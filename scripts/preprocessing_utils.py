import spikeinterface.preprocessing as st
import spikeinterface.extractors as se
import spikeinterface.full as si
from matplotlib import pyplot as plt
from probeinterface.plotting import plot_probe, plot_probe_group
from pathlib import Path
import numpy as np
import time
import os
try:
    from kilosort import io, run_kilosort
except ImportError:
    print("Kilosort not installed")
    pass
import json
try:
    import torch
except ImportError:
    print("Torch not installed")
    torch = None
    pass

import spikeinterface.sorters as ss



def find_recinfo_file(base_path, target_file='structure.oebin'):
    file_path = list(Path(base_path).rglob(target_file))
    assert len(file_path) == 1, f"Found {len(file_path)} files with name {target_file} in {base_path}. Please specify a single file."
    return file_path[0]


def read_continuous_data_info(recinfo_file):
    with open(recinfo_file, 'r') as f:
        file_contents = f.read()
        rec_info = json.loads(file_contents)
    # pprint(rec_info)
    return rec_info["continuous"]


def get_channel_names(continuous_data_info):
    string_to_exclude = ['LFP', 'NI-DAQ']
    # recinfo = load_recinfo(base_path)
    all_folders = []
    ap_folders = []
    for folder in continuous_data_info:
        stream_name = folder["recorded_processor"] + " " + str(folder["recorded_processor_id"]) + "#" + folder['folder_name'][:-1]
        all_folders.append(stream_name)

        if all([string not in folder['folder_name'] for string in string_to_exclude]):
            ap_folders.append(stream_name)
    
    return all_folders, ap_folders


def make_probe_plots(recording_extractor, hpsf_recording_extractor, path, stream_name):
    path = Path(path)
    # noise levels plot:
    noise_levels_microV = si.get_noise_levels(recording_extractor, return_scaled=True)
    fig, ax = plt.subplots()
    _ = ax.hist(noise_levels_microV, bins=np.arange(5, 30, 2.5))
    ax.set_xlabel('noise  [microV]')
    fig.savefig(path / 'preprocFigs_noiseLevels.png')

    # Valid probe channels plot:
    probe = hpsf_recording_extractor.get_probe() 

    plot_probe(probe)  # for me to see (just which shanks I am dealing with)
    plt.gcf().savefig(path / 'preprocFigs_probe_goodChannels.png')

    df = probe.to_dataframe(complete=True)  # complete=True includes channel number
    probetablename = Path(path / f"probe_table_{stream_name}.csv")
    df.to_csv(probetablename, index=False)


def standard_preprocessing(recording_extractor):
    """Aggregate lab standard preprocessing steps for Neuropixels data."""

    recording = st.correct_lsb(recording_extractor, verbose=1)
    recording = st.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = st.phase_shift(recording) #lazy
    [bad_channel_ids, channel_labels] = st.detect_bad_channels(recording=recording)  
    recording = recording.remove_channels(remove_channel_ids=bad_channel_ids)  # could be interpolated instead, but why?

    # split in groups and apply spatial filtering, then reaggregate. KS4 can now handle multiple shanks
    grouped_recordings = recording.split_by(property='group')
    recgrouplist_hpsf = [st.highpass_spatial_filter(recording=grouped_recordings[k]) for k in grouped_recordings.keys()]  # cmr is slightly faster. results are similar
    recording_hpsf = si.aggregate_channels(recgrouplist_hpsf)

    return recording_hpsf


def call_ks(preprocessed_recording, stream_name, working_folder_path, callKSfromSI=True, remove_binary=True):
    working_folder_path = Path(working_folder_path)
    
    DTYPE = np.int16
    
    probe = preprocessed_recording.get_probe()

    if callKSfromSI: #takes forever even with latest spikeinterface version. stopped it.
        # call kilosort from here, without saving the .bin (will have to call something else to run phy)
        print("Starting KS4")
        t_start = time.perf_counter()
        sorting_ks4_prop = ss.run_sorter_by_property("kilosort4", recording=preprocessed_recording, 
                        grouping_property="group", working_folder=working_folder_path, verbose=True)
        t_stop = time.perf_counter()
        elapsed_prop = np.round(t_stop - t_start, 2)
        print(f"Elapsed time by property: {elapsed_prop} s")

    else:
        # kilosort export .bin continuous data (this can be read by phy too)
        # haven't tried the wrapper, pessimistic about the time it takes
        probename = "probe_{}.prb".format(stream_name)
        filename, N, c, s, fs, probe_path = io.spikeinterface_to_binary(
            preprocessed_recording, working_folder_path, data_name='data.bin', dtype=DTYPE,
            chunksize=60000, export_probe=True, probe_name=probename
            )
        
        #run KS programmatically
        settings = {'fs': fs, 'n_chan_bin': probe.get_contact_count()}
        probe_dict = io.load_probe(probe_path)
        kilosort_optional_params = {}
        if torch is not None:
            kilosort_optional_params["device"] = torch.device("cuda")

        ops, st_ks, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = run_kilosort(
                            settings=settings, probe=probe_dict, filename=filename, data_dtype=DTYPE, **kilosort_optional_params
        )

        # decide whether you want to delete data.bin afterwards
        bin_path = working_folder_path / 'data.bin'
        if bin_path.exists() and remove_binary:
            bin_path.unlink()