import spikeinterface.preprocessing as st
import spikeinterface.extractors as se
from spikeinterface import create_sorting_analyzer
import spikeinterface.full as si
from matplotlib import pyplot as plt
from probeinterface.plotting import plot_probe, plot_probe_group
from probeinterface import write_prb, ProbeGroup
from pathlib import Path
import numpy as np
import time
import os
import shutil
import logging


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

try:
    from nwb_conv.oephys import OEPhysDataFolder
except ImportError:
    print("nwb-conv not installed")
    pass

import spikeinterface.sorters as ss


def get_logger(log_file_path, print_to_console=True):
    logger = logging.getLogger("Processing logger")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if print_to_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def copy_folder(src, dst):
    src = Path(src)
    dst = Path(dst)
    
    dst = dst / src.name
    dst.mkdir(exist_ok=True, parents=True)

    shutil.copytree(src, dst, dirs_exist_ok=True)

    return dst


def get_job_kwargs(chunk_duration="1s", progress_bar=True):
    n_cpus = os.cpu_count()
    n_jobs = n_cpus // 2
    job_kwargs = dict(progress_bar=progress_bar, 
                    chunk_duration=chunk_duration, 
                    n_jobs=n_jobs)
    return job_kwargs

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
    if recording.get_channel_groups().max() > 1:
        grouped_recordings = recording.split_by(property='group')
        recgrouplist_hpsf = [st.highpass_spatial_filter(recording=grouped_recordings[k]) for k in grouped_recordings.keys()]  # cmr is slightly faster. results are similar
        recording_hpsf = si.aggregate_channels(recgrouplist_hpsf)
    else:
        recording_hpsf = st.highpass_spatial_filter(recording=recording)

    return recording_hpsf


def call_ks(preprocessed_recording, stream_name, working_folder_path, 
            callKSfromSI=True, remove_binary=True, drift_correction=True, n_jobs=None):
    working_folder_path = Path(working_folder_path)

    if n_jobs is None:
        n_jobs = os.cpu_count() - 1

    
    DTYPE = np.int16
    
    probe = preprocessed_recording.get_probe()

    if callKSfromSI: 
        #Maybe old comment: takes forever even with latest spikeinterface version. stopped it.
        # call kilosort from here, without saving the .bin (will have to call something else to run phy)

        probename = "probe_{}.prb".format(stream_name)
        pg = ProbeGroup()
        pg.add_probe(probe)
        write_prb(working_folder_path / probename, pg)


        t_start = time.perf_counter()
        sorting_ks4 = ss.run_sorter_by_property(sorter_name="kilosort4", 
                                                recording=preprocessed_recording, 
                                                grouping_property="group", 
                                                engine="joblib",
                                                engine_kwargs={"n_jobs": n_jobs},
                                                folder=working_folder_path, 
                                                verbose=True)
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
        settings = {'fs': fs, 'n_chan_bin': probe.get_contact_count(), 'n_blocks': int(drift_correction)}
        probe_dict = io.load_probe(probe_path)
        kilosort_optional_params = {}
        if torch is not None:
            kilosort_optional_params["device"] = torch.device("cuda")

        ops, st_ks, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = run_kilosort(
                            settings=settings, 
                            probe=probe_dict, 
                            filename=filename, 
                            data_dtype=DTYPE, 
                            **kilosort_optional_params
        )

        # decide whether you want to delete data.bin afterwards
        bin_path = working_folder_path / 'data.bin'
        if bin_path.exists() and remove_binary:
            print(f"Removing {bin_path}")
            bin_path.unlink()
        else:
            print(f"data.bin file to remove not found at {bin_path}")

        sorting_ks4 = si.read_kilosort(folder_path=working_folder_path)

    return sorting_ks4


def compute_stats(sorting_data_folder, sorter_object, recording, **job_kwargs):
     # first fix the params.py file saved by KS
    found_paths = []
    found_paths = sorting_data_folder.rglob("params.py")
    # search_files(ks_folder, "params.py")
    for file_path in found_paths:   
        print(file_path)
        sortinganalyzerfolder = file_path.parent / "analyser"

        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        # Todo understand this mod here:
        for i, line in enumerate(lines):
            if "class numpy.int16" in line:
                print(lines[i])
                lines[i] = "dtype = '<class int16>'\n"  
                break
        with open(file_path, 'w') as file:
            file.writelines(lines)
        print("params.py file updated successfully.")

        # run the analyzer
        analyzer = create_sorting_analyzer(
            sorting=sorter_object,
            recording=recording,
            folder=sortinganalyzerfolder,
            format="binary_folder",
            sparse=True,
            overwrite=True
        )

        
        logging.info("compute random spikes...")
        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500) #parent extension
        logging.info("compute waveforms...")
        analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0, **job_kwargs)
        logging.info("compute templates...")
        analyzer.compute("templates", operators=["average", "median", "std"])
        print(analyzer)

        si.compute_noise_levels(analyzer)   # return_scaled=True  #???
        si.compute_spike_amplitudes(analyzer)

        logging.info("compute quality metrics...")
        start_time = time.time()
        dqm_params = si.get_default_qm_params()
        qms = analyzer.compute(input={"principal_components": dict(n_components=3, mode="by_channel_local"),
                                        "quality_metrics": dict(skip_pc_metrics=False, qm_params=dqm_params)}, 
                                verbose=True, **job_kwargs)
        qms = analyzer.get_extension(extension_name="quality_metrics")
        metrics = qms.get_data()
        print(metrics.columns)
        assert 'isolation_distance' in metrics.columns
        elapsed_time = time.time() - start_time
        print(f"Elapsed time: {elapsed_time} seconds")


def find_laser_onsets_npx_idxs(oephys_data: OEPhysDataFolder, channel_name: str):
    """Find indexes of laser ON from OEPhysDataFolder object with NIDAQX recording.
    Assumes existence of a channel named laser-log.
    """

    # TODO: This shold be read from metadata in the future:
    oephys_data.nidaq_channel_map = {0: "frames-log", 
                                    1: "laser-log", 
                                    2: "-", 
                                    3: "motor-log", 
                                    4: "barcode", 
                                    5: "-", 
                                    6: "-", 
                                    7: "-"}
    
    npx_barcode = oephys_data.load_channel_barcode(channel_name)

    nidaq_data = oephys_data.nidaq_recording

    nidaq_barcode = nidaq_data.barcode
    laser_data = nidaq_data.continuous_signals["laser-log"]

    return nidaq_barcode.transform_idxs_to(npx_barcode, laser_data.onsets).astype(int)


# additional plots for snippets of trace:
def plot_snippets(data_interface, n_snippets=4, padding_s=30, snippet_length_s=0.04, axs=None, additional_info=None,
                  vmax=None):
    """
    Plot snippets of traces from the recording to check for drifts, artifacts, etc.
    Parameters
    ----------
    data_interface : RecordingExtractor
        Recording extractor.
    n_snippets : int
        Number of snippets to plot.
    padding_s : float
        Padding at beginning and end of recording to avoid edge effects.
    snippet_length_s : float
        Length of each snippet in seconds.
    """
    DEFAULT_WIDTH = 3.5
    DEFAULT_HEIGHT = 5

    snippet_length = int(snippet_length_s * data_interface.sampling_frequency)

    # intersperse snippets sampling beginning, middle and end of recording:
    snippet_start_s = np.linspace(padding_s, data_interface.get_total_duration() - padding_s, n_snippets)
    snippet_start_idxs = (snippet_start_s * data_interface.sampling_frequency).astype(int)

    if axs is None:
        fig, axs = plt.subplots(1, n_snippets, figsize=(DEFAULT_WIDTH* n_snippets, DEFAULT_HEIGHT), 
                                sharey=True, gridspec_kw={'wspace': 0.1})
    else:
        fig = axs[0].figure
    
    chan_ids = data_interface.get_channel_ids()
    for i, idx in   enumerate(snippet_start_idxs):
        snippet = data_interface.get_traces(start_frame=idx, end_frame=idx + snippet_length)
        if i == 0 and vmax is None:
            vmax = np.percentile(np.abs(snippet), 99)

        axs[i].imshow(snippet.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="upper",
                    extent=[0, snippet_length_s*1000, chan_ids[-1], chan_ids[0]])
        axs[i].set(xlabel="Time (ms)")
        if i == 0:
            axs[i].set(ylabel="Channel" if additional_info is None else additional_info + "\n\nChannel")
        else:
            axs[i].set(yticklabels=[])
            
        axs[i].set_title(f"t={snippet_start_s[i]:.1f} s", fontsize=10)

    # add small axis for cmap deriving its position from the last subplot location on the figure:
    cax = fig.add_axes([axs[-1].get_position().x1 + 0.01, axs[-1].get_position().y0, 
                        axs[-1].get_position().width*0.05, axs[-1].get_position().height / 5])
    plt.colorbar(axs[-1].images[0], cax=cax)
    
    return axs


def plot_raw_and_preprocessed(raw_extractor, preprocessed_extractor, n_snippets=4, padding_s=30, snippet_length_s=0.04,
                              saving_path=None, v_maxs=(100, 50), stream_name=None):
    """
    Plot snippets of trace and preprocessed trace from the recording to check for drifts, artifacts, etc.
    Parameters
    ----------
    raw_extractor : RecordingExtractor
        Raw recording extractor.
    preprocessed_extractor : RecordingExtractor
        Preprocessed recording extractor.
    n_snippets : int
        Number of snippets to plot.
    padding_s : float
        Padding at beginning and end of recording to avoid edge effects.
    snippet_length_s : float
        Length of each snippet in seconds.
    """

    if stream_name is None:
        stream_name = raw_extractor.stream_name

    fig, all_axs = plt.subplots(2, n_snippets, figsize=(3.5 * n_snippets, 10), sharex=True, 
                            gridspec_kw={'wspace': 0.1, 'hspace': 0.3})
    
    for ax, data, info, vmax in zip(all_axs, 
                                    [raw_extractor, preprocessed_extractor], 
                                    [f"Raw ({stream_name})", "Preprocessed"], 
                                    v_maxs):
        plot_snippets(data, n_snippets=n_snippets, padding_s=padding_s, snippet_length_s=snippet_length_s, 
                      axs=ax, additional_info=info, vmax=vmax)
        
    if saving_path is not None:
        plt.savefig(saving_path, dpi=300)


def show_laser_triggered(oe_extractor, laser_onset_idxs, n_to_take=100, padding_s=0.05, channel_idx=150, v_maxs=100, ax=None):
    """Visualize laser-triggered snippets for a given channel, to test correct artefact zeroing.
    """
    fs = oe_extractor.sampling_frequency

    skip = len(laser_onset_idxs) // n_to_take
    pre_int = int(padding_s * fs)
    post_int = int(padding_s * fs)

    sel_laser_onset_idxs = laser_onset_idxs[::skip]  # take every n-th laser onset for visualization
    cropped = np.zeros((len(sel_laser_onset_idxs), pre_int + post_int))

    for i, laser_idx in enumerate(sel_laser_onset_idxs):
        cropped[i, :] = oe_extractor.get_traces(start_frame = laser_idx - pre_int, end_frame = laser_idx + post_int, 
                                                channel_ids=[oe_extractor.get_channel_ids()[channel_idx]]).flat
        cropped[i, :] = cropped[i, :] - np.mean(cropped[i, :])
    
    if ax is None:
        f, ax = plt.subplots()
    ax.imshow(cropped, cmap="RdBu_r", aspect="auto", vmin=-v_maxs, vmax=v_maxs)
    ax.axvline(pre_int, color="green", lw=2)


def show_laser_trigger_preprost(raw_extractor, preprocessed_extractor, laser_onset_idxs, 
                                n_to_take=100, padding_s=0.05, channel_idx=150, v_maxs=(100, 50), saving_path=None):
    """Visualize laser-triggered snippets for a given channel, to test correct artefact zeroing."""
    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    
    show_laser_triggered(raw_extractor, laser_onset_idxs, n_to_take, padding_s, channel_idx, v_maxs[0], ax=ax[0])
    ax[0].set_title("Raw data")
    
    show_laser_triggered(preprocessed_extractor, laser_onset_idxs, n_to_take, padding_s, channel_idx, v_maxs[1], ax=ax[1])
    ax[1].set_title("Preprocessed data")

    if saving_path is not None:
        f.savefig(saving_path)