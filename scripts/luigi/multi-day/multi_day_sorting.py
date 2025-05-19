"""Batch spike sorting on all data from a batch of recordings"""

# %%
# ========================================================================
# Parameters for running the script:
test_mode = True  # if true, find folders to analyze but then use test recording
source_dir = r"N:\SNeuroBiology_shared\P07_PREY_HUNTING_YE\e01_ephys _recordings"
test_rec_tuple = (r"D:\luigi\mouse_data_electrode_tips\2024-12-18_14-52-41",
                    "Record Node 111#Neuropix-PXI-110.ProbeA")
main_working_dir = r"D:\temp_processing"  # Local disk location to temporarily move files from the NAS


# ========================================================================
# Imports and functions:
import re
from pathlib import Path
from pprint import pprint
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt
import spikeinterface.sorters as ss
from spikeinterface.exporters import export_to_phy, export_report
from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor
import spikeinterface.preprocessing as spre
from spikeinterface import aggregate_channels
from spikeinterface.core import concatenate_recordings
from spikeinterface import create_sorting_analyzer

import os
import shutil
import time

source_dir = Path(source_dir)
main_working_dir = Path(main_working_dir)
test_rec_tuple = (Path(test_rec_tuple[0]), test_rec_tuple[1])

assert test_rec_tuple[0].exists(), f"Test recording {test_rec_tuple[0]} does not exist."


def get_stream_name(folder):
    """Matching file pattern to get stream name. Currently taylored to find 
    first NPX2 recording, has to change for multiprobe/NPX1"""

    STREAM_NAME_MATCH = "Record Node *\\experiment*\\recording*\\continuous\\Neuropix-PXI-*.Probe*"
    stream_path = folder.glob(STREAM_NAME_MATCH)
    stream_path = next(stream_path)
    record_node = str(stream_path).split("Record Node ")[1].split("\\")[0]
    probe = stream_path.name

    return f"Record Node {record_node}#{probe}"

def copy_folder(src, dst):
    """Copy a folder to a new location, creating the destination folder if it doesn't exist.
    Args:
        src (str): Path to the source folder.
        dst (str): Path to the destination folder PARENT
    """
    src, dts = Path(src), Path(dst)
    dst = dst / src.name
    dst.mkdir(exist_ok=True, parents=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return dst


def get_job_kwargs(chunk_duration="1s", progress_bar=True):
    return dict(n_jobs=os.cpu_count() // 2,  # physical cores hald of virtual total
                progress_bar=progress_bar, 
                # backend="loky",
                chunk_duration=chunk_duration)


# %%
# ========================================================================
# Locate all data sources:

mouse_paths = {f.name : f for f in source_dir.glob("*M[0-9][0-9]")}
print("Mouse paths to analyze:")
pprint(mouse_paths)

pooled_dirs_to_process = dict()
for mouse_id, mouse_path in list(mouse_paths.items())[:1]:
# mouse_id, mouse_path = list(mouse_paths.items())[0]
    print(f"Searching {mouse_id}")
    # Find all unique folder named with a timestamp in the form YYYY-MM-DD_HH-MM-SS for the mouse

    # Regex pattern for timestamp format YYYY-MM-DD_HH-MM-SS
    timestamp_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")

    # Find all unique folders matching the timestamp pattern
    timestamp_folders = sorted(
        folder for folder in mouse_path.rglob("*-*_*/")
        if folder.is_dir() and timestamp_pattern.match(folder.name)
    )

    print(f"Found {len(timestamp_folders)} OpenEphys folders with timestamps:")
    pprint(timestamp_folders)

    for timestamp_folder in timestamp_folders:
        try:
            # Check if the folder contains a recording
            stream_name = get_stream_name(timestamp_folder)
            recording_raw = OpenEphysBinaryRecordingExtractor(timestamp_folder, stream_name=stream_name)
            # recording_raw = concatenate_recordings([recording_raw])
        except Exception as e:
            print(f"Error processing {timestamp_folder}: {e}; \n is this a valid OpenEphys folder?")
            continue

        # put together folders from paired recordings:
        if "split_" in timestamp_folder.parent.name:
            key = timestamp_folder.parent.name
            folder_list, _ = pooled_dirs_to_process.get(key, [[], None])
            folder_list.append(timestamp_folder)
            pooled_dirs_to_process[key] = [folder_list, stream_name]
        else:
            # If the folder is not paired, add it to the dictionary
            pooled_dirs_to_process[timestamp_folder.name] = [timestamp_folder, stream_name]

print(f"Found {len(pooled_dirs_to_process)} total folders to process:")
pprint(pooled_dirs_to_process)

# ========================================================================
# Process all folders:
if test_mode:
    pooled_dirs_to_process = [test_rec_tuple]
    print(f"Test mode: using {test_rec_tuple[0]} as test recording.")



# %%

folder, stream_name = pooled_dirs_to_process[0]
print(f"Processing {folder} with stream name {stream_name}")

start_t = time.time()
if len(list((main_working_dir / folder.name).rglob("*.dat"))) == 0:
    print(f"Copying {folder} to {main_working_dir}")
    local_folder = copy_folder(folder, main_working_dir)
    print(f"Copying took {time.time()-start_t:.2f} seconds")
else:
    local_folder = main_working_dir / folder.name
    print(f"Already copied {folder} to {local_folder}")
# %%
def standard_preprocessing(recording_extractor):
    """Aggregate lab standard preprocessing steps for Neuropixels data."""

    recording = spre.correct_lsb(recording_extractor, verbose=1)
    recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = spre.phase_shift(recording) #lazy
    bad_channel_ids, _ = spre.detect_bad_channels(recording=recording)  
    # recording = recording.remove_channels(remove_channel_ids=bad_channel_ids)  # could be interpolated instead, but why?
    recording_clean = spre.interpolate_bad_channels(recording, bad_channel_ids=bad_channel_ids)

    # split in groups and apply spatial filtering, then reaggregate. KS4 can now handle multiple shanks
    if recording.get_channel_groups().max() > 1:
        grouped_recordings = recording.split_by(property='group')
        recgrouplist_hpsf = [spre.highpass_spatial_filter(recording=grouped_recordings[k]) for k in grouped_recordings.keys()]  # cmr is slightly faster. results are similar
        recording_hpsf = aggregate_channels(recgrouplist_hpsf)
    else:
        recording_hpsf = spre.highpass_spatial_filter(recording=recording)

    return recording_hpsf

def load_data_from_folder(folder, stream_name):
    """Load data from a folder using OpenEphysBinaryRecordingExtractor."""
    if isinstance(folder, str) or isinstance(folder, Path):
        # If a single folder is provided, use it directly
        folder = [folder]
    return concatenate_recordings(
                [OpenEphysBinaryRecordingExtractor(f, stream_name=stream_name) for f in folder]
            )


def compute_stats(sorting_data_folder, sorter_object, recording, logger=None, **job_kwargs):
    """Compute stats for a given sorting data folder and sorter object."""
     # first fix the params.py file saved by KS
    diagnostics_message_streamer = logger.info if logger is not None else print

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

        
        diagnostics_message_streamer("compute random spikes...")
        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500) #parent extension
        diagnostics_message_streamer("compute waveforms...")
        analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0, **job_kwargs)
        diagnostics_message_streamer("compute templates...")
        analyzer.compute("templates", operators=["average", "median", "std"])
        print(analyzer)

        si.compute_noise_levels(analyzer)   # return_scaled=True  #???
        si.compute_spike_amplitudes(analyzer)

        diagnostics_message_streamer("compute quality metrics...")
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
        diagnostics_message_streamer(f"Elapsed time: {elapsed_time} seconds")

        _ = analyzer.compute('correlograms')

        # # the export process is fast because everything is pre-computed
        export_to_phy(sorting_analyzer=analyzer, output_folder=sortinganalyzerfolder / "phy", copy_binary=False)
        export_report(sorting_analyzer=analyzer, output_folder=sortinganalyzerfolder / "report")


recording_extractor = load_data_from_folder(local_folder, stream_name=stream_name)


job_kwargs = get_job_kwargs(chunk_duration="1s", progress_bar=True)
preprocessed = standard_preprocessing(recording_extractor)

# %%


working_folder_path = local_folder / "kilosort4"
if working_folder_path.exists():
    # remove the folder if it exists
    print(f"Removing existing working folder: {working_folder_path}")
    shutil.rmtree(working_folder_path)

working_folder_path.mkdir(parents=True, exist_ok=True)

sorting_ks4 = ss.run_sorter_by_property(sorter_name="kilosort4", 
                                                recording=preprocessed, 
                                                folder=working_folder_path, 
                                                grouping_property="group", 
                                                engine="joblib",
                                                engine_kwargs=job_kwargs,
                                                verbose=True)


compute_stats(working_folder_path, sorting_ks4, preprocessed, **job_kwargs)


# %%
