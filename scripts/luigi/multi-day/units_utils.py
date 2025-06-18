from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface.curation as scur
from neuroconv.datainterfaces import (KiloSortSortingInterface,
                                      OpenEphysRecordingInterface)
from recording_processor import get_stream_name
from spikeinterface.extractors import read_openephys
from spikeinterface.full import load_sorting_analyzer, concatenate_recordings
import pynapple as nap
from zoneinfo import ZoneInfo


def _get_units_df(probe_sorted_units):
    """Read dataframe of units info with position on the probe."""
    templates = np.load(probe_sorted_units / "templates.npy")
    units_df = pd.read_csv(probe_sorted_units / "cluster_KSLabel.tsv", sep="\t")
    probe_channel_positions = np.load(probe_sorted_units / "channel_positions.npy")

    peak_chan_idxs = templates.mean(1).argmax(1)
    unit_channel_position = probe_channel_positions[peak_chan_idxs]
    units_df["probe_x"] = unit_channel_position[:, 0]
    units_df["probe_depth"] = unit_channel_position[:, 1]
    units_df["probe_channel"] = peak_chan_idxs

    return units_df


def load_units_df(folder: Path) -> pd.DataFrame:
    date = datetime.strptime(folder.parent.name.split("_")[0], "%Y-%m-%d")
    mouse_id = (
        folder.parent.parent.parent.name
        if "split" in folder.parent.parent.name
        else folder.parent.parent.name
    )

    analyzer = load_sorting_analyzer(folder / "analyser", format="binary_folder")
    units_df = _get_units_df(folder / "sorter_output")
    metrics = analyzer.get_extension(extension_name="quality_metrics").get_data()

    units_df["recording_date"] = date
    units_df["mouse_id"] = mouse_id
    units_df = pd.concat([units_df, metrics], axis=1)
    return units_df


def read_patched_openephys_folders(recording_folders):
    if len(recording_folders) == 1:
        return read_patched_openephys(recording_folders[0])
    else:
        all_extractors = [read_patched_openephys(recording_folder) for recording_folder in recording_folders]
    
    global concatenated_times
    concatenated_times = np.concatenate([extractor.get_times() for extractor in all_extractors])
    
    def _get_times(segment_index=0):
        return concatenated_times
    
    patched_recording_extractor = all_extractors[0]
    patched_recording_extractor.get_times = _get_times

    return patched_recording_extractor


def read_patched_openephys(recording_folder):
    # My eyes have never bled this much
    recording_extractor = read_openephys(recording_folder, stream_name=get_stream_name(recording_folder), load_sync_timestamps=True)

    all_timestamps = []
    if recording_extractor.get_num_segments() > 1:
        patched_recording_extractor = read_openephys(recording_folder, 
                                                     stream_name=get_stream_name(recording_folder), 
                                                     experiment_names=["experiment1", "experiment2"],
                                                     load_sync_timestamps=False)
        for segment_index in range(recording_extractor.get_num_segments()):
            all_timestamps.append(recording_extractor.get_times(segment_index=segment_index))

        global concatenated_times
        concatenated_times = np.concatenate(all_timestamps)
        def _get_times(segment_index=0):
            return concatenated_times
        patched_recording_extractor.get_times = _get_times
        recording_extractor = patched_recording_extractor

    return recording_extractor


def spikes_interface_loder(input_data_folder: Path) -> pd.DataFrame:
    folder_path = Path(input_data_folder) / "kilosort4"
    assert folder_path.exists(), f"Folder {folder_path} does not exist"
    # Change the folder_path to the location of the data in your system
    try:
        recording_folder = next((folder_path.parent / "NPXData").glob("2025-*"))
        recording = OpenEphysRecordingInterface(
            folder_path=recording_folder,
            stream_name=get_stream_name(recording_folder),
        )
        recording_extractor = read_patched_openephys(recording_folder)
        # recording_extractor = read_openephys(
        #         recording_folder,
        #         stream_name=get_stream_name(recording_folder),
        #         load_sync_timestamps=True,
        #         #block_index=i,
        #         # experiment_names=["experiment1", "experiment2"]#experiment_name
        #     )
            
        # tstamps = recording_extractor.get_times()
        recording.recording_extractor = recording_extractor
    except StopIteration:
        print(
            f"No recording folder found in {folder_path.parent / 'NPXData'}. Won't use recording timestamps."
        )
        recording = None
    
    try:
        times = recording_extractor.get_times()
        print(f"Times shape: {times.shape}, min: {np.min(times)}, max: {np.max(times)}")
    except ValueError as e:
        if "Multi-segment object" in str(e):
            print(f"Multi-segment object issue: {e}")
            # In this case, we have a multisegment issue
        else:
            raise e
    
    ks_interface = KiloSortSortingInterface(
        folder_path=folder_path / "sorter_output", verbose=False, keep_good_only=False
    )
    # Deal with stupid KS bug of fake spikes at negative or too high indexes. 
    # There seems to be a bug, the resulting trace is too short. Maybe not bug, just an issue with 
    # my stupid dummy local data!
    
    # for attr in ["get_duration", "get_total_duration", "get_end_time", "get_start_time"]:
      #   print(attr, getattr(recording_extractor, attr)())
    #ks_interface.sorting_extractor = scur.remove_excess_spikes(
    #    ks_interface.sorting_extractor, recording_extractor
    # )
    # AAAAAAARAAARGGGGGGHHHH  this come with the monkeypatch. No way of reading a 1-segment without concatenating otherwise
    recording.recording_extractor.get_num_segments = lambda: ks_interface.sorting_extractor.get_num_segments()
    ks_interface.register_recording(recording)
    
    return ks_interface  # nap.load_file(nwb_path)



if  __name__ == "__main__":
    from nwb_tester import test_on_temp_nwb_file
    from time import time
    from pprint import pprint
    # example_path = "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126"
    #example_path = '/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250508/155144'
    example_main_path = Path("/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings/")
    assert example_main_path.exists(), f"Folder {example_main_path} does not exist"
    all_paths_to_test = sorted(list(example_main_path.glob("M*_WT002/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]/[0-9][0-9][0-9][0-9][0-9][0-9]")))
    pprint(all_paths_to_test)
    for example_path in all_paths_to_test:
        print("-"*100)
        print(example_path)
        try:
            start_time = time()
            # example_path = "/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings/M29_WT002/20250508/155144"
            
            
            data_folder = Path(example_path)
            assert data_folder.exists(), f"Folder {data_folder} does not exist"
            nwb_path = data_folder / "sorter_output.nwb"
            interface = spikes_interface_loder(data_folder)
            nwb_file_data = test_on_temp_nwb_file(interface, nwb_path)
            print(nwb_file_data)
            print("Successfully loaded spikes interface")
            good_units_mask = nwb_file_data["units"]["KSLabel"] == "good"

            good_units = nwb_file_data["units"]#[good_units_mask]
            good_units_tsd = good_units.to_tsd()
            print(f"Converted to Tsd with shape: {good_units_tsd.shape}")
            print(
                "First spike time: ",
                np.min(good_units_tsd.t),
                "Last spike time: ",
                np.max(good_units_tsd.t),
            )
        except Exception as e:
            print(f"Error loading spikes interface: {e}")
        print(f"Time taken: {time() - start_time} seconds")
