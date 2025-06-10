from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface.curation as scur
from neuroconv.datainterfaces import (KiloSortSortingInterface,
                                      OpenEphysRecordingInterface)
from recording_processor import get_stream_name
from spikeinterface.extractors import read_openephys
from spikeinterface.full import load_sorting_analyzer
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
        # block_index=0)  # read_openephys(recording_folder, stream_name=get_stream_name(recording_folder))
        recording_extractor = read_openephys(
            recording_folder,
            stream_name=get_stream_name(recording_folder),
            load_sync_timestamps=True,
        )
        recording.recording_extractor = recording_extractor
    except StopIteration:
        print(
            f"No recording folder found in {folder_path.parent / 'NPXData'}. Won't use recording timestamps."
        )
        recording = None
    
    times = recording_extractor.get_times()
    print(f"Times shape: {times.shape}, min: {np.min(times)}, max: {np.max(times)}")
    ks_interface = KiloSortSortingInterface(
        folder_path=folder_path / "sorter_output", verbose=False, keep_good_only=False
    )
    # Deal with stupid KS bug of fake spikes at negative or too high indexes
    ks_interface.sorting_extractor = scur.remove_excess_spikes(
        ks_interface.sorting_extractor, recording_extractor
    )
    ks_interface.register_recording(recording)
    
    return ks_interface  # nap.load_file(nwb_path)


def test_on_temp_nwb_file(input_data_folder: Path):
    folder_path = Path(input_data_folder) / "kilosort4"
    assert folder_path.exists(), f"Folder {folder_path} does not exist"
    nwb_path = folder_path.parent / "sorter_output.nwb"
    if nwb_path.exists():
        nwb_path.unlink()
    
    ks_interface = KiloSortSortingInterface(
        folder_path=folder_path / "sorter_output", verbose=False, keep_good_only=False
    )

    metadata = ks_interface.get_metadata()
    session_start_time = datetime(2020, 1, 1, 12, 30, 0, tzinfo=ZoneInfo("US/Pacific"))
    metadata["NWBFile"].update(session_start_time=session_start_time)
    ks_interface.run_conversion(nwbfile_path=nwb_path, metadata=metadata)

    nwb = nap.load_file(nwb_path)
    return nwb


if __name__ == "__main__":
    # example_path = "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126"
    example_path = '/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250508/155144'
    data_folder = Path(example_path)
    nwb_file_data = test_on_temp_nwb_file(data_folder)
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
    # Read metadata from folder and subject log:
    # metadata = parse_folder_metadata(data_folder)
    # units_df = load_units_df(data_folder / "kilosort4")
    # print(units_df)
