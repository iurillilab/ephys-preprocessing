# %%

from spikeinterface.extractors import read_openephys, read_openephys_event
from recording_processor import RecordingProcessor, get_stream_name
import numpy as np
from pathlib import Path
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
import spikeinterface.curation as scur
import pynapple as nap

from neuroconv.datainterfaces import KiloSortSortingInterface, OpenEphysRecordingInterface
# %%

# create units table
sample_folder = Path("/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126")  # Path("/Users/vigji/Desktop/short_recording_oneshank/2025-01-22_16-56-15")
folder_path = sample_folder / "kilosort4"
spike_times = np.load('/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126/kilosort4/sorter_output/spike_times.npy')
timestamps_file = '/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126/NPXData/2025-05-09_12-04-15/Record Node 107/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeA/timestamps.npy'
timestamps = np.load(timestamps_file)
# save timestamps to ".dat" file of binary data, making second dim 1:
timestamps_reshaped = timestamps.reshape(-1, 1)
timestamps_dat_path = Path(timestamps_file).parent / "continuous.dat"
timestamps_reshaped.astype(np.float64).tofile(timestamps_dat_path)

recording_folder = next((folder_path.parent / "NPXData").glob("2025-*"))
# recording_folder = Path('/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys _recordings/M29/2025-05-09_12-04-15')
stream_name = get_stream_name(recording_folder)
print(stream_name)
recording_extractor = read_openephys(recording_folder, 
                                         stream_name=stream_name,
                                         load_sync_timestamps=True)
times = recording_extractor.get_times(segment_index=0)



print(spike_times.min(), spike_times.max())
print(len(times), times.min(), times.max())
print(len(timestamps), timestamps.min(), timestamps.max())

# %%

sample_folder = Path("/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126")  # Path("/Users/vigji/Desktop/short_recording_oneshank/2025-01-22_16-56-15")
folder_path = sample_folder / "kilosort4"


def dirty_nwb_like_loder(folder_path: Path) -> pd.DataFrame:
    # Change the folder_path to the location of the data in your system
    try:
        recording_folder = next((folder_path.parent / "NPXData").glob("2025-*"))
        recording = OpenEphysRecordingInterface(folder_path=recording_folder, 
                                                stream_name=get_stream_name(recording_folder),)
                                                # block_index=0)  # read_openephys(recording_folder, stream_name=get_stream_name(recording_folder))
        recording_extractor = read_openephys(recording_folder, 
                                            stream_name=get_stream_name(recording_folder),
                                            load_sync_timestamps=True)
        recording.recording_extractor = recording_extractor
    except StopIteration:
        print(f"No recording folder found in {folder_path.parent / 'NPXData'}. Won't use recording timestamps.")
        recording = None

    nwb_path = folder_path.parent / "sorter_output.nwb"

    ## if nwb_path.exists():
        #return nap.load_file(nwb_path)
    recording_extractor.get_times
    ks_interface = KiloSortSortingInterface(folder_path=folder_path / "sorter_output", 
                                         verbose=False,
                                         keep_good_only=False)
    ks_interface.sorting_extractor = scur.remove_excess_spikes(ks_interface.sorting_extractor, recording_extractor)
    ks_interface.register_recording(recording)
    metadata = ks_interface.get_metadata()
    session_start_time = datetime(2020, 1, 1, 12, 30, 0, tzinfo=ZoneInfo("US/Pacific"))
    metadata["NWBFile"].update(session_start_time=session_start_time)
    ks_interface.run_conversion(nwbfile_path=nwb_path, metadata=metadata)
    return nap.load_file(nwb_path)

#sample_folder = Path('/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250508/155144')
#sorter = read_kilosort(sample_folder / "kilosort4" / "sorter_output", keep_good_only=False)
# %%
# units_df = load_units_df(sample_folder / "kilosort4")
nwb_file = dirty_nwb_like_loder(sample_folder / "kilosort4")
