# %%

from spikeinterface.extractors import read_openephys, read_openephys_event
from recording_processor import RecordingProcessor, get_stream_name
import numpy as np
from pathlib import Path

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
