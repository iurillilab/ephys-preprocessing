from spikeinterface.extractors import read_openephys
from spikeinterface import concatenate_recordings
from pathlib import Path
from recording_processor import get_stream_name
from pprint import pprint
import numpy as np

# example_path = "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126"
example_path = '/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250508/155144'

folder_path = Path(example_path) / "NPXData"
assert folder_path.exists(), f"Folder {folder_path} does not exist"
recording_folder = next((folder_path).glob("2025-*"))

def read_patched_openephys(recording_folder):
    # My eyes have never bled this much
    recording_extractor = read_openephys(recording_folder, stream_name=get_stream_name(recording_folder), load_sync_timestamps=True)

    all_timestamps = []
    if recording_extractor.get_num_segments() > 1:
        patched_recording_extractor = read_openephys(recording_folder, stream_name=get_stream_name(recording_folder), load_sync_timestamps=False)
        for segment_index in range(recording_extractor.get_num_segments()):
            all_timestamps.append(recording_extractor.get_times(segment_index=segment_index))

        global concatenated_times
        concatenated_times = np.concatenate(all_timestamps)
        def _get_times(segment_index=0):
            return concatenated_times
        patched_recording_extractor.get_times = _get_times
        recording_extractor = patched_recording_extractor

    return recording_extractor
        # recording_extractor = concatenate_recordings([recording_extractor], ignore_times=True)
    # print(recording_extractor)
    # tstamps = recording_extractor.get_times()
    # print(tstamps.shape, tstamps.min(), tstamps.max())

# print(recording_extractor.get_times())

recording_extractor = read_openephys(recording_folder, 
                                     stream_name=get_stream_name(recording_folder), 
                                     load_sync_timestamps=True,
                                     block_index=1,
                                     experiment_names=["experiment1"])
print(recording_extractor)
print(recording_extractor.get_num_segments())