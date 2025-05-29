# %%
from nwb_conv import check_lablogs_location
from nwb_conv import parse_folder_metadata
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from spikeinterface.full import read_kilosort, load_sorting_analyzer
from recording_processor import RecordingProcessor
from pathlib import Path
from pprint import pprint


from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import pynapple as nap
from rastermap import Rastermap
import plotly.io as pio
import cv2
import pandas as pd

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from recording_processor import get_stream_name


sample_folder = Path("/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250508/155144")  # Path("/Users/vigji/Desktop/short_recording_oneshank/2025-01-22_16-56-15")
assert sample_folder.exists()
pprint(parse_folder_metadata(sample_folder))


from neuroconv.datainterfaces import SLEAPInterface
from neuroconv.utils import dict_deep_update

# Change the file_path so it points to the slp file in your system
file_path = "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250508/155144/videos/Object/multicam_video_2025-05-08T16_36_18_cropped_20250528161845/multicam_video_2025-05-08T16_36_18_centralpredictions.slp"  # sample_folder / "sleap" / "predictions_1.2.7_provenance_and_tracking.slp"
interface = SLEAPInterface(file_path=file_path, verbose=False)
# Then, run:
# interface.set_aligned_timestamps

# Extract what metadata we can from the source files
metadata = interface.get_metadata()
metadata = dict_deep_update(metadata, parse_folder_metadata(sample_folder))

# Choose a path for saving the nwb file and run the conversion
nwbfile_path = f"/Users/vigji/Desktop/delete_me.nwb"  # This should be something like: "saved_file.nwb"
interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)

# %%
def get_timestamps_no_si(event_data_folder, recording_number=1, cam_ch=2):
    binary_data_folder_pattern = f"Record Node */experiment*/recording{recording_number}/events/NI-DAQmx-*.PXIe-6341/TTL"
    binary_data_folder = next(event_data_folder.glob(binary_data_folder_pattern))

    timestamps_file = next(binary_data_folder.glob("timestamps.npy"))
    timestamps = np.load(timestamps_file)
    #words_file = next(binary_data_folder.glob("full_words.npy"))
    #words = np.load(words_file)
    states_file = next(binary_data_folder.glob("states.npy"))
    states = np.load(states_file)

    video_trigger_times = timestamps[states == cam_ch]
    return video_trigger_times

def get_timestamps_si(data_folder, recording_number=1, cam_ch=2):
    event_data_folder_pattern = f"Record Node */experiment*/recording{recording_number}/events/NI-DAQmx-*.PXIe-6341/TTL"
    print(data_folder, event_data_folder_pattern)
    binary_data_folder = next(data_folder.glob(event_data_folder_pattern))
    stream_name = binary_data_folder.parts[-2]

    event_interface = OpenEphysBinaryEventExtractor(folder_path=data_folder, stream_id=stream_name)
    return event_interface

# get_timestamps_si(sample_folder, recording_number=1, cam_ch=2)
# OpenEphysBinaryEventExtractor()
from spikeinterface.extractors import OpenEphysBinaryEventExtractor
f = list(sample_folder.glob("NPXData/2025-*"))[0]
OpenEphysBinaryEventExtractor(folder_path=f)
# %%



video_files = list(sample_folder.glob("videos/*/*.avi"))
print(video_files)
video_timestamps_files = list(sample_folder.glob("videos/*/*.csv"))
print(video_timestamps_files)


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def get_timestamps_info(timestamps_path):
    df = pd.read_csv(timestamps_path)
    return len(df)

# video_split = np.argwhere(np.diff(video_trigger_times) > 10)[0, 0]
#video_triggers_video0 = video_trigger_times[:video_split]
#video_triggers_video1 = video_trigger_times[video_split:]
video_triggers_video0 = get_timestamps(event_data_folder, recording_number=1, cam_ch=2)
video_triggers_video1 = get_timestamps(event_data_folder, recording_number=2, cam_ch=2)

total_frames = 0
print("\nVideo and timestamp analysis:")
for video_file, ts_file, video_triggers in zip(video_files, 
                                               video_timestamps_files, 
                                               [video_triggers_video0, video_triggers_video1]):
    video_frames = get_video_info(video_file)
    timestamp_count = get_timestamps_info(ts_file)
    print(f"\n{video_file.name}:")
    print(f"Video Frames: {video_frames}")
    print(f"Bonsai csv timestamps: {timestamp_count}")
    print(f"DAQ TTL Triggers: {len(video_triggers)}")
    print(f"Percent mismatch n of frames: {100 * (video_frames - timestamp_count) / video_frames}%")
    df = pd.read_csv(ts_file, header=None)
    df[0] = pd.to_datetime(df[0])
    df[0] = (df[0] - df[0].iloc[0]).dt.total_seconds() 
    dt_csv = np.median(np.diff(df[0]))
    dt_daq = np.median(np.diff(video_triggers))
    print(f"dt_csv: {dt_csv}, dt_daq: {dt_daq}")
    print(f"Percent mismatch dt: {100 * (dt_csv - dt_daq) / dt_csv}%")
    total_frames += video_frames



# %%
px.histogram(x=video_split, log_y=True)
# %%



# %%
pd.read_csv(video_timestamps_files[0], header=None)

# %%
# Parse timestamps to datetime
df = pd.read_csv(video_timestamps_files[0], header=None)
df[0] = pd.to_datetime(df[0])
df[0] = (df[0] - df[0].iloc[0]).dt.total_seconds()
print(df.head())
print(f"Timestamp column dtype: {df[0].dtype}")

np.median(np.diff(df[0]))

# %%
print(sum(states == cam_ch))
# %%
import plotly.io as pio
pio.renderers.default = "vscode"

# %%
np.median(np.diff(timestamps[states == cam_ch]))
# %%
px.histogram(x=np.diff(df[0]), log_y=True)
# %%
