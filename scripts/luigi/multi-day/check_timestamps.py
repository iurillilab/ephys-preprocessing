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
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import pynapple as nap
from rastermap import Rastermap
import plotly.io as pio
import cv2
import pandas as pd

from recording_processor import get_stream_name
from spikeinterface.extractors import read_openephys_event



sample_folder = Path("/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126")  # Path("/Users/vigji/Desktop/short_recording_oneshank/2025-01-22_16-56-15")
assert sample_folder.exists()
pprint(parse_folder_metadata(sample_folder))


from neuroconv.datainterfaces import SLEAPInterface
from neuroconv.utils import dict_deep_update

# Change the file_path so it points to the slp file in your system
file_path = '/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126/videos/cricket/multicam_video_2025-05-09T12_25_28_cropped_20250528161845/multicam_video_2025-05-09T12_25_28_centralpredictions.slp'  # sample_folder / "sleap" / "predictions_1.2.7_provenance_and_tracking.slp"
assert Path(file_path).exists()
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
def get_timestamps_si(data_folder, recording_number=0, cam_ch=2):
    event_interface = read_openephys_event(folder_path=data_folder)
    events = event_interface.get_events(channel_id='PXIe-6341Digital Input Line', segment_index=recording_number)
    timestamps = np.array([v[0] for v in events])
    channels = np.array([int(v[2]) for v in events])
    return timestamps[channels == cam_ch]


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


cricket_video_file = next(sample_folder.glob("videos/cricket/*.avi"))
object_video_file = next(sample_folder.glob("videos/object/*.avi"))

# n_frames_cricket, n_frames_object = get_video_info(cricket_video_file), get_video_info(object_video_file)


is_split = False
sample_folder = Path("/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250508/155144")  # Path("/Users/vigji/Desktop/short_recording_oneshank/2025-01-22_16-56-15")
#sample_folder = Path("/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126")

MIN_T_TO_SPLIT = 10
possible_sessions = ["object", "cricket", "roach"]
cam_events = {cam_name: None for cam_name in possible_sessions}
video_files = list(sample_folder.glob("videos/*/*.avi"))
video_files.sort(key=lambda x: x.name)  # sort by filename with timestamp
actual_sessions = [v.parts[-2] for v in video_files]
assert set(actual_sessions).issubset(set(possible_sessions)), f"Session should be one of {possible_sessions}. From video I see: {set(actual_sessions)}"
assert len(set(actual_sessions)) == 2, f"Only 2 session type is supported for now. From video I see: {set(actual_sessions)}"
print(f"Sessions: {actual_sessions}")

session_triggers = []
if not is_split:
    npx_data_folder = next(sample_folder.glob("NPXData/2025-*"))
    recs_parent_folder_list = list(npx_data_folder.glob("Record Node */experiment*"))
    assert len(recs_parent_folder_list) == 1
    recs_folder = list(recs_parent_folder_list[0].glob("recording*"))

    if len(recs_folder) > 1:
        assert len(recs_folder) == 2, "Only two recordings are supported for now"
        print("Multiple recordings found")
        session_triggers = [get_timestamps_si(npx_data_folder, recording_number=n) for n in range(2)]
        
    else:
        print("Single recording found, splitting camera trigger events")
        all_events = get_timestamps_si(npx_data_folder)
        delta_ts = np.diff(all_events)
        video_split = np.argwhere(delta_ts > MIN_T_TO_SPLIT)[0, 0] + 1
        session_triggers = [all_events[:video_split], all_events[video_split:]]

assert len(session_triggers) == len(actual_sessions), f"Number of session triggers ({len(session_triggers)}) should be equal to number of sessions ({len(actual_sessions)})"


for i, video_file in enumerate(video_files):
    n_frames = get_video_info(video_file)
    assert n_frames < len(session_triggers[i]), f"Number of frames in video {video_file.name} is greater than number of session triggers. From video I see: {n_frames}"
    print(f"Trimming session triggers from {len(session_triggers[i])} to {n_frames} frames")
    session_triggers[i] = session_triggers[i][:n_frames]



# %%

# %%






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
