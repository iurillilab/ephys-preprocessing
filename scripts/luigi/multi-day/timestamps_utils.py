# %%
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from nwb_conv import parse_folder_metadata
from spikeinterface.extractors import read_openephys_event

# %%


def get_timestamps_si(data_folder, recording_number=0, cam_ch=2):
    event_interface = read_openephys_event(folder_path=data_folder)
    events = event_interface.get_events(
        channel_id="PXIe-6341Digital Input Line", segment_index=recording_number
    )
    timestamps = np.array([v[0] for v in events])
    channels = np.array([int(v[2]) for v in events])
    return timestamps[channels == cam_ch]


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def get_video_timestamps(input_data_folder):
    """Get the timestamps of the video frames for each session.

    Parameters
    ----------
    input_data_folder : Path
        Full DATA path (NOT video folder) (main session directory)

    Returns
    -------
    session_triggers : list of np.ndarray
        Timestamps of the video frames for each session.
    """

    input_data_folder = Path(input_data_folder)
    assert (
        input_data_folder.exists()
    ), f"Input video folder {input_data_folder} does not exist"
    # Determine if the data is split over two sessions:
    # TODO: still missing the case of two separate recordings!
    is_split = False

    MIN_T_TO_SPLIT = 10
    possible_sessions = ["object", "cricket", "roach"]
    cam_events = {cam_name: None for cam_name in possible_sessions}
    video_files = list(input_data_folder.glob("videos/*/*.avi"))
    video_files.sort(key=lambda x: x.name)  # sort by filename with timestamp
    print(video_files)
    actual_sessions = [v.parts[-2] for v in video_files]
    assert set(actual_sessions).issubset(
        set(possible_sessions)
    ), f"Session should be one of {possible_sessions}. From video I see: {set(actual_sessions)}"
    assert (
        len(set(actual_sessions)) == 2
    ), f"Only 2 session type is supported for now. From video I see: {set(actual_sessions)}"
    print(f"Sessions: {actual_sessions}")

    session_triggers = []
    if not is_split:
        npx_data_folder = next(input_data_folder.glob("NPXData/2025-*"))
        recs_parent_folder_list = list(
            npx_data_folder.glob("Record Node */experiment*")
        )
        assert len(recs_parent_folder_list) == 1
        recs_folder = list(recs_parent_folder_list[0].glob("recording*"))

        if len(recs_folder) > 1:
            assert len(recs_folder) == 2, "Only two recordings are supported for now"
            print("Multiple recordings found")
            session_triggers = [
                get_timestamps_si(npx_data_folder, recording_number=n) for n in range(2)
            ]

        else:
            print("Single recording found, splitting camera trigger events")
            all_events = get_timestamps_si(npx_data_folder)
            delta_ts = np.diff(all_events)
            video_split = np.argwhere(delta_ts > MIN_T_TO_SPLIT)[0, 0] + 1
            session_triggers = [all_events[:video_split], all_events[video_split:]]

    assert len(session_triggers) == len(
        actual_sessions
    ), f"Number of session triggers ({len(session_triggers)}) should be equal to number of sessions ({len(actual_sessions)})"

    for i, video_file in enumerate(video_files):
        n_frames = get_video_info(video_file)
        assert n_frames < len(
            session_triggers[i]
        ), f"Number of frames in video {video_file.name} is greater than number of session triggers. From video I see: {n_frames}"
        print(
            f"Trimming session triggers from {len(session_triggers[i])} to {n_frames} frames"
        )
        session_triggers[i] = session_triggers[i][:n_frames+2]

    return session_triggers, actual_sessions


if __name__ == "__main__":
    #sample_folder = "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250508/155144"
    sample_folder = "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126"
    sample_folder = Path(sample_folder)
    session_triggers, actual_sessions = get_video_timestamps(sample_folder)
    print(actual_sessions)
    print(session_triggers)

# %%
