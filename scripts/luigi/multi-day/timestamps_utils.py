# %%
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from nwb_conv import parse_folder_metadata
from spikeinterface.extractors import read_openephys_event


MIN_T_TO_SPLIT = 4
MIN_SEGMENT_LENGTH = 3000
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


def split_session_triggers(session_triggers, min_t_to_split=MIN_T_TO_SPLIT):
    """Split triggers based on gaps in the triggers.
    
    Parameters
    ----------
    session_triggers : np.ndarray
        Array of trigger timestamps
    min_t_to_split : float
        Minimum time gap to consider a split (in seconds)
        
    Returns
    -------
    list of np.ndarray
        List of trigger arrays, one for each session
    """
    delta_ts = np.diff(session_triggers)
    split_indices = np.argwhere(delta_ts > min_t_to_split).flatten()
    
    if len(split_indices) == 0:
        raise ValueError("No splits found!")
    
    # Convert to split points (add 1 because diff shifts indices)
    split_points = split_indices + 1
    
    # Create list of session trigger arrays
    sessions = []
    start_idx = 0
    
    for split_point in split_points:
        sessions.append(session_triggers[start_idx:split_point])
        start_idx = split_point
    
    # Add the final session (from last split to end)
    sessions.append(session_triggers[start_idx:])
    
    return sessions


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

    
    possible_sessions = ["object", "cricket", "roach"]

    video_files = sorted(list(input_data_folder.glob("videos/*/*.avi")))
    video_files.sort(key=lambda x: x.name)  # sort by filename with timestamp
    actual_sessions = [v.parts[-2] for v in video_files]

    # this part is to deal with multiple video folders for the same session when camera was started multiple times
    more_video_files = list(input_data_folder.glob("videos/*/*/*.avi"))
    more_video_files.sort(key=lambda x: x.name)  # sort by filename with timestamp
    more_actual_sessions = [v.parts[-3] for v in more_video_files]
    actual_sessions = actual_sessions + more_actual_sessions


    assert set(actual_sessions).issubset(
        set(possible_sessions)
    ), f"Session should be one of {possible_sessions}. From video I see: {set(actual_sessions)}"
    # assert len()
    print(actual_sessions)
    if len(set(actual_sessions)) >= 2:
        print("!!!!!!!!!! warning !!!!!!!!!!")
        print(f"Usuall 2 session types is supported for now. From video I see: {set(actual_sessions)}")
        print("I assume this is a special case with a prolonged acquisition")
    #else:
    #    assert (
    #        len(set(actual_sessions)) == 2
    #    ), f"Usuall 2 session types is supported for now. From video I see: {set(actual_sessions)}"
    print(f"Sessions: {actual_sessions}")

    npx_data_folders = list(input_data_folder.glob("NPXData/2025-*"))
    npx_data_folders.sort(key=lambda x: x.name)
    assert len(npx_data_folders) <= 2, f"Expected 2 NPX data folders, got {len(npx_data_folders)}"
    is_split = len(npx_data_folders) == 2
    session_triggers = []
    # print(npx_data_folders)

    # TODO: right now this is a tangled mess, but we have to deal with many possible cases...
    if is_split:
        session_triggers = []
        for npx_data_folder in npx_data_folders:
            recs_parent_folder_list = list(
                npx_data_folder.glob("Record Node */experiment*")
            )
            assert len(recs_parent_folder_list) == 1
            recs_folder = list(recs_parent_folder_list[0].glob("recording*"))
            print(recs_folder)
            if len(recs_folder) > 1:
                print("Multiple recordings found for a single session, concatenating them")
                triggers_array = np.concatenate([
                    get_timestamps_si(npx_data_folder, recording_number=n) for n in range(2)
                ])
                session_triggers.append(triggers_array) 

            else:
                print("Single recording found for this session, good")
                all_events = get_timestamps_si(npx_data_folder)
                session_triggers.append(all_events)
    else:
        npx_data_folder = npx_data_folders[0]
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
            
            np.save("all_events.npy", all_events)

            session_triggers = split_session_triggers(all_events)
            print(f"Found {len(session_triggers)} sessions triggers of length:")
            print([len(t) for t in session_triggers])
            session_triggers = [t for t in session_triggers if len(t) > MIN_SEGMENT_LENGTH]
            print(f"After filtering, found {len(session_triggers)} sessions triggers of length:")
            print([len(t) for t in session_triggers])

    assert len(session_triggers) == len(
        actual_sessions
    ), f"Number of session triggers ({len(session_triggers)}) should be equal to number of sessions ({len(actual_sessions)})"

    if "M29_WT002" in str(input_data_folder) and "20250507" in str(input_data_folder):
        print("Funny patch required because M29 on 20250507 was dropping one every 2 frames in the first session")
        session_triggers[0] = session_triggers[0][::2]

    for i, video_file in enumerate(video_files):
        n_frames = get_video_info(video_file)
        mismatch_n = n_frames - len(session_triggers[i])
        
        assert (mismatch_n) < 10, f"Number of frames in video {video_file.name} is greater than number of session triggers. From video I see: {n_frames}, from session triggers I see: {len(session_triggers[i])}"
            #print("!!!!!!!!!! warning !!!!!!!!!!")
            #print(f"Number of frames in video {video_file.name} is greater than number of session triggers. From video I see: {n_frames}, from session triggers I see: {len(session_triggers[i])}")
        #else:
            #assert abs(mismatch_n) < 10, f"Number of frames in video {video_file.name} is greater than number of session triggers. From video I see: {n_frames}, from session triggers I see: {len(session_triggers[i])}"
        # assert n_frames < len(
        #    session_triggers[i]
        #), f"Number of frames in video {video_file.name} is greater than number of session triggers. From video I see: {n_frames}"
        #print(
        #    f"Trimming session triggers from {len(session_triggers[i])} to {n_frames} frames"
        #)
        if mismatch_n < 0:
            print(f"Trimming session triggers from {len(session_triggers[i])} to {n_frames} frames")
            session_triggers[i] = session_triggers[i][:n_frames+2]
        else:
            print(f"Padding session triggers from {len(session_triggers[i])} to {n_frames} frames")
            dt = np.median(np.diff(session_triggers[i]))
            padding = np.arange(mismatch_n) * dt + session_triggers[i][-1]
            session_triggers[i] = np.concatenate([session_triggers[i], padding])

    return session_triggers, actual_sessions

# %%
if __name__ == "__main__":
    #sample_folder = "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250508/155144"    # sample_folder = "/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings/M30_WT002/20250513/103449"
    sample_folder = "/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings/M31_WT002/20250510/121940"
    sample_folder = Path(sample_folder)
    session_triggers, actual_sessions = get_video_timestamps(sample_folder)
    print(actual_sessions)
    #fname = "/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings/M30_WT002/20250511/164220/NPXData/2025-05-11_17-21-59"    
    #tstamps = get_timestamps_si(fname)

# %%