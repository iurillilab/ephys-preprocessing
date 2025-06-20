from pathlib import Path
import sys
from pprint import pprint
from timestamps_utils import get_video_timestamps
from neuroconv import ConverterPipe
from neuroconv.datainterfaces import SLEAPInterface, DeepLabCutInterface, ExternalVideoInterface
import pandas as pd
import numpy as np

dlc_config_path_dict = {
        'cricket': Path('/Users/vigji/Desktop/final_models/dlc3_cricket-YaduLuigi-2025-06-10/config.yaml'),
        # 'mouse': Path('/Users/vigji/Desktop/final_models/dlc3_mouse-YaduLuigi-2025-06-10/config.yaml'), 
        'mouse': Path('/Users/vigji/Desktop/final_models/mouse-bottom-new/config.yaml'), 
        'roach': Path('/Users/vigji/Desktop/final_models/dlc3_roach-YaduLuigi-2025-06-10/config.yaml'),
        'object': Path('/Users/vigji/Desktop/final_models/dlc3_object-YaduLuigi-2025-06-10/config.yaml'),
    }

def load_video_interfaces(data_path: Path):
    """Load the video interface for a video."""

    all_video_timestamps, sessions_names = get_video_timestamps(data_path)

    # Funny patch requires because M29 on 20250507 was dropping one every 2 frames in the first session:
    # print(str(data_path))

    interfaces_list = []
    video_file_paths_list = []
    n_iterations_per_session = {}
    for session_name, video_timestamps in zip(sessions_names, all_video_timestamps):
        print(session_name)
        n_iter = n_iterations_per_session.get(session_name, 0)
        session_video_path = data_path / "videos" / session_name
        assert session_video_path.exists(), f"Session video path {session_video_path} does not exist"

        # SLEAP file:
        # slp_files = sorted(list(session_video_path.glob("multicam_video_*_*/*centralpredictions.slp")))
        #assert len(slp_files) == 1, f"Expected 1 SLP file, got {len(slp_files)}"
        #slp_file_path = slp_files[0]

        # DeepLabCut file:
        for model_name in ["mouse", session_name]:
            # print(f"multicam_video_*_*/*{session_name}*{model_name}*.h5")
            # print(session_video_path)
            # print(list(session_video_path.glob("multicam_video_*_*/*.h5")))
            dlc_files = sorted(list(session_video_path.glob(f"multicam_video_*_*/*{model_name}*.h5")))
            if len(dlc_files) == 0:
                metadata_key = f"PoseEstimationDeepLabCutSession{session_name.capitalize()}Entity{model_name.capitalize()}Video{n_iter}"
                dlc_files = sorted(list(session_video_path.glob(f"*/multicam_video_*_*/*{model_name}*.h5")))
                assert len(dlc_files) > 1, f"Expected at least 2 DLC files, got {len(dlc_files)} in {session_video_path}"
                dlc_file = dlc_files[n_iter]
            else:
                metadata_key = f"PoseEstimationDeepLabCutSession{session_name.capitalize()}Entity{model_name.capitalize()}"
                assert len(dlc_files) == 1, f"Expected 1 DLC file, got {len(dlc_files)} in {session_video_path}"
                dlc_file = dlc_files[0]

            # if len(dlc_files) == 1 f"Expected 1 DLC file, got {len(dlc_files)} in {session_video_path}"

            # rep here deals with multiple videos aquired during the same session
            
            dlc_interface = DeepLabCutInterface(file_path=dlc_file, 
                                                config_file_path=dlc_config_path_dict[model_name],
                                                pose_estimation_metadata_key=metadata_key)#, config_file_path=video_file_path)
            print(dlc_file)
            dlc_df = pd.read_hdf(dlc_file, key="df_with_missing")

            # TODO fix this abomination:
            len_tracking_frames = len(dlc_df)
            tstamps = video_timestamps[:len_tracking_frames]
            #print(len(video_timestamps), len(video_timestamps[:10]), len(video_timestamps[:len_tracking_frames]))
            #print(len(video_timestamps[:len_tracking_frames]), len_tracking_frames, len(tstamps))
            assert len_tracking_frames == len(video_timestamps[:len_tracking_frames]), f"Number of frames in DLC file {dlc_files[0]} and video timestamps {len(tstamps)} do not match"
            dlc_interface.set_aligned_timestamps(tstamps)    

            interfaces_list.append(dlc_interface)

        n_iterations_per_session[session_name] = n_iter + 1

        # Video file:
        video_file_path = next(dlc_files[0].parent.glob("*central*.mp4"))
        assert video_file_path.exists(), f"Video file path {video_file_path} does not exist"
        video_file_paths_list.append(str(video_file_path))
        

    # Video interface:
    print(video_file_path)
    video_interface = ExternalVideoInterface(file_paths=video_file_paths_list, 
                                                video_name=f"BottomVideo",
                                                verbose=True)
    video_interface.set_aligned_timestamps(all_video_timestamps)
    interfaces_list.append(video_interface)

    conversion_options = {"ExternalVideoInterface": dict(starting_frames=np.array([0,] + [len(t) for t in all_video_timestamps[:-1]]))}

    return interfaces_list, conversion_options


if __name__ == "__main__":
    from nwb_tester import test_on_temp_nwb_file
    #  = '/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126/videos/cricket/multicam_video_2025-05-09T12_25_28_cropped_20250528161845/multicam_video_2025-05-09T12_25_28_centralpredictions.slp'
    # data_path = Path('/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126')
    main_data_path = Path('/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings')
    data_paths = sorted(list(main_data_path.glob("M3*_WT*/*/[0-9][0-9][0-9][0-9][0-9][0-9]")))    
    # data_paths = [Path("/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings/M30_WT002/20250513/103449")]
    for data_path in data_paths:
        print(data_path)
        nwb_path = data_path / "tracking_output.nwb"
        interfaces_list, conversion_options = load_video_interfaces(data_path)
        nwb_file = test_on_temp_nwb_file(ConverterPipe(interfaces_list), nwb_path, conversion_options)
        print(nwb_file)

