from pathlib import Path
import sys
from pprint import pprint
from timestamps_utils import get_video_timestamps
from neuroconv import ConverterPipe
from neuroconv.datainterfaces import SLEAPInterface, DeepLabCutInterface, ExternalVideoInterface
import pandas as pd
import numpy as np

dlc_config_path_dict = {
        'cricket': Path('/Users/vigji/Desktop/cricket-below-Luigi Petrucco-2024-02-05/config.yaml'),
        'roach': None,
        'object': Path('/Users/vigji/Desktop/cricket-below-Luigi Petrucco-2024-02-05/config.yaml'),
    }

def load_video_interfaces(data_path: Path):
    """Load the video interface for a video."""

    all_video_timestamps, sessions_names = get_video_timestamps(data_path)

    interfaces_list = []
    video_file_paths_list = []
    for session_name, video_timestamps in zip(sessions_names, all_video_timestamps):
        print(session_name)
        session_video_path = data_path / "videos" / session_name
        assert session_video_path.exists(), f"Session video path {session_video_path} does not exist"

        # SLEAP file:
        slp_files = sorted(list(session_video_path.glob("multicam_video_*_*/*centralpredictions.slp")))
        assert len(slp_files) == 1, f"Expected 1 SLP file, got {len(slp_files)}"
        slp_file_path = slp_files[0]

        # DeepLabCut file:
        dlc_files = sorted(list(session_video_path.glob(f"multicam_video_*_*/*{session_name}*.h5")))
        assert len(dlc_files) == 1, f"Expected 1 DLC file, got {len(dlc_files)}"
        dlc_file_path = dlc_files[0]

        # Video file:
        video_file_path = next(slp_file_path.parent.glob("*central*.mp4"))
        assert video_file_path.exists(), f"Video file path {video_file_path} does not exist"
        video_file_paths_list.append(str(video_file_path))

        # DLC interface:
        dlc_interface = DeepLabCutInterface(file_path=dlc_file_path, pose_estimation_metadata_key=f"PoseEstimationDeepLabCut{session_name.capitalize()}")#, config_file_path=video_file_path)
        dlc_df = pd.read_hdf(dlc_file_path, key="df_with_missing")
        # print(len(dlc_df), len(video_timestamps))
        dlc_interface.set_aligned_timestamps(video_timestamps)
        
        # SLEAP interface:   
        # TODO: for the future, find a way to integrate better names:     
        sleap_interface = SLEAPInterface(file_path=slp_file_path, 
                                         video_file_path=str(video_file_path))
        sleap_interface.set_aligned_timestamps(video_timestamps)

        # Order matters: need to add sleap first, then dlc
        interfaces_list.append(sleap_interface)
        interfaces_list.append(dlc_interface)

    # Video interface:
    print(video_file_path)
    video_interface = ExternalVideoInterface(file_paths=video_file_paths_list, 
                                                video_name=f"BottomVideo",
                                                verbose=True)
    video_interface.set_aligned_timestamps(all_video_timestamps)
    interfaces_list.append(video_interface)

    conversion_options = {"ExternalVideoInterface": dict(starting_frames=np.array([0, len(all_video_timestamps[0])]))}

    return interfaces_list, conversion_options




if __name__ == "__main__":
    from nwb_tester import test_on_temp_nwb_file
    #  = '/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126/videos/cricket/multicam_video_2025-05-09T12_25_28_cropped_20250528161845/multicam_video_2025-05-09T12_25_28_centralpredictions.slp'
    data_path = Path('/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126')

    nwb_path = data_path / "tracking_output.nwb"
    
    interfaces_list, conversion_options = load_video_interfaces(data_path)
    nwb_file = test_on_temp_nwb_file(ConverterPipe(interfaces_list), nwb_path, conversion_options)
    print(nwb_file)
        # print(video_timestamps)
    # print(video_timestamps)

    
    # sleap_interface = SLEAPInterface(file_path=slp_file_path)
    # sleap_interface.set_aligned_timestamps()
    # print(sleap_interface)
    # # ConverterPipe
    # test_on_temp_nwb_file(sleap_interface, nwb_path)



# All of this maybe can go:
"""
if len(sys.argv) > 1:
    main_path = Path(sys.argv[1])
else:
    main_path = Path("/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings")
    print(f"No path provided, using example: {main_path}")

if len(sys.argv) > 2:
    config_path = Path(sys.argv[2])
else:
    config_path = Path("/Users/vigji/Desktop/cricket-below-Luigi Petrucco-2024-02-05/config.yaml")
    print(f"No config path provided, using example: {config_path}")

all_sessions = list(main_path.glob("M*_WT*/*/*")) 
videos_to_process = [str(path) for path in main_path.glob("M*_WT*/*/*/videos/concatenated_central.mp4")]
print(f"Found {len(videos_to_process)} videos to process")
pprint(videos_to_process)

if len(videos_to_process) > 0:
    import deeplabcut
    deeplabcut.analyze_videos(config_path, videos_to_process)
"""
