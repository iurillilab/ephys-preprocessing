import deeplabcut
from deeplabcut.pose_estimation_pytorch import train_network
from deeplabcut.modelzoo import build_weight_init
from pathlib import Path

superanimal_name = "superanimal_topviewmouse"

config_path = '/Users/vigji/Desktop/dlc3_mouse_object_cricket_roach-YaduLuigi-2025-06-10/config.yaml'

# net_type = "hrnet_w32"
# weight_init = build_weight_init(cfg=config_path, super_animal=superanimal_name, model_name=net_type, detector_name=net_type, with_decoder=False)
# deeplabcut.create_training_dataset(config_path, weight_init=weight_init, net_type=net_type)

# train_network(config_path, shuffle=7)


# After creating the training dataset folder structure, update the torch_config with the following:
"""
data:
      rotation: 270
  # augmentation if your bodyparts are [snout, eye_L, eye_R, ear_L, ear_R]
  hflip:
    p: 0.5  # apply a horizontal flip with 50% probability
    symmetries: [[1, 5], [2, 4]]  # the indices of symmetric keypoints
crop_bodyparts: true

logger:
  type: WandbLogger
  project_name: DLC_object_cricket_roach  # the name of the project where the run should be logged
"""


# 3. Launch training on shuffle index 1
#    - displayiters: how often to print loss (e.g. every 1k iters)
#    - saveiters: how often to save a snapshot (matches your config)

# 4. After training, evaluate performance on the held‐out set
# deeplabcut.evaluate_network(config_path, shuffle=1,plotting=True)

# 5. Run inference on videos using model from shuffle 7
videos_to_analyze = [
    '/Users/vigji/Desktop/model_videos/multicam_video_2025-05-08T16_36_18_central.mp4',
    '/Users/vigji/Desktop/model_videos/multicam_video_2025-05-08T16_57_05_central.mp4',
    '/Users/vigji/Desktop/model_videos/multicam_video_2025-05-09T12_05_00_central.mp4',
    '/Users/vigji/Desktop/model_videos/multicam_video_2025-05-09T12_25_28_central.mp4',
    # Add more video paths as needed
]

for f in videos_to_analyze:
    assert Path(f).exists(), f"File {f} does not exist"

# Analyze videos using the trained model from shuffle 7
deeplabcut.analyze_videos(config_path, videos_to_analyze, shuffle=7)

# Optionally, create labeled videos with predictions
# deeplabcut.create_labeled_video(config_path, videos_to_analyze, shuffle=7)

# Optionally, plot trajectories
# deeplabcut.plot_trajectories(config_path, videos_to_analyze, shuffle=7)


