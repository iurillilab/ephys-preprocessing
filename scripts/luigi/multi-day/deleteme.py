
# %%
import numpy as np
import json
from matplotlib import pyplot as plt
# %%
all_timestamps = []
for file in [
    '/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings/M29_WT002/20250507/100459/NPXData/2025-05-07_10-12-22/Record Node 107/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeA/timestamps.npy',
    '/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings/M29_WT002/20250507/100459/NPXData/2025-05-07_14-10-50/Record Node 107/experiment1/recording2/continuous/Neuropix-PXI-100.ProbeA/timestamps.npy',
    '/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings/M29_WT002/20250507/100459/NPXData/2025-05-07_14-10-50/Record Node 107/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeA/timestamps.npy'

]:
    timestamps = np.load(file)
    all_timestamps.append(timestamps)
    print(len(timestamps), timestamps[0], timestamps[-1], timestamps[-1] - timestamps[0]),

# %%
spike_frames_filename = '/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings/M29_WT002/20250507/100459/kilosort4/sorter_output/spike_times.npy'
spike_frames = np.load(spike_frames_filename)
# %%
spike_frames[-1]
# %%
spike_frames[-1] - len(np.concatenate(all_timestamps))
# %%
plt.figure(figsize=(10, 5))
plt.hist(spike_frames, bins=100)
# plt.plot(spike_frames)
plt.show()

# %%
# %%
recording_info_file = "/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings/M29_WT002/20250507/100459/kilosort4/spikeinterface_recording.json"

with open(recording_info_file, 'r') as f:
    recording_info = json.load(f)

# %%
si_log_file = '/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings/M29_WT002/20250507/100459/kilosort4/spikeinterface_log.json'
with open(si_log_file, 'r') as f:
    si_log = json.load(f)

# %%
recording_info.keys()
# %%
to_probe = recording_info["kwargs"]["recording_list"][3]["kwargs"]["recording"]["kwargs"]["parent_recording"]

for max_n_iters in range(100):
    if "kwargs" in to_probe:
        to_probe = to_probe["kwargs"]
    elif "recording" in to_probe:
        to_probe = to_probe["recording"]
    elif "recording_list" in to_probe:
        to_probe = to_probe["recording_list"][2]
    else:
        break

print(to_probe["folder_path"], to_probe["experiment_names"], to_probe["stream_name"])
# print(to_probe["properties"].keys())

# %%
to_probe
# ["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording"]["kwargs"]["recording"].keys()#["properties"].keys()






# %%
# extra
t = np.load('/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings/M30_WT002/20250513/103449/NPXData/2025-05-13_11-12-55/Record Node 107/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeA/timestamps.npy')
print(len(t), t[0], t[-1], t[-1] - t[0])

# %%
