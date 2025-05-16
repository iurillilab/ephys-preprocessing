# %%
from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor
import spikeinterface.preprocessing as spre
from spikeinterface.core import concatenate_recordings
import re
from pathlib import Path
from pprint import pprint
import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt

source_dir = Path("N:\SNeuroBiology_shared\P07_PREY_HUNTING_YE\e01_ephys _recordings")

mouse_paths = {f.name : f for f in source_dir.glob("*M[0-9][0-9]")}
pprint(mouse_paths)
# %%
window_s = 2  # second window
num_windows = 50

all_rms_list_dict = {}
all_timestamps_dict = {}

for mouse_id, mouse_path in mouse_paths.items():
    print(f"Processing {mouse_id}")
    # for every unique timestamp, fetch first folder matching that timestamp:
    unique_timestamps = sorted(list(set(folder.name.split("_")[0] for folder in timestamp_folders)))
    pprint(unique_timestamps)

    all_rms_list = []
    for timestamp in unique_timestamps:
        print(f"Processing {timestamp}")
        # Find all folders matching the timestamp
        first_matching_folder = next(folder for folder in timestamp_folders if folder.name.startswith(timestamp))
        first_matching_folder

        stream_name = "Record Node 107#Neuropix-PXI-100.ProbeA"
        recording_raw = OpenEphysBinaryRecordingExtractor(first_matching_folder, stream_name=stream_name)
        recording_raw = concatenate_recordings([recording_raw])
        #bad_channel_ids, _ = spre.detect_bad_channels(recording_raw)
        #recording_clean = spre.interpolate_bad_channels(recording_raw, bad_channel_ids=bad_channel_ids)
        filtered = spre.bandpass_filter(recording_raw, freq_min=300, freq_max=3000)

        fs = filtered.get_sampling_frequency()
        num_samples = int(window_s * fs)
        window_starts = np.linspace(0, filtered.get_num_frames() - num_samples, num_windows, dtype=int)
        channel_ids = filtered.channel_ids
        # print(window_starts)
        rms_map = np.zeros((len(channel_ids), num_windows))
        for i in trange(num_windows):
            traces = filtered.get_traces(start_frame=i*num_samples, end_frame=(i+1)*num_samples)
            rms_map[:, i] = np.sqrt(np.mean(traces**2, axis=0))

        all_rms_list.append(rms_map)
    

    all_timestamps_dict[mouse_id] = unique_timestamps
    all_rms = np.concatenate(all_rms_list, axis=1)

    all_rms_list_dict[mouse_id] = all_rms_list

# %%
# Plot RMS over time as an image (drift map)
for (mouse_id, all_rms_list), lims in zip(all_rms_list_dict.items(), [(68, 72), (66, 72), (66, 72)]):
    unique_timestamps = all_timestamps_dict[mouse_id]
    # Plot the RMS map
    f, axs = plt.subplots(1, len(all_rms_list), figsize=(13, 5), sharey=True)
    for i, rms_map in enumerate(all_rms_list):
        axs[i].imshow(rms_map, aspect='auto', origin='lower', cmap='viridis',
                    extent=[0, filtered.get_num_frames() / fs, 0, len(channel_ids)], vmin=lims[0], vmax=lims[1])
        axs[i].set_xlabel("Time (s)")
        axs[i].set_title(f"{unique_timestamps[i]}", fontsize=10)
    axs[0].set_ylabel("Channel")
    plt.suptitle(f"RMS for {mouse_id}")
    plt.colorbar(axs[0].images[0], ax=axs, orientation='vertical', label='RMS (uV)')
    plt.show()
    f.savefig(Path(r"D:\SNeurobiology\Desktop\multiday_ephys") / f"rms_{mouse_id}.png", dpi=300, bbox_inches='tight')
    plt.close(f)
# %%
