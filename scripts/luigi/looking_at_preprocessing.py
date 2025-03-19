# %%
%matplotlib widget
import matplotlib.pyplot as plt
import spikeinterface.extractors as se
from spikeinterface import aggregate_channels
import spikeinterface.preprocessing as st
import pandas as pd
from pathlib import Path
import numpy as np
data_path = Path("/Users/vigji/Desktop/mouse_data")

# %%
recording_extractor = se.read_openephys(data_path, stream_name="Record Node 111#Neuropix-PXI-110.ProbeA")
recording = st.correct_lsb(recording_extractor, verbose=1)
recording = st.bandpass_filter(recording, freq_min=300, freq_max=4000)
recording = st.phase_shift(recording) #lazy
[bad_channel_ids, channel_labels] = st.detect_bad_channels(recording=recording)  
recording = recording.remove_channels(remove_channel_ids=bad_channel_ids)  # could be interpolated instead, but why?

# split in groups and apply spatial filtering, then reaggregate. KS4 can now handle multiple shanks
if len(recording.get_probes()) > 1 or np.unique(recording.get_probes()[0].shank_ids).size > 1:
    grouped_recordings = recording.split_by(property='group')
    recgrouplist_hpsf = [st.highpass_spatial_filter(recording=grouped_recordings[k]) for k in grouped_recordings.keys()]  # cmr is slightly faster. results are similar
    recording_hpsf = aggregate_channels(recgrouplist_hpsf)
else:
    recording_hpsf = st.highpass_spatial_filter(recording=recording)

# %%

start_time = 150
window_range = 0.5
vlim = 20
start_frame = int(start_time * recording_hpsf.get_sampling_frequency())
end_frame = start_frame + int(window_range * recording_hpsf.get_sampling_frequency())

# Create side by side comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

# Get traces from both recordings
traces_raw = recording.get_traces(start_frame=start_frame, end_frame=end_frame)
traces_filtered = recording_hpsf.get_traces(start_frame=start_frame, end_frame=end_frame)

# Plot raw data
im1 = ax1.imshow(traces_raw.T, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
ax1.set_title("Raw Recording")

# Plot filtered data
im2 = ax2.imshow(traces_filtered.T, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
ax2.set_title("High-pass Spatially Filtered")

# Add a single colorbar
# fig.colorbar(im2, ax=(ax1, ax2), location='right', shrink=0.8, label='Amplitude')

plt.tight_layout()

# %%

p = recording.get_probes()[0]

# %%

# %%
