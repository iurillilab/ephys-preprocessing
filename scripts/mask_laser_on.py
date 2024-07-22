# %%

data_path_list = ["/Volumes/Extreme SSD/P02_MPAOPTO_LP/e04_ephys-contrapag-stim/v01/M22/2024-04-23_10-39-40"]
run_barcodeSync = False
run_preprocessing = True # run preprocessing and spikesorting
callKSfromSI = False

# %%
%matplotlib widget
from matplotlib import pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.widgets as sw
import spikeinterface.preprocessing as st

from spikeinterface import get_noise_levels, aggregate_channels
from pathlib import Path
import os
import numpy as np

from preprocessing_utils import *
from nwb_conv.oephys import OEPhysDataFolder


# %%
oephys_data = OEPhysDataFolder(data_path_list[0])

all_stream_names, ap_stream_names = oephys_data.stream_names, oephys_data.ap_stream_names

# %%
npx_barcode = oephys_data.reference_npx_barcode
nidaq_data = oephys_data.nidaq_recording

nidaq_barcode = nidaq_data.barcode
laser_data = nidaq_data.continuous_signals["laser-log"]

reader = se.read_openephys(oephys_data.path, stream_name=oephys_data.ap_stream_names[0])

laser_idxs = nidaq_barcode.map_indexes_to(npx_barcode, laser_data.onsets)

# %%
chan_idx = 200
trace = reader.get_traces(channel_ids=[reader.get_channel_ids()[chan_idx]])
# %%
n_to_take = 200
skip = len(laser_idxs) // n_to_take
pre_int_sec = 0.05
post_int_sec = 0.05
fs = reader.sampling_frequency
pre_int = int(pre_int_sec * fs)
post_int = int(post_int_sec * fs)

cropped = np.zeros((len(laser_idxs[::skip]), pre_int + post_int))
for i, laser_idx in enumerate(laser_idxs[::skip]):
    cropped[i, :] = np.array(trace[laser_idx - pre_int:laser_idx + post_int]).flat

cropped.shape
plt.figure()
plt.imshow(cropped, cmap="RdBu_r", aspect="auto", vmin=-1000, vmax=1000)
plt.axvline(pre_int, color="k")


# %%
zeroed = st.RemoveArtifactsRecording(reader, laser_idxs, ms_after=12)
# %%
# zeroed_trace = zeroed.get_traces(channel_ids=[reader.get_channel_ids()[chan_idx]])
cropped = np.zeros((len(laser_idxs[::skip]), pre_int + post_int))
for i, laser_idx in enumerate(laser_idxs[::skip]):
    trace = zeroed.get_traces(start_frame=laser_idx - pre_int, 
                              end_frame=laser_idx + post_int,
                              channel_ids=[reader.get_channel_ids()[chan_idx]])
    trace = np.array(trace).flat
    trace = trace - np.mean(trace)
    cropped[i, :] = trace
    

cropped.shape
plt.figure()
# plt.plot(cropped, cmap="RdBu_r", aspect="auto", vmin=-1000, vmax=1000)
plt.plot(cropped.T)
plt.axvline(pre_int, color="k")

# %%
?zeroed.get_traces
# %%
