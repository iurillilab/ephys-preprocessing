# %%
%matplotlib widget
import spikeinterface.extractors as se
from spikeinterface.core import get_noise_levels
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Path to a valid OpenEphys recording:
recording_path = Path("/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/setup-calibrations/ephys-mpm-rig_synch_test/long/2024-11-13_14-39-11")

# To read the recording, we need to look in the folder the name of the record node
# and the streams that we want to read. The full stream name is "Record Node XXX#stream name"
recording_npx1ap = se.read_openephys(recording_path, stream_name="Record Node 111#Neuropix-PXI-110.ProbeA-AP")
recording_npx1lfp = se.read_openephys(recording_path, stream_name="Record Node 111#Neuropix-PXI-110.ProbeA-LFP")
recording_npx2 = se.read_openephys(recording_path, stream_name="Record Node 111#Neuropix-PXI-110.ProbeB")
recording_daq = se.read_openephys(recording_path, stream_name="Record Node 111#NI-DAQmx-112.PXIe-6341")
dir(recording_daq)
# %%
noise_levels = {}
for rec, key in zip([recording_npx1ap, recording_npx1lfp, recording_npx2], ["npx1ap", "npx1lfp", "npx2"]):
    # restrict recording to 8 seconds
    n_seconds = 8
    n_samples = n_seconds * rec.get_sampling_frequency()
    rec = rec.frame_slice(start_frame=0, end_frame=n_samples)
    # (start_frame=0, end_frame=n_samples)

    noise_levels[key] = get_noise_levels(rec, method="std")
    print("================", key)
    print(noise_levels[key])


# %%
rec
# %%
rec = recording_npx1ap
n_seconds = 5
n_samples = n_seconds * rec.get_sampling_frequency()
channels = rec.channel_ids[:10]
trace_memmap = rec.get_traces(start_frame=0, end_frame=n_samples, channel_ids=channels, return_scaled=True)
trace = np.array(trace_memmap)

bins = np.arange(-100, 100, 5)
n_channels = trace.shape[1]
hist_mat = np.zeros((n_channels, bins.size - 1))
for i in tqdm(range(n_channels)):
    hist_mat[i], _ = np.histogram(trace[:, i], bins=bins, density=True)

hist_mat.shape
# %%
# plot all histograms as lines
plt.figure(figsize=(6, 6))
for i in range(n_channels):
    plt.plot(bins[:-1], hist_mat[i, :] + 0.02 *i)# %%

# %%
