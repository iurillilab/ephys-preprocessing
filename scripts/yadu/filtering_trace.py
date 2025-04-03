#%%
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spkp
import spikeinterface.full as si
from pathlib import Path
import numpy as np
import time
import os
import spikeinterface.sorters as ss
import pandas as pd

#%% Path to an OpenEphys recording:
parent_path = Path(r'Y:')
recording_path = parent_path/ '20250124'/ 'M20' / 'test_npx1' / '2025-01-24_19-56-04'
#%% To read the recording, we need to look in the folder the name of the record node and the streams that we want to read. The full stream name is "Record Node xxx#stream name"
# recording_daq = se.read_openephys(recording_path, stream_name="Record Node 102#NI-DAQmx-102.PXIe-6341")
recording_npx1 = se.read_openephys(recording_path, stream_name="Record Node 102#Neuropix-PXI-100.ProbeB-AP")

#%% Loading information on the recording
#Sampling frequency:
sampling_frequency = recording_npx1.get_sampling_frequency()

# Read channel info:
chan = recording_npx1.get_channel_ids() # channel names
chan.size

# # Read time duration info:
print("Number of frames:", recording_npx1.get_num_frames())  # number of samples
print("Duration in seconds:", recording_npx1.get_total_duration())

#%% We need to be careful with time info if we look at raw signals: the starting point of each stream do not necessarily match!
#print(recording_daq.get_time_info())
print(recording_npx1.get_time_info())

#%% We do not want to load the full traces in memory, We want to read the first initial seconds:lazy loading
n_seconds = 0.1
start_frame_cust = 1000000 + int(0.4*30000)
n_samples = int(n_seconds * recording_npx1.get_sampling_frequency())
npx1_trace = recording_npx1.get_traces(start_frame=start_frame_cust, end_frame=start_frame_cust + n_samples, channel_ids=['AP345']) 

# To really load the data in memory, we can use the np.array() function, Until we do this we do not have to wait for loading time from the disk! 
# Useful for big data on slow disks like the NAS.
npx1_trace = np.array(npx1_trace)

#if you dont have a DAQ trace, we create a time series to plot the npx trace based off on the size of the NPX trace
time_trace = np.arange(npx1_trace.shape[0]) / sampling_frequency
#%%
rec = spkp.phase_shift(recording = recording_npx1 , margin_ms=40.0, inter_sample_shift=None, dtype=None)
rec_cmr = spkp.common_reference(recording=recording_npx1, operator="median", reference="global")
# %%
rec = spkp.highpass_filter(recording=recording_npx1)
# %%
bad_channel_ids = spkp.detect_bad_channels(recording=rec)
bad_channel_ids = np.where(bad_channel_ids == 'bad')[0]
bad_channel_ids
# %%
rec = spkp.interpolate_bad_channels(recording=rec, bad_channel_ids=bad_channel_ids)
# %%
rec = spkp.highpass_spatial_filter(recording=rec)
# %%
folder_path= r'Y:\20250124\M20\test_npx1\2025-01-24_19-56-04\Record Node 102\experiment1\recording1\continuous\Neuropix-PXI-100.ProbeB-AP'
output_folder = r'Y:\20250124\M20\test_npx1\2025-01-24_19-56-04\Record Node 102\experiment1\recording1\continuous\Neuropix-PXI-100.ProbeB-AP\kilosort4'
sort = ss.run_sorter('kilosort4', rec, output_folder= output_folder, verbose=True)
 # %%
# outputs saved to results_dir
results_dir = Path(r'Y:\20250124\M20\test_npx1\2025-01-24_19-56-04\Record Node 102\experiment1\recording1\continuous\Neuropix-PXI-100.ProbeB-AP\kilosort4\sorter_output')
ops = np.load(results_dir / 'ops.npy', allow_pickle=True).item()
camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values
chan_map =  np.load(results_dir / 'channel_map.npy')
templates =  np.load(results_dir / 'templates.npy')
chan_best = (templates**2).sum(axis=1).argmax(axis=-1)
chan_best = chan_map[chan_best]
amplitudes = np.load(results_dir / 'amplitudes.npy')
st = np.load(results_dir / 'spike_times.npy')
clu = np.load(results_dir / 'spike_clusters.npy')
firing_rates = np.unique(clu, return_counts=True)[1] * 30000 / st.max()
dshift = ops['dshift']