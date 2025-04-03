#%%
%matplotlib widget
import spikeinterface.extractors as se
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%
# Path to a valid OpenEphys recording:
recording_path = Path(r'Y:\20250124\M20\test_npx1\2025-01-24_19-56-04')
#%%
# To read the recording, we need to look in the folder the name of the record node
# and the streams that we want to read. The full stream name is "Record Node xxx#stream name"
# recording_daq = se.read_openephys(recording_path, stream_name="Record Node 102#NI-DAQmx-102.PXIe-6341")
recording_npx1 = se.read_openephys(recording_path, stream_name="Record Node 102#Neuropix-PXI-100.ProbeB-AP")
# recording_npx2 = se.read_openephys(recording_path, stream_name="Record Node 107#Neuropix-PXI-100.ProbeB-LFP")

# %%
dir(recording_npx1)
# %%
# Loading information on the recording
# Sampling frequency:
recording_npx1.get_sampling_frequency()

# Read channel info:
chan = recording_npx1.get_channel_ids() # channel names
chan.size

# # Read time duration info:
print("Number of frames:", recording_npx1.get_num_frames())  # number of samples
print("Duration in seconds:", recording_npx1.get_total_duration())

# %%
# We need to be careful with time info if we look at raw signals: the starting point of each stream do not necessarily match!
# print(recording_daq.get_time_info())
print(recording_npx1.get_time_info())
#%%
# We do not want to load the full traces in memory:
# We want to read the first initial seconds:
sampling_frequency = recording_npx1.get_sampling_frequency()
n_seconds = 0.1
start_frame_cust = 1000000 + int(0.4*30000)
n_samples = int(n_seconds * recording_npx1.get_sampling_frequency())
npx1_trace = recording_npx1.get_traces(start_frame=start_frame_cust, end_frame=start_frame_cust + n_samples, channel_ids=['AP345']) 

# To really load the data in memory, we can use the np.array() function.
# Until we do this we do not have to wait for loading time from the disk! 
# Useful for big data on slow disks like the NAS.
# trace_numpy = np.array(trace)
# type(trace_numpy)
# trace_numpy
npx1_trace = np.array(npx1_trace)

#if you dont have a DAQ trace, we create a time series to plot the npx1 trace based off on the size of the NPX trace
time_trace = np.arange(npx1_trace.shape[0]) / sampling_frequency

#now lets plot the trace and see our data
plt.figure()
# plt.plot(time_trace, npx1_trace[:, 0])
plt.plot(time_trace, npx1_trace)
#%%
len(time_trace)
#%%
# Define 1-second trace extraction points: beginning, middle, and end
total_frames = recording_npx1.get_num_frames()
sampling_frequency = recording_npx1.get_sampling_frequency()
time_range = 1
n_samples = int(time_range*sampling_frequency)  # 1 second worth of samples
start_frames = [start_frame_cust, total_frames // 2 - n_samples // 2, total_frames - n_samples]

# Extract traces for each point
traces = []
time_traces = []
for start_frame in start_frames:
    trace = recording_npx1.get_traces(start_frame=start_frame, end_frame=start_frame + n_samples)
    trace = np.array(trace)
    traces.append(trace)
    time_traces.append(np.arange(trace.shape[0]) / sampling_frequency)

# Plot the traces using imshow
plt.figure(figsize=(10, 6))
labels = ["Beginning", "Middle", "End"]
for i, trace in enumerate(traces):
    plt.subplot(1, 3, i + 1)  # Create a subplot for each trace
    time_extent = [0, trace.shape[0] / sampling_frequency]  # Scale x-axis to seconds
    channel_extent = [0, trace.shape[1]]  # Channels on y-axis
    plt.imshow(trace.T, aspect='auto', cmap='RdBu', vmin=-300, vmax=300, extent=[*time_extent, *channel_extent])
    plt.title(f'{labels[i]} (1s)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Channels')
    plt.colorbar(label='Amplitude')

plt.tight_layout()
plt.show()

# Convert results_dir to a Path object
results_dir = Path(r'Y:\20250124\M20\test_npx1\2025-01-24_19-56-04\Record Node 102\experiment1\recording1\continuous\Neuropix-PXI-100.ProbeB-AP\kilosort4')

# Load ops.npy
ops = np.load(results_dir / 'ops.npy', allow_pickle=True).item()

# Load other data
camps = pd.read_csv(results_dir / 'cluster_Amplitude.tsv', sep='\t')['Amplitude'].values
contam_pct = pd.read_csv(results_dir / 'cluster_ContamPct.tsv', sep='\t')['ContamPct'].values

#%%
