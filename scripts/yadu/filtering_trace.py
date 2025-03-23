#%%
%matplotlib widget
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spkp
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

#%% Path to an OpenEphys recording:
recording_path = Path(r'Y:\20250124\M20\test_npx1\2025-01-24_19-56-04')

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

#now lets plot the trace and see our data
plt.figure()
# plt.plot(time_trace, npx1_trace[:, 0])
plt.plot(time_trace, npx1_trace)

#%%
# Define 1-second trace extraction points: beginning, middle, and end
total_frames = recording_npx1.get_num_frames()
sampling_frequency = recording_npx1.get_sampling_frequency()
time_range = 0.20
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

#%%
rec = spkp.phase_shift(recording = recording_npx1 , margin_ms=40.0, inter_sample_shift=None, dtype=None)
#%%
rec_cmr = spkp.common_reference(recording=recording_npx1, operator="median", reference="global")
# %%
npx1_trace = rec_cmr.get_traces(start_frame=start_frame_cust, end_frame=start_frame_cust + n_samples, channel_ids=['AP345'])
total_frames = recording_npx1.get_num_frames()
sampling_frequency = recording_npx1.get_sampling_frequency()
time_range = 0.20
n_samples = int(time_range*sampling_frequency)  # 1 second worth of samples
start_frames = [start_frame_cust, total_frames // 2 - n_samples // 2, total_frames - n_samples]

# Extract traces for each point
traces = []
time_traces = []
for start_frame in start_frames:
    trace = rec_cmr.get_traces(start_frame=start_frame, end_frame=start_frame + n_samples)
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
# %%
rec = spkp.highpass_filter(recording=recording_npx1)
rec = spkp.phase_shift(recording=rec)
# %%
bad_channel_ids = spkp.detect_bad_channels(recording=rec)
#%%
bad_channel_ids
bad_channel_ids = np.where(bad_channel_ids == 'bad')[0]
bad_channel_ids
# %%
rec = spkp.interpolate_bad_channels(recording=rec, bad_channel_ids=bad_channel_ids)

# %%
rec = spkp.highpass_spatial_filter(recording=rec)

# %%
traces = []
time_traces = []
for start_frame in start_frames:
    trace = rec.get_traces(start_frame=start_frame, end_frame=start_frame + n_samples)
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

# %%
kilosort

# %%
import kilosort
# %%
from kilosort import run_kilosort
# %%
from kilosort import run_kilosort, DEFAULT_SETTINGS

settings = DEFAULT_SETTINGS
# ( path to drive if mounted: /content/drive/MyDrive/ )
settings['data_dir'] = r'Y:\20250124\M20\test_npx1\2025-01-24_19-56-04\Record Node 102\experiment1\recording1\continuous\Neuropix-PXI-100.ProbeB-AP'
settings['n_chan_bin'] = 384

ops, st, clu, tF, Wall, similar_templates, is_ref, est_contam_rate, kept_spikes = \
    run_kilosort(settings=settings, probe_name='neuropixPhase3B1_kilosortChanMap.mat')
 # %%
import pandas as pd
#%%

# outputs saved to results_dir
results_dir = Path(r'Y:\20250124\M20\test_npx1\2025-01-24_19-56-04\Record Node 102\experiment1\recording1\continuous\Neuropix-PXI-100.ProbeB-AP\kilosort4')
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
# %%
%matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import gridspec, rcParams
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
gray = .5 * np.ones(3)

fig = plt.figure(figsize=(10,10), dpi=100)
grid = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.5)

ax = fig.add_subplot(grid[0,0])
ax.plot(np.arange(0, ops['Nbatches'])*2, dshift);
ax.set_xlabel('time (sec.)')
ax.set_ylabel('drift (um)')

ax = fig.add_subplot(grid[0,1:])
t0 = 0
t1 = np.nonzero(st > ops['fs']*5)[0][0]
ax.scatter(st[t0:t1]/30000., chan_best[clu[t0:t1]], s=0.5, color='k', alpha=0.25)
ax.set_xlim([0, 5])
ax.set_ylim([chan_map.max(), 0])
ax.set_xlabel('time (sec.)')
ax.set_ylabel('channel')
ax.set_title('spikes from units')

ax = fig.add_subplot(grid[1,0])
nb=ax.hist(firing_rates, 20, color=gray)
ax.set_xlabel('firing rate (Hz)')
ax.set_ylabel('# of units')

ax = fig.add_subplot(grid[1,1])
nb=ax.hist(camps, 20, color=gray)
ax.set_xlabel('amplitude')
ax.set_ylabel('# of units')

ax = fig.add_subplot(grid[1,2])
nb=ax.hist(np.minimum(100, contam_pct), np.arange(0,105,5), color=gray)
ax.plot([10, 10], [0, nb[0].max()], 'k--')
ax.set_xlabel('% contamination')
ax.set_ylabel('# of units')
ax.set_title('< 10% = good units')

for k in range(2):
    ax = fig.add_subplot(grid[2,k])
    is_ref = contam_pct<10.
    ax.scatter(firing_rates[~is_ref], camps[~is_ref], s=3, color='r', label='mua', alpha=0.25)
    ax.scatter(firing_rates[is_ref], camps[is_ref], s=3, color='b', label='good', alpha=0.25)
    ax.set_ylabel('amplitude (a.u.)')
    ax.set_xlabel('firing rate (Hz)')
    ax.legend()
    if k==1:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('loglog')

# %%
