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
probe = ops['probe']
# x and y position of probe sites
xc, yc = probe['xc'], probe['yc']
nc = 16 # number of channels to show
good_units = np.nonzero(contam_pct <= 0.1)[0]
mua_units = np.nonzero(contam_pct > 0.1)[0]


gstr = ['good', 'mua']
for j in range(2):
    print(f'~~~~~~~~~~~~~~ {gstr[j]} units ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('title = number of spikes from each unit')
    units = good_units if j==0 else mua_units
    fig = plt.figure(figsize=(12,3), dpi=150)
    grid = gridspec.GridSpec(2,20, figure=fig, hspace=0.25, wspace=0.5)

    for k in range(40):
        wi = units[np.random.randint(len(units))]
        wv = templates[wi].copy()
        cb = chan_best[wi]
        nsp = (clu==wi).sum()

        ax = fig.add_subplot(grid[k//20, k%20])
        n_chan = wv.shape[-1]
        ic0 = max(0, cb-nc//2)
        ic1 = min(n_chan, cb+nc//2)
        wv = wv[:, ic0:ic1]
        x0, y0 = xc[ic0:ic1], yc[ic0:ic1]

        amp = 4
        for ii, (xi,yi) in enumerate(zip(x0,y0)):
            t = np.arange(-wv.shape[0]//2,wv.shape[0]//2,1,'float32')
            t /= wv.shape[0] / 20
            ax.plot(xi + t, yi + wv[:,ii]*amp, lw=0.5, color='k')

        ax.set_title(f'{nsp}', fontsize='small')
        ax.axis('off')
    plt.show()
# %%
