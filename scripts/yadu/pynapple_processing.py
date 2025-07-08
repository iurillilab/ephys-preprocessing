#%%
%matplotlib widget
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pynapple as nap
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import ttest_1samp
# %%
parent_path = Path(r'Y:/Anaesthetised')
nwbfile_path = parent_path / 'M26_D879' / '20250307' / 'M26_D879.nwb'
#%%
#load the nwb file using pynapple
data = nap.load_file(nwbfile_path)
data
#%%
#initialise the pynapple object to access the data
data.key_to_id
# %%
# Accessing the spikes, trials, and video data and adding them to the data object
#printing the dataset to visualise the data
spikes = data["units"]
trials = data["trials"]
video = data["TopCamera__video_2025-03-07T16_14_50.avi"]
print("Spikes:", spikes, "\nTrials:", trials, "\nVideo:", video)
#%%
spike_tsd = spikes.to_tsd()
unit_id = spikes["unit_name"]
# %%
#trial parameters
start_time = trials["start"]
start_time_ts = nap.Ts(trials['start'])
end_time = trials["end"]
end_time_ts = nap.Ts(trials['end'])
set_movement_on = trials["set_movement_on"]
set_movement_off = trials["set_movement_off"]
home_movement_on = trials["home_movement_on"]
home_movement_off = trials["home_movement_off"]
radius = trials["radius"]
theta = trials["theta"]
#%%
# Plotting the spike times with unit IDs and shading each trial/section
plt.figure(figsize=(10, 6))
plt.plot(spike_tsd.index.to_numpy(), spike_tsd.to_numpy(), '|', markersize=1, alpha=0.5)
plt.xlabel('Spike Time (s)')
plt.ylabel('Unit ID')

# Shade each trial/section between start and end
for s, e in zip(start_time, end_time):
    plt.axvspan(s, e, color='orange', alpha=0.7)

plt.title('All spikes with unit IDs and recording section')
plt.show()
#%%
# Plotting the spike times with unit IDs aligned to the first event in the trials
unit_ids = np.unique(spike_tsd.values)
event_time = end_time_ts.index[1]  # Get the scalar time

plt.figure(figsize=(10, 6))

for i, unit in enumerate(unit_ids):
    unit_spikes = spike_tsd[spike_tsd.values == unit]
    aligned_spikes = unit_spikes.index - event_time
    mask = (aligned_spikes >= -10) & (aligned_spikes <= 10)
    plt.plot(aligned_spikes[mask], np.full(np.sum(mask), unit), '|', markersize=5, color='black')

plt.xlabel("Time from event (s)")
plt.ylabel("Unit ID")
plt.title("Spikes for all units aligned to single event")
plt.xlim(-0.5, 0.5)
plt.axvline(0.0, color='blue')
plt.show()
#%%
#average firing rate for neurons around trials in a 10s window with 100ms bins
# Parameters
window = 10  # seconds before and after
bin_size = 0.1  # 100 ms
bins = np.arange(-window, window + bin_size, bin_size)
bin_centers = bins[:-1] + bin_size / 2

unit_ids = np.unique(spike_tsd.values)
n_units = len(unit_ids)
n_bins = len(bins) - 1

# Initialize: rows=units, cols=bins
fr_matrix = np.zeros((n_units, n_bins))

for i, unit in enumerate(unit_ids):
    unit_spikes = spike_tsd[spike_tsd.values == unit].index.values
    aligned_counts = np.zeros(n_bins)
    n_trials = 0
    for s in start_time:
        # Align spikes to trial start
        aligned_spikes = unit_spikes - s
        # Only keep spikes within window
        mask = (aligned_spikes >= -window) & (aligned_spikes <= window)
        aligned_spikes = aligned_spikes[mask]
        # Histogram
        counts, _ = np.histogram(aligned_spikes, bins=bins)
        aligned_counts += counts
        n_trials += 1
    # Average firing rate per bin (Hz)
    fr_matrix[i, :] = aligned_counts / (n_trials * bin_size)

# Plot: average across all units
mean_fr = np.mean(fr_matrix, axis=0)
sem_fr = np.std(fr_matrix, axis=0) / np.sqrt(n_units)

plt.figure(figsize=(10, 6))
plt.plot(bin_centers, mean_fr, label='Mean firing rate')
plt.fill_between(bin_centers, mean_fr - sem_fr, mean_fr + sem_fr, alpha=0.3, label='SEM')
plt.axvline(0, color='red', linestyle='--', label='Trial start')
plt.xlabel('Time from trial start (s)')
plt.ylabel('Firing rate (Hz)')
plt.title('Average firing rate around trial start (all units)')
plt.legend()
plt.tight_layout()
plt.show()
#%%
#average firing rate for neurons around each events, 2 seconds before and after the event in a 100ms bin

window = 2  # seconds before and after
bin_size = 0.1  # 100 ms
bins = np.arange(-window, window + bin_size, bin_size)
bin_centers = bins[:-1] + bin_size / 2

event_dict = {
    "set_movement_on": set_movement_on,
    "set_movement_off": set_movement_off,
    "home_movement_on": home_movement_on,
    "home_movement_off": home_movement_off
}

unit_ids = np.unique(spike_tsd.values)
n_units = len(unit_ids)
n_bins = len(bins) - 1

fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axs = axs.flatten()

for idx, (event_name, event_times) in enumerate(event_dict.items()):
    fr_matrix = np.zeros((n_units, n_bins))
    for i, unit in enumerate(unit_ids):
        unit_spikes = spike_tsd[spike_tsd.values == unit].index.values
        aligned_counts = np.zeros(n_bins)
        n_events = 0
        for s in event_times:
            aligned_spikes = unit_spikes - s
            mask = (aligned_spikes >= -window) & (aligned_spikes <= window)
            aligned_spikes = aligned_spikes[mask]
            counts, _ = np.histogram(aligned_spikes, bins=bins)
            aligned_counts += counts
            n_events += 1
        if n_events > 0:
            fr_matrix[i, :] = aligned_counts / (n_events * bin_size)
    mean_fr = np.mean(fr_matrix, axis=0)
    sem_fr = np.std(fr_matrix, axis=0) / np.sqrt(n_units)
    axs[idx].plot(bin_centers, mean_fr, label='Mean firing rate')
    axs[idx].fill_between(bin_centers, mean_fr - sem_fr, mean_fr + sem_fr, alpha=0.3, label='SEM')
    axs[idx].axvline(0, color='red', linestyle='--', label='Event')
    axs[idx].set_title(event_name)
    axs[idx].set_xlabel('Time from event (s)')
    axs[idx].set_ylabel('Firing rate (Hz)')
    axs[idx].legend()

plt.suptitle('Average firing rate around each event (all units)')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
#%%
#z-score and make a tuning map for each neuron and position

from scipy.stats import zscore

unit_ids = np.unique(spike_tsd.values)
positions = list(set(zip(radius, theta)))  # unique (radius, theta) pairs

# --- 1s window centered on start time ---
all_firing_rates_start = {unit: [] for unit in unit_ids}
trial_info_start = {unit: [] for unit in unit_ids}

for unit in unit_ids:
    unit_spikes = spike_tsd[spike_tsd.values == unit]
    for pos in positions:
        r, t = pos
        trial_mask = (radius == r) & (theta == t)
        trial_starts = np.array(start_time)[trial_mask]
        for s in trial_starts:
            win_start = s - 0.5
            win_end = s + 0.5
            duration = win_end - win_start
            if duration > 0:
                count = np.sum((unit_spikes.index >= win_start) & (unit_spikes.index < win_end))
                rate = count / duration
                all_firing_rates_start[unit].append(rate)
                trial_info_start[unit].append(pos)

zscore_maps_start = {}
for unit in unit_ids:
    rates = np.array(all_firing_rates_start[unit])
    if len(rates) > 1:
        zscores = zscore(rates)
    else:
        zscores = np.zeros_like(rates)
    tuning_map = {}
    for pos, z in zip(trial_info_start[unit], zscores):
        if pos not in tuning_map:
            tuning_map[pos] = []
        tuning_map[pos].append(z)
    for pos in tuning_map:
        tuning_map[pos] = np.mean(tuning_map[pos])
    zscore_maps_start[unit] = tuning_map

tuning_df_start = pd.DataFrame(zscore_maps_start).T
tuning_df_start.index.name = "unit_id"
tuning_df_start.columns = pd.MultiIndex.from_tuples(tuning_df_start.columns, names=["radius", "theta"])

# --- 1s window centered on end time ---
all_firing_rates_end = {unit: [] for unit in unit_ids}
trial_info_end = {unit: [] for unit in unit_ids}

for unit in unit_ids:
    unit_spikes = spike_tsd[spike_tsd.values == unit]
    for pos in positions:
        r, t = pos
        trial_mask = (radius == r) & (theta == t)
        trial_ends = np.array(end_time)[trial_mask]
        for e in trial_ends:
            win_start = e - 0.5
            win_end = e + 0.5
            duration = win_end - win_start
            if duration > 0:
                count = np.sum((unit_spikes.index >= win_start) & (unit_spikes.index < win_end))
                rate = count / duration
                all_firing_rates_end[unit].append(rate)
                trial_info_end[unit].append(pos)

zscore_maps_end = {}
for unit in unit_ids:
    rates = np.array(all_firing_rates_end[unit])
    if len(rates) > 1:
        zscores = zscore(rates)
    else:
        zscores = np.zeros_like(rates)
    tuning_map = {}
    for pos, z in zip(trial_info_end[unit], zscores):
        if pos not in tuning_map:
            tuning_map[pos] = []
        tuning_map[pos].append(z)
    for pos in tuning_map:
        tuning_map[pos] = np.mean(tuning_map[pos])
    zscore_maps_end[unit] = tuning_map

tuning_df_end = pd.DataFrame(zscore_maps_end).T
tuning_df_end.index.name = "unit_id"
tuning_df_end.columns = pd.MultiIndex.from_tuples(tuning_df_end.columns, names=["radius", "theta"])

print("Tuning matrix around start time:\n", tuning_df_start)
print("Tuning matrix around end time:\n", tuning_df_end)
#%%
#tuning map before Z-score
# unit_ids = np.unique(spike_tsd.values)
# positions = list(set(zip(radius, theta)))  # unique (radius, theta) pairs

# # Prepare a DataFrame to store the tuning map for each neuron
# tuning_maps = {}

# for unit in unit_ids:
#     tuning_map = {}
#     unit_spikes = spike_tsd[spike_tsd.values == unit]
#     for pos in positions:
#         r, t = pos
#         # Find trials with this (radius, theta)
#         trial_mask = (radius == r) & (theta == t)
#         trial_starts = np.array(start_time)[trial_mask]
#         trial_ends = np.array(end_time)[trial_mask]
#         firing_rates = []
#         for s, e in zip(trial_starts, trial_ends):
#             # Count spikes in this trial for this unit
#             count = np.sum((unit_spikes.index >= s) & (unit_spikes.index < e))
#             duration = e - s  # bin size in seconds
#             if duration > 0:
#                 firing_rates.append(count / duration)
#         # Average firing rate for this position
#         if len(firing_rates) > 0:
#             tuning_map[pos] = np.mean(firing_rates)
#         else:
#             tuning_map[pos] = np.nan  # or 0
#     tuning_maps[unit] = tuning_map

# # Convert to DataFrame: rows=unit, columns=(radius, theta), values=mean firing rate (Hz)
# tuning_df = pd.DataFrame(tuning_maps).T
# tuning_df.index.name = "unit_id"
# tuning_df.columns = pd.MultiIndex.from_tuples(tuning_df.columns, names=["radius", "theta"])
# print(tuning_df)
# %%# Plotting the tuning map for a specific neuron
# Pick a neuron to show (e.g., the first one)
unit = tuning_df_start.index[93]
tuning = tuning_df_start.loc[unit]

# Convert MultiIndex columns to a matrix for plotting
radii = sorted(set([r for r, t in tuning_df_start.columns]))
thetas = sorted(set([t for r, t in tuning_df_start.columns]))
tuning_matrix = np.full((len(radii), len(thetas)), np.nan)

for i, r in enumerate(radii):
    for j, t in enumerate(thetas):
        if (r, t) in tuning:
            tuning_matrix[i, j] = tuning[(r, t)]

plt.figure(figsize=(8, 6))
im = plt.imshow(
    tuning_matrix,
    aspect='auto',
    origin='lower',
    extent=[min(thetas), max(thetas), min(radii), max(radii)],
    cmap='viridis'
)
plt.colorbar(im, label='Avg. spike count')
plt.xlabel('Theta')
plt.ylabel('Radius')
plt.title(f'Tuning map for unit {unit}')
plt.show()
# %%
# Calculate the delta firing rate for each unit and position
delta_maps = {}

for unit in unit_ids:
    delta_map = {}
    unit_spikes = spike_tsd[spike_tsd.values == unit]
    for pos in positions:
        r, t = pos
        trial_mask = (radius == r) & (theta == t)
        trial_starts = np.array(start_time)[trial_mask]
        trial_ends = np.array(end_time)[trial_mask]
        delta_rates = []
        for s, e in zip(trial_starts, trial_ends):
            duration = e - s
            if duration > 0:
                # Firing rate after (during trial)
                count_after = np.sum((unit_spikes.index >= s) & (unit_spikes.index < e))
                rate_after = count_after / duration
                # Firing rate before (same duration before trial)
                before_start = s - duration
                before_end = s
                count_before = np.sum((unit_spikes.index >= before_start) & (unit_spikes.index < before_end))
                rate_before = count_before / duration
                # Delta firing rate
                delta_rates.append(rate_after - rate_before)
        # Average delta for this position
        if len(delta_rates) > 0:
            delta_map[pos] = np.mean(delta_rates)
        else:
            delta_map[pos] = np.nan
    delta_maps[unit] = delta_map

# Convert to DataFrame: rows=unit, columns=(radius, theta), values=mean delta firing rate (Hz)
delta_df = pd.DataFrame(delta_maps).T
delta_df.index.name = "unit_id"
delta_df.columns = pd.MultiIndex.from_tuples(delta_df.columns, names=["radius", "theta"])
print(delta_df)
#%%
# Perform a t-test for each unit and position to see if the delta firing rate is significantly different from zero
# Prepare a DataFrame to store p-values for each unit and (radius, theta)
unit_pval_map = {}

for unit in delta_maps:
    pval_map = {}
    for pos in delta_df.columns:
        # For this unit and position, get all delta rates (across trials)
        r, t = pos
        # Find all trials for this unit and position
        trial_mask = (radius == r) & (theta == t)
        trial_starts = np.array(start_time)[trial_mask]
        trial_ends = np.array(end_time)[trial_mask]
        unit_spikes = spike_tsd[spike_tsd.values == unit]
        delta_rates = []
        for s, e in zip(trial_starts, trial_ends):
            duration = e - s
            if duration > 0:
                count_after = np.sum((unit_spikes.index >= s) & (unit_spikes.index < e))
                rate_after = count_after / duration
                before_start = s - duration
                before_end = s
                count_before = np.sum((unit_spikes.index >= before_start) & (unit_spikes.index < before_end))
                rate_before = count_before / duration
                delta_rates.append(rate_after - rate_before)
        # t-test for this unit and position
        if len(delta_rates) > 1:
            t_stat, p_val = ttest_1samp(delta_rates, 0)
            pval_map[pos] = p_val
        else:
            pval_map[pos] = np.nan
    unit_pval_map[unit] = pval_map

# Convert to DataFrame: rows=unit, columns=(radius, theta), values=p-value
unit_pval_df = pd.DataFrame(unit_pval_map).T
unit_pval_df.index.name = "unit_id"
unit_pval_df.columns = pd.MultiIndex.from_tuples(unit_pval_df.columns, names=["radius", "theta"])
print(unit_pval_df)
#%%
# Plotting the firing rate map for a specific neuron (gradient visualization)

from scipy.ndimage import gaussian_filter

# Pick a neuron to show (e.g., the 93rd one)
unit = tuning_df.index[100]  # or tuning_df_z for z-scored
tuning = tuning_df.loc[unit]  # or tuning_df_z.loc[unit]

# Prepare the matrix for plotting
radii = sorted(set([r for r, t in tuning_df.columns]))
thetas = sorted(set([t for r, t in tuning_df.columns]))
tuning_matrix = np.full((len(radii), len(thetas)), np.nan)

for i, r in enumerate(radii):
    for j, t in enumerate(thetas):
        if (r, t) in tuning:
            tuning_matrix[i, j] = tuning[(r, t)]

# Smooth the matrix for better gradients
sigma = 1.0  # Increase for more smoothing if needed
tuning_matrix_smooth = gaussian_filter(tuning_matrix, sigma=sigma, mode='nearest')

plt.figure(figsize=(8, 6))
im = plt.imshow(
    tuning_matrix_smooth,
    aspect='auto',
    origin='lower',
    cmap='turbo',  # vibrant, multi-color gradient
    vmin=np.nanmin(tuning_matrix_smooth),
    vmax=np.nanmax(tuning_matrix_smooth)
)
plt.colorbar(im, label='Firing rate (Hz)' if tuning_df is not tuning_df_z else 'Z-scored firing rate')
plt.xticks(np.arange(len(thetas)), [f"{t:.2f}" for t in thetas])
plt.yticks(np.arange(len(radii)), [f"{r:.2f}" for r in radii])
plt.xlabel('Theta')
plt.ylabel('Radius')
plt.title(f'Firing rate map for unit {unit}')

# Optionally, overlay the original (unsmoothed) values as text
for i in range(len(radii)):
    for j in range(len(thetas)):
        val = tuning_matrix[i, j]
        if not np.isnan(val):
            plt.text(
                j, i, f"{val:.2g}",
                ha='center', va='center',
                color='white' if val > np.nanmean(tuning_matrix) else 'black',
                fontsize=8, fontweight='bold' if val > np.nanmean(tuning_matrix) else 'normal'
            )

plt.tight_layout()
plt.show()
#%%
# Plotting the p-value map for a specific neuron
# Pick a unit to show (e.g., the first one)
unit = unit_pval_df.index[93]
pval_map = unit_pval_df.loc[unit]

# Prepare the matrix for plotting
radii = sorted(set([r for r, t in unit_pval_df.columns]))
thetas = sorted(set([t for r, t in unit_pval_df.columns]))
pval_matrix = np.full((len(radii), len(thetas)), np.nan)

for i, r in enumerate(radii):
    for j, t in enumerate(thetas):
        if (r, t) in pval_map:
            pval_matrix[i, j] = pval_map[(r, t)]
plt.figure(figsize=(8, 6))
im = plt.imshow(
    pval_matrix,
    aspect='auto',
    origin='lower',
    cmap='viridis_r',  # reversed so low p-values are dark
    vmin=0, vmax=1
)
plt.colorbar(im, label='p-value')

# Set axis ticks to actual theta/radius values
plt.xticks(np.arange(len(thetas)), [f"{t:.2f}" for t in thetas])
plt.yticks(np.arange(len(radii)), [f"{r:.2f}" for r in radii])
plt.xlabel('Theta')
plt.ylabel('Radius')
plt.title(f'P-value map for unit {unit}')

# Add p-value text to each cell, centered
for i in range(len(radii)):
    for j in range(len(thetas)):
        val = pval_matrix[i, j]
        if not np.isnan(val):
            plt.text(
                j, i, f"{val:.2g}",
                ha='center', va='center',
                color='white' if val < 0.01 else 'black',
                fontsize=8, fontweight='bold' if val < 0.01 else 'normal'
            )

plt.tight_layout()
plt.show()
# %%
# unit_pval_df: rows=unit, columns=(radius, theta), values=p-value
radii = sorted(set([r for r, t in unit_pval_df.columns]))
thetas = sorted(set([t for r, t in unit_pval_df.columns]))
pval_matrix = np.full((len(radii), len(thetas)), np.nan)

# Compute proportion of units with p < 0.05 for each (radius, theta)
prop_matrix = np.full((len(radii), len(thetas)), np.nan)
for i, r in enumerate(radii):
    for j, t in enumerate(thetas):
        col = (r, t)
        if col in unit_pval_df.columns:
            pvals = unit_pval_df[col].dropna()
            if len(pvals) > 0:
                prop_matrix[i, j] = np.mean(pvals < 0.01)

plt.figure(figsize=(8, 6))
im = plt.imshow(
    prop_matrix,
    aspect='auto',
    origin='lower',
    cmap='plasma',
    vmin=0, vmax=1
)
plt.colorbar(im, label='Proportion of significant units (p < 0.05)')
plt.xticks(np.arange(len(thetas)), [f"{t:.2f}" for t in thetas])
plt.yticks(np.arange(len(radii)), [f"{r:.2f}" for r in radii])
plt.xlabel('Theta')
plt.ylabel('Radius')
plt.title('Proportion of significant units per position')
plt.tight_layout()
plt.show()
# %%
# Plotting p-value histograms for each (radius, theta) combination

# Prepare grid
nrows = len(radii)
ncols = len(thetas)
fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2.5*nrows), sharex=True, sharey=True)

for i, r in enumerate(radii):
    for j, t in enumerate(thetas):
        col = (r, t)
        ax = axes[i, j] if nrows > 1 and ncols > 1 else axes[max(i, j)]
        if col in unit_pval_df.columns:
            pvals = unit_pval_df[col].dropna()
            if len(pvals) > 0:
                # Histogram bins
                bins = np.linspace(0, 1, 51)
                counts, edges = np.histogram(pvals, bins=bins)
                # Highlight bars with p < 0.05
                bar_colors = ['red' if edges[k] < 0.01 else 'gray' for k in range(len(counts))]
                ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge', color=bar_colors, edgecolor='black')
                print("p-values < 0.01:", len(pvals[pvals < 0.01].values))
        ax.set_title(f"r={r:.2f}, θ={t:.2f}")
        if i == nrows - 1:
            ax.set_xlabel("p-value")
        if j == 0:
            ax.set_ylabel("Number of units")
        ax.set_xlim(0, 0.5)
plt.tight_layout()
plt.suptitle("P-value histograms for each (radius, theta) combination\n(red: p < 0.01)", y=1.02)
# plt.show()
# %%
# Plotting -log10(p) histograms for each (radius, theta) combination
radii = sorted(set([r for r, t in unit_pval_df.columns]))
thetas = sorted(set([t for r, t in unit_pval_df.columns]))
pval_matrix = np.full((len(radii), len(thetas)), np.nan)

# Prepare grid
nrows = len(radii)
ncols = len(thetas)
fig, axes = plt.subplots(nrows, ncols, figsize=(3*ncols, 2.5*nrows), sharex=True, sharey=True)

for i, r in enumerate(radii):
    for j, t in enumerate(thetas):
        col = (r, t)
        ax = axes[i, j] if nrows > 1 and ncols > 1 else axes[max(i, j)]
        if col in unit_pval_df.columns:
            pvals = unit_pval_df[col].dropna()
            if len(pvals) > 0:
                pvals = np.clip(pvals, 1e-20, 1)
                logp = -np.log10(pvals)
                bins = np.linspace(0, 10, 100)
                counts, edges = np.histogram(logp, bins=bins)
                # Highlight bars with -log10(p) > 2 (p < 0.01)
                bar_colors = ['red' if (edges[k] + edges[k+1]) / 2 > 2 else 'gray' for k in range(len(counts))]
                ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge', color=bar_colors, edgecolor='black')
                ax.axvline(2, color='blue', linestyle='--', linewidth=1)
                # Count and annotate number of significant neurons
                n_sig = np.sum(pvals < 0.01)
                ax.text(
                    0.98, 0.95, f"n={n_sig}", 
                    ha='right', va='top', transform=ax.transAxes,
                    fontsize=10, color='red', fontweight='bold'
                )
        ax.set_title(f"r={r:.2f}, θ={t:.2f}")
        if i == nrows - 1:
            ax.set_xlabel("-log10(p-value)")
        if j == 0:
            ax.set_ylabel("Number of units")
        ax.set_xlim(0, 5)
plt.tight_layout()
plt.suptitle("-log10(p) histograms for each (radius, theta) combination\n(red: p < 0.01)", y=1.02)
plt.show()
#%%
# 1. Select units with p < 0.05 in at least one (radius, theta)
sig_units = unit_pval_df.index[(unit_pval_df < 0.05).any(axis=1)]

# 2. Build feature matrix X from delta_df for significant units
# delta_df: rows=unit, columns=(radius, theta)
# We want X: (n_trials, n_units)
# For each trial, get the delta firing rate for the corresponding (radius, theta) for each unit

# Build X: (n_trials, n_units), each entry is the delta firing rate for that trial and unit
n_trials = len(trials)
n_units = len(sig_units)
X = np.zeros((n_trials, n_units))

for i, unit in enumerate(sig_units):
    unit_spikes = spike_tsd[spike_tsd.values == unit]
    for j, (s, e) in enumerate(zip(trials['start'], trials['end'])):
        duration = e - s
        if duration > 0:
            # Firing rate after (during trial)
            count_after = np.sum((unit_spikes.index >= s) & (unit_spikes.index < e))
            rate_after = count_after / duration
            # Firing rate before (same duration before trial)
            before_start = s - duration
            before_end = s
            count_before = np.sum((unit_spikes.index >= before_start) & (unit_spikes.index < before_end))
            rate_before = count_before / duration
            # Delta firing rate for this trial
            X[j, i] = rate_after - rate_before
        else:
            X[j, i] = 0

# Targets: (radius, theta) for each trial
y_radius = np.array(trials['radius']).astype(int)
y_theta = np.array(trials['theta']).astype(int)
y_pair = pd.Categorical(list(zip(y_radius, y_theta))).codes

# Use a single split for all targets
X_train, X_test, y_radius_train, y_radius_test, y_theta_train, y_theta_test, y_pair_train, y_pair_test = train_test_split(
    X, y_radius, y_theta, y_pair, test_size=0.2, random_state=42
)

# Now fit and evaluate as before
clf_radius = RandomForestClassifier()
clf_radius.fit(X_train, y_radius_train)
radius_pred = clf_radius.predict(X_test)
print("Radius accuracy:", accuracy_score(y_radius_test, radius_pred))

clf_theta = RandomForestClassifier()
clf_theta.fit(X_train, y_theta_train)
theta_pred = clf_theta.predict(X_test)
print("Theta accuracy:", accuracy_score(y_theta_test, theta_pred))

clf_pair = RandomForestClassifier()
clf_pair.fit(X_train, y_pair_train)
pair_pred = clf_pair.predict(X_test)
print("Radius,Theta pair accuracy:", accuracy_score(y_pair_test, pair_pred))

#%%
print("Unique rows in X:", np.unique(X, axis=0).shape[0])
print("Number of trials:", X.shape[0])
# %%
# 1. Scatter plot: True vs Predicted for radius and theta
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Radius
axs[0].scatter(y_radius_test, radius_pred, alpha=0.7)
axs[0].plot([min(y_radius_test), max(y_radius_test)], [min(y_radius_test), max(y_radius_test)], 'r--')
axs[0].set_xlabel('True Radius')
axs[0].set_ylabel('Predicted Radius')
axs[0].set_title('True vs Predicted Radius')

# Theta
axs[1].scatter(y_theta_test, theta_pred, alpha=0.7)
axs[1].plot([min(y_theta_test), max(y_theta_test)], [min(y_theta_test), max(y_theta_test)], 'r--')
axs[1].set_xlabel('True Theta')
axs[1].set_ylabel('Predicted Theta')
axs[1].set_title('True vs Predicted Theta')

plt.tight_layout()
plt.show()

# 2. Confusion matrix for (radius, theta) pair
cm = confusion_matrix(y_pair_test, pair_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted (radius, theta) class')
plt.ylabel('True (radius, theta) class')
plt.title('Confusion Matrix for (radius, theta) Pair Classification')
plt.tight_layout()
plt.show()
# %%
from sklearn.svm import SVC

# SVM for radius
svm_radius = SVC()
svm_radius.fit(X_train, y_radius_train)
radius_pred_svm = svm_radius.predict(X_test)
print("SVM Radius accuracy:", accuracy_score(y_radius_test, radius_pred_svm))

# SVM for theta
svm_theta = SVC()
svm_theta.fit(X_train, y_theta_train)
theta_pred_svm = svm_theta.predict(X_test)
print("SVM Theta accuracy:", accuracy_score(y_theta_test, theta_pred_svm))

# SVM for (radius, theta) pair
svm_pair = SVC()
svm_pair.fit(X_train, y_pair_train)
pair_pred_svm = svm_pair.predict(X_test)
print("SVM Radius,Theta pair accuracy:", accuracy_score(y_pair_test, pair_pred_svm))
# %%
# 1. Scatter plot: True vs Predicted for radius and theta (SVM)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Radius
axs[0].scatter(y_radius_test, radius_pred_svm, alpha=0.7)
axs[0].plot([min(y_radius_test), max(y_radius_test)], [min(y_radius_test), max(y_radius_test)], 'r--')
axs[0].set_xlabel('True Radius')
axs[0].set_ylabel('Predicted Radius (SVM)')
axs[0].set_title('SVM: True vs Predicted Radius')

# Theta
axs[1].scatter(y_theta_test, theta_pred_svm, alpha=0.7)
axs[1].plot([min(y_theta_test), max(y_theta_test)], [min(y_theta_test), max(y_theta_test)], 'r--')
axs[1].set_xlabel('True Theta')
axs[1].set_ylabel('Predicted Theta (SVM)')
axs[1].set_title('SVM: True vs Predicted Theta')

plt.tight_layout()
plt.show()

# 2. Confusion matrix for (radius, theta) pair (SVM)
cm_svm = confusion_matrix(y_pair_test, pair_pred_svm)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted (radius, theta) class (SVM)')
plt.ylabel('True (radius, theta) class')
plt.title('SVM: Confusion Matrix for (radius, theta) Pair Classification')
plt.tight_layout()
plt.show()
# %%
from sklearn.model_selection import GridSearchCV

# Define parameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

# Grid search for radius
svm_radius = SVC()
grid_radius = GridSearchCV(svm_radius, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_radius.fit(X_train, y_radius_train)
print("Best SVM params for radius:", grid_radius.best_params_)
radius_pred_svm = grid_radius.predict(X_test)
print("Best SVM Radius accuracy:", accuracy_score(y_radius_test, radius_pred_svm))

# Grid search for theta
svm_theta = SVC()
grid_theta = GridSearchCV(svm_theta, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_theta.fit(X_train, y_theta_train)
print("Best SVM params for theta:", grid_theta.best_params_)
theta_pred_svm = grid_theta.predict(X_test)
print("Best SVM Theta accuracy:", accuracy_score(y_theta_test, theta_pred_svm))

# Grid search for (radius, theta) pair
svm_pair = SVC()
grid_pair = GridSearchCV(svm_pair, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_pair.fit(X_train, y_pair_train)
print("Best SVM params for (radius, theta) pair:", grid_pair.best_params_)
pair_pred_svm = grid_pair.predict(X_test)
print("Best SVM Radius,Theta pair accuracy:", accuracy_score(y_pair_test, pair_pred_svm))
# %%
from sklearn.model_selection import GridSearchCV

# Define parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}

# Grid search for radius
rf_radius = RandomForestClassifier(random_state=42)
grid_rf_radius = GridSearchCV(rf_radius, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf_radius.fit(X_train, y_radius_train)
print("Best RF params for radius:", grid_rf_radius.best_params_)
radius_pred_rf = grid_rf_radius.predict(X_test)
print("Best RF Radius accuracy:", accuracy_score(y_radius_test, radius_pred_rf))

# Grid search for theta
rf_theta = RandomForestClassifier(random_state=42)
grid_rf_theta = GridSearchCV(rf_theta, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf_theta.fit(X_train, y_theta_train)
print("Best RF params for theta:", grid_rf_theta.best_params_)
theta_pred_rf = grid_rf_theta.predict(X_test)
print("Best RF Theta accuracy:", accuracy_score(y_theta_test, theta_pred_rf))

# Grid search for (radius, theta) pair
rf_pair = RandomForestClassifier(random_state=42)
grid_rf_pair = GridSearchCV(rf_pair, rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf_pair.fit(X_train, y_pair_train)
print("Best RF params for (radius, theta) pair:", grid_rf_pair.best_params_)
pair_pred_rf = grid_rf_pair.predict(X_test)
print("Best RF Radius,Theta pair accuracy:", accuracy_score(y_pair_test, pair_pred_rf))
# %%
#Accuracy for radius, theta, and (radius, theta) pair using Random Forest with Grid Search
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

n_splits = 10
test_size = 0.2

rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', None]
}

# For radius
sss_radius = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
radius_accuracies = []
for train_idx, test_idx in sss_radius.split(X, y_radius):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_radius[train_idx], y_radius[test_idx]
    grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    radius_accuracies.append(acc)
    print(f"Radius: Best params: {grid.best_params_}, Test accuracy: {acc:.3f}")

print(f"\nMean test accuracy for radius: {np.mean(radius_accuracies):.3f} ± {np.std(radius_accuracies):.3f}\n")

# For theta
sss_theta = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
theta_accuracies = []
for train_idx, test_idx in sss_theta.split(X, y_theta):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_theta[train_idx], y_theta[test_idx]
    grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    theta_accuracies.append(acc)
    print(f"Theta: Best params: {grid.best_params_}, Test accuracy: {acc:.3f}")

print(f"\nMean test accuracy for theta: {np.mean(theta_accuracies):.3f} ± {np.std(theta_accuracies):.3f}\n")

# For (radius, theta) pair
sss_pair = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
pair_accuracies = []
for train_idx, test_idx in sss_pair.split(X, y_pair):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_pair[train_idx], y_pair[test_idx]
    grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    y_pred = grid.best_estimator_.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pair_accuracies.append(acc)
    print(f"Pair: Best params: {grid.best_params_}, Test accuracy: {acc:.3f}")

print(f"\nMean test accuracy for (radius, theta) pair: {np.mean(pair_accuracies):.3f} ± {np.std(pair_accuracies):.3f}")
# %%
