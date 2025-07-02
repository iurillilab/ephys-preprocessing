#%%
%matplotlib widget
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import pynapple as nap
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
    mask = (aligned_spikes >= -0.5) & (aligned_spikes <= 0.5)
    plt.plot(aligned_spikes[mask], np.full(np.sum(mask), unit), '|', markersize=5, color='black')

plt.xlabel("Time from event (s)")
plt.ylabel("Unit ID")
plt.title("Spikes for all units aligned to single event")
plt.xlim(-0.5, 0.5)
plt.axvline(0.0, color='blue')
plt.show()
# %%
unit_ids = np.unique(spike_tsd.values)
positions = list(set(zip(radius, theta)))  # unique (radius, theta) pairs

# Prepare a DataFrame to store the tuning map for each neuron
tuning_maps = {}

for unit in unit_ids:
    tuning_map = {}
    unit_spikes = spike_tsd[spike_tsd.values == unit]
    for pos in positions:
        r, t = pos
        # Find trials with this (radius, theta)
        trial_mask = (radius == r) & (theta == t)
        trial_starts = np.array(start_time)[trial_mask]
        trial_ends = np.array(end_time)[trial_mask]
        firing_rates = []
        for s, e in zip(trial_starts, trial_ends):
            # Count spikes in this trial for this unit
            count = np.sum((unit_spikes.index >= s) & (unit_spikes.index < e))
            duration = e - s  # bin size in seconds
            if duration > 0:
                firing_rates.append(count / duration)
        # Average firing rate for this position
        if len(firing_rates) > 0:
            tuning_map[pos] = np.mean(firing_rates)
        else:
            tuning_map[pos] = np.nan  # or 0
    tuning_maps[unit] = tuning_map

# Convert to DataFrame: rows=unit, columns=(radius, theta), values=mean firing rate (Hz)
tuning_df = pd.DataFrame(tuning_maps).T
tuning_df.index.name = "unit_id"
tuning_df.columns = pd.MultiIndex.from_tuples(tuning_df.columns, names=["radius", "theta"])
print(tuning_df)
# %%# Plotting the tuning map for a specific neuron
# Pick a neuron to show (e.g., the first one)
unit = tuning_df.index[700]
tuning = tuning_df.loc[unit]

# Convert MultiIndex columns to a matrix for plotting
radii = sorted(set([r for r, t in tuning_df.columns]))
thetas = sorted(set([t for r, t in tuning_df.columns]))
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
from scipy.stats import ttest_1samp
import numpy as np
import pandas as pd

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
# %%
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
                color='white' if val < 0.05 else 'black',
                fontsize=8, fontweight='bold' if val < 0.05 else 'normal'
            )

plt.tight_layout()
plt.show()
# %%
