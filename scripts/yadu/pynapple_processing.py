#%%
%matplotlib widget
import numpy as np
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
unit_id = spikes["unit_name"].to_numpy()
#%%
#plotting spikes and indices to visualize the data
plt.plot(spike_tsd.index.to_numpy(), spike_tsd.to_numpy(), 'o', markersize=1, alpha=0.5)
plt.xlabel('Spike Time (s)')
plt.ylabel('Unit ID')
plt.show()
# %%
#trial parameters
start_time = trials["start"]
end_time = trials["end"]
set_movement_on = trials["set_movement_on"]
set_movement_off = trials["set_movement_off"]
home_movement_on = trials["home_movement_on"]
home_movement_off = trials["home_movement_off"]
radius = trials["radius"]
theta = trials["theta"]
#%%
spikes_in_trials = spike_tsd.restrict(trials)
spikes_in_trials
# %%
