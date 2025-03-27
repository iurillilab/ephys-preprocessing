#%%
%matplotlib widget
import spikeinterface.extractors as se
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
#%% Path to an OpenEphys recording:
recording_path = r'X:\SNeuroBiology_shared\P05_3DRIG_YE-LP\YE_3D_rig\anaesthetised\20250309\M28\2025-03-09_15-37-57'

#%%
# Loading events
# Extract events. aka digital signals:
recording_path_nas = r'X:\SNeuroBiology_shared\P05_3DRIG_YE-LP\YE_3D_rig\anaesthetised\20250309\M28\2025-03-09_15-37-57'
events = se.read_openephys_event(recording_path)

events_array = events.get_events(channel_id='PXIe-6341Digital Input Line')
events_array
# %%
recording_path_events =Path(r'X:\SNeuroBiology_shared\P05_3DRIG_YE-LP\YE_3D_rig\anaesthetised\20250309\M28\2025-03-09_15-37-57\Record Node 107\experiment1\recording1\events\NI-DAQmx-102.PXIe-6341\TTL')
evts_path = recording_path_events
full_words = np.load(evts_path / "full_words.npy")
sample_numbers = np.load(evts_path / "sample_numbers.npy")
states = np.load(evts_path / "states.npy")
timestamps = np.load(evts_path / "timestamps.npy")
# %%
len(full_words)
# %%
np.unique(full_words)
# %%
stacked = np.stack([timestamps, full_words, states], axis=1)
clock = stacked[np.abs(stacked[:, -1]) == 1]
laser = stacked[np.abs(stacked[:, -1]) == 2]

laser[0, 0] - clock[0, 0]
# %%
plt.figure()
# plt.plot(np.diff(clock[:, 0]))
plt.plot(np.diff(laser[:, 0]))
# %%
motor_log = pd.read_csv(r'X:\SNeuroBiology_shared\P05_3DRIG_YE-LP\YE_3D_rig\anaesthetised\20250309\M28\153729\motor-log_2025-03-09T15_37_29.csv')
motor_log
# %%
motor_log[['Value.Radius', 'Value.Theta']] = motor_log[['Value.Theta', 'Value.Radius']].values
motor_log
# %%
motor_log[['Value.Direction', 'Value.Theta']] = motor_log[['Value.Theta', 'Value.Direction']].values
motor_log
# %%
