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
# event_times = events.get_event_times(channel_id='PXIe-6341Digital Input Line')
# event_times
events_array
# %% Sorting events_array by the label field
# To sort the structured numpy array by the 'label' field:
sorted_events_array = np.sort(events_array, order='label')
print("Sorted events_array by label:")
print(sorted_events_array)

# Alternatively, if you want to only select events with label "2":
events_with_label_3 = events_array[events_array['label'] == "2"]
print("Events with label '3':")
print(events_with_label_3)
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
# %% Calculate difference between rows in the 'Value.Radius' column of motor_log
motor_log['timestamp_diff'] = pd.to_datetime(motor_log['Timestamp']).diff()
print("Differences in 'Timestamp':")
print(motor_log['timestamp_diff'])
#%%
plt.figure
plt.plot(motor_log['timestamp_diff'])
# %%
# %% Histogram of timestamp differences for motor_log
# Convert timestamp differences to seconds for plotting purposes
motor_log['timestamp_diff_sec'] = motor_log['timestamp_diff'].dt.total_seconds()
plt.figure()
plt.hist(motor_log['timestamp_diff_sec'].dropna(), bins=20)
plt.xlabel("Time Difference (seconds)")
plt.ylabel("Frequency")
plt.title("Histogram of Timestamp Differences")
plt.show()
# %%
events_with_label_1 = events_array[events_array['label'] == "1"]
print("Events with label '1':")
print(events_with_label_1)

events_with_label_2 = events_array[events_array['label'] == "2"]
events_with_label_4 = events_array[events_array['label'] == "4"]

plt.figure()
plt.plot(events_with_label_1['label'], 'o')
plt.plot(events_with_label_2['label'], 'o')
plt.plot(events_with_label_4['label'], 'o')

#%%
len(events_with_label_4)# %%

# %%

