#%%
import spikeinterface.extractors as se
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
#%% Path to an OpenEphys recording:
parent_path = Path(r'D:/')
recording_path = parent_path/'Anaesthetised'/ 'M26_D879'/ '20250307'/ '2025-03-07_16-14-43'
events_path = recording_path / 'Record Node 107' / 'experiment1' / 'recording1' / 'events'
#%%
# Loading events
# Extract events. aka digital signals:
events = se.read_openephys_event(recording_path)
events_array = events.get_events(channel_id='PXIe-6341Digital Input Line')
events_array
# %% loading events from the events_path
recording_path_events = events_path / 'NI-DAQmx-102.PXIe-6341' / 'TTL'
evts_path = recording_path_events
full_words = np.load(evts_path / "full_words.npy")
sample_numbers = np.load(evts_path / "sample_numbers.npy")
states = np.load(evts_path / "states.npy")
timestamps = np.load(evts_path / "timestamps.npy")
# %%
stacked = np.stack([timestamps, full_words, states], axis=1)
clock = stacked[np.abs(stacked[:, -1]) == 1]
laser = stacked[np.abs(stacked[:, -1]) == 2]
# %%
#loading motor log from the servo motor
motor_log = pd.read_csv(parent_path/'Anaesthetised'/ 'M26_D879'/ '20250307' /'20250307' / '161450'/ 'motor-log_2025-03-07T16_14_50.csv')
motor_log[['Value.Radius', 'Value.Theta']] = motor_log[['Value.Theta', 'Value.Radius']].values   #swap the columns because they are swapped in the file
# %%
#find the difference in timestamps to compute delay between events
motor_log['timestamp_diff'] = pd.to_datetime(motor_log['Timestamp']).diff()
# Convert timestamp differences to seconds for plotting purposes
motor_log['timestamp_diff_sec'] = motor_log['timestamp_diff'].dt.total_seconds()
motor_log = motor_log.drop(['timestamp_diff', 'Timestamp', 'Value.Theta'], axis=1)  # Drop the original timestamp_diff and Timestamp column
#%%
motor_log = motor_log.rename(columns={'Value.Radius': 'radius', 'Value.Direction': 'theta'})  # Rename columns for clarity
# %%
#finding events with specific labels and identifying them with their timestamps 
events_with_label_1 = events_array[events_array['label'] == "1"]
print("Events with label '1':")
print(events_with_label_1)

events_with_label_2 = events_array[events_array['label'] == "2"]
events_with_label_4 = events_array[events_array['label'] == "4"]

plt.figure()
plt.plot(events_with_label_1['label'], 'o')
plt.plot(events_with_label_2['label'], 'o')
plt.plot(events_with_label_4['label'], 'o')
# %%
#label 4 event is the NIDAQ event that we are interested in
NIDAQ_time = events_with_label_4['time']
NIDAQ_duration = events_with_label_4['duration']
# %%
#assing the NIDAQ timestamps and duration to the motor log
motor_log['timestamp_NIDAQ'] = NIDAQ_time
motor_log['duration_NIDAQ'] = NIDAQ_duration
motor_log
#%%
motor_log = motor_log.drop(index=0)  # Drop the first row if the Value.Radius is 5.0 (we dont want the first movement to be home movement)
motor_log
# %%
# Create new columns for set_movement_on and set_movement_off
#the set_movement_on is the timestamp of the NIDAQ event where the motor moves towards the mouse
# the set_movement_off is the timestamp of the NIDAQ event where the motor stops moving 
motor_log['set_movement_on'] = motor_log['timestamp_NIDAQ']
motor_log['set_movement_off'] = motor_log['timestamp_NIDAQ'] + motor_log['duration_NIDAQ']
motor_log
#%%
# Filter rows where Value.Radius equals 5 and add the new columns:
#home_movement_on is the timestamp of the NIDAQ event where the motor moves towards the home position
#home_movement_off is the timestamp of the NIDAQ event where the motor stops moving

# Step 1: Find indices where Value.Radius == 5.0
home_idx = motor_log.index[motor_log['radius'] == 5.0]

# Step 2: Compute home_movement_on and home_movement_off for those rows
home_on = motor_log.loc[home_idx, 'timestamp_NIDAQ']
home_off = motor_log.loc[home_idx, 'timestamp_NIDAQ'] + motor_log.loc[home_idx, 'duration_NIDAQ']

# Step 3: Assign to previous row (index - 1)
motor_log.loc[home_idx - 1, 'home_movement_on'] = home_on.values
motor_log.loc[home_idx - 1, 'home_movement_off'] = home_off.values

# Step 4: Remove all rows where Value.Radius == 5.0
motor_log = motor_log[motor_log['radius'] != 5.0].reset_index(drop=True)
motor_log
#%%
#drop the unnecessary columns
motor_log = motor_log.drop(['timestamp_diff_sec','timestamp_NIDAQ', 'duration_NIDAQ'], axis=1)
#%%
#defining start and stop times for the motor log, this is needed for converting in to nwb file
#the staert_time is the timestamp of the NIDAQ event where the motor starts moving towards the mouse (set_movement_on)
#the stop_time is the timestamp of the NIDAQ event where the motor stops moving (home_movement_off)
motor_log['start_time'] = motor_log['set_movement_on']
motor_log['stop_time'] = motor_log['home_movement_off']
motor_log
#%%
trial_log = motor_log
trial_log.to_csv(parent_path/'Anaesthetised'/ 'M26_D879' / '20250307' / 'trial_log.csv', index=False)