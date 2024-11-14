# %%
%matplotlib widget
import spikeinterface.extractors as se
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Path to a valid OpenEphys recording:
recording_path = Path("/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/setup-calibrations/ephys-mpm_rig-testing/20241114/nidaq_rig_synch_test/long/2024-11-13_14-39-11")

# To read the recording, we need to look in the folder the name of the record node
# and the streams that we want to read. The full stream name is "Record Node XXX#stream name"
recording_daq = se.read_openephys(recording_path, stream_name="Record Node 111#NI-DAQmx-112.PXIe-6341")
recording_npx1 = se.read_openephys(recording_path, stream_name="Record Node 111#Neuropix-PXI-110.ProbeA-AP")
recording_npx2 = se.read_openephys(recording_path, stream_name="Record Node 111#Neuropix-PXI-110.ProbeB")

# %%
dir(recording_daq)
# %%
# Sampling frequency:
recording_daq.get_sampling_frequency()

# Read channel info:
recording_daq.get_channel_ids()

# Read time duration info:
print("Number of frames:", recording_daq.get_num_frames())  # number of samples
print("Duration in seconds:", recording_daq.get_total_duration())


# %%
# We need to be careful with time info if we look at raw signals: the starting point
# of each stream do not necessarily match!
print(recording_daq.get_time_info())
print(recording_npx2.get_time_info())
# %%
# We do not want to load the full traces in memory:
# We want to read 50 seconds:
n_seconds = 5
n_samples = n_seconds * recording_daq.get_sampling_frequency()
trace = recording_daq.get_traces(start_frame=0, end_frame=n_samples, channel_ids=["AI0", "AI1"])
type(trace)  # this is not a numpy array! this is a memory view
# What does this mean?
# Data is not loaded in memory until we use it, but we can still know stuff about the array:
trace.shape

# %%
# To really load the data in memory, we can use the np.array() function.
# Until we do this we do not have to wait for loading time from the disk! 
# Useful for big data on slow disks like the NAS.
trace_numpy = np.array(trace)
type(trace_numpy)

# %%
# Extract events. aka digital signals:
events = se.read_openephys_event(recording_path)
# %%
events_array = events.get_events(channel_id='PXIe-6341Digital Input Line')
# Note: this currently works only if there is a single digital input line.
# For multiple ones there is a known bug in spikeinterface, which hopefully 
# will be fixed soon: https://github.com/NeuralEnsemble/python-neo/issues/1437
# %%
# The alternative is to read the events from the .npy files directly:
evts_path = recording_path  / "Record Node 111/experiment1/recording3/events/NI-DAQmx-112.PXIe-6341/TTL"

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
