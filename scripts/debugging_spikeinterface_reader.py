# %%
import spikeinterface.extractors as se
from pathlib import Path
import numpy as np

plotting = True

if plotting:
    # %matplotlib widget
    import matplotlib.pyplot as plt

# Path to a valid OpenEphys recording:
recording_path = Path("/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/setup-calibrations/ephys-rigs-tests/mpm-rig/20241115/nidaq_rig_synch_test/long/2024-11-13_14-39-11")

# recording_path = Path("/Users/vigji/ephy_testing_data/openephysbinary/v0.6.x_neuropixels_with_sync")

events = se.read_openephys_event(recording_path)
print("=========")
# events_array = events.get_events(channel_id='Rhythm_FPGA-100.0')
# Note: this currently works only if there is a single digital input line.
# For multiple ones there is a known bug in spikeinterface, which hopefully 
# will be fixed soon: https://github.com/NeuralEnsemble/python-neo/issues/1437
# events_array
dir(events)
events.channel_ids
# %%
events_array = events.get_events(channel_id='PXIe-6341Digital Input Line')
# events = se.read_openephys_event(recording_path)
print("=====================")

timesstamps, durations, states = [], [], []
for t, d, s in events_array:
    timesstamps.append(t)
    durations.append(d)
    states.append(s)

print(len(timesstamps), len(durations), len(states))
print(np.unique(states))
print(durations[:100])
# %%
if plotting:
    plt.figure()
    plt.plot(timesstamps, durations)
# %%
# The alternative is to read the events from the .npy files directly:
evts_path = Path('/Users/vigji/ephy_testing_data/openephysbinary/v0.6.x_neuropixels_with_sync/Record Node 104/experiment1/recording1/events/NI-DAQmx-103.PXIe-6341/TTL')

full_words = np.load(evts_path / "full_words.npy")
sample_numbers = np.load(evts_path / "sample_numbers.npy")
states = np.load(evts_path / "states.npy")
timestamps = np.load(evts_path / "timestamps.npy")
# %%
states
# %%
if plotting:
    plt.figure()
    for s in np.unique(np.abs(states)):
        plt.plot(timestamps[np.abs(states) == s], states[np.abs(states) == s], label=f'state {s}')
    plt.legend()
plt.show()
# %%
