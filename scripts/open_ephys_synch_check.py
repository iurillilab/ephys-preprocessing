# %%
%matplotlib widget
import spikeinterface.extractors as se
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

recording_path = Path("/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/setup-calibrations/ephys-mpm_rig-testing/20241114/nidaq_rig_synch_test/long/2024-11-13_14-39-11")
recording_daq = se.read_openephys(recording_path, stream_name="Record Node 111#NI-DAQmx-112.PXIe-6341")
recording_npx2 = se.read_openephys(recording_path, stream_name="Record Node 111#Neuropix-PXI-110.ProbeB")

# %%

# Read time arrays:
time_npx2 = recording_npx2.get_times()
time_daq = recording_daq.get_times()

# Read traces:
start_time = time.time()
trace_npx2 = np.array(recording_npx2.get_traces(channel_ids=recording_npx2.channel_ids[:1]))
trace_daq = np.array(recording_daq.get_traces(channel_ids=recording_daq.channel_ids[1:]))
print(f"Time taken to load traces: {time.time() - start_time:.2f} seconds")

# %%
plt.plot(time_npx2, trace_npx2, label="npx2")
plt.plot(time_daq, trace_daq, label="daq")
plt.legend()
# %%
