# %%
%matplotlib widget
import spikeinterface.extractors as se
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import time

recording_path = Path("/Volumes/Extreme SSD/2024-11-13_14-39-11")
recording_daq = se.read_openephys(recording_path, 
                                  load_sync_timestamps=True,
                                  stream_name="Record Node 111#NI-DAQmx-112.PXIe-6341")
recording_npx2 = se.read_openephys(recording_path, 
                                   load_sync_timestamps=True,
                                   stream_name="Record Node 111#Neuropix-PXI-110.ProbeB")

# %%

timing_start = time.time()
# Read time arrays:
time_npx2 = recording_npx2.get_times()
time_daq = recording_daq.get_times()
print(f"Time taken to load times: {time.time() - timing_start:.2f} seconds")


# Read traces:
timing_start = time.time()
start_time = 0
end_time = start_time + 1500
trace_npx2 = np.array(recording_npx2.get_traces(# start_frame=start_time*recording_npx2.get_sampling_frequency(),
                                                # end_frame=end_time*recording_npx2.get_sampling_frequency(),
                                                channel_ids=recording_npx2.channel_ids[:1]))
trace_daq = np.array(recording_daq.get_traces(# start_frame=start_time*recording_daq.get_sampling_frequency(),
                                                # end_frame=end_time*recording_daq.get_sampling_frequency(),
                                                channel_ids=recording_daq.channel_ids[1:]))

print(f"Time taken to load traces: {time.time() - timing_start:.2f} seconds")

# %%
plt.figure(figsize=(10, 5))
plt.plot(time_npx2[::30], # [int(start_time*recording_npx2.get_sampling_frequency()):int(end_time*recording_npx2.get_sampling_frequency())], 
         trace_npx2[::30], label="npx2")
plt.plot(time_daq[::6], # [int(start_time*recording_daq.get_sampling_frequency()):int(end_time*recording_daq.get_sampling_frequency())], 
         trace_daq[::6], label="daq")
plt.legend()
# %%
plt.close()
# %%
npx_timestamps = np.load("/Volumes/Extreme SSD/2024-11-13_14-39-11/Record Node 111/experiment1/recording3/continuous/Neuropix-PXI-110.ProbeB/timestamps.npy")
daq_timestamps = np.load("/Volumes/Extreme SSD/2024-11-13_14-39-11/Record Node 111/experiment1/recording3/continuous/NI-DAQmx-112.PXIe-6341/timestamps.npy")

# %%
plt.figure(figsize=(10, 5))
plt.plot(npx_timestamps[::30], # time_npx2[::30], # [int(start_time*recording_npx2.get_sampling_frequency()):int(end_time*recording_npx2.get_sampling_frequency())], 
         trace_npx2[::30], label="npx2")
plt.plot(daq_timestamps[::6], # time_daq[::6], # [int(start_time*recording_daq.get_sampling_frequency()):int(end_time*recording_daq.get_sampling_frequency())], 
         trace_daq[::6], label="daq")
plt.legend()
# %%
