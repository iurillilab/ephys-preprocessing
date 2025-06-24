#%%
# %matplotlib inline 
from datetime import datetime
import matplotlib.pyplot as plt
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import neuroconv as nwb_conv

from neuroconv.datainterfaces import KiloSortSortingInterface
print('hello world')
#%%
from nwb_conv import check_lablogs_location
check_lablogs_location()
#%%

parent_path = Path(r'D:\Anaesthetised')
probe_sorted_units_path = parent_path / 'M26_D879' / '20250307' / 'kilosort4' / 'sorter_output'
# Change the folder_path to the location of the data in your system
interface = KiloSortSortingInterface(folder_path=probe_sorted_units_path, verbose=False)

metadata = interface.get_metadata()
session_start_time = datetime(2025, 3, 7, 16, 45, 0, tzinfo=ZoneInfo("US/Pacific"))
metadata["NWBFile"].update(session_start_time=session_start_time)
nwbfile_path = parent_path / 'M26_D879' / '20250307' / 'recording.nwb'
interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)
# %%
import pynapple as nap
# %%
data = nap.load_file(nwbfile_path)
data.key_to_id
# %%
spikes = data["units"]
spikes
# %%
dir(spikes)
#%%
a = spikes.to_tsd()
spike_times = np.array(a)
unit_id = spikes["unit_name"].to_numpy()
spike_times
#%%
plt.plot(a.index.to_numpy(), a .to_numpy(), 'o')
plt.xlabel('Spike Time (s)')
plt.ylabel('Unit ID')
plt.show()

# %%
# %%
