#%%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pynapple as nap
# %%
parent_path = Path(r'D:/Anaesthetised')
nwbfile_path = parent_path / 'M26_D879' / '20250307' / 'M26_D879.nwb'
#%%
data = nap.load_file(nwbfile_path)
data.key_to_id
# %%
spikes = data["units"]
trials = data["trials"]
video = data["TopCamera__video_2025-03-07T16_14_50.avi"]
video
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
