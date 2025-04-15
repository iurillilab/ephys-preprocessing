# %%
%matplotlib widget
import pynapple as nap
import numpy as np
from matplotlib import pyplot as plt
data_path = "/Users/vigji/Downloads/test (1).nwb"
units = nap.load_file(data_path)["units"]
tsd = units.to_tsd()
f, ax = plt.subplots()
ax.scatter(tsd.t, tsd.d, s=1)
ax.set(xlabel="Time (s)", ylabel="Unit")



# %%
t_range = np.arange(0, 300, 0.05)
counts_ctx, _ = np.histogram(tsd[tsd.d > 470].t, bins=t_range, density=True)
counts_sc, _ = np.histogram(tsd[(tsd.d < 250) & (tsd.d > 150)].t, bins=t_range, density=True)
# %%
plt.figure(figsize=(10, 5))
plt.plot(t_range[:-1], counts_ctx, label="ctx")
plt.plot(t_range[:-1], counts_sc, label="sc")
plt.legend()
# %%
