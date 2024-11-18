# %%
# %matplotlib widget
import spikeinterface.extractors as se
from spikeinterface.core import get_noise_levels
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from nwb_conv.oephys import OEPhysDataFolder
# %%
# Path to a valid OpenEphys recording:
# main_dir = Path("/Users/vigji/Desktop/noise_tests")
main_dir = Path("/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/setup-calibrations/ephys-rigs-tests")
date = "20241115"


def _get_recording(setup, probe_combination, probe_name):
    setup = main_dir / setup / date
    probe_dir = setup / f"{probe_combination}_test_noise"
    # print(probe_dir, probe_dir.exists())
    recording_path = next(probe_dir.glob("[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9]"))
    rec_folder = OEPhysDataFolder(recording_path)
    # print(rec_folder.reference_stream_name)
    probe_stream_name = [stream for stream in rec_folder.stream_names if probe_name in stream][0]
    print(probe_stream_name)
    recording = se.read_openephys(recording_path, stream_name=probe_stream_name)
    return recording

def get_recording_noise_levels(setup, probe_combination, probe_name, method="std"):
    recording = _get_recording(setup, probe_combination, probe_name)
    return get_noise_levels(recording, method=method)

# %%
# noise_levs1 = get_recording_noise_levels("mpm-rig", "NPX1", "ProbeA")
# noise_levs2 = get_recording_noise_levels("mpm-rig", "NPX1-NPX2", "ProbeA")

# room1 vs room2
# noise_levs1 = get_recording_noise_levels("mpm-rig", "NPX1", "ProbeA")
# noise_levs2 = get_recording_noise_levels("ephysroom-rig", "NPX1", "ProbeA")

# noise_levs1 = get_recording_noise_levels("ephysroom-rig", "NPX1", "ProbeA")
# noise_levs2 = get_recording_noise_levels("ephysroom-rig", "NPX1-NPX2", "ProbeA")


setup1, key1 = "mpm-rig", "NPX2"
setup2, key2 = "ephysroom-rig", "NPX2"
noise_levs1 = get_recording_noise_levels(setup1, key1, "ProbeB")
noise_levs2 = get_recording_noise_levels(setup2, key2, "ProbeA")

f, ax = plt.subplots(figsize=(3, 3))
ax.scatter(noise_levs1, noise_levs2, s=10, alpha=0.5, lw=0)
# plt.axis("equal")
xlims = (0, max(np.max(noise_levs1), np.max(noise_levs2))*1.02)
ax.plot(*[[1, xlims[1]*0.9]]*2, "k--", lw=1)
ax.set_xlim(xlims)
ax.set_ylim(xlims)
ax.set_xlabel(f"{setup1} {key1}")
ax.set_ylabel(f"{setup2} {key2}")
ax.set_yticks(ax.get_xticks())
plt.tight_layout()

setup1, key1 = "ephysroom-rig", "NPX1-NPX2"
setup2, key2 = "ephysroom-rig", "NPX1-NPX2_light"
noise_levs1 = get_recording_noise_levels(setup1, key1, "ProbeA")
noise_levs2 = get_recording_noise_levels(setup2, key2, "ProbeA")

f, ax = plt.subplots(figsize=(3, 3))
ax.scatter(noise_levs1, noise_levs2, s=10, alpha=0.5, lw=0)
# plt.axis("equal")
xlims = (0, max(np.max(noise_levs1), np.max(noise_levs2))*1.02)
ax.plot(*[[1, xlims[1]*0.9]]*2, "k--", lw=1)
ax.set_xlim(xlims)
ax.set_ylim(xlims)
ax.set_xlabel(f"{setup1} {key1}")
ax.set_ylabel(f"{setup2} {key2}")
ax.set_yticks(ax.get_xticks())
plt.tight_layout()

setup1, key1 = "ephysroom-rig", "NPX1-NPX2"
setup2, key2 = "ephysroom-rig", "NPX1-NPX2_light_cover"
noise_levs1 = get_recording_noise_levels(setup1, key1, "ProbeA")
noise_levs2 = get_recording_noise_levels(setup2, key2, "ProbeA")


f, ax = plt.subplots(figsize=(3, 3))
ax.scatter(noise_levs1, noise_levs2, s=10, alpha=0.5, lw=0)
# plt.axis("equal")
xlims = (0, max(np.max(noise_levs1), np.max(noise_levs2))*1.02)
ax.plot(*[[1, xlims[1]*0.9]]*2, "k--", lw=1)
ax.set_xlim(xlims)
ax.set_ylim(xlims)
ax.set_xlabel(f"{setup1} {key1}")
ax.set_ylabel(f"{setup2} {key2}")
ax.set_yticks(ax.get_xticks())
plt.tight_layout()
# %%
