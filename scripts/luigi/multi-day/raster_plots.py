# %%
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pynapple as nap
from matplotlib import pyplot as plt
from neuroconv.datainterfaces import (KiloSortSortingInterface,
                                      OpenEphysRecordingInterface)
from plotly.subplots import make_subplots
from rastermap import Rastermap
from recording_processor import RecordingProcessor, get_stream_name
from spikeinterface.extractors import read_openephys, read_openephys_event
from spikeinterface.full import load_sorting_analyzer, read_kilosort
from tqdm import tqdm

# %%


# create units table:
def _get_units_df(probe_sorted_units):
    """Read dataframe of units info with position on the probe."""
    templates = np.load(probe_sorted_units / "templates.npy")
    units_df = pd.read_csv(probe_sorted_units / "cluster_KSLabel.tsv", sep="\t")
    probe_channel_positions = np.load(probe_sorted_units / "channel_positions.npy")

    peak_chan_idxs = templates.mean(1).argmax(1)
    unit_channel_position = probe_channel_positions[peak_chan_idxs]
    units_df["probe_x"] = unit_channel_position[:, 0]
    units_df["probe_depth"] = unit_channel_position[:, 1]
    units_df["probe_channel"] = peak_chan_idxs

    return units_df


def load_units_df(folder: Path) -> pd.DataFrame:
    date = datetime.strptime(folder.parent.name.split("_")[0], "%Y-%m-%d")
    mouse_id = (
        folder.parent.parent.parent.name
        if "split" in folder.parent.parent.name
        else folder.parent.parent.name
    )

    analyzer = load_sorting_analyzer(folder / "analyser", format="binary_folder")
    units_df = _get_units_df(folder / "sorter_output")
    metrics = analyzer.get_extension(extension_name="quality_metrics").get_data()

    units_df["recording_date"] = date
    units_df["mouse_id"] = mouse_id
    units_df = pd.concat([units_df, metrics], axis=1)
    return units_df


# %%

# %%

good_units_mask = nwb_file["units"]["KSLabel"] == "good"

good_units = nwb_file["units"][good_units_mask]
good_units_tsd = good_units.to_tsd()
print(
    "First spike time: ", good_units_tsd.t[0], "Last spike time: ", good_units_tsd.t[-1]
)
# %%
sample_folder = Path(
    "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126"
)  # Path("/Users/vigji/Desktop/short_recording_oneshank/2025-01-22_16-56-15")
folder_path = sample_folder / "kilosort4"
recording_folder = next((folder_path.parent / "NPXData").glob("2025-*"))
stream_name = get_stream_name(recording_folder)
print(stream_name)
recording_extractor = read_openephys(recording_folder, stream_name=stream_name)
# load_sync_timestamps=True)
times = recording_extractor.get_times(segment_index=0)


spike_times = np.load(
    "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126/kilosort4/sorter_output/spike_times.npy"
)
timestamps = np.load(
    "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126/NPXData/2025-05-09_12-04-15/Record Node 107/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeA/timestamps.npy"
)
print(spike_times.min(), spike_times.max())
print(len(times), times.min(), times.max())
print(len(timestamps), timestamps.min(), timestamps.max())
# %%
sample_folder = "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126/NPXData/2025-05-09_12-04-15"
stream_name = get_stream_name(sample_folder)
rec = read_openephys(sample_folder, stream_name=stream_name)
rec
# %%
# Filter good units
good_units_mask = nwb_file["units"]["KSLabel"] == "good"

good_units = nwb_file["units"][good_units_mask]
print(f"Total units: {len(nwb_file['units'])}")
print(f"Good units: {len(good_units)}")

# Convert TsGroup to Tsd
good_units_tsd = good_units.to_tsd()
print(f"Converted to Tsd with shape: {good_units_tsd.shape}")
print(
    "First spike time: ", good_units_tsd.t[0], "Last spike time: ", good_units_tsd.t[-1]
)
# Calculate firing rates using 1s bins
firing_rates = good_units
# print(f"Firing rates shape: {firing_rates.shape}")
# %%
dir(good_units)

# %%
# Calculate spike counts for each unit
bin_size = 0.5
firing_rates = good_units.count(bin_size=bin_size).d / bin_size
zsc_firing_rates = (firing_rates - firing_rates.mean(axis=0)) / firing_rates.std(axis=0)
# print(f"Spike counts per unit:\n{spike_counts}")
# %%

px.imshow(
    zsc_firing_rates.T, aspect="auto", color_continuous_scale="Gray_r", zmin=0, zmax=1
)
# %%
# spks is neurons by time

# fit rastermap
model = Rastermap(n_PCs=200, n_clusters=20, locality=0, time_lag_window=5).fit(
    zsc_firing_rates.T
)
# y = model.embedding # neurons x 1
isort = model.isort

# Create time axis based on bin size
t = np.arange(0, zsc_firing_rates.shape[0]) * bin_size

px.imshow(
    zsc_firing_rates[:, isort].T,
    aspect="auto",
    color_continuous_scale="Gray_r",
    zmin=0,
    zmax=1,
    x=t,
    labels={"x": "Time (s)", "y": "Neuron ID"},
)
# %%
zsc_firing_rates.shape
# %%
np.load(
    "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126/NPXData/2025-05-09_12-04-15/Record Node 107/experiment1/recording1/continuous/Neuropix-PXI-100.ProbeA/timestamps.npy"
)
# %%
np.load(
    "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126/kilosort4/sorter_output/spike_positions.npy"
)
# %%
np.load(
    "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126/kilosort4/sorter_output/templates_ind.npy"
)
np.load(
    "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126/kilosort4/sorter_output/spike_times.npy"
)
