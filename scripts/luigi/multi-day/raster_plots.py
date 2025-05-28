# %%
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from spikeinterface.full import read_kilosort, load_sorting_analyzer
from recording_processor import RecordingProcessor
from pathlib import Path

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import pynapple as nap
from rastermap import Rastermap
 

from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from neuroconv.datainterfaces import KiloSortSortingInterface
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

def dirty_nwb_like_loder(folder_path: Path) -> pd.DataFrame:
    # folder_path = f"{ECEPHY_DATA_PATH}/phy/phy_example_0"
    # Change the folder_path to the location of the data in your system
    nwb_path = folder_path / "sorter_output.nwb"

    if nwb_path.exists():
        return nap.load_file(nwb_path)
    
    interface = KiloSortSortingInterface(folder_path=folder_path / "sorter_output", verbose=False)

    metadata = interface.get_metadata()
    session_start_time = datetime(2020, 1, 1, 12, 30, 0, tzinfo=ZoneInfo("US/Pacific"))
    metadata["NWBFile"].update(session_start_time=session_start_time)
    interface.run_conversion(nwbfile_path=nwb_path, metadata=metadata)
    return nap.load_file(nwb_path)

sample_folder = Path("/Users/vigji/Desktop/M29/2025-05-09_12-04-15")  # Path("/Users/vigji/Desktop/short_recording_oneshank/2025-01-22_16-56-15")
# sorter = read_kilosort(sample_folder / "kilosort4" / "sorter_output", keep_good_only=False)

def load_units_df(folder: Path) -> pd.DataFrame:
    date = datetime.strptime(folder.parent.name.split("_")[0], "%Y-%m-%d")
    mouse_id = folder.parent.parent.parent.name if "split" in folder.parent.parent.name else folder.parent.parent.name

    analyzer = load_sorting_analyzer(folder / "analyser", format="binary_folder")
    units_df = _get_units_df(folder / "sorter_output")
    metrics = analyzer.get_extension(extension_name="quality_metrics").get_data()

    units_df["recording_date"] = date
    units_df["mouse_id"] = mouse_id
    units_df = pd.concat([units_df, metrics], axis=1)
    return units_df

# units_df = load_units_df(sample_folder / "kilosort4")
nwb_file = dirty_nwb_like_loder(sample_folder / "kilosort4")


# %%
# Filter good units
good_units_mask = (nwb_file["units"]["KSLabel"] == "good") 

good_units = nwb_file["units"][good_units_mask]
print(f"Total units: {len(nwb_file['units'])}")
print(f"Good units: {len(good_units)}")

# Convert TsGroup to Tsd
good_units_tsd = good_units.to_tsd()
print(f"Converted to Tsd with shape: {good_units_tsd.shape}")

# Calculate firing rates using 1s bins
firing_rates = good_units
# print(f"Firing rates shape: {firing_rates.shape}")
# %%
dir(good_units)
# %%
# Filter first 500 seconds and create scatter plot
mask = good_units_tsd.t < 500
fig = go.Figure()

# Add scatter plot with vertical line markers using scattergl for better performance
fig.add_trace(go.Scattergl(
    x=good_units_tsd.t[mask],
    y=good_units_tsd.d[mask],
    mode='markers',
    marker=dict(
        symbol='line-ns',  # vertical line marker
        size=8,  # slightly smaller for better performance
        color='black',
        line=dict(width=1)
    ),
    name='Spikes'
))

# Update layout
fig.update_layout(
    title='Spike Raster Plot (First 500s)',
    xaxis_title='Time (s)',
    yaxis_title='Unit ID',
    template='plotly_white',
    showlegend=False
)

# Update axes
fig.update_xaxes(range=[0, 500])
fig.update_yaxes(range=[good_units_tsd.d.min()-0.5, good_units_tsd.d.max()+0.5])

fig.show()

# %%
# Calculate spike counts for each unit
bin_size = .5
firing_rates = good_units.count(bin_size=bin_size).d / bin_size
zsc_firing_rates = (firing_rates - firing_rates.mean(axis=0)) / firing_rates.std(axis=0)
#print(f"Spike counts per unit:\n{spike_counts}")
# %%

px.imshow(zsc_firing_rates.T, aspect="auto",
          color_continuous_scale="Gray_r", zmin=0, zmax=1)
# %%
# spks is neurons by time

# fit rastermap
model = Rastermap(n_PCs=200, n_clusters=20, 
                  locality=0, time_lag_window=5).fit(zsc_firing_rates.T)
# y = model.embedding # neurons x 1
isort = model.isort

# Create time axis based on bin size
t = np.arange(0, zsc_firing_rates.shape[0]) * bin_size

px.imshow(zsc_firing_rates[:, isort].T, aspect="auto",
          color_continuous_scale="Gray_r", zmin=0, zmax=1,
          x=t, labels={'x': 'Time (s)', 'y': 'Neuron ID'})
# %%
zsc_firing_rates.shape
# %%
