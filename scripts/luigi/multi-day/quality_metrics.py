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

# from neuroconv.datainterfaces import KiloSortSortingInterface
# interface = KiloSortSortingInterface(folder_path=sample_folder / "kilosort4" / "sorter_output", verbose=False)

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

# %%
# NAS:
#####
nas_data_path = Path("/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys _recordings/M30")

local_csv_filename = Path("nas_units_df.csv")
if local_csv_filename.exists():
    all_dfs = pd.read_csv(local_csv_filename)
    all_dfs["recording_date"] = pd.to_datetime(all_dfs["recording_date"])
else:
    ks_paths = list(nas_data_path.glob("*/kilosort4")) + list(nas_data_path.glob("*/*/kilosort4"))
    print(len(ks_paths))
    all_dfs = []
    for ks_path in tqdm(ks_paths):
        units_df = _get_units_df(ks_path / "sorter_output")
        date = datetime.strptime(ks_path.parent.name.split("_")[0], "%Y-%m-%d")
        mouse_id = ks_path.parent.parent.parent.name if "split" in ks_path.parent.parent.name else ks_path.parent.parent.name

        units_df["ks_path"] = ks_path
        units_df["recording_date"] = date
        units_df["mouse_id"] = mouse_id
        analyzer = load_sorting_analyzer(ks_path / "analyser", format="binary_folder")
        metrics = analyzer.get_extension(extension_name="quality_metrics").get_data()
        good_filter = units_df["KSLabel"] == "good"
        assert len(units_df) == len(metrics)
        units_df = pd.concat([units_df, metrics], axis=1)

        all_dfs.append(units_df)
    all_dfs = pd.concat(all_dfs, axis=0)
    all_dfs["recording_date"] = pd.to_datetime(all_dfs["recording_date"])
    all_dfs.to_csv(local_csv_filename)
all_dfs

# %%
# %%
# Allen criteria:
# isi_violations_ratio < 0.5 (under a specific definition)
# amplitude_cutoff < 0.1 (from distribution of amplitudes)
# presence_ratio > 0.9
# They actually discourage noise cutoff. They keep 40% of units
# IBL:
# Sliding Refractory Period (max_confidence > 0.9, min_contamination < 10%)
# Noise Cutoff (cutoff value < 5, lowest bin < 10% of peak)
# Amplitude (amp_median > 50 µV)
# They keep 14% of units
# missing noise cutoff!

all_dfs["KS_filter"] = all_dfs["KSLabel"] == "good"
all_dfs["allen_filter"] = (all_dfs.isi_violations_ratio < 0.5) & (all_dfs.amplitude_cutoff < 0.1) & (all_dfs.presence_ratio > 0.9)
all_dfs["ibl_filter"] = (all_dfs.sliding_rp_violation < 0.1) & (np.abs(all_dfs.amplitude_median) > 50)


print(f"Good units: {all_dfs['KS_filter'].sum()}")
print(f"Allen criteria: {all_dfs['allen_filter'].sum()}")
print(f"IBL criteria: {all_dfs['ibl_filter'].sum()}")
print(f"Allen & IBL criteria: {(all_dfs['allen_filter'] & all_dfs['ibl_filter']).sum()}")
print(f"KS and IBL criteria: {(all_dfs['ibl_filter'] & all_dfs['KS_filter']).sum()}")
print(f"KS and Allen criteria: {(all_dfs['allen_filter'] & all_dfs['KS_filter']).sum()}")
print(f"All three criteria: {(all_dfs['allen_filter'] & all_dfs['ibl_filter'] & all_dfs['KS_filter']).sum()}")

good_filter = all_dfs['KS_filter'] & all_dfs['allen_filter'] & all_dfs['ibl_filter']
# %%
# Plotting
good_filter
# %%
# Try with explicit bin edges
all_dfs#.loc[good_filter, :]
# %%
metrics_to_plot = ["snr", "amplitude_median", "isi_violations_ratio", "amplitude_cutoff", "presence_ratio", "sliding_rp_violation"]

# Create multi-panel figure with plotly
n_metrics = len(metrics_to_plot)
n_cols = 3
n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division

fig = make_subplots(
    rows=n_rows, 
    cols=n_cols,
    subplot_titles=[metric.replace('_', ' ').title() for metric in metrics_to_plot],
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

for idx, metric in enumerate(metrics_to_plot):
    row = idx // n_cols + 1
    col = idx % n_cols + 1
    
    # Add histogram for good units
    fig.add_trace(
        go.Histogram(
            x=all_dfs.loc[good_filter, metric],
            name='Good units',
            opacity=0.7,
            nbinsx=30,
            marker_color='red',
            showlegend=idx==0
        ),
        row=row, col=col
    )

# Update layout
fig.update_layout(
    height=300 * n_rows,
    width=1000,
    #title_text="Unit Quality Metrics",
    showlegend=True,
    template="plotly_white"
)

# Update axes labels
for i in range(1, n_metrics + 1):
    row = (i-1) // n_cols + 1
    col = (i-1) % n_cols + 1
    #fig.update_xaxes(title_text=metrics_to_plot[i-1].replace('_', ' ').title(), row=row, col=col)
    fig.update_yaxes(title_text="Count", row=row, col=col)

fig.show()

# %%
# Plotting overlap between criteria
from matplotlib_venn import venn3
import matplotlib.pyplot as plt

# Create the Venn diagram
plt.figure(figsize=(10, 10))
v = venn3(
    subsets=(
        set(all_dfs[all_dfs['KS_filter']].index),
        set(all_dfs[all_dfs['allen_filter']].index),
        set(all_dfs[all_dfs['ibl_filter']].index),
    ),
    set_labels=('Kilosort', 'Allen', 'IBL'),
    set_colors=('blue', 'green', 'red'),
    alpha=0.5
)

# Color the labels
for label, color in zip(v.set_labels, ('blue', 'green', 'red')):
    if label is not None:
        label.set_color(color)
        label.set_fontweight('bold')

# Add text for total number of units
total_units = len(all_dfs)
plt.text(0.5, -0.1, f'Total units: {total_units}', 
         horizontalalignment='center', 
         verticalalignment='center',
         transform=plt.gca().transAxes,
         fontsize=12,
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

# Add a title that emphasizes these are subsets
plt.title('Unit Quality Criteria Overlap\n(All sets are subsets of total units)', pad=20, size=14)
plt.show()


# %%
mouse_id = "M29"

mouse_df = all_dfs[all_dfs["mouse_id"] == mouse_id]
mouse_df.groupby("recording_date").agg({"KS_filter": "sum", "allen_filter": "sum", "ibl_filter": "sum"})
daily_counts = mouse_df.groupby("recording_date")[["KS_filter", "allen_filter", "ibl_filter"]].sum().reset_index()

# Sort by date
daily_counts = daily_counts.sort_values("recording_date")

# Create sequential day numbers
daily_counts["day_number"] = range(len(daily_counts))

# Melt the dataframe for plotting
daily_counts_melted = daily_counts.melt(
    id_vars=["recording_date", "day_number"],
    value_vars=["KS_filter", "allen_filter", "ibl_filter"],
    var_name="Filter",
    value_name="Number of Units"
)

# Create the line plot
fig = px.line(
    daily_counts_melted,
    x="day_number",
    y="Number of Units",
    color="Filter",
    markers=True,
    title="Number of Units per Day by Quality Filter - M29"
)

# Update layout
fig.update_layout(
    xaxis_title="Recording Day",
    yaxis_title="Number of Units",
    template="plotly_white",
    hovermode="x unified",
    width=800,
    height=500,
    yaxis=dict(range=[0, 600])
)

# Update x-axis to show actual dates
fig.update_xaxes(
    ticktext=daily_counts["recording_date"].dt.strftime("%Y-%m-%d"),
    tickvals=daily_counts["day_number"]
)

# Update colors and line styles
fig.update_traces(
    line=dict(width=2),
    marker=dict(size=8)
)

fig.show()

# %%
if __name__ == "__main__":
    ...
# %%
# %%