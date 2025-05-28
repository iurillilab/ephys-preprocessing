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
    interface = KiloSortSortingInterface(folder_path=folder_path, verbose=False)

    metadata = interface.get_metadata()
    session_start_time = datetime(2020, 1, 1, 12, 30, 0, tzinfo=ZoneInfo("US/Pacific"))
    metadata["NWBFile"].update(session_start_time=session_start_time)
    nwbfile_path = f"{path_to_save_nwbfile}"  # This should be something like: "./saved_file.nwb"
    interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)

# %%
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

units_df = load_units_df(sample_folder / "kilosort4")

interface = KiloSortSortingInterface(folder_path=sample_folder / "kilosort4" / "sorter_output", verbose=False)

# %%
