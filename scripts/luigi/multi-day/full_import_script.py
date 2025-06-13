"""Load data into NWB file, and save to disk."""

# create units table:
from datetime import datetime
from pathlib import Path
from nwb_tester import test_on_temp_nwb_file
import numpy as np
import pandas as pd
import spikeinterface.curation as scur
from neuroconv.datainterfaces import SLEAPInterface
from neuroconv.utils import dict_deep_update
from spikeinterface.extractors import read_openephys, read_openephys_event
from spikeinterface.full import load_sorting_analyzer, read_kilosort
from timestamps_utils import get_video_timestamps
from units_utils import load_units_df, spikes_interface_loder
from tracking_utils import load_video_interfaces
from neuroconv import ConverterPipe
from nwb_conv import parse_folder_metadata

example_path = "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126"
example_path = "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250508/155144"
data_folder = Path(example_path)
# Read metadata from folder and subject log:
metadata = parse_folder_metadata(data_folder)

########### Spikes ###########
spikes_interface = spikes_interface_loder(data_folder)

########### Behavior ###########
# Tracking data:
# Change the file_path so it points to the slp file in your system
# interfaces_list = load_video_interfaces(data_folder)

########### Videos ###########
video_interfaces = load_video_interfaces( data_folder )



interface = ConverterPipe([spikes_interface] + video_interfaces)
########### Metadata ###########
interface_metadata = interface.get_metadata()
metadata = dict_deep_update(interface_metadata, metadata)

data = test_on_temp_nwb_file(interface, data_folder / "full_output.nwb")
print(data)