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

example_paths = [
 "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126",
"/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250508/155144"]
main_data_path = Path('/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings')
assert main_data_path.exists(), f"Main data path {main_data_path} does not exist"
data_paths = sorted(list(main_data_path.glob("M*_WT*/*/[0-9][0-9][0-9][0-9][0-9][0-9]")))
print(len(data_paths))
# data_paths = [f for f in data_paths if len(list(f.glob("[0-9][0-9][0-9][0-9][0-9][0-9].nwb"))) == 0]
# data_paths = [Path("/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys_recordings/M30_WT002/20250513/103449")]
main_dest_folder = Path("/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys_recordings")

to_redo = sorted(list(main_data_path.glob("M*_WT*/*/[0-9][0-9][0-9][0-9][0-9][0-9]")))
data_paths = [f for f in to_redo if len(list((f / "NPXData").glob("[0-9]*"))) > 1]

if main_dest_folder is not None:
    main_dest_folder.mkdir(parents=True, exist_ok=True)

# %%
for data_path in data_paths:
    #if "0508" in str(data_path) or "0509" in str(data_path):
    #    continue
    print("--------------------------------")
    print(data_path)
    try:
        data_folder = Path(data_path)

        #if (data_folder / f"{data_folder.name}.nwb").exists():
        #    print(f"Skipping {data_folder} because it already exists")
        #    continue

        ########### Spikes ###########
        spikes_interface = spikes_interface_loder(data_folder)
        # print(type(spikes_interface))

        ########### Behavior ###########
        # Tracking data:
        # Change the file_path so it points to the slp file in your system
        # interfaces_list = load_video_interfaces(data_folder)

        ########### Videos ###########
        video_interfaces, conv_options = load_video_interfaces(data_folder )

        print(conv_options)
        full_interfaces_list = [spikes_interface, *video_interfaces]
        interface = ConverterPipe(full_interfaces_list)
        ########### Metadata ###########

        interface_metadata = spikes_interface.get_metadata()

        # Read metadata from folder and subject log:
        # metadata = parse_folder_metadata(data_folder)
        # metadata = dict_deep_update(interface_metadata, metadata)
        if main_dest_folder is not None:
            print(main_dest_folder / data_folder.parent.parent.name / data_folder.parent.name / data_folder.name / f"{data_folder.name}.nwb")
            data = test_on_temp_nwb_file(interface, main_dest_folder / data_folder.parent.parent.name / data_folder.parent.name / data_folder.name / f"{data_folder.name}.nwb", 
                                        conversion_options=conv_options)
        # else:
        #     dest_folder = data_folder
        

        data = test_on_temp_nwb_file(interface, data_folder / f"{data_folder.name}.nwb", 
                                    conversion_options=conv_options)
    except Exception as e:
        print(f"Error processing {data_folder}: {e}")
        continue
        
# %%
