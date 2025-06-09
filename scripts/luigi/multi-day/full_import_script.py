"""Load data into NWB file, and save to disk."""

# create units table:
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface.curation as scur
from neuroconv.datainterfaces import (KiloSortSortingInterface,
                                      OpenEphysRecordingInterface,
                                      SLEAPInterface)
from neuroconv.utils import dict_deep_update
from recording_processor import get_stream_name
from spikeinterface.extractors import read_openephys, read_openephys_event
from spikeinterface.full import load_sorting_analyzer, read_kilosort
from timestamps_utils import get_video_timestamps
from units_utils import load_units_df, spikes_interface_loder

example_path = "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126"
data_folder = Path(example_path)
# Read metadata from folder and subject log:
# metadata = parse_folder_metadata(data_folder)


########### Spikes ###########
spikes_interface_loder(data_folder / "kilosort4")

########### Behavior ###########

# Tracking data:
# Change the file_path so it points to the slp file in your system
file_path = "/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings/M29_WT002/20250509/113126/videos/cricket/multicam_video_2025-05-09T12_25_28_cropped_20250528161845/multicam_video_2025-05-09T12_25_28_centralpredictions.slp"  # sample_folder / "sleap" / "predictions_1.2.7_provenance_and_tracking.slp"
assert Path(file_path).exists()
interface = SLEAPInterface(file_path=file_path, verbose=False)

# Frames timing:
# After loading, run:
# interface.set_aligned_timestamps


########### Metadata ###########
# metadata = interface.get_metadata()
# metadata = dict_deep_update(metadata, parse_folder_metadata(sample_folder))
