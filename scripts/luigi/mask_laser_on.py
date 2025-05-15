# %%

data_path_list = [r"E:\P02_MPAOPTO_LP\e04_ephys-contrapag-stim\v01\M21",
                  r"F:\Luigi\M19_D558\20240419\133356\NpxData"]
run_barcodeSync = False
run_preprocessing = True # run preprocessing and spikesorting
callKSfromSI = False

# %%
# %matplotlib widget
from matplotlib import pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as st

from pathlib import Path
import numpy as np

from preprocessing_utils import *
from nwb_conv.oephys import OEPhysDataFolder
from datetime import datetime

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")


data_path = Path(data_path_list[0])

oephys_data = OEPhysDataFolder(data_path)  # custom class to simplify parsing of info from oephys data folder

all_stream_names, ap_stream_names = oephys_data.stream_names, oephys_data.ap_stream_names

for current_stream in ap_stream_names:
    # standard preprocessing pipeline:
    oephys_extractor = se.read_openephys(oephys_data.path, stream_name=current_stream)

    # read laser onsets in NPX indexing from oephys data (takes care of barcode alignment internally):
    laser_onset_idxs = find_laser_onsets_npx_idxs(oephys_data, current_stream)
    zeroed_extractor = st.RemoveArtifactsRecording(oephys_extractor, laser_onset_idxs, ms_after=11)

    preprocessed_extractor = standard_preprocessing(zeroed_extractor)

    # Debugging plots:
    figures_folder = data_path / f"ephys_processing_figs_{timestamp}" / current_stream
    figures_folder.mkdir(exist_ok=True, parents=True)

    make_probe_plots(oephys_extractor, preprocessed_extractor, figures_folder, current_stream)
    plot_raw_and_preprocessed(oephys_extractor, preprocessed_extractor, saving_path=figures_folder / f"snippets_{current_stream}.png")

    show_laser_trigger_preprost(oephys_extractor, preprocessed_extractor, laser_onset_idxs, 
                                    n_to_take=200, saving_path=figures_folder / f"laser_trigger_preprost_{current_stream}.png")
    
    plt.close('all')
