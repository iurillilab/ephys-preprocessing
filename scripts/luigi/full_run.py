# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 11:24:35 2024

@author: SNeurobiology
"""

# %%
import numpy as np

# INPUT HERE ALL FOLDERS TO BE PREPROCESSED:
# all recordings and probes (except for LFP) within the following paths will be pre-processed.
# specify the path to a general folder containing multiple experiments, or be more specific if you need to exclude something:

base_npxpaths = [r"E:\Luigi_ephys\M24"]
                # r"G:\Paola Storage do not delete\RL_E1\E1_M18",
                # r"G:\Paola Storage do not delete\RL_E1\E1_M16",
run_barcodeSync = False
run_preprocessing = True # run preprocessing and spikesorting
callKSfromSI = False  # this remains insanely slow.

dtype = np.int16

# %%
import sys
import scipy.signal
import time
import os
import shutil
import json
import subprocess
from pathlib import Path
import torch

import spikeinterface.extractors as se
import spikeinterface.full as si
import spikeinterface.preprocessing as st

sys.path.append(r"C:\Users\SNeurobiology\code\ephys-preprocessing\scripts")
sys.path
from preprocessing_utils import make_probe_plots, standard_preprocessing, run_kilosort, get_channel_names, read_continuous_data_info, find_recinfo_file

from probeinterface import ProbeGroup, write_prb
from probeinterface.plotting import plot_probe, plot_probe_group

from kilosort import io, run_kilosort

from pathlib import Path
from datetime import datetime
#%matplotlib  widget

n_cpus = os.cpu_count()
n_jobs = n_cpus - 4
print(n_jobs)
job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=True)

dtype = np.int16

# %%
base_path = base_npxpaths[0]

recinfo_file = find_recinfo_file(base_path)
continuous_data_info = read_continuous_data_info(recinfo_file)
all_stream_names, ap_stream_names = get_channel_names(continuous_data_info)

# %%


for base_path in base_npxpaths:
    recinfo_file = find_recinfo_file(base_path)
    continuous_data_info = read_continuous_data_info(recinfo_file)
    all_stream_names, ap_stream_names = get_channel_names(continuous_data_info)

    # extra step to correct for bad timestamps from failed online synchronization here:
    sync_folder = Path(base_path) / 'SynchData'  # please do not change the spelling
    
    if run_barcodeSync:
        sync_folder.mkdir(exist_ok=True)
        
        for continuous_stream_info in continuous_data_info:
            foldername = continuous_stream_info['folder_name']
            stream_name = continuous_stream_info['stream_name']
            sample_rate = continuous_stream_info['sample_rate']
            workdir = os.path.join(path, 'continuous', foldername)
            sn = np.load(os.path.join(workdir, 'sample_numbers.npy'))
            ts = sn.astype(np.float64)
            ts = ts/sample_rate
            ts = ts.reshape(-1,1) #make it consistent with what is required downstream, and with what is saved by matlab
            print(ts.shape)
            np.save(os.path.join(sync_folder, "timestamps_{}.npy".format(stream_name)), ts)
            # also retrieve all the barcode files and copy them to the same sync folder
            workdir_events = os.path.join(path, 'events', foldername, 'TTL')
            source_path = os.path.join(workdir_events, 'sample_numbers.npy')
            destination_path = os.path.join(sync_folder, "sample_numbers_{}.npy".format(stream_name))
            shutil.copy(source_path, destination_path)
        
        # run barcode synchronization:
        command = ['python', 'barcode_sync_full.py', sync_folder]
        result = subprocess.run(command, capture_output=True, text=True)
        print("Output:", result.stdout)
        if result.stderr:
            print("Error:", result.stderr)
        

    # collect spikedata streams info:
    spikedata = {
        "SpikeData": []
    }
    for continuous_stream_info in continuous_data_info:
        foldername = continuous_stream_info['folder_name']
        stream_name = continuous_stream_info['stream_name']

        ks_folder = Path(os.path.join(path, "SpikeData_{}".format(stream_name)))
        new_spikedata = {
            "stream_name": stream_name,
            "path": ks_folder.parts
        }
        spikedata["SpikeData"].append(new_spikedata)
    
    p = Path(path)
    # update and write metadata file:
    if run_barcodeSync:
        new_recording = {
            "recording_path": p.parts,
            "SynchData_path": sync_folder.parts,
            "barcodeSync_output": result.stdout,
            "barcodeSync_error": result.stderr,
            "SpikeData": spikedata["SpikeData"]
        }
    else:
        new_recording = {
            "recording_path": p.parts,
            "SynchData_path": sync_folder.parts,
            "SpikeData": spikedata["SpikeData"]
        }
    log_metadata["recording"].append(new_recording)
    with open(logfilename, 'w') as file:
        json.dump(log_metadata, file, indent=4)
    
    
    if run_preprocessing:
        # go on with preprocessing
        for stream_name in ap_stream_names:
            # stream_name = continuous_data[index]['stream_name']
            # print(index, stream_name)

            ks_folder = Path(path) / "SpikeData_{}".format(stream_name)
            ks_folder.mkdir(exist_ok=True)
            
            print("new dir: ", ks_folder)
            recording_raw = se.read_openephys(folder_path=path, stream_name=stream_name, load_sync_channel=False) # this maps all streams even if you specify only one
            print("Stream name: ", recording_raw.stream_name)
            print(recording_raw)

            fs = recording_raw.get_sampling_frequency()


            # preprocessing steps
            recording_hpsf = standard_preprocessing(recording_raw)

            # save probe information (as dataframe, plus figure)
            make_probe_plots(recording_raw, recording_hpsf, ks_folder)


# %%


# %%
