import spikeinterface.extractors as se
import numpy as np
from scipy.io import savemat
from pathlib import Path

# 
path_string = input("Recording path: ")
rec_path = Path(path_string)
STREAM_NAME ='Record Node 101#Neuropix-PXI-100.ProbeA'

n_blocks = len([f for f in rec_path.glob("*/*") if f.is_dir()])

for block_index in range(n_blocks):
    print(block_index)
    openephys_exp = se.read_openephys(rec_path,
                     stream_name=STREAM_NAME, block_index=block_index)
    print(openephys_exp)

    probe = openephys_exp.get_probe()
    print(probe)
    probe_df = probe.to_dataframe()
    n_chans = len(probe_df)

    config_dict = dict(name="NP2", 
         xcoords=probe_df["x"].values[:, np.newaxis],
         ycoords=probe_df["y"].values[:, np.newaxis],
         kcoords=np.ones(n_chans)[:, np.newaxis],
         connected=np.ones(n_chans, dtype=bool)[:, np.newaxis],
         #chanMap=np.ones(n_chans)[:, np.newaxis] + 1, # old of Luigi, was giving error
         #chanMap0ind=np.ones(n_chans)[:, np.newaxis], # old of Luigi, was giving error
         chanMap=np.array(list(range(1, n_chans + 1)))[:, np.newaxis], # filip trying this one
         chanMap0ind=np.array(list(range(0, n_chans)))[:, np.newaxis], # filip trying this one
        )
    filename = "probeConfig.mat" if block_index == 0 else f"probeConfig_{block_index + 1}.mat"
    savemat(rec_path / STREAM_NAME.split("#")[0] / filename, config_dict)