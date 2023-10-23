# %%
from pathlib import Path
import spikeinterface.extractors as se 
from spikeinterface.core import load_extractor
import spikeinterface.full as si
fragment_dest_dir = r"N:\SNeuroBiology_shared\LP_E002_MPA_OPTO\ephys\test_dataset"
fragment = load_extractor(fragment_dest_dir)
# %%
data_path = Path(r"N:\SNeuroBiology_shared\LP_E002_MPA_OPTO\ephys\M11\NPXData")
recording_raw = se.read_openephys(data_path, stream_name="Record Node 105#Neuropix-PXI-100.ProbeA-AP")

# %%
fs = recording_raw.get_sampling_frequency()
recording_sub = recording_raw.frame_slice(start_frame=500*fs, end_frame=800*fs)
# %%
si.plot_traces(recording_sub, backend="matplotlib", # "ipywidgets", 
               mode='map', time_range=(0, 0.04), 
               channel_ids=recording_sub.channel_ids)
# %%
fragment_dest_dir =  r"N:\SNeuroBiology_shared\LP_E002_MPA_OPTO\ephys\test_dataset-corrected"
recording_sub.save(folder=fragment_dest_dir,
                   #n_jobs=20, 
                  # chunk_duration='4s', 
                   progres_bar=True)
# %%
from spikeinterface.core import load_extractor
fragment_dest_dir = r"N:\SNeuroBiology_shared\LP_E002_MPA_OPTO\ephys\test_dataset-corrected"
fragment = load_extractor(fragment_dest_dir)

# %%
si.plot_traces(fragment, backend="matplotlib", # "ipywidgets", 
               mode='map', time_range=(0, 0.04), 
               channel_ids=fragment.channel_ids)

# %%


# %%

