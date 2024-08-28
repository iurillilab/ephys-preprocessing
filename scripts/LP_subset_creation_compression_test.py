# %%
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from spikeinterface import read_zarr

from pathlib import Path

from preprocessing_utils import *
from nwb_conv.oephys import OEPhysDataFolder
from datetime import datetime

job_kwargs = get_job_kwargs(chunk_duration="10s")
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# %%
data_to_subsample = Path(r"D:\luigi\M24")
subsampled_folder = Path(r"D:\luigi\short_test_trace_parallel")

oephys_data = OEPhysDataFolder(data_to_subsample)  # custom class to simplify parsing of info from oephys data folder

all_stream_names, ap_stream_names = oephys_data.stream_names, oephys_data.ap_stream_names

current_stream = ap_stream_names[0]

oephys_extractor = se.read_openephys(oephys_data.path, stream_name=current_stream)

# take the first 10 minutes of the recording:
n_seconds_subset = 600
start_frame = 0
end_frame = int(n_seconds_subset * oephys_extractor.get_sampling_frequency())
oephys_extractor = oephys_extractor.frame_slice(start_frame, end_frame)
lsb_corrected = spre.correct_lsb(oephys_extractor, verbose=True)  # Ensure least significant bit is 1

# %%
if __name__ == "__main__":
    start_time = datetime.now()
    lsb_corrected.save_to_zarr(folder=subsampled_folder, compressor=None, overwrite=True, **job_kwargs)
    print(f"Time taken to save uncompressed: {datetime.now() - start_time}")

# %%
