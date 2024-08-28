# %%
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
from spikeinterface import read_zarr
import numcodecs

from pathlib import Path
import numpy as np

from nwb_conv.oephys import OEPhysDataFolder

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# %%
data_to_subsample = Path(r"D:\luigi\M24")
subsampled_folder = data_to_subsample / "short_test_trace"

oephys_data = OEPhysDataFolder(data_to_subsample)  # custom class to simplify parsing of info from oephys data folder

all_stream_names, ap_stream_names = oephys_data.stream_names, oephys_data.ap_stream_names

current_stream = ap_stream_names[0]

oephys_extractor = se.read_openephys(oephys_data.path, stream_name=current_stream)

# take the first 10 minutes of the recording:
n_seconds_subset = 100
start_frame = 0
end_frame = int(n_seconds_subset * oephys_extractor.get_sampling_frequency())

# %%
start_time = datetime.now()
oephys_extractor = oephys_extractor.frame_slice(start_frame, end_frame)
lsb_corrected = spre.correct_lsb(oephys_extractor, verbose=True)  # Ensure least significant bit is 1

lsb_corrected.save_to_zarr(folder=subsampled_folder.parent / (subsampled_folder.name + f"_uncompressed"))
print(f"Time taken to save uncompressed: {datetime.now() - start_time}")

# save with compression and time it:
for clevel in [3, 5]:
    compressor = numcodecs.Blosc(cname="zstd", clevel=clevel, shuffle=numcodecs.Blosc.BITSHUFFLE)
    start_time = datetime.now()
    lsb_corrected.save_to_zarr(folder=subsampled_folder.parent / (subsampled_folder.name + f"_compressed_{clevel}"), 
                            compressor=compressor)
    print(f"Time taken to save compressed at level {clevel}: {datetime.now() - start_time}")
# start_time = datetime.now()
# oephys_extractor.save_to_zarr(folder=subsampled_folder.parent / (subsampled_folder.name + "_compressed"), 
#                               compressor=compressor)
print(f"Time taken to save compressed: {datetime.now() - start_time}")

# %%
# test decompression time:
saved_files = data_to_subsample.parent.glob("short_test_trace*compressed*") 


# take the first 10 minutes of the recording:
n_seconds_subset = 100
start_frame = 0
end_frame = int(n_seconds_subset * oephys_extractor.get_sampling_frequency())

for file in saved_files:
    file = Path(file)
    loaded_data = read_zarr(file)

    start_time = datetime.now()
    # resave to different uncompressed folder to test speed:
    arr = np.array(loaded_data.get_traces(start_frame=start_frame, end_frame=end_frame))
    print(f"Time taken to load 100 s in {file.name}: {datetime.now() - start_time}")
# %%
