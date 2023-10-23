import spikeinterface.full as si
import spikeinterface.extractors as se
from pathlib import Path
import numcodecs
import spikeinterface.preprocessing as spre
from multiprocessing import freeze_support

freeze_support()

from datetime import datetime

# load_extractor is a function that can be used to load data already imported in a 
# spikeinterface-compatible structure:
data_path = Path(r"E:\Luigi-temp\M10\NPXData\Rec_2023-09-02_09-58-30")
test_data = se.read_openephys(data_path, stream_name="Record Node 105#Neuropix-PXI-100.ProbeA-AP")


# This compression has been optimized to get to high compression ratios while taking quite some time to run.
# This is because we expect to compress/decompress the data only once a spike sorting is final, and we do not really care 
# if this takes a few minutes. Expected time is approx. 5X real time (duration of the recording) for 30kHz sampling rate on SSD.

# To learn a bit more about other compression options, check this out: https://iopscience.iop.org/article/10.1088/1741-2552/acf5a4#jneacf5a4s5

# Ensure least significant bit is 1
comp_data_lsb_corr = test_data #spre.correct_lsb(test_data, verbose=True)


cname = "lzma"
level = 1  # higher ratios makes it much longer without much gain in CR

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
compressed_path = data_path.parent / f"test_dataset_compressed_{cname}_{level}_{timestamp}"

compressor = numcodecs.registry.codec_registry[cname](preset=level)

# Save using compression in zarr format:
comp_data_lsb_corr.save(folder=compressed_path,
                    format="zarr",
                    compressor=compressor,
                    channel_chunk_size=-1,
                    chunk_duration=10.,
                    n_jobs=5,
                    filters=[]  # prefiltering does not help with lzma
                )