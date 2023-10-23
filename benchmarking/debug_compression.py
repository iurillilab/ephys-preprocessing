from pathlib import Path
from datetime import datetime

from spikeinterface.core import load_extractor

from multiprocessing import freeze_support

freeze_support()


data_path = Path(r"E:\Luigi-temp\test-data-fixed")
test_data = load_extractor(data_path)

temp_path = data_path.parent / f"test_dataset_compressed_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
test_data.save(folder=temp_path, n_jobs=5)

# prep_data = correct_motion(test_data, preset="kilosort_like", n_jobs=20, chunk_duration='4s')
# from spikeinterface.preprocessing import correct_motion

# import numcodecs
#compressor = numcodecs.registry.codec_registry["lzma"](preset=1)
# prep_data.save(folder=compressed_path,
#                     # format="zarr",
#                     # compressor=compressor,
#                     channel_chunk_size=-1,
#                     chunk_duration=1.,
#                     # n_jobs=5,
#                 )