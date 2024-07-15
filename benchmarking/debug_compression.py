from pathlib import Path
from datetime import datetime
from spikeinterface.core import load_extractor
import spikeinterface.extractors as se

#data_path = Path(r"E:\Luigi-temp\test-data-fixed")
#test_data = load_extractor(data_path)

data_path = Path(r"E:\Luigi-temp\M10\NPXData\Rec_2023-09-02_09-58-30")
test_data = se.read_openephys(data_path, stream_name="Record Node 105#Neuropix-PXI-100.ProbeA-AP")


temp_path = data_path.parent / f"test_dataset_resaved_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
print("Saving...")
test_data.save(folder=temp_path, n_jobs=2)

# prep_data = correct_motion(test_data, preset="kilosort_like", n_jobs=20, chunk_duration='4s')
# from spikeinterface.preprocessing import correct_motion
"""
import numcodecs
compressor = numcodecs.registry.codec_registry["lzma"](preset=1)
temp_path = data_path.parent / f"test_dataset_resaved_compr_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
test_data.save(folder=temp_path,
                    format="zarr",
                    compressor=compressor,
                    channel_chunk_size=-1,
                    chunk_duration=1.,
                    n_jobs=-1,
                )
                """