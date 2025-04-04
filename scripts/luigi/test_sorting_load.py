from spikeinterface.sorters import read_sorter_folder
from nwb_conv.oephys import OEPhysDataFolder
from pathlib import Path

data_path = Path("/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P02_MPAOPTO_LP/e05_doubleservoarm-ephys-pagfiber/v01/M20_D545/20240424/154810")

npx_folder = OEPhysDataFolder(data_path / "NpxData")
stream_name = npx_folder.ap_stream_names[0]
sorting = read_sorter_folder(data_path / "Sorting" / stream_name / "0", register_recording=False)
print(sorting)