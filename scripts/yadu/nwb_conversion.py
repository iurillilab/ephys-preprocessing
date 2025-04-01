#%%
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from neuroconv.datainterfaces import KiloSortSortingInterface
#%%

folder_path = r'Y:\20250124\M20\test_npx1\2025-01-24_19-56-04\Record Node 102\experiment1\recording1\continuous\Neuropix-PXI-100.ProbeB-AP\kilosort4\sorter_output'
# Change the folder_path to the location of the data in your system
interface = KiloSortSortingInterface(folder_path=folder_path, verbose=False)

metadata = interface.get_metadata()
session_start_time = datetime(2020, 1, 1, 12, 30, 0, tzinfo=ZoneInfo("US/Pacific"))
metadata["NWBFile"].update(session_start_time=session_start_time)
nwbfile_path = r'Y:\20250124\M20\test_npx1\2025-01-24_19-56-04\Record Node 102\experiment1\recording1\continuous\Neuropix-PXI-100.ProbeB-AP\kilosort4'
interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)
# %%
