from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import pynapple as nap


def test_on_temp_nwb_file(interface, nwb_path: Path, conversion_options: dict = None):
    assert nwb_path.parent.exists(), f"Folder {nwb_path} does not exist"
    # nwb_path = folder_path.parent / "tracking_output.nwb"
    if nwb_path.exists():
        nwb_path.unlink()
    
    metadata = interface.get_metadata()
    session_start_time = datetime(2020, 1, 1, 12, 30, 0, tzinfo=ZoneInfo("US/Pacific"))
    metadata["NWBFile"].update(session_start_time=session_start_time)

    try:
        interface.run_conversion(nwbfile_path=nwb_path, metadata=metadata)  # conversion_options=conversion_options)
    except TypeError as e:
        if "BaseSortingExtractorInterface.add_to_nwbfile() got an unexpected keyword argument 'conversion_options'" in str(e):
            interface.run_conversion(nwbfile_path=nwb_path, metadata=metadata)
        else:
            raise e

    return nap.load_file(nwb_path)

