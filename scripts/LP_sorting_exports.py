import spikeinterface as si # core module only
from spikeinterface.postprocessing import compute_spike_amplitudes, compute_principal_components
from spikeinterface.exporters import export_to_phy, export_report
from pathlib import Path
from spikeinterface import read_zarr

from spikeinterface.sorters import Kilosort4Sorter

sorter_folder_path = Path(r"E:\ks_processed\20240903-161925_['M20', 'M24', 'M21', 'M23', 'M22', 'M22b', 'M19']_debug-False\M20\Record Node 103#Neuropix-PXI-100.ProbeA-AP\0")
recording_path = Path(r"D:\luigi\short_test_trace.zarr\Record Node 103#Neuropix-PXI-100.ProbeA-AP_preprocessed.zarr")

recording = read_zarr(recording_path)


sorting = Kilosort4Sorter.get_result_from_folder(sorter_folder_path)
# # the waveforms are sparse so it is faster to export to phy
sorting_analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording)
# sorting_analyzer = si.load_sorting_analyzer(sorter_folder / "analyzer")

# # some computations are done before to control all options
sorting_analyzer.compute(['random_spikes', 'waveforms', 'templates', 'noise_levels'])
_ = sorting_analyzer.compute('spike_amplitudes')
_ = sorting_analyzer.compute('principal_components', n_components = 5, mode="by_channel_local")
_ = sorting_analyzer.compute('quality_metrics')
_ = sorting_analyzer.compute('correlograms')

# # the export process is fast because everything is pre-computed
# export_to_phy(sorting_analyzer=sorting_analyzer, output_folder=sorter_folder_path / "phy")
export_report(sorting_analyzer=sorting_analyzer, output_folder=sorter_folder_path / "report")

