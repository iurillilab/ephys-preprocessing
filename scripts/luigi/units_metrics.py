#%%
import spikeinterface.core as si
import spikeinterface.extractors as se
from spikeinterface.postprocessing import compute_principal_components
from spikeinterface.qualitymetrics import (
    compute_snrs,
    compute_firing_rates,
    compute_isi_violations,
    calculate_pc_metrics,
    compute_quality_metrics,
)
from spikeinterface.extractors import read_openephys
from spikeinterface.sorters import read_sorter_folder
from multiprocessing import freeze_support
from nwb_conv.oephys import OEPhysDataFolder
from pathlib import Path
from datetime import datetime

from spikeinterface import set_global_job_kwargs

set_global_job_kwargs(n_jobs=-1)

if __name__ == '__main__':
    freeze_support()
    # local_path = si.download_dataset(remote_path="mearec/mearec_test_10s.h5")
    # recording, sorting = se.read_mearec(local_path)
    # print(recording)
    #print(sorting)
    data_path = Path("/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P02_MPAOPTO_LP/e05_doubleservoarm-ephys-pagfiber/v01/M20_D545/20240424/154810")

    npx_folder = OEPhysDataFolder(data_path / "NpxData")
    for stream_name in npx_folder.ap_stream_names:
        start_time = datetime.now()
        recording = read_openephys(npx_folder.path, stream_name=stream_name)
        # %%
        print(stream_name)
        spike_data_path = data_path / "Sorting" / stream_name
        sorting = read_sorter_folder(spike_data_path / "0",
                                     register_recording=False)
        # %%

        analyzer = si.create_sorting_analyzer(sorting=sorting, recording=recording, format="memory")
        print(analyzer)
        analyzer.save_as(format="binary_folder", folder=spike_data_path / "analyzer")

        analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=600, seed=2205)
        analyzer.compute("waveforms", ms_before=1.3, ms_after=2.6)
        analyzer.compute("templates", operators=["average", "median", "std"])
        analyzer.compute("noise_levels")

        print(analyzer)

        firing_rates = compute_firing_rates(analyzer)
        print(firing_rates)
        isi_violation_ratio, isi_violations_count = compute_isi_violations(analyzer)
        print(isi_violation_ratio)
        snrs = compute_snrs(analyzer)
        print(snrs)

        #metrics = compute_quality_metrics(analyzer, metric_names=["firing_rate", "snr", "amplitude_cutoff"])
        #print(metrics)

        analyzer.compute("principal_components", n_components=3, mode="by_channel_global", whiten=True)

        metrics = compute_quality_metrics(
            analyzer,
            metric_names=[
                "num_spikes",
                "firing_rate",
                "presence_ratio",
                "snr",
                "isi_violation",
                "rp_violation",
                "sliding_rp_violation",
                "amplitude_cutoff",
                "amplitude_median",
                "amplitude_cv",
                "synchrony",
                "firing_range",
                "drift",
                "sd_ratio",
            ],
        )
        print(metrics)
        print(metrics.columns)
        end_time = datetime.now()
        metrics.to_csv(spike_data_path / "metrics.csv")

        print(f"Time taken: {(end_time - start_time).total_seconds()}")