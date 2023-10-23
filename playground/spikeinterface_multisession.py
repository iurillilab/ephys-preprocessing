# %%
import spikeinterface.extractors as se
import spikeinterface.full as si
from matplotlib import pyplot as plt
from pathlib import Path
from spikeinterface import append_recordings
# %%
from spikeinterface.preprocessing import common_reference, phase_shift, highpass_filter, bandpass_filter, \
        detect_bad_channels, interpolate_bad_channels, highpass_spatial_filter, correct_motion

from spikeinterface.sorters import run_sorter, Kilosort3Sorter
Kilosort3Sorter.set_kilosort3_path(r"C:\Users\SNeurobiology\Documents\MATLAB\kilosort3-master")

# %%
master_path = Path(r"E:\Luigi-temp\M10\NPXData")
all_data_sessions = sorted(master_path.glob("*"))
all_si_raw_reads = [se.read_openephys(str(f), stream_name="Record Node 105#Neuropix-PXI-100.ProbeA-AP") 
                    for f in all_data_sessions]
pipeline_steps_dict = dict()
pipeline_steps_dict["raw"] = append_recordings(all_si_raw_reads)
#channel_ids = pipeline_steps_dict["raw"].channel_ids
pipeline_steps_dict["raw"]

# %%
pipeline_steps_dict["hp_raw"]  = highpass_filter(pipeline_steps_dict["raw"])
pipeline_steps_dict["shift_hp_raw"] = phase_shift(pipeline_steps_dict["hp_raw"])
bad_channel_ids = detect_bad_channels(pipeline_steps_dict["shift_hp_raw"])
pipeline_steps_dict["shift_hp_raw"] = interpolate_bad_channels(pipeline_steps_dict["shift_hp_raw"], bad_channel_ids[0])

#pipeline_steps_dict["sp_car_shift_hp_raw"] = common_reference(pipeline_steps_dict["shift_hp_raw"], operator="median", reference="global")
#pipeline_steps_dict["sp_car_shift_hp_raw"] = highpass_spatial_filter(pipeline_steps_dict["sp_car_shift_hp_raw"])
# pipeline_steps_dict["whiten_sp_car_shift_hp_raw"] = whiten(pipeline_steps_dict["sp_car_shift_hp_raw"])

pipeline_steps_dict["sp_glcar_shift_hp_raw"] = common_reference(pipeline_steps_dict["shift_hp_raw"], operator="median", 
                                                                reference="local", local_radius=(100, 400))
pipeline_steps_dict["sp_glcar_shift_hp_raw"] = highpass_spatial_filter(pipeline_steps_dict["sp_glcar_shift_hp_raw"])

#pipeline_steps_dict["sp_car_hp_raw"] = common_reference(pipeline_steps_dict["hp_raw"], operator="median", reference="global")
#pipeline_steps_dict["sp_car_hp_raw"] = highpass_spatial_filter(pipeline_steps_dict["sp_car_hp_raw"])


# %%
# time_range=(621, 624)
# fig, ax = plt.subplots(figsize=(20, 10), sharex=True)
# sel_k = "hp_raw"
# si.plot_traces({sel_k: pipeline_steps_dict[sel_k]}, 
#                time_range=time_range,
#                 backend='matplotlib', mode='line', ax=ax, channel_ids=channel_ids[[150]])
# fig.show()

# time_range = (622, 623)
# some_chans = channel_ids[[10, 50, 300, 350]]
# sel_steps = ["hp_raw", "sp_car_shift_hp_raw", "sp_glcar_shift_hp_raw", "bp_sp_glcar_shift_hp_raw"]
# # plot some channels
# fig, axs = plt.subplots(nrows=len(sel_steps), figsize=(20, 10), sharex=True)
# some_chans = channel_ids[[10, 50, 100, 150, 300]]
# for i, key in enumerate(sel_steps):
#     si.plot_traces({key: pipeline_steps_dict[key]}, time_range=time_range,
#                    backend='matplotlib', mode='line', ax=axs[i], channel_ids=some_chans)
# fig.show()

# fig, axs = plt.subplots(ncols=4, figsize=(20, 10), sharex=True, sharey=True)
# sel_steps = ["hp_raw", "sp_car_shift_hp_raw", "sp_glcar_shift_hp_raw", "bp_sp_glcar_shift_hp_raw"]
# for i, key in enumerate(sel_steps):
#     si.plot_traces(pipeline_steps_dict[key], backend='matplotlib',
#                    seconds_per_row=0.02,
#                    order_channel_by_depth=True,  
#                    time_range=time_range,
#                    ax=axs[i])
#     axs[i].set_title(key)
# fig.show()

# %%
from spikeinterface.core import load_extractor
# each import is needed
from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
from spikeinterface.sortingcomponents.motion_interpolation import interpolate_motion

# job_kwargs = dict(chunk_duration="1s", n_jobs=20, progress_bar=True)
# Step 1 : activity profile
# peaks = detect_peaks(pipeline_steps_dict["sp_glcar_shift_hp_raw"], 
#                      method="locally_exclusive_torch", detect_threshold=10.0, **job_kwargs)
# # (optional) sub-select some peaks to speed up the localization
# # %%
# peaks = select_peaks(peaks, ...)
# peak_locations = localize_peaks(pipeline_steps_dict["shift_hp_raw"], peaks, 
#                                 method="monopolar_triangulation",radius_um=75.0,
#                                 max_distance_um=150.0, **job_kwargs)

# # Step 2: motion inference
# motion, temporal_bins, spatial_bins = estimate_motion(rec, peaks, peak_locations,
#                                                       method="decentralized",
#                                                       direction="y",
#                                                       bin_duration_s=2.0,
#                                                       bin_um=5.0,
#                                                       win_step_um=50.0,
#                                                       **job_kwargs)

# pipeline_steps_dict["on_disk"] = pipeline_steps_dict["sp_glcar_shift_hp_raw"].save(folder=str(Path(f).parent / "clean_data_3"), n_jobs=10, chunk_duration='1s', progres_bar=True)
# filtered_on_disk = load_extractor(f.parent / clean_data_dirname)
# pipeline_steps_dict["bandpass"]  = bandpass_filter(pipeline_steps_dict["on_disk"], 200, 5000)
dir(pipeline_steps_dict["sp_glcar_shift_hp_raw"])
pipeline_steps_dict["sp_glcar_shift_hp_raw"].select_segments(0)


job_kwargs = dict(chunk_duration="1s", n_jobs=10, progress_bar=True)
recs_corrected = [correct_motion(pipeline_steps_dict["sp_glcar_shift_hp_raw"].select_segments(i), 
                                preset="kilosort_like", 
                                **job_kwargs) for i in range(2)]
# rec_corrected = correct_motion(pipeline_steps_dict["shift_hp_raw"] , preset="nonrigid_accurate",
#                                detect_kwargs=dict(
#                                    detect_threshold=10.),
#                                estimate_motion_kwargs=dict(
#                                    histogram_depth_smooth_um=8.,
#                                    time_horizon_s=120.,
#                                ),
#                                interpolate_motion_kwargs=dict(
#                                     spatial_interpolation_method="kriging",
#                                )
#                                )

# %%

corrected_data_dirname = master_path / "drift_corrected"
pipeline_steps_dict["motion_corr"] = rec_corrected.save(folder=str(corrected_data_dirname), 
                                                        n_jobs=20, 
                                                        chunk_duration='4s', 
                                                        progres_bar=True)

# %%
motion_corr = load_extractor(f.parent / corrected_data_dirname)
sorted_data_foldername = "sorted_kilosort4"

sorting_KS3 = run_sorter("kilosort3", 
                         motion_corr, 
                         output_folder=str(f.parent / sorted_data_foldername),  
                         do_correction=False, 
                         sig=20,
                         nblocks=5,
                         projection_threshold=[9, 9],
                         freq_min=300,
                         preclust_threshold=6, 
                         skip_kilosort_preprocessing=True,
                         car=False)

# %%
from spikeinterface.postprocessing import compute_spike_amplitudes, compute_correlograms
from spikeinterface import extract_waveforms
from spikeinterface.qualitymetrics import compute_quality_metrics
from spikeinterface.exporters import export_report, export_to_phy


# the waveforms are sparse for more interpretable figures
we = extract_waveforms(motion_corr, sorting_KS3, folder=str(f.parent / "waveforms"), sparse=True, n_jobs=20)

# some computations are done before to control all options
compute_spike_amplitudes(we, n_jobs=20)
compute_correlograms(we)
compute_quality_metrics(we, metric_names=['snr', 'isi_violation', 'presence_ratio'], n_jobs=20)

# the export process
export_report(we, output_folder=str(f.parent / "spikes_report"), n_jobs=20)
export_to_phy(we, output_folder=str(f.parent / "phy_export"), n_jobs=20)
# %%
