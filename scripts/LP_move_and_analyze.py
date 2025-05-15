# %%
from pathlib import Path
import shutil
from datetime import datetime
from matplotlib import pyplot as plt
import spikeinterface.extractors as se
import spikeinterface.preprocessing as st
from spikeinterface import read_zarr, create_sorting_analyzer, load_sorting_analyzer

from pathlib import Path
import numcodecs
from preprocessing_utils import *
from nwb_conv.oephys import OEPhysDataFolder
from datetime import datetime
import warnings

# suppress all warnings:
warnings.filterwarnings("ignore")

# %%
# CUSTOM PARAMETERS TO MODIFY:

debug_mode = False
test_on_subset = False
remove_laser_artifacts = True
drift_correction_spikeinterface = False
callKSfromSI = True

debug_data_path = Path(r"D:\luigi\short_test_trace.zarr")

# data locations to search for data, in decreasing order of priority:
data_locat_first_opt = [Path(r"N:\SNeuroBiology_shared\P02_MPAOPTO_LP\e05_doubleservoarm-ephys-pagfiber\v01"),
                        Path(r"F:\Luigi"),
                        Path(r"E:\local_loc")]

# folder to copy data to before processing:
master_temporary_folder = Path(r"D:\luigi")
master_final_data_loc = Path(r"E:\ks_processed")

# generate all tuples of (mouse, data_location, stream_name) to process:
all_mice = ["M23"]  #["M20", "M24", "M21", "M23", "M22", "M22b", "M19"] 

n_seconds_subset = 300
start_frame = 0

# %%
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # final location to save processed data:
    final_data_loc = master_final_data_loc / f"{timestamp}_{all_mice}_debug-{debug_mode}"
    final_data_loc.mkdir(exist_ok=True, parents=True)

    # save log file in final data location, and also print to console:
    logger = get_logger(final_data_loc / "processing_log.log")
    logger.info(f"Processing started at {timestamp}. Debug mode: {debug_mode}. Laser artifacts removed: {remove_laser_artifacts}. Drift correction in spikeinterface: {drift_correction_spikeinterface}. Call KS from SI: {callKSfromSI}")


    tuples_to_process = []
    for mouse in all_mice:
        data_location = None
        for folder_to_search in data_locat_first_opt: # + [temporary_folder]:
            try:
                data_location = next(folder_to_search.glob(f"*{mouse}*"))

                oephys_data = OEPhysDataFolder(data_location)
            except (StopIteration, AssertionError):
                continue

        if data_location is None:
            print(f"Could not find data for mouse {mouse}")
            continue
        
        all_stream_names, ap_stream_names = oephys_data.stream_names, oephys_data.ap_stream_names

        new_tuple = (mouse, data_location, tuple(ap_stream_names[::-1]))
        print(new_tuple)
        tuples_to_process.append(new_tuple)

    if debug_mode:
        tuples_to_process = [("M24dummy", debug_data_path, ("Record Node 103#Neuropix-PXI-100.ProbeA-AP",))]

    logger.info(f"Processing the following tuples: {tuples_to_process}")

    # %%
    job_kwargs = get_job_kwargs(chunk_duration="1s")
    si.set_global_job_kwargs(**job_kwargs)


    for mouse, raw_data_location, ap_stream_names in tuples_to_process:
        # retrieve mouse data:
        logger.info(f"Processing mouse {mouse}...")

        mouse_temporary_data_folder = master_temporary_folder / mouse
        mouse_temporary_data_folder.mkdir(exist_ok=True, parents=True)

        # Prepare folder where results will be saved:
        mouse_processed_data_dir = final_data_loc / f"{mouse}"
        mouse_processed_data_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"processing data from {raw_data_location}. Preprocessed data in {mouse_temporary_data_folder}. Final data in {mouse_processed_data_dir}")


        # custom class to simplify parsing of info from oephys data folder
        oephys_data = OEPhysDataFolder(raw_data_location) if not debug_mode else None

        for current_stream in ap_stream_names:
            # temporary folder for the current stream:
            stream_temporary_data_folder = mouse_temporary_data_folder / current_stream
            stream_temporary_data_folder.mkdir(exist_ok=True, parents=True)

            # Data folder for the current stream:
            stream_final_data_folder = mouse_processed_data_dir / current_stream
            stream_final_data_folder.mkdir(exist_ok=True, parents=True)

            logger.info(f"Processing stream {current_stream}. Final data will be saved to {stream_final_data_folder}")

            # standard preprocessing pipeline:
            if "zarr" in data_location.name:
                oephys_extractor = read_zarr(data_location)
                
            else:
                oephys_extractor = se.read_openephys(oephys_data.path, stream_name=current_stream)


            # standard filtering pipeline:
            # Export to zarr array only if it does not exist:
            preprocessed_zarr_path = stream_temporary_data_folder / f"{current_stream}_preprocessed.zarr"

            if not preprocessed_zarr_path.exists():
                # remove laser artifacts:
                if remove_laser_artifacts and not debug_mode and not test_on_subset:
                    logger.info(f"Removing laser artifacts from {current_stream}...")
                    # read laser onsets in NPX indexing from oephys data (takes care of barcode alignment internally):
                    laser_onset_idxs = find_laser_onsets_npx_idxs(oephys_data, current_stream)

                    zeroed_extractor = st.remove_artifacts(oephys_extractor, laser_onset_idxs, ms_after=11, ms_before=0.5)
                else:
                    zeroed_extractor = oephys_extractor

                if test_on_subset:
                    end_frame = int(n_seconds_subset * zeroed_extractor.get_sampling_frequency())
                    zeroed_extractor = zeroed_extractor.frame_slice(start_frame, end_frame)

                preprocessed_interface = standard_preprocessing(zeroed_extractor)

                if drift_correction_spikeinterface:
                    # drift correction in spikeinterface:
                    logger.info(f"Drift correction for {current_stream}...")
                    preprocessed_interface = st.correct_motion(recording=preprocessed_interface, 
                                                               preset="kilosort_like", 
                                                               folder=stream_temporary_data_folder)  # TODO check what is saved here

                figures_folder = stream_final_data_folder / f"ephys_processing_figs"
                figures_folder.mkdir(exist_ok=True, parents=True)

                # make all diagnostic plots:
                make_probe_plots(oephys_extractor, preprocessed_interface, figures_folder, current_stream)
                plot_raw_and_preprocessed(oephys_extractor, preprocessed_interface, saving_path=figures_folder / f"snippets_{current_stream}.png",
                                        stream_name=current_stream
                                        )

                if remove_laser_artifacts and not debug_mode and not test_on_subset:
                    show_laser_trigger_preprost(oephys_extractor, preprocessed_interface, laser_onset_idxs, 
                                                n_to_take=200, saving_path=figures_folder / f"laser_trigger_preprost_{current_stream}.png")
                

                # save preprocessed data to zarr:
                logger.info(f"Saving preprocessed trace to {preprocessed_zarr_path}...")
                compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    recording_saved = preprocessed_interface.save_to_zarr(folder = preprocessed_zarr_path, **job_kwargs)

                logger.info(f"Preprocessing complete for {current_stream}. Data saved to {stream_final_data_folder}")
            else:
                logger.info(f"Preprocessed zarr file already exists at {preprocessed_zarr_path}; loading...")
                recording_saved = read_zarr(preprocessed_zarr_path)

                if test_on_subset:
                    end_frame = int(n_seconds_subset * recording_saved.get_sampling_frequency())
                    recording_saved = recording_saved.frame_slice(start_frame, end_frame)

            sorting_ks4 = call_ks(recording_saved, 
                                  current_stream, 
                                  stream_final_data_folder, 
                                  callKSfromSI=callKSfromSI, 
                                  remove_binary=True, 
                                  drift_correction=True,
                                  n_jobs=job_kwargs["n_jobs"])
            
            # compute_stats(stream_final_data_folder, sorting_ks4, recording_saved, **job_kwargs)


        if not debug_mode:
            print(f"Cleaning up temporary folder {mouse_temporary_data_folder}")
            shutil.rmtree(mouse_temporary_data_folder)
            print("Temporary folder cleaned up.")

# %%
