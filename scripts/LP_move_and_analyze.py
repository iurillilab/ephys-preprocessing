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

# %%
# CUSTOM PARAMETERS TO MODIFY:

debug_mode = True
remove_laser_artifacts = True
drift_correction_spikeinterface = False
callKSfromSI = True

debug_data_path = Path(r"D:\luigi\short_test_trace_parallel.zarr")

# data locations to search for data, in decreasing order of priority:
data_locat_first_opt = [Path(r"E:\local_loc"), 
                        Path(r"F:\Luigi"), 
                        Path(r"N:\SNeuroBiology_shared\P02_MPAOPTO_LP\e05_doubleservoarm-ephys-pagfiber\v01")]

# folder to copy data to before processing:
temporary_folder = Path(r"D:\luigi")

# generate all tuples of (mouse, data_location, stream_name) to process:
all_mice = ["M24",] # ["M22b"] + [f"M{i}" for i in range(19, 25)]

# %%

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # final location to save processed data:
    final_data_loc = Path(r"E:\ks_processed") / f"{timestamp}_{all_mice}_debug-{debug_mode}"
    final_data_loc.mkdir(exist_ok=True, parents=True)

    # save log file in final data location, and also print to console:
    logger = get_logger(final_data_loc / "processing_log.log")
    logger.info(f"Processing started at {timestamp}. Debug mode: {debug_mode}. Laser artifacts removed: {remove_laser_artifacts}. Drift correction in spikeinterface: {drift_correction_spikeinterface}. Call KS from SI: {callKSfromSI}")


    tuples_to_process = []
    for mouse in all_mice:
        data_location = None
        for folder_to_search in data_locat_first_opt + [temporary_folder]:
            try:
                data_location = next(folder_to_search.glob(f"*{mouse}*"))

                oephys_data = OEPhysDataFolder(data_location)
            except (StopIteration, AssertionError):
                continue

        if data_location is None:
            print(f"Could not find data for mouse {mouse}")
            continue
        
        all_stream_names, ap_stream_names = oephys_data.stream_names, oephys_data.ap_stream_names

        new_tuple = (mouse, data_location, tuple(ap_stream_names))
        print(new_tuple)
        tuples_to_process.append(new_tuple)

    if debug_mode:
        tuples_to_process = [("M24dummy", debug_data_path, ("Record Node 103#Neuropix-PXI-100.ProbeA-AP",))]

    logger.info(f"Processing the following tuples: {tuples_to_process}")

    # %%
    job_kwargs = get_job_kwargs(chunk_duration="10s")



    for mouse, data_location, ap_stream_names in tuples_to_process:
        # retrieve mouse data:
        logger.info(f"Processing mouse {mouse}...")
        expected_source_dir = temporary_folder / mouse
        if not expected_source_dir.exists():  
            start_time = timestamp.now()
            logger.info(f"Copying data from {source_data_dir} to {temporary_folder}")
            source_data_dir = copy_folder(source_data_dir, temporary_folder)

            logger.info(f"Data copied (Time: {timestamp.now() - start_time} s)")

        else:
            source_data_dir = expected_source_dir

        # Prepare folder where results will be saved:
        processed_data_dir = final_data_loc / f"{mouse}"
        processed_data_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"processing data from {source_data_dir}")

        # custom class to simplify parsing of info from oephys data folder
        oephys_data = OEPhysDataFolder(source_data_dir) if not debug_mode else None

        for current_stream in ap_stream_names:
            processed_stream_folder = processed_data_dir / current_stream
            processed_stream_folder.mkdir(exist_ok=True, parents=True)
            logger.info(f"Processing stream {current_stream}. Data saved to {processed_stream_folder}")

            # standard preprocessing pipeline:
            if "zarr" in data_location.name:
                oephys_extractor = read_zarr(data_location)
                
            else:
                oephys_extractor = se.read_openephys(oephys_data.path, stream_name=current_stream)


            # standard filtering pipeline:
            # Export to zarr array only if it does not exist:
            preprocessed_zarr_path = source_data_dir / f"{current_stream}_preprocessed.zarr"

            if not preprocessed_zarr_path.exists():
                # remove laser artifacts:
                if remove_laser_artifacts and not debug_mode:
                    logger.info(f"Removing laser artifacts from {current_stream}...")
                    # read laser onsets in NPX indexing from oephys data (takes care of barcode alignment internally):
                    laser_onset_idxs = find_laser_onsets_npx_idxs(oephys_data, current_stream)

                    zeroed_extractor = st.remove_artifacts(oephys_extractor, laser_onset_idxs, ms_after=11, ms_before=0.5)
                else:
                    zeroed_extractor = oephys_extractor

                preprocessed_interface = standard_preprocessing(zeroed_extractor)


                if drift_correction_spikeinterface:
                    # drift correction in spikeinterface:
                    logger.info(f"Drift correction for {current_stream}...")
                    preprocessed_interface = st.correct_motion(recording=preprocessed_interface, 
                                                            preset="kilosort_like", 
                                                            folder=processed_stream_folder)

                figures_folder = processed_stream_folder / f"ephys_processing_figs"
                figures_folder.mkdir(exist_ok=True, parents=True)

                # make all diagnostic plots:
                make_probe_plots(oephys_extractor, preprocessed_interface, figures_folder, current_stream)
                plot_raw_and_preprocessed(oephys_extractor, preprocessed_interface, saving_path=figures_folder / f"snippets_{current_stream}.png",
                                        stream_name=current_stream
                                        )

                if remove_laser_artifacts and not debug_mode:
                    show_laser_trigger_preprost(oephys_extractor, preprocessed_interface, laser_onset_idxs, 
                                                n_to_take=200, saving_path=figures_folder / f"laser_trigger_preprost_{current_stream}.png")
                

                # save preprocessed data to zarr:
                compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.BITSHUFFLE)
                recording_saved = preprocessed_interface.save_to_zarr(folder = preprocessed_zarr_path)

                logger.info(f"Preprocessing complete for {current_stream}. Data saved to {processed_stream_folder}")
            else:
                logger.info(f"Preprocessed zarr file already exists at {preprocessed_zarr_path}; loading...")
                recording_saved = read_zarr(preprocessed_zarr_path)

            sorting_ks4 = call_ks(recording_saved, 
                    current_stream, 
                    processed_stream_folder, 
                    callKSfromSI=callKSfromSI, 
                    remove_binary=True, 
                    drift_correction=True)
            
            compute_stats(processed_stream_folder, sorting_ks4, recording_saved, **job_kwargs)

            # plt.close('all')

        if not debug_mode:
            print(f"Cleaning up temporary folder {source_data_dir}")
            shutil.rmtree(source_data_dir)
            print("Temporary folder cleaned up.")

# %%
