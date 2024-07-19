"""
    Original scripts written by Optogenetics and Neural Engineering Core ONE Core
    University of Colorado, School of Medicine
    31.Oct.2021
    See bit.ly/onecore for more information, including a more detailed write up.
    ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    This version was adapted by PP @iurilliLab - 04.May.2024.
    
    Pass the path to the SynchData folder as argument to the script.
    There, it should find:
      - a barcode-event file  (sample_numbers_{stream_name}.npy)  for each available
        stream, in the format of concatenated onsets and offsets (no continuous data);
      - a timestamp file to be aligned  (timetamps_{stream_name}.npy)  for each
        available stream.
      - data from at least two different streams. If a stream_name contains "Probe", it
        will be deemed as the primary stream to align to. If multiple "Probe" streams
        are available, a NPX1 Probe will be preferred as primary stream.
    All output is saved as:  timestamps_{stream_name}_alignedTo_{primary_stream}.npy
    in the same SynchData folder.
    
    Example call:
    python barcode_sync_full.py '/Users/galileo/Data/RL_E1/E1_M16/SynchData'
"""
import numpy as np
import sys
import os
import re
from scipy.signal import find_peaks

# Parse system input for the data synchronization folder
# (a single SyncData folder containing all data needed for alignment)
sync_folder_path = sys.argv[1]

# Regex pattern to identify timestamp files
pattern = re.compile(r'^timestamps_(.*?)\.npy$')
all_streams = []
all_stream_samplerates = []
all_ts_files = []
primary_stream = None
fallback_probe_stream = None  # Fallback for the first file containing "Probe" (ephys file)

for file in os.listdir(sync_folder_path):
    match = pattern.match(file)
    if match:
        all_ts_files.append(file)
        # Extract the captured group, which is the stream name
        stream_name = match.group(1)
        all_streams.append(stream_name)
        
        # extract sample rate for each stream:
        mmap_array = np.load(os.path.join(sync_folder_path, file), mmap_mode='r')
        # Access only the first two points
        ts = mmap_array[:2]
        sample_rate = 1 / (ts[1] - ts[0]) #denominator is never zero
        all_stream_samplerates.append(sample_rate)
        
        # Check if the extra string contains "-AP" (i.e., NPX 1.0) and flag it as the primary file
        if "-AP" in stream_name and primary_stream is None:
            primary_stream = stream_name
            continue
        # Check for "Probe" to set as a fallback if needed
        if "Probe" in stream_name and fallback_probe_stream is None:
            fallback_probe_stream = stream_name
            continue
        
if not primary_stream and fallback_probe_stream:
    primary_stream = fallback_probe_stream
        
# Check if a primary stream has been identified
if primary_stream:
    print("Primary stream for synchronization:", primary_stream)
# Check if there are at least two streams and no primary stream has been set
elif len(all_streams) >= 2:
    primary_stream = all_streams[0]  # Arbitrarily choose the first stream as primary
    print("No primary or suitable fallback was found, using first available stream as primary:", primary_stream)
else:
    print("Not enough streams to perform any synchronization. Exiting.")
    sys.exit()  # Exit the script


def extract_barcodes(all_streams, all_stream_samplerates, sync_folder_path):
    # Extracting barcodes from all streams
    for index, stream_name in enumerate(all_streams):
        print(f"{stream_name}")
        raw_data_format = False
        signals_column = 0
        expected_sample_rate = all_stream_samplerates[index]
        global_tolerance = .20
        barcodes_name = f"extractedBCDs_{stream_name}"

        nbits = 32
        inter_barcode_interval = 5000
        ind_wrap_duration = 10
        ind_bar_duration = 30

        # Set Global Variables/Tolerances Based on User Input
        wrap_duration = 3 * ind_wrap_duration # Off-On-Off
        total_barcode_duration = nbits * ind_bar_duration + 2 * wrap_duration

        # Tolerance conversions
        min_wrap_duration = ind_wrap_duration - ind_wrap_duration * global_tolerance
        max_wrap_duration = ind_wrap_duration + ind_wrap_duration * global_tolerance
        min_bar_duration = ind_bar_duration - ind_bar_duration * global_tolerance
        max_bar_duration = ind_bar_duration + ind_bar_duration * global_tolerance
        sample_conversion = 1000 / expected_sample_rate # Convert sampling rate to msec

        # Select Data Input File / Barcodes Output Directory ###
        signals_file = os.path.join(sync_folder_path, f"sample_numbers_{stream_name}.npy")
        
        # Signals Data Extraction & Manipulation
        try:
            signals_numpy_data = np.load(signals_file)
            signals_located = True
        except:
            signals_numpy_data = ''
            print("Signals .npy file not located; please check your filepath")
            print(f"{stream_name}")
            signals_located = False

        # Check whether signals_numpy_data exists; if not, end script with sys.exit().
        if signals_located:
            #LJ = If data is in raw format and has not been sorted by "peaks"
            if raw_data_format:

                # Extract the signals_column from the raw data
                barcode_column = signals_numpy_data[:, signals_column]
                barcode_array = barcode_column.transpose()
                # Extract the indices of all events when TTL pulse changed value.
                event_index, _ = find_peaks(np.abs(np.diff(barcode_array)), height=0.9)
                # Convert the event_index to indexed_times to align with later code.
                indexed_times = event_index # Just take the index values of the raw data

            # NP = Collect the pre-extracted indices from the signals_column.
            else:
                    indexed_times = signals_numpy_data.flatten()

            # Find time difference between index values (ms), and extract barcode wrappers.
            events_time_diff = np.diff(indexed_times) * sample_conversion # convert to ms
            wrapper_array = indexed_times[np.where(
                            np.logical_and(min_wrap_duration < events_time_diff,
                                           events_time_diff  < max_wrap_duration))[0]]

            # Isolate the wrapper_array to wrappers with ON values, to avoid any
            # "OFF wrappers" created by first binary value.
            false_wrapper_check = np.diff(wrapper_array) * sample_conversion # Convert to ms
            # Locate indices where two wrappers are next to each other.
            false_wrappers = np.where(
                             false_wrapper_check < max_wrap_duration)[0]
            # Delete the "second" wrapper (it's an OFF wrapper going into an ON bar)
            wrapper_array = np.delete(wrapper_array, false_wrappers+1)

            # Find the barcode "start" wrappers, set these to wrapper_start_times, then
            # save the "real" barcode start times to signals_barcode_start_times, which
            # will be combined with barcode values for the output .npy file.
            wrapper_time_diff = np.diff(wrapper_array) * sample_conversion # convert to ms
            barcode_index = np.where(wrapper_time_diff < total_barcode_duration)[0]
            wrapper_start_times = wrapper_array[barcode_index]
            signals_barcode_start_times = wrapper_start_times - ind_wrap_duration / sample_conversion
            # Actual barcode start is 10 ms before first 10 ms ON value.

            # Using the wrapper_start_times, collect the rest of the indexed_times events
            # into on_times and off_times for barcode value extraction.
            on_times = []
            off_times = []
            for idx, ts in enumerate(indexed_times):    # Go through indexed_times
                # Find where ts = first wrapper start time
                if ts == wrapper_start_times[0]:
                    # All on_times include current ts and every second value after ts.
                    on_times = indexed_times[idx::2]
                    off_times = indexed_times[idx+1::2] # Everything else is off_times

            # Convert wrapper_start_times, on_times, and off_times to ms
            wrapper_start_times = wrapper_start_times * sample_conversion
            on_times = on_times * sample_conversion
            off_times = off_times * sample_conversion

            signals_barcodes = []
            for start_time in wrapper_start_times:
                oncode = on_times[
                    np.where(
                        np.logical_and(on_times > start_time,
                                       on_times < start_time + total_barcode_duration)
                    )[0]
                ]
                offcode = off_times[
                    np.where(
                        np.logical_and(off_times > start_time,
                                       off_times < start_time + total_barcode_duration)
                    )[0]
                ]
                curr_time = offcode[0] + ind_wrap_duration # Jumps ahead to start of barcode
                bits = np.zeros((nbits,))
                interbit_ON = False # Changes to "True" during multiple ON bars

                for bit in range(0, nbits):
                    next_on = np.where(oncode >= (curr_time - ind_bar_duration * global_tolerance))[0]
                    next_off = np.where(offcode >= (curr_time - ind_bar_duration * global_tolerance))[0]

                    if next_on.size > 1:    # Don't include the ending wrapper
                        next_on = oncode[next_on[0]]
                    else:
                        next_on = start_time + inter_barcode_interval

                    if next_off.size > 1:    # Don't include the ending wrapper
                        next_off = offcode[next_off[0]]
                    else:
                        next_off = start_time + inter_barcode_interval

                    # Recalculate min/max bar duration around curr_time
                    min_bar_duration = curr_time - ind_bar_duration * global_tolerance
                    max_bar_duration = curr_time + ind_bar_duration * global_tolerance

                    if min_bar_duration <= next_on <= max_bar_duration:
                        bits[bit] = 1
                        interbit_ON = True
                    elif min_bar_duration <= next_off <= max_bar_duration:
                        interbit_ON = False
                    elif interbit_ON == True:
                        bits[bit] = 1

                    curr_time += ind_bar_duration

                barcode = 0

                for bit in range(0, nbits):             # least sig left
                    barcode += bits[bit] * pow(2, bit)

                signals_barcodes.append(barcode)

        else: # If signals_located = False
            sys.exit("Data not found. Program has stopped.")

        # Print out final output and save to chosen file format(s)

        # Create merged array with timestamps stacked above their barcode values
        signals_time_and_bars_array = np.vstack((signals_barcode_start_times,
                                                 np.array(signals_barcodes)))
        print(f"Extracted barcodes for {stream_name}")
        print("Saving to disk...")

        output_file = os.path.join(sync_folder_path, barcodes_name)
        np.save(output_file, signals_time_and_bars_array)


def align_barcodes(all_streams, primary_stream, all_stream_samplerates, sync_folder_path):
    # aligning barcodes to primary stream
    primary_index = all_streams.index(primary_stream)
    for index, stream_name in enumerate(all_streams):
        if stream_name != primary_stream:
            print(f"Registering stream: {stream_name} to {primary_stream}")
            main_sample_rate = all_stream_samplerates[primary_index]
            secondary_sample_rate = all_stream_samplerates[index]
            convert_timestamp_column = 0 # Column that timestamps are located in secondary data
            alignment_name = f"timestamps_{stream_name}_alignedTo_{primary_stream}"  # Name of output file.
            main_dir_and_name = os.path.join(sync_folder_path, f"extractedBCDs_{primary_stream}.npy")
            secondary_dir_and_name = os.path.join(sync_folder_path, f"extractedBCDs_{stream_name}.npy")
            secondary_raw_data = os.path.join(sync_folder_path, f"timestamps_{stream_name}.npy")
            main_numpy_data = np.load(main_dir_and_name)
            secondary_numpy_data = np.load(secondary_dir_and_name)
            secondary_data_original = np.load(secondary_raw_data)

            # Extract Barcodes and Index Values, then Calculate Linear Variables
            # Pull the barcode row from the data. 1st column is timestamps, second is barcodes
            barcode_timestamps_row = 0 # Same for both main and secondary, because we used our own code
            barcodes_row = 1 # Same for both main and secondary

            main_numpy_barcode = main_numpy_data[barcodes_row, :]
            secondary_numpy_barcode = secondary_numpy_data[barcodes_row, :]

            main_numpy_timestamp = main_numpy_data[barcode_timestamps_row, :]
            secondary_numpy_timestamp = secondary_numpy_data[barcode_timestamps_row, :]

            # Pull the index values from barcodes shared by both groups of data
            shared_barcodes, main_index, second_index = np.intersect1d(main_numpy_barcode,
                                                        secondary_numpy_barcode, return_indices=True)
            # Note: To intersect more than two arrays, use functools.reduce

            # Use main_index and second_index arrays to extract related timestamps
            main_shared_barcode_times = main_numpy_timestamp[main_index]
            secondary_shared_barcode_times = secondary_numpy_timestamp[second_index]

            # Determine slope (m) between main/secondary timestamps
            m = (main_shared_barcode_times[-1]-main_shared_barcode_times[0])/(secondary_shared_barcode_times[-1]-secondary_shared_barcode_times[0])
            # Determine offset (b) between main and secondary barcode timestamps
            b = main_shared_barcode_times[0] - secondary_shared_barcode_times[0] * m

            print('Linear conversion from secondary timestamps to main:\ny = ', m, 'x + ', b)

            # Apply Linear Conversion to Secondary Data (in .npy Format) ###
            secondary_data_original[:,convert_timestamp_column] = secondary_data_original[:,convert_timestamp_column] * secondary_sample_rate * m + b
            secondary_data_converted = secondary_data_original # To show conversion complete.
            
            # Clean up conversion of values to nearest whole number
            #print("Total number of index values: ", len(secondary_data_converted[:,convert_timestamp_column]))
            for index in range(0,len(secondary_data_converted[:,convert_timestamp_column])):
                value = secondary_data_converted[index, convert_timestamp_column]
                rounded_val = value.astype('int')
                secondary_data_converted[index, convert_timestamp_column] = rounded_val
                
            # Make it in actual timestamps (previous output is sample numbers in primary coordinates)
            # I am sure this previous step could be done better
            secondary_data_converted.astype(np.float64)
            secondary_data_converted = secondary_data_converted/main_sample_rate
            
            # Print out final output and save to chosen file format(s)
            print(f"Final output for {stream_name}:\n{secondary_data_converted}")
            output_file = os.path.join(sync_folder_path, alignment_name)
            np.save(output_file, secondary_data_converted)
            print("Aligned timestamps saved to disk. Deleting the original timestamps...")
            try:
                os.remove(secondary_raw_data)
                print(f"File {secondary_raw_data} successfully deleted.")
            except FileNotFoundError:
                print(f"File {secondary_raw_data} does not exist and could not be deleted.")
            except PermissionError:
                print(f"Permission denied: Unable to delete {secondary_raw_data}.")
            except Exception as e:
                print(f"An error occurred: {e}")


# step 1: barcode extraction (ex scripts s1 and s2)
extract_barcodes(all_streams, all_stream_samplerates, sync_folder_path)

# step 2: align barcodes (ex scritps s3 and s4)
align_barcodes(all_streams, primary_stream, all_stream_samplerates, sync_folder_path)

# when successfully executed, clean up intermediate output that is no longer needed:
print("Deleting intermediate output files...")
for stream_name in all_streams:
    signals_file = os.path.join(sync_folder_path, f"sample_numbers_{stream_name}.npy")
    barcode_file = os.path.join(sync_folder_path, f"extractedBCDs_{stream_name}.npy")
    os.remove(signals_file)
    os.remove(barcode_file)
    
print("Alignment operations concluded successfully.")
