"""Recording processor for spike sorting pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import shutil
import time
import os
import subprocess
import traceback
import re
from pprint import pprint
import hashlib

import spikeinterface.sorters as ss
from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor
import spikeinterface.preprocessing as spre
from spikeinterface import aggregate_channels
from spikeinterface.core import concatenate_recordings
from spikeinterface import create_sorting_analyzer
import spikeinterface.full as si
import spikeinterface.curation as scur

import torch


N_JOBS = os.cpu_count() // 2

# Example paths for testing
EXAMPLE_PATHS = {
    'source_dir':  Path('/mnt/d/temp_processing'),
    'working_dir': Path('/mnt/d/temp_processing'),
    # 'source_dir':  Path("/Users/vigji/Desktop/short_recording_oneshank"),
    # 'working_dir': None,
    'test_recording': (
        Path("/Volumes/Extreme SSD/2024-11-13_14-39-11"),
        "Record Node 111#Neuropix-PXI-110.ProbeA"
    )
}

#################################
# Utils
#################################

def _n_seconds_to_formatted_time(n_seconds: int) -> str:
    """Convert seconds to formatted time string."""
    n_seconds = int(n_seconds)
    hours, remainder = divmod(n_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def is_timestamp_folder(folder: Path) -> bool:
    """Check if a folder name matches the timestamp pattern."""
    timestamp_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
    return bool(timestamp_pattern.match(folder.name)) and folder.is_dir()

def copy_folder(src: Union[str, Path], dst: Union[str, Path], overwrite: bool = True) -> Path:
    """Copy a folder to a new location.
    
    Args:
        src: Source folder path
        dst: Destination folder path
        overwrite: If True, overwrites files that are:
            - Different size than source
            - Newer in source (mtime)
    """
    src, dst = Path(src), Path(dst)
    dst = dst / src.name
    dst.mkdir(exist_ok=True, parents=True)
    
    for item in src.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(src)
            dest_path = dst / rel_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if we should overwrite
            should_copy = False
            if not dest_path.exists():
                should_copy = True
            elif overwrite:
                # Check if source is newer or size is different
                if (item.stat().st_mtime != dest_path.stat().st_mtime or 
                    item.stat().st_size != dest_path.stat().st_size):
                    should_copy = True
            
            if should_copy:
                shutil.copy2(item, dest_path)
    
    return dst

def get_stream_name(folder: Path) -> str:
    """Get stream name from folder."""
    STREAM_NAME_MATCH = "Record Node */experiment*/recording*/continuous/Neuropix-PXI-*.Probe*"
    try:
        stream_path = next(folder.glob(STREAM_NAME_MATCH))
    except StopIteration:
        raise ValueError(f"No stream found in {folder} matching {STREAM_NAME_MATCH}")
    
    parts = stream_path.parts
    record_node = next(part for part in parts if part.startswith("Record Node "))
    record_node_number = record_node.split("Record Node ")[1]
    probe = stream_path.name
    return f"Record Node {record_node_number}#{probe}"



##################################
# RecordingProcessor class
##################################

@dataclass
class RecordingProcessor:
    """Handles path management and processing state for a recording."""
    source_path: Path
    working_dir: Path  # Optional[Path]
    stream_name: str
    is_split_recording: bool = False
    
    def __post_init__(self):
        # self.source_path = Path(self.source_path)
        # If working_dir is None, use the parent directory of the source_path
        # self.working_dir = Path(self.working_dir) if self.working_dir is not None else self.source_path.parent

        self.local_folder = self.working_dir / self.source_path.name
        self.kilosort_folder = self.local_folder / "kilosort4"
        self.analyzer_folder = self.kilosort_folder / "analyser"
    
    def try_load_recording(self) -> Optional[OpenEphysBinaryRecordingExtractor]:
        """Try to load the recording, return None if it fails."""
        try:
            if self.is_split_recording:
                folders = sorted(f for f in self.local_folder.glob("*") if is_timestamp_folder(f))
                if not folders:
                    return None
                return concatenate_recordings([
                    OpenEphysBinaryRecordingExtractor(f, stream_name=self.stream_name) 
                    for f in folders
                ])
            else:
                return OpenEphysBinaryRecordingExtractor(self.local_folder, stream_name=self.stream_name)
        except Exception as e:
            print(f"Failed to load recording in {self.local_folder} (Split: {self.is_split_recording} ): {e}")
            return None
    
    def try_load_sorter(self) -> Optional[si.BaseSorting]:
        """Try to load the sorter results, return None if it fails."""
        try:
            return si.read_kilosort(self.kilosort_folder / "sorter_output", keep_good_only=False)
        except Exception as e:
            print(f"Failed to load sorter from {self.kilosort_folder}: {e}")
            return None
    
    def try_load_analyzer(self) -> Optional[si.SortingAnalyzer]:
        """Try to load the analyzer results, return None if it fails."""
        try:
            return si.load_sorting_analyzer(self.analyzer_folder, format="binary_folder")
        except Exception as e:
            print(f"Failed to load analyzer from {self.analyzer_folder}: {e}")
            return None
    
    def copy_to_working_dir(self) -> Path:
        """Copy recording data to working directory."""
        self.local_folder = copy_folder(self.source_path, self.working_dir)
        return self.local_folder

    def copy_results_to_source_dir(self) -> None:
        """Copy results to source directory, only copying new or modified files.
        
        Will also overwrite files that are:
        - Empty (0 bytes)
        - Corrupted (destination file size is less than 95% of source file size)
        """
        for item in self.local_folder.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(self.local_folder)
                dest_path = self.source_path / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Check if we should overwrite
                should_copy = False
                if not dest_path.exists():
                    should_copy = True
                else:
                    # Check if source is newer
                    if item.stat().st_mtime > dest_path.stat().st_mtime:
                        should_copy = True
                    # Check if destination is empty
                    elif dest_path.stat().st_size == 0:
                        should_copy = True
                    # Check if destination might be corrupted
                    else:
                        source_size = item.stat().st_size
                        dest_size = dest_path.stat().st_size
                        if dest_size < 0.95 * source_size:
                            should_copy = True
                
                if should_copy:
                    shutil.copy2(item, dest_path)



#################################
# Actual processing functions
#################################

def standard_preprocessing(recording_extractor: OpenEphysBinaryRecordingExtractor) -> OpenEphysBinaryRecordingExtractor:
    """Aggregate lab standard preprocessing steps for Neuropixels data."""
    recording = spre.correct_lsb(recording_extractor, verbose=1)
    recording = spre.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = spre.phase_shift(recording)
    bad_channel_ids, _ = spre.detect_bad_channels(recording=recording)
    recording = spre.interpolate_bad_channels(recording, bad_channel_ids=bad_channel_ids)

    if recording.get_channel_groups().max() > 1:
        grouped_recordings = recording.split_by(property='group')
        recgrouplist_hpsf = [spre.highpass_spatial_filter(recording=grouped_recordings[k]) 
                            for k in grouped_recordings.keys()]
        recording_hpsf = aggregate_channels(recgrouplist_hpsf)
    else:
        recording_hpsf = spre.highpass_spatial_filter(recording=recording)

    return recording_hpsf


def run_sorting(recording: OpenEphysBinaryRecordingExtractor, kilosort_folder: Path, overwrite: bool) -> si.BaseSorting:
    sorting = ss.run_sorter(
        sorter_name="kilosort4",
        recording=recording,
        folder=kilosort_folder,
        remove_existing_folder=overwrite,
        n_jobs=N_JOBS,
        verbose=True,
    )
    return scur.remove_excess_spikes(sorting, recording)

def compute_stats(sorting_ks: si.BaseSorting, recording: OpenEphysBinaryRecordingExtractor, sortinganalyzerfolder: Path) -> None:
    analyzer = create_sorting_analyzer(
        sorting=sorting_ks,
        recording=recording,
        folder=sortinganalyzerfolder,
        format="binary_folder",
        sparse=True,
        overwrite=True,
        n_jobs=N_JOBS
    )

    job_kwargs = dict(n_jobs=N_JOBS, chunk_duration="1s", progress_bar=True)
    analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500, **job_kwargs)
    analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0, **job_kwargs)
    analyzer.compute("templates", operators=["average", "median", "std"], **job_kwargs)

    si.compute_noise_levels(analyzer, **job_kwargs)
    si.compute_spike_amplitudes(analyzer, **job_kwargs)

    dqm_params = si.get_default_qm_params()
    qms = analyzer.compute(
        input={
            "principal_components": dict(n_components=3, mode="by_channel_local"),
            "quality_metrics": dict(skip_pc_metrics=False, qm_params=dqm_params)
        }, 
        verbose=True, 
        **job_kwargs
    )
    qms = analyzer.get_extension(extension_name="quality_metrics")
    metrics = qms.get_data()
    assert 'isolation_distance' in metrics.columns
    return metrics


def process_recording(processor: RecordingProcessor, dry_run: bool = False, overwrite: bool = False) -> None:
    """Process a single recording."""
    print(f"\nProcessing {processor.source_path.name} with stream name {processor.stream_name}")
    global_t_start = time.time()
    
    # Step 1: Try to load or copy recording data
    recording = processor.try_load_recording()
    if recording is None:
        if dry_run:
            print("[DRY RUN] Would copy and load recording data")
        else:
            print("Copying recording data...")
            copy_t_start = time.time()
            processor.copy_to_working_dir()
            print(f"Copying took {_n_seconds_to_formatted_time(time.time()-copy_t_start)}")
            
            recording = processor.try_load_recording()
            if recording is None:
                raise ValueError(f"Failed to load recording after copying")
    else:
        print("Loaded existing recording data.")
    
    # Step 2: Try to load or run spike sorting
    sorting = processor.try_load_sorter()
    if sorting is None:
        print("No existing sorting found")
    else:
        print(f"Found existing sorting with {len(sorting.get_unit_ids())} units")

    if sorting is None or overwrite:
        if dry_run:
            print("[DRY RUN] Would run spike sorting here")
        else:
            print("Running spike sorting...")
            sorting_t_start = time.time()
            
            print("Preprocessing recording...")
            recording = standard_preprocessing(recording)
            
            processor.kilosort_folder.mkdir(parents=True, exist_ok=True)
            sorting = run_sorting(
                recording=recording,
                kilosort_folder=processor.kilosort_folder,
                overwrite=overwrite
            )
            print(f"Sorting duration: {_n_seconds_to_formatted_time(time.time()-sorting_t_start)}")
    else:
        print("Using existing sorting results")
    
    # Step 3: Try to load or compute metrics
    analyzer = processor.try_load_analyzer()
    if analyzer is None:
        print("No existing metrics found")
    else:
        print("Found existing metrics")

    if analyzer is None or overwrite:
        if dry_run:
            print("[DRY RUN] Would compute metrics")
        else:
            print("Computing metrics...")
            stats_t_start = time.time()
            metrics = compute_stats(sorting, recording, processor.analyzer_folder)
            print(f"Computed metrics: {metrics.columns.tolist()}")
            print(f"Stats duration: {_n_seconds_to_formatted_time(time.time()-stats_t_start)}")
    else:
        print("Using existing metrics")
    
    if not dry_run:
        print(f"Total duration: {_n_seconds_to_formatted_time(time.time()-global_t_start)}")


#################################
# Search for recordings to process
#################################


def find_recordings_to_process(source_dir: Path, working_dir: Path) -> List[RecordingProcessor]:
    """Find all recordings to process."""
    source_dir = Path(source_dir)
    working_dir = source_dir.parent if working_dir is None else Path(working_dir)
    processors = []

    # Find all timestamp folders up to 3 levels deep using explicit glob patterns for speed
    glob_patterns = ["*-*_*/", "*/*-*_*/", "*/*/*-*_*/"]
    timestamp_folders = sorted(
        folder for pattern in glob_patterns
        for folder in source_dir.glob(pattern)
        if is_timestamp_folder(folder)
    )
    
    print(f"Found {len(timestamp_folders)} timestamp folders to process")
    
    for timestamp_folder in timestamp_folders:
        try:
            stream_name = get_stream_name(timestamp_folder)
            if "split_" in timestamp_folder.parent.name:
                # For split recordings, create processor for parent folder
                parent_folder = timestamp_folder.parent
                if not any(p.source_path == parent_folder for p in processors):
                    processors.append(RecordingProcessor(
                        source_path=parent_folder,
                        working_dir=working_dir,
                        stream_name=stream_name,
                        is_split_recording=True
                    ))
            else:
                processors.append(RecordingProcessor(
                    source_path=timestamp_folder,
                    working_dir=working_dir,
                    stream_name=stream_name
                ))
        except Exception as e:
            print(f"Error processing {timestamp_folder}: {repr(e)} ({type(e)})")
            traceback.print_exc()
            continue
    
    return processors

def main():
    """Example usage of the recording processor."""
    print(EXAMPLE_PATHS)
    # Example with dry run
    
    # Example with real processing
    print("\n=== Real Processing Example ===")
    processors = find_recordings_to_process(
        "/Volumes/SystemsNeuroBiology/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys _recordings/M29",
        working_dir=None,
        # EXAMPLE_PATHS['source_dir'],
        # EXAMPLE_PATHS['working_dir']
    )
    # print(processors)
    for processor in processors:
        print("\n" + "-"*100)
        print(f"Processing {processor.source_path} with stream name {processor.stream_name}")
        process_recording(processor, dry_run=True, overwrite=False)

if __name__ == "__main__":
    main() 