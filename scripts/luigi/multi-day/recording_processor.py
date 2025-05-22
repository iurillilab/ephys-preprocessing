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

import spikeinterface.sorters as ss
from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor
import spikeinterface.preprocessing as spre
from spikeinterface import aggregate_channels
from spikeinterface.core import concatenate_recordings
from spikeinterface import create_sorting_analyzer
import spikeinterface.full as si
import spikeinterface.curation as scur

# Example paths for testing
EXAMPLE_PATHS = {
    'source_dir': Path("/Volumes/Extreme SSD/test_yadu_rec"),
    'working_dir': Path("/Volumes/Extreme SSD/test_yadu_rec/M29"),
    'test_recording': (
        Path("/Volumes/Extreme SSD/2024-11-13_14-39-11"),
        "Record Node 111#Neuropix-PXI-110.ProbeA"
    )
}

def _n_seconds_to_formatted_time(n_seconds: int) -> str:
    """Convert seconds to formatted time string."""
    n_seconds = int(n_seconds)
    hours, remainder = divmod(n_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def copy_folder(src: Union[str, Path], dst: Union[str, Path]) -> Path:
    """Copy a folder to a new location."""
    src, dst = Path(src), Path(dst)
    dst = dst / src.name
    dst.mkdir(exist_ok=True, parents=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
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

@dataclass
class RecordingProcessor:
    """Handles path management and processing state for a recording."""
    source_path: Path
    working_dir: Path
    stream_name: str
    is_split_recording: bool = False
    dry_run: bool = False
    
    def __post_init__(self):
        self.source_path = Path(self.source_path)
        self.working_dir = Path(self.working_dir)
        self.local_folder = self.working_dir / self.source_path.name
        self.kilosort_folder = self.local_folder / "kilosort4"
        self.analyzer_folder = self.kilosort_folder / "analyser"
    
    def try_load_recording(self) -> Optional[OpenEphysBinaryRecordingExtractor]:
        """Try to load the recording, return None if it fails."""
        try:
            if self.is_split_recording:
                folders = sorted(self.local_folder.glob("*-*_*"))
                if not folders:
                    return None
                return concatenate_recordings([
                    OpenEphysBinaryRecordingExtractor(f, stream_name=self.stream_name) 
                    for f in folders
                ])
            else:
                return OpenEphysBinaryRecordingExtractor(self.local_folder, stream_name=self.stream_name)
        except Exception as e:
            if self.dry_run:
                print(f"[DRY RUN] Failed to load recording: {e}")
            return None
    
    def try_load_sorter(self) -> Optional[si.BaseSorting]:
        """Try to load the sorter results, return None if it fails."""
        try:
            return si.read_sorter_folder(self.kilosort_folder / "0")
        except Exception as e:
            if self.dry_run:
                print(f"[DRY RUN] Failed to load sorter: {e}")
            return None
    
    def try_load_analyzer(self) -> Optional[si.SortingAnalyzer]:
        """Try to load the analyzer results, return None if it fails."""
        try:
            return si.load_sorting_analyzer(self.analyzer_folder, format="binary_folder")
        except Exception as e:
            if self.dry_run:
                print(f"[DRY RUN] Failed to load analyzer: {e}")
            return None
    
    def copy_to_working_dir(self) -> None:
        """Copy recording data to working directory."""
        if self.dry_run:
            print(f"[DRY RUN] Would copy {self.source_path} to {self.working_dir}")
            return
            
        print(f"Copying {self.source_path} to {self.working_dir}")
        start_t = time.time()
        self.local_folder = copy_folder(self.source_path, self.working_dir)
        print(f"Copying took {_n_seconds_to_formatted_time(time.time()-start_t)}")

def process_recording(processor: RecordingProcessor, overwrite: bool = False) -> None:
    """Process a single recording."""
    print(f"\nProcessing {processor.source_path.name} with stream name {processor.stream_name}")
    global_t_start = time.time()
    
    # Step 1: Try to load or copy recording data
    recording = processor.try_load_recording()
    if recording is None:
        if processor.dry_run:
            print("[DRY RUN] Would copy and load recording data")
        else:
            processor.copy_to_working_dir()
            recording = processor.try_load_recording()
            if recording is None:
                raise ValueError(f"Failed to load recording after copying")
    else:
        print("Found existing recording data")
    
    # Step 2: Try to load or run spike sorting
    sorting = processor.try_load_sorter()
    if sorting is None or overwrite:
        if processor.dry_run:
            print("[DRY RUN] Would run spike sorting")
        else:
            print("Running spike sorting...")
            sorting_t_start = time.time()
            recording = standard_preprocessing(recording)
            processor.kilosort_folder.mkdir(parents=True, exist_ok=True)
            
            sorting = ss.run_sorter_by_property(
                sorter_name="kilosort4",
                recording=recording,
                folder=processor.kilosort_folder,
                grouping_property="group",
                n_jobs=14,
                verbose=True
            )
            sorting = scur.remove_excess_spikes(sorting, recording)
            print(f"Sorting duration: {_n_seconds_to_formatted_time(time.time()-sorting_t_start)}")
    else:
        print("Found existing sorting results")
    
    # Step 3: Try to load or compute metrics
    analyzer = processor.try_load_analyzer()
    if analyzer is None or overwrite:
        if processor.dry_run:
            print("[DRY RUN] Would compute metrics")
        else:
            print("Computing metrics...")
            stats_t_start = time.time()
            compute_stats(processor.kilosort_folder, sorting, recording)
            print(f"Stats duration: {_n_seconds_to_formatted_time(time.time()-stats_t_start)}")
    else:
        print("Found existing metrics")
    
    if not processor.dry_run:
        print(f"Total duration: {_n_seconds_to_formatted_time(time.time()-global_t_start)}")

def find_recordings_to_process(source_dir: Path, working_dir: Path, dry_run: bool = False) -> List[RecordingProcessor]:
    """Find all recordings to process."""
    processors = []
    mouse_paths = {f.name: f for f in source_dir.glob("*M[0-9][0-9]")}
    
    for mouse_id, mouse_path in mouse_paths.items():
        print(f"Searching {mouse_id}")
        timestamp_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
        
        timestamp_folders = sorted(
            folder for folder in mouse_path.rglob("*-*_*/")
            if folder.is_dir() and timestamp_pattern.match(folder.name)
        )
        
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
                            is_split_recording=True,
                            dry_run=dry_run
                        ))
                else:
                    processors.append(RecordingProcessor(
                        source_path=timestamp_folder,
                        working_dir=working_dir,
                        stream_name=stream_name,
                        dry_run=dry_run
                    ))
            except Exception as e:
                print(f"Error processing {timestamp_folder}: {repr(e)} ({type(e)})")
                traceback.print_exc()
                continue
    
    return processors

def main():
    """Example usage of the recording processor."""
    # Example with dry run
    print("=== Dry Run Example ===")
    test_processor = RecordingProcessor(
        source_path=EXAMPLE_PATHS['test_recording'][0],
        working_dir=EXAMPLE_PATHS['working_dir'],
        stream_name=EXAMPLE_PATHS['test_recording'][1],
        dry_run=True
    )
    process_recording(test_processor)
    
    # Example with real processing
    print("\n=== Real Processing Example ===")
    processors = find_recordings_to_process(
        EXAMPLE_PATHS['source_dir'],
        EXAMPLE_PATHS['working_dir'],
        dry_run=True
    )
    for processor in processors:
        process_recording(processor)

if __name__ == "__main__":
    main() 