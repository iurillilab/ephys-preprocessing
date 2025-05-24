"""Recording processor for spike sorting pipeline."""
# Before running, run 
# subprocess.run("sudo mount -t drvfs \\\\10.231.128.151\\SystemsNeuroBiology /mnt/nas")

from dataclasses import dataclass
from pathlib import Path
import shutil
import time
import os
import traceback
import re
import subprocess

import spikeinterface.sorters as ss
from spikeinterface.extractors import OpenEphysBinaryRecordingExtractor
import spikeinterface.preprocessing as spre
from spikeinterface import aggregate_channels
from spikeinterface.core import concatenate_recordings
from spikeinterface import create_sorting_analyzer
import spikeinterface.full as si
import spikeinterface.curation as scur

from pprint import pprint
from tqdm import tqdm


N_JOBS = os.cpu_count() // 2

#################################
# Utils
#################################

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

def _n_seconds_to_formatted_time(n_seconds: int) -> str:
    """Convert seconds to formatted time string."""
    n_seconds = int(n_seconds)
    hours, remainder = divmod(n_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def is_timestamp_folder(folder: Path) -> bool:
    """Check if a folder name matches the timestamp pattern."""
    if not folder.is_dir():
        return False
    timestamp_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$")
    if not timestamp_pattern.match(folder.name):
        return False
    try:
        get_stream_name(folder)
    except ValueError:
        return False
    
    return True


def copy_folder(src: str | Path, dst: str | Path, overwrite: bool = True) -> Path:
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
    
    for item in tqdm(list(src.rglob("*")), desc="Copying files", unit="file"):
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
                shutil.copyfile(item, dest_path)  # or copy2 for metadata?
    
    return dst



##################################
# RecordingProcessor class
##################################

@dataclass
class RecordingProcessor:
    """Handles path management and processing state for a recording."""
    source_path: Path
    root_working_dir: Path | None  # Optional[Path]
    stream_name: str
    is_split_recording: bool = False

    def __post_init__(self):
        self.source_path = Path(self.source_path)
        self.root_working_dir = Path(self.root_working_dir) if self.root_working_dir is not None else None
        
        # Initialize all paths
        self.processed_in_different_folder = self.root_working_dir is not None
        self.local_folder = self.root_working_dir / self.source_path.name if self.processed_in_different_folder else self.source_path
        self.kilosort_folder = self.local_folder / "kilosort4"
        self.analyzer_folder = self.kilosort_folder / "analyser"

    def try_load_recording(self) -> OpenEphysBinaryRecordingExtractor | None:
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
    
    def try_load_sorter(self) -> si.BaseSorting | None:
        """Try to load the sorter results, return None if it fails."""
        try:
            return si.read_kilosort(self.kilosort_folder / "sorter_output", keep_good_only=False)
        except Exception as e:
            print(f"Failed to load sorter from {self.kilosort_folder}: {e}")
            return None
    
    def try_load_analyzer(self) -> si.SortingAnalyzer | None:
        """Try to load the analyzer results, return None if it fails."""
        try:
            return si.load_sorting_analyzer(self.analyzer_folder, format="binary_folder")
        except Exception as e:
            print(f"Failed to load analyzer from {self.analyzer_folder}: {e}")
            return None
    
    def copy_to_working_dir(self) -> Path:
        """Copy recording data to working directory."""
        if self.processed_in_different_folder:
            self.local_folder = copy_folder(self.source_path, self.root_working_dir)
        return self.local_folder

    def copy_results_to_source_dir(self, overwrite: bool = True) -> None:
        """Copy results to source directory, only copying new or modified files."""
        if self.processed_in_different_folder:
            copy_folder(self.local_folder, self.source_path.parent, overwrite=overwrite)
        else:
            print(f"Would copy results to source directory, but we are in the same folder: {self.local_folder}")

    def remove_sorting_results(self) -> None:
        """Remove sorting results."""
        shutil.rmtree(self.kilosort_folder)

    def remove_analyzer_results(self) -> None:
        """Remove analyzer results."""
        shutil.rmtree(self.analyzer_folder)

    def cleanup_working_dir(self) -> None:
        """Cleanup working directory."""
        if self.processed_in_different_folder:
            shutil.rmtree(self.local_folder)
        else:
            print(f"Would cleanup working directory, but we are in the same folder: {self.local_folder}")


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
        remove_existing_folder=True,
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
    print(job_kwargs)
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


def process_recording(processor: RecordingProcessor, dry_run: bool = False, overwrite: bool = False, copy_upstream: bool = True, cleanup: bool = True) -> None:
    """Process a single recording."""
    print(f"\nProcessing {processor.source_path} with stream name {processor.stream_name}")
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
            run_sorting(
                recording=recording,
                kilosort_folder=processor.kilosort_folder,
                overwrite=overwrite
            )
            sorting = processor.try_load_sorter()
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

            analyzer = processor.try_load_analyzer()
    else:
        print("Using existing metrics")

    if copy_upstream:
        if not dry_run:
            processor.copy_results_to_source_dir()
        else:
            print("[DRY RUN] Would copy results to source directory")

    if cleanup:
        if not dry_run:
            processor.cleanup_working_dir()
        else:
            print("[DRY RUN] Would cleanup working directory")
    
    # if not dry_run:
    print(f"Total duration: {_n_seconds_to_formatted_time(time.time()-global_t_start)}")


#################################
# Search for recordings to process
#################################


def find_recordings_to_process(source_dir: Path, root_working_dir: Path) -> list[RecordingProcessor]:
    """Find all recordings to process."""
    source_dir = Path(source_dir)
    root_working_dir = Path(root_working_dir) if root_working_dir is not None else None
    processors = []

    # Find all timestamp folders up to 3 levels deep using explicit glob patterns for speed
    glob_patterns = ["*-*_*/", "*/*-*_*/", "*/*/*-*_*/"]
    timestamp_folders = sorted(
        folder for pattern in glob_patterns
        for folder in source_dir.glob(pattern)
        if is_timestamp_folder(folder)
    )
    
    print(f"Found {len(timestamp_folders)} timestamp folders to process: ")
    pprint(timestamp_folders)
    
    for timestamp_folder in timestamp_folders:
        try:
            stream_name = get_stream_name(timestamp_folder)
            if "split_" in timestamp_folder.parent.name:
                # For split recordings, create processor for parent folder
                parent_folder = timestamp_folder.parent
                if not any(p.source_path == parent_folder for p in processors):
                    processors.append(RecordingProcessor(
                        source_path=parent_folder,
                        root_working_dir=root_working_dir,
                        stream_name=stream_name,
                        is_split_recording=True
                    ))
            else:
                processors.append(RecordingProcessor(
                    source_path=timestamp_folder,
                    root_working_dir=root_working_dir,
                    stream_name=stream_name
                ))
        except Exception as e:
            print(f"Error processing {timestamp_folder}: {repr(e)} ({type(e)})")
            traceback.print_exc()
            continue
    
    return processors

@dataclass
class RunArgs:
    source_dir: Path
    root_working_dir: Path
    dry_run: bool
    overwrite: bool
    copy_upstream: bool
    cleanup: bool

luigi_macbook_kwargs = RunArgs(
    source_dir=Path("/Users/vigji/Desktop/short_recording_oneshank"),  # Path("/Volumes/Extreme SSD/mouse_data_electrode_tips"),
    root_working_dir=Path("/Users/vigji/Desktop/temp_dir"),
    dry_run=False,
    overwrite=False,
    copy_upstream=True,
    cleanup=True
)

wsl_run_kwargs = RunArgs(
    source_dir=Path("/mnt/nas/SNeuroBiology_shared/P07_PREY_HUNTING_YE/e01_ephys _recordings"),
    root_working_dir=Path("/mnt/d/temp_processing"),
    dry_run=False,
    overwrite=False,
    copy_upstream=True,
    cleanup=True
)


def main():
    """Example usage of the recording processor."""
    # run_kwargs = luigi_macbook_kwargs
    run_kwargs = wsl_run_kwargs
    
    # Example with real processing
    print("\n=== Real Processing Example ===")
    processors = find_recordings_to_process(run_kwargs.source_dir, run_kwargs.root_working_dir)
    pprint(processors)
    first_split = next((p for p in processors if p.is_split_recording), None)
    first_not_split = next((p for p in processors if not p.is_split_recording), None)

    for processor in processors:
        print("\n" + "-"*100 + "\n")
        process_recording(processor, dry_run=run_kwargs.dry_run, overwrite=run_kwargs.overwrite, copy_upstream=run_kwargs.copy_upstream, cleanup=run_kwargs.cleanup)

    
    # for processor in [p for p in [first_split, first_not_split] if p is not None]:
    #     print("\n" + "-"*100 + "\n")
    #     process_recording(processor, dry_run=run_kwargs.dry_run, overwrite=run_kwargs.overwrite, copy_upstream=run_kwargs.copy_upstream, cleanup=run_kwargs.cleanup)

if __name__ == "__main__":
    main() 