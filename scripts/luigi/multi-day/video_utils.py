"""Script to concatenate videos for all chunks (object, cricket, roach) for one session.
DEPRECATED!
"""


import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def check_video_health(video_path):
    """Check if a video file can be opened and read by OpenCV.
    
    Parameters
    ----------
    video_path : Path or str
        Path to the video file
        
    Returns
    -------
    dict
        Dictionary with health check results
    """
    video_path = Path(video_path)
    result = {
        'path': video_path,
        'exists': video_path.exists(),
        'size_mb': 0,
        'can_open': False,
        'frame_count': 0,
        'fps': 0,
        'duration_sec': 0,
        'codec': None,
        'error': None
    }
    
    if not video_path.exists():
        result['error'] = 'File does not exist'
        return result
    
    result['size_mb'] = video_path.stat().st_size / (1024 * 1024)
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        result['can_open'] = cap.isOpened()
        
        if result['can_open']:
            result['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            result['fps'] = cap.get(cv2.CAP_PROP_FPS)
            if result['fps'] > 0:
                result['duration_sec'] = result['frame_count'] / result['fps']
            result['codec'] = cap.get(cv2.CAP_PROP_FOURCC)
        else:
            result['error'] = 'Cannot open with OpenCV'
        
        cap.release()
    except Exception as e:
        result['error'] = str(e)
    
    return result


def find_and_remove_dummy_videos(videos_folder, min_size_mb=1, dry_run=True):
    """Find and optionally remove tiny video files that are likely dummy/corrupted.
    
    Parameters
    ----------
    videos_folder : Path or str
        Path to the videos folder
    min_size_mb : float
        Minimum size in MB for a valid video file
    dry_run : bool
        If True, only print what would be removed without actually removing
        
    Returns
    -------
    list
        List of files that were found (and removed if dry_run=False)
    """
    videos_folder = Path(videos_folder)
    dummy_files = []
    
    for video_file in videos_folder.rglob("*.mp4"):
        size_mb = video_file.stat().st_size / (1024 * 1024)
        if size_mb < min_size_mb:
            dummy_files.append(video_file)
            if dry_run:
                print(f"Would remove dummy file: {video_file} ({size_mb:.3f} MB)")
            else:
                print(f"Removing dummy file: {video_file} ({size_mb:.3f} MB)")
                video_file.unlink()
    
    return dummy_files


def get_video_interface(session_path):
    """Get the video interface for a session."""
    pass

def concatenate_central_videos(session_folder):
    """Concatenate central videos from all session subfolders using ffmpeg.
    
    Parameters
    ----------
    session_folder : str or Path
        Path to the session folder containing the videos subfolder
        
    Returns
    -------
    Path
        Path to the concatenated video file
    """
    session_folder = Path(session_folder)
    videos_folder = session_folder / "videos"
    
    if not videos_folder.exists():
        raise FileNotFoundError(f"Videos folder not found: {videos_folder}")
    
    # Find all central videos in session subfolders
    central_videos = []
    for session_subfolder in videos_folder.iterdir():
        if session_subfolder.is_dir() and session_subfolder.name not in [".", ".."]:
            # Look for cropped subfolder
            cropped_folders = list(session_subfolder.glob("*cropped*"))
            if cropped_folders:
                # If multiple cropped folders, take the latest one by modification time
                if len(cropped_folders) > 1:
                    cropped_folders.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    print(f"Multiple cropped folders found in {session_subfolder.name}, using latest: {cropped_folders[0].name}")
                
                # Look for central video in the selected cropped folder
                # Try different patterns for central videos, prioritizing larger files
                central_patterns = ["multicam_video_*central*.mp4", "*central*.avi.mp4", "*central*.mp4"]
                central_files = []
                
                for pattern in central_patterns:
                    found_files = list(cropped_folders[0].glob(pattern))
                    if found_files:
                        central_files.extend(found_files)
                
                if central_files:
                    # Filter out tiny files (likely corrupted or dummy files)
                    valid_files = [f for f in central_files if f.stat().st_size > 1000000]  # > 1MB
                    if valid_files:
                        # Sort by file size (largest first) to get the main video
                        valid_files.sort(key=lambda x: x.stat().st_size, reverse=True)
                        central_videos.append(valid_files[0])
                        if len(valid_files) > 1:
                            print(f"Multiple central videos found in {session_subfolder.name}, using largest: {valid_files[0].name}")
                    else:
                        print(f"Warning: Found central video files in {session_subfolder.name} but all are too small (< 1MB)")
    
    if not central_videos:
        raise FileNotFoundError("No central videos found in the expected structure")
    
    # Sort videos to ensure consistent order
    central_videos.sort(key=lambda x: x.parent.parent.name)
    
    print(f"Found {len(central_videos)} central videos to concatenate:")
    for video in central_videos:
        print(f"  - {video.parent.parent.name}: {video.name}")
    
    # Create temporary file list for ffmpeg
    output_path = videos_folder / "concatenated_central.mp4"
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file_list = Path(f.name)
        for video in central_videos:
            f.write(f"file '{video.absolute()}'\n")
    
    try:
        # Run ffmpeg concatenation
        cmd = [
            'ffmpeg', 
            '-f', 'concat',
            '-safe', '0',
            '-i', str(temp_file_list),
            '-c', 'copy',
            '-y',  # overwrite output file if it exists
            str(output_path)
        ]
        
        print(f"Running ffmpeg concatenation...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with error: {result.stderr}")
        
        print(f"Successfully created concatenated video: {output_path}")
        return output_path
        
    finally:
        # Clean up temporary file
        temp_file_list.unlink()


if __name__ == "__main__":
    # Example usage - replace with your actual session path
    import sys
    from tqdm import tqdm
    if len(sys.argv) > 1:
        main_path = Path(sys.argv[1])
    else:
        main_path = Path("/Users/vigji/Desktop/07_PREY_HUNTING_YE/e01_ephys _recordings")
        print(f"No path provided, using example: {main_path}")

    all_sessions = list(main_path.glob("M*_WT*/*/*")) 
    for session_path in tqdm(all_sessions):
        print(session_path)
        try:
            output_path = concatenate_central_videos(session_path)
        except Exception as e:
            print(f"Error: {e}")
    


