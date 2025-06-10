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
                central_files = list(cropped_folders[0].glob("*central*.mp4"))
                if central_files:
                    central_videos.append(central_files[0])
    
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
    


