#!/usr/bin/env python3
import os
import shutil
import random
from pathlib import Path

def copy_random_files():
    source_dir = Path("/data/erich/raj/data/test")
    dest_dir = Path("./data/classification")
    
    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all files from source directory
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        return
    
    all_files = [f for f in source_dir.iterdir() if f.is_file()]
    
    if len(all_files) == 0:
        print(f"Error: No files found in {source_dir}")
        return
    
    if len(all_files) < 20:
        print(f"Warning: Only {len(all_files)} files available, copying all of them")
        files_to_copy = all_files
    else:
        files_to_copy = random.sample(all_files, 20)
    
    print(f"Copying {len(files_to_copy)} files from {source_dir} to {dest_dir}")
    
    for file_path in files_to_copy:
        try:
            dest_file = dest_dir / file_path.name
            shutil.copy2(file_path, dest_file)
            print(f"Copied: {file_path.name}")
        except Exception as e:
            print(f"Error copying {file_path.name}: {e}")
    
    print("Copy operation completed")

if __name__ == "__main__":
    copy_random_files()