#!/usr/bin/env python3
import random
import shutil
from pathlib import Path


def copy_random_files():
    source_dir = Path("/data/erich/raj/data/test/")
    dest_dir = Path("../../data/classification")

    # Create destination directory if it doesn't exist
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if source directory exists
    if not source_dir.exists():
        print(f"Error: Source directory {source_dir} does not exist")
        return

    # Get all subdirectories in the source directory
    subdirs = [d for d in source_dir.iterdir() if d.is_dir()]

    if len(subdirs) == 0:
        print(f"Error: No subdirectories found in {source_dir}")
        return

    print(f"Found {len(subdirs)} subdirectories in {source_dir}")

    total_copied = 0

    # Iterate through each subdirectory
    for subdir in subdirs:
        print(f"Processing folder: {subdir.name}")

        # Get all files in this subdirectory
        files_in_subdir = [f for f in subdir.iterdir() if f.is_file()]

        if len(files_in_subdir) == 0:
            print(f"  Warning: No files found in {subdir.name}")
            continue

        # Select 2 random files (or all files if less than 2)
        files_to_copy = random.sample(files_in_subdir, min(2, len(files_in_subdir)))

        print(f"  Copying {len(files_to_copy)} files from {subdir.name}")

        # Create corresponding subdirectory in destination
        dest_subdir = dest_dir / subdir.name
        dest_subdir.mkdir(parents=True, exist_ok=True)
        
        # Copy selected files to the appropriate subfolder
        for file_path in files_to_copy:
            try:
                dest_file = dest_subdir / file_path.name
                shutil.copy2(file_path, dest_file)
                print(f"  Copied: {file_path.name} -> {subdir.name}/")
                total_copied += 1
            except Exception as e:
                print(f"  Error copying {file_path.name}: {e}")

    print(f"Copy operation completed. Total files copied: {total_copied}")


if __name__ == "__main__":
    copy_random_files()
