#!/usr/bin/env python3
import os
import shutil
from pathlib import Path

def flatten_results():
    """Flatten the results directory structure by moving JSON files from subdirs to main dir."""
    results_dir = Path(".")
    
    # Get all subdirectories (excluding the script itself and other files)
    subdirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("tool-calling-")]
    
    print(f"Found {len(subdirs)} subdirectories to process")
    
    for subdir in subdirs:
        print(f"Processing {subdir.name}...")
        
        # Find all JSON files in the subdirectory
        json_files = list(subdir.glob("*.json"))
        
        for json_file in json_files:
            # Create new filename: subdir-filename.json
            new_filename = f"{subdir.name}-{json_file.name}"
            new_path = results_dir / new_filename
            
            # Check if destination file already exists
            if new_path.exists():
                print(f"  Warning: {new_filename} already exists, skipping {json_file.name}")
                continue
            
            # Move the file
            try:
                shutil.move(str(json_file), str(new_path))
                print(f"  Moved: {json_file.name} -> {new_filename}")
            except Exception as e:
                print(f"  Error moving {json_file.name}: {e}")
        
        # Remove the now-empty subdirectory
        try:
            subdir.rmdir()
            print(f"  Removed empty directory: {subdir.name}")
        except Exception as e:
            print(f"  Warning: Could not remove directory {subdir.name}: {e}")

if __name__ == "__main__":
    flatten_results()
    print("Flattening complete!") 