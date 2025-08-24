#!/usr/bin/env python3
import json
import os
import glob
import shutil
from pathlib import Path

def analyze_json_file(file_path):
    """Analyze a single JSON file to determine if it's complete and calculate average reward."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Check if file is truncated (ends with incomplete JSON)
        if not content.strip().endswith(']'):
            # print("incomplete file")
            return None, None, None, None, None
        
        data = json.loads(content)
        
        if not isinstance(data, list) or len(data) == 0:
            # print("not a list")
            return None, None, None, None, None
        
        # Count tasks
        task_count = len(data)
        
        # Get task IDs to determine range
        task_ids = [task.get('task_id', -1) for task in data if 'task_id' in task]
        if not task_ids:
            # print("no task ids")
            return None, None, None, None, None
        
        min_task_id = min(task_ids)
        max_task_id = max(task_ids)
        unique_task_ids = len(set(task_ids))
        
        # if len(task_ids) != 50 and len(task_ids) != 115:
        #     print("75",unique_task_ids)
        
        # Determine category from filename or metadata
        category = None
        filename = os.path.basename(file_path)
        
        if 'airline' in filename:
            category = 'airline'
            expected_tasks = 50
        elif 'retail' in filename:
            category = 'retail'
            expected_tasks = 115
        else:
            # Try to get category from metadata in first task
            if data and 'metadata' in data[0] and data[0]['metadata']:
                metadata = data[0]['metadata']
                if 'environment' in metadata:
                    category = metadata['environment']
                    if category == 'airline':
                        expected_tasks = 50
                    elif category == 'retail':
                        expected_tasks = 115
                    else:
                        return None, None, None, None, None
                else:
                    return None, None, None, None, None
            else:
                # If no metadata, try to infer from task count and range
                if unique_task_ids == 50 and max_task_id == 49:
                    category = 'airline'
                    expected_tasks = 50
                elif unique_task_ids == 115 and max_task_id == 114:
                    category = 'retail'
                    expected_tasks = 115
                else:
                    return None, None, None, None, None
        
        # Check if file is complete
        is_complete = unique_task_ids == expected_tasks
        
        # Calculate average reward
        total_reward = 0
        valid_tasks = 0
        
        for task in data:
            if 'reward' in task and isinstance(task['reward'], (int, float)):
                total_reward += task['reward']
                valid_tasks += 1
        
        avg_reward = total_reward / valid_tasks if valid_tasks > 0 else 0
        
        # Check user model from filename
        user_model = None
        if 'gpt-4o-20240806' in filename:
            user_model = 'gpt-4o-20240806'
        
        return category, file_path, is_complete, avg_reward, user_model
        
    except Exception as e:
        # Silently skip files with errors
        print(e)
        return None, None, None, None, None

def main():
    results_dir = "."
    
    # Create results_trials directory if it doesn't exist
    trials_dir = "../results_trials"
    os.makedirs(trials_dir, exist_ok=True)
    
    # Find all JSON files in current directory and subdirectories
    json_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    print(f"Found {len(json_files)} JSON files")
    print("=" * 80)
    
    complete_files = []
    filtered_files = []
    
    for file_path in json_files:
        # if "gpt-4o-20240806-llm_0819111930" in file_path and "R1" in file_path: 
        #     print("75",file_path)
        #     print(analyze_json_file(file_path))
        category, full_path, is_complete, avg_reward, user_model = analyze_json_file(file_path)
        
        if category and is_complete and avg_reward > 0 and user_model == 'gpt-4o-20240806':
            complete_files.append((category, full_path, avg_reward))
            print(f"Category: {category}, Filename: {full_path}, Avg Reward: {avg_reward:.4f}, Model: {user_model}")
        else:
            # This file is filtered out, add to list for moving
            filtered_files.append(file_path)
    
    # Move filtered files to results_trials directory
    moved_count = 0
    for file_path in filtered_files:
        try:
            # Get relative path from current directory to preserve subdirectory structure
            rel_path = os.path.relpath(file_path, results_dir)
            # Create destination path preserving subdirectory structure
            dest_path = os.path.join(trials_dir, rel_path)
            
            # Create subdirectories if they don't exist
            dest_dir = os.path.dirname(dest_path)
            if dest_dir:
                os.makedirs(dest_dir, exist_ok=True)
            
            # If destination file already exists, add a number suffix
            counter = 1
            original_dest_path = dest_path
            while os.path.exists(dest_path):
                name, ext = os.path.splitext(original_dest_path)
                dest_path = f"{name}_{counter}{ext}"
                counter += 1
            
            # Move the file
            shutil.move(file_path, dest_path)
            moved_count += 1
            print(f"Moved: {file_path} -> {dest_path}")
        except Exception as e:
            print(f"Error moving {file_path}: {e}")
    
    print("=" * 80)
    print(f"Total complete files found: {len(complete_files)}")
    print(f"Total filtered files moved: {moved_count}")
    
    # Summary by category
    airline_files = [f for f in complete_files if f[0] == 'airline']
    retail_files = [f for f in complete_files if f[0] == 'retail']
    
    print(f"Complete airline files: {len(airline_files)}")
    print(f"Complete retail files: {len(retail_files)}")

if __name__ == "__main__":
    main() 