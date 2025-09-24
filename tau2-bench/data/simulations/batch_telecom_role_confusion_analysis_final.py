#!/usr/bin/env python3
"""
Batch script to analyze role confusion patterns in all telecom simulation files.
Generates a CSV with task_id x model matrix where 1 = role confusion, 0 = no role confusion.
"""

import json
import os
import re
import csv
import glob
from typing import List, Dict, Any, Set
from collections import defaultdict

def detect_telecom_role_confusion(content: str) -> bool:
    """Detect if user message shows role confusion by using agent-like language"""
    if not content or not isinstance(content, str):
        return False
        
    content_lower = content.lower()
    
    # Patterns that indicate role confusion - user speaking as if they are the agent
    confusion_patterns = [
        r'\byour\s+(phone|device|computer|laptop|system|account|order|reservation|flight|booking|service|plan|bill|account|number)\b',
    ]
    
    for pattern in confusion_patterns:
        if re.search(pattern, content_lower):
            return True
    
    return False

def extract_model_name_from_filename(filename: str) -> str:
    """Extract model name from the filename"""
    # Pattern: timestamp_telecom_llm_agent_MODELNAME_user_simulator_...
    # Remove the .json extension first
    base_name = filename.replace('.json', '')
    parts = base_name.split('_')
    
    # Find the pattern: telecom_llm_agent_MODELNAME_user_simulator
    try:
        telecom_idx = parts.index('telecom')
        if telecom_idx + 4 < len(parts) and parts[telecom_idx + 1] == 'llm' and parts[telecom_idx + 2] == 'agent':
            # The model name is the part right after 'agent' and before 'user'
            if parts[telecom_idx + 4] == 'user':
                return parts[telecom_idx + 3]  # This is the model name
    except (ValueError, IndexError):
        pass
    
    return 'unknown'

def analyze_telecom_file(filepath: str) -> Dict[str, Any]:
    """Analyze a single telecom simulation file for role confusion"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract model name from filename
        filename = os.path.basename(filepath)
        model_name = extract_model_name_from_filename(filename)
        
        # Check if file has simulations
        if 'simulations' not in data or not data['simulations']:
            print(f"No simulations found in {filename}")
            return {'model': model_name, 'task_confusion': {}, 'total_simulations': 0}
        
        # Check if this is a telecom file
        if 'environment_info' in data and 'domain_name' in data['environment_info']:
            domain = data['environment_info']['domain_name']
            if domain != 'telecom':
                print(f"Warning: {filename} is for domain '{domain}', not telecom")
                return {'model': model_name, 'task_confusion': {}, 'total_simulations': 0}
        
        print(f"Analyzing {len(data['simulations'])} simulations in {filename} (model: {model_name})")
        
        # Track role confusion per task
        task_confusion = defaultdict(bool)  # task_id -> has_confusion
        
        for sim_idx, simulation in enumerate(data['simulations']):
            if 'messages' not in simulation:
                continue
                
            task_id = simulation.get('task_id', 'unknown')
            
            # Check if this simulation has role confusion
            has_confusion = False
            for msg_idx, message in enumerate(simulation['messages']):
                if message.get('role') == 'user' and 'content' in message:
                    content = message['content']
                    if detect_telecom_role_confusion(content):
                        has_confusion = True
                        break
            
            # If this task has any confusion, mark it
            if has_confusion:
                task_confusion[task_id] = True
        
        return {
            'model': model_name,
            'task_confusion': dict(task_confusion),
            'total_simulations': len(data['simulations']),
            'filename': filename
        }
    
    except Exception as e:
        print(f'Error processing {filepath}: {e}')
        return {'model': 'error', 'task_confusion': {}, 'total_simulations': 0, 'filename': os.path.basename(filepath)}

def main():
    # Find all telecom JSON files
    telecom_files = glob.glob('*_telecom_llm_agent_*_user_simulator_*.json')
    
    if not telecom_files:
        print("No telecom simulation files found in current directory")
        return
    
    print(f"Found {len(telecom_files)} telecom simulation files")
    print("=" * 80)
    
    # Test model name extraction on a few files
    print("Testing model name extraction:")
    for i, filepath in enumerate(telecom_files[:5]):
        filename = os.path.basename(filepath)
        model_name = extract_model_name_from_filename(filename)
        print(f"  {filename} -> {model_name}")
    print()
    
    # Analyze all files
    all_results = []
    all_task_ids = set()
    
    for filepath in telecom_files:
        result = analyze_telecom_file(filepath)
        all_results.append(result)
        all_task_ids.update(result['task_confusion'].keys())
    
    # Convert to sorted lists for consistent ordering
    all_task_ids = sorted(list(all_task_ids))
    models = [r['model'] for r in all_results if r['model'] != 'error']
    models = sorted(list(set(models)))  # Remove duplicates and sort
    
    print(f"\nFound {len(all_task_ids)} unique task IDs")
    print(f"Found {len(models)} unique models: {models}")
    
    # Create CSV
    csv_filename = 'telecom_role_confusion_matrix.csv'
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        header = ['task_id'] + models
        writer.writerow(header)
        
        # Write data rows
        for task_id in all_task_ids:
            row = [task_id]
            for model in models:
                # Find if this task has confusion for this model
                has_confusion = False
                for result in all_results:
                    if result['model'] == model and task_id in result['task_confusion']:
                        has_confusion = result['task_confusion'][task_id]
                        break
                
                row.append(1 if has_confusion else 0)
            writer.writerow(row)
    
    print(f"\nCSV file created: {csv_filename}")
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("=" * 80)
    
    for model in models:
        confusion_count = 0
        total_tasks = 0
        
        for result in all_results:
            if result['model'] == model:
                total_tasks += len(result['task_confusion'])
                confusion_count += sum(1 for has_confusion in result['task_confusion'].values() if has_confusion)
        
        if total_tasks > 0:
            confusion_rate = confusion_count / total_tasks * 100
            print(f"{model}: {confusion_count}/{total_tasks} tasks with role confusion ({confusion_rate:.1f}%)")

if __name__ == '__main__':
    main()
