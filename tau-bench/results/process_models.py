#!/usr/bin/env python3
import os
import glob
import subprocess
import sys
from pathlib import Path

def check_files_exist(model_name, directory="."):
    """Check if both airline and retail files exist for a model."""
    airline_pattern = f"*airline*{model_name}*.json"
    retail_pattern = f"*retail*{model_name}*.json"
    
    airline_files = glob.glob(os.path.join(directory, airline_pattern))
    retail_files = glob.glob(os.path.join(directory, retail_pattern))
    
    return airline_files, retail_files

def convert_model_files(model_name, output_dir="."):
    """Convert airline and retail files for a specific model."""
    print(f"\nProcessing model: {model_name}")
    
    # Check if files exist
    airline_files, retail_files = check_files_exist(model_name)
    
    if not airline_files:
        print(f"  ❌ No airline files found for {model_name}")
        return 0
    
    if not retail_files:
        print(f"  ❌ No retail files found for {model_name}")
        return 0
    
    print(f"  ✅ Found {len(airline_files)} airline file(s): {[os.path.basename(f) for f in airline_files]}")
    print(f"  ✅ Found {len(retail_files)} retail file(s): {[os.path.basename(f) for f in retail_files]}")
    
    # Create output filename
    output_filename = f"{model_name}.jsonl"
    output_path = os.path.join(output_dir, output_filename)
    
    # Run conversion
    pattern = f"*{model_name}*.json"
    cmd = [
        sys.executable, "convert_taubench.py", 
        "--airline-retail", pattern, 
        output_path
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print(f"  ✅ Successfully created: {output_filename}")
            if result.stdout:
                print(f"    {result.stdout.strip()}")
            
            # Count lines in output file
            try:
                with open(output_path, 'r') as f:
                    line_count = sum(1 for _ in f)
                print(f"    📊 Total JSONL entries: {line_count}")
                return line_count
            except FileNotFoundError:
                print(f"    ❌ Output file not found: {output_path}")
                return 0
        else:
            print(f"  ❌ Failed to convert {model_name}")
            if result.stderr:
                print(f"    Error: {result.stderr.strip()}")
            return 0
    except Exception as e:
        print(f"  ❌ Exception while processing {model_name}: {e}")
        return 0

def main():
    model_names = [
        "grok-4",
        "Kimi-K2-Instruct", 
        "Qwen3-8B",
        "Qwen3-32B",
        "Qwen3-235B-A22B-Thinking-2507-FP8",
        "Qwen3-235B-A22B-FP8",
        "Qwen3-235B-A22B-Instruct-2507-FP8",
        "o4-mini-high",
        "o3-high",
        "gpt-4o-20240806",
        "gpt-4o-mini",
        "gpt-4.1",
        "DeepSeek-V3-0324",
        "DeepSeek-R1-0528",
        "claude-4-sonnet-thinking-on-10k",
        "claude-4-sonnet-thinking-off"
    ]
    
    print("Taubench to SGLang JSONL Conversion")
    print("=" * 50)
    print(f"Processing {len(model_names)} models...")
    
    total_jsonl_entries = 0
    successful_models = 0
    
    for i, model_name in enumerate(model_names, 1):
        print(f"\n[{i}/{len(model_names)}] Processing: {model_name}")
        print("-" * 40)
        
        count = convert_model_files(model_name)
        if count > 0:
            total_jsonl_entries += count
            successful_models += 1
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Successfully processed: {successful_models}/{len(model_names)} models")
    print(f"Total JSONL entries: {total_jsonl_entries}")
    
    if successful_models > 0:
        avg_per_model = total_jsonl_entries / successful_models
        print(f"Average entries per model: {avg_per_model:.1f}")
    
    print("\nConversion completed!")

if __name__ == "__main__":
    main() 