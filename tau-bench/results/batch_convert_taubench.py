#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

def create_directory_if_not_exists(directory_path):
    """Create directory if it doesn't exist."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    print(f"Directory ensured: {directory_path}")

# def run_conversion_for_model(model_name, base_dir="~/sgl-eval-data/agent/taubench"):
def run_conversion_for_model(model_name, base_dir="~/agenthard/downloaded_datasets/tau-bench-evaluation"):
    """Run conversion for a specific model."""
    # Expand the base directory path
    base_dir = os.path.expanduser(base_dir)
    
    # Create model-specific directory
    model_dir = os.path.join(base_dir, model_name)
    # create_directory_if_not_exists(model_dir)
    
    # Construct the pattern and output path
    pattern = f"*{model_name}*user*-gpt-4o-20240806*.json"
    # "tool-calling-airline-DeepSeek-V3-0324-0.0_range_0--1_user-openai-gpt-4o-20240806-llm_0813191457.json"
    # output_path = os.path.join(model_dir, "output_rs0.jsonl")
    output_path = os.path.join(base_dir, f"{model_name}.jsonl")
    # Run the conversion command
    cmd = [
        sys.executable, "convert_taubench.py", 
        "--airline-retail", pattern, 
        output_path
    ]
    
    print(f"\nProcessing model: {model_name}")
    print(f"Pattern: {pattern}")
    print(f"Output: {output_path}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if result.returncode == 0:
            print(f"✅ Successfully processed {model_name}")
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
        else:
            print(f"❌ Failed to process {model_name}")
            if result.stderr:
                print(f"Error: {result.stderr.strip()}")
            if result.stdout:
                print(f"Output: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Exception while processing {model_name}: {e}")

def main():
    # List of model names to process
    model_names = [
        # "gpt-4o-20240806",
        # "gpt-4o-mini",
        # "gpt-4.1",
        # "claude-4-sonnet-thinking-on-10k",
        # "claude-4-sonnet-thinking-off",
        # "Qwen3-8B",
        # "Qwen3-32B",
        # "Qwen3-235B-A22B-Thinking-2507-FP8",
        # "Qwen3-235B-A22B-FP8",
        # "Qwen3-235B-A22B-Instruct-2507-FP8",
        # "DeepSeek-R1-0528",
        # "DeepSeek-V3-0324",
        # "Kimi-K2-Instruct",
        # "grok-4",
        # "o4-mini-high",
        # "o3-high",
        "gpt-4.1-mini",
        "gpt-5",
        "claude-4-opus-thinking-on-10k",
        "claude-4-opus-thinking-off",
        "Qwen3-Coder-480B-A35B-Instruct-FP8",
        "DeepSeek-V3.1-thinking-off",
        "DeepSeek-V3.1-thinking-on",
    ]
    
    print(f"Starting batch conversion for {len(model_names)} models...")
    print("=" * 60)
    
    for i, model_name in enumerate(model_names, 1):
        print(f"\n[{i}/{len(model_names)}] Processing: {model_name}")
        print("-" * 40)
        run_conversion_for_model(model_name)
    
    print("\n" + "=" * 60)
    print("Batch conversion completed!")

if __name__ == "__main__":
    main() 