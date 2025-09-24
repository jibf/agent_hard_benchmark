#!/usr/bin/env python3
"""
Example script demonstrating multi-GPU embedding usage in the benchmark filtering pipeline.

This script shows how to use the new multi-GPU support for embedding models.
"""

import subprocess
import sys
import torch

def run_pipeline_with_multi_gpu():
    """Example of running the pipeline with multi-GPU embedding support."""
    
    # Check if multiple GPUs are available
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    if gpu_count < 2:
        print("Warning: Multi-GPU embedding requires at least 2 GPUs")
        print("Running with single GPU instead...")
        use_multi_gpu = False
    else:
        use_multi_gpu = True
    
    # Base command
    cmd = [
        "python", "main.py",
        "--target-benchmark", "complex-func-bench",  # Example benchmark
        "--embedding-model", "Qwen/Qwen3-Embedding-8B",
        "--embedding-batch-size", "16",  # Larger batch size for multi-GPU
    ]
    
    if use_multi_gpu:
        # Add multi-GPU arguments
        cmd.extend([
            "--use-multi-gpu-embedding",
            "--embedding-device-map", "auto",  # or "balanced", "balanced_low_0"
        ])
        print("Running with multi-GPU embedding support...")
    else:
        print("Running with single GPU embedding...")
    
    # Add other useful arguments
    cmd.extend([
        "--skip-visualization",  # Skip visualization for faster testing
        "--llm-filter-mode", "specific",
    ])
    
    print(f"Command: {' '.join(cmd)}")
    
    # Run the pipeline
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Pipeline completed successfully!")
        print("STDOUT:", result.stdout[-500:])  # Last 500 chars
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed with error: {e}")
        print("STDERR:", e.stderr[-500:])  # Last 500 chars
        return False
    
    return True

def show_device_map_examples():
    """Show examples of different device_map configurations."""
    
    print("\n" + "="*60)
    print("DEVICE MAP EXAMPLES")
    print("="*60)
    
    examples = [
        {
            "name": "Auto (Recommended)",
            "value": "auto",
            "description": "Automatically distributes model layers across available GPUs"
        },
        {
            "name": "Balanced",
            "value": "balanced", 
            "description": "Balances model layers across GPUs with equal memory usage"
        },
        {
            "name": "Balanced Low 0",
            "value": "balanced_low_0",
            "description": "Like balanced but keeps some layers on GPU 0"
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        print(f"  Value: {example['value']}")
        print(f"  Description: {example['description']}")
        print(f"  Usage: --embedding-device-map {example['value']}")

def show_performance_tips():
    """Show performance optimization tips for multi-GPU embedding."""
    
    print("\n" + "="*60)
    print("PERFORMANCE OPTIMIZATION TIPS")
    print("="*60)
    
    tips = [
        "Increase embedding batch size when using multi-GPU (e.g., --embedding-batch-size 32)",
        "Use 'auto' device_map for most cases - it automatically optimizes layer distribution",
        "Monitor GPU memory usage to ensure optimal distribution",
        "For very large models, consider using 'balanced_low_0' to keep some layers on GPU 0",
        "Multi-GPU is most beneficial for large embedding models (>1B parameters)",
        "Ensure all GPUs have similar memory capacity for best performance"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"{i}. {tip}")

if __name__ == "__main__":
    print("Multi-GPU Embedding Example for Benchmark Filtering Pipeline")
    print("="*60)
    
    # Show examples and tips
    show_device_map_examples()
    show_performance_tips()
    
    # Ask user if they want to run the example
    print("\n" + "="*60)
    response = input("Do you want to run the pipeline example? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\nRunning pipeline example...")
        success = run_pipeline_with_multi_gpu()
        if success:
            print("\nExample completed successfully!")
        else:
            print("\nExample failed. Check the error messages above.")
    else:
        print("Example skipped. You can run the pipeline manually using the commands shown above.")
