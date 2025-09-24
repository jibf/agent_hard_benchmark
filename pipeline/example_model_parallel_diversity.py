#!/usr/bin/env python3
"""
Example script demonstrating model-parallel diversity computation in the benchmark filtering pipeline.

This script shows how to use model parallelism (splitting the model across GPUs) for diversity calculations.
"""

import subprocess
import sys
import torch

def run_model_parallel_diversity_example():
    """Example of running the pipeline with model-parallel diversity support."""
    
    # Check if multiple GPUs are available
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    if gpu_count < 2:
        print("Warning: Model-parallel diversity requires at least 2 GPUs")
        print("Running with single GPU instead...")
        use_multi_gpu = False
    else:
        use_multi_gpu = True
    
    # Base command
    cmd = [
        "python", "main.py",
        "--target-benchmark", "complex-func-bench",  # Example benchmark
        "--embedding-model", "Qwen/Qwen3-Embedding-8B",
        "--embedding-batch-size", "16",  # Can use larger batch size with model parallelism
    ]
    
    if use_multi_gpu:
        # Add model-parallel diversity arguments
        cmd.extend([
            "--use-multi-gpu-diversity",
            "--embedding-device-map", "auto",  # or "balanced", "balanced_low_0"
        ])
        print("Running with model-parallel diversity computation...")
    else:
        print("Running with single GPU diversity computation...")
    
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

def show_model_parallelism_info():
    """Show information about model parallelism for diversity computation."""
    
    print("\n" + "="*60)
    print("MODEL PARALLELISM FOR DIVERSITY COMPUTATION")
    print("="*60)
    
    info = [
        "Model parallelism splits the embedding model across multiple GPUs",
        "Each GPU handles different layers of the model",
        "Much more memory efficient than data parallelism",
        "Allows larger models to fit in GPU memory",
        "Single model instance spans multiple GPUs automatically",
        "HuggingFace device_map handles the distribution automatically"
    ]
    
    for i, item in enumerate(info, 1):
        print(f"{i}. {item}")

def show_device_map_options():
    """Show different device_map options."""
    
    print("\n" + "="*60)
    print("DEVICE MAP OPTIONS")
    print("="*60)
    
    options = [
        {
            "name": "auto",
            "description": "Automatically distributes model layers across available GPUs (recommended)",
            "best_for": "Most use cases"
        },
        {
            "name": "balanced", 
            "description": "Distributes layers evenly across GPUs with equal memory usage",
            "best_for": "Models with uniform layer sizes"
        },
        {
            "name": "balanced_low_0",
            "description": "Like balanced but keeps some layers on GPU 0",
            "best_for": "Very large models or when GPU 0 has more memory"
        }
    ]
    
    for option in options:
        print(f"\n{option['name'].upper()}:")
        print(f"  Description: {option['description']}")
        print(f"  Best for: {option['best_for']}")

def show_advantages_over_data_parallelism():
    """Show advantages of model parallelism over data parallelism."""
    
    print("\n" + "="*60)
    print("ADVANTAGES OF MODEL PARALLELISM")
    print("="*60)
    
    advantages = [
        "Memory Efficiency: Model layers distributed across GPUs, not duplicated",
        "Larger Models: Can fit models that don't fit on single GPU",
        "Better Utilization: Each GPU processes different parts of the model",
        "Simpler Code: Single model instance, no need to manage multiple models",
        "Automatic Optimization: HuggingFace handles optimal layer distribution",
        "Lower Memory Overhead: No need to store multiple copies of the model"
    ]
    
    for advantage in advantages:
        print(f"• {advantage}")

def show_usage_examples():
    """Show usage examples for model-parallel diversity."""
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    examples = [
        {
            "name": "Basic Model Parallelism",
            "command": "python main.py --use-multi-gpu-diversity --target-benchmark complex-func-bench"
        },
        {
            "name": "Balanced Device Mapping",
            "command": "python main.py --use-multi-gpu-diversity --embedding-device-map balanced --target-benchmark complex-func-bench"
        },
        {
            "name": "High-Performance with Large Batch",
            "command": "python main.py --use-multi-gpu-diversity --embedding-device-map auto --embedding-batch-size 32 --target-benchmark complex-func-bench"
        },
        {
            "name": "Large Model with Balanced Low 0",
            "command": "python main.py --use-multi-gpu-diversity --embedding-device-map balanced_low_0 --embedding-model Qwen/Qwen3-Embedding-8B --target-benchmark complex-func-bench"
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        print(f"  {example['command']}")

def show_memory_comparison():
    """Show memory usage comparison between approaches."""
    
    print("\n" + "="*60)
    print("MEMORY USAGE COMPARISON")
    print("="*60)
    
    print("For Qwen/Qwen3-Embedding-8B (8GB model):")
    print()
    print("Data Parallelism (old approach):")
    print("  • 2 GPUs: 16GB total (8GB per GPU)")
    print("  • 4 GPUs: 32GB total (8GB per GPU)")
    print("  • 8 GPUs: 64GB total (8GB per GPU)")
    print()
    print("Model Parallelism (new approach):")
    print("  • 2 GPUs: ~8GB total (4GB per GPU)")
    print("  • 4 GPUs: ~8GB total (2GB per GPU)")
    print("  • 8 GPUs: ~8GB total (1GB per GPU)")
    print()
    print("✅ Model parallelism uses much less memory!")

if __name__ == "__main__":
    print("Model-Parallel Diversity Computation Example")
    print("="*60)
    
    show_model_parallelism_info()
    show_device_map_options()
    show_advantages_over_data_parallelism()
    show_memory_comparison()
    show_usage_examples()
    
    # Check GPU availability
    gpu_count = torch.cuda.device_count()
    print(f"\nCurrent GPU setup: {gpu_count} GPU(s) available")
    
    if gpu_count >= 2:
        print("✅ Model-parallel diversity is available!")
        print("💡 This approach is much more memory efficient than data parallelism")
    else:
        print("⚠️  Model-parallel diversity requires at least 2 GPUs")
    
    print("\n" + "="*60)
    response = input("Do you want to run the model-parallel diversity example? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\nRunning model-parallel diversity example...")
        success = run_model_parallel_diversity_example()
        if success:
            print("\nExample completed successfully!")
        else:
            print("\nExample failed. Check the error messages above.")
    else:
        print("Example skipped. You can run the pipeline manually using the commands shown above.")
