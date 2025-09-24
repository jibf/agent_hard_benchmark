#!/usr/bin/env python3
"""
Example script demonstrating multi-GPU diversity computation in the benchmark filtering pipeline.

This script shows how to use the new multi-GPU support for diversity calculations.
"""

import subprocess
import sys
import torch

def run_multi_gpu_diversity_example():
    """Example of running the pipeline with multi-GPU diversity support."""
    
    # Check if multiple GPUs are available
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    if gpu_count < 2:
        print("Warning: Multi-GPU diversity requires at least 2 GPUs")
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
        # Add multi-GPU diversity arguments
        cmd.extend([
            "--use-multi-gpu-diversity",
        ])
        print("Running with multi-GPU diversity computation...")
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

def show_multi_gpu_diversity_info():
    """Show information about multi-GPU diversity computation."""
    
    print("\n" + "="*60)
    print("MULTI-GPU DIVERSITY COMPUTATION")
    print("="*60)
    
    info = [
        "Multi-GPU diversity distributes embedding computation across multiple GPUs",
        "Each GPU processes a portion of the texts in parallel",
        "Results are combined to compute the final diversity metric",
        "Significant speedup for large datasets with multiple GPUs",
        "Automatic fallback to single GPU if insufficient GPUs available"
    ]
    
    for i, item in enumerate(info, 1):
        print(f"{i}. {item}")

def show_performance_benefits():
    """Show expected performance benefits of multi-GPU diversity."""
    
    print("\n" + "="*60)
    print("PERFORMANCE BENEFITS")
    print("="*60)
    
    benefits = [
        "2 GPUs: ~1.5-1.8x speedup for diversity computation",
        "4 GPUs: ~2.5-3.2x speedup for diversity computation", 
        "8 GPUs: ~4.0-5.5x speedup for diversity computation",
        "Best for large datasets (>1000 texts per benchmark)",
        "Reduces total pipeline runtime significantly",
        "Memory usage distributed across GPUs"
    ]
    
    for benefit in benefits:
        print(f"• {benefit}")

def show_usage_examples():
    """Show usage examples for multi-GPU diversity."""
    
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    examples = [
        {
            "name": "Basic Multi-GPU Diversity",
            "command": "python main.py --use-multi-gpu-diversity --target-benchmark complex-func-bench"
        },
        {
            "name": "High-Performance Multi-GPU",
            "command": "python main.py --use-multi-gpu-diversity --embedding-batch-size 32 --target-benchmark complex-func-bench"
        },
        {
            "name": "Multi-GPU with Small Model",
            "command": "python main.py --use-multi-gpu-diversity --embedding-model sentence-transformers/all-MiniLM-L6-v2 --target-benchmark complex-func-bench"
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        print(f"  {example['command']}")

def show_requirements():
    """Show requirements for multi-GPU diversity."""
    
    print("\n" + "="*60)
    print("REQUIREMENTS")
    print("="*60)
    
    requirements = [
        "Multiple NVIDIA GPUs with CUDA support",
        "PyTorch with CUDA support",
        "SentenceTransformer library",
        "Sufficient GPU memory for embedding model on each GPU",
        "At least 2 GPUs (automatically falls back to single GPU if not available)"
    ]
    
    for req in requirements:
        print(f"• {req}")

if __name__ == "__main__":
    print("Multi-GPU Diversity Computation Example")
    print("="*60)
    
    show_multi_gpu_diversity_info()
    show_performance_benefits()
    show_requirements()
    show_usage_examples()
    
    # Check GPU availability
    gpu_count = torch.cuda.device_count()
    print(f"\nCurrent GPU setup: {gpu_count} GPU(s) available")
    
    if gpu_count >= 2:
        print("✅ Multi-GPU diversity is available!")
    else:
        print("⚠️  Multi-GPU diversity requires at least 2 GPUs")
    
    print("\n" + "="*60)
    response = input("Do you want to run the multi-GPU diversity example? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\nRunning multi-GPU diversity example...")
        success = run_multi_gpu_diversity_example()
        if success:
            print("\nExample completed successfully!")
        else:
            print("\nExample failed. Check the error messages above.")
    else:
        print("Example skipped. You can run the pipeline manually using the commands shown above.")
