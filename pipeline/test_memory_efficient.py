#!/usr/bin/env python3
"""
Test script for memory-efficient embedding computation.
This script helps you test different configurations to avoid CUDA OOM errors.
"""

import subprocess
import sys
import torch

def test_memory_efficient_configs():
    """Test different memory-efficient configurations."""
    
    print("Testing Memory-Efficient Embedding Configurations")
    print("=" * 60)
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Available GPUs: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name} ({memory_gb:.1f} GB)")
    else:
        print("No CUDA GPUs available")
        return
    
    # Test configurations from most memory-efficient to least
    configs = [
        {
            "name": "Ultra Memory-Efficient (CPU + Small Model)",
            "args": [
                "--embedding-model", "sentence-transformers/all-MiniLM-L6-v2",
                "--embedding-batch-size", "1",
                "--skip-visualization",
                "--target-benchmark", "complex-func-bench"
            ],
            "description": "Uses CPU with smallest embedding model"
        },
        {
            "name": "Memory-Efficient (Small Model + Small Batch)",
            "args": [
                "--embedding-model", "sentence-transformers/all-MiniLM-L6-v2", 
                "--embedding-batch-size", "2",
                "--skip-visualization",
                "--target-benchmark", "complex-func-bench"
            ],
            "description": "Uses small model with very small batch size"
        },
        {
            "name": "Default Memory-Efficient (Large Model + Small Batch)",
            "args": [
                "--embedding-model", "Qwen/Qwen3-Embedding-8B",
                "--embedding-batch-size", "2", 
                "--skip-visualization",
                "--target-benchmark", "complex-func-bench"
            ],
            "description": "Uses large model with small batch size (default fix)"
        },
        {
            "name": "Standard (Large Model + Default Batch)",
            "args": [
                "--embedding-model", "Qwen/Qwen3-Embedding-8B",
                "--embedding-batch-size", "4",
                "--skip-visualization", 
                "--target-benchmark", "complex-func-bench"
            ],
            "description": "Uses large model with default batch size"
        }
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   Description: {config['description']}")
        print(f"   Command: python main.py {' '.join(config['args'])}")
        
        response = input(f"   Test this configuration? (y/n/q to quit): ").lower().strip()
        
        if response == 'q':
            break
        elif response in ['y', 'yes']:
            print(f"   Testing {config['name']}...")
            
            cmd = ["python", "main.py"] + config['args']
            
            try:
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300  # 5 minute timeout
                )
                
                if result.returncode == 0:
                    print(f"   ✅ SUCCESS: {config['name']} completed without errors")
                    # Show last few lines of output
                    output_lines = result.stdout.strip().split('\n')
                    if output_lines:
                        print(f"   Last output: {output_lines[-1]}")
                else:
                    print(f"   ❌ FAILED: {config['name']} failed")
                    if "out of memory" in result.stderr.lower():
                        print(f"   Error: CUDA OOM detected")
                    else:
                        print(f"   Error: {result.stderr[-200:]}")
                        
            except subprocess.TimeoutExpired:
                print(f"   ⏰ TIMEOUT: {config['name']} took too long")
            except Exception as e:
                print(f"   ❌ ERROR: {config['name']} - {e}")

def show_memory_tips():
    """Show memory optimization tips."""
    
    print("\n" + "=" * 60)
    print("MEMORY OPTIMIZATION TIPS")
    print("=" * 60)
    
    tips = [
        "Use smaller embedding models: 'sentence-transformers/all-MiniLM-L6-v2' (22MB) vs 'Qwen/Qwen3-Embedding-8B' (8GB)",
        "Reduce batch size: Start with --embedding-batch-size 1 or 2",
        "Skip visualization: Use --skip-visualization to save memory",
        "Use CPU fallback: The pipeline automatically falls back to CPU if GPU OOM occurs",
        "Monitor GPU memory: Use 'nvidia-smi -l 1' to watch memory usage",
        "Process smaller datasets: Use --target-benchmark to process one benchmark at a time",
        "Clear GPU cache: The pipeline now automatically clears GPU memory between operations"
    ]
    
    for i, tip in enumerate(tips, 1):
        print(f"{i}. {tip}")

def show_quick_fixes():
    """Show quick command-line fixes for CUDA OOM."""
    
    print("\n" + "=" * 60)
    print("QUICK FIXES FOR CUDA OOM")
    print("=" * 60)
    
    fixes = [
        {
            "name": "Immediate Fix (CPU + Small Model)",
            "command": "python main.py --embedding-model sentence-transformers/all-MiniLM-L6-v2 --embedding-batch-size 1 --skip-visualization"
        },
        {
            "name": "Memory-Efficient Fix (Small Batch)",
            "command": "python main.py --embedding-batch-size 2 --skip-visualization"
        },
        {
            "name": "Skip Embedding Computation",
            "command": "python main.py --skip-measurement"
        }
    ]
    
    for fix in fixes:
        print(f"\n{fix['name']}:")
        print(f"  {fix['command']}")

if __name__ == "__main__":
    print("CUDA OOM Memory-Efficient Testing Tool")
    print("=" * 60)
    
    show_memory_tips()
    show_quick_fixes()
    
    print("\n" + "=" * 60)
    response = input("Do you want to test different configurations? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        test_memory_efficient_configs()
    else:
        print("Testing skipped. Use the quick fixes above to resolve CUDA OOM issues.")
