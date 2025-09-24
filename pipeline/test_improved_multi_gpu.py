#!/usr/bin/env python3
"""
Test script for the improved multi-GPU diversity computation with better memory management.
"""

import subprocess
import sys
import torch

def test_improved_multi_gpu():
    """Test the improved multi-GPU setup with memory management."""
    
    print("Testing Improved Multi-GPU Diversity Computation")
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
            "name": "Ultra Memory-Efficient (Small Model + Batch Size 1)",
            "args": [
                "--embedding-model", "sentence-transformers/all-MiniLM-L6-v2",
                "--embedding-batch-size", "1",
                "--skip-visualization",
                "--target-benchmark", "tau2-bench"
            ],
            "description": "Uses smallest model with batch size 1"
        },
        {
            "name": "Memory-Efficient Multi-GPU (Small Model)",
            "args": [
                "--embedding-model", "sentence-transformers/all-MiniLM-L6-v2",
                "--embedding-batch-size", "2",
                "--use-multi-gpu-diversity",
                "--skip-visualization",
                "--target-benchmark", "tau2-bench"
            ],
            "description": "Uses small model with multi-GPU and small batch"
        },
        {
            "name": "Improved Multi-GPU (Large Model + Memory Limits)",
            "args": [
                "--embedding-model", "Qwen/Qwen3-Embedding-8B",
                "--embedding-batch-size", "2",
                "--use-multi-gpu-diversity",
                "--embedding-device-map", "balanced",
                "--skip-visualization",
                "--target-benchmark", "tau2-bench"
            ],
            "description": "Uses large model with balanced device mapping and memory limits"
        },
        {
            "name": "Auto Multi-GPU (Large Model + Auto Mapping)",
            "args": [
                "--embedding-model", "Qwen/Qwen3-Embedding-8B",
                "--embedding-batch-size", "4",
                "--use-multi-gpu-diversity",
                "--skip-visualization",
                "--target-benchmark", "tau2-bench"
            ],
            "description": "Uses large model with automatic device mapping"
        }
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   Description: {config['description']}")
        print(f"   Command: CUDA_VISIBLE_DEVICES=6,7 python main.py {' '.join(config['args'])}")
        
        response = input(f"   Test this configuration? (y/n/q to quit): ").lower().strip()
        
        if response == 'q':
            break
        elif response in ['y', 'yes']:
            print(f"   Testing {config['name']}...")
            
            cmd = ["CUDA_VISIBLE_DEVICES=6,7", "python", "main.py"] + config['args']
            
            try:
                result = subprocess.run(
                    " ".join(cmd), 
                    shell=True,
                    capture_output=True, 
                    text=True, 
                    timeout=600  # 10 minute timeout
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

def show_improvements():
    """Show the improvements made to the multi-GPU setup."""
    
    print("\n" + "="*60)
    print("IMPROVEMENTS MADE TO MULTI-GPU SETUP")
    print("="*60)
    
    improvements = [
        "Added max_memory limits to prevent OOM (20GB per GPU)",
        "Added low_cpu_mem_usage=True for better memory management",
        "Automatic fallback to smaller model if large model fails",
        "Reduced batch size for multi-GPU (max 4 instead of 32)",
        "Added memory clearing before/after encoding operations",
        "Added OOM error handling with automatic batch size reduction",
        "Better error messages and logging for debugging"
    ]
    
    for i, improvement in enumerate(improvements, 1):
        print(f"{i}. {improvement}")

def show_quick_fixes():
    """Show quick fixes for the current OOM issue."""
    
    print("\n" + "="*60)
    print("QUICK FIXES FOR CURRENT OOM ISSUE")
    print("="*60)
    
    fixes = [
        {
            "name": "Immediate Fix (Small Model)",
            "command": "CUDA_VISIBLE_DEVICES=6,7 python main.py --embedding-model sentence-transformers/all-MiniLM-L6-v2 --embedding-batch-size 1 --skip-visualization --target-benchmark tau2-bench"
        },
        {
            "name": "Multi-GPU with Small Model",
            "command": "CUDA_VISIBLE_DEVICES=6,7 python main.py --embedding-model sentence-transformers/all-MiniLM-L6-v2 --embedding-batch-size 2 --use-multi-gpu-diversity --skip-visualization --target-benchmark tau2-bench"
        },
        {
            "name": "Skip Diversity Computation",
            "command": "CUDA_VISIBLE_DEVICES=6,7 python main.py --skip-measurement --target-benchmark tau2-bench"
        }
    ]
    
    for fix in fixes:
        print(f"\n{fix['name']}:")
        print(f"  {fix['command']}")

if __name__ == "__main__":
    print("Improved Multi-GPU Diversity Testing Tool")
    print("="*60)
    
    show_improvements()
    show_quick_fixes()
    
    print("\n" + "="*60)
    response = input("Do you want to test the improved configurations? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        test_improved_multi_gpu()
    else:
        print("Testing skipped. Use the quick fixes above to resolve the OOM issue.")
