#!/usr/bin/env python3
"""
Example script demonstrating enhanced progress tracking in tau-bench.

This script shows how to run tau-bench with comprehensive progress tracking,
including overall progress, trial progress, and individual task progress
with real-time ETA calculations.
"""

import os
import sys
from tau_bench.run import run
from tau_bench.types import RunConfig

def main():
    """
    Example usage of tau-bench with enhanced progress tracking.
    
    This will run a small subset of tasks to demonstrate the progress tracking
    features including:
    - Overall progress bar with ETA
    - Trial-specific progress bars
    - Individual task progress with action tracking
    - Real-time timing information
    """
    
    # Example configuration for demonstration
    config = RunConfig(
        model_provider="openai",
        user_model_provider="openai", 
        model="gpt-4o",
        user_model="gpt-4o",
        num_trials=2,  # Run 2 trials for demonstration
        env="retail",
        agent_strategy="tool-calling",
        temperature=0.0,
        task_split="test",
        start_index=0,
        end_index=3,  # Only run first 3 tasks for demo
        task_ids=None,
        log_dir="results",
        max_concurrency=1,  # Use 1 for demo to see individual progress
        seed=42,
        shuffle=0,
        user_strategy="llm",
        few_shot_displays_path=None,
    )
    
    print("🚀 Starting tau-bench with enhanced progress tracking...")
    print("=" * 60)
    print("Progress tracking features:")
    print("• Overall progress bar with ETA calculation")
    print("• Trial-specific progress bars")
    print("• Individual task progress with action tracking")
    print("• Real-time timing and performance metrics")
    print("=" * 60)
    
    # Run the benchmark
    results = run(config)
    
    print("\n" + "=" * 60)
    print("✅ Benchmark completed!")
    print(f"📊 Total results: {len(results)}")
    print(f"🎯 Successful tasks: {sum(1 for r in results if r.reward == 1)}")
    print(f"📈 Success rate: {sum(1 for r in results if r.reward == 1) / len(results):.2%}")
    print("=" * 60)

if __name__ == "__main__":
    # Check if API keys are set
    required_keys = ["OPENAI_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"❌ Missing required environment variables: {missing_keys}")
        print("Please set your API keys before running this example:")
        print("export OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)
    
    main() 