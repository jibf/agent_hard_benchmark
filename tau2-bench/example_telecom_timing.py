#!/usr/bin/env python3
"""
Example script demonstrating timing estimates and progress tracking for telecom domain in tau2-bench.

This script shows how to run tau2-bench with comprehensive progress tracking,
including overall progress, individual task progress, and real-time ETA calculations
for the telecom domain.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path so we can import tau2
sys.path.insert(0, str(Path(__file__).parent / "src"))

from tau2.data_model.simulation import RunConfig
from tau2.run import run_domain

def main():
    """
    Example usage of tau2-bench with enhanced progress tracking for telecom domain.
    
    This will run a small subset of telecom tasks to demonstrate the progress tracking
    features including:
    - Overall progress bar with ETA
    - Individual task progress with step tracking
    - Real-time timing information
    - Performance metrics
    """
    
    # Example configuration for telecom domain demonstration
    config = RunConfig(
        domain="telecom",
        agent="llm_agent",
        user="user_simulator",
        task_ids=["telecom_task_1", "telecom_task_2"],  # Run specific tasks for demo
        llm_agent="openai/gpt-4o-mini",
        llm_args_agent={"temperature": 0.0},
        llm_user="openai/gpt-4o-mini",
        llm_args_user={"temperature": 0.0},
        num_trials=2,  # Run 2 trials for demonstration
        max_steps=50,  # Limit steps for demo
        max_errors=5,
        max_concurrency=1,  # Use 1 for demo to see detailed progress
        seed=42,
        save_to="telecom_demo_results",
    )
    
    print("🚀 Starting tau2-bench with enhanced progress tracking for telecom domain...")
    print("=" * 70)
    print("Progress tracking features:")
    print("• Overall progress bar with ETA calculation")
    print("• Individual task progress with step tracking")
    print("• Real-time timing and performance metrics")
    print("• Task completion status and rewards")
    print("=" * 70)
    
    # Run the benchmark
    results = run_domain(config)
    
    print("\n" + "=" * 70)
    print("✅ Telecom benchmark completed!")
    print(f"📊 Total simulations: {len(results.simulations)}")
    
    # Calculate success rate
    successful_sims = sum(1 for sim in results.simulations 
                         if sim.reward_info and sim.reward_info.reward == 1.0)
    success_rate = successful_sims / len(results.simulations) if results.simulations else 0
    print(f"🎯 Successful simulations: {successful_sims}")
    print(f"📈 Success rate: {success_rate:.2%}")
    
    # Show timing statistics
    if results.simulations:
        avg_duration = sum(sim.duration for sim in results.simulations) / len(results.simulations)
        print(f"⏱️  Average simulation duration: {avg_duration:.1f} seconds")
        print(f"🚀 Simulations per second: {len(results.simulations) / sum(sim.duration for sim in results.simulations):.2f}")
    
    print("=" * 70)
    print("\n💡 Tips for telecom domain:")
    print("• Telecom tasks can be complex and time-consuming")
    print("• Use lower concurrency (1-2) for detailed progress tracking")
    print("• Monitor ETA to estimate completion times")
    print("• Check individual task progress for long-running simulations")
    print("\nTo run with your own configuration:")
    print("python -m tau2 run --domain telecom --agent llm_agent --user user_simulator")
    print("  --agent-llm openai/gpt-4o --user-llm openai/gpt-4o")
    print("  --max-concurrency 2 --num-trials 1")

if __name__ == "__main__":
    main() 