#!/usr/bin/env python3
"""
Script to analyze telecom tasks and provide recommendations for optimizing max_steps parameter.

This script analyzes the telecom tasks to understand:
1. Typical number of actions required per task
2. Distribution of task complexity
3. Recommended max_steps values for different scenarios
"""

import json
import sys
from pathlib import Path
from collections import Counter
import statistics

def analyze_telecom_tasks():
    """Analyze telecom tasks to understand complexity and recommend max_steps values."""
    
    # Load telecom tasks
    tasks_file = Path("data/tau2/domains/telecom/tasks.json")
    if not tasks_file.exists():
        print(f"❌ Tasks file not found: {tasks_file}")
        return
    
    with open(tasks_file, 'r') as f:
        tasks = json.load(f)
    
    print(f"📊 Analyzing {len(tasks)} telecom tasks...")
    print("=" * 60)
    
    # Analyze action counts
    action_counts = []
    task_complexities = []
    
    for task in tasks:
        if "evaluation_criteria" in task and "actions" in task["evaluation_criteria"]:
            action_count = len(task["evaluation_criteria"]["actions"])
            action_counts.append(action_count)
            
            # Categorize complexity
            if action_count <= 2:
                complexity = "Simple"
            elif action_count <= 4:
                complexity = "Medium"
            elif action_count <= 6:
                complexity = "Complex"
            else:
                complexity = "Very Complex"
            
            task_complexities.append(complexity)
    
    if not action_counts:
        print("❌ No tasks with actions found")
        return
    
    # Calculate statistics
    min_actions = min(action_counts)
    max_actions = max(action_counts)
    avg_actions = statistics.mean(action_counts)
    median_actions = statistics.median(action_counts)
    p90_actions = sorted(action_counts)[int(len(action_counts) * 0.9)]
    p95_actions = sorted(action_counts)[int(len(action_counts) * 0.95)]
    
    # Count complexity distribution
    complexity_counts = Counter(task_complexities)
    
    print("📈 Task Complexity Analysis:")
    print(f"   • Total tasks analyzed: {len(action_counts)}")
    print(f"   • Actions per task:")
    print(f"     - Minimum: {min_actions}")
    print(f"     - Maximum: {max_actions}")
    print(f"     - Average: {avg_actions:.1f}")
    print(f"     - Median: {median_actions}")
    print(f"     - 90th percentile: {p90_actions}")
    print(f"     - 95th percentile: {p95_actions}")
    
    print("\n📊 Complexity Distribution:")
    for complexity, count in complexity_counts.most_common():
        percentage = (count / len(task_complexities)) * 100
        print(f"   • {complexity}: {count} tasks ({percentage:.1f}%)")
    
    # Calculate recommended max_steps
    # Each action typically requires 2-3 steps (agent action + user response + environment update)
    steps_per_action = 3
    buffer_steps = 10  # Extra steps for exploration, clarification, etc.
    
    print("\n🎯 Recommended max_steps Values:")
    print("=" * 60)
    
    # Conservative approach (covers 95% of tasks)
    conservative_steps = (p95_actions * steps_per_action) + buffer_steps
    print(f"   • Conservative (95% coverage): {conservative_steps} steps")
    print(f"     - Based on {p95_actions} actions × {steps_per_action} steps/action + {buffer_steps} buffer")
    
    # Balanced approach (covers 90% of tasks)
    balanced_steps = (p90_actions * steps_per_action) + buffer_steps
    print(f"   • Balanced (90% coverage): {balanced_steps} steps")
    print(f"     - Based on {p90_actions} actions × {steps_per_action} steps/action + {buffer_steps} buffer")
    
    # Aggressive approach (covers 75% of tasks)
    p75_actions = sorted(action_counts)[int(len(action_counts) * 0.75)]
    aggressive_steps = (p75_actions * steps_per_action) + buffer_steps
    print(f"   • Aggressive (75% coverage): {aggressive_steps} steps")
    print(f"     - Based on {p75_actions} actions × {steps_per_action} steps/action + {buffer_steps} buffer")
    
    # Current default
    current_default = 200
    print(f"   • Current default: {current_default} steps")
    
    print("\n💡 Recommendations:")
    print("=" * 60)
    
    if balanced_steps < current_default:
        print(f"✅ You can reduce max_steps from {current_default} to {balanced_steps}")
        print(f"   - This will cover 90% of tasks and reduce execution time")
        print(f"   - Time savings: ~{((current_default - balanced_steps) / current_default * 100):.1f}%")
    else:
        print(f"⚠️  Consider increasing max_steps from {current_default} to {balanced_steps}")
        print(f"   - This will ensure 90% of tasks can complete")
    
    print(f"\n🚀 For fastest execution (75% coverage): {aggressive_steps} steps")
    print(f"🛡️  For maximum coverage (95%): {conservative_steps} steps")
    
    # Show some example tasks
    print("\n📋 Example Tasks by Complexity:")
    print("=" * 60)
    
    simple_tasks = [task for task in tasks if "evaluation_criteria" in task and len(task["evaluation_criteria"]["actions"]) <= 2][:3]
    complex_tasks = [task for task in tasks if "evaluation_criteria" in task and len(task["evaluation_criteria"]["actions"]) >= 5][:3]
    
    print("Simple Tasks (≤2 actions):")
    for task in simple_tasks:
        action_count = len(task["evaluation_criteria"]["actions"])
        print(f"   • {task['id'][:50]}... ({action_count} actions)")
    
    print("\nComplex Tasks (≥5 actions):")
    for task in complex_tasks:
        action_count = len(task["evaluation_criteria"]["actions"])
        print(f"   • {task['id'][:50]}... ({action_count} actions)")
    
    print("\n🔧 Usage Examples:")
    print("=" * 60)
    print("For development/testing (fast iteration):")
    print(f"   python -m tau2 run --domain telecom --max-steps {aggressive_steps}")
    print("\nFor production runs (balanced):")
    print(f"   python -m tau2 run --domain telecom --max-steps {balanced_steps}")
    print("\nFor comprehensive evaluation (maximum coverage):")
    print(f"   python -m tau2 run --domain telecom --max-steps {conservative_steps}")

if __name__ == "__main__":
    analyze_telecom_tasks() 