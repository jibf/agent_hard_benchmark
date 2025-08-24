#!/usr/bin/env python3
"""
Simple Taubench Results Viewer
A lightweight tool to visualize taubench JSON result files without external dependencies.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse

class SimpleTaubenchViewer:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.data = self.load_data()
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load and parse the taubench JSON file."""
        with open(self.file_path, 'r') as f:
            return json.load(f)
    
    def print_separator(self, title: str = ""):
        """Print a separator line with optional title."""
        if title:
            print(f"\n{'='*20} {title} {'='*20}")
        else:
            print("\n" + "="*60)
    
    def display_summary(self):
        """Display overall summary of the results."""
        total_tasks = len(self.data)
        successful_tasks = sum(1 for item in self.data if item.get("reward", 0) > 0.5)
        avg_reward = sum(item.get("reward", 0) for item in self.data) / total_tasks
        total_cost = sum(item.get("info", {}).get("user_cost", 0) or 0 for item in self.data)
        
        self.print_separator("RESULTS SUMMARY")
        print(f"📊 Total Tasks: {total_tasks}")
        print(f"✅ Successful Tasks: {successful_tasks} ({successful_tasks/total_tasks*100:.1f}%)")
        print(f"📈 Average Reward: {avg_reward:.3f}")
        print(f"💰 Total Cost: ${total_cost:.4f}")
        print(f"📁 File: {self.file_path.name}")
    
    def display_task_list(self, only_failed: bool = False):
        """Display a list of all tasks with basic info."""
        self.print_separator("TASK LIST")
        
        print(f"{'#':<4} {'Task ID':<8} {'Reward':<8} {'Status':<12} {'Cost':<10} {'User ID':<15}")
        print("-" * 70)
        
        for i, item in enumerate(self.data):
            task_id = item.get("task_id", i)
            reward = item.get("reward", 0)
            status = "✅ Success" if reward > 0.5 else "❌ Failed"
            cost = item.get("info", {}).get("user_cost", 0) or 0
            user_id = item.get("info", {}).get("task", {}).get("user_id", "N/A")
            
            # Skip successful tasks if only showing failed
            if only_failed and reward > 0.5:
                continue
                
            print(f"{i+1:<4} {task_id:<8} {reward:<8.3f} {status:<12} ${cost:<9.4f} {user_id:<15}")
    
    def display_task_details(self, task_index: int):
        """Display detailed information about a specific task."""
        if task_index < 0 or task_index >= len(self.data):
            print(f"❌ Invalid task index: {task_index}")
            return
        
        item = self.data[task_index]
        task_id = item.get("task_id", task_index)
        
        self.print_separator(f"TASK {task_id} DETAILS")
        
        # Task Info
        task_info = item.get("info", {}).get("task", {})
        instruction = task_info.get("instruction", "No instruction available")
        user_id = task_info.get("user_id", "N/A")
        actions = task_info.get("actions", [])
        
        print(f"📋 Task ID: {task_id}")
        print(f"👤 User ID: {user_id}")
        print(f"🏆 Reward: {item.get('reward', 0):.3f}")
        print(f"💰 Cost: ${item.get('info', {}).get('user_cost', 0) or 0:.4f}")
        
        # Instruction
        self.print_separator("USER INSTRUCTION")
        print(instruction)
        
        # Expected Actions
        if actions:
            self.print_separator("EXPECTED ACTIONS")
            for i, action in enumerate(actions, 1):
                print(f"{i}. {action.get('name', 'Unknown')}")
                if 'kwargs' in action:
                    print(f"   Arguments: {json.dumps(action['kwargs'], indent=2)}")
                print()
        
        # Reward Info
        reward_info = item.get("info", {}).get("reward_info", {})
        if reward_info:
            self.print_separator("REWARD INFORMATION")
            print(f"Reward: {reward_info.get('reward', 0):.3f}")
            print(f"Actions Score: {reward_info.get('info', {}).get('r_actions', 0):.3f}")
            print(f"Data Hash: {reward_info.get('info', {}).get('gt_data_hash', 'N/A')}")
    
    def display_conversation(self, task_index: int):
        """Display the conversation flow for a specific task."""
        if task_index < 0 or task_index >= len(self.data):
            print(f"❌ Invalid task index: {task_index}")
            return
        
        item = self.data[task_index]
        task_id = item.get("task_id", task_index)
        trajectory = item.get("traj", [])
        
        self.print_separator(f"CONVERSATION FOR TASK {task_id}")
        
        for i, msg in enumerate(trajectory):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "system":
                # Skip system messages
                continue
            elif role == "user":
                print(f"\n👤 USER (Message {i+1}):")
                print(f"{'='*50}")
                print(content)
            elif role == "assistant":
                # Handle tool calls
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    for tool_call in tool_calls:
                        func_name = tool_call.get("function", {}).get("name", "unknown")
                        func_args = tool_call.get("function", {}).get("arguments", "{}")
                        
                        print(f"\n🔧 ASSISTANT TOOL CALL (Message {i+1}):")
                        print(f"{'='*50}")
                        print(f"Tool: {func_name}")
                        print(f"Arguments: {func_args}")
                
                if content:
                    print(f"\n🤖 ASSISTANT (Message {i+1}):")
                    print(f"{'='*50}")
                    print(content)
            elif role == "tool":
                tool_name = msg.get("name", "unknown")
                tool_response = msg.get("content", "")
                
                print(f"\n⚙️ TOOL RESPONSE (Message {i+1}):")
                print(f"{'='*50}")
                print(f"Tool: {tool_name}")
                print(f"Response: {tool_response}")
    
    def display_metrics(self):
        """Display detailed metrics and statistics."""
        rewards = [item.get("reward", 0) for item in self.data]
        costs = [item.get("info", {}).get("user_cost", 0) or 0 for item in self.data]
        
        # Calculate statistics
        total_tasks = len(self.data)
        successful_tasks = sum(1 for r in rewards if r > 0.5)
        avg_reward = sum(rewards) / total_tasks
        avg_cost = sum(costs) / total_tasks
        min_reward = min(rewards)
        max_reward = max(rewards)
        
        # Reward distribution
        reward_ranges = {
            "0.0-0.2": sum(1 for r in rewards if 0 <= r <= 0.2),
            "0.2-0.4": sum(1 for r in rewards if 0.2 < r <= 0.4),
            "0.4-0.6": sum(1 for r in rewards if 0.4 < r <= 0.6),
            "0.6-0.8": sum(1 for r in rewards if 0.6 < r <= 0.8),
            "0.8-1.0": sum(1 for r in rewards if 0.8 < r <= 1.0),
        }
        
        self.print_separator("DETAILED METRICS")
        print(f"📊 Total Tasks: {total_tasks}")
        print(f"✅ Successful Tasks: {successful_tasks} ({successful_tasks/total_tasks*100:.1f}%)")
        print(f"📈 Average Reward: {avg_reward:.3f}")
        print(f"📉 Min Reward: {min_reward:.3f}")
        print(f"📈 Max Reward: {max_reward:.3f}")
        print(f"💰 Average Cost: ${avg_cost:.4f}")
        print(f"💰 Total Cost: ${sum(costs):.4f}")
        
        self.print_separator("REWARD DISTRIBUTION")
        print(f"{'Range':<10} {'Count':<8} {'Percentage':<12}")
        print("-" * 30)
        for range_name, count in reward_ranges.items():
            percentage = count / total_tasks * 100
            print(f"{range_name:<10} {count:<8} {percentage:<12.1f}%")
    
    def interactive_mode(self):
        """Run the viewer in interactive mode."""
        while True:
            print("\n" + "="*50)
            print("🎯 TAUBENCH VIEWER MENU")
            print("="*50)
            print("1. Show summary")
            print("2. List all tasks")
            print("3. List failed tasks only")
            print("4. View task details")
            print("5. View conversation")
            print("6. Show metrics")
            print("7. Exit")
            print("="*50)
            
            try:
                choice = input("\nSelect option (1-7): ").strip()
                
                if choice == "1":
                    self.display_summary()
                elif choice == "2":
                    self.display_task_list()
                elif choice == "3":
                    self.display_task_list(only_failed=True)
                elif choice == "4":
                    try:
                        task_num = int(input("Enter task number (1-based): "))
                        self.display_task_details(task_num - 1)
                    except ValueError:
                        print("❌ Invalid task number")
                elif choice == "5":
                    try:
                        task_num = int(input("Enter task number (1-based): "))
                        self.display_conversation(task_num - 1)
                    except ValueError:
                        print("❌ Invalid task number")
                elif choice == "6":
                    self.display_metrics()
                elif choice == "7":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid option. Please select 1-7.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

def main():
    parser = argparse.ArgumentParser(description="Simple Taubench Results Viewer")
    parser.add_argument("file", help="Path to taubench JSON result file")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument("--list", action="store_true", help="List all tasks")
    parser.add_argument("--failed", action="store_true", help="List failed tasks only")
    parser.add_argument("--task", type=int, help="Show details for specific task (1-based)")
    parser.add_argument("--conversation", type=int, help="Show conversation for specific task (1-based)")
    parser.add_argument("--metrics", action="store_true", help="Show detailed metrics")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"❌ Error: File {file_path} not found")
        sys.exit(1)
    
    viewer = SimpleTaubenchViewer(file_path)
    
    if args.interactive:
        viewer.interactive_mode()
    else:
        if args.summary:
            viewer.display_summary()
        if args.list:
            viewer.display_task_list()
        if args.failed:
            viewer.display_task_list(only_failed=True)
        if args.task is not None:
            viewer.display_task_details(args.task - 1)
        if args.conversation is not None:
            viewer.display_conversation(args.conversation - 1)
        if args.metrics:
            viewer.display_metrics()
        
        # If no specific options provided, show summary
        if not any([args.summary, args.list, args.failed, args.task, args.conversation, args.metrics]):
            viewer.display_summary()

if __name__ == "__main__":
    main() 