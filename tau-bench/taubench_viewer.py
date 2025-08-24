#!/usr/bin/env python3
"""
Taubench Results Viewer
A tool to visualize and analyze taubench JSON result files.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import IntPrompt, Prompt
from rich.syntax import Syntax
from rich.layout import Layout
from rich.columns import Columns
from rich import box

console = Console()

class TaubenchViewer:
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.data = self.load_data()
        
    def load_data(self) -> List[Dict[str, Any]]:
        """Load and parse the taubench JSON file."""
        with open(self.file_path, 'r') as f:
            return json.load(f)
    
    def display_summary(self):
        """Display overall summary of the results."""
        total_tasks = len(self.data)
        successful_tasks = sum(1 for item in self.data if item.get("reward", 0) > 0.5)
        avg_reward = sum(item.get("reward", 0) for item in self.data) / total_tasks
        total_cost = sum(item.get("info", {}).get("user_cost", 0) for item in self.data)
        
        summary_table = Table(title="📊 Results Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Value", style="green")
        
        summary_table.add_row("Total Tasks", str(total_tasks))
        summary_table.add_row("Successful Tasks", f"{successful_tasks} ({successful_tasks/total_tasks*100:.1f}%)")
        summary_table.add_row("Average Reward", f"{avg_reward:.3f}")
        summary_table.add_row("Total Cost", f"${total_cost:.4f}")
        summary_table.add_row("File", str(self.file_path.name))
        
        console.print(summary_table)
        console.print()
    
    def display_task_list(self, only_failed: bool = False):
        """Display a list of all tasks with basic info."""
        table = Table(title="📋 Task List", box=box.ROUNDED)
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column("Task ID", style="blue")
        table.add_column("Reward", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Cost", style="magenta")
        table.add_column("User ID", style="white")
        
        for i, item in enumerate(self.data):
            task_id = item.get("task_id", i)
            reward = item.get("reward", 0)
            status = "✅ Success" if reward > 0.5 else "❌ Failed"
            cost = item.get("info", {}).get("user_cost", 0)
            user_id = item.get("info", {}).get("task", {}).get("user_id", "N/A")
            
            # Skip successful tasks if only showing failed
            if only_failed and reward > 0.5:
                continue
                
            table.add_row(
                str(i + 1),
                str(task_id),
                f"{reward:.3f}",
                status,
                f"${cost:.4f}",
                user_id
            )
        
        console.print(table)
        console.print()
    
    def display_task_details(self, task_index: int):
        """Display detailed information about a specific task."""
        if task_index < 0 or task_index >= len(self.data):
            console.print(f"[red]Invalid task index: {task_index}[/]")
            return
        
        item = self.data[task_index]
        task_id = item.get("task_id", task_index)
        
        # Task Info Panel
        task_info = item.get("info", {}).get("task", {})
        instruction = task_info.get("instruction", "No instruction available")
        user_id = task_info.get("user_id", "N/A")
        actions = task_info.get("actions", [])
        
        console.print(Panel(
            f"[bold blue]Task {task_id}[/]\n"
            f"[cyan]User ID:[/] {user_id}\n"
            f"[cyan]Reward:[/] {item.get('reward', 0):.3f}\n"
            f"[cyan]Cost:[/] ${item.get('info', {}).get('user_cost', 0):.4f}",
            title="📋 Task Information",
            border_style="blue"
        ))
        
        # Instruction Panel
        console.print(Panel(
            instruction,
            title="📝 User Instruction",
            border_style="green"
        ))
        
        # Expected Actions Panel
        if actions:
            actions_text = ""
            for i, action in enumerate(actions, 1):
                actions_text += f"[bold]{i}.[/] {action.get('name', 'Unknown')}\n"
                if 'kwargs' in action:
                    actions_text += f"   Args: {json.dumps(action['kwargs'], indent=2)}\n"
                actions_text += "\n"
            
            console.print(Panel(
                actions_text,
                title="🎯 Expected Actions",
                border_style="yellow"
            ))
        
        # Reward Info Panel
        reward_info = item.get("info", {}).get("reward_info", {})
        if reward_info:
            console.print(Panel(
                f"[cyan]Reward:[/] {reward_info.get('reward', 0):.3f}\n"
                f"[cyan]Actions Score:[/] {reward_info.get('info', {}).get('r_actions', 0):.3f}\n"
                f"[cyan]Data Hash:[/] {reward_info.get('info', {}).get('gt_data_hash', 'N/A')}",
                title="🏆 Reward Information",
                border_style="green"
            ))
        
        console.print()
    
    def display_conversation(self, task_index: int):
        """Display the conversation flow for a specific task."""
        if task_index < 0 or task_index >= len(self.data):
            console.print(f"[red]Invalid task index: {task_index}[/]")
            return
        
        item = self.data[task_index]
        task_id = item.get("task_id", task_index)
        trajectory = item.get("traj", [])
        
        console.print(Panel(
            f"[bold blue]Conversation for Task {task_id}[/]",
            title="💬 Conversation Flow",
            border_style="blue"
        ))
        
        for i, msg in enumerate(trajectory):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            if role == "system":
                # Skip system messages or show them differently
                continue
            elif role == "user":
                console.print(Panel(
                    content,
                    title=f"👤 User (Message {i+1})",
                    border_style="green",
                    title_align="left"
                ))
            elif role == "assistant":
                # Handle tool calls
                tool_calls = msg.get("tool_calls", [])
                if tool_calls:
                    for tool_call in tool_calls:
                        func_name = tool_call.get("function", {}).get("name", "unknown")
                        func_args = tool_call.get("function", {}).get("arguments", "{}")
                        
                        console.print(Panel(
                            f"[bold]Tool Call:[/] {func_name}\n"
                            f"[bold]Arguments:[/]\n{func_args}",
                            title=f"🔧 Assistant Tool Call (Message {i+1})",
                            border_style="yellow",
                            title_align="left"
                        ))
                
                if content:
                    console.print(Panel(
                        content,
                        title=f"🤖 Assistant (Message {i+1})",
                        border_style="blue",
                        title_align="left"
                    ))
            elif role == "tool":
                tool_name = msg.get("name", "unknown")
                tool_response = msg.get("content", "")
                
                console.print(Panel(
                    f"[bold]Tool:[/] {tool_name}\n"
                    f"[bold]Response:[/]\n{tool_response}",
                    title=f"⚙️ Tool Response (Message {i+1})",
                    border_style="red",
                    title_align="left"
                ))
        
        console.print()
    
    def display_metrics(self):
        """Display detailed metrics and statistics."""
        rewards = [item.get("reward", 0) for item in self.data]
        costs = [item.get("info", {}).get("user_cost", 0) for item in self.data]
        
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
        
        metrics_table = Table(title="📈 Detailed Metrics", box=box.ROUNDED)
        metrics_table.add_column("Metric", style="cyan", no_wrap=True)
        metrics_table.add_column("Value", style="green")
        
        metrics_table.add_row("Total Tasks", str(total_tasks))
        metrics_table.add_row("Successful Tasks", f"{successful_tasks} ({successful_tasks/total_tasks*100:.1f}%)")
        metrics_table.add_row("Average Reward", f"{avg_reward:.3f}")
        metrics_table.add_row("Min Reward", f"{min_reward:.3f}")
        metrics_table.add_row("Max Reward", f"{max_reward:.3f}")
        metrics_table.add_row("Average Cost", f"${avg_cost:.4f}")
        metrics_table.add_row("Total Cost", f"${sum(costs):.4f}")
        
        console.print(metrics_table)
        console.print()
        
        # Reward distribution
        dist_table = Table(title="📊 Reward Distribution", box=box.ROUNDED)
        dist_table.add_column("Range", style="cyan")
        dist_table.add_column("Count", style="green")
        dist_table.add_column("Percentage", style="yellow")
        
        for range_name, count in reward_ranges.items():
            percentage = count / total_tasks * 100
            dist_table.add_row(range_name, str(count), f"{percentage:.1f}%")
        
        console.print(dist_table)
        console.print()
    
    def interactive_mode(self):
        """Run the viewer in interactive mode."""
        while True:
            console.print("\n[bold yellow]Taubench Viewer Menu:[/]")
            console.print("1. Show summary")
            console.print("2. List all tasks")
            console.print("3. List failed tasks only")
            console.print("4. View task details")
            console.print("5. View conversation")
            console.print("6. Show metrics")
            console.print("7. Exit")
            
            choice = Prompt.ask("\nSelect option", choices=["1", "2", "3", "4", "5", "6", "7"])
            
            if choice == "1":
                console.clear()
                self.display_summary()
            elif choice == "2":
                console.clear()
                self.display_task_list()
            elif choice == "3":
                console.clear()
                self.display_task_list(only_failed=True)
            elif choice == "4":
                task_num = IntPrompt.ask("Enter task number (1-based)", default=1)
                console.clear()
                self.display_task_details(task_num - 1)
            elif choice == "5":
                task_num = IntPrompt.ask("Enter task number (1-based)", default=1)
                console.clear()
                self.display_conversation(task_num - 1)
            elif choice == "6":
                console.clear()
                self.display_metrics()
            elif choice == "7":
                console.print("[green]Goodbye![/]")
                break

def main():
    parser = argparse.ArgumentParser(description="Taubench Results Viewer")
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
        console.print(f"[red]Error: File {file_path} not found[/]")
        sys.exit(1)
    
    viewer = TaubenchViewer(file_path)
    
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