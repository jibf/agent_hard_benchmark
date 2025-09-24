#!/usr/bin/env python3
"""
Extract flawed tasks from results and show their context using the bench loader.
"""

import json
import sys
import os
from typing import Dict, List, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.bench_loaders import get_bench_loader
from src.utils.types import Benchmark, UniqueQuestionID

def load_flawed_tasks_from_results(results_file: str) -> List[Dict[str, Any]]:
    """Load tasks with is_flawed=True from the results JSONL file, ordered by issue frequency (infrequent first)."""
    flawed_tasks = []
    
    # Define issue type priority (infrequent first)
    issue_priority = {
        'Vague instruction': 1,
        'Contradictory': 2, 
        'Ground-Truth': 3,
        'Redundant/ungrounded function calls': 4,
        'Flawed function design': 5,
        'Insufficient toolsets': 6,
        'Unjustified/Hallucinated Parameters': 7,
        'Policy Violation': 8,
        'Incorrect function calls': 9,
        'Malformed function calls': 10
    }
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                task_data = json.loads(line.strip())
                
                # Check if this task is flawed
                llm_judge_output = task_data.get('llm_judge_output', {})
                specific_filter = llm_judge_output.get('specific_filter', {})
                
                if specific_filter.get('is_flawed', False):
                    # Skip special_* task types
                    task_name = task_data.get('task_name', '')
                    if task_name.startswith('special_'):
                        continue
                    
                    error_category = specific_filter.get('error_category', 'Unknown')
                    priority = issue_priority.get(error_category, 999)  # Unknown issues go last
                    
                    flawed_tasks.append({
                        'line_number': line_num,
                        'task_data': task_data,
                        'priority': priority,
                        'error_category': error_category
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    # Sort by priority (infrequent issue types first)
    flawed_tasks.sort(key=lambda x: x['priority'])
    
    return flawed_tasks

def get_task_context(benchmark_name: str, task_name: str, question_id: str) -> Dict[str, Any]:
    """Get the context for a specific task using the bench loader."""
    try:
        # Map benchmark name to Benchmark enum
        benchmark_map = {
            'ACEBench': Benchmark.ACE_BENCH,
            'TauBench': Benchmark.TAU_BENCH,
            'Tau2Bench': Benchmark.TAU2_BENCH,
            'NexusBench': Benchmark.NEXUS_BENCH,
            'ToolSandbox': Benchmark.TOOL_SANDBOX,
            'ComplexFuncBench': Benchmark.COMPLEX_FUNC_BENCH,
            'DrafterBench': Benchmark.DRAFTER_BENCH,
            'BFCL': Benchmark.BFCL,
            'MultiChallenge': Benchmark.MULTI_CHALLENGE
        }
        
        benchmark = benchmark_map.get(benchmark_name)
        if not benchmark:
            return {"error": f"Unknown benchmark: {benchmark_name}"}
        
        # Get the appropriate loader
        loader_class = get_bench_loader(benchmark)
        loader = loader_class()
        
        # Load all questions
        questions = loader.load_questions()
        
        # Find the specific question
        target_question = None
        for question in questions:
            if (question.question_id == question_id and 
                question.task_name == task_name and 
                question.benchmark == benchmark):
                target_question = question
                break
        
        if not target_question:
            return {"error": f"Question not found: {question_id}"}
        
        # Extract relevant context based on benchmark type
        context = {
            "benchmark": benchmark_name,
            "task_name": task_name,
            "question_id": question_id,
            "instruction": getattr(target_question, 'instruction', ''),
            "available_function_list": getattr(target_question, 'available_function_list', []),
            "gt_conv_traj": getattr(target_question, 'gt_conv_traj', []),
        }
        
        # Add benchmark-specific fields
        if hasattr(target_question, 'agent_system_prompt'):
            context["agent_system_prompt"] = target_question.agent_system_prompt
        if hasattr(target_question, 'previous_conversation_history'):
            context["previous_conversation_history"] = target_question.previous_conversation_history
        if hasattr(target_question, 'time'):
            context["time"] = target_question.time
        if hasattr(target_question, 'user_context'):
            context["user_context"] = target_question.user_context
        if hasattr(target_question, 'meta'):
            context["meta"] = target_question.meta
            
        return context
        
    except Exception as e:
        return {"error": f"Error loading context: {str(e)}"}

def print_task_analysis(flawed_task: Dict[str, Any]):
    """Print detailed analysis of a flawed task."""
    task_data = flawed_task['task_data']
    line_num = flawed_task['line_number']
    error_category = flawed_task.get('error_category', 'Unknown')
    priority = flawed_task.get('priority', 999)
    
    print("=" * 80)
    print(f"FLAWED TASK (Line {line_num}) - Priority {priority}")
    print("=" * 80)
    
    # Basic task info
    benchmark = task_data.get('benchmark', 'Unknown')
    task_name = task_data.get('task_name', 'Unknown')
    question_id = task_data.get('question_id', 'Unknown')
    
    print(f"Benchmark: {benchmark}")
    print(f"Task Name: {task_name}")
    print(f"Question ID: {question_id}")
    print(f"Issue Category: {error_category} (Priority: {priority})")
    print()
    
    # LLM-as-a-judge output
    llm_judge_output = task_data.get('llm_judge_output', {})
    specific_filter = llm_judge_output.get('specific_filter', {})
    
    print("LLM-as-a-Judge Analysis:")
    print(f"  Is Flawed: {specific_filter.get('is_flawed', 'N/A')}")
    print(f"  Error Category: {specific_filter.get('error_category', 'N/A')}")
    print(f"  Reasoning: {specific_filter.get('reasoning', 'N/A')}")
    print(f"  Reasoning Summary: {specific_filter.get('reasoning_summary', 'N/A')}")
    print()
    
    # Get context using bench loader
    print("Loading context using bench loader...")
    context = get_task_context(benchmark, task_name, question_id)
    
    if "error" in context:
        print(f"Error loading context: {context['error']}")
        return
    
    # Print context
    print("Task Context:")
    print("-" * 40)
    
    if context.get('instruction'):
        print(f"Instruction: {context['instruction']}")
        print()
    
    if context.get('previous_conversation_history'):
        print(f"Conversation History: {context['previous_conversation_history']}")
        print()
    
    if context.get('agent_system_prompt'):
        print(f"Agent System Prompt: {context['agent_system_prompt']}")
        print()
    
    if context.get('available_function_list'):
        print("Available Functions:")
        for i, func in enumerate(context['available_function_list'], 1):
            print(f"  {i}. {json.dumps(func, indent=4)}")
        print()
    
    if context.get('gt_conv_traj'):
        print(f"Ground Truth Trajectory: {json.dumps(context['gt_conv_traj'], indent=2)}")
        print()
    
    if context.get('time'):
        print(f"Time Context: {context['time']}")
        print()
    
    if context.get('meta'):
        print(f"Metadata: {json.dumps(context['meta'], indent=2)}")
        print()

def main():
    if len(sys.argv) != 2:
        print("Usage: python extract_flawed_tasks.py <results_file.jsonl>")
        sys.exit(1)
    
    results_file = sys.argv[1]
    
    if not os.path.exists(results_file):
        print(f"Error: Results file '{results_file}' not found")
        sys.exit(1)
    
    print(f"Loading flawed tasks from: {results_file}")
    flawed_tasks = load_flawed_tasks_from_results(results_file)
    
    print(f"Found {len(flawed_tasks)} flawed tasks")
    print()
    
    if not flawed_tasks:
        print("No flawed tasks found in the results file.")
        return
    
    # Show ordering summary
    print("Tasks are ordered by issue frequency (infrequent first):")
    print("Note: Excluding special_* task types (special_error_param, special_incomplete, special_irrelevant)")
    print("1. Vague instruction (1 occurrence)")
    print("2. Contradictory (1 occurrence)")
    print("3. Ground-Truth (2 occurrences)")
    print("4. Redundant/ungrounded function calls (3 occurrences)")
    print("5. Flawed function design (5 occurrences)")
    print("6. Insufficient toolsets (8 occurrences)")
    print("7. Unjustified/Hallucinated Parameters (14 occurrences)")
    print("8. Policy Violation (14 occurrences)")
    print("9. Incorrect function calls (71 occurrences)")
    print("10. Malformed function calls (162 occurrences)")
    print()
    
    # Process each flawed task and save to file
    output_file = "flawed_tasks_analysis.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, flawed_task in enumerate(flawed_tasks, 1):
            print(f"Processing flawed task {i}/{len(flawed_tasks)}...")
            
            # Capture the output
            import io
            import contextlib
            
            output_buffer = io.StringIO()
            with contextlib.redirect_stdout(output_buffer):
                print_task_analysis(flawed_task)
            
            task_output = output_buffer.getvalue()
            
            # Write to file
            f.write(task_output)
            f.write("\n" + "="*80 + "\n\n")
            
            # Also print to console
            print(task_output)
    
    print(f"\nAll {len(flawed_tasks)} flawed tasks have been saved to: {output_file}")

if __name__ == "__main__":
    main()
