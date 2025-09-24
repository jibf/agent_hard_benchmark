#!/usr/bin/env python3
"""
Review non-flawed tasks using LLM to criticize the analysis.
"""

import json
import sys
import os
import re
import random
from typing import Dict, List, Any
from openai import OpenAI
import time
from dotenv import load_dotenv
from tqdm import tqdm
from multiprocessing import Pool

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.bench_loaders import get_bench_loader
from src.utils.types import Benchmark, UniqueQuestionID

load_dotenv()

def load_non_flawed_tasks_from_results(results_file: str) -> List[Dict[str, Any]]:
    """Load tasks with is_flawed=False from the results JSONL file."""
    non_flawed_tasks = []
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                task_data = json.loads(line.strip())
                
                # Check if this task is NOT flawed
                llm_judge_output = task_data.get('llm_judge_output', {})
                specific_filter = llm_judge_output.get('specific_filter', {})
                
                if not specific_filter.get('is_flawed', True):  # Default to True if not specified
                    # Skip special_* task types
                    task_name = task_data.get('task_name', '')
                    if task_name.startswith('special_'):
                        continue
                    
                    non_flawed_tasks.append({
                        'line_number': line_num,
                        'task_data': task_data
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return non_flawed_tasks

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

def format_task_context_for_prompt(context: Dict[str, Any], llm_judge_output: Dict[str, Any]) -> str:
    """Format the task context and LLM judge output for the review prompt."""
    
    prompt_parts = []
    
    # Basic task info
    prompt_parts.append(f"Task: {context['task_name']}")
    prompt_parts.append(f"Question ID: {context['question_id']}")
    prompt_parts.append(f"Benchmark: {context['benchmark']}")
    prompt_parts.append("")
    
    # Instruction
    if context.get('instruction'):
        prompt_parts.append(f"Instruction: {context['instruction']}")
        prompt_parts.append("")
    
    # Conversation history
    if context.get('previous_conversation_history'):
        prompt_parts.append(f"Conversation History: {context['previous_conversation_history']}")
        prompt_parts.append("")
    
    # Agent system prompt
    if context.get('agent_system_prompt'):
        prompt_parts.append(f"Agent System Prompt: {context['agent_system_prompt']}")
        prompt_parts.append("")
    
    # Available functions
    if context.get('available_function_list'):
        prompt_parts.append("Available Functions:")
        for i, func in enumerate(context['available_function_list'], 1):
            prompt_parts.append(f"  {i}. {json.dumps(func, indent=4)}")
        prompt_parts.append("")
    
    # Ground truth trajectory
    if context.get('gt_conv_traj'):
        prompt_parts.append(f"Ground Truth Trajectory: {json.dumps(context['gt_conv_traj'], indent=2)}")
        prompt_parts.append("")
    
    # Time context
    if context.get('time'):
        prompt_parts.append(f"Time Context: {context['time']}")
        prompt_parts.append("")
    
    # Metadata
    if context.get('meta'):
        prompt_parts.append(f"Metadata: {json.dumps(context['meta'], indent=2)}")
        prompt_parts.append("")
    
    # LLM Judge Analysis
    specific_filter = llm_judge_output.get('specific_filter', {})
    prompt_parts.append("LLM-as-a-Judge Analysis:")
    prompt_parts.append(f"  Is Flawed: {specific_filter.get('is_flawed', 'N/A')}")
    prompt_parts.append(f"  Error Category: {specific_filter.get('error_category', 'N/A')}")
    prompt_parts.append(f"  Reasoning: {specific_filter.get('reasoning', 'N/A')}")
    prompt_parts.append(f"  Reasoning Summary: {specific_filter.get('reasoning_summary', 'N/A')}")
    prompt_parts.append("")
    
    # Review prompt
    prompt_parts.append("Review the LLM model's analysis of saying this task doesn't have issues and criticize it. If the task actually has issues, explain why, if not state that this task doesn't have issues.")
    
    return "\n".join(prompt_parts)

def _extract_json_from_response(content: str) -> dict:
    """Extract JSON from response content that might be wrapped in code blocks."""
    # Try to parse as is first
    try:
        return json.loads(content.strip())
    except json.JSONDecodeError:
        pass

    # Look for JSON in code blocks
    json_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Look for JSON without code blocks
    json_pattern = r'(\{.*?\})'
    match = re.search(json_pattern, content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from response: {content}")

def make_llm_call(prompt: str, max_retries: int = 3, retry_delay: float = 1.0) -> str:
    """Make an LLM API call to review the task using the same setup as llm_judge_filtering.py."""
    client = OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL")
    )
    
    # model = "openai/gpt-4.1"
    model = "google/gemini-2.5-pro-thinking-on"
    is_gemini = "gemini" in model
    
    for attempt in range(max_retries):
        try:
            params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }

            # For gemini models, don't use extra_body or response_format as they cause 500 errors
            if not is_gemini:
                params["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**params)

            response_content = response.choices[0].message.content
            if not response_content or response_content.strip() == "":
                raise ValueError("Empty response from API")

            # For gemini models, we expect plain text response, not JSON
            return response_content.strip()

        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                return f"Error: {str(e)}"
            time.sleep(retry_delay)
    
    return "Error: Max retries exceeded"

def _review_task_worker(args):
    """Worker function for multiprocessing task review."""
    task_data, line_num, benchmark, task_name, question_id, model, api_key, base_url, max_retries, retry_delay = args
    
    # Get context using bench loader
    context = get_task_context(benchmark, task_name, question_id)
    
    if "error" in context:
        return {
            "line_number": line_num,
            "benchmark": benchmark,
            "task_name": task_name,
            "question_id": question_id,
            "error": context["error"],
            "llm_review": "Context loading failed"
        }
    
    # Get LLM judge output
    llm_judge_output = task_data.get('llm_judge_output', {})
    
    # Format prompt
    review_prompt = format_task_context_for_prompt(context, llm_judge_output)
    
    # Make LLM call
    client = OpenAI(api_key=api_key, base_url=base_url)
    llm_review = make_llm_call_with_client(client, model, review_prompt, max_retries, retry_delay)
    
    # Create result object
    result = {
        "line_number": line_num,
        "benchmark": benchmark,
        "task_name": task_name,
        "question_id": question_id,
        "context": context,
        "llm_judge_output": llm_judge_output,
        "llm_review": llm_review
    }
    
    return result

def make_llm_call_with_client(client: OpenAI, model: str, prompt: str, max_retries: int = 3, retry_delay: float = 1.0) -> str:
    """Make an LLM API call using an existing client."""
    is_gemini = "gemini" in model
    
    for attempt in range(max_retries):
        try:
            params = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            }

            # For gemini models, don't use extra_body or response_format as they cause 500 errors
            if not is_gemini:
                params["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**params)

            response_content = response.choices[0].message.content
            if not response_content or response_content.strip() == "":
                raise ValueError("Empty response from API")

            # For gemini models, we expect plain text response, not JSON
            return response_content.strip()

        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                return f"Error: {str(e)}"
            time.sleep(retry_delay)
    
    return "Error: Max retries exceeded"

def review_non_flawed_tasks(results_file: str, output_file: str, sample_size: int = 80, num_proc: int = 4):
    """Review a random sample of non-flawed tasks and save results to JSONL."""
    
    print(f"Loading non-flawed tasks from: {results_file}")
    non_flawed_tasks = load_non_flawed_tasks_from_results(results_file)
    
    print(f"Found {len(non_flawed_tasks)} non-flawed tasks")
    
    if not non_flawed_tasks:
        print("No non-flawed tasks found in the results file.")
        return
    
    # Randomly sample tasks
    if len(non_flawed_tasks) > sample_size:
        print(f"Randomly sampling {sample_size} tasks from {len(non_flawed_tasks)} available tasks")
        random.seed(42)  # For reproducibility
        non_flawed_tasks = random.sample(non_flawed_tasks, sample_size)
    else:
        print(f"Using all {len(non_flawed_tasks)} available tasks (less than requested sample size)")
    
    print(f"Reviewing {len(non_flawed_tasks)} non-flawed tasks using {num_proc} processes")
    print()
    
    # Prepare arguments for multiprocessing
    # model = "openai/gpt-4.1"
    model = "google/gemini-2.5-pro-thinking-on"
    api_key = os.getenv("API_KEY")
    base_url = os.getenv("BASE_URL")
    max_retries = 3
    retry_delay = 1.0
    
    args_list = []
    for non_flawed_task in non_flawed_tasks:
        task_data = non_flawed_task['task_data']
        line_num = non_flawed_task['line_number']
        benchmark = task_data.get('benchmark', 'Unknown')
        task_name = task_data.get('task_name', 'Unknown')
        question_id = task_data.get('question_id', 'Unknown')
        
        args_list.append((
            task_data, line_num, benchmark, task_name, question_id,
            model, api_key, base_url, max_retries, retry_delay
        ))
    
    # Process tasks using multiprocessing
    results = []
    if num_proc == 1:  # Single process
        for args in tqdm(args_list, desc="Reviewing non-flawed tasks"):
            result = _review_task_worker(args)
            results.append(result)
    else:  # Multiprocessing
        with Pool(processes=num_proc) as pool:
            with tqdm(total=len(args_list), desc="Reviewing non-flawed tasks (multiprocessing)") as pbar:
                for result in pool.imap(_review_task_worker, args_list):
                    results.append(result)
                    pbar.update(1)
    
    # Write results to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"All {len(non_flawed_tasks)} non-flawed task reviews have been saved to: {output_file}")

def main():
    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python review_non_flawed_tasks.py <results_file.jsonl> [sample_size] [num_processes]")
        print("Default sample_size is 80, default num_processes is 4")
        sys.exit(1)
    
    results_file = sys.argv[1]
    sample_size = int(sys.argv[2]) if len(sys.argv) >= 3 else 80
    num_proc = int(sys.argv[3]) if len(sys.argv) == 4 else 4
    num_proc = 40
    
    if not os.path.exists(results_file):
        print(f"Error: Results file '{results_file}' not found")
        sys.exit(1)
    
    output_file = f"non_flawed_tasks_review_sample_{sample_size}.jsonl"
    
    review_non_flawed_tasks(results_file, output_file, sample_size, num_proc)

if __name__ == "__main__":
    main()
