#!/usr/bin/env python3
"""
BaseBenchmark Runner

A universal script to run any benchmark supported by BaseBenchmark framework.
"""

import argparse
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from tqdm import tqdm

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from basebenchmark import BFCLAdapter, BenchmarkConfig, UnifiedBenchmarkResult, EvaluationMetrics
import json
import logging

Benchmark_Adapter_Mapping = {
    "bfcl": BFCLAdapter,
}


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def get_available_benchmarks():
    """Get list of available benchmarks"""
    benchmarks = {
        "bfcl": {
            "name": "Berkeley Function Calling Leaderboard (BFCL)",
            "description": "Comprehensive function calling evaluation for LLMs",
            "adapter_class": "BFCLAdapter",
            "categories": [
                # Individual categories (with ground truth files) - RECOMMENDED
                "simple", "parallel", "multiple", "parallel_multiple",
                "java", "javascript", "live_simple", "live_multiple", "live_parallel", 
                "live_parallel_multiple", "multi_turn_base", "multi_turn_miss_func", 
                "multi_turn_miss_param", "multi_turn_long_context",
                # Combined categories (may cause errors due to missing ground truth files)
                # "all", "non_live", "live", "multi_turn"
            ]
        }
        # Add more benchmarks here as they are implemented
        # "toolsandbox": {
        #     "name": "ToolSandbox",
        #     "description": "Tool usage evaluation",
        #     "adapter_class": "ToolSandboxAdapter",
        #     "categories": ["basic", "advanced"]
        # }
    }
    return benchmarks


def list_available_benchmarks():
    """List available benchmarks"""
    print("Available Benchmarks:")
    print("=" * 50)
    
    benchmarks = get_available_benchmarks()
    
    for benchmark_id, info in benchmarks.items():
        print(f"{info['name']}")
        print(f"   ID: {benchmark_id}")
        print(f"   Description: {info['description']}")
        print(f"   Categories: {len(info['categories'])} available")
        print(f"   Categories: {', '.join(info['categories'])}")
        print()
    
    print("Usage:")
    print("   python run_benchmark.py --benchmark bfcl --list")
    print("   python run_benchmark.py --benchmark bfcl --model-name 'your-model' --api-key 'your-key'")


def build_adapter(benchmark_name: str, config: BenchmarkConfig):
    if benchmark_name not in Benchmark_Adapter_Mapping:
        raise NotImplementedError(f"Benchmark {benchmark_name} not yet implemented")
    return Benchmark_Adapter_Mapping[benchmark_name](config=config)


def process_single_task(task, adapter, config, task_index, total_tasks):
    """Process a single task with proper error handling"""
    try:
        # Convert task to dict format for adapter
        task_data = {
            "task_id": task.task_id,
            "test_category": task.category_type,
            "original_data": task.original_data
        }
        
        # Get response using adapter's method
        response = adapter.get_response_results(task_data)
        
        # Check if response is an error string
        if isinstance(response, str) and response.startswith("BFCL execution error"):
            # Handle error case
            error_metrics = EvaluationMetrics(
                accuracy=0.0,
                success_rate=0.0,
                task_specific_metrics={"error": response}
            )
            unified_result = adapter.create_unified_result(task, "", error_metrics, config)
            
            # Save error result immediately
            safe_model_name = config.model_name.replace("/", "_")
            results_file = f"{config.output_dir}/{safe_model_name}_{config.benchmark_name}.jsonl"
            os.makedirs(config.output_dir, exist_ok=True)
            
            with open(results_file, 'a') as f:
                json.dump(unified_result.__dict__, f, ensure_ascii=False, default=str)
                f.write('\n')
            
            return {
                "task_id": task.task_id,
                "success": False,
                "error": response,
                "result": unified_result
            }
        
        # Evaluate response
        eval_data = {
            "task_id": task.task_id,
            "test_category": task.category_type,
            "prompt": task.original_data,
            "possible_answer": task.ground_truth
        }
        metrics_dict = adapter.evaluate(response, eval_data)
        # Convert to EvaluationMetrics object
        metrics = EvaluationMetrics(
            accuracy=metrics_dict["accuracy"],
            success_rate=metrics_dict["valid"],
            task_specific_metrics=metrics_dict
        )
        
        # Create unified result format
        # Extract content and messages from response if it's a dict
        if isinstance(response, dict):
            response_content = response.get("content", "")
            messages = response.get("messages", [])
            # Store messages in task metadata for create_unified_result to use
            task.metadata["messages"] = messages
        else:
            response_content = response
        
        unified_result = adapter.create_unified_result(task, response_content, metrics, config)
        
        # Save unified result immediately
        safe_model_name = config.model_name.replace("/", "_")
        results_file = f"{config.output_dir}/{safe_model_name}_{config.benchmark_name}.jsonl"
        os.makedirs(config.output_dir, exist_ok=True)
        
        with open(results_file, 'a') as f:
            json.dump(unified_result.__dict__, f, ensure_ascii=False, default=str)
            f.write('\n')
        
        return {
            "task_id": task.task_id,
            "success": True,
            "accuracy": metrics.accuracy or 0.0,
            "result": unified_result
        }
        
    except Exception as e:
        # Create error result in unified format
        error_metrics = EvaluationMetrics(
            accuracy=0.0,
            success_rate=0.0,
            task_specific_metrics={"error": str(e)}
        )
        unified_result = adapter.create_unified_result(task, "", error_metrics, config)
        
        return {
            "task_id": task.task_id,
            "success": False,
            "error": str(e),
            "result": unified_result
        }


def run_benchmark(args):
    """Run specified benchmark"""
    print(f"Running {args.benchmark.upper()} Benchmark")
    print("=" * 50)
    
    # Set environment variables from command line arguments
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url
    
    # Create configuration
    config = BenchmarkConfig(
        model_name=args.model_name,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
        output_dir=args.output_dir
    )
    
    # Set benchmark-specific parameters
    config.benchmark_name = args.benchmark
    if args.benchmark == "bfcl":
        config.benchmark_params = {"test_category": [args.category] if args.category else ["simple"]}
    
    print(f"Configuration:")
    print(f"   Benchmark: {args.benchmark}")
    print(f"   Model: {config.model_name}")
    print(f"   Base URL: {config.base_url}")
    print(f"   Temperature: {config.temperature}")
    print(f"   Max Tokens: {config.max_tokens}")
    print(f"   Output Dir: {config.output_dir}")
    if args.category:
        print(f"   Category: {args.category}")
    print()
    
    # Create adapter
    print("Creating benchmark adapter...")
    adapter = build_adapter(args.benchmark, config)
    
    print("Adapter created successfully")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = adapter.load_dataset(config)
    
    # Limit tasks for demo
    if args.max_tasks and len(dataset.tasks) > args.max_tasks:
        dataset.tasks = dataset.tasks[:args.max_tasks]
        print(f"Limiting to {args.max_tasks} tasks for demo")
    
    print(f"Running {len(dataset.tasks)} tasks")
    
    # Run benchmark with multi-threading
    print("Starting benchmark execution with multi-threading...")
    
    # Determine number of workers
    max_workers = min(args.max_workers, len(dataset.tasks))
    print(f"Using {max_workers} workers")
    
    results = []
    successful_tasks = 0
    failed_tasks = 0
    
    # Create thread-safe lock for file writing
    file_lock = threading.Lock()
    
    # Use ThreadPoolExecutor for concurrent execution
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_single_task, task, adapter, config, i, len(dataset.tasks)): task
            for i, task in enumerate(dataset.tasks, 1)
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(dataset.tasks), desc="Processing tasks", unit="task") as pbar:
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result["result"])
                    
                    if result["success"]:
                        successful_tasks += 1
                        pbar.set_postfix({
                            "Success": successful_tasks,
                            "Failed": failed_tasks,
                            "Accuracy": f"{result['accuracy']:.3f}"
                        })
                    else:
                        failed_tasks += 1
                        pbar.set_postfix({
                            "Success": successful_tasks,
                            "Failed": failed_tasks,
                            "Error": result["error"][:20] + "..." if len(result["error"]) > 20 else result["error"]
                        })
                    
                except Exception as e:
                    failed_tasks += 1
                    pbar.set_postfix({
                        "Success": successful_tasks,
                        "Failed": failed_tasks,
                        "Error": str(e)[:20] + "..." if len(str(e)) > 20 else str(e)
                    })
                
                pbar.update(1)
    
    print()
    print("Benchmark completed!")
    
    # Calculate final KPI
    safe_model_name = config.model_name.replace("/", "_")
    results_file = f"{config.output_dir}/{safe_model_name}_{config.benchmark_name}.jsonl"
    kpi = adapter.get_final_kpi(results_file)
    
    print("Final Results:")
    print(f"   Total Tasks: {kpi['total_tasks']}")
    print(f"   Valid Tasks: {kpi['valid_tasks']}")
    print(f"   Average Accuracy: {kpi['average_accuracy']:.3f}")
    print(f"   Success Rate: {kpi['success_rate']:.3f}")
    print(f"   Error Breakdown: {kpi['error_breakdown']}")
    
    return True


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="BaseBenchmark Universal Runner")
    
    # Benchmark selection
    parser.add_argument("--benchmark", help="Benchmark to run (e.g., bfcl)")
    
    # Basic arguments
    parser.add_argument("--api-key", default="", help="API key for the model")
    parser.add_argument("--base-url", default="", help="Base URL for the API")
    parser.add_argument("--model-name", default="", help="Model name to test")
    
    # Optional arguments
    parser.add_argument("--temperature", type=float, default=0.001, help="Temperature for generation")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Maximum tokens to generate")
    parser.add_argument("--max-tasks", type=int, help="Maximum number of tasks to run (for demo)")
    parser.add_argument("--max-workers", type=int, default=8, help="Maximum number of worker threads")
    parser.add_argument("--output-dir", default="./results", help="Output directory for results")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    parser.add_argument("--category", help="Benchmark category to run (e.g., simple, java for BFCL)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # List benchmarks if requested
    if args.list:
        list_available_benchmarks()
        return

    # Validate benchmark is provided
    if not args.benchmark:
        print("Benchmark must be specified with --benchmark")
        print("Available benchmarks:")
        available_benchmarks = get_available_benchmarks()
        for benchmark_id in available_benchmarks.keys():
            print(f"   - {benchmark_id}")
        print("\nUse --list to see detailed information about available benchmarks")
        sys.exit(1)

    # Validate benchmark exists
    available_benchmarks = get_available_benchmarks()
    if args.benchmark not in available_benchmarks:
        print(f"Unknown benchmark: {args.benchmark}")
        print("Available benchmarks:")
        for benchmark_id in available_benchmarks.keys():
            print(f"   - {benchmark_id}")
        sys.exit(1)

    # Run benchmark
    success = run_benchmark(args)
    
    if success:
        print(f"\n{args.benchmark.upper()} benchmark completed successfully!")
    else:
        print(f"\n{args.benchmark.upper()} benchmark failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
