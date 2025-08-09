from typing import Any, Callable, List, Dict, Tuple

from os import getenv

import statistics

import time

import json
import os
from pathlib import Path
from datetime import datetime

from traceback import format_exc, print_exc

from functools import wraps

from concurrent.futures import ThreadPoolExecutor, as_completed

from tabulate import tabulate

from tqdm import tqdm

# ---------------------------------------------------------------------------
# JSON SERIALIZATION UTILITIES
# ---------------------------------------------------------------------------
# Some benchmark tools may legitimately return values that the standard
# json module cannot natively serialise (for example complex numbers or
# custom classes).  When those values are included in the intermediate
# context that we store or forward to the model, a naive json.dumps call
# will raise a `TypeError: Object of type X is not JSON serializable` and
# break the whole benchmark run.  We provide a drop-in replacement JSON
# encoder that gracefully degrades by falling back to `str(obj)` for any
# value it does not understand.  We then patch the json module once
# (early in `entrypoint.py`) so that *all* subsequent dumps use this safe
# encoder without having to touch every call individually.

class SafeJSONEncoder(json.JSONEncoder):
    """A JSONEncoder that is tolerant to exotic Python objects.

    * complex -> {"real": <float>, "imag": <float>} (round-trippable)
    * anything else -> str(obj)
    """

    def default(self, obj):  # pylint: disable=method-hidden
        # Handle native complex numbers explicitly so that information is not
        # lost while still remaining JSON serialisable.
        if isinstance(obj, complex):
            return {"real": obj.real, "imag": obj.imag}
        # Fall back to default behaviour or convert to string if that fails.
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def _dumps_safe(obj, *args, **kwargs):
    """Replacement for json.dumps that always uses SafeJSONEncoder."""
    kwargs.setdefault("cls", SafeJSONEncoder)
    return _json_dumps_orig(obj, *args, **kwargs)


def _dump_safe(obj, fp, *args, **kwargs):  # type: ignore[override]
    """Replacement for json.dump that always uses SafeJSONEncoder."""
    kwargs.setdefault("cls", SafeJSONEncoder)
    return _json_dump_orig(obj, fp, *args, **kwargs)


# Keep original references so we can call them from our wrappers
_json_dumps_orig = json.dumps
_json_dump_orig = json.dump


def patch_json_serialization() -> None:
    """Globally patch json.dump/json.dumps to use SafeJSONEncoder.

    This should be called exactly once at application start-up (see
    `entrypoint.py`).  After patching, *any* call to json.dump(s) made by
    NexusBench or third-party libraries within the same process will gain
    the enhanced serialisation capability automatically.
    """
    # Idempotency guard – if we've already patched, bail out.
    if json.dumps is _dumps_safe and json.dump is _dump_safe:
        return

    json.dumps = _dumps_safe  # type: ignore[assignment]
    json.dump = _dump_safe  # type: ignore[assignment]

# ---------------------------------------------------------------------------


DEFAULT_MAX_EXECUTION_RETRIES = 1
EXECUTION_RETRY_DELAY = 1


def handle_exceptions(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_execution_retries = getenv("MAX_EXECUTION_RETRIES")
        if max_execution_retries is not None:
            max_execution_retries = json.loads(max_execution_retries)
        else:
            max_execution_retries = DEFAULT_MAX_EXECUTION_RETRIES

        debug = json.loads(getenv("DEBUG", "false"))

        retry_delay = EXECUTION_RETRY_DELAY
        for attempt in range(max_execution_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                e_str = format_exc() if debug else str(e)
                if attempt < max_execution_retries - 1:
                    print(
                        f"Attempt {attempt + 1} failed: {e_str}. Retrying in {retry_delay} seconds..."
                    )
                    time.sleep(retry_delay)
                else:
                    print(f"Max retries reached. Last error: {e_str}")
                    # pylint: disable=raise-missing-from
                    raise ValueError("Max Retries Reached")

    return wrapper


def parallelize():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            max_workers = json.loads(getenv("NUM_SAMPLES_PARALLEL"))
            debug = json.loads(getenv("DEBUG"))

            futures = []
            total = len(args[1])
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for arg in args[1]:
                    future = executor.submit(func, arg)
                    futures.append(future)

                results = []
                with tqdm(total=total, desc=f"Processing {func.__name__}") as pbar:
                    for future, arg in zip(as_completed(futures), args[1]):
                        try:
                            results.append(future.result())
                        except Exception as e:
                            if debug:
                                print_exc()
                            else:
                                print(e)
                            results.append(e)
                        finally:
                            pbar.update(1)
            return results

        return wrapper

    return decorator


def save_benchmark_results_to_files(results: List[Tuple], output_dir: str, model_name: str, client_name: str):
    """
    Save benchmark results to organized local files.
    
    Args:
        results: List of (benchmark_class, metrics, correct_calls, time_taken_s) tuples
        output_dir: Directory to save results
        model_name: Name of the model used
        client_name: Name of the client used
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create subdirectories for better organization
    results_dir = output_path / "results"
    metrics_dir = output_path / "metrics"
    trajectories_dir = output_path / "trajectories"
    
    for dir_path in [results_dir, metrics_dir, trajectories_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # Save overall summary
    summary_data = {
        "timestamp": timestamp,
        "model": model_name,
        "client": client_name,
        "total_benchmarks": len(results),
        "benchmarks": []
    }
    
    for benchmark_class, metrics, correct_calls, time_taken_s in results:
        benchmark_name = benchmark_class.__name__
        
        # Calculate statistics
        total_samples = len(correct_calls)
        successful_samples = sum(1 for call in correct_calls if not isinstance(call, Exception))
        error_samples = total_samples - successful_samples
        
        benchmark_summary = {
            "benchmark_name": benchmark_name,
            "metrics": metrics,
            "time_taken_s": time_taken_s,
            "total_samples": total_samples,
            "successful_samples": successful_samples,
            "error_samples": error_samples,
            "success_rate": successful_samples / total_samples if total_samples > 0 else 0
        }
        
        summary_data["benchmarks"].append(benchmark_summary)
        
        # Save detailed results for each benchmark
        benchmark_data = {
            "benchmark_name": benchmark_name,
            "model": model_name,
            "client": client_name,
            "timestamp": timestamp,
            "summary": benchmark_summary,
            "detailed_results": []
        }
        
        # Process each sample result
        for i, call in enumerate(correct_calls):
            if isinstance(call, Exception):
                sample_result = {
                    "sample_index": i,
                    "error": str(call),
                    "final_accuracy": False,
                    "has_trajectory": False
                }
            else:
                # Extract trajectory information
                context = call.get("context", [])
                trajectory = []
                for ctx in context:
                    if isinstance(ctx, dict):
                        trajectory.append({
                            "prompt": ctx.get("prompt", ""),
                            "query": ctx.get("query", ""),
                            "raw_model_call": ctx.get("raw_model_call", ""),
                            "model_call": ctx.get("model_call", ""),
                            "result": ctx.get("result", ""),
                            "timestamp": ctx.get("timestamp", "")
                        })
                
                sample_result = {
                    "sample_index": i,
                    "final_accuracy": call.get("Final Accuracy", False),
                    "ground_truth": call.get("ground_truth", ""),
                    "max_turns_hit": call.get("Max Turns Hit", False),
                    "invalid_plan": call.get("Invalid Plan", False),
                    "has_trajectory": len(trajectory) > 0,
                    "trajectory_length": len(trajectory)
                }
                
                # Save trajectory separately if it exists
                if trajectory:
                    trajectory_file = trajectories_dir / f"{model_name}_{benchmark_name}_sample_{i}_{timestamp}.json"
                    with open(trajectory_file, 'w') as f:
                        json.dump({
                            "benchmark_name": benchmark_name,
                            "sample_index": i,
                            "model": model_name,
                            "timestamp": timestamp,
                            "trajectory": trajectory
                        }, f, indent=2, default=str)
            
            benchmark_data["detailed_results"].append(sample_result)
        
        # Save benchmark results
        benchmark_file = results_dir / f"{model_name}_{benchmark_name}_{timestamp}.json"
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2, default=str)
        
        # Save metrics separately
        metrics_file = metrics_dir / f"{model_name}_{benchmark_name}_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                "benchmark_name": benchmark_name,
                "model": model_name,
                "timestamp": timestamp,
                "metrics": metrics,
                "time_taken_s": time_taken_s,
                "sample_count": total_samples
            }, f, indent=2, default=str)
        
        print(f"Saved {benchmark_name} results to {benchmark_file}")
        print(f"Saved {benchmark_name} metrics to {metrics_file}")
    
    # Save overall summary
    summary_file = output_path / f"{model_name}_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    # Save metrics table as CSV
    csv_file = output_path / f"{model_name}_metrics_{timestamp}.csv"
    with open(csv_file, 'w') as f:
        f.write("Benchmark,Accuracy,Max Turns Hit,Invalid Plan,Time (s),Total Samples,Successful Samples,Error Samples,Success Rate\n")
        for benchmark_class, metrics, correct_calls, time_taken_s in results:
            benchmark_name = benchmark_class.__name__
            accuracy = metrics.get("Accuracy", 0)
            max_turns_hit = metrics.get("Max Turns Hit", 0)
            invalid_plan = metrics.get("Invalid Plan", 0)
            total_samples = len(correct_calls)
            successful_samples = sum(1 for call in correct_calls if not isinstance(call, Exception))
            error_samples = total_samples - successful_samples
            success_rate = successful_samples / total_samples if total_samples > 0 else 0
            
            f.write(f"{benchmark_name},{accuracy:.4f},{max_turns_hit:.4f},{invalid_plan:.4f},{time_taken_s:.2f},{total_samples},{successful_samples},{error_samples},{success_rate:.4f}\n")
    
    print(f"Saved summary results to {summary_file}")
    print(f"Saved metrics CSV to {csv_file}")
    print(f"Results saved to directory: {output_path}")


def print_benchmark_results(accuracies: List[Tuple[str, Dict[str, float], Any, float]]):
    table_data = []
    all_metrics = set()
    for _, metrics, _, _ in accuracies:
        all_metrics.update(metrics.keys())

    # Sort metrics to ensure consistent order
    sorted_metrics = sorted(all_metrics)

    # Prepare headers
    headers = ["Benchmark"] + sorted_metrics + ["Time taken (s)"]

    # Populate table data
    for name, metrics, _, time_taken_s in accuracies:
        row = [name.__name__]
        for metric in sorted_metrics:
            value = metrics.get(metric, "N/A")
            row.append(f"{value:.2%}" if isinstance(value, float) else value)
        row.append(f"{time_taken_s:.0f}")
        table_data.append(row)

    # Calculate and add average row
    avg_row = ["Average"]
    for metric in sorted_metrics:
        values = [
            metrics[metric] for _, metrics, _, _ in accuracies if metric in metrics
        ]
        if values:
            avg_value = statistics.mean(values)
            avg_row.append(f"{avg_value:.2%}")
        else:
            avg_row.append("N/A")

        times_taken_s = [
            time_taken_s for _, _, _, time_taken_s in accuracies if metric in metrics
        ]
        avg_time_taken_s = f"{statistics.mean(times_taken_s):.0f}"
        avg_row.append(avg_time_taken_s)

    table_data.append(avg_row)

    # Print the results in a pretty table
    print("\nBenchmark Results:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
