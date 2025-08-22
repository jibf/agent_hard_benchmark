"""
BFCL Adapter for Unified Framework

This adapter implements the BenchmarkAdapter interface for BFCL benchmark.
"""

import json
import os
import subprocess
import sys
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import ast

from ..core.models import (
    BenchmarkType, BenchmarkTask, 
    EvaluationMetrics, BenchmarkDataset, BenchmarkConfig, UnifiedBenchmarkResult
)

# Add BFCL path to sys.path to handle the hyphen in folder name
bfcl_root = None
possible_paths = [
    Path(__file__).parent.parent / "gorilla" / "berkeley-function-call-leaderboard",
    Path.cwd() / "gorilla" / "berkeley-function-call-leaderboard",
    Path.home() / "gorilla" / "berkeley-function-call-leaderboard"
]

for path in possible_paths:
    if path.exists() and (path / "bfcl_eval").exists():
        bfcl_root = path
        break

if bfcl_root:
    # Add BFCL root to Python path
    sys.path.insert(0, str(bfcl_root))
    
    # Now we can import BFCL modules directly
    from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
    from bfcl_eval._llm_response_generation import get_involved_test_entries, process_multi_turn_test_case, sort_key
    from bfcl_eval.utils import parse_test_category_argument, extract_test_category_from_id
    from bfcl_eval.eval_checker.eval_runner import bfcl_evaluator
    # Import BFCL evaluation modules
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker
    from bfcl_eval.eval_checker.eval_runner import is_multi_turn, is_relevance_or_irrelevance
    from bfcl_eval.eval_checker.eval_runner import is_empty_execute_response, is_empty_output
    from bfcl_eval.eval_checker.eval_runner import is_java, is_js
    from bfcl_eval.eval_checker.eval_runner import is_function_calling_format_output
    from bfcl_eval.constants.eval_config import POSSIBLE_ANSWER_PATH
    from bfcl_eval.utils import find_file_with_suffix, load_file
else:
    raise ImportError("BFCL root directory not found. Please ensure BFCL is properly installed.")

class BFCLAdapter:
    """
    BFCL (Berkeley Function Calling Leaderboard) adapter
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.results_file = None
        self.completed_tasks = set()
        self.config = config
        self.model_name = config.model_name
        self.temperature = config.temperature
        # Create handler with config parameters
        self.handler = OpenAICompletionsHandler(model_name=config.model_name, temperature=config.temperature)
        
    @property
    def benchmark_type(self) -> BenchmarkType:
        return BenchmarkType.BFCL
    
    @property
    def name(self) -> str:
        return "bfcl"
    
    @property
    def version(self) -> str:
        return "v3"
    
    def load_dataset(self, config: BenchmarkConfig) -> BenchmarkDataset:
        """
        Load BFCL dataset and filter completed tasks
        """
        self.results_file = f"unified_results/{config.model_name}_bfcl.jsonl"
        
        # Create results directory
        os.makedirs("unified_results", exist_ok=True)
        
        # Load completed tasks
        self._load_completed_tasks()
        
        test_category = config.benchmark_params.get("test_category", ["simple"])
        
        (
            all_test_file_paths,
            all_test_categories,
            all_test_entries_involved,
        ) = get_involved_test_entries(test_category, False)

        test_cases_to_generate = [
            test_case
            for test_case in all_test_entries_involved
            if test_case["id"] not in self.completed_tasks
        ]
        test_cases_to_generate = process_multi_turn_test_case(test_cases_to_generate)

        test_cases_total = sorted(test_cases_to_generate, key=sort_key)

        # Load ground truth for each category
        ground_truth_by_category = {}
        for test_category in all_test_categories:
            possible_answer_file = find_file_with_suffix(POSSIBLE_ANSWER_PATH, test_category)
            ground_truth_data = load_file(possible_answer_file, sort_by_id=True)
            ground_truth_by_category[test_category] = ground_truth_data

        # Convert to BenchmarkTask format
        tasks = []
        for test_case in test_cases_total:
            test_category = extract_test_category_from_id(test_case["id"])
            
            # Find corresponding ground truth
            ground_truth = None
            if test_category in ground_truth_by_category:
                for gt_entry in ground_truth_by_category[test_category]:
                    if gt_entry["id"] == test_case["id"]:
                        ground_truth = gt_entry
                        break
            
            task = BenchmarkTask(
                task_id=test_case["id"],
                benchmark_type=BenchmarkType.BFCL,
                category_type=test_category,
                original_data=test_case,
                ground_truth=ground_truth,
                metadata={
                    "is_multi_turn": "multi_turn" in test_category
                }
            )
            tasks.append(task)
        
        return BenchmarkDataset(
            name="BFCL Dataset",
            benchmark_type=BenchmarkType.BFCL,
            tasks=tasks,
            metadata={
                "total_count": len(all_test_entries_involved),
                "pending_count": len(tasks)
            }
        )
    
    def get_response_results(self, data: Dict) -> Dict:
        """
        Get final response for one data, handle all interactions internally
        
        Args:
            data: Task data
            
        Returns:
            Complete response data with messages
        """
        task_id = data["task_id"]
        test_category = data["test_category"]
        
        # Run BFCL generation
        try:
            # Get original data from metadata
            original_data = data["original_data"]
            
            result, metadata = self.handler.inference(
                deepcopy(original_data), True, False
            )

            # Extract messages from the inference process
            messages = self._extract_messages_from_bfcl(original_data, result, metadata)
            
            result_to_write = {
                "id": task_id,
                "content": result,
                "success": True,
                "test_category": test_category,
                "messages": messages
            }
            
            return result_to_write
            
        except Exception as e:
            return {
                "id": task_id,
                "content": f"Error during inference: {str(e)}",
                "success": False,
                "test_category": test_category,
                "messages": []
            }
    
    def _extract_messages_from_bfcl(self, original_data: Dict, result: str, metadata: Dict = None) -> List[Dict]:
        """Extract messages from BFCL inference process"""
        # Try to get complete messages from metadata
        if metadata and "inference_log" in metadata:
            for log_entry in metadata["inference_log"]:
                if log_entry["role"] == "inference_input":
                    content = log_entry["content"]
                    if "message" in content:
                        # Direct string to list conversion
                        message_list = ast.literal_eval(content["message"])
                        messages = [{"role": msg["role"], "content": msg["content"]} 
                                  for msg in message_list]
                        
                        # Add assistant response
                        if result:
                            messages.append({"role": "assistant", "content": result})
                        return messages
        
        # Fallback: construct from original data
        messages = []
        for turn_messages in original_data["question"]:
            for message in turn_messages:
                if message["role"] == "user":
                    messages.append({"role": "user", "content": message["content"]})
        
        if result:
            messages.append({"role": "assistant", "content": result})
        
        return messages
    
    def get_model_prompt(self, task: BenchmarkTask, config: BenchmarkConfig) -> str:
        """Generate model prompt for BFCL task"""
        # Extract prompt from original data
        return task.original_data.get("prompt", "N/A")
    
    def evaluate_task(self, task: BenchmarkTask, response: str, config: BenchmarkConfig) -> EvaluationMetrics:
        """Evaluate a single BFCL task"""
        # Convert task to data format expected by evaluate method
        data = {
            "task_id": task.task_id,
            "test_category": task.category_type,
            "prompt": task.original_data,
            "possible_answer": task.ground_truth,
            "ground_truth": task.ground_truth
        }
        
        response_data = {
            "content": response
        }
        
        metrics = self.evaluate(response_data, data)
        
        return EvaluationMetrics(
            accuracy=metrics["accuracy"],
            success_rate=metrics["valid"],
            metadata=metrics
        )
    
    def evaluate(self, response: Dict, data: Dict) -> Dict[str, float]:
        """
        Evaluate a single response and return metrics
        
        Args:
            response: Model response data
            data: Original task data
            
        Returns:
            Evaluation metrics
        """
        # Extract data
        task_id = data["task_id"]
        test_category = data["test_category"]
        model_response = response["content"]
        prompt_data = data["prompt"]
        possible_answer = data["possible_answer"]

        
        # Determine language
        language = "Python"
        if is_java(test_category):
            language = "Java"
        elif is_js(test_category):
            language = "JavaScript"
        
        # Initialize metrics
        metrics = {
            "accuracy": 0.0,
            "valid": False,
            "error_type": None,
            "error_message": None,
            "test_category": test_category,
            "task_id": task_id
        }
        
        # Handle different test categories
        if is_relevance_or_irrelevance(test_category):
            # Relevance/Irrelevance test
            contain_func_call = False
            try:
                decoded_result = self.handler.decode_ast(model_response, language)
                contain_func_call = True
                if is_empty_output(decoded_result):
                    contain_func_call = False
            except Exception as e:
                contain_func_call = False
            
            # For relevance test, model should output function call
            # For irrelevance test, model should NOT output function call
            if test_category == "relevance":
                metrics["valid"] = contain_func_call
                metrics["accuracy"] = 1.0 if contain_func_call else 0.0
            else:  # irrelevance
                metrics["valid"] = not contain_func_call
                metrics["accuracy"] = 1.0 if not contain_func_call else 0.0
            
            if not metrics["valid"]:
                metrics["error_type"] = "relevance_checker:wrong_output"
                metrics["error_message"] = f"Expected {'function call' if test_category == 'relevance' else 'no function call'}, got {'function call' if contain_func_call else 'no function call'}"
        
        elif is_multi_turn(test_category):
            # Multi-turn test
            try:
                # Parse multi-turn response
                if isinstance(model_response, str):
                    # Try to parse as JSON if it's a string
                    multi_turn_response = json.loads(model_response)
                else:
                    multi_turn_response = model_response
                
                # Decode multi-turn response
                multi_turn_decoded = []
                for turn_response in multi_turn_response:
                    turn_decoded = []
                    for step_response in turn_response:
                        try:
                            decoded_result = self.handler.decode_execute(step_response)
                            if not is_empty_execute_response(decoded_result):
                                turn_decoded.append(decoded_result)
                        except Exception:
                            continue
                    multi_turn_decoded.append(turn_decoded)
                
                # Get ground truth
                ground_truth = possible_answer.get("ground_truth", [])
                
                # Check multi-turn correctness
                checker_result = multi_turn_checker(
                    multi_turn_decoded,
                    ground_truth,
                    prompt_data,
                    test_category,
                    self.model_name
                )
                
                metrics["valid"] = checker_result.get("valid", False)
                metrics["accuracy"] = 1.0 if metrics["valid"] else 0.0
                
                if not metrics["valid"]:
                    metrics["error_type"] = checker_result.get("error_type", "multi_turn:unknown_error")
                    metrics["error_message"] = checker_result.get("error_message", "Multi-turn evaluation failed")
            
            except Exception as e:
                metrics["valid"] = False
                metrics["accuracy"] = 0.0
                metrics["error_type"] = "multi_turn:parsing_error"
                metrics["error_message"] = str(e)
        
        else:
            # Single-turn test
            try:
                # Decode AST
                model_result_decoded = self.handler.decode_ast(model_response, language)
                
                # Check if output is in correct format
                if not is_function_calling_format_output(model_result_decoded):
                    metrics["valid"] = False
                    metrics["accuracy"] = 0.0
                    metrics["error_type"] = "ast_decoder:wrong_output_format"
                    metrics["error_message"] = "Model output is not in function calling format"
                else:
                    # Get function description and ground truth
                    func_description = prompt_data["function"]
                    ground_truth = possible_answer["ground_truth"]
                    
                    # Check correctness
                    checker_result = ast_checker(
                        func_description,
                        model_result_decoded,
                        ground_truth,
                        language,
                        test_category,
                        self.model_name
                    )
                    
                    metrics["valid"] = checker_result.get("valid", False)
                    metrics["accuracy"] = 1.0 if metrics["valid"] else 0.0
                    
                    if not metrics["valid"]:
                        metrics["error_type"] = checker_result.get("error_type", "ast_checker:unknown_error")
                        metrics["error_message"] = checker_result.get("error", ["Unknown error"])
            
            except Exception as e:
                metrics["valid"] = False
                metrics["accuracy"] = 0.0
                metrics["error_type"] = "ast_decoder:decoder_failed"
                metrics["error_message"] = str(e)
        
        return metrics
    
    def create_unified_result(
        self, 
        task: BenchmarkTask, 
        response: str, 
        metrics: Union[EvaluationMetrics, Dict],
        config: BenchmarkConfig
    ) -> UnifiedBenchmarkResult:
        """Create unified benchmark result format for BFCL"""
        
        # Handle both EvaluationMetrics object and dict
        if isinstance(metrics, dict):
            score = metrics["accuracy"]
            accuracy = metrics["accuracy"]
            success_rate = metrics["valid"]
            task_specific_metrics = metrics
        else:
            score = metrics.accuracy
            accuracy = metrics.accuracy
            success_rate = metrics.success_rate
            task_specific_metrics = metrics.task_specific_metrics
        
        # Create messages format - use actual messages from BFCL if available
        # Note: response is a string, not a dict, so we need to get messages from the task data
        messages = []
        
        # Get messages from task metadata
        messages = task.metadata['messages']
        
        # Create evaluation result - only keep KPI calculation fields
        eval_result = {
            "score": score,
            "accuracy": accuracy,
            "success_rate": success_rate
        }
        
        # Create metadata
        meta = {
            "task_id": task.task_id,
            "test_category": task.category_type,
            "is_multi_turn": task.metadata["is_multi_turn"],
            "finish_reason": "success" if score > 0 else "failed",
            "run_timestamp": datetime.now().isoformat() + "Z",
            "original_data": task.original_data,
            "ground_truth": task.ground_truth,
            "task_specific_metrics": task_specific_metrics
        }
        
        return UnifiedBenchmarkResult(
            model_path=config.model_name,
            benchmark_name="bfcl",
            task_name=task.category_type,  # Use test_category as task_name
            sampling_params={
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p
            },
            messages=messages,
            eval_result=eval_result,
            meta=meta
        )
    
    def get_final_kpi(self, results_file: str) -> Dict[str, float]:
        """
        Get final KPI from JSONL results file
        
        Args:
            results_file: Path to JSONL results file
            
        Returns:
            Final KPI metrics
        """
        if not os.path.exists(results_file):
            return {
                "total_tasks": 0,
                "average_accuracy": 0.0,
                "success_rate": 0.0,
                "valid_tasks": 0,
                "error_breakdown": {}
            }
        
        results = []
        with open(results_file, 'r') as f:
            for line in f:
                results.append(json.loads(line))
        
        if not results:
            return {
                "total_tasks": 0,
                "average_accuracy": 0.0,
                "success_rate": 0.0,
                "valid_tasks": 0,
                "error_breakdown": {}
            }
        
        # Calculate metrics
        accuracies = []
        valid_tasks = 0
        error_breakdown = {}
        
        for result in results:
            # Handle new unified result format
            if "eval_result" in result:
                # New unified format
                eval_result = result["eval_result"]
                score = eval_result.get("score", 0.0)
                accuracy = eval_result.get("accuracy", 0.0)
                valid = score > 0.5  # Consider score > 0.5 as valid
                
                if valid:
                    valid_tasks += 1
                    accuracies.append(accuracy)
                
                # Count error types from meta
                meta = result.get("meta", {})
                error_type = meta.get("error_type", "unknown")
                if error_type not in error_breakdown:
                    error_breakdown[error_type] = 0
                error_breakdown[error_type] += 1
            else:
                # Legacy format
                metrics = result.get("metrics", {})
                accuracy = metrics.get("accuracy", 0.0)
                valid = metrics.get("valid", False)
                
                if valid:
                    valid_tasks += 1
                    accuracies.append(accuracy)
                
                # Count error types
                error_type = metrics.get("error_type", "unknown")
                if error_type not in error_breakdown:
                    error_breakdown[error_type] = 0
                error_breakdown[error_type] += 1
        
        # Calculate averages
        average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
        success_rate = valid_tasks / len(results) if results else 0.0
        
        return {
            "total_tasks": len(results),
            "average_accuracy": average_accuracy,
            "success_rate": success_rate,
            "valid_tasks": valid_tasks,
            "error_breakdown": error_breakdown,
            "test_categories": list(set(r.get("meta", {}).get("test_category", "unknown") for r in results))
        }
    
    def validate_config(self, config: BenchmarkConfig) -> bool:
        """Validate configuration for BFCL"""
        # Check if BFCL is available
        if not bfcl_root:
            return False
        
        # Check if model name is provided
        if not config.model_name:
            return False
        
        return True

    def get_benchmark_info(self) -> Dict[str, Any]:
        """Get benchmark information"""
        return {
            "name": "Berkeley Function Calling Leaderboard (BFCL)",
            "version": "v3",
            "description": "Comprehensive function calling evaluation for LLMs",
            "url": "https://gorilla.cs.berkeley.edu/leaderboard.html",
            "paper": "https://arxiv.org/abs/2310.13023",
            "categories": [
                "simple", "irrelevance", "parallel", "multiple", "parallel_multiple",
                "java", "javascript", "live_simple", "live_multiple", "live_parallel",
                "live_parallel_multiple", "live_irrelevance", "live_relevance",
                "multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param",
                "multi_turn_long_context"
            ]
        }
    
    def _load_completed_tasks(self):
        """Load completed tasks from results file"""
        if os.path.exists(self.results_file):
            with open(self.results_file, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        self.completed_tasks.add(result.get("id"))
                    except:
                        continue
