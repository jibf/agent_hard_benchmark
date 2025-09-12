#!/usr/bin/env python3
"""
Main pipeline for benchmark filtering.
Orchestrates the complete filtering process with rule-based and LLM-as-Judge stages.
"""

import sys
import os
import json
import logging
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from math import comb
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.comprehensive_rule_filtering import ComprehensiveRuleFilter
from src.rule_filtering_orchestrator import RuleFilteringOrchestrator
from src.llm_judge_filtering import LLMJudge, LLMJudgeConfig, LLMJudgeStep
from src.data_loader import BenchmarkDataLoader
from src.utils.types import Benchmark

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BenchmarkFilteringPipeline:
    """Complete benchmark filtering pipeline."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.data_loader = BenchmarkDataLoader()
        self.use_specific_filters = self.config.get('use_specific_filters', False)
        
        self.llm_config = LLMJudgeConfig(
            model=self.config.get("llm_model", "openai/gpt-4.1"),
            max_samples=self.config.get("llm_max_samples", None),
            batch_size=self.config.get("llm_batch_size", 10),
            max_retries=self.config.get("llm_max_retries", 3),
            retry_delay=self.config.get("llm_retry_delay", 1.0),
            num_proc=self.config.get("num_proc", 1),
            steps=[LLMJudgeStep.FILTER] if self.config.get("llm_filter_only", False) else [LLMJudgeStep.FILTER, LLMJudgeStep.SCORE]
        )
    
    def _make_json_serializable(self, obj):
        """Make objects JSON serializable by converting enums and other non-serializable types."""
        if isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'value'):  # Handle enums like BenchmarkType
            return obj.value
        else:
            return obj
    
    def _convert_response_list_to_qid_list(self, response_list: List[Dict]) -> List[Dict]:
        """Convert response list to unique question ID list, removing duplicates."""
        questions = set()
        for response in response_list:
            unique_key = (
                response['benchmark_name'], 
                response['task_name'], 
                response['meta']['id']
            )
            questions.add(unique_key)
        
        return [{ "benchmark_name": key[0], "task_name": key[1], "question_id": key[2] } for key in questions ]
        
        
    def run_pipeline(self, skip_llm_judge: bool = False, skip_rule_based: bool = False) -> Tuple[List[Dict], List[Dict]]:
        """Run the complete filtering pipeline."""
        logger.info("Starting benchmark filtering pipeline")
        
        # Step 1: Rule-based filtering
        all_samples = self._load_benchmark_data()
        separability_dict = self._compute_separability(all_samples)
        logger.info(f"Benchmark separability before filtering: {json.dumps(separability_dict, indent=2)}")
        
        # Calculate model ranking from original dataset for consistent ordering
        model_ranking = None
        if not self.config.get('skip_visualization', False):
            model_ranking = self._calculate_model_ranking(all_samples)
            self._visualize_model_performance(all_samples, "Original Dataset", "original_performance", model_ranking=model_ranking)

        if skip_rule_based:
            logger.info("Skipping Step 1: Rule-based filtering")
            logger.info("Running LLM-as-Judge on questions independently")
            return self._run_llm_judge(all_samples)
        
        
        logger.info("Step 1: Comprehensive rule-based filtering")
        step1_passed, step1_dropped = self._run_step1_rule_filtering(all_samples)
        separability_dict = self._compute_separability(step1_passed)
        logger.info(f"Benchmark separability after Step 1: {json.dumps(separability_dict, indent=2)}")
        
        # Visualize Step 1 filtered performance
        if not self.config.get('skip_visualization', False):
            self._visualize_model_performance(step1_passed, "After Step 1 (Rule-based Filtering)", "step1_filtered_performance", model_ranking=model_ranking)
        
        print("step1_passed", len(step1_passed)) #, step1_passed[0])
        step1_N = len(step1_passed)
        
        # Count unique tasks in step1_passed
        step1_unique_tasks = self._count_unique_tasks(step1_passed)
        logger.info(f"Step 1 passed: {len(step1_passed)} samples from {step1_unique_tasks} unique tasks")
        
        # Create baseline sample set by randomly sampling N tasks from original samples
        baseline_samples = self._create_task_wise_baseline_sample_set(all_samples, step1_unique_tasks)
        logger.info(f"Created task-wise baseline sample set with {len(baseline_samples)} samples from {step1_unique_tasks} tasks")
        
        # Compute separability for baseline for comparison
        baseline_separability = self._compute_separability(baseline_samples)
        logger.info(f"Benchmark separability for baseline: {json.dumps(baseline_separability, indent=2)}")
        
        # Visualize baseline performance
        if not self.config.get('skip_visualization', False):
            self._visualize_model_performance(baseline_samples, "Baseline (Random Sampling)", "baseline_performance", model_ranking=model_ranking)
        

        baseline_samples = self._create_task_wise_baseline_sample_set(all_samples, 80)
        logger.info(f"Created task-wise baseline sample set with {len(baseline_samples)} samples from 80 tasks")
        
        # Compute separability for baseline for comparison
        baseline_separability = self._compute_separability(baseline_samples)
        logger.info(f"Benchmark separability for baseline: {json.dumps(baseline_separability, indent=2)}")
        

        self._save_step1_results(step1_passed, step1_dropped)
        
        if skip_llm_judge:
            logger.info("Skipping Step 2: LLM-as-Judge filtering")
            return step1_passed, step1_dropped
        
        # Step 2: LLM-as-Judge filtering
        logger.info("Step 2: LLM-as-Judge filtering")
        step2_passed, step2_dropped = self._run_llm_judge(step1_passed)
        step2_N = len(step2_passed)
        
        # Count unique tasks in step2_passed
        step2_unique_tasks = self._count_unique_tasks(step2_passed)
        logger.info(f"Step 2 passed: {len(step2_passed)} samples from {step2_unique_tasks} unique tasks")
        
        # Create baseline sample set by randomly sampling N tasks from step1_passed
        baseline_samples_step1 = self._create_task_wise_baseline_sample_set(step1_passed, step2_unique_tasks)
        baseline_separability = self._compute_separability(baseline_samples_step1)
        logger.info(f"Benchmark separability for baseline (from step1_passed): {json.dumps(baseline_separability, indent=2)}")

        # Create baseline sample set by randomly sampling N tasks from all_samples
        baseline_samples_all = self._create_task_wise_baseline_sample_set(all_samples, step2_unique_tasks)
        baseline_separability = self._compute_separability(baseline_samples_all)
        logger.info(f"Benchmark separability for baseline (from all_samples): {json.dumps(baseline_separability, indent=2)}")


        separability_dict = self._compute_separability(step2_passed)
        logger.info(f"Benchmark separability after Step 2: {json.dumps(separability_dict, indent=2)}")
        
        # Visualize Step 2 filtered performance
        if not self.config.get('skip_visualization', False):
            self._visualize_model_performance(step2_passed, "After Step 2 (LLM-as-Judge Filtering)", "step2_filtered_performance", model_ranking=model_ranking)
        
        self._save_results(step2_passed, step2_dropped, "step2_llm_judge")
        
        # Final summary and results
        self._print_final_summary(step1_passed, step2_passed)
        return step2_passed, step1_dropped + step2_dropped

    
    def _load_benchmark_data(self) -> List[Dict]:
        """Load benchmark data based on configuration."""
        logger.info("Loading benchmark data...")
        
        target_benchmark = None
        if self.use_specific_filters:
            target_benchmark = self.config.get('target_benchmark')
            logger.info(f"Loading only target benchmark: {target_benchmark}")
        
        all_samples = self.data_loader.load_benchmark_data("benchmark", target_benchmark)
        logger.info(f"Loaded {len(all_samples):,} total samples")
        
        return all_samples
    
    
    def _run_step1_rule_filtering(self, all_samples: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Run Step 1: Rule-based filtering (general or benchmark-specific)."""
        target_benchmark = self.config.get('target_benchmark') if self.use_specific_filters else None
        
        # Save unified dataset before filtering
        self._save_unified_dataset(all_samples)
        
        if self.use_specific_filters:
            logger.info("Using benchmark-specific filtering...")
            orchestrator = RuleFilteringOrchestrator()
            passed_samples, dropped_samples = orchestrator.filter_samples(
                all_samples, 
                use_specific_filters=True,
                target_benchmark=target_benchmark
            )
        else:
            logger.info("Using general comprehensive filtering...")
            rule_filter = ComprehensiveRuleFilter()
            passed_samples, dropped_samples = rule_filter.filter_samples(all_samples)
        
        logger.info(f"Step 1 completed: {len(passed_samples):,} samples passed")
        return passed_samples, dropped_samples
    
    def _save_step1_results(self, step1_passed: List[Dict], step1_dropped: List[Dict]):
        """Save all Step 1 results in various formats."""
        self._save_results(step1_passed, step1_dropped, "step1_rule_based")
        self._save_benchmark_specific_results(step1_passed, step1_dropped, "step1_rule_based")
        self._save_unified_step1_results(step1_passed, step1_dropped)
    

    def _compute_separability(self, samples: List[Dict], n_bootstrap: int=1000, ci: float=0.95) -> float:

        score_dict = {}
        separability_dict = {}
        for sample in samples:
            model_name = sample["model_path"]
            benchmark_name = sample["benchmark_name"]
            score = sample["eval_result"]["score"]
            if benchmark_name not in score_dict:
                score_dict[benchmark_name] = {}
            if model_name not in score_dict[benchmark_name]:
                score_dict[benchmark_name][model_name] = []
            score_dict[benchmark_name][model_name].append(score)
        
        for benchmark in score_dict:
            score_matrix = np.array([score_dict[benchmark][model] for model in sorted(score_dict[benchmark].keys())])
            num_models = score_matrix.shape[0]
            intervals = []

            for i in range(num_models):
                i_ci = self._bootstrap_confidence_interval(score_matrix[i], n_bootstrap=n_bootstrap, ci=ci)
                intervals.append(i_ci)
            intervals.sort(key=lambda x: x[0])

            overlapping_pairs = []
            total_pairs = comb(num_models, 2)
            for i in range(len(intervals)):
                for j in range(i+1, len(intervals)):
                    # If the start time of the second interval is less than the end time of the first, they overlap
                    if intervals[j][0] < intervals[i][1]:
                        # Check if the pair is already in the list
                        if (intervals[i], intervals[j]) not in overlapping_pairs and (intervals[j], intervals[i]) not in overlapping_pairs:
                            overlapping_pairs.append((intervals[i], intervals[j]))
                    else:
                        break
            separability = 1 - len(overlapping_pairs) / total_pairs if total_pairs > 0 else 0
            separability_dict[benchmark] = separability

        return separability_dict

    def _calculate_model_ranking(self, samples: List[Dict]) -> Dict[str, List[str]]:
        """Calculate model ranking based on performance for consistent ordering across visualizations."""
        benchmark_data = {}
        for sample in samples:
            benchmark_name = sample["benchmark_name"]
            model_name = sample["model_path"]
            score = sample["eval_result"]["score"]
            
            if benchmark_name not in benchmark_data:
                benchmark_data[benchmark_name] = {}
            if model_name not in benchmark_data[benchmark_name]:
                benchmark_data[benchmark_name][model_name] = []
            benchmark_data[benchmark_name][model_name].append(score)
        
        # Calculate ranking for each benchmark
        model_ranking = {}
        for benchmark_name, model_scores in benchmark_data.items():
            model_means = {}
            for model_name, scores in model_scores.items():
                model_means[model_name] = np.mean(scores)
            # Sort by performance (descending)
            model_ranking[benchmark_name] = sorted(model_means.keys(), key=lambda x: model_means[x], reverse=True)
        
        return model_ranking

    def _visualize_model_performance(self, samples: List[Dict], title: str, filename: str, n_bootstrap: int=100, ci: float=0.95, model_ranking: Dict[str, List[str]] = None):
        """Create bar graph visualization of model-wise performance with confidence intervals."""
        logger.info(f"Creating visualization: {title}")
        
        # Create output directory for plots
        plots_dir = "pipeline_results/plots"
        os.makedirs(plots_dir, exist_ok=True)
        
        # Group samples by benchmark
        benchmark_data = {}
        for sample in samples:
            benchmark_name = sample["benchmark_name"]
            model_name = sample["model_path"]
            score = sample["eval_result"]["score"]
            
            if benchmark_name not in benchmark_data:
                benchmark_data[benchmark_name] = {}
            if model_name not in benchmark_data[benchmark_name]:
                benchmark_data[benchmark_name][model_name] = []
            benchmark_data[benchmark_name][model_name].append(score)
        
        # Create separate plots for each benchmark
        for benchmark_name, model_scores in benchmark_data.items():
            # Determine model ordering
            if model_ranking and benchmark_name in model_ranking:
                # Use provided ranking, but only include models that exist in current data
                available_models = set(model_scores.keys())
                ordered_models = [model for model in model_ranking[benchmark_name] if model in available_models]
                # Add any models not in ranking at the end
                remaining_models = sorted(available_models - set(ordered_models))
                model_order = ordered_models + remaining_models
            else:
                # Calculate ranking based on current data (for original dataset)
                model_means = {}
                for model_name, scores in model_scores.items():
                    model_means[model_name] = np.mean(scores)
                model_order = sorted(model_means.keys(), key=lambda x: model_means[x], reverse=True)
            
            # Calculate means and confidence intervals for each model in the determined order
            model_names = []
            means = []
            ci_lowers = []
            ci_uppers = []
            
            for model_name in model_order:
                scores = np.array(model_scores[model_name])
                mean_score = np.mean(scores)
                ci_lower, ci_upper = self._bootstrap_confidence_interval(scores, n_bootstrap, ci)
                
                model_names.append(model_name)
                means.append(mean_score)
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)
            
            # Create the plot with smaller figure size
            plt.figure(figsize=(10, 6))
            x_pos = np.arange(len(model_names))
            
            # Create bars with error bars
            bars = plt.bar(x_pos, means, yerr=[np.array(means) - np.array(ci_lowers), 
                                              np.array(ci_uppers) - np.array(means)], 
                          capsize=3, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.8)
            
            # Customize the plot
            plt.xlabel('Model', fontsize=10, fontweight='bold')
            plt.ylabel('Performance Score', fontsize=10, fontweight='bold')
            plt.title(f'{title} - {benchmark_name}\nModel Performance with {int(ci*100)}% Confidence Intervals', 
                     fontsize=12, fontweight='bold', pad=15)
            
            # Set x-axis labels
            plt.xticks(x_pos, model_names, rotation=45, ha='right', fontsize=9)
            
            # Add value labels on top of bars (smaller font)
            for i, (bar, mean, ci_low, ci_high) in enumerate(zip(bars, means, ci_lowers, ci_uppers)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (ci_high - mean) + 0.005,
                        f'{mean:.3f}\n[{ci_low:.3f}, {ci_high:.3f}]',
                        ha='center', va='bottom', fontsize=7, fontweight='bold')
            
            # Add grid for better readability
            plt.grid(True, alpha=0.3, axis='y')
            
            # Adjust layout to prevent label cutoff
            plt.tight_layout()
            
            # Save the plot with reduced DPI
            safe_benchmark_name = benchmark_name.replace('/', '_').replace(' ', '_')
            plot_filename = os.path.join(plots_dir, f"{filename}_{safe_benchmark_name}.png")
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', 
                       format='png')
            plt.close()
            
            logger.info(f"Saved plot: {plot_filename}")
            
            # Print summary statistics
            logger.info(f"\n{benchmark_name} - Model Performance Summary:")
            logger.info("=" * 60)
            for model_name, mean, ci_low, ci_high in zip(model_names, means, ci_lowers, ci_uppers):
                logger.info(f"{model_name:30} | Mean: {mean:.4f} | CI: [{ci_low:.4f}, {ci_high:.4f}]")

    def _bootstrap_confidence_interval(self, scores: np.ndarray, n_bootstrap: int=100, ci: float=0.95):
        n = len(scores)
        means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=n, replace=True)
            means.append(np.mean(sample))
        lower = np.percentile(means, (1 - ci) / 2 * 100)
        upper = np.percentile(means, (1 + ci) / 2 * 100)
        return lower, upper
        
    def _save_results(self, passed_samples: List[Dict], dropped_samples: List[Dict], step_name: str):
        """Save results for a pipeline step."""
        logger.info(f"Saving {step_name} results...")
        
        # Save passed samples
        passed_filename = f"{step_name}_passed_samples.jsonl"
        with open(passed_filename, "w") as f:
            for sample in passed_samples:
                serializable_sample = self._make_json_serializable(sample)
                f.write(json.dumps(serializable_sample) + "\n")
        
        # Save dropped samples
        dropped_filename = f"{step_name}_dropped_samples.jsonl"
        with open(dropped_filename, "w") as f:
            for sample in dropped_samples:
                serializable_sample = self._make_json_serializable(sample)
                f.write(json.dumps(serializable_sample) + "\n")
        
        logger.info(f"Results saved to {passed_filename} and {dropped_filename}")
    
    def _save_benchmark_specific_results(self, passed_samples: List[Dict], dropped_samples: List[Dict], step_name: str):
        """Save benchmark-specific passed and pruned files after step 1."""
        logger.info(f"Saving benchmark-specific {step_name} results...")
        
        # Create output directory
        output_dir = "pipeline_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Group samples by benchmark
        benchmark_passed = {}
        benchmark_dropped = {}
        
        for sample in passed_samples:
            benchmark_name = sample.get('benchmark_name', 'unknown')
            if benchmark_name not in benchmark_passed:
                benchmark_passed[benchmark_name] = []
            benchmark_passed[benchmark_name].append(sample)
        
        for sample in dropped_samples:
            benchmark_name = sample.get('benchmark_name', 'unknown')
            if benchmark_name not in benchmark_dropped:
                benchmark_dropped[benchmark_name] = []
            benchmark_dropped[benchmark_name].append(sample)
        
        # Save benchmark-specific passed files
        for benchmark_name, samples in benchmark_passed.items():
            if samples:  # Only save if there are samples
                # Create benchmark-specific folder
                benchmark_dir = os.path.join(output_dir, benchmark_name)
                os.makedirs(benchmark_dir, exist_ok=True)
                
                # Save with benchmark name + passed
                filename = os.path.join(benchmark_dir, f"{benchmark_name}_passed.jsonl")
                with open(filename, "w") as f:
                    for sample in samples:
                        serializable_sample = self._make_json_serializable(sample)
                        f.write(json.dumps(serializable_sample) + "\n")
                logger.info(f"Saved {len(samples):,} passed samples for {benchmark_name} to {filename}")
        
        # Save benchmark-specific dropped files
        for benchmark_name, samples in benchmark_dropped.items():
            if samples:  # Only save if there are samples
                # Create benchmark-specific folder
                benchmark_dir = os.path.join(output_dir, benchmark_name)
                os.makedirs(benchmark_dir, exist_ok=True)
                
                # Save with benchmark name + pruned
                filename = os.path.join(benchmark_dir, f"{benchmark_name}_pruned.jsonl")
                with open(filename, "w") as f:
                    for sample in samples:
                        serializable_sample = self._make_json_serializable(sample)
                        f.write(json.dumps(serializable_sample) + "\n")
                logger.info(f"Saved {len(samples):,} pruned samples for {benchmark_name} to {filename}")
        
        logger.info(f"Benchmark-specific results saved for {len(benchmark_passed)} benchmarks")
    
    def _save_unified_dataset(self, all_samples: List[Dict]):
        """Save the unified dataset before any filtering."""
        logger.info("Saving unified dataset before filtering...")
        output_dir = "pipeline_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save unified dataset
        unified_filename = os.path.join(output_dir, "unified_dataset_all_samples.jsonl")
        with open(unified_filename, "w") as f:
            for sample in all_samples:
                serializable_sample = self._make_json_serializable(sample)
                f.write(json.dumps(serializable_sample) + "\n")
        
        logger.info(f"Unified dataset saved to {unified_filename} with {len(all_samples):,} samples")
        return unified_filename
    
    def _save_unified_step1_results(self, passed_samples: List[Dict], dropped_samples: List[Dict]):
        """Save unified passed and pruned files after step 1."""
        logger.info("Saving unified Step 1 results...")
        
        # Create output directory
        output_dir = "pipeline_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save unified passed samples
        unified_passed_filename = os.path.join(output_dir, "unified_step1_passed_samples.jsonl")
        with open(unified_passed_filename, "w") as f:
            for sample in passed_samples:
                serializable_sample = self._make_json_serializable(sample)
                f.write(json.dumps(serializable_sample) + "\n")
        logger.info(f"Unified passed samples saved to {unified_passed_filename} with {len(passed_samples):,} samples")
        
        # Save unified dropped samples
        unified_dropped_filename = os.path.join(output_dir, "unified_step1_dropped_samples.jsonl")
        with open(unified_dropped_filename, "w") as f:
            for sample in dropped_samples:
                serializable_sample = self._make_json_serializable(sample)
                f.write(json.dumps(serializable_sample) + "\n")
        logger.info(f"Unified dropped samples saved to {unified_dropped_filename} with {len(dropped_samples):,} samples")
    
    def _print_final_summary(self, step1_passed: List[Dict], step2_passed: List[Dict]):
        """Print final pipeline summary."""
        logger.info("Pipeline completed - Final summary")
        
        # Count samples per benchmark
        step1_benchmarks = {}
        step2_benchmarks = {}
        
        for sample in step1_passed:
            benchmark = sample.get('benchmark_name', 'unknown')
            step1_benchmarks[benchmark] = step1_benchmarks.get(benchmark, 0) + 1
        
        for sample in step2_passed:
            benchmark = sample.get('benchmark_name', 'unknown')
            step2_benchmarks[benchmark] = step2_benchmarks.get(benchmark, 0) + 1
        
        logger.info(f"\nStep 1 (Rule-based) results:")
        for benchmark, count in sorted(step1_benchmarks.items()):
            logger.info(f"  {benchmark}: {count:,}")
        
        logger.info(f"\nStep 2 (LLM-as-Judge) results:")
        for benchmark, count in sorted(step2_benchmarks.items()):
            logger.info(f"  {benchmark}: {count:,}")
        
        # Count unique questions
        step1_questions = self._count_unique_questions(step1_passed)
        step2_questions = self._count_unique_questions(step2_passed)
        
        logger.info(f"\nUnique questions:")
        logger.info(f"  After Step 1: {step1_questions:,}")
        logger.info(f"  After Step 2: {step2_questions:,}")
        
        logger.info(f"\nPipeline complete! Final results saved to step2_llm_judge_passed_samples.jsonl")
    
    def _count_unique_questions(self, samples: List[Dict]) -> int:
        """Count unique questions in samples."""
        return len(self._convert_response_list_to_qid_list(samples))
    
    def _filter_responses_by_benchmark(self, responses: List[Dict], benchmark_name: str) -> List[Dict]:
        if benchmark_name not in list(benchmark.value for benchmark in Benchmark):
            raise ValueError(f"Invalid benchmark name {benchmark_name}")
        
        result = []
        for response in responses:
            if response['benchmark_name'].value == benchmark_name:  # benchmark_name is enum, compare with .value
                result.append(response)
        return result


    def _run_llm_judge(self, responses: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Run LLM judge independently on questions from benchmark datasets."""
        # Determine which benchmarks to process based on target_benchmark config
        passed_responses, filtered_responses = [], []

        for benchmark in Benchmark:
            benchmark_responses = self._filter_responses_by_benchmark(responses, benchmark.value)
            if len(benchmark_responses) == 0:
                continue

            logger.info(f"Processing {benchmark.value} benchmark: {len(benchmark_responses)} responses")
            judge = LLMJudge(benchmark, self.llm_config)
            benchmark_results = judge.assess_questions()

            # Create mapping from question ID to benchmark results
            result_map = {}
            for benchmark_result in benchmark_results:
                result_map[benchmark_result.question_id] = benchmark_result

            passed_question_ids = [] 
            for benchmark_result in benchmark_results:
                if not benchmark_result.is_flawed:
                    passed_question_ids.append(benchmark_result.question_id)
            
            assert len(passed_question_ids) > 0, f"{benchmark.value} No question passes filtering; Check question ID format"
            for response in benchmark_responses:
                possible_question_ids = [response['meta']['id'], f"{response['task_name']}_{response['meta']['id']}",  f"{response['task_name']}-{response['meta']['id']}"]
                matched_question_id = None
                
                for possible_question_id in possible_question_ids:
                    if possible_question_id in result_map:
                        matched_question_id = possible_question_id
                        break
                
                # Add benchmark result fields to response
                if matched_question_id:
                    benchmark_result = result_map[matched_question_id]
                    response['llm_judge_result'] = {
                        'is_flawed': benchmark_result.is_flawed,
                        'error_category': benchmark_result.error_category,
                        'reasoning_summary': benchmark_result.reasoning_summary,
                        'reasoning': benchmark_result.reasoning
                    }
                    
                    # Use is_flawed from benchmark_result to determine if passed
                    if not benchmark_result.is_flawed:
                        passed_responses.append(response)
                    else:
                        filtered_responses.append(response)
            
        return passed_responses, filtered_responses
    
    def _count_unique_tasks(self, samples: List[Dict]) -> int:
        """Count unique tasks in samples."""
        task_ids = set()
        for sample in samples:
            task_id = self._extract_task_id(sample)
            task_ids.add(task_id)
        return len(task_ids)
    
    def _extract_task_id(self, sample: Dict) -> str:
        """Extract a unique task identifier from a sample."""
        # Use the same logic as in comprehensive_rule_filtering.py
        if 'task_name' in sample and 'meta' in sample and 'id' in sample['meta']:
            return sample['task_name'] + "_" + sample['meta']['id']
        elif 'task_name' in sample:
            return sample['task_name']
        elif 'question' in sample:
            return sample['question']
        elif 'prompt' in sample:
            return sample['prompt']
        elif 'messages' in sample and sample['messages']:
            # Use first message content as task identifier
            first_msg = sample['messages'][0]
            if isinstance(first_msg, dict) and 'content' in first_msg:
                return first_msg['content'][:100]  # First 100 chars
        elif 'conversation' in sample and sample['conversation']:
            # Use first turn as task identifier
            first_turn = sample['conversation'][0]
            if isinstance(first_turn, dict) and 'content' in first_turn:
                return first_turn['content'][:100]
        
        # Fallback: use a hash of the entire sample
        import hashlib
        return hashlib.md5(json.dumps(sample, sort_keys=True).encode()).hexdigest()
    
    def _create_task_wise_baseline_sample_set(self, all_samples: List[Dict], target_task_count: int) -> List[Dict]:
        """Create a baseline sample set by randomly sampling N tasks from original samples."""
        logger.info(f"Creating task-wise baseline: sampling {target_task_count} tasks from {len(all_samples)} total samples")
        
        # Group samples by task
        task_groups = {}
        for sample in all_samples:
            task_id = self._extract_task_id(sample)
            if task_id not in task_groups:
                task_groups[task_id] = []
            task_groups[task_id].append(sample)
        
        total_tasks = len(task_groups)
        logger.info(f"Found {total_tasks} unique tasks in dataset")
        
        if target_task_count >= total_tasks:
            logger.info("Target task count >= total tasks, returning all samples")
            return all_samples.copy()
        
        # Use random sampling with a fixed seed for reproducibility
        random.seed(42)
        selected_task_ids = random.sample(list(task_groups.keys()), target_task_count)
        random.seed()  # Reset seed
        
        # Collect all samples for selected tasks
        baseline_samples = []
        for task_id in selected_task_ids:
            baseline_samples.extend(task_groups[task_id])
        
        logger.info(f"Task-wise baseline created: {len(baseline_samples)} samples from {len(selected_task_ids)} tasks")
        return baseline_samples
    
    def _create_baseline_sample_set(self, all_samples: List[Dict], target_count: int) -> List[Dict]:
        """Create a baseline sample set by randomly sampling N samples from original samples."""
        logger.info(f"Creating sample-wise baseline: sampling {target_count} samples from {len(all_samples)} total samples")
        
        if target_count >= len(all_samples):
            logger.info("Target count >= total samples, returning all samples")
            return all_samples.copy()
        
        # Use random sampling with a fixed seed for reproducibility
        random.seed(42)
        baseline_samples = random.sample(all_samples, target_count)
        random.seed()  # Reset seed
        
        logger.info(f"Sample-wise baseline created with {len(baseline_samples)} samples")
        return baseline_samples
    
    
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark Filtering Pipeline")
    parser.add_argument(
        "--skip-llm-judge", 
        action="store_true", 
        help="Skip Step 2 (LLM-as-Judge filtering)"
    )
    parser.add_argument(
        "--skip-rule-based", 
        action="store_true", 
        help="Skip Step 1 (rule-based filtering) and run LLM judge on questions independently"
    )
    parser.add_argument(
        "--specific-step1", 
        action="store_true", 
        help="Use benchmark-specific filtering for Step 1 instead of general filtering"
    )
    parser.add_argument(
        "--llm-model", 
        default="openai/gpt-4.1",
        help="LLM model to use for Step 2 (default: gpt-4.1)"
    )
    parser.add_argument(
        "--llm-max-samples", 
        type=int,
        help="Maximum samples to process in Step 2 (default: all)"
    )
    parser.add_argument(
        "--llm-batch-size", 
        type=int,
        default=10,
        help="Batch size for LLM processing (default: 10)"
    )
    parser.add_argument(
        "--llm-max-retries", 
        type=int,
        default=3,
        help="Maximum retries for LLM calls (default: 3)"
    )
    parser.add_argument(
        "--llm-retry-delay", 
        type=float,
        default=1.0,
        help="Delay between retries in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--num-proc", 
        type=int,
        default=1,
        help="Number of processes for multiprocessing (default: 1)"
    )
    parser.add_argument(
        "--target_benchmark", "--target-benchmark",
        choices=list(benchmark.value for benchmark in Benchmark),
        help="Target benchmark to process (default: all available benchmarks)"
    )
    parser.add_argument(
        "--llm-filter-only", 
        action="store_true", 
        default=True,
        help="Run only LLM filtering step, skip scoring step"
    )
    parser.add_argument(
        "--skip-visualization", 
        action="store_true", 
        help="Skip creating performance visualization plots"
    )
    
    args = parser.parse_args()
    
    # Configuration
    config = {
        "llm_model": args.llm_model,
        "llm_max_samples": args.llm_max_samples,
        "llm_batch_size": args.llm_batch_size,
        "llm_max_retries": args.llm_max_retries,
        "llm_retry_delay": args.llm_retry_delay,
        "num_proc": args.num_proc,
        "target_benchmark": args.target_benchmark,
        "use_specific_filters": args.specific_step1,
        "llm_filter_only": args.llm_filter_only,
        "skip_visualization": args.skip_visualization
    }
    
    # Validate arguments
    if args.skip_rule_based and args.skip_llm_judge:
        logger.error("Cannot skip both rule-based and LLM judge filtering")
        sys.exit(1)
    
    # Run pipeline
    pipeline = BenchmarkFilteringPipeline(config)
    passed_samples, dropped_samples = pipeline.run_pipeline(
        skip_llm_judge=args.skip_llm_judge,
        skip_rule_based=args.skip_rule_based
    )
    
    logger.info(f"\nPipeline completed successfully!")
    logger.info(f"Final passed samples: {len(passed_samples):,}")
    logger.info(f"Total dropped samples: {len(dropped_samples):,}")

if __name__ == "__main__":
    main()
