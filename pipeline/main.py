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
import csv
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from math import comb
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime

from sentence_transformers import SentenceTransformer
import torch
from sklearn.manifold import TSNE

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.comprehensive_rule_filtering import ComprehensiveRuleFilter
from src.rule_filtering_orchestrator import RuleFilteringOrchestrator
from src.llm_judge_filtering import LLMJudge, LLMJudgeConfig, LLMJudgeStep
from src.data_loader import BenchmarkDataLoader
from src.utils.types import (
    Benchmark,
    UniqueQuestionID,
    LLMJudgeOutput,
    PipelineOutput,
    RuleBasedOutput,
)
from src.utils import group_responses_by_question, log_confusion_matrix_human_labelled
from src.bench_loaders import get_bench_loader
from metric.irt_metric import compute_irt_metric

# Logger will be configured in main() function
logger = logging.getLogger(__name__)

COMMON_MODEL_SET = {
    "claude-4-sonnet-thinking-on-10k",
    "Kimi-K2-Instruct",
    "Qwen3-235B-A22B-Instruct-2507-FP8",
    "o4-mini-high",
    "Qwen3-235B-A22B-Thinking-2507-FP8",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o-20240806",
    "claude-4-sonnet-thinking-off",
    "Qwen3-235B-A22B-FP8",
    "o3-high",
}


class BenchmarkFilteringPipeline:
    """Complete benchmark filtering pipeline."""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.data_loader = BenchmarkDataLoader()
        self.orchestrator = RuleFilteringOrchestrator()
        self.use_specific_filters = self.config.get("use_specific_filters", False)

        # Store metrics for final summary
        results_template = {
            "separability": [],
            "separability_baseline": [],
            "diversity": None,
            "irt": None,
            "agreement": {},
            "human_alignment": {
                "precision": None,
                "recall": None,
                "f1": None,
                "tp": None,
                "fp": None,
                "fn": None,
                "tn": None
            },
            "retention_ratio": None,
            "total_num": None,
            "subtask_size": {},
            "model_performance": {},
        }
        self.metrics_summary = {
            "original": deepcopy(results_template),
            "step1": deepcopy(results_template),
            "step2": deepcopy(results_template),
            "step3": deepcopy(results_template),
            "step4": deepcopy(results_template),
        }

        # Determine which LLM judge steps to run based on filtering scheme
        filter_mode = self.config.get("llm_filter_mode", "both")
        skip_scoring = self.config.get("skip_scoring", False)

        if filter_mode == "common":
            llm_steps = [LLMJudgeStep.UNIVERSAL_FILTER]
        elif filter_mode == "specific":
            llm_steps = [LLMJudgeStep.SPECIFIC_FILTER]
        elif filter_mode == "both":
            llm_steps = [LLMJudgeStep.UNIVERSAL_FILTER, LLMJudgeStep.SPECIFIC_FILTER]
            if not skip_scoring:
                llm_steps.append(LLMJudgeStep.SCORE)
        else:
            raise ValueError(
                f"Invalid llm_filter_mode: {filter_mode}. Must be 'common', 'specific', or 'both'"
            )

        self.llm_config = LLMJudgeConfig(
            model=self.config.get("llm_model", "openai/gpt-4.1"),
            max_samples=self.config.get("llm_max_samples", None),
            max_retries=self.config.get("llm_max_retries", 3),
            retry_delay=self.config.get("llm_retry_delay", 1.0),
            num_proc=self.config.get("num_proc", 1),
            steps=llm_steps,
        )

        # ----- Embedding model for semantic diversity -----
        model_name = self.config.get("embedding_model", "Qwen/Qwen3-Embedding-8B")
        self.embedder = None
        if self.config.get("skip_measurement", False):
            pass
        else:
            self.embedder = SentenceTransformer(
                model_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                model_kwargs={"torch_dtype": torch.float16},
            )
        self.embedding_batch_size = self.config.get("embedding_batch_size", 8)

        # Whether to embed the concatenation of all initial prompts (system & user)
        # before the first assistant/tool call, instead of only the last user prompt.
        self.embed_all_initial_prompts: bool = self.config.get(
            "embed_all_initial_prompts", False
        )
        self.fitering_template = {
            "Benchmark": None,
            "task_type": None,
            "task_id": None,
            "specific_rule_passed": None,
            "specific_llm_passed": None,
            "topk_selection_passed": None,
            "comp_passed": None,
        }
        self.filtering_summary = {}

    def _make_json_serializable(self, obj):
        """Make objects JSON serializable by converting enums and other non-serializable types."""
        if isinstance(obj, dict):
            return {
                key: self._make_json_serializable(value) for key, value in obj.items()
            }
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, "value"):  # Handle enums like BenchmarkType
            return obj.value
        else:
            return obj

    def _convert_response_list_to_qid_list(
        self, response_list: List[Dict]
    ) -> List[UniqueQuestionID]:
        """Convert response list to unique question ID list, removing duplicates."""
        questions = set()
        for response in response_list:
            unique_question_id = UniqueQuestionID(
                benchmark=response["benchmark_name"],  # Already enum after data loading
                task_name=response.get("task_name", None),
                question_id=response["meta"]["id"],
            )
            questions.add(unique_question_id)

        return list(questions)

    def filter_illegal_data(
        self, response_dict: Dict[UniqueQuestionID, List[Dict]]
    ) -> Dict[UniqueQuestionID, List[Dict]]:
        """Filter out illegal data entries from the response dictionary.
        1. duplicate questions from same model
        3. questions do not have all model's responses
        4. models that do not have responses for all questions
        """
        filtered_dict = {}

        # filter out duplicate questions from same model
        for qid, responses in response_dict.items():
            # hardcode for BFCL, only loading multi_turn data
            if qid.benchmark.value == "BFCL" and "multi_turn" not in qid.task_name:
                continue
            model_names = []
            valid_responses = []
            for response in responses:
                model_name = response["model_name"]
                if model_name not in model_names:
                    model_names.append(model_name)
                    valid_responses.append(response)
            filtered_dict[qid] = valid_responses
        response_count = sum([len(responses) for responses in response_dict.values()])
        valid_count = sum([len(responses) for responses in filtered_dict.values()])
        logging.info(
            f"Filtered {response_count - valid_count}/{response_count} duplicated responses."
        )

        # filter out models which do not have enough responses in each benchmark
        non_standard_prefixes = [
            "anthropic-",
            "openai-",
        ]  # TODO: hack for NexusBench, need to fix ASAP
        benchmark_question_statistics = {}
        for qid, responses in filtered_dict.items():
            for response in responses:
                for prefix in non_standard_prefixes:
                    if response["model_name"].startswith(prefix):
                        response["model_name"] = response["model_name"].replace(
                            prefix, ""
                        )

                benchmark_name = response["benchmark_name"]
                model_name = response["model_name"]
                if benchmark_name not in benchmark_question_statistics:
                    benchmark_question_statistics[benchmark_name] = {}
                if model_name not in benchmark_question_statistics[benchmark_name]:
                    benchmark_question_statistics[benchmark_name][model_name] = 0
                benchmark_question_statistics[benchmark_name][model_name] += 1

        benchmark_reamined_model = {}
        for benchmark_name, statics in benchmark_question_statistics.items():
            benchmark_reamined_model[benchmark_name] = set()
            avg_question_count = sum(statics.values()) / len(statics)
            for model_name, v in statics.items():
                if v > 0.8 * avg_question_count:
                    benchmark_reamined_model[benchmark_name].add(model_name)
                else:
                    logging.info(
                        f"Filtered model {model_name} for benchmark {benchmark_name} due to lacking valid responses."
                    )

        # filter out models do not have all results for all benchmarks
        for idx, benchmark_name in enumerate(benchmark_reamined_model):
            if idx == 0:
                remained_model_cross_all_benchmark = benchmark_reamined_model[
                    benchmark_name
                ]
                continue
            remained_model_cross_all_benchmark = (
                remained_model_cross_all_benchmark.intersection(
                    benchmark_reamined_model[benchmark_name]
                )
            )
        remained_model_cross_all_benchmark = (
            remained_model_cross_all_benchmark.intersection(COMMON_MODEL_SET)
        )
        for idx, benchmark_name in enumerate(benchmark_reamined_model):
            filterd_model = (
                benchmark_reamined_model[benchmark_name]
                - remained_model_cross_all_benchmark
            )
            logging.info(
                f"Filtered model {filterd_model} for benchmark {benchmark_name} to maintain consistency cross all benchmarks."
            )

        final_dict = {}
        for qid, responses in filtered_dict.items():
            filterd_response_list = []
            unique_model_set = set()
            for response in responses:
                model_name = response["model_name"]
                if model_name in remained_model_cross_all_benchmark:
                    filterd_response_list.append(response)
                    unique_model_set.add(model_name)

            if unique_model_set == remained_model_cross_all_benchmark:
                final_dict[qid] = filterd_response_list

        response_count = sum([len(responses) for responses in filtered_dict.values()])
        valid_count = sum([len(responses) for responses in final_dict.values()])

        logging.info(
            f"Filtered {response_count - valid_count}/{response_count} responses to maintain consistency cross all benchmarks."
        )

        return final_dict

    def _is_question_flawed(self, llm_output):
        """Check if question is flawed based on any available filter results"""
        if not llm_output:
            return False
        # Check specific filter first, then universal filter
        if llm_output.specific_filter:
            return llm_output.specific_filter.is_flawed
        elif llm_output.universal_filter:
            return llm_output.universal_filter.is_flawed
        return False

    def _get_all_questions_from_loader(self):
        benchmark_names = self.config.get("target_benchmark", None)
        all_questions = {}
        benchmark_names = benchmark_names if benchmark_names is not None else [i.value for i in Benchmark]

        for benchmark_name in benchmark_names:
            benchmark = Benchmark(benchmark_name)
            loader_class = get_bench_loader(benchmark)
            loader = loader_class()
            benchmark_questions = loader.load_questions()
            all_questions[benchmark_name] = benchmark_questions

        return all_questions


    def run_pipeline(
        self, skip_llm_judge: bool = False, skip_rule_based: bool = False
    ) -> Dict[UniqueQuestionID, PipelineOutput]:
        """Run the complete filtering pipeline."""
        logger.info("Starting benchmark filtering pipeline")

        all_responses = self._load_benchmark_data()
        self.all_questions = self._get_all_questions_from_loader()
        self.human_labelled_details = self.data_loader.load_human_labelled_ground_truth(self.config.get("target_benchmark"))
        current_human_labelled = set(self.human_labelled_details.keys())

        responses_by_question = group_responses_by_question(all_responses)
        responses_by_question = self.filter_illegal_data(responses_by_question)

        pipeline_outputs = {k: PipelineOutput() for k in responses_by_question.keys()}
        self.initial_baseline_ids = set(responses_by_question.keys())

        # Compute and log metrics for original/initial dataset
        self.compute_and_log_metrics(
            passed_responses=responses_by_question,
            input_ids=self.initial_baseline_ids,
            phase="initial",
            human_labelled_questions=current_human_labelled,
            baseline_source=responses_by_question,
        )

        current_responses = responses_by_question

        if not skip_rule_based:
            # Step 1: Apply benchmark-specific filtering for benchmarks that have specific filters
            logger.info("Step 1: Benchmark-specific filtering")
            step1_passed, step1_dropped = self._run_benchmark_specific_filtering(
                responses_by_question
            )
            # Compute and log all metrics for step1 (including baseline)
            self.compute_and_log_metrics(
                passed_responses=step1_passed,
                input_ids=set(responses_by_question.keys()),
                phase="step1",
                human_labelled_questions=current_human_labelled,
                baseline_source=responses_by_question,
            )
            current_responses = step1_passed
            # Update human labelled questions to only include those that passed step1
            current_human_labelled = current_human_labelled & set(step1_passed.keys())

            # Update pipeline_outputs with step1 results
            step1_passed_questions = set(step1_passed.keys())
            for question_id in pipeline_outputs.keys():
                passed = question_id in step1_passed_questions
                pipeline_outputs[question_id].rule_based_output = RuleBasedOutput(
                    passed=passed, reason=None
                )

            # Count unique tasks and samples in step1_passed
            step1_unique_tasks = len(step1_passed)
            step1_sample_count = sum(
                len(responses) for responses in step1_passed.values()
            )
            logger.info(
                f"Step 1 passed: {step1_sample_count} samples from {step1_unique_tasks} unique tasks"
            )
        else:
            logger.info("Skipping Step 1: Rule-based filtering")

        # Step 2: LLM-as-Judge filtering
        if not skip_llm_judge:
            logger.info("Step 2: LLM-as-Judge filtering")

            step2_result = self._run_llm_judge(current_responses)
            # Update pipeline_outputs with step2 results
            for question_id, llm_output in step2_result.items():
                pipeline_outputs[question_id].llm_judge_output = llm_output

            # Create step2_passed using the same logic as final summary
            # This ensures consistency between run_pipeline and _print_final_summary

            # Get step1_passed question IDs for consistent calculation
            step1_passed_qids = [
                qid
                for qid, output in pipeline_outputs.items()
                if not output.rule_based_output or output.rule_based_output.passed
            ]

            step2_passed = {
                qid: responses_by_question[qid]
                for qid in step1_passed_qids
                if not self._is_question_flawed(pipeline_outputs[qid].llm_judge_output)
            }
            # Compute and log all metrics for step2 (including baseline)
            self.compute_and_log_metrics(
                passed_responses=step2_passed,
                input_ids=set(step1_passed_qids),
                phase="step2",
                human_labelled_questions=current_human_labelled,
                baseline_source=responses_by_question,
            )
            current_responses = step2_passed
            # Update human labelled questions to only include those that passed step2
            current_human_labelled = current_human_labelled & set(step2_passed.keys())
            # Count unique tasks and samples in step2_passed
            step2_unique_tasks = len(step2_passed)
            step2_sample_count = sum(
                len(responses) for responses in step2_passed.values()
            )
            logger.info(
                f"Step 2 passed: {step2_sample_count} samples from {step2_unique_tasks} unique tasks"
            )

            # Step 3: Top-K selection based on scores
            if LLMJudgeStep.SCORE in self.llm_config.steps:
                logger.info("Step 3: Selecting top 50 samples based on total scores")
                step3_passed = self._run_step3_top_k_selection(
                    step2_result, responses_by_question, 50
                )
                # Compute and log all metrics for step3 (including baseline)
                self.compute_and_log_metrics(
                    passed_responses=step3_passed,
                    input_ids=set(step2_result.keys()),
                    phase="step3",
                    human_labelled_questions=current_human_labelled,
                    baseline_source=responses_by_question,
                )
                current_responses = step3_passed
                # Update human labelled questions to only include those that passed step3
                current_human_labelled = current_human_labelled & set(step3_passed.keys())

                # Count unique tasks and samples in step3_passed
                step3_unique_tasks = len(step3_passed)
                step3_sample_count = sum(
                    len(responses) for responses in step3_passed.values()
                )
                logger.info(
                    f"Step 3 passed: {step3_sample_count} samples from {step3_unique_tasks} unique tasks"
                )

            else:
                logger.info("Skipping Step 3: Scoring not enabled")
        else:
            logger.info("Skipping Step 2: LLM-as-Judge filtering")

        # Step 4: Apply comprehensive rule-based filtering (moved from Step 0)
        if not skip_rule_based:
            logger.info("Step 4: Comprehensive rule-based filtering (final stage)")
            step4_passed, step4_dropped = self._run_comprehensive_filtering(
                current_responses
            )
            # Compute and log all metrics for step4 (including baseline)
            self.compute_and_log_metrics(
                passed_responses=step4_passed,
                input_ids=set(current_responses.keys()),
                phase="step4",
                human_labelled_questions=current_human_labelled,
                baseline_source=responses_by_question,
            )

            # Count unique tasks and samples in step4_passed
            step4_unique_tasks = len(step4_passed)
            step4_sample_count = sum(
                len(responses) for responses in step4_passed.values()
            )
            logger.info(
                f"Step 4 passed: {step4_sample_count} samples from {step4_unique_tasks} unique tasks"
            )
            # Update human labelled questions to only include those that passed step4
            current_human_labelled = current_human_labelled & set(step4_passed.keys())
        else:
            logger.info("Skipping Step 4: Comprehensive rule-based filtering")

        self._save_results(pipeline_outputs)
        self._print_final_summary(pipeline_outputs)
        return pipeline_outputs

    def _load_benchmark_data(self) -> List[Dict]:
        """Load benchmark data based on configuration."""
        logger.info("Loading benchmark data...")

        target_benchmark = self.config.get("target_benchmark")
        if target_benchmark:
            logger.info(f"Loading only target benchmark: {target_benchmark}")

        all_samples = self.data_loader.load_benchmark_data(
            "benchmark", target_benchmark
        )
        logger.info(f"Loaded {len(all_samples):,} total samples")

        return all_samples

    def _run_comprehensive_filtering(
        self, responses_by_question: Dict[str, List[Dict]]
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """Run Step 4: Comprehensive rule-based filtering."""
        # Save unified dataset before filtering
        all_samples = [
            sample
            for responses in responses_by_question.values()
            for sample in responses
        ]
        self._save_unified_dataset(all_samples)

        logger.info("Applying comprehensive rule-based filtering...")
        rule_filter = ComprehensiveRuleFilter()
        passed_responses_by_question, dropped_responses_by_question = (
            rule_filter.filter_samples(responses_by_question)
        )

        passed_count = sum(
            len(responses) for responses in passed_responses_by_question.values()
        )
        logger.info(f"Step 0 completed: {passed_count:,} samples passed")
        return passed_responses_by_question, dropped_responses_by_question

    def _run_benchmark_specific_filtering(
        self, responses_by_question: Dict[str, List[Dict]]
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """Run Step 1: Benchmark-specific filtering."""
        logger.info(
            "Step 1: Applying benchmark-specific filtering for benchmarks with specific filters..."
        )

        # Group responses by benchmark, but flatten to samples for compatibility with existing filters
        benchmark_groups = {}
        for responses in responses_by_question.values():
            benchmark_name = responses[0].get("benchmark_name", "unknown")
            if benchmark_name not in benchmark_groups:
                benchmark_groups[benchmark_name] = []
            benchmark_groups[benchmark_name].extend(responses)

        all_passed_samples = []
        all_dropped_samples = []

        # Process each benchmark
        for benchmark_name, benchmark_samples in benchmark_groups.items():
            if benchmark_name in self.orchestrator.benchmark_filters:
                logger.info(
                    f"Applying {str(benchmark_name)}-specific filtering to {len(benchmark_samples)} samples"
                )
                passed_samples, dropped_samples = self.orchestrator.filter_samples(
                    benchmark_samples,
                    use_specific_filters=True,
                    target_benchmark=benchmark_name,
                )
                all_passed_samples.extend(passed_samples)
                all_dropped_samples.extend(dropped_samples)
                logger.info(
                    f"{benchmark_name}: {len(passed_samples)} passed, {len(dropped_samples)} dropped"
                )
            else:
                # No specific filter available, keep all samples from this benchmark
                logger.info(
                    f"No specific filter for {benchmark_name}, keeping all {len(benchmark_samples)} samples"
                )
                all_passed_samples.extend(benchmark_samples)

        # Convert back to responses_by_question format
        from src.utils import group_responses_by_question

        passed_responses_by_question = group_responses_by_question(all_passed_samples)
        dropped_responses_by_question = group_responses_by_question(all_dropped_samples)

        passed_count = len(all_passed_samples)
        dropped_count = len(all_dropped_samples)
        logger.info(
            f"Step 1 completed: {passed_count:,} samples passed, {dropped_count:,} samples dropped"
        )
        return passed_responses_by_question, dropped_responses_by_question

    def _save_step1_results(self, step1_passed: List[Dict], step1_dropped: List[Dict]):
        """Save all Step 1 results in various formats."""
        self._save_results(step1_passed, step1_dropped, "step1_rule_based")
        self._save_benchmark_specific_results(
            step1_passed, step1_dropped, "step1_rule_based"
        )
        self._save_unified_step1_results(step1_passed, step1_dropped)

    def compute_and_log_metrics(
        self,
        passed_responses: Dict,
        input_ids: set,
        phase: str,
        human_labelled_questions: set,
        baseline_source: Dict = None
    ) -> None:
        """Compute and log all metrics for a given filtering phase.

        Args:
            passed_responses: Dict of responses that passed the filter (contains question IDs as keys)
            input_ids: Set of question IDs that were input to the filter
            phase: Phase name (e.g., 'initial', 'step1', 'step2', 'step3', 'step4')
            human_labelled_questions: Set of all human labelled questions for current filtering stage
            baseline_source: Source data for creating baseline (original responses_by_question)
        """
        benchmark_name = self.config.get("target_benchmark", None)[0]
        skip_measurement = self.config.get("skip_measurement", False)
        if skip_measurement:
            return

        logger.info(f"\n=== Computing Metrics for {phase.upper()} ===")
        summary_key = "original" if phase == "initial" else phase

        # Extract passed_ids from passed_responses
        passed_ids = set(passed_responses.keys()) if passed_responses else set()
        baseline_task_count = len(passed_ids)
        self.metrics_summary[summary_key]["total_num"] = baseline_task_count

        # ============================== save filtered csv ==============================
        self._write_filter_summary(
            passed_ids=passed_ids,
            input_ids=input_ids,
            phase=phase,
        )

        # ============================== model ranking ==============================
        model_ranking, model_performance = self._calculate_model_ranking(passed_responses)
        if phase == "initial":
            self.model_ranking = model_ranking

        # Store model performance data (assuming single benchmark)
        self.metrics_summary[summary_key]["model_performance"] = model_performance

        # ============================== compute embeddings (initial phase only) ==============================
        if phase == "initial":
            embed_file = f"./{benchmark_name}_embed_dict.pkl"
            if os.path.exists(embed_file):
                with open(embed_file, "rb") as f:
                    self.embeddings_dict = pickle.load(f)
            else:
                self._compute_embeddings_dict(passed_responses)
                with open(embed_file, "wb") as f:
                    pickle.dump(self.embeddings_dict, f)

        # ============================== compute Separability ==============================
        sep_list = []
        for i in range(3):
            separability_dict = self._compute_separability(passed_responses)
            phase_label = "before filtering" if phase == "initial" else f"after {phase}"
            logger.info(f"Benchmark separability {phase_label}: {json.dumps(separability_dict, indent=2)}")
            sep_list.append(separability_dict)
        self.metrics_summary[summary_key]["separability"] = sep_list

        # ============================== compute IRT metric ==============================
        if phase == "initial":
            # Initial calculation: compute IRT discrimination for all questions
            irt_file = f"./{benchmark_name}_irt_dict.pkl"
            if os.path.exists(irt_file):
                with open(irt_file, "rb") as f:
                    self.irt_discrimination_dict = pickle.load(f)
            else:
                self.irt_discrimination_dict = compute_irt_metric(passed_responses, threshold=0.5)
                with open(irt_file, "wb") as f:
                    pickle.dump(self.irt_discrimination_dict, f)

            irt_discrimination = np.mean(list(self.irt_discrimination_dict.values()))
            logger.info(f"Benchmark IRT discrimination before filtering: {irt_discrimination:.4f}")
        else:
            passed_irt_values = [self.irt_discrimination_dict[qid] for qid in passed_ids if qid in self.irt_discrimination_dict]
            irt_discrimination = np.mean(passed_irt_values)
            logger.info(f"Benchmark IRT discrimination after {phase}: {irt_discrimination:.4f}")
        self.metrics_summary[summary_key]["irt"] = irt_discrimination

        # ============================== compute diversity metrics ==============================
        diversity_dict = self._compute_diversity(passed_ids)
        self.metrics_summary[summary_key]["diversity"] = diversity_dict
        phase_label = "before filtering" if phase == "initial" else f"for {phase}"
        logger.info(f"Benchmark semantic diversity {phase_label}: {json.dumps(diversity_dict, indent=2)}")

        # ============================== visualize agreement & diversity metrics ==============================
        phase_titles = {
            "initial": "Original Dataset",
            "step1": "After Step 1 (Rule-based Filtering)",
            "step2": "After Step 2 (LLM-as-Judge Filtering)",
            "step3": "After Step 3 (Top-K Selection)",
            "step4": "After Step 4 (Comprehensive Rule-based Filtering)"
        }
        title = phase_titles.get(phase, f"After {phase}")
        filename = f"{phase}_filtered_performance" if phase != "initial" else "original_performance"
        diversity_filename = f"{phase}_filtered_diversity" if phase != "initial" else "original_diversity"

        self._visualize_diversity(passed_ids, title, diversity_filename)
        self._visualize_model_performance(passed_responses, title, filename, model_ranking=model_ranking)

        # ============================== compute agreement ==============================
        # Store agreement stats by benchmark (computed in `_visualize_model_performance`)
        agreement_by_benchmark = {}
        for benchmark_name, benchmark_stats in self._current_agreement_stats.items():
            agreement_by_benchmark[benchmark_name] = {
                "min": float(benchmark_stats["min_agreement"]),
                "max": float(benchmark_stats["max_agreement"]),
                "avg": float(benchmark_stats["avg_agreement"])
            }

        self.metrics_summary[summary_key]["agreement"] = agreement_by_benchmark

        # ============================== compute retention ratio ==============================
        retention_metrics = self.compute_retention_ratio(passed_ids, self.initial_baseline_ids)
        self.metrics_summary[summary_key]["retention_ratio"] = retention_metrics["retention_ratio"]
        self.metrics_summary[summary_key]["subtask_size"] = retention_metrics["subtask_size"]
        # Store complete retention metrics including subtask_details
        self.metrics_summary[summary_key]["retention_metrics"] = retention_metrics
        if phase == "initial":
            self.metrics_summary[summary_key]["question_num"] = len(human_labelled_questions)

        if phase != "initial":
            # Confusion matrix calculation using current human labelled set
            human_alignment_metrics = log_confusion_matrix_human_labelled(
                human_labelled_questions=human_labelled_questions,
                human_labelled_details=self.human_labelled_details,
                passed_ids=passed_ids,
                input_ids=input_ids,
            )
            self.metrics_summary[summary_key]["human_alignment"] = human_alignment_metrics
            # =====================================================================
            # Compute baseline metrics based on same number of samples
            baseline_responses = self._create_task_wise_baseline_sample_set(baseline_source, baseline_task_count)
            baseline_sample_count = sum(len(responses) for responses in baseline_responses.values())
            logger.info(f"Created task-wise baseline sample set with {baseline_sample_count} samples from {baseline_task_count} tasks")

            # Compute baseline separability (run 3 times for stability)
            sep_list = []
            for i in range(3):
                baseline_separability = self._compute_separability(baseline_responses)
                logger.info(f"Benchmark separability baseline (vs. {phase}): {json.dumps(baseline_separability, indent=2)}")
                sep_list.append(baseline_separability)
            self.metrics_summary[summary_key]["separability_baseline"] = sep_list

            stored_model_ranking = getattr(self, 'model_ranking', None)
            self._visualize_model_performance(
                baseline_responses,
                "Baseline (Random Sampling)",
                f"{phase}_baseline_performance",
                model_ranking=stored_model_ranking
            )
            
            self._visualize_diversity(
                set(baseline_responses.keys()),
                "Baseline (Random Sampling)",
                f"{phase}_baseline_diversity"
            )
        
        output_path = self.config.get("report_filename")
        self.write_metrics_to_csv(output_path)

    def _compute_separability(
        self,
        responses_by_question: Dict[str, List[Dict]],
        n_bootstrap: int = 10000,
        ci: float = 0.95,
    ) -> float:
        score_dict = {}
        separability_dict = {}
        for responses in responses_by_question.values():
            for sample in responses:
                model_name = sample["model_path"]
                benchmark_name = str(sample["benchmark_name"])
                score = sample["eval_result"]["score"]
                if benchmark_name not in score_dict:
                    score_dict[benchmark_name] = {}
                if model_name not in score_dict[benchmark_name]:
                    score_dict[benchmark_name][model_name] = []
                score_dict[benchmark_name][model_name].append(score)

        for benchmark in score_dict:
            # Calculate mean scores for each model

            models_with_scores = sorted(
                [model for model, scores in score_dict[benchmark].items() if scores]
            )
            if len(models_with_scores) < 2:
                separability_dict[benchmark] = (
                    1.0 if len(models_with_scores) < 2 else 0.0
                )
                continue

            score_matrix = [
                score_dict[benchmark][model] for model in models_with_scores
            ]
            num_models = len(score_matrix)
            intervals = []

            for i in range(num_models):
                i_ci = self._bootstrap_confidence_interval(
                    np.array(score_matrix[i]), n_bootstrap=n_bootstrap, ci=ci
                )
                intervals.append(i_ci)
            intervals.sort(key=lambda x: x[0])

            overlapping_pairs = []
            total_pairs = comb(num_models, 2)
            for i in range(len(intervals)):
                for j in range(i + 1, len(intervals)):
                    # If the start time of the second interval is less than the end time of the first, they overlap
                    if intervals[j][0] < intervals[i][1]:
                        # Check if the pair is already in the list
                        if (intervals[i], intervals[j]) not in overlapping_pairs and (
                            intervals[j],
                            intervals[i],
                        ) not in overlapping_pairs:
                            overlapping_pairs.append((intervals[i], intervals[j]))
                    else:
                        break
            separability = (
                1 - len(overlapping_pairs) / total_pairs if total_pairs > 0 else 0
            )
            separability_dict[benchmark] = separability

        return separability_dict

    def _bootstrap_confidence_interval(
        self, scores: np.ndarray, n_bootstrap: int = 100, ci: float = 0.95
    ):
        if len(scores) == 0:
            return (0, 0)
        if len(scores) == 1:
            return (scores[0], scores[0])

        n = len(scores)
        means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=n, replace=True)
            means.append(np.mean(sample))
        lower = np.percentile(means, (1 - ci) / 2 * 100)
        upper = np.percentile(means, (1 + ci) / 2 * 100)
        return lower, upper

    # Semantic diversity metric
    def _compute_diversity(
        self, question_ids: set
    ) -> Dict[str, float]:
        """Compute semantic diversity for each benchmark using average pairwise cosine distance.

        Uses pre-computed embeddings for efficiency.
        The metric is bounded in [0, 1] per the expression: (2 / (N * (N - 1))) * sum{i<j} [1 - cos(e_i, e_j)]
        where N is the number of samples and cos(·,·) is cosine similarity.

        Args:
            question_ids: Set of question IDs to compute diversity for
        """
        diversity_dict: Dict[str, float] = {}

        for benchmark_name in self.all_questions:
            # Get pre-computed embeddings for these questions
            embeddings = np.array([self.embeddings_dict[qid] for qid in question_ids if qid.benchmark.value == benchmark_name])

            # Cosine similarity matrix via dot product because embeddings are normalised
            sim_matrix = np.matmul(embeddings, embeddings.T)

            # Extract upper triangle (i < j)
            triu_indices = np.triu_indices(len(embeddings), k=1)
            cosine_sims = sim_matrix[triu_indices]
            avg_distance = np.mean(1 - cosine_sims)

            diversity_dict[benchmark_name] = float(avg_distance)

        return diversity_dict

    def compute_retention_ratio(self, passed_ids: set, baseline_ids: set) -> Dict:
        """Compute retention ratio metrics.

        Args:
            passed_ids: Set of question IDs that passed current filter
            baseline_ids: Set of question IDs from initial baseline (for overall retention ratio)

        Returns:
            Dict: {
                "retention_ratio": float,  # overall retention ratio
                "subtask_size": Dict[str, float],  # retention ratio by task type
                "subtask_details": Dict[str, Dict]  # detailed stats for each subtask
            }
        """
        # Overall retention ratio
        overall_retention_ratio = len(passed_ids) / len(baseline_ids) if len(baseline_ids) > 0 else 0.0

        # Task-level retention ratio
        retention_task_types = [question_id.task_name for question_id in passed_ids]
        baseline_task_types = [question_id.task_name for question_id in baseline_ids]

        retention_stat = Counter(retention_task_types)
        baseline_stat = Counter(baseline_task_types)

        subtask_retention = {}
        subtask_details = {}

        for task_type in baseline_stat:
            base_num = baseline_stat[task_type]
            retention_num = retention_stat.get(task_type, 0)
            ratio = retention_num / base_num if base_num > 0 else 0.0

            subtask_retention[task_type] = ratio
            subtask_details[task_type] = {
                "base_num": base_num,
                "retention_num": retention_num,
                "ratio": ratio
            }

        return {
            "retention_ratio": overall_retention_ratio,
            "subtask_size": subtask_retention,
            "subtask_details": subtask_details
        }

    def _compute_embeddings_dict(self, responses_by_question: Dict[str, List[Dict]]) -> None:
        """Compute embeddings for all questions and store them for reuse.

        This function should be called once at the beginning to compute embeddings
        for all questions in the dataset.

        Args:
            responses_by_question: Dict mapping question IDs to their responses
        """
        logger.info("Computing embeddings for all questions (one-time calculation)...")
        self.embeddings_dict = {}

        # Process each benchmark
        for benchmark_name, question_list in self.all_questions.items():
            texts = []
            question_ids = []

            for question in question_list:
                if question not in responses_by_question:
                    continue
                # Extract text for embedding
                system_prompt = getattr(question, "agent_system_prompt",
                                      getattr(question, "system_prompt", "")
                                    )
                instruction = question.instruction
                text = system_prompt + "\n" + instruction

                texts.append(text)
                question_ids.append(question)

            if len(texts) == 0:
                continue

            # Compute embeddings for this benchmark
            embeddings = self.embedder.encode(
                texts,
                batch_size=self.embedding_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # Store embeddings by question ID (same type as passed_ids)
            for i, question_id in enumerate(question_ids):
                self.embeddings_dict[question_id] = embeddings[i]

    def _visualize_diversity(
        self, question_ids: set, title: str, filename: str
    ):
        """Create visualizations for semantic diversity: 2D embedding projection and pairwise distance histogram.

        Uses pre-computed embeddings for efficiency.

        Args:
            question_ids: Set of question IDs to visualize
            title: Plot title
            filename: Base filename for saved plots
        """
        logger.info(f"Creating diversity visualization: {title}")

        plots_dir = "pipeline_results/plots"
        os.makedirs(plots_dir, exist_ok=True)

        for benchmark_name in self.all_questions:
            # Get pre-computed embeddings for these questions
            embeddings = np.array([self.embeddings_dict[qid] for qid in question_ids if qid.benchmark.value == benchmark_name])
            task_names = [qid.task_name for qid in question_ids if qid.benchmark.value == benchmark_name]
            # 1. 2D Embedding Representation (t-SNE)
            if len(embeddings) > 1:
                tsne = TSNE(
                    n_components=2,
                    random_state=42,
                    perplexity=min(30, len(embeddings) - 1),
                )
                embeddings_2d = tsne.fit_transform(embeddings)

                plt.figure(figsize=(12, 10))

                unique_task_names = sorted(list(set(task_names)))
                palette = sns.color_palette("husl", len(unique_task_names))

                scatter = sns.scatterplot(
                    x=embeddings_2d[:, 0],
                    y=embeddings_2d[:, 1],
                    hue=task_names,
                    hue_order=unique_task_names,
                    palette=palette,
                    alpha=0.7,
                    edgecolor="none",
                )

                plt.title(
                    f"2D t-SNE Embedding Visualization for {benchmark_name}\n({title})",
                    fontsize=14,
                    fontweight="bold",
                )
                plt.xlabel("t-SNE Component 1", fontsize=12)
                plt.ylabel("t-SNE Component 2", fontsize=12)
                plt.grid(True, alpha=0.3)

                if len(unique_task_names) > 10:
                    scatter.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=1)
                    plt.tight_layout(rect=[0, 0, 0.85, 1])
                else:
                    scatter.legend(loc="best")
                    plt.tight_layout()

                safe_benchmark_name = (
                    str(benchmark_name).replace("/", "_").replace(" ", "_")
                )
                plot_filename_tsne = os.path.join(
                    plots_dir, f"{filename}_{safe_benchmark_name}_tsne.png"
                )
                plt.savefig(plot_filename_tsne, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"Saved t-SNE plot: {plot_filename_tsne}")

            # 2. Pairwise Embedding Distance Histogram
            sim_matrix = np.matmul(embeddings, embeddings.T)
            triu_indices = np.triu_indices(len(embeddings), k=1)
            distances = 1 - sim_matrix[triu_indices]

            plt.figure(figsize=(10, 6))
            sns.histplot(distances, bins=50, kde=True)
            mean_dist = np.mean(distances)
            median_dist = np.median(distances)
            plt.axvline(
                mean_dist, color="r", linestyle="--", label=f"Mean: {mean_dist:.3f}"
            )
            plt.axvline(
                median_dist,
                color="g",
                linestyle="-",
                label=f"Median: {median_dist:.3f}",
            )

            plt.title(
                f"Pairwise Embedding Distance Histogram for {benchmark_name}\n({title})",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Cosine Distance (1 - Cosine Similarity)", fontsize=12)
            plt.ylabel("Frequency", fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            safe_benchmark_name = (
                str(benchmark_name).replace("/", "_").replace(" ", "_")
            )
            plot_filename_hist = os.path.join(
                plots_dir, f"{filename}_{safe_benchmark_name}_distance_hist.png"
            )
            plt.savefig(plot_filename_hist, dpi=150)
            plt.close()
            logger.info(f"Saved distance histogram: {plot_filename_hist}")

    def _calculate_model_ranking(
        self, responses_by_question: Dict[str, List[Dict]]
    ) -> tuple[Dict[str, List[str]], Dict]:
        """Calculate model ranking and detailed performance data.

        Returns:
            tuple: (model_ranking_dict, model_performance_dict)
                - model_ranking_dict: Dict[benchmark_name, List[model_name]] sorted by performance
                - model_performance_dict: Dict with overall and subtask performance data
        """
        benchmark_data = {}
        task_data = {}  # For subtask-level performance

        for responses in responses_by_question.values():
            for sample in responses:
                benchmark_name = str(sample["benchmark_name"])
                model_name = sample["model_path"]
                task_name = sample.get("task_name", "unknown")
                score = sample["eval_result"]["score"]

                # Benchmark-level data
                if benchmark_name not in benchmark_data:
                    benchmark_data[benchmark_name] = {}
                if model_name not in benchmark_data[benchmark_name]:
                    benchmark_data[benchmark_name][model_name] = []
                benchmark_data[benchmark_name][model_name].append(score)

                # Task-level data
                if benchmark_name not in task_data:
                    task_data[benchmark_name] = {}
                if task_name not in task_data[benchmark_name]:
                    task_data[benchmark_name][task_name] = {}
                if model_name not in task_data[benchmark_name][task_name]:
                    task_data[benchmark_name][task_name][model_name] = []
                task_data[benchmark_name][task_name][model_name].append(score)

        # Calculate ranking for each benchmark
        model_ranking = {}
        model_performance = {}

        for benchmark_name, model_scores in benchmark_data.items():
            model_means = {}
            for model_name, scores in model_scores.items():
                model_means[model_name] = np.mean(scores)

            # Sort by performance (descending)
            model_ranking[benchmark_name] = sorted(
                model_means.keys(), key=lambda x: model_means[x], reverse=True
            )

            # Store performance data
            model_performance[benchmark_name] = {}
            for model_name, avg_score in model_means.items():
                model_performance[benchmark_name][model_name] = {
                    "overall_score": float(avg_score),
                    "subtask_scores": {}
                }

                # Add subtask scores
                if benchmark_name in task_data:
                    for task_name, task_models in task_data[benchmark_name].items():
                        if model_name in task_models:
                            task_avg = np.mean(task_models[model_name])
                            model_performance[benchmark_name][model_name]["subtask_scores"][task_name] = float(task_avg)

        return model_ranking, model_performance

    def _visualize_model_performance(
        self,
        responses_by_question: Dict[str, List[Dict]],
        title: str,
        filename: str,
        n_bootstrap: int = 100,
        ci: float = 0.95,
        model_ranking: Dict[str, List[str]] = None,
    ):
        """Create bar graph visualization of model-wise performance with confidence intervals."""
        logger.info(f"Creating visualization: {title}")

        # Create output directory for plots
        plots_dir = "pipeline_results/plots"
        os.makedirs(plots_dir, exist_ok=True)

        # Group samples by benchmark
        benchmark_data = {}
        for responses in responses_by_question.values():
            for sample in responses:
                benchmark_name = str(sample["benchmark_name"])
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
                ordered_models = [
                    model
                    for model in model_ranking[benchmark_name]
                    if model in available_models
                ]
                # Add any models not in ranking at the end
                remaining_models = sorted(available_models - set(ordered_models))
                model_order = ordered_models + remaining_models
            else:
                # Calculate ranking based on current data (for original dataset)
                model_means = {}
                for model_name, scores in model_scores.items():
                    model_means[model_name] = np.mean(scores)
                model_order = sorted(
                    model_means.keys(), key=lambda x: model_means[x], reverse=True
                )

            # Calculate means and confidence intervals for each model in the determined order
            model_names = []
            means = []
            ci_lowers = []
            ci_uppers = []

            for model_name in model_order:
                scores = np.array(model_scores[model_name])
                mean_score = np.mean(scores)
                ci_lower, ci_upper = self._bootstrap_confidence_interval(
                    scores, n_bootstrap, ci
                )

                model_names.append(model_name)
                means.append(mean_score)
                ci_lowers.append(ci_lower)
                ci_uppers.append(ci_upper)

            # Create the plot with smaller figure size
            plt.figure(figsize=(10, 6))
            x_pos = np.arange(len(model_names))

            # Create bars with error bars
            bars = plt.bar(
                x_pos,
                means,
                yerr=[
                    np.array(means) - np.array(ci_lowers),
                    np.array(ci_uppers) - np.array(means),
                ],
                capsize=3,
                alpha=0.7,
                color="skyblue",
                edgecolor="navy",
                linewidth=0.8,
            )

            # Customize the plot
            plt.xlabel("Model", fontsize=10, fontweight="bold")
            plt.ylabel("Performance Score", fontsize=10, fontweight="bold")
            plt.title(
                f"{title} - {benchmark_name}\nModel Performance with {int(ci * 100)}% Confidence Intervals",
                fontsize=12,
                fontweight="bold",
                pad=15,
            )

            # Set x-axis labels
            plt.xticks(x_pos, model_names, rotation=45, ha="right", fontsize=9)

            # Add value labels on top of bars (smaller font)
            for bar, mean, ci_low, ci_high in zip(bars, means, ci_lowers, ci_uppers):
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (ci_high - mean) + 0.005,
                    f"{mean:.3f}\n[{ci_low:.3f}, {ci_high:.3f}]",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

            # Add grid for better readability
            plt.grid(True, alpha=0.3, axis="y")

            # Adjust layout to prevent label cutoff
            plt.tight_layout()

            # Save the plot with reduced DPI
            safe_benchmark_name = benchmark_name.replace("/", "_").replace(" ", "_")
            plot_filename = os.path.join(
                plots_dir, f"{filename}_{safe_benchmark_name}.png"
            )
            plt.savefig(
                plot_filename,
                dpi=150,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
                format="png",
            )
            plt.close()

            logger.info(f"Saved plot: {plot_filename}")

            # Create model agreement heatmap
            self._create_model_agreement_heatmap(
                responses_by_question, benchmark_name, title, filename, model_order
            )

    def _create_model_agreement_heatmap(
        self,
        responses_by_question: Dict[str, List[Dict]],
        benchmark_name: str,
        title: str,
        filename: str,
        model_order: List[str],
    ):
        """Create a heatmap showing model agreement based on task correctness."""
        logger.info(f"Creating model agreement heatmap for {benchmark_name}")

        # Create output directory for plots
        plots_dir = "pipeline_results/plots"
        os.makedirs(plots_dir, exist_ok=True)

        # Group responses by question ID and model
        question_model_scores = {}
        for question_id, responses in responses_by_question.items():
            for response in responses:
                if str(response["benchmark_name"]) == benchmark_name:
                    model_name = response["model_path"]
                    score = response["eval_result"]["score"]

                    if question_id not in question_model_scores:
                        question_model_scores[question_id] = {}
                    question_model_scores[question_id][model_name] = score

        # Filter to only include models that are in model_order and have data
        available_models = set()
        for question_id, model_scores in question_model_scores.items():
            available_models.update(model_scores.keys())

        # Use only models that exist in both model_order and available_models
        filtered_model_order = [
            model for model in model_order if model in available_models
        ]

        if len(filtered_model_order) < 2:
            logger.warning(
                f"Not enough models ({len(filtered_model_order)}) to create agreement heatmap for {benchmark_name}"
            )
            return

        # Calculate agreement matrix
        n_models = len(filtered_model_order)
        agreement_matrix = np.zeros((n_models, n_models))

        # For each pair of models, calculate agreement
        for i, model1 in enumerate(filtered_model_order):
            for j, model2 in enumerate(filtered_model_order):
                if i == j:
                    agreement_matrix[i, j] = 1.0  # Perfect agreement with self
                else:
                    agreements = 0
                    total_comparisons = 0

                    # Compare on each question where both models have responses
                    for question_id, model_scores in question_model_scores.items():
                        if model1 in model_scores and model2 in model_scores:
                            score1 = model_scores[model1]
                            score2 = model_scores[model2]

                            # Convert scores to binary correctness (assuming score > 0.5 means correct)
                            correct1 = 1 if score1 > 0.5 else 0
                            correct2 = 1 if score2 > 0.5 else 0

                            # Agreement if both correct or both incorrect
                            if correct1 == correct2:
                                agreements += 1
                            total_comparisons += 1

                    if total_comparisons > 0:
                        agreement_matrix[i, j] = agreements / total_comparisons
                    else:
                        agreement_matrix[i, j] = 0.0

        # Create the heatmap
        plt.figure(figsize=(max(8, n_models * 0.8), max(6, n_models * 0.6)))

        # Create heatmap with custom colormap
        sns.heatmap(
            agreement_matrix,
            xticklabels=filtered_model_order,
            yticklabels=filtered_model_order,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Agreement Rate"},
            square=True,
            linewidths=0.5,
            annot_kws={"fontsize": 8},
        )

        plt.title(
            f"{title} - {benchmark_name}\nModel Agreement Heatmap",
            fontsize=12,
            fontweight="bold",
            pad=15,
        )
        plt.xlabel("Model", fontsize=10, fontweight="bold")
        plt.ylabel("Model", fontsize=10, fontweight="bold")

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)

        plt.tight_layout()

        # Save the heatmap
        safe_benchmark_name = benchmark_name.replace("/", "_").replace(" ", "_")
        heatmap_filename = os.path.join(
            plots_dir, f"{filename}_{safe_benchmark_name}_agreement_heatmap.png"
        )
        plt.savefig(
            heatmap_filename,
            dpi=150,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
            format="png",
        )
        plt.close()

        logger.info(f"Saved agreement heatmap: {heatmap_filename}")

        # Log some summary statistics
        # Calculate average agreement (excluding diagonal)
        mask = ~np.eye(n_models, dtype=bool)
        avg_agreement = np.mean(agreement_matrix[mask])
        min_agreement = np.min(agreement_matrix[mask])
        max_agreement = np.max(agreement_matrix[mask])

        logger.info(f"Agreement statistics for {benchmark_name}:")
        logger.info(f"  Average agreement: {avg_agreement:.3f}")
        logger.info(f"  Min agreement: {min_agreement:.3f}")
        logger.info(f"  Max agreement: {max_agreement:.3f}")

        # Store agreement statistics for final summary
        if not hasattr(self, "_current_agreement_stats"):
            self._current_agreement_stats = {}
        self._current_agreement_stats[benchmark_name] = {
            "avg_agreement": avg_agreement,
            "min_agreement": min_agreement,
            "max_agreement": max_agreement,
            "num_models": n_models,
        }

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
        if "task_name" in sample and "meta" in sample and "id" in sample["meta"]:
            return sample["task_name"] + "_" + sample["meta"]["id"]
        elif "task_name" in sample:
            return sample["task_name"]
        elif "question" in sample:
            return sample["question"]
        elif "prompt" in sample:
            return sample["prompt"]
        elif "messages" in sample and sample["messages"]:
            # Use first message content as task identifier
            first_msg = sample["messages"][0]
            if isinstance(first_msg, dict) and "content" in first_msg:
                return first_msg["content"][:100]  # First 100 chars
        elif "conversation" in sample and sample["conversation"]:
            # Use first turn as task identifier
            first_turn = sample["conversation"][0]
            if isinstance(first_turn, dict) and "content" in first_turn:
                return first_turn["content"][:100]

        # Fallback: use a hash of the entire sample
        import hashlib

        return hashlib.md5(json.dumps(sample, sort_keys=True).encode()).hexdigest()

    def _create_task_wise_baseline_sample_set(
        self, responses_by_question: Dict[str, List[Dict]], target_task_count: int
    ) -> Dict[str, List[Dict]]:
        """Create a baseline sample set by randomly sampling N tasks from original samples."""
        total_tasks = len(responses_by_question)
        total_samples = sum(
            len(responses) for responses in responses_by_question.values()
        )
        logger.info(
            f"Creating task-wise baseline: sampling {target_task_count} tasks from {total_samples} total samples ({total_tasks} unique tasks)"
        )
        print("Total tasks for BASELINE: ", target_task_count)
        if target_task_count >= total_tasks:
            logger.info("Target task count >= total tasks, returning all samples")
            return responses_by_question.copy()

        # Random sample with fixed seed for reproducibility
        random.seed(42)
        selected_task_ids = random.sample(
            list(responses_by_question.keys()), target_task_count
        )
        random.seed()  # Reset seed
        # print("Selected task ids for BASELINE: ", [(task_id.task_name, task_id.question_id) for task_id in selected_task_ids])
        # Create baseline using dict comprehension
        baseline = {
            task_id: responses_by_question[task_id] for task_id in selected_task_ids
        }

        baseline_samples = sum(len(responses) for responses in baseline.values())
        logger.info(
            f"Task-wise baseline created: {baseline_samples} samples from {len(selected_task_ids)} tasks"
        )
        return baseline

    def _write_filter_summary(
        self, passed_ids: set, input_ids: set, phase: str
    ):
        """Record filtering summary for each phase.

        Args:
            passed_ids: Set of question IDs that passed current phase
            input_ids: Set of question IDs that input to current phase
            phase: Phase name ("initial", "step0", "step1", "step2", "step3", "final")
        """
        logger.info(f"Recording filter summary for phase: {phase}")

        if phase == "initial":
            # Initialize all questions with basic info
            for question_id in passed_ids:
                if question_id not in self.filtering_summary:
                    # Create new entry from template
                    entry = self.fitering_template.copy()

                    # Fill basic info from question_id
                    entry["Benchmark"] = (
                        question_id.benchmark.value
                        if hasattr(question_id.benchmark, "value")
                        else str(question_id.benchmark)
                    )
                    entry["task_type"] = (
                        question_id.task_name if question_id.task_name else ""
                    )
                    entry["task_id"] = question_id.question_id

                    self.filtering_summary[question_id] = entry

        else:
            # Mark questions that were NOT in passed_ids as failed for this phase
            field_map = {
                "step1": "specific_rule_passed",
                "step2": "specific_llm_passed",
                "step3": "topk_selection_passed",
                "step4": "comp_passed",
            }

            if phase in field_map:
                field_name = field_map[phase]
                for question_id in input_ids:
                    entry = self.filtering_summary[question_id]

                    # If this question is not in passed_ids and hasn't been marked as failed yet
                    if question_id not in passed_ids:
                        entry[field_name] = False
                    else:
                        entry[field_name] = True

            # Use the new compute_retention_ratio function
            retention_metrics = self.compute_retention_ratio(passed_ids, self.initial_baseline_ids)

            logging.info(f"==================== Retention ratio after {phase} ====================")
            logging.info(f"Overall retention: {len(passed_ids)}/{len(self.initial_baseline_ids)} = {retention_metrics['retention_ratio']*100:.2f}%")

            for task_type, ratio in retention_metrics['subtask_size'].items():
                logging.info(f"{task_type}: {ratio*100:.2f}%")

        self._save_filter_summary_csv()

    def _save_filter_summary_csv(self):
        """Save filtering summary to CSV file, ordered by filtering stage (latest filtered first)."""
        if not self.filtering_summary:
            logger.info("No filtering summary to save")
            return

        # Sort questions by when they were filtered (reverse order - latest filtered first)
        def get_filter_stage(entry):
            # Return stage number where question was filtered (higher = later)
            order = 0
            for stage in ["topk_selection_passed", "specific_llm_passed", "specific_rule_passed", "comp_passed"]:
                if entry[stage] == True:
                    order += 1
            if entry.get("is_issue") == True:
                order += 0.5
            return order

        # Sort by filter stage (descending) then by question_id for consistency
        sorted_items = sorted(
            self.filtering_summary.items(),
            key=lambda x: (get_filter_stage(x[1]), str(x[0])),
            reverse=True,
        )

        csv_path = self.config.get("csv_filename")

        # Write CSV
        fieldnames = list(self.fitering_template.keys())

        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for question_id, entry in sorted_items:
                row = entry.copy()

                # Replace None values with empty strings for CSV
                for key, value in row.items():
                    if value is None:
                        row[key] = ""

                writer.writerow(row)

        logger.info(f"Filtering summary saved to: {csv_path}")

    def _save_results(self, pipeline_outputs: Dict[UniqueQuestionID, PipelineOutput]):
        """Save results for the pipeline."""
        logger.info("Saving pipeline results...")

        # Save pipeline outputs
        timestamp = datetime.now().strftime("%m%d_%H%M")
        output_dir = "pipeline_results"
        os.makedirs(output_dir, exist_ok=True)
        results_filename = os.path.join(output_dir, f"result_{timestamp}.jsonl")
        with open(results_filename, "w") as f:
            for question_id, output in pipeline_outputs.items():
                result_dict = {**question_id.model_dump(), **output.model_dump()}
                serializable_result = self._make_json_serializable(result_dict)
                f.write(json.dumps(serializable_result) + "\n")

        logger.info(f"Pipeline results saved to {results_filename}")

    def _save_benchmark_specific_results(
        self, passed_samples: List[Dict], dropped_samples: List[Dict], step_name: str
    ):
        """Save benchmark-specific passed and pruned files after step 1."""
        logger.info(f"Saving benchmark-specific {step_name} results...")

        # Create output directory
        output_dir = "pipeline_results"
        os.makedirs(output_dir, exist_ok=True)

        # Group samples by benchmark
        benchmark_passed = {}
        benchmark_dropped = {}

        for sample in passed_samples:
            benchmark_name = sample.get("benchmark_name", "unknown")
            if benchmark_name not in benchmark_passed:
                benchmark_passed[benchmark_name] = []
            benchmark_passed[benchmark_name].append(sample)

        for sample in dropped_samples:
            benchmark_name = sample.get("benchmark_name", "unknown")
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
                logger.info(
                    f"Saved {len(samples):,} passed samples for {benchmark_name} to {filename}"
                )

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
                logger.info(
                    f"Saved {len(samples):,} pruned samples for {benchmark_name} to {filename}"
                )

        logger.info(
            f"Benchmark-specific results saved for {len(benchmark_passed)} benchmarks"
        )

    def write_metrics_to_csv(self, output_path: str = "pipeline_results/metrics_summary.csv"):
        """Write metrics_summary to CSV file with benchmark comparisons and model performance."""
        logger.info(f"Writing metrics summary to {output_path}")

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        rows = []

        # Helper function to handle None values and format numbers
        def format_value(value):
            if value is None:
                return "-"
            if isinstance(value, (int, float)):
                return f"{value:.3f}"
            return str(value)

        # Header row for metrics comparison
        rows.append(["Metrics Comparison", "Baseline", "Step1", "Step2", "Step4"])

        # Get baseline data and benchmark name (from "original" step)
        baseline_data = self.metrics_summary.get("original", {})
        steps = ["step1", "step2", "step4"]

        # Get benchmark name from the first available data (assuming single benchmark)
        benchmark_name = list(baseline_data["diversity"].keys())[0]

        # Add benchmark name row
        rows.append([f"Benchmark: {benchmark_name or 'Unknown'}", "", "", "", ""])
        rows.append([])  # Empty row for separation

        # Agreement in format (avg/min/max) - extract using benchmark_name
        agreement_row = ["Agreement (avg/min/max)"]
        baseline_agreement_data = baseline_data.get("agreement", {})
        baseline_bench_data = baseline_agreement_data[benchmark_name]
        baseline_avg = baseline_bench_data.get("avg")
        baseline_min = baseline_bench_data.get("min")
        baseline_max = baseline_bench_data.get("max")
        baseline_formatted = f"{format_value(baseline_avg)}/{format_value(baseline_min)}/{format_value(baseline_max)}"
        agreement_row.append(baseline_formatted)

        for step in steps:
            step_data = self.metrics_summary.get(step, {})
            current_agreement_data = step_data.get("agreement", {})
            current_bench_data = current_agreement_data.get(benchmark_name, {})
            current_avg = current_bench_data.get("avg")
            current_min = current_bench_data.get("min")
            current_max = current_bench_data.get("max")
            current_formatted = f"{format_value(current_avg)}/{format_value(current_min)}/{format_value(current_max)}"
            agreement_row.append(current_formatted)
        rows.append(agreement_row)

        # CI Overlap (separability with step-specific baselines)
        ci_overlap_row = ["CI Overlap (sampled_baseline/after_step)"]
        # For baseline column, use separability from original step
        baseline_separability = baseline_data.get("separability", [])
        baseline_sep_value = baseline_separability[0][benchmark_name]
        ci_overlap_row.append(format_value(baseline_sep_value))

        for step in steps:
            step_data = self.metrics_summary.get(step, {})
            # Current step separability
            current_separability = step_data.get("separability", [])
            current_sep_value = current_separability[0].get(benchmark_name) if len(current_separability) > 0 else None

            # Step-specific baseline separability
            step_baseline_separability = step_data.get("separability_baseline", [])
            step_baseline_value = step_baseline_separability[0].get(benchmark_name) if len(step_baseline_separability) > 0 else None

            baseline_val = format_value(step_baseline_value)
            current_val = format_value(current_sep_value)
            ci_overlap_row.append(f"{baseline_val}/{current_val}")
        rows.append(ci_overlap_row)

        # Diversity (extract value using benchmark_name)
        diversity_row = ["Diversity"]
        baseline_div_value = baseline_data["diversity"][benchmark_name]
        diversity_row.append(format_value(baseline_div_value))

        for step in steps:
            current_diversity = self.metrics_summary[step].get("diversity", {})
            current_div_value = current_diversity.get(benchmark_name) if current_diversity is not None else None
            diversity_row.append(format_value(current_div_value))
        rows.append(diversity_row)

        # IRT (direct values, no baseline comparison)
        irt_row = ["IRT"]
        baseline_irt = baseline_data.get("irt")
        irt_row.append(format_value(baseline_irt))
        for step in steps:
            step_data = self.metrics_summary.get(step, {})
            current_irt = step_data.get("irt")
            irt_row.append(format_value(current_irt))
        rows.append(irt_row)

        # Precision
        precision_row = ["Precision"]
        precision_row.append("-")  # No baseline precision
        for step in steps:
            step_data = self.metrics_summary.get(step, {})
            human_alignment = step_data.get("human_alignment", {})
            precision_row.append(format_value(human_alignment.get("precision")))
        rows.append(precision_row)

        # Recall
        recall_row = ["Recall"]
        recall_row.append("-")  # No baseline recall
        for step in steps:
            step_data = self.metrics_summary.get(step, {})
            human_alignment = step_data.get("human_alignment", {})
            recall_row.append(format_value(human_alignment.get("recall")))
        rows.append(recall_row)

        # Question Num (TP + FP + TN + FN from human alignment)
        question_num_row = ["Question Num"]
        question_num_row.append(self.metrics_summary["original"]["question_num"])
        for step in steps:
            step_data = self.metrics_summary.get(step, {})
            human_alignment = step_data.get("human_alignment", {})
            tn = human_alignment.get("tn", 0) or 0
            fn = human_alignment.get("fn", 0) or 0
            question_remain = tn + fn
            question_num_row.append(str(question_remain) if question_remain > 0 else "-")
        rows.append(question_num_row)

        # Total Num (from metrics_summary.total_num)
        total_num_row = ["Total Num"]
        baseline_total_num = baseline_data.get("total_num")
        total_num_row.append(str(baseline_total_num))
        for step in steps:
            step_data = self.metrics_summary.get(step, {})
            current_total_num = step_data.get("total_num")
            total_num_row.append(str(current_total_num))
        rows.append(total_num_row)

        # Retention Ratio
        retention_row = ["Retention Ratio"]
        retention_row.append("-")  # No baseline retention ratio
        for step in steps:
            step_data = self.metrics_summary.get(step, {})
            retention_ratio = step_data.get("retention_ratio")
            retention_row.append(format_value(retention_ratio))
        rows.append(retention_row)

        # Add empty rows for separation
        rows.append([])
        rows.append([])

        # Model Performance section
        rows.append(["Model Performance"])
        rows.append([])

        # Get all models and subtasks from the data
        all_models = set()
        all_subtasks = set()

        # Include baseline data
        baseline_model_performance = baseline_data.get("model_performance", {})
        for benchmark_data in baseline_model_performance.values():
            for model_name, model_data in benchmark_data.items():
                all_models.add(model_name)
                subtask_scores = model_data.get("subtask_scores", {})
                all_subtasks.update(subtask_scores.keys())

        # Include step data
        for step in steps:
            step_data = self.metrics_summary.get(step, {})
            model_performance = step_data.get("model_performance", {})

            # Model performance is nested: {benchmark: {model: {scores...}}}
            for benchmark_data in model_performance.values():
                for model_name, model_data in benchmark_data.items():
                    all_models.add(model_name)
                    subtask_scores = model_data.get("subtask_scores", {})
                    all_subtasks.update(subtask_scores.keys())

        all_models = sorted(list(all_models))
        all_subtasks = sorted(list(all_subtasks))

        # Create baseline model performance table
        baseline_model_performance = baseline_data.get("model_performance", {})
        if baseline_model_performance:
            rows.append(["BASELINE - Model Performance"])

            # Headers: Model, Overall, subtask1, subtask2, ...
            header = ["Model", "Overall"] + all_subtasks
            rows.append(header)

            # Get benchmark data (using benchmark_name)
            benchmark_data = baseline_model_performance.get(benchmark_name, {}) if benchmark_name else {}

            # Model data rows (sorted by overall score descending)
            model_rows = []
            for model_name in all_models:
                if model_name in benchmark_data:
                    model_data = benchmark_data[model_name]
                    overall_score_raw = model_data.get("overall_score")
                    overall_score = format_value(overall_score_raw)
                    subtask_scores = model_data.get("subtask_scores", {})

                    row = [model_name, overall_score]
                    for subtask in all_subtasks:
                        row.append(format_value(subtask_scores.get(subtask)))
                    model_rows.append((overall_score_raw or 0, row))

            # Sort by overall score (descending) and add to rows
            model_rows.sort(key=lambda x: x[0], reverse=True)
            for _, row in model_rows:
                rows.append(row)

            rows.append([])  # Empty row after baseline

        # Create model performance table for each step
        for step in steps:
            step_data = self.metrics_summary.get(step, {})
            model_performance = step_data.get("model_performance", {})

            if not model_performance:
                continue

            # Step header
            rows.append([f"{step.upper()} - Model Performance"])

            # Headers: Model, Overall, subtask1, subtask2, ...
            header = ["Model", "Overall"] + all_subtasks
            rows.append(header)

            # Get benchmark data (using benchmark_name)
            benchmark_data = model_performance.get(benchmark_name, {}) if benchmark_name else {}

            # Model data rows (sorted by overall score descending)
            model_rows = []
            for model_name in all_models:
                if model_name in benchmark_data:
                    model_data = benchmark_data[model_name]
                    overall_score_raw = model_data.get("overall_score")
                    overall_score = format_value(overall_score_raw)
                    subtask_scores = model_data.get("subtask_scores", {})

                    row = [model_name, overall_score]
                    for subtask in all_subtasks:
                        row.append(format_value(subtask_scores.get(subtask)))
                    model_rows.append((overall_score_raw or 0, row))

            # Sort by overall score (descending) and add to rows
            model_rows.sort(key=lambda x: x[0], reverse=True)
            for _, row in model_rows:
                rows.append(row)

            # Add subtask statistics table after model performance
            rows.append([])  # Empty row before subtask stats
            rows.append([f"{step.upper()} - Subtask Statistics"])

            # Get subtask details from retention ratio computation
            retention_metrics = step_data.get("retention_metrics")
            if isinstance(retention_metrics, dict) and "subtask_details" in retention_metrics:
                subtask_details = retention_metrics["subtask_details"]

                # Headers: Subtask, Base Num, Retention Num, Ratio
                subtask_header = ["Subtask", "Base Num", "Retention Num", "Ratio"]
                rows.append(subtask_header)

                # Data rows
                for subtask_name, details in subtask_details.items():
                    base_num = details.get("base_num", 0)
                    retention_num = details.get("retention_num", 0)
                    ratio = details.get("ratio", 0.0)

                    subtask_row = [
                        subtask_name,
                        str(base_num),
                        str(retention_num),
                        format_value(ratio)
                    ]
                    rows.append(subtask_row)
            else:
                # If no subtask details available, show placeholder
                rows.append(["No subtask statistics available"])

            rows.append([])  # Empty row between steps

        # Write to CSV
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for row in rows:
                writer.writerow(row)

        logger.info(f"Metrics summary CSV written to {output_path}")

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

        logger.info(
            f"Unified dataset saved to {unified_filename} with {len(all_samples):,} samples"
        )
        return unified_filename

    def _save_unified_step1_results(
        self, passed_samples: List[Dict], dropped_samples: List[Dict]
    ):
        """Save unified passed and pruned files after step 1."""
        logger.info("Saving unified Step 1 results...")

        # Create output directory
        output_dir = "pipeline_results"
        os.makedirs(output_dir, exist_ok=True)

        # Save unified passed samples
        unified_passed_filename = os.path.join(
            output_dir, "unified_step1_passed_samples.jsonl"
        )
        with open(unified_passed_filename, "w") as f:
            for sample in passed_samples:
                serializable_sample = self._make_json_serializable(sample)
                f.write(json.dumps(serializable_sample) + "\n")
        logger.info(
            f"Unified passed samples saved to {unified_passed_filename} with {len(passed_samples):,} samples"
        )

        # Save unified dropped samples
        unified_dropped_filename = os.path.join(
            output_dir, "unified_step1_dropped_samples.jsonl"
        )
        with open(unified_dropped_filename, "w") as f:
            for sample in dropped_samples:
                serializable_sample = self._make_json_serializable(sample)
                f.write(json.dumps(serializable_sample) + "\n")
        logger.info(
            f"Unified dropped samples saved to {unified_dropped_filename} with {len(dropped_samples):,} samples"
        )

    def _print_final_summary(
        self, pipeline_outputs: Dict[UniqueQuestionID, PipelineOutput]
    ):
        logger.info("Pipeline completed - Final summary")

        step1_passed = [
            qid
            for qid, output in pipeline_outputs.items()
            if not output.rule_based_output or output.rule_based_output.passed
        ]

        step2_passed = [
            qid
            for qid in step1_passed
            if not self._is_question_flawed(pipeline_outputs[qid].llm_judge_output)
        ]

        step1_benchmarks = Counter(qid.benchmark.value for qid in step1_passed)
        step2_benchmarks = Counter(qid.benchmark.value for qid in step2_passed)

        logger.info(f"\nStep 1 (Rule-based) results:")
        for benchmark, count in sorted(step1_benchmarks.items()):
            logger.info(f"  {benchmark}: {count:,}")

        logger.info(f"\nStep 2 (LLM-as-Judge) results:")
        for benchmark, count in sorted(step2_benchmarks.items()):
            logger.info(f"  {benchmark}: {count:,}")

        logger.info(f"\nUnique questions:")
        logger.info(f"  After Step 1: {len(step1_passed):,}")
        logger.info(f"  After Step 2: {len(step2_passed):,}")

        # Print compiled metrics summary
        self._print_compiled_metrics_summary()

        logger.info(f"\nPipeline complete!")

    def _print_compiled_metrics_summary(self):
        """Print a compiled summary of all metrics collected during the pipeline."""
        logger.info(f"\n{'=' * 80}")
        logger.info(f"COMPILED METRICS SUMMARY")
        logger.info(f"{'=' * 80}")

        step_names = {
            "original": "Original Dataset",
            "step1": "After Step 1 (Rule-based)",
            "step2": "After Step 2 (LLM-as-Judge)",
            "step3": "After Step 3 (Top-K Selection)",
            "step4": "After Step 4 (Comprehensive Rule-based)",
            "step1_baseline": "Step 1 Baseline",
            "step2_baseline": "Step 2 Baseline",
            "step3_baseline": "Step 3 Baseline",
            "step4_baseline": "Step 4 Baseline",
        }

        for step_key, step_name in step_names.items():
            if step_key not in self.metrics_summary:
                continue

            step_metrics = self.metrics_summary[step_key]
            if not any(step_metrics.values()):  # Skip if no metrics available
                continue

            logger.info(f"\n{step_name}:")
            logger.info(f"{'-' * 50}")

            available_keys = set(step_metrics.keys())

            # Separability
            if "separability" in available_keys and step_metrics["separability"]:
                logger.info(f"Separability by benchmark:")
                # Handle both dict and list of dicts (for repeated runs)
                if isinstance(step_metrics["separability"], dict):
                    sep_dicts = [step_metrics["separability"]]
                else:
                    sep_dicts = step_metrics["separability"]
                # Print each run
                for run_idx, sep_dict in enumerate(sep_dicts):
                    if len(sep_dicts) > 1:
                        logger.info(f"  Run {run_idx + 1}:")
                    for benchmark, separability in sep_dict.items():
                        logger.info(f"    {benchmark}: {separability:.3f}")
                    # avg_separability = np.mean(list(sep_dict.values()))
                    # logger.info(f"    Average: {avg_separability:.3f}")
            else:
                logger.info(f"Separability: Not computed")

            # Only print diversity if present
            if "diversity" in available_keys and step_metrics["diversity"]:
                logger.info(f"Semantic diversity by benchmark:")
                for benchmark, diversity in step_metrics["diversity"].items():
                    logger.info(f"  {benchmark}: {diversity:.3f}")
                # avg_diversity = np.mean(list(step_metrics["diversity"].values()))
                # logger.info(f"  Average: {avg_diversity:.3f}")
            elif "diversity" in available_keys:
                logger.info(f"Semantic diversity: Not computed")

            # Only print agreement if present
            if "agreement" in available_keys and step_metrics["agreement"]:
                logger.info(f"Model agreement statistics:")
                for benchmark, stats in step_metrics["agreement"].items():
                    logger.info(f"  {benchmark}:")
                    logger.info(f"    Average agreement: {stats['avg']:.3f}")
                    logger.info(f"    Min agreement: {stats['min']:.3f}")
                    logger.info(f"    Max agreement: {stats['max']:.3f}")
            elif "agreement" in available_keys:
                logger.info(f"Model agreement: Not computed")

    def _count_unique_questions(self, samples: List[Dict]) -> int:
        """Count unique questions in samples."""
        return len(self._convert_response_list_to_qid_list(samples))

    def _filter_responses_by_benchmark(
        self, responses: List[Dict], benchmark_name: str
    ) -> List[Dict]:
        if benchmark_name not in list(benchmark.value for benchmark in Benchmark):
            raise ValueError(f"Invalid benchmark name {benchmark_name}")

        result = []
        for response in responses:
            if (
                response["benchmark_name"].value == benchmark_name
            ):  # benchmark_name is enum, compare with .value
                result.append(response)
        return result

    def _filter_questions_by_benchmark(
        self, questions: List[UniqueQuestionID], benchmark: Benchmark
    ) -> List[UniqueQuestionID]:
        result = []
        for question in questions:
            if question.benchmark == benchmark:
                result.append(question)
        return result

    def _run_llm_judge(
        self, responses_by_question: Dict[UniqueQuestionID, List[Dict]]
    ) -> Dict[UniqueQuestionID, LLMJudgeOutput]:
        """Run LLM judge independently on questions from benchmark datasets."""
        # Determine which benchmarks to process based on target_benchmark config
        judge = LLMJudge(self.llm_config)
        return judge.judge_questions(responses_by_question)

    def _run_step3_top_k_selection(
        self,
        step2_result: Dict[UniqueQuestionID, LLMJudgeOutput],
        responses_by_question: Dict[UniqueQuestionID, List[Dict]],
        top_k: int,
    ) -> Dict[UniqueQuestionID, List[Dict]]:
        """Select top-K samples based on total scores from LLM judge results."""

        # Create list of (question_id, total_score) pairs
        scored_questions = []
        for question_id, llm_output in step2_result.items():
            if llm_output.scores and "total_score" in llm_output.scores:
                total_score = llm_output.scores["total_score"]
                scored_questions.append((question_id, total_score))
            else:
                logger.warning(f"No total_score found for question {question_id}")

        # Sort by total_score in descending order and select top-K
        scored_questions.sort(key=lambda x: x[1], reverse=True)
        top_k_questions = [q[0] for q in scored_questions[:top_k]]

        logger.info(
            f"Selected top {len(top_k_questions)} questions out of {len(scored_questions)} scored questions"
        )
        if scored_questions:
            logger.info(
                f"Score range: {scored_questions[0][1]:.3f} (highest) to {scored_questions[-1][1]:.3f} (lowest)"
            )
            if top_k_questions:
                logger.info(
                    f"Top-K score range: {scored_questions[0][1]:.3f} to {scored_questions[top_k - 1][1]:.3f}"
                )

        # Collect responses for selected questions
        step3_responses_by_question = {}
        for question_id in top_k_questions:
            if question_id in responses_by_question:
                step3_responses_by_question[question_id] = responses_by_question[
                    question_id
                ]

        return step3_responses_by_question


def main():
    """Main entry point."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    parser = argparse.ArgumentParser(description="Benchmark Filtering Pipeline")
    parser.add_argument(
        "--skip-llm-judge",
        action="store_true",
        help="Skip Step 2 (LLM-as-Judge filtering)",
    )
    parser.add_argument(
        "--skip-rule-based",
        action="store_true",
        help="Skip Step 1 (rule-based filtering) and run LLM judge on questions independently",
    )
    parser.add_argument(
        "--llm-model",
        default="google/gemini-2.5-pro-thinking-on",
        help="LLM model to use for Step 2 (default: google/gemini-2.5-pro-thinking-on)",
    )
    parser.add_argument(
        "--llm-max-samples",
        type=int,
        help="Maximum samples to process in Step 2 (default: all)",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of processes for multiprocessing (default: 1)",
    )
    parser.add_argument(
        "--target-benchmark",
        nargs="+",
        choices=[benchmark.value for benchmark in Benchmark],
        help="Target benchmark(s) to process (default: all available benchmarks)",
    )
    parser.add_argument(
        "--llm-filter-mode",
        choices=["common", "specific", "both"],
        default="specific",
        help="LLM filtering scheme: 'common' (universal filter only), 'specific' (benchmark-specific filter only), 'both' (universal + specific filters)",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip scoring step even when using 'both' filtering scheme",
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-8B",
        help="SentenceTransformer model for semantic diversity computation (default: Qwen/Qwen3-Embedding-8B)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=4,
        help="Batch size when encoding texts for diversity (default: 8)",
    )

    parser.add_argument(
        "--embed-all-initial-prompts",
        action="store_true",
        help="Diversity calculation: If set, embed the concatenation of all initial (system & user) prompts before the first assistant call instead of only the last user prompt.",
    )
    parser.add_argument(
        "--skip-measurement",
        action="store_true",
        help="Skip diversity and IRT measurement calculations to speed up processing",
    )

    args = parser.parse_args()

    output_dir = "pipeline_results"
    os.makedirs(output_dir, exist_ok=True)

    # Generate file prefix based on target_benchmark
    if args.target_benchmark:
        benchmark_prefix = "_".join(args.target_benchmark)
    else:
        benchmark_prefix = "all"

    # Generate filenames with unified naming convention
    log_filename = f"{output_dir}/{benchmark_prefix}_{timestamp}_pipeline.log"
    csv_filename = f"{output_dir}/{benchmark_prefix}_{timestamp}_filtering_summary.csv"
    report_filename = f"{output_dir}/{benchmark_prefix}_{timestamp}_report.csv"

    # Configuration
    config = {
        "llm_model": args.llm_model,
        "llm_max_samples": args.llm_max_samples,
        "num_proc": args.num_proc,
        "target_benchmark": args.target_benchmark,
        "llm_filter_mode": args.llm_filter_mode,
        "skip_scoring": args.skip_scoring,
        "embedding_model": args.embedding_model,
        "embedding_batch_size": args.embedding_batch_size,
        "embed_all_initial_prompts": args.embed_all_initial_prompts,
        "skip_measurement": args.skip_measurement,
        "csv_filename": csv_filename,
        "report_filename": report_filename,
    }

    # Set up logging with dynamic filename
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
        force=True  # This allows reconfiguring if already configured
    )

    # Validate arguments
    if args.skip_rule_based and args.skip_llm_judge:
        logger.error("Cannot skip both rule-based and LLM judge filtering")
        sys.exit(1)

    # Run pipeline
    pipeline = BenchmarkFilteringPipeline(config)
    pipeline.run_pipeline(
        skip_llm_judge=args.skip_llm_judge, skip_rule_based=args.skip_rule_based
    )


if __name__ == "__main__":
    main()
