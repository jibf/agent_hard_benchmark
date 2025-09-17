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
from src.utils import group_responses_by_question, log_confusion_matrix

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler()],
)
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
        self.metrics_summary = {
            "original": {"separability": None, "diversity": None, "agreement": None},
            "step1": {"separability": None, "diversity": None, "agreement": None},
            "step1_baseline": {"separability": None},
            "step2": {"separability": None, "diversity": None, "agreement": None},
            "step2_baseline": {"separability": None},
            "step3": {"separability": None, "diversity": None, "agreement": None},
            "step3_baseline": {"separability": None},
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
        if self.config.get("skip_diversity_measurement", False):
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
            "is_issue": None,
            "issue_type": None,
            "comp_passed": None,
            "specific_rule_passed": None,
            "specific_llm_passed": None,
            "topk_selection_passed": None,
        }
        self.filering_summary = {}

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

    def run_pipeline(
        self, skip_llm_judge: bool = False, skip_rule_based: bool = False
    ) -> Dict[UniqueQuestionID, PipelineOutput]:
        """Run the complete filtering pipeline."""
        logger.info("Starting benchmark filtering pipeline")

        all_responses = self._load_benchmark_data()
        problematic_issues = self.data_loader.load_problematic_issues()
        responses_by_question = group_responses_by_question(all_responses)
        responses_by_question = self.filter_illegal_data(responses_by_question)

        pipeline_outputs = {k: PipelineOutput() for k in responses_by_question.keys()}

        sep_list = []
        for i in range(3):
            ori_separability = self._compute_separability(responses_by_question)
            logger.info(
                f"Benchmark separability before filtering: {json.dumps(ori_separability, indent=2)}"
            )
            sep_list.append(ori_separability)

        # Store for final summary
        self.metrics_summary["original"]["separability"] = sep_list

        if not self.config.get("skip_diversity_measurement", False):
            diversity_dict = self._compute_diversity(responses_by_question)
            logger.info(
                f"Benchmark semantic diversity before filtering: {json.dumps(diversity_dict, indent=2)}"
            )
            # Store for final summary
            self.metrics_summary["original"]["diversity"] = diversity_dict

        # Calculate model ranking from original dataset for consistent ordering
        model_ranking = None
        if not self.config.get("skip_visualization", False):
            model_ranking = self._calculate_model_ranking(responses_by_question)
            self._visualize_model_performance(
                responses_by_question,
                "Original Dataset",
                "original_performance",
                model_ranking=model_ranking,
            )
            if not self.config.get("skip_diversity_measurement", False):
                self._visualize_diversity(
                    responses_by_question, "Original Dataset", "original_diversity"
                )
            # Store agreement stats for original dataset
            if hasattr(self, "_current_agreement_stats"):
                self.metrics_summary["original"]["agreement"] = (
                    self._current_agreement_stats.copy()
                )
                self._current_agreement_stats = {}

        # Step 0: Always apply comprehensive filtering first
        current_responses = responses_by_question
        remaining_problematic_issues = deepcopy(problematic_issues)
        self._write_filter_summary(
            passed_ids=set(responses_by_question.keys()),
            input_problematic_ids=set(remaining_problematic_issues.keys()),
            phase="initial",
            problematic_issues=problematic_issues,
        )
        if not skip_rule_based:
            logger.info("Step 0: Comprehensive rule-based filtering (always applied)")
            step0_passed, step0_dropped = self._run_comprehensive_filtering(
                responses_by_question
            )
            log_confusion_matrix(
                problematic_issues=remaining_problematic_issues,
                passed_ids=set(step0_passed.keys()),
                total_num=len(current_responses),
            )
            self._write_filter_summary(
                passed_ids=set(step0_passed.keys()),
                input_problematic_ids=set(remaining_problematic_issues.keys()),
                phase="step0",
                problematic_issues=problematic_issues,
            )
            remaining_problematic_issues = {
                question_id: problematic_issues
                for question_id in remaining_problematic_issues
                if question_id in step0_passed
            }
            # Step 1: Apply benchmark-specific filtering for benchmarks that have specific filters
            step1_passed, step1_dropped = self._run_benchmark_specific_filtering(
                step0_passed
            )
            log_confusion_matrix(
                problematic_issues=remaining_problematic_issues,
                passed_ids=set(step1_passed.keys()),
                total_num=len(step0_passed),
            )
            self._write_filter_summary(
                passed_ids=set(step1_passed.keys()),
                input_problematic_ids=set(remaining_problematic_issues.keys()),
                phase="step1",
                problematic_issues=problematic_issues,
            )
            remaining_problematic_issues = {
                question_id: problematic_issues
                for question_id in remaining_problematic_issues
                if question_id in step1_passed
            }
            current_responses = step1_passed

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

            sep_list = []
            for i in range(3):
                separability_dict = self._compute_separability(step1_passed)
                logger.info(
                    f"Benchmark separability after Step 1: {json.dumps(separability_dict, indent=2)}"
                )
                sep_list.append(separability_dict)
            # Store for final summary
            self.metrics_summary["step1"]["separability"] = sep_list

            # Visualize Step 1 filtered performance
            if not self.config.get("skip_visualization", False):
                self._visualize_model_performance(
                    step1_passed,
                    "After Step 1 (Rule-based Filtering)",
                    "step1_filtered_performance",
                    model_ranking=model_ranking,
                )
                if not self.config.get("skip_diversity_measurement", False):
                    self._visualize_diversity(
                        step1_passed,
                        "After Step 1 (Rule-based Filtering)",
                        "step1_filtered_diversity",
                    )
                if hasattr(self, "_current_agreement_stats"):
                    self.metrics_summary["step1"]["agreement"] = (
                        self._current_agreement_stats.copy()
                    )
                    self._current_agreement_stats = {}

            # Create baseline sample set by randomly sampling N tasks from original samples
            logger.info(f"Step 1 BASELINE...")
            baseline_responses = self._create_task_wise_baseline_sample_set(
                responses_by_question, step1_unique_tasks
            )
            baseline_sample_count = sum(
                len(responses) for responses in baseline_responses.values()
            )
            logger.info(
                f"Created task-wise baseline sample set with {baseline_sample_count} samples from {step1_unique_tasks} tasks"
            )

            # Compute separability for baseline for comparison
            sep_list = []
            for i in range(3):
                baseline_separability = self._compute_separability(baseline_responses)
                logger.info(
                    f"Benchmark separability baseline (vs. step1): {json.dumps(baseline_separability, indent=2)}"
                )
                sep_list.append(baseline_separability)
            self.metrics_summary["step1_baseline"]["separability"] = sep_list

            # Visualize baseline performance
            if not self.config.get("skip_visualization", False):
                self._visualize_model_performance(
                    baseline_responses,
                    "Baseline (Random Sampling)",
                    "baseline_performance",
                    model_ranking=model_ranking,
                )
                if not self.config.get("skip_diversity_measurement", False):
                    self._visualize_diversity(
                        baseline_responses,
                        "Baseline (Random Sampling)",
                        "baseline_diversity",
                    )

            if not self.config.get("skip_diversity_measurement", False):
                diversity_dict = self._compute_diversity(step1_passed)
                logger.info(
                    f"Benchmark semantic diversity after Step 1: {json.dumps(diversity_dict, indent=2)}"
                )
                # Store for final summary
                self.metrics_summary["step1"]["diversity"] = diversity_dict
        else:
            logger.info("Skipping Step 0 & 1: Rule-based filtering")

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
            log_confusion_matrix(
                problematic_issues=remaining_problematic_issues,
                passed_ids=set(step2_passed.keys()),
                total_num=len(current_responses),
            )
            self._write_filter_summary(
                passed_ids=set(step2_passed.keys()),
                input_problematic_ids=set(remaining_problematic_issues.keys()),
                phase="step2",
                problematic_issues=problematic_issues,
            )
            remaining_problematic_issues = {
                question_id: problematic_issues
                for question_id in remaining_problematic_issues
                if question_id in step2_passed
            }

            # Count unique tasks and samples in step2_passed
            step2_unique_tasks = len(step2_passed)
            step2_sample_count = sum(
                len(responses) for responses in step2_passed.values()
            )
            logger.info(
                f"Step 2 passed: {step2_sample_count} samples from {step2_unique_tasks} unique tasks"
            )

            sep_list = []
            for i in range(3):
                separability_dict = self._compute_separability(step2_passed)
                logger.info(
                    f"Benchmark separability after Step 2: {json.dumps(separability_dict, indent=2)}"
                )
                sep_list.append(separability_dict)

            # Store for final summary
            self.metrics_summary["step2"]["separability"] = sep_list
            # Visualize Step 2 filtered performance
            if not self.config.get("skip_visualization", False):
                self._visualize_model_performance(
                    step2_passed,
                    "After Step 2 (LLM-as-Judge Filtering)",
                    "step2_filtered_performance",
                    model_ranking=model_ranking,
                )
                if not self.config.get("skip_diversity_measurement", False):
                    self._visualize_diversity(
                        step2_passed,
                        "After Step 2 (LLM-as-Judge Filtering)",
                        "step2_filtered_diversity",
                    )

            # Store agreement stats for step2
            if hasattr(self, "_current_agreement_stats"):
                self.metrics_summary["step2"]["agreement"] = (
                    self._current_agreement_stats.copy()
                )
                self._current_agreement_stats = {}

            # Create baseline sample set by randomly sampling N tasks from step1_passed
            # logger.info(f"Step 2 BASELINE (vs. Step 1 from step1_passed)...")
            # baseline_from_step1 = self._create_task_wise_baseline_sample_set(
            #     step1_passed, step2_unique_tasks
            # )
            # baseline_separability = self._compute_separability(baseline_from_step1)

            # logger.info(
            #     f"Benchmark separability baseline (vs. Step 2 from step1_passed): {json.dumps(baseline_separability, indent=2)}"
            # )

            # Create baseline sample set by randomly sampling N tasks from all_samples
            logger.info(f"Step 2 BASELINE (vs. Step 1 from all_samples)...")
            baseline_from_all = self._create_task_wise_baseline_sample_set(
                responses_by_question, step2_unique_tasks
            )
            sep_list = []
            for i in range(3):
                baseline_separability = self._compute_separability(baseline_from_all)
                logger.info(
                    f"Benchmark separability baseline (vs. Step 2 from all_samples): {json.dumps(baseline_separability, indent=2)}"
                )
                sep_list.append(baseline_separability)
            self.metrics_summary["step2_baseline"]["separability"] = sep_list

            if not self.config.get("skip_diversity_measurement", False):
                diversity_dict = self._compute_diversity(step2_passed)
                logger.info(
                    f"Benchmark semantic diversity after Step 2: {json.dumps(diversity_dict, indent=2)}"
                )
                # Store for final summary
                self.metrics_summary["step2"]["diversity"] = diversity_dict

            # Step 3: Top-K selection based on scores
            if LLMJudgeStep.SCORE in self.llm_config.steps:
                logger.info("Step 3: Selecting top 50 samples based on total scores")
                step3_passed = self._run_step3_top_k_selection(
                    step2_result, responses_by_question, 50
                )
                log_confusion_matrix(
                    problematic_issues=remaining_problematic_issues,
                    passed_ids=set(step3_passed.keys()),
                    total_num=len(step2_result),
                )
                self._write_filter_summary(
                    passed_ids=set(step3_passed.keys()),
                    input_problematic_ids=set(remaining_problematic_issues.keys()),
                    phase="step3",
                    problematic_issues=problematic_issues,
                )
                remaining_problematic_issues = {
                    question_id: problematic_issues
                    for question_id in remaining_problematic_issues
                    if question_id in step3_passed
                }

                # Count unique tasks and samples in step3_passed
                step3_unique_tasks = len(step3_passed)
                step3_sample_count = sum(
                    len(responses) for responses in step3_passed.values()
                )
                logger.info(
                    f"Step 3 passed: {step3_sample_count} samples from {step3_unique_tasks} unique tasks"
                )

                # # Create baseline sample sets for step3 comparison
                # logger.info(f"Step 3 BASELINE (vs Step 2 from step2_passed)...")
                # baseline_from_step2 = self._create_task_wise_baseline_sample_set(
                #     step2_passed, step3_unique_tasks
                # )
                # baseline_separability_step2 = self._compute_separability(
                #     baseline_from_step2
                # )
                # logger.info(
                #     f"Benchmark separability for baseline (vs Step 3 from step2_passed): {json.dumps(baseline_separability_step2, indent=2)}"
                # )

                # Compute separability for step3 results
                sep_list = []
                for i in range(3):
                    separability_dict_step3 = self._compute_separability(step3_passed)
                    logger.info(
                        f"Benchmark separability after Step 3: {json.dumps(separability_dict_step3, indent=2)}"
                    )
                    sep_list.append(separability_dict_step3)
                # Store for final summary
                self.metrics_summary["step3"]["separability"] = sep_list

                # Visualize Step 3 filtered performance
                if not self.config.get("skip_visualization", False):
                    self._visualize_model_performance(
                        step3_passed,
                        "After Step 3 (Top-K Selection)",
                        "step3_filtered_performance",
                        model_ranking=model_ranking,
                    )
                    if not self.config.get("skip_diversity_measurement", False):
                        self._visualize_diversity(
                            step3_passed,
                            "After Step 3 (Top-K Selection)",
                            "step3_filtered_diversity",
                        )

                # Store agreement stats for step3
                if hasattr(self, "_current_agreement_stats"):
                    self.metrics_summary["step3"]["agreement"] = (
                        self._current_agreement_stats.copy()
                    )
                    self._current_agreement_stats = {}
                if not self.config.get("skip_diversity_measurement", False):
                    diversity_dict = self._compute_diversity(step3_passed)
                    logger.info(
                        f"Benchmark semantic diversity after Step 2: {json.dumps(diversity_dict, indent=2)}"
                    )
                    # Store for final summary
                    self.metrics_summary["step3"]["diversity"] = diversity_dict

                logger.info(f"Step 3 BASELINE (vs Step 3 from all_samples)...")
                baseline_from_all_step3 = self._create_task_wise_baseline_sample_set(
                    responses_by_question, step3_unique_tasks
                )
                sep_list = []
                for i in range(3):
                    baseline_separability_all_step3 = self._compute_separability(
                        baseline_from_all_step3
                    )
                    logger.info(
                        f"Benchmark separability for baseline (vs Step 3 from all_samples): {json.dumps(baseline_separability_all_step3, indent=2)}"
                    )
                    sep_list.append(baseline_separability_all_step3)
                self.metrics_summary["step3_baseline"]["separability"] = sep_list

            else:
                logger.info("Skipping Step 3: Scoring not enabled")
        else:
            logger.info("Skipping Step 2: LLM-as-Judge filtering")

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
        """Run Step 0: Comprehensive rule-based filtering (always applied)."""
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
        """Run Step 1: Benchmark-specific filtering (applied after comprehensive filtering)."""
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
        self, responses_by_question: Dict[str, List[Dict]]
    ) -> Dict[str, float]:
        """Compute semantic diversity for each benchmark using average pairwise cosine distance.

        For each benchmark, we embed the `messages` field of every sample and then
        compute the average cosine distance across all unique pairs. The metric
        is bounded in [0, 1] per the expression: (2 / (N * (N - 1))) * sum{i<j} [1 - cos(e_i, e_j)]
        where N is the number of samples and cos(·,·) is cosine similarity.
        """
        id_to_data_by_benchmark = self._extract_texts_for_diversity(
            responses_by_question
        )

        diversity_dict: Dict[str, float] = {}
        for benchmark_name, id_to_data in id_to_data_by_benchmark.items():
            texts = [data["text"] for data in id_to_data.values()]
            N = len(texts)
            if N < 2:
                diversity_dict[benchmark_name] = 0.0
                continue
            print("diversity texts[0]:", texts[0])
            # Encode texts -> unit-normalised embeddings
            embeddings = self.embedder.encode(
                texts,
                batch_size=self.embedding_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # Cosine similarity matrix via dot product because embeddings are normalised
            sim_matrix = np.matmul(embeddings, embeddings.T)

            # Extract upper triangle (i < j)
            triu_indices = np.triu_indices(N, k=1)
            cosine_sims = sim_matrix[triu_indices]
            avg_distance = np.mean(1 - cosine_sims)

            diversity_dict[benchmark_name] = float(avg_distance)

        return diversity_dict

    def _extract_texts_for_diversity(
        self, responses_by_question: Dict[str, List[Dict]]
    ) -> Dict[str, Dict[str, Dict]]:
        """Helper to extract unique texts and task names from samples for diversity computation/visualization."""
        id_to_data_by_benchmark: Dict[str, Dict[str, Dict]] = {}
        for responses in responses_by_question.values():
            for sample in responses:
                benchmark_name = str(sample["benchmark_name"])
                meta_id = str(sample["meta"]["id"])
                if not meta_id or meta_id in id_to_data_by_benchmark.get(
                    benchmark_name, {}
                ):
                    continue

                messages_field = sample.get("messages")
                if not messages_field:
                    continue

                if self.embed_all_initial_prompts:
                    initial_contents = []
                    for m in messages_field:
                        role = m.get("role")
                        if role not in {"system", "user"}:
                            break
                        if m.get("content"):
                            initial_contents.append(m["content"].strip())
                    msg_content = "\n".join(initial_contents).strip()
                else:
                    msg_content = ""
                    for m in messages_field:
                        role = m.get("role")
                        if role == "system":
                            continue
                        if role == "user":
                            msg_content = m["content"]
                            continue
                        break

                if msg_content:
                    task_name = sample.get("task_name", "unknown")
                    id_to_data_by_benchmark.setdefault(benchmark_name, {})[meta_id] = {
                        "text": msg_content,
                        "task_name": task_name,
                    }
        return id_to_data_by_benchmark

    def _visualize_diversity(
        self, responses_by_question: Dict[str, List[Dict]], title: str, filename: str
    ):
        """Create visualizations for semantic diversity: 2D embedding projection and pairwise distance histogram."""
        logger.info(f"Creating diversity visualization: {title}")

        plots_dir = "pipeline_results/plots"
        os.makedirs(plots_dir, exist_ok=True)

        id_to_data_by_benchmark = self._extract_texts_for_diversity(
            responses_by_question
        )
        print(
            "vis diversity lens:",
            len(responses_by_question),
            len(id_to_data_by_benchmark),
        )
        for benchmark_name, id_to_data in id_to_data_by_benchmark.items():
            if len(id_to_data) < 2:
                continue

            texts = [data["text"] for data in id_to_data.values()]
            task_names = [data["task_name"] for data in id_to_data.values()]

            logger.info(
                f"Generating diversity plots for {benchmark_name} with {len(texts)} unique questions."
            )

            embeddings = self.embedder.encode(
                texts,
                batch_size=self.embedding_batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

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
    ) -> Dict[str, List[str]]:
        """Calculate model ranking based on performance for consistent ordering across visualizations."""
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

        # Calculate ranking for each benchmark
        model_ranking = {}
        for benchmark_name, model_scores in benchmark_data.items():
            model_means = {}
            for model_name, scores in model_scores.items():
                model_means[model_name] = np.mean(scores)
            # Sort by performance (descending)
            model_ranking[benchmark_name] = sorted(
                model_means.keys(), key=lambda x: model_means[x], reverse=True
            )

        return model_ranking

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

            # Print summary statistics
            # logger.info(f"\n{benchmark_name} - Model Performance Summary:")
            # logger.info("=" * 60)
            # for model_name, mean, ci_low, ci_high in zip(
            #     model_names, means, ci_lowers, ci_uppers
            # ):
            #     logger.info(
            #         f"{model_name:30} | Mean: {mean:.4f} | CI: [{ci_low:.4f}, {ci_high:.4f}]"
            #     )

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
        self, passed_ids: set, input_problematic_ids: set, phase: str, problematic_issues: Dict = None
    ):
        """Record filtering summary for each phase.

        Args:
            passed_ids: Set of question IDs that passed current phase
            input_problematic_ids: Set of question IDs that input to current phase
            phase: Phase name ("initial", "step0", "step1", "step2", "step3", "final")
            problematic_issues: Dict of problematic issues for getting issue reasons
        """
        logger.info(f"Recording filter summary for phase: {phase}")

        if phase == "initial":
            # Initialize all questions with basic info
            for question_id in passed_ids:
                if question_id not in self.filering_summary:
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

                    # Check if this is a problematic question
                    is_problematic = False
                    issue_reason = None
                    if question_id in problematic_issues:
                        is_problematic = True
                        issue_reason = problematic_issues[question_id]["reason"]

                    entry["is_issue"] = is_problematic
                    entry["issue_type"] = issue_reason

                    self.filering_summary[question_id] = entry

        else:
            # Mark questions that were NOT in passed_ids as failed for this phase
            field_map = {
                "step0": "comp_passed",
                "step1": "specific_rule_passed",
                "step2": "specific_llm_passed",
                "step3": "topk_selection_passed",
            }

            if phase in field_map:
                field_name = field_map[phase]
                for question_id in input_problematic_ids:
                    entry = self.filering_summary[question_id]
                    # If this question is not in passed_ids and hasn't been marked as failed yet
                    if question_id not in passed_ids:
                        entry[field_name] = False
                    else:
                        entry[field_name] = True

        self._save_filter_summary_csv()

    def _save_filter_summary_csv(self):
        """Save filtering summary to CSV file, ordered by filtering stage (latest filtered first)."""
        if not self.filering_summary:
            logger.info("No filtering summary to save")
            return

        # Sort questions by when they were filtered (reverse order - latest filtered first)
        def get_filter_stage(entry):
            # Return stage number where question was filtered (higher = later)
            if entry.get("is_issue") == False:
                return 0
            elif entry.get("topk_selection_passed") == True:
                return 5
            elif entry.get("topk_selection_passed") == False:
                return 4
            elif entry.get("specific_llm_passed") == False:
                return 3
            elif entry.get("specific_rule_passed") == False:
                return 2
            elif entry.get("comp_passed") == False:
                return 1
            else:
                return 5

        # Sort by filter stage (descending) then by question_id for consistency
        sorted_items = sorted(
            self.filering_summary.items(),
            key=lambda x: (get_filter_stage(x[1]), str(x[0])),
            reverse=True,
        )

        # Use fixed filename (will overwrite in same run)
        csv_path = "filtering_summary.csv"

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
            "step1_baseline": "Step 1 Baseline",
            "step2_baseline": "Step 2 Baseline",
            "step3_baseline": "Step 3 Baseline",
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
                    logger.info(f"    Average agreement: {stats['avg_agreement']:.3f}")
                    logger.info(f"    Min agreement: {stats['min_agreement']:.3f}")
                    logger.info(f"    Max agreement: {stats['max_agreement']:.3f}")
                    logger.info(f"    Models: {stats['num_models']}")
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
        default="openai/gpt-4.1",
        help="LLM model to use for Step 2 (default: gpt-4.1)",
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
        default="both",
        help="LLM filtering scheme: 'common' (universal filter only), 'specific' (benchmark-specific filter only), 'both' (universal + specific filters + scoring by default)",
    )
    parser.add_argument(
        "--skip-scoring",
        action="store_true",
        help="Skip scoring step even when using 'both' filtering scheme",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip creating performance visualization plots",
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-8B",
        help="SentenceTransformer model for semantic diversity computation (default: Qwen/Qwen3-Embedding-8B)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=8,
        help="Batch size when encoding texts for diversity (default: 8)",
    )

    parser.add_argument(
        "--embed-all-initial-prompts",
        action="store_true",
        help="Diversity calculation: If set, embed the concatenation of all initial (system & user) prompts before the first assistant call instead of only the last user prompt.",
    )
    parser.add_argument(
        "--skip-diversity-measurement",
        action="store_true",
        help="Skip diversity measurement calculation to speed up processing",
    )

    args = parser.parse_args()

    # Configuration
    config = {
        "llm_model": args.llm_model,
        "llm_max_samples": args.llm_max_samples,
        "num_proc": args.num_proc,
        "target_benchmark": args.target_benchmark,
        "llm_filter_mode": args.llm_filter_mode,
        "skip_scoring": args.skip_scoring,
        "skip_visualization": args.skip_visualization,
        "embedding_model": args.embedding_model,
        "embedding_batch_size": args.embedding_batch_size,
        "embed_all_initial_prompts": args.embed_all_initial_prompts,
        "skip_diversity_measurement": args.skip_diversity_measurement,
    }

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
