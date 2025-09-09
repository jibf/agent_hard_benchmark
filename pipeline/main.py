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
from math import comb
from typing import Dict, List, Tuple, Optional
from collections import Counter
from datetime import datetime
from sentence_transformers import SentenceTransformer
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.comprehensive_rule_filtering import ComprehensiveRuleFilter
from src.rule_filtering_orchestrator import RuleFilteringOrchestrator
from src.llm_judge_filtering import LLMJudge, LLMJudgeConfig, LLMJudgeStep
from src.data_loader import BenchmarkDataLoader
from src.utils.types import Benchmark, UniqueQuestionID, LLMJudgeOutput, PipelineOutput, RuleBasedOutput
from src.utils import group_responses_by_question, get_benchmark_from_name

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

        # ----- Embedding model for semantic diversity -----
        model_name = self.config.get("embedding_model", "Qwen/Qwen3-Embedding-8B")
        self.embedder = SentenceTransformer(
            model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_kwargs={"torch_dtype": torch.float16},
        )
        self.embedding_batch_size = self.config.get("embedding_batch_size", 8)
    
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
    
    def _convert_response_list_to_qid_list(self, response_list: List[Dict]) -> List[UniqueQuestionID]:
        """Convert response list to unique question ID list, removing duplicates."""
        questions = set()
        for response in response_list:
            unique_question_id = UniqueQuestionID(
                benchmark=get_benchmark_from_name(response['benchmark_name']),
                task_name=response.get('task_name', None),
                question_id=response['meta']['id']
            )
            questions.add(unique_question_id)
        
        return list(questions)
        
        
    def run_pipeline(self, skip_llm_judge: bool = False, skip_rule_based: bool = False) -> Dict[UniqueQuestionID, PipelineOutput]:
        """Run the complete filtering pipeline."""
        logger.info("Starting benchmark filtering pipeline")
        
        all_responses = self._load_benchmark_data()
        responses_by_question = group_responses_by_question(all_responses)
        pipeline_outputs = {k: PipelineOutput() for k in responses_by_question.keys()}

        separability_dict = self._compute_separability(all_responses)
        logger.info(f"Benchmark separability before filtering: {json.dumps(separability_dict, indent=2)}")

        diversity_dict = self._compute_diversity(all_responses)
        logger.info(f"Benchmark semantic diversity before filtering: {json.dumps(diversity_dict, indent=2)}")

        # Step 1: Rule-based filtering
        current_responses = all_responses
        if not skip_rule_based:
            logger.info("Step 1: Comprehensive rule-based filtering")
            step1_passed, step1_dropped = self._run_step1_rule_filtering(all_responses)
            current_responses = step1_passed
            
            # Update pipeline_outputs with step1 results
            step1_passed_questions = set(group_responses_by_question(step1_passed).keys())
            for question_id in pipeline_outputs.keys():
                passed = question_id in step1_passed_questions
                pipeline_outputs[question_id].rule_based_output = RuleBasedOutput(passed=passed, reason=None)
            
            separability_dict = self._compute_separability(step1_passed)
            logger.info(f"Benchmark separability after Step 1: {json.dumps(separability_dict, indent=2)}")

            diversity_dict = self._compute_diversity(step1_passed)
            logger.info(f"Benchmark semantic diversity after Step 1: {json.dumps(diversity_dict, indent=2)}")
        else:
            logger.info("Skipping Step 1: Rule-based filtering")

        # Step 2: LLM-as-Judge filtering  
        if not skip_llm_judge:
            logger.info("Step 2: LLM-as-Judge filtering")
            current_questions = list(group_responses_by_question(current_responses).keys())
            step2_result = self._run_llm_judge(current_questions)
            
            # Update pipeline_outputs with step2 results
            for question_id, llm_output in step2_result.items():
                pipeline_outputs[question_id].llm_judge_output = llm_output
            
            step2_passed_responses = [response for qid in step2_result.keys() for response in responses_by_question.get(qid, [])]
            separability_dict = self._compute_separability(step2_passed_responses)
            logger.info(f"Benchmark separability after Step 2: {json.dumps(separability_dict, indent=2)}")

            diversity_dict = self._compute_diversity(step2_passed_responses)
            logger.info(f"Benchmark semantic diversity after Step 2: {json.dumps(diversity_dict, indent=2)}")
        else:
            logger.info("Skipping Step 2: LLM-as-Judge filtering")
        
        self._save_results(pipeline_outputs)
        self._print_final_summary(pipeline_outputs)
        return pipeline_outputs 

    
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
    

    def _compute_separability(self, samples: List[Dict], n_bootstrap: int=100, ci: float=0.95) -> float:

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

    def _bootstrap_confidence_interval(self, scores: np.ndarray, n_bootstrap: int=100, ci: float=0.95):
        n = len(scores)
        means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=n, replace=True)
            means.append(np.mean(sample))
        lower = np.percentile(means, (1 - ci) / 2 * 100)
        upper = np.percentile(means, (1 + ci) / 2 * 100)
        return lower, upper

    # Semantic diversity metric
    def _compute_diversity(self, samples: List[Dict]) -> Dict[str, float]:
        """Compute semantic diversity for each benchmark using average pairwise cosine distance.

        For each benchmark, we embed the `messages` field of every sample and then
        compute the average cosine distance across all unique pairs. The metric
        is bounded in [0, 1] per the expression: (2 / (N * (N - 1))) * sum{i<j} [1 - cos(e_i, e_j)]
        where N is the number of samples and cos(·,·) is cosine similarity.
        """
        # Collect one non-empty user prompt per unique meta.id for each benchmark
        # Structure: {benchmark: {meta_id: text}}
        id_to_text_by_benchmark: Dict[str, Dict[str, str]] = {}

        for sample in samples:
            benchmark_name = sample["benchmark_name"]

            # Unique question id (mandatory)
            meta_id = str(sample["meta"]["id"])
            if not meta_id:
                # Skip if meta.id missing
                continue

            # Already captured non-empty text for this id → skip
            if meta_id in id_to_text_by_benchmark.get(benchmark_name, {}):
                continue

            messages_field = sample["messages"]
            if not messages_field:
                continue

            # Extract latest 'user' message before first non-user after system messages
            msg_content = ""
            for m in messages_field:
                if m["role"] == "system":
                    # Skip system messages at the start
                    continue
                if m["role"] == "user":
                    msg_content = m["content"]  # update latest user candidate
                    continue
                # First non-user encountered (assistant/tool etc.)
                break

            if not msg_content:
                continue

            id_to_text_by_benchmark.setdefault(benchmark_name, {})[meta_id] = msg_content

        print(id_to_text_by_benchmark)
        diversity_dict: Dict[str, float] = {}
        for benchmark_name, id_to_text in id_to_text_by_benchmark.items():
            texts = list(id_to_text.values())
            N = len(texts)
            if N < 2:
                diversity_dict[benchmark_name] = 0.0
                continue

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
        
    def _save_results(self, pipeline_outputs: Dict[UniqueQuestionID, PipelineOutput]):
        """Save results for the pipeline."""
        logger.info("Saving pipeline results...")
        
        # Save pipeline outputs
        timestamp = datetime.now().strftime("%m%d_%H%M")
        results_filename = f"pipeline_results_{timestamp}.jsonl"
        with open(results_filename, "w") as f:
            for question_id, output in pipeline_outputs.items():
                result_dict = {
                    **question_id.model_dump(),
                    **output.model_dump()
                }
                serializable_result = self._make_json_serializable(result_dict)
                f.write(json.dumps(serializable_result) + "\n")
        
        logger.info(f"Pipeline results saved to {results_filename}")
    
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
    
    def _print_final_summary(self, pipeline_outputs: Dict[UniqueQuestionID, PipelineOutput]):
        logger.info("Pipeline completed - Final summary")

        step1_passed = [qid for qid, output in pipeline_outputs.items() 
                       if not output.rule_based_output or output.rule_based_output.passed]
        
        step2_passed = [qid for qid in step1_passed 
                       if (not pipeline_outputs[qid].llm_judge_output or 
                           not pipeline_outputs[qid].llm_judge_output.is_flawed)]
        
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
        
        logger.info(f"\nPipeline complete!")
    
    def _count_unique_questions(self, samples: List[Dict]) -> int:
        """Count unique questions in samples."""
        return len(self._convert_response_list_to_qid_list(samples))
    
    def _filter_responses_by_benchmark(self, responses: List[Dict], benchmark_name: str) -> List[Dict]:
        if benchmark_name not in list(benchmark.value for benchmark in Benchmark):
            raise ValueError(f"Invalid benchmark name {benchmark_name}")
        
        def normalize_name(name: str) -> str:
            return name.lower().replace('-', '').replace('_', '')
        
        result = []
        for response in responses:
            if normalize_name(response['benchmark_name']) == normalize_name(benchmark_name):
                result.append(response)
        return result
    
    def _filter_questions_by_benchmark(self, questions: List[UniqueQuestionID], benchmark: Benchmark) -> List[UniqueQuestionID]:
        result = []
        for question in questions:
            if question.benchmark == benchmark:
                result.append(question)
        return result



    def _run_llm_judge(self, questions: List[UniqueQuestionID]) -> Dict[UniqueQuestionID, LLMJudgeOutput]:
        """Run LLM judge independently on questions from benchmark datasets."""
        # Determine which benchmarks to process based on target_benchmark config
        results = dict()

        for benchmark in Benchmark:
            benchmark_responses = self._filter_questions_by_benchmark(questions, benchmark)
            if len(benchmark_responses) == 0:
                continue
            logger.info(f"Processing {benchmark.value} benchmark: {len(benchmark_responses)} responses")
            judge = LLMJudge(benchmark, self.llm_config)
            benchmark_results = judge.get_results()
            results.update(benchmark_results)
            
        return results 
    
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
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-8B",
        help="SentenceTransformer model for semantic diversity computation (default: Qwen/Qwen3-Embedding-8B)"
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=8,
        help="Batch size when encoding texts for diversity (default: 8)"
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
        "embedding_model": args.embedding_model,
        "embedding_batch_size": args.embedding_batch_size,
    }
    
    # Validate arguments
    if args.skip_rule_based and args.skip_llm_judge:
        logger.error("Cannot skip both rule-based and LLM judge filtering")
        sys.exit(1)
    
    # Run pipeline
    pipeline = BenchmarkFilteringPipeline(config)
    pipeline.run_pipeline(
        skip_llm_judge=args.skip_llm_judge,
        skip_rule_based=args.skip_rule_based
    )
    
if __name__ == "__main__":
    main()
