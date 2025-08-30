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
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.comprehensive_rule_filtering import ComprehensiveRuleFilter
from src.llm_judge_filtering import LLMJudgeAssessor, LLMJudgeConfig, Step
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
        
    def run_pipeline(self, skip_llm_judge: bool = False, skip_rule_based: bool = False) -> Tuple[List[Dict], List[Dict]]:
        """Run the complete filtering pipeline."""
        # TODO: Not tested yet
        logger.info("=" * 60)
        logger.info("STARTING BENCHMARK FILTERING PIPELINE")
        logger.info("=" * 60)
        
        if skip_rule_based:
            # Skip rule-based filtering and run only LLM judge on questions
            logger.info("\n" + "=" * 40)
            logger.info("SKIPPING STEP 1: RULE-BASED FILTERING")
            logger.info("RUNNING LLM-AS-JUDGE ON QUESTIONS INDEPENDENTLY")
            logger.info("=" * 40)
            
            return self._run_llm_judge_independently()
        else:  # Step 1: Rule-based filtering
            logger.info("\n" + "=" * 40)
            logger.info("STEP 1: COMPREHENSIVE RULE-BASED FILTERING")
            logger.info("=" * 40)
            step1_passed, step1_dropped = self._run_step1_rule_filtering()
            # Save Step 1 results
            self._save_results(step1_passed, step1_dropped, "step1_rule_based")
        
        if skip_llm_judge:
            logger.info("\n" + "=" * 40)
            logger.info("SKIPPING STEP 2: LLM-AS-JUDGE FILTERING")
            logger.info("=" * 40)
            return step1_passed, step1_dropped
        else:
            # Step 2: LLM-as-Judge filtering
            logger.info("\n" + "=" * 40)
            logger.info("STEP 2: LLM-AS-JUDGE FILTERING")
            logger.info("=" * 40)
            
            step2_passed, step2_dropped = self._run_llm_judge(step1_passed)
            
            # Save Step 2 results
            self._save_results(step2_passed, step2_dropped, "step2_llm_judge")
            
        # Final summary
        self._print_final_summary(step1_passed, step2_passed)
        
        return step2_passed, step1_dropped + step2_dropped
    
    def _run_step1_rule_filtering(self) -> Tuple[List[Dict], List[Dict]]:
        """Run Step 1: Comprehensive rule-based filtering."""
        logger.info("Loading benchmark data...")
        all_samples = self.data_loader.load_benchmark_data("benchmark")
        logger.info(f"Loaded {len(all_samples):,} total samples")
        
        logger.info("Applying comprehensive rule-based filtering...")
        rule_filter = ComprehensiveRuleFilter()
        passed_samples, dropped_samples = rule_filter.filter_samples(all_samples)
        
        logger.info(f"Step 1 completed: {len(passed_samples):,} samples passed")
        return passed_samples, dropped_samples
    
    def _run_llm_judge(self, step1_passed: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Run Step 2: LLM-as-Judge filtering."""
        logger.info(f"Starting LLM-as-Judge filtering on {len(step1_passed):,} samples from Step 1")
        
        # Configure LLM-as-Judge
        llm_config = LLMJudgeConfig(
            model=self.config.get("llm_model", "gpt-4o-mini"),
            max_samples=self.config.get("llm_max_samples", None),  # None = process all
            batch_size=self.config.get("llm_batch_size", 10),
            max_retries=self.config.get("llm_max_retries", 3),
            retry_delay=self.config.get("llm_retry_delay", 1.0),
            num_proc=self.config.get("num_proc", 1)
        )
        
        llm_filter = LLMJudgeAssessor(llm_config)
        passed_samples, dropped_samples = llm_filter.filter_samples(step1_passed)
        
        logger.info(f"Step 2 completed: {len(passed_samples):,} samples passed")
        return passed_samples, dropped_samples
    
    def _save_results(self, passed_samples: List[Dict], dropped_samples: List[Dict], step_name: str):
        """Save results for a pipeline step."""
        logger.info(f"Saving {step_name} results...")
        
        def make_json_serializable(obj):
            """Make objects JSON serializable by converting enums and other non-serializable types."""
            if isinstance(obj, dict):
                return {key: make_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, 'value'):  # Handle enums like BenchmarkType
                return obj.value
            else:
                return obj
        
        # Save passed samples
        passed_filename = f"{step_name}_passed_samples.jsonl"
        with open(passed_filename, "w") as f:
            for sample in passed_samples:
                serializable_sample = make_json_serializable(sample)
                f.write(json.dumps(serializable_sample) + "\n")
        
        # Save dropped samples
        dropped_filename = f"{step_name}_dropped_samples.jsonl"
        with open(dropped_filename, "w") as f:
            for sample in dropped_samples:
                serializable_sample = make_json_serializable(sample)
                f.write(json.dumps(serializable_sample) + "\n")
        
        logger.info(f"Results saved to {passed_filename} and {dropped_filename}")
    
    def _print_final_summary(self, step1_passed: List[Dict], step2_passed: List[Dict]):
        """Print final pipeline summary."""
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE COMPLETED - FINAL SUMMARY")
        logger.info("=" * 60)
        
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
        import hashlib
        question_ids = set()
        
        for sample in samples:
            messages = sample.get("messages", [])
            user_prompt = ""
            
            for msg in messages:
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if content:
                        if isinstance(content, list):
                            text_parts = []
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    text_parts.append(part.get("text", ""))
                                elif isinstance(part, str):
                                    text_parts.append(part)
                            content = " ".join(text_parts)
                        
                        if isinstance(content, str):
                            user_prompt = content
                            break
            
            task_name = sample.get("task_name", "unknown")
            benchmark_name = sample.get("benchmark_name", "unknown")
            question_text = f"{benchmark_name}|||{task_name}|||{user_prompt}"
            question_id = hashlib.md5(question_text.encode()).hexdigest()
            question_ids.add(question_id)
        
        return len(question_ids)
    
    def _run_llm_judge_independently(self) -> Tuple[List[Dict], List[Dict]]:
        """Run LLM judge independently on questions from benchmark datasets."""
        # Determine which benchmarks to process based on target_benchmark config
        target_benchmark = self.config.get("target_benchmark")
        
        if target_benchmark:
            # Map string names to Benchmark enum values
            benchmark_map = {
                "tau_bench": Benchmark.TAU_BENCH,
                "complex_func_bench": Benchmark.COMPLEX_FUNC_BENCH
            }
            
            if target_benchmark in benchmark_map:
                benchmarks = [benchmark_map[target_benchmark]]
                logger.info(f"Processing {target_benchmark}")
            else:
                logger.warning(f"Unknown target benchmark: {target_benchmark}. Processing all available benchmarks.")
                benchmarks = [Benchmark.TAU_BENCH, Benchmark.COMPLEX_FUNC_BENCH]
        else:
            benchmarks = [Benchmark.TAU_BENCH, Benchmark.COMPLEX_FUNC_BENCH]
            logger.info("Processing all available benchmarks")

        llm_config = LLMJudgeConfig(
            model=self.config.get("llm_model", "gpt-4o-mini"),
            max_samples=self.config.get("llm_max_samples", None),
            batch_size=self.config.get("llm_batch_size", 10),
            max_retries=self.config.get("llm_max_retries", 3),
            retry_delay=self.config.get("llm_retry_delay", 1.0),
            num_proc=self.config.get("num_proc", 1)
        )

        all_results = []
        for benchmark in benchmarks:
            logger.info(f"Processing {benchmark.value} benchmark...")
            assessor = LLMJudgeAssessor(benchmark, llm_config)
            benchmark_results = assessor.load_benchmark_and_get_results()
            all_results.extend(benchmark_results)
            
        # Save combined results as JSON
        output_filename = "llm_judge_results.json"
        with open(output_filename, "w", encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Results saved to {output_filename}")
        logger.info(f"Total questions processed: {len(all_results)}")
        
        # For compatibility, return empty lists since we're saving to JSON
        return [], []
    
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
        "--llm-model", 
        default="openai/gpt-4.1",
        help="LLM model to use for Step 2 (default: gpt-4o-mini)"
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
        "--target_benchmark",
        choices=["tau_bench", "complex_func_bench", "bfcl", "nexus_bench", "drafter_bench"],
        help="Target benchmark to process (default: all available benchmarks)"
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
        "target_benchmark": args.target_benchmark
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
