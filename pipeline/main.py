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

from comprehensive_rule_filtering import ComprehensiveRuleFilter
from llm_judge_filtering import LLMJudgeFilter, LLMJudgeConfig, Step
from data_loader import BenchmarkDataLoader
from benchmark_types import BenchmarkType
from utils.formatters.tau_formatter import TauBenchFormatter

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
            
            return self._run_llm_judge_independent()
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
            
            step2_passed, step2_dropped = self._run_step2_llm_judge(step1_passed)
            
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
    
    def _run_step2_llm_judge(self, step1_passed: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Run Step 2: LLM-as-Judge filtering."""
        logger.info(f"Starting LLM-as-Judge filtering on {len(step1_passed):,} samples from Step 1")
        
        # Configure LLM-as-Judge
        llm_config = LLMJudgeConfig(
            model=self.config.get("llm_model", "gpt-4o-mini"),
            max_samples=self.config.get("llm_max_samples", None),  # None = process all
            batch_size=self.config.get("llm_batch_size", 10),
            max_retries=self.config.get("llm_max_retries", 3),
            retry_delay=self.config.get("llm_retry_delay", 1.0)
        )
        
        logger.info(f"LLM-as-Judge configuration:")
        logger.info(f"  Model: {llm_config.model}")
        logger.info(f"  Max samples: {llm_config.max_samples or 'All'}")
        logger.info(f"  Batch size: {llm_config.batch_size}")
        
        llm_filter = LLMJudgeFilter(llm_config)
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
    
    def _run_llm_judge_independent(self) -> Tuple[List[Dict], List[Dict]]:
        """Run LLM judge independently on questions from benchmark datasets."""
        logger.info("Loading questions from benchmark datasets...")
        
        # For now, we'll process ComplexFuncBench questions
        # This can be extended to support other benchmark types
        questions = self._load_benchmark_questions()
        
        if not questions:
            logger.warning("No questions found to process")
            return [], []
        
        logger.info(f"Loaded {len(questions):,} questions")
        
        # Configure LLM-as-Judge
        llm_config = LLMJudgeConfig(
            model=self.config.get("llm_model", "gpt-4o-mini"),
            max_samples=self.config.get("llm_max_samples", None),
            batch_size=self.config.get("llm_batch_size", 10),
            max_retries=self.config.get("llm_max_retries", 3),
            retry_delay=self.config.get("llm_retry_delay", 1.0)
        )
        
        logger.info(f"LLM-as-Judge configuration:")
        logger.info(f"  Model: {llm_config.model}")
        logger.info(f"  Max samples: {llm_config.max_samples or 'All'}")
        logger.info(f"  Batch size: {llm_config.batch_size}")
        
        llm_filter = LLMJudgeFilter(llm_config)
        
        # Process questions independently
        step = Step.FILTER  # Default to filter step
        
        # Determine benchmark type based on target
        target_benchmark = self.config.get("target_benchmark")
        if target_benchmark == "tau_bench":
            benchmark_type = BenchmarkType.TAU_BENCH
        else:
            benchmark_type = BenchmarkType.COMPLEX_FUNC_BENCH  # Default benchmark type
            
        proc_num = self.config.get("proc_num", 1)
        
        results = llm_filter.assess_questions(questions, step, benchmark_type, proc_num)
        
        # Separate passed and dropped questions based on assessment
        passed_questions = []
        dropped_questions = []
        
        for result in results:
            assessment = result.get("assessment", {})
            if not assessment.get("error"):
                if step == Step.FILTER:
                    # For filter step, check if_flawed
                    if not assessment.get("is_flawed", True):
                        passed_questions.append(result)
                    else:
                        dropped_questions.append(result)
                else:
                    # For score step, could add different logic
                    passed_questions.append(result)
            else:
                dropped_questions.append(result)
        
        # Save results
        self._save_results(passed_questions, dropped_questions, "llm_judge_independent")
        
        logger.info(f"Independent LLM judge completed:")
        logger.info(f"  Passed: {len(passed_questions):,}")
        logger.info(f"  Dropped: {len(dropped_questions):,}")
        
        return passed_questions, dropped_questions
    
    def _load_benchmark_questions(self) -> List[Dict]:
        """Load questions from benchmark datasets for independent assessment."""
        target_benchmark = self.config.get("target_benchmark")
        questions = []
        
        # Map benchmark names to their directories and file patterns
        benchmark_mapping = {
            "tau_bench": "data",  # tau_bench files are stored in data/ directory
            "complex_func_bench": "data",
            "bfcl": "benchmark/BFCL-evaluation",
            "nexus_bench": "benchmark/NexusBench-evaluation", 
            "drafter_bench": "benchmark/DrafterBench-evaluation"
        }
        
        # Map benchmark names to their file patterns
        file_patterns = {
            "tau_bench": ["tau_bench_*.jsonl"],  # Look for converted tau_bench files
            "complex_func_bench": ["ComplexFuncBench.jsonl"],
            "bfcl": ["*.jsonl"],
            "nexus_bench": ["*.jsonl"],
            "drafter_bench": ["*.jsonl"]
        }
        
        if target_benchmark:
            # Load specific benchmark
            benchmarks_to_load = [target_benchmark]
        else:
            # Load all available benchmarks
            benchmarks_to_load = list(benchmark_mapping.keys())
        
        for benchmark_name in benchmarks_to_load:
            benchmark_dir = benchmark_mapping.get(benchmark_name)
            patterns = file_patterns.get(benchmark_name, ["*.jsonl"])
            
            if not benchmark_dir or not os.path.exists(benchmark_dir):
                logger.warning(f"Benchmark directory {benchmark_dir} for {benchmark_name} not found")
                continue
                
            # Special handling for tau_bench
            if benchmark_name == "tau_bench":
                questions.extend(self._load_tau_bench_questions())
                continue
            
            # Load files matching patterns
            files_loaded = 0
            for pattern in patterns:
                import glob
                file_pattern = os.path.join(benchmark_dir, pattern)
                matching_files = glob.glob(file_pattern)
                
                for file_path in matching_files:
                    logger.info(f"Loading questions from {file_path}")
                    file_questions = self._load_questions_from_file(file_path, benchmark_name)
                    questions.extend(file_questions)
                    files_loaded += 1
                    
                    # Limit files per benchmark to avoid overwhelming
                    if files_loaded >= 5:  # Limit to first 5 files per benchmark
                        break
                        
                if files_loaded >= 5:
                    break
            
            if files_loaded == 0:
                logger.warning(f"No files found for benchmark {benchmark_name} in {benchmark_dir}")
        
        logger.info(f"Total questions loaded: {len(questions)}")
        return questions
    
    def _load_tau_bench_questions(self) -> List[Dict]:
        """Load tau_bench questions using the original assess_dataset_mine logic."""
        questions = []
        
        # Use TauBenchFormatter to convert tau-bench tasks
        try:
            from assess_dataset_mine import DatasetAssessor
            from benchmark_types import BenchmarkType
            
            # Create assessor to use tau formatter
            assessor = DatasetAssessor()
            
            # Try airline domain first
            domains = ['airline', 'retail']
            for domain in domains:
                try:
                    logger.info(f"Generating tool schemas for tau-bench {domain}...")
                    assessor.tau_formatter.get_tool_schemas(domain)
                    
                    logger.info(f"Converting tau-bench {domain} tasks...")
                    converted_tasks = assessor.tau_formatter.process_tau_bench_tasks(domain)
                    
                    if converted_tasks:
                        for i, task in enumerate(converted_tasks):
                            task_dict = task.model_dump() if hasattr(task, 'model_dump') else task
                            task_dict['id'] = f'tau_{domain}_{i+1}'
                            task_dict['benchmark_name'] = 'tau_bench'
                            task_dict['domain'] = domain
                            questions.append(task_dict)
                        
                        logger.info(f"Loaded {len(converted_tasks)} questions from tau-bench {domain}")
                    else:
                        logger.warning(f"No converted tasks found for tau-bench {domain}")
                        
                except Exception as e:
                    logger.warning(f"Failed to load tau-bench {domain}: {e}")
                    continue
                    
        except ImportError as e:
            logger.warning(f"Could not import tau-bench dependencies: {e}")
            
        return questions
    
    def _load_questions_from_file(self, file_path: str, benchmark_name: str) -> List[Dict]:
        """Load questions from a single JSONL file."""
        questions = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        question_data = json.loads(line.strip())
                        question_data['id'] = question_data.get('id', f'{benchmark_name}_{os.path.basename(file_path)}_{line_num}')
                        question_data['benchmark_name'] = benchmark_name
                        question_data['source_file'] = file_path
                        questions.append(question_data)
                        
                        # Limit questions per file to avoid overwhelming
                        if len(questions) >= 100:  # Limit to first 100 questions per file
                            logger.info(f"Limited to first 100 questions from {file_path}")
                            break
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Error parsing line {line_num} in {file_path}: {e}")
                        
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {e}")
            
        logger.info(f"Loaded {len(questions)} questions from {file_path}")
        return questions

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
        "--proc-num", 
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
        "proc_num": args.proc_num,
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
