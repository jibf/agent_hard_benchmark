#!/usr/bin/env python3
"""
Data loader module for benchmark filtering pipeline.
Handles loading and preprocessing of benchmark data from JSONL files.
"""

import json
import csv
from pathlib import Path
from typing import Dict, List, Optional
import logging
from src.utils import normalize_benchmark_name
from src.utils.types import Benchmark, UniqueQuestionID

logger = logging.getLogger(__name__)


class BenchmarkDataLoader:
    """Loads and preprocesses benchmark data from JSONL files."""

    def __init__(self):
        self.loaded_samples = []

    def load_benchmark_data(
        self, benchmarks_dir: str, target_benchmark: Optional[List[str]] = None
    ) -> List[Dict]:
        """Load benchmark data from the specified directory.

        Args:
            benchmarks_dir: Directory containing benchmark data
            target_benchmark: If specified, only load these specific benchmarks
        """
        logger.info(f"Loading benchmark data from {benchmarks_dir}")

        benchmarks_path = Path(benchmarks_dir)
        if not benchmarks_path.exists():
            raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_dir}")

        all_samples = []

        if target_benchmark:
            # Load only the target benchmarks
            logger.info(f"Loading target benchmarks: {target_benchmark}")
            for benchmark in target_benchmark:
                target_dir = benchmarks_path / f"{benchmark}-evaluation"
                if target_dir.exists() and target_dir.is_dir():
                    logger.info(f"Loading benchmark: {benchmark}")
                    benchmark_samples = self._load_benchmark_directory(
                        target_dir, benchmark
                    )
                    all_samples.extend(benchmark_samples)
                else:
                    logger.warning(
                        f"Target benchmark directory not found: {target_dir}"
                    )
            if not all_samples:
                return []
        else:
            # Process each benchmark directory (original behavior)
            for benchmark_dir in benchmarks_path.iterdir():
                if benchmark_dir.is_dir() and benchmark_dir.name.endswith(
                    "-evaluation"
                ):
                    benchmark_name = benchmark_dir.name.replace("-evaluation", "")
                    logger.info(f"Processing benchmark: {benchmark_name}")

                    benchmark_samples = self._load_benchmark_directory(
                        benchmark_dir, benchmark_name
                    )
                    all_samples.extend(benchmark_samples)

        logger.info(f"Loaded {len(all_samples)} total samples")
        for sample in all_samples:
            self._convert_benchmark_name_to_enum(sample)
        return all_samples

    def _load_benchmark_directory(
        self, benchmark_dir: Path, benchmark_name: str
    ) -> List[Dict]:
        """Load all JSONL files from a benchmark directory."""
        samples = []

        for file_path in benchmark_dir.rglob("*.jsonl"):
            file_samples = self._load_jsonl_file(file_path, benchmark_name)
            samples.extend(file_samples)

        return samples

    def _load_jsonl_file(self, file_path: Path, benchmark_name: str) -> List[Dict]:
        """Load data from a single JSONL file."""
        data = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())

                        # Add benchmark name if not present
                        if "benchmark_name" not in sample:
                            sample["benchmark_name"] = benchmark_name

                        # Extract model name from filename if not present
                        if "model_name" not in sample:
                            model_name = self._extract_model_name_from_filename(
                                file_path.name
                            )
                            sample["model_name"] = model_name

                        data.append(sample)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in {file_path}:{line_num}: {e}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

        logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data

    def _extract_model_name_from_filename(self, filename: str) -> str:
        """Extract model name from filename."""
        # Remove .jsonl extension
        name = filename.replace(".jsonl", "")

        # Split by underscore and take the first part as model name
        parts = name.split("_")
        if len(parts) > 1:
            return parts[0]
        else:
            return name

    def extract_user_prompt(self, messages: List[Dict]) -> str:
        """Extract user prompt from messages."""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content is None:
                    content = ""
                return content
        return ""

    def extract_tools_schema(self, messages: List[Dict]) -> Dict:
        """Extract tools schema from messages."""
        tools = set()
        for msg in messages:
            if msg.get("role") == "assistant" and "tool_calls" in msg:
                for tool_call in msg["tool_calls"]:
                    if "function" in tool_call and "name" in tool_call["function"]:
                        tools.add(tool_call["function"]["name"])
        return {"tools": list(tools)} if tools else {}

    def extract_ground_truth_conversation(self, messages: List[Dict]) -> List[Dict]:
        """Extract ground truth conversation from messages."""
        return messages

    def _convert_benchmark_name_to_enum(self, sample: Dict) -> Dict:
        if "benchmark_name" not in sample:
            raise ValueError("Sample missing 'benchmark_name' field")
        benchmark_name = sample["benchmark_name"]
        normalized_name = normalize_benchmark_name(benchmark_name)
        if normalized_name in map(
            normalize_benchmark_name, Benchmark.__members__.keys()
        ):
            original_key = next(
                k
                for k in Benchmark.__members__.keys()
                if normalize_benchmark_name(k) == normalized_name
            )
            sample["benchmark_name"] = Benchmark[original_key]
        else:
            raise ValueError(f"Unknown benchmark name: {benchmark_name}")

    def load_problematic_issues(self, target_benchmark: Optional[List[str]] = None) -> Dict[str, Dict[str, Dict]]:
        """Load problematic issues from CSV files in problematic_questions directory."""
        logger.info("Loading problematic issues...")
        target_list = [name.lower().replace("-", "") for name in target_benchmark]

        # Get the problematic_questions directory relative to the pipeline root
        pipeline_root = Path(__file__).parent.parent  # Go up from src to pipeline
        problematic_dir = pipeline_root / "problematic_questions"
        problematic_issues = {}

        if not problematic_dir.exists():
            logger.warning(
                f"Problematic questions directory not found: {problematic_dir}"
            )
            return problematic_issues

        # Find all CSV files in the directory
        csv_files = list(problematic_dir.glob("*.csv"))

        for csv_file in csv_files:
            # Extract benchmark name from filename (e.g., "DrafterBench_problematic_questions.csv" -> "DrafterBench")
            benchmark_name = csv_file.stem.replace("_problematic_questions", "")
            if benchmark_name.lower().replace("-", "") not in target_list:
                continue
            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    benchmark_issues = {}

                    for row in reader:
                        task_name = row.get("task_name")
                        task_id = row.get("task_id")
                        reason = row.get("issue_reason", "")
                        source = row.get("issue_source", "manually")

                        if task_name and task_id:
                            # Create UniqueQuestionID
                            unique_id = UniqueQuestionID(
                                benchmark=Benchmark(benchmark_name),
                                task_name=task_name,
                                question_id=task_id,
                            )
                            benchmark_issues[unique_id] = {
                                "reason": reason,
                                "source": source,
                            }

                    if benchmark_issues:
                        problematic_issues.update(benchmark_issues)
                        logger.info(
                            f"Loaded {len(benchmark_issues)} problematic issues for {benchmark_name}"
                        )

            except Exception as e:
                logger.error(f"Error loading problematic issues from {csv_file}: {e}")

        return problematic_issues

    def load_human_labelled_ground_truth(self, target_benchmark: Optional[List[str]] = None) -> tuple[Dict[str, Dict[str, Dict]], set]:
        """Load human labelled ground truth data from CSV files in human_labelled_ground_truth directory.

        Returns:
            tuple: (problematic_issues_dict, all_labelled_question_ids_set)
                - problematic_issues_dict: Dict mapping problematic question IDs to their info
                - all_labelled_question_ids_set: Set of all question IDs in the labelled dataset
        """
        logger.info("Loading human labelled ground truth data...")
        target_list = [name.lower().replace("-", "") for name in target_benchmark] if target_benchmark else []

        # Get the human_labelled_ground_truth directory relative to the pipeline root
        pipeline_root = Path(__file__).parent.parent  # Go up from src to pipeline
        ground_truth_dir = pipeline_root / "human_labelled_ground_truth"
        ground_truth_issues = {}
        all_labelled_questions = set()

        if not ground_truth_dir.exists():
            logger.warning(
                f"Human labelled ground truth directory not found: {ground_truth_dir}"
            )
            return ground_truth_issues, all_labelled_questions

        # Find all CSV files in the directory
        csv_files = list(ground_truth_dir.glob("*.csv"))

        for csv_file in csv_files:
            # Extract benchmark name from filename (e.g., "DrafterBench.csv" -> "DrafterBench")
            benchmark_name = csv_file.stem
            if target_list and benchmark_name.lower().replace("-", "") not in target_list:
                continue

            try:
                with open(csv_file, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    benchmark_issues = {}
                    benchmark_all_questions = set()

                    for row in reader:
                        task_name = row.get("task_name")
                        task_id = row.get("task_id")
                        is_issue = row.get("is_issue", "0")
                        issue_type = row.get("issue_type", "")

                        if task_name and task_id:
                            # Create UniqueQuestionID
                            unique_id = UniqueQuestionID(
                                benchmark=Benchmark(benchmark_name),
                                task_name=task_name,
                                question_id=task_id,
                            )

                            # Add to all_labelled_questions regardless of is_issue value
                            benchmark_all_questions.add(unique_id)

                            # Only add to ground_truth_issues if is_issue == "1"
                            if is_issue == "1":
                                benchmark_issues[unique_id] = {
                                    "reason": issue_type,
                                    "source": "human_labelled",
                                }

                    if benchmark_issues:
                        ground_truth_issues.update(benchmark_issues)
                        logger.info(
                            f"Loaded {len(benchmark_issues)} problematic issues from human labelled ground truth for {benchmark_name}"
                        )

                    if benchmark_all_questions:
                        all_labelled_questions.update(benchmark_all_questions)
                        logger.info(
                            f"Loaded {len(benchmark_all_questions)} total labelled questions for {benchmark_name}"
                        )

            except Exception as e:
                logger.error(f"Error loading human labelled ground truth from {csv_file}: {e}")

        return ground_truth_issues, all_labelled_questions
