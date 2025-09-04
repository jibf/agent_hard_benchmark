#!/usr/bin/env python3
"""
Data loader module for benchmark filtering pipeline.
Handles loading and preprocessing of benchmark data from JSONL files.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class BenchmarkDataLoader:
    """Loads and preprocesses benchmark data from JSONL files."""
    
    def __init__(self):
        self.loaded_samples = []
    
    def load_benchmark_data(self, benchmarks_dir: str, target_benchmark: Optional[str] = None) -> List[Dict]:
        """Load benchmark data from the specified directory.
        
        Args:
            benchmarks_dir: Directory containing benchmark data
            target_benchmark: If specified, only load this specific benchmark
        """
        logger.info(f"Loading benchmark data from {benchmarks_dir}")
        
        benchmarks_path = Path(benchmarks_dir)
        if not benchmarks_path.exists():
            raise FileNotFoundError(f"Benchmarks directory not found: {benchmarks_dir}")
        
        all_samples = []
        
        if target_benchmark:
            # Load only the target benchmark
            if target_benchmark in ["tau_bench", "tau2_bench"]:
                target_dir = benchmarks_path / f"{target_benchmark.replace("_", "-")}-evaluation"
            else:
                target_dir = benchmarks_path / f"{target_benchmark}-evaluation"
            if target_dir.exists() and target_dir.is_dir():
                logger.info(f"Loading only target benchmark: {target_benchmark}")
                benchmark_samples = self._load_benchmark_directory(target_dir, target_benchmark)
                all_samples.extend(benchmark_samples)
            else:
                logger.warning(f"Target benchmark directory not found: {target_dir}")
                return []
        else:
            # Process each benchmark directory (original behavior)
            for benchmark_dir in benchmarks_path.iterdir():
                if benchmark_dir.is_dir() and benchmark_dir.name.endswith('-evaluation'):
                    benchmark_name = benchmark_dir.name.replace('-evaluation', '')
                    logger.info(f"Processing benchmark: {benchmark_name}")
                    
                    benchmark_samples = self._load_benchmark_directory(benchmark_dir, benchmark_name)
                    all_samples.extend(benchmark_samples)
        
        logger.info(f"Loaded {len(all_samples)} total samples")
        return all_samples
    
    def _load_benchmark_directory(self, benchmark_dir: Path, benchmark_name: str) -> List[Dict]:
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
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        sample = json.loads(line.strip())
                        
                        # Add benchmark name if not present
                        if 'benchmark_name' not in sample:
                            sample['benchmark_name'] = benchmark_name
                        
                        # Extract model name from filename if not present
                        if 'model_name' not in sample:
                            model_name = self._extract_model_name_from_filename(file_path.name)
                            sample['model_name'] = model_name
                        
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
        name = filename.replace('.jsonl', '')
        
        # Split by underscore and take the first part as model name
        parts = name.split('_')
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
