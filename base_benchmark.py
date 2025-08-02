# system import
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Iterable
import concurrent.futures

# third-party import
import pandas as pd
from openai import OpenAI
from tqdm import tqdm


@dataclass
class BenchmarkConfig:
    """Base configuration for benchmarking"""
    model_name: str = "internVL-1B"
    api_key: str = "YOUR-API-KEY"
    base_url: str = "YOUR-URL"
    max_tokens: int = 1024
    temperature: float = 0.0
    max_workers: int = 4
    cache_interval: int = 20
    cache_dir: str = f"./benchmark_cache/{model_name}"
    results_dir: str = f"./benchmark_results/{model_name}"


class BaseBenchmark(ABC):
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        os.makedirs(config.cache_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)

        # Download dataset if needed
        self.dataset = self.prepare_data()

        # Resume evaluation from cache
        self.results = self._load_cache()
        self.processed_ids = set(self.results.keys())

    @property
    def cache_path(self) -> Path:
        """Get path for cached responses"""
        return Path(self.config.cache_dir) / f"{self.__class__.__name__}_cache.json"

    @property
    def results_path(self) -> Path:
        """Get path for evaluation results"""
        return Path(self.config.results_dir) / f"{self.__class__.__name__}_results.csv"

    @abstractmethod
    def prepare_data(self) -> List[Dict[str, Any]]:
        """Load benchmark dataset, downloading if necessary"""
        raise NotImplementedError("Implement `prepare_data` in subclass")

    def build_dataloader(self) -> Iterable[Dict[str, Any]]:
        """Create data loader for pending items"""
        pending_items = [
            item for item in self.dataset 
            if item["id"] not in self.processed_ids
        ]
        return pending_items

    def get_model_response(self, messages: List[Dict]) -> str:
        """Get model response synchronously"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"API call failed: {str(e)}")
            return ""

    @abstractmethod
    def evaluate_single(self, output: str, reference: str) -> Dict:
        """Evaluate single sample"""
        raise NotImplementedError("Implement `evaluate_single` in subclass")

    def run_evaluation(self, progress_bar: bool = True):
        """Execute evaluation pipeline with concurrent processing"""
        pending_items = self.build_dataloader()
        if not pending_items:
            logging.info("All items already processed")
            return

        # Process items concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_workers
        ) as executor:
            futures = {
                executor.submit(self._process_item, item): item["id"]
                for item in pending_items
            }
            
            completed = 0
            with tqdm(
                total=len(futures), 
                desc="Processing items",
                disable=not progress_bar
            ) as pbar:
                for future in concurrent.futures.as_completed(futures):
                    item_id = futures[future]
                    try:
                        result = future.result()
                        self.results[item_id] = result
                        completed += 1
                        
                        # Save progress periodically
                        if completed % self.config.cache_interval == 0:
                            self._save_cache()
                            
                    except Exception as e:
                        logging.error(f"Failed processing item {item_id}: {str(e)}")
                    finally:
                        pbar.update(1)

        # Final save
        self._save_cache()

    def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Process single evaluation item"""
        output = self.get_model_response(item["input"])
        scores = self.evaluate_single(output, item["reference"])
        
        return {
            "id": item["id"],
            "input": item["input"],
            "output": output,
            "reference": item["reference"],
            **scores
        }

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        """Load cached results"""
        if not self.cache_path.exists():
            return {}

        try:
            with open(self.cache_path, "r") as f:
                results = json.load(f)
            logging.info(f"Resumed {len(results)} evaluation results from {self.cache_path}")
            return results
        except Exception as e:
            logging.error(f"Cache loading failed: {str(e)}")
            return {}

    def _save_cache(self):
        """Save results to cache"""
        try:
            with open(self.cache_path, "w") as f:
                json.dump(self.results, f, indent=2)
            logging.info(f"Saved {len(self.results)} results to cache")
        except Exception as e:
            logging.error(f"Cache saving failed: {str(e)}")

    def generate_scores(self) -> str:
        """Generate evaluation report"""
        if not self.results:
            return "No results available - run evaluation first"

        # Convert to DataFrame
        results_list = list(self.results.values())
        df = pd.DataFrame(results_list)
        
        # Calculate aggregate metrics
        metric_cols = [col for col in df.columns if col not in {"id", "input", "output", "reference"}]
        summary = {"num_samples": len(df)}
        
        # Save results
        df.to_csv(self.results_path, index=False)
        summary_path = self.results_path.with_suffix(".summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Generate report
        report = f"# {self.__class__.__name__} Evaluation Report\n\n"
        report += "## Aggregate Metrics\n"
        report += "| Metric | Value |\n|-------|-------|\n"
        for k, v in summary.items():
            if isinstance(v, float):
                report += f"| {k} | {v:.4f} |\n"
            else:
                report += f"| {k} | {v} |\n"
        
        report += "\n## Sample Results\n"
        report += df.head(5).to_markdown(index=False)

        return report