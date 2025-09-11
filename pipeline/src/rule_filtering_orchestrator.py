import json
import os
from typing import Dict, List, Any, Tuple
from .comprehensive_rule_filtering import ComprehensiveRuleFilter
from .benchmark_specific_filters import (
    DrafterBenchFilter, MultiChallengeFilter, ACEBenchFilter,
    ComplexFuncBenchFilter, BFCLFilter, NexusBenchFilter, TAUBenchFilter
)
from .utils import normalize_benchmark_name, EnumJSONEncoder


class RuleFilteringOrchestrator:
    """Orchestrates rule-based filtering with benchmark-specific and general filters."""
    
    def __init__(self):
        self.comprehensive_filter = ComprehensiveRuleFilter()
        self.benchmark_filters = {
            'DrafterBench': DrafterBenchFilter(),
            'multi_challenge': MultiChallengeFilter(),
            'ACEBench': ACEBenchFilter(),
            'complex-func-bench': ComplexFuncBenchFilter(),
            'BFCL': BFCLFilter(),
            'NexusBench': NexusBenchFilter(),
            'tau-bench': TAUBenchFilter()
        }
    
    def filter_samples(self, samples: List[Dict], use_specific_filters: bool = False, 
                      target_benchmark: str = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter samples using benchmark-specific filtering.
        Note: This is now called AFTER comprehensive filtering has been applied.
        
        Args:
            samples: List of samples to filter (already filtered by comprehensive filter)
            use_specific_filters: Whether to use benchmark-specific filters
            target_benchmark: Specific benchmark to filter for
            
        Returns:
            Tuple of (passed_samples, dropped_samples)
        """
        if use_specific_filters and target_benchmark:
            # Use benchmark-specific filter (comprehensive filtering already applied)
            if normalize_benchmark_name(target_benchmark) in map(normalize_benchmark_name, self.benchmark_filters.keys()):
                # Find the original key that matches the normalized target
                original_key = next(k for k in self.benchmark_filters.keys() if normalize_benchmark_name(k) == normalize_benchmark_name(target_benchmark))
                filter_instance = self.benchmark_filters[original_key]
                passed_samples, dropped_samples = filter_instance.filter_samples(samples)
                filter_name = filter_instance.get_filter_name()
                
                # Save results
                self._save_filtered_results(passed_samples, filter_name, target_benchmark)
                
                return passed_samples, dropped_samples
            else:
                print(f"Warning: No specific filter found for {target_benchmark}, returning samples as-is")
                # Return samples unchanged since comprehensive filtering was already applied
                return samples, []
        else:
            # Return samples unchanged since comprehensive filtering was already applied
            return samples, []
    
    def _save_filtered_results(self, filtered_samples: List[Dict[str, Any]], 
                              filter_name: str, target_benchmark: str = None) -> Dict[str, str]:
        """
        Save filtered results to both unified and benchmark-specific files.
        
        Args:
            filtered_samples: List of filtered samples
            filter_name: Name of the filter used
            target_benchmark: Specific benchmark if applicable
            
        Returns:
            Dict containing file paths where results were saved
        """
        # Create output directory
        output_dir = "filtered_datasets"
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        # 1. Save unified pruned dataset
        unified_file = os.path.join(output_dir, f"unified_pruned_{filter_name}.jsonl")
        self._save_samples_to_jsonl(filtered_samples, unified_file)
        saved_files['unified'] = unified_file
        
        # 2. Save benchmark-specific files
        if target_benchmark:
            # Save single benchmark file
            benchmark_file = os.path.join(output_dir, f"{target_benchmark}_pruned_{filter_name}.jsonl")
            self._save_samples_to_jsonl(filtered_samples, benchmark_file)
            saved_files['benchmark_specific'] = benchmark_file
        else:
            # Group samples by benchmark and save separate files
            benchmark_groups = self._group_samples_by_benchmark(filtered_samples)
            
            for benchmark_name, benchmark_samples in benchmark_groups.items():
                if benchmark_samples:  # Only save if there are samples
                    benchmark_file = os.path.join(output_dir, f"{benchmark_name}_pruned_{filter_name}.jsonl")
                    self._save_samples_to_jsonl(benchmark_samples, benchmark_file)
                    saved_files[benchmark_name] = benchmark_file
        
        print(f"Saved filtered results:")
        print(f"  - Unified dataset: {saved_files['unified']}")
        for key, path in saved_files.items():
            if key != 'unified':
                print(f"  - {key}: {path}")
        
        return saved_files
    
    def _group_samples_by_benchmark(self, samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group samples by benchmark name."""
        benchmark_groups = {}
        
        for sample in samples:
            benchmark_name = sample.get('benchmark_name', 'unknown')
            if benchmark_name not in benchmark_groups:
                benchmark_groups[benchmark_name] = []
            benchmark_groups[benchmark_name].append(sample)
        
        return benchmark_groups
    
    def _save_samples_to_jsonl(self, samples: List[Dict[str, Any]], file_path: str):
        """Save samples to a JSONL file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    json.dump(sample, f, ensure_ascii=False, cls=EnumJSONEncoder)
                    f.write('\n')
            print(f"Saved {len(samples)} samples to {file_path}")
        except Exception as e:
            print(f"Error saving to {file_path}: {e}")

