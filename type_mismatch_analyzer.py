import json
import os
import re
from typing import Dict, List, Tuple, Any
from collections import defaultdict

class TypeMismatchAnalyzer:
    def __init__(self):
        self.type_patterns = {
            'string_bool': [r'"true"', r'"false"'],
            'string_number': [r'"\d+"', r'"\d+\.\d+"'],
            'string_null': [r'"null"', r'"None"'],
            'mixed_quotes': [r"'[^']*'", r'"[^"]*"']  # Different quote styles
        }
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single JSON file for type mismatches."""
        results = {
            'total_entries': 0,
            'type_errors': defaultdict(int),
            'examples': defaultdict(list),
            'file_path': file_path
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    
                    results['total_entries'] += 1
                    
                    # Check for type mismatch patterns in the raw line
                    for error_type, patterns in self.type_patterns.items():
                        for pattern in patterns:
                            matches = re.findall(pattern, line)
                            if matches:
                                results['type_errors'][error_type] += len(matches)
                                if len(results['examples'][error_type]) < 5:  # Keep max 5 examples
                                    results['examples'][error_type].extend(matches[:5])
                    
                    # Also check specific function call patterns
                    self._check_function_call_patterns(line, results, line_num)
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
        return results
    
    def _check_function_call_patterns(self, line: str, results: Dict, line_num: int):
        """Check for type mismatches in function call patterns."""
        # Look for boolean parameters that should be boolean but are strings
        bool_param_pattern = r'(\w+)=(["\'])(?:true|false|True|False)\2'
        matches = re.findall(bool_param_pattern, line)
        for param_name, quote in matches:
            results['type_errors']['param_bool_string'] += 1
            if len(results['examples']['param_bool_string']) < 5:
                results['examples']['param_bool_string'].append(f"Line {line_num}: {param_name}={quote}true{quote} or {quote}false{quote}")
        
        # Look for numeric parameters that are strings
        num_param_pattern = r'(\w+)=(["\'])(\d+(?:\.\d+)?)\2'
        matches = re.findall(num_param_pattern, line)
        for param_name, quote, number in matches:
            results['type_errors']['param_number_string'] += 1
            if len(results['examples']['param_number_string']) < 5:
                results['examples']['param_number_string'].append(f"Line {line_num}: {param_name}={quote}{number}{quote}")
    
    def compare_original_vs_fixed(self, original_path: str, fixed_path: str) -> Dict[str, Any]:
        """Compare original and fixed files."""
        original_results = self.analyze_file(original_path)
        fixed_results = self.analyze_file(fixed_path)
        
        comparison = {
            'original': original_results,
            'fixed': fixed_results,
            'improvements': {}
        }
        
        # Calculate improvements
        for error_type in original_results['type_errors']:
            original_count = original_results['type_errors'][error_type]
            fixed_count = fixed_results['type_errors'].get(error_type, 0)
            improvement = original_count - fixed_count
            improvement_percentage = (improvement / original_count * 100) if original_count > 0 else 0
            
            comparison['improvements'][error_type] = {
                'original_count': original_count,
                'fixed_count': fixed_count,
                'improvement': improvement,
                'improvement_percentage': improvement_percentage
            }
        
        return comparison

def main():
    analyzer = TypeMismatchAnalyzer()
    
    base_paths = [
        "E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-off",
        "E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-on-10k"
    ]
    
    all_results = {}
    
    for base_path in base_paths:
        model_name = os.path.basename(base_path)
        all_results[model_name] = {}
        
        # Analyze original files
        print(f"\nAnalyzing {model_name}...")
        for filename in os.listdir(base_path):
            if filename.endswith('.json'):
                file_path = os.path.join(base_path, filename)
                results = analyzer.analyze_file(file_path)
                all_results[model_name][filename] = results
                
                total_errors = sum(results['type_errors'].values())
                if total_errors > 0:
                    print(f"  {filename}: {total_errors} type errors out of {results['total_entries']} entries")
        
        # Analyze fixed files if they exist
        fixed_path = os.path.join(base_path, "fixed")
        if os.path.exists(fixed_path):
            print(f"\nAnalyzing fixed files for {model_name}...")
            for filename in os.listdir(fixed_path):
                if filename.endswith('.json'):
                    original_name = filename.replace('_fixed', '')
                    original_file = os.path.join(base_path, original_name)
                    fixed_file = os.path.join(fixed_path, filename)
                    
                    if os.path.exists(original_file):
                        comparison = analyzer.compare_original_vs_fixed(original_file, fixed_file)
                        print(f"  {filename}:")
                        for error_type, improvement in comparison['improvements'].items():
                            if improvement['original_count'] > 0:
                                print(f"    {error_type}: {improvement['original_count']} -> {improvement['fixed_count']} ({improvement['improvement_percentage']:.1f}% improvement)")
    
    # Calculate overall statistics
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    
    for model_name, files in all_results.items():
        print(f"\n{model_name.upper()}:")
        total_entries = sum(file_data['total_entries'] for file_data in files.values())
        total_type_errors = sum(sum(file_data['type_errors'].values()) for file_data in files.values())
        
        if total_entries > 0:
            error_percentage = (total_type_errors / total_entries) * 100
            print(f"  Total entries: {total_entries}")
            print(f"  Total type errors: {total_type_errors}")
            print(f"  Error percentage: {error_percentage:.2f}%")
            
            # Show error type breakdown
            error_breakdown = defaultdict(int)
            for file_data in files.values():
                for error_type, count in file_data['type_errors'].items():
                    error_breakdown[error_type] += count
            
            print("  Error type breakdown:")
            for error_type, count in sorted(error_breakdown.items()):
                percentage = (count / total_type_errors) * 100 if total_type_errors > 0 else 0
                print(f"    {error_type}: {count} ({percentage:.1f}%)")
                
            # Show examples
            print("\n  Examples of type mismatches:")
            for file_data in files.values():
                for error_type, examples in file_data['examples'].items():
                    if examples and error_breakdown[error_type] > 0:
                        print(f"    {error_type}:")
                        for example in examples[:3]:  # Show max 3 examples
                            print(f"      - {example}")

if __name__ == "__main__":
    main()