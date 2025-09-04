#!/usr/bin/env python3
"""
Final Comprehensive Boolean Parameter Type Strictness Analysis
Creates a detailed analysis comparing thinking-off vs thinking-on performance.
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path


class FinalBooleanAnalyzer:
    def __init__(self):
        self.results = {
            'thinking-off': {'total_calls': 0, 'boolean_issues': [], 'files': 0},
            'thinking-on': {'total_calls': 0, 'boolean_issues': [], 'files': 0}
        }
        
        # Common boolean parameter patterns we care about
        self.boolean_indicators = [
            r'\bis_\w+', r'\bhas_\w+', r'\benable\w*', r'\bdisable\w*',
            r'\binclude\w*', r'\bexclude\w*', r'\bshow_\w+', r'\bhide_\w+',
            r'\buse\w*', r'\ballow\w*', r'\bforce\w*', r'\bstrict\w*',
            r'\bformatted?\b', r'\bdebug\b', r'\bverbose\b', r'\brecursive\b',
            r'\bascending\b', r'\bdescending\b', r'\breverse\b', r'\bsorted\b',
            r'\bcase_?insensitive\b', r'\bignore_case\b',
            r'\bgood_for_\w+', r'\bfree_\w+', r'\bin_unit_\w+'
        ]
        
        self.string_booleans = ['true', 'false', 'yes', 'no', 'on', 'off']

    def is_boolean_parameter(self, param_name, value):
        """Check if parameter should be boolean based on name and value."""
        param_lower = param_name.lower()
        value_lower = value.lower() if isinstance(value, str) else str(value).lower()
        
        # Check parameter name patterns
        name_suggests_boolean = any(
            re.search(pattern, param_lower, re.IGNORECASE) 
            for pattern in self.boolean_indicators
        )
        
        # Check if value is string boolean
        value_is_string_boolean = value_lower in self.string_booleans
        
        return name_suggests_boolean and value_is_string_boolean

    def extract_parameters(self, function_call):
        """Extract parameters from function call."""
        # Match param=value patterns (handle quoted and unquoted values)
        pattern = r'(\w+)\s*=\s*(["\']?)([^,\]]+?)\2(?=\s*[,\)]|\s*$)'
        matches = re.findall(pattern, function_call)
        
        parameters = []
        for param_name, quote_char, value in matches:
            is_quoted = bool(quote_char)
            parameters.append((param_name, value.strip(), is_quoted))
        
        return parameters

    def analyze_function_call(self, function_call, model_type):
        """Analyze single function call for boolean issues."""
        self.results[model_type]['total_calls'] += 1
        
        parameters = self.extract_parameters(function_call)
        
        for param_name, value, is_quoted in parameters:
            if is_quoted and self.is_boolean_parameter(param_name, value):
                issue = {
                    'function_call': function_call,
                    'param_name': param_name,
                    'string_value': value,
                    'expected_boolean': self.convert_to_boolean(value)
                }
                self.results[model_type]['boolean_issues'].append(issue)

    def convert_to_boolean(self, string_value):
        """Convert string to expected boolean value."""
        value_lower = string_value.lower()
        if value_lower in ['true', 'yes', 'on', '1']:
            return 'True'
        elif value_lower in ['false', 'no', 'off', '0']:
            return 'False'
        return f'convert("{string_value}")'

    def analyze_directory(self, directory_path, model_type):
        """Analyze all result files in directory."""
        directory = Path(directory_path)
        if not directory.exists():
            print(f"Directory not found: {directory_path}")
            return
            
        result_files = list(directory.glob("BFCL_v3_*.json"))
        print(f"Analyzing {len(result_files)} files for {model_type}")
        
        for file_path in result_files:
            self.analyze_file(file_path, model_type)
            
        self.results[model_type]['files'] = len(result_files)

    def analyze_file(self, file_path, model_type):
        """Analyze single result file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'result' in data and isinstance(data['result'], str):
                            # Extract function calls from result
                            function_calls = re.findall(r'\[([^\]]+)\]', data['result'])
                            for call in function_calls:
                                self.analyze_function_call(call, model_type)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def generate_comprehensive_report(self):
        """Generate detailed comparison report."""
        off_issues = len(self.results['thinking-off']['boolean_issues'])
        on_issues = len(self.results['thinking-on']['boolean_issues'])
        off_calls = self.results['thinking-off']['total_calls']
        on_calls = self.results['thinking-on']['total_calls']
        
        report = f"""
=== Comprehensive Boolean Parameter Type Strictness Analysis ===

Claude 4 Sonnet Thinking Off:
  - Files Analyzed: {self.results['thinking-off']['files']}
  - Total Function Calls: {off_calls:,}
  - Boolean Type Issues: {off_issues:,}
  - Issue Rate: {(off_issues/off_calls*100):.2f}%

Claude 4 Sonnet Thinking On (10k):
  - Files Analyzed: {self.results['thinking-on']['files']}
  - Total Function Calls: {on_calls:,}
  - Boolean Type Issues: {on_issues:,}
  - Issue Rate: {(on_issues/on_calls*100):.2f}%

=== Comparison ===
"""
        
        if off_calls > 0 and on_calls > 0:
            improvement = ((off_issues/off_calls) - (on_issues/on_calls)) * 100
            if improvement > 0:
                report += f"Thinking mode IMPROVED boolean handling by {improvement:.2f} percentage points\n"
            elif improvement < 0:
                report += f"Thinking mode WORSENED boolean handling by {abs(improvement):.2f} percentage points\n"
            else:
                report += "No significant difference between thinking modes\n"
        
        # Analyze most common problematic parameters
        off_params = Counter(issue['param_name'] for issue in self.results['thinking-off']['boolean_issues'])
        on_params = Counter(issue['param_name'] for issue in self.results['thinking-on']['boolean_issues'])
        
        report += f"""
=== Most Problematic Boolean Parameters ===

Thinking Off - Top 10:
"""
        for param, count in off_params.most_common(10):
            percentage = (count/off_issues*100) if off_issues > 0 else 0
            report += f"  {param}: {count} issues ({percentage:.1f}%)\n"
            
        report += f"""
Thinking On - Top 10:
"""
        for param, count in on_params.most_common(10):
            percentage = (count/on_issues*100) if on_issues > 0 else 0
            report += f"  {param}: {count} issues ({percentage:.1f}%)\n"

        # String boolean value distribution
        off_values = Counter(issue['string_value'].lower() for issue in self.results['thinking-off']['boolean_issues'])
        on_values = Counter(issue['string_value'].lower() for issue in self.results['thinking-on']['boolean_issues'])
        
        report += f"""
=== String Boolean Value Distribution ===

Thinking Off:
"""
        for value, count in off_values.most_common():
            percentage = (count/off_issues*100) if off_issues > 0 else 0
            report += f'  "{value}": {count} ({percentage:.1f}%)\n'

        report += f"""
Thinking On:
"""
        for value, count in on_values.most_common():
            percentage = (count/on_issues*100) if on_issues > 0 else 0
            report += f'  "{value}": {count} ({percentage:.1f}%)\n'

        # Sample issues from each model
        report += f"""
=== Sample Issues ===

Thinking Off Examples:
"""
        for i, issue in enumerate(self.results['thinking-off']['boolean_issues'][:5]):
            report += f"""
Example {i+1}:
  Function: {issue['function_call'][:100]}{'...' if len(issue['function_call']) > 100 else ''}
  Parameter: {issue['param_name']} = "{issue['string_value']}" → should be {issue['expected_boolean']}
"""

        report += f"""
Thinking On Examples:
"""
        for i, issue in enumerate(self.results['thinking-on']['boolean_issues'][:5]):
            report += f"""
Example {i+1}:
  Function: {issue['function_call'][:100]}{'...' if len(issue['function_call']) > 100 else ''}
  Parameter: {issue['param_name']} = "{issue['string_value']}" → should be {issue['expected_boolean']}
"""

        # Calculate key statistics
        total_issues = off_issues + on_issues
        total_calls = off_calls + on_calls
        
        report += f"""
=== Summary Statistics ===
- Total Function Calls Analyzed: {total_calls:,}
- Total Boolean Type Issues: {total_issues:,}
- Overall Issue Rate: {(total_issues/total_calls*100):.2f}%
- Most Common String Boolean: "{max(off_values.keys(), key=lambda k: off_values[k] + on_values[k])}"
- Most Problematic Parameter: "{max(set(off_params.keys()) | set(on_params.keys()), key=lambda k: off_params[k] + on_params[k])}"

=== Conclusion ===
"""
        if improvement > 0.5:
            report += f"Claude with thinking enabled shows significantly better boolean parameter handling, with {improvement:.2f} percentage points fewer type errors."
        elif improvement < -0.5:
            report += f"Claude with thinking enabled shows worse boolean parameter handling, with {abs(improvement):.2f} percentage points more type errors."
        else:
            report += "No significant difference in boolean parameter handling between thinking modes."
            
        report += f"""

Common Issues Observed:
1. Parameters like 'good_for_kids', 'free_entry', 'has_laundry_service' frequently use string "True" instead of boolean True
2. String values "true" and "false" are most commonly misused instead of proper boolean types
3. Boolean parameters with clear naming conventions (is_, has_, enable_, etc.) still receive string values
4. This suggests a systematic issue with type strictness in function calling benchmarks
"""

        return report


def main():
    analyzer = FinalBooleanAnalyzer()
    
    directories = {
        'thinking-off': "E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-off",
        'thinking-on': "E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-on-10k"
    }
    
    print("Starting Comprehensive Boolean Parameter Analysis")
    print("=" * 55)
    
    for model_type, directory in directories.items():
        analyzer.analyze_directory(directory, model_type)
    
    report = analyzer.generate_comprehensive_report()
    print(report)
    
    # Save comprehensive report
    report_file = "E:/Users/김현준/Downloads/agent_hard_benchmark_2/final_boolean_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
        
        # Add detailed issue lists for both models
        f.write("\n\n=== Complete Issue Lists ===\n")
        
        for model_type in ['thinking-off', 'thinking-on']:
            f.write(f"\n--- {model_type.title()} Issues ---\n")
            issues = analyzer.results[model_type]['boolean_issues']
            for i, issue in enumerate(issues):
                f.write(f"\nIssue #{i+1}:\n")
                f.write(f"  Function: {issue['function_call']}\n")
                f.write(f"  Parameter: {issue['param_name']}\n")
                f.write(f"  Current: \"{issue['string_value']}\"\n")
                f.write(f"  Expected: {issue['expected_boolean']}\n")
    
    print(f"\nDetailed report saved to: {report_file}")


if __name__ == "__main__":
    main()