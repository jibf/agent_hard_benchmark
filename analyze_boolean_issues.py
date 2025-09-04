#!/usr/bin/env python3
"""
Boolean Parameter Type Strictness Analysis for BFCL Claude Results
Analyzes boolean-related errors in BFCL benchmark results for Claude models.
"""

import json
import os
import re
from collections import defaultdict, Counter
from pathlib import Path


class BooleanIssueAnalyzer:
    def __init__(self):
        self.boolean_patterns = [
            # String boolean values that should be proper booleans
            r'=\s*["\']?(true|false)["\']?',
            r'=\s*["\']?(True|False)["\']?',
            r'=\s*["\']?(1|0)["\']?',
            r'=\s*["\']?(yes|no)["\']?',
            r'=\s*["\']?(YES|NO)["\']?',
            r'=\s*["\']?(on|off)["\']?',
            r'=\s*["\']?(ON|OFF)["\']?',
        ]
        
        self.boolean_param_indicators = [
            r'\b(is_|has_|enable_|disable_|should_|can_|will_)',
            r'\b(formatted|debug|verbose|strict|force)',
            r'\b(includeFrom|includeTo|ascending|descending)',
            r'\b(live_conversion|include_|exclude_)',
            r'\b(monitoringEnabled|powerSaveEnabled)',
            r'\b(useShortName|caseInsensitive|inlined)',
            r'\b(ensureNoSelfReferences|writeStartAndEndHeaders)',
            r'\b(allowJavaNames|inContent|aligned)',
        ]
        
        self.stats = {
            'total_function_calls': 0,
            'boolean_issues': [],
            'common_boolean_params': Counter(),
            'string_boolean_values': Counter(),
            'files_analyzed': 0,
        }

    def analyze_function_call(self, result_text):
        """Extract and analyze function calls from result text."""
        # Pattern to match function calls in brackets
        function_call_pattern = r'\[([^\]]+)\]'
        matches = re.findall(function_call_pattern, result_text)
        
        for match in matches:
            self.stats['total_function_calls'] += 1
            self.analyze_single_function_call(match)

    def analyze_single_function_call(self, function_call):
        """Analyze a single function call for boolean issues."""
        # Look for parameter=value patterns with string boolean values
        param_value_pattern = r'(\w+)\s*=\s*["\']?(true|false|True|False|1|0|yes|no|YES|NO|on|off|ON|OFF)["\']?'
        matches = re.finditer(param_value_pattern, function_call, re.IGNORECASE)
        
        for match in matches:
            param_name = match.group(1)
            boolean_value = match.group(2)
            
            # Skip if it's actually a proper boolean (not quoted)
            full_match = match.group(0)
            if not ('"' in full_match or "'" in full_match) and boolean_value in ['True', 'False']:
                continue
            
            # Check if this looks like a boolean parameter that should use proper boolean type
            is_boolean_param = (
                boolean_value.lower() in ['true', 'false', '1', '0', 'yes', 'no', 'on', 'off'] or
                any(indicator in param_name.lower() for indicator in [
                    'is_', 'has_', 'enable', 'disable', 'should_', 'can_', 'will_',
                    'formatted', 'debug', 'verbose', 'strict', 'force', 'include',
                    'exclude', 'ascending', 'descending', 'live_', 'monitor',
                    'power', 'use', 'case', 'inline', 'ensure', 'write', 'allow',
                    'content', 'align'
                ])
            )
            
            if is_boolean_param:
                self.stats['boolean_issues'].append({
                    'function_call': function_call,
                    'param_name': param_name,
                    'boolean_value': boolean_value,
                    'issue_type': 'string_boolean'
                })
                
                self.stats['string_boolean_values'][boolean_value.lower()] += 1
                self.stats['common_boolean_params'][param_name] += 1

    def extract_param_name(self, function_call, match_start):
        """Extract parameter name before the boolean value."""
        before_match = function_call[:match_start]
        param_match = re.search(r'(\w+)\s*=\s*$', before_match)
        return param_match.group(1) if param_match else None

    def analyze_result_file(self, file_path):
        """Analyze a single result file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'result' in data and isinstance(data['result'], str):
                            self.analyze_function_call(data['result'])
                    except json.JSONDecodeError:
                        continue
            self.stats['files_analyzed'] += 1
        except Exception as e:
            print(f"Error analyzing file {file_path}: {e}")

    def analyze_directory(self, directory_path):
        """Analyze all result files in a directory."""
        directory = Path(directory_path)
        if not directory.exists():
            print(f"Directory not found: {directory_path}")
            return

        result_files = list(directory.glob("BFCL_v3_*.json"))
        print(f"Found {len(result_files)} result files in {directory_path}")

        for file_path in result_files:
            print(f"Analyzing: {file_path.name}")
            self.analyze_result_file(file_path)

    def generate_report(self):
        """Generate comprehensive analysis report."""
        total_issues = len(self.stats['boolean_issues'])
        total_calls = self.stats['total_function_calls']
        
        report = f"""
=== Boolean Parameter Type Strictness Analysis ===

Files Analyzed: {self.stats['files_analyzed']}
Total Function Calls: {total_calls:,}
Boolean-related Issues: {total_issues:,}
Issue Rate: {(total_issues/total_calls*100):.2f}% of all function calls

=== Most Common Boolean Parameters with Issues ===
"""
        for param, count in self.stats['common_boolean_params'].most_common(15):
            percentage = (count/total_issues*100) if total_issues > 0 else 0
            report += f"{param}: {count} issues ({percentage:.1f}%)\n"

        report += f"""
=== String Boolean Values Used ===
"""
        for value, count in self.stats['string_boolean_values'].most_common():
            percentage = (count/total_issues*100) if total_issues > 0 else 0
            report += f'"{value}": {count} occurrences ({percentage:.1f}%)\n'

        report += f"""
=== Sample Boolean Issues ===
"""
        # Show first 10 issues as examples
        for i, issue in enumerate(self.stats['boolean_issues'][:10]):
            report += f"""
Issue #{i+1}:
  Function: {issue['function_call'][:100]}{'...' if len(issue['function_call']) > 100 else ''}
  Parameter: {issue['param_name']}
  Value: "{issue['boolean_value']}"
  Type: {issue['issue_type']}
"""

        return report


def main():
    analyzer = BooleanIssueAnalyzer()
    
    # Analyze both Claude model directories
    directories = [
        "E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-off",
        "E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-on-10k"
    ]
    
    print("Starting Boolean Parameter Type Strictness Analysis for Claude Models")
    print("=" * 70)
    
    for directory in directories:
        print(f"\nAnalyzing directory: {directory}")
        analyzer.analyze_directory(directory)
    
    # Generate and display report
    report = analyzer.generate_report()
    print(report)
    
    # Save detailed report to file
    report_file = "E:/Users/김현준/Downloads/agent_hard_benchmark_2/boolean_analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
        
        # Add detailed issue list
        f.write(f"\n\n=== Detailed Issue List ===\n")
        for i, issue in enumerate(analyzer.stats['boolean_issues']):
            f.write(f"\nIssue #{i+1}:\n")
            f.write(f"  Function: {issue['function_call']}\n")
            f.write(f"  Parameter: {issue['param_name']}\n") 
            f.write(f"  Value: \"{issue['boolean_value']}\"\n")
            f.write(f"  Type: {issue['issue_type']}\n")
    
    print(f"\nDetailed report saved to: {report_file}")


if __name__ == "__main__":
    main()