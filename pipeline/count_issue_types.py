#!/usr/bin/env python3
"""
Count the frequency of each issue type from the flawed tasks analysis file.
"""

import re
from collections import Counter

def count_issue_types(analysis_file: str):
    """Count the frequency of each issue type."""
    issue_types = []
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Look for lines with "Error Category:"
            if "Error Category:" in line:
                # Extract the issue type after "Error Category: "
                match = re.search(r"Error Category: (.+)", line)
                if match:
                    issue_type = match.group(1).strip()
                    issue_types.append(issue_type)
    
    # Count frequencies
    issue_counts = Counter(issue_types)
    
    # Print results sorted by frequency (descending)
    print("Issue Type Frequency (excluding special_* categories):")
    print("=" * 60)
    
    total_issues = sum(issue_counts.values())
    print(f"Total flawed tasks: {total_issues}")
    print()
    
    for issue_type, count in issue_counts.most_common():
        percentage = (count / total_issues) * 100
        print(f"{issue_type}: {count} ({percentage:.1f}%)")
    
    return issue_counts

if __name__ == "__main__":
    analysis_file = "flawed_tasks_analysis.txt"
    count_issue_types(analysis_file)


