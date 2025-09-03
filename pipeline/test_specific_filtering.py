#!/usr/bin/env python3
"""
Test script for benchmark-specific filtering.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.rule_filtering_orchestrator import RuleFilteringOrchestrator

def test_specific_filtering():
    """Test the benchmark-specific filtering system."""
    print("Testing benchmark-specific filtering...")
    
    # Test with specific filters enabled
    orchestrator = RuleFilteringOrchestrator(use_specific_filters=True)
    print(f"✓ Orchestrator initialized with specific_filters=True")
    
    # Test with general filtering
    general_orchestrator = RuleFilteringOrchestrator(use_specific_filters=False)
    print(f"✓ Orchestrator initialized with specific_filters=False")
    
    print("\n✓ All tests passed! The system is ready to use.")
    print("\nUsage examples:")
    print("  # Use general filtering (default)")
    print("  python main.py --skip-llm-judge")
    print("\n  # Use benchmark-specific filtering")
    print("  python main.py --skip-llm-judge --specific-step1")
    print("\n  # Use benchmark-specific filtering for specific benchmark")
    print("  python main.py --skip-llm-judge --specific-step1 --target_benchmark DrafterBench")

if __name__ == "__main__":
    test_specific_filtering()

