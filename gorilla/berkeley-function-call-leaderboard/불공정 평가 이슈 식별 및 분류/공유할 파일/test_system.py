#!/usr/bin/env python3
"""
Simple test script for BFCL Unfair Evaluation Detection System
"""

import sys
import traceback
from pathlib import Path

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from data_loader import load_bfcl_results, BFCLDataLoader
        print("  - data_loader: OK")
    except Exception as e:
        print(f"  - data_loader: FAILED - {e}")
        return False
        
    try:
        from unfair_evaluation_detector import UnfairEvaluationDetector
        print("  - unfair_evaluation_detector: OK")
    except Exception as e:
        print(f"  - unfair_evaluation_detector: FAILED - {e}")
        return False
        
    return True

def test_data_structure():
    """Test data structure validation"""
    print("\nTesting data structure...")
    
    try:
        from data_loader import BFCLDataLoader
        
        base_path = r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard"
        loader = BFCLDataLoader(base_path)
        
        is_valid = loader.validate_data_structure()
        print(f"  - Data structure validation: {'PASSED' if is_valid else 'FAILED'}")
        
        return is_valid
        
    except Exception as e:
        print(f"  - Data structure test: FAILED - {e}")
        return False

def test_data_loading():
    """Test data loading with small sample"""
    print("\nTesting data loading...")
    
    try:
        from data_loader import BFCLDataLoader
        
        base_path = r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard"
        loader = BFCLDataLoader(base_path)
        
        # Try to load just one model's data
        result_path = loader.result_path
        model_dirs = [d for d in result_path.iterdir() if d.is_dir()]
        
        if not model_dirs:
            print("  - No model directories found")
            return False
            
        # Use first model for testing
        test_model = model_dirs[0]
        print(f"  - Testing with model: {test_model.name}")
        
        json_files = list(test_model.glob("*.json"))
        if not json_files:
            print("  - No JSON files found in model directory")
            return False
            
        print(f"  - Found {len(json_files)} JSON files")
        
        # Try loading full data (this might take a while)
        df = loader.load_all_results()
        
        if df is not None and len(df) > 0:
            print(f"  - Successfully loaded {len(df)} records")
            print(f"  - Columns: {list(df.columns)[:5]}..." if len(df.columns) > 5 else f"  - Columns: {list(df.columns)}")
            return True
        else:
            print("  - No data loaded")
            return False
            
    except Exception as e:
        print(f"  - Data loading test: FAILED - {e}")
        traceback.print_exc()
        return False

def test_detector():
    """Test detector with minimal data"""
    print("\nTesting detector...")
    
    try:
        import pandas as pd
        from unfair_evaluation_detector import UnfairEvaluationDetector
        
        # Create minimal test data
        test_data = [
            {
                'model_name': 'test_model',
                'test_category': 'test_category',
                'task_id': 'task_1',
                'score': 0.0,
                'error_message': 'max_tokens required',
                'model_result': '',
                'input_tokens': 0,
                'output_tokens': 0,
                'is_summary': False
            },
            {
                'model_name': 'test_model',
                'test_category': 'test_category',
                'task_id': 'task_2',
                'score': 1.0,
                'error_message': '',
                'model_result': 'success',
                'input_tokens': 100,
                'output_tokens': 50,
                'is_summary': False
            }
        ]
        
        df = pd.DataFrame(test_data)
        detector = UnfairEvaluationDetector(df)
        
        # Test individual detection methods
        tech_errors = detector.detect_technical_errors()
        api_bias = detector.detect_api_configuration_bias()
        
        print(f"  - Technical errors detected: {tech_errors}")
        print(f"  - API bias detected: {api_bias}")
        print("  - Detector test: PASSED")
        
        return True
        
    except Exception as e:
        print(f"  - Detector test: FAILED - {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("BFCL Unfair Evaluation Detection System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Structure Test", test_data_structure),
        ("Data Loading Test", test_data_loading),
        ("Detector Test", test_detector),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  EXCEPTION in {test_name}: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        print(f"{test_name:25}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("All tests passed! System appears to be working correctly.")
        return 0
    else:
        print("Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)