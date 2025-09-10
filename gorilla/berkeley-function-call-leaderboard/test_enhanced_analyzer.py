#!/usr/bin/env python3
"""
Test script for Enhanced BFCL Functionality Analyzer
Tests core functionality with a small dataset before full deployment
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List
import time

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test if all required imports work"""
    print("Testing imports...")
    
    try:
        from enhanced_functionality_analyzer import EnhancedFunctionalityAnalyzer
        print("+ Enhanced analyzer import successful")
    except ImportError as e:
        print(f"- Enhanced analyzer import failed: {e}")
        return False
    
    try:
        import requests
        import psutil
        from dotenv import load_dotenv
        print("+ External dependencies import successful")
    except ImportError as e:
        print(f"- External dependencies import failed: {e}")
        return False
    
    return True

def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    # Check .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("- .env file not found")
        return False
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('API_KEY')
    base_url = os.getenv('BASE_URL')
    
    if not api_key or not base_url:
        print("- API_KEY or BASE_URL not found in .env")
        return False
    
    print("+ Configuration loaded successfully")
    print(f"  Base URL: {base_url}")
    print(f"  API Key: {api_key[:20]}...")
    
    return True

def test_data_loading():
    """Test data loading functionality"""
    print("\nTesting data loading...")
    
    try:
        from enhanced_functionality_analyzer import EnhancedFunctionalityAnalyzer
        analyzer = EnhancedFunctionalityAnalyzer(num_workers=1)
        
        # Test loading a simple task
        test_tasks = ['simple', 'irrelevance']  # Start with smaller datasets
        
        for task in test_tasks:
            # Check if data file exists
            data_file = Path(f"bfcl_eval/data/BFCL_v3_{task}.json")
            if not data_file.exists():
                print(f"- Data file not found: {data_file}")
                continue
            
            # Try to load data
            test_cases = analyzer.load_test_data_with_context(task)
            if test_cases:
                print(f"+ Loaded {len(test_cases)} cases for {task}")
                
                # Test first case structure
                first_case = test_cases[0]
                print(f"  Sample case ID: {first_case.get('id', 'unknown')}")
                print(f"  Has functions: {len(first_case.get('function', []))}")
                print(f"  Has question: {bool(first_case.get('question'))}")
            else:
                print(f"- No test cases loaded for {task}")
                return False
    
    except Exception as e:
        print(f"- Data loading failed: {e}")
        return False
    
    return True

def test_prompt_generation():
    """Test enhanced prompt generation"""
    print("\nTesting prompt generation...")
    
    try:
        from enhanced_functionality_analyzer import EnhancedFunctionalityAnalyzer
        analyzer = EnhancedFunctionalityAnalyzer(num_workers=1)
        
        # Create a test case
        test_case = {
            'id': 'test_case_1',
            'question': [[{"role": "user", "content": "Find the area of a triangle with base 10 and height 5."}]],
            'function': [{
                'name': 'calculate_triangle_area',
                'description': 'Calculate the area of a triangle.',
                'parameters': {
                    'type': 'object',
                    'properties': {
                        'base': {'type': 'number', 'description': 'Base of triangle'},
                        'height': {'type': 'number', 'description': 'Height of triangle'}
                    },
                    'required': ['base', 'height']
                }
            }],
            'task_name': 'simple'
        }
        
        # Test prompt generation
        prompt = analyzer.create_enhanced_analysis_prompt(test_case, "openai/gpt-4.1")
        
        if len(prompt) > 100 and "**VERDICT**:" in prompt:
            print("+ Enhanced prompt generated successfully")
            print(f"  Prompt length: {len(prompt)} characters")
            print(f"  Contains BFCL context: {'BFCL System Prompt' in prompt}")
            print(f"  Contains analysis format: {'**DETAILED_ANALYSIS**' in prompt}")
        else:
            print("- Prompt generation failed or incomplete")
            return False
            
    except Exception as e:
        print(f"- Prompt generation failed: {e}")
        return False
    
    return True

def test_api_call():
    """Test API call functionality"""
    print("\nTesting API call...")
    
    try:
        from enhanced_functionality_analyzer import EnhancedFunctionalityAnalyzer
        analyzer = EnhancedFunctionalityAnalyzer(num_workers=1)
        
        # Simple test prompt
        test_prompt = "Respond with 'API TEST SUCCESSFUL' if you can see this message."
        model_config = analyzer.get_model_config("openai/gpt-4.1")
        
        response = analyzer.call_gpt4(test_prompt, model_config)
        
        if "ERROR:" not in response:
            print("+ API call successful")
            print(f"  Response: {response[:100]}...")
            return True
        else:
            print(f"- API call failed: {response}")
            return False
            
    except Exception as e:
        print(f"- API test failed: {e}")
        return False

def test_single_case_analysis():
    """Test analysis of a single case"""
    print("\nTesting single case analysis...")
    
    try:
        from enhanced_functionality_analyzer import EnhancedFunctionalityAnalyzer
        analyzer = EnhancedFunctionalityAnalyzer(num_workers=1)
        
        # Load a simple case
        test_cases = analyzer.load_test_data_with_context('simple')
        if not test_cases:
            print("- No test cases available for testing")
            return False
        
        # Test first case
        test_case = test_cases[0]
        model_config = analyzer.get_model_config("openai/gpt-4.1")
        
        print(f"Analyzing case: {test_case.get('id')}")
        
        result = analyzer.analyze_single_case(test_case, model_config)
        
        if result and 'verdict' in result:
            print("+ Single case analysis successful")
            print(f"  Case ID: {result['case_id']}")
            print(f"  Verdict: {result['verdict']}")
            print(f"  Mismatch Type: {result['mismatch_type']}")
            return True
        else:
            print("- Single case analysis failed")
            return False
            
    except Exception as e:
        print(f"- Single case analysis failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("="*60)
    print("ENHANCED BFCL ANALYZER - COMPREHENSIVE TEST")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Data Loading Test", test_data_loading),
        ("Prompt Generation Test", test_prompt_generation),
        ("API Call Test", test_api_call),
        ("Single Case Analysis Test", test_single_case_analysis)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-'*40}")
        print(f"Running: {test_name}")
        print(f"{'-'*40}")
        
        try:
            start_time = time.time()
            success = test_func()
            duration = time.time() - start_time
            
            results.append((test_name, success, duration))
            
            if success:
                print(f"+ {test_name} PASSED ({duration:.2f}s)")
            else:
                print(f"- {test_name} FAILED ({duration:.2f}s)")
                
        except Exception as e:
            results.append((test_name, False, 0))
            print(f"- {test_name} ERROR: {e}")
    
    # Print final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, duration in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<30} {status:<6} ({duration:.2f}s)")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: ALL TESTS PASSED - Ready for deployment!")
        return True
    else:
        print("FAILED: SOME TESTS FAILED - Fix issues before deployment")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)