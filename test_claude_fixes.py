#!/usr/bin/env python3
"""
BFCL Claude 수정사항 테스트 스크립트
"""

import sys
import os
import json

# BFCL 모듈을 import하기 위한 경로 설정
sys.path.append('gorilla/berkeley-function-call-leaderboard')

from bfcl_eval.model_handler.api_inference.claude import ClaudeHandler
from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_parse, type_checker

def test_claude_response_format_parsing():
    """Response Format Discrimination 개선 테스트"""
    print("=== Response Format Discrimination 테스트 ===")
    
    # ClaudeHandler 인스턴스 생성
    handler = ClaudeHandler("claude-4-sonnet-thinking-off", 0.0)
    
    # 테스트 케이스들
    test_cases = [
        {
            "name": "Claude 자연어 + [function()] 혼합 형식",
            "response": "I'll help you with that. [get_weather(location='New York', unit='celsius')]",
            "expected": [{"get_weather": {"location": "New York", "unit": "celsius"}}]
        },
        {
            "name": "Claude 여러 함수 호출",
            "response": "Let me check both. [get_weather(location='Tokyo')] and [get_time(timezone='Asia/Tokyo')]",
            "expected": [
                {"get_weather": {"location": "Tokyo"}},
                {"get_time": {"timezone": "Asia/Tokyo"}}
            ]
        },
        {
            "name": "기존 형식 (호환성 테스트)",
            "response": "[calculate(a=5, b=3)]",
            "expected": [{"calculate": {"a": 5, "b": 3}}]
        }
    ]
    
    success_count = 0
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"Input: {case['response']}")
        
        try:
            result = handler.decode_ast(case['response'])
            print(f"Output: {result}")
            
            if result == case['expected']:
                print("✅ PASS")
                success_count += 1
            else:
                print(f"❌ FAIL - Expected: {case['expected']}")
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\nResponse Format 테스트 결과: {success_count}/{len(test_cases)} 성공")
    return success_count == len(test_cases)

def test_boolean_parameter_strictness():
    """Boolean Parameter Strictness 개선 테스트"""
    print("\n=== Boolean Parameter Strictness 테스트 ===")
    
    # 테스트 케이스들 - boolean 타입 변환
    test_cases = [
        {
            "name": "문자열 'true' → boolean True",
            "value": "true",
            "expected_type": "boolean",
            "should_pass": True
        },
        {
            "name": "문자열 'false' → boolean False",
            "value": "false", 
            "expected_type": "boolean",
            "should_pass": True
        },
        {
            "name": "문자열 'True' (대소문자 무관)",
            "value": "True",
            "expected_type": "boolean", 
            "should_pass": True
        },
        {
            "name": "숫자 1 → boolean True",
            "value": 1,
            "expected_type": "boolean",
            "should_pass": True
        },
        {
            "name": "숫자 0 → boolean False", 
            "value": 0,
            "expected_type": "boolean",
            "should_pass": True
        }
    ]
    
    success_count = 0
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"Value: {case['value']} (type: {type(case['value']).__name__})")
        
        try:
            # type_checker 호출 (간소화된 테스트)
            result = type_checker(
                param_name="test_param",
                expected_type_description=case['expected_type'],
                value=case['value'],
                possible_answer=[case['value']]  # 단순화
            )
            
            success = result['valid'] == case['should_pass']
            if success:
                print("✅ PASS")
                success_count += 1
            else:
                print(f"❌ FAIL - Expected: {case['should_pass']}, Got: {result['valid']}")
                if result.get('error'):
                    print(f"Error: {result['error']}")
                    
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\nBoolean Parameter 테스트 결과: {success_count}/{len(test_cases)} 성공")
    return success_count == len(test_cases)

def test_type_system_inconsistency():
    """Type System Inconsistency 개선 테스트"""
    print("\n=== Type System Inconsistency 테스트 ===")
    
    test_cases = [
        {
            "name": "문자열 '123' → integer 123",
            "value": "123",
            "expected_type": "integer",
            "should_pass": True
        },
        {
            "name": "문자열 '123.45' → float 123.45",
            "value": "123.45",
            "expected_type": "float", 
            "should_pass": True
        },
        {
            "name": "integer 123 → string '123'",
            "value": 123,
            "expected_type": "string",
            "should_pass": True
        },
        {
            "name": "float 123.0 → integer 123", 
            "value": 123.0,
            "expected_type": "integer",
            "should_pass": True
        }
    ]
    
    success_count = 0
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"Value: {case['value']} (type: {type(case['value']).__name__})")
        
        try:
            result = type_checker(
                param_name="test_param",
                expected_type_description=case['expected_type'],
                value=case['value'],
                possible_answer=[case['value']]
            )
            
            success = result['valid'] == case['should_pass']
            if success:
                print("✅ PASS")
                success_count += 1
            else:
                print(f"❌ FAIL - Expected: {case['should_pass']}, Got: {result['valid']}")
                if result.get('error'):
                    print(f"Error: {result['error']}")
                    
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\nType System 테스트 결과: {success_count}/{len(test_cases)} 성공")
    return success_count == len(test_cases)

def main():
    """메인 테스트 함수"""
    print("BFCL Claude 개선사항 종합 테스트 시작")
    print("=" * 50)
    
    results = []
    
    # 각 개선사항 테스트
    results.append(test_claude_response_format_parsing())
    results.append(test_boolean_parameter_strictness())  
    results.append(test_type_system_inconsistency())
    
    # 최종 결과
    total_passed = sum(results)
    total_tests = len(results)
    
    print("\n" + "=" * 50)
    print("최종 테스트 결과")
    print(f"통과한 테스트 그룹: {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        print("🎉 모든 개선사항이 정상적으로 작동합니다!")
        return True
    else:
        print("⚠️ 일부 개선사항에 문제가 있습니다.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)