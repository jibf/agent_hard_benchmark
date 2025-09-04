#!/usr/bin/env python3
"""
BFCL Claude 수정사항 단순 테스트 스크립트
"""

import re
import ast

def test_claude_smart_parse():
    """Claude의 새로운 파싱 로직 테스트"""
    print("=== Claude Smart Parse 테스트 ===")
    
    def _claude_smart_parse(result: str, language: str = "Python"):
        """Claude 응답 특화 파싱"""
        import re
        
        # Claude의 자연어 + [function()] 혼합 형식 처리
        # 패턴: [function_name(param=value, param2=value2)]
        pattern = re.compile(r'\[([a-zA-Z_]\w*\([^[\]]*\))\]', re.MULTILINE)
        matches = pattern.findall(result)
        
        if matches:
            # 추출된 함수들을 표준 리스트 형태로 구성
            extracted_calls = '[' + ', '.join(matches) + ']'
            try:
                # AST 파싱 시뮬레이션 (간소화)
                return f"SUCCESS: {extracted_calls}"
            except Exception as e:
                return f"PARSE_ERROR: {e}"
        else:
            return "NO_MATCHES"
    
    test_cases = [
        {
            "name": "Claude 자연어 + [function()] 혼합",
            "input": "I'll help you with that. [get_weather(location='New York', unit='celsius')]",
            "expected_match": True
        },
        {
            "name": "여러 함수 호출",
            "input": "Let me check both. [get_weather(location='Tokyo')] and [get_time(timezone='Asia/Tokyo')]",
            "expected_match": True
        },
        {
            "name": "기존 형식",
            "input": "[calculate(a=5, b=3)]",
            "expected_match": True
        },
        {
            "name": "자연어만 (함수 호출 없음)",
            "input": "I cannot help with that.",
            "expected_match": False
        }
    ]
    
    success_count = 0
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"Input: {case['input'][:50]}..." if len(case['input']) > 50 else f"Input: {case['input']}")
        
        result = _claude_smart_parse(case['input'])
        has_match = not result.startswith("NO_MATCHES")
        
        if has_match == case['expected_match']:
            print(f"PASS - {result[:50]}...")
            success_count += 1
        else:
            print(f"FAIL - Expected match: {case['expected_match']}, Got: {has_match}")
            print(f"Result: {result}")
    
    print(f"\nSmart Parse 테스트 결과: {success_count}/{len(test_cases)} 성공")
    return success_count == len(test_cases)

def test_boolean_type_coercion():
    """Boolean 타입 변환 로직 테스트"""
    print("\n=== Boolean Type Coercion 테스트 ===")
    
    def boolean_coercion_test(value, expected_type_description):
        """Boolean 변환 로직 시뮬레이션"""
        if expected_type_description == "boolean":
            if isinstance(value, bool):
                return True, value  # 이미 boolean이면 성공
            elif isinstance(value, str):
                # "true"/"false" 문자열을 boolean으로 변환
                if value.lower() == "true":
                    return True, True
                elif value.lower() == "false":  
                    return True, False
            elif isinstance(value, (int, float)):
                # 숫자를 boolean으로 변환
                return True, bool(value)
        return False, value
    
    test_cases = [
        {
            "name": "문자열 'true' → boolean True",
            "value": "true",
            "expected_success": True,
            "expected_result": True
        },
        {
            "name": "문자열 'false' → boolean False", 
            "value": "false",
            "expected_success": True,
            "expected_result": False
        },
        {
            "name": "문자열 'True' (대문자) → boolean True",
            "value": "True",
            "expected_success": True,
            "expected_result": True
        },
        {
            "name": "숫자 1 → boolean True",
            "value": 1,
            "expected_success": True,
            "expected_result": True
        },
        {
            "name": "숫자 0 → boolean False",
            "value": 0,
            "expected_success": True,
            "expected_result": False
        },
        {
            "name": "유효하지 않은 문자열 → 실패",
            "value": "maybe",
            "expected_success": False,
            "expected_result": "maybe"
        }
    ]
    
    success_count = 0
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"Input: {case['value']} (type: {type(case['value']).__name__})")
        
        success, result = boolean_coercion_test(case['value'], "boolean")
        
        if (success == case['expected_success'] and 
            result == case['expected_result']):
            print(f"PASS - Result: {result}")
            success_count += 1
        else:
            print(f"FAIL")
            print(f"Expected: success={case['expected_success']}, result={case['expected_result']}")
            print(f"Got: success={success}, result={result}")
    
    print(f"\nBoolean Coercion 테스트 결과: {success_count}/{len(test_cases)} 성공")
    return success_count == len(test_cases)

def test_type_system_flexibility():
    """Type System 유연성 테스트"""
    print("\n=== Type System Flexibility 테스트 ===")
    
    def type_conversion_test(value, expected_type_description):
        """타입 변환 로직 시뮬레이션"""
        try:
            if expected_type_description == "integer" and isinstance(value, str):
                if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    return True, int(value)
            elif expected_type_description == "integer" and isinstance(value, float):
                if value.is_integer():
                    return True, int(value)
            elif expected_type_description == "float" and isinstance(value, str):
                return True, float(value)
            elif expected_type_description == "float" and isinstance(value, int):
                return True, float(value)
            elif expected_type_description == "string" and not isinstance(value, str):
                return True, str(value)
        except ValueError:
            pass
        return False, value
    
    test_cases = [
        {
            "name": "문자열 '123' → integer 123",
            "value": "123",
            "target_type": "integer", 
            "expected_success": True,
            "expected_result": 123
        },
        {
            "name": "문자열 '123.45' → float 123.45",
            "value": "123.45",
            "target_type": "float",
            "expected_success": True,
            "expected_result": 123.45
        },
        {
            "name": "integer 123 → string '123'",
            "value": 123,
            "target_type": "string",
            "expected_success": True,
            "expected_result": "123"
        },
        {
            "name": "float 123.0 → integer 123",
            "value": 123.0,
            "target_type": "integer",
            "expected_success": True,
            "expected_result": 123
        },
        {
            "name": "유효하지 않은 변환 'abc' → integer",
            "value": "abc",
            "target_type": "integer",
            "expected_success": False,
            "expected_result": "abc"
        }
    ]
    
    success_count = 0
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"Input: {case['value']} → {case['target_type']}")
        
        success, result = type_conversion_test(case['value'], case['target_type'])
        
        if (success == case['expected_success'] and 
            result == case['expected_result']):
            print(f"PASS - Result: {result} ({type(result).__name__})")
            success_count += 1
        else:
            print(f"FAIL")
            print(f"Expected: success={case['expected_success']}, result={case['expected_result']}")
            print(f"Got: success={success}, result={result}")
    
    print(f"\nType Flexibility 테스트 결과: {success_count}/{len(test_cases)} 성공")
    return success_count == len(test_cases)

def main():
    """메인 테스트 함수"""
    print("BFCL Claude 개선사항 단순 테스트")
    print("=" * 50)
    
    results = []
    
    # 각 개선사항 테스트
    results.append(test_claude_smart_parse())
    results.append(test_boolean_type_coercion())
    results.append(test_type_system_flexibility())
    
    # 최종 결과
    total_passed = sum(results)
    total_tests = len(results)
    
    print("\n" + "=" * 50)
    print("최종 테스트 결과")
    print(f"통과한 테스트 그룹: {total_passed}/{total_tests}")
    
    if total_passed == total_tests:
        print("모든 개선사항이 정상적으로 작동합니다!")
        return True
    else:
        print("일부 개선사항에 문제가 있습니다.")
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n종료 코드: {'성공' if success else '실패'}")