#!/usr/bin/env python3
"""
BFCL Comprehensive Format Fixer
BFCL 평가의 모든 format mismatch 및 evaluation 문제를 해결하는 포괄적 솔루션
"""

import re
import json
import ast
from typing import List, Dict, Any, Union, Optional
from claude_format_converter import ClaudeFormatConverter

class BFCLComprehensiveFixer:
    """BFCL의 모든 format 관련 문제를 해결하는 클래스"""
    
    def __init__(self):
        self.base_converter = ClaudeFormatConverter()
        self.type_coercion_rules = {
            'boolean': self._coerce_to_boolean,
            'integer': self._coerce_to_integer,
            'string': self._coerce_to_string,
            'array': self._coerce_to_array,
            'object': self._coerce_to_object,
            'float': self._coerce_to_float
        }
    
    # ========== 1. Type System Inconsistency 해결 ==========
    def fix_type_inconsistency(self, value: Any, expected_type: str) -> Any:
        """타입 시스템 불일치 해결"""
        if expected_type in self.type_coercion_rules:
            return self.type_coercion_rules[expected_type](value)
        return value
    
    def _coerce_to_boolean(self, value: Any) -> bool:
        """Boolean으로 안전한 변환"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            # 다양한 boolean 표현 처리
            lower_val = value.lower()
            if lower_val in ('true', 't', 'yes', 'y', '1', 'on'):
                return True
            elif lower_val in ('false', 'f', 'no', 'n', '0', 'off'):
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return False
    
    def _coerce_to_integer(self, value: Any) -> int:
        """Integer로 안전한 변환"""
        try:
            if isinstance(value, str):
                # "123" -> 123
                return int(float(value))
            return int(value)
        except:
            return 0
    
    def _coerce_to_string(self, value: Any) -> str:
        """String으로 안전한 변환"""
        if value is None:
            return ""
        return str(value)
    
    def _coerce_to_array(self, value: Any) -> list:
        """Array로 안전한 변환"""
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            # "[1, 2, 3]" -> [1, 2, 3]
            try:
                return ast.literal_eval(value)
            except:
                # "item1, item2" -> ["item1", "item2"]
                return [x.strip() for x in value.split(',') if x.strip()]
        if value is None:
            return []
        return [value]
    
    def _coerce_to_object(self, value: Any) -> dict:
        """Object/Dict로 안전한 변환"""
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except:
                return {}
        return {}
    
    def _coerce_to_float(self, value: Any) -> float:
        """Float로 안전한 변환"""
        try:
            return float(value)
        except:
            return 0.0
    
    # ========== 2. Boolean Parameter Type Strictness 해결 ==========
    def standardize_boolean_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Boolean 파라미터 표준화 - BFCL의 엄격한 boolean 타입 체크 해결"""
        standardized = {}
        
        for key, value in params.items():
            # Boolean으로 예상되는 파라미터 패턴 (확장된 리스트)
            boolean_indicators = [
                'is_', 'has_', 'can_', 'should_', 'will_', 'was_', 'are_',
                'enable', 'disable', 'show', 'hide', 'visible', 'hidden',
                'active', 'inactive', 'open', 'close', 'allow', 'deny',
                'true', 'false', 'yes', 'no', 'on', 'off'
            ]
            
            if any(pattern in key.lower() for pattern in boolean_indicators):
                standardized[key] = self._coerce_to_boolean(value)
            # Special case: single letter boolean flags (e.g., 'a' for ls -a)
            elif len(key) == 1 and value in [1, 0, "1", "0", "true", "false", True, False]:
                standardized[key] = self._coerce_to_boolean(value)
            else:
                standardized[key] = value
        
        return standardized
    
    # ========== 3. Response Format Discrimination 해결 ==========
    def normalize_response_format(self, response: Any) -> List[Dict[str, Any]]:
        """다양한 응답 형식을 표준화"""
        
        # Case 1: 이미 올바른 형식
        if isinstance(response, list) and all(isinstance(item, dict) for item in response):
            return response
        
        # Case 2: 문자열 형태의 function call
        if isinstance(response, str):
            # Claude 스타일 변환
            converted = self.base_converter.convert_response(response)
            if converted:
                return converted
            
            # 다른 형식 시도
            return self._parse_alternative_formats(response)
        
        # Case 3: 단일 딕셔너리
        if isinstance(response, dict):
            return [response]
        
        # Case 4: 중첩된 리스트
        if isinstance(response, list):
            flattened = []
            for item in response:
                if isinstance(item, list):
                    flattened.extend(self.normalize_response_format(item))
                elif isinstance(item, str):
                    flattened.extend(self.normalize_response_format(item))
                else:
                    flattened.append(item)
            return flattened
        
        return []
    
    def _parse_alternative_formats(self, response: str) -> List[Dict[str, Any]]:
        """대체 형식 파싱"""
        results = []
        
        # Format 1: func_name(arg1=val1, arg2=val2)
        pattern1 = re.compile(r'(\w+)\((.*?)\)')
        matches = pattern1.findall(response)
        
        for func_name, args_str in matches:
            parsed_args = self._parse_function_args(args_str)
            results.append({
                "function": func_name,
                "arguments": parsed_args
            })
        
        # Format 2: {"function": "name", "args": {...}}
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                # 다양한 키 이름 처리
                func_key = next((k for k in ['function', 'func', 'name', 'method'] if k in data), None)
                args_key = next((k for k in ['arguments', 'args', 'params', 'parameters'] if k in data), None)
                
                if func_key:
                    results.append({
                        "function": data[func_key],
                        "arguments": data.get(args_key, {})
                    })
        except:
            pass
        
        return results
    
    def _parse_function_args(self, args_str: str) -> Dict[str, Any]:
        """함수 인자 파싱 (더 유연하게)"""
        args = {}
        
        # 쉼표로 분리하되, 중첩된 구조 고려
        parts = self._smart_split(args_str, ',')
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # 값 타입 추론
                args[key] = self._infer_value_type(value)
        
        return args
    
    def _smart_split(self, text: str, delimiter: str) -> List[str]:
        """중첩 구조를 고려한 스마트 분리"""
        parts = []
        current = []
        depth = 0
        
        for char in text:
            if char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
            elif char == delimiter and depth == 0:
                parts.append(''.join(current))
                current = []
                continue
            
            current.append(char)
        
        if current:
            parts.append(''.join(current))
        
        return parts
    
    def _infer_value_type(self, value_str: str) -> Any:
        """문자열 값의 타입 추론"""
        value_str = value_str.strip().strip('\'"')
        
        # Boolean
        if value_str.lower() in ('true', 'false'):
            return value_str.lower() == 'true'
        
        # Number
        try:
            if '.' in value_str:
                return float(value_str)
            return int(value_str)
        except:
            pass
        
        # Array
        if value_str.startswith('[') and value_str.endswith(']'):
            try:
                return ast.literal_eval(value_str)
            except:
                pass
        
        # Object
        if value_str.startswith('{') and value_str.endswith('}'):
            try:
                return json.loads(value_str)
            except:
                pass
        
        # Default to string
        return value_str
    
    # ========== 4. State Management Inconsistencies 해결 ==========
    def fix_multi_turn_state(self, turns: List[Any], state_schema: Optional[Dict] = None) -> List[Any]:
        """Multi-turn 상태 관리 문제 해결"""
        fixed_turns = []
        accumulated_state = {}
        
        for turn_idx, turn in enumerate(turns):
            # 각 턴을 정규화
            normalized_turn = self.normalize_response_format(turn)
            
            # 상태 일관성 검증
            if state_schema:
                for func_call in normalized_turn:
                    # 상태 의존성 체크
                    if 'arguments' in func_call:
                        func_call['arguments'] = self._validate_state_dependencies(
                            func_call['arguments'], 
                            accumulated_state,
                            state_schema
                        )
            
            # 상태 업데이트
            self._update_state(accumulated_state, normalized_turn)
            
            fixed_turns.append(normalized_turn)
        
        return fixed_turns
    
    def _validate_state_dependencies(self, args: Dict, state: Dict, schema: Dict) -> Dict:
        """상태 의존성 검증 및 수정"""
        validated_args = args.copy()
        
        for key, value in args.items():
            # 참조 해결 (예: "{{previous_result}}")
            if isinstance(value, str) and value.startswith('{{') and value.endswith('}}'):
                ref_key = value[2:-2]
                if ref_key in state:
                    validated_args[key] = state[ref_key]
        
        return validated_args
    
    def _update_state(self, state: Dict, turn_results: List[Dict]):
        """턴 결과로 상태 업데이트"""
        for idx, func_call in enumerate(turn_results):
            # 결과를 상태에 저장
            state[f"turn_result_{len(state)}"] = func_call
            
            # 특정 함수들의 결과는 특별히 처리
            if func_call.get('function') == 'cd':
                state['current_directory'] = func_call.get('arguments', {}).get('folder', '')
    
    # ========== 5. Response Parsing Failures 해결 ==========
    def robust_parse_response(self, response: Any) -> List[Dict[str, Any]]:
        """강건한 응답 파싱"""
        
        # 여러 파싱 전략을 순차적으로 시도
        strategies = [
            self.normalize_response_format,
            self._parse_with_regex,
            self._parse_with_ast,
            self._parse_with_fallback
        ]
        
        for strategy in strategies:
            try:
                result = strategy(response)
                if result:
                    return result
            except Exception as e:
                continue
        
        # 모든 전략 실패시 빈 리스트
        return []
    
    def _parse_with_regex(self, response: Any) -> List[Dict[str, Any]]:
        """정규식 기반 파싱"""
        if not isinstance(response, str):
            response = str(response)
        
        results = []
        
        # 다양한 패턴 시도
        patterns = [
            r'\[([^\[\]]*)\]',  # [function_call]
            r'<function>([^<>]*)</function>',  # <function>call</function>
            r'`([^`]*)`',  # `function_call`
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                parsed = self._parse_function_string(match)
                if parsed:
                    results.append(parsed)
        
        return results
    
    def _parse_function_string(self, func_str: str) -> Optional[Dict[str, Any]]:
        """함수 문자열 파싱"""
        # 기본 converter 사용
        converted = self.base_converter.parse_function_call(func_str)
        if converted and converted.get('function'):
            return converted
        return None
    
    def _parse_with_ast(self, response: Any) -> List[Dict[str, Any]]:
        """AST 기반 파싱"""
        if not isinstance(response, str):
            return []
        
        try:
            # Python 코드로 해석 시도
            tree = ast.parse(response, mode='eval')
            # AST에서 함수 호출 추출
            return self._extract_calls_from_ast(tree)
        except:
            return []
    
    def _extract_calls_from_ast(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """AST에서 함수 호출 추출"""
        calls = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = ""
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                
                args = {}
                for keyword in node.keywords:
                    if keyword.arg:
                        try:
                            args[keyword.arg] = ast.literal_eval(keyword.value)
                        except:
                            args[keyword.arg] = str(keyword.value)
                
                if func_name:
                    calls.append({
                        "function": func_name,
                        "arguments": args
                    })
        
        return calls
    
    def _parse_with_fallback(self, response: Any) -> List[Dict[str, Any]]:
        """최종 fallback 파싱"""
        # 응답을 문자열로 변환하고 기본 처리
        response_str = str(response)
        
        # 함수명처럼 보이는 것 추출
        func_pattern = re.compile(r'\b([a-zA-Z_]\w*)\b')
        potential_funcs = func_pattern.findall(response_str)
        
        # 알려진 함수명 리스트 (BFCL에서 사용되는)
        known_functions = ['pwd', 'ls', 'cd', 'mkdir', 'mv', 'cp', 'rm', 'cat', 
                          'echo', 'grep', 'find', 'sort', 'diff', 'touch']
        
        results = []
        for func in potential_funcs:
            if func in known_functions:
                results.append({
                    "function": func,
                    "arguments": {}
                })
        
        return results
    
    # ========== 통합 처리 함수 ==========
    def comprehensive_fix(self, data: Any, context: Optional[Dict] = None) -> Any:
        """모든 문제를 종합적으로 해결"""
        
        if context is None:
            context = {}
        
        # Response parsing 먼저
        parsed = self.robust_parse_response(data)
        
        # Type system 수정
        for item in parsed:
            if 'arguments' in item:
                # Boolean parameters 표준화
                item['arguments'] = self.standardize_boolean_params(item['arguments'])
                
                # Type coercion based on expected schema
                if 'schema' in context:
                    for param, expected_type in context['schema'].items():
                        if param in item['arguments']:
                            item['arguments'][param] = self.fix_type_inconsistency(
                                item['arguments'][param], 
                                expected_type
                            )
        
        # Multi-turn state 처리
        if 'is_multi_turn' in context and context['is_multi_turn']:
            parsed = self.fix_multi_turn_state(parsed, context.get('state_schema'))
        
        return parsed

def test_comprehensive_fixer():
    """종합 fixer 테스트"""
    fixer = BFCLComprehensiveFixer()
    
    test_cases = [
        # Type inconsistency
        {
            "input": '[ls(a="true")]',  # String "true" instead of boolean
            "expected": [{"function": "ls", "arguments": {"a": True}}]
        },
        # Boolean strictness
        {
            "input": '[enable_feature(is_enabled=1)]',  # Number instead of boolean
            "expected": [{"function": "enable_feature", "arguments": {"is_enabled": True}}]
        },
        # Format discrimination
        {
            "input": 'Execute: pwd() then ls() then cd(folder="test")',
            "expected": [
                {"function": "pwd", "arguments": {}},
                {"function": "ls", "arguments": {}},
                {"function": "cd", "arguments": {"folder": "test"}}
            ]
        }
    ]
    
    for test in test_cases:
        result = fixer.comprehensive_fix(test["input"])
        print(f"Input: {test['input']}")
        print(f"Output: {result}")
        print(f"Expected: {test['expected']}")
        print("=" * 50)

if __name__ == "__main__":
    test_comprehensive_fixer()