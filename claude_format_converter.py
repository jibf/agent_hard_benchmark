#!/usr/bin/env python3
"""
Claude BFCL Format Converter
자동으로 Claude의 자연어 + function call 응답을 BFCL 평가 형식으로 변환
"""

import re
import json
import ast
from typing import List, Dict, Any, Union

class ClaudeFormatConverter:
    """Claude 응답을 BFCL 평가 형식으로 변환하는 클래스"""
    
    def __init__(self):
        # Function call 패턴 매칭을 위한 정규표현식
        self.function_pattern = re.compile(r'\[([^\[\]]+)\]', re.MULTILINE)
    
    def extract_function_calls(self, response_text: str) -> List[str]:
        """텍스트에서 [function()] 형태의 함수 호출 추출"""
        matches = self.function_pattern.findall(response_text)
        return [match.strip() for match in matches if self._is_valid_function_call(match)]
    
    def _is_valid_function_call(self, call: str) -> bool:
        """유효한 function call인지 검증"""
        call = call.strip()
        if not call:
            return False
        
        # 기본 function call 패턴 검증
        if '(' in call and ')' in call:
            return True
        
        return False
    
    def parse_function_call(self, call_str: str) -> Dict[str, Any]:
        """개별 함수 호출을 파싱하여 딕셔너리로 변환"""
        call_str = call_str.strip()
        
        # 함수명과 인자 분리
        if '(' not in call_str:
            return {"function": call_str, "arguments": {}}
        
        func_name = call_str.split('(')[0].strip()
        args_part = call_str[call_str.find('(')+1:call_str.rfind(')')].strip()
        
        # 인자 파싱
        arguments = {}
        if args_part:
            try:
                # Python 문법으로 파싱 시도
                parsed_args = self._parse_python_args(args_part)
                arguments = parsed_args
            except Exception as e:
                # 실패시 키워드 인자만 추출
                arguments = self._parse_keyword_args(args_part)
        
        return {"function": func_name, "arguments": arguments}
    
    def _parse_python_args(self, args_str: str) -> Dict[str, Any]:
        """Python 함수 인자를 파싱"""
        # 간단한 eval 대신 ast.literal_eval 사용
        try:
            # 함수 호출 형태로 만들어서 ast 파싱
            fake_func = f"dummy({args_str})"
            tree = ast.parse(fake_func, mode='eval')
            call_node = tree.body
            
            args_dict = {}
            
            # 키워드 인자 처리
            for keyword in call_node.keywords:
                if keyword.arg:
                    args_dict[keyword.arg] = ast.literal_eval(keyword.value)
            
            # 위치 인자 처리 (간단한 경우만)
            if call_node.args and not args_dict:
                for i, arg in enumerate(call_node.args):
                    args_dict[f"arg_{i}"] = ast.literal_eval(arg)
            
            return args_dict
            
        except Exception:
            # 실패시 키워드 파싱으로 fallback
            return self._parse_keyword_args(args_str)
    
    def _parse_keyword_args(self, args_str: str) -> Dict[str, Any]:
        """키워드 인자만 파싱 (fallback 방법)"""
        arguments = {}
        
        # 간단한 키워드=값 패턴 매칭
        kv_pattern = re.compile(r'(\w+)\s*=\s*([^,]+)')
        matches = kv_pattern.findall(args_str)
        
        for key, value in matches:
            # 값 정제
            value = value.strip().strip('\'"')
            
            # 타입 추론
            if value.isdigit():
                arguments[key] = int(value)
            elif value.lower() in ('true', 'false'):
                arguments[key] = value.lower() == 'true'
            else:
                arguments[key] = value
        
        return arguments
    
    def convert_response(self, claude_response: str) -> List[Dict[str, Any]]:
        """Claude 응답을 BFCL 형식으로 변환"""
        # Function call 추출
        function_calls = self.extract_function_calls(claude_response)
        
        # 각 함수 호출을 딕셔너리로 변환
        converted_calls = []
        for call in function_calls:
            try:
                parsed_call = self.parse_function_call(call)
                converted_calls.append(parsed_call)
            except Exception as e:
                print(f"Error parsing function call '{call}': {e}")
                continue
        
        return converted_calls
    
    def convert_multi_turn_result(self, result_data: List[List[str]]) -> List[List[Dict[str, Any]]]:
        """Multi-turn 결과 전체를 변환"""
        converted_turns = []
        
        for turn in result_data:
            converted_turn = []
            for response in turn:
                converted_response = self.convert_response(response)
                converted_turn.append(converted_response)
            converted_turns.append(converted_turn)
        
        return converted_turns

def test_converter():
    """변환기 테스트"""
    converter = ClaudeFormatConverter()
    
    # 테스트 케이스
    test_response = """I'll help you move the 'final_report.pdf' file to a 'temp' directory within the document directory. First, let me check the current directory structure and then create the temp directory before moving the file.

[pwd()]

[ls()]

[cd(folder="document")]

[ls()]

[mkdir(dir_name="temp")]

[mv(source="final_report.pdf", destination="temp")]"""
    
    print("Original Claude Response:")
    print(test_response)
    print("\n" + "="*50 + "\n")
    
    converted = converter.convert_response(test_response)
    print("Converted Format:")
    print(json.dumps(converted, indent=2))
    
    expected_format = [
        {"function": "pwd", "arguments": {}},
        {"function": "ls", "arguments": {}},
        {"function": "cd", "arguments": {"folder": "document"}},
        {"function": "ls", "arguments": {}},
        {"function": "mkdir", "arguments": {"dir_name": "temp"}},
        {"function": "mv", "arguments": {"source": "final_report.pdf", "destination": "temp"}}
    ]
    
    print("\nExpected Format:")
    print(json.dumps(expected_format, indent=2))
    
    print(f"\nConversion successful: {len(converted) == len(expected_format)}")

if __name__ == "__main__":
    test_converter()