#!/usr/bin/env python3
"""
BFCL Claude Result Fixer
Claude의 multi-turn 결과를 자동으로 수정하여 BFCL 평가가 정상적으로 되도록 함
"""

import json
import os
from claude_format_converter import ClaudeFormatConverter
from typing import Dict, List, Any

class BFCLClaudeFixer:
    """BFCL Claude 결과 파일 수정기"""
    
    def __init__(self):
        self.converter = ClaudeFormatConverter()
        self.stats = {
            'processed_files': 0,
            'processed_items': 0,
            'converted_responses': 0,
            'errors': 0
        }
    
    def fix_result_file(self, input_file: str, output_file: str = None) -> Dict[str, Any]:
        """단일 결과 파일 수정 (JSONL 형식 지원)"""
        if output_file is None:
            output_file = input_file.replace('.json', '_fixed.json')
        
        print(f"Processing: {input_file}")
        
        try:
            # JSONL 형식 파일 읽기 (각 줄이 하나의 JSON 객체)
            data = []
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            data.append(item)
                        except json.JSONDecodeError:
                            continue
            
            fixed_data = []
            
            for item in data:
                if 'result' not in item:
                    fixed_data.append(item)
                    continue
                
                self.stats['processed_items'] += 1
                
                # result를 고정
                original_result = item['result']
                fixed_result = self._fix_result_turns(original_result)
                
                # 고정된 아이템 생성
                fixed_item = item.copy()
                fixed_item['result'] = fixed_result
                fixed_data.append(fixed_item)
            
            # 고정된 결과 저장 (JSONL 형식으로)
            with open(output_file, 'w', encoding='utf-8') as f:
                for item in fixed_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            self.stats['processed_files'] += 1
            print(f"Fixed file saved: {output_file}")
            
            return {
                'input_file': input_file,
                'output_file': output_file,
                'items_processed': len(data),
                'success': True
            }
            
        except Exception as e:
            self.stats['errors'] += 1
            print(f"Error processing {input_file}: {e}")
            return {
                'input_file': input_file,
                'success': False,
                'error': str(e)
            }
    
    def _fix_result_turns(self, result_turns: List[List[str]]) -> List[List[Any]]:
        """Multi-turn result 수정"""
        fixed_turns = []
        
        for turn in result_turns:
            fixed_turn = []
            for response in turn:
                if isinstance(response, str):
                    # 문자열 응답을 function call로 변환
                    converted = self.converter.convert_response(response)
                    if converted:  # 변환에 성공한 경우
                        fixed_turn.append(converted)
                        self.stats['converted_responses'] += 1
                    else:
                        # 변환 실패시 원본 유지
                        fixed_turn.append([response])
                else:
                    # 이미 올바른 형식이면 유지
                    fixed_turn.append(response)
            
            fixed_turns.append(fixed_turn)
        
        return fixed_turns
    
    def fix_directory(self, input_dir: str, output_dir: str = None, file_pattern: str = "*multi_turn*.json"):
        """디렉토리의 모든 multi-turn 파일 수정"""
        if output_dir is None:
            output_dir = os.path.join(input_dir, "fixed")
        
        os.makedirs(output_dir, exist_ok=True)
        
        import glob
        pattern_path = os.path.join(input_dir, file_pattern)
        files = glob.glob(pattern_path)
        
        print(f"Found {len(files)} files matching pattern: {file_pattern}")
        
        results = []
        for file_path in files:
            filename = os.path.basename(file_path)
            output_path = os.path.join(output_dir, filename.replace('.json', '_fixed.json'))
            result = self.fix_result_file(file_path, output_path)
            results.append(result)
        
        return results
    
    def print_stats(self):
        """통계 출력"""
        print("\n" + "="*50)
        print("BFCL Claude Fixer Statistics")
        print("="*50)
        print(f"Processed files: {self.stats['processed_files']}")
        print(f"Processed items: {self.stats['processed_items']}")
        print(f"Converted responses: {self.stats['converted_responses']}")
        print(f"Errors: {self.stats['errors']}")
        print("="*50)

def main():
    """메인 함수"""
    fixer = BFCLClaudeFixer()
    
    # Claude thinking-off 결과 수정
    claude_off_dir = "E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-off"
    
    print("Fixing Claude 4 Sonnet (Thinking Off) multi-turn results...")
    results_off = fixer.fix_directory(claude_off_dir)
    
    # Claude thinking-on 결과 수정
    claude_on_dir = "E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-on-10k"
    
    print("\nFixing Claude 4 Sonnet (Thinking On) multi-turn results...")
    results_on = fixer.fix_directory(claude_on_dir)
    
    # 통계 출력
    fixer.print_stats()
    
    # 결과 요약
    print("\nFiles processed:")
    for result in results_off + results_on:
        if result['success']:
            print(f"OK {result['input_file']} -> {result['output_file']}")
        else:
            print(f"ERROR {result['input_file']} - Error: {result['error']}")

if __name__ == "__main__":
    main()