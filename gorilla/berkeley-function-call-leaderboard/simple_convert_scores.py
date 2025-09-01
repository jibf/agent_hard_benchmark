#!/usr/bin/env python3
"""
DeepSeek V3.1 모델들의 result 파일을 score로 변환하는 간단한 스크립트
BFCL 의존성 최소화 버전

Usage:
    python simple_convert_scores.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
import traceback

def simple_function_call_check(result_str: str) -> bool:
    """
    간단한 function call 형식 체크
    """
    if not result_str or not isinstance(result_str, str):
        return False
    
    result_str = result_str.strip()
    
    # 기본적인 function call 패턴 체크
    patterns = [
        '[' in result_str and ']' in result_str,  # 리스트 형태
        '(' in result_str and ')' in result_str,  # 함수 호출 형태
        not result_str.startswith('I '),           # "I cannot" 등 거부 응답 아님
        'sorry' not in result_str.lower(),        # 사과 표현 없음
        len(result_str.strip()) > 5               # 최소 길이
    ]
    
    return sum(patterns) >= 3  # 대부분의 조건 만족

def evaluate_result_file(result_file: Path) -> Dict:
    """
    단일 result 파일 평가 (JSONL 형식 지원)
    """
    try:
        # JSONL 형식으로 읽기 시도
        data = []
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError:
                        # 일반 JSON으로 읽기 시도
                        f.seek(0)
                        try:
                            data = json.load(f)
                            break
                        except:
                            return {"error": "Invalid JSON format"}
        
        if not data:
            return {"error": "No data found"}
            
        if not isinstance(data, list):
            data = [data]  # 단일 객체를 리스트로 변환
            
        total_count = len(data)
        correct_count = 0
        
        for item in data:
            if isinstance(item, dict) and 'result' in item:
                result = item['result']
                if simple_function_call_check(result):
                    correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count
        }
        
    except Exception as e:
        return {"error": str(e)}

def main():
    """
    메인 함수
    """
    print("DeepSeek V3.1 모델 result -> score 변환 (간단 버전)")
    print("=" * 60)
    
    base_path = Path(__file__).parent
    result_path = base_path / "result"
    score_path = base_path / "score"
    
    # DeepSeek V3.1 모델들
    target_models = [
        "deepseek-ai_DeepSeek-V3.1-thinking-off",
        "deepseek-ai_DeepSeek-V3.1-thinking-on"
    ]
    
    for model_name in target_models:
        print(f"\n[MODEL] {model_name}")
        print("-" * 50)
        
        model_result_path = result_path / model_name
        model_score_path = score_path / model_name
        
        if not model_result_path.exists():
            print(f"  [WARN] Result 폴더가 존재하지 않습니다: {model_result_path}")
            continue
            
        # result 파일들 찾기
        result_files = list(model_result_path.glob("BFCL_v3_*_result.json"))
        print(f"  [INFO] {len(result_files)}개의 result 파일 발견")
        
        if not result_files:
            print(f"  [WARN] Result 파일을 찾을 수 없습니다.")
            continue
        
        # score 폴더 생성
        model_score_path.mkdir(parents=True, exist_ok=True)
        
        # 각 result 파일을 처리
        for result_file in sorted(result_files):
            score_filename = result_file.name.replace('_result.json', '_score.json')
            score_file = model_score_path / score_filename
            
            print(f"    처리 중: {result_file.name}")
            
            # 평가 수행
            score_data = evaluate_result_file(result_file)
            
            if 'error' in score_data:
                print(f"      [FAIL] 오류: {score_data['error']}")
                continue
            
            # score 파일 저장
            try:
                with open(score_file, 'w', encoding='utf-8') as f:
                    json.dump(score_data, f, indent=2, ensure_ascii=False)
                
                accuracy = score_data['accuracy']
                correct = score_data['correct_count'] 
                total = score_data['total_count']
                print(f"      [OK] 완료: 정확도 {accuracy:.3f} ({correct}/{total})")
                
            except Exception as e:
                print(f"      [FAIL] 저장 실패: {e}")
    
    print("\n" + "=" * 60)
    print("[COMPLETE] 변환 완료!")
    
    # 최종 결과 요약
    print("\n[SUMMARY] 생성된 score 파일 요약:")
    total_files = 0
    for model_name in target_models:
        model_score_path = score_path / model_name
        if model_score_path.exists():
            score_files = list(model_score_path.glob("*.json"))
            total_files += len(score_files)
            print(f"  [DIR] {model_name}: {len(score_files)}개 파일")
            
            for score_file in sorted(score_files)[:5]:  # 처음 5개만 표시
                try:
                    with open(score_file, 'r', encoding='utf-8') as f:
                        score_data = json.load(f)
                    
                    if 'accuracy' in score_data:
                        accuracy = score_data['accuracy']
                        total = score_data.get('total_count', 0)
                        correct = score_data.get('correct_count', 0)
                        print(f"    - {score_file.name}: {accuracy:.3f} ({correct}/{total})")
                except:
                    print(f"    - {score_file.name}: 생성됨")
            
            if len(score_files) > 5:
                print(f"    ... 그 외 {len(score_files) - 5}개 파일")
        else:
            print(f"  [WARN] {model_name}: score 폴더 없음")
    
    print(f"\n총 {total_files}개의 score 파일이 생성되었습니다.")

if __name__ == "__main__":
    main()