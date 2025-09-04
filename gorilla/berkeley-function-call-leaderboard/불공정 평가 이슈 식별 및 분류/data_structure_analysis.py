"""
BFCL Data Structure Analysis - Identify All Issues
실제 JSON 구조와 우리 코드의 불일치점 분석
"""

import json
import os
from pathlib import Path

def analyze_data_structure_issues():
    """실제 데이터 구조 vs 코드 불일치점 분석"""
    
    print("="*80)
    print("BFCL DATA STRUCTURE ANALYSIS - CRITICAL ISSUES FOUND")
    print("="*80)
    
    # 1. 실제 Result JSON 구조 확인
    print("\n1. ACTUAL RESULT JSON STRUCTURE:")
    sample_result = 'E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/result/anthropic_claude-4-sonnet-thinking-off/BFCL_v3_simple_result.json'
    
    if os.path.exists(sample_result):
        with open(sample_result, 'r') as f:
            actual_result = json.loads(f.readline().strip())
        
        print(f"   Real keys: {sorted(actual_result.keys())}")
        for key, value in actual_result.items():
            print(f"   {key}: {type(value).__name__} = {str(value)[:60]}...")
    
    # 2. 실제 Score JSON 구조 확인  
    print("\n2. ACTUAL SCORE JSON STRUCTURE:")
    sample_score = 'E:/Users/김현준/Downloads/agent_hard_benchmark_2/gorilla/berkeley-function-call-leaderboard/score/anthropic_claude-4-sonnet-thinking-off/BFCL_v3_simple_score.json'
    
    if os.path.exists(sample_score):
        with open(sample_score, 'r') as f:
            actual_score = json.loads(f.readline().strip())
        
        print(f"   Real keys: {sorted(actual_score.keys())}")
        for key, value in actual_score.items():
            print(f"   {key}: {type(value).__name__} = {value}")
    
    # 3. 데이터 로더가 찾으려는 필드들 vs 실제
    print("\n3. DATA LOADER ISSUES - 존재하지 않는 필드들:")
    
    # Result JSON에서 찾으려 하지만 존재하지 않는 필드들
    missing_fields = [
        'valid',           # ❌ 존재하지 않음 - 디폴트 0 사용
        'score',           # ❌ score 파일에서 'accuracy'로 가져와야 함  
        'error',           # ❌ 성공한 케이스에는 error 필드 없음
        'error_type',      # ❌ 성공한 케이스에는 error_type 필드 없음
        'error_message',   # ❌ 성공한 케이스에는 error_message 필드 없음
        'model_result',    # ❌ 'result' 필드를 사용해야 함
        'expected_output', # ❌ result JSON에 없음 - 별도 평가 로직
        'decoded_result',  # ❌ result JSON에 없음 - 'result'가 이미 디코딩된 것
        'prompt',          # ❌ result JSON에 없음 - 별도 파일 
        'question',        # ❌ result JSON에 없음 - 별도 파일
        'function',        # ❌ result JSON에 없음 - 별도 파일
        'total_cost',      # ❌ result JSON에 없음 - 디폴트 0 사용
        'execution_time'   # ❌ 'latency'를 사용해야 함
    ]
    
    for field in missing_fields:
        print(f"   ❌ {field}: 존재하지 않음 - 기본값/잘못된 필드 사용 중")
    
    # 4. 올바른 필드 매핑
    print("\n4. CORRECT FIELD MAPPING:")
    correct_mapping = {
        'task_id': 'id',                    # ✅ 올바름
        'model_result': 'result',           # ❌ 잘못된 필드명 사용
        'score': 'accuracy (from score JSON)', # ❌ 잘못된 소스 
        'input_token_count': 'input_token_count',  # ✅ 올바름
        'output_token_count': 'output_token_count', # ✅ 올바름  
        'execution_time': 'latency',        # ❌ 잘못된 필드명 사용
    }
    
    for our_field, real_field in correct_mapping.items():
        if our_field in ['model_result', 'execution_time']:
            print(f"   ❌ {our_field} ← {real_field}")
        else:
            print(f"   ✅ {our_field} ← {real_field}")
    
    # 5. 점수 계산 문제
    print("\n5. SCORE CALCULATION ISSUES:")
    print("   ❌ 우리는 result JSON에서 'score' 필드를 찾으려 함")
    print("   ✅ 실제로는 score JSON의 'accuracy' 필드를 사용해야 함")
    print("   ❌ score JSON과 result JSON을 제대로 연결하지 못함")
    
    # 6. 성공/실패 판단 기준 문제
    print("\n6. SUCCESS/FAILURE DETECTION ISSUES:")
    print("   ❌ 현재: input_token_count=0 and output_token_count=0 을 실패로 봄")
    print("   ✅ 실제: input_token_count>0 and output_token_count>0 and latency>0.1 이면 성공")
    print("   ❌ 'error' 필드로 실패 판단하려 하지만 성공 케이스에는 error 필드 없음")
    
    # 7. 영향 분석
    print("\n7. IMPACT ANALYSIS:")
    print("   🔥 CRITICAL: score 필드를 잘못 가져와서 성능 분석 부정확")
    print("   🔥 CRITICAL: 성공/실패 판단 기준이 틀려서 infrastructure 상태 오판")
    print("   🔥 CRITICAL: model_result 필드 누락으로 실제 모델 응답 분석 불가")
    print("   ⚠️  HIGH: 많은 필드가 기본값으로 채워져서 분석 의미 없음")
    
    return {
        'actual_result_keys': list(actual_result.keys()) if 'actual_result' in locals() else [],
        'actual_score_keys': list(actual_score.keys()) if 'actual_score' in locals() else [],
        'missing_fields': missing_fields,
        'incorrect_mappings': ['model_result', 'score', 'execution_time']
    }

if __name__ == "__main__":
    issues = analyze_data_structure_issues()