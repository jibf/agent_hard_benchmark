"""
CORRECTED BFCL Data Loader - Fixed to match actual JSON structure
실제 JSON 구조에 맞춘 완전히 수정된 데이터 로더
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class CorrectedBFCLDataLoader:
    """
    실제 BFCL JSON 구조에 맞춘 데이터 로더
    """
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.result_path = self.base_path / "result" 
        self.score_path = self.base_path / "score"
        
    def load_all_results(self) -> pd.DataFrame:
        """모든 모델의 결과와 점수를 로드해서 통합 DataFrame 생성"""
        
        print("Loading BFCL data with CORRECTED structure mapping...")
        all_data = []
        
        # 결과 디렉토리의 모든 모델 폴더 탐색
        model_dirs = [d for d in self.result_path.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            model_name = model_dir.name
            print(f"  Processing {model_name}...")
            
            # 각 모델의 모든 결과 파일 처리
            result_files = list(model_dir.glob("*.json"))
            
            for result_file in result_files:
                try:
                    # 점수 파일 찾기 (먼저 로드)
                    score_file = self.score_path / model_name / result_file.name.replace('_result.json', '_score.json')
                    
                    score_data = None
                    if score_file.exists():
                        try:
                            with open(score_file, 'r', encoding='utf-8') as f:
                                # Score 파일은 첫 번째 줄이 summary이고 나머지는 개별 케이스들
                                first_line = f.readline().strip()
                                if first_line:
                                    score_data = json.loads(first_line)  # 첫 번째 줄만 사용 (summary)
                        except Exception as e:
                            print(f"    Warning: Could not load score file {score_file}: {e}")
                    
                    # 결과 파일 로드 (JSONLines 형식)
                    result_data = []
                    with open(result_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    result_data.append(json.loads(line))
                                except json.JSONDecodeError as e:
                                    print(f"    Warning: Could not parse JSON line in {result_file}: {line[:50]}...")
                    
                    # 데이터 변환 (수정된 매핑 사용)
                    processed_data = self._process_model_data_corrected(
                        model_name, result_file.stem, result_data, score_data
                    )
                    all_data.extend(processed_data)
                    
                except Exception as e:
                    print(f"    Error processing {result_file}: {e}")
                    continue
        
        if not all_data:
            raise ValueError("No data could be loaded. Please check the file paths and structure.")
            
        # DataFrame으로 변환
        df = pd.DataFrame(all_data)
        print(f"SUCCESS: Loaded {len(df):,} evaluation records from {len(model_dirs)} models")
        print(f"Actual columns: {df.columns.tolist()}")
        
        return df
    
    def _process_model_data_corrected(
        self, 
        model_name: str, 
        test_name: str, 
        result_data: List[Dict], 
        score_data: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """실제 JSON 구조에 맞춘 데이터 처리"""
        
        processed_records = []
        
        # Score 데이터가 요약 정보인 경우 (accuracy, correct_count, total_count)
        if isinstance(score_data, dict) and 'accuracy' in score_data:
            # 요약 정보 레코드 생성
            record = {
                'model_name': model_name,
                'test_category': test_name,
                'task_id': f'{test_name}_summary',
                'is_summary': True,
                'accuracy': score_data.get('accuracy', 0),
                'correct_count': score_data.get('correct_count', 0),
                'total_count': score_data.get('total_count', 0),
                'score': score_data.get('accuracy', 0),  # accuracy -> score 매핑
                # 토큰 정보는 개별 케이스에서 집계
                'input_token_count': 0,
                'output_token_count': 0,
                'latency': 0,
            }
            processed_records.append(record)
        
        # 개별 결과 데이터 처리
        if isinstance(result_data, list):
            for item in result_data:
                if isinstance(item, dict):
                    record = self._extract_corrected_record_info(
                        model_name, test_name, item, score_data
                    )
                    processed_records.append(record)
                    
        return processed_records
    
    def _extract_corrected_record_info(
        self, 
        model_name: str, 
        test_name: str, 
        item: Dict[str, Any], 
        score_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """실제 JSON 구조에 맞춘 정보 추출"""
        
        # 기본 정보 (실제 필드 매핑)
        record = {
            'model_name': model_name,
            'test_category': test_name,
            'task_id': item.get('id', ''),           # ✅ 올바른 매핑
            'is_summary': False
        }
        
        # 모델 응답 정보 (실제 필드 매핑)
        record['model_result'] = item.get('result', '')  # ✅ result -> model_result
        
        # 안전한 숫자 변환 함수
        def safe_num_convert(val):
            try:
                if isinstance(val, (int, float)):
                    return float(val)
                elif isinstance(val, list) and len(val) > 0:
                    return float(val[0])
                elif val is not None:
                    return float(val)
                else:
                    return 0.0
            except:
                return 0.0
        
        # 토큰 및 레이턴시 정보 (실제 필드 매핑 + 안전한 변환)
        record['input_token_count'] = safe_num_convert(item.get('input_token_count', 0))
        record['output_token_count'] = safe_num_convert(item.get('output_token_count', 0))
        record['latency'] = safe_num_convert(item.get('latency', 0))
        record['execution_time'] = record['latency']  # 호환성
        
        # 점수 정보 (score JSON에서 가져오거나 추정)
        if score_data and isinstance(score_data, dict) and 'accuracy' in score_data:
            # 전체 정확도를 개별 케이스 점수로 사용 (근사치)
            record['score'] = score_data.get('accuracy', 0)
            record['accuracy'] = score_data.get('accuracy', 0)
        else:
            # score 파일이 없는 경우 성공/실패를 토큰으로 판단
            has_tokens = (
                record['input_token_count'] > 0 and 
                record['output_token_count'] > 0 and
                record['latency'] > 0.1
            )
            record['score'] = 1.0 if has_tokens else 0.0
            record['accuracy'] = record['score']
        
        # 기술적 실행 성공/실패 판단 (토큰 생성 여부)
        record['is_successful'] = (
            record['input_token_count'] > 0 and
            record['output_token_count'] > 0 and
            record['latency'] > 0.1
        )
        
        # 성능/점수 기반 성공 판단 (점수 >= 0.8을 성공으로 간주)
        record['is_performance_successful'] = record['score'] >= 0.8
        
        # 기타 필드 (실제로는 존재하지 않지만 호환성을 위해)
        record['error'] = [] if record['is_successful'] else ['execution_failed']
        record['error_message'] = '' if record['is_successful'] else 'Execution failed - no tokens generated'
        record['error_type'] = '' if record['is_successful'] else 'execution_failure'
        record['error_code'] = None
        
        # 추가 정보 (실제 JSON에 없는 필드들은 빈 값)
        record['expected_output'] = ''    # 별도 파일에 있음
        record['decoded_result'] = []     # result가 이미 디코딩됨
        record['question'] = ''           # 별도 파일에 있음  
        record['function'] = []           # 별도 파일에 있음
        record['total_cost'] = 0          # 없음
        
        return record

def load_corrected_bfcl_results(base_path: str = None) -> pd.DataFrame:
    """수정된 BFCL 결과 로더"""
    
    if base_path is None:
        base_path = r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard"
    
    loader = CorrectedBFCLDataLoader(base_path)
    
    # 데이터 구조 검증
    if not loader.result_path.exists():
        raise ValueError(f"Result path does not exist: {loader.result_path}")
    if not loader.score_path.exists():
        raise ValueError(f"Score path does not exist: {loader.score_path}")
    
    # 데이터 로드
    df = loader.load_all_results()
    
    return df

if __name__ == "__main__":
    # 테스트 실행
    df = load_corrected_bfcl_results()
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Sample data:")
    print(df.head())