import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class BFCLDataLoader:
    """
    BFCL 벤치마크 결과 및 점수 데이터를 로드하는 클래스
    """
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.result_path = self.base_path / "result" 
        self.score_path = self.base_path / "score"
        
    def load_all_results(self) -> pd.DataFrame:
        """모든 모델의 결과와 점수를 로드해서 통합 DataFrame 생성"""
        
        print("Loading BFCL evaluation data...")
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
                    # 결과 파일 로드 (JSONLines 형식)
                    result_data = []
                    with open(result_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    result_data.append(json.loads(line))
                                except json.JSONDecodeError as e:
                                    print(f"    Warning: Could not parse JSON line in {result_file}: {line[:100]}...")
                    
                    # 점수 파일 찾기
                    score_file = self.score_path / model_name / result_file.name.replace('_result.json', '_score.json')
                    
                    score_data = None
                    if score_file.exists():
                        try:
                            with open(score_file, 'r', encoding='utf-8') as f:
                                score_data = json.load(f)
                        except Exception as e:
                            print(f"    Warning: Could not load score file {score_file}: {e}")
                    
                    # 데이터 변환
                    processed_data = self._process_model_data(
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
        
        return df
    
    def _process_model_data(
        self, 
        model_name: str, 
        test_name: str, 
        result_data: Any, 
        score_data: Optional[Any]
    ) -> List[Dict[str, Any]]:
        """개별 모델 데이터를 처리해서 표준화된 형식으로 변환"""
        
        processed_records = []
        
        # 결과 데이터가 딕셔너리인 경우 (점수 파일)
        if isinstance(result_data, dict) and 'accuracy' in result_data:
            # 점수 파일인 경우 - 요약 정보만 포함
            record = {
                'model_name': model_name,
                'test_category': test_name,
                'accuracy': result_data.get('accuracy', 0),
                'correct_count': result_data.get('correct_count', 0),
                'total_count': result_data.get('total_count', 0),
                'score': result_data.get('accuracy', 0),
                'is_summary': True
            }
            processed_records.append(record)
            
        # 결과 데이터가 리스트인 경우 (개별 테스트 케이스들)
        elif isinstance(result_data, list):
            for item in result_data:
                if isinstance(item, dict):
                    record = self._extract_record_info(model_name, test_name, item, score_data)
                    processed_records.append(record)
                    
        # 결과 데이터가 딕셔너리이지만 개별 테스트 케이스인 경우
        elif isinstance(result_data, dict) and 'id' in result_data:
            record = self._extract_record_info(model_name, test_name, result_data, score_data)
            processed_records.append(record)
            
        return processed_records
    
    def _extract_record_info(
        self, 
        model_name: str, 
        test_name: str, 
        item: Dict[str, Any], 
        score_data: Optional[Any]
    ) -> Dict[str, Any]:
        """개별 테스트 케이스에서 정보 추출"""
        
        # 기본 정보
        record = {
            'model_name': model_name,
            'test_category': test_name,
            'task_id': item.get('id', ''),
            'is_summary': False
        }
        
        # 점수 정보
        if 'valid' in item:
            record['score'] = 1.0 if item['valid'] else 0.0
            record['is_valid'] = item['valid']
        else:
            record['score'] = item.get('score', 0)
            
        # 오류 정보
        record['error'] = item.get('error', [])
        record['error_type'] = item.get('error_type', '')
        record['error_message'] = str(item.get('error', '')) if item.get('error') else ''
        
        # 모델 응답 정보
        record['model_result'] = item.get('model_result', '')
        record['expected_output'] = item.get('expected_output', '')
        record['decoded_result'] = item.get('decoded_result', [])
        
        # 프롬프트 정보
        if 'prompt' in item:
            prompt_info = item['prompt']
            if isinstance(prompt_info, dict):
                record['question'] = str(prompt_info.get('question', ''))
                record['function'] = prompt_info.get('function', [])
                
        # 토큰 및 비용 정보 (있는 경우)
        record['input_token_count'] = item.get('input_token_count', 0)
        record['output_token_count'] = item.get('output_token_count', 0) 
        record['total_cost'] = item.get('total_cost', 0)
        record['latency'] = item.get('latency', 0)  # 실제 BFCL이 사용하는 컬럼명
        record['execution_time'] = item.get('latency', item.get('execution_time', 0))  # 호환성
        
        # HTTP 오류 코드 추출
        if record['error_message']:
            # 일반적인 HTTP 오류 코드 패턴 매칭
            import re
            error_code_match = re.search(r'\b(400|401|403|404|429|500|502|503|504)\b', record['error_message'])
            if error_code_match:
                record['error_code'] = int(error_code_match.group(1))
            else:
                record['error_code'] = None
        else:
            record['error_code'] = None
            
        return record
    
    def load_summary_data(self) -> pd.DataFrame:
        """CSV 요약 데이터 로드 (있는 경우)"""
        
        csv_files = {
            'overall': self.score_path / 'data_overall.csv',
            'live': self.score_path / 'data_live.csv', 
            'non_live': self.score_path / 'data_non_live.csv',
            'multi_turn': self.score_path / 'data_multi_turn.csv'
        }
        
        summary_data = {}
        
        for data_type, file_path in csv_files.items():
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    summary_data[data_type] = df
                    print(f"OK: Loaded {data_type} summary: {len(df)} models")
                except Exception as e:
                    print(f"WARNING: Could not load {file_path}: {e}")
                    
        return summary_data
    
    def validate_data_structure(self) -> bool:
        """데이터 구조 유효성 검증"""
        
        print("[검색] Validating data structure...")
        
        if not self.base_path.exists():
            print(f"ERROR: Base path does not exist: {self.base_path}")
            return False
            
        if not self.result_path.exists():
            print(f"ERROR: Result path does not exist: {self.result_path}")
            return False
            
        if not self.score_path.exists():
            print(f"ERROR: Score path does not exist: {self.score_path}")
            return False
            
        # 최소한 하나의 모델 디렉토리가 있는지 확인
        model_dirs = [d for d in self.result_path.iterdir() if d.is_dir()]
        if not model_dirs:
            print("ERROR: No model directories found in result path")
            return False
            
        print(f"OK: Found {len(model_dirs)} model directories")
        
        # 각 모델 디렉토리에 최소한 하나의 JSON 파일이 있는지 확인
        valid_models = 0
        for model_dir in model_dirs:
            json_files = list(model_dir.glob("*.json"))
            if json_files:
                valid_models += 1
                
        print(f"OK: {valid_models} models have JSON result files")
        
        if valid_models == 0:
            print("ERROR: No models with valid JSON files found")
            return False
            
        return True

def load_bfcl_results(base_path: str = None) -> pd.DataFrame:
    """BFCL 결과를 로드하는 편의 함수"""
    
    if base_path is None:
        base_path = r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard"
    
    loader = BFCLDataLoader(base_path)
    
    # 데이터 구조 검증
    if not loader.validate_data_structure():
        raise ValueError("Data structure validation failed")
    
    # 데이터 로드
    df = loader.load_all_results()
    
    return df