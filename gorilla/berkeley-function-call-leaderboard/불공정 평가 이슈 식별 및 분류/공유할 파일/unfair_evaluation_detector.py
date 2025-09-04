import pandas as pd
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class UnfairEvaluationDetector:
    """
    BFCL 벤치마크에서 불공정한 평가 사례를 탐지하고 분류하는 시스템
    
    Priority Levels:
    - P0 (Critical): Technical Errors & API Configuration Bias - 모델 능력과 무관한 문제
    - P1 (High): Parsing Failures & State Management Issues - 평가 시스템 문제
    - P2 (Medium): Infrastructure Dependencies & Format Discrimination - 평가 방법론 문제
    """
    
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df.copy()
        self.issue_categories = {}
        self.detection_results = {}
        self.base_path = Path(r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard")
        
    def detect_technical_errors(self) -> int:
        """
        Priority P0: Technical Errors Counted as Model Failures
        이런 오류들은 절대 모델 능력으로 평가되어서는 안됨
        """
        print("[검색] Detecting Technical Errors...")
        
        conditions = {}
        technical_mask = pd.Series(False, index=self.df.index)
        
        # API Error 400 (Bad Request)
        if 'error_code' in self.df.columns:
            conditions['api_error_400'] = self.df['error_code'] == 400
        else:
            conditions['api_error_400'] = pd.Series(False, index=self.df.index)
            
        # Zero Token Failures (모델이 아예 실행되지 않은 경우)
        if 'input_tokens' in self.df.columns and 'output_tokens' in self.df.columns:
            conditions['zero_token_failure'] = (
                (self.df['input_tokens'] == 0) & (self.df['output_tokens'] == 0)
            )
        else:
            conditions['zero_token_failure'] = pd.Series(False, index=self.df.index)
            
        # Max Tokens Error (특히 Claude 모델에서 발생)
        if 'error_message' in self.df.columns:
            conditions['max_tokens_error'] = self.df['error_message'].str.contains(
                'max_tokens.*required', case=False, na=False
            )
        else:
            conditions['max_tokens_error'] = pd.Series(False, index=self.df.index)
            
        # Execution Time Failures (처리 전 실패)
        if 'execution_time' in self.df.columns:
            conditions['timeout_before_execution'] = self.df['execution_time'] < 0.001
        else:
            conditions['timeout_before_execution'] = pd.Series(False, index=self.df.index)
            
        # Infrastructure Errors
        if 'error_message' in self.df.columns:
            conditions['infrastructure_error'] = self.df['error_message'].str.contains(
                'timeout|connection|rate limit|502|503|500|network|ssl', case=False, na=False
            )
        else:
            conditions['infrastructure_error'] = pd.Series(False, index=self.df.index)
            
        # Systematic Model Failures
        conditions['systematic_model_failure'] = self._detect_systematic_failures()
        
        # 조건들을 결합하고 개별적으로 저장
        for condition_name, mask in conditions.items():
            technical_mask |= mask
            self.df[f'technical_error_{condition_name}'] = mask
            print(f"  - {condition_name}: {mask.sum()} cases")
            
        self.df['is_technical_error'] = technical_mask
        
        total_technical = technical_mask.sum()
        print(f"SUCCESS: Total Technical Errors: {total_technical}")
        return total_technical
    
    def detect_api_configuration_bias(self) -> int:
        """
        Priority P0: API Configuration Bias  
        모델별 API 제약사항의 불공정함
        """
        print("[검색] Detecting API Configuration Bias...")
        
        bias_patterns = {}
        api_bias_mask = pd.Series(False, index=self.df.index)
        
        # Claude-specific max_tokens requirement
        if 'model_name' in self.df.columns and 'error_message' in self.df.columns:
            claude_max_tokens = (
                self.df['model_name'].str.contains('claude', case=False, na=False) & 
                self.df['error_message'].str.contains('max_tokens', case=False, na=False)
            )
            bias_patterns['claude_max_tokens_bias'] = claude_max_tokens
            print(f"  - Claude max_tokens bias: {claude_max_tokens.sum()} cases")
        else:
            bias_patterns['claude_max_tokens_bias'] = pd.Series(False, index=self.df.index)
            
        # Model-specific timeout thresholds
        timeout_bias = self._detect_timeout_bias()
        bias_patterns['timeout_bias'] = timeout_bias
        print(f"  - Timeout bias: {timeout_bias.sum()} cases")
        
        # Different response format expectations per model
        format_bias = self._detect_format_bias()
        bias_patterns['format_bias'] = format_bias
        print(f"  - Format bias: {format_bias.sum()} cases")
        
        # 조건들을 결합
        for pattern_name, mask in bias_patterns.items():
            api_bias_mask |= mask
            self.df[f'api_bias_{pattern_name}'] = mask
            
        self.df['is_api_bias'] = api_bias_mask
        
        total_bias = api_bias_mask.sum()
        print(f"SUCCESS: Total API Configuration Bias: {total_bias}")
        return total_bias
    
    def detect_parsing_failures(self) -> int:
        """
        Priority P1: Response Parsing Failures
        모델이 올바른 답을 했지만 파싱이 실패한 경우
        """
        print("[검색] Detecting Parsing Failures...")
        
        parsing_issues = []
        
        for idx, row in self.df.iterrows():
            # 점수가 0이거나 NaN인 경우만 체크
            if pd.isna(row.get('score', np.nan)) or row.get('score', 0) == 0:
                if self._is_response_semantically_correct(row):
                    parsing_issues.append(idx)
        
        parsing_mask = self.df.index.isin(parsing_issues)
        self.df['is_parsing_failure'] = parsing_mask
        
        total_parsing = parsing_mask.sum()
        print(f"SUCCESS: Total Parsing Failures: {total_parsing}")
        return total_parsing
    
    def detect_state_management_issues(self) -> int:
        """
        Priority P1: State Management Inconsistencies
        Multi-turn 대화에서 상태 손실
        """
        print("[검색] Detecting State Management Issues...")
        
        state_issues = []
        
        # Conversation ID가 있는 경우
        if 'conversation_id' in self.df.columns:
            for conv_id, group in self.df.groupby('conversation_id'):
                if self._has_state_inconsistency(group):
                    state_issues.extend(group.index.tolist())
        else:
            # Multi-turn 관련 데이터에서 상태 문제를 추론
            if 'test_category' in self.df.columns:
                multi_turn_data = self.df[
                    self.df['test_category'].str.contains('multi_turn', case=False, na=False)
                ]
                # 간단한 휴리스틱: multi-turn에서 첫 번째 턴은 성공하고 이후 턴이 실패하는 패턴
                for model in multi_turn_data['model_name'].unique():
                    model_data = multi_turn_data[multi_turn_data['model_name'] == model]
                    # 여기서 더 정교한 상태 관리 문제 탐지 로직 구현 가능
                    
        state_mask = self.df.index.isin(state_issues)
        self.df['is_state_issue'] = state_mask
        
        total_state = state_mask.sum()
        print(f"SUCCESS: Total State Management Issues: {total_state}")
        return total_state
    
    def detect_infrastructure_dependencies(self) -> int:
        """
        Priority P2: Model-Specific Infrastructure Dependencies
        모델별 인프라 의존성 문제
        """
        print("[검색] Detecting Infrastructure Dependencies...")
        
        infra_patterns = {
            'function_calling_format': self._detect_function_format_issues(),
            'tool_use_format': self._detect_tool_format_issues(), 
            'model_specific_prompting': self._detect_prompting_bias()
        }
        
        infra_mask = pd.Series(False, index=self.df.index)
        for pattern_name, mask in infra_patterns.items():
            infra_mask |= mask
            self.df[f'infra_{pattern_name}'] = mask
            print(f"  - {pattern_name}: {mask.sum()} cases")
            
        self.df['is_infrastructure_issue'] = infra_mask
        
        total_infra = infra_mask.sum()
        print(f"SUCCESS: Total Infrastructure Dependencies: {total_infra}")
        return total_infra
    
    def detect_format_discrimination(self) -> int:
        """
        Semantic Level: Response Format Discrimination
        의미적으로는 올바르지만 형식이 다른 경우
        """
        print("[검색] Detecting Format Discrimination...")
        
        # 실제 응답과 예상 출력을 비교하는 로직
        format_mask = self._analyze_format_discrimination()
        self.df['is_format_discrimination'] = format_mask
        
        total_format = format_mask.sum()
        print(f"SUCCESS: Total Format Discrimination: {total_format}")
        return total_format

    def _detect_systematic_failures(self) -> pd.Series:
        """모든 태스크가 동일한 오류로 실패하는 체계적 실패 탐지"""
        systematic_failures = pd.Series(False, index=self.df.index)
        
        if 'model_name' not in self.df.columns:
            return systematic_failures
            
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model]
            
            if len(model_data) > 10:  # 충분한 샘플이 필요
                if 'error_message' in self.df.columns and pd.notna(model_data['error_message']).any():
                    error_counts = model_data['error_message'].value_counts()
                    if len(error_counts) > 0:
                        most_common_error_rate = error_counts.iloc[0] / len(model_data)
                        
                        if most_common_error_rate > 0.9:  # 90% 이상이 같은 오류
                            most_common_error = error_counts.index[0]
                            mask = (
                                (self.df['model_name'] == model) & 
                                (self.df['error_message'] == most_common_error)
                            )
                            systematic_failures |= mask
                            print(f"  - Systematic failure detected in {model}: {mask.sum()} cases")
        
        return systematic_failures

    def _detect_timeout_bias(self) -> pd.Series:
        """모델별 타임아웃 편향 탐지"""
        return pd.Series(False, index=self.df.index)  # 구현 예정

    def _detect_format_bias(self) -> pd.Series:
        """응답 형식 편향 탐지"""
        return pd.Series(False, index=self.df.index)  # 구현 예정

    def _is_response_semantically_correct(self, row: pd.Series) -> bool:
        """응답이 의미적으로 올바른지 확인"""
        if not isinstance(row, pd.Series):
            return False
            
        response = row.get('model_result', '')
        expected = row.get('expected_output', '')
        
        if pd.isna(response) or pd.isna(expected):
            return False
        
        response = str(response).lower().strip()
        expected = str(expected).lower().strip()
        
        # 간단한 의미적 매칭 패턴
        semantic_matches = [
            # 불린 동등성
            (response in ['yes', 'true', '1', 'correct'] and expected in ['yes', 'true', '1', 'correct']),
            (response in ['no', 'false', '0', 'incorrect'] and expected in ['no', 'false', '0', 'incorrect']),
            # 숫자 형식 변형
            self._numbers_equivalent(response, expected),
            # 함수 호출 동등성
            self._function_calls_equivalent(response, expected)
        ]
        
        return any(semantic_matches)

    def _numbers_equivalent(self, resp: str, exp: str) -> bool:
        """숫자 동등성 검사"""
        try:
            return float(resp) == float(exp)
        except (ValueError, TypeError):
            return False

    def _function_calls_equivalent(self, resp: str, exp: str) -> bool:
        """함수 호출 동등성 검사"""
        # JSON 형식의 함수 호출을 파싱해서 비교
        try:
            if resp.startswith('[') and exp.startswith('['):
                resp_parsed = json.loads(resp)
                exp_parsed = json.loads(exp)
                return resp_parsed == exp_parsed
        except (json.JSONDecodeError, TypeError):
            pass
        return False

    def _has_state_inconsistency(self, group: pd.DataFrame) -> bool:
        """상태 불일치 탐지"""
        # 구현 예정
        return False

    def _detect_function_format_issues(self) -> pd.Series:
        """함수 호출 형식 문제 탐지"""
        return pd.Series(False, index=self.df.index)

    def _detect_tool_format_issues(self) -> pd.Series:
        """도구 사용 형식 문제 탐지"""
        return pd.Series(False, index=self.df.index)

    def _detect_prompting_bias(self) -> pd.Series:
        """프롬프팅 편향 탐지"""
        return pd.Series(False, index=self.df.index)

    def _analyze_format_discrimination(self) -> pd.Series:
        """형식 차별 분석"""
        return pd.Series(False, index=self.df.index)

    def _detect_claude_sonnet_case(self) -> pd.Series:
        """Claude-4-Sonnet irrelevance 태스크 특정 문제 탐지"""
        if 'model_name' not in self.df.columns or 'test_category' not in self.df.columns:
            return pd.Series(False, index=self.df.index)
            
        claude_sonnet_pattern = (
            self.df['model_name'].str.contains('claude.*sonnet', case=False, na=False) &
            self.df['test_category'].str.contains('irrelevance', case=False, na=False)
        )
        
        if 'error_message' in self.df.columns:
            claude_sonnet_pattern &= self.df['error_message'].str.contains(
                'max_tokens.*required', case=False, na=False
            )
            
        if 'input_tokens' in self.df.columns and 'output_tokens' in self.df.columns:
            claude_sonnet_pattern &= (
                (self.df['input_tokens'] == 0) &
                (self.df['output_tokens'] == 0)
            )
        
        affected_count = claude_sonnet_pattern.sum()
        if affected_count > 200:  # 많은 태스크에 영향을 미치는 경우
            print(f"CRITICAL: DETECTED CLAUDE-SONNET SYSTEMATIC ISSUE: {affected_count} tasks affected")
        
        return claude_sonnet_pattern

    def classify_all_issues(self) -> Dict[str, int]:
        """모든 탐지 방법을 실행하고 최종 분류 생성"""
        
        print("[검색] Starting Unfair Evaluation Detection...")
        print("=" * 60)
        
        # 모든 탐지기 실행
        results = {
            'technical_errors': self.detect_technical_errors(),
            'api_bias': self.detect_api_configuration_bias(), 
            'parsing_failures': self.detect_parsing_failures(),
            'state_issues': self.detect_state_management_issues(),
            'infrastructure': self.detect_infrastructure_dependencies(),
            'format_discrimination': self.detect_format_discrimination()
        }
        
        print("\n" + "=" * 60)
        
        # 특별 사례 탐지
        claude_sonnet_issues = self._detect_claude_sonnet_case()
        results['claude_sonnet_specific'] = claude_sonnet_issues.sum()
        
        # 최종 분류 생성 (우선순위 순서로, 나중 것이 앞의 것을 덮어씀)
        self.df['issue_classification'] = 'Fair Evaluation'  # 기본값
        
        # 우선순위 순서로 분류 (P2 -> P1 -> P0 순으로, 더 심각한 것이 덮어씀)
        self.df.loc[self.df['is_format_discrimination'], 'issue_classification'] = 'Format Discrimination (P2)'
        self.df.loc[self.df['is_infrastructure_issue'], 'issue_classification'] = 'Infrastructure Dependency (P2)'
        self.df.loc[self.df['is_state_issue'], 'issue_classification'] = 'State Management Issue (P1)'
        self.df.loc[self.df['is_parsing_failure'], 'issue_classification'] = 'Parsing Failure (P1)'
        self.df.loc[self.df['is_api_bias'], 'issue_classification'] = 'API Configuration Bias (P0)'
        self.df.loc[self.df['is_technical_error'], 'issue_classification'] = 'Technical Error (P0)'
        
        # Claude-Sonnet 특별 사례 (가장 높은 우선순위)
        self.df.loc[claude_sonnet_issues, 'issue_classification'] = 'Claude-Sonnet Systematic Issue (P0)'
        
        self.detection_results = results
        return results

    def generate_unfair_evaluation_report(self) -> Dict[str, Any]:
        """불공정 평가에 대한 종합 리포트 생성"""
        
        # 요약 통계
        classification_summary = self.df['issue_classification'].value_counts()
        
        print("\n[분석] UNFAIR EVALUATION CLASSIFICATION SUMMARY")
        print("=" * 60)
        
        total_evaluations = len(self.df)
        fair_evaluations = classification_summary.get('Fair Evaluation', 0)
        unfair_evaluations = total_evaluations - fair_evaluations
        
        print(f"Total Evaluations: {total_evaluations:,}")
        print(f"Fair Evaluations: {fair_evaluations:,} ({(fair_evaluations/total_evaluations)*100:.1f}%)")
        print(f"Unfair Evaluations: {unfair_evaluations:,} ({(unfair_evaluations/total_evaluations)*100:.1f}%)")
        print("\nDETAILED BREAKDOWN:")
        
        for category, count in classification_summary.items():
            percentage = (count / total_evaluations) * 100
            if category != 'Fair Evaluation':
                priority = "P0" if "P0" in category else "P1" if "P1" in category else "P2" if "P2" in category else ""
                print(f"  {category:45}: {count:6,} ({percentage:5.1f}%) {priority}")
        
        # 모델별 세부 분석
        print(f"\n[분석] MODEL-SPECIFIC BREAKDOWN")
        print("=" * 60)
        
        if 'model_name' in self.df.columns:
            model_breakdown = self.df.groupby(['model_name', 'issue_classification']).size().unstack(fill_value=0)
            
            # 불공정 평가 비율이 높은 모델 순으로 정렬
            model_unfair_rates = {}
            for model in model_breakdown.index:
                model_data = self.df[self.df['model_name'] == model]
                unfair_count = len(model_data[model_data['issue_classification'] != 'Fair Evaluation'])
                model_unfair_rates[model] = unfair_count / len(model_data)
            
            sorted_models = sorted(model_unfair_rates.items(), key=lambda x: x[1], reverse=True)
            
            for model, unfair_rate in sorted_models[:10]:  # 상위 10개 모델만 출력
                model_data = self.df[self.df['model_name'] == model]
                total_model_evals = len(model_data)
                unfair_model_evals = len(model_data[model_data['issue_classification'] != 'Fair Evaluation'])
                
                print(f"{model:40}: {unfair_model_evals:4}/{total_model_evals:4} ({unfair_rate*100:5.1f}%) unfair")
                
                # P0 문제가 있는 모델은 특별히 표시
                p0_issues = model_data[model_data['issue_classification'].str.contains('P0', na=False)]
                if len(p0_issues) > 0:
                    print(f"{'':42}WARNING: {len(p0_issues)} CRITICAL P0 issues detected!")
        
        # 리포트 파일 저장
        print(f"\n[저장] SAVING REPORTS...")
        print("=" * 60)
        
        # 1. 전체 분류된 데이터셋
        output_file = self.base_path / 'unfair_evaluation_analysis.csv'
        self.df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"OK: Full analysis: {output_file}")
        
        # 2. 카테고리별 요약
        summary_file = self.base_path / 'issue_classification_summary.csv'
        classification_summary.to_csv(summary_file, encoding='utf-8-sig')
        print(f"OK: Category summary: {summary_file}")
        
        # 3. 모델별 세부 분석
        if 'model_name' in self.df.columns:
            model_file = self.base_path / 'model_issue_breakdown.csv'
            model_breakdown.to_csv(model_file, encoding='utf-8-sig')
            print(f"OK: Model breakdown: {model_file}")
        
        # 4. 우선순위 높은 수정 필요 사항
        priority_issues = self.df[
            self.df['issue_classification'].str.contains('P0', na=False)
        ].copy()
        
        if len(priority_issues) > 0:
            priority_file = self.base_path / 'priority_fixes_required.csv'
            priority_issues.to_csv(priority_file, index=False, encoding='utf-8-sig')
            print(f"OK: Priority fixes (P0): {priority_file}")
            print(f"CRITICAL: {len(priority_issues)} CRITICAL issues require immediate attention!")
        
        # 5. Claude-Sonnet 특별 분석
        claude_issues = self.df[
            self.df['issue_classification'].str.contains('Claude-Sonnet', na=False)
        ].copy()
        
        if len(claude_issues) > 0:
            claude_file = self.base_path / 'claude_sonnet_systematic_issues.csv'
            claude_issues.to_csv(claude_file, index=False, encoding='utf-8-sig')
            print(f"OK: Claude-Sonnet issues: {claude_file}")
            print(f"[검색] {len(claude_issues)} Claude-Sonnet systematic issues identified")
        
        # 6. 상세 탐지 결과 JSON (numpy 타입 변환)
        detection_file = self.base_path / 'detection_results.json'
        
        # numpy 타입을 Python 네이티브 타입으로 변환
        def convert_numpy_types(obj):
            if hasattr(obj, 'dtype'):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        json_data = {
            'summary': {
                'total_evaluations': int(total_evaluations),
                'fair_evaluations': int(fair_evaluations),
                'unfair_evaluations': int(unfair_evaluations),
                'unfair_percentage': float((unfair_evaluations/total_evaluations)*100)
            },
            'detection_results': convert_numpy_types(self.detection_results),
            'classification_counts': convert_numpy_types(classification_summary.to_dict())
        }
        
        with open(detection_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"OK: Detection results JSON: {detection_file}")
        
        return {
            'total_unfair': unfair_evaluations,
            'priority_fixes': len(priority_issues),
            'classification_breakdown': classification_summary.to_dict(),
            'unfair_percentage': (unfair_evaluations/total_evaluations)*100,
            'most_affected_models': sorted_models[:5] if 'model_name' in self.df.columns else []
        }