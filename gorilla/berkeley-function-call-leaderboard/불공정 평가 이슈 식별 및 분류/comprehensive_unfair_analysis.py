#!/usr/bin/env python3
"""
포괄적 BFCL 불공정 평가 분석 시스템
- 전체 16개 모델 분석
- Performance Inversion 탐지 (P0)
- 패밀리별 편향 패턴 분석 (P1) 
- 정량적 영향도 분석
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# 로컬 모듈
from data_loader import load_bfcl_results
from unfair_evaluation_detector import UnfairEvaluationDetector

class ComprehensiveUnfairAnalysis:
    """포괄적 불공정 평가 분석 시스템"""
    
    def __init__(self):
        self.base_path = Path(r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard")
        self.df = None
        self.model_families = self._define_model_families()
        self.performance_baseline = self._define_performance_baseline()
        
    def _define_model_families(self):
        """모델 패밀리 정의"""
        return {
            'OpenAI': ['openai_gpt-4.1', 'openai_gpt-4o-20240806', 'openai_gpt-4o-mini', 'openai_o3-high', 'openai_o4-mini-high'],
            'Anthropic': ['anthropic_claude-4-sonnet-thinking-off', 'anthropic_claude-4-sonnet-thinking-on-10k'],
            'DeepSeek': ['deepseek-ai_DeepSeek-R1-0528', 'deepseek-ai_DeepSeek-V3-0324'],
            'Qwen': ['togetherai_Qwen_Qwen3-235B-A22B-FP8', 'togetherai_Qwen_Qwen3-235B-A22B-Instruct-2507-FP8', 
                     'togetherai_Qwen_Qwen3-235B-A22B-Thinking-2507-FP8'],
            'Other': ['togetherai_moonshotai_Kimi-K2-Instruct', 'xai_grok-4']
        }
        
    def _define_performance_baseline(self):
        """성능 기준선 정의 (예상 성능 순위)"""
        return {
            'openai_gpt-4.1': 95,
            'openai_o3-high': 90, 
            'openai_o4-mini-high': 85,
            'openai_gpt-4o-20240806': 80,
            'togetherai_Qwen_Qwen3-235B-A22B-Instruct-2507-FP8': 75,
            'togetherai_Qwen_Qwen3-235B-A22B-FP8': 72,
            'anthropic_claude-4-sonnet-thinking-on-10k': 70,
            'deepseek-ai_DeepSeek-V3-0324': 65,
            'deepseek-ai_DeepSeek-R1-0528': 60,
            'anthropic_claude-4-sonnet-thinking-off': 55,
            'togetherai_Qwen_Qwen3-235B-A22B-Thinking-2507-FP8': 50,
            'openai_gpt-4o-mini': 45,
            'togetherai_moonshotai_Kimi-K2-Instruct': 40,
            'xai_grok-4': 35
        }
    
    def load_and_analyze_all_data(self):
        """전체 데이터 로드 및 기본 분석"""
        print("=" * 80)
        print("COMPREHENSIVE BFCL UNFAIR EVALUATION ANALYSIS")
        print("=" * 80)
        
        # 전체 데이터 로드
        print("\n[1] Loading complete dataset...")
        self.df = load_bfcl_results()
        
        if self.df is None or len(self.df) == 0:
            raise ValueError("No data could be loaded")
            
        print(f"SUCCESS: Loaded {len(self.df):,} evaluation records")
        
        # 기본 통계
        self._basic_statistics()
        
        return self.df
    
    def _basic_statistics(self):
        """기본 통계 분석"""
        print(f"\n[BASIC STATISTICS]")
        print(f"Total Records: {len(self.df):,}")
        
        if 'model_name' in self.df.columns:
            model_counts = self.df['model_name'].value_counts()
            print(f"Models: {len(model_counts)}")
            
            print(f"\nModel Distribution:")
            for model, count in model_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"  {model:45}: {count:6,} ({percentage:5.1f}%)")
                
        if 'test_category' in self.df.columns:
            category_counts = self.df['test_category'].value_counts()
            print(f"\nTest Categories: {len(category_counts)}")
            for category, count in category_counts.head(10).items():
                percentage = (count / len(self.df)) * 100
                print(f"  {category:25}: {count:6,} ({percentage:5.1f}%)")
    
    def detect_performance_inversions(self):
        """Performance Inversion (P0) 탐지 - 가장 심각한 불공정 패턴"""
        print(f"\n[2] DETECTING PERFORMANCE INVERSIONS (P0 - CRITICAL)")
        print("=" * 60)
        
        if 'model_name' not in self.df.columns or 'test_category' not in self.df.columns:
            print("ERROR: Required columns missing for performance inversion analysis")
            return []
            
        inversions = []
        
        # 테스트 카테고리별로 모델 성능 계산
        for category in self.df['test_category'].unique():
            category_data = self.df[self.df['test_category'] == category]
            
            # 모델별 성공률 계산
            model_performance = {}
            for model in category_data['model_name'].unique():
                model_data = category_data[category_data['model_name'] == model]
                if len(model_data) > 0:
                    # score가 있으면 사용, 없으면 valid 사용
                    if 'score' in model_data.columns:
                        success_rate = model_data['score'].mean()
                    elif 'is_valid' in model_data.columns:
                        success_rate = model_data['is_valid'].mean()
                    else:
                        success_rate = 0.5  # 기본값
                        
                    model_performance[model] = {
                        'success_rate': success_rate,
                        'expected_performance': self.performance_baseline.get(model, 50),
                        'sample_size': len(model_data)
                    }
            
            # Performance Inversion 탐지
            for weak_model, weak_stats in model_performance.items():
                for strong_model, strong_stats in model_performance.items():
                    if weak_model == strong_model:
                        continue
                        
                    # 예상 성능: strong_model > weak_model 이어야 함
                    expected_strong = strong_stats['expected_performance']
                    expected_weak = weak_stats['expected_performance']
                    
                    # 실제 성능: weak_model > strong_model (역전!)
                    actual_strong = strong_stats['success_rate'] 
                    actual_weak = weak_stats['success_rate']
                    
                    # 역전 조건
                    if (expected_strong > expected_weak + 10 and  # 예상 차이 10점 이상
                        actual_weak > actual_strong + 0.1 and    # 실제 역전 10%p 이상
                        min(weak_stats['sample_size'], strong_stats['sample_size']) >= 10):  # 충분한 샘플
                        
                        inversion = {
                            'category': category,
                            'weak_model': weak_model,
                            'strong_model': strong_model,
                            'expected_gap': expected_strong - expected_weak,
                            'actual_gap': actual_weak - actual_strong,
                            'inversion_magnitude': (actual_weak - actual_strong) + (expected_strong - expected_weak) / 100,
                            'weak_performance': actual_weak,
                            'strong_performance': actual_strong,
                            'weak_expected': expected_weak,
                            'strong_expected': expected_strong,
                            'weak_sample_size': weak_stats['sample_size'],
                            'strong_sample_size': strong_stats['sample_size']
                        }
                        inversions.append(inversion)
        
        # 결과 출력
        if inversions:
            print(f"CRITICAL: {len(inversions)} Performance Inversions detected!")
            
            # 심각도 순 정렬
            inversions.sort(key=lambda x: x['inversion_magnitude'], reverse=True)
            
            print(f"\nTop 10 Most Severe Inversions:")
            for i, inv in enumerate(inversions[:10], 1):
                print(f"\n{i}. Category: {inv['category']}")
                print(f"   Weak Model Outperforming: {inv['weak_model']} ({inv['weak_performance']:.3f})")
                print(f"   Strong Model Underperforming: {inv['strong_model']} ({inv['strong_performance']:.3f})")
                print(f"   Performance Gap: {inv['actual_gap']:.3f} (Expected: {-inv['expected_gap']:.3f})")
                print(f"   Inversion Magnitude: {inv['inversion_magnitude']:.3f}")
                print(f"   Sample Sizes: {inv['weak_sample_size']} vs {inv['strong_sample_size']}")
        else:
            print("No significant performance inversions detected.")
            
        return inversions
    
    def analyze_family_bias_patterns(self):
        """모델 패밀리별 체계적 편향 패턴 분석 (P1)"""
        print(f"\n[3] ANALYZING FAMILY BIAS PATTERNS (P1 - HIGH PRIORITY)")
        print("=" * 60)
        
        family_analysis = {}
        
        for family_name, models in self.model_families.items():
            print(f"\n[{family_name} Family Analysis]")
            
            # 해당 패밀리 데이터 필터링
            family_data = self.df[self.df['model_name'].isin(models)]
            
            if len(family_data) == 0:
                print(f"  No data found for {family_name} family")
                continue
                
            print(f"  Models: {len([m for m in models if m in self.df['model_name'].unique()])} out of {len(models)}")
            print(f"  Total Records: {len(family_data):,}")
            
            # 패밀리별 불공정 패턴 분석
            bias_patterns = self._analyze_family_specific_patterns(family_name, family_data)
            family_analysis[family_name] = bias_patterns
            
            # 결과 출력
            if bias_patterns['systematic_issues'] > 0:
                print(f"  SYSTEMATIC ISSUES: {bias_patterns['systematic_issues']} detected")
                
            if bias_patterns['common_errors']:
                print(f"  Common Error Patterns:")
                for error, count in bias_patterns['common_errors'].most_common(3):
                    print(f"    - {error}: {count} cases")
                    
        return family_analysis
    
    def _analyze_family_specific_patterns(self, family_name, family_data):
        """패밀리별 세부 패턴 분석"""
        patterns = {
            'systematic_issues': 0,
            'common_errors': Counter(),
            'performance_issues': [],
            'bias_indicators': []
        }
        
        # 오류 메시지 패턴 분석
        if 'error_message' in family_data.columns:
            error_messages = family_data['error_message'].dropna()
            if len(error_messages) > 0:
                patterns['common_errors'] = Counter(error_messages)
                
                # 체계적 이슈 탐지 (같은 오류가 많은 비율을 차지)
                most_common_error = patterns['common_errors'].most_common(1)
                if most_common_error:
                    error_rate = most_common_error[0][1] / len(family_data)
                    if error_rate > 0.3:  # 30% 이상이 같은 오류
                        patterns['systematic_issues'] += 1
        
        # 패밀리별 특수 편향 패턴
        if family_name == 'OpenAI':
            patterns['bias_indicators'].extend(self._detect_openai_bias(family_data))
        elif family_name == 'Anthropic':
            patterns['bias_indicators'].extend(self._detect_anthropic_bias(family_data))
        elif family_name == 'DeepSeek':
            patterns['bias_indicators'].extend(self._detect_deepseek_bias(family_data))
            
        return patterns
    
    def _detect_openai_bias(self, data):
        """OpenAI 모델 특화 편향 탐지"""
        bias_indicators = []
        
        # OpenAI API 특화 이슈
        if 'error_message' in data.columns:
            openai_specific_errors = data['error_message'].str.contains(
                'rate limit|usage limit|content filter', case=False, na=False
            ).sum()
            
            if openai_specific_errors > len(data) * 0.1:  # 10% 이상
                bias_indicators.append({
                    'type': 'openai_api_limits',
                    'count': openai_specific_errors,
                    'rate': openai_specific_errors / len(data)
                })
        
        return bias_indicators
    
    def _detect_anthropic_bias(self, data):
        """Anthropic 모델 특화 편향 탐지"""
        bias_indicators = []
        
        # max_tokens 요구사항 편향
        if 'error_message' in data.columns:
            max_tokens_issues = data['error_message'].str.contains(
                'max_tokens.*required', case=False, na=False
            ).sum()
            
            if max_tokens_issues > len(data) * 0.1:  # 10% 이상
                bias_indicators.append({
                    'type': 'max_tokens_requirement_bias',
                    'count': max_tokens_issues,
                    'rate': max_tokens_issues / len(data)
                })
        
        return bias_indicators
    
    def _detect_deepseek_bias(self, data):
        """DeepSeek 모델 특화 편향 탐지"""
        bias_indicators = []
        
        # DeepSeek 특화 이슈 (예: 언어별 성능 차이)
        if 'test_category' in data.columns:
            # 다국어 태스크에서 성능 편향 체크
            multilingual_tasks = data[
                data['test_category'].str.contains('java|javascript', case=False, na=False)
            ]
            
            if len(multilingual_tasks) > 0:
                success_rate = multilingual_tasks.get('score', multilingual_tasks.get('is_valid', pd.Series([0.5]))).mean()
                if success_rate < 0.3:  # 30% 미만 성공률
                    bias_indicators.append({
                        'type': 'multilingual_performance_bias',
                        'success_rate': success_rate,
                        'sample_size': len(multilingual_tasks)
                    })
        
        return bias_indicators
    
    def detect_diverse_unfair_patterns(self):
        """다양한 불공정 유형 탐지 - Parsing/Format 실패 등"""
        print(f"\n[4] DETECTING DIVERSE UNFAIR PATTERNS")
        print("=" * 60)
        
        # 전체 데이터에 대해 탐지기 실행
        detector = UnfairEvaluationDetector(self.df)
        results = detector.classify_all_issues()
        
        print(f"\nDiversified Pattern Detection Results:")
        print(f"  Technical Errors: {results['technical_errors']:,}")
        print(f"  API Configuration Bias: {results['api_bias']:,}")
        print(f"  Parsing Failures: {results['parsing_failures']:,}")  
        print(f"  State Management Issues: {results['state_issues']:,}")
        print(f"  Infrastructure Dependencies: {results['infrastructure']:,}")
        print(f"  Format Discrimination: {results['format_discrimination']:,}")
        
        # 새로운 패턴 탐지를 위한 확장 분석
        extended_patterns = self._detect_extended_unfair_patterns()
        
        return {**results, **extended_patterns}
    
    def _detect_extended_unfair_patterns(self):
        """확장 불공정 패턴 탐지"""
        extended = {}
        
        # 1. 언어별 편향 패턴
        language_bias = self._detect_language_bias()
        extended['language_bias'] = language_bias
        
        # 2. 테스트 길이별 편향
        length_bias = self._detect_length_bias() 
        extended['length_bias'] = length_bias
        
        # 3. 복잡도별 편향
        complexity_bias = self._detect_complexity_bias()
        extended['complexity_bias'] = complexity_bias
        
        print(f"  Language Bias Cases: {language_bias}")
        print(f"  Length Bias Cases: {length_bias}")
        print(f"  Complexity Bias Cases: {complexity_bias}")
        
        return extended
    
    def _detect_language_bias(self):
        """프로그래밍 언어별 편향 탐지"""
        language_bias_count = 0
        
        if 'test_category' in self.df.columns:
            # 언어별 카테고리 식별
            language_categories = ['javascript', 'java', 'python']
            
            for lang in language_categories:
                lang_data = self.df[
                    self.df['test_category'].str.contains(lang, case=False, na=False)
                ]
                
                if len(lang_data) > 100:  # 충분한 샘플
                    # 모델별 언어 성능 편향 체크
                    for model in lang_data['model_name'].unique():
                        model_lang_data = lang_data[lang_data['model_name'] == model]
                        other_data = self.df[
                            (self.df['model_name'] == model) & 
                            (~self.df['test_category'].str.contains(lang, case=False, na=False))
                        ]
                        
                        if len(model_lang_data) > 10 and len(other_data) > 10:
                            lang_score = model_lang_data.get('score', pd.Series([0.5])).mean()
                            other_score = other_data.get('score', pd.Series([0.5])).mean()
                            
                            # 특정 언어에서 비정상적으로 낮은 성능
                            if abs(lang_score - other_score) > 0.3:  # 30%p 차이
                                language_bias_count += 1
                                
        return language_bias_count
    
    def _detect_length_bias(self):
        """테스트 길이별 편향 탐지"""
        length_bias_count = 0
        
        if 'question' in self.df.columns:
            # 질문 길이 계산
            self.df['question_length'] = self.df['question'].astype(str).str.len()
            
            # 길이별 성능 편향 체크
            short_tasks = self.df[self.df['question_length'] < 200]  # 짧은 질문
            long_tasks = self.df[self.df['question_length'] > 1000]   # 긴 질문
            
            if len(short_tasks) > 100 and len(long_tasks) > 100:
                for model in self.df['model_name'].unique():
                    model_short = short_tasks[short_tasks['model_name'] == model]
                    model_long = long_tasks[long_tasks['model_name'] == model]
                    
                    if len(model_short) > 5 and len(model_long) > 5:
                        short_score = model_short.get('score', pd.Series([0.5])).mean()
                        long_score = model_long.get('score', pd.Series([0.5])).mean()
                        
                        # 비정상적인 길이 편향
                        if abs(short_score - long_score) > 0.4:  # 40%p 차이
                            length_bias_count += 1
                            
        return length_bias_count
    
    def _detect_complexity_bias(self):
        """복잡도별 편향 탐지"""
        complexity_bias_count = 0
        
        # 복잡도 추정 (function 개수, 파라미터 개수 등)
        if 'function' in self.df.columns:
            self.df['function_count'] = self.df['function'].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            
            simple_tasks = self.df[self.df['function_count'] <= 1]    # 단순 태스크
            complex_tasks = self.df[self.df['function_count'] >= 3]   # 복잡 태스크
            
            if len(simple_tasks) > 100 and len(complex_tasks) > 100:
                for model in self.df['model_name'].unique():
                    model_simple = simple_tasks[simple_tasks['model_name'] == model]
                    model_complex = complex_tasks[complex_tasks['model_name'] == model]
                    
                    if len(model_simple) > 5 and len(model_complex) > 5:
                        simple_score = model_simple.get('score', pd.Series([0.5])).mean()
                        complex_score = model_complex.get('score', pd.Series([0.5])).mean()
                        
                        # 예상과 다른 복잡도 편향 (복잡한 태스크가 더 높은 점수?)
                        if complex_score > simple_score + 0.2:  # 복잡한 게 20%p 더 높음
                            complexity_bias_count += 1
                            
        return complexity_bias_count
    
    def calculate_ranking_impact(self, inversions, unfair_patterns):
        """정량적 영향도 분석 - 순위 변화 시뮬레이션"""
        print(f"\n[5] QUANTITATIVE IMPACT ANALYSIS")
        print("=" * 60)
        
        # 현재 순위 계산 (전체 점수 기준)
        current_rankings = self._calculate_current_rankings()
        
        # 불공정 평가 제외 시 순위 계산
        fair_rankings = self._calculate_fair_rankings(unfair_patterns)
        
        # 순위 변화 분석
        ranking_changes = self._analyze_ranking_changes(current_rankings, fair_rankings)
        
        # Performance Inversion의 순위 영향
        inversion_impact = self._calculate_inversion_impact(inversions, current_rankings)
        
        print(f"Ranking Impact Summary:")
        print(f"  Models with Significant Ranking Changes: {ranking_changes['significant_changes']}")
        print(f"  Average Ranking Change: {ranking_changes['avg_change']:.1f} positions")
        print(f"  Maximum Ranking Change: {ranking_changes['max_change']} positions")
        print(f"  Models Affected by Performance Inversions: {inversion_impact['affected_models']}")
        
        return {
            'current_rankings': current_rankings,
            'fair_rankings': fair_rankings,
            'ranking_changes': ranking_changes,
            'inversion_impact': inversion_impact
        }
    
    def _calculate_current_rankings(self):
        """현재 순위 계산"""
        model_scores = {}
        
        for model in self.df['model_name'].unique():
            model_data = self.df[self.df['model_name'] == model]
            
            if 'score' in model_data.columns:
                avg_score = model_data['score'].mean()
            elif 'is_valid' in model_data.columns:
                avg_score = model_data['is_valid'].mean() 
            else:
                avg_score = 0.5
                
            model_scores[model] = avg_score
        
        # 점수 기준 순위
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        rankings = {model: rank+1 for rank, (model, score) in enumerate(sorted_models)}
        
        return rankings
    
    def _calculate_fair_rankings(self, unfair_patterns):
        """불공정 평가 제외 시 순위"""
        # 불공정으로 분류된 케이스 제외
        if 'issue_classification' in self.df.columns:
            fair_data = self.df[self.df['issue_classification'] == 'Fair Evaluation']
        else:
            # 아직 분류가 안된 경우, 기술적 오류가 없는 것으로 가정
            fair_data = self.df[
                (self.df.get('input_token_count', pd.Series([1])).astype(float) > 0) | 
                (self.df.get('output_token_count', pd.Series([1])).astype(float) > 0) |
                (self.df.get('latency', self.df.get('execution_time', pd.Series([1]))).astype(float) >= 0.001)
            ]
        
        if len(fair_data) == 0:
            return self._calculate_current_rankings()  # fallback
        
        model_scores = {}
        
        for model in fair_data['model_name'].unique():
            model_data = fair_data[fair_data['model_name'] == model]
            
            if len(model_data) > 0:
                if 'score' in model_data.columns:
                    avg_score = model_data['score'].mean()
                elif 'is_valid' in model_data.columns:
                    avg_score = model_data['is_valid'].mean()
                else:
                    avg_score = 0.5
            else:
                avg_score = 0.0  # 공정한 평가 데이터가 없음
                
            model_scores[model] = avg_score
        
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        rankings = {model: rank+1 for rank, (model, score) in enumerate(sorted_models)}
        
        return rankings
    
    def _analyze_ranking_changes(self, current, fair):
        """순위 변화 분석"""
        changes = {}
        position_changes = []
        
        for model in current.keys():
            if model in fair:
                change = current[model] - fair[model]  # 음수면 순위 상승
                changes[model] = change
                position_changes.append(abs(change))
        
        significant_changes = sum(1 for change in position_changes if change >= 3)
        avg_change = np.mean(position_changes) if position_changes else 0
        max_change = max(position_changes) if position_changes else 0
        
        return {
            'changes': changes,
            'significant_changes': significant_changes,
            'avg_change': avg_change,
            'max_change': max_change
        }
    
    def _calculate_inversion_impact(self, inversions, current_rankings):
        """Performance Inversion의 순위 영향"""
        affected_models = set()
        
        for inversion in inversions:
            affected_models.add(inversion['weak_model'])
            affected_models.add(inversion['strong_model'])
        
        return {
            'affected_models': len(affected_models),
            'total_inversions': len(inversions)
        }
    
    def _count_unfair_evaluations(self):
        """불공정 평가 개수 계산"""
        if 'issue_classification' in self.df.columns:
            return len(self.df[self.df['issue_classification'] != 'Fair Evaluation'])
        else:
            # 기술적 오류로 추정되는 케이스들
            unfair_cases = (
                (self.df.get('input_token_count', pd.Series([1])).astype(float) == 0) & 
                (self.df.get('output_token_count', pd.Series([1])).astype(float) == 0)
            ).sum()
            return unfair_cases
    
    def generate_comprehensive_report(self, all_results):
        """종합 리포트 생성"""
        print(f"\n[6] GENERATING COMPREHENSIVE REPORT")
        print("=" * 60)
        
        # 리포트 파일 경로
        report_path = self.base_path / "불공정 평가 이슈 식별 및 분류"
        
        # 1. Performance Inversion 리포트
        inversion_df = pd.DataFrame(all_results['inversions'])
        if len(inversion_df) > 0:
            inversion_file = report_path / 'performance_inversions_p0_critical.csv'
            inversion_df.to_csv(inversion_file, index=False, encoding='utf-8-sig')
            print(f"OK: Performance Inversions (P0): {inversion_file}")
        
        # 2. 패밀리 편향 리포트
        family_report = report_path / 'family_bias_patterns_p1.json'
        with open(family_report, 'w', encoding='utf-8') as f:
            json.dump(all_results['family_analysis'], f, indent=2, ensure_ascii=False, default=str)
        print(f"OK: Family Bias Patterns (P1): {family_report}")
        
        # 3. 순위 영향도 리포트
        ranking_report = report_path / 'ranking_impact_analysis.json'
        with open(ranking_report, 'w', encoding='utf-8') as f:
            json.dump(all_results['ranking_impact'], f, indent=2, ensure_ascii=False, default=str)
        print(f"OK: Ranking Impact Analysis: {ranking_report}")
        
        # 4. 종합 요약 리포트
        self._generate_executive_summary(all_results, report_path)
        
        return report_path
    
    def _generate_executive_summary(self, results, report_path):
        """경영진 요약 리포트"""
        summary = {
            'benchmark_credibility_assessment': self._assess_benchmark_credibility(results),
            'critical_findings': self._extract_critical_findings(results),
            'recommended_actions': self._generate_recommendations(results),
            'quantitative_impact': {
                'total_unfair_evaluations': self._count_unfair_evaluations(),
                'unfair_percentage': (self._count_unfair_evaluations() / len(self.df)) * 100,
                'performance_inversions': len(results['inversions']),
                'models_affected_by_inversions': results['ranking_impact']['inversion_impact']['affected_models'],
                'significant_ranking_changes': results['ranking_impact']['ranking_changes']['significant_changes']
            }
        }
        
        summary_file = report_path / 'executive_summary_benchmark_credibility.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"OK: Executive Summary: {summary_file}")
        
        # 텍스트 요약도 생성
        self._write_text_summary(summary, report_path)
    
    def _assess_benchmark_credibility(self, results):
        """벤치마크 신뢰도 평가"""
        total_unfair = self._count_unfair_evaluations()
        unfair_rate = total_unfair / len(self.df)
        
        if unfair_rate > 0.3:
            credibility = 'SEVERELY COMPROMISED'
        elif unfair_rate > 0.15:
            credibility = 'MODERATELY COMPROMISED' 
        elif unfair_rate > 0.05:
            credibility = 'MILDLY COMPROMISED'
        else:
            credibility = 'ACCEPTABLE'
            
        return {
            'overall_assessment': credibility,
            'unfair_evaluation_rate': unfair_rate,
            'performance_inversion_count': len(results['inversions']),
            'systemic_bias_detected': any(
                family_data.get('systematic_issues', 0) > 0 
                for family_data in results['family_analysis'].values()
            )
        }
    
    def _extract_critical_findings(self, results):
        """핵심 발견사항 추출"""
        return {
            'most_problematic_models': self._identify_most_problematic_models(results),
            'worst_performance_inversions': sorted(results['inversions'], key=lambda x: x['inversion_magnitude'], reverse=True)[:5],
            'systemic_family_biases': [
                family for family, data in results['family_analysis'].items() 
                if data.get('systematic_issues', 0) > 0
            ]
        }
    
    def _identify_most_problematic_models(self, results):
        """가장 문제가 많은 모델 식별"""
        model_problems = defaultdict(int)
        
        # Performance Inversion에서 문제 모델
        for inv in results['inversions']:
            model_problems[inv['weak_model']] += 1
            model_problems[inv['strong_model']] += 1
        
        # 불공정 평가 비율
        if hasattr(self, 'df') and 'model_name' in self.df.columns:
            for model in self.df['model_name'].unique():
                model_data = self.df[self.df['model_name'] == model]
                if 'issue_classification' in model_data.columns:
                    unfair_count = len(model_data[model_data['issue_classification'] != 'Fair Evaluation'])
                else:
                    unfair_count = ((model_data.get('input_token_count', pd.Series([1])).astype(float) == 0) & 
                                   (model_data.get('output_token_count', pd.Series([1])).astype(float) == 0)).sum()
                unfair_rate = unfair_count / len(model_data)
                
                if unfair_rate > 0.5:  # 50% 이상 불공정
                    model_problems[model] += 10
                elif unfair_rate > 0.2:  # 20% 이상 불공정
                    model_problems[model] += 5
        
        # 상위 5개 문제 모델
        return sorted(model_problems.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _generate_recommendations(self, results):
        """권고사항 생성"""
        recommendations = []
        
        # Performance Inversion 대응
        if results['inversions']:
            recommendations.append({
                'priority': 'P0_CRITICAL',
                'action': 'IMMEDIATE_REVALUATION_REQUIRED',
                'description': f"{len(results['inversions'])} performance inversions detected. These fundamentally undermine benchmark credibility and require immediate investigation and revaluation.",
                'affected_models': len(set([inv['weak_model'] for inv in results['inversions']] + [inv['strong_model'] for inv in results['inversions']]))
            })
        
        # 체계적 편향 대응
        systemic_families = [f for f, data in results['family_analysis'].items() if data.get('systematic_issues', 0) > 0]
        if systemic_families:
            recommendations.append({
                'priority': 'P1_HIGH',
                'action': 'EVALUATION_METHODOLOGY_REVIEW',
                'description': f"Systematic bias detected in {len(systemic_families)} model families: {', '.join(systemic_families)}. Evaluation methodology needs review.",
                'affected_families': systemic_families
            })
        
        # 불공정 평가율 높은 경우
        total_unfair = self._count_unfair_evaluations()
        unfair_rate = total_unfair / len(self.df)
        
        if unfair_rate > 0.15:
            recommendations.append({
                'priority': 'P1_HIGH',
                'action': 'BENCHMARK_INFRASTRUCTURE_OVERHAUL',
                'description': f"Unfair evaluation rate of {unfair_rate:.1%} indicates serious infrastructure issues. Comprehensive overhaul required.",
                'unfair_rate': unfair_rate
            })
        
        return recommendations
    
    def _write_text_summary(self, summary, report_path):
        """텍스트 요약 작성"""
        text_file = report_path / 'COMPREHENSIVE_ANALYSIS_SUMMARY.txt'
        
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write("BFCL BENCHMARK COMPREHENSIVE UNFAIR EVALUATION ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            # 신뢰도 평가
            cred = summary['benchmark_credibility_assessment']
            f.write(f"BENCHMARK CREDIBILITY: {cred['overall_assessment']}\n")
            f.write(f"Unfair Evaluation Rate: {cred['unfair_evaluation_rate']:.1%}\n")
            f.write(f"Performance Inversions: {cred['performance_inversion_count']}\n")
            f.write(f"Systemic Bias Detected: {cred['systemic_bias_detected']}\n\n")
            
            # 정량적 영향
            quant = summary['quantitative_impact']
            f.write("QUANTITATIVE IMPACT:\n")
            f.write(f"  Total Unfair Evaluations: {quant['total_unfair_evaluations']:,}\n")
            f.write(f"  Unfair Percentage: {quant['unfair_percentage']:.1f}%\n")
            f.write(f"  Performance Inversions: {quant['performance_inversions']}\n")
            f.write(f"  Models Affected by Inversions: {quant['models_affected_by_inversions']}\n")
            f.write(f"  Significant Ranking Changes: {quant['significant_ranking_changes']}\n\n")
            
            # 권고사항
            f.write("RECOMMENDED ACTIONS:\n")
            for i, rec in enumerate(summary['recommended_actions'], 1):
                f.write(f"{i}. [{rec['priority']}] {rec['action']}\n")
                f.write(f"   {rec['description']}\n\n")
        
        print(f"OK: Text Summary: {text_file}")

def main():
    """메인 실행"""
    try:
        analyzer = ComprehensiveUnfairAnalysis()
        
        # 1. 전체 데이터 로드
        analyzer.load_and_analyze_all_data()
        
        # 2. Performance Inversion 탐지
        inversions = analyzer.detect_performance_inversions()
        
        # 3. 패밀리별 편향 분석
        family_analysis = analyzer.analyze_family_bias_patterns()
        
        # 4. 다양한 불공정 패턴 탐지
        diverse_patterns = analyzer.detect_diverse_unfair_patterns()
        
        # 5. 순위 영향도 분석
        ranking_impact = analyzer.calculate_ranking_impact(inversions, diverse_patterns)
        
        # 6. 종합 리포트 생성
        all_results = {
            'inversions': inversions,
            'family_analysis': family_analysis,
            'diverse_patterns': diverse_patterns,
            'ranking_impact': ranking_impact
        }
        
        report_path = analyzer.generate_comprehensive_report(all_results)
        
        print(f"\n" + "=" * 80)
        print("COMPREHENSIVE ANALYSIS COMPLETED!")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)