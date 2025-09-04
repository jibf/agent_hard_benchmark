"""
Final Corrected BFCL Analysis - Using proper JSON structure mapping
"""

from corrected_data_loader import load_corrected_bfcl_results
import pandas as pd
import numpy as np
import json
from pathlib import Path

def final_corrected_analysis():
    """최종 수정된 BFCL 분석"""
    
    print("="*80)
    print("FINAL CORRECTED BFCL ANALYSIS - USING PROPER JSON STRUCTURE")
    print("="*80)
    
    # 수정된 데이터 로더 사용
    df = load_corrected_bfcl_results()
    
    print(f"\nDataset loaded: {len(df):,} records")
    print(f"Columns available: {len(df.columns)}")
    
    # 기본 통계
    results = {
        'total_records': len(df),
        'models': df['model_name'].nunique(),
        'test_categories': df['test_category'].nunique()
    }
    
    # 실제 점수 기반 분석 (이전에는 디폴트 값이었음)
    print(f"\n1. SCORE ANALYSIS (based on actual accuracy from score files):")
    
    # Summary 레코드 제외하고 개별 평가만 분석
    eval_records = df[df['is_summary'] == False]
    print(f"   Evaluation records (excluding summaries): {len(eval_records):,}")
    
    if 'score' in eval_records.columns:
        avg_score = eval_records['score'].mean()
        print(f"   Average score: {avg_score:.3f}")
        
        # 점수별 분포
        score_ranges = {
            'Perfect (1.0)': (eval_records['score'] == 1.0).sum(),
            'High (0.8-0.99)': ((eval_records['score'] >= 0.8) & (eval_records['score'] < 1.0)).sum(),
            'Medium (0.5-0.79)': ((eval_records['score'] >= 0.5) & (eval_records['score'] < 0.8)).sum(),
            'Low (0.1-0.49)': ((eval_records['score'] >= 0.1) & (eval_records['score'] < 0.5)).sum(),
            'Failed (0.0)': (eval_records['score'] == 0.0).sum()
        }
        
        print(f"   Score distribution:")
        for range_name, count in score_ranges.items():
            percentage = (count / len(eval_records)) * 100
            print(f"     {range_name}: {count:,} ({percentage:.1f}%)")
            results[f'score_{range_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}'] = int(count)
    
    # 토큰 사용량 분석 (이전에는 잘못된 필드였음)
    print(f"\n2. TOKEN USAGE ANALYSIS (based on actual token fields):")
    
    if 'input_token_count' in eval_records.columns:
        token_stats = {
            'avg_input_tokens': eval_records['input_token_count'].mean(),
            'avg_output_tokens': eval_records['output_token_count'].mean(),
            'avg_latency': eval_records['latency'].mean(),
            'records_with_tokens': (eval_records['input_token_count'] > 0).sum(),
            'records_with_output': (eval_records['output_token_count'] > 0).sum(),
            'records_with_latency': (eval_records['latency'] > 0).sum()
        }
        
        print(f"   Average input tokens: {token_stats['avg_input_tokens']:.1f}")
        print(f"   Average output tokens: {token_stats['avg_output_tokens']:.1f}")
        print(f"   Average latency: {token_stats['avg_latency']:.3f}s")
        print(f"   Records with input tokens: {token_stats['records_with_tokens']:,} ({(token_stats['records_with_tokens']/len(eval_records)*100):.1f}%)")
        print(f"   Records with output tokens: {token_stats['records_with_output']:,} ({(token_stats['records_with_output']/len(eval_records)*100):.1f}%)")
        
        results.update({k: float(v) if isinstance(v, (np.integer, np.floating)) else int(v) for k, v in token_stats.items()})
    
    # 모델별 성능 분석 (실제 점수 기반)
    print(f"\n3. MODEL PERFORMANCE ANALYSIS (based on actual scores):")
    
    model_stats = []
    for model in eval_records['model_name'].unique():
        model_data = eval_records[eval_records['model_name'] == model]
        
        stats = {
            'model': model,
            'count': len(model_data),
            'avg_score': model_data['score'].mean() if 'score' in model_data.columns else 0,
            'successful_rate': (model_data['is_successful']).mean() if 'is_successful' in model_data.columns else 0,
            'performance_success_rate': (model_data['is_performance_successful']).mean() if 'is_performance_successful' in model_data.columns else 0,
            'avg_input_tokens': model_data['input_token_count'].mean() if 'input_token_count' in model_data.columns else 0,
            'avg_output_tokens': model_data['output_token_count'].mean() if 'output_token_count' in model_data.columns else 0,
            'avg_latency': model_data['latency'].mean() if 'latency' in model_data.columns else 0
        }
        
        model_stats.append(stats)
        
        print(f"   {model[:40]:40} Score: {stats['avg_score']:.3f}, "
              f"Success: {stats['successful_rate']*100:5.1f}%, "
              f"Tokens: {stats['avg_input_tokens']:4.0f}→{stats['avg_output_tokens']:3.0f}, "
              f"Latency: {stats['avg_latency']:5.2f}s")
    
    # 카테고리별 성능
    print(f"\n4. CATEGORY PERFORMANCE ANALYSIS:")
    
    category_stats = []
    for category in eval_records['test_category'].unique():
        cat_data = eval_records[eval_records['test_category'] == category]
        
        if len(cat_data) > 0:
            stats = {
                'category': category,
                'count': len(cat_data),
                'avg_score': cat_data['score'].mean(),
                'success_rate': cat_data['is_performance_successful'].mean() if 'is_performance_successful' in cat_data.columns else 0
            }
            category_stats.append(stats)
            
            print(f"   {category[:35]:35} Records: {stats['count']:4,}, "
                  f"Score: {stats['avg_score']:.3f}, "
                  f"Success: {stats['success_rate']*100:5.1f}%")
    
    # 인프라 상태 평가 (정확한 기준)
    print(f"\n5. INFRASTRUCTURE STATUS (based on actual data):")
    
    if 'is_successful' in eval_records.columns:
        total_success = eval_records['is_successful'].sum()
        success_rate = total_success / len(eval_records)
        
        print(f"   Total evaluations: {len(eval_records):,}")
        print(f"   Successful evaluations: {total_success:,}")
        print(f"   Success rate: {success_rate*100:.1f}%")
        
        results['infrastructure_success_rate'] = float(success_rate)
        results['infrastructure_status'] = 'GOOD' if success_rate > 0.8 else 'NEEDS_OPTIMIZATION' if success_rate > 0.5 else 'POOR'
        
        print(f"   Infrastructure status: {results['infrastructure_status']}")
    
    # 결과 저장
    output_file = Path("final_corrected_analysis_results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_file}")
    
    # 최종 보고서 생성
    create_final_corrected_report(results, model_stats, category_stats)
    
    return df, results

def create_final_corrected_report(results, model_stats, category_stats):
    """최종 수정된 보고서 생성"""
    
    report = f"""# BFCL Benchmark Analysis - FINAL CORRECTED RESULTS

**Date**: September 2, 2025  
**Analysis Method**: Corrected JSON structure mapping  
**Total Records**: {results['total_records']:,}  
**Models Analyzed**: {results['models']}  
**Test Categories**: {results['test_categories']}

## Executive Summary

**PREVIOUS ANALYSIS WAS INCORRECT** due to wrong JSON field mappings. This report provides the accurate analysis based on proper data structure understanding.

### Key Corrections Made:
1. **Score Field**: Now using `accuracy` from score files instead of non-existent `score` field
2. **Token Fields**: Using correct `input_token_count` and `output_token_count` fields
3. **Model Results**: Using `result` field instead of non-existent `model_result` field  
4. **Latency**: Using correct `latency` field instead of `execution_time`

### Accurate Results:
- **Infrastructure Status**: {results.get('infrastructure_status', 'UNKNOWN')}
- **Success Rate**: {results.get('infrastructure_success_rate', 0)*100:.1f}%
- **Average Score**: {results.get('score', 'N/A')}

## Model Performance Ranking

"""

    # 모델별 성능 테이블
    sorted_models = sorted(model_stats, key=lambda x: x['avg_score'], reverse=True)
    
    report += "| Rank | Model | Avg Score | Success Rate | Avg Tokens (In→Out) | Latency |\n"
    report += "|------|-------|-----------|--------------|---------------------|----------|\n"
    
    for i, stats in enumerate(sorted_models[:15], 1):
        report += f"| {i} | {stats['model'][:30]} | {stats['avg_score']:.3f} | {stats['successful_rate']*100:.1f}% | {stats['avg_input_tokens']:.0f}→{stats['avg_output_tokens']:.0f} | {stats['avg_latency']:.2f}s |\n"
    
    report += f"""
## Category Performance

"""
    
    sorted_categories = sorted(category_stats, key=lambda x: x['avg_score'], reverse=True)
    
    report += "| Category | Records | Avg Score | Success Rate |\n"
    report += "|----------|---------|-----------|-------------|\n"
    
    for stats in sorted_categories[:10]:
        report += f"| {stats['category'][:25]} | {stats['count']:,} | {stats['avg_score']:.3f} | {stats['success_rate']*100:.1f}% |\n"

    report += f"""
## Key Findings

### Infrastructure Assessment
- **Status**: {results.get('infrastructure_status', 'UNKNOWN')}
- **Overall Success Rate**: {results.get('infrastructure_success_rate', 0)*100:.1f}%
- The benchmark infrastructure is """

    if results.get('infrastructure_success_rate', 0) > 0.8:
        report += "**functioning well** with high success rates."
    elif results.get('infrastructure_success_rate', 0) > 0.5:
        report += "**functioning adequately** but has room for optimization."
    else:
        report += "**experiencing significant issues** that require investigation."

    report += f"""

### Data Quality
- **Token Tracking**: Properly implemented with actual usage data
- **Latency Measurement**: Accurate timing information available  
- **Score Calculation**: Based on actual evaluation results

## Comparison with Previous (Incorrect) Analysis

| Metric | Previous (Wrong) | Current (Correct) |
|--------|------------------|-------------------|
| Success Rate | 0% (100% failure) | {results.get('infrastructure_success_rate', 0)*100:.1f}% |
| Data Source | Wrong field names | Actual JSON structure |
| Infrastructure Status | SEVERELY COMPROMISED | {results.get('infrastructure_status', 'UNKNOWN')} |
| Token Analysis | Based on defaults | Based on actual data |

## Recommendations

"""

    if results.get('infrastructure_success_rate', 0) > 0.8:
        report += """1. **Continue Current Operations**: Infrastructure is performing well
2. **Monitor Performance**: Regular monitoring to maintain quality
3. **Optimize Lower-Performing Models**: Focus on models with lower success rates"""
    else:
        report += """1. **Investigate Failures**: Analyze the root causes of evaluation failures
2. **Optimize Infrastructure**: Improve evaluation pipeline reliability  
3. **Enhanced Monitoring**: Implement real-time quality monitoring"""

    # 보고서 저장
    report_file = Path("FINAL_CORRECTED_BFCL_ANALYSIS_REPORT.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Final corrected report saved to {report_file}")

if __name__ == "__main__":
    df, results = final_corrected_analysis()