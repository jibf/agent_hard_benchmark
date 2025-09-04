#!/usr/bin/env python3
"""
BFCL Unfair Evaluation Detection & Classification System
Main Execution Script

이 스크립트는 BFCL 벤치마크 결과에서 불공정한 평가 사례를 자동으로 탐지하고 분류합니다.

Usage:
    python main_unfair_evaluation_analysis.py [--base-path PATH] [--output-dir PATH]
"""

import argparse
import sys
import traceback
from pathlib import Path
import pandas as pd
from typing import Optional

# 로컬 모듈 import
try:
    from data_loader import load_bfcl_results, BFCLDataLoader
    from unfair_evaluation_detector import UnfairEvaluationDetector
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure data_loader.py and unfair_evaluation_detector.py are in the same directory")
    sys.exit(1)


def main():
    """메인 실행 함수"""
    
    # Command line arguments 파싱
    parser = argparse.ArgumentParser(
        description="BFCL Unfair Evaluation Detection & Classification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main_unfair_evaluation_analysis.py
    python main_unfair_evaluation_analysis.py --base-path "C:/path/to/bfcl/data"
    python main_unfair_evaluation_analysis.py --output-dir "./analysis_results"
        """
    )
    
    parser.add_argument(
        '--base-path', 
        type=str,
        default=r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard",
        help='Base path to BFCL benchmark data (default: current detected path)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for analysis results (default: same as base-path)'
    )
    
    parser.add_argument(
        '--models',
        type=str,
        nargs='*',
        default=None,
        help='Specific model names to analyze (default: all models)'
    )
    
    parser.add_argument(
        '--test-categories',
        type=str,
        nargs='*',
        default=None,
        help='Specific test categories to analyze (default: all categories)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    print("[BFCL] UNFAIR EVALUATION DETECTION SYSTEM")
    print("=" * 60)
    print(f"Base Path: {args.base_path}")
    if args.output_dir:
        print(f"Output Directory: {args.output_dir}")
    if args.models:
        print(f"Target Models: {', '.join(args.models)}")
    if args.test_categories:
        print(f"Target Categories: {', '.join(args.test_categories)}")
    print("=" * 60)
    
    try:
        # Step 1: 데이터 로딩
        print("\n[STEP 1] Loading BFCL Data")
        print("-" * 40)
        
        df = load_bfcl_results(args.base_path)
        
        if df is None or len(df) == 0:
            print("ERROR: No data loaded. Exiting.")
            return 1
            
        print(f"SUCCESS: Successfully loaded {len(df):,} evaluation records")
        
        # 필터링 (선택적)
        if args.models:
            if 'model_name' in df.columns:
                original_count = len(df)
                df = df[df['model_name'].isin(args.models)]
                print(f"🔍 Filtered to {len(df):,} records from specified models ({original_count - len(df):,} filtered out)")
            else:
                print("⚠️  Warning: model_name column not found, ignoring --models filter")
                
        if args.test_categories:
            if 'test_category' in df.columns:
                original_count = len(df)
                df = df[df['test_category'].isin(args.test_categories)]
                print(f"🔍 Filtered to {len(df):,} records from specified categories ({original_count - len(df):,} filtered out)")
            else:
                print("⚠️  Warning: test_category column not found, ignoring --test-categories filter")
        
        # 데이터 개요 출력
        print("\n📊 DATA OVERVIEW")
        print("-" * 40)
        print(f"Total Records: {len(df):,}")
        
        if 'model_name' in df.columns:
            model_counts = df['model_name'].value_counts()
            print(f"Models: {len(model_counts)} total")
            if args.verbose:
                for model, count in model_counts.head(10).items():
                    print(f"  - {model}: {count:,} records")
                    
        if 'test_category' in df.columns:
            category_counts = df['test_category'].value_counts()
            print(f"Test Categories: {len(category_counts)} total")
            if args.verbose:
                for category, count in category_counts.head(10).items():
                    print(f"  - {category}: {count:,} records")
        
        # Step 2: 불공정 평가 탐지
        print("\n🔍 STEP 2: Unfair Evaluation Detection")
        print("-" * 40)
        
        # 출력 디렉토리 설정
        if args.output_dir:
            output_base = Path(args.output_dir)
            output_base.mkdir(parents=True, exist_ok=True)
        else:
            output_base = Path(args.base_path)
            
        # 탐지기 초기화 (출력 경로 업데이트)
        detector = UnfairEvaluationDetector(df)
        detector.base_path = output_base
        
        # 탐지 실행
        detection_results = detector.classify_all_issues()
        
        # Step 3: 리포트 생성
        print("\n📄 STEP 3: Report Generation")
        print("-" * 40)
        
        report_results = detector.generate_unfair_evaluation_report()
        
        # Step 4: 결과 요약
        print("\n🎯 ANALYSIS COMPLETE!")
        print("=" * 60)
        
        print(f"📊 FINAL SUMMARY:")
        print(f"  Total Evaluations: {len(df):,}")
        print(f"  Unfair Evaluations: {report_results['total_unfair']:,} ({report_results['unfair_percentage']:.1f}%)")
        print(f"  Critical Issues (P0): {report_results['priority_fixes']:,}")
        
        if report_results['most_affected_models']:
            print(f"\n🔍 MOST AFFECTED MODELS:")
            for i, (model, unfair_rate) in enumerate(report_results['most_affected_models'], 1):
                print(f"  {i}. {model}: {unfair_rate*100:.1f}% unfair evaluations")
        
        print(f"\n💾 OUTPUT FILES SAVED TO: {output_base}")
        
        # 중요한 발견사항 하이라이트
        if report_results['priority_fixes'] > 0:
            print(f"\n🚨 ATTENTION REQUIRED:")
            print(f"   {report_results['priority_fixes']} critical P0 issues detected!")
            print(f"   These should be fixed immediately as they unfairly penalize model capabilities.")
            
        # Claude-Sonnet 특별 사례 체크
        if 'claude_sonnet_specific' in detection_results and detection_results['claude_sonnet_specific'] > 0:
            print(f"\n🔍 SPECIAL CASE DETECTED:")
            print(f"   {detection_results['claude_sonnet_specific']} Claude-Sonnet systematic issues found.")
            print(f"   This appears to be a configuration problem affecting all Claude-Sonnet tasks.")
        
        print(f"\n✅ Analysis completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        if args.verbose:
            print("\nFull traceback:")
            traceback.print_exc()
        return 1


def quick_test():
    """빠른 테스트를 위한 함수 (개발용)"""
    
    print("🧪 Running Quick Test...")
    
    try:
        # 최소한의 데이터로 테스트
        base_path = r"E:\Users\김현준\Downloads\agent_hard_benchmark_2\gorilla\berkeley-function-call-leaderboard"
        loader = BFCLDataLoader(base_path)
        
        # 구조 검증만
        is_valid = loader.validate_data_structure()
        print(f"Data structure valid: {is_valid}")
        
        if is_valid:
            # 하나의 모델만 로드해서 테스트
            df = load_bfcl_results(base_path)
            if len(df) > 0:
                # 첫 100개 레코드만으로 테스트
                test_df = df.head(100)
                detector = UnfairEvaluationDetector(test_df)
                
                # 간단한 탐지만 실행
                tech_errors = detector.detect_technical_errors()
                api_bias = detector.detect_api_configuration_bias()
                
                print(f"Technical errors found: {tech_errors}")
                print(f"API bias cases found: {api_bias}")
                print("✅ Quick test completed successfully!")
            else:
                print("❌ No data could be loaded")
        else:
            print("❌ Data structure validation failed")
            
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # 환경 변수로 테스트 모드 확인
    import os
    if os.getenv('BFCL_TEST_MODE') == '1':
        quick_test()
    else:
        exit_code = main()
        sys.exit(exit_code)