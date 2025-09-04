#!/usr/bin/env python3
"""
Quick run script for BFCL Unfair Evaluation Analysis
"""

import sys
from data_loader import load_bfcl_results
from unfair_evaluation_detector import UnfairEvaluationDetector

def main():
    print("BFCL Unfair Evaluation Analysis - Quick Run")
    print("=" * 60)
    
    try:
        # Load data
        print("\n[1] Loading data...")
        df = load_bfcl_results()
        
        if df is None or len(df) == 0:
            print("ERROR: No data loaded")
            return 1
            
        print(f"SUCCESS: Loaded {len(df):,} evaluation records")
        
        # Sample data for testing (use first 1000 records)
        if len(df) > 1000:
            print(f"Using sample of 1000 records for testing (out of {len(df):,})")
            df_sample = df.head(1000)
        else:
            df_sample = df
            
        print(f"Analyzing {len(df_sample):,} records...")
        
        # Initialize detector
        detector = UnfairEvaluationDetector(df_sample)
        
        # Run analysis
        print("\n[2] Running unfair evaluation detection...")
        results = detector.classify_all_issues()
        
        # Generate report
        print("\n[3] Generating reports...")
        report_results = detector.generate_unfair_evaluation_report()
        
        # Summary
        print(f"\n[4] ANALYSIS COMPLETE!")
        print(f"Total unfair evaluations: {report_results['total_unfair']:,}")
        print(f"Critical issues (P0): {report_results['priority_fixes']:,}")
        print(f"Unfair percentage: {report_results['unfair_percentage']:.1f}%")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)