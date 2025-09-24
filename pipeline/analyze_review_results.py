#!/usr/bin/env python3
"""
Analyze the review results to find samples where LLM agrees/disagrees with the original judgment.
"""

import json
import re
from typing import Dict, List, Any
from collections import Counter

def load_review_results(review_file: str) -> List[Dict[str, Any]]:
    """Load review results from JSONL file."""
    results = []
    
    with open(review_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                result = json.loads(line.strip())
                results.append(result)
            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
    
    return results

def analyze_llm_review(llm_review: str) -> Dict[str, Any]:
    """Analyze the LLM review to determine agreement/disagreement."""
    
    # Check for API errors
    if "Error:" in llm_review or "error" in llm_review.lower():
        return {
            "status": "error",
            "agreement": "unknown",
            "reasoning": "API error occurred"
        }
    
    # Look for agreement indicators
    agreement_indicators = [
        "agree", "correct", "valid", "appropriate", "well-justified", 
        "reasonable", "logical", "no issues", "not flawed", "doesn't have issues"
    ]
    
    disagreement_indicators = [
        "disagree", "incorrect", "invalid", "inappropriate", "flawed", 
        "has issues", "problem", "error", "wrong", "should be"
    ]
    
    review_lower = llm_review.lower()
    
    # Count agreement and disagreement indicators
    agreement_count = sum(1 for indicator in agreement_indicators if indicator in review_lower)
    disagreement_count = sum(1 for indicator in disagreement_indicators if indicator in review_lower)
    
    # Determine agreement status
    if agreement_count > disagreement_count:
        agreement = "agree"
    elif disagreement_count > agreement_count:
        agreement = "disagree"
    else:
        agreement = "neutral"
    
    return {
        "status": "success",
        "agreement": agreement,
        "agreement_count": agreement_count,
        "disagreement_count": disagreement_count,
        "reasoning": llm_review[:200] + "..." if len(llm_review) > 200 else llm_review
    }

def main():
    review_file = "non_flawed_tasks_review_sample_80.jsonl"
    
    print(f"Loading review results from: {review_file}")
    results = load_review_results(review_file)
    
    print(f"Found {len(results)} review results")
    print()
    
    # Analyze each result
    analysis_results = []
    error_count = 0
    agree_count = 0
    disagree_count = 0
    neutral_count = 0
    
    for result in results:
        llm_review = result.get('llm_review', '')
        analysis = analyze_llm_review(llm_review)
        
        analysis_results.append({
            "question_id": result.get('question_id', 'Unknown'),
            "task_name": result.get('task_name', 'Unknown'),
            "original_judgment": result.get('llm_judge_output', {}).get('specific_filter', {}).get('is_flawed', 'Unknown'),
            "llm_review": llm_review,
            "analysis": analysis
        })
        
        if analysis["status"] == "error":
            error_count += 1
        elif analysis["agreement"] == "agree":
            agree_count += 1
        elif analysis["agreement"] == "disagree":
            disagree_count += 1
        else:
            neutral_count += 1
    
    # Print summary
    print("=== REVIEW ANALYSIS SUMMARY ===")
    print(f"Total tasks reviewed: {len(results)}")
    print(f"API errors: {error_count}")
    print(f"LLM agrees with original judgment: {agree_count}")
    print(f"LLM disagrees with original judgment: {disagree_count}")
    print(f"LLM neutral/uncertain: {neutral_count}")
    print()
    
    # Show samples where LLM agrees (no issues)
    print("=== SAMPLES WHERE LLM AGREES TASK HAS NO ISSUES ===")
    agree_samples = [r for r in analysis_results if r["analysis"]["agreement"] == "agree"]
    
    if agree_samples:
        print(f"Found {len(agree_samples)} samples where LLM agrees:")
        print()
        
        for i, sample in enumerate(agree_samples[:5], 1):  # Show first 5
            print(f"{i}. Task: {sample['task_name']} ({sample['question_id']})")
            print(f"   Original judgment: {sample['original_judgment']}")
            print(f"   LLM review: {sample['analysis']['reasoning']}")
            print()
    else:
        print("No samples found where LLM clearly agrees.")
    
    # Show samples where LLM disagrees
    print("=== SAMPLES WHERE LLM DISAGREES (THINKS TASK HAS ISSUES) ===")
    disagree_samples = [r for r in analysis_results if r["analysis"]["agreement"] == "disagree"]
    
    if disagree_samples:
        print(f"Found {len(disagree_samples)} samples where LLM disagrees:")
        print()
        
        for i, sample in enumerate(disagree_samples[:5], 1):  # Show first 5
            print(f"{i}. Task: {sample['task_name']} ({sample['question_id']})")
            print(f"   Original judgment: {sample['original_judgment']}")
            print(f"   LLM review: {sample['analysis']['reasoning']}")
            print()
    else:
        print("No samples found where LLM clearly disagrees.")
    
    # Show error samples
    if error_count > 0:
        print("=== SAMPLES WITH API ERRORS ===")
        error_samples = [r for r in analysis_results if r["analysis"]["status"] == "error"]
        
        for i, sample in enumerate(error_samples[:3], 1):  # Show first 3
            print(f"{i}. Task: {sample['task_name']} ({sample['question_id']})")
            print(f"   Error: {sample['llm_review']}")
            print()

if __name__ == "__main__":
    main()
