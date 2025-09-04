# BFCL Benchmark Analysis - Corrected Results

**Date**: September 2, 2025
**Total Records Analyzed**: 69,604
**Models**: 16
**Test Categories**: 17

## Executive Summary

Based on corrected column names and proper data type handling:

- **Success Rate**: 81.5% of evaluations completed successfully
- **Input Token Coverage**: 81.5% of records have input tokens
- **Output Token Coverage**: 81.5% of records have output tokens
- **Normal Latency**: 81.5% of records have latency > 1s

## Key Findings

### ✅ Infrastructure Generally Functional

The analysis shows:
- Majority of evaluations completed successfully
- Token tracking and latency measurements working
- Some optimization opportunities may exist

## Model-Specific Results

| Model | Records | Input OK | Output OK | Latency OK |
|-------|---------|----------|-----------|------------|
| anthropic_claude-4-sonnet-thin | 4441 | 81.0% | 81.0% | 81.0% |
| anthropic_claude-4-sonnet-thin | 4441 | 81.7% | 81.7% | 81.7% |
| deepseek-ai_DeepSeek-R1-0528 | 4441 | 82.0% | 82.0% | 82.0% |
| deepseek-ai_DeepSeek-V3-0324 | 4441 | 82.0% | 82.0% | 82.0% |
| openai_gpt-4.1 | 4441 | 82.0% | 82.0% | 82.0% |
| openai_gpt-4o-20240806 | 4441 | 82.0% | 82.0% | 82.0% |
| openai_gpt-4o-mini | 4441 | 82.0% | 81.9% | 82.0% |
| openai_o3-high | 4441 | 81.1% | 81.1% | 81.1% |
| openai_o4-mini-high | 4441 | 81.0% | 81.0% | 81.0% |
| togetherai_moonshotai_Kimi-K2- | 4441 | 77.1% | 77.1% | 77.1% |

## Recommendations

1. **Optimize Failed Cases**: Investigate and fix specific failure patterns
2. **Performance Monitoring**: Implement continuous monitoring
3. **Data Quality**: Enhance data validation and error handling
