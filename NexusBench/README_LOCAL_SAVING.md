# NexusBench Local Result Saving

This document explains how to use the new local result saving functionality in NexusBench.

## Overview

NexusBench now supports saving benchmark results locally instead of (or in addition to) uploading to Hugging Face Hub. This allows you to:

- Save results to your local filesystem
- Analyze results offline
- Keep results private
- Have better control over data storage

## Usage

### Command Line Interface

Add the `--output_dir` argument to save results locally:

```bash
python -m nexusbench.entrypoint \
    --client openai \
    --model gpt-4 \
    --suite per_task \
    --output_dir ./my_results \
    --limit 10
```

### Available Arguments

- `--output_dir`: Directory to save results (required for local saving)
- `--upload`: Still upload to Hugging Face Hub (optional)
- All other existing arguments work as before

## Output Structure

When you specify `--output_dir`, the following directory structure is created:

```
output_dir/
├── summary_YYYYMMDD_HHMMSS.json     # Overall summary
├── metrics_YYYYMMDD_HHMMSS.csv      # Metrics table
├── results/                          # Individual benchmark results
│   ├── LangChainMath_YYYYMMDD_HHMMSS.json
│   ├── VirusTotalBenchmark_YYYYMMDD_HHMMSS.json
│   └── ...
├── metrics/                          # Individual benchmark metrics
│   ├── LangChainMath_metrics_YYYYMMDD_HHMMSS.json
│   ├── VirusTotalBenchmark_metrics_YYYYMMDD_HHMMSS.json
│   └── ...
└── trajectories/                     # Conversation trajectories
    ├── LangChainMath_sample_0_YYYYMMDD_HHMMSS.json
    ├── LangChainMath_sample_1_YYYYMMDD_HHMMSS.json
    └── ...
```

## File Formats

### Summary File (`summary_*.json`)

Contains overall information about the benchmark run:

```json
{
  "timestamp": "20241201_143022",
  "model": "gpt-4",
  "client": "openai",
  "total_benchmarks": 2,
  "benchmarks": [
    {
      "benchmark_name": "LangChainMath",
      "metrics": {"Accuracy": 0.8},
      "time_taken_s": 45.2,
      "total_samples": 10,
      "successful_samples": 8,
      "error_samples": 2,
      "success_rate": 0.8
    }
  ]
}
```

### Metrics CSV (`metrics_*.csv`)

Comma-separated values file with benchmark metrics:

```csv
Benchmark,Accuracy,Max Turns Hit,Invalid Plan,Time (s),Total Samples,Successful Samples,Error Samples,Success Rate
LangChainMath,0.8000,0.2000,0.1000,45.20,10,8,2,0.8000
```

### Individual Benchmark Results (`results/*.json`)

Detailed results for each benchmark:

```json
{
  "benchmark_name": "LangChainMath",
  "model": "gpt-4",
  "client": "openai",
  "timestamp": "20241201_143022",
  "summary": {...},
  "detailed_results": [
    {
      "sample_index": 0,
      "final_accuracy": true,
      "ground_truth": "42",
      "max_turns_hit": false,
      "invalid_plan": false,
      "has_trajectory": true,
      "trajectory_length": 3
    }
  ]
}
```

### Trajectory Files (`trajectories/*.json`)

Complete conversation trajectories for each sample:

```json
{
  "benchmark_name": "LangChainMath",
  "sample_index": 0,
  "model": "gpt-4",
  "timestamp": "20241201_143022",
  "trajectory": [
    {
      "prompt": "System: You are a helpful assistant...",
      "query": "What is 6 * 7?",
      "raw_model_call": "multiply(6, 7)",
      "model_call": "multiply(6, 7)",
      "result": "42",
      "timestamp": "20241201_143022"
    }
  ]
}
```

## Examples

### Basic Usage

```bash
# Run a single benchmark with local saving
python -m nexusbench.entrypoint \
    --client openai \
    --model gpt-4 \
    --benchmarks LangChainMath \
    --output_dir ./results \
    --limit 5
```

### Multiple Benchmarks

```bash
# Run multiple benchmarks
python -m nexusbench.entrypoint \
    --client openai \
    --model gpt-4 \
    --benchmarks LangChainMath VirusTotalBenchmark \
    --output_dir ./multi_benchmark_results \
    --limit 10
```

### With Hugging Face Upload

```bash
# Save locally AND upload to Hugging Face
python -m nexusbench.entrypoint \
    --client openai \
    --model gpt-4 \
    --suite per_task \
    --output_dir ./local_results \
    --upload
```

### Using the Example Script

```bash
# Run the example script
python example_local_saving.py
```

## Analysis

The saved files can be used for:

- **Performance Analysis**: Use the CSV file for quick metrics comparison
- **Error Analysis**: Check individual sample results for patterns
- **Trajectory Analysis**: Study conversation flows in trajectory files
- **Reproducibility**: All inputs and outputs are preserved

## Tips

1. **Use meaningful output directories**: Include model name and date
   ```bash
   --output_dir ./results/gpt4_20241201
   ```

2. **Limit samples for testing**: Use `--limit` to test with fewer samples
   ```bash
   --limit 5
   ```

3. **Check file sizes**: Trajectory files can be large for complex benchmarks

4. **Backup important results**: Copy results to safe locations

## Troubleshooting

- **Permission errors**: Ensure the output directory is writable
- **Large files**: Trajectory files can be large; consider disk space
- **Memory issues**: For large benchmarks, consider reducing parallel processing

## Migration from Hugging Face Only

If you were previously using only Hugging Face upload:

1. Add `--output_dir` to your existing commands
2. Keep `--upload` if you still want Hugging Face uploads
3. Remove `--upload` if you only want local saving

The local saving functionality is designed to be non-breaking and can be used alongside existing Hugging Face upload functionality. 