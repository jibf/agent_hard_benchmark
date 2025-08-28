# AgentHard Dataset Downloader

This script downloads multiple AgentHard evaluation datasets from Hugging Face in one go, with optional file filtering capabilities.

## Features

- **Batch Download**: Download all six AgentHard evaluation datasets at once
- **Selective Download**: Choose specific repositories to download
- **File Filtering**: Download only specific file types (e.g., only JSONL files)
- **Custom Output Directory**: Specify where to save the datasets
- **Error Handling**: Graceful error handling with detailed error messages
- **Resume Support**: Automatically resumes interrupted downloads

## Prerequisites

1. **Python 3.6+** installed
2. **huggingface_hub** library installed:
   ```bash
   pip install -U huggingface_hub
   ```
3. **Hugging Face Authentication** (optional, for gated datasets):
   ```bash
   huggingface-cli login
   ```
   *Note: All AgentHard datasets are public and don't require authentication*

## Usage

### Basic Usage

Download all six AgentHard evaluation datasets to the current directory:

```bash
python3 download_agenthard.py
```

### Download Specific Repositories

Download only specific datasets:

```bash
python3 download_agenthard.py --repo AgentHard/NexusBench-evaluation --repo AgentHard/DrafterBench-evaluation
```

### Filter Files by Type

Download only JSONL files from all datasets:

```bash
python3 download_agenthard.py --patterns "*.jsonl"
```

Download only CSV files from scores directories:

```bash
python3 download_agenthard.py --patterns "scores/*.csv"
```

### Custom Output Directory

Download to a specific directory:

```bash
python3 download_agenthard.py --out ./datasets
```

### Combined Options

Download specific repositories with file filtering to a custom directory:

```bash
python3 download_agenthard.py --repo AgentHard/NexusBench-evaluation --patterns "*.jsonl" --out ./my_datasets
```

## Available Datasets

The script downloads these six AgentHard evaluation datasets by default:

1. **AgentHard/NexusBench-evaluation** - Multi-domain agent evaluation
2. **AgentHard/DrafterBench-evaluation** - Code generation evaluation
3. **AgentHard/complex-func-bench-evaluation** - Complex function calling evaluation
4. **AgentHard/multi_challenge-evaluation** - Multi-challenge evaluation
5. **AgentHard/BFCL-evaluation** - Benchmark for Function Calling Language
6. **AgentHard/ToolSandbox-evaluation** - Tool usage evaluation

## Output Structure

Each dataset is downloaded to a separate directory named after the repository suffix:

```
output_directory/
├── NexusBench-evaluation/
│   ├── *.jsonl files
│   └── other files...
├── DrafterBench-evaluation/
│   ├── *.jsonl files
│   └── other files...
└── ...
```

## Command Line Options

- `--repo REPO`: HF dataset repo_id (can be repeated for multiple repos)
- `--patterns PATTERNS`: Glob patterns to filter files (e.g., "*.jsonl", "scores/*.csv")
- `--out OUT`: Base output directory (default: current directory)
- `-h, --help`: Show help message

## Examples

### Download all datasets with only JSONL files
```bash
python3 download_agenthard.py --patterns "*.jsonl"
```

### Download specific datasets to a custom directory
```bash
python3 download_agenthard.py --repo AgentHard/NexusBench-evaluation --repo AgentHard/DrafterBench-evaluation --out ./agent_datasets
```

### Download all files from a single dataset
```bash
python3 download_agenthard.py --repo AgentHard/NexusBench-evaluation
```

## Troubleshooting

### Authentication Issues
If you encounter authentication errors for gated datasets:
1. Run `huggingface-cli login`
2. Enter your Hugging Face token
3. Retry the download

## Script Files

- `download_agenthard.py`

