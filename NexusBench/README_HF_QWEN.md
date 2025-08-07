# HuggingFace Qwen Models in NexusBench

This guide explains how to use local HuggingFace Qwen models (like Qwen2.5-7B-Instruct) with NexusBench.

## Prerequisites

1. **GPU Requirements**: 
   - NVIDIA GPU with at least 8GB VRAM for 7B models
   - NVIDIA GPU with at least 16GB VRAM for 13B+ models
   - CUDA 11.8 or later

2. **Python Dependencies**:
   ```bash
   pip install transformers accelerate torch vllm
   ```

## Quick Setup

1. **Install NexusBench with HF support**:
   ```bash
   cd NexusBench
   pip install -e .
   ```

2. **Setup your environment**:
   ```bash
   cp env.template .env
   python setup_hf_qwen.py
   ```

3. **Run a test benchmark**:
   ```bash
   nexusbench --benchmarks NVDLibraryBenchmark
   ```

## Manual Setup

If you prefer manual setup:

1. **Copy the environment template**:
   ```bash
   cp env.template .env
   ```

2. **Edit your `.env` file**:
   ```bash
   # API Configuration (not used for local models)
   API_KEY=not_used
   BASE_URL=not_used

   # Model Configuration
   CLIENT=HuggingFaceQwen
   MODEL=/path/to/your/Qwen2.5-7B-Instruct

   # Optional: Parallel Processing Configuration
   NUM_SAMPLES_PARALLEL=16  # Reduce for GPU memory constraints
   NUM_BENCHMARKS_PARALLEL=1
   ```

## Model Paths

### Local Models
- **Qwen2.5-7B-Instruct**: `/path/to/Qwen2.5-7B-Instruct`
- **Qwen2.5-14B-Instruct**: `/path/to/Qwen2.5-14B-Instruct`
- **Qwen2.5-32B-Instruct**: `/path/to/Qwen2.5-32B-Instruct`

### HuggingFace Hub Models
You can also use models directly from HuggingFace Hub:
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `Qwen/Qwen2.5-32B-Instruct`

## Performance Optimization

### GPU Memory Optimization

1. **Reduce batch size** in your `.env` file:
   ```bash
   NUM_SAMPLES_PARALLEL=8  # For 7B models
   NUM_SAMPLES_PARALLEL=4  # For 13B+ models
   ```

2. **Use vLLM for faster inference** (automatically detected):
   ```bash
   pip install vllm
   ```

3. **Adjust tensor parallelism** for multi-GPU setups:
   ```python
   # In HuggingFaceQwenClient.create_client()
   tensor_parallel_size=2  # For 2 GPUs
   ```

### Memory Requirements

| Model Size | VRAM Required | Recommended GPU |
|------------|---------------|-----------------|
| 7B         | 8-12GB       | RTX 3080/4080  |
| 14B        | 16-24GB      | RTX 4090/A100  |
| 32B        | 32-48GB      | A100/H100      |

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**:
   - Reduce `NUM_SAMPLES_PARALLEL`
   - Use smaller model variants
   - Enable gradient checkpointing

2. **Slow Inference**:
   - Install vLLM: `pip install vllm`
   - Use GPU with more VRAM
   - Reduce batch size

3. **Import Errors**:
   ```bash
   pip install transformers accelerate torch vllm
   ```

### Debug Mode

Enable debug mode to see detailed error messages:
```bash
DEBUG=true nexusbench --benchmarks NVDLibraryBenchmark
```

## Example Usage

### Basic Benchmark
```bash
nexusbench --benchmarks NVDLibraryBenchmark VirusTotalBenchmark
```

### Full Suite
```bash
nexusbench --suite all
```

### Custom Configuration
```bash
nexusbench \
  --client HuggingFaceQwen \
  --model /path/to/Qwen2.5-7B-Instruct \
  --benchmarks NVDLibraryBenchmark \
  --limit 10
```

## Advanced Configuration

### Custom Model Parameters

You can modify the model loading parameters in `nexusbench/clients.py`:

```python
# For vLLM
self.llm = LLM(
    model=self.model,
    trust_remote_code=True,
    tensor_parallel_size=1,  # Adjust for multi-GPU
    gpu_memory_utilization=0.8,  # Adjust memory usage
    max_model_len=8192,  # Adjust context length
)

# For Transformers
self.model = AutoModelForCausalLM.from_pretrained(
    self.model,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # Use bfloat16 for newer GPUs
    device_map="auto",
    load_in_8bit=True,  # For 8-bit quantization
)
```

### Function Calling Format

The HuggingFace Qwen client expects function calls in this format:
```
function_name(arg1=value1, arg2=value2)
```

Example:
```
search_vulnerabilities(query="CVE-2021-44228", limit=10)
```

## Performance Comparison

| Setup | Speed (tokens/sec) | Memory Usage | Quality |
|-------|-------------------|--------------|---------|
| vLLM (7B) | ~50-100 | 8-12GB | High |
| Transformers (7B) | ~20-40 | 8-12GB | High |
| vLLM (14B) | ~30-60 | 16-24GB | Higher |
| Transformers (14B) | ~10-25 | 16-24GB | Higher |

## Support

For issues specific to HuggingFace Qwen models:
1. Check the [Qwen GitHub repository](https://github.com/QwenLM/Qwen)
2. Check the [vLLM documentation](https://docs.vllm.ai/)
3. Check the [Transformers documentation](https://huggingface.co/docs/transformers/)

For NexusBench-specific issues, please open an issue in the NexusBench repository. 