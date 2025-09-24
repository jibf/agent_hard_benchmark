# Multi-GPU Support for Embedding Models

This document explains how to use the new multi-GPU support for embedding models in the benchmark filtering pipeline.

## Overview

The pipeline now supports running embedding models across multiple GPUs, which can significantly improve performance when:
- Using large embedding models (>1B parameters)
- Processing large datasets
- Working with models that don't fit in single GPU memory

## Requirements

- Multiple NVIDIA GPUs with CUDA support
- PyTorch with CUDA support
- HuggingFace transformers and accelerate libraries
- SentenceTransformer library

## Usage

### Basic Multi-GPU Usage

```bash
python main.py --use-multi-gpu-embedding --target-benchmark complex-func-bench
```

### Advanced Configuration

```bash
python main.py \
    --use-multi-gpu-embedding \
    --embedding-device-map auto \
    --embedding-batch-size 32 \
    --embedding-model Qwen/Qwen3-Embedding-8B \
    --target-benchmark complex-func-bench
```

## Command Line Arguments

### `--use-multi-gpu-embedding`
- **Type**: Flag (boolean)
- **Description**: Enables multi-GPU support for the embedding model
- **Default**: False
- **Requirements**: At least 2 GPUs available

### `--embedding-device-map`
- **Type**: String
- **Description**: Specifies how to distribute the model across GPUs
- **Options**:
  - `auto` (default): Automatically distributes layers across available GPUs
  - `balanced`: Balances model layers across GPUs with equal memory usage
  - `balanced_low_0`: Like balanced but keeps some layers on GPU 0
- **Default**: `auto` when multi-GPU is enabled

## Device Map Options Explained

### `auto`
- **Best for**: Most use cases
- **Behavior**: Automatically determines optimal layer distribution
- **Memory**: Balances memory usage across GPUs
- **Performance**: Generally optimal for most models

### `balanced`
- **Best for**: Models with uniform layer sizes
- **Behavior**: Distributes layers evenly across all GPUs
- **Memory**: Equal memory usage on each GPU
- **Performance**: Good for models with consistent layer sizes

### `balanced_low_0`
- **Best for**: Very large models or when GPU 0 has more memory
- **Behavior**: Like balanced but keeps some layers on GPU 0
- **Memory**: Slightly more memory usage on GPU 0
- **Performance**: Good for models that benefit from keeping some layers on the primary GPU

## Performance Optimization Tips

1. **Increase Batch Size**: When using multi-GPU, increase the embedding batch size:
   ```bash
   --embedding-batch-size 32  # or higher depending on GPU memory
   ```

2. **Monitor GPU Usage**: Use `nvidia-smi` to monitor GPU memory and utilization:
   ```bash
   watch -n 1 nvidia-smi
   ```

3. **Choose Appropriate Model**: Multi-GPU is most beneficial for large models:
   - Models >1B parameters see significant speedup
   - Smaller models may not benefit much from multi-GPU

4. **Memory Considerations**: Ensure all GPUs have similar memory capacity for optimal performance

## Example Scripts

### Basic Multi-GPU Example
```bash
python main.py \
    --use-multi-gpu-embedding \
    --target-benchmark complex-func-bench \
    --embedding-batch-size 16
```

### High-Performance Multi-GPU Example
```bash
python main.py \
    --use-multi-gpu-embedding \
    --embedding-device-map balanced \
    --embedding-batch-size 64 \
    --embedding-model Qwen/Qwen3-Embedding-8B \
    --target-benchmark complex-func-bench \
    --skip-visualization
```

### Custom Device Map Example
```bash
python main.py \
    --use-multi-gpu-embedding \
    --embedding-device-map balanced_low_0 \
    --embedding-batch-size 32 \
    --target-benchmark complex-func-bench
```

## Troubleshooting

### Common Issues

1. **"Only 1 GPU available" Warning**
   - **Cause**: Multi-GPU requested but only 1 GPU detected
   - **Solution**: Pipeline automatically falls back to single GPU mode

2. **CUDA Out of Memory**
   - **Cause**: Batch size too large for available GPU memory
   - **Solution**: Reduce `--embedding-batch-size` or use `balanced_low_0` device map

3. **Slow Performance**
   - **Cause**: Model too small to benefit from multi-GPU
   - **Solution**: Use larger embedding model or stick with single GPU

### Debugging

Enable verbose logging to see GPU usage:
```bash
python main.py --use-multi-gpu-embedding --target-benchmark complex-func-bench 2>&1 | grep -i gpu
```

Check GPU memory usage during execution:
```bash
nvidia-smi -l 1  # Updates every second
```

## Implementation Details

The multi-GPU support is implemented using:
- HuggingFace's `device_map` functionality
- SentenceTransformer's support for model parallelism
- Automatic fallback to single GPU when needed
- Proper tensor movement to CPU for computation

## Performance Benchmarks

Expected speedup with multi-GPU (approximate):
- 2 GPUs: 1.5-1.8x speedup
- 4 GPUs: 2.5-3.2x speedup
- 8 GPUs: 4.0-5.5x speedup

*Actual performance depends on model size, batch size, and GPU specifications.*

## Limitations

1. **Model Size**: Very small models may not benefit from multi-GPU
2. **Memory**: All GPUs should have similar memory capacity
3. **Communication Overhead**: Some overhead exists for inter-GPU communication
4. **Batch Size**: Very small batch sizes may not utilize multiple GPUs effectively

## Future Improvements

Potential enhancements for future versions:
- Dynamic batch size adjustment based on GPU memory
- Support for custom device mapping configurations
- Integration with distributed training frameworks
- Automatic performance profiling and optimization
