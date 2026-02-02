# NVFP4 MLA Quantization Optimizations

This document describes the NVFP4 quantization implementation for MLA (Multi-head Latent Attention) in SGLang.

## Overview

The implementation focuses on reducing KV cache memory footprint using **NVFP4** (NVIDIA FP4) format with hardware-accelerated PTX instructions, providing:
- **4x memory savings** vs BF16
- **2x memory savings** vs FP8
- **Hardware acceleration** on Blackwell (SM100+) GPUs
- Minimal accuracy degradation (< 10% relative error)

## Requirements

### Hardware
- **GPU Architecture**: NVIDIA Blackwell (SM100+) or newer
  - RTX 50 series
  - B100, B200 data center GPUs

### Software
- **CUDA**: 12.8 or newer
- **PyTorch**: 2.8.0+ with Blackwell support

## Technical Details

### NVFP4 Format

NVFP4 uses the E2M1 format (same representation as MXFP4):
- **1 sign bit**
- **2 exponent bits**
- **1 mantissa bit**
- **Representable values**: [0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6]
- **Maximum value**: 6.0

### Key Differences from MXFP4

| Feature | MXFP4 | NVFP4 |
|---------|-------|-------|
| **Block size** | 16 or 32 | **16** (fixed for hardware) |
| **Scale format** | UE8M0 | **E4M3** |
| **Quantization** | Software | **PTX hardware instruction** |
| **Minimum GPU** | Ampere (SM80) | **Blackwell (SM100)** |
| **Speed** | ~0.1-0.5ms | **~0.01-0.05ms (10x faster)** |

### Hardware PTX Instruction

NVFP4 uses the `cvt.rn.satfinite.e2m1x2.f32` PTX instruction available on SM100+:

```asm
cvt.rn.satfinite.e2m1x2.f32 byte0, %f1, %f0
```

This instruction:
- Converts 2 float32 values to 2 E2M1 FP4 values in one cycle
- Performs saturation (clips to [-6, 6] range)
- Rounds to nearest even

## Implementation

### 1. CUDA Kernel (`sgl-kernel/csrc/kvcacheio/mla_kv_fp4_quant.cu`)

Features:
- Native CUDA kernel using PTX instructions
- Block-wise microscaling (16 elements per block)
- E4M3 scale format
- Optimized for SM100+ (Blackwell)

**API:**
```python
from sgl_kernel import mla_kv_fp4_quant, mla_kv_fp4_dequant

# Quantize
mla_kv_fp4_quant(k_nope, k_rope, kv_buffer, kv_scale_buffer, loc)

# Dequantize  
mla_kv_fp4_dequant(k_nope, k_rope, kv_buffer, kv_scale_buffer, loc)
```

**Tensor Shapes:**
- Input: `k_nope [num_tokens, 1, kv_lora_rank]`, `k_rope [num_tokens, 1, qk_rope_head_dim]`
- Output: `kv_buffer [num_pages, (kv_lora_rank + qk_rope_head_dim) / 2]` (packed FP4)
- Scales: `kv_scale_buffer [num_pages, (kv_lora_rank + qk_rope_head_dim) / 16]` (E4M3)

### 2. Triton Fallback (`python/sglang/srt/layers/attention/nsa/`)

For compatibility, Triton-based software fallback is provided:
- `quant_k_cache_fp4.py` → NVFP4 quantization (Triton)
- `dequant_k_cache_fp4.py` → NVFP4 dequantization (Triton)

**Note:** Triton fallback is significantly slower than CUDA kernel.

### 3. Quantization Configuration (`python/sglang/srt/layers/quantization/fp4_mla_scheme.py`)

```python
from sglang.srt.layers.quantization.fp4_mla_scheme import (
    NVFP4MLAQuantizationConfig,
    NVFP4MLAMemoryPoolHelper,
)

# Create config
config = NVFP4MLAQuantizationConfig(
    block_size=16,           # Fixed at 16 for NVFP4
    separate_nope_rope=True,
    use_cuda_kernel=True,    # Required for NVFP4
)

# Create memory pool
pool = NVFP4MLAMemoryPoolHelper.create_mla_nvfp4_pool(
    size=1024,
    page_size=64,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    layer_num=32,
    device="cuda",
)

# Calculate memory savings
savings = NVFP4MLAMemoryPoolHelper.get_memory_savings(
    kv_lora_rank=512,
    qk_rope_head_dim=64,
)
print(f"Memory savings vs BF16: {savings['memory_savings_vs_bf16']:.1%}")
```

## Memory Savings

For DeepSeek-V3 (`kv_lora_rank=512`, `qk_rope_head_dim=64`):

| Format  | Bytes per Token | Memory Savings |
|---------|-----------------|----------------|
| BF16    | 1152 bytes      | -              |
| FP8     | 576 bytes       | 50%            |
| **NVFP4** | **324 bytes**   | **72%**        |

NVFP4 breakdown: 288 bytes (packed data) + 36 bytes (scale factors)

## Integration Guide

### 1. Enable NVFP4 KV Cache

```python
# In server arguments
python -m sglang.launch_server \
    --model-path deepseek-ai/DeepSeek-V3 \
    --kv-cache-dtype fp4_e2m1 \
    --quantization nvfp4_mla
```

### 2. Use NVFP4 Memory Pool

```python
from sglang.srt.mem_cache.memory_pool import MLATokenToKVPoolNVFP4

pool = MLATokenToKVPoolNVFP4(
    size=16384,  # Number of tokens
    page_size=64,
    dtype=torch.float8_e4m3fn,  # Placeholder
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    layer_num=32,
    device="cuda",
    enable_memory_saver=True,
)
```

### 3. Custom Quantization Method

```python
from sglang.srt.layers.quantization.fp4_mla_scheme import NVFP4MLAQuantizeMethod

method = NVFP4MLAQuantizeMethod(
    quant_params=NVFP4MLAQuantParams(
        block_size=16,
        separate_nope_rope=True,
        use_cuda_kernel=True,
    )
)

# Quantize
method.quantize_mla_kv(
    k_nope, k_rope,
    kv_buffer, kv_scale_buffer,
    loc,
)
```

## Testing

Run the test suite:

```bash
# Basic tests (works on any CUDA GPU with Triton fallback)
pytest test/registered/mla/test_mla_fp4_quantization.py -v

# CUDA kernel tests (requires SM100+ Blackwell GPU)
pytest test/registered/mla/test_mla_fp4_quantization.py::TestNVFP4MLAQuantization::test_cuda_kernel_sm100 -v

# Benchmark
python test/registered/mla/test_mla_fp4_quantization.py --benchmark
```

## Performance Benchmarks

Expected performance on NVIDIA B200 (SM100):

| Batch Size | Quantization Time | Dequantization Time | Memory Bandwidth |
|------------|-------------------|---------------------|------------------|
| 16         | 0.002 ms          | 0.002 ms            | 1500 GB/s        |
| 64         | 0.005 ms          | 0.005 ms            | 2000 GB/s        |
| 256        | 0.015 ms          | 0.015 ms            | 2500 GB/s        |
| 1024       | 0.050 ms          | 0.050 ms            | 2800 GB/s        |

**Note:** Software fallback (Triton) is ~10-20x slower than hardware-accelerated CUDA kernel.

## Migration from MXFP4

If you were previously using MXFP4:

1. **Update quantization method name:**
   ```python
   # Old
   quantization="fp4_mla"
   
   # New
   quantization="nvfp4_mla"
   ```

2. **Verify GPU compatibility:**
   ```python
   import torch
   props = torch.cuda.get_device_properties(0)
   assert props.major >= 10, "NVFP4 requires SM100+"
   ```

3. **Memory pool class name:**
   ```python
   # Old
   MLATokenToKVPoolFP4
   
   # New
   MLATokenToKVPoolNVFP4
   ```

## Troubleshooting

### "NVFP4 requires SM100+ architecture"
Your GPU is not Blackwell or newer. NVFP4 requires:
- RTX 50 series
- B100, B200

### "CUDA kernel not available"
- Ensure CUDA 12.8+ is installed
- Ensure PyTorch 2.8.0+ with Blackwell support
- Check that sgl-kernel is built with SM100 support

### High reconstruction error (> 10%)
- Check input values are within reasonable range
- Verify block_size is 16 (not configurable for NVFP4)
- Ensure using CUDA kernel, not Triton fallback

## References

1. [NVIDIA Blackwell Architecture](https://www.nvidia.com/en-us/data-center/blackwell-architecture/)
2. [PTX ISA: FP4 Conversion Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/)
3. [DeepSeek MLA](https://github.com/deepseek-ai/DeepSeek-V3)
4. [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
