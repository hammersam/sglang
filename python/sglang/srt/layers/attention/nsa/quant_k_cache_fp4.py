# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
NVFP4 Quantization for NSA KV Cache

This module provides NVFP4 (NVIDIA FP4) quantization for Native Sparse Attention (NSA).
NVFP4 uses E2M1 format with hardware-accelerated PTX instructions on SM100+.

For KV Cache quantization:
- Block size: 16 elements
- Scale format: E4M3 (shared with NVFP4 weight quantization)
- Hardware PTX: cvt.rn.satfinite.e2m1x2.f32 (SM100+ only)

Note: This Triton implementation provides software fallback. For optimal performance
on Blackwell GPUs, use the native CUDA kernels in sgl-kernel.
"""

import torch
import triton
import triton.language as tl


# NVFP4 E2M1 constants
FP4_E2M1_MAX = 6.0
FP4_BLOCK_SIZE = 16  # NVFP4 block size (must be 16 for hardware compatibility)

# FP4 lookup table values: [0, 0.5, 1, 1.5, 2, 3, 4, 6]
# Used for software-based encoding/decoding


def quantize_k_cache_nvfp4(cache_k: torch.Tensor) -> torch.Tensor:
    """
    Quantize K cache to NVFP4 format for NSA.
    
    Args:
        cache_k: [num_blocks, block_size, h_k, d] BF16/FP16 tensor
                 where d = kv_lora_rank + qk_rope_head_dim (typically 512 + 64 = 576)
    
    Returns:
        Quantized tensor with shape [num_blocks, block_size, h_k, packed_dim]
        where packed_dim accounts for:
        - FP4 packed data (d / 2 bytes)
        - E4M3 scale factors (d / FP4_BLOCK_SIZE bytes)
    """
    return _quantize_k_cache_nvfp4_fast_wrapped(cache_k)


def quantize_k_cache_nvfp4_separate(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize k_nope and k_rope separately to NVFP4 format.
    
    Args:
        k_nope: [num_tokens, dim_nope] or [num_tokens, 1, dim_nope]
        k_rope: [num_tokens, dim_rope] or [num_tokens, 1, dim_rope]
    
    Returns:
        Tuple of (nope_part, rope_part) as uint8 tensors
        - nope_part: [num_tokens, 1, packed_bytes_nope]
        - rope_part: [num_tokens, 1, packed_bytes_rope]
    """
    k_nope_2d = k_nope.squeeze(1) if k_nope.ndim == 3 else k_nope
    k_rope_2d = k_rope.squeeze(1) if k_rope.ndim == 3 else k_rope
    
    return _quantize_k_cache_nvfp4_separate(k_nope_2d, k_rope_2d)


def _quantize_k_cache_nvfp4_fast_wrapped(
    input_k_cache: torch.Tensor,
    dim_nope: int = 512,
) -> torch.Tensor:
    """Wrapper to handle 4D input shape."""
    num_blocks, block_size, _, dim_total = input_k_cache.shape
    
    # Flatten to 2D for processing
    input_k_cache = input_k_cache.view(-1, dim_total)
    k_nope = input_k_cache[:, :dim_nope]
    k_rope = input_k_cache[:, dim_nope:]
    
    output = _quantize_k_cache_nvfp4_fast(k_nope, k_rope)
    
    # Reshape back to 4D
    return output.view(num_blocks, block_size, 1, -1)


def _quantize_k_cache_nvfp4_fast(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
) -> torch.Tensor:
    """
    Fast NVFP4 quantization using Triton kernel (software fallback).
    
    For optimal performance on SM100+, use the native CUDA kernel instead.
    
    Args:
        k_nope: [num_tokens, dim_nope]
        k_rope: [num_tokens, dim_rope]
    
    Returns:
        [num_tokens, packed_dim] uint8 tensor
    """
    assert k_nope.dtype in (torch.bfloat16, torch.float16)
    assert k_rope.dtype == k_nope.dtype
    
    num_tokens, dim_nope = k_nope.shape
    _, dim_rope = k_rope.shape
    
    # Calculate output size
    # FP4 data: (dim_nope + dim_rope) / 2 bytes (2 FP4 values per byte)
    # Scale factors: (dim_nope + dim_rope) / FP4_BLOCK_SIZE bytes
    total_dim = dim_nope + dim_rope
    packed_data_bytes = total_dim // 2
    scale_bytes = total_dim // FP4_BLOCK_SIZE
    output_bytes = packed_data_bytes + scale_bytes
    
    output = torch.empty(
        (num_tokens, output_bytes),
        dtype=torch.uint8,
        device=k_nope.device,
    )
    
    # Split output buffer
    output_data = output[:, :packed_data_bytes]
    output_scales = output[:, packed_data_bytes:]
    
    # Launch Triton kernel
    num_blocks_per_token = triton.cdiv(total_dim, FP4_BLOCK_SIZE)
    num_nope_blocks = dim_nope // FP4_BLOCK_SIZE
    
    _quantize_nvfp4_kernel[(num_tokens, num_blocks_per_token)](
        output_data,
        output_scales,
        k_nope,
        k_rope,
        output_data.stride(0),
        output_scales.stride(0),
        k_nope.stride(0),
        k_rope.stride(0),
        NUM_NOPE_BLOCKS=num_nope_blocks,
        BLOCK_SIZE=FP4_BLOCK_SIZE,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )
    
    return output


def _quantize_k_cache_nvfp4_separate(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize k_nope and k_rope separately to NVFP4.
    
    Returns:
        (nope_part_u8, rope_part_u8) each with shape [num_tokens, 1, packed_bytes]
    """
    num_tokens, dim_nope = k_nope.shape
    _, dim_rope = k_rope.shape
    
    k_nope = k_nope.contiguous()
    k_rope = k_rope.contiguous()
    
    # Calculate packed sizes
    nope_data_bytes = dim_nope // 2
    nope_scale_bytes = dim_nope // FP4_BLOCK_SIZE
    nope_total_bytes = nope_data_bytes + nope_scale_bytes
    
    rope_data_bytes = dim_rope // 2
    rope_scale_bytes = dim_rope // FP4_BLOCK_SIZE
    rope_total_bytes = rope_data_bytes + rope_scale_bytes
    
    # Allocate output buffers
    nope_part = torch.empty(
        (num_tokens, nope_total_bytes), dtype=torch.uint8, device=k_nope.device
    )
    rope_part = torch.empty(
        (num_tokens, rope_total_bytes), dtype=torch.uint8, device=k_rope.device
    )
    
    nope_data = nope_part[:, :nope_data_bytes]
    nope_scales = nope_part[:, nope_data_bytes:]
    rope_data = rope_part[:, :rope_data_bytes]
    rope_scales = rope_part[:, rope_data_bytes:]
    
    # Launch kernel for nope
    num_nope_blocks = triton.cdiv(dim_nope, FP4_BLOCK_SIZE)
    _quantize_nvfp4_kernel_nope_only[(num_tokens, num_nope_blocks)](
        nope_data,
        nope_scales,
        k_nope,
        nope_data.stride(0),
        nope_scales.stride(0),
        k_nope.stride(0),
        BLOCK_SIZE=FP4_BLOCK_SIZE,
        DIM_NOPE=dim_nope,
    )
    
    # Launch kernel for rope
    num_rope_blocks = triton.cdiv(dim_rope, FP4_BLOCK_SIZE)
    _quantize_nvfp4_kernel_rope_only[(num_tokens, num_rope_blocks)](
        rope_data,
        rope_scales,
        k_rope,
        rope_data.stride(0),
        rope_scales.stride(0),
        k_rope.stride(0),
        BLOCK_SIZE=FP4_BLOCK_SIZE,
        DIM_ROPE=dim_rope,
    )
    
    # Add middle dimension for compatibility
    return nope_part.unsqueeze(1), rope_part.unsqueeze(1)


@triton.jit
def _quantize_nvfp4_kernel(
    output_data_ptr,
    output_scales_ptr,
    k_nope_ptr,
    k_rope_ptr,
    output_data_stride: int,
    output_scales_stride: int,
    k_nope_stride: int,
    k_rope_stride: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    """Triton kernel for NVFP4 quantization (software fallback)."""
    token_id = tl.program_id(0)
    block_id = tl.program_id(1)
    
    # Determine if we're processing nope or rope
    if block_id < NUM_NOPE_BLOCKS:
        # Process nope block
        block_start = block_id * BLOCK_SIZE
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < DIM_NOPE
        
        ptr = k_nope_ptr + token_id * k_nope_stride + offs
        values = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
    else:
        # Process rope block
        rope_block_id = block_id - NUM_NOPE_BLOCKS
        block_start = rope_block_id * BLOCK_SIZE
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < DIM_ROPE
        
        ptr = k_rope_ptr + token_id * k_rope_stride + offs
        values = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
    
    # Compute scale factor (max abs value / FP4 max)
    max_abs = tl.max(tl.abs(values))
    
    # Compute E4M3 scale (NVFP4 uses E4M3, not UE8M0)
    # scale = max_abs / FP4_E2M1_MAX, clamp to E4M3 range
    scale_val = max_abs / FP4_E2M1_MAX
    scale_exp = tl.log2(scale_val + 1e-10)
    # E4M3: 1 sign bit, 4 exp bits, 3 mantissa bits
    # Bias is 7, max exp is 15 (biased to 15), clamp to valid range
    scale_e4m3 = tl.clamp(tl.ceil(scale_exp) + 7, 0, 255).to(tl.uint8)
    
    # Apply scale and quantize
    scale_f = tl.exp2(scale_exp)
    scaled_vals = values / scale_f
    
    # Quantize to FP4 (E2M1) - software implementation
    # Sign bit: 1 bit, Magnitude: 3 bits
    abs_vals = tl.abs(scaled_vals)
    signs = (scaled_vals < 0).to(tl.int32)
    
    # Map to FP4 magnitude values [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    mag = tl.where(abs_vals < 0.25, 0,
          tl.where(abs_vals < 0.75, 1,
          tl.where(abs_vals < 1.25, 2,
          tl.where(abs_vals < 1.75, 3,
          tl.where(abs_vals < 2.5, 4,
          tl.where(abs_vals < 3.5, 5,
          tl.where(abs_vals < 5.0, 6, 7)))))))
    
    fp4_vals = (signs << 3) | mag
    
    # Pack two FP4 values into one byte
    packed_vals = tl.zeros((BLOCK_SIZE // 2,), dtype=tl.uint8)
    for i in range(0, BLOCK_SIZE, 2):
        lo = tl.sum(tl.where(tl.arange(0, BLOCK_SIZE) == i, fp4_vals, 0))
        hi = tl.sum(tl.where(tl.arange(0, BLOCK_SIZE) == i + 1, fp4_vals, 0))
        packed_idx = i // 2
        packed_vals = tl.where(
            tl.arange(0, BLOCK_SIZE // 2) == packed_idx,
            (hi << 4) | (lo & 0xF),
            packed_vals
        )
    
    # Store packed data
    data_offs = (block_id * BLOCK_SIZE) // 2 + tl.arange(0, BLOCK_SIZE // 2)
    data_mask = data_offs < (DIM_NOPE + DIM_ROPE) // 2
    data_ptr = output_data_ptr + token_id * output_data_stride + data_offs
    tl.store(data_ptr, packed_vals, mask=data_mask)
    
    # Store scale
    scale_ptr = output_scales_ptr + token_id * output_scales_stride + block_id
    tl.store(scale_ptr, scale_e4m3)


@triton.jit
def _quantize_nvfp4_kernel_nope_only(
    output_data_ptr,
    output_scales_ptr,
    k_nope_ptr,
    output_data_stride: int,
    output_scales_stride: int,
    k_nope_stride: int,
    BLOCK_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
):
    """NVFP4 quantization kernel for k_nope only."""
    token_id = tl.program_id(0)
    block_id = tl.program_id(1)
    
    block_start = block_id * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < DIM_NOPE
    
    ptr = k_nope_ptr + token_id * k_nope_stride + offs
    values = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
    
    # Compute scale
    max_abs = tl.max(tl.abs(values))
    scale_val = max_abs / FP4_E2M1_MAX
    scale_exp = tl.log2(scale_val + 1e-10)
    scale_e4m3 = tl.clamp(tl.ceil(scale_exp) + 7, 0, 255).to(tl.uint8)
    
    # Quantize
    scale_f = tl.exp2(scale_exp)
    scaled_vals = values / scale_f
    
    abs_vals = tl.abs(scaled_vals)
    signs = (scaled_vals < 0).to(tl.int32)
    mag = tl.where(abs_vals < 0.25, 0,
          tl.where(abs_vals < 0.75, 1,
          tl.where(abs_vals < 1.25, 2,
          tl.where(abs_vals < 1.75, 3,
          tl.where(abs_vals < 2.5, 4,
          tl.where(abs_vals < 3.5, 5,
          tl.where(abs_vals < 5.0, 6, 7)))))))
    
    fp4_vals = (signs << 3) | mag
    
    # Pack
    packed_vals = tl.zeros((BLOCK_SIZE // 2,), dtype=tl.uint8)
    for i in range(0, BLOCK_SIZE, 2):
        lo = tl.sum(tl.where(tl.arange(0, BLOCK_SIZE) == i, fp4_vals, 0))
        hi = tl.sum(tl.where(tl.arange(0, BLOCK_SIZE) == i + 1, fp4_vals, 0))
        packed_idx = i // 2
        packed_vals = tl.where(
            tl.arange(0, BLOCK_SIZE // 2) == packed_idx,
            (hi << 4) | (lo & 0xF),
            packed_vals
        )
    
    # Store
    data_offs = (block_id * BLOCK_SIZE) // 2 + tl.arange(0, BLOCK_SIZE // 2)
    data_mask = data_offs < DIM_NOPE // 2
    data_ptr = output_data_ptr + token_id * output_data_stride + data_offs
    tl.store(data_ptr, packed_vals, mask=data_mask)
    
    scale_ptr = output_scales_ptr + token_id * output_scales_stride + block_id
    tl.store(scale_ptr, scale_e4m3)


@triton.jit
def _quantize_nvfp4_kernel_rope_only(
    output_data_ptr,
    output_scales_ptr,
    k_rope_ptr,
    output_data_stride: int,
    output_scales_stride: int,
    k_rope_stride: int,
    BLOCK_SIZE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    """NVFP4 quantization kernel for k_rope only."""
    token_id = tl.program_id(0)
    block_id = tl.program_id(1)
    
    block_start = block_id * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < DIM_ROPE
    
    ptr = k_rope_ptr + token_id * k_rope_stride + offs
    values = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
    
    # Compute scale
    max_abs = tl.max(tl.abs(values))
    scale_val = max_abs / FP4_E2M1_MAX
    scale_exp = tl.log2(scale_val + 1e-10)
    scale_e4m3 = tl.clamp(tl.ceil(scale_exp) + 7, 0, 255).to(tl.uint8)
    
    # Quantize
    scale_f = tl.exp2(scale_exp)
    scaled_vals = values / scale_f
    
    abs_vals = tl.abs(scaled_vals)
    signs = (scaled_vals < 0).to(tl.int32)
    mag = tl.where(abs_vals < 0.25, 0,
          tl.where(abs_vals < 0.75, 1,
          tl.where(abs_vals < 1.25, 2,
          tl.where(abs_vals < 1.75, 3,
          tl.where(abs_vals < 2.5, 4,
          tl.where(abs_vals < 3.5, 5,
          tl.where(abs_vals < 5.0, 6, 7)))))))
    
    fp4_vals = (signs << 3) | mag
    
    # Pack
    packed_vals = tl.zeros((BLOCK_SIZE // 2,), dtype=tl.uint8)
    for i in range(0, BLOCK_SIZE, 2):
        lo = tl.sum(tl.where(tl.arange(0, BLOCK_SIZE) == i, fp4_vals, 0))
        hi = tl.sum(tl.where(tl.arange(0, BLOCK_SIZE) == i + 1, fp4_vals, 0))
        packed_idx = i // 2
        packed_vals = tl.where(
            tl.arange(0, BLOCK_SIZE // 2) == packed_idx,
            (hi << 4) | (lo & 0xF),
            packed_vals
        )
    
    # Store
    data_offs = (block_id * BLOCK_SIZE) // 2 + tl.arange(0, BLOCK_SIZE // 2)
    data_mask = data_offs < DIM_ROPE // 2
    data_ptr = output_data_ptr + token_id * output_data_stride + data_offs
    tl.store(data_ptr, packed_vals, mask=data_mask)
    
    scale_ptr = output_scales_ptr + token_id * output_scales_stride + block_id
    tl.store(scale_ptr, scale_e4m3)
