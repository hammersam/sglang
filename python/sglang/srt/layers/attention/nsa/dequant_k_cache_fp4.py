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
NVFP4 Dequantization for NSA KV Cache

Software fallback dequantization for NVFP4 format.
For optimal performance on SM100+, use native CUDA kernels.
"""

import torch
import triton
import triton.language as tl


FP4_E2M1_MAX = 6.0
FP4_BLOCK_SIZE = 16


@triton.jit
def _dequantize_nvfp4_kernel(
    output_ptr,
    input_data_ptr,
    input_scales_ptr,
    output_stride: int,
    input_data_stride: int,
    input_scales_stride: int,
    DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for NVFP4 dequantization (software fallback)."""
    token_id = tl.program_id(0)
    block_id = tl.program_id(1)
    
    block_start = block_id * BLOCK_SIZE
    offs = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offs < DIM
    
    # Load scale factor (E4M3 format for NVFP4)
    scale_ptr = input_scales_ptr + token_id * input_scales_stride + block_id
    scale_e4m3 = tl.load(scale_ptr).to(tl.int32)
    # E4M3 bias is 7
    scale_exp = scale_e4m3 - 7
    scale = tl.exp2(scale_exp.to(tl.float32))
    
    # Load packed FP4 data
    packed_idx = block_start // 2 + tl.arange(0, BLOCK_SIZE // 2)
    packed_mask = packed_idx < DIM // 2
    data_ptr = input_data_ptr + token_id * input_data_stride + packed_idx
    packed_vals = tl.load(data_ptr, mask=packed_mask, other=0)
    
    # Unpack FP4 values
    lo = packed_vals & 0xF
    hi = (packed_vals >> 4) & 0xF
    sign_lo = (lo >> 3) & 1
    sign_hi = (hi >> 3) & 1
    mag_lo = lo & 0x7
    mag_hi = hi & 0x7
    
    # FP4 lookup table: [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    fp4_table = tl.full((8,), 0.0, dtype=tl.float32)
    fp4_table = tl.where(tl.arange(0, 8) == 0, 0.0, fp4_table)
    fp4_table = tl.where(tl.arange(0, 8) == 1, 0.5, fp4_table)
    fp4_table = tl.where(tl.arange(0, 8) == 2, 1.0, fp4_table)
    fp4_table = tl.where(tl.arange(0, 8) == 3, 1.5, fp4_table)
    fp4_table = tl.where(tl.arange(0, 8) == 4, 2.0, fp4_table)
    fp4_table = tl.where(tl.arange(0, 8) == 5, 3.0, fp4_table)
    fp4_table = tl.where(tl.arange(0, 8) == 6, 4.0, fp4_table)
    fp4_table = tl.where(tl.arange(0, 8) == 7, 6.0, fp4_table)
    
    # Dequantize
    dequant_lo = fp4_table[mag_lo] * scale
    dequant_hi = fp4_table[mag_hi] * scale
    
    # Apply sign
    dequant_lo = tl.where(sign_lo == 1, -dequant_lo, dequant_lo)
    dequant_hi = tl.where(sign_hi == 1, -dequant_hi, dequant_hi)
    
    # Store output
    out_ptr_lo = output_ptr + token_id * output_stride + block_start + tl.arange(0, BLOCK_SIZE, 2)
    out_ptr_hi = output_ptr + token_id * output_stride + block_start + tl.arange(1, BLOCK_SIZE, 2)
    
    tl.store(out_ptr_lo, dequant_lo.to(output_ptr.dtype.element_ty), mask=mask & (tl.arange(0, BLOCK_SIZE) % 2 == 0))
    tl.store(out_ptr_hi, dequant_hi.to(output_ptr.dtype.element_ty), mask=mask & (tl.arange(0, BLOCK_SIZE) % 2 == 1))


def dequantize_k_cache_nvfp4(
    quantized: torch.Tensor,
    dim_nope: int = 512,
    dim_rope: int = 64,
) -> torch.Tensor:
    """
    Dequantize NVFP4 K cache.
    
    Args:
        quantized: [num_tokens, packed_bytes] uint8 tensor
        dim_nope: Dimension of k_nope
        dim_rope: Dimension of k_rope
    
    Returns:
        [num_tokens, dim_nope + dim_rope] BF16/FP16 tensor
    """
    num_tokens = quantized.shape[0]
    total_dim = dim_nope + dim_rope
    
    packed_data_bytes = total_dim // 2
    scale_bytes = total_dim // FP4_BLOCK_SIZE
    
    data = quantized[:, :packed_data_bytes]
    scales = quantized[:, packed_data_bytes:]
    
    output = torch.empty(
        (num_tokens, total_dim),
        dtype=torch.bfloat16,
        device=quantized.device,
    )
    
    num_blocks = triton.cdiv(total_dim, FP4_BLOCK_SIZE)
    
    _dequantize_nvfp4_kernel[(num_tokens, num_blocks)](
        output,
        data,
        scales,
        output.stride(0),
        data.stride(0),
        scales.stride(0),
        DIM=total_dim,
        BLOCK_SIZE=FP4_BLOCK_SIZE,
    )
    
    return output


def dequantize_k_cache_nvfp4_separate(
    nope_part: torch.Tensor,
    rope_part: torch.Tensor,
    dim_nope: int = 512,
    dim_rope: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dequantize NVFP4 K cache (separate nope and rope).
    
    Args:
        nope_part: [num_tokens, packed_bytes_nope] uint8 tensor
        rope_part: [num_tokens, packed_bytes_rope] uint8 tensor
    
    Returns:
        (k_nope, k_rope) as BF16/FP16 tensors
    """
    num_tokens = nope_part.shape[0]
    
    # Dequantize nope
    nope_data_bytes = dim_nope // 2
    nope_scale_bytes = dim_nope // FP4_BLOCK_SIZE
    
    nope_data = nope_part[:, :nope_data_bytes]
    nope_scales = nope_part[:, nope_data_bytes:]
    
    k_nope = torch.empty(
        (num_tokens, dim_nope),
        dtype=torch.bfloat16,
        device=nope_part.device,
    )
    
    num_nope_blocks = triton.cdiv(dim_nope, FP4_BLOCK_SIZE)
    _dequantize_nvfp4_kernel[(num_tokens, num_nope_blocks)](
        k_nope,
        nope_data,
        nope_scales,
        k_nope.stride(0),
        nope_data.stride(0),
        nope_scales.stride(0),
        DIM=dim_nope,
        BLOCK_SIZE=FP4_BLOCK_SIZE,
    )
    
    # Dequantize rope
    rope_data_bytes = dim_rope // 2
    rope_scale_bytes = dim_rope // FP4_BLOCK_SIZE
    
    rope_data = rope_part[:, :rope_data_bytes]
    rope_scales = rope_part[:, rope_data_bytes:]
    
    k_rope = torch.empty(
        (num_tokens, dim_rope),
        dtype=torch.bfloat16,
        device=rope_part.device,
    )
    
    num_rope_blocks = triton.cdiv(dim_rope, FP4_BLOCK_SIZE)
    _dequantize_nvfp4_kernel[(num_tokens, num_rope_blocks)](
        k_rope,
        rope_data,
        rope_scales,
        k_rope.stride(0),
        rope_data.stride(0),
        rope_scales.stride(0),
        DIM=dim_rope,
        BLOCK_SIZE=FP4_BLOCK_SIZE,
    )
    
    return k_nope, k_rope
