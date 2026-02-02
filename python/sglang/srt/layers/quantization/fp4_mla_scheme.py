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
NVFP4 Quantization Scheme for MLA KV Cache

This module provides NVFP4 (NVIDIA FP4) quantization scheme for MLA KV cache,
using hardware-accelerated PTX instructions on Blackwell (SM100+) GPUs.

NVFP4 uses:
- E2M1 format (same as MXFP4)
- Block-wise microscaling with 16-element blocks
- E4M3 scale factor format
- PTX hardware instructions for quantization

Minimum requirement: SM100 (Blackwell) architecture
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Tuple

import torch
from torch.nn.parameter import Parameter

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.utils import is_cuda, is_hip

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention


# NVFP4 E2M1 constants
FP4_E2M1_MAX = 6.0
FP4_BLOCK_SIZE = 16  # NVFP4 block size


@dataclass
class NVFP4MLAQuantParams:
    """Parameters for NVFP4 MLA quantization."""
    
    # Block size for microscaling (fixed at 16 for NVFP4)
    block_size: int = 16
    
    # Whether to quantize nope and rope separately
    separate_nope_rope: bool = True
    
    # Whether to use native CUDA kernel (NVFP4 requires this)
    use_cuda_kernel: bool = True


class NVFP4MLAQuantizationConfig(QuantizationConfig):
    """
    Configuration for NVFP4 MLA quantization.
    
    This config provides NVFP4 KV cache quantization using hardware-accelerated
    PTX instructions on NVIDIA Blackwell (SM100+) GPUs.
    """
    
    def __init__(
        self,
        block_size: int = 16,
        separate_nope_rope: bool = True,
        use_cuda_kernel: bool = True,
    ):
        self.quant_params = NVFP4MLAQuantParams(
            block_size=block_size,
            separate_nope_rope=separate_nope_rope,
            use_cuda_kernel=use_cuda_kernel,
        )
    
    @classmethod
    def get_name(cls) -> str:
        return "nvfp4_mla"
    
    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]
    
    @classmethod
    def get_min_capability(cls) -> int:
        # NVFP4 requires compute capability 10.0+ (Blackwell)
        # Hardware-accelerated PTX instructions only available on SM100+
        return 100
    
    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []
    
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "NVFP4MLAQuantizationConfig":
        """Create config from dictionary (e.g., from model config)."""
        return cls(
            block_size=config.get("block_size", 16),
            separate_nope_rope=config.get("separate_nope_rope", True),
            use_cuda_kernel=config.get("use_cuda_kernel", True),
        )
    
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        """Get quantization method for a layer."""
        from sglang.srt.layers.radix_attention import RadixAttention
        
        if isinstance(layer, RadixAttention):
            return NVFP4MLAQuantizeMethod(self.quant_params)
        return None
    
    def get_scaled_act_names(self) -> list[str]:
        return []


class NVFP4MLAQuantizeMethod(QuantizeMethodBase):
    """
    Quantization method for NVFP4 MLA.
    
    This method handles the quantization and dequantization of MLA KV cache
    using NVFP4 format with hardware-accelerated PTX instructions on SM100+.
    """
    
    def __init__(self, quant_params: NVFP4MLAQuantParams):
        self.quant_params = quant_params
        self._cuda_kernel_available = self._check_cuda_kernel()
    
    def _check_cuda_kernel(self) -> bool:
        """Check if native CUDA kernel is available."""
        if not is_cuda():
            return False
        try:
            from sgl_kernel import mla_kv_fp4_quant
            return True
        except ImportError:
            return False
    
    def create_weights(
        self,
        layer: torch.nn.Module,
        **kwargs,
    ) -> None:
        """Create quantization parameters (if any)."""
        # NVFP4 quantization doesn't require learnable parameters
        # Scale factors are computed dynamically
        pass
    
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Process weights after loading."""
        pass
    
    def quantize_mla_kv(
        self,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        kv_buffer: torch.Tensor,
        kv_scale_buffer: torch.Tensor,
        loc: torch.Tensor,
    ) -> None:
        """
        Quantize MLA KV cache to NVFP4.
        
        Args:
            k_nope: [num_tokens, 1, kv_lora_rank]
            k_rope: [num_tokens, 1, qk_rope_head_dim]
            kv_buffer: Output buffer for packed FP4 data
            kv_scale_buffer: Output buffer for E4M3 scale factors
            loc: Token locations
        """
        if self.quant_params.use_cuda_kernel and self._cuda_kernel_available:
            # Use native CUDA kernel with PTX instructions
            from sgl_kernel import mla_kv_fp4_quant
            mla_kv_fp4_quant(k_nope, k_rope, kv_buffer, kv_scale_buffer, loc)
        else:
            raise RuntimeError(
                "NVFP4 quantization requires native CUDA kernel with SM100+ support. "
                "Please ensure you're running on Blackwell (SM100+) GPU."
            )
    
    def dequantize_mla_kv(
        self,
        k_nope: torch.Tensor,
        k_rope: torch.Tensor,
        kv_buffer: torch.Tensor,
        kv_scale_buffer: torch.Tensor,
        loc: torch.Tensor,
    ) -> None:
        """
        Dequantize MLA KV cache from NVFP4.
        
        Args:
            k_nope: Output [num_tokens, 1, kv_lora_rank]
            k_rope: Output [num_tokens, 1, qk_rope_head_dim]
            kv_buffer: Input packed FP4 data
            kv_scale_buffer: Input E4M3 scale factors
            loc: Token locations
        """
        if self.quant_params.use_cuda_kernel and self._cuda_kernel_available:
            # Use native CUDA kernel
            from sgl_kernel import mla_kv_fp4_dequant
            mla_kv_fp4_dequant(k_nope, k_rope, kv_buffer, kv_scale_buffer, loc)
        else:
            raise RuntimeError(
                "NVFP4 dequantization requires native CUDA kernel with SM100+ support. "
                "Please ensure you're running on Blackwell (SM100+) GPU."
            )


class NVFP4MLAMemoryPoolHelper:
    """
    Helper class for managing NVFP4 MLA memory pools.
    
    This class provides utilities for creating and managing NVFP4-quantized
    KV cache memory pools for Blackwell GPUs.
    """
    
    @staticmethod
    def create_mla_nvfp4_pool(
        size: int,
        page_size: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        layer_num: int,
        device: str,
        use_nsa: bool = False,
        **kwargs,
    ):
        """
        Create an MLA NVFP4 memory pool.
        
        Args:
            size: Number of tokens in the pool
            page_size: Page size for paged attention
            kv_lora_rank: Dimension of KV LoRA rank
            qk_rope_head_dim: Dimension of QK RoPE head
            layer_num: Number of layers
            device: Device to allocate on (must be SM100+)
            use_nsa: Whether to use NSA (Native Sparse Attention)
            **kwargs: Additional arguments
        
        Returns:
            MLATokenToKVPoolNVFP4 or NSATokenToKVPoolNVFP4 instance
        """
        if use_nsa:
            from sglang.srt.mem_cache.memory_pool import NSATokenToKVPoolNVFP4
            return NSATokenToKVPoolNVFP4(
                size=size,
                page_size=page_size,
                kv_lora_rank=kv_lora_rank,
                dtype=torch.float8_e4m3fn,  # Not used for FP4 storage
                qk_rope_head_dim=qk_rope_head_dim,
                layer_num=layer_num,
                device=device,
                index_head_dim=kwargs.get("index_head_dim", 128),
                enable_memory_saver=kwargs.get("enable_memory_saver", False),
            )
        else:
            from sglang.srt.mem_cache.memory_pool import MLATokenToKVPoolNVFP4
            return MLATokenToKVPoolNVFP4(
                size=size,
                page_size=page_size,
                dtype=torch.float8_e4m3fn,  # Not used for FP4 storage
                kv_lora_rank=kv_lora_rank,
                qk_rope_head_dim=qk_rope_head_dim,
                layer_num=layer_num,
                device=device,
                enable_memory_saver=kwargs.get("enable_memory_saver", False),
            )
    
    @staticmethod
    def get_memory_savings(
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
    ) -> dict[str, float]:
        """
        Calculate memory savings of NVFP4 vs other formats.
        
        Returns:
            Dictionary with memory usage ratios
        """
        total_dim = kv_lora_rank + qk_rope_head_dim
        
        # BF16: 2 bytes per element
        bf16_bytes = total_dim * 2
        
        # FP8: 1 byte per element
        fp8_bytes = total_dim * 1
        
        # NVFP4: 0.5 bytes per element + scale overhead
        fp4_data_bytes = total_dim // 2
        fp4_scale_bytes = total_dim // FP4_BLOCK_SIZE
        fp4_total_bytes = fp4_data_bytes + fp4_scale_bytes
        
        return {
            "bf16_bytes": bf16_bytes,
            "fp8_bytes": fp8_bytes,
            "nvfp4_bytes": fp4_total_bytes,
            "nvfp4_vs_bf16_ratio": fp4_total_bytes / bf16_bytes,
            "nvfp4_vs_fp8_ratio": fp4_total_bytes / fp8_bytes,
            "memory_savings_vs_bf16": 1 - (fp4_total_bytes / bf16_bytes),
            "memory_savings_vs_fp8": 1 - (fp4_total_bytes / fp8_bytes),
        }


def register_nvfp4_mla_quantization():
    """
    Register NVFP4 MLA quantization with the quantization registry.
    
    This function should be called during module initialization.
    """
    from sglang.srt.layers.quantization import QUANTIZATION_METHODS
    
    # Register as primary name
    QUANTIZATION_METHODS["nvfp4_mla"] = NVFP4MLAQuantizationConfig
    
    # Also register legacy aliases for compatibility
    QUANTIZATION_METHODS["fp4_mla"] = NVFP4MLAQuantizationConfig
    QUANTIZATION_METHODS["fp4_kv"] = NVFP4MLAQuantizationConfig
