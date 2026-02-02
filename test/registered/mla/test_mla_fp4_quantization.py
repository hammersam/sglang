#!/usr/bin/env python3
"""
Test NVFP4 Quantization for MLA KV Cache

This test validates:
1. NVFP4 quantization/dequantization accuracy
2. CUDA kernel correctness (requires SM100+ Blackwell GPU)
3. Memory savings calculation
4. Integration with MLATokenToKVPoolNVFP4

Note: NVFP4 requires SM100+ (Blackwell) architecture for hardware acceleration.
"""

import math
import unittest

import pytest
import torch

# Skip if CUDA is not available
cuda_available = torch.cuda.is_available()

# Check if GPU is Blackwell (SM100+)
def is_blackwell():
    if not cuda_available:
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major >= 10


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
class TestNVFP4MLAQuantization(unittest.TestCase):
    """Test NVFP4 quantization for MLA KV Cache."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.batch_size = 4
        self.kv_lora_rank = 512
        self.qk_rope_head_dim = 64
        self.total_dim = self.kv_lora_rank + self.qk_rope_head_dim
        
        # NVFP4 constants
        self.fp4_block_size = 16
        self.fp4_max_val = 6.0
    
    def test_nvfp4_quantization_dequantization(self):
        """Test basic NVFP4 quantization and dequantization."""
        from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil
        
        # Create random input
        x = torch.randn(
            self.batch_size, 1, self.total_dim,
            dtype=self.dtype,
            device=self.device,
        )
        
        # Quantize
        x_fp4, scale = KVFP4QuantizeUtil.batched_quantize(x)
        
        # Dequantize
        x_dequant = KVFP4QuantizeUtil.batched_dequantize(x_fp4, scale, self.dtype)
        
        # Check shapes
        self.assertEqual(x_fp4.shape, (self.batch_size, 1, self.total_dim // 2))
        self.assertEqual(scale.shape, (self.batch_size, 1, self.total_dim // self.fp4_block_size))
        
        # Check reconstruction error is reasonable
        rel_error = torch.mean(torch.abs(x - x_dequant) / (torch.abs(x) + 1e-8))
        print(f"Relative error: {rel_error.item():.4f}")
        self.assertLess(rel_error.item(), 0.1)  # Expect < 10% relative error
    
    def test_nvfp4_memory_savings(self):
        """Test memory savings calculation."""
        from sglang.srt.layers.quantization.fp4_mla_scheme import NVFP4MLAMemoryPoolHelper
        
        savings = NVFP4MLAMemoryPoolHelper.get_memory_savings(
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
        )
        
        print(f"Memory savings report:")
        print(f"  BF16 bytes: {savings['bf16_bytes']}")
        print(f"  FP8 bytes: {savings['fp8_bytes']}")
        print(f"  NVFP4 bytes: {savings['nvfp4_bytes']}")
        print(f"  NVFP4 vs BF16 ratio: {savings['nvfp4_vs_bf16_ratio']:.2%}")
        print(f"  NVFP4 vs FP8 ratio: {savings['nvfp4_vs_fp8_ratio']:.2%}")
        print(f"  Memory savings vs BF16: {savings['memory_savings_vs_bf16']:.2%}")
        print(f"  Memory savings vs FP8: {savings['memory_savings_vs_fp8']:.2%}")
        
        # NVFP4 should use less memory than FP8
        self.assertLess(savings['nvfp4_bytes'], savings['fp8_bytes'])
        # NVFP4 should use significantly less memory than BF16
        self.assertLess(savings['nvfp4_vs_bf16_ratio'], 0.5)
    
    def test_nvfp4_mla_pool(self):
        """Test MLATokenToKVPoolNVFP4 creation."""
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPoolNVFP4
        
        pool_size = 1024
        page_size = 64
        layer_num = 2
        
        pool = MLATokenToKVPoolNVFP4(
            size=pool_size,
            page_size=page_size,
            dtype=torch.float8_e4m3fn,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=self.qk_rope_head_dim,
            layer_num=layer_num,
            device=self.device,
            enable_memory_saver=False,
        )
        
        # Check pool properties
        self.assertEqual(pool.kv_lora_rank, self.kv_lora_rank)
        self.assertEqual(pool.qk_rope_head_dim, self.qk_rope_head_dim)
        
        # Check buffer sizes
        expected_packed_dim = (self.kv_lora_rank + self.qk_rope_head_dim) // 2
        expected_scale_dim = (self.kv_lora_rank + self.qk_rope_head_dim) // 16
        
        for layer_id in range(layer_num):
            kv_buffer = pool.kv_buffer[layer_id]
            scale_buffer = pool.kv_scale_buffer[layer_id]
            
            self.assertEqual(kv_buffer.shape[-1], expected_packed_dim)
            self.assertEqual(scale_buffer.shape[-1], expected_scale_dim)
        
        print(f"MLATokenToKVPoolNVFP4 created successfully")
        print(f"  KV buffer shape: {pool.kv_buffer[0].shape}")
        print(f"  Scale buffer shape: {pool.kv_scale_buffer[0].shape}")
        print(f"  Memory usage: {pool.mem_usage:.2f} GB")
    
    def test_nvfp4_config(self):
        """Test NVFP4MLAQuantizationConfig."""
        from sglang.srt.layers.quantization.fp4_mla_scheme import (
            NVFP4MLAQuantizationConfig,
            NVFP4MLAQuantizeMethod,
        )
        
        config = NVFP4MLAQuantizationConfig(
            block_size=16,
            separate_nope_rope=True,
            use_cuda_kernel=True,
        )
        
        self.assertEqual(config.get_name(), "nvfp4_mla")
        self.assertEqual(config.quant_params.block_size, 16)
        self.assertEqual(config.get_min_capability(), 100)  # SM100+
        
        # Test method creation
        method = NVFP4MLAQuantizeMethod(config.quant_params)
        self.assertIsNotNone(method)
        
        print(f"NVFP4MLAQuantizationConfig test passed")
    
    @pytest.mark.skipif(not is_blackwell(), reason="Requires SM100+ Blackwell GPU")
    def test_cuda_kernel_sm100(self):
        """Test CUDA kernel on SM100+ Blackwell GPU."""
        try:
            from sgl_kernel import mla_kv_fp4_quant, mla_kv_fp4_dequant
            print("NVFP4 CUDA kernel is available (SM100+ detected)")
            kernel_available = True
        except ImportError:
            print("NVFP4 CUDA kernel not available")
            kernel_available = False
        
        self.assertTrue(kernel_available, "NVFP4 CUDA kernel should be available on SM100+")
        
        # Test actual kernel execution
        num_tokens = 4
        k_nope = torch.randn(num_tokens, 1, self.kv_lora_rank, dtype=self.dtype, device=self.device)
        k_rope = torch.randn(num_tokens, 1, self.qk_rope_head_dim, dtype=self.dtype, device=self.device)
        
        # Allocate output buffers
        total_dim = self.kv_lora_rank + self.qk_rope_head_dim
        kv_buffer = torch.empty(num_tokens, total_dim // 2, dtype=torch.uint8, device=self.device)
        kv_scale_buffer = torch.empty(num_tokens, total_dim // 16, dtype=torch.uint8, device=self.device)
        loc = torch.arange(num_tokens, dtype=torch.int64, device=self.device)
        
        # Test quantization
        mla_kv_fp4_quant(k_nope, k_rope, kv_buffer, kv_scale_buffer, loc)
        print("NVFP4 quantization kernel executed successfully")
        
        # Test dequantization
        k_nope_out = torch.empty_like(k_nope)
        k_rope_out = torch.empty_like(k_rope)
        mla_kv_fp4_dequant(k_nope_out, k_rope_out, kv_buffer, kv_scale_buffer, loc)
        print("NVFP4 dequantization kernel executed successfully")
        
        # Check reconstruction error
        nope_rel_error = torch.mean(torch.abs(k_nope - k_nope_out) / (torch.abs(k_nope) + 1e-8))
        rope_rel_error = torch.mean(torch.abs(k_rope - k_rope_out) / (torch.abs(k_rope) + 1e-8))
        
        print(f"  nope relative error: {nope_rel_error.item():.4f}")
        print(f"  rope relative error: {rope_rel_error.item():.4f}")
        
        self.assertLess(nope_rel_error.item(), 0.1)
        self.assertLess(rope_rel_error.item(), 0.1)


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
class TestNSANVFP4Quantization(unittest.TestCase):
    """Test NVFP4 quantization for NSA KV Cache."""
    
    def setUp(self):
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.dim_nope = 512
        self.dim_rope = 64
    
    def test_nsa_nvfp4_quantization(self):
        """Test NSA NVFP4 quantization."""
        from sglang.srt.layers.attention.nsa.quant_k_cache_fp4 import (
            quantize_k_cache_nvfp4_separate,
        )
        
        num_tokens = 16
        k_nope = torch.randn(num_tokens, self.dim_nope, dtype=self.dtype, device=self.device)
        k_rope = torch.randn(num_tokens, self.dim_rope, dtype=self.dtype, device=self.device)
        
        # Quantize
        nope_part, rope_part = quantize_k_cache_nvfp4_separate(k_nope, k_rope)
        
        # Check shapes
        # nope_part: [num_tokens, 1, packed_bytes_nope]
        # packed_bytes_nope = dim_nope/2 (data) + dim_nope/16 (scales)
        expected_nope_bytes = self.dim_nope // 2 + self.dim_nope // 16
        expected_rope_bytes = self.dim_rope // 2 + self.dim_rope // 16
        
        self.assertEqual(nope_part.shape, (num_tokens, 1, expected_nope_bytes))
        self.assertEqual(rope_part.shape, (num_tokens, 1, expected_rope_bytes))
        
        print(f"NSA NVFP4 quantization test passed")
        print(f"  nope_part shape: {nope_part.shape}")
        print(f"  rope_part shape: {rope_part.shape}")
    
    def test_nsa_nvfp4_dequantization(self):
        """Test NSA NVFP4 dequantization."""
        from sglang.srt.layers.attention.nsa.quant_k_cache_fp4 import (
            quantize_k_cache_nvfp4_separate,
        )
        from sglang.srt.layers.attention.nsa.dequant_k_cache_fp4 import (
            dequantize_k_cache_nvfp4_separate,
        )
        
        num_tokens = 16
        k_nope = torch.randn(num_tokens, self.dim_nope, dtype=self.dtype, device=self.device)
        k_rope = torch.randn(num_tokens, self.dim_rope, dtype=self.dtype, device=self.device)
        
        # Quantize
        nope_part, rope_part = quantize_k_cache_nvfp4_separate(k_nope, k_rope)
        
        # Dequantize
        k_nope_dequant, k_rope_dequant = dequantize_k_cache_nvfp4_separate(
            nope_part.squeeze(1), rope_part.squeeze(1)
        )
        
        # Check reconstruction
        nope_rel_error = torch.mean(torch.abs(k_nope - k_nope_dequant) / (torch.abs(k_nope) + 1e-8))
        rope_rel_error = torch.mean(torch.abs(k_rope - k_rope_dequant) / (torch.abs(k_rope) + 1e-8))
        
        print(f"NSA NVFP4 dequantization test passed")
        print(f"  nope relative error: {nope_rel_error.item():.4f}")
        print(f"  rope relative error: {rope_rel_error.item():.4f}")
        
        self.assertLess(nope_rel_error.item(), 0.1)
        self.assertLess(rope_rel_error.item(), 0.1)


def run_benchmark():
    """Run performance benchmark."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return
    
    from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil
    
    device = "cuda"
    dtype = torch.bfloat16
    batch_sizes = [16, 64, 256, 1024]
    dim = 576  # 512 + 64
    
    print("\n" + "=" * 60)
    print("NVFP4 Quantization Benchmark (Software Fallback)")
    print("=" * 60)
    print("Note: For optimal performance on SM100+, use native CUDA kernels")
    print("-" * 60)
    
    for bs in batch_sizes:
        x = torch.randn(bs, 1, dim, dtype=dtype, device=device)
        
        # Warmup
        for _ in range(10):
            x_fp4, scale = KVFP4QuantizeUtil.batched_quantize(x)
            _ = KVFP4QuantizeUtil.batched_dequantize(x_fp4, scale, dtype)
        
        torch.cuda.synchronize()
        
        # Benchmark quantization
        import time
        n_iters = 100
        
        start = time.time()
        for _ in range(n_iters):
            x_fp4, scale = KVFP4QuantizeUtil.batched_quantize(x)
        torch.cuda.synchronize()
        quant_time = (time.time() - start) / n_iters * 1000
        
        # Benchmark dequantization
        start = time.time()
        for _ in range(n_iters):
            _ = KVFP4QuantizeUtil.batched_dequantize(x_fp4, scale, dtype)
        torch.cuda.synchronize()
        dequant_time = (time.time() - start) / n_iters * 1000
        
        print(f"Batch size {bs:4d}: Quant={quant_time:.3f}ms, Dequant={dequant_time:.3f}ms")


if __name__ == "__main__":
    # Run tests
    unittest.main(exit=False)
    
    # Run benchmark
    run_benchmark()
