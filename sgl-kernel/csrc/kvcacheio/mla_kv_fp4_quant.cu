/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/all.h>

#include <cstdint>

#ifndef USE_ROCM
#define WARP_SIZE 32
#include "pytorch_extension_utils.h"
#else
#include "pytorch_extension_utils_rocm.h"
#include "utils.h"
#endif

// NVFP4 E2M1 format constants
constexpr float FP4_E2M1_MAX = 6.0f;
constexpr int FP4_BLOCK_SIZE = 16;  // NVFP4 block size (16 elements per scale factor)
constexpr int ELTS_PER_THREAD = 8;  // Each thread processes 8 elements

// Fast reciprocal
inline __device__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
  return b;
}

// Convert 8 float32 values into 8 e2m1 values (packed into one uint32_t)
// Uses PTX instructions available on SM100+ (Blackwell)
inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0]),
        "f"(array[1]),
        "f"(array[2]),
        "f"(array[3]),
        "f"(array[4]),
        "f"(array[5]),
        "f"(array[6]),
        "f"(array[7]));
  return val;
#else
  // Fallback for older architectures (should not be called on SM < 100)
  return 0;
#endif
}

// Convert float2 array to e2m1 (same output, different input layout)
inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4]) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}"
      : "=r"(val)
      : "f"(array[0].x),
        "f"(array[0].y),
        "f"(array[1].x),
        "f"(array[1].y),
        "f"(array[2].x),
        "f"(array[2].y),
        "f"(array[3].x),
        "f"(array[3].y));
  return val;
#else
  return 0;
#endif
}

// Type converter for half/bfloat16
template <typename T>
struct TypeConverter {
  using Type = half2;
};

template <>
struct TypeConverter<half2> {
  using Type = half;
};

template <>
struct TypeConverter<half> {
  using Type = half2;
};

template <>
struct TypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

// Packed vector type for loading 8 elements
template <typename T>
struct PackedVec {
  using Type = typename TypeConverter<T>::Type;
  Type elts[ELTS_PER_THREAD / 2];  // 4 half2 or 4 bfloat162
};

// NVFP4 Quantization kernel for MLA KV Cache
// Uses hardware PTX instructions on SM100+ (Blackwell)
template <typename T>
__global__ void __launch_bounds__(128, 2) mla_kv_fp4_quant_kernel(
    const T* __restrict__ k_nope,
    const T* __restrict__ k_rope,
    uint8_t* __restrict__ kv_buffer,       // Output: packed FP4 data (uint32_t per 8 values)
    uint8_t* __restrict__ kv_scale_buffer, // Output: UE8M0/E4M3 scale factors
    const int64_t* __restrict__ loc,       // Token locations
    int num_tokens,
    int kv_lora_rank,
    int qk_rope_head_dim,
    int page_size) {
  
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane_id = tid % WARP_SIZE;
  
  if (token_idx >= num_tokens) return;
  
  const int64_t page_id = loc[token_idx];
  
  // Calculate output offsets
  const int total_dim = kv_lora_rank + qk_rope_head_dim;
  const int packed_dim = total_dim / 2;  // 2 FP4 values per byte, but we pack 8 per uint32_t
  const int num_blocks = total_dim / FP4_BLOCK_SIZE;
  
  // Output layout: uint32_t per 8 elements (4 bytes per 8 values = 0.5 bytes per value)
  uint32_t* out_kv = reinterpret_cast<uint32_t*>(kv_buffer) + page_id * (packed_dim / 4);
  uint8_t* out_scale = kv_scale_buffer + page_id * num_blocks;
  
  using PackedType = PackedVec<T>;
  constexpr int THREADS_PER_SF = FP4_BLOCK_SIZE / ELTS_PER_THREAD;  // 2 threads per SF
  
  // Process k_nope part
  for (int block_start = 0; block_start < kv_lora_rank; block_start += FP4_BLOCK_SIZE) {
    // Each thread loads 8 elements
    float local_vals[ELTS_PER_THREAD];
    
    // Load and convert to float
    #pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++) {
      int elem_idx = block_start + (tid % THREADS_PER_SF) * ELTS_PER_THREAD + i;
      if (elem_idx < kv_lora_rank) {
        int idx = token_idx * kv_lora_rank + elem_idx;
        local_vals[i] = static_cast<float>(k_nope[idx]);
      } else {
        local_vals[i] = 0.0f;
      }
    }
    
    // Find max abs in this thread's 8 values
    float local_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++) {
      local_max = fmaxf(local_max, fabsf(local_vals[i]));
    }
    
    // Warp reduce: pair of threads (16 elements) share one scale factor
    // Shuffle with thread XOR 1 to exchange between paired threads
    float other_max = __shfl_xor_sync(0xffffffff, local_max, 1);
    float block_max = fmaxf(local_max, other_max);
    
    // Compute scale factor (only one thread per pair writes)
    uint8_t scale_val = 0;
    float scale_f = 1.0f;
    
    if ((lane_id % THREADS_PER_SF) == 0) {
      // Compute SF = max / 6.0
      float sf = block_max * reciprocal_approximate_ftz(FP4_E2M1_MAX);
      
      // Convert to E4M3 (used as UE4M3 for positive values)
      __nv_fp8_e4m3 sf_fp8 = __nv_fp8_e4m3(sf);
      scale_val = sf_fp8.__x;
      scale_f = static_cast<float>(sf_fp8);
      
      // Write scale to global memory
      int block_idx = block_start / FP4_BLOCK_SIZE;
      out_scale[block_idx] = scale_val;
    }
    
    // Broadcast scale to both threads in the pair
    scale_val = __shfl_sync(0xffffffff, scale_val, lane_id & ~1);
    scale_f = __shfl_sync(0xffffffff, scale_f, lane_id & ~1);
    
    // Compute output scale (reciprocal of SF)
    float output_scale = scale_f != 0.0f ? reciprocal_approximate_ftz(scale_f) : 0.0f;
    
    // Scale the values
    float scaled_vals[ELTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++) {
      scaled_vals[i] = local_vals[i] * output_scale;
    }
    
    // Convert to FP4 using PTX instruction
    uint32_t fp4_packed = fp32_vec_to_e2m1(scaled_vals);
    
    // Write output (each thread writes one uint32_t for its 8 values)
    int thread_in_block = tid % THREADS_PER_SF;
    int out_idx = (block_start / 8) + thread_in_block;
    if (block_start + thread_in_block * ELTS_PER_THREAD < kv_lora_rank) {
      out_kv[out_idx] = fp4_packed;
    }
  }
  
  // Process k_rope part
  const int rope_kv_offset = kv_lora_rank / 8;  // uint32_t offset
  const int scale_rope_offset = kv_lora_rank / FP4_BLOCK_SIZE;
  uint32_t* out_kv_rope = out_kv + rope_kv_offset;
  uint8_t* out_scale_rope = out_scale + scale_rope_offset;
  
  for (int block_start = 0; block_start < qk_rope_head_dim; block_start += FP4_BLOCK_SIZE) {
    float local_vals[ELTS_PER_THREAD];
    
    #pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++) {
      int elem_idx = block_start + (tid % THREADS_PER_SF) * ELTS_PER_THREAD + i;
      if (elem_idx < qk_rope_head_dim) {
        int idx = token_idx * qk_rope_head_dim + elem_idx;
        local_vals[i] = static_cast<float>(k_rope[idx]);
      } else {
        local_vals[i] = 0.0f;
      }
    }
    
    float local_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++) {
      local_max = fmaxf(local_max, fabsf(local_vals[i]));
    }
    
    float other_max = __shfl_xor_sync(0xffffffff, local_max, 1);
    float block_max = fmaxf(local_max, other_max);
    
    uint8_t scale_val = 0;
    float scale_f = 1.0f;
    
    if ((lane_id % THREADS_PER_SF) == 0) {
      float sf = block_max * reciprocal_approximate_ftz(FP4_E2M1_MAX);
      __nv_fp8_e4m3 sf_fp8 = __nv_fp8_e4m3(sf);
      scale_val = sf_fp8.__x;
      scale_f = static_cast<float>(sf_fp8);
      
      int block_idx = block_start / FP4_BLOCK_SIZE;
      out_scale_rope[block_idx] = scale_val;
    }
    
    scale_val = __shfl_sync(0xffffffff, scale_val, lane_id & ~1);
    scale_f = __shfl_sync(0xffffffff, scale_f, lane_id & ~1);
    
    float output_scale = scale_f != 0.0f ? reciprocal_approximate_ftz(scale_f) : 0.0f;
    
    float scaled_vals[ELTS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ELTS_PER_THREAD; i++) {
      scaled_vals[i] = local_vals[i] * output_scale;
    }
    
    uint32_t fp4_packed = fp32_vec_to_e2m1(scaled_vals);
    
    int thread_in_block = tid % THREADS_PER_SF;
    int out_idx = (block_start / 8) + thread_in_block;
    if (block_start + thread_in_block * ELTS_PER_THREAD < qk_rope_head_dim) {
      out_kv_rope[out_idx] = fp4_packed;
    }
  }
#endif
}

// NVFP4 Dequantization kernel
template <typename T>
__global__ void __launch_bounds__(128, 2) mla_kv_fp4_dequant_kernel(
    T* __restrict__ k_nope,
    T* __restrict__ k_rope,
    const uint8_t* __restrict__ kv_buffer,
    const uint8_t* __restrict__ kv_scale_buffer,
    const int64_t* __restrict__ loc,
    int num_tokens,
    int kv_lora_rank,
    int qk_rope_head_dim,
    int page_size) {
  
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  const int token_idx = blockIdx.x;
  const int tid = threadIdx.x;
  
  if (token_idx >= num_tokens) return;
  
  const int64_t page_id = loc[token_idx];
  
  const int total_dim = kv_lora_rank + qk_rope_head_dim;
  const int packed_dim = total_dim / 2;
  const int num_blocks = total_dim / FP4_BLOCK_SIZE;
  
  const uint32_t* in_kv = reinterpret_cast<const uint32_t*>(kv_buffer) + page_id * (packed_dim / 4);
  const uint8_t* in_scale = kv_scale_buffer + page_id * num_blocks;
  
  // FP4 lookup table for dequantization (E2M1 format)
  // Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
  const float fp4_table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  
  // Dequantize k_nope
  for (int elem_idx = tid; elem_idx < kv_lora_rank; elem_idx += blockDim.x) {
    int block_idx = elem_idx / FP4_BLOCK_SIZE;
    int in_block_offset = elem_idx % FP4_BLOCK_SIZE;
    
    // Read scale factor
    uint8_t scale_val = in_scale[block_idx];
    float scale_f;
    
    // Convert E4M3 scale to float (same as NVFP4 quantization)
    __nv_fp8_e4m3 sf_fp8;
    sf_fp8.__x = scale_val;
    scale_f = static_cast<float>(sf_fp8);
    
    // Read packed FP4 data
    int packed_idx = elem_idx / 8;
    uint32_t packed = in_kv[packed_idx];
    
    // Extract the correct byte (4 bytes per uint32_t, 2 values per byte)
    int byte_idx = (elem_idx % 8) / 2;
    uint8_t byte_val = (packed >> (byte_idx * 8)) & 0xFF;
    
    // Extract FP4 value (4 bits each)
    uint8_t fp4_val;
    if ((elem_idx % 2) == 0) {
      fp4_val = byte_val & 0xF;  // Low nibble
    } else {
      fp4_val = (byte_val >> 4) & 0xF;  // High nibble
    }
    
    // Dequantize
    uint8_t mag = fp4_val & 0x7;
    float sign = (fp4_val & 0x8) ? -1.0f : 1.0f;
    float dequant_val = sign * fp4_table[mag] * scale_f;
    
    int idx = token_idx * kv_lora_rank + elem_idx;
    k_nope[idx] = static_cast<T>(dequant_val);
  }
  
  // Dequantize k_rope
  const int rope_kv_offset = kv_lora_rank / 8;
  const int scale_rope_offset = kv_lora_rank / FP4_BLOCK_SIZE;
  const uint32_t* in_kv_rope = in_kv + rope_kv_offset;
  const uint8_t* in_scale_rope = in_scale + scale_rope_offset;
  
  for (int elem_idx = tid; elem_idx < qk_rope_head_dim; elem_idx += blockDim.x) {
    int block_idx = elem_idx / FP4_BLOCK_SIZE;
    int in_block_offset = elem_idx % FP4_BLOCK_SIZE;
    
    uint8_t scale_val = in_scale_rope[block_idx];
    __nv_fp8_e4m3 sf_fp8;
    sf_fp8.__x = scale_val;
    float scale_f = static_cast<float>(sf_fp8);
    
    int packed_idx = elem_idx / 8;
    uint32_t packed = in_kv_rope[packed_idx];
    
    int byte_idx = (elem_idx % 8) / 2;
    uint8_t byte_val = (packed >> (byte_idx * 8)) & 0xFF;
    
    uint8_t fp4_val;
    if ((elem_idx % 2) == 0) {
      fp4_val = byte_val & 0xF;
    } else {
      fp4_val = (byte_val >> 4) & 0xF;
    }
    
    uint8_t mag = fp4_val & 0x7;
    float sign = (fp4_val & 0x8) ? -1.0f : 1.0f;
    float dequant_val = sign * fp4_table[mag] * scale_f;
    
    int idx = token_idx * qk_rope_head_dim + elem_idx;
    k_rope[idx] = static_cast<T>(dequant_val);
  }
#endif
}

// Host entry points
void mla_kv_fp4_quant(
    torch::Tensor const& k_nope,
    torch::Tensor const& k_rope,
    torch::Tensor& kv_buffer,
    torch::Tensor& kv_scale_buffer,
    torch::Tensor const& loc) {
  
  TORCH_CHECK(k_nope.is_cuda(), "k_nope must be CUDA tensor");
  TORCH_CHECK(k_rope.is_cuda(), "k_rope must be CUDA tensor");
  TORCH_CHECK(kv_buffer.is_cuda(), "kv_buffer must be CUDA tensor");
  TORCH_CHECK(kv_scale_buffer.is_cuda(), "kv_scale_buffer must be CUDA tensor");
  TORCH_CHECK(loc.is_cuda(), "loc must be CUDA tensor");
  
  // Check SM version (NVFP4 requires SM100+)
  int device_idx = k_nope.get_device();
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_idx);
  TORCH_CHECK(prop.major >= 10, "NVFP4 quantization requires SM100+ (Blackwell) architecture");
  
  int num_tokens = k_nope.size(0);
  int kv_lora_rank = k_nope.size(2);
  int qk_rope_head_dim = k_rope.size(2);
  
  TORCH_CHECK(kv_lora_rank % FP4_BLOCK_SIZE == 0, 
              "kv_lora_rank must be divisible by ", FP4_BLOCK_SIZE);
  TORCH_CHECK(qk_rope_head_dim % FP4_BLOCK_SIZE == 0, 
              "qk_rope_head_dim must be divisible by ", FP4_BLOCK_SIZE);
  
  const int threads = 128;
  const int blocks = num_tokens;
  
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    k_nope.scalar_type(),
    "mla_kv_fp4_quant",
    [&] {
      mla_kv_fp4_quant_kernel<scalar_t><<<blocks, threads>>>(
        k_nope.data_ptr<scalar_t>(),
        k_rope.data_ptr<scalar_t>(),
        kv_buffer.data_ptr<uint8_t>(),
        kv_scale_buffer.data_ptr<uint8_t>(),
        loc.data_ptr<int64_t>(),
        num_tokens,
        kv_lora_rank,
        qk_rope_head_dim,
        1  // page_size
      );
    }
  );
  
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void mla_kv_fp4_dequant(
    torch::Tensor& k_nope,
    torch::Tensor& k_rope,
    torch::Tensor const& kv_buffer,
    torch::Tensor const& kv_scale_buffer,
    torch::Tensor const& loc) {
  
  TORCH_CHECK(k_nope.is_cuda(), "k_nope must be CUDA tensor");
  TORCH_CHECK(k_rope.is_cuda(), "k_rope must be CUDA tensor");
  TORCH_CHECK(kv_buffer.is_cuda(), "kv_buffer must be CUDA tensor");
  TORCH_CHECK(kv_scale_buffer.is_cuda(), "kv_scale_buffer must be CUDA tensor");
  TORCH_CHECK(loc.is_cuda(), "loc must be CUDA tensor");
  
  // Check SM version
  int device_idx = k_nope.get_device();
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_idx);
  TORCH_CHECK(prop.major >= 10, "NVFP4 dequantization requires SM100+ (Blackwell) architecture");
  
  int num_tokens = k_nope.size(0);
  int kv_lora_rank = k_nope.size(2);
  int qk_rope_head_dim = k_rope.size(2);
  
  const int threads = 128;
  const int blocks = num_tokens;
  
  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::ScalarType::Half,
    at::ScalarType::BFloat16,
    k_nope.scalar_type(),
    "mla_kv_fp4_dequant",
    [&] {
      mla_kv_fp4_dequant_kernel<scalar_t><<<blocks, threads>>>(
        k_nope.data_ptr<scalar_t>(),
        k_rope.data_ptr<scalar_t>(),
        kv_buffer.data_ptr<uint8_t>(),
        kv_scale_buffer.data_ptr<uint8_t>(),
        loc.data_ptr<int64_t>(),
        num_tokens,
        kv_lora_rank,
        qk_rope_head_dim,
        1  // page_size
      );
    }
  );
  
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
