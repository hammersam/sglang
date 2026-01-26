#!/usr/bin/env python3
"""
Standalone benchmark script for topk.cu kernel occupancy analysis.

This script compiles topk.cu using PyTorch JIT and benchmarks the kernel
for nsys/ncu profiling to measure SM occupancy improvements.

Usage:
    # Basic benchmark
    python benchmark_topk_occupancy.py

    # With nsys profiling
    nsys profile -o topk_profile python benchmark_topk_occupancy.py

    # With ncu profiling (detailed occupancy metrics)
    ncu --set full -o topk_metrics python benchmark_topk_occupancy.py

    # ncu with specific metrics for occupancy
    ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__maximum_warps_per_active_cycle_pct \
        python benchmark_topk_occupancy.py
"""

import os
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.resolve()
TOPK_CU_PATH = SCRIPT_DIR / "csrc" / "elementwise" / "topk.cu"


def compile_topk_kernel():
    """Compile topk.cu using PyTorch JIT."""
    print(f"Compiling {TOPK_CU_PATH}...")

    # Create a wrapper cpp file for pybind11 bindings
    wrapper_code = '''
#include <torch/extension.h>
#include <optional>

// Forward declarations from topk.cu
void fast_topk_interface(
    const at::Tensor& score,
    at::Tensor& indices,
    const at::Tensor& lengths,
    std::optional<at::Tensor> row_starts_opt);

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fast_topk", &fast_topk_interface,
          "Fast TopK kernel",
          py::arg("score"),
          py::arg("indices"),
          py::arg("lengths"),
          py::arg("row_starts") = py::none());
}
'''

    # Write wrapper to temp file
    wrapper_path = SCRIPT_DIR / "_topk_wrapper.cpp"
    with open(wrapper_path, "w") as f:
        f.write(wrapper_code)

    try:
        # Compile with JIT
        topk_module = load(
            name="topk_benchmark",
            sources=[str(wrapper_path), str(TOPK_CU_PATH)],
            extra_cuda_cflags=[
                "-O3",
                "--use_fast_math",
                "-lineinfo",  # For ncu source correlation
            ],
            verbose=True,
        )
        print("Compilation successful!")
        return topk_module
    finally:
        # Cleanup wrapper file
        if wrapper_path.exists():
            wrapper_path.unlink()


def fast_topk_v2(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: torch.Tensor = None,
    module=None,
) -> torch.Tensor:
    """Wrapper for fast_topk kernel matching sgl_kernel interface."""
    assert topk == 2048, "Only topk=2048 is supported"
    batch_size = score.size(0)
    indices = torch.empty((batch_size, topk), dtype=torch.int32, device=score.device)
    module.fast_topk(score, indices, lengths, row_starts)
    return indices


def warmup(module, score, lengths, topk, num_warmup=10):
    """Warmup the kernel to ensure stable measurements."""
    for _ in range(num_warmup):
        fast_topk_v2(score, lengths, topk, module=module)
    torch.cuda.synchronize()


def benchmark_kernel(
    module,
    batch_size: int,
    seq_len: int,
    topk: int = 2048,
    num_iterations: int = 100,
    with_row_starts: bool = False,
):
    """Benchmark the topk kernel with given parameters."""
    max_seq_len = 131072

    # Create input tensors
    torch.manual_seed(42)
    score = torch.randn(batch_size, max_seq_len, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")

    if with_row_starts:
        row_starts = torch.randint(0, 2048, (batch_size,), dtype=torch.int32, device="cuda")
    else:
        row_starts = None

    # Warmup
    warmup(module, score, lengths, topk)

    # Benchmark
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(num_iterations):
        fast_topk_v2(score, lengths, topk, row_starts=row_starts, module=module)
    end_event.record()

    torch.cuda.synchronize()
    elapsed_ms = start_event.elapsed_time(end_event)
    avg_ms = elapsed_ms / num_iterations

    return avg_ms


def run_profiling_workload(module, num_iterations: int = 50):
    """
    Run a fixed workload suitable for nsys/ncu profiling.
    This function runs multiple configurations to get comprehensive profiling data.
    """
    print("\n" + "=" * 60)
    print("Running profiling workload for nsys/ncu analysis")
    print("=" * 60)

    # Mark the profiling region for nsys
    torch.cuda.nvtx.range_push("topk_profiling_workload")

    configs = [
        # (batch_size, seq_len, with_row_starts)
        # (1, 4096, False),
        # (1, 16384, False),
        # (1, 65536, False),
        # (32, 4096, False),
        # (32, 16384, False),
        # (32, 65536, False),
        # (128, 4096, False),
        # (128, 16384, False),
        # (256, 4096, False),
        # (256, 16384, False),
        # (512, 131072, False)
        (4, 131072, False),
        (256, 131072, False),
    ]

    max_seq_len = 131072
    torch.manual_seed(42)

    for batch_size, seq_len, with_row_starts in configs:
        config_name = f"bs{batch_size}_seq{seq_len}"
        if with_row_starts:
            config_name += "_rowstarts"

        torch.cuda.nvtx.range_push(config_name)

        score = torch.randn(batch_size, max_seq_len, dtype=torch.float32, device="cuda")
        lengths = torch.full((batch_size,), seq_len, dtype=torch.int32, device="cuda")
        row_starts = None
        if with_row_starts:
            row_starts = torch.randint(0, 2048, (batch_size,), dtype=torch.int32, device="cuda")

        # Warmup
        warmup(module, score, lengths, 2048, num_warmup=5)

        # Profile iterations
        for i in range(num_iterations):
            fast_topk_v2(score, lengths, 2048, row_starts=row_starts, module=module)

        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_pop()
    print("Profiling workload completed.")


def get_max_shared_memory_per_sm():
    """Get max shared memory per SM using CUDA runtime API."""
    import ctypes

    # Try to get the value via cudaDeviceGetAttribute
    # cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81
    try:
        libcudart = ctypes.CDLL("libcudart.so")
    except OSError:
        try:
            libcudart = ctypes.CDLL("libcudart.dylib")
        except OSError:
            # Fallback: use known values based on compute capability
            return None

    device = torch.cuda.current_device()
    value = ctypes.c_int()
    # cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81
    result = libcudart.cudaDeviceGetAttribute(ctypes.byref(value), 81, device)
    if result == 0:  # cudaSuccess
        return value.value
    return None


def get_shared_memory_per_sm_by_cc(major: int, minor: int) -> int:
    """Get max shared memory per SM based on compute capability."""
    # Known values from CUDA documentation
    cc_to_smem = {
        (7, 0): 96 * 1024,   # V100
        (7, 5): 64 * 1024,   # T4
        (8, 0): 164 * 1024,  # A100
        (8, 6): 100 * 1024,  # RTX 3090
        (8, 9): 100 * 1024,  # RTX 4090
        (9, 0): 228 * 1024,  # H100
    }
    return cc_to_smem.get((major, minor), 96 * 1024)  # Default fallback


def print_theoretical_occupancy():
    """Print theoretical occupancy information based on kernel parameters."""
    print("\n" + "=" * 60)
    print("Theoretical Occupancy Analysis")
    print("=" * 60)

    # Kernel parameters from topk.cu
    threads_per_block = 1024
    shared_mem_bytes = 8 * 1024 * 4  # 32KB (changed from 128KB)

    # Get device properties
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)

    # Get max shared memory per SM
    max_smem_per_sm = get_max_shared_memory_per_sm()
    if max_smem_per_sm is None:
        max_smem_per_sm = get_shared_memory_per_sm_by_cc(props.major, props.minor)
        smem_source = "estimated from CC"
    else:
        smem_source = "from CUDA API"

    print(f"Device: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"Max threads per SM: {props.max_threads_per_multi_processor}")
    print(f"Max shared memory per SM: {max_smem_per_sm // 1024} KB ({smem_source})")
    print(f"Max shared memory per block: {props.shared_memory_per_block // 1024} KB")
    print(f"Number of SMs: {props.multi_processor_count}")
    print()
    print("Kernel Configuration:")
    print(f"  Threads per block: {threads_per_block}")
    print(f"  Dynamic shared memory: {shared_mem_bytes // 1024} KB")
    print()

    # Calculate theoretical max blocks per SM
    max_blocks_by_threads = props.max_threads_per_multi_processor // threads_per_block
    max_blocks_by_smem = max_smem_per_sm // shared_mem_bytes if shared_mem_bytes > 0 else float('inf')

    theoretical_blocks_per_sm = min(max_blocks_by_threads, int(max_blocks_by_smem))

    print(f"Theoretical max blocks per SM (by threads): {max_blocks_by_threads}")
    print(f"Theoretical max blocks per SM (by shared mem): {int(max_blocks_by_smem)}")
    print(f"Theoretical blocks per SM: {theoretical_blocks_per_sm}")

    theoretical_occupancy = (theoretical_blocks_per_sm * threads_per_block) / props.max_threads_per_multi_processor * 100
    print(f"Theoretical occupancy: {theoretical_occupancy:.1f}%")

    # Compare with old configuration (128KB)
    old_shared_mem = 32 * 1024 * 4  # 128KB
    old_max_blocks = min(max_blocks_by_threads, max_smem_per_sm // old_shared_mem)
    old_occupancy = (old_max_blocks * threads_per_block) / props.max_threads_per_multi_processor * 100
    print()
    print(f"Previous config (128KB shared mem):")
    print(f"  Max blocks per SM: {old_max_blocks}")
    print(f"  Occupancy: {old_occupancy:.1f}%")
    print(f"  Improvement: {theoretical_occupancy - old_occupancy:.1f}% absolute")


def main():
    print("=" * 60)
    print("TopK Kernel Occupancy Benchmark")
    print("=" * 60)

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return

    device = torch.cuda.current_device()
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    print()

    # Compile the kernel
    try:
        module = compile_topk_kernel()
    except Exception as e:
        print(f"Failed to compile kernel: {e}")
        return

    # Print theoretical occupancy analysis
    print_theoretical_occupancy()

    # Run benchmarks
    print("\n" + "=" * 60)
    print("Performance Benchmarks")
    print("=" * 60)

    configs = [
        # (batch_size, seq_len)
        (1, 4096),
        (1, 16384),
        (1, 65536),
        (32, 4096),
        (32, 16384),
        (32, 65536),
        (128, 4096),
        (128, 16384),
        (256, 4096),
        (256, 16384),
    ]

    print(f"\n{'Batch Size':<12} {'Seq Len':<10} {'Avg Time (ms)':<15} {'Throughput (K elem/s)':<20}")
    print("-" * 60)

    for batch_size, seq_len in configs:
        avg_ms = benchmark_kernel(module, batch_size, seq_len, num_iterations=100)
        total_elements = batch_size * seq_len
        throughput = total_elements / (avg_ms * 1e-3) / 1000  # K elements per second
        print(f"{batch_size:<12} {seq_len:<10} {avg_ms:<15.4f} {throughput:<20.2f}")

    # Run profiling workload
    run_profiling_workload(module)

    print("\n" + "=" * 60)
    print("Benchmark completed!")
    print()
    print("To analyze occupancy with ncu, run:")
    print("  ncu --set full -o topk_metrics python benchmark_topk_occupancy.py")
    print()
    print("Key metrics to look for in ncu report:")
    print("  - sm__warps_active.avg.pct_of_peak_sustained_active (achieved occupancy)")
    print("  - sm__maximum_warps_per_active_cycle_pct (theoretical occupancy)")
    print("  - launch__occupancy_limit_* (what's limiting occupancy)")
    print("=" * 60)


if __name__ == "__main__":
    module = compile_topk_kernel()
    run_profiling_workload(module)
