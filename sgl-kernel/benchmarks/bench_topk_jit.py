"""Benchmark: JIT adaptive top-k vs AOT single-CTA top-k.

Usage:
    python bench_topk_jit.py [--include-aot]

Measures latency (ms) across batch sizes and sequence lengths.
When --include-aot is set, also benchmarks the AOT kernel for comparison.
"""
import argparse
import time

import torch


def bench_fn(fn, warmup=10, iters=100):
    """Benchmark a CUDA function, return median latency in ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    times.sort()
    return times[len(times) // 2]  # median


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-aot", action="store_true", help="Also benchmark AOT kernel")
    args = parser.parse_args()

    from sglang.jit_kernel.topk import can_use_jit_topk, fast_topk_v2_jit

    assert can_use_jit_topk(), "JIT topk not available"

    topk = 2048
    max_len = 131072

    batch_sizes = [1, 16, 64, 256]
    seq_lens = [4096, 16384, 32768, 65536, 131072]

    # Warm up JIT compilation
    print("Warming up JIT compilation...")
    score_warmup = torch.randn(1, max_len, dtype=torch.float32, device="cuda")
    lengths_warmup = torch.tensor([4096], dtype=torch.int32, device="cuda")
    fast_topk_v2_jit(score_warmup, lengths_warmup, topk)
    torch.cuda.synchronize()
    print("JIT compilation done.\n")

    aot_fn = None
    if args.include_aot:
        try:
            def _aot_topk(score, lengths, topk, row_starts=None):
                topk_indices = score.new_empty((score.size(0), topk), dtype=torch.int32)
                torch.ops.sgl_kernel.fast_topk(score, topk_indices, lengths, row_starts)
                return topk_indices
            aot_fn = _aot_topk
            print("AOT kernel available for comparison.\n")
        except Exception as e:
            print(f"AOT kernel not available: {e}\n")

    # Header
    header = f"{'bs':>6} {'seq_len':>8} {'JIT (ms)':>10}"
    if aot_fn:
        header += f" {'AOT (ms)':>10} {'speedup':>8}"
    print(header)
    print("-" * len(header))

    for bs in batch_sizes:
        for seq_len in seq_lens:
            score = torch.randn(bs, max_len, dtype=torch.float32, device="cuda")
            lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

            jit_ms = bench_fn(lambda: fast_topk_v2_jit(score, lengths, topk))

            line = f"{bs:>6} {seq_len:>8} {jit_ms:>10.3f}"

            if aot_fn and seq_len <= max_len:
                try:
                    aot_ms = bench_fn(lambda: aot_fn(score, lengths, topk))
                    speedup = aot_ms / jit_ms if jit_ms > 0 else float("inf")
                    line += f" {aot_ms:>10.3f} {speedup:>7.2f}x"
                except Exception:
                    line += f" {'N/A':>10} {'N/A':>8}"

            print(line)

        print()


if __name__ == "__main__":
    main()
