"""Tests for JIT adaptive top-k kernel.

Tests correctness against torch.topk reference across various batch sizes
and sequence lengths, including long sequences (131K+) where the multi-CTA
path activates.
"""
from typing import Optional

import pytest
import torch


def _check_jit_available():
    try:
        from sglang.jit_kernel.topk import can_use_jit_topk

        return can_use_jit_topk()
    except (ImportError, Exception):
        return False


requires_jit = pytest.mark.skipif(
    not _check_jit_available(), reason="JIT topk kernel not available"
)


def _ref_topk(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Reference implementation using torch.topk."""
    B = score.size(0)
    results = []
    for i in range(B):
        length = lengths[i].item()
        rs = row_starts[i].item() if row_starts is not None else 0
        row_score = score[i, rs : rs + length]
        if length <= topk:
            # Trivial: return all valid indices + padding with -1
            idx = torch.arange(length, dtype=torch.int32, device=score.device)
            pad = torch.full(
                (topk - length,), -1, dtype=torch.int32, device=score.device
            )
            results.append(torch.cat([idx, pad]))
        else:
            _, idx = torch.topk(row_score, topk, dim=-1, sorted=False)
            results.append(idx.int())
    return torch.stack(results)


def _sets_match(
    score: torch.Tensor,
    ref: torch.Tensor,
    our: torch.Tensor,
    max_errors: int = 5,
    row_starts: Optional[torch.Tensor] = None,
):
    """Check that two sets of topk indices select the same top values."""
    B = ref.size(0)
    total_errors = 0
    for i in range(B):
        rs = row_starts[i].item() if row_starts is not None else 0
        ref_set = set(ref[i].cpu().tolist())
        our_set = set(our[i].cpu().tolist())
        ref_set.discard(-1)
        our_set.discard(-1)
        extra = our_set - ref_set
        missing = ref_set - our_set
        if extra or missing:
            # Allow ties (same value at different indices)
            extra_vals = sorted(score[i, rs + idx].item() for idx in extra)
            missing_vals = sorted(score[i, rs + idx].item() for idx in missing)
            if extra_vals != missing_vals:
                total_errors += len(extra)
    assert total_errors <= max_errors, f"Too many mismatches: {total_errors}"


MAX_SEQ_LEN = 131072


@requires_jit
@pytest.mark.parametrize("bs", [1, 16, 64, 256])
@pytest.mark.parametrize("topk", [2048])
@pytest.mark.parametrize("seq_len", [2048, 4096, 16384, 65536, 131072])
@pytest.mark.parametrize("has_row_starts", [True, False])
@torch.inference_mode()
def test_jit_topk_basic(bs: int, topk: int, seq_len: int, has_row_starts: bool):
    """Test basic topk (indices only) against torch.topk reference."""
    from sglang.jit_kernel.topk import fast_topk_v2_jit

    torch.manual_seed(42)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

    row_starts = None
    if has_row_starts:
        max_offset = MAX_SEQ_LEN - seq_len
        if max_offset > 0:
            row_starts = torch.randint(
                0, min(max_offset, 2048), (bs,), dtype=torch.int32, device="cuda"
            )
        else:
            row_starts = torch.zeros(bs, dtype=torch.int32, device="cuda")

    ref = _ref_topk(score, lengths, topk, row_starts)
    out = fast_topk_v2_jit(score, lengths, topk, row_starts)

    _sets_match(score, ref, out, max_errors=5, row_starts=row_starts)


@requires_jit
@pytest.mark.parametrize("bs", [1, 16, 64])
@pytest.mark.parametrize("topk", [2048])
@pytest.mark.parametrize("seq_len", [2048, 16384, 65536])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
@torch.inference_mode()
def test_jit_topk_pagetable(bs: int, topk: int, seq_len: int, mode: str):
    """Test page table transform variant."""
    from sglang.jit_kernel.topk import fast_topk_transform_fused_jit

    torch.manual_seed(42)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

    # Identity page table for easy verification
    src_page_table = torch.arange(seq_len, dtype=torch.int32, device="cuda")
    src_page_table = src_page_table.unsqueeze(0).expand(bs, -1).contiguous()

    if mode == "decode":
        cu_seqlens_q = torch.arange(bs + 1, dtype=torch.int32, device="cuda")
        row_starts = None
    else:
        step = 4 if bs >= 4 else 1
        prefill_bs = bs // step
        cu_seqlens_q = torch.arange(
            0, bs + 1, step=step, dtype=torch.int32, device="cuda"
        )
        src_page_table = src_page_table[:prefill_bs]
        row_starts = torch.randint(
            0, min(MAX_SEQ_LEN - seq_len, 2048), (bs,), dtype=torch.int32, device="cuda"
        )

    ref = _ref_topk(score, lengths, topk, row_starts)
    # With identity page table, page_table[ref_idx] == ref_idx
    out = fast_topk_transform_fused_jit(
        score, lengths, src_page_table, cu_seqlens_q, topk, row_starts
    )

    _sets_match(score, ref, out, max_errors=5, row_starts=row_starts)


@requires_jit
@pytest.mark.parametrize("bs", [1, 16, 64])
@pytest.mark.parametrize("topk", [2048])
@pytest.mark.parametrize("seq_len", [2048, 16384, 65536])
@torch.inference_mode()
def test_jit_topk_ragged(bs: int, topk: int, seq_len: int):
    """Test ragged transform variant."""
    from sglang.jit_kernel.topk import fast_topk_transform_ragged_fused_jit

    torch.manual_seed(42)
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")
    row_starts = torch.randint(
        0, min(MAX_SEQ_LEN - seq_len, 2048), (bs,), dtype=torch.int32, device="cuda"
    )
    offsets = torch.randint(0, 1024, (bs,), dtype=torch.int32, device="cuda")

    ref = _ref_topk(score, lengths, topk, row_starts)
    out = fast_topk_transform_ragged_fused_jit(
        score, lengths, offsets, topk, row_starts
    )

    # Ragged adds offset: out[i] = ref[i] + offsets[i] for valid entries
    ref_with_offset = ref.clone()
    for i in range(bs):
        mask = ref_with_offset[i] != -1
        ref_with_offset[i][mask] += offsets[i]

    _sets_match(
        score, ref_with_offset, out, max_errors=5, row_starts=row_starts
    )


@requires_jit
@pytest.mark.parametrize("seq_len", [512, 1024])
@torch.inference_mode()
def test_jit_topk_trivial_case(seq_len: int):
    """Test when length <= topk (trivial case)."""
    from sglang.jit_kernel.topk import fast_topk_v2_jit

    torch.manual_seed(42)
    bs = 4
    topk = 2048
    score = torch.randn(bs, MAX_SEQ_LEN, dtype=torch.float32, device="cuda")
    lengths = torch.full((bs,), seq_len, dtype=torch.int32, device="cuda")

    out = fast_topk_v2_jit(score, lengths, topk)

    # For trivial case: indices [0, length) should be valid, rest should be -1
    for i in range(bs):
        valid = out[i, :seq_len]
        padding = out[i, seq_len:]
        valid_set = set(valid.cpu().tolist())
        expected_set = set(range(seq_len))
        assert valid_set == expected_set, f"Row {i}: expected {expected_set}, got {valid_set}"
        assert (padding == -1).all(), f"Row {i}: padding should be -1"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
