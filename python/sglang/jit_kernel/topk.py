from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING, Dict, Optional

import flashinfer
import torch

from sglang.jit_kernel.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module

logger = logging.getLogger(__name__)

# RadixRowState: histogram[3][256] (3072) + 5 scalars (20) = 3092 bytes.
# Round up to 4096 to account for compiler padding and potential future field additions.
_RADIX_ROW_STATE_SIZE = 4096
# Extra per-row for row_to_batch mapping in pagetable mode
_ROW_TO_BATCH_SIZE = 4  # int32


@cache_once
def _jit_topk_module(topk: int, dtype: torch.dtype) -> Module:
    args = make_cpp_args(topk, dtype)
    flashinfer_dir = pathlib.Path(flashinfer.__file__).parent.resolve()
    assert (
        flashinfer_dir / "data" / "include"
    ).exists(), (
        f"flashinfer headers are missing {str(flashinfer_dir / 'data' / 'include')}"
    )
    flashinfer_include_path = str((flashinfer_dir / "data" / "include").resolve())
    return load_jit(
        "topk",
        *args,
        cuda_files=["elementwise/topk.cuh"],
        cuda_wrappers=[
            ("topk_basic", f"TopKKernel<{args}>::run_basic"),
            ("topk_pagetable", f"TopKKernel<{args}>::run_pagetable"),
            ("topk_ragged", f"TopKKernel<{args}>::run_ragged"),
        ],
        extra_include_paths=[flashinfer_include_path],
    )


# ---- Workspace management ----
_workspace_buffers: Dict[torch.device, torch.Tensor] = {}


def _get_workspace(num_rows: int, device: torch.device) -> torch.Tensor:
    """Get or grow a workspace buffer for the topk kernel."""
    needed = num_rows * (_RADIX_ROW_STATE_SIZE + _ROW_TO_BATCH_SIZE)
    buf = _workspace_buffers.get(device)
    if buf is None or buf.numel() < needed:
        # Allocate with some headroom (at least 16MB or needed)
        alloc_size = max(needed, 16 * 1024 * 1024)
        _workspace_buffers[device] = torch.zeros(
            alloc_size, dtype=torch.uint8, device=device
        )
    return _workspace_buffers[device]


# ---- Availability check ----
@cache_once
def can_use_jit_topk() -> bool:
    """Check if the JIT topk kernel can be compiled and loaded."""
    try:
        _jit_topk_module(2048, torch.float32)
        return True
    except Exception as e:
        logger.warning(f"JIT topk kernel not available: {e}")
        return False


# ---- Public API ----
def fast_topk_v2_jit(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """JIT adaptive top-k: returns indices [B, topk] as int32."""
    assert score.dim() == 2
    B = score.size(0)
    topk_indices = score.new_empty((B, topk), dtype=torch.int32)
    workspace = _get_workspace(B, score.device)
    module = _jit_topk_module(topk, score.dtype)
    module.topk_basic(score, lengths, topk_indices, workspace, row_starts)
    return topk_indices


def fast_topk_transform_fused_jit(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """JIT adaptive top-k with page table transform: returns page indices [B, topk] as int32."""
    assert score.dim() == 2
    B = score.size(0)
    dst_page_table = score.new_empty((B, topk), dtype=torch.int32)
    workspace = _get_workspace(B, score.device)
    module = _jit_topk_module(topk, score.dtype)
    module.topk_pagetable(
        score, lengths, dst_page_table, page_table_size_1,
        cu_seqlens_q, workspace, row_starts
    )
    return dst_page_table


def fast_topk_transform_ragged_fused_jit(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices_offset: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """JIT adaptive top-k with ragged offset addition: returns indices [B, topk] as int32."""
    assert score.dim() == 2
    B = score.size(0)
    topk_indices_ragged = score.new_empty((B, topk), dtype=torch.int32)
    workspace = _get_workspace(B, score.device)
    module = _jit_topk_module(topk, score.dtype)
    module.topk_ragged(
        score, lengths, topk_indices_ragged,
        topk_indices_offset, workspace, row_starts
    )
    return topk_indices_ragged
