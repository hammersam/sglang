/**
 * JIT kernel wrapper for flashinfer adaptive top-k.
 * Provides TVM FFI interface to TopKDispatch / TopKPageTableTransformDispatch /
 * TopKRaggedTransformDispatch from topk_fi.cuh.
 */
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/topk_fi.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_runtime.h>

#include <cstdint>

namespace {

using namespace flashinfer::sampling;

// ---------------------------------------------------------------------------
// Helper: convert cu_seqlens_q → row_to_batch mapping
// cu_seqlens_q: [prefill_bs + 1] — cumulative head counts per batch element
// row_to_batch: [num_rows]       — batch_idx for each row
// ---------------------------------------------------------------------------
__global__ void cu_seqlens_to_row_to_batch_kernel(
    const int32_t* __restrict__ cu_seqlens_q,
    int32_t* __restrict__ row_to_batch,
    int32_t prefill_bs,
    int32_t num_rows) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_rows) return;
  // Binary search for the batch element containing this row.
  // cu_seqlens_q is monotonically non-decreasing with prefill_bs+1 entries.
  int lo = 0, hi = prefill_bs;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (cu_seqlens_q[mid + 1] <= tid) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  row_to_batch[tid] = lo;
}

// ---------------------------------------------------------------------------
// TopK kernel struct — templated on topk and dtype.
// Provides three entry points matching the sgl_kernel Python API.
// ---------------------------------------------------------------------------
template <int64_t kTopK, typename DType>
struct TopKKernel {
  // -----------------------------------------------------------------------
  // Basic top-k: returns indices only
  //   score:     [B, stride], float32
  //   lengths:   [B], int32
  //   output:    [B, topk], int32
  //   workspace: [ws_bytes], uint8
  //   row_starts: optional [B], int32
  // -----------------------------------------------------------------------
  static void run_basic(
      tvm::ffi::TensorView score,
      tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView workspace,
      tvm::ffi::Optional<tvm::ffi::TensorView> row_starts) {

    auto device = host::SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto B = host::SymbolicSize{"B"};
    auto S = host::SymbolicSize{"stride"};

    host::TensorMatcher({B, S})
        .with_strides({S, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(score);
    host::TensorMatcher({B})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(lengths);
    host::TensorMatcher({B, kTopK})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(output);

    const uint32_t num_rows = static_cast<uint32_t>(B.unwrap());
    const uint32_t max_len = static_cast<uint32_t>(S.unwrap());
    const cudaStream_t stream = host::LaunchKernel::resolve_device(device.unwrap());

    auto* input_ptr = static_cast<DType*>(score.data_ptr());
    auto* lengths_ptr = static_cast<int32_t*>(lengths.data_ptr());
    auto* output_ptr = static_cast<int32_t*>(output.data_ptr());
    const int32_t* row_starts_ptr = row_starts.has_value()
        ? static_cast<const int32_t*>(row_starts.value().data_ptr())
        : nullptr;

    // Workspace: RadixRowState per row
    auto* row_states = reinterpret_cast<RadixRowState*>(workspace.data_ptr());

    auto status = TopKDispatch<DType, int32_t>(
        input_ptr, output_ptr, /*output_values=*/nullptr,
        lengths_ptr, row_starts_ptr,
        num_rows, kTopK, max_len,
        row_states, stream);
    RuntimeCheck(status == cudaSuccess, "TopKDispatch failed: ", cudaGetErrorString(status));
  }

  // -----------------------------------------------------------------------
  // Page table transform: topk + lookup through page table
  //   score:          [B, stride]
  //   lengths:        [B]
  //   dst_page_table: [B, topk], int32 — output
  //   src_page_table: [prefill_bs, src_stride], int32
  //   cu_seqlens_q:   [prefill_bs + 1], int32
  //   workspace:      [ws_bytes], uint8
  //   row_starts:     optional [B], int32
  // -----------------------------------------------------------------------
  static void run_pagetable(
      tvm::ffi::TensorView score,
      tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView dst_page_table,
      tvm::ffi::TensorView src_page_table,
      tvm::ffi::TensorView cu_seqlens_q,
      tvm::ffi::TensorView workspace,
      tvm::ffi::Optional<tvm::ffi::TensorView> row_starts) {

    auto device = host::SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto B = host::SymbolicSize{"B"};
    auto S = host::SymbolicSize{"stride"};
    auto PB = host::SymbolicSize{"prefill_bs"};
    auto SS = host::SymbolicSize{"src_stride"};

    host::TensorMatcher({B, S})
        .with_strides({S, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(score);
    host::TensorMatcher({B})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(lengths);
    host::TensorMatcher({B, kTopK})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(dst_page_table);
    host::TensorMatcher({PB, SS})
        .with_strides({SS, 1})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(src_page_table);
    auto CU = host::SymbolicSize{"cu_seqlens_q_len"};
    host::TensorMatcher({CU})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(cu_seqlens_q);

    const uint32_t num_rows = static_cast<uint32_t>(B.unwrap());
    const uint32_t max_len = static_cast<uint32_t>(S.unwrap());
    const int64_t prefill_bs = PB.unwrap();
    const int64_t src_stride = SS.unwrap();
    const cudaStream_t stream = host::LaunchKernel::resolve_device(device.unwrap());

    auto* input_ptr = static_cast<DType*>(score.data_ptr());
    auto* lengths_ptr = static_cast<int32_t*>(lengths.data_ptr());
    auto* dst_ptr = static_cast<int32_t*>(dst_page_table.data_ptr());
    auto* src_ptr = static_cast<const int32_t*>(src_page_table.data_ptr());
    auto* cu_seqlens_ptr = static_cast<const int32_t*>(cu_seqlens_q.data_ptr());
    const int32_t* row_starts_ptr = row_starts.has_value()
        ? static_cast<const int32_t*>(row_starts.value().data_ptr())
        : nullptr;

    // Workspace layout: [RadixRowState * num_rows | row_to_batch * num_rows]
    auto* ws_ptr = static_cast<uint8_t*>(workspace.data_ptr());
    auto* row_states = reinterpret_cast<RadixRowState*>(ws_ptr);
    auto* row_to_batch = reinterpret_cast<int32_t*>(
        ws_ptr + num_rows * sizeof(RadixRowState));

    // Decode vs prefill dispatch (same logic as AOT kernel)
    const bool is_decode = !row_starts.has_value() && prefill_bs == static_cast<int64_t>(num_rows);

    const int32_t* row_to_batch_ptr = nullptr;
    if (!is_decode) {
      // Prefill: compute row_to_batch mapping from cu_seqlens_q
      const int threads = 256;
      const int blocks = (num_rows + threads - 1) / threads;
      cu_seqlens_to_row_to_batch_kernel<<<blocks, threads, 0, stream>>>(
          cu_seqlens_ptr, row_to_batch, static_cast<int32_t>(prefill_bs), num_rows);
      row_to_batch_ptr = row_to_batch;
    }

    auto status = TopKPageTableTransformDispatch<DType, int32_t>(
        input_ptr, dst_ptr, src_ptr, src_stride,
        row_to_batch_ptr, lengths_ptr, row_starts_ptr,
        num_rows, kTopK, max_len,
        row_states, stream);
    RuntimeCheck(status == cudaSuccess,
                 "TopKPageTableTransformDispatch failed: ", cudaGetErrorString(status));
  }

  // -----------------------------------------------------------------------
  // Ragged transform: topk + offset addition
  //   score:                [B, stride]
  //   lengths:              [B]
  //   output:               [B, topk], int32
  //   topk_indices_offset:  [B], int32
  //   workspace:            [ws_bytes], uint8
  //   row_starts:           optional [B], int32
  // -----------------------------------------------------------------------
  static void run_ragged(
      tvm::ffi::TensorView score,
      tvm::ffi::TensorView lengths,
      tvm::ffi::TensorView output,
      tvm::ffi::TensorView topk_indices_offset,
      tvm::ffi::TensorView workspace,
      tvm::ffi::Optional<tvm::ffi::TensorView> row_starts) {

    auto device = host::SymbolicDevice{};
    device.set_options<kDLCUDA>();
    auto B = host::SymbolicSize{"B"};
    auto S = host::SymbolicSize{"stride"};

    host::TensorMatcher({B, S})
        .with_strides({S, 1})
        .with_dtype<DType>()
        .with_device(device)
        .verify(score);
    host::TensorMatcher({B})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(lengths);
    host::TensorMatcher({B, kTopK})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(output);
    host::TensorMatcher({B})
        .with_dtype<int32_t>()
        .with_device(device)
        .verify(topk_indices_offset);

    const uint32_t num_rows = static_cast<uint32_t>(B.unwrap());
    const uint32_t max_len = static_cast<uint32_t>(S.unwrap());
    const cudaStream_t stream = host::LaunchKernel::resolve_device(device.unwrap());

    auto* input_ptr = static_cast<DType*>(score.data_ptr());
    auto* lengths_ptr = static_cast<int32_t*>(lengths.data_ptr());
    auto* output_ptr = static_cast<int32_t*>(output.data_ptr());
    auto* offsets_ptr = static_cast<const int32_t*>(topk_indices_offset.data_ptr());
    const int32_t* row_starts_ptr = row_starts.has_value()
        ? static_cast<const int32_t*>(row_starts.value().data_ptr())
        : nullptr;

    auto* row_states = reinterpret_cast<RadixRowState*>(workspace.data_ptr());

    auto status = TopKRaggedTransformDispatch<DType, int32_t>(
        input_ptr, output_ptr, offsets_ptr,
        lengths_ptr, row_starts_ptr,
        num_rows, kTopK, max_len,
        row_states, stream);
    RuntimeCheck(status == cudaSuccess,
                 "TopKRaggedTransformDispatch failed: ", cudaGetErrorString(status));
  }
};

}  // namespace
