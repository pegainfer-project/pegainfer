// GLM5.2 TP4 prefill MoE routing shims: hand-written kernels (not vendored
// library code) bridging the router's top-k output to the FlashInfer grouped
// GEMM's m_indptr/gathered-row contract, plus the weighted expert combine.
//
//   glm52_prefill_moe_route_cuda        topk_idx -> {m_indptr, gather_rows,
//                                       route_slot} grouped-route metadata
//   glm52_prefill_moe_gather_rows_cuda  bf16 row gather into GEMM slot order
//   glm52_prefill_moe_gather_fp8_cuda   fp8 row + group-scale gather (the
//                                       caller quantizes once, gathers fp8)
//   glm52_prefill_moe_combine_cuda      deterministic weighted combine of the
//                                       W2 output with the shared-expert rows
//
// Determinism contract: the histogram pass assigns each (row, j) route its
// within-expert rank with atomicAdd, so slot order inside one expert is
// nondeterministic run to run. Downstream results are deterministic anyway:
// every grouped-GEMM row depends only on its own gathered source row, and
// the combine kernel reads through route_slot in fixed (row, j) order with
// f32 accumulation and a single bf16 rounding — no atomics on the value
// path. Out-of-range expert ids device-trap (the glm52_deepgemm_grouped
// metadata treatment): a bad id would multiply a bogus weight slab into real
// outputs with no error anywhere downstream.

#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <climits>

namespace {

constexpr int kThreads = 256;
constexpr int kScanThreads = 256;
constexpr int kMaxBlocks = 65535;
constexpr int kVecWidth = 8;  // bf16 per float4

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

int grid_blocks(long long work) {
  const long long blocks = (work + kThreads - 1) / kThreads;
  return blocks > kMaxBlocks ? kMaxBlocks : static_cast<int>(blocks);
}

// Pass 1: per-expert histogram; route_slot temporarily holds each route's
// within-expert rank (atomicAdd order — see the determinism contract above).
__global__ void route_histogram_kernel(const int* __restrict__ topk_idx,
                                       int* __restrict__ expert_counts,
                                       int* __restrict__ route_slot, int total,
                                       int num_experts) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += blockDim.x * gridDim.x) {
    const int expert = topk_idx[i];
    if (expert < 0 || expert >= num_experts) {
      __trap();
    }
    route_slot[i] = atomicAdd(expert_counts + expert, 1);
  }
}

// Pass 2: single-block exclusive scan of expert_counts into m_indptr
// (m_indptr[g] = rows before expert g, m_indptr[num_experts] = total).
// Chunked Hillis-Steele in shared memory with a running carry, so
// num_experts is not bounded by the block size.
__global__ void route_scan_kernel(const int* __restrict__ expert_counts,
                                  int* __restrict__ m_indptr,
                                  int num_experts) {
  __shared__ int prefix[kScanThreads];
  __shared__ int carry;
  if (threadIdx.x == 0) {
    carry = 0;
    m_indptr[0] = 0;
  }
  __syncthreads();
  for (int base = 0; base < num_experts; base += kScanThreads) {
    const int expert = base + threadIdx.x;
    prefix[threadIdx.x] = expert < num_experts ? expert_counts[expert] : 0;
    __syncthreads();
    for (int offset = 1; offset < kScanThreads; offset <<= 1) {
      const int addend =
          threadIdx.x >= offset ? prefix[threadIdx.x - offset] : 0;
      __syncthreads();
      prefix[threadIdx.x] += addend;
      __syncthreads();
    }
    const int inclusive = prefix[threadIdx.x] + carry;
    if (expert < num_experts) {
      m_indptr[expert + 1] = inclusive;
    }
    __syncthreads();
    if (threadIdx.x == kScanThreads - 1) {
      carry = inclusive;  // zero-padded chunk tail keeps this exact
    }
    __syncthreads();
  }
}

// Pass 3: slot = expert segment start + within-expert rank; record the
// slot -> source-row gather map and overwrite route_slot with the final
// (row, j) -> slot map. topk_idx was already range-checked in pass 1.
__global__ void route_placement_kernel(const int* __restrict__ topk_idx,
                                       const int* __restrict__ m_indptr,
                                       int* __restrict__ gather_rows,
                                       int* __restrict__ route_slot, int total,
                                       int topk) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
       i += blockDim.x * gridDim.x) {
    const int slot = m_indptr[topk_idx[i]] + route_slot[i];
    gather_rows[slot] = i / topk;
    route_slot[i] = slot;
  }
}

// output[slot] = input[gather_rows[slot]], float4-of-bf16x8 vectors. The
// destination is dense in slot order, so the flat vector index i addresses
// it directly.
__global__ void gather_rows_kernel(const __nv_bfloat16* __restrict__ input,
                                   const int* __restrict__ gather_rows,
                                   __nv_bfloat16* __restrict__ output,
                                   long long total_vecs, int vecs_per_row) {
  const long long stride =
      static_cast<long long>(blockDim.x) * gridDim.x;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) +
                     threadIdx.x;
       i < total_vecs; i += stride) {
    const int src_row = gather_rows[i / vecs_per_row];
    const int col = static_cast<int>(i % vecs_per_row);
    reinterpret_cast<float4*>(output)[i] = reinterpret_cast<const float4*>(
        input)[static_cast<long long>(src_row) * vecs_per_row + col];
  }
}

// fp8 payload + per-row group-scale gather: output[slot] =
// input[gather_rows[slot]] as an int4 byte copy (16 fp8/vector), and
// output_scale[slot] = input_scale[gather_rows[slot]] (k/128 floats). One
// kernel, two flat grid-stride ranges — the scale range is 1/128th of the
// payload range so it adds one tail iteration at most.
__global__ void gather_fp8_kernel(const unsigned char* __restrict__ input,
                                  const float* __restrict__ input_scale,
                                  const int* __restrict__ gather_rows,
                                  unsigned char* __restrict__ output,
                                  float* __restrict__ output_scale,
                                  long long total_vecs, int vecs_per_row,
                                  long long total_sf, int sf_per_row) {
  const long long stride =
      static_cast<long long>(blockDim.x) * gridDim.x;
  const long long start =
      blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  for (long long i = start; i < total_vecs; i += stride) {
    const int src_row = gather_rows[i / vecs_per_row];
    const int col = static_cast<int>(i % vecs_per_row);
    reinterpret_cast<int4*>(output)[i] = reinterpret_cast<const int4*>(
        input)[static_cast<long long>(src_row) * vecs_per_row + col];
  }
  for (long long i = start; i < total_sf; i += stride) {
    const int src_row = gather_rows[i / sf_per_row];
    const int col = static_cast<int>(i % sf_per_row);
    output_scale[i] =
        input_scale[static_cast<long long>(src_row) * sf_per_row + col];
  }
}

union Bf16x8 {
  float4 vec;
  __nv_bfloat162 pairs[4];
};

// output[row] = shared_out[row] + sum_j topk_weight[row, j] *
// w2_out[route_slot[row, j]], f32 accumulation in fixed j order, one bf16
// rounding at the end.
__global__ void combine_kernel(const __nv_bfloat16* __restrict__ w2_out,
                               const int* __restrict__ route_slot,
                               const float* __restrict__ topk_weight,
                               const __nv_bfloat16* __restrict__ shared_out,
                               __nv_bfloat16* __restrict__ output,
                               long long total_vecs, int vecs_per_row,
                               int topk) {
  const long long stride =
      static_cast<long long>(blockDim.x) * gridDim.x;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) +
                     threadIdx.x;
       i < total_vecs; i += stride) {
    const long long row = i / vecs_per_row;
    const int col = static_cast<int>(i % vecs_per_row);
    Bf16x8 in;
    in.vec = reinterpret_cast<const float4*>(shared_out)[i];
    float acc[kVecWidth];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      const float2 f = __bfloat1622float2(in.pairs[p]);
      acc[2 * p] = f.x;
      acc[2 * p + 1] = f.y;
    }
    for (int j = 0; j < topk; ++j) {
      const long long route = row * topk + j;
      const float weight = topk_weight[route];
      const long long slot = route_slot[route];
      in.vec = reinterpret_cast<const float4*>(
          w2_out)[slot * vecs_per_row + col];
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        const float2 f = __bfloat1622float2(in.pairs[p]);
        acc[2 * p] += weight * f.x;
        acc[2 * p + 1] += weight * f.y;
      }
    }
    Bf16x8 out;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      out.pairs[p] = __floats2bfloat162_rn(acc[2 * p], acc[2 * p + 1]);
    }
    reinterpret_cast<float4*>(output)[i] = out.vec;
  }
}

}  // namespace

extern "C" CUresult glm52_prefill_moe_route_cuda(
    const int* topk_idx, int rows, int topk, int num_experts,
    int* expert_counts, int* m_indptr, int* gather_rows, int* route_slot,
    CUstream stream) {
  if (!topk_idx || !expert_counts || !m_indptr || !gather_rows ||
      !route_slot || rows <= 0 || topk <= 0 || num_experts <= 0 ||
      static_cast<long long>(rows) * topk > INT_MAX) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  PEGAINFER_FFI_GUARD_BEGIN
  const auto s = reinterpret_cast<cudaStream_t>(stream);
  const int total = rows * topk;
  const cudaError_t err = cudaMemsetAsync(
      expert_counts, 0, static_cast<size_t>(num_experts) * sizeof(int), s);
  if (err != cudaSuccess) return map_cuda_error(err);
  const int blocks = grid_blocks(total);
  route_histogram_kernel<<<blocks, kThreads, 0, s>>>(
      topk_idx, expert_counts, route_slot, total, num_experts);
  route_scan_kernel<<<1, kScanThreads, 0, s>>>(expert_counts, m_indptr,
                                               num_experts);
  route_placement_kernel<<<blocks, kThreads, 0, s>>>(
      topk_idx, m_indptr, gather_rows, route_slot, total, topk);
  return consume_last_cuda_error();
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

extern "C" CUresult glm52_prefill_moe_gather_rows_cuda(
    const __nv_bfloat16* input, const int* gather_rows, __nv_bfloat16* output,
    int total, int hidden, CUstream stream) {
  if (!input || !gather_rows || !output || total < 0 || hidden <= 0 ||
      (hidden % kVecWidth) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (total == 0) {
    return CUDA_SUCCESS;
  }
  PEGAINFER_FFI_GUARD_BEGIN
  const int vecs_per_row = hidden / kVecWidth;
  const long long total_vecs =
      static_cast<long long>(total) * vecs_per_row;
  gather_rows_kernel<<<grid_blocks(total_vecs), kThreads, 0,
                       reinterpret_cast<cudaStream_t>(stream)>>>(
      input, gather_rows, output, total_vecs, vecs_per_row);
  return consume_last_cuda_error();
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

extern "C" CUresult glm52_prefill_moe_gather_fp8_cuda(
    const unsigned char* input, const float* input_scale,
    const int* gather_rows, unsigned char* output, float* output_scale,
    int total, int k, CUstream stream) {
  if (!input || !input_scale || !gather_rows || !output || !output_scale ||
      total < 0 || k <= 0 || (k % 128) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (total == 0) {
    return CUDA_SUCCESS;
  }
  PEGAINFER_FFI_GUARD_BEGIN
  const int vecs_per_row = k / 16;  // int4 = 16 fp8 bytes; 16 | 128 | k
  const int sf_per_row = k / 128;
  const long long total_vecs =
      static_cast<long long>(total) * vecs_per_row;
  gather_fp8_kernel<<<grid_blocks(total_vecs), kThreads, 0,
                      reinterpret_cast<cudaStream_t>(stream)>>>(
      input, input_scale, gather_rows, output, output_scale, total_vecs,
      vecs_per_row, static_cast<long long>(total) * sf_per_row, sf_per_row);
  return consume_last_cuda_error();
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

extern "C" CUresult glm52_prefill_moe_combine_cuda(
    const __nv_bfloat16* w2_out, const int* route_slot,
    const float* topk_weight, const __nv_bfloat16* shared_out,
    __nv_bfloat16* output, int rows, int topk, int hidden, CUstream stream) {
  if (!w2_out || !route_slot || !topk_weight || !shared_out || !output ||
      rows < 0 || topk <= 0 || hidden <= 0 || (hidden % kVecWidth) != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (rows == 0) {
    return CUDA_SUCCESS;
  }
  PEGAINFER_FFI_GUARD_BEGIN
  const int vecs_per_row = hidden / kVecWidth;
  const long long total_vecs =
      static_cast<long long>(rows) * vecs_per_row;
  combine_kernel<<<grid_blocks(total_vecs), kThreads, 0,
                   reinterpret_cast<cudaStream_t>(stream)>>>(
      w2_out, route_slot, topk_weight, shared_out, output, total_vecs,
      vecs_per_row, topk);
  return consume_last_cuda_error();
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}
