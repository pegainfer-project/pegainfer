// Kimi-K3 batched-decode MoE expert chain: the step-time kernels that wrap the
// FP8xFP4 masked grouped GEMM (`k3_deepgemm_fp8_fp4_grouped_sm100.cu`) into a
// complete routed-expert forward.
//
// Production routing goes through the fused MegaMoE kernel
// (`k3_mega_moe_sm100.cu`). This chain is the numerics anchor behind it: it is
// what the certified f32 reference is checked against, and what the fused
// kernel's golden gate is A/B'd against. Single rank only.
//
// ---------------------------------------------------------------------------
// Chain
// ---------------------------------------------------------------------------
//   latent [tokens, hidden] bf16 + router topk (idx [tokens, topk] i32,
//   weights [tokens, topk] f32)
//
//   1. k3_moe_local_route_metadata      -> masked_m[groups], slot_map[entries]
//   2/3. k3_moe_gather_fp8_quant_masked -> W13 A operand + f32 group scales
//        k3_fp8_scale_pack_ue8m0        -> packed UE8M0 i32 SFA
//   4. masked grouped GEMM W13 (other TU) -> [groups, cap, 2 * inter] bf16
//   5. k3_situ_and_mul_fp8_quant_masked -> W2 A operand + f32 group scales
//        k3_fp8_scale_pack_ue8m0        -> packed UE8M0 i32 SFA
//   6. masked grouped GEMM W2 (other TU)  -> [groups, cap, hidden] bf16
//   7. k3_moe_weighted_combine          -> [tokens, hidden] bf16
//
// The local gather is fused into the step-3 quant (the quant kernel reads the
// token-major latent row through the routing map and writes the masked slot),
// so no bf16 expert-major staging buffer exists.
//
// ---------------------------------------------------------------------------
// Routing map contract
// ---------------------------------------------------------------------------
// An "entry" is one expanded (token, topk-slot) pair, index `t * topk + s`, and
// the entry order IS the deterministic order everything downstream uses.
//
//   topk_idx[entry]  GLOBAL expert id, or anything outside `[local_expert_base,
//                    local_expert_base + groups)` to mark the entry inactive
//                    (that is how padded batch rows are excluded)
//   slot_map[entry]  `local_expert * masked_cap + rank` for active entries, -1
//                    for inactive ones, where `local_expert` is
//                    `topk_idx[entry] - local_expert_base` and `rank` counts
//                    the active entries of that expert that precede this one
//                    in entry order
//   masked_m[expert] number of active entries claiming that local expert
//
// `local_expert_base` shifts that window; the chain passes `0` with `groups`
// covering every routed expert. The parameter survives from the expert-parallel
// era and is kept because it costs one subtract and keeps the active-entry test
// in one place — every consumer re-derives the local expert the same way.
//
// Every consumer re-reads `topk_idx` and skips inactive entries rather than
// trusting `slot_map` alone, so a stale map can never be silently consumed.
//
// ---------------------------------------------------------------------------
// Activation quant recipe (A operand of both GEMMs)
// ---------------------------------------------------------------------------
// Identical to the GLM5.2 recipe the masked GEMM already consumes: fp8 e4m3,
// one scale per 1x128 K block, the scale rounded UP to a power of two (UE8M0 —
// the Blackwell packed-SF contract rejects arbitrary f32 scales), scales stored
// MN-major `[groups, k / 128, masked_cap]` f32 and then packed 4-per-i32 into
// `[groups, k / 512, masked_cap]`.
//
// ---------------------------------------------------------------------------
// CUDA-graph safety
// ---------------------------------------------------------------------------
// No allocation, no host readback, no device-side launch. Launch geometry
// depends only on (groups, masked_cap, hidden/inter) plus the token count,
// which is fixed for a captured graph; the per-entry loops are grid-strided so
// a capacity-shaped grid replays unchanged. Accumulation in the combine is a
// fixed topk-slot order with no atomics, so a replay is bit-reproducible.

#include "../common.cuh"
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

constexpr int kGroupSize = 128;
constexpr float kFp8Min = -448.0f;
constexpr float kFp8Max = 448.0f;
constexpr float kPerTokenGroupQuantEps = 1.0e-10f;

// Row-grid cap for the grid-strided per-entry kernels: enough blocks to fill
// the SMs at 128 threads/block, small enough that a capacity-shaped launch does
// not pay a block schedule per idle entry.
constexpr int kMaxEntryBlocks = 256;
constexpr int kMetadataThreads = 256;
constexpr int kMetadataWarps = kMetadataThreads / WARP_SIZE;
constexpr int kCombineThreads = 256;

__device__ __forceinline__ unsigned char quantize_e4m3(float value,
                                                       float scale) {
  float q = fminf(fmaxf(value / scale, kFp8Min), kFp8Max);
  return __nv_cvt_float_to_fp8(q, __NV_SATFINITE, __NV_E4M3);
}

// Next power of two >= s, by bumping the mantissa into the exponent field.
// Exact bit manipulation, no log2f rounding hazard; s is always positive,
// normal, and far from f32 max here.
__device__ __forceinline__ float round_up_pow2(float s) {
  return __uint_as_float((__float_as_uint(s) + 0x007FFFFFu) & 0x7F800000u);
}

// Block-wide max over one 128-wide K group, then the UE8M0 group scale.
// `shared` must hold blockDim.x floats; on return every thread sees the scale
// in `shared[0]`.
__device__ __forceinline__ float group_ue8m0_scale(float value, float* shared) {
  const int tid = threadIdx.x;
  shared[tid] = fabsf(value);
  __syncthreads();
#pragma unroll
  for (int stride = kGroupSize / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    shared[0] = round_up_pow2(fmaxf(shared[0], kPerTokenGroupQuantEps) / kFp8Max);
  }
  __syncthreads();
  return shared[0];
}

// Routing metadata, one block per local expert.
//
// Each block streams the whole entry array in entry order and compacts its own
// expert's entries with a ballot-based block scan, so `rank` — and therefore
// the masked row every entry lands in — is a pure function of the routing
// input, independent of block scheduling.
//
// The -1 fill for inactive entries is NOT done by the claiming block (no block
// claims them); entry `i` is filled by block `i % groups`, a partition disjoint
// from every claimed write, so the two write sets never race.
//
// An expert claiming more than `masked_cap` entries would alias the next
// expert's rows; that traps rather than silently corrupting, matching the
// GLM5.2 metadata kernel's contract.
__global__ void moe_local_route_metadata_kernel(const int* __restrict__ topk_idx,
                                                int* __restrict__ masked_m,
                                                int* __restrict__ slot_map,
                                                int entries, int groups,
                                                int masked_cap,
                                                int local_expert_base) {
  const int expert = blockIdx.x;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp = threadIdx.x / WARP_SIZE;

  __shared__ int warp_counts[kMetadataWarps];
  __shared__ int claimed;
  if (threadIdx.x == 0) {
    claimed = 0;
  }
  __syncthreads();

  for (int chunk = 0; chunk < entries; chunk += kMetadataThreads) {
    const int entry = chunk + threadIdx.x;
    const int routed = entry < entries ? topk_idx[entry] : -1;
    // `routed` is a global expert id; `local` is its index in this rank's
    // window. At base 0 this is the identity and the test below is exactly the
    // single-rank `routed >= 0 && routed < groups`.
    const int local = routed - local_expert_base;
    const bool active = routed >= 0 && local >= 0 && local < groups;
    if (entry < entries && !active && entry % groups == expert) {
      slot_map[entry] = -1;
    }

    const int mine = active && local == expert ? 1 : 0;
    const unsigned int ballot = __ballot_sync(0xFFFFFFFFu, mine);
    if (lane == 0) {
      warp_counts[warp] = __popc(ballot);
    }
    __syncthreads();

    if (mine != 0) {
      int rank = claimed + __popc(ballot & ((1u << lane) - 1u));
      for (int w = 0; w < warp; ++w) {
        rank += warp_counts[w];
      }
      if (rank >= masked_cap) {
        __trap();
      }
      slot_map[entry] = expert * masked_cap + rank;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      int total = 0;
#pragma unroll
      for (int w = 0; w < kMetadataWarps; ++w) {
        total += warp_counts[w];
      }
      claimed += total;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    masked_m[expert] = claimed;
  }
}

// Local gather fused with the W13 A-operand quant. Grid-strided over entries
// (blockIdx.x) x K groups (blockIdx.y); the entry index, and therefore the
// branch, is uniform across the block, so the __syncthreads in the reduction
// stay collective.
__global__ void gather_fp8_quant_bf16_k128_masked_kernel(
    const __nv_bfloat16* __restrict__ latent, const int* __restrict__ topk_idx,
    const int* __restrict__ slot_map, unsigned char* __restrict__ output,
    float* __restrict__ scales, int entries, int topk, int hidden, int groups,
    int masked_cap, int local_expert_base) {
  const int k_group = blockIdx.y;
  const int col = k_group * kGroupSize + threadIdx.x;
  const int scale_cols = hidden / kGroupSize;

  __shared__ float shared[kGroupSize];
  for (int entry = blockIdx.x; entry < entries; entry += gridDim.x) {
    const int expert = topk_idx[entry] - local_expert_base;
    if (expert < 0 || expert >= groups) {
      continue;
    }
    const int slot = slot_map[entry];
    const int token = entry / topk;
    const float value =
        __bfloat162float(latent[(size_t)token * hidden + col]);

    const float scale = group_ue8m0_scale(value, shared);
    if (threadIdx.x == 0) {
      const int row = slot % masked_cap;
      scales[((size_t)expert * scale_cols + k_group) * masked_cap + row] = scale;
    }
    output[(size_t)slot * hidden + col] = quantize_e4m3(value, scale);
    __syncthreads();
  }
}

// K3 "situ" activation + the W2 A-operand quant. The gate|up rows are already
// in the masked layout (the W13 GEMM wrote them), gate in the first `inter`
// columns and up in the second.
//
// Certified spelling, in f32 over the bf16 GEMM output (that bf16 store IS the
// activation's rounding step):
//     act = 4 * tanh(g / 4) * sigmoid(g) * 25 * tanh(u / 25)
// Router weights are deliberately not applied here; they are applied once, in
// the combine, after W2.
__global__ void situ_and_mul_fp8_quant_bf16_k128_masked_kernel(
    const __nv_bfloat16* __restrict__ gate_up, const int* __restrict__ topk_idx,
    const int* __restrict__ slot_map, unsigned char* __restrict__ output,
    float* __restrict__ scales, int entries, int topk, int inter, int groups,
    int masked_cap, int local_expert_base) {
  const int k_group = blockIdx.y;
  const int group_start = k_group * kGroupSize;
  const int col = group_start + threadIdx.x;
  const int input_stride = inter * 2;
  const int scale_cols = inter / kGroupSize;

  __shared__ float shared[kGroupSize];
  for (int entry = blockIdx.x; entry < entries; entry += gridDim.x) {
    const int expert = topk_idx[entry] - local_expert_base;
    if (expert < 0 || expert >= groups) {
      continue;
    }
    const int slot = slot_map[entry];
    const __nv_bfloat16* row_gate =
        gate_up + (size_t)slot * input_stride + group_start;
    const __nv_bfloat16* row_up = row_gate + inter;
    const float g = __bfloat162float(row_gate[threadIdx.x]);
    const float u = __bfloat162float(row_up[threadIdx.x]);
    const float sigmoid_g = 1.0f / (1.0f + expf(-g));
    const float activated =
        4.0f * tanhf(g * 0.25f) * sigmoid_g * (25.0f * tanhf(u / 25.0f));

    const float scale = group_ue8m0_scale(activated, shared);
    if (threadIdx.x == 0) {
      const int row = slot % masked_cap;
      scales[((size_t)expert * scale_cols + k_group) * masked_cap + row] = scale;
    }
    output[(size_t)slot * inter + col] = quantize_e4m3(activated, scale);
    __syncthreads();
  }
}

// f32 MN-major scales [groups, scale_cols, cap] -> packed UE8M0 i32
// [groups, scale_cols / 4, cap], LSB-first exponent bytes
// (u32 = b0 | b1<<8 | b2<<16 | b3<<24). Dense full-cover pass: every packed
// word is rewritten each step, so a stale byte never reaches the GEMM. Inputs
// must already be powers of two (the quant kernels above emit them); the SM100
// kernel device-asserts exponent-only values.
__global__ void fp8_scale_pack_ue8m0_kernel(const float* __restrict__ scales,
                                            int* __restrict__ packed,
                                            int groups, int scale_cols,
                                            int cap) {
  const int packed_cols = scale_cols / 4;
  const size_t total = (size_t)groups * packed_cols * cap;
  for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
       idx += (size_t)gridDim.x * blockDim.x) {
    const int m = idx % cap;
    const size_t gi = idx / cap;
    const float* base = scales + (gi * 4) * (size_t)cap + m;
    unsigned int word = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const unsigned int bits = __float_as_uint(base[(size_t)j * cap]);
      word |= ((bits >> 23) & 0xFFu) << (8 * j);
    }
    packed[idx] = static_cast<int>(word);
  }
}

// Weighted combine: masked expert rows -> token-major hidden states.
//
// One block column chunk per (token, column block); each thread owns one column
// and walks the token's topk slots IN SLOT ORDER, accumulating in f32 with no
// atomics, and rounds to bf16 exactly once at the end. Tokens with no active
// entry land an exact zero, so the output is fully covered every step.
__global__ void moe_weighted_combine_kernel(
    const __nv_bfloat16* __restrict__ expert_out,
    const int* __restrict__ topk_idx, const int* __restrict__ slot_map,
    const float* __restrict__ topk_weight, __nv_bfloat16* __restrict__ out,
    int topk, int hidden, int groups, int masked_cap) {
  const int token = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (col >= hidden) {
    return;
  }
  float acc = 0.0f;
  for (int slot_index = 0; slot_index < topk; ++slot_index) {
    const int entry = token * topk + slot_index;
    const int expert = topk_idx[entry];
    if (expert < 0 || expert >= groups) {
      continue;
    }
    const int slot = slot_map[entry];
    const float value =
        __bfloat162float(expert_out[(size_t)slot * hidden + col]);
    acc = fmaf(topk_weight[entry], value, acc);
  }
  out[(size_t)token * hidden + col] = __float2bfloat16_rn(acc);
}

CUresult map_cuda_error(cudaError_t err) {
  if (err == cudaSuccess) return CUDA_SUCCESS;
  if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (err == cudaErrorMemoryAllocation) return CUDA_ERROR_OUT_OF_MEMORY;
  if (err == cudaErrorNotSupported) return CUDA_ERROR_NOT_SUPPORTED;
  return CUDA_ERROR_LAUNCH_FAILED;
}

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

int entry_grid(int entries) {
  return entries < kMaxEntryBlocks ? entries : kMaxEntryBlocks;
}

bool valid_route_shape(int tokens, int topk, int groups, int masked_cap) {
  return tokens > 0 && topk > 0 && groups > 0 && masked_cap > 0 &&
         masked_cap % kGroupSize == 0 &&
         (long long)tokens * topk <= (long long)INT32_MAX;
}

}  // namespace

extern "C" {

// Routing metadata for one rank's expert window. `topk_idx` is `[tokens, topk]`
// i32 global expert ids; an entry is active when `topk_idx - local_expert_base`
// lands in [0, groups). Writes `masked_m[groups]` and `slot_map[tokens * topk]`.
// A single-rank chain passes `local_expert_base = 0`.
CUresult k3_moe_local_route_metadata_cuda(const int* topk_idx, int* masked_m,
                                          int* slot_map, int tokens, int topk,
                                          int groups, int masked_cap,
                                          int local_expert_base,
                                          cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (topk_idx == nullptr || masked_m == nullptr || slot_map == nullptr ||
      local_expert_base < 0 ||
      !valid_route_shape(tokens, topk, groups, masked_cap)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  moe_local_route_metadata_kernel<<<groups, kMetadataThreads, 0, stream>>>(
      topk_idx, masked_m, slot_map, tokens * topk, groups, masked_cap,
      local_expert_base);
  return consume_last_cuda_error();
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

// Local gather + FP8 e4m3 per-token group-128 quant into the masked layout.
// `latent` is `[tokens, hidden]` bf16; `output` is
// `[groups * masked_cap, hidden]` fp8; `scales` is
// `[groups, hidden / 128, masked_cap]` f32 (UE8M0 values, MN-major).
CUresult k3_moe_gather_fp8_quant_masked_cuda(
    const __nv_bfloat16* latent, const int* topk_idx, const int* slot_map,
    unsigned char* output, float* scales, int tokens, int topk, int hidden,
    int groups, int masked_cap, int local_expert_base, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (latent == nullptr || topk_idx == nullptr || slot_map == nullptr ||
      output == nullptr || scales == nullptr || local_expert_base < 0 ||
      !valid_route_shape(tokens, topk, groups, masked_cap) || hidden <= 0 ||
      hidden % kGroupSize != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int entries = tokens * topk;
  dim3 grid(entry_grid(entries), hidden / kGroupSize, 1);
  gather_fp8_quant_bf16_k128_masked_kernel<<<grid, kGroupSize, 0, stream>>>(
      latent, topk_idx, slot_map, output, scales, entries, topk, hidden, groups,
      masked_cap, local_expert_base);
  return consume_last_cuda_error();
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

// K3 situ activation over the masked gate|up rows + FP8 quant of the product.
// `gate_up` is `[groups * masked_cap, 2 * inter]` bf16 (gate first), `output`
// is `[groups * masked_cap, inter]` fp8, `scales` is
// `[groups, inter / 128, masked_cap]` f32.
CUresult k3_situ_and_mul_fp8_quant_masked_cuda(
    const __nv_bfloat16* gate_up, const int* topk_idx, const int* slot_map,
    unsigned char* output, float* scales, int tokens, int topk, int inter,
    int groups, int masked_cap, int local_expert_base, cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (gate_up == nullptr || topk_idx == nullptr || slot_map == nullptr ||
      output == nullptr || scales == nullptr || local_expert_base < 0 ||
      !valid_route_shape(tokens, topk, groups, masked_cap) || inter <= 0 ||
      inter % kGroupSize != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int entries = tokens * topk;
  dim3 grid(entry_grid(entries), inter / kGroupSize, 1);
  situ_and_mul_fp8_quant_bf16_k128_masked_kernel<<<grid, kGroupSize, 0,
                                                   stream>>>(
      gate_up, topk_idx, slot_map, output, scales, entries, topk, inter, groups,
      masked_cap, local_expert_base);
  return consume_last_cuda_error();
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

// f32 MN-major group scales -> the packed UE8M0 i32 SFA tensor the masked GEMM
// reads. `scale_cols` is `k / 128` and must be a multiple of 4.
CUresult k3_fp8_scale_pack_ue8m0_cuda(const float* scales, int* packed,
                                      int groups, int scale_cols, int cap,
                                      cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (scales == nullptr || packed == nullptr || groups <= 0 || cap <= 0 ||
      scale_cols <= 0 || scale_cols % 4 != 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const size_t total = (size_t)groups * (scale_cols / 4) * cap;
  const int threads = 256;
  const size_t needed = (total + threads - 1) / threads;
  const int blocks = static_cast<int>(needed < 256 ? needed : 256);
  fp8_scale_pack_ue8m0_kernel<<<blocks, threads, 0, stream>>>(
      scales, packed, groups, scale_cols, cap);
  return consume_last_cuda_error();
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

// Weighted combine of the masked W2 output back to token-major hidden states.
// `expert_out` is `[groups * masked_cap, hidden]` bf16, `topk_weight` is
// `[tokens, topk]` f32, `out` is `[tokens, hidden]` bf16.
CUresult k3_moe_weighted_combine_cuda(const __nv_bfloat16* expert_out,
                                      const int* topk_idx, const int* slot_map,
                                      const float* topk_weight,
                                      __nv_bfloat16* out, int tokens, int topk,
                                      int hidden, int groups, int masked_cap,
                                      cudaStream_t stream) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (expert_out == nullptr || topk_idx == nullptr || slot_map == nullptr ||
      topk_weight == nullptr || out == nullptr ||
      !valid_route_shape(tokens, topk, groups, masked_cap) || hidden <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  dim3 grid(tokens, (hidden + kCombineThreads - 1) / kCombineThreads, 1);
  moe_weighted_combine_kernel<<<grid, kCombineThreads, 0, stream>>>(
      expert_out, topk_idx, slot_map, topk_weight, out, topk, hidden, groups,
      masked_cap);
  return consume_last_cuda_error();
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

}  // extern "C"
