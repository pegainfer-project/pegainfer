// GLM5.2 plain/batched weight-only FP8 linear projections: bf16 activation,
// on-the-fly e4m3 block-scale weight dequant, and f32 accumulation. Dequant
// matches the host reference: deq(W) = float(e4m3(W)) * weight_scale_inv.
//
// One __device__ row-tile core serves the plain linear path: one weight matrix
// and one broadcast activation row.
//
// Activation is NOT staged in shared: it is read straight from global (L2-resident,
// reused across every block). Staging it cost a 32KB-shared prologue + __syncthreads +
// a shared-occupancy cap that throttled the long-K GEMVs hardest. Dropping the stage
// (H200 sm_90 microbench, bit-identical checksum):
//   W13 gate_up 67->83%, W2 down 63->75%, o_proj 54->73%, q_b 59->79% HBM BW.
// (Blog "Twelve Attempts": keep the reused vector hot in L2, not shared; shared only
// pays off for re-read data, and a streamed GEMV re-reads nothing.) rows/warp is the
// second lever: short-K amortises per-block overhead with 4 rows/warp, while long-K
// fills the single grid column with 1 row/warp.

#include "../common.cuh"  // warp_reduce_sum
#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kWarpSize     = 32;
constexpr int kWarpsPerBlk  = 8;
constexpr int kBlockThreads = kWarpSize * kWarpsPerBlk;       // 256
constexpr int kVec          = 16;                             // 16 e4m3 = 128-bit LDG
constexpr int kFp8Block     = 128;
constexpr int kStep         = kWarpSize * kVec;               // 512 k per warp step

// The plain path dispatch follows the executing architecture. Hopper
// amortises short-K launches with four rows/warp; Blackwell uses one row/warp
// for the measured decode shapes. A row's accumulation order is unchanged.
constexpr int kRowsPlainShortKHopper = 4;  // k <= 2048, measured on SM90
constexpr int kRowsPlainBlackwell    = 1;  // all K, measured on SM103
constexpr int kRowsPlainLongK        = 1;  // k > 2048 on every architecture

CUresult consume_last_cuda_error() { return map_cuda_error(cudaGetLastError()); }

// The activation is read with 16-byte float4/uint4 loads straight from global (no shared
// staging absorbs misalignment any more), so its base must be 16-byte aligned. CudaSlice
// allocation bases are 256B-aligned and satisfy this; guard here so a future misaligned
// sub-view crashes early at the boundary instead of faulting mid-kernel.
bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15u) == 0; }

// One warp owns ROWS output rows starting at n0; sweeps the whole K of a single weight
// matrix (w_base/scale_base already resolved to this expert) against the bf16 activation
// row (xs, read straight from global/L2 -- not staged). No split-K, no cross-warp
// reduction. ROWS is the per-path tile (grouped 4; plain dispatched on k, see launcher).
template <int ROWS>
__device__ __forceinline__ void gemv_row_tile(
    const __nv_bfloat16* __restrict__ xs,        // activation [k], read from global/L2
    const unsigned char* __restrict__ w_base,    // fp8 e4m3 [n, k] for this expert
    const float* __restrict__ scale_base,        // f32 [n/128, k/128] for this expert
    __nv_bfloat16* __restrict__ out_row,         // [n] output for this slot
    int n, int k, int block_y) {
  constexpr int kRowsPerBlock = kWarpsPerBlk * ROWS;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n0   = block_y * kRowsPerBlock + warp * ROWS;
  const int scale_cols = k >> 7;                 // k/128
  // All ROWS rows share one weight-scale row: n0 is a multiple of ROWS and ROWS | 128,
  // so n0..n0+ROWS-1 never straddle a /128 boundary.
  const float* scale_row = scale_base + (size_t)(n0 >> 7) * scale_cols;
  const float4* xs4 = reinterpret_cast<const float4*>(xs);

  float acc[ROWS];
#pragma unroll
  for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;

  for (int kk = lane * kVec; kk < k; kk += kStep) {
    const float scale = scale_row[kk >> 7];      // one f32 per (lane, k-chunk)
    // Keep the 16 activations as raw bf16 bits (8 regs), convert inline at use --
    // materialising float x[16] cost 16 regs and capped occupancy (ncu: reg-limited).
    float4 xv0 = xs4[(kk >> 3)];
    float4 xv1 = xs4[(kk >> 3) + 1];
    const __nv_bfloat16* xh0 = reinterpret_cast<const __nv_bfloat16*>(&xv0);
    const __nv_bfloat16* xh1 = reinterpret_cast<const __nv_bfloat16*>(&xv1);
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      // 128-bit coalesced e4m3 weight load: 16 contiguous weights of row (n0+r).
      // __ldcs = cache-streaming (evict-first): the weight is read exactly once per
      // GEMV, so keeping it out of L1 stops one-shot data from evicting the staged
      // activation's working set. Microbench (long-K o_proj k=16384): 1.23x HBM BW,
      // neutral on short-K, zero accuracy/register cost. (KernelWiki: memory-bound
      // GEMV -> cache-policy differentiation is THE lever, not cp.async/ILP.)
      const uint4 wp = __ldcs(reinterpret_cast<const uint4*>(
          w_base + ((size_t)(n0 + r) * k) + kk));
      // Vectorised dequant: decode the 16 e4m3 weights two-at-a-time via the hardware
      // fp8x2->half2 path (8 cvts, not 16). e4m3 (3 mantissa bits) is exactly
      // representable in f16, so this is bit-identical to scalar `float(e4m3)` while
      // halving the decode-instruction count that caps this memory-bound GEMV below
      // peak BW. Activation stays inline bf16 (materialising float x[16] is reg-limited,
      // see above). Microbench grouped R4: +5-7% HBM BW, re 0.0000. (KernelWiki
      // contest-gpumode-p1: vectorise the low-precision dequant.)
      const __nv_fp8x2_e4m3* w2 = reinterpret_cast<const __nv_fp8x2_e4m3*>(&wp);
      float partial = 0.0f;
      // Per-term accumulation (NOT pair-grouped): preserves the original left-to-right
      // FMA association, so `partial` is bit-identical to the scalar path -- greedy
      // decode over 78 layers is sensitive to sub-ULP wobble and would otherwise
      // diverge. The only change vs scalar is the weight *decode* (fp8x2->half2), and
      // __low2float(h) == float(e4m3) exactly (e4m3 mantissa fits in f16).
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __half2 h = static_cast<__half2>(w2[j]);
        partial += __low2float(h) * __bfloat162float(xh0[2 * j]);
        partial += __high2float(h) * __bfloat162float(xh0[2 * j + 1]);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __half2 h = static_cast<__half2>(w2[4 + j]);
        partial += __low2float(h) * __bfloat162float(xh1[2 * j]);
        partial += __high2float(h) * __bfloat162float(xh1[2 * j + 1]);
      }
      acc[r] += scale * partial;                 // block-scale hoisted (const over 128 cols)
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r) {
    float v = warp_reduce_sum(acc[r]);           // common.cuh
    if (lane == 0) out_row[n0 + r] = __float2bfloat16(v);
  }
}

// Plain linear GEMV: single weight matrix, broadcast activation (groups=1). ROWS is
// dispatched on k by the launcher (short-K 4, long-K 1).
template <int ROWS>
__global__ void glm52_fp8_weight_only_gemv_kernel(
    const __nv_bfloat16* __restrict__ activation,  // [k]
    const unsigned char* __restrict__ weight,      // [n, k] e4m3
    const float* __restrict__ weight_scale,        // [n/128, k/128]
    __nv_bfloat16* __restrict__ out,               // [n]
    int n, int k) {
  gemv_row_tile<ROWS>(activation, weight, weight_scale, out, n, k,
                      blockIdx.y);  // x straight from L2
}

// Two independent projections with one activation and one graph node. The
// grids are concatenated, not the weights: q_a and kv_a keep their checkpoint
// layout and write their existing output buffers. This removes the second
// launch while preserving each row's exact dot-product order.
__global__ void glm52_fp8_weight_only_gemv_pair_kernel(
    const __nv_bfloat16* __restrict__ activation,
    const unsigned char* __restrict__ weight_a,
    const float* __restrict__ weight_scale_a,
    __nv_bfloat16* __restrict__ out_a, int n_a,
    const unsigned char* __restrict__ weight_b,
    const float* __restrict__ weight_scale_b,
    __nv_bfloat16* __restrict__ out_b, int n_b, int k) {
  const int blocks_a = n_a / kWarpsPerBlk;
  if (blockIdx.y < blocks_a) {
    gemv_row_tile<1>(activation, weight_a, weight_scale_a, out_a, n_a, k,
                     blockIdx.y);
  } else {
    gemv_row_tile<1>(activation, weight_b, weight_scale_b, out_b, n_b, k,
                     blockIdx.y - blocks_a);
  }
}

// Batched weight-stationary GEMV: a warp owns ROWS output rows and sweeps K once,
// carrying ROWS x BATCH accumulators. The weight packet is read exactly once per row
// (the whole point of batching a weight-memory-bound GEMV) and reused across all
// BATCH activation rows; the activation chunk is read once per warp into registers
// and reused across the ROWS weight rows. Each row's dot uses the same lane sweep,
// per-term order, and warp reduction as gemv_row_tile<1>, so every row is
// bit-identical to the m=1 kernel run alone.
//
// Why ROWS matters (H200 microbench + in-situ ncu, 2026-07-05): at BATCH=8 the
// 1-row/warp layout issues 16 activation LDGs per weight LDG and saturates the
// L1TEX port (93% L1/TEX vs 12% DRAM) — this was the dominant cost of the
// bucket-8 decode step. Shared staging does not help (LDS shares the L1TEX
// pipe); registers are the only storage off that port, so reusing the chunk
// across ROWS=4 rows cuts activation loads 4x: o_proj 132→66 µs, dense_dn
// 99→51, q_b 42→26. ROWS=8 spills; WARPS=4 keeps 3 blocks/SM resident at
// 158 regs measured (ptxas sm_90; BATCH=4 is 128). BATCH=2
// stays at ROWS=1: activation traffic is 2:16 against the weight there and
// the 1-row/warp shape measures faster on 7 of 10 model shapes.
template <int BATCH, int ROWS, int WARPS>
__global__ __launch_bounds__(WARPS* kWarpSize) void
glm52_fp8_weight_only_gemv_batched_kernel(
    const __nv_bfloat16* __restrict__ activation,  // [BATCH, k]
    const unsigned char* __restrict__ weight,      // [n, k] e4m3
    const float* __restrict__ weight_scale,        // [n/128, k/128]
    __nv_bfloat16* __restrict__ out,               // [BATCH, n]
    int n, int k) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n0   = blockIdx.y * (WARPS * ROWS) + warp * ROWS;
  const int scale_cols = k >> 7;
  // All ROWS rows share one weight-scale row: n0 is a multiple of ROWS and ROWS | 128.
  const float* scale_row = weight_scale + (size_t)(n0 >> 7) * scale_cols;

  float acc[ROWS][BATCH];
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int b = 0; b < BATCH; ++b) acc[r][b] = 0.0f;

  for (int kk = lane * kVec; kk < k; kk += kStep) {
    const float scale = scale_row[kk >> 7];
    float4 xv[BATCH][2];
#pragma unroll
    for (int b = 0; b < BATCH; ++b) {
      const float4* xs4 = reinterpret_cast<const float4*>(activation + (size_t)b * k);
      xv[b][0] = xs4[(kk >> 3)];
      xv[b][1] = xs4[(kk >> 3) + 1];
    }
#pragma unroll
    for (int r = 0; r < ROWS; ++r) {
      const uint4 wp = __ldcs(reinterpret_cast<const uint4*>(
          weight + ((size_t)(n0 + r) * k) + kk));
      const __nv_fp8x2_e4m3* w2 = reinterpret_cast<const __nv_fp8x2_e4m3*>(&wp);
#pragma unroll
      for (int b = 0; b < BATCH; ++b) {
        const __nv_bfloat16* xh0 = reinterpret_cast<const __nv_bfloat16*>(&xv[b][0]);
        const __nv_bfloat16* xh1 = reinterpret_cast<const __nv_bfloat16*>(&xv[b][1]);
        float partial = 0.0f;
        // Same per-term left-to-right association as gemv_row_tile — bit parity
        // per row is the contract (see that kernel's comment).
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          __half2 h = static_cast<__half2>(w2[j]);
          partial += __low2float(h) * __bfloat162float(xh0[2 * j]);
          partial += __high2float(h) * __bfloat162float(xh0[2 * j + 1]);
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          __half2 h = static_cast<__half2>(w2[4 + j]);
          partial += __low2float(h) * __bfloat162float(xh1[2 * j]);
          partial += __high2float(h) * __bfloat162float(xh1[2 * j + 1]);
        }
        acc[r][b] += scale * partial;
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROWS; ++r)
#pragma unroll
    for (int b = 0; b < BATCH; ++b) {
      float v = warp_reduce_sum(acc[r][b]);
      if (lane == 0) out[(size_t)b * n + n0 + r] = __float2bfloat16(v);
    }
}

// ---------------------------------------------------------------------------
// Tensor-core batched path (batches 4/8, whitelisted shapes where it wins).
//
// At batch 8 the register-tile kernel above is compute-walled: the per-term
// f32 FMA chain runs BATCH x 16 terms per 16-byte weight packet and caps
// o_proj at ~66 us against a 28.5 us weight-read floor (H200, ncu: 65% SM,
// occupancy/cache tweaks measured flat). mma.m16n8k16.bf16 retires that chain
// on the tensor cores: fp8 e4m3 is exactly representable in bf16, so the
// fp8 -> half2 -> f32 -> bf16x2 decode is lossless and the only numerics
// change is the f32 accumulation order inside the mma (fixed by hardware, so
// per-bucket deterministic and replay-stable — but NOT bit-identical to the
// m=1 kernel; buckets 1/2 keep their exact paths, and cross-bucket FP
// divergence is already the accepted whole-step contract since the expert
// GEMM reassociates across buckets anyway).
//
// The kernel reads the ORIGINAL row-major [n, k] fp8 layout — no repack, no
// second weight copy. The mma k-slot -> column map is a free permutation as
// long as A and B slots agree (dot products are permutation-invariant):
// sigma(step s, tid, d) = tid*16 + 4*s + d over a k64 super-chunk makes each
// lane's A bytes one contiguous 16-byte LDG per owned row, with mma step s
// consuming word s of that uint4, and the matching B slots one 8-byte load.
//
// Structure per warp: NTILES independent 16-row mma chains (shared B
// fragments, 2-deep weight prefetch per chain) x KSPLIT k-slices in separate
// blocks writing f32 partials to scratch; a tiny epilogue reduces the slices
// in fixed order (deterministic) and converts to bf16. KSPLIT is the
// occupancy lever the bit-parity kernels could never use.
//
// Measured (jz-38 H200 microbench, batch 8, us): o_proj 66.5 -> 45.8,
// dense_gu 99.4 -> 58.6, dense_dn 50.9 -> 33.0, q_b 26.2 -> 16.9,
// q_a 16.4 -> 10.7, kv_a 13.2 -> 7.7, shared_gu 19.3 -> 13.6,
// shared_dn 11.5 -> 9.3, idx_wq_b 8.8 -> 7.8, idx_wk 12.5 -> 7.5
// (~ -66 us per MoE layer). Batch 4 wins only on the k=6144 column (see
// mma_config); the rest stay on the register tile.
constexpr int kMmaWarps = 4;
// The f32 partial scratch [KSPLIT, BATCH, n] is CALLER-OWNED and passed per
// launch. Ownership matters: the layer forward deliberately overlaps the ctx
// and aux streams (indexer fork, shared-expert fork), and both sides run
// batched GEMVs — a shared per-device buffer would race on the device even
// though the host is single-threaded per rank. Each Rust-side scratch struct
// owns its own buffer, so two streams can never see the same pointer.

// fp8 e4m3 pair -> packed bf16x2, exact (e4m3 c bf16 via the f16/f32 path).
__device__ __forceinline__ unsigned mma_cvt_pair(unsigned char b0, unsigned char b1) {
  __nv_fp8x2_e4m3 p;
  p.__x = (unsigned short)(b0 | (b1 << 8));
  __half2 h = static_cast<__half2>(p);
  float2 f = __half22float2(h);
  __nv_bfloat162 bb = __float22bfloat162_rn(f);
  return *reinterpret_cast<unsigned*>(&bb);
}

template <int BATCH, int KSPLIT, int NTILES>
__global__ __launch_bounds__(kMmaWarps * kWarpSize) void
glm52_gemv_batched_mma_kernel(
    const __nv_bfloat16* __restrict__ activation,  // [BATCH, k]
    const unsigned char* __restrict__ weight,      // [n, k] e4m3, original layout
    const float* __restrict__ weight_scale,        // [n/128, k/128]
    float* __restrict__ partial,                   // [KSPLIT, BATCH, n] f32
    int n, int k) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tile0 = (blockIdx.y * kMmaWarps + warp) * NTILES;  // 16-row tiles
  if (tile0 * 16 >= n) return;
  const int gid = lane >> 2, tid = lane & 3;
  const int kslice = k / KSPLIT;  // multiple of 128
  const int k_begin = blockIdx.x * kslice;
  const int scale_cols = k >> 7;

  float macc[NTILES][4], cacc[NTILES][4];
#pragma unroll
  for (int t = 0; t < NTILES; ++t)
#pragma unroll
    for (int i = 0; i < 4; ++i) { macc[t][i] = 0.f; cacc[t][i] = 0.f; }

  // Per chain: rows (gid, gid+8) of its tile; one 16B packet per row per k64.
  const unsigned char* w0[NTILES];
  const unsigned char* w1[NTILES];
#pragma unroll
  for (int t = 0; t < NTILES; ++t) {
    const int n0 = (tile0 + t) * 16;
    w0[t] = weight + (size_t)(n0 + gid) * k + k_begin + tid * 16;
    w1[t] = weight + (size_t)(n0 + gid + 8) * k + k_begin + tid * 16;
  }

  // 2-deep weight-packet pipeline per chain: 2*NTILES LDGs in flight while the
  // previous chunk's (serializing) asm mma chain retires.
  uint4 wp0[NTILES], wp1[NTILES];
#pragma unroll
  for (int t = 0; t < NTILES; ++t) {
    wp0[t] = __ldcs(reinterpret_cast<const uint4*>(w0[t]));
    wp1[t] = __ldcs(reinterpret_cast<const uint4*>(w1[t]));
  }
  for (int kk = k_begin; kk < k_begin + kslice; kk += 64) {
    uint4 c0[NTILES], c1[NTILES];
#pragma unroll
    for (int t = 0; t < NTILES; ++t) {
      c0[t] = wp0[t]; c1[t] = wp1[t];
      w0[t] += 64; w1[t] += 64;
    }
    if (kk + 64 < k_begin + kslice) {
#pragma unroll
      for (int t = 0; t < NTILES; ++t) {
        wp0[t] = __ldcs(reinterpret_cast<const uint4*>(w0[t]));
        wp1[t] = __ldcs(reinterpret_cast<const uint4*>(w1[t]));
      }
    }
#pragma unroll
    for (int s = 0; s < 4; ++s) {  // four k16 mma steps per k64 super-chunk
      unsigned b01 = 0, b23 = 0;
      if (gid < BATCH) {
        // sigma: B slots (tid*2, +1, +8, +9) = cols tid*16 + 4s + {0,1,2,3}.
        const __nv_bfloat16* xrow =
            activation + (size_t)gid * k + kk + tid * 16 + 4 * s;
        const uint2 bv = *reinterpret_cast<const uint2*>(xrow);
        b01 = bv.x; b23 = bv.y;
      }
#pragma unroll
      for (int t = 0; t < NTILES; ++t) {
        const unsigned char* p0 = reinterpret_cast<const unsigned char*>(&c0[t]) + 4 * s;
        const unsigned char* p1 = reinterpret_cast<const unsigned char*>(&c1[t]) + 4 * s;
        unsigned a0 = mma_cvt_pair(p0[0], p0[1]);  // row gid,   slots tid*2/+1
        unsigned a1 = mma_cvt_pair(p1[0], p1[1]);  // row gid+8, slots tid*2/+1
        unsigned a2 = mma_cvt_pair(p0[2], p0[3]);  // row gid,   slots tid*2+8/+9
        unsigned a3 = mma_cvt_pair(p1[2], p1[3]);  // row gid+8, slots tid*2+8/+9
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(cacc[t][0]), "+f"(cacc[t][1]), "+f"(cacc[t][2]), "+f"(cacc[t][3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b01), "r"(b23));
      }
    }
    if (((kk + 64) & 127) == 0) {  // end of a 128-col scale group
#pragma unroll
      for (int t = 0; t < NTILES; ++t) {
        // A 16-row tile never straddles a /128 scale-row boundary (16 | 128).
        const float scale =
            weight_scale[(size_t)(((tile0 + t) * 16) >> 7) * scale_cols + (kk >> 7)];
#pragma unroll
        for (int i = 0; i < 4; ++i) { macc[t][i] += scale * cacc[t][i]; cacc[t][i] = 0.f; }
      }
    }
  }
  // C fragment: c0=(row gid, col tid*2) c1=(gid, +1) c2=(gid+8, tid*2) c3=(gid+8, +1);
  // cols are batch indices.
  float* out_slice = partial + (size_t)blockIdx.x * BATCH * n;
  const int col0 = tid * 2;
#pragma unroll
  for (int t = 0; t < NTILES; ++t) {
    const int n0 = (tile0 + t) * 16;
    if (col0 < BATCH)     out_slice[(size_t)col0 * n + n0 + gid] = macc[t][0];
    if (col0 + 1 < BATCH) out_slice[(size_t)(col0 + 1) * n + n0 + gid] = macc[t][1];
    if (col0 < BATCH)     out_slice[(size_t)col0 * n + n0 + gid + 8] = macc[t][2];
    if (col0 + 1 < BATCH) out_slice[(size_t)(col0 + 1) * n + n0 + gid + 8] = macc[t][3];
  }
}

// Fixed-order k-slice reduction + bf16 store (deterministic epilogue).
template <int BATCH, int KSPLIT>
__global__ void glm52_gemv_batched_mma_reduce_kernel(
    const float* __restrict__ partial, __nv_bfloat16* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= BATCH * n) return;
  float v = partial[i];
#pragma unroll
  for (int s = 1; s < KSPLIT; ++s) v += partial[(size_t)s * BATCH * n + i];
  out[i] = __float2bfloat16(v);
}

// (ksplit, ntiles) per (arch, batch, whitelisted shape); ksplit == 0 keeps the
// register tile. Measured winners ONLY — Hopper table from an 8xH200
// locked-clock sweep 2026-07-05, Blackwell batch-8 table from a GB300 sm_103
// locked-clock sweep 2026-07-13 (full KSPLIT x NTILES grid per shape; method
// in docs/models/glm52/). Batch-4 has not been re-swept on Blackwell and
// keeps the Hopper picks.
struct MmaConfig { int ksplit; int ntiles; };

bool arch_is_blackwell() {
  // Config selection only — on a failed attribute query fall back to the
  // Hopper table (still correct everywhere) instead of failing the launch.
  static const bool blackwell = [] {
    int device = 0;
    int major = 0;
    if (cudaGetDevice(&device) != cudaSuccess ||
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess) {
      return false;
    }
    return major >= 10;
  }();
  return blackwell;
}

MmaConfig mma_config(int batch, int n, int k) {
  if (batch == 8 && arch_is_blackwell()) {
    if (n == 2048  && k == 6144)  return {48, 2};  // q_a / shared gate,up
    if (n == 2624  && k == 6144)  return {16, 1};  // q_a|kv_a pack (7.3us vs 12.6 separate)
    if (n == 16384 && k == 2048)  return {4, 1};   // q_b
    if (n == 576   && k == 6144)  return {16, 1};  // kv_a
    if (n == 6144  && k == 16384) return {16, 2};  // o_proj
    if (n == 24576 && k == 6144)  return {8, 2};   // dense gate|up
    if (n == 6144  && k == 12288) return {16, 2};  // dense down
    if (n == 4096  && k == 6144)  return {16, 2};  // shared gate|up
    if (n == 6144  && k == 2048)  return {16, 2};  // shared down
    if (n == 4096  && k == 2048)  return {16, 1};  // indexer wq_b
    if (n == 128   && k == 6144)  return {48, 1};  // indexer wk
  }
  if (batch == 8) {
    if (n == 2048  && k == 6144)  return {16, 2};  // q_a / shared gate,up
    if (n == 16384 && k == 2048)  return {8, 4};   // q_b
    if (n == 576   && k == 6144)  return {16, 2};  // kv_a
    if (n == 6144  && k == 16384) return {16, 2};  // o_proj
    if (n == 24576 && k == 6144)  return {16, 2};  // dense gate|up
    if (n == 6144  && k == 12288) return {16, 2};  // dense down
    if (n == 4096  && k == 6144)  return {16, 2};  // shared gate|up
    if (n == 6144  && k == 2048)  return {8, 1};   // shared down
    if (n == 4096  && k == 2048)  return {8, 1};   // indexer wq_b
    if (n == 128   && k == 6144)  return {16, 2};  // indexer wk
  }
  if (batch == 4) {
    // Only the shapes where mma measured ahead of the register tile at batch 4.
    if (n == 2048  && k == 6144) return {16, 2};  // q_a / shared gate,up
    if (n == 576   && k == 6144) return {16, 2};  // kv_a
    if (n == 24576 && k == 6144) return {16, 2};  // dense gate|up
    if (n == 4096  && k == 6144) return {16, 2};  // shared gate|up
    if (n == 128   && k == 6144) return {16, 2};  // indexer wk
  }
  return {0, 0};
}

template <int BATCH, int KSPLIT, int NTILES>
CUresult launch_gemv_batched_mma(const __nv_bfloat16* activation,
                                 const unsigned char* weight,
                                 const float* weight_scale, __nv_bfloat16* out,
                                 float* scratch, size_t scratch_floats, int n,
                                 int k, cudaStream_t stream,
                                 bool skip_reduce = false) {
  if (n % 16 != 0 || k % (128 * KSPLIT) != 0 || (n / 16) % NTILES != 0 ||
      scratch == nullptr || (size_t)KSPLIT * BATCH * n > scratch_floats) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int tiles = n / 16;
  const dim3 grid(KSPLIT, (tiles / NTILES + kMmaWarps - 1) / kMmaWarps, 1);
  glm52_gemv_batched_mma_kernel<BATCH, KSPLIT, NTILES>
      <<<grid, kMmaWarps * kWarpSize, 0, stream>>>(activation, weight,
                                                   weight_scale, scratch, n, k);
  if (!skip_reduce) {
    const int rthreads = 256;
    glm52_gemv_batched_mma_reduce_kernel<BATCH, KSPLIT>
        <<<(BATCH * n + rthreads - 1) / rthreads, rthreads, 0, stream>>>(
            scratch, out, n);
  }
  return consume_last_cuda_error();
}

// Plain SiLU(gate)*up -> bf16 for dense/shared MLPs.
__global__ void glm52_silu_and_mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,    // [rows, 2*inter] (gate|up)
    __nv_bfloat16* __restrict__ output,         // [rows, inter]
    int rows, int inter) {
  const int row = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= rows || col >= inter) return;
  const __nv_bfloat16* gate = input + (size_t)row * (2 * inter);
  const __nv_bfloat16* up = gate + inter;
  const float g = __bfloat162float(gate[col]);
  const float u = __bfloat162float(up[col]);
  const float sg = 1.0f / (1.0f + expf(-g));
  output[(size_t)row * inter + col] = __float2bfloat16(g * sg * u);
}

// Fixed-order k-slice reduce fused with the SwiGLU: `partial` is the packed
// gate|up mma scratch ([ksplit, rows, 2*inter] f32). Each of the two sums is
// rounded to bf16 before the SiLU math — bit-identical to the standalone
// reduce -> silu pair this replaces, one launch instead of two. ksplit is a
// runtime loop bound: the kernel is bandwidth-trivial and shared across every
// per-shape split factor.
__global__ void glm52_gemv_reduce_silu_mul_kernel(
    const float* __restrict__ partial,       // [ksplit, rows, 2*inter]
    __nv_bfloat16* __restrict__ output,      // [rows, inter]
    int rows, int inter, int ksplit) {
  const int row = blockIdx.x;
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (row >= rows || col >= inter) return;
  const size_t slice = (size_t)rows * 2 * inter;
  const size_t base = (size_t)row * (2 * inter);
  float g = partial[base + col];
  float u = partial[base + inter + col];
  for (int s = 1; s < ksplit; ++s) {
    g += partial[(size_t)s * slice + base + col];
    u += partial[(size_t)s * slice + base + inter + col];
  }
  g = __bfloat162float(__float2bfloat16(g));
  u = __bfloat162float(__float2bfloat16(u));
  const float sg = 1.0f / (1.0f + expf(-g));
  output[(size_t)row * inter + col] = __float2bfloat16(g * sg * u);
}

// Fixed-order k-slice reduce for a horizontally packed projection pair:
// partials are [ksplit, rows, n_a + n_b] (one mma launch over the packed
// weight); the reduce de-interleaves into the two compact per-projection
// outputs so every downstream consumer keeps its contiguous [rows, n] layout.
// Same summation order and rounding as the plain reduce kernel.
__global__ void glm52_gemv_split_reduce_kernel(
    const float* __restrict__ partial,   // [ksplit, rows, n_a + n_b]
    __nv_bfloat16* __restrict__ out_a,   // [rows, n_a]
    __nv_bfloat16* __restrict__ out_b,   // [rows, n_b]
    int rows, int n_a, int n_b, int ksplit) {
  const int n = n_a + n_b;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= rows * n) return;
  const int row = i / n;
  const int col = i % n;
  float v = partial[i];
  for (int s = 1; s < ksplit; ++s) v += partial[(size_t)s * rows * n + i];
  const __nv_bfloat16 h = __float2bfloat16(v);
  if (col < n_a) {
    out_a[(size_t)row * n_a + col] = h;
  } else {
    out_b[(size_t)row * n_b + (col - n_a)] = h;
  }
}

// Kernel-coverage + scale-index invariants. k%128 keeps the scale column stride
// exact; n%rpb and k%512 keep the row-tile / lane sweep exact (no tail masking). n
// need NOT be %128: a warp owns ROWS consecutive rows from an rpb-row block based at a
// multiple of rpb (ROWS | 128), so its rows never straddle a /128 scale boundary -- the
// caller just sizes the scale buffer with div_ceil(n,128) rows (MLA kv_a n=576 -> 5
// rows). Encoding these turns any future off-shape into a crash, not a silent read.
bool valid_tiling(int n, int k, int rpb) {
  return n > 0 && k > 0 && k % kFp8Block == 0 && n % rpb == 0 && k % kStep == 0;
}

// Rows/warp follows the executing architecture. The original table was tuned
// on H200; an exact production-kernel sweep on GB300 found ROWS=1 ahead for
// 4096x2048 (3.70us vs 4.47us for ROWS=4).
CUresult plain_rows_per_warp(int k, int* rows) {
  // Attribute query only — this runs per projection launch during graph
  // capture, and cudaGetDeviceProperties is orders of magnitude slower than
  // the single cudaDevAttrComputeCapabilityMajor lookup needed here.
  int device = 0;
  int major = 0;
  cudaError_t err = cudaGetDevice(&device);
  if (err == cudaSuccess) {
    err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  }
  if (err != cudaSuccess) return map_cuda_error(err);
  *rows = major >= 10 ? kRowsPlainBlackwell
                      : (k <= 2048 ? kRowsPlainShortKHopper : kRowsPlainLongK);
  return CUDA_SUCCESS;
}

// The batched kernel's supported batches (one instantiation per decode
// bucket beyond 1). Rust's GLM52_DECODE_BUCKETS must match — the launcher
// rejects any other batch, so a drift crashes at the boundary.
constexpr int kBatchedGemvBatch2 = 2;
constexpr int kBatchedGemvBatch4 = 4;
constexpr int kBatchedGemvBatchFull = 8;
// Verify-span buckets (#812): correct-first register-tile instantiations at
// ROWS=1 (the per-thread activation preload dominates registers at these
// batches and spills regardless of ROWS). No measured mma table yet —
// profile before trusting their latency; the expected endgame for m >= 16
// is the fp8 GEMM path, not a wider GEMV.
constexpr int kBatchedGemvBatch16 = 16;
constexpr int kBatchedGemvBatch32 = 32;
constexpr int kBatchedGemvBatch48 = 48;
// Per-batch (ROWS, WARPS) tile — measured, see the kernel comment.
constexpr int kBatchedRows  = 4;
constexpr int kBatchedWarps = 4;

// Whitelist of the model's linear shapes (shared by the plain and batched
// launchers; the tiling check differs — rpb is 8 or 32 for plain, 8 (batch 2)
// or 16 (batch 4/8) for batched — so it is applied by each launcher, not here).
bool whitelisted_linear_shape(int n, int k) {
  if (n == 2048  && k == 6144)  return true;  // q_a / shared gate,up
  if (n == 2624  && k == 6144)  return true;  // q_a|kv_a horizontal pack
  if (n == 16384 && k == 2048)  return true;  // q_b
  if (n == 4096  && k == 2048)  return true;  // q_b attention-TP 16-head shard
  if (n == 2048  && k == 2048)  return true;  // q_b attention-TP 8-head shard
  if (n == 576   && k == 6144)  return true;  // kv_a + rope (partial-N: 576%128!=0)
  if (n == 6144  && k == 16384) return true;  // o_proj
  if (n == 6144  && k == 4096)  return true;  // o_proj attention-TP 16-head shard
  if (n == 24576 && k == 6144)  return true;  // dense gate|up (packed)
  if (n == 6144  && k == 12288) return true;  // dense down
  if (n == 4096  && k == 6144)  return true;  // shared gate|up (packed)
  if (n == 6144  && k == 2048)  return true;  // shared down
  if (n == 4096  && k == 2048)  return true;  // indexer wq_b
  if (n == 128   && k == 6144)  return true;  // indexer wk
  return false;
}

// Encoding these turns any future off-shape into a crash, not a silent read.
bool valid_linear_shape(int n, int k, int rows) {
  return valid_tiling(n, k, kWarpsPerBlk * rows) &&
         whitelisted_linear_shape(n, k);
}

}  // namespace

extern "C" {

static CUresult fp8_weight_only_gemv(
    const __nv_bfloat16* activation, const unsigned char* weight,
    const float* weight_scale, __nv_bfloat16* out, int n, int k,
    cudaStream_t stream) {
  int rows = 0;
  CUresult rows_status = plain_rows_per_warp(k, &rows);
  if (rows_status != CUDA_SUCCESS) return rows_status;
  if (activation == nullptr || weight == nullptr || weight_scale == nullptr ||
      out == nullptr || !aligned16(activation) ||
      !valid_linear_shape(n, k, rows)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const dim3 grid(1, n / (kWarpsPerBlk * rows), 1);
  if (rows == kRowsPlainShortKHopper) {
    glm52_fp8_weight_only_gemv_kernel<kRowsPlainShortKHopper>
        <<<grid, kBlockThreads, 0, stream>>>(activation, weight, weight_scale, out, n, k);
  } else {
    glm52_fp8_weight_only_gemv_kernel<kRowsPlainLongK>
        <<<grid, kBlockThreads, 0, stream>>>(activation, weight, weight_scale, out, n, k);
  }
  return consume_last_cuda_error();
}

CUresult glm52_fp8_weight_only_gemv_pair_cuda(
    const __nv_bfloat16* activation, const unsigned char* weight_a,
    const float* weight_scale_a, __nv_bfloat16* out_a, int n_a,
    const unsigned char* weight_b, const float* weight_scale_b,
    __nv_bfloat16* out_b, int n_b, int k, cudaStream_t stream) {
  // This ABI is deliberately the one measured MLA fusion, not a generic
  // paired-GEMV surface whose unmeasured shapes could silently spread.
  if (activation == nullptr || weight_a == nullptr || weight_scale_a == nullptr ||
      out_a == nullptr || weight_b == nullptr || weight_scale_b == nullptr ||
      out_b == nullptr || !aligned16(activation) || n_a != 2048 || n_b != 576 ||
      k != 6144 || !valid_tiling(n_a, k, kWarpsPerBlk) ||
      !valid_tiling(n_b, k, kWarpsPerBlk)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const dim3 grid(1, (n_a + n_b) / kWarpsPerBlk, 1);
  glm52_fp8_weight_only_gemv_pair_kernel<<<grid, kBlockThreads, 0, stream>>>(
      activation, weight_a, weight_scale_a, out_a, n_a, weight_b,
      weight_scale_b, out_b, n_b, k);
  return consume_last_cuda_error();
}

// Batched plain GEMV: `out[b] = deq(weight) @ activation[b]` for b in [0, batch).
// batch == 1 routes to the m=1 kernel; batch 2 keeps the bit-parity register
// tile; batches 4/8 dispatch per shape between the tensor-core mma path and
// the register tile (mma_config — deterministic per bucket, not bit-identical
// to m=1; see the mma block comment for the numerics contract).
// `scratch`/`scratch_floats` = caller-owned f32 partial buffer for the
// tensor-core path (see the mma block comment for the ownership contract:
// one buffer per stream, never shared across the ctx/aux overlap). Batches
// that stay on the register tile ignore it; an mma-routed launch with a
// null/short buffer fails INVALID_VALUE instead of racing.
//
// `partials_only`: an mma-routed launch stops after the mma kernel, leaves the
// f32 k-slice partials in `scratch`, and reports the ksplit via `ksplit_out`
// so a fused consumer (reduce+SwiGLU below) can take over the fixed-order
// reduction. Non-mma paths write bf16 `out` as usual and report ksplit 0.
static CUresult gemv_batched_dispatch(
    const __nv_bfloat16* activation, const unsigned char* weight,
    const float* weight_scale, __nv_bfloat16* out, float* scratch,
    size_t scratch_floats, int batch, int n, int k, cudaStream_t stream,
    bool partials_only, int* ksplit_out) {
  if (activation == nullptr || weight == nullptr || weight_scale == nullptr ||
      out == nullptr || !aligned16(activation) ||
      !whitelisted_linear_shape(n, k)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (ksplit_out != nullptr) *ksplit_out = 0;
  if (batch == 1) {
    return fp8_weight_only_gemv(activation, weight, weight_scale, out, n, k,
                                stream);
  }
#define GLM52_MMA_CASE(BATCH_, KS_, NT_)                                     \
  if (cfg.ksplit == (KS_) && cfg.ntiles == (NT_)) {                         \
    if (partials_only && ksplit_out != nullptr) *ksplit_out = (KS_);        \
    return launch_gemv_batched_mma<BATCH_, KS_, NT_>(                       \
        activation, weight, weight_scale, out, scratch, scratch_floats, n,  \
        k, stream, partials_only);                                          \
  }
  switch (batch) {
    case kBatchedGemvBatch2: {
      // Batch 2 keeps the 1-row/warp shape (see the kernel comment).
      if (!valid_tiling(n, k, kWarpsPerBlk)) return CUDA_ERROR_INVALID_VALUE;
      const dim3 grid(1, n / kWarpsPerBlk, 1);
      glm52_fp8_weight_only_gemv_batched_kernel<kBatchedGemvBatch2, 1, kWarpsPerBlk>
          <<<grid, kBlockThreads, 0, stream>>>(activation, weight, weight_scale,
                                               out, n, k);
      break;
    }
    case kBatchedGemvBatch4: {
      const MmaConfig cfg = mma_config(batch, n, k);
      GLM52_MMA_CASE(kBatchedGemvBatch4, 16, 2)
      // A table entry whose (ksplit, ntiles) has no case above is a config/
      // dispatch drift — crash at capture, never silently take the register
      // tile (the sweep winner would quietly not run).
      if (cfg.ksplit != 0) return CUDA_ERROR_INVALID_VALUE;
      if (!valid_tiling(n, k, kBatchedWarps * kBatchedRows))
        return CUDA_ERROR_INVALID_VALUE;
      const dim3 grid(1, n / (kBatchedWarps * kBatchedRows), 1);
      glm52_fp8_weight_only_gemv_batched_kernel<kBatchedGemvBatch4, kBatchedRows,
                                                kBatchedWarps>
          <<<grid, kBatchedWarps * kWarpSize, 0, stream>>>(
              activation, weight, weight_scale, out, n, k);
      break;
    }
    case kBatchedGemvBatchFull: {
      // Table shapes route to the tensor-core path; anything off-table (e.g.
      // the attention-TP shard shapes) keeps the register tile below.
      const MmaConfig cfg = mma_config(batch, n, k);
      GLM52_MMA_CASE(kBatchedGemvBatchFull, 16, 2)
      GLM52_MMA_CASE(kBatchedGemvBatchFull, 8, 4)
      GLM52_MMA_CASE(kBatchedGemvBatchFull, 8, 1)
      GLM52_MMA_CASE(kBatchedGemvBatchFull, 48, 2)
      GLM52_MMA_CASE(kBatchedGemvBatchFull, 48, 1)
      GLM52_MMA_CASE(kBatchedGemvBatchFull, 16, 1)
      GLM52_MMA_CASE(kBatchedGemvBatchFull, 8, 2)
      GLM52_MMA_CASE(kBatchedGemvBatchFull, 4, 1)
      // See the batch-4 arm: an uninstantiated table entry must crash here.
      if (cfg.ksplit != 0) return CUDA_ERROR_INVALID_VALUE;
      if (!valid_tiling(n, k, kBatchedWarps * kBatchedRows))
        return CUDA_ERROR_INVALID_VALUE;
      const dim3 grid(1, n / (kBatchedWarps * kBatchedRows), 1);
      glm52_fp8_weight_only_gemv_batched_kernel<kBatchedGemvBatchFull, kBatchedRows,
                                                kBatchedWarps>
          <<<grid, kBatchedWarps * kWarpSize, 0, stream>>>(
              activation, weight, weight_scale, out, n, k);
      break;
    }
    case kBatchedGemvBatch16: {
      if (!valid_tiling(n, k, kWarpsPerBlk)) return CUDA_ERROR_INVALID_VALUE;
      const dim3 grid(1, n / kWarpsPerBlk, 1);
      glm52_fp8_weight_only_gemv_batched_kernel<kBatchedGemvBatch16, 1, kWarpsPerBlk>
          <<<grid, kBlockThreads, 0, stream>>>(activation, weight, weight_scale,
                                               out, n, k);
      break;
    }
    case kBatchedGemvBatch32: {
      if (!valid_tiling(n, k, kWarpsPerBlk)) return CUDA_ERROR_INVALID_VALUE;
      const dim3 grid(1, n / kWarpsPerBlk, 1);
      glm52_fp8_weight_only_gemv_batched_kernel<kBatchedGemvBatch32, 1, kWarpsPerBlk>
          <<<grid, kBlockThreads, 0, stream>>>(activation, weight, weight_scale,
                                               out, n, k);
      break;
    }
    case kBatchedGemvBatch48: {
      if (!valid_tiling(n, k, kWarpsPerBlk)) return CUDA_ERROR_INVALID_VALUE;
      const dim3 grid(1, n / kWarpsPerBlk, 1);
      glm52_fp8_weight_only_gemv_batched_kernel<kBatchedGemvBatch48, 1, kWarpsPerBlk>
          <<<grid, kBlockThreads, 0, stream>>>(activation, weight, weight_scale,
                                               out, n, k);
      break;
    }
    default:
      return CUDA_ERROR_INVALID_VALUE;
  }
  return consume_last_cuda_error();
#undef GLM52_MMA_CASE
}

CUresult glm52_fp8_weight_only_gemv_batched_cuda(
    const __nv_bfloat16* activation, const unsigned char* weight,
    const float* weight_scale, __nv_bfloat16* out, float* scratch,
    size_t scratch_floats, int batch, int n, int k, cudaStream_t stream) {
  return gemv_batched_dispatch(activation, weight, weight_scale, out, scratch,
                               scratch_floats, batch, n, k, stream,
                               /*partials_only=*/false, nullptr);
}

// Partials-producing twin of the batched GEMV: mma-routed shapes stop at the
// f32 k-slice partials (`*ksplit_out` > 0, bf16 `out` untouched); everything
// else behaves exactly like the plain entry and reports `*ksplit_out` == 0.
CUresult glm52_fp8_weight_only_gemv_partials_cuda(
    const __nv_bfloat16* activation, const unsigned char* weight,
    const float* weight_scale, __nv_bfloat16* out, float* scratch,
    size_t scratch_floats, int batch, int n, int k, cudaStream_t stream,
    int* ksplit_out) {
  if (ksplit_out == nullptr) return CUDA_ERROR_INVALID_VALUE;
  return gemv_batched_dispatch(activation, weight, weight_scale, out, scratch,
                               scratch_floats, batch, n, k, stream,
                               /*partials_only=*/true, ksplit_out);
}

CUresult glm52_silu_and_mul_bf16_cuda(
    const __nv_bfloat16* input, __nv_bfloat16* output, int rows, int inter,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || rows <= 0 || inter <= 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int threads = 256;
  const dim3 grid(rows, (inter + threads - 1) / threads, 1);
  glm52_silu_and_mul_bf16_kernel<<<grid, threads, 0, stream>>>(input, output, rows,
                                                               inter);
  return consume_last_cuda_error();
}

CUresult glm52_gemv_reduce_silu_mul_cuda(
    const float* partial, __nv_bfloat16* output, int rows, int inter,
    int ksplit, cudaStream_t stream) {
  if (partial == nullptr || output == nullptr || rows <= 0 || inter <= 0 ||
      ksplit < 1) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int threads = 256;
  const dim3 grid(rows, (inter + threads - 1) / threads, 1);
  glm52_gemv_reduce_silu_mul_kernel<<<grid, threads, 0, stream>>>(
      partial, output, rows, inter, ksplit);
  return consume_last_cuda_error();
}

CUresult glm52_gemv_split_reduce_cuda(
    const float* partial, __nv_bfloat16* out_a, __nv_bfloat16* out_b, int rows,
    int n_a, int n_b, int ksplit, cudaStream_t stream) {
  if (partial == nullptr || out_a == nullptr || out_b == nullptr || rows <= 0 ||
      n_a <= 0 || n_b <= 0 || ksplit < 1) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  const int threads = 256;
  const int total = rows * (n_a + n_b);
  glm52_gemv_split_reduce_kernel<<<(total + threads - 1) / threads, threads, 0,
                                   stream>>>(partial, out_a, out_b, rows, n_a,
                                             n_b, ksplit);
  return consume_last_cuda_error();
}

// Capability query for the horizontal-pack route: reports the ksplit the mma
// table would pick for (batch, n, k), 0 when the shape would take the
// register tile. The rust side must NOT run a packed batched launch without a
// table entry — the register tile would write the packed interleaved layout
// no consumer understands.
CUresult glm52_gemv_mma_ksplit_cuda(int batch, int n, int k, int* ksplit_out) {
  if (ksplit_out == nullptr) return CUDA_ERROR_INVALID_VALUE;
  *ksplit_out =
      (batch == kBatchedGemvBatch4 || batch == kBatchedGemvBatchFull)
          ? mma_config(batch, n, k).ksplit
          : 0;
  return CUDA_SUCCESS;
}

}  // extern "C"
