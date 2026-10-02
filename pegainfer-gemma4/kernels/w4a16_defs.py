"""TileLang W4A16 decode GEMMs for the compressed-tensors QAT checkpoints.

y[m, n] = sum_k x[m, k] * (q[n, k] - 8) * s[n, k // 32] for m up to 16 rows.

The loader rewrites each linear once (`gemma4_w4a16_pack_cuda`) so that, for
every 16 x 16 (n, k) tile, each lane's eight m16n8k16 A-fragment values sit in
one int32 -- lane (g, t) = (lane / 4, lane % 4) holds rows g and g + 8,
columns 2t, 2t + 1, 2t + 8, 2t + 9 -- with the nibble slots holding fragment
locals [0, 2, 4, 6, 1, 3, 5, 7]. Four consecutive k16 tiles are one 16-byte
run per lane. Each lane's two row scales travel as one bf16x2 word.

Both kernels are stream-K over (64-column tile, 256-deep K stage) units on P
persistent CTAs whose warps split each stage's K, so x is read from shared
memory once per CTA. A tile split across CTAs is finished in-kernel by
the CTA holding its first units: the later parts are the other CTAs' first
tiles, written early, and are added in k order once their flags are set. The
split depends on the shape and P alone, so a row's bits do not depend on how
many rows ride with it, and every CTA must be resident at once. That residency
is what ties the warp count and pipeline depth to the target's shared memory
per SM; `pick_tiling` owns the choice.

Up to eight rows run with the operands swapped (weights as the MMA A operand,
x as the n8 B operand, one MMA per weight tile); sixteen rows take x as the
m16 A operand through ldmatrix and the weights as B.

For gate|up the loader interleaves the two halves 32 rows at a time, so a
64-column tile holds 32 gate columns and their up columns, and the kernel
finishing a tile writes gelu(gate) * up with the MLP activation kernel's
arithmetic (`gelu_tanh_mul_kernel`), rounding gate and up to bf16 first as
their stored values would be.
"""

import tilelang.language as T
from tilelang.cuda.intrinsics.layout.mma_layout import make_mma_swizzle_layout
from tilelang.cuda.intrinsics.layout.mma_layout import (
    mma_store_32x8_to_shared_16x16_layout,
)
from tilelang.cuda.intrinsics.macro.mma_macro_generator import TensorCoreIntrinEmitter

GROUP = 32
BLOCK_N = 64
BLOCK_K = 256
BUCKETS = (1, 2, 4, 8, 16)

# The stream-K fix-up keeps two CTAs resident per SM, so one CTA's dynamic
# shared memory has to fit half the device's per-SM budget. More warps split a
# stage's K wider and more stages hide the load latency deeper, so the
# candidates are ordered most capable first and `pick_tiling` takes the first
# that fits: (4, 3) is the Hopper default, (2, 2) is what Ada's 100 KB admits.
TILINGS = ((4, 3), (4, 2), (2, 3), (2, 2))
# The driver reserves this much shared memory per block on top of the dynamic
# size the kernel asks for.
SMEM_PER_BLOCK_RESERVED = 1024
# The widest row bucket: `rows = 16` runs unswapped with `xr = 16`, so it is
# the one the budget has to satisfy.
SMEM_XR = 16


def smem_bytes(warps, stages):
    """Dynamic shared bytes one CTA asks for at the widest row bucket: the
    `stages`-deep pipeline over the A, weight and scale tiles, the per-warp
    accumulator tile and the cross-warp reduction tile."""
    cols = BLOCK_N // 16
    slices = BLOCK_K // 64
    staged = (
        SMEM_XR * BLOCK_K * 2  # x_sh
        + cols * slices * 32 * 4 * 4  # w_sh
        + cols * (BLOCK_K // GROUP) * 8 * 4  # s_sh
    )
    return stages * staged + warps * SMEM_XR * BLOCK_N * 4 + SMEM_XR * BLOCK_N * 4


def pick_tiling(smem_per_sm):
    """The most capable tiling whose two resident CTAs fit `smem_per_sm`."""
    for warps, stages in TILINGS:
        if 2 * (smem_bytes(warps, stages) + SMEM_PER_BLOCK_RESERVED) <= smem_per_sm:
            return warps, stages
    raise ValueError(
        f"no W4A16 tiling fits {smem_per_sm} B of shared memory per SM: the "
        f"cheapest needs "
        f"{2 * (smem_bytes(*TILINGS[-1]) + SMEM_PER_BLOCK_RESERVED)} B for its "
        f"two resident CTAs"
    )


# One prelude for every kernel in the unit, so their preambles agree.
PRELUDE = r"""
#include <cuda_bf16.h>
// (lane - 136) * scale over the four nibble pairs of one packed word, the
// pair's row alternating g, g + 8. `order` places pair k at out[order[k]].
__device__ __forceinline__ void w4a16_decode8(const int* src, const int* scales, void* dst,
                                              int o0, int o1, int o2, int o3) {
  const unsigned w = *reinterpret_cast<const unsigned*>(src);
  const unsigned sw = *reinterpret_cast<const unsigned*>(scales);
  const __nv_bfloat162 sp = *reinterpret_cast<const __nv_bfloat162*>(&sw);
  const __nv_bfloat162 s_lo = __halves2bfloat162(__low2bfloat16(sp), __low2bfloat16(sp));
  const __nv_bfloat162 s_hi = __halves2bfloat162(__high2bfloat16(sp), __high2bfloat16(sp));
  const __nv_bfloat162 bias = __float2bfloat162_rn(136.0f);
  unsigned* out = reinterpret_cast<unsigned*>(dst);
  const int order[4] = {o0, o1, o2, o3};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    unsigned v = ((w >> (4 * k)) & 0x000F000Fu) | 0x43004300u;
    __nv_bfloat162 b = __hsub2(*reinterpret_cast<__nv_bfloat162*>(&v), bias);
    b = __hmul2(b, (k & 1) ? s_hi : s_lo);
    out[order[k]] = *reinterpret_cast<unsigned*>(&b);
  }
}
__device__ __forceinline__ void w4a16_flag_set(int* f) {
  __threadfence();
  asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(f), "r"(1) : "memory");
}
__device__ __forceinline__ void w4a16_flag_wait(int* f) {
  int v;
  do {
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(f) : "memory");
  } while (v == 0);
}
__device__ __forceinline__ void w4a16_flag_clear(int* f) { *f = 0; }
// gelu_pytorch_tanh(gate) * up as gelu_tanh_mul_kernel computes it from the
// bf16 gate and up; the caller rounds the result to bf16.
__device__ __forceinline__ float w4a16_gelu_mul(float gate, float up) {
  const float kSqrt2OverPi = 0.7978845608028654f;
  const float g = __bfloat162float(__float2bfloat16(gate));
  const float u = __bfloat162float(__float2bfloat16(up));
  float inner = kSqrt2OverPi * (g + 0.044715f * g * g * g);
  float gelu_g = 0.5f * g * (1.0f + tanhf(inner));
  return __bfloat162float(__float2bfloat16(gelu_g)) * u;
}
"""

# A-fragment pairs in place; for the B fragment the middle two trade places.
A_ORDER = (0, 1, 2, 3)
B_ORDER = (0, 2, 1, 3)


def plan(nb, kb, P):
    """Per CTA, the CTAs whose parts of the split tile it finishes, in k order."""
    U = nb * kb
    lo = [c * U // P for c in range(P)]
    hi = [(c + 1) * U // P for c in range(P)]
    offsets, flat = [0], []
    for c in range(P):
        if lo[c] < hi[c]:
            t = (hi[c] - 1) // kb
            if t * kb >= lo[c] and (t + 1) * kb > hi[c]:
                c2 = c + 1
                while c2 < P and lo[c2] < (t + 1) * kb:
                    if lo[c2] < hi[c2]:
                        flat.append(c2)
                    c2 += 1
        offsets.append(len(flat))
    return offsets, flat


def gemm(N, K, rows, P, gelu_mul=False, tiling=TILINGS[0]):
    """The prim_func for one (shape, bucket); `plan(N // BLOCK_N, K // BLOCK_K, P)`
    gives its `fin_off` / `fin_list` arguments. With `gelu_mul` the weight is
    an interleaved gate|up stack and `y` is `N // 2` wide. `tiling` is a
    `(warps, stages)` pair, chosen against the target's shared memory budget
    (`pick_tiling`)."""
    warps, stages = tiling
    assert rows in BUCKETS and N % BLOCK_N == 0 and K % BLOCK_K == 0
    swapped = rows <= 8
    xr = 8 if swapped else 16
    cols = BLOCK_N // 16
    slices = BLOCK_K // 64
    assert 1 <= warps <= slices, f"{warps} warps cannot split {slices} K slices"
    per_warp = slices // warps
    nb, kb = N // BLOCK_N, K // BLOCK_K
    U = nb * kb
    steps = -(-U // P)
    n_flat = max(len(plan(nb, kb, P)[1]), 1)
    emitter = TensorCoreIntrinEmitter(
        a_dtype="bfloat16",
        b_dtype="bfloat16",
        accum_dtype="float32",
        a_transposed=False,
        b_transposed=True,
        block_row_warps=1,
        block_col_warps=1,
        warp_row_tiles=16,
        warp_col_tiles=BLOCK_N,
        chunk=BLOCK_K,
    )
    lo_c = emitter.local_size_out
    acc_len = cols * 4 if swapped else cols * lo_c
    order = A_ORDER if swapped else B_ORDER
    half = BLOCK_N // 2

    @T.macro
    def store(y, tot, t):
        if gelu_mul:
            for i, j in T.Parallel(rows, half):
                y[i, t * half + j] = T.cast(
                    T.call_extern(
                        "float", "w4a16_gelu_mul", tot[i, j], tot[i, j + half]
                    ),
                    "bfloat16",
                )
        else:
            for i, j in T.Parallel(rows, BLOCK_N):
                y[i, t * BLOCK_N + j] = T.cast(tot[i, j], "bfloat16")

    @T.prim_func
    def main(
        x: T.Tensor((rows, K), "bfloat16"),
        wq: T.Tensor((N // 16, K // 64, 32, 4), "int32"),
        sq: T.Tensor((N // 16, K // GROUP, 8), "int32"),
        y: T.Tensor((rows, N // 2 if gelu_mul else N), "bfloat16"),
        part: T.Tensor((P, rows, BLOCK_N), "float"),
        flags: T.Tensor((P,), "int32"),
        fin_off: T.Tensor((P + 1,), "int32"),
        fin_list: T.Tensor((n_flat,), "int32"),
    ):
        with T.Kernel(P, threads=32 * warps, prelude=PRELUDE) as c:
            x_sh = T.alloc_shared((xr, BLOCK_K), "bfloat16")
            w_sh = T.alloc_shared((cols, slices, 32, 4), "int32")
            s_sh = T.alloc_shared((cols, BLOCK_K // GROUP, 8), "int32")
            c_sh = T.alloc_shared((warps, xr, BLOCK_N), "float")
            tot = T.alloc_shared((xr, BLOCK_N), "float")
            if not swapped:
                T.annotate_layout({x_sh: make_mma_swizzle_layout(x_sh)})
            w_loc = T.alloc_local((cols * 8,), "bfloat16")
            x_loc = T.alloc_local((8,), "bfloat16")
            b_q = T.alloc_local((cols * 4,), "int32")
            s_q = T.alloc_local((cols * 2,), "int32")
            acc = T.alloc_local((acc_len,), "float")
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx % 32
            g = lane // 4
            t4 = lane % 4
            lo_u = c * U // P
            hi_u = (c + 1) * U // P
            T.clear(acc)
            for it in T.Pipelined(steps, num_stages=stages):
                u = T.min(lo_u + it, hi_u - 1)
                t = u // kb
                kt = u % kb
                k0 = kt * BLOCK_K
                T.copy(x[0:rows, k0 : k0 + BLOCK_K], x_sh[0:rows, :], disable_tma=True)
                T.copy(
                    wq[t * cols : (t + 1) * cols, k0 // 64 : k0 // 64 + slices, :, :],
                    w_sh,
                    disable_tma=True,
                )
                T.copy(
                    sq[
                        t * cols : (t + 1) * cols,
                        k0 // GROUP : k0 // GROUP + BLOCK_K // GROUP,
                        :,
                    ],
                    s_sh,
                    disable_tma=True,
                )
                if lo_u + it < hi_u:
                    for p in T.serial(per_warp):
                        kq = warp * per_warp + p
                        for j in T.serial(cols):
                            for v in T.vectorized(4):
                                b_q[j * 4 + v] = w_sh[j, kq, lane, v]
                            for v in T.serial(2):
                                s_q[j * 2 + v] = s_sh[j, kq * 2 + v, g]
                        for ki in T.serial(4):
                            if swapped:
                                kk = kq * 64 + ki * 16
                                for v in T.vectorized(2):
                                    x_loc[v] = x_sh[g, kk + 2 * t4 + v]
                                for v in T.vectorized(2):
                                    x_loc[2 + v] = x_sh[g, kk + 8 + 2 * t4 + v]
                            else:
                                emitter.ldmatrix_a(x_loc, x_sh, kq * 4 + ki)
                            for j in T.serial(cols):
                                T.call_extern(
                                    "handle",
                                    "w4a16_decode8",
                                    T.address_of(b_q[j * 4 + ki]),
                                    T.address_of(s_q[j * 2 + ki // 2]),
                                    T.address_of(w_loc[j * 8]),
                                    *order,
                                )
                                if swapped:
                                    T.ptx_mma(
                                        "float32",
                                        "m16n8k16",
                                        "row",
                                        "col",
                                        "bf16",
                                        "bf16",
                                        "fp32",
                                        w_loc.data,
                                        j * 8,
                                        x_loc.data,
                                        0,
                                        acc.data,
                                        j * 4,
                                        T.bool(False),
                                    )
                            if not swapped:
                                emitter.mma(x_loc, w_loc, acc)
                    if kt == kb - 1 or u == hi_u - 1:
                        if swapped:
                            for j in T.serial(cols):
                                for local in T.serial(4):
                                    c_sh[
                                        warp,
                                        2 * t4 + local % 2,
                                        j * 16 + g + 8 * (local // 2),
                                    ] = acc[j * 4 + local]
                        else:
                            for j in T.serial(cols):
                                for local in T.serial(lo_c):
                                    r, cc = T.meta_var(
                                        mma_store_32x8_to_shared_16x16_layout(
                                            lane, local
                                        )
                                    )
                                    c_sh[warp, r, j * 16 + cc] = acc[j * lo_c + local]
                        T.sync_threads()
                        for i, j in T.Parallel(rows, BLOCK_N):
                            total = T.alloc_var("float")
                            total = c_sh[0, i, j]
                            for w in T.serial(1, warps):
                                total = total + c_sh[w, i, j]
                            tot[i, j] = total
                        T.sync_threads()
                        if t * kb < lo_u:
                            for i, j in T.Parallel(rows, BLOCK_N):
                                part[c, i, j] = tot[i, j]
                            T.sync_threads()
                            if tx == 0:
                                T.call_extern(
                                    "handle", "w4a16_flag_set", T.address_of(flags[c])
                                )
                        elif kt == kb - 1:
                            store(y, tot, t)
                        else:
                            for r in T.serial(fin_off[c], fin_off[c + 1]):
                                if tx == 0:
                                    T.call_extern(
                                        "handle",
                                        "w4a16_flag_wait",
                                        T.address_of(flags[fin_list[r]]),
                                    )
                                T.sync_threads()
                                for i, j in T.Parallel(rows, BLOCK_N):
                                    tot[i, j] = tot[i, j] + part[fin_list[r], i, j]
                                T.sync_threads()
                                if tx == 0:
                                    T.call_extern(
                                        "handle",
                                        "w4a16_flag_clear",
                                        T.address_of(flags[fin_list[r]]),
                                    )
                            store(y, tot, t)
                        T.sync_threads()
                        T.clear(acc)

    return main
