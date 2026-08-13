"""Vendored TileLang definitions for the K3 batched decode kernels.

Trimmed copy of the certified K3 TileLang kernel set: only the three batched
kernels this crate AOT-compiles (`router_topk_batched`,
`attnres_scores_batched`, `attnres_mix_batched`) plus the module-level
helpers they need (`DT`/`ACC`/`NEG`, the cache switch, `_compile`).

The prim_func bodies are byte-identical to the upstream Python source and
must stay that way. They were certified row-by-row against the reference
implementation, so any respelling — even a semantically equivalent one —
invalidates the certification and has to be re-gated upstream first. Edit
upstream, then re-vendor; never patch here.

Every kernel is row-independent: batch size B (and expert count E / attention
residual block count NB) are STATIC compile dimensions, one instantiation per
bucket. `generate.py` walks the bucket lists and dumps
`get_kernel_source()` for each.
"""
from functools import lru_cache

import tilelang
import tilelang.language as T

DT = "bfloat16"
ACC = "float32"
NEG = -1.0e30

tilelang.disable_cache()


@lru_cache(maxsize=None)
def _compile(prim):
    return tilelang.compile(prim)


# --------------------------------------------------------------------------- #
# batched variants (for EP / high-concurrency serving; every row is independent
# and the per-row spelling is word-for-word identical to the bs=1 version, so
# they are gated **bitwise** against the bs=1 certified kernels, see
# fast/check_batched.py.
# B is a static compile-time dimension: serving goes through lru_cache per
# batch bucket, one compile per bucket.
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def router_topk_batched(E: int, TOPK: int, B: int, threads: int = 256):
    """Batched version of ``router_topk``. The input S (B, E) holds **f32
    score rows** (output of the framework-side f32 GEMM -- the authored
    spelling already casts to f32 before the matmul; the ascending f32 merge
    over the SK segments in the bs=1 version is equivalent to the row being
    merged upstream). One row per block; sigmoid, bias add, serial top-k
    selection, un-biased gather, and normalization times routed_scale are
    word-for-word identical to the bs=1 version (lowest-index tie-break, Bias
    taken in f32)."""
    EP = ((E + threads - 1) // threads) * threads

    @T.prim_func
    def main(
        S: T.Tensor((B, E), ACC),
        Bias: T.Tensor((E,), ACC),
        Rs: T.Tensor((1,), DT),
        Idx: T.Tensor((B, TOPK), "int32"),
        Wts: T.Tensor((B, TOPK), ACC),
    ):
        with T.Kernel(B, threads=threads) as bb:
            scores = T.alloc_shared((E,), ACC)
            biased = T.alloc_shared((E,), ACC)
            best = T.alloc_var(ACC)
            bi = T.alloc_var("int32")
            den = T.alloc_var(ACC)
            for e in T.Parallel(EP):
                with T.If(e < E):
                    with T.Then():
                        scores[e] = T.sigmoid(S[bb, e])
                        biased[e] = T.sigmoid(S[bb, e]) + Bias[e].astype(ACC)
            T.sync_threads()
            if T.get_thread_binding() == 0:
                den = 0.0
                for t in T.serial(TOPK):
                    best = NEG
                    bi = 0
                    for e in T.serial(E):
                        with T.If(biased[e] > best):
                            with T.Then():
                                best = biased[e]
                                bi = e
                    Idx[bb, t] = bi
                    Wts[bb, t] = scores[bi]
                    biased[bi] = NEG
                    den += scores[bi]
                for t in T.serial(TOPK):
                    Wts[bb, t] = Wts[bb, t] / (den + 1e-20) * Rs[0].astype(ACC)

    return _compile(main)


@lru_cache(maxsize=None)
def attnres_scores_batched(NB: int, H: int, B: int, eps: float,
                           threads: int = 256):
    """Batched version of ``attnres_scores``: block (b, c) computes candidate c
    of row b, and within a row (weightless rms normalization -> dot with the
    f32 fused scoring vector) it is word-for-word identical to the bs=1
    version. Bl holds each row's own snapshot history (B, NB, H)."""
    @T.prim_func
    def main(
        Ps: T.Tensor((B, H), DT),
        Bl: T.Tensor((B, NB, H), DT),
        Sw: T.Tensor((H,), ACC),
        Sc: T.Tensor((B, NB + 1), ACC),
    ):
        with T.Kernel(B, NB + 1, threads=threads) as (bb, bc):
            xs = T.alloc_fragment((H,), ACC)
            sq = T.alloc_fragment((H,), ACC)
            dp = T.alloc_fragment((H,), ACC)
            tot = T.alloc_fragment((1,), ACC)
            dtot = T.alloc_fragment((1,), ACC)
            for i in T.Parallel(H):
                xs[i] = T.if_then_else(
                    bc < NB, Bl[bb, T.min(bc, NB - 1), i], Ps[bb, i]
                ).astype(ACC)
                sq[i] = xs[i] * xs[i]
            T.reduce_sum(sq, tot, dim=0)
            for i in T.Parallel(H):
                dp[i] = xs[i] * T.rsqrt(tot[0] / H + eps) * Sw[i]
            T.reduce_sum(dp, dtot, dim=0)
            if T.get_thread_binding() == 0:
                Sc[bb, bc] = dtot[0]

    return _compile(main)


@lru_cache(maxsize=None)
def attnres_mix_batched(NB: int, H: int, B: int, threads: int = 256):
    """Batched version of ``attnres_mix``: block (b, x) mixes column segment x
    of row b, each block redoes that row's NB+1 term softmax (same as the
    bs=1 version), mixes the un-normalized candidates by probability, and
    lands bf16 once. Requires threads|H."""
    @T.prim_func
    def main(
        Ps: T.Tensor((B, H), DT),
        Bl: T.Tensor((B, NB, H), DT),
        Sc: T.Tensor((B, NB + 1), ACC),
        O: T.Tensor((B, H), DT),
    ):
        with T.Kernel(B, H // threads, threads=threads) as (bb, bx):
            p = T.alloc_shared((NB + 1,), ACC)
            mx = T.alloc_var(ACC)
            den = T.alloc_var(ACC)
            if T.get_thread_binding() == 0:
                mx = NEG
                for c in T.serial(NB + 1):
                    mx = T.max(mx, Sc[bb, c])
                den = 0.0
                for c in T.serial(NB + 1):
                    p[c] = T.exp(Sc[bb, c] - mx)
                    den += T.exp(Sc[bb, c] - mx)
                for c in T.serial(NB + 1):
                    p[c] = p[c] / den
            T.sync_threads()
            acc = T.alloc_fragment((threads,), ACC)
            T.clear(acc)
            for j in T.Parallel(threads):
                for c in T.serial(NB):
                    acc[j] += p[c] * Bl[bb, c, bx * threads + j].astype(ACC)
                acc[j] += p[NB] * Ps[bb, bx * threads + j].astype(ACC)
            for j in T.Parallel(threads):
                O[bb, bx * threads + j] = T.Cast(DT, acc[j])

    return _compile(main)
