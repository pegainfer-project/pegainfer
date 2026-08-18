"""Vendored TileLang kernel definitions for the K3 batched decode step.

This file is a **verbatim** subset of the certified upstream kernel module: the
shared prologue and the eleven batched kernel factories, copied character for
character. Nothing here is re-spelled, re-indented or "cleaned up", and no
kernel body is edited to fit this repository. The upstream module is the
authority on what these kernels compute; it carries the bitwise parity gates
that certified them, and any divergence here would silently un-certify the
result.

Only the batched factories are vendored. The batch size ``B`` is a static
compile-time dimension, so ``B = 1`` is a first-class instantiation whose
per-row spelling is word for word the certified single-row kernel -- one family
therefore serves both single-stream and high-concurrency decode. The upstream
single-row factories, the split-K ``gemv`` and the MXFP4 ``expert_gemv`` /
``packed_expert_gemv`` families are deliberately absent: dense projections are
served by cuBLASLt and the routed experts by the DeepGEMM masked grouped-GEMM
chain, so vendoring them here would ship kernels nothing launches.

Two properties of these definitions constrain how the generator may use them:

* Some bodies branch or loop in **plain Python** around the TileLang IR (tile
  counts, unrolled tables, window taps). Those branches run at trace time and
  pick different IR per shape, which is exactly why a whole factory is vendored
  rather than a per-shape transcription -- there is no single body to
  transcribe.
* Every factory is ``lru_cache``d on its argument tuple and returns a compiled
  kernel for one static shape. The generator walks argument tuples; it never
  reaches inside a body.

`_compile` targets the local GPU, which a build host need not have. The
generator rebinds this module's `_compile` to pin an explicit target instead of
editing the line below -- see `generate.py:_pin_arch`.
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
# Batched decode kernels, vendored verbatim (including the upstream section
# comment below, which names the parity gates that certified them).
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# batched variants (for EP / high-concurrency serving; every row is independent
# and the per-row spelling is word-for-word identical to the bs=1 version, so
# they are gated **bitwise** against the bs=1 certified kernels, see
# fast/check_batched.py (router / attn_res) and fast/check_batched2.py
# (conv / KDA / MLA / the row-parallel elementwise and reduction kernels).
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


# --- row-parallel elementwise / reduction ---------------------------------- #


@lru_cache(maxsize=None)
def rms_norm_rbs_batched(H: int, B: int, eps: float, threads: int = 256):
    """Batched ``rms_norm_rbs``: one row per block. Gamma is a weight and is
    shared by every row; the per-row body (f32 squares, block reduce, bf16
    landing before the gamma product) is word-for-word the bs=1 body."""
    @T.prim_func
    def main(X: T.Tensor((B, H), DT), G: T.Tensor((H,), DT),
             O: T.Tensor((B, H), DT)):
        with T.Kernel(B, threads=threads) as bb:
            xs = T.alloc_fragment((H,), ACC)
            sq = T.alloc_fragment((H,), ACC)
            tot = T.alloc_fragment((1,), ACC)
            for i in T.Parallel(H):
                xs[i] = X[bb, i].astype(ACC)
                sq[i] = xs[i] * xs[i]
            T.reduce_sum(sq, tot, dim=0)
            for i in T.Parallel(H):
                O[bb, i] = (xs[i] * T.rsqrt(tot[0] / H + eps)).astype(DT) * G[i]

    return _compile(main)


@lru_cache(maxsize=None)
def land_batched(NT: int, N: int, OFF: int, SK: int, B: int, threads: int = 256):
    """Batched ``land``: block (b, x) merges row b's partials for one column
    span. The serial ascending-s merge and the single bf16 landing are the
    bs=1 body verbatim."""
    NPAD = ((N + threads - 1) // threads) * threads

    @T.prim_func
    def main(P: T.Tensor((B, SK, NT), ACC), O: T.Tensor((B, N), DT)):
        with T.Kernel(B, NPAD // threads, threads=threads) as (bb, bx):
            acc = T.alloc_fragment((threads,), ACC)
            T.clear(acc)
            for j in T.Parallel(threads):
                for s in T.serial(SK):
                    acc[j] += P[bb, s, OFF + T.min(bx * threads + j, N - 1)]
            for j in T.Parallel(threads):
                with T.If(bx * threads + j < N):
                    with T.Then():
                        O[bb, bx * threads + j] = T.Cast(DT, acc[j])

    return _compile(main)


@lru_cache(maxsize=None)
def land_rms_norm_rbs_batched(NT: int, N: int, OFF: int, SK: int, B: int,
                              eps: float, threads: int = 256):
    """Batched ``land_rms_norm_rbs``: one row per block, merge -> bf16 landing
    -> round-before-scale norm, word-for-word the bs=1 body."""
    @T.prim_func
    def main(P: T.Tensor((B, SK, NT), ACC), G: T.Tensor((N,), DT),
             O: T.Tensor((B, N), DT)):
        with T.Kernel(B, threads=threads) as bb:
            xs = T.alloc_fragment((N,), ACC)
            sq = T.alloc_fragment((N,), ACC)
            tot = T.alloc_fragment((1,), ACC)
            T.clear(xs)
            for i in T.Parallel(N):
                for s in T.serial(SK):
                    xs[i] += P[bb, s, OFF + i]
            for i in T.Parallel(N):
                xs[i] = T.Cast(DT, xs[i]).astype(ACC)   # matmul's bf16 landing
                sq[i] = xs[i] * xs[i]
            T.reduce_sum(sq, tot, dim=0)
            for i in T.Parallel(N):
                O[bb, i] = (xs[i] * T.rsqrt(tot[0] / N + eps)).astype(DT) * G[i]

    return _compile(main)


@lru_cache(maxsize=None)
def add2_batched(N: int, B: int, threads: int = 256):
    """Batched ``add2``: block (b, x) adds one column segment of row b in bf16.
    Requires threads|N."""
    @T.prim_func
    def main(A: T.Tensor((B, N), DT), Bt: T.Tensor((B, N), DT),
             O: T.Tensor((B, N), DT)):
        with T.Kernel(B, N // threads, threads=threads) as (bb, bx):
            for j in T.Parallel(threads):
                O[bb, bx * threads + j] = (A[bb, bx * threads + j]
                                           + Bt[bb, bx * threads + j])

    return _compile(main)


@lru_cache(maxsize=None)
def mul_sigmoid_batched(N: int, B: int, threads: int = 256):
    """Batched ``mul_sigmoid``: block (b, x) gates one column segment of row b.
    f32 sigmoid landing back in bf16, then the bf16 product -- the bs=1 body
    verbatim. Requires threads|N."""
    @T.prim_func
    def main(A: T.Tensor((B, N), DT), Bt: T.Tensor((B, N), DT),
             O: T.Tensor((B, N), DT)):
        with T.Kernel(B, N // threads, threads=threads) as (bb, bx):
            for j in T.Parallel(threads):
                n = bx * threads + j
                O[bb, n] = A[bb, n] * T.Cast(DT, T.sigmoid(Bt[bb, n].astype(ACC)))

    return _compile(main)


@lru_cache(maxsize=None)
def situ_batched(N: int, B: int, threads: int = 256):
    """Batched ``situ``: block (b, x) applies the activation to one column
    segment of row b; beta=4 / linear_beta=25 stay compiled in and the f32
    chain lands bf16 once, as in the bs=1 body. Requires threads|N."""
    @T.prim_func
    def main(G: T.Tensor((B, N), DT), U: T.Tensor((B, N), DT),
             O: T.Tensor((B, N), DT)):
        with T.Kernel(B, N // threads, threads=threads) as (bb, bx):
            for j in T.Parallel(threads):
                n = bx * threads + j
                g = G[bb, n].astype(ACC)
                u = U[bb, n].astype(ACC)
                O[bb, n] = T.Cast(
                    DT,
                    4.0 * T.tanh(g / 4.0) * T.sigmoid(g) * (25.0 * T.tanh(u / 25.0)),
                )

    return _compile(main)


# --- KDA ------------------------------------------------------------------- #


@lru_cache(maxsize=None)
def conv_silu_batched(KP: int, W: int, SK: int, B: int, threads: int = 256):
    """Batched ``conv_silu``: block (b, x) advances row b's conv window over one
    column segment. The conv weights Cw are a weight (f32, no batch axis); the
    window state Cs/Sn gains a leading batch axis -- one independent window per
    sequence, laid out [B, W-1, KP] so that a row's window is the contiguous
    bs=1 [W-1, KP] block. Merge -> bf16 landing -> f32 window products (Cw is
    f32 by contract) -> f32 silu landing bf16 is the bs=1 body verbatim.
    Requires threads|KP."""
    WS = W - 1

    @T.prim_func
    def main(
        P: T.Tensor((B, SK, KP), ACC),
        Cw: T.Tensor((W, KP), ACC),
        Cs: T.Tensor((B, WS, KP), DT),
        X: T.Tensor((B, KP), DT),
        Y: T.Tensor((B, KP), DT),
        Sn: T.Tensor((B, WS, KP), DT),
    ):
        with T.Kernel(B, KP // threads, threads=threads) as (bb, bx):
            xa = T.alloc_fragment((threads,), ACC)
            ca = T.alloc_fragment((threads,), ACC)
            T.clear(xa)
            for j in T.Parallel(threads):
                c = bx * threads + j
                for s in T.serial(SK):
                    xa[j] += P[bb, s, c]
            T.clear(ca)
            for j in T.Parallel(threads):
                c = bx * threads + j
                xb = T.Cast(DT, xa[j])
                X[bb, c] = xb
                for t in T.serial(WS):
                    ca[j] += Cs[bb, t, c].astype(ACC) * Cw[t, c]
                    Sn[bb, t, c] = T.if_then_else(
                        t + 1 < WS, Cs[bb, T.min(t + 1, WS - 1), c], xb
                    )
                ca[j] += xb.astype(ACC) * Cw[WS, c]
            for j in T.Parallel(threads):
                c = bx * threads + j
                sb = T.Cast(DT, ca[j]).astype(ACC)
                Y[bb, c] = T.Cast(DT, sb * T.sigmoid(sb))

    return _compile(main)


@lru_cache(maxsize=None)
def kda_core_batched(KH: int, KD: int, SKG: int, B: int, lb: float, eps: float):
    """Batched ``kda_core``: block (b, h) runs row b's head h, thread count =
    head_dim. Dt (dt_bias), Alog (A_log) and Go (o_norm gamma) are weights and
    keep their bs=1 shapes; the recurrent state gains a leading batch axis,
    [B, head, v_dim, k_dim], so a row's state is the contiguous bs=1
    [heads, v_dim, k_dim] block. Everything inside a block -- the bf16 l2norm
    chain, the f32 delta rule with all per-dv steps kept inside one Parallel
    loop, the single attn bf16 landing, the f32 rms_norm and the bf16 sigmoid
    gate -- is the bs=1 body verbatim."""
    KP = KH * KD
    scale = float(KD) ** -0.5

    @T.prim_func
    def main(
        Q: T.Tensor((B, KP), DT),
        K: T.Tensor((B, KP), DT),
        V: T.Tensor((B, KP), DT),
        GP: T.Tensor((B, SKG, KP), ACC),
        Dt: T.Tensor((KP,), ACC),
        Alog: T.Tensor((KH,), ACC),
        Bt: T.Tensor((B, KH), DT),
        G2: T.Tensor((B, KP), DT),
        Go: T.Tensor((KD,), ACC),  # o_norm gamma: checkpoint stores F32 (as conv)
        State: T.Tensor((B, KH, KD, KD), ACC),
        StateN: T.Tensor((B, KH, KD, KD), ACC),
        Out: T.Tensor((B, KP), DT),
    ):
        with T.Kernel(B, KH, threads=KD) as (bb, bh):
            qsq = T.alloc_fragment((KD,), ACC)
            ksq = T.alloc_fragment((KD,), ACC)
            qtot = T.alloc_fragment((1,), ACC)
            ktot = T.alloc_fragment((1,), ACC)
            ga = T.alloc_fragment((KD,), ACC)
            mfr = T.alloc_fragment((KD,), ACC)
            dfr = T.alloc_fragment((KD,), ACC)
            afr = T.alloc_fragment((KD,), ACC)
            asq = T.alloc_fragment((KD,), ACC)
            atot = T.alloc_fragment((1,), ACC)
            qss = T.alloc_shared((KD,), ACC)
            kns = T.alloc_shared((KD,), ACC)
            decs = T.alloc_shared((KD,), ACC)
            attnb = T.alloc_shared((KD,), DT)

            # Squares land in bf16 per term (authored tf.square is on bf16),
            # the sum is f32.
            for d in T.Parallel(KD):
                qsq[d] = (Q[bb, bh * KD + d] * Q[bb, bh * KD + d]).astype(ACC)
                ksq[d] = (K[bb, bh * KD + d] * K[bb, bh * KD + d]).astype(ACC)
            T.reduce_sum(qsq, qtot, dim=0)
            T.reduce_sum(ksq, ktot, dim=0)
            T.clear(ga)
            for d in T.Parallel(KD):
                # authored l2_normalize is an all-bf16 chain: the sum lands
                # bf16, then +eps, rsqrt and the product with x land in bf16
                # step by step; afterwards cast to f32 (q also times scale).
                qr = T.Cast(DT, T.rsqrt(T.Cast(DT, qtot[0]).astype(ACC) + 1e-6))
                kr = T.Cast(DT, T.rsqrt(T.Cast(DT, ktot[0]).astype(ACC) + 1e-6))
                qss[d] = (Q[bb, bh * KD + d] * qr).astype(ACC) * scale
                kns[d] = (K[bb, bh * KD + d] * kr).astype(ACC)
                for s in T.serial(SKG):
                    ga[d] += GP[bb, s, bh * KD + d]
                raw = T.Cast(DT, ga[d]).astype(ACC) + Dt[bh * KD + d]
                decs[d] = T.exp(lb * T.sigmoid(T.exp(Alog[bh]) * raw))
            T.sync_threads()

            # delta rule, all f32; one v_dim row per thread, serial over k_dim.
            # All steps for a given dv are kept inside the same Parallel loop:
            # a fragment's thread mapping may differ between separate Parallel
            # loops (different vectorization layouts), so reading it by index
            # across loops mismatches -- a lesson measured in this repo.
            T.clear(mfr)
            T.clear(afr)
            for dv in T.Parallel(KD):
                for k in T.serial(KD):
                    mfr[dv] += State[bb, bh, dv, k] * decs[k] * kns[k]
                dfr[dv] = (
                    V[bb, bh * KD + dv].astype(ACC) - mfr[dv]
                ) * T.sigmoid(Bt[bb, bh].astype(ACC))
                for k in T.serial(KD):
                    StateN[bb, bh, dv, k] = (State[bb, bh, dv, k] * decs[k]
                                             + dfr[dv] * kns[k])
                    afr[dv] += StateN[bb, bh, dv, k] * qss[k]
                attnb[dv] = T.Cast(DT, afr[dv])     # attn's single bf16 landing
            T.sync_threads()

            # tf.rms_norm: compute in f32, multiply gamma in f32, land once;
            # then multiply by the bf16 gate.
            for d in T.Parallel(KD):
                asq[d] = attnb[d].astype(ACC) * attnb[d].astype(ACC)
            T.reduce_sum(asq, atot, dim=0)
            for d in T.Parallel(KD):
                Out[bb, bh * KD + d] = T.Cast(
                    DT,
                    attnb[d].astype(ACC) * T.rsqrt(atot[0] / KD + eps)
                    * Go[d].astype(ACC),
                ) * T.Cast(DT, T.sigmoid(G2[bb, bh * KD + d].astype(ACC)))

    return _compile(main)


def o_norm_gate_batched(KH: int, KD: int, B: int, eps: float):
    """``kda_core``'s tail on its own: per (row, head) the f32 rms_norm of the
    bf16 attention landing times the o_norm gamma, landed once, times the bf16
    sigmoid of the output-gate projection -- word-for-word the last loop of
    ``kda_core_batched``. Chunked prefill computes the attention elsewhere
    (FlashKDA) and finishes each row through this identical spelling."""
    KP = KH * KD

    @T.prim_func
    def main(
        X: T.Tensor((B, KP), DT),
        G2: T.Tensor((B, KP), DT),
        Go: T.Tensor((KD,), ACC),
        Out: T.Tensor((B, KP), DT),
    ):
        with T.Kernel(B, KH, threads=KD) as (bb, bh):
            asq = T.alloc_fragment((KD,), ACC)
            atot = T.alloc_fragment((1,), ACC)
            for d in T.Parallel(KD):
                asq[d] = X[bb, bh * KD + d].astype(ACC) * X[bb, bh * KD + d].astype(ACC)
            T.reduce_sum(asq, atot, dim=0)
            for d in T.Parallel(KD):
                Out[bb, bh * KD + d] = T.Cast(
                    DT,
                    X[bb, bh * KD + d].astype(ACC) * T.rsqrt(atot[0] / KD + eps)
                    * Go[d].astype(ACC),
                ) * T.Cast(DT, T.sigmoid(G2[bb, bh * KD + d].astype(ACC)))

    return _compile(main)
