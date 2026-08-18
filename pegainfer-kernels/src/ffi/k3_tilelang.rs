//! K3 TileLang-generated batched decode kernels (AOT).
//!
//! Built by the `k3 tilelang` section of `build.rs` from
//! `pegainfer-k3/kernels/generate.py`. Each symbol is a hand-written dispatch
//! launcher over the per-shape kernel instantiations: it returns a raw
//! `cudaError_t` as `int`, with `cudaErrorInvalidValue` for a configuration
//! that was never instantiated and `cudaErrorNotSupported` when the build fell
//! back to the stub tier (no TileLang on the build host).
//!
//! Parameter names follow the certified kernel definitions, so a call site
//! reads like the Python engine's launch sequence. `void*` operands are bf16;
//! `f32` side inputs (router bias, conv weights, KDA gate and o_norm gamma)
//! are f32 because the checkpoint stores them so, and narrowing them to bf16
//! measurably moves routing decisions.
//!
//! Every kernel is batched and `b` is a *compile-time* bucket, not a live row
//! count. The rounding rule and the buffer-size contract that follows from it
//! live in `ops::k3_tilelang`; nothing here validates them.

use core::ffi::c_void;

use cudarc::driver::sys::CUstream;

unsafe extern "C" {
    /// KimiRMSNorm, round-before-scale: `O = bf16(X * rsqrt(mean + eps)) * G`
    /// per row of `X [b, h]`. `G [h]` is a weight shared by every row.
    pub fn k3_rms_norm_rbs_batched(
        x: *const c_void,
        g: *const c_void,
        o: *mut c_void,
        b: i32,
        h: i32,
        stream: CUstream,
    ) -> i32;

    /// Merge the column span `[off, off + n)` of each row's `P [b, split_k, nt]`
    /// f32 partial and land `O [b, n]` bf16 once. `split_k = 1` is the
    /// single-partial case a framework GEMM produces.
    pub fn k3_land_batched(
        p: *const f32,
        o: *mut c_void,
        b: i32,
        nt: i32,
        n: i32,
        off: i32,
        split_k: i32,
        stream: CUstream,
    ) -> i32;

    /// `k3_land_batched` followed by the round-before-scale norm against the
    /// shared gamma `G [n]`.
    pub fn k3_land_rms_norm_rbs_batched(
        p: *const f32,
        g: *const c_void,
        o: *mut c_void,
        b: i32,
        nt: i32,
        n: i32,
        off: i32,
        split_k: i32,
        stream: CUstream,
    ) -> i32;

    /// `O = A + Bt` in bf16 addition, all `[b, n]`.
    pub fn k3_add2_batched(
        a: *const c_void,
        bt: *const c_void,
        o: *mut c_void,
        b: i32,
        n: i32,
        stream: CUstream,
    ) -> i32;

    /// `O = A * bf16(sigmoid(Bt))`, the MLA sigmoid output gate. All `[b, n]`.
    pub fn k3_mul_sigmoid_batched(
        a: *const c_void,
        bt: *const c_void,
        o: *mut c_void,
        b: i32,
        n: i32,
        stream: CUstream,
    ) -> i32;

    /// K3's situ activation `4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25)`, computed
    /// in f32 and landed bf16 once. All `[b, n]`.
    pub fn k3_situ_batched(
        g: *const c_void,
        u: *const c_void,
        o: *mut c_void,
        b: i32,
        n: i32,
        stream: CUstream,
    ) -> i32;

    /// Causal depthwise convolution over the `width`-slot window plus silu.
    /// `P [b, split_k, kp]` f32 partials land into `X [b, kp]` bf16, which is
    /// also the newest window slot; `Cs`/`Sn [b, width - 1, kp]` are the carried
    /// window and its successor; `Y [b, kp]` is the activated output. Conv
    /// weights `Cw [width, kp]` are f32 and carry no batch axis.
    pub fn k3_conv_silu_batched(
        p: *const f32,
        cw: *const f32,
        cs: *const c_void,
        x: *mut c_void,
        y: *mut c_void,
        sn: *mut c_void,
        b: i32,
        kp: i32,
        width: i32,
        split_k: i32,
        stream: CUstream,
    ) -> i32;

    /// One KDA delta-rule step per row. `Q`/`K`/`V`/`G2`/`Out` are
    /// `[b, num_heads * head_dim]` bf16, `Bt [b, num_heads]` bf16, `GP
    /// [b, split_k_gate, num_heads * head_dim]` f32. `Dt`, `Alog` and `Go` are
    /// weights with no batch axis. `State`/`StateN [b, num_heads, head_dim,
    /// head_dim]` f32 are the recurrent state and its successor and must not
    /// alias.
    pub fn k3_kda_core_batched(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        gp: *const f32,
        dt: *const f32,
        alog: *const f32,
        bt: *const c_void,
        g2: *const c_void,
        go: *const f32,
        state: *const f32,
        state_n: *mut f32,
        out: *mut c_void,
        b: i32,
        num_heads: i32,
        head_dim: i32,
        split_k_gate: i32,
        stream: CUstream,
    ) -> i32;

    /// `kda_core`'s tail on its own: per (row, head) f32 rms_norm of the bf16
    /// attention landing `X` times the o_norm gamma `Go [head_dim]`, landed
    /// once, times the bf16 sigmoid of the output gate `G2`. eps compiled in.
    pub fn k3_o_norm_gate_batched(
        x: *const c_void,
        g2: *const c_void,
        go: *const f32,
        out: *mut c_void,
        b: i32,
        num_heads: i32,
        head_dim: i32,
        stream: CUstream,
    ) -> i32;

    /// Sigmoid router plus biased top-k over merged f32 score rows
    /// `S [b, num_experts]`, with `Bias [num_experts]` f32 and the bf16 routed
    /// scale `Rs [1]`. Writes `Idx [b, topk]` i32 and `Wts [b, topk]` f32.
    pub fn k3_router_topk_batched(
        s: *const f32,
        bias: *const f32,
        rs: *const c_void,
        idx: *mut i32,
        wts: *mut f32,
        b: i32,
        num_experts: i32,
        topk: i32,
        stream: CUstream,
    ) -> i32;

    /// Attention-residual candidate scoring: weightless RMS normalization then
    /// a dot with the fused f32 scoring vector `Sw [h]`. `Ps [b, h]` is the
    /// running prefix sum, `Bl [b, blocks, h]` that row's snapshot history;
    /// `Sc [b, blocks + 1]` receives one score per candidate.
    pub fn k3_attnres_scores_batched(
        ps: *const c_void,
        bl: *const c_void,
        sw: *const f32,
        sc: *mut f32,
        b: i32,
        blocks: i32,
        h: i32,
        stream: CUstream,
    ) -> i32;

    /// Softmax over each row's `blocks + 1` scores, then a probability-weighted
    /// mix of the *un-normalized* candidates landing `O [b, h]` bf16 once.
    pub fn k3_attnres_mix_batched(
        ps: *const c_void,
        bl: *const c_void,
        sc: *const f32,
        o: *mut c_void,
        b: i32,
        blocks: i32,
        h: i32,
        stream: CUstream,
    ) -> i32;
}
