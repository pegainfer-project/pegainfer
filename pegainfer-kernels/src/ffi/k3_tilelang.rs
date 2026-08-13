//! K3 TileLang-generated decode kernels (AOT).
//!
//! Built by the `k3 tilelang` section of `build.rs` from
//! `pegainfer-k3/kernels/generate.py`. Each symbol is a hand-written dispatch
//! launcher over the per-shape kernel instantiations: it returns a raw
//! `cudaError_t` as `int`, with `cudaErrorInvalidValue` for a shape that was
//! never instantiated and `cudaErrorNotSupported` when the build fell back to
//! the stub tier (no TileLang on the build host).
//!
//! `batch` is a compile-time bucket, not the live row count — see
//! `ops::k3_tilelang` for the rounding and the buffer-size contract.

use core::ffi::c_void;

use cudarc::driver::sys::CUstream;

unsafe extern "C" {
    /// Sigmoid router + biased top-k: `scores [batch, num_experts]` f32 and
    /// `bias [num_experts]` f32 in, `idx [batch, topk]` i32 and
    /// `weights [batch, topk]` f32 out, weights normalized and scaled by the
    /// bf16 scalar `routed_scale [1]`.
    pub fn k3_router_topk(
        scores: *const f32,
        bias: *const f32,
        routed_scale: *const c_void,
        idx: *mut i32,
        weights: *mut f32,
        num_experts: i32,
        topk: i32,
        batch: i32,
        stream: CUstream,
    ) -> i32;

    /// Attention-residual candidate scoring: `prefix_sum [batch, hidden]` and
    /// `blocks_in [batch, num_blocks, hidden]` bf16 plus the fused f32 scoring
    /// vector `score_weight [hidden]` produce `scores_out [batch, num_blocks + 1]`
    /// f32 (candidate `num_blocks` is the prefix sum itself).
    pub fn k3_attnres_scores(
        prefix_sum: *const c_void,
        blocks_in: *const c_void,
        score_weight: *const f32,
        scores_out: *mut f32,
        num_blocks: i32,
        hidden: i32,
        batch: i32,
        stream: CUstream,
    ) -> i32;

    /// Attention-residual mixing: softmax `scores [batch, num_blocks + 1]`,
    /// then mix the un-normalized candidates into `out [batch, hidden]` bf16.
    pub fn k3_attnres_mix(
        prefix_sum: *const c_void,
        blocks_in: *const c_void,
        scores: *const f32,
        out: *mut c_void,
        num_blocks: i32,
        hidden: i32,
        batch: i32,
        stream: CUstream,
    ) -> i32;
}
