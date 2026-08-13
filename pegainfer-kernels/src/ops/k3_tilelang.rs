//! K3 TileLang decode kernels: safe wrappers over the AOT dispatch launchers
//! in `ffi::k3_tilelang`.
//!
//! Every kernel here is row-independent, and the batch dimension is a static
//! compile dimension of the generated code. A step with `rows` live rows runs
//! the next bucket up ([`K3_BATCH_BUCKETS`]) and discards the tail rows, so
//! the caller must hand in buffers sized for the *bucket*, not for `rows`.
//! Allocating every buffer at [`K3_MAX_BATCH`] once and reusing it is the
//! intended shape — it also keeps the pointers stable for CUDA Graph capture.

use core::ffi::c_void;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

/// Batch buckets the generator instantiates, ascending.
pub const K3_BATCH_BUCKETS: [usize; 10] = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128];
/// Largest bucket, i.e. the row capacity every reusable buffer should have.
pub const K3_MAX_BATCH: usize = 128;

/// Routed-expert counts instantiated for the router.
pub const K3_ROUTER_EXPERTS: [usize; 2] = [224, 896];
/// Route width baked into the router instantiations.
pub const K3_ROUTER_TOPK: usize = 16;

/// Hidden size baked into the attention-residual instantiations.
pub const K3_ATTNRES_HIDDEN: usize = 7168;
/// Largest attention-residual candidate block count (the history grows 1..8
/// across the layer stack).
pub const K3_ATTNRES_MAX_BLOCKS: usize = 8;

/// Round a live row count up to the nearest instantiated batch bucket.
pub fn k3_batch_bucket(rows: usize) -> Result<usize> {
    ensure!(rows > 0, "K3 batch bucket needs rows > 0");
    K3_BATCH_BUCKETS
        .into_iter()
        .find(|bucket| *bucket >= rows)
        .ok_or_else(|| {
            anyhow!("K3 kernels support at most {K3_MAX_BATCH} rows per step, got {rows}")
        })
}

/// The launchers return a raw `cudaError_t`; 0 is `cudaSuccess`.
fn check(rc: i32, what: &str) -> Result<()> {
    ensure!(rc == 0, "{what} failed with cudaError={rc}");
    Ok(())
}

/// Sigmoid router + biased top-k selection over `rows` live rows.
///
/// `scores` is the f32 router logit matrix `[bucket, num_experts]` (the
/// framework-side GEMM already lands in f32), `bias` the f32 per-expert
/// correction `[num_experts]`, and `routed_scale` the bf16 scalar the
/// normalized weights are multiplied by. Writes `idx [bucket, K3_ROUTER_TOPK]`
/// and `weights [bucket, K3_ROUTER_TOPK]`.
pub fn k3_router_topk_launch(
    ctx: &DeviceContext,
    num_experts: usize,
    rows: usize,
    scores: &CudaSlice<f32>,
    bias: &CudaSlice<f32>,
    routed_scale: &CudaSlice<bf16>,
    idx: &mut CudaSlice<i32>,
    weights: &mut CudaSlice<f32>,
) -> Result<()> {
    let batch = k3_batch_bucket(rows)?;
    ensure!(
        K3_ROUTER_EXPERTS.contains(&num_experts),
        "K3 router is instantiated for {K3_ROUTER_EXPERTS:?} experts, got {num_experts}"
    );
    ensure!(
        scores.len() >= batch * num_experts
            && bias.len() >= num_experts
            && !routed_scale.is_empty()
            && idx.len() >= batch * K3_ROUTER_TOPK
            && weights.len() >= batch * K3_ROUTER_TOPK,
        "K3 router buffers too small for bucket {batch} x {num_experts} experts: scores {}, bias {}, routed_scale {}, idx {}, weights {}",
        scores.len(),
        bias.len(),
        routed_scale.len(),
        idx.len(),
        weights.len()
    );
    let (scores_ptr, _scores_guard) = scores.device_ptr(&ctx.stream);
    let (bias_ptr, _bias_guard) = bias.device_ptr(&ctx.stream);
    let (scale_ptr, _scale_guard) = routed_scale.device_ptr(&ctx.stream);
    let (idx_ptr, _idx_guard) = idx.device_ptr_mut(&ctx.stream);
    let (weights_ptr, _weights_guard) = weights.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_router_topk(
            scores_ptr as *const f32,
            bias_ptr as *const f32,
            scale_ptr as *const c_void,
            idx_ptr as *mut i32,
            weights_ptr as *mut f32,
            num_experts as i32,
            K3_ROUTER_TOPK as i32,
            batch as i32,
            ctx.stream.cu_stream(),
        )
    };
    check(rc, &format!("K3 router top-k (E={num_experts}, B={batch})"))
}

/// Attention-residual candidate scoring: weightless RMS normalization of each
/// candidate followed by a dot with the fused f32 scoring vector.
///
/// `prefix_sum [bucket, H]` and `blocks [bucket, num_blocks, H]` are bf16,
/// `score_weight [H]` is f32, and `scores_out [bucket, num_blocks + 1]` gets
/// one score per candidate — index `num_blocks` is the prefix sum itself.
pub fn k3_attnres_scores_launch(
    ctx: &DeviceContext,
    num_blocks: usize,
    rows: usize,
    prefix_sum: &CudaSlice<bf16>,
    blocks: &CudaSlice<bf16>,
    score_weight: &CudaSlice<f32>,
    scores_out: &mut CudaSlice<f32>,
) -> Result<()> {
    let batch = k3_batch_bucket(rows)?;
    let hidden = K3_ATTNRES_HIDDEN;
    ensure!(
        (1..=K3_ATTNRES_MAX_BLOCKS).contains(&num_blocks),
        "K3 attention-residual scoring is instantiated for 1..={K3_ATTNRES_MAX_BLOCKS} blocks, got {num_blocks}"
    );
    ensure!(
        prefix_sum.len() >= batch * hidden
            && blocks.len() >= batch * num_blocks * hidden
            && score_weight.len() >= hidden
            && scores_out.len() >= batch * (num_blocks + 1),
        "K3 attention-residual scoring buffers too small for bucket {batch} x {num_blocks} blocks: prefix_sum {}, blocks {}, score_weight {}, scores_out {}",
        prefix_sum.len(),
        blocks.len(),
        score_weight.len(),
        scores_out.len()
    );
    let (prefix_ptr, _prefix_guard) = prefix_sum.device_ptr(&ctx.stream);
    let (blocks_ptr, _blocks_guard) = blocks.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = score_weight.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = scores_out.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_attnres_scores(
            prefix_ptr as *const c_void,
            blocks_ptr as *const c_void,
            weight_ptr as *const f32,
            out_ptr as *mut f32,
            num_blocks as i32,
            hidden as i32,
            batch as i32,
            ctx.stream.cu_stream(),
        )
    };
    check(
        rc,
        &format!("K3 attention-residual scoring (NB={num_blocks}, B={batch})"),
    )
}

/// Attention-residual mixing: per-row softmax over the `num_blocks + 1`
/// candidate scores, then a probability-weighted mix of the *un-normalized*
/// candidates landing in bf16 once. `out [bucket, H]` is bf16.
pub fn k3_attnres_mix_launch(
    ctx: &DeviceContext,
    num_blocks: usize,
    rows: usize,
    prefix_sum: &CudaSlice<bf16>,
    blocks: &CudaSlice<bf16>,
    scores: &CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    let batch = k3_batch_bucket(rows)?;
    let hidden = K3_ATTNRES_HIDDEN;
    ensure!(
        (1..=K3_ATTNRES_MAX_BLOCKS).contains(&num_blocks),
        "K3 attention-residual mixing is instantiated for 1..={K3_ATTNRES_MAX_BLOCKS} blocks, got {num_blocks}"
    );
    ensure!(
        prefix_sum.len() >= batch * hidden
            && blocks.len() >= batch * num_blocks * hidden
            && scores.len() >= batch * (num_blocks + 1)
            && out.len() >= batch * hidden,
        "K3 attention-residual mixing buffers too small for bucket {batch} x {num_blocks} blocks: prefix_sum {}, blocks {}, scores {}, out {}",
        prefix_sum.len(),
        blocks.len(),
        scores.len(),
        out.len()
    );
    let (prefix_ptr, _prefix_guard) = prefix_sum.device_ptr(&ctx.stream);
    let (blocks_ptr, _blocks_guard) = blocks.device_ptr(&ctx.stream);
    let (scores_ptr, _scores_guard) = scores.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_attnres_mix(
            prefix_ptr as *const c_void,
            blocks_ptr as *const c_void,
            scores_ptr as *const f32,
            out_ptr as *mut c_void,
            num_blocks as i32,
            hidden as i32,
            batch as i32,
            ctx.stream.cu_stream(),
        )
    };
    check(
        rc,
        &format!("K3 attention-residual mixing (NB={num_blocks}, B={batch})"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_bucket_rounds_up_and_rejects_overflow() {
        assert_eq!(k3_batch_bucket(1).unwrap(), 1);
        assert_eq!(k3_batch_bucket(3).unwrap(), 4);
        assert_eq!(k3_batch_bucket(48).unwrap(), 48);
        assert_eq!(k3_batch_bucket(49).unwrap(), 64);
        assert_eq!(k3_batch_bucket(128).unwrap(), 128);
        assert!(k3_batch_bucket(0).is_err());
        assert!(k3_batch_bucket(129).is_err());
    }
}
