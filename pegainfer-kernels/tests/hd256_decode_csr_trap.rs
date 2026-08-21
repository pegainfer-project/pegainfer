//! The batched decode prep's CSR window trap. One trap per binary, because a
//! `__trap()` poisons the context for everything that follows.
//!
//! Manual gate — CI compiles this but never runs it.

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::qkv_norm_rope_paged_decode_hd256_plain_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

const HD: usize = 256;
const EPS: f32 = 1e-6;
const NUM_Q_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 2;
const Q_DIM: usize = NUM_Q_HEADS * HD;
const KV_DIM: usize = NUM_KV_HEADS * HD;

/// A request's page window lives on the device, so the host cannot check it
/// without a synchronization; the kernel checks it against the array length
/// the wrapper hands in.
#[test]
fn a_window_past_the_page_array_traps() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let layout = PagedKvLayout::new(1, NUM_KV_HEADS, HD, 4);
    let q = HiddenStates::zeros(ctx, Q_DIM, 1).expect("q alloc");
    let k = HiddenStates::zeros(ctx, KV_DIM, 1).expect("k alloc");
    let v = HiddenStates::zeros(ctx, KV_DIM, 1).expect("v alloc");
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, 1).expect("q_out alloc");
    let pool: CudaSlice<bf16> = ctx
        .stream
        .alloc_zeros(layout.page_stride)
        .expect("pool alloc");
    let qn = DeviceVec::zeros(ctx, HD).expect("qn alloc");
    let kn = DeviceVec::zeros(ctx, HD).expect("kn alloc");
    let cos_dev = DeviceVec::zeros(ctx, 4 * HD).expect("cos alloc");
    let sin_dev = DeviceVec::zeros(ctx, 4 * HD).expect("sin alloc");
    // One page in the array, but the window says two.
    let pages: CudaSlice<i32> = ctx.stream.clone_htod(&[0i32]).expect("pages H2D");
    let indptr: CudaSlice<i32> = ctx.stream.clone_htod(&[0i32, 2]).expect("indptr H2D");
    let origins: CudaSlice<i32> = ctx.stream.clone_htod(&[0i32]).expect("origins H2D");
    let positions: CudaSlice<i32> = ctx.stream.clone_htod(&[0i32]).expect("positions H2D");

    let res = qkv_norm_rope_paged_decode_hd256_plain_into(
        ctx,
        &q,
        &k,
        &v,
        &mut q_out,
        0,
        &pool,
        &layout,
        &qn,
        &kn,
        &cos_dev,
        &sin_dev,
        0, // layer
        &pages,
        &indptr,
        &origins,
        &positions,
        4, // cos_max_pos
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HD,
        EPS,
    );
    match res {
        Ok(()) => assert!(
            ctx.sync().is_err(),
            "a window of two pages out of a one-page array must fail visibly"
        ),
        // A sticky trap can surface at the launcher instead of at the sync.
        Err(e) => {
            let report = format!("{e:#}");
            assert!(
                report.contains("qkv_norm_rope_paged_decode_hd256_plain_cuda"),
                "every host check passes here, so only the kernel can refuse; got: {report}"
            );
        }
    }
}
