//! The hd512 paged-KV page-id trap. One trap per binary, for the reason
//! given in hd512_qk_rope_trap.rs.
//!
//! Manual gate — CI compiles this but never runs it.

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::qk_norm_partial_rope_paged_prefill_hd512_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

const HD: usize = 512;
const ROTARY_DIM: usize = 128;
const EPS: f32 = 1e-6;
const NUM_Q_HEADS: usize = 2;
const NUM_KV_HEADS: usize = 2;
const Q_DIM: usize = NUM_Q_HEADS * HD;
const KV_DIM: usize = NUM_KV_HEADS * HD;
const SEQ_LEN: usize = 4;

/// Checking `page_indices` on the host would require a D2H synchronization.
/// The async trap may surface at the launcher or the next sync; either
/// counts, while a silent `Ok` does not.
#[test]
fn paged_page_id_trap_is_visible() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    // One page in the pool → num_pages = 1; page_indices = [5] is out of
    // range, but every wrapper check must still PASS so the launch happens
    // and the kernel traps: coverage 1 page * 4 slots >= start_pos 0 +
    // seq_len 4, tables 4 rows (cos_max_pos 4), positions 0..=3 all read
    // page_indices[0] = 5.
    let layout = PagedKvLayout::new(1, NUM_KV_HEADS, HD, 4);
    let q = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q alloc");
    let k = HiddenStates::zeros(ctx, KV_DIM, SEQ_LEN).expect("k alloc");
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q_out alloc");
    let pool: CudaSlice<bf16> = ctx
        .stream
        .alloc_zeros(layout.page_stride)
        .expect("pool alloc");
    let qn = DeviceVec::zeros(ctx, HD).expect("qn alloc");
    let kn = DeviceVec::zeros(ctx, HD).expect("kn alloc");
    let cos_dev = DeviceVec::zeros(ctx, 4 * ROTARY_DIM).expect("cos alloc");
    let sin_dev = DeviceVec::zeros(ctx, 4 * ROTARY_DIM).expect("sin alloc");
    let page_indices: CudaSlice<i32> = ctx.stream.clone_htod(&[5]).expect("page_indices H2D");

    let res = qk_norm_partial_rope_paged_prefill_hd512_into(
        ctx,
        &q,
        &k,
        &mut q_out,
        0,
        &pool,
        &layout,
        &qn,
        &kn,
        &cos_dev,
        &sin_dev,
        0, // layer
        &page_indices,
        0,
        0, // start_pos
        4, // cos_max_pos
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        ROTARY_DIM,
        EPS,
    );
    if res.is_ok() {
        assert!(
            ctx.sync().is_err(),
            "page id 5 beyond num_pages 1 must fail visibly"
        );
    }
}
