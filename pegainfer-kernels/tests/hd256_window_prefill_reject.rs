//! The attention kernels compute addresses from page ids and the last-page
//! length with no device-side bounds check, so a plan's host-side bounds are
//! the only bounds. This drives them through the public wrapper.

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::batch_prefill_paged_window_hd256_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

const HD: usize = 256;
const NUM_Q_HEADS: usize = 16;
const NUM_KV_HEADS: usize = 8;
const PAGE_SIZE: usize = 2;
const NUM_LAYERS: usize = 2;
const POOL_PAGES: usize = 8;
const SEQ_LEN: usize = 2;

fn run(ctx: &DeviceContext, page_indices: &[i32], last_page_len: usize) -> anyhow::Result<()> {
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let pool: CudaSlice<bf16> = ctx.stream.alloc_zeros(layout.page_stride * POOL_PAGES)?;
    let plan = PrefillPagedPlan::new_with_cta_tile_q(
        ctx,
        page_indices,
        last_page_len,
        0, // start_pos
        SEQ_LEN,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HD,
        0, // auto tile
    )?;
    let q = HiddenStates::zeros(ctx, NUM_Q_HEADS * HD, SEQ_LEN)?;
    let mut output = HiddenStates::zeros(ctx, NUM_Q_HEADS * HD, SEQ_LEN)?;
    batch_prefill_paged_window_hd256_into(
        ctx,
        &q,
        &pool,
        &layout,
        0, // layer
        &plan,
        &mut output,
        NUM_Q_HEADS,
        1.0,
        -1,
    )
}

#[test]
fn invalid_plan_metadata_is_rejected_before_launch() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let err = run(ctx, &[99], PAGE_SIZE).expect_err("page id past the pool must be rejected");
    assert!(err.to_string().contains("references page"), "{err}");
    let err = run(ctx, &[0], PAGE_SIZE + 1).expect_err("oversized last page must be rejected");
    assert!(err.to_string().contains("last-page length"), "{err}");
    run(ctx, &[0], PAGE_SIZE).expect("in-range metadata must launch");

    // The batch plan narrows the same host bound, where a length past i32
    // would truncate into a legal-looking one.
    // A per-request page list is what the kernels derive that request's KV
    // length from, so an empty one is malformed even when the batch as a
    // whole carries pages.
    let err = PrefillPagedPlan::new_batch_with_cta_tile_q(
        ctx,
        &[Vec::new(), vec![0]],
        &[PAGE_SIZE, PAGE_SIZE],
        &[0, 0],
        &[SEQ_LEN, SEQ_LEN],
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HD,
        0, // auto tile
    )
    .map(|_| ())
    .expect_err("a request with no pages must be rejected");
    assert!(err.to_string().contains("has no pages"), "{err}");

    let err = PrefillPagedPlan::new_batch_with_cta_tile_q(
        ctx,
        &[vec![0]],
        &[u32::MAX as usize + 2],
        &[0],
        &[SEQ_LEN],
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HD,
        0, // auto tile
    )
    .map(|_| ())
    .expect_err("last-page length past i32 must be rejected");
    assert!(err.to_string().contains("does not fit i32"), "{err}");

    let err = PrefillPagedPlan::new_with_cta_tile_q(
        ctx,
        &[0],
        PAGE_SIZE,
        0, // start_pos
        SEQ_LEN,
        NUM_Q_HEADS,
        0, // num_kv_heads
        HD,
        0, // auto tile
    )
    .map(|_| ())
    .expect_err("zero kv heads must be rejected");
    assert!(err.to_string().contains("num_kv_heads"), "{err}");
}
