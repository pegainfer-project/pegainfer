//! The generated windowed prefill against the incumbent windowed read it
//! replaces for the sliding family's prompt rows, on a pool and a ragged
//! plan built the way the serving path builds them.
//!
//! Three requests in one step: a prompt longer than the window, whose later
//! rows must drop keys the page-aligned residency still holds; a chunk that
//! continues a resident context; and a prompt shorter than a query tile.
//! Both reads take the same rows, pool and plan, so a disagreement is the
//! kernel's.
//!
//! Without a device it skips; `PEGAINFER_REQUIRE_GPU=1` turns that into a
//! failure.

#![cfg(feature = "gemma4")]

mod common;

use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::batch_prefill_paged_window_hd256_into;
use pegainfer_kernels::ops::gemma4_hd256_prefill_window_into;
use pegainfer_kernels::ops::gemma4_hd512_prefill_is_built;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

const HD: usize = 256;
const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 16;
const PAGE_SIZE: usize = 64;
const NUM_LAYERS: usize = 50;
const LAYER: usize = 7;
const WINDOW: usize = 1024;

/// One step of `requests` (resident context before the rows, rows) through
/// both windowed reads.
fn check(ctx: &DeviceContext, layers: usize, requests: &[(usize, usize)]) {
    let layout = PagedKvLayout::new(layers, NUM_KV_HEADS, HD, PAGE_SIZE);
    let mut page_indices: Vec<Vec<i32>> = Vec::new();
    let mut last_page_lens = Vec::new();
    let mut start_positions = Vec::new();
    let mut seq_lens = Vec::new();
    let mut next_page = 1i32;
    for &(start, rows) in requests {
        let kv_len = start + rows;
        let pages = kv_len.div_ceil(PAGE_SIZE);
        // Pages scattered over the pool, two apart, in request order.
        page_indices.push((0..pages).map(|p| next_page + 2 * p as i32).collect());
        next_page += 2 * pages as i32 + 1;
        last_page_lens.push((kv_len - 1) % PAGE_SIZE + 1);
        start_positions.push(start);
        seq_lens.push(rows);
    }
    let pool_pages = next_page as usize + 1;
    let pool = ctx
        .stream
        .clone_htod(&common::fill(0x5EED, pool_pages * layout.page_stride))
        .expect("pool upload");
    let total_rows: usize = seq_lens.iter().sum();
    let q = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&common::fill(0xC0_FFEE, total_rows * NUM_Q_HEADS * HD))
            .expect("q"),
        seq_len: total_rows,
        hidden_dim: NUM_Q_HEADS * HD,
    };
    let plan = PrefillPagedPlan::new_batch_with_cta_tile_q(
        ctx,
        &page_indices,
        &last_page_lens,
        &start_positions,
        &seq_lens,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HD,
        0,
    )
    .expect("prefill plan");
    let window_left = WINDOW - 1;

    let mut incumbent = HiddenStates::zeros(ctx, NUM_Q_HEADS * HD, total_rows).expect("out a");
    batch_prefill_paged_window_hd256_into(
        ctx,
        &q,
        &pool,
        &layout,
        LAYER,
        &plan,
        &mut incumbent,
        NUM_Q_HEADS,
        1.0,
        window_left as i32,
    )
    .expect("windowed prefill read");
    let mut replacement = HiddenStates::zeros(ctx, NUM_Q_HEADS * HD, total_rows).expect("out b");
    gemma4_hd256_prefill_window_into(
        ctx,
        &q,
        &pool,
        &layout,
        LAYER,
        &plan,
        &mut replacement,
        NUM_Q_HEADS,
        1.0,
        window_left,
    )
    .expect("generated windowed prefill");

    let a = incumbent.to_host(ctx).expect("D2H a");
    let b = replacement.to_host(ctx).expect("D2H b");
    let worst = common::worst_delta(&a, &b);
    let row = worst.1 / (NUM_Q_HEADS * HD);
    eprintln!(
        "{} requests over {layers} layers: worst |delta| {} at row {row} head {} lane {}; \
         incumbent {} replacement {}",
        requests.len(),
        worst.0,
        worst.1 % (NUM_Q_HEADS * HD) / HD,
        worst.1 % HD,
        a[worst.1],
        b[worst.1]
    );
    assert!(
        worst.0 <= 0.05,
        "{} requests: the two windowed reads disagree by {} at row {row}",
        requests.len(),
        worst.0
    );
}

#[test]
fn the_generated_windowed_prefill_matches_the_windowed_read() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    assert!(
        gemma4_hd512_prefill_is_built(),
        "this build carries the stub"
    );
    // A fresh prompt across the window, a continuation of a resident window,
    // a prompt under one tile.
    check(&ctx, NUM_LAYERS, &[(0, 1500), (1088, 700), (0, 40)]);
    // A step at the slot ceiling: sixteen requests, prompts across the
    // window, continuations, decoders holding a window and a page. A shallow
    // tower keeps the pool small.
    let full: Vec<(usize, usize)> = (0..16)
        .map(|i| match i % 4 {
            0 => (0, 1100 + i),
            1 => (0, 200 + 7 * i),
            2 => (1040, 1),
            _ => (64, 130),
        })
        .collect();
    check(&ctx, LAYER + 1, &full);
}
