//! The generated global-attention prefill against the one it stands in for,
//! on a pool and plan built the way the serving path builds them. Needs
//! neither weights nor a tokenizer, so it names which entry is wrong where
//! the serving gate only says the logits diverged; both entries are handed
//! the same plan, the same pool and the same query rows.
//!
//! Without a device it skips; `PEGAINFER_REQUIRE_GPU=1` turns that into a
//! failure.

#![cfg(feature = "gemma4")]

mod common;

use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::batch_prefill_paged_hd512_into;
use pegainfer_kernels::ops::gemma4_hd512_prefill_is_built;
use pegainfer_kernels::ops::gemma4_hd512_prefill_varlen_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

// Gemma 4's global family: 32 query heads over 4 key/value heads at head dim
// 512, paged at the attention key block.
const HD: usize = 512;
const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 4;
const PAGE_SIZE: usize = 64;
const NUM_LAYERS: usize = 10;
// The layer under test is not the first, so a wrong layer offset shows up.
const LAYER: usize = 5;
// Long enough to walk several key blocks and to leave the last page partial.
const SEQ_LEN: usize = 500;

/// One step of `requests` (resident context before the rows, rows) through
/// both prefills.
fn check(ctx: &DeviceContext, requests: &[(usize, usize)]) {
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    // Page ids are scattered and the pool holds more pages than the requests
    // use, so a stride mistake reads somebody else's rows rather than its
    // own neighbour's.
    let mut page_indices: Vec<Vec<i32>> = Vec::new();
    let mut last_page_lens = Vec::new();
    let mut start_positions = Vec::new();
    let mut seq_lens = Vec::new();
    let mut next_page = 1i32;
    for &(start, rows) in requests {
        let kv_len = start + rows;
        let pages = kv_len.div_ceil(PAGE_SIZE);
        page_indices.push((0..pages).map(|p| next_page + 3 * p as i32).collect());
        next_page += 3 * pages as i32 + 1;
        last_page_lens.push((kv_len - 1) % PAGE_SIZE + 1);
        start_positions.push(start);
        seq_lens.push(rows);
    }
    let pool_pages = next_page as usize + 1;
    let pool_host = common::fill(0x51DE, pool_pages * layout.page_stride);
    let pool = ctx.stream.clone_htod(&pool_host).expect("pool upload");
    let total_rows: usize = seq_lens.iter().sum();
    let q_host = common::fill(0xC0_FFEE, total_rows * NUM_Q_HEADS * HD);
    let q = HiddenStates {
        data: ctx.stream.clone_htod(&q_host).expect("q upload"),
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

    let mut incumbent = HiddenStates::zeros(ctx, NUM_Q_HEADS * HD, total_rows).expect("out a");
    let mut replacement = HiddenStates::zeros(ctx, NUM_Q_HEADS * HD, total_rows).expect("out b");
    batch_prefill_paged_hd512_into(
        ctx,
        &q,
        &pool,
        &layout,
        LAYER,
        &plan,
        &mut incumbent,
        NUM_Q_HEADS,
        1.0,
    )
    .expect("incumbent prefill");
    gemma4_hd512_prefill_varlen_into(
        ctx,
        &q,
        &pool,
        &layout,
        LAYER,
        &plan,
        &mut replacement,
        NUM_Q_HEADS,
        1.0,
    )
    .expect("generated prefill");

    let a = incumbent.to_host(ctx).expect("incumbent D2H");
    let b = replacement.to_host(ctx).expect("replacement D2H");
    assert_eq!(a.len(), b.len());
    let (worst, worst_at) = common::worst_delta(&a, &b);
    let row = worst_at / (NUM_Q_HEADS * HD);
    let head = (worst_at % (NUM_Q_HEADS * HD)) / HD;
    eprintln!(
        "{} requests: worst |delta| {worst} at row {row} head {head} lane {}; \
         incumbent {} replacement {}",
        requests.len(),
        worst_at % HD,
        a[worst_at],
        b[worst_at]
    );
    assert!(
        worst <= 0.05,
        "{} requests: the two kernels disagree by {worst} at row {row} head {head}",
        requests.len()
    );
}

#[test]
fn the_generated_global_prefill_matches_the_one_it_replaces() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    assert!(
        gemma4_hd512_prefill_is_built(),
        "this build carries the stub, so there is nothing to compare against"
    );
    check(&ctx, &[(0, SEQ_LEN)]);
    // A step at the slot ceiling: sixteen requests, fresh prompts,
    // continuations over a resident context and single decode rows.
    let full: Vec<(usize, usize)> = (0..16)
        .map(|i| match i % 4 {
            0 => (0, SEQ_LEN + 3 * i),
            1 => (0, 130 + 7 * i),
            2 => (700 + 5 * i, 1),
            _ => (300, 90),
        })
        .collect();
    check(&ctx, &full);
}
