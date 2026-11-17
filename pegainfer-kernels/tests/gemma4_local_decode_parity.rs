//! The generated windowed decode against the windowed prefill read it
//! replaces for the sliding family's decode rows, on a pool and plan built
//! the way the serving path builds them.
//!
//! A decode row's resident pages hold up to a page more than the window,
//! since pages release whole; both reads have to mask those keys the same
//! way. The two entries are handed the same rows, the same pool and the
//! same page table, so a disagreement is the kernel's, not the plumbing's.
//! Plan arrays are exactly as long as the step uses, then padded with values
//! a kernel reading past the step would choke on.
//!
//! Without a device it skips; `PEGAINFER_REQUIRE_GPU=1` turns that into a
//! failure.

#![cfg(feature = "gemma4")]

mod common;

use half::bf16;
use pegainfer_kernels::ops::Hd512DecodeMetadata;
use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::batch_prefill_paged_window_hd256_into;
use pegainfer_kernels::ops::gemma4_hd256_decode_window_into;
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
/// The serving path's chunk for this family.
const CHUNK_TOKENS: usize = 64;
const HOSTILE: i32 = -10_000_000;

/// One decode row over `resident` tokens of resident pages, scattered over a
/// pool with junk everywhere else.
fn check(ctx: &DeviceContext, resident: usize) {
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let pages = resident.div_ceil(PAGE_SIZE);
    let pool_pages = pages * 3;
    let page_indices: Vec<i32> = (0..pages).map(|p| (p * 3 + 1) as i32).collect();
    let pool = ctx
        .stream
        .clone_htod(&common::fill(
            0x51DE ^ resident as u64,
            pool_pages * layout.page_stride,
        ))
        .expect("pool upload");
    let q = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&common::fill(0xC0_FFEE, NUM_Q_HEADS * HD))
            .expect("q"),
        seq_len: 1,
        hidden_dim: NUM_Q_HEADS * HD,
    };
    let last_page_len = (resident - 1) % PAGE_SIZE + 1;
    let window_left = WINDOW - 1;

    // The incumbent: the windowed prefill read over one query row at the
    // window's end.
    let plan = PrefillPagedPlan::new_with_cta_tile_q(
        ctx,
        &page_indices,
        last_page_len,
        resident - 1,
        1,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HD,
        0,
    )
    .expect("prefill plan");
    let mut incumbent = HiddenStates::zeros(ctx, NUM_Q_HEADS * HD, 1).expect("out a");
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

    // The replacement: the split plan over the same pages, one slot per
    // chunk, hostile past the step.
    let chunks = resident.div_ceil(CHUNK_TOKENS);
    let mut ids = page_indices;
    let mut page_indptr = vec![0, pages as i32];
    let mut last = vec![last_page_len as i32];
    let mut req: Vec<i32> = vec![0; chunks];
    let mut tile: Vec<i32> = (0..chunks as i32).collect();
    let mut valid = vec![1u8; chunks];
    let mut o_indptr = vec![0, chunks as i32];
    for i in 0..4 {
        page_indptr.push(pages as i32 + HOSTILE * (i + 1));
        last.push(HOSTILE);
        o_indptr.push(chunks as i32 + HOSTILE * (i + 1));
    }
    for _ in 0..8 {
        req.push(HOSTILE);
        tile.push(HOSTILE);
        valid.push(1);
        ids.push(HOSTILE);
    }
    let up = |v: &[i32]| ctx.stream.clone_htod(v).expect("upload");
    let (ids_d, indptr_d, last_d, req_d, tile_d, o_indptr_d) = (
        up(&ids),
        up(&page_indptr),
        up(&last),
        up(&req),
        up(&tile),
        up(&o_indptr),
    );
    let chunk_d = up(&[CHUNK_TOKENS as i32]);
    let valid_d = ctx.stream.clone_htod(&valid).expect("valid mask");
    let meta = Hd512DecodeMetadata::new(
        &ids_d,
        &indptr_d,
        &last_d,
        &req_d,
        &tile_d,
        &chunk_d,
        CHUNK_TOKENS,
    );
    let mut tmp_v = ctx
        .stream
        .alloc_zeros::<bf16>(chunks * NUM_Q_HEADS * HD)
        .expect("tmp_v");
    let mut tmp_s = ctx
        .stream
        .alloc_zeros::<f32>(chunks * NUM_Q_HEADS)
        .expect("tmp_s");
    let mut replacement = HiddenStates::zeros(ctx, NUM_Q_HEADS * HD, 1).expect("out b");
    gemma4_hd256_decode_window_into(
        ctx,
        &q,
        0,
        &pool,
        &layout,
        LAYER,
        &meta,
        &o_indptr_d,
        &valid_d,
        &mut tmp_v,
        &mut tmp_s,
        chunks,
        &mut replacement,
        NUM_Q_HEADS,
        1.0,
        window_left,
    )
    .expect("generated windowed decode");

    let a = incumbent.to_host(ctx).expect("D2H a");
    let b = replacement.to_host(ctx).expect("D2H b");
    let worst = common::worst_delta(&a, &b);
    eprintln!(
        "resident {resident}: worst |delta| {} at head {} lane {}; incumbent {} replacement {}",
        worst.0,
        worst.1 / HD,
        worst.1 % HD,
        a[worst.1],
        b[worst.1]
    );
    assert!(
        worst.0 <= 0.05,
        "resident {resident}: the two windowed reads disagree by {} at head {}",
        worst.0,
        worst.1 / HD
    );
}

#[test]
fn the_generated_windowed_decode_matches_the_windowed_prefill_read() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    assert!(
        gemma4_hd512_prefill_is_built(),
        "this build carries the stub"
    );
    // A window plus a page, where the oldest page's keys must be masked; a
    // window exactly; less than a window; a single token.
    for resident in [WINDOW + PAGE_SIZE, WINDOW, 528, 1] {
        check(&ctx, resident);
    }
}
