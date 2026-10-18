//! The generated split-KV decode against the one it stands in for, on a pool,
//! plan and workspace built the way the serving path builds them, so a
//! disagreement is the kernel's. Plan arrays are exactly as long as the step
//! uses, then padded with values a kernel reading past it would choke on.
//!
//! Without a device it skips; `PEGAINFER_REQUIRE_GPU=1` turns that into a
//! failure.

#![cfg(feature = "gemma4")]

mod common;

use half::bf16;
use pegainfer_kernels::ops::Hd512DecodeMetadata;
use pegainfer_kernels::ops::gemma4_hd512_decode_split_kv_into;
use pegainfer_kernels::ops::gemma4_hd512_prefill_is_built;
use pegainfer_kernels::ops::paged_attention_batch_decode_split_kv_hd512_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

const HD: usize = 512;
const NUM_Q_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 4;
const PAGE_SIZE: usize = 64;
const NUM_LAYERS: usize = 10;
const LAYER: usize = 5;
/// The serving path's split chunk, so the tile indices count the same thing.
const CHUNK_TOKENS: usize = 256;
/// A value a walk that reads it cannot survive.
const HOSTILE: i32 = -10_000_000;

/// One decode step over `kv_lens` requests whose rows start at `row_offset`,
/// as the serving path lays it out: scattered pages, one slot per chunk,
/// padded slots masked, hostile values past every array's real length.
struct Step {
    q: HiddenStates,
    pool: cudarc::driver::CudaSlice<bf16>,
    layout: PagedKvLayout,
    page_indices: cudarc::driver::CudaSlice<i32>,
    page_indptr: cudarc::driver::CudaSlice<i32>,
    last_page_len: cudarc::driver::CudaSlice<i32>,
    request_indices: cudarc::driver::CudaSlice<i32>,
    kv_tile_indices: cudarc::driver::CudaSlice<i32>,
    chunk_size: cudarc::driver::CudaSlice<i32>,
    o_indptr: cudarc::driver::CudaSlice<i32>,
    valid_mask: cudarc::driver::CudaSlice<u8>,
    padded_slots: usize,
    rows: usize,
}

fn step(ctx: &DeviceContext, kv_lens: &[usize], row_offset: usize) -> Step {
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let batch = kv_lens.len();
    let rows = row_offset + batch;
    let total_pages: usize = kv_lens.iter().map(|n| n.div_ceil(PAGE_SIZE)).sum();
    let pool_pages = total_pages * 2 + 3;
    let pool_host = common::fill(0x51DE, pool_pages * layout.page_stride);
    let pool = ctx.stream.clone_htod(&pool_host).expect("pool upload");
    // Scatter each request's pages over the pool with a stride that keeps
    // them apart from the next request's.
    let (mut page_ids, mut page_indptr, mut last) = (Vec::new(), vec![0i32], Vec::new());
    let (mut req_idx, mut tile_idx, mut valid, mut o_indptr) =
        (Vec::new(), Vec::new(), Vec::new(), vec![0i32]);
    let mut next_page = 1usize;
    let mut slots_per = Vec::new();
    for (r, &n) in kv_lens.iter().enumerate() {
        let pages = n.div_ceil(PAGE_SIZE);
        for _ in 0..pages {
            page_ids.push(next_page as i32);
            next_page += 2;
        }
        page_indptr.push(page_ids.len() as i32);
        last.push(((n - 1) % PAGE_SIZE + 1) as i32);
        let chunks = n.div_ceil(CHUNK_TOKENS);
        slots_per.push(chunks);
        for t in 0..chunks {
            req_idx.push(r as i32);
            tile_idx.push(t as i32);
            valid.push(1u8);
        }
        o_indptr.push(req_idx.len() as i32);
    }
    let cap = *slots_per.iter().max().unwrap();
    let padded_slots = batch * cap;
    while req_idx.len() < padded_slots {
        req_idx.push(0);
        tile_idx.push(0);
        valid.push(0);
    }
    // Hostile tails past what the step uses.
    for i in 0..4 {
        page_indptr.push(page_indptr[batch] + HOSTILE * (i + 1));
        last.push(HOSTILE);
        o_indptr.push(o_indptr[batch] + HOSTILE * (i + 1));
    }
    for _ in 0..8 {
        req_idx.push(HOSTILE);
        tile_idx.push(HOSTILE);
        valid.push(1);
        page_ids.push(HOSTILE);
    }
    let up = |v: &[i32]| ctx.stream.clone_htod(v).expect("upload");
    Step {
        q: HiddenStates {
            data: ctx
                .stream
                .clone_htod(&common::fill(0xC0_FFEE, rows * NUM_Q_HEADS * HD))
                .expect("q"),
            seq_len: rows,
            hidden_dim: NUM_Q_HEADS * HD,
        },
        pool,
        layout,
        page_indices: up(&page_ids),
        page_indptr: up(&page_indptr),
        last_page_len: up(&last),
        request_indices: up(&req_idx),
        kv_tile_indices: up(&tile_idx),
        chunk_size: up(&[CHUNK_TOKENS as i32]),
        o_indptr: up(&o_indptr),
        valid_mask: ctx.stream.clone_htod(&valid).expect("valid mask"),
        padded_slots,
        rows,
    }
}

fn check(ctx: &DeviceContext, name: &str, kv_lens: &[usize], row_offset: usize) {
    let s = step(ctx, kv_lens, row_offset);
    let meta = Hd512DecodeMetadata::new(
        &s.page_indices,
        &s.page_indptr,
        &s.last_page_len,
        &s.request_indices,
        &s.kv_tile_indices,
        &s.chunk_size,
        CHUNK_TOKENS,
    );
    let mut out_a = HiddenStates::zeros(ctx, NUM_Q_HEADS * HD, s.rows).expect("out a");
    let mut out_b = HiddenStates::zeros(ctx, NUM_Q_HEADS * HD, s.rows).expect("out b");
    let mut tmp_v = ctx
        .stream
        .alloc_zeros::<bf16>(s.padded_slots * NUM_Q_HEADS * HD)
        .expect("tmp_v");
    let mut tmp_s = ctx
        .stream
        .alloc_zeros::<f32>(s.padded_slots * NUM_Q_HEADS)
        .expect("tmp_s");
    paged_attention_batch_decode_split_kv_hd512_into(
        ctx,
        &s.q,
        row_offset,
        &s.pool,
        &s.layout,
        LAYER,
        &meta,
        &s.o_indptr,
        &s.valid_mask,
        &mut tmp_v,
        &mut tmp_s,
        s.padded_slots,
        &mut out_a,
        NUM_Q_HEADS,
        1.0,
    )
    .expect("incumbent decode");
    ctx.stream.memset_zeros(&mut tmp_v).expect("clear tmp_v");
    ctx.stream.memset_zeros(&mut tmp_s).expect("clear tmp_s");
    gemma4_hd512_decode_split_kv_into(
        ctx,
        &s.q,
        row_offset,
        &s.pool,
        &s.layout,
        LAYER,
        &meta,
        &s.o_indptr,
        &s.valid_mask,
        &mut tmp_v,
        &mut tmp_s,
        s.padded_slots,
        &mut out_b,
        NUM_Q_HEADS,
        1.0,
    )
    .expect("generated decode");
    let a = out_a.to_host(ctx).expect("D2H a");
    let b = out_b.to_host(ctx).expect("D2H b");
    // Only the decode rows are written; the rows before them stay zero on
    // both sides, which the comparison covers as well.
    let (worst, at) = common::worst_delta(&a, &b);
    let row = at / (NUM_Q_HEADS * HD);
    let head = (at % (NUM_Q_HEADS * HD)) / HD;
    eprintln!(
        "{name}: worst |delta| {worst} at row {row} head {head} lane {}; incumbent {} replacement {}",
        at % HD,
        a[at],
        b[at]
    );
    assert!(
        worst <= 0.05,
        "{name}: the two decode kernels disagree by {worst} at row {row} head {head}"
    );
}

#[test]
fn the_generated_global_decode_matches_the_one_it_replaces() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    assert!(
        gemma4_hd512_prefill_is_built(),
        "this build carries the stub"
    );
    // One request over several chunks with a partial last page and a
    // partial last chunk; then a ragged batch whose decode rows sit after
    // prompt rows, so the row offset and the per-request slot ranges are
    // both exercised.
    check(&ctx, "single", &[500], 0);
    check(&ctx, "ragged", &[2045, 701, 1501], 37);
}
