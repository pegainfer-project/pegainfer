//! Device gate for csrc/shared/prefill_attention_hd512.cu.
//!
//! Manual gate: CI compiles this but never runs it. Run on a GPU box with
//! PEGAINFER_REQUIRE_GPU=1, which turns a missing device into a failure
//! rather than a skip. Traps are in their own binaries — __trap() poisons
//! the context for whatever runs next.

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::Hd512DecodeMetadata;
use pegainfer_kernels::ops::paged_attention_batch_decode_split_kv_hd512_into;
use pegainfer_kernels::ops::qk_norm_partial_rope_batched_decode_hd512_into;
use pegainfer_kernels::ops::qk_norm_partial_rope_paged_decode_hd512_into;
use pegainfer_kernels::ops::qk_norm_partial_rope_paged_prefill_hd512_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

const HD: usize = 512;
const ROTARY_DIM: usize = 128;
const HALF_ROTARY: usize = ROTARY_DIM / 2;
const EPS: f32 = 1e-6;
const NUM_Q_HEADS: usize = 16;
const NUM_KV_HEADS: usize = 1;
// Distinct inputs make a Q/K pointer swap observable.
const Q_INPUT: f32 = 1.0;
const K_INPUT: f32 = 3.0;
const Q_DIM: usize = NUM_Q_HEADS * HD;
const KV_DIM: usize = NUM_KV_HEADS * HD;
const SEQ_LEN: usize = 4;
const PAGE_SIZE: usize = 2;
// Exercise nonzero positions and more than one RoPE row.
const START_POS: usize = 1;
// Positions 1..=4 map to pages 3, 7, 7, 5; page 9 is unreferenced.
const PAGE_INDICES: [i32; 4] = [3, 7, 5, 9];
// Non-monotonic rows distinguish request positions from token indices.
const POSITIONS: [i32; 4] = [0, 1, 3, 1];
// Two layers make a wrong K-layer offset observable.
const NUM_LAYERS: usize = 2;
const PAGE_STRIDE: i64 = 8 * HD as i64;
// Covers page 7 while leaving unreferenced pages to check for stray writes.
const POOL_LEN: usize = 8 * PAGE_STRIDE as usize;

/// Constant rows make mean(x^2) exact in f32, so the kernel's 512-way
/// reduction collapses to a constant and never needs reproducing here.
fn inv_rms(x: f32) -> f32 {
    1.0f32 / (x * x + EPS).sqrt()
}

fn normed(x: f32, w: &[bf16], inv: f32, d: usize) -> f32 {
    bf16::from_f32(x * inv * w[d].to_f32()).to_f32()
}

/// Period 3, so every in-bounds row maps to a well-defined transform:
///   0 → ( 1, 0) identity     1 → ( 0, 1) swap-negate     2 → (-1, 0) negate
/// Unit coefficients keep every expectation exact in bf16.
fn rope_row(row: usize) -> (f32, f32) {
    match row % 3 {
        0 => (1.0, 0.0),
        1 => (0.0, 1.0),
        _ => (-1.0, 0.0),
    }
}

/// Laid out as the kernel indexes them: [pos * rotary_dim + d].
fn cos_sin_tables(ctx: &DeviceContext, rows: usize) -> (DeviceVec, DeviceVec) {
    let mut cos = Vec::with_capacity(rows * ROTARY_DIM);
    let mut sin = Vec::with_capacity(rows * ROTARY_DIM);
    for row in 0..rows {
        let (c, s) = rope_row(row);
        cos.extend(vec![bf16::from_f32(c); ROTARY_DIM]);
        sin.extend(vec![bf16::from_f32(s); ROTARY_DIM]);
    }
    (
        DeviceVec::from_host(ctx, &cos).expect("cos H2D"),
        DeviceVec::from_host(ctx, &sin).expect("sin H2D"),
    )
}

fn expected_prep(x: f32, w: &[bf16], inv: f32, d: usize, row: usize) -> f32 {
    if d < HALF_ROTARY {
        let lo = normed(x, w, inv, d);
        let hi = normed(x, w, inv, d + HALF_ROTARY);
        match row % 3 {
            0 => lo,
            1 => -hi,
            _ => -lo,
        }
    } else if d < ROTARY_DIM {
        let lo = normed(x, w, inv, d - HALF_ROTARY);
        let hi = normed(x, w, inv, d);
        match row % 3 {
            0 => hi,
            1 => lo,
            _ => -hi,
        }
    } else {
        normed(x, w, inv, d)
    }
}

fn expected_full(
    x: f32,
    w: &[bf16],
    inv: f32,
    dim: usize,
    row_of: impl Fn(usize) -> usize,
) -> Vec<f32> {
    let mut full = vec![0.0f32; dim * SEQ_LEN];
    for t in 0..SEQ_LEN {
        let row = row_of(t);
        for d in 0..dim {
            full[t * dim + d] = expected_prep(x, w, inv, d % HD, row);
        }
    }
    full
}

/// K and V blocks; everything else stays 0.0. The offsets are derived from
/// the layout, so oracle and kernel share no hand-picked raw offset. V is
/// the K=V fork: the same raw vector under the shared inv_rms, weightless
/// and un-rotated.
fn expected_pool(x: f32, w: &[bf16], inv: f32, layout: &PagedKvLayout, layer: usize) -> Vec<f32> {
    let mut exp = vec![0.0f32; POOL_LEN];
    let layer_offset = (layer * layout.layer_stride) as i64;
    let v_val = bf16::from_f32(x * inv).to_f32();
    for t in 0..SEQ_LEN {
        let pos = START_POS + t;
        let page = PAGE_INDICES[pos / PAGE_SIZE] as i64;
        for h in 0..NUM_KV_HEADS {
            let base = page * PAGE_STRIDE
                + layer_offset
                + (pos % PAGE_SIZE) as i64 * KV_DIM as i64
                + h as i64 * HD as i64;
            for d in 0..HD {
                exp[(base + d as i64) as usize] = expected_prep(x, w, inv, d, pos);
                exp[(base + layout.kv_block_len as i64 + d as i64) as usize] = v_val;
            }
        }
    }
    exp
}

fn assert_close(got: &[f32], expected: &[f32], what: &str) {
    assert_eq!(got.len(), expected.len());
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert!(
            (g - e).abs() < 0.02,
            "{what}[{i}]: got {g}, expected {e} (tolerance 0.02)"
        );
    }
}

/// Exact zero is the assertion, not sloppiness: it marks a slot the kernel
/// must never have written — unreferenced pages, the other layer, and the
/// slots outside the request's positions.
#[allow(clippy::float_cmp)]
fn assert_pool(got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len());
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        if e == 0.0 {
            assert_eq!(g, 0.0, "pool[{i}]: expected untouched, got {g}");
        } else {
            assert!(
                (g - e).abs() < 0.02,
                "pool[{i}]: got {g}, expected {e} (tolerance 0.02)"
            );
        }
    }
}

/// Starts at 1: w[0] = 0 would make dim 0 normalise to 0.0, which
/// assert_pool cannot tell from an untouched slot.
fn q_norm_weights() -> Vec<bf16> {
    (1..=HD).map(|d| bf16::from_f32(d as f32)).collect()
}

/// Negated relative to the Q side, so swapping the two weight pointers is
/// a sign flip in every slot. bf16 rounds some magnitudes, which does not
/// matter: oracle and kernel read back the same converted value.
fn k_norm_weights() -> Vec<bf16> {
    (1..=HD).map(|d| bf16::from_f32(-(d as f32))).collect()
}

#[test]
fn prefill_prep_matches_closed_form() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let qw = q_norm_weights();
    let kw = k_norm_weights();
    let layer = 1;
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    assert_eq!(
        layout.page_stride as i64, PAGE_STRIDE,
        "test geometry drift: layout page_stride must match PAGE_STRIDE"
    );
    let ones_q = vec![bf16::from_f32(Q_INPUT); Q_DIM * SEQ_LEN];
    let q = HiddenStates::from_host(ctx, &ones_q, Q_DIM, SEQ_LEN).expect("q H2D");
    let ones_k = vec![bf16::from_f32(K_INPUT); KV_DIM * SEQ_LEN];
    let k = HiddenStates::from_host(ctx, &ones_k, KV_DIM, SEQ_LEN).expect("k H2D");
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q_out alloc");

    let (cos_dev, sin_dev) = cos_sin_tables(ctx, 8);
    let qn = DeviceVec::from_host(ctx, &qw).expect("q_norm_weight H2D");
    let kn = DeviceVec::from_host(ctx, &kw).expect("k_norm_weight H2D");
    let pool: CudaSlice<bf16> = ctx.stream.alloc_zeros(POOL_LEN).expect("pool alloc");
    let page_indices: CudaSlice<i32> = ctx
        .stream
        .clone_htod(&PAGE_INDICES)
        .expect("page_indices H2D");

    qk_norm_partial_rope_paged_prefill_hd512_into(
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
        layer,
        &page_indices,
        0,
        START_POS,
        8, // cos_max_pos
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        ROTARY_DIM,
        EPS,
    )
    .expect("prefill prep launch");

    let qo = q_out.to_host(ctx).expect("q_out D2H");
    assert_close(
        &qo,
        &expected_full(Q_INPUT, &qw, inv_rms(Q_INPUT), Q_DIM, |t| START_POS + t),
        "prefill pairing/tail L1",
    );
    let pool_host: Vec<bf16> = ctx.stream.clone_dtoh(&pool).expect("pool D2H");
    let pool_f: Vec<f32> = pool_host.iter().map(|x| x.to_f32()).collect();
    assert_pool(
        &pool_f,
        &expected_pool(K_INPUT, &kw, inv_rms(K_INPUT), &layout, layer),
    );
}

#[test]
fn paged_decode_prep_matches_closed_form() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let qw = q_norm_weights();
    let kw = k_norm_weights();
    let layer = 1;
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let batch = 2usize;
    let positions: [i32; 2] = [3, 1];
    let pages_cat: [i32; 3] = [3, 7, 5];
    let indptr: [i32; 3] = [0, 2, 3];

    let ones_q = vec![bf16::from_f32(Q_INPUT); Q_DIM * batch];
    let q = HiddenStates::from_host(ctx, &ones_q, Q_DIM, batch).expect("q H2D");
    let ones_k = vec![bf16::from_f32(K_INPUT); KV_DIM * batch];
    let k = HiddenStates::from_host(ctx, &ones_k, KV_DIM, batch).expect("k H2D");
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, batch).expect("q_out alloc");

    let (cos_dev, sin_dev) = cos_sin_tables(ctx, 8);
    let qn = DeviceVec::from_host(ctx, &qw).expect("q_norm_weight H2D");
    let kn = DeviceVec::from_host(ctx, &kw).expect("k_norm_weight H2D");
    let pool: CudaSlice<bf16> = ctx.stream.alloc_zeros(POOL_LEN).expect("pool alloc");
    let pages_d: CudaSlice<i32> = ctx.stream.clone_htod(&pages_cat).expect("pages H2D");
    let indptr_d: CudaSlice<i32> = ctx.stream.clone_htod(&indptr).expect("indptr H2D");
    let origins_zero_d: CudaSlice<i32> = ctx.stream.clone_htod(&[0i32; 2]).expect("origins H2D");
    let positions_d: CudaSlice<i32> = ctx.stream.clone_htod(&positions).expect("positions H2D");

    qk_norm_partial_rope_paged_decode_hd512_into(
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
        layer,
        &pages_d,
        &indptr_d,
        &origins_zero_d,
        &positions_d,
        8, // cos_max_pos
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        ROTARY_DIM,
        EPS,
    )
    .expect("paged decode prep launch");

    let qo = q_out.to_host(ctx).expect("q_out D2H");
    let q_inv = inv_rms(Q_INPUT);
    let mut q_exp = vec![0.0f32; Q_DIM * batch];
    for (t, &pos) in positions.iter().enumerate() {
        for h in 0..NUM_Q_HEADS {
            for d in 0..HD {
                q_exp[t * Q_DIM + h * HD + d] = expected_prep(Q_INPUT, &qw, q_inv, d, pos as usize);
            }
        }
    }
    assert_close(&qo, &q_exp, "paged decode pairing/tail");

    let inv = inv_rms(K_INPUT);
    let v_val = bf16::from_f32(K_INPUT * inv).to_f32();
    let mut exp = vec![0.0f32; POOL_LEN];
    let layer_offset = (layer * layout.layer_stride) as i64;
    // (row, its page, its position): row 0's window [3, 7] covers pos 3 in
    // page 7; row 1's window [5] covers pos 1 in page 5.
    for (page, pos) in [(7i64, 3usize), (5, 1)] {
        for h in 0..NUM_KV_HEADS {
            let base = page * PAGE_STRIDE
                + layer_offset
                + (pos % PAGE_SIZE) as i64 * KV_DIM as i64
                + h as i64 * HD as i64;
            for d in 0..HD {
                exp[(base + d as i64) as usize] = expected_prep(K_INPUT, &kw, inv, d, pos);
                exp[(base + layout.kv_block_len as i64 + d as i64) as usize] = v_val;
            }
        }
    }
    let pool_host: Vec<bf16> = ctx.stream.clone_dtoh(&pool).expect("pool D2H");
    let pool_f: Vec<f32> = pool_host.iter().map(|x| x.to_f32()).collect();
    assert_pool(&pool_f, &exp);
}

#[test]
fn decode_prep_matches_closed_form() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let qw = q_norm_weights();
    let kw = k_norm_weights();
    let ones_q = vec![bf16::from_f32(Q_INPUT); Q_DIM * SEQ_LEN];
    let q = HiddenStates::from_host(ctx, &ones_q, Q_DIM, SEQ_LEN).expect("q H2D");
    let ones_k = vec![bf16::from_f32(K_INPUT); KV_DIM * SEQ_LEN];
    let mut k = HiddenStates::from_host(ctx, &ones_k, KV_DIM, SEQ_LEN).expect("k H2D");
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q_out alloc");

    let (cos_dev, sin_dev) = cos_sin_tables(ctx, 4);
    let qn = DeviceVec::from_host(ctx, &qw).expect("q_norm_weight H2D");
    let kn = DeviceVec::from_host(ctx, &kw).expect("k_norm_weight H2D");
    let positions: CudaSlice<i32> = ctx.stream.clone_htod(&POSITIONS).expect("positions H2D");

    qk_norm_partial_rope_batched_decode_hd512_into(
        ctx,
        &q,
        &mut q_out,
        &mut k,
        &qn,
        &kn,
        &cos_dev,
        &sin_dev,
        &positions,
        4, // cos_max_pos
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        ROTARY_DIM,
        EPS,
    )
    .expect("decode prep launch");

    let qo = q_out.to_host(ctx).expect("q_out D2H");
    assert_close(
        &qo,
        &expected_full(Q_INPUT, &qw, inv_rms(Q_INPUT), Q_DIM, |t| {
            POSITIONS[t] as usize
        }),
        "decode pairing/tail",
    );
    let k_host = k.to_host(ctx).expect("k D2H");
    assert_close(
        &k_host,
        &expected_full(K_INPUT, &kw, inv_rms(K_INPUT), KV_DIM, |t| {
            POSITIONS[t] as usize
        }),
        "decode pairing/tail",
    );
}

#[test]
fn rejects_bad_rotary_dim() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    // Zeroed buffers suffice: the launcher must reject before touching any
    // device memory. Tables are sized for the worst case checked here
    // (8 x 1024), so the wrapper's table checks pass and both values
    // actually reach the launcher.
    let q = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q alloc");
    let k = HiddenStates::zeros(ctx, KV_DIM, SEQ_LEN).expect("k alloc");
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q_out alloc");
    let mut k_mut = HiddenStates::zeros(ctx, KV_DIM, SEQ_LEN).expect("k_mut alloc");
    let pool: CudaSlice<bf16> = ctx.stream.alloc_zeros(POOL_LEN).expect("pool alloc");
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let cos_dev = DeviceVec::zeros(ctx, 8 * 1024).expect("cos alloc");
    let sin_dev = DeviceVec::zeros(ctx, 8 * 1024).expect("sin alloc");
    let qn = DeviceVec::zeros(ctx, HD).expect("qn alloc");
    let kn = DeviceVec::zeros(ctx, HD).expect("kn alloc");
    let page_indices: CudaSlice<i32> = ctx
        .stream
        .clone_htod(&PAGE_INDICES)
        .expect("page_indices H2D");
    let positions: CudaSlice<i32> = ctx.stream.clone_htod(&POSITIONS).expect("positions H2D");

    // 127: odd — index 126 would never be written. 1024: wider than the
    // head — smem and the output slices would walk past 512. Both must be
    // rejected, not silently accepted.
    for bad in [127, 1024] {
        let prefill_err = qk_norm_partial_rope_paged_prefill_hd512_into(
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
            8, // cos_max_pos
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            bad,
            EPS,
        );
        assert!(
            prefill_err.is_err(),
            "prefill: rotary_dim={bad} must be rejected"
        );
        let decode_err = qk_norm_partial_rope_batched_decode_hd512_into(
            ctx,
            &q,
            &mut q_out,
            &mut k_mut,
            &qn,
            &kn,
            &cos_dev,
            &sin_dev,
            &positions,
            8, // cos_max_pos
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            bad,
            EPS,
        );
        assert!(
            decode_err.is_err(),
            "decode: rotary_dim={bad} must be rejected"
        );
    }
}

#[test]
fn prefill_rejects_position_beyond_cos_table() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    // Zeroed buffers suffice: rejected on the host, before any launch.
    // With cos_max_pos 4, start_pos 1 + seq_len 4 reaches row 5 — past the
    // tables; the page check passes (8 >= 5 slots), so the position check
    // is the one that fires.
    let q = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q alloc");
    let k = HiddenStates::zeros(ctx, KV_DIM, SEQ_LEN).expect("k alloc");
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q_out alloc");
    let pool: CudaSlice<bf16> = ctx.stream.alloc_zeros(POOL_LEN).expect("pool alloc");
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let cos_dev = DeviceVec::zeros(ctx, 4 * ROTARY_DIM).expect("cos alloc");
    let sin_dev = DeviceVec::zeros(ctx, 4 * ROTARY_DIM).expect("sin alloc");
    let qn = DeviceVec::zeros(ctx, HD).expect("qn alloc");
    let kn = DeviceVec::zeros(ctx, HD).expect("kn alloc");
    let page_indices: CudaSlice<i32> = ctx
        .stream
        .clone_htod(&PAGE_INDICES)
        .expect("page_indices H2D");

    let err = qk_norm_partial_rope_paged_prefill_hd512_into(
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
        1, // start_pos
        4, // cos_max_pos
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        ROTARY_DIM,
        EPS,
    );
    assert!(
        err.is_err(),
        "prefill start_pos + seq_len beyond cos_max_pos must be rejected on the host"
    );
}

#[test]
fn prefill_rejects_undersized_kv_pool() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    // The wrapper validates pool CAPACITY, not just page-table coverage:
    // the kernel would otherwise write 512 K elements per token into a
    // 1-element pool. Two checks guard this, and the cases are chosen so
    // each is reachable on its own — otherwise deleting one would leave
    // the test green because the other still fires: 0 is a multiple of
    // page_stride, so only the num_pages >= 1 check can reject it;
    // page_stride + 1 is over a page (num_pages = 1) but not a multiple,
    // so only divisibility can.
    let q = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q alloc");
    let k = HiddenStates::zeros(ctx, KV_DIM, SEQ_LEN).expect("k alloc");
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q_out alloc");
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let cos_dev = DeviceVec::zeros(ctx, 8 * ROTARY_DIM).expect("cos alloc");
    let sin_dev = DeviceVec::zeros(ctx, 8 * ROTARY_DIM).expect("sin alloc");
    let qn = DeviceVec::zeros(ctx, HD).expect("qn alloc");
    let kn = DeviceVec::zeros(ctx, HD).expect("kn alloc");
    let page_indices: CudaSlice<i32> = ctx
        .stream
        .clone_htod(&PAGE_INDICES)
        .expect("page_indices H2D");

    for bad_len in [0usize, PAGE_STRIDE as usize + 1] {
        let pool: CudaSlice<bf16> = ctx.stream.alloc_zeros(bad_len).expect("pool alloc");
        let err = qk_norm_partial_rope_paged_prefill_hd512_into(
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
            8, // cos_max_pos
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            ROTARY_DIM,
            EPS,
        );
        let Err(e) = err else {
            panic!("kv_pool len {bad_len} must be rejected on the host");
        };
        // 0 IS a multiple of page_stride: without the num_pages check the
        // call would still fail — but from the kernel's page trap, which
        // poisons the context. Pin the host message so that regression
        // cannot pass as "still an error".
        if bad_len == 0 {
            let msg = format!("{e:#}");
            assert!(
                msg.contains("no whole page"),
                "len 0 must be rejected on the host, not by the device trap; got: {msg}"
            );
        }
    }
}

/// The row-offset suffix contract for the hd512 decode prep: the prefix row
/// of `q_out` stays untouched, and the suffix outputs and pool writes equal
/// a zero-offset run over the same two suffix rows.
#[test]
fn decode_prep_row_offset_serves_only_the_suffix() {
    const SENTINEL: f32 = 777.0;
    const FILLER: f32 = 9.25;
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let qw = q_norm_weights();
    let kw = k_norm_weights();
    let layer = 1;
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let batch = 2usize;
    let positions: [i32; 2] = [3, 1];
    let pages_cat: [i32; 3] = [3, 7, 5];
    let indptr: [i32; 3] = [0, 2, 3];

    // Per-row distinct suffix bytes shared between the arms; the offset
    // arm's prefix row is filler the prep must neither read nor overwrite.
    let suffix = |base: f32, dim: usize| -> Vec<bf16> {
        (0..batch)
            .flat_map(|t| vec![bf16::from_f32(base + t as f32); dim])
            .collect()
    };
    let with_prefix = |sfx: &[bf16], dim: usize| -> Vec<bf16> {
        let mut host = vec![bf16::from_f32(FILLER); dim];
        host.extend_from_slice(sfx);
        host
    };
    let (qs, ks) = (suffix(Q_INPUT, Q_DIM), suffix(K_INPUT, KV_DIM));
    let (cos_dev, sin_dev) = cos_sin_tables(ctx, 8);
    let qn = DeviceVec::from_host(ctx, &qw).expect("q_norm_weight H2D");
    let kn = DeviceVec::from_host(ctx, &kw).expect("k_norm_weight H2D");
    let pages_d: CudaSlice<i32> = ctx.stream.clone_htod(&pages_cat).expect("pages H2D");
    let indptr_d: CudaSlice<i32> = ctx.stream.clone_htod(&indptr).expect("indptr H2D");
    let origins_zero_d: CudaSlice<i32> = ctx.stream.clone_htod(&[0i32; 2]).expect("origins H2D");
    let positions_d: CudaSlice<i32> = ctx.stream.clone_htod(&positions).expect("positions H2D");

    let run = |offset: usize, q_host: &[bf16], k_host: &[bf16]| {
        let rows = offset + batch;
        let q = HiddenStates::from_host(ctx, q_host, Q_DIM, rows).expect("q H2D");
        let k = HiddenStates::from_host(ctx, k_host, KV_DIM, rows).expect("k H2D");
        let sentinel = vec![bf16::from_f32(SENTINEL); Q_DIM * rows];
        let mut q_out = HiddenStates::from_host(ctx, &sentinel, Q_DIM, rows).expect("q_out H2D");
        let pool: CudaSlice<bf16> = ctx.stream.alloc_zeros(POOL_LEN).expect("pool alloc");
        qk_norm_partial_rope_paged_decode_hd512_into(
            ctx,
            &q,
            &k,
            &mut q_out,
            offset,
            &pool,
            &layout,
            &qn,
            &kn,
            &cos_dev,
            &sin_dev,
            layer,
            &pages_d,
            &indptr_d,
            &origins_zero_d,
            &positions_d,
            8,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            ROTARY_DIM,
            EPS,
        )
        .expect("decode prep launch");
        let out: Vec<u32> = q_out
            .to_host(ctx)
            .expect("q_out D2H")
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let pool_host: Vec<bf16> = ctx.stream.clone_dtoh(&pool).expect("pool D2H");
        let pool_bits: Vec<u16> = pool_host.iter().map(|x| x.to_bits()).collect();
        (out, pool_bits)
    };

    let (out_a, pool_a) = run(1, &with_prefix(&qs, Q_DIM), &with_prefix(&ks, KV_DIM));
    let (out_b, pool_b) = run(0, &qs, &ks);

    let sentinel_bits = bf16::from_f32(SENTINEL).to_f32().to_bits();
    assert!(
        out_a[..Q_DIM].iter().all(|&b| b == sentinel_bits),
        "the prefix row of q_out must stay untouched"
    );
    assert_eq!(
        out_a[Q_DIM..],
        out_b[..],
        "suffix q_out rows must match the zero-offset run bit for bit"
    );
    assert_eq!(
        pool_a, pool_b,
        "pool writes must match the zero-offset run bit for bit"
    );
}

/// The row-offset suffix contract for the split-KV hd512 read: over a pool
/// both arms share, the offset arm's prefix output row stays untouched and
/// its suffix rows equal a zero-offset read of the same two requests. Eight
/// query heads over the single KV head keep the GQA group dispatchable.
#[test]
fn split_read_row_offset_serves_only_the_suffix() {
    const SENTINEL: f32 = 777.0;
    const FILLER: f32 = 9.25;
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let layer = 1;
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let batch = 2usize;
    let read_heads = 8usize;
    let read_dim = read_heads * HD;

    // Populate the pool through the zero-offset prep: request 0 writes
    // position 3 into page 7, request 1 writes position 1 into page 5.
    let qw = q_norm_weights();
    let kw = k_norm_weights();
    let prep_q = HiddenStates::from_host(
        ctx,
        &vec![bf16::from_f32(Q_INPUT); Q_DIM * batch],
        Q_DIM,
        batch,
    )
    .expect("prep q H2D");
    let prep_k = HiddenStates::from_host(
        ctx,
        &vec![bf16::from_f32(K_INPUT); KV_DIM * batch],
        KV_DIM,
        batch,
    )
    .expect("prep k H2D");
    let mut prep_q_out = HiddenStates::zeros(ctx, Q_DIM, batch).expect("prep q_out alloc");
    let (cos_dev, sin_dev) = cos_sin_tables(ctx, 8);
    let qn = DeviceVec::from_host(ctx, &qw).expect("q_norm_weight H2D");
    let kn = DeviceVec::from_host(ctx, &kw).expect("k_norm_weight H2D");
    let pool: CudaSlice<bf16> = ctx.stream.alloc_zeros(POOL_LEN).expect("pool alloc");
    let positions: [i32; 2] = [3, 1];
    let pages_cat: [i32; 3] = [3, 7, 5];
    let indptr: [i32; 3] = [0, 2, 3];
    let pages_d: CudaSlice<i32> = ctx.stream.clone_htod(&pages_cat).expect("pages H2D");
    let indptr_d: CudaSlice<i32> = ctx.stream.clone_htod(&indptr).expect("indptr H2D");
    let origins_zero_d: CudaSlice<i32> = ctx.stream.clone_htod(&[0i32; 2]).expect("origins H2D");
    let positions_d: CudaSlice<i32> = ctx.stream.clone_htod(&positions).expect("positions H2D");
    qk_norm_partial_rope_paged_decode_hd512_into(
        ctx,
        &prep_q,
        &prep_k,
        &mut prep_q_out,
        0,
        &pool,
        &layout,
        &qn,
        &kn,
        &cos_dev,
        &sin_dev,
        layer,
        &pages_d,
        &indptr_d,
        &origins_zero_d,
        &positions_d,
        8,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        ROTARY_DIM,
        EPS,
    )
    .expect("pool populate");

    // Read metadata: kv_len 4 (pages [3, 7], last 2) and kv_len 2
    // (page [5], last 2); one chunk per request.
    let last: [i32; 2] = [2, 2];
    let req: [i32; 2] = [0, 1];
    let tile: [i32; 2] = [0, 0];
    let chunk: [i32; 1] = [8];
    let o_indptr: [i32; 3] = [0, 1, 2];
    let mask: [u8; 2] = [1, 1];
    let last_d: CudaSlice<i32> = ctx.stream.clone_htod(&last).expect("last H2D");
    let req_d: CudaSlice<i32> = ctx.stream.clone_htod(&req).expect("req H2D");
    let tile_d: CudaSlice<i32> = ctx.stream.clone_htod(&tile).expect("tile H2D");
    let chunk_d: CudaSlice<i32> = ctx.stream.clone_htod(&chunk).expect("chunk H2D");
    let o_indptr_d: CudaSlice<i32> = ctx.stream.clone_htod(&o_indptr).expect("o_indptr H2D");
    let mask_d: CudaSlice<u8> = ctx.stream.clone_htod(&mask).expect("mask H2D");

    let suffix_q: Vec<bf16> = (0..batch)
        .flat_map(|t| vec![bf16::from_f32(0.5 + t as f32); read_dim])
        .collect();
    let run = |offset: usize, q_host: &[bf16]| {
        let rows = offset + batch;
        let q = HiddenStates::from_host(ctx, q_host, read_dim, rows).expect("read q H2D");
        let sentinel = vec![bf16::from_f32(SENTINEL); read_dim * rows];
        let mut out = HiddenStates::from_host(ctx, &sentinel, read_dim, rows).expect("out H2D");
        let mut tmp_v: CudaSlice<bf16> = ctx.stream.alloc_zeros(2 * read_dim).expect("tmp_v alloc");
        let mut tmp_s: CudaSlice<f32> =
            ctx.stream.alloc_zeros(2 * read_heads).expect("tmp_s alloc");
        let meta =
            Hd512DecodeMetadata::new(&pages_d, &indptr_d, &last_d, &req_d, &tile_d, &chunk_d);
        paged_attention_batch_decode_split_kv_hd512_into(
            ctx,
            &q,
            offset,
            &pool,
            &layout,
            layer,
            &meta,
            &o_indptr_d,
            &mask_d,
            &mut tmp_v,
            &mut tmp_s,
            2,
            &mut out,
            read_heads,
            1.0,
        )
        .expect("split read launch");
        let bits: Vec<u32> = out
            .to_host(ctx)
            .expect("out D2H")
            .iter()
            .map(|v| v.to_bits())
            .collect();
        bits
    };

    let mut with_prefix = vec![bf16::from_f32(FILLER); read_dim];
    with_prefix.extend_from_slice(&suffix_q);
    let out_a = run(1, &with_prefix);
    let out_b = run(0, &suffix_q);

    let sentinel_bits = bf16::from_f32(SENTINEL).to_f32().to_bits();
    assert!(
        out_a[..read_dim].iter().all(|&b| b == sentinel_bits),
        "the prefix output row must stay untouched"
    );
    assert_eq!(
        out_a[read_dim..],
        out_b[..],
        "suffix outputs must match the zero-offset read bit for bit"
    );
}

/// The row and page-table windows of the hd512 prefill prep: the offset arm
/// carries a filler prefix row and a junk leading table entry the kernel
/// must neither read nor dereference; its suffix outputs and pool writes
/// must equal a zero-offset run bit for bit, and the prefix row of `q_out`
/// stays untouched.
#[test]
fn prefill_prep_row_offset_serves_only_the_suffix() {
    const SENTINEL: f32 = 777.0;
    const FILLER: f32 = 9.25;
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let qw = q_norm_weights();
    let kw = k_norm_weights();
    let layer = 1;
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let (cos_dev, sin_dev) = cos_sin_tables(ctx, 8);
    let qn = DeviceVec::from_host(ctx, &qw).expect("q_norm_weight H2D");
    let kn = DeviceVec::from_host(ctx, &kw).expect("k_norm_weight H2D");

    // Per-row distinct suffix bytes shared between the arms.
    let suffix = |base: f32, dim: usize| -> Vec<bf16> {
        (0..SEQ_LEN)
            .flat_map(|t| vec![bf16::from_f32(base + t as f32); dim])
            .collect()
    };
    let with_prefix = |sfx: &[bf16], dim: usize| -> Vec<bf16> {
        let mut host = vec![bf16::from_f32(FILLER); dim];
        host.extend_from_slice(sfx);
        host
    };
    let (qs, ks) = (suffix(Q_INPUT, Q_DIM), suffix(K_INPUT, KV_DIM));

    let run =
        |offset: usize, q_host: &[bf16], k_host: &[bf16], table: &[i32], pages_offset: usize| {
            let rows = offset + SEQ_LEN;
            let q = HiddenStates::from_host(ctx, q_host, Q_DIM, rows).expect("q H2D");
            let k = HiddenStates::from_host(ctx, k_host, KV_DIM, rows).expect("k H2D");
            let sentinel = vec![bf16::from_f32(SENTINEL); Q_DIM * rows];
            let mut q_out =
                HiddenStates::from_host(ctx, &sentinel, Q_DIM, rows).expect("q_out H2D");
            let pool: CudaSlice<bf16> = ctx.stream.alloc_zeros(POOL_LEN).expect("pool alloc");
            let page_indices: CudaSlice<i32> = ctx.stream.clone_htod(table).expect("pages H2D");
            qk_norm_partial_rope_paged_prefill_hd512_into(
                ctx,
                &q,
                &k,
                &mut q_out,
                offset,
                &pool,
                &layout,
                &qn,
                &kn,
                &cos_dev,
                &sin_dev,
                layer,
                &page_indices,
                pages_offset,
                START_POS,
                8,
                NUM_Q_HEADS,
                NUM_KV_HEADS,
                ROTARY_DIM,
                EPS,
            )
            .expect("prefill prep launch");
            let out: Vec<u32> = q_out
                .to_host(ctx)
                .expect("q_out D2H")
                .iter()
                .map(|x| x.to_bits())
                .collect();
            let pool_host: Vec<bf16> = ctx.stream.clone_dtoh(&pool).expect("pool D2H");
            let pool_bits: Vec<u16> = pool_host.iter().map(|x| x.to_bits()).collect();
            (out, pool_bits)
        };

    // The junk entry is a valid, unreferenced page: a wrong dereference
    // lands visibly in the pool comparison instead of out of bounds.
    let mut junk_table = vec![6i32];
    junk_table.extend_from_slice(&PAGE_INDICES);
    let (out_a, pool_a) = run(
        1,
        &with_prefix(&qs, Q_DIM),
        &with_prefix(&ks, KV_DIM),
        &junk_table,
        1,
    );
    let (out_b, pool_b) = run(0, &qs, &ks, &PAGE_INDICES, 0);

    let sentinel_bits = bf16::from_f32(SENTINEL).to_f32().to_bits();
    assert!(
        out_a[..Q_DIM].iter().all(|&b| b == sentinel_bits),
        "the prefix row of q_out must stay untouched"
    );
    assert_eq!(
        out_a[Q_DIM..],
        out_b[..],
        "suffix q_out rows must match the zero-offset run bit for bit"
    );
    assert_eq!(
        pool_a, pool_b,
        "pool writes must match the zero-offset run bit for bit"
    );
}
