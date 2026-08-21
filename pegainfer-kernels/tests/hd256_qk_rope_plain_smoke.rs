//! Device gate for csrc/shared/prefill_attention_hd256_plain.cu.
//!
//! Manual gate: CI compiles this but never runs it. Run on a GPU box with
//! PEGAINFER_REQUIRE_GPU=1, which turns a missing device into a failure
//! rather than a skip.

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::qk_norm_rope_prefill_hd256_plain_into;
use pegainfer_kernels::ops::qkv_norm_rope_paged_decode_hd256_plain_into;
use pegainfer_kernels::ops::qkv_norm_rope_paged_prefill_hd256_plain_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;

const HD: usize = 256;
const EPS: f32 = 1e-6;
// The Gemma 4 12B local-layer geometry: 16 query heads over 8 KV heads.
const NUM_Q_HEADS: usize = 16;
const NUM_KV_HEADS: usize = 8;
// The norm collapses a constant row to sign(x) * w[d] (x / sqrt(x^2 + eps)
// is ±1 up to eps), so input magnitude cannot tell heads or tokens apart
// after normalisation — only sign survives. The per-(head, token) sign
// patterns below carry source identity through the norm, so a source-stride
// error (always reading head 0 or token 0, the qwen35 gated 2x head stride,
// an off-by-one head) lands a wrong sign somewhere. K uses the leading
// 8 entries; the pattern is aperiodic under h -> 2h, h % 8 and h ± 1.
const Q_BASE: f32 = 1.0;
const K_BASE: f32 = 3.0;
// The weightless V norm erases magnitude (x * inv_rms(x) is ±1 up to eps),
// so only signs distinguish V from a mis-read source; the global flip
// against Q/K makes a V-input mix-up land a wrong sign in every slot.
const V_BASE: f32 = -5.0;
const HEAD_SIGNS: [f32; NUM_Q_HEADS] = [
    1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
];
const Q_DIM: usize = NUM_Q_HEADS * HD;
const KV_DIM: usize = NUM_KV_HEADS * HD;
const SEQ_LEN: usize = 4;
// Exercise nonzero positions and more than one RoPE row.
const START_POS: usize = 1;
const COS_MAX_POS: usize = 8;

fn signed(base: f32, head: usize, token: usize) -> f32 {
    let tok_sign = if token.is_multiple_of(2) { 1.0 } else { -1.0 };
    base * HEAD_SIGNS[head] * tok_sign
}

/// Rows are constant within each (token, head), so mean(x^2) is exact in f32
/// and the kernel's 256-way reduction collapses to a constant.
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
fn cos_sin_tables(ctx: &DeviceContext, rows: usize, rotary_dim: usize) -> (DeviceVec, DeviceVec) {
    let mut cos = Vec::with_capacity(rows * rotary_dim);
    let mut sin = Vec::with_capacity(rows * rotary_dim);
    for row in 0..rows {
        let (c, s) = rope_row(row);
        cos.extend(vec![bf16::from_f32(c); rotary_dim]);
        sin.extend(vec![bf16::from_f32(s); rotary_dim]);
    }
    (
        DeviceVec::from_host(ctx, &cos).expect("cos H2D"),
        DeviceVec::from_host(ctx, &sin).expect("sin H2D"),
    )
}

fn expected_prep(x: f32, w: &[bf16], inv: f32, d: usize, row: usize, rotary_dim: usize) -> f32 {
    let half_rotary = rotary_dim / 2;
    if d < half_rotary {
        let lo = normed(x, w, inv, d);
        let hi = normed(x, w, inv, d + half_rotary);
        match row % 3 {
            0 => lo,
            1 => -hi,
            _ => -lo,
        }
    } else if d < rotary_dim {
        let lo = normed(x, w, inv, d - half_rotary);
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

fn expected_full(base: f32, w: &[bf16], dim: usize, rotary_dim: usize) -> Vec<f32> {
    let num_heads = dim / HD;
    let mut full = vec![0.0f32; dim * SEQ_LEN];
    for t in 0..SEQ_LEN {
        let row = START_POS + t;
        for h in 0..num_heads {
            let x = signed(base, h, t);
            let inv = inv_rms(x);
            for d in 0..HD {
                full[t * dim + h * HD + d] = expected_prep(x, w, inv, d, row, rotary_dim);
            }
        }
    }
    full
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

/// Starts at 1: w[0] = 0 would make dim 0 normalise to 0.0, which an
/// uninitialised output slot could imitate.
fn q_norm_weights() -> Vec<bf16> {
    (1..=HD).map(|d| bf16::from_f32(d as f32)).collect()
}

/// Negated relative to the Q side, so swapping the two weight pointers is
/// a sign flip in every slot. bf16 rounds some magnitudes, which does not
/// matter: oracle and kernel read back the same converted value.
fn k_norm_weights() -> Vec<bf16> {
    (1..=HD).map(|d| bf16::from_f32(-(d as f32))).collect()
}

fn hidden_input(ctx: &DeviceContext, base: f32, num_heads: usize, rows: usize) -> HiddenStates {
    let mut host = Vec::with_capacity(num_heads * HD * rows);
    for t in 0..rows {
        for h in 0..num_heads {
            host.extend(vec![bf16::from_f32(signed(base, h, t)); HD]);
        }
    }
    HiddenStates::from_host(ctx, &host, num_heads * HD, rows).expect("input H2D")
}

fn run_prep(ctx: &DeviceContext, rotary_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let qw = q_norm_weights();
    let kw = k_norm_weights();
    let q = hidden_input(ctx, Q_BASE, NUM_Q_HEADS, SEQ_LEN);
    let k = hidden_input(ctx, K_BASE, NUM_KV_HEADS, SEQ_LEN);
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q_out alloc");
    let mut k_out = HiddenStates::zeros(ctx, KV_DIM, SEQ_LEN).expect("k_out alloc");

    let (cos_dev, sin_dev) = cos_sin_tables(ctx, COS_MAX_POS, rotary_dim);
    let qn = DeviceVec::from_host(ctx, &qw).expect("q_norm_weight H2D");
    let kn = DeviceVec::from_host(ctx, &kw).expect("k_norm_weight H2D");

    qk_norm_rope_prefill_hd256_plain_into(
        ctx,
        &q,
        &k,
        &mut q_out,
        &mut k_out,
        &qn,
        &kn,
        &cos_dev,
        &sin_dev,
        START_POS,
        COS_MAX_POS,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        rotary_dim,
        EPS,
    )
    .expect("prep launch");

    let qo = q_out.to_host(ctx).expect("q_out D2H");
    let ko = k_out.to_host(ctx).expect("k_out D2H");
    (qo, ko)
}

// Pool geometry for the serving-form case. Positions 1..=4 map through
// PAGE_INDICES to pages 3, 7, 5; page id 9 is an out-of-range sentinel the
// kernel must never dereference. The 8-page pool leaves in-range pages
// unreferenced, so stray writes have somewhere visible to land.
const PAGE_SIZE: usize = 2;
const NUM_LAYERS: usize = 2;
const PAGE_INDICES: [i32; 4] = [3, 7, 5, 9];
const POOL_PAGES: usize = 8;

/// K and V blocks at their layout-derived offsets; everything else stays
/// 0.0. V is the weightless norm of the v input — never rotated, no weight.
fn expected_pool(layout: &PagedKvLayout, layer: usize, kw: &[bf16]) -> Vec<f32> {
    let mut exp = vec![0.0f32; layout.page_stride * POOL_PAGES];
    let layer_offset = (layer * layout.layer_stride) as i64;
    for t in 0..SEQ_LEN {
        let pos = START_POS + t;
        let page = PAGE_INDICES[pos / PAGE_SIZE] as i64;
        for h in 0..NUM_KV_HEADS {
            let k_x = signed(K_BASE, h, t);
            let k_inv = inv_rms(k_x);
            let v_x = signed(V_BASE, h, t);
            let v_val = bf16::from_f32(v_x * inv_rms(v_x)).to_f32();
            let base = page * layout.page_stride as i64
                + layer_offset
                + (pos % PAGE_SIZE) as i64 * KV_DIM as i64
                + h as i64 * HD as i64;
            for d in 0..HD {
                exp[(base + d as i64) as usize] = expected_prep(k_x, kw, k_inv, d, pos, HD);
                exp[(base + layout.kv_block_len as i64 + d as i64) as usize] = v_val;
            }
        }
    }
    exp
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

#[test]
fn pool_write_matches_closed_form_and_touches_nothing_else() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let qw = q_norm_weights();
    let kw = k_norm_weights();
    let layer = 1;
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let q = hidden_input(ctx, Q_BASE, NUM_Q_HEADS, SEQ_LEN);
    let k = hidden_input(ctx, K_BASE, NUM_KV_HEADS, SEQ_LEN);
    let v = hidden_input(ctx, V_BASE, NUM_KV_HEADS, SEQ_LEN);
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q_out alloc");
    let (cos_dev, sin_dev) = cos_sin_tables(ctx, COS_MAX_POS, HD);
    let qn = DeviceVec::from_host(ctx, &qw).expect("q_norm_weight H2D");
    let kn = DeviceVec::from_host(ctx, &kw).expect("k_norm_weight H2D");
    let pool: CudaSlice<bf16> = ctx
        .stream
        .alloc_zeros(layout.page_stride * POOL_PAGES)
        .expect("pool alloc");
    let page_indices: CudaSlice<i32> = ctx
        .stream
        .clone_htod(&PAGE_INDICES)
        .expect("page_indices H2D");

    qkv_norm_rope_paged_prefill_hd256_plain_into(
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
        layer,
        &page_indices,
        0,
        0, // page_origin: the row starts at the sequence's first page
        START_POS,
        COS_MAX_POS,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HD,
        EPS,
    )
    .expect("pool prep launch");

    let qo = q_out.to_host(ctx).expect("q_out D2H");
    assert_close(
        &qo,
        &expected_full(Q_BASE, &qw, Q_DIM, HD),
        "pool-write Q pairing",
    );
    let pool_host: Vec<bf16> = ctx.stream.clone_dtoh(&pool).expect("pool D2H");
    let pool_f: Vec<f32> = pool_host.iter().map(|x| x.to_f32()).collect();
    assert_pool(&pool_f, &expected_pool(&layout, layer, &kw));
}

/// rotary_dim = 256 is the Gemma 4 local-layer case: the full head rotates
/// and the pass-through tail is empty.
#[test]
fn full_rotation_matches_closed_form() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let qw = q_norm_weights();
    let kw = k_norm_weights();
    let (qo, ko) = run_prep(ctx, HD);
    assert_close(
        &qo,
        &expected_full(Q_BASE, &qw, Q_DIM, HD),
        "full-rotation Q pairing",
    );
    assert_close(
        &ko,
        &expected_full(K_BASE, &kw, KV_DIM, HD),
        "full-rotation K pairing",
    );
}

/// rotary_dim = 128 exercises the pass-through tail, which the full-width
/// case never reaches; a tail written rotated (or not written) fails here.
#[test]
fn partial_rotation_exercises_tail() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let qw = q_norm_weights();
    let kw = k_norm_weights();
    let (qo, ko) = run_prep(ctx, 128);
    assert_close(
        &qo,
        &expected_full(Q_BASE, &qw, Q_DIM, 128),
        "partial-rotation Q pairing/tail",
    );
    assert_close(
        &ko,
        &expected_full(K_BASE, &kw, KV_DIM, 128),
        "partial-rotation K pairing/tail",
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
    // (8 x 512), so the wrapper's table checks pass and both values
    // actually reach the launcher.
    let q = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q alloc");
    let k = HiddenStates::zeros(ctx, KV_DIM, SEQ_LEN).expect("k alloc");
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q_out alloc");
    let mut k_out = HiddenStates::zeros(ctx, KV_DIM, SEQ_LEN).expect("k_out alloc");
    let cos_dev = DeviceVec::zeros(ctx, COS_MAX_POS * 512).expect("cos alloc");
    let sin_dev = DeviceVec::zeros(ctx, COS_MAX_POS * 512).expect("sin alloc");
    let qn = DeviceVec::zeros(ctx, HD).expect("qn alloc");
    let kn = DeviceVec::zeros(ctx, HD).expect("kn alloc");

    // 127: odd — index 126 would never be written. 512: wider than the
    // head — smem and the output slices would walk past 256. Both must be
    // rejected, not silently accepted.
    for bad in [127, 512] {
        let err = qk_norm_rope_prefill_hd256_plain_into(
            ctx,
            &q,
            &k,
            &mut q_out,
            &mut k_out,
            &qn,
            &kn,
            &cos_dev,
            &sin_dev,
            0, // start_pos
            COS_MAX_POS,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            bad,
            EPS,
        );
        assert!(err.is_err(), "rotary_dim={bad} must be rejected");
    }
}

#[test]
fn rejects_position_beyond_cos_table() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    // Rejected on the host, before any launch: with cos_max_pos 4,
    // start_pos 1 + seq_len 4 reaches row 5 — past the tables.
    let q = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q alloc");
    let k = HiddenStates::zeros(ctx, KV_DIM, SEQ_LEN).expect("k alloc");
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q_out alloc");
    let mut k_out = HiddenStates::zeros(ctx, KV_DIM, SEQ_LEN).expect("k_out alloc");
    let cos_dev = DeviceVec::zeros(ctx, 4 * HD).expect("cos alloc");
    let sin_dev = DeviceVec::zeros(ctx, 4 * HD).expect("sin alloc");
    let qn = DeviceVec::zeros(ctx, HD).expect("qn alloc");
    let kn = DeviceVec::zeros(ctx, HD).expect("kn alloc");

    let err = qk_norm_rope_prefill_hd256_plain_into(
        ctx,
        &q,
        &k,
        &mut q_out,
        &mut k_out,
        &qn,
        &kn,
        &cos_dev,
        &sin_dev,
        1, // start_pos
        4, // cos_max_pos
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HD,
        EPS,
    );
    assert!(
        err.is_err(),
        "start_pos + seq_len beyond cos_max_pos must be rejected on the host"
    );
}

#[test]
fn batched_decode_prep_matches_closed_form() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let qw = q_norm_weights();
    let kw = k_norm_weights();
    let layer = 1;
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let batch = 2usize;
    // [7] at origin 1 maps pos 3 to page 7; [3, 5] at origin 0 maps it to page 5.
    let positions: [i32; 2] = [3, 3];
    let origins: [i32; 2] = [1, 0];
    let pages_cat: [i32; 3] = [7, 3, 5];
    let indptr: [i32; 3] = [0, 1, 3];

    let q = hidden_input(ctx, Q_BASE, NUM_Q_HEADS, batch);
    let k = hidden_input(ctx, K_BASE, NUM_KV_HEADS, batch);
    let v = hidden_input(ctx, V_BASE, NUM_KV_HEADS, batch);
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, batch).expect("q_out alloc");
    let (cos_dev, sin_dev) = cos_sin_tables(ctx, COS_MAX_POS, HD);
    let qn = DeviceVec::from_host(ctx, &qw).expect("q_norm_weight H2D");
    let kn = DeviceVec::from_host(ctx, &kw).expect("k_norm_weight H2D");
    let pool: CudaSlice<bf16> = ctx
        .stream
        .alloc_zeros(layout.page_stride * POOL_PAGES)
        .expect("pool alloc");
    let pages_d: CudaSlice<i32> = ctx.stream.clone_htod(&pages_cat).expect("pages H2D");
    let indptr_d: CudaSlice<i32> = ctx.stream.clone_htod(&indptr).expect("indptr H2D");
    let origins_d: CudaSlice<i32> = ctx.stream.clone_htod(&origins).expect("origins H2D");
    let positions_d: CudaSlice<i32> = ctx.stream.clone_htod(&positions).expect("positions H2D");

    qkv_norm_rope_paged_decode_hd256_plain_into(
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
        layer,
        &pages_d,
        &indptr_d,
        &origins_d,
        &positions_d,
        COS_MAX_POS,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HD,
        EPS,
    )
    .expect("batched decode prep launch");

    let qo = q_out.to_host(ctx).expect("q_out D2H");
    let mut q_exp = vec![0.0f32; Q_DIM * batch];
    for (row, &pos) in positions.iter().enumerate() {
        for h in 0..NUM_Q_HEADS {
            let x = signed(Q_BASE, h, row);
            let inv = inv_rms(x);
            for d in 0..HD {
                q_exp[row * Q_DIM + h * HD + d] = expected_prep(x, &qw, inv, d, pos as usize, HD);
            }
        }
    }
    assert_close(&qo, &q_exp, "batched decode pairing/tail");

    let pool_host: Vec<bf16> = ctx.stream.clone_dtoh(&pool).expect("pool D2H");
    let got: Vec<f32> = pool_host.iter().map(|x| x.to_f32()).collect();
    let mut exp = vec![0.0f32; layout.page_stride * POOL_PAGES];
    let layer_offset = (layer * layout.layer_stride) as i64;
    for (row, (&pos, &page)) in positions.iter().zip([7i32, 5i32].iter()).enumerate() {
        for h in 0..NUM_KV_HEADS {
            let k_x = signed(K_BASE, h, row);
            let k_inv = inv_rms(k_x);
            let v_x = signed(V_BASE, h, row);
            let v_val = bf16::from_f32(v_x * inv_rms(v_x)).to_f32();
            let base = page as i64 * layout.page_stride as i64
                + layer_offset
                + (pos as usize % PAGE_SIZE) as i64 * KV_DIM as i64
                + h as i64 * HD as i64;
            for d in 0..HD {
                exp[(base + d as i64) as usize] =
                    expected_prep(k_x, &kw, k_inv, d, pos as usize, HD);
                exp[(base + layout.kv_block_len as i64 + d as i64) as usize] = v_val;
            }
        }
    }
    assert_pool(&got, &exp);
}

/// The row-offset suffix contract: with `row_offset = 1` over three rows,
/// the decode prep must leave the prefix row of `q_out` untouched, and its
/// suffix outputs and pool writes must equal a zero-offset run over the
/// same two suffix rows.
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
    let positions: [i32; 2] = [3, 3];
    let origins: [i32; 2] = [1, 0];
    let pages_cat: [i32; 3] = [7, 3, 5];
    let indptr: [i32; 3] = [0, 1, 3];

    // Shared suffix bytes: the offset arm's rows [1..3) are byte-identical
    // to the zero-offset arm's rows [0..2); its prefix row is filler the
    // prep must neither read through nor overwrite.
    let suffix = |base: f32, heads: usize| -> Vec<bf16> {
        let mut host = Vec::new();
        for t in 0..batch {
            for h in 0..heads {
                host.extend(vec![bf16::from_f32(signed(base, h, t)); HD]);
            }
        }
        host
    };
    let with_prefix = |sfx: &[bf16], heads: usize| -> Vec<bf16> {
        let mut host = vec![bf16::from_f32(FILLER); heads * HD];
        host.extend_from_slice(sfx);
        host
    };
    let (qs, ks, vs) = (
        suffix(Q_BASE, NUM_Q_HEADS),
        suffix(K_BASE, NUM_KV_HEADS),
        suffix(V_BASE, NUM_KV_HEADS),
    );
    let (cos_dev, sin_dev) = cos_sin_tables(ctx, COS_MAX_POS, HD);
    let qn = DeviceVec::from_host(ctx, &qw).expect("q_norm_weight H2D");
    let kn = DeviceVec::from_host(ctx, &kw).expect("k_norm_weight H2D");
    let pages_d: CudaSlice<i32> = ctx.stream.clone_htod(&pages_cat).expect("pages H2D");
    let indptr_d: CudaSlice<i32> = ctx.stream.clone_htod(&indptr).expect("indptr H2D");
    let origins_d: CudaSlice<i32> = ctx.stream.clone_htod(&origins).expect("origins H2D");
    let positions_d: CudaSlice<i32> = ctx.stream.clone_htod(&positions).expect("positions H2D");

    let run = |offset: usize, q_host: &[bf16], k_host: &[bf16], v_host: &[bf16]| {
        let rows = offset + batch;
        let q = HiddenStates::from_host(ctx, q_host, Q_DIM, rows).expect("q H2D");
        let k = HiddenStates::from_host(ctx, k_host, KV_DIM, rows).expect("k H2D");
        let v = HiddenStates::from_host(ctx, v_host, KV_DIM, rows).expect("v H2D");
        let sentinel = vec![bf16::from_f32(SENTINEL); Q_DIM * rows];
        let mut q_out = HiddenStates::from_host(ctx, &sentinel, Q_DIM, rows).expect("q_out H2D");
        let pool: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(layout.page_stride * POOL_PAGES)
            .expect("pool alloc");
        qkv_norm_rope_paged_decode_hd256_plain_into(
            ctx,
            &q,
            &k,
            &v,
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
            &origins_d,
            &positions_d,
            COS_MAX_POS,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HD,
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

    let (out_a, pool_a) = run(
        1,
        &with_prefix(&qs, NUM_Q_HEADS),
        &with_prefix(&ks, NUM_KV_HEADS),
        &with_prefix(&vs, NUM_KV_HEADS),
    );
    let (out_b, pool_b) = run(0, &qs, &ks, &vs);

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

/// The row and page-table windows of the hd256 prefill prep: the offset arm
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
    let (cos_dev, sin_dev) = cos_sin_tables(ctx, COS_MAX_POS, HD);
    let qn = DeviceVec::from_host(ctx, &qw).expect("q_norm_weight H2D");
    let kn = DeviceVec::from_host(ctx, &kw).expect("k_norm_weight H2D");

    let suffix = |base: f32, heads: usize| -> Vec<bf16> {
        let mut host = Vec::new();
        for t in 0..SEQ_LEN {
            for h in 0..heads {
                host.extend(vec![bf16::from_f32(signed(base, h, t)); HD]);
            }
        }
        host
    };
    let with_prefix = |sfx: &[bf16], heads: usize| -> Vec<bf16> {
        let mut host = vec![bf16::from_f32(FILLER); heads * HD];
        host.extend_from_slice(sfx);
        host
    };
    let (qs, ks, vs) = (
        suffix(Q_BASE, NUM_Q_HEADS),
        suffix(K_BASE, NUM_KV_HEADS),
        suffix(V_BASE, NUM_KV_HEADS),
    );

    let run = |offset: usize,
               q_host: &[bf16],
               k_host: &[bf16],
               v_host: &[bf16],
               table: &[i32],
               pages_offset: usize| {
        let rows = offset + SEQ_LEN;
        let q = HiddenStates::from_host(ctx, q_host, Q_DIM, rows).expect("q H2D");
        let k = HiddenStates::from_host(ctx, k_host, KV_DIM, rows).expect("k H2D");
        let v = HiddenStates::from_host(ctx, v_host, KV_DIM, rows).expect("v H2D");
        let sentinel = vec![bf16::from_f32(SENTINEL); Q_DIM * rows];
        let mut q_out = HiddenStates::from_host(ctx, &sentinel, Q_DIM, rows).expect("q_out H2D");
        let pool: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(layout.page_stride * POOL_PAGES)
            .expect("pool alloc");
        let page_indices: CudaSlice<i32> = ctx.stream.clone_htod(table).expect("pages H2D");
        qkv_norm_rope_paged_prefill_hd256_plain_into(
            ctx,
            &q,
            &k,
            &v,
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
            0,
            START_POS,
            COS_MAX_POS,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HD,
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
        &with_prefix(&qs, NUM_Q_HEADS),
        &with_prefix(&ks, NUM_KV_HEADS),
        &with_prefix(&vs, NUM_KV_HEADS),
        &junk_table,
        1,
    );
    let (out_b, pool_b) = run(0, &qs, &ks, &vs, &PAGE_INDICES, 0);

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
