//! Device gate for csrc/shared/prefill_attention_hd256_plain.cu.
//!
//! Manual gate: CI compiles this but never runs it. Run on a GPU box with
//! PEGAINFER_REQUIRE_GPU=1, which turns a missing device into a failure
//! rather than a skip.
//!
//! Two closed-form anchors, then a chain. `full_rotation_matches_closed_form`
//! and `partial_rotation_exercises_tail` are the only tests here that restate
//! the operator, and they are two because rotary_dim == HD and rotary_dim < HD
//! are different control flow: at full width no thread reaches the
//! pass-through tail at all, and full width is the Gemma 4 local-layer
//! production config. From there each gate is measured against an arm the
//! anchors certify — the contiguous kernel certifies the values the paged
//! prefill must land at its layout addresses, and the paged prefill certifies
//! the bits the paged decode must reproduce. Nothing below the anchors
//! recomputes the norm, the RoPE, or the paged address formula.
//!
//! Extend the chain, do not grow a host oracle: a new dispatch variant is
//! compared against the arm one link above it. See
//! docs/subsystems/kernels/qk-rope-smoke-oracles.md for why, and for the
//! negative controls that keep a GPU-vs-GPU gate honest.

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

/// The pool slot a (position, kv head, element) triple owns, read straight
/// off the production `PagedKvLayout` rather than restated from the page
/// geometry. This is the only address arithmetic left in the file: the
/// decode gate below compares two pools element for element and needs none.
fn pool_k_offset(
    layout: &PagedKvLayout,
    layer: usize,
    page: usize,
    pos: usize,
    kv_head: usize,
) -> usize {
    page * layout.page_stride
        + layer * layout.layer_stride
        + (pos % layout.page_size) * layout.num_kv_heads * layout.head_dim
        + kv_head * layout.head_dim
}

/// Bitwise, not approximate: every expectation here is a value another GPU
/// arm produced from the same input at the same position, so the two cannot
/// differ at all. The lone host-computed expectation (the weightless V norm)
/// is exact too, for reasons worth reading before loosening this: see
/// docs/subsystems/kernels/qk-rope-smoke-oracles.md.
///
/// A zero expectation in a pool comparison is a slot the kernel must never
/// have written — unreferenced pages, the other layer, the slots outside the
/// request's positions — so it is asserted as hard as any other value.
///
/// Reports the first differing element. `assert_eq!` on the whole slice
/// would dump two vectors of tens of thousands of values and bury it.
fn assert_bits_eq(got: &[u16], expected: &[u16], what: &str) {
    assert_eq!(got.len(), expected.len(), "{what}: length differs");
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            g,
            e,
            "{what}[{i}]: got {}, expected {}",
            bf16::from_bits(g),
            bf16::from_bits(e)
        );
    }
}

/// bf16 bits, not the widened f32: the pool stores bf16, so reading both
/// sides the same way lets one comparison cover Q rows and pool slots alike.
fn row_bits(rows: &[f32]) -> Vec<u16> {
    rows.iter().map(|&v| bf16::from_f32(v).to_bits()).collect()
}

/// Norm weights and RoPE tables on device: the setup both paged gates share.
fn device_fixture(ctx: &DeviceContext) -> (DeviceVec, DeviceVec, DeviceVec, DeviceVec) {
    let (cos_dev, sin_dev) = cos_sin_tables(ctx, COS_MAX_POS, HD);
    (
        DeviceVec::from_host(ctx, &q_norm_weights()).expect("q_norm_weight H2D"),
        DeviceVec::from_host(ctx, &k_norm_weights()).expect("k_norm_weight H2D"),
        cos_dev,
        sin_dev,
    )
}

/// The paged prefill over the fixture window, read back as (Q rows, pool)
/// bit patterns. Both gates below need this arm — one as the subject, one as
/// the oracle the decode form must reproduce — and the wrapper takes
/// twenty-two arguments, so it is spelled once here rather than twice.
fn run_paged_prefill(
    ctx: &DeviceContext,
    layout: &PagedKvLayout,
    layer: usize,
    qn: &DeviceVec,
    kn: &DeviceVec,
    cos_dev: &DeviceVec,
    sin_dev: &DeviceVec,
) -> (Vec<u16>, Vec<u16>) {
    let q = hidden_input(ctx, Q_BASE, NUM_Q_HEADS, SEQ_LEN);
    let k = hidden_input(ctx, K_BASE, NUM_KV_HEADS, SEQ_LEN);
    let v = hidden_input(ctx, V_BASE, NUM_KV_HEADS, SEQ_LEN);
    let mut q_out = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q_out alloc");
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
        layout,
        qn,
        kn,
        cos_dev,
        sin_dev,
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
    .expect("paged prefill launch");
    let pool_host: Vec<bf16> = ctx.stream.clone_dtoh(&pool).expect("pool D2H");
    (
        row_bits(&q_out.to_host(ctx).expect("q_out D2H")),
        pool_host.iter().map(|x| x.to_bits()).collect(),
    )
}

/// The paged prefill's addressing, against the contiguous arm rather than a
/// second copy of the operator. `full_rotation_matches_closed_form`
/// certifies that arm's norm and RoPE at this rotary width, so driving it on
/// the same rows from the same start_pos yields the Q and K this prefill
/// must produce, and the only open question left here is whether the paged
/// form lands those values in the slots `PagedKvLayout` names and leaves
/// every other slot alone.
///
/// V has no contiguous counterpart — the flat kernel carries no V band — so
/// its weightless norm stays as a one-line expectation. That is one multiply
/// over a reduction the anchor already pins, not a second operator.
#[test]
fn paged_prefill_lands_flat_values_at_layout_addresses() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let layer = 1;
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let (qn, kn, cos_dev, sin_dev) = device_fixture(ctx);

    // Oracle arm: the same rows over the same position window, through the
    // contiguous kernel the closed-form anchors certify.
    let oracle_q = hidden_input(ctx, Q_BASE, NUM_Q_HEADS, SEQ_LEN);
    let oracle_k = hidden_input(ctx, K_BASE, NUM_KV_HEADS, SEQ_LEN);
    let mut oracle_q_out = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("oracle q_out alloc");
    let mut oracle_k_out = HiddenStates::zeros(ctx, KV_DIM, SEQ_LEN).expect("oracle k_out alloc");
    qk_norm_rope_prefill_hd256_plain_into(
        ctx,
        &oracle_q,
        &oracle_k,
        &mut oracle_q_out,
        &mut oracle_k_out,
        &qn,
        &kn,
        &cos_dev,
        &sin_dev,
        START_POS,
        COS_MAX_POS,
        NUM_Q_HEADS,
        NUM_KV_HEADS,
        HD,
        EPS,
    )
    .expect("oracle flat prep launch");
    let oracle_q_bits = row_bits(&oracle_q_out.to_host(ctx).expect("oracle q_out D2H"));
    let oracle_k_vals = oracle_k_out.to_host(ctx).expect("oracle k_out D2H");

    let (q_bits, pool_bits) = run_paged_prefill(ctx, &layout, layer, &qn, &kn, &cos_dev, &sin_dev);
    assert_bits_eq(
        &q_bits,
        &oracle_q_bits,
        "paged Q vs the contiguous arm over the same window",
    );

    let mut expected = vec![0u16; layout.page_stride * POOL_PAGES];
    for t in 0..SEQ_LEN {
        let pos = START_POS + t;
        let page = PAGE_INDICES[pos / PAGE_SIZE] as usize;
        for h in 0..NUM_KV_HEADS {
            let v_x = bf16::from_f32(signed(V_BASE, h, t)).to_f32();
            let v_val = bf16::from_f32(v_x * inv_rms(v_x)).to_bits();
            let base = pool_k_offset(&layout, layer, page, pos, h);
            for d in 0..HD {
                expected[base + d] =
                    bf16::from_f32(oracle_k_vals[t * KV_DIM + h * HD + d]).to_bits();
                expected[base + layout.kv_block_len + d] = v_val;
            }
        }
    }
    assert_bits_eq(&pool_bits, &expected, "pool");
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

/// The per-token metadata form against the whole-window form. Both are
/// instantiations of one template, differing only in where `pos`, the page
/// window and the window's origin come from, so driving the decode arm with
/// a table replicated per row and the origin the prefill arm was handed must
/// reproduce that arm bit for bit.
///
/// Everything the two share cancels: the norm, the RoPE, the V band, the
/// pool addressing. What is left under test is exactly the routing —
/// `positions[]`, the CSR window, the per-token origin — which is the only
/// code `PER_TOKEN_META` switches on. No host expectation is computed here
/// and no address is derived.
#[test]
fn paged_decode_equals_paged_prefill_over_the_same_positions() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let ctx = &ctx;
    let layer = 1;
    let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HD, PAGE_SIZE);
    let (qn, kn, cos_dev, sin_dev) = device_fixture(ctx);

    // Whole-window arm: one table for the whole prompt, positions derived
    // from start_pos, one scalar origin for every row.
    let (prefill_q, prefill_kv) =
        run_paged_prefill(ctx, &layout, layer, &qn, &kn, &cos_dev, &sin_dev);

    // Per-token arm: equivalent metadata in the form the decode path takes.
    // Each row's window is compressed to start at the page holding that row's
    // own position — a caller that released the front passes exactly this —
    // so the origins differ across rows (0, 1, 1, 2) and no two rows share a
    // table span. The per-token origin and the CSR window are both
    // load-bearing here, not defaulted: a form that ignored
    // page_origins[token] would take row pos/page_size into a window that no
    // longer starts at page 0 and land somewhere else. Page 9 stays past
    // every reachable row in both arms, so dereferencing the out-of-range
    // sentinel traps rather than passing quietly.
    let mut per_token_pages: Vec<i32> = Vec::new();
    let mut indptr: Vec<i32> = vec![0];
    let mut origins: Vec<i32> = Vec::new();
    let mut positions: Vec<i32> = Vec::new();
    for t in 0..SEQ_LEN {
        let pos = START_POS + t;
        let origin = pos / PAGE_SIZE;
        per_token_pages.extend_from_slice(&PAGE_INDICES[origin..]);
        indptr.push(per_token_pages.len() as i32);
        origins.push(origin as i32);
        positions.push(pos as i32);
    }
    let q = hidden_input(ctx, Q_BASE, NUM_Q_HEADS, SEQ_LEN);
    let k = hidden_input(ctx, K_BASE, NUM_KV_HEADS, SEQ_LEN);
    let v = hidden_input(ctx, V_BASE, NUM_KV_HEADS, SEQ_LEN);
    let mut decode_q_out = HiddenStates::zeros(ctx, Q_DIM, SEQ_LEN).expect("q_out alloc");
    let decode_pool: CudaSlice<bf16> = ctx
        .stream
        .alloc_zeros(layout.page_stride * POOL_PAGES)
        .expect("pool alloc");
    let pages_d: CudaSlice<i32> = ctx.stream.clone_htod(&per_token_pages).expect("pages H2D");
    let indptr_d: CudaSlice<i32> = ctx.stream.clone_htod(&indptr).expect("indptr H2D");
    let origins_d: CudaSlice<i32> = ctx.stream.clone_htod(&origins).expect("origins H2D");
    let positions_d: CudaSlice<i32> = ctx.stream.clone_htod(&positions).expect("positions H2D");
    qkv_norm_rope_paged_decode_hd256_plain_into(
        ctx,
        &q,
        &k,
        &v,
        &mut decode_q_out,
        0,
        &decode_pool,
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
    .expect("paged decode prep launch");

    let decode_q = row_bits(&decode_q_out.to_host(ctx).expect("q_out D2H"));
    let decode_pool_host: Vec<bf16> = ctx.stream.clone_dtoh(&decode_pool).expect("pool D2H");
    let decode_kv: Vec<u16> = decode_pool_host.iter().map(|x| x.to_bits()).collect();
    assert_bits_eq(&decode_q, &prefill_q, "per-token Q vs the whole-window run");
    assert_bits_eq(
        &decode_kv,
        &prefill_kv,
        "per-token pool writes vs the whole-window run",
    );
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
