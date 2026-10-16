//! Qwen3.8-Flash-Next QSA sparse attention against the frozen reference golden.
//!
//! `test_data/qwen38fn-qsa-attn-golden.safetensors` was dumped from
//! transformers 5.16.1 `qwen4_exp` by
//! `tools/accuracy/dump_qwen38fn_qsa_attn_golden.py`. Each case carries one
//! full-attention layer's captured activations — the fused `q_proj` rows of the
//! evaluated query rows, every token's `k_proj` and `v_proj` rows, the effective
//! sparse+causal visibility — plus the reference's own softmax weights and the
//! gated output it hands `o_proj`. So this gate replays those inputs through the
//! kernel and compares both observables. The four plain projections are not
//! covered here: the golden starts after `q_proj`/`k_proj`/`v_proj` and ends
//! before `o_proj`, and the indexer that produced the mask has its own oracle.
//!
//! The cases cover S=8/10/64 with every row evaluated, and the real budget
//! boundary at S=2055 with six evaluated rows (2049..2055) — the last of which
//! is where the 2048-token budget starts dropping a 4-token block, so its mask
//! is sparse where the first three cases' masks are exactly causal. Case 3 is
//! also the only one whose evaluated rows are a strict subset of its key array.
//!
//! Provenance is asserted, not assumed: the fixture records the reference
//! version and the sha256 of the frozen config it was dumped from, and that
//! config's attention geometry is held against the geometry this crate compiles
//! for. A fixture from another checkpoint or another semantics fails here
//! instead of being skipped.
//!
//! Three negative controls keep the gate from being vacuous — each one breaks a
//! single semantic and must break the comparison: the stored mask replaced by
//! plain causality (which the first three cases cannot see and the boundary case
//! must), the two KV heads swapped (a wrong GQA mapping), and the output gate
//! saturated so `sigmoid` is 1 (which must leave the weights alone and break
//! only the output).
//!
//! What this fixture cannot pin: its `positions` are `arange`, so on the key side
//! `positions[s]` and the token index `s` are the same number and a kernel that
//! rotated keys by their index instead of by their position would pass. The query
//! side *is* pinned — case 3's six evaluated rows are tokens 2049..2054, so a
//! slot-indexed rotation cannot match there. Paged KV and the decode path are a
//! later stage's business and are not gated here at all.
//!
//! Without a device it skips; `PEGAINFER_REQUIRE_GPU=1` turns that into a
//! failure.
//!
//!   cargo test --release -p pegainfer-kernels --features qwen4 \
//!     --test qwen4_qsa_attn_parity -- --nocapture

#![cfg(feature = "qwen4")]

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::QsaAttnGeometry;
use pegainfer_kernels::ops::qsa_attn_forward_into;
use pegainfer_kernels::tensor::DeviceContext;
use safetensors::SafeTensors;
use sha2::Digest;
use sha2::Sha256;

const FIXTURE: &str = "qwen38fn-qsa-attn-golden.safetensors";
const METADATA_KEY: &str = "qwen4_qsa_attn_golden";
/// Frozen config the golden was dumped from, relative to the workspace root.
const CONFIG: &str = "pegainfer-qwen4/tests/frozen_config.json";
const FORMAT_VERSION: u64 = 2;
/// The reference implementation the golden's semantics are frozen against.
const REF_TRANSFORMERS: &str = "5.16.1";
const CASES: usize = 4;

/// Weight parity bound, `abs(got - want) <= ATOL + RTOL * abs(want)`.
///
/// All 102240 weights of the first three cases are bit-identical, and so are
/// 295332 of case 3's 295920 — the one with 2055 keys and positions past 63. Of
/// its 588 deviations, 532 are one bf16 quantum, 56 are over one and 7 over two,
/// the worst 1.68e-2 of the weight (8 quanta at that magnitude).
///
/// That is a rounding decision, not a semantic difference, and it is the
/// reference's own floor rather than this kernel's error. A weight is
/// `exp(score - max) / sum`, and the score it reads is the bf16 rounding of a
/// 256-wide float32 dot, so one ulp of a score at this fixture's largest (6.28)
/// moves its weight by `exp(0.049) - 1` = 5%. The reference rounds its float32
/// reductions — the norm's 256-wide mean of squares, the GEMM's dot, the
/// softmax's sum — in a torch-defined order that differs from this kernel's
/// shuffle trees, so where a value sits within a float32 ulp of a bf16 midpoint
/// the two round apart; one flipped norm weight on a key element is worth the 72
/// scores that read it, which is why case 3 carries a few hundred deviations and
/// the short cases, whose means land further from a midpoint, carry none. The
/// deviations are also sign-balanced around a worst of 8 quanta, which a
/// systematic error does not produce.
///
/// So the bound is one score ulp at this fixture's largest score — 5e-2 relative,
/// against a measured worst of a third of it — plus an absolute term far under the
/// smallest weight a visible key can have here (`exp(-12)`). A real fault is
/// orders out: the three controls below land at 4.1e4, 5.5e4 and 1.3e2 of the
/// bound.
const WEIGHT_ATOL: f32 = 1.0e-7;
const WEIGHT_RTOL: f32 = 5.0e-2;
/// Output parity bound, same form. The output folds every visible key's rounded
/// weight, so it inherits the weights' flips and adds the context reduction's own
/// order: case 2 leaves 9 of 393216 elements one ulp off, case 3 is 95.4%
/// bit-identical with 479 elements over a ulp and 181 over two, max abs 4.9e-4
/// and max rel 3.0e-2 over the elements big enough to read relatively.
///
/// The absolute term is what the near-zero elements need, and why one relative
/// bound cannot serve both observables: case 3's outputs cancel down through
/// zero, where the float32 accumulation's own noise (a 2050-term sum of ~5e-4
/// products, plus the weights' flips at ~1e-5 each) is large against the value
/// however small it is in absolute terms. The worst element by bound fraction is
/// 2.19e-4 against 2.67e-4 — 50 quanta and 18% of it, but only 4.8e-5 absolute,
/// which is the same order as the max abs over all 36864 elements. That is the
/// reading `REL_FLOOR` keeps out of the relative column: below it, quantum
/// counts and percentages describe the cancellation, not the kernel.
///
/// Measured, the relative term alone leaves 3.4e-5 uncovered; the absolute term
/// is set at three times that, against outputs that reach 1.6 in case 2. The
/// closest a control comes to passing is 2.2e1 of the bound, so the term is not
/// what makes this gate pass.
const OUT_ATOL: f32 = 1.0e-4;
const OUT_RTOL: f32 = 5.0e-2;

/// Written into both output buffers before every launch. Both are fully written
/// by the kernel, so a surviving NaN is a launch that did not cover the shape
/// rather than a rounding difference, and it fails the finiteness check with that
/// reading instead of as an unexplained deviation.
const SENTINEL: bf16 = bf16::NAN;

fn fixture_path() -> String {
    format!("{}/../test_data/{FIXTURE}", env!("CARGO_MANIFEST_DIR"))
}

/// Bit equality for the scalars the kernel pins and the exact derivations
/// beside them. The kernel refuses any other value, so a tolerance here would
/// let a config drift through.
fn same_f32(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits()
}

fn same_f64(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    Sha256::digest(bytes)
        .iter()
        .fold(String::with_capacity(64), |mut hex, byte| {
            let _ = write!(hex, "{byte:02x}");
            hex
        })
}

/// The fixture's own record of what it was dumped from, held against the
/// reference version, the frozen config's bytes and the attention geometry this
/// crate compiles for. Fails loudly on any disagreement: a golden from another
/// checkpoint or another reference version is not a skippable nuisance, it is a
/// comparison against the wrong semantics.
fn assert_provenance(bytes: &[u8]) {
    let (_, metadata) = SafeTensors::read_metadata(bytes).expect("fixture metadata");
    let recorded = metadata
        .metadata()
        .as_ref()
        .expect("fixture metadata map")
        .get(METADATA_KEY)
        .expect("fixture manifest key")
        .clone();
    let manifest: serde_json::Value =
        serde_json::from_str(&recorded).expect("fixture manifest is not JSON");

    assert_eq!(
        manifest["format_version"].as_u64(),
        Some(FORMAT_VERSION),
        "fixture format_version; this gate reads version {FORMAT_VERSION}"
    );
    assert_eq!(
        manifest["transformers_version"].as_str(),
        Some(REF_TRANSFORMERS),
        "fixture was dumped from another reference version, and the frozen \
         semantics are that one's"
    );
    assert_eq!(
        manifest["config"].as_str(),
        Some(CONFIG),
        "fixture was dumped from another config"
    );

    let config_path = format!("{}/../{CONFIG}", env!("CARGO_MANIFEST_DIR"));
    let config_bytes =
        std::fs::read(&config_path).unwrap_or_else(|e| panic!("read {config_path}: {e}"));
    let expected = sha256_hex(&config_bytes);
    assert_eq!(
        manifest["config_sha256"].as_str(),
        Some(expected.as_str()),
        "{CONFIG} is not the config the fixture was dumped from"
    );

    let config: serde_json::Value =
        serde_json::from_slice(&config_bytes).expect("frozen config is not JSON");
    let text = &config["text_config"];
    let geometry = QsaAttnGeometry::qwen38_flash_next();
    let field = |name: &str| text[name].as_u64().unwrap_or_else(|| panic!("{name}"));
    assert_eq!(field("num_attention_heads"), geometry.heads as u64);
    assert_eq!(field("num_key_value_heads"), geometry.kv_heads as u64);
    assert_eq!(field("head_dim"), geometry.head_dim as u64);
    assert_eq!(
        geometry.heads % geometry.kv_heads,
        0,
        "the GQA group must divide the query heads"
    );
    let rms_eps = text["rms_norm_eps"].as_f64().expect("rms_norm_eps");
    assert!(
        same_f32(rms_eps as f32, geometry.rms_eps),
        "rms_norm_eps {rms_eps} is not the compiled {}",
        geometry.rms_eps
    );
    let rope_theta = text["rope_parameters"]["rope_theta"]
        .as_f64()
        .expect("rope_theta");
    assert!(
        same_f32(rope_theta as f32, geometry.rope_theta),
        "rope_theta {rope_theta} is not the compiled {}",
        geometry.rope_theta
    );
    // rotary_dim = head_dim * partial_rotary_factor: the rotated prefix of an
    // attention head, which is a quarter of it here.
    let partial = text["rope_parameters"]["partial_rotary_factor"]
        .as_f64()
        .expect("partial_rotary_factor");
    assert!(
        same_f64(
            field("head_dim") as f64 * partial,
            geometry.rotary_dim as f64
        ),
        "rotary_dim {} is not head_dim {} * partial_rotary_factor {partial}",
        geometry.rotary_dim,
        field("head_dim")
    );
    assert!(geometry.rotary_dim <= geometry.head_dim);
    // Recorded, not asserted: the host the golden was dumped on decides the
    // order its float32 reductions round in, which is where the parity bound's
    // residual comes from, and any other host's order is an equally valid
    // reference.
    println!(
        "golden provenance: transformers {}, torch {}, {}",
        manifest["transformers_version"].as_str().unwrap_or("?"),
        manifest["torch_version"].as_str().unwrap_or("?"),
        manifest["platform"].as_str().unwrap_or("?"),
    );
}

fn i32_tensor(fixture: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<i32>) {
    let view = fixture
        .tensor(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let host = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| i32::from_le_bytes(*b))
        .collect();
    (view.shape().to_vec(), host)
}

fn bf16_tensor(fixture: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<bf16>) {
    let view = fixture
        .tensor(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let host = view
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| bf16::from_bits(u16::from_le_bytes(*b)))
        .collect();
    (view.shape().to_vec(), host)
}

fn bool_tensor(fixture: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<u8>) {
    let view = fixture
        .tensor(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    (view.shape().to_vec(), view.data().to_vec())
}

/// One case's inputs and the reference's two answers for its evaluated rows.
struct Case {
    index: usize,
    tokens: usize,
    /// The evaluated query rows, as token indices into the key array.
    rows: Vec<i32>,
    positions: Vec<i32>,
    /// `[rows, heads * head_dim * 2]`: each head's query dims, then its gate.
    queries: Vec<bf16>,
    keys: Vec<bf16>,
    values: Vec<bf16>,
    /// `[rows, tokens]`, non-zero where the token is visible.
    mask: Vec<u8>,
    /// `[heads, rows, tokens]`.
    weights: Vec<bf16>,
    /// `[rows, heads * head_dim]`, the input to `o_proj`.
    out: Vec<bf16>,
}

fn load_case(fixture: &SafeTensors<'_>, index: usize, geometry: &QsaAttnGeometry) -> Case {
    let (query_shape, queries) = bf16_tensor(fixture, &format!("case{index}_queries"));
    let (key_shape, keys) = bf16_tensor(fixture, &format!("case{index}_keys"));
    let (value_shape, values) = bf16_tensor(fixture, &format!("case{index}_values"));
    let (position_shape, positions) = i32_tensor(fixture, &format!("case{index}_positions"));
    let (rows_shape, rows) = i32_tensor(fixture, &format!("case{index}_rows"));
    let (mask_shape, mask) = bool_tensor(fixture, &format!("case{index}_mask"));
    let (weight_shape, weights) = bf16_tensor(fixture, &format!("case{index}_attn_weights"));
    let (out_shape, out) = bf16_tensor(fixture, &format!("case{index}_attn_out"));
    let (seq_len_shape, seq_len) = i32_tensor(fixture, &format!("case{index}_seq_len"));

    let tokens = key_shape[0];
    let evaluated = rows_shape[0];
    assert_eq!(seq_len_shape, vec![1], "case {index}: seq_len shape");
    assert_eq!(
        seq_len[0], tokens as i32,
        "case {index}: seq_len against the key array"
    );
    assert_eq!(key_shape, vec![tokens, geometry.kv_width()]);
    assert_eq!(value_shape, key_shape, "case {index}: values against keys");
    assert_eq!(position_shape, vec![tokens]);
    assert_eq!(query_shape, vec![evaluated, geometry.query_width()]);
    assert_eq!(mask_shape, vec![evaluated, tokens]);
    assert_eq!(weight_shape, vec![1, geometry.heads, evaluated, tokens]);
    assert_eq!(out_shape, vec![evaluated, geometry.out_width()]);
    assert!(evaluated > 0, "case {index}: no evaluated rows");
    assert!(
        tokens <= QsaAttnGeometry::MAX_TOKENS,
        "case {index}: {tokens} tokens exceed what one call accepts"
    );
    for (slot, row) in rows.iter().enumerate() {
        assert!(
            (0..tokens as i32).contains(row),
            "case {index}: evaluated row {slot} is token {row}, outside the key array"
        );
    }
    for (token, position) in positions.iter().enumerate() {
        assert!(
            *position >= 0,
            "case {index}: token {token} has position {position}"
        );
    }
    // The kernel scores a hidden token at `finfo(bf16).min` rather than -inf,
    // which the reference's softmax treats the same only while one key in the
    // row is visible; a fully hidden row would come out uniform here and NaN
    // there. Causality alone guarantees one, so this is a precondition of the
    // comparison rather than a fact about the model.
    for (slot, row_mask) in mask.chunks_exact(tokens).enumerate() {
        assert!(
            row_mask.iter().any(|visible| *visible != 0),
            "case {index}: row {slot} sees no key at all, and the softmax of an \
             all-hidden row is not the reference's"
        );
    }
    Case {
        index,
        tokens,
        rows,
        positions,
        queries,
        keys,
        values,
        mask,
        weights,
        out,
    }
}

impl Case {
    fn evaluated(&self) -> usize {
        self.rows.len()
    }

    /// Visible keys per evaluated row.
    fn visible_counts(&self) -> Vec<usize> {
        self.mask
            .chunks_exact(self.tokens)
            .map(|row| row.iter().filter(|v| **v != 0).count())
            .collect()
    }

    /// Whether this case's visibility is exactly causality — the first three
    /// cases are, and the boundary case is not, which is what makes the
    /// causal-mask control a control.
    fn mask_is_causal(&self) -> bool {
        self.rows.iter().enumerate().all(|(slot, row)| {
            let row_mask = &self.mask[slot * self.tokens..][..self.tokens];
            row_mask
                .iter()
                .enumerate()
                .all(|(token, visible)| (*visible != 0) == (token as i32 <= *row))
        })
    }
}

/// One semantic of the replay, either the fixture's own or a broken one that the
/// comparison must catch.
#[derive(Clone, Copy, Debug)]
enum Replay {
    /// The fixture's own inputs.
    Reference,
    /// The stored visibility replaced by plain causality, so the indexer's
    /// selection is dropped from the mask.
    CausalMask,
    /// The two KV heads exchanged in both key and value arrays, so every query
    /// head reads the other group's keys — a wrong GQA mapping.
    SwappedKvHeads,
    /// The gate half of the fused projection rows set to `bf16::MAX`, so
    /// `sigmoid` saturates at 1 and the output is the ungated context.
    UngatedOutput,
}

/// The two KV halves of every token exchanged.
fn swap_kv_heads(rows: &[bf16], geometry: &QsaAttnGeometry) -> Vec<bf16> {
    let width = geometry.kv_width();
    let head_dim = geometry.head_dim;
    let mut swapped = rows.to_vec();
    for row in swapped.chunks_exact_mut(width) {
        for dim in 0..head_dim {
            row.swap(dim, head_dim + dim);
        }
    }
    swapped
}

/// Every head's gate dims saturated, leaving its query dims alone.
fn saturate_gate(rows: &[bf16], geometry: &QsaAttnGeometry) -> Vec<bf16> {
    let width = geometry.query_width();
    let head_dim = geometry.head_dim;
    let mut saturated = rows.to_vec();
    for row in saturated.chunks_exact_mut(width) {
        for head in 0..geometry.heads {
            for gate in &mut row[head * 2 * head_dim + head_dim..][..head_dim] {
                *gate = bf16::MAX;
            }
        }
    }
    saturated
}

/// Plain causality over the evaluated rows: token `s` is visible to the row that
/// belongs to token `rows[slot]`.
fn causal_mask(case: &Case) -> Vec<u8> {
    let mut mask = vec![0u8; case.evaluated() * case.tokens];
    for (slot, row) in case.rows.iter().enumerate() {
        for (token, visible) in mask[slot * case.tokens..][..case.tokens]
            .iter_mut()
            .enumerate()
        {
            *visible = u8::from(token as i32 <= *row);
        }
    }
    mask
}

/// Runs one case through the kernel and brings both observables back.
fn replay(
    ctx: &DeviceContext,
    geometry: &QsaAttnGeometry,
    q_norm: &CudaSlice<bf16>,
    k_norm: &CudaSlice<bf16>,
    case: &Case,
    mode: Replay,
) -> (Vec<bf16>, Vec<bf16>) {
    let (queries, keys, values, mask) = match mode {
        Replay::Reference => (
            case.queries.clone(),
            case.keys.clone(),
            case.values.clone(),
            case.mask.clone(),
        ),
        Replay::CausalMask => (
            case.queries.clone(),
            case.keys.clone(),
            case.values.clone(),
            causal_mask(case),
        ),
        Replay::SwappedKvHeads => (
            case.queries.clone(),
            swap_kv_heads(&case.keys, geometry),
            swap_kv_heads(&case.values, geometry),
            case.mask.clone(),
        ),
        Replay::UngatedOutput => (
            saturate_gate(&case.queries, geometry),
            case.keys.clone(),
            case.values.clone(),
            case.mask.clone(),
        ),
    };

    let queries_dev = ctx.stream.clone_htod(&queries).expect("queries upload");
    let keys_dev = ctx.stream.clone_htod(&keys).expect("keys upload");
    let values_dev = ctx.stream.clone_htod(&values).expect("values upload");
    let mask_dev = ctx.stream.clone_htod(&mask).expect("mask upload");
    let rows_dev = ctx.stream.clone_htod(&case.rows).expect("rows upload");
    let positions_dev = ctx
        .stream
        .clone_htod(&case.positions)
        .expect("positions upload");
    let mut weights_dev = ctx
        .stream
        .clone_htod(&vec![
            SENTINEL;
            geometry.heads * case.evaluated() * case.tokens
        ])
        .expect("weight buffer");
    let mut out_dev = ctx
        .stream
        .clone_htod(&vec![SENTINEL; case.evaluated() * geometry.out_width()])
        .expect("output buffer");
    let mut normed_keys_dev = ctx
        .stream
        .alloc_zeros::<bf16>(case.tokens * geometry.kv_width())
        .expect("normed key workspace");

    qsa_attn_forward_into(
        ctx,
        geometry,
        &queries_dev,
        &keys_dev,
        &values_dev,
        &mask_dev,
        &rows_dev,
        &positions_dev,
        q_norm,
        k_norm,
        &mut weights_dev,
        &mut out_dev,
        &mut normed_keys_dev,
    )
    .expect("QSA attention forward");
    ctx.sync().expect("sync");
    (
        ctx.stream
            .clone_dtoh(&weights_dev)
            .expect("weight download"),
        ctx.stream.clone_dtoh(&out_dev).expect("output download"),
    )
}

/// What one compared tensor says: how much of it is bit-identical, the worst
/// element and where, the absolute term the relative bound cannot cover, and how
/// many elements miss by more than one and two bf16 ulps.
///
/// The ulp counts separate the two animals a max alone cannot: a handful of
/// one-ulp rounding flips is this kernel's float32 reduction order against the
/// reference's, while anything many ulps out is a semantic difference.
#[derive(Clone, Copy)]
struct Stats {
    compared: usize,
    exact: usize,
    max_abs: f32,
    /// Worst `abs(got - want) / abs(want)`, over the elements whose reference is
    /// big enough for a relative reading to mean anything.
    max_rel: f32,
    /// The smallest absolute term that would cover every element under this
    /// bound's relative term: `max(abs(delta) - rtol * abs(want))`.
    atol_needed: f32,
    over_ulp: usize,
    over_2ulp: usize,
    max_scaled: f32,
    worst: (usize, f32, f32),
}

/// One bf16 quantum at a reference value: the distance to the next representable
/// magnitude, which is what a single rounding decision anywhere in the chain is
/// worth at that value.
fn bf16_ulp(reference: f32) -> f32 {
    let magnitude = bf16::from_f32(reference.abs());
    f32::from(bf16::from_bits(magnitude.to_bits() + 1)) - f32::from(magnitude)
}

/// `abs(want)` below which a relative reading is noise: case 3's outputs cancel
/// down through zero, and there the absolute term is the whole statement.
const REL_FLOOR: f32 = 1.0e-3;

fn stats(got: &[bf16], want: &[bf16], atol: f32, rtol: f32, what: &str) -> Stats {
    assert_eq!(got.len(), want.len(), "{what}: compared runs differ");
    let mut summary = Stats {
        compared: got.len(),
        exact: 0,
        max_abs: 0.0,
        max_rel: 0.0,
        atol_needed: 0.0,
        over_ulp: 0,
        over_2ulp: 0,
        max_scaled: 0.0,
        worst: (0, 0.0, 0.0),
    };
    for (index, (got, reference)) in got.iter().zip(want).enumerate() {
        let got = f32::from(*got);
        let reference = f32::from(*reference);
        assert!(
            got.is_finite() && reference.is_finite(),
            "{what} element {index} holds {got} against the reference's \
             {reference}; a NaN on this side is either the sentinel, so the \
             launch never wrote it, or a softmax over a row with no visible key"
        );
        let delta = (got - reference).abs();
        summary.exact += usize::from(delta == 0.0);
        summary.max_abs = summary.max_abs.max(delta);
        if reference.abs() > REL_FLOOR {
            summary.max_rel = summary.max_rel.max(delta / reference.abs());
        }
        summary.atol_needed = summary
            .atol_needed
            .max(delta - rtol * reference.abs())
            .max(0.0);
        let ulp = bf16_ulp(reference);
        summary.over_ulp += usize::from(delta > ulp);
        summary.over_2ulp += usize::from(delta > 2.0 * ulp);
        let scaled = delta / (atol + rtol * reference.abs());
        if scaled > summary.max_scaled {
            summary.max_scaled = scaled;
            summary.worst = (index, got, reference);
        }
    }
    summary
}

fn assert_within(summary: Stats, atol: f32, rtol: f32, what: &str, case: usize) {
    assert!(
        summary.max_scaled <= 1.0,
        "case {case} {what}: element {} scored {} against the reference's {}, \
         {:.3e} off; the bound is ATOL {atol} + RTOL {rtol} * |want|, {:.3} of it",
        summary.worst.0,
        summary.worst.1,
        summary.worst.2,
        (summary.worst.1 - summary.worst.2).abs(),
        summary.max_scaled
    );
}

fn report(label: &str, summary: Stats, atol: f32, rtol: f32) {
    println!(
        "  {label}: {} elements, {} bit-identical ({:.4}%), max abs {:.3e}, max \
         rel {:.3e} over |want| > {REL_FLOOR}, atol needed under rtol {rtol} \
         {:.3e}, over 1 ulp {}, over 2 ulp {}, worst at {} ({:.9} against \
         {:.9}), bound fraction {:.4} (ATOL {atol})",
        summary.compared,
        summary.exact,
        100.0 * summary.exact as f64 / summary.compared as f64,
        summary.max_abs,
        summary.max_rel,
        summary.atol_needed,
        summary.over_ulp,
        summary.over_2ulp,
        summary.worst.0,
        summary.worst.1,
        summary.worst.2,
        summary.max_scaled,
    );
}

/// One case's two observables against the reference, compared and reported but
/// not asserted: the negative controls need the same numbers on the other side
/// of the line.
fn measure(
    ctx: &DeviceContext,
    geometry: &QsaAttnGeometry,
    q_norm: &CudaSlice<bf16>,
    k_norm: &CudaSlice<bf16>,
    case: &Case,
    mode: Replay,
) -> (Stats, Stats) {
    let (weights, out) = replay(ctx, geometry, q_norm, k_norm, case, mode);
    (
        stats(
            &weights,
            &case.weights,
            WEIGHT_ATOL,
            WEIGHT_RTOL,
            &format!("case {} weights", case.index),
        ),
        stats(
            &out,
            &case.out,
            OUT_ATOL,
            OUT_RTOL,
            &format!("case {} output", case.index),
        ),
    )
}

/// The two per-head norm weights on the device, held against the compiled head
/// width first: they are the only weights the fixture carries, and a narrower
/// one would silently read a neighbour's.
fn upload_norms(
    ctx: &DeviceContext,
    fixture: &SafeTensors<'_>,
    geometry: &QsaAttnGeometry,
) -> (CudaSlice<bf16>, CudaSlice<bf16>) {
    let (_, q_norm) = bf16_tensor(fixture, "q_norm");
    let (_, k_norm) = bf16_tensor(fixture, "k_norm");
    assert_eq!(q_norm.len(), geometry.head_dim, "q_norm width");
    assert_eq!(k_norm.len(), geometry.head_dim, "k_norm width");
    (
        ctx.stream.clone_htod(&q_norm).expect("q norm upload"),
        ctx.stream.clone_htod(&k_norm).expect("k norm upload"),
    )
}

/// The fixture's provenance, independent of any device: this half must never be
/// skippable.
#[test]
fn fixture_records_the_frozen_reference() {
    let bytes = std::fs::read(fixture_path()).expect("read fixture");
    assert_provenance(&bytes);
}

/// The kernel's own half of the refusal contract, likewise device-free: the
/// entry point validates before it forms a launch, so its refusals are pinnable
/// without a device and must never be skippable. `rotary_dim` and `rope_theta`
/// are compiled constants nothing else can be held against, and the token cap is
/// the shared-memory budget a longer context would overflow.
///
/// Every call below leaves through a guard, before the first launch, and a
/// launch is the only thing that would touch the pointers; that is why they can
/// dangle and why this needs no device. The two zero-extent calls come first as
/// the positive control — the compiled geometry with an empty extent, which the
/// guards must let through — because an entry point that refused everything
/// would make every refusal below vacuous.
#[test]
fn the_entry_point_refuses_a_geometry_it_was_not_compiled_for() {
    use cudarc::driver::sys::CUresult;
    use pegainfer_kernels::ffi::Half;
    use pegainfer_kernels::ffi::qwen4_qsa_attn_forward_cuda as entry;

    // Dangling and well-aligned, never dereferenced.
    let half = std::ptr::NonNull::<Half>::dangling().as_ptr();
    let byte = std::ptr::NonNull::<u8>::dangling().as_ptr();
    let int = std::ptr::NonNull::<i32>::dangling().as_ptr();
    let invalid = CUresult::CUDA_ERROR_INVALID_VALUE as i32;
    let success = CUresult::CUDA_SUCCESS as i32;

    // (query_rows, tokens, rotary_dim, rope_theta, rms_eps)
    let call = |args: (i32, i32, i32, f32, f32)| {
        let (query_rows, tokens, rotary_dim, rope_theta, rms_eps) = args;
        (unsafe {
            entry(
                half,
                half,
                half,
                byte,
                int,
                int,
                half,
                half,
                half,
                half,
                half,
                query_rows,
                tokens,
                rotary_dim,
                rope_theta,
                rms_eps,
                std::ptr::null_mut(),
            )
        }) as i32
    };

    // Zero extent, so the guards are the whole of the call and no launch follows.
    assert_eq!(
        call((0, 8, 64, 1.0e7, 1.0e-6)),
        success,
        "the compiled geometry must pass the guards"
    );
    assert_eq!(
        call((1, 0, 64, 1.0e7, 1.0e-6)),
        success,
        "an empty call must pass the guards"
    );

    for (what, result) in [
        (
            "a rotary split that is not the compiled 64 of the 256-dim head",
            call((0, 8, 32, 1.0e7, 1.0e-6)),
        ),
        (
            "a rotary split over the whole attention head",
            call((0, 8, 256, 1.0e7, 1.0e-6)),
        ),
        (
            "a rope_theta that is not the compiled 1e7",
            call((0, 8, 64, 1.0e6, 1.0e-6)),
        ),
        (
            // One f32 ulp: the geometry's own bit-equality check, from the
            // kernel's side, and what a config drifting in the last place hits.
            "a rope_theta one ulp from the compiled 1e7",
            call((0, 8, 64, f32::from_bits(1.0e7f32.to_bits() + 1), 1.0e-6)),
        ),
        ("a negative rms_eps", call((0, 8, 64, 1.0e7, -1.0e-6))),
        (
            // `!(rms_eps >= 0)` refuses NaN, which would otherwise poison every
            // norm silently rather than fail loudly.
            "a NaN rms_eps",
            call((0, 8, 64, 1.0e7, f32::NAN)),
        ),
        (
            "a negative query-row count",
            call((-1, 8, 64, 1.0e7, 1.0e-6)),
        ),
        ("a negative token count", call((0, -1, 64, 1.0e7, 1.0e-6))),
        (
            // One row of float32 scores lives in dynamic shared memory, so a
            // key array past the budget is a launch that cannot be formed.
            "a key array past the shared-memory budget",
            call((
                1,
                (QsaAttnGeometry::MAX_TOKENS + 1) as i32,
                64,
                1.0e7,
                1.0e-6,
            )),
        ),
    ] {
        assert_eq!(result, invalid, "{what} must be refused");
    }

    // One position per kind of argument — a fused-projection input, the
    // visibility, the two outputs and the workspace — so the chain is pinned to
    // cover the pointers, not just the extents. All five entries are `*mut`, the
    // type the dangling handles above carry, so the rows share one type and the
    // const parameters take them by coercion.
    for (what, queries, mask, attn_weights, attn_out, normed_keys) in [
        (
            "a null queries",
            std::ptr::null_mut(),
            byte,
            half,
            half,
            half,
        ),
        ("a null mask", half, std::ptr::null_mut(), half, half, half),
        (
            "a null attn_weights",
            half,
            byte,
            std::ptr::null_mut(),
            half,
            half,
        ),
        (
            "a null attn_out",
            half,
            byte,
            half,
            std::ptr::null_mut(),
            half,
        ),
        (
            "a null normed_keys workspace",
            half,
            byte,
            half,
            half,
            std::ptr::null_mut(),
        ),
    ] {
        let result = unsafe {
            entry(
                queries,
                half,
                half,
                mask,
                int,
                int,
                half,
                half,
                attn_weights,
                attn_out,
                normed_keys,
                0,
                8,
                64,
                1.0e7,
                1.0e-6,
                std::ptr::null_mut(),
            )
        } as i32;
        assert_eq!(result, invalid, "{what} must be refused");
    }
}

/// The token cap is a launch limit, not a policy: one row of float32 scores lives
/// in dynamic shared memory, and the cap is the longest row that fits what a launch
/// may ask for without opting in. This runs exactly that shape — `MAX_TOKENS`
/// keys, the widest shared memory this kernel ever requests, and positions past
/// anything the fixture carries — so the cap is taken at the cap instead of at
/// 2055 and at a refusal. The device-free test above pins the refusing side with
/// the same constant, which is what keeps the Rust cap and the CUDA one from
/// drifting apart: raise either alone and one of the two tests fails.
///
/// The weights' row sums are the check that the largest launch is not merely
/// legal but complete: a softmax that never ran, or ran over a truncated row,
/// would not sum to 1 in any of the 24 heads.
#[test]
fn the_token_cap_is_a_launch_it_can_actually_form() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let geometry = QsaAttnGeometry::qwen38_flash_next();
    let tokens = QsaAttnGeometry::MAX_TOKENS;

    let queries_dev = ctx
        .stream
        .clone_htod(&common::fill(0x5151_5151, geometry.query_width()))
        .expect("queries upload");
    let keys_dev = ctx
        .stream
        .clone_htod(&common::fill(0x6161_6161, tokens * geometry.kv_width()))
        .expect("keys upload");
    let values_dev = ctx
        .stream
        .clone_htod(&common::fill(0x7171_7171, tokens * geometry.kv_width()))
        .expect("values upload");
    let q_norm_dev = ctx
        .stream
        .clone_htod(&common::fill(0x8181_8181, geometry.head_dim))
        .expect("q norm upload");
    let k_norm_dev = ctx
        .stream
        .clone_htod(&common::fill(0x9191_9191, geometry.head_dim))
        .expect("k norm upload");
    let positions: Vec<i32> = (0..tokens).map(|position| position as i32).collect();
    let positions_dev = ctx.stream.clone_htod(&positions).expect("positions upload");
    let rows_dev = ctx.stream.clone_htod(&[0i32]).expect("rows upload");
    // Every key visible to the one evaluated row, so the softmax runs over the
    // whole row of scores.
    let mask_dev = ctx
        .stream
        .clone_htod(&vec![1u8; tokens])
        .expect("mask upload");
    let mut weights_dev = ctx
        .stream
        .clone_htod(&vec![SENTINEL; geometry.heads * tokens])
        .expect("weight buffer");
    let mut out_dev = ctx
        .stream
        .clone_htod(&vec![SENTINEL; geometry.out_width()])
        .expect("output buffer");
    let mut normed_keys_dev = ctx
        .stream
        .alloc_zeros::<bf16>(tokens * geometry.kv_width())
        .expect("normed key workspace");

    qsa_attn_forward_into(
        &ctx,
        &geometry,
        &queries_dev,
        &keys_dev,
        &values_dev,
        &mask_dev,
        &rows_dev,
        &positions_dev,
        &q_norm_dev,
        &k_norm_dev,
        &mut weights_dev,
        &mut out_dev,
        &mut normed_keys_dev,
    )
    .unwrap_or_else(|e| panic!("{tokens} tokens must be launchable: {e}"));
    ctx.sync().expect("sync");

    let weights = ctx.stream.clone_dtoh(&weights_dev).expect("weights");
    let out = ctx.stream.clone_dtoh(&out_dev).expect("output");
    for head in 0..geometry.heads {
        let row = &weights[head * tokens..][..tokens];
        assert!(
            row.iter().all(|weight| f32::from(*weight).is_finite()),
            "head {head} has a weight that is not finite"
        );
        let sum: f32 = row.iter().map(|weight| f32::from(*weight)).sum();
        // bf16 rounds each of the 8192 weights, ~1.2e-4 apiece, so the row sum
        // carries their rounding and nothing else.
        assert!(
            (sum - 1.0).abs() < 2.0e-3,
            "head {head}: its {tokens} weights sum to {sum}, so the softmax did \
             not run over the whole row"
        );
    }
    assert!(
        out.iter().all(|element| f32::from(*element).is_finite()),
        "an output element at the token cap is not finite"
    );
    println!(
        "token cap {tokens}: {} weights over {} heads and {} output elements, \
         all finite, every head's row sum within 2e-3 of 1",
        weights.len(),
        geometry.heads,
        out.len()
    );
}

#[test]
fn qsa_attention_matches_the_reference() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let bytes = std::fs::read(fixture_path()).expect("read fixture");
    assert_provenance(&bytes);
    let fixture = SafeTensors::deserialize(&bytes).expect("parse fixture");
    let geometry = QsaAttnGeometry::qwen38_flash_next();
    let (q_norm, k_norm) = upload_norms(&ctx, &fixture, &geometry);

    let mut weight_elements = 0usize;
    let mut out_elements = 0usize;
    let mut worst_weight = 0.0f32;
    let mut worst_out = 0.0f32;
    let mut subset = 0usize;
    let mut sparse = 0usize;
    let mut boundary = 0usize;
    for index in 0..CASES {
        let case = load_case(&fixture, index, &geometry);
        let visible = case.visible_counts();
        let is_causal = case.mask_is_causal();
        // Coverage the fixture claims, asserted rather than assumed: a case
        // whose evaluated rows are a strict subset of its key array, and a case
        // whose visibility is not causality — the indexer's selection actually
        // dropping keys the row could otherwise see.
        subset += usize::from(case.evaluated() < case.tokens);
        sparse += usize::from(!is_causal);
        boundary += usize::from(
            !is_causal
                && visible
                    .iter()
                    .zip(visible.iter().skip(1))
                    .any(|(before, after)| after < before),
        );

        let (weights, out) = measure(&ctx, &geometry, &q_norm, &k_norm, &case, Replay::Reference);
        report("weights", weights, WEIGHT_ATOL, WEIGHT_RTOL);
        report("output ", out, OUT_ATOL, OUT_RTOL);
        assert_within(weights, WEIGHT_ATOL, WEIGHT_RTOL, "weights", index);
        assert_within(out, OUT_ATOL, OUT_RTOL, "output", index);
        println!(
            "case {index}: {} rows of {} tokens ({} weight elements, {} output \
             elements), visible per row {:?}, mask is causality: {is_causal}",
            case.evaluated(),
            case.tokens,
            geometry.heads * case.evaluated() * case.tokens,
            case.evaluated() * geometry.out_width(),
            visible,
        );
        weight_elements += weights.compared;
        out_elements += out.compared;
        worst_weight = worst_weight.max(weights.max_scaled);
        worst_out = worst_out.max(out.max_scaled);
    }
    println!(
        "parity over {CASES} cases: {weight_elements} weights and {out_elements} \
         output elements, worst bound fraction {worst_weight:.4} (weights, ATOL \
         {WEIGHT_ATOL} RTOL {WEIGHT_RTOL}) and {worst_out:.4} (output, ATOL \
         {OUT_ATOL} RTOL {OUT_RTOL})"
    );
    assert_eq!(
        subset, 1,
        "exactly one case must evaluate a strict subset of its key array, or the \
         row-subset path this gate is the only reader of is not covered"
    );
    assert_eq!(
        sparse, 1,
        "exactly one case's visibility must be sparser than causality, or the \
         mask this gate replays could be the causal one"
    );
    assert_eq!(
        boundary, 1,
        "exactly one case sits at the budget boundary, where the visible count \
         starts falling as the sequence grows"
    );
}

/// Control (a): the stored visibility replaced by plain causality. The first
/// three cases' masks *are* causal, so they must still match — that is what makes
/// the fourth case's failure the indexer's selection and not something else this
/// control perturbs.
#[test]
fn a_causal_mask_alone_loses_the_budget_boundary() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let bytes = std::fs::read(fixture_path()).expect("read fixture");
    assert_provenance(&bytes);
    let fixture = SafeTensors::deserialize(&bytes).expect("parse fixture");
    let geometry = QsaAttnGeometry::qwen38_flash_next();
    let (q_norm, k_norm) = upload_norms(&ctx, &fixture, &geometry);

    let mut broken = 0usize;
    for index in 0..CASES {
        let case = load_case(&fixture, index, &geometry);
        let (weights, out) = measure(&ctx, &geometry, &q_norm, &k_norm, &case, Replay::CausalMask);
        let fails = weights.max_scaled > 1.0 || out.max_scaled > 1.0;
        println!(
            "case {index} with a causal mask: weights {:.3} of the bound, output \
             {:.3} of the bound, mask is causality: {} -> {}",
            weights.max_scaled,
            out.max_scaled,
            case.mask_is_causal(),
            if fails { "MISMATCH" } else { "still parity" }
        );
        if case.mask_is_causal() {
            assert!(
                !fails,
                "case {index}: its stored mask is causality, so replacing it with \
                 causality must change nothing, and it moved the weights to {} \
                 of the bound",
                weights.max_scaled
            );
        } else {
            assert!(
                fails,
                "case {index}: dropping the indexer's selection from the mask \
                 must break the comparison, and both observables stayed inside \
                 the bound ({:.3} and {:.3} of it) — the gate would pass without \
                 the sparse mask",
                weights.max_scaled, out.max_scaled
            );
            broken += 1;
        }
    }
    assert_eq!(broken, 1, "the boundary case is the only sparse one");
}

/// Control (b): a wrong GQA mapping. Exchanging the two KV heads sends every
/// query group to the other head's keys and values, which is the mistake a
/// `h % kv_heads` or `h / group` mix-up makes, and no case may survive it.
#[test]
fn a_wrong_gqa_mapping_does_not_reproduce_the_reference() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let bytes = std::fs::read(fixture_path()).expect("read fixture");
    assert_provenance(&bytes);
    let fixture = SafeTensors::deserialize(&bytes).expect("parse fixture");
    let geometry = QsaAttnGeometry::qwen38_flash_next();
    let (q_norm, k_norm) = upload_norms(&ctx, &fixture, &geometry);

    for index in 0..CASES {
        let case = load_case(&fixture, index, &geometry);
        let (weights, out) = measure(
            &ctx,
            &geometry,
            &q_norm,
            &k_norm,
            &case,
            Replay::SwappedKvHeads,
        );
        println!(
            "case {index} with the KV heads exchanged: weights {:.3} of the bound \
             ({} bit-identical), output {:.3} of the bound ({} bit-identical)",
            weights.max_scaled, weights.exact, out.max_scaled, out.exact
        );
        assert!(
            weights.max_scaled > 1.0 && out.max_scaled > 1.0,
            "case {index}: reading the other KV head must break both observables, \
             and they came in at {:.3} and {:.3} of the bound — the gate would \
             pass with the group mapping wrong",
            weights.max_scaled,
            out.max_scaled
        );
    }
}

/// Control (c): the output gate dropped. Saturating it makes `sigmoid` exactly 1,
/// so the compared output is the ungated context. The weights must be untouched
/// — the gate sits after them — and the output must break in every case.
#[test]
fn an_ungated_output_does_not_reproduce_the_reference() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let bytes = std::fs::read(fixture_path()).expect("read fixture");
    assert_provenance(&bytes);
    let fixture = SafeTensors::deserialize(&bytes).expect("parse fixture");
    let geometry = QsaAttnGeometry::qwen38_flash_next();
    let (q_norm, k_norm) = upload_norms(&ctx, &fixture, &geometry);

    for index in 0..CASES {
        let case = load_case(&fixture, index, &geometry);
        let (weights, out) = measure(
            &ctx,
            &geometry,
            &q_norm,
            &k_norm,
            &case,
            Replay::UngatedOutput,
        );
        println!(
            "case {index} with the gate saturated: weights {:.3} of the bound, \
             output {:.3} of the bound ({} bit-identical)",
            weights.max_scaled, out.max_scaled, out.exact
        );
        assert_within(weights, WEIGHT_ATOL, WEIGHT_RTOL, "weights", index);
        assert!(
            out.max_scaled > 1.0,
            "case {index}: dropping the output gate must break the output, and it \
             came in at {:.3} of the bound — the gate is not load-bearing for \
             this gate",
            out.max_scaled
        );
    }
}
