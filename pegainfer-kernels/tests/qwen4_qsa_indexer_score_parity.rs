//! Qwen3.8-Flash-Next QSA token-indexer block scores against the frozen
//! reference golden.
//!
//! `test_data/qwen38fn-qsa-indexer-golden.safetensors` was dumped from
//! transformers 5.16.1 `qwen4_exp` by
//! `tools/accuracy/dump_qwen38fn_qsa_indexer_golden.py`. Each case carries the
//! indexer's inputs *after* `index_qk_proj` plus the reference's per-row block
//! scores, so this gate replays those inputs through the kernel and compares.
//! Cases cover the tail (5), a block-aligned short input (8), a medium one
//! (64), the truncation branch under a shrunk budget (14, 16, 20) and the real
//! budget boundary at 2049..2054 visible tokens.
//!
//! Provenance is asserted, not assumed: the fixture records the reference
//! version and the sha256 of the frozen config it was dumped from, and that
//! config's indexer geometry is held against the geometry this crate compiles
//! for. Without a device it skips; `PEGAINFER_REQUIRE_GPU=1` turns that into a
//! failure.
//!
//!   cargo test --release -p pegainfer-kernels --features qwen4 \
//!     --test qwen4_qsa_indexer_score_parity -- --nocapture

#![cfg(feature = "qwen4")]

mod common;

use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::QsaIndexerGeometry;
use pegainfer_kernels::ops::qsa_indexer_block_scores_into;
use pegainfer_kernels::tensor::DeviceContext;
use safetensors::SafeTensors;
use sha2::Digest;
use sha2::Sha256;

const FIXTURE: &str = "qwen38fn-qsa-indexer-golden.safetensors";
const METADATA_KEY: &str = "qwen4_qsa_indexer_golden";
/// Frozen config the golden was dumped from, relative to the workspace root.
const CONFIG: &str = "pegainfer-qwen4/tests/frozen_config.json";
const FORMAT_VERSION: u64 = 2;
/// The reference implementation the golden's semantics are frozen against.
const REF_TRANSFORMERS: &str = "5.16.1";
const CASES: usize = 7;

/// Score parity bound, `abs(got - want) <= ATOL + RTOL * abs(want)`.
///
/// Six cases land within 9.6e-7 — every bf16 rounding decision the reference
/// makes is reproduced bit-for-bit. The residual sits in the seventh (the only
/// case with positions past 63): 152 of 3076 scores above 1e-4, worst 2.4e-3,
/// sign-balanced with a -5.2e-7 bias — the shape of a flipped bf16 rounding
/// decision, not a semantic difference. The reference rounds its float32
/// reductions in a torch-defined order; where a value sits within one f32 ulp
/// of a bf16 midpoint the kernel's butterfly tree rounds apart from it, and
/// one flipped element moves every score it touches. The reference does the
/// same to itself: re-deriving case 2 on x86 (torch 2.14.1 vs the golden's
/// 2.14.0/arm64) moves one of its 496 scores by 2.2e-3, three orders more than
/// this kernel's 9.5e-7 there.
///
/// So the bound holds ~6 flips at a typical score of ~2, against a measured
/// worst of 0.16 of it, while any real fault — a block off by one, a missing
/// relu, an unrotated half — moves a score by ~1, two orders out. Selection
/// must treat a block tied at the cut as an equally valid pick, for the same
/// reason.
const ATOL: f32 = 5.0e-3;
const RTOL: f32 = 5.0e-3;

/// Written over the score buffer before every launch, to prove the kernel writes
/// exactly the valid prefix of each row and nothing else.
const SENTINEL: f32 = -12345.0;

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
/// reference version, the frozen config's bytes and the geometry this crate
/// compiles for. Returns the manifest.
fn assert_provenance(bytes: &[u8]) -> serde_json::Value {
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
    let geometry = QsaIndexerGeometry::qwen38_flash_next();
    let field = |name: &str| text[name].as_u64().unwrap_or_else(|| panic!("{name}"));
    assert_eq!(field("indexer_n_heads"), geometry.heads as u64);
    assert_eq!(field("indexer_kv_heads"), geometry.kv_heads as u64);
    assert_eq!(field("indexer_head_dim"), geometry.head_dim as u64);
    assert_eq!(
        field("indexer_compress_ratio"),
        geometry.compress_ratio as u64
    );
    assert_eq!(field("indexer_budget"), geometry.budget as u64);
    let rope_theta = text["rope_parameters"]["rope_theta"]
        .as_f64()
        .expect("rope_theta");
    assert!(
        same_f32(rope_theta as f32, geometry.rope_theta),
        "rope_theta {rope_theta} is not the compiled {}",
        geometry.rope_theta
    );
    let rms_eps = text["rms_norm_eps"].as_f64().expect("rms_norm_eps");
    assert!(
        same_f32(rms_eps as f32, geometry.rms_eps),
        "rms_norm_eps {rms_eps} is not the compiled {}",
        geometry.rms_eps
    );
    // rotary_dim = the attention head's head_dim * partial_rotary_factor, and
    // the indexer head is long enough to hold it.
    let head_dim = field("head_dim") as f64;
    let partial = text["rope_parameters"]["partial_rotary_factor"]
        .as_f64()
        .expect("partial_rotary_factor");
    assert!(
        same_f64(head_dim * partial, geometry.rotary_dim as f64),
        "rotary_dim {} is not head_dim {head_dim} * partial_rotary_factor {partial}",
        geometry.rotary_dim
    );
    assert!(geometry.rotary_dim <= geometry.head_dim);
    let full_layers = text["layer_types"]
        .as_array()
        .expect("layer_types")
        .iter()
        .filter(|kind| kind.as_str() == Some("full_attention"))
        .count();
    assert_eq!(full_layers, 12, "the checkpoint's sparse-attention layers");
    // Recorded, not asserted: the host the golden was dumped on decides the
    // order its float32 reductions round in, which is where the parity bound's
    // residual comes from, and any other host's order is an equally valid
    // reference. Printed because the bound's justification leans on it and it
    // is otherwise invisible to whoever reads this gate's output.
    println!(
        "golden provenance: transformers {}, torch {}, {}",
        manifest["transformers_version"].as_str().unwrap_or("?"),
        manifest["torch_version"].as_str().unwrap_or("?"),
        manifest["platform"].as_str().unwrap_or("?"),
    );
    manifest
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

fn f32_tensor(fixture: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<f32>) {
    let view = fixture
        .tensor(name)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    let host = view
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
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

fn budget_tensor(fixture: &SafeTensors<'_>, name: &str) -> u32 {
    let (shape, host) = i32_tensor(fixture, name);
    assert_eq!(shape, vec![1], "{name} shape");
    u32::try_from(host[0]).unwrap_or_else(|_| panic!("{name} is negative"))
}

/// One case's inputs and the reference's answer for its evaluated rows.
struct Case {
    index: usize,
    tokens: usize,
    budget: u32,
    keys: Vec<bf16>,
    queries: Vec<bf16>,
    rows: Vec<i32>,
    reference: Vec<f32>,
    score_cols: usize,
}

fn load_case(fixture: &SafeTensors<'_>, index: usize, geometry: &QsaIndexerGeometry) -> Case {
    let (key_shape, keys) = bf16_tensor(fixture, &format!("case{index}_keys"));
    let (query_shape, queries) = bf16_tensor(fixture, &format!("case{index}_queries"));
    let (rows_shape, rows) = i32_tensor(fixture, &format!("case{index}_rows"));
    let (score_shape, reference) = f32_tensor(fixture, &format!("case{index}_scores"));

    let tokens = key_shape[0];
    assert_eq!(key_shape, vec![tokens, geometry.head_dim]);
    let query_rows = query_shape[0];
    assert_eq!(
        query_shape,
        vec![query_rows, geometry.heads, geometry.head_dim]
    );
    assert_eq!(rows_shape, vec![query_rows]);
    assert_eq!(score_shape.len(), 2);
    assert_eq!(score_shape[0], query_rows);
    assert!(
        score_shape[1] >= tokens / geometry.compress_ratio,
        "case {index}: {} score columns cannot hold the {} blocks of {tokens} tokens",
        score_shape[1],
        tokens / geometry.compress_ratio
    );
    let case = Case {
        index,
        tokens,
        budget: budget_tensor(fixture, &format!("case{index}_budget")),
        keys,
        queries,
        rows,
        reference,
        score_cols: score_shape[1],
    };
    assert!(
        case.budget.is_multiple_of(geometry.compress_ratio as u32),
        "case {index}: the reference requires the budget to divide by the \
         compress ratio, and its block_topk is that quotient"
    );
    case
}

/// The fused `index_qk_proj` rows the kernel reads: the keys for every token,
/// then the queries at the rows the reference evaluated.
fn fused_rows(case: &Case, geometry: &QsaIndexerGeometry) -> Vec<bf16> {
    let width = geometry.fused_width();
    let q_width = geometry.heads * geometry.head_dim;
    let mut fused = vec![bf16::ZERO; case.tokens * width];
    for token in 0..case.tokens {
        let key_row = &case.keys[token * geometry.head_dim..][..geometry.head_dim];
        fused[token * width + q_width..][..geometry.head_dim].copy_from_slice(key_row);
    }
    for (slot, row) in case.rows.iter().enumerate() {
        let row = usize::try_from(*row).expect("non-negative query row");
        assert!(row < case.tokens, "case {}: row {row}", case.index);
        fused[row * width..][..q_width].copy_from_slice(&case.queries[slot * q_width..][..q_width]);
    }
    fused
}

/// This case's visible counts: causal text with no prefix eviction, so a query
/// at position p sees tokens [0, p]. The kernel takes the visible count
/// separately because the reference's block grid starts at the first visible
/// token, which this line serves as token 0.
fn visible_counts(case: &Case) -> Vec<i32> {
    case.rows
        .iter()
        .map(|row| {
            let visible = row.checked_add(1).expect("visible length");
            assert!(
                usize::try_from(visible).expect("visible") <= case.tokens,
                "case {}: row {row} sees past the key array",
                case.index
            );
            visible
        })
        .collect()
}

/// One case's parity: how much was compared, the worst element and where, and
/// the shape of the residual — the sign split and the bias catch a
/// *systematic* error (a lean one way, deviations piling up in one row)
/// that a max alone cannot.
struct CaseSummary {
    compared: usize,
    max_abs: f32,
    max_scaled: f32,
    /// `(position, block, scored, reference)`
    worst: (i32, usize, f32, f32),
    over_1e4: usize,
    over_1e5: usize,
    pos_1e4: usize,
    /// `(position, deviating blocks in that row)`
    widest_row: (i32, usize),
    bias: f32,
}

/// Runs one case.
fn check_case(
    ctx: &DeviceContext,
    geometry: &QsaIndexerGeometry,
    q_norm: &CudaSlice<bf16>,
    k_norm: &CudaSlice<bf16>,
    case: &Case,
) -> CaseSummary {
    let query_rows = case.rows.len();
    let score_cols = case.score_cols;
    let block_cols = geometry.max_blocks(case.tokens);
    let visible_len = visible_counts(case);

    let fused = ctx
        .stream
        .clone_htod(&fused_rows(case, geometry))
        .expect("fused upload");
    let rows_dev = ctx.stream.clone_htod(&case.rows).expect("rows upload");
    let visible_dev = ctx.stream.clone_htod(&visible_len).expect("visible upload");
    let mut scores_dev = ctx
        .stream
        .clone_htod(&vec![SENTINEL; query_rows * score_cols])
        .expect("score buffer");
    let mut pooled_dev = ctx
        .stream
        .alloc_zeros::<bf16>(block_cols * geometry.head_dim)
        .expect("pooled workspace");

    qsa_indexer_block_scores_into(
        ctx,
        geometry,
        &fused,
        q_norm,
        k_norm,
        &rows_dev,
        &visible_dev,
        case.tokens,
        &mut scores_dev,
        score_cols,
        &mut pooled_dev,
    )
    .expect("QSA indexer block scores");
    ctx.sync().expect("sync");
    let got = ctx.stream.clone_dtoh(&scores_dev).expect("score download");

    let mut compared = 0usize;
    let mut max_abs = 0.0f32;
    let mut max_scaled = 0.0f32;
    let mut worst = (0i32, 0usize, 0.0f32, 0.0f32);
    let mut over_1e4 = 0usize;
    let mut over_1e5 = 0usize;
    let mut pos_1e4 = 0usize;
    let mut widest_row = (0i32, 0usize);
    let mut signed_sum = 0.0f64;
    for (slot, row) in case.rows.iter().enumerate() {
        let blocks = usize::try_from(visible_len[slot]).expect("visible") / geometry.compress_ratio;
        assert!(blocks <= block_cols, "case {}: beyond the pool", case.index);
        let row_ref = &case.reference[slot * score_cols..][..score_cols];
        let row_got = &got[slot * score_cols..][..score_cols];
        // The reference pads its un-computed columns with NaN; the gate reads
        // the explicit prefix instead of comparing through it.
        for (block, want) in row_ref.iter().enumerate().skip(blocks) {
            assert!(
                want.is_nan(),
                "case {}: row {row} column {block} is not the reference's padding",
                case.index
            );
        }
        // The kernel writes the prefix and leaves the rest bit-for-bit as it
        // was, so the sentinel check is a bit comparison.
        for (block, x) in row_got.iter().enumerate().skip(blocks) {
            assert!(
                same_f32(*x, SENTINEL),
                "case {}: row {row} column {block} is past the valid prefix of {blocks} \
                 and must be left untouched, but holds {x}",
                case.index
            );
        }
        let mut case_worst: Option<(usize, f32, f32, f32)> = None;
        let mut row_over_1e4 = 0usize;
        for (block, (x, want)) in row_got[..blocks].iter().zip(&row_ref[..blocks]).enumerate() {
            assert!(
                x.is_finite() && want.is_finite(),
                "case {}: row {row} block {block} scored {x} against {want}",
                case.index
            );
            let delta = x - want;
            let abs = delta.abs();
            let scaled = abs / (ATOL + RTOL * want.abs());
            max_abs = max_abs.max(abs);
            signed_sum += f64::from(delta);
            over_1e4 += usize::from(abs > 1.0e-4);
            over_1e5 += usize::from(abs > 1.0e-5);
            pos_1e4 += usize::from(abs > 1.0e-4 && delta > 0.0);
            row_over_1e4 += usize::from(abs > 1.0e-4);
            if scaled > max_scaled {
                max_scaled = scaled;
                case_worst = Some((block, *x, *want, abs));
                worst = (*row, block, *x, *want);
            }
        }
        if row_over_1e4 > widest_row.1 {
            widest_row = (*row, row_over_1e4);
        }
        if max_scaled > 1.0 {
            let (block, x, want, abs) = case_worst.expect("a worst element");
            panic!(
                "case {}: row {row} block {block} scored {x} against the reference's \
                 {want}, {abs:.3e} off; the bound is ATOL {ATOL} + RTOL {RTOL} * |want|",
                case.index
            );
        }
        compared += blocks;
    }
    assert!(compared > 0, "case {}: nothing compared", case.index);
    CaseSummary {
        compared,
        max_abs,
        max_scaled,
        worst,
        over_1e4,
        over_1e5,
        pos_1e4,
        widest_row,
        bias: (signed_sum / compared as f64) as f32,
    }
}

/// The fixture's provenance, independent of any device: this half must never be
/// skippable.
#[test]
fn fixture_records_the_frozen_reference() {
    let bytes = std::fs::read(fixture_path()).expect("read fixture");
    assert_provenance(&bytes);
}

/// The kernel's own half of the refusal contract, device-free: the entry point
/// rejects a geometry it was not compiled for *before* it forms a launch. It
/// is the only place the compiled row stride is checked — the reason
/// `fused_width` is an argument at all — and `rotary_dim` and `rope_theta`
/// are compiled constants nothing else can be held against.
///
/// Every call below leaves through a guard before the first launch, so the
/// pointers can dangle. The two zero-extent calls come first as the positive
/// control: an entry point that refused everything would make every refusal
/// below vacuous.
#[test]
fn the_entry_point_refuses_a_geometry_it_was_not_compiled_for() {
    use cudarc::driver::sys::CUresult;
    use pegainfer_kernels::ffi::Half;
    use pegainfer_kernels::ffi::qwen4_qsa_indexer_block_scores_cuda as entry;

    // Dangling and well-aligned, never dereferenced.
    let half = std::ptr::NonNull::<Half>::dangling().as_ptr();
    let int = std::ptr::NonNull::<i32>::dangling().as_ptr();
    let float = std::ptr::NonNull::<f32>::dangling().as_ptr();
    let pool = std::ptr::NonNull::<Half>::dangling().as_ptr();
    let invalid = CUresult::CUDA_ERROR_INVALID_VALUE as i32;
    let success = CUresult::CUDA_SUCCESS as i32;

    // (query_rows, tokens, score_cols, fused_width, rotary_dim, rope_theta, rms_eps)
    let call = |args: (i32, i32, i32, i32, i32, f32, f32)| {
        let (query_rows, tokens, score_cols, fused_width, rotary_dim, rope_theta, rms_eps) = args;
        (unsafe {
            entry(
                half,
                half,
                half,
                int,
                int,
                float,
                pool,
                query_rows,
                tokens,
                score_cols,
                fused_width,
                rotary_dim,
                rope_theta,
                rms_eps,
                std::ptr::null_mut(),
            )
        }) as i32
    };

    // Zero extent, so the guards are the whole of the call and no launch follows.
    assert_eq!(
        call((0, 8, 2, 640, 64, 1.0e7, 1.0e-6)),
        success,
        "the compiled geometry must pass the guards"
    );
    assert_eq!(
        call((1, 0, 0, 640, 64, 1.0e7, 1.0e-6)),
        success,
        "an empty call must pass the guards"
    );

    for (what, result) in [
        (
            "a fused row stride that is not the compiled 640 wide",
            call((0, 8, 2, 639, 64, 1.0e7, 1.0e-6)),
        ),
        (
            "a fused row stride wider than the compiled one",
            call((0, 8, 2, 641, 64, 1.0e7, 1.0e-6)),
        ),
        (
            "a rotary split that is not the compiled 64 of the 128-dim head",
            call((0, 8, 2, 640, 32, 1.0e7, 1.0e-6)),
        ),
        (
            "a rotary split over the whole indexer head",
            call((0, 8, 2, 640, 128, 1.0e7, 1.0e-6)),
        ),
        (
            "a rope_theta that is not the compiled 1e7",
            call((0, 8, 2, 640, 64, 1.0e6, 1.0e-6)),
        ),
        (
            // One f32 ulp: the geometry's own bit-equality check, from the
            // kernel's side, and what a config drifting in the last place hits.
            "a rope_theta one ulp from the compiled 1e7",
            call((
                0,
                8,
                2,
                640,
                64,
                f32::from_bits(1.0e7f32.to_bits() + 1),
                1.0e-6,
            )),
        ),
        (
            "a negative rms_eps",
            call((0, 8, 2, 640, 64, 1.0e7, -1.0e-6)),
        ),
        (
            // `!(rms_eps >= 0)` refuses NaN, which would otherwise poison every
            // norm silently rather than fail loudly.
            "a NaN rms_eps",
            call((0, 8, 2, 640, 64, 1.0e7, f32::NAN)),
        ),
        (
            "a negative query-row count",
            call((-1, 8, 2, 640, 64, 1.0e7, 1.0e-6)),
        ),
        (
            "a negative token count",
            call((0, -1, 2, 640, 64, 1.0e7, 1.0e-6)),
        ),
        (
            "a negative score stride",
            call((0, 8, -1, 640, 64, 1.0e7, 1.0e-6)),
        ),
        (
            // 8 tokens is 2 complete blocks, so a 1-wide row is the
            // out-of-bounds write the stride check exists for.
            "a score stride under the block grid",
            call((1, 8, 1, 640, 64, 1.0e7, 1.0e-6)),
        ),
    ] {
        assert_eq!(result, invalid, "{what} must be refused");
    }

    // One position per kind of argument — an input, the score output and the
    // pool workspace — so the chain is pinned to cover the pointers, not just
    // the extents.
    for (what, fused_qk, scores, pooled_keys) in [
        ("a null fused_qk", std::ptr::null(), float, pool),
        ("a null scores", half, std::ptr::null_mut(), pool),
        ("a null pooled_keys", half, float, std::ptr::null_mut()),
    ] {
        let result = unsafe {
            entry(
                fused_qk,
                half,
                half,
                int,
                int,
                scores,
                pooled_keys,
                0,
                8,
                2,
                640,
                64,
                1.0e7,
                1.0e-6,
                std::ptr::null_mut(),
            )
        } as i32;
        assert_eq!(result, invalid, "{what} must be refused");
    }
}

#[test]
fn qsa_indexer_block_scores_match_the_reference() {
    let Some(ctx) = common::device_or_skip() else {
        return;
    };
    let bytes = std::fs::read(fixture_path()).expect("read fixture");
    assert_provenance(&bytes);
    let fixture = SafeTensors::deserialize(&bytes).expect("parse fixture");
    let geometry = QsaIndexerGeometry::qwen38_flash_next();

    let (_, q_norm) = bf16_tensor(&fixture, "q_layernorm");
    let (_, k_norm) = bf16_tensor(&fixture, "k_layernorm");
    assert_eq!(q_norm.len(), geometry.head_dim);
    assert_eq!(k_norm.len(), geometry.head_dim);
    let q_norm_dev = ctx.stream.clone_htod(&q_norm).expect("q norm upload");
    let k_norm_dev = ctx.stream.clone_htod(&k_norm).expect("k norm upload");

    let mut total = 0usize;
    let mut worst_abs = 0.0f32;
    let mut worst_scaled = 0.0f32;
    let mut truncating = 0usize;
    let mut ragged = 0usize;
    let mut boundary = 0usize;
    let mut over_1e4 = 0usize;
    let mut over_1e5 = 0usize;
    for index in 0..CASES {
        let case = load_case(&fixture, index, &geometry);
        let visible_len = visible_counts(&case);
        let kept = case.budget as usize / geometry.compress_ratio;
        let scored = visible_len
            .iter()
            .map(|visible| usize::try_from(*visible).expect("visible") / geometry.compress_ratio)
            .max()
            .expect("a query row");
        // Coverage the fixture claims, asserted rather than assumed: a case that
        // scores more blocks than the budget keeps (so scoring cannot be the
        // truncating stage), a case with a tail, and the real 512-block edge.
        truncating += usize::from(scored > kept);
        ragged += usize::from(visible_len.iter().any(|v| {
            // `visible_len` is non-negative, so the unsigned cast is exact.
            !(*v as u32).is_multiple_of(geometry.compress_ratio as u32)
        }));
        boundary += usize::from(kept == geometry.budget / geometry.compress_ratio && scored > kept);

        let summary = check_case(&ctx, &geometry, &q_norm_dev, &k_norm_dev, &case);
        total += summary.compared;
        worst_abs = worst_abs.max(summary.max_abs);
        worst_scaled = worst_scaled.max(summary.max_scaled);
        over_1e4 += summary.over_1e4;
        over_1e5 += summary.over_1e5;
        println!(
            "case {index}: {} block scores over {} rows, tokens {}, budget {}, \
             blocks scored {scored} vs kept {kept}: max abs {:.3e} at row {} block {} \
             ({:.9} against {:.9}), above 1e-4 {}/{} ({} positive), above 1e-5 {}/{}, \
             bias {:+.2e}, widest row {} with {}, bound fraction {:.3}",
            summary.compared,
            case.rows.len(),
            case.tokens,
            case.budget,
            summary.max_abs,
            summary.worst.0,
            summary.worst.1,
            summary.worst.2,
            summary.worst.3,
            summary.over_1e4,
            summary.compared,
            summary.pos_1e4,
            summary.over_1e5,
            summary.compared,
            summary.bias,
            summary.widest_row.0,
            summary.widest_row.1,
            summary.max_scaled,
        );
    }
    println!(
        "parity over {CASES} cases and {total} block scores: max abs {worst_abs:.3e}, \
         worst bound fraction {worst_scaled:.3} (ATOL {ATOL}, RTOL {RTOL}); \
         {over_1e4} of {total} above 1e-4, {over_1e5} above 1e-5"
    );
    assert!(
        truncating >= 1,
        "a case must score past what its budget's top-k would keep, or scoring \
         could be confused with the truncating stage"
    );
    assert!(ragged >= 1, "the incomplete tail must be covered");
    assert_eq!(boundary, 1, "the 512-block budget edge must be covered");
    assert!(
        worst_scaled <= 1.0,
        "worst block score is {worst_scaled} of the bound"
    );
}
