//! Qwen3.8-Flash-Next QSA token indexer against the frozen reference golden.
//!
//! Replays `test_data/qwen38fn-qsa-indexer-golden.safetensors` (dumped from
//! transformers 5.16.1 `qwen4_exp` by
//! `tools/accuracy/dump_qwen38fn_qsa_indexer_golden.py`) through both halves of
//! #1105's indexer. `score` replays the fixture's post-`index_qk_proj` inputs
//! through the block-scoring kernels and holds the scores to the reference;
//! `select` replays the reference's recorded block scores through the selection
//! kernel and holds the mask to the reference, bitwise on a strict cut and
//! structurally where `torch.topk` left a tie.
//!
//! Provenance is asserted, not assumed: the fixture records the reference
//! version and the sha256 of the frozen config it was dumped from, and each
//! half derives its geometry from that config, so a golden from another
//! checkpoint or another semantics fails loudly instead of skipping. Without a
//! device the device halves skip and `PEGAINFER_REQUIRE_GPU=1` turns that into a
//! failure; the provenance test and the launcher-refusal test need no device
//! and always run.
//!
//! Manual gate: no CI job enables the kernels crate's `qwen4` feature, so CI
//! neither builds nor runs it. Run it with `--features qwen4` and
//! `PEGAINFER_REQUIRE_GPU=1`:
//!
//!   cargo test --release -p pegainfer-kernels --features qwen4 \
//!     --test qwen4_qsa_indexer_parity -- --nocapture
#![cfg(feature = "qwen4")]
// Exact float comparison is the point: selection is a chain of comparisons
// over recorded scores, and a tied cut is defined by score-equal blocks, so a
// margin here would erase the distinction the gate exists for.
#![allow(clippy::float_cmp)]

mod common;

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

fn fixture_path() -> String {
    format!("{}/../test_data/{FIXTURE}", env!("CARGO_MANIFEST_DIR"))
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
/// reference version and the frozen config's bytes. Returns the parsed config,
/// from which the gate derives the geometry and builds the cos/sin tables.
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
    let rope_theta = config["text_config"]["rope_parameters"]["rope_theta"]
        .as_f64()
        .expect("rope_theta");
    assert!(
        rope_theta.is_finite() && rope_theta > 0.0,
        "rope_theta {rope_theta} cannot build a table"
    );
    // Recorded, not asserted: the host the golden was dumped on decides the
    // order its float32 reductions round in, and any other host's order is an
    // equally valid reference. That order is one candidate for the long case's
    // residual, which is not traced yet. Printed because the bound's
    // justification leans on it and it is otherwise invisible to whoever reads
    // this gate's output.
    println!(
        "golden provenance: transformers {}, torch {}, {}",
        manifest["transformers_version"].as_str().unwrap_or("?"),
        manifest["torch_version"].as_str().unwrap_or("?"),
        manifest["platform"].as_str().unwrap_or("?"),
    );
    config
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

/// The fixture's provenance, independent of any device: this half must never be
/// skippable.
#[test]
fn fixture_records_the_frozen_reference() {
    let bytes = std::fs::read(fixture_path()).expect("read fixture");
    assert_provenance(&bytes);
}

mod score {
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
    //! version and the sha256 of the frozen config it was dumped from. The gate
    //! derives its geometry from that config and cross-checks it against the
    //! fixture's own tensor shapes, so a config or checkpoint drift reaches the
    //! kernel instead of a duplicated Rust contract. Without a device it skips;
    //! `PEGAINFER_REQUIRE_GPU=1` turns that into a failure.

    use cudarc::driver::CudaSlice;
    use half::bf16;
    use pegainfer_kernels::ops::qsa_indexer_block_scores_into;
    use pegainfer_kernels::tensor::DeviceContext;
    use pegainfer_kernels::tensor::DeviceVec;

    use super::*;

    /// Score parity bound, `abs(got - want) <= ATOL + RTOL * abs(want)`.
    ///
    /// Cases 0..6 sit at ≤9.6e-7 (measured: 8.9e-8 for the 5-token case, 4.8e-7 for
    /// the 8-token one, and 9.5e-7 or less on the rest), which is a handful of f32
    /// ulps, so they carry a tight bound. The long case is the only one with
    /// positions past 63 and 513 blocks; its 2.4e-3 residual has the shape of a
    /// flipped bf16 rounding decision — the reference reproduces the same size
    /// across environments: re-deriving its case 2 on x86 from the fixture's own
    /// inputs moves one of its 496 scores by 2.2e-3 — and keeps the loose bound
    /// until that residual is traced.
    ///
    /// The tight bound's discriminative floor is the measured mutation floor, not an
    /// assumption about the arithmetic: keeping the pooled mean in f32 instead of
    /// its bf16 round trip moves scores by 1.55e-2, 100x `rms_eps` by 1.25e-2,
    /// dropping `rms_eps` or pooling by sum by 3.3e-3, moving the pooled key's RoPE
    /// to the block end by 2.7, and a post-sum relu by 0.83 — every one of them
    /// two to six orders past the tight bound. Architecture variance is not a
    /// reason to loosen it: the gate prints identical numbers on sm_89 and sm_120,
    /// and the kernel reads the same host-built table the attention uses, so there
    /// is no device trig to vary.
    const ATOL_TIGHT: f32 = 2.0e-6;
    const RTOL_TIGHT: f32 = 2.0e-6;
    const ATOL_LONG: f32 = 5.0e-3;
    const RTOL_LONG: f32 = 5.0e-3;
    /// The only case whose residual is not yet traced.
    const LONG_CASE: usize = 6;

    fn parity_bound(case: usize) -> (f32, f32) {
        if case == LONG_CASE {
            (ATOL_LONG, RTOL_LONG)
        } else {
            (ATOL_TIGHT, RTOL_TIGHT)
        }
    }

    /// Written over the score buffer before every launch, to prove the kernel writes
    /// exactly the valid prefix of each row and nothing else.
    const SENTINEL: f32 = -12345.0;

    fn config_theta(config: &serde_json::Value) -> f32 {
        config["text_config"]["rope_parameters"]["rope_theta"]
            .as_f64()
            .expect("rope_theta") as f32
    }

    /// The attention line's cos/sin tables, built like
    /// `pegainfer_core::rope::RopeTableSpec::tables` with `frequency_dim` equal to
    /// `rotary_dim`: `[max_pos, rotary_dim]`, position-major, half-split
    /// duplicated, bf16. Equal dims are what the reference's partial RoPE uses
    /// (its `inv_freq` spans the 64 rotated dims); a spec with the head dim as
    /// `frequency_dim` builds a different table. The gate builds them here because
    /// `pegainfer-kernels` cannot depend on `pegainfer-core` (the crate dependency
    /// runs the other way); production builds the same table with that function
    /// and passes it in, which is the point of the kernel reading it.
    fn rope_tables(rotary_dim: usize, theta: f32, max_pos: usize) -> (Vec<bf16>, Vec<bf16>) {
        assert!(rotary_dim > 0 && rotary_dim.is_multiple_of(2) && max_pos > 0);
        let half_dim = rotary_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(i as f32 * 2.0 / rotary_dim as f32))
            .collect();
        let mut cos = vec![bf16::ZERO; max_pos * rotary_dim];
        let mut sin = vec![bf16::ZERO; max_pos * rotary_dim];
        for pos in 0..max_pos {
            let base = pos * rotary_dim;
            for i in 0..half_dim {
                let angle = pos as f32 * inv_freq[i];
                let c = bf16::from_f32(angle.cos());
                let s = bf16::from_f32(angle.sin());
                cos[base + i] = c;
                cos[base + i + half_dim] = c;
                sin[base + i] = s;
                sin[base + i + half_dim] = s;
            }
        }
        (cos, sin)
    }

    /// The geometry this gate runs, derived from the frozen config and cross-checked
    /// against the fixture's own tensor shapes — not from a Rust copy of the
    /// kernel's constants. The compiled kernel produces the scores, so a drift
    /// between the config and what it was compiled for shows up as a score
    /// mismatch instead of as a duplicated contract.
    #[derive(Clone, Copy)]
    struct Geometry {
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        compress_ratio: usize,
        rotary_dim: usize,
        rms_eps: f32,
    }

    impl Geometry {
        fn fused_width(&self) -> usize {
            (self.heads + self.kv_heads) * self.head_dim
        }

        fn max_blocks(&self, tokens: usize) -> usize {
            tokens / self.compress_ratio
        }
    }

    fn fixture_geometry(config: &serde_json::Value, fixture: &SafeTensors<'_>) -> Geometry {
        let text = &config["text_config"];
        let field = |name: &str| text[name].as_u64().unwrap_or_else(|| panic!("{name}")) as usize;
        let attention_head_dim = text["head_dim"].as_f64().expect("head_dim");
        let partial = text["rope_parameters"]["partial_rotary_factor"]
            .as_f64()
            .expect("partial_rotary_factor");
        let geometry = Geometry {
            heads: field("indexer_n_heads"),
            kv_heads: field("indexer_kv_heads"),
            head_dim: field("indexer_head_dim"),
            compress_ratio: field("indexer_compress_ratio"),
            rotary_dim: (attention_head_dim * partial) as usize,
            rms_eps: text["rms_norm_eps"].as_f64().expect("rms_norm_eps") as f32,
        };
        // The fixture's own shapes must agree with the config the golden was dumped
        // from; the scores then hold the compiled kernel to both.
        let (key_shape, _) = bf16_tensor(fixture, "case0_keys");
        assert_eq!(key_shape[1], geometry.head_dim, "case0 key head dim");
        let (query_shape, _) = bf16_tensor(fixture, "case0_queries");
        assert_eq!(
            &query_shape[1..],
            [geometry.heads, geometry.head_dim],
            "case0 query heads"
        );
        assert!(geometry.heads > 0 && geometry.kv_heads == 1);
        assert!(geometry.compress_ratio > 0 && geometry.rotary_dim > 0);
        assert!(geometry.rotary_dim <= geometry.head_dim);
        assert!(geometry.rms_eps >= 0.0);
        geometry
    }

    /// Bit equality for the scalars the kernel pins and the exact derivations
    /// beside them. The kernel refuses any other value, so a tolerance here would
    /// let a config drift through.
    fn same_f32(a: f32, b: f32) -> bool {
        a.to_bits() == b.to_bits()
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

    fn load_case(fixture: &SafeTensors<'_>, index: usize, geometry: &Geometry) -> Case {
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
        Case {
            index,
            tokens,
            budget: budget_tensor(fixture, &format!("case{index}_budget")),
            keys,
            queries,
            rows,
            reference,
            score_cols: score_shape[1],
        }
    }

    /// The fused `index_qk_proj` rows the kernel reads: the keys for every token,
    /// then the queries at the rows the reference evaluated.
    fn fused_rows(case: &Case, geometry: &Geometry) -> Vec<bf16> {
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
            fused[row * width..][..q_width]
                .copy_from_slice(&case.queries[slot * q_width..][..q_width]);
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
    /// the shape of the residual. The sign split, the bias and the widest row are
    /// read-outs — printed as evidence, not asserted — so the assertion is the
    /// bound and each metric keeps its own coordinates.
    struct CaseSummary {
        compared: usize,
        max_abs: f32,
        /// `(position, block, scored, reference)` of the largest absolute error.
        max_abs_at: (i32, usize, f32, f32),
        max_scaled: f32,
        /// `(position, block, scored, reference)` of the largest scaled error.
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
        geometry: &Geometry,
        q_norm: &CudaSlice<bf16>,
        k_norm: &CudaSlice<bf16>,
        cos: &DeviceVec,
        sin: &DeviceVec,
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
            &fused,
            q_norm,
            k_norm,
            cos,
            sin,
            &rows_dev,
            &visible_dev,
            case.tokens,
            &mut scores_dev,
            score_cols,
            &mut pooled_dev,
            geometry.rms_eps,
        )
        .expect("QSA indexer block scores");
        ctx.sync().expect("sync");
        let got = ctx.stream.clone_dtoh(&scores_dev).expect("score download");

        let (atol, rtol) = parity_bound(case.index);
        let mut compared = 0usize;
        let mut max_abs = 0.0f32;
        let mut max_abs_at = (0i32, 0usize, 0.0f32, 0.0f32);
        let mut max_scaled = 0.0f32;
        let mut worst = (0i32, 0usize, 0.0f32, 0.0f32);
        let mut over_1e4 = 0usize;
        let mut over_1e5 = 0usize;
        let mut pos_1e4 = 0usize;
        let mut widest_row = (0i32, 0usize);
        let mut signed_sum = 0.0f64;
        for (slot, row) in case.rows.iter().enumerate() {
            let blocks =
                usize::try_from(visible_len[slot]).expect("visible") / geometry.compress_ratio;
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
                let scaled = abs / (atol + rtol * want.abs());
                if abs > max_abs {
                    max_abs = abs;
                    max_abs_at = (*row, block, *x, *want);
                }
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
                     {want}, {abs:.3e} off; the bound is ATOL {atol} + RTOL {rtol} * |want|",
                    case.index
                );
            }
            compared += blocks;
        }
        assert!(compared > 0, "case {}: nothing compared", case.index);
        CaseSummary {
            compared,
            max_abs,
            max_abs_at,
            max_scaled,
            worst,
            over_1e4,
            over_1e5,
            pos_1e4,
            widest_row,
            bias: (signed_sum / compared as f64) as f32,
        }
    }

    #[test]
    fn qsa_indexer_block_scores_match_the_reference() {
        let Some(ctx) = common::device_or_skip() else {
            return;
        };
        let bytes = std::fs::read(fixture_path()).expect("read fixture");
        let config = assert_provenance(&bytes);
        let fixture = SafeTensors::deserialize(&bytes).expect("parse fixture");
        let geometry = fixture_geometry(&config, &fixture);
        let theta = config_theta(&config);

        let (_, q_norm) = bf16_tensor(&fixture, "q_layernorm");
        let (_, k_norm) = bf16_tensor(&fixture, "k_layernorm");
        assert_eq!(q_norm.len(), geometry.head_dim);
        assert_eq!(k_norm.len(), geometry.head_dim);
        let q_norm_dev = ctx.stream.clone_htod(&q_norm).expect("q norm upload");
        let k_norm_dev = ctx.stream.clone_htod(&k_norm).expect("k norm upload");

        let mut total = 0usize;
        let mut worst_abs = 0.0f32;
        let mut worst_scaled = 0.0f32;
        let mut ragged = 0usize;
        let mut over_1e4 = 0usize;
        let mut over_1e5 = 0usize;
        for index in 0..CASES {
            let case = load_case(&fixture, index, &geometry);
            let visible_len = visible_counts(&case);
            let kept = case.budget as usize / geometry.compress_ratio;
            let scored = visible_len
                .iter()
                .map(|visible| {
                    usize::try_from(*visible).expect("visible") / geometry.compress_ratio
                })
                .max()
                .expect("a query row");
            // The fixture's coverage is printed, not asserted: how many blocks each
            // row scores versus what the budget would keep is selection-stage
            // bookkeeping, and the `ragged` check below is the part this stage
            // needs (the incomplete tail must exist).
            ragged += usize::from(visible_len.iter().any(|v| {
                // `visible_len` is non-negative, so the unsigned cast is exact.
                !(*v as u32).is_multiple_of(geometry.compress_ratio as u32)
            }));

            let (cos_host, sin_host) = rope_tables(geometry.rotary_dim, theta, case.tokens);
            let cos_dev = DeviceVec::from_host(&ctx, &cos_host).expect("cos upload");
            let sin_dev = DeviceVec::from_host(&ctx, &sin_host).expect("sin upload");
            let summary = check_case(
                &ctx,
                &geometry,
                &q_norm_dev,
                &k_norm_dev,
                &cos_dev,
                &sin_dev,
                &case,
            );
            total += summary.compared;
            worst_abs = worst_abs.max(summary.max_abs);
            worst_scaled = worst_scaled.max(summary.max_scaled);
            over_1e4 += summary.over_1e4;
            over_1e5 += summary.over_1e5;
            let (atol, rtol) = parity_bound(index);
            println!(
                "case {index}: {} block scores over {} rows, tokens {}, budget {}, \
                 blocks scored {scored} vs kept {kept}: max abs {:.3e} at row {} block {} \
                 ({:.9} against {:.9}); worst bound fraction {:.3} at row {} block {} \
                 ({:.9} against {:.9}); bound ATOL {atol:.1e} + RTOL {rtol:.1e} * |want|; \
                 above 1e-4 {}/{} ({} positive), above 1e-5 {}/{}, bias {:+.2e}, \
                 widest row {} with {}",
                summary.compared,
                case.rows.len(),
                case.tokens,
                case.budget,
                summary.max_abs,
                summary.max_abs_at.0,
                summary.max_abs_at.1,
                summary.max_abs_at.2,
                summary.max_abs_at.3,
                summary.max_scaled,
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
            );
        }
        println!(
            "parity over {CASES} cases and {total} block scores: max abs {worst_abs:.3e}, \
             worst bound fraction {worst_scaled:.3}; {over_1e4} of {total} above 1e-4, \
             {over_1e5} above 1e-5"
        );
        assert!(ragged >= 1, "the incomplete tail must be covered");
    }

    /// A poisoned key must not read as a valid lowest score. `torch.relu` keeps a
    /// NaN, and the kernel's relu is written to match: the micro-block that pools
    /// the poisoned token scores NaN for every row that can see it, while the other
    /// blocks stay finite and the untouched columns stay sentinel.
    #[test]
    fn a_poisoned_key_keeps_its_nan_score() {
        let Some(ctx) = common::device_or_skip() else {
            return;
        };
        let bytes = std::fs::read(fixture_path()).expect("read fixture");
        let config = assert_provenance(&bytes);
        let fixture = SafeTensors::deserialize(&bytes).expect("parse fixture");
        let geometry = fixture_geometry(&config, &fixture);
        let (_, q_norm) = bf16_tensor(&fixture, "q_layernorm");
        let (_, k_norm) = bf16_tensor(&fixture, "k_layernorm");
        let q_norm_dev = ctx.stream.clone_htod(&q_norm).expect("q norm upload");
        let k_norm_dev = ctx.stream.clone_htod(&k_norm).expect("k norm upload");

        // Case 2: 64 tokens and 16 blocks, so every row from position 3 on sees
        // block 0, which is the block the poisoned token pools into.
        let mut case = load_case(&fixture, 2, &geometry);
        let (cos_host, sin_host) =
            rope_tables(geometry.rotary_dim, config_theta(&config), case.tokens);
        let cos_dev = DeviceVec::from_host(&ctx, &cos_host).expect("cos upload");
        let sin_dev = DeviceVec::from_host(&ctx, &sin_host).expect("sin upload");
        case.keys[0] = bf16::from_f32(f32::NAN);
        let visible_len = visible_counts(&case);
        let query_rows = case.rows.len();
        let score_cols = case.score_cols;

        let fused = ctx
            .stream
            .clone_htod(&fused_rows(&case, &geometry))
            .expect("fused upload");
        let rows_dev = ctx.stream.clone_htod(&case.rows).expect("rows upload");
        let visible_dev = ctx.stream.clone_htod(&visible_len).expect("visible upload");
        let mut scores_dev = ctx
            .stream
            .clone_htod(&vec![SENTINEL; query_rows * score_cols])
            .expect("score buffer");
        let mut pooled_dev = ctx
            .stream
            .alloc_zeros::<bf16>(geometry.max_blocks(case.tokens) * geometry.head_dim)
            .expect("pooled workspace");
        qsa_indexer_block_scores_into(
            &ctx,
            &fused,
            &q_norm_dev,
            &k_norm_dev,
            &cos_dev,
            &sin_dev,
            &rows_dev,
            &visible_dev,
            case.tokens,
            &mut scores_dev,
            score_cols,
            &mut pooled_dev,
            geometry.rms_eps,
        )
        .expect("QSA indexer block scores");
        ctx.sync().expect("sync");
        let got = ctx.stream.clone_dtoh(&scores_dev).expect("score download");

        let mut saw_nan = 0usize;
        for (slot, row) in case.rows.iter().enumerate() {
            let blocks =
                usize::try_from(visible_len[slot]).expect("visible") / geometry.compress_ratio;
            let row_got = &got[slot * score_cols..][..score_cols];
            if blocks > 0 {
                assert!(
                    row_got[0].is_nan(),
                    "case 2 row {row}: the poisoned block must stay NaN, got {}",
                    row_got[0]
                );
                saw_nan += 1;
                for (block, score) in row_got.iter().enumerate().take(blocks).skip(1) {
                    assert!(
                        score.is_finite(),
                        "case 2 row {row} block {block}: an unpoisoned block must stay \
                         finite, got {score}"
                    );
                }
            }
            for (block, score) in row_got.iter().enumerate().skip(blocks) {
                assert!(
                    same_f32(*score, SENTINEL),
                    "case 2 row {row} column {block}: past the prefix must stay sentinel, \
                     got {score}"
                );
            }
        }
        assert!(saw_nan > 0, "no row saw the poisoned block");
        println!("poisoned key: block 0 is NaN on {saw_nan} rows, later blocks finite");
    }
}

mod select {
    //! `test_data/qwen38fn-qsa-indexer-golden.safetensors` (with the scoring gate's
    //! fixture) stores, per case, the reference's per-row block scores and the final
    //! `[rows, tokens]` visibility mask. This gate replays those recorded scores
    //! through the selection kernel and compares the mask. Selection is
    //! comparison-only — no arithmetic touches a score — so there is no tolerance
    //! here: on a row whose cut is strict the mask must match **bitwise**, and the
    //! only rows held to less are the ones where `torch.topk` left a block tied at
    //! the cut (see below).
    //!
    //! Provenance is asserted, not assumed: the fixture records the reference
    //! version and the sha256 of the frozen config it was dumped from, and the gate
    //! derives the block ratio and the default budget from that config, holding
    //! them against the fixture's shapes and budgets. Provenance failure panics; it
    //! is never a skip. Without a device the parity half skips and
    //! `PEGAINFER_REQUIRE_GPU=1` turns that into a failure; the provenance test and
    //! the launcher-refusal test need no device and always run.
    //!
    //! Ties: `torch.topk` leaves the pick at the cut open, so a block tied with the
    //! last kept one is an equally valid pick. The kernel's canonical order is
    //! descending score, ascending block index. For each row the gate computes the
    //! cut score; if no other block shares it the row is compared bitwise, and
    //! otherwise both masks are required to be structurally valid (a causal prefix,
    //! only complete four-token blocks plus the exact tail, exactly
    //! `min(budget / 4, blocks) * 4 + tail` bits set), to select only scores at
    //! least as high as every block they exclude, and to keep the same number of
    //! tied-at-cut blocks. Measured on this fixture: 133 rows, 129 compared
    //! bitwise, 4 compared up to ties (all in case 6, at cut score 0.0 with 31 to
    //! 43 tied blocks). **Ungated boundary:** every tie here is at zero; the fixture
    //! has no row tied at a non-zero cut, so that case is not gated. Case 6's rows
    //! are 2049..2054, so positions before 2049 at that length are not exercised,
    //! and the selector derives a row's visible list from its position (token 0 is
    //! the first visible token, the scoring pass's grid contract), so a
    //! prefix-evicted visible list is out of contract and has no case here.
    //!
    //! The fixture covers both selection paths: rows that keep every block take
    //! the all-kept fast path, and the truncating rows run the masked argmax
    //! loop.

    use std::cmp::Ordering;

    use cudarc::driver::sys::CUresult;
    use pegainfer_kernels::ffi::qwen4_qsa_indexer_select_cuda;
    use pegainfer_kernels::ops::qsa_indexer_select_into;
    use pegainfer_kernels::tensor::DeviceContext;

    use super::*;

    /// The four tokens per micro-block the kernel compiles in. The config's
    /// `indexer_compress_ratio` must equal it; a drift is caught in `load_case`
    /// rather than silently selecting blocks of another width.
    const RATIO: usize = 4;
    /// Written over the mask buffer before every launch, to prove the kernel writes
    /// every cell of every row.
    const SENTINEL: u8 = 0xAA;

    /// The reference's mask is `torch.bool`, one byte per cell.
    fn mask_tensor(fixture: &SafeTensors<'_>, name: &str) -> (Vec<usize>, Vec<u8>) {
        let view = fixture
            .tensor(name)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(view.dtype(), safetensors::Dtype::BOOL, "{name} dtype");
        (view.shape().to_vec(), view.data().to_vec())
    }

    fn budget_tensor(fixture: &SafeTensors<'_>, name: &str) -> i32 {
        let (shape, host) = i32_tensor(fixture, name);
        assert_eq!(shape, vec![1], "{name} shape");
        host[0]
    }

    /// One case's recorded scores, rows, budget and the reference's mask.
    struct Case {
        index: usize,
        tokens: usize,
        budget: i32,
        rows: Vec<i32>,
        score_cols: usize,
        /// `[rows, score_cols]`, the NaN padding included.
        scores: Vec<f32>,
        /// `[rows, tokens]`, one byte per cell.
        golden: Vec<u8>,
    }

    fn load_case(fixture: &SafeTensors<'_>, index: usize, scenario: &Scenario) -> Case {
        let (rows_shape, rows) = i32_tensor(fixture, &format!("case{index}_rows"));
        let (score_shape, scores) = f32_tensor(fixture, &format!("case{index}_scores"));
        let (mask_shape, golden) = mask_tensor(fixture, &format!("case{index}_mask"));
        let budget = budget_tensor(fixture, &format!("case{index}_budget"));

        assert_eq!(rows_shape, vec![rows.len()], "case {index} rows shape");
        assert_eq!(score_shape.len(), 2, "case {index} score rank");
        assert_eq!(mask_shape.len(), 2, "case {index} mask rank");
        assert_eq!(score_shape[0], rows.len(), "case {index} score rows");
        assert_eq!(mask_shape[0], rows.len(), "case {index} mask rows");
        let tokens = mask_shape[1];
        assert_eq!(golden.len(), rows.len() * tokens, "case {index} mask cells");
        assert!(tokens >= 1, "case {index} has no tokens");
        let score_cols = score_shape[1];

        // The config's ratio and the compiled one must agree before any shape is
        // read through them: the mask's block structure is the reference's, and the
        // kernel selects blocks of exactly RATIO tokens.
        assert_eq!(
            scenario.ratio, RATIO,
            "config indexer_compress_ratio {} is not the compiled {RATIO}",
            scenario.ratio
        );
        assert!(
            budget > 0 && (budget as usize).is_multiple_of(scenario.ratio),
            "case {index} budget {budget} is not a positive multiple of the ratio"
        );
        let budget_blocks = budget as usize / scenario.ratio;
        assert!(
            budget_blocks <= scenario.max_budget_blocks,
            "case {index} budget keeps {budget_blocks} blocks, past the {} the config ceiling allows",
            scenario.max_budget_blocks
        );
        // One row can see at most `tokens / ratio` blocks; the score row must hold
        // them, and every cell past the row's own prefix is the reference's NaN.
        let block_ceiling = tokens / scenario.ratio;
        assert!(
            score_cols >= block_ceiling,
            "case {index}: {score_cols} score columns cannot hold {block_ceiling} blocks"
        );
        for (slot, row) in rows.iter().enumerate() {
            assert!(
                *row >= 0 && (*row as usize) < tokens,
                "case {index} row {row} is outside the {tokens} tokens"
            );
            let blocks = (*row as usize + 1) / scenario.ratio;
            let row_scores = &scores[slot * score_cols..][..score_cols];
            for (block, score) in row_scores.iter().take(blocks).enumerate() {
                assert!(
                    score.is_finite(),
                    "case {index} row {row} block {block} is not finite: {score}"
                );
            }
            for (block, score) in row_scores.iter().enumerate().skip(blocks) {
                assert!(
                    score.is_nan(),
                    "case {index} row {row} column {block} is not the reference's padding"
                );
            }
        }
        Case {
            index,
            tokens,
            budget,
            rows,
            score_cols,
            scores,
            golden,
        }
    }

    /// What the frozen config says about this gate's geometry.
    struct Scenario {
        ratio: usize,
        /// The config's own budget, in blocks; the fixture must exercise it.
        default_budget: i32,
        /// The compiled budget ceiling in blocks.
        max_budget_blocks: usize,
    }

    impl Scenario {
        fn from_config(config: &serde_json::Value) -> Scenario {
            let text = &config["text_config"];
            let ratio = text["indexer_compress_ratio"]
                .as_u64()
                .expect("indexer_compress_ratio") as usize;
            let default_budget = text["indexer_budget"].as_u64().expect("indexer_budget") as i32;
            // The selection kernel refuses a budget past 512 blocks; the config's
            // own budget must stay inside that.
            let max_budget_blocks = 512;
            assert!(ratio > 0, "indexer_compress_ratio");
            assert!(default_budget > 0, "indexer_budget");
            assert!(default_budget as usize / ratio <= max_budget_blocks);
            Scenario {
                ratio,
                default_budget,
                max_budget_blocks,
            }
        }

        fn blocks_of(&self, budget: i32) -> usize {
            budget as usize / self.ratio
        }
    }

    impl Case {
        fn row_scores(&self, slot: usize) -> &[f32] {
            &self.scores[slot * self.score_cols..][..self.score_cols]
        }

        fn blocks(&self, slot: usize) -> usize {
            (self.rows[slot] as usize + 1) / RATIO
        }

        fn tail(&self, slot: usize) -> usize {
            (self.rows[slot] as usize + 1) % RATIO
        }

        fn keep(&self, slot: usize, budget_blocks: usize) -> usize {
            budget_blocks.min(self.blocks(slot))
        }

        fn golden_row(&self, slot: usize) -> &[u8] {
            &self.golden[slot * self.tokens..][..self.tokens]
        }
    }

    /// The canonical order over a score row: descending score, ascending block
    /// index, the kernel's tie rule.
    fn canonical_order(scores: &[f32]) -> Vec<usize> {
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| {
            let by_score = if scores[a] == scores[b] {
                Ordering::Equal
            } else {
                scores[b].partial_cmp(&scores[a]).expect("finite scores")
            };
            by_score.then(a.cmp(&b))
        });
        order
    }

    /// A host-built mask for one row: the tokens of `selected` blocks plus the
    /// first `tail_bits` tokens of the row's incomplete block, which starts at
    /// `blocks * RATIO`.
    fn host_mask(tokens: usize, blocks: usize, selected: &[usize], tail_bits: usize) -> Vec<u8> {
        assert!(blocks * RATIO + tail_bits <= tokens, "row mask bounds");
        let mut mask = vec![0u8; tokens];
        for &block in selected {
            mask[block * RATIO..][..RATIO].fill(1);
        }
        mask[blocks * RATIO..][..tail_bits].fill(1);
        mask
    }

    fn differing(a: &[u8], b: &[u8]) -> Vec<usize> {
        assert_eq!(a.len(), b.len(), "compared rows differ in length");
        (0..a.len()).filter(|&i| a[i] != b[i]).collect()
    }

    fn assert_same_row(label: &str, got: &[u8], want: &[u8]) {
        if let Some(at) = differing(got, want).first() {
            panic!(
                "{label}: first difference at token {at}: got {}, want {}. \
                 Full difference: {:?}",
                got[*at],
                want[*at],
                differing(got, want)
            );
        }
    }

    /// The structure every valid row mask has, returning its selected blocks: only
    /// `0` or `1`, complete four-token blocks below the row's own token, nothing
    /// past it, the exact tail set, and exactly `selected * 4 + tail` bits.
    fn structure(
        label: &str,
        row_mask: &[u8],
        blocks: usize,
        tail: usize,
        pos: usize,
    ) -> Vec<usize> {
        assert!(
            row_mask.iter().all(|cell| *cell <= 1),
            "{label}: the mask holds a cell other than 0 or 1"
        );
        let mut selected = Vec::new();
        for block in 0..blocks {
            let bits = &row_mask[block * RATIO..][..RATIO];
            if bits.iter().all(|cell| *cell == 1) {
                selected.push(block);
            } else {
                assert!(
                    bits.iter().all(|cell| *cell == 0),
                    "{label}: block {block} is only partly set: {bits:?}"
                );
            }
        }
        for (i, cell) in row_mask[blocks * RATIO..=pos].iter().enumerate() {
            assert_eq!(
                *cell, 1,
                "{label}: tail token {i} of {tail} must be visible"
            );
        }
        for (i, cell) in row_mask[pos + 1..].iter().enumerate() {
            assert_eq!(
                *cell,
                0,
                "{label}: token {} is past the row's own position and must not be visible",
                pos + 1 + i
            );
        }
        let set: usize = row_mask.iter().map(|cell| *cell as usize).sum();
        assert_eq!(
            set,
            selected.len() * RATIO + tail,
            "{label}: {set} set bits is not the selected blocks' tokens plus the tail"
        );
        selected
    }

    /// Runs one case's mask through the production launch and returns the
    /// downloaded cells.
    fn run_case(ctx: &DeviceContext, case: &Case) -> Vec<u8> {
        let scores = ctx.stream.clone_htod(&case.scores).expect("score upload");
        let rows = ctx.stream.clone_htod(&case.rows).expect("rows upload");
        let mut mask = ctx
            .stream
            .clone_htod(&vec![SENTINEL; case.rows.len() * case.tokens])
            .expect("mask buffer");
        qsa_indexer_select_into(
            ctx,
            &scores,
            case.score_cols,
            &rows,
            case.budget,
            case.tokens,
            &mut mask,
        )
        .unwrap_or_else(|e| panic!("case {}: launch: {e:#}", case.index));
        ctx.sync().expect("sync");
        let host = ctx.stream.clone_dtoh(&mask).expect("mask download");
        assert!(
            host.iter().all(|cell| *cell <= 1),
            "case {}: left a cell unwritten or wrote past 1",
            case.index
        );
        host
    }

    /// One case's measured comparison.
    #[derive(Default)]
    struct CaseReport {
        rows: usize,
        bitwise: usize,
        tie: usize,
        truncating: usize,
        /// Rows that keep every block and take the all-kept fast path.
        all_kept: usize,
    }

    fn check_case(ctx: &DeviceContext, case: &Case, scenario: &Scenario) -> CaseReport {
        let budget_blocks = scenario.blocks_of(case.budget);
        let rows = case.rows.len();
        let tokens = case.tokens;

        let production = run_case(ctx, case);

        let mut report = CaseReport {
            rows,
            ..CaseReport::default()
        };
        for slot in 0..rows {
            let row = case.rows[slot];
            let pos = row as usize;
            let blocks = case.blocks(slot);
            let tail = case.tail(slot);
            let keep = case.keep(slot, budget_blocks);
            let scores = &case.row_scores(slot)[..blocks];
            let canonical = canonical_order(scores);
            let label = format!("case {} row {row}", case.index);

            // The kernel and the host's canonical order must agree on every row.
            // On a tie row this still pins the kernel (same total order), while the
            // golden is held to validity instead of to its open pick.
            let ours = &production[slot * tokens..][..tokens];
            assert_same_row(
                &label,
                ours,
                &host_mask(tokens, blocks, &canonical[..keep], tail),
            );

            let golden = case.golden_row(slot);
            let cut = if blocks > keep {
                let mut ranked = scores.to_vec();
                ranked.sort_by(|a, b| b.partial_cmp(a).expect("finite scores"));
                Some(ranked[keep - 1])
            } else {
                None
            };
            let tie = cut.is_some_and(|cut| {
                let at_cut = scores.iter().filter(|score| **score == cut).count();
                let kept_at_cut = keep - scores.iter().filter(|score| **score > cut).count();
                at_cut > kept_at_cut
            });
            if tie {
                report.tie += 1;
                let cut = cut.expect("a tie needs a cut");
                let at_cut = scores.iter().filter(|score| **score == cut).count();
                println!(
                    "{label}: tie at cut {cut}, {at_cut} blocks tied, {keep} kept; \
                     compared up to the tie"
                );
                let ours_selected = structure(&format!("{label} (ours)"), ours, blocks, tail, pos);
                let golden_selected =
                    structure(&format!("{label} (golden)"), golden, blocks, tail, pos);
                assert_eq!(ours_selected.len(), keep, "{label}: kept blocks");
                assert_eq!(golden_selected.len(), keep, "{label}: golden kept blocks");
                // Every kept score at least every excluded one, on both masks.
                for (name, selected) in [("ours", &ours_selected), ("golden", &golden_selected)] {
                    let lowest = selected
                        .iter()
                        .map(|&block| scores[block])
                        .fold(f32::INFINITY, f32::min);
                    let highest_excluded = (0..blocks)
                        .filter(|block| !selected.contains(block))
                        .map(|block| scores[block])
                        .fold(f32::NEG_INFINITY, f32::max);
                    assert!(
                        lowest >= highest_excluded,
                        "{label}: {name} keeps a block scoring {lowest} below the excluded \
                         {highest_excluded}"
                    );
                }
                let ours_at_cut = ours_selected
                    .iter()
                    .filter(|&&block| scores[block] == cut)
                    .count();
                let golden_at_cut = golden_selected
                    .iter()
                    .filter(|&&block| scores[block] == cut)
                    .count();
                assert_eq!(
                    ours_at_cut, golden_at_cut,
                    "{label}: tied-at-cut picks ({cut}) differ between the kernel and the reference"
                );
            } else {
                report.bitwise += 1;
                assert_same_row(&label, ours, golden);
            }
            if blocks > keep {
                report.truncating += 1;
            } else {
                report.all_kept += 1;
            }
        }
        report
    }

    #[test]
    fn qsa_indexer_select_matches_the_reference() {
        let Some(ctx) = common::device_or_skip() else {
            return;
        };
        let bytes = std::fs::read(fixture_path()).expect("read fixture");
        let config = assert_provenance(&bytes);
        let fixture = SafeTensors::deserialize(&bytes).expect("parse fixture");
        let scenario = Scenario::from_config(&config);

        let mut total = CaseReport::default();
        let mut budgets = Vec::new();
        for index in 0..CASES {
            let case = load_case(&fixture, index, &scenario);
            let report = check_case(&ctx, &case, &scenario);
            budgets.push(case.budget);
            println!(
                "case {index}: {} rows, {} bitwise, {} tie, {} truncating, {} all-kept; \
                 tokens {}, budget {} ({} blocks)",
                report.rows,
                report.bitwise,
                report.tie,
                report.truncating,
                report.all_kept,
                case.tokens,
                case.budget,
                scenario.blocks_of(case.budget),
            );
            total.rows += report.rows;
            total.bitwise += report.bitwise;
            total.tie += report.tie;
            total.truncating += report.truncating;
            total.all_kept += report.all_kept;
        }
        println!(
            "selection parity over {CASES} cases: {} rows, {} bitwise, {} up to ties, \
             {} truncating, {} all-kept",
            total.rows, total.bitwise, total.tie, total.truncating, total.all_kept,
        );

        assert_eq!(
            total.rows,
            total.bitwise + total.tie,
            "every row is compared, bitwise or up to a tie"
        );
        assert!(
            total.bitwise >= total.rows - total.truncating,
            "every row that does not truncate must compare bitwise"
        );
        assert!(
            total.tie <= total.truncating,
            "only a truncating row can have a tied cut"
        );
        assert!(
            budgets.contains(&scenario.default_budget),
            "the config's own budget {} is not exercised",
            scenario.default_budget
        );
        assert!(
            total.truncating > 0,
            "the fixture must exercise the truncating path"
        );
        assert!(
            total.all_kept > 0,
            "the fixture must exercise the all-kept fast path"
        );
    }

    /// The C entry's guards, with no device and no launch: every one of them must
    /// answer `CUDA_ERROR_INVALID_VALUE` before any pointer is dereferenced, which
    /// is what lets a broken caller fail loudly instead of reading junk.
    #[test]
    fn the_launch_entry_refuses_bad_extents_without_a_device() {
        fn launch(
            scores: *const f32,
            rows: *const i32,
            mask: *mut u8,
            query_rows: i32,
            tokens: i32,
            score_cols: i32,
            budget: i32,
        ) -> CUresult {
            unsafe {
                qwen4_qsa_indexer_select_cuda(
                    scores,
                    rows,
                    mask,
                    query_rows,
                    tokens,
                    score_cols,
                    budget,
                    std::ptr::null_mut(),
                )
            }
        }
        let scores = std::ptr::NonNull::<f32>::dangling().as_ptr();
        let rows = std::ptr::NonNull::<i32>::dangling().as_ptr();
        let mask = std::ptr::NonNull::<u8>::dangling().as_ptr();
        let valid = |budget| launch(scores, rows, mask, 1, 8, 2, budget);

        let refuse = |label: &str, result: CUresult| {
            assert_eq!(
                result,
                CUresult::CUDA_ERROR_INVALID_VALUE,
                "{label} must be refused before the launch"
            );
        };
        refuse(
            "null scores",
            launch(std::ptr::null(), rows, mask, 1, 8, 2, 8),
        );
        refuse(
            "null rows",
            launch(scores, std::ptr::null(), mask, 1, 8, 2, 8),
        );
        refuse(
            "null mask",
            launch(scores, rows, std::ptr::null_mut(), 1, 8, 2, 8),
        );
        for query_rows in [-1, 0] {
            refuse(
                "query rows",
                launch(scores, rows, mask, query_rows, 8, 2, 8),
            );
        }
        for tokens in [-1, 0] {
            refuse("tokens", launch(scores, rows, mask, 1, tokens, 2, 8));
        }
        refuse(
            "negative score columns",
            launch(scores, rows, mask, 1, 8, -1, 8),
        );
        for budget in [-4, 0, 2, 7, 2052] {
            refuse("budget", valid(budget));
        }
        // A row sees up to tokens / 4 blocks, so one score column cannot serve an
        // eight-token row.
        refuse("narrow score row", launch(scores, rows, mask, 1, 8, 1, 8));
        // The exclusion bitmap is one bit per block: 1.6M tokens is past the 48 KiB
        // dynamic-shared-memory ceiling a launch may use without an opt-in.
        refuse(
            "oversized exclusion bitmap",
            launch(scores, rows, mask, 1, 1_600_000, 400_000, 8),
        );
        // A launch that passes every guard would dereference these placeholder
        // pointers, so only refusals belong in this test.
    }
}
