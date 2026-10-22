//! Qwen3.8-Flash-Next QSA token-indexer block selection against the frozen
//! reference golden.
//!
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
//! Negative controls run the same launch with the kernel's gate-only `variant`
//! and are held to computed floors, never a vacuous non-empty check: dropping
//! the tail must differ exactly on the fixture's tail bits, keeping one block
//! fewer exactly on one block per row, keeping one block more exactly on the
//! truncating rows (`blocks > budget / 4`), ignoring causality exactly on the
//! columns past each row's own token, and inverting the order must reproduce a
//! host-computed bottom-k mask and differ from production on exactly two blocks
//! per truncating row.
//!
//! Manual gate: no CI job enables the kernels crate's `qwen4` feature, so CI
//! neither builds nor runs it. Run it with `--features qwen4` and
//! `PEGAINFER_REQUIRE_GPU=1`:
//!
//!   cargo test --release -p pegainfer-kernels --features qwen4 \
//!     --test qwen4_qsa_indexer_select_parity -- --nocapture

#![cfg(feature = "qwen4")]
// Exact float comparison is the point: the kernel's order is a chain of
// comparisons over recorded scores, and a tied cut is defined by score-equal
// blocks, so a margin here would erase the distinction the gate exists for.
#![allow(clippy::float_cmp)]

mod common;

use std::cmp::Ordering;

use cudarc::driver::sys::CUresult;
use pegainfer_kernels::ffi::qwen4_qsa_indexer_select_cuda;
use pegainfer_kernels::ops::QSA_SELECT_DROP_TAIL;
use pegainfer_kernels::ops::QSA_SELECT_IGNORE_CAUSALITY;
use pegainfer_kernels::ops::QSA_SELECT_INVERT_ORDER;
use pegainfer_kernels::ops::QSA_SELECT_KEEP_FEWER;
use pegainfer_kernels::ops::QSA_SELECT_KEEP_MORE;
use pegainfer_kernels::ops::QSA_SELECT_PRODUCTION;
use pegainfer_kernels::ops::qsa_indexer_select_into;
use pegainfer_kernels::ops::qsa_indexer_select_variant_into;
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
/// The four tokens per micro-block the kernel compiles in. The config's
/// `indexer_compress_ratio` must equal it; a drift is caught in `load_case`
/// rather than silently selecting blocks of another width.
const RATIO: usize = 4;
/// Written over the mask buffer before every launch, to prove the kernel writes
/// every cell of every row.
const SENTINEL: u8 = 0xAA;

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
/// from which the gate derives the block ratio and the default budget.
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
/// index. `invert` flips only the score direction (the gate's negative control);
/// the tie rule stays "lower index first" either way, like the kernel's.
fn canonical_order(scores: &[f32], invert: bool) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        let by_score = if scores[a] == scores[b] {
            Ordering::Equal
        } else if invert {
            scores[a].partial_cmp(&scores[b]).expect("finite scores")
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
fn structure(label: &str, row_mask: &[u8], blocks: usize, tail: usize, pos: usize) -> Vec<usize> {
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

/// Runs one case's mask through one variant and returns the downloaded cells.
fn run_case(ctx: &DeviceContext, case: &Case, variant: i32) -> Vec<u8> {
    let scores = ctx.stream.clone_htod(&case.scores).expect("score upload");
    let rows = ctx.stream.clone_htod(&case.rows).expect("rows upload");
    let mut mask = ctx
        .stream
        .clone_htod(&vec![SENTINEL; case.rows.len() * case.tokens])
        .expect("mask buffer");
    let launch = if variant == QSA_SELECT_PRODUCTION {
        qsa_indexer_select_into(ctx, &scores, &rows, case.budget, case.tokens, &mut mask)
    } else {
        qsa_indexer_select_variant_into(
            ctx,
            &scores,
            &rows,
            case.budget,
            case.tokens,
            &mut mask,
            variant,
        )
    };
    launch.unwrap_or_else(|e| panic!("case {}: variant {variant} launch: {e:#}", case.index));
    ctx.sync().expect("sync");
    let host = ctx.stream.clone_dtoh(&mask).expect("mask download");
    assert!(
        host.iter().all(|cell| *cell <= 1),
        "case {}: variant {variant} left a cell unwritten or wrote past 1",
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
    /// Rows that keep at least one block, i.e. rows with a score prefix.
    keeping_rows: usize,
    tail_bits: usize,
    /// The fixture's own tail width summed over the rows, for the control.
    tail_expected: usize,
    keep_fewer_diff: usize,
    keep_more_diff: usize,
    causality_diff: usize,
    /// The fixture's own "columns past the row's token" summed over the rows.
    causality_expected: usize,
    invert_diff: usize,
    /// Truncating rows whose mask the inverted order changes.
    invert_rows: usize,
    /// Blocks production drops that the inverted order keeps.
    invert_dropped_kept: usize,
}

fn check_case(ctx: &DeviceContext, case: &Case, scenario: &Scenario) -> CaseReport {
    let budget_blocks = scenario.blocks_of(case.budget);
    let rows = case.rows.len();
    let tokens = case.tokens;

    let production = run_case(ctx, case, QSA_SELECT_PRODUCTION);
    let drop_tail = run_case(ctx, case, QSA_SELECT_DROP_TAIL);
    let keep_fewer = run_case(ctx, case, QSA_SELECT_KEEP_FEWER);
    let keep_more = run_case(ctx, case, QSA_SELECT_KEEP_MORE);
    let ignore_causality = run_case(ctx, case, QSA_SELECT_IGNORE_CAUSALITY);
    let invert = run_case(ctx, case, QSA_SELECT_INVERT_ORDER);

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
        let canonical = canonical_order(scores, false);
        let inverted = canonical_order(scores, true);
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
        }
        report.keeping_rows += usize::from(keep > 0);
        report.tail_expected += tail;

        // Control 1: dropping the tail differs exactly on the tail tokens.
        let want = host_mask(tokens, blocks, &canonical[..keep], 0);
        let got = &drop_tail[slot * tokens..][..tokens];
        assert_same_row(&format!("{label} (drop tail)"), got, &want);
        let diff = differing(got, ours);
        let want: Vec<usize> = (blocks * RATIO..blocks * RATIO + tail).collect();
        assert_eq!(
            diff, want,
            "{label}: the tail tokens are the only difference"
        );
        report.tail_bits += diff.len();

        // Control 2: one block fewer differs exactly on the last kept block.
        let fewer = keep.saturating_sub(1);
        let want = host_mask(tokens, blocks, &canonical[..fewer], tail);
        let got = &keep_fewer[slot * tokens..][..tokens];
        assert_same_row(&format!("{label} (keep fewer)"), got, &want);
        let diff = differing(got, ours);
        report.keep_fewer_diff += diff.len();
        if keep == 0 {
            assert!(diff.is_empty(), "{label}: nothing to drop");
        } else {
            let dropped = canonical[keep - 1] * RATIO;
            assert_eq!(
                diff,
                (dropped..dropped + RATIO).collect::<Vec<_>>(),
                "{label}: the block the budget had to drop is the only difference"
            );
        }

        // Control 3: one block more differs exactly on the truncating rows,
        // where an extra block exists; a row under the budget cannot change.
        let more = (keep + 1).min(blocks);
        let want = host_mask(tokens, blocks, &canonical[..more], tail);
        let got = &keep_more[slot * tokens..][..tokens];
        assert_same_row(&format!("{label} (keep more)"), got, &want);
        let diff = differing(got, ours);
        report.keep_more_diff += diff.len();
        if blocks > keep {
            let added = canonical[keep] * RATIO;
            assert_eq!(
                diff,
                (added..added + RATIO).collect::<Vec<_>>(),
                "{label}: the first block past the budget is the only difference"
            );
        } else {
            assert!(
                diff.is_empty(),
                "{label}: keeping more than the budget is capped at {blocks} blocks"
            );
        }

        // Control 4: ignoring causality differs exactly on the columns past the
        // row's own token, and nowhere else.
        let mut want = host_mask(tokens, blocks, &canonical[..keep], tail);
        want[pos + 1..].fill(1);
        let got = &ignore_causality[slot * tokens..][..tokens];
        assert_same_row(&format!("{label} (ignore causality)"), got, &want);
        let diff = differing(got, ours);
        let want: Vec<usize> = (pos + 1..tokens).collect();
        assert_eq!(
            diff, want,
            "{label}: only tokens past the row's own position may differ"
        );
        report.causality_diff += diff.len();
        report.causality_expected += tokens - 1 - pos;

        // Control 5: the inverted order reproduces a host-computed bottom-k
        // mask and differs from production on exactly the blocks the two orders
        // disagree on — every block production had to drop comes back, and the
        // top block goes.
        let got = &invert[slot * tokens..][..tokens];
        assert_same_row(
            &format!("{label} (invert)"),
            got,
            &host_mask(tokens, blocks, &inverted[..keep], tail),
        );
        let diff = differing(got, ours);
        report.invert_diff += diff.len();
        let mut traded: Vec<usize> = canonical[..keep]
            .iter()
            .filter(|block| !inverted[..keep].contains(*block))
            .copied()
            .collect();
        traded.extend(
            inverted[..keep]
                .iter()
                .filter(|block| !canonical[..keep].contains(*block))
                .copied(),
        );
        let mut want = Vec::new();
        for block in &traded {
            want.extend(block * RATIO..block * RATIO + RATIO);
        }
        want.sort_unstable();
        assert_eq!(
            diff, want,
            "{label}: the inverted order must trade exactly the blocks the two orders \
             disagree on"
        );
        if blocks > keep {
            report.invert_rows += usize::from(!diff.is_empty());
            report.invert_dropped_kept += canonical[keep..]
                .iter()
                .filter(|block| inverted[..keep].contains(*block))
                .count();
        } else {
            assert!(
                diff.is_empty(),
                "{label}: every block is kept, so order cannot matter"
            );
        }
    }
    report
}

/// The fixture's provenance, independent of any device: this half must never be
/// skippable.
#[test]
fn fixture_records_the_frozen_reference() {
    let bytes = std::fs::read(fixture_path()).expect("read fixture");
    assert_provenance(&bytes);
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
            "case {index}: {} rows, {} bitwise, {} tie, {} truncating (blocks > budget/4); \
             tokens {}, budget {} ({} blocks); controls differ: drop-tail {} bits, \
             keep-fewer {} bits, keep-more {} bits, ignore-causality {} bits, \
             invert {} bits",
            report.rows,
            report.bitwise,
            report.tie,
            report.truncating,
            case.tokens,
            case.budget,
            scenario.blocks_of(case.budget),
            report.tail_bits,
            report.keep_fewer_diff,
            report.keep_more_diff,
            report.causality_diff,
            report.invert_diff,
        );
        total.rows += report.rows;
        total.bitwise += report.bitwise;
        total.tie += report.tie;
        total.truncating += report.truncating;
        total.tail_bits += report.tail_bits;
        total.tail_expected += report.tail_expected;
        total.keeping_rows += report.keeping_rows;
        total.causality_expected += report.causality_expected;
        total.keep_fewer_diff += report.keep_fewer_diff;
        total.keep_more_diff += report.keep_more_diff;
        total.causality_diff += report.causality_diff;
        total.invert_diff += report.invert_diff;
        total.invert_rows += report.invert_rows;
        total.invert_dropped_kept += report.invert_dropped_kept;
    }
    println!(
        "selection parity over {CASES} cases: {} rows, {} bitwise, {} up to ties, \
         {} truncating; controls differ from production: drop-tail {} bits, \
         keep-fewer {} bits, keep-more {} bits, ignore-causality {} bits, invert {} \
         bits ({} blocks production dropped that the inverted order keeps)",
        total.rows,
        total.bitwise,
        total.tie,
        total.truncating,
        total.tail_bits,
        total.keep_fewer_diff,
        total.keep_more_diff,
        total.causality_diff,
        total.invert_diff,
        total.invert_dropped_kept,
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
    // Computed floors: each control must bite exactly as widely as the fixture
    // allows, and the two truncation-dependent controls need truncating rows to
    // mean anything.
    assert!(
        total.tail_expected > 0,
        "the fixture must give some row a tail to drop"
    );
    assert_eq!(
        total.tail_bits, total.tail_expected,
        "drop-tail must clear exactly the fixture's tail bits"
    );
    assert!(
        total.keeping_rows > 0,
        "the fixture must give some row a block to keep"
    );
    assert_eq!(
        total.keep_fewer_diff,
        RATIO * total.keeping_rows,
        "keep-fewer must remove exactly one block per row that keeps one"
    );
    assert_eq!(
        total.keep_more_diff,
        RATIO * total.truncating,
        "keep-more must add exactly one block on every truncating row"
    );
    assert!(
        total.causality_expected > 0,
        "the fixture must give some row a token past its own"
    );
    assert_eq!(
        total.causality_diff, total.causality_expected,
        "ignore-causality must set exactly the columns past each row's token"
    );
    assert!(
        total.truncating > 0,
        "the fixture must express a dropped block"
    );
    assert_eq!(
        total.invert_rows, total.truncating,
        "inverting must change every truncating row and no other"
    );
    assert_eq!(
        total.invert_dropped_kept, total.truncating,
        "inverting must keep every block production dropped"
    );
    assert!(
        total.invert_diff >= RATIO * total.truncating,
        "inverting must differ on at least one block per truncating row"
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
        variant: i32,
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
                variant,
                std::ptr::null_mut(),
            )
        }
    }
    let scores = std::ptr::NonNull::<f32>::dangling().as_ptr();
    let rows = std::ptr::NonNull::<i32>::dangling().as_ptr();
    let mask = std::ptr::NonNull::<u8>::dangling().as_ptr();
    let valid = |budget, variant| launch(scores, rows, mask, 1, 8, 2, budget, variant);

    let refuse = |label: &str, result: CUresult| {
        assert_eq!(
            result,
            CUresult::CUDA_ERROR_INVALID_VALUE,
            "{label} must be refused before the launch"
        );
    };
    refuse(
        "null scores",
        launch(std::ptr::null(), rows, mask, 1, 8, 2, 8, 0),
    );
    refuse(
        "null rows",
        launch(scores, std::ptr::null(), mask, 1, 8, 2, 8, 0),
    );
    refuse(
        "null mask",
        launch(scores, rows, std::ptr::null_mut(), 1, 8, 2, 8, 0),
    );
    for query_rows in [-1, 0] {
        refuse(
            "query rows",
            launch(scores, rows, mask, query_rows, 8, 2, 8, 0),
        );
    }
    for tokens in [-1, 0] {
        refuse("tokens", launch(scores, rows, mask, 1, tokens, 2, 8, 0));
    }
    refuse(
        "negative score columns",
        launch(scores, rows, mask, 1, 8, -1, 8, 0),
    );
    for budget in [-4, 0, 2, 7, 2052] {
        refuse("budget", valid(budget, 0));
    }
    for variant in [-1, 6] {
        refuse("variant", valid(8, variant));
    }
    // A row sees up to tokens / 4 blocks, so one score column cannot serve an
    // eight-token row.
    refuse(
        "narrow score row",
        launch(scores, rows, mask, 1, 8, 1, 8, 0),
    );
    // The exclusion bitmap is one bit per block: 1.6M tokens is past the 48 KiB
    // dynamic-shared-memory ceiling a launch may use without an opt-in.
    refuse(
        "oversized exclusion bitmap",
        launch(scores, rows, mask, 1, 1_600_000, 400_000, 8, 0),
    );
    // A launch that passes every guard would dereference these placeholder
    // pointers, so only refusals belong in this test.
}
