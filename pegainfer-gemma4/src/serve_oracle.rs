//! GPU + checkpoint gates for the KV serving path. The A/B oracle needs only
//! the existing golden fixture; the generate gate needs the HF `generate()`
//! fixture dumped on the test box.

use anyhow::Result;

use super::*;
use crate::forward::full_forward;
use crate::kv::admit_tokens;
use crate::testkit::GOLDEN_PATH;
use crate::testkit::METADATA_KEY;
use crate::testkit::assert_checkpoint_matches;
use crate::testkit::f32_tensor;
use crate::testkit::i32_tensor;
use crate::testkit::log_softmax_at;
use crate::testkit::model_path;

/// Compare one logit row against the oracle's: both must be finite (a NaN
/// would rank highest under `total_cmp` and be ignored by `f32::max`), the
/// argmaxes must agree, and the worst absolute gap is what the caller gates.
fn compare_row(ours: &[f32], theirs: &[f32], what: &str) -> f32 {
    assert!(
        ours.iter().chain(theirs.iter()).all(|v| v.is_finite()),
        "{what}: non-finite logit"
    );
    assert_eq!(argmax(ours), argmax(theirs), "{what}: argmax diverged");
    ours.iter()
        .zip(theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn fixture_manifest(bytes: &[u8], key: &str) -> serde_json::Value {
    let (_, meta) = safetensors::SafeTensors::read_metadata(bytes).expect("fixture metadata");
    serde_json::from_str(
        meta.metadata()
            .as_ref()
            .expect("fixture metadata map")
            .get(key)
            .expect("fixture manifest key"),
    )
    .expect("parse fixture manifest")
}

fn stack_with(max_context: usize, pages: usize) -> (DeviceContext, GemmaServe, String) {
    let dir = model_path();
    let config = Gemma4Config::from_file(&dir).expect("config");
    let (weights, _) = Gemma4Weights::from_safetensors(&dir, 0, config).expect("load 12B weights");
    let ctx = DeviceContext::new_with_device(0).expect("device context");
    let serve = GemmaServe::new(&ctx, weights, max_context, pages, pages).expect("serve");
    (ctx, serve, dir)
}

fn load_stack() -> (DeviceContext, GemmaServe, String) {
    // One request at the window, plus each pool's padding page.
    stack_with(1024, 66)
}

/// What agreement is available at this depth, measured on the reference
/// itself: the largest gap its two backends have with each other over the
/// ids both rank, and how often they pick the same top-1. Neither certifies
/// anything smaller as correct; they say what a correct implementation can
/// still be asked for.
fn backend_floor(
    ref_ids: &[i32],
    ref_lps: &[f32],
    eager_ids: &[i32],
    eager_lps: &[f32],
    positions: usize,
    top_k: usize,
) -> (f32, usize) {
    let mut floor = 0.0f32;
    let mut top1 = 0usize;
    for pos in 0..positions {
        let eager: std::collections::HashMap<i32, f32> = (0..top_k)
            .map(|k| (eager_ids[pos * top_k + k], eager_lps[pos * top_k + k]))
            .collect();
        for k in 0..top_k {
            if let Some(&e) = eager.get(&ref_ids[pos * top_k + k]) {
                floor = floor.max((ref_lps[pos * top_k + k] - e).abs());
            }
        }
        if ref_ids[pos * top_k] == eager_ids[pos * top_k] {
            top1 += 1;
        }
    }
    (floor, top1)
}

const WINDOW_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../test_data/gemma4-12b-hf-window-golden.safetensors"
);

/// One case's run through the serving path: the prompt prefilled in steps of
/// `chunk` tokens (the whole prompt when zero), then its teacher-forced
/// continuation one token at a time.
struct Run {
    rows: Vec<Vec<f32>>,
    kv_len: usize,
    local_pages: usize,
    global_pages: usize,
    /// Whether a multi-token step ran with the resident row already shifted.
    shifted_multi_token: bool,
}

fn run_case(
    ctx: &DeviceContext,
    serve: &GemmaServe,
    fixture: &safetensors::SafeTensors<'_>,
    case: &str,
    chunk: usize,
) -> Run {
    let (_, prompt_i32) = i32_tensor(fixture, &format!("{case}_prompt"));
    let (_, teacher_i32) = i32_tensor(fixture, &format!("{case}_teacher"));
    let prompt: Vec<u32> = prompt_i32
        .iter()
        .map(|&t| u32::try_from(t).expect("token id"))
        .collect();
    let step_size = if chunk == 0 { prompt.len() } else { chunk };
    let last_chunk = prompt.len().div_ceil(step_size) - 1;

    let mut kv = serve.alloc_kv();
    let mut arena = serve
        .alloc_step_arena(ctx, 1, false)
        .expect("oracle step arena");
    let mut rows = Vec::with_capacity(teacher_i32.len() + 1);
    let mut shifted_multi_token = false;
    for (i, piece) in prompt.chunks(step_size).enumerate() {
        admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, piece.len())
            .expect("admit prompt");
        shifted_multi_token |= kv.local.origin_pages() > 0 && piece.len() > 1;
        let logits = serve
            .step(ctx, &mut kv, piece, LogitsSpan::LastRow)
            .expect("prefill");
        if i == last_chunk {
            rows.push(logits.to_host(ctx).expect("D2H"));
        }
    }
    for &t in &teacher_i32 {
        let token = u32::try_from(t).expect("token id");
        rows.push(decode_serving(serve, ctx, &mut arena, &mut kv, token).expect("decode"));
    }
    Run {
        rows,
        kv_len: prompt.len() + teacher_i32.len(),
        local_pages: kv.local.held_pages(),
        global_pages: kv.global.held_pages(),
        shifted_multi_token,
    }
}

/// A case's reference rows, with the tolerance and the top-1 floor its own
/// two backends earn.
fn reference(
    fixture: &safetensors::SafeTensors<'_>,
    case: &str,
) -> (Vec<i32>, Vec<f32>, usize, usize, f32, usize) {
    let (shape, ids) = i32_tensor(fixture, &format!("{case}_sdpa_ids"));
    let (_, lps) = f32_tensor(fixture, &format!("{case}_sdpa_logprobs"));
    let (_, eager_ids) = i32_tensor(fixture, &format!("{case}_eager_ids"));
    let (_, eager_lps) = f32_tensor(fixture, &format!("{case}_eager_logprobs"));
    let (positions, top_k) = (shape[0], shape[1]);
    let (floor, backend_top1) = backend_floor(&ids, &lps, &eager_ids, &eager_lps, positions, top_k);
    (
        ids,
        lps,
        positions,
        top_k,
        (2.0 * floor).max(1.0),
        backend_top1,
    )
}

// Worst absolute gap against the reference's own top-k, plus how often our
// argmax lands on its top-1.
fn score_rows(
    rows: &[Vec<f32>],
    ref_ids: &[i32],
    ref_lps: &[f32],
    top_k: usize,
    case: &str,
) -> (f32, usize) {
    let mut max_abs = 0.0f32;
    let mut top1 = 0usize;
    for (pos, row) in rows.iter().enumerate() {
        let ids = &ref_ids[pos * top_k..(pos + 1) * top_k];
        let (ours, argmax) = log_softmax_at(row, ids);
        assert!(
            ours.iter().all(|v| v.is_finite()),
            "{case}: non-finite logprob at position {pos}"
        );
        if argmax == usize::try_from(ids[0]).expect("token id") {
            top1 += 1;
        }
        for k in 0..top_k {
            max_abs = max_abs.max((ours[k] - ref_lps[pos * top_k + k]).abs());
        }
    }
    (max_abs, top1)
}

const LONGCTX_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../test_data/gemma4-12b-hf-longctx-golden.safetensors"
);

/// A case's sdpa rows with a borrowed tolerance: where eager could not fit
/// next to the tower, the widest dual-backend case lends its floor — the
/// reference says what agreement is reachable, not that less is correct.
fn reference_sdpa_only(
    fixture: &safetensors::SafeTensors<'_>,
    case: &str,
    tolerance: f32,
    backend_top1_share: f64,
) -> (Vec<i32>, Vec<f32>, usize, usize, f32, usize) {
    let (shape, ids) = i32_tensor(fixture, &format!("{case}_sdpa_ids"));
    let (_, lps) = f32_tensor(fixture, &format!("{case}_sdpa_logprobs"));
    let (positions, top_k) = (shape[0], shape[1]);
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let backend_top1 = (backend_top1_share * positions as f64).floor() as usize;
    (ids, lps, positions, top_k, tolerance, backend_top1)
}

/// The raised ceiling's numeric waypoints: 16384 and 32768 teacher-forced
/// against the Hugging Face reference — proportional rope and global
/// attention far past the window fixture's 4096 — each gated at twice its
/// own backends' measured gap, the widest dual-backend floor standing in
/// where eager could not fit. The widest case runs again in 2048-token
/// chunks, the raised ceiling's production prefill shape.
#[test]
#[ignore = "requires the pinned 12B checkpoint and the longctx fixture"]
fn longctx_waypoints_match_hf() {
    // A 32776-token prefill holds every local page at once (append then
    // attend); pools sized for one request plus padding.
    let (ctx, serve, dir) = stack_with(32900, 2200);
    let bytes = std::fs::read(LONGCTX_FIXTURE).expect("read longctx fixture");
    let golden = fixture_manifest(
        &std::fs::read(GOLDEN_PATH).expect("read golden"),
        METADATA_KEY,
    );
    assert_checkpoint_matches(&golden, &dir);
    let manifest = fixture_manifest(&bytes, "gemma4_longctx_golden");
    assert_eq!(
        manifest["revision"], golden["revision"],
        "the longctx fixture was dumped from a different revision than the golden one"
    );
    let eager_skipped: Vec<String> = manifest["eager_skipped"]
        .as_array()
        .expect("eager_skipped list")
        .iter()
        .map(|v| v.as_str().expect("case name").to_string())
        .collect();
    let fixture = safetensors::SafeTensors::deserialize(&bytes).expect("parse fixture");
    let page = serve.local_pool.layout().page_size;

    // Neither waypoint fits eager next to the tower on this device, so no
    // in-fixture dual-backend floor exists; the window fixture's deepest
    // dual case lends its own — the widest measured agreement bound
    // available. A depth-grown gap past it fails loud and is widened only
    // with a written justification.
    let window_bytes = std::fs::read(WINDOW_FIXTURE).expect("read window fixture");
    let window_manifest = fixture_manifest(&window_bytes, "gemma4_window_golden");
    assert_eq!(
        window_manifest["revision"], golden["revision"],
        "the window fixture was dumped from a different revision than the golden one"
    );
    assert_eq!(
        manifest["transformers"], window_manifest["transformers"],
        "a borrowed floor is only meaningful under the donor's own reference release"
    );
    let window_fixture =
        safetensors::SafeTensors::deserialize(&window_bytes).expect("parse window fixture");
    let (_, _, donor_positions, _, donor_tolerance, donor_top1) =
        reference(&window_fixture, "w4096");
    #[allow(clippy::cast_precision_loss)]
    let donor_share = donor_top1 as f64 / donor_positions as f64;

    let mut over: Vec<String> = Vec::new();
    for (case, chunk) in [("w16384", 0), ("w32768", 0), ("w32768", 2048)] {
        let label = if chunk == 0 {
            case.to_string()
        } else {
            format!("{case}-chunked")
        };
        let (ref_ids, ref_lps, positions, top_k, tolerance, backend_top1) =
            if eager_skipped.contains(&case.to_string()) {
                reference_sdpa_only(&fixture, case, donor_tolerance, donor_share)
            } else {
                reference(&fixture, case)
            };
        let run = run_case(&ctx, &serve, &fixture, case, chunk);
        assert_eq!(run.rows.len(), positions, "{label}: fixture positions");
        assert_eq!(
            chunk > 0,
            run.shifted_multi_token,
            "{label}: a shifted multi-token step is exactly what chunking adds"
        );

        let (max_abs, top1) = score_rows(&run.rows, &ref_ids, &ref_lps, top_k, &label);
        eprintln!(
            "{label}: max |dlogprob| {max_abs} (tol {tolerance:.2}), top-1 {top1}/{positions} \
             (backend bar {backend_top1}/{positions}), local pages {}, global {}",
            run.local_pages, run.global_pages
        );
        assert!(
            top1 >= backend_top1,
            "{label}: top-1 {top1}/{positions} below the backends' own \
             {backend_top1}/{positions}"
        );
        let released = run.kv_len.saturating_sub(serve.sliding_window) / page;
        assert_eq!(
            run.local_pages,
            run.kv_len.div_ceil(page) - released,
            "{label}: resident pages after {released} released"
        );
        assert_eq!(
            run.global_pages,
            run.kv_len.div_ceil(page),
            "{label}: the global family must keep every page"
        );
        if max_abs > tolerance {
            over.push(format!("{label} ({max_abs} > {tolerance})"));
        }
    }
    assert!(
        over.is_empty(),
        "cases over their calibrated floor: {over:?}"
    );
}

/// Gated distribution-level because a greedy chain is not reachable at this
/// depth: the reference's own backends continue the same prompt in different
/// directions, so each case is gated at twice its measured sdpa-vs-eager
/// gap. A window-semantics bug misses by orders more than that floor.
#[test]
#[ignore = "requires the pinned 12B checkpoint and the window fixture"]
fn window_crossing_matches_hf() {
    // A 4096-token prefill holds every local page at once (append then
    // attend); decode releases down to the window afterwards.
    let (ctx, serve, dir) = stack_with(4200, 300);
    let bytes = std::fs::read(WINDOW_FIXTURE).expect("read window fixture");
    // The golden fixture fingerprints the checkpoint files; the window one
    // only names a revision, so pin with the former and cross-check the
    // latter against it.
    let golden = fixture_manifest(
        &std::fs::read(GOLDEN_PATH).expect("read golden"),
        METADATA_KEY,
    );
    assert_checkpoint_matches(&golden, &dir);
    let window = fixture_manifest(&bytes, "gemma4_window_golden");
    assert_eq!(
        window["revision"], golden["revision"],
        "the window fixture was dumped from a different revision than the golden one"
    );
    let fixture = safetensors::SafeTensors::deserialize(&bytes).expect("parse fixture");
    let page = serve.local_pool.layout().page_size;

    let mut over: Vec<String> = Vec::new();
    // Single-shot prefill for every case, then the widest one again in
    // window-sized chunks: a prefill that fits in one step always runs at
    // origin 0, since the front is only released afterwards, so chunking is
    // the only way to reach a multi-token step with the row already shifted.
    for (case, chunk) in [
        ("w1023", 0),
        ("w1024", 0),
        ("w1025", 0),
        ("w4096", 0),
        ("w4096", 1024),
    ] {
        let label = if chunk == 0 {
            case.to_string()
        } else {
            format!("{case}-chunked")
        };
        let (ref_ids, ref_lps, positions, top_k, tolerance, backend_top1) =
            reference(&fixture, case);
        let run = run_case(&ctx, &serve, &fixture, case, chunk);
        assert_eq!(run.rows.len(), positions, "{label}: fixture positions");
        assert_eq!(
            chunk > 0,
            run.shifted_multi_token,
            "{label}: a shifted multi-token step is exactly what chunking adds"
        );

        let (max_abs, top1) = score_rows(&run.rows, &ref_ids, &ref_lps, top_k, &label);
        eprintln!(
            "{label}: max |dlogprob| {max_abs} (tol {tolerance:.2}), top-1 {top1}/{positions} \
             (backends agree {backend_top1}/{positions}), local pages {}, global {}",
            run.local_pages, run.global_pages
        );
        // A ranking regression can hide under a logprob tolerance this wide,
        // so top-1 is gated too — against what the reference's own backends
        // manage with each other, which is all that is reachable here.
        assert!(
            top1 >= backend_top1,
            "{label}: top-1 {top1}/{positions} below the backends' own \
             {backend_top1}/{positions}"
        );
        // Exactly what the release rule leaves resident: every page the
        // frontier accounts for, minus the ones whose last token has aged out
        // of every future window. One page late still fails here.
        let released = run.kv_len.saturating_sub(serve.sliding_window) / page;
        assert_eq!(
            run.local_pages,
            run.kv_len.div_ceil(page) - released,
            "{label}: resident pages after {released} released"
        );
        assert_eq!(
            run.global_pages,
            run.kv_len.div_ceil(page),
            "{label}: the global family must keep every page"
        );
        if max_abs > tolerance {
            over.push(format!("{label} ({max_abs} > {tolerance})"));
        }
    }
    assert!(
        over.is_empty(),
        "cases over their calibrated floor: {over:?}"
    );
}

/// `window_left` masks out-of-window keys whether or not their pages are
/// still resident, so releasing them need not change a single generated
/// token.
#[test]
#[ignore = "requires the pinned 12B checkpoint and the window fixture"]
fn eviction_is_footprint_only() {
    let (ctx, mut serve, _dir) = stack_with(1300, 120);
    let bytes = std::fs::read(WINDOW_FIXTURE).expect("read window fixture");
    let fixture = safetensors::SafeTensors::deserialize(&bytes).expect("parse fixture");
    let (_, prompt_i32) = i32_tensor(&fixture, "w1023_prompt");
    let prompt: Vec<u32> = prompt_i32
        .iter()
        .map(|&t| u32::try_from(t).expect("token id"))
        .collect();

    let run = |serve: &GemmaServe| -> (Vec<u32>, usize) {
        let mut kv = serve.alloc_kv();
        let mut arena = serve
            .alloc_step_arena(&ctx, 1, false)
            .expect("oracle step arena");
        admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, prompt.len())
            .expect("admit prompt");
        let logits = serve
            .step(&ctx, &mut kv, &prompt, LogitsSpan::LastRow)
            .expect("prefill");
        let mut next = argmax_last(&ctx, &logits).expect("argmax");
        let mut tokens = vec![next];
        for _ in 1..30 {
            let row = decode_serving(serve, &ctx, &mut arena, &mut kv, next).expect("decode");
            next = u32::try_from(argmax(&row)).expect("token id");
            tokens.push(next);
        }
        (tokens, kv.local.held_pages())
    };

    let (evicted, pages_evicted) = run(&serve);
    serve.set_release_for_test(false);
    let (retained, pages_retained) = run(&serve);
    assert_eq!(evicted, retained, "releasing changed the generated tokens");
    eprintln!("local pages at end: released {pages_evicted} vs retained {pages_retained}");
    assert!(
        pages_evicted < pages_retained,
        "release must shrink the resident footprint ({pages_evicted} vs {pages_retained})"
    );
}

/// The paged serving path against the no-KV oracle forward on the same
/// weights and tokens: each decode step is compared against a full recompute
/// of the grown sequence, so a KV write that drifts shows up immediately.
#[test]
#[ignore = "requires the pinned 12B checkpoint via PEGAINFER_TEST_MODEL_PATH and a GPU"]
fn serve_matches_oracle_forward() {
    let (ctx, serve, dir) = load_stack();
    let fixture_bytes = std::fs::read(GOLDEN_PATH).expect("read fixture");
    let fixture = safetensors::SafeTensors::deserialize(&fixture_bytes).expect("parse fixture");
    let (_, tokens_i32) = i32_tensor(&fixture, "short_tokens");
    let mut tokens: Vec<u32> = tokens_i32
        .iter()
        .map(|&t| u32::try_from(t).expect("token id"))
        .collect();

    // The same fixture that carries the tokens fingerprints the checkpoint
    // they were dumped from.
    assert_checkpoint_matches(&fixture_manifest(&fixture_bytes, METADATA_KEY), &dir);
    let config = &serve.weights.config;
    let local_geom = LayerGeometry::local_of(config);
    let global_geom = LayerGeometry::global_of(config);
    let (scos, ssin) = pegainfer_core::rope::precompute_rope(
        &ctx,
        &RopeTableSpec {
            rotary_dim: local_geom.head_dim,
            frequency_dim: local_geom.head_dim,
            max_seq_len: 1024,
            theta: config.sliding_rope_theta,
        },
    )
    .expect("sliding tables");
    let (gcos, gsin) = build_proportional_rope_tables(
        &ctx,
        config.global_rope_theta,
        global_geom.head_dim,
        config.global_rotary_dim,
        1024,
    )
    .expect("global tables");

    let oracle = |tokens: &[u32]| -> Vec<f32> {
        let logits = full_forward(
            &ctx,
            &serve.weights,
            tokens,
            (&scos, &ssin),
            (&gcos, &gsin),
            1024,
        )
        .expect("oracle forward");
        logits.to_host(&ctx).expect("D2H")
    };

    let mut kv = serve.alloc_kv();
    admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, tokens.len())
        .expect("admit prompt");
    let serve_logits = serve
        .step(&ctx, &mut kv, &tokens, LogitsSpan::All)
        .expect("serve prefill");
    let vocab = serve_logits.hidden_dim;
    let serve_host = serve_logits.to_host(&ctx).expect("D2H");
    let oracle_host = oracle(&tokens);
    let mut max_abs = 0.0f32;
    for pos in 0..tokens.len() {
        let s = &serve_host[pos * vocab..(pos + 1) * vocab];
        let o = &oracle_host[pos * vocab..(pos + 1) * vocab];
        max_abs = max_abs.max(compare_row(s, o, &format!("prefill pos {pos}")));
    }
    eprintln!(
        "prefill: {} positions, max |dlogit| {max_abs}",
        tokens.len()
    );
    // Calibrated on the box: different kernels (fused paged prep + batch
    // prefill vs two-pass contiguous + single_prefill) measured 1.31
    // peak over prefill and 0.95 over decode on softcapped logits.
    assert!(
        max_abs <= 2.0,
        "prefill |dlogit| {max_abs} above calibrated 2.0"
    );

    // Feed the ORACLE's continuation each step so one divergence cannot
    // cascade; its last row doubles as the next step's greedy pick.
    let mut oracle_last = oracle_host[(tokens.len() - 1) * vocab..tokens.len() * vocab].to_vec();
    let mut arena = serve
        .alloc_step_arena(&ctx, 1, false)
        .expect("oracle step arena");
    for step in 0..4 {
        let next = u32::try_from(argmax(&oracle_last)).expect("token id");
        let step_host =
            decode_serving(&serve, &ctx, &mut arena, &mut kv, next).expect("serve decode step");
        tokens.push(next);
        let o = oracle(&tokens);
        oracle_last = o[(tokens.len() - 1) * vocab..tokens.len() * vocab].to_vec();
        let s_row = &step_host[0..vocab];
        let m = compare_row(s_row, &oracle_last, &format!("decode step {step}"));
        eprintln!("decode step {step}: max |dlogit| {m}");
        assert!(
            m <= 2.0,
            "decode step {step} |dlogit| {m} above calibrated 2.0"
        );
    }
}

/// DoD gate: greedy continuation matches HF `generate()` token for
/// token on three prompts. The fixture is dumped on the box by
/// tools/accuracy/dump_gemma4_generate.py (prompt + up to 50 greedy
/// tokens per case).
#[test]
#[ignore = "requires the pinned 12B checkpoint and the generate fixture"]
fn greedy_matches_hf_generate() {
    let (ctx, serve, dir) = load_stack();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../test_data/gemma4-12b-generate.safetensors"
    );
    let bytes = std::fs::read(path).expect("read generate fixture (dump on the box first)");
    // Provenance: the golden fixture fingerprints the checkpoint files, so
    // it pins what is loaded here; the generate fixture then has to name
    // that same revision, or these tokens came from another model.
    let golden_bytes = std::fs::read(GOLDEN_PATH).expect("read golden fixture");
    let golden = fixture_manifest(&golden_bytes, METADATA_KEY);
    assert_checkpoint_matches(&golden, &dir);
    let generate = fixture_manifest(&bytes, "gemma4_generate");
    assert_eq!(
        generate["revision"], golden["revision"],
        "the generate fixture was dumped from a different revision than the golden one"
    );
    let fixture = safetensors::SafeTensors::deserialize(&bytes).expect("parse fixture");
    let mut diverged: Vec<String> = Vec::new();
    for case in ["a", "b", "c"] {
        let (_, prompt_i32) = i32_tensor(&fixture, &format!("{case}_prompt"));
        let (_, expect_i32) = i32_tensor(&fixture, &format!("{case}_generated"));
        let prompt: Vec<u32> = prompt_i32
            .iter()
            .map(|&t| u32::try_from(t).expect("token id"))
            .collect();
        let mut kv = serve.alloc_kv();
        let ours = generate_greedy(&serve, &ctx, &mut kv, &prompt, expect_i32.len())
            .expect("greedy generation");
        let expect: Vec<u32> = expect_i32
            .iter()
            .map(|&t| u32::try_from(t).expect("token id"))
            .collect();
        assert_eq!(
            ours.len(),
            expect.len(),
            "case {case}: generated {} tokens against the fixture's {}",
            ours.len(),
            expect.len()
        );
        match ours.iter().zip(&expect).position(|(a, b)| a != b) {
            None => eprintln!("case {case}: {} tokens match HF generate", expect.len()),
            Some(at) => {
                eprintln!(
                    "case {case}: diverged at {at}/{}: ours {:?} vs HF {:?}",
                    expect.len(),
                    &ours[at..(at + 4).min(ours.len())],
                    &expect[at..(at + 4).min(expect.len())]
                );
                diverged.push(format!("{case}@{at}"));
            }
        }
    }
    assert!(
        diverged.is_empty(),
        "cases diverged from HF generate: {diverged:?}"
    );
}

fn mixed_gate_argmax(host: &[f32], row: usize, vocab: usize) -> u32 {
    u32::try_from(argmax(&host[row * vocab..(row + 1) * vocab])).expect("token id")
}

fn mixed_gate_decode_rounds(
    ctx: &DeviceContext,
    serve: &GemmaServe,
    arena: &mut StepArena,
    lanes: &mut Vec<(usize, GemmaKv, u32)>,
    produced: &mut [Vec<u32>],
    budgets: &[usize],
    rounds: usize,
) {
    for _ in 0..rounds {
        if lanes.is_empty() {
            break;
        }
        let tokens: Vec<u32> = lanes.iter().map(|(_, _, next)| *next).collect();
        for (_, kv, _) in lanes.iter_mut() {
            admit_tokens(&serve.local_pool, &serve.global_pool, kv, 1).expect("admit decode");
        }
        let (vocab, host);
        {
            let mut kvs: Vec<&mut GemmaKv> = lanes.iter_mut().map(|(_, kv, _)| kv).collect();
            let logits = serve
                .decode_batch_step(ctx, arena, &mut kvs, &tokens)
                .expect("batched decode");
            vocab = logits.hidden_dim;
            host = logits.to_host(ctx).expect("logits D2H");
        }
        let mut retire: Vec<usize> = Vec::new();
        for (row, (req, _, next)) in lanes.iter_mut().enumerate() {
            let token = mixed_gate_argmax(&host, row, vocab);
            produced[*req].push(token);
            if produced[*req].len() >= budgets[*req] {
                retire.push(row);
            } else {
                *next = token;
            }
        }
        for row in retire.into_iter().rev() {
            lanes.swap_remove(row);
        }
    }
}

/// A mixed admission — the prompt sharing one step with the live decode
/// batch — must be token-exact against the serial path, for the newcomer
/// (logits row 0) and every incumbent lane (rows 1..), across two
/// admissions at different batch sizes and the pure-decode rounds between
/// them.
#[test]
#[ignore = "requires the pinned 12B checkpoint and the generate fixture"]
fn mixed_step_matches_serial_greedy() {
    let (ctx, serve, _dir) = stack_with(1024, 200);
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../test_data/gemma4-12b-generate.safetensors"
    );
    let bytes = std::fs::read(path).expect("read generate fixture (dump on the box first)");
    let fixture = safetensors::SafeTensors::deserialize(&bytes).expect("parse fixture");
    let cases = ["a", "b", "c"];
    let prompts: Vec<Vec<u32>> = cases
        .iter()
        .map(|case| {
            let (_, prompt_i32) = i32_tensor(&fixture, &format!("{case}_prompt"));
            prompt_i32
                .iter()
                .map(|&t| u32::try_from(t).expect("token id"))
                .collect()
        })
        .collect();
    let budgets = [50usize, 37, 44];

    let serial: Vec<Vec<u32>> = prompts
        .iter()
        .zip(budgets)
        .map(|(prompt, budget)| {
            let mut kv = serve.alloc_kv();
            generate_greedy(&serve, &ctx, &mut kv, prompt, budget).expect("serial greedy")
        })
        .collect();

    let mut arena = serve.alloc_step_arena(&ctx, 4, false).expect("step arena");
    let mut lanes: Vec<(usize, GemmaKv, u32)> = Vec::new();
    let mut produced: Vec<Vec<u32>> = vec![Vec::new(); cases.len()];

    // Lane a arrives alone — the plain prefill path — then three decode
    // rounds so the mixed admissions below meet a warm batch.
    {
        let mut kv = serve.alloc_kv();
        admit_tokens(
            &serve.local_pool,
            &serve.global_pool,
            &mut kv,
            prompts[0].len(),
        )
        .expect("admit prompt a");
        let logits = serve
            .step(&ctx, &mut kv, &prompts[0], LogitsSpan::LastRow)
            .expect("prefill a");
        let first = argmax_last(&ctx, &logits).expect("first token");
        produced[0].push(first);
        lanes.push((0, kv, first));
    }
    mixed_gate_decode_rounds(
        &ctx,
        &serve,
        &mut arena,
        &mut lanes,
        &mut produced,
        &budgets,
        3,
    );

    // Admissions b and c ride the live batch TOGETHER: the k=2 step
    // prefills both prompts as segments and samples both first tokens
    // (logits rows 0..2) ahead of the incumbent rows.
    {
        let mut kv_b = serve.alloc_kv();
        admit_tokens(
            &serve.local_pool,
            &serve.global_pool,
            &mut kv_b,
            prompts[1].len(),
        )
        .expect("admit prompt b");
        let mut kv_c = serve.alloc_kv();
        admit_tokens(
            &serve.local_pool,
            &serve.global_pool,
            &mut kv_c,
            prompts[2].len(),
        )
        .expect("admit prompt c");
        for (_, lane_kv, _) in &mut lanes {
            admit_tokens(&serve.local_pool, &serve.global_pool, lane_kv, 1).expect("admit decode");
        }
        let tokens: Vec<u32> = lanes.iter().map(|(_, _, next)| *next).collect();
        let (vocab, host);
        {
            let mut kvs: Vec<&mut GemmaKv> = lanes.iter_mut().map(|(_, kv, _)| kv).collect();
            let mut prefills = [
                (&mut kv_b, prompts[1].as_slice()),
                (&mut kv_c, prompts[2].as_slice()),
            ];
            let logits = serve
                .mixed_prefill_decode_step(&ctx, &mut arena, &mut prefills, &mut kvs, &tokens)
                .expect("k=2 mixed step");
            vocab = logits.hidden_dim;
            host = logits.to_host(&ctx).expect("logits D2H");
        }
        let mut retire: Vec<usize> = Vec::new();
        for (row, (req, _, next)) in lanes.iter_mut().enumerate() {
            let token = mixed_gate_argmax(&host, row + 2, vocab);
            produced[*req].push(token);
            if produced[*req].len() >= budgets[*req] {
                retire.push(row);
            } else {
                *next = token;
            }
        }
        for row in retire.into_iter().rev() {
            lanes.swap_remove(row);
        }
        let first_b = mixed_gate_argmax(&host, 0, vocab);
        produced[1].push(first_b);
        lanes.push((1, kv_b, first_b));
        let first_c = mixed_gate_argmax(&host, 1, vocab);
        produced[2].push(first_c);
        lanes.push((2, kv_c, first_c));
        mixed_gate_decode_rounds(
            &ctx,
            &serve,
            &mut arena,
            &mut lanes,
            &mut produced,
            &budgets,
            3,
        );
    }

    // Drain everyone to their budgets.
    mixed_gate_decode_rounds(
        &ctx,
        &serve,
        &mut arena,
        &mut lanes,
        &mut produced,
        &budgets,
        usize::MAX,
    );
    for (i, case) in cases.iter().enumerate() {
        assert_eq!(
            produced[i], serial[i],
            "case {case}: mixed admission diverged from the serial path"
        );
        eprintln!("case {case}: {} tokens mixed == serial", serial[i].len());
    }
}

/// A mixed admission whose prompt crosses the 1024-token sliding window —
/// the window front releases inside the same step that live lanes decode
/// in — must be token-exact against a serial admission with the same batch
/// composition everywhere else: both arms share the opening rounds and the
/// batch-2 rounds after admission, and differ only in the admission itself
/// (one mixed step versus a plain prefill plus one live decode round).
/// Synthetic ids suffice: the gate is a self-A/B over window arithmetic.
#[test]
#[ignore = "requires the pinned 12B checkpoint via PEGAINFER_TEST_MODEL_PATH and a GPU"]
fn mixed_step_crosses_the_window_like_serial() {
    let (ctx, serve, _dir) = stack_with(2048, 512);
    let partner: Vec<u32> = (0..40u32).map(|i| 1000 + i * 31).collect();
    let long_prompt: Vec<u32> = (0..1500u32).map(|i| 1000 + (i * 37) % 50000).collect();
    let budgets = [24usize, 20];

    let run_arm = |mixed: bool| -> Vec<Vec<u32>> {
        let mut arena = serve.alloc_step_arena(&ctx, 2, false).expect("step arena");
        let mut lanes: Vec<(usize, GemmaKv, u32)> = Vec::new();
        let mut produced: Vec<Vec<u32>> = vec![Vec::new(); 2];

        {
            let mut kv = serve.alloc_kv();
            admit_tokens(
                &serve.local_pool,
                &serve.global_pool,
                &mut kv,
                partner.len(),
            )
            .expect("admit partner");
            let logits = serve
                .step(&ctx, &mut kv, &partner, LogitsSpan::LastRow)
                .expect("prefill partner");
            let first = argmax_last(&ctx, &logits).expect("first token");
            produced[0].push(first);
            lanes.push((0, kv, first));
        }
        mixed_gate_decode_rounds(
            &ctx,
            &serve,
            &mut arena,
            &mut lanes,
            &mut produced,
            &budgets,
            3,
        );

        let mut kv = serve.alloc_kv();
        admit_tokens(
            &serve.local_pool,
            &serve.global_pool,
            &mut kv,
            long_prompt.len(),
        )
        .expect("admit long prompt");
        let first = if mixed {
            // The long prompt rides the live lane; its prefill crosses the
            // window inside the mixed step.
            for (_, lane_kv, _) in &mut lanes {
                admit_tokens(&serve.local_pool, &serve.global_pool, lane_kv, 1)
                    .expect("admit decode");
            }
            let tokens: Vec<u32> = lanes.iter().map(|(_, _, next)| *next).collect();
            let (vocab, host);
            {
                let mut kvs: Vec<&mut GemmaKv> = lanes.iter_mut().map(|(_, kv, _)| kv).collect();
                let mut prefills = [(&mut kv, long_prompt.as_slice())];
                let logits = serve
                    .mixed_prefill_decode_step(&ctx, &mut arena, &mut prefills, &mut kvs, &tokens)
                    .expect("mixed step");
                vocab = logits.hidden_dim;
                host = logits.to_host(&ctx).expect("logits D2H");
            }
            assert!(
                kv.local.origin_pages() > 0,
                "the mixed prefill must have released its window front (origin {})",
                kv.local.origin_pages()
            );
            for (row, (req, _, next)) in lanes.iter_mut().enumerate() {
                let token = mixed_gate_argmax(&host, row + 1, vocab);
                produced[*req].push(token);
                *next = token;
            }
            mixed_gate_argmax(&host, 0, vocab)
        } else {
            let logits = serve
                .step(&ctx, &mut kv, &long_prompt, LogitsSpan::LastRow)
                .expect("prefill long prompt");
            let first = argmax_last(&ctx, &logits).expect("first token");
            mixed_gate_decode_rounds(
                &ctx,
                &serve,
                &mut arena,
                &mut lanes,
                &mut produced,
                &budgets,
                1,
            );
            first
        };
        produced[1].push(first);
        lanes.push((1, kv, first));

        mixed_gate_decode_rounds(
            &ctx,
            &serve,
            &mut arena,
            &mut lanes,
            &mut produced,
            &budgets,
            usize::MAX,
        );
        produced
    };

    let serial = run_arm(false);
    let produced = run_arm(true);
    for (i, name) in ["partner", "newcomer"].iter().enumerate() {
        assert_eq!(
            produced[i], serial[i],
            "{name}: window-crossing mixed admission diverged from the serial path"
        );
        eprintln!(
            "{name}: {} tokens mixed crossing == serial",
            serial[i].len()
        );
    }
}

/// The KV ledger after a segmented admission must match the whole-prompt
/// admission: same frontier, same origin, same resident page count. Pure
/// bookkeeping arithmetic — no forward pass — so a divergence indicts the
/// advance/release cycling, not any kernel. Page ids may legally differ
/// (pool recycling), so the assertion stops at the ledger.
#[test]
#[ignore = "requires the pinned 12B checkpoint via PEGAINFER_TEST_MODEL_PATH and a GPU"]
fn segmented_admission_matches_whole_kv_ledger() {
    let (_ctx, serve, dir) = stack_with(2048, 512);
    let window = Gemma4Config::from_file(&dir)
        .expect("config")
        .sliding_window;
    let total = 1500usize;

    let mut whole = serve.alloc_kv();
    admit_tokens(&serve.local_pool, &serve.global_pool, &mut whole, total).expect("admit whole");
    whole
        .local
        .advance_and_release(total, window)
        .expect("whole advance");
    whole.global.advance(total);

    let mut seg = serve.alloc_kv();
    admit_tokens(&serve.local_pool, &serve.global_pool, &mut seg, total).expect("admit segmented");
    let mut left = total;
    while left > 0 {
        let step = left.min(128);
        seg.local
            .advance_and_release(step, window)
            .expect("segment advance");
        seg.global.advance(step);
        left -= step;
    }

    assert_eq!(seg.local.seq_len(), whole.local.seq_len(), "local frontier");
    assert_eq!(
        seg.global.seq_len(),
        whole.global.seq_len(),
        "global frontier"
    );
    assert_eq!(
        seg.local.origin_pages(),
        whole.local.origin_pages(),
        "released-front origin"
    );
    assert_eq!(
        seg.local.page_row().len(),
        whole.local.page_row().len(),
        "resident page count"
    );

    // Third arm: reserve per segment instead of up front. The ledger must
    // land in the same place, and the local residency must stay bounded by
    // window plus segment the whole way — the property that frees a long
    // prompt's admission from holding its full context in pages.
    let page = PAGE_SIZE;
    let residency_cap = window.div_ceil(page) + 128usize.div_ceil(page) + 1;
    let mut segadmit = serve.alloc_kv();
    let mut left = total;
    while left > 0 {
        let step = left.min(128);
        admit_tokens(&serve.local_pool, &serve.global_pool, &mut segadmit, step)
            .expect("admit segment");
        segadmit
            .local
            .advance_and_release(step, window)
            .expect("segment advance");
        segadmit.global.advance(step);
        left -= step;
        assert!(
            segadmit.local.page_row().len() <= residency_cap,
            "local residency {} exceeds window plus segment ({residency_cap})",
            segadmit.local.page_row().len()
        );
    }
    assert_eq!(
        segadmit.local.seq_len(),
        whole.local.seq_len(),
        "segment-admitted local frontier"
    );
    assert_eq!(
        segadmit.global.seq_len(),
        whole.global.seq_len(),
        "segment-admitted global frontier"
    );
    assert_eq!(
        segadmit.local.origin_pages(),
        whole.local.origin_pages(),
        "segment-admitted released-front origin"
    );
    assert_eq!(
        segadmit.local.page_row().len(),
        whole.local.page_row().len(),
        "segment-admitted resident page count"
    );
    eprintln!(
        "ledger: seq {} origin {} pages {} (segment-admitted residency capped at {residency_cap})",
        seg.local.seq_len(),
        seg.local.origin_pages(),
        seg.local.page_row().len()
    );
}

/// The overlap-safe prefill under a lane-stream override must be bit-equal
/// to the sync step: identical row-0 logits, the same released-window
/// shape after the deferred release, and greedy decode over the
/// lane-written KV matching the sync arm token for token — for a short
/// prompt and one crossing the sliding window.
#[test]
#[ignore = "requires the pinned 12B checkpoint via PEGAINFER_TEST_MODEL_PATH and a GPU"]
fn overlapped_prefill_matches_the_sync_step() {
    let (ctx, serve, _dir) = stack_with(2048, 512);
    let mut arena = serve.alloc_step_arena(&ctx, 1, false).expect("step arena");
    let budgets = [13usize];
    for (name, len) in [("short", 40usize), ("crossing", 1500)] {
        let prompt: Vec<u32> = (0..len as u32).map(|i| 1000 + (i * 37) % 50000).collect();

        let mut kv_sync = serve.alloc_kv();
        admit_tokens(
            &serve.local_pool,
            &serve.global_pool,
            &mut kv_sync,
            prompt.len(),
        )
        .expect("admit sync");
        let logits_sync = serve
            .step(&ctx, &mut kv_sync, &prompt, LogitsSpan::LastRow)
            .expect("sync prefill");
        let bits_sync: Vec<u32> = logits_sync
            .to_host(&ctx)
            .expect("sync logits D2H")
            .iter()
            .map(|v| v.to_bits())
            .collect();

        let mut kv_lane = serve.alloc_kv();
        admit_tokens(
            &serve.local_pool,
            &serve.global_pool,
            &mut kv_lane,
            prompt.len(),
        )
        .expect("admit lane");
        let lane = crate::green_ctx::PrefillLaneStream::shared().expect("lane stream");
        let pass = {
            let _guard =
                unsafe { pegainfer_core::tensor::StreamOverrideGuard::activate(lane.stream) };
            serve
                .prefill_into_logits(&ctx, &mut kv_lane, &prompt)
                .expect("lane prefill")
        };
        let sync = unsafe { cudarc::driver::sys::cuStreamSynchronize(lane.stream) };
        assert_eq!(
            sync,
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "{name}: lane stream drain"
        );
        serve
            .release_prefill_window(&mut kv_lane)
            .expect("deferred release");
        let bits_lane: Vec<u32> = pass
            .logits
            .to_host(&ctx)
            .expect("lane logits D2H")
            .iter()
            .map(|v| v.to_bits())
            .collect();

        assert_eq!(
            bits_lane, bits_sync,
            "{name}: overlapped prefill logits diverged from the sync step"
        );
        assert_eq!(
            (kv_lane.local.origin_pages(), kv_lane.local.seq_len()),
            (kv_sync.local.origin_pages(), kv_sync.local.seq_len()),
            "{name}: released-window shape diverged"
        );

        let mut tokens = [Vec::new(), Vec::new()];
        for (slot, kv) in [kv_sync, kv_lane].into_iter().enumerate() {
            let first = argmax_last(&ctx, &logits_sync).expect("first token");
            let mut lanes = vec![(0usize, kv, first)];
            let mut produced: Vec<Vec<u32>> = vec![vec![first]];
            mixed_gate_decode_rounds(
                &ctx,
                &serve,
                &mut arena,
                &mut lanes,
                &mut produced,
                &budgets,
                usize::MAX,
            );
            tokens[slot] = produced.swap_remove(0);
        }
        assert_eq!(
            tokens[1], tokens[0],
            "{name}: greedy decode over the lane-written KV diverged"
        );
        eprintln!(
            "{name}: {} prompt tokens, logits bit-equal, {} greedy tokens equal",
            prompt.len(),
            tokens[0].len()
        );
    }
}

/// The prefix cache's GPU halves, gated bit-level: two arms run the same
/// kernel sequence — prefill turn 1, then a suffix step and greedy decode —
/// and differ only in that the cache arm captures after turn 1, drops its
/// KV entirely (returning the original pages for reuse), and restores from
/// the cache-owned copies. Suffix logits must match to the bit and greedy
/// tokens exactly, for a short entry (origin 0) and one past the window
/// (released front); a divergence below the window floor must miss.
#[test]
#[ignore = "requires the pinned 12B checkpoint via PEGAINFER_TEST_MODEL_PATH and a GPU"]
fn prefix_restore_matches_cold_path() {
    use crate::prefix_cache::PrefixCache;
    let (ctx, serve, dir) = stack_with(4096, 512);
    let window = Gemma4Config::from_file(&dir)
        .expect("config")
        .sliding_window;
    let mut arena = serve.alloc_step_arena(&ctx, 1, false).expect("step arena");
    let budget = 16usize;

    for (name, turn1_len) in [("short", 200usize), ("long", 1500)] {
        let turn1: Vec<u32> = (0..turn1_len as u32)
            .map(|i| 1000 + (i * 37) % 50000)
            .collect();
        let mut turn2 = turn1.clone();
        turn2.extend((0..64u32).map(|i| 2000 + i * 13));

        let (ref_bits, ref_tokens) = {
            let mut kv = serve.alloc_kv();
            admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, turn1.len())
                .expect("admit t1");
            serve
                .step(&ctx, &mut kv, &turn1, LogitsSpan::LastRow)
                .expect("prefill t1");
            suffix_and_greedy(&serve, &ctx, &mut arena, &mut kv, &turn2, budget)
        };

        let (warm_bits, warm_tokens) = {
            let mut cache = PrefixCache::new(2, window);
            {
                let mut kv = serve.alloc_kv();
                admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, turn1.len())
                    .expect("admit t1");
                serve
                    .step(&ctx, &mut kv, &turn1, LogitsSpan::LastRow)
                    .expect("prefill t1");
                let entry = serve
                    .capture_checkpoint(&ctx, &kv, turn1.clone())
                    .expect("capture");
                cache.insert(entry, None);
            }
            // A divergence below the window floor must miss.
            if name == "long" {
                let mut early = turn2.clone();
                early[8] = 7;
                assert!(
                    cache.resolve(&early).is_none(),
                    "{name}: early divergence must not resolve"
                );
            }
            let (entry, t) = cache.resolve(&turn2).expect("hit");
            assert_eq!(t, turn1.len(), "{name}: resume point");
            let mut kv = serve
                .restore_from_checkpoint(&ctx, entry, t)
                .expect("restore");
            suffix_and_greedy(&serve, &ctx, &mut arena, &mut kv, &turn2, budget)
        };

        assert_eq!(
            warm_bits, ref_bits,
            "{name}: restored suffix logits diverged from the uncached arm"
        );
        assert_eq!(
            warm_tokens, ref_tokens,
            "{name}: restored greedy tokens diverged from the uncached arm"
        );
        eprintln!(
            "{name}: resume {} of {}, suffix logits bit-equal, {} greedy tokens equal",
            turn1.len(),
            turn2.len(),
            ref_tokens.len()
        );
    }

    // Past the per-entry allotment (half the serving context) the capture
    // must refuse — the cache can never hold more of the pool than its
    // entries paid for.
    let over: Vec<u32> = (0..2100u32).map(|i| 1000 + (i * 37) % 50000).collect();
    let mut kv = serve.alloc_kv();
    admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, over.len()).expect("admit over");
    serve
        .step(&ctx, &mut kv, &over, LogitsSpan::LastRow)
        .expect("prefill over");
    assert!(
        serve.capture_checkpoint(&ctx, &kv, over.clone()).is_none(),
        "a prompt past the per-entry allotment must not capture"
    );
    eprintln!(
        "over: {} tokens refused capture at the allotment",
        over.len()
    );
}

/// Admit and prefill `prompt`'s unseen suffix, then decode `budget` greedy
/// tokens on the serving path; returns the suffix logits row's bits and the
/// tokens.
fn suffix_and_greedy(
    serve: &GemmaServe,
    ctx: &DeviceContext,
    arena: &mut StepArena,
    kv: &mut GemmaKv,
    prompt: &[u32],
    budget: usize,
) -> (Vec<u32>, Vec<u32>) {
    let start = kv.local.seq_len();
    admit_tokens(
        &serve.local_pool,
        &serve.global_pool,
        kv,
        prompt.len() - start,
    )
    .expect("admit suffix");
    let logits = serve
        .step(ctx, kv, &prompt[start..], LogitsSpan::LastRow)
        .expect("suffix prefill");
    let host = logits.to_host(ctx).expect("D2H");
    let vocab = logits.hidden_dim;
    let last = &host[(logits.seq_len - 1) * vocab..logits.seq_len * vocab];
    let bits: Vec<u32> = last.iter().map(|v| v.to_bits()).collect();
    let mut next = u32::try_from(argmax(last)).expect("token id");
    let mut tokens = vec![next];
    for _ in 1..budget {
        let row = decode_serving(serve, ctx, arena, kv, next).expect("decode");
        next = u32::try_from(argmax(&row)).expect("token id");
        tokens.push(next);
    }
    (bits, tokens)
}

/// One serving-path decode step for a single request: the batched decode
/// entry at batch one on an eager arena. The graph path is anchored to this
/// by the ragged determinism gate's replay-vs-eager comparison.
fn decode_serving(
    serve: &GemmaServe,
    ctx: &DeviceContext,
    arena: &mut StepArena,
    kv: &mut GemmaKv,
    token: u32,
) -> Result<Vec<f32>> {
    admit_tokens(&serve.local_pool, &serve.global_pool, kv, 1)?;
    let mut borrowed: [&mut GemmaKv; 1] = [kv];
    let logits = serve.decode_batch_step(ctx, arena, &mut borrowed, &[token])?;
    logits.to_host(ctx)
}

fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .expect("non-empty row")
}

/// Greedy continuation: prefill the prompt, then decode `max_new`
/// tokens one at a time. Host argmax over the last position — the
/// correctness path; sampling belongs to the serving frontend.
pub(crate) fn generate_greedy(
    serve: &GemmaServe,
    ctx: &DeviceContext,
    kv: &mut GemmaKv,
    prompt: &[u32],
    max_new: usize,
) -> Result<Vec<u32>> {
    anyhow::ensure!(!prompt.is_empty(), "empty prompt");
    anyhow::ensure!(max_new > 0, "generate_greedy needs max_new >= 1");
    admit_tokens(&serve.local_pool, &serve.global_pool, kv, prompt.len())?;
    let mut arena = serve.alloc_step_arena(ctx, 1, false)?;
    let logits = serve.step(ctx, kv, prompt, LogitsSpan::LastRow)?;
    let mut next = argmax_last(ctx, &logits)?;
    let mut out = vec![next];
    for _ in 1..max_new {
        let row = decode_serving(serve, ctx, &mut arena, kv, next)?;
        next = u32::try_from(argmax(&row)).context("token id")?;
        out.push(next);
    }
    Ok(out)
}

fn argmax_last(ctx: &DeviceContext, logits: &HiddenStates) -> Result<u32> {
    let host = logits.to_host(ctx)?;
    let vocab = logits.hidden_dim;
    let row = &host[(logits.seq_len - 1) * vocab..logits.seq_len * vocab];
    anyhow::ensure!(
        row.iter().all(|v| v.is_finite()),
        "non-finite logit in the row an argmax is about to rank"
    );
    let argmax = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .context("non-empty vocab")?;
    u32::try_from(argmax).context("token id fits u32")
}

/// A fixed-width ragged batch must preserve each request's logits as rows move
/// between steps. Distinct token streams make cross-row reads observable.
#[test]
#[ignore = "requires the pinned 12B checkpoint and a GPU"]
fn a_ragged_batch_does_not_depend_on_row_order() {
    const STEPS: usize = 10;
    // Three real requests in a four-row arena: the fourth row is the pad
    // row, writing the pools' reserved padding pages.
    let lengths = [1100usize, 40, 17];
    // Exactly the pages the three requests reach by the last step, plus the
    // pool's padding page.
    let pages = lengths
        .iter()
        .map(|len| (len + STEPS).div_ceil(PAGE_SIZE))
        .sum::<usize>()
        + 1;
    let (ctx, serve, _dir) = stack_with(2048, pages);

    let prompt_of = |request: usize| -> Vec<u32> {
        (0..lengths[request] as u32)
            .map(|i| 1000 + request as u32 * 2000 + i)
            .collect()
    };
    let feed_of = |request: usize, step: usize| -> u32 { 20000 + (request * STEPS + step) as u32 };

    // `orders[step % orders.len()]` lists request ids by the row each one
    // occupies at that step; out[request][step] collects results back by
    // request, wherever it sat.
    let run = |orders: &[Vec<usize>], graphs: bool| -> Vec<Vec<Vec<f32>>> {
        let mut arena = serve.alloc_step_arena(&ctx, 4, graphs).expect("step arena");
        if graphs {
            serve
                .precapture_decode_graphs(&ctx, &mut arena)
                .expect("precapture");
        }
        let mut kvs: Vec<GemmaKv> = lengths.iter().map(|_| serve.alloc_kv()).collect();
        for (request, kv) in kvs.iter_mut().enumerate() {
            let prompt = prompt_of(request);
            admit_tokens(&serve.local_pool, &serve.global_pool, kv, prompt.len())
                .expect("admit prompt");
            serve
                .step(&ctx, kv, &prompt, LogitsSpan::LastRow)
                .expect("prefill");
        }
        assert!(
            kvs[0].local.origin_pages() > 0,
            "the {} token row should have released its window front",
            lengths[0]
        );
        for (request, kv) in kvs.iter().enumerate().skip(1) {
            assert_eq!(
                kv.local.origin_pages(),
                0,
                "request {request} ({} prompt tokens) should still hold its front",
                lengths[request]
            );
        }
        let mut out = vec![Vec::with_capacity(STEPS); lengths.len()];
        for step in 0..STEPS {
            let order = &orders[step % orders.len()];
            for kv in &mut kvs {
                admit_tokens(&serve.local_pool, &serve.global_pool, kv, 1).expect("admit token");
            }
            let mut slots: Vec<Option<&mut GemmaKv>> = kvs.iter_mut().map(Some).collect();
            let mut borrowed = Vec::with_capacity(order.len());
            let mut tokens = Vec::with_capacity(order.len());
            for &request in order {
                borrowed.push(slots[request].take().expect("each request once"));
                tokens.push(feed_of(request, step));
            }
            let logits = serve
                .decode_batch_step(&ctx, &mut arena, &mut borrowed, &tokens)
                .expect("decode");
            let host = logits.to_host(&ctx).expect("D2H");
            let vocab = logits.hidden_dim;
            for (row, &request) in order.iter().enumerate() {
                out[request].push(host[row * vocab..(row + 1) * vocab].to_vec());
            }
        }
        out
    };

    let forward: Vec<usize> = (0..lengths.len()).collect();
    let reversed: Vec<usize> = forward.iter().rev().copied().collect();
    let first = run(std::slice::from_ref(&forward), true);
    let replayed = run(std::slice::from_ref(&forward), true);
    let shuffled = run(&[forward.clone(), reversed], true);
    let eager = run(std::slice::from_ref(&forward), false);

    for (label, other) in [
        ("replaying", &replayed),
        ("moving its row between steps", &shuffled),
        ("running eagerly instead of replaying the graph", &eager),
    ] {
        for (request, (rows_a, rows_b)) in first.iter().zip(other).enumerate() {
            for (step, (x, y)) in rows_a.iter().zip(rows_b).enumerate() {
                let differing = x
                    .iter()
                    .zip(y)
                    .enumerate()
                    .find(|(_, (p, q))| p.to_bits() != q.to_bits());
                assert!(
                    differing.is_none(),
                    "request {request} ({} prompt tokens) step {step}: {label} changed its \
                     logits at {:?}",
                    lengths[request],
                    differing.map(|(i, (p, q))| (i, *p, *q))
                );
            }
        }
    }
    println!(
        "ragged batch: {} rows at {:?} tokens replay identically and survive per step row moves",
        lengths.len(),
        lengths
    );
}
