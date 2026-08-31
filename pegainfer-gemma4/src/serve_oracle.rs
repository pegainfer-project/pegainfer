//! GPU + checkpoint gates for the KV serving path, every one of them the
//! production path against an external reference or against itself under a
//! different admission shape.

use anyhow::Result;

use super::*;
use crate::kv::admit_tokens;
use crate::testkit::f32_tensor;
use crate::testkit::fixture_manifest;
use crate::testkit::golden_bytes;
use crate::testkit::i32_tensor;
use crate::testkit::log_softmax_at;
use crate::testkit::model_path;
use crate::testkit::u32_tensor;

fn stack_with(max_context: usize, pages: usize) -> (DeviceContext, GemmaServe, String) {
    let dir = model_path();
    let config = Gemma4Config::from_file(&dir).expect("config");
    let (weights, _) =
        Gemma4Weights::from_safetensors(&dir, 0, config).expect("load checkpoint weights");
    let ctx = DeviceContext::new_with_device(0).expect("device context");
    let serve = GemmaServe::new(&ctx, weights, max_context, pages, pages).expect("serve");
    (ctx, serve, dir)
}

fn load_stack() -> (DeviceContext, GemmaServe, String) {
    // One request at the window, plus each pool's padding page.
    stack_with(1024, 66)
}

fn synthetic_tokens(len: usize, salt: u32) -> Vec<u32> {
    (0..len as u32)
        .map(|i| 1000 + (i * 37 + salt) % 50000)
        .collect()
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
    let (_, prompt) = u32_tensor(fixture, &format!("{case}_prompt"));
    let (_, teacher_i32) = i32_tensor(fixture, &format!("{case}_teacher"));
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
        let logits = serve.step(ctx, &mut kv, piece).expect("prefill");
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

#[derive(Clone, Copy)]
struct BorrowedFloor {
    tolerance: f32,
    top1_share: f64,
}

#[derive(Clone, Copy)]
struct Waypoint<'a> {
    case: &'a str,
    chunk: usize,
    floor: Option<BorrowedFloor>,
}

fn waypoint_reference(
    fixture: &safetensors::SafeTensors<'_>,
    point: Waypoint<'_>,
) -> (Vec<i32>, Vec<f32>, usize, usize, f32, usize) {
    match point.floor {
        Some(floor) => reference_sdpa_only(fixture, point.case, floor.tolerance, floor.top1_share),
        None => reference(fixture, point.case),
    }
}

fn gate_waypoint(
    ctx: &DeviceContext,
    serve: &GemmaServe,
    fixture: &safetensors::SafeTensors<'_>,
    point: Waypoint<'_>,
) -> Option<String> {
    let label = match point.chunk {
        0 => point.case.to_string(),
        _ => format!("{}-chunked", point.case),
    };
    let (ids, lps, positions, top_k, tolerance, backend_top1) = waypoint_reference(fixture, point);
    let run = run_case(ctx, serve, fixture, point.case, point.chunk);
    assert_eq!(run.rows.len(), positions, "{label}: fixture positions");
    assert_eq!(
        point.chunk > 0,
        run.shifted_multi_token,
        "{label}: shifted multi-token coverage"
    );
    let (max_abs, top1) = score_rows(&run.rows, &ids, &lps, top_k, &label);
    assert!(
        top1 >= backend_top1,
        "{label}: top-1 {top1}/{positions} below backend bar {backend_top1}/{positions}"
    );
    let page = serve.local_pool.layout().page_size;
    let released = run.kv_len.saturating_sub(serve.sliding_window) / page;
    assert_eq!(run.local_pages, run.kv_len.div_ceil(page) - released);
    assert_eq!(run.global_pages, run.kv_len.div_ceil(page));
    eprintln!(
        "{label}: max |dlogprob| {max_abs} (tol {tolerance:.2}), top-1 \
         {top1}/{positions}, local pages {}, global {}",
        run.local_pages, run.global_pages
    );
    (max_abs > tolerance).then(|| format!("{label} ({max_abs} > {tolerance})"))
}

fn gate_waypoints(
    ctx: &DeviceContext,
    serve: &GemmaServe,
    fixture: &safetensors::SafeTensors<'_>,
    points: &[Waypoint<'_>],
) {
    let over: Vec<String> = points
        .iter()
        .filter_map(|&point| gate_waypoint(ctx, serve, fixture, point))
        .collect();
    assert!(
        over.is_empty(),
        "cases over their calibrated floor: {over:?}"
    );
}

fn validate_waypoint_provenance(dir: &str, window_bytes: &[u8], long_bytes: &[u8]) {
    let (_, golden) = golden_bytes(dir);
    let window = fixture_manifest(window_bytes, "gemma4_window_golden");
    let long = fixture_manifest(long_bytes, "gemma4_longctx_golden");
    assert_eq!(window["revision"], golden["revision"], "window revision");
    assert_eq!(long["revision"], golden["revision"], "longctx revision");
    assert_eq!(
        long["transformers"], window["transformers"],
        "borrowed floor reference release"
    );
    let skipped = long["eager_skipped"].as_array().expect("eager_skipped");
    for case in ["w16384", "w32768"] {
        assert!(
            skipped.iter().any(|value| value == case),
            "{case} eager skip"
        );
    }
}

/// Window crossing and raised-ceiling waypoints share one 12B tower load.
/// Dual-backend window cases carry their own floor; long-context sdpa cases
/// borrow the deepest window floor under the same reference release.
#[test]
#[ignore = "requires the pinned 12B checkpoint, fixtures, and a GPU"]
fn context_waypoints_match_hf() {
    let (ctx, serve, dir) = stack_with(32900, 2200);
    let window_bytes = std::fs::read(WINDOW_FIXTURE).expect("read window fixture");
    let long_bytes = std::fs::read(LONGCTX_FIXTURE).expect("read longctx fixture");
    validate_waypoint_provenance(&dir, &window_bytes, &long_bytes);
    let window = safetensors::SafeTensors::deserialize(&window_bytes).expect("window fixture");
    let long = safetensors::SafeTensors::deserialize(&long_bytes).expect("longctx fixture");
    let (_, _, positions, _, tolerance, top1) = reference(&window, "w4096");
    #[allow(clippy::cast_precision_loss)]
    let floor = BorrowedFloor {
        tolerance,
        top1_share: top1 as f64 / positions as f64,
    };
    let window_points = [
        Waypoint {
            case: "w1023",
            chunk: 0,
            floor: None,
        },
        Waypoint {
            case: "w1024",
            chunk: 0,
            floor: None,
        },
        Waypoint {
            case: "w1025",
            chunk: 0,
            floor: None,
        },
        Waypoint {
            case: "w4096",
            chunk: 0,
            floor: None,
        },
        Waypoint {
            case: "w4096",
            chunk: 1024,
            floor: None,
        },
    ];
    let long_points = [
        Waypoint {
            case: "w16384",
            chunk: 0,
            floor: Some(floor),
        },
        Waypoint {
            case: "w32768",
            chunk: 0,
            floor: Some(floor),
        },
        Waypoint {
            case: "w32768",
            chunk: 2048,
            floor: Some(floor),
        },
    ];
    gate_waypoints(&ctx, &serve, &window, &window_points);
    gate_waypoints(&ctx, &serve, &long, &long_points);
}

/// `window_left` masks out-of-window keys whether or not their pages are
/// still resident, so releasing them need not change a single generated
/// token.
#[test]
#[ignore = "requires the pinned 12B checkpoint and a GPU"]
fn eviction_is_footprint_only() {
    let (ctx, mut serve, _dir) = stack_with(1300, 120);
    let prompt = synthetic_tokens(1023, 5);

    let run = |serve: &GemmaServe| -> (Vec<u32>, usize) {
        let mut kv = serve.alloc_kv();
        let mut arena = serve
            .alloc_step_arena(&ctx, 1, false)
            .expect("oracle step arena");
        admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, prompt.len())
            .expect("admit prompt");
        let logits = serve.step(&ctx, &mut kv, &prompt).expect("prefill");
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

/// DoD gate: greedy continuation matches HF `generate()` token for
/// token on three prompts. The fixture is dumped on the box by
/// tools/accuracy/dump_gemma4_generate.py (prompt + up to 50 greedy
/// tokens per case).
#[test]
#[ignore = "requires the pinned 12B checkpoint, fixtures, and a GPU"]
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
    let (_, golden) = golden_bytes(&dir);
    let generate = fixture_manifest(&bytes, "gemma4_generate");
    assert_eq!(
        generate["revision"], golden["revision"],
        "the generate fixture was dumped from a different revision than the golden one"
    );
    let fixture = safetensors::SafeTensors::deserialize(&bytes).expect("parse fixture");
    let mut diverged: Vec<String> = Vec::new();
    for case in ["a", "b", "c"] {
        let (_, prompt) = u32_tensor(&fixture, &format!("{case}_prompt"));
        let (_, expect_i32) = i32_tensor(&fixture, &format!("{case}_generated"));
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

/// Reserve a lane's whole prompt. These gates drive the serving primitives
/// directly, so a pool that cannot hold it fails here rather than in a step.
fn gate_admit_kv(serve: &GemmaServe, prompt: &[u32], what: &str) -> GemmaKv {
    let mut kv = serve.alloc_kv();
    admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, prompt.len())
        .unwrap_or_else(|err| panic!("admit {what}: {err:#}"));
    kv
}

fn gate_open_lane(
    ctx: &DeviceContext,
    serve: &GemmaServe,
    prompt: &[u32],
    what: &str,
) -> (GemmaKv, u32) {
    let mut kv = gate_admit_kv(serve, prompt, what);
    let logits = serve
        .step(ctx, &mut kv, prompt)
        .unwrap_or_else(|err| panic!("prefill {what}: {err:#}"));
    let first = argmax_last(ctx, &logits).expect("first token");
    (kv, first)
}

fn gate_step_tokens(serve: &GemmaServe, lanes: &mut [(usize, GemmaKv, u32)]) -> Vec<u32> {
    for (_, kv, _) in lanes.iter_mut() {
        admit_tokens(&serve.local_pool, &serve.global_pool, kv, 1).expect("admit decode");
    }
    lanes.iter().map(|(_, _, next)| *next).collect()
}

fn gate_host_logits(ctx: &DeviceContext, logits: &HiddenStates) -> (usize, Vec<f32>) {
    (logits.hidden_dim, logits.to_host(ctx).expect("logits D2H"))
}

/// Read a step's incumbent rows — `row_base` past any newcomer rows — into the
/// live lanes, so a mixed round and a pure one advance the batch alike.
fn settle_gate_lanes(
    host: &[f32],
    vocab: usize,
    row_base: usize,
    lanes: &mut Vec<(usize, GemmaKv, u32)>,
    produced: &mut [Vec<u32>],
    budgets: &[usize],
) {
    let mut retire: Vec<usize> = Vec::new();
    for (row, (req, _, next)) in lanes.iter_mut().enumerate() {
        let token = mixed_gate_argmax(host, row + row_base, vocab);
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
        let tokens = gate_step_tokens(serve, lanes);
        let (vocab, host) = {
            let mut kvs: Vec<&mut GemmaKv> = lanes.iter_mut().map(|(_, kv, _)| kv).collect();
            let logits = serve
                .decode_batch_step(ctx, arena, &mut kvs, &tokens)
                .expect("batched decode");
            gate_host_logits(ctx, logits)
        };
        settle_gate_lanes(&host, vocab, 0, lanes, produced, budgets);
    }
}

/// A mixed admission — the prompt sharing one step with the live decode
/// batch — must be token-exact against the serial path, for the newcomer
/// (logits row 0) and every incumbent lane (rows 1..), across two
/// admissions at different batch sizes and the pure-decode rounds between
/// them.
fn assert_mixed_admissions_match_serial(ctx: &DeviceContext, serve: &GemmaServe) {
    let cases = ["a", "b", "c"];
    let prompts = crate::testkit::generate_fixture_prompts();
    let budgets = [50usize, 37, 44];

    let serial: Vec<Vec<u32>> = prompts
        .iter()
        .zip(budgets)
        .map(|(prompt, budget)| {
            let mut kv = serve.alloc_kv();
            generate_greedy(serve, ctx, &mut kv, prompt, budget).expect("serial greedy")
        })
        .collect();

    let mut arena = serve.alloc_step_arena(ctx, 4, false).expect("step arena");
    let mut lanes: Vec<(usize, GemmaKv, u32)> = Vec::new();
    let mut produced: Vec<Vec<u32>> = vec![Vec::new(); cases.len()];

    // Lane a arrives alone — the plain prefill path — then three decode
    // rounds so the mixed admissions below meet a warm batch.
    let (kv_a, first_a) = gate_open_lane(ctx, serve, &prompts[0], "prompt a");
    produced[0].push(first_a);
    lanes.push((0, kv_a, first_a));
    mixed_gate_decode_rounds(
        ctx,
        serve,
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
        let mut kv_b = gate_admit_kv(serve, &prompts[1], "prompt b");
        let mut kv_c = gate_admit_kv(serve, &prompts[2], "prompt c");
        let tokens = gate_step_tokens(serve, &mut lanes);
        let (vocab, host) = {
            let mut kvs: Vec<&mut GemmaKv> = lanes.iter_mut().map(|(_, kv, _)| kv).collect();
            let mut prefills = [
                (&mut kv_b, prompts[1].as_slice()),
                (&mut kv_c, prompts[2].as_slice()),
            ];
            let logits = serve
                .mixed_prefill_decode_step(ctx, &mut arena, &mut prefills, &mut kvs, &tokens)
                .expect("k=2 mixed step");
            gate_host_logits(ctx, logits)
        };
        settle_gate_lanes(&host, vocab, 2, &mut lanes, &mut produced, &budgets);
        let first_b = mixed_gate_argmax(&host, 0, vocab);
        produced[1].push(first_b);
        lanes.push((1, kv_b, first_b));
        let first_c = mixed_gate_argmax(&host, 1, vocab);
        produced[2].push(first_c);
        lanes.push((2, kv_c, first_c));
        mixed_gate_decode_rounds(
            ctx,
            serve,
            &mut arena,
            &mut lanes,
            &mut produced,
            &budgets,
            3,
        );
    }

    // Drain everyone to their budgets.
    mixed_gate_decode_rounds(
        ctx,
        serve,
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
fn assert_mixed_window_crossing_matches_serial(ctx: &DeviceContext, serve: &GemmaServe) {
    let partner: Vec<u32> = (0..40u32).map(|i| 1000 + i * 31).collect();
    let long_prompt: Vec<u32> = (0..1500u32).map(|i| 1000 + (i * 37) % 50000).collect();
    let budgets = [24usize, 20];

    let run_arm = |mixed: bool| -> Vec<Vec<u32>> {
        let mut arena = serve.alloc_step_arena(ctx, 2, false).expect("step arena");
        let mut lanes: Vec<(usize, GemmaKv, u32)> = Vec::new();
        let mut produced: Vec<Vec<u32>> = vec![Vec::new(); 2];

        let (kv_partner, first_partner) = gate_open_lane(ctx, serve, &partner, "partner");
        produced[0].push(first_partner);
        lanes.push((0, kv_partner, first_partner));
        mixed_gate_decode_rounds(
            ctx,
            serve,
            &mut arena,
            &mut lanes,
            &mut produced,
            &budgets,
            3,
        );

        let mut kv = gate_admit_kv(serve, &long_prompt, "long prompt");
        let first = if mixed {
            // The long prompt rides the live lane; its prefill crosses the
            // window inside the mixed step.
            let tokens = gate_step_tokens(serve, &mut lanes);
            let (vocab, host) = {
                let mut kvs: Vec<&mut GemmaKv> = lanes.iter_mut().map(|(_, kv, _)| kv).collect();
                let mut prefills = [(&mut kv, long_prompt.as_slice())];
                let logits = serve
                    .mixed_prefill_decode_step(ctx, &mut arena, &mut prefills, &mut kvs, &tokens)
                    .expect("mixed step");
                gate_host_logits(ctx, logits)
            };
            assert!(
                kv.local.origin_pages() > 0,
                "the mixed prefill must have released its window front (origin {})",
                kv.local.origin_pages()
            );
            settle_gate_lanes(&host, vocab, 1, &mut lanes, &mut produced, &budgets);
            mixed_gate_argmax(&host, 0, vocab)
        } else {
            let logits = serve
                .step(ctx, &mut kv, &long_prompt)
                .expect("prefill long prompt");
            let first = argmax_last(ctx, &logits).expect("first token");
            mixed_gate_decode_rounds(
                ctx,
                serve,
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
            ctx,
            serve,
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

#[test]
#[ignore = "requires the pinned 12B checkpoint and a GPU"]
fn mixed_step_matches_serial() {
    let (ctx, serve, _dir) = stack_with(2048, 512);
    assert_mixed_admissions_match_serial(&ctx, &serve);
    assert_mixed_window_crossing_matches_serial(&ctx, &serve);
}

/// The overlap-safe prefill under a lane-stream override must be bit-equal
/// to the sync step: identical row-0 logits, the same released-window
/// shape after the deferred release, and greedy decode over the
/// lane-written KV matching the sync arm token for token — for a short
/// prompt and one crossing the sliding window.
#[test]
#[ignore = "requires a Gemma 4 checkpoint and a GPU"]
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
            .step(&ctx, &mut kv_sync, &prompt)
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
#[ignore = "requires the pinned 12B checkpoint and a GPU"]
fn prefix_restore_matches_cold_path() {
    use crate::prefix_cache::PrefixCache;
    let (ctx, serve, _dir) = stack_with(4096, 512);
    let window = serve.weights.config.sliding_window;
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
            serve.step(&ctx, &mut kv, &turn1).expect("prefill t1");
            suffix_and_greedy(&serve, &ctx, &mut arena, &mut kv, &turn2, budget)
        };

        let (warm_bits, warm_tokens) = {
            let mut cache = PrefixCache::new(2, window);
            {
                let mut kv = serve.alloc_kv();
                admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, turn1.len())
                    .expect("admit t1");
                serve.step(&ctx, &mut kv, &turn1).expect("prefill t1");
                let entry = serve
                    .capture_checkpoint(&ctx, &kv, &turn1)
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
    serve.step(&ctx, &mut kv, &over).expect("prefill over");
    assert!(
        serve.capture_checkpoint(&ctx, &kv, &over).is_none(),
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
        .step(ctx, kv, &prompt[start..])
        .expect("suffix prefill");
    let host = logits.to_host(ctx).expect("D2H");
    let vocab = logits.hidden_dim;
    let last = &host[(logits.seq_len - 1) * vocab..logits.seq_len * vocab];
    let bits: Vec<u32> = last.iter().map(|v| v.to_bits()).collect();
    let first = u32::try_from(argmax(last)).expect("token id");
    let tokens = greedy_continuation(serve, ctx, arena, kv, first, budget).expect("greedy suffix");
    (bits, tokens)
}

/// Decode `budget` greedy tokens, counting the `first` a prefill already picked.
fn greedy_continuation(
    serve: &GemmaServe,
    ctx: &DeviceContext,
    arena: &mut StepArena,
    kv: &mut GemmaKv,
    first: u32,
    budget: usize,
) -> Result<Vec<u32>> {
    let mut next = first;
    let mut out = vec![next];
    for _ in 1..budget {
        let row = decode_serving(serve, ctx, arena, kv, next)?;
        next = u32::try_from(argmax(&row)).context("token id")?;
        out.push(next);
    }
    Ok(out)
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
    let logits = serve.step(ctx, kv, prompt)?;
    let first = argmax_last(ctx, &logits)?;
    greedy_continuation(serve, ctx, &mut arena, kv, first, max_new)
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
#[ignore = "requires a Gemma 4 checkpoint and a GPU"]
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
            serve.step(&ctx, kv, &prompt).expect("prefill");
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
