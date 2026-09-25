//! GPU + checkpoint gates for the KV serving path, every one of them the
//! production path against an external reference or against itself under a
//! different admission shape.

use anyhow::Result;
use pegainfer_core::kv_pool::KvFormat;
use pegainfer_core::kv_pool::KvPool;
use pegainfer_core::kv_pool::KvStorage;

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
    stack_with_storage(
        max_context,
        pages,
        crate::engine::local_kv_storage(&crate::engine::read_env).expect("PEGAINFER_KV_FP8"),
    )
}

fn stack_with_storage(
    max_context: usize,
    pages: usize,
    storage: KvStorage,
) -> (DeviceContext, GemmaServe, String) {
    let dir = model_path();
    let config = Gemma4Config::from_file(&dir).expect("config");
    let weights =
        Gemma4Weights::from_safetensors(&dir, 0, config).expect("load checkpoint weights");
    let ctx = DeviceContext::new_with_device(0).expect("device context");
    // The oracle measures the incumbent kernel; the opt-in one has its own
    // gate.
    let serve = GemmaServe::new(
        &ctx,
        weights,
        max_context,
        storage,
        pages,
        pages,
        GlobalAttn::Incumbent,
    )
    .expect("serve");
    eprintln!("oracle stack storage: {storage:?}");
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

fn window_fixture() -> String {
    crate::testkit::fixture_path(
        "PEGAINFER_GEMMA4_WINDOW_GOLDEN",
        "gemma4-12b-hf-window-golden.safetensors",
    )
}

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

fn longctx_fixture() -> String {
    crate::testkit::fixture_path(
        "PEGAINFER_GEMMA4_LONGCTX_GOLDEN",
        "gemma4-12b-hf-longctx-golden.safetensors",
    )
}

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

fn waypoint_label(point: Waypoint<'_>) -> String {
    match point.chunk {
        0 => point.case.to_string(),
        _ => format!("{}-chunked", point.case),
    }
}

/// What one waypoint left behind: its failures, or nothing because the card
/// could not hold the pass.
enum WaypointOutcome {
    Ran(Vec<String>),
    Skipped,
}

#[derive(Default)]
struct WaypointReport {
    ran: Vec<String>,
    failures: Vec<String>,
    skipped: Vec<String>,
}

fn gate_waypoint(
    ctx: &DeviceContext,
    serve: &GemmaServe,
    fixture: &safetensors::SafeTensors<'_>,
    point: Waypoint<'_>,
) -> WaypointOutcome {
    let label = waypoint_label(point);
    // Chunk 0 is the whole-prompt pass.
    if point.chunk == 0 {
        let (_, prompt) = u32_tensor(fixture, &format!("{}_prompt", point.case));
        // Before the probe: afterwards the allocator still holds what it
        // touched, so the same call would report the probe's own state.
        let (free, total) = cudarc::driver::result::mem_get_info().expect("cuMemGetInfo");
        let fits = serve
            .single_pass_scratch_fits(ctx, prompt.len())
            .unwrap_or_else(|e| {
                panic!("{label}: the scratch probe failed for its own reason: {e:#}")
            });
        if !fits {
            let gib = |bytes: usize| bytes as f64 / (1u64 << 30) as f64;
            eprintln!(
                "{label}: skipped -- a whole-prompt pass over {} rows cannot take its scratch \
                 with {:.1} of {:.1} GiB free; the chunked pass carries this length",
                prompt.len(),
                gib(free),
                gib(total),
            );
            return WaypointOutcome::Skipped;
        }
    }
    let (ids, lps, positions, top_k, tolerance, backend_top1) = waypoint_reference(fixture, point);
    let run = run_case(ctx, serve, fixture, point.case, point.chunk);
    assert_eq!(run.rows.len(), positions, "{label}: fixture positions");
    assert_eq!(
        point.chunk > 0,
        run.shifted_multi_token,
        "{label}: shifted multi-token coverage"
    );
    let (max_abs, top1) = score_rows(&run.rows, &ids, &lps, top_k, &label);
    let page = serve.local_pool.layout().page_size;
    let released = run.kv_len.saturating_sub(serve.sliding_window) / page;
    assert_eq!(run.local_pages, run.kv_len.div_ceil(page) - released);
    assert_eq!(run.global_pages, run.kv_len.div_ceil(page));
    eprintln!(
        "{label}: max |dlogprob| {max_abs} (tol {tolerance:.2}), top-1 \
         {top1}/{positions}, local pages {}, global {}",
        run.local_pages, run.global_pages
    );
    let mut failures = Vec::new();
    if top1 < backend_top1 {
        failures.push(format!(
            "{label}: top-1 {top1}/{positions} below backend bar {backend_top1}/{positions}"
        ));
    }
    if max_abs > tolerance {
        failures.push(format!("{label} ({max_abs} > {tolerance})"));
    }
    WaypointOutcome::Ran(failures)
}

fn gate_waypoints(
    ctx: &DeviceContext,
    serve: &GemmaServe,
    fixture: &safetensors::SafeTensors<'_>,
    points: &[Waypoint<'_>],
    report: &mut WaypointReport,
) {
    for &point in points {
        let label = waypoint_label(point);
        match gate_waypoint(ctx, serve, fixture, point) {
            WaypointOutcome::Ran(failures) => {
                report.ran.push(label);
                report.failures.extend(failures);
            }
            WaypointOutcome::Skipped => report.skipped.push(label),
        }
    }
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
    // The deepest waypoint's prompt and its teacher tokens, whole in the
    // pools: the pages a raised ceiling would hold, plus each pool's padding
    // page and one more.
    let max_context = 32900;
    let (ctx, serve, dir) = stack_with(
        max_context,
        max_context.div_ceil(crate::kv::LOCAL_PAGE_SIZE) + 2,
    );
    let window_bytes = std::fs::read(window_fixture()).expect("read window fixture");
    let long_bytes = std::fs::read(longctx_fixture()).expect("read longctx fixture");
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
    // The chunked 32K pass runs before the whole-prompt one: the chunked
    // walk is what a raised ceiling serves through, and it is the pass a
    // card too small for the whole-prompt scratch still has to carry.
    let long_points = [
        Waypoint {
            case: "w16384",
            chunk: 0,
            floor: Some(floor),
        },
        Waypoint {
            case: "w32768",
            chunk: 2048,
            floor: Some(floor),
        },
        Waypoint {
            case: "w32768",
            chunk: 0,
            floor: Some(floor),
        },
    ];
    let mut report = WaypointReport::default();
    gate_waypoints(&ctx, &serve, &window, &window_points, &mut report);
    gate_waypoints(&ctx, &serve, &long, &long_points, &mut report);
    // A skipped whole-prompt case is only evidence of memory, never of
    // numerics: its length has to have passed through the chunked pass.
    for skipped in &report.skipped {
        let twin = format!("{skipped}-chunked");
        assert!(
            report.ran.contains(&twin) && !report.failures.iter().any(|f| f.starts_with(&twin)),
            "{skipped} was skipped for memory and {twin} did not pass in its place"
        );
    }
    assert!(
        report.failures.is_empty(),
        "cases over their calibrated floor: {:?}",
        report.failures
    );
    eprintln!(
        "waypoints: {} ran, {} skipped for memory {:?}",
        report.ran.len(),
        report.skipped.len(),
        report.skipped
    );
}

fn incremental_argmaxes(ctx: &DeviceContext, serve: &GemmaServe, prompt: &[u32]) -> Vec<usize> {
    let mut kv = serve.alloc_kv();
    let mut arena = serve
        .alloc_step_arena(ctx, 1, false)
        .expect("oracle step arena");
    admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, 1).expect("admit first token");
    let first = serve.step(ctx, &mut kv, &prompt[..1]).expect("first step");
    let mut choices = vec![argmax(&first.to_host(ctx).expect("first logits D2H"))];
    for &token in &prompt[1..] {
        let row = decode_serving(serve, ctx, &mut arena, &mut kv, token).expect("decode");
        choices.push(argmax(&row));
    }
    choices
}

fn recomputed_argmaxes(
    ctx: &DeviceContext,
    serve: &GemmaServe,
    prompt: &[u32],
    positions: &[usize],
) -> Vec<usize> {
    positions
        .iter()
        .map(|&position| argmax(&serving_recompute(ctx, serve, &prompt[..=position])))
        .collect()
}

fn agreement(left: &[usize], right: &[usize]) -> usize {
    left.iter().zip(right).filter(|(a, b)| a == b).count()
}

struct AgreementCase {
    tensor_name: &'static str,
    prompt: Vec<u32>,
    sampled: Vec<usize>,
}

fn agreement_case(
    fixture: &safetensors::SafeTensors<'_>,
    tensor_name: &'static str,
    truncate_to: usize,
    stride: usize,
) -> AgreementCase {
    let (_, mut prompt) = u32_tensor(fixture, tensor_name);
    prompt.truncate(truncate_to);
    let sampled: Vec<usize> = (0..prompt.len()).step_by(stride).collect();
    AgreementCase {
        tensor_name,
        prompt,
        sampled,
    }
}

fn bf16_agreement(
    ctx: &DeviceContext,
    serve: &GemmaServe,
    case: &AgreementCase,
) -> (Vec<usize>, usize) {
    let incremental = incremental_argmaxes(ctx, serve, &case.prompt);
    // The same-schedule run-to-run baseline, measured on this exact prompt
    // and schedule: a replay must agree everywhere, which is why a lossy
    // storage is judged against the cross-shape floor below instead.
    let replay = incremental_argmaxes(ctx, serve, &case.prompt);
    let replay_matches = agreement(&incremental, &replay);
    eprintln!(
        "{}: same-schedule bf16 run-to-run agreement {replay_matches}/{}",
        case.tensor_name,
        incremental.len()
    );
    assert_eq!(
        replay_matches,
        incremental.len(),
        "{}: the same-schedule bf16 replay must be deterministic",
        case.tensor_name
    );
    let recomputed = recomputed_argmaxes(ctx, serve, &case.prompt, &case.sampled);
    let sampled = case
        .sampled
        .iter()
        .map(|&pos| incremental[pos])
        .collect::<Vec<_>>();
    let floor_matches = agreement(&sampled, &recomputed);
    (sampled, floor_matches)
}

fn judge_agreement(case: &AgreementCase, floor: usize, fp8: usize) {
    let samples = case.sampled.len();
    let samples_f64 = f64::from(u32::try_from(samples).expect("sample count fits u32"));
    let floor_rate = f64::from(u32::try_from(floor).expect("match count fits u32")) / samples_f64;
    let fp8_rate = f64::from(u32::try_from(fp8).expect("match count fits u32")) / samples_f64;
    eprintln!(
        "{}: argmax agreement: bf16 incremental/recompute {floor_rate:.6} \
         ({floor}/{samples}), fp8/bf16 incremental {fp8_rate:.6} ({fp8}/{samples})",
        case.tensor_name
    );
    assert!(
        floor * 2 > samples,
        "{}: degenerate bf16 incremental/recompute floor {floor}/{samples}",
        case.tensor_name
    );
    assert!(
        fp8 >= floor,
        "{}: fp8/bf16 argmax agreement {fp8}/{samples} is below the bf16 \
         incremental/recompute floor {floor}/{samples}",
        case.tensor_name
    );
}

#[test]
#[ignore = "requires the pinned 12B checkpoint, fixtures, and a GPU"]
fn fp8_argmax_agreement_meets_the_bf16_floor() {
    let bytes = std::fs::read(window_fixture()).expect("read window fixture");
    let fixture = safetensors::SafeTensors::deserialize(&bytes).expect("window fixture");
    let cases = [("w1023_prompt", usize::MAX, 2), ("w4096_prompt", 2048, 8)]
        .map(|(name, cut, stride)| agreement_case(&fixture, name, cut, stride));
    let max_context = cases
        .iter()
        .map(|case| {
            case.prompt.len().div_ceil(crate::kv::LOCAL_PAGE_SIZE) * crate::kv::LOCAL_PAGE_SIZE
        })
        .max()
        .expect("agreement cases");
    let pages = max_context.div_ceil(crate::kv::LOCAL_PAGE_SIZE) + 2;

    let (ctx, bf16, _) = stack_with_storage(max_context, pages, KvStorage::Bf16);
    let bf16_results = cases
        .each_ref()
        .map(|case| bf16_agreement(&ctx, &bf16, case));
    drop(bf16);
    drop(ctx);

    let (ctx, fp8, _) = stack_with_storage(max_context, pages, KvStorage::E4m3);
    let fp8_results: Vec<usize> = cases
        .iter()
        .zip(bf16_results.iter())
        .map(|(case, (bf16, _))| {
            let incremental = incremental_argmaxes(&ctx, &fp8, &case.prompt);
            let sampled = case
                .sampled
                .iter()
                .map(|&pos| incremental[pos])
                .collect::<Vec<_>>();
            agreement(&sampled, bf16)
        })
        .collect();
    drop(fp8);
    drop(ctx);

    for ((case, (_, floor)), fp8) in cases.iter().zip(bf16_results).zip(fp8_results) {
        judge_agreement(case, floor, fp8);
    }
}

/// The shape of one logit row, for when two of them disagree: the top few ids
/// and the range say whether a row is a distribution or garbage.
fn describe_row(what: &str, row: &[f32]) {
    let mut ranked: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    let lo = row.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let finite = row.iter().filter(|v| v.is_finite()).count();
    eprintln!(
        "{what}: range [{lo}, {hi}], finite {finite}/{}, top5 {:?}",
        row.len(),
        &ranked[..5.min(ranked.len())]
    );
}

/// The two arms run different launch shapes (one-token decode against a
/// whole-prompt prefill), which this engine does not promise bit-equal, so
/// the callers bound the raw-logit drift by the calibrated ceiling — but the
/// decision must not move: the argmax has to be identical, measured zero
/// flips across the fixture.
fn compare_row(ours: &[f32], theirs: &[f32], what: &str) -> f32 {
    assert!(
        ours.iter().chain(theirs.iter()).all(|v| v.is_finite()),
        "{what}: non-finite logit"
    );
    let (a, b) = (argmax(ours), argmax(theirs));
    assert_eq!(a, b, "{what}: argmax diverged ({a} vs {b})");
    ours.iter()
        .zip(theirs)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The whole-prefix recompute of the serving path, reduced to its last row.
fn serving_recompute(ctx: &DeviceContext, serve: &GemmaServe, tokens: &[u32]) -> Vec<f32> {
    let mut kv = serve.alloc_kv();
    admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, tokens.len())
        .expect("admit recompute");
    let logits = serve.step(ctx, &mut kv, tokens).expect("serving recompute");
    let host = logits.to_host(ctx).expect("recompute D2H");
    let vocab = logits.hidden_dim;
    host[(logits.seq_len - 1) * vocab..].to_vec()
}

/// Greedy continuation through the serving path: the prompt in one prefill
/// step, then `steps` decode steps each fed the previous row's argmax. Every
/// row is returned, so a divergence is placed at the step it appears, and so
/// is the context it walked, which a recompute of the same tokens needs.
fn continue_greedy(
    ctx: &DeviceContext,
    serve: &GemmaServe,
    tokens: &[u32],
    steps: usize,
) -> (Vec<Vec<f32>>, Vec<u32>) {
    let mut kv = serve.alloc_kv();
    let mut arena = serve
        .alloc_step_arena(ctx, 1, false)
        .expect("oracle step arena");
    admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, tokens.len())
        .expect("admit prompt");
    let logits = serve.step(ctx, &mut kv, tokens).expect("prefill");
    let host = logits.to_host(ctx).expect("prefill D2H");
    let vocab = logits.hidden_dim;
    let mut rows = vec![host[(logits.seq_len - 1) * vocab..].to_vec()];
    let mut walked = tokens.to_vec();
    for _ in 0..steps {
        let token = u32::try_from(argmax(rows.last().unwrap())).expect("token id");
        walked.push(token);
        rows.push(decode_serving(serve, ctx, &mut arena, &mut kv, token).expect("decode"));
    }
    (rows, walked)
}

/// The decode gates' raw-logit line. Correctness at this depth is carried by
/// the argmax, which [`compare_row`] holds on every row of every cell; this
/// catches a drift the argmax would not, and is set above the spread a
/// sixteen-cell sweep measured (worst 7.91) rather than at the edge of it,
/// because the quantity is chaotic and a tighter line would only flake. What
/// bounds the kernel's own error is the AOT gate, at 0.002 against fp32.
const DRIFT_LINE: f32 = 12.0;

/// What a decode row's logits move by when nothing about the maths changes:
/// every row of a greedy walk against a single prefill of the context that
/// row saw, one arm two ways, reduced the way the arms are compared.
///
/// Printed beside each cell so the reader has the magnitude an already
/// accepted implementation difference reaches here. It is not the line: a
/// sweep of sixteen cells put this at 0.31 to 5.75 and the replacement at
/// 0.56 to 7.91, with neither tracking prompt or length -- lengths a page
/// apart differ sevenfold and the worst row lands anywhere from the prompt
/// row to the last. A maximum over a walk of these is a draw from a heavy
/// tail, so a line derived from it in-run would move with the draw.
fn neutral_scale(
    ctx: &DeviceContext,
    serve: &GemmaServe,
    rows: &[Vec<f32>],
    prompt_len: usize,
    walked: &[u32],
) -> f32 {
    let mut worst = 0.0f32;
    for (i, row) in rows.iter().enumerate() {
        let recomputed = serving_recompute(ctx, serve, &walked[..prompt_len + i]);
        worst = worst.max(compare_row(
            row,
            &recomputed,
            &format!("neutral scale row {i}"),
        ));
    }
    worst
}

/// The contexts the decode gates sweep: every fixture prompt at three
/// lengths, so a line comes from a spread rather than from one cell.
fn decode_sweep_cells() -> Vec<(usize, usize)> {
    let mut cells = Vec::new();
    for prompt in 0..3 {
        for len in [512usize, 1500, 3000] {
            cells.push((prompt, len));
        }
    }
    cells
}

/// The replacement global-attention decode against the one it stands in for,
/// over every fixture prompt at three lengths. Each step is compared on its
/// own row, argmax first, since the arms pick their own next token: that
/// equality is the correctness the gate carries, and it holds on every row.
/// The raw-logit line is [`DRIFT_LINE`], and every cell prints its own gap
/// beside [`neutral_scale`], so the spread is on the record.
#[test]
#[ignore = "requires a Gemma 4 checkpoint, a GPU, and --test-threads=1"]
fn the_replacement_global_decode_matches_the_incumbent() {
    const STEPS: usize = 16;
    let (ctx, mut serve, _dir) = stack_with(4096, 300);
    assert!(
        pegainfer_kernels::ops::gemma4_hd512_prefill_is_built(),
        "this build has no TileLang kernel to compare; the gate needs one that does"
    );
    let prompts = crate::testkit::generate_fixture_prompts();

    let mut cells = Vec::new();
    for (prompt, len) in decode_sweep_cells() {
        let tokens: Vec<u32> = prompts[prompt].iter().cycle().copied().take(len).collect();

        serve.tilelang_global_attn = false;
        let (incumbent, walked) = continue_greedy(&ctx, &serve, &tokens, STEPS);
        for (i, row) in incumbent.iter().enumerate() {
            eprintln!(
                "prompt {prompt} at {len}: incumbent step {i} fingerprint {:016x}",
                fingerprint(row)
            );
        }
        let (again, _) = continue_greedy(&ctx, &serve, &tokens, STEPS);
        for (i, (a, b)) in incumbent.iter().zip(&again).enumerate() {
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "prompt {prompt} at {len}: the incumbent is not bit-identical run to \
                 run at step {i}, so there is no floor to measure the replacement against"
            );
        }
        let scale = neutral_scale(&ctx, &serve, &incumbent, tokens.len(), &walked);

        serve.tilelang_global_attn = true;
        let (replacement, _) = continue_greedy(&ctx, &serve, &tokens, STEPS);
        let mut worst = (0.0f32, 0usize);
        for (i, (a, b)) in incumbent.iter().zip(&replacement).enumerate() {
            let gap = compare_row(a, b, &format!("prompt {prompt} at {len}, decode step {i}"));
            if gap > worst.0 {
                worst = (gap, i);
            }
        }
        eprintln!(
            "prompt {prompt} at {len} tokens: floor 0, neutral scale {scale}, \
             replacement |dlogit| {} at step {}",
            worst.0, worst.1
        );
        cells.push((prompt, len, scale, worst));
    }

    let widest_scale = cells.iter().fold(0.0f32, |m, c| m.max(c.2));
    let worst_cell = cells
        .iter()
        .max_by(|a, b| a.3.0.total_cmp(&b.3.0))
        .expect("the sweep ran");
    let line = DRIFT_LINE;
    eprintln!(
        "global decode over {} cells: neutral scale at most {widest_scale}, line {line}, \
         worst replacement |dlogit| {} at prompt {} length {} step {}",
        cells.len(),
        worst_cell.3.0,
        worst_cell.0,
        worst_cell.1,
        worst_cell.3.1
    );
    for (prompt, len, _, (gap, step)) in &cells {
        assert!(
            *gap <= line,
            "prompt {prompt} at {len}: replacement |dlogit| {gap} at step {step} above \
             the drift line of {line}"
        );
    }
}

/// The folded global pool against the split one, both read by the generated
/// kernels: the rows differ in where K's norm weight is applied, one bf16
/// rounding apart per element. Every fixture prompt at three lengths. The
/// split arms run and are kept first, so one pool swap covers the sweep. The
/// decode rows take [`DRIFT_LINE`]; the prompt row keeps the prefill gate's
/// 2.0, which one row of one kernel pass can hold.
#[test]
#[ignore = "requires a Gemma 4 checkpoint, a GPU, and --test-threads=1"]
fn the_folded_pool_matches_the_split_one() {
    const STEPS: usize = 16;
    let (ctx, mut serve, _dir) = stack_with(4096, 300);
    assert!(
        pegainfer_kernels::ops::gemma4_hd512_prefill_is_built(),
        "this build has no TileLang kernel to compare; the gate needs one that does"
    );
    let prompts = crate::testkit::generate_fixture_prompts();
    serve.tilelang_global_attn = true;

    let cells = decode_sweep_cells();
    let mut split_arms = Vec::new();
    for &(prompt, len) in &cells {
        let tokens: Vec<u32> = prompts[prompt].iter().cycle().copied().take(len).collect();
        let (split, walked) = continue_greedy(&ctx, &serve, &tokens, STEPS);
        let (again, _) = continue_greedy(&ctx, &serve, &tokens, STEPS);
        for (i, (a, b)) in split.iter().zip(&again).enumerate() {
            assert!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "prompt {prompt} at {len}: the split arm is not bit-identical run to \
                 run at step {i}, so there is no floor to measure the folded one against"
            );
        }
        let scale = neutral_scale(&ctx, &serve, &split, tokens.len(), &walked);
        split_arms.push((tokens, split, scale));
    }

    // The same budget of pages, in the other format; the serving path
    // allocates the pool once in whichever format the knob names.
    let (layers, heads, head_dim, page_size, pages) = {
        let layout = serve.global_pool.layout();
        (
            layout.num_layers,
            layout.num_kv_heads,
            layout.head_dim,
            layout.page_size,
            serve.global_pool.capacity_pages(),
        )
    };
    let rotary = serve.weights.config.global_rotary_dim;
    serve.global_pool = KvPool::with_storage_and_format(
        &ctx,
        layers,
        heads,
        head_dim,
        page_size,
        pages,
        KvStorage::Bf16,
        KvFormat::Folded { rotary },
    )
    .expect("folded global pool");
    eprintln!(
        "global pool re-allocated as {:?}: {} elements per page against the split's",
        serve.global_pool.layout().format,
        serve.global_pool.layout().page_stride
    );

    let widest_scale = split_arms.iter().fold(0.0f32, |m, c| m.max(c.2));
    let line = DRIFT_LINE;
    let mut worst_overall = (0.0f32, 0usize, 0usize, 0usize);
    for ((prompt, len), (tokens, split, scale)) in cells.iter().zip(&split_arms) {
        let (folded, _) = continue_greedy(&ctx, &serve, tokens, STEPS);
        let mut worst = (0.0f32, 0usize);
        for (i, (a, b)) in split.iter().zip(&folded).enumerate() {
            let gap = compare_row(a, b, &format!("prompt {prompt} at {len}, row {i}"));
            if i == 0 {
                assert!(
                    gap <= 2.0,
                    "prompt {prompt} at {len}: folded prefill |dlogit| {gap} above the \
                     prefill line of 2.0"
                );
            } else if gap > worst.0 {
                worst = (gap, i);
            }
        }
        eprintln!(
            "prompt {prompt} at {len} tokens: floor 0, neutral scale {scale}, \
             folded decode |dlogit| {} at row {}",
            worst.0, worst.1
        );
        assert!(
            worst.0 <= line,
            "prompt {prompt} at {len}: folded decode |dlogit| {} at row {} above the \
             drift line of {line}",
            worst.0,
            worst.1
        );
        if worst.0 > worst_overall.0 {
            worst_overall = (worst.0, worst.1, *prompt, *len);
        }
    }
    eprintln!(
        "folded against split over {} cells: neutral scale at most {widest_scale}, line \
         {line}, worst decode |dlogit| {} at prompt {} length {} row {}",
        cells.len(),
        worst_overall.0,
        worst_overall.2,
        worst_overall.3,
        worst_overall.1
    );
}

/// FNV-1a over the row's bits.
fn fingerprint(row: &[f32]) -> u64 {
    row.iter().fold(0xcbf2_9ce4_8422_2325, |h, x| {
        (h ^ u64::from(x.to_bits())).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// The replacement global-attention kernel against the one it stands in for,
/// both through the production serving path. The incumbent is asked twice
/// first, so the tolerance is a measured floor rather than a chosen number.
/// The prompt leaves the last page partial, where tail handling could differ.
#[test]
#[ignore = "requires a Gemma 4 checkpoint, a GPU, and --test-threads=1"]
fn the_replacement_global_kernel_matches_the_incumbent() {
    let (ctx, mut serve, dir) = stack_with(4096, 300);
    assert!(
        pegainfer_kernels::ops::gemma4_hd512_prefill_is_built(),
        "this build has no TileLang kernel to compare; the gate needs one that does"
    );
    // The runner selects this gate only where the geometry matches.
    let config = crate::config::Gemma4Config::from_file(&dir).expect("config");
    crate::engine::tilelang_geometry_refusal(&config).expect(
        "this gate compares the generated kernel against the incumbent, so it needs a \
         checkpoint whose global geometry the build was compiled for",
    );
    let prompts = crate::testkit::generate_fixture_prompts();
    let tokens: Vec<u32> = prompts[0].iter().cycle().copied().take(1500).collect();
    let page = serve.global_pool.layout().page_size;
    assert!(
        !tokens.len().is_multiple_of(page),
        "the prompt has to leave the final global page partial, and {} tokens \
         divides the pool's {page}-row page",
        tokens.len()
    );

    assert!(!serve.tilelang_global_attn);
    let incumbent = serving_recompute(&ctx, &serve, &tokens);
    // The incumbent's bits on this checkpoint, to compare across trees.
    eprintln!("incumbent fingerprint {:016x}", fingerprint(&incumbent));
    let again = serving_recompute(&ctx, &serve, &tokens);
    let floor = compare_row(&incumbent, &again, "incumbent against itself");
    assert!(
        incumbent
            .iter()
            .zip(&again)
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "the incumbent is not bit-identical run to run, so there is no floor \
         to measure the replacement against"
    );

    serve.tilelang_global_attn = true;
    let replacement = serving_recompute(&ctx, &serve, &tokens);
    // Report before asserting: when the two disagree, the magnitude and the
    // shape of each row say which kind of wrong it is, and `compare_row`
    // stops at the first divergence it finds.
    describe_row("incumbent", &incumbent);
    describe_row("replacement", &replacement);
    let spread = incumbent
        .iter()
        .zip(&replacement)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max |dlogit| between the two kernels: {spread}");
    let gap = compare_row(&incumbent, &replacement, "replacement against incumbent");
    eprintln!(
        "global prefill over {} tokens: floor {floor}, replacement |dlogit| {gap}",
        tokens.len()
    );
    assert!(
        gap <= 2.0,
        "replacement |dlogit| {gap} above the line's calibrated 2.0"
    );
}

/// One forward path answers for itself: every prompt position's incremental
/// logits (one token at a time through the decode arena) match a whole-prompt
/// recompute of the same serving path, and four decode steps fed the
/// recompute's own greedy picks match it too — so one divergence cannot
/// cascade. Both arms run the production release path; this short
/// trajectory never crosses the window — the crossing itself is pinned by
/// the waypoint, mixed-window, overlap and ragged gates.
#[test]
#[ignore = "requires the pinned 12B checkpoint, the golden fixture, and a GPU"]
fn incremental_serving_matches_recompute() {
    let (ctx, serve, _dir) = load_stack();
    // The golden fixture's short prompt: real text, and the tokens the
    // ceiling below was calibrated on. Read for its prompt only.
    let path = crate::testkit::golden_path();
    let bytes = std::fs::read(path).expect("read golden fixture (dump it first)");
    let fixture = safetensors::SafeTensors::deserialize(&bytes).expect("parse fixture");
    let (_, tokens_i32) = i32_tensor(&fixture, "short_tokens");
    let mut tokens: Vec<u32> = tokens_i32
        .iter()
        .map(|&t| u32::try_from(t).expect("token id"))
        .collect();

    let mut kv = serve.alloc_kv();
    let mut arena = serve
        .alloc_step_arena(&ctx, 1, false)
        .expect("oracle step arena");
    let mut max_abs = 0.0f32;
    for pos in 0..tokens.len() {
        let incremental = if pos == 0 {
            admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, 1)
                .expect("admit first prompt token");
            serve
                .step(&ctx, &mut kv, &tokens[..1])
                .expect("first prompt token")
                .to_host(&ctx)
                .expect("D2H")
        } else {
            decode_serving(&serve, &ctx, &mut arena, &mut kv, tokens[pos])
                .expect("teacher-forced prompt token")
        };
        let recomputed = serving_recompute(&ctx, &serve, &tokens[..=pos]);
        let gap = compare_row(&incremental, &recomputed, &format!("prefill pos {pos}"));
        eprintln!("prefill pos {pos}: max |dlogit| {gap}");
        max_abs = max_abs.max(gap);
    }
    eprintln!(
        "prefill: {} positions, max |dlogit| {max_abs}",
        tokens.len()
    );
    assert!(
        max_abs <= 2.0,
        "prefill |dlogit| {max_abs} above calibrated 2.0"
    );

    let mut oracle_last = serving_recompute(&ctx, &serve, &tokens);
    for step in 0..4 {
        let next = u32::try_from(argmax(&oracle_last)).expect("token id");
        let step_host =
            decode_serving(&serve, &ctx, &mut arena, &mut kv, next).expect("serve decode step");
        tokens.push(next);
        oracle_last = serving_recompute(&ctx, &serve, &tokens);
        let gap = compare_row(&step_host, &oracle_last, &format!("decode step {step}"));
        eprintln!("decode step {step}: max |dlogit| {gap}");
        assert!(
            gap <= 2.0,
            "decode step {step} |dlogit| {gap} above calibrated 2.0"
        );
    }
}

/// Greedy continuation matches HF `generate()` token for token on three
/// prompts. The fixture is dumped by tools/accuracy/dump_gemma4_generate.py
/// (prompt + up to 50 greedy tokens per case).
#[test]
#[ignore = "requires the pinned 12B checkpoint, fixtures, and a GPU"]
fn greedy_matches_hf_generate() {
    let (ctx, serve, dir) = load_stack();
    let path = crate::testkit::fixture_path(
        "PEGAINFER_GEMMA4_GENERATE",
        "gemma4-12b-generate.safetensors",
    );
    let bytes = std::fs::read(path).expect("read generate fixture (dump it first)");
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

/// The production sampler draws from the true vocabulary; a padded lm_head
/// column is never a candidate, so neither is it here.
fn mixed_gate_argmax(host: &[f32], row: usize, stride: usize, bound: usize) -> u32 {
    u32::try_from(argmax(&host[row * stride..row * stride + bound])).expect("token id")
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
    stride: usize,
    bound: usize,
    row_base: usize,
    lanes: &mut Vec<(usize, GemmaKv, u32)>,
    produced: &mut [Vec<u32>],
    budgets: &[usize],
) {
    let mut retire: Vec<usize> = Vec::new();
    for (row, (req, _, next)) in lanes.iter_mut().enumerate() {
        let token = mixed_gate_argmax(host, row + row_base, stride, bound);
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
        settle_gate_lanes(
            &host,
            vocab,
            serve.weights.config.vocab_size,
            0,
            lanes,
            produced,
            budgets,
        );
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
        settle_gate_lanes(
            &host,
            vocab,
            serve.weights.config.vocab_size,
            2,
            &mut lanes,
            &mut produced,
            &budgets,
        );
        let first_b = mixed_gate_argmax(&host, 0, vocab, serve.weights.config.vocab_size);
        produced[1].push(first_b);
        lanes.push((1, kv_b, first_b));
        let first_c = mixed_gate_argmax(&host, 1, vocab, serve.weights.config.vocab_size);
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
/// Synthetic ids suffice: both sides run the same window arithmetic.
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
            settle_gate_lanes(
                &host,
                vocab,
                serve.weights.config.vocab_size,
                1,
                &mut lanes,
                &mut produced,
                &budgets,
            );
            mixed_gate_argmax(&host, 0, vocab, serve.weights.config.vocab_size)
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
    // This is a bf16 bit-exactness contract; distribution and waypoint gates judge fp8.
    let (ctx, serve, _dir) = stack_with_storage(2048, 512, KvStorage::Bf16);
    assert_mixed_admissions_match_serial(&ctx, &serve);
    assert_mixed_window_crossing_matches_serial(&ctx, &serve);
}

fn assert_finite_gate_logits(host: &[f32], what: &str) {
    assert!(
        host.iter().all(|value| value.is_finite()),
        "{what}: mixed step produced non-finite logits"
    );
}

fn assert_gate_page_accounting(serve: &GemmaServe, kv: &GemmaKv, what: &str) {
    let page = serve.local_pool.layout().page_size;
    let kv_len = kv.local.seq_len();
    assert_eq!(kv.global.seq_len(), kv_len, "{what}: KV lengths");
    let released = kv_len.saturating_sub(serve.sliding_window) / page;
    assert_eq!(
        kv.local.held_pages(),
        kv_len.div_ceil(page) - released,
        "{what}: local pages"
    );
    assert_eq!(
        kv.global.held_pages(),
        kv_len.div_ceil(page),
        "{what}: global pages"
    );
}

fn fp8_plain_mixed_walk(ctx: &DeviceContext, serve: &GemmaServe) {
    let prompts = crate::testkit::generate_fixture_prompts();
    let budgets = [50usize, 37, 44];
    let mut arena = serve.alloc_step_arena(ctx, 4, false).expect("step arena");
    let mut lanes = Vec::new();
    let mut produced = vec![Vec::new(); prompts.len()];

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
    assert_finite_gate_logits(&host, "plain fp8 walk");
    settle_gate_lanes(
        &host,
        vocab,
        serve.weights.config.vocab_size,
        2,
        &mut lanes,
        &mut produced,
        &budgets,
    );
    for (req, kv, row) in [(1, kv_b, 0), (2, kv_c, 1)] {
        let first = mixed_gate_argmax(&host, row, vocab, serve.weights.config.vocab_size);
        produced[req].push(first);
        lanes.push((req, kv, first));
    }
    mixed_gate_decode_rounds(
        ctx,
        serve,
        &mut arena,
        &mut lanes,
        &mut produced,
        &budgets,
        3,
    );
    mixed_gate_decode_rounds(
        ctx,
        serve,
        &mut arena,
        &mut lanes,
        &mut produced,
        &budgets,
        usize::MAX,
    );
    for (tokens, budget) in produced.iter().zip(budgets) {
        assert_eq!(tokens.len(), budget, "fp8 mixed lane token budget");
    }
}

fn fp8_window_mixed_walk(ctx: &DeviceContext, serve: &GemmaServe) {
    let partner: Vec<u32> = (0..40u32).map(|i| 1000 + i * 31).collect();
    let long_prompt: Vec<u32> = (0..1500u32).map(|i| 1000 + (i * 37) % 50000).collect();
    let budgets = [24usize, 20];
    let mut arena = serve.alloc_step_arena(ctx, 2, false).expect("step arena");
    let mut produced = vec![Vec::new(); 2];
    let (kv_partner, first_partner) = gate_open_lane(ctx, serve, &partner, "partner");
    let mut lanes = vec![(0, kv_partner, first_partner)];
    produced[0].push(first_partner);
    mixed_gate_decode_rounds(
        ctx,
        serve,
        &mut arena,
        &mut lanes,
        &mut produced,
        &budgets,
        3,
    );

    let mut kv_long = gate_admit_kv(serve, &long_prompt, "long prompt");
    let tokens = gate_step_tokens(serve, &mut lanes);
    let (vocab, host) = {
        let mut kvs: Vec<&mut GemmaKv> = lanes.iter_mut().map(|(_, kv, _)| kv).collect();
        let mut prefills = [(&mut kv_long, long_prompt.as_slice())];
        let logits = serve
            .mixed_prefill_decode_step(ctx, &mut arena, &mut prefills, &mut kvs, &tokens)
            .expect("window-crossing mixed step");
        gate_host_logits(ctx, logits)
    };
    assert_finite_gate_logits(&host, "window-crossing fp8 walk");
    assert_gate_page_accounting(serve, &kv_long, "long prompt mixed step");
    settle_gate_lanes(
        &host,
        vocab,
        serve.weights.config.vocab_size,
        1,
        &mut lanes,
        &mut produced,
        &budgets,
    );
    let first_long = mixed_gate_argmax(&host, 0, vocab, serve.weights.config.vocab_size);
    produced[1].push(first_long);
    lanes.push((1, kv_long, first_long));

    for (req, mut kv, mut next) in lanes {
        while produced[req].len() < budgets[req] {
            let host = decode_serving(serve, ctx, &mut arena, &mut kv, next).expect("decode");
            let bound = serve.weights.config.vocab_size.min(host.len());
            next = u32::try_from(argmax(&host[..bound])).expect("token id");
            produced[req].push(next);
        }
        assert_gate_page_accounting(serve, &kv, ["partner", "long prompt"][req]);
    }
    for (tokens, budget) in produced.iter().zip(budgets) {
        assert_eq!(tokens.len(), budget, "fp8 window lane token budget");
    }
}

#[test]
#[ignore = "requires the pinned 12B checkpoint, generate prompts, and a GPU"]
fn fp8_mixed_walk_holds_its_structure() {
    let (ctx, serve, _dir) = stack_with_storage(2048, 512, KvStorage::E4m3);
    fp8_plain_mixed_walk(&ctx, &serve);
    fp8_window_mixed_walk(&ctx, &serve);
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
    // The cache's page copies are bf16-only; distribution and waypoint gates judge fp8.
    let (ctx, serve, _dir) = stack_with_storage(4096, 512, KvStorage::Bf16);
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
fn generate_greedy(
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
        .map(|len| (len + STEPS).div_ceil(LOCAL_PAGE_SIZE))
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
