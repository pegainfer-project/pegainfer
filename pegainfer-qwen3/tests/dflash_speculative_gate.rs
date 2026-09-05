//! DFlash speculative-decoding losslessness gate.
//!
//! Greedy speculative decoding must be *lossless*: every draft is verified by a
//! target forward and only the matching-argmax prefix (plus one bonus) is
//! committed, so the accepted tokens are the target model's own greedy
//! continuation. The catch is pure numerics — the verify path runs the
//! *prefill* attention kernel over the K+1 span while a plain decode runs the
//! *decode* kernel, and the two differ by ~1 bf16 ULP. On a near-tie that flips
//! one argmax, and from there two greedy runs fan out completely.
//!
//! So an exact `spec == baseline` token match is the wrong gate: it false-fails
//! on a benign tie flip. We use a *regret* test like `hf_golden_gate`: at the
//! first position the two sequences disagree (where they still share an
//! identical context, so the comparison is valid) we ask how far below the
//! argmax the speculative pick sits — measured *in the prefill kernel's own
//! distribution*, because that is the kernel the verify path runs. A re-prefill
//! of the shared context (`prefill_next`) gives that reference distribution.
//! The verify path's committed KV is built incrementally across batched
//! speculative spans, while a one-shot prefill builds it in a single forward;
//! the two K/V differ by a few bf16 ULP, so on a near-tie the argmax flips.
//! Within `MARGIN_TOL` of the prefill argmax ⇒ a benign numerical tie. Clearly
//! worse (or outside the prefill top-K) ⇒ the verify/accept/capture logic chose
//! a token the forward never favored — a real bug. A systematic bug corrupts
//! the non-tie positions too, so it cannot hide behind the tie band.
//!
//! (Empirically the one prompt that flips — "The capital of France is" — sits on
//! a Germany-vs-Paris near-tie: the prefill kernel scores them -0.71 vs -0.83,
//! a 0.12-nat gap, well inside `MARGIN_TOL`. The other four prompts are bit
//! identical. A real verify bug would not single out the one degenerate prompt.)
//!
//! The baseline runs with logprobs on (plain decode); the speculative engine
//! runs with logprobs off (logprobs force the spec path off by design), so it
//! reports chosen tokens only — exactly what the regret check needs.
//!
//! Runs the two engines sequentially (baseline dropped before the speculative
//! engine loads) so only one Qwen3-4B is resident at a time.
//!
//! Requires a CUDA GPU, Qwen3-4B weights, and the DFlash drafter. Set
//! `PEGAINFER_TEST_MODEL_PATH` (target) and `PEGAINFER_DFLASH_TEST_MODEL_PATH`
//! (drafter); skips cleanly when either is absent.

use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen3::DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES;
use pegainfer_qwen3::DEFAULT_KV_PAGE_SIZE;
use pegainfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS;
use pegainfer_qwen3::DecodeOverlap;
use pegainfer_qwen3::Qwen3LaunchOptions;
use pegainfer_qwen3::Qwen3MemoryOptions;
use pegainfer_qwen3::Qwen3OffloadOptions;
use vllm_text::tokenizer::DynTokenizer;

mod common;

use common::harness::EngineHarness;
use common::harness::Outcome;
use common::harness::request;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const DRAFT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B-DFlash-b16");
const GENERATED_TOKENS: usize = 64;
/// Top-K logprobs requested from the baseline; wide enough that the speculative
/// pick is in the set on any real tie (a pick outside the top-K is itself a red
/// flag the gate should catch).
const LOGPROBS: usize = 20;
/// Max acceptable regret: how far below the baseline's argmax (in the baseline's
/// own logprobs) the speculative pick may sit at the divergence point. ~3 bf16
/// ULP at typical logit magnitudes — mirrors `hf_golden_gate`'s `MARGIN_TOL`.
const MARGIN_TOL: f32 = 0.20;

/// Both tests launch a Qwen3-4B engine, and two at once overflow a 16 GB card.
/// Cargo runs tests in one binary concurrently, so serialize the engine-holding
/// bodies — only one engine is ever resident on the GPU.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn target_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping dflash gate: {MODEL_PATH}/config.json missing; set PEGAINFER_TEST_MODEL_PATH"
            );
            None
        }
    }
}

fn draft_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_DFLASH_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(DRAFT_PATH).join("config.json").exists() => {
            Some(DRAFT_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping dflash gate: {DRAFT_PATH}/config.json missing; set PEGAINFER_DFLASH_TEST_MODEL_PATH"
            );
            None
        }
    }
}

fn launch_options(draft: Option<PathBuf>) -> Qwen3LaunchOptions {
    Qwen3LaunchOptions {
        device_ordinal: 0,
        tp_size: 1,
        cuda_graph: true,
        dump_graph_png: None,
        offload: Qwen3OffloadOptions::disabled(),
        // The speculative engine forces the prefix cache off; match it on the
        // baseline so both take the same cold prefill path.
        no_prefix_cache: true,
        max_prefill_tokens: DEFAULT_MAX_PREFILL_TOKENS,
        memory: Qwen3MemoryOptions::new(
            0.85,
            DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES,
            DEFAULT_KV_PAGE_SIZE,
        )
        .validate()
        .expect("valid memory options"),
        lora: None,
        decode_overlap: DecodeOverlap::Off,
        batch_invariant: false,
        dflash_draft_model_path: draft,
    }
}

/// One decoded position: the chosen token and (when requested) the top-K
/// `(token, logprob)` distribution that produced it.
struct Step {
    id: u32,
    top_logprobs: Vec<(u32, f32)>,
}

/// Fold a finished request's outcome into decoded steps: tokens paired with
/// their requested top-K distributions (empty when logprobs were off).
fn to_steps(outcome: Outcome) -> Vec<Step> {
    let Outcome {
        tokens, logprobs, ..
    } = outcome;
    tokens
        .into_iter()
        .enumerate()
        .map(|(i, id)| Step {
            id,
            top_logprobs: logprobs
                .get(i)
                .and_then(Option::as_ref)
                .map(|lp| lp.top_logprobs.clone())
                .unwrap_or_default(),
        })
        .collect()
}

/// Submit one greedy request and collect the decoded steps until it finishes.
fn generate(
    engine: &EngineHarness,
    prompt_tokens: Vec<u32>,
    logprobs: usize,
    max_tokens: usize,
) -> Vec<Step> {
    let mut req = request(prompt_tokens, SamplingParams::default(), max_tokens);
    req.logprobs = logprobs;
    to_steps(engine.submit(req).expect_finished())
}

/// Submit several greedy requests at once, then collect each one's steps. They
/// run concurrently in the one engine — the scheduler batches them — so with
/// heterogeneous `max_tokens` the verify spans differ across a batch, exercising
/// the real bs>1 draft+verify path. Each tuple is `(prompt_tokens, max_tokens)`;
/// logprobs are off so the speculative path stays active. Returns one step list
/// per request, in submission order.
fn generate_concurrent(engine: &EngineHarness, requests: Vec<(Vec<u32>, usize)>) -> Vec<Vec<Step>> {
    // Submit all up front so they coexist in the engine and form real batches.
    let streams: Vec<_> = requests
        .into_iter()
        .map(|(prompt_tokens, max_tokens)| {
            engine.submit(request(
                prompt_tokens,
                SamplingParams::default(),
                max_tokens,
            ))
        })
        .collect();

    // Fold each stream to its terminal (updates are buffered per request, so
    // the fold order doesn't matter — they all ran concurrently).
    streams
        .into_iter()
        .map(|stream| to_steps(stream.expect_finished()))
        .collect()
}

/// Prefill `context` (echo) and return the next-token distribution the *prefill*
/// kernel produces — the kernel the speculative verify path also uses. This is
/// the reference the spec pick should match (vs the plain-decode baseline, whose
/// kernel resolves bifurcation ties to the other side). Returns the first
/// generated token's `(id, top_logprobs)`.
fn prefill_next(engine: &EngineHarness, context: Vec<u32>, logprobs: usize) -> Step {
    let mut req = request(context, SamplingParams::default(), 1);
    req.logprobs = logprobs;
    req.echo = true;
    let mut generated = to_steps(engine.submit(req).expect_finished());
    assert!(
        !generated.is_empty(),
        "echo prefill finished without a token"
    );
    generated.swap_remove(0)
}

/// Compare one prompt's speculative `spec` steps against its plain-greedy `base`,
/// tolerating only the benign prefill-vs-decode kernel-gap tie flip (the spec
/// pick sits within `MARGIN_TOL` of the prefill kernel's own argmax, measured in
/// the prefill distribution the verify path actually runs). `Ok(())` ⇒ lossless
/// or a benign tie; `Err(diagnostic)` ⇒ a real spec bug. `engine` must be the
/// live speculative engine — at a divergence it re-prefills the shared context
/// (`prompt_tokens` + the matched prefix) to read that prefill-kernel reference.
fn check_lossless(
    engine: &EngineHarness,
    tokenizer: &DynTokenizer,
    i: usize,
    prompt: &str,
    prompt_tokens: &[u32],
    base: &[Step],
    spec: &[Step],
) -> Result<(), String> {
    let matched = base
        .iter()
        .zip(spec)
        .take_while(|(b, s)| b.id == s.id)
        .count();

    // Identical sequences (or one a prefix of the other): perfectly lossless.
    if matched == base.len().min(spec.len()) {
        eprintln!(
            "prompt {i} ({prompt:?}): {matched}/{} tokens identical (100% lossless)",
            base.len()
        );
        return Ok(());
    }

    let spec_id = spec[matched].id;
    let decode_argmax = base[matched].top_logprobs[0].0;

    // Diagnostic: show the exact branch point.
    {
        let lo = matched.saturating_sub(2);
        let hi = (matched + 3).min(base.len()).min(spec.len());
        let base_ids: Vec<u32> = base[..hi].iter().map(|s| s.id).collect();
        let spec_ids: Vec<u32> = spec[..hi].iter().map(|s| s.id).collect();
        eprintln!("  [diag] prompt {i} matched={matched}");
        eprintln!(
            "  [diag] context+gen base ids {:?} = {:?}",
            base_ids,
            tokenizer.decode(&base_ids, false).unwrap_or_default()
        );
        eprintln!(
            "  [diag] base[{lo}..{hi}] = {:?}",
            base[lo..hi]
                .iter()
                .map(|s| (s.id, tokenizer.decode(&[s.id], false).unwrap_or_default()))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "  [diag] spec[{lo}..{hi}] = {:?}",
            spec[lo..hi]
                .iter()
                .map(|s| (s.id, tokenizer.decode(&[s.id], false).unwrap_or_default()))
                .collect::<Vec<_>>()
        );
        let _ = spec_ids;
    }

    // The verify path runs the prefill kernel, so the right reference for the
    // spec pick is a plain *prefill* of the same shared context — not the
    // plain-decode baseline, whose kernel resolves a bifurcation tie to the
    // other side and amplifies the gap.
    let mut context = prompt_tokens.to_vec();
    context.extend(base[..matched].iter().map(|s| s.id));
    let prefill_ref = prefill_next(engine, context, LOGPROBS);

    if prefill_ref.id == spec_id {
        // Spec faithfully reproduced the prefill-kernel greedy pick; the
        // divergence is purely the pre-existing prefill-vs-decode kernel gap.
        let decode_lp = base[matched]
            .top_logprobs
            .iter()
            .find(|(t, _)| *t == spec_id)
            .map(|(_, lp)| base[matched].top_logprobs[0].1 - lp);
        eprintln!(
            "prompt {i} ({prompt:?}): kernel-gap flip at token {matched} — verify(prefill)→{spec_id}, \
             decode→{decode_argmax}; spec matches prefill greedy (decode-margin {:?}). Not a spec bug.",
            decode_lp
        );
        return Ok(());
    }

    // Spec's greedy pick differs from the prefill-kernel argmax too. The verify
    // path builds its committed KV incrementally across batched speculative
    // spans while this reference prefill builds it in one shot; the two differ by
    // a few bf16 ULP. Within MARGIN_TOL of the prefill argmax ⇒ a benign tie
    // flip; clearly worse ⇒ the verify/accept/capture logic picked a token the
    // forward never favored — a real bug.
    let prefill_regret = prefill_ref
        .top_logprobs
        .iter()
        .find(|(t, _)| *t == spec_id)
        .map(|(_, lp)| prefill_ref.top_logprobs[0].1 - lp);

    if let Some(regret) = prefill_regret {
        if regret <= MARGIN_TOL {
            eprintln!(
                "prompt {i} ({prompt:?}): tie flip at token {matched} — \
                 verify(prefill)→{}, spec→{spec_id}, decode→{decode_argmax}; \
                 spec pick is #2 in the prefill distribution (regret {regret:.3} ≤ {MARGIN_TOL}). \
                 Not a spec bug.",
                prefill_ref.id,
            );
            return Ok(());
        }
    }

    // Either the spec pick is outside the prefill top-K entirely, or it sits
    // clearly below the prefill argmax — neither is a benign tie.
    let decode_regret = base[matched]
        .top_logprobs
        .iter()
        .find(|(t, _)| *t == spec_id)
        .map(|(_, lp)| base[matched].top_logprobs[0].1 - lp);
    Err(format!(
        "prompt {i}: at token {matched} spec chose {spec_id} but prefill greedy says {} and \
         decode greedy says {decode_argmax} (spec regret in prefill dist: {prefill_regret:?} > \
         {MARGIN_TOL}; in decode dist: {decode_regret:?}) — real spec bug",
        prefill_ref.id,
    ))
}

#[test]
fn dflash_speculative_greedy_matches_plain_greedy() {
    common::harness::init_capture_logging();
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let prompts = [
        "The capital of France is",
        "Here is a short story about a dragon. Once upon a time",
        "def fibonacci(n):",
        "Q: What is 17 multiplied by 23? A: Let's think step by step.",
        "The three primary colors are",
    ];

    let tokenizer = common::load_tokenizer(&model_path);
    let encoded: Vec<Vec<u32>> = prompts
        .iter()
        .map(|p| tokenizer.encode(p, false).expect("encode failed"))
        .collect();

    // 1. Baseline: plain greedy decode (speculative off), with logprobs so the
    //    regret check has the reference distribution at the divergence point.
    let baseline: Vec<Vec<Step>> = {
        let engine = EngineHarness::new(
            pegainfer_qwen3::launch(Path::new(&model_path), launch_options(None))
                .expect("failed to start baseline engine"),
        );
        let out = encoded
            .iter()
            .map(|t| generate(&engine, t.clone(), LOGPROBS, GENERATED_TOKENS))
            .collect();
        drop(engine);
        // Let the scheduler thread tear down and free GPU memory before the
        // speculative engine loads the same 8 GB target.
        std::thread::sleep(Duration::from_secs(2));
        out
    };

    // 2. Speculative: DFlash draft + verify (logprobs off ⇒ spec path active).
    //    Keep the engine alive through analysis: at a divergence we re-prefill
    //    the shared context to read the prefill-kernel reference (the kernel the
    //    verify path uses), which the plain-decode baseline cannot provide.
    let engine = EngineHarness::new(
        pegainfer_qwen3::launch(
            Path::new(&model_path),
            launch_options(Some(PathBuf::from(&draft_path))),
        )
        .expect("failed to start speculative engine"),
    );

    let mut failures = Vec::new();
    for (i, &prompt) in prompts.iter().enumerate() {
        let spec = generate(&engine, encoded[i].clone(), 0, GENERATED_TOKENS);
        if let Err(failure) = check_lossless(
            &engine,
            &tokenizer,
            i,
            prompt,
            &encoded[i],
            &baseline[i],
            &spec,
        ) {
            failures.push(failure);
        }
    }

    drop(engine);

    assert!(
        failures.is_empty(),
        "speculative greedy decode is not lossless:\n{}",
        failures.join("\n")
    );
}

/// Verify-graph capture-shape regression (heterogeneous `max_tokens`).
///
/// The piecewise verify CUDA Graph keys its captured dense segments by
/// `batch_size` alone, but a request near its output budget shortens its verify
/// span (`scheduler::plan` truncates the span to the remaining budget), so
/// `total_tokens` — the row count the captured segments bake into their launch
/// grid — varies at a *fixed* batch size. A graph captured at a short span and
/// then replayed at a longer one processes too few rows: the trailing requests
/// read stale logits, silently breaking the lossless contract.
///
/// Neither existing check can see this. The bs=1 gate above issues each request
/// sequentially, so every fresh request's first verify is a *full* span and
/// bucket bs=1 is always first-captured at the maximal shape (only the harmless
/// over-compute direction occurs). A homogeneous concurrent benchmark is no
/// better: lockstep requests capture every bucket at full span during ramp-up,
/// and all truncation happens later as they finish together (still the safe
/// direction). The dangerous direction needs *heterogeneous* progress.
///
/// This reproduces it deterministically, single-stream: a `max_tokens=8` request
/// (span < `block_size`) captures the bucket-bs=1 graph at a truncated shape,
/// then a `max_tokens=64` request on the *same* engine replays that poisoned
/// graph at the full span. On the buggy code the long request diverges from
/// plain greedy; with full-shape gating (truncated spans run eager, so the graph
/// is only ever captured/replayed at the maximal shape) it stays lossless.
#[test]
fn dflash_short_then_long_verify_capture_is_lossless() {
    // << block_size (16): the poison request's only verify step is a short
    // truncated span, capturing the bucket-bs=1 graph at total_tokens far below
    // the full span. The fewer valid rows, the sooner a full-span replay hits the
    // stale tail — so the victim diverges early, well clear of its token budget.
    const POISON_MAX_TOKENS: usize = 4;

    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    let poison_prompt = "Hello, world! Tell me a story.";
    let victim_prompt = "Q: What is 17 multiplied by 23? A: Let's think step by step.";

    let tokenizer = common::load_tokenizer(&model_path);
    let poison_tokens = tokenizer
        .encode(poison_prompt, false)
        .expect("encode failed");
    let victim_tokens = tokenizer
        .encode(victim_prompt, false)
        .expect("encode failed");

    // 1. Baseline: the victim's plain-greedy decode (spec off) with logprobs, for
    //    the regret reference at any divergence.
    let baseline = {
        let engine = EngineHarness::new(
            pegainfer_qwen3::launch(Path::new(&model_path), launch_options(None))
                .expect("failed to start baseline engine"),
        );
        let out = generate(&engine, victim_tokens.clone(), LOGPROBS, GENERATED_TOKENS);
        drop(engine);
        // Free the target before the speculative engine loads the same 8 GB.
        std::thread::sleep(Duration::from_secs(2));
        out
    };

    // 2. Speculative engine, shared across both requests so the bucket-bs=1
    //    capture from the poison request persists into the victim's replay.
    let engine = EngineHarness::new(
        pegainfer_qwen3::launch(
            Path::new(&model_path),
            launch_options(Some(PathBuf::from(&draft_path))),
        )
        .expect("failed to start speculative engine"),
    );

    // Poison: a short request whose only verify step has total_tokens < span,
    // first-capturing the bucket-bs=1 graph at the truncated shape.
    let poison = generate(&engine, poison_tokens, 0, POISON_MAX_TOKENS);
    assert!(
        poison.len() <= POISON_MAX_TOKENS,
        "poison request emitted {} tokens, expected <= {POISON_MAX_TOKENS}",
        poison.len()
    );

    // Victim: a full-span replay of the poisoned bucket-bs=1 graph.
    let spec = generate(&engine, victim_tokens.clone(), 0, GENERATED_TOKENS);

    let result = check_lossless(
        &engine,
        &tokenizer,
        0,
        victim_prompt,
        &victim_tokens,
        &baseline,
        &spec,
    );
    drop(engine);

    assert!(
        result.is_ok(),
        "verify capture-shape bug: the long request diverged from plain greedy after a short \
         request poisoned the bucket-bs=1 graph at a truncated span:\n{}",
        result.unwrap_err()
    );
}

/// Concurrent, heterogeneous-`max_tokens` losslessness coverage for the bs>1
/// draft+verify path. The bs=1 gate and the homogeneous c8/c16 benches never
/// exercise a real batch with requests at *different* verify-span lengths; this
/// runs several greedy requests concurrently with staggered budgets and asserts
/// each stays lossless vs its own plain-greedy baseline (tolerating only the
/// benign bf16 tie-flip via the shared regret check). A batched-draft indexing
/// regression or a capture-shape mismatch at bs>1 would surface here as a real
/// (non-tie) divergence.
#[test]
fn dflash_concurrent_heterogeneous_is_lossless() {
    common::harness::init_capture_logging();
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // Distinct prompts with staggered budgets: at any tick the in-flight batch
    // mixes full and near-budget (truncated) verify spans.
    let cases: [(&str, usize); 4] = [
        ("def fibonacci(n):", 64),
        ("The three primary colors are", 24),
        (
            "Q: What is 17 multiplied by 23? A: Let's think step by step.",
            48,
        ),
        ("Here is a short story about a dragon. Once upon a time", 40),
    ];

    let tokenizer = common::load_tokenizer(&model_path);
    let encoded: Vec<Vec<u32>> = cases
        .iter()
        .map(|(p, _)| tokenizer.encode(p, false).expect("encode failed"))
        .collect();

    // 1. Baselines: each prompt's plain-greedy decode (spec off) at ITS budget,
    //    with logprobs for the regret reference. Sequential, one engine.
    let baselines: Vec<Vec<Step>> = {
        let engine = EngineHarness::new(
            pegainfer_qwen3::launch(Path::new(&model_path), launch_options(None))
                .expect("failed to start baseline engine"),
        );
        let out = encoded
            .iter()
            .zip(&cases)
            .map(|(t, (_, max_tokens))| generate(&engine, t.clone(), LOGPROBS, *max_tokens))
            .collect();
        drop(engine);
        std::thread::sleep(Duration::from_secs(2));
        out
    };

    // 2. Speculative engine: submit all four at once so they form real batches.
    let engine = EngineHarness::new(
        pegainfer_qwen3::launch(
            Path::new(&model_path),
            launch_options(Some(PathBuf::from(&draft_path))),
        )
        .expect("failed to start speculative engine"),
    );
    let specs = generate_concurrent(
        &engine,
        encoded
            .iter()
            .zip(&cases)
            .map(|(t, (_, max_tokens))| (t.clone(), *max_tokens))
            .collect(),
    );

    let mut failures = Vec::new();
    for (i, (prompt, _)) in cases.iter().enumerate() {
        if let Err(failure) = check_lossless(
            &engine,
            &tokenizer,
            i,
            prompt,
            &encoded[i],
            &baselines[i],
            &specs[i],
        ) {
            failures.push(failure);
        }
    }
    drop(engine);

    assert!(
        failures.is_empty(),
        "concurrent heterogeneous speculative decode is not lossless:\n{}",
        failures.join("\n")
    );
}

/// P2 regression: a request that fits the target context window but lands in the
/// draft's `block_size` in-fill headroom (`max_pos - block_size < prompt +
/// max_tokens <= max_pos`) must be rejected cleanly at admission. Before the
/// admission cap, such a request was admitted on the target's limit and then
/// panicked mid-prefill when the draft allocated KV past its own max positions.
#[test]
fn dflash_request_in_draft_headroom_is_rejected_not_panicked() {
    let (Some(model_path), Some(draft_path)) = (target_path_or_skip(), draft_path_or_skip()) else {
        return;
    };
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);

    // Read both context windows so the boundary is exact for ANY checkpoint
    // pairing: the request must clear the target's own limit but land inside
    // the draft's final in-fill block `(draft_max - block_size, draft_max]`,
    // where the DFlash admission cap (`draft_max - block_size`) rejects it.
    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(&model_path).join("config.json")).expect("read config"),
    )
    .expect("parse config");
    let max_pos = config["max_position_embeddings"]
        .as_u64()
        .expect("max_position_embeddings") as usize;
    let draft_config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(&draft_path).join("config.json"))
            .expect("read draft config"),
    )
    .expect("parse draft config");
    let block_size = draft_config["block_size"]
        .as_u64()
        .expect("draft block_size") as usize;
    let draft_max = draft_config["max_position_embeddings"]
        .as_u64()
        .map_or(max_pos, |v| v as usize);
    let prompt_len = 16usize;
    let total = (draft_max - block_size / 2).min(max_pos);
    if total <= draft_max - block_size {
        eprintln!(
            "skipping draft-headroom case: target max {max_pos} cannot reach the draft in-fill window (draft max {draft_max}, block {block_size})"
        );
        return;
    }
    let max_tokens = total - prompt_len;

    let engine = EngineHarness::new(
        pegainfer_qwen3::launch(
            Path::new(&model_path),
            launch_options(Some(PathBuf::from(&draft_path))),
        )
        .expect("failed to start speculative engine"),
    );

    let outcome = engine
        .submit(request(
            vec![100u32; prompt_len],
            SamplingParams::default(),
            max_tokens,
        ))
        .outcome();
    assert!(
        outcome.tokens.is_empty(),
        "draft-headroom request was admitted instead of rejected"
    );
    match outcome.terminal {
        Terminal::Rejected { reason, .. } => {
            eprintln!("draft-headroom request rejected as expected: {reason}");
        }
        Terminal::Finished { .. } => {
            panic!("draft-headroom request was admitted instead of rejected")
        }
        Terminal::Failed { message, .. } => {
            panic!(
                "draft-headroom request errored mid-flight instead of clean rejection: {message}"
            )
        }
    }
}

/// Execution gate for the hedge ladder: re-runs the two losslessness tests
/// above in child processes, since the hedge config is read once per process
/// and cannot be toggled in-process.
///
/// Scope: this proves the copy-back and discard branches were ENTERED, not
/// that a later round consumes the winner's state correctly — `check_lossless`
/// returns at the first benign tie flip, so the suffix past it is never
/// compared. A tie is also a non-win (the win test is strictly-greater), so
/// the counters cannot separate a tie from a shorter chain.
///
/// Strict token equality against an unhedged run is NOT a valid contract:
/// hedged rounds change the verify batch shape, which legally flips bf16 ties,
/// and `--batch-invariant` rejects DFlash.
#[test]
fn hedged_ladder_passes_the_lossless_gates() {
    // Strict mode (CI / validation boxes): every skip below is a failure —
    // the gate must not go green without actually executing the hedge.
    let strict = std::env::var("PEGAINFER_REQUIRE_HEDGE_GATE").is_ok_and(|v| v == "1");
    let (target, draft) = (target_path_or_skip(), draft_path_or_skip());
    let (Some(_), Some(draft_path)) = (target, draft) else {
        assert!(
            !strict,
            "hedged gate requires model fixtures when PEGAINFER_REQUIRE_HEDGE_GATE is set"
        );
        return;
    };
    // A plain DFlash drafter (markov_rank == 0) never hedges; a green run
    // against it would be vacuous.
    let draft_config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(Path::new(&draft_path).join("config.json"))
            .expect("read draft config"),
    )
    .expect("parse draft config");
    if draft_config["markov_rank"].as_u64().unwrap_or(0) == 0 {
        assert!(
            !strict,
            "hedged gate requires a DSpark drafter but {draft_path} has markov_rank 0"
        );
        eprintln!(
            "skipping hedged gate: {draft_path} is a plain DFlash drafter (markov_rank 0); \
             point PEGAINFER_DFLASH_TEST_MODEL_PATH at a DSpark checkpoint"
        );
        return;
    }
    // Hold this process's GPU slot for the whole child run: the children own
    // the card, and sibling tests here must not load an engine beside them.
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let exe = std::env::current_exe().expect("test binary path");
    // One child per lossless suite, each with a single exact filter, so the
    // invocation shape is beyond dispute on any libtest version.
    // A hedge-free child would make its lossless pass vacuous, so each child
    // must show expanded spans in the executor's per-round trace.
    let mut total_spans = 0usize;
    let mut total_wins = 0usize;
    let mut total_rounds = 0usize;
    for child_test in [
        "dflash_speculative_greedy_matches_plain_greedy",
        "dflash_concurrent_heterogeneous_is_lossless",
    ] {
        let output = Command::new(&exe)
            .args(["--exact", child_test, "--test-threads=1", "--nocapture"])
            .env("PEGAINFER_SPEC_HEDGE", "8")
            .env("PEGAINFER_SPEC_HEDGE_POSITIONS", "0,1,2")
            .env("PEGAINFER_TEST_LOG", "1")
            .env_remove("PEGAINFER_SPEC_HEDGE_AUTO")
            .output()
            .expect("spawn hedged child gate");
        let child_stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        assert!(
            output.status.success(),
            "hedged lossless gate '{child_test}' failed:\n{}\n{child_stderr}",
            String::from_utf8_lossy(&output.stdout),
        );
        let mut rounds = 0usize;
        let mut spans = 0usize;
        let mut wins = 0usize;
        for line in child_stderr.lines() {
            let Some(rest) = line.split("DFlash hedge: ").nth(1) else {
                continue;
            };
            let mut nums = rest
                .split(|ch: char| !ch.is_ascii_digit())
                .filter(|tok| !tok.is_empty())
                .map(|tok| tok.parse::<usize>().expect("hedge trace number"));
            spans += nums.next().expect("span count");
            wins += nums.next().expect("win count");
            rounds += 1;
        }
        assert!(
            rounds > 0 && spans > 0,
            "child '{child_test}' executed no hedged verify round:\n{child_stderr}"
        );
        total_rounds += rounds;
        total_spans += spans;
        total_wins += wins;
    }
    assert!(
        total_wins > 0,
        "no hedge chain ever won across {total_rounds} hedged rounds"
    );
    assert!(
        total_spans > total_wins,
        "every hedge span won ({total_wins}/{total_spans}) — the discard path never executed"
    );
}
