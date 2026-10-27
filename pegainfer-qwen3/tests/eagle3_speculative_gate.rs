//! EAGLE-3 speculative-decoding gate (greedy, one request per draft round).
//!
//! Greedy speculation must be lossless: every committed token is the target's
//! own pick, so the output matches plain greedy decode up to the benign
//! prefill-vs-decode tie flips that `common::lossless` tolerates. The tests
//! also read the engine's speculative counters, so a run that should draft
//! fails when it silently decodes plainly. Losslessness does not check the
//! draft forward itself: the verifier rejects a wrong draft, which only lowers
//! acceptance.
//!
//! Requires a CUDA GPU, Qwen3-4B weights and an EAGLE-3 drafter for it (e.g.
//! `AngelSlim/Qwen3-4B_eagle3`). Set `PEGAINFER_TEST_MODEL_PATH` (target) and
//! `PEGAINFER_EAGLE3_TEST_MODEL_PATH` (drafter); the tests skip when either is
//! absent. The two engines run one after the other so only one target is
//! resident at a time.

use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use pegainfer_frontend::engine::SpecDecodeCounters;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen3::DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES;
use pegainfer_qwen3::DEFAULT_KV_PAGE_SIZE;
use pegainfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS;
use pegainfer_qwen3::DecodeOverlap;
use pegainfer_qwen3::Qwen3LaunchOptions;
use pegainfer_qwen3::Qwen3MemoryOptions;
use pegainfer_qwen3::Qwen3OffloadOptions;

mod common;

use common::harness::EngineHarness;
use common::harness::RequestStream;
use common::harness::request;
use common::lossless::LOGPROBS;
use common::lossless::Step;
use common::lossless::check_lossless;
use common::lossless::generate;
use common::lossless::to_steps;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const DRAFT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B_eagle3");
const GENERATED_TOKENS: usize = 64;
/// The abort test's victim ignores EOS and has room for far more tokens than
/// one request decodes in `RETIRE_TIMEOUT`, so going idle within it means the
/// abort retired the victim.
const VICTIM_TOKENS: usize = 8192;
const RETIRE_TIMEOUT: Duration = Duration::from_secs(2);

const PROMPTS: [&str; 5] = [
    "The capital of France is",
    "Here is a short story about a dragon. Once upon a time",
    "def fibonacci(n):",
    "Q: What is 17 multiplied by 23? A: Let's think step by step.",
    "The three primary colors are",
];

/// One Qwen3-4B engine at a time fits the card; cargo runs tests concurrently.
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn path_or_skip(env: &str, default: &str) -> Option<String> {
    match std::env::var(env) {
        Ok(path) => Some(path),
        Err(_) if Path::new(default).join("config.json").exists() => Some(default.to_string()),
        Err(_) => {
            eprintln!("skipping eagle3 gate: {default}/config.json missing; set {env}");
            None
        }
    }
}

fn paths_or_skip() -> Option<(String, String)> {
    Some((
        path_or_skip("PEGAINFER_TEST_MODEL_PATH", MODEL_PATH)?,
        path_or_skip("PEGAINFER_EAGLE3_TEST_MODEL_PATH", DRAFT_PATH)?,
    ))
}

fn launch(model_path: &str, draft: Option<&str>) -> EngineHarness {
    let options = Qwen3LaunchOptions {
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
        dflash_draft_model_path: None,
        eagle3_draft_model_path: draft.map(PathBuf::from),
    };
    EngineHarness::new(
        pegainfer_qwen3::launch(Path::new(model_path), options).expect("failed to start engine"),
    )
}

/// Plain greedy decode (speculation off) with logprobs, one prompt at a time.
/// Each case is `(prompt_tokens, max_tokens)`.
fn baseline(model_path: &str, cases: &[(Vec<u32>, usize)]) -> Vec<Vec<Step>> {
    let engine = launch(model_path, None);
    let out = cases
        .iter()
        .map(|(tokens, max_tokens)| generate(&engine, tokens.clone(), LOGPROBS, *max_tokens))
        .collect();
    drop(engine);
    out
}

fn counters(engine: &EngineHarness) -> SpecDecodeCounters {
    engine
        .metrics()
        .spec_decode
        .expect("EAGLE-3 engine publishes speculative counters")
}

/// Drafts proposed and accepted since `before`, with the accepted count per
/// draft position (the accepted-length distribution).
fn report(name: &str, before: &SpecDecodeCounters, after: &SpecDecodeCounters) -> (u64, u64) {
    let proposed = after.num_draft_tokens - before.num_draft_tokens;
    let accepted = after.num_accepted_tokens - before.num_accepted_tokens;
    let width = usize::try_from(after.num_spec_tokens).expect("spec width fits usize");
    let per_pos: Vec<u64> = (0..width)
        .map(|i| after.num_accepted_tokens_per_pos[i] - before.num_accepted_tokens_per_pos[i])
        .collect();
    eprintln!(
        "eagle3 {name}: rounds={}, proposed={proposed}, accepted={accepted}, per_position={per_pos:?}",
        after.num_drafts - before.num_drafts
    );
    (proposed, accepted)
}

/// Read updates from `stream` until it has produced at least `tokens` tokens,
/// and return them.
fn take_tokens(stream: &mut RequestStream, tokens: usize) -> Vec<Step> {
    let mut steps = Vec::new();
    while steps.len() < tokens {
        let update = stream.recv().expect("engine closed the stream");
        assert!(
            update.terminal.is_none(),
            "request finished after {} tokens",
            steps.len()
        );
        steps.extend(update.tokens.iter().map(|&id| Step {
            id,
            top_logprobs: Vec::new(),
        }));
    }
    steps
}

fn encode(model_path: &str, prompts: &[&str]) -> Vec<Vec<u32>> {
    let tokenizer = common::load_tokenizer(model_path);
    prompts
        .iter()
        .map(|p| tokenizer.encode(p, false).expect("encode failed"))
        .collect()
}

#[test]
fn eagle3_greedy_matches_plain_greedy() {
    common::harness::init_capture_logging();
    let Some((model_path, draft_path)) = paths_or_skip() else {
        return;
    };
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tokenizer = common::load_tokenizer(&model_path);
    let encoded = encode(&model_path, &PROMPTS);
    let cases: Vec<_> = encoded
        .iter()
        .map(|t| (t.clone(), GENERATED_TOKENS))
        .collect();
    let base = baseline(&model_path, &cases);

    let engine = launch(&model_path, Some(&draft_path));
    let before = counters(&engine);
    let specs: Vec<Vec<Step>> = encoded
        .iter()
        .map(|t| generate(&engine, t.clone(), 0, GENERATED_TOKENS))
        .collect();
    let (proposed, accepted) = report("greedy", &before, &counters(&engine));
    assert!(
        proposed > 0 && accepted > 0,
        "EAGLE-3 never drafted or never had a draft accepted"
    );
    let failures: Vec<String> = PROMPTS
        .iter()
        .enumerate()
        .filter_map(|(i, prompt)| {
            check_lossless(
                &engine,
                &tokenizer,
                i,
                prompt,
                &encoded[i],
                &base[i],
                &specs[i],
            )
            .err()
        })
        .collect();
    assert!(
        failures.is_empty(),
        "EAGLE-3 greedy decode is not lossless:\n{}",
        failures.join("\n")
    );
}

/// The scheduler clamps each verify span to the request's remaining output
/// budget, so short budgets end a request on spans of every length from one
/// token up to the full chain, and the draft rebuild runs on each.
#[test]
fn eagle3_output_budget_edges_match_plain_greedy() {
    let Some((model_path, draft_path)) = paths_or_skip() else {
        return;
    };
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tokenizer = common::load_tokenizer(&model_path);
    let prompt = PROMPTS[1];
    let tokens = encode(&model_path, &[prompt]).remove(0);
    let budgets = [1usize, 2, 3, 4, 5, 6, 9];
    let cases: Vec<_> = budgets.iter().map(|&b| (tokens.clone(), b)).collect();
    let base = baseline(&model_path, &cases);

    let engine = launch(&model_path, Some(&draft_path));
    let before = counters(&engine);
    let specs: Vec<Vec<Step>> = budgets
        .iter()
        .map(|&b| generate(&engine, tokens.clone(), 0, b))
        .collect();
    let (proposed, _) = report("budget edges", &before, &counters(&engine));
    assert!(proposed > 0, "EAGLE-3 never drafted");
    for (i, (base, spec)) in base.iter().zip(&specs).enumerate() {
        check_lossless(&engine, &tokenizer, i, prompt, &tokens, base, spec)
            .unwrap_or_else(|failure| panic!("budget {}: {failure}", budgets[i]));
    }
}

/// A request aborted while it is drafting is retired, and the next request
/// drafts and stays lossless.
#[test]
fn eagle3_abort_mid_speculation_keeps_the_engine_lossless() {
    let Some((model_path, draft_path)) = paths_or_skip() else {
        return;
    };
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tokenizer = common::load_tokenizer(&model_path);
    let encoded = encode(&model_path, &PROMPTS[..2]);
    let base = baseline(&model_path, &[(encoded[1].clone(), GENERATED_TOKENS)]);

    let engine = launch(&model_path, Some(&draft_path));
    let start = counters(&engine);
    let victim_params = SamplingParams {
        ignore_eos: true,
        ..SamplingParams::default()
    };
    let mut victim = engine.submit(request(encoded[0].clone(), victim_params, VICTIM_TOKENS));
    take_tokens(&mut victim, 16);
    let (proposed, _) = report("victim", &start, &counters(&engine));
    assert!(proposed > 0, "the victim never drafted before the abort");
    victim.control.abort();
    drop(victim);
    let deadline = Instant::now() + RETIRE_TIMEOUT;
    while engine.metrics().num_running_reqs > 0 {
        assert!(
            Instant::now() < deadline,
            "the aborted request was never retired"
        );
        std::thread::sleep(Duration::from_millis(5));
    }

    let before = counters(&engine);
    let spec = generate(&engine, encoded[1].clone(), 0, GENERATED_TOKENS);
    let (proposed, accepted) = report("after abort", &before, &counters(&engine));
    assert!(
        proposed > 0 && accepted > 0,
        "EAGLE-3 stopped drafting after an abort"
    );
    check_lossless(
        &engine,
        &tokenizer,
        1,
        PROMPTS[1],
        &encoded[1],
        &base[0],
        &spec,
    )
    .unwrap_or_else(|failure| panic!("after abort: {failure}"));
}

/// A second request joining a drafting one exceeds the one-request draft
/// round: the batch decodes plainly, neither request drafts again, and both
/// outputs match plain greedy.
#[test]
fn eagle3_overlapping_requests_fall_back_and_stay_lossless() {
    let Some((model_path, draft_path)) = paths_or_skip() else {
        return;
    };
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tokenizer = common::load_tokenizer(&model_path);
    let encoded = encode(&model_path, &PROMPTS[2..4]);
    let cases: Vec<_> = encoded
        .iter()
        .map(|t| (t.clone(), GENERATED_TOKENS))
        .collect();
    let base = baseline(&model_path, &cases);

    let engine = launch(&model_path, Some(&draft_path));
    let start = counters(&engine);
    let mut first = engine.submit(request(
        encoded[0].clone(),
        SamplingParams::default(),
        GENERATED_TOKENS,
    ));
    let mut first_steps = take_tokens(&mut first, 8);
    let (proposed, _) = report("alone", &start, &counters(&engine));
    assert!(proposed > 0, "the first request never drafted alone");

    let mut second = engine.submit(request(
        encoded[1].clone(),
        SamplingParams::default(),
        GENERATED_TOKENS,
    ));
    let mut second_steps = take_tokens(&mut second, 1);
    // One snapshot for both checks, so no step can land between them.
    let snapshot = engine.metrics();
    assert_eq!(
        snapshot.num_running_reqs, 2,
        "the two requests never overlapped"
    );
    let overlap = snapshot
        .spec_decode
        .expect("EAGLE-3 engine publishes speculative counters");

    first_steps.extend(to_steps(first.expect_finished()));
    second_steps.extend(to_steps(second.expect_finished()));
    let specs = [first_steps, second_steps];
    let (proposed, _) = report("after overlap", &overlap, &counters(&engine));
    assert_eq!(proposed, 0, "a request drafted after the overlap");
    for (i, (base, spec)) in base.iter().zip(&specs).enumerate() {
        check_lossless(
            &engine,
            &tokenizer,
            i,
            PROMPTS[2 + i],
            &encoded[i],
            base,
            spec,
        )
        .unwrap_or_else(|failure| panic!("concurrent: {failure}"));
    }
}
