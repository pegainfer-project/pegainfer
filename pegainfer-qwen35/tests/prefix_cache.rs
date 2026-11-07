//! Qwen3.5 joint full-attention KV and recurrent/conv prefix-cache gate.
//!
//! The first request publishes the 256-token boundary. The second identical
//! request must restore both state families at that boundary and report the
//! joint hit through `RequestUpdate::cached_tokens`.

use std::path::Path;

use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen35::Qwen35LaunchOptions;
use pegainfer_qwen35::Qwen35SchedulerPolicy;

mod common;

use common::EngineHarness;
use common::RequestGuard;

const PREFIX_BOUNDARY: usize = 256;
const PROMPT_TOKENS: usize = 320;
const TRACE_TOKENS: usize = 8;
const TOP_LOGPROBS: usize = 16;
const QWEN35_4B_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");
// Qwen3.5-4B uses 49.125 MiB per snapshot, so this is exactly two slots.
const PREFIX_CACHE_MIB: usize = 128;

fn model_path_or_skip() -> Option<String> {
    common::model_path_or_skip("prefix_cache")
}

fn start_engine(model_path: &str, tp_size: usize, prefix_cache_mib: usize) -> EngineHarness {
    start_engine_with_graph(model_path, tp_size, prefix_cache_mib, tp_size == 1)
}

fn start_engine_with_graph(
    model_path: &str,
    tp_size: usize,
    prefix_cache_mib: usize,
    cuda_graph: bool,
) -> EngineHarness {
    let handle = pegainfer_qwen35::launch_with_options_policy_and_overlap(
        Path::new(model_path),
        Qwen35LaunchOptions::new(0, tp_size, cuda_graph, 2, 1024, prefix_cache_mib),
        Qwen35SchedulerPolicy::Off,
        pegainfer_qwen35::Qwen35DecodeOverlap::Off,
    )
    .unwrap_or_else(|err| panic!("failed to start Qwen3.5 TP{tp_size} prefix-cache engine: {err}"));
    EngineHarness::new(handle)
}

struct Generation {
    cached_tokens: usize,
    tokens: Vec<u32>,
    logprobs: Vec<Option<TokenLogprob>>,
}

fn request(prompt_tokens: Vec<u32>, max_tokens: usize, logprobs: usize) -> Request {
    let mut request = common::request(
        prompt_tokens,
        SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        },
        max_tokens,
    );
    request.logprobs = (logprobs > 0).then_some(logprobs);
    request
}

fn submit(
    handle: &EngineHarness,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    logprobs: usize,
) -> RequestGuard {
    handle.submit(request(prompt_tokens, max_tokens, logprobs))
}

fn generate(
    handle: &mut EngineHarness,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    logprobs: usize,
) -> Generation {
    run(handle, request(prompt_tokens, max_tokens, logprobs))
}

fn run(handle: &mut EngineHarness, request: Request) -> Generation {
    let max_tokens = request.max_tokens;
    let control = handle.submit(request);
    let mut cached_tokens = None;
    let mut generated_tokens = Vec::with_capacity(max_tokens);
    let mut generated_logprobs = Vec::with_capacity(max_tokens);
    loop {
        let update = handle.next(control.id());
        if let Some(hit) = update.cached_tokens {
            cached_tokens = Some(hit);
        }
        generated_tokens.extend(update.tokens);
        generated_logprobs.extend(update.logprobs);
        if let Some(terminal) = update.terminal {
            match terminal {
                Terminal::Finished {
                    reason,
                    completion_tokens,
                    ..
                } => {
                    assert_eq!(reason, FinishReason::Length);
                    assert_eq!(completion_tokens, max_tokens);
                    return Generation {
                        cached_tokens: cached_tokens.expect("request did not report cached_tokens"),
                        tokens: generated_tokens,
                        logprobs: generated_logprobs,
                    };
                }
                terminal => panic!("generation did not finish: {terminal:?}"),
            }
        }
    }
}

fn generate_one(handle: &mut EngineHarness, prompt_tokens: Vec<u32>) -> (usize, u32) {
    let result = generate(handle, prompt_tokens, 1, 0);
    (
        result.cached_tokens,
        *result.tokens.first().expect("request emitted no token"),
    )
}

fn prompt_tokens(
    tokenizer: &vllm_text::tokenizer::DynTokenizer,
    text: &str,
    token_len: usize,
) -> Vec<u32> {
    // Fixture sentences encode to at least 8 tokens, so this covers long prompts.
    let prompt = text.repeat(80.max(token_len.div_ceil(8)));
    let mut tokens = tokenizer.encode(&prompt, false).expect("encode failed");
    assert!(
        tokens.len() >= token_len,
        "test fixture encoded to only {} tokens",
        tokens.len()
    );
    tokens.truncate(token_len);
    tokens
}

/// Engines in one process share their timed decode GEMM winners (`g_lt_tuned_algos`).
fn assert_trace_exact(label: &str, cold: &Generation, warm: &Generation) {
    assert_eq!(
        cold.tokens, warm.tokens,
        "{label}: generated token trace changed"
    );
    // Missing logprobs on both sides would compare equal, so require one per token.
    for (side, trace) in [("cold", cold), ("warm", warm)] {
        assert_eq!(
            trace.logprobs.len(),
            trace.tokens.len(),
            "{label}: {side} trace needs one logprob per token"
        );
        if let Some(i) = trace.logprobs.iter().position(Option::is_none) {
            panic!("{label}: {side} position {i} has no logprob");
        }
    }
    let diverged = cold
        .logprobs
        .iter()
        .zip(&warm.logprobs)
        .position(|(cold_lp, warm_lp)| cold_lp != warm_lp);
    if let Some(i) = diverged {
        panic!(
            "{label}: logprobs diverge at position {i}: {:?} vs {:?}",
            cold.logprobs[i], warm.logprobs[i]
        );
    }
}

#[test]
fn joint_restore_and_unpinned_lru_eviction_preserve_output() {
    if !Path::new(QWEN35_4B_MODEL_PATH)
        .join("config.json")
        .is_file()
    {
        eprintln!(
            "skipping Qwen3.5-4B two-slot LRU test: {QWEN35_4B_MODEL_PATH}/config.json is missing"
        );
        return;
    }
    let model_path = QWEN35_4B_MODEL_PATH;
    let tokenizer = common::load_tokenizer(model_path);
    let prompt_a = prompt_tokens(
        &tokenizer,
        "Alpha prefix exercises full-attention KV plus every recurrent and convolution state. ",
        PROMPT_TOKENS,
    );
    let prompt_b = prompt_tokens(
        &tokenizer,
        "Beta prefix is deliberately distinct and occupies a second recurrent snapshot slot. ",
        PROMPT_TOKENS,
    );
    let prompt_c = prompt_tokens(
        &tokenizer,
        "Gamma prefix creates pressure and must evict the least-recently-used unpinned snapshot. ",
        PROMPT_TOKENS,
    );

    let mut handle = start_engine(model_path, 1, PREFIX_CACHE_MIB);
    let (cold_cached, cold_token) = generate_one(&mut handle, prompt_a.clone());
    assert_eq!(cold_cached, 0, "first request must be cold");

    let (warm_cached, warm_token) = generate_one(&mut handle, prompt_a.clone());
    assert_eq!(
        warm_cached, PREFIX_BOUNDARY,
        "the longest jointly committed boundary should be restored"
    );
    assert_eq!(
        warm_token, cold_token,
        "joint restore must preserve greedy output"
    );

    let (beta_cold_cached, beta_token) = generate_one(&mut handle, prompt_b.clone());
    assert_eq!(beta_cold_cached, 0, "new beta prefix must be cold");

    let (alpha_touched_cached, _) = generate_one(&mut handle, prompt_a);
    assert_eq!(
        alpha_touched_cached, PREFIX_BOUNDARY,
        "alpha lookup must refresh its snapshot LRU position"
    );

    let (gamma_cold_cached, _) = generate_one(&mut handle, prompt_c);
    assert_eq!(
        gamma_cold_cached, 0,
        "new gamma prefix must insert under snapshot pressure"
    );

    let (beta_after_eviction_cached, beta_after_eviction_token) =
        generate_one(&mut handle, prompt_b);
    assert_eq!(
        beta_after_eviction_cached, 0,
        "beta KV may remain resident, but its evicted snapshot must force a joint miss"
    );
    assert_eq!(
        beta_after_eviction_token, beta_token,
        "snapshot pressure may change hit rate but must not change output"
    );
}

#[test]
fn boundary_selection_and_multitoken_restore_preserve_logits() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let tokenizer = common::load_tokenizer(&model_path);
    let long_prompt = prompt_tokens(
        &tokenizer,
        "Boundary coverage checks exact alignment, prefix extension, and joint recurrent state restore. ",
        576,
    );
    let mut handle = start_engine(&model_path, 1, 512);

    // The cold run stops only at its last boundary, 512, so both runs cut there.
    let cold = generate(&mut handle, long_prompt.clone(), TRACE_TOKENS, TOP_LOGPROBS);
    assert_eq!(cold.cached_tokens, 0);
    let warm = generate(&mut handle, long_prompt.clone(), TRACE_TOKENS, TOP_LOGPROBS);
    assert_eq!(warm.cached_tokens, 512);
    assert_trace_exact("tp1 576-token restore", &cold, &warm);

    let exact_512 = generate(&mut handle, long_prompt[..512].to_vec(), 1, 0);
    assert_eq!(
        exact_512.cached_tokens, 0,
        "an exactly aligned prompt must retain one token, so the 512 snapshot cannot serve it"
    );
    // This prompt ends on a boundary and publishes it for the extension below.
    let exact_256 = generate(&mut handle, long_prompt[..256].to_vec(), 1, 0);
    assert_eq!(exact_256.cached_tokens, 0);

    let extended = generate(&mut handle, long_prompt[..320].to_vec(), 1, 0);
    assert_eq!(extended.cached_tokens, 256);
    let longest = generate(&mut handle, long_prompt, 1, 0);
    assert_eq!(
        longest.cached_tokens, 512,
        "the longest joint boundary must win"
    );
}

/// Cold cuts depend only on budget and prompt: this prompt's last boundary is a step
/// end, so the cache off and two cache sizes cut it, and decode it, identically.
#[test]
fn cold_output_ignores_cache_budget() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let tokenizer = common::load_tokenizer(&model_path);
    let prompt = prompt_tokens(
        &tokenizer,
        "Cold prefill cuts depend only on the step budget and on the prompt itself. ",
        4160,
    );
    let cold = |prefix_cache_mib| {
        let mut handle = start_engine(&model_path, 1, prefix_cache_mib);
        let generation = generate(&mut handle, prompt.clone(), TRACE_TOKENS, TOP_LOGPROBS);
        assert_eq!(generation.cached_tokens, 0);
        generation
    };
    let cache_off = cold(0);
    assert_trace_exact("cache off vs 128 MiB", &cache_off, &cold(PREFIX_CACHE_MIB));
    assert_trace_exact("cache off vs 512 MiB", &cache_off, &cold(512));
}

/// A hit on another prompt's stop replays the cold run exactly when its cut history
/// matches; when the hit moves a cut, only its greedy tokens must hold.
#[test]
fn cross_prompt_hit_is_exact_when_cuts_match() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let tokenizer = common::load_tokenizer(&model_path);
    // The donor's last stop is 3840. The probe shares those tokens, but its own cold
    // run cuts at 3072 and 4096, so restoring the donor's snapshot moves one cut.
    let donor = prompt_tokens(
        &tokenizer,
        "A donor prompt publishes its last boundary for a later request to reuse. ",
        4000,
    );
    let mut probe = donor[..3840].to_vec();
    probe.extend(prompt_tokens(
        &tokenizer,
        "The probe continues past the shared prefix with different text. ",
        320,
    ));
    let cold = |history_tokens| {
        let mut handle = start_engine(&model_path, 1, PREFIX_CACHE_MIB);
        let mut cold_request = request(probe.clone(), TRACE_TOKENS, TOP_LOGPROBS);
        cold_request.history_tokens = history_tokens;
        let generation = run(&mut handle, cold_request);
        assert_eq!(generation.cached_tokens, 0);
        generation
    };
    let cold_own = cold(None);
    let cold_stop_3840 = cold(Some(3840));

    let mut handle = start_engine(&model_path, 1, PREFIX_CACHE_MIB);
    assert_eq!(generate(&mut handle, donor, 1, 0).cached_tokens, 0);
    // Stopping at 3840 too, the probe's hit keeps its cold cuts.
    let mut same_cuts = request(probe.clone(), TRACE_TOKENS, TOP_LOGPROBS);
    same_cuts.history_tokens = Some(3840);
    let same_cuts = run(&mut handle, same_cuts);
    assert_eq!(same_cuts.cached_tokens, 3840);
    assert_trace_exact("cross-prompt hit, same cuts", &cold_stop_3840, &same_cuts);
    // Without the stop, the hit adds a 3840 cut its own cold run lacks.
    let moved_cut = generate(&mut handle, probe, TRACE_TOKENS, TOP_LOGPROBS);
    assert_eq!(moved_cut.cached_tokens, 3840);
    assert_eq!(
        moved_cut.tokens, cold_own.tokens,
        "cross-prompt hit, moved cut: generated token trace changed"
    );
}

/// A warm hinted turn must not resume past its history stop, even when a longer
/// snapshot of the same tokens exists, or the next turn loses that boundary.
#[test]
fn warm_hinted_turn_still_stops_at_history_boundary() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let tokenizer = common::load_tokenizer(&model_path);
    let mut handle = start_engine(&model_path, 1, PREFIX_CACHE_MIB);
    let turn = prompt_tokens(
        &tokenizer,
        "A warm chat turn first arrives without the frontend's history hint. ",
        4100,
    );
    // Unhinted, the turn's step ends include 3072 and 4096 but not 3840.
    assert_eq!(generate(&mut handle, turn.clone(), 1, 0).cached_tokens, 0);
    let mut hinted = request(turn.clone(), 1, 0);
    hinted.history_tokens = Some(4094);
    assert_eq!(
        run(&mut handle, hinted).cached_tokens,
        3072,
        "a hinted lookup must not resume past its 3840 stop"
    );
    let mut next_turn = turn[..4094].to_vec();
    next_turn.extend(prompt_tokens(
        &tokenizer,
        "The user follows up with another question. ",
        200,
    ));
    assert_eq!(generate(&mut handle, next_turn, 1, 0).cached_tokens, 3840);
}

fn restore_during_live_decode(
    tp_size: usize,
    cuda_graph: bool,
    overlap: pegainfer_qwen35::Qwen35DecodeOverlap,
) {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let tokenizer = common::load_tokenizer(&model_path);
    let prompt = prompt_tokens(
        &tokenizer,
        "Joint prefix restore must preserve logits while another request decodes. ",
        576,
    );
    let start_engine = || {
        pegainfer_qwen35::launch_with_options_policy_and_overlap(
            Path::new(&model_path),
            Qwen35LaunchOptions::new(0, tp_size, cuda_graph, 2, 1024, 128),
            Qwen35SchedulerPolicy::Off,
            overlap,
        )
        .map(EngineHarness::new)
        .unwrap()
    };
    let generate_with_background = |handle: &mut EngineHarness| {
        let background = submit(handle, vec![9707], 1024, 0);
        let mut background_tokens = loop {
            let update = handle.next(background.id());
            assert!(
                update.terminal.is_none(),
                "background decode ended before the probe"
            );
            if !update.tokens.is_empty() {
                break update.tokens.len();
            }
        };
        let generation = generate(handle, prompt.clone(), TRACE_TOKENS, TOP_LOGPROBS);
        while let Some(update) = handle.try_next(background.id()) {
            background_tokens += update.tokens.len();
            assert!(
                update.terminal.is_none(),
                "background finished before the mixed-load probe"
            );
        }
        assert!(background_tokens < 1024);
        background.abort();
        generation
    };

    // Keep a cold reference under the same background decode as the warm request.
    let mixed_cold = {
        let mut reference = start_engine();
        let generation = generate_with_background(&mut reference);
        assert_eq!(generation.cached_tokens, 0);
        generation
    };
    let mut handle = start_engine();
    let cold = generate(&mut handle, prompt.clone(), TRACE_TOKENS, TOP_LOGPROBS);
    assert_eq!(cold.cached_tokens, 0);
    assert_eq!(mixed_cold.tokens, cold.tokens);
    let warm = generate_with_background(&mut handle);
    assert_eq!(warm.cached_tokens, 512);
    assert_trace_exact("restore during live decode", &mixed_cold, &warm);
    let again = generate(&mut handle, prompt, TRACE_TOKENS, TOP_LOGPROBS);
    assert_eq!(again.cached_tokens, 512);
    assert_trace_exact("restore after cancellation", &cold, &again);
}

#[test]
fn async_prefill_prefix_restore_during_live_decode() {
    restore_during_live_decode(1, true, pegainfer_qwen35::Qwen35DecodeOverlap::SharedSm);
}

#[test]
#[ignore = "requires two CUDA devices and Qwen3.5 weights"]
fn tp2_prefix_restore_during_mixed_step() {
    restore_during_live_decode(2, false, pegainfer_qwen35::Qwen35DecodeOverlap::Off);
}

#[test]
#[ignore = "requires two CUDA devices and Qwen3.5 weights"]
fn tp2_graph_prefix_restore_during_mixed_step() {
    restore_during_live_decode(2, true, pegainfer_qwen35::Qwen35DecodeOverlap::Off);
}
