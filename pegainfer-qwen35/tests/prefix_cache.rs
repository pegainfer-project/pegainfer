//! Qwen3.5 joint full-attention KV and recurrent/conv prefix-cache gate.
//!
//! The first request publishes the 256-token boundary. The second identical
//! request must restore both state families at that boundary and report the
//! joint hit through `RequestUpdate::cached_tokens`.

use std::path::Path;

use pegainfer_frontend::engine::FinishReason;
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

fn submit(
    handle: &EngineHarness,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    logprobs: usize,
) -> RequestGuard {
    let mut request = common::request(
        prompt_tokens,
        SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        },
        max_tokens,
    );
    request.logprobs = (logprobs > 0).then_some(logprobs);
    handle.submit(request)
}

fn generate(
    handle: &mut EngineHarness,
    prompt_tokens: Vec<u32>,
    max_tokens: usize,
    logprobs: usize,
) -> Generation {
    let control = submit(handle, prompt_tokens, max_tokens, logprobs);
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
    let prompt = text.repeat(80);
    let mut tokens = tokenizer.encode(&prompt, false).expect("encode failed");
    assert!(
        tokens.len() >= token_len,
        "test fixture encoded to only {} tokens",
        tokens.len()
    );
    tokens.truncate(token_len);
    tokens
}

fn assert_trace_close(label: &str, cold: &Generation, warm: &Generation) {
    assert_eq!(
        cold.tokens, warm.tokens,
        "{label}: generated token trace changed"
    );
    assert_eq!(cold.logprobs.len(), warm.logprobs.len());
    let mut deltas = Vec::new();
    for (position, (cold_lp, warm_lp)) in cold.logprobs.iter().zip(&warm.logprobs).enumerate() {
        let cold_lp = cold_lp
            .as_ref()
            .unwrap_or_else(|| panic!("{label}: cold position {position} has no logprob"));
        let warm_lp = warm_lp
            .as_ref()
            .unwrap_or_else(|| panic!("{label}: warm position {position} has no logprob"));
        let cold_map: std::collections::HashMap<u32, f32> =
            cold_lp.top_logprobs.iter().copied().collect();
        for &(token, warm_value) in warm_lp.top_logprobs.iter().take(8) {
            if let Some(cold_value) = cold_map.get(&token) {
                deltas.push((warm_value - cold_value).abs());
            }
        }
    }
    assert!(!deltas.is_empty(), "{label}: no top-logprob overlap");
    deltas.sort_by(f32::total_cmp);
    let mean = deltas.iter().sum::<f32>() / deltas.len() as f32;
    let p99 = deltas[((deltas.len() as f64 * 0.99) as usize).min(deltas.len() - 1)];
    eprintln!(
        "{label}: {} logprob deltas, mean {mean:.4}, p99 {p99:.4}",
        deltas.len()
    );
    assert!(mean <= 0.06, "{label}: mean logprob delta {mean:.4} > 0.06");
    assert!(p99 <= 0.20, "{label}: p99 logprob delta {p99:.4} > 0.20");
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

    let cold = generate(&mut handle, long_prompt.clone(), TRACE_TOKENS, TOP_LOGPROBS);
    assert_eq!(cold.cached_tokens, 0);
    let warm = generate(&mut handle, long_prompt.clone(), TRACE_TOKENS, TOP_LOGPROBS);
    assert_eq!(warm.cached_tokens, 512);
    assert_trace_close("tp1 576-token restore", &cold, &warm);

    let exact_512 = generate(&mut handle, long_prompt[..512].to_vec(), 1, 0);
    assert_eq!(
        exact_512.cached_tokens, 256,
        "an exactly aligned prompt must retain one token for final prefill"
    );
    let exact_256 = generate(&mut handle, long_prompt[..256].to_vec(), 1, 0);
    assert_eq!(exact_256.cached_tokens, 0);

    let extended = generate(&mut handle, long_prompt[..320].to_vec(), 1, 0);
    assert_eq!(extended.cached_tokens, 256);
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
    assert_trace_close("restore during live decode", &mixed_cold, &warm);
    let again = generate(&mut handle, prompt, TRACE_TOKENS, TOP_LOGPROBS);
    assert_eq!(again.cached_tokens, 512);
    assert_trace_close("restore after cancellation", &cold, &again);
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
