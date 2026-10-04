//! E2E scheduler integration test for Qwen3.5-4B.
//!
//! Tests the Qwen3.5 reduced-capacity scheduler path (batch prefill +
//! CUDA Graph decode) with sequential, concurrent, and cancelled requests.
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use common::EngineHarness;
use common::RequestGuard;
use log::info;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::RequestUpdate;
use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::sampler::SamplingParams;
use vllm_text::tokenizer::DynTokenizer;

use crate::test_fixture as common;
use crate::test_fixture::GdnAcceptance;

const CASES: &[TestCase] = &[
    TestCase {
        name: "tell_story",
        prompt: "Tell me a story",
        max_new_tokens: 50,
    },
    TestCase {
        name: "my_name",
        prompt: "My name is",
        max_new_tokens: 50,
    },
    TestCase {
        name: "math",
        prompt: "What is 2 + 2?",
        max_new_tokens: 30,
    },
    TestCase {
        name: "chinese_weather",
        prompt: "The weather is nice today",
        max_new_tokens: 50,
    },
    TestCase {
        name: "chinese_capital",
        prompt: "Introduce the capital city of China",
        max_new_tokens: 50,
    },
    TestCase {
        name: "python_code",
        prompt: "Write a Python function to reverse a string",
        max_new_tokens: 50,
    },
    TestCase {
        name: "kanye_album",
        prompt: "My favorite Kanye West album is",
        max_new_tokens: 50,
    },
    TestCase {
        name: "coldplay_ghost",
        prompt: "Coldplay's Ghost Stories album feels",
        max_new_tokens: 50,
    },
    TestCase {
        name: "oyster_riddle",
        prompt: "An oyster cooked in a pan becomes",
        max_new_tokens: 50,
    },
    TestCase {
        name: "monkey_king_lake",
        prompt: "A clever monkey jumps into a lake and returns as",
        max_new_tokens: 50,
    },
];

fn max_position_embeddings(model_path: &str) -> usize {
    let config_path = std::path::Path::new(model_path).join("config.json");
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&config_path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", config_path.display())),
    )
    .expect("config.json must be valid JSON");
    config
        .pointer("/text_config/max_position_embeddings")
        .or_else(|| config.pointer("/max_position_embeddings"))
        .and_then(serde_json::Value::as_u64)
        .expect("Qwen3.5 config must expose max_position_embeddings") as usize
}

struct TestCase {
    name: &'static str,
    prompt: &'static str,
    max_new_tokens: usize,
}

struct GenerationResult {
    tokens: Vec<u32>,
    logprobs: Vec<Option<TokenLogprob>>,
    finish_reason: FinishReason,
}

fn generate_tokens(
    handle: &mut EngineHarness,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
) -> (Vec<u32>, FinishReason) {
    let result = generate_tokens_with_logprobs(handle, tokenizer, prompt, max_tokens, None);
    (result.tokens, result.finish_reason)
}

fn generate_tokens_with_logprobs(
    handle: &mut EngineHarness,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
    logprobs: Option<usize>,
) -> GenerationResult {
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
    let mut request = common::request(prompt_tokens, SamplingParams::default(), max_tokens);
    request.logprobs = logprobs;
    let control = handle.submit(request);
    collect_generation(handle, &control, prompt, logprobs)
}

fn submit_repeated_token_request(
    handle: &EngineHarness,
    request_id: &str,
    token: u32,
    prompt_len: usize,
    max_tokens: usize,
) -> RequestGuard {
    let mut request = common::request(
        vec![token; prompt_len],
        SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        },
        max_tokens,
    );
    request.client_label = Some(request_id.into());
    handle.submit(request)
}

fn wait_for_first_token(
    handle: &mut EngineHarness,
    control: &RequestGuard,
    request_id: &str,
) -> Vec<u32> {
    let deadline = Instant::now() + std::time::Duration::from_secs(30);
    loop {
        let update = recv_event_before(handle, control, request_id, deadline);
        assert!(
            update.terminal.is_none(),
            "{request_id} ended before its first token"
        );
        if !update.tokens.is_empty() {
            return update.tokens;
        }
    }
}

fn recv_event_before(
    handle: &mut EngineHarness,
    control: &RequestGuard,
    request_id: &str,
    deadline: Instant,
) -> RequestUpdate {
    loop {
        if let Some(update) = handle.try_next(control.id()) {
            return update;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {request_id} scheduler event"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn assert_no_generated_event(handle: &mut EngineHarness, control: &RequestGuard, request_id: &str) {
    while let Some(update) = handle.try_next(control.id()) {
        assert!(
            update.tokens.is_empty() && update.terminal.is_none(),
            "{request_id} emitted {update:?} before the overlap bound"
        );
    }
}

fn drain_tokens(handle: &mut EngineHarness, control: &RequestGuard, request_id: &str) -> Vec<u32> {
    let mut tokens = Vec::new();
    while let Some(update) = handle.try_next(control.id()) {
        assert!(
            update.terminal.is_none(),
            "{request_id} ended while it must remain active"
        );
        tokens.extend(update.tokens);
    }
    tokens
}

fn wait_for_running_requests(handle: &EngineHarness, expected: u64, timeout: std::time::Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        let snapshot = handle.metrics();
        if snapshot.num_running_reqs == expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {expected} running requests; last snapshot: {snapshot:?}"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn collect_generation(
    handle: &mut EngineHarness,
    control: &RequestGuard,
    name: &str,
    logprobs: Option<usize>,
) -> GenerationResult {
    collect_generation_until(handle, control, name, logprobs, None)
}

fn collect_generation_with_timeout(
    handle: &mut EngineHarness,
    control: &RequestGuard,
    name: &str,
    logprobs: Option<usize>,
    timeout: std::time::Duration,
) -> GenerationResult {
    collect_generation_until(
        handle,
        control,
        name,
        logprobs,
        Some(Instant::now() + timeout),
    )
}

fn collect_generation_until(
    handle: &mut EngineHarness,
    control: &RequestGuard,
    name: &str,
    logprobs: Option<usize>,
    deadline: Option<Instant>,
) -> GenerationResult {
    let mut tokens = Vec::new();
    let mut token_logprobs = Vec::new();
    loop {
        let update = match deadline {
            Some(deadline) => recv_event_before(handle, control, name, deadline),
            None => handle.next(control.id()),
        };
        for (&id, logprob) in update.tokens.iter().zip(&update.logprobs) {
            if let Some(count) = logprobs {
                let lp = logprob
                    .as_ref()
                    .unwrap_or_else(|| panic!("{name}: logprobs={count} returned None"));
                assert!(
                    lp.logprob.is_finite(),
                    "{name}: sampled token logprob must be finite"
                );
                assert_eq!(
                    lp.top_logprobs.len(),
                    count,
                    "{name}: top_logprobs length should match the request"
                );
                assert!(
                    lp.top_logprobs.iter().all(|&(_, v)| v.is_finite()),
                    "{name}: top_logprobs must be finite"
                );
                if count > 0 {
                    assert_eq!(
                        lp.top_logprobs.first().map(|&(token, _)| token),
                        Some(id),
                        "{name}: greedy sampled token should match top-1 logprob row"
                    );
                }
            } else {
                assert!(
                    logprob.is_none(),
                    "{name}: disabled logprobs should return None"
                );
            }
        }
        tokens.extend(update.tokens);
        token_logprobs.extend(update.logprobs);
        if let Some(terminal) = update.terminal {
            match terminal {
                Terminal::Finished { reason, .. } => {
                    return GenerationResult {
                        tokens,
                        logprobs: token_logprobs,
                        finish_reason: reason,
                    };
                }
                terminal => panic!("{name}: generation did not finish: {terminal:?}"),
            }
        }
    }
}

fn concurrent_params(case_idx: usize) -> SamplingParams {
    if case_idx.is_multiple_of(2) {
        SamplingParams::default()
    } else {
        SamplingParams {
            temperature: 0.9,
            top_k: 32,
            top_p: 0.9,
            ..SamplingParams::default()
        }
    }
}

fn expect_context_window_rejection(handle: &mut EngineHarness, max_context_tokens: usize) {
    let mut request = common::request(vec![1; max_context_tokens], SamplingParams::default(), 1);
    request.client_label = Some("over-context-window".into());
    let control = handle.submit(request);
    let update = handle.next(control.id());
    match update.terminal {
        Some(Terminal::Rejected {
            reason,
            prompt_tokens,
        }) => {
            assert_eq!(prompt_tokens, max_context_tokens);
            assert_eq!(
                reason,
                RejectReason::ContextLength {
                    prompt_tokens: max_context_tokens,
                    max_tokens: 1,
                    limit: max_context_tokens,
                }
            );
        }
        terminal => panic!("expected context-window rejection, got: {terminal:?}"),
    }
}

/// Token-loop collapse of one completion (the Qwen3.5-9B untied-lm_head
/// symptom): distinct-token ratio, longest same-token run, and an exact
/// repeated-tail period each catch a different loop shape.
struct Collapse {
    distinct_ratio: f64,
    max_run: usize,
    tail_period: Option<usize>,
    len: usize,
}

impl Collapse {
    fn measure(tokens: &[u32]) -> Self {
        let distinct: HashSet<u32> = tokens.iter().copied().collect();
        let mut max_run = 0usize;
        let mut run = 0usize;
        let mut prev = None;
        for &t in tokens {
            run = if prev == Some(t) { run + 1 } else { 1 };
            max_run = max_run.max(run);
            prev = Some(t);
        }
        // Periods 1-2 already trip max_run / distinct_ratio; starting at 3 keeps
        // benign short echoes ("no, no") out of this check.
        let tail_period = (3..=tokens.len() / 2).find(|&p| {
            tokens[tokens.len() - 2 * p..tokens.len() - p] == tokens[tokens.len() - p..]
        });
        Self {
            // An empty completion (immediate EOS) is a valid stop, not a loop.
            distinct_ratio: if tokens.is_empty() {
                1.0
            } else {
                distinct.len() as f64 / tokens.len() as f64
            },
            max_run,
            tail_period,
            len: tokens.len(),
        }
    }

    fn is_degenerate(&self) -> bool {
        const DISTINCT_RATIO_FLOOR: f64 = 0.25;
        const MAX_RUN_CEILING: usize = 8;
        self.distinct_ratio < DISTINCT_RATIO_FLOOR
            || self.max_run >= MAX_RUN_CEILING
            || self.tail_period.is_some()
    }
}

fn assert_no_model_wide_collapse(collapses: &[(&str, Collapse)]) {
    let degenerate = collapses.iter().filter(|(_, c)| c.is_degenerate()).count();
    if degenerate * 2 >= collapses.len() {
        for (name, c) in collapses {
            eprintln!(
                "{}  {name} len={} distinct_ratio={:.3} max_run={} tail_period={:?}",
                if c.is_degenerate() {
                    "DEGENERATE"
                } else {
                    "ok        "
                },
                c.len,
                c.distinct_ratio,
                c.max_run,
                c.tail_period,
            );
        }
        panic!(
            "{degenerate}/{} sequential completions are degenerate — model-wide broken generation",
            collapses.len()
        );
    }
}

fn run_full_scheduler_e2e(
    handle: &mut EngineHarness,
    tokenizer: &DynTokenizer,
    max_context_tokens: usize,
    label: &str,
) {
    // logging intentionally left to the test harness

    // ── 0. Static context-window rejection ─────────────────────────────
    info!("=== Phase 0: Context-window rejection ===");
    expect_context_window_rejection(handle, max_context_tokens);
    info!("  PASS: over-context request rejected before prefill");

    // ── 1. logprobs must not change greedy tokens ─────────────────────
    info!("=== Phase 1: logprobs/no-logprobs token parity ===");
    for case in CASES.iter().take(3) {
        let max_tokens = case.max_new_tokens.min(16);
        let no_logprobs =
            generate_tokens_with_logprobs(handle, tokenizer, case.prompt, max_tokens, None);
        let with_logprobs =
            generate_tokens_with_logprobs(handle, tokenizer, case.prompt, max_tokens, Some(1));
        assert_eq!(no_logprobs.finish_reason, with_logprobs.finish_reason);
        assert_eq!(
            no_logprobs.tokens, with_logprobs.tokens,
            "greedy token ids must not depend on whether logprobs are requested for {:?}",
            case.name
        );
        assert!(
            no_logprobs.logprobs.iter().all(Option::is_none),
            "logprobs=None should keep the no-host-logprobs path for {:?}",
            case.name
        );
        assert!(
            with_logprobs.logprobs.iter().all(Option::is_some),
            "logprobs=1 should attach token logprobs for {:?}",
            case.name
        );
        assert!(
            !no_logprobs.tokens.is_empty(),
            "logprobs parity regression prompt {:?} produced no tokens",
            case.name
        );
        info!(
            "  PASS: {:?} logprobs=None and logprobs=Some(1) produced identical greedy tokens",
            case.name
        );
    }

    // ── 2. Sequential scheduler requests ────────────────────────────────
    info!("=== Phase 2: Qwen3.5 sequential scheduler requests ===");
    let mut collapses = Vec::new();
    for case in CASES {
        info!("--- {:?} ---", case.name);
        let start = Instant::now();
        let (tokens, finish_reason) =
            generate_tokens(handle, tokenizer, case.prompt, case.max_new_tokens);
        let elapsed = start.elapsed();

        let text = tokenizer.decode(&tokens, true).expect("decode failed");
        let tok_s = tokens.len() as f64 / elapsed.as_secs_f64();

        info!(
            "  {} tokens in {:.2?} ({:.1} tok/s) finish={:?}",
            tokens.len(),
            elapsed,
            tok_s,
            finish_reason
        );

        assert!(!text.is_empty(), "empty output for: {:?}", case.name);
        if tokens.len() >= case.max_new_tokens {
            assert_eq!(finish_reason, FinishReason::Length);
        }

        collapses.push((case.name, Collapse::measure(&tokens)));
        info!("  PASS: {:?}", case.name);
    }
    assert_no_model_wide_collapse(&collapses);

    // ── 3. Multi-request (scheduler state reuse) ────────────────────────
    info!("=== Phase 3: Multi-request ===");
    for case in CASES {
        let (tokens, _) = generate_tokens(handle, tokenizer, case.prompt, case.max_new_tokens);
        let text = tokenizer.decode(&tokens, true).expect("decode failed");
        assert!(
            !text.is_empty(),
            "empty output on second run for: {:?}",
            case.name
        );
        info!("  PASS: {:?} → {} tokens", case.name, tokens.len());
    }

    // ── 4. Concurrent requests ──────────────────────────────────────────
    info!("=== Phase 4: Concurrent requests ===");
    {
        let mut requests = Vec::new();

        // Submit all cases concurrently, alternating greedy and sampled rows so
        // batch decode covers the mixed token-selection path from #284.
        for (case_idx, case) in CASES.iter().enumerate() {
            let prompt_tokens = tokenizer.encode(case.prompt, false).expect("encode failed");
            let control = handle.submit(common::request(
                prompt_tokens,
                concurrent_params(case_idx),
                case.max_new_tokens,
            ));
            requests.push((case.name, control));
        }

        // Collect all results
        for (name, control) in requests {
            let result = collect_generation(handle, &control, name, None);
            let text = tokenizer
                .decode(&result.tokens, true)
                .expect("decode failed");
            assert!(!text.is_empty(), "empty output for concurrent: {:?}", name);
            info!("  PASS: {:?} → {} tokens", name, result.tokens.len());
        }
    }

    // ── 4b. Mixed concurrent logprobs requests ─────────────────────────
    info!("=== Phase 4b: Mixed concurrent logprobs ===");
    {
        let mixed = [
            ("mixed_no_logprobs", CASES[0].prompt, None),
            ("mixed_chosen_logprob", CASES[1].prompt, Some(0)),
            ("mixed_top_logprobs", CASES[1].prompt, Some(1)),
        ];
        let mut requests = Vec::new();
        for (name, prompt, logprobs) in mixed {
            let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
            let mut request = common::request(prompt_tokens, SamplingParams::default(), 8);
            request.client_label = Some(name.into());
            request.logprobs = logprobs;
            requests.push((name, logprobs, handle.submit(request)));
        }
        for (name, logprobs, control) in requests {
            let result = collect_generation(handle, &control, name, logprobs);
            assert!(!result.tokens.is_empty(), "{name}: produced no tokens");
            info!("  PASS: {name} → {} tokens", result.tokens.len());
        }
    }

    // ── 5. Cancellation safety ─────────────────────────────────────────
    info!("=== Phase 5: Request cancellation ===");
    {
        let prompt_tokens = tokenizer.encode("Hello", false).expect("encode failed");
        let control = handle.submit(common::request(
            prompt_tokens,
            SamplingParams::default(),
            10,
        ));
        control.abort();
    }

    // Verify scheduler survives
    let (tokens, _) = generate_tokens(handle, tokenizer, "Hello", 5);
    let text = tokenizer.decode(&tokens, true).expect("decode failed");
    assert!(!text.is_empty(), "scheduler dead after cancellation");
    info!("  PASS: scheduler survived cancellation");

    info!("All Qwen3.5 scheduler tests passed for {label}!");
}

#[test]
fn test_e2e_qwen35_scheduler() {
    run_scheduler_e2e(&GdnAcceptance::Triton);
}

#[test]
#[ignore = "requires SM120, a validated candidate identity, and Qwen3.5 weights"]
fn candidate_e2e_qwen35_scheduler() {
    run_scheduler_e2e(&GdnAcceptance::candidate().expect("candidate prerequisites"));
}

fn run_scheduler_e2e(acceptance: &GdnAcceptance) {
    let Some(model_path) = acceptance
        .model_path("test_e2e_qwen35_scheduler")
        .expect("model prerequisite")
    else {
        return;
    };

    info!("Loading Qwen3.5 model for scheduler test...");
    let start = Instant::now();
    let tokenizer = common::load_tokenizer(&model_path);
    let overlap = if acceptance.is_candidate() {
        crate::Qwen35DecodeOverlap::SharedSm
    } else {
        crate::Qwen35DecodeOverlap::Off
    };
    let handle = acceptance
        .launch_engine(
            &model_path,
            8,
            crate::DEFAULT_MAX_PREFILL_TOKENS,
            crate::Qwen35SchedulerPolicy::Off,
            overlap,
        )
        .expect("Failed to start Qwen3.5 scheduler");
    let mut handle = EngineHarness::new(handle);
    info!("scheduler loaded in {:.2?}", start.elapsed());

    let max_context_tokens = max_position_embeddings(&model_path);
    run_full_scheduler_e2e(&mut handle, &tokenizer, max_context_tokens, "TP1");
}

#[test]
fn test_e2e_qwen35_shared_sm_last_decoder() {
    run_shared_sm_last_decoder(&GdnAcceptance::Triton);
}

#[test]
#[ignore = "requires SM120, a validated candidate identity, and Qwen3.5 weights"]
fn candidate_e2e_qwen35_shared_sm_last_decoder() {
    run_shared_sm_last_decoder(&GdnAcceptance::candidate().expect("candidate prerequisites"));
}

fn run_shared_sm_last_decoder(acceptance: &GdnAcceptance) {
    pegainfer_core::logging::init_default();
    let Some(model_path) = acceptance
        .model_path("test_e2e_qwen35_shared_sm_last_decoder")
        .expect("model prerequisite")
    else {
        return;
    };
    let tokenizer = common::load_tokenizer(&model_path);
    let seed_token = tokenizer
        .encode("Hello", false)
        .expect("encode failed")
        .into_iter()
        .next()
        .expect("test prompt must contain a token");

    let (off_reference_tokens, off_decoder_tokens) = {
        let off_handle = acceptance
            .launch_engine(
                &model_path,
                4,
                8192,
                crate::Qwen35SchedulerPolicy::Off,
                crate::Qwen35DecodeOverlap::Off,
            )
            .expect("Failed to start Qwen3.5 default-Off scheduler");
        let mut off_handle = EngineHarness::new(off_handle);
        let off_request = submit_repeated_token_request(
            &off_handle,
            "overlap-off-reference",
            seed_token,
            8192,
            2,
        );
        let off = collect_generation_with_timeout(
            &mut off_handle,
            &off_request,
            "overlap-off-reference",
            None,
            std::time::Duration::from_secs(30),
        );
        assert_eq!(
            off.tokens.len(),
            2,
            "default-Off reference request must finish before Shared-SM parity"
        );
        let decoder_request = submit_repeated_token_request(
            &off_handle,
            "overlap-off-decoder-reference",
            seed_token,
            512,
            128,
        );
        let decoder = collect_generation_with_timeout(
            &mut off_handle,
            &decoder_request,
            "overlap-off-decoder-reference",
            None,
            std::time::Duration::from_secs(30),
        );
        assert_eq!(decoder.tokens.len(), 128);
        (off.tokens, decoder.tokens)
    };

    // The auto policy may now combine with Shared-SM overlap: it only reshapes
    // per-step prefill chunk budgets, so chunk boundaries change while greedy
    // tokens must not.
    {
        let auto_handle = acceptance
            .launch_engine(
                &model_path,
                4,
                8192,
                crate::Qwen35SchedulerPolicy::Auto,
                crate::Qwen35DecodeOverlap::SharedSm,
            )
            .expect("Failed to start Qwen3.5 auto + shared-SM scheduler");
        let mut auto_handle = EngineHarness::new(auto_handle);

        let auto_active_request = submit_repeated_token_request(
            &auto_handle,
            "overlap-auto-last-decoder",
            seed_token,
            512,
            128,
        );
        wait_for_first_token(
            &mut auto_handle,
            &auto_active_request,
            "overlap-auto-last-decoder",
        );
        let _ = drain_tokens(
            &mut auto_handle,
            &auto_active_request,
            "overlap-auto-last-decoder",
        );
        let auto_prefill_request = submit_repeated_token_request(
            &auto_handle,
            "overlap-auto-inflight-prefill",
            seed_token,
            8192,
            2,
        );
        wait_for_running_requests(&auto_handle, 2, std::time::Duration::from_secs(10));
        assert_no_generated_event(
            &mut auto_handle,
            &auto_prefill_request,
            "overlap-auto-inflight-prefill",
        );
        auto_active_request.abort();
        let auto_prefill = collect_generation_with_timeout(
            &mut auto_handle,
            &auto_prefill_request,
            "overlap-auto-inflight-prefill",
            None,
            std::time::Duration::from_secs(30),
        );
        assert_eq!(
            auto_prefill.tokens.len(),
            2,
            "auto + shared-SM in-flight prefill must finish after the last decoder is cancelled"
        );
        assert_eq!(
            auto_prefill.tokens, off_reference_tokens,
            "auto + shared-SM overlapped prefill must match the greedy default-Off reference"
        );
    }

    let handle = acceptance
        .launch_engine(
            &model_path,
            4,
            8192,
            crate::Qwen35SchedulerPolicy::Off,
            crate::Qwen35DecodeOverlap::SharedSm,
        )
        .expect("Failed to start Qwen3.5 shared-SM scheduler");
    let mut handle = EngineHarness::new(handle);

    let active_request =
        submit_repeated_token_request(&handle, "overlap-last-decoder", seed_token, 512, 128);
    let mut active_tokens =
        wait_for_first_token(&mut handle, &active_request, "overlap-last-decoder");
    active_tokens.extend(drain_tokens(
        &mut handle,
        &active_request,
        "overlap-last-decoder",
    ));
    let prefill_request =
        submit_repeated_token_request(&handle, "overlap-inflight-prefill", seed_token, 8192, 2);

    wait_for_running_requests(&handle, 2, std::time::Duration::from_secs(10));
    active_tokens.extend(drain_tokens(
        &mut handle,
        &active_request,
        "overlap-last-decoder",
    ));
    for _ in 0..2 {
        active_tokens.extend(wait_for_first_token(
            &mut handle,
            &active_request,
            "overlap-last-decoder",
        ));
        assert_no_generated_event(&mut handle, &prefill_request, "overlap-inflight-prefill");
    }
    assert_eq!(
        active_tokens,
        off_decoder_tokens[..active_tokens.len()],
        "Shared-SM active decoder must match the greedy default-Off reference"
    );
    active_request.abort();
    let prefill = collect_generation_with_timeout(
        &mut handle,
        &prefill_request,
        "overlap-inflight-prefill",
        None,
        std::time::Duration::from_secs(30),
    );
    assert_eq!(
        prefill.tokens.len(),
        2,
        "in-flight prefill must finish after the last decoder is cancelled"
    );
    assert_eq!(
        prefill.tokens, off_reference_tokens,
        "Shared-SM overlapped prefill must match the greedy default-Off reference"
    );

    let streaming_decoder =
        submit_repeated_token_request(&handle, "overlap-streaming-decoder", seed_token, 512, 1024);
    wait_for_first_token(&mut handle, &streaming_decoder, "overlap-streaming-decoder");
    let _ = drain_tokens(&mut handle, &streaming_decoder, "overlap-streaming-decoder");
    let streaming_prefill =
        submit_repeated_token_request(&handle, "overlap-streaming-prefill", seed_token, 128, 2);
    let deadline = Instant::now() + std::time::Duration::from_secs(30);
    let first = loop {
        let update = recv_event_before(
            &mut handle,
            &streaming_prefill,
            "overlap-streaming-prefill",
            deadline,
        );
        if !update.tokens.is_empty() || update.terminal.is_some() {
            break update;
        }
    };
    let decoder_tokens = drain_tokens(&mut handle, &streaming_decoder, "overlap-streaming-decoder");
    streaming_decoder.abort();
    assert_eq!(
        first.tokens.len(),
        1,
        "completed async prefill must publish its first token before the next decode"
    );
    assert!(first.terminal.is_none());
    assert!(
        !decoder_tokens.is_empty(),
        "the existing decoder must remain active"
    );
    let remaining = collect_generation_with_timeout(
        &mut handle,
        &streaming_prefill,
        "overlap-streaming-prefill",
        None,
        std::time::Duration::from_secs(30),
    );
    assert_eq!(remaining.tokens.len(), 1);
    assert_eq!(remaining.finish_reason, FinishReason::Length);
    wait_for_running_requests(&handle, 0, std::time::Duration::from_secs(10));
    assert_eq!(handle.metrics().num_waiting_reqs, 0);

    let (tokens, finish_reason) = generate_tokens(&mut handle, &tokenizer, "Hello again", 2);
    assert_eq!(
        tokens.len(),
        2,
        "scheduler must accept work after overlap wait"
    );
    assert_eq!(finish_reason, FinishReason::Length);

    let shutdown_active_request =
        submit_repeated_token_request(&handle, "overlap-shutdown-decoder", seed_token, 512, 128);
    wait_for_first_token(
        &mut handle,
        &shutdown_active_request,
        "overlap-shutdown-decoder",
    );
    let _ = drain_tokens(
        &mut handle,
        &shutdown_active_request,
        "overlap-shutdown-decoder",
    );
    let shutdown_prefill_request =
        submit_repeated_token_request(&handle, "overlap-shutdown-prefill", seed_token, 8192, 2);
    wait_for_running_requests(&handle, 2, std::time::Duration::from_secs(10));
    assert_no_generated_event(
        &mut handle,
        &shutdown_prefill_request,
        "overlap-shutdown-prefill",
    );
    shutdown_active_request.abort();
    shutdown_prefill_request.abort();

    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let shutdown = std::thread::spawn(move || {
        drop(handle);
        let _ = done_tx.send(());
    });
    done_rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .expect("dropping the last handle must drain in-flight prefill and return");
    shutdown.join().expect("scheduler shutdown thread panicked");
}

#[test]
#[ignore = "requires two CUDA devices, NCCL, and Qwen3.5 weights"]
fn test_e2e_qwen35_scheduler_tp2() {
    let Some(model_path) = common::model_path_or_skip("test_e2e_qwen35_scheduler_tp2") else {
        return;
    };

    info!("Loading Qwen3.5 TP2 model for scheduler test...");
    let start = Instant::now();
    let tokenizer = common::load_tokenizer(&model_path);
    let handle = crate::start_engine_with_capacity(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: false,
            device_ordinals: common::tp2_device_ordinals(),
            seed: 42,
            ..EngineLoadOptions::default()
        },
        8,
        crate::DEFAULT_MAX_PREFILL_TOKENS,
    )
    .expect("Failed to start Qwen3.5 TP2 scheduler");
    let mut handle = EngineHarness::new(handle);
    info!("TP2 scheduler loaded in {:.2?}", start.elapsed());

    let max_context_tokens = max_position_embeddings(&model_path);
    run_full_scheduler_e2e(&mut handle, &tokenizer, max_context_tokens, "TP2");
}

#[test]
#[ignore = "requires two CUDA devices, NCCL, and Qwen3.5 weights"]
fn test_e2e_qwen35_scheduler_tp2_graph() {
    let Some(model_path) = common::model_path_or_skip("test_e2e_qwen35_scheduler_tp2_graph") else {
        return;
    };

    info!("Loading Qwen3.5 TP2 model with CUDA Graph for scheduler test...");
    let start = Instant::now();
    let tokenizer = common::load_tokenizer(&model_path);
    // P2c: decode replays pre-captured CUDA Graphs when the TP-local decode
    // GQA group has a compiled kernel (4B/9B); uncompiled groups (27B group 6)
    // keep the batched eager path under the same request flow.
    let handle = crate::start_engine_with_capacity(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            device_ordinals: common::tp2_device_ordinals(),
            seed: 42,
            ..EngineLoadOptions::default()
        },
        8,
        crate::DEFAULT_MAX_PREFILL_TOKENS,
    )
    .expect("Failed to start Qwen3.5 TP2 graph scheduler");
    let mut handle = EngineHarness::new(handle);
    info!("TP2 graph scheduler loaded in {:.2?}", start.elapsed());

    let max_context_tokens = max_position_embeddings(&model_path);
    run_full_scheduler_e2e(&mut handle, &tokenizer, max_context_tokens, "TP2 graph");
}
