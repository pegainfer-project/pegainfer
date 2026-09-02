//! E2E scheduler integration test for Qwen3.5-4B.
//!
//! Tests the Qwen3.5 reduced-capacity scheduler path (batch prefill +
//! CUDA Graph decode) with sequential, concurrent, and client-abort requests.
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use log::info;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::RequestUpdate;
use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::sampler::SamplingParams;
use vllm_text::tokenizer::DynTokenizer;

mod common;

use common::harness::EngineHarness;
use common::harness::RequestStream;

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
    engine: &EngineHarness,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
) -> (Vec<u32>, FinishReason) {
    let result = generate_tokens_with_logprobs(engine, tokenizer, prompt, max_tokens, 0);
    (result.tokens, result.finish_reason)
}

fn generate_tokens_with_logprobs(
    engine: &EngineHarness,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
    logprobs: usize,
) -> GenerationResult {
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
    let mut request =
        common::harness::request(prompt_tokens, SamplingParams::default(), max_tokens);
    request.logprobs = logprobs;
    collect_generation(engine.submit(request), prompt, logprobs)
}

fn submit_repeated_token_request(
    engine: &EngineHarness,
    request_id: &str,
    token: u32,
    prompt_len: usize,
    max_tokens: usize,
) -> RequestStream {
    let mut request = common::harness::request(
        vec![token; prompt_len],
        SamplingParams {
            ignore_eos: true,
            ..SamplingParams::default()
        },
        max_tokens,
    );
    request.client_label = Some(Arc::from(request_id));
    engine.submit(request)
}

fn wait_for_first_token(stream: &mut RequestStream, request_id: &str) {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        match recv_update_before(stream, request_id, deadline) {
            Some(update) if !update.tokens.is_empty() => return,
            Some(update) if update.terminal.is_some() => {
                panic!(
                    "{request_id} emitted {:?} before its first token",
                    update.terminal
                )
            }
            Some(_) => {}
            None => panic!("scheduler closed before {request_id} emitted a token"),
        }
    }
}

fn recv_update_before(
    stream: &mut RequestStream,
    request_id: &str,
    deadline: Instant,
) -> Option<RequestUpdate> {
    loop {
        if let Some(update) = stream.try_recv() {
            return Some(update);
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {request_id} scheduler event"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn assert_no_generated_event(stream: &mut RequestStream, request_id: &str) {
    while let Some(update) = stream.try_recv() {
        assert!(
            update.tokens.is_empty(),
            "{request_id} emitted tokens {:?} before the overlap bound",
            update.tokens
        );
        assert!(
            update.terminal.is_none(),
            "{request_id} emitted {:?} before the overlap bound",
            update.terminal
        );
    }
}

fn drain_tokens(stream: &mut RequestStream, request_id: &str) -> usize {
    let mut tokens = 0;
    while let Some(update) = stream.try_recv() {
        assert!(
            update.terminal.is_none(),
            "{request_id} emitted {:?} while it must remain active",
            update.terminal
        );
        tokens += update.tokens.len();
    }
    tokens
}

fn wait_for_running_requests(engine: &EngineHarness, expected: u64, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    loop {
        let snapshot = engine.metrics();
        if snapshot.num_running_reqs == expected {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {expected} running requests; last snapshot: {snapshot:?}"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn collect_generation(stream: RequestStream, name: &str, logprobs: usize) -> GenerationResult {
    collect_generation_until(stream, name, logprobs, None)
}

fn collect_generation_with_timeout(
    stream: RequestStream,
    name: &str,
    logprobs: usize,
    timeout: Duration,
) -> GenerationResult {
    collect_generation_until(stream, name, logprobs, Some(Instant::now() + timeout))
}

fn collect_generation_until(
    mut stream: RequestStream,
    name: &str,
    logprobs: usize,
    deadline: Option<Instant>,
) -> GenerationResult {
    let mut tokens = Vec::new();
    let mut token_logprobs = Vec::new();
    loop {
        let update = match deadline {
            Some(deadline) => recv_update_before(&mut stream, name, deadline),
            None => stream.recv(),
        };
        match update {
            Some(update) => {
                apply_update(name, logprobs, &update, &mut tokens, &mut token_logprobs);
                if let Some(terminal) = update.terminal {
                    return generation_from_terminal(name, tokens, token_logprobs, terminal);
                }
            }
            None => panic!("{name}: scheduler channel closed without Finished"),
        }
    }
}

fn apply_update(
    name: &str,
    requested_logprobs: usize,
    update: &RequestUpdate,
    tokens: &mut Vec<u32>,
    token_logprobs: &mut Vec<Option<TokenLogprob>>,
) {
    if requested_logprobs == 0 {
        assert!(
            update.logprobs.iter().all(Option::is_none),
            "{name}: logprobs=0 should not return token logprobs"
        );
    } else if !update.tokens.is_empty() {
        assert_eq!(
            update.logprobs.len(),
            update.tokens.len(),
            "{name}: logprobs must be parallel to tokens"
        );
        for (id, logprob) in update.tokens.iter().copied().zip(update.logprobs.iter()) {
            let lp = logprob
                .as_ref()
                .unwrap_or_else(|| panic!("{name}: logprobs={requested_logprobs} returned None"));
            assert!(
                lp.logprob.is_finite(),
                "{name}: sampled token logprob must be finite"
            );
            assert_eq!(
                lp.top_logprobs.len(),
                requested_logprobs,
                "{name}: top_logprobs length should match the request"
            );
            assert!(
                lp.top_logprobs.iter().all(|&(_, v)| v.is_finite()),
                "{name}: top_logprobs must be finite"
            );
            assert_eq!(
                lp.top_logprobs.first().map(|&(token, _)| token),
                Some(id),
                "{name}: greedy sampled token should match top-1 logprob row"
            );
        }
    }
    tokens.extend_from_slice(&update.tokens);
    token_logprobs.extend(update.logprobs.iter().cloned());
}

fn generation_from_terminal(
    name: &str,
    tokens: Vec<u32>,
    logprobs: Vec<Option<TokenLogprob>>,
    terminal: Terminal,
) -> GenerationResult {
    match terminal {
        Terminal::Finished {
            reason: finish_reason,
            ..
        } => GenerationResult {
            tokens,
            logprobs,
            finish_reason,
        },
        Terminal::Rejected { reason, .. } => panic!("{name}: generation rejected: {reason}"),
        Terminal::Failed { message, .. } => panic!("{name}: generation failed: {message}"),
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

fn expect_context_window_rejection(engine: &EngineHarness, max_context_tokens: usize) {
    let mut request =
        common::harness::request(vec![1; max_context_tokens], SamplingParams::default(), 1);
    request.client_label = Some(Arc::from("over-context-window"));

    match engine.submit(request).outcome().terminal {
        Terminal::Rejected {
            reason:
                RejectReason::ContextLength {
                    prompt_tokens,
                    max_tokens,
                    limit,
                },
            prompt_tokens: reported_prompt,
        } => {
            assert_eq!(reported_prompt, max_context_tokens);
            assert_eq!(prompt_tokens, max_context_tokens);
            assert_eq!(max_tokens, 1);
            assert_eq!(limit, max_context_tokens);
            assert_eq!(
                prompt_tokens.saturating_add(max_tokens),
                max_context_tokens + 1
            );
        }
        Terminal::Rejected { reason, .. } => {
            panic!("expected context-window rejection, got: {reason}")
        }
        Terminal::Failed { message, .. } => {
            panic!("oversized prompt errored instead of clean rejection: {message}")
        }
        Terminal::Finished { .. } => panic!("expected context-window rejection"),
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
    engine: &EngineHarness,
    tokenizer: &DynTokenizer,
    max_context_tokens: usize,
    label: &str,
) {
    // logging intentionally left to the test harness

    // ── 0. Static context-window rejection ─────────────────────────────
    info!("=== Phase 0: Context-window rejection ===");
    expect_context_window_rejection(engine, max_context_tokens);
    info!("  PASS: over-context request rejected before prefill");

    // ── 1. logprobs must not change greedy tokens ─────────────────────
    info!("=== Phase 1: logprobs/no-logprobs token parity ===");
    for case in CASES.iter().take(3) {
        let max_tokens = case.max_new_tokens.min(16);
        let no_logprobs =
            generate_tokens_with_logprobs(engine, tokenizer, case.prompt, max_tokens, 0);
        let with_logprobs =
            generate_tokens_with_logprobs(engine, tokenizer, case.prompt, max_tokens, 1);
        assert_eq!(no_logprobs.finish_reason, with_logprobs.finish_reason);
        assert_eq!(
            no_logprobs.tokens, with_logprobs.tokens,
            "greedy token ids must not depend on whether logprobs are requested for {:?}",
            case.name
        );
        assert!(
            no_logprobs.logprobs.iter().all(Option::is_none),
            "logprobs=0 should keep the no-host-logprobs path for {:?}",
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
            "  PASS: {:?} logprobs=0 and logprobs=1 produced identical greedy tokens",
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
            generate_tokens(engine, tokenizer, case.prompt, case.max_new_tokens);
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
        let (tokens, _) = generate_tokens(engine, tokenizer, case.prompt, case.max_new_tokens);
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
        let mut streams: Vec<(String, usize, RequestStream)> = Vec::new();

        // Submit all cases concurrently, alternating greedy and sampled rows so
        // batch decode covers the mixed token-selection path from #284.
        for (case_idx, case) in CASES.iter().enumerate() {
            let prompt_tokens = tokenizer.encode(case.prompt, false).expect("encode failed");
            let mut request = common::harness::request(
                prompt_tokens,
                concurrent_params(case_idx),
                case.max_new_tokens,
            );
            request.client_label = Some(Arc::from(case.name));
            streams.push((case.name.to_string(), 0, engine.submit(request)));
        }

        for (name, logprobs, stream) in streams {
            let result = collect_generation(stream, &name, logprobs);
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
            ("mixed_no_logprobs", CASES[0].prompt, 0usize),
            ("mixed_with_logprobs", CASES[1].prompt, 1usize),
        ];
        let mut streams: Vec<(&str, usize, RequestStream)> = Vec::new();

        for (name, prompt, logprobs) in mixed {
            let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
            let mut request = common::harness::request(prompt_tokens, SamplingParams::default(), 8);
            request.logprobs = logprobs;
            request.client_label = Some(Arc::from(name));
            streams.push((name, logprobs, engine.submit(request)));
        }

        for (name, logprobs, stream) in streams {
            let result = collect_generation(stream, name, logprobs);
            assert!(!result.tokens.is_empty(), "{name}: produced no tokens");
            if logprobs == 0 {
                assert!(
                    result.logprobs.iter().all(Option::is_none),
                    "{name}: no-logprobs request should stay on the no-copy path"
                );
            } else {
                assert!(
                    result.logprobs.iter().all(Option::is_some),
                    "{name}: requested logprobs should be present"
                );
            }
            info!("  PASS: {name} → {} tokens", result.tokens.len());
        }
    }

    // ── 5. Client abort safety ──────────────────────────────────────────
    info!("=== Phase 5: Client abort ===");
    {
        let prompt_tokens = tokenizer.encode("Hello", false).expect("encode failed");
        let stream = engine.submit(common::harness::request(
            prompt_tokens,
            SamplingParams::default(),
            10,
        ));
        stream.control.abort();
        drop(stream);
        std::thread::sleep(Duration::from_millis(500));
        info!("  PASS: client abort handled");
    }

    // Verify scheduler survives
    let (tokens, _) = generate_tokens(engine, tokenizer, "Hello", 5);
    let text = tokenizer.decode(&tokens, true).expect("decode failed");
    assert!(!text.is_empty(), "scheduler dead after client abort");
    info!("  PASS: scheduler survived client abort");

    info!("All Qwen3.5 scheduler tests passed for {label}!");
}

#[test]
fn test_e2e_qwen35_scheduler() {
    let Some(model_path) = common::model_path_or_skip("test_e2e_qwen35_scheduler") else {
        return;
    };

    info!("Loading Qwen3.5 model for scheduler test...");
    let start = Instant::now();
    let tokenizer = common::load_tokenizer(&model_path);
    // Load through `start_engine_with_capacity` so recurrent-state reservation
    // matches the intended 8-slot 16GB budget. `from_safetensors_with_options`
    // still sizes to MAX_BATCH=64 and OOMs on a 16GB card before start.
    let engine: Engine = pegainfer_qwen35::start_engine_with_capacity(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            device_ordinals: vec![0],
            seed: 42,
            ..EngineLoadOptions::default()
        },
        8,
        pegainfer_qwen35::DEFAULT_MAX_PREFILL_TOKENS,
    )
    .expect("Failed to start Qwen3.5 scheduler");
    let engine = EngineHarness::new(engine);
    info!("scheduler loaded in {:.2?}", start.elapsed());

    let max_context_tokens = max_position_embeddings(&model_path);
    run_full_scheduler_e2e(&engine, &tokenizer, max_context_tokens, "TP1");
}

#[test]
fn test_e2e_qwen35_shared_sm_last_decoder() {
    pegainfer_core::logging::init_default();
    let Some(model_path) = common::model_path_or_skip("test_e2e_qwen35_shared_sm_last_decoder")
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

    let off_reference_tokens = {
        let off_engine: Engine = pegainfer_qwen35::start_engine_with_capacity_policy_and_overlap(
            Path::new(&model_path),
            EngineLoadOptions {
                enable_cuda_graph: true,
                device_ordinals: vec![0],
                seed: 42,
                ..EngineLoadOptions::default()
            },
            4,
            8192,
            pegainfer_qwen35::Qwen35SchedulerPolicy::Off,
            pegainfer_qwen35::Qwen35DecodeOverlap::Off,
        )
        .expect("Failed to start Qwen3.5 default-Off scheduler");
        let off_engine = EngineHarness::new(off_engine);
        let off_stream = submit_repeated_token_request(
            &off_engine,
            "overlap-off-reference",
            seed_token,
            8192,
            2,
        );
        let off = collect_generation_with_timeout(
            off_stream,
            "overlap-off-reference",
            0,
            Duration::from_secs(30),
        );
        assert_eq!(
            off.tokens.len(),
            2,
            "default-Off reference request must finish before Shared-SM parity"
        );
        off.tokens
    };

    let engine: Engine = pegainfer_qwen35::start_engine_with_capacity_policy_and_overlap(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            device_ordinals: vec![0],
            seed: 42,
            ..EngineLoadOptions::default()
        },
        4,
        8192,
        pegainfer_qwen35::Qwen35SchedulerPolicy::Off,
        pegainfer_qwen35::Qwen35DecodeOverlap::SharedSm,
    )
    .expect("Failed to start Qwen3.5 shared-SM scheduler");
    let engine = EngineHarness::new(engine);

    let mut active =
        submit_repeated_token_request(&engine, "overlap-last-decoder", seed_token, 512, 128);
    wait_for_first_token(&mut active, "overlap-last-decoder");
    let _ = drain_tokens(&mut active, "overlap-last-decoder");
    let mut prefill =
        submit_repeated_token_request(&engine, "overlap-inflight-prefill", seed_token, 8192, 2);

    wait_for_running_requests(&engine, 2, Duration::from_secs(10));
    let _ = drain_tokens(&mut active, "overlap-last-decoder");
    for _ in 0..2 {
        wait_for_first_token(&mut active, "overlap-last-decoder");
        assert_no_generated_event(&mut prefill, "overlap-inflight-prefill");
    }
    active.control.abort();
    drop(active);
    let prefill = collect_generation_with_timeout(
        prefill,
        "overlap-inflight-prefill",
        0,
        Duration::from_secs(30),
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

    let (tokens, finish_reason) = generate_tokens(&engine, &tokenizer, "Hello again", 2);
    assert_eq!(
        tokens.len(),
        2,
        "scheduler must accept work after overlap wait"
    );
    assert_eq!(finish_reason, FinishReason::Length);

    let mut shutdown_active =
        submit_repeated_token_request(&engine, "overlap-shutdown-decoder", seed_token, 512, 128);
    wait_for_first_token(&mut shutdown_active, "overlap-shutdown-decoder");
    let _ = drain_tokens(&mut shutdown_active, "overlap-shutdown-decoder");
    let mut shutdown_prefill =
        submit_repeated_token_request(&engine, "overlap-shutdown-prefill", seed_token, 8192, 2);
    wait_for_running_requests(&engine, 2, Duration::from_secs(10));
    assert_no_generated_event(&mut shutdown_prefill, "overlap-shutdown-prefill");
    shutdown_active.control.abort();
    shutdown_prefill.control.abort();
    drop(shutdown_active);
    drop(shutdown_prefill);

    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let shutdown = std::thread::spawn(move || {
        drop(engine);
        let _ = done_tx.send(());
    });
    done_rx
        .recv_timeout(Duration::from_secs(30))
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
    // TP Phase 1 is eager-only; CUDA Graph must stay disabled for multi-device startup.
    let engine: Engine = pegainfer_qwen35::start_engine_with_capacity(
        Path::new(&model_path),
        EngineLoadOptions {
            enable_cuda_graph: false,
            device_ordinals: common::tp2_device_ordinals(),
            seed: 42,
            ..EngineLoadOptions::default()
        },
        8,
        pegainfer_qwen35::DEFAULT_MAX_PREFILL_TOKENS,
    )
    .expect("Failed to start Qwen3.5 TP2 scheduler");
    let engine = EngineHarness::new(engine);
    info!("TP2 scheduler loaded in {:.2?}", start.elapsed());

    let max_context_tokens = max_position_embeddings(&model_path);
    run_full_scheduler_e2e(&engine, &tokenizer, max_context_tokens, "TP2");
}
