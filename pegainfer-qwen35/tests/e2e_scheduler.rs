/// E2E scheduler integration test for Qwen3.5-4B.
///
/// Tests the Qwen3.5 reduced-capacity scheduler path (batch prefill +
/// CUDA Graph decode) with sequential, concurrent, and consumer-drop requests.
use std::collections::HashSet;
/// E2E scheduler integration test for Qwen3.5-4B.
///
/// Tests the Qwen3.5 reduced-capacity scheduler path (batch prefill +
/// CUDA Graph decode) with sequential, concurrent, and consumer-drop requests.
use std::path::Path;
/// E2E scheduler integration test for Qwen3.5-4B.
///
/// Tests the Qwen3.5 reduced-capacity scheduler path (batch prefill +
/// CUDA Graph decode) with sequential, concurrent, and consumer-drop requests.
use std::time::Instant;

use log::info;
use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::engine::TokenSink;
use pegainfer_frontend::engine::TokenStreamReceiver;
use pegainfer_frontend::sampler::SamplingParams;
use vllm_text::tokenizer::DynTokenizer;

mod common;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");

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

fn get_model_path() -> String {
    std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string())
}

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
    handle: &EngineHandle,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
) -> (Vec<u32>, FinishReason) {
    let result = generate_tokens_with_logprobs(handle, tokenizer, prompt, max_tokens, 0);
    (result.tokens, result.finish_reason)
}

fn generate_tokens_with_logprobs(
    handle: &EngineHandle,
    tokenizer: &DynTokenizer,
    prompt: &str,
    max_tokens: usize,
    logprobs: usize,
) -> GenerationResult {
    let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
    let (token_tx, mut token_rx) = TokenSink::standalone();

    handle
        .submit(GenerateRequest {
            trace_parent: None,
            request_id: None,
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens,
            lora_adapter: None,
            kv_transfer_params: None,
            token_tx,
            logprobs,
            echo: false,
        })
        .expect("submit failed");

    collect_generation(&mut token_rx, prompt, logprobs)
}

fn collect_generation(
    token_rx: &mut TokenStreamReceiver,
    name: &str,
    logprobs: usize,
) -> GenerationResult {
    let mut tokens = Vec::new();
    let mut token_logprobs = Vec::new();
    loop {
        match token_rx.blocking_recv().map(|(_, event)| event) {
            Some(TokenEvent::Token { id, logprob }) => {
                if logprobs == 0 {
                    assert!(
                        logprob.is_none(),
                        "{name}: logprobs=0 should not return token logprobs"
                    );
                } else {
                    let lp = logprob
                        .as_ref()
                        .unwrap_or_else(|| panic!("{name}: logprobs={logprobs} returned None"));
                    assert!(
                        lp.logprob.is_finite(),
                        "{name}: sampled token logprob must be finite"
                    );
                    assert_eq!(
                        lp.top_logprobs.len(),
                        logprobs,
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
                tokens.push(id);
                token_logprobs.push(logprob);
            }
            Some(
                TokenEvent::PromptTokens { .. }
                | TokenEvent::Scheduled { .. }
                | TokenEvent::KvTransfer { .. },
            ) => {}
            Some(TokenEvent::Finished { finish_reason, .. }) => {
                return GenerationResult {
                    tokens,
                    logprobs: token_logprobs,
                    finish_reason,
                };
            }
            Some(TokenEvent::Error { message, .. }) => panic!("generation failed: {message}"),
            Some(TokenEvent::Rejected { message, .. }) => panic!("generation rejected: {message}"),
            None => panic!("{name}: scheduler channel closed without Finished"),
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

fn expect_context_window_rejection(handle: &EngineHandle, max_context_tokens: usize) {
    let (token_tx, mut token_rx) = TokenSink::standalone();

    handle
        .submit(GenerateRequest {
            trace_parent: None,
            request_id: Some("over-context-window".to_string()),
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens: vec![1; max_context_tokens],
            params: SamplingParams::default(),
            max_tokens: 1,
            lora_adapter: None,
            kv_transfer_params: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })
        .expect("submit over-context request");

    match token_rx.blocking_recv().map(|(_, event)| event) {
        Some(TokenEvent::Rejected {
            message,
            prompt_tokens,
            completion_tokens,
        }) => {
            assert_eq!(prompt_tokens, max_context_tokens);
            assert_eq!(completion_tokens, 0);
            assert!(
                message.contains("maximum context length"),
                "expected context-window rejection, got: {message}"
            );
            assert!(
                message.contains(&(max_context_tokens + 1).to_string()),
                "rejection should report prompt + max_tokens, got: {message}"
            );
        }
        Some(_) => panic!("expected context-window rejection"),
        None => panic!("scheduler channel closed without rejection"),
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
    handle: &EngineHandle,
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
            generate_tokens_with_logprobs(handle, tokenizer, case.prompt, max_tokens, 0);
        let with_logprobs =
            generate_tokens_with_logprobs(handle, tokenizer, case.prompt, max_tokens, 1);
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
        let mut receivers: Vec<(String, usize, TokenStreamReceiver)> = Vec::new();

        // Submit all cases concurrently, alternating greedy and sampled rows so
        // batch decode covers the mixed token-selection path from #284.
        for (case_idx, case) in CASES.iter().enumerate() {
            let prompt_tokens = tokenizer.encode(case.prompt, false).expect("encode failed");
            let (token_tx, token_rx) = TokenSink::standalone();
            handle
                .submit(GenerateRequest {
                    trace_parent: None,
                    request_id: None,
                    queued_at_unix_s: None,
                    data_parallel_rank: None,
                    prompt_tokens,
                    params: concurrent_params(case_idx),
                    max_tokens: case.max_new_tokens,
                    lora_adapter: None,
                    kv_transfer_params: None,
                    token_tx,
                    logprobs: 0,
                    echo: false,
                })
                .expect("submit failed");
            receivers.push((case.name.to_string(), 0, token_rx));
        }

        // Collect all results
        for (name, logprobs, mut rx) in receivers {
            let result = collect_generation(&mut rx, &name, logprobs);
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
        let mut receivers: Vec<(&str, usize, TokenStreamReceiver)> = Vec::new();

        for (name, prompt, logprobs) in mixed {
            let prompt_tokens = tokenizer.encode(prompt, false).expect("encode failed");
            let (token_tx, token_rx) = TokenSink::standalone();
            handle
                .submit(GenerateRequest {
                    trace_parent: None,
                    request_id: Some(name.to_string()),
                    queued_at_unix_s: None,
                    data_parallel_rank: None,
                    prompt_tokens,
                    params: SamplingParams::default(),
                    max_tokens: 8,
                    lora_adapter: None,
                    kv_transfer_params: None,
                    token_tx,
                    logprobs,
                    echo: false,
                })
                .expect("submit failed");
            receivers.push((name, logprobs, token_rx));
        }

        for (name, logprobs, mut rx) in receivers {
            let result = collect_generation(&mut rx, name, logprobs);
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

    // ── 5. Consumer drop safety ─────────────────────────────────────────
    info!("=== Phase 5: Consumer drop ===");
    {
        let prompt_tokens = tokenizer.encode("Hello", false).expect("encode failed");
        let (token_tx, rx) = TokenSink::standalone();
        drop(rx);
        handle
            .submit(GenerateRequest {
                trace_parent: None,
                request_id: None,
                queued_at_unix_s: None,
                data_parallel_rank: None,
                prompt_tokens,
                params: SamplingParams::default(),
                max_tokens: 10,
                lora_adapter: None,
                kv_transfer_params: None,
                token_tx,
                logprobs: 0,
                echo: false,
            })
            .expect("submit failed");
        std::thread::sleep(std::time::Duration::from_millis(500));
        info!("  PASS: consumer drop handled");
    }

    // Verify scheduler survives
    let (tokens, _) = generate_tokens(handle, tokenizer, "Hello", 5);
    let text = tokenizer.decode(&tokens, true).expect("decode failed");
    assert!(!text.is_empty(), "scheduler dead after consumer drop");
    info!("  PASS: scheduler survived consumer drop");

    info!("All Qwen3.5 scheduler tests passed for {label}!");
}

fn context_limit_for(handle: &EngineHandle, model_path: &str) -> usize {
    handle
        .servable_len()
        .map_or_else(|| max_position_embeddings(model_path), |len| len as usize)
}

#[test]
fn test_e2e_qwen35_scheduler() {
    let model_path = get_model_path();

    info!("Loading Qwen3.5 model for scheduler test...");
    let start = Instant::now();
    let model =
        pegainfer_qwen35::runtime::Qwen35Model::from_safetensors_with_options(&model_path, true)
            .expect("Failed to load model");
    let tokenizer = common::load_tokenizer(&model_path);
    // Use reduced batch capacity (8) to fit on 16GB GPUs alongside the model.
    let handle = pegainfer_qwen35::runtime::start_with_capacity(
        model,
        42,
        8,
        pegainfer_qwen35::DEFAULT_MAX_PREFILL_TOKENS,
    )
    .expect("Failed to start Qwen3.5 scheduler");
    info!("scheduler loaded in {:.2?}", start.elapsed());

    let max_context_tokens = context_limit_for(&handle, &model_path);
    run_full_scheduler_e2e(&handle, &tokenizer, max_context_tokens, "TP1");
}

#[test]
#[ignore = "requires an SM120 GPU, Qwen3.5-4B weights, and the validated Hv32 FlashInfer artifact"]
fn test_e2e_qwen35_scheduler_flashinfer_gdn() {
    let model_path = get_model_path();
    info!("Loading Qwen3.5 model for FlashInfer scheduler test...");
    let start = Instant::now();
    let tokenizer = common::load_tokenizer(&model_path);
    let (handle, evidence) =
        pegainfer_qwen35::runtime::start_engine_with_flashinfer_gdn_for_accuracy(
            Path::new(&model_path),
            0,
            8,
            pegainfer_qwen35::DEFAULT_MAX_PREFILL_TOKENS,
        )
        .expect("Failed to start FlashInfer Qwen3.5 scheduler");
    let initial = evidence.snapshot();
    assert_eq!(initial.successful_launches, 0);
    info!(
        "FlashInfer identity: object_sha256={} object_bytes={}",
        initial.artifact_sha256, initial.artifact_size_bytes
    );
    info!("FlashInfer scheduler loaded in {:.2?}", start.elapsed());

    let max_context_tokens = context_limit_for(&handle, &model_path);
    run_full_scheduler_e2e(
        &handle,
        &tokenizer,
        max_context_tokens,
        "TP1 FlashInfer GDN",
    );
    let final_evidence = evidence.snapshot();
    assert!(
        final_evidence.successful_launches > 0,
        "scheduler e2e completed without a successful FlashInfer GDN launch"
    );
    info!(
        "FlashInfer scheduler successful launches: {}",
        final_evidence.successful_launches
    );
}

#[test]
#[ignore = "requires two CUDA devices, NCCL, and Qwen3.5 weights"]
fn test_e2e_qwen35_scheduler_tp2() {
    let model_path = get_model_path();

    info!("Loading Qwen3.5 TP2 model for scheduler test...");
    let start = Instant::now();
    let tokenizer = common::load_tokenizer(&model_path);
    // TP Phase 1 is eager-only; CUDA Graph must stay disabled for multi-device startup.
    let handle = pegainfer_qwen35::start_engine_with_capacity(
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
    info!("TP2 scheduler loaded in {:.2?}", start.elapsed());

    let max_context_tokens = context_limit_for(&handle, &model_path);
    run_full_scheduler_e2e(&handle, &tokenizer, max_context_tokens, "TP2");
}
