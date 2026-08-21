//! DeepSeek-V2-Lite EP2 correctness and integration gate.
//!
//! This test verifies model loading, host-staged/NCCL generation, request-flow
//! isolation, token/text hashes, and EP route/collective accounting. It does
//! not measure serving latency, throughput, SLO compliance, soak behavior, or
//! production readiness; those claims belong to retained benchmark artifacts.

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Barrier;
use std::thread;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use pegainfer_deepseek_v2_lite::DeepSeekV2LiteEp2Generator;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::EosPolicy;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::StopPolicy;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_frontend::engine::TokenSink;
use pegainfer_frontend::engine::TokenStreamReceiver;
use pegainfer_frontend::sampler::SamplingParams;
use serde::Deserialize;
use sha2::Digest;
use sha2::Sha256;
use vllm_text::tokenizer::HuggingFaceTokenizer;
use vllm_text::tokenizer::Tokenizer;

const EXPECTED_GENERATED_TOKENS: usize = 16;
const EXPECTED_OUTPUT_SHA256_PAIRS: &[(&str, &str, &str)] = &[
    (
        "4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225",
        "0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347",
        "DeepSeek-V2-Lite snapshot 604d5664 on 2x RTX 5090, torch 2.7.0, transformers 4.40.2",
    ),
    (
        "d05a7b0f0ac6435fb51040582a337d8b6d72844dd61194daa1b3090fa0e16ce8",
        "4aaafbe4b3a46bc5b9ab5ea8d09d5fad71225006c2e234e87a928e3265b387c6",
        "DeepSeek-V2-Lite snapshot 604d5664 on 2x A800-SXM4-80GB, torch 2.7.0, transformers 4.40.2",
    ),
];
const DSV2_LITE_HIDDEN_SIZE: usize = 2048;
const DSV2_LITE_MOE_LAYERS: usize = 26;
const E2E_JSON_OUT_ENV: &str = "PEGAINFER_DSV2_LITE_E2E_JSON_OUT";
const E2E_CASE_SET_ENV: &str = "PEGAINFER_DSV2_LITE_E2E_CASE_SET";

fn request_stop_policy(ignore_eos: bool) -> StopPolicy {
    StopPolicy {
        eos: if ignore_eos {
            EosPolicy::Ignore
        } else {
            EosPolicy::ModelDefault
        },
        token_ids: vec![],
    }
}

#[derive(Debug, Deserialize)]
struct CaseSet {
    cases: Vec<GateCase>,
}

#[derive(Debug, Clone, Deserialize)]
struct GateCase {
    id: String,
    prompt: String,
    output_len: usize,
    #[serde(default = "default_batch_size")]
    batch_size: usize,
    #[serde(default)]
    ignore_eos: bool,
}

#[test]
fn test_deepseek_v2_lite_ep2_rust_generation() -> Result<()> {
    let model_path_label = env::var("PEGAINFER_TEST_MODEL_PATH")
        .context("PEGAINFER_TEST_MODEL_PATH must point to DeepSeek-V2-Lite weights")?;
    let model_path = resolve_model_path(&model_path_label);
    ensure!(
        model_path.join("config.json").exists(),
        "missing config.json under {}",
        model_path.display()
    );

    let duplicate_ordinal_err = DeepSeekV2LiteEp2Generator::load(
        &model_path,
        EngineLoadOptions {
            enable_cuda_graph: false,
            device_ordinals: vec![0, 0],
            seed: 42,
            ..EngineLoadOptions::default()
        },
    )
    .err()
    .context("duplicate CUDA device ordinals unexpectedly loaded")?;
    ensure!(
        format!("{duplicate_ordinal_err:#}").contains("two distinct CUDA device ordinals"),
        "duplicate CUDA ordinal error should mention distinct devices, got {duplicate_ordinal_err:#}"
    );

    run_rust_generation(&model_path_label, &model_path)?;
    run_mixed_serving_generation(&model_path, &model_path_label)
}

fn run_rust_generation(model_path_label: &str, model_path: &Path) -> Result<()> {
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = HuggingFaceTokenizer::new(&tokenizer_path).map_err(|err| {
        anyhow::anyhow!(
            "failed to load tokenizer {}: {err:?}",
            tokenizer_path.display()
        )
    })?;
    if let Some((case_set_path, cases)) = load_case_set_from_env()? {
        return run_case_set_generation(
            model_path_label,
            model_path,
            &tokenizer,
            &case_set_path,
            &cases,
        );
    }

    let prompt = "Hello";
    let prompt_tokens = tokenizer
        .encode(prompt, false)
        .map_err(|err| anyhow::anyhow!("encode prompt failed: {err:?}"))?;
    ensure!(!prompt_tokens.is_empty(), "tokenizer returned empty prompt");

    let mut generator = DeepSeekV2LiteEp2Generator::load(
        model_path,
        EngineLoadOptions {
            enable_cuda_graph: false,
            device_ordinals: vec![0, 1],
            seed: 42,
            ..EngineLoadOptions::default()
        },
    )?;
    let result = generator.generate_greedy(&prompt_tokens, 16, false)?;
    ensure!(
        !result.tokens.is_empty(),
        "DeepSeek-V2-Lite Rust generation produced no tokens"
    );
    ensure!(
        result.stats.ep_size == 2,
        "DeepSeek-V2-Lite E2E expected ep_size=2, got {}",
        result.stats.ep_size
    );
    ensure!(
        result.stats.device_ordinals == vec![0, 1],
        "DeepSeek-V2-Lite E2E expected devices [0, 1], got {:?}",
        result.stats.device_ordinals
    );
    ensure!(
        result.stats.generated_tokens == EXPECTED_GENERATED_TOKENS,
        "DeepSeek-V2-Lite E2E generated {} tokens, expected {}",
        result.stats.generated_tokens,
        EXPECTED_GENERATED_TOKENS
    );
    ensure!(
        result.finish_reason == FinishReason::Length,
        "DeepSeek-V2-Lite E2E finish_reason drift: got {:?}, expected Length",
        result.finish_reason
    );
    ensure!(
        result.stats.ep_backend == current_backend(),
        "DeepSeek-V2-Lite E2E backend mismatch: got {}, expected {}",
        result.stats.ep_backend,
        current_backend()
    );
    match result.stats.ep_backend.as_str() {
        "host-staged" => {
            ensure!(
                result.stats.host_dispatch_calls > 0
                    && result.stats.host_combine_calls == result.stats.host_dispatch_calls
                    && result.stats.host_dispatch_elements > 0
                    && result.stats.host_combine_elements == result.stats.host_dispatch_elements,
                "host-staged EP gate did not record dispatch/combine counts"
            );
            ensure!(
                result.stats.host_dispatch_remote_routes > 0,
                "host-staged EP gate did not exercise any remote routed expert"
            );
            ensure!(
                result.stats.host_dispatch_local_routes > 0,
                "host-staged EP gate did not exercise any local routed expert"
            );
            ensure!(
                result.stats.nccl_dense_exchange_calls == 0
                    && result.stats.nccl_combine_calls == 0
                    && result.stats.nccl_dense_exchange_elements == 0
                    && result.stats.nccl_combine_elements == 0,
                "host-staged EP gate unexpectedly recorded NCCL collectives"
            );
        }
        "nccl" => {
            ensure!(
                result.stats.nccl_dispatch_remote_routes > 0,
                "NCCL EP gate did not exercise any remote routed expert"
            );
            ensure!(
                result.stats.nccl_dispatch_local_routes > 0,
                "NCCL EP gate did not exercise any local routed expert"
            );
            ensure!(
                result.stats.nccl_combine_routes
                    == result.stats.nccl_dispatch_local_routes
                        + result.stats.nccl_dispatch_remote_routes,
                "NCCL combine route accounting drift"
            );
            let expected_moe_calls = result.stats.generated_tokens * DSV2_LITE_MOE_LAYERS;
            let expected_collective_elements = expected_moe_calls * DSV2_LITE_HIDDEN_SIZE;
            ensure!(
                result.stats.nccl_dense_exchange_calls == expected_moe_calls,
                "NCCL dense hidden exchange call count drift: got {}, expected {}",
                result.stats.nccl_dense_exchange_calls,
                expected_moe_calls
            );
            ensure!(
                result.stats.nccl_combine_calls == expected_moe_calls,
                "NCCL combine call count drift: got {}, expected {}",
                result.stats.nccl_combine_calls,
                expected_moe_calls
            );
            ensure!(
                result.stats.nccl_dense_exchange_elements == expected_collective_elements,
                "NCCL dense hidden exchange element count drift: got {}, expected {}",
                result.stats.nccl_dense_exchange_elements,
                expected_collective_elements
            );
            ensure!(
                result.stats.nccl_combine_elements == expected_collective_elements,
                "NCCL combine element count drift: got {}, expected {}",
                result.stats.nccl_combine_elements,
                expected_collective_elements
            );
        }
        other => anyhow::bail!("unexpected DeepSeek-V2-Lite EP backend in E2E: {other}"),
    }

    let output_text = tokenizer
        .decode(&result.tokens, false)
        .map_err(|err| anyhow::anyhow!("decode output failed: {err:?}"))?;
    let mut hasher = Sha256::new();
    hasher.update(output_text.as_bytes());
    let output_text_sha256 = hex::encode(hasher.finalize());
    let matched_output_oracle =
        matched_expected_output_oracle(&result.stats.output_token_sha256, &output_text_sha256);
    let payload = serde_json::json!({
        "schema_version": 1,
        "kind": "deepseek_v2_lite_ep2_correctness",
        "report_intent": "correctness_integration",
        "model_path": model_path_label,
        "gpu_count": 2,
        "ep_size": result.stats.ep_size,
        "ep_backend": result.stats.ep_backend,
        "devices": &result.stats.device_ordinals,
        "prompt": prompt,
        "prompt_tokens": result.stats.prompt_tokens,
        "prompt_token_ids": &prompt_tokens,
        "max_new_tokens": 16,
        "generated_tokens": result.stats.generated_tokens,
        "generated_token_ids": &result.tokens,
        "generated_text": &output_text,
        "output_token_sha256": result.stats.output_token_sha256,
        "output_text_sha256": output_text_sha256,
        "matched_output_oracle": matched_output_oracle,
        "token_sha256_algorithm": "sha256 over generated token ids encoded as little-endian u32",
        "text_sha256_algorithm": "sha256 over UTF-8 generated text bytes",
        "host_dispatch_calls": result.stats.host_dispatch_calls,
        "host_dispatch_elements": result.stats.host_dispatch_elements,
        "host_combine_calls": result.stats.host_combine_calls,
        "host_combine_elements": result.stats.host_combine_elements,
        "host_dispatch_local_routes": result.stats.host_dispatch_local_routes,
        "host_dispatch_remote_routes": result.stats.host_dispatch_remote_routes,
        "nccl_dispatch_local_routes": result.stats.nccl_dispatch_local_routes,
        "nccl_dispatch_remote_routes": result.stats.nccl_dispatch_remote_routes,
        "nccl_combine_routes": result.stats.nccl_combine_routes,
        "nccl_dense_exchange_calls": result.stats.nccl_dense_exchange_calls,
        "nccl_combine_calls": result.stats.nccl_combine_calls,
        "nccl_dense_exchange_elements": result.stats.nccl_dense_exchange_elements,
        "nccl_combine_elements": result.stats.nccl_combine_elements,
        "output_text": &output_text,
        "claim_boundary": "Correctness and integration evidence for the covered DeepSeek-V2-Lite EP2 generation path. This artifact makes no latency, throughput, serving SLO, soak, or production-readiness claim.",
    });
    let payload_text = serde_json::to_string_pretty(&payload)?;
    if let Ok(path) = env::var(E2E_JSON_OUT_ENV) {
        if !path.is_empty() {
            let path = PathBuf::from(path);
            let path = resolve_workspace_path(path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create {}", parent.display()))?;
            }
            fs::write(&path, format!("{payload_text}\n"))
                .with_context(|| format!("write {}", path.display()))?;
        }
    }
    println!("{payload_text}");
    ensure!(
        matched_output_oracle.is_some(),
        "DeepSeek-V2-Lite E2E hash drift: got token_sha256={} text_sha256={}, expected one HF-confirmed pair from {:?}",
        result.stats.output_token_sha256,
        output_text_sha256,
        EXPECTED_OUTPUT_SHA256_PAIRS
    );
    Ok(())
}

fn run_case_set_generation(
    model_path_label: &str,
    model_path: &Path,
    tokenizer: &HuggingFaceTokenizer,
    case_set_path: &Path,
    cases: &[GateCase],
) -> Result<()> {
    ensure!(
        !cases.is_empty(),
        "DeepSeek-V2-Lite E2E case set must contain at least one case"
    );
    let mut generator = DeepSeekV2LiteEp2Generator::load(
        model_path,
        EngineLoadOptions {
            enable_cuda_graph: false,
            device_ordinals: vec![0, 1],
            seed: 42,
            ..EngineLoadOptions::default()
        },
    )?;

    // The generator owns loaded weights and reusable backend scratch. Each
    // generate call below builds fresh DecodeCache and GenerationStats values,
    // so the case set can share one model load without sharing KV state.
    let mut case_payloads = Vec::with_capacity(cases.len());
    for case in cases {
        case_payloads.push(run_case_set_case(tokenizer, &mut generator, case)?);
    }

    let payload = serde_json::json!({
        "schema_version": 1,
        "kind": "deepseek_v2_lite_ep2_correctness_case_set",
        "report_intent": "correctness_integration",
        "schema": 2,
        "report_type": "deepseek-v2-lite-ep2-rust-e2e-case-set",
        "model_path": model_path_label,
        "case_set_json": case_set_path.display().to_string(),
        "gpu_count": 2,
        "ep_size": 2,
        "ep_backend": current_backend(),
        "devices": [0, 1],
        "case_count": case_payloads.len(),
        "token_sha256_algorithm": "sha256 over generated token ids encoded as little-endian u32",
        "text_sha256_algorithm": "sha256 over UTF-8 generated text bytes",
        "cases": case_payloads,
        "claim_boundary": "Correctness and integration evidence for the covered DeepSeek-V2-Lite EP2 case set. This artifact makes no latency, throughput, serving SLO, soak, or production-readiness claim.",
    });
    let payload_text = serde_json::to_string_pretty(&payload)?;
    write_payload_if_requested(&payload_text)?;
    println!("{payload_text}");
    Ok(())
}

fn run_case_set_case(
    tokenizer: &HuggingFaceTokenizer,
    generator: &mut DeepSeekV2LiteEp2Generator,
    case: &GateCase,
) -> Result<serde_json::Value> {
    let prompt_tokens = tokenizer
        .encode(&case.prompt, false)
        .map_err(|err| anyhow::anyhow!("encode prompt for case {} failed: {err:?}", case.id))?;
    ensure!(
        !prompt_tokens.is_empty(),
        "tokenizer returned empty prompt for case {}",
        case.id
    );

    if case.batch_size == 1 {
        let result = generator.generate_greedy(&prompt_tokens, case.output_len, case.ignore_eos)?;
        ensure!(
            !result.tokens.is_empty(),
            "case {} produced no tokens",
            case.id
        );
        ensure!(
            result.tokens.len() <= case.output_len,
            "case {} generated {} tokens, expected at most {}",
            case.id,
            result.tokens.len(),
            case.output_len
        );
        if case.ignore_eos {
            ensure!(
                result.tokens.len() == case.output_len,
                "case {} uses ignore_eos=true and must generate exactly {} tokens, got {}",
                case.id,
                case.output_len,
                result.tokens.len()
            );
        }
        validate_generation_stats(
            &result.stats,
            case,
            prompt_tokens.len(),
            result.tokens.len(),
        )?;
        let generated_text = decode_tokens(tokenizer, &result.tokens, &case.id)?;
        let text_sha256 = sha256_text(&generated_text);
        let matched_output_oracle =
            matched_expected_output_oracle(&result.stats.output_token_sha256, &text_sha256);

        Ok(serde_json::json!({
            "id": &case.id,
            "prompt": &case.prompt,
            "prompt_token_ids": &prompt_tokens,
            "prompt_tokens": prompt_tokens.len(),
            "batch_size": case.batch_size,
            "output_len": case.output_len,
            "ignore_eos": case.ignore_eos,
            "generated_tokens": result.tokens.len(),
            "generated_token_ids": &result.tokens,
            "generated_text": generated_text,
            "output_token_sha256": result.stats.output_token_sha256,
            "output_text_sha256": text_sha256,
            "matched_output_oracle": matched_output_oracle,
            "finish_reason": format!("{:?}", result.finish_reason),
            "ep": ep_payload(&result.stats),
        }))
    } else {
        ensure!(
            case.ignore_eos,
            "batch case {} must set ignore_eos=true so each row has the requested output length",
            case.id
        );
        let result = generator.generate_greedy_batch_same_prompt_with_timings(
            &prompt_tokens,
            case.batch_size,
            case.output_len,
            case.ignore_eos,
        )?;
        ensure!(
            result.tokens.len() == case.batch_size,
            "case {} returned {} rows, expected batch_size={}",
            case.id,
            result.tokens.len(),
            case.batch_size
        );
        ensure!(
            result
                .tokens
                .iter()
                .all(|tokens| tokens.len() == case.output_len),
            "case {} batch rows must all generate exactly {} tokens",
            case.id,
            case.output_len
        );
        validate_generation_stats(
            &result.stats,
            case,
            prompt_tokens.len(),
            result.tokens[0].len(),
        )?;

        let mut generated_text_by_row = Vec::with_capacity(result.tokens.len());
        let mut token_sha256_by_row = Vec::with_capacity(result.tokens.len());
        let mut text_sha256_by_row = Vec::with_capacity(result.tokens.len());
        for row in &result.tokens {
            let generated_text = decode_tokens(tokenizer, row, &case.id)?;
            token_sha256_by_row.push(token_sha256(row));
            text_sha256_by_row.push(sha256_text(&generated_text));
            generated_text_by_row.push(generated_text);
        }
        ensure!(
            token_sha256_by_row
                .iter()
                .all(|hash| hash == &token_sha256_by_row[0])
                && text_sha256_by_row
                    .iter()
                    .all(|hash| hash == &text_sha256_by_row[0]),
            "case {} same-prompt batch rows are not hash-identical",
            case.id
        );
        let matched_output_oracle =
            matched_expected_output_oracle(&token_sha256_by_row[0], &text_sha256_by_row[0]);

        Ok(serde_json::json!({
            "id": &case.id,
            "prompt": &case.prompt,
            "prompt_token_ids": &prompt_tokens,
            "prompt_tokens": prompt_tokens.len(),
            "batch_size": case.batch_size,
            "output_len": case.output_len,
            "ignore_eos": case.ignore_eos,
            "generated_tokens": result.tokens[0].len(),
            "generated_tokens_total": result.stats.generated_tokens,
            "generated_token_ids": &result.tokens[0],
            "generated_text": &generated_text_by_row[0],
            "output_token_sha256": &token_sha256_by_row[0],
            "output_text_sha256": &text_sha256_by_row[0],
            "generated_token_ids_by_row": &result.tokens,
            "generated_text_by_row": &generated_text_by_row,
            "token_sha256_by_row": &token_sha256_by_row,
            "text_sha256_by_row": &text_sha256_by_row,
            "same_prompt_rows_exact": true,
            "matched_output_oracle": matched_output_oracle,
            "ep": ep_payload(&result.stats),
        }))
    }
}

fn run_mixed_serving_generation(model_path: &Path, model_path_label: &str) -> Result<()> {
    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = HuggingFaceTokenizer::new(&tokenizer_path).map_err(|err| {
        anyhow::anyhow!(
            "failed to load tokenizer {}: {err:?}",
            tokenizer_path.display()
        )
    })?;
    let cases = [
        ("mixed-hello", "Hello", 8usize, false),
        ("mixed-world", "World", 8usize, false),
        ("mixed-paris", "Paris", 8usize, false),
        ("mixed-china", "China", 8usize, false),
    ];

    let mut sequential = DeepSeekV2LiteEp2Generator::load(
        model_path,
        EngineLoadOptions {
            enable_cuda_graph: false,
            device_ordinals: vec![0, 1],
            seed: 42,
            ..EngineLoadOptions::default()
        },
    )?;
    let mut encoded_cases = Vec::with_capacity(cases.len());
    let mut expected = Vec::with_capacity(cases.len());
    for (id, prompt, max_tokens, ignore_eos) in cases {
        let prompt_tokens = tokenizer
            .encode(prompt, false)
            .map_err(|err| anyhow::anyhow!("encode prompt for {id} failed: {err:?}"))?;
        ensure!(
            !prompt_tokens.is_empty(),
            "tokenizer returned empty prompt for mixed-serving case {id}"
        );
        let result = sequential.generate_greedy(&prompt_tokens, max_tokens, ignore_eos)?;
        expected.push((id.to_string(), result.tokens, result.finish_reason));
        encoded_cases.push((
            id.to_string(),
            prompt.to_string(),
            prompt_tokens,
            max_tokens,
            ignore_eos,
        ));
    }
    let hello_len = encoded_cases
        .iter()
        .find(|(id, _, _, _, _)| id == "mixed-hello")
        .map(|(_, _, prompt_tokens, _, _)| prompt_tokens.len())
        .context("mixed-hello case missing")?;
    let world_len = encoded_cases
        .iter()
        .find(|(id, _, _, _, _)| id == "mixed-world")
        .map(|(_, _, prompt_tokens, _, _)| prompt_tokens.len())
        .context("mixed-world case missing")?;
    ensure!(
        hello_len == world_len,
        "mixed-serving e2e requires Hello and World to tokenize to the same length so the batch decode path is exercised; got Hello={hello_len}, World={world_len}"
    );
    for (id, _, prompt_tokens, _, _) in &encoded_cases {
        ensure!(
            prompt_tokens.len() == hello_len,
            "mixed-serving e2e requires every main prompt to share one tokenized length so the batch decode path is exercised; case {id} has {}, expected {hello_len}",
            prompt_tokens.len()
        );
    }
    let isolation_id = "mixed-valid-beside-invalid";
    let isolation_prompt = "A valid request beside a rejected request";
    let isolation_prompt_tokens = tokenizer
        .encode(isolation_prompt, false)
        .map_err(|err| anyhow::anyhow!("encode prompt for {isolation_id} failed: {err:?}"))?;
    ensure!(
        !isolation_prompt_tokens.is_empty(),
        "tokenizer returned empty prompt for mixed-serving case {isolation_id}"
    );
    let isolation_expected = sequential.generate_greedy(&isolation_prompt_tokens, 6, false)?;

    let fallback_cases = [
        ("mixed-short-single-row", "Hello", 6usize, false),
        (
            "mixed-long-single-row",
            "A valid request beside a rejected request",
            6usize,
            false,
        ),
    ];
    let mut fallback_encoded = Vec::with_capacity(fallback_cases.len());
    let mut fallback_expected = Vec::with_capacity(fallback_cases.len());
    for (id, prompt, max_tokens, ignore_eos) in fallback_cases {
        let prompt_tokens = tokenizer
            .encode(prompt, false)
            .map_err(|err| anyhow::anyhow!("encode prompt for {id} failed: {err:?}"))?;
        ensure!(
            !prompt_tokens.is_empty(),
            "tokenizer returned empty prompt for mixed-serving case {id}"
        );
        let result = sequential.generate_greedy(&prompt_tokens, max_tokens, ignore_eos)?;
        fallback_expected.push((id.to_string(), result.tokens, result.finish_reason));
        fallback_encoded.push((id.to_string(), prompt_tokens, max_tokens, ignore_eos));
    }
    let fallback_prompt_lengths: Vec<_> = fallback_encoded
        .iter()
        .map(|(_, prompt_tokens, _, _)| prompt_tokens.len())
        .collect();
    ensure!(
        fallback_prompt_lengths
            .windows(2)
            .any(|pair| pair[0] != pair[1]),
        "mixed-serving fallback e2e requires at least two prompt lengths so single-row decode is exercised; got {fallback_prompt_lengths:?}"
    );
    drop(sequential);

    let handle = pegainfer_deepseek_v2_lite::start_engine(
        model_path,
        EngineLoadOptions {
            enable_cuda_graph: false,
            device_ordinals: vec![0, 1],
            seed: 42,
            ..EngineLoadOptions::default()
        },
    )?;

    let mut requests = Vec::with_capacity(encoded_cases.len());
    let mut receivers = Vec::with_capacity(encoded_cases.len());
    for (id, _prompt, prompt_tokens, max_tokens, ignore_eos) in encoded_cases {
        let (token_tx, token_rx) = TokenSink::standalone();
        let req = GenerateRequest {
            trace_parent: None,
            request_id: Some(id.clone()),
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens,
            params: SamplingParams {
                ignore_eos,
                ..SamplingParams::default()
            },
            stop_policy: request_stop_policy(ignore_eos),
            max_tokens,
            lora_adapter: None,
            kv_transfer_params: None,
            token_tx,
            logprobs: 0,
            echo: false,
        };
        receivers.push((id, token_rx));
        requests.push(req);
    }
    submit_concurrently(&handle, requests)?;

    let mut actual = Vec::with_capacity(receivers.len());
    for (id, mut token_rx) in receivers {
        let (tokens, finish_reason) = drain_engine_stream(&id, &mut token_rx)?;
        actual.push((id, tokens, finish_reason));
    }

    ensure!(
        actual == expected,
        "mixed-serving output drift on {model_path_label}: actual={actual:?} expected={expected:?}"
    );

    run_mixed_serving_position_fallback(
        &handle,
        fallback_encoded,
        &fallback_expected,
        &mut actual,
    )?;

    run_mixed_serving_rejection_isolation(
        &handle,
        isolation_id,
        isolation_prompt_tokens,
        &isolation_expected.tokens,
        isolation_expected.finish_reason,
        &mut actual,
    )?;

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema_version": 1,
            "kind": "deepseek_v2_lite_ep2_mixed_serving_correctness",
            "report_intent": "correctness_integration",
            "schema": 1,
            "report_type": "deepseek-v2-lite-ep2-mixed-serving-e2e",
            "model_path": model_path_label,
            "ep_backend": current_backend(),
            "case_count": actual.len(),
            "cases": actual
                .iter()
                .map(|(id, tokens, finish_reason)| {
                    serde_json::json!({
                        "id": id,
                        "generated_tokens": tokens.len(),
                        "output_token_sha256": token_sha256(tokens),
                        "finish_reason": format!("{finish_reason:?}"),
                    })
                })
                .collect::<Vec<_>>(),
            "claim_boundary": "Correctness and integration evidence for mixed DeepSeek-V2-Lite EP2 request flow and failure isolation. This artifact makes no latency, throughput, serving SLO, soak, or production-readiness claim.",
        }))?
    );
    Ok(())
}

fn run_mixed_serving_position_fallback(
    handle: &pegainfer_frontend::engine::EngineHandle,
    encoded_cases: Vec<(String, Vec<u32>, usize, bool)>,
    expected: &[(String, Vec<u32>, FinishReason)],
    actual: &mut Vec<(String, Vec<u32>, FinishReason)>,
) -> Result<()> {
    let mut requests = Vec::with_capacity(encoded_cases.len());
    let mut receivers = Vec::with_capacity(encoded_cases.len());
    for (id, prompt_tokens, max_tokens, ignore_eos) in encoded_cases {
        let (token_tx, token_rx) = TokenSink::standalone();
        let req = GenerateRequest {
            trace_parent: None,
            request_id: Some(id.clone()),
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens,
            params: SamplingParams {
                ignore_eos,
                ..SamplingParams::default()
            },
            stop_policy: request_stop_policy(ignore_eos),
            max_tokens,
            lora_adapter: None,
            kv_transfer_params: None,
            token_tx,
            logprobs: 0,
            echo: false,
        };
        receivers.push((id, token_rx));
        requests.push(req);
    }
    submit_concurrently(handle, requests)?;

    let mut fallback_actual = Vec::with_capacity(receivers.len());
    for (id, mut token_rx) in receivers {
        let (tokens, finish_reason) = drain_engine_stream(&id, &mut token_rx)?;
        fallback_actual.push((id, tokens, finish_reason));
    }

    ensure!(
        fallback_actual == expected,
        "mixed-serving single-row fallback output drift: actual={fallback_actual:?} expected={expected:?}"
    );
    actual.extend(fallback_actual);
    Ok(())
}

fn submit_concurrently(
    handle: &pegainfer_frontend::engine::EngineHandle,
    requests: Vec<GenerateRequest>,
) -> Result<()> {
    let barrier = Arc::new(Barrier::new(requests.len() + 1));
    let mut threads = Vec::with_capacity(requests.len());
    for req in requests {
        let handle = handle.clone();
        let barrier = Arc::clone(&barrier);
        threads.push(thread::spawn(move || -> Result<()> {
            barrier.wait();
            handle
                .submit(req)
                .map_err(|_| anyhow::anyhow!("DeepSeek-V2-Lite mixed-serving engine closed"))?;
            Ok(())
        }));
    }
    barrier.wait();
    for submit_thread in threads {
        submit_thread
            .join()
            .map_err(|_| anyhow::anyhow!("mixed-serving submit thread panicked"))??;
    }
    Ok(())
}

fn run_mixed_serving_rejection_isolation(
    handle: &pegainfer_frontend::engine::EngineHandle,
    valid_id: &str,
    valid_prompt_tokens: Vec<u32>,
    expected_tokens: &[u32],
    expected_finish_reason: FinishReason,
    actual: &mut Vec<(String, Vec<u32>, FinishReason)>,
) -> Result<()> {
    let (invalid_tx, mut invalid_rx) = TokenSink::standalone();
    let invalid_req = GenerateRequest {
        trace_parent: None,
        request_id: Some("mixed-invalid-logprobs".to_string()),
        queued_at_unix_s: None,
        data_parallel_rank: None,
        prompt_tokens: vec![1, 2, 3],
        params: SamplingParams::default(),
        stop_policy: StopPolicy::default(),
        max_tokens: 4,
        lora_adapter: None,
        kv_transfer_params: None,
        token_tx: invalid_tx,
        logprobs: 1,
        echo: false,
    };

    let (valid_tx, mut valid_rx) = TokenSink::standalone();
    let valid_req = GenerateRequest {
        trace_parent: None,
        request_id: Some(valid_id.to_string()),
        queued_at_unix_s: None,
        data_parallel_rank: None,
        prompt_tokens: valid_prompt_tokens,
        params: SamplingParams::default(),
        stop_policy: StopPolicy::default(),
        max_tokens: 6,
        lora_adapter: None,
        kv_transfer_params: None,
        token_tx: valid_tx,
        logprobs: 0,
        echo: false,
    };
    submit_concurrently(handle, vec![invalid_req, valid_req])?;

    let mut saw_rejection = false;
    while let Some((_tag, event)) = invalid_rx.blocking_recv() {
        match event {
            TokenEvent::Scheduled { .. } => {}
            TokenEvent::Rejected { message, .. } => {
                ensure!(
                    message.contains("logprobs"),
                    "invalid request rejection should mention logprobs, got {message}"
                );
                saw_rejection = true;
                break;
            }
            TokenEvent::Token { .. }
            | TokenEvent::PromptTokens { .. }
            | TokenEvent::KvTransfer { .. }
            | TokenEvent::Finished { .. }
            | TokenEvent::Error { .. } => {
                anyhow::bail!("invalid mixed-serving request reached unexpected event")
            }
        }
    }
    ensure!(
        saw_rejection,
        "invalid mixed-serving request stream closed without Rejected"
    );

    let (tokens, finish_reason) = drain_engine_stream(valid_id, &mut valid_rx)?;
    ensure!(
        tokens == expected_tokens && finish_reason == expected_finish_reason,
        "valid mixed-serving request drifted next to invalid request: actual_tokens={tokens:?} expected_tokens={expected_tokens:?} actual_finish={finish_reason:?} expected_finish={expected_finish_reason:?}"
    );
    ensure!(
        !tokens.is_empty(),
        "valid mixed-serving request produced no tokens after neighboring rejection"
    );
    actual.push((valid_id.to_string(), tokens, finish_reason));
    Ok(())
}

fn drain_engine_stream(
    case_id: &str,
    token_rx: &mut TokenStreamReceiver,
) -> Result<(Vec<u32>, FinishReason)> {
    let mut tokens = Vec::new();
    loop {
        match token_rx.blocking_recv() {
            Some((
                _tag,
                TokenEvent::Scheduled { .. }
                | TokenEvent::PromptTokens { .. }
                | TokenEvent::KvTransfer { .. },
            )) => {}
            Some((_tag, TokenEvent::Token { id, .. })) => tokens.push(id),
            Some((_tag, TokenEvent::Finished { finish_reason, .. })) => {
                return Ok((tokens, finish_reason));
            }
            Some((_tag, TokenEvent::Error { message, .. })) => {
                anyhow::bail!("mixed-serving case {case_id} failed: {message}")
            }
            Some((_tag, TokenEvent::Rejected { message, .. })) => {
                anyhow::bail!("mixed-serving case {case_id} rejected: {message}")
            }
            None => anyhow::bail!("mixed-serving case {case_id} stream closed before Finished"),
        }
    }
}

fn load_case_set_from_env() -> Result<Option<(PathBuf, Vec<GateCase>)>> {
    let Ok(raw_path) = env::var(E2E_CASE_SET_ENV) else {
        return Ok(None);
    };
    if raw_path.is_empty() {
        return Ok(None);
    }
    let path = resolve_workspace_path(PathBuf::from(raw_path));
    let file = fs::File::open(&path).with_context(|| format!("open {}", path.display()))?;
    let case_set: CaseSet =
        serde_json::from_reader(file).with_context(|| format!("parse {}", path.display()))?;
    ensure!(
        !case_set.cases.is_empty(),
        "{} must contain at least one case",
        path.display()
    );
    for case in &case_set.cases {
        validate_case(case)?;
    }
    Ok(Some((path, case_set.cases)))
}

fn validate_case(case: &GateCase) -> Result<()> {
    ensure!(!case.id.is_empty(), "case id must not be empty");
    ensure!(
        !case.prompt.is_empty(),
        "case {} prompt must not be empty",
        case.id
    );
    ensure!(
        case.output_len > 0,
        "case {} output_len must be positive",
        case.id
    );
    ensure!(
        (1..=8).contains(&case.batch_size),
        "case {} batch_size must be in 1..=8, got {}",
        case.id,
        case.batch_size
    );
    if case.batch_size > 1 {
        ensure!(
            case.ignore_eos,
            "case {} batch_size={} requires ignore_eos=true",
            case.id,
            case.batch_size
        );
    }
    Ok(())
}

fn validate_generation_stats(
    stats: &pegainfer_deepseek_v2_lite::GenerationStats,
    case: &GateCase,
    prompt_tokens_per_row: usize,
    generated_tokens_per_row: usize,
) -> Result<()> {
    ensure!(
        stats.ep_size == 2,
        "case {} expected ep_size=2, got {}",
        case.id,
        stats.ep_size
    );
    ensure!(
        stats.device_ordinals == vec![0, 1],
        "case {} expected devices [0, 1], got {:?}",
        case.id,
        stats.device_ordinals
    );
    ensure!(
        stats.ep_backend == current_backend(),
        "case {} backend mismatch: got {}, expected {}",
        case.id,
        stats.ep_backend,
        current_backend()
    );
    let expected_prompt_tokens = prompt_tokens_per_row * case.batch_size;
    ensure!(
        stats.prompt_tokens == expected_prompt_tokens,
        "case {} prompt token count drift: got {}, expected {}",
        case.id,
        stats.prompt_tokens,
        expected_prompt_tokens
    );
    let expected_generated_tokens = generated_tokens_per_row * case.batch_size;
    ensure!(
        stats.generated_tokens == expected_generated_tokens,
        "case {} generated token count drift: got {}, expected {}",
        case.id,
        stats.generated_tokens,
        expected_generated_tokens
    );

    match stats.ep_backend.as_str() {
        "host-staged" => {
            ensure!(
                stats.host_dispatch_calls > 0
                    && stats.host_combine_calls == stats.host_dispatch_calls
                    && stats.host_dispatch_elements > 0
                    && stats.host_combine_elements == stats.host_dispatch_elements,
                "case {} host-staged EP gate did not record dispatch/combine counts",
                case.id
            );
            ensure!(
                stats.host_dispatch_remote_routes > 0,
                "case {} host-staged EP gate did not exercise any remote routed expert",
                case.id
            );
            ensure!(
                stats.host_dispatch_local_routes > 0,
                "case {} host-staged EP gate did not exercise any local routed expert",
                case.id
            );
            ensure!(
                stats.nccl_dense_exchange_calls == 0
                    && stats.nccl_combine_calls == 0
                    && stats.nccl_dense_exchange_elements == 0
                    && stats.nccl_combine_elements == 0,
                "case {} host-staged EP gate unexpectedly recorded NCCL collectives",
                case.id
            );
        }
        "nccl" => {
            ensure!(
                stats.nccl_dispatch_remote_routes > 0,
                "case {} NCCL EP gate did not exercise any remote routed expert",
                case.id
            );
            ensure!(
                stats.nccl_dispatch_local_routes > 0,
                "case {} NCCL EP gate did not exercise any local routed expert",
                case.id
            );
            ensure!(
                stats.nccl_combine_routes
                    == stats.nccl_dispatch_local_routes + stats.nccl_dispatch_remote_routes,
                "case {} NCCL combine route accounting drift",
                case.id
            );
            ensure!(
                stats.nccl_dense_exchange_calls > 0
                    && stats.nccl_combine_calls == stats.nccl_dense_exchange_calls,
                "case {} NCCL EP gate did not record matching dense exchange/combine calls: exchange={}, combine={}",
                case.id,
                stats.nccl_dense_exchange_calls,
                stats.nccl_combine_calls
            );
            ensure!(
                stats.nccl_dense_exchange_elements > 0
                    && stats.nccl_combine_elements == stats.nccl_dense_exchange_elements,
                "case {} NCCL EP gate did not record matching dense exchange/combine elements: exchange={}, combine={}",
                case.id,
                stats.nccl_dense_exchange_elements,
                stats.nccl_combine_elements
            );
        }
        other => anyhow::bail!(
            "case {} unexpected DeepSeek-V2-Lite EP backend in E2E: {other}",
            case.id
        ),
    }
    Ok(())
}

fn ep_payload(stats: &pegainfer_deepseek_v2_lite::GenerationStats) -> serde_json::Value {
    serde_json::json!({
        "host_dispatch_calls": stats.host_dispatch_calls,
        "host_dispatch_elements": stats.host_dispatch_elements,
        "host_combine_calls": stats.host_combine_calls,
        "host_combine_elements": stats.host_combine_elements,
        "host_dispatch_local_routes": stats.host_dispatch_local_routes,
        "host_dispatch_remote_routes": stats.host_dispatch_remote_routes,
        "nccl_dispatch_local_routes": stats.nccl_dispatch_local_routes,
        "nccl_dispatch_remote_routes": stats.nccl_dispatch_remote_routes,
        "nccl_combine_routes": stats.nccl_combine_routes,
        "nccl_dense_exchange_calls": stats.nccl_dense_exchange_calls,
        "nccl_combine_calls": stats.nccl_combine_calls,
        "nccl_dense_exchange_elements": stats.nccl_dense_exchange_elements,
        "nccl_combine_elements": stats.nccl_combine_elements,
    })
}

fn decode_tokens(
    tokenizer: &HuggingFaceTokenizer,
    tokens: &[u32],
    case_id: &str,
) -> Result<String> {
    tokenizer
        .decode(tokens, false)
        .map_err(|err| anyhow::anyhow!("decode output for case {case_id} failed: {err:?}"))
}

fn token_sha256(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

fn sha256_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
}

fn write_payload_if_requested(payload_text: &str) -> Result<()> {
    if let Ok(path) = env::var(E2E_JSON_OUT_ENV) {
        if !path.is_empty() {
            let path = PathBuf::from(path);
            let path = resolve_workspace_path(path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("create {}", parent.display()))?;
            }
            fs::write(&path, format!("{payload_text}\n"))
                .with_context(|| format!("write {}", path.display()))?;
        }
    }
    Ok(())
}

fn default_batch_size() -> usize {
    1
}

fn matched_expected_output_oracle(token_sha256: &str, text_sha256: &str) -> Option<&'static str> {
    EXPECTED_OUTPUT_SHA256_PAIRS
        .iter()
        .find(|(expected_token, expected_text, _)| {
            token_sha256 == *expected_token && text_sha256 == *expected_text
        })
        .map(|(_, _, source)| *source)
}

fn current_backend() -> String {
    env::var("PEGAINFER_DSV2_LITE_EP_BACKEND").unwrap_or_else(|_| "host-staged".to_string())
}

fn resolve_model_path(raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.join("config.json").exists() {
        return path;
    }
    let workspace_path = resolve_workspace_path(path.clone());
    if workspace_path.join("config.json").exists() {
        return workspace_path;
    }
    path
}

fn resolve_workspace_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path;
    }
    workspace_root().join(path)
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("model crate must live under the workspace root")
        .to_path_buf()
}
