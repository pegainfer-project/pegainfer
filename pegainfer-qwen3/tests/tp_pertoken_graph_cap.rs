//! TP=2 regression for the PerToken CUDA-Graph bucket cap.
//!
//! TP startup pre-captures allowed graph buckets on every rank. The test checks
//! bucket 32 replay and eager fallback for batch 33, which maps to bucket 40.

use pegainfer_core::sampler::SamplingParams;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::per_token_served;
use pegainfer_kernels::ops::reset_numeric_policy_counters;
use pegainfer_kernels::ops::set_numeric_policy;
use pegainfer_qwen3::runtime::DecodePlan;
use pegainfer_qwen3::runtime::DecodeStepItem;
use pegainfer_qwen3::runtime::PrefillPlan;
use pegainfer_qwen3::runtime::PrefillStepItem;
use pegainfer_qwen3::runtime::Qwen3Executor;
use pegainfer_qwen3::runtime::RequestId;

const LOGPROBS: usize = 0;
const MAX_OUTPUT_TOKENS: usize = 4;
const PROMPT_LEN: usize = 8;
const AT_CAP_BATCH: usize = 32;
const ABOVE_CAP_BATCH: usize = 33;

/// Reslove the model path or skip this GPU integration test
fn model_path_or_skip() -> Option<String> {
    let Ok(path) = std::env::var("PEGAINFER_TEST_MODEL_PATH") else {
        eprintln!("skipping tp_pertoken_graph_cap: set PEGAINFER_TEST_MODEL_PATH");
        return None;
    };
    Some(path)
}

/// Number of CUDA devices visible to this test process
fn cuda_device_count() -> usize {
    cudarc::driver::CudaContext::device_count().map_or(0, |count| count.max(0) as usize)
}

/// Build a deterministic short prompt
fn prompt(seed: u32) -> Vec<u32> {
    (0..PROMPT_LEN as u32)
        .map(|position| 1000 + (seed * 131 + position * 7) % 50000)
        .collect()
}

/// Prefill one batch and return request ids plus first generated tokens
fn prefill_batch(
    executor: &mut Qwen3Executor,
    batch_size: usize,
    request_base: u64,
) -> (Vec<RequestId>, Vec<u32>) {
    let requests: Vec<(RequestId, Vec<u32>)> = (0..batch_size)
        .map(|index| {
            (
                RequestId::new(request_base + index as u64),
                prompt(index as u32),
            )
        })
        .collect();

    let prefill_items: Vec<PrefillStepItem> = requests
        .iter()
        .map(|(request_id, tokens)| {
            PrefillStepItem::new(
                *request_id,
                tokens.clone(),
                MAX_OUTPUT_TOKENS,
                SamplingParams::default(),
                LOGPROBS,
                false,
            )
        })
        .collect();

    let result = executor
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &prefill_items,
            echo: false,
        })
        .expect("TP PerToken prefill");

    assert_eq!(result.requests.len(), batch_size);
    assert!(
        result.requests.iter().all(|request| request.completed),
        "short probe prompt prompt unexpectedly required chunked prefill"
    );

    let ids = requests
        .into_iter()
        .map(|(request_id, _)| request_id)
        .collect();

    let tokens = result
        .requests
        .into_iter()
        .map(|request| request.first_token)
        .collect();

    (ids, tokens)
}

/// Execute one decode step and return sampled token ids.
fn decode_batch(
    executor: &mut Qwen3Executor,
    request_ids: &[RequestId],
    tokens: &[u32],
) -> Vec<u32> {
    let decode_items: Vec<DecodeStepItem> = request_ids
        .iter()
        .zip(tokens.iter().copied())
        .map(|(request_id, token)| {
            DecodeStepItem::new(*request_id, token, SamplingParams::default(), LOGPROBS)
        })
        .collect();

    executor
        .execute_decode(DecodePlan {
            sample_seed: 0,
            requests: &decode_items,
        })
        .expect("TP PerToken decode")
        .requests
        .into_iter()
        .map(|request| request.token)
        .collect()
}

/// Measure PerToken GEMM calls across two decode steps.
fn per_token_counter_probe(
    executor: &mut Qwen3Executor,
    batch_size: usize,
    request_base: u64,
) -> (u64, u64) {
    let (request_ids, mut tokens) = prefill_batch(executor, batch_size, request_base);

    // Measure decode GEMMs only, not prefill GEMMs
    reset_numeric_policy_counters();

    tokens = decode_batch(executor, &request_ids, &tokens);
    let after_first_decode = per_token_served();

    let _ = decode_batch(executor, &request_ids, &tokens);
    let after_second_decode = per_token_served();

    for request_id in request_ids {
        executor
            .drop_request(request_id)
            .expect("drop TP PerToken probe request");
    }

    (after_first_decode, after_second_decode)
}

/// Verify TP2 startup capture at bucket 32 and eager fallback above the cap.
#[test]
fn tp2_pertoken_graph_cap() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };

    let gpus = cuda_device_count();
    if gpus < 2 {
        eprintln!("skipping TP2 test: needs at least two GPUs, have {gpus}");
        return;
    }

    // The policy must be set before TP graph pre-capture starts.
    set_numeric_policy(NumericPolicy::PerToken);

    let mut executor = Qwen3Executor::from_runtime(&model_path, true, &[0, 1])
        .expect("build TP2 PerToken graph executor");

    executor.set_prefix_cache_enabled(false);

    let (graph_first, graph_second) = per_token_counter_probe(&mut executor, AT_CAP_BATCH, 10_000);

    assert_eq!(
        graph_first, 0,
        "TP batch 32 ran PerToken GEMMs after startup capture"
    );
    assert_eq!(
        graph_second, 0,
        "TP batch 32 ran PerToken GEMMs on the second decode"
    );

    let (eager_first, eager_second) =
        per_token_counter_probe(&mut executor, ABOVE_CAP_BATCH, 20_000);

    assert!(eager_first > 0);
    assert!(
        eager_second > eager_first,
        "TP batch 33 did not execute eager GEMMs twice"
    );

    eprintln!(
        "TP2 PerToken graph cap probe: bs=32 served {graph_first}->{graph_second}, \
        bs=33 served {eager_first}->{eager_second}"
    );
}
