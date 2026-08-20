//! Shared fixtures for the scheduler module tests.

use pegainfer_frontend::engine::EosPolicy;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::StopCause;
use pegainfer_frontend::engine::StopPolicy;
use pegainfer_kv_store::BlockPool;
use pegainfer_kv_store::RequestKv;

use super::PAGE;
use super::slot::Glm52SlotState;
use super::slot::Glm52StepOutcome;

pub(super) const EOS: &[u32] = &[7];

pub(super) fn state(prompt: Vec<u32>, max_tokens: usize, ignore_eos: bool) -> Glm52SlotState {
    Glm52SlotState::new(prompt, max_tokens, stop_policy(ignore_eos), 0)
}

pub(super) fn stop_policy(ignore_eos: bool) -> StopPolicy {
    StopPolicy {
        eos: if ignore_eos {
            EosPolicy::Ignore
        } else {
            EosPolicy::ModelDefault
        },
        token_ids: Vec::new(),
    }
}

/// A standalone `RequestKv` for tests that never schedule KV (the pool
/// is leaked so the kvbm internals outlive the test value).
pub(super) fn test_kv(prompt: Vec<u32>, max_tokens: usize) -> RequestKv {
    let pool: &'static BlockPool = Box::leak(Box::new(BlockPool::new(PAGE, 64)));
    pool.new_request(prompt, max_tokens, None)
}

pub(super) fn commit(
    committed: &[u32],
    emit: usize,
    finish: Option<FinishReason>,
    context_rows: usize,
) -> Glm52StepOutcome {
    let stop_cause = matches!(finish, Some(FinishReason::Stop))
        .then(|| StopCause::Eos(*committed.last().expect("a stop commit has a trigger token")));
    Glm52StepOutcome::Commit {
        committed: committed.to_vec(),
        emit,
        finish,
        stop_cause,
        context_rows,
    }
}

pub(super) fn request(
    prompt: Vec<u32>,
    params: pegainfer_sample::SamplingParams,
    max_tokens: usize,
) -> GenerateRequest {
    let (token_tx, _token_rx) = pegainfer_frontend::engine::TokenSink::standalone();
    let stop_policy = stop_policy(params.ignore_eos);
    GenerateRequest {
        trace_parent: None,
        request_id: None,
        queued_at_unix_s: None,
        data_parallel_rank: None,
        prompt_tokens: prompt,
        params,
        stop_policy,
        max_tokens,
        lora_adapter: None,
        kv_transfer_params: None,
        token_tx,
        logprobs: 0,
        echo: false,
    }
}

pub(super) fn sampled(temperature: f32) -> pegainfer_sample::SamplingParams {
    pegainfer_sample::SamplingParams {
        temperature,
        ..Default::default()
    }
}
