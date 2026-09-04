use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::thread;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use crossbeam_channel as channel;
use cudarc::driver::DevicePtr;
use pegainfer_core::cuda_graph::CudaGraphDumpSummary;
use pegainfer_core::kv_pool::KvLayout;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_core::weight_loader::WeightPrefetch;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_frontend::engine::DeferredFinish;
use pegainfer_frontend::engine::LoadLoraAdapterRequest;
use pegainfer_frontend::engine::SpecDecodeCounters;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::engine::UnloadLoraAdapterRequest;
use pegainfer_frontend::engine::panic_message;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kv_cache::KvBlockGuard;
use pegainfer_kv_cache::KvBuffer;
use pegainfer_kv_cache::KvCacheManager;
use pegainfer_kv_cache::KvView;
use pegainfer_kv_cache::LoadReservation;
use pegainfer_kv_cache::PrefixProbe;
use pegainfer_kv_offload::LoadHandle;
use pegainfer_kv_offload::OffloadConfig;
use pegainfer_kv_offload::OffloadEngine;

use crate::Qwen3LoraOptions;
use crate::Qwen3OffloadOptions;
use crate::batch_decode::DecodeGraphUse;
use crate::batch_decode_buffers::BATCH_BUCKETS;
use crate::batch_decode_buffers::BatchDecodeBuffers;
use crate::config::Config;
use crate::config::TensorParallelConfig;
use crate::weights::KvBudget;
use crate::weights::ModelRuntimeConfig;
use crate::weights::Qwen3MemoryOptions;
use crate::weights::Qwen3Model;

mod auto_hedge;
mod dflash_lane;
mod dflash_prefill;
mod remote_fetch;
use remote_fetch::QueryView;
use remote_fetch::RemoteFetchAction;
use remote_fetch::remote_fetch_action;
mod spec;

use dflash_lane::DFlashLaneState;
use dflash_prefill::DFlashPrefillAction;
use dflash_prefill::dflash_prefill_action;
/// The contract's request id, end to end: the wiring mints it at submit and
/// every internal queue, effect, and KV key uses it unchanged — there is no
/// second id space to map across.
pub use pegainfer_frontend::engine::RequestId;

use crate::dflash::DFlashDraftModel;
use crate::speculative::DraftPlan;
use crate::speculative::DraftResult;
use crate::speculative::DraftStepItem;
use crate::speculative::VerifyPlan;
use crate::speculative::VerifyRequestResult;
use crate::speculative::VerifyResult;
use crate::speculative::VerifyStepItem;
use crate::speculative::build_verify_results;
use crate::verify_graph::VerifyGraphBuffers;

#[derive(Clone)]
pub struct PrefillStepItem {
    pub(crate) request_id: RequestId,
    pub(crate) prompt_tokens: Vec<u32>,
    pub(crate) max_output_tokens: usize,
    pub(crate) params: SamplingParams,
    pub(crate) logprobs: usize,
    pub(crate) echo: bool,
    pub(crate) lora_adapter: Option<String>,
    /// Leading prompt tokens whose KV came from the prefix cache.
    /// Set by the executor after matching; the forward pass only computes
    /// the remaining suffix.
    pub(crate) cached_tokens: usize,
    /// Scheduler-set cap on prompt tokens forwarded this step (chunked
    /// prefill). The executor clamps it to the tokens actually remaining.
    pub(crate) chunk_budget: usize,
    /// First prompt position forwarded this step. Set by the executor from
    /// the request's KV position (covers both prefix-cache hits and chunks
    /// applied in earlier steps).
    pub(crate) chunk_start: usize,
    /// Prompt tokens forwarded this step. Set by the executor.
    pub(crate) chunk_tokens: usize,
}

impl PrefillStepItem {
    pub fn new(
        request_id: RequestId,
        prompt_tokens: Vec<u32>,
        max_output_tokens: usize,
        params: SamplingParams,
        logprobs: usize,
        echo: bool,
    ) -> Self {
        let chunk_tokens = prompt_tokens.len();
        Self {
            request_id,
            prompt_tokens,
            max_output_tokens,
            params,
            logprobs,
            echo,
            lora_adapter: None,
            cached_tokens: 0,
            chunk_budget: usize::MAX,
            chunk_start: 0,
            chunk_tokens,
        }
    }

    #[must_use]
    pub fn with_lora_adapter(mut self, lora_adapter: Option<String>) -> Self {
        self.lora_adapter = lora_adapter;
        self
    }

    /// Cap the prompt tokens forwarded per `execute_prefill` call; re-issue the
    /// item until its result is `completed` to prefill a long prompt in chunks.
    #[must_use]
    pub fn with_chunk_budget(mut self, chunk_budget: usize) -> Self {
        self.chunk_budget = chunk_budget;
        self
    }

    /// Prompt tokens forwarded this step.
    fn as_slice(&self) -> &[u32] {
        &self.prompt_tokens[self.chunk_start..self.chunk_start + self.chunk_tokens]
    }

    /// Whether this step's chunk reaches the end of the prompt (and so
    /// produces the first generated token).
    fn is_final_chunk(&self) -> bool {
        self.chunk_start + self.chunk_tokens == self.prompt_tokens.len()
    }
}

#[derive(Clone)]
pub struct DecodeStepItem {
    pub(crate) request_id: RequestId,
    pub(crate) token_id: u32,
    pub(crate) params: SamplingParams,
    pub(crate) logprobs: usize,
    pub(crate) lora_adapter: Option<String>,
}

impl DecodeStepItem {
    pub fn new(
        request_id: RequestId,
        token_id: u32,
        params: SamplingParams,
        logprobs: usize,
    ) -> Self {
        Self {
            request_id,
            token_id,
            params,
            logprobs,
            lora_adapter: None,
        }
    }

    #[must_use]
    pub fn with_lora_adapter(mut self, lora_adapter: Option<String>) -> Self {
        self.lora_adapter = lora_adapter;
        self
    }
}

fn gather_decode_logprobs(
    lane: &LocalQwen3Lane,
    requests: &[DecodeStepItem],
    logits: &HiddenStates,
    row_offset: usize,
    tokens: &[u32],
) -> Result<Vec<Option<TokenLogprob>>> {
    let wanted: Vec<usize> = requests
        .iter()
        .enumerate()
        .filter(|(_, req)| req.logprobs > 0)
        .map(|(i, _)| i)
        .collect();
    let lp_requests: Vec<pegainfer_sample::LogprobRequest> = wanted
        .iter()
        .map(|&i| pegainfer_sample::LogprobRequest {
            row: row_offset + i,
            picked: tokens[row_offset + i],
            top_k: requests[i].logprobs,
        })
        .collect();
    let results =
        pegainfer_sample::token_logprobs_batch(lane.model.device_ctx(), logits, &lp_requests)?;
    let mut logprobs: Vec<Option<TokenLogprob>> = vec![None; requests.len()];
    for (i, lp) in wanted.into_iter().zip(results) {
        logprobs[i] = Some(lp);
    }
    Ok(logprobs)
}

fn build_prefill_request_results(
    lane: &LocalQwen3Lane,
    requests: &[PrefillStepItem],
    logits: &HiddenStates,
    tokens: &[u32],
    all_position_logits: Option<&HiddenStates>,
    compute_prompt_logprobs: bool,
) -> Result<Vec<PrefillRequestResult>> {
    let ctx = lane.model.device_ctx();

    let first_token_wanted: Vec<usize> = requests
        .iter()
        .enumerate()
        .filter(|(_, req)| req.is_final_chunk() && req.logprobs > 0)
        .map(|(i, _)| i)
        .collect();
    let first_token_requests: Vec<pegainfer_sample::LogprobRequest> = first_token_wanted
        .iter()
        .map(|&i| pegainfer_sample::LogprobRequest {
            row: i,
            picked: tokens[i],
            top_k: requests[i].logprobs,
        })
        .collect();
    let mut first_token_logprobs: Vec<Option<TokenLogprob>> = vec![None; requests.len()];
    for (i, lp) in first_token_wanted
        .into_iter()
        .zip(pegainfer_sample::token_logprobs_batch(
            ctx,
            logits,
            &first_token_requests,
        )?)
    {
        first_token_logprobs[i] = Some(lp);
    }

    let mut prompt_requests: Vec<pegainfer_sample::LogprobRequest> = Vec::new();
    if compute_prompt_logprobs && all_position_logits.is_some() {
        let mut token_offset = 0usize;
        for req in requests {
            if req.echo {
                for j in 1..req.prompt_tokens.len() {
                    prompt_requests.push(pegainfer_sample::LogprobRequest {
                        row: token_offset + j - 1,
                        picked: req.prompt_tokens[j],
                        top_k: req.logprobs,
                    });
                }
            }
            token_offset += req.chunk_tokens;
        }
    }
    let mut prompt_results = match all_position_logits {
        Some(all_logits) if !prompt_requests.is_empty() => Some(
            pegainfer_sample::token_logprobs_batch(ctx, all_logits, &prompt_requests)?.into_iter(),
        ),
        _ => None,
    };

    let mut outputs = Vec::with_capacity(requests.len());
    for (i, req) in requests.iter().enumerate() {
        let completed = req.is_final_chunk();
        let prompt_logprobs = if req.echo {
            if compute_prompt_logprobs {
                let mut echo_logprobs: Vec<Option<TokenLogprob>> =
                    Vec::with_capacity(req.prompt_tokens.len());
                echo_logprobs.push(None);
                match &mut prompt_results {
                    Some(results) => {
                        for _ in 1..req.prompt_tokens.len() {
                            echo_logprobs.push(results.next());
                        }
                    }
                    None => echo_logprobs.resize(req.prompt_tokens.len(), None),
                }
                Some(echo_logprobs)
            } else {
                Some(vec![None; req.prompt_tokens.len()])
            }
        } else {
            None
        };
        outputs.push(PrefillRequestResult {
            request_id: req.request_id,
            first_token: tokens[i],
            first_token_logprob: first_token_logprobs[i].take(),
            prompt_logprobs,
            cached_tokens: req.cached_tokens,
            completed,
            prefill_pos: req.chunk_start + req.chunk_tokens,
        });
    }
    Ok(outputs)
}

fn build_decode_request_results(
    lane: &LocalQwen3Lane,
    requests: &[DecodeStepItem],
    logits: &HiddenStates,
    row_offset: usize,
    tokens: &[u32],
) -> Result<Vec<DecodeRequestResult>> {
    let mut logprobs = gather_decode_logprobs(lane, requests, logits, row_offset, tokens)?;
    Ok(requests
        .iter()
        .enumerate()
        .map(|(i, req)| DecodeRequestResult {
            request_id: req.request_id,
            token: tokens[row_offset + i],
            logprob: logprobs[i].take(),
        })
        .collect())
}

fn build_batch_decode_request_results(
    lane: &mut LocalQwen3Lane,
    requests: &[DecodeStepItem],
    sample_seed: u64,
) -> Result<Vec<DecodeRequestResult>> {
    let params: Vec<&SamplingParams> = requests.iter().map(|req| &req.params).collect();
    lane.steps_buf.clear();
    lane.steps_buf.resize(params.len(), 0);
    let tokens = pegainfer_sample::select_batch(
        lane.model.device_ctx(),
        &lane.bufs.logits,
        &params,
        &lane.steps_buf,
        sample_seed,
        &mut lane.sample_scratch,
    )?;

    let mut logprobs = gather_decode_logprobs(lane, requests, &lane.bufs.logits, 0, &tokens)?;
    Ok(requests
        .iter()
        .enumerate()
        .map(|(i, req)| DecodeRequestResult {
            request_id: req.request_id,
            token: tokens[i],
            logprob: logprobs[i].take(),
        })
        .collect())
}

fn execute_step_on_lane(
    lane: &mut LocalQwen3Lane,
    step: &StepCommand,
    collect_result: bool,
) -> Result<WorkerStepOutcome> {
    match step {
        StepCommand::Prefill {
            requests,
            kv_views,
            echo,
            sample_seed,
        } => {
            let prompts: Vec<&[u32]> = requests.iter().map(PrefillStepItem::as_slice).collect();
            let lora_adapters: Vec<Option<&str>> = requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();
            // When DFlash is loaded, capture target hidden states for eligible
            // requests so they can seed the draft model after prefill finishes.
            let capture_requested = lane.should_capture_dflash_prefill_context(requests);
            let capture_layer_ids = if capture_requested {
                lane.dflash_capture_layer_ids()
            } else {
                None
            };
            let (logits, all_position_logits, captured_hidden) = lane.execute_prefill(
                &prompts,
                kv_views,
                &lora_adapters,
                *echo,
                capture_layer_ids.as_deref(),
            )?;
            let dflash_context_captured_requests = lane.record_prefill_dflash_context(
                requests,
                capture_requested,
                captured_hidden.as_ref(),
            )?;
            if collect_result {
                let params: Vec<&SamplingParams> = requests.iter().map(|r| &r.params).collect();
                let tokens = lane.select_step_tokens(&logits, &params, *sample_seed)?;
                Ok(WorkerStepOutcome::Prefill(PrefillResult {
                    requests: build_prefill_request_results(
                        lane,
                        requests,
                        &logits,
                        &tokens,
                        all_position_logits.as_ref(),
                        *echo,
                    )?,
                    dflash_context_captured_requests,
                }))
            } else {
                Ok(WorkerStepOutcome::Ack)
            }
        }
        StepCommand::Decode {
            requests,
            kv_views,
            sample_seed,
        } => {
            // Under TP, replay is only safe once the sweep captured every graph;
            // reaching here without it would be a mid-serving capture (#481).
            anyhow::ensure!(
                !lane.model.tp_graph_enabled() || lane.precapture_complete,
                "TP decode with CUDA Graph requires the startup pre-capture sweep"
            );
            let token_ids: Vec<u32> = requests.iter().map(|req| req.token_id).collect();
            let lora_adapters: Vec<Option<&str>> = requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();
            lane.execute_decode(&token_ids, kv_views, &lora_adapters)?;
            if collect_result {
                Ok(WorkerStepOutcome::Decode(DecodeResult {
                    requests: build_batch_decode_request_results(lane, requests, *sample_seed)?,
                }))
            } else {
                Ok(WorkerStepOutcome::Ack)
            }
        }
        StepCommand::Unified {
            prefill_requests,
            prefill_kv_views,
            decode_requests,
            decode_kv_views,
            sample_seed,
        } => {
            let prefill_prompts: Vec<&[u32]> = prefill_requests
                .iter()
                .map(PrefillStepItem::as_slice)
                .collect();
            let decode_tokens: Vec<u32> = decode_requests.iter().map(|req| req.token_id).collect();
            let prefill_lora_adapters: Vec<Option<&str>> = prefill_requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();
            let decode_lora_adapters: Vec<Option<&str>> = decode_requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();
            let logits = lane.execute_unified(
                &prefill_prompts,
                prefill_kv_views,
                &prefill_lora_adapters,
                &decode_tokens,
                decode_kv_views,
                &decode_lora_adapters,
            )?;
            if collect_result {
                // Logits columns: prefill requests first, then decode rows.
                let params: Vec<&SamplingParams> = prefill_requests
                    .iter()
                    .map(|r| &r.params)
                    .chain(decode_requests.iter().map(|r| &r.params))
                    .collect();
                let tokens = lane.select_step_tokens(&logits, &params, *sample_seed)?;
                Ok(WorkerStepOutcome::Unified(UnifiedResult {
                    prefill_requests: build_prefill_request_results(
                        lane,
                        prefill_requests,
                        &logits,
                        &tokens,
                        None,
                        false,
                    )?,
                    decode_requests: build_decode_request_results(
                        lane,
                        decode_requests,
                        &logits,
                        prefill_requests.len(),
                        &tokens,
                    )?,
                }))
            } else {
                Ok(WorkerStepOutcome::Ack)
            }
        }
        StepCommand::SplitConcurrent {
            prefill_requests,
            prefill_kv_views,
            decode_requests,
            decode_kv_views,
            prefill_stream,
            decode_stream,
            sample_seed,
        } => {
            use pegainfer_kernels::tensor::StreamOverrideGuard;

            anyhow::ensure!(
                lane.inflight_prefill.is_none(),
                "SplitConcurrent dispatched with an unresolved in-flight prefill"
            );

            let prefill_prompts: Vec<&[u32]> = prefill_requests
                .iter()
                .map(PrefillStepItem::as_slice)
                .collect();
            let prefill_lora_adapters: Vec<Option<&str>> = prefill_requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();
            let decode_tokens: Vec<u32> = decode_requests.iter().map(|req| req.token_id).collect();
            let decode_lora_adapters: Vec<Option<&str>> = decode_requests
                .iter()
                .map(|req| req.lora_adapter.as_deref())
                .collect();

            // Declared before the bin so that on any early return (`?` or panic)
            // the bin drops first: its Drop synchronizes the prefill stream before
            // `prefill_logits` and the parked temporaries the bin owns are freed.
            let prefill_logits;
            let mut prefill_temp_bin = crate::prefill::PrefillTempBin::armed(prefill_stream.0);
            {
                let _prefill_override = unsafe { StreamOverrideGuard::activate(prefill_stream.0) };
                let (logits, _, _) = lane.execute_prefill(
                    &prefill_prompts,
                    prefill_kv_views,
                    &prefill_lora_adapters,
                    false,
                    None,
                )?;
                prefill_logits = logits;
            }
            // Close the prefill parking window before decode, so decode's own
            // temporaries don't land in it.
            prefill_temp_bin.close();

            {
                let _decode_guard = DecodeStreamGuard {
                    stream: decode_stream.0,
                };
                let _decode_override = unsafe { StreamOverrideGuard::activate(decode_stream.0) };
                lane.execute_decode(&decode_tokens, decode_kv_views, &decode_lora_adapters)?;
            }

            let decode_result =
                build_batch_decode_request_results(lane, decode_requests, *sample_seed)?;

            let event = lane
                .model
                .device_ctx()
                .ctx
                .new_event(None)
                .map_err(|e| anyhow::anyhow!("cuEventCreate(prefill poll) failed: {e}"))?;
            unsafe {
                // The prefill stream may be a Green Context stream; cuEventRecord
                // needs the event and stream in one context, so record via the
                // green context when the stream has one (stream mode has none).
                let mut gctx: cudarc::driver::sys::CUgreenCtx = std::ptr::null_mut();
                let get = cudarc::driver::sys::cuStreamGetGreenCtx(prefill_stream.0, &raw mut gctx);
                let record = if get != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                    get
                } else if gctx.is_null() {
                    cudarc::driver::sys::cuEventRecord(event.cu_event(), prefill_stream.0)
                } else {
                    cudarc::driver::sys::cuGreenCtxRecordEvent(gctx, event.cu_event())
                };
                anyhow::ensure!(
                    record == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                    "recording prefill poll event failed: {record:?}"
                );
            }

            lane.inflight_prefill = Some(InflightPrefillState {
                temp_bin: prefill_temp_bin,
                prefill_logits,
                prefill_requests: prefill_requests.clone(),
                sample_seed: *sample_seed,
            });

            Ok(WorkerStepOutcome::SplitDecodeReady {
                decode: DecodeResult {
                    requests: decode_result,
                },
                prefill_event: event,
            })
        }
        StepCommand::SpeculativeVerify {
            requests,
            kv_views,
            sample_seed,
        } => {
            // One target forward over each request's K+1 draft span with a
            // speculative KV view. The fixed-buffer verify path computes all-
            // position logits (accept_prefix_match needs the target's committed
            // token at each span position) and captures the target hidden states
            // (at the DFlash layers) to seed the next draft — all into reused,
            // pointer-stable scratch (`VerifyGraphBuffers`).
            let result = lane.execute_dflash_verify(requests, kv_views, *sample_seed)?;
            Ok(WorkerStepOutcome::SpeculativeVerify(result))
        }
        StepCommand::SpeculativeDraft { requests } => Ok(WorkerStepOutcome::SpeculativeDraft(
            lane.execute_dflash_draft(requests)?,
        )),
    }
}

struct CublasThreadGuard;

impl Drop for CublasThreadGuard {
    fn drop(&mut self) {
        unsafe {
            pegainfer_core::ffi::cublas_destroy();
        }
    }
}

fn bind_model_thread(model: &Qwen3Model) -> Result<()> {
    unsafe {
        let err = pegainfer_core::ffi::cuda_set_device(model.device_ctx().device_ordinal as i32);
        if err != 0 {
            return Err(anyhow::anyhow!(
                "Failed to set CUDA device {} on worker thread: cudaError={}",
                model.device_ctx().device_ordinal,
                err
            ));
        }
    }
    model
        .device_ctx()
        .ctx
        .bind_to_thread()
        .map_err(|e| anyhow::anyhow!("Failed to bind CUDA context to thread: {e}"))?;
    unsafe {
        pegainfer_core::ffi::cublas_init();
    }
    Ok(())
}

/// Prepare decode GEMM algos before capture, per the active numeric policy: under `Pin`, eagerly pin
/// one algo per projection {M,K} (reused for all N); otherwise tune the fastest cublasLt algo per
/// decode shape (buckets up to `GEMM_LT_MAX_N`, every layer's weights in the L2-cold timing rotation).
/// Adds startup cost per thread: warmup on every worker, plus the Pin self-check
/// on the serving worker when `run_envelope_check` is true.
fn tune_decode_gemm_algos(
    model: &Qwen3Model,
    max_prefill_tokens: usize,
    run_envelope_check: bool,
) -> Result<()> {
    use pegainfer_kernels::ops::NumericPolicy;
    use pegainfer_kernels::ops::numeric_policy;

    let ctx = model.device_ctx();
    let hidden = model.config().hidden_size;
    let vocab = model.config().vocab_size;
    let q_dim = model.local_q_dim();
    let kv_dim = model.local_kv_dim();
    let intermediate = model.local_intermediate_size();
    let fused_qkv = model.fused_decode_qkv();

    if numeric_policy() == NumericPolicy::Pin {
        // Eager pin before capture: the lazy pin-workspace alloc is illegal mid-capture.
        crate::batch_decode_buffers::warmup_decode_projection_pins(
            hidden,
            q_dim,
            kv_dim,
            intermediate,
            vocab,
        )?;
        let max_context = model.config().max_position_embeddings;
        log::info!(
            "Qwen3 split-KV decode chunk pinned: {} tokens (max_context_tokens={max_context})",
            crate::batch_decode_buffers::pin_chunk_size(max_context)
        );
        // The profile worker skips the full sweep; the long-lived serving worker verifies its own
        // (thread-local) warmed plans before capture, so the envelope is guaranteed pre-serving.
        if run_envelope_check {
            verify_pin_envelope(model, max_prefill_tokens)?;
        }
        return Ok(());
    }

    let layers = &model.layers;

    let q_samples: Vec<_> = layers.iter().map(|l| (&l.attention.qkv_proj, 0)).collect();
    let kv_samples: Vec<_> = layers
        .iter()
        .flat_map(|l| {
            [
                (&l.attention.qkv_proj, q_dim),
                (&l.attention.qkv_proj, q_dim + kv_dim),
            ]
        })
        .collect();
    let o_samples: Vec<_> = layers.iter().map(|l| (&l.attention.o_proj, 0)).collect();
    let gate_up_samples: Vec<_> = layers
        .iter()
        .flat_map(|l| {
            [
                (&l.mlp.gate_up_proj, 0),
                (&l.mlp.gate_up_proj, intermediate),
            ]
        })
        .collect();
    let down_samples: Vec<_> = layers.iter().map(|l| (&l.mlp.down_proj, 0)).collect();
    let lm_head_samples = [(model.output_projection(), 0)];

    for &n in BATCH_BUCKETS.iter().filter(|&&b| b <= ops::GEMM_LT_MAX_N) {
        if fused_qkv {
            ops::gemm_lt_tune(ctx, &q_samples, q_dim + 2 * kv_dim, n)?;
        } else {
            ops::gemm_lt_tune(ctx, &q_samples, q_dim, n)?;
            ops::gemm_lt_tune(ctx, &kv_samples, kv_dim, n)?;
        }
        ops::gemm_lt_tune(ctx, &o_samples, hidden, n)?;
        ops::gemm_lt_tune(ctx, &gate_up_samples, intermediate, n)?;
        ops::gemm_lt_tune(ctx, &down_samples, hidden, n)?;
        ops::gemm_lt_tune(ctx, &lm_head_samples, vocab, n)?;
    }
    Ok(())
}

/// Boot-time Pin envelope check: errors unless, under `Pin`, the pinned algo serves the FULL production N
/// envelope — EVERY reachable N, not a sample — with zero per-token fallback. Each N is checked
/// by the host-side `gemm_lt_pin_check`, so the dense full-N sweep is a startup-only cost. Runs post-warmup,
/// pre-capture, on the GEMM thread (per TP rank via `bind`); `bail!`s naming the first unserved
/// {M,N,K}. Each projection's N upper bound is set below.
fn verify_pin_envelope(model: &Qwen3Model, max_prefill_tokens: usize) -> Result<()> {
    use pegainfer_kernels::ops::gemm_lt_pin_check;

    let hidden = model.config().hidden_size;
    let q_dim = model.local_q_dim();
    let kv_dim = model.local_kv_dim();
    let intermediate = model.local_intermediate_size();
    // Unified-N peak: one prefill chunk (≤ max_prefill_tokens) + (max_bucket−1) concurrent decoders;
    // rests on max_decode_batch_size == BATCH_BUCKETS.last().
    let ceiling = max_prefill_tokens + (*BATCH_BUCKETS.last().unwrap()).saturating_sub(1);
    // lm_head (vocab×hidden) runs on the sampled-position count, not the token count: decode-only
    // pads to a bucket (≤ max_decode_batch_size) and unified gathers ≤ that many requests, while
    // echo/all-position runs up to max_prefill_tokens — true max N = max(max_prefill, max_decode_batch).
    let lm_head_max_n = max_prefill_tokens.max(*BATCH_BUCKETS.last().unwrap());
    let shapes = crate::batch_decode_buffers::decode_projection_pin_shapes(
        hidden,
        q_dim,
        kv_dim,
        intermediate,
        model.config().vocab_size,
    );
    // Per-shape N upper bound, in the same lm_head-last order as `shapes`: the unified-N peak for
    // every projection, the sampled-position count for lm_head.
    let max_ns = [ceiling, ceiling, ceiling, ceiling, ceiling, lm_head_max_n];
    let mut checks = 0usize;
    for (&(m, k), &max_n) in shapes.iter().zip(max_ns.iter()) {
        for n in 1..=max_n {
            if !gemm_lt_pin_check(m, n, k)? {
                anyhow::bail!(
                    "batch-invariant pin self-check FAILED: pinned cuBLASLt algo cannot serve \
                     N={n} at projection {{M={m}, K={k}}} — this GPU/cuBLAS combo cannot serve the \
                     full envelope, so the pinned GEMM would bail at runtime rather than serve it. \
                     Run without --batch-invariant or report the {{M,N,K}}."
                );
            }
            checks += 1;
        }
    }
    log::info!(
        "batch-invariant pin envelope verified: every N up to {ceiling} (lm_head to {lm_head_max_n}), {checks} checks, 0 fallback"
    );
    Ok(())
}

pub struct PrefillPlan<'a> {
    pub requests: &'a [PrefillStepItem],
    pub echo: bool,
    pub sample_seed: u64,
}

pub struct DecodePlan<'a> {
    pub requests: &'a [DecodeStepItem],
    pub sample_seed: u64,
}

pub struct UnifiedPlan<'a> {
    pub prefill_requests: &'a [PrefillStepItem],
    pub decode_requests: &'a [DecodeStepItem],
    pub sample_seed: u64,
}

#[derive(Clone, Debug)]
pub struct PrefillRequestResult {
    pub(crate) request_id: RequestId,
    pub first_token: u32,
    pub first_token_logprob: Option<TokenLogprob>,
    pub(crate) prompt_logprobs: Option<Vec<Option<TokenLogprob>>>,
    /// Prompt tokens served from the prefix cache (KV reused, not recomputed).
    pub cached_tokens: usize,
    /// Whether the prompt is fully prefilled. When false this step ran a
    /// non-final chunk and `first_token` is meaningless.
    pub completed: bool,
    /// Prompt tokens with KV computed after this step (authoritative —
    /// includes prefix-cache hits the scheduler can't see).
    pub(crate) prefill_pos: usize,
}

#[derive(Clone, Debug)]
pub struct DecodeRequestResult {
    pub(crate) request_id: RequestId,
    pub token: u32,
    pub logprob: Option<TokenLogprob>,
}

pub struct PrefillResult {
    pub requests: Vec<PrefillRequestResult>,
    /// Requests whose DFlash target context was captured this prefill step.
    /// Empty unless speculative decoding is enabled. The executor folds these
    /// into its `dflash_ready_requests` set once the prompt is fully prefilled.
    pub(crate) dflash_context_captured_requests: Vec<RequestId>,
}

pub struct DecodeResult {
    pub requests: Vec<DecodeRequestResult>,
}

pub struct UnifiedResult {
    pub prefill_requests: Vec<PrefillRequestResult>,
    pub decode_requests: Vec<DecodeRequestResult>,
}

pub(crate) trait ModelExecutor: Send {
    fn block_size(&self) -> usize;
    fn max_request_blocks(&self) -> usize;
    fn max_context_tokens(&self) -> usize;
    fn max_decode_batch_size(&self) -> usize;
    fn available_blocks(&self) -> usize;
    fn is_stop_token(&self, token_id: u32) -> bool;
    fn drop_request(&mut self, request_id: RequestId) -> Result<()>;

    fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult>;
    fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<DecodeResult>;
    fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult>;

    /// Run one speculative draft round (propose `K` tokens per request). Only
    /// meaningful when [`Self::speculative_enabled`] is true.
    fn execute_speculative_draft(&mut self, _plan: DraftPlan<'_>) -> Result<DraftResult> {
        anyhow::bail!("speculative draft is not implemented for this executor")
    }

    /// Verify a draft span with one target forward and accept the greedy prefix.
    fn execute_speculative_verify(&mut self, _plan: VerifyPlan<'_>) -> Result<VerifyResult> {
        anyhow::bail!("speculative verification is not implemented for this executor")
    }

    /// Whether a draft model is loaded and speculative decoding is active.
    fn speculative_enabled(&self) -> bool {
        false
    }

    /// Cumulative spec-decode acceptance counters, or `None` when no draft
    /// model is loaded.
    fn spec_decode_counters(&self) -> Option<SpecDecodeCounters> {
        None
    }

    /// Whether `request_id` has captured draft context and can be drafted.
    fn speculative_request_ready(&self, _request_id: RequestId) -> bool {
        false
    }

    fn load_lora_adapter(&mut self, request: &LoadLoraAdapterRequest) -> Result<()> {
        anyhow::bail!(
            "Qwen3 LoRA adapter loading is not implemented yet: name={}, path={}",
            request.lora_name,
            request.lora_path.display()
        )
    }

    fn unload_lora_adapter(&mut self, request: &UnloadLoraAdapterRequest) -> Result<()> {
        anyhow::bail!(
            "Qwen3 LoRA adapter unloading is not implemented yet: name={}",
            request.lora_name
        )
    }

    fn list_lora_adapters(&self) -> Vec<String> {
        Vec::new()
    }

    // ── KV-offload prefetch hooks (no-op unless offload is enabled) ─────

    /// Offer a freshly-submitted request for async CPU-tier KV prefetch.
    /// Returns `true` if a load is now in flight and the scheduler must park
    /// the request until [`Self::drain_ready_prefetch`] reports it ready.
    ///
    /// `reserve_floor` is the number of free blocks already promised to
    /// admitted requests (active decode growth + remaining prefill chunks);
    /// the prefetch must not reserve into it, or a mid-prefill request's next
    /// chunk fails allocation and the whole step errors out.
    fn begin_kv_prefetch(
        &mut self,
        _request_id: RequestId,
        _prompt_tokens: &[u32],
        _lora_adapter: Option<&str>,
        _reserve_floor: usize,
    ) -> bool {
        false
    }

    /// Non-blocking sweep: request ids whose prefetch just settled (now
    /// prefill-eligible). `reserve_floor` guards the remote-fetch re-query
    /// path the same way it guards `begin_kv_prefetch`: a fetch that resolves
    /// this tick must not reserve into blocks already promised to admitted
    /// requests.
    fn drain_ready_prefetch(&mut self, _reserve_floor: usize) -> Vec<RequestId> {
        Vec::new()
    }

    /// Block until at least one in-flight prefetch settles (idle-only), then
    /// sweep the rest.
    fn wait_ready_prefetch(&mut self, _reserve_floor: usize) -> Vec<RequestId> {
        Vec::new()
    }

    /// Blocks `request_id` already holds via a settled prefetch (its restored
    /// prefix). These were taken out of the free pool for this request and
    /// become its cached prefill prefix, so admission credits them against the
    /// request's block need to avoid double-counting. Zero unless a prefetch
    /// has committed for `request_id`.
    fn prefetched_blocks(&self, _request_id: RequestId) -> usize {
        0
    }

    /// Whether this executor withholds `Finished` past the step. Only the
    /// P/D prefill role does (`flush_on_finish`): its finishes leave through
    /// [`Self::release_finished_events`] once KV saves are peer-visible.
    /// Everyone else finishes through the emitter, so the terminal rides the
    /// committed step — shipped after the driver publishes load, which keeps
    /// a finishing batch's send-time stats reading the drained occupancy
    /// instead of racing the publish.
    fn withholds_finishes(&self) -> bool {
        false
    }

    /// Deliver this step's withheld finishes once the KV saves + MetaServer
    /// registrations are query-visible to peers — the client treats the HTTP
    /// response as the KV-ready signal — so the scheduler thread never waits
    /// on the flush barrier. Each [`DeferredFinish`] carries the request's
    /// whole final record (tokens included), so late delivery cannot reorder
    /// against the step stream. Called only when
    /// [`Self::withholds_finishes`] is true.
    fn release_finished_events(&self, _finishes: Vec<DeferredFinish>) {
        unreachable!("executor withholds finishes without a delivery override");
    }

    // ── Decode-overlap async prefill ─────────────────────────────────────

    /// Whether prefill/decode overlap is enabled (async prefill supported).
    fn has_decode_overlap(&self) -> bool {
        false
    }

    /// Poll whether the async prefill has completed. Returns `Some(result)` if
    /// done, `None` if still in-flight.
    fn poll_async_prefill(&mut self) -> Option<PrefillResult> {
        None
    }
}

/// Deliver withheld finishes; a closed frontend needs no handling this late
/// in a request's life (the send is infallible-by-contract).
fn send_finished_events(finishes: Vec<DeferredFinish>) {
    for finish in finishes {
        finish.send();
    }
}

struct Qwen3ExecutorMetadata {
    block_size: usize,
    stop_token_ids: Vec<u32>,
    config: Config,
}

pub struct Qwen3Executor {
    metadata: Qwen3ExecutorMetadata,
    kv_mgr: KvCacheManager,
    request_kvs: HashMap<RequestId, pegainfer_kv_cache::RequestKv>,
    primary: RankWorker,
    workers: Vec<RankWorker>,
    loaded_lora_adapters: HashSet<String>,
    /// Requested prefix-cache state; read through `prefix_cache_enabled()`,
    /// which also honours the drafter override.
    prefix_cache_requested: bool,
    lora_options: Qwen3LoraOptions,
    /// pegaflow KV-offload bridge; `None` unless offload is opted in on the
    /// single-GPU path. Drives both the SAVE hook and the async LOAD prefetch.
    offload: Option<OffloadEngine>,
    /// Per-request count of sealed blocks already saved to the host tier, so
    /// each step only saves blocks that newly sealed. Initialized to the
    /// GPU-hit prefix (already resident) on first save.
    saved_cursor: HashMap<RequestId, usize>,
    /// In-flight CPU→GPU prefetches keyed by request, parked until their load
    /// settles and the blocks register into the prefix cache.
    prefetch: HashMap<RequestId, PrefetchState>,
    /// Offload pure-L2 mode. When set, completed blocks are not kept for
    /// cross-request HBM reuse: the prefetch probe drains the inactive pool
    /// first, so every probe sees `gpu_hit == 0` and the whole cacheable prefix
    /// is restored from the host tier. This is what `--no-prefix-cache` means
    /// once offload is on (the L2 restore still rides on `match_and_add_prefix`,
    /// so prefix matching itself stays enabled). Set via
    /// [`Self::set_no_prefix_cache`].
    l1_retention_disabled: bool,
    /// P/D prefill role: withhold each step's `Finished` events until offload
    /// saves + MetaServer registrations are peer-visible, so the HTTP response
    /// doubles as the KV-ready signal (see `Qwen3P2pOptions::flush_on_finish`).
    flush_offload_on_finish: bool,
    /// P/D decode role with a vLLM prefill peer: offload query keys derive
    /// with vLLM's hash scheme, a zero hit waits out the producer's
    /// registration tail, and self-saves are skipped (this node's kvbm keys
    /// would be unfindable in the vLLM-keyed content domain).
    vllm_compat: Option<VllmCompatState>,
    /// Overlap streams (`--decode-overlap`); `None` when off.
    overlap: Option<crate::green_ctx::OverlapStreams>,
    /// In-flight async prefill state. Populated by the SplitConcurrent step,
    /// consumed by `poll_async_prefill`.
    async_prefill: Option<cudarc::driver::CudaEvent>,
    /// DFlash draft metadata; `Some` once a draft model is loaded into the
    /// primary lane. Speculative decoding is enabled iff this is set.
    speculative: Option<DFlashMeta>,
    /// Cumulative acceptance counters for the DFlash path, set and cleared
    /// together with `speculative` above. Written in
    /// `execute_speculative_verify_impl`, read by `publish_load`.
    spec_decode_counters: Option<SpecDecodeCounters>,
    /// Requests whose DFlash context is captured and ready to draft. A request
    /// enters this set when its prompt finishes prefilling with captured target
    /// context, and leaves on retire or a plain (non-speculative) decode.
    dflash_ready_requests: HashSet<RequestId>,
    /// CUDA device ordinal this executor was built on. Used by
    /// [`enable_decode_overlap`] to create overlap streams on the correct
    /// device (the model, KV cache, and compute stream all live here).
    device_ordinal: usize,
}

/// One request's in-flight CPU-tier KV prefetch.
///
/// `probe` holds the GPU-hit prefix resident for the request's whole parked
/// life; `phase` tracks where the missing prefix currently is.
struct PrefetchState {
    probe: PrefixProbe,
    phase: PrefetchPhase,
}

enum PrefetchPhase {
    /// pegaflow is pulling the missing prefix from a remote peer (P2P RDMA)
    /// into the local host tier; the executor re-queries each scheduler tick
    /// until the fetch resolves or `deadline` passes (then: prefill from
    /// scratch). Only entered with P2P configured.
    RemoteFetch {
        query_hashes: Vec<Vec<u8>>,
        deadline: std::time::Instant,
        /// vLLM-compat P/D handoff race guard: until this instant a zero hit
        /// keeps the request parked (the producer's registration hasn't
        /// landed yet) instead of degrading to prefill-from-scratch. Set to
        /// the park time (i.e. already expired) outside vLLM-compat mode.
        miss_deadline: std::time::Instant,
        /// When the request was parked — for the degradation warning.
        parked_at: std::time::Instant,
        /// Last re-query instant: ticks inside [`REMOTE_REQUERY_INTERVAL`]
        /// skip the RPC so N parked requests cannot turn every scheduler
        /// tick into N serial MetaServer round-trips.
        last_query: std::time::Instant,
    },
    /// Host→GPU DMA into reserved local blocks is in flight.
    Loading {
        reservation: LoadReservation,
        handle: LoadHandle,
    },
    /// Load landed and blocks are committed; `probe` keeps the GPU+CPU prefix
    /// resident until the request prefills.
    Committed,
}

/// How long a parked request waits on a remote (P2P) prefix fetch before
/// giving up and prefilling from scratch. A safety net for a hung peer — the
/// normal failure path (peer evicted the blocks, RDMA error) resolves through
/// pegaflow's own fetch timeout into a plain local hit count well before this.
const REMOTE_FETCH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(15);

/// Minimum spacing between re-query RPCs for one parked request. The idle
/// scheduler loop already throttles at ~5ms; this bounds the busy path too,
/// where decode ticks can come faster than the RPC is worth repeating.
const REMOTE_REQUERY_INTERVAL: std::time::Duration = std::time::Duration::from_millis(5);

/// vLLM-compat miss breaker: after this many consecutive requests each
/// exhausted the whole zero-hit wait window, new requests skip the wait (the
/// prefill peer is evidently not publishing — misconfig or down) instead of
/// taxing every cold request the full window. Any remote hit re-arms waiting.
const MISS_BREAKER_THRESHOLD: u32 = 3;

/// vLLM-compat P/D mode, derived from [`crate::Qwen3VllmCompatOptions`] at
/// executor build time (the hasher needs the resolved KV block size).
struct VllmCompatState {
    hasher: pegainfer_kv_offload::VllmBlockHasher,
    /// Zero-hit wait window: how long a cold request re-queries before
    /// giving up on the expected remote KV (see `RemoteFetch::miss_deadline`).
    miss_wait: std::time::Duration,
    /// Requests in a row that exhausted the whole wait window with zero hits.
    /// At [`MISS_BREAKER_THRESHOLD`] the breaker opens and new requests skip
    /// the wait; any remote hit resets it.
    consecutive_miss_windows: u32,
}

fn ensure_pertoken_single_gpu(policy: NumericPolicy, device_count: usize) -> Result<()> {
    anyhow::ensure!(
        policy != NumericPolicy::PerToken || device_count <= 1,
        "NumericPolicy::PerToken is only supported on the single-GPU path; got {device_count} devices"
    );
    Ok(())
}

fn ensure_pertoken_overlap_disabled(
    policy: NumericPolicy,
    overlap: crate::DecodeOverlap,
) -> Result<()> {
    anyhow::ensure!(
        policy != NumericPolicy::PerToken || matches!(overlap, crate::DecodeOverlap::Off),
        "NumericPolicy::PerToken is not compatible with decode-overlap: PerToken CUDA-Graph memory is budgeted only for the full-SM graph cache"
    );
    Ok(())
}

impl Qwen3Executor {
    pub(crate) fn dump_decode_graph_png(&self, png_path: &Path) -> Result<CudaGraphDumpSummary> {
        self.primary.dump_decode_graph_png(png_path.to_path_buf())
    }

    fn single(
        model: Qwen3Model,
        offload_opts: &Qwen3OffloadOptions,
        max_prefill_tokens: usize,
        dflash_kv_bytes_per_token: usize,
        memory_options: Qwen3MemoryOptions,
        hedge_scratch_pages: usize,
    ) -> Result<Self> {
        let (model, budget) = profile_kv_budget_on_worker(
            model,
            max_prefill_tokens,
            dflash_kv_bytes_per_token,
            memory_options,
        )?;
        let kv_mgr = KvCacheManager::new_with_scratch_pages(
            &model.device_ctx().stream,
            budget.num_layers,
            budget.num_kv_heads,
            budget.head_dim,
            budget.block_size,
            budget.num_blocks,
            hedge_scratch_pages,
        )?;
        let device_ordinal = model.device_ctx().device_ordinal;
        let metadata = Qwen3ExecutorMetadata {
            block_size: budget.block_size,
            stop_token_ids: model.config().stop_token_ids.clone(),
            config: model.config().clone(),
        };
        let kv_buffer = kv_mgr.buffer().clone();
        // Build the offload engine while the model's stream is still in hand
        // (it moves into the RankWorker below). Registers the fused KV buffer.
        let offload = build_offload(offload_opts, &kv_mgr, model.config(), model.device_ctx())?;
        let total_blocks = kv_mgr.pool().total_blocks();
        let padding_block_id = kv_mgr.pool().padding_block_id();
        let vllm_compat = match offload_opts.vllm_compat.as_ref() {
            None => None,
            Some(c) => {
                ensure!(
                    c.miss_wait < REMOTE_FETCH_DEADLINE,
                    "kv-pd miss wait ({:?}) must stay below the {:?} remote-fetch \
                     deadline, which would otherwise silently cap it",
                    c.miss_wait,
                    REMOTE_FETCH_DEADLINE,
                );
                let hasher = pegainfer_kv_offload::VllmBlockHasher::new(
                    &c.python_hash_seed,
                    budget.block_size,
                );
                // Cross-engine fingerprint. Every P/D mismatch (seed,
                // namespace, block size, geometry) otherwise presents as
                // nothing but slow cold requests — this line is what an
                // operator diffs against the vLLM peer's startup config.
                log::info!(
                    "vLLM-compat P/D active: seed={} namespace={} block_size={} \
                     none_hash={:032x} layers={} kv_heads={} head_dim={} miss_wait={:?}",
                    c.python_hash_seed,
                    c.namespace,
                    budget.block_size,
                    u128::from_be_bytes(hasher.none_hash()),
                    budget.num_layers,
                    budget.num_kv_heads,
                    budget.head_dim,
                    c.miss_wait,
                );
                Some(VllmCompatState {
                    hasher,
                    miss_wait: c.miss_wait,
                    consecutive_miss_windows: 0,
                })
            }
        };
        Ok(Self {
            metadata,
            kv_mgr,
            request_kvs: HashMap::new(),
            primary: RankWorker::spawn(
                0,
                LocalQwen3Lane::new(
                    model,
                    kv_buffer,
                    total_blocks,
                    padding_block_id,
                    max_prefill_tokens,
                )?,
            )?,
            workers: Vec::new(),
            loaded_lora_adapters: HashSet::new(),
            prefix_cache_requested: true,
            lora_options: Qwen3LoraOptions::default(),
            offload,
            saved_cursor: HashMap::new(),
            prefetch: HashMap::new(),
            l1_retention_disabled: false,
            // Derived here, not via a post-construction setter, so every
            // launch path (plain and LoRA alike) honors the P/D contract.
            flush_offload_on_finish: offload_opts
                .p2p
                .as_ref()
                .is_some_and(|p2p| p2p.flush_on_finish),
            vllm_compat,
            overlap: None,
            async_prefill: None,
            speculative: None,
            spec_decode_counters: None,
            dflash_ready_requests: HashSet::new(),
            device_ordinal,
        })
    }

    pub fn from_runtime(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
    ) -> Result<Self> {
        Self::from_runtime_with_lora_options(
            model_path,
            enable_cuda_graph,
            device_ordinals,
            Qwen3LoraOptions::default(),
            Qwen3OffloadOptions::disabled(),
            crate::scheduler::DEFAULT_MAX_PREFILL_TOKENS,
            None,
            Qwen3MemoryOptions::default(),
        )
    }

    #[allow(
        clippy::needless_pass_by_value,
        reason = "executor construction is a one-shot ownership boundary"
    )]
    pub fn from_runtime_with_lora_options(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
        lora_options: Qwen3LoraOptions,
        offload_options: Qwen3OffloadOptions,
        max_prefill_tokens: usize,
        dflash_draft_path: Option<&str>,
        memory_options: Qwen3MemoryOptions,
    ) -> Result<Self> {
        Self::from_runtime_with_decode_environment(
            model_path,
            enable_cuda_graph,
            device_ordinals,
            lora_options,
            offload_options,
            max_prefill_tokens,
            dflash_draft_path,
            memory_options,
            crate::DecodeOverlap::Off,
        )
    }

    #[allow(
        clippy::needless_pass_by_value,
        clippy::too_many_arguments,
        reason = "executor construction is a one-shot ownership boundary"
    )]
    pub(crate) fn from_runtime_with_decode_environment(
        model_path: &str,
        enable_cuda_graph: bool,
        device_ordinals: &[usize],
        lora_options: Qwen3LoraOptions,
        offload_options: Qwen3OffloadOptions,
        max_prefill_tokens: usize,
        dflash_draft_path: Option<&str>,
        memory_options: Qwen3MemoryOptions,
        decode_overlap: crate::DecodeOverlap,
    ) -> Result<Self> {
        let mut memory_options = memory_options.validate()?;
        let lora_options = lora_options.validate()?;
        anyhow::ensure!(
            !device_ordinals.is_empty(),
            "Qwen3 executor requires at least one device"
        );
        anyhow::ensure!(
            !offload_options.enabled || device_ordinals.len() == 1,
            "KV offload is only supported on the single-GPU path (tensor parallel \
             shards KV per rank); got {} devices",
            device_ordinals.len()
        );
        let policy = pegainfer_kernels::ops::numeric_policy();
        ensure_pertoken_single_gpu(policy, device_ordinals.len())?;
        ensure_pertoken_overlap_disabled(policy, decode_overlap)?;

        if policy == NumericPolicy::PerToken {
            if enable_cuda_graph {
                let max_graph_bucket = crate::batch_decode::DecodeGraphPlan::new(policy)
                    .retained_buckets()
                    .last()
                    .expect("BATCH_BUCKETS is non-empty");
                log::info!(
                    "PerToken decode graph policy: eligible non-LoRA buckets <= \
                    {max_graph_bucket} use lazy CUDA Graph capture/replay; larger \
                    buckets execute eagerly"
                );
            } else {
                log::info!(
                    "PerToken decode graph policy: CUDA Graph disabled; all decode \
                    buckets execute eagerly"
                );
            }
        }
        if device_ordinals.len() == 1 {
            let model = Qwen3Model::from_safetensors_with_runtime(
                model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph,
                    tensor_parallel: None,
                    device_ordinal: device_ordinals[0],
                    max_loras: lora_options.max_loras,
                    max_lora_rank: lora_options.max_lora_rank,
                    decode_overlap,
                    dflash_enabled: dflash_draft_path.is_some(),
                },
            )?;
            // The DFlash draft model loads after profiling but lives outside the
            // paged KV pool, so reserve its footprint up front from the draft
            // config: fixed bytes (weights + block scratch) via the margin, and
            // pool-scaling per-token bytes folded into the block budget.
            let max_verify_batch = *BATCH_BUCKETS.last().unwrap();
            let (dflash_kv_bytes_per_token, hedge_scratch_pages) = match dflash_draft_path {
                Some(path) => {
                    let reservation =
                        crate::dflash::DFlashMemoryReservation::from_path(path, max_verify_batch)?;
                    memory_options.kv_cache_memory_margin_bytes = crate::sizing::sum(&[
                        memory_options.kv_cache_memory_margin_bytes,
                        reservation.fixed_bytes,
                    ])?;
                    // Hedge-chain scratch KV pages (PEGAINFER_SPEC_HEDGE)
                    // live in the KV buffer but outside the pool; bill them
                    // via the margin so the profiled block budget shrinks to
                    // fit instead of OOMing. Two bounds: the effective
                    // geometry (worst-case span pages per chain) and the
                    // expanded verify batch, which holds at most
                    // `max_verify_batch - 1` hedge spans at once.
                    let geometry = dflash_lane::spec_hedge_effective(reservation.block_size);
                    let pages_per_chain = dflash_lane::hedge_pages_per_chain(
                        reservation.block_size,
                        memory_options.page_size,
                    );
                    let pages = if reservation.uses_markov_head {
                        geometry
                            .cap
                            .saturating_mul(pages_per_chain)
                            .saturating_mul(geometry.positions.len())
                            .min(pages_per_chain.saturating_mul(max_verify_batch - 1))
                    } else {
                        // A plain DFlash drafter can never hedge; a nonzero
                        // hedge env must not reserve scratch it cannot use.
                        0
                    };
                    (reservation.kv_bytes_per_token, pages)
                }
                None => (0, 0),
            };
            if hedge_scratch_pages > 0 {
                let page_bytes = model.kv_page_bytes(memory_options.page_size)?;
                memory_options.kv_cache_memory_margin_bytes = crate::sizing::sum(&[
                    memory_options.kv_cache_memory_margin_bytes,
                    crate::sizing::product(&[hedge_scratch_pages, page_bytes])?,
                ])?;
            }
            let mut executor = Self::single(
                model,
                &offload_options,
                max_prefill_tokens,
                dflash_kv_bytes_per_token,
                memory_options,
                hedge_scratch_pages,
            )?;
            executor.lora_options = lora_options;
            return Ok(executor);
        }
        anyhow::ensure!(
            dflash_draft_path.is_none(),
            "speculative decoding requires the single-GPU path (got {} devices)",
            device_ordinals.len()
        );

        let world_size = device_ordinals.len();
        let mut models = Vec::with_capacity(world_size);
        // TP ranks load sequentially and suppress per-rank prefetch, so keep one
        // whole-checkpoint prefetch alive across the loop.
        let (shard_paths, _) = load_shard_info(model_path)?;
        let prefetch = WeightPrefetch::spawn(&shard_paths);
        for (rank, &device_ordinal) in device_ordinals.iter().enumerate() {
            models.push(Qwen3Model::from_safetensors_with_runtime(
                model_path,
                ModelRuntimeConfig {
                    enable_cuda_graph,
                    tensor_parallel: Some(TensorParallelConfig { rank, world_size }),
                    device_ordinal,
                    max_loras: lora_options.max_loras,
                    max_lora_rank: lora_options.max_lora_rank,
                    decode_overlap,
                    dflash_enabled: false,
                },
            )?);
        }
        drop(prefetch);
        // Profile each rank independently and use the minimum shared block
        // count. The logical scheduler uses one block budget for all ranks, but
        // free memory and worker-thread runtime allocations are per device.
        let mut profiled_models = Vec::with_capacity(world_size);
        let mut budgets = Vec::with_capacity(world_size);
        for model in models {
            // DFlash is single-GPU only, so the TP path reserves nothing for it.
            let (model, budget) =
                profile_kv_budget_on_worker(model, max_prefill_tokens, 0, memory_options)?;
            profiled_models.push(model);
            budgets.push(budget);
        }
        let mut models = profiled_models;
        let mut budget = budgets[0];
        budget.num_blocks = budgets
            .iter()
            .map(|budget| budget.num_blocks)
            .min()
            .expect("at least one TP rank");
        log::info!(
            "TP KV budget: using {} blocks (minimum across {} ranks)",
            budget.num_blocks,
            world_size
        );

        // Create the centralized KvCacheManager on rank 0's stream.
        let kv_mgr = KvCacheManager::new(
            &models[0].device_ctx().stream,
            budget.num_layers,
            budget.num_kv_heads,
            budget.head_dim,
            budget.block_size,
            budget.num_blocks,
        )?;

        let metadata = Qwen3ExecutorMetadata {
            block_size: budget.block_size,
            stop_token_ids: models[0].config().stop_token_ids.clone(),
            config: models[0].config().clone(),
        };

        // Create extra KvBuffers for ranks 1+ on their respective streams.
        let mut extra_kv_buffers = Vec::with_capacity(world_size - 1);
        for model in &models[1..] {
            extra_kv_buffers.push(KvBuffer::new(
                &model.device_ctx().stream,
                budget.num_layers,
                budget.num_kv_heads,
                budget.head_dim,
                budget.block_size,
                budget.num_blocks,
            )?);
        }

        // Each comm is built on its rank's compute stream — the same stream
        // decode capture runs on — so its all-reduces land inside the captured
        // graph; a comm on any other stream would execute them eagerly instead.
        let streams = models
            .iter()
            .map(|m| m.device_ctx().stream.clone())
            .collect();
        let comms = cudarc::nccl::safe::Comm::from_devices(streams)
            .map_err(|e| anyhow::anyhow!("failed to initialize NCCL comms: {e:?}"))?;
        for (model, comm) in models.iter_mut().zip(comms) {
            model.attach_tp_comm(comm);
        }

        let total_blocks = kv_mgr.pool().total_blocks();
        let padding_block_id = kv_mgr.pool().padding_block_id();

        // Primary rank gets the KvBuffer from the centralized manager.
        let primary_buffer = kv_mgr.buffer().clone();
        let mut models_iter = models.into_iter();
        let primary_model = models_iter.next().unwrap();
        let primary = RankWorker::spawn(
            0,
            LocalQwen3Lane::new(
                primary_model,
                primary_buffer,
                total_blocks,
                padding_block_id,
                max_prefill_tokens,
            )?,
        )?;

        // Worker ranks get their own extra KvBuffers.
        let workers = models_iter
            .zip(extra_kv_buffers)
            .enumerate()
            .map(|(index, (model, buffer))| {
                let lane = LocalQwen3Lane::new(
                    model,
                    buffer,
                    total_blocks,
                    padding_block_id,
                    max_prefill_tokens,
                )?;
                RankWorker::spawn(index + 1, lane)
            })
            .collect::<Result<Vec<_>>>()?;

        // Pre-capture every reachable decode graph now (see [`PrecapturePhase`]):
        // after every RankWorker is spawned (each launch blocks on its peers) and
        // exactly once (a mid-serving re-run is the #481 capture window).
        // Uncompiled GQA groups reroute decode to the eager unified path and skip it.
        if enable_cuda_graph && metadata.config.decode_group_is_compiled() {
            // NCCL has no device timeout, so a desynced sweep wedges forever;
            // this watchdog aborts on the deadline. abort() not exit() — exit's
            // cudart atexit teardown takes the same wedged driver lock — and it
            // disarms only on the explicit success send (drop-on-error stays armed).
            let (sweep_done_tx, sweep_done_rx) = channel::bounded::<()>(1);
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(600);
            thread::Builder::new()
                .name("qwen3-tp-precapture-watchdog".into())
                .spawn(move || {
                    if sweep_done_rx.recv_deadline(deadline).is_ok() {
                        return;
                    }
                    // A sender drop (error path) can wake us early; stay armed
                    // to the deadline before deciding startup is wedged.
                    std::thread::sleep(deadline.saturating_duration_since(std::time::Instant::now()));
                    eprintln!(
                        "qwen3 TP decode graph pre-capture did not complete within 600s — NCCL wedge suspected, aborting"
                    );
                    log::error!(
                        "qwen3 TP decode graph pre-capture did not complete within 600s — NCCL wedge suspected, aborting"
                    );
                    std::process::abort();
                })
                .map_err(|e| anyhow::anyhow!("failed to spawn pre-capture watchdog: {e}"))?;
            let started = std::time::Instant::now();
            let run_phase = |phase: PrecapturePhase| -> Result<()> {
                let pending = std::iter::once(&primary)
                    .chain(workers.iter())
                    .map(|worker| worker.precapture(phase))
                    .collect::<Result<Vec<_>>>()?;
                let ranks = pending.len();
                for (rank, recv) in pending.into_iter().enumerate() {
                    let outcome = recv.recv().map_err(|_| {
                        anyhow::anyhow!(
                            "tensor-parallel rank {rank} dropped during graph pre-capture {phase:?}"
                        )
                    });
                    if let Err(e) = outcome.and_then(|r| {
                        r.map_err(|e| {
                            anyhow::anyhow!(
                                "tensor-parallel rank {rank} graph pre-capture {phase:?} failed: {e:#}"
                            )
                        })
                    }) {
                        // Peers past this rank may be wedged in an unpaired
                        // collective; name them so the wedge is attributable.
                        log::error!(
                            "graph pre-capture aborting in {phase:?}; ranks {:?} were not yet collected and may be wedged in NCCL collectives",
                            (rank + 1..ranks).collect::<Vec<_>>()
                        );
                        return Err(e);
                    }
                }
                Ok(())
            };
            run_phase(PrecapturePhase::Warmup)?;
            for bucket_idx in 0..BATCH_BUCKETS.len() {
                run_phase(PrecapturePhase::Capture { bucket_idx })?;
                run_phase(PrecapturePhase::Launch { bucket_idx })?;
            }
            run_phase(PrecapturePhase::Finalize)?;
            let _ = sweep_done_tx.send(());
            log::info!(
                "TP decode graph pre-capture: {} buckets per rank captured in {:.2}s",
                BATCH_BUCKETS.len(),
                started.elapsed().as_secs_f64()
            );
        }

        Ok(Self {
            metadata,
            kv_mgr,
            request_kvs: HashMap::new(),
            primary,
            workers,
            loaded_lora_adapters: HashSet::new(),
            prefix_cache_requested: true,
            lora_options,
            // Offload is single-GPU only (asserted above); never built here.
            offload: None,
            saved_cursor: HashMap::new(),
            prefetch: HashMap::new(),
            l1_retention_disabled: false,
            flush_offload_on_finish: false,
            vllm_compat: None,
            overlap: None,
            async_prefill: None,
            speculative: None,
            spec_decode_counters: None,
            dflash_ready_requests: HashSet::new(),
            device_ordinal: device_ordinals[0],
        })
    }

    pub fn drop_request(&mut self, request_id: RequestId) -> Result<()> {
        <Self as ModelExecutor>::drop_request(self, request_id)
    }

    pub fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        <Self as ModelExecutor>::execute_prefill(self, plan)
    }

    pub fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<DecodeResult> {
        <Self as ModelExecutor>::execute_decode(self, plan)
    }

    pub fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult> {
        <Self as ModelExecutor>::execute_unified(self, plan)
    }

    pub fn load_lora_adapter(&mut self, request: &LoadLoraAdapterRequest) -> Result<()> {
        <Self as ModelExecutor>::load_lora_adapter(self, request)
    }

    /// Prefix caching is on by default; tests that assert bit-identical
    /// replay disable it (a cache hit changes prefill GEMM shapes, which
    /// drifts logits by bf16 ULPs). Disabling it also disables retention of
    /// completed request blocks: retaining blocks that can never be matched
    /// creates duplicate primaries outside request-level capacity accounting.
    pub fn set_prefix_cache_enabled(&mut self, enabled: bool) {
        let retained_before = self.retains_completed_kv_blocks();
        self.prefix_cache_requested = enabled;
        self.finish_retention_transition(retained_before);
    }

    /// Whether prefix reuse is actually in effect. A loaded drafter forces it
    /// off: speculative capture needs uncached hidden states, and the KV budget
    /// charges the draft's out-of-pool cache per POOL block, which only holds
    /// while blocks are not shared between requests. `load_dflash_draft_model`
    /// also clears the requested flag; this makes the override structural.
    fn prefix_cache_enabled(&self) -> bool {
        self.prefix_cache_requested && self.speculative.is_none()
    }

    /// Whether a completed request should leave its registered GPU blocks in
    /// the inactive L1 cache for a later prefix match.
    fn retains_completed_kv_blocks(&self) -> bool {
        self.prefix_cache_enabled() && !self.l1_retention_disabled
    }

    /// Drain blocks retained under the old policy once when L1 retention is
    /// disabled. Active blocks are untouched; `drop_request` marks them for
    /// reset when their final owners finish.
    fn finish_retention_transition(&mut self, retained_before: bool) {
        if retained_before && !self.retains_completed_kv_blocks() {
            self.kv_mgr.pool().evict_inactive();
        }
    }

    /// Configure two-stream prefill/decode overlap (see [`crate::DecodeOverlap`]).
    /// A no-op for [`crate::DecodeOverlap::Off`]; otherwise sets up the streams,
    /// returning an error if the GPU/driver cannot honor the requested mode.
    pub(crate) fn enable_decode_overlap(&mut self, overlap: crate::DecodeOverlap) -> Result<()> {
        // Pre-capture backstop: a runtime caller could set Pin then enable overlap here, bypassing the
        // engine-entry guard. launch_gemm_pin also bails on the resulting stream override, but only mid
        // graph-capture/replay (a hot-path failure) — rejecting here moves it to a safe point.
        anyhow::ensure!(
            pegainfer_kernels::ops::numeric_policy() != pegainfer_kernels::ops::NumericPolicy::Pin
                || matches!(overlap, crate::DecodeOverlap::Off),
            "--batch-invariant (NumericPolicy::Pin) is not compatible with decode-overlap: the stream override would force the pinned GEMM to bail at runtime"
        );
        // PerToken profiles only the full-SM graph cache. Overlap captures into
        // the independent split-stream cache, which would otherwise be unbudgeted.
        ensure_pertoken_overlap_disabled(pegainfer_kernels::ops::numeric_policy(), overlap)?;

        // Overlap's split-concurrent decode stream uses the group-limited decode
        // kernel this GQA group can't instantiate; reject at load, not mid-serving.
        anyhow::ensure!(
            self.metadata.config.decode_group_is_compiled()
                || matches!(overlap, crate::DecodeOverlap::Off),
            "decode-overlap is unsupported for GQA group size {}/{}: no compiled decode kernel",
            self.metadata.config.num_attention_heads,
            self.metadata.config.num_key_value_heads,
        );
        // TP decode graphs are pre-captured on the compute stream only; a split
        // decode stream would route capture into the graphs_split cache with no
        // cross-rank alignment (and the streams below are built on device 0).
        anyhow::ensure!(
            self.workers.is_empty() || matches!(overlap, crate::DecodeOverlap::Off),
            "decode-overlap is unsupported under tensor parallelism"
        );
        let device_ordinal = self.device_ordinal;
        self.overlap = crate::green_ctx::OverlapStreams::create(device_ordinal, overlap)?;
        Ok(())
    }

    /// vLLM-style `--no-prefix-cache`. Behaviour depends on whether offload is
    /// active:
    ///   * **No offload** — classic: disable prefix matching outright, so every
    ///     prefill recomputes the full prompt.
    ///   * **With offload** — pure-L2 mode: keep matching on (the host-tier
    ///     restore registers blocks and relies on `match_and_add_prefix` to pick
    ///     them up) but stop retaining completed blocks in HBM, so no request
    ///     ever serves its prefix from a cross-request L1 hit. Every reuse then
    ///     comes from the host tier, which is the point of the L2 benchmark.
    ///
    /// A resident HBM block and its host-tier copy share one content hash, so
    /// the cache cannot be told to prefer L2 for a block still in HBM — the only
    /// way to force the bytes from L2 is to not keep the HBM copy around.
    pub(crate) fn set_no_prefix_cache(&mut self, on: bool) {
        if self.offload.is_some() {
            let retained_before = self.retains_completed_kv_blocks();
            self.l1_retention_disabled = on;
            self.finish_retention_transition(retained_before);
        } else {
            self.set_prefix_cache_enabled(!on);
        }
    }

    /// Enable speculative decoding by loading a DFlash draft model into the
    /// primary lane.
    ///
    /// Requires the single-GPU topology (tensor parallel shards KV per rank) and
    /// is incompatible with KV offload. Disables the prefix cache: speculative
    /// capture needs clean, uncached target hidden states for every prompt
    /// token, and a prefix-cache hit skips the forward that would produce them.
    pub(crate) fn load_dflash_draft_model(&mut self, draft_path: &str) -> Result<()> {
        anyhow::ensure!(
            self.workers.is_empty(),
            "speculative decoding requires the single-GPU path (got {} extra ranks)",
            self.workers.len()
        );
        anyhow::ensure!(
            self.offload.is_none(),
            "speculative decoding is not supported together with KV offload"
        );
        let meta = self.primary.load_dflash(draft_path.to_string())?;
        let counters = SpecDecodeCounters::new(meta.num_spec_tokens)?;
        log::info!(
            "Qwen3 DFlash speculative decoding enabled: draft block size {}",
            meta.block_size
        );
        self.set_prefix_cache_enabled(false);
        self.spec_decode_counters = Some(counters);
        self.speculative = Some(meta);
        Ok(())
    }

    /// Whether KV offload is active on this executor.
    pub fn offload_enabled(&self) -> bool {
        self.offload.is_some()
    }

    /// Flush pending offload saves into the host read cache so a following
    /// query can see them. A persistence barrier for handoff and tests; no-op
    /// without offload.
    pub fn flush_offload_saves(&self) {
        if let Some(offload) = &self.offload {
            offload.flush_saves();
        }
    }

    /// Drop every cached-but-unused GPU prefix block. With offload on, this
    /// forces a cold prefix to be restored from the host tier on its next
    /// request (rather than served from HBM).
    pub fn evict_cached_blocks(&self) {
        self.kv_mgr.pool().evict_inactive();
    }

    /// Begin an async CPU-tier KV prefetch for `request_id`; see the
    /// [`ModelExecutor`] hook. Public so admission drivers and tests can park a
    /// request on its load. Returns `true` when a load is in flight.
    pub fn begin_kv_prefetch(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[u32],
        lora_adapter: Option<&str>,
        reserve_floor: usize,
    ) -> bool {
        <Self as ModelExecutor>::begin_kv_prefetch(
            self,
            request_id,
            prompt_tokens,
            lora_adapter,
            reserve_floor,
        )
    }

    /// Block until at least one in-flight prefetch settles, then sweep the
    /// rest; returns the settled request ids (now prefill-eligible).
    pub fn wait_ready_prefetch(&mut self, reserve_floor: usize) -> Vec<RequestId> {
        <Self as ModelExecutor>::wait_ready_prefetch(self, reserve_floor)
    }

    // ── KV-offload SAVE ────────────────────────────────────────────────

    /// Save every block that sealed since this request's last save to the host
    /// tier (fire-and-forget). Safe to call right after `apply_prefill`/
    /// `apply_decode`: the producing step's token read-back has already
    /// synchronized the compute stream, so the sealed KV is fully written.
    fn save_sealed_blocks(&mut self, request_id: RequestId) {
        if self.offload.is_none() {
            return;
        }
        if self.vllm_compat.is_some() {
            // The content domain is keyed with vLLM's hash scheme; this node's
            // kvbm-keyed self-saves would be unfindable there. Remote blocks
            // it fetched are already host-cached (under the vLLM keys) by
            // pegaflow's read path, so multi-turn reuse doesn't need them.
            return;
        }
        let Some(rkv) = self.request_kvs.get(&request_id) else {
            return;
        };
        // `assigned_block_hashes` lists only sealed (registered) blocks; the
        // partial tail block has no hash and never appears here.
        let assigned = rkv.assigned_block_hashes();
        let prefix_matched = rkv.prefix_matched_blocks();
        let cursor = self
            .saved_cursor
            .entry(request_id)
            .or_insert(prefix_matched);
        if assigned.len() <= *cursor {
            return;
        }
        let fresh = &assigned[*cursor..];
        let block_ids: Vec<i32> = fresh.iter().map(|(id, _)| *id).collect();
        let block_hashes: Vec<Vec<u8>> = fresh.iter().map(|(_, h)| h.to_vec()).collect();
        // Pin exactly the blocks being saved (aligned 1:1 with `assigned`) for
        // the duration of the async D2H, so a finished request can't hand the
        // slot to a new request that overwrites it before the copy lands.
        let pins: Vec<KvBlockGuard> = rkv
            .assigned_block_guards()
            .into_iter()
            .skip(*cursor)
            .collect();
        *cursor = assigned.len();
        self.offload
            .as_ref()
            .expect("offload present")
            .save(&block_ids, &block_hashes, pins);
    }

    // ── Chunked prefill ────────────────────────────────────────────────

    /// Prepare one prefill step for `req`: create its `RequestKv` on the
    /// first chunk (matching the prefix cache), then clamp the scheduler's
    /// chunk budget to the prompt tokens actually remaining and allocate KV
    /// for them. Sets `chunk_start`/`chunk_tokens` on the item.
    fn schedule_prefill_chunk(&mut self, req: &mut PrefillStepItem) -> Result<()> {
        if !self.request_kvs.contains_key(&req.request_id) {
            let mut rkv = self.kv_mgr.pool().new_request(
                req.prompt_tokens.clone(),
                req.max_output_tokens,
                req.lora_adapter.as_deref(),
            );
            // Echo needs logits for every prompt position; cached positions
            // are never forwarded, so echo requests prefill from scratch.
            if self.prefix_cache_enabled() && !req.echo {
                req.cached_tokens = rkv.match_and_add_prefix(self.kv_mgr.pool())?;
            }
            self.request_kvs.insert(req.request_id, rkv);
            // match_and_add_prefix above already absorbed any CPU-prefetched
            // blocks (now held by the request's sequence), so release the
            // prefetch's separate hold.
            self.prefetch.remove(&req.request_id);
        }
        let rkv = self
            .request_kvs
            .get_mut(&req.request_id)
            .expect("inserted above");
        req.chunk_start = rkv.kv_position();
        let remaining = req.prompt_tokens.len() - req.chunk_start;
        // Echo must produce all-position logits in a single forward, so it is
        // exempt from chunking (the scheduler never splits echo requests).
        req.chunk_tokens = if req.echo {
            remaining
        } else {
            remaining.min(req.chunk_budget)
        };
        assert!(
            req.chunk_tokens > 0,
            "zero-token prefill chunk for {:?} (budget {})",
            req.request_id,
            req.chunk_budget
        );
        rkv.schedule_prefill(req.chunk_tokens, self.kv_mgr.pool())
            .map_err(|e| anyhow::anyhow!("schedule_prefill failed for {:?}: {e}", req.request_id))
    }

    /// Register a finished prefill step on the request's KV: the final chunk
    /// carries the first generated token, non-final chunks only advance the
    /// KV position.
    fn apply_prefill_result(&mut self, result: &PrefillRequestResult) -> Result<()> {
        let rkv = self
            .request_kvs
            .get_mut(&result.request_id)
            .expect("request must exist after prefill");
        if result.completed {
            rkv.apply_prefill(result.first_token, self.kv_mgr.pool())
        } else {
            rkv.apply_prefill_chunk(self.kv_mgr.pool())
        }
    }

    // ── KV-offload LOAD (async CPU-tier prefetch) ──────────────────────
    // The trait-facing prefetch hooks (`begin_kv_prefetch`,
    // `drain_ready_prefetch`, `wait_ready_prefetch`, `has_pending_prefetch`)
    // live in the `ModelExecutor` impl below; `settle_prefetch` is their shared
    // helper.

    /// Finalize one prefetch whose load returned `result`. On success the
    /// reserved blocks are staged + registered (held by the probe until the
    /// request prefills); on failure the state is dropped so the request
    /// prefills from scratch.
    fn settle_prefetch(
        &mut self,
        id: RequestId,
        reservation: LoadReservation,
        result: Result<(), pegainfer_kv_offload::EngineError>,
    ) {
        match result {
            Ok(()) => {
                let st = self.prefetch.get_mut(&id).expect("prefetch present");
                st.phase = PrefetchPhase::Committed;
                self.kv_mgr
                    .pool()
                    .commit_loaded_blocks(&mut st.probe, reservation);
            }
            Err(e) => {
                log::warn!("KV offload load failed for {id:?} (prefill from scratch): {e}");
                self.prefetch.remove(&id);
            }
        }
    }

    /// Re-query a request parked on a remote (P2P) prefix fetch. Terminal
    /// transitions return `true` (request is prefill-eligible): a `Ready`
    /// answer either starts the H2D load (probe → Loading, still parked → and
    /// so returns `false`) or, on zero hit / reservation pressure / deadline,
    /// drops the prefetch so the request prefills from scratch.
    fn poll_remote_fetch(&mut self, id: RequestId, reserve_floor: usize) -> bool {
        let Some(st) = self.prefetch.get(&id) else {
            return true;
        };
        let PrefetchPhase::RemoteFetch {
            query_hashes,
            deadline,
            miss_deadline,
            parked_at,
            last_query,
        } = &st.phase
        else {
            return false;
        };
        let now = std::time::Instant::now();
        let timed_out = now > *deadline;
        if timed_out {
            log::warn!("remote KV fetch timed out for {id:?}; prefill from scratch");
        }
        if !timed_out && now.duration_since(*last_query) < REMOTE_REQUERY_INTERVAL {
            return false; // stay parked; too soon for another MetaServer RPC
        }
        // The breaker cuts already-parked waiters short too: a request can
        // enter this phase past the breaker via a transient Loading answer,
        // and "the peer is evidently not publishing" applies to it as well.
        let breaker_closed = self
            .vllm_compat
            .as_ref()
            .is_none_or(|c| c.consecutive_miss_windows < MISS_BREAKER_THRESHOLD);
        let wait_on_miss = self.vllm_compat.is_some() && breaker_closed && now <= *miss_deadline;
        let miss_deadline = *miss_deadline;
        let parked_for = now.duration_since(*parked_at);
        let queried_blocks = query_hashes.len();
        let query_hashes = query_hashes.clone();
        if let Some(st) = self.prefetch.get_mut(&id)
            && let PrefetchPhase::RemoteFetch { last_query, .. } = &mut st.phase
        {
            *last_query = now;
        }
        let available_blocks = self.kv_mgr.pool().available_blocks();
        let mut query_errored = false;
        let action = {
            let offload = self.offload.as_ref().expect("offload present in prefetch");
            remote_fetch_action(
                timed_out,
                wait_on_miss,
                breaker_closed,
                || {
                    offload
                        // Partial remote hits are a win here: the miss is
                        // recomputed locally, so never hold out for the
                        // full prefix.
                        .query(&id.raw().to_string(), &query_hashes, false)
                        .map(QueryView::from)
                        .map_err(|e| {
                            query_errored = true;
                            log::warn!(
                                "remote KV re-query failed for {id:?} (prefill from scratch): {e}"
                            );
                        })
                },
                available_blocks,
                reserve_floor,
            )
        };
        match action {
            RemoteFetchAction::Wait => false,
            RemoteFetchAction::Scratch => {
                // A vLLM-compat request that waited out the whole miss window
                // is the sole symptom of every P/D misconfiguration (seed,
                // namespace, block size, peer down) — never degrade silently.
                // The 15s hard timeout (a Loading-stuck peer) counts toward
                // the breaker too, with its own warning already emitted.
                // Requests cut short by an open breaker (now before the
                // deadline) scratch quietly: the breaker warning already
                // announced the mode.
                let window_exhausted = self.vllm_compat.is_some()
                    && (timed_out || (!query_errored && now > miss_deadline));
                if window_exhausted {
                    if !timed_out {
                        log::warn!(
                            "expected remote KV never appeared for {id:?} \
                             ({queried_blocks} blocks, waited {parked_for:?}); prefill from \
                             scratch — check P/D seed/namespace/block-size alignment"
                        );
                    }
                    self.note_miss_window_exhausted();
                }
                self.prefetch.remove(&id);
                true
            }
            RemoteFetchAction::Release(lease) => {
                self.note_remote_hit();
                let offload = self.offload.as_ref().expect("offload present in prefetch");
                offload.release_query_lease(lease);
                self.prefetch.remove(&id);
                true
            }
            RemoteFetchAction::Load(lease, num_blocks) => {
                self.note_remote_hit();
                let probe = self
                    .prefetch
                    .remove(&id)
                    .expect("prefetch present in RemoteFetch")
                    .probe;
                match self.start_prefetch_load(id, probe, lease, num_blocks) {
                    Ok(()) => false,
                    Err(()) => true,
                }
            }
        }
    }

    /// Miss-breaker bookkeeping: one more request exhausted the whole
    /// zero-hit wait window. At the threshold the breaker opens and
    /// [`Self::begin_kv_prefetch`] stops parking new requests.
    fn note_miss_window_exhausted(&mut self) {
        let Some(compat) = self.vllm_compat.as_mut() else {
            return;
        };
        compat.consecutive_miss_windows = compat.consecutive_miss_windows.saturating_add(1);
        if compat.consecutive_miss_windows == MISS_BREAKER_THRESHOLD {
            log::warn!(
                "P/D miss breaker open: {MISS_BREAKER_THRESHOLD} consecutive requests \
                 exhausted the remote-KV wait window; new requests prefill from scratch \
                 immediately until a remote hit lands"
            );
        }
    }

    /// Miss-breaker bookkeeping: remote content is visible again (leased
    /// hit), so cold requests may wait on the P/D handoff race once more.
    fn note_remote_hit(&mut self) {
        if let Some(compat) = self.vllm_compat.as_mut() {
            compat.consecutive_miss_windows = 0;
        }
    }

    /// Reserve GPU destination blocks for a leased host-tier hit, submit the
    /// H2D load, and park the request as a `Loading` prefetch (taking ownership
    /// of `probe`, which keeps the GPU-hit prefix resident meanwhile).
    /// `Err(())` means the prefetch was abandoned (lease released, no state
    /// kept) and the request should prefill from scratch.
    fn start_prefetch_load(
        &mut self,
        id: RequestId,
        probe: PrefixProbe,
        lease: pegainfer_kv_offload::QueryLeaseId,
        num_blocks: usize,
    ) -> Result<(), ()> {
        let offload = self.offload.as_ref().expect("offload present in prefetch");
        let Some(reservation) = self.kv_mgr.pool().reserve_loaded_blocks(num_blocks) else {
            // Block pressure: release the lease so its pinned host blocks
            // aren't held for the full lease TTL, and prefill from scratch
            // rather than stall.
            offload.release_query_lease(lease);
            return Err(());
        };
        let page_ids = reservation.page_ids();
        match offload.load(lease, page_ids) {
            Ok(handle) => {
                self.prefetch.insert(
                    id,
                    PrefetchState {
                        probe,
                        phase: PrefetchPhase::Loading {
                            reservation,
                            handle,
                        },
                    },
                );
                Ok(())
            }
            Err(e) => {
                log::warn!("KV offload load submit failed for {id:?} (skipping): {e}");
                // `load` consumes the lease only past its early validation; a
                // submit error may leave it pinned, so release it (no-op if it
                // was already consumed).
                offload.release_query_lease(lease);
                Err(())
            }
        }
    }

    fn wait_for_step_ack(
        pending: Vec<channel::Receiver<Result<WorkerStepOutcome>>>,
        op_name: &'static str,
    ) -> Result<()> {
        for recv in pending {
            match recv
                .recv()
                .map_err(|_| anyhow::anyhow!("tensor-parallel {op_name} worker dropped"))??
            {
                WorkerStepOutcome::Ack => {}
                other => {
                    return Err(anyhow::anyhow!(
                        "tensor-parallel {op_name} worker returned unexpected payload: {}",
                        other.kind()
                    ));
                }
            }
        }
        Ok(())
    }

    fn run_step(&self, step: &StepCommand) -> Result<WorkerStepOutcome> {
        let primary = self.primary.run_step(step.clone(), true)?;
        let mut pending = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            pending.push(worker.run_step(step.clone(), false)?);
        }
        let primary_result = primary
            .recv()
            .map_err(|_| anyhow::anyhow!("primary worker dropped step response"))??;
        Self::wait_for_step_ack(pending, step.kind())?;
        Ok(primary_result)
    }
}

fn profile_kv_budget_on_worker(
    model: Qwen3Model,
    max_prefill_tokens: usize,
    dflash_kv_bytes_per_token: usize,
    memory_options: Qwen3MemoryOptions,
) -> Result<(Qwen3Model, KvBudget)> {
    let handle = thread::Builder::new()
        .name(format!(
            "qwen3-memory-profile-dev{}",
            model.device_ctx().device_ordinal
        ))
        .spawn(move || -> Result<(Qwen3Model, KvBudget)> {
            bind_model_thread(&model)?;
            let _guard = CublasThreadGuard;
            tune_decode_gemm_algos(&model, max_prefill_tokens, false)?;
            let budget = model.profiled_kv_budget(
                max_prefill_tokens,
                *BATCH_BUCKETS.last().unwrap(),
                dflash_kv_bytes_per_token,
                memory_options,
            )?;
            Ok((model, budget))
        })
        .map_err(|e| anyhow::anyhow!("failed to spawn Qwen3 memory profile worker: {e}"))?;
    handle
        .join()
        .map_err(|_| anyhow::anyhow!("Qwen3 memory profile worker panicked"))?
}

/// Build the KV-offload engine for the single-GPU path, or `None` when offload
/// is disabled. Registers the fused KV buffer with pegaflow against the model's
/// device/stream — must be called while that stream is still owned by the model
/// (before it moves into the `RankWorker`).
fn build_offload(
    opts: &Qwen3OffloadOptions,
    kv_mgr: &KvCacheManager,
    config: &Config,
    ctx: &DeviceContext,
) -> Result<Option<OffloadEngine>> {
    if !opts.enabled {
        return Ok(None);
    }
    let device_id = ctx.device_ordinal as i32;
    let layout = kv_mgr.buffer().layout();
    // Content-addressing domain: two engines may cross-hit only when the same
    // token prefix produces interchangeable KV bytes. That needs the *model*
    // to match, not just the KV geometry — Qwen3-4B and 8B share
    // layers/heads/head_dim and a tokenizer, so geometry alone would let a
    // mixed mesh silently feed one model the other's KV. hidden_size +
    // intermediate_size + vocab_size discriminate the model line's sizes;
    // the layout fields pin the block geometry the transfer relies on.
    // vLLM-compat mode joins the *P side's* content domain instead: the
    // pegaflow connector derives an 8-hex namespace from vLLM config (and logs
    // it at startup); reproducing that derivation would mean chasing Python
    // repr of vLLM internals, so the operator passes it through explicitly.
    let namespace = match &opts.vllm_compat {
        Some(compat) => compat.namespace.clone(),
        None => format!(
            "pegainfer-qwen3-hs{}-is{}-v{}-l{}h{}d{}p{}",
            config.hidden_size,
            config.intermediate_size,
            config.vocab_size,
            layout.num_layers,
            layout.num_kv_heads,
            layout.head_dim,
            layout.page_size
        ),
    };
    let mut config = OffloadConfig::new(
        format!("qwen3-dev{device_id}"),
        device_id,
        opts.pinned_pool_bytes,
    )
    .with_namespace(namespace);
    config.use_hugepages = opts.use_hugepages;
    if let Some(p2p) = &opts.p2p {
        config = config.with_p2p(pegainfer_kv_offload::P2pConfig {
            metaserver_addr: p2p.metaserver_addr.clone(),
            advertise_addr: p2p.advertise_addr.clone(),
            rdma_nics: p2p.rdma_nics.clone(),
        });
    }
    let engine = OffloadEngine::new(config, kv_mgr.buffer(), &ctx.stream)
        .map_err(|e| anyhow::anyhow!("KV offload engine init failed: {e}"))?;
    log::info!(
        "KV offload enabled on device {device_id} ({} MiB host tier, p2p={})",
        opts.pinned_pool_bytes >> 20,
        opts.p2p.is_some(),
    );
    Ok(Some(engine))
}

fn ensure_lora_capacity(
    loaded_lora_adapters: &HashSet<String>,
    lora_name: &str,
    max_loras: usize,
    load_inplace: bool,
) -> Result<()> {
    if loaded_lora_adapters.contains(lora_name) {
        anyhow::ensure!(
            load_inplace,
            "Qwen3 LoRA adapter {lora_name} is already loaded"
        );
        return Ok(());
    }
    anyhow::ensure!(
        loaded_lora_adapters.len() < max_loras,
        "Qwen3 LoRA adapter capacity exceeded: max_loras={}, loaded_adapters={}, requested={}",
        max_loras,
        loaded_lora_adapters.len(),
        lora_name
    );
    Ok(())
}

impl ModelExecutor for Qwen3Executor {
    fn block_size(&self) -> usize {
        self.metadata.block_size
    }

    fn max_request_blocks(&self) -> usize {
        self.kv_mgr.pool().max_request_blocks()
    }

    fn max_context_tokens(&self) -> usize {
        let target = self.metadata.config.max_position_embeddings;
        match &self.speculative {
            // The draft's fixed-width in-fill block writes `block_size` positions
            // past the committed length each step, so a request may use at most
            // `draft.max_pos - block_size` tokens before the draft cache would
            // overflow. Reject the rest at admission instead of crashing mid-prefill.
            Some(meta) => target.min(meta.max_position_embeddings.saturating_sub(meta.block_size)),
            None => target,
        }
    }

    fn max_decode_batch_size(&self) -> usize {
        *BATCH_BUCKETS.last().unwrap()
    }

    fn available_blocks(&self) -> usize {
        self.kv_mgr.pool().available_blocks()
    }

    fn spec_decode_counters(&self) -> Option<SpecDecodeCounters> {
        self.spec_decode_counters
    }

    fn is_stop_token(&self, token_id: u32) -> bool {
        self.metadata.stop_token_ids.contains(&token_id)
    }

    fn prefetched_blocks(&self, request_id: RequestId) -> usize {
        self.prefetch
            .get(&request_id)
            .map_or(0, |st| st.probe.held_blocks())
    }

    fn withholds_finishes(&self) -> bool {
        self.flush_offload_on_finish
    }

    fn release_finished_events(&self, finishes: Vec<DeferredFinish>) {
        // P/D prefill role: the peer treats our HTTP response as the
        // KV-ready signal, so `Finished` may leave only after this step's
        // saves + MetaServer registrations are peer-visible. The barrier
        // runs on the offload runtime; the scheduler thread never waits.
        let offload = self
            .offload
            .as_ref()
            .expect("flush_offload_on_finish implies an offload runtime");
        offload.flush_saves_then(move || send_finished_events(finishes));
    }

    fn drop_request(&mut self, request_id: RequestId) -> Result<()> {
        // Remove and drop — RAII on SchedulableSequence's block guards
        // returns all allocated blocks regardless of lifecycle state. The same
        // RAII frees any parked prefetch's reserved/held blocks.
        let retain_completed_kv = self.retains_completed_kv_blocks();
        let mut removed = self.request_kvs.remove(&request_id);
        if let Some(rkv) = removed.as_mut()
            && !retain_completed_kv
        {
            rkv.mark_blocks_reset_on_release();
        }
        drop(removed);
        // A parked prefetch may still have a load in flight: pegaflow's worker
        // is writing the reserved GPU blocks (H2D). Dropping the reservation now
        // frees those physical pages for immediate reuse while the DMA keeps
        // landing on them — silent KV corruption, the load-side mirror of the
        // SAVE keep-alive pin. Block until the copy finishes before the
        // reservation drops. The scheduler is a dedicated synchronous thread, so
        // this brief wait costs nothing it could spend elsewhere. (A RemoteFetch
        // phase has no local reservation yet — pegaflow owns that fetch and its
        // pinned-pool destination; dropping the state simply orphans the query,
        // which pegaflow's own req-scoped prefetch GC cleans up.)
        if let Some(state) = self.prefetch.remove(&request_id) {
            if let PrefetchPhase::Loading { handle, .. } = state.phase {
                let _ = handle.wait();
            }
        }
        self.saved_cursor.remove(&request_id);
        if self.speculative.is_some() {
            self.dflash_ready_requests.remove(&request_id);
            self.primary.drop_dflash_request(request_id)?;
        }
        Ok(())
    }

    fn begin_kv_prefetch(
        &mut self,
        request_id: RequestId,
        prompt_tokens: &[u32],
        lora_adapter: Option<&str>,
        reserve_floor: usize,
    ) -> bool {
        let Some(offload) = self.offload.as_ref() else {
            return false;
        };
        if !self.prefix_cache_enabled() {
            return false;
        }
        if self.l1_retention_disabled {
            // Pure-L2 mode: drop any cross-request HBM retention so the probe
            // sees gpu_hit == 0 and queries the whole cacheable prefix from the
            // host tier. Only inactive (completed, unheld) blocks are drained —
            // the current request holds nothing yet, and in-flight prefetches
            // keep their reserved blocks, so this never touches live KV.
            self.kv_mgr.pool().evict_inactive();
        }
        if self.vllm_compat.is_some() && lora_adapter.is_some() {
            // vLLM salts LoRA block hashes via extra_keys; that derivation is
            // not replicated, so LoRA requests skip the cross-engine lookup.
            return false;
        }
        let probe = self
            .kv_mgr
            .pool()
            .probe_prefix(prompt_tokens.to_vec(), lora_adapter);
        let query_hashes = match &self.vllm_compat {
            None => probe.cpu_query_hashes(),
            // Same query window ([gpu_hit .. cacheable) blocks of the prompt),
            // keyed with vLLM's hash scheme so the lookup can find what the
            // vLLM prefill peer registered. Local GPU-tier naming stays kvbm:
            // the loaded bytes are committed under the probe's own hashes.
            Some(compat) => {
                let window = probe.cpu_query_window();
                let start = probe.gpu_hit_blocks();
                // In bounds by construction: the probe's reuse cap leaves the
                // prompt's final token out, so start + window ≤ ⌊len/bs⌋ =
                // chain.len() even for block-aligned prompts.
                let chain = compat.hasher.key_chain(prompt_tokens);
                chain[start..start + window].to_vec()
            }
        };
        if query_hashes.is_empty() {
            return false;
        }
        // Breaker open: the peer demonstrably isn't publishing, so treat a
        // zero hit as a plain miss instead of parking for the whole window,
        // and don't park on `Loading` either — in compat mode the first shot
        // is always `Loading` (the query only starts the async fetch), so
        // parking would stall every cold request for the full fetch deadline.
        // The first-shot query below still runs — a hit re-arms waiting.
        let expect_remote = self
            .vllm_compat
            .as_ref()
            .is_some_and(|c| c.consecutive_miss_windows < MISS_BREAKER_THRESHOLD);
        let park_on_loading = self.vllm_compat.is_none() || expect_remote;
        let available_blocks = self.kv_mgr.pool().available_blocks();
        let action = remote_fetch_action(
            false,
            expect_remote,
            park_on_loading,
            || {
                offload
                    // As in the re-query: partial hits shorten the local
                    // prefill, so don't wait for the full prefix.
                    .query(&request_id.raw().to_string(), &query_hashes, false)
                    .map(QueryView::from)
                    .map_err(|e| {
                        log::warn!("KV offload query failed for {request_id:?} (skipping): {e}");
                    })
            },
            available_blocks,
            reserve_floor,
        );
        match action {
            RemoteFetchAction::Wait => {
                // pegaflow is pulling the missing prefix from a P2P peer (or
                // SSD) into the local host tier — or, in vLLM-compat mode, the
                // producer's registration hasn't landed yet. Park the request
                // and re-query each tick; the probe keeps the GPU-hit prefix
                // resident.
                let now = std::time::Instant::now();
                let miss_wait = self
                    .vllm_compat
                    .as_ref()
                    .map_or(std::time::Duration::ZERO, |c| c.miss_wait);
                self.prefetch.insert(
                    request_id,
                    PrefetchState {
                        probe,
                        phase: PrefetchPhase::RemoteFetch {
                            query_hashes,
                            deadline: now + REMOTE_FETCH_DEADLINE,
                            miss_deadline: now + miss_wait,
                            parked_at: now,
                            last_query: now,
                        },
                    },
                );
                true
            }
            RemoteFetchAction::Scratch => false, // miss or query error
            RemoteFetchAction::Release(lease) => {
                self.note_remote_hit();
                let offload = self.offload.as_ref().expect("offload checked above");
                offload.release_query_lease(lease);
                false
            }
            RemoteFetchAction::Load(lease, num_blocks) => {
                self.note_remote_hit();
                match self.start_prefetch_load(request_id, probe, lease, num_blocks) {
                    Ok(()) => true,
                    Err(()) => false,
                }
            }
        }
    }

    fn drain_ready_prefetch(&mut self, reserve_floor: usize) -> Vec<RequestId> {
        let ids: Vec<RequestId> = self.prefetch.keys().copied().collect();
        let mut done = Vec::new();
        for id in ids {
            match self.prefetch.get_mut(&id).map(|st| &mut st.phase) {
                Some(PrefetchPhase::Loading { handle, .. }) => {
                    let Some(result) = handle.poll() else {
                        continue;
                    };
                    let st = self.prefetch.get_mut(&id).expect("prefetch present");
                    let PrefetchPhase::Loading { reservation, .. } =
                        std::mem::replace(&mut st.phase, PrefetchPhase::Committed)
                    else {
                        unreachable!("phase matched Loading above");
                    };
                    self.settle_prefetch(id, reservation, result);
                    done.push(id);
                }
                Some(PrefetchPhase::RemoteFetch { .. }) => {
                    if self.poll_remote_fetch(id, reserve_floor) {
                        done.push(id);
                    }
                }
                Some(PrefetchPhase::Committed) | None => {} // awaiting prefill
            }
        }
        done
    }

    fn wait_ready_prefetch(&mut self, reserve_floor: usize) -> Vec<RequestId> {
        let mut done = Vec::new();
        if let Some(id) = self
            .prefetch
            .iter()
            .find(|(_, st)| matches!(st.phase, PrefetchPhase::Loading { .. }))
            .map(|(id, _)| *id)
        {
            let st = self.prefetch.get_mut(&id).expect("prefetch present");
            let PrefetchPhase::Loading {
                reservation,
                handle,
            } = std::mem::replace(&mut st.phase, PrefetchPhase::Committed)
            else {
                unreachable!("phase matched Loading above");
            };
            let result = handle.wait();
            self.settle_prefetch(id, reservation, result);
            done.push(id);
        } else if let Some(id) = self
            .prefetch
            .iter()
            .find(|(_, st)| matches!(st.phase, PrefetchPhase::RemoteFetch { .. }))
            .map(|(id, _)| *id)
        {
            // Nothing locally in flight but a remote fetch is: there is no
            // completion handle to block on (pegaflow owns the fetch), so
            // sleep one poll interval to avoid a busy idle loop, then fall
            // through to the sweep, which re-queries it.
            std::thread::sleep(std::time::Duration::from_millis(5));
            let _ = id;
        }
        // Sweep any others that completed concurrently.
        for id in self.drain_ready_prefetch(reserve_floor) {
            if !done.contains(&id) {
                done.push(id);
            }
        }
        done
    }

    fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        // 1. Create RequestKvs (first chunk only), clamp chunk budgets,
        // schedule KV for this step's tokens
        let mut requests = plan.requests.to_vec();
        for req in &mut requests {
            self.schedule_prefill_chunk(req)?;
        }

        // 2. Build KvViews (seq_len = chunk_start + this chunk)
        let kv_views: Vec<KvView> = requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].prefill_view(req.chunk_tokens))
            .collect();

        // 3. Execute forward
        let step = StepCommand::Prefill {
            requests,
            kv_views,
            echo: plan.echo,
            sample_seed: plan.sample_seed,
        };
        let outcome = self.run_step(&step)?;

        // 4. Apply prefill
        let result = match outcome {
            WorkerStepOutcome::Prefill(result) => result,
            other => {
                return Err(anyhow::anyhow!(
                    "prefill returned unexpected: {}",
                    other.kind()
                ));
            }
        };
        for req_result in &result.requests {
            self.apply_prefill_result(req_result)?;
        }
        // A request becomes draft-ready once its prompt is fully prefilled with
        // captured target context. Partial chunks stay pending; ineligible
        // requests drop any stale worker state.
        if self.speculative.is_some() {
            for req_result in &result.requests {
                let captured = result
                    .dflash_context_captured_requests
                    .contains(&req_result.request_id);
                match dflash_prefill_action(captured, req_result.completed) {
                    DFlashPrefillAction::MarkReady => {
                        self.dflash_ready_requests.insert(req_result.request_id);
                    }
                    DFlashPrefillAction::KeepPending => {
                        self.dflash_ready_requests.remove(&req_result.request_id);
                    }
                    DFlashPrefillAction::Drop => {
                        self.dflash_ready_requests.remove(&req_result.request_id);
                        self.primary.drop_dflash_request(req_result.request_id)?;
                    }
                }
            }
        }
        // 5. Offload the blocks this prefill just sealed (post-step-sync).
        for req_result in &result.requests {
            self.save_sealed_blocks(req_result.request_id);
        }

        Ok(result)
    }

    fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<DecodeResult> {
        if !self.metadata.config.decode_group_is_compiled() {
            let unified = self.execute_unified(UnifiedPlan {
                prefill_requests: &[],
                decode_requests: plan.requests,
                sample_seed: plan.sample_seed,
            })?;
            return Ok(DecodeResult {
                requests: unified.decode_requests,
            });
        }

        // 1. Schedule decode for all active requests
        for req in plan.requests {
            let rkv = self
                .request_kvs
                .get_mut(&req.request_id)
                .ok_or_else(|| anyhow::anyhow!("missing RequestKv for {:?}", req.request_id))?;
            rkv.schedule_decode(self.kv_mgr.pool()).map_err(|e| {
                anyhow::anyhow!("schedule_decode failed for {:?}: {e}", req.request_id)
            })?;
        }

        // 2. Build KvViews
        let kv_views: Vec<KvView> = plan
            .requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].decode_view())
            .collect();

        // 3. Execute forward
        let step = StepCommand::Decode {
            requests: plan.requests.to_vec(),
            kv_views,
            sample_seed: plan.sample_seed,
        };
        let outcome = self.run_step(&step)?;

        // 4. Apply decode
        let result = match outcome {
            WorkerStepOutcome::Decode(result) => result,
            other => {
                return Err(anyhow::anyhow!(
                    "decode returned unexpected: {}",
                    other.kind()
                ));
            }
        };
        for req_result in &result.requests {
            let rkv = self
                .request_kvs
                .get_mut(&req_result.request_id)
                .expect("request must exist after decode");
            rkv.apply_decode(req_result.token, self.kv_mgr.pool())?;
        }
        // A plain decode advances the sequence outside the speculative path, so
        // any captured draft context is now stale — drop it.
        if self.speculative.is_some() {
            for req_result in &result.requests {
                if self.dflash_ready_requests.remove(&req_result.request_id) {
                    self.primary.drop_dflash_request(req_result.request_id)?;
                }
            }
        }
        // 5. Offload any block this decode step just sealed (post-step-sync).
        for req_result in &result.requests {
            self.save_sealed_blocks(req_result.request_id);
        }

        Ok(result)
    }

    fn execute_speculative_draft(&mut self, plan: DraftPlan<'_>) -> Result<DraftResult> {
        self.execute_speculative_draft_impl(plan)
    }

    fn execute_speculative_verify(&mut self, plan: VerifyPlan<'_>) -> Result<VerifyResult> {
        self.execute_speculative_verify_impl(plan)
    }

    fn speculative_enabled(&self) -> bool {
        self.speculative.is_some()
    }

    fn speculative_request_ready(&self, request_id: RequestId) -> bool {
        self.dflash_ready_requests.contains(&request_id)
    }

    fn execute_unified(&mut self, plan: UnifiedPlan<'_>) -> Result<UnifiedResult> {
        // The scheduler resolves any prior async prefill before this step; a
        // pending one here is a broken contract. Checked before any KV commit.
        anyhow::ensure!(
            self.async_prefill.is_none(),
            "async prefill invariant violated: a previous prefill is still pending at a new unified step"
        );
        // Low-level callers can bypass the startup guard; LoRA prefill scratch is
        // unordered across ctx.stream and the overlap stream.
        anyhow::ensure!(
            self.overlap.is_none()
                || plan
                    .prefill_requests
                    .iter()
                    .all(|request| request.lora_adapter.is_none()),
            "LoRA prefill does not support decode-overlap"
        );

        // 1. Create RequestKvs for prefill requests (first chunk only), clamp
        // chunk budgets, schedule KV for this step's tokens
        let mut prefill_requests = plan.prefill_requests.to_vec();
        for req in &mut prefill_requests {
            self.schedule_prefill_chunk(req)?;
        }

        // Schedule decode for active requests
        for req in plan.decode_requests {
            let rkv = self
                .request_kvs
                .get_mut(&req.request_id)
                .ok_or_else(|| anyhow::anyhow!("missing RequestKv for {:?}", req.request_id))?;
            rkv.schedule_decode(self.kv_mgr.pool()).map_err(|e| {
                anyhow::anyhow!("schedule_decode failed for {:?}: {e}", req.request_id)
            })?;
        }

        // 2. Build KvViews
        let prefill_kv_views: Vec<KvView> = prefill_requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].prefill_view(req.chunk_tokens))
            .collect();
        let decode_kv_views: Vec<KvView> = plan
            .decode_requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].decode_view())
            .collect();

        // 3. Execute forward — use split-concurrent if overlap streams are active
        let step = if let Some(ref overlap) = self.overlap {
            StepCommand::SplitConcurrent {
                prefill_requests,
                prefill_kv_views,
                decode_requests: plan.decode_requests.to_vec(),
                decode_kv_views,
                prefill_stream: overlap.prefill_stream,
                decode_stream: overlap.decode_stream,
                sample_seed: plan.sample_seed,
            }
        } else {
            StepCommand::Unified {
                prefill_requests,
                prefill_kv_views,
                decode_requests: plan.decode_requests.to_vec(),
                decode_kv_views,
                sample_seed: plan.sample_seed,
            }
        };
        let outcome = self.run_step(&step)?;

        // 4. Apply results
        match outcome {
            WorkerStepOutcome::Unified(result) => {
                // Normal unified path: both results ready
                for req_result in &result.prefill_requests {
                    self.apply_prefill_result(req_result)?;
                }
                for req_result in &result.decode_requests {
                    let rkv = self
                        .request_kvs
                        .get_mut(&req_result.request_id)
                        .expect("request must exist after unified decode");
                    rkv.apply_decode(req_result.token, self.kv_mgr.pool())?;
                }
                // A plain decode via the fused unified step advances the sequence
                // outside the speculative path, so any captured draft context is
                // now stale — drop it, mirroring execute_decode. (Eligible pending
                // are routed to a dedicated prefill step, so unified prefills never
                // need DFlash mark-ready here.)
                if self.speculative.is_some() {
                    for req_result in &result.decode_requests {
                        if self.dflash_ready_requests.remove(&req_result.request_id) {
                            self.primary.drop_dflash_request(req_result.request_id)?;
                        }
                    }
                }
                for req_result in &result.prefill_requests {
                    self.save_sealed_blocks(req_result.request_id);
                }
                for req_result in &result.decode_requests {
                    self.save_sealed_blocks(req_result.request_id);
                }
                Ok(result)
            }
            WorkerStepOutcome::SplitDecodeReady {
                decode: decode_result,
                prefill_event: event,
            } => {
                // Prefill may still be in flight (the `event` signals completion),
                // writing this step's KV pages. An error return releases those pages,
                // so sync the event first; success leaves the prefill in flight.
                let applied =
                    decode_result
                        .requests
                        .iter()
                        .try_for_each(|req_result| -> Result<()> {
                            let rkv = self
                                .request_kvs
                                .get_mut(&req_result.request_id)
                                .expect("request must exist after split decode");
                            rkv.apply_decode(req_result.token, self.kv_mgr.pool())?;
                            Ok(())
                        });
                if let Err(e) = applied {
                    let sync = unsafe { cudarc::driver::sys::cuEventSynchronize(event.cu_event()) };
                    if sync != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                        log::error!(
                            "FATAL: cuEventSynchronize(prefill) failed on the decode-error \
                             path ({sync:?}); aborting to avoid releasing KV pages the \
                             prefill stream may still be writing"
                        );
                        std::process::abort();
                    }
                    let rx = self.primary.resolve_prefill()?;
                    rx.recv().map_err(|_| {
                        anyhow::anyhow!("worker dropped resolve_prefill on decode-error path")
                    })??;
                    return Err(e);
                }
                for req_result in &decode_result.requests {
                    self.save_sealed_blocks(req_result.request_id);
                }
                self.async_prefill = Some(event);
                // Return a UnifiedResult with empty prefill — scheduler will
                // get prefill results via poll_async_prefill.
                Ok(UnifiedResult {
                    prefill_requests: Vec::new(),
                    decode_requests: decode_result.requests,
                })
            }
            other => Err(anyhow::anyhow!(
                "unified returned unexpected: {}",
                other.kind()
            )),
        }
    }

    fn load_lora_adapter(&mut self, request: &LoadLoraAdapterRequest) -> Result<()> {
        ensure_lora_capacity(
            &self.loaded_lora_adapters,
            &request.lora_name,
            self.lora_options.max_loras,
            request.load_inplace,
        )?;
        let adapter = crate::lora::load_lora_adapter(
            &request.lora_path,
            &self.metadata.config,
            self.lora_options.max_lora_rank,
        )?;
        let world_size = self.workers.len() + 1;
        let projection_count: usize = adapter
            .layers
            .iter()
            .map(|layer| layer.projections.len())
            .sum();
        let element_count: usize = adapter
            .layers
            .iter()
            .flat_map(|layer| layer.projections.values())
            .map(|projection| projection.a.data.len() + projection.b.data.len())
            .sum();
        let shape_elems: usize = adapter
            .layers
            .iter()
            .flat_map(|layer| layer.projections.values())
            .map(|projection| {
                projection.a.rows * projection.a.cols + projection.b.rows * projection.b.cols
            })
            .sum();
        debug_assert_eq!(element_count, shape_elems);
        let rank = adapter.manifest.rank;
        let targets = adapter.manifest.target_modules.join(", ");
        let path = adapter.manifest.path.display().to_string();
        let mut sharded_adapters = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            sharded_adapters.push(adapter.shard_for_tensor_parallel(
                &self.metadata.config,
                TensorParallelConfig { rank, world_size },
            )?);
        }

        let mut sharded_adapters = sharded_adapters.into_iter();
        let primary_adapter = sharded_adapters
            .next()
            .expect("rank 0 adapter must exist for nonzero world_size");
        let primary_response = self.primary.load_lora_adapter(
            request.lora_name.clone(),
            primary_adapter,
            request.load_inplace,
        )?;
        let mut pending = Vec::with_capacity(self.workers.len());
        let mut errors = Vec::new();
        for (index, worker) in self.workers.iter().enumerate() {
            let rank = index + 1;
            let rank_adapter = sharded_adapters
                .next()
                .expect("worker adapter must exist for every tensor-parallel rank");
            match worker.load_lora_adapter(
                request.lora_name.clone(),
                rank_adapter,
                request.load_inplace,
            ) {
                Ok(response) => pending.push((rank, response)),
                Err(err) => errors.push(format!("rank {rank} dispatch: {err:#}")),
            }
        }

        match primary_response.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => errors.push(format!("rank 0: {err:#}")),
            Err(_) => errors.push("rank 0: dropped LoRA load response".to_string()),
        }
        for (rank, response) in pending {
            match response.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => errors.push(format!("rank {rank}: {err:#}")),
                Err(_) => errors.push(format!("rank {rank}: dropped LoRA load response")),
            }
        }
        if !errors.is_empty() {
            let mut cleanup_errors = Vec::new();
            match self.primary.discard_lora_adapter(request.lora_name.clone()) {
                Ok(response) => match response.recv() {
                    Ok(Ok(())) => {}
                    Ok(Err(err)) => cleanup_errors.push(format!("rank 0 cleanup: {err:#}")),
                    Err(_) => cleanup_errors
                        .push("rank 0 cleanup: dropped LoRA discard response".to_string()),
                },
                Err(err) => cleanup_errors.push(format!("rank 0 cleanup dispatch: {err:#}")),
            }
            for (index, worker) in self.workers.iter().enumerate() {
                let rank = index + 1;
                match worker.discard_lora_adapter(request.lora_name.clone()) {
                    Ok(response) => match response.recv() {
                        Ok(Ok(())) => {}
                        Ok(Err(err)) => {
                            cleanup_errors.push(format!("rank {rank} cleanup: {err:#}"));
                        }
                        Err(_) => cleanup_errors.push(format!(
                            "rank {rank} cleanup: dropped LoRA discard response"
                        )),
                    },
                    Err(err) => {
                        cleanup_errors.push(format!("rank {rank} cleanup dispatch: {err:#}"));
                    }
                }
            }
            if cleanup_errors.is_empty() {
                self.loaded_lora_adapters.remove(&request.lora_name);
            }
            let cleanup_suffix = if cleanup_errors.is_empty() {
                String::new()
            } else {
                format!("; cleanup errors: {}", cleanup_errors.join("; "))
            };
            anyhow::bail!(
                "failed to load Qwen3 LoRA adapter {} on tensor-parallel ranks: {}{}",
                request.lora_name,
                errors.join("; "),
                cleanup_suffix
            );
        }

        log::info!(
            "Loaded Qwen3 LoRA adapter {} from {} (rank={}, targets={}, projections={}, bf16_elements={}, tp_world_size={}, load_inplace={})",
            request.lora_name,
            path,
            rank,
            targets,
            projection_count,
            element_count,
            world_size,
            request.load_inplace
        );
        self.loaded_lora_adapters.insert(request.lora_name.clone());
        Ok(())
    }

    fn unload_lora_adapter(&mut self, request: &UnloadLoraAdapterRequest) -> Result<()> {
        let primary_response = self
            .primary
            .unload_lora_adapter(request.lora_name.clone())?;
        let mut pending = Vec::with_capacity(self.workers.len());
        for (index, worker) in self.workers.iter().enumerate() {
            pending.push((
                index + 1,
                worker.unload_lora_adapter(request.lora_name.clone())?,
            ));
        }

        let mut errors = Vec::new();
        match primary_response.recv() {
            Ok(Ok(())) => {}
            Ok(Err(err)) => errors.push(format!("rank 0: {err:#}")),
            Err(_) => errors.push("rank 0: dropped LoRA unload response".to_string()),
        }
        for (rank, response) in pending {
            match response.recv() {
                Ok(Ok(())) => {}
                Ok(Err(err)) => errors.push(format!("rank {rank}: {err:#}")),
                Err(_) => errors.push(format!("rank {rank}: dropped LoRA unload response")),
            }
        }
        if !errors.is_empty() {
            anyhow::bail!(
                "failed to unload Qwen3 LoRA adapter {} on tensor-parallel ranks: {}",
                request.lora_name,
                errors.join("; ")
            );
        }

        log::info!("Unloaded Qwen3 LoRA adapter {}", request.lora_name);
        self.loaded_lora_adapters.remove(&request.lora_name);
        Ok(())
    }

    fn list_lora_adapters(&self) -> Vec<String> {
        let mut names: Vec<_> = self.loaded_lora_adapters.iter().cloned().collect();
        names.sort();
        names
    }

    fn has_decode_overlap(&self) -> bool {
        self.overlap.is_some()
    }

    fn poll_async_prefill(&mut self) -> Option<PrefillResult> {
        let status =
            unsafe { cudarc::driver::sys::cuEventQuery(self.async_prefill.as_ref()?.cu_event()) };
        match status {
            cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => return None,
            cudarc::driver::sys::CUresult::CUDA_SUCCESS => {}
            other => {
                log::error!(
                    "FATAL: cuEventQuery(prefill poll) returned {other:?}; async prefill/context \
                     state untrustworthy — aborting"
                );
                std::process::abort();
            }
        }
        let _ = self.async_prefill.take();
        let rx = self.primary.resolve_prefill().unwrap_or_else(|e| {
            log::error!(
                "FATAL: resolve_prefill failed after the async prefill completed ({e}); aborting"
            );
            std::process::abort();
        });
        let result = match rx.recv() {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => {
                log::error!("FATAL: worker resolve_prefill errored ({e}); aborting");
                std::process::abort();
            }
            Err(_) => {
                log::error!("FATAL: worker dropped the resolve_prefill response; aborting");
                std::process::abort();
            }
        };
        for req_result in &result.requests {
            self.apply_prefill_result(req_result).unwrap_or_else(|e| {
                log::error!("FATAL: apply_prefill_result failed ({e}); aborting");
                std::process::abort();
            });
        }
        for req_result in &result.requests {
            self.save_sealed_blocks(req_result.request_id);
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::ensure_lora_capacity;

    #[test]
    fn lora_capacity_rejects_new_adapter_at_limit() {
        let loaded = HashSet::from(["adapter-a".to_string()]);

        let error = ensure_lora_capacity(&loaded, "adapter-b", 1, false)
            .expect_err("new adapter should exceed capacity")
            .to_string();

        assert!(error.contains("max_loras=1"));
        assert!(error.contains("requested=adapter-b"));
    }

    #[test]
    fn lora_capacity_allows_existing_adapter_replacement_at_limit_with_load_inplace() {
        let loaded = HashSet::from(["adapter-a".to_string()]);

        ensure_lora_capacity(&loaded, "adapter-a", 1, true)
            .expect("existing adapter should fit with load_inplace");
    }

    #[test]
    fn lora_capacity_rejects_duplicate_without_load_inplace() {
        let loaded = HashSet::from(["adapter-a".to_string()]);

        let error = ensure_lora_capacity(&loaded, "adapter-a", 1, false)
            .expect_err("duplicate without load_inplace should fail")
            .to_string();

        assert!(error.contains("already loaded"));
    }
}

impl Drop for Qwen3Executor {
    fn drop(&mut self) {
        self.primary.shutdown();
        for worker in &mut self.workers {
            worker.shutdown();
        }
    }
}

/// What the executor learns about the draft model after loading it on the
/// worker: the draft block size (`K` candidates per round) and which target
/// layers feed the draft (the worker captures these; kept for diagnostics).
#[derive(Clone, Debug)]
struct DFlashMeta {
    block_size: usize,
    /// Max drafts proposed per verify step (`K`). The span leads with the anchor
    /// (`[anchor, draft_1, …]`), so this is `verify_span - 1` under either block
    /// layout — anchor-first vs anchor-drop is already folded into `verify_span()`.
    num_spec_tokens: usize,
    /// Draft's max cacheable position; with the `block_size` in-fill headroom
    /// this caps the DFlash-effective context to `max_position_embeddings - block_size`.
    max_position_embeddings: usize,
    #[allow(dead_code)]
    target_layer_ids: Vec<usize>,
}

struct LocalQwen3Lane {
    /// Before `model` on purpose: NCCL comm teardown polls until every graph
    /// that recorded its collectives is destroyed, so these graphs must drop
    /// before `model.tp_comm`.
    bufs: BatchDecodeBuffers,
    model: Qwen3Model,
    kv_buffer: KvBuffer,
    layout: KvLayout,
    sample_scratch: pegainfer_sample::SampleScratch,
    /// Request-local decode steps handed to `select_batch`, reused across
    /// steps to keep the sampling hot path allocation-free. All zeros until
    /// the scheduler wires generated counts through (sampling-parity 1b).
    steps_buf: Vec<u64>,
    /// Prefill-chunk token cap; bounds the Pin self-check's unified-N envelope in `bind`.
    max_prefill_tokens: usize,
    /// In-flight prefill from a previous SplitConcurrent step (not yet synced).
    inflight_prefill: Option<InflightPrefillState>,
    /// DFlash draft lane (the draft model + per-request draft state). `None`
    /// unless speculative decoding is enabled; only the primary rank carries it.
    dflash: Option<DFlashLaneState>,
    /// Fixed, pre-allocated scratch for the DFlash verify forward. Lazily built
    /// on the first verify step (its shape depends on the loaded draft model's
    /// block size and the target's capture layers). Pointer-stable for the
    /// upcoming verify CUDA Graph.
    verify_bufs: Option<VerifyGraphBuffers>,
    /// KV pool block count — the worst-case page-list bound for `verify_bufs`.
    total_blocks: usize,
    /// Shared dead page backing both production padding rows and the sweep's
    /// synthetic rows.
    padding_block_id: i32,
    /// Set by the sweep's `Finalize`; arms the TP replay fail-loud in serving.
    precapture_complete: bool,
}

/// Stored state for an async prefill that was launched but not yet synced.
struct InflightPrefillState {
    /// Field order is load-bearing: `temp_bin` must drop before `prefill_logits`.
    temp_bin: crate::prefill::PrefillTempBin,
    prefill_logits: HiddenStates,
    prefill_requests: Vec<PrefillStepItem>,
    /// Per-step sampling seed captured when the prefill was launched, replayed
    /// when its tokens are sampled after the deferred sync.
    sample_seed: u64,
}

// SAFETY: InflightPrefillState lives entirely within the worker thread that
// owns the GPU context. It is never shared across threads.
unsafe impl Send for InflightPrefillState {}

/// Drains the decode stream on scope exit (success, `?`, or panic) before this
/// step's KV pages can be released; a failed drain aborts (fail-closed).
struct DecodeStreamGuard {
    stream: cudarc::driver::sys::CUstream,
}

impl Drop for DecodeStreamGuard {
    fn drop(&mut self) {
        let r = unsafe { cudarc::driver::sys::cuStreamSynchronize(self.stream) };
        if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            log::error!(
                "FATAL: cuStreamSynchronize(decode) failed ({r:?}); aborting rather than \
                 release KV pages the decode stream may still be reading"
            );
            std::process::abort();
        }
    }
}

impl LocalQwen3Lane {
    fn new(
        model: Qwen3Model,
        kv_buffer: KvBuffer,
        total_blocks: usize,
        padding_block_id: i32,
        max_prefill_tokens: usize,
    ) -> Result<Self> {
        let buf_layout = kv_buffer.layout();
        let layout = KvLayout::new(
            buf_layout.num_layers,
            buf_layout.num_kv_heads,
            buf_layout.head_dim,
            buf_layout.page_size,
        )
        .expect("kv layout geometry");
        let max_bucket = *BATCH_BUCKETS.last().unwrap();
        let bufs = BatchDecodeBuffers::new(
            model.device_ctx(),
            model.config().hidden_size,
            model.local_q_dim(),
            model.local_kv_dim(),
            model.local_intermediate_size(),
            model.config().vocab_size,
            max_bucket,
            buf_layout.page_size,
            padding_block_id,
            model.local_num_attention_heads(),
            model.config().max_position_embeddings,
            model.fused_decode_qkv(),
        )?;
        let sample_scratch = pegainfer_sample::SampleScratch::new(
            model.device_ctx(),
            model.config().vocab_size,
            max_bucket,
        )?;
        Ok(Self {
            model,
            kv_buffer,
            layout,
            bufs,
            sample_scratch,
            steps_buf: Vec::new(),
            max_prefill_tokens,
            inflight_prefill: None,
            dflash: None,
            verify_bufs: None,
            total_blocks,
            padding_block_id,
            precapture_complete: false,
        })
    }

    /// One phase of the TP startup sweep ([`PrecapturePhase`]). Synthetic rows
    /// are token 0 at position 0 over the shared padding page; outputs unused.
    fn precapture_phase(&mut self, phase: PrecapturePhase) -> Result<()> {
        match phase {
            PrecapturePhase::Warmup => self.model.warmup_tp_collective(),
            PrecapturePhase::Capture { bucket_idx } => {
                self.precapture_decode(bucket_idx, DecodeGraphUse::CaptureOnly)
            }
            PrecapturePhase::Launch { bucket_idx } => {
                self.precapture_decode(bucket_idx, DecodeGraphUse::Replay)
            }
            PrecapturePhase::Finalize => {
                for (bucket_idx, &bucket) in BATCH_BUCKETS.iter().enumerate() {
                    let path = BatchDecodeBuffers::attention_path(
                        bucket,
                        self.bufs.policy_at_construction,
                    );
                    let graph_idx = BatchDecodeBuffers::graph_index(bucket_idx, path);
                    anyhow::ensure!(
                        self.bufs.graphs[graph_idx].is_captured(),
                        "TP decode graph pre-capture left bucket {bucket} ({path:?}) uncaptured"
                    );
                }
                self.precapture_complete = true;
                Ok(())
            }
        }
    }

    fn precapture_decode(&mut self, bucket_idx: usize, graph_use: DecodeGraphUse) -> Result<()> {
        let bucket = BATCH_BUCKETS[bucket_idx];
        let token_ids = vec![0u32; bucket];
        let kv_views: Vec<KvView> = (0..bucket)
            .map(|_| KvView::new(vec![self.padding_block_id], 1, self.layout.page_size))
            .collect();
        let lora_adapters: Vec<Option<&str>> = vec![None; bucket];
        self.model.batch_decode(
            &token_ids,
            &kv_views,
            &lora_adapters,
            self.kv_buffer.buffer(),
            &self.layout,
            &mut self.bufs,
            graph_use,
        )?;
        // Capture acks only after the async cuGraphUpload lands; Launch acks
        // only after the collectives drained.
        self.model.device_ctx().stream.synchronize()?;
        Ok(())
    }

    fn dump_decode_graph_png(&mut self, png_path: &Path) -> Result<CudaGraphDumpSummary> {
        let bucket_idx = 0;
        let bucket = BATCH_BUCKETS[bucket_idx];
        let attention_path =
            BatchDecodeBuffers::attention_path(bucket, self.bufs.policy_at_construction);
        let graph_idx = BatchDecodeBuffers::graph_index(bucket_idx, attention_path);
        if !self.bufs.graphs[graph_idx].is_captured() {
            anyhow::ensure!(
                !self.model.tp_graph_enabled(),
                "Qwen3 TP batch-1 graph was not captured by the startup sweep"
            );
            self.precapture_decode(bucket_idx, DecodeGraphUse::CaptureOnly)?;
        }
        let title = format!("Qwen3 decode CUDA Graph · bs={bucket} · {attention_path:?}");
        self.bufs.graphs[graph_idx]
            .dump_png(png_path, &title)
            .with_context(|| {
                format!("dump Qwen3 rank-0 batch-{bucket} {attention_path:?} decode CUDA Graph")
            })
    }

    /// Load the DFlash draft model into this lane (primary rank only). The draft
    /// model is built here on the worker thread because it reads the co-located
    /// target model's embeddings and head.
    fn load_dflash(&mut self, draft_path: &str) -> Result<DFlashMeta> {
        let model = DFlashDraftModel::from_safetensors_for_target(
            self.model.device_ctx(),
            draft_path,
            &self.model,
        )?;
        model.tune_gemm_algos(&self.model)?;
        let meta = DFlashMeta {
            block_size: model.block_size(),
            num_spec_tokens: model.verify_span().saturating_sub(1),
            max_position_embeddings: model.max_position_embeddings(),
            target_layer_ids: model.target_layer_ids().to_vec(),
        };
        let max_decode_batch_size = *BATCH_BUCKETS.last().unwrap();
        self.dflash = Some(DFlashLaneState::new(
            self.model.device_ctx(),
            model,
            max_decode_batch_size,
        )?);
        Ok(meta)
    }

    fn bind(&self) -> Result<CublasThreadGuard> {
        bind_model_thread(&self.model)?;
        let guard = CublasThreadGuard;
        tune_decode_gemm_algos(&self.model, self.max_prefill_tokens, true)?;
        Ok(guard)
    }

    /// Sync the in-flight prefill stream and sample prefill tokens.
    fn resolve_inflight_prefill(&mut self) -> Result<PrefillResult> {
        let mut state = self
            .inflight_prefill
            .take()
            .ok_or_else(|| anyhow::anyhow!("no inflight prefill to resolve"))?;

        state.temp_bin.synchronize();

        // Sample prefill tokens
        let params: Vec<&SamplingParams> =
            state.prefill_requests.iter().map(|r| &r.params).collect();
        let tokens = self.select_step_tokens(&state.prefill_logits, &params, state.sample_seed)?;

        // Build prefill result
        let results = build_prefill_request_results(
            self,
            &state.prefill_requests,
            &state.prefill_logits,
            &tokens,
            None,
            false,
        )?;

        // Split-concurrent prefill never runs with DFlash (capture needs the
        // synchronous result), so no context is captured here.
        Ok(PrefillResult {
            requests: results,
            dflash_context_captured_requests: Vec::new(),
        })
    }

    /// Pick one token per logits column (batched argmax for greedy rows,
    /// one batched sampler call for non-greedy rows). Grows the sampling
    /// scratch when a step is wider than the decode bucket it was sized for.
    fn select_step_tokens(
        &mut self,
        logits: &HiddenStates,
        params: &[&SamplingParams],
        sample_seed: u64,
    ) -> Result<Vec<u32>> {
        if params.len() > self.sample_scratch.max_rows() {
            self.sample_scratch = pegainfer_sample::SampleScratch::new(
                self.model.device_ctx(),
                self.model.config().vocab_size,
                params.len(),
            )?;
        }
        self.steps_buf.clear();
        self.steps_buf.resize(params.len(), 0);
        pegainfer_sample::select_batch(
            self.model.device_ctx(),
            logits,
            params,
            &self.steps_buf,
            sample_seed,
            &mut self.sample_scratch,
        )
    }

    fn execute_prefill(
        &mut self,
        prompts: &[&[u32]],
        kv_views: &[KvView],
        lora_adapters: &[Option<&str>],
        echo: bool,
        capture_layer_ids: Option<&[usize]>,
    ) -> Result<(HiddenStates, Option<HiddenStates>, Option<HiddenStates>)> {
        self.model.batch_prefill(
            prompts,
            kv_views,
            lora_adapters,
            self.kv_buffer.buffer(),
            &self.layout,
            echo,
            capture_layer_ids,
        )
    }

    /// Parallel multi-chain hedge (`PEGAINFER_SPEC_HEDGE`): verify each hedged
    /// request's greedy chain (chain A) **and** its hedge-ladder alternative
    /// chains in one expanded forward, then commit whichever chain the target
    /// accepts further. Each hedge chain's span KV is written to lane-owned
    /// scratch pages via a synthesized [`KvView`] (prefix pages shared
    /// read-only, the partial committed page copied first); on a hedge win the
    /// touched pages are copied back into the request's reservation before
    /// `apply_speculative`, and the winning chain's captured hidden rows are
    /// compacted onto chain A's offsets so
    /// `record_verify_dflash_context` sees a plain N-request layout. The KV
    /// transaction interface is untouched. Returns `Ok(None)` when nothing is
    /// hedgeable this round (caller falls back to the plain pass).
    fn try_execute_hedged_verify(
        &mut self,
        requests: &[VerifyStepItem],
        kv_views: &[KvView],
        capture_layer_ids: &[usize],
        sample_seed: u64,
        bufs: &mut VerifyGraphBuffers,
    ) -> Result<Option<VerifyResult>> {
        let page_size = self.layout.page_size;
        let scratch_end = self.kv_buffer.num_blocks();
        let max_batch = *BATCH_BUCKETS.last().unwrap();
        let cap = self.dflash.as_ref().map_or(0, DFlashLaneState::hedge_cap);
        if cap == 0 || scratch_end == self.total_blocks {
            return Ok(None);
        }

        let ctx = self.model.device_ctx().clone();
        // Single pass: expanded layout keeps the N chain-A spans first
        // (offsets unchanged) and appends every accepted hedge span, walking a
        // scratch-page cursor. `hedge_spans[slot]` = (request idx, replaced
        // (original page, scratch page) pairs) for the winner copy-back.
        let mut expanded: Vec<VerifyStepItem> = requests.to_vec();
        let mut views: Vec<KvView> = kv_views.to_vec();
        let mut hedge_spans: Vec<(usize, Vec<(usize, usize)>)> = Vec::new();
        let mut next_scratch = self.total_blocks;
        {
            let Some(dflash) = self.dflash.as_ref() else {
                return Ok(None);
            };
            if !dflash.model.uses_markov_head() {
                return Ok(None);
            }
            let drafts_start = usize::from(!dflash.model.anchor_first());
            let pages_per_chain =
                dflash_lane::hedge_pages_per_chain(dflash.model.block_size(), page_size);
            let mut hedged_requests = 0usize;
            for (idx, req) in requests.iter().enumerate() {
                if hedged_requests == cap {
                    break;
                }
                if !req.params.is_greedy() || req.token_ids.len() < 2 {
                    continue;
                }
                let Some(chains) = dflash.hedge_blocks.get(&req.request_id) else {
                    continue;
                };
                let v = &kv_views[idx];
                let committed = v.seq_len() - req.token_ids.len();
                let first_span_page = committed / page_size;
                let span_pages = v.page_indices().len() - first_span_page;
                if span_pages > pages_per_chain {
                    continue;
                }
                let keep = req.token_ids.len() - 1;
                let mut added: Vec<Vec<u32>> = Vec::new();
                for block in chains {
                    if expanded.len() == max_batch || next_scratch + span_pages > scratch_end {
                        break;
                    }
                    let mut ids = Vec::with_capacity(keep + 1);
                    ids.push(req.token_ids[0]);
                    ids.extend(block[drafts_start..].iter().take(keep).copied());
                    if ids.len() != req.token_ids.len()
                        || ids[1..] == req.token_ids[1..]
                        || added.iter().any(|prev| prev[1..] == ids[1..])
                    {
                        continue;
                    }
                    let mut pages = v.page_indices().to_vec();
                    let mut replaced = Vec::with_capacity(span_pages);
                    for (k, page_slot) in (first_span_page..pages.len()).enumerate() {
                        let orig = pages[page_slot] as usize;
                        let scratch_page = next_scratch + k;
                        if k == 0 && !committed.is_multiple_of(page_size) {
                            // The hedge chain shares this page's committed
                            // prefix rows: give it a full copy to write its
                            // span into.
                            self.kv_buffer.copy_page(&ctx.stream, orig, scratch_page)?;
                        }
                        pages[page_slot] = scratch_page as i32;
                        replaced.push((orig, scratch_page));
                    }
                    next_scratch += span_pages;
                    expanded.push(VerifyStepItem::new(req.request_id, ids.clone(), req.params));
                    views.push(KvView::new(pages, v.seq_len(), page_size));
                    hedge_spans.push((idx, replaced));
                    added.push(ids);
                }
                if !added.is_empty() {
                    hedged_requests += 1;
                }
            }
        }
        if hedge_spans.is_empty() {
            return Ok(None);
        }

        let spans: Vec<&[u32]> = expanded.iter().map(VerifyStepItem::as_slice).collect();
        self.model.batch_prefill_into(
            &spans,
            &views,
            self.kv_buffer.buffer(),
            &self.layout,
            capture_layer_ids,
            bufs,
        )?;
        let params: Vec<&SamplingParams> = expanded
            .iter()
            .flat_map(|req| std::iter::repeat_n(&req.params, req.as_slice().len()))
            .collect();
        let target_tokens = self.select_step_tokens(bufs.all_logits(), &params, sample_seed)?;
        let all_results = build_verify_results(&expanded, &target_tokens)?;
        let (results_a, results_b) = all_results.split_at(requests.len());

        // Per request keep the best-accepting chain; ties keep chain A (no
        // copies). A later chain of the same request only replaces the
        // running winner when strictly better, so the final page/hidden
        // copies always belong to the final winner.
        let a_total_rows: usize = requests.iter().map(|r| r.token_ids.len()).sum();
        let hidden_dim = bufs.captured_hidden().hidden_dim;
        let elem = std::mem::size_of::<half::bf16>();
        let mut final_requests: Vec<VerifyStepItem> = requests.to_vec();
        let mut final_results: Vec<VerifyRequestResult> = results_a.to_vec();
        let mut b_wins = 0usize;
        let mut b_row_offset = a_total_rows;
        for (slot, (idx, replaced)) in hedge_spans.iter().enumerate() {
            let span_len = requests[*idx].token_ids.len();
            let res_b = &results_b[slot];
            if res_b.accepted_tokens.len() > final_results[*idx].accepted_tokens.len() {
                // Winning chain: canonical pages take its span KV, and its
                // captured hidden rows land on A's row offsets in place.
                for &(orig, scratch_page) in replaced {
                    self.kv_buffer.copy_page(&ctx.stream, scratch_page, orig)?;
                }
                let a_row_offset: usize = requests[..*idx].iter().map(|r| r.token_ids.len()).sum();
                let (hid_ptr, _guard) = bufs.captured_hidden().data.device_ptr(&ctx.stream);
                let src = hid_ptr + (b_row_offset * hidden_dim * elem) as u64;
                let dst = hid_ptr + (a_row_offset * hidden_dim * elem) as u64;
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        dst,
                        src,
                        span_len * hidden_dim * elem,
                        ctx.stream.cu_stream(),
                    )
                }
                .map_err(|e| anyhow::anyhow!("hedge hidden compaction failed: {e}"))?;
                b_wins += 1;
                final_requests[*idx] = expanded[requests.len() + slot].clone();
                final_results[*idx] = res_b.clone();
            }
            b_row_offset += span_len;
        }
        let b_extra: usize = final_results
            .iter()
            .zip(results_a)
            .map(|(f, a)| f.accepted_tokens.len() - a.accepted_tokens.len())
            .sum();
        log::debug!(
            "Qwen3 DFlash hedge: {} chain span(s), {b_wins} win event(s), +{b_extra} committed token(s)",
            hedge_spans.len()
        );

        // The expanded forward left seq_len at A+B rows; the winner layout is
        // exactly the N chain-A offsets.
        bufs.captured_hidden_mut().seq_len = a_total_rows;
        self.record_verify_dflash_context(
            &final_requests,
            &final_results,
            Some(bufs.captured_hidden()),
        )?;
        Ok(Some(VerifyResult {
            requests: final_results,
        }))
    }

    /// DFlash verify forward over each request's `block_size`-token span, using
    /// the fixed pre-allocated [`VerifyGraphBuffers`] (no per-step allocation),
    /// lazily built on first use. Numerically equivalent to the
    /// `batch_prefill(echo=true)` verify path it replaces.
    fn execute_dflash_verify(
        &mut self,
        requests: &[VerifyStepItem],
        kv_views: &[KvView],
        sample_seed: u64,
    ) -> Result<VerifyResult> {
        let capture_layer_ids = self.dflash_capture_layer_ids().ok_or_else(|| {
            anyhow::anyhow!("DFlash verify requested but no draft model is loaded")
        })?;
        // Verify span = anchor + drafts: `block_size` for DFlash, `block_size + 1`
        // for DSpark (anchor-first). The graph buffers + replay shape key off this.
        let verify_span = self
            .dflash
            .as_ref()
            .expect("DFlash present when capture layers exist")
            .model
            .verify_span();

        if self.verify_bufs.is_none() {
            let max_batch = *BATCH_BUCKETS.last().unwrap();
            // Each hedged request repeats its full prefix page list once per
            // chain, so the plan's page-index capacity must budget the
            // duplicates on top of the pool-wide worst case.
            // Base page lists are disjoint pool pages (sum <= total_blocks):
            // loading a drafter force-disables the prefix cache (see
            // `load_dflash_draft_model`), so no two active requests share
            // pages on any path that can hedge. Each chain layer duplicates at
            // most one pool-wide list, so the multiplier is the chain count
            // alone, bounded by the expanded verify batch.
            let hedge_dup = self.dflash.as_ref().map_or(0, |dflash| {
                if dflash.hedge_cap() == 0 {
                    0
                } else {
                    dflash.hedge_branch_count().min(max_batch - 1)
                }
            });
            self.verify_bufs = Some(VerifyGraphBuffers::new(
                &self.model,
                max_batch,
                verify_span,
                capture_layer_ids.len(),
                self.total_blocks
                    .saturating_mul(hedge_dup.saturating_add(1)),
            )?);
        }

        // Take the buffers out of `self` so the forward (borrows `&self.model`,
        // `&mut bufs`) and the subsequent sampling (`&mut self.sample_scratch`)
        // and context record (`&mut self.dflash`) don't alias a `self` borrow.
        let mut bufs = self.verify_bufs.take().expect("verify buffers just set");
        let result = (|| -> Result<VerifyResult> {
            if self.dflash.as_ref().is_some_and(|d| d.hedge_cap() > 0) {
                if let Some(result) = self.try_execute_hedged_verify(
                    requests,
                    kv_views,
                    &capture_layer_ids,
                    sample_seed,
                    &mut bufs,
                )? {
                    return Ok(result);
                }
                // Fell back to the plain pass: whatever the draft prepared,
                // this round executed unhedged — the controller must not book
                // ladder cost to a chain count that never ran a verify span.
                if let Some(dflash) = self.dflash.as_mut() {
                    dflash.clear_round_chains();
                }
            }
            let spans: Vec<&[u32]> = requests.iter().map(VerifyStepItem::as_slice).collect();
            self.model.batch_prefill_into(
                &spans,
                kv_views,
                self.kv_buffer.buffer(),
                &self.layout,
                &capture_layer_ids,
                &mut bufs,
            )?;

            // One committed token per span row, each row governed by ITS
            // request's params — argmax rows batch through the fused path,
            // sampled rows ride the regular batched sampler (sampled-verify,
            // #512). Steps stay zeroed exactly like the plain-decode call:
            // when sampling-parity 1b wires request-local decode steps
            // through, verify rows must move with it (row k of a span is the
            // request's step `completion + k`) or seeded replay breaks.
            let params: Vec<&SamplingParams> = requests
                .iter()
                .flat_map(|req| std::iter::repeat_n(&req.params, req.as_slice().len()))
                .collect();
            let target_tokens = self.select_step_tokens(bufs.all_logits(), &params, sample_seed)?;
            let request_results = build_verify_results(requests, &target_tokens)?;

            self.record_verify_dflash_context(
                requests,
                &request_results,
                Some(bufs.captured_hidden()),
            )?;
            Ok(VerifyResult {
                requests: request_results,
            })
        })();
        self.verify_bufs = Some(bufs);
        result
    }

    fn execute_decode(
        &mut self,
        token_ids: &[u32],
        kv_views: &[KvView],
        lora_adapters: &[Option<&str>],
    ) -> Result<()> {
        self.model.batch_decode(
            token_ids,
            kv_views,
            lora_adapters,
            self.kv_buffer.buffer(),
            &self.layout,
            &mut self.bufs,
            DecodeGraphUse::Serve,
        )
    }

    fn execute_unified(
        &mut self,
        prefill_prompts: &[&[u32]],
        prefill_views: &[KvView],
        prefill_lora_adapters: &[Option<&str>],
        decode_tokens: &[u32],
        decode_views: &[KvView],
        decode_lora_adapters: &[Option<&str>],
    ) -> Result<HiddenStates> {
        self.model.unified_step(
            prefill_prompts,
            prefill_views,
            prefill_lora_adapters,
            decode_tokens,
            decode_views,
            decode_lora_adapters,
            &mut self.bufs,
            self.kv_buffer.buffer(),
            &self.layout,
        )
    }

    fn load_lora_adapter(
        &mut self,
        name: String,
        adapter: crate::lora::LoraAdapter,
        load_inplace: bool,
    ) -> Result<()> {
        let device_adapter =
            crate::lora::load_device_lora_adapter(self.model.device_ctx(), name, adapter)?;
        self.model
            .install_lora_adapter(device_adapter, load_inplace)
    }

    fn unload_lora_adapter(&mut self, name: &str) -> Result<()> {
        self.model.uninstall_lora_adapter(name)
    }

    fn discard_lora_adapter(&mut self, name: &str) -> Result<()> {
        self.model.discard_lora_adapter(name)
    }
}

#[derive(Clone)]
enum StepCommand {
    Prefill {
        requests: Vec<PrefillStepItem>,
        kv_views: Vec<KvView>,
        echo: bool,
        sample_seed: u64,
    },
    Decode {
        requests: Vec<DecodeStepItem>,
        kv_views: Vec<KvView>,
        sample_seed: u64,
    },
    Unified {
        prefill_requests: Vec<PrefillStepItem>,
        prefill_kv_views: Vec<KvView>,
        decode_requests: Vec<DecodeStepItem>,
        decode_kv_views: Vec<KvView>,
        sample_seed: u64,
    },
    /// Split-concurrent: prefill and decode launch on separate overlap streams
    /// (SM-partitioned under `green-ctx`, shared under `stream`) for concurrency.
    SplitConcurrent {
        prefill_requests: Vec<PrefillStepItem>,
        prefill_kv_views: Vec<KvView>,
        decode_requests: Vec<DecodeStepItem>,
        decode_kv_views: Vec<KvView>,
        prefill_stream: crate::green_ctx::SendStream,
        decode_stream: crate::green_ctx::SendStream,
        sample_seed: u64,
    },
    /// Speculative verify: one target forward over each request's `K + 1` draft
    /// span (with a speculative KV view), capturing target hidden states for the
    /// next draft round. Each position selects the request's *committed* token —
    /// argmax for greedy rows, a regular sample for non-greedy rows — and the
    /// tokens drive [`accept_prefix_match`] (sampled-verify, #512).
    SpeculativeVerify {
        requests: Vec<VerifyStepItem>,
        kv_views: Vec<KvView>,
        sample_seed: u64,
    },
    /// Speculative draft: roll the DFlash draft model forward one block per
    /// request. Uses the draft's own KV — no target KV views.
    SpeculativeDraft { requests: Vec<DraftStepItem> },
}

impl StepCommand {
    fn kind(&self) -> &'static str {
        match self {
            Self::Prefill { .. } => "prefill",
            Self::Decode { .. } => "decode",
            Self::Unified { .. } => "unified",
            Self::SplitConcurrent { .. } => "split_concurrent",
            Self::SpeculativeVerify { .. } => "speculative_verify",
            Self::SpeculativeDraft { .. } => "speculative_draft",
        }
    }
}

enum WorkerCommand {
    RunStep {
        step: StepCommand,
        collect_result: bool,
        resp: channel::Sender<Result<WorkerStepOutcome>>,
    },
    LoadLoraAdapter {
        name: String,
        adapter: crate::lora::LoraAdapter,
        load_inplace: bool,
        resp: channel::Sender<Result<()>>,
    },
    UnloadLoraAdapter {
        name: String,
        resp: channel::Sender<Result<()>>,
    },
    DiscardLoraAdapter {
        name: String,
        resp: channel::Sender<Result<()>>,
    },
    /// Sync the in-flight prefill from a previous SplitConcurrent step and
    /// return the sampled prefill result.
    ResolvePrefill {
        resp: channel::Sender<Result<PrefillResult>>,
    },
    /// Load the DFlash draft model into the primary lane (built on the worker
    /// thread because it reads the co-located target model).
    LoadDflash {
        draft_path: String,
        resp: channel::Sender<Result<DFlashMeta>>,
    },
    /// Drop a request's DFlash draft state (request retired, or it fell back to
    /// a plain decode that advanced the sequence outside the speculative path).
    DropDflash {
        request_id: RequestId,
        resp: channel::Sender<Result<()>>,
    },
    /// Startup-only (TP + CUDA Graph): one phase of the decode-graph
    /// pre-capture sweep. See [`LocalQwen3Lane::precapture_phase`].
    Precapture {
        phase: PrecapturePhase,
        resp: channel::Sender<Result<()>>,
    },
    DumpDecodeGraph {
        png_path: PathBuf,
        resp: channel::Sender<Result<CudaGraphDumpSummary>>,
    },
    Shutdown,
}

/// One controller-barriered phase of the TP decode-graph pre-capture sweep.
///
/// Capture and launch are separate phases because a captured collective's first
/// launch blocks on its peers: overlapping that with a peer still in capture/
/// instantiate/upload (which contend driver locks and allocate device memory)
/// deadlocks the driver. So every rank finishes capturing a bucket before any
/// rank launches it.
#[derive(Clone, Copy, Debug)]
enum PrecapturePhase {
    /// One eager all-reduce per bucket message size, so the size-selected NCCL
    /// algorithm connects before any `cuStreamBeginCapture` records it.
    Warmup,
    /// Record + instantiate + upload one bucket; no launch, no cross-rank dependency.
    Capture { bucket_idx: usize },
    /// Launch one bucket (pure enqueue after `Capture`) + sync; collectives pair across ranks.
    Launch { bucket_idx: usize },
    /// Verify every graph captured, mark the lane serve-ready.
    Finalize,
}

enum WorkerStepOutcome {
    Ack,
    Prefill(PrefillResult),
    Decode(DecodeResult),
    Unified(UnifiedResult),
    /// Split-concurrent: decode result is ready; prefill is still in-flight
    /// on the prefill stream. The executor must call a follow-up to sync+sample
    /// prefill before using prefill scratch buffers again.
    SplitDecodeReady {
        decode: DecodeResult,
        /// Event recorded on prefill stream after all prefill kernels;
        /// query this to check if prefill is done without blocking.
        prefill_event: cudarc::driver::CudaEvent,
    },
    SpeculativeVerify(VerifyResult),
    SpeculativeDraft(DraftResult),
}

impl WorkerStepOutcome {
    fn kind(&self) -> &'static str {
        match self {
            Self::Ack => "ack",
            Self::Prefill(_) => "prefill",
            Self::Decode(_) => "decode",
            Self::Unified(_) => "unified",
            Self::SplitDecodeReady { .. } => "split_decode_ready",
            Self::SpeculativeVerify(_) => "speculative_verify",
            Self::SpeculativeDraft(_) => "speculative_draft",
        }
    }
}

struct RankWorker {
    tx: channel::Sender<WorkerCommand>,
    handle: Option<thread::JoinHandle<()>>,
}

impl RankWorker {
    fn spawn(rank: usize, mut lane: LocalQwen3Lane) -> Result<Self> {
        let (tx, rx) = channel::unbounded();
        let (startup_tx, startup_rx) = channel::bounded(1);
        let handle = thread::Builder::new()
            .name(format!("qwen3-tp-rank-{rank}"))
            .spawn(move || {
                let startup = lane.bind();
                match startup {
                    Ok(_guard) => {
                        let _ = startup_tx.send(Ok(()));
                        while let Ok(cmd) = rx.recv() {
                            match cmd {
                                WorkerCommand::RunStep {
                                    step,
                                    collect_result,
                                    resp,
                                } => {
                                    let result =
                                        execute_step_on_lane(&mut lane, &step, collect_result);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::LoadLoraAdapter {
                                    name,
                                    adapter,
                                    load_inplace,
                                    resp,
                                } => {
                                    let result =
                                        lane.load_lora_adapter(name, adapter, load_inplace);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::UnloadLoraAdapter { name, resp } => {
                                    let result = lane.unload_lora_adapter(&name);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::DiscardLoraAdapter { name, resp } => {
                                    let result = lane.discard_lora_adapter(&name);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::ResolvePrefill { resp } => {
                                    let result = lane.resolve_inflight_prefill();
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::LoadDflash { draft_path, resp } => {
                                    let result = lane.load_dflash(&draft_path);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::DropDflash { request_id, resp } => {
                                    lane.drop_dflash_request(request_id);
                                    let _ = resp.send(Ok(()));
                                }
                                WorkerCommand::Precapture { phase, resp } => {
                                    let result = lane.precapture_phase(phase);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::DumpDecodeGraph { png_path, resp } => {
                                    let result = lane.dump_decode_graph_png(&png_path);
                                    let _ = resp.send(result);
                                }
                                WorkerCommand::Shutdown => break,
                            }
                        }
                    }
                    Err(err) => {
                        let _ = startup_tx.send(Err(err));
                    }
                }
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn tensor-parallel worker {rank}: {e}"))?;
        let Ok(startup) = startup_rx.recv() else {
            let panic_note = match handle.join() {
                Err(panic) => format!(" (thread panicked: {})", panic_message(panic.as_ref())),
                Ok(()) => String::new(),
            };
            anyhow::bail!("tensor-parallel worker {rank} exited during startup{panic_note}");
        };
        startup?;
        Ok(Self {
            tx,
            handle: Some(handle),
        })
    }

    fn run_step(
        &self,
        step: StepCommand,
        collect_result: bool,
    ) -> Result<channel::Receiver<Result<WorkerStepOutcome>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::RunStep {
                step,
                collect_result,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("tensor-parallel worker step channel closed"))?;
        Ok(resp_rx)
    }

    /// Start one sweep phase; returns the ack receiver. The controller fans a
    /// phase out to all ranks before collecting acks so their collectives pair.
    fn precapture(&self, phase: PrecapturePhase) -> Result<channel::Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::Precapture {
                phase,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("worker channel closed on precapture {phase:?}"))?;
        Ok(resp_rx)
    }

    fn dump_decode_graph_png(&self, png_path: PathBuf) -> Result<CudaGraphDumpSummary> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::DumpDecodeGraph {
                png_path,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("worker channel closed on CUDA Graph dump"))?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("worker dropped CUDA Graph dump response"))?
    }

    /// Ask the worker to sync its in-flight prefill and return the result.
    fn resolve_prefill(&self) -> Result<channel::Receiver<Result<PrefillResult>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::ResolvePrefill { resp: resp_tx })
            .map_err(|_| anyhow::anyhow!("worker channel closed on resolve_prefill"))?;
        Ok(resp_rx)
    }

    /// Load the DFlash draft model into this worker's lane and return its
    /// metadata. Blocks until the worker finishes loading.
    fn load_dflash(&self, draft_path: String) -> Result<DFlashMeta> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::LoadDflash {
                draft_path,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("worker channel closed on load_dflash"))?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("worker dropped load_dflash response"))?
    }

    /// Drop a request's DFlash state. Blocks until the worker acknowledges.
    fn drop_dflash_request(&self, request_id: RequestId) -> Result<()> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::DropDflash {
                request_id,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("worker channel closed on drop_dflash"))?;
        resp_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("worker dropped drop_dflash response"))?
    }

    fn load_lora_adapter(
        &self,
        name: String,
        adapter: crate::lora::LoraAdapter,
        load_inplace: bool,
    ) -> Result<channel::Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::LoadLoraAdapter {
                name,
                adapter,
                load_inplace,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("tensor-parallel worker channel closed on LoRA load"))?;
        Ok(resp_rx)
    }

    fn unload_lora_adapter(&self, name: String) -> Result<channel::Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::UnloadLoraAdapter {
                name,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("tensor-parallel worker channel closed on LoRA unload"))?;
        Ok(resp_rx)
    }

    fn discard_lora_adapter(&self, name: String) -> Result<channel::Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = channel::bounded(1);
        self.tx
            .send(WorkerCommand::DiscardLoraAdapter {
                name,
                resp: resp_tx,
            })
            .map_err(|_| {
                anyhow::anyhow!("tensor-parallel worker channel closed on LoRA discard")
            })?;
        Ok(resp_rx)
    }

    fn shutdown(&mut self) {
        let _ = self.tx.send(WorkerCommand::Shutdown);
        if let Some(handle) = self.handle.take() {
            if let Err(panic) = handle.join() {
                log::warn!(
                    "tensor-parallel worker thread panicked during shutdown: {}",
                    panic_message(panic.as_ref())
                );
            }
        }
    }
}
