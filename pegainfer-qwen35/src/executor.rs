//! Minimal Qwen3.5 logits executor for model-local accuracy gates.
//!
//! The production server uses the scheduler. This executor exists so tests can
//! teacher-force fixed token sequences through prefill + decode and inspect
//! logits without widening the northbound engine API.

use std::collections::HashSet;

use anyhow::Result;
use pegainfer_core::kv_pool::KvState;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::sampler::SamplingParams;

use crate::batch_decode_graph::BatchDecodeGraphState;
use crate::decode_buffers::BatchDecodeBuffers35;
use crate::logprobs::snapshot_requested_logprobs;
use crate::prefill::GdnPrefillRuntimeEvidence;
use crate::recurrent_state::RecurrentState;
use crate::weights::Qwen35Model;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct RequestId(u64);

impl RequestId {
    pub fn new(value: u64) -> Self {
        Self(value)
    }

    pub(crate) fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone)]
pub struct PrefillStepItem {
    pub(crate) request_id: RequestId,
    pub(crate) prompt_tokens: Vec<u32>,
    pub(crate) logprobs: usize,
}

impl PrefillStepItem {
    pub fn new(request_id: RequestId, prompt_tokens: Vec<u32>, logprobs: usize) -> Self {
        Self {
            request_id,
            prompt_tokens,
            logprobs,
        }
    }
}

#[derive(Clone)]
pub struct DecodeStepItem {
    pub(crate) request_id: RequestId,
    pub(crate) token_id: u32,
    pub(crate) logprobs: usize,
}

impl DecodeStepItem {
    pub fn new(request_id: RequestId, token_id: u32, logprobs: usize) -> Self {
        Self {
            request_id,
            token_id,
            logprobs,
        }
    }
}

#[derive(Clone, Copy)]
pub struct PrefillPlan<'a> {
    pub requests: &'a [PrefillStepItem],
}

#[derive(Clone, Copy)]
pub struct DecodePlan<'a> {
    pub requests: &'a [DecodeStepItem],
}

#[derive(Clone, Debug)]
pub struct PrefillRequestResult {
    pub(crate) request_id: RequestId,
    pub(crate) first_token: u32,
    pub first_token_logprob: Option<TokenLogprob>,
}

#[derive(Clone, Debug)]
pub struct DecodeRequestResult {
    pub(crate) request_id: RequestId,
    pub(crate) token: u32,
    pub logprob: Option<TokenLogprob>,
}

#[derive(Debug)]
pub struct PrefillResult {
    pub requests: Vec<PrefillRequestResult>,
}

#[derive(Debug)]
pub struct DecodeResult {
    pub requests: Vec<DecodeRequestResult>,
}

struct ActiveRequest {
    request_id: RequestId,
    kv: KvState,
    graph_slot_idx: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExecutorGdnPrefillBackend {
    Auto,
    Triton,
    FlashInfer,
}

pub struct Qwen35Executor {
    model: Qwen35Model,
    graph_state: BatchDecodeGraphState,
    active: Vec<ActiveRequest>,
    gdn_prefill_backend: ExecutorGdnPrefillBackend,
}

impl Qwen35Executor {
    pub fn from_runtime(model_path: &str, device_ordinal: usize, max_batch: usize) -> Result<Self> {
        let model = Qwen35Model::from_safetensors(model_path, device_ordinal, max_batch)?;
        Self::from_model(model, ExecutorGdnPrefillBackend::Auto)
    }

    pub fn from_runtime_with_triton_gdn(
        model_path: &str,
        device_ordinal: usize,
        max_batch: usize,
    ) -> Result<Self> {
        let model = Qwen35Model::from_safetensors(model_path, device_ordinal, max_batch)?;
        Self::from_model(model, ExecutorGdnPrefillBackend::Triton)
    }

    /// Build the low-level accuracy executor with the pinned FlashInfer GDN
    /// candidate selected explicitly for every prefill chunk. This is a
    /// test/benchmark seam over the same production dispatch boundary.
    pub fn from_runtime_with_flashinfer_gdn(
        model_path: &str,
        device_ordinal: usize,
        max_batch: usize,
    ) -> Result<Self> {
        let model = Qwen35Model::from_safetensors(model_path, device_ordinal, max_batch)?;
        model.require_flashinfer_gdn_for_test()?;
        Self::from_model(model, ExecutorGdnPrefillBackend::FlashInfer)
    }

    fn from_model(
        model: Qwen35Model,
        gdn_prefill_backend: ExecutorGdnPrefillBackend,
    ) -> Result<Self> {
        model.tune_decode_gemm_algos()?;
        let graph_state = model.create_batch_decode_graph_state()?;
        Ok(Self {
            model,
            graph_state,
            active: Vec::new(),
            gdn_prefill_backend,
        })
    }

    /// Return candidate identity and launch proof for the explicit FlashInfer
    /// constructor. A standard Triton executor has no such evidence.
    pub fn flashinfer_gdn_runtime_evidence(&self) -> Result<Option<GdnPrefillRuntimeEvidence>> {
        match self.gdn_prefill_backend {
            ExecutorGdnPrefillBackend::Auto => self
                .model
                .flashinfer_gdn_runtime_evidence()
                .map(Some)
                .or_else(|_| Ok(None)),
            ExecutorGdnPrefillBackend::Triton => Ok(None),
            ExecutorGdnPrefillBackend::FlashInfer => {
                self.model.flashinfer_gdn_runtime_evidence().map(Some)
            }
        }
    }

    pub fn execute_prefill(&mut self, plan: PrefillPlan<'_>) -> Result<PrefillResult> {
        anyhow::ensure!(
            !plan.requests.is_empty(),
            "Qwen3.5 prefill plan requires at least one request"
        );
        anyhow::ensure!(
            self.active.len() + plan.requests.len() <= self.graph_state.slot_states.len(),
            "Qwen3.5 prefill would exceed logits executor capacity"
        );
        let mut seen = HashSet::with_capacity(plan.requests.len());
        for req in plan.requests {
            anyhow::ensure!(
                !req.prompt_tokens.is_empty(),
                "Qwen3.5 logits executor prefill request {} has an empty prompt",
                req.request_id.get()
            );
            anyhow::ensure!(
                seen.insert(req.request_id),
                "duplicate Qwen3.5 request id {} in prefill plan",
                req.request_id.get()
            );
            anyhow::ensure!(
                !self
                    .active
                    .iter()
                    .any(|active| active.request_id == req.request_id),
                "duplicate Qwen3.5 request id {}",
                req.request_id.get()
            );
        }

        let prompts: Vec<&[u32]> = plan
            .requests
            .iter()
            .map(|req| req.prompt_tokens.as_slice())
            .collect();
        let mut kv_states: Vec<KvState> = plan
            .requests
            .iter()
            .map(|_| self.model.alloc_kv())
            .collect();
        let mut recurrent_states: Vec<RecurrentState> = plan
            .requests
            .iter()
            .map(|_| RecurrentState::new(self.model.device_ctx(), self.model.config()))
            .collect::<Result<_>>()?;
        let mut recurrent_refs: Vec<&mut RecurrentState> = recurrent_states.iter_mut().collect();
        let logits = match self.gdn_prefill_backend {
            ExecutorGdnPrefillBackend::Auto => {
                self.model
                    .batch_prefill_logits(&prompts, &mut kv_states, &mut recurrent_refs)?
            }
            ExecutorGdnPrefillBackend::Triton => self.model.batch_prefill_logits_triton(
                &prompts,
                &mut kv_states,
                &mut recurrent_refs,
            )?,
            ExecutorGdnPrefillBackend::FlashInfer => self.model.batch_prefill_logits_flashinfer(
                &prompts,
                &mut kv_states,
                &mut recurrent_refs,
            )?,
        };

        let requested_logprobs: Vec<usize> = plan.requests.iter().map(|req| req.logprobs).collect();
        let cpu_logits =
            snapshot_requested_logprobs(self.model.device_ctx(), &logits, &requested_logprobs)?;
        let tokens =
            select_default_tokens_from_logits(&self.model, &logits, &mut self.graph_state.buffers)?;

        let mut results = Vec::with_capacity(plan.requests.len());
        for (i, (req, kv)) in plan.requests.iter().zip(kv_states).enumerate() {
            let first_token = tokens[i];
            let first_token_logprob = cpu_logits[i].as_ref().and_then(|row| {
                pegainfer_sample::token_logprob_from_row(row, first_token, req.logprobs)
            });
            let slot_idx = self.active.len();
            self.graph_state.copy_state_to_slot(
                self.model.device_ctx(),
                &recurrent_states[i],
                slot_idx,
            )?;
            self.active.push(ActiveRequest {
                request_id: req.request_id,
                kv,
                graph_slot_idx: slot_idx,
            });
            results.push(PrefillRequestResult {
                request_id: req.request_id,
                first_token,
                first_token_logprob,
            });
        }

        Ok(PrefillResult { requests: results })
    }

    pub fn execute_decode(&mut self, plan: DecodePlan<'_>) -> Result<DecodeResult> {
        anyhow::ensure!(
            !plan.requests.is_empty(),
            "Qwen3.5 decode plan requires at least one request"
        );
        anyhow::ensure!(
            plan.requests.len() == self.active.len(),
            "Qwen3.5 logits executor decode must include all active requests in slot order"
        );
        for (i, req) in plan.requests.iter().enumerate() {
            anyhow::ensure!(
                self.active[i].request_id == req.request_id,
                "Qwen3.5 decode request order differs from active slot order"
            );
        }

        let token_ids: Vec<u32> = plan.requests.iter().map(|req| req.token_id).collect();
        let mut kv_refs: Vec<&mut KvState> =
            self.active.iter_mut().map(|req| &mut req.kv).collect();
        self.model
            .batch_decode_graph(&token_ids, &mut kv_refs, &mut self.graph_state)?;

        let requested_logprobs: Vec<usize> = plan.requests.iter().map(|req| req.logprobs).collect();
        let cpu_logits = snapshot_requested_logprobs(
            self.model.device_ctx(),
            &self.graph_state.buffers.logits,
            &requested_logprobs,
        )?;
        let params = vec![SamplingParams::default(); plan.requests.len()];
        let params_refs: Vec<&SamplingParams> = params.iter().collect();
        let tokens = self.model.select_tokens_batch_varied(
            &mut self.graph_state.buffers,
            &params_refs,
            0,
        )?;

        let mut results = Vec::with_capacity(plan.requests.len());
        for (i, req) in plan.requests.iter().enumerate() {
            let token = tokens[i];
            let logprob = cpu_logits[i]
                .as_ref()
                .and_then(|row| pegainfer_sample::token_logprob_from_row(row, token, req.logprobs));
            results.push(DecodeRequestResult {
                request_id: req.request_id,
                token,
                logprob,
            });
        }
        Ok(DecodeResult { requests: results })
    }

    pub fn drop_request(&mut self, request_id: RequestId) -> Result<()> {
        let Some(idx) = self
            .active
            .iter()
            .position(|active| active.request_id == request_id)
        else {
            return Ok(());
        };
        self.compact_slot(idx)
    }

    fn compact_slot(&mut self, idx: usize) -> Result<()> {
        let last = self.active.len() - 1;
        self.active.swap_remove(idx);

        if idx < self.active.len() {
            anyhow::ensure!(
                self.active[idx].graph_slot_idx == last,
                "Qwen3.5 logits executor slot invariant broken: active slot {} moved from graph slot {}, expected {}",
                idx,
                self.active[idx].graph_slot_idx,
                last
            );
            for layer_idx in 0..self.graph_state.slot_states[last].layers.len() {
                let (src_part, dst_part) = if idx < last {
                    let (left, right) = self.graph_state.slot_states.split_at_mut(last);
                    (
                        &right[0].layers[layer_idx],
                        &mut left[idx].layers[layer_idx],
                    )
                } else {
                    unreachable!("idx < active.len() <= last");
                };
                self.model
                    .device_ctx()
                    .stream
                    .memcpy_dtod(&src_part.state, &mut dst_part.state)
                    .map_err(|e| {
                        anyhow::anyhow!("compact Qwen3.5 logits executor state copy failed: {e}")
                    })?;
                self.model
                    .device_ctx()
                    .stream
                    .memcpy_dtod(&src_part.conv_state.data, &mut dst_part.conv_state.data)
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "compact Qwen3.5 logits executor conv_state copy failed: {e}"
                        )
                    })?;
            }
            self.graph_state.slot_states[idx].seq_len = self.graph_state.slot_states[last].seq_len;
            self.active[idx].graph_slot_idx = idx;
        }
        Ok(())
    }
}

fn select_default_tokens_from_logits(
    model: &Qwen35Model,
    logits: &HiddenStates,
    bufs: &mut BatchDecodeBuffers35,
) -> Result<Vec<u32>> {
    let params = vec![SamplingParams::default(); logits.seq_len];
    let params_refs: Vec<&SamplingParams> = params.iter().collect();
    model.select_tokens_from_logits_varied(logits, bufs, &params_refs, 0)
}
