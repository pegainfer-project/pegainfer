//! Single-GPU and TP execution backends south of the ledger.
//!
//! Named [`Qwen35Backend`] so it does not collide with the frontend's
//! `engine::SchedulerBackend` (the driver-side wiring).

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

use anyhow::Result;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaStream;
use cudarc::driver::sys;
use pegainfer_core::kv_pool::KvState;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::RequestId as EngineRequestId;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::sampler::SamplingParams;

use super::plan;
use super::plan::slot_for_new_request;
use crate::Qwen35DecodeOverlap;
use crate::batch_decode_graph::BatchDecodeGraphState;
use crate::executor::DecodeRequestResult;
use crate::executor::DecodeResult;
use crate::executor::PrefillRequestResult;
use crate::executor::PrefillResult;
use crate::executor::RequestId as BackendRequestId;
use crate::logprobs::snapshot_requested_logprobs;
use crate::recurrent_state::RecurrentState;
use crate::tp_executor::DropExpectation;
use crate::tp_executor::Qwen35TpExecutor;
use crate::tp_executor::TpDecodeStepItem;
use crate::tp_executor::TpPrefillChunkItem;
use crate::tp_executor::TpUnifiedPlan;
use crate::weights::Qwen35Model;

/// An in-flight request being decoded. Recurrent state lives in the
/// `BatchDecodeGraphState` at `graph_slot_idx` — NOT owned here.
pub(super) struct ActiveRequest35 {
    pub(super) id: EngineRequestId,
    pub(super) backend_state: ActiveBackendState,
    pub(super) last_token: u32,
    pub(super) generated_count: usize,
    pub(super) max_tokens: usize,
    pub(super) prompt_len: usize,
    pub(super) params: SamplingParams,
    /// Number of top logprobs to return (0 = disabled).
    pub(super) logprobs: usize,
}

/// A request whose prompt is being prefilled across multiple scheduler steps.
pub(super) struct PrefillingRequest35 {
    pub(super) id: EngineRequestId,
    pub(super) request: Request,
    pub(super) backend_state: PrefillBackendState,
    /// Prompt tokens prefilled so far.
    pub(super) cursor: usize,
    /// Tokens to prefill in the step currently scheduled (set by `take_prefill_chunks`).
    pub(super) step_chunk: usize,
}

pub(super) enum ActiveBackendState {
    Single {
        kv: KvState,
        /// Index into `BatchDecodeGraphState.slot_states`.
        graph_slot_idx: usize,
    },
    Tp {
        request_id: BackendRequestId,
    },
}

pub(super) enum PrefillBackendState {
    Single { kv: KvState, rec: RecurrentState },
    Tp { request_id: BackendRequestId },
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct PrefillArtifact {
    pub(super) token: u32,
    pub(super) logprob: Option<TokenLogprob>,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct DecodeArtifact {
    pub(super) token: u32,
    pub(super) logprob: Option<TokenLogprob>,
}

pub(super) struct AlignedUnifiedArtifacts {
    pub(super) prefill: Vec<Option<PrefillArtifact>>,
    pub(super) decode: Vec<DecodeArtifact>,
}

pub(super) enum PrefillStepArtifacts {
    Single {
        tokens: Vec<u32>,
        logprobs: Vec<Option<TokenLogprob>>,
    },
    Tp(Vec<Option<PrefillArtifact>>),
}

impl PrefillStepArtifacts {
    pub(super) fn final_artifact(&self, idx: usize) -> PrefillArtifact {
        match self {
            Self::Single { tokens, logprobs } => PrefillArtifact {
                token: tokens[idx],
                logprob: logprobs[idx].clone(),
            },
            Self::Tp(artifacts) => artifacts[idx]
                .clone()
                .expect("validated TP final-prefill row must contain an artifact"),
        }
    }
}

pub(super) struct SingleGpuBackend {
    model: Qwen35Model,
    graph_state: BatchDecodeGraphState,
    prefill_stream: Option<Arc<CudaStream>>,
}

// One instance per scheduler; the size asymmetry costs nothing here.
#[allow(clippy::large_enum_variant)]
pub(super) enum Qwen35Backend {
    Single(SingleGpuBackend),
    Tp(TpSchedulerBackend),
}

pub(super) struct AsyncPrefillOutput {
    logits: Option<HiddenStates>,
    done: CudaEvent,
    stream: Arc<CudaStream>,
    completed: bool,
}

impl AsyncPrefillOutput {
    pub(super) fn is_ready(&mut self) -> bool {
        match unsafe { sys::cuEventQuery(self.done.cu_event()) } {
            sys::CUresult::CUDA_SUCCESS => {
                self.completed = true;
                true
            }
            sys::CUresult::CUDA_ERROR_NOT_READY => false,
            err => fatal_cuda_lifecycle(&format!(
                "query Qwen3.5 async prefill event failed: {err:?}"
            )),
        }
    }

    pub(super) fn into_logits(mut self) -> HiddenStates {
        if !self.completed {
            if let Err(err) = self.done.synchronize() {
                fatal_cuda_lifecycle(&format!("wait for Qwen3.5 async prefill failed: {err}"));
            }
            self.completed = true;
        }
        self.logits
            .take()
            .expect("async prefill logits must be consumed exactly once")
    }
}

impl Drop for AsyncPrefillOutput {
    fn drop(&mut self) {
        if self.completed {
            return;
        }
        if let Err(err) = self.stream.synchronize() {
            fatal_cuda_lifecycle(&format!(
                "drain Qwen3.5 async prefill during cleanup failed: {err}"
            ));
        }
    }
}

fn fatal_cuda_lifecycle(message: &str) -> ! {
    log::error!("FATAL: {message}; aborting before CUDA-referenced state is released");
    std::process::abort();
}

pub(super) struct TpSchedulerBackend {
    executor: Qwen35TpExecutor,
    next_request_id: u64,
}

impl SingleGpuBackend {
    pub(super) fn new(
        model: Qwen35Model,
        max_batch: usize,
        decode_overlap: Qwen35DecodeOverlap,
    ) -> Result<Self> {
        anyhow::ensure!(max_batch > 0, "Qwen3.5 max_batch must be > 0");
        let graph_capacity = crate::batch_decode_graph::bucket_for(max_batch);
        let graph_state = model.create_batch_decode_graph_state_with_capacity(graph_capacity)?;
        let prefill_stream = match decode_overlap {
            Qwen35DecodeOverlap::Off => None,
            Qwen35DecodeOverlap::SharedSm => Some(
                model
                    .device_ctx()
                    .ctx
                    .new_stream()
                    .map_err(|err| anyhow::anyhow!("create Qwen3.5 prefill stream: {err}"))?,
            ),
        };
        Ok(Self {
            model,
            graph_state,
            prefill_stream,
        })
    }

    pub(super) fn model(&self) -> &Qwen35Model {
        &self.model
    }

    fn max_batch(&self) -> usize {
        // #470: admit the requested `--max-batch`, which may sit below the loaded
        // graph bucket (e.g. 5 on bucket 8); never exceed the physical slots.
        self.model
            .decode_admission_batch
            .min(self.graph_state.slot_states.len())
            .max(1)
    }

    fn page_size(&self) -> usize {
        self.model.kv_pool().layout().page_size
    }

    fn available_pages(&self) -> usize {
        self.model.kv_pool().available_pages()
    }

    fn capacity_pages_for_requests(&self) -> usize {
        self.model.kv_pool().capacity_pages().saturating_sub(1)
    }

    fn max_position_embeddings(&self) -> usize {
        self.model.config().max_position_embeddings
    }

    fn alloc_kv(&self) -> KvState {
        self.model.alloc_kv()
    }

    fn alloc_recurrent(&self) -> Result<RecurrentState> {
        RecurrentState::new(self.model.device_ctx(), self.model.config())
    }

    pub(super) fn batch_prefill_logits(&self, chunk: &mut ScheduledChunk) -> Result<HiddenStates> {
        let window_refs: Vec<&[u32]> = chunk.windows.iter().map(Vec::as_slice).collect();
        let ScheduledChunkBackendState::Single { kvs, recs } = &mut chunk.backend_state else {
            anyhow::bail!("single-GPU prefill received TP chunk state");
        };
        let mut rec_refs: Vec<&mut RecurrentState> = recs.iter_mut().collect();
        self.model
            .batch_prefill_logits(&window_refs, kvs, &mut rec_refs)
    }

    pub(super) fn overlap_enabled(&self) -> bool {
        self.prefill_stream.is_some()
    }

    pub(super) fn launch_async_prefill(
        &mut self,
        chunk: &mut ScheduledChunk,
    ) -> Result<AsyncPrefillOutput> {
        let prefill_stream = self
            .prefill_stream
            .clone()
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 decode overlap is disabled"))?;

        // Request KV/recurrent state was allocated on the model stream. Order
        // those producers before the prefill stream without blocking the host.
        prefill_stream
            .join(&self.model.device_ctx().stream)
            .map_err(|err| anyhow::anyhow!("join Qwen3.5 prefill stream: {err}"))?;

        let window_refs: Vec<&[u32]> = chunk.windows.iter().map(Vec::as_slice).collect();
        let ScheduledChunkBackendState::Single { kvs, recs } = &mut chunk.backend_state else {
            anyhow::bail!("single-GPU async prefill received TP chunk state");
        };
        let mut rec_refs: Vec<&mut RecurrentState> = recs.iter_mut().collect();
        let logits = match self.model.batch_prefill_logits_on_stream(
            Arc::clone(&prefill_stream),
            &window_refs,
            kvs,
            &mut rec_refs,
        ) {
            Ok(logits) => logits,
            Err(err) => {
                if let Err(sync_err) = prefill_stream.synchronize() {
                    fatal_cuda_lifecycle(&format!(
                        "Qwen3.5 async prefill failed ({err}); stream drain failed: {sync_err}"
                    ));
                }
                return Err(err);
            }
        };
        let done = match prefill_stream.record_event(None) {
            Ok(done) => done,
            Err(err) => {
                if let Err(sync_err) = prefill_stream.synchronize() {
                    fatal_cuda_lifecycle(&format!(
                        "record Qwen3.5 async prefill event failed ({err}); stream drain failed: {sync_err}"
                    ));
                }
                return Err(anyhow::anyhow!("record Qwen3.5 async prefill event: {err}"));
            }
        };
        Ok(AsyncPrefillOutput {
            logits: Some(logits),
            done,
            stream: prefill_stream,
            completed: false,
        })
    }

    pub(super) fn unified_step(
        &mut self,
        chunk: &mut ScheduledChunk,
        active: &mut [ActiveRequest35],
    ) -> Result<crate::unified_forward::UnifiedStepOutput> {
        let window_refs: Vec<&[u32]> = chunk.windows.iter().map(Vec::as_slice).collect();
        let ScheduledChunkBackendState::Single { kvs, recs } = &mut chunk.backend_state else {
            anyhow::bail!("single-GPU unified step received TP chunk state");
        };
        let mut rec_refs: Vec<&mut RecurrentState> = recs.iter_mut().collect();
        let decode_tokens: Vec<u32> = active.iter().map(|r| r.last_token).collect();
        let mut decode_kv_refs: Vec<&mut KvState> = active
            .iter_mut()
            .map(|r| match &mut r.backend_state {
                ActiveBackendState::Single { kv, .. } => kv,
                ActiveBackendState::Tp { .. } => {
                    panic!("single-GPU unified step received TP active state")
                }
            })
            .collect();
        self.model.unified_step(
            &window_refs,
            kvs,
            &mut rec_refs,
            &decode_tokens,
            &mut decode_kv_refs,
            &mut self.graph_state,
        )
    }

    pub(super) fn decode_graph(&mut self, active: &mut [ActiveRequest35]) -> Result<()> {
        let token_ids: Vec<u32> = active.iter().map(|r| r.last_token).collect();
        let mut kv_refs: Vec<&mut KvState> = active
            .iter_mut()
            .map(|r| match &mut r.backend_state {
                ActiveBackendState::Single { kv, .. } => kv,
                ActiveBackendState::Tp { .. } => {
                    panic!("single-GPU decode received TP active state")
                }
            })
            .collect();
        self.model
            .batch_decode_graph(&token_ids, &mut kv_refs, &mut self.graph_state)
    }

    pub(super) fn sample_prefill_logits(
        &mut self,
        pending: &[Request],
        logits: &HiddenStates,
        sample_seed: u64,
    ) -> Result<(Vec<u32>, Vec<Option<TokenLogprob>>)> {
        debug_assert_eq!(
            logits.seq_len,
            pending.len(),
            "Qwen3.5 prefill logits rows must preserve pending request order"
        );
        let requested_logprobs: Vec<usize> = pending.iter().map(|r| r.logprobs).collect();
        let cpu_logits =
            snapshot_requested_logprobs(self.model.device_ctx(), logits, &requested_logprobs)?;
        let params_refs: Vec<&SamplingParams> = pending.iter().map(|r| &r.params).collect();
        let tokens = self.model.select_tokens_from_logits_varied(
            logits,
            &mut self.graph_state.buffers,
            &params_refs,
            sample_seed,
        )?;

        let logprobs = cpu_logits
            .into_iter()
            .enumerate()
            .map(|(i, logits_opt)| {
                logits_opt.and_then(|logits_f32| {
                    pegainfer_sample::token_logprob_from_row(
                        &logits_f32,
                        tokens[i],
                        pending[i].logprobs,
                    )
                })
            })
            .collect();
        Ok((tokens, logprobs))
    }

    pub(super) fn sample_decode_logits(
        &mut self,
        active: &[ActiveRequest35],
        sample_seed: u64,
    ) -> Result<(Vec<u32>, Vec<Option<TokenLogprob>>)> {
        let requested_logprobs: Vec<usize> = active.iter().map(|r| r.logprobs).collect();
        let cpu_logits = snapshot_requested_logprobs(
            self.model.device_ctx(),
            &self.graph_state.buffers.logits,
            &requested_logprobs,
        )?;
        let params_refs: Vec<&SamplingParams> = active.iter().map(|r| &r.params).collect();
        let tokens = self.model.select_tokens_batch_varied(
            &mut self.graph_state.buffers,
            &params_refs,
            sample_seed,
        )?;

        let logprobs = cpu_logits
            .into_iter()
            .enumerate()
            .map(|(i, logits_opt)| {
                logits_opt.and_then(|logits_f32| {
                    pegainfer_sample::token_logprob_from_row(
                        &logits_f32,
                        tokens[i],
                        active[i].logprobs,
                    )
                })
            })
            .collect();
        Ok((tokens, logprobs))
    }

    fn is_stop_token(&self, token: u32) -> bool {
        self.model.is_stop_token(token)
    }

    fn copy_recurrent_to_slot(
        &mut self,
        recurrent: &RecurrentState,
        slot_idx: usize,
    ) -> Result<()> {
        self.graph_state
            .copy_state_to_slot(self.model.device_ctx(), recurrent, slot_idx)
    }

    fn compact_slot(&mut self, active: &mut [ActiveRequest35], compaction: plan::SlotCompaction) {
        let src_slot = match active[compaction.moved_to].backend_state {
            ActiveBackendState::Single { graph_slot_idx, .. } => graph_slot_idx,
            ActiveBackendState::Tp { .. } => {
                panic!("single-GPU slot compaction received TP active state")
            }
        };
        debug_assert_eq!(src_slot, compaction.moved_from);

        let ctx = self.model.device_ctx();
        let src = &self.graph_state.slot_states[compaction.moved_from];
        for layer_idx in 0..src.layers.len() {
            let (src_part, dst_part) = if compaction.moved_to < compaction.moved_from {
                let (left, right) = self
                    .graph_state
                    .slot_states
                    .split_at_mut(compaction.moved_from);
                (
                    &right[0].layers[layer_idx],
                    &mut left[compaction.moved_to].layers[layer_idx],
                )
            } else {
                unreachable!("idx < active.len() <= last");
            };

            ctx.stream
                .memcpy_dtod(&src_part.state, &mut dst_part.state)
                .expect("compact slot state copy failed");
            ctx.stream
                .memcpy_dtod(&src_part.conv_state.data, &mut dst_part.conv_state.data)
                .expect("compact slot conv_state copy failed");
        }
        self.graph_state.slot_states[compaction.moved_to].seq_len =
            self.graph_state.slot_states[compaction.moved_from].seq_len;

        match &mut active[compaction.moved_to].backend_state {
            ActiveBackendState::Single { graph_slot_idx, .. } => {
                *graph_slot_idx = compaction.moved_to;
            }
            ActiveBackendState::Tp { .. } => {
                panic!("single-GPU slot compaction received TP active state")
            }
        }
    }
}

impl TpSchedulerBackend {
    pub(super) fn new(
        model_path: &str,
        device_ordinals: &[usize],
        max_batch: usize,
        max_prefill_tokens: usize,
    ) -> Result<Self> {
        let executor = Qwen35TpExecutor::from_runtime_with_limits(
            model_path,
            false,
            device_ordinals,
            max_batch,
            max_prefill_tokens,
        )?;
        Ok(Self {
            executor,
            next_request_id: 1,
        })
    }

    fn alloc_request_id(&mut self) -> BackendRequestId {
        let id = BackendRequestId::new(self.next_request_id);
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        id
    }

    fn max_batch(&self) -> usize {
        self.executor.max_batch()
    }

    pub(super) fn page_size(&self) -> usize {
        self.executor.page_size()
    }

    pub(super) fn capacity_pages_for_requests(&self) -> usize {
        self.executor.capacity_pages_for_requests()
    }

    pub(super) fn max_position_embeddings(&self) -> usize {
        self.executor.max_position_embeddings()
    }

    fn is_stop_token(&self, token: u32) -> bool {
        self.executor.is_stop_token(token)
    }

    fn available_pages(
        &self,
        active: &[ActiveRequest35],
        prefilling: &[PrefillingRequest35],
    ) -> usize {
        let page_size = self.page_size();
        let active_pages: usize = active
            .iter()
            .map(|req| pages_needed(current_active_tokens(req), page_size))
            .sum();
        let prefilling_pages: usize = prefilling
            .iter()
            .map(|req| pages_needed(req.cursor, page_size))
            .sum();
        self.capacity_pages_for_requests()
            .saturating_sub(active_pages.saturating_add(prefilling_pages))
    }

    pub(super) fn execute_prefill_chunk(
        &self,
        chunk: &ScheduledChunk,
        sample_seed: u64,
    ) -> Result<Vec<Option<PrefillArtifact>>> {
        let items = tp_prefill_items(chunk)?;
        let result = self
            .executor
            .execute_prefill_chunks_with_seed(&items, sample_seed)?;
        align_prefill_results(chunk, &result)
            .map_err(|err| self.executor.poison_artifact_contract("prefill", &err))
    }

    pub(super) fn execute_decode(
        &self,
        active: &[ActiveRequest35],
        sample_seed: u64,
    ) -> Result<Vec<DecodeArtifact>> {
        let items = tp_decode_items(active)?;
        let result = self.executor.execute_decode_items(&items, sample_seed)?;
        align_decode_results(active, &result)
            .map_err(|err| self.executor.poison_artifact_contract("decode", &err))
    }

    pub(super) fn execute_unified(
        &self,
        chunk: &ScheduledChunk,
        active: &[ActiveRequest35],
        decode_sample_seed: u64,
        prefill_sample_seed: u64,
    ) -> Result<AlignedUnifiedArtifacts> {
        let plan = TpUnifiedPlan {
            prefill: tp_prefill_items(chunk)?,
            decode: tp_decode_items(active)?,
            prefill_sample_seed,
            decode_sample_seed,
        };
        let result = self.executor.execute_unified(&plan)?;
        let prefill = align_prefill_results(chunk, &result.prefill).map_err(|err| {
            self.executor
                .poison_artifact_contract("unified prefill", &err)
        })?;
        let decode = align_decode_results(active, &result.decode).map_err(|err| {
            self.executor
                .poison_artifact_contract("unified decode", &err)
        })?;
        Ok(AlignedUnifiedArtifacts { prefill, decode })
    }

    fn drop_request(
        &self,
        request_id: BackendRequestId,
        expectation: DropExpectation,
    ) -> Result<()> {
        self.executor.drop_request(request_id, expectation)
    }
}

impl Qwen35Backend {
    pub(super) fn max_batch(&self) -> usize {
        match self {
            Self::Single(backend) => backend.max_batch(),
            Self::Tp(backend) => backend.max_batch(),
        }
    }

    pub(super) fn page_size(&self) -> usize {
        match self {
            Self::Single(backend) => backend.page_size(),
            Self::Tp(backend) => backend.page_size(),
        }
    }

    pub(super) fn available_pages(
        &self,
        active: &[ActiveRequest35],
        prefilling: &[PrefillingRequest35],
    ) -> usize {
        match self {
            Self::Single(backend) => backend.available_pages(),
            Self::Tp(backend) => backend.available_pages(active, prefilling),
        }
    }

    pub(super) fn capacity_pages_for_requests(&self) -> usize {
        match self {
            Self::Single(backend) => backend.capacity_pages_for_requests(),
            Self::Tp(backend) => backend.capacity_pages_for_requests(),
        }
    }

    pub(super) fn max_position_embeddings(&self) -> usize {
        match self {
            Self::Single(backend) => backend.max_position_embeddings(),
            Self::Tp(backend) => backend.max_position_embeddings(),
        }
    }

    pub(super) fn alloc_prefill_state(&mut self) -> Result<PrefillBackendState> {
        match self {
            Self::Single(backend) => Ok(PrefillBackendState::Single {
                kv: backend.alloc_kv(),
                rec: backend.alloc_recurrent()?,
            }),
            Self::Tp(backend) => Ok(PrefillBackendState::Tp {
                request_id: backend.alloc_request_id(),
            }),
        }
    }

    pub(super) fn is_stop_token(&self, token: u32) -> bool {
        match self {
            Self::Single(backend) => backend.is_stop_token(token),
            Self::Tp(backend) => backend.is_stop_token(token),
        }
    }
}

fn current_active_tokens(req: &ActiveRequest35) -> usize {
    req.prompt_len
        .saturating_add(req.generated_count.saturating_sub(1))
}

fn pages_needed(token_count: usize, page_size: usize) -> usize {
    token_count.div_ceil(page_size)
}

fn tp_prefill_items(chunk: &ScheduledChunk) -> Result<Vec<TpPrefillChunkItem>> {
    let ScheduledChunkBackendState::Tp { request_ids } = &chunk.backend_state else {
        anyhow::bail!("TP prefill received single-GPU chunk state");
    };
    anyhow::ensure!(
        chunk.reqs.len() == request_ids.len()
            && chunk.reqs.len() == chunk.windows.len()
            && chunk.reqs.len() == chunk.ends.len(),
        "Qwen3.5 TP scheduled prefill vectors are misaligned"
    );
    Ok(chunk
        .reqs
        .iter()
        .zip(request_ids)
        .zip(&chunk.windows)
        .zip(&chunk.ends)
        .map(|(((req, request_id), window), end)| {
            TpPrefillChunkItem::new_with_sampling(
                *request_id,
                window.clone(),
                req.logprobs,
                req.params,
                *end == req.prompt_tokens.len(),
            )
        })
        .collect())
}

fn tp_decode_items(active: &[ActiveRequest35]) -> Result<Vec<TpDecodeStepItem>> {
    active
        .iter()
        .map(|req| {
            let ActiveBackendState::Tp { request_id } = &req.backend_state else {
                anyhow::bail!("TP decode received single-GPU active state");
            };
            Ok(TpDecodeStepItem::new(
                *request_id,
                req.last_token,
                req.logprobs,
                req.params,
            ))
        })
        .collect()
}

fn align_prefill_results(
    chunk: &ScheduledChunk,
    result: &PrefillResult,
) -> Result<Vec<Option<PrefillArtifact>>> {
    let ScheduledChunkBackendState::Tp { request_ids } = &chunk.backend_state else {
        anyhow::bail!("align_prefill_results requires TP chunk state");
    };
    anyhow::ensure!(
        request_ids.len() == chunk.reqs.len() && chunk.ends.len() == chunk.reqs.len(),
        "Qwen3.5 TP prefill alignment vectors are misaligned"
    );
    let expected: HashSet<BackendRequestId> = request_ids
        .iter()
        .zip(&chunk.reqs)
        .zip(&chunk.ends)
        .filter_map(|((&request_id, req), &end)| {
            (end == req.prompt_tokens.len()).then_some(request_id)
        })
        .collect();
    let mut by_id = HashMap::with_capacity(result.requests.len());
    for PrefillRequestResult {
        request_id,
        first_token,
        first_token_logprob,
    } in &result.requests
    {
        anyhow::ensure!(
            expected.contains(request_id),
            "Qwen3.5 TP prefill returned unknown or non-final request id {}",
            request_id.get()
        );
        let artifact = PrefillArtifact {
            token: *first_token,
            logprob: first_token_logprob.clone(),
        };
        anyhow::ensure!(
            by_id.insert(*request_id, artifact).is_none(),
            "Qwen3.5 TP prefill returned duplicate request id {}",
            request_id.get()
        );
    }
    anyhow::ensure!(
        by_id.len() == expected.len(),
        "Qwen3.5 TP prefill result is missing final request IDs"
    );

    request_ids
        .iter()
        .zip(&chunk.reqs)
        .zip(&chunk.ends)
        .map(|((&request_id, req), &end)| {
            if end == req.prompt_tokens.len() {
                by_id.remove(&request_id).map(Some).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Qwen3.5 TP prefill result is missing final request id {}",
                        request_id.get()
                    )
                })
            } else {
                Ok(None)
            }
        })
        .collect()
}

fn align_decode_results(
    active: &[ActiveRequest35],
    result: &DecodeResult,
) -> Result<Vec<DecodeArtifact>> {
    let expected: Vec<BackendRequestId> = active
        .iter()
        .map(|active_req| {
            let ActiveBackendState::Tp { request_id } = active_req.backend_state else {
                anyhow::bail!("align_decode_results requires TP active state");
            };
            Ok(request_id)
        })
        .collect::<Result<_>>()?;
    let expected_set: HashSet<_> = expected.iter().copied().collect();
    anyhow::ensure!(
        expected_set.len() == expected.len(),
        "Qwen3.5 TP active decode IDs contain duplicates"
    );
    let mut by_id = HashMap::with_capacity(result.requests.len());
    for DecodeRequestResult {
        request_id,
        token,
        logprob,
    } in &result.requests
    {
        anyhow::ensure!(
            expected_set.contains(request_id),
            "Qwen3.5 TP decode returned unknown request id {}",
            request_id.get()
        );
        let artifact = DecodeArtifact {
            token: *token,
            logprob: logprob.clone(),
        };
        anyhow::ensure!(
            by_id.insert(*request_id, artifact).is_none(),
            "Qwen3.5 TP decode returned duplicate request id {}",
            request_id.get()
        );
    }
    expected
        .into_iter()
        .map(|request_id| {
            by_id.remove(&request_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen3.5 TP decode result is missing request id {}",
                    request_id.get()
                )
            })
        })
        .collect()
}

pub(super) fn split_decode_artifacts(
    artifacts: &[DecodeArtifact],
) -> (Vec<u32>, Vec<Option<TokenLogprob>>) {
    artifacts
        .iter()
        .map(|artifact| (artifact.token, artifact.logprob.clone()))
        .unzip()
}

pub(super) fn servable_len(max_context: usize, max_pages: usize, page_size: usize) -> u32 {
    max_context
        .min(max_pages.saturating_mul(page_size))
        .try_into()
        .unwrap_or(u32::MAX)
}

pub(super) struct CublasThreadGuard;

impl Drop for CublasThreadGuard {
    fn drop(&mut self) {
        unsafe {
            crate::ffi::cublas_destroy();
        }
    }
}

/// Bind the CUDA context and init thread-local cuBLAS on the scheduler
/// (driver) thread. Must run on first `step`, not the load thread.
pub(super) fn bind_model_thread(model: &Qwen35Model) -> Result<CublasThreadGuard> {
    let ctx = model.device_ctx();
    unsafe {
        let err = crate::ffi::cuda_set_device(ctx.device_ordinal as i32);
        if err != 0 {
            return Err(anyhow::anyhow!(
                "Failed to set CUDA device {} on Qwen3.5 scheduler thread: cudaError={}",
                ctx.device_ordinal,
                err
            ));
        }
    }
    ctx.ctx.bind_to_thread().map_err(|e| {
        anyhow::anyhow!("Failed to bind CUDA context to Qwen3.5 scheduler thread: {e}")
    })?;
    unsafe {
        crate::ffi::cublas_init();
    }
    model.tune_decode_gemm_algos()?;
    Ok(CublasThreadGuard)
}

pub(super) trait DecodeDispatchBackend {
    fn is_stop_token(&self, token: u32) -> bool;
    fn take_active_request(
        &mut self,
        active: &mut Vec<ActiveRequest35>,
        idx: usize,
    ) -> ActiveRequest35;
    fn drop_active_state(&mut self, state: &ActiveBackendState) -> Result<()>;
}

impl DecodeDispatchBackend for SingleGpuBackend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.is_stop_token(token)
    }

    fn take_active_request(
        &mut self,
        active: &mut Vec<ActiveRequest35>,
        idx: usize,
    ) -> ActiveRequest35 {
        compact_single_slot(self, active, idx)
    }

    fn drop_active_state(&mut self, _state: &ActiveBackendState) -> Result<()> {
        Ok(())
    }
}

impl DecodeDispatchBackend for Qwen35Backend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.is_stop_token(token)
    }

    fn take_active_request(
        &mut self,
        active: &mut Vec<ActiveRequest35>,
        idx: usize,
    ) -> ActiveRequest35 {
        match self {
            Qwen35Backend::Single(backend) => compact_single_slot(backend, active, idx),
            Qwen35Backend::Tp(_) => active.swap_remove(idx),
        }
    }

    fn drop_active_state(&mut self, state: &ActiveBackendState) -> Result<()> {
        match (self, state) {
            (Qwen35Backend::Single(_), ActiveBackendState::Single { .. }) => Ok(()),
            (Qwen35Backend::Tp(backend), ActiveBackendState::Tp { request_id }) => {
                backend.drop_request(*request_id, DropExpectation::MustExist)
            }
            _ => anyhow::bail!("mismatched Qwen3.5 scheduler backend state during retirement"),
        }
    }
}

/// Remove single-GPU request at `idx` via swap_remove and compact graph slots.
fn compact_single_slot(
    backend: &mut SingleGpuBackend,
    active: &mut Vec<ActiveRequest35>,
    idx: usize,
) -> ActiveRequest35 {
    let compaction = plan::compaction_after_retire(active.len(), idx);
    let removed = active.swap_remove(idx);

    if let Some(compaction) = compaction {
        backend.compact_slot(active, compaction);
    }
    removed
}

pub(super) struct ScheduledChunk {
    pub(super) ids: Vec<EngineRequestId>,
    pub(super) reqs: Vec<Request>,
    pub(super) backend_state: ScheduledChunkBackendState,
    /// Prompt cursor after this step's chunk
    pub(super) ends: Vec<usize>,
    /// This step's chunked token slice per request
    pub(super) windows: Vec<Vec<u32>>,
}

pub(super) struct InflightPrefill {
    // Fields drop in declaration order. Drain the stream before request state
    // can return KV pages or release recurrent/convolution buffers on unwind.
    pub(super) output: AsyncPrefillOutput,
    pub(super) chunk: ScheduledChunk,
    pub(super) sample_seed: u64,
}

pub(super) enum ScheduledChunkBackendState {
    Single {
        kvs: Vec<KvState>,
        recs: Vec<RecurrentState>,
    },
    Tp {
        request_ids: Vec<BackendRequestId>,
    },
}

impl From<Vec<PrefillingRequest35>> for ScheduledChunk {
    fn from(scheduled: Vec<PrefillingRequest35>) -> Self {
        let n = scheduled.len();
        let is_tp = scheduled
            .first()
            .is_some_and(|p| matches!(p.backend_state, PrefillBackendState::Tp { .. }));
        let mut chunk = ScheduledChunk {
            ids: Vec::with_capacity(n),
            reqs: Vec::with_capacity(n),
            backend_state: if is_tp {
                ScheduledChunkBackendState::Tp {
                    request_ids: Vec::with_capacity(n),
                }
            } else {
                ScheduledChunkBackendState::Single {
                    kvs: Vec::with_capacity(n),
                    recs: Vec::with_capacity(n),
                }
            },
            ends: Vec::with_capacity(n),
            windows: Vec::with_capacity(n),
        };
        for p in scheduled {
            let end = p.cursor + p.step_chunk;
            chunk
                .windows
                .push(p.request.prompt_tokens[p.cursor..end].to_vec());
            chunk.ends.push(end);
            chunk.ids.push(p.id);
            chunk.reqs.push(p.request);
            match (&mut chunk.backend_state, p.backend_state) {
                (
                    ScheduledChunkBackendState::Single { kvs, recs },
                    PrefillBackendState::Single { kv, rec },
                ) => {
                    kvs.push(kv);
                    recs.push(rec);
                }
                (
                    ScheduledChunkBackendState::Tp { request_ids },
                    PrefillBackendState::Tp { request_id },
                ) => request_ids.push(request_id),
                _ => unreachable!("mixed Qwen3.5 scheduler backend states in one chunk"),
            }
        }
        chunk
    }
}

pub(super) trait PrefillPromoteBackend {
    fn is_stop_token(&self, token: u32) -> bool;
    fn promote_prefill_state(
        &mut self,
        active_len: usize,
        state: PrefillBackendState,
    ) -> ActiveBackendState;
    fn drop_prefill_state(
        &mut self,
        state: &PrefillBackendState,
        expectation: DropExpectation,
    ) -> Result<()>;
}

impl PrefillPromoteBackend for SingleGpuBackend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.is_stop_token(token)
    }

    fn promote_prefill_state(
        &mut self,
        active_len: usize,
        state: PrefillBackendState,
    ) -> ActiveBackendState {
        let PrefillBackendState::Single { kv, rec } = state else {
            panic!("single-GPU promotion received TP prefill state");
        };
        let slot_idx = slot_for_new_request(active_len, self.max_batch())
            .expect("admission must reserve a graph slot");
        self.copy_recurrent_to_slot(&rec, slot_idx)
            .expect("copy recurrent state to slot failed");
        ActiveBackendState::Single {
            kv,
            graph_slot_idx: slot_idx,
        }
    }

    fn drop_prefill_state(
        &mut self,
        _state: &PrefillBackendState,
        _expectation: DropExpectation,
    ) -> Result<()> {
        Ok(())
    }
}

impl PrefillPromoteBackend for Qwen35Backend {
    fn is_stop_token(&self, token: u32) -> bool {
        self.is_stop_token(token)
    }

    fn promote_prefill_state(
        &mut self,
        active_len: usize,
        state: PrefillBackendState,
    ) -> ActiveBackendState {
        match (self, state) {
            (Qwen35Backend::Single(single), PrefillBackendState::Single { kv, rec }) => {
                let slot_idx = slot_for_new_request(active_len, single.max_batch())
                    .expect("admission must reserve a graph slot");
                single
                    .copy_recurrent_to_slot(&rec, slot_idx)
                    .expect("copy recurrent state to slot failed");
                ActiveBackendState::Single {
                    kv,
                    graph_slot_idx: slot_idx,
                }
            }
            (Qwen35Backend::Tp(_), PrefillBackendState::Tp { request_id }) => {
                ActiveBackendState::Tp { request_id }
            }
            _ => panic!("mismatched Qwen3.5 scheduler backend state during promotion"),
        }
    }

    fn drop_prefill_state(
        &mut self,
        state: &PrefillBackendState,
        expectation: DropExpectation,
    ) -> Result<()> {
        match (self, state) {
            (Qwen35Backend::Single(_), PrefillBackendState::Single { .. }) => Ok(()),
            (Qwen35Backend::Tp(backend), PrefillBackendState::Tp { request_id }) => {
                backend.drop_request(*request_id, expectation)
            }
            _ => anyhow::bail!("mismatched Qwen3.5 scheduler backend state during prefill drop"),
        }
    }
}

pub(super) fn split_scheduled_backend_state(
    backend_state: ScheduledChunkBackendState,
) -> Vec<PrefillBackendState> {
    match backend_state {
        ScheduledChunkBackendState::Single { kvs, recs } => kvs
            .into_iter()
            .zip(recs)
            .map(|(kv, rec)| PrefillBackendState::Single { kv, rec })
            .collect(),
        ScheduledChunkBackendState::Tp { request_ids } => request_ids
            .into_iter()
            .map(|request_id| PrefillBackendState::Tp { request_id })
            .collect(),
    }
}
