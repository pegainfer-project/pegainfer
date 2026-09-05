//! Qwen3.5 scheduler backend abstraction (single-GPU + TP).

use super::*;

pub(super) struct SingleGpuBackend {
    model: Qwen35Model,
    graph_state: BatchDecodeGraphState,
    prefill_stream: Option<Arc<CudaStream>>,
}

// One instance per scheduler; the size asymmetry costs nothing here.
#[allow(clippy::large_enum_variant)]
pub(super) enum SchedulerBackend {
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

pub(super) fn fatal_cuda_lifecycle(message: &str) -> ! {
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

    pub(super) fn max_batch(&self) -> usize {
        // #470: admit the requested `--max-batch`, which may sit below the loaded
        // graph bucket (e.g. 5 on bucket 8); never exceed the physical slots.
        self.model
            .decode_admission_batch
            .min(self.graph_state.slot_states.len())
            .max(1)
    }

    pub(super) fn page_size(&self) -> usize {
        self.model.kv_pool().layout().page_size
    }

    pub(super) fn available_pages(&self) -> usize {
        self.model.kv_pool().available_pages()
    }

    pub(super) fn capacity_pages_for_requests(&self) -> usize {
        self.model.kv_pool().capacity_pages().saturating_sub(1)
    }

    pub(super) fn max_position_embeddings(&self) -> usize {
        self.model.config().max_position_embeddings
    }

    pub(super) fn alloc_kv(&self) -> KvState {
        self.model.alloc_kv()
    }

    pub(super) fn alloc_recurrent(&self) -> Result<RecurrentState> {
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
        pending: &[SchedulerRequest],
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

    pub(super) fn is_stop_token(&self, token: u32) -> bool {
        self.model.is_stop_token(token)
    }

    pub(super) fn copy_recurrent_to_slot(
        &mut self,
        recurrent: &RecurrentState,
        slot_idx: usize,
    ) -> Result<()> {
        self.graph_state
            .copy_state_to_slot(self.model.device_ctx(), recurrent, slot_idx)
    }

    pub(super) fn compact_slot(
        &mut self,
        active: &mut [ActiveRequest35],
        compaction: plan::SlotCompaction,
    ) {
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

    pub(super) fn alloc_request_id(&mut self) -> RequestId {
        let id = RequestId::new(self.next_request_id);
        self.next_request_id = self.next_request_id.wrapping_add(1).max(1);
        id
    }

    pub(super) fn max_batch(&self) -> usize {
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

    pub(super) fn is_stop_token(&self, token: u32) -> bool {
        self.executor.is_stop_token(token)
    }

    pub(super) fn available_pages(
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

    pub(super) fn drop_request(
        &self,
        request_id: RequestId,
        expectation: DropExpectation,
    ) -> Result<()> {
        self.executor.drop_request(request_id, expectation)
    }
}

impl SchedulerBackend {
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
