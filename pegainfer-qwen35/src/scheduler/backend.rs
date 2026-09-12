//! Qwen3.5 scheduler backend abstraction (single-GPU + TP).

use super::*;
use crate::logprobs::LogprobSnapshot;
use crate::tp_executor::TpBeginRequestError;

pub(super) struct SingleGpuBackend {
    pub(super) model: Qwen35Model,
    pub(super) kv_cache: Qwen35PrefixCache,
    recurrent_store: RecurrentStateStore,
    graph_state: BatchDecodeGraphState,
    prefill_stream: Option<Arc<CudaStream>>,
}

// One instance per scheduler; the size asymmetry costs nothing here.
#[allow(clippy::large_enum_variant)]
pub(super) enum SchedulerBackend {
    Single(SingleGpuBackend),
    Tp(TpSchedulerBackend),
}

impl Drop for SchedulerBackend {
    fn drop(&mut self) {
        self.log_prefix_cache_stats();
    }
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

/// Pair each sampled token with its host logprob row, where one was requested.
fn attached_logprobs(
    cpu_logits: Vec<Option<LogprobSnapshot>>,
    tokens: &[u32],
) -> Vec<Option<TokenLogprob>> {
    cpu_logits
        .into_iter()
        .zip(tokens)
        .map(|(row, &token)| {
            row.and_then(|(row, top_k)| {
                pegainfer_sample::token_logprob_from_row(&row, token, top_k)
            })
        })
        .collect()
}

pub(super) struct TpSchedulerBackend {
    pub(super) executor: Qwen35TpExecutor,
    next_request_id: u64,
    /// Slot move derived by the in-flight `take_active_request`; consumed by
    /// the paired `drop_active_state` so the workers apply the same move.
    pub(super) pending_compaction: Option<TpSlotCompaction>,
}

pub(super) enum AdmissionError {
    Recoverable(anyhow::Error),
    Fatal(anyhow::Error),
}

impl From<TpBeginRequestError> for AdmissionError {
    fn from(error: TpBeginRequestError) -> Self {
        match error {
            TpBeginRequestError::Recoverable(error) => Self::Recoverable(error),
            TpBeginRequestError::Fatal(error) => Self::Fatal(error),
        }
    }
}

impl SingleGpuBackend {
    fn schedule_prefill_views(
        &self,
        kvs: &mut [Box<RequestKv>],
        windows: &[Vec<u32>],
    ) -> Result<Vec<KvView>> {
        debug_assert_eq!(kvs.len(), windows.len());
        for (scheduled, (kv, window)) in kvs.iter_mut().zip(windows).enumerate() {
            if let Err(error) = self.kv_cache.schedule_prefill(kv, window.len()) {
                revert_scheduled_requests(
                    &self.kv_cache,
                    kvs.iter_mut().take(scheduled).map(Box::as_mut),
                );
                return Err(error);
            }
        }
        Ok(kvs
            .iter()
            .zip(windows)
            .map(|(kv, window)| self.kv_cache.prefill_view(kv, window.len()))
            .collect())
    }

    fn schedule_decode_views(&self, active: &mut [ActiveRequest35]) -> Result<Vec<KvView>> {
        for (scheduled, request) in active.iter_mut().enumerate() {
            let ActiveBackendState::Single { kv, .. } = &mut request.backend_state else {
                panic!("single-GPU decode received TP active state")
            };
            if let Err(error) = self.kv_cache.schedule_decode(kv) {
                revert_scheduled_requests(
                    &self.kv_cache,
                    active
                        .iter_mut()
                        .take(scheduled)
                        .filter_map(active_request_kv),
                );
                return Err(error);
            }
        }
        Ok(active
            .iter()
            .map(|request| match &request.backend_state {
                ActiveBackendState::Single { kv, .. } => self.kv_cache.decode_view(kv),
                ActiveBackendState::Tp { .. } => {
                    panic!("single-GPU decode received TP active state")
                }
            })
            .collect())
    }

    pub(super) fn apply_prefill(
        &mut self,
        chunk: &mut ScheduledChunk,
        tokens: &[u32],
    ) -> Result<()> {
        let ScheduledChunkBackendState::Single { kvs, recs } = &mut chunk.backend_state else {
            anyhow::bail!("single-GPU commit received TP chunk state")
        };
        for (i, (kv, rec)) in kvs.iter_mut().zip(recs.iter()).enumerate() {
            let is_final = chunk.ends[i] == chunk.reqs[i].prompt_tokens.len();
            let boundary = self
                .kv_cache
                .apply_prefill(kv, is_final.then_some(tokens[i]))?;
            anyhow::ensure!(
                rec.seq_len == boundary,
                "Qwen3.5 prefill apply position mismatch: kv={boundary}, recurrent={}",
                rec.seq_len
            );
            if let Some(reservation) = self.kv_cache.reserve_snapshot(kv, boundary)? {
                if let Err(error) = self.recurrent_store.save(
                    self.model.device_ctx(),
                    reservation.recurrent_slot(),
                    rec,
                ) {
                    self.kv_cache.abort_snapshot(reservation);
                    return Err(error);
                }
                self.kv_cache.publish_snapshot(reservation);
            }
        }
        Ok(())
    }

    pub(super) fn apply_decode(
        &self,
        active: &mut [ActiveRequest35],
        tokens: &[u32],
    ) -> Result<()> {
        anyhow::ensure!(active.len() == tokens.len(), "decode apply row mismatch");
        for (req, &token) in active.iter_mut().zip(tokens) {
            let ActiveBackendState::Single { kv, .. } = &mut req.backend_state else {
                anyhow::bail!("single-GPU decode apply received TP state")
            };
            self.kv_cache.apply_decode(kv, token)?;
        }
        Ok(())
    }

    pub(super) fn log_prefix_cache_stats(&self) {
        let cache = &self.kv_cache;
        let stats = cache.stats();
        info!(
            "Qwen3.5 prefix cache summary: joint_hits={}, hit_tokens={}, kv_only_fallbacks={}, snapshot_misses={}, inserts={}, evictions={}, occupancy={}/{}",
            stats.joint_hits,
            stats.joint_hit_tokens,
            stats.kv_only_fallbacks,
            stats.snapshot_misses,
            stats.inserts,
            stats.evictions,
            cache.snapshot_occupancy(),
            cache.snapshot_slots(),
        );
    }

    pub(super) fn new(
        model: Qwen35Model,
        max_batch: usize,
        decode_overlap: Qwen35DecodeOverlap,
    ) -> Result<Self> {
        anyhow::ensure!(max_batch > 0, "Qwen3.5 max_batch must be > 0");
        let manager =
            KvCacheManager::from_buffer(model.kv_buffer().clone(), model.kv_buffer().num_blocks())?;
        let kv_cache = Qwen35PrefixCache::new(manager, model.prefix_snapshot_slots())?;
        let recurrent_store = RecurrentStateStore::new(
            model.device_ctx(),
            model.config(),
            model.geometry,
            model.prefix_snapshot_slots(),
        )?;
        debug_assert_eq!(recurrent_store.len(), kv_cache.snapshot_slots());
        let graph_capacity = crate::batch_decode_graph::bucket_for(max_batch);
        let graph_state = model.create_batch_decode_graph_state_with_capacity(
            graph_capacity,
            kv_cache.pool().padding_block_id(),
        )?;
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
            kv_cache,
            recurrent_store,
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
        self.kv_cache.pool().block_size()
    }

    pub(super) fn available_pages(&self) -> usize {
        self.kv_cache.pool().available_blocks()
    }

    pub(super) fn active_joint_prefix_pages(&self, req: &SchedulerRequest) -> usize {
        self.kv_cache
            .active_joint_prefix_pages(&req.prompt_tokens, req.lora_adapter.as_deref())
    }

    pub(super) fn capacity_pages_for_requests(&self) -> usize {
        self.kv_cache.pool().max_request_blocks()
    }

    pub(super) fn max_position_embeddings(&self) -> usize {
        self.model.config().max_position_embeddings
    }

    pub(super) fn alloc_prefill_state(
        &mut self,
        req: &SchedulerRequest,
    ) -> Result<(PrefillBackendState, usize), AdmissionError> {
        let mut rec = self
            .alloc_recurrent()
            .map_err(AdmissionError::Recoverable)?;
        let (mut kv, restore) = self
            .kv_cache
            .begin_request(
                &req.prompt_tokens,
                req.max_tokens,
                req.lora_adapter.as_deref(),
                true,
            )
            .map_err(AdmissionError::Recoverable)?;
        let cached_tokens = if let Some(restore) = restore {
            if let Err(error) = self.recurrent_store.restore(
                self.model.device_ctx(),
                restore.recurrent_slot(),
                &mut rec,
            ) {
                let _ = self.kv_cache.release_request(&mut kv);
                return Err(AdmissionError::Recoverable(error));
            }
            match self.kv_cache.finish_restore(&kv, restore, &[rec.seq_len]) {
                Ok(tokens) => tokens,
                Err(error) => {
                    let _ = self.kv_cache.release_request(&mut kv);
                    return Err(AdmissionError::Recoverable(error));
                }
            }
        } else {
            0
        };
        Ok((
            PrefillBackendState::Single {
                kv: Box::new(kv),
                rec,
            },
            cached_tokens,
        ))
    }

    pub(super) fn alloc_recurrent(&self) -> Result<RecurrentState> {
        RecurrentState::new(
            self.model.device_ctx(),
            self.model.config(),
            self.model.geometry,
        )
    }

    pub(super) fn batch_prefill_logits(&self, chunk: &mut ScheduledChunk) -> Result<HiddenStates> {
        let window_refs: Vec<&[u32]> = chunk.windows.iter().map(Vec::as_slice).collect();
        let ScheduledChunkBackendState::Single { kvs, recs } = &mut chunk.backend_state else {
            anyhow::bail!("single-GPU prefill received TP chunk state");
        };
        let views = self.schedule_prefill_views(kvs, &chunk.windows)?;
        let mut rec_refs: Vec<&mut RecurrentState> = recs.iter_mut().collect();
        let result = self.model.batch_prefill_logits(
            &window_refs,
            &views,
            &mut rec_refs,
            self.kv_cache.buffer(),
        );
        if result.is_err() {
            revert_scheduled_requests(&self.kv_cache, kvs.iter_mut().map(Box::as_mut));
        }
        result
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
        let views = self.schedule_prefill_views(kvs, &chunk.windows)?;
        let mut rec_refs: Vec<&mut RecurrentState> = recs.iter_mut().collect();
        let logits = match self.model.batch_prefill_logits_on_stream(
            Arc::clone(&prefill_stream),
            &window_refs,
            &views,
            self.kv_cache.buffer(),
            &mut rec_refs,
        ) {
            Ok(logits) => logits,
            Err(err) => {
                if let Err(sync_err) = prefill_stream.synchronize() {
                    fatal_cuda_lifecycle(&format!(
                        "Qwen3.5 async prefill failed ({err}); stream drain failed: {sync_err}"
                    ));
                }
                revert_scheduled_requests(&self.kv_cache, kvs.iter_mut().map(Box::as_mut));
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
        let prefill_views = self.schedule_prefill_views(kvs, &chunk.windows)?;
        let mut rec_refs: Vec<&mut RecurrentState> = recs.iter_mut().collect();
        let decode_tokens: Vec<u32> = active.iter().map(|r| r.last_token).collect();
        let decode_views = match self.schedule_decode_views(active) {
            Ok(views) => views,
            Err(error) => {
                revert_scheduled_requests(&self.kv_cache, kvs.iter_mut().map(Box::as_mut));
                return Err(error);
            }
        };
        let result = self.model.unified_step(
            &window_refs,
            &prefill_views,
            &mut rec_refs,
            &decode_tokens,
            &decode_views,
            self.kv_cache.buffer(),
            &mut self.graph_state,
        );
        if result.is_err() {
            revert_scheduled_requests(&self.kv_cache, kvs.iter_mut().map(Box::as_mut));
            revert_scheduled_requests(
                &self.kv_cache,
                active.iter_mut().filter_map(active_request_kv),
            );
        }
        result
    }

    pub(super) fn decode_graph(&mut self, active: &mut [ActiveRequest35]) -> Result<()> {
        let token_ids: Vec<u32> = active.iter().map(|r| r.last_token).collect();
        let views = self.schedule_decode_views(active)?;
        let result = self.model.batch_decode_graph(
            &token_ids,
            &views,
            self.kv_cache.buffer(),
            &mut self.graph_state,
            crate::batch_decode::DecodeGraphUse::Serve,
        );
        if result.is_err() {
            revert_scheduled_requests(
                &self.kv_cache,
                active.iter_mut().filter_map(active_request_kv),
            );
        }
        result
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
        let requested_logprobs: Vec<Option<usize>> = pending.iter().map(|r| r.logprobs).collect();
        let cpu_logits =
            snapshot_requested_logprobs(self.model.device_ctx(), logits, &requested_logprobs)?;
        let params_refs: Vec<&SamplingParams> = pending.iter().map(|r| &r.params).collect();
        let tokens = self.model.select_tokens_from_logits_varied(
            logits,
            &mut self.graph_state.buffers,
            &params_refs,
            sample_seed,
        )?;

        let logprobs = attached_logprobs(cpu_logits, &tokens);
        Ok((tokens, logprobs))
    }

    pub(super) fn sample_decode_logits(
        &mut self,
        active: &[ActiveRequest35],
        sample_seed: u64,
    ) -> Result<(Vec<u32>, Vec<Option<TokenLogprob>>)> {
        let requested_logprobs: Vec<Option<usize>> = active.iter().map(|r| r.logprobs).collect();
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

        let logprobs = attached_logprobs(cpu_logits, &tokens);
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
    pub(super) fn alloc_prefill_state(
        &mut self,
        req: &SchedulerRequest,
    ) -> Result<(PrefillBackendState, usize), AdmissionError> {
        let request_id = self.alloc_request_id();
        let cached_tokens = self
            .executor
            .begin_request(
                request_id,
                &req.prompt_tokens,
                req.max_tokens,
                req.lora_adapter.as_deref(),
                true,
            )
            .map_err(AdmissionError::from)?;
        Ok((PrefillBackendState::Tp { request_id }, cached_tokens))
    }

    pub(super) fn active_joint_prefix_pages(&self, req: &SchedulerRequest) -> usize {
        self.executor
            .active_joint_prefix_pages(&req.prompt_tokens, req.lora_adapter.as_deref())
    }

    pub(super) fn new(
        model_path: &str,
        device_ordinals: &[usize],
        max_batch: usize,
        max_prefill_tokens: usize,
        enable_cuda_graph: bool,
        prefix_snapshot_bytes: usize,
    ) -> Result<Self> {
        let executor = Qwen35TpExecutor::from_runtime_with_limits_and_prefix(
            model_path,
            enable_cuda_graph,
            device_ordinals,
            max_batch,
            max_prefill_tokens,
            prefix_snapshot_bytes,
        )?;
        Ok(Self {
            executor,
            next_request_id: 1,
            pending_compaction: None,
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
        _active: &[ActiveRequest35],
        _prefilling: &[PrefillingRequest35],
    ) -> usize {
        self.executor.available_pages()
    }

    pub(super) fn execute_prefill_chunk(
        &mut self,
        chunk: &ScheduledChunk,
        sample_seed: u64,
    ) -> Result<Vec<Option<PrefillArtifact>>> {
        let items = tp_prefill_items(chunk)?;
        let result = self
            .executor
            .execute_prefill_chunks_with_seed(&items, sample_seed)?;
        align_prefill_results(chunk, &result).map_err(|err| {
            self.executor
                .poison_after_mutation("prefill artifacts", &err)
        })
    }

    pub(super) fn execute_decode(
        &mut self,
        active: &[ActiveRequest35],
        sample_seed: u64,
    ) -> Result<Vec<DecodeArtifact>> {
        let items = tp_decode_items(active)?;
        let result = self.executor.execute_decode_items(&items, sample_seed)?;
        align_decode_results(active, &result).map_err(|err| {
            self.executor
                .poison_after_mutation("decode artifacts", &err)
        })
    }

    pub(super) fn execute_unified(
        &mut self,
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
                .poison_after_mutation("unified prefill artifacts", &err)
        })?;
        let decode = align_decode_results(active, &result.decode).map_err(|err| {
            self.executor
                .poison_after_mutation("unified decode artifacts", &err)
        })?;
        Ok(AlignedUnifiedArtifacts { prefill, decode })
    }

    pub(super) fn drop_request(
        &mut self,
        request_id: RequestId,
        expectation: DropExpectation,
    ) -> Result<()> {
        self.executor
            .drop_request_with_compaction(request_id, expectation, None)
    }

    /// Remove the TP request at `idx` via swap_remove and stash the resulting
    /// slot compaction for the paired `drop_active_state`. Mirrors
    /// `compact_single_slot`: after the swap, slots `0..active.len()` stay
    /// dense because the moved request's slot follows it.
    pub(super) fn take_active_request(
        &mut self,
        active: &mut Vec<ActiveRequest35>,
        idx: usize,
    ) -> ActiveRequest35 {
        let compaction = compaction_after_retire(active.len(), idx);
        let removed = active.swap_remove(idx);

        self.pending_compaction = compaction.map(|compaction| {
            let moved = &mut active[idx];
            let ActiveBackendState::Tp {
                request_id,
                slot_idx,
            } = &mut moved.backend_state
            else {
                panic!("TP scheduler received single-GPU active state")
            };
            debug_assert_eq!(*slot_idx, compaction.moved_from);
            *slot_idx = compaction.moved_to;
            TpSlotCompaction {
                moved_request_id: *request_id,
                from: compaction.moved_from,
                to: compaction.moved_to,
            }
        });
        removed
    }
}

impl SchedulerBackend {
    pub(super) fn log_prefix_cache_stats(&self) {
        match self {
            Self::Single(backend) => backend.log_prefix_cache_stats(),
            Self::Tp(backend) => backend.executor.log_prefix_cache_stats(),
        }
    }

    pub(super) fn active_joint_prefix_pages(&self, req: &SchedulerRequest) -> usize {
        match self {
            Self::Single(backend) => backend.active_joint_prefix_pages(req),
            Self::Tp(backend) => backend.active_joint_prefix_pages(req),
        }
    }

    pub(super) fn snapshot_stride(&self) -> Option<usize> {
        match self {
            Self::Single(backend) if backend.kv_cache.enabled() => {
                Some(crate::prefix_cache::SNAPSHOT_STRIDE_TOKENS)
            }
            Self::Tp(backend) if backend.executor.prefix_cache_enabled() => {
                Some(crate::prefix_cache::SNAPSHOT_STRIDE_TOKENS)
            }
            Self::Single(_) | Self::Tp(_) => None,
        }
    }

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

    pub(super) fn alloc_prefill_state(
        &mut self,
        req: &SchedulerRequest,
    ) -> Result<(PrefillBackendState, usize), AdmissionError> {
        match self {
            Self::Single(backend) => backend.alloc_prefill_state(req),
            Self::Tp(backend) => backend.alloc_prefill_state(req),
        }
    }

    pub(super) fn is_stop_token(&self, token: u32) -> bool {
        match self {
            Self::Single(backend) => backend.is_stop_token(token),
            Self::Tp(backend) => backend.is_stop_token(token),
        }
    }
}
