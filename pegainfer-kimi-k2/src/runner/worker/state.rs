use super::forward::*;
use super::runtime::*;
use super::*;
use crate::config::KIMI_K2_VOCAB;

impl KimiRankThreadState {
    /// Collective: every rank's worker thread must execute this concurrently
    /// (the DeepEP context create blocks until all ranks join the NCCL
    /// communicator and register the symmetric window).
    pub(super) fn enable_deepep(&mut self, unique_id: &[u8; 128], num_ranks: usize) -> Result<()> {
        ensure!(
            self.deepep.is_none(),
            "Kimi rank {} DeepEP already enabled",
            self.sliced_load_plan.rank
        );
        ensure!(
            self.tp_comm.is_none(),
            "Kimi rank {} DeepEP is the TP1/DP8 EP path; TP8 uses the NCCL MoE backend",
            self.sliced_load_plan.rank
        );
        self.ctx.set_current()?;
        let rank = self.sliced_load_plan.rank;
        self.deepep = Some(crate::runner::moe_deepep::KimiMoeDeepEpState::new(
            &self.ctx.as_device_context(),
            unique_id,
            num_ranks,
            rank,
        )?);
        // The DeepEP decode path is host-quiet (no allocs, no D2H, no seq_len
        // mutations; persistent worst-case buffers), so it is graph-capturable
        // (#227): the decode step threads the EP state into the capture
        // closure and replays only at full arena occupancy.
        Ok(())
    }

    pub(super) fn init_tp_comm(&mut self, id: Id, world_size: usize) -> Result<()> {
        ensure!(
            self.tp_comm.is_none(),
            "Kimi rank {} TP comm already attached",
            self.sliced_load_plan.rank
        );
        self.ctx.set_current()?;
        let rank = self.sliced_load_plan.rank;
        let device_ctx = self.ctx.as_device_context();
        let comm = Comm::from_rank(device_ctx.stream, rank, world_size, id)
            .map_err(|err| anyhow::anyhow!("Kimi rank {rank} NCCL init failed: {err:?}"))?;
        self.tp_comm = Some(OwnedRankComm(comm));
        Ok(())
    }

    pub(super) fn load_sliced_weights(
        &mut self,
        model_path: &Path,
    ) -> Result<KimiRankWeightLoadReport> {
        let started = Instant::now();
        let rank = self.sliced_load_plan.rank;
        debug!("rank {rank} start rank weight init");
        let load_output = load_rank_sliced_weights_to_gpu(
            &self.ctx,
            model_path,
            &self.sliced_load_plan,
            &self.weight_names,
        )
        .with_context(|| {
            format!(
                "failed to load Kimi rank {} sliced weights to GPU",
                self.sliced_load_plan.rank
            )
        })?;
        let weights = load_output.weights;
        let expert_kernel_weights = load_output.expert_kernel_weights;
        let tensor_count = load_output.loaded_tensor_count;
        let total_bytes = load_output.loaded_total_bytes;
        debug!("rank {rank} start one-token forward cache build");
        let cache_started = Instant::now();
        let one_token_cache =
            KimiOneTokenForwardCache::from_gpu_weights(&self.ctx, &weights, &self.weight_names)
                .with_context(|| {
                    format!(
                        "failed to build Kimi rank {} one-token forward cache",
                        self.sliced_load_plan.rank
                    )
                })?;
        debug!(
            "rank {rank} one-token forward cache build cost {:.2}s",
            cache_started.elapsed().as_secs_f64()
        );
        // Allocate the shared KV pool eagerly: an OOM should kill bringup,
        // not the first request that fills the pool.
        let kv_pool_started = Instant::now();
        let kv_pool = KimiWorkerKvPool::new(
            &self.ctx.as_device_context(),
            KIMI_K2_LAYERS,
            self.kv_pool_pages,
        )
        .with_context(|| {
            format!(
                "failed to allocate Kimi rank {rank} KV pool ({} pages)",
                self.kv_pool_pages
            )
        })?;
        debug!(
            "rank {rank} KV pool ({} pages, {} tokens) alloc cost {:.2}s",
            self.kv_pool_pages,
            self.kv_pool_pages * KIMI_KV_PAGE_SIZE,
            kv_pool_started.elapsed().as_secs_f64()
        );
        let decode_arenas = KimiWorkerDecodeArenas::new(
            one_token_cache.vocab_rows,
            &self.local_dims,
            self.kv_pool_pages,
        );
        let report = KimiRankWeightLoadReport::from_loaded_weights(
            tensor_count,
            total_bytes,
            &expert_kernel_weights,
        );
        let loaded = KimiRankLoadedWeights {
            gpu: weights,
            expert_kernels: expert_kernel_weights,
            one_token_cache,
            kv_pool,
            decode_arenas,
        };
        ensure!(
            loaded.gpu.rank == report.rank,
            "Kimi loaded rank {} does not match report rank {}",
            loaded.gpu.rank,
            report.rank
        );
        ensure!(
            loaded.expert_kernels.layers.len() == report.expert_kernel_layers,
            "Kimi expert kernel layer count {} does not match report count {}",
            loaded.expert_kernels.layers.len(),
            report.expert_kernel_layers
        );
        self.loaded = Some(loaded);
        debug!(
            "rank {rank} rank weight init cost {:.2}s: tensors={}, bytes={}, expert_layers={}",
            started.elapsed().as_secs_f64(),
            tensor_count,
            ByteSize(total_bytes as u64),
            report.expert_kernel_layers
        );
        Ok(report)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_prompt_next_token(
        &mut self,
        slot: usize,
        decode_batch_size: usize,
        input_ids: &[u32],
        cached_tokens: usize,
        ep_max_seq_len: usize,
        kv_pages: &KimiKvStepPages,
        row: KimiRowOptions,
        seed: u64,
    ) -> Result<KimiOneTokenForwardReport> {
        self.forward_prompt_next_token_inner(
            slot,
            decode_batch_size,
            input_ids,
            cached_tokens,
            ep_max_seq_len,
            kv_pages,
            row,
            seed,
        )
    }

    pub(super) fn ensure_decode_arena(&mut self, decode_batch_size: usize) -> Result<()> {
        self.ctx.set_current()?;
        let device_ctx = self.ctx.as_device_context();
        let loaded = self.loaded.as_mut().ok_or_else(|| {
            anyhow::anyhow!("Kimi rank weights must be loaded before decode arena allocation")
        })?;
        loaded
            .decode_arenas
            .get_mut(&device_ctx, decode_batch_size)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_decode_batch_next_tokens(
        &mut self,
        token_ids: &[u32],
        append_positions: &[usize],
        slots: &[usize],
        decode_batch_size: usize,
        kv_pages: &KimiKvStepPages,
        rows: &[KimiRowOptions],
        seed: u64,
    ) -> Result<Vec<KimiOneTokenForwardReport>> {
        ensure!(!token_ids.is_empty(), "Kimi batch decode requires tokens");
        ensure!(
            token_ids.len() == append_positions.len() && token_ids.len() == slots.len(),
            "Kimi batch decode input length mismatch: tokens={}, positions={}, slots={}",
            token_ids.len(),
            append_positions.len(),
            slots.len()
        );
        ensure!(
            rows.len() == token_ids.len(),
            "Kimi batch decode row options length mismatch: tokens={}, rows={}",
            token_ids.len(),
            rows.len()
        );
        self.ctx.set_current()?;
        let loaded = self.loaded.as_mut().ok_or_else(|| {
            anyhow::anyhow!("Kimi rank weights must be loaded before batch decode")
        })?;
        let tp_comm_ref = self.tp_comm.as_ref().map(super::OwnedRankComm::get);
        let device_ctx = self.ctx.as_device_context();
        let decode_aux_ctx = DeviceContext {
            ctx: Arc::clone(&self.decode_aux_ctx.ctx),
            stream: Arc::clone(&self.decode_aux_ctx.stream),
            device_ordinal: self.decode_aux_ctx.device_ordinal,
        };
        let KimiRankLoadedWeights {
            gpu,
            expert_kernels,
            one_token_cache: cache,
            kv_pool,
            decode_arenas,
        } = loaded;
        let rank = gpu.rank;
        let active_len = token_ids.len();
        ensure!(
            (1..=KIMI_DECODE_MAX_BATCH).contains(&decode_batch_size),
            "Kimi decode batch size {decode_batch_size} must be in 1..={KIMI_DECODE_MAX_BATCH}"
        );
        ensure!(
            active_len <= decode_batch_size,
            "Kimi active decode rows {active_len} exceed decode batch size {decode_batch_size}"
        );
        let decode_arena = decode_arenas.get_mut(&device_ctx, decode_batch_size)?;
        #[cfg(feature = "kernel-call-trace")]
        if rank == 0 && call_trace::is_enabled() {
            let kv_len = append_positions
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
                .saturating_add(1);
            for call in crate::batch_decode_trace::trace_decode_kernel_calls(
                "",
                decode_arena.batch_size,
                kv_len,
            )? {
                call_trace::record_call(call);
            }
        }
        decode_arena
            .configure_batch_decode(&device_ctx, slots, append_positions, kv_pages)
            .with_context(|| format!("Kimi rank {rank} configure batch decode KV page table"))?;
        decode_arena
            .upload_batch_tokens(&device_ctx, token_ids)
            .with_context(|| format!("Kimi rank {rank} upload batch decode tokens"))?;

        let local_heads = self.local_dims.local_heads;
        let deepep = self.deepep.as_mut();
        let forward_result = if self.enable_cuda_graph && deepep.is_some() {
            if active_len == decode_batch_size {
                // Full arena occupancy: the captured shape matches exactly, so
                // replay is safe. No cross-rank barrier — DeepEP decode has no
                // host-side collectives, and DP ranks reach full occupancy on
                // different steps (a shared barrier would deadlock). During one
                // rank's capture the others' device-side spins resolve as soon
                // as its first graph launch executes the captured step. The
                // safety margin for that pause is the DeepEP device timeout
                // (`kTimeoutCycles` ≈ 100 s): a peer already spinning in
                // dispatch traps on the PEER rank if our capture+instantiate
                // ever exceeds it. Today the window is tens of ms — keep it
                // far below that ceiling.
                let mut graph = std::mem::take(&mut decode_arena.graph);
                let result = graph.run_or_capture(&device_ctx, || {
                    forward_decode_batch_next_token_kernels(
                        &device_ctx,
                        &decode_aux_ctx,
                        tp_comm_ref,
                        cache,
                        expert_kernels,
                        kv_pool,
                        decode_arena,
                        active_len,
                        local_heads,
                        deepep,
                    )
                });
                decode_arena.graph = graph;
                result
            } else {
                // Partial bucket: run eager — a graph captured at this
                // active_len would bake the row count and silently compute
                // the wrong rows on replay.
                forward_decode_batch_next_token_kernels(
                    &device_ctx,
                    &decode_aux_ctx,
                    tp_comm_ref,
                    cache,
                    expert_kernels,
                    kv_pool,
                    decode_arena,
                    active_len,
                    local_heads,
                    deepep,
                )
            }
        } else if self.enable_cuda_graph {
            let mut graph = std::mem::take(&mut decode_arena.graph);
            let graph_barrier = Arc::clone(&self.collective_barrier);
            let result = graph.run_or_capture_synchronized(
                &device_ctx,
                |_| {
                    graph_barrier.wait();
                },
                || {
                    forward_decode_batch_next_token_kernels(
                        &device_ctx,
                        &decode_aux_ctx,
                        tp_comm_ref,
                        cache,
                        expert_kernels,
                        kv_pool,
                        decode_arena,
                        active_len,
                        local_heads,
                        None,
                    )
                },
            );
            decode_arena.graph = graph;
            result
        } else {
            forward_decode_batch_next_token_kernels(
                &device_ctx,
                &decode_aux_ctx,
                tp_comm_ref,
                cache,
                expert_kernels,
                kv_pool,
                decode_arena,
                active_len,
                local_heads,
                deepep,
            )
        };
        forward_result?;

        // Non-greedy rows: one batched FlashInfer sampling pass over the
        // logits arena (its own sync, in addition to the argmax read below).
        // All-greedy batches skip this entirely — the greedy path is unchanged.
        let sampling_rows: Vec<pegainfer_sample::BatchSamplingRow> = rows
            .iter()
            .enumerate()
            .filter(|(_, r)| !r.sampling.is_greedy())
            .map(|(i, r)| pegainfer_sample::BatchSamplingRow {
                row: i,
                temperature: r.sampling.temperature,
                top_k: r.sampling.top_k,
                top_p: r.sampling.top_p,
                min_p: r.sampling.min_p,
            })
            .collect();
        let sampled = if sampling_rows.is_empty() {
            Vec::new()
        } else {
            ensure!(
                cache.vocab_start == 0 && cache.vocab_rows == KIMI_K2_VOCAB,
                "Kimi sampling requires an unsharded vocab (TP1); a vocab shard \
                 cannot sample the global distribution (#237, #226)"
            );
            let scratch = decode_arena.scratch.sampling.batch_sampling(&device_ctx)?;
            pegainfer_sample::gpu_sample_batch_into(
                &device_ctx,
                decode_arena.logits.as_ref(),
                &sampling_rows,
                seed,
                scratch,
            )
            .with_context(|| format!("Kimi rank {rank} batched decode sampling"))?
        };

        let local_top1 = read_local_top1_batch_values(
            &device_ctx,
            &decode_arena.logits,
            active_len,
            &mut decode_arena.scratch.sampling.top1_value_scratch,
            &mut decode_arena.scratch.sampling.top1_out,
        )?;
        let mut picks: Vec<(u32, f32)> = local_top1;
        for (sampling_row, token) in sampling_rows.iter().zip(&sampled) {
            // Keep the argmax logit as the reported top logit; the pick itself
            // is the sampled token.
            picks[sampling_row.row].0 = *token;
        }

        let host_logits = if rows.iter().any(|r| r.logprobs.is_some()) {
            ensure!(
                cache.vocab_start == 0 && cache.vocab_rows == KIMI_K2_VOCAB,
                "Kimi logprobs require an unsharded vocab (TP1); a vocab shard's \
                 logsumexp is not the global one (#236)"
            );
            Some(
                device_ctx
                    .stream
                    .clone_dtoh(&decode_arena.logits.data)
                    .with_context(|| format!("Kimi rank {rank} D2H decode logits for logprobs"))?,
            )
        } else {
            None
        };
        let mut reports = Vec::with_capacity(active_len);
        for (row, (local_next, local_top_logit_f32)) in picks.into_iter().enumerate() {
            let logprob = match (&host_logits, rows[row].logprobs) {
                (Some(host), Some(top_k)) => pegainfer_sample::token_logprob_from_row(
                    &host[row * cache.vocab_rows..(row + 1) * cache.vocab_rows],
                    local_next,
                    top_k,
                ),
                _ => None,
            };
            reports.push(KimiOneTokenForwardReport {
                rank,
                batch_slot: slots[row],
                input_token_id: token_ids[row],
                local_next_token_id: local_next,
                local_next_token_global_id: cache.vocab_start as u32 + local_next,
                local_top_logit_f32,
                vocab_start: cache.vocab_start,
                vocab_rows: cache.vocab_rows,
                dense_layers_executed: KIMI_K2_DENSE_LAYERS,
                moe_layers_executed: KIMI_K2_MOE_LAYERS,
                logprob,
            });
        }
        Ok(reports)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_prompt_next_token_inner(
        &mut self,
        slot: usize,
        decode_batch_size: usize,
        input_ids: &[u32],
        cached_tokens: usize,
        ep_max_seq_len: usize,
        kv_pages: &KimiKvStepPages,
        row: KimiRowOptions,
        seed: u64,
    ) -> Result<KimiOneTokenForwardReport> {
        ensure!(!input_ids.is_empty(), "Kimi prompt forward requires tokens");
        self.ctx.set_current()?;
        let loaded = self
            .loaded
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Kimi rank weights must be loaded before forward"))?;
        let tp_comm_ref = self.tp_comm.as_ref().map(super::OwnedRankComm::get);
        let device_ctx = self.ctx.as_device_context();
        let KimiRankLoadedWeights {
            gpu,
            expert_kernels,
            one_token_cache: cache,
            kv_pool,
            decode_arenas,
        } = loaded;
        let rank = gpu.rank;
        let seq_len = input_ids.len();
        let input_token_id = *input_ids
            .last()
            .ok_or_else(|| anyhow::anyhow!("Kimi prompt ids unexpectedly empty"))?;
        let decode_arena = decode_arenas.get_mut(&device_ctx, decode_batch_size)?;
        let slot_pages_start = decode_arena
            .configure_slot_prefill(&device_ctx, slot, seq_len, cached_tokens, kv_pages)
            .with_context(|| {
                format!("Kimi rank {rank} configure slot {slot} prefill KV page table")
            })?;

        let mut hidden = GpuTensor::<KIMI_K2_HIDDEN>::zeros(&device_ctx, seq_len)?;
        let token_ids = device_ctx.stream.clone_htod(input_ids)?;
        typed_ops::embedding_vocab_shard_into(
            &device_ctx,
            &cache.token_embedding,
            &token_ids,
            &mut hidden,
            cache.vocab_start as u32,
        )?;
        self.collective_barrier.wait();
        if let Some(comm) = tp_comm_ref {
            device_ctx
                .sync()
                .with_context(|| format!("Kimi rank {} sync before first TP all-reduce", rank))?;
            comm.all_reduce_in_place(&mut hidden.data, &ReduceOp::Sum)
                .map_err(|err| {
                    anyhow::anyhow!("Kimi TP all-reduce bf16 hidden failed: status={:?}", err.0)
                })?;
        }

        // RoPE table must cover absolute positions up to cached + seq_len - 1:
        // the suffix rotates at positions cached_tokens.. on a cache hit.
        let (cos_host, sin_host) = build_yarn_rope_cache(cached_tokens + seq_len);
        let cos = device_ctx.stream.clone_htod(&cos_host)?;
        let sin = device_ctx.stream.clone_htod(&sin_host)?;
        let mut normed = GpuTensor::<KIMI_K2_HIDDEN>::zeros(&device_ctx, seq_len)?;
        let mut next_hidden = GpuTensor::<KIMI_K2_HIDDEN>::zeros(&device_ctx, seq_len)?;
        let decode_aux_ctx = DeviceContext {
            ctx: Arc::clone(&self.decode_aux_ctx.ctx),
            stream: Arc::clone(&self.decode_aux_ctx.stream),
            device_ordinal: self.decode_aux_ctx.device_ordinal,
        };
        let mut deepep_prefill = if tp_comm_ref.is_none() && ep_max_seq_len > 0 {
            Some(
                crate::runner::moe_deepep::KimiMoeDeepEpPrefill::new(&device_ctx, ep_max_seq_len)
                    .with_context(|| {
                    format!(
                        "Kimi rank {rank} DeepEP prefill buffers (ep_max_seq_len={ep_max_seq_len})"
                    )
                })?,
            )
        } else {
            None
        };

        let mut dense_layers_executed = 0usize;
        let mut moe_layers_executed = 0usize;
        for layer in &cache.layers {
            Self::forward_mla_prefill(
                &device_ctx,
                tp_comm_ref,
                layer.layer_idx,
                &layer.attention,
                &cos,
                &sin,
                kv_pool,
                decode_arena,
                &mut hidden,
                &mut normed,
                &mut next_hidden,
                cached_tokens,
                slot_pages_start,
                self.local_dims.local_heads,
            )
            .with_context(|| format!("Kimi MLA prefill layer {}", layer.layer_idx))?;
            match &layer.kind {
                KimiLayerForwardKindCache::Dense(dense) => {
                    Self::forward_dense_mlp(
                        &device_ctx,
                        tp_comm_ref,
                        dense,
                        &layer.attention.post_attention_norm,
                        &mut hidden,
                        &mut normed,
                        &mut next_hidden,
                    )
                    .with_context(|| format!("Kimi dense MLP layer {}", layer.layer_idx))?;
                    dense_layers_executed += 1;
                }
                KimiLayerForwardKindCache::Moe(moe) => {
                    if let Some(prefill) = deepep_prefill.as_mut() {
                        let ep = self
                            .deepep
                            .as_ref()
                            .ok_or_else(|| {
                                anyhow::anyhow!("Kimi rank {rank} TP1 prefill requires DeepEP")
                            })?
                            .ep();
                        crate::runner::moe_deepep::forward_moe_layer_prefill_deepep(
                            &device_ctx,
                            &decode_aux_ctx,
                            ep,
                            layer.layer_idx,
                            moe,
                            &layer.attention.post_attention_norm,
                            expert_kernels,
                            &mut hidden,
                            &mut normed,
                            &mut next_hidden,
                            prefill,
                        )
                        .with_context(|| {
                            format!("Kimi MoE DeepEP prefill layer {}", layer.layer_idx)
                        })?;
                    } else {
                        Self::forward_moe_layer(
                            &device_ctx,
                            tp_comm_ref,
                            layer.layer_idx,
                            moe,
                            &layer.attention.post_attention_norm,
                            expert_kernels,
                            &mut hidden,
                            &mut normed,
                            &mut next_hidden,
                        )
                        .with_context(|| format!("Kimi MoE layer {}", layer.layer_idx))?;
                    }
                    moe_layers_executed += 1;
                }
            }
        }

        typed_ops::rms_norm_into(
            &device_ctx,
            &hidden,
            &cache.final_norm,
            KIMI_K2_RMS_NORM_EPS,
            &mut normed,
        )?;
        let mut logits_hidden = HiddenStates::zeros(&device_ctx, cache.vocab_rows, seq_len)?;
        typed_ops::gemm_runtime_out_into(&device_ctx, &cache.lm_head, &normed, &mut logits_hidden)?;
        let logits_offset = (seq_len - 1) * cache.vocab_rows;
        let logits_last = logits_hidden
            .data
            .slice(logits_offset..logits_offset + cache.vocab_rows);
        let mut logits_data = device_ctx.stream.alloc_zeros(cache.vocab_rows)?;
        device_ctx
            .stream
            .memcpy_dtod(&logits_last, &mut logits_data)?;
        let logits = DeviceVec {
            data: logits_data,
            len: cache.vocab_rows,
        };
        let (mut local_next, local_top_logit_f32) =
            sample_local_top1_with_value(&device_ctx, &logits)?;
        if !row.sampling.is_greedy() {
            ensure!(
                cache.vocab_start == 0 && cache.vocab_rows == KIMI_K2_VOCAB,
                "Kimi sampling requires an unsharded vocab (TP1); a vocab shard \
                 cannot sample the global distribution (#237, #226)"
            );
            let sampling_rows = [pegainfer_sample::BatchSamplingRow {
                row: 0,
                temperature: row.sampling.temperature,
                top_k: row.sampling.top_k,
                top_p: row.sampling.top_p,
                min_p: row.sampling.min_p,
            }];
            let scratch = decode_arena.scratch.sampling.batch_sampling(&device_ctx)?;
            let sampled = pegainfer_sample::gpu_sample_batch_into(
                &device_ctx,
                pegainfer_kernels::tensor::HiddenStatesRef {
                    data: &logits.data,
                    hidden_dim: logits.len,
                    seq_len: 1,
                },
                &sampling_rows,
                seed,
                scratch,
            )
            .with_context(|| format!("Kimi rank {rank} prefill sampling"))?;
            local_next = sampled[0];
        }
        let logprob = if let Some(top_k) = row.logprobs {
            ensure!(
                cache.vocab_start == 0 && cache.vocab_rows == KIMI_K2_VOCAB,
                "Kimi logprobs require an unsharded vocab (TP1); a vocab \
                 shard's logsumexp is not the global one (#236)"
            );
            let host = device_ctx
                .stream
                .clone_dtoh(&logits.data)
                .with_context(|| format!("Kimi rank {rank} D2H prefill logits"))?;
            pegainfer_sample::token_logprob_from_row(&host, local_next, top_k)
        } else {
            None
        };

        let report = KimiOneTokenForwardReport {
            rank,
            batch_slot: slot,
            input_token_id,
            local_next_token_id: local_next,
            local_next_token_global_id: cache.vocab_start as u32 + local_next,
            local_top_logit_f32,
            vocab_start: cache.vocab_start,
            vocab_rows: cache.vocab_rows,
            dense_layers_executed,
            moe_layers_executed,
            logprob,
        };
        Ok(report)
    }

    /// One MLA prefill layer over the `seq_len`-token suffix. On a prefix
    /// cache hit (`cached_tokens > 0`) the cached latent is gathered from
    /// pool pages, decompressed through the same kv_b GEMM, and assembled
    /// into k/v rows `0..cached_tokens`; the suffix fills the rest and
    /// attends over all `cached_tokens + seq_len` rows.
    #[allow(clippy::too_many_arguments)]
    fn forward_mla_prefill(
        ctx: &DeviceContext,
        comm: Option<&Comm>,
        layer_idx: usize,
        attention: &KimiAttentionForwardCache,
        cos: &CudaSlice<half::bf16>,
        sin: &CudaSlice<half::bf16>,
        kv_pool: &mut KimiWorkerKvPool,
        decode_arena: &mut KimiWorkerDecodeArena,
        hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
        normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
        next_hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
        cached_tokens: usize,
        slot_pages_start: usize,
        local_heads: usize,
    ) -> Result<()> {
        let seq_len = hidden.seq_len;
        let kv_len = cached_tokens + seq_len;
        let q_proj_out = local_heads * KIMI_K2_MLA_Q_HEAD_DIM;
        let kv_b_out = attention.kv_b_proj.rows;
        pegainfer_kernels::typed_pipeline! {
            ctx = ctx, eps = KIMI_K2_RMS_NORM_EPS, seq_len = seq_len, gemm = prefill;
            tensor qkv_a: KIMI_K2_MLA_QKV_A_OUT;
            tensor q_a: KIMI_K2_Q_LORA_RANK;
            tensor q_a_normed: KIMI_K2_Q_LORA_RANK;
            tensor compressed_kv: KIMI_K2_MLA_KV_LORA_RANK;
            tensor k_rope: KIMI_K2_MLA_ROPE_DIM;
            tensor compressed_normed: KIMI_K2_MLA_KV_LORA_RANK;
            tensor append_kpe: KIMI_K2_MLA_ROPE_DIM;

            rms_norm(hidden => normed, attention.input_norm);
            gemm(normed => &mut qkv_a, attention.fused_qkv_a_proj);
            try kimi_mla_split_qkv_a(ctx, &qkv_a, &mut q_a, &mut compressed_kv, &mut k_rope);
            rms_norm(&q_a => &mut q_a_normed, attention.q_a_norm);
        }
        let mut q_proj = HiddenStates::zeros(ctx, q_proj_out, seq_len)?;
        typed_ops::gemm_dm_typed_to_hs(ctx, &attention.q_b_proj, &q_a_normed, &mut q_proj)?;
        typed_ops::rms_norm_into(
            ctx,
            &compressed_kv,
            &attention.kv_a_norm,
            KIMI_K2_RMS_NORM_EPS,
            &mut compressed_normed,
        )?;
        kimi_mla_rope_apply_kpe(
            ctx,
            &k_rope,
            cos,
            sin,
            &decode_arena.positions_d,
            &mut append_kpe,
        )?;
        decode_arena.append_prefill_layer_kv(
            ctx,
            kv_pool,
            layer_idx,
            &compressed_normed,
            &append_kpe,
        )?;
        let mut kv_b = HiddenStates::zeros(ctx, kv_b_out, seq_len)?;
        typed_ops::gemm_dm_typed_to_hs(ctx, &attention.kv_b_proj, &compressed_normed, &mut kv_b)?;

        let mut k_cache = ctx
            .stream
            .alloc_zeros(kv_len * local_heads * KIMI_K2_MLA_Q_HEAD_DIM)?;
        let mut v_cache = ctx
            .stream
            .alloc_zeros(kv_len * local_heads * KIMI_K2_MLA_V_HEAD_DIM)?;
        if cached_tokens > 0 {
            let layer_cache = kv_pool.layer_mut(layer_idx)?;
            let mut ckv_gathered =
                GpuTensor::<KIMI_K2_MLA_KV_LORA_RANK>::zeros(ctx, cached_tokens)?;
            kimi_mla_gather_cached_ckv_rt(
                ctx,
                &layer_cache.ckv_cache,
                &decode_arena.page_indices_d,
                slot_pages_start,
                &decode_arena.layout,
                &mut ckv_gathered,
            )?;
            let mut kv_b_cached = HiddenStates::zeros(ctx, kv_b_out, cached_tokens)?;
            typed_ops::gemm_dm_typed_to_hs(
                ctx,
                &attention.kv_b_proj,
                &ckv_gathered,
                &mut kv_b_cached,
            )?;
            kimi_mla_assemble_cached_kv_rt(
                ctx,
                &kv_b_cached,
                &layer_cache.kpe_cache,
                &decode_arena.page_indices_d,
                slot_pages_start,
                &decode_arena.layout,
                &mut k_cache,
                &mut v_cache,
                kv_len,
                local_heads,
            )?;
        }
        let mut q_attn = HiddenStates::zeros(ctx, q_proj_out, seq_len)?;
        kimi_mla_rope_assemble_prefill_rt(
            ctx,
            &q_proj,
            &k_rope,
            &kv_b,
            cos,
            sin,
            &mut q_attn,
            &mut k_cache,
            &mut v_cache,
            cached_tokens,
            local_heads,
        )?;

        let o_proj_in = local_heads * KIMI_K2_MLA_V_HEAD_DIM;
        let mut attn_out = HiddenStates::zeros(ctx, o_proj_in, seq_len)?;
        kimi_flashinfer_single_prefill_mla_rt(
            ctx,
            &q_attn,
            &k_cache,
            &v_cache,
            &mut attn_out,
            kimi_mla_softmax_scale(),
            kv_len,
            local_heads,
        )?;
        let mut projected = GpuTensor::<KIMI_K2_HIDDEN>::zeros(ctx, seq_len)?;
        typed_ops::gemm_dm_hs_to_typed(ctx, &attention.o_proj, &attn_out, &mut projected)?;
        if let Some(comm) = comm {
            comm.all_reduce_in_place(&mut projected.data, &ReduceOp::Sum)
                .map_err(|err| {
                    anyhow::anyhow!("Kimi TP all-reduce bf16 hidden failed: status={:?}", err.0)
                })?;
        }
        typed_ops::add_into(ctx, hidden, &projected, next_hidden)?;
        std::mem::swap(hidden, next_hidden);
        Ok(())
    }

    fn forward_dense_mlp(
        ctx: &DeviceContext,
        comm: Option<&Comm>,
        dense: &KimiDenseForwardCache,
        post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
        hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
        normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
        next_hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    ) -> Result<()> {
        forward_dense_mlp_batch_into(
            ctx,
            comm,
            dense,
            post_attention_norm,
            hidden,
            normed,
            next_hidden,
        )
    }

    fn forward_moe_layer(
        ctx: &DeviceContext,
        comm: Option<&Comm>,
        layer_idx: usize,
        moe: &KimiMoeForwardCache,
        post_attention_norm: &NormWeight<KIMI_K2_HIDDEN>,
        expert_kernels: &KimiRankExpertMarlinWeights,
        hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
        normed: &mut GpuTensor<KIMI_K2_HIDDEN>,
        next_hidden: &mut GpuTensor<KIMI_K2_HIDDEN>,
    ) -> Result<()> {
        crate::runner::moe_nccl::forward_moe_layer_batch_into(
            ctx,
            comm,
            layer_idx,
            moe,
            post_attention_norm,
            expert_kernels,
            hidden,
            normed,
            next_hidden,
        )
    }
}
