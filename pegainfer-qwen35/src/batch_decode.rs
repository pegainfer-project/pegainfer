//! Batched decode for Qwen3.5: N requests, 1 token each, shared full-attn kernels
//! and per-request recurrent-state updates for linear attention.

use anyhow::Context;
use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use pegainfer_core::kv_pool::KvLayout;
use pegainfer_core::kv_pool::KvState;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::sampler::SamplingParams;

use super::batch_decode_graph::BATCH_BUCKETS;
use super::batch_decode_graph::BatchDecodeGraphState;
use super::batch_decode_graph::bucket_for;
use super::decode_buffers::BatchDecodeBuffers35;
use super::recurrent_state::LinearStatePointerTables;
use super::recurrent_state::RecurrentState;
use super::weights::FullAttentionLayer;
use super::weights::LayerKind;
use super::weights::LinearAttentionLayer;
use super::weights::Qwen35Model;
use super::weights::TransformerBlock35;
use crate::ops;

static LOG_UNCOMPILED_DECODE_ROUTE: std::sync::Once = std::sync::Once::new();

impl Qwen35Model {
    pub(crate) fn select_tokens_from_logits_varied(
        &self,
        logits: &HiddenStates,
        bufs: &mut BatchDecodeBuffers35,
        params: &[&SamplingParams],
        sample_seed: u64,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(
            params.len() == logits.seq_len,
            "Qwen3.5 sampling params/logits row mismatch: params={}, logits_rows={}",
            params.len(),
            logits.seq_len
        );
        anyhow::ensure!(
            params.len() <= bufs.max_batch_size,
            "Qwen3.5 sampling batch {} exceeds scratch capacity {}",
            params.len(),
            bufs.max_batch_size
        );
        bufs.steps.clear();
        bufs.steps.resize(params.len(), 0);
        pegainfer_sample::select_batch(
            &self.ctx,
            logits,
            params,
            &bufs.steps,
            sample_seed,
            &mut bufs.sample,
        )
    }

    pub(crate) fn select_tokens_batch_varied(
        &self,
        bufs: &mut BatchDecodeBuffers35,
        params: &[&SamplingParams],
        sample_seed: u64,
    ) -> Result<Vec<u32>> {
        anyhow::ensure!(
            !params.is_empty(),
            "Qwen3.5 decode sampling requires at least one request"
        );
        anyhow::ensure!(
            params.len() <= bufs.logits.seq_len,
            "Qwen3.5 decode sampling params/logits row mismatch: params={}, logits_rows={}",
            params.len(),
            bufs.logits.seq_len
        );
        anyhow::ensure!(
            params.len() <= bufs.max_batch_size,
            "Qwen3.5 decode sampling batch {} exceeds scratch capacity {}",
            params.len(),
            bufs.max_batch_size
        );
        bufs.steps.clear();
        bufs.steps.resize(params.len(), 0);
        pegainfer_sample::select_batch(
            &self.ctx,
            &bufs.logits,
            params,
            &bufs.steps,
            sample_seed,
            &mut bufs.sample,
        )
    }

    fn batch_decode_full_attention(
        &self,
        attn: &FullAttentionLayer,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        layer_idx: usize,
        bs: usize,
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let tp = self.tensor_parallel;
        let num_attention_heads = self.config.local_num_attention_heads(tp);
        let num_key_value_heads = self.config.local_num_key_value_heads(tp);

        ops::gemm_into(&self.ctx, &attn.q_proj, &bufs.normed, &mut bufs.q_full);
        ops::gemm_into(&self.ctx, &attn.k_proj, &bufs.normed, &mut bufs.k_attn);
        ops::gemm_into(&self.ctx, &attn.v_proj, &bufs.normed, &mut bufs.v_attn);

        ops::qk_norm_partial_rope_batched_decode_hd256_into(
            &self.ctx,
            &bufs.q_full,
            &mut bufs.q_attn,
            &mut bufs.k_attn,
            &attn.q_norm,
            &attn.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.positions_d,
            num_attention_heads,
            num_key_value_heads,
            self.config.rotary_dim,
            eps,
        )?;

        ops::paged_attention_batch_decode_hd256_into(
            &self.ctx,
            &bufs.q_attn,
            &bufs.k_attn,
            &bufs.v_attn,
            kv_buffer,
            layout,
            layer_idx,
            &bufs.page_indices_d,
            &bufs.page_indptr_d,
            &bufs.last_page_len_d,
            &bufs.positions_d,
            &bufs.request_indices_d,
            &bufs.kv_tile_indices_d,
            &bufs.kv_chunk_size_d,
            &mut bufs.attn_out_full,
            num_attention_heads,
            bs,
        )?;

        unsafe {
            let (qf_ptr, _gqf) = bufs.q_full.data.device_ptr(&self.ctx.stream);
            let (out_ptr, _go) = bufs.attn_out_full.data.device_ptr_mut(&self.ctx.stream);
            crate::ffi::attention_gate_batch_hd256_cuda(
                qf_ptr as *const crate::ffi::Half,
                out_ptr as *mut crate::ffi::Half,
                num_attention_heads as i32,
                bs as i32,
                self.ctx.stream.cu_stream(),
            );
        }

        ops::gemm_into(
            &self.ctx,
            &attn.o_proj,
            &bufs.attn_out_full,
            &mut bufs.attn_results,
        );
        self.all_reduce_hidden(&mut bufs.attn_results)?;
        Ok(())
    }

    fn batch_decode_full_attention_via_prefill(
        &self,
        attn: &FullAttentionLayer,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        plan: &ops::PrefillPagedPlan,
        layer_idx: usize,
        bs: usize,
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        ops::gemm_into(&self.ctx, &attn.q_proj, &bufs.normed, &mut bufs.q_full);
        ops::gemm_into(&self.ctx, &attn.k_proj, &bufs.normed, &mut bufs.k_attn);
        ops::gemm_into(&self.ctx, &attn.v_proj, &bufs.normed, &mut bufs.v_attn);

        ops::qk_norm_partial_rope_batched_decode_hd256_into(
            &self.ctx,
            &bufs.q_full,
            &mut bufs.q_attn,
            &mut bufs.k_attn,
            &attn.q_norm,
            &attn.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &bufs.positions_d,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.rotary_dim,
            eps,
        )?;

        ops::paged_attention_batch_decode_via_prefill_hd256_into(
            &self.ctx,
            &bufs.q_attn,
            &bufs.k_attn,
            &bufs.v_attn,
            kv_buffer,
            layout,
            layer_idx,
            plan,
            &mut bufs.attn_out_full,
            self.config.num_attention_heads,
        )?;

        unsafe {
            let (qf_ptr, _gqf) = bufs.q_full.data.device_ptr(&self.ctx.stream);
            let (out_ptr, _go) = bufs.attn_out_full.data.device_ptr_mut(&self.ctx.stream);
            crate::ffi::attention_gate_batch_hd256_cuda(
                qf_ptr as *const crate::ffi::Half,
                out_ptr as *mut crate::ffi::Half,
                self.config.num_attention_heads as i32,
                bs as i32,
                self.ctx.stream.cu_stream(),
            );
        }

        ops::gemm_into(
            &self.ctx,
            &attn.o_proj,
            &bufs.attn_out_full,
            &mut bufs.attn_results,
        );
        self.all_reduce_hidden(&mut bufs.attn_results)?;
        Ok(())
    }

    /// Eager batch decode step.
    ///
    /// Unlike `batch_decode_graph`, this does not pad to a CUDA Graph bucket and
    /// does not capture/replay. Recurrent state is supplied directly by the
    /// caller, which is the shape TP workers need for rank-local request state.
    pub(crate) fn batch_decode_eager_logits(
        &self,
        token_ids: &[u32],
        kv_states: &mut [&mut KvState],
        recurrent_states: &mut [&mut RecurrentState],
        linear_pointer_tables: &LinearStatePointerTables,
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        let bs = token_ids.len();
        anyhow::ensure!(
            bs > 0,
            "batch_decode_eager_logits requires at least one request"
        );
        anyhow::ensure!(bs == kv_states.len(), "token_ids / kv_states len mismatch");
        anyhow::ensure!(
            bs == recurrent_states.len(),
            "token_ids / recurrent_states len mismatch"
        );
        anyhow::ensure!(
            bs <= bufs.max_batch_size,
            "batch size {bs} exceeds eager decode buffer capacity {}",
            bufs.max_batch_size
        );
        linear_pointer_tables.validate_for(&self.config, bs, "Qwen3.5 eager decode")?;

        let mut positions = Vec::with_capacity(bs);
        for (i, kv) in kv_states.iter_mut().enumerate() {
            let pos = kv.seq_len();
            self.ensure_rope_cache_covers(pos + 1)?;
            kv.ensure_capacity(pos + 1)?;
            kv.advance(1);
            recurrent_states[i].seq_len += 1;
            positions.push(pos as i32);
        }

        bufs.set_batch_size(bs);
        self.ctx
            .stream
            .memcpy_htod(token_ids, &mut bufs.token_ids_d)?;
        self.ctx
            .stream
            .memcpy_htod(&positions, &mut bufs.positions_d)?;

        let kv_refs: Vec<&KvState> = kv_states.iter().map(|s| &**s).collect();
        bufs.sync_paged_meta(&self.ctx, &kv_refs, bs)?;

        let kv_buffer = kv_states[0].buffer();
        let layout = *kv_states[0].layout();
        self.batch_decode_kernels_graph(
            kv_buffer,
            &layout,
            bs,
            &linear_pointer_tables.state_ptrs,
            &linear_pointer_tables.conv_state_ptrs,
            bufs,
        )
    }

    // =========================================================================
    // CUDA Graph batch decode
    // =========================================================================

    /// CUDA Graph batch decode step.
    ///
    /// Pads the batch to the nearest bucket size, fills padding positions with
    /// dummy KV metadata (pointing to the reserved padding page), then
    /// captures or replays a per-bucket CUDA Graph for the full kernel sequence.
    ///
    /// Recurrent state is owned by `graph_state.slot_states`: the caller must
    /// pack active requests into positions 0..batch_size before calling. After
    /// the call, `slot_states[i]` contains the updated state for request i.
    pub(crate) fn batch_decode_graph(
        &self,
        token_ids: &[u32],
        kv_states: &mut [&mut KvState],
        graph_state: &mut BatchDecodeGraphState,
    ) -> Result<()> {
        let bs = token_ids.len();
        anyhow::ensure!(bs > 0, "batch_decode_graph requires at least one request");
        anyhow::ensure!(bs == kv_states.len(), "token_ids / kv_states len mismatch");
        anyhow::ensure!(
            bs <= graph_state.slot_states.len(),
            "batch size {bs} exceeds decode capacity {}",
            graph_state.slot_states.len()
        );

        if !self.config.decode_group_is_compiled() {
            LOG_UNCOMPILED_DECODE_ROUTE.call_once(|| {
                let group = self.config.num_attention_heads / self.config.num_key_value_heads;
                log::info!(
                    "Qwen3.5 decode GQA group {group} ({} q heads / {} kv heads) has no compiled BatchDecode kernel; batched hybrid eager fallback active, bs_capacity={}",
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    graph_state.buffers.max_batch_size,
                );
            });
            // Paged-prefill attention stays eager; verify_graph records that
            // captured prefill attention under-reads growing decode KV.
            return self.batch_decode_batched_hybrid(token_ids, kv_states, graph_state);
        }

        let padded_bs = bucket_for(bs);
        graph_state.linear_pointer_tables.validate_for(
            &self.config,
            padded_bs,
            "Qwen3.5 graph decode",
        )?;

        // Advance KV states and collect positions. Slot seq_len is incremented
        // on the CPU outside the graph so it never appears inside the capture.
        let mut positions = Vec::with_capacity(bs);
        for (i, kv) in kv_states.iter_mut().enumerate() {
            let pos = kv.seq_len();
            self.ensure_rope_cache_covers(pos + 1)?;
            kv.ensure_capacity(pos + 1)?;
            kv.advance(1);
            graph_state.slot_states[i].seq_len += 1;
            positions.push(pos as i32);
        }

        graph_state.buffers.set_batch_size(padded_bs);

        // H2D: token_ids and positions — zero-padded to bucket size.
        let mut token_ids_padded = token_ids.to_vec();
        token_ids_padded.resize(padded_bs, 0);
        positions.resize(padded_bs, 0);
        self.ctx
            .stream
            .memcpy_htod(&token_ids_padded, &mut graph_state.buffers.token_ids_d)?;
        self.ctx
            .stream
            .memcpy_htod(&positions, &mut graph_state.buffers.positions_d)?;

        // H2D: paged KV metadata with padding slots pointing to padding_page_id.
        let kv_refs: Vec<&KvState> = kv_states.iter().map(|s| &**s).collect();
        graph_state
            .buffers
            .sync_paged_meta(&self.ctx, &kv_refs, padded_bs)?;

        let kv_buffer = kv_states[0].buffer();
        let layout = *kv_states[0].layout();
        let bucket_idx = BATCH_BUCKETS.iter().position(|&b| b == padded_bs).unwrap();

        // Take graphs out of graph_state to avoid split-borrow in the closure.
        let mut graphs = std::mem::take(&mut graph_state.graphs);
        let linear_state_ptrs = &graph_state.linear_pointer_tables.state_ptrs;
        let linear_conv_state_ptrs = &graph_state.linear_pointer_tables.conv_state_ptrs;
        let result = graphs[bucket_idx].run_or_capture(&self.ctx, || {
            self.batch_decode_kernels_graph(
                kv_buffer,
                &layout,
                padded_bs,
                linear_state_ptrs,
                linear_conv_state_ptrs,
                &mut graph_state.buffers,
            )
        });
        graph_state.graphs = graphs;
        result
    }

    fn batch_decode_batched_hybrid(
        &self,
        token_ids: &[u32],
        kv_states: &mut [&mut KvState],
        graph_state: &mut BatchDecodeGraphState,
    ) -> Result<()> {
        let bs = token_ids.len();
        graph_state.linear_pointer_tables.validate_for(
            &self.config,
            bs,
            "Qwen3.5 hybrid decode",
        )?;
        let mut positions_i32 = Vec::with_capacity(bs);
        let mut start_positions = Vec::with_capacity(bs);
        for (i, kv) in kv_states.iter_mut().enumerate() {
            let pos = kv.seq_len();
            self.ensure_rope_cache_covers(pos + 1)
                .with_context(|| format!("hybrid decode rope cache pos={} slot={i}", pos + 1))?;
            kv.ensure_capacity(pos + 1)
                .with_context(|| format!("hybrid decode KV capacity pos={} slot={i}", pos + 1))?;
            kv.advance(1);
            graph_state.slot_states[i].seq_len += 1;
            positions_i32.push(pos as i32);
            start_positions.push(pos);
        }

        let bufs = &mut graph_state.buffers;
        bufs.set_batch_size(bs);
        self.ctx
            .stream
            .memcpy_htod(token_ids, &mut bufs.token_ids_d)
            .map_err(|e| {
                anyhow::anyhow!(
                    "hybrid decode H2D token_ids bs={bs}, cap={}: {e}",
                    bufs.max_batch_size
                )
            })?;
        self.ctx
            .stream
            .memcpy_htod(&positions_i32, &mut bufs.positions_d)
            .map_err(|e| {
                anyhow::anyhow!(
                    "hybrid decode H2D positions bs={bs}, cap={}: {e}",
                    bufs.max_batch_size
                )
            })?;

        let page_indices: Vec<Vec<i32>> =
            kv_states.iter().map(|kv| kv.page_indices_i32()).collect();
        let last_page_lens: Vec<usize> = kv_states.iter().map(|kv| kv.last_page_len()).collect();
        let seq_lens = vec![1usize; bs];
        // cta_tile_q 0 = the kernel's own FA2 derivation; the hd256 FFI takes no override.
        let plan = ops::PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
            &self.ctx,
            &page_indices,
            &last_page_lens,
            &start_positions,
            &seq_lens,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            0,
        )
        .with_context(|| {
            format!(
                "hybrid decode build PrefillPagedPlan bs={bs}, pages={}, heads={}/{}, head_dim={}",
                page_indices.iter().map(Vec::len).sum::<usize>(),
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim
            )
        })?;

        let kv_buffer = kv_states[0].buffer();
        let layout = *kv_states[0].layout();
        anyhow::ensure!(
            layout.num_kv_heads == self.config.num_key_value_heads
                && layout.head_dim == self.config.head_dim,
            "hybrid decode KV layout mismatch bs={bs}: layout kv_heads={}, head_dim={}; config kv_heads={}, head_dim={}",
            layout.num_kv_heads,
            layout.head_dim,
            self.config.num_key_value_heads,
            self.config.head_dim
        );
        self.batch_decode_batched_hybrid_kernels(
            kv_buffer,
            &layout,
            &plan,
            bs,
            &graph_state.linear_pointer_tables.state_ptrs,
            &graph_state.linear_pointer_tables.conv_state_ptrs,
            bufs,
        )
    }

    fn batch_decode_kernels_graph(
        &self,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        padded_bs: usize,
        linear_state_ptrs: &[CudaSlice<u64>],
        linear_conv_state_ptrs: &[CudaSlice<u64>],
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.token_ids_d,
            &mut bufs.hidden,
        )?;

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        for layer in &self.layers {
            ops::rms_norm_batch_offset_into(
                &self.ctx,
                &bufs.hidden,
                &layer.input_layernorm,
                eps,
                &mut bufs.normed,
            )?;

            match &layer.attn {
                LayerKind::FullAttention(attn) => {
                    self.batch_decode_full_attention(
                        attn, kv_buffer, layout, full_idx, padded_bs, bufs,
                    )?;
                    full_idx += 1;
                }
                LayerKind::LinearAttention(attn) => {
                    self.batch_decode_linear_attention_slots(
                        attn,
                        &linear_state_ptrs[linear_idx],
                        &linear_conv_state_ptrs[linear_idx],
                        padded_bs,
                        bufs,
                    );
                    linear_idx += 1;
                }
            }

            ops::add_batch_into(
                &self.ctx,
                &bufs.hidden,
                &bufs.attn_results,
                &mut bufs.hidden_mid,
            )?;

            ops::rms_norm_batch_offset_into(
                &self.ctx,
                &bufs.hidden_mid,
                &layer.post_attention_layernorm,
                eps,
                &mut bufs.normed,
            )?;

            ops::gemm_into(
                &self.ctx,
                &layer.mlp.gate_up_proj,
                &bufs.normed,
                &mut bufs.gate_up_out,
            );
            ops::silu_mul_fused_batch_into(&self.ctx, &bufs.gate_up_out, &mut bufs.act_out)?;
            ops::gemm_into(
                &self.ctx,
                &layer.mlp.down_proj,
                &bufs.act_out,
                &mut bufs.mlp_out,
            );
            self.all_reduce_hidden(&mut bufs.mlp_out)?;

            ops::add_batch_into(&self.ctx, &bufs.hidden_mid, &bufs.mlp_out, &mut bufs.hidden)?;
        }

        ops::rms_norm_batch_offset_into(
            &self.ctx,
            &bufs.hidden,
            &self.norm,
            eps,
            &mut bufs.normed,
        )?;
        ops::gemm_rows_into_checked(
            &self.ctx,
            self.output_projection(),
            0,
            self.config.selection_vocab,
            &bufs.normed,
            &mut bufs.logits,
        )?;
        debug_assert_eq!(bufs.logits.seq_len, padded_bs);

        Ok(())
    }

    fn batch_decode_batched_hybrid_kernels(
        &self,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        plan: &ops::PrefillPagedPlan,
        bs: usize,
        linear_state_ptrs: &[CudaSlice<u64>],
        linear_conv_state_ptrs: &[CudaSlice<u64>],
        bufs: &mut BatchDecodeBuffers35,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;

        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.token_ids_d,
            &mut bufs.hidden,
        )?;

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            ops::rms_norm_batch_offset_into(
                &self.ctx,
                &bufs.hidden,
                &layer.input_layernorm,
                eps,
                &mut bufs.normed,
            )?;

            match &layer.attn {
                LayerKind::FullAttention(attn) => {
                    self.batch_decode_full_attention_via_prefill(
                        attn, kv_buffer, layout, plan, full_idx, bs, bufs,
                    )
                    .with_context(|| {
                        format!("hybrid decode full-attn layer_idx={layer_idx}, full_idx={full_idx}, bs={bs}")
                    })?;
                    full_idx += 1;
                }
                LayerKind::LinearAttention(attn) => {
                    self.batch_decode_linear_attention_slots(
                        attn,
                        &linear_state_ptrs[linear_idx],
                        &linear_conv_state_ptrs[linear_idx],
                        bs,
                        bufs,
                    );
                    linear_idx += 1;
                }
            }

            self.batch_decode_mlp_tail(layer, bufs, eps)
                .with_context(|| {
                    format!("hybrid decode MLP tail layer_idx={layer_idx}, bs={bs}")
                })?;
        }

        ops::rms_norm_batch_offset_into(
            &self.ctx,
            &bufs.hidden,
            &self.norm,
            eps,
            &mut bufs.normed,
        )?;
        ops::gemm_rows_into_checked(
            &self.ctx,
            self.output_projection(),
            0,
            self.config.selection_vocab,
            &bufs.normed,
            &mut bufs.logits,
        )?;
        debug_assert_eq!(bufs.logits.seq_len, bs);
        Ok(())
    }

    fn batch_decode_mlp_tail(
        &self,
        layer: &TransformerBlock35,
        bufs: &mut BatchDecodeBuffers35,
        eps: f32,
    ) -> Result<()> {
        ops::add_batch_into(
            &self.ctx,
            &bufs.hidden,
            &bufs.attn_results,
            &mut bufs.hidden_mid,
        )?;

        ops::rms_norm_batch_offset_into(
            &self.ctx,
            &bufs.hidden_mid,
            &layer.post_attention_layernorm,
            eps,
            &mut bufs.normed,
        )?;

        ops::gemm_into(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            &bufs.normed,
            &mut bufs.gate_up_out,
        );
        ops::silu_mul_fused_batch_into(&self.ctx, &bufs.gate_up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.mlp_out,
        );

        ops::add_batch_into(&self.ctx, &bufs.hidden_mid, &bufs.mlp_out, &mut bufs.hidden)
    }

    /// Linear attention decode over slot-indexed recurrent state.
    ///
    /// Iterates 0..`padded_bs`. Real requests are in 0..real_bs; padding slots
    /// (real_bs..padded_bs) run but their output columns are ignored by the caller.
    /// All GPU addresses are stable per slot index, making this CUDA Graph safe.
    fn batch_decode_linear_attention_slots(
        &self,
        attn: &LinearAttentionLayer,
        state_ptrs: &CudaSlice<u64>,
        conv_state_ptrs: &CudaSlice<u64>,
        padded_bs: usize,
        bufs: &mut BatchDecodeBuffers35,
    ) {
        ops::gemm_into(&self.ctx, &attn.in_proj_qkv, &bufs.normed, &mut bufs.qkv);
        ops::gemm_into(&self.ctx, &attn.in_proj_z, &bufs.normed, &mut bufs.z);
        ops::gemm_into(&self.ctx, &attn.in_proj_b, &bufs.normed, &mut bufs.b_proj);
        ops::gemm_into(&self.ctx, &attn.in_proj_a, &bufs.normed, &mut bufs.a_proj);

        ops::conv1d_decode_batch_into(
            &self.ctx,
            &bufs.qkv,
            &attn.conv1d_weight,
            conv_state_ptrs,
            &mut bufs.qkv_conv,
            self.config.linear_conv_kernel_dim,
        );
        ops::gated_delta_rule_decode_batch_into(
            &self.ctx,
            &bufs.qkv_conv,
            &bufs.b_proj,
            &bufs.a_proj,
            &attn.dt_bias,
            &attn.a_log,
            state_ptrs,
            &mut bufs.gdr_out,
            padded_bs,
            self.config.linear_num_key_heads,
            self.config.linear_num_value_heads,
            self.config.linear_key_head_dim,
            self.config.linear_value_head_dim,
        );

        ops::rms_norm_gated_batch_into(
            &self.ctx,
            &bufs.gdr_out,
            &attn.norm_weight,
            &bufs.z,
            &mut bufs.normed_gated,
            self.config.linear_num_value_heads,
            self.config.linear_value_head_dim,
            self.config.rms_norm_eps,
        );
        ops::gemm_into(
            &self.ctx,
            &attn.out_proj,
            &bufs.normed_gated,
            &mut bufs.attn_results,
        );
    }
}
