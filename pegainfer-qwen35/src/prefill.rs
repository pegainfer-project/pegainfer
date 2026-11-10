use anyhow::Context;
use anyhow::Result;

/// Sequence length used for conservative prefill scratch reservation.
///
/// This is not an admission cap. Actual prompt admission is governed by the
/// paged KV pool, RoPE cache coverage, and allocation success. Prompts longer
/// than this are handled by chunking prefill at `PREFILL_CHUNK_LEN` rather than
/// being rejected (see `prefill_last_hidden`).
pub(crate) const SCRATCH_ESTIMATE_SEQ: usize = 20_000;

/// Maximum number of tokens processed in a single prefill forward pass.
///
/// Prefill is chunked at this granularity so the per-pass GDR scratch
/// (`GdrChunkwiseScratch35`, which scales linearly with the pass length) never
/// exceeds the memory reserved at startup. Kept equal to `SCRATCH_ESTIMATE_SEQ`
/// so the reservation in `weights.rs` covers exactly one chunk.
pub(crate) const PREFILL_CHUNK_LEN: usize = SCRATCH_ESTIMATE_SEQ;

use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::GdrChunkwiseScratch35;
use pegainfer_kernels::ops::attention_prefill_hd256_into;
use pegainfer_kv_cache::KvBuffer;
use pegainfer_kv_cache::KvView;

use super::flashinfer_gdn::FlashInferGdnChunkResources;
use super::prefill_buffers::gdr_chunkwise_scratch;
use super::recurrent_state::RecurrentState;
use super::weights::FullAttentionLayer;
use super::weights::LayerKind;
use super::weights::LinearAttentionLayer;
use super::weights::Qwen35Model;
use super::weights::TransformerBlock35;
use crate::ops;
use crate::ops::PrefillPagedPlan;

enum GdnPrefillChunkScratch {
    Triton(Box<GdrChunkwiseScratch35>),
    FlashInfer(Box<FlashInferGdnChunkResources>),
}

struct PrefillKvAccess<'a> {
    buffer: &'a cudarc::driver::CudaSlice<half::bf16>,
    layout: pegainfer_core::kv_pool::KvLayout,
    plan: &'a PrefillPagedPlan,
    base_pos: usize,
}

fn checked_prefill_end_pos(
    base_pos: usize,
    seq_len: usize,
    max_position_embeddings: usize,
) -> Result<usize> {
    let end_pos = base_pos.checked_add(seq_len).ok_or_else(|| {
        anyhow::anyhow!("Qwen3.5 prefill position overflow: base_pos={base_pos}, seq_len={seq_len}")
    })?;
    anyhow::ensure!(
        end_pos <= max_position_embeddings,
        "Qwen3.5 prefill requested end_pos={end_pos}, beyond max_position_embeddings={max_position_embeddings}"
    );
    Ok(end_pos)
}

impl Qwen35Model {
    pub(super) fn prefill_last_hidden(
        &self,
        token_ids: &[u32],
        full_view: &KvView,
        kv_buffer: &KvBuffer,
        recurrent: &mut RecurrentState,
    ) -> Result<DeviceVec> {
        anyhow::ensure!(
            !token_ids.is_empty(),
            "Qwen3.5 prefill requires at least one token"
        );
        // Validate the full target range up front (position overflow + RoPE cache
        // coverage) so an out-of-range prompt is rejected before any chunk mutates
        // the KV / recurrent state, rather than failing partway through.
        let base_pos = recurrent.seq_len;
        let end_pos = checked_prefill_end_pos(
            base_pos,
            token_ids.len(),
            self.config.max_position_embeddings,
        )?;
        anyhow::ensure!(
            full_view.seq_len() == end_pos,
            "Qwen3.5 prefill view ends at {}, expected {end_pos}",
            full_view.seq_len()
        );
        self.ensure_rope_cache_covers(end_pos)?;

        // Run prefill in serial chunks of at most `PREFILL_CHUNK_LEN` tokens. Each
        // chunk advances the paged KV and linear-attention recurrent/conv state in
        // place, so the next chunk continues from the previous one. This caps the
        // per-pass GDR scratch (which grows with the pass length) at the budget
        // reserved at startup, so prompts longer than one chunk prefill without OOM.
        let page_size = kv_buffer.layout().page_size;
        let mut hidden_batch = None;
        let mut offset = 0usize;
        for chunk in token_ids.chunks(PREFILL_CHUNK_LEN) {
            // Free the previous chunk's hidden states before allocating the next
            // chunk's scratch so peak memory stays within one chunk's reservation.
            drop(hidden_batch.take());
            let (chunk_hidden, chunk_scratch) = self.prepare_prefill_chunk(chunk)?;
            let chunk_end = base_pos + offset + chunk.len();
            let page_count = chunk_end.div_ceil(page_size);
            let chunk_view = KvView::new(
                full_view.page_indices()[..page_count].to_vec(),
                chunk_end,
                page_size,
            );
            let page_indices = vec![chunk_view.page_indices().to_vec()];
            let plan = PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
                &self.ctx,
                &page_indices,
                &[chunk_view.last_page_len()],
                &[base_pos + offset],
                &[chunk.len()],
                self.geometry.local_num_attention_heads(),
                self.geometry.local_num_key_value_heads(),
                self.config.head_dim,
                0,
            )?;
            let access = PrefillKvAccess {
                buffer: kv_buffer.buffer(),
                layout: pegainfer_core::kv_pool::KvLayout::new(
                    kv_buffer.layout().num_layers,
                    kv_buffer.layout().num_kv_heads,
                    kv_buffer.layout().head_dim,
                    page_size,
                )?,
                plan: &plan,
                base_pos: base_pos + offset,
            };
            hidden_batch = Some(self.prefill_chunk_forward(
                chunk_hidden,
                chunk_scratch,
                &access,
                recurrent,
            )?);
            offset += chunk.len();
        }
        // `seq_len > 0` guarantees at least one chunk produced hidden states.
        let hidden_batch = hidden_batch.expect("non-empty prefill produced no chunk");

        // Last-token logic runs once, on the final chunk's output.
        ops::extract_vec(&self.ctx, &hidden_batch, hidden_batch.seq_len - 1)
    }

    pub(super) fn batch_last_hidden_logits(
        &self,
        last_hiddens: &[DeviceVec],
    ) -> Result<HiddenStates> {
        let n = last_hiddens.len();
        anyhow::ensure!(n > 0, "batch_last_hidden_logits requires at least one row");
        let hidden_dim = self.config.hidden_size;

        let mut batched = HiddenStates::zeros(&self.ctx, hidden_dim, n)?;
        for (request_idx, last_hidden) in last_hiddens.iter().enumerate() {
            anyhow::ensure!(
                last_hidden.len == hidden_dim,
                "Qwen3.5 last hidden row {request_idx} has len {}, expected {hidden_dim}",
                last_hidden.len
            );
            ops::write_vec_into(&self.ctx, last_hidden, &mut batched, request_idx)?;
        }

        let mut normed = HiddenStates::zeros(&self.ctx, hidden_dim, n)?;
        ops::rms_norm_batch_offset_into(
            &self.ctx,
            &batched,
            &self.norm,
            self.config.rms_norm_eps,
            &mut normed,
        )?;
        let mut logits = HiddenStates::zeros(&self.ctx, self.config.selection_vocab, n)?;
        self.output_logits_into(&normed, &mut logits)?;
        debug_assert_eq!(logits.seq_len, n);
        Ok(logits)
    }

    fn prepare_prefill_chunk(
        &self,
        token_ids: &[u32],
    ) -> Result<(HiddenStates, GdnPrefillChunkScratch)> {
        let seq_len = token_ids.len();
        anyhow::ensure!(
            seq_len > 0 && seq_len <= PREFILL_CHUNK_LEN,
            "prefill chunk length {seq_len} out of range 1..={PREFILL_CHUNK_LEN}"
        );
        let c = &self.config;

        // Embeddings for this chunk.
        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(token_ids)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {e}"))?;
        let mut hidden_batch = HiddenStates::zeros(&self.ctx, c.hidden_size, seq_len)?;
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &token_ids_gpu,
            &mut hidden_batch,
        )?;

        let gdn_scratch = match &self.flashinfer_gdn {
            None => GdnPrefillChunkScratch::Triton(Box::new(gdr_chunkwise_scratch(
                &self.ctx,
                c,
                self.geometry,
                seq_len,
            )?)),
            Some(backend) => GdnPrefillChunkScratch::FlashInfer(Box::new(
                FlashInferGdnChunkResources::new(&self.ctx, backend, seq_len)?,
            )),
        };
        Ok((hidden_batch, gdn_scratch))
    }

    fn prefill_chunk_forward(
        &self,
        mut hidden_batch: HiddenStates,
        mut gdn_scratch: GdnPrefillChunkScratch,
        kv: &PrefillKvAccess<'_>,
        recurrent: &mut RecurrentState,
    ) -> Result<HiddenStates> {
        let seq_len = hidden_batch.seq_len;
        // Process layers
        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_batch = self.prefill_layer(
                layer_idx,
                layer,
                &hidden_batch,
                &mut gdn_scratch,
                &mut linear_idx,
                &mut full_idx,
                kv,
                recurrent,
            )?;
        }

        // Advance recurrent token count for the next chunk / decode step; the
        // paged KV position is committed by the caller.
        recurrent.seq_len += seq_len;

        Ok(hidden_batch)
    }

    /// Process one layer during prefill. Returns updated hidden_batch.
    #[allow(clippy::too_many_arguments)]
    fn prefill_layer(
        &self,
        _layer_idx: usize,
        layer: &TransformerBlock35,
        hidden_batch: &HiddenStates,
        gdn_scratch: &mut GdnPrefillChunkScratch,
        linear_idx: &mut usize,
        full_idx: &mut usize,
        kv: &PrefillKvAccess<'_>,
        recurrent: &mut RecurrentState,
    ) -> Result<HiddenStates> {
        let c = &self.config;
        let eps = c.rms_norm_eps;
        let seq_len = hidden_batch.seq_len;

        // 1. Input layernorm — per-token (no batched offset norm kernel yet)
        // Use standard batched norm and add the offset correction manually
        // Actually we need the (1+w) variant. Process token by token for now.
        let mut normed_batch =
            self.batched_rms_norm_offset(hidden_batch, &layer.input_layernorm, eps)?;

        // 2. Attention / Linear attention — per-token for correctness
        let geom = self.geometry;
        let attn_out_dim = match &layer.attn {
            LayerKind::FullAttention(_) => geom.local_full_attn_q_dim(),
            LayerKind::LinearAttention(_) => geom.local_linear_z_dim(),
        };

        // Batch project, then per-token attention/recurrent
        let attn_results = match &layer.attn {
            LayerKind::FullAttention(attn) => self.prefill_full_attention(
                attn,
                &normed_batch,
                full_idx,
                kv,
                attn_out_dim,
                seq_len,
            )?,
            LayerKind::LinearAttention(attn) => self.prefill_linear_attention(
                attn,
                &normed_batch,
                linear_idx,
                recurrent,
                gdn_scratch,
                seq_len,
            )?,
        };

        // 3. Residual + post-attention layernorm
        let hidden_plus_attn = ops::add_batch(&self.ctx, hidden_batch, &attn_results)?;

        // Post-attention layernorm (1+weight offset, batched per-token)
        normed_batch =
            self.batched_rms_norm_offset(&hidden_plus_attn, &layer.post_attention_layernorm, eps)?;

        // 4. MLP (batched)
        let gate_up_out = ops::gemm(&self.ctx, &layer.mlp.gate_up_proj, &normed_batch)?;
        let mut act_out = HiddenStates::zeros(&self.ctx, geom.local_intermediate_size(), seq_len)?;
        ops::silu_mul_fused_batch_into(&self.ctx, &gate_up_out, &mut act_out)?;
        let mut mlp_out = ops::gemm(&self.ctx, &layer.mlp.down_proj, &act_out)?;
        self.all_reduce_hidden(&mut mlp_out)?;

        // 5. Residual
        ops::add_batch(&self.ctx, &hidden_plus_attn, &mlp_out)
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_full_attention(
        &self,
        attn: &FullAttentionLayer,
        normed_batch: &HiddenStates,
        full_idx: &mut usize,
        kv: &PrefillKvAccess<'_>,
        _attn_out_dim: usize,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        let c = &self.config;
        let attn_out_dim = self.geometry.local_full_attn_q_dim();
        let q_full_batch = ops::gemm(&self.ctx, &attn.q_proj, normed_batch)?;
        let k_batch = ops::gemm(&self.ctx, &attn.k_proj, normed_batch)?;
        let v_batch = ops::gemm(&self.ctx, &attn.v_proj, normed_batch)?;
        let mut q_prepped = HiddenStates::zeros(&self.ctx, attn_out_dim, seq_len)?;
        let mut attn_out_batch = HiddenStates::zeros(&self.ctx, attn_out_dim, seq_len)?;
        attention_prefill_hd256_into(
            &self.ctx,
            &q_full_batch,
            &k_batch,
            &v_batch,
            &attn.q_norm,
            &attn.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            kv.buffer,
            &kv.layout.kernel_layout(),
            *full_idx,
            kv.plan,
            kv.base_pos,
            c.rotary_dim,
            c.rms_norm_eps,
            &mut q_prepped,
            &mut attn_out_batch,
        )?;
        *full_idx += 1;

        let mut projected = ops::gemm(&self.ctx, &attn.o_proj, &attn_out_batch)?;
        self.all_reduce_hidden(&mut projected)?;
        Ok(projected)
    }

    fn prefill_linear_attention(
        &self,
        attn: &LinearAttentionLayer,
        normed_batch: &HiddenStates,
        linear_idx: &mut usize,
        recurrent: &mut RecurrentState,
        gdn_scratch: &mut GdnPrefillChunkScratch,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        let c = &self.config;
        let geom = self.geometry;

        // Batch projections. The weights are the fused ones the decode path
        // issues a single GEMM against; a prefill step reads the row range it
        // wants into its own buffer, so every kernel downstream is unchanged.
        let qkv_dim = geom.local_linear_qkv_dim();
        let z_dim = geom.local_linear_z_dim();
        let value_heads = geom.local_linear_num_value_heads();

        let mut qkv_batch = HiddenStates::zeros(&self.ctx, qkv_dim, seq_len)?;
        ops::gemm_rows_into(
            &self.ctx,
            &attn.in_proj_qkvz,
            0,
            qkv_dim,
            normed_batch,
            &mut qkv_batch,
        );
        let mut z_batch = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
        ops::gemm_rows_into(
            &self.ctx,
            &attn.in_proj_qkvz,
            qkv_dim,
            z_dim,
            normed_batch,
            &mut z_batch,
        );
        let mut b_batch = HiddenStates::zeros(&self.ctx, value_heads, seq_len)?;
        ops::gemm_rows_into(
            &self.ctx,
            &attn.in_proj_ba,
            0,
            value_heads,
            normed_batch,
            &mut b_batch,
        );
        let mut a_batch = HiddenStates::zeros(&self.ctx, value_heads, seq_len)?;
        ops::gemm_rows_into(
            &self.ctx,
            &attn.in_proj_ba,
            value_heads,
            value_heads,
            normed_batch,
            &mut a_batch,
        );

        let layer_state = &mut recurrent.layers[*linear_idx];

        let mut qkv_conv_batch = HiddenStates::zeros(&self.ctx, qkv_dim, seq_len)?;
        ops::conv1d_prefill_batch_into(
            &self.ctx,
            &qkv_batch,
            &attn.conv1d_weight,
            &mut layer_state.conv_state,
            &mut qkv_conv_batch,
            c.linear_conv_kernel_dim,
        );

        let mut normed_out_batch = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
        {
            let mut triton_output;
            let gdr_output = match gdn_scratch {
                GdnPrefillChunkScratch::Triton(scratch) => {
                    triton_output = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
                    ops::gated_delta_rule_prefill_chunkwise_into(
                        &self.ctx,
                        &qkv_conv_batch,
                        &b_batch,
                        &a_batch,
                        &attn.dt_bias,
                        &attn.a_log,
                        &mut layer_state.state,
                        scratch,
                        &mut triton_output,
                        geom.local_linear_num_key_heads(),
                        geom.local_linear_num_value_heads(),
                        c.linear_key_head_dim,
                        c.linear_value_head_dim,
                    )?;
                    &triton_output
                }
                GdnPrefillChunkScratch::FlashInfer(resources) => {
                    ops::gated_delta_rule_prefill_native_prepare_into(
                        &self.ctx,
                        &qkv_conv_batch,
                        &b_batch,
                        &a_batch,
                        &attn.dt_bias,
                        &attn.a_log,
                        &mut resources.prepare,
                    )?;
                    resources.launch_in_place(
                        &self.ctx,
                        self.flashinfer_gdn
                            .as_ref()
                            .context("FlashInfer GDN was not selected when loading this model")?,
                        &mut layer_state.state,
                    )?;
                    &resources.output
                }
            };
            ops::rms_norm_gated_batch_into(
                &self.ctx,
                gdr_output,
                &attn.norm_weight,
                &z_batch,
                &mut normed_out_batch,
                geom.local_linear_num_value_heads(),
                c.linear_value_head_dim,
                c.rms_norm_eps,
                ops::GatedNormActivation::Silu,
            );
        }

        *linear_idx += 1;

        // Output projection (batched)
        let mut projected = ops::gemm(&self.ctx, &attn.out_proj, &normed_out_batch)?;
        self.all_reduce_hidden(&mut projected)?;
        Ok(projected)
    }

    fn batched_rms_norm_offset(
        &self,
        x: &HiddenStates,
        weight: &DeviceVec,
        eps: f32,
    ) -> Result<HiddenStates> {
        let mut out = HiddenStates::zeros(&self.ctx, x.hidden_dim, x.seq_len)?;
        ops::rms_norm_batch_offset_into(&self.ctx, x, weight, eps, &mut out)?;
        Ok(out)
    }
}

#[cfg(test)]
mod tests;
