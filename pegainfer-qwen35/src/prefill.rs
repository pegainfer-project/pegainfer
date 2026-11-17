use std::mem::size_of;

use anyhow::Context;
use anyhow::Result;
use half::bf16;

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
use crate::config::Config35;
use crate::config::LocalGeometry;
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

/// Device buffers one prefill pass allocates once and reuses for every layer.
///
/// Layers of the same kind need the same tensor shapes, so one buffer per role
/// replaces the per-layer allocation of every intermediate tensor. A 1024-token
/// pass previously issued roughly 460 `cudaMallocAsync` calls, one per
/// intermediate tensor of every layer; the pass now issues one per role.
pub(crate) struct PrefillPassBuffers {
    normed: HiddenStates,
    q: HiddenStates,
    k: HiddenStates,
    v: HiddenStates,
    q_prepped: HiddenStates,
    attn_out: HiddenStates,
    qkv: HiddenStates,
    z: HiddenStates,
    b: HiddenStates,
    a: HiddenStates,
    conv: HiddenStates,
    gdr: HiddenStates,
    normed_out: HiddenStates,
    gate_up: HiddenStates,
    act: HiddenStates,
    projected: HiddenStates,
}

impl PrefillPassBuffers {
    /// Bytes one pass-wide buffer set holds for a `seq_len`-token chunk.
    ///
    /// Every role is allocated once and lives for the whole pass, so the full
    /// attention, linear attention and MLP buffers all exist at the same time.
    /// The two hidden blocks the chunk loop ping-pongs are counted here too;
    /// `prefill_last_hidden` allocates them next to this set.
    pub(crate) fn estimate_bytes(
        config: &Config35,
        geometry: LocalGeometry,
        seq_len: usize,
    ) -> usize {
        let hidden = config.hidden_size;
        let intermediate = geometry.local_intermediate_size();
        let value_heads = geometry.local_linear_num_value_heads();
        let q_dim = geometry.local_full_attn_q_dim();
        let kv_dim = geometry.local_full_attn_kv_dim();
        let qkv_dim = geometry.local_linear_qkv_dim();
        let z_dim = geometry.local_linear_z_dim();

        let full_attn = geometry.local_full_attn_gated_q_dim() // q
            + 2 * kv_dim // k, v
            + 2 * q_dim; // q_prepped, attn_out
        let linear = 2 * qkv_dim // qkv, conv
            + 3 * z_dim // z, gdr, normed_out
            + 2 * value_heads; // b, a
        let mlp = 3 * intermediate; // gate_up twice, act once
        let hidden_roles = 4 * hidden; // normed, projected, two ping-pong blocks

        (full_attn + linear + mlp + hidden_roles) * seq_len * size_of::<bf16>()
    }
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
            let chunk_len = chunk.len();
            let mut chunk_hidden =
                HiddenStates::zeros(&self.ctx, self.config.hidden_size, chunk_len)?;
            self.embed_prefill_chunk(chunk, &mut chunk_hidden)?;
            let mut chunk_next =
                HiddenStates::zeros(&self.ctx, self.config.hidden_size, chunk_len)?;
            let mut chunk_scratch = self.gdn_prefill_chunk_scratch(chunk_len)?;
            let mut pass_buffers = self.prefill_pass_buffers(chunk_len)?;
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
            self.prefill_chunk_forward(
                &mut chunk_hidden,
                &mut chunk_next,
                &mut pass_buffers,
                &mut chunk_scratch,
                &access,
                recurrent,
            )?;
            hidden_batch = Some(chunk_hidden);
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

    /// Allocate the pass-wide buffers for one chunk of `seq_len` tokens.
    ///
    /// Every layer of the same kind uses the same shapes, so the pass owns one
    /// buffer per role instead of allocating each intermediate tensor per layer.
    fn prefill_pass_buffers(&self, seq_len: usize) -> Result<PrefillPassBuffers> {
        let ctx = &self.ctx;
        let geom = self.geometry;
        let hidden = self.config.hidden_size;
        let intermediate = geom.local_intermediate_size();
        let value_heads = geom.local_linear_num_value_heads();
        let q_dim = geom.local_full_attn_q_dim();
        let kv_dim = geom.local_full_attn_kv_dim();
        let qkv_dim = geom.local_linear_qkv_dim();
        let z_dim = geom.local_linear_z_dim();
        Ok(PrefillPassBuffers {
            normed: HiddenStates::zeros(ctx, hidden, seq_len)?,
            q: HiddenStates::zeros(ctx, geom.local_full_attn_gated_q_dim(), seq_len)?,
            k: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            v: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            q_prepped: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            attn_out: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            qkv: HiddenStates::zeros(ctx, qkv_dim, seq_len)?,
            z: HiddenStates::zeros(ctx, z_dim, seq_len)?,
            b: HiddenStates::zeros(ctx, value_heads, seq_len)?,
            a: HiddenStates::zeros(ctx, value_heads, seq_len)?,
            conv: HiddenStates::zeros(ctx, qkv_dim, seq_len)?,
            gdr: HiddenStates::zeros(ctx, z_dim, seq_len)?,
            normed_out: HiddenStates::zeros(ctx, z_dim, seq_len)?,
            gate_up: HiddenStates::zeros(ctx, 2 * intermediate, seq_len)?,
            act: HiddenStates::zeros(ctx, intermediate, seq_len)?,
            projected: HiddenStates::zeros(ctx, hidden, seq_len)?,
        })
    }

    /// Allocate the GDN scratch for one chunk of `seq_len` tokens.
    ///
    /// The pass length is bounded by `PREFILL_CHUNK_LEN`, so one scratch serves
    /// every layer of the chunk.
    fn gdn_prefill_chunk_scratch(&self, seq_len: usize) -> Result<GdnPrefillChunkScratch> {
        anyhow::ensure!(
            seq_len > 0 && seq_len <= PREFILL_CHUNK_LEN,
            "prefill chunk length {seq_len} out of range 1..={PREFILL_CHUNK_LEN}"
        );
        let c = &self.config;
        Ok(match &self.flashinfer_gdn {
            None => GdnPrefillChunkScratch::Triton(Box::new(gdr_chunkwise_scratch(
                &self.ctx,
                c,
                self.geometry,
                seq_len,
            )?)),
            Some(backend) => GdnPrefillChunkScratch::FlashInfer(Box::new(
                FlashInferGdnChunkResources::new(&self.ctx, backend, seq_len)?,
            )),
        })
    }

    /// Embed one prefill chunk into a caller-allocated buffer.
    fn embed_prefill_chunk(&self, token_ids: &[u32], hidden: &mut HiddenStates) -> Result<()> {
        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(token_ids)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {e}"))?;
        ops::embedding_batch(&self.ctx, &self.embed_tokens, &token_ids_gpu, hidden)
    }

    /// Run the layer stack of one chunk.
    ///
    /// The hidden state alternates between two caller-owned buffers because every
    /// layer adds two residuals; it is in `hidden` again when this returns.
    #[allow(clippy::too_many_arguments)]
    fn prefill_chunk_forward(
        &self,
        hidden: &mut HiddenStates,
        next: &mut HiddenStates,
        buffers: &mut PrefillPassBuffers,
        gdn_scratch: &mut GdnPrefillChunkScratch,
        kv: &PrefillKvAccess<'_>,
        recurrent: &mut RecurrentState,
    ) -> Result<()> {
        let seq_len = hidden.seq_len;
        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        let mut input_is_hidden = true;

        for layer in &self.layers {
            let (input, output) = if input_is_hidden {
                (&*hidden, &mut *next)
            } else {
                (&*next, &mut *hidden)
            };
            self.prefill_layer(
                layer,
                input,
                output,
                buffers,
                gdn_scratch,
                &mut linear_idx,
                &mut full_idx,
                kv,
                recurrent,
            )?;
            input_is_hidden = !input_is_hidden;
        }
        if !input_is_hidden {
            std::mem::swap(hidden, next);
        }

        // Advance recurrent token count for the next chunk / decode step; the
        // paged KV position is committed by the caller.
        recurrent.seq_len += seq_len;

        Ok(())
    }

    /// Process one layer during prefill.
    ///
    /// Reads `input`, writes `input + attention + mlp` into `output`, and reuses
    /// the pass buffers for every intermediate tensor.
    #[allow(clippy::too_many_arguments)]
    fn prefill_layer(
        &self,
        layer: &TransformerBlock35,
        input: &HiddenStates,
        output: &mut HiddenStates,
        buffers: &mut PrefillPassBuffers,
        gdn_scratch: &mut GdnPrefillChunkScratch,
        linear_idx: &mut usize,
        full_idx: &mut usize,
        kv: &PrefillKvAccess<'_>,
        recurrent: &mut RecurrentState,
    ) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        debug_assert_eq!(buffers.normed.seq_len, input.seq_len);

        // 1. Input layernorm (1+weight offset, batched per token)
        ops::rms_norm_batch_offset_into(
            &self.ctx,
            input,
            &layer.input_layernorm,
            eps,
            &mut buffers.normed,
        )?;

        // 2. Attention / linear attention, projected into `buffers.projected`.
        match &layer.attn {
            LayerKind::FullAttention(attn) => {
                self.prefill_full_attention(attn, buffers, full_idx, kv)?;
            }
            LayerKind::LinearAttention(attn) => {
                self.prefill_linear_attention(attn, buffers, linear_idx, recurrent, gdn_scratch)?;
            }
        }

        // 3. Residual, then post-attention layernorm into `buffers.normed`.
        ops::add_batch_into(&self.ctx, input, &buffers.projected, output)?;
        ops::rms_norm_batch_offset_into(
            &self.ctx,
            output,
            &layer.post_attention_layernorm,
            eps,
            &mut buffers.normed,
        )?;

        // 4. MLP into `buffers.projected`.
        ops::gemm_into_checked(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            &buffers.normed,
            &mut buffers.gate_up,
        )?;
        ops::silu_mul_fused_batch_into(&self.ctx, &buffers.gate_up, &mut buffers.act)?;
        ops::gemm_into_checked(
            &self.ctx,
            &layer.mlp.down_proj,
            &buffers.act,
            &mut buffers.projected,
        )?;
        self.all_reduce_hidden(&mut buffers.projected)?;

        // 5. Residual: `output` is the accumulator and also its own addend.
        ops::add_batch_inplace(&self.ctx, output, &buffers.projected)
    }

    fn prefill_full_attention(
        &self,
        attn: &FullAttentionLayer,
        buffers: &mut PrefillPassBuffers,
        full_idx: &mut usize,
        kv: &PrefillKvAccess<'_>,
    ) -> Result<()> {
        let c = &self.config;
        ops::gemm_into_checked(&self.ctx, &attn.q_proj, &buffers.normed, &mut buffers.q)?;
        ops::gemm_into_checked(&self.ctx, &attn.k_proj, &buffers.normed, &mut buffers.k)?;
        ops::gemm_into_checked(&self.ctx, &attn.v_proj, &buffers.normed, &mut buffers.v)?;

        attention_prefill_hd256_into(
            &self.ctx,
            &buffers.q,
            &buffers.k,
            &buffers.v,
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
            &mut buffers.q_prepped,
            &mut buffers.attn_out,
        )?;
        *full_idx += 1;

        ops::gemm_into_checked(
            &self.ctx,
            &attn.o_proj,
            &buffers.attn_out,
            &mut buffers.projected,
        )?;
        self.all_reduce_hidden(&mut buffers.projected)
    }

    fn prefill_linear_attention(
        &self,
        attn: &LinearAttentionLayer,
        buffers: &mut PrefillPassBuffers,
        linear_idx: &mut usize,
        recurrent: &mut RecurrentState,
        gdn_scratch: &mut GdnPrefillChunkScratch,
    ) -> Result<()> {
        let c = &self.config;
        let geom = self.geometry;

        // Batch projections. The weights are the fused ones the decode path
        // issues a single GEMM against; a prefill step reads the row range it
        // wants into its own buffer, so every kernel downstream is unchanged.
        let qkv_dim = geom.local_linear_qkv_dim();
        let z_dim = geom.local_linear_z_dim();
        let value_heads = geom.local_linear_num_value_heads();

        ops::gemm_rows_into_checked(
            &self.ctx,
            &attn.in_proj_qkvz,
            0,
            qkv_dim,
            &buffers.normed,
            &mut buffers.qkv,
        )?;
        ops::gemm_rows_into_checked(
            &self.ctx,
            &attn.in_proj_qkvz,
            qkv_dim,
            z_dim,
            &buffers.normed,
            &mut buffers.z,
        )?;
        ops::gemm_rows_into_checked(
            &self.ctx,
            &attn.in_proj_ba,
            0,
            value_heads,
            &buffers.normed,
            &mut buffers.b,
        )?;
        ops::gemm_rows_into_checked(
            &self.ctx,
            &attn.in_proj_ba,
            value_heads,
            value_heads,
            &buffers.normed,
            &mut buffers.a,
        )?;

        let layer_state = &mut recurrent.layers[*linear_idx];

        ops::conv1d_prefill_batch_into(
            &self.ctx,
            &buffers.qkv,
            &attn.conv1d_weight,
            &mut layer_state.conv_state,
            &mut buffers.conv,
            c.linear_conv_kernel_dim,
        );

        let gdr_output = match gdn_scratch {
            GdnPrefillChunkScratch::Triton(scratch) => {
                ops::gated_delta_rule_prefill_chunkwise_into(
                    &self.ctx,
                    &buffers.conv,
                    &buffers.b,
                    &buffers.a,
                    &attn.dt_bias,
                    &attn.a_log,
                    &mut layer_state.state,
                    scratch,
                    &mut buffers.gdr,
                    geom.local_linear_num_key_heads(),
                    geom.local_linear_num_value_heads(),
                    c.linear_key_head_dim,
                    c.linear_value_head_dim,
                )?;
                &buffers.gdr
            }
            GdnPrefillChunkScratch::FlashInfer(resources) => {
                ops::gated_delta_rule_prefill_native_prepare_into(
                    &self.ctx,
                    &buffers.conv,
                    &buffers.b,
                    &buffers.a,
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
            &buffers.z,
            &mut buffers.normed_out,
            geom.local_linear_num_value_heads(),
            c.linear_value_head_dim,
            c.rms_norm_eps,
            ops::GatedNormActivation::Silu,
        );

        *linear_idx += 1;

        // Output projection into the layer's output slot.
        ops::gemm_into_checked(
            &self.ctx,
            &attn.out_proj,
            &buffers.normed_out,
            &mut buffers.projected,
        )?;
        self.all_reduce_hidden(&mut buffers.projected)
    }
}

#[cfg(test)]
mod tests;
