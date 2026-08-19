use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

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
const HEAD_DIM: usize = 256;

use pegainfer_core::kv_pool::KvState;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

use super::flashinfer_gdn::FlashInferGdnChunkResources;
use super::flashinfer_gdn::GdnPrefillBackend;
pub use super::flashinfer_gdn::GdnPrefillRuntimeEvidence;
pub use super::flashinfer_gdn::GdnPrefillRuntimeEvidenceHandle;
use super::prefill_buffers::GdrChunkwiseScratch35;
use super::recurrent_state::RecurrentState;
use super::weights::FullAttentionLayer;
use super::weights::LayerKind;
use super::weights::LinearAttentionLayer;
use super::weights::Qwen35Model;
use super::weights::TransformerBlock35;
use crate::ffi;
use crate::ops;
use crate::ops::PrefillPagedPlan;

enum GdnPrefillChunkScratch {
    Triton(Box<GdrChunkwiseScratch35>),
    FlashInfer(Box<FlashInferGdnChunkResources>),
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
    /// Require the build-linked candidate for an explicit same-path A/B gate.
    /// Artifact selection and validation happen in `pegainfer-kernels` at build
    /// time; model code never consumes an artifact path at runtime.
    pub(crate) fn require_flashinfer_gdn_for_test(&self) -> Result<()> {
        anyhow::ensure!(
            self.flashinfer_gdn.is_some(),
            "FlashInfer GDN is not available; set PEGAINFER_QWEN35_GDN_AOT_BUNDLE at build time"
        );
        Ok(())
    }

    pub(super) fn prefill_last_hidden(
        &self,
        token_ids: &[u32],
        kv_state: &mut KvState,
        recurrent: &mut RecurrentState,
    ) -> Result<DeviceVec> {
        let seq_len = token_ids.len();
        anyhow::ensure!(
            seq_len > 0,
            "Qwen3.5 prefill_last_hidden requires at least one token"
        );
        let c = &self.config;

        // Validate the full target range up front (position overflow + RoPE cache
        // coverage) so an out-of-range prompt is rejected before any chunk mutates
        // the KV / recurrent state, rather than failing partway through.
        let base_pos = kv_state.seq_len();
        let end_pos = checked_prefill_end_pos(base_pos, seq_len, c.max_position_embeddings)?;
        self.ensure_rope_cache_covers(end_pos)?;

        // Run prefill in serial chunks of at most `PREFILL_CHUNK_LEN` tokens. Each
        // chunk advances the paged KV and linear-attention recurrent/conv state in
        // place, so the next chunk continues from the previous one. This caps the
        // per-pass GDR scratch (which grows with the pass length) at the budget
        // reserved at startup, so prompts longer than one chunk prefill without OOM.
        let mut hidden_batch: Option<HiddenStates> = None;
        let gdn_backend = self.resolved_gdn_backend();
        for chunk in token_ids.chunks(PREFILL_CHUNK_LEN) {
            // Free the previous chunk's hidden states before allocating the next
            // chunk's scratch so peak memory stays within one chunk's reservation.
            drop(hidden_batch.take());
            hidden_batch =
                Some(self.prefill_chunk_forward(chunk, kv_state, recurrent, gdn_backend)?);
        }
        // `seq_len > 0` guarantees at least one chunk produced hidden states.
        let hidden_batch = hidden_batch.expect("prefill produced no chunk despite seq_len > 0");

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
        ops::gemm_rows_into_checked(
            &self.ctx,
            self.output_projection(),
            0,
            self.config.selection_vocab,
            &normed,
            &mut logits,
        )?;
        debug_assert_eq!(logits.seq_len, n);
        Ok(logits)
    }

    /// Forward one prefill chunk through all layers, advancing the paged KV state
    /// and the linear-attention recurrent/conv state in place.
    ///
    /// `token_ids.len()` must be in `1..=PREFILL_CHUNK_LEN` so the per-chunk GDR
    /// scratch stays within the startup reservation. Returns the chunk's hidden
    /// states for every token; only the final chunk's last token feeds the LM head.
    fn prefill_chunk_forward(
        &self,
        token_ids: &[u32],
        kv_state: &mut KvState,
        recurrent: &mut RecurrentState,
        gdn_backend: GdnPrefillBackend,
    ) -> Result<HiddenStates> {
        let seq_len = token_ids.len();
        anyhow::ensure!(
            seq_len > 0 && seq_len <= PREFILL_CHUNK_LEN,
            "prefill chunk length {seq_len} out of range 1..={PREFILL_CHUNK_LEN}"
        );
        let c = &self.config;
        let base_pos = kv_state.seq_len();
        let end_pos = checked_prefill_end_pos(base_pos, seq_len, c.max_position_embeddings)?;
        self.ensure_rope_cache_covers(end_pos)?;

        // Embeddings for this chunk.
        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(token_ids)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;

        let hidden_dim = c.hidden_size;
        let mut hidden_batch = HiddenStates::zeros(&self.ctx, hidden_dim, seq_len)?;
        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &token_ids_gpu,
            &mut hidden_batch,
        )?;

        // Allocate the chunk scratch before advancing the KV state. It is the
        // largest, most allocation-prone buffer here, so failing first leaves
        // `kv_state` untouched and the request can be rejected cleanly.
        let mut gdn_scratch = match gdn_backend {
            GdnPrefillBackend::Triton => GdnPrefillChunkScratch::Triton(Box::new(
                GdrChunkwiseScratch35::new(&self.ctx, c, seq_len)?,
            )),
            GdnPrefillBackend::FlashInfer => {
                let backend = self.flashinfer_gdn()?;
                GdnPrefillChunkScratch::FlashInfer(Box::new(FlashInferGdnChunkResources::new(
                    &self.ctx,
                    &self.config,
                    backend,
                    seq_len,
                )?))
            }
        };

        // Advance paged KV state and build this chunk's prefill plan.
        kv_state.ensure_capacity(end_pos)?;
        kv_state.advance(seq_len);
        let kv_desc = kv_state.desc();
        let tp = self.tensor_parallel;
        let prefill_plan = PrefillPagedPlan::new(
            &self.ctx,
            &kv_desc,
            base_pos,
            seq_len,
            c.local_num_attention_heads(tp),
            c.local_num_key_value_heads(tp),
            c.head_dim,
        )?;

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
                kv_state,
                &prefill_plan,
                recurrent,
            )?;
        }

        if let GdnPrefillChunkScratch::FlashInfer(resources) = &gdn_scratch {
            resources.ensure_prepare_inputs_finite(&self.ctx)?;
        }

        // Advance recurrent token count for the next chunk / decode step; the
        // paged KV position is tracked by `kv_state` (advanced above).
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
        kv_state: &KvState,
        prefill_plan: &PrefillPagedPlan,
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
        let tp = self.tensor_parallel;
        let attn_out_dim = match &layer.attn {
            LayerKind::FullAttention(_) => c.local_full_attn_q_dim(tp),
            LayerKind::LinearAttention(_) => c.linear_attn_z_dim(),
        };

        // Batch project, then per-token attention/recurrent
        let attn_results = match &layer.attn {
            LayerKind::FullAttention(attn) => self.prefill_full_attention(
                attn,
                &normed_batch,
                full_idx,
                kv_state,
                prefill_plan,
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
        let mut act_out = HiddenStates::zeros(&self.ctx, c.local_intermediate_size(tp), seq_len)?;
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
        kv_state: &KvState,
        prefill_plan: &PrefillPagedPlan,
        _attn_out_dim: usize,
        seq_len: usize,
    ) -> Result<HiddenStates> {
        let c = &self.config;
        let tp = self.tensor_parallel;
        let num_attention_heads = c.local_num_attention_heads(tp);
        let num_key_value_heads = c.local_num_key_value_heads(tp);
        let attn_out_dim = c.local_full_attn_q_dim(tp);
        let eps = c.rms_norm_eps;
        let q_full_batch = ops::gemm(&self.ctx, &attn.q_proj, normed_batch)?;
        let k_batch = ops::gemm(&self.ctx, &attn.k_proj, normed_batch)?;
        let v_batch = ops::gemm(&self.ctx, &attn.v_proj, normed_batch)?;
        let mut attn_out_batch = HiddenStates::zeros(&self.ctx, attn_out_dim, seq_len)?;

        // `kv_state` was advanced by `seq_len` before the layer loop, so the
        // base write position for this prefill is `seq_len()` minus this batch.
        let base_pos = kv_state.seq_len() - seq_len;
        let mut q_prepped = HiddenStates::zeros(&self.ctx, attn_out_dim, seq_len)?;
        let start_pos_cpu: CudaSlice<i32> = self
            .ctx
            .stream
            .clone_htod(&[base_pos as i32])
            .map_err(|e| anyhow::anyhow!("H2D start_pos failed: {e}"))?;
        let layout = kv_state.layout();
        let layer_k_off = (*full_idx * layout.layer_stride) as i64;
        let layer_v_off = layer_k_off + layout.kv_block_len as i64;
        let stride_page = layout.page_stride as i64;

        // Step 1: QK norm + partial RoPE + direct paged K/V write.
        unsafe {
            let (qf_ptr, _) = q_full_batch.data.device_ptr(&self.ctx.stream);
            let (k_ptr, _) = k_batch.data.device_ptr(&self.ctx.stream);
            let (v_ptr, _) = v_batch.data.device_ptr(&self.ctx.stream);
            let (qn_ptr, _) = attn.q_norm.data.device_ptr(&self.ctx.stream);
            let (kn_ptr, _) = attn.k_norm.data.device_ptr(&self.ctx.stream);
            let (cos_ptr, _) = self.cos_cache.data.device_ptr(&self.ctx.stream);
            let (sin_ptr, _) = self.sin_cache.data.device_ptr(&self.ctx.stream);
            let (qp_ptr, _) = q_prepped.data.device_ptr_mut(&self.ctx.stream);
            let (buf_ptr, _) = kv_state.buffer().device_ptr(&self.ctx.stream);
            let (pi_ptr, _) = prefill_plan.page_indices_d().device_ptr(&self.ctx.stream);
            let (sp_ptr, _) = start_pos_cpu.device_ptr(&self.ctx.stream);
            ffi::prefill_attention_hd256_prep_paged_cuda(
                qf_ptr as *const ffi::Half,
                k_ptr as *const ffi::Half,
                v_ptr as *const ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                qp_ptr as *mut ffi::Half,
                buf_ptr as *mut ffi::Half,
                layer_k_off,
                layer_v_off,
                pi_ptr as *const i32,
                num_attention_heads as i32,
                num_key_value_heads as i32,
                seq_len as i32,
                sp_ptr as *const i32,
                c.rotary_dim as i32,
                eps,
                layout.page_size as i32,
                stride_page,
                self.ctx.stream.cu_stream(),
            );
        }

        // Step 2: Batch prefill paged attention (HD=256).
        let sm_scale = 1.0f32 / f32::sqrt(HEAD_DIM as f32);
        {
            let (buf_ptr, _gbuf) = kv_state.buffer().device_ptr(&self.ctx.stream);
            let (qp_ptr, _gqp) = q_prepped.data.device_ptr(&self.ctx.stream);
            let (out_ptr, _go) = attn_out_batch.data.device_ptr_mut(&self.ctx.stream);
            let (pi_ptr, _gpi) = prefill_plan.page_indices_d().device_ptr(&self.ctx.stream);
            let (pip_ptr, _gpip) = prefill_plan.page_indptr_d().device_ptr(&self.ctx.stream);
            let (lpl_ptr, _glpl) = prefill_plan.last_page_len_d().device_ptr(&self.ctx.stream);
            let (qi_ptr, _gqi) = prefill_plan.q_indptr_d().device_ptr(&self.ctx.stream);
            let (ri_ptr, _gri) = prefill_plan
                .request_indices_d()
                .device_ptr(&self.ctx.stream);
            let (qti_ptr, _gqti) = prefill_plan
                .qo_tile_indices_d()
                .device_ptr(&self.ctx.stream);
            let (kti_ptr, _gkti) = prefill_plan
                .kv_tile_indices_d()
                .device_ptr(&self.ctx.stream);
            let (kcs_ptr, _gkcs) = prefill_plan.kv_chunk_size_d().device_ptr(&self.ctx.stream);
            let (tnr_ptr, _gtnr) = prefill_plan.total_num_rows_d().device_ptr(&self.ctx.stream);
            let result = unsafe {
                ffi::batch_prefill_paged_cuda_hd256(
                    qp_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    buf_ptr as *const ffi::Half,
                    layer_k_off,
                    layer_v_off,
                    pi_ptr as *const i32,
                    pip_ptr as *const i32,
                    lpl_ptr as *const i32,
                    qi_ptr as *const i32,
                    ri_ptr as *const i32,
                    qti_ptr as *const i32,
                    kti_ptr as *const i32,
                    kcs_ptr as *const i32,
                    tnr_ptr as *const u32,
                    num_attention_heads as i32,
                    num_key_value_heads as i32,
                    HEAD_DIM as i32,
                    layout.page_size as i32,
                    seq_len as i32,
                    prefill_plan.batch_size(),
                    prefill_plan.num_tiles(),
                    stride_page,
                    sm_scale,
                    self.ctx.stream.cu_stream(),
                )
            };
            anyhow::ensure!(
                result == 0,
                "batch_prefill_paged_cuda_hd256 failed: {result}{}",
                pegainfer_kernels::ops::ffi_exception_message(result)
            );
        }

        // Step 3: Apply gate from q_full_batch.
        {
            let (qf_ptr, _gqf) = q_full_batch.data.device_ptr(&self.ctx.stream);
            let (out_ptr, _go) = attn_out_batch.data.device_ptr_mut(&self.ctx.stream);
            unsafe {
                ffi::attention_gate_batch_hd256_cuda(
                    qf_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    num_attention_heads as i32,
                    seq_len as i32,
                    self.ctx.stream.cu_stream(),
                );
            }
        }

        *full_idx += 1;

        // O projection (batched)
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

        // Batch projections
        let qkv_batch = ops::gemm(&self.ctx, &attn.in_proj_qkv, normed_batch)?;
        let z_batch = ops::gemm(&self.ctx, &attn.in_proj_z, normed_batch)?;
        let b_batch = ops::gemm(&self.ctx, &attn.in_proj_b, normed_batch)?;
        let a_batch = ops::gemm(&self.ctx, &attn.in_proj_a, normed_batch)?;

        let qkv_dim = c.linear_attn_qkv_dim();
        let z_dim = c.linear_attn_z_dim();
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
        match gdn_scratch {
            GdnPrefillChunkScratch::Triton(scratch) => {
                let mut gdr_out_batch = HiddenStates::zeros(&self.ctx, z_dim, seq_len)?;
                ops::gated_delta_rule_prefill_chunkwise_into(
                    &self.ctx,
                    &qkv_conv_batch,
                    &b_batch,
                    &a_batch,
                    &attn.dt_bias,
                    &attn.a_log,
                    &mut layer_state.state,
                    scratch,
                    &mut gdr_out_batch,
                    c.linear_num_key_heads,
                    c.linear_num_value_heads,
                    c.linear_key_head_dim,
                    c.linear_value_head_dim,
                )?;
                ops::rms_norm_gated_batch_into(
                    &self.ctx,
                    &gdr_out_batch,
                    &attn.norm_weight,
                    &z_batch,
                    &mut normed_out_batch,
                    c.linear_num_value_heads,
                    c.linear_value_head_dim,
                    c.rms_norm_eps,
                );
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
                    c.linear_num_key_heads,
                    c.linear_num_key_heads,
                    c.linear_num_value_heads,
                    c.linear_key_head_dim,
                )?;
                resources.launch_in_place(
                    &self.ctx,
                    self.flashinfer_gdn()?,
                    &mut layer_state.state,
                )?;
                ops::rms_norm_gated_batch_into(
                    &self.ctx,
                    &resources.output,
                    &attn.norm_weight,
                    &z_batch,
                    &mut normed_out_batch,
                    c.linear_num_value_heads,
                    c.linear_value_head_dim,
                    c.rms_norm_eps,
                );
            }
        }

        *linear_idx += 1;

        // Output projection (batched)
        ops::gemm(&self.ctx, &attn.out_proj, &normed_out_batch)
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
mod tests {
    use std::path::Path;

    use anyhow::Result;
    use half::bf16;
    use pegainfer_core::tensor::DeviceVec;

    use super::GdnPrefillBackend;
    use super::checked_prefill_end_pos;
    use crate::recurrent_state::RecurrentState;
    use crate::weights::Qwen35Model;

    // Frozen by the Stage 13 FP32 recurrence gate; chunk continuation does
    // not receive a looser state envelope than the original operator proof.
    const CHUNK_STATE_ATOL: f32 = 5.0e-3;
    const CHUNK_STATE_RTOL: f32 = 2.0e-3;
    const LOGIT_MEAN_TOL: f32 = 0.06;
    const LOGIT_P99_TOL: f32 = 0.20;
    const LOGIT_ARGMAX_REGRET_TOL: f32 = 0.20;

    fn required_model_path() -> String {
        let default = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");
        let path =
            std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| default.to_string());
        assert!(
            Path::new(&path).join("config.json").is_file(),
            "required chunk-continuation gate cannot read {path}/config.json; set PEGAINFER_TEST_MODEL_PATH"
        );
        path
    }

    fn initialize_non_symmetric_state(
        model: &Qwen35Model,
        recurrent: &mut RecurrentState,
    ) -> Result<()> {
        let ctx = model.device_ctx();
        let h_v = model.config().linear_num_value_heads;
        let head_dim = model.config().linear_key_head_dim;
        for (layer_idx, layer) in recurrent.layers.iter_mut().enumerate() {
            assert_eq!(layer.state.len(), h_v * head_dim * head_dim);
            let state = (0..layer.state.len())
                .map(|index| {
                    let head = index / (head_dim * head_dim);
                    let rem = index % (head_dim * head_dim);
                    let key = rem / head_dim;
                    let value = rem % head_dim;
                    (layer_idx * 1_000_000 + head * 100_000 + key * 100 + value) as f32 * 1.0e-7
                })
                .collect::<Vec<_>>();
            layer.state = ctx.stream.clone_htod(&state)?;

            let conv = (0..layer.conv_state.len)
                .map(|index| {
                    let signed = ((index * 29 + layer_idx * 17) % 257) as i32 - 128;
                    bf16::from_f32(signed as f32 * 1.0e-3)
                })
                .collect::<Vec<_>>();
            layer.conv_state = DeviceVec::from_host(ctx, &conv)?;
        }
        recurrent.seq_len = 0;
        Ok(())
    }

    fn assert_exact_bf16(label: &str, expected: &[bf16], actual: &[bf16]) {
        assert_eq!(expected.len(), actual.len(), "{label} length mismatch");
        if let Some(index) = expected
            .iter()
            .zip(actual)
            .position(|(left, right)| left.to_bits() != right.to_bits())
        {
            panic!(
                "{label} first bitwise mismatch at {index}: expected={} actual={}",
                expected[index].to_f32(),
                actual[index].to_f32()
            );
        }
    }

    #[derive(Debug)]
    struct F32DifferenceStats {
        first_violation: Option<(usize, f32, f32, f32)>,
        violations: usize,
        max_abs: f32,
        mean_abs: f32,
        p99_abs: f32,
        max_rel: f32,
    }

    fn difference_stats_f32(
        expected: &[f32],
        actual: &[f32],
        atol: f32,
        rtol: f32,
    ) -> F32DifferenceStats {
        assert_eq!(
            expected.len(),
            actual.len(),
            "f32 comparison length mismatch"
        );
        let mut absolute = Vec::with_capacity(expected.len());
        let mut max_relative = 0.0_f32;
        let mut first_violation = None;
        let mut violation_count = 0usize;
        for (index, (&left, &right)) in expected.iter().zip(actual).enumerate() {
            let diff = (left - right).abs();
            absolute.push(diff);
            max_relative = max_relative.max(diff / left.abs().max(right.abs()).max(1.0e-12));
            let violation = !left.is_finite()
                || !right.is_finite()
                || diff > atol + rtol * left.abs().max(right.abs());
            if violation {
                violation_count += 1;
                if first_violation.is_none() {
                    first_violation = Some((index, left, right, diff));
                }
            }
        }
        absolute.sort_by(f32::total_cmp);
        let max = absolute.last().copied().unwrap_or(0.0);
        let mean = if absolute.is_empty() {
            0.0
        } else {
            absolute.iter().sum::<f32>() / absolute.len() as f32
        };
        let p99_index = absolute.len().saturating_sub(1) * 99 / 100;
        let p99 = absolute.get(p99_index).copied().unwrap_or(0.0);
        F32DifferenceStats {
            first_violation,
            violations: violation_count,
            max_abs: max,
            mean_abs: mean,
            p99_abs: p99,
            max_rel: max_relative,
        }
    }

    fn report_close_f32(
        label: &str,
        expected: &[f32],
        actual: &[f32],
        atol: f32,
        rtol: f32,
    ) -> F32DifferenceStats {
        let stats = difference_stats_f32(expected, actual, atol, rtol);
        eprintln!(
            "{label}: elements={} violations={} max_abs={:.8} mean_abs={:.8} p99_abs={:.8} max_rel={:.8} atol={atol} rtol={rtol}",
            expected.len(),
            stats.violations,
            stats.max_abs,
            stats.mean_abs,
            stats.p99_abs,
            stats.max_rel,
        );
        stats
    }

    fn assert_close_f32(label: &str, expected: &[f32], actual: &[f32], atol: f32, rtol: f32) {
        let stats = report_close_f32(label, expected, actual, atol, rtol);
        assert!(
            stats.first_violation.is_none(),
            "{label} first violation {:?}; violations={}/{} max_abs={} mean_abs={} p99_abs={} max_rel={}",
            stats.first_violation,
            stats.violations,
            expected.len(),
            stats.max_abs,
            stats.mean_abs,
            stats.p99_abs,
            stats.max_rel,
        );
    }

    fn report_layer_state_pair(
        model: &Qwen35Model,
        label: &str,
        expected: &RecurrentState,
        actual: &RecurrentState,
        layer_idx: usize,
    ) -> Result<()> {
        let ctx = model.device_ctx();
        let expected_host = ctx.stream.clone_dtoh(&expected.layers[layer_idx].state)?;
        let actual_host = ctx.stream.clone_dtoh(&actual.layers[layer_idx].state)?;
        ctx.sync()?;
        report_close_f32(
            label,
            &expected_host,
            &actual_host,
            CHUNK_STATE_ATOL,
            CHUNK_STATE_RTOL,
        );
        Ok(())
    }

    fn assert_recurrent_close(
        model: &Qwen35Model,
        expected: &RecurrentState,
        actual: &RecurrentState,
    ) -> Result<()> {
        assert_eq!(expected.seq_len, actual.seq_len);
        assert_eq!(expected.layers.len(), actual.layers.len());
        let ctx = model.device_ctx();
        let mut copies = Vec::with_capacity(expected.layers.len());
        for (layer_idx, (left, right)) in expected.layers.iter().zip(&actual.layers).enumerate() {
            copies.push((
                layer_idx,
                ctx.stream.clone_dtoh(&left.state)?,
                ctx.stream.clone_dtoh(&right.state)?,
                ctx.stream.clone_dtoh(&left.conv_state.data)?,
                ctx.stream.clone_dtoh(&right.conv_state.data)?,
            ));
        }
        ctx.sync()?;
        for (layer_idx, expected_state, actual_state, expected_conv, actual_conv) in copies {
            assert_close_f32(
                &format!("final layer {layer_idx} recurrent state"),
                &expected_state,
                &actual_state,
                CHUNK_STATE_ATOL,
                CHUNK_STATE_RTOL,
            );
            assert_exact_bf16(
                &format!("final layer {layer_idx} conv state"),
                &expected_conv,
                &actual_conv,
            );
        }
        Ok(())
    }

    fn log_softmax(values: &[f32]) -> Vec<f32> {
        let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_sum = values
            .iter()
            .map(|value| (*value - max).exp())
            .sum::<f32>()
            .ln();
        values.iter().map(|value| *value - max - log_sum).collect()
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .expect("logits must be non-empty")
    }

    fn assert_logit_parity(label: &str, expected: &[f32], actual: &[f32]) {
        assert_eq!(
            expected.len(),
            actual.len(),
            "{label} logit length mismatch"
        );
        let expected_lp = log_softmax(expected);
        let actual_lp = log_softmax(actual);
        assert!(
            expected_lp.iter().all(|value| value.is_finite())
                && actual_lp.iter().all(|value| value.is_finite()),
            "{label} contains non-finite log-probabilities"
        );
        let expected_token = argmax(&expected_lp);
        let actual_token = argmax(&actual_lp);
        let regret = expected_lp[expected_token] - expected_lp[actual_token];
        assert!(
            regret <= LOGIT_ARGMAX_REGRET_TOL,
            "{label} actual argmax {actual_token} has baseline regret {regret} > {LOGIT_ARGMAX_REGRET_TOL}"
        );
        assert_eq!(
            actual_token, expected_token,
            "{label} greedy token parity failed"
        );

        let mut deltas = expected_lp
            .iter()
            .zip(&actual_lp)
            .map(|(left, right)| (*left - *right).abs())
            .collect::<Vec<_>>();
        deltas.sort_by(f32::total_cmp);
        let max = deltas.last().copied().unwrap_or(0.0);
        let mean = deltas.iter().sum::<f32>() / deltas.len() as f32;
        let p99 = deltas[deltas.len().saturating_sub(1) * 99 / 100];
        eprintln!(
            "{label}: vocab={} expected_tokens=[{expected_token}] actual_tokens=[{actual_token}] max_logprob_delta={max:.6} mean={mean:.6} p99={p99:.6} regret={regret:.6}",
            deltas.len()
        );
        assert!(
            mean <= LOGIT_MEAN_TOL,
            "{label} mean {mean} > {LOGIT_MEAN_TOL}"
        );
        assert!(p99 <= LOGIT_P99_TOL, "{label} p99 {p99} > {LOGIT_P99_TOL}");
    }

    fn last_token_logits(
        model: &Qwen35Model,
        hidden: &pegainfer_core::tensor::HiddenStates,
    ) -> Result<Vec<f32>> {
        let last = crate::ops::extract_vec(model.device_ctx(), hidden, hidden.seq_len - 1)?;
        let logits = model.batch_last_hidden_logits(&[last])?;
        logits.to_host(model.device_ctx())
    }

    fn run_prefill_case(
        model: &Qwen35Model,
        tokens: &[u32],
        backend: GdnPrefillBackend,
        split_at: Option<usize>,
    ) -> Result<(pegainfer_core::kv_pool::KvState, RecurrentState, Vec<f32>)> {
        let mut kv = model.alloc_kv();
        let mut recurrent = RecurrentState::new(model.device_ctx(), model.config())?;
        initialize_non_symmetric_state(model, &mut recurrent)?;
        let hidden = match split_at {
            Some(split) => {
                assert!(split > 0 && split < tokens.len());
                let first = model.prefill_chunk_forward(
                    &tokens[..split],
                    &mut kv,
                    &mut recurrent,
                    backend,
                )?;
                drop(first);
                model.prefill_chunk_forward(&tokens[split..], &mut kv, &mut recurrent, backend)?
            }
            None => model.prefill_chunk_forward(tokens, &mut kv, &mut recurrent, backend)?,
        };
        let logits = last_token_logits(model, &hidden)?;
        drop(hidden);
        Ok((kv, recurrent, logits))
    }

    fn first_decode_logits(
        model: &Qwen35Model,
        token: u32,
        kv: &mut pegainfer_core::kv_pool::KvState,
        recurrent: &RecurrentState,
    ) -> Result<Vec<f32>> {
        let mut graph = model.create_batch_decode_graph_state_with_capacity(1)?;
        graph.copy_state_to_slot(model.device_ctx(), recurrent, 0)?;
        let mut kv_refs = vec![kv];
        model.batch_decode_graph(&[token], &mut kv_refs, &mut graph)?;
        graph.buffers.logits.to_host(model.device_ctx())
    }

    #[test]
    fn checked_prefill_end_pos_accepts_config_limit() {
        assert_eq!(
            checked_prefill_end_pos(0, 262_144, 262_144).unwrap(),
            262_144
        );
        assert_eq!(
            checked_prefill_end_pos(262_143, 1, 262_144).unwrap(),
            262_144
        );
    }

    #[test]
    fn checked_prefill_end_pos_rejects_past_config_limit() {
        let err = checked_prefill_end_pos(0, 262_145, 262_144)
            .unwrap_err()
            .to_string();
        assert!(err.contains("beyond max_position_embeddings=262144"));
        assert!(err.contains("requested end_pos=262145"));
    }

    #[test]
    fn checked_prefill_end_pos_rejects_overflow() {
        let err = checked_prefill_end_pos(usize::MAX, 1, 262_144)
            .unwrap_err()
            .to_string();
        assert!(err.contains("prefill position overflow"));
    }

    #[test]
    #[ignore = "requires an SM120 GPU, Qwen3.5-4B weights, and a build-linked validated FlashInfer bundle"]
    fn flashinfer_gdn_chunked_prefill_matches_unchunked_state() -> Result<()> {
        let model_path = required_model_path();
        let model = Qwen35Model::from_safetensors(&model_path, 0, 1)?;
        model.require_flashinfer_gdn_for_test()?;
        assert_eq!(model.resolved_gdn_backend(), GdnPrefillBackend::FlashInfer);
        let evidence_before = model.flashinfer_gdn_runtime_evidence()?;
        assert_eq!(evidence_before.selected_backend, "flashinfer");
        assert_ne!(evidence_before.artifact_sha256, "unavailable");
        assert_eq!(evidence_before.artifact_sha256.len(), 64);
        assert_eq!(evidence_before.successful_launches, 0);

        let tokens = (0..128)
            .map(|index| 100 + (index * 17 % 1000) as u32)
            .collect::<Vec<_>>();

        let (mut chunked_kv, chunked_state, chunked_prefill_logits) =
            run_prefill_case(&model, &tokens, GdnPrefillBackend::FlashInfer, Some(64))?;
        let (mut unchunked_kv, unchunked_state, unchunked_prefill_logits) =
            run_prefill_case(&model, &tokens, GdnPrefillBackend::FlashInfer, None)?;

        // Temporary Stage 18 attribution inside the existing production gate.
        // This is not an additional acceptance test: it determines whether a
        // deterministic FlashInfer chunk mismatch is shared by the established
        // Triton implementation or belongs to one FlashInfer partition shape.
        let (_triton_chunked_kv, triton_chunked_state, _triton_chunked_logits) =
            run_prefill_case(&model, &tokens, GdnPrefillBackend::Triton, Some(64))?;
        let (_triton_unchunked_kv, triton_unchunked_state, _triton_unchunked_logits) =
            run_prefill_case(&model, &tokens, GdnPrefillBackend::Triton, None)?;

        report_layer_state_pair(
            &model,
            "diagnostic layer 0 FlashInfer unchunked vs chunked",
            &unchunked_state,
            &chunked_state,
            0,
        )?;
        report_layer_state_pair(
            &model,
            "diagnostic layer 0 Triton unchunked vs chunked",
            &triton_unchunked_state,
            &triton_chunked_state,
            0,
        )?;
        report_layer_state_pair(
            &model,
            "diagnostic layer 0 chunked Triton vs FlashInfer",
            &triton_chunked_state,
            &chunked_state,
            0,
        )?;
        report_layer_state_pair(
            &model,
            "diagnostic layer 0 unchunked Triton vs FlashInfer",
            &triton_unchunked_state,
            &unchunked_state,
            0,
        )?;

        assert_eq!(chunked_state.seq_len, 128);
        assert_eq!(unchunked_state.seq_len, 128);
        assert_recurrent_close(&model, &unchunked_state, &chunked_state)?;
        assert_logit_parity(
            "final prefill",
            &unchunked_prefill_logits,
            &chunked_prefill_logits,
        );

        let decode_token = 42;
        let unchunked_decode =
            first_decode_logits(&model, decode_token, &mut unchunked_kv, &unchunked_state)?;
        let chunked_decode =
            first_decode_logits(&model, decode_token, &mut chunked_kv, &chunked_state)?;
        assert_logit_parity("first decode", &unchunked_decode, &chunked_decode);

        let evidence_after = model.flashinfer_gdn_runtime_evidence()?;
        assert_eq!(evidence_after.selected_backend, "flashinfer");
        assert_eq!(
            evidence_after.artifact_sha256,
            evidence_before.artifact_sha256
        );
        let linear_layers =
            model.config().num_hidden_layers - model.config().num_full_attention_layers();
        assert_eq!(
            evidence_after.successful_launches - evidence_before.successful_launches,
            (3 * linear_layers) as u64,
            "chunk continuation gate did not execute two chunks plus one unchunked FlashInfer pass"
        );
        Ok(())
    }
}
