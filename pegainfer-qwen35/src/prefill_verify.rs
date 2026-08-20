use anyhow::Result;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use pegainfer_core::kv_pool::KvState;

use crate::ffi;
use crate::ops;
use crate::prefill::PREFILL_CHUNK_LEN;
use crate::recurrent_state::RecurrentState;
use crate::verify_buffers::VerifyBuffers35;
use crate::weights::FullAttentionLayer;
use crate::weights::LayerKind;
use crate::weights::LinearAttentionLayer;
use crate::weights::Qwen35Model;
use crate::weights::TransformerBlock35;

impl Qwen35Model {
    pub(crate) fn prefill_verify_into(
        &self,
        spans: &[&[u32]],
        kv_states: &mut [&mut KvState],
        recurrent_states: &mut [&mut RecurrentState],
        bufs: &mut VerifyBuffers35,
    ) -> Result<()> {
        anyhow::ensure!(
            !pegainfer_kernels::tensor::has_stream_override(),
            "Qwen3.5 verify prefill does not support a CUDA stream override"
        );
        anyhow::ensure!(!spans.is_empty(), "Qwen3.5 verify needs at least one span");
        anyhow::ensure!(
            spans.len() == kv_states.len() && spans.len() == recurrent_states.len(),
            "Qwen3.5 verify spans/KV/recurrent mismatch: spans={}, kv={}, recurrent={}",
            spans.len(),
            kv_states.len(),
            recurrent_states.len()
        );
        let kv_buffer = kv_states[0].buffer();
        anyhow::ensure!(
            kv_states
                .iter()
                .all(|kv| std::ptr::eq(kv.buffer(), kv_buffer)),
            "Qwen3.5 verify KV states must share one pool"
        );
        anyhow::ensure!(
            spans.len() <= bufs.max_batch(),
            "Qwen3.5 verify batch {} exceeds buffer capacity {}",
            spans.len(),
            bufs.max_batch()
        );
        for span in spans {
            anyhow::ensure!(
                !span.is_empty() && span.len() <= PREFILL_CHUNK_LEN,
                "Qwen3.5 verify span len {} out of range",
                span.len()
            );
        }

        let total_rows = bufs.stage_tokens(&self.ctx, spans)?;
        let seq_lens: Vec<usize> = spans.iter().map(|span| span.len()).collect();
        let start_positions: Vec<usize> = kv_states.iter().map(|kv| kv.seq_len()).collect();
        for (kv, (&base_pos, &seq_len)) in kv_states
            .iter_mut()
            .zip(start_positions.iter().zip(seq_lens.iter()))
        {
            let end_pos = base_pos.checked_add(seq_len).ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen3.5 verify position overflow: base_pos={base_pos}, seq_len={seq_len}"
                )
            })?;
            anyhow::ensure!(
                end_pos <= self.config.max_position_embeddings,
                "Qwen3.5 verify requested end_pos={end_pos}, beyond max_position_embeddings={}",
                self.config.max_position_embeddings
            );
            self.ensure_rope_cache_covers(end_pos)?;
            kv.ensure_capacity(end_pos)?;
            kv.advance(seq_len);
        }

        let page_indices: Vec<Vec<i32>> =
            kv_states.iter().map(|kv| kv.page_indices_i32()).collect();
        let last_page_lens: Vec<usize> = kv_states.iter().map(|kv| kv.last_page_len()).collect();
        bufs.plan.update_batch_with_cta_tile_q(
            &self.ctx,
            &page_indices,
            &last_page_lens,
            &start_positions,
            &seq_lens,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.head_dim,
            0,
        )?;

        ops::embedding_batch(
            &self.ctx,
            &self.embed_tokens,
            &bufs.token_ids_d,
            &mut bufs.hidden,
        )?;

        let mut linear_idx = 0usize;
        let mut full_idx = 0usize;
        for layer in &self.layers {
            self.prefill_verify_layer_into(
                layer,
                &seq_lens,
                kv_states,
                recurrent_states,
                &mut linear_idx,
                &mut full_idx,
                bufs,
            )?;
        }

        for (recurrent, &seq_len) in recurrent_states.iter_mut().zip(seq_lens.iter()) {
            recurrent.seq_len += seq_len;
        }

        ops::rms_norm_batch_offset_into(
            &self.ctx,
            &bufs.hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut bufs.logits_normed,
        )?;
        ops::gemm_rows_into_checked(
            &self.ctx,
            self.output_projection(),
            0,
            self.config.selection_vocab,
            &bufs.logits_normed,
            &mut bufs.logits,
        )?;
        debug_assert_eq!(bufs.logits.seq_len, total_rows);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_verify_layer_into(
        &self,
        layer: &TransformerBlock35,
        seq_lens: &[usize],
        kv_states: &[&mut KvState],
        recurrent_states: &mut [&mut RecurrentState],
        linear_idx: &mut usize,
        full_idx: &mut usize,
        bufs: &mut VerifyBuffers35,
    ) -> Result<()> {
        ops::rms_norm_batch_offset_into(
            &self.ctx,
            &bufs.hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        match &layer.attn {
            LayerKind::FullAttention(attn) => {
                self.prefill_verify_full_attention_into(attn, kv_states, *full_idx, bufs)?;
                *full_idx += 1;
            }
            LayerKind::LinearAttention(attn) => {
                self.prefill_verify_linear_attention_into(
                    attn,
                    seq_lens,
                    recurrent_states,
                    *linear_idx,
                    bufs,
                )?;
                *linear_idx += 1;
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
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;
        ops::gemm_into_checked(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            &bufs.normed,
            &mut bufs.gate_up_out,
        )?;
        ops::silu_mul_fused_batch_into(&self.ctx, &bufs.gate_up_out, &mut bufs.act_out)?;
        ops::gemm_into_checked(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.mlp_out,
        )?;
        ops::add_batch_into(
            &self.ctx,
            &bufs.hidden_mid,
            &bufs.mlp_out,
            &mut bufs.hidden_next,
        )?;
        std::mem::swap(&mut bufs.hidden, &mut bufs.hidden_next);
        Ok(())
    }

    fn prefill_verify_full_attention_into(
        &self,
        attn: &FullAttentionLayer,
        kv_states: &[&mut KvState],
        full_idx: usize,
        bufs: &mut VerifyBuffers35,
    ) -> Result<()> {
        let c = &self.config;
        ops::gemm_into_checked(&self.ctx, &attn.q_proj, &bufs.normed, &mut bufs.q_full)?;
        ops::gemm_into_checked(&self.ctx, &attn.k_proj, &bufs.normed, &mut bufs.k_full)?;
        ops::gemm_into_checked(&self.ctx, &attn.v_proj, &bufs.normed, &mut bufs.v_full)?;

        ops::qk_norm_partial_rope_batched_decode_hd256_into(
            &self.ctx,
            &bufs.q_full,
            &mut bufs.q_prepped,
            &mut bufs.k_full,
            &attn.q_norm,
            &attn.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            bufs.plan.positions_d(),
            c.num_attention_heads,
            c.num_key_value_heads,
            c.rotary_dim,
            c.rms_norm_eps,
        )?;

        ops::paged_attention_batch_decode_via_prefill_hd256_into(
            &self.ctx,
            &bufs.q_prepped,
            &bufs.k_full,
            &bufs.v_full,
            kv_states[0].buffer(),
            kv_states[0].layout(),
            full_idx,
            &bufs.plan,
            &mut bufs.attn_out_full,
            c.num_attention_heads,
        )?;

        unsafe {
            let (qf_ptr, _gqf) = bufs.q_full.data.device_ptr(&self.ctx.stream);
            let (out_ptr, _go) = bufs.attn_out_full.data.device_ptr_mut(&self.ctx.stream);
            ffi::attention_gate_batch_hd256_cuda(
                qf_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                c.num_attention_heads as i32,
                bufs.q_prepped.seq_len as i32,
                self.ctx.stream.cu_stream(),
            );
        }
        ops::gemm_into_checked(
            &self.ctx,
            &attn.o_proj,
            &bufs.attn_out_full,
            &mut bufs.attn_results,
        )?;
        Ok(())
    }

    fn prefill_verify_linear_attention_into(
        &self,
        attn: &LinearAttentionLayer,
        seq_lens: &[usize],
        recurrent_states: &mut [&mut RecurrentState],
        linear_idx: usize,
        bufs: &mut VerifyBuffers35,
    ) -> Result<()> {
        let c = &self.config;
        ops::gemm_into_checked(&self.ctx, &attn.in_proj_qkv, &bufs.normed, &mut bufs.qkv)?;
        ops::gemm_into_checked(&self.ctx, &attn.in_proj_z, &bufs.normed, &mut bufs.z)?;
        ops::gemm_into_checked(&self.ctx, &attn.in_proj_b, &bufs.normed, &mut bufs.b_proj)?;
        ops::gemm_into_checked(&self.ctx, &attn.in_proj_a, &bufs.normed, &mut bufs.a_proj)?;

        let mut row_offset = 0usize;
        for (recurrent, &seq_len) in recurrent_states.iter_mut().zip(seq_lens.iter()) {
            let layer_state = &mut recurrent.layers[linear_idx];
            bufs.set_compact_rows(seq_len);
            ops::copy_hidden_token_range_into(
                &self.ctx,
                &bufs.qkv,
                row_offset,
                &mut bufs.compact_qkv,
                0,
                seq_len,
            )?;
            ops::conv1d_prefill_batch_into(
                &self.ctx,
                &bufs.compact_qkv,
                &attn.conv1d_weight,
                &mut layer_state.conv_state,
                &mut bufs.compact_qkv_conv,
                c.linear_conv_kernel_dim,
            );
            ops::copy_hidden_token_range_into(
                &self.ctx,
                &bufs.b_proj,
                row_offset,
                &mut bufs.compact_b,
                0,
                seq_len,
            )?;
            ops::copy_hidden_token_range_into(
                &self.ctx,
                &bufs.a_proj,
                row_offset,
                &mut bufs.compact_a,
                0,
                seq_len,
            )?;
            ops::gated_delta_rule_prefill_chunkwise_into(
                &self.ctx,
                &bufs.compact_qkv_conv,
                &bufs.compact_b,
                &bufs.compact_a,
                &attn.dt_bias,
                &attn.a_log,
                &mut layer_state.state,
                &mut bufs.gdr_scratch,
                &mut bufs.compact_gdr,
                c.linear_num_key_heads,
                c.linear_num_value_heads,
                c.linear_key_head_dim,
                c.linear_value_head_dim,
            )?;
            ops::copy_hidden_token_range_into(
                &self.ctx,
                &bufs.compact_gdr,
                0,
                &mut bufs.gdr_out,
                row_offset,
                seq_len,
            )?;
            row_offset += seq_len;
        }
        bufs.gdr_scratch.set_rows(bufs.qkv.seq_len);

        ops::rms_norm_gated_batch_into(
            &self.ctx,
            &bufs.gdr_out,
            &attn.norm_weight,
            &bufs.z,
            &mut bufs.normed_gated,
            c.linear_num_value_heads,
            c.linear_value_head_dim,
            c.rms_norm_eps,
        );
        ops::gemm_into_checked(
            &self.ctx,
            &attn.out_proj,
            &bufs.normed_gated,
            &mut bufs.attn_results,
        )?;
        Ok(())
    }
}
