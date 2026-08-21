//! Unified forward pass: prefill + decode tokens in a single forward pass.
//!
//! GEMM ops process all tokens together. Tuned, and GQA groups with no compiled split-KV decode
//! kernel, use BatchPrefill for every row; Pin/PerToken route decode rows through the decode
//! attention ops, keeping a unified decode row on the same kernel path as a pure decode step.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::kv_pool::KvLayout;
use pegainfer_core::ops;
use pegainfer_core::ops::PrefillPagedPlan;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::numeric_policy;
use pegainfer_kv_cache::KvBuffer;
use pegainfer_kv_cache::KvView;

use super::batch_decode_buffers::BatchDecodeBuffers;
use super::config::PREFILL_ATTENTION_CTA_TILE_Q;
use super::prefill::PrefillBuffers;
use super::weights::Qwen3Model;
use super::weights::TransformerBlock;
use crate::lora::DeviceLoraTokenGroup;
use crate::lora::build_lora_token_ranges;
use crate::lora::prepare_lora_token_groups;

impl Qwen3Model {
    pub(crate) fn profile_unified_step_memory(
        &self,
        max_prefill_tokens: usize,
        profile_decode_rows: usize,
        kv_buffer: &KvBuffer,
        decode_bufs: &mut BatchDecodeBuffers,
        sample_scratch: &mut pegainfer_sample::SampleScratch,
        mark_peak: &mut impl FnMut() -> Result<()>,
    ) -> Result<()> {
        anyhow::ensure!(
            max_prefill_tokens > 0,
            "profile prefill tokens must be positive"
        );
        anyhow::ensure!(
            profile_decode_rows > 0,
            "profile decode rows must be positive"
        );

        let layout = KvLayout::new(
            kv_buffer.layout().num_layers,
            kv_buffer.layout().num_kv_heads,
            kv_buffer.layout().head_dim,
            kv_buffer.layout().page_size,
        )
        .expect("kv layout geometry");
        let page_size = layout.page_size;
        let num_prefill_reqs = 1;
        let prefill_pages = max_prefill_tokens.div_ceil(page_size);
        let decode_views: Vec<KvView> = (0..profile_decode_rows)
            .map(|i| KvView::new(vec![(1 + prefill_pages + i) as i32], 1, page_size))
            .collect();

        let decode_tokens = vec![0u32; profile_decode_rows];
        let decode_adapters = vec![None; profile_decode_rows];

        // Force the decode CUDA-Graph/buffer path before the unified peak
        // sample. The synthetic views are short, but the pre-allocated decode
        // arena and graph state are the same serving objects used later. Skip it
        // for uncompiled-group models: the unified sample below bounds their KV.
        // Eager under TP: ranks profile uncoordinated, so an in-profile capture
        // would hit the same deadlock the sweep avoids (see `PrecapturePhase`).
        if self.config.decode_group_is_compiled() {
            let graph_use = if self.tensor_parallel.world_size > 1 {
                crate::batch_decode::DecodeGraphUse::Eager
            } else {
                crate::batch_decode::DecodeGraphUse::Serve
            };
            self.batch_decode(
                &decode_tokens,
                &decode_views,
                &decode_adapters,
                kv_buffer.buffer(),
                &layout,
                decode_bufs,
                graph_use,
            )?;
            mark_peak()?;
        }

        // Reachable worst-case unified profile: admission caps
        // active decode rows + admitted/prefilling rows at the decode batch
        // capacity, while max_prefill_tokens caps total prefill tokens. One
        // max-sized prefill row plus the remaining decode slots exercises the
        // largest activation shape without inventing unreachable row count.
        let prefill_tokens_per_req = vec![0u32; max_prefill_tokens];
        let prefill_tokens_list: Vec<&[u32]> = vec![prefill_tokens_per_req.as_slice()];
        let prefill_single_views: Vec<KvView> = vec![KvView::new(
            (1..=prefill_pages).map(|page| page as i32).collect(),
            max_prefill_tokens,
            page_size,
        )];
        let prefill_adapters: Vec<Option<&str>> = vec![None; num_prefill_reqs];

        let logits = self.unified_step_with_peak(
            &prefill_tokens_list,
            &prefill_single_views,
            &prefill_adapters,
            &decode_tokens,
            &decode_views,
            &decode_adapters,
            decode_bufs,
            kv_buffer.buffer(),
            &layout,
            mark_peak,
        )?;
        mark_peak()?;

        let total_reqs = num_prefill_reqs + profile_decode_rows;
        let params = vec![SamplingParams::default(); total_reqs];
        let param_refs: Vec<&SamplingParams> = params.iter().collect();
        let steps = vec![0u64; param_refs.len()];
        let _ = pegainfer_sample::select_batch(
            self.device_ctx(),
            &logits,
            &param_refs,
            &steps,
            0,
            sample_scratch,
        )?;
        mark_peak()?;
        self.ctx.sync()?;
        Ok(())
    }

    /// Unified step: prefill + decode in one forward pass.
    ///
    /// Returns batched last-token logits `[vocab_size, n_prefill + n_decode]`:
    /// prefill request columns first (in request order), then decode columns.
    pub(crate) fn unified_step(
        &self,
        prefill_prompts: &[&[u32]],
        prefill_views: &[KvView],
        prefill_lora_adapters: &[Option<&str>],
        decode_tokens: &[u32],
        decode_views: &[KvView],
        decode_lora_adapters: &[Option<&str>],
        decode_bufs: &mut BatchDecodeBuffers,
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
    ) -> Result<HiddenStates> {
        let mut mark_peak = || Ok(());
        self.unified_step_with_peak(
            prefill_prompts,
            prefill_views,
            prefill_lora_adapters,
            decode_tokens,
            decode_views,
            decode_lora_adapters,
            decode_bufs,
            kv_buffer,
            layout,
            &mut mark_peak,
        )
    }

    fn unified_step_with_peak(
        &self,
        prefill_prompts: &[&[u32]],
        prefill_views: &[KvView],
        prefill_lora_adapters: &[Option<&str>],
        decode_tokens: &[u32],
        decode_views: &[KvView],
        decode_lora_adapters: &[Option<&str>],
        decode_bufs: &mut BatchDecodeBuffers,
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
        mark_peak: &mut dyn FnMut() -> Result<()>,
    ) -> Result<HiddenStates> {
        let num_prefill_reqs = prefill_prompts.len();
        let num_decode_reqs = decode_tokens.len();
        assert_eq!(num_prefill_reqs, prefill_views.len());
        assert_eq!(num_prefill_reqs, prefill_lora_adapters.len());
        assert_eq!(num_decode_reqs, decode_views.len());
        assert_eq!(num_decode_reqs, decode_lora_adapters.len());
        // Decode-only (empty prefill) is the fallback for GQA groups with no
        // compiled decode kernel.
        assert!(num_prefill_reqs > 0 || num_decode_reqs > 0);

        let prefill_seq_lens: Vec<usize> = prefill_prompts.iter().map(|p| p.len()).collect();
        let total_prefill: usize = prefill_seq_lens.iter().sum();
        let total_tokens = total_prefill + num_decode_reqs;
        let mut lora_ranges = build_lora_token_ranges(
            prefill_seq_lens.iter().copied(),
            prefill_lora_adapters.iter().copied(),
        );
        lora_ranges.extend(
            build_lora_token_ranges(
                std::iter::repeat_n(1, num_decode_reqs),
                decode_lora_adapters.iter().copied(),
            )
            .into_iter()
            .map(|mut range| {
                range.token_offset += total_prefill;
                range
            }),
        );
        let lora_groups = prepare_lora_token_groups(&self.ctx, &lora_ranges)?;

        // ── 1. Concatenate all tokens and get embeddings ──────────────
        let mut all_tokens: Vec<u32> = Vec::with_capacity(total_tokens);
        for prompt in prefill_prompts {
            all_tokens.extend_from_slice(prompt);
        }
        all_tokens.extend_from_slice(decode_tokens);
        let hidden = self.get_embeddings_batch(&all_tokens)?;
        mark_peak()?;

        // ── 2. Derive positions from views ────────────────────────────
        let prefill_start_positions: Vec<usize> = prefill_views
            .iter()
            .zip(prefill_seq_lens.iter())
            .map(|(v, &slen)| v.seq_len() - slen)
            .collect();

        let decode_positions: Vec<usize> = decode_views.iter().map(|v| v.seq_len() - 1).collect();

        // ── 3. Build metadata ─────────────────────────────────────────

        // Unconditional, as in `batch_decode`: the decode buffers' split workspace is sized for the
        // construction policy, and for an uncompiled-group model this is the only decode path.
        assert_eq!(
            numeric_policy(),
            decode_bufs.policy_at_construction,
            "NumericPolicy changed after executor construction (policy-key-trap); build a fresh executor per policy"
        );
        let split_decode_attention = matches!(
            numeric_policy(),
            NumericPolicy::Pin | NumericPolicy::PerToken
        ) && num_decode_reqs > 0
            && self.config.decode_group_is_compiled();
        let plan = if split_decode_attention {
            let positions: Vec<i32> = decode_positions.iter().map(|&pos| pos as i32).collect();
            self.ctx
                .stream
                .memcpy_htod(&positions, &mut decode_bufs.positions_d)?;
            let decode_refs: Vec<&KvView> = decode_views.iter().collect();
            decode_bufs.sync_paged_meta(&self.ctx, &decode_refs, num_decode_reqs)?;

            if num_prefill_reqs > 0 {
                let page_indices: Vec<Vec<i32>> = prefill_views
                    .iter()
                    .map(|v| v.page_indices().to_vec())
                    .collect();
                let last_page_lens: Vec<usize> = prefill_views
                    .iter()
                    .map(pegainfer_kv_cache::KvView::last_page_len)
                    .collect();
                Some(PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
                    &self.ctx,
                    &page_indices,
                    &last_page_lens,
                    &prefill_start_positions,
                    &prefill_seq_lens,
                    self.local_num_attention_heads(),
                    self.local_num_key_value_heads(),
                    self.config.head_dim,
                    PREFILL_ATTENTION_CTA_TILE_Q,
                )?)
            } else {
                None
            }
        } else {
            // One attention plan over prefill requests + decode rows (qo_len=1,
            // start at the decode position so the row attends its full history).
            let page_indices: Vec<Vec<i32>> = prefill_views
                .iter()
                .chain(decode_views.iter())
                .map(|v| v.page_indices().to_vec())
                .collect();
            let last_page_lens: Vec<usize> = prefill_views
                .iter()
                .chain(decode_views.iter())
                .map(pegainfer_kv_cache::KvView::last_page_len)
                .collect();
            let mut start_positions = prefill_start_positions;
            start_positions.extend_from_slice(&decode_positions);
            let mut seq_lens = prefill_seq_lens.clone();
            seq_lens.extend(std::iter::repeat_n(1, num_decode_reqs));
            Some(PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
                &self.ctx,
                &page_indices,
                &last_page_lens,
                &start_positions,
                &seq_lens,
                self.local_num_attention_heads(),
                self.local_num_key_value_heads(),
                self.config.head_dim,
                PREFILL_ATTENTION_CTA_TILE_Q,
            )?)
        };
        mark_peak()?;

        // ── 4. Process layers ─────────────────────────────────────────
        let hidden = self.unified_layers_with_peak(
            hidden,
            total_tokens,
            plan.as_ref(),
            decode_bufs,
            split_decode_attention,
            total_prefill,
            num_decode_reqs,
            &lora_groups,
            kv_buffer,
            layout,
            mark_peak,
        )?;

        // ── 5. Extract logits ─────────────────────────────────────────
        // Last token of each prefill sequence, then every decode token —
        // one gather + one batched lm_head GEMM for the whole step.
        let mut last_indices = Vec::with_capacity(num_prefill_reqs + num_decode_reqs);
        let mut offset = 0usize;
        for &seq_len in &prefill_seq_lens {
            last_indices.push((offset + seq_len - 1) as i32);
            offset += seq_len;
        }
        for i in 0..num_decode_reqs {
            last_indices.push((total_prefill + i) as i32);
        }
        let logits = self.batch_token_logits(&hidden, &last_indices)?;
        mark_peak()?;
        Ok(logits)
    }

    fn unified_layers_with_peak(
        &self,
        mut hidden: HiddenStates,
        total_tokens: usize,
        plan: Option<&PrefillPagedPlan>,
        decode_bufs: &mut BatchDecodeBuffers,
        split_decode_attention: bool,
        total_prefill: usize,
        num_decode: usize,
        lora_groups: &[DeviceLoraTokenGroup<'_>],
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
        mark_peak: &mut dyn FnMut() -> Result<()>,
    ) -> Result<HiddenStates> {
        let inter_dim = self.local_intermediate_size();
        let q_dim = self.local_q_dim();
        let kv_dim = self.local_kv_dim();

        let mut bufs = PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            total_tokens,
        )?;
        mark_peak()?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.unified_forward_layer(
                layer_idx,
                layer,
                &mut hidden,
                &mut bufs,
                plan,
                decode_bufs,
                split_decode_attention,
                total_prefill,
                num_decode,
                lora_groups,
                kv_buffer,
                layout,
            )?;
        }
        mark_peak()?;

        Ok(hidden)
    }

    #[allow(clippy::too_many_arguments)]
    fn unified_forward_layer(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        bufs: &mut PrefillBuffers,
        plan: Option<&PrefillPagedPlan>,
        decode_bufs: &mut BatchDecodeBuffers,
        split_decode_attention: bool,
        total_prefill: usize,
        num_decode: usize,
        lora_groups: &[DeviceLoraTokenGroup<'_>],
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
    ) -> Result<()> {
        let num_heads = self.local_num_attention_heads();
        let num_kv_heads = self.local_num_key_value_heads();
        let head_dim = self.config.head_dim;

        // ── 1. RMSNorm → normed [all tokens] ─────────────────────────
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );

        // ── 2. QKV projections from fused qkv_proj [all tokens] ─────
        let q_dim_l = layer.attention.q_dim;
        let kv_dim_l = layer.attention.kv_dim;
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            0,
            q_dim_l,
            &bufs.normed,
            &mut bufs.q_batch,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.q_proj.as_ref(),
            &bufs.normed,
            &mut bufs.q_batch,
            0,
        )?;
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim_l,
            kv_dim_l,
            &bufs.normed,
            &mut bufs.k_batch,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.k_proj.as_ref(),
            &bufs.normed,
            &mut bufs.k_batch,
            0,
        )?;
        ops::gemm_rows_into(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim_l + kv_dim_l,
            kv_dim_l,
            &bufs.normed,
            &mut bufs.v_batch,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.v_proj.as_ref(),
            &bufs.normed,
            &mut bufs.v_batch,
            0,
        )?;

        // `plan` is prefill-only under split routing (None on a decode-only step), all-row otherwise.
        if let Some(plan) = plan {
            ops::prefill_attention_paged_into(
                &self.ctx,
                &mut bufs.q_batch,
                &mut bufs.k_batch,
                &bufs.v_batch,
                &layer.attention.q_norm,
                &layer.attention.k_norm,
                &self.cos_cache,
                &self.sin_cache,
                kv_buffer,
                layout,
                layer_idx,
                plan,
                &mut bufs.attn_output,
                num_heads,
                num_kv_heads,
                head_dim,
                self.config.rms_norm_eps,
            )?;
        }
        if split_decode_attention {
            ops::qk_norm_rope_batch_decode_into(
                &self.ctx,
                &mut bufs.q_batch,
                &mut bufs.k_batch,
                total_prefill,
                num_decode,
                &layer.attention.q_norm,
                &layer.attention.k_norm,
                &self.cos_cache,
                &self.sin_cache,
                &decode_bufs.positions_d,
                num_heads,
                num_kv_heads,
                head_dim,
                self.config.rms_norm_eps,
            )?;
            ops::paged_attention_batch_decode_split_kv_into(
                &self.ctx,
                &bufs.q_batch,
                &bufs.k_batch,
                &bufs.v_batch,
                total_prefill,
                kv_buffer,
                layout,
                layer_idx,
                &decode_bufs.page_indices_d,
                &decode_bufs.page_indptr_d,
                &decode_bufs.last_page_len_d,
                &decode_bufs.positions_d,
                &decode_bufs.request_indices_d,
                &decode_bufs.split_request_indices_d,
                &decode_bufs.split_kv_tile_indices_d,
                &decode_bufs.split_kv_chunk_size_d,
                &decode_bufs.split_o_indptr_d,
                &decode_bufs.split_block_valid_mask_d,
                &mut decode_bufs.split_tmp_v,
                &mut decode_bufs.split_tmp_s,
                decode_bufs.split_padded_slots,
                &mut bufs.attn_output,
                num_heads,
                num_decode,
            )?;
        }

        // ── 6. O projection [all tokens] ─────────────────────────────
        ops::gemm_into(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.o_proj.as_ref(),
            &bufs.attn_output,
            &mut bufs.o_buf,
            0,
        )?;
        self.all_reduce_hidden(&mut bufs.o_buf)?;

        // ── 7+8. Residual add + MLP RMSNorm (fused) ─────────────────
        pegainfer_kernels::ops::fused_add_rms_norm_round_batch_into(
            &self.ctx,
            hidden,
            &bufs.o_buf,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        ops::gemm_rows_into(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            0,
            self.local_intermediate_size(),
            &bufs.normed,
            &mut bufs.gate_out,
        );
        ops::gemm_rows_into(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            self.local_intermediate_size(),
            self.local_intermediate_size(),
            &bufs.normed,
            &mut bufs.up_out,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.gate_proj.as_ref(),
            &bufs.normed,
            &mut bufs.gate_out,
            0,
        )?;
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.up_proj.as_ref(),
            &bufs.normed,
            &mut bufs.up_out,
            0,
        )?;
        ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
        ops::gemm_into(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        );
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.down_proj.as_ref(),
            &bufs.act_out,
            &mut bufs.o_buf,
            0,
        )?;
        self.all_reduce_hidden(&mut bufs.o_buf)?;

        // ── 9. Residual add → hidden_out ─────────────────────────────
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        std::mem::swap(hidden, &mut bufs.hidden_out);

        Ok(())
    }
}
