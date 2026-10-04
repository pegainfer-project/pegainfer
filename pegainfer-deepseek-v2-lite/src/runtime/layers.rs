use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

use super::DeepSeekV2LiteEp2Generator;
use super::backend::EpBackendKind;
use super::types::GenerationStats;
use crate::attribution::DecodeAttributionProfile;
use crate::device::activate;
use crate::host_ops::DecodeCache;
use crate::host_ops::LayerCache;
use crate::host_ops::append_kv_and_build_queries;
use crate::host_ops::append_kv_and_build_queries_decode_batch;
use crate::host_ops::compute_attention_host;
use crate::host_ops::compute_attention_host_decode_batch;
use crate::host_ops::hidden_from_bf16_host;
use crate::host_ops::hidden_from_f32_host;
use crate::host_ops::hidden_to_f32;
use crate::host_ops::normalize_compressed_kv;
use crate::host_ops::rms_norm_hidden_host;
use crate::host_ops::rms_norm_host;
use crate::model::AttentionWeights;
use crate::model::MlpWeights;
use crate::model::dense_mlp_forward;
use crate::model::dense_mlp_forward_per_token;

impl DeepSeekV2LiteEp2Generator {
    pub(super) fn embed_tokens(&self, token_ids: &[u32]) -> Result<HiddenStates> {
        activate(&self.rank0.ctx)?;
        let token_ids_gpu = self.rank0.ctx.stream.clone_htod(token_ids)?;
        let mut out =
            HiddenStates::zeros(&self.rank0.ctx, self.config.hidden_size, token_ids.len())?;
        ops::embedding_batch(
            &self.rank0.ctx,
            &self.rank0.embed_tokens,
            &token_ids_gpu,
            &mut out,
        )?;
        Ok(out)
    }

    pub(super) fn forward_layers(
        &mut self,
        mut hidden: HiddenStates,
        start_pos: usize,
        cache: &mut DecodeCache,
        stats: &mut GenerationStats,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
    ) -> Result<HiddenStates> {
        ensure!(
            cache.layers.len() == self.rank0.layers.len(),
            "decode cache layer count mismatch"
        );
        let deferred_device_routes = self.begin_device_routed_forward(hidden.seq_len)?;
        for layer_idx in 0..self.rank0.layers.len() {
            hidden = self
                .forward_layer(
                    layer_idx,
                    &hidden,
                    start_pos,
                    &mut cache.layers[layer_idx],
                    stats,
                    attribution,
                    phase,
                    token_index,
                )
                .with_context(|| format!("DeepSeek-V2-Lite layer {layer_idx}"))?;
        }
        if deferred_device_routes {
            let (local_routes, remote_routes) = self.finish_device_routed_forward()?;
            stats.record_routes(EpBackendKind::Nccl, local_routes, remote_routes);
        }
        Ok(hidden)
    }

    pub(super) fn forward_layers_decode_batch(
        &mut self,
        mut hidden: HiddenStates,
        position: usize,
        caches: &mut [DecodeCache],
        stats: &mut GenerationStats,
        attribution: &mut DecodeAttributionProfile,
        token_index: usize,
    ) -> Result<HiddenStates> {
        ensure!(
            caches.len() == hidden.seq_len,
            "batched decode cache count {} must match hidden seq_len {}",
            caches.len(),
            hidden.seq_len
        );
        ensure!(
            caches
                .iter()
                .all(|cache| cache.layers.len() == self.rank0.layers.len()),
            "batched decode cache layer count mismatch"
        );
        let deferred_device_routes = self.begin_device_routed_forward(hidden.seq_len)?;
        for layer_idx in 0..self.rank0.layers.len() {
            let mut layer_caches: Vec<_> = caches
                .iter_mut()
                .map(|cache| &mut cache.layers[layer_idx])
                .collect();
            hidden = self
                .forward_layer_decode_batch(
                    layer_idx,
                    &hidden,
                    position,
                    &mut layer_caches,
                    stats,
                    attribution,
                    token_index,
                )
                .with_context(|| format!("DeepSeek-V2-Lite batched layer {layer_idx}"))?;
        }
        if deferred_device_routes {
            let (local_routes, remote_routes) = self.finish_device_routed_forward()?;
            stats.record_routes(EpBackendKind::Nccl, local_routes, remote_routes);
        }
        Ok(hidden)
    }

    fn forward_layer(
        &mut self,
        layer_idx: usize,
        hidden: &HiddenStates,
        start_pos: usize,
        cache: &mut LayerCache,
        stats: &mut GenerationStats,
        attribution: &mut DecodeAttributionProfile,
        phase: &'static str,
        token_index: Option<usize>,
    ) -> Result<HiddenStates> {
        activate(&self.rank0.ctx)?;

        let layer = &self.rank0.layers[layer_idx];
        let normed = attribution.record_result(
            phase,
            "host_rms_norm",
            || format!("layer.{layer_idx}.input_rms_norm"),
            Some(layer_idx),
            token_index,
            || self.rms_norm_hidden(hidden, &layer.input_layernorm_host),
        )?;

        let attn = attribution.record_result(
            phase,
            "attention_host_path",
            || format!("layer.{layer_idx}.attention_host_path"),
            Some(layer_idx),
            token_index,
            || self.attention_forward(&normed, &layer.attention, start_pos, cache),
        )?;
        activate(&self.rank0.ctx)?;
        let attn_projected = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "gpu_o_proj_enqueue",
            || format!("layer.{layer_idx}.attention_o_proj"),
            Some(layer_idx),
            token_index,
            || ops::gemm(&self.rank0.ctx, &layer.attention.o_proj, &attn),
        )?;
        let after_attn = attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "gpu_residual_add_enqueue",
            || format!("layer.{layer_idx}.attention_residual_add"),
            Some(layer_idx),
            token_index,
            || ops::add_batch(&self.rank0.ctx, hidden, &attn_projected),
        )?;

        let ffn_norm = attribution.record_result(
            phase,
            "host_rms_norm",
            || format!("layer.{layer_idx}.post_attention_rms_norm"),
            Some(layer_idx),
            token_index,
            || self.rms_norm_hidden(&after_attn, &layer.post_attention_layernorm_host),
        )?;

        let (ffn_out, local_routes, remote_routes) = match &layer.mlp {
            MlpWeights::Dense(dense) => (
                attribution.record_gpu_result(
                    &self.rank0.ctx,
                    phase,
                    "dense_mlp_enqueue",
                    || format!("layer.{layer_idx}.dense_mlp"),
                    Some(layer_idx),
                    token_index,
                    || dense_mlp_forward(&self.rank0.ctx, dense, &ffn_norm),
                )?,
                0,
                0,
            ),
            MlpWeights::Moe(moe) => {
                let (ffn_out, local_routes, remote_routes) = self.moe_forward(
                    layer_idx,
                    &ffn_norm,
                    moe,
                    attribution,
                    phase,
                    token_index,
                    false,
                )?;
                match self.backend.kind() {
                    EpBackendKind::HostStaged => {
                        stats.record_host_staged_moe(
                            ffn_norm.hidden_dim,
                            local_routes + remote_routes,
                        );
                    }
                    EpBackendKind::Nccl => {
                        stats.record_nccl_moe_collectives(ffn_norm.hidden_dim, ffn_norm.seq_len);
                    }
                }
                (ffn_out, local_routes, remote_routes)
            }
        };
        stats.record_routes(self.backend.kind(), local_routes, remote_routes);
        attribution.record_gpu_result(
            &self.rank0.ctx,
            phase,
            "gpu_residual_add_enqueue",
            || format!("layer.{layer_idx}.ffn_residual_add"),
            Some(layer_idx),
            token_index,
            || ops::add_batch(&self.rank0.ctx, &after_attn, &ffn_out),
        )
    }

    fn forward_layer_decode_batch(
        &mut self,
        layer_idx: usize,
        hidden: &HiddenStates,
        position: usize,
        caches: &mut [&mut LayerCache],
        stats: &mut GenerationStats,
        attribution: &mut DecodeAttributionProfile,
        token_index: usize,
    ) -> Result<HiddenStates> {
        ensure!(
            hidden.seq_len == caches.len(),
            "batched layer hidden seq_len {} must match cache rows {}",
            hidden.seq_len,
            caches.len()
        );
        activate(&self.rank0.ctx)?;

        let layer = &self.rank0.layers[layer_idx];
        let normed = attribution.record_result(
            "decode",
            "host_rms_norm",
            || format!("layer.{layer_idx}.batch_input_rms_norm"),
            Some(layer_idx),
            Some(token_index),
            || self.rms_norm_hidden(hidden, &layer.input_layernorm_host),
        )?;

        let attn = attribution.record_result(
            "decode",
            "attention_host_path",
            || format!("layer.{layer_idx}.batch_attention_host_path"),
            Some(layer_idx),
            Some(token_index),
            || self.attention_forward_decode_batch(&normed, &layer.attention, position, caches),
        )?;
        activate(&self.rank0.ctx)?;
        let attn_projected = attribution.record_gpu_result(
            &self.rank0.ctx,
            "decode",
            "gpu_o_proj_enqueue",
            || format!("layer.{layer_idx}.batch_attention_o_proj"),
            Some(layer_idx),
            Some(token_index),
            || ops::gemm_per_token(&self.rank0.ctx, &layer.attention.o_proj, &attn),
        )?;
        let after_attn = attribution.record_gpu_result(
            &self.rank0.ctx,
            "decode",
            "gpu_residual_add_enqueue",
            || format!("layer.{layer_idx}.batch_attention_residual_add"),
            Some(layer_idx),
            Some(token_index),
            || ops::add_batch(&self.rank0.ctx, hidden, &attn_projected),
        )?;

        let ffn_norm = attribution.record_result(
            "decode",
            "host_rms_norm",
            || format!("layer.{layer_idx}.batch_post_attention_rms_norm"),
            Some(layer_idx),
            Some(token_index),
            || self.rms_norm_hidden(&after_attn, &layer.post_attention_layernorm_host),
        )?;

        let (ffn_out, local_routes, remote_routes) = match &layer.mlp {
            MlpWeights::Dense(dense) => (
                attribution.record_gpu_result(
                    &self.rank0.ctx,
                    "decode",
                    "dense_mlp_enqueue",
                    || format!("layer.{layer_idx}.batch_dense_mlp"),
                    Some(layer_idx),
                    Some(token_index),
                    || dense_mlp_forward_per_token(&self.rank0.ctx, dense, &ffn_norm),
                )?,
                0,
                0,
            ),
            MlpWeights::Moe(moe) => {
                let (ffn_out, local_routes, remote_routes) = self.moe_forward(
                    layer_idx,
                    &ffn_norm,
                    moe,
                    attribution,
                    "decode",
                    Some(token_index),
                    true,
                )?;
                match self.backend.kind() {
                    EpBackendKind::HostStaged => {
                        stats.record_host_staged_moe(
                            ffn_norm.hidden_dim,
                            local_routes + remote_routes,
                        );
                    }
                    EpBackendKind::Nccl => {
                        stats.record_nccl_moe_collectives(ffn_norm.hidden_dim, ffn_norm.seq_len);
                    }
                }
                (ffn_out, local_routes, remote_routes)
            }
        };
        stats.record_routes(self.backend.kind(), local_routes, remote_routes);
        attribution.record_gpu_result(
            &self.rank0.ctx,
            "decode",
            "gpu_residual_add_enqueue",
            || format!("layer.{layer_idx}.batch_ffn_residual_add"),
            Some(layer_idx),
            Some(token_index),
            || ops::add_batch(&self.rank0.ctx, &after_attn, &ffn_out),
        )
    }

    fn attention_forward(
        &self,
        input: &HiddenStates,
        attn: &AttentionWeights,
        start_pos: usize,
        cache: &mut LayerCache,
    ) -> Result<HiddenStates> {
        activate(&self.rank0.ctx)?;
        ensure!(
            cache.len(&self.config) == start_pos,
            "attention cache position mismatch: cache_len={}, start_pos={start_pos}",
            cache.len(&self.config)
        );

        let q = ops::gemm(&self.rank0.ctx, &attn.q_proj, input)?;
        let kv_a = ops::gemm(&self.rank0.ctx, &attn.kv_a_proj, input)?;
        let q_host = hidden_to_f32(&self.rank0.ctx, &q)?;
        let kv_a_host = hidden_to_f32(&self.rank0.ctx, &kv_a)?;

        let compressed_norm = normalize_compressed_kv(
            &self.config,
            &kv_a_host,
            &attn.kv_a_norm_host,
            input.seq_len,
        );
        let compressed = hidden_from_bf16_host(
            &self.rank0.ctx,
            &compressed_norm,
            self.config.kv_lora_rank,
            input.seq_len,
        )?;
        activate(&self.rank0.ctx)?;
        let kv_b = ops::gemm(&self.rank0.ctx, &attn.kv_b_proj, &compressed)?;
        let kv_b_host = hidden_to_f32(&self.rank0.ctx, &kv_b)?;

        let mut queries =
            vec![
                0.0f32;
                input.seq_len * self.config.num_attention_heads * self.config.query_head_dim()
            ];
        append_kv_and_build_queries(
            &self.config,
            &q_host,
            &kv_a_host,
            &kv_b_host,
            start_pos,
            input.seq_len,
            &mut queries,
            cache,
        );

        let out_host =
            compute_attention_host(&self.config, &queries, cache, start_pos, input.seq_len);
        hidden_from_f32_host(
            &self.rank0.ctx,
            &out_host,
            self.config.o_proj_cols(),
            input.seq_len,
        )
    }

    fn attention_forward_decode_batch(
        &self,
        input: &HiddenStates,
        attn: &AttentionWeights,
        position: usize,
        caches: &mut [&mut LayerCache],
    ) -> Result<HiddenStates> {
        activate(&self.rank0.ctx)?;
        ensure!(
            input.seq_len == caches.len(),
            "batched attention input seq_len {} must match cache rows {}",
            input.seq_len,
            caches.len()
        );
        for (row, cache) in caches.iter().enumerate() {
            ensure!(
                cache.len(&self.config) == position,
                "batched attention cache row {row} position mismatch: cache_len={}, position={position}",
                cache.len(&self.config)
            );
        }

        let q = ops::gemm_per_token(&self.rank0.ctx, &attn.q_proj, input)?;
        let kv_a = ops::gemm_per_token(&self.rank0.ctx, &attn.kv_a_proj, input)?;
        let q_host = hidden_to_f32(&self.rank0.ctx, &q)?;
        let kv_a_host = hidden_to_f32(&self.rank0.ctx, &kv_a)?;

        let compressed_norm = normalize_compressed_kv(
            &self.config,
            &kv_a_host,
            &attn.kv_a_norm_host,
            input.seq_len,
        );
        let compressed = hidden_from_bf16_host(
            &self.rank0.ctx,
            &compressed_norm,
            self.config.kv_lora_rank,
            input.seq_len,
        )?;
        activate(&self.rank0.ctx)?;
        let kv_b = ops::gemm_per_token(&self.rank0.ctx, &attn.kv_b_proj, &compressed)?;
        let kv_b_host = hidden_to_f32(&self.rank0.ctx, &kv_b)?;

        let mut queries =
            vec![
                0.0f32;
                input.seq_len * self.config.num_attention_heads * self.config.query_head_dim()
            ];
        append_kv_and_build_queries_decode_batch(
            &self.config,
            &q_host,
            &kv_a_host,
            &kv_b_host,
            position,
            &mut queries,
            caches,
        );

        let cache_views: Vec<_> = caches.iter().map(|cache| &**cache as &LayerCache).collect();
        let out_host =
            compute_attention_host_decode_batch(&self.config, &queries, &cache_views, position);
        hidden_from_f32_host(
            &self.rank0.ctx,
            &out_host,
            self.config.o_proj_cols(),
            input.seq_len,
        )
    }

    pub(super) fn sample_last_token(&self, hidden: &HiddenStates) -> Result<u32> {
        self.sample_token_at(hidden, hidden.seq_len - 1)
    }

    pub(super) fn sample_tokens(&self, hidden: &HiddenStates) -> Result<Vec<u32>> {
        activate(&self.rank0.ctx)?;
        ensure!(hidden.seq_len != 0, "cannot sample an empty hidden batch");

        let normed = self.rms_norm_hidden(hidden, &self.rank0.norm_host)?;
        activate(&self.rank0.ctx)?;
        let logits = ops::gemm_per_token(&self.rank0.ctx, &self.rank0.lm_head, &normed)?;
        let mut values = self.rank0.ctx.stream.alloc_zeros(hidden.seq_len)?;
        let mut out = self.rank0.ctx.stream.alloc_zeros(hidden.seq_len)?;
        ops::argmax_batch_bf16_into(&self.rank0.ctx, &logits, &mut values, &mut out)?;

        let out_host = self.rank0.ctx.stream.clone_dtoh(&out)?;
        self.rank0
            .ctx
            .stream
            .synchronize()
            .context("sync sampled token D2H stream")?;

        out_host
            .into_iter()
            .map(|token| {
                ensure!(token >= 0, "argmax returned negative token id {token}");
                Ok(token as u32)
            })
            .collect()
    }

    fn sample_token_at(&self, hidden: &HiddenStates, row: usize) -> Result<u32> {
        activate(&self.rank0.ctx)?;
        ensure!(
            row < hidden.seq_len,
            "sample row {row} out of range for seq_len {}",
            hidden.seq_len
        );
        let state = ops::extract_vec(&self.rank0.ctx, hidden, row)?;
        let normed = self.rms_norm_vec(&state, &self.rank0.norm_host)?;
        let logits = ops::linear(&self.rank0.ctx, &normed, &self.rank0.lm_head)?;
        ops::argmax(&self.rank0.ctx, &logits)
    }

    fn rms_norm_hidden(&self, hidden: &HiddenStates, weight: &[f32]) -> Result<HiddenStates> {
        activate(&self.rank0.ctx)?;
        let input_host = hidden_to_f32(&self.rank0.ctx, hidden)?;
        let out = rms_norm_hidden_host(&self.config, &input_host, weight, hidden.seq_len);
        hidden_from_bf16_host(
            &self.rank0.ctx,
            &out,
            self.config.hidden_size,
            hidden.seq_len,
        )
    }

    fn rms_norm_vec(&self, input: &DeviceVec, weight: &[f32]) -> Result<DeviceVec> {
        activate(&self.rank0.ctx)?;
        let input_host = input.to_host(&self.rank0.ctx)?;
        let mut out = vec![bf16::ZERO; input.len];
        rms_norm_host(&input_host, weight, self.config.rms_norm_eps, &mut out);
        DeviceVec::from_host(&self.rank0.ctx, &out)
    }
}
