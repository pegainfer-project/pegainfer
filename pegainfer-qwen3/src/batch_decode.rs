//! Batched decode: process N requests' tokens in one forward pass.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::kv_pool::KvLayout;
use pegainfer_core::ops;
use pegainfer_kernels::tensor::KvDim;
use pegainfer_kernels::tensor::QDim;
use pegainfer_kv_cache::KvView;

use super::batch_decode_buffers::BATCH_BUCKETS;
use super::batch_decode_buffers::BatchDecodeBuffers;
use super::batch_decode_buffers::DecodeAttentionPath;
use super::batch_decode_buffers::bucket_for;
use super::batch_decode_dag::BatchDecodeDag;
use super::weights::PackedLoraProjection;
use super::weights::Qwen3Model;
use super::weights::TransformerBlock;
use crate::lora::LoraProjectionKind;

#[cfg(feature = "kernel-call-trace")]
macro_rules! dag_label {
    ($label:expr) => {
        $label.to_string()
    };
}

#[cfg(not(feature = "kernel-call-trace"))]
macro_rules! dag_label {
    ($label:expr) => {
        ()
    };
}

#[cfg(feature = "kernel-call-trace")]
macro_rules! trace_decode_kv_len {
    ($kv_len:expr, $body:block) => {
        ops::call_trace::with_decode_kv_len($kv_len, || $body)
    };
}

#[cfg(not(feature = "kernel-call-trace"))]
macro_rules! trace_decode_kv_len {
    ($kv_len:expr, $body:block) => {{ $body }};
}

/// How a `batch_decode` call interacts with the per-bucket CUDA graphs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DecodeGraphUse {
    /// Replay if captured, lazily capture otherwise (single-GPU serving).
    Serve,
    /// Eager kernels even with graphs enabled; the TP memory profile uses it to
    /// avoid an uncoordinated capture (see [`PrecapturePhase`]).
    Eager,
    /// Record + instantiate + upload, no launch (the TP sweep's `Capture`).
    CaptureOnly,
    /// Replay only; error if never captured (the TP sweep's `Launch`).
    Replay,
}

impl Qwen3Model {
    /// Batch decode step: N requests, 1 new token each, one forward pass.
    ///
    /// When `enable_cuda_graph` is set and the batch does not use LoRA, pads
    /// to the nearest bucket size and uses per-bucket CUDA Graph capture/replay.
    /// LoRA batches currently run eager because packed LoRA slots are not part
    /// of the CUDA Graph cache key yet.
    pub(crate) fn batch_decode(
        &self,
        token_ids: &[u32],
        kv_views: &[KvView],
        lora_adapters: &[Option<&str>],
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
        bufs: &mut BatchDecodeBuffers,
        graph_use: DecodeGraphUse,
    ) -> Result<()> {
        let bs = token_ids.len();
        assert_eq!(bs, kv_views.len());
        assert_eq!(bs, lora_adapters.len());
        assert!(bs > 0);
        // Before the first live-policy read (sync_paged_meta) and on every decode path; see
        // `policy_at_construction` for the workspace-overflow / stale-graph trap this guards.
        assert_eq!(
            pegainfer_kernels::ops::numeric_policy(),
            bufs.policy_at_construction,
            "NumericPolicy changed after executor construction (policy-key-trap); build a fresh executor per policy"
        );
        let lora_slots = self.decode_lora_slots(lora_adapters)?;
        let use_lora = lora_slots.is_some();
        let graphs_available = self.enable_cuda_graph && lora_slots.is_none();
        anyhow::ensure!(
            graphs_available || matches!(graph_use, DecodeGraphUse::Serve | DecodeGraphUse::Eager),
            "batch_decode {graph_use:?} requires CUDA graphs enabled and no LoRA rows"
        );
        let use_cuda_graph = graphs_available && graph_use != DecodeGraphUse::Eager;

        // Derive positions from views (seq_len - 1 = position of the new token)
        let mut positions: Vec<i32> = kv_views.iter().map(|v| (v.seq_len() - 1) as i32).collect();

        // Pad to bucket size for CUDA Graph stability
        let padded_bs = if use_cuda_graph { bucket_for(bs) } else { bs };

        // Set batch size on all buffers (padded — kernels run at bucket width)
        bufs.set_batch_size(padded_bs);

        // Sync metadata to GPU (pad token_ids/positions with 0 for padding slots)
        let mut token_ids_padded = token_ids.to_vec();
        token_ids_padded.resize(padded_bs, 0);
        positions.resize(padded_bs, 0);

        self.ctx
            .stream
            .memcpy_htod(&token_ids_padded, &mut bufs.token_ids_d)?;
        self.ctx
            .stream
            .memcpy_htod(&positions, &mut bufs.positions_d)?;
        if let Some(lora_slots) = &lora_slots {
            self.ctx
                .stream
                .memcpy_htod(lora_slots, &mut bufs.lora_token_slots_d)?;
        }

        let kv_refs: Vec<&KvView> = kv_views.iter().collect();
        bufs.sync_paged_meta(&self.ctx, &kv_refs, padded_bs)?;
        crate::green_ctx::fence_producers_before_override(&self.ctx)?;
        let attention_path =
            BatchDecodeBuffers::attention_path(padded_bs, bufs.policy_at_construction);
        #[cfg(feature = "kernel-call-trace")]
        let trace_kv_len = kv_views.iter().map(KvView::seq_len).max().unwrap_or(0);
        if use_cuda_graph {
            let bucket_idx = BATCH_BUCKETS.iter().position(|&b| b == padded_bs).unwrap();
            let graph_idx = BatchDecodeBuffers::graph_index(bucket_idx, attention_path);
            // Override-stream captures use the split cache; full-SM captures use the normal cache.
            let on_split_stream = pegainfer_kernels::tensor::has_stream_override();
            // Take graphs out of bufs to avoid split-borrow conflict with closure
            let mut graphs = if on_split_stream {
                std::mem::take(&mut bufs.graphs_split)
            } else {
                std::mem::take(&mut bufs.graphs)
            };
            let graph = &mut graphs[graph_idx];
            let kernels = || {
                trace_decode_kv_len!(trace_kv_len, {
                    self.batch_decode_kernels(
                        kv_buffer,
                        layout,
                        padded_bs,
                        attention_path,
                        use_lora,
                        bufs,
                    )
                })
            };
            let result = match graph_use {
                DecodeGraphUse::Serve => graph.run_or_capture(&self.ctx, kernels),
                DecodeGraphUse::CaptureOnly => graph.capture_only(&self.ctx, kernels),
                DecodeGraphUse::Replay => graph.launch_captured(&self.ctx),
                DecodeGraphUse::Eager => unreachable!("use_cuda_graph excludes Eager"),
            };
            if on_split_stream {
                bufs.graphs_split = graphs;
            } else {
                bufs.graphs = graphs;
            }
            result?;
        } else {
            trace_decode_kv_len!(trace_kv_len, {
                self.batch_decode_kernels(
                    kv_buffer,
                    layout,
                    padded_bs,
                    attention_path,
                    use_lora,
                    bufs,
                )
            })?;
        }

        Ok(())
    }

    fn batch_decode_kernels(
        &self,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &KvLayout,
        bs: usize,
        attention_path: DecodeAttentionPath,
        use_lora: bool,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        let num_layers = self.layers.len();
        let dag = BatchDecodeDag::new(self, kv_buffer, layout, bs, attention_path);

        // Embedding: N token_ids → hidden [hidden_dim, bs]
        dag.embedding(dag_label!("embedding"), &bufs.token_ids_d, &mut bufs.hidden)?;

        // First layer norm
        dag.rms_norm(
            dag_label!("input.rms_norm"),
            &bufs.hidden,
            &self.layers[0].input_layernorm,
            &mut bufs.normed,
        );

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            self.batch_decode_layer(layer_idx, layer, &dag, use_lora, bufs)?;

            let next_weight = if layer_idx + 1 < num_layers {
                &self.layers[layer_idx + 1].input_layernorm
            } else {
                &self.norm
            };
            // Without kernel-call-trace, dag_label! expands to `()` and the
            // branches collapse into identical unit blocks.
            #[cfg_attr(
                not(feature = "kernel-call-trace"),
                allow(
                    clippy::if_same_then_else,
                    clippy::let_unit_value,
                    clippy::semicolon_if_nothing_returned
                )
            )]
            let label = if layer_idx + 1 < num_layers {
                dag_label!(format!("L{layer_idx}.mlp.fused_add_rms_norm"))
            } else {
                dag_label!("final.rms_norm")
            };
            dag.fused_add_rms_norm(
                label,
                &mut bufs.hidden,
                &bufs.mlp_out,
                next_weight,
                &mut bufs.normed,
            )?;
        }

        // Output projection: logits [vocab_size, bs]
        dag.lm_head(
            dag_label!("lm_head"),
            self.output_projection(),
            &bufs.normed,
            &mut bufs.logits,
        );

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_decode_layer(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        dag: &BatchDecodeDag<'_>,
        use_lora: bool,
        bufs: &mut BatchDecodeBuffers,
    ) -> Result<()> {
        let q_dim = layer.attention.q_dim;
        let kv_dim = layer.attention.kv_dim;
        let fused_qkv = self.fused_decode_qkv();
        if fused_qkv {
            let qkv_out = bufs
                .qkv_out
                .as_mut()
                .expect("fused QKV plan requires qkv_out");
            dag.qkv_proj(
                dag_label!(format!("L{layer_idx}.attn.qkv_proj")),
                &layer.attention.qkv_proj,
                &bufs.normed,
                qkv_out,
            );
            dag.split_qkv(
                dag_label!(format!("L{layer_idx}.attn.split_qkv")),
                qkv_out,
                &mut bufs.q,
                &mut bufs.k,
                &mut bufs.v,
            )?;
        } else {
            // The split path is the numerical baseline retained by #75.
            dag.gemm_rows::<QDim>(
                dag_label!(format!("L{layer_idx}.attn.q_proj")),
                &layer.attention.qkv_proj,
                0,
                q_dim,
                &bufs.normed,
                &mut bufs.q,
            );
            dag.gemm_rows::<KvDim>(
                dag_label!(format!("L{layer_idx}.attn.k_proj")),
                &layer.attention.qkv_proj,
                q_dim,
                kv_dim,
                &bufs.normed,
                &mut bufs.k,
            );
            dag.gemm_rows::<KvDim>(
                dag_label!(format!("L{layer_idx}.attn.v_proj")),
                &layer.attention.qkv_proj,
                q_dim + kv_dim,
                kv_dim,
                &bufs.normed,
                &mut bufs.v,
            );
        }
        self.apply_decode_lora_projection_group3(
            layer_idx,
            LoraProjectionKind::Q,
            LoraProjectionKind::K,
            LoraProjectionKind::V,
            use_lora,
            &bufs.normed,
            &mut bufs.q,
            &mut bufs.k,
            &mut bufs.v,
            &bufs.lora_token_slots_d,
        )?;

        // QK norm + RoPE (batched, per-request positions)
        dag.qk_norm_rope(
            dag_label!(format!("L{layer_idx}.attn.qk_norm_rope")),
            &mut bufs.q,
            &mut bufs.k,
            &layer.attention.q_norm,
            &layer.attention.k_norm,
            &bufs.positions_d,
        )?;

        // KV append + paged attention decode (FlashInfer, batched)
        dag.paged_decode_attention(
            dag_label!(format!("L{layer_idx}.attn.paged_decode")),
            layer_idx,
            bufs,
        )?;

        // O projection (GEMM)
        dag.o_proj(
            dag_label!(format!("L{layer_idx}.attn.o_proj")),
            &layer.attention.o_proj,
            &bufs.attn_out,
            &mut bufs.attn_proj,
        );
        self.apply_decode_lora_projection(
            layer_idx,
            LoraProjectionKind::O,
            use_lora,
            &bufs.attn_out,
            &mut bufs.attn_proj,
            0,
            &bufs.lora_token_slots_d,
        )?;
        dag.all_reduce_hidden(
            dag_label!(format!("L{layer_idx}.attn.all_reduce")),
            &mut bufs.attn_proj,
        )?;

        // Residual + LayerNorm
        dag.fused_add_rms_norm(
            dag_label!(format!("L{layer_idx}.attn.fused_add_rms_norm")),
            &mut bufs.hidden,
            &bufs.attn_proj,
            &layer.post_attention_layernorm,
            &mut bufs.normed,
        )?;

        dag.mlp_gate_proj(
            dag_label!(format!("L{layer_idx}.mlp.gate_proj")),
            &layer.mlp.gate_up_proj,
            &bufs.normed,
            &mut bufs.gate_out,
        );
        dag.mlp_up_proj(
            dag_label!(format!("L{layer_idx}.mlp.up_proj")),
            &layer.mlp.gate_up_proj,
            &bufs.normed,
            &mut bufs.up_out,
        );
        self.apply_decode_lora_projection_group2(
            layer_idx,
            LoraProjectionKind::Gate,
            LoraProjectionKind::Up,
            use_lora,
            &bufs.normed,
            &mut bufs.gate_out,
            &mut bufs.up_out,
            &bufs.lora_token_slots_d,
        )?;
        dag.silu_mul_split(
            dag_label!(format!("L{layer_idx}.mlp.silu_mul")),
            &bufs.gate_out,
            &bufs.up_out,
            &mut bufs.mlp_act,
        )?;
        dag.down_proj(
            dag_label!(format!("L{layer_idx}.mlp.down_proj")),
            &layer.mlp.down_proj,
            &bufs.mlp_act,
            &mut bufs.mlp_out,
        );
        self.apply_decode_lora_projection(
            layer_idx,
            LoraProjectionKind::Down,
            use_lora,
            &bufs.mlp_act,
            &mut bufs.mlp_out,
            0,
            &bufs.lora_token_slots_d,
        )?;
        dag.all_reduce_hidden(
            dag_label!(format!("L{layer_idx}.mlp.all_reduce")),
            &mut bufs.mlp_out,
        )?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_decode_lora_projection_group2(
        &self,
        layer_idx: usize,
        kind0: LoraProjectionKind,
        kind1: LoraProjectionKind,
        use_lora: bool,
        input: &pegainfer_core::tensor::HiddenStates,
        out0: &mut pegainfer_core::tensor::HiddenStates,
        out1: &mut pegainfer_core::tensor::HiddenStates,
        token_slots: &CudaSlice<i32>,
    ) -> Result<()> {
        if !use_lora {
            return Ok(());
        }

        let p0 = self.decode_grouped_lora_projection(layer_idx, kind0, out0);
        let p1 = self.decode_grouped_lora_projection(layer_idx, kind1, out1);
        ops::lora_decode_fused_delta_group3_into(&self.ctx, token_slots, input, p0, p1, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_decode_lora_projection_group3(
        &self,
        layer_idx: usize,
        kind0: LoraProjectionKind,
        kind1: LoraProjectionKind,
        kind2: LoraProjectionKind,
        use_lora: bool,
        input: &pegainfer_core::tensor::HiddenStates,
        out0: &mut pegainfer_core::tensor::HiddenStates,
        out1: &mut pegainfer_core::tensor::HiddenStates,
        out2: &mut pegainfer_core::tensor::HiddenStates,
        token_slots: &CudaSlice<i32>,
    ) -> Result<()> {
        if !use_lora {
            return Ok(());
        }

        let p0 = self.decode_grouped_lora_projection(layer_idx, kind0, out0);
        let p1 = self.decode_grouped_lora_projection(layer_idx, kind1, out1);
        let p2 = self.decode_grouped_lora_projection(layer_idx, kind2, out2);
        ops::lora_decode_fused_delta_group3_into(&self.ctx, token_slots, input, p0, p1, p2)
    }

    fn decode_grouped_lora_projection<'a>(
        &'a self,
        layer_idx: usize,
        kind: LoraProjectionKind,
        out: &'a mut pegainfer_core::tensor::HiddenStates,
    ) -> Option<ops::LoraDecodeGroupedProjection<'a>> {
        let packed = self.packed_lora_projection(layer_idx, kind)?;
        Some(grouped_projection_from_packed(packed, out))
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_decode_lora_projection(
        &self,
        layer_idx: usize,
        kind: LoraProjectionKind,
        use_lora: bool,
        input: &pegainfer_core::tensor::HiddenStates,
        out: &mut pegainfer_core::tensor::HiddenStates,
        row_offset: usize,
        token_slots: &CudaSlice<i32>,
    ) -> Result<()> {
        if !use_lora {
            return Ok(());
        }

        let Some(packed) = self.packed_lora_projection(layer_idx, kind) else {
            return Ok(());
        };
        ops::lora_decode_fused_delta_into(
            &self.ctx,
            &packed.a,
            &packed.b,
            &packed.scales,
            token_slots,
            input,
            out,
            packed.max_loras,
            packed.max_rank,
            packed.rank,
            packed.out_dim,
            row_offset,
        )
    }
}

fn grouped_projection_from_packed<'a>(
    packed: &'a PackedLoraProjection,
    out: &'a mut pegainfer_core::tensor::HiddenStates,
) -> ops::LoraDecodeGroupedProjection<'a> {
    ops::LoraDecodeGroupedProjection {
        a_packed: &packed.a,
        b_packed: &packed.b,
        scales: &packed.scales,
        out,
        max_loras: packed.max_loras,
        max_rank: packed.max_rank,
        rank: packed.rank,
        out_dim: packed.out_dim,
    }
}
