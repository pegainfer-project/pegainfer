use std::any::Any;

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::kv_pool::KvLayout;
use pegainfer_core::ops;
use pegainfer_core::ops::PrefillPagedPlan;
use pegainfer_core::tensor::DeviceVec;

use super::config::PREFILL_ATTENTION_CTA_TILE_Q;
use super::weights::Qwen3Model;
use super::weights::TransformerBlock;
use crate::lora::DeviceLoraTokenGroup;
use crate::lora::build_lora_token_ranges;
use crate::lora::prepare_lora_token_groups;

// Prefill temporaries free on ctx.stream but are consumed by override-stream
// kernels; `park()` defers them into the open parking window until that
// stream syncs. No window open (non-overlap) → drops in place.
thread_local! {
    static PREFILL_TEMP_WINDOW: std::cell::RefCell<Option<Vec<Box<dyn Any>>>> =
        const { std::cell::RefCell::new(None) };
}

fn park<T: 'static>(val: T) {
    PREFILL_TEMP_WINDOW.with(|w| {
        if let Some(items) = w.borrow_mut().as_mut() {
            items.push(Box::new(val));
        }
    });
}

/// Owns parked prefill temporaries until the override stream is synchronized.
pub(crate) struct PrefillTempBin {
    stream: Option<cudarc::driver::sys::CUstream>,
    items: Vec<Box<dyn Any>>,
}

impl PrefillTempBin {
    pub(crate) fn armed(stream: cudarc::driver::sys::CUstream) -> Self {
        PREFILL_TEMP_WINDOW.with(|w| *w.borrow_mut() = Some(Vec::new()));
        Self {
            stream: Some(stream),
            items: Vec::new(),
        }
    }

    pub(crate) fn close(&mut self) {
        if let Some(mut captured) = PREFILL_TEMP_WINDOW.with(|w| w.borrow_mut().take()) {
            self.items.append(&mut captured);
        }
    }

    /// Drains the prefill stream; aborts if synchronization fails.
    pub(crate) fn synchronize(&mut self) {
        self.close();
        if let Some(stream) = self.stream {
            let r = unsafe { cudarc::driver::sys::cuStreamSynchronize(stream) };
            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                log::error!(
                    "FATAL: cuStreamSynchronize(prefill) failed ({r:?}); aborting rather than \
                     free buffers the prefill stream may still be reading"
                );
                std::process::abort();
            }
            self.stream = None;
            self.items.clear();
        }
    }
}

impl Drop for PrefillTempBin {
    fn drop(&mut self) {
        self.synchronize();
    }
}
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kv_cache::KvView;

/// Pre-allocated scratch buffers for one prefill forward pass.
/// Created once per prefill pass, eliminating
/// per-layer `cuMemAllocAsync` overhead (~11k calls / 88ms at seq=2048).
///
/// Buffer reuse across steps (all kernels serialized on a single stream):
///   `normed`  reused for `normed2`  (steps 1-4 done before step 8)
///   `o_buf`   reused for `mlp_out`  (step 7 done before step 12)
pub(super) struct PrefillBuffers {
    /// Output ping-pong: layer writes result here; caller swaps with the incoming hidden.
    pub(super) hidden_out: HiddenStates, // hidden_dim × seq_len
    pub(super) normed: HiddenStates, // hidden_dim × seq_len (reused for normed2)
    pub(super) q_batch: HiddenStates, // q_dim × seq_len
    pub(super) k_batch: HiddenStates, // kv_dim × seq_len
    pub(super) v_batch: HiddenStates, // kv_dim × seq_len
    pub(super) o_buf: HiddenStates,  // hidden_dim × seq_len (reused for mlp_out)
    pub(super) gate_out: HiddenStates, // inter_dim × seq_len
    pub(super) up_out: HiddenStates, // inter_dim × seq_len
    pub(super) act_out: HiddenStates, // inter_dim × seq_len
    pub(super) attn_output: HiddenStates, // q_dim × seq_len
}

/// One named last-token snapshot from a selected prefill layer.
pub(crate) struct PrefillStageSnapshot {
    pub(crate) name: String,
    pub(crate) values: DeviceVec,
}

impl PrefillBuffers {
    pub(super) fn new(
        ctx: &DeviceContext,
        hidden_dim: usize,
        q_dim: usize,
        kv_dim: usize,
        inter_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        Ok(Self {
            hidden_out: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            normed: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, seq_len)?,
            k_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            v_batch: HiddenStates::zeros(ctx, kv_dim, seq_len)?,
            o_buf: HiddenStates::zeros(ctx, hidden_dim, seq_len)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, seq_len)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, seq_len)?,
        })
    }

    /// Point every scratch buffer's logical row count at `rows` without
    /// reallocating. Used by the fixed-buffer verify path (see
    /// [`crate::verify_graph`]); the buffers must have been allocated for at
    /// least `rows`.
    pub(super) fn set_rows(&mut self, rows: usize) {
        self.hidden_out.seq_len = rows;
        self.normed.seq_len = rows;
        self.q_batch.seq_len = rows;
        self.k_batch.seq_len = rows;
        self.v_batch.seq_len = rows;
        self.o_buf.seq_len = rows;
        self.gate_out.seq_len = rows;
        self.up_out.seq_len = rows;
        self.act_out.seq_len = rows;
        self.attn_output.seq_len = rows;
    }
}

impl Qwen3Model {
    fn prefill_plan_for_single_prompt(
        &self,
        prompt: &[u32],
        kv_view: &KvView,
    ) -> Result<PrefillPagedPlan> {
        anyhow::ensure!(!prompt.is_empty(), "prompt must not be empty");
        let start_position = kv_view.seq_len() - prompt.len();
        PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
            &self.ctx,
            &[kv_view.page_indices().to_vec()],
            &[kv_view.last_page_len()],
            &[start_position],
            &[prompt.len()],
            self.local_num_attention_heads(),
            self.local_num_key_value_heads(),
            self.config.head_dim,
            PREFILL_ATTENTION_CTA_TILE_Q,
        )
    }

    pub(super) fn get_embeddings_batch(&self, token_ids: &[u32]) -> Result<HiddenStates> {
        let seq_len = token_ids.len();
        let hidden_dim = self.config.hidden_size;

        let token_ids_gpu = self
            .ctx
            .stream
            .clone_htod(token_ids)
            .map_err(|e| anyhow::anyhow!("H2D copy failed: {}", e))?;

        let mut out = HiddenStates::zeros(&self.ctx, hidden_dim, seq_len)?;
        crate::green_ctx::fence_producers_before_override(&self.ctx)?;
        let launched =
            ops::embedding_batch(&self.ctx, &self.embed_tokens, &token_ids_gpu, &mut out);
        park(token_ids_gpu);
        if let Err(e) = launched {
            park(out);
            return Err(e);
        }

        Ok(out)
    }

    /// Embed a device-resident token buffer into a pre-allocated output, with
    /// no host round-trip or allocation — used by the DFlash draft rollout's
    /// graph-stable scratch.
    pub(super) fn get_embeddings_batch_into(
        &self,
        token_ids_gpu: &cudarc::driver::CudaSlice<u32>,
        out: &mut HiddenStates,
    ) -> Result<()> {
        anyhow::ensure!(
            out.hidden_dim == self.config.hidden_size,
            "embedding output hidden_dim {} does not match model hidden_size {}",
            out.hidden_dim,
            self.config.hidden_size
        );
        ops::embedding_batch(&self.ctx, &self.embed_tokens, token_ids_gpu, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_batch_paged(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &pegainfer_core::kv_pool::KvLayout,
        plan: &PrefillPagedPlan,
        lora_groups: &[DeviceLoraTokenGroup<'_>],
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        self.forward_layer_pre_attn(layer_idx, layer, hidden, lora_groups, bufs)?;
        self.forward_layer_attn(layer_idx, layer, kv_buffer, layout, plan, bufs)?;
        self.forward_layer_post_attn(layer_idx, layer, hidden, lora_groups, bufs)?;
        Ok(())
    }

    /// Pre-attention dense ops: input RMSNorm + fused QKV projections (+ LoRA).
    /// Reads `hidden`; writes `bufs.normed` / `bufs.q_batch` / `bufs.k_batch` /
    /// `bufs.v_batch`. Graph-safe — shapes depend only on the fixed row count, not
    /// on KV length — so the verify piecewise CUDA Graph captures it.
    pub(crate) fn forward_layer_pre_attn(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &HiddenStates,
        lora_groups: &[DeviceLoraTokenGroup<'_>],
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        // 1. RMSNorm → bufs.normed
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &layer.input_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        );

        // 2. QKV projections from fused qkv_proj
        let q_dim = layer.attention.q_dim;
        let kv_dim = layer.attention.kv_dim;
        ops::gemm_rows_into_checked(
            &self.ctx,
            &layer.attention.qkv_proj,
            0,
            q_dim,
            &bufs.normed,
            &mut bufs.q_batch,
        )?;
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.q_proj.as_ref(),
            &bufs.normed,
            &mut bufs.q_batch,
            0,
        )?;
        ops::gemm_rows_into_checked(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.k_batch,
        )?;
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.k_proj.as_ref(),
            &bufs.normed,
            &mut bufs.k_batch,
            0,
        )?;
        ops::gemm_rows_into_checked(
            &self.ctx,
            &layer.attention.qkv_proj,
            q_dim + kv_dim,
            kv_dim,
            &bufs.normed,
            &mut bufs.v_batch,
        )?;
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.v_proj.as_ref(),
            &bufs.normed,
            &mut bufs.v_batch,
            0,
        )?;
        Ok(())
    }

    /// The attention op: q/k norm + RoPE + paged KV append + paged attention.
    /// This is the ONLY part of the layer whose KV iteration count tracks the
    /// (growing) context length, so the verify piecewise graph keeps it EAGER —
    /// capturing it would freeze the KV length at capture time (`num_iterations`
    /// in FlashInfer's prefill kernel is fixed when the graph is recorded).
    pub(crate) fn forward_layer_attn(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        layout: &pegainfer_core::kv_pool::KvLayout,
        plan: &PrefillPagedPlan,
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        let num_heads = self.local_num_attention_heads();
        let num_kv_heads = self.local_num_key_value_heads();
        let head_dim = self.config.head_dim;

        // 3. Paged prefill: norm+RoPE → append K/V to paged → batch attention
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
        Ok(())
    }

    /// Post-attention dense ops: O projection + residual + MLP + final residual
    /// add. Reads `bufs.attn_output` and `hidden`; writes the layer output back
    /// into `hidden` via the ping-pong buffer swap. Graph-safe (no KV-length
    /// dependence) — captured into the verify piecewise CUDA Graph.
    pub(crate) fn forward_layer_post_attn(
        &self,
        layer_idx: usize,
        layer: &TransformerBlock,
        hidden: &mut HiddenStates,
        lora_groups: &[DeviceLoraTokenGroup<'_>],
        bufs: &mut PrefillBuffers,
    ) -> Result<()> {
        // 4. O projection → bufs.o_buf (as o_batch)
        ops::gemm_into_checked(
            &self.ctx,
            &layer.attention.o_proj,
            &bufs.attn_output,
            &mut bufs.o_buf,
        )?;
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.o_proj.as_ref(),
            &bufs.attn_output,
            &mut bufs.o_buf,
            0,
        )?;
        self.all_reduce_hidden(&mut bufs.o_buf)?;

        // 5+6. Residual add + MLP RMSNorm (fused): hidden += o_buf; normed = rms_norm(hidden)
        pegainfer_kernels::ops::fused_add_rms_norm_round_batch_into(
            &self.ctx,
            hidden,
            &bufs.o_buf,
            &layer.post_attention_layernorm,
            self.config.rms_norm_eps,
            &mut bufs.normed,
        )?;

        // 7. MLP: split gate/up GEMMs → silu_mul → down → bufs.o_buf
        let inter_dim = self.local_intermediate_size();
        ops::gemm_rows_into_checked(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            0,
            inter_dim,
            &bufs.normed,
            &mut bufs.gate_out,
        )?;
        ops::gemm_rows_into_checked(
            &self.ctx,
            &layer.mlp.gate_up_proj,
            inter_dim,
            inter_dim,
            &bufs.normed,
            &mut bufs.up_out,
        )?;
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
        ops::gemm_into_checked(
            &self.ctx,
            &layer.mlp.down_proj,
            &bufs.act_out,
            &mut bufs.o_buf,
        )?;
        self.apply_lora_projection_ranges(
            layer_idx,
            lora_groups,
            |layer| layer.down_proj.as_ref(),
            &bufs.act_out,
            &mut bufs.o_buf,
            0,
        )?;
        self.all_reduce_hidden(&mut bufs.o_buf)?;

        // 8. Residual add: attn_residual + mlp_out → bufs.hidden_out (old hidden_in, free to overwrite)
        ops::add_batch_into(&self.ctx, hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
        // Swap: hidden = layer output, bufs.hidden_out = attn_residual (free next layer)
        std::mem::swap(hidden, &mut bufs.hidden_out);

        Ok(())
    }

    // ── Batch prefill ──────────────────────────────────────────────────

    /// Batch prefill: process multiple prompts in a single forward pass.
    ///
    /// Compute logits for ALL positions in the hidden states.
    ///
    /// Used when `echo=true` to return prompt token log-probabilities.
    /// Applies final RMS norm + lm_head projection in a single batched GEMM.
    /// Returns `HiddenStates` with shape `[vocab_size, total_tokens]`.
    fn compute_all_position_logits(&self, hidden: &HiddenStates) -> Result<HiddenStates> {
        let mut normed = HiddenStates::zeros(&self.ctx, hidden.hidden_dim, hidden.seq_len)?;
        ops::rms_norm_batch_into(
            &self.ctx,
            hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut normed,
        );
        ops::gemm(&self.ctx, self.output_projection(), &normed)
    }

    /// Batched last-token logits: gather the given token columns out of
    /// `hidden`, then apply final RMSNorm and lm_head as single batched ops.
    /// Returns `HiddenStates [vocab_size, n]`, one column per index.
    pub(crate) fn batch_token_logits(
        &self,
        hidden: &HiddenStates,
        token_indices: &[i32],
    ) -> Result<HiddenStates> {
        let n = token_indices.len();
        // Allocate all buffers up front so one producer fence orders them ahead
        // of the override-stream gather/norm/GEMM.
        let indices_d = self.ctx.stream.clone_htod(token_indices)?;
        let mut gathered = HiddenStates::zeros(&self.ctx, hidden.hidden_dim, n)?;
        let mut normed = HiddenStates::zeros(&self.ctx, hidden.hidden_dim, n)?;
        let mut logits = HiddenStates::zeros(&self.ctx, self.output_projection().rows, n)?;
        crate::green_ctx::fence_producers_before_override(&self.ctx)?;

        let gather =
            ops::gather_hidden_tokens_into(&self.ctx, hidden, &indices_d, n, &mut gathered);
        park(indices_d);
        if let Err(e) = gather {
            park(gathered);
            return Err(e);
        }
        ops::rms_norm_batch_into(
            &self.ctx,
            &gathered,
            &self.norm,
            self.config.rms_norm_eps,
            &mut normed,
        );
        let gemm =
            ops::gemm_into_checked(&self.ctx, self.output_projection(), &normed, &mut logits);
        park(gathered);
        park(normed);
        if let Err(e) = gemm {
            park(logits);
            return Err(e);
        }
        Ok(logits)
    }

    /// Concatenates all prompts' tokens, runs one GEMM per layer for the
    /// entire batch, and uses FlashInfer's multi-request causal attention.
    /// Returns batched last-token logits `[vocab_size, batch]`, one column
    /// per request.
    ///
    /// If `echo` is true, also returns all-position logits as a
    /// `HiddenStates [vocab_size, total_tokens]` for prompt logprobs.
    /// Batch prefill forward.
    ///
    /// `capture_layer_ids`, when set, copies the residual-stream hidden states
    /// after the listed (strictly increasing) transformer layers into an extra
    /// `[hidden_size * layers, total_tokens]` buffer returned as the third tuple
    /// element. This feeds the DFlash draft model its target context; `None`
    /// behaves identically to a plain prefill and returns `None` there.
    pub(crate) fn batch_prefill(
        &self,
        prompts: &[&[u32]],
        kv_views: &[KvView],
        lora_adapters: &[Option<&str>],
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
        echo: bool,
        capture_layer_ids: Option<&[usize]>,
    ) -> Result<(HiddenStates, Option<HiddenStates>, Option<HiddenStates>)> {
        let batch_size = prompts.len();
        assert_eq!(batch_size, kv_views.len());
        assert_eq!(batch_size, lora_adapters.len());

        let seq_lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
        let lora_ranges =
            build_lora_token_ranges(seq_lens.iter().copied(), lora_adapters.iter().copied());
        let mut lora_groups = prepare_lora_token_groups(&self.ctx, &lora_ranges)?;
        let start_positions: Vec<usize> = kv_views
            .iter()
            .zip(prompts.iter())
            .map(|(v, p)| v.seq_len() - p.len())
            .collect();

        // Build the paged plan before launching embedding, so a plan-build
        // failure can't strand an in-flight embedding the early return would skip.
        let page_indices: Vec<Vec<i32>> =
            kv_views.iter().map(|v| v.page_indices().to_vec()).collect();
        let last_page_lens: Vec<usize> = kv_views
            .iter()
            .map(pegainfer_kv_cache::KvView::last_page_len)
            .collect();
        let plan = PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
            &self.ctx,
            &page_indices,
            &last_page_lens,
            &start_positions,
            &seq_lens,
            self.local_num_attention_heads(),
            self.local_num_key_value_heads(),
            self.config.head_dim,
            PREFILL_ATTENTION_CTA_TILE_Q,
        )?;

        let all_tokens: Vec<u32> = prompts.iter().flat_map(|p| p.iter().copied()).collect();
        let mut hidden = self.get_embeddings_batch(&all_tokens)?;

        // Failed launches may still leave kernels reading these inputs.
        let result = self
            .process_all_layers_batch_multi(
                &mut hidden,
                layout,
                kv_buffer,
                &plan,
                &lora_groups,
                capture_layer_ids,
            )
            .and_then(|captured_hidden| {
                let all_logits = if echo {
                    Some(self.compute_all_position_logits(&hidden)?)
                } else {
                    None
                };

                let mut last_indices = Vec::with_capacity(batch_size);
                let mut offset = 0usize;
                for &seq_len in &seq_lens {
                    last_indices.push((offset + seq_len - 1) as i32);
                    offset += seq_len;
                }
                let logits = self.batch_token_logits(&hidden, &last_indices)?;
                Ok((logits, all_logits, captured_hidden))
            });

        for group in &mut lora_groups {
            if let Some(indices) = group.token_indices_d.take() {
                park(indices);
            }
        }
        park(hidden);
        park(plan);

        result
    }

    /// Run one prompt prefill and return the final RMSNorm'ed hidden state for
    /// the last prompt token. This narrow diagnostic hook lets model adapters
    /// compare the exact hidden-state contract consumed by downstream heads.
    pub(crate) fn prefill_last_normed_hidden(
        &self,
        prompt: &[u32],
        kv_view: &KvView,
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
    ) -> Result<DeviceVec> {
        let plan = self.prefill_plan_for_single_prompt(prompt, kv_view)?;
        let mut hidden = self.get_embeddings_batch(prompt)?;
        let lora_groups: Vec<DeviceLoraTokenGroup<'_>> = Vec::new();
        self.process_all_layers_batch_multi(
            &mut hidden,
            layout,
            kv_buffer,
            &plan,
            &lora_groups,
            None,
        )?;
        let last_hidden = ops::extract_vec(&self.ctx, &hidden, prompt.len() - 1)?;
        let normed = ops::rms_norm(
            &self.ctx,
            &last_hidden,
            &self.norm,
            self.config.rms_norm_eps,
        )?;
        park(hidden);
        park(plan);
        Ok(normed)
    }

    /// Run one prompt prefill and return HuggingFace-style last-token hidden
    /// snapshots: embedding output, each transformer block output before final
    /// model norm, and the final normed hidden.
    pub(crate) fn prefill_last_hidden_layer_snapshots(
        &self,
        prompt: &[u32],
        kv_view: &KvView,
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
    ) -> Result<(DeviceVec, Vec<DeviceVec>, DeviceVec)> {
        let plan = self.prefill_plan_for_single_prompt(prompt, kv_view)?;
        let mut hidden = self.get_embeddings_batch(prompt)?;
        let embedding_snapshot = ops::extract_vec(&self.ctx, &hidden, prompt.len() - 1)?;
        let total_tokens = hidden.seq_len;
        let inter_dim = self.local_intermediate_size();
        let q_dim = self.local_q_dim();
        let kv_dim = self.local_kv_dim();
        let last_token_idx = prompt.len() - 1;
        let lora_groups: Vec<DeviceLoraTokenGroup<'_>> = Vec::new();
        let mut bufs = PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            total_tokens,
        )?;
        let mut layer_snapshots = Vec::with_capacity(self.layers.len());

        crate::green_ctx::fence_producers_before_override(&self.ctx)?;
        let run = (|| -> Result<()> {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                self.forward_layer_batch_paged(
                    layer_idx,
                    layer,
                    &mut hidden,
                    kv_buffer,
                    layout,
                    &plan,
                    &lora_groups,
                    &mut bufs,
                )?;
                layer_snapshots.push(ops::extract_vec(&self.ctx, &hidden, last_token_idx)?);
            }
            Ok(())
        })();
        park(bufs);
        run?;

        let last_hidden = ops::extract_vec(&self.ctx, &hidden, last_token_idx)?;
        let final_normed = ops::rms_norm(
            &self.ctx,
            &last_hidden,
            &self.norm,
            self.config.rms_norm_eps,
        )?;
        park(hidden);
        park(plan);
        Ok((embedding_snapshot, layer_snapshots, final_normed))
    }

    /// Run layers before `layer_idx` normally, then expose named last-token
    /// snapshots across the selected layer's pre-attention, attention, and MLP
    /// stages without changing the production layer forward path.
    pub(crate) fn prefill_layer_stage_snapshots(
        &self,
        layer_idx: usize,
        prompt: &[u32],
        kv_view: &KvView,
        kv_buffer: &CudaSlice<bf16>,
        layout: &KvLayout,
    ) -> Result<Vec<PrefillStageSnapshot>> {
        anyhow::ensure!(
            layer_idx < self.layers.len(),
            "layer_idx {} out of range for {} layers",
            layer_idx,
            self.layers.len()
        );

        let plan = self.prefill_plan_for_single_prompt(prompt, kv_view)?;
        let mut hidden = self.get_embeddings_batch(prompt)?;
        let total_tokens = hidden.seq_len;
        let inter_dim = self.local_intermediate_size();
        let q_dim = self.local_q_dim();
        let kv_dim = self.local_kv_dim();
        let last_token_idx = prompt.len() - 1;
        let lora_groups: Vec<DeviceLoraTokenGroup<'_>> = Vec::new();
        let mut stages = Vec::new();
        let mut bufs = PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            total_tokens,
        )?;

        crate::green_ctx::fence_producers_before_override(&self.ctx)?;
        let run = (|| -> Result<()> {
            for (previous_idx, previous_layer) in self.layers.iter().take(layer_idx).enumerate() {
                self.forward_layer_batch_paged(
                    previous_idx,
                    previous_layer,
                    &mut hidden,
                    kv_buffer,
                    layout,
                    &plan,
                    &lora_groups,
                    &mut bufs,
                )?;
            }

            let layer = &self.layers[layer_idx];
            let stage_prefix = format!("layer{layer_idx}");
            stages.push(PrefillStageSnapshot {
                name: format!("{stage_prefix}.input_hidden.bf16"),
                values: ops::extract_vec(&self.ctx, &hidden, last_token_idx)?,
            });

            self.forward_layer_pre_attn(layer_idx, layer, &hidden, &lora_groups, &mut bufs)?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.input_norm.bf16"),
                &self.ctx,
                &bufs.normed,
                last_token_idx,
            )?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.q_proj.bf16"),
                &self.ctx,
                &bufs.q_batch,
                last_token_idx,
            )?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.k_proj.bf16"),
                &self.ctx,
                &bufs.k_batch,
                last_token_idx,
            )?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.v_proj.bf16"),
                &self.ctx,
                &bufs.v_batch,
                last_token_idx,
            )?;

            let mut q_norm_debug = HiddenStates {
                data: bufs
                    .q_batch
                    .data
                    .try_clone()
                    .map_err(|e| anyhow::anyhow!("D2D q norm debug clone failed: {e}"))?,
                hidden_dim: bufs.q_batch.hidden_dim,
                seq_len: bufs.q_batch.seq_len,
            };
            let mut k_norm_debug = HiddenStates {
                data: bufs
                    .k_batch
                    .data
                    .try_clone()
                    .map_err(|e| anyhow::anyhow!("D2D k norm debug clone failed: {e}"))?,
                hidden_dim: bufs.k_batch.hidden_dim,
                seq_len: bufs.k_batch.seq_len,
            };
            let zero_positions = vec![0i32; total_tokens];
            let zero_positions_d = self
                .ctx
                .stream
                .clone_htod(&zero_positions)
                .map_err(|e| anyhow::anyhow!("H2D q/k norm debug positions failed: {e}"))?;
            ops::qk_norm_rope_batch_decode_into(
                &self.ctx,
                &mut q_norm_debug,
                &mut k_norm_debug,
                0,
                total_tokens,
                &layer.attention.q_norm,
                &layer.attention.k_norm,
                &self.cos_cache,
                &self.sin_cache,
                &zero_positions_d,
                self.local_num_attention_heads(),
                self.local_num_key_value_heads(),
                self.config.head_dim,
                self.config.rms_norm_eps,
            )?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.q_norm.bf16"),
                &self.ctx,
                &q_norm_debug,
                last_token_idx,
            )?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.k_norm.bf16"),
                &self.ctx,
                &k_norm_debug,
                last_token_idx,
            )?;
            park(zero_positions_d);
            park(q_norm_debug);
            park(k_norm_debug);

            self.forward_layer_attn(layer_idx, layer, kv_buffer, layout, &plan, &mut bufs)?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.q_norm_rope.bf16"),
                &self.ctx,
                &bufs.q_batch,
                last_token_idx,
            )?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.k_norm_rope.bf16"),
                &self.ctx,
                &bufs.k_batch,
                last_token_idx,
            )?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.attn_output.bf16"),
                &self.ctx,
                &bufs.attn_output,
                last_token_idx,
            )?;

            ops::gemm_into_checked(
                &self.ctx,
                &layer.attention.o_proj,
                &bufs.attn_output,
                &mut bufs.o_buf,
            )?;
            self.all_reduce_hidden(&mut bufs.o_buf)?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.o_proj.bf16"),
                &self.ctx,
                &bufs.o_buf,
                last_token_idx,
            )?;

            pegainfer_kernels::ops::fused_add_rms_norm_round_batch_into(
                &self.ctx,
                &mut hidden,
                &bufs.o_buf,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
                &mut bufs.normed,
            )?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.post_attn_norm.bf16"),
                &self.ctx,
                &bufs.normed,
                last_token_idx,
            )?;

            ops::gemm_rows_into_checked(
                &self.ctx,
                &layer.mlp.gate_up_proj,
                0,
                inter_dim,
                &bufs.normed,
                &mut bufs.gate_out,
            )?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.gate_proj.bf16"),
                &self.ctx,
                &bufs.gate_out,
                last_token_idx,
            )?;
            ops::gemm_rows_into_checked(
                &self.ctx,
                &layer.mlp.gate_up_proj,
                inter_dim,
                inter_dim,
                &bufs.normed,
                &mut bufs.up_out,
            )?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.up_proj.bf16"),
                &self.ctx,
                &bufs.up_out,
                last_token_idx,
            )?;
            ops::silu_mul_batch_into(&self.ctx, &bufs.gate_out, &bufs.up_out, &mut bufs.act_out)?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.silu_mul.bf16"),
                &self.ctx,
                &bufs.act_out,
                last_token_idx,
            )?;
            ops::gemm_into_checked(
                &self.ctx,
                &layer.mlp.down_proj,
                &bufs.act_out,
                &mut bufs.o_buf,
            )?;
            self.all_reduce_hidden(&mut bufs.o_buf)?;
            push_stage(
                &mut stages,
                format!("{stage_prefix}.down_proj.bf16"),
                &self.ctx,
                &bufs.o_buf,
                last_token_idx,
            )?;
            ops::add_batch_into(&self.ctx, &hidden, &bufs.o_buf, &mut bufs.hidden_out)?;
            std::mem::swap(&mut hidden, &mut bufs.hidden_out);
            push_stage(
                &mut stages,
                format!("{stage_prefix}.output_hidden.bf16"),
                &self.ctx,
                &hidden,
                last_token_idx,
            )?;
            Ok(())
        })();
        park(bufs);
        run?;
        park(hidden);
        park(plan);
        Ok(stages)
    }

    fn process_all_layers_batch_multi(
        &self,
        hidden: &mut HiddenStates,
        layout: &KvLayout,
        kv_buffer: &cudarc::driver::CudaSlice<half::bf16>,
        plan: &PrefillPagedPlan,
        lora_groups: &[DeviceLoraTokenGroup<'_>],
        capture_layer_ids: Option<&[usize]>,
    ) -> Result<Option<HiddenStates>> {
        let total_tokens = hidden.seq_len;
        let inter_dim = self.local_intermediate_size();
        let q_dim = self.local_q_dim();
        let kv_dim = self.local_kv_dim();

        let capture_layer_ids = capture_layer_ids.unwrap_or(&[]);
        anyhow::ensure!(
            capture_layer_ids.windows(2).all(|pair| pair[0] < pair[1]),
            "target hidden capture layer ids must be strictly increasing"
        );
        anyhow::ensure!(
            capture_layer_ids
                .iter()
                .all(|&layer| layer < self.layers.len()),
            "target hidden capture layer id out of range"
        );
        let mut captured_hidden = if capture_layer_ids.is_empty() {
            None
        } else {
            Some(HiddenStates::zeros(
                &self.ctx,
                self.config.hidden_size * capture_layer_ids.len(),
                total_tokens,
            )?)
        };
        let mut next_capture = 0usize;

        let mut bufs = PrefillBuffers::new(
            &self.ctx,
            self.config.hidden_size,
            q_dim,
            kv_dim,
            inter_dim,
            total_tokens,
        )?;

        crate::green_ctx::fence_producers_before_override(&self.ctx)?;

        let run = (|| -> Result<()> {
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                self.forward_layer_batch_paged(
                    layer_idx,
                    layer,
                    hidden,
                    kv_buffer,
                    layout,
                    plan,
                    lora_groups,
                    &mut bufs,
                )?;
                if capture_layer_ids.get(next_capture) == Some(&layer_idx) {
                    let out = captured_hidden
                        .as_mut()
                        .expect("capture buffer exists when ids are non-empty");
                    ops::copy_hidden_rows_into(
                        &self.ctx,
                        hidden,
                        out,
                        next_capture * self.config.hidden_size,
                    )?;
                    next_capture += 1;
                }
            }
            Ok(())
        })();
        park(bufs);
        run?;

        Ok(captured_hidden)
    }
}

fn push_stage(
    stages: &mut Vec<PrefillStageSnapshot>,
    name: String,
    ctx: &DeviceContext,
    batch: &HiddenStates,
    token_idx: usize,
) -> Result<()> {
    stages.push(PrefillStageSnapshot {
        name,
        values: ops::extract_vec(ctx, batch, token_idx)?,
    });
    Ok(())
}
