use anyhow::Context;
use anyhow::Result;
use cudarc::driver::CudaSlice;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

use crate::config::DFlashConfig;
use crate::dspark::MarkovHead;
use crate::dspark::MarkovScratch;
use crate::weights::Qwen3Model;
use crate::weights::TransformerBlock;

mod loading;
mod reservation;

pub(crate) use reservation::DFlashMemoryReservation;

pub(crate) struct DFlashDraftModel {
    config: DFlashConfig,
    layers: Vec<TransformerBlock>,
    norm: DeviceVec,
    hidden_norm: DeviceVec,
    fc: DeviceMatrix,
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
    /// DSpark Markov head; `None` for plain DFlash drafters. When present, the
    /// draft proposes via [`MarkovHead::sample_block`] (anchor-first, all
    /// `block_size` positions) instead of an independent per-position argmax.
    markov: Option<MarkovHead>,
}

pub(crate) struct DFlashRequestState {
    layers: Vec<DFlashLayerCache>,
    pending_context: DFlashPendingContext,
    /// Projected target context for the current draft round. Computed once from
    /// `pending_context` and read by every layer's tail concat, so it lives with
    /// the request (the batched scratch only holds one request's varlen tail).
    context: DFlashContextScratch,
    committed_len: usize,
    max_cache_len: usize,
}

struct DFlashLayerCache {
    k: HiddenStates,
    v: HiddenStates,
}

struct DFlashPendingContext {
    buffer: HiddenStates,
    len: usize,
    capacity: usize,
}

/// Per-request projected context. The fc projection + hidden_norm turn the
/// captured target hidden context into draft hidden space once per draft round;
/// every layer's tail concat reads `context_hidden`, so it must persist across
/// the layer loop and therefore lives in the request (not the shared scratch).
struct DFlashContextScratch {
    max_context_len: usize,
    context_projected: HiddenStates,
    context_hidden: HiddenStates,
}

/// Lane-level batched draft scratch, allocated once for the whole decode batch.
///
/// Dense buffers (`hidden`, `normed`, `q_batch`, `attn_output`, the MLP buffers,
/// and `logits`) hold `max_batch * block_size` rows so the GEMM / rms_norm /
/// silu / add / logits / embedding ops run ONCE over the batched buffer. The
/// varlen tail buffers (`tail_input`, `k_tail`, `v_tail`) stay sized for a single
/// request and are reused inside the per-request loop, because their ops (tail
/// concat, k/v GEMMs, rope, KV copy, attention) still loop per request — Step 2
/// will batch those via CUDA-kernel changes.
pub(crate) struct DFlashBatchScratch {
    max_batch_block_rows: usize,
    max_tail_len: usize,
    block_token_ids_h: Vec<u32>,
    token_ids_d: CudaSlice<u32>,
    hidden: HiddenStates,
    hidden_out: HiddenStates,
    normed: HiddenStates,
    q_batch: HiddenStates,
    attn_output: HiddenStates,
    o_buf: HiddenStates,
    gate_out: HiddenStates,
    up_out: HiddenStates,
    act_out: HiddenStates,
    logits_normed: HiddenStates,
    logits: HiddenStates,
    // Shared single-request varlen tail scratch (reused inside the per-request loop).
    tail_input: HiddenStates,
    k_tail: HiddenStates,
    v_tail: HiddenStates,
    // DSpark Markov sample-loop scratch; `None` for plain DFlash drafters.
    markov: Option<MarkovScratch>,
}

impl DFlashRequestState {
    pub(crate) fn pending_context_len(&self) -> Option<usize> {
        (self.pending_context.len > 0).then_some(self.pending_context.len)
    }
}

impl DFlashPendingContext {
    fn new(ctx: &DeviceContext, hidden_dim: usize, capacity: usize) -> Result<Self> {
        anyhow::ensure!(
            capacity > 0,
            "DFlash pending context capacity must be non-zero"
        );
        let mut buffer = HiddenStates::zeros(ctx, hidden_dim, capacity)?;
        buffer.seq_len = 0;
        Ok(Self {
            buffer,
            len: 0,
            capacity,
        })
    }

    fn append_from(
        &mut self,
        ctx: &DeviceContext,
        src: &HiddenStates,
        src_token_offset: usize,
        token_count: usize,
        max_capacity: usize,
    ) -> Result<()> {
        let required_len = self
            .len
            .checked_add(token_count)
            .context("DFlash pending context length overflow")?;
        anyhow::ensure!(
            required_len <= max_capacity,
            "DFlash pending context length {} exceeds request capacity {}",
            required_len,
            max_capacity
        );
        self.ensure_capacity(ctx, required_len, max_capacity)?;
        self.buffer.seq_len = self.capacity;
        ops::copy_hidden_token_range_into(
            ctx,
            src,
            src_token_offset,
            &mut self.buffer,
            self.len,
            token_count,
        )?;
        self.len = required_len;
        self.buffer.seq_len = self.len;
        Ok(())
    }

    fn ensure_capacity(
        &mut self,
        ctx: &DeviceContext,
        required_len: usize,
        max_capacity: usize,
    ) -> Result<()> {
        if required_len <= self.capacity {
            return Ok(());
        }
        let doubled = self
            .capacity
            .checked_mul(2)
            .context("DFlash pending context capacity overflow")?;
        let new_capacity = required_len.max(doubled).min(max_capacity);
        anyhow::ensure!(
            new_capacity >= required_len,
            "DFlash pending context capacity {} cannot fit {} tokens",
            new_capacity,
            required_len
        );
        let mut next = HiddenStates::zeros(ctx, self.buffer.hidden_dim, new_capacity)?;
        if self.len > 0 {
            self.buffer.seq_len = self.capacity;
            ops::copy_hidden_token_range_into(ctx, &self.buffer, 0, &mut next, 0, self.len)?;
        }
        next.seq_len = self.len;
        self.buffer = next;
        self.capacity = new_capacity;
        Ok(())
    }

    fn activate_for_read(&mut self) {
        self.buffer.seq_len = self.len;
    }

    fn clear(&mut self) {
        self.len = 0;
        self.buffer.seq_len = 0;
    }
}

impl DFlashContextScratch {
    fn new(ctx: &DeviceContext, hidden_size: usize, max_context_len: usize) -> Result<Self> {
        Ok(Self {
            max_context_len,
            context_projected: HiddenStates::zeros(ctx, hidden_size, max_context_len)?,
            context_hidden: HiddenStates::zeros(ctx, hidden_size, max_context_len)?,
        })
    }

    fn ensure_capacity(
        &mut self,
        ctx: &DeviceContext,
        hidden_size: usize,
        context_len: usize,
    ) -> Result<()> {
        if context_len > self.max_context_len {
            *self = Self::new(ctx, hidden_size, context_len)?;
        }
        self.context_projected.seq_len = context_len;
        self.context_hidden.seq_len = context_len;
        Ok(())
    }
}

impl DFlashBatchScratch {
    fn new(
        ctx: &DeviceContext,
        config: &DFlashConfig,
        max_decode_batch_size: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            max_decode_batch_size > 0,
            "DFlash batch scratch needs a non-zero batch size"
        );
        let block_size = config.block_size;
        let hidden_size = config.hidden_size;
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let inter_dim = config.intermediate_size;
        // Dense buffers span the whole decode batch so the dense ops run once.
        let batch_rows = block_size * max_decode_batch_size;
        // The shared varlen tail starts at one block (no committed context yet)
        // and grows on demand via `ensure_tail_capacity`.
        let tail_capacity = block_size;

        Ok(Self {
            max_batch_block_rows: batch_rows,
            max_tail_len: tail_capacity,
            block_token_ids_h: vec![config.mask_token_id; batch_rows],
            token_ids_d: ctx.stream.alloc_zeros(batch_rows)?,
            hidden: HiddenStates::zeros(ctx, hidden_size, batch_rows)?,
            hidden_out: HiddenStates::zeros(ctx, hidden_size, batch_rows)?,
            normed: HiddenStates::zeros(ctx, hidden_size, batch_rows)?,
            q_batch: HiddenStates::zeros(ctx, q_dim, batch_rows)?,
            attn_output: HiddenStates::zeros(ctx, q_dim, batch_rows)?,
            o_buf: HiddenStates::zeros(ctx, hidden_size, batch_rows)?,
            gate_out: HiddenStates::zeros(ctx, inter_dim, batch_rows)?,
            up_out: HiddenStates::zeros(ctx, inter_dim, batch_rows)?,
            act_out: HiddenStates::zeros(ctx, inter_dim, batch_rows)?,
            logits_normed: HiddenStates::zeros(ctx, hidden_size, batch_rows)?,
            logits: HiddenStates::zeros(ctx, config.vocab_size, batch_rows)?,
            tail_input: HiddenStates::zeros(ctx, hidden_size, tail_capacity)?,
            k_tail: HiddenStates::zeros(ctx, kv_dim, tail_capacity)?,
            v_tail: HiddenStates::zeros(ctx, kv_dim, tail_capacity)?,
            markov: config
                .uses_markov_head()
                .then(|| MarkovScratch::new(ctx, config, max_decode_batch_size))
                .transpose()?,
        })
    }

    /// Markov scratch's max batch (chain-rows), `0` for a plain DFlash
    /// drafter — the hedge ladder's capacity guard.
    pub(crate) fn markov_max_batch(&self) -> usize {
        self.markov
            .as_ref()
            .map_or(0, crate::dspark::MarkovScratch::max_batch)
    }

    /// The batched backbone draft logits `[active_batch * block_size, vocab]`,
    /// request-major, as produced by `DFlashDraftModel::draft_logits_batched`.
    pub(crate) fn logits(&self) -> &HiddenStates {
        &self.logits
    }

    /// Point every dense buffer at the active `batch_block_rows = active_batch *
    /// block_size` prefix. Allocated for the max decode batch, so this only
    /// shrinks `seq_len`; it never reallocates.
    fn activate_dense(&mut self, batch_block_rows: usize) {
        assert!(
            batch_block_rows <= self.max_batch_block_rows,
            "DFlash batched draft {} block rows exceeds scratch capacity {}",
            batch_block_rows,
            self.max_batch_block_rows
        );
        self.hidden.seq_len = batch_block_rows;
        self.hidden_out.seq_len = batch_block_rows;
        self.normed.seq_len = batch_block_rows;
        self.q_batch.seq_len = batch_block_rows;
        self.attn_output.seq_len = batch_block_rows;
        self.o_buf.seq_len = batch_block_rows;
        self.gate_out.seq_len = batch_block_rows;
        self.up_out.seq_len = batch_block_rows;
        self.act_out.seq_len = batch_block_rows;
        self.logits_normed.seq_len = batch_block_rows;
        self.logits.seq_len = batch_block_rows;
    }

    /// Size the shared varlen tail buffers for one request's `tail_len =
    /// context_len + block_size`, growing the allocation if needed.
    fn ensure_tail_capacity(
        &mut self,
        ctx: &DeviceContext,
        config: &DFlashConfig,
        tail_len: usize,
    ) -> Result<()> {
        if tail_len > self.max_tail_len {
            let hidden_size = config.hidden_size;
            let kv_dim = config.num_key_value_heads * config.head_dim;
            self.tail_input = HiddenStates::zeros(ctx, hidden_size, tail_len)?;
            self.k_tail = HiddenStates::zeros(ctx, kv_dim, tail_len)?;
            self.v_tail = HiddenStates::zeros(ctx, kv_dim, tail_len)?;
            self.max_tail_len = tail_len;
        }
        self.tail_input.seq_len = tail_len;
        self.k_tail.seq_len = tail_len;
        self.v_tail.seq_len = tail_len;
        Ok(())
    }
}

impl DFlashDraftModel {
    pub(crate) fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Largest sequence position the draft can cache. `validate_for_target`
    /// guarantees this is `>=` the target's, but the draft's per-step in-fill
    /// block writes `block_size` transient positions past the committed length,
    /// so the usable context is `max_position_embeddings - block_size`.
    pub(crate) fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }

    fn mask_token_id(&self) -> u32 {
        self.config.mask_token_id
    }

    pub(crate) fn target_layer_ids(&self) -> &[usize] {
        &self.config.target_layer_ids
    }

    pub(crate) fn uses_markov_head(&self) -> bool {
        self.markov.is_some()
    }

    /// Anchor-first block layout (a checkpoint property, see
    /// [`DFlashConfig::anchor_first`]) — drives both the verify span and the
    /// draft-block slice start, independently of the markov head.
    pub(crate) fn anchor_first(&self) -> bool {
        self.config.anchor_first()
    }

    /// Length of each request's verify span = anchor (1) + proposed drafts.
    /// Anchor-drop checkpoints discard block position 0, proposing
    /// `block_size - 1` drafts (span `block_size`); anchor-first checkpoints
    /// propose all `block_size` drafts (span `block_size + 1`). The verify
    /// CUDA-graph buffers are sized to this.
    pub(crate) fn verify_span(&self) -> usize {
        if self.anchor_first() {
            self.block_size() + 1
        } else {
            self.block_size()
        }
    }

    pub(crate) fn tune_gemm_algos(&self, target: &Qwen3Model) -> Result<()> {
        let ctx = target.device_ctx();
        let block_size = self.block_size().min(ops::GEMM_LT_MAX_N);
        let hidden = self.config.hidden_size;
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let context_dim = self.context_feature_dim();

        let fc_samples = [(&self.fc, 0)];
        for n in 1..=block_size {
            ops::gemm_lt_tune(ctx, &fc_samples, hidden, n)?;
        }

        let kv_samples: Vec<_> = self
            .layers
            .iter()
            .flat_map(|layer| {
                [
                    (&layer.attention.qkv_proj, q_dim),
                    (&layer.attention.qkv_proj, q_dim + kv_dim),
                ]
            })
            .collect();
        let min_tail_n = self.block_size() + 1;
        let max_tail_n = (self.block_size() * 2).min(ops::GEMM_LT_MAX_N);
        for n in min_tail_n..=max_tail_n {
            ops::gemm_lt_tune(ctx, &kv_samples, kv_dim, n)?;
        }

        log::info!(
            "Qwen3 DFlash cublasLt tuned: fc M={} K={} N=1..{}, kv M={} K={} N={}..{}",
            hidden,
            context_dim,
            block_size,
            kv_dim,
            hidden,
            min_tail_n,
            max_tail_n,
        );
        Ok(())
    }

    /// Allocate the lane-level batched draft scratch once, sized for the whole
    /// decode batch. The per-request `DFlashRequestState` no longer owns scratch.
    pub(crate) fn new_batch_scratch(
        &self,
        ctx: &DeviceContext,
        max_decode_batch_size: usize,
    ) -> Result<DFlashBatchScratch> {
        DFlashBatchScratch::new(ctx, &self.config, max_decode_batch_size)
    }

    pub(crate) fn new_request_state(
        &self,
        ctx: &DeviceContext,
        max_cache_len: usize,
    ) -> Result<DFlashRequestState> {
        anyhow::ensure!(
            max_cache_len <= self.config.max_position_embeddings,
            "DFlash request cache length {} exceeds max_position_embeddings {}",
            max_cache_len,
            self.config.max_position_embeddings
        );
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let mut layers = Vec::with_capacity(self.layers.len());
        for _ in 0..self.layers.len() {
            layers.push(DFlashLayerCache {
                k: HiddenStates::zeros(ctx, kv_dim, max_cache_len)?,
                v: HiddenStates::zeros(ctx, kv_dim, max_cache_len)?,
            });
        }
        Ok(DFlashRequestState {
            layers,
            pending_context: DFlashPendingContext::new(
                ctx,
                self.context_feature_dim(),
                self.config.block_size.min(max_cache_len),
            )?,
            context: DFlashContextScratch::new(
                ctx,
                self.config.hidden_size,
                self.config.block_size,
            )?,
            committed_len: 0,
            max_cache_len,
        })
    }

    pub(crate) fn append_pending_context(
        &self,
        ctx: &DeviceContext,
        state: &mut DFlashRequestState,
        captured_hidden: &HiddenStates,
        token_offset: usize,
        token_count: usize,
    ) -> Result<()> {
        anyhow::ensure!(token_count > 0, "DFlash context append needs tokens");
        anyhow::ensure!(
            captured_hidden.hidden_dim == self.context_feature_dim(),
            "DFlash captured hidden dim {} does not match expected {}",
            captured_hidden.hidden_dim,
            self.context_feature_dim()
        );
        anyhow::ensure!(
            token_offset + token_count <= captured_hidden.seq_len,
            "DFlash captured hidden token range exceeds source"
        );
        let required_committed_len = state
            .committed_len
            .checked_add(state.pending_context.len)
            .and_then(|len| len.checked_add(token_count))
            .and_then(|len| len.checked_add(self.block_size()))
            .context("DFlash pending context cache length overflow")?;
        anyhow::ensure!(
            required_committed_len <= state.max_cache_len,
            "DFlash pending context would exceed cache: committed={}, pending={}, append={}, block={}, max={}",
            state.committed_len,
            state.pending_context.len,
            token_count,
            self.block_size(),
            state.max_cache_len
        );
        state.pending_context.append_from(
            ctx,
            captured_hidden,
            token_offset,
            token_count,
            state.max_cache_len,
        )?;
        Ok(())
    }

    /// Batched draft forward over all active requests at once.
    ///
    /// The *dense* ops (embedding, rms_norm, q / o / gate_up / down GEMMs, silu,
    /// add, fused_add_rms_norm, logits) run ONCE over an `active_batch *
    /// block_size` batched buffer. The *varlen* ops (context projection, tail
    /// concat, k/v GEMMs, rope, KV copy, attention) still loop per request,
    /// slicing each request's `block_size` rows at offset `i * block_size` in the
    /// batched buffers — those are Step 2/3's job to batch via CUDA-kernel changes.
    ///
    /// Returns the batched logits (`active_batch * block_size` rows): request `i`
    /// owns rows `[i * block_size, (i + 1) * block_size)`.
    pub(crate) fn draft_logits_batched<'a>(
        &self,
        target: &Qwen3Model,
        states: &mut [&mut DFlashRequestState],
        current_tokens: &[u32],
        scratch: &'a mut DFlashBatchScratch,
    ) -> Result<&'a HiddenStates> {
        let ctx = target.device_ctx();
        let active_batch = states.len();
        anyhow::ensure!(
            active_batch > 0,
            "DFlash batched draft needs active requests"
        );
        anyhow::ensure!(
            states.len() == current_tokens.len(),
            "DFlash batched draft: {} states vs {} current tokens",
            states.len(),
            current_tokens.len()
        );
        let block_size = self.block_size();
        let batch_block_rows = active_batch * block_size;

        // Each request's committed context length for this round; advancing
        // `committed_len` is deferred until after the layer loop (the rope start
        // positions and KV write offsets read the pre-advance value).
        let mut context_lens = Vec::with_capacity(active_batch);
        for (i, state) in states.iter().enumerate() {
            let Some(context_len) = state.pending_context_len() else {
                anyhow::bail!(
                    "DFlash draft requested before target hidden context is available (request slot {i})"
                );
            };
            let tail_len = context_len + block_size;
            anyhow::ensure!(
                state.committed_len + tail_len <= state.max_cache_len,
                "DFlash draft cache overflow: committed={}, tail={}, max={}",
                state.committed_len,
                tail_len,
                state.max_cache_len
            );
            context_lens.push(context_len);
        }

        scratch.activate_dense(batch_block_rows);

        // Build the batched token id buffer: each request's block is
        // [current_token, mask, mask, ...].
        scratch.block_token_ids_h[..batch_block_rows].fill(self.mask_token_id());
        for (i, &current_token) in current_tokens.iter().enumerate() {
            scratch.block_token_ids_h[i * block_size] = current_token;
        }
        // token_ids_d holds `max_batch * block_size` ids; copy only the active
        // prefix. The embedding kernel reads `out.seq_len = batch_block_rows` ids
        // from the buffer start, so the active prefix is what it consumes.
        let mut token_ids_dst = scratch.token_ids_d.slice_mut(..batch_block_rows);
        ctx.stream.memcpy_htod(
            &scratch.block_token_ids_h[..batch_block_rows],
            &mut token_ids_dst,
        )?;
        target.get_embeddings_batch_into(&scratch.token_ids_d, &mut scratch.hidden)?;

        // Per-request context projection: varlen (each request's committed
        // prefix differs), persisted in the request so every layer can read it.
        for (i, state) in states.iter_mut().enumerate() {
            let context_len = context_lens[i];
            state
                .context
                .ensure_capacity(ctx, self.config.hidden_size, context_len)?;
            state.pending_context.activate_for_read();
            self.project_context_into(ctx, &state.pending_context.buffer, &mut state.context);
            state.pending_context.clear();
        }

        let hidden_size = self.config.hidden_size;
        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        let inter_dim = self.config.intermediate_size;
        debug_assert_eq!(scratch.hidden.hidden_dim, hidden_size);
        debug_assert_eq!(scratch.q_batch.hidden_dim, q_dim);
        debug_assert_eq!(scratch.k_tail.hidden_dim, kv_dim);
        debug_assert_eq!(scratch.gate_out.hidden_dim, inter_dim);

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            // Dense: input layernorm over the whole batch.
            ops::rms_norm_batch_into(
                ctx,
                &scratch.hidden,
                &layer.input_layernorm,
                self.config.rms_norm_eps,
                &mut scratch.normed,
            );

            // Dense: Q projection over the whole batch (per-token, no cross-request
            // mixing). Computed before the per-request loop reads `normed`, and
            // before the post-attention norm overwrites it.
            ops::gemm_rows_into(
                ctx,
                &layer.attention.qkv_proj,
                0,
                q_dim,
                &scratch.normed,
                &mut scratch.q_batch,
            );

            // Per-request varlen attention: tail concat, k/v GEMMs, rope, KV copy,
            // single-request prefill. Each request slices its `block_size` rows at
            // offset `i * block_size` of the batched `normed`/`q_batch`/`attn_output`.
            for (i, state) in states.iter_mut().enumerate() {
                let context_len = context_lens[i];
                let tail_len = context_len + block_size;
                let row_offset = i * block_size;
                scratch.ensure_tail_capacity(ctx, &self.config, tail_len)?;

                // tail_input = [context_hidden(context_len) | normed_block(block_size)].
                ops::copy_hidden_token_range_into(
                    ctx,
                    &state.context.context_hidden,
                    0,
                    &mut scratch.tail_input,
                    0,
                    context_len,
                )?;
                ops::copy_hidden_token_range_into(
                    ctx,
                    &scratch.normed,
                    row_offset,
                    &mut scratch.tail_input,
                    context_len,
                    block_size,
                )?;

                ops::gemm_rows_into(
                    ctx,
                    &layer.attention.qkv_proj,
                    q_dim,
                    kv_dim,
                    &scratch.tail_input,
                    &mut scratch.k_tail,
                );
                ops::gemm_rows_into(
                    ctx,
                    &layer.attention.qkv_proj,
                    q_dim + kv_dim,
                    kv_dim,
                    &scratch.tail_input,
                    &mut scratch.v_tail,
                );

                ops::dflash_qk_norm_rope_into(
                    ctx,
                    &mut scratch.q_batch,
                    row_offset,
                    block_size,
                    &mut scratch.k_tail,
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.cos_cache,
                    &self.sin_cache,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    state.committed_len + context_len,
                    state.committed_len,
                    self.config.rms_norm_eps,
                )?;

                let cache = &mut state.layers[layer_idx];
                ops::copy_hidden_token_range_into(
                    ctx,
                    &scratch.k_tail,
                    0,
                    &mut cache.k,
                    state.committed_len,
                    tail_len,
                )?;
                ops::copy_hidden_token_range_into(
                    ctx,
                    &scratch.v_tail,
                    0,
                    &mut cache.v,
                    state.committed_len,
                    tail_len,
                )?;
                ops::single_prefill_nhd_noncausal_into(
                    ctx,
                    &scratch.q_batch,
                    row_offset,
                    block_size,
                    &cache.k,
                    &cache.v,
                    &mut scratch.attn_output,
                    self.config.num_attention_heads,
                    self.config.num_key_value_heads,
                    self.config.head_dim,
                    state.committed_len + tail_len,
                )?;
            }

            // Dense: o_proj + residual + post-attention norm + MLP over the batch.
            ops::gemm_into(
                ctx,
                &layer.attention.o_proj,
                &scratch.attn_output,
                &mut scratch.o_buf,
            );
            pegainfer_kernels::ops::fused_add_rms_norm_round_batch_into(
                ctx,
                &mut scratch.hidden,
                &scratch.o_buf,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
                &mut scratch.normed,
            )?;

            ops::gemm_rows_into(
                ctx,
                &layer.mlp.gate_up_proj,
                0,
                inter_dim,
                &scratch.normed,
                &mut scratch.gate_out,
            );
            ops::gemm_rows_into(
                ctx,
                &layer.mlp.gate_up_proj,
                inter_dim,
                inter_dim,
                &scratch.normed,
                &mut scratch.up_out,
            );
            ops::silu_mul_batch_into(
                ctx,
                &scratch.gate_out,
                &scratch.up_out,
                &mut scratch.act_out,
            )?;
            ops::gemm_into(
                ctx,
                &layer.mlp.down_proj,
                &scratch.act_out,
                &mut scratch.o_buf,
            );
            ops::add_batch_into(
                ctx,
                &scratch.hidden,
                &scratch.o_buf,
                &mut scratch.hidden_out,
            )?;
            std::mem::swap(&mut scratch.hidden, &mut scratch.hidden_out);
        }

        for (i, state) in states.iter_mut().enumerate() {
            state.committed_len += context_lens[i];
        }
        self.compute_logits_with_target_head_into(target, scratch);
        Ok(&scratch.logits)
    }

    /// DSpark propose: sample all `block_size` positions per request with the
    /// Markov head (anchor-first), reading the backbone logits already produced
    /// by [`Self::draft_logits_batched`] into `scratch.logits`. Returns the
    /// `active_batch * block_size` request-major drafts.
    pub(crate) fn markov_draft_tokens(
        &self,
        ctx: &DeviceContext,
        current_tokens: &[u32],
        scratch: &mut DFlashBatchScratch,
    ) -> Result<Vec<u32>> {
        let markov = self
            .markov
            .as_ref()
            .context("markov_draft_tokens called on a non-DSpark drafter")?;
        // Split-borrow: the backbone logits (read) and the Markov scratch (write)
        // are disjoint fields of the same scratch.
        let DFlashBatchScratch {
            logits,
            markov: markov_scratch,
            ..
        } = scratch;
        let markov_scratch = markov_scratch
            .as_mut()
            .context("Markov scratch was not allocated for a DSpark drafter")?;
        markov.sample_block(
            ctx,
            logits,
            current_tokens,
            self.block_size(),
            markov_scratch,
        )
    }

    /// Chain-A markov proposal plus per-branch-position runner-up capture
    /// (see [`crate::dspark::MarkovHead::sample_block_with_runners`]).
    pub(crate) fn markov_draft_with_runners(
        &self,
        ctx: &DeviceContext,
        current_tokens: &[u32],
        branch_positions: &[usize],
        scratch: &mut DFlashBatchScratch,
    ) -> Result<Vec<u32>> {
        let markov = self
            .markov
            .as_ref()
            .context("markov_draft_with_runners called on a non-DSpark drafter")?;
        let DFlashBatchScratch {
            logits,
            markov: markov_scratch,
            ..
        } = scratch;
        let markov_scratch = markov_scratch
            .as_mut()
            .context("Markov scratch was not allocated for a DSpark drafter")?;
        markov.sample_block_with_runners(
            ctx,
            logits,
            current_tokens,
            self.block_size(),
            branch_positions,
            markov_scratch,
        )
    }

    /// Batched hedge-chain ladder (see
    /// [`crate::dspark::MarkovHead::sample_chain_ladder`]).
    pub(crate) fn markov_chain_ladder(
        &self,
        ctx: &DeviceContext,
        chain_a: &[u32],
        req_map: &[u32],
        positions: &[usize],
        scratch: &mut DFlashBatchScratch,
    ) -> Result<Vec<u32>> {
        let markov = self
            .markov
            .as_ref()
            .context("markov_chain_ladder called on a non-DSpark drafter")?;
        let DFlashBatchScratch {
            logits,
            markov: markov_scratch,
            ..
        } = scratch;
        let markov_scratch = markov_scratch
            .as_mut()
            .context("Markov scratch was not allocated for a DSpark drafter")?;
        markov.sample_chain_ladder(
            ctx,
            logits,
            chain_a,
            req_map,
            positions,
            self.block_size(),
            markov_scratch,
        )
    }

    fn context_feature_dim(&self) -> usize {
        self.config.hidden_size * self.target_layer_ids().len()
    }

    fn project_context_into(
        &self,
        ctx: &DeviceContext,
        context_features: &HiddenStates,
        context: &mut DFlashContextScratch,
    ) {
        ops::gemm_into(
            ctx,
            &self.fc,
            context_features,
            &mut context.context_projected,
        );
        ops::rms_norm_batch_into(
            ctx,
            &context.context_projected,
            &self.hidden_norm,
            self.config.rms_norm_eps,
            &mut context.context_hidden,
        );
    }

    fn compute_logits_with_target_head_into(
        &self,
        target: &Qwen3Model,
        scratch: &mut DFlashBatchScratch,
    ) {
        let ctx = target.device_ctx();
        ops::rms_norm_batch_into(
            ctx,
            &scratch.hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut scratch.logits_normed,
        );
        ops::gemm_into(
            ctx,
            target.output_projection(),
            &scratch.logits_normed,
            &mut scratch.logits,
        );
    }
}

#[cfg(test)]
pub(crate) fn validate_dflash_config_for_target(
    dflash_path: &str,
    target_config: &crate::config::Config,
) -> Result<DFlashConfig> {
    let config = DFlashConfig::from_file(dflash_path)?;
    config.validate_for_target(target_config)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::validate_dflash_config_for_target;
    use crate::config::Config;

    #[test]
    fn downloaded_dflash_config_matches_qwen3_4b() {
        let target_path = std::env::var("PEGAINFER_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "/data/models/Qwen3-4B".to_string());
        let dflash_path = std::env::var("PEGAINFER_DFLASH_TEST_MODEL_PATH")
            .unwrap_or_else(|_| "/data/models/Qwen3-4B-DFlash-b16".to_string());
        if !Path::new(&target_path).join("config.json").exists()
            || !Path::new(&dflash_path).join("config.json").exists()
        {
            eprintln!(
                "skipping DFlash config test; set PEGAINFER_TEST_MODEL_PATH and PEGAINFER_DFLASH_TEST_MODEL_PATH"
            );
            return;
        }

        let target = Config::from_file(&target_path).expect("target config");
        let dflash = validate_dflash_config_for_target(&dflash_path, &target)
            .expect("DFlash config should match target");

        assert_eq!(dflash.block_size, 16);
        assert_eq!(dflash.mask_token_id, 151_669);
        assert_eq!(dflash.target_layer_ids, vec![1, 9, 17, 25, 33]);

        // Pin the memory reservation the KV budget bills against. The per-token
        // term (draft KV 5*2*1024*2 + scratch-context (3*2560+2*1024)*2 + pending
        // 2560*5*2) drives the ~12% block haircut; a layer-count or geometry
        // regression here would silently over/under-reserve and risk OOM.
        let reservation =
            super::DFlashMemoryReservation::from_config(&dflash, /*max_decode_batch*/ 256)
                .expect("reservation sizing");
        assert_eq!(
            reservation.kv_bytes_per_token, 65_536,
            "draft KV(20480) + scratch-ctx(19456) + pending(25600) per token"
        );
        // Weights (~1.1 GiB) dominate the fixed term at batch=1; the block-sized
        // per-request scratch (~6.5 MiB, logits-heavy) plus the one-block KV/tail
        // headroom (~0.5 MiB) add across the decode batch.
        let fixed_batch1 = super::DFlashMemoryReservation::from_config(&dflash, 1)
            .expect("reservation sizing")
            .fixed_bytes;
        assert!(
            (1_150_000_000..1_220_000_000).contains(&fixed_batch1),
            "draft weights ~1.1GiB, got {fixed_batch1}"
        );
        assert!(
            (2_900_000_000..3_000_000_000).contains(&reservation.fixed_bytes),
            "weights + 256 * (~6.5MiB scratch + ~0.5MiB block headroom), got {}",
            reservation.fixed_bytes
        );
    }
}
