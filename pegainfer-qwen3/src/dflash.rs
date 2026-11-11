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

mod conv;
mod draft_kv_pool;
mod loading;
mod reservation;
mod selector;

use conv::ConvScratch;
use conv::LayerConvs;
pub(crate) use draft_kv_pool::DFlashDraftKvPool;
pub(crate) use reservation::DFlashMemoryReservation;
use selector::SelectorHead;
use selector::SelectorScratch;

/// `PEGAINFER_SPEC_MAX_DRAFTS=<k>` — cap on the draft candidates proposed per
/// round (0/unset = the checkpoint's natural block). Read once; a malformed
/// value refuses to start (see [`crate::spec_env`]).
fn max_drafts() -> usize {
    static MAX_DRAFTS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MAX_DRAFTS.get_or_init(|| crate::spec_env::read(crate::spec_env::MAX_DRAFTS))
}

pub(crate) struct DFlashDraftModel {
    config: DFlashConfig,
    layers: Vec<TransformerBlock>,
    convs: Option<Vec<LayerConvs>>,
    norm: DeviceVec,
    hidden_norm: DeviceVec,
    fc: DeviceMatrix,
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
    /// DSpark Markov head; `None` for plain DFlash drafters. When present, the
    /// draft proposes via [`MarkovHead::sample_block`] (anchor-first, all
    /// `block_size` positions) instead of an independent per-position argmax.
    markov: Option<MarkovHead>,
    selector: Option<SelectorHead>,
}

pub(crate) struct DFlashRequestState {
    layers: Vec<DFlashLayerCache>,
    /// Draft K/V rows already materialized into `layers` (and mirrored into the
    /// paged pool) at positions `[0, committed_len)`.
    pub(crate) committed_len: usize,
    max_cache_len: usize,
}

struct DFlashLayerCache {
    k: HiddenStates,
    v: HiddenStates,
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
    selector: Option<SelectorScratch>,
    conv: Option<ConvScratch>,
}

/// Lane-level scratch for materializing a step's target context into draft K/V.
/// One instance serves the whole lane: prefill chunks and verify spans reuse it.
pub(crate) struct DFlashKvMaterializeScratch {
    capture_rows: HiddenStates,
    ctx_projected: HiddenStates,
    ctx_hidden: HiddenStates,
    q: HiddenStates,
    k: HiddenStates,
    v: HiddenStates,
    locs: CudaSlice<i32>,
    locs_host: Vec<i32>,
    max_rows: usize,
}

impl DFlashKvMaterializeScratch {
    pub(crate) fn new(ctx: &DeviceContext, config: &DFlashConfig, max_rows: usize) -> Result<Self> {
        anyhow::ensure!(max_rows > 0, "DFlash materialize scratch needs rows");
        let capture_dim = config.hidden_size * config.target_layer_ids.len();
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        Ok(Self {
            capture_rows: HiddenStates::zeros(ctx, capture_dim, max_rows)?,
            ctx_projected: HiddenStates::zeros(ctx, config.hidden_size, max_rows)?,
            ctx_hidden: HiddenStates::zeros(ctx, config.hidden_size, max_rows)?,
            q: HiddenStates::zeros(ctx, q_dim, max_rows)?,
            k: HiddenStates::zeros(ctx, kv_dim, max_rows)?,
            v: HiddenStates::zeros(ctx, kv_dim, max_rows)?,
            locs: ctx.stream.alloc_zeros(max_rows)?,
            locs_host: vec![0; max_rows],
            max_rows,
        })
    }

    pub(crate) fn max_rows(&self) -> usize {
        self.max_rows
    }
}

impl DFlashRequestState {
    /// Token lifetime this request's buffers were sized for; the lane charges
    /// its draft-state budget per this token count.
    pub(crate) fn max_cache_len(&self) -> usize {
        self.max_cache_len
    }
}

impl DFlashBatchScratch {
    pub(crate) fn selector_proposal(
        &self,
    ) -> Option<(&CudaSlice<u32>, &CudaSlice<f32>, &CudaSlice<u32>)> {
        self.selector.as_ref().map(SelectorScratch::proposal)
    }

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
            logits: HiddenStates::zeros(ctx, config.logits_rows, batch_rows)?,
            tail_input: HiddenStates::zeros(ctx, hidden_size, tail_capacity)?,
            k_tail: HiddenStates::zeros(ctx, kv_dim, tail_capacity)?,
            v_tail: HiddenStates::zeros(ctx, kv_dim, tail_capacity)?,
            conv: config
                .conv
                .as_ref()
                .map(|_| ConvScratch::new(ctx, config, batch_rows))
                .transpose()?,
            selector: (config.selector_rank > 0)
                .then(|| SelectorScratch::new(ctx, config, max_decode_batch_size))
                .transpose()?,
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

    /// The batched backbone draft logits `[active_batch * block_size, logits_rows]`,
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

    /// How many draft candidates the block layout yields before any trim:
    /// anchor-drop checkpoints discard block position 0 (the anchor slot; only
    /// the mask positions draft), so they propose `block_size - 1`; anchor-first
    /// checkpoints have position 0 already predict the first draft, so they
    /// propose all `block_size` — a one-token-longer verify span. This is a
    /// checkpoint property, not a markov one: a `markov_rank == 0` DeepSpec
    /// checkpoint is still anchor-first.
    fn natural_drafts(&self) -> usize {
        if self.anchor_first() {
            self.block_size()
        } else {
            self.block_size() - 1
        }
    }

    /// `PEGAINFER_SPEC_MAX_DRAFTS=<k>` trims how many draft candidates are
    /// proposed per round: the drafter still rolls its whole block out (its
    /// cache still writes the `block_size`-slot in-fill tail every round, so
    /// the headroom is unchanged), but only the first `k` candidates are
    /// verified, so the target forward carries `1 + k` rows per request
    /// instead of the full span. `0`/unset = off.
    pub(crate) fn effective_drafts(&self) -> usize {
        let natural = self.natural_drafts();
        let cap = max_drafts();
        if cap > 0 { natural.min(cap) } else { natural }
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

    pub(crate) fn new_draft_kv_pool(
        &self,
        ctx: &DeviceContext,
        total_blocks: usize,
        block_size: usize,
    ) -> Result<DFlashDraftKvPool> {
        DFlashDraftKvPool::new(
            ctx,
            self.layers.len(),
            self.kv_dim(),
            total_blocks,
            block_size,
        )
    }

    pub(crate) fn new_materialize_scratch(
        &self,
        ctx: &DeviceContext,
        max_rows: usize,
    ) -> Result<DFlashKvMaterializeScratch> {
        DFlashKvMaterializeScratch::new(ctx, &self.config, max_rows)
    }

    pub(crate) fn kv_dim(&self) -> usize {
        self.config.num_key_value_heads * self.config.head_dim
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
            committed_len: 0,
            max_cache_len,
        })
    }

    /// Materialize one step's target context rows into draft K/V at their
    /// absolute positions. The rows are written both into the request's dense
    /// draft K/V buffer (the draft attention's mirror) and into the paged pool
    /// at the target block slots, so a later prefix hit can restore them without
    /// re-projecting the target hidden.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn materialize_context_kv(
        &self,
        ctx: &DeviceContext,
        state: &mut DFlashRequestState,
        captured_hidden: &HiddenStates,
        token_offset: usize,
        token_count: usize,
        start_pos: usize,
        page_indices: &[i32],
        mut pool: Option<&mut DFlashDraftKvPool>,
        scratch: &mut DFlashKvMaterializeScratch,
    ) -> Result<()> {
        anyhow::ensure!(token_count > 0, "DFlash materialize needs tokens");
        anyhow::ensure!(
            token_count <= scratch.max_rows(),
            "DFlash materialize rows {token_count} exceed scratch {}",
            scratch.max_rows()
        );
        anyhow::ensure!(
            captured_hidden.hidden_dim == self.context_feature_dim(),
            "DFlash captured hidden dim {} does not match expected {}",
            captured_hidden.hidden_dim,
            self.context_feature_dim()
        );
        anyhow::ensure!(
            token_offset + token_count <= captured_hidden.seq_len,
            "DFlash materialize source range [{}, {}) exceeds seq_len {}",
            token_offset,
            token_offset + token_count,
            captured_hidden.seq_len
        );
        anyhow::ensure!(
            start_pos + token_count <= state.max_cache_len,
            "DFlash materialize destination [{}, {}) exceeds max cache {}",
            start_pos,
            start_pos + token_count,
            state.max_cache_len
        );
        if let Some(pool) = pool.as_deref() {
            anyhow::ensure!(
                pool.k.len() == self.layers.len(),
                "DFlash draft KV pool has {} layers, drafter has {}",
                pool.k.len(),
                self.layers.len()
            );
            let needed_pages = (start_pos + token_count).div_ceil(pool.block_size);
            anyhow::ensure!(
                page_indices.len() >= needed_pages,
                "DFlash materialize needs {needed_pages} target pages, got {}",
                page_indices.len()
            );
            let block = pool.block_size;
            for i in 0..token_count {
                let position = start_pos + i;
                let page = page_indices[position / block];
                anyhow::ensure!(
                    page >= 0 && (page as usize) < pool.total_blocks,
                    "DFlash materialize page {page} out of range {}",
                    pool.total_blocks
                );
                scratch.locs_host[i] = (page as usize * block + position % block) as i32;
            }
            let mut locs_dst = scratch.locs.slice_mut(..token_count);
            ctx.stream
                .memcpy_htod(&scratch.locs_host[..token_count], &mut locs_dst)?;
        }

        scratch.capture_rows.seq_len = token_count;
        ops::copy_hidden_token_range_into(
            ctx,
            captured_hidden,
            token_offset,
            &mut scratch.capture_rows,
            0,
            token_count,
        )?;
        scratch.ctx_projected.seq_len = token_count;
        ops::gemm_into(
            ctx,
            &self.fc,
            &scratch.capture_rows,
            &mut scratch.ctx_projected,
        );
        scratch.ctx_hidden.seq_len = token_count;
        ops::rms_norm_batch_into(
            ctx,
            &scratch.ctx_projected,
            &self.hidden_norm,
            self.config.rms_norm_eps,
            &mut scratch.ctx_hidden,
        );

        let q_dim = self.config.num_attention_heads * self.config.head_dim;
        let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            scratch.k.seq_len = token_count;
            scratch.v.seq_len = token_count;
            ops::gemm_rows_into(
                ctx,
                &layer.attention.qkv_proj,
                q_dim,
                kv_dim,
                &scratch.ctx_hidden,
                &mut scratch.k,
            );
            ops::gemm_rows_into(
                ctx,
                &layer.attention.qkv_proj,
                q_dim + kv_dim,
                kv_dim,
                &scratch.ctx_hidden,
                &mut scratch.v,
            );
            scratch.q.seq_len = token_count;
            ops::dflash_qk_norm_rope_into(
                ctx,
                &mut scratch.q,
                0,
                token_count,
                &mut scratch.k,
                &layer.attention.q_norm,
                &layer.attention.k_norm,
                &self.cos_cache,
                &self.sin_cache,
                self.config.num_attention_heads,
                self.config.num_key_value_heads,
                self.config.head_dim,
                start_pos,
                start_pos,
                self.config.rms_norm_eps,
            )?;
            ops::copy_hidden_token_range_into(
                ctx,
                &scratch.k,
                0,
                &mut state.layers[layer_idx].k,
                start_pos,
                token_count,
            )?;
            ops::copy_hidden_token_range_into(
                ctx,
                &scratch.v,
                0,
                &mut state.layers[layer_idx].v,
                start_pos,
                token_count,
            )?;

            if let Some(pool) = pool.as_mut() {
                let pool: &mut DFlashDraftKvPool = pool;
                pegainfer_kernels::ops::draft_kv_scatter_into(
                    ctx,
                    &mut [
                        (&scratch.k, &mut pool.k[layer_idx]),
                        (&scratch.v, &mut pool.v[layer_idx]),
                    ],
                    &scratch.locs,
                    token_count,
                )?;
            }
        }
        Ok(())
    }

    /// Gather a matched prefix's draft K/V out of the paged pool into the
    /// request's dense mirror. The caller has already verified that every block
    /// is tagged with the matched content.
    pub(crate) fn gather_restored_kv(
        ctx: &DeviceContext,
        state: &mut DFlashRequestState,
        pool: &DFlashDraftKvPool,
        locs: &CudaSlice<i32>,
        tokens: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            pool.k.len() == state.layers.len(),
            "DFlash restore pool/drafter layer mismatch: {} vs {}",
            pool.k.len(),
            state.layers.len()
        );
        anyhow::ensure!(
            tokens <= state.max_cache_len,
            "DFlash restore tokens {tokens} exceed max cache {}",
            state.max_cache_len
        );
        {
            let mut pairs: Vec<(&HiddenStates, &mut HiddenStates)> =
                Vec::with_capacity(2 * state.layers.len());
            for (layer_idx, cache) in state.layers.iter_mut().enumerate() {
                pairs.push((&pool.k[layer_idx], &mut cache.k));
                pairs.push((&pool.v[layer_idx], &mut cache.v));
            }
            pegainfer_kernels::ops::draft_kv_gather_into(ctx, &mut pairs, locs, tokens)?;
        }
        state.committed_len = tokens;
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

        // The prefix K/V is already materialized by the prefill/verify path;
        // this round only appends the in-fill block at `committed_len`.
        for state in states.iter() {
            anyhow::ensure!(
                state.committed_len + block_size <= state.max_cache_len,
                "DFlash draft cache overflow: committed={}, block={}, max={}",
                state.committed_len,
                block_size,
                state.max_cache_len
            );
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

            if let (Some(convs), Some(conv_scratch)) = (&self.convs, scratch.conv.as_mut()) {
                convs[layer_idx]
                    .attention
                    .prepare(ctx, &mut scratch.normed, conv_scratch)?;
            }

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
                let row_offset = i * block_size;
                scratch.ensure_tail_capacity(ctx, &self.config, block_size)?;

                // The prefix K/V is already in `state.layers`; the tail is just
                // this request's in-fill block, starting at its current token.
                ops::copy_hidden_token_range_into(
                    ctx,
                    &scratch.normed,
                    row_offset,
                    &mut scratch.tail_input,
                    0,
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
                    state.committed_len,
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
                    block_size,
                )?;
                ops::copy_hidden_token_range_into(
                    ctx,
                    &scratch.v_tail,
                    0,
                    &mut cache.v,
                    state.committed_len,
                    block_size,
                )?;
                // The trained mask keeps W target-context tokens before the
                // anchor and the entire noncausal draft block for every query.
                let kv_start = self
                    .config
                    .sliding_window
                    .map_or(0, |window| state.committed_len.saturating_sub(window));
                pegainfer_kernels::ops::single_prefill_nhd_noncausal_range_into(
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
                    kv_start..state.committed_len + block_size,
                )?;
            }

            // Dense: o_proj + residual + post-attention norm + MLP over the batch.
            ops::gemm_into(
                ctx,
                &layer.attention.o_proj,
                &scratch.attn_output,
                &mut scratch.o_buf,
            );
            if let (Some(convs), Some(conv_scratch)) = (&self.convs, scratch.conv.as_mut()) {
                convs[layer_idx]
                    .attention
                    .finish(ctx, &mut scratch.o_buf, conv_scratch)?;
            }
            pegainfer_kernels::ops::fused_add_rms_norm_round_batch_into(
                ctx,
                &mut scratch.hidden,
                &scratch.o_buf,
                &layer.post_attention_layernorm,
                self.config.rms_norm_eps,
                &mut scratch.normed,
            )?;

            if let (Some(convs), Some(conv_scratch)) = (&self.convs, scratch.conv.as_mut()) {
                convs[layer_idx]
                    .mlp
                    .prepare(ctx, &mut scratch.normed, conv_scratch)?;
            }

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
            if let (Some(convs), Some(conv_scratch)) = (&self.convs, scratch.conv.as_mut()) {
                convs[layer_idx]
                    .mlp
                    .finish(ctx, &mut scratch.o_buf, conv_scratch)?;
            }
            ops::add_batch_into(
                ctx,
                &scratch.hidden,
                &scratch.o_buf,
                &mut scratch.hidden_out,
            )?;
            std::mem::swap(&mut scratch.hidden, &mut scratch.hidden_out);
        }

        self.compute_logits_into(target, scratch)?;
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

    pub(crate) fn uses_selector(&self) -> bool {
        self.selector.is_some()
    }

    pub(crate) fn selected_draft_tokens(
        &self,
        ctx: &DeviceContext,
        anchors: &[u32],
        temperatures: &[f32],
        uniforms: &[f32],
        scratch: &mut DFlashBatchScratch,
    ) -> Result<Vec<u32>> {
        let head = self
            .selector
            .as_ref()
            .context("DFlash2 selector was not loaded")?;
        let selector = scratch
            .selector
            .as_mut()
            .context("DFlash2 scratch was not allocated")?;
        head.select(
            ctx,
            &scratch.logits_normed,
            &scratch.logits,
            anchors,
            temperatures,
            uniforms,
            self.block_size(),
            selector,
        )
    }

    pub(crate) fn context_feature_dim(&self) -> usize {
        self.config.hidden_size * self.target_layer_ids().len()
    }

    fn compute_logits_into(
        &self,
        target: &Qwen3Model,
        scratch: &mut DFlashBatchScratch,
    ) -> Result<()> {
        let ctx = target.device_ctx();
        ops::rms_norm_batch_into(
            ctx,
            &scratch.hidden,
            &self.norm,
            self.config.rms_norm_eps,
            &mut scratch.logits_normed,
        );
        target.project_logits(ctx, &scratch.logits_normed, &mut scratch.logits)
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::DFlashConfig;
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
        let dflash = DFlashConfig::from_file(&dflash_path).expect("DFlash config");
        dflash
            .validate_for_target(&target)
            .expect("DFlash config should match target");

        assert_eq!(dflash.block_size, 16);
        assert_eq!(dflash.mask_token_id, 151_669);
        assert_eq!(dflash.target_layer_ids, vec![1, 9, 17, 25, 33]);

        // Pin the memory reservation the KV budget bills against. The per-token
        // term (draft KV 5*2*1024*2 + scratch-context (3*2560+2*1024)*2 + pending
        // 2560*5*2) drives the ~12% block haircut; a layer-count or geometry
        // regression here would silently over/under-reserve and risk OOM.
        let reservation = super::DFlashMemoryReservation::from_config(
            &dflash, /*max_decode_batch*/ 256, 1024,
        )
        .expect("reservation sizing");
        assert_eq!(
            reservation.kv_bytes_per_token, 65_536,
            "conservative draft term: pool + mirror + tail + retired capture buffers"
        );
        assert_eq!(
            reservation.request_state_bytes_per_token, 20_480,
            "per-request dense draft K/V mirror only; the paged pool is shared"
        );
        // Weights (~1.1 GiB) dominate the fixed term at batch=1; the block-sized
        // per-request scratch (~6.5 MiB, logits-heavy) plus the one-block KV/tail
        // headroom (~0.5 MiB) add across the decode batch.
        let fixed_batch1 = super::DFlashMemoryReservation::from_config(&dflash, 1, 1024)
            .expect("reservation sizing")
            .fixed_bytes;
        assert!(
            (1_150_000_000..1_280_000_000).contains(&fixed_batch1),
            "draft weights ~1.1GiB + materialize scratch, got {fixed_batch1}"
        );
        assert!(
            (2_900_000_000..3_120_000_000).contains(&reservation.fixed_bytes),
            "weights + 256 * (~6.5MiB scratch + ~0.5MiB block headroom) + materialize, got {}",
            reservation.fixed_bytes
        );
    }
}
