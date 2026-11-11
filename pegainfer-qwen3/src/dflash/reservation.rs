use anyhow::Result;

use super::selector::SelectorHead;
use crate::config::DFlashConfig;
use crate::dspark::MarkovHead;
use crate::sizing;

/// GPU memory DFlash needs on top of the target KV pool, derived from the draft
/// config so the KV budget can reserve it *before* the draft model loads (the
/// draft buffers live outside the paged `KvCacheManager`). Split by how it scales:
///
/// - `kv_bytes_per_token` scales with the KV pool (billed by shrinking the target
///   block count): the draft's own KV cache plus the per-request context-projection
///   and pending-context buffers, which currently persist at prompt length per
///   request (see `dflash-speculative-decoding.md` — collapsing that persistence
///   is a tracked follow-up that would shrink this term to the draft KV alone).
/// - `fixed_bytes` does not scale with the pool (billed via the memory margin):
///   the draft weights plus the lane-level batched scratch sized for the whole
///   decode batch.
// TODO: the draft scratch is now a single lane-level `DFlashBatchScratch`
// allocated once (dense buffers sized `max_batch * block_size`, plus one shared
// varlen tail), not a per-request buffer. The per-token `tail_scratch` term and
// the per-request `block_headroom` tail term are therefore over-estimates — kept
// as a conservative upper bound until the accounting is retuned against the
// batched allocation.
pub(crate) struct DFlashMemoryReservation {
    pub(crate) kv_bytes_per_token: usize,
    /// The subset of [`Self::kv_bytes_per_token`] each *request* owns: the draft
    /// KV, the projected-context scratch, and the pending capture buffer. The
    /// tail scratch is lane-level and shared, so it is excluded here. The lane
    /// charges this per token of a request's `max_cache_len` against the draft
    /// budget, which is what keeps shared prefixes from overcommitting state
    /// the pool reservation only billed once.
    pub(crate) request_state_bytes_per_token: usize,
    pub(crate) fixed_bytes: usize,
    /// Drafter block size, surfaced so startup budgets (hedge scratch pages,
    /// verify-plan metadata) can share the config read.
    pub(crate) block_size: usize,
    /// Whether the drafter has a Markov head: a plain DFlash checkpoint can
    /// never hedge, so startup must not reserve hedge scratch for it even
    /// under a nonzero hedge env.
    pub(crate) uses_markov_head: bool,
}

impl DFlashMemoryReservation {
    pub(crate) fn from_path(
        draft_path: &str,
        logits_rows: usize,
        max_decode_batch_size: usize,
        max_prefill_tokens: usize,
    ) -> Result<Self> {
        let mut config = DFlashConfig::from_file(draft_path)?;
        config.logits_rows = logits_rows;
        Self::from_config(&config, max_decode_batch_size, max_prefill_tokens)
    }

    pub(crate) fn from_config(
        config: &DFlashConfig,
        max_decode_batch_size: usize,
        max_prefill_tokens: usize,
    ) -> Result<Self> {
        const BF16: usize = 2;
        let hidden = config.hidden_size;
        let kv_dim = sizing::product(&[config.num_key_value_heads, config.head_dim])?;
        let q_dim = sizing::product(&[config.num_attention_heads, config.head_dim])?;
        let inter = config.intermediate_size;
        let capture_layers = config.target_layer_ids.len();

        // Per-sequence-token, pool-scaling buffers.
        // DFlashLayerCache k+v
        let draft_kv = sizing::product(&[config.num_hidden_layers, 2, kv_dim, BF16])?;
        // Scratch split by what it tracks: `context_*` grows with the committed
        // prefix; `tail_input` grows with the in-fill tail, one block past it.
        let context_scratch = sizing::product(&[2, hidden, BF16])?; // context_projected + context_hidden
        let tail_scratch = sizing::product(&[hidden, BF16])?;
        let pending = sizing::product(&[hidden, capture_layers, BF16])?; // context_feature_dim
        // Conservative per-token term: it covers the paged draft KV pool
        // (draft_kv), the per-request mirror (draft_kv), the shared tail scratch,
        // and the (now unused) capture/context buffers. The stage-2 pool is
        // addressed by target block IDs and shares the target radix cache's
        // lifetime; a later pass can tighten this term once the lane-level
        // materialize scratch is billed explicitly.
        let kv_bytes_per_token = sizing::sum(&[draft_kv, context_scratch, tail_scratch, pending])?;
        // Per-request state is just the dense draft K/V mirror; the paged pool
        // is shared by every request.
        let request_state_bytes_per_token = draft_kv;

        // Lane-level batched dense scratch: every dense buffer is sized for the
        // whole decode batch (`max_batch * block_size` rows), allocated once.
        // Same total magnitude as the old per-request scratch summed over the
        // batch, but now one contiguous allocation.
        let dense_scratch_per_block_row = sizing::product(&[
            BF16,
            sizing::sum(&[
                config.logits_rows,
                sizing::product(&[5, hidden])?,
                sizing::product(&[2, q_dim])?,
                sizing::product(&[3, inter])?,
            ])?,
        ])?;
        let scratch_total = sizing::product(&[
            dense_scratch_per_block_row,
            config.block_size,
            max_decode_batch_size,
        ])?;

        // Draft weights (5 transformer layers + the context projection), +10% slack
        // for norms, rope caches, and allocator alignment.
        let per_layer = sizing::product(&[
            BF16,
            sizing::sum(&[
                sizing::product(&[
                    hidden,
                    sizing::sum(&[q_dim, sizing::product(&[2, kv_dim])?])?,
                ])?, // qkv_proj
                sizing::product(&[q_dim, hidden])?,    // o_proj
                sizing::product(&[hidden, 2, inter])?, // gate_up_proj
                sizing::product(&[inter, hidden])?,    // down_proj
            ])?,
        ])?;
        // context projection
        let fc = sizing::product(&[BF16, hidden, hidden, capture_layers])?;
        let weights = sizing::sum(&[sizing::product(&[per_layer, config.num_hidden_layers])?, fc])?;
        let weights = sizing::sum(&[weights, weights / 10])?;

        // The durable draft KV and the tail scratch are sized to `context +
        // block_size` — one in-fill block past the lifetime the KV pool reserves
        // for the request. The per-token term bills only the pool's tokens, so
        // reserve that one-block headroom per concurrently decoding request to
        // keep the reservation an upper bound.
        let block_headroom = sizing::product(&[
            max_decode_batch_size,
            config.block_size,
            sizing::sum(&[draft_kv, tail_scratch])?,
        ])?;

        let native = if let Some(conv) = &config.conv {
            let dynamic_width = sizing::product(&[2, conv.taps, hidden / conv.group_size])?;
            let conv_weights = sizing::product(&[
                config.num_hidden_layers,
                2,
                BF16,
                hidden,
                sizing::sum(&[2 * conv.taps, dynamic_width])?,
            ])?;
            let conv_scratch = sizing::product(&[
                max_decode_batch_size,
                config.block_size,
                BF16,
                sizing::sum(&[hidden, dynamic_width])?,
            ])?;
            sizing::sum(&[conv_weights, conv_scratch])?
        } else {
            0
        };

        // Lane-level materialize scratch: captured rows, the fused context, a
        // dummy query window, one layer's K/V, and the slot index list. Sized
        // for the largest prefill chunk (or one draft block), allocated once.
        let materialize_rows = max_prefill_tokens.max(config.block_size.saturating_add(1));
        let capture_dim = sizing::product(&[hidden, capture_layers])?;
        let materialize = sizing::sum(&[
            sizing::product(&[capture_dim, materialize_rows, BF16])?,
            sizing::product(&[2 * hidden, materialize_rows, BF16])?,
            sizing::product(&[q_dim, materialize_rows, BF16])?,
            sizing::product(&[2 * kv_dim, materialize_rows, BF16])?,
            sizing::product(&[materialize_rows, 4])?,
        ])?;

        // DSpark Markov head: weights (2 × vocab × rank) + sample scratch (the
        // per-step bias is the dominant term). Zero for plain DFlash drafters.
        let markov = MarkovHead::reservation_bytes(config, max_decode_batch_size)?;
        let selector = SelectorHead::reservation_bytes(config, max_decode_batch_size)?;
        let sampling = if config.selector_rank > 0 {
            pegainfer_kernels::ops::SparseVerifyScratch::reservation_bytes(
                max_decode_batch_size,
                config.block_size,
            )
        } else {
            0
        };
        Ok(Self {
            kv_bytes_per_token,
            request_state_bytes_per_token,
            fixed_bytes: sizing::sum(&[
                weights,
                scratch_total,
                block_headroom,
                materialize,
                markov,
                native,
                selector,
                sampling,
            ])?,
            block_size: config.block_size,
            uses_markov_head: config.uses_markov_head(),
        })
    }
}
