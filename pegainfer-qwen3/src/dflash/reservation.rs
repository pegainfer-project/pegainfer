use anyhow::Result;

use crate::config::DFlashConfig;
use crate::config::DFlashHeadSource;
use crate::config::DFlashProposal;
use crate::dflash::selector::SelectorScratch;
use crate::dspark::MarkovHead;

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
    pub(crate) fixed_bytes: usize,
}

impl DFlashMemoryReservation {
    pub(crate) fn from_path(draft_path: &str, max_decode_batch_size: usize) -> Result<Self> {
        let config = DFlashConfig::from_file(draft_path)?;
        config.validate_runtime_capabilities()?;
        Ok(Self::from_config(&config, max_decode_batch_size))
    }

    pub(crate) fn from_config(config: &DFlashConfig, max_decode_batch_size: usize) -> Self {
        const BF16: usize = 2;
        let hidden = config.hidden_size;
        let target_hidden = config.target_hidden_size;
        let kv_dim = config.num_key_value_heads * config.head_dim;
        let q_dim = config.num_attention_heads * config.head_dim;
        let inter = config.intermediate_size;
        let capture_layers = config.target_layer_ids.len();

        // Per-sequence-token, pool-scaling buffers.
        let draft_kv = config.num_hidden_layers * 2 * kv_dim * BF16; // DFlashLayerCache k+v
        // Scratch split by what it tracks: `context_*` grows with the committed
        // prefix; `tail_*` (tail_input + k_tail + v_tail) grows with the in-fill
        // tail, which is one block past the prefix.
        let context_scratch = 2 * hidden * BF16; // context_projected + context_hidden
        let tail_scratch = (hidden + 2 * kv_dim) * BF16; // tail_input + k_tail + v_tail
        let pending = target_hidden * capture_layers * BF16; // context_feature_dim
        let kv_bytes_per_token = draft_kv + context_scratch + tail_scratch + pending;

        // Lane-level batched dense scratch: every dense buffer is sized for the
        // whole decode batch (`max_batch * block_size` rows), allocated once.
        // Same total magnitude as the old per-request scratch summed over the
        // batch, but now one contiguous allocation.
        let dense_scratch_per_block_row =
            BF16 * (config.vocab_size + 5 * hidden + 2 * q_dim + 3 * inter);
        let scratch_total = dense_scratch_per_block_row * config.block_size * max_decode_batch_size;

        // Draft weights (5 transformer layers + the context projection), +10% slack
        // for norms, rope caches, and allocator alignment.
        let per_layer = BF16
            * (hidden * (q_dim + 2 * kv_dim) // qkv_proj
                + q_dim * hidden // o_proj
                + hidden * 2 * inter // gate_up_proj
                + inter * hidden); // down_proj
        let fc = BF16 * hidden * (target_hidden * capture_layers); // context projection
        let weights = per_layer * config.num_hidden_layers + fc;
        let weights = weights + weights / 10;

        // The durable draft KV and the tail scratch are sized to `context +
        // block_size` — one in-fill block past the lifetime the KV pool reserves
        // for the request. The per-token term bills only the pool's tokens, so
        // reserve that one-block headroom per concurrently decoding request to
        // keep the reservation an upper bound.
        let block_headroom = max_decode_batch_size * config.block_size * (draft_kv + tail_scratch);

        // DSpark Markov head: weights (2 × vocab × rank) + sample scratch (the
        // per-step bias is the dominant term). Zero for plain DFlash drafters.
        let markov = MarkovHead::reservation_bytes(config, max_decode_batch_size);
        let selector = match &config.proposal {
            DFlashProposal::TopKSelector { rank, .. } => {
                // Selector weights and persistent launch scratch.
                let weights = BF16 * (rank * hidden + 2 * config.draft_vocab_size * rank);
                weights + SelectorScratch::bytes(config, max_decode_batch_size)
            }
            _ => 0,
        };
        let native_head = match config.head_source {
            DFlashHeadSource::DraftOutput => BF16 * config.draft_vocab_size * hidden,
            DFlashHeadSource::Target => 0,
        };

        Self {
            kv_bytes_per_token,
            fixed_bytes: weights + scratch_total + block_headroom + markov + selector + native_head,
        }
    }
}
