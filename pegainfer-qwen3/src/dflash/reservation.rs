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
///   decode batch, and (for DSpine) the bridge weights and vocabulary-sized
///   TG/TSV tables. The bridge scratch is billed by the allocation PR.
// TODO: the draft scratch is now a single lane-level `DFlashBatchScratch`
// allocated once (dense buffers sized `max_batch * block_size`, plus one shared
// varlen tail), not a per-request buffer. The per-token `tail_scratch` term and
// the per-request `block_headroom` tail term are therefore over-estimates — kept
// as a conservative upper bound until the accounting is retuned against the
// batched allocation.
pub(crate) struct DFlashMemoryReservation {
    pub(crate) kv_bytes_per_token: usize,
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
    ) -> Result<Self> {
        let mut config = DFlashConfig::from_file(draft_path)?;
        config.logits_rows = logits_rows;
        Self::from_config(&config, max_decode_batch_size)
    }

    pub(crate) fn from_config(config: &DFlashConfig, max_decode_batch_size: usize) -> Result<Self> {
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
        // prefix; `tail_*` (tail_input + k_tail + v_tail) grows with the in-fill
        // tail, which is one block past the prefix.
        let context_scratch = sizing::product(&[2, hidden, BF16])?; // context_projected + context_hidden
        // tail_input + k_tail + v_tail
        let tail_scratch = sizing::product(&[
            sizing::sum(&[hidden, sizing::product(&[2, kv_dim])?])?,
            BF16,
        ])?;
        let pending = sizing::product(&[hidden, capture_layers, BF16])?; // context_feature_dim
        let kv_bytes_per_token = sizing::sum(&[draft_kv, context_scratch, tail_scratch, pending])?;

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

        // DSpark Markov head: weights (2 × vocab × rank) + sample scratch (the
        // per-step bias is the dominant term). Zero for plain DFlash drafters.
        let markov = MarkovHead::reservation_bytes(config, max_decode_batch_size)?;
        let selector = SelectorHead::reservation_bytes(config, max_decode_batch_size)?;
        let dspine = dspine_bytes(config)?;

        Ok(Self {
            kv_bytes_per_token,
            fixed_bytes: sizing::sum(&[
                weights,
                scratch_total,
                block_headroom,
                markov,
                native,
                selector,
                dspine,
            ])?,
            block_size: config.block_size,
            uses_markov_head: config.uses_markov_head(),
        })
    }
}

/// Fixed DSpine footprint: bridge weights and vocabulary-sized TG/TSV tables,
/// the dimensions of which come straight from the checkpoint header. `0` for
/// non-DSpine drafters. The bridge/serial scratch is billed together with the
/// allocation that defines its layout.
fn dspine_bytes(config: &DFlashConfig) -> Result<usize> {
    const BF16: usize = 2;
    if !config.uses_dspine() {
        return Ok(0);
    }
    let dspine = config
        .dspine
        .as_ref()
        .expect("uses_dspine implies a parsed bridge");
    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let layers = config.num_hidden_layers;
    let code = dspine.code_size;
    let rank = dspine.rank;

    sizing::sum(&[
        sizing::product(&[vocab, code, BF16])?,  // codebook
        sizing::product(&[code, hidden, BF16])?, // read_weight
        sizing::product(&[layers.saturating_sub(1), code, hidden, BF16])?, // early reads
        sizing::product(&[layers, hidden, rank, BF16])?, // feature writes
        sizing::product(&[3, rank, code, BF16])?, // packed gate/predecessor/prev
        sizing::product(&[rank, BF16])?,         // gate bias
        sizing::product(&[vocab, rank, BF16])?,  // TG
        sizing::product(&[vocab, 2, rank, BF16])?, // TSV
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tests::dspine_8b_config;
    use crate::config::tests::plain_dflash_config;

    const MAX_BATCH: usize = 256;
    /// codebook + read/early-read/write + packed/bias + TG/TSV (843,973,632 B).
    const DSPINE_FIXED_BYTES: usize = 843_973_632;

    #[test]
    fn dspine_bridge_is_billed_in_fixed_bytes() {
        let plain = DFlashMemoryReservation::from_config(&plain_dflash_config(), MAX_BATCH)
            .expect("plain DFlash reservation");
        let dspine = DFlashMemoryReservation::from_config(&dspine_8b_config(), MAX_BATCH)
            .expect("DSpine reservation");

        assert_eq!(dspine.kv_bytes_per_token, plain.kv_bytes_per_token);
        assert_eq!(dspine.fixed_bytes - plain.fixed_bytes, DSPINE_FIXED_BYTES);
        assert_eq!(
            dspine_bytes(&plain_dflash_config()).unwrap(),
            0,
            "plain DFlash must not bill DSpine memory"
        );
    }
}
