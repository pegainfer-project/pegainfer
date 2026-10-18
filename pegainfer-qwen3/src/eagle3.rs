// EAGLE-3 drafter (e.g. `AngelSlim/Qwen3-4B_eagle3`).
#![allow(dead_code)]

use anyhow::Result;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::DeviceVec;

use crate::config::Eagle3Config;

mod loading;
mod reservation;

// Wired into the KV budget by the forward-pass PR; kept here as the skeleton lands.
#[allow(unused_imports)]
pub(crate) use reservation::Eagle3MemoryReservation;

/// Number of tokens the EAGLE-3 chain drafts per speculative round (γ; the verify
/// span is `EAGLE3_CHAIN_LENGTH + 1`: the current token plus γ drafts). v1 is a
/// fixed top-1 **chain** — *not* a tree. γ=3 is the measured chain optimum on a Qwen3-4B GSM8K A/B (RTX 5070 Ti)
pub(crate) const EAGLE3_CHAIN_LENGTH: usize = 3;

/// The three *target* layers (low/mid/high) whose **post-layer** hidden states
/// EAGLE-3 captures and fuses via `fc`.
///
/// SafeAILab's `traineagle3` gates on `(2, N/2, N-3)` but appends the residual
/// stream *before* running those layers (vLLM matches it with a post-layer
/// `idx + 1` capture) — so both land on the residual stream *entering* those
/// layers. Our capture path is post-layer / 0-based (`prefill.rs`,
/// `verify_graph.rs`), so the equivalent indices are one block earlier:
/// `(1, N/2-1, N-4)` — post-layer outputs `[1, 17, 32]` for the 36-layer
/// Qwen3-4B target. See PR #662 discussion.
pub(crate) fn aux_hidden_state_layers(target_num_layers: usize) -> Result<[usize; 3]> {
    let low = 1;
    let mid = (target_num_layers / 2).saturating_sub(1);
    let high = target_num_layers.saturating_sub(4);
    anyhow::ensure!(
        low < mid && mid < high && high < target_num_layers,
        "EAGLE-3 aux layers ({low}, {mid}, {high}) are not strictly increasing within \
         {target_num_layers} target layers"
    );
    Ok([low, mid, high])
}

/// The single EAGLE-3 decoder block (`midlayer`).
///
/// Differs from the Qwen3 target's [`crate::weights::TransformerBlock`] in two
/// load-bearing ways:
/// 1. attention input is `2 * hidden_size` — EAGLE-3 concatenates
///    `[input_layernorm(embed), hidden_norm(fused_hidden)]` before q/k/v, so the
///    q/k/v projections have `2 * hidden_size` input columns;
/// 2. there is no QK-norm
pub(crate) struct Eagle3Layer {
    /// RMSNorm applied to the input token embedding.
    pub(crate) input_layernorm: DeviceVec,
    /// RMSNorm applied to the fused target hidden state.
    pub(crate) hidden_norm: DeviceVec,
    /// `vstack(q_proj, k_proj, v_proj)`; input dim is `2 * hidden_size`.
    pub(crate) qkv_proj: DeviceMatrix,
    pub(crate) o_proj: DeviceMatrix,
    pub(crate) post_attention_layernorm: DeviceVec,
    /// `vstack(gate_proj, up_proj)`.
    pub(crate) gate_up_proj: DeviceMatrix,
    pub(crate) down_proj: DeviceMatrix,
    /// Rows of `q_proj` (`num_attention_heads * head_dim`).
    pub(crate) q_dim: usize,
    /// Rows of `k_proj`/`v_proj` (`num_key_value_heads * head_dim`).
    pub(crate) kv_dim: usize,
}

/// EAGLE-3 draft model. Reuses the target's `embed_tokens` (the checkpoint ships
/// no embedding), so the embedding is not stored here.
pub(crate) struct Eagle3DraftModel {
    pub(crate) config: Eagle3Config,
    /// Fuses the captured low/mid/high target hidden states:
    /// `[3 * hidden_size] -> [hidden_size]`.
    pub(crate) fc: DeviceMatrix,
    pub(crate) midlayer: Eagle3Layer,
    /// Final RMSNorm before the draft head.
    pub(crate) norm: DeviceVec,
    /// Draft head over the reduced `draft_vocab_size`.
    pub(crate) lm_head: DeviceMatrix,
    /// Draft→target vocab offset map: `target_id = draft_id + d2t[draft_id]`.
    /// Length `draft_vocab_size`. Host-resident (small lookup, not a GEMM input).
    pub(crate) d2t: Vec<i64>,
    /// Target→draft presence mask: `t2d[target_id]` is true iff that target token
    /// exists in the draft vocab. Length `vocab_size`.
    pub(crate) t2d: Vec<bool>,
    pub(crate) cos_cache: DeviceVec,
    pub(crate) sin_cache: DeviceVec,
}

#[cfg(test)]
mod tests {
    use super::aux_hidden_state_layers;

    #[test]
    fn aux_layers_reject_tiny_target() {
        // Too few layers to place a strictly-increasing low/mid/high triple.
        assert!(aux_hidden_state_layers(6).is_err());
        assert!(aux_hidden_state_layers(4).is_err());
    }
}
