//! The text tower's head and tail: the embedding scale, the token-id
//! validation every entry point runs, and the final norm plus tied LM head
//! with its logit softcap.

use anyhow::Result;
use half::bf16;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;

use crate::weights::Gemma4Weights;

pub(crate) const MULTIMODAL_PLACEHOLDER_IDS: [u32; 6] =
    [255_999, 256_000, 258_880, 258_881, 258_882, 258_883];

/// The embedding multiplier is the bf16 rounding of `sqrt(hidden_size)` —
/// the reference casts the scale buffer to the weight dtype before the
/// multiply, so at hidden 3840 this is exactly 62.0, not 61.9677, and the
/// 5.2e-4 relative gap is far too large to pass off as accumulation noise.
pub(crate) fn embed_scale_bf16(hidden_size: usize) -> f32 {
    bf16::from_f32((hidden_size as f32).sqrt()).to_f32()
}

/// Validate host token ids before the text-only embedding kernel, which has
/// neither bounds checks nor a multimodal embedder.
pub(crate) fn validate_tokens(
    weights: &Gemma4Weights,
    hidden_size: usize,
    tokens: &[u32],
) -> Result<()> {
    let vocab_size = weights.embed_tokens.rows;
    anyhow::ensure!(
        weights.embed_tokens.cols == hidden_size,
        "embedding width {} != config hidden_size {hidden_size}",
        weights.embed_tokens.cols
    );
    for (position, &token) in tokens.iter().enumerate() {
        anyhow::ensure!(
            !MULTIMODAL_PLACEHOLDER_IDS.contains(&token),
            "text-only Gemma 4 cannot embed multimodal placeholder token {token} \
             at position {position}"
        );
        anyhow::ensure!(
            (token as usize) < vocab_size,
            "token {token} at position {position} is outside the embedding's {vocab_size} rows"
        );
    }
    Ok(())
}

/// Shared with the KV-backed serving path, which runs it over whichever rows
/// it needs logits for.
pub(crate) fn logits_tail(
    ctx: &DeviceContext,
    weights: &Gemma4Weights,
    hidden: &HiddenStates,
    rms_norm_eps: f32,
    final_logit_softcapping: f32,
) -> Result<HiddenStates> {
    let seq_len = hidden.seq_len;
    let mut normed = HiddenStates::zeros(ctx, hidden.hidden_dim, seq_len)?;
    let mut logits = HiddenStates::zeros(ctx, weights.embed_tokens.rows, seq_len)?;
    logits_tail_into(
        ctx,
        weights,
        hidden,
        rms_norm_eps,
        final_logit_softcapping,
        &mut normed,
        &mut logits,
    )?;
    Ok(logits)
}

/// The arena form: `normed` and `logits` are caller-owned buffers reshaped to
/// this call's row count, so the decode loop reaches the head without
/// allocating.
pub(crate) fn logits_tail_into(
    ctx: &DeviceContext,
    weights: &Gemma4Weights,
    hidden: &HiddenStates,
    rms_norm_eps: f32,
    final_logit_softcapping: f32,
    normed: &mut HiddenStates,
    logits: &mut HiddenStates,
) -> Result<()> {
    use anyhow::Context as _;
    let seq_len = hidden.seq_len;
    let vocab_size = weights.embed_tokens.rows;
    // The ops assert that the tensors agree with each other, not that the
    // allocation behind them is long enough, so both buffers are checked
    // before the first kernel.
    let normed_elems = hidden
        .hidden_dim
        .checked_mul(seq_len)
        .context("head normed extent overflows")?;
    anyhow::ensure!(
        normed.data.len() >= normed_elems,
        "head normed buffer holds {} elements, not {normed_elems}",
        normed.data.len()
    );
    let logits_elems = vocab_size
        .checked_mul(seq_len)
        .context("head logits extent overflows")?;
    anyhow::ensure!(
        logits.data.len() >= logits_elems,
        "head logits buffer holds {} elements, not {logits_elems}",
        logits.data.len()
    );
    normed.hidden_dim = hidden.hidden_dim;
    normed.seq_len = seq_len;
    ops::rms_norm_batch_into(ctx, hidden, &weights.norm, rms_norm_eps, normed);

    // Tied embeddings: the LM head is the embedding matrix itself.
    logits.hidden_dim = vocab_size;
    logits.seq_len = seq_len;
    ops::gemm_rows_into_checked(ctx, &weights.embed_tokens, 0, vocab_size, normed, logits)?;
    ops::softcap_bf16_in_place(ctx, logits, final_logit_softcapping)
}
