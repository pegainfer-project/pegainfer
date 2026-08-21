//! Full 12B text-tower forward, prefill form: scaled embeddings, all 48
//! layers in their real local/global order, the final norm, the tied LM
//! head, and the final-logit softcap.
//!
//! No KV cache, no batching, no sliding-window truncation: every prompt this
//! runs is at or below the 1024-token window, where causal attention IS
//! sliding attention (the window first evicts at 1025 tokens, established by
//! measurement in the golden fixture). Crossing the window belongs to the KV
//! cache work, not here.

use anyhow::Result;
use half::bf16;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

use crate::config::LayerKind;
use crate::layer::LayerGeometry;
use crate::layer::global_layer_forward;
use crate::layer::local_layer_forward;
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

/// Runs the full text tower over `tokens` at positions `0..len`, returning
/// softcapped logits (`[vocab_size, seq_len]`).
pub(crate) fn full_forward(
    ctx: &DeviceContext,
    weights: &Gemma4Weights,
    tokens: &[u32],
    sliding_tables: (&DeviceVec, &DeviceVec),
    global_tables: (&DeviceVec, &DeviceVec),
    cos_max_pos: usize,
) -> Result<HiddenStates> {
    let config = &weights.config;
    let local_geom = LayerGeometry::local_of(config);
    let global_geom = LayerGeometry::global_of(config);
    let seq_len = tokens.len();
    anyhow::ensure!(seq_len > 0, "full forward needs at least one token");
    anyhow::ensure!(
        seq_len <= cos_max_pos,
        "full forward seq_len {seq_len} exceeds the rope tables' {cos_max_pos} rows"
    );
    validate_tokens(weights, config.hidden_size, tokens)?;

    let ids = ctx
        .stream
        .clone_htod(tokens)
        .map_err(|e| anyhow::anyhow!("token ids H2D failed: {e}"))?;
    let mut hidden = HiddenStates::zeros(ctx, config.hidden_size, seq_len)?;
    ops::embedding_batch(ctx, &weights.embed_tokens, &ids, &mut hidden)?;
    ops::scale_bf16_in_place(ctx, &mut hidden, embed_scale_bf16(config.hidden_size))?;

    for (index, kind) in config.layer_types.iter().enumerate() {
        let layer = &weights.layers[index];
        hidden = match kind {
            LayerKind::Sliding => local_layer_forward(
                ctx,
                layer,
                &local_geom,
                &hidden,
                0,
                config.sliding_window,
                sliding_tables.0,
                sliding_tables.1,
                cos_max_pos,
            )?,
            LayerKind::Global => global_layer_forward(
                ctx,
                layer,
                &global_geom,
                &hidden,
                0,
                global_tables.0,
                global_tables.1,
                cos_max_pos,
            )?,
        };
    }

    logits_tail(
        ctx,
        weights,
        &hidden,
        config.rms_norm_eps,
        config.final_logit_softcapping,
    )
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

#[cfg(test)]
mod oracle {
    use super::*;
    use crate::config::Gemma4Config;
    use crate::layer::build_proportional_rope_tables;
    use crate::testkit::GOLDEN_PATH;
    use crate::testkit::METADATA_KEY;
    use crate::testkit::assert_checkpoint_matches;
    use crate::testkit::f32_tensor;
    use crate::testkit::i32_tensor;
    use crate::testkit::log_softmax_at;
    use crate::testkit::model_path;

    #[test]
    #[ignore = "requires the pinned 12B checkpoint via PEGAINFER_TEST_MODEL_PATH and a GPU"]
    // float_cmp: the embed-scale assert is intentionally exact — both sides
    // are the same bf16-representable value.
    #[allow(clippy::float_cmp)]
    fn full_forward_matches_hf_topk() {
        let dir = model_path();
        let fixture_bytes = std::fs::read(GOLDEN_PATH).expect("read fixture");
        let (_, meta) =
            safetensors::SafeTensors::read_metadata(&fixture_bytes).expect("fixture metadata");
        let manifest: serde_json::Value = serde_json::from_str(
            meta.metadata()
                .as_ref()
                .expect("fixture metadata map")
                .get(METADATA_KEY)
                .expect("gemma4_golden metadata key"),
        )
        .expect("parse fixture manifest");
        assert_checkpoint_matches(&manifest, &dir);
        let fixture = safetensors::SafeTensors::deserialize(&fixture_bytes).expect("parse fixture");

        let config = Gemma4Config::from_file(&dir).expect("config");
        // The scale chain is load-bearing enough that the fixture records it;
        // our derivation must agree before it multiplies anything.
        assert_eq!(
            f64::from(embed_scale_bf16(config.hidden_size)),
            manifest["embed_scale_bf16"]
                .as_f64()
                .expect("embed_scale_bf16"),
            "embed scale disagrees with the fixture's recorded bf16 value"
        );
        let (weights, _) =
            Gemma4Weights::from_safetensors(&dir, 0, config.clone()).expect("load 12B weights");
        let ctx = DeviceContext::new_with_device(0).expect("device context");

        let cos_max_pos = 1024;
        let (scos, ssin) = pegainfer_core::rope::precompute_rope(
            &ctx,
            &pegainfer_core::rope::RopeTableSpec {
                rotary_dim: config.head_dim,
                frequency_dim: config.head_dim,
                max_seq_len: cos_max_pos,
                theta: config.sliding_rope_theta,
            },
        )
        .expect("sliding rope tables");
        let (gcos, gsin) = build_proportional_rope_tables(
            &ctx,
            config.global_rope_theta,
            config.global_head_dim,
            config.global_rotary_dim,
            cos_max_pos,
        )
        .expect("proportional rope tables");

        // Per-case gates declared from measurement. The one- and nine-token
        // cases sit at the bf16 rounding floor (measured maxima 0.162 and
        // 0.638) and gate tightly with top-1 exact. At 1024 tokens two HF
        // attention backends on the same weights and the same sequence —
        // sdpa vs eager — agree on top-1 at only 802/1024 with max
        // |dlogprob| 16.4, spread across all four quarters
        // (224/186/199/193, worst 72.7%); ours measured 821/1024 and 16.19
        // against the sdpa-dumped fixture, at that floor. The edge gate is
        // therefore distributional — a global top-1 floor and a chaos
        // ceiling on the gap — plus a per-quarter top-1 floor with headroom
        // below the baseline's worst quarter, bounding localized damage: a
        // whole-quarter failure cannot hide inside a passing global rate.
        let mut over_tolerance: Vec<String> = Vec::new();
        for (case, tol, top1_floor, quarter_floor) in [
            ("single", 0.3f32, 1.0f64, 1.0f64),
            ("short", 0.9, 1.0, 1.0),
            ("edge", 20.0, 0.75, 0.65),
        ] {
            let (tshape, tokens_i32) = i32_tensor(&fixture, &format!("{case}_tokens"));
            assert_eq!(tshape.len(), 1, "{case}_tokens rank");
            let tokens: Vec<u32> = tokens_i32
                .iter()
                .map(|&t| u32::try_from(t).expect("token id fits u32"))
                .collect();

            let logits = full_forward(
                &ctx,
                &weights,
                &tokens,
                (&scos, &ssin),
                (&gcos, &gsin),
                cos_max_pos,
            )
            .expect("full forward");

            let vocab = weights.embed_tokens.rows;
            let host = logits.to_host(&ctx).expect("logits D2H");
            let (id_shape, ref_ids) = i32_tensor(&fixture, &format!("{case}_topk_ids"));
            let (lp_shape, ref_lps) = f32_tensor(&fixture, &format!("{case}_topk_logprobs"));
            assert_eq!(id_shape, lp_shape, "{case} topk shapes");
            let (seq_len, top_k) = (id_shape[0], id_shape[1]);
            assert_eq!(seq_len, tokens.len(), "{case} topk seq length");

            let mut max_abs = 0.0f32;
            let mut worst = (0usize, 0usize);
            let mut top1_hits = 0usize;
            let mut first_miss: Option<usize> = None;
            let quarter = seq_len.div_ceil(4);
            let mut q_hits = [0usize; 4];
            let mut q_total = [0usize; 4];
            let mut q_max = [0.0f32; 4];
            for pos in 0..seq_len {
                let row = &host[pos * vocab..(pos + 1) * vocab];
                let ids = &ref_ids[pos * top_k..(pos + 1) * top_k];
                let lps = &ref_lps[pos * top_k..(pos + 1) * top_k];
                let (ours, argmax) = log_softmax_at(row, ids);
                let q = (pos / quarter).min(3);
                q_total[q] += 1;
                if argmax == usize::try_from(ids[0]).expect("top1 id") {
                    top1_hits += 1;
                    q_hits[q] += 1;
                } else if first_miss.is_none() {
                    first_miss = Some(pos);
                }
                for k in 0..top_k {
                    assert!(
                        ours[k].is_finite() && lps[k].is_finite(),
                        "{case}: non-finite logprob at (pos {pos}, k {k})"
                    );
                    let abs = (ours[k] - lps[k]).abs();
                    q_max[q] = q_max[q].max(abs);
                    if abs > max_abs {
                        max_abs = abs;
                        worst = (pos, k);
                    }
                }
            }
            eprintln!(
                "{case}: max |dlogprob| {max_abs} at (pos {}, k {}), top-1 agreement {top1_hits}/{seq_len}",
                worst.0, worst.1
            );
            eprintln!(
                "{case}: first top-1 miss {first_miss:?}; per-quarter top-1 {q_hits:?}/{q_total:?}, per-quarter max |dlogprob| {q_max:?}"
            );
            // Our own determinism: a second run must be bitwise identical —
            // the reference gap is then not run-to-run jitter on our side.
            if case == "edge" {
                let logits2 = full_forward(
                    &ctx,
                    &weights,
                    &tokens,
                    (&scos, &ssin),
                    (&gcos, &gsin),
                    cos_max_pos,
                )
                .expect("full forward replay");
                let host2 = logits2.to_host(&ctx).expect("logits D2H");
                assert_eq!(host.len(), host2.len(), "{case}: replay length");
                assert!(
                    host.iter()
                        .zip(&host2)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "{case}: replay differs bitwise from the first run"
                );
                eprintln!("{case}: replay bitwise identical");
            }
            let quarters_ok = (0..4)
                .all(|q| q_total[q] == 0 || q_hits[q] as f64 >= quarter_floor * q_total[q] as f64);
            if max_abs > tol || (top1_hits as f64) < top1_floor * seq_len as f64 || !quarters_ok {
                over_tolerance.push(format!(
                    "{case} (max {max_abs}, top-1 {top1_hits}/{seq_len}, per-quarter \
                     {q_hits:?}/{q_total:?})"
                ));
            }
        }
        assert!(
            over_tolerance.is_empty(),
            "cases over tolerance: {over_tolerance:?}"
        );
    }
}
