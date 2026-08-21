#[cfg(feature = "gdn-validation")]
use std::path::Path;

#[cfg(feature = "gdn-validation")]
use anyhow::Result;

#[cfg(feature = "gdn-validation")]
use super::GdnPrefillBackend;
use super::checked_prefill_end_pos;
#[cfg(feature = "gdn-validation")]
use crate::recurrent_state::RecurrentState;
#[cfg(feature = "gdn-validation")]
use crate::weights::Qwen35Model;

// Stage 18 measured the real-model FP32 recurrent-state partition floor at
// mean=1.3146e-4 and p99=1.0670e-3. Stage 19 then calibrated the BF16 conv-state
// distribution on the real zero-state SM120 continuation gate. These bounds
// retain margin over those observed floors without turning token parity into
// the only continuation criterion.
#[cfg(feature = "gdn-validation")]
const RECURRENT_STATE_MEAN_TOL: f32 = 2.5e-4;
#[cfg(feature = "gdn-validation")]
const RECURRENT_STATE_P99_TOL: f32 = 2.0e-3;
#[cfg(feature = "gdn-validation")]
const CONV_STATE_MEAN_TOL: f32 = 1.5625e-2;
#[cfg(feature = "gdn-validation")]
const CONV_STATE_P99_TOL: f32 = 6.25e-2;
#[cfg(feature = "gdn-validation")]
const LOGIT_MEAN_TOL: f32 = 0.06;
#[cfg(feature = "gdn-validation")]
const LOGIT_P99_TOL: f32 = 0.20;
#[cfg(feature = "gdn-validation")]
const LOGIT_ARGMAX_REGRET_TOL: f32 = 0.20;

#[cfg(feature = "gdn-validation")]
fn required_model_path() -> String {
    let default = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");
    let path = std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| default.to_string());
    assert!(
        Path::new(&path).join("config.json").is_file(),
        "required chunk-continuation gate cannot read {path}/config.json; set PEGAINFER_TEST_MODEL_PATH"
    );
    path
}

#[cfg(feature = "gdn-validation")]
fn assert_distribution_close(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    mean_tolerance: f32,
    p99_tolerance: f32,
) {
    assert_eq!(expected.len(), actual.len(), "{label} length mismatch");
    assert!(!expected.is_empty(), "{label} must not be empty");

    let mut deltas = Vec::with_capacity(expected.len());
    for (index, (&left, &right)) in expected.iter().zip(actual).enumerate() {
        assert!(
            left.is_finite() && right.is_finite(),
            "{label} contains a non-finite value at {index}: expected={left} actual={right}"
        );
        deltas.push((left - right).abs());
    }
    deltas.sort_by(f32::total_cmp);

    let mean =
        (deltas.iter().map(|&value| f64::from(value)).sum::<f64>() / deltas.len() as f64) as f32;
    let p50 = deltas[deltas.len().saturating_sub(1) * 50 / 100];
    let p99 = deltas[deltas.len().saturating_sub(1) * 99 / 100];
    let max = *deltas.last().expect("non-empty deltas");
    eprintln!(
        "{label}: elements={} mean_abs={mean:.8} p50_abs={p50:.8} p99_abs={p99:.8} max_abs={max:.8} mean_tol={mean_tolerance} p99_tol={p99_tolerance}",
        deltas.len()
    );

    assert!(
        mean <= mean_tolerance,
        "{label} mean_abs {mean} exceeds {mean_tolerance}"
    );
    assert!(
        p99 <= p99_tolerance,
        "{label} p99_abs {p99} exceeds {p99_tolerance}"
    );
}

#[cfg(feature = "gdn-validation")]
fn assert_recurrent_continuation(
    model: &Qwen35Model,
    unchunked: &RecurrentState,
    chunked: &RecurrentState,
) -> Result<()> {
    assert_eq!(unchunked.seq_len, 128);
    assert_eq!(chunked.seq_len, 128);
    assert_eq!(
        unchunked.layers.len(),
        chunked.layers.len(),
        "linear recurrent layer count mismatch"
    );

    let ctx = model.device_ctx();
    for (layer, (expected, actual)) in unchunked.layers.iter().zip(&chunked.layers).enumerate() {
        let expected_state = ctx.stream.clone_dtoh(&expected.state)?;
        let actual_state = ctx.stream.clone_dtoh(&actual.state)?;
        let expected_conv = expected.conv_state.to_host(ctx)?;
        let actual_conv = actual.conv_state.to_host(ctx)?;
        ctx.sync()?;

        assert_distribution_close(
            &format!("real-model layer {layer} recurrent state"),
            &expected_state,
            &actual_state,
            RECURRENT_STATE_MEAN_TOL,
            RECURRENT_STATE_P99_TOL,
        );
        assert_distribution_close(
            &format!("real-model layer {layer} conv state"),
            &expected_conv,
            &actual_conv,
            CONV_STATE_MEAN_TOL,
            CONV_STATE_P99_TOL,
        );
    }
    Ok(())
}

#[cfg(feature = "gdn-validation")]
fn assert_logits_close(label: &str, expected: &[f32], actual: &[f32]) -> u32 {
    assert_distribution_close(label, expected, actual, LOGIT_MEAN_TOL, LOGIT_P99_TOL);

    let expected_top = pegainfer_sample::token_logprob_from_row(expected, 0, 1)
        .and_then(|summary| summary.top_logprobs.into_iter().next())
        .expect("baseline logits must contain a top token");
    let actual_top = pegainfer_sample::token_logprob_from_row(actual, 0, 1)
        .and_then(|summary| summary.top_logprobs.into_iter().next())
        .expect("candidate logits must contain a top token");
    let actual_token_in_baseline =
        pegainfer_sample::token_logprob_from_row(expected, actual_top.0, 0)
            .expect("candidate token must be in the baseline vocabulary");
    let regret = expected_top.1 - actual_token_in_baseline.logprob;

    eprintln!(
        "{label}: expected_token={} actual_token={} expected_logprob={:.6} actual_logprob={:.6} regret={regret:.6}",
        expected_top.0, actual_top.0, expected_top.1, actual_top.1
    );
    assert!(
        regret <= LOGIT_ARGMAX_REGRET_TOL,
        "{label} candidate token {} has baseline regret {regret} > {LOGIT_ARGMAX_REGRET_TOL}",
        actual_top.0
    );
    assert_eq!(
        actual_top.0, expected_top.0,
        "{label} greedy token parity failed"
    );
    expected_top.0
}

#[cfg(feature = "gdn-validation")]
fn last_token_logits(
    model: &Qwen35Model,
    hidden: &pegainfer_core::tensor::HiddenStates,
) -> Result<Vec<f32>> {
    let last = crate::ops::extract_vec(model.device_ctx(), hidden, hidden.seq_len - 1)?;
    model
        .batch_last_hidden_logits(&[last])?
        .to_host(model.device_ctx())
}

#[cfg(feature = "gdn-validation")]
fn run_prefill_case(
    model: &Qwen35Model,
    tokens: &[u32],
    backend: GdnPrefillBackend,
    split_at: Option<usize>,
) -> Result<(pegainfer_core::kv_pool::KvState, RecurrentState, Vec<f32>)> {
    let mut kv = model.alloc_kv();
    let mut recurrent = RecurrentState::new(model.device_ctx(), model.config())?;
    let hidden = match split_at {
        Some(split) => {
            assert!(split > 0 && split < tokens.len());
            drop(model.prefill_chunk_forward(
                &tokens[..split],
                &mut kv,
                &mut recurrent,
                backend,
            )?);
            model.prefill_chunk_forward(&tokens[split..], &mut kv, &mut recurrent, backend)?
        }
        None => model.prefill_chunk_forward(tokens, &mut kv, &mut recurrent, backend)?,
    };
    let logits = last_token_logits(model, &hidden)?;
    Ok((kv, recurrent, logits))
}

#[cfg(feature = "gdn-validation")]
fn first_decode_logits(
    model: &Qwen35Model,
    token: u32,
    kv: &mut pegainfer_core::kv_pool::KvState,
    recurrent: &RecurrentState,
) -> Result<Vec<f32>> {
    let mut graph = model.create_batch_decode_graph_state_with_capacity(1)?;
    graph.copy_state_to_slot(model.device_ctx(), recurrent, 0)?;
    let mut kv_refs = vec![kv];
    model.batch_decode_graph(&[token], &mut kv_refs, &mut graph)?;
    graph.buffers.logits.to_host(model.device_ctx())
}

#[test]
fn checked_prefill_end_pos_accepts_config_limit() {
    assert_eq!(
        checked_prefill_end_pos(0, 262_144, 262_144).unwrap(),
        262_144
    );
    assert_eq!(
        checked_prefill_end_pos(262_143, 1, 262_144).unwrap(),
        262_144
    );
}

#[test]
fn checked_prefill_end_pos_rejects_past_config_limit() {
    let err = checked_prefill_end_pos(0, 262_145, 262_144)
        .unwrap_err()
        .to_string();
    assert!(err.contains("beyond max_position_embeddings=262144"));
    assert!(err.contains("requested end_pos=262145"));
}

#[test]
fn checked_prefill_end_pos_rejects_overflow() {
    let err = checked_prefill_end_pos(usize::MAX, 1, 262_144)
        .unwrap_err()
        .to_string();
    assert!(err.contains("prefill position overflow"));
}

#[cfg(feature = "gdn-validation")]
#[test]
#[ignore = "requires an SM120 GPU, Qwen3.5-4B weights, and a build-linked validated FlashInfer bundle"]
fn flashinfer_gdn_chunk_continuation_and_model_outputs_match() -> Result<()> {
    let model_path = required_model_path();
    let model = Qwen35Model::from_safetensors(&model_path, 0, 1)?;
    let backend = model.resolved_gdn_backend();
    assert_eq!(backend, GdnPrefillBackend::FlashInfer);

    let evidence_before = model.flashinfer_gdn_runtime_evidence()?;
    assert_eq!(evidence_before.selected_backend, "flashinfer");
    assert_ne!(evidence_before.artifact_sha256, "unavailable");
    assert_eq!(evidence_before.artifact_sha256.len(), 64);
    assert_eq!(evidence_before.successful_launches, 0);

    // These deterministic token ids are only model inputs. All hidden values,
    // Q/K/V/gates, recurrent state, and logits come from the real 4B weights.
    let tokens = (0..128)
        .map(|index| 100 + (index * 17 % 1000) as u32)
        .collect::<Vec<_>>();
    let (mut unchunked_kv, unchunked_state, unchunked_prefill_logits) =
        run_prefill_case(&model, &tokens, backend, None)?;
    let (mut chunked_kv, chunked_state, chunked_prefill_logits) =
        run_prefill_case(&model, &tokens, backend, Some(64))?;

    assert_recurrent_continuation(&model, &unchunked_state, &chunked_state)?;
    let decode_token = assert_logits_close(
        "real-model last-token logits",
        &unchunked_prefill_logits,
        &chunked_prefill_logits,
    );

    let unchunked_decode =
        first_decode_logits(&model, decode_token, &mut unchunked_kv, &unchunked_state)?;
    let chunked_decode =
        first_decode_logits(&model, decode_token, &mut chunked_kv, &chunked_state)?;
    assert_logits_close(
        "real-model first-decode logits",
        &unchunked_decode,
        &chunked_decode,
    );

    let evidence_after = model.flashinfer_gdn_runtime_evidence()?;
    assert_eq!(evidence_after.selected_backend, "flashinfer");
    assert_eq!(
        evidence_after.artifact_sha256,
        evidence_before.artifact_sha256
    );
    let linear_layers =
        model.config().num_hidden_layers - model.config().num_full_attention_layers();
    assert_eq!(
        evidence_after.successful_launches - evidence_before.successful_launches,
        (3 * linear_layers) as u64,
        "chunk continuation gate did not execute one unchunked pass and two resumed model chunks"
    );
    Ok(())
}
