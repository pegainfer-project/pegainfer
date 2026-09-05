use anyhow::Result;

use crate::recurrent_state::RecurrentState;
use crate::weights::Qwen35Model;

// Bounds calibrated for shape-dependent BF16 rounding in whole-model continuation.
const RECURRENT_STATE_MEAN_TOL: f32 = 2.5e-4;
const RECURRENT_STATE_P99_TOL: f32 = 2.0e-3;
const CONV_STATE_MEAN_TOL: f32 = 1.5625e-2;
const CONV_STATE_P99_TOL: f32 = 6.25e-2;
const LOGIT_MEAN_TOL: f32 = 0.06;
const LOGIT_P99_TOL: f32 = 0.20;
const LOGIT_ARGMAX_REGRET_TOL: f32 = 0.20;

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

fn last_token_logits(
    model: &Qwen35Model,
    hidden: &pegainfer_core::tensor::HiddenStates,
) -> Result<Vec<f32>> {
    let last = crate::ops::extract_vec(model.device_ctx(), hidden, hidden.seq_len - 1)?;
    model
        .batch_last_hidden_logits(&[last])?
        .to_host(model.device_ctx())
}

fn run_prefill_case(
    model: &Qwen35Model,
    tokens: &[u32],
    split_at: Option<usize>,
) -> Result<(pegainfer_core::kv_pool::KvState, RecurrentState, Vec<f32>)> {
    let mut kv = model.alloc_kv();
    let mut recurrent = RecurrentState::new(model.device_ctx(), model.config())?;
    let hidden = match split_at {
        Some(split) => {
            assert!(split > 0 && split < tokens.len());
            drop(model.prefill_chunk_forward(&tokens[..split], &mut kv, &mut recurrent)?);
            assert_eq!(recurrent.seq_len, split);
            for (layer, state) in recurrent.layers.iter().enumerate() {
                let matrix = model.device_ctx().stream.clone_dtoh(&state.state)?;
                let conv = state.conv_state.to_host(model.device_ctx())?;
                assert!(
                    matrix.iter().any(|&value| value != 0.0)
                        && conv.iter().any(|&value| value != 0.0),
                    "first chunk left layer {layer} continuation state empty"
                );
            }
            model.prefill_chunk_forward(&tokens[split..], &mut kv, &mut recurrent)?
        }
        None => model.prefill_chunk_forward(tokens, &mut kv, &mut recurrent)?,
    };
    let logits = last_token_logits(model, &hidden)?;
    Ok((kv, recurrent, logits))
}

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
#[ignore = "requires an SM120 GPU, Qwen3.5-4B weights, and a build-linked validated FlashInfer bundle"]
fn flashinfer_gdn_chunk_continuation_and_model_outputs_match() -> Result<()> {
    let model_path = crate::test_fixture::model_path_or_skip(
        "flashinfer_gdn_chunk_continuation_and_model_outputs_match",
    )
    .expect("chunk-continuation gate requires PEGAINFER_TEST_MODEL_PATH");
    let model = Qwen35Model::from_safetensors_with_options(
        &model_path,
        &crate::Qwen35LaunchOptions {
            max_batch: 1,
            gdn_backend: crate::Qwen35GdnBackend::FlashInferCandidate,
            ..Default::default()
        },
    )?;
    assert!(model.flashinfer_gdn.is_some());

    // These deterministic token ids are only model inputs. All hidden values,
    // Q/K/V/gates, recurrent state, and logits come from the real 4B weights.
    let tokens = (0..128)
        .map(|index| 100 + (index * 17 % 1000) as u32)
        .collect::<Vec<_>>();
    let (mut unchunked_kv, unchunked_state, unchunked_prefill_logits) =
        run_prefill_case(&model, &tokens, None)?;
    let (mut chunked_kv, chunked_state, chunked_prefill_logits) =
        run_prefill_case(&model, &tokens, Some(64))?;

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

    Ok(())
}
