use std::path::Path;

use anyhow::Result;
use half::bf16;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::Qwen35GdnGeometry;

use super::GdnPrefillBackend;
use super::checked_prefill_end_pos;
use crate::recurrent_state::RecurrentState;
use crate::weights::Qwen35Model;

const CHUNK_STATE_ATOL: f32 = 5.0e-3;
const CHUNK_STATE_RTOL: f32 = 2.0e-3;
const CHUNK_OUTPUT_ATOL: f32 = 1.0 / 64.0;
const CHUNK_OUTPUT_RTOL: f32 = 2.0e-3;
const LOGIT_MEAN_TOL: f32 = 0.06;
const LOGIT_P99_TOL: f32 = 0.20;
const LOGIT_ARGMAX_REGRET_TOL: f32 = 0.20;

fn required_model_path() -> String {
    let default = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");
    let path = std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| default.to_string());
    assert!(
        Path::new(&path).join("config.json").is_file(),
        "required chunk-continuation gate cannot read {path}/config.json; set PEGAINFER_TEST_MODEL_PATH"
    );
    path
}

#[derive(Debug)]
struct F32DifferenceStats {
    first_violation: Option<(usize, f32, f32, f32)>,
    violations: usize,
    max_abs: f32,
    mean_abs: f32,
    p99_abs: f32,
    max_rel: f32,
}

fn difference_stats_f32(
    expected: &[f32],
    actual: &[f32],
    atol: f32,
    rtol: f32,
) -> F32DifferenceStats {
    assert_eq!(
        expected.len(),
        actual.len(),
        "f32 comparison length mismatch"
    );
    let mut absolute = Vec::with_capacity(expected.len());
    let mut max_relative = 0.0_f32;
    let mut first_violation = None;
    let mut violation_count = 0usize;
    for (index, (&left, &right)) in expected.iter().zip(actual).enumerate() {
        let diff = (left - right).abs();
        absolute.push(diff);
        max_relative = max_relative.max(diff / left.abs().max(right.abs()).max(1.0e-12));
        let violation = !left.is_finite()
            || !right.is_finite()
            || diff > atol + rtol * left.abs().max(right.abs());
        if violation {
            violation_count += 1;
            if first_violation.is_none() {
                first_violation = Some((index, left, right, diff));
            }
        }
    }
    absolute.sort_by(f32::total_cmp);
    let max = absolute.last().copied().unwrap_or(0.0);
    let mean = if absolute.is_empty() {
        0.0
    } else {
        absolute.iter().sum::<f32>() / absolute.len() as f32
    };
    let p99_index = absolute.len().saturating_sub(1) * 99 / 100;
    let p99 = absolute.get(p99_index).copied().unwrap_or(0.0);
    F32DifferenceStats {
        first_violation,
        violations: violation_count,
        max_abs: max,
        mean_abs: mean,
        p99_abs: p99,
        max_rel: max_relative,
    }
}

fn report_close_f32(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    atol: f32,
    rtol: f32,
) -> F32DifferenceStats {
    let stats = difference_stats_f32(expected, actual, atol, rtol);
    eprintln!(
        "{label}: elements={} violations={} max_abs={:.8} mean_abs={:.8} p99_abs={:.8} max_rel={:.8} atol={atol} rtol={rtol}",
        expected.len(),
        stats.violations,
        stats.max_abs,
        stats.mean_abs,
        stats.p99_abs,
        stats.max_rel,
    );
    stats
}

fn assert_close_f32(label: &str, expected: &[f32], actual: &[f32], atol: f32, rtol: f32) {
    let stats = report_close_f32(label, expected, actual, atol, rtol);
    assert!(
        stats.first_violation.is_none(),
        "{label} first violation {:?}; violations={}/{} max_abs={} mean_abs={} p99_abs={} max_rel={}",
        stats.first_violation,
        stats.violations,
        expected.len(),
        stats.max_abs,
        stats.mean_abs,
        stats.p99_abs,
        stats.max_rel,
    );
}

fn log_softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let log_sum = values
        .iter()
        .map(|value| (*value - max).exp())
        .sum::<f32>()
        .ln();
    values.iter().map(|value| *value - max - log_sum).collect()
}

fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .expect("logits must be non-empty")
}

fn assert_logit_parity(label: &str, expected: &[f32], actual: &[f32]) -> usize {
    assert_eq!(
        expected.len(),
        actual.len(),
        "{label} logit length mismatch"
    );
    let expected_lp = log_softmax(expected);
    let actual_lp = log_softmax(actual);
    assert!(
        expected_lp.iter().all(|value| value.is_finite())
            && actual_lp.iter().all(|value| value.is_finite()),
        "{label} contains non-finite log-probabilities"
    );
    let expected_token = argmax(&expected_lp);
    let actual_token = argmax(&actual_lp);
    let regret = expected_lp[expected_token] - expected_lp[actual_token];
    let mut deltas = expected_lp
        .iter()
        .zip(&actual_lp)
        .map(|(left, right)| (*left - *right).abs())
        .collect::<Vec<_>>();
    deltas.sort_by(f32::total_cmp);
    let max = deltas.last().copied().unwrap_or(0.0);
    let mean = deltas.iter().sum::<f32>() / deltas.len() as f32;
    let p99 = deltas[deltas.len().saturating_sub(1) * 99 / 100];
    eprintln!(
        "{label}: vocab={} expected_tokens=[{expected_token}] actual_tokens=[{actual_token}] max_logprob_delta={max:.6} mean={mean:.6} p99={p99:.6} regret={regret:.6}",
        deltas.len()
    );
    assert!(
        regret <= LOGIT_ARGMAX_REGRET_TOL,
        "{label} actual argmax {actual_token} has baseline regret {regret} > {LOGIT_ARGMAX_REGRET_TOL}"
    );
    assert_eq!(
        actual_token, expected_token,
        "{label} greedy token parity failed"
    );
    assert!(
        mean <= LOGIT_MEAN_TOL,
        "{label} mean {mean} > {LOGIT_MEAN_TOL}"
    );
    assert!(p99 <= LOGIT_P99_TOL, "{label} p99 {p99} > {LOGIT_P99_TOL}");
    expected_token
}

struct PreparedGdnFixture {
    geometry: Qwen35GdnGeometry,
    tokens: usize,
    q: Vec<bf16>,
    k: Vec<bf16>,
    v: Vec<bf16>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
    initial_state: Vec<f32>,
}

struct CpuGdnResult {
    output: Vec<f32>,
    final_state: Vec<f32>,
}

fn normalized_bf16_rows(tokens: usize, heads: usize, dim: usize, salt: usize) -> Vec<bf16> {
    let mut result = Vec::with_capacity(tokens * heads * dim);
    for token in 0..tokens {
        for head in 0..heads {
            let row = (0..dim)
                .map(|index| {
                    let value = (token * 37 + head * 19 + index * salt + 11) % 251;
                    value as f32 - 125.0
                })
                .collect::<Vec<_>>();
            let inv_norm = row
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                .sqrt()
                .recip();
            result.extend(
                row.into_iter()
                    .map(|value| bf16::from_f32(value * inv_norm)),
            );
        }
    }
    result
}

fn prepared_gdn_fixture(geometry: Qwen35GdnGeometry, tokens: usize) -> PreparedGdnFixture {
    assert_eq!(geometry.h_q, geometry.h_k);
    assert_eq!(geometry.h_v % geometry.h_k, 0);
    let v = (0..tokens * geometry.h_v * geometry.head_dim)
        .map(|index| {
            let signed = ((index * 29 + 7) % 97) as i32 - 48;
            bf16::from_f32(signed as f32 / 128.0)
        })
        .collect();
    let alpha = (0..tokens * geometry.h_v)
        .map(|index| 0.980_468_75 + (index % 17) as f32 / 1024.0)
        .collect();
    let beta = (0..tokens * geometry.h_v)
        .map(|index| 0.25 + (index % 17) as f32 / 32.0)
        .collect();
    let initial_state = (0..geometry.h_v)
        .flat_map(|head| {
            (0..geometry.head_dim).flat_map(move |key| {
                (0..geometry.head_dim)
                    .map(move |value| (head * 100_000 + key * 100 + value) as f32 * 1.0e-7 - 0.1)
            })
        })
        .collect();
    PreparedGdnFixture {
        geometry,
        tokens,
        q: normalized_bf16_rows(tokens, geometry.h_q, geometry.head_dim, 23),
        k: normalized_bf16_rows(tokens, geometry.h_k, geometry.head_dim, 31),
        v,
        alpha,
        beta,
        initial_state,
    }
}

fn cpu_gdn_recurrence(fixture: &PreparedGdnFixture) -> CpuGdnResult {
    let g = fixture.geometry;
    let mut state = fixture.initial_state.clone();
    let mut output = vec![0.0_f32; fixture.tokens * g.h_v * g.head_dim];
    let scale = (g.head_dim as f32).sqrt().recip();
    for token in 0..fixture.tokens {
        for value_head in 0..g.h_v {
            let key_head = value_head * g.h_k / g.h_v;
            let q_base = (token * g.h_q + key_head) * g.head_dim;
            let k_base = (token * g.h_k + key_head) * g.head_dim;
            let v_base = (token * g.h_v + value_head) * g.head_dim;
            let state_base = value_head * g.head_dim * g.head_dim;
            let alpha = fixture.alpha[token * g.h_v + value_head];
            let beta = fixture.beta[token * g.h_v + value_head];

            for value in &mut state[state_base..state_base + g.head_dim * g.head_dim] {
                *value *= alpha;
            }
            for value in 0..g.head_dim {
                let mut memory = 0.0_f32;
                for key in 0..g.head_dim {
                    memory += state[state_base + key * g.head_dim + value]
                        * fixture.k[k_base + key].to_f32();
                }
                let delta = (fixture.v[v_base + value].to_f32() - memory) * beta;
                let mut out = 0.0_f32;
                for key in 0..g.head_dim {
                    let state_index = state_base + key * g.head_dim + value;
                    state[state_index] += delta * fixture.k[k_base + key].to_f32();
                    out += state[state_index] * fixture.q[q_base + key].to_f32() * scale;
                }
                output[v_base + value] = bf16::from_f32(out).to_f32();
            }
        }
    }
    CpuGdnResult {
        output,
        final_state: state,
    }
}

fn launch_prepared_gdn_segment(
    model: &Qwen35Model,
    fixture: &PreparedGdnFixture,
    start: usize,
    end: usize,
    state: &mut cudarc::driver::CudaSlice<f32>,
) -> Result<Vec<f32>> {
    assert!(start < end && end <= fixture.tokens);
    let ctx = model.device_ctx();
    let g = fixture.geometry;
    let tokens = end - start;
    let q_width = g.h_q * g.head_dim;
    let k_width = g.h_k * g.head_dim;
    let v_width = g.h_v * g.head_dim;
    let q = HiddenStates::from_host(
        ctx,
        &fixture.q[start * q_width..end * q_width],
        q_width,
        tokens,
    )?;
    let k = HiddenStates::from_host(
        ctx,
        &fixture.k[start * k_width..end * k_width],
        k_width,
        tokens,
    )?;
    let v = HiddenStates::from_host(
        ctx,
        &fixture.v[start * v_width..end * v_width],
        v_width,
        tokens,
    )?;
    let alpha = ctx
        .stream
        .clone_htod(&fixture.alpha[start * g.h_v..end * g.h_v])?;
    let beta = ctx
        .stream
        .clone_htod(&fixture.beta[start * g.h_v..end * g.h_v])?;
    let mut output = HiddenStates::zeros(ctx, v_width, tokens)?;
    let backend = model.flashinfer_gdn()?;
    let mut workspace = backend.allocate_workspace(ctx, tokens)?;
    backend.launch_in_place(
        ctx,
        &q,
        &k,
        &v,
        &alpha,
        &beta,
        state,
        &mut output,
        &mut workspace,
    )?;
    output.to_host(ctx)
}

fn assert_operator_continuation(model: &Qwen35Model) -> Result<()> {
    const TOKENS: usize = 128;
    const SPLIT: usize = 64;
    let geometry = crate::flashinfer_gdn::model_geometry(model.config());
    let fixture = prepared_gdn_fixture(geometry, TOKENS);
    let cpu = cpu_gdn_recurrence(&fixture);
    let ctx = model.device_ctx();

    let mut unchunked_state = ctx.stream.clone_htod(&fixture.initial_state)?;
    let unchunked_output =
        launch_prepared_gdn_segment(model, &fixture, 0, TOKENS, &mut unchunked_state)?;
    let unchunked_state = ctx.stream.clone_dtoh(&unchunked_state)?;

    let mut chunked_state = ctx.stream.clone_htod(&fixture.initial_state)?;
    let mut chunked_output =
        launch_prepared_gdn_segment(model, &fixture, 0, SPLIT, &mut chunked_state)?;
    chunked_output.extend(launch_prepared_gdn_segment(
        model,
        &fixture,
        SPLIT,
        TOKENS,
        &mut chunked_state,
    )?);
    let chunked_state = ctx.stream.clone_dtoh(&chunked_state)?;
    ctx.sync()?;

    assert_close_f32(
        "operator CPU oracle vs unchunked output",
        &cpu.output,
        &unchunked_output,
        CHUNK_OUTPUT_ATOL,
        CHUNK_OUTPUT_RTOL,
    );
    assert_close_f32(
        "operator CPU oracle vs chunked output",
        &cpu.output,
        &chunked_output,
        CHUNK_OUTPUT_ATOL,
        CHUNK_OUTPUT_RTOL,
    );
    assert_close_f32(
        "operator unchunked vs chunked output",
        &unchunked_output,
        &chunked_output,
        CHUNK_OUTPUT_ATOL,
        CHUNK_OUTPUT_RTOL,
    );
    assert_close_f32(
        "operator CPU oracle vs unchunked final state",
        &cpu.final_state,
        &unchunked_state,
        CHUNK_STATE_ATOL,
        CHUNK_STATE_RTOL,
    );
    assert_close_f32(
        "operator CPU oracle vs chunked final state",
        &cpu.final_state,
        &chunked_state,
        CHUNK_STATE_ATOL,
        CHUNK_STATE_RTOL,
    );
    assert_close_f32(
        "operator unchunked vs chunked final state",
        &unchunked_state,
        &chunked_state,
        CHUNK_STATE_ATOL,
        CHUNK_STATE_RTOL,
    );
    Ok(())
}

fn last_token_logits(
    model: &Qwen35Model,
    hidden: &pegainfer_core::tensor::HiddenStates,
) -> Result<Vec<f32>> {
    let last = crate::ops::extract_vec(model.device_ctx(), hidden, hidden.seq_len - 1)?;
    let logits = model.batch_last_hidden_logits(&[last])?;
    logits.to_host(model.device_ctx())
}

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
            let first =
                model.prefill_chunk_forward(&tokens[..split], &mut kv, &mut recurrent, backend)?;
            drop(first);
            model.prefill_chunk_forward(&tokens[split..], &mut kv, &mut recurrent, backend)?
        }
        None => model.prefill_chunk_forward(tokens, &mut kv, &mut recurrent, backend)?,
    };
    let logits = last_token_logits(model, &hidden)?;
    drop(hidden);
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

#[test]
#[ignore = "requires an SM120 GPU, Qwen3.5-4B weights, and a build-linked validated FlashInfer bundle"]
fn flashinfer_gdn_chunk_continuation_and_model_outputs_match() -> Result<()> {
    let model_path = required_model_path();
    let model = Qwen35Model::from_safetensors(&model_path, 0, 1)?;
    model.require_flashinfer_gdn_for_test()?;
    assert_eq!(model.resolved_gdn_backend(), GdnPrefillBackend::FlashInfer);
    let evidence_before = model.flashinfer_gdn_runtime_evidence()?;
    assert_eq!(evidence_before.selected_backend, "flashinfer");
    assert_ne!(evidence_before.artifact_sha256, "unavailable");
    assert_eq!(evidence_before.artifact_sha256.len(), 64);
    assert_eq!(evidence_before.successful_launches, 0);

    // Keep arbitrary non-zero HKV state at the operator boundary, where
    // both executions consume byte-identical prepared inputs and an
    // independent serial recurrence can determine correctness.
    assert_operator_continuation(&model)?;

    // The model-level comparison starts from the real new-request zero
    // state. Splitting a whole model changes GEMM/attention association,
    // so the production contract here is full-vocabulary output parity,
    // not applying the operator's state tolerance to different inputs.
    let tokens = (0..128)
        .map(|index| 100 + (index * 17 % 1000) as u32)
        .collect::<Vec<_>>();

    let (mut chunked_kv, chunked_state, chunked_prefill_logits) =
        run_prefill_case(&model, &tokens, GdnPrefillBackend::FlashInfer, Some(64))?;
    let (mut unchunked_kv, unchunked_state, unchunked_prefill_logits) =
        run_prefill_case(&model, &tokens, GdnPrefillBackend::FlashInfer, None)?;

    assert_eq!(chunked_state.seq_len, 128);
    assert_eq!(unchunked_state.seq_len, 128);
    let decode_token = assert_logit_parity(
        "final prefill",
        &unchunked_prefill_logits,
        &chunked_prefill_logits,
    ) as u32;

    let unchunked_decode =
        first_decode_logits(&model, decode_token, &mut unchunked_kv, &unchunked_state)?;
    let chunked_decode =
        first_decode_logits(&model, decode_token, &mut chunked_kv, &chunked_state)?;
    assert_logit_parity("first decode", &unchunked_decode, &chunked_decode);

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
        (3 * linear_layers + 3) as u64,
        "chunk continuation gate did not execute one operator full pass, two operator continuation passes, two model chunks, and one unchunked model pass"
    );
    Ok(())
}
