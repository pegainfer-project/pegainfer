use anyhow::Result;
use half::bf16;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

use super::gated_delta_rule_prefill_native_prepare_into;
use crate::prefill_buffers::GdnPrepareScratch35;

fn bf16_vec(data: &[f32]) -> Vec<bf16> {
    data.iter().map(|&x| bf16::from_f32(x)).collect()
}

fn assert_f32_close_with_stats(
    label: &str,
    expected: &[f32],
    actual: &[f32],
    atol: f32,
    rtol: f32,
) {
    assert_eq!(expected.len(), actual.len(), "{label} length mismatch");
    let mut deltas = Vec::with_capacity(expected.len());
    let mut max_relative = 0.0_f32;
    let mut violation_count = 0usize;
    let mut first_violation = None;
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        let delta = (expected - actual).abs();
        let relative = delta / expected.abs().max(actual.abs()).max(1.0e-12);
        deltas.push(delta);
        max_relative = max_relative.max(relative);
        let violation = !expected.is_finite()
            || !actual.is_finite()
            || delta > atol + rtol * expected.abs().max(actual.abs());
        if violation {
            violation_count += 1;
            if first_violation.is_none() {
                first_violation = Some((index, expected, actual, delta));
            }
        }
    }
    deltas.sort_by(f32::total_cmp);
    let max = deltas.last().copied().unwrap_or(0.0);
    let mean = if deltas.is_empty() {
        0.0
    } else {
        deltas.iter().sum::<f32>() / deltas.len() as f32
    };
    let p99 = deltas
        .get(deltas.len().saturating_sub(1) * 99 / 100)
        .copied()
        .unwrap_or(0.0);
    eprintln!(
        "{label}: elements={} violations={violation_count} max_abs={max:.8} mean_abs={mean:.8} p99_abs={p99:.8} max_rel={max_relative:.8} atol={atol} rtol={rtol}",
        deltas.len()
    );
    assert!(
        first_violation.is_none(),
        "{label} first violation {:?}; violations={violation_count}/{} max_abs={max} mean_abs={mean} p99_abs={p99} max_rel={max_relative}",
        first_violation,
        expected.len(),
    );
}

fn assert_bf16_bits_equal(label: &str, expected: &[bf16], actual: &[bf16]) {
    assert_eq!(expected.len(), actual.len(), "{label} length mismatch");
    let first_mismatch = expected
        .iter()
        .zip(actual)
        .position(|(expected, actual)| expected.to_bits() != actual.to_bits());
    assert!(
        first_mismatch.is_none(),
        "{label} first bitwise mismatch at {:?}: expected={:?} actual={:?}",
        first_mismatch,
        first_mismatch.map(|index| expected[index].to_f32()),
        first_mismatch.map(|index| actual[index].to_f32()),
    );
    eprintln!("{label}: elements={} bitwise_mismatches=0", expected.len());
}

#[test]
#[ignore = "requires a CUDA GPU"]
fn test_gdn_native_prepare_matches_cpu_reference_on_finite_inputs() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let h_q = 16usize;
    let h_k = 16usize;
    let h_v = 32usize;
    let d = 128usize;
    let qkv_dim = (h_q + h_k + h_v) * d;
    let dt_host = bf16_vec(
        &(0..h_v)
            .map(|head| (head as f32 - h_v as f32 / 2.0) / 64.0)
            .collect::<Vec<_>>(),
    );
    let a_log_host = (0..h_v)
        .map(|head| match head {
            // Keep negative-softplus outputs away from 1 so branch errors remain visible.
            0 => 32.0,
            1 | 2 => 20.0,
            _ => -2.5 + head as f32 / h_v as f32,
        })
        .collect::<Vec<_>>();
    let dt_bias = DeviceVec::from_host(&ctx, &dt_host)?;
    let a_log = ctx.stream.clone_htod(&a_log_host)?;

    // Each CTA owns one token/head; cover a single token and the large production grid.
    for tokens in [1usize, 2048] {
        let qkv_host = bf16_vec(
            &(0..tokens * qkv_dim)
                .map(|index| {
                    let signed = ((index * 37 + 11) % 251) as i32 - 125;
                    let value = signed as f32 / 31.0;
                    // Exercise epsilon-dominated Q/K norms without erasing head asymmetry.
                    match (index / d) % (h_q + h_k + h_v) {
                        0 | 16 => 0.0,
                        1 | 17 => value * 1.0e-8,
                        _ => value,
                    }
                })
                .collect::<Vec<_>>(),
        );
        let b_host = bf16_vec(
            &(0..tokens * h_v)
                .map(|index| match index % h_v {
                    0 => -100.0,
                    1 => 0.0,
                    2 => 100.0,
                    _ => ((index * 13 % 41) as f32 - 20.0) / 7.0,
                })
                .collect::<Vec<_>>(),
        );
        let a_host = bf16_vec(
            &(0..tokens * h_v)
                .map(|index| match index % h_v {
                    // Bias is at most 0.25: these straddle both softplus thresholds.
                    0 => -32.0,
                    1 => -20.5,
                    2 => -19.5,
                    3 => 19.5,
                    4 => 20.5,
                    5 => 32.0,
                    _ => ((index * 17 % 47) as f32 - 23.0) / 9.0,
                })
                .collect::<Vec<_>>(),
        );
        let qkv = HiddenStates {
            data: ctx.stream.clone_htod(&qkv_host)?,
            hidden_dim: qkv_dim,
            seq_len: tokens,
        };
        let b = HiddenStates {
            data: ctx.stream.clone_htod(&b_host)?,
            hidden_dim: h_v,
            seq_len: tokens,
        };
        let a = HiddenStates {
            data: ctx.stream.clone_htod(&a_host)?,
            hidden_dim: h_v,
            seq_len: tokens,
        };
        let mut prepared = GdnPrepareScratch35::new(&ctx, tokens)?;
        gated_delta_rule_prefill_native_prepare_into(
            &ctx,
            &qkv,
            &b,
            &a,
            &dt_bias,
            &a_log,
            &mut prepared,
        )?;

        let q_actual = ctx.stream.clone_dtoh(&prepared.q.data)?;
        let k_actual = ctx.stream.clone_dtoh(&prepared.k.data)?;
        let v_actual = ctx.stream.clone_dtoh(&prepared.v.data)?;
        let alpha_actual = ctx.stream.clone_dtoh(&prepared.alpha)?;
        let beta_actual = ctx.stream.clone_dtoh(&prepared.beta)?;
        ctx.sync()?;

        let mut q_expected = Vec::with_capacity(tokens * h_q * d);
        let mut k_expected = Vec::with_capacity(tokens * h_k * d);
        let mut v_expected = Vec::with_capacity(tokens * h_v * d);
        for token in 0..tokens {
            let token_qkv = token * qkv_dim;
            for head in 0..h_q {
                let input = token_qkv + head * d;
                let output = (token * h_q + head) * d;
                let sum_sq = qkv_host[input..input + d]
                    .iter()
                    .map(|value| value.to_f32().powi(2))
                    .sum::<f32>();
                let inv_norm = (sum_sq + 1.0e-12).sqrt().recip();
                for lane in 0..d {
                    q_expected.push(qkv_host[input + lane].to_f32() * inv_norm);
                }
                debug_assert_eq!(q_expected.len(), output + d);
            }
            for head in 0..h_k {
                let input = token_qkv + h_q * d + head * d;
                let output = (token * h_k + head) * d;
                let sum_sq = qkv_host[input..input + d]
                    .iter()
                    .map(|value| value.to_f32().powi(2))
                    .sum::<f32>();
                let inv_norm = (sum_sq + 1.0e-12).sqrt().recip();
                for lane in 0..d {
                    k_expected.push(qkv_host[input + lane].to_f32() * inv_norm);
                }
                debug_assert_eq!(k_expected.len(), output + d);
            }
            let v_input = token_qkv + (h_q + h_k) * d;
            v_expected.extend_from_slice(&qkv_host[v_input..v_input + h_v * d]);
        }
        let q_actual_f32 = q_actual
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let k_actual_f32 = k_actual
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        assert_f32_close_with_stats(
            &format!("native prepare Q [T={tokens},H={h_q},D={d},bf16]"),
            &q_expected,
            &q_actual_f32,
            1.0 / 256.0,
            0.0,
        );
        assert_f32_close_with_stats(
            &format!("native prepare K [T={tokens},H={h_k},D={d},bf16]"),
            &k_expected,
            &k_actual_f32,
            1.0 / 256.0,
            0.0,
        );
        assert_bf16_bits_equal(
            &format!("native prepare V [T={tokens},H={h_v},D={d},bf16]"),
            &v_expected,
            &v_actual,
        );
        let mut alpha_expected = Vec::with_capacity(tokens * h_v);
        let mut beta_expected = Vec::with_capacity(tokens * h_v);
        for index in 0..tokens * h_v {
            let head = index % h_v;
            let a_value = a_host[index].to_f32();
            let b_value = b_host[index].to_f32();
            let x = f64::from(a_value) + f64::from(dt_host[head].to_f32());
            let softplus = x.max(0.0) + (-x.abs()).exp().ln_1p();
            let expected_alpha = (-f64::from(a_log_host[head]).exp() * softplus).exp() as f32;
            let expected_beta = (1.0 / (1.0 + (-f64::from(b_value)).exp())) as f32;
            alpha_expected.push(expected_alpha);
            beta_expected.push(expected_beta);
        }
        assert_f32_close_with_stats(
            &format!("native prepare alpha [T={tokens},H={h_v},f32]"),
            &alpha_expected,
            &alpha_actual,
            2.0e-6,
            2.0e-6,
        );
        assert_f32_close_with_stats(
            &format!("native prepare beta [T={tokens},H={h_v},f32]"),
            &beta_expected,
            &beta_actual,
            2.0e-6,
            2.0e-6,
        );
    }

    Ok(())
}
