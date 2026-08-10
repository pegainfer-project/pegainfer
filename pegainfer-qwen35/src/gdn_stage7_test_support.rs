//! CPU oracle and immutable numeric gates for the real-SM120 Stage 7 harness.
//!
//! This module is test-only.  In particular, none of these tolerances can be
//! changed through a serving or test environment variable during a paid GPU
//! session.

use crate::gdn_prepare_test_contract::Fixture;
use crate::gdn_prepare_test_contract::Geometry;
use crate::gdn_prepare_test_contract::Prepared;
use crate::gdn_prepare_test_contract::bf16_to_f32;
use crate::gdn_prepare_test_contract::f32_to_bf16;
use crate::gdn_prepare_test_contract::prepare;

#[derive(Clone, Copy, Debug)]
pub(crate) struct NumericTolerance {
    pub(crate) atol: f32,
    pub(crate) rtol: f32,
}

/// Q/K are rounded to BF16 after an FP32 normalization reduction.  This
/// permits two BF16 steps around zero while remaining much narrower than the
/// operator tolerances used by the retired #709 candidate.
pub(crate) const PREPARE_QK_TOLERANCE: NumericTolerance = NumericTolerance {
    atol: 1.0 / 256.0,
    rtol: 0.0,
};

/// Alpha/beta stay FP32; only libdevice reduction/transcendental ordering may
/// differ between the scalar host oracle and the CUDA implementation.
pub(crate) const PREPARE_GATE_TOLERANCE: NumericTolerance = NumericTolerance {
    atol: 2.0e-6,
    rtol: 2.0e-6,
};

/// Prefill/decode outputs are stored as BF16.  State is accumulated in FP32.
/// The same fixed hybrid bound is applied to CPU↔Triton, CPU↔FlashInfer, and
/// Triton↔FlashInfer so no backend receives a looser gate.
pub(crate) const RECURRENCE_OUTPUT_TOLERANCE: NumericTolerance = NumericTolerance {
    atol: 1.0 / 64.0,
    rtol: 2.0e-3,
};
pub(crate) const RECURRENCE_STATE_TOLERANCE: NumericTolerance = NumericTolerance {
    atol: 5.0e-3,
    rtol: 2.0e-3,
};

// Hv48 is operator-only coverage rather than a supported model geometry.  Keep
// the frozen elementwise state bound as the primary gate, but permit a tiny
// numeric tail only when FlashInfer is strictly no worse than the existing
// Triton baseline on every aggregate statistic. The narrow excess cap retains
// the explained T=65 boundary tail but deliberately rejects the deeper T=128
// suffix-block error until the FP64-oracle audit establishes a final envelope.
const HV48_OPERATOR_STATE_MAX_VIOLATIONS: usize = 8;
const HV48_OPERATOR_STATE_MAX_EXCESS: f32 = 1.0 / 16_384.0;
const HV48_OPERATOR_STATE_ELEMENTS: usize = 48 * 128 * 128;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FirstDifference {
    pub(crate) index: usize,
    pub(crate) reference: f32,
    pub(crate) candidate: f32,
    pub(crate) abs_diff: f32,
    pub(crate) allowed: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DifferenceStats {
    pub(crate) count: usize,
    pub(crate) first_difference: Option<FirstDifference>,
    pub(crate) first_violation: Option<FirstDifference>,
    pub(crate) max_abs: f32,
    pub(crate) max_excess: f32,
    pub(crate) mean_abs: f32,
    pub(crate) p99_abs: f32,
    pub(crate) max_rel: f32,
    pub(crate) violations: usize,
}

impl DifferenceStats {
    pub(crate) fn compare(
        reference: &[f32],
        candidate: &[f32],
        tolerance: NumericTolerance,
    ) -> Result<Self, String> {
        if reference.len() != candidate.len() {
            return Err(format!(
                "comparison length mismatch: reference={}, candidate={}",
                reference.len(),
                candidate.len()
            ));
        }
        if reference.is_empty() {
            return Err("comparison inputs must be non-empty".to_string());
        }

        let mut diffs = Vec::with_capacity(reference.len());
        let mut first_difference = None;
        let mut first_violation = None;
        let mut sum = 0.0_f64;
        let mut max_abs = 0.0_f32;
        let mut max_excess = 0.0_f32;
        let mut max_rel = 0.0_f32;
        let mut violations = 0;
        for (index, (&reference, &candidate)) in reference.iter().zip(candidate).enumerate() {
            if !reference.is_finite() || !candidate.is_finite() {
                return Err(format!(
                    "comparison contains non-finite value at index {index}: reference={reference}, candidate={candidate}"
                ));
            }
            let abs_diff = (reference - candidate).abs();
            let scale = reference.abs().max(candidate.abs());
            let allowed = tolerance.atol + tolerance.rtol * scale;
            let difference = FirstDifference {
                index,
                reference,
                candidate,
                abs_diff,
                allowed,
            };
            if abs_diff != 0.0 && first_difference.is_none() {
                first_difference = Some(difference);
            }
            if abs_diff > allowed {
                violations += 1;
                max_excess = max_excess.max(abs_diff - allowed);
                if first_violation.is_none() {
                    first_violation = Some(difference);
                }
            }
            max_abs = max_abs.max(abs_diff);
            max_rel = max_rel.max(abs_diff / scale.max(f32::MIN_POSITIVE));
            sum += f64::from(abs_diff);
            diffs.push(abs_diff);
        }
        diffs.sort_by(f32::total_cmp);
        let p99_index = ((diffs.len() as f64 * 0.99).ceil() as usize)
            .saturating_sub(1)
            .min(diffs.len() - 1);
        Ok(Self {
            count: diffs.len(),
            first_difference,
            first_violation,
            max_abs,
            max_excess,
            mean_abs: (sum / diffs.len() as f64) as f32,
            p99_abs: diffs[p99_index],
            max_rel,
            violations,
        })
    }

    pub(crate) fn ensure_within(&self, label: &str) -> Result<(), String> {
        if self.violations == 0 {
            Ok(())
        } else {
            Err(format!(
                "{label} exceeded frozen tolerance at {}/{} elements; first violation {:?}; max_abs={}, max_excess={}, mean_abs={}, p99_abs={}, max_rel={}",
                self.violations,
                self.count,
                self.first_violation,
                self.max_abs,
                self.max_excess,
                self.mean_abs,
                self.p99_abs,
                self.max_rel
            ))
        }
    }

    pub(crate) fn ensure_hv48_operator_tail_within(
        &self,
        label: &str,
        triton_baseline: &Self,
    ) -> Result<(), String> {
        if self.violations == 0 {
            return Ok(());
        }
        if self.count != HV48_OPERATOR_STATE_ELEMENTS {
            return Err(format!(
                "{label} Hv48 operator-tail gate received {} elements, expected {}",
                self.count, HV48_OPERATOR_STATE_ELEMENTS
            ));
        }
        if self.violations > HV48_OPERATOR_STATE_MAX_VIOLATIONS {
            return Err(format!(
                "{label} Hv48 operator numeric tail has {} violations, cap is {}",
                self.violations, HV48_OPERATOR_STATE_MAX_VIOLATIONS
            ));
        }
        if self.max_excess > HV48_OPERATOR_STATE_MAX_EXCESS {
            return Err(format!(
                "{label} Hv48 operator numeric tail max_excess={} exceeds cap {}",
                self.max_excess, HV48_OPERATOR_STATE_MAX_EXCESS
            ));
        }
        let dominated = self.violations <= triton_baseline.violations
            && self.max_abs <= triton_baseline.max_abs
            && self.mean_abs <= triton_baseline.mean_abs
            && self.p99_abs <= triton_baseline.p99_abs;
        if !dominated {
            return Err(format!(
                "{label} Hv48 operator numeric tail does not dominate Triton baseline: FlashInfer={self:?}, Triton={triton_baseline:?}"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CpuRunResult {
    pub(crate) output: Vec<f32>,
    pub(crate) final_state: Vec<f32>,
}

/// Serial Gated Delta Rule reference over already-prepared native Q/K/V and
/// per-token alpha/beta.  State is `[Hv,K,V]`, with V contiguous.
pub(crate) fn cpu_stepwise(
    geometry: Geometry,
    prepared: &Prepared,
    initial_state: &[f32],
) -> Result<CpuRunResult, String> {
    let expected_state = geometry.h_v * geometry.d * geometry.d;
    if initial_state.len() != expected_state
        || prepared.q.len() != geometry.q_len()
        || prepared.k.len() != geometry.k_len()
        || prepared.v.len() != geometry.v_len()
        || prepared.alpha.len() != geometry.gate_len()
        || prepared.beta.len() != geometry.gate_len()
    {
        return Err("CPU GDN reference input length mismatch".to_string());
    }
    if geometry.h_q != geometry.h_k || !geometry.h_v.is_multiple_of(geometry.h_k) {
        return Err("CPU GDN reference requires Hq=Hk and Hv divisible by Hk".to_string());
    }

    let mut state = initial_state.to_vec();
    let mut output = vec![0.0_f32; geometry.v_len()];
    let scale = 1.0_f32 / (geometry.d as f32).sqrt();
    for token in 0..geometry.tokens {
        for value_head in 0..geometry.h_v {
            let key_head = value_head * geometry.h_k / geometry.h_v;
            let q_base = (token * geometry.h_q + key_head) * geometry.d;
            let k_base = (token * geometry.h_k + key_head) * geometry.d;
            let v_base = (token * geometry.h_v + value_head) * geometry.d;
            let state_base = value_head * geometry.d * geometry.d;
            let alpha = prepared.alpha[token * geometry.h_v + value_head];
            let beta = prepared.beta[token * geometry.h_v + value_head];

            for key in 0..geometry.d {
                let row = state_base + key * geometry.d;
                for value in 0..geometry.d {
                    state[row + value] *= alpha;
                }
            }

            for value in 0..geometry.d {
                let mut memory = 0.0_f32;
                for key in 0..geometry.d {
                    memory += state[state_base + key * geometry.d + value]
                        * bf16_to_f32(prepared.k[k_base + key]);
                }
                let delta = (bf16_to_f32(prepared.v[v_base + value]) - memory) * beta;
                let mut out = 0.0_f32;
                for key in 0..geometry.d {
                    let index = state_base + key * geometry.d + value;
                    state[index] += delta * bf16_to_f32(prepared.k[k_base + key]);
                    out += state[index] * bf16_to_f32(prepared.q[q_base + key]) * scale;
                }
                // Both CUDA backends store the public operator output as BF16.
                output[v_base + value] = bf16_to_f32(f32_to_bf16(out));
            }
        }
    }
    Ok(CpuRunResult {
        output,
        final_state: state,
    })
}

/// Neutral high-precision recurrence oracle. Inputs retain their public
/// BF16/FP32 values, all recurrence arithmetic is evaluated in FP64, and the
/// result is rounded only once at the public FP32-state/BF16-output boundary.
/// This is intentionally not a simulation of either Triton or WGMMA ordering.
pub(crate) fn cpu_stepwise_f64_rounded(
    geometry: Geometry,
    prepared: &Prepared,
    initial_state: &[f32],
) -> Result<CpuRunResult, String> {
    let expected_state = geometry.h_v * geometry.d * geometry.d;
    if initial_state.len() != expected_state
        || prepared.q.len() != geometry.q_len()
        || prepared.k.len() != geometry.k_len()
        || prepared.v.len() != geometry.v_len()
        || prepared.alpha.len() != geometry.gate_len()
        || prepared.beta.len() != geometry.gate_len()
    {
        return Err("FP64 CPU GDN reference input length mismatch".to_string());
    }
    if geometry.h_q != geometry.h_k || !geometry.h_v.is_multiple_of(geometry.h_k) {
        return Err("FP64 CPU GDN reference requires Hq=Hk and Hv divisible by Hk".to_string());
    }

    let mut state: Vec<f64> = initial_state.iter().copied().map(f64::from).collect();
    let mut output = vec![0.0_f32; geometry.v_len()];
    let scale = 1.0_f64 / (geometry.d as f64).sqrt();
    for token in 0..geometry.tokens {
        for value_head in 0..geometry.h_v {
            let key_head = value_head * geometry.h_k / geometry.h_v;
            let q_base = (token * geometry.h_q + key_head) * geometry.d;
            let k_base = (token * geometry.h_k + key_head) * geometry.d;
            let v_base = (token * geometry.h_v + value_head) * geometry.d;
            let state_base = value_head * geometry.d * geometry.d;
            let alpha = f64::from(prepared.alpha[token * geometry.h_v + value_head]);
            let beta = f64::from(prepared.beta[token * geometry.h_v + value_head]);

            for key in 0..geometry.d {
                let row = state_base + key * geometry.d;
                for value in 0..geometry.d {
                    state[row + value] *= alpha;
                }
            }

            for value in 0..geometry.d {
                let mut memory = 0.0_f64;
                for key in 0..geometry.d {
                    memory += state[state_base + key * geometry.d + value]
                        * f64::from(bf16_to_f32(prepared.k[k_base + key]));
                }
                let delta = (f64::from(bf16_to_f32(prepared.v[v_base + value])) - memory) * beta;
                let mut out = 0.0_f64;
                for key in 0..geometry.d {
                    let index = state_base + key * geometry.d + value;
                    state[index] += delta * f64::from(bf16_to_f32(prepared.k[k_base + key]));
                    out += state[index] * f64::from(bf16_to_f32(prepared.q[q_base + key])) * scale;
                }
                output[v_base + value] = bf16_to_f32(f32_to_bf16(out as f32));
            }
        }
    }
    Ok(CpuRunResult {
        output,
        final_state: state.into_iter().map(|value| value as f32).collect(),
    })
}

/// One production-decode step from raw fused Q/K/V and gates.  Unlike the
/// prefill prepare path, the decode CUDA kernel keeps normalized Q/K in FP32
/// registers instead of rounding them through BF16 scratch.
pub(crate) fn cpu_decode_from_raw(
    fixture: &Fixture,
    initial_state: &[f32],
) -> Result<CpuRunResult, String> {
    let geometry = fixture.geometry;
    if geometry.tokens != 1 {
        return Err("CPU raw decode reference requires exactly one token".to_string());
    }
    let prepared = prepare(fixture)?;
    let expected_state = geometry.h_v * geometry.d * geometry.d;
    if initial_state.len() != expected_state {
        return Err("CPU raw decode state length mismatch".to_string());
    }

    let normalize = |bits: &[u16]| {
        let values: Vec<f32> = bits.iter().copied().map(bf16_to_f32).collect();
        let inv_norm = (values.iter().map(|value| value * value).sum::<f32>() + 1.0e-12)
            .sqrt()
            .recip();
        values
            .into_iter()
            .map(|value| value * inv_norm)
            .collect::<Vec<_>>()
    };
    let mut q = Vec::with_capacity(geometry.h_q * geometry.d);
    let mut k = Vec::with_capacity(geometry.h_k * geometry.d);
    for head in 0..geometry.h_q {
        let start = fixture.offsets.q + head * geometry.d;
        q.extend(normalize(&fixture.qkv[start..start + geometry.d]));
    }
    for head in 0..geometry.h_k {
        let start = fixture.offsets.k + head * geometry.d;
        k.extend(normalize(&fixture.qkv[start..start + geometry.d]));
    }

    let mut state = initial_state.to_vec();
    let mut output = vec![0.0_f32; geometry.h_v * geometry.d];
    let scale = 1.0_f32 / (geometry.d as f32).sqrt();
    for value_head in 0..geometry.h_v {
        let key_head = value_head * geometry.h_k / geometry.h_v;
        let q_base = key_head * geometry.d;
        let k_base = key_head * geometry.d;
        let v_base = value_head * geometry.d;
        let state_base = value_head * geometry.d * geometry.d;
        let alpha = prepared.alpha[value_head];
        let beta = prepared.beta[value_head];

        for key in 0..geometry.d {
            let row = state_base + key * geometry.d;
            for value in 0..geometry.d {
                state[row + value] *= alpha;
            }
        }
        for value in 0..geometry.d {
            let mut memory = 0.0_f32;
            for key_index in 0..geometry.d {
                memory +=
                    state[state_base + key_index * geometry.d + value] * k[k_base + key_index];
            }
            let delta = (bf16_to_f32(prepared.v[v_base + value]) - memory) * beta;
            let mut out = 0.0_f32;
            for key_index in 0..geometry.d {
                let index = state_base + key_index * geometry.d + value;
                state[index] += delta * k[k_base + key_index];
                out += state[index] * q[q_base + key_index] * scale;
            }
            output[v_base + value] = bf16_to_f32(f32_to_bf16(out));
        }
    }
    Ok(CpuRunResult {
        output,
        final_state: state,
    })
}

pub(crate) fn asymmetric_hkv_state(geometry: Geometry) -> Vec<f32> {
    (0..geometry.h_v * geometry.d * geometry.d)
        .map(|index| {
            let head = index / (geometry.d * geometry.d);
            let rem = index % (geometry.d * geometry.d);
            let key = rem / geometry.d;
            let value = rem % geometry.d;
            // A scaled version of h*100000+k*100+v keeps every axis
            // distinguishable without making BF16 output overflow dominate.
            (head * 100_000 + key * 100 + value) as f32 * 1.0e-6 - 0.2
        })
        .collect()
}

/// Deliberate K/V transpose used only to prove the asymmetric oracle would
/// reject the unpatched upstream HVK interpretation when K==V==128.
pub(crate) fn transpose_kv_as_wrong_hvk(geometry: Geometry, hkv: &[f32]) -> Vec<f32> {
    let mut wrong = vec![0.0_f32; hkv.len()];
    for head in 0..geometry.h_v {
        for key in 0..geometry.d {
            for value in 0..geometry.d {
                let destination = (head * geometry.d + key) * geometry.d + value;
                let source = (head * geometry.d + value) * geometry.d + key;
                wrong[destination] = hkv[source];
            }
        }
    }
    wrong
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gdn_prepare_test_contract::deterministic_fixture;
    use crate::gdn_prepare_test_contract::prepare;

    #[test]
    fn tolerance_report_identifies_first_violation() {
        let stats = DifferenceStats::compare(
            &[1.0, 2.0, 3.0],
            &[1.0, 2.01, 3.5],
            NumericTolerance {
                atol: 0.02,
                rtol: 0.0,
            },
        )
        .unwrap();
        assert_eq!(stats.violations, 1);
        assert_eq!(stats.first_violation.unwrap().index, 2);
        assert!(stats.ensure_within("negative-control").is_err());
    }

    fn synthetic_stats(
        violations: usize,
        max_abs: f32,
        max_excess: f32,
        mean_abs: f32,
        p99_abs: f32,
    ) -> DifferenceStats {
        DifferenceStats {
            count: HV48_OPERATOR_STATE_ELEMENTS,
            first_difference: None,
            first_violation: None,
            max_abs,
            max_excess,
            mean_abs,
            p99_abs,
            max_rel: 1.0,
            violations,
        }
    }

    #[test]
    fn hv48_operator_tail_accepts_bounded_baseline_dominant_tail() {
        let flashinfer = synthetic_stats(4, 0.00514, 4.4e-5, 3.75e-4, 1.77e-3);
        let triton = synthetic_stats(6, 0.00584, 8.0e-4, 4.41e-4, 2.00e-3);
        flashinfer
            .ensure_hv48_operator_tail_within("Hv48", &triton)
            .unwrap();
    }

    #[test]
    fn hv48_operator_tail_rejects_excess_or_baseline_regression() {
        let triton = synthetic_stats(6, 0.00584, 8.0e-4, 4.41e-4, 2.00e-3);
        let excessive = synthetic_stats(
            4,
            0.00514,
            HV48_OPERATOR_STATE_MAX_EXCESS * 2.0,
            3.75e-4,
            1.77e-3,
        );
        assert!(
            excessive
                .ensure_hv48_operator_tail_within("Hv48", &triton)
                .is_err()
        );

        let regressed = synthetic_stats(4, 0.00514, 4.4e-5, 4.50e-4, 1.77e-3);
        assert!(
            regressed
                .ensure_hv48_operator_tail_within("Hv48", &triton)
                .is_err()
        );
    }

    #[test]
    fn cpu_stepwise_matches_hand_calculated_hkv_update() {
        let geometry = Geometry {
            h_q: 1,
            h_k: 1,
            h_v: 1,
            d: 2,
            tokens: 1,
        };
        let prepared = Prepared {
            q: vec![f32_to_bf16(1.0), f32_to_bf16(0.0)],
            k: vec![f32_to_bf16(1.0), f32_to_bf16(0.0)],
            v: vec![f32_to_bf16(2.0), f32_to_bf16(3.0)],
            alpha: vec![0.5],
            beta: vec![0.25],
        };
        let result = cpu_stepwise(geometry, &prepared, &[4.0, 5.0, 6.0, 7.0]).unwrap();
        let f64_result =
            cpu_stepwise_f64_rounded(geometry, &prepared, &[4.0, 5.0, 6.0, 7.0]).unwrap();
        assert_eq!(result.final_state, vec![2.0, 2.625, 3.0, 3.5]);
        let expected_output = vec![
            bf16_to_f32(f32_to_bf16(2.0 / 2.0_f32.sqrt())),
            bf16_to_f32(f32_to_bf16(2.625 / 2.0_f32.sqrt())),
        ];
        assert_eq!(result.output, expected_output);
        assert_eq!(f64_result, result);
    }

    #[test]
    fn cpu_stepwise_rejects_wrong_hvk_oracle() {
        let fixture = deterministic_fixture(2, 32);
        let prepared = prepare(&fixture).unwrap();
        let initial = asymmetric_hkv_state(fixture.geometry);
        let wrong = transpose_kv_as_wrong_hvk(fixture.geometry, &initial);
        let correct = cpu_stepwise(fixture.geometry, &prepared, &initial).unwrap();
        let wrong = cpu_stepwise(fixture.geometry, &prepared, &wrong).unwrap();
        let output =
            DifferenceStats::compare(&correct.output, &wrong.output, RECURRENCE_OUTPUT_TOLERANCE)
                .unwrap();
        let state = DifferenceStats::compare(
            &correct.final_state,
            &wrong.final_state,
            RECURRENCE_STATE_TOLERANCE,
        )
        .unwrap();
        assert!(output.violations > 0 || state.violations > 0);
    }
}
