//! CPU reference and host-only gates for the native GDN prepare stage.
//!
//! Kept dependency-free so it can be compiled with `rustc --test` even when
//! the workspace CUDA toolchain is unavailable. Inputs and Q/K/V outputs are
//! represented as raw BF16 bits to freeze rounding and split semantics.

pub(crate) const BOUNDARY_TOKENS: [usize; 7] = [1, 2, 63, 64, 65, 127, 128];
pub(crate) const D: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Geometry {
    pub(crate) h_q: usize,
    pub(crate) h_k: usize,
    pub(crate) h_v: usize,
    pub(crate) d: usize,
    pub(crate) tokens: usize,
}

impl Geometry {
    fn validate(self) -> Result<(), String> {
        if self.h_q != 16 || self.h_k != 16 || !matches!(self.h_v, 32 | 48) || self.d != D {
            return Err(format!(
                "native GDN prepare supports Hq/Hk/Hv/D=16/16/{{32,48}}/128, got {}/{}/{}/{}",
                self.h_q, self.h_k, self.h_v, self.d
            ));
        }
        if self.tokens == 0 {
            return Err("native GDN prepare requires T>=1".into());
        }
        Ok(())
    }

    pub(crate) fn q_len(self) -> usize {
        self.tokens * self.h_q * self.d
    }

    pub(crate) fn k_len(self) -> usize {
        self.tokens * self.h_k * self.d
    }

    pub(crate) fn v_len(self) -> usize {
        self.tokens * self.h_v * self.d
    }

    pub(crate) fn gate_len(self) -> usize {
        self.tokens * self.h_v
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ProjectionOffsets {
    pub(crate) q: usize,
    pub(crate) k: usize,
    pub(crate) v: usize,
    pub(crate) total: usize,
}

impl ProjectionOffsets {
    fn canonical(g: Geometry) -> Self {
        let q = 0;
        let k = g.h_q * g.d;
        let v = k + g.h_k * g.d;
        let total = v + g.h_v * g.d;
        Self { q, k, v, total }
    }

    fn validate(self, g: Geometry) -> Result<(), String> {
        let expected = Self::canonical(g);
        if self != expected {
            return Err(format!(
                "fused QKV offsets mismatch: got {self:?}, expected {expected:?}"
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Fixture {
    pub(crate) geometry: Geometry,
    pub(crate) offsets: ProjectionOffsets,
    pub(crate) qkv: Vec<u16>,
    pub(crate) b: Vec<u16>,
    pub(crate) a: Vec<u16>,
    pub(crate) dt_bias: Vec<u16>,
    pub(crate) a_log: Vec<f32>,
}

#[derive(Clone, Debug)]
pub(crate) struct Prepared {
    pub(crate) q: Vec<u16>,
    pub(crate) k: Vec<u16>,
    pub(crate) v: Vec<u16>,
    pub(crate) alpha: Vec<f32>,
    pub(crate) beta: Vec<f32>,
}

pub(crate) fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

pub(crate) fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let round = 0x7fff + ((bits >> 16) & 1);
    ((bits.wrapping_add(round)) >> 16) as u16
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

fn sigmoid(value: f32) -> f32 {
    let magnitude_exp = if value < 0.0 {
        value.exp()
    } else {
        (-value).exp()
    };
    if value >= 0.0 {
        1.0 / (1.0 + magnitude_exp)
    } else {
        magnitude_exp / (1.0 + magnitude_exp)
    }
}

fn normalize_bf16(input: &[u16], name: &str) -> Result<Vec<u16>, String> {
    let mut sum_sq = 0.0_f32;
    let mut values = Vec::with_capacity(input.len());
    for &bits in input {
        let value = bf16_to_f32(bits);
        if !value.is_finite() {
            return Err(format!("non-finite {name} input"));
        }
        sum_sq += value * value;
        values.push(value);
    }
    let inv_norm = (sum_sq + 1.0e-12).sqrt().recip();
    if !inv_norm.is_finite() {
        return Err(format!("non-finite {name} normalization"));
    }
    Ok(values
        .into_iter()
        .map(|value| f32_to_bf16(value * inv_norm))
        .collect())
}

pub(crate) fn prepare(fixture: &Fixture) -> Result<Prepared, String> {
    let g = fixture.geometry;
    g.validate()?;
    fixture.offsets.validate(g)?;
    let expected_qkv = g.tokens * fixture.offsets.total;
    if fixture.qkv.len() != expected_qkv
        || fixture.a.len() != g.gate_len()
        || fixture.b.len() != g.gate_len()
        || fixture.dt_bias.len() != g.h_v
        || fixture.a_log.len() != g.h_v
    {
        return Err("native GDN prepare input length mismatch".into());
    }

    let mut output = Prepared {
        q: Vec::with_capacity(g.q_len()),
        k: Vec::with_capacity(g.k_len()),
        v: Vec::with_capacity(g.v_len()),
        alpha: Vec::with_capacity(g.gate_len()),
        beta: Vec::with_capacity(g.gate_len()),
    };
    for token in 0..g.tokens {
        let token_base = token * fixture.offsets.total;
        for head in 0..g.h_q {
            let start = token_base + fixture.offsets.q + head * g.d;
            output
                .q
                .extend(normalize_bf16(&fixture.qkv[start..start + g.d], "Q")?);
        }
        for head in 0..g.h_k {
            let start = token_base + fixture.offsets.k + head * g.d;
            output
                .k
                .extend(normalize_bf16(&fixture.qkv[start..start + g.d], "K")?);
        }
        let v_start = token_base + fixture.offsets.v;
        for &bits in &fixture.qkv[v_start..v_start + g.h_v * g.d] {
            if !bf16_to_f32(bits).is_finite() {
                return Err("non-finite V input".into());
            }
            output.v.push(bits);
        }
        for head in 0..g.h_v {
            let gate = token * g.h_v + head;
            let a = bf16_to_f32(fixture.a[gate]);
            let b = bf16_to_f32(fixture.b[gate]);
            let bias = bf16_to_f32(fixture.dt_bias[head]);
            let log_a = fixture.a_log[head];
            if !a.is_finite() || !b.is_finite() || !bias.is_finite() || !log_a.is_finite() {
                return Err(format!(
                    "non-finite gate input at token={token}, head={head}"
                ));
            }
            let alpha = (-log_a.exp() * softplus(a + bias)).exp();
            let beta = sigmoid(b);
            if !alpha.is_finite() || !beta.is_finite() {
                return Err(format!(
                    "non-finite gate output at token={token}, head={head}"
                ));
            }
            output.alpha.push(alpha);
            output.beta.push(beta);
        }
    }
    Ok(output)
}

pub(crate) fn deterministic_fixture(tokens: usize, h_v: usize) -> Fixture {
    let geometry = Geometry {
        h_q: 16,
        h_k: 16,
        h_v,
        d: D,
        tokens,
    };
    let offsets = ProjectionOffsets::canonical(geometry);
    let qkv = (0..tokens * offsets.total)
        .map(|index| {
            let signed = ((index * 37 + 11) % 251) as i32 - 125;
            f32_to_bf16(signed as f32 / 31.0)
        })
        .collect();
    let b = (0..geometry.gate_len())
        .map(|index| f32_to_bf16(((index * 13 % 41) as f32 - 20.0) / 7.0))
        .collect();
    let a = (0..geometry.gate_len())
        .map(|index| f32_to_bf16(((index * 17 % 47) as f32 - 23.0) / 9.0))
        .collect();
    let dt_bias = (0..h_v)
        .map(|head| f32_to_bf16((head as f32 - h_v as f32 / 2.0) / 64.0))
        .collect();
    let a_log = (0..h_v)
        .map(|head| -2.5 + head as f32 / h_v as f32)
        .collect();
    Fixture {
        geometry,
        offsets,
        qkv,
        b,
        a,
        dt_bias,
        a_log,
    }
}

fn norm(values: &[u16]) -> f32 {
    values
        .iter()
        .map(|&bits| {
            let value = bf16_to_f32(bits);
            value * value
        })
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_preserves_native_head_counts_and_raw_v_bits() {
        let fixture = deterministic_fixture(2, 32);
        let prepared = prepare(&fixture).unwrap();
        assert_eq!(prepared.q.len(), fixture.geometry.q_len());
        assert_eq!(prepared.k.len(), fixture.geometry.k_len());
        assert_eq!(prepared.v.len(), fixture.geometry.v_len());
        assert!(prepared.q.len() < prepared.v.len());
        assert!(prepared.k.len() < prepared.v.len());

        let mut expected_v = Vec::new();
        for token in 0..fixture.geometry.tokens {
            let start = token * fixture.offsets.total + fixture.offsets.v;
            expected_v.extend_from_slice(
                &fixture.qkv[start..start + fixture.geometry.h_v * fixture.geometry.d],
            );
        }
        assert_eq!(prepared.v, expected_v);
    }

    #[test]
    fn q_and_k_are_independently_normalized_in_fp32() {
        let fixture = deterministic_fixture(2, 32);
        let prepared = prepare(&fixture).unwrap();
        for head in [0, 7, 15] {
            let q_start = head * D;
            let k_start = head * D;
            assert!((norm(&prepared.q[q_start..q_start + D]) - 1.0).abs() < 0.01);
            assert!((norm(&prepared.k[k_start..k_start + D]) - 1.0).abs() < 0.01);
        }
        assert_ne!(prepared.q[..D], prepared.k[..D]);
    }

    #[test]
    fn alpha_and_beta_are_per_token_values_not_log_or_cumulative() {
        let fixture = deterministic_fixture(2, 32);
        let prepared = prepare(&fixture).unwrap();
        for (token, head) in [(0, 0), (0, 17), (1, 0), (1, 31)] {
            let index = token * fixture.geometry.h_v + head;
            let a = bf16_to_f32(fixture.a[index]);
            let b = bf16_to_f32(fixture.b[index]);
            let bias = bf16_to_f32(fixture.dt_bias[head]);
            let log_alpha = -fixture.a_log[head].exp() * softplus(a + bias);
            assert!((prepared.alpha[index] - log_alpha.exp()).abs() < 1.0e-7);
            assert!((prepared.beta[index] - sigmoid(b)).abs() < 1.0e-7);
            assert!(prepared.alpha[index] > 0.0 && prepared.alpha[index] <= 1.0);
            assert!((0.0..=1.0).contains(&prepared.beta[index]));
            assert_ne!(prepared.alpha[index], log_alpha);
        }
        assert_ne!(prepared.alpha[0], prepared.alpha[fixture.geometry.h_v]);
    }

    #[test]
    fn boundary_lengths_and_hv48_complete() {
        for tokens in BOUNDARY_TOKENS {
            for h_v in [32, 48] {
                let fixture = deterministic_fixture(tokens, h_v);
                let prepared = prepare(&fixture).unwrap();
                assert_eq!(prepared.alpha.len(), tokens * h_v);
                assert_eq!(prepared.beta.len(), tokens * h_v);
            }
        }
    }

    #[test]
    fn small_and_large_finite_norm_inputs_remain_finite() {
        for magnitude in [1.0e-5_f32, 1.0e3_f32] {
            let mut fixture = deterministic_fixture(1, 32);
            for value in &mut fixture.qkv[..D] {
                *value = f32_to_bf16(magnitude);
            }
            for value in &mut fixture.qkv[fixture.offsets.k..fixture.offsets.k + D] {
                *value = f32_to_bf16(-magnitude);
            }
            let prepared = prepare(&fixture).unwrap();
            assert!(
                prepared.q[..D]
                    .iter()
                    .all(|&bits| bf16_to_f32(bits).is_finite())
            );
            assert!(
                prepared.k[..D]
                    .iter()
                    .all(|&bits| bf16_to_f32(bits).is_finite())
            );
            assert!((norm(&prepared.q[..D]) - 1.0).abs() < 0.01);
            assert!((norm(&prepared.k[..D]) - 1.0).abs() < 0.01);
        }
    }

    #[test]
    fn rejects_wrong_offsets_geometry_lengths_and_non_finite_inputs() {
        let mut fixture = deterministic_fixture(1, 32);
        fixture.offsets.k += D;
        assert!(prepare(&fixture).unwrap_err().contains("offsets mismatch"));

        let mut fixture = deterministic_fixture(1, 32);
        fixture.geometry.h_q = 8;
        assert!(prepare(&fixture).unwrap_err().contains("supports Hq"));

        let mut fixture = deterministic_fixture(1, 32);
        fixture.b.pop();
        assert!(prepare(&fixture).unwrap_err().contains("length mismatch"));

        let mut fixture = deterministic_fixture(1, 32);
        fixture.qkv[0] = f32_to_bf16(f32::NAN);
        assert!(prepare(&fixture).unwrap_err().contains("non-finite Q"));

        let mut fixture = deterministic_fixture(1, 32);
        fixture.a_log[0] = f32::INFINITY;
        assert!(prepare(&fixture).unwrap_err().contains("non-finite gate"));
    }

    #[test]
    fn cuda_source_uses_native_qk_grid_and_direct_target_layouts() {
        let source = include_str!("../../pegainfer-kernels/csrc/qwen35/gdn_prepare.cu");
        assert!(source.contains("const dim3 grid(tokens, h_q + h_k + h_v)"));
        assert!(source.contains("token) * h_q + head) * head_dim + d"));
        assert!(source.contains("token) * h_k + head) * head_dim + d"));
        assert!(source.contains("token) * h_v + head) * head_dim + d"));
        assert!(!source.contains("v_head * h_k / h_v"));
        assert!(!source.contains("q_expanded"));
        assert!(!source.contains("k_expanded"));
    }
}
