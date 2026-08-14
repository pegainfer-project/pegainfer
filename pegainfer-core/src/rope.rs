//! RoPE cos/sin table generation.

use anyhow::Result;
use half::bf16;

use crate::tensor::DeviceContext;
use crate::tensor::DeviceVec;

/// Geometry of a RoPE cos/sin cache.
pub struct RopeTableSpec {
    /// Components that get rotated. Drives the cache extent and the half-split.
    pub rotary_dim: usize,
    /// Denominator of `theta^(2i / frequency_dim)`, and nothing else.
    pub frequency_dim: usize,
    pub max_seq_len: usize,
    pub theta: f32,
}

impl RopeTableSpec {
    /// Everything the table generation assumes. Each rejected case would
    /// otherwise yield a table that looks well-formed and is wrong.
    fn validate(&self) -> Result<usize> {
        let RopeTableSpec {
            rotary_dim: r,
            frequency_dim: f,
            max_seq_len: n,
            theta,
        } = *self;
        anyhow::ensure!(r > 0, "RoPE rotary_dim must be positive");
        anyhow::ensure!(r % 2 == 0, "RoPE rotary_dim must be even, got {r}");
        anyhow::ensure!(
            r <= f,
            "RoPE rotary_dim ({r}) must not exceed frequency_dim ({f})"
        );
        anyhow::ensure!(n > 0, "RoPE max_seq_len must be positive");
        anyhow::ensure!(
            theta.is_finite() && theta > 0.0,
            "RoPE theta must be finite and positive, got {theta}"
        );
        n.checked_mul(r)
            .ok_or_else(|| anyhow::anyhow!("RoPE table size {n} x {r} overflows usize"))
    }
}

/// Upload a spec's tables as contiguous GPU buffers.
pub fn precompute_rope(
    ctx: &DeviceContext,
    spec: &RopeTableSpec,
) -> Result<(DeviceVec, DeviceVec)> {
    let (cos_host, sin_host) = spec.tables()?;
    Ok((
        DeviceVec::from_host(ctx, &cos_host)?,
        DeviceVec::from_host(ctx, &sin_host)?,
    ))
}

impl RopeTableSpec {
    /// Host-side cos/sin tables, laid out `[max_seq_len * rotary_dim]` with
    /// position `pos` at offset `pos * rotary_dim`.
    fn tables(&self) -> Result<(Vec<bf16>, Vec<bf16>)> {
        let total = self.validate()?;
        let RopeTableSpec {
            rotary_dim,
            frequency_dim,
            max_seq_len,
            theta,
        } = *self;
        let half_dim = rotary_dim / 2;

        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(i as f32 * 2.0 / frequency_dim as f32))
            .collect();

        let mut cos_host = vec![bf16::ZERO; total];
        let mut sin_host = vec![bf16::ZERO; total];

        for pos in 0..max_seq_len {
            let base = pos * rotary_dim;
            for i in 0..half_dim {
                let freq = pos as f32 * inv_freq[i];
                let cos_val = bf16::from_f32(freq.cos());
                let sin_val = bf16::from_f32(freq.sin());
                // Half-split layout: [cos(0)..cos(half-1), cos(0)..cos(half-1)]
                cos_host[base + i] = cos_val;
                cos_host[base + i + half_dim] = cos_val;
                sin_host[base + i] = sin_val;
                sin_host[base + i + half_dim] = sin_val;
            }
        }

        Ok((cos_host, sin_host))
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::RopeTableSpec;

    const THETA: f32 = 1e6;

    fn spec(rotary_dim: usize, frequency_dim: usize, max_seq_len: usize) -> RopeTableSpec {
        RopeTableSpec {
            rotary_dim,
            frequency_dim,
            max_seq_len,
            theta: THETA,
        }
    }

    fn tables(
        rotary_dim: usize,
        frequency_dim: usize,
        max_seq_len: usize,
    ) -> (Vec<bf16>, Vec<bf16>) {
        spec(rotary_dim, frequency_dim, max_seq_len)
            .tables()
            .expect("test geometry must be valid")
    }

    #[test]
    fn rope_tables_backward_compatible_with_single_param() {
        // Reference comes from the formula, not from a second call.
        let (d, n) = (512usize, 2048usize);
        let (cos, sin) = tables(d, d, n);

        let half = d / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / THETA.powf(i as f32 * 2.0 / d as f32))
            .collect();
        for pos in 0..n {
            let base = pos * d;
            for i in 0..half {
                let freq = pos as f32 * inv_freq[i];
                let c = bf16::from_f32(freq.cos());
                let s = bf16::from_f32(freq.sin());
                assert_eq!(cos[base + i].to_bits(), c.to_bits(), "cos pos={pos} i={i}");
                assert_eq!(
                    cos[base + i + half].to_bits(),
                    c.to_bits(),
                    "cos dup pos={pos} i={i}"
                );
                assert_eq!(sin[base + i].to_bits(), s.to_bits(), "sin pos={pos} i={i}");
                assert_eq!(
                    sin[base + i + half].to_bits(),
                    s.to_bits(),
                    "sin dup pos={pos} i={i}"
                );
            }
        }
    }

    #[test]
    fn rope_tables_proportional_matches_full_rope_leading_half() {
        // The two rows duplicate at different offsets — 64 and 256 — so the
        // identity covers the first rotary_dim / 2 entries, not a whole-row
        // prefix. Exact arithmetic on both sides, so 1024 rows suffice.
        let n = 1024usize;
        let (prop_cos, prop_sin) = tables(128, 512, n);
        let (full_cos, full_sin) = tables(512, 512, n);

        for pos in [0usize, 1023] {
            let (pb, fb) = (pos * 128, pos * 512);
            assert_eq!(
                prop_cos[pb..pb + 64],
                full_cos[fb..fb + 64],
                "cos leading half diverged at pos={pos}"
            );
            assert_eq!(
                prop_sin[pb..pb + 64],
                full_sin[fb..fb + 64],
                "sin leading half diverged at pos={pos}"
            );
            if pos > 0 {
                assert_ne!(
                    prop_cos[pb + 64..pb + 128],
                    full_cos[fb + 64..fb + 128],
                    "expected the second halves to diverge at pos={pos}"
                );
            }
        }
    }

    #[test]
    fn rope_tables_proportional_is_not_ordinary_partial() {
        // Only i = 0 is denominator-independent (theta^0 both ways). From i = 1
        // the two formulas separate immediately: at pos = 1 the cos entries
        // differ by ~0.109, some 28 bf16 ulps.
        let (prop_cos, _) = tables(128, 512, 2);
        let (partial_cos, _) = tables(128, 128, 2);

        assert_eq!(
            prop_cos[128].to_bits(),
            partial_cos[128].to_bits(),
            "i=0 must agree"
        );
        assert_ne!(
            prop_cos[129].to_bits(),
            partial_cos[129].to_bits(),
            "i=1 at pos=1 must separate the two denominators"
        );
    }

    #[test]
    fn rope_spec_rejects_unusable_geometry() {
        for (r, f, what) in [
            (0usize, 512usize, "zero rotary_dim"),
            (127, 512, "odd rotary_dim"),
            (512, 128, "rotary_dim above frequency_dim"),
        ] {
            assert!(spec(r, f, 8).tables().is_err(), "{what} must be rejected");
        }
        assert!(spec(128, 512, 0).tables().is_err(), "zero max_seq_len");
        assert!(
            spec(128, 512, usize::MAX).tables().is_err(),
            "table size overflow"
        );
        for (theta, what) in [
            (0.0f32, "zero theta"),
            (-1e6, "negative theta"),
            (f32::NAN, "NaN theta"),
            (f32::INFINITY, "infinite theta"),
        ] {
            let s = RopeTableSpec {
                theta,
                ..spec(128, 512, 8)
            };
            assert!(s.tables().is_err(), "{what} must be rejected");
        }
    }
}
