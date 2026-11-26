#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    /// Minimum probability threshold relative to the most likely token:
    /// tokens with `p < min_p * p_max` are masked. `0.0` disables (the
    /// default); valid range is [0, 1).
    pub min_p: f32,
    /// Per-request sampling seed, handed to the engine as the request gave
    /// it. What a seed replays is the engine's promise: a fixed seed gives
    /// the same tokens for the same request where the engine's draw depends
    /// on nothing else, or at the same batch layout where it does.
    pub seed: Option<u64>,
    pub ignore_eos: bool,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.0,
            seed: None,
            ignore_eos: false,
        }
    }
}

impl SamplingParams {
    /// Greedy means argmax: temperature below the sampling epsilon (the
    /// temperature -> 0 limit is argmax regardless of top_p, and 1/temperature
    /// overflows long before that; vLLM draws the same line at 1e-5) or
    /// top_k == 1 (a single token survives the mask). Everything else requires
    /// a real sampling pass.
    pub fn is_greedy(&self) -> bool {
        self.temperature < 1e-5 || self.top_k == 1
    }

    /// Matches the wire greedy conversion and the sampler kernel's host validation,
    /// so admission can reject bad parameters before batching.
    pub fn validate(&self) -> Result<(), &'static str> {
        // Wire forces greedy when temperature <= 0, rewriting top_p/min_p to valid values.
        if self.temperature <= 0.0 {
            return Ok(());
        }
        if !self.temperature.is_finite() {
            return Err("temperature must be finite and positive");
        }
        if !self.top_p.is_finite() || self.top_p <= 0.0 || self.top_p > 1.0 {
            return Err("top_p must be in (0, 1]");
        }
        if !self.min_p.is_finite() || self.min_p < 0.0 || self.min_p >= 1.0 {
            return Err("min_p must be in [0, 1)");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::SamplingParams;

    #[test]
    fn rejects_out_of_range_sampling_parameters() {
        let bad_top_p = SamplingParams {
            temperature: 0.8,
            top_p: 0.0,
            ..SamplingParams::default()
        };
        assert_eq!(bad_top_p.validate(), Err("top_p must be in (0, 1]"));

        let bad_min_p = SamplingParams {
            temperature: 0.8,
            top_p: 1.0,
            min_p: 1.0,
            ..SamplingParams::default()
        };
        assert_eq!(bad_min_p.validate(), Err("min_p must be in [0, 1)"));
    }

    #[test]
    fn zero_temperature_skips_sampling_validation() {
        let greedy = SamplingParams::default();
        assert!(greedy.validate().is_ok());

        // top_k=1 only changes the kernel sampling path; it does not exempt top_p from host validation.
        let top_k_one = SamplingParams {
            temperature: 0.8,
            top_k: 1,
            top_p: 0.0,
            ..SamplingParams::default()
        };
        assert_eq!(top_k_one.validate(), Err("top_p must be in (0, 1]"));
    }
}
