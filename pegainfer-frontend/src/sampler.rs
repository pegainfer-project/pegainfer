#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    /// Minimum probability threshold relative to the most likely token:
    /// tokens with `p < min_p * p_max` are masked. `0.0` disables (the
    /// default); valid range is [0, 1).
    pub min_p: f32,
    /// Per-request sampling seed. `Some` makes the request's sampled tokens a
    /// pure function of (seed, request step, distribution) — independent of
    /// batch composition — so a fixed-seed request replays identically.
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
}
