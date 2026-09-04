//! Split-KV decode chunking config: the chunk-size formula and the
//! `split_kv_{chunk}x{max}` label/parse round-trip for the manifest variant sweep.

use serde::Deserialize;
use serde::Serialize;

/// One split-KV decode configuration: a per-chunk token floor and a per-request chunk cap.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SplitKvConfig {
    pub chunk_tokens: usize,
    pub max_chunks_per_request: usize,
}

impl SplitKvConfig {
    pub const fn new(chunk_tokens: usize, max_chunks_per_request: usize) -> Self {
        // A zero `max_chunks_per_request` divides by zero in `actual_chunk_size` (and a zero
        // `chunk_tokens` does in `active_chunks` at kv_len==0); reject loud at construction
        // (compile-time for the const sites) rather than panic mid-decode.
        assert!(
            chunk_tokens != 0 && max_chunks_per_request != 0,
            "SplitKvConfig requires non-zero chunk_tokens and max_chunks_per_request"
        );
        Self {
            chunk_tokens,
            max_chunks_per_request,
        }
    }

    /// Chunk size for a `kv_len`-token request: the floor, coarsened to keep the
    /// request within `max_chunks_per_request` chunks.
    pub fn actual_chunk_size(self, kv_len: usize) -> usize {
        self.chunk_tokens
            .max(kv_len.div_ceil(self.max_chunks_per_request))
    }

    pub fn active_chunks(self, kv_len: usize) -> usize {
        kv_len.div_ceil(self.actual_chunk_size(kv_len)).max(1)
    }

    /// The `split_kv_{chunk}x{max}` manifest variant name.
    pub fn label(self) -> String {
        format!(
            "split_kv_{}x{}",
            self.chunk_tokens, self.max_chunks_per_request
        )
    }

    /// Inverse of [`label`](Self::label): parse `split_kv_{chunk}x{max}` back into a config.
    /// `None` on a malformed label or a zero field.
    pub fn parse(label: &str) -> Option<Self> {
        let (chunk, max_chunks) = label.strip_prefix("split_kv_")?.split_once('x')?;
        let chunk_tokens: usize = chunk.parse().ok()?;
        let max_chunks_per_request: usize = max_chunks.parse().ok()?;
        if chunk_tokens == 0 || max_chunks_per_request == 0 {
            return None;
        }
        Some(Self::new(chunk_tokens, max_chunks_per_request))
    }
}

#[cfg(test)]
mod tests {
    use super::SplitKvConfig;

    #[test]
    fn label_parse_round_trips() {
        for cfg in [
            SplitKvConfig::new(64, 64),
            SplitKvConfig::new(64, 256),
            SplitKvConfig::new(256, 64),
            SplitKvConfig::new(512, 64),
        ] {
            assert_eq!(SplitKvConfig::parse(&cfg.label()), Some(cfg));
        }
    }

    #[test]
    fn parse_rejects_malformed() {
        assert_eq!(SplitKvConfig::parse("non_partition"), None);
        assert_eq!(SplitKvConfig::parse("split_kv_64"), None);
        assert_eq!(SplitKvConfig::parse("split_kv_axb"), None);
        assert_eq!(SplitKvConfig::parse("split_kv_64x0"), None);
        assert_eq!(SplitKvConfig::parse("split_kv_0x64"), None);
    }
}
