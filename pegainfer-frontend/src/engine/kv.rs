use std::any::Any;
use std::fmt;

use super::request::GenerateRequest;

/// A request's KV-prefix resolution, produced by the KV store *before* the
/// request reaches a scheduler: how many leading prompt tokens are already
/// materialized in the target rank's GPU prefix cache, plus an opaque RAII
/// hold keeping those blocks resident until the scheduler's prefix match
/// consumes them.
///
/// The engine contract deliberately does not know KV internals (this crate is
/// CUDA-free): the hold is minted by `pegainfer-kv-store`, carried opaquely,
/// and only ever *dropped* — after the scheduler's `match_and_add_prefix`, or
/// with the request if it dies first. Dropping releases the anti-eviction pin.
///
/// Degraded resolutions (timeout, pool pressure) are not a distinct state:
/// they surface as a smaller `hit_tokens` — the number alone carries all
/// downstream semantics (disaggregated-decode admission asserts it against
/// the handoff's committed length; everyone else just prefills from
/// `hit_tokens`).
pub struct KvPrefix {
    hit_tokens: usize,
    hold: Option<Box<dyn Any + Send>>,
    /// The scheduler partition the resolution ran against. The hold pins
    /// blocks on THIS rank, so routing anywhere else silently loses the hit
    /// and wastes the pin — [`crate::engine::EngineHandle::submit_resolved`]
    /// routes by it.
    rank: Option<usize>,
}

impl KvPrefix {
    /// No resolution ran (or it degraded to nothing): prefill from scratch.
    /// The scheduler's own GPU prefix match still applies as usual.
    #[must_use]
    pub fn none() -> Self {
        Self {
            hit_tokens: 0,
            hold: None,
            rank: None,
        }
    }

    /// A resolved prefix: `hit_tokens` are materialized on partition `rank`,
    /// pinned by `hold` until this value is dropped.
    #[must_use]
    pub fn resolved(hit_tokens: usize, rank: usize, hold: Box<dyn Any + Send>) -> Self {
        Self {
            hit_tokens,
            hold: Some(hold),
            rank: Some(rank),
        }
    }

    pub fn hit_tokens(&self) -> usize {
        self.hit_tokens
    }

    /// The partition the resolution is bound to, if one ran.
    pub(crate) fn rank(&self) -> Option<usize> {
        self.rank
    }

    /// Whether a hold is still pinning resolved blocks.
    pub fn has_hold(&self) -> bool {
        self.hold.is_some()
    }

    /// The opaque hold, for the store that minted it to downcast. Every
    /// other crate treats the hold as drop-only.
    pub fn hold_any(&self) -> Option<&(dyn Any + Send)> {
        self.hold.as_deref()
    }
}

impl fmt::Debug for KvPrefix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KvPrefix")
            .field("hit_tokens", &self.hit_tokens)
            .field("rank", &self.rank)
            .field("hold", &self.hold.as_ref().map(|_| "<opaque>"))
            .finish()
    }
}

/// What a scheduler partition's submit channel carries: the request plus its
/// KV-prefix resolution. Unresolved paths submit [`KvPrefix::none`]; the
/// tuple (rather than a wrapper struct) states the fact plainly — the store's
/// output is a prefix resolution, not a new kind of request.
pub type SubmittedRequest = (GenerateRequest, KvPrefix);

/// KV pool capacity as the scheduler actually allocates it: whole blocks of
/// `block_size` tokens. A request of `L` tokens occupies `⌈L / block_size⌉`
/// blocks no matter how `L` divides, so a fit check must round per request —
/// summing raw token counts under-counts and can admit a batch that the
/// scheduler then has to defer. Lets a caller (e.g. the prefill/decode bench)
/// decide up front whether a batch fits without computing per-token KV by hand.
#[derive(Clone, Copy, Debug)]
pub struct KvCapacity {
    /// Blocks available for requests when the pool is empty.
    pub total_blocks: usize,
    /// Tokens per block.
    pub block_size: usize,
}

impl KvCapacity {
    /// Total tokens the pool can hold (`total_blocks × block_size`).
    #[must_use]
    pub(crate) fn total_tokens(self) -> usize {
        self.total_blocks.saturating_mul(self.block_size)
    }

    /// Blocks a single request of `tokens` tokens occupies — whole-block
    /// allocation rounds up.
    #[must_use]
    pub fn blocks_for(self, tokens: usize) -> usize {
        tokens.div_ceil(self.block_size.max(1))
    }
}
