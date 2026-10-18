//! The caller-facing vocabulary of the store: cancellation, cache scoping,
//! per-request read policy, and save bookkeeping.

use pegainfer_frontend::engine::TokenSink;
use tokio::sync::oneshot;

/// Whether the request wants the resolve abandoned. Implemented by the
/// engine's `TokenSink` (the abort atomic the frontend flips); the store
/// observes it between operations and short-circuits remaining I/O.
pub trait CancelProbe: Send + Sync {
    fn is_cancelled(&self) -> bool;
}

impl CancelProbe for TokenSink {
    fn is_cancelled(&self) -> bool {
        self.is_closed()
    }
}

/// For callers without a request context (tests, warmup).
pub struct NeverCancelled;

impl CancelProbe for NeverCancelled {
    fn is_cancelled(&self) -> bool {
        false
    }
}

/// Prefix-cache identity of a resolve, mirroring
/// `BlockPool::probe_prefix_with_cache_salt`: the producer request and the
/// resolve must derive identical block hashes or the query keys are
/// unrelated.
#[derive(Clone, Copy, Debug, Default)]
pub struct CacheScope<'a> {
    pub(crate) cache_salt: Option<&'a str>,
    pub(crate) lora_name: Option<&'a str>,
}

impl<'a> CacheScope<'a> {
    /// Extra cache identity beyond the tokens, for a model that mixes
    /// per-request state into its KV pages: identical tokens under different
    /// salts must never hit each other's blocks.
    #[must_use]
    pub fn cache_salt(mut self, salt: &'a str) -> Self {
        self.cache_salt = Some(salt);
        self
    }

    /// Weight identity: blocks computed under one adapter never match
    /// another's.
    #[must_use]
    pub fn lora(mut self, name: &'a str) -> Self {
        self.lora_name = Some(name);
        self
    }
}

/// Per-request read policy for [`crate::KvStore::resolve_prefix`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ResolvePolicy {
    pub(crate) wait_for_full_hit: bool,
    pub(crate) full_pages: bool,
}

impl ResolvePolicy {
    /// For a caller that cannot recompute a miss: on the decode side of a
    /// disaggregated prefill/decode pair, admission requires the hit to
    /// cover the length the handoff committed to, and a partial hit is
    /// worthless. The tier query therefore becomes all-or-nothing, and a
    /// `Miss` is read as "the producing prefill's registration has not
    /// landed yet" — keep waiting under the deadline instead of concluding
    /// the cache is cold.
    #[must_use]
    pub fn wait_for_full_hit(mut self) -> Self {
        self.wait_for_full_hit = true;
        self
    }

    /// Resolve every complete page of the chain, boundary page included.
    /// The default cap excludes the last page (a plain request's final
    /// chunk must forward to emit a token, so a full-prompt hit is
    /// unusable); the decode side of a handoff resolves a pad-aligned
    /// chain it never prefills.
    #[must_use]
    pub fn full_pages(mut self) -> Self {
        self.full_pages = true;
        self
    }
}

/// Save quality-of-service, from the design doc's QoS split.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SaveClass {
    /// Fire-and-forget cacheability: a lost save forfeits a future hit, never
    /// correctness.
    Cacheable,
    /// Must-complete saves for a disaggregated prefill/decode handoff:
    /// [`crate::KvStore::retire`] parks the request's KV until these settle,
    /// so the consuming peer's checkpoint exists before the KV goes away.
    /// Lease semantics against that peer are future work.
    Handoff,
}

/// Per-request save bookkeeping, owned by the scheduler next to the
/// `RequestKv` (no hidden map in the store). Starts past the prefix-cache
/// hit — those blocks were stored by whoever first sealed them.
#[derive(Default)]
pub struct SaveCursor {
    pub(crate) saved_blocks: usize,
    /// Completion outcomes of this request's `Handoff`-class saves, awaited
    /// by [`crate::KvStore::retire`] before the KV releases.
    pub(crate) pending: Vec<oneshot::Receiver<Result<(), String>>>,
}

impl SaveCursor {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}
