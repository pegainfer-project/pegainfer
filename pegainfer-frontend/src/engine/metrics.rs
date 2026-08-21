//! What the scheduler republishes about itself: [`SchedulerMetrics`], the
//! per-iteration snapshot of occupancy gauges plus whatever richer counters a
//! model line serves (today: cumulative speculative-decode acceptance, when a
//! draft model is loaded).

use std::error::Error;
use std::fmt;

/// The metrics snapshot a scheduler republishes after every step.
///
/// `kv_used_blocks` is the load signal an out-of-band consumer (e.g. a Dynamo
/// KV router) scores against; `kv_total_blocks` is the engine's whole-pool
/// capacity (the same number advertised as the servable ceiling), so the
/// consumer can derive fractional usage without a second query. Readers only
/// ever see the latest snapshot — stepped engines publish into a shared cell
/// ([`super::MetricsPublisher`]), legacy engines over a `watch` channel — so
/// counters carried here must be cumulative totals, not per-step deltas (see
/// [`SpecDecodeCounters`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchedulerMetrics {
    pub kv_used_blocks: u64,
    pub kv_total_blocks: u64,
    /// Requests currently occupying a decode/prefill slot.
    pub num_running_reqs: u64,
    /// Requests admitted but not yet running (KV pressure, prefetch wait).
    pub num_waiting_reqs: u64,
    /// Cumulative spec-decode counters, or `None` when no draft model is loaded.
    pub spec_decode: Option<SpecDecodeCounters>,
}

/// Upper bound on a drafter's `K`, fixing the width of
/// [`SpecDecodeCounters::num_accepted_tokens_per_pos`].
pub const MAX_SPEC_TOKENS: usize = 32;

#[derive(Debug, Eq, PartialEq)]
pub struct SpecWidthUnsupported {
    pub num_spec_tokens: usize,
}

impl fmt::Display for SpecWidthUnsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "draft checkpoint proposes K={} tokens per verify step, above the \
             MAX_SPEC_TOKENS={MAX_SPEC_TOKENS} the acceptance metrics support",
            self.num_spec_tokens
        )
    }
}

impl Error for SpecWidthUnsupported {}

/// Cumulative speculative-decode acceptance counters, monotone since the draft
/// model was loaded. Use totals rather than per-step deltas, because a
/// [`SchedulerMetrics`] reader only keeps the newest value: a reader that
/// misses a step would lose that delta for good, but it can still read totals
/// correctly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecDecodeCounters {
    /// The drafter's configured `K`, and the number of leading
    /// [`Self::num_accepted_tokens_per_pos`] entries that carry meaning.
    pub num_spec_tokens: u64,
    /// Draft proposals — one per request per verify step.
    pub num_drafts: u64,
    /// Draft tokens proposed for verification (pre-acceptance) in total.
    pub num_draft_tokens: u64,
    /// Draft tokens accepted in total, excluding the bonus token.
    pub num_accepted_tokens: u64,
    /// `[i]` is how often the `i`-th draft position was accepted. Only
    /// `[..num_spec_tokens]` is meaningful; the tail stays zero.
    pub num_accepted_tokens_per_pos: [u64; MAX_SPEC_TOKENS],
}

impl SpecDecodeCounters {
    pub fn new(num_spec_tokens: usize) -> Result<Self, SpecWidthUnsupported> {
        if num_spec_tokens > MAX_SPEC_TOKENS {
            return Err(SpecWidthUnsupported { num_spec_tokens });
        }
        Ok(Self {
            num_spec_tokens: num_spec_tokens as u64,
            ..Self::default()
        })
    }

    pub fn observe_draft(&mut self, num_draft_tokens: usize, num_accepted: usize) {
        self.num_drafts += 1;
        self.num_draft_tokens += num_draft_tokens as u64;
        self.num_accepted_tokens += num_accepted as u64;
        let tallied = num_accepted.min(self.num_spec_tokens as usize);
        for slot in self.num_accepted_tokens_per_pos.iter_mut().take(tallied) {
            *slot += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observe_draft_tallies_a_prefix_per_position() {
        let mut counters = SpecDecodeCounters::new(3).expect("K within bounds");
        // Two drafts proposed, both accepted; then three proposed, one accepted.
        counters.observe_draft(2, 2);
        counters.observe_draft(3, 1);
        assert_eq!(counters.num_drafts, 2);
        assert_eq!(counters.num_draft_tokens, 5);
        assert_eq!(counters.num_accepted_tokens, 3);
        // Acceptance is prefix-shaped: `n` accepted tokens credit positions
        // `0..n`, so a per-position histogram never rises.
        assert_eq!(&counters.num_accepted_tokens_per_pos[..3], &[2, 1, 0]);
    }

    #[test]
    fn oversized_k_is_rejected_not_truncated() {
        assert!(SpecDecodeCounters::new(MAX_SPEC_TOKENS).is_ok());
        assert_eq!(
            SpecDecodeCounters::new(MAX_SPEC_TOKENS + 1),
            Err(SpecWidthUnsupported {
                num_spec_tokens: MAX_SPEC_TOKENS + 1
            })
        );
    }
}
