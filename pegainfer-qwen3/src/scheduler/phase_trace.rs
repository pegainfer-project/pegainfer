//! Per-request host-side phase tracing for the scheduler.
//!
//! Emits `queue` → `prefill` → `decode` spans as children of the request span
//! the frontend opened (passed via [`GenerateRequest::trace_parent`]). Unlike
//! attributing phases from event arrival at the frontend demux — where the
//! `Scheduled` event and first token land microseconds apart and prefill looks
//! like ~0 — these spans are opened and closed *where the work happens* on the
//! scheduler thread, so their durations are the real host-side phase times.
//!
//! Span semantics (pinned by the contract test below):
//!
//! - `queue`: admission → the request's first prompt chunk reaches the GPU.
//! - `prefill`: first chunk on GPU → first token. This is prompt-phase *wall
//!   time*, not pure forward compute: under chunked prefill the span stays
//!   open between the request's own chunks, so it also covers the interleaved
//!   decode steps, other requests' prefills, and scheduler delay in between.
//!   `queue + prefill` reads as TTFT. To see what a slow request waited on,
//!   check which spans of *other* requests overlap its gaps in the trace
//!   waterfall — per-chunk compute spans would only be worth adding if that
//!   view ever proves insufficient.
//! - `decode`: first token → finish (any termination path).
//!
//! Live [`Span`]s are non-`Clone`, so they never enter the scheduler's `Clone`
//! request state; they live here in a side-table keyed by [`RequestId`]. The
//! request's [`SpanContext`] (which *is* `Copy`) rides through request state and
//! is handed to the tracker at admission.
//!
//! Every method is a cheap no-op when tracing is off: the parent context is
//! `None`, so no span is ever created and the table stays empty.

use std::collections::HashMap;

use fastrace::Span;
use fastrace::collector::SpanContext;

use crate::executor::RequestId;

/// The phase a request's open span currently represents. Guards against
/// double-transitions (e.g. a chunked prefill spanning several steps must open
/// `prefill` once, not once per chunk).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Phase {
    Queue,
    Prefill,
    Decode,
}

struct Tracked {
    parent: SpanContext,
    phase: Phase,
    /// The currently-open phase span. Dropping it ends the phase; reassigning
    /// closes the old phase and opens the new one.
    span: Span,
}

/// Side-table of in-flight request phase spans. Owned by the scheduler loop.
#[derive(Default)]
pub(crate) struct PhaseTracker {
    tracked: HashMap<RequestId, Tracked>,
}

impl PhaseTracker {
    /// Begin tracing a request in the `queue` phase. `parent` is the frontend's
    /// request span context; `None` (tracing off) makes this and every later
    /// call a no-op for this request.
    pub(crate) fn enter_queue(&mut self, id: RequestId, parent: Option<SpanContext>) {
        let Some(parent) = parent else { return };
        let span = Span::root("queue", parent);
        self.tracked.insert(
            id,
            Tracked {
                parent,
                phase: Phase::Queue,
                span,
            },
        );
    }

    /// The request's prompt work first reached the GPU: close `queue`, open
    /// `prefill`. Idempotent across chunked-prefill steps — the span opened on
    /// the first chunk stays open until `enter_decode`, which is what makes
    /// `prefill` prompt-phase wall time rather than per-chunk compute time
    /// (see module docs).
    pub(crate) fn enter_prefill(&mut self, id: RequestId) {
        if let Some(t) = self.tracked.get_mut(&id) {
            if t.phase != Phase::Queue {
                return;
            }
            t.phase = Phase::Prefill;
            t.span = Span::root("prefill", t.parent);
        }
    }

    /// The first token was produced: close `prefill`, open `decode`.
    pub(crate) fn enter_decode(&mut self, id: RequestId) {
        if let Some(t) = self.tracked.get_mut(&id) {
            if t.phase == Phase::Decode {
                return;
            }
            t.phase = Phase::Decode;
            t.span = Span::root("decode", t.parent);
        }
    }

    /// The request finished (or was dropped/rejected): close its open span and
    /// stop tracking it. Dropping the whole request trace is the frontend root
    /// span's job; this only ends the last phase span.
    pub(crate) fn finish(&mut self, id: RequestId) {
        self.tracked.remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use fastrace::collector::Config;
    use fastrace::collector::SpanRecord;
    use fastrace::collector::TestReporter;

    use super::*;

    /// Contract test for the span tree the scheduler emits: span names,
    /// parentage under the frontend's request root, transition order,
    /// chunked-prefill idempotency, cleanup on early finish, and the
    /// tracing-off no-op. Drives the tracker directly — no GPU, no scheduler.
    ///
    /// Must stay the only test in the crate that installs a fastrace reporter:
    /// the reporter is global, so parallel reporter tests would race.
    #[test]
    fn phase_span_contract() {
        let (reporter, spans) = TestReporter::new();
        fastrace::set_reporter(reporter, Config::default());

        let mut tracker = PhaseTracker::default();

        // Happy path, mimicking the frontend: open the request root span and
        // hand its context to the tracker at admission.
        let root = Span::root("request", SpanContext::random());
        let ctx = SpanContext::from_span(&root).expect("root span is sampled");
        let id = RequestId::new(1);
        tracker.enter_queue(id, Some(ctx));
        tracker.enter_prefill(id);
        // Chunked prefill re-enters once per step: must not open a second span.
        tracker.enter_prefill(id);
        tracker.enter_decode(id);
        tracker.finish(id);
        drop(root);

        // Rejected before prefill: queue span only, closed by finish; later
        // transitions are no-ops.
        let rejected_root = Span::root("request", SpanContext::random());
        let rejected_ctx = SpanContext::from_span(&rejected_root).unwrap();
        let rejected = RequestId::new(2);
        tracker.enter_queue(rejected, Some(rejected_ctx));
        tracker.finish(rejected);
        tracker.enter_decode(rejected);
        drop(rejected_root);

        // Tracing off: no parent context, no spans, no lingering state.
        let untraced = RequestId::new(3);
        tracker.enter_queue(untraced, None);
        tracker.enter_prefill(untraced);
        tracker.enter_decode(untraced);
        tracker.finish(untraced);
        assert!(tracker.tracked.is_empty());

        fastrace::flush();
        let spans = spans.lock().clone();

        // Two request roots, one per request.
        assert_eq!(spans.iter().filter(|s| s.name == "request").count(), 2);

        // A request's phase spans share its trace and parent onto its root.
        let phases_of = |root_ctx: &SpanContext| -> Vec<&SpanRecord> {
            let mut phases: Vec<&SpanRecord> = spans
                .iter()
                .filter(|s| s.trace_id == root_ctx.trace_id && s.name != "request")
                .collect();
            for span in &phases {
                assert_eq!(span.parent_id, root_ctx.span_id);
            }
            phases.sort_by_key(|s| s.begin_time_unix_ns);
            phases
        };

        // Happy path: queue → prefill → decode, exactly one span per phase —
        // the repeated enter_prefill (chunked prefill) opened nothing new.
        let happy = phases_of(&ctx);
        let names: Vec<&str> = happy.iter().map(|s| s.name.as_ref()).collect();
        assert_eq!(names, ["queue", "prefill", "decode"]);

        // Rejected path: queue only; the post-finish enter_decode was a no-op.
        let rejected = phases_of(&rejected_ctx);
        let names: Vec<&str> = rejected.iter().map(|s| s.name.as_ref()).collect();
        assert_eq!(names, ["queue"]);
    }
}
