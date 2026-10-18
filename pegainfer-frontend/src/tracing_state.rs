//! Global tracing-enabled flag, shared across crates.
//!
//! The frontend and the model schedulers both need to know whether request
//! tracing is on *before* doing any span work, so the check must live in this
//! low-level contract crate that both depend on. `pegainfer-core::tracing` owns
//! the reporter and flips this flag once (and only once) a reporter is actually
//! installed. When off, callers skip span creation entirely — fastrace is
//! compiled with `enable`, so an unguarded `Span::root` would otherwise build
//! and immediately discard a real span on every request.

use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

static TRACING_ENABLED: AtomicBool = AtomicBool::new(false);

/// Mark request tracing as active. Called once by the reporter installer.
pub fn set_enabled(enabled: bool) {
    TRACING_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether request tracing is active. A relaxed load on the request hot path;
/// callers use it to avoid building spans that would be discarded.
#[inline]
pub(crate) fn is_enabled() -> bool {
    TRACING_ENABLED.load(Ordering::Relaxed)
}
