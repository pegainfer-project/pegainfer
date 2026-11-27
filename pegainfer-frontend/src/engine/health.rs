//! Engine health state machine: loading, ready, draining, failed, and scheduler heartbeat.
//!
//! HTTP mapping belongs to the protocol stack; this module owns state and decisions only.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

/// A scheduler that goes longer than this without a heartbeat is considered stalled.
pub const HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EnginePhase {
    Loading,
    Ready,
    Draining,
    Failed,
}

impl EnginePhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Loading => "loading",
            Self::Ready => "ready",
            Self::Draining => "draining",
            Self::Failed => "failed",
        }
    }

    fn to_u8(self) -> u8 {
        match self {
            Self::Loading => 0,
            Self::Ready => 1,
            Self::Draining => 2,
            Self::Failed => 3,
        }
    }

    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Loading,
            1 => Self::Ready,
            2 => Self::Draining,
            _ => Self::Failed,
        }
    }
}

struct Inner {
    phase: AtomicU8,
    last_beat_ms: AtomicU64,
    started: Instant,
    failure: Mutex<Option<String>>,
    #[cfg(test)]
    skew_ms: AtomicU64,
}

/// Cloneable engine health handle: the scheduler writes beats; the watchdog and HTTP read state.
#[derive(Clone)]
pub struct EngineHealth {
    inner: Arc<Inner>,
}

impl EngineHealth {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Inner {
                phase: AtomicU8::new(EnginePhase::Loading.to_u8()),
                last_beat_ms: AtomicU64::new(0),
                started: Instant::now(),
                failure: Mutex::new(None),
                #[cfg(test)]
                skew_ms: AtomicU64::new(0),
            }),
        }
    }

    pub fn phase(&self) -> EnginePhase {
        EnginePhase::from_u8(self.inner.phase.load(Ordering::Acquire))
    }

    pub fn failure(&self) -> Option<String> {
        self.inner
            .failure
            .lock()
            .expect("health failure mutex poisoned")
            .clone()
    }

    /// Called by the scheduler every loop; heartbeats stop after Failed.
    pub fn beat(&self) {
        if self.phase() == EnginePhase::Failed {
            return;
        }
        let elapsed_ms =
            u64::try_from(self.inner.started.elapsed().as_millis()).unwrap_or(u64::MAX);
        self.inner.last_beat_ms.store(elapsed_ms, Ordering::Release);
    }

    pub fn mark_ready(&self) {
        let _ = self.inner.phase.compare_exchange(
            EnginePhase::Loading.to_u8(),
            EnginePhase::Ready.to_u8(),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        self.beat();
    }

    pub fn mark_draining(&self) {
        let _ = self.inner.phase.compare_exchange(
            EnginePhase::Ready.to_u8(),
            EnginePhase::Draining.to_u8(),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    /// Returns whether this call entered Failed for the first time.
    pub fn mark_failed(&self, reason: impl Into<String>) -> bool {
        let previous = self
            .inner
            .phase
            .swap(EnginePhase::Failed.to_u8(), Ordering::AcqRel);
        if EnginePhase::from_u8(previous) == EnginePhase::Failed {
            return false;
        }
        *self
            .inner
            .failure
            .lock()
            .expect("health failure mutex poisoned") = Some(reason.into());
        true
    }

    pub fn heartbeat_age(&self) -> Duration {
        let elapsed_ms =
            u64::try_from(self.inner.started.elapsed().as_millis()).unwrap_or(u64::MAX);
        let beat_ms = self.inner.last_beat_ms.load(Ordering::Acquire);
        let age_ms = elapsed_ms.saturating_sub(beat_ms);
        #[cfg(test)]
        let age_ms = age_ms.saturating_add(self.inner.skew_ms.load(Ordering::Acquire));
        Duration::from_millis(age_ms)
    }

    /// Watchdog decision: return the failure reason once failed; otherwise mark failed in place when the heartbeat exceeds 30s.
    pub fn watchdog_failure(&self) -> Option<String> {
        if self.phase() == EnginePhase::Failed {
            return self.failure();
        }
        if self.phase() != EnginePhase::Ready {
            return None;
        }
        let age = self.heartbeat_age();
        if age <= HEARTBEAT_TIMEOUT {
            return None;
        }
        let reason = format!(
            "scheduler heartbeat stalled for {}s (timeout {}s)",
            age.as_secs(),
            HEARTBEAT_TIMEOUT.as_secs()
        );
        self.mark_failed(reason.clone());
        Some(reason)
    }

    #[cfg(test)]
    fn force_beat_age(&self, age: Duration) {
        let age_ms = u64::try_from(age.as_millis()).unwrap_or(u64::MAX);
        self.inner.skew_ms.store(age_ms, Ordering::Release);
    }
}

impl Default for EngineHealth {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::EngineHealth;
    use super::EnginePhase;
    use super::HEARTBEAT_TIMEOUT;

    #[test]
    fn failed_state_is_terminal_and_keeps_first_reason() {
        let health = EngineHealth::new();
        health.mark_ready();
        assert!(health.mark_failed("rank 0 prefill failed"));
        assert!(!health.mark_failed("later failure"));
        assert_eq!(health.phase(), EnginePhase::Failed);
        assert_eq!(health.failure().as_deref(), Some("rank 0 prefill failed"));
    }

    #[test]
    fn watchdog_flags_stalled_scheduler() {
        let health = EngineHealth::new();
        health.mark_ready();
        health.force_beat_age(HEARTBEAT_TIMEOUT + Duration::from_secs(1));
        let reason = health
            .watchdog_failure()
            .expect("stalled scheduler must fail");
        assert!(reason.contains("heartbeat stalled"));
        assert_eq!(health.phase(), EnginePhase::Failed);
    }

    #[test]
    fn draining_rejects_new_work_but_stays_alive() {
        let health = EngineHealth::new();
        health.mark_ready();
        health.mark_draining();
        assert_eq!(health.phase(), EnginePhase::Draining);
        assert!(health.watchdog_failure().is_none());
    }
}
