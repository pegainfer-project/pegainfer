//! Test-only probes for the free-running DP engine gates
//! (`docs/models/glm52/free-running-dp.md` §8, gates 4–5).
//!
//! Gate 4 (padding byte constancy) needs every step's router output bytes
//! per rank; gate 5 (MTP fixed chain) needs per-round emptiness and wall
//! time. Both are recorded from the serving path itself — the whole point
//! is probing the production step/round code, not an isolated kernel — so
//! the hooks live in `runner::step` and `scheduler::mtp::run_mtp_round`
//! behind [`enabled`], and the gate drains the records between phases.

use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

static ENABLED: AtomicBool = AtomicBool::new(false);
static STEP_ROUTES: Mutex<Vec<StepRouteRecord>> = Mutex::new(Vec::new());
static MTP_ROUNDS: Mutex<Vec<MtpRoundRecord>> = Mutex::new(Vec::new());

/// One rank's router output after one decode step: the last MoE layer's
/// top-k routing for the step's bucket rows. For an all-padding step these
/// bytes must be constant step over step — the padding-as-protocol gate.
pub(crate) struct StepRouteRecord {
    pub(crate) rank: usize,
    pub(crate) bucket: usize,
    pub(crate) active_rows: usize,
    pub(crate) topk_idx: Vec<i32>,
    /// f32 weights as raw bits — the gate compares bytes, not values.
    pub(crate) topk_weight_bits: Vec<u32>,
}

/// One rank's MTP round: whether it entered with no work, and how long its
/// own submit-to-join took. Each engine records its own round — the
/// coordinator-era record held all ranks because only the coordinator saw
/// them.
pub(crate) struct MtpRoundRecord {
    pub(crate) rank: usize,
    pub(crate) empty: bool,
    pub(crate) elapsed: Duration,
}

pub(crate) fn set_enabled(on: bool) {
    ENABLED.store(on, Ordering::SeqCst);
}

pub(crate) fn enabled() -> bool {
    ENABLED.load(Ordering::SeqCst)
}

pub(crate) fn record_step_route(record: StepRouteRecord) {
    STEP_ROUTES
        .lock()
        .expect("freerun step-route probe lock poisoned")
        .push(record);
}

pub(crate) fn record_mtp_round(rank: usize, empty: bool, elapsed: Duration) {
    if !enabled() {
        return;
    }
    MTP_ROUNDS
        .lock()
        .expect("freerun MTP-round probe lock poisoned")
        .push(MtpRoundRecord {
            rank,
            empty,
            elapsed,
        });
}

pub(crate) fn take_step_routes() -> Vec<StepRouteRecord> {
    std::mem::take(
        &mut STEP_ROUTES
            .lock()
            .expect("freerun step-route probe lock poisoned"),
    )
}

pub(crate) fn take_mtp_rounds() -> Vec<MtpRoundRecord> {
    std::mem::take(
        &mut MTP_ROUNDS
            .lock()
            .expect("freerun MTP-round probe lock poisoned"),
    )
}
