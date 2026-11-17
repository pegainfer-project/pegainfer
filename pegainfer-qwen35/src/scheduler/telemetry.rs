//! Env-gated per-step ITL logging (`PEGAINFER_ITL_DEBUG`): one `ITL_STEP`
//! line per executed scheduler step with the plan kind, chunk/decode widths,
//! and CPU wall-time. Off by default; the check is cached so the serving path
//! pays one static lookup.

use super::*;

/// Env-gated per-step ITL diagnostics (issue #470). When `PEGAINFER_ITL_DEBUG`
/// is set, the scheduler emits one `ITL_STEP` line per executed step, tagging
/// the plan kind, the *actual* prefill-chunk token count associated with the
/// action, the active decode width, and the CPU wall-time. This lets the
/// mixed-load bench separate serial Unified stalls from overlap launch,
/// decode, completion, and wait actions instead of relying on the coarse
/// `[submit, last-token]` injection window. Off by default: no cost on the
/// normal bench path.
pub(super) fn itl_debug_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PEGAINFER_ITL_DEBUG").is_some())
}

/// Monotonic microseconds since the first ITL step, so `ITL_STEP` timestamps
/// are correlatable within one process run (paired with wall-clock epoch us).
fn itl_debug_mono_us() -> u128 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_micros()
}

pub(super) fn log_itl_step(
    step_start: Option<Instant>,
    plan: &str,
    prefill_tokens: usize,
    prefill_reqs: usize,
    decode_n: usize,
) {
    let Some(step_start) = step_start else {
        return;
    };
    let dur_us = step_start.elapsed().as_micros();
    let epoch_us = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_micros());
    info!(
        "ITL_STEP mono_us={} epoch_us={} plan={} prefill_tok={} prefill_reqs={} decode_n={} dur_us={}",
        itl_debug_mono_us(),
        epoch_us,
        plan,
        prefill_tokens,
        prefill_reqs,
        decode_n,
        dur_us
    );
}
