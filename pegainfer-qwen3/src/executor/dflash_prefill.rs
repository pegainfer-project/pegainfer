//! DFlash prefill-capture eligibility predicates.
//!
//! A request can seed the DFlash draft only if its prefill produces clean target
//! hidden states: no LoRA, no prefix-cache hit, no prompt logprobs, no
//! completion logprobs. Sampling params are irrelevant here — the prompt's
//! hidden states are sampling-independent, and sampled-verify (#512) speculates
//! the full sampling surface, so capture is as valid for a sampled request as
//! for a greedy one.

use super::PrefillStepItem;
use super::RequestId;

/// Whether a prefill request is eligible to capture DFlash target context.
fn dflash_prefill_supported(req: &PrefillStepItem) -> bool {
    req.lora_adapter.is_none()
        && req.cached_tokens == 0
        && req.logprobs.is_none()
        && req.prompt_logprobs.is_none()
}

/// Eligible AND continuous: either the first chunk, or a later chunk whose
/// earlier chunks already captured context (no gaps in the pending buffer).
pub(super) fn dflash_prefill_can_capture(
    req: &PrefillStepItem,
    pending_state_exists: bool,
) -> bool {
    dflash_prefill_supported(req) && (req.chunk_start == 0 || pending_state_exists)
}

/// Capture hidden states during this prefill step iff any request is eligible.
pub(super) fn should_capture_dflash_prefill_context(
    requests: &[PrefillStepItem],
    pending_state_exists: impl Fn(RequestId) -> bool,
) -> bool {
    !requests.is_empty()
        && requests
            .iter()
            .any(|req| dflash_prefill_can_capture(req, pending_state_exists(req.request_id)))
}

/// What to do with a request's DFlash state after a prefill step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DFlashPrefillAction {
    /// Context captured and prefill finished → ready to draft.
    MarkReady,
    /// Context captured but more chunks remain → keep the pending state.
    KeepPending,
    /// Ineligible → drop any stale state.
    Drop,
}

pub(super) fn dflash_prefill_action(
    captured_context: bool,
    completed: bool,
) -> DFlashPrefillAction {
    match (captured_context, completed) {
        (true, true) => DFlashPrefillAction::MarkReady,
        (true, false) => DFlashPrefillAction::KeepPending,
        (false, _) => DFlashPrefillAction::Drop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_action_table() {
        assert_eq!(
            dflash_prefill_action(true, true),
            DFlashPrefillAction::MarkReady
        );
        assert_eq!(
            dflash_prefill_action(true, false),
            DFlashPrefillAction::KeepPending
        );
        assert_eq!(
            dflash_prefill_action(false, true),
            DFlashPrefillAction::Drop
        );
        assert_eq!(
            dflash_prefill_action(false, false),
            DFlashPrefillAction::Drop
        );
    }
}
