//! DFlash prefill-capture eligibility predicates.
//!
//! A request can seed the DFlash draft only if its prefill produces clean target
//! hidden states: no LoRA, no prompt logprobs, no completion logprobs. A prefix
//! hit is fine when the drafter's paged K/V was restored for the matched block
//! IDs; the continuity check below requires that state for any non-zero
//! `chunk_start`.
//! Sampling params are irrelevant here — the prompt's hidden states are
//! sampling-independent, and sampled-verify (#512) speculates the full sampling
//! surface, so capture is as valid for a sampled request as for a greedy one.

use super::PrefillStepItem;
use super::RequestId;

/// Whether a prefill produces clean target hidden states to seed a drafter
/// (DFlash and EAGLE-3 alike).
pub(super) fn prefill_capture_supported(req: &PrefillStepItem) -> bool {
    req.lora_adapter.is_none() && req.logprobs.is_none() && req.prompt_logprobs.is_none()
}

/// Eligible AND continuous: either the first chunk, or a later chunk whose
/// earlier chunks already captured context (no gaps in the pending buffer).
pub(super) fn dflash_prefill_can_capture(
    req: &PrefillStepItem,
    pending_state_exists: bool,
) -> bool {
    prefill_capture_supported(req) && (req.chunk_start == 0 || pending_state_exists)
}

/// Requests that must not ride a unified prefill step. The fused forward
/// captures no target hidden context, so under speculation an eligible request
/// that took that path would never become draft-ready and its drafter would
/// silently no-op for the rest of the conversation. The scheduler routes these
/// to a dedicated prefill with a deliberately loose superset
/// (`scheduler/plan.rs`); this is the strict, fail-closed side of that rule.
pub(super) fn capture_requests_in_unified(
    speculative: bool,
    requests: &[PrefillStepItem],
) -> Vec<RequestId> {
    if !speculative {
        return Vec::new();
    }
    requests
        .iter()
        .filter(|req| prefill_capture_supported(req))
        .map(|req| req.request_id)
        .collect()
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

    fn request(id: u64) -> PrefillStepItem {
        PrefillStepItem::new(
            RequestId::new(id),
            vec![1, 2, 3],
            8,
            pegainfer_frontend::sampler::SamplingParams::default(),
            None,
            None,
        )
    }

    #[test]
    fn unified_refuses_capture_eligible_prefills_only_under_speculation() {
        let eligible = request(1);
        assert!(
            capture_requests_in_unified(false, std::slice::from_ref(&eligible)).is_empty(),
            "without a drafter the fused path is fine"
        );
        assert_eq!(
            capture_requests_in_unified(true, std::slice::from_ref(&eligible)),
            vec![eligible.request_id],
            "a capture-eligible request under speculation must be refused"
        );

        // LoRA and logprob requests produce no clean capture, so a dedicated
        // prefill would drop them too; the fused path may keep them.
        let mut lora = request(2);
        lora.lora_adapter = Some("adapter".to_string());
        let mut logprobs = request(3);
        logprobs.logprobs = Some(0);
        let mut prompt_logprobs = request(4);
        prompt_logprobs.prompt_logprobs = Some(0);
        for ineligible in [lora, logprobs, prompt_logprobs] {
            assert!(
                capture_requests_in_unified(true, std::slice::from_ref(&ineligible)).is_empty(),
                "{:?} is not capture-eligible",
                ineligible.request_id
            );
        }
    }

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
