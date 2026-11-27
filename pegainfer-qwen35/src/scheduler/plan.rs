/// Decode cohort identifier: single-GPU dual-cohort alternates A/B; TP and single-cohort use A only.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum CohortId {
    A = 0,
    B = 1,
}

impl CohortId {
    pub(super) const fn index(self) -> usize {
        self as usize
    }

    pub(super) const fn other(self) -> Self {
        match self {
            Self::A => Self::B,
            Self::B => Self::A,
        }
    }
}

/// Selects the target cohort on admission: fills A first so `c <= decode_width` stays on A only,
/// spilling into B only once A is full.
pub(super) fn assign_cohort(cohort_free: [usize; 2]) -> Option<CohortId> {
    if cohort_free[CohortId::A.index()] > 0 {
        Some(CohortId::A)
    } else if cohort_free[CohortId::B.index()] > 0 {
        Some(CohortId::B)
    } else {
        None
    }
}

/// Selects which cohort decodes on this step and the next turn cursor:
/// - Only A active: run A (`c <= 64` behavior unchanged);
/// - Only B active: run B;
/// - Both A and B active: strictly alternate (A -> B -> A -> B).
pub(super) fn next_decode_cohort(
    active_counts: [usize; 2],
    next_turn: CohortId,
) -> Option<(CohortId, CohortId)> {
    let a = active_counts[CohortId::A.index()] > 0;
    let b = active_counts[CohortId::B.index()] > 0;
    match (a, b) {
        (false, false) => None,
        (true, false) => Some((CohortId::A, CohortId::A)),
        (false, true) => Some((CohortId::B, CohortId::B)),
        (true, true) => Some((next_turn, next_turn.other())),
    }
}

pub(super) struct AdmissionOutcome<T> {
    pub(super) pending: Vec<T>,
    pub(super) deferred: Vec<T>,
    pub(super) rejected: Vec<(T, RejectReason)>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ActiveKvBudget {
    pub(super) prompt_len: usize,
    pub(super) generated_count: usize,
    pub(super) max_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PrefillQueueState {
    pub(super) remaining_tokens: usize,
}

/// Prompt rows in one standard chunk.
///
/// The serving cells that set Qwen3.5's tail target were tuned against a
/// 1024-row prefill step, so the default budget is this one chunk.
pub(super) const STANDARD_PROMPT_CHUNK: usize = 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SlotCompaction {
    pub(super) moved_from: usize,
    pub(super) moved_to: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RejectReason {
    ZeroMaxTokens,
    ContextLength { limit: usize },
    KvBudget,
}

/// FIFO prompt slices and whether the active decode batch advances.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct StepPlan {
    pub(super) decode: bool,
    /// Row counts for consecutive prompts at the front of the prefill queue.
    pub(super) prefill: Vec<usize>,
}

impl StepPlan {
    pub(super) fn is_empty(&self) -> bool {
        !self.decode && self.prefill.is_empty()
    }
}

/// Pack FIFO prompt slices into `budget` and at most `max_prefill_reqs` prompts
/// (bounded by decode `SampleScratch` width); decode rows do not consume `budget`.
/// A prompt crossing the budget stays at the queue front and resumes on the next step.
pub(super) fn plan_step(
    active_decodes: usize,
    budget: usize,
    max_prefill_reqs: usize,
    prefilling: &[PrefillQueueState],
) -> StepPlan {
    assert!(
        budget > 0,
        "Qwen3.5 scheduler requires a positive prefill budget"
    );
    let mut remaining = budget;
    let mut prefill = Vec::new();
    for state in prefilling {
        if prefill.len() >= max_prefill_reqs {
            break;
        }
        let rows = state.remaining_tokens.min(remaining);
        if rows == 0 {
            break;
        }
        prefill.push(rows);
        remaining -= rows;
        if rows < state.remaining_tokens {
            break;
        }
    }
    StepPlan {
        decode: active_decodes > 0,
        prefill,
    }
}

pub(super) fn admit_pending_requests<T>(
    pending: Vec<T>,
    active: &[ActiveKvBudget],
    max_batch: usize,
    page_size: usize,
    available_pages: usize,
    max_request_pages: usize,
    max_context_tokens: usize,
    mut prompt_len: impl FnMut(&T) -> usize,
    mut max_tokens: impl FnMut(&T) -> usize,
    mut reusable_pages: impl FnMut(&T) -> usize,
) -> AdmissionOutcome<T> {
    assert!(page_size > 0, "Qwen3.5 KV page size must be non-zero");

    let mut page_budget = available_pages.saturating_sub(active_future_pages(active, page_size));
    let slot_budget = max_batch.saturating_sub(active.len());
    let mut admitted = Vec::new();
    let mut still_deferred = Vec::new();
    let mut rejected = Vec::new();
    let mut blocked = false;

    for req in pending {
        let prompt_len = prompt_len(&req);
        let max_tokens = max_tokens(&req);
        if max_tokens == 0 {
            rejected.push((req, RejectReason::ZeroMaxTokens));
            continue;
        }
        if prompt_len.saturating_add(max_tokens) > max_context_tokens {
            rejected.push((
                req,
                RejectReason::ContextLength {
                    limit: max_context_tokens,
                },
            ));
            continue;
        }

        let request_pages = request_lifetime_pages(prompt_len, max_tokens, page_size);
        if request_pages > max_request_pages {
            rejected.push((req, RejectReason::KvBudget));
            continue;
        }

        if blocked || admitted.len() >= slot_budget {
            blocked = true;
            still_deferred.push(req);
            continue;
        }

        let fresh_pages = request_pages.saturating_sub(reusable_pages(&req));
        if fresh_pages > page_budget {
            blocked = true;
            still_deferred.push(req);
            continue;
        }

        page_budget -= fresh_pages;
        admitted.push(req);
    }

    AdmissionOutcome {
        pending: admitted,
        deferred: still_deferred,
        rejected,
    }
}

fn pages_needed(token_count: usize, page_size: usize) -> usize {
    token_count.div_ceil(page_size)
}

// One-token completions finish after prefill, before schedule_decode provisions
// a dangling generation block. Multi-token requests can draw that extra block,
// so admission reserves prompt + max_tokens for their lifetime peak.
fn request_lifetime_pages(prompt_len: usize, max_tokens: usize, page_size: usize) -> usize {
    pages_needed(request_lifetime_tokens(prompt_len, max_tokens), page_size)
}

pub(super) fn request_lifetime_tokens(prompt_len: usize, max_tokens: usize) -> usize {
    if max_tokens <= 1 {
        prompt_len
    } else {
        prompt_len.saturating_add(max_tokens)
    }
}

fn current_active_tokens(req: ActiveKvBudget) -> usize {
    req.prompt_len
        .saturating_add(req.generated_count.saturating_sub(1))
}

fn active_future_pages(active: &[ActiveKvBudget], page_size: usize) -> usize {
    active
        .iter()
        .map(|req| {
            let max_pages = request_lifetime_pages(req.prompt_len, req.max_tokens, page_size);
            let current_pages = pages_needed(current_active_tokens(*req), page_size);
            assert!(
                current_pages <= max_pages,
                "active Qwen3.5 request exceeded its admitted KV lifetime budget"
            );
            max_pages.saturating_sub(current_pages)
        })
        .sum()
}

pub(super) fn slot_for_new_request(active_count: usize, max_batch: usize) -> Option<usize> {
    (active_count < max_batch).then_some(active_count)
}

/// KV-lifetime budget of a request for chunked prefill
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct PrefillKvBudget {
    pub(super) current_tokens: usize,
    pub(super) prompt_len: usize,
    pub(super) max_tokens: usize,
}

/// Pages an in-flight chunked prefill will still allocate before it finishes
pub(super) fn prefilling_future_pages(prefilling: &[PrefillKvBudget], page_size: usize) -> usize {
    prefilling
        .iter()
        .map(|req| {
            let max_pages = request_lifetime_pages(req.prompt_len, req.max_tokens, page_size);
            let current_pages = pages_needed(req.current_tokens, page_size);
            assert!(
                current_pages <= max_pages,
                "Qwen3.5 chunked prefill exceeded its admitted KV lifetime budget"
            );
            max_pages.saturating_sub(current_pages)
        })
        .sum()
}

pub(super) fn compaction_after_retire(
    active_len_before: usize,
    retired_idx: usize,
) -> Option<SlotCompaction> {
    assert!(
        retired_idx < active_len_before,
        "retired Qwen3.5 slot index must be active"
    );

    let last = active_len_before - 1;
    (retired_idx < last).then_some(SlotCompaction {
        moved_from: last,
        moved_to: retired_idx,
    })
}

#[cfg(test)]
mod tests {
    use pegainfer_kv_cache::BlockPool;

    use super::*;

    #[derive(Clone, Debug)]
    struct Pending {
        id: u32,
        prompt_len: usize,
        max_tokens: usize,
    }

    fn pending(id: u32, prompt_len: usize) -> Pending {
        Pending {
            id,
            prompt_len,
            max_tokens: 1,
        }
    }

    fn pending_with_max(id: u32, prompt_len: usize, max_tokens: usize) -> Pending {
        Pending {
            id,
            prompt_len,
            max_tokens,
        }
    }

    fn ids(reqs: &[Pending]) -> Vec<u32> {
        reqs.iter().map(|req| req.id).collect()
    }

    fn rejected_ids(reqs: &[(Pending, RejectReason)]) -> Vec<u32> {
        reqs.iter().map(|(req, _)| req.id).collect()
    }

    fn four_standard_prompts() -> [PrefillQueueState; 4] {
        [PrefillQueueState {
            remaining_tokens: STANDARD_PROMPT_CHUNK,
        }; 4]
    }

    fn prompts(remaining: &[usize]) -> Vec<PrefillQueueState> {
        remaining
            .iter()
            .map(|&remaining_tokens| PrefillQueueState { remaining_tokens })
            .collect()
    }

    #[test]
    fn default_budget_is_one_standard_prompt_chunk() {
        let plan = plan_step(
            1,
            crate::DEFAULT_MAX_PREFILL_TOKENS,
            64,
            &four_standard_prompts(),
        );
        assert!(plan.decode);
        assert_eq!(plan.prefill, vec![STANDARD_PROMPT_CHUNK]);
    }

    const PACK_BUDGET: usize = STANDARD_PROMPT_CHUNK * 4;

    #[test]
    fn plan_packs_whole_prompts_in_fifo_order() {
        let plan = plan_step(1, PACK_BUDGET, 64, &prompts(&[300, 700, 900, 1500]));
        assert!(plan.decode);
        assert_eq!(plan.prefill, vec![300, 700, 900, 1500]);
    }

    #[test]
    fn plan_slices_the_prompt_that_crosses_the_budget() {
        let plan = plan_step(1, PACK_BUDGET, 64, &prompts(&[6000]));
        assert_eq!(plan.prefill, vec![PACK_BUDGET]);
    }

    #[test]
    fn plan_slices_against_the_budget_left_by_earlier_prompts() {
        let plan = plan_step(1, PACK_BUDGET, 64, &prompts(&[2000, 3000, 1]));
        assert_eq!(plan.prefill, vec![2000, 2096]);
    }

    #[test]
    fn plan_resumes_the_sliced_tail_before_the_next_prompt() {
        // The 6000-token prompt was sliced at 4096; its tail is now the front
        // of the queue and must be finished before the next prompt moves.
        let plan = plan_step(1, PACK_BUDGET, 64, &prompts(&[1904, 1024]));
        assert_eq!(plan.prefill, vec![1904, 1024]);
    }

    #[test]
    fn plan_packs_everything_under_a_huge_budget() {
        let plan = plan_step(
            1,
            usize::MAX,
            64,
            &prompts(&[STANDARD_PROMPT_CHUNK, STANDARD_PROMPT_CHUNK, 7]),
        );
        assert_eq!(
            plan.prefill,
            vec![STANDARD_PROMPT_CHUNK, STANDARD_PROMPT_CHUNK, 7]
        );
    }

    #[test]
    fn plan_caps_prompt_count_to_decode_width() {
        let queue = prompts(&vec![1; 128]);
        let plan = plan_step(1, 1024, 64, &queue);
        assert_eq!(plan.prefill.len(), 64);
    }

    #[test]
    fn plan_never_widens_a_smaller_budget() {
        let plan = plan_step(1, 512, 64, &prompts(&[STANDARD_PROMPT_CHUNK; 4]));
        assert_eq!(plan.prefill, vec![512]);
    }

    #[test]
    fn plan_breaks_on_a_finished_prompt() {
        let plan = plan_step(
            1,
            crate::DEFAULT_MAX_PREFILL_TOKENS,
            64,
            &prompts(&[0, STANDARD_PROMPT_CHUNK]),
        );
        assert!(plan.prefill.is_empty());
        assert!(plan.decode);
        assert!(!plan.is_empty());
    }

    #[test]
    fn plan_is_prefill_only_without_decode_work() {
        let plan = plan_step(0, crate::DEFAULT_MAX_PREFILL_TOKENS, 64, &prompts(&[700]));
        assert!(!plan.decode);
        assert_eq!(plan.prefill, vec![700]);
        assert!(!plan.is_empty());
    }

    #[test]
    fn plan_is_decode_only_without_prompt_work() {
        let plan = plan_step(1, crate::DEFAULT_MAX_PREFILL_TOKENS, 64, &[]);
        assert!(plan.decode);
        assert!(plan.prefill.is_empty());
        assert!(!plan.is_empty());
    }

    #[test]
    fn plan_is_empty_without_prompt_or_decode_work() {
        let plan = plan_step(0, crate::DEFAULT_MAX_PREFILL_TOKENS, 64, &[]);
        assert!(plan.is_empty());
    }

    #[test]
    #[should_panic(expected = "positive prefill budget")]
    fn plan_rejects_a_zero_budget() {
        plan_step(1, 0, 64, &[]);
    }

    #[test]
    fn cohort_assignment_fills_a_before_spilling_to_b() {
        assert_eq!(assign_cohort([64, 64]), Some(CohortId::A));
        assert_eq!(assign_cohort([1, 64]), Some(CohortId::A));
        assert_eq!(assign_cohort([0, 64]), Some(CohortId::B));
        assert_eq!(assign_cohort([0, 1]), Some(CohortId::B));
        assert_eq!(assign_cohort([0, 0]), None);
    }

    #[test]
    fn decode_cohort_selection_stays_on_a_under_64_and_alternates_above_64() {
        assert_eq!(next_decode_cohort([0, 0], CohortId::A), None);
        assert_eq!(
            next_decode_cohort([64, 0], CohortId::A),
            Some((CohortId::A, CohortId::A))
        );
        assert_eq!(
            next_decode_cohort([32, 0], CohortId::B),
            Some((CohortId::A, CohortId::A))
        );
        assert_eq!(
            next_decode_cohort([0, 12], CohortId::A),
            Some((CohortId::B, CohortId::B))
        );
        let (first, turn1) = next_decode_cohort([64, 32], CohortId::A).unwrap();
        let (second, turn2) = next_decode_cohort([64, 32], turn1).unwrap();
        let (third, turn3) = next_decode_cohort([64, 32], turn2).unwrap();
        assert_eq!(
            (first, second, third),
            (CohortId::A, CohortId::B, CohortId::A)
        );
        assert_eq!(turn3, CohortId::B);
    }

    #[test]
    fn admission_admits_128_resident_requests_and_defers_129th() {
        let pending: Vec<Pending> = (0..129).map(|i| pending_with_max(i, 16, 16)).collect();
        let outcome = admit_pending_requests(
            pending,
            &[],
            128,
            16,
            10_000,
            10_000,
            262_144,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 0,
        );
        assert_eq!(outcome.pending.len(), 128);
        assert_eq!(ids(&outcome.deferred), vec![128]);
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_respects_slot_capacity_and_active_decode_reserve() {
        let active = [
            ActiveKvBudget {
                prompt_len: 16,
                generated_count: 1,
                max_tokens: 18,
            },
            ActiveKvBudget {
                prompt_len: 16,
                generated_count: 1,
                max_tokens: 1,
            },
        ];
        let outcome = admit_pending_requests(
            vec![pending(1, 16), pending(2, 16), pending(3, 16)],
            &active,
            4,
            16,
            6,
            6,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 0,
        );

        assert_eq!(
            ids(&outcome.pending),
            vec![1, 2],
            "active future KV growth is reserved before admitting new requests"
        );
        assert_eq!(
            ids(&outcome.deferred),
            vec![3],
            "requests beyond the remaining slot/page budget stay deferred"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_is_fcfs_and_keeps_later_requests_deferred_after_first_miss() {
        let outcome = admit_pending_requests(
            vec![pending(1, 16), pending(2, 33), pending(3, 16)],
            &[],
            8,
            16,
            3,
            8,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 0,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert_eq!(
            ids(&outcome.deferred),
            vec![2, 3],
            "a later smaller request must not jump ahead of an earlier budget miss"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_keeps_order_when_first_pending_request_misses_budget() {
        let outcome = admit_pending_requests(
            vec![pending(1, 33), pending(2, 16)],
            &[],
            8,
            16,
            2,
            8,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 0,
        );

        assert!(outcome.pending.is_empty());
        assert_eq!(
            ids(&outcome.deferred),
            vec![1, 2],
            "a later smaller request must not bypass the first deferred request"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_uses_ceil_div_at_page_boundaries() {
        let outcome = admit_pending_requests(
            vec![pending(1, 15), pending(2, 16), pending(3, 17)],
            &[],
            8,
            16,
            3,
            8,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 0,
        );

        assert_eq!(
            ids(&outcome.pending),
            vec![1, 2],
            "15 and 16 tokens each use one page"
        );
        assert_eq!(
            ids(&outcome.deferred),
            vec![3],
            "17 tokens needs two pages and waits when only one page remains"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_rejects_invalid_requests() {
        let outcome = admit_pending_requests(
            vec![
                pending_with_max(1, 16, 16), // 32 context tokens: admitted.
                pending_with_max(2, 16, 17), // 33 context tokens: rejected.
                pending_with_max(3, 40, 1),  // prompt alone exceeds the window.
                pending_with_max(4, 16, 0),  // no output token requested.
            ],
            &[],
            8,
            16,
            1000,
            1000,
            32,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 0,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert!(outcome.deferred.is_empty());
        assert_eq!(rejected_ids(&outcome.rejected), vec![2, 3, 4]);
        for (_, reason) in &outcome.rejected[..2] {
            assert!(
                matches!(reason, RejectReason::ContextLength { limit: 32 }),
                "over-window requests must be rejected on context length"
            );
        }
        assert!(matches!(outcome.rejected[2].1, RejectReason::ZeroMaxTokens));
    }

    #[test]
    fn context_window_rejection_takes_precedence_over_kv_budget() {
        let outcome = admit_pending_requests(
            vec![pending_with_max(1, 40, 40)], // exceeds both 32-token context and 1 page.
            &[],
            8,
            16,
            1000,
            1,
            32,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 0,
        );

        assert!(outcome.pending.is_empty());
        assert!(outcome.deferred.is_empty());
        assert_eq!(rejected_ids(&outcome.rejected), vec![1]);
        assert!(
            matches!(
                outcome.rejected[0].1,
                RejectReason::ContextLength { limit: 32 }
            ),
            "a static context-window violation should not be reported as KV pressure"
        );
    }

    #[test]
    fn admission_returns_all_pending_when_active_batch_is_at_slot_capacity() {
        let outcome = admit_pending_requests(
            vec![pending(1, 1), pending(2, 1)],
            &[
                ActiveKvBudget {
                    prompt_len: 1,
                    generated_count: 1,
                    max_tokens: 1,
                },
                ActiveKvBudget {
                    prompt_len: 1,
                    generated_count: 1,
                    max_tokens: 1,
                },
                ActiveKvBudget {
                    prompt_len: 1,
                    generated_count: 1,
                    max_tokens: 1,
                },
                ActiveKvBudget {
                    prompt_len: 1,
                    generated_count: 1,
                    max_tokens: 1,
                },
            ],
            4,
            16,
            10,
            10,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 0,
        );

        assert!(outcome.pending.is_empty());
        assert_eq!(ids(&outcome.deferred), vec![1, 2]);
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn active_scheduler_decodes_when_no_pending_request_is_admitted() {
        let outcome = admit_pending_requests(
            vec![pending(1, 16)],
            &[ActiveKvBudget {
                prompt_len: 16,
                generated_count: 1,
                max_tokens: 17,
            }],
            4,
            16,
            1,
            4,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 0,
        );

        assert!(outcome.pending.is_empty());
        assert_eq!(ids(&outcome.deferred), vec![1]);
        assert!(outcome.rejected.is_empty());
        let plan = plan_step(1, crate::DEFAULT_MAX_PREFILL_TOKENS, 64, &[]);
        assert!(
            plan.decode && plan.prefill.is_empty(),
            "active requests keep decoding when pending requests are all deferred"
        );
    }

    #[test]
    fn admission_counts_pending_generation_budget() {
        let outcome = admit_pending_requests(
            vec![
                pending_with_max(1, 16, 17), // 33-token peak -> 3 pages
                pending(2, 16),              // 16-token peak -> 1 page
            ],
            &[],
            8,
            16,
            3,
            8,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 0,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert_eq!(
            ids(&outcome.deferred),
            vec![2],
            "pending request 1 reserves its future decode KV page"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_credits_legal_joint_prefix_pages() {
        // Each request has an 18-page lifetime and shares a 256-token (16-page) prefix.
        let outcome = admit_pending_requests(
            vec![pending_with_max(1, 272, 16), pending_with_max(2, 272, 16)],
            &[],
            2,
            16,
            4,
            18,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 16,
        );

        assert_eq!(ids(&outcome.pending), vec![1, 2]);
        assert!(outcome.deferred.is_empty());
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn active_future_pages_counts_only_remaining_growth() {
        let active = [
            ActiveKvBudget {
                prompt_len: 16,
                generated_count: 1, // current 16 tokens -> 1 page
                max_tokens: 33,     // 49-token lifetime peak -> 4 pages
            },
            ActiveKvBudget {
                prompt_len: 16,
                generated_count: 17, // current 32 tokens -> 2 pages
                max_tokens: 17,      // 33-token lifetime peak -> 3 pages
            },
            ActiveKvBudget {
                prompt_len: 9,
                generated_count: 8, // current 16 tokens -> 1 page
                max_tokens: 24,     // 33-token lifetime peak -> 3 pages
            },
        ];

        assert_eq!(
            active_future_pages(&active, 16),
            6,
            "active admission reserves only future page growth, not pages already held"
        );
    }

    #[test]
    fn active_future_reservation_can_defer_pending_by_page_budget() {
        let active = [ActiveKvBudget {
            prompt_len: 16,
            generated_count: 1, // current 16 tokens -> 1 page
            max_tokens: 49,     // 65-token lifetime peak -> 5 pages; future growth = 4 pages
        }];
        let outcome = admit_pending_requests(
            vec![pending(1, 16), pending(2, 16)],
            &active,
            8,
            16,
            5,
            8,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 0,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert_eq!(
            ids(&outcome.deferred),
            vec![2],
            "slot budget is available, but active future KV growth leaves only one page"
        );
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn admission_rejects_impossible_request_without_blocking_later_fit() {
        let outcome = admit_pending_requests(
            vec![
                pending_with_max(1, 16, 65), // 81-token peak -> 6 pages
                pending(2, 16),
            ],
            &[],
            8,
            16,
            4,
            4,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 0,
        );

        assert_eq!(ids(&outcome.pending), vec![2]);
        assert!(outcome.deferred.is_empty());
        assert_eq!(rejected_ids(&outcome.rejected), vec![1]);
        assert!(
            matches!(outcome.rejected[0].1, RejectReason::KvBudget),
            "over-budget requests should keep the existing KV rejection reason"
        );
    }

    #[test]
    fn admission_allows_request_at_single_request_page_cap() {
        let outcome = admit_pending_requests(
            vec![pending_with_max(1, 16, 48)], // 64-token peak -> 4 pages
            &[],
            1,
            16,
            4,
            4,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 0,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert!(outcome.deferred.is_empty());
        assert!(outcome.rejected.is_empty());
    }

    fn kvbm_peak_pages(prompt_len: usize, max_tokens: usize, page_size: usize) -> usize {
        let pool = BlockPool::new(page_size, 256).expect("test block pool");
        let baseline = pool.available_blocks();
        let mut peak = 0;
        let mut kv = pool.new_request(vec![1; prompt_len], max_tokens, None);

        kv.schedule_prefill(prompt_len, &pool)
            .expect("schedule prefill");
        peak = peak.max(baseline - pool.available_blocks());
        kv.apply_prefill(100, &pool).expect("apply prefill");
        for step in 1..max_tokens {
            kv.schedule_decode(&pool).expect("schedule decode");
            peak = peak.max(baseline - pool.available_blocks());
            kv.apply_decode(100 + step as u32, &pool)
                .expect("apply decode");
        }
        kv.release().expect("release request KV");
        assert_eq!(pool.available_blocks(), baseline);
        peak
    }

    #[test]
    fn lifetime_pages_cover_request_kv_peak_draw() {
        let page_size = 16;
        assert_eq!(request_lifetime_pages(16, 17, page_size), 3);
        assert_eq!(kvbm_peak_pages(16, 17, page_size), 3);

        for prompt_len in 1..=32 {
            for max_tokens in 1..=32 {
                let reserved = request_lifetime_pages(prompt_len, max_tokens, page_size);
                let peak = kvbm_peak_pages(prompt_len, max_tokens, page_size);
                assert!(
                    reserved >= peak,
                    "prompt={prompt_len}, max_tokens={max_tokens}: reserved={reserved}, peak={peak}"
                );
            }
        }
    }

    #[test]
    fn one_token_completion_on_page_boundary_uses_only_prompt_page() {
        assert_eq!(request_lifetime_pages(16, 1, 16), 1);
        let outcome = admit_pending_requests(
            vec![pending_with_max(1, 16, 1)],
            &[],
            1,
            16,
            1,
            1,
            usize::MAX,
            |req| req.prompt_len,
            |req| req.max_tokens,
            |_| 0,
        );

        assert_eq!(ids(&outcome.pending), vec![1]);
        assert!(outcome.deferred.is_empty());
        assert!(outcome.rejected.is_empty());
    }

    #[test]
    fn graph_slot_assignment_stays_dense_after_retirement() {
        assert_eq!(slot_for_new_request(0, 4), Some(0));
        assert_eq!(slot_for_new_request(3, 4), Some(3));
        assert_eq!(slot_for_new_request(4, 4), None);

        assert_eq!(
            compaction_after_retire(4, 1),
            Some(SlotCompaction {
                moved_from: 3,
                moved_to: 1
            }),
            "retiring a middle slot moves the last dense slot into the hole"
        );
        assert_eq!(
            compaction_after_retire(4, 0),
            Some(SlotCompaction {
                moved_from: 3,
                moved_to: 0
            }),
            "retiring the first slot also moves the last dense slot into the hole"
        );
        assert_eq!(
            compaction_after_retire(4, 3),
            None,
            "retiring the last slot does not need a recurrent-state copy"
        );
        assert_eq!(
            slot_for_new_request(3, 4),
            Some(3),
            "after compaction, the next request reuses the next dense slot"
        );
    }

    #[test]
    fn prefilling_future_pages_reserves_only_remaining_growth() {
        // Current 16 tokens use 1 page; the 49-token lifetime peak uses 4.
        let prefilling = [PrefillKvBudget {
            current_tokens: 16,
            prompt_len: 16,
            max_tokens: 33,
        }];
        assert_eq!(prefilling_future_pages(&prefilling, 16), 3);
    }

    #[test]
    fn prefilling_future_pages_sums_across_requests() {
        let prefilling = [
            PrefillKvBudget {
                current_tokens: 0, // just admitted, nothing in KV yet -> 0 pages
                prompt_len: 16,
                max_tokens: 17, // 33-token lifetime peak -> 3 pages
            },
            PrefillKvBudget {
                current_tokens: 32, // 2 pages held
                prompt_len: 40,
                max_tokens: 9, // 49-token lifetime peak -> 4 pages; future = 2
            },
        ];
        assert_eq!(prefilling_future_pages(&prefilling, 16), 5);
    }
}
