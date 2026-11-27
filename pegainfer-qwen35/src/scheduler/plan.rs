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

/// Linear per-step cost fit used to bound the prefill rows of one step.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepCostModel {
    pub base_us: u64,
    pub us_per_prefill_row: u64,
    pub us_per_decode_row: u64,
}

impl StepCostModel {
    /// H20 Qwen3.5-4B `PEGAINFER_ITL_DEBUG` fit: n=1497 unified steps,
    /// R^2=0.9972.
    pub const H20_QWEN35_4B: Self = Self {
        base_us: 8_200,
        us_per_prefill_row: 74,
        us_per_decode_row: 106,
    };

    pub const fn predict_step_us(&self, prefill_rows: usize, active_decodes: usize) -> u64 {
        self.base_us
            .saturating_add(self.us_per_prefill_row.saturating_mul(prefill_rows as u64))
            .saturating_add(self.us_per_decode_row.saturating_mul(active_decodes as u64))
    }

    /// Fastest mixed step the model can represent: one prefill row alongside one
    /// active decoder.
    pub const fn min_reachable_stall_us(&self) -> u64 {
        self.predict_step_us(1, 1)
    }

    /// Prompt rows that keep the fitted step cost at or below `max_stall_us`.
    ///
    /// Launch validation rejects a target below [`Self::min_reachable_stall_us`],
    /// so one row fits a one-decoder mix. The keepalive floor still matters for
    /// a much wider active batch: the fixed decode cost can exceed the target,
    /// and the single row keeps the prefill queue from starving.
    pub fn max_prefill_rows_for_stall(&self, max_stall_us: u64, active_decodes: usize) -> usize {
        assert!(
            self.us_per_prefill_row > 0,
            "step cost model requires positive prefill slope"
        );
        let fixed_us = self
            .base_us
            .saturating_add(self.us_per_decode_row.saturating_mul(active_decodes as u64));
        let headroom_us = max_stall_us.saturating_sub(fixed_us);
        usize::try_from((headroom_us / self.us_per_prefill_row).max(1)).unwrap_or(usize::MAX)
    }
}

/// Queue pressure visible to the step planner.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct StepLoadSignals {
    pub(super) pending_reqs: usize,
    pub(super) queued_prefill_tokens: usize,
}

impl StepLoadSignals {
    pub(super) fn from_queue(pending_reqs: usize, prefilling: &[PrefillQueueState]) -> Self {
        Self {
            pending_reqs,
            queued_prefill_tokens: prefilling.iter().fold(0usize, |total, state| {
                total.saturating_add(state.remaining_tokens)
            }),
        }
    }
}

/// Which prefill row policy a scheduler instance applies.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum StepPolicy {
    /// Pack FIFO prompt slices into the configured fixed token budget.
    #[default]
    Fixed,
    /// Bound prefill rows by a fitted per-step cost target.
    StallBounded {
        max_stall_us: u64,
        cost_model: StepCostModel,
    },
}

impl StepPolicy {
    pub const fn stall_bounded_ms(max_stall_ms: u64) -> Self {
        Self::StallBounded {
            max_stall_us: max_stall_ms.saturating_mul(1_000),
            cost_model: StepCostModel::H20_QWEN35_4B,
        }
    }
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

/// Pack FIFO prompt slices into `remaining_budget`; decode rows do not consume
/// it. A prompt crossing the budget resumes at the queue front on the next step.
pub(super) fn plan_prefill_rows(
    active_decodes: usize,
    prefilling: &[PrefillQueueState],
    remaining_budget: usize,
    load: StepLoadSignals,
    policy: StepPolicy,
) -> Vec<usize> {
    if remaining_budget == 0 || prefilling.is_empty() {
        return Vec::new();
    }
    match policy {
        StepPolicy::Fixed => pack_fifo(prefilling, remaining_budget),
        StepPolicy::StallBounded {
            max_stall_us,
            cost_model,
        } => {
            let stall_cap = cost_model
                .max_prefill_rows_for_stall(max_stall_us, active_decodes)
                .min(remaining_budget);
            if active_decodes == 0 {
                // No active decoder shares this step, so a long prompt may use
                // the whole budget instead of being sliced by the stall cap.
                let first_rem = prefilling[0].remaining_tokens;
                let first = first_rem.min(remaining_budget);
                if first == 0 {
                    return Vec::new();
                }
                if first < first_rem {
                    return vec![first];
                }
                pack_whole_followers(first, &prefilling[1..], stall_cap)
            } else {
                let step_cap = stall_cap;
                let first_rem = prefilling[0].remaining_tokens;
                if first_rem == 0 {
                    return Vec::new();
                }
                if first_rem <= step_cap {
                    return pack_whole_followers(first_rem, &prefilling[1..], step_cap);
                }
                let has_backlog = prefilling.len() > 1
                    || load.pending_reqs > 0
                    || load.queued_prefill_tokens > first_rem;
                let tail = first_rem % step_cap;
                let next_fits_with_tail = tail > 0
                    && prefilling.get(1).is_some_and(|next| {
                        next.remaining_tokens > 0
                            && tail.saturating_add(next.remaining_tokens) <= step_cap
                    });
                if has_backlog && next_fits_with_tail {
                    // Finish the tail in the next step beside the whole follower,
                    // so this step takes a full slice instead of a fragment.
                    return vec![step_cap];
                }
                // Same number of steps as repeated capped slices, lower peak ITL.
                let chunks = first_rem.div_ceil(step_cap);
                vec![first_rem.div_ceil(chunks)]
            }
        }
    }
}

/// Pack the head plus only those FIFO followers that fit whole in `row_cap`.
///
/// A follower that cannot fit is left for the next step; slicing it here would
/// add a prefill request launch for a tiny fragment.
fn pack_whole_followers(
    first_rows: usize,
    followers: &[PrefillQueueState],
    row_cap: usize,
) -> Vec<usize> {
    let mut prefill = vec![first_rows];
    let mut total = first_rows;
    for state in followers {
        if state.remaining_tokens == 0 {
            break;
        }
        if total.saturating_add(state.remaining_tokens) <= row_cap {
            total = total.saturating_add(state.remaining_tokens);
            prefill.push(state.remaining_tokens);
        } else {
            break;
        }
    }
    prefill
}

/// FIFO prompt slices under a hard row budget.
fn pack_fifo(prefilling: &[PrefillQueueState], budget: usize) -> Vec<usize> {
    let mut remaining = budget;
    let mut prefill = Vec::new();
    for state in prefilling {
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
    prefill
}

/// Pack FIFO prompt slices into `budget`; decode rows do not consume it.
/// A prompt crossing the budget resumes at the queue front on the next step.
pub(super) fn plan_step(
    active_decodes: usize,
    budget: usize,
    prefilling: &[PrefillQueueState],
    load: StepLoadSignals,
    policy: StepPolicy,
) -> StepPlan {
    assert!(
        budget > 0,
        "Qwen3.5 scheduler requires a positive prefill budget"
    );
    let prefill = plan_prefill_rows(active_decodes, prefilling, budget, load, policy);
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

    /// Build a policy whose fitted prefill row cap is exactly `step_cap` for
    /// `active_decodes`.
    fn stall_policy_for_cap(step_cap: usize, active_decodes: usize) -> StepPolicy {
        let cost_model = StepCostModel::H20_QWEN35_4B;
        let max_stall_us = cost_model
            .base_us
            .saturating_add(
                cost_model
                    .us_per_decode_row
                    .saturating_mul(active_decodes as u64),
            )
            .saturating_add(
                cost_model
                    .us_per_prefill_row
                    .saturating_mul(step_cap as u64),
            );
        StepPolicy::StallBounded {
            max_stall_us,
            cost_model,
        }
    }

    #[test]
    fn default_budget_is_one_standard_prompt_chunk() {
        let plan = plan_step(
            1,
            crate::DEFAULT_MAX_PREFILL_TOKENS,
            &four_standard_prompts(),
            StepLoadSignals::default(),
            StepPolicy::Fixed,
        );
        assert!(plan.decode);
        assert_eq!(plan.prefill, vec![STANDARD_PROMPT_CHUNK]);
    }

    const PACK_BUDGET: usize = STANDARD_PROMPT_CHUNK * 4;

    #[test]
    fn plan_packs_whole_prompts_in_fifo_order() {
        let plan = plan_step(
            1,
            PACK_BUDGET,
            &prompts(&[300, 700, 900, 1500]),
            StepLoadSignals::default(),
            StepPolicy::Fixed,
        );
        assert!(plan.decode);
        assert_eq!(plan.prefill, vec![300, 700, 900, 1500]);
    }

    #[test]
    fn plan_slices_the_prompt_that_crosses_the_budget() {
        let plan = plan_step(
            1,
            PACK_BUDGET,
            &prompts(&[6000]),
            StepLoadSignals::default(),
            StepPolicy::Fixed,
        );
        assert_eq!(plan.prefill, vec![PACK_BUDGET]);
    }

    #[test]
    fn plan_slices_against_the_budget_left_by_earlier_prompts() {
        let plan = plan_step(
            1,
            PACK_BUDGET,
            &prompts(&[2000, 3000, 1]),
            StepLoadSignals::default(),
            StepPolicy::Fixed,
        );
        assert_eq!(plan.prefill, vec![2000, 2096]);
    }

    #[test]
    fn plan_resumes_the_sliced_tail_before_the_next_prompt() {
        // The 6000-token prompt was sliced at 4096; its tail is now the front
        // of the queue and must be finished before the next prompt moves.
        let plan = plan_step(
            1,
            PACK_BUDGET,
            &prompts(&[1904, 1024]),
            StepLoadSignals::default(),
            StepPolicy::Fixed,
        );
        assert_eq!(plan.prefill, vec![1904, 1024]);
    }

    #[test]
    fn plan_packs_everything_under_a_huge_budget() {
        let plan = plan_step(
            1,
            usize::MAX,
            &prompts(&[STANDARD_PROMPT_CHUNK, STANDARD_PROMPT_CHUNK, 7]),
            StepLoadSignals::default(),
            StepPolicy::Fixed,
        );
        assert_eq!(
            plan.prefill,
            vec![STANDARD_PROMPT_CHUNK, STANDARD_PROMPT_CHUNK, 7]
        );
    }

    #[test]
    fn plan_never_widens_a_smaller_budget() {
        let plan = plan_step(
            1,
            512,
            &prompts(&[STANDARD_PROMPT_CHUNK; 4]),
            StepLoadSignals::default(),
            StepPolicy::Fixed,
        );
        assert_eq!(plan.prefill, vec![512]);
    }

    #[test]
    fn plan_breaks_on_a_finished_prompt() {
        let plan = plan_step(
            1,
            crate::DEFAULT_MAX_PREFILL_TOKENS,
            &prompts(&[0, STANDARD_PROMPT_CHUNK]),
            StepLoadSignals::default(),
            StepPolicy::Fixed,
        );
        assert!(plan.prefill.is_empty());
        assert!(plan.decode);
        assert!(!plan.is_empty());
    }

    #[test]
    fn plan_is_prefill_only_without_decode_work() {
        let plan = plan_step(
            0,
            crate::DEFAULT_MAX_PREFILL_TOKENS,
            &prompts(&[700]),
            StepLoadSignals::default(),
            StepPolicy::Fixed,
        );
        assert!(!plan.decode);
        assert_eq!(plan.prefill, vec![700]);
        assert!(!plan.is_empty());
    }

    #[test]
    fn plan_is_decode_only_without_prompt_work() {
        let plan = plan_step(
            1,
            crate::DEFAULT_MAX_PREFILL_TOKENS,
            &[],
            StepLoadSignals::default(),
            StepPolicy::Fixed,
        );
        assert!(plan.decode);
        assert!(plan.prefill.is_empty());
        assert!(!plan.is_empty());
    }

    #[test]
    fn plan_is_empty_without_prompt_or_decode_work() {
        let plan = plan_step(
            0,
            crate::DEFAULT_MAX_PREFILL_TOKENS,
            &[],
            StepLoadSignals::default(),
            StepPolicy::Fixed,
        );
        assert!(plan.is_empty());
    }

    #[test]
    #[should_panic(expected = "positive prefill budget")]
    fn plan_rejects_a_zero_budget() {
        plan_step(1, 0, &[], StepLoadSignals::default(), StepPolicy::Fixed);
    }

    #[test]
    fn h20_cost_model_predicts_and_saturates() {
        let model = StepCostModel::H20_QWEN35_4B;
        assert_eq!(model.predict_step_us(0, 0), 8_200);
        assert_eq!(
            model.predict_step_us(1_024, 32),
            8_200 + 74 * 1_024 + 106 * 32
        );
        assert_eq!(model.min_reachable_stall_us(), 8_380);
        assert_eq!(model.min_reachable_stall_us(), model.predict_step_us(1, 1));
        assert_eq!(model.predict_step_us(usize::MAX, usize::MAX), u64::MAX);
    }

    #[test]
    fn h20_cost_model_rows_shrink_with_active_decodes() {
        let model = StepCostModel::H20_QWEN35_4B;
        assert_eq!(model.max_prefill_rows_for_stall(60_000, 0), 700);
        assert_eq!(model.max_prefill_rows_for_stall(60_000, 1), 698);
        assert_eq!(
            model.max_prefill_rows_for_stall(0, 0),
            1,
            "an exhausted budget still admits one row"
        );
    }

    #[test]
    #[should_panic(expected = "positive prefill slope")]
    fn cost_model_rejects_a_zero_prefill_slope() {
        let model = StepCostModel {
            base_us: 0,
            us_per_prefill_row: 0,
            us_per_decode_row: 0,
        };
        model.max_prefill_rows_for_stall(60_000, 0);
    }

    #[test]
    fn stall_bounded_policy_converts_milliseconds_saturating() {
        assert_eq!(
            StepPolicy::stall_bounded_ms(60),
            StepPolicy::StallBounded {
                max_stall_us: 60_000,
                cost_model: StepCostModel::H20_QWEN35_4B,
            }
        );
        assert_eq!(
            StepPolicy::stall_bounded_ms(0),
            StepPolicy::StallBounded {
                max_stall_us: 0,
                cost_model: StepCostModel::H20_QWEN35_4B,
            }
        );
        assert_eq!(
            StepPolicy::stall_bounded_ms(u64::MAX),
            StepPolicy::StallBounded {
                max_stall_us: u64::MAX,
                cost_model: StepCostModel::H20_QWEN35_4B,
            }
        );
    }

    #[test]
    fn step_load_signals_sum_queue_tokens_saturating() {
        assert_eq!(
            StepLoadSignals::from_queue(3, &prompts(&[300, 700])),
            StepLoadSignals {
                pending_reqs: 3,
                queued_prefill_tokens: 1_000,
            }
        );
        assert_eq!(
            StepLoadSignals::from_queue(
                0,
                &[
                    PrefillQueueState {
                        remaining_tokens: usize::MAX,
                    },
                    PrefillQueueState {
                        remaining_tokens: 1,
                    },
                ],
            )
            .queued_prefill_tokens,
            usize::MAX
        );
    }

    #[test]
    fn stall_bounded_returns_no_rows_for_an_empty_queue() {
        let policy = StepPolicy::stall_bounded_ms(60);
        for active_decodes in [0, 1, 8] {
            assert!(
                plan_prefill_rows(
                    active_decodes,
                    &[],
                    1_024,
                    StepLoadSignals::default(),
                    policy,
                )
                .is_empty()
            );
        }
    }

    #[test]
    fn stall_bounded_zero_budget_yields_no_rows() {
        let policy = StepPolicy::stall_bounded_ms(60);
        assert!(
            plan_prefill_rows(0, &prompts(&[1_024]), 0, StepLoadSignals::default(), policy)
                .is_empty()
        );
        assert!(
            plan_prefill_rows(3, &prompts(&[1_024]), 0, StepLoadSignals::default(), policy)
                .is_empty()
        );
        assert!(
            plan_prefill_rows(
                0,
                &prompts(&[1_024]),
                0,
                StepLoadSignals::default(),
                StepPolicy::Fixed,
            )
            .is_empty()
        );
    }

    #[test]
    fn stall_bounded_empty_plan_without_prompt_or_decode_work() {
        let plan = plan_step(
            0,
            1_024,
            &[],
            StepLoadSignals::default(),
            StepPolicy::stall_bounded_ms(60),
        );
        assert!(plan.is_empty());
    }

    #[test]
    fn stall_bounded_plan_step_keeps_prefill_only_and_mixed_marks() {
        let pure = plan_step(
            0,
            2_048,
            &prompts(&[3_000]),
            StepLoadSignals::default(),
            StepPolicy::stall_bounded_ms(60),
        );
        assert!(!pure.decode);
        assert_eq!(pure.prefill, vec![2_048]);

        let queue = prompts(&[1_024]);
        let mixed = plan_step(
            2,
            4_096,
            &queue,
            StepLoadSignals::from_queue(0, &queue),
            StepPolicy::stall_bounded_ms(60),
        );
        assert!(mixed.decode);
        assert_eq!(mixed.prefill, vec![512]);
    }

    #[test]
    #[should_panic(expected = "positive prefill budget")]
    fn stall_bounded_plan_step_rejects_a_zero_budget() {
        plan_step(
            1,
            0,
            &prompts(&[1_024]),
            StepLoadSignals::default(),
            StepPolicy::stall_bounded_ms(60),
        );
    }

    #[test]
    fn stall_bounded_decode_only_step_keeps_decoding() {
        let plan = plan_step(
            4,
            1_024,
            &[],
            StepLoadSignals {
                pending_reqs: 3,
                queued_prefill_tokens: 0,
            },
            StepPolicy::stall_bounded_ms(60),
        );
        assert!(plan.decode);
        assert!(plan.prefill.is_empty());
    }

    #[test]
    fn stall_bounded_pure_prefill_uses_the_whole_budget_for_one_long_prompt() {
        let rows = plan_prefill_rows(
            0,
            &prompts(&[3_000]),
            2_048,
            StepLoadSignals::default(),
            StepPolicy::stall_bounded_ms(60),
        );
        assert_eq!(rows, vec![2_048]);
    }

    #[test]
    fn stall_bounded_pure_prefill_does_not_pack_a_1024_prompt_burst() {
        let rows = plan_prefill_rows(
            0,
            &prompts(&[1_024, 1_024, 1_024]),
            4_096,
            StepLoadSignals::default(),
            StepPolicy::stall_bounded_ms(60),
        );
        assert_eq!(
            rows,
            vec![1_024],
            "a second 1024-row prompt must not delay the first prompt's TTFT"
        );
    }

    #[test]
    fn stall_bounded_pure_prefill_packs_short_prompts_within_the_stall_cap() {
        let rows = plan_prefill_rows(
            0,
            &prompts(&[300, 200, 100, 100]),
            4_096,
            StepLoadSignals::default(),
            StepPolicy::stall_bounded_ms(60),
        );
        assert_eq!(rows, vec![300, 200, 100, 100]);
        assert_eq!(rows.iter().sum::<usize>(), 700);
    }

    #[test]
    fn stall_bounded_pure_prefill_does_not_slice_a_whole_follower() {
        let rows = plan_prefill_rows(
            0,
            &prompts(&[300, 500]),
            4_096,
            StepLoadSignals::default(),
            StepPolicy::stall_bounded_ms(60),
        );
        assert_eq!(
            rows,
            vec![300],
            "a follower that does not fit whole waits for the next step"
        );
    }

    #[test]
    fn stall_bounded_pure_prefill_slicing_takes_the_hard_budget() {
        let rows = plan_prefill_rows(
            0,
            &prompts(&[3_000, 100]),
            500,
            StepLoadSignals::default(),
            StepPolicy::stall_bounded_ms(60),
        );
        assert_eq!(
            rows,
            vec![500],
            "the hard budget caps the slice and the unfinished prompt blocks the follower"
        );
    }

    #[test]
    fn stall_bounded_mixed_step_slices_balanced_without_backlog() {
        let policy = StepPolicy::stall_bounded_ms(60);
        let rows = plan_prefill_rows(
            1,
            &prompts(&[1_024]),
            4_096,
            StepLoadSignals::from_queue(0, &prompts(&[1_024])),
            policy,
        );
        assert_eq!(rows, vec![512]);
        assert!(
            StepCostModel::H20_QWEN35_4B.predict_step_us(rows[0], 1) <= 60_000,
            "the balanced slice must respect the stall target"
        );
    }

    #[test]
    fn stall_bounded_mixed_step_respects_a_budget_below_the_stall_cap() {
        let rows = plan_prefill_rows(
            1,
            &prompts(&[3_000]),
            300,
            StepLoadSignals::default(),
            StepPolicy::stall_bounded_ms(60),
        );
        assert_eq!(rows, vec![300]);
    }

    #[test]
    fn stall_bounded_mixed_step_does_not_slice_a_whole_follower() {
        let policy = stall_policy_for_cap(1_062, 1);
        let queue = prompts(&[1_024, 1_024]);
        let rows = plan_prefill_rows(
            1,
            &queue,
            4_096,
            StepLoadSignals::from_queue(0, &queue),
            policy,
        );
        assert_eq!(
            rows,
            vec![1_024],
            "must not emit a [1024, 38] follower fragment"
        );
    }

    #[test]
    fn stall_bounded_mixed_step_packs_a_whole_follower_when_it_fits() {
        let policy = stall_policy_for_cap(2_100, 1);
        let queue = prompts(&[1_024, 1_024]);
        let rows = plan_prefill_rows(
            1,
            &queue,
            4_096,
            StepLoadSignals::from_queue(0, &queue),
            policy,
        );
        assert_eq!(rows, vec![1_024, 1_024]);
    }

    #[test]
    fn stall_bounded_mixed_step_full_slices_to_merge_the_tail_with_a_whole_follower() {
        let policy = stall_policy_for_cap(1_024, 1);
        let target_us = StepCostModel::H20_QWEN35_4B.predict_step_us(1_024, 1);
        let queue = prompts(&[1_500, 500]);
        let first = plan_prefill_rows(
            1,
            &queue,
            4_096,
            StepLoadSignals::from_queue(0, &queue),
            policy,
        );
        assert_eq!(first, vec![1_024]);

        let remaining = prompts(&[476, 500]);
        let second = plan_prefill_rows(
            1,
            &remaining,
            4_096,
            StepLoadSignals::from_queue(0, &remaining),
            policy,
        );
        assert_eq!(second, vec![476, 500]);
        assert!(
            StepCostModel::H20_QWEN35_4B.predict_step_us(second.iter().sum::<usize>(), 1)
                <= target_us
        );
    }

    #[test]
    fn stall_bounded_mixed_step_balances_without_slicing_the_follower() {
        let policy = stall_policy_for_cap(550, 1);
        let queue = prompts(&[1_024, 1_024]);
        let first = plan_prefill_rows(
            1,
            &queue,
            4_096,
            StepLoadSignals::from_queue(0, &queue),
            policy,
        );
        assert_eq!(first, vec![512]);

        let remaining = prompts(&[512, 1_024]);
        let second = plan_prefill_rows(
            1,
            &remaining,
            4_096,
            StepLoadSignals::from_queue(0, &remaining),
            policy,
        );
        assert_eq!(
            second,
            vec![512],
            "the tail step stays whole and never emits a follower fragment"
        );
    }

    #[test]
    fn stall_bounded_mixed_step_packs_whole_short_prompts() {
        let policy = stall_policy_for_cap(700, 1);
        let queue = prompts(&[256, 256, 512]);
        let rows = plan_prefill_rows(
            1,
            &queue,
            4_096,
            StepLoadSignals::from_queue(0, &queue),
            policy,
        );
        assert_eq!(rows, vec![256, 256]);
    }

    #[test]
    fn stall_bounded_mixed_step_balances_when_pending_cannot_supply_a_mergeable_tail() {
        let policy = StepPolicy::stall_bounded_ms(60);
        let rows = plan_prefill_rows(
            1,
            &prompts(&[1_024]),
            4_096,
            StepLoadSignals {
                pending_reqs: 1,
                queued_prefill_tokens: 1_024,
            },
            policy,
        );
        assert_eq!(
            rows,
            vec![512],
            "a pending request is not a follower that can absorb the sliced tail"
        );
    }

    #[test]
    fn stall_bounded_mixed_step_shrinks_continuously_with_active_decodes() {
        let policy = StepPolicy::stall_bounded_ms(60);
        let model = StepCostModel::H20_QWEN35_4B;
        let queue = prompts(&[4_096, 4_096]);
        let mut previous_cap = usize::MAX;
        for active_decodes in 1..=200 {
            let cap = model
                .max_prefill_rows_for_stall(60_000, active_decodes)
                .min(4_096);
            let rows = plan_prefill_rows(
                active_decodes,
                &queue,
                4_096,
                StepLoadSignals::from_queue(0, &queue),
                policy,
            );
            let total = rows.iter().sum::<usize>();
            assert!(total >= 1);
            assert!(
                total <= cap,
                "active_decodes={active_decodes} planned {total} rows above cap {cap}"
            );
            assert!(
                model.predict_step_us(total, active_decodes) <= 60_000,
                "active_decodes={active_decodes} planned rows exceed the stall target"
            );
            assert!(
                cap <= previous_cap,
                "the row cap must not widen as active decodes grow"
            );
            assert!(
                previous_cap == usize::MAX || previous_cap - cap <= 2,
                "active_decodes={active_decodes} moved the cap by more than one decode row step"
            );
            previous_cap = cap;
        }
    }

    #[test]
    fn stall_bounded_keeps_one_row_under_an_extreme_bound() {
        let policy = StepPolicy::StallBounded {
            max_stall_us: 1,
            cost_model: StepCostModel::H20_QWEN35_4B,
        };
        let single = plan_prefill_rows(
            64,
            &prompts(&[1_024]),
            4_096,
            StepLoadSignals::default(),
            policy,
        );
        assert_eq!(single, vec![1]);
        let backlogged = plan_prefill_rows(
            64,
            &prompts(&[1_024]),
            4_096,
            StepLoadSignals {
                pending_reqs: 1,
                queued_prefill_tokens: 1_024,
            },
            policy,
        );
        assert_eq!(backlogged, vec![1]);
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
        let plan = plan_step(
            1,
            crate::DEFAULT_MAX_PREFILL_TOKENS,
            &[],
            StepLoadSignals::default(),
            StepPolicy::Fixed,
        );
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
