//! Scheduling mechanics: admission, KV-prefetch parking, chunked-prefill
//! batching, and plan selection for the dedicated GPU thread.
//!
//! This module is engine-contract-free: it owns queues of [`PendingRequest`]s
//! and produces pure [`effects::StepEffects`] data. The serve loop itself
//! belongs to the contract's driver, and everything that touches the engine
//! contract (tickets, emitter calls, the `Scheduler` trait) lives in
//! [`crate::frontend_adapter`] — read that file to see how Qwen3 plugs into
//! the frontend.

pub(crate) mod effects;
pub(crate) mod phase_trace;
pub(crate) mod plan;
pub(crate) mod resolve;

use std::collections::HashSet;

use log::debug;
use log::warn;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::sampler::SamplingParams;

use crate::executor::ModelExecutor;
use crate::executor::RequestId;

// ── Internal types ──────────────────────────────────────────────────────

/// An in-flight request being decoded.
pub(crate) struct ActiveRequestState {
    pub(crate) request_id: RequestId,
    pub(crate) lora_adapter: Option<String>,
    pub(crate) last_token: u32,
    pub(crate) generated_count: usize,
    pub(crate) max_tokens: usize,
    pub(crate) prompt_len: usize,
    pub(crate) params: SamplingParams,
    /// Number of top logprobs to return (0 = disabled).
    pub(crate) logprobs: usize,
}

/// A request between intake and promotion, `Clone` because the decode-overlap
/// path keeps a copy for async-prefill poll resolution. Its engine-contract
/// handle lives in the adapter's registry, keyed by `request_id` — never here.
#[derive(Clone)]
pub(crate) struct PendingRequest {
    pub(crate) request_id: RequestId,
    pub(crate) lora_adapter: Option<String>,
    pub(crate) prompt_tokens: Vec<u32>,
    pub(crate) params: SamplingParams,
    pub(crate) max_tokens: usize,
    pub(crate) logprobs: usize,
    pub(crate) echo: bool,
    /// Whether this request has already been offered to async KV prefetch.
    /// Offered at most once; a no-hit offer leaves the request in the normal
    /// admission flow with this set so it isn't re-probed every tick.
    pub(crate) prefetch_offered: bool,
    /// Prompt tokens whose KV is already computed (prefix-cache hits plus
    /// chunks applied in earlier steps). Updated from the executor's
    /// authoritative position after every chunk.
    pub(crate) prefill_pos: usize,
    /// Prompt tokens to forward in the upcoming step. Set by
    /// `take_prefill_chunks` when the request is packed into a step.
    pub(crate) step_chunk: usize,
    /// Prefix-cache hits reported by the first chunk, carried across later
    /// chunks so the final result still reports them truthfully.
    pub(crate) cached_tokens: usize,
}

impl PendingRequest {
    pub(crate) fn from_request(request_id: RequestId, req: Request) -> Self {
        Self {
            request_id,
            lora_adapter: req.lora_adapter,
            prompt_tokens: req.prompt_tokens,
            params: req.params,
            max_tokens: req.max_tokens,
            logprobs: req.logprobs,
            echo: req.echo,
            prefetch_offered: false,
            prefill_pos: 0,
            step_chunk: 0,
            cached_tokens: 0,
        }
    }

    fn remaining_prompt_tokens(&self) -> usize {
        self.prompt_tokens.len() - self.prefill_pos
    }
}

/// Pull the next prefill step set off the front of `prefilling`, capping the
/// step's total forwarded tokens at `max_prefill_tokens`. Each taken request
/// gets its per-step chunk recorded in `step_chunk`. Echo requests need
/// logits for every prompt position in one forward, so they only run when
/// their whole remainder fits the profiled prefill bound. Under request-local
/// chunking, a request takes `min(remaining, max_prefill_tokens)` whole or skips
/// the step, so its chunk boundaries depend only on its own length and are
/// batch-invariant.
pub(crate) fn take_prefill_chunks(
    prefilling: &mut Vec<PendingRequest>,
    max_prefill_tokens: usize,
    request_local: bool,
) -> Vec<PendingRequest> {
    let mut budget = max_prefill_tokens;
    let mut taken: Vec<PendingRequest> = Vec::new();
    let mut i = 0;
    while i < prefilling.len() && budget > 0 {
        let remaining = prefilling[i].remaining_prompt_tokens();
        let chunk = if prefilling[i].echo {
            if remaining > budget {
                i += 1;
                continue;
            }
            remaining
        } else if request_local {
            let desired = remaining.min(max_prefill_tokens);
            if desired > budget {
                i += 1;
                continue;
            }
            desired
        } else {
            remaining.min(budget)
        };
        let mut req = prefilling.remove(i);
        req.step_chunk = chunk;
        budget = budget.saturating_sub(chunk);
        taken.push(req);
    }
    // Whole-or-skip may select later requests; sort to match request-id-ordered results.
    taken.sort_by_key(|req| req.request_id);
    taken
}

pub(crate) fn servable_len(max_context: usize, max_blocks: usize, block_size: usize) -> u32 {
    max_context
        .min(max_blocks.saturating_mul(block_size))
        .try_into()
        .unwrap_or(u32::MAX)
}

// ── KV-offload prefetch admission helpers ────────────────────────────────

/// Move requests whose async CPU-tier prefetch just settled from `loading`
/// back to the front of `deferred` — their KV is hot, so admit them first.
pub(crate) fn reclaim_ready_prefetch<E: ModelExecutor>(
    executor: &mut E,
    deferred: &mut Vec<PendingRequest>,
    loading: &mut Vec<PendingRequest>,
    // Free blocks already promised to admitted requests; a remote fetch that
    // resolves during this sweep must not reserve into them.
    reserve_floor: usize,
) {
    promote_ready(
        executor.drain_ready_prefetch(reserve_floor),
        deferred,
        loading,
    );
}

/// Offer each not-yet-offered `deferred` request to async CPU-tier prefetch,
/// moving the ones that start loading out of `deferred` into `loading`. A
/// request that doesn't start a load (pure GPU hit, miss, or block pressure)
/// stays in `deferred`, flagged so it isn't re-probed next tick.
///
/// Echo requests are never offered: their prefill forwards the whole prompt to
/// recover prompt logprobs and so skips `match_and_add_prefix` (see
/// `execute_prefill`). Prefetched blocks would never be matched/reused — they
/// would only park restored KV that admission credits but prefill can't spend,
/// starving the request under tight budgets. Leaving `prefetch_offered` unset
/// for echo is harmless: the `!req.echo` guard keeps them from being probed.
pub(crate) fn offer_prefetch<E: ModelExecutor>(
    executor: &mut E,
    deferred: &mut Vec<PendingRequest>,
    loading: &mut Vec<PendingRequest>,
    // Free blocks already promised to admitted requests; the prefetch must
    // leave them untouched (see `ModelExecutor::begin_kv_prefetch`).
    reserve_floor: usize,
) {
    let mut keep = Vec::with_capacity(deferred.len());
    for mut req in deferred.drain(..) {
        if !req.prefetch_offered && !req.echo {
            req.prefetch_offered = true;
            if executor.begin_kv_prefetch(
                req.request_id,
                &req.prompt_tokens,
                req.lora_adapter.as_deref(),
                reserve_floor,
            ) {
                loading.push(req);
                continue;
            }
        }
        keep.push(req);
    }
    *deferred = keep;
}

/// Block until at least one in-flight prefetch settles, then promote the
/// settled requests to `deferred`. Called only when the scheduler is otherwise
/// idle, so blocking on the DMA costs nothing.
pub(crate) fn block_on_loading<E: ModelExecutor>(
    executor: &mut E,
    deferred: &mut Vec<PendingRequest>,
    loading: &mut Vec<PendingRequest>,
    reserve_floor: usize,
) {
    promote_ready(
        executor.wait_ready_prefetch(reserve_floor),
        deferred,
        loading,
    );
}

fn promote_ready(
    ready: Vec<RequestId>,
    deferred: &mut Vec<PendingRequest>,
    loading: &mut Vec<PendingRequest>,
) {
    for id in ready {
        if let Some(pos) = loading.iter().position(|p| p.request_id == id) {
            deferred.insert(0, loading.remove(pos));
        }
    }
}

/// Release any executor-side state a request accumulated before it was rejected
/// at admission. A rejected request never prefills, so the only state it can
/// hold is a settled KV prefetch — committed prefix blocks parked in the
/// executor while the request waited in `deferred`. Without this they would
/// leak (blocks pinned, map entry stranded) for the engine's lifetime. Idempotent
/// and harmless for requests that were never prefetched.
pub(crate) fn release_rejected<E: ModelExecutor>(
    executor: &mut E,
    tracker: &mut phase_trace::PhaseTracker,
    req: &PendingRequest,
) {
    // Close the request's queue span and drop its tracker entry; a rejected
    // request never reaches prefill/decode, so this is its only cleanup point.
    tracker.finish(req.request_id);
    if let Err(e) = executor.drop_request(req.request_id) {
        warn!(
            "failed to release state for rejected {:?}: {e}",
            req.request_id
        );
    }
}

// ── Admission ───────────────────────────────────────────────────────────

/// Why a request was rejected at admission, so the client gets an accurate error.
#[derive(Clone, Copy)]
pub(crate) enum RejectReason {
    /// Worst-case length exceeds the model's position-encoding window.
    ContextLength { limit: usize },
    /// Echo needs all-position logits in one forward, so it must fit the
    /// profiled prefill bound.
    EchoPrefillTokens { limit: usize },
    /// Worst-case length needs more KV blocks than this instance can ever provide.
    KvBudget,
}

pub(crate) struct AdmissionOutcome {
    pub(crate) pending: Vec<PendingRequest>,
    pub(crate) deferred: Vec<PendingRequest>,
    pub(crate) rejected: Vec<(PendingRequest, RejectReason)>,
}

pub(crate) struct LoraValidationOutcome {
    pub(crate) accepted: Vec<PendingRequest>,
    pub(crate) rejected: Vec<PendingRequest>,
}

pub(crate) fn reject_unknown_lora_requests(
    deferred: Vec<PendingRequest>,
    executor: &impl ModelExecutor,
) -> LoraValidationOutcome {
    if !deferred.iter().any(|req| req.lora_adapter.is_some()) {
        return LoraValidationOutcome {
            accepted: deferred,
            rejected: Vec::new(),
        };
    }

    let loaded_lora_adapters = executor.list_lora_adapters();
    let loaded_lora_adapters: HashSet<_> = loaded_lora_adapters.into_iter().collect();
    let mut accepted = Vec::new();
    let mut rejected = Vec::new();

    for req in deferred {
        match req.lora_adapter.as_ref() {
            Some(adapter) if !loaded_lora_adapters.contains(adapter) => rejected.push(req),
            _ => accepted.push(req),
        }
    }

    LoraValidationOutcome { accepted, rejected }
}

fn blocks_needed(token_count: usize, block_size: usize) -> usize {
    token_count.div_ceil(block_size)
}

// Prefill samples the first output token but does not write its KV. A generated
// token's KV is written only when it is fed as the next decode input. Therefore
// N returned completion tokens occupy at most N - 1 generated-token KV slots.
pub(crate) fn max_request_tokens(req: &PendingRequest) -> usize {
    req.prompt_tokens
        .len()
        .saturating_add(req.max_tokens.saturating_sub(1))
}

#[cfg(test)]
fn max_active_tokens(req: &ActiveRequestState) -> usize {
    req.prompt_len
        .saturating_add(req.max_tokens.saturating_sub(1))
}

fn current_active_tokens(req: &ActiveRequestState) -> usize {
    req.prompt_len
        .saturating_add(req.generated_count.saturating_sub(1))
}

// Pool blocks a request can draw over its lifetime. One-token completions
// finish after prefill, so schedule_decode never provisions a dangling block.
// Multi-token requests can draw that final dangling decode block, so admission
// reserves prompt + max_tokens for them.
fn request_lifetime_blocks(prompt_len: usize, max_tokens: usize, block_size: usize) -> usize {
    let lifetime_tokens = if max_tokens <= 1 {
        prompt_len
    } else {
        prompt_len.saturating_add(max_tokens)
    };
    lifetime_tokens.div_ceil(block_size).max(1)
}

fn pending_lifetime_blocks(req: &PendingRequest, block_size: usize) -> usize {
    request_lifetime_blocks(req.prompt_tokens.len(), req.max_tokens, block_size)
}

fn active_lifetime_blocks(req: &ActiveRequestState, block_size: usize) -> usize {
    request_lifetime_blocks(req.prompt_len, req.max_tokens, block_size)
}

fn active_future_blocks(active: &[ActiveRequestState], block_size: usize) -> usize {
    active
        .iter()
        .map(|req| {
            active_lifetime_blocks(req, block_size)
                .saturating_sub(blocks_needed(current_active_tokens(req), block_size))
        })
        .sum()
}

fn echo_exceeds_prefill_bound(req: &PendingRequest, max_prefill_tokens: usize) -> bool {
    req.echo && req.prompt_tokens.len() > max_prefill_tokens
}

/// Free blocks already promised to admitted requests (active decode growth +
/// remaining prefill chunks). A KV prefetch reservation must stay out of this
/// floor or a later chunk/decode fails allocation and kills the whole step.
pub(crate) fn admitted_future_blocks<E: ModelExecutor>(
    executor: &E,
    active: &[ActiveRequestState],
    prefilling: &[PendingRequest],
) -> usize {
    let block_size = executor.block_size();
    active_future_blocks(active, block_size)
        + prefilling_future_blocks(prefilling, block_size, |id| executor.prefetched_blocks(id))
}

fn prefilling_future_blocks(
    prefilling: &[PendingRequest],
    block_size: usize,
    // Blocks a request already holds via a settled prefetch (zero once its
    // first chunk absorbs them). They are out of the free pool, so counting
    // them as future need would double-charge the budget.
    prefetch_credit: impl Fn(RequestId) -> usize,
) -> usize {
    prefilling
        .iter()
        .map(|req| {
            pending_lifetime_blocks(req, block_size)
                .saturating_sub(blocks_needed(req.prefill_pos, block_size))
                .saturating_sub(prefetch_credit(req.request_id))
        })
        .sum()
}

/// Default for `max_prefill_tokens`: prompt tokens forwarded in a single step
/// (chunked prefill). Prefill activation scratch scales with the step's total
/// prompt tokens (~22 KB/token measured on Qwen3-4B), so an unbounded prefill
/// batch can eat the post-KV-pool VRAM headroom and OOM mid-serving under a
/// request burst. Prompts longer than the budget are split across steps, so
/// long prompts can't monopolize a step and starve running decodes.
/// Echo requests need all-position logits in one forward and are rejected when
/// their prompt exceeds this bound.
///
/// A unified step's duration scales with its prefill tokens, and every decode
/// request in the batch stalls for the whole step — the budget bounds that
/// stall. 1024 halves ITL p99 vs 2048 at mid-load with the same mean TPOT;
/// 512 chunks no longer amortize the per-step fixed cost, so prefill falls
/// behind arrivals and TTFT queues up.
pub const DEFAULT_MAX_PREFILL_TOKENS: usize = 1024;

#[allow(clippy::too_many_arguments)]
pub(crate) fn admit_deferred_requests(
    deferred: Vec<PendingRequest>,
    active: &[ActiveRequestState],
    // Admitted requests still mid-prefill: they hold KV for their applied
    // chunks and will take a decode slot when they promote, so admission
    // must reserve both or completing chunks can overshoot capacity.
    prefilling: &[PendingRequest],
    block_size: usize,
    available_blocks: usize,
    max_request_blocks: usize,
    max_context_tokens: usize,
    max_decode_batch_size: usize,
    max_prefill_tokens: usize,
    // Blocks a request already holds from a settled prefetch. These are already
    // out of `available_blocks`, so they must be credited against the request's
    // need or admission double-counts them and can wedge a near-budget CPU-hit
    // request forever (never admitted, prefetch never released).
    prefetch_credit: impl Fn(RequestId) -> usize,
) -> AdmissionOutcome {
    let mut budget = available_blocks
        .saturating_sub(active_future_blocks(active, block_size))
        .saturating_sub(prefilling_future_blocks(
            prefilling,
            block_size,
            &prefetch_credit,
        ));
    let mut decode_slots = max_decode_batch_size
        .saturating_sub(active.len())
        .saturating_sub(prefilling.len());
    let mut pending = Vec::new();
    let mut still_deferred = Vec::new();
    let mut rejected = Vec::new();

    for req in deferred {
        // Reject if the full sequence overflows the position-encoding window
        if req.prompt_tokens.len().saturating_add(req.max_tokens) > max_context_tokens {
            rejected.push((
                req,
                RejectReason::ContextLength {
                    limit: max_context_tokens,
                },
            ));
            continue;
        }

        if echo_exceeds_prefill_bound(&req, max_prefill_tokens) {
            rejected.push((
                req,
                RejectReason::EchoPrefillTokens {
                    limit: max_prefill_tokens,
                },
            ));
            continue;
        }

        // Full physical footprint gates the per-request cap (a request occupies
        // all of it, prefetched or not)…
        let footprint = pending_lifetime_blocks(&req, block_size);
        if footprint > max_request_blocks {
            rejected.push((req, RejectReason::KvBudget));
            continue;
        }

        // …but only the blocks not already held by this request's prefetch must
        // come from the free-pool budget.
        let fresh_needed = footprint.saturating_sub(prefetch_credit(req.request_id));
        if fresh_needed <= budget && decode_slots > 0 {
            budget -= fresh_needed;
            decode_slots -= 1;
            debug!(
                "request admitted: request_id={:?} prompt_len={} max_tokens={}",
                req.request_id,
                req.prompt_tokens.len(),
                req.max_tokens
            );
            pending.push(req);
        } else {
            still_deferred.push(req);
        }
    }

    AdmissionOutcome {
        pending,
        deferred: still_deferred,
        rejected,
    }
}

/// Choose the step plan, preferring a speculative-decode step when the whole
/// active batch is draft-ready. Prefill of new arrivals still takes priority —
/// a speculative step only runs when there is nothing to prefill, so the two
/// never mix in one step.
pub(crate) fn runtime_plan(
    executor: &impl ModelExecutor,
    active: &[ActiveRequestState],
    pending: Vec<PendingRequest>,
) -> Option<plan::ExecutionPlan> {
    if plan::should_speculative_decode(executor, active) {
        if pending.is_empty() {
            Some(plan::ExecutionPlan::SpeculativeDecode)
        } else {
            Some(plan::ExecutionPlan::Prefill { pending })
        }
    } else {
        plan::build_next_plan(!active.is_empty(), pending, executor.speculative_enabled())
    }
}

/// The requests a plan's execution failure kills: the batch it was about to
/// touch. Computed before execution so the ids survive the error path.
pub(crate) fn failure_target_ids(
    active: &[ActiveRequestState],
    plan: &plan::ExecutionPlan,
) -> Vec<RequestId> {
    let mut targets = Vec::new();
    match plan {
        plan::ExecutionPlan::Prefill { pending } => {
            targets.extend(pending.iter().map(|req| req.request_id));
        }
        plan::ExecutionPlan::Decode | plan::ExecutionPlan::SpeculativeDecode => {
            targets.extend(active.iter().map(|req| req.request_id));
        }
        plan::ExecutionPlan::Unified { pending } => {
            targets.extend(active.iter().map(|req| req.request_id));
            targets.extend(pending.iter().map(|req| req.request_id));
        }
    }
    targets
}

#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;
