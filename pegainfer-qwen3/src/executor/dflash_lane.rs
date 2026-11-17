//! Worker-side DFlash draft lane: the draft model plus per-request draft state.
//!
//! This lives on the worker thread next to the target model because the draft
//! rollout reads the target's embeddings/head and its captured hidden states.
//! The draft/verify boundary stays a pure token span — the target hidden rows
//! are private to this lane and materialize into draft K/V before the scheduler
//! can match the block.

use std::collections::HashMap;

use anyhow::Result;
use cudarc::driver::CudaSlice;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::sampler::SamplingParams;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::LocalQwen3Lane;
use super::PrefillStepItem;
use super::RequestId;
use super::dflash_prefill::dflash_prefill_can_capture;
use super::dflash_prefill::should_capture_dflash_prefill_context;
use crate::dflash::DFlashBatchScratch;
use crate::dflash::DFlashDraftKvPool;
use crate::dflash::DFlashDraftModel;
use crate::dflash::DFlashKvMaterializeScratch;
use crate::dflash::DFlashRequestState;
use crate::speculative::DraftRequestResult;
use crate::speculative::DraftResult;
use crate::speculative::DraftStepItem;
use crate::speculative::VerifyRequestResult;
use crate::speculative::VerifyStepItem;

pub(super) struct DFlashLaneState {
    pub(super) model: DFlashDraftModel,
    requests: HashMap<RequestId, DFlashRequestState>,
    /// Lane-level batched draft scratch, allocated once for the whole decode
    /// batch so the dense draft ops run once instead of once per request.
    pub(super) scratch: DFlashBatchScratch,
    /// Sampled proposals from this round: request -> (scratch row, round seed).
    pub(super) proposal_rows: HashMap<RequestId, (usize, u64)>,
    pub(super) sampling: Option<pegainfer_kernels::ops::SparseVerifyScratch>,
    verified_draft_tokens: usize,
    accepted_draft_tokens: usize,
    /// Hedge chains proposed by the last draft round (`PEGAINFER_SPEC_HEDGE`):
    /// per request, one full `block_size` chain per configured branch position
    /// (`PEGAINFER_SPEC_HEDGE_POSITIONS`, default `0`) — chain `j` follows the
    /// greedy chain to `positions[j]`, takes the exact Markov runner-up there,
    /// and continues greedily. Consumed by the verify step's parallel hedge
    /// expansion; refilled every draft round.
    pub(super) hedge_blocks: HashMap<RequestId, Vec<Vec<u32>>>,
    /// Runtime chain-count controller (`PEGAINFER_SPEC_HEDGE_AUTO`); `None`
    /// = fixed chain count from `PEGAINFER_SPEC_HEDGE_POSITIONS`.
    auto: Option<super::auto_hedge::AutoHedge>,
    /// Chain count the LAST draft round actually ran (0 when the capacity
    /// guard or cap disabled hedging) — the controller books rounds to the
    /// executed configuration, not the requested one.
    round_chains: usize,
    /// Effective hedge geometry, resolved once against this drafter's block
    /// size (`spec_hedge_effective`): the clamped request cap — zero when no
    /// configured position can fire — and the in-block branch positions. The
    /// draft path, the verify dispatch, and the controller all read THESE,
    /// never the raw env values.
    hedge_cap: usize,
    hedge_branch_positions: Vec<usize>,
    /// Draft K/V addressed by the target KV's block IDs. It shares the target
    /// radix cache's lifetime: every registered block was materialized before
    /// it became matchable. `None` with `--no-prefix-cache` (no match can
    /// consume it; the request mirror is still materialized for the draft).
    draft_pool: Option<DFlashDraftKvPool>,
    /// Lane-level scratch for materializing a step's target context.
    materialize: DFlashKvMaterializeScratch,
    /// Target KV page size and pool block count.
    page_size: usize,
    total_blocks: usize,
    /// Locations for gathering a restored prefix out of the pool.
    restore_locs: CudaSlice<i32>,
    restore_locs_host: Vec<i32>,
    /// Bytes the KV reservation set aside for request-owned draft state. Shared
    /// prefix blocks are billed once by the pool, but each request still owns
    /// its own dense draft K/V mirror, so the lane must bound the sum or a
    /// shared-prefix batch allocates far past the reservation.
    draft_state_budget_bytes: usize,
    /// Per-token charge for one request's draft state.
    draft_state_bytes_per_token: usize,
}

impl DFlashLaneState {
    pub(super) fn new(
        ctx: &DeviceContext,
        model: DFlashDraftModel,
        max_decode_batch_size: usize,
        page_size: usize,
        total_blocks: usize,
        max_prefill_tokens: usize,
        prefix_cache: bool,
        draft_state_budget_bytes: usize,
        draft_state_bytes_per_token: usize,
    ) -> Result<Self> {
        let scratch = model.new_batch_scratch(ctx, max_decode_batch_size)?;
        let sampling = model
            .uses_selector()
            .then(|| {
                pegainfer_kernels::ops::SparseVerifyScratch::new(
                    ctx,
                    max_decode_batch_size,
                    model.block_size(),
                )
            })
            .transpose()?;
        let geometry = spec_hedge_effective(model.block_size());
        let hedge_branch_positions = geometry.positions;
        let hedge_cap = if model.uses_markov_head() && !hedge_branch_positions.is_empty() {
            geometry.cap
        } else {
            0
        };
        if hedge_cap > 0 {
            // The parser drops unparsable entries and falls back to position
            // 0, so a benchmark can otherwise measure a narrower ladder than
            // it configured.
            log::info!(
                "spec hedge on: up to {hedge_cap} request(s)/round, branch positions {:?} (block size {})",
                hedge_branch_positions,
                model.block_size()
            );
        }
        let auto = (spec_hedge_auto() && hedge_cap > 0).then(|| {
            let max_chains = hedge_branch_positions.len();
            log::info!("spec hedge auto: self-pricing controller on (C in 0..={max_chains})");
            super::auto_hedge::AutoHedge::new(max_chains)
        });
        let draft_pool = if prefix_cache {
            let pool = model.new_draft_kv_pool(ctx, total_blocks, page_size)?;
            log::info!(
                "Qwen3 DFlash draft KV pool: {} layers x {} slots x {} dim ({} MiB)",
                pool.k.len(),
                pool.slots(),
                model.kv_dim(),
                pool.slots() * model.kv_dim() * 2 * pool.k.len() * 2 / (1024 * 1024)
            );
            Some(pool)
        } else {
            None
        };
        let materialize_rows = max_prefill_tokens.max(model.block_size() + 1);
        let materialize = model.new_materialize_scratch(ctx, materialize_rows)?;
        if draft_state_budget_bytes > 0 {
            log::info!(
                "Qwen3 DFlash draft state budget: {} MiB ({} B/token per request)",
                draft_state_budget_bytes / (1024 * 1024),
                draft_state_bytes_per_token
            );
        }
        let restore_rows = model.max_position_embeddings();
        let restore_locs = ctx.stream.alloc_zeros(restore_rows)?;
        Ok(Self {
            model,
            requests: HashMap::new(),
            scratch,
            proposal_rows: HashMap::new(),
            sampling,
            verified_draft_tokens: 0,
            accepted_draft_tokens: 0,
            hedge_blocks: HashMap::new(),
            auto,
            round_chains: 0,
            hedge_cap,
            hedge_branch_positions,
            draft_pool,
            materialize,
            page_size,
            total_blocks,
            restore_locs,
            restore_locs_host: vec![0; restore_rows],
            draft_state_budget_bytes,
            draft_state_bytes_per_token,
        })
    }

    /// Whether one more request of `max_cache_len` tokens fits under the draft
    /// state budget. A request that does not fit takes the capture-miss path:
    /// it keeps its target KV hit and plain-decodes.
    fn can_create_state(&self, max_cache_len: usize) -> bool {
        if self.draft_state_bytes_per_token == 0 {
            return true;
        }
        let needed = max_cache_len.saturating_mul(self.draft_state_bytes_per_token);
        let used: usize = self
            .requests
            .values()
            .map(|state| {
                state
                    .max_cache_len()
                    .saturating_mul(self.draft_state_bytes_per_token)
            })
            .sum();
        used.saturating_add(needed) <= self.draft_state_budget_bytes
    }

    /// Copy a matched prefix's draft K/V out of the paged pool. The executor
    /// has already checked every page's content tag; an empty page list is the
    /// fallback (target KV hit, no draft state).
    pub(super) fn restore_prefix_captures(
        &mut self,
        ctx: &DeviceContext,
        requests: &[PrefillStepItem],
    ) -> Result<()> {
        let Some(draft_pool) = self.draft_pool.as_ref() else {
            return Ok(());
        };
        for req in requests {
            if req.capture_prefix_pages.is_empty()
                || self.requests.contains_key(&req.request_id)
                || !super::dflash_prefill::prefill_capture_supported(req)
            {
                continue;
            }
            let block = self.page_size;
            let pages = &req.capture_prefix_pages;
            let cached = pages.len() * block;
            anyhow::ensure!(
                cached == req.cached_tokens.unwrap_or(0),
                "DFlash restore pages ({} x {}) disagree with cached_tokens {} for {:?}",
                pages.len(),
                block,
                req.cached_tokens.unwrap_or(0),
                req.request_id
            );
            anyhow::ensure!(
                cached <= self.restore_locs_host.len(),
                "DFlash restore tokens {cached} exceed restore scratch {}",
                self.restore_locs_host.len()
            );
            let max_cache_len =
                (req.prompt_tokens.len() + req.max_output_tokens + self.model.block_size())
                    .min(self.model.max_position_embeddings());
            if !self.can_create_state(max_cache_len) {
                log::debug!(
                    "Qwen3 DFlash draft state budget is full; {:?} keeps its KV hit but will not \
                     draft",
                    req.request_id
                );
                continue;
            }
            for (i, page) in pages.iter().enumerate() {
                anyhow::ensure!(
                    *page >= 0 && (*page as usize) < self.total_blocks,
                    "DFlash restore page {page} out of range {}",
                    self.total_blocks
                );
                for j in 0..block {
                    self.restore_locs_host[i * block + j] = (*page as usize * block + j) as i32;
                }
            }
            let mut locs_dst = self.restore_locs.slice_mut(..cached);
            ctx.stream
                .memcpy_htod(&self.restore_locs_host[..cached], &mut locs_dst)?;
            let mut state = self.model.new_request_state(ctx, max_cache_len)?;
            if let Err(error) = DFlashDraftModel::gather_restored_kv(
                ctx,
                &mut state,
                draft_pool,
                &self.restore_locs,
                cached,
            ) {
                log::warn!(
                    "Qwen3 DFlash prefix draft KV restore failed for {:?}: {error}",
                    req.request_id
                );
                continue;
            }
            self.requests.insert(req.request_id, state);
        }
        Ok(())
    }
}

impl LocalQwen3Lane {
    /// Target layers whose hidden states the draft model consumes (None when
    /// DFlash is not loaded).
    pub(super) fn dflash_capture_layer_ids(&self) -> Option<Vec<usize>> {
        self.dflash
            .as_ref()
            .map(|dflash| dflash.model.target_layer_ids().to_vec())
    }

    pub(super) fn restore_dflash_prefix_captures(
        &mut self,
        requests: &[PrefillStepItem],
    ) -> Result<()> {
        let ctx = self.model.device_ctx().clone();
        if let Some(dflash) = self.dflash.as_mut() {
            dflash.restore_prefix_captures(&ctx, requests)?;
        }
        Ok(())
    }

    pub(super) fn should_capture_dflash_prefill_context(
        &self,
        requests: &[PrefillStepItem],
    ) -> bool {
        let Some(dflash) = self.dflash.as_ref() else {
            return false;
        };
        should_capture_dflash_prefill_context(requests, |request_id| {
            dflash.requests.contains_key(&request_id)
        })
    }

    /// Materialize each eligible request's prefill chunk into draft K/V. The
    /// rows go to both the request mirror and the paged pool at the target's
    /// block slots. Returns the requests that now have a full draft state.
    pub(super) fn record_prefill_dflash_context(
        &mut self,
        requests: &[PrefillStepItem],
        kv_views: &[pegainfer_kv_cache::KvView],
        capture_requested: bool,
        captured_hidden: Option<&HiddenStates>,
    ) -> Result<Vec<RequestId>> {
        let Some(captured_hidden) = captured_hidden else {
            anyhow::ensure!(
                !capture_requested,
                "DFlash prefill context capture was requested but no hidden states were returned"
            );
            return Ok(Vec::new());
        };
        anyhow::ensure!(
            capture_requested,
            "DFlash prefill hidden states were returned without a capture request"
        );
        anyhow::ensure!(
            requests.len() == kv_views.len(),
            "DFlash prefill has {} requests but {} KV views",
            requests.len(),
            kv_views.len()
        );
        let Some(dflash) = self.dflash.as_mut() else {
            anyhow::bail!("DFlash prefill context record requested without DFlash");
        };
        let expected_tokens: usize = requests.iter().map(|req| req.chunk_tokens).sum();
        anyhow::ensure!(
            captured_hidden.seq_len == expected_tokens,
            "DFlash prefill captured {} hidden rows for {} scheduled tokens",
            captured_hidden.seq_len,
            expected_tokens
        );
        let ctx = self.model.device_ctx().clone();
        let mut captured_requests = Vec::new();
        let mut token_offset = 0usize;
        for (i, req) in requests.iter().enumerate() {
            let state_exists = dflash.requests.contains_key(&req.request_id);
            if dflash_prefill_can_capture(req, state_exists) {
                let max_cache_len =
                    (req.prompt_tokens.len() + req.max_output_tokens + dflash.model.block_size())
                        .min(dflash.model.max_position_embeddings());
                let existing = dflash.requests.remove(&req.request_id);
                if existing.is_none() && !dflash.can_create_state(max_cache_len) {
                    log::debug!(
                        "Qwen3 DFlash draft state budget is full; {:?} will plain-decode",
                        req.request_id
                    );
                    token_offset += req.chunk_tokens;
                    continue;
                }
                let mut state = if let Some(state) = existing {
                    state
                } else {
                    anyhow::ensure!(
                        req.chunk_start == 0,
                        "DFlash prefill chunk for {:?} starts at {} without a restored state",
                        req.request_id,
                        req.chunk_start
                    );
                    dflash.model.new_request_state(&ctx, max_cache_len)?
                };
                anyhow::ensure!(
                    state.committed_len == req.chunk_start,
                    "DFlash prefill context for {:?} is discontinuous: committed={}, chunk_start={}",
                    req.request_id,
                    state.committed_len,
                    req.chunk_start
                );
                let start_pos = state.committed_len;
                dflash.model.materialize_context_kv(
                    &ctx,
                    &mut state,
                    captured_hidden,
                    token_offset,
                    req.chunk_tokens,
                    start_pos,
                    kv_views[i].page_indices(),
                    dflash.draft_pool.as_mut(),
                    &mut dflash.materialize,
                )?;
                state.committed_len += req.chunk_tokens;
                dflash.requests.insert(req.request_id, state);
                captured_requests.push(req.request_id);
            } else {
                dflash.requests.remove(&req.request_id);
            }
            token_offset += req.chunk_tokens;
        }
        Ok(captured_requests)
    }

    /// Materialize the accepted verify rows' target hidden into draft K/V and
    /// keep the cumulative acceptance trace at debug level. The verifier
    /// forwarded the anchor plus the accepted drafts; the newly sampled bonus
    /// stays dangling and is materialized by the next step.
    pub(super) fn record_verify_dflash_context(
        &mut self,
        requests: &[VerifyStepItem],
        kv_views: &[pegainfer_kv_cache::KvView],
        results: &[VerifyRequestResult],
        captured_hidden: Option<&HiddenStates>,
        verify_round: u64,
        gated: bool,
    ) -> Result<()> {
        let Some(captured_hidden) = captured_hidden else {
            anyhow::bail!("DFlash verify context capture requested but no hidden states returned");
        };
        anyhow::ensure!(
            requests.len() == kv_views.len(),
            "DFlash verify has {} requests but {} KV views",
            requests.len(),
            kv_views.len()
        );
        let Some(dflash) = self.dflash.as_mut() else {
            anyhow::bail!("DFlash verify context record requested without DFlash");
        };
        anyhow::ensure!(
            requests.len() == results.len(),
            "DFlash verify result count {} does not match request count {}",
            results.len(),
            requests.len()
        );
        let expected_tokens: usize = requests.iter().map(|req| req.token_ids.len()).sum();
        anyhow::ensure!(
            captured_hidden.seq_len == expected_tokens,
            "DFlash verify captured {} hidden rows for {} scheduled tokens",
            captured_hidden.seq_len,
            expected_tokens
        );
        let ctx = self.model.device_ctx().clone();
        let mut token_offset = 0usize;
        for (i, (req, result)) in requests.iter().zip(results).enumerate() {
            anyhow::ensure!(
                req.request_id == result.request_id,
                "DFlash verify result {:?} does not match request {:?}",
                result.request_id,
                req.request_id
            );
            let mut state = dflash.requests.remove(&req.request_id).ok_or_else(|| {
                anyhow::anyhow!("missing DFlash state after verify for {:?}", req.request_id)
            })?;
            let accepted = result.accepted_tokens.len();
            let start_pos = state.committed_len;
            dflash.model.materialize_context_kv(
                &ctx,
                &mut state,
                captured_hidden,
                token_offset,
                accepted,
                start_pos,
                kv_views[i].page_indices(),
                dflash.draft_pool.as_mut(),
                &mut dflash.materialize,
            )?;
            state.committed_len += accepted;
            let committed_len = state.committed_len;
            dflash.requests.insert(req.request_id, state);
            dflash.verified_draft_tokens += req.token_ids.len().saturating_sub(1);
            dflash.accepted_draft_tokens += result.matched_draft_tokens;
            let rate = if dflash.verified_draft_tokens == 0 {
                0.0
            } else {
                dflash.accepted_draft_tokens as f64 / dflash.verified_draft_tokens as f64
            };
            let verified_draft = req.token_ids.len().saturating_sub(1);
            log::debug!(
                "Qwen3 DFlash context round={} request={} accepted_draft={} verified_draft={} gated={} committed_tokens={} context_len={} cumulative_accept_rate={:.3}",
                verify_round,
                req.request_id,
                result.matched_draft_tokens,
                verified_draft,
                u8::from(gated),
                accepted,
                committed_len,
                rate,
            );
            token_offset += req.token_ids.len();
        }
        if let Some(controller) = dflash.auto.as_mut() {
            let committed: usize = results
                .iter()
                .map(|result| result.accepted_tokens.len())
                .sum();
            if gated {
                controller.break_window();
            } else {
                controller.tick(dflash.round_chains, committed);
            }
        }
        Ok(())
    }

    /// Roll out one draft span per request using the checkpoint's proposal head.
    /// Returns the verify span `[current_token, draft_1, …]`.
    pub(super) fn execute_dflash_draft(
        &mut self,
        requests: &[DraftStepItem],
    ) -> Result<DraftResult> {
        anyhow::ensure!(
            !requests.is_empty(),
            "DFlash draft requested without active requests"
        );
        // Take the lane out of `self` so the draft forward (which borrows
        // `dflash.model`/`dflash.scratch`) and the argmax (which borrows
        // `self.sample_scratch`) don't collide on a `self` borrow.
        let Some(mut dflash) = self.dflash.take() else {
            anyhow::bail!("DFlash draft requested but DFlash is not loaded");
        };
        let result = (|| -> Result<Vec<DraftRequestResult>> {
            dflash.proposal_rows.clear();
            // Pull every active request's state out of the map so the batched
            // forward can hold `&mut` to all of them at once. Re-inserted below.
            let mut taken: Vec<(RequestId, DFlashRequestState)> =
                Vec::with_capacity(requests.len());
            for req in requests {
                let state = dflash.requests.remove(&req.request_id).ok_or_else(|| {
                    anyhow::anyhow!("missing DFlash state for {:?}", req.request_id)
                })?;
                taken.push((req.request_id, state));
            }

            let block_size = dflash.model.block_size();
            let current_tokens: Vec<u32> = requests.iter().map(|req| req.current_token).collect();
            let DFlashLaneState {
                model,
                scratch,
                requests: state_map,
                hedge_blocks,
                auto,
                round_chains,
                hedge_cap,
                hedge_branch_positions,
                proposal_rows,
                ..
            } = &mut dflash;
            let mut state_refs: Vec<&mut DFlashRequestState> =
                taken.iter_mut().map(|(_, state)| state).collect();

            // Backbone forward → base block logits in `scratch.logits`. Scoped so
            // the returned borrow of `scratch` ends before the DSpark propose path
            // re-borrows `scratch` mutably for its Markov sample loop.
            let draft_len = {
                let draft_logits = model.draft_logits_batched(
                    &self.model,
                    &mut state_refs,
                    &current_tokens,
                    scratch,
                )?;
                let draft_len = draft_logits.seq_len;
                anyhow::ensure!(
                    draft_len == requests.len() * block_size,
                    "DFlash batched draft produced {} logits rows for {} requests x block {}",
                    draft_len,
                    requests.len(),
                    block_size
                );
                draft_len
            };

            // Plain DFlash uses independent argmax; the selector adds learned
            // pair scores over top-16 candidates. DSpark uses its Markov head
            // with an anchor-first layout (all `block_size` positions).
            let markov = model.uses_markov_head();
            let hedge_positions = if *hedge_cap > 0 {
                hedge_branch_positions.as_slice()
            } else {
                &[]
            };
            // Parallel hedge (PEGAINFER_SPEC_HEDGE): the chain-A loop also
            // captures the exact Markov runner-up at each configured branch
            // position, and one batched ladder pass proposes a chain per
            // (request, position) over the same backbone logits. Stashed
            // lane-side; the draft/verify token-span seam is unchanged (the
            // scheduler never sees hedge chains). Skipped when the ladder
            // would exceed the Markov scratch's max batch.
            hedge_blocks.clear();
            // Auto mode: the controller picks how many of the configured
            // positions to ladder this round. Always a PREFIX — runner
            // stripes are indexed by position order, so prefixes keep the
            // runner pass (always run over the full list, stable graph
            // shape) aligned with the ladder's stripe reads.
            // Batch-adaptive policy (PEGAINFER_SPEC_HEDGE_KNEE) takes precedence
            // over both the fixed count and the auto controller: it sets the
            // chain count from the free zone at THIS round's batch size, so
            // hedging fades to zero as concurrency fills the verify pass.
            let knee = spec_hedge_knee();
            let (ladder_chains, round_cap) = if knee > 0 {
                // Batch-adaptive, bucket-aware: the chain count keeps the
                // expanded verify batch on a CUDA-graph bucket inside the free
                // zone, so it fades to the baseline as concurrency rises. The
                // whole batch hedges at that count (the cap is not the knob).
                let c = batch_adaptive_chains(
                    requests.len(),
                    model.block_size(),
                    knee,
                    hedge_positions.len(),
                    crate::batch_decode_buffers::BATCH_BUCKETS,
                );
                log::debug!(
                    "spec hedge plan: batch {} -> {c} chain(s) ({} verify spans)",
                    requests.len(),
                    requests.len() * (1 + c)
                );
                (c, requests.len())
            } else {
                let c = match auto.as_ref() {
                    Some(controller) => controller.current_c().min(hedge_positions.len()),
                    None => hedge_positions.len(),
                };
                (c, *hedge_cap)
            };
            // Chains are built for the SAME requests the verify side will
            // hedge: eligibility travels on the draft item, and selection is
            // capped by the per-round request cap and the ladder's scratch
            // budget at the chain count that actually runs this round.
            let hedged = select_hedged_requests(
                requests,
                round_cap,
                match ladder_chains {
                    0 => 0,
                    // Slots are bounded by the ladder's Markov scratch AND by
                    // the verify batch's remaining span capacity — chains the
                    // expansion would drop at `max_batch` must not be built.
                    c => (scratch.markov_max_batch() / c).min(
                        (*crate::batch_decode_buffers::BATCH_BUCKETS.last().unwrap())
                            .saturating_sub(requests.len())
                            / c,
                    ),
                },
                hedge_positions.first().copied().unwrap_or(usize::MAX),
                model.block_size(),
            );
            let hedging = ladder_chains > 0 && !hedged.is_empty();
            *round_chains = if hedging { ladder_chains } else { 0 };
            let selector = model.uses_selector();
            let sampled = if selector {
                let mut temperatures = Vec::with_capacity(requests.len());
                let mut uniforms = Vec::with_capacity(requests.len() * (block_size - 1));
                for (row, req) in requests.iter().enumerate() {
                    let greedy = pegainfer_sample::effectively_greedy(
                        &req.params,
                        self.model.config().decodable_vocab,
                    );
                    temperatures.push(if greedy { 0.0 } else { req.params.temperature });
                    if greedy {
                        uniforms.resize(uniforms.len() + block_size - 1, 0.0);
                    } else {
                        // Separate proposal draws from acceptance and correction draws.
                        let mut rng =
                            StdRng::seed_from_u64(pegainfer_sample::mix_seed(req.sample_seed, 0));
                        uniforms.extend((0..block_size - 1).map(|_| rng.random::<f32>()));
                        proposal_rows.insert(req.request_id, (row, req.sample_seed));
                    }
                }
                model.selected_draft_tokens(
                    self.model.device_ctx(),
                    &current_tokens,
                    &temperatures,
                    &uniforms,
                    scratch,
                )?
            } else if markov {
                if hedging {
                    let ctx = self.model.device_ctx();
                    let chain_a = model.markov_draft_with_runners(
                        ctx,
                        &current_tokens,
                        hedge_positions,
                        scratch,
                    )?;
                    let bs = model.block_size();
                    let req_map: Vec<u32> = hedged.iter().map(|&i| i as u32).collect();
                    let ladder = model.markov_chain_ladder(
                        ctx,
                        &chain_a,
                        &req_map,
                        &hedge_positions[..ladder_chains],
                        scratch,
                    )?;
                    let c = ladder_chains;
                    for (slot, &i) in hedged.iter().enumerate() {
                        let chains: Vec<Vec<u32>> = (0..c)
                            .map(|j| {
                                let row = slot * c + j;
                                ladder[row * bs..(row + 1) * bs].to_vec()
                            })
                            .collect();
                        hedge_blocks.insert(requests[i].request_id, chains);
                    }
                    chain_a
                } else {
                    model.markov_draft_tokens(self.model.device_ctx(), &current_tokens, scratch)?
                }
            } else {
                let greedy = SamplingParams::default();
                let params: Vec<&SamplingParams> = vec![&greedy; draft_len];
                self.select_step_tokens(scratch.logits(), &params, 0)?
            };

            // Re-insert every request's state before splitting the result.
            for (request_id, state) in taken {
                state_map.insert(request_id, state);
            }

            // Selector already drops the anchor slot. Legacy proposals still
            // carry full blocks, with the checkpoint-specific layout below.
            let sampled_stride = block_size - usize::from(selector);
            anyhow::ensure!(
                sampled.len() == requests.len() * sampled_stride,
                "DFlash batched draft sampled {} tokens for {} requests x stride {}",
                sampled.len(),
                requests.len(),
                sampled_stride
            );

            // Each request owns sampled_stride entries. Selector output already
            // excludes its anchor; legacy anchor-drop skips position 0 here.
            // Anchor-first checkpoints use every sample. Prepend the actual
            // dangling token once for all three paths before target verification.
            let drafts_start = usize::from(!selector && !model.anchor_first());
            let keep = model.effective_drafts();
            let mut outputs = Vec::with_capacity(requests.len());
            for (i, req) in requests.iter().enumerate() {
                let block = &sampled[i * sampled_stride..(i + 1) * sampled_stride];
                let drafts = &block[drafts_start..];
                anyhow::ensure!(
                    !drafts.is_empty(),
                    "draft block {} produced no draft tokens (block_size {})",
                    i,
                    block_size
                );
                anyhow::ensure!(
                    keep <= drafts.len(),
                    "draft trim {} exceeds the block's {} draft tokens",
                    keep,
                    drafts.len()
                );
                let mut token_ids = Vec::with_capacity(keep + 1);
                token_ids.push(req.current_token);
                token_ids.extend(drafts[..keep].iter().copied());
                outputs.push(DraftRequestResult {
                    request_id: req.request_id,
                    token_ids,
                });
            }
            Ok(outputs)
        })();
        self.dflash = Some(dflash);
        Ok(DraftResult { requests: result? })
    }

    pub(super) fn drop_dflash_request(&mut self, request_id: RequestId) {
        if let Some(dflash) = self.dflash.as_mut() {
            dflash.requests.remove(&request_id);
        }
    }
}

/// `PEGAINFER_SPEC_HEDGE_AUTO=1` puts the hedge chain count under the
/// runtime self-pricing controller ([`crate::executor::auto_hedge`]): the
/// lane explores C ∈ {0..=len(positions)} (position-list prefixes) and
/// commits to the measured tokens/sec argmax. Requires the hedge to be
/// enabled (`PEGAINFER_SPEC_HEDGE` > 0). Read once.
pub(super) fn spec_hedge_auto() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("PEGAINFER_SPEC_HEDGE_AUTO").is_ok_and(|v| v == "1"))
}

/// Effective hedge geometry for a drafter `block_size`: the clamped request
/// cap and the configured branch positions that can actually fire
/// (`p < block_size`). Startup scratch billing, verify-plan sizing, and the
/// draft path all budget from this ONE shape — the sites drifting apart is
/// how positions past the block ended up reserving pages no round could use.
pub(super) struct HedgeGeometry {
    pub(super) cap: usize,
    pub(super) positions: Vec<usize>,
    /// Most hedge chain spans one verify round can carry.
    pub(super) chain_spans: usize,
}

pub(super) fn spec_hedge_effective(block_size: usize) -> HedgeGeometry {
    let positions: Vec<usize> = spec_hedge_positions()
        .iter()
        .copied()
        .filter(|&p| p < block_size)
        .collect();
    let cap = spec_hedge_cap();
    let chain_spans = match spec_hedge_knee() {
        _ if cap == 0 || positions.is_empty() => 0,
        0 => cap.saturating_mul(positions.len()),
        knee => knee_chain_spans(
            block_size,
            knee,
            positions.len(),
            crate::batch_decode_buffers::BATCH_BUCKETS,
        ),
    };
    HedgeGeometry {
        cap,
        positions,
        chain_spans,
    }
}

/// Worst-case scratch KV pages one hedge chain's verify span can touch: the
/// span is at most the anchor plus `block_size` drafts, at an arbitrary page
/// offset. Startup scratch billing and the verify-time span guard both
/// budget from this ONE value; the two sites carrying independent literals
/// is how a large-but-legal block would reserve pages the verify pass then
/// silently refuses to use.
pub(super) fn hedge_pages_per_chain(block_size: usize, page_size: usize) -> usize {
    // Saturating: this bills against raw config dims, where over-estimating
    // is safe but wrapping would under-reserve.
    block_size.div_ceil(page_size.max(1)).saturating_add(1)
}

/// `PEGAINFER_SPEC_HEDGE=<H>` enables the parallel hedge and caps the number
/// of hedged requests per verify round at `H`, unless
/// `PEGAINFER_SPEC_HEDGE_KNEE` plans the round. Each hedged request costs one
/// extra verify span and [`hedge_pages_per_chain`] lane scratch KV pages PER
/// CHAIN, so a request branching at C positions costs C of each.
/// `0`/unset/unparsable = off. Read once.
pub(super) fn spec_hedge_cap() -> usize {
    static CAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("PEGAINFER_SPEC_HEDGE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
            // Scratch pages, plan metadata, and ladder rows all scale with
            // the cap; clamp so a pathological value cannot blow any budget.
            .min(64)
    })
}

/// `PEGAINFER_SPEC_HEDGE_KNEE=<rows>` turns on the batch-adaptive policy: the
/// per-round chain count is chosen so the expanded verify batch stays within
/// the roofline free zone of `<rows>` rows — the compute knee for this model
/// and GPU, below which the verify pass is weight-bandwidth-bound and extra
/// rows are nearly free. `0`/unset = off, keeping the fixed or auto chain
/// count. Read once.
pub(super) fn spec_hedge_knee() -> usize {
    static KNEE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *KNEE.get_or_init(|| {
        std::env::var("PEGAINFER_SPEC_HEDGE_KNEE")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0)
    })
}

/// Uniform hedge chain count for this batch, chosen to keep the expanded
/// verify batch BOTH inside the roofline free zone AND on a CUDA-graph bucket.
///
/// Every hedged request is verified over `1 + C` spans, so the batch spans
/// `batch * (1 + C)`. Two constraints: the rows must stay under the compute
/// knee (`batch * (1 + C) * span <= knee_rows`, below which extra rows are
/// nearly free), and the span count must be a captured graph bucket — a count
/// that falls between buckets runs eager and gives back the hedge gain. Return
/// the largest `C <= max_chains` satisfying both, or `0` when no bucket-aligned
/// count fits, which is how the policy fades to the baseline as concurrency
/// fills the zone.
///
/// A fractional free-zone remainder (a batch that fits, say, four extra chain
/// spans across eight requests) is deliberately NOT spent: the only span
/// counts between one bucket and the next are off-bucket, so partial hedging
/// runs eager and loses more than it gains.
pub(super) fn batch_adaptive_chains(
    batch: usize,
    block_size: usize,
    knee_rows: usize,
    max_chains: usize,
    buckets: &[usize],
) -> usize {
    if batch == 0 || knee_rows == 0 || max_chains == 0 {
        return 0;
    }
    let span = block_size + 1;
    let mut best = 0;
    for c in 1..=max_chains {
        let spans = batch * (1 + c);
        if spans * span > knee_rows {
            break; // past the free zone; larger C only goes further past
        }
        if buckets.contains(&spans) {
            best = c;
        }
    }
    best
}

/// Most hedge chain spans one round can carry under the knee policy, which
/// hedges the whole batch at the chain count [`batch_adaptive_chains`] picks:
/// the planner's own `batch * C` maximum over every batch size.
fn knee_chain_spans(
    block_size: usize,
    knee_rows: usize,
    max_chains: usize,
    buckets: &[usize],
) -> usize {
    (1..=buckets.last().copied().unwrap_or(0))
        .map(|batch| {
            batch * batch_adaptive_chains(batch, block_size, knee_rows, max_chains, buckets)
        })
        .max()
        .unwrap_or(0)
}

/// `PEGAINFER_SPEC_HEDGE_POSITIONS="0,1,2"` — draft positions at which hedge
/// chains branch (each takes the exact Markov runner-up at that position,
/// rank 2). Sorted/deduped; default `0` (the original single-chain hedge).
/// Read once.
pub(super) fn spec_hedge_positions() -> &'static [usize] {
    static POSITIONS: std::sync::OnceLock<Vec<usize>> = std::sync::OnceLock::new();
    POSITIONS.get_or_init(|| {
        let mut positions: Vec<usize> = std::env::var("PEGAINFER_SPEC_HEDGE_POSITIONS")
            .ok()
            .map(|v| {
                v.split(',')
                    .filter_map(|t| t.trim().parse::<usize>().ok())
                    .collect()
            })
            .unwrap_or_default();
        positions.sort_unstable();
        positions.dedup();
        positions.truncate(16);
        if positions.is_empty() {
            positions.push(0);
        }
        positions
    })
}

impl DFlashLaneState {
    /// Verify-side correction for the controller's books: the round fell back
    /// to the plain pass (no expanded spans), so it executed as C = 0
    /// whatever the draft side prepared.
    pub(super) fn clear_round_chains(&mut self) {
        self.round_chains = 0;
    }

    /// Effective hedged-request cap (0 = hedging cannot run on this drafter
    /// and the verify dispatch must not try).
    pub(super) fn hedge_cap(&self) -> usize {
        self.hedge_cap
    }

    /// Number of in-block branch positions (the maximum chain count).
    pub(super) fn hedge_branch_count(&self) -> usize {
        self.hedge_branch_positions.len()
    }
}

/// First-come selection of hedge-eligible requests under the per-round cap
/// and the ladder's scratch budget (both in requests). A request qualifies
/// only when its chains can SURVIVE the verify-side span clamp: after the
/// clamp it keeps `min(block_size, budget - 1)` drafts, and the earliest
/// branch position must land inside them — otherwise every chain collapses
/// onto chain A and is deduplicated away, wasting the slot. Draft and verify
/// pick from this same list, so neither a sampled request nor a
/// budget-starved greedy one ahead in the batch can burn a hedge slot the
/// verify side then refuses.
fn select_hedged_requests(
    requests: &[DraftStepItem],
    cap: usize,
    slots: usize,
    first_position: usize,
    block_size: usize,
) -> Vec<usize> {
    requests
        .iter()
        .enumerate()
        .filter(|(_, r)| r.hedge_budget >= 2 && first_position < block_size.min(r.hedge_budget - 1))
        .map(|(i, _)| i)
        .take(cap.min(slots))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(id: u64, budget: usize) -> DraftStepItem {
        DraftStepItem::new(RequestId::new(id), 7, budget, SamplingParams::default(), 0)
    }

    const BLOCK: usize = 7;

    #[test]
    fn batch_adaptive_chains_picks_the_largest_bucket_aligned_count_under_the_knee() {
        // block 7 -> span 8 rows; knee 96 rows = 12 spans.
        let bk = crate::batch_decode_buffers::BATCH_BUCKETS;
        let c = |b| batch_adaptive_chains(b, BLOCK, 96, 7, bk);
        // B=1: 1*(1+C) must be a bucket <= 12 -> {2,4,8}; largest gives C=7 (8 spans).
        assert_eq!(c(1), 7);
        // B=2: 2*(1+C) in buckets <= 12 -> {2,4,8}; 8 -> C=3, NOT the free-zone C=5.
        assert_eq!(c(2), 3);
        // B=4: 4*(1+C) in buckets <= 12 -> 8 -> C=1 (12 would be off-bucket).
        assert_eq!(c(4), 1);
        // B=8: 8*(1+C): C=0 gives 8 (baseline), C=1 gives 16 = past the knee.
        // 9..15 spans are all off-bucket -> no hedge fits, fade to baseline.
        assert_eq!(c(8), 0);
        assert_eq!(c(16), 0); // past the knee outright
        // every chosen count lands on a bucket and inside the free zone
        for b in 1..=32 {
            let cc = c(b);
            if cc > 0 {
                let spans = b * (1 + cc);
                assert!(
                    bk.contains(&spans),
                    "b={b} c={cc}: {spans} spans off-bucket"
                );
                assert!(spans * (BLOCK + 1) <= 96, "b={b} c={cc}: past the knee");
            }
        }
        // off / degenerate inputs
        assert_eq!(batch_adaptive_chains(1, BLOCK, 0, 7, bk), 0);
        assert_eq!(batch_adaptive_chains(0, BLOCK, 96, 7, bk), 0);
        assert_eq!(batch_adaptive_chains(1, BLOCK, 96, 0, bk), 0);
    }

    #[test]
    fn knee_chain_spans_is_the_planners_maximum() {
        let bk = crate::batch_decode_buffers::BATCH_BUCKETS;
        // block 7, knee 96 rows = 12 spans. One position: only batches 1, 2
        // and 4 land on a bucket with a chain each, so at most 4 chain spans.
        assert_eq!(knee_chain_spans(BLOCK, 96, 1, bk), 4);
        // Three positions: batch 2 at 3 chains (8 spans) carries 6.
        assert_eq!(knee_chain_spans(BLOCK, 96, 3, bk), 6);
        // Seven positions: batch 1 at 7 chains carries 7.
        assert_eq!(knee_chain_spans(BLOCK, 96, 7, bk), 7);
        assert_eq!(knee_chain_spans(BLOCK, 0, 7, bk), 0);
    }

    #[test]
    fn hedge_selection_is_order_invariant_for_mixed_batches() {
        // H=1: the single slot must land on the greedy request in BOTH
        // orders — a sampled request (budget 0) must never consume it.
        let sampled_first = [item(1, 0), item(2, 64)];
        let greedy_first = [item(2, 64), item(1, 0)];
        assert_eq!(
            select_hedged_requests(&sampled_first, 1, usize::MAX, 0, BLOCK),
            [1]
        );
        assert_eq!(
            select_hedged_requests(&greedy_first, 1, usize::MAX, 0, BLOCK),
            [0]
        );
    }

    #[test]
    fn hedge_selection_skips_spans_the_clamp_would_collapse() {
        // Branch position 1 with remaining budget 2: the clamp keeps one
        // draft, the branch cannot show, the chain would dedupe onto A. The
        // slot must go to the request whose span can carry the branch — in
        // BOTH orders.
        let starved_first = [item(1, 2), item(2, 64)];
        let healthy_first = [item(2, 64), item(1, 2)];
        assert_eq!(
            select_hedged_requests(&starved_first, 1, usize::MAX, 1, BLOCK),
            [1]
        );
        assert_eq!(
            select_hedged_requests(&healthy_first, 1, usize::MAX, 1, BLOCK),
            [0]
        );
        // The same budget IS enough for a position-0 branch.
        assert_eq!(
            select_hedged_requests(&starved_first, 2, usize::MAX, 0, BLOCK),
            [0, 1]
        );
    }

    #[test]
    fn hedge_selection_respects_cap_and_slots() {
        let batch = [item(1, 64), item(2, 0), item(3, 64), item(4, 64)];
        assert_eq!(
            select_hedged_requests(&batch, 2, usize::MAX, 0, BLOCK),
            [0, 2]
        );
        assert_eq!(select_hedged_requests(&batch, usize::MAX, 1, 0, BLOCK), [0]);
        assert!(select_hedged_requests(&batch, 0, usize::MAX, 0, BLOCK).is_empty());
    }
}
