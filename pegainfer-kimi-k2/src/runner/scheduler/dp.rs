use anyhow::Result;
use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use crossbeam_channel::bounded;
use log::error;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::StopPolicy;
use pegainfer_frontend::engine::SubmittedRequest;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_frontend::engine::TokenSink;
use pegainfer_kv_cache::BlockPool;
use pegainfer_kv_cache::RequestKv;
use rand::rngs::StdRng;
use tokio::sync::mpsc;

use super::emit_sampled_token;
use super::lifecycle::preflight_prefill_candidate;
use super::lifecycle::request_lifetime_blocks;
use super::lifecycle::send_scheduled;
use super::lifecycle::validate_kv_capacity;
use super::row_options;
use crate::runner::executor::DP_MAX_BATCH_PER_RANK;
use crate::runner::executor::ForwardExecutor;
use crate::runner::load_balancer::DpLoadBalancer;
use crate::runner::moe_deepep::DEEPEP_MAX_DISPATCH_TOKENS;
use crate::runner::worker::KimiKvStepPages;
use crate::runner::worker::KimiOneTokenForwardReport;
use crate::runner::worker::KimiRowOptions;

/// Stable per-rank decode arena capacity. Logical slot IDs are arena rows, so
/// every TP1 DP8 decode/prefill command must keep this capacity stable.
const MAX_BATCH_PER_DP: usize = DP_MAX_BATCH_PER_RANK;

/// Coordinated DP engine: one coordinator thread drives all DP ranks in
/// lock-step. Every decode step, ALL ranks execute forward simultaneously
/// (active ranks with real tokens, idle ranks with padding). This satisfies
/// the DeepEP contract that requires all ranks to participate in every
/// MoE layer's dispatch/combine collective.
pub(in crate::runner) struct DpCoordinator {
    dp_world: usize,
    ranks: Vec<DpRankState>,
    /// One logical block pool per rank, mirroring that rank's physical
    /// worker KV pool (same page count and page size).
    pools: Vec<BlockPool>,
    executors: Vec<Box<dyn ForwardExecutor + Send>>,
    step_txs: Vec<Sender<StepCommand>>,
    result_rxs: Vec<Receiver<StepResult>>,
    stop_token_ids: Vec<u32>,
    /// Drives non-greedy sampling: every step command carries a fresh philox
    /// seed per rank (rows within a rank decorrelate through the philox
    /// subsequence; ranks must not share a seed or same-index rows across
    /// ranks would draw the same uniform).
    rng: StdRng,
}

pub(in crate::runner) struct DpRankState {
    slots: Vec<Option<RequestState>>,
}

struct RequestState {
    token_tx: TokenSink,
    stop_policy: StopPolicy,
    prompt_len: usize,
    completion_tokens: usize,
    max_tokens: usize,
    last_token: u32,
    options: KimiRowOptions,
    /// Pool pages backing this request's KV; dropping the state (retire,
    /// disconnect, failure) releases them back to the rank's pool.
    kv: RequestKv,
}

impl RequestState {
    /// Blocks this request may still pull from the pool: peak lifetime draw
    /// (`request_lifetime_blocks`: prompt + max_tokens, because kvbm
    /// provisions the final dangling token's block even though its KV is
    /// never written) minus a lower bound on blocks already drawn
    /// (`ceil((prompt + completion − 1)/bs)` — the KV written so far; kvbm
    /// always holds at least that). Computed from request fields, not kvbm
    /// block state — the qwen3 #85 pattern.
    fn future_blocks(&self, block_size: usize) -> usize {
        let lifetime_tokens = self.prompt_len + self.max_tokens;
        let current_tokens = self.prompt_len + self.completion_tokens.saturating_sub(1);
        lifetime_tokens
            .div_ceil(block_size)
            .saturating_sub(current_tokens.div_ceil(block_size))
    }
}

struct DecodeAdmission {
    slot: usize,
    req: GenerateRequest,
    /// Created at admission with the single prompt-token page already
    /// scheduled; applied once the admission decode step reports back.
    kv: RequestKv,
}

struct DecodeInput {
    token_id: u32,
    append_position: usize,
    slot: usize,
    options: KimiRowOptions,
    pages: Vec<i32>,
}

enum DecodeBatchRow {
    Active(DecodeInput),
    Admission(Box<DecodeAdmission>),
}

impl DecodeBatchRow {
    fn token_id(&self) -> u32 {
        match self {
            Self::Active(input) => input.token_id,
            Self::Admission(admission) => admission.req.prompt_tokens[0],
        }
    }

    fn append_position(&self) -> usize {
        match self {
            Self::Active(input) => input.append_position,
            Self::Admission(_) => 0,
        }
    }

    fn slot(&self) -> usize {
        match self {
            Self::Active(input) => input.slot,
            Self::Admission(admission) => admission.slot,
        }
    }

    fn options(&self) -> KimiRowOptions {
        match self {
            Self::Active(input) => input.options,
            Self::Admission(admission) => row_options(&admission.req),
        }
    }

    fn pages(&self) -> Vec<i32> {
        match self {
            Self::Active(input) => input.pages.clone(),
            Self::Admission(admission) => admission.kv.step_page_indices(1),
        }
    }
}

enum StepCommand {
    Decode {
        token_ids: Vec<u32>,
        positions: Vec<usize>,
        slots: Vec<usize>,
        decode_batch_size: usize,
        kv_pages: KimiKvStepPages,
        rows: Vec<KimiRowOptions>,
        seed: u64,
    },
    Prefill {
        input_ids: Vec<u32>,
        slot: usize,
        decode_batch_size: usize,
        cached_tokens: usize,
        ep_max_seq_len: usize,
        kv_pages: KimiKvStepPages,
        row: KimiRowOptions,
        seed: u64,
    },
    Shutdown,
}

enum StepResult {
    Decode(Result<Vec<KimiOneTokenForwardReport>>),
    Prefill(Result<KimiOneTokenForwardReport>),
}

impl DpCoordinator {
    pub(in crate::runner) fn new(
        executors: Vec<Box<dyn ForwardExecutor + Send>>,
        stop_token_ids: Vec<u32>,
        seed: u64,
        pools: Vec<BlockPool>,
    ) -> Self {
        let dp_world = executors.len();
        assert_eq!(
            pools.len(),
            dp_world,
            "Kimi-K2 DP coordinator needs one KV pool per rank"
        );
        let mut ranks = Vec::with_capacity(dp_world);
        for _ in 0..dp_world {
            ranks.push(DpRankState {
                slots: (0..MAX_BATCH_PER_DP).map(|_| None).collect(),
            });
        }

        Self {
            dp_world,
            ranks,
            pools,
            executors,
            step_txs: Vec::new(),
            result_rxs: Vec::new(),
            stop_token_ids,
            rng: rand::SeedableRng::seed_from_u64(seed),
        }
    }

    /// Blocks the rank's pool can still promise to new requests: free blocks
    /// minus what active requests may still allocate over their lifetimes
    /// (full-lifetime reservation, the qwen3 #85 pattern — admitted requests
    /// can never run out of pages mid-decode).
    fn rank_kv_budget(&self, dp_rank: usize) -> usize {
        let block_size = self.pools[dp_rank].block_size();
        let future: usize = self.ranks[dp_rank]
            .slots
            .iter()
            .flatten()
            .map(|state| state.future_blocks(block_size))
            .sum();
        self.pools[dp_rank]
            .available_blocks()
            .saturating_sub(future)
    }

    fn next_step_seed(&mut self) -> u64 {
        rand::RngExt::random(&mut self.rng)
    }

    /// Spawn per-rank forward threads and run the coordinated decode loop.
    /// This consumes self and blocks until shutdown.
    pub(in crate::runner) fn run(
        mut self,
        mut submit_rx: mpsc::UnboundedReceiver<SubmittedRequest>,
        lb: DpLoadBalancer,
    ) {
        let mut step_txs = Vec::with_capacity(self.dp_world);
        let mut result_rxs = Vec::with_capacity(self.dp_world);
        let mut handles = Vec::with_capacity(self.dp_world);

        for (dp_rank, executor) in self.executors.drain(..).enumerate() {
            let (cmd_tx, cmd_rx) = bounded::<StepCommand>(1);
            let (res_tx, res_rx) = bounded::<StepResult>(1);
            step_txs.push(cmd_tx);
            result_rxs.push(res_rx);

            let handle = std::thread::Builder::new()
                .name(format!("kimi-k2-dp-fwd-{dp_rank}"))
                .spawn(move || {
                    rank_forward_loop(executor, cmd_rx, res_tx);
                })
                .expect("failed to spawn DP rank forward thread");
            handles.push(handle);
        }

        self.step_txs = step_txs;
        self.result_rxs = result_rxs;

        let mut pending_reqs: Vec<GenerateRequest> = Vec::new();

        loop {
            // 1. Drain new requests from submit channel
            if self.global_active_count() == 0 && pending_reqs.is_empty() {
                match submit_rx.blocking_recv() {
                    Some((req, _kv_prefix)) => pending_reqs.push(req),
                    None => break,
                }
            }
            while let Ok((req, _kv_prefix)) = submit_rx.try_recv() {
                pending_reqs.push(req);
            }

            // 2. Admit pending requests to DP ranks via load balancer
            self.admit_pending_requests(&mut pending_reqs, lb);

            // 3. Run one synchronized step across ALL ranks
            if self.global_active_count() > 0 {
                self.synchronized_decode_step();
            }
        }

        // Shutdown all forward threads
        for tx in &self.step_txs {
            let _ = tx.send(StepCommand::Shutdown);
        }
        for handle in handles {
            let _ = handle.join();
        }
    }

    fn global_active_count(&self) -> usize {
        self.ranks.iter().map(DpRankState::active_count).sum()
    }

    fn admit_pending_requests(
        &mut self,
        pending_reqs: &mut Vec<GenerateRequest>,
        lb: DpLoadBalancer,
    ) {
        let mut still_pending = Vec::new();
        let mut decode_admissions = self.empty_decode_admissions();
        let mut reserved_free_slots = self.free_slot_lists();
        // Future blocks promised to this round's queued (not yet installed)
        // decode admissions, per rank.
        let mut round_reserved = vec![0usize; self.dp_world];

        for req in pending_reqs.drain(..) {
            let Some(req) = preflight_prefill_candidate(req) else {
                continue;
            };

            // Honor-or-reject (#239): requests that can never fit are
            // rejected here; requests that merely don't fit *now* wait.
            if let Err(message) = validate_kv_capacity(
                &req,
                self.pools[0].block_size(),
                self.pools[0].max_request_blocks(),
                Some(DEEPEP_MAX_DISPATCH_TOKENS),
            ) {
                send_scheduled(&req);
                let _ = req.token_tx.send(TokenEvent::Rejected {
                    message,
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
                continue;
            }
            let blocks_needed = request_lifetime_blocks(&req, self.pools[0].block_size());

            if req.prompt_tokens.len() == 1 {
                let rank = reserved_free_slots
                    .iter()
                    .enumerate()
                    .filter(|(rank, slots)| {
                        !slots.is_empty()
                            && self
                                .rank_kv_budget(*rank)
                                .saturating_sub(round_reserved[*rank])
                                >= blocks_needed
                    })
                    .max_by_key(|(_, slots)| slots.len())
                    .map(|(rank, _)| rank);
                let Some(rank) = rank else {
                    still_pending.push(req);
                    continue;
                };
                let slot = reserved_free_slots[rank].remove(0);
                let mut kv =
                    self.pools[rank].new_request(req.prompt_tokens.clone(), req.max_tokens, None);
                if let Err(err) = kv.schedule_prefill(1, &self.pools[rank]) {
                    let message = format!(
                        "Kimi-K2 admission KV block accounting violated full-lifetime reservation: {err}"
                    );
                    error!("{message}");
                    send_scheduled(&req);
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message,
                        prompt_tokens: req.prompt_tokens.len(),
                        completion_tokens: 0,
                    });
                    continue;
                }
                round_reserved[rank] += blocks_needed.saturating_sub(1);
                send_scheduled(&req);
                decode_admissions[rank].push(DecodeAdmission { slot, req, kv });
                continue;
            }

            self.flush_decode_admissions(&mut decode_admissions);
            reserved_free_slots = self.free_slot_lists();
            round_reserved = vec![0usize; self.dp_world];

            let dp_rank = lb.pick_rank(&self.ranks);
            match dp_rank {
                Some(rank) => {
                    if self.rank_kv_budget(rank) < blocks_needed {
                        still_pending.push(req);
                        continue;
                    }
                    let Some(slot) = self.ranks[rank].find_free_slot() else {
                        still_pending.push(req);
                        continue;
                    };
                    let Some(prefill_slots) = self.prefill_slots_for(rank, slot) else {
                        still_pending.push(req);
                        continue;
                    };
                    self.admit_request(rank, slot, &prefill_slots, req);
                    reserved_free_slots = self.free_slot_lists();
                }
                None => still_pending.push(req),
            }
        }

        self.flush_decode_admissions(&mut decode_admissions);
        *pending_reqs = still_pending;
    }

    fn empty_decode_admissions(&self) -> Vec<Vec<DecodeAdmission>> {
        (0..self.dp_world).map(|_| Vec::new()).collect()
    }

    fn free_slot_lists(&self) -> Vec<Vec<usize>> {
        self.ranks.iter().map(DpRankState::free_slots).collect()
    }

    fn prefill_slots_for(&self, owning_rank: usize, owning_slot: usize) -> Option<Vec<usize>> {
        let mut slots = Vec::with_capacity(self.dp_world);
        for dp_rank in 0..self.dp_world {
            if dp_rank == owning_rank {
                slots.push(owning_slot);
            } else {
                slots.push(self.ranks[dp_rank].find_free_slot()?);
            }
        }
        Some(slots)
    }

    fn flush_decode_admissions(&mut self, batch: &mut Vec<Vec<DecodeAdmission>>) {
        if batch.iter().all(Vec::is_empty) {
            return;
        }
        let ready = std::mem::replace(batch, self.empty_decode_admissions());
        self.synchronized_decode_admissions(ready);
    }

    fn admit_request(
        &mut self,
        dp_rank: usize,
        slot: usize,
        prefill_slots: &[usize],
        req: GenerateRequest,
    ) {
        send_scheduled(&req);

        let mut kv =
            self.pools[dp_rank].new_request(req.prompt_tokens.clone(), req.max_tokens, None);
        let cached_tokens = match kv.match_and_add_prefix(&self.pools[dp_rank]) {
            Ok(cached) => cached,
            Err(err) => {
                let message = format!("Kimi-K2 prefix cache matching failed: {err:#}");
                error!("{message}");
                let _ = req.token_tx.send(TokenEvent::Error {
                    message,
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
                return;
            }
        };
        let suffix_len = req.prompt_tokens.len() - cached_tokens;
        if let Err(err) = kv.schedule_prefill(suffix_len, &self.pools[dp_rank]) {
            let message = format!(
                "Kimi-K2 prefill KV block accounting violated full-lifetime reservation: {err}"
            );
            error!("{message}");
            let _ = req.token_tx.send(TokenEvent::Error {
                message,
                prompt_tokens: req.prompt_tokens.len(),
                completion_tokens: 0,
            });
            return;
        }
        let kv_pages = KimiKvStepPages::single(
            kv.step_page_indices(suffix_len),
            self.pools[dp_rank].padding_block_id(),
        );

        // Prefill: all ranks run prefill in lock-step so DeepEP collectives
        // align. Owning rank processes the uncached suffix; padding ranks
        // process a single dummy token into a free slot (output discarded).
        self.synchronized_prefill(dp_rank, prefill_slots, &req, cached_tokens, &kv_pages);

        let prompt_len = req.prompt_tokens.len();

        // Owner first: it processes the real tokens, so it is the realistic
        // fast-fail (padding ranks run a fixed dummy input). Any rank error
        // poisons the lock-step — see abort_poisoned_step.
        let owner_report = match self.result_rxs[dp_rank].recv() {
            Ok(StepResult::Prefill(Ok(report))) => report,
            Ok(StepResult::Prefill(Err(err))) => {
                abort_poisoned_step(dp_rank, "prefill", &format!("{err:#}"))
            }
            Ok(StepResult::Decode(_)) => {
                abort_poisoned_step(dp_rank, "prefill", "returned decode result during prefill")
            }
            Err(_) => abort_dropped_result_channel(dp_rank, "prefill"),
        };
        for r in 0..self.dp_world {
            if r == dp_rank {
                continue;
            }
            match self.result_rxs[r].recv() {
                Ok(StepResult::Prefill(Ok(_))) => {}
                Ok(StepResult::Prefill(Err(err))) => {
                    abort_poisoned_step(r, "padding prefill", &format!("{err:#}"))
                }
                Ok(StepResult::Decode(_)) => abort_poisoned_step(
                    r,
                    "padding prefill",
                    "returned decode result during prefill",
                ),
                Err(_) => abort_dropped_result_channel(r, "prefill"),
            }
        }

        let last_token = owner_report.local_next_token_global_id;
        if let Err(err) = kv.apply_prefill(last_token, &self.pools[dp_rank]) {
            let message = format!("Kimi-K2 prefill KV bookkeeping failed: {err:#}");
            error!("{message}");
            let _ = req.token_tx.send(TokenEvent::Error {
                message,
                prompt_tokens: prompt_len,
                completion_tokens: 0,
            });
            return;
        }
        let completion_tokens = 1;
        if !emit_sampled_token(
            &req.token_tx,
            &req.stop_policy,
            &self.stop_token_ids,
            last_token,
            owner_report.logprob,
            prompt_len,
            completion_tokens,
            req.max_tokens,
        ) {
            return;
        }

        let options = row_options(&req);
        self.ranks[dp_rank].slots[slot] = Some(RequestState {
            token_tx: req.token_tx,
            stop_policy: req.stop_policy,
            prompt_len,
            completion_tokens,
            max_tokens: req.max_tokens,
            last_token,
            options,
            kv,
        });
    }

    fn synchronized_decode_admissions(&mut self, batch: Vec<Vec<DecodeAdmission>>) {
        let mut rows_by_rank = Vec::with_capacity(self.dp_world);
        for (dp_rank, rank_batch) in batch.into_iter().enumerate() {
            let seed = self.next_step_seed();
            let padding_page = self.pools[dp_rank].padding_block_id();
            let mut rows = self.ranks[dp_rank]
                .schedule_active_decode_inputs(&self.pools[dp_rank])
                .into_iter()
                .map(DecodeBatchRow::Active)
                .collect::<Vec<_>>();
            rows.extend(
                rank_batch
                    .into_iter()
                    .map(|admission| DecodeBatchRow::Admission(Box::new(admission))),
            );
            if rows.len() > MAX_BATCH_PER_DP {
                let message = format!(
                    "Kimi-K2 DP rank {dp_rank} decode rows exceed arena capacity {MAX_BATCH_PER_DP}"
                );
                self.fail_decode_rows(dp_rank, rows, &message);
                send_step_command(
                    &self.step_txs[dp_rank],
                    dp_rank,
                    "decode admission overflow padding",
                    build_padding_decode_command(seed, padding_page),
                );
                rows_by_rank.push(Vec::new());
                continue;
            }

            let cmd = if rows.is_empty() {
                build_padding_decode_command(seed, padding_page)
            } else {
                build_decode_command_from_rows(&rows, seed, padding_page)
            };
            send_step_command(&self.step_txs[dp_rank], dp_rank, "decode admission", cmd);
            rows_by_rank.push(rows);
        }

        for (dp_rank, rows) in rows_by_rank.into_iter().enumerate() {
            let reports = match self.result_rxs[dp_rank].recv() {
                Ok(StepResult::Decode(Ok(reports))) => reports,
                Ok(StepResult::Decode(Err(err))) => {
                    abort_poisoned_step(dp_rank, "decode admission", &format!("{err:#}"))
                }
                Ok(StepResult::Prefill(_)) => abort_poisoned_step(
                    dp_rank,
                    "decode admission",
                    "returned prefill result during decode admission",
                ),
                Err(_) => abort_dropped_result_channel(dp_rank, "decode admission"),
            };

            if rows.is_empty() {
                continue;
            }

            for (row, report) in rows.into_iter().zip(reports) {
                match row {
                    DecodeBatchRow::Active(input) => {
                        let pool = &self.pools[dp_rank];
                        self.ranks[dp_rank].process_decode_report(
                            input.slot,
                            &report,
                            &self.stop_token_ids,
                            pool,
                        );
                    }
                    DecodeBatchRow::Admission(admission) => {
                        self.install_decode_admission_result(dp_rank, *admission, &report);
                    }
                }
            }
        }
    }

    fn install_decode_admission_result(
        &mut self,
        dp_rank: usize,
        admission: DecodeAdmission,
        report: &KimiOneTokenForwardReport,
    ) {
        let DecodeAdmission { slot, req, mut kv } = admission;
        let token_id = report.local_next_token_global_id;
        if let Err(err) = kv.apply_prefill(token_id, &self.pools[dp_rank]) {
            let message = format!("Kimi-K2 admission KV bookkeeping failed: {err:#}");
            error!("{message}");
            let _ = req.token_tx.send(TokenEvent::Error {
                message,
                prompt_tokens: req.prompt_tokens.len(),
                completion_tokens: 0,
            });
            return;
        }
        let completion_tokens = 1;
        if !emit_sampled_token(
            &req.token_tx,
            &req.stop_policy,
            &self.stop_token_ids,
            token_id,
            report.logprob.clone(),
            req.prompt_tokens.len(),
            completion_tokens,
            req.max_tokens,
        ) {
            return;
        }

        let options = row_options(&req);
        self.ranks[dp_rank].slots[slot] = Some(RequestState {
            token_tx: req.token_tx,
            stop_policy: req.stop_policy,
            prompt_len: req.prompt_tokens.len(),
            completion_tokens,
            max_tokens: req.max_tokens,
            last_token: token_id,
            options,
            kv,
        });
    }

    fn fail_decode_rows(&mut self, dp_rank: usize, rows: Vec<DecodeBatchRow>, message: &str) {
        if rows
            .iter()
            .any(|row| matches!(row, DecodeBatchRow::Active(_)))
        {
            let err = anyhow::anyhow!(message.to_string());
            self.ranks[dp_rank].fail_all_active(&err);
        }

        for row in rows {
            let DecodeBatchRow::Admission(admission) = row else {
                continue;
            };
            let _ = admission.req.token_tx.send(TokenEvent::Error {
                message: message.to_string(),
                prompt_tokens: admission.req.prompt_tokens.len(),
                completion_tokens: 0,
            });
        }
    }

    /// `cached_tokens` is the request's prefix-cache hit length: the owning
    /// rank forwards only the uncached suffix, so EP scratch and padding
    /// ranks size against the suffix, not the full prompt.
    fn synchronized_prefill(
        &mut self,
        owning_rank: usize,
        prefill_slots: &[usize],
        req: &GenerateRequest,
        cached_tokens: usize,
        kv_pages: &KimiKvStepPages,
    ) {
        let suffix = &req.prompt_tokens[cached_tokens..];
        let ep_max_seq_len = suffix.len();
        debug_assert_eq!(prefill_slots.len(), self.dp_world);
        for (dp_rank, slot) in prefill_slots
            .iter()
            .copied()
            .enumerate()
            .take(self.dp_world)
        {
            let seed = self.next_step_seed();
            let cmd = if dp_rank == owning_rank {
                StepCommand::Prefill {
                    input_ids: suffix.to_vec(),
                    slot,
                    decode_batch_size: MAX_BATCH_PER_DP,
                    cached_tokens,
                    ep_max_seq_len,
                    kv_pages: kv_pages.clone(),
                    row: row_options(req),
                    seed,
                }
            } else {
                // All ranks run prefill so they traverse layers at the same
                // pace, making exactly 1 DeepEP dispatch/combine per MoE layer.
                // The dummy token's KV write lands on the rank's padding page.
                let padding_page = self.pools[dp_rank].padding_block_id();
                StepCommand::Prefill {
                    input_ids: vec![0],
                    slot,
                    decode_batch_size: MAX_BATCH_PER_DP,
                    cached_tokens: 0,
                    ep_max_seq_len,
                    kv_pages: KimiKvStepPages::single(vec![padding_page], padding_page),
                    row: KimiRowOptions::default(),
                    seed,
                }
            };
            send_step_command(&self.step_txs[dp_rank], dp_rank, "prefill", cmd);
        }
    }

    fn synchronized_decode_step(&mut self) {
        // Build per-rank decode commands
        for dp_rank in 0..self.dp_world {
            let seed = self.next_step_seed();
            let cmd = self.ranks[dp_rank].build_decode_command(&self.pools[dp_rank], seed);
            send_step_command(&self.step_txs[dp_rank], dp_rank, "decode", cmd);
        }

        // Collect results from all ranks. A rank-level error means that rank
        // bailed without completing the step's EP collectives while the other
        // ranks are parked inside them — there is no recovery, only a silent
        // engine-wide hang. Crash loudly instead (#239 verification found
        // exactly this deadlock).
        for dp_rank in 0..self.dp_world {
            let result = match self.result_rxs[dp_rank].recv() {
                Ok(StepResult::Decode(Ok(reports))) => reports,
                Ok(StepResult::Decode(Err(err))) => {
                    abort_poisoned_step(dp_rank, "decode", &format!("{err:#}"))
                }
                Ok(StepResult::Prefill(_)) => {
                    abort_poisoned_step(dp_rank, "decode", "returned prefill result during decode")
                }
                Err(_) => abort_dropped_result_channel(dp_rank, "decode"),
            };

            let pool = &self.pools[dp_rank];
            self.ranks[dp_rank].process_decode_results(result, &self.stop_token_ids, pool);
        }
    }
}

impl DpRankState {
    fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }

    pub(in crate::runner) fn free_slot_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_none()).count()
    }

    pub(in crate::runner) fn has_free_slot(&self) -> bool {
        self.slots.iter().any(Option::is_none)
    }

    fn find_free_slot(&self) -> Option<usize> {
        self.slots.iter().position(Option::is_none)
    }

    fn free_slots(&self) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(idx, slot)| slot.is_none().then_some(idx))
            .collect()
    }

    fn build_decode_command(&mut self, pool: &BlockPool, seed: u64) -> StepCommand {
        let inputs = self.schedule_active_decode_inputs(pool);
        if inputs.is_empty() {
            return build_padding_decode_command(seed, pool.padding_block_id());
        }

        build_decode_command_from_inputs(inputs, seed, pool.padding_block_id())
    }

    /// Allocate this step's KV pages for every active request and collect
    /// decode inputs. Full-lifetime reservation makes allocation failure an
    /// accounting bug — such a request is failed and dropped, keeping the
    /// command's rows aligned with the slots a result will be paired with.
    fn schedule_active_decode_inputs(&mut self, pool: &BlockPool) -> Vec<DecodeInput> {
        let mut inputs = Vec::new();
        for slot in 0..self.slots.len() {
            let Some(state) = self.slots[slot].as_mut() else {
                continue;
            };
            if let Err(err) = state.kv.schedule_decode(pool) {
                let message = format!(
                    "Kimi-K2 decode KV block accounting violated full-lifetime reservation: {err}"
                );
                error!("{message}");
                let _ = state.token_tx.send(TokenEvent::Error {
                    message,
                    prompt_tokens: state.prompt_len,
                    completion_tokens: state.completion_tokens,
                });
                self.slots[slot] = None;
                continue;
            }
            inputs.push(DecodeInput {
                token_id: state.last_token,
                append_position: state.prompt_len + state.completion_tokens - 1,
                slot,
                options: state.options,
                pages: state.kv.step_page_indices(1),
            });
        }
        inputs
    }

    fn process_decode_results(
        &mut self,
        reports: Vec<KimiOneTokenForwardReport>,
        stop_token_ids: &[u32],
        pool: &BlockPool,
    ) {
        let active_slots: Vec<usize> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|_| i))
            .collect();

        if active_slots.is_empty() {
            return;
        }

        for (slot_idx, report) in active_slots.into_iter().zip(reports) {
            self.process_decode_report(slot_idx, &report, stop_token_ids, pool);
        }
    }

    fn process_decode_report(
        &mut self,
        slot_idx: usize,
        report: &KimiOneTokenForwardReport,
        stop_token_ids: &[u32],
        pool: &BlockPool,
    ) {
        let Some(req) = self.slots[slot_idx].as_mut() else {
            return;
        };

        let token_id = report.local_next_token_global_id;
        req.completion_tokens += 1;

        if let Err(err) = req.kv.apply_decode(token_id, pool) {
            let message = format!("Kimi-K2 decode KV bookkeeping failed: {err:#}");
            error!("{message}");
            let _ = req.token_tx.send(TokenEvent::Error {
                message,
                prompt_tokens: req.prompt_len,
                completion_tokens: req.completion_tokens,
            });
            self.slots[slot_idx] = None;
            return;
        }

        if !emit_sampled_token(
            &req.token_tx,
            &req.stop_policy,
            stop_token_ids,
            token_id,
            report.logprob.clone(),
            req.prompt_len,
            req.completion_tokens,
            req.max_tokens,
        ) {
            self.slots[slot_idx] = None;
        } else {
            req.last_token = token_id;
        }
    }

    fn fail_all_active(&mut self, err: &anyhow::Error) {
        let message = format!("{err:#}");
        for slot in &mut self.slots {
            if let Some(req) = slot.take() {
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: message.clone(),
                    prompt_tokens: req.prompt_len,
                    completion_tokens: req.completion_tokens,
                });
            }
        }
    }
}

/// Padding command for idle ranks: 1 dummy token so the rank participates
/// in EP collectives without producing real output. The dummy KV write
/// lands on the rank's padding page.
fn build_padding_decode_command(seed: u64, padding_page: i32) -> StepCommand {
    StepCommand::Decode {
        token_ids: vec![0],
        positions: vec![0],
        slots: vec![0],
        decode_batch_size: MAX_BATCH_PER_DP,
        kv_pages: KimiKvStepPages::single(vec![padding_page], padding_page),
        rows: vec![KimiRowOptions::default()],
        seed,
    }
}

fn build_decode_command_from_rows(
    rows: &[DecodeBatchRow],
    seed: u64,
    padding_page: i32,
) -> StepCommand {
    StepCommand::Decode {
        token_ids: rows.iter().map(DecodeBatchRow::token_id).collect(),
        positions: rows.iter().map(DecodeBatchRow::append_position).collect(),
        slots: rows.iter().map(DecodeBatchRow::slot).collect(),
        decode_batch_size: MAX_BATCH_PER_DP,
        kv_pages: KimiKvStepPages::new(
            rows.iter().map(DecodeBatchRow::pages).collect(),
            padding_page,
        ),
        rows: rows.iter().map(DecodeBatchRow::options).collect(),
        seed,
    }
}

fn build_decode_command_from_inputs(
    inputs: Vec<DecodeInput>,
    seed: u64,
    padding_page: i32,
) -> StepCommand {
    StepCommand::Decode {
        token_ids: inputs.iter().map(|input| input.token_id).collect(),
        positions: inputs.iter().map(|input| input.append_position).collect(),
        slots: inputs.iter().map(|input| input.slot).collect(),
        decode_batch_size: MAX_BATCH_PER_DP,
        rows: inputs.iter().map(|input| input.options).collect(),
        kv_pages: KimiKvStepPages::new(
            inputs.into_iter().map(|input| input.pages).collect(),
            padding_page,
        ),
        seed,
    }
}

fn send_step_command(tx: &Sender<StepCommand>, dp_rank: usize, phase: &str, command: StepCommand) {
    if tx.send(command).is_err() {
        error!("fatal: DP rank {dp_rank} forward thread dropped before {phase}");
        std::process::abort();
    }
}

fn abort_dropped_result_channel(dp_rank: usize, phase: &str) -> ! {
    error!("fatal: DP rank {dp_rank} forward thread dropped during {phase}");
    std::process::abort();
}

/// A rank failed (or broke protocol) mid lock-step: it never completed the
/// step's EP collectives, so the remaining ranks are parked inside them.
/// There is no recovery path — only a silent engine-wide hang (#239's H200
/// verification hit exactly this). Crash loudly instead. The message also
/// goes to stderr directly so it survives processes without a tracing
/// subscriber (tests).
fn abort_poisoned_step(dp_rank: usize, phase: &str, detail: &str) -> ! {
    let message = format!("kimi-k2: fatal: DP rank {dp_rank} poisoned the {phase} step: {detail}");
    error!("{message}");
    eprintln!("{message}");
    std::process::abort();
}

fn rank_forward_loop(
    executor: Box<dyn ForwardExecutor + Send>,
    cmd_rx: Receiver<StepCommand>,
    res_tx: Sender<StepResult>,
) {
    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            StepCommand::Decode {
                token_ids,
                positions,
                slots,
                decode_batch_size,
                kv_pages,
                rows,
                seed,
            } => {
                let result = executor.forward_decode_batch(
                    &token_ids,
                    &positions,
                    &slots,
                    decode_batch_size,
                    &kv_pages,
                    &rows,
                    seed,
                );
                let _ = res_tx.send(StepResult::Decode(result));
            }
            StepCommand::Prefill {
                input_ids,
                slot,
                decode_batch_size,
                cached_tokens,
                ep_max_seq_len,
                kv_pages,
                row,
                seed,
            } => {
                let result = executor.forward_prefill(
                    &input_ids,
                    slot,
                    decode_batch_size,
                    cached_tokens,
                    ep_max_seq_len,
                    &kv_pages,
                    row,
                    seed,
                );
                let _ = res_tx.send(StepResult::Prefill(result));
            }
            StepCommand::Shutdown => break,
        }
    }
    drop(executor);
    drop(cmd_rx);
    drop(res_tx);
}

#[cfg(test)]
mod tests {
    use pegainfer_frontend::engine::EosPolicy;
    use pegainfer_frontend::engine::StopCause;
    use pegainfer_frontend::sampler::SamplingParams;

    use super::*;

    fn dummy_request(prompt_tokens: Vec<u32>, max_tokens: usize) -> GenerateRequest {
        let (token_tx, _token_rx) = TokenSink::standalone();
        GenerateRequest {
            trace_parent: None,
            request_id: None,
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens,
            params: SamplingParams::default(),
            stop_policy: StopPolicy::default(),
            max_tokens,
            lora_adapter: None,
            kv_transfer_params: None,
            token_tx,
            logprobs: 0,
            echo: false,
        }
    }

    fn test_pool() -> BlockPool {
        BlockPool::new(16, 1024).expect("test pool")
    }

    /// Walk a fresh request through the real kvbm lifecycle to the state
    /// "prompt prefilled, `completion_tokens` generated, last one dangling",
    /// so page accounting matches what decode scheduling expects.
    fn dummy_kv(
        pool: &BlockPool,
        prompt_len: usize,
        completion_tokens: usize,
        max_tokens: usize,
        last_token: u32,
    ) -> RequestKv {
        // Prompt content derives from last_token so distinct requests get
        // distinct block-hash chains (identical chains would pin each
        // other's registered blocks via the registry).
        let mut kv = pool.new_request(vec![last_token; prompt_len], max_tokens, None);
        kv.schedule_prefill(prompt_len, pool)
            .expect("prefill blocks");
        kv.apply_prefill(last_token, pool).expect("apply prefill");
        for _ in 1..completion_tokens {
            kv.schedule_decode(pool).expect("decode block");
            kv.apply_decode(last_token, pool).expect("apply decode");
        }
        kv
    }

    fn dummy_state(
        pool: &BlockPool,
        prompt_len: usize,
        completion_tokens: usize,
        max_tokens: usize,
        last_token: u32,
    ) -> RequestState {
        let (token_tx, _token_rx) = TokenSink::standalone();
        RequestState {
            token_tx,
            stop_policy: StopPolicy::default(),
            prompt_len,
            completion_tokens,
            max_tokens,
            last_token,
            options: KimiRowOptions::default(),
            kv: dummy_kv(pool, prompt_len, completion_tokens, max_tokens, last_token),
        }
    }

    fn dummy_report(token_id: u32) -> KimiOneTokenForwardReport {
        KimiOneTokenForwardReport {
            rank: 0,
            batch_slot: 0,
            input_token_id: 0,
            local_next_token_id: token_id,
            local_next_token_global_id: token_id,
            local_top_logit_f32: 0.0,
            vocab_start: 0,
            vocab_rows: 0,
            dense_layers_executed: 0,
            moe_layers_executed: 0,
            logprob: None,
        }
    }

    fn test_coordinator(dp_world: usize) -> DpCoordinator {
        DpCoordinator {
            dp_world,
            ranks: (0..dp_world)
                .map(|_| DpRankState {
                    slots: (0..MAX_BATCH_PER_DP).map(|_| None).collect(),
                })
                .collect(),
            pools: (0..dp_world).map(|_| test_pool()).collect(),
            executors: Vec::new(),
            step_txs: Vec::new(),
            result_rxs: Vec::new(),
            stop_token_ids: Vec::new(),
            rng: rand::SeedableRng::seed_from_u64(0),
        }
    }

    #[test]
    fn sparse_decode_slot_keeps_stable_arena_capacity() {
        let pool = test_pool();
        let mut rank = DpRankState {
            slots: (0..MAX_BATCH_PER_DP).map(|_| None).collect(),
        };
        rank.slots[MAX_BATCH_PER_DP - 1] = Some(dummy_state(&pool, 4, 3, 16, 123));

        let StepCommand::Decode {
            token_ids,
            positions,
            slots,
            decode_batch_size,
            kv_pages,
            rows,
            seed,
        } = rank.build_decode_command(&pool, 7)
        else {
            panic!("decode command expected");
        };

        assert_eq!(decode_batch_size, MAX_BATCH_PER_DP);
        assert_eq!(token_ids, vec![123]);
        assert_eq!(positions, vec![6]);
        assert_eq!(slots, vec![MAX_BATCH_PER_DP - 1]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].logprobs, 0);
        // 6 KV tokens + this step's append fit one 16-token page.
        assert_eq!(kv_pages.rows(), 1);
        assert_eq!(kv_pages.row(0).expect("row 0").len(), 1);
        assert_eq!(seed, 7);
    }

    #[test]
    fn decode_rows_merge_active_decode_and_new_admission() {
        let pool = test_pool();
        let mut sampling_req = dummy_request(vec![99], 8);
        sampling_req.params.temperature = 0.8;
        sampling_req.params.top_p = 0.9;
        let mut admission_kv = pool.new_request(vec![99], 8, None);
        admission_kv
            .schedule_prefill(1, &pool)
            .expect("admission prefill block");
        let batch_rows = vec![
            DecodeBatchRow::Active(DecodeInput {
                token_id: 11,
                append_position: 5,
                slot: 3,
                options: KimiRowOptions {
                    logprobs: 4,
                    sampling: SamplingParams::default(),
                },
                pages: vec![5],
            }),
            DecodeBatchRow::Admission(Box::new(DecodeAdmission {
                slot: 7,
                req: sampling_req,
                kv: admission_kv,
            })),
        ];

        let StepCommand::Decode {
            token_ids,
            positions,
            slots,
            decode_batch_size,
            kv_pages,
            rows,
            seed,
        } = build_decode_command_from_rows(&batch_rows, 42, pool.padding_block_id())
        else {
            panic!("decode command expected");
        };

        assert_eq!(decode_batch_size, MAX_BATCH_PER_DP);
        assert_eq!(token_ids, vec![11, 99]);
        assert_eq!(positions, vec![5, 0]);
        assert_eq!(slots, vec![3, 7]);
        assert_eq!(rows[0].logprobs, 4);
        assert!(rows[0].sampling.is_greedy());
        assert_eq!(rows[1].logprobs, 0);
        assert!(!rows[1].sampling.is_greedy());
        assert!((rows[1].sampling.temperature - 0.8).abs() < f32::EPSILON);
        assert_eq!(kv_pages.rows(), 2);
        assert_eq!(kv_pages.row(0).expect("active row"), &[5]);
        assert_eq!(kv_pages.row(1).expect("admission row").len(), 1);
        assert_eq!(seed, 42);
    }

    #[test]
    fn padding_decode_uses_stable_arena_capacity() {
        let StepCommand::Decode {
            token_ids,
            positions,
            slots,
            decode_batch_size,
            kv_pages,
            rows,
            seed: _,
        } = build_padding_decode_command(1, 0)
        else {
            panic!("decode command expected");
        };

        assert_eq!(decode_batch_size, MAX_BATCH_PER_DP);
        assert_eq!(token_ids, vec![0]);
        assert_eq!(positions, vec![0]);
        assert_eq!(slots, vec![0]);
        assert_eq!(rows.len(), 1);
        assert!(rows[0].sampling.is_greedy());
        // The dummy row's single KV write lands on the padding page.
        assert_eq!(kv_pages.rows(), 1);
        assert_eq!(kv_pages.row(0).expect("padding row"), &[0]);
    }

    #[test]
    fn decode_report_finishes_with_stop_at_eos() {
        let pool = test_pool();
        let mut rank = DpRankState {
            slots: (0..MAX_BATCH_PER_DP).map(|_| None).collect(),
        };
        let (token_tx, mut token_rx) = TokenSink::standalone();
        let mut kv = dummy_kv(&pool, 4, 1, 16, 7);
        kv.schedule_decode(&pool).expect("decode block");
        rank.slots[0] = Some(RequestState {
            token_tx,
            stop_policy: StopPolicy::default(),
            prompt_len: 4,
            completion_tokens: 1,
            max_tokens: 16,
            last_token: 7,
            options: KimiRowOptions::default(),
            kv,
        });

        rank.process_decode_report(0, &dummy_report(163_586), &[163_586], &pool);

        assert!(rank.slots[0].is_none());
        let Ok((_, TokenEvent::Token { id, .. })) = token_rx.try_recv() else {
            panic!("expected trigger Token event");
        };
        assert_eq!(id, 163_586);
        let Ok((
            _,
            TokenEvent::Finished {
                finish_reason,
                stop_cause,
                completion_tokens,
                ..
            },
        )) = token_rx.try_recv()
        else {
            panic!("expected Finished event");
        };
        assert_eq!(finish_reason, FinishReason::Stop);
        assert_eq!(stop_cause, Some(StopCause::Eos(163_586)));
        assert_eq!(completion_tokens, 2);
        assert!(token_rx.try_recv().is_err());
    }

    #[test]
    fn decode_report_honors_request_eos_policy() {
        let pool = test_pool();
        let mut rank = DpRankState {
            slots: (0..MAX_BATCH_PER_DP).map(|_| None).collect(),
        };
        let (token_tx, mut token_rx) = TokenSink::standalone();
        let mut kv = dummy_kv(&pool, 4, 1, 16, 7);
        kv.schedule_decode(&pool).expect("decode block");
        rank.slots[0] = Some(RequestState {
            token_tx,
            stop_policy: StopPolicy {
                eos: EosPolicy::Ignore,
                token_ids: Vec::new(),
            },
            prompt_len: 4,
            completion_tokens: 1,
            max_tokens: 16,
            last_token: 7,
            options: KimiRowOptions {
                logprobs: 0,
                sampling: SamplingParams {
                    ignore_eos: true,
                    ..SamplingParams::default()
                },
            },
            kv,
        });

        rank.process_decode_report(0, &dummy_report(163_586), &[163_586], &pool);

        assert!(rank.slots[0].is_some());
        let Ok((_, TokenEvent::Token { id, .. })) = token_rx.try_recv() else {
            panic!("expected Token event");
        };
        assert_eq!(id, 163_586);
    }

    #[test]
    fn decode_report_stops_on_explicit_token_while_eos_is_ignored() {
        let pool = test_pool();
        let mut rank = DpRankState {
            slots: (0..MAX_BATCH_PER_DP).map(|_| None).collect(),
        };
        let (token_tx, mut token_rx) = TokenSink::standalone();
        let mut kv = dummy_kv(&pool, 4, 1, 16, 7);
        kv.schedule_decode(&pool).expect("decode block");
        rank.slots[0] = Some(RequestState {
            token_tx,
            stop_policy: StopPolicy {
                eos: EosPolicy::Ignore,
                token_ids: vec![42],
            },
            prompt_len: 4,
            completion_tokens: 1,
            max_tokens: 16,
            last_token: 7,
            options: KimiRowOptions::default(),
            kv,
        });

        rank.process_decode_report(0, &dummy_report(42), &[163_586], &pool);

        assert!(rank.slots[0].is_none());
        assert!(matches!(
            token_rx.try_recv(),
            Ok((_, TokenEvent::Token { id: 42, .. }))
        ));
        let Ok((_, TokenEvent::Finished { stop_cause, .. })) = token_rx.try_recv() else {
            panic!("expected terminal Finished event");
        };
        assert_eq!(stop_cause, Some(StopCause::Token(42)));
    }

    #[test]
    fn rank_kv_budget_reserves_remaining_lifetime_of_active_requests() {
        let mut coordinator = test_coordinator(1);
        let baseline_available = coordinator.pools[0].available_blocks();
        let baseline_budget = coordinator.rank_kv_budget(0);
        assert_eq!(baseline_budget, baseline_available);

        // prompt 16, max_tokens 100: peak lifetime draw is
        // ceil((16 + 100)/16) = 8 blocks (kvbm provisions the final
        // dangling token's block even though its KV is never written).
        // At every decode point, held + future must cover that peak —
        // otherwise concurrent long-output requests are over-admitted and
        // die mid-decode, the exact failure #239 exists to prevent. The
        // upper bound caps the waste from the held-blocks lower bound
        // (kvbm keeps a staged spare beyond ceil(kv_tokens/bs)).
        let lifetime = (16usize + 100).div_ceil(16);
        for completion in [1usize, 10, 16, 17, 99] {
            // Distinct token content per iteration: identical chains would
            // hash-match earlier iterations' registered blocks and pin them,
            // skewing the held-block measurement.
            let mut state = dummy_state(
                &coordinator.pools[0],
                16,
                completion,
                100,
                completion as u32,
            );
            state.max_tokens = 100;
            coordinator.ranks[0].slots[0] = Some(state);

            let reserved = baseline_budget - coordinator.rank_kv_budget(0);
            let held = baseline_available - coordinator.pools[0].available_blocks();
            assert!(
                reserved >= lifetime,
                "completion={completion}: held {held} + future only covers \
                 {reserved} of the {lifetime}-block lifetime"
            );
            assert!(
                reserved <= lifetime + 2,
                "completion={completion}: reserving {reserved} blocks for a \
                 {lifetime}-block lifetime wastes the pool"
            );
            coordinator.ranks[0].slots[0] = None;
        }
        assert_eq!(coordinator.rank_kv_budget(0), baseline_budget);
    }

    #[test]
    fn prefill_padding_slots_avoid_active_requests() {
        let mut coordinator = test_coordinator(2);
        let state = dummy_state(&coordinator.pools[0], 4, 1, 16, 10);
        coordinator.ranks[0].slots[0] = Some(state);

        let slots = coordinator
            .prefill_slots_for(1, 3)
            .expect("rank 0 has a free padding slot");

        assert_ne!(slots[0], 0);
        assert_eq!(slots[1], 3);
    }

    #[test]
    fn prefill_waits_when_any_padding_rank_is_full() {
        let mut coordinator = test_coordinator(2);
        for slot in 0..MAX_BATCH_PER_DP {
            let state = dummy_state(&coordinator.pools[0], 4, 1, 16, slot as u32);
            coordinator.ranks[0].slots[slot] = Some(state);
        }

        assert!(coordinator.prefill_slots_for(1, 3).is_none());
    }
}
