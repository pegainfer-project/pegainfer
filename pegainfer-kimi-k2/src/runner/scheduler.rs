use std::collections::VecDeque;
use std::thread;
use std::time::Duration;
use std::time::Instant;

pub(in crate::runner) mod dp;
mod lifecycle;

use anyhow::Context;
use anyhow::Result;
use lifecycle::preflight_prefill_candidate;
use lifecycle::send_scheduled;
use lifecycle::validate_kv_capacity;
use log::error;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::StopPolicy;
use pegainfer_frontend::engine::SubmittedRequest;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::engine::TokenSink;
use pegainfer_kv_cache::BlockPool;
use pegainfer_kv_cache::RequestKv;
use tokio::sync::mpsc;

use crate::runner::executor::ForwardExecutor;
use crate::runner::worker::KimiKvStepPages;
use crate::runner::worker::KimiRowOptions;

const KIMI_RUNNER_MAX_BATCH: usize = 64;
const KIMI_DECODE_ADMISSION_MICROBATCH: usize = 64;
const KIMI_PREFILL_BATCH_COALESCE: Duration = Duration::from_millis(100);
const KIMI_PREFILL_BATCH_POLL: Duration = Duration::from_micros(50);

pub(super) struct KimiK2Scheduler {
    executor: Box<dyn ForwardExecutor + Send>,
    stop_token_ids: Vec<u32>,
    pool: BlockPool,
}

fn row_options(req: &GenerateRequest) -> KimiRowOptions {
    KimiRowOptions {
        logprobs: req.logprobs,
        sampling: req.params,
    }
}

/// Deliver a sampled token before its terminal event so the protocol layer
/// receives the exact trigger token and logprob. Stop classification stays on
/// the host and does not change the GPU sampling row or CUDA graph inputs.
fn emit_sampled_token(
    token_tx: &TokenSink,
    stop_policy: &StopPolicy,
    model_eos: &[u32],
    token_id: u32,
    logprob: Option<TokenLogprob>,
    prompt_tokens: usize,
    completion_tokens: usize,
    max_tokens: usize,
) -> bool {
    if token_tx
        .send(TokenEvent::Token {
            id: token_id,
            logprob,
        })
        .is_err()
    {
        return false;
    }

    if let Some(stop_cause) = stop_policy.classify(token_id, |id| model_eos.contains(&id)) {
        let _ = token_tx.send(TokenEvent::Finished {
            finish_reason: FinishReason::Stop,
            stop_cause: Some(stop_cause),
            prompt_tokens,
            completion_tokens,
        });
        return false;
    }

    if completion_tokens >= max_tokens {
        let _ = token_tx.send(TokenEvent::Finished {
            finish_reason: FinishReason::Length,
            stop_cause: None,
            prompt_tokens,
            completion_tokens,
        });
        return false;
    }

    true
}

struct ActiveKimiRequest {
    token_tx: TokenSink,
    stop_policy: StopPolicy,
    prompt_len: usize,
    completion_tokens: usize,
    max_tokens: usize,
    last_token: u32,
    slot: usize,
    decode_batch_size: usize,
    options: KimiRowOptions,
    /// Pool pages backing this request's KV; dropping the request releases
    /// them back to the pool.
    kv: RequestKv,
}

impl KimiK2Scheduler {
    pub(super) fn new(
        executor: Box<dyn ForwardExecutor + Send>,
        stop_token_ids: Vec<u32>,
        pool: BlockPool,
    ) -> Result<Self> {
        executor
            .ensure_decode_batch(KIMI_RUNNER_MAX_BATCH)
            .with_context(|| {
                format!("Kimi-K2 warm decode arena bs{KIMI_RUNNER_MAX_BATCH} before serving")
            })?;
        let warm_tokens = (0..KIMI_RUNNER_MAX_BATCH)
            .map(|idx| 100 + (idx % 1000) as u32)
            .collect::<Vec<_>>();
        let warm_positions = vec![0; KIMI_RUNNER_MAX_BATCH];
        let warm_slots = (0..KIMI_RUNNER_MAX_BATCH).collect::<Vec<_>>();
        let warm_rows = vec![KimiRowOptions::default(); KIMI_RUNNER_MAX_BATCH];
        // Warm rows all ride the padding page: garbage in, output discarded.
        let warm_pages = KimiKvStepPages::new(
            vec![vec![pool.padding_block_id()]; KIMI_RUNNER_MAX_BATCH],
            pool.padding_block_id(),
        );
        let _ = executor
            .forward_decode_batch(
                &warm_tokens,
                &warm_positions,
                &warm_slots,
                KIMI_RUNNER_MAX_BATCH,
                &warm_pages,
                &warm_rows,
                0,
            )
            .with_context(|| {
                format!("Kimi-K2 warm decode admission bs{KIMI_RUNNER_MAX_BATCH} before serving")
            })?;
        Ok(Self {
            executor,
            stop_token_ids,
            pool,
        })
    }

    pub(in crate::runner) fn run(
        &mut self,
        mut submit_rx: mpsc::UnboundedReceiver<SubmittedRequest>,
    ) {
        let mut pending = VecDeque::new();
        loop {
            if pending.is_empty() {
                match submit_rx.blocking_recv() {
                    Some((req, _kv_prefix)) => pending.push_back(req),
                    None => return,
                }
            }

            while let Ok((req, _kv_prefix)) = submit_rx.try_recv() {
                pending.push_back(req);
            }
            let deadline = Instant::now() + KIMI_PREFILL_BATCH_COALESCE;
            while pending.len() < KIMI_RUNNER_MAX_BATCH && Instant::now() < deadline {
                match submit_rx.try_recv() {
                    Ok((req, _kv_prefix)) => pending.push_back(req),
                    Err(mpsc::error::TryRecvError::Empty) => {
                        thread::sleep(KIMI_PREFILL_BATCH_POLL);
                    }
                    Err(mpsc::error::TryRecvError::Disconnected) => break,
                }
            }

            let mut batch = Vec::with_capacity(KIMI_RUNNER_MAX_BATCH);
            while batch.len() < KIMI_RUNNER_MAX_BATCH {
                let Some(req) = pending.pop_front() else {
                    break;
                };
                batch.push(req);
            }
            if !batch.is_empty() {
                // Requests deferred by the KV budget go back to the queue
                // front: the wave just drained, so the next wave starts from
                // a full pool and FCFS order is preserved.
                let deferred = self.handle_request_batch(batch);
                for req in deferred.into_iter().rev() {
                    pending.push_front(req);
                }
            }
        }
    }

    fn handle_request_batch(&mut self, reqs: Vec<GenerateRequest>) -> Vec<GenerateRequest> {
        let mut prefill_reqs = Vec::with_capacity(reqs.len());
        let mut deferred = Vec::new();
        // Full-lifetime reservation (#239, the qwen3 #85 pattern): a request
        // is only admitted when the pool can hold its prompt plus every
        // token it may generate, so decode can never run out of pages
        // mid-flight and poison the whole batch.
        let mut budget = self.pool.available_blocks();
        for req in reqs {
            let Some(req) = preflight_prefill_candidate(req) else {
                continue;
            };
            // Honor-or-reject (#237): this scheduler drives the TP8 path,
            // where each rank holds a vocab shard and cannot sample the
            // global distribution (#226). Rejecting here keeps one bad
            // request from failing the whole decode batch in the executor.
            if !req.params.is_greedy() {
                send_scheduled(&req);
                let _ = req.token_tx.send(TokenEvent::Rejected {
                    message: "Kimi TP8 path does not support sampling yet: use \
                              TP1/DP8 or temperature=0 (#237, #226)"
                        .to_string(),
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
                continue;
            }
            if let Err(message) = validate_kv_capacity(
                &req,
                self.pool.block_size(),
                self.pool.max_request_blocks(),
                None,
            ) {
                send_scheduled(&req);
                let _ = req.token_tx.send(TokenEvent::Rejected {
                    message,
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
                continue;
            }
            let blocks_needed = lifecycle::request_lifetime_blocks(&req, self.pool.block_size());
            if blocks_needed > budget {
                deferred.push(req);
                continue;
            }
            budget -= blocks_needed;
            send_scheduled(&req);
            prefill_reqs.push(req);
        }
        if prefill_reqs.is_empty() {
            return deferred;
        }

        let decode_batch_size = prefill_reqs.len();
        if let Err(err) = self.executor.ensure_decode_batch(decode_batch_size) {
            let message = format!(
                "Kimi-K2 decode arena allocation failed for batch size {decode_batch_size} after {}/{} ranks loaded: {err:#}",
                self.executor.gpu_weight_ready_count(),
                self.executor.worker_count()
            );
            error!("{message}");
            for req in prefill_reqs {
                let _ = req.token_tx.send(TokenEvent::Error {
                    message: message.clone(),
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
            }
            return deferred;
        }
        let mut active = Vec::with_capacity(prefill_reqs.len());
        let mut decode_admissions = Vec::with_capacity(KIMI_DECODE_ADMISSION_MICROBATCH);
        for (slot, req) in prefill_reqs.into_iter().enumerate() {
            if req.prompt_tokens.len() == 1 {
                decode_admissions.push((slot, req));
                if decode_admissions.len() == KIMI_DECODE_ADMISSION_MICROBATCH {
                    self.decode_admission_microbatch(
                        std::mem::take(&mut decode_admissions),
                        decode_batch_size,
                        &mut active,
                    );
                }
                continue;
            }
            if !decode_admissions.is_empty() {
                self.decode_admission_microbatch(
                    std::mem::take(&mut decode_admissions),
                    decode_batch_size,
                    &mut active,
                );
            }
            if let Some(active_req) = self.prefill_request(req, slot, decode_batch_size) {
                active.push(active_req);
            }
        }
        if !decode_admissions.is_empty() {
            self.decode_admission_microbatch(decode_admissions, decode_batch_size, &mut active);
        }

        while !active.is_empty() {
            let decode_batch_size = active[0].decode_batch_size;
            debug_assert!(
                active
                    .iter()
                    .all(|req| req.decode_batch_size == decode_batch_size)
            );
            let token_ids = active.iter().map(|req| req.last_token).collect::<Vec<_>>();
            let append_positions = active
                .iter()
                .map(|req| req.prompt_len + req.completion_tokens - 1)
                .collect::<Vec<_>>();
            let slots = active.iter().map(|req| req.slot).collect::<Vec<_>>();
            let rows = active.iter().map(|req| req.options).collect::<Vec<_>>();
            // Allocate this step's pages. Full-lifetime reservation makes
            // exhaustion impossible; a failure here is an accounting bug.
            let mut kv_rows = Vec::with_capacity(active.len());
            let mut schedule_err = None;
            for req in &mut active {
                if let Err(err) = req.kv.schedule_decode(&self.pool) {
                    schedule_err = Some(err);
                    break;
                }
                kv_rows.push(req.kv.step_page_indices(1));
            }
            if let Some(err) = schedule_err {
                let message = format!(
                    "Kimi-K2 decode KV block accounting violated full-lifetime reservation: {err}"
                );
                error!("{message}");
                for req in active.drain(..) {
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message: message.clone(),
                        prompt_tokens: req.prompt_len,
                        completion_tokens: req.completion_tokens,
                    });
                }
                return deferred;
            }
            let kv_pages = KimiKvStepPages::new(kv_rows, self.pool.padding_block_id());
            let reports = match self.executor.forward_decode_batch(
                &token_ids,
                &append_positions,
                &slots,
                decode_batch_size,
                &kv_pages,
                &rows,
                0,
            ) {
                Ok(reports) => reports,
                Err(err) => {
                    let message = format!(
                        "Kimi-K2 batch decode forward failed after {}/{} ranks loaded: {err:#}",
                        self.executor.gpu_weight_ready_count(),
                        self.executor.worker_count()
                    );
                    error!("{message}");
                    for req in active.drain(..) {
                        let _ = req.token_tx.send(TokenEvent::Error {
                            message: message.clone(),
                            prompt_tokens: req.prompt_len,
                            completion_tokens: req.completion_tokens,
                        });
                    }
                    return deferred;
                }
            };
            let mut retire = Vec::new();
            for (idx, report) in reports.into_iter().enumerate() {
                let req = &mut active[idx];
                let token_id = report.local_next_token_global_id;
                req.completion_tokens += 1;
                if let Err(err) = req.kv.apply_decode(token_id, &self.pool) {
                    let message = format!("Kimi-K2 decode KV bookkeeping failed: {err:#}");
                    error!("{message}");
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message,
                        prompt_tokens: req.prompt_len,
                        completion_tokens: req.completion_tokens,
                    });
                    retire.push(idx);
                    continue;
                }
                if !emit_sampled_token(
                    &req.token_tx,
                    &req.stop_policy,
                    &self.stop_token_ids,
                    token_id,
                    report.logprob,
                    req.prompt_len,
                    req.completion_tokens,
                    req.max_tokens,
                ) {
                    retire.push(idx);
                } else {
                    req.last_token = token_id;
                }
            }
            for idx in retire.into_iter().rev() {
                active.swap_remove(idx);
            }
        }
        deferred
    }

    /// Create the request's KV state: match the prompt prefix against the
    /// cache, then allocate pages for the uncached suffix. Returns the KV
    /// handle and the cached-token count. `None` means the request was
    /// failed (event already sent); allocation can only fail if the
    /// admission budget arithmetic is wrong.
    fn schedule_request_kv(&self, req: &GenerateRequest) -> Option<(RequestKv, usize)> {
        let mut kv = self
            .pool
            .new_request(req.prompt_tokens.clone(), req.max_tokens, None);
        let cached_tokens = match kv.match_and_add_prefix(&self.pool) {
            Ok(cached) => cached,
            Err(err) => {
                let message = format!("Kimi-K2 prefix cache matching failed: {err:#}");
                error!("{message}");
                let _ = req.token_tx.send(TokenEvent::Error {
                    message,
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
                return None;
            }
        };
        let suffix_len = req.prompt_tokens.len() - cached_tokens;
        if let Err(err) = kv.schedule_prefill(suffix_len, &self.pool) {
            let message = format!(
                "Kimi-K2 prefill KV block accounting violated full-lifetime reservation: {err}"
            );
            error!("{message}");
            let _ = req.token_tx.send(TokenEvent::Error {
                message,
                prompt_tokens: req.prompt_tokens.len(),
                completion_tokens: 0,
            });
            return None;
        }
        Some((kv, cached_tokens))
    }

    fn prefill_request(
        &mut self,
        req: GenerateRequest,
        slot: usize,
        decode_batch_size: usize,
    ) -> Option<ActiveKimiRequest> {
        let completion_tokens = 0usize;
        let (mut kv, cached_tokens) = self.schedule_request_kv(&req)?;
        let suffix_len = req.prompt_tokens.len() - cached_tokens;
        let kv_pages = KimiKvStepPages::single(
            kv.step_page_indices(suffix_len),
            self.pool.padding_block_id(),
        );
        let last_token = match self.executor.forward_prefill(
            &req.prompt_tokens[cached_tokens..],
            slot,
            decode_batch_size,
            cached_tokens,
            0,
            &kv_pages,
            row_options(&req),
            0,
        ) {
            Ok(report) => {
                let token_id = report.local_next_token_global_id;
                if let Err(err) = kv.apply_prefill(token_id, &self.pool) {
                    let message = format!("Kimi-K2 prefill KV bookkeeping failed: {err:#}");
                    error!("{message}");
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message,
                        prompt_tokens: req.prompt_tokens.len(),
                        completion_tokens: 0,
                    });
                    return None;
                }
                if !emit_sampled_token(
                    &req.token_tx,
                    &req.stop_policy,
                    &self.stop_token_ids,
                    token_id,
                    report.logprob,
                    req.prompt_tokens.len(),
                    1,
                    req.max_tokens,
                ) {
                    return None;
                }
                token_id
            }
            Err(err) => {
                let message = format!(
                    "Kimi-K2 prompt forward failed for slot {slot} after {}/{} ranks loaded: {err:#}",
                    self.executor.gpu_weight_ready_count(),
                    self.executor.worker_count()
                );
                let _ = req.token_tx.send(TokenEvent::Error {
                    message,
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens,
                });
                return None;
            }
        };
        let completion_tokens = completion_tokens + 1;
        let options = row_options(&req);
        Some(ActiveKimiRequest {
            token_tx: req.token_tx,
            stop_policy: req.stop_policy,
            prompt_len: req.prompt_tokens.len(),
            completion_tokens,
            max_tokens: req.max_tokens,
            last_token,
            slot,
            decode_batch_size,
            options,
            kv,
        })
    }

    fn decode_admission_microbatch(
        &mut self,
        group: Vec<(usize, GenerateRequest)>,
        decode_batch_size: usize,
        active: &mut Vec<ActiveKimiRequest>,
    ) {
        // 1-token prompts run their "prefill" through the decode path; the
        // KV lifecycle is still a prefill of one token. Prefix matching
        // always leaves ≥1 token uncached, so cached_tokens is 0 here.
        let mut group_kv = Vec::with_capacity(group.len());
        for (slot, req) in group {
            let Some((kv, _cached_tokens)) = self.schedule_request_kv(&req) else {
                continue;
            };
            group_kv.push((slot, req, kv));
        }
        if group_kv.is_empty() {
            return;
        }
        let token_ids = group_kv
            .iter()
            .map(|(_, req, _)| req.prompt_tokens[0])
            .collect::<Vec<_>>();
        let append_positions = vec![0; token_ids.len()];
        let slots = group_kv
            .iter()
            .map(|(slot, _, _)| *slot)
            .collect::<Vec<_>>();
        let rows = group_kv
            .iter()
            .map(|(_, req, _)| row_options(req))
            .collect::<Vec<_>>();
        let kv_pages = KimiKvStepPages::new(
            group_kv
                .iter()
                .map(|(_, _, kv)| kv.step_page_indices(1))
                .collect(),
            self.pool.padding_block_id(),
        );
        let reports = match self.executor.forward_decode_batch(
            &token_ids,
            &append_positions,
            &slots,
            decode_batch_size,
            &kv_pages,
            &rows,
            0,
        ) {
            Ok(reports) => reports,
            Err(err) => {
                let message = format!(
                    "Kimi-K2 decode admission forward failed after {}/{} ranks loaded: {err:#}",
                    self.executor.gpu_weight_ready_count(),
                    self.executor.worker_count()
                );
                error!("{message}");
                for (_, req, _) in group_kv {
                    let _ = req.token_tx.send(TokenEvent::Error {
                        message: message.clone(),
                        prompt_tokens: req.prompt_tokens.len(),
                        completion_tokens: 0,
                    });
                }
                return;
            }
        };
        for ((slot, req, mut kv), report) in group_kv.into_iter().zip(reports) {
            let token_id = report.local_next_token_global_id;
            if let Err(err) = kv.apply_prefill(token_id, &self.pool) {
                let message = format!("Kimi-K2 admission KV bookkeeping failed: {err:#}");
                error!("{message}");
                let _ = req.token_tx.send(TokenEvent::Error {
                    message,
                    prompt_tokens: req.prompt_tokens.len(),
                    completion_tokens: 0,
                });
                continue;
            }
            let completion_tokens = 1usize;
            if !emit_sampled_token(
                &req.token_tx,
                &req.stop_policy,
                &self.stop_token_ids,
                token_id,
                report.logprob,
                req.prompt_tokens.len(),
                completion_tokens,
                req.max_tokens,
            ) {
                continue;
            }
            let options = row_options(&req);
            active.push(ActiveKimiRequest {
                token_tx: req.token_tx,
                stop_policy: req.stop_policy,
                prompt_len: req.prompt_tokens.len(),
                completion_tokens,
                max_tokens: req.max_tokens,
                last_token: token_id,
                slot,
                decode_batch_size,
                options,
                kv,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;

    use pegainfer_frontend::engine::EosPolicy;
    use pegainfer_frontend::engine::StopCause;
    use pegainfer_frontend::sampler::SamplingParams;

    use super::*;
    use crate::runner::worker::KimiOneTokenForwardReport;

    #[derive(Debug, Eq, PartialEq)]
    enum ForwardCall {
        EnsureDecodeBatch(usize),
        Prefill {
            input_ids: Vec<u32>,
            slot: usize,
            decode_batch_size: usize,
            cached_tokens: usize,
            kv_row_pages: Vec<usize>,
        },
        Decode {
            token_ids: Vec<u32>,
            append_positions: Vec<usize>,
            slots: Vec<usize>,
            decode_batch_size: usize,
            kv_row_pages: Vec<usize>,
        },
    }

    struct RecordingExecutor {
        calls: Arc<Mutex<Vec<ForwardCall>>>,
    }

    impl RecordingExecutor {
        fn new(calls: Arc<Mutex<Vec<ForwardCall>>>) -> Self {
            Self { calls }
        }
    }

    fn kv_row_pages(kv_pages: &KimiKvStepPages) -> Vec<usize> {
        (0..kv_pages.rows())
            .map(|row| kv_pages.row(row).expect("CSR row").len())
            .collect()
    }

    impl ForwardExecutor for RecordingExecutor {
        fn ensure_decode_batch(&self, decode_batch_size: usize) -> Result<()> {
            self.calls
                .lock()
                .unwrap()
                .push(ForwardCall::EnsureDecodeBatch(decode_batch_size));
            Ok(())
        }

        fn forward_prefill(
            &self,
            input_ids: &[u32],
            slot: usize,
            decode_batch_size: usize,
            cached_tokens: usize,
            _ep_max_seq_len: usize,
            kv_pages: &KimiKvStepPages,
            _row: KimiRowOptions,
            _seed: u64,
        ) -> Result<KimiOneTokenForwardReport> {
            self.calls.lock().unwrap().push(ForwardCall::Prefill {
                input_ids: input_ids.to_vec(),
                slot,
                decode_batch_size,
                cached_tokens,
                kv_row_pages: kv_row_pages(kv_pages),
            });
            Ok(report(slot, *input_ids.last().unwrap(), 1000 + slot as u32))
        }

        fn forward_decode_batch(
            &self,
            token_ids: &[u32],
            append_positions: &[usize],
            slots: &[usize],
            decode_batch_size: usize,
            kv_pages: &KimiKvStepPages,
            _rows: &[KimiRowOptions],
            _seed: u64,
        ) -> Result<Vec<KimiOneTokenForwardReport>> {
            self.calls.lock().unwrap().push(ForwardCall::Decode {
                token_ids: token_ids.to_vec(),
                append_positions: append_positions.to_vec(),
                slots: slots.to_vec(),
                decode_batch_size,
                kv_row_pages: kv_row_pages(kv_pages),
            });
            Ok(token_ids
                .iter()
                .zip(slots)
                .enumerate()
                .map(|(row, (token_id, slot))| report(*slot, *token_id, 2000 + row as u32))
                .collect())
        }

        fn worker_count(&self) -> usize {
            1
        }

        fn gpu_weight_ready_count(&self) -> usize {
            1
        }
    }

    fn report(slot: usize, input_token_id: u32, next_token_id: u32) -> KimiOneTokenForwardReport {
        KimiOneTokenForwardReport {
            rank: 0,
            batch_slot: slot,
            input_token_id,
            local_next_token_id: next_token_id,
            local_next_token_global_id: next_token_id,
            local_top_logit_f32: 0.0,
            vocab_start: 0,
            vocab_rows: 1,
            dense_layers_executed: 0,
            moe_layers_executed: 0,
            logprob: None,
        }
    }

    fn test_pool() -> BlockPool {
        BlockPool::new(16, 1024).expect("test pool")
    }

    fn test_scheduler(calls: &Arc<Mutex<Vec<ForwardCall>>>, pool: BlockPool) -> KimiK2Scheduler {
        KimiK2Scheduler {
            executor: Box::new(RecordingExecutor::new(Arc::clone(calls))),
            stop_token_ids: Vec::new(),
            pool,
        }
    }

    fn request(prompt_tokens: Vec<u32>) -> GenerateRequest {
        request_with_channel(prompt_tokens, 1).0
    }

    fn request_with_channel(
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> (
        GenerateRequest,
        pegainfer_frontend::engine::TokenStreamReceiver,
    ) {
        let (token_tx, token_rx) = TokenSink::standalone();
        let req = GenerateRequest {
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
        };
        (req, token_rx)
    }

    #[test]
    fn sampled_model_eos_is_emitted_before_stop_finish() {
        let (req, mut token_rx) = request_with_channel(vec![11, 22], 4);
        let logprob = TokenLogprob {
            logprob: -0.5,
            top_logprobs: vec![(99, -0.5)],
        };

        assert!(!emit_sampled_token(
            &req.token_tx,
            &req.stop_policy,
            &[99],
            99,
            Some(logprob.clone()),
            2,
            1,
            req.max_tokens,
        ));

        let Ok((_, TokenEvent::Token { id, logprob: seen })) = token_rx.try_recv() else {
            panic!("expected trigger Token event");
        };
        assert_eq!(id, 99);
        assert_eq!(seen, Some(logprob));
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
            panic!("expected terminal Finished event");
        };
        assert_eq!(finish_reason, FinishReason::Stop);
        assert_eq!(stop_cause, Some(StopCause::Eos(99)));
        assert_eq!(completion_tokens, 1);
    }

    #[test]
    fn ignored_eos_still_honors_request_stop_token() {
        let (mut req, mut token_rx) = request_with_channel(vec![11, 22], 4);
        req.stop_policy = StopPolicy {
            eos: EosPolicy::Ignore,
            token_ids: vec![99],
        };

        assert!(!emit_sampled_token(
            &req.token_tx,
            &req.stop_policy,
            &[99],
            99,
            None,
            2,
            1,
            req.max_tokens,
        ));
        assert!(matches!(
            token_rx.try_recv(),
            Ok((_, TokenEvent::Token { id: 99, .. }))
        ));
        let Ok((_, TokenEvent::Finished { stop_cause, .. })) = token_rx.try_recv() else {
            panic!("expected terminal Finished event");
        };
        assert_eq!(stop_cause, Some(StopCause::Token(99)));
    }

    #[test]
    fn mixed_prompt_batch_routes_single_token_requests_to_decode() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut scheduler = test_scheduler(&calls, test_pool());

        let deferred = scheduler.handle_request_batch(vec![
            request(vec![11]),
            request(vec![22, 33]),
            request(vec![44]),
        ]);

        assert!(deferred.is_empty());
        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                ForwardCall::EnsureDecodeBatch(3),
                ForwardCall::Decode {
                    token_ids: vec![11],
                    append_positions: vec![0],
                    slots: vec![0],
                    decode_batch_size: 3,
                    kv_row_pages: vec![1],
                },
                ForwardCall::Prefill {
                    input_ids: vec![22, 33],
                    slot: 1,
                    decode_batch_size: 3,
                    cached_tokens: 0,
                    kv_row_pages: vec![1],
                },
                ForwardCall::Decode {
                    token_ids: vec![44],
                    append_positions: vec![0],
                    slots: vec![2],
                    decode_batch_size: 3,
                    kv_row_pages: vec![1],
                },
            ]
        );
    }

    #[test]
    fn tp8_scheduler_rejects_sampling_requests_before_forward() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut scheduler = test_scheduler(&calls, test_pool());

        let (mut sampling_req, mut token_rx) = request_with_channel(vec![11, 22], 4);
        sampling_req.params = SamplingParams {
            temperature: 0.8,
            top_k: -1,
            top_p: 0.9,
            min_p: 0.0,
            seed: None,
            ignore_eos: false,
        };

        scheduler.handle_request_batch(vec![sampling_req]);

        assert!(
            calls.lock().unwrap().is_empty(),
            "a rejected sampling request must not reach the executor"
        );
        // Scheduled event precedes the rejection.
        let Ok((_, TokenEvent::Scheduled { .. })) = token_rx.try_recv() else {
            panic!("expected Scheduled event");
        };
        let Ok((_, TokenEvent::Rejected { message, .. })) = token_rx.try_recv() else {
            panic!("expected Rejected event");
        };
        assert!(
            message.contains("TP8"),
            "rejection names the path: {message}"
        );
    }

    #[test]
    fn echo_request_is_rejected_before_forward() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut scheduler = test_scheduler(&calls, test_pool());

        let (mut echo_req, mut token_rx) = request_with_channel(vec![11, 22], 4);
        echo_req.echo = true;

        scheduler.handle_request_batch(vec![echo_req]);

        assert!(
            calls.lock().unwrap().is_empty(),
            "a rejected echo request must not reach the executor"
        );
        let Ok((_, TokenEvent::Scheduled { .. })) = token_rx.try_recv() else {
            panic!("expected Scheduled event");
        };
        let Ok((_, TokenEvent::Rejected { message, .. })) = token_rx.try_recv() else {
            panic!("expected Rejected event");
        };
        assert!(
            message.contains("echo"),
            "rejection names the unsupported field: {message}"
        );
    }

    #[test]
    fn over_capacity_request_is_rejected_before_forward() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut scheduler = test_scheduler(&calls, test_pool());

        // prompt (2) + max_tokens (KIMI_MAX_REQUEST_TOKENS) - 1 exceeds the
        // per-request KV capacity by one token.
        let (req, mut token_rx) =
            request_with_channel(vec![11, 22], crate::runner::worker::KIMI_MAX_REQUEST_TOKENS);

        let deferred = scheduler.handle_request_batch(vec![req]);

        assert!(
            deferred.is_empty(),
            "over-cap is a rejection, not a deferral"
        );
        assert!(
            calls.lock().unwrap().is_empty(),
            "a rejected over-cap request must not reach the executor"
        );
        let Ok((_, TokenEvent::Scheduled { .. })) = token_rx.try_recv() else {
            panic!("expected Scheduled event");
        };
        let Ok((_, TokenEvent::Rejected { message, .. })) = token_rx.try_recv() else {
            panic!("expected Rejected event");
        };
        assert!(
            message.contains("per-request capacity"),
            "rejection names the limit: {message}"
        );
    }

    #[test]
    fn pool_budget_defers_requests_to_the_next_wave() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        // 4 pages total, 1 reserved for padding: budget = 3 blocks.
        let mut scheduler = test_scheduler(&calls, BlockPool::new(16, 4).expect("tiny pool"));

        // 33-token prompt + max_tokens 1 → lifetime ceil(34/16) = 3 blocks:
        // admitted, budget drained to zero.
        let (big, _big_rx) = request_with_channel((0..33).collect(), 1);
        // 1 more block needed: over budget, deferred without any event.
        let (small, mut small_rx) = request_with_channel(vec![5], 1);

        let deferred = scheduler.handle_request_batch(vec![big, small]);

        assert_eq!(deferred.len(), 1);
        assert_eq!(deferred[0].prompt_tokens, vec![5]);
        assert!(
            small_rx.try_recv().is_err(),
            "deferral is silent: the request just waits for the next wave"
        );
        assert_eq!(
            *calls.lock().unwrap(),
            vec![
                ForwardCall::EnsureDecodeBatch(1),
                ForwardCall::Prefill {
                    input_ids: (0..33).collect(),
                    slot: 0,
                    decode_batch_size: 1,
                    cached_tokens: 0,
                    kv_row_pages: vec![3],
                },
            ]
        );
    }

    #[test]
    fn prefix_cache_hit_prefills_only_the_suffix() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let mut scheduler = test_scheduler(&calls, test_pool());

        // First request: a 33-token prompt fills two complete 16-token
        // blocks, which register in the prefix cache at apply_prefill.
        let prompt_a: Vec<u32> = (100..133).collect();
        scheduler.handle_request_batch(vec![request(prompt_a.clone())]);

        // Second request shares the block-aligned 32-token prefix.
        let mut prompt_b: Vec<u32> = prompt_a[..32].to_vec();
        prompt_b.extend([900, 901]);
        scheduler.handle_request_batch(vec![request(prompt_b.clone())]);

        let calls = calls.lock().unwrap();
        let prefills: Vec<_> = calls
            .iter()
            .filter(|call| matches!(call, ForwardCall::Prefill { .. }))
            .collect();
        let [
            ForwardCall::Prefill {
                input_ids: cold_ids,
                cached_tokens: cold_cached,
                ..
            },
            ForwardCall::Prefill {
                input_ids: warm_ids,
                cached_tokens: warm_cached,
                kv_row_pages: warm_pages,
                ..
            },
        ] = prefills[..]
        else {
            panic!("expected two prefill forwards, got {calls:?}");
        };
        assert_eq!(*cold_cached, 0);
        assert_eq!(*cold_ids, prompt_a);
        assert_eq!(*warm_cached, 32, "two full blocks hit the prefix cache");
        assert_eq!(*warm_ids, prompt_b[32..], "only the suffix is forwarded");
        assert_eq!(
            *warm_pages,
            vec![3],
            "page row covers cached prefix + suffix"
        );
    }
}
