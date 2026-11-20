use std::collections::VecDeque;

use anyhow::Result;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::QueuedRequest;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::RequestLedger;
use pegainfer_frontend::engine::Scheduler;
use pegainfer_frontend::engine::SchedulerMetrics;

use super::Model;
use super::ModelOptions;
use super::TokenInput;

// The scheduler contract can be exercised without initializing a GPU, including
// cancellation during forward and fatal execution errors. Numeric qualification
// continues to use the real Model and the independent HF golden oracle.
pub(super) trait Executor: Send {
    fn options(&self) -> ModelOptions;
    fn vocab_size(&self) -> usize;
    fn eos_ids(&self) -> &[u32];
    fn reset_slot(&mut self, slot: usize) -> Result<()>;
    fn forward(&mut self, input: &[TokenInput], outputs: &[usize]) -> Result<Vec<u32>>;
}

impl Executor for Model {
    fn options(&self) -> ModelOptions {
        Self::options(self)
    }
    fn vocab_size(&self) -> usize {
        Self::vocab_size(self)
    }
    fn eos_ids(&self) -> &[u32] {
        Self::eos_ids(self)
    }
    fn reset_slot(&mut self, slot: usize) -> Result<()> {
        Self::reset_slot(self, slot)
    }
    fn forward(&mut self, input: &[TokenInput], outputs: &[usize]) -> Result<Vec<u32>> {
        Self::forward(self, input, outputs)
    }
}
struct Active {
    id: RequestId,
    request: Request,
    slot: usize,
    cursor: usize,
    generated: usize,
    last: u32,
}
pub(super) struct RocmScheduler<M = Model> {
    model: M,
    waiting: VecDeque<QueuedRequest>,
    active: Vec<Active>,
    free: Vec<usize>,
}
impl<M: Executor> RocmScheduler<M> {
    pub(super) fn new(model: M) -> Self {
        let slots = model.options().slots;
        Self {
            model,
            waiting: VecDeque::new(),
            active: Vec::new(),
            free: (0..slots).rev().collect(),
        }
    }
    fn unsupported(&self, r: &Request) -> Option<String> {
        if !r.params.is_greedy() {
            return Some("ROCm first release supports greedy sampling (temperature=0)".into());
        }
        if r.logprobs.is_some() || r.prompt_logprobs.is_some() {
            return Some(
                "ROCm first release does not support HTTP logprobs/prompt echo scoring".into(),
            );
        }
        if r.lora_adapter.is_some() || r.kv_transfer_params.is_some() {
            return Some("ROCm first release does not support LoRA or KV transfer".into());
        }
        if r.prompt_tokens.is_empty() || r.max_tokens == 0 {
            return Some("empty prompt or max_tokens=0".into());
        }
        if r.prompt_tokens
            .iter()
            .any(|&x| x as usize >= self.model.vocab_size())
        {
            return Some("prompt token outside model vocabulary".into());
        }
        None
    }
}
impl<M: Executor> Scheduler for RocmScheduler<M> {
    fn submit(&mut self, r: QueuedRequest) {
        self.waiting.push_back(r);
    }
    fn metrics(&self) -> SchedulerMetrics {
        let o = self.model.options();
        SchedulerMetrics {
            num_running_reqs: self.active.len() as _,
            num_waiting_reqs: self.waiting.len() as _,
            kv_used_blocks: (self.active.len() * o.max_context) as _,
            kv_total_blocks: (o.slots * o.max_context) as _,
            ..Default::default()
        }
    }
    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        let mut keep = Vec::new();
        for a in self.active.drain(..) {
            if ledger.is_aborted(a.id) {
                ledger.retire(a.id);
                self.model.reset_slot(a.slot)?;
                self.free.push(a.slot);
            } else {
                keep.push(a);
            }
        }
        self.active = keep;
        let mut waiting = VecDeque::new();
        while let Some(q) = self.waiting.pop_front() {
            if ledger.is_aborted(q.id) {
                ledger.retire(q.id);
                continue;
            }
            if let Some(feature) = self.unsupported(&q.request) {
                ledger.reject(q.id, RejectReason::Unsupported { feature });
                continue;
            }
            let r = &q.request;
            let limit = self.model.options().max_context;
            if r.prompt_tokens.len().saturating_add(r.max_tokens) > limit {
                ledger.reject(
                    q.id,
                    RejectReason::ContextLength {
                        prompt_tokens: r.prompt_tokens.len(),
                        max_tokens: r.max_tokens,
                        limit,
                    },
                );
                continue;
            }
            if let Some(slot) = self.free.pop() {
                ledger.admit(q.id);
                self.active.push(Active {
                    id: q.id,
                    request: q.request,
                    slot,
                    cursor: 0,
                    generated: 0,
                    last: 0,
                });
            } else {
                waiting.push_back(q);
            }
        }
        self.waiting = waiting;
        if self.active.is_empty() {
            return Ok(());
        }
        let cap = self
            .model
            .options()
            .chunk_size
            .max(self.model.options().slots);
        let mut input = Vec::new();
        let mut outputs = Vec::new();
        let mut owners = Vec::new();
        // Decode rows first, then fair shares of prefill. All selected rows execute
        // in a single ragged batch; causal masking uses each row's own position.
        for (idx, a) in self.active.iter().enumerate() {
            if a.cursor == a.request.prompt_tokens.len() {
                outputs.push(input.len());
                owners.push(idx);
                input.push(TokenInput {
                    token: a.last,
                    slot: a.slot,
                    position: a.cursor + a.generated - 1,
                });
            }
        }
        let prefills = self
            .active
            .iter()
            .filter(|a| a.cursor < a.request.prompt_tokens.len())
            .count();
        let share = (cap - input.len()).checked_div(prefills).unwrap_or(0);
        for (idx, a) in self.active.iter_mut().enumerate() {
            if a.cursor == a.request.prompt_tokens.len() {
                continue;
            }
            let take = share
                .max(1)
                .min(a.request.prompt_tokens.len() - a.cursor)
                .min(cap - input.len());
            for pos in a.cursor..a.cursor + take {
                input.push(TokenInput {
                    token: a.request.prompt_tokens[pos],
                    slot: a.slot,
                    position: pos,
                });
            }
            a.cursor += take;
            if take > 0 && a.cursor == a.request.prompt_tokens.len() {
                outputs.push(input.len() - 1);
                owners.push(idx);
            }
        }
        // A pure intermediate prefill still advances state; discard its one scored row.
        let scoring = if outputs.is_empty() {
            vec![input.len() - 1]
        } else {
            outputs.clone()
        };
        let tokens = self.model.forward(&input, &scoring)?;
        let mut finished = vec![false; self.active.len()];
        for (token, idx) in tokens.into_iter().zip(owners) {
            let a = &mut self.active[idx];
            if ledger.is_aborted(a.id) {
                ledger.retire(a.id);
                finished[idx] = true;
                continue;
            }
            a.last = token;
            a.generated += 1;
            ledger.push_tokens(a.id, &[token], &[]);
            let stop = a
                .request
                .stop_policy
                .classify(token, |id| self.model.eos_ids().contains(&id));
            if let Some(cause) = stop {
                ledger.finish_with_cause(a.id, FinishReason::Stop, Some(cause));
                finished[idx] = true;
            } else if a.generated >= a.request.max_tokens {
                ledger.finish(a.id, FinishReason::Length);
                finished[idx] = true;
            }
        }
        let mut keep = Vec::new();
        for (idx, a) in self.active.drain(..).enumerate() {
            if finished[idx] {
                self.model.reset_slot(a.slot)?;
                self.free.push(a.slot);
            } else {
                keep.push(a);
            }
        }
        self.active = keep;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Mutex;
    use std::time::Duration;
    use std::time::Instant;

    use pegainfer_frontend::engine::RequestControl;
    use pegainfer_frontend::engine::StepOutputs;
    use pegainfer_frontend::engine::StopPolicy;
    use pegainfer_frontend::engine::Terminal;
    use pegainfer_frontend::engine::drive;
    use pegainfer_frontend::engine::scheduler_pair;

    use super::*;

    #[derive(Default)]
    struct Observed {
        batches: Vec<Vec<(usize, usize)>>,
        resets: Vec<usize>,
        abort_during_forward: Option<Arc<RequestControl>>,
        fail: bool,
    }
    struct TestExecutor {
        options: ModelOptions,
        observed: Arc<Mutex<Observed>>,
        lengths: Vec<usize>,
    }
    impl Executor for TestExecutor {
        fn options(&self) -> ModelOptions {
            self.options
        }
        fn vocab_size(&self) -> usize {
            128
        }
        fn eos_ids(&self) -> &[u32] {
            &[99]
        }
        fn reset_slot(&mut self, slot: usize) -> Result<()> {
            self.observed.lock().unwrap().resets.push(slot);
            self.lengths[slot] = 0;
            Ok(())
        }
        fn forward(&mut self, input: &[TokenInput], outputs: &[usize]) -> Result<Vec<u32>> {
            let mut state = self.observed.lock().unwrap();
            if let Some(control) = state.abort_during_forward.take() {
                control.abort();
            }
            if state.fail {
                anyhow::bail!("injected forward failure");
            }
            assert!(!input.is_empty());
            assert!(input.len() <= self.options.chunk_size.max(self.options.slots));
            for token in input {
                assert_eq!(
                    token.position, self.lengths[token.slot],
                    "noncontiguous request KV"
                );
                self.lengths[token.slot] += 1;
            }
            assert!(outputs.iter().all(|&row| row < input.len()));
            state
                .batches
                .push(input.iter().map(|row| (row.slot, row.position)).collect());
            Ok(vec![42; outputs.len()])
        }
    }
    fn request(prompt: usize, max_tokens: usize) -> Request {
        Request {
            prompt_tokens: vec![3; prompt],
            params: pegainfer_frontend::sampler::SamplingParams::default(),
            history_tokens: None,
            history_marks: vec![],
            stop_policy: StopPolicy::default(),
            max_tokens,
            lora_adapter: None,
            kv_transfer_params: None,
            logprobs: None,
            prompt_logprobs: None,
            trace_parent: None,
            client_label: None,
        }
    }
    fn executor(
        slots: usize,
        chunk_size: usize,
        max_context: usize,
    ) -> (TestExecutor, Arc<Mutex<Observed>>) {
        let state = Arc::new(Mutex::new(Observed::default()));
        (
            TestExecutor {
                options: ModelOptions {
                    slots,
                    chunk_size,
                    max_context,
                    device: 0,
                },
                observed: state.clone(),
                lengths: vec![0; slots],
            },
            state,
        )
    }
    fn receive(receiver: &mut tokio::sync::mpsc::UnboundedReceiver<StepOutputs>) -> StepOutputs {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Ok(step) = receiver.try_recv() {
                return step;
            }
            assert!(Instant::now() < deadline, "scheduler made no progress");
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    #[test]
    fn rejects_unsupported_and_over_context_requests_before_forward() {
        let (model, observed) = executor(1, 2, 8);
        let (mut handle, backend) = scheduler_pair();
        let mut steps = handle.take_steps().unwrap();
        let mut invalid = request(1, 1);
        invalid.params.temperature = 0.5;
        handle.submit(invalid);
        handle.submit(request(8, 1));
        handle.submit(request(0, 1));
        handle.submit(request(1, 0));
        let join = std::thread::spawn(move || drive(RocmScheduler::new(model), backend));
        let mut rejected = 0;
        while rejected < 4 {
            for update in receive(&mut steps).updates {
                assert!(update.tokens.is_empty());
                if let Some(terminal) = update.terminal {
                    assert!(matches!(terminal, Terminal::Rejected { .. }));
                    rejected += 1;
                }
            }
        }
        drop(handle);
        join.join().unwrap();
        assert!(observed.lock().unwrap().batches.is_empty());
    }
    #[test]
    fn ragged_chunks_decode_and_waiting_requests_reuse_slots() {
        let (model, observed) = executor(2, 3, 32);
        let (mut handle, backend) = scheduler_pair();
        let mut steps = handle.take_steps().unwrap();
        for prompt in [1, 11, 3, 7] {
            handle.submit(request(prompt, 3));
        }
        let join = std::thread::spawn(move || drive(RocmScheduler::new(model), backend));
        let (mut finished, mut tokens) = (0, 0);
        while finished < 4 {
            for update in receive(&mut steps).updates {
                tokens += update.tokens.len();
                if let Some(terminal) = update.terminal {
                    assert!(matches!(
                        terminal,
                        Terminal::Finished {
                            reason: FinishReason::Length,
                            ..
                        }
                    ));
                    finished += 1;
                }
            }
        }
        drop(handle);
        join.join().unwrap();
        let state = observed.lock().unwrap();
        assert_eq!(tokens, 12);
        assert_eq!(state.resets.len(), 4);
        assert!(state.batches.iter().any(|batch| batch.len() == 3));
        assert!(
            state
                .batches
                .iter()
                .flatten()
                .any(|&(_, position)| position >= 11)
        );
    }
    #[test]
    fn cancellation_before_and_during_forward_reclaims_slot() {
        let (model, observed) = executor(1, 2, 32);
        let (mut handle, backend) = scheduler_pair();
        let mut steps = handle.take_steps().unwrap();
        let before = handle.submit(request(4, 2));
        before.abort();
        let during = Arc::new(handle.submit(request(2, 2)));
        observed.lock().unwrap().abort_during_forward = Some(during.clone());
        let normal = handle.submit(request(2, 2));
        let join = std::thread::spawn(move || drive(RocmScheduler::new(model), backend));
        loop {
            let step = receive(&mut steps);
            assert!(
                step.updates
                    .iter()
                    .all(|update| update.id != before.id() && update.id != during.id())
            );
            if step
                .updates
                .iter()
                .any(|update| update.id == normal.id() && update.terminal.is_some())
            {
                break;
            }
        }
        drop(handle);
        join.join().unwrap();
        assert_eq!(observed.lock().unwrap().resets, [0, 0]);
    }
    #[test]
    fn fatal_forward_failure_answers_active_and_waiting_accounts() {
        let (model, observed) = executor(1, 2, 32);
        observed.lock().unwrap().fail = true;
        let (mut handle, backend) = scheduler_pair();
        let mut steps = handle.take_steps().unwrap();
        handle.submit(request(2, 2));
        handle.submit(request(2, 2));
        let join = std::thread::spawn(move || drive(RocmScheduler::new(model), backend));
        let mut failed = 0;
        while failed < 2 {
            for update in receive(&mut steps).updates {
                assert!(update.tokens.is_empty());
                if let Some(terminal) = update.terminal {
                    assert!(matches!(terminal, Terminal::Failed { .. }));
                    failed += 1;
                }
            }
        }
        drop(handle);
        join.join().unwrap();
    }
}
