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
use super::TokenInput;
struct Active {
    id: RequestId,
    request: Request,
    slot: usize,
    cursor: usize,
    generated: usize,
    last: u32,
}
pub(super) struct RocmScheduler {
    model: Model,
    waiting: VecDeque<QueuedRequest>,
    active: Vec<Active>,
    free: Vec<usize>,
}
impl RocmScheduler {
    pub(super) fn new(model: Model) -> Self {
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
impl Scheduler for RocmScheduler {
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
