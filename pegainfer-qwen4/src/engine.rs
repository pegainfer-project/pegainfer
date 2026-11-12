//! Serving: one scheduler over the model split across every visible card. A
//! running sequence holds one slot (its K/V, GDN state and n-gram rows on
//! every card). Each step admits at most one prompt, prefilled alone with its
//! first token picked from the last row, then decodes every running slot as
//! one batch. Attention runs dense, which is the reference's QSA only while a
//! sequence fits the selection budget, so a request whose prompt and
//! completion could pass that length is refused.

use std::collections::VecDeque;
use std::path::Path;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use cudarc::driver::CudaContext;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineInfo;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::QueuedRequest;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::RequestLedger;
use pegainfer_frontend::engine::Scheduler;
use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_frontend::engine::StopPolicy;
use pegainfer_frontend::engine::spawn_scheduler;
use pegainfer_frontend::sampler::SamplingParams;

use crate::model::split::Output;
use crate::model::split::Readout;
use crate::model::split::Sampling;
use crate::model::split::SplitModel;
use crate::weights::Geometry;
use crate::weights::load_config;

/// Sequences decoded together.
const SLOTS: usize = 8;

struct Running {
    id: RequestId,
    params: SamplingParams,
    stop: StopPolicy,
    max_tokens: usize,
    /// Tokens picked so far; the last one is the next step's input.
    produced: usize,
    last: u32,
}

pub struct Qwen4Scheduler {
    model: SplitModel,
    eos: Vec<u32>,
    /// Prompt and completion tokens one sequence may reach.
    limit: usize,
    slots: Vec<Option<Running>>,
    waiting: VecDeque<QueuedRequest>,
    /// Sampler calls so far, mixed into each call's seed.
    calls: u64,
}

impl Qwen4Scheduler {
    fn next_seed(&mut self) -> u64 {
        self.calls = self.calls.wrapping_add(1);
        self.calls.rotate_left(17)
    }

    /// Why this engine cannot serve `request`, if it cannot.
    fn refusal(&self, request: &Request) -> Option<RejectReason> {
        let unsupported = |feature: &str| {
            Some(RejectReason::Unsupported {
                feature: feature.to_string(),
            })
        };
        if request.prompt_tokens.is_empty() {
            return unsupported("an empty prompt");
        }
        if request.logprobs.is_some() {
            return unsupported("logprobs");
        }
        if request.prompt_logprobs.is_some() {
            return unsupported("prompt logprobs");
        }
        if request.lora_adapter.is_some() {
            return unsupported("LoRA adapters");
        }
        if request.prompt_tokens.len() + request.max_tokens > self.limit {
            return Some(RejectReason::ContextLength {
                prompt_tokens: request.prompt_tokens.len(),
                max_tokens: request.max_tokens,
                limit: self.limit,
            });
        }
        None
    }

    /// Record the token picked for `slot`'s sequence and free the slot when
    /// the sequence stops.
    fn emit(&mut self, ledger: &mut RequestLedger, slot: usize, token: u32) {
        let running = self.slots[slot].as_mut().expect("a running slot");
        ledger.push_tokens(running.id, &[token], &[]);
        running.produced += 1;
        running.last = token;
        let eos = &self.eos;
        let finish = match running.stop.classify(token, |id| eos.contains(&id)) {
            Some(cause) => Some((FinishReason::Stop, Some(cause))),
            None if running.produced >= running.max_tokens => Some((FinishReason::Length, None)),
            None => None,
        };
        if let Some((reason, cause)) = finish {
            ledger.finish_with_cause(running.id, reason, cause);
            self.slots[slot] = None;
        }
    }

    /// Admit the first waiting request that can run into a free slot, and
    /// prefill it.
    fn admit(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        let Some(slot) = self.slots.iter().position(Option::is_none) else {
            return Ok(());
        };
        while let Some(QueuedRequest { id, request }) = self.waiting.pop_front() {
            if ledger.is_aborted(id) {
                ledger.retire(id);
                continue;
            }
            if let Some(reason) = self.refusal(&request) {
                ledger.reject(id, reason);
                continue;
            }
            ledger.admit(id);
            if request.max_tokens == 0 {
                ledger.finish(id, FinishReason::Length);
                continue;
            }
            self.model.reset(slot)?;
            let sampling = Sampling {
                params: vec![request.params],
                steps: vec![0],
                seed: self.next_seed(),
            };
            let Output::Tokens(tokens) =
                self.model
                    .prefill(slot, &request.prompt_tokens, Readout::Sample(sampling))?
            else {
                bail!("the cards handed back a stream for a prompt");
            };
            self.slots[slot] = Some(Running {
                id,
                params: request.params,
                stop: request.stop_policy,
                max_tokens: request.max_tokens,
                produced: 0,
                last: 0,
            });
            self.emit(ledger, slot, *tokens.first().context("a token per prompt")?);
            return Ok(());
        }
        Ok(())
    }

    /// One token for every running slot, as one batch.
    fn decode(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        let slots: Vec<usize> = (0..self.slots.len())
            .filter(|&slot| self.slots[slot].is_some())
            .collect();
        if slots.is_empty() {
            return Ok(());
        }
        let running = || {
            slots
                .iter()
                .map(|&slot| self.slots[slot].as_ref().expect("running"))
        };
        let ids: Vec<u32> = running().map(|r| r.last).collect();
        let params: Vec<SamplingParams> = running().map(|r| r.params).collect();
        let steps: Vec<u64> = running().map(|r| r.produced as u64).collect();
        let sampling = Sampling {
            params,
            steps,
            seed: self.next_seed(),
        };
        let Output::Tokens(tokens) = self.model.decode(&slots, &ids, Readout::Sample(sampling))?
        else {
            bail!("the cards handed back a stream for a decode step");
        };
        for (&slot, token) in slots.iter().zip(tokens) {
            self.emit(ledger, slot, token);
        }
        Ok(())
    }
}

impl Scheduler for Qwen4Scheduler {
    fn submit(&mut self, request: QueuedRequest) {
        self.waiting.push_back(request);
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        for slot in &mut self.slots {
            if let Some(running) = slot.take_if(|running| ledger.is_aborted(running.id)) {
                ledger.retire(running.id);
            }
        }
        self.admit(ledger)?;
        self.decode(ledger)
    }

    fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            num_running_reqs: self.slots.iter().flatten().count() as u64,
            num_waiting_reqs: self.waiting.len() as u64,
            ..SchedulerMetrics::default()
        }
    }
}

/// Load every layer across every visible card and start the scheduler.
pub fn launch(model_path: &Path) -> Result<Engine> {
    let config = load_config(model_path)?;
    let g = Geometry::from_config(&config);
    let cards = CudaContext::device_count().context("counting the visible cards")?;
    let devices: Vec<usize> = (0..usize::try_from(cards)?).collect();
    let layers: Vec<usize> = (0..config.layer_kinds.len()).collect();
    let limit = g.dense_attention_rows();
    log::info!(
        "qwen4: {} layers over {} cards, {SLOTS} slots of {limit} tokens",
        layers.len(),
        devices.len()
    );
    let model = SplitModel::load(model_path, &devices, &layers, SLOTS, limit)?;
    let eos = pegainfer_core::eos::eos_token_ids(model_path, g.vocab)?;
    let scheduler = Qwen4Scheduler {
        model,
        eos,
        limit,
        slots: (0..SLOTS).map(|_| None).collect(),
        waiting: VecDeque::new(),
        calls: 0,
    };
    Ok(Engine {
        schedulers: vec![spawn_scheduler("qwen4-scheduler", scheduler)],
        info: EngineInfo {
            kv_capacity: None,
            servable_len: Some(u32::try_from(limit)?),
        },
        lora: None,
    })
}
