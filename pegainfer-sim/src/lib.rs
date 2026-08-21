use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use anyhow::ensure;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineInfo;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::PromptEcho;
use pegainfer_frontend::engine::QueuedRequest;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::RequestLedger;
use pegainfer_frontend::engine::Scheduler;
use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::engine::spawn_scheduler;

/// Cap on how long `step` parks while waiting for the next due token. New
/// submissions only drain between steps, so a full TTFT/TPOT sleep would
/// stall admission; 1ms keeps the CPU-only sim from spinning a core.
const WAIT_SLICE: Duration = Duration::from_millis(1);

#[derive(Clone, Debug)]
pub struct SimulatedEngineConfig {
    base_ttft_ms: f64,
    prefill_tokens_per_ms: f64,
    tpot_ms: f64,
    fallback_token_id: u32,
    /// Explicit completion token-id sequence to replay verbatim. Empty (the
    /// default) keeps the legacy behaviour of cycling the prompt tokens.
    scripted_completion: Vec<u32>,
}

impl SimulatedEngineConfig {
    pub fn new(
        base_ttft_ms: f64,
        prefill_tokens_per_ms: f64,
        tpot_ms: f64,
        fallback_token_id: u32,
    ) -> Result<Self> {
        ensure!(
            base_ttft_ms.is_finite() && base_ttft_ms >= 0.0,
            "base TTFT must be finite and non-negative"
        );
        ensure!(
            prefill_tokens_per_ms.is_finite() && prefill_tokens_per_ms > 0.0,
            "prefill throughput must be finite and positive"
        );
        ensure!(
            tpot_ms.is_finite() && tpot_ms >= 0.0,
            "TPOT must be finite and non-negative"
        );

        Ok(Self {
            base_ttft_ms,
            prefill_tokens_per_ms,
            tpot_ms,
            fallback_token_id,
            scripted_completion: Vec::new(),
        })
    }

    /// Replay `ids` verbatim as the completion for every request
    #[must_use]
    pub fn with_scripted_completion(mut self, ids: Vec<u32>) -> Self {
        self.scripted_completion = ids;
        self
    }

    fn ttft(&self, prompt_tokens: usize) -> Duration {
        duration_from_ms(self.base_ttft_ms + prompt_tokens as f64 / self.prefill_tokens_per_ms)
    }

    fn tpot(&self) -> Duration {
        duration_from_ms(self.tpot_ms)
    }
}

impl Default for SimulatedEngineConfig {
    fn default() -> Self {
        Self {
            base_ttft_ms: 5.0,
            prefill_tokens_per_ms: 100.0,
            tpot_ms: 12.0,
            fallback_token_id: 0,
            scripted_completion: Vec::new(),
        }
    }
}

/// One scheduler, no KV, no LoRA. `partitions` is the frontend-visible engine
/// count (tests that declare N engines must spawn N schedulers).
pub fn start_engine(config: &SimulatedEngineConfig) -> Engine {
    start_engine_with_partitions(config, 1)
}

pub fn start_engine_with_partitions(config: &SimulatedEngineConfig, partitions: usize) -> Engine {
    assert!(
        partitions > 0,
        "an engine must expose at least one scheduler"
    );
    Engine {
        schedulers: (0..partitions)
            .map(|index| {
                spawn_scheduler(
                    &format!("pegainfer-sim-{index}"),
                    SimScheduler::new(config.clone()),
                )
            })
            .collect(),
        info: EngineInfo {
            kv_capacity: None,
            servable_len: None,
        },
        lora: None,
    }
}

struct SimScheduler {
    config: SimulatedEngineConfig,
    queued: Vec<QueuedRequest>,
    running: Vec<RunningRequest>,
}

struct RunningRequest {
    id: RequestId,
    /// Tokens not yet emitted, stored reversed for cheap pops.
    pending: Vec<u32>,
    next_token_at: Instant,
    finish_reason: FinishReason,
    logprobs: usize,
}

impl SimScheduler {
    fn new(config: SimulatedEngineConfig) -> Self {
        Self {
            config,
            queued: Vec::new(),
            running: Vec::new(),
        }
    }

    fn park_if_waiting(&self) {
        let Some(next) = self.running.iter().map(|r| r.next_token_at).min() else {
            return;
        };
        let now = Instant::now();
        if next <= now {
            return;
        }
        std::thread::sleep((next - now).min(WAIT_SLICE));
    }
}

impl Scheduler for SimScheduler {
    fn submit(&mut self, request: QueuedRequest) {
        self.queued.push(request);
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        for QueuedRequest { id, request } in self.queued.drain(..) {
            if ledger.is_aborted(id) {
                ledger.retire(id);
                continue;
            }
            if request.echo {
                ledger.echo_prompt(
                    id,
                    PromptEcho {
                        ids: request.prompt_tokens.clone(),
                        logprobs: vec![None; request.prompt_tokens.len()],
                    },
                );
            }
            let prompt_len = request.prompt_tokens.len();
            let (pending, finish_reason) =
                planned_completion(&self.config, &request.prompt_tokens, request.max_tokens);
            ledger.admit(id);
            if pending.is_empty() {
                ledger.finish(id, finish_reason);
                continue;
            }
            self.running.push(RunningRequest {
                id,
                pending,
                next_token_at: Instant::now() + self.config.ttft(prompt_len),
                finish_reason,
                logprobs: request.logprobs,
            });
        }

        let now = Instant::now();
        let mut still_running = Vec::new();
        for mut running in self.running.drain(..) {
            if ledger.is_aborted(running.id) {
                ledger.retire(running.id);
                continue;
            }
            if now < running.next_token_at {
                still_running.push(running);
                continue;
            }
            let Some(token) = running.pending.pop() else {
                ledger.finish(running.id, running.finish_reason);
                continue;
            };
            let logprob = (running.logprobs > 0).then_some(TokenLogprob {
                logprob: 0.0,
                top_logprobs: Vec::new(),
            });
            let logprobs = match logprob {
                Some(lp) => vec![Some(lp)],
                None => Vec::new(),
            };
            ledger.push_tokens(running.id, &[token], &logprobs);
            if running.pending.is_empty() {
                ledger.finish(running.id, running.finish_reason);
            } else {
                running.next_token_at = Instant::now() + self.config.tpot();
                still_running.push(running);
            }
        }
        self.running = still_running;
        self.park_if_waiting();
        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            num_running_reqs: self.running.len() as u64,
            num_waiting_reqs: self.queued.len() as u64,
            ..SchedulerMetrics::default()
        }
    }
}

/// Remaining tokens (reversed) plus the terminal reason. Empty pending means
/// finish immediately after admit.
fn planned_completion(
    config: &SimulatedEngineConfig,
    prompt_tokens: &[u32],
    max_tokens: usize,
) -> (Vec<u32>, FinishReason) {
    let script = &config.scripted_completion;
    let emit_count = if script.is_empty() {
        max_tokens
    } else {
        max_tokens.min(script.len())
    };
    let finish_reason = if !script.is_empty() && emit_count == script.len() {
        FinishReason::Stop
    } else {
        FinishReason::Length
    };
    let mut pending: Vec<u32> = if script.is_empty() {
        (0..emit_count)
            .map(|index| fake_token_id(prompt_tokens, index, config.fallback_token_id))
            .collect()
    } else {
        script[..emit_count].to_vec()
    };
    pending.reverse();
    (pending, finish_reason)
}

fn fake_token_id(prompt_tokens: &[u32], index: usize, fallback_token_id: u32) -> u32 {
    if prompt_tokens.is_empty() {
        return fallback_token_id;
    }
    prompt_tokens[index % prompt_tokens.len()]
}

fn duration_from_ms(ms: f64) -> Duration {
    Duration::from_secs_f64(ms / 1000.0)
}

#[cfg(test)]
mod tests {
    use pegainfer_frontend::engine::Request;
    use pegainfer_frontend::engine::Terminal;
    use pegainfer_frontend::sampler::SamplingParams;

    use super::*;

    fn request(prompt_tokens: Vec<u32>, max_tokens: usize, logprobs: usize) -> Request {
        Request {
            prompt_tokens,
            params: SamplingParams::default(),
            max_tokens,
            lora_adapter: None,
            kv_transfer_params: None,
            logprobs,
            echo: false,
            trace_parent: None,
            client_label: None,
        }
    }

    fn collect_completion(
        config: &SimulatedEngineConfig,
        req: Request,
    ) -> (Vec<u32>, Option<usize>, Terminal) {
        let mut engine = start_engine(config);
        assert_eq!(engine.schedulers.len(), 1);
        let mut partition = engine.schedulers.remove(0);
        let mut steps = partition.handle.take_steps().expect("step stream");
        let _control = partition.handle.submit(req);

        let mut tokens = Vec::new();
        let mut prompt_tokens = None;
        let mut terminal = None;
        while terminal.is_none() {
            let step = steps.blocking_recv().expect("step message");
            for update in step.updates {
                if let Some(scheduled) = update.scheduled {
                    prompt_tokens = Some(scheduled.prompt_tokens);
                }
                tokens.extend(update.tokens);
                if let Some(t) = update.terminal {
                    terminal = Some(t);
                }
            }
        }
        drop(partition.handle);
        partition.join.join().expect("driver thread exits");
        (tokens, prompt_tokens, terminal.expect("terminal"))
    }

    #[test]
    fn fake_token_id_cycles_prompt_tokens() {
        assert_eq!(fake_token_id(&[7, 9], 0, 42), 7);
        assert_eq!(fake_token_id(&[7, 9], 1, 42), 9);
        assert_eq!(fake_token_id(&[7, 9], 2, 42), 7);
        assert_eq!(fake_token_id(&[], 0, 42), 42);
    }

    #[test]
    fn scripted_completion_replays_ids_and_stops() {
        let config = SimulatedEngineConfig::new(0.0, 100.0, 0.0, 0)
            .unwrap()
            .with_scripted_completion(vec![11, 22, 33]);
        let (tokens, _, terminal) = collect_completion(&config, request(vec![7, 9], 8, 0));
        assert_eq!(tokens, [11, 22, 33]);
        assert!(matches!(
            terminal,
            Terminal::Finished {
                reason: FinishReason::Stop,
                completion_tokens: 3,
                ..
            }
        ));
    }

    #[test]
    fn scripted_completion_truncated_by_max_tokens_is_length() {
        let config = SimulatedEngineConfig::new(0.0, 100.0, 0.0, 0)
            .unwrap()
            .with_scripted_completion(vec![11, 22, 33]);
        let (tokens, _, terminal) = collect_completion(&config, request(vec![7], 2, 0));
        assert_eq!(tokens, [11, 22]);
        assert!(matches!(
            terminal,
            Terminal::Finished {
                reason: FinishReason::Length,
                completion_tokens: 2,
                ..
            }
        ));
    }

    #[test]
    fn config_rejects_invalid_timing_values() {
        assert!(SimulatedEngineConfig::new(-1.0, 100.0, 12.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(5.0, 0.0, 12.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(5.0, 100.0, -1.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(f64::NAN, 100.0, 12.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(5.0, f64::INFINITY, 12.0, 0).is_err());
        assert!(SimulatedEngineConfig::new(5.0, 100.0, f64::INFINITY, 0).is_err());
    }

    #[test]
    fn simulated_request_emits_scheduled_tokens_and_finished() {
        let config = SimulatedEngineConfig::new(0.0, 100.0, 0.0, 42).unwrap();
        let (tokens, prompt_tokens, terminal) =
            collect_completion(&config, request(vec![7, 9], 3, 1));
        assert_eq!(prompt_tokens, Some(2));
        assert_eq!(tokens, [7, 9, 7]);
        assert!(matches!(
            terminal,
            Terminal::Finished {
                reason: FinishReason::Length,
                prompt_tokens: 2,
                completion_tokens: 3,
            }
        ));
    }

    #[test]
    fn start_engine_with_partitions_exposes_that_many_schedulers() {
        let engine = start_engine_with_partitions(&SimulatedEngineConfig::default(), 3);
        assert_eq!(engine.schedulers.len(), 3);
    }
}
