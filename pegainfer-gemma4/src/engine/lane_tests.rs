use std::path::Path;
use std::time::Duration;
use std::time::Instant;

use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::LiveScheduler;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::RequestControl;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_frontend::engine::spawn_scheduler;

pub(super) use super::lane_step_collector::Drained;
use super::lane_step_collector::StepCollector;

pub(super) struct Harness {
    scheduler: Option<LiveScheduler>,
    pub(super) steps: StepCollector,
    pub(super) servable_len: Option<u32>,
}

impl Harness {
    fn from_engine(mut engine: Engine) -> Self {
        let mut scheduler = engine.schedulers.remove(0);
        let steps = scheduler.handle.take_steps().expect("fresh step stream");
        Self {
            scheduler: Some(scheduler),
            steps: StepCollector::new(steps),
            servable_len: engine.info.servable_len,
        }
    }

    pub(super) fn from_state(state: super::EngineState) -> Self {
        let servable_len = Some(u32::try_from(state.max_context).expect("servable ceiling"));
        let mut scheduler = spawn_scheduler("gemma4-test", super::Gemma4Scheduler::new(state));
        let steps = scheduler.handle.take_steps().expect("fresh step stream");
        Self {
            scheduler: Some(scheduler),
            steps: StepCollector::new(steps),
            servable_len,
        }
    }

    pub(super) fn submit(&self, prompt_tokens: Vec<u32>, max_tokens: usize) -> RequestControl {
        self.scheduler
            .as_ref()
            .expect("live scheduler")
            .handle
            .submit(Request {
                prompt_tokens,
                params: pegainfer_frontend::sampler::SamplingParams {
                    ignore_eos: true,
                    ..pegainfer_frontend::sampler::SamplingParams::default()
                },
                stop_policy: pegainfer_frontend::engine::StopPolicy::default(),
                max_tokens,
                lora_adapter: None,
                kv_transfer_params: None,
                logprobs: None,
                prompt_logprobs: None,
                trace_parent: None,
                client_label: None,
            })
    }

    pub(super) fn metrics(&self) -> SchedulerMetrics {
        self.scheduler
            .as_ref()
            .expect("live scheduler")
            .handle
            .metrics()
    }

    pub(super) fn shutdown(&mut self, aborts: &[&RequestControl]) {
        for control in aborts {
            control.abort();
        }
        let finished = self
            .scheduler
            .as_ref()
            .expect("live scheduler")
            .join
            .is_finished();
        if !finished {
            let drained = wait_until(Duration::from_secs(10), || {
                let metrics = self.metrics();
                metrics.num_running_reqs == 0 && metrics.num_waiting_reqs == 0
            });
            if !drained {
                let metrics = self.metrics();
                panic!(
                    "scheduler did not drain within 10s: running {} waiting {}",
                    metrics.num_running_reqs, metrics.num_waiting_reqs
                );
            }
        }
        let scheduler = self
            .scheduler
            .take()
            .expect("scheduler not shut down twice");
        drop(scheduler.handle);
        scheduler.join.join().expect("scheduler thread exits");
    }
}

/// The serving knobs a test names, and nothing from the process environment.
pub(super) fn knob_table<'a>(
    overrides: &'a [(&'a str, &'a str)],
) -> impl Fn(&str) -> anyhow::Result<Option<String>> + 'a {
    move |name: &str| {
        Ok(overrides
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, value)| (*value).to_string()))
    }
}

pub(super) fn launch(overrides: &[(&str, &str)]) -> Harness {
    let dir = crate::testkit::model_path();
    let engine = super::start_with_knobs(
        Path::new(&dir),
        &EngineLoadOptions::default(),
        &knob_table(overrides),
    )
    .expect("engine start");
    Harness::from_engine(engine)
}

pub(super) fn load_state(overrides: &[(&str, &str)]) -> anyhow::Result<super::EngineState> {
    let dir = crate::testkit::model_path();
    let config = crate::config::Gemma4Config::from_file(&dir)?;
    let knobs = super::ServingKnobs::resolve(&knob_table(overrides), &config)?;
    let policy = super::generation_policy(&dir)?;
    super::EngineState::load(&dir, config, knobs, 0, policy, 0x5EED, true)
}

pub(super) fn ids(len: usize, salt: u32) -> Vec<u32> {
    (0..len as u32)
        .map(|i| 1000 + (i * 37 + salt) % 50000)
        .collect()
}

pub(super) fn warm_prompt(prefix: &[u32]) -> Vec<u32> {
    let mut prompt = prefix.to_vec();
    prompt.extend(ids(60, 11));
    prompt
}

pub(super) fn pin_live_stream(harness: &mut Harness) -> RequestControl {
    let streamer = harness.submit(ids(40, 0), 1024);
    harness.steps.wait_tokens(streamer.id(), 2);
    streamer
}

pub(super) fn assert_warm_result(harness: &mut Harness, id: RequestId, cached: usize, label: &str) {
    let warm = harness.steps.drain(id, label);
    assert_eq!((warm.tokens, warm.finish), (4, FinishReason::Length));
    assert_eq!(warm.cached, cached, "warm admission resume frontier");
}

pub(super) fn wait_until(timeout: Duration, mut predicate: impl FnMut() -> bool) -> bool {
    let start = Instant::now();
    while start.elapsed() < timeout {
        if predicate() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    false
}
