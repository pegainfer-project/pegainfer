//! Blocking test harness over the engine contract.
//!
//! Integration tests predate the step-batched wire protocol and want the old
//! per-request stream ergonomics; this pump restores them the same way the
//! production bridge does — one thread demultiplexes `StepOutputs` into
//! per-request queues. Everything here is synchronous: GPU tests drive
//! requests from plain test threads.

// Each integration test crate compiles this module independently and uses a
// different slice of it.
#![allow(dead_code)]

use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;

use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineInfo;
use pegainfer_frontend::engine::LoraClient;
use pegainfer_frontend::engine::PromptEcho;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::RequestControl;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::RequestUpdate;
use pegainfer_frontend::engine::SchedulerHandle;
use pegainfer_frontend::engine::Terminal;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::sampler::SamplingParams;

/// A contract request with test defaults; adjust fields on the result for
/// echo/logprobs/LoRA variants.
pub(crate) fn request(
    prompt_tokens: Vec<u32>,
    params: SamplingParams,
    max_tokens: usize,
) -> Request {
    Request {
        prompt_tokens,
        params,
        max_tokens,
        lora_adapter: None,
        kv_transfer_params: None,
        logprobs: 0,
        echo: false,
        trace_parent: None,
        client_label: None,
    }
}

pub(crate) struct EngineHarness {
    handle: Option<SchedulerHandle>,
    lora: Option<LoraClient>,
    scheduler_join: Option<std::thread::JoinHandle<()>>,
    pump_join: Option<std::thread::JoinHandle<()>>,
    inbox: Arc<Inbox>,
    pub(crate) info: EngineInfo,
}

struct Inbox {
    state: Mutex<InboxState>,
    cv: Condvar,
}

struct InboxState {
    updates: HashMap<RequestId, VecDeque<RequestUpdate>>,
    closed: bool,
}

impl EngineHarness {
    pub(crate) fn new(mut engine: Engine) -> Self {
        assert_eq!(
            engine.schedulers.len(),
            1,
            "test harness drives single-scheduler engines"
        );
        let mut scheduler = engine.schedulers.remove(0);
        let mut steps = scheduler
            .handle
            .take_steps()
            .expect("a fresh scheduler yields its step stream once");
        let inbox = Arc::new(Inbox {
            state: Mutex::new(InboxState {
                updates: HashMap::new(),
                closed: false,
            }),
            cv: Condvar::new(),
        });
        let pump_inbox = Arc::clone(&inbox);
        let pump_join = std::thread::spawn(move || {
            while let Some(step) = steps.blocking_recv() {
                let mut state = pump_inbox.state.lock().unwrap();
                for update in step.updates {
                    state
                        .updates
                        .entry(update.id)
                        .or_default()
                        .push_back(update);
                }
                drop(state);
                pump_inbox.cv.notify_all();
            }
            pump_inbox.state.lock().unwrap().closed = true;
            pump_inbox.cv.notify_all();
        });
        Self {
            handle: Some(scheduler.handle),
            lora: engine.lora,
            scheduler_join: Some(scheduler.join),
            pump_join: Some(pump_join),
            inbox,
            info: engine.info,
        }
    }

    pub(crate) fn submit(&self, request: Request) -> RequestStream {
        let control = self
            .handle
            .as_ref()
            .expect("harness handle lives until drop")
            .submit(request);
        RequestStream {
            id: control.id(),
            control,
            inbox: Arc::clone(&self.inbox),
        }
    }

    /// The engine's LoRA client; panics when the engine serves no adapter
    /// control — the `Option` on `Engine::lora` is the capability.
    pub(crate) fn lora_client(&self) -> LoraClient {
        self.lora.clone().expect("engine exposes LoRA control")
    }

    /// Submit one request and return its generated token ids, panicking on any
    /// non-`Finished` terminal — the shape most GPU tests want.
    pub(crate) fn generate(
        &self,
        prompt_tokens: Vec<u32>,
        params: SamplingParams,
        max_tokens: usize,
    ) -> Vec<u32> {
        self.submit(request(prompt_tokens, params, max_tokens))
            .expect_finished()
            .tokens
    }
}

impl Drop for EngineHarness {
    fn drop(&mut self) {
        // Closing the submission channel lets the scheduler drain and exit;
        // the step stream then closes and the pump follows.
        drop(self.handle.take());
        if let Some(join) = self.scheduler_join.take() {
            let _ = join.join();
        }
        if let Some(join) = self.pump_join.take() {
            let _ = join.join();
        }
    }
}

/// One request's demultiplexed view of the step stream. `Send`, so tests may
/// drive concurrent requests from separate threads.
pub(crate) struct RequestStream {
    id: RequestId,
    pub(crate) control: RequestControl,
    inbox: Arc<Inbox>,
}

impl RequestStream {
    pub(crate) fn id(&self) -> RequestId {
        self.id
    }

    /// Next update for this request; `None` once the engine is gone.
    pub(crate) fn recv(&mut self) -> Option<RequestUpdate> {
        let mut state = self.inbox.state.lock().unwrap();
        loop {
            if let Some(update) = state
                .updates
                .get_mut(&self.id)
                .and_then(VecDeque::pop_front)
            {
                return Some(update);
            }
            if state.closed {
                return None;
            }
            state = self.inbox.cv.wait(state).unwrap();
        }
    }

    /// Fold the stream to its terminal. Panics if the engine dies first — a
    /// vanished stream is a test failure, not an outcome.
    pub(crate) fn outcome(mut self) -> Outcome {
        let mut outcome = Outcome {
            tokens: Vec::new(),
            logprobs: Vec::new(),
            cached_tokens: None,
            prompt_echo: None,
            terminal: Terminal::Failed {
                message: String::new(),
                prompt_tokens: 0,
                completion_tokens: 0,
            },
        };
        loop {
            let update = self
                .recv()
                .expect("engine closed the stream without a terminal");
            outcome.tokens.extend_from_slice(&update.tokens);
            outcome.logprobs.extend(update.logprobs);
            if update.cached_tokens.is_some() {
                assert!(
                    outcome.cached_tokens.is_none(),
                    "cached_tokens must be reported at most once per request"
                );
                outcome.cached_tokens = update.cached_tokens;
            }
            if update.prompt_echo.is_some() {
                outcome.prompt_echo = update.prompt_echo;
            }
            if let Some(terminal) = update.terminal {
                outcome.terminal = terminal;
                return outcome;
            }
        }
    }

    /// The common success path: fold to the terminal and require `Finished`.
    pub(crate) fn expect_finished(self) -> Outcome {
        let outcome = self.outcome();
        match &outcome.terminal {
            Terminal::Finished { .. } => outcome,
            Terminal::Rejected { reason, .. } => panic!("generation rejected: {reason}"),
            Terminal::Failed { message, .. } => panic!("generation failed: {message}"),
        }
    }
}

/// A finished request, folded: everything the old per-request event stream
/// delivered incrementally.
pub(crate) struct Outcome {
    pub(crate) tokens: Vec<u32>,
    pub(crate) logprobs: Vec<Option<TokenLogprob>>,
    pub(crate) cached_tokens: Option<usize>,
    pub(crate) prompt_echo: Option<PromptEcho>,
    pub(crate) terminal: Terminal,
}

/// Minimal stderr logger for gate children (`PEGAINFER_TEST_LOG=1`): the
/// hedged execution gate counts the executor's per-round hedge trace lines,
/// and cargo test binaries install no logger of their own. Forwarding only
/// this crate's records keeps child stderr parseable.
struct TestLogger;

impl log::Log for TestLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.target().starts_with("pegainfer_qwen3")
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            eprintln!("[{}] {}", record.level(), record.args());
        }
    }

    fn flush(&self) {}
}

static TEST_LOGGER: TestLogger = TestLogger;

pub(crate) fn init_capture_logging() {
    if std::env::var("PEGAINFER_TEST_LOG").is_ok() {
        let _ = log::set_logger(&TEST_LOGGER);
        log::set_max_level(log::LevelFilter::Debug);
    }
}
