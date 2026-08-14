//! Echo model server: the full pegainfer serving stack (OpenAI-compatible
//! HTTP -> vLLM engine protocol -> the `Scheduler` contract) over an engine
//! that echoes the prompt back instead of running a model. No GPU, no
//! weights.
//!
//! It doubles as the reference for wiring a model line in: the echo line is
//! implemented in full below, and `main` at the bottom mirrors
//! `pegainfer-server/src/main.rs`'s serving flow with the compiled-in
//! registry reduced to the echo line alone.
//!
//! Run it (from the workspace root or anywhere else — the "model" is a
//! checked-in fixture next to this file):
//!
//!   cargo run -p pegainfer-frontend --example echo-server
//!
//! Talk to it:
//!
//!   curl http://127.0.0.1:8000/v1/completions \
//!     -H 'content-type: application/json' \
//!     -d '{"model": "echo", "prompt": "hello, pegainfer", "max_tokens": 64}'
//!
//!   # the completion text is the prompt back: "hello, pegainfer"

use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use clap::FromArgMatches;
use pegainfer_frontend::engine::ActiveRequest;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineInfo;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::engine::LoadSnapshot;
use pegainfer_frontend::engine::QueuedRequest;
use pegainfer_frontend::engine::Scheduler;
use pegainfer_frontend::engine::StepEmitter;
use pegainfer_frontend::engine::spawn_scheduler;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;
use pegainfer_frontend::model_line::ModelLineRegistry;
use pegainfer_frontend::model_line::SharedArgs;
use pegainfer_frontend::model_line::provided_args;
use pegainfer_frontend::vllm;

/// The echo "model" is checked in here: a `config.json` the echo line's
/// `probe` claims, plus a byte-level tokenizer for the serving stack.
const FIXTURE_MODEL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/examples/echo-model");

// ---------------------------------------------------------------------------
// The echo model line, complete. A real model line lives in its own crate;
// everything it must implement against the frontend contract fits between
// here and `main`.
// ---------------------------------------------------------------------------

/// The instance a server binary registers with its `ModelLineRegistry`.
static MODEL_LINE: EchoLine = EchoLine;

/// Every new model line's checklist, minimal edition: claim a `config.json`
/// identity in `probe`, spawn schedulers in `launch`. No exclusive CLI flags
/// and no shared flags beyond the core ones.
struct EchoLine;

impl ModelLine for EchoLine {
    fn name(&self) -> &'static str {
        "Echo"
    }

    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
        if model_type == Some("echo") {
            Ok(())
        } else {
            Err(format!("model_type {model_type:?} is not \"echo\""))
        }
    }

    fn launch(&self, _ctx: &LaunchContext<'_>) -> Result<LaunchedEngine> {
        // A real line loads config/weights and may spawn several schedulers
        // (one per DP replica). The echo engine is one scheduler reporting no
        // KV pool and no servable-length limit of its own.
        Ok(Engine {
            schedulers: vec![spawn_scheduler("echo", EchoScheduler::default())],
            info: EngineInfo {
                kv_capacity: None,
                servable_len: None,
            },
            lora: None,
        }
        .into())
    }
}

/// Answers each request with its own prompt tokens: echo
/// `prompt[..min(len, max_tokens)]` back, one token per step, then finish —
/// `FinishReason::Stop` for a full echo, `FinishReason::Length` when
/// `max_tokens` cut it short. One token per step keeps the per-token
/// streaming path warm instead of collapsing the response into one batch.
#[derive(Default)]
struct EchoScheduler {
    queued: Vec<QueuedRequest>,
    running: Vec<RunningRequest>,
}

/// One admission onward: the request handle plus its echo cursor — prompt
/// tokens not yet echoed and whether `max_tokens` trimmed the echo short.
struct RunningRequest {
    active: ActiveRequest,
    /// Prompt tokens not yet echoed, stored reversed for cheap pops.
    pending: Vec<u32>,
    /// Whether `max_tokens` trimmed the prompt (decides the finish reason).
    truncated: bool,
}

impl Scheduler for EchoScheduler {
    fn submit(&mut self, req: QueuedRequest) {
        self.queued.push(req);
    }

    fn step(&mut self, emitter: &mut StepEmitter) -> Result<()> {
        for req in self.queued.drain(..) {
            if req.is_aborted() {
                emitter.retire_queued(req);
                continue;
            }
            let request = req.request();
            let echo_len = request.prompt_tokens.len().min(request.max_tokens);
            let truncated = echo_len < request.prompt_tokens.len();
            let mut pending = Vec::from(&request.prompt_tokens[..echo_len]);
            pending.reverse();
            let active = emitter.admit(req);
            self.running.push(RunningRequest {
                active,
                pending,
                truncated,
            });
        }
        let mut still_running = Vec::new();
        for mut running in self.running.drain(..) {
            if running.active.is_aborted() {
                emitter.retire(running.active);
                continue;
            }
            if let Some(token) = running.pending.pop() {
                emitter.push_tokens(&mut running.active, &[token], &[]);
                still_running.push(running);
            } else {
                let reason = if running.truncated {
                    FinishReason::Length
                } else {
                    FinishReason::Stop
                };
                emitter.finish(running.active, reason);
            }
        }
        self.running = still_running;
        Ok(())
    }

    fn load(&self) -> LoadSnapshot {
        LoadSnapshot {
            num_running_reqs: self.running.len() as u64,
            num_waiting_reqs: self.queued.len() as u64,
            ..LoadSnapshot::default()
        }
    }
}

// ---------------------------------------------------------------------------
// Serving: same shape as pegainfer-server/src/main.rs — registry, detect by
// config.json, launch, serve — minus the lines echo doesn't need.
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    let registry = ModelLineRegistry::new(vec![&MODEL_LINE]);
    let cmd = registry.build_command(
        clap::Command::new("echo-server").about("pegainfer echo model server (no GPU, no weights)"),
    );
    let matches = cmd.clone().get_matches();
    let mut shared = SharedArgs::from_arg_matches(&matches)
        .map_err(|error| anyhow::anyhow!("invalid CLI args: {error}"))?;

    // Default to the checked-in fixture; --model-path still works if you keep
    // your own copy of these few files somewhere else.
    let provided = provided_args(&matches, &cmd);
    if !provided.contains("model_path") {
        shared.model_path = PathBuf::from(FIXTURE_MODEL_DIR);
    }

    let config_path = shared.model_path.join("config.json");
    let content = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;

    let line = registry.detect(&config)?;
    registry.validate_provided(line, &provided, &cmd)?;
    shared.validate(&provided)?;

    let ctx = LaunchContext {
        model_path: &shared.model_path,
        config: &config,
        shared: &shared,
        matches: &matches,
    };
    let plan = line.serve_plan(&ctx)?;
    // Echo has nothing to load; a real line hides a blocking weights load
    // behind this call (see pegainfer-server's spawn_blocking).
    let engine = line.launch(&ctx)?;

    let served_model_name = vec![
        shared
            .served_model_name
            .clone()
            .unwrap_or_else(|| "echo".to_owned()),
    ];
    eprintln!(
        "serving the echo model on port {} (model={:?}); Ctrl-C to stop",
        shared.port, served_model_name[0],
    );
    vllm::serve_with_engine_count(
        std::future::ready(Ok(engine)),
        &shared.model_path,
        served_model_name,
        shared.port,
        None,
        plan.scheduler_partition_count,
        vllm::shutdown_token_from_ctrl_c(),
    )
    .await
}
