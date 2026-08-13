use std::collections::HashSet;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use anyhow::ensure;
use axum::Router;
use log::warn;
use serde::Deserialize;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::TransportMode;
use vllm_server::ApiServerOptions;
use vllm_server::ChatTemplateContentFormatOption;
use vllm_server::Config;
use vllm_server::CoordinatorMode;
use vllm_server::CorsConfig;
use vllm_server::DEFAULT_KEEP_ALIVE_TIMEOUT;
use vllm_server::HttpListenerMode;
use vllm_server::ParserSelection;
use vllm_server::RendererSelection;

use crate::engine::LaunchedEngine;

mod bridge;
mod lora;
mod request_contract;
mod wire;

use bridge::LocalEngineBridge;
use bridge::ipc_endpoint;
use bridge::local_ipc_namespace;
pub use lora::LoraModule;
use lora::load_startup_lora_modules;
use lora::lora_openai_routes;
use lora::lora_routes;
pub use lora::parse_lora_modules_arg;

#[derive(Debug, Deserialize)]
struct ModelLenConfig {
    max_position_embeddings: Option<u32>,
    text_config: Option<Box<ModelLenConfig>>,
}

impl ModelLenConfig {
    fn max_model_len(&self) -> Option<u32> {
        self.max_position_embeddings
            .or_else(|| self.text_config.as_ref()?.max_model_len())
    }
}

/// Serve while the engine is still loading: the HTTP frontend (tokenizer,
/// chat templates) starts immediately and the engine bridge attaches once
/// `engine` resolves. HTTP binds only after the bridge registers, so a
/// reachable port still means the engine is ready.
///
/// Pass `max_model_len: None` to read `max_position_embeddings` from
/// `model_path/config.json`; pass `Some(n)` when the path has no config
/// (e.g. a HuggingFace model id for the sim frontend).
pub async fn serve(
    engine: impl Future<Output = Result<LaunchedEngine>> + Send + 'static,
    model_path: &Path,
    served_model_name: Vec<String>,
    port: u16,
    max_model_len: Option<u32>,
    shutdown: CancellationToken,
) -> Result<()> {
    serve_with_engine_count(
        engine,
        model_path,
        served_model_name,
        port,
        max_model_len,
        1,
        shutdown,
    )
    .await
}

/// Serve one HTTP endpoint backed by `engine_count` frontend-visible engine
/// identities. Data-parallel model lines use one identity per scheduler
/// partition; ordinary model lines call [`serve`] and get the single-engine
/// case.
pub async fn serve_with_engine_count(
    engine: impl Future<Output = Result<LaunchedEngine>> + Send + 'static,
    model_path: &Path,
    served_model_name: Vec<String>,
    port: u16,
    max_model_len: Option<u32>,
    engine_count: usize,
    shutdown: CancellationToken,
) -> Result<()> {
    serve_model_on_host(
        engine,
        model_path.to_string_lossy().into_owned(),
        served_model_name,
        "0.0.0.0".to_string(),
        port,
        resolve_max_model_len(model_path, max_model_len),
        engine_count,
        shutdown,
    )
    .await
}

/// Serve an endpoint that requires `max_tokens=1`.
#[allow(clippy::too_many_arguments)]
pub async fn serve_prefill_only_with_engine_count(
    engine: impl Future<Output = Result<LaunchedEngine>> + Send + 'static,
    model_path: &Path,
    served_model_name: Vec<String>,
    port: u16,
    max_model_len: Option<u32>,
    engine_count: usize,
    shutdown: CancellationToken,
) -> Result<()> {
    serve_model_on_host_with_router_extension(
        engine,
        model_path.to_string_lossy().into_owned(),
        served_model_name,
        "0.0.0.0".to_string(),
        port,
        resolve_max_model_len(model_path, max_model_len),
        engine_count,
        shutdown,
        request_contract::prefill_only_routes,
    )
    .await
}

pub async fn serve_model_with_lora_routes(
    engine: crate::engine::Engine,
    model_id: impl Into<String>,
    served_model_name: Vec<String>,
    lora_modules: Vec<LoraModule>,
    port: u16,
    max_model_len: u32,
    shutdown: CancellationToken,
) -> Result<()> {
    let model_id = model_id.into();
    let adapter_names = Arc::new(RwLock::new(HashSet::new()));
    // The Option is the capability: only engines that minted a LoRA channel
    // can serve these routes.
    let control = engine
        .lora
        .clone()
        .ok_or_else(|| anyhow::anyhow!("engine does not expose LoRA adapter control"))?;
    load_startup_lora_modules(&control, &adapter_names, &lora_modules).await?;
    let base_model_name = served_model_name
        .first()
        .cloned()
        .unwrap_or_else(|| model_id.clone());
    serve_model_on_host_with_router_extension(
        std::future::ready(Ok(LaunchedEngine::Stepped(engine))),
        model_id,
        served_model_name.clone(),
        "0.0.0.0".to_string(),
        port,
        max_model_len,
        1,
        shutdown,
        move |router| {
            let lora_router = lora_routes(control, Arc::clone(&adapter_names));
            let openai_router = lora_openai_routes(
                router.clone(),
                base_model_name,
                served_model_name,
                Arc::clone(&adapter_names),
            );
            openai_router.merge(lora_router).fallback_service(router)
        },
    )
    .await
}

async fn serve_model_on_host(
    engine: impl Future<Output = Result<LaunchedEngine>> + Send + 'static,
    model_id: String,
    served_model_name: Vec<String>,
    host: String,
    port: u16,
    max_model_len: u32,
    engine_count: usize,
    shutdown: CancellationToken,
) -> Result<()> {
    serve_model_on_host_with_router_extension(
        engine,
        model_id,
        served_model_name,
        host,
        port,
        max_model_len,
        engine_count,
        shutdown,
        |router| router,
    )
    .await
}

async fn serve_model_on_host_with_router_extension<F>(
    engine: impl Future<Output = Result<LaunchedEngine>> + Send + 'static,
    model_id: String,
    served_model_name: Vec<String>,
    host: String,
    port: u16,
    max_model_len: u32,
    engine_count: usize,
    shutdown: CancellationToken,
    extend_router: F,
) -> Result<()>
where
    F: FnOnce(Router) -> Router,
{
    ensure!(engine_count > 0, "frontend engine_count must be positive");
    let data_parallel_size = u32::try_from(engine_count)
        .map_err(|_| anyhow::anyhow!("frontend engine_count {engine_count} exceeds u32"))?;
    let namespace = local_ipc_namespace()?;
    let input_address = ipc_endpoint(&namespace, "input.sock");
    let output_address = ipc_endpoint(&namespace, "output.sock");

    // The HTTP server runs concurrently with the engine load: vllm-server
    // spends ~1s loading the tokenizer and chat templates before it waits for
    // an engine to register, so neither waits on the other. This task attaches
    // the bridge once the engine resolves and runs it to completion; on engine
    // failure it cancels the server so the error surfaces instead of hanging
    // in the registration wait.
    let server_shutdown = shutdown.child_token();
    let bridge_shutdown = shutdown.child_token();
    let engine_task = tokio::spawn({
        let server_shutdown = server_shutdown.clone();
        let bridge_shutdown = bridge_shutdown.clone();
        let input_address = input_address.clone();
        let output_address = output_address.clone();
        async move {
            let launched = match engine.await {
                Ok(launched) => launched,
                Err(error) => {
                    server_shutdown.cancel();
                    return Err(error);
                }
            };
            let mut bridges = tokio::task::JoinSet::new();
            // Stepped engines own their scheduler threads; reaped below after
            // the bridges (and with them the scheduler handles) are gone.
            let mut scheduler_joins: Vec<std::thread::JoinHandle<()>> = Vec::new();
            match launched {
                LaunchedEngine::Handle(handle) => {
                    let actual_partitions = handle.scheduler_partition_count();
                    if actual_partitions != engine_count {
                        server_shutdown.cancel();
                        anyhow::bail!(
                            "frontend declared {engine_count} engines but the resolved handle \
                             exposes {actual_partitions} scheduler partitions"
                        );
                    }
                    let servable_limit = handle.servable_len().map(|cap| max_model_len.min(cap));
                    let max_model_len = servable_limit.unwrap_or(max_model_len);
                    for engine_index in 0..engine_count {
                        let bridge = LocalEngineBridge {
                            input_address: input_address.clone(),
                            output_address: output_address.clone(),
                            handle: handle.clone(),
                            max_model_len,
                            engine_index: engine_index as u32,
                            data_parallel_size,
                            load_watch: handle.load_watch_for(engine_index),
                        };
                        let shutdown = bridge_shutdown.clone();
                        bridges.spawn(async move { (engine_index, bridge.run(shutdown).await) });
                    }
                    drop(handle);
                }
                LaunchedEngine::Stepped(engine) => {
                    let actual_schedulers = engine.schedulers.len();
                    if actual_schedulers != engine_count {
                        server_shutdown.cancel();
                        anyhow::bail!(
                            "frontend declared {engine_count} engines but the launched engine \
                             exposes {actual_schedulers} schedulers"
                        );
                    }
                    let info = engine.info;
                    let servable_limit = info.servable_len.map(|cap| max_model_len.min(cap));
                    let max_model_len = servable_limit.unwrap_or(max_model_len);
                    for (engine_index, scheduler) in engine.schedulers.into_iter().enumerate() {
                        scheduler_joins.push(scheduler.join);
                        let bridge = bridge::SteppedEngineBridge {
                            input_address: input_address.clone(),
                            output_address: output_address.clone(),
                            scheduler: scheduler.handle,
                            kv_capacity: info.kv_capacity,
                            max_model_len,
                            engine_index: engine_index as u32,
                            data_parallel_size,
                        };
                        let shutdown = bridge_shutdown.clone();
                        bridges.spawn(async move { (engine_index, bridge.run(shutdown).await) });
                    }
                }
            }

            let mut bridge_error = None;
            while let Some(joined) = bridges.join_next().await {
                match joined {
                    Ok((_, Ok(()))) if bridge_shutdown.is_cancelled() => {}
                    Ok((engine_index, Ok(()))) => {
                        bridge_error = Some(anyhow::anyhow!(
                            "local vLLM engine {engine_index} bridge exited unexpectedly"
                        ));
                        break;
                    }
                    Ok((engine_index, Err(error))) => {
                        bridge_error = Some(
                            error
                                .context(format!("local vLLM engine {engine_index} bridge failed")),
                        );
                        break;
                    }
                    Err(error) => {
                        bridge_error = Some(anyhow::anyhow!(
                            "local vLLM engine bridge task panicked: {error}"
                        ));
                        break;
                    }
                }
            }
            if bridge_error.is_some() {
                server_shutdown.cancel();
                bridge_shutdown.cancel();
                bridges.abort_all();
                while bridges.join_next().await.is_some() {}
            }
            // The bridges are gone, and with them the partition handles:
            // intake channels are disconnected, so the drivers drain and
            // exit. Reap their threads to surface scheduler panics.
            if !scheduler_joins.is_empty() {
                let _ = tokio::task::spawn_blocking(move || {
                    for join in scheduler_joins {
                        if join.join().is_err() {
                            warn!("scheduler thread panicked during shutdown");
                        }
                    }
                })
                .await;
            }
            match bridge_error {
                Some(error) => Err(error),
                None => Ok(()),
            }
        }
    });

    let config = Config {
        transport_mode: TransportMode::Bootstrapped {
            input_address,
            output_address,
            engine_start_index: 0,
            engine_count,
            // The in-process bridge registers once the engine future resolves,
            // so this bounds the whole engine load (multi-GPU MoE models take
            // minutes, cold starts longer). Load *failure* already cancels the
            // server via the engine task; this only catches a truly hung load.
            ready_timeout: Duration::from_mins(30),
        },
        coordinator_mode: CoordinatorMode::None,
        model: model_id,
        served_model_name,
        listener_mode: HttpListenerMode::BindTcp { host, port },
        tool_call_parser: ParserSelection::default(),
        reasoning_parser: ParserSelection::default(),
        renderer: RendererSelection::default(),
        chat_template: None,
        default_chat_template_kwargs: None,
        chat_template_content_format: ChatTemplateContentFormatOption::default(),
        max_logprobs: None,
        language_model_only: true,
        cors: CorsConfig::default(),
        api_keys: Vec::new(),
        api_server_options: ApiServerOptions {
            enable_log_requests: true,
            enable_prompt_tokens_details: true,
            enable_request_id_headers: false,
        },
        disable_log_stats: true,
        grpc_port: None,
        shutdown_timeout: Duration::from_secs(10),
        keep_alive_timeout: DEFAULT_KEEP_ALIVE_TIMEOUT,
        profiler: None,
        tls: None,
    };

    let result =
        vllm_server::serve_with_router_extension(config, server_shutdown, extend_router).await;
    // Stop the bridge (no-op if the caller's shutdown already cancelled it),
    // then collect the engine task. If the server failed while the engine is
    // still loading, the uncancellable blocking load must finish first.
    bridge_shutdown.cancel();
    if result.is_err() && !engine_task.is_finished() {
        warn!("HTTP server failed; waiting for the in-flight engine load to finish before exit");
    }
    let result = match engine_task.await {
        Ok(Ok(())) => result,
        // Engine failed: the server saw a cancel and returned Ok — the engine
        // error is the one worth reporting.
        Ok(Err(engine_error)) => result.and(Err(engine_error)),
        Err(join_error) => result.and(Err(anyhow::anyhow!(
            "engine startup task panicked: {join_error}"
        ))),
    };
    let _ = std::fs::remove_dir_all(namespace);
    result
}

pub fn load_max_model_len(model_path: &Path) -> Option<u32> {
    let content = std::fs::read_to_string(model_path.join("config.json")).ok()?;
    serde_json::from_str::<ModelLenConfig>(&content)
        .ok()?
        .max_model_len()
}

fn resolve_max_model_len(model_path: &Path, max_model_len: Option<u32>) -> u32 {
    max_model_len.unwrap_or_else(|| {
        load_max_model_len(model_path).unwrap_or_else(|| {
            const FALLBACK_MAX_MODEL_LEN: u32 = 4096;
            warn!(
                "max_position_embeddings not found in {}/config.json; capping max_model_len at {FALLBACK_MAX_MODEL_LEN}. \
                 Requests are limited to this length — set max_position_embeddings in the model config if it supports more.",
                model_path.display()
            );
            FALLBACK_MAX_MODEL_LEN
        })
    })
}

/// Cancel `token` on the first CTRL+C. Installing the handler replaces the
/// default SIGINT kill behavior — only call this once whatever the token
/// guards can actually wind down (e.g. after an uncancellable blocking engine
/// load has finished), otherwise CTRL+C turns into a no-op wait.
pub fn cancel_token_on_ctrl_c(token: &CancellationToken) {
    let shutdown = token.clone();
    tokio::spawn(async move {
        if let Err(error) = tokio::signal::ctrl_c().await {
            warn!("failed to install CTRL+C handler: {error}");
        }
        shutdown.cancel();
    });
}

pub fn shutdown_token_from_ctrl_c() -> CancellationToken {
    let token = CancellationToken::new();
    cancel_token_on_ctrl_c(&token);
    token
}
