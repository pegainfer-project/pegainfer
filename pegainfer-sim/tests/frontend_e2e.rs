use std::fs;
use std::net::TcpListener;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use pegainfer_frontend::engine::LoadSnapshot;
use pegainfer_frontend::engine::SpecDecodeCounters;
use pegainfer_sim::SimulatedEngineConfig;
use pegainfer_sim::start_engine;
use reqwest::Client;
use serde_json::Value;
use serde_json::json;
use tempfile::TempDir;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const HTTP_TIMEOUT: Duration = Duration::from_secs(10);
const MODEL_NAME: &str = "pegainfer-sim-e2e";
const METRICS_MODEL_NAME: &str = "pegainfer-sim-e2e-metrics";
const CLOSED_FEED_MODEL_NAME: &str = "pegainfer-sim-e2e-closed-feed";
const SPEC_METRICS_MODEL_NAME: &str = "pegainfer-sim-e2e-spec-metrics";
const SERVER_START_ATTEMPTS: usize = 5;

struct SimServer {
    base_url: String,
    model_name: String,
    shutdown: CancellationToken,
    task: JoinHandle<Result<()>>,
    load_txs: Vec<watch::Sender<LoadSnapshot>>,
    _model_dir: TempDir,
}

impl SimServer {
    async fn spawn() -> Result<Self> {
        Self::spawn_with_model_dir(model_dir_with_minimal_metadata()?).await
    }

    async fn spawn_partitioned() -> Result<Self> {
        Self::spawn_with_model_dir_and_engine_count(
            model_dir_with_minimal_metadata()?,
            2,
            METRICS_MODEL_NAME,
        )
        .await
    }

    async fn spawn_with_spec_metrics() -> Result<Self> {
        Self::spawn_with_model_dir_and_engine_count(
            model_dir_with_minimal_metadata()?,
            2,
            SPEC_METRICS_MODEL_NAME,
        )
        .await
    }

    async fn spawn_with_closed_load_feed() -> Result<Self> {
        Self::spawn_with_model_dir_and_engine_count(
            model_dir_with_minimal_metadata()?,
            2,
            CLOSED_FEED_MODEL_NAME,
        )
        .await
    }

    async fn spawn_with_model_dir(model_dir: TempDir) -> Result<Self> {
        Self::spawn_with_model_dir_and_engine_count(model_dir, 1, MODEL_NAME).await
    }

    async fn spawn_with_model_dir_and_engine_count(
        model_dir: TempDir,
        engine_count: usize,
        model_name: &str,
    ) -> Result<Self> {
        let mut last_error = None;
        for attempt in 1..=SERVER_START_ATTEMPTS {
            match Self::spawn_once(&model_dir, engine_count, model_name).await {
                Ok(started) => {
                    return Ok(Self {
                        base_url: started.base_url,
                        model_name: started.model_name,
                        shutdown: started.shutdown,
                        task: started.task,
                        load_txs: started.load_txs,
                        _model_dir: model_dir,
                    });
                }
                Err(error) => {
                    last_error = Some(error);
                    if attempt < SERVER_START_ATTEMPTS {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow!("sim frontend startup was not attempted")))
            .with_context(|| {
                format!("failed to start sim frontend after {SERVER_START_ATTEMPTS} attempts")
            })
    }

    async fn spawn_once(
        model_dir: &TempDir,
        engine_count: usize,
        model_name: &str,
    ) -> Result<StartedSimServer> {
        let port = reserve_loopback_port()?;
        let base_url = format!("http://127.0.0.1:{port}");
        let shutdown = CancellationToken::new();
        let mut engine = start_engine(SimulatedEngineConfig::new(0.0, 1000.0, 0.0, 1)?);
        let load_txs = if engine_count > 1 {
            let (load_txs, load_watches) = (0..engine_count)
                .map(|_| watch::channel(LoadSnapshot::default()))
                .unzip();
            engine = engine.with_load_watches(load_watches);
            load_txs
        } else {
            Vec::new()
        };
        let server_shutdown = shutdown.clone();
        let served_model_name = model_name.to_string();
        let started_model_name = served_model_name.clone();
        let model_path_buf = model_dir.path().to_path_buf();
        let mut task = tokio::spawn(async move {
            pegainfer_frontend::vllm::serve_with_engine_count(
                std::future::ready(Ok(engine.into())),
                &model_path_buf,
                vec![served_model_name],
                port,
                Some(128),
                engine_count,
                server_shutdown,
            )
            .await
        });

        let client = test_client()?;
        let health_result = tokio::select! {
            result = wait_for_health(&client, &base_url) => result,
            result = &mut task => {
                return match result {
                    Ok(Ok(())) => Err(anyhow!("sim frontend exited before becoming healthy")),
                    Ok(Err(error)) => Err(error).context("sim frontend exited before becoming healthy"),
                    Err(error) => Err(error).context("sim frontend task panicked"),
                };
            }
        };

        if let Err(error) = health_result {
            shutdown.cancel();
            let _ = tokio::time::timeout(Duration::from_secs(10), task).await;
            return Err(error);
        }

        Ok(StartedSimServer {
            base_url,
            model_name: started_model_name,
            shutdown,
            task,
            load_txs,
        })
    }

    fn publish_load(&self, partition: usize, snapshot: &LoadSnapshot) -> Result<()> {
        let sender = self
            .load_txs
            .get(partition)
            .ok_or_else(|| anyhow!("sim frontend has no load feed for partition {partition}"))?;
        let _ = sender.send_replace(*snapshot);
        Ok(())
    }

    async fn shutdown(self) -> Result<()> {
        self.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(10), self.task)
            .await
            .context("timed out waiting for sim frontend shutdown")?
            .context("sim frontend task panicked")?
    }
}

struct StartedSimServer {
    base_url: String,
    model_name: String,
    shutdown: CancellationToken,
    task: JoinHandle<Result<()>>,
    load_txs: Vec<watch::Sender<LoadSnapshot>>,
}

fn empty_model_dir() -> Result<TempDir> {
    tempfile::tempdir().context("failed to create temp model dir")
}

fn model_dir_with_minimal_metadata() -> Result<TempDir> {
    let dir = empty_model_dir()?;

    // The simulated frontend still builds the normal vLLM text/chat stack.
    // Token-id prompts avoid tokenizer encode work, but generated ids still
    // need a tokenizer for detokenization and a tiny config for metadata.
    fs::write(dir.path().join("tokenizer.json"), TINY_TOKENIZER_JSON)
        .context("failed to write tiny tokenizer.json")?;
    fs::write(
        dir.path().join("tokenizer_config.json"),
        TINY_TOKENIZER_CONFIG_JSON,
    )
    .context("failed to write tiny tokenizer_config.json")?;
    fs::write(dir.path().join("config.json"), TINY_CONFIG_JSON)
        .context("failed to write tiny config.json")?;

    Ok(dir)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simulated_engine_serves_openai_completions_over_http() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;

    assert_models_endpoint(&client, &server.base_url, &server.model_name).await?;
    assert_non_streaming_completion_has_output(&client, &server.base_url, &server.model_name)
        .await?;
    assert_streaming_completion_emits_done(&client, &server.base_url, &server.model_name).await?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn one_http_endpoint_exports_per_engine_scheduler_metrics() -> Result<()> {
    let server = SimServer::spawn_partitioned().await?;
    let client = test_client()?;

    assert_non_streaming_completion_has_output(&client, &server.base_url, &server.model_name)
        .await?;
    wait_for_metrics(
        &client,
        &server.base_url,
        &[
            ("vllm:num_requests_running", "0", 0.0),
            ("vllm:num_requests_running", "1", 0.0),
            ("vllm:num_requests_waiting", "0", 0.0),
            ("vllm:num_requests_waiting", "1", 0.0),
            ("vllm:kv_cache_usage_perc", "0", 0.0),
            ("vllm:kv_cache_usage_perc", "1", 0.0),
        ],
        &server.model_name,
    )
    .await?;

    server.publish_load(
        0,
        &LoadSnapshot {
            kv_used_blocks: 25,
            kv_total_blocks: 100,
            num_running_reqs: 1,
            num_waiting_reqs: 0,
            spec_decode: None,
        },
    )?;
    server.publish_load(
        1,
        &LoadSnapshot {
            kv_used_blocks: 50,
            kv_total_blocks: 100,
            num_running_reqs: 0,
            num_waiting_reqs: 2,
            spec_decode: None,
        },
    )?;
    wait_for_metrics(
        &client,
        &server.base_url,
        &[
            ("vllm:num_requests_running", "0", 1.0),
            ("vllm:num_requests_waiting", "1", 2.0),
            ("vllm:kv_cache_usage_perc", "0", 0.25),
            ("vllm:kv_cache_usage_perc", "1", 0.5),
        ],
        &server.model_name,
    )
    .await?;

    server.publish_load(
        0,
        &LoadSnapshot {
            kv_used_blocks: 75,
            kv_total_blocks: 100,
            num_running_reqs: 3,
            num_waiting_reqs: 4,
            spec_decode: None,
        },
    )?;
    server.publish_load(
        1,
        &LoadSnapshot {
            kv_used_blocks: 25,
            kv_total_blocks: 100,
            num_running_reqs: 5,
            num_waiting_reqs: 6,
            spec_decode: None,
        },
    )?;
    wait_for_metrics(
        &client,
        &server.base_url,
        &[
            ("vllm:num_requests_running", "0", 3.0),
            ("vllm:num_requests_running", "1", 5.0),
            ("vllm:num_requests_waiting", "0", 4.0),
            ("vllm:num_requests_waiting", "1", 6.0),
            ("vllm:kv_cache_usage_perc", "0", 0.75),
            ("vllm:kv_cache_usage_perc", "1", 0.25),
        ],
        &server.model_name,
    )
    .await?;

    server.publish_load(0, &LoadSnapshot::default())?;
    server.publish_load(1, &LoadSnapshot::default())?;
    wait_for_metrics(
        &client,
        &server.base_url,
        &[
            ("vllm:num_requests_running", "0", 0.0),
            ("vllm:num_requests_running", "1", 0.0),
            ("vllm:num_requests_waiting", "0", 0.0),
            ("vllm:num_requests_waiting", "1", 0.0),
            ("vllm:kv_cache_usage_perc", "0", 0.0),
            ("vllm:kv_cache_usage_perc", "1", 0.0),
        ],
        &server.model_name,
    )
    .await?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn spec_decode_counters_reach_prometheus_as_cumulative_totals() -> Result<()> {
    let server = SimServer::spawn_with_spec_metrics().await?;
    let client = test_client()?;

    let mut totals = SpecDecodeCounters::new(2).expect("K within bounds");
    totals.observe_draft(2, 1);
    server.publish_load(
        0,
        &LoadSnapshot {
            spec_decode: Some(totals),
            ..LoadSnapshot::default()
        },
    )?;
    // `_total` is appended at exposition; the registered name scrapes empty.
    wait_for_metrics(
        &client,
        &server.base_url,
        &[
            ("vllm:spec_decode_num_drafts_total", "0", 1.0),
            ("vllm:spec_decode_num_draft_tokens_total", "0", 2.0),
            ("vllm:spec_decode_num_accepted_tokens_total", "0", 1.0),
        ],
        &server.model_name,
    )
    .await?;

    // Two verify steps coalesced into one snapshot, as the watch would. What
    // must come back is the scheduler's cumulative, not this interval's delta:
    // a delta counted twice or dropped breaks that equality, and nothing below
    // the exposition layer can see it.
    totals.observe_draft(2, 2);
    totals.observe_draft(2, 2);
    server.publish_load(
        0,
        &LoadSnapshot {
            spec_decode: Some(totals),
            ..LoadSnapshot::default()
        },
    )?;
    wait_for_metrics(
        &client,
        &server.base_url,
        &[
            (
                "vllm:spec_decode_num_drafts_total",
                "0",
                totals.num_drafts as f64,
            ),
            (
                "vllm:spec_decode_num_draft_tokens_total",
                "0",
                totals.num_draft_tokens as f64,
            ),
            (
                "vllm:spec_decode_num_accepted_tokens_total",
                "0",
                totals.num_accepted_tokens as f64,
            ),
        ],
        &server.model_name,
    )
    .await?;

    wait_for_labeled_metrics(
        &client,
        &server.base_url,
        &[
            (
                "vllm:spec_decode_num_accepted_tokens_per_pos_total",
                "0",
                &[("position", "0")][..],
                totals.num_accepted_tokens_per_pos[0] as f64,
            ),
            (
                "vllm:spec_decode_num_accepted_tokens_per_pos_total",
                "0",
                &[("position", "1")][..],
                totals.num_accepted_tokens_per_pos[1] as f64,
            ),
        ],
        &server.model_name,
    )
    .await?;

    // Exposition is one series per draft slot the drafter actually has, so the
    // fixed `MAX_SPEC_TOKENS` array width must not leak past `K = 2`.
    let metrics = client
        .get(format!("{}/metrics", server.base_url))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let leaked: Vec<_> = metrics
        .lines()
        .filter(|line| {
            line.contains(&format!("model_name=\"{}\"", server.model_name))
                && line.contains("position=\"2\"")
        })
        .collect();
    assert!(
        leaked.is_empty(),
        "per-position series past K=2 were exposed: {leaked:?}"
    );

    // Engine 1 never drafted. Its counters are pre-registered, so they must be
    // present and zero — which is also what pins each delta to one engine
    // instead of being applied to both.
    wait_for_metrics(
        &client,
        &server.base_url,
        &[
            ("vllm:spec_decode_num_drafts_total", "1", 0.0),
            ("vllm:spec_decode_num_draft_tokens_total", "1", 0.0),
            ("vllm:spec_decode_num_accepted_tokens_total", "1", 0.0),
        ],
        &server.model_name,
    )
    .await?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn closed_scheduler_load_feed_stops_the_endpoint() -> Result<()> {
    let mut server = SimServer::spawn_with_closed_load_feed().await?;
    drop(server.load_txs.remove(0));

    let service_result = tokio::time::timeout(Duration::from_secs(10), &mut server.task)
        .await
        .context("frontend did not stop after a scheduler load feed closed")?
        .context("sim frontend task panicked")?;
    let error = service_result.expect_err("closed scheduler load feed must fail the endpoint");
    let message = format!("{error:#}");
    if !message.contains("scheduler load feed closed") {
        bail!("closed scheduler load feed returned unexpected error: {message}");
    }

    Ok(())
}

async fn wait_for_metrics(
    client: &Client,
    base_url: &str,
    expected: &[(&str, &str, f64)],
    model_name: &str,
) -> Result<()> {
    let expected: Vec<_> = expected
        .iter()
        .map(|(metric, engine, value)| (*metric, *engine, &[][..], *value))
        .collect();
    wait_for_labeled_metrics(client, base_url, &expected, model_name).await
}

/// One expected exposition line: metric name, `engine` label, any further
/// labels, and the value it must carry.
type ExpectedMetric<'a> = (&'a str, &'a str, &'a [(&'a str, &'a str)], f64);

/// [`wait_for_metrics`] plus arbitrary extra labels, for families keyed by more
/// than engine and model — the per-position spec-decode counter adds `position`.
async fn wait_for_labeled_metrics(
    client: &Client,
    base_url: &str,
    expected: &[ExpectedMetric<'_>],
    model_name: &str,
) -> Result<()> {
    let metrics_url = format!("{base_url}/metrics");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let mut last_metrics = String::new();
    loop {
        if tokio::time::Instant::now() >= deadline {
            bail!("timed out waiting for scheduler metrics at {metrics_url}:\n{last_metrics}");
        }

        if let Ok(response) = client.get(&metrics_url).send().await {
            if let Ok(response) = response.error_for_status() {
                if let Ok(metrics) = response.text().await {
                    let all_found =
                        expected
                            .iter()
                            .all(|(metric, engine, labels, expected_value)| {
                                metrics.lines().any(|line| {
                                    line.starts_with(metric)
                                        && line.contains(&format!("engine=\"{engine}\""))
                                        && line.contains(&format!("model_name=\"{model_name}\""))
                                        && labels.iter().all(|(name, value)| {
                                            line.contains(&format!("{name}=\"{value}\""))
                                        })
                                        && line
                                            .rsplit_once(' ')
                                            .and_then(|(_, value)| value.parse::<f64>().ok())
                                            .is_some_and(|value| {
                                                (value - expected_value).abs() < 1e-9
                                            })
                                })
                            });
                    if all_found {
                        return Ok(());
                    }
                    last_metrics = metrics;
                }
            }
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn frontend_rejects_engine_partition_mismatch() -> Result<()> {
    let model_dir = model_dir_with_minimal_metadata()?;
    let port = reserve_loopback_port()?;
    let engine = start_engine(SimulatedEngineConfig::new(0.0, 1000.0, 0.0, 1)?);
    let result = tokio::time::timeout(
        Duration::from_secs(10),
        pegainfer_frontend::vllm::serve_with_engine_count(
            std::future::ready(Ok(engine.into())),
            model_dir.path(),
            vec![MODEL_NAME.to_string()],
            port,
            Some(128),
            2,
            CancellationToken::new(),
        ),
    )
    .await
    .context("partition-mismatch server did not stop")?;
    let error = result.expect_err("one scheduler partition cannot register as two engines");
    if !error
        .to_string()
        .contains("declared 2 engines but the resolved handle exposes 1 scheduler partitions")
    {
        bail!("unexpected partition-mismatch error: {error:#}");
    }
    Ok(())
}

// The old "mounted LoRA routes report unsupported" sim test is gone by
// construction: `serve_model_with_lora_routes` now requires an `Engine` whose
// `lora` capability is `Some(LoraClient)` — the `Option` is the capability —
// and the simulated handle engine cannot provide one. The gone-engine HTTP
// mapping is pinned by the `vllm::lora` route tests.

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_streaming_completion_returns_nonempty_output_for_positive_max_tokens() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;

    assert_non_streaming_completion_has_output(&client, &server.base_url, &server.model_name)
        .await?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_lora_xarg_rejects_only_its_request() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;
    let mut body = completion_body(&server.model_name, false);
    body["vllm_xargs"] = json!({ "pegainfer_lora_adapter": 123 });

    let response = client
        .post(format!("{}/v1/completions", server.base_url))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body.to_string())
        .send()
        .await?;
    let status = response.status();
    let response_body = response.text().await?;
    if status.is_success() {
        bail!("invalid pegainfer_lora_adapter unexpectedly succeeded: {response_body}");
    }

    assert_non_streaming_completion_has_output(&client, &server.base_url, &server.model_name)
        .await?;
    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_completion_emits_terminal_done() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;

    assert_streaming_completion_emits_done(&client, &server.base_url, &server.model_name).await?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_completions_returns_correct_format() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;

    let response: Value = client
        .post(format!("{}/v1/chat/completions", server.base_url))
        .json(&json!({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "alpha beta"}],
            "max_tokens": 4,
            "temperature": 0.0
        }))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    assert_eq!(
        response["object"].as_str(),
        Some("chat.completion"),
        "object field must be chat.completion: {response}"
    );

    let choice = &response["choices"][0];
    assert_eq!(
        choice["message"]["role"].as_str(),
        Some("assistant"),
        "message role must be assistant: {response}"
    );
    assert!(
        choice["message"]["content"]
            .as_str()
            .is_some_and(|s| !s.is_empty()),
        "message content must be a non-empty string: {response}"
    );
    assert_eq!(
        choice["finish_reason"].as_str(),
        Some("length"),
        "finish_reason must be length when max_tokens is exhausted: {response}"
    );

    let usage = &response["usage"];
    let prompt_tokens = usage["prompt_tokens"]
        .as_u64()
        .ok_or_else(|| anyhow!("usage.prompt_tokens missing: {response}"))?;
    let completion_tokens = usage["completion_tokens"]
        .as_u64()
        .ok_or_else(|| anyhow!("usage.completion_tokens missing: {response}"))?;
    let total_tokens = usage["total_tokens"]
        .as_u64()
        .ok_or_else(|| anyhow!("usage.total_tokens missing: {response}"))?;
    assert!(
        prompt_tokens > 0,
        "prompt_tokens must be positive: {response}"
    );
    assert_eq!(
        completion_tokens, 4,
        "completion_tokens must equal max_tokens: {response}"
    );
    assert_eq!(
        total_tokens,
        prompt_tokens + completion_tokens,
        "total_tokens must equal prompt + completion: {response}"
    );

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_completions_streaming_emits_role_content_and_done() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;

    let response = client
        .post(format!("{}/v1/chat/completions", server.base_url))
        .json(&json!({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "alpha"}],
            "max_tokens": 2,
            "temperature": 0.0,
            "stream": true
        }))
        .send()
        .await?
        .error_for_status()?;
    assert_event_stream_content_type(&response)?;
    let stream_text = response.text().await?;

    let chunks = parse_terminal_sse_chunks(&stream_text)?;

    assert!(
        !chunks.is_empty(),
        "streaming response must contain at least one chunk"
    );

    for chunk in &chunks {
        assert_eq!(
            chunk["object"].as_str(),
            Some("chat.completion.chunk"),
            "chunk object must be chat.completion.chunk: {chunk}"
        );
    }

    assert_eq!(
        chunks[0]["choices"][0]["delta"]["role"].as_str(),
        Some("assistant"),
        "first chunk must declare assistant role: {}",
        chunks[0]
    );

    let content: String = chunks
        .iter()
        .filter_map(|chunk| chunk["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(
        content, " alpha alpha",
        "streaming response must emit the simulated content: {stream_text}"
    );

    let last_chunk = chunks.last().expect("at least one chunk");
    assert_eq!(
        last_chunk["choices"][0]["finish_reason"].as_str(),
        Some("length"),
        "last chunk must carry finish_reason: {last_chunk}"
    );

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chat_completions_usage_with_stream_options() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;

    let response = client
        .post(format!("{}/v1/chat/completions", server.base_url))
        .json(&json!({
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "alpha"}],
            "max_tokens": 3,
            "temperature": 0.0,
            "stream": true,
            "stream_options": {"include_usage": true}
        }))
        .send()
        .await?
        .error_for_status()?;
    assert_event_stream_content_type(&response)?;
    let stream_text = response.text().await?;

    let chunks = parse_terminal_sse_chunks(&stream_text)?;
    let usage_indices: Vec<usize> = chunks
        .iter()
        .enumerate()
        .filter_map(|(index, chunk)| (!chunk["usage"].is_null()).then_some(index))
        .collect();
    assert_eq!(
        usage_indices.len(),
        1,
        "stream_options.include_usage must emit exactly one usage chunk: {stream_text}"
    );

    let usage_index = usage_indices[0];
    let usage_chunk = &chunks[usage_index];
    assert_eq!(
        usage_chunk["choices"].as_array().map(Vec::len),
        Some(0),
        "usage chunk must not contain choices: {usage_chunk}"
    );

    let finish_index = chunks
        .iter()
        .position(|chunk| chunk["choices"][0]["finish_reason"] == "length")
        .ok_or_else(|| anyhow!("streaming response has no length finish chunk: {stream_text}"))?;
    assert_eq!(
        usage_index,
        finish_index + 1,
        "usage chunk must immediately follow the finish chunk: {stream_text}"
    );
    assert_eq!(
        usage_index,
        chunks.len() - 1,
        "usage chunk must be the final JSON chunk before [DONE]: {stream_text}"
    );

    let usage = &usage_chunk["usage"];
    let prompt_tokens = usage["prompt_tokens"]
        .as_u64()
        .ok_or_else(|| anyhow!("usage.prompt_tokens missing: {usage_chunk}"))?;
    let completion_tokens = usage["completion_tokens"]
        .as_u64()
        .ok_or_else(|| anyhow!("usage.completion_tokens missing: {usage_chunk}"))?;
    let total_tokens = usage["total_tokens"]
        .as_u64()
        .ok_or_else(|| anyhow!("usage.total_tokens missing: {usage_chunk}"))?;
    assert!(
        prompt_tokens > 0,
        "usage.prompt_tokens must be positive: {usage_chunk}"
    );
    assert_eq!(
        completion_tokens, 3,
        "usage.completion_tokens must equal max_tokens: {usage_chunk}"
    );
    assert_eq!(
        total_tokens,
        prompt_tokens + completion_tokens,
        "usage.total_tokens must equal prompt + completion: {usage_chunk}"
    );

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simulated_frontend_metadata_contract_is_executable() -> Result<()> {
    let model_dir = model_dir_with_minimal_metadata()?;
    for file in ["tokenizer.json", "tokenizer_config.json", "config.json"] {
        if !model_dir.path().join(file).is_file() {
            bail!("minimal simulated frontend metadata fixture is missing {file}");
        }
    }

    let server = SimServer::spawn_with_model_dir(model_dir).await?;
    server.shutdown().await?;

    let error = match SimServer::spawn_with_model_dir(empty_model_dir()?).await {
        Ok(server) => {
            server.shutdown().await?;
            bail!("empty local model metadata directory should fail frontend startup");
        }
        Err(error) => error,
    };
    let message = format!("{error:#}");
    if !message.contains("supported tokenizer file") || !message.contains("tokenizer.json") {
        bail!("empty metadata dir failed with unexpected error: {message}");
    }

    Ok(())
}

async fn assert_models_endpoint(client: &Client, base_url: &str, model_name: &str) -> Result<()> {
    let models: Value = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let advertised = models["data"]
        .as_array()
        .ok_or_else(|| anyhow!("/v1/models response has no data array"))?;
    if !advertised.iter().any(|model| model["id"] == model_name) {
        bail!("/v1/models did not advertise {model_name}: {models}");
    }

    Ok(())
}

async fn assert_non_streaming_completion_has_output(
    client: &Client,
    base_url: &str,
    model_name: &str,
) -> Result<()> {
    let completion: Value = post_completion(client, base_url, model_name, false).await?;
    let text = completion["choices"][0]["text"]
        .as_str()
        .ok_or_else(|| anyhow!("non-streaming completion has no text: {completion}"))?;
    if text.is_empty() {
        bail!("non-streaming completion returned empty text for max_tokens > 0");
    }

    Ok(())
}

async fn assert_streaming_completion_emits_done(
    client: &Client,
    base_url: &str,
    model_name: &str,
) -> Result<()> {
    let stream = post_completion_stream(client, base_url, model_name).await?;
    if !stream.lines().any(|line| line.trim() == "data: [DONE]") {
        bail!("streaming completion did not emit terminal data: [DONE]: {stream}");
    }

    Ok(())
}

async fn post_completion(
    client: &Client,
    base_url: &str,
    model_name: &str,
    stream: bool,
) -> Result<Value> {
    let response = client
        .post(format!("{base_url}/v1/completions"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(completion_body(model_name, stream).to_string())
        .send()
        .await?
        .error_for_status()?;
    response
        .json()
        .await
        .context("failed to parse non-streaming completion response")
}

async fn post_completion_stream(
    client: &Client,
    base_url: &str,
    model_name: &str,
) -> Result<String> {
    client
        .post(format!("{base_url}/v1/completions"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(completion_body(model_name, true).to_string())
        .send()
        .await?
        .error_for_status()?
        .text()
        .await
        .context("failed to read streaming completion response")
}

fn test_client() -> Result<Client> {
    Client::builder()
        .timeout(HTTP_TIMEOUT)
        .build()
        .context("failed to build HTTP test client")
}

fn completion_body(model_name: &str, stream: bool) -> Value {
    json!({
        "model": model_name,
        "prompt": [1, 2],
        "max_tokens": 3,
        "temperature": 0.0,
        "ignore_eos": true,
        "stream": stream
    })
}

fn assert_event_stream_content_type(response: &reqwest::Response) -> Result<()> {
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .ok_or_else(|| anyhow!("streaming response is missing Content-Type"))?
        .to_str()
        .context("streaming response has an invalid Content-Type")?;
    let media_type = content_type.split(';').next().unwrap_or_default().trim();
    if media_type != "text/event-stream" {
        bail!("streaming response must use text/event-stream, got {content_type}");
    }
    Ok(())
}

fn parse_terminal_sse_chunks(stream: &str) -> Result<Vec<Value>> {
    let normalized = stream.replace("\r\n", "\n").replace('\r', "\n");
    let mut payloads = Vec::new();
    let mut data_lines = Vec::new();

    for line in normalized.split_terminator('\n') {
        if line.is_empty() {
            if !data_lines.is_empty() {
                payloads.push(data_lines.join("\n"));
                data_lines.clear();
            }
            continue;
        }
        if line.starts_with(':') {
            continue;
        }

        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        if field == "data" {
            data_lines.push(value.strip_prefix(' ').unwrap_or(value));
        }
    }
    if !data_lines.is_empty() {
        bail!("streaming response ended before its final SSE event was dispatched: {stream}");
    }

    let done_count = payloads
        .iter()
        .filter(|payload| payload.as_str() == "[DONE]")
        .count();
    if done_count != 1 {
        bail!("streaming response must contain exactly one data: [DONE]: {stream}");
    }
    if payloads.last().map(String::as_str) != Some("[DONE]") {
        bail!("streaming response must end with data: [DONE]: {stream}");
    }

    payloads[..payloads.len() - 1]
        .iter()
        .map(|payload| serde_json::from_str(payload))
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("failed to parse streaming chunks")
}

#[test]
fn sse_parser_requires_dispatched_events_and_unique_terminal_done() -> Result<()> {
    let valid = "data: {\"object\":\"chat.completion.chunk\"}\n\ndata:[DONE]\n\n";
    assert_eq!(parse_terminal_sse_chunks(valid)?.len(), 1);

    let missing_separator = "data: {\"object\":\"chat.completion.chunk\"}\ndata:[DONE]\n\n";
    assert!(parse_terminal_sse_chunks(missing_separator).is_err());

    let duplicate_done = "data: {}\n\ndata:[DONE]\n\ndata: [DONE]\n\n";
    assert!(parse_terminal_sse_chunks(duplicate_done).is_err());

    let undispatched_done = "data: {}\n\ndata:[DONE]";
    assert!(parse_terminal_sse_chunks(undispatched_done).is_err());

    let line_terminated_but_undispatched_done = "data: {}\n\ndata:[DONE]\n";
    assert!(parse_terminal_sse_chunks(line_terminated_but_undispatched_done).is_err());

    Ok(())
}

async fn wait_for_health(client: &Client, base_url: &str) -> Result<()> {
    let health_url = format!("{base_url}/health");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    loop {
        if tokio::time::Instant::now() >= deadline {
            bail!("timed out waiting for sim frontend health at {health_url}");
        }

        match client
            .get(&health_url)
            .timeout(Duration::from_secs(1))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(_) | Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
        }
    }
}

fn reserve_loopback_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .context("failed to reserve loopback port for sim e2e test")?;
    Ok(listener.local_addr()?.port())
}

const TINY_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {
      "id": 0,
      "content": "<unk>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Whitespace"
  },
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "<unk>": 0,
      "alpha": 1,
      "beta": 2
    },
    "unk_token": "<unk>"
  }
}"#;

const TINY_TOKENIZER_CONFIG_JSON: &str = r#"{
  "unk_token": "<unk>",
  "tokenizer_class": "PreTrainedTokenizerFast",
  "chat_template": "{% for message in messages %}{{ message.content }}{% endfor %}"
}"#;

const TINY_CONFIG_JSON: &str = r#"{
  "model_type": "pegainfer_sim",
  "max_position_embeddings": 128,
  "vocab_size": 3
}"#;
