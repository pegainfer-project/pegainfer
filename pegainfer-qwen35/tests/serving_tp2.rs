use std::net::TcpListener;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use pegainfer_frontend::engine::EngineLoadOptions;
use reqwest::Client;
use serde_json::Value;
use serde_json::json;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

mod common;

const DEFAULT_MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.5-4B");
const MODEL_NAME: &str = "qwen35-tp2-serving-smoke";
const HTTP_TIMEOUT: Duration = Duration::from_secs(120);

struct Qwen35Tp2Server {
    base_url: String,
    shutdown: CancellationToken,
    task: JoinHandle<Result<()>>,
}

impl Qwen35Tp2Server {
    async fn shutdown(self) -> Result<()> {
        self.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(30), self.task)
            .await
            .context("timed out waiting for Qwen3.5 TP2 frontend shutdown")?
            .context("Qwen3.5 TP2 frontend task panicked")?
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires two CUDA devices, CUDA-12 NCCL, Qwen3.5 weights, and real HTTP frontend startup"]
async fn qwen35_tp2_serves_openai_completions_over_http() -> Result<()> {
    let engine_model_path = PathBuf::from(get_model_path());
    let frontend_model_path = PathBuf::from(get_frontend_model_path(&engine_model_path));
    let server = spawn_ready_server(engine_model_path, frontend_model_path, 1).await?;
    let client = test_client()?;

    assert_models_endpoint(&client, &server.base_url).await?;
    assert_non_streaming_completion(&client, &server.base_url).await?;
    assert_streaming_completion(&client, &server.base_url).await?;
    assert_concurrent_completions(&client, &server.base_url).await?;
    assert_invalid_cuda_graph_tp_startup_fails(&get_model_path())?;

    server.shutdown().await
}

async fn spawn_ready_server(
    engine_model_path: PathBuf,
    frontend_model_path: PathBuf,
    max_prefill_tokens: usize,
) -> Result<Qwen35Tp2Server> {
    let device_ordinals = common::tp2_device_ordinals();
    let handle = tokio::task::spawn_blocking(move || {
        pegainfer_qwen35::start_engine_with_capacity(
            &engine_model_path,
            EngineLoadOptions {
                enable_cuda_graph: false,
                device_ordinals,
                seed: 42,
                ..EngineLoadOptions::default()
            },
            8,
            max_prefill_tokens,
        )
    })
    .await
    .context("Qwen3.5 TP2 engine loader thread panicked")??;

    let port = reserve_loopback_port()?;
    let base_url = format!("http://127.0.0.1:{port}");
    let shutdown = CancellationToken::new();
    let server_shutdown = shutdown.clone();
    let mut task = tokio::spawn(async move {
        pegainfer_frontend::vllm::serve(
            std::future::ready(Ok(pegainfer_frontend::engine::LaunchedEngine::Handle(
                handle,
            ))),
            &frontend_model_path,
            vec![MODEL_NAME.to_string()],
            port,
            None,
            server_shutdown,
        )
        .await
    });

    let client = test_client()?;
    let health_result = tokio::select! {
        result = wait_for_health(&client, &base_url) => result,
        result = &mut task => {
            return match result {
                Ok(Ok(())) => Err(anyhow!("Qwen3.5 TP2 frontend exited before becoming healthy")),
                Ok(Err(error)) => Err(error).context("Qwen3.5 TP2 frontend exited before becoming healthy"),
                Err(error) => Err(error).context("Qwen3.5 TP2 frontend task panicked"),
            };
        }
    };

    if let Err(error) = health_result {
        shutdown.cancel();
        let _ = tokio::time::timeout(Duration::from_secs(30), task).await;
        return Err(error).context("Qwen3.5 TP2 frontend did not become healthy");
    }

    Ok(Qwen35Tp2Server {
        base_url,
        shutdown,
        task,
    })
}

async fn assert_models_endpoint(client: &Client, base_url: &str) -> Result<()> {
    let models: Value = client
        .get(format!("{base_url}/v1/models"))
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let advertised = models["data"]
        .as_array()
        .ok_or_else(|| anyhow!("/v1/models response has no data array: {models}"))?;
    if !advertised.iter().any(|model| model["id"] == MODEL_NAME) {
        bail!("/v1/models did not advertise {MODEL_NAME}: {models}");
    }
    Ok(())
}

async fn assert_non_streaming_completion(client: &Client, base_url: &str) -> Result<()> {
    let completion = post_completion(client, base_url, completion_body(false, 5, 1)).await?;
    let choice = &completion["choices"][0];
    let text = choice["text"]
        .as_str()
        .ok_or_else(|| anyhow!("non-streaming completion has no text: {completion}"))?;
    if text.is_empty() {
        bail!("non-streaming completion returned empty text for max_tokens > 0");
    }
    let finish_reason = choice["finish_reason"]
        .as_str()
        .ok_or_else(|| anyhow!("non-streaming completion has no finish_reason: {completion}"))?;
    if finish_reason != "length" {
        bail!("expected length finish_reason for ignore_eos request, got {completion}");
    }
    assert_usage(&completion, 5)?;
    assert_logprobs(&completion)?;
    Ok(())
}

async fn assert_streaming_completion(client: &Client, base_url: &str) -> Result<()> {
    let stream = post_completion_stream(client, base_url, completion_body(true, 4, 0)).await?;
    let data_lines: Vec<&str> = stream
        .lines()
        .filter(|line| line.starts_with("data: "))
        .collect();
    if !data_lines.iter().any(|line| line.trim() == "data: [DONE]") {
        bail!("streaming completion did not emit terminal data: [DONE]: {stream}");
    }
    if !data_lines
        .iter()
        .filter(|line| line.trim() != "data: [DONE]")
        .any(|line| line.contains("\"choices\""))
    {
        bail!("streaming completion did not emit any choice payloads: {stream}");
    }
    Ok(())
}

async fn assert_concurrent_completions(client: &Client, base_url: &str) -> Result<()> {
    let first = post_completion(client, base_url, completion_body(false, 3, 0));
    let second = post_completion(client, base_url, alternate_completion_body(false, 3, 1));
    let (first, second) = tokio::try_join!(first, second)?;
    assert_usage(&first, 3)?;
    assert_usage(&second, 3)?;
    assert_logprobs(&second)?;
    Ok(())
}

fn assert_invalid_cuda_graph_tp_startup_fails(model_path: &str) -> Result<()> {
    let Err(error) = pegainfer_qwen35::start_engine_with_capacity(
        Path::new(model_path),
        EngineLoadOptions {
            enable_cuda_graph: true,
            device_ordinals: common::tp2_device_ordinals(),
            seed: 42,
            ..EngineLoadOptions::default()
        },
        8,
        1,
    ) else {
        bail!("TP2 + CUDA Graph must fail before serving requests");
    };
    let message = error.to_string();
    if !message.contains("eager execution only") {
        bail!("unexpected TP2 + CUDA Graph startup error: {message}");
    }
    Ok(())
}

async fn post_completion(client: &Client, base_url: &str, body: Value) -> Result<Value> {
    client
        .post(format!("{base_url}/v1/completions"))
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await
        .context("failed to parse non-streaming completion response")
}

async fn post_completion_stream(client: &Client, base_url: &str, body: Value) -> Result<String> {
    client
        .post(format!("{base_url}/v1/completions"))
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await
        .context("failed to read streaming completion response")
}

fn completion_body(stream: bool, max_tokens: usize, logprobs: usize) -> Value {
    let mut body = json!({
        "model": MODEL_NAME,
        "prompt": [151_644, 872, 198, 9707, 151_645, 198, 151_644, 77091, 198],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": true,
        "stream": stream
    });
    if logprobs > 0 {
        body["logprobs"] = json!(logprobs);
    }
    body
}

fn alternate_completion_body(stream: bool, max_tokens: usize, logprobs: usize) -> Value {
    let mut body = json!({
        "model": MODEL_NAME,
        "prompt": [151_644, 872, 198, 3838, 374, 220, 17, 489, 220, 17, 30, 151_645, 198, 151_644, 77091, 198],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "ignore_eos": true,
        "stream": stream
    });
    if logprobs > 0 {
        body["logprobs"] = json!(logprobs);
    }
    body
}

fn assert_usage(completion: &Value, expected_completion_tokens: usize) -> Result<()> {
    let completion_tokens = completion["usage"]["completion_tokens"]
        .as_u64()
        .ok_or_else(|| {
            anyhow!("completion response has no usage.completion_tokens: {completion}")
        })?;
    if completion_tokens != expected_completion_tokens as u64 {
        bail!(
            "expected {expected_completion_tokens} completion tokens, got {completion_tokens}: {completion}"
        );
    }
    let prompt_tokens = completion["usage"]["prompt_tokens"]
        .as_u64()
        .ok_or_else(|| anyhow!("completion response has no usage.prompt_tokens: {completion}"))?;
    if prompt_tokens == 0 {
        bail!("completion response reported zero prompt tokens: {completion}");
    }
    Ok(())
}

fn assert_logprobs(completion: &Value) -> Result<()> {
    let logprobs = &completion["choices"][0]["logprobs"];
    if logprobs.is_null() {
        bail!("completion requested logprobs but response has null logprobs: {completion}");
    }
    let token_logprobs = logprobs["token_logprobs"]
        .as_array()
        .ok_or_else(|| anyhow!("logprobs.token_logprobs is not an array: {completion}"))?;
    if token_logprobs.is_empty() {
        bail!("logprobs.token_logprobs is empty: {completion}");
    }
    if !token_logprobs.iter().all(|value| value.as_f64().is_some()) {
        bail!("logprobs.token_logprobs contains non-finite values: {completion}");
    }
    Ok(())
}

async fn wait_for_health(client: &Client, base_url: &str) -> Result<()> {
    let health_url = format!("{base_url}/health");
    let deadline = tokio::time::Instant::now() + Duration::from_secs(180);
    loop {
        if tokio::time::Instant::now() >= deadline {
            bail!("timed out waiting for Qwen3.5 TP2 frontend health at {health_url}");
        }

        match client
            .get(&health_url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(response) if response.status().is_success() => return Ok(()),
            Ok(_) | Err(_) => tokio::time::sleep(Duration::from_millis(500)).await,
        }
    }
}

fn test_client() -> Result<Client> {
    Client::builder()
        .no_proxy()
        .timeout(HTTP_TIMEOUT)
        .build()
        .context("failed to build HTTP test client")
}

fn reserve_loopback_port() -> Result<u16> {
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .context("failed to reserve loopback port for Qwen3.5 TP2 serving test")?;
    Ok(listener.local_addr()?.port())
}

fn get_model_path() -> String {
    std::env::var("PEGAINFER_TEST_MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL_PATH.to_string())
}

fn get_frontend_model_path(engine_model_path: &Path) -> String {
    std::env::var("PEGAINFER_TEST_FRONTEND_MODEL_PATH")
        .unwrap_or_else(|_| engine_model_path.to_string_lossy().into_owned())
}
