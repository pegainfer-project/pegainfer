//! Tool-call round-trip contract test for the OpenAI `/v1/chat/completions`
//! path, driven entirely on CPU through `pegainfer-sim`.
//!
//! This is the "protocol layer testable on pegainfer-sim" deliverable from the
//! coding-agent workstream (roadmap-2026-h2 §5). It does NOT test the upstream
//! vLLM tool parser in isolation (upstream already has `roundtrip.rs` for that).
//! It tests the seam pegainfer owns: a scripted engine byte stream ->
//! pegainfer's `TokenEvent` contract + detokenize -> upstream `Auto` parser
//! selection -> OpenAI `tool_calls` output, in both non-streaming and streaming
//! form.
//!
//! Several non-obvious constraints make this work — including two real wiring
//! gaps this test caught by failing (the frontend runs a *second* `Auto` parser
//! for reasoning; the upstream frontend suppresses the final token id on `Stop`,
//! so the script must pad a trailing EOS the engine never actually emits — see
//! #584). Each is documented at the code site it justifies rather than in a list
//! here, so the rationale stays next to the line it explains and can't drift.

use std::fs;
use std::net::TcpListener;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use pegainfer_sim::SimulatedEngineConfig;
use pegainfer_sim::start_engine;
use reqwest::Client;
use serde_json::Map;
use serde_json::Value;
use serde_json::json;
use tempfile::TempDir;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

const HTTP_TIMEOUT: Duration = Duration::from_secs(10);
const MODEL_NAME: &str = "pegainfer-sim-toolcall";

// The three scripted completion chunks to test toolcall parsing in streaming.
// Concatenated they form exactly:
//   <tool_call>
//   {"name": "get_weather", "arguments": {"location": "SF"}}
//   </tool_call>
//
// The split points fall *inside* the JSON ("argu|ments", "S|F") so the streaming
// path must accumulate a tool call across multiple detokenized deltas.
const CHUNK_1: &str = "<tool_call>\n{\"name\": \"get_weather\", \"argu";
const CHUNK_2: &str = "ments\": {\"location\": \"S";
const CHUNK_3: &str = "F\"}}\n</tool_call>";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn non_streaming_tool_call_round_trips_to_openai_tool_calls() -> Result<()> {
    let server = ToolCallSimServer::spawn().await?;
    let client = test_client()?;

    let response = client
        .post(format!("{}/v1/chat/completions", server.base_url))
        .json(&chat_request(false))
        .send()
        .await?;
    let status = response.status();
    let text = response.text().await?;
    if !status.is_success() {
        bail!("chat completion returned {status}: {text}");
    }
    let body: Value =
        serde_json::from_str(&text).context("failed to parse non-streaming chat response")?;

    let tool_calls = body["choices"][0]["message"]["tool_calls"]
        .as_array()
        .ok_or_else(|| anyhow!("response has no message.tool_calls array: {body}"))?;
    if tool_calls.len() != 1 {
        bail!(
            "expected exactly one tool call, got {}: {body}",
            tool_calls.len()
        );
    }
    assert_tool_call_is_get_weather_sf(&tool_calls[0], &body)?;

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_tool_call_reassembles_across_chunks() -> Result<()> {
    let server = ToolCallSimServer::spawn().await?;
    let client = test_client()?;

    let raw = client
        .post(format!("{}/v1/chat/completions", server.base_url))
        .json(&chat_request(true))
        .send()
        .await?
        .error_for_status()?
        .text()
        .await
        .context("failed to read streaming chat response")?;

    // Accumulate the streamed tool-call name + arguments across SSE frames.
    let mut name = String::new();
    let mut arguments = String::new();
    let mut argument_fragments = 0usize;
    let mut saw_done = false;

    for line in raw.lines() {
        let payload = match line.strip_prefix("data: ") {
            Some(p) => p.trim(),
            None => continue,
        };
        if payload == "[DONE]" {
            saw_done = true;
            continue;
        }
        let frame: Value = serde_json::from_str(payload)
            .with_context(|| format!("failed to parse SSE frame: {payload}"))?;
        let Some(deltas) = frame["choices"][0]["delta"]["tool_calls"].as_array() else {
            continue;
        };
        for delta in deltas {
            if let Some(n) = delta["function"]["name"].as_str() {
                name.push_str(n);
            }
            if let Some(args) = delta["function"]["arguments"].as_str() {
                if !args.is_empty() {
                    arguments.push_str(args);
                    argument_fragments += 1;
                }
            }
        }
    }

    if !saw_done {
        bail!("streaming tool-call response did not terminate with data: [DONE]:\n{raw}");
    }
    if name != "get_weather" {
        bail!("streamed tool name reassembled to {name:?}, expected \"get_weather\":\n{raw}");
    }
    // The whole point of this test: the arguments must have arrived in more than
    // one delta and still reassemble to valid JSON.
    if argument_fragments < 2 {
        bail!(
            "expected tool-call arguments to stream across >= 2 fragments, got {argument_fragments}:\n{raw}"
        );
    }
    let parsed: Value = serde_json::from_str(&arguments).with_context(|| {
        format!("reassembled streamed arguments are not valid JSON: {arguments:?}")
    })?;
    if parsed["location"] != json!("SF") {
        bail!("streamed arguments location != SF: {parsed}");
    }

    server.shutdown().await
}

fn assert_tool_call_is_get_weather_sf(tool_call: &Value, body: &Value) -> Result<()> {
    if tool_call["type"] != json!("function") {
        bail!("tool call is not type=function: {body}");
    }
    if tool_call["function"]["name"] != json!("get_weather") {
        bail!("tool call name != get_weather: {body}");
    }
    let arguments = tool_call["function"]["arguments"]
        .as_str()
        .ok_or_else(|| anyhow!("tool call arguments are not a string: {body}"))?;
    let parsed: Value = serde_json::from_str(arguments)
        .with_context(|| format!("tool call arguments are not valid JSON: {arguments:?}"))?;
    if parsed["location"] != json!("SF") {
        bail!("tool call arguments location != SF: {parsed}");
    }
    Ok(())
}

fn chat_request(stream: bool) -> Value {
    json!({
        "model": MODEL_NAME,
        // Prompt content is irrelevant: scripted mode ignores the prompt tokens
        // and replays the tool-call chunks regardless.
        "messages": [{ "role": "user", "content": "call the weather tool" }],
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": { "location": { "type": "string" } },
                    "required": ["location"]
                }
            }
        }],
        "tool_choice": "auto",
        "max_tokens": 16,
        "temperature": 0.0,
        "stream": stream
    })
}

struct ToolCallSimServer {
    base_url: String,
    shutdown: CancellationToken,
    task: JoinHandle<Result<()>>,
    _model_dir: TempDir,
}

impl ToolCallSimServer {
    async fn spawn() -> Result<Self> {
        // Script = the three tool-call chunks (ids 1/2/3, see fuse_tokenizer_json)
        // plus 6 (EOS). The trailing EOS is *not* incremental-detokenizer buffering:
        // the upstream frontend's terminal-stop-token suppression drops the last
        // token id whenever a request finishes with `Stop`. pegainfer engines never
        // emit an EOS token — they signal `Finished(Stop)` with no preceding token
        // event — so without this pad the suppression would eat the final *content*
        // token and truncate the JSON to `{"location": "S`. The throwaway EOS
        // absorbs the suppression instead. Engine-side contract alignment: #584.
        let script = vec![1u32, 2, 3, 6];
        let model_dir = qwen_model_dir()?;
        let port = reserve_loopback_port()?;
        let base_url = format!("http://127.0.0.1:{port}");

        let shutdown = CancellationToken::new();
        let server_shutdown = shutdown.clone();
        let engine = start_engine(
            SimulatedEngineConfig::new(0.0, 1000.0, 0.0, 0)?.with_scripted_completion(script),
        );
        let model_path = model_dir.path().to_path_buf();
        let mut task = tokio::spawn(async move {
            pegainfer_frontend::vllm::serve(
                std::future::ready(Ok(engine.into())),
                &model_path,
                vec![MODEL_NAME.to_string()],
                port,
                Some(128),
                server_shutdown,
            )
            .await
        });

        let client = test_client()?;
        let health = tokio::select! {
            result = wait_for_health(&client, &base_url) => result,
            result = &mut task => {
                return match result {
                    Ok(Ok(())) => Err(anyhow!("sim frontend exited before becoming healthy")),
                    Ok(Err(error)) => Err(error).context("sim frontend exited before becoming healthy"),
                    Err(error) => Err(error).context("sim frontend task panicked"),
                };
            }
        };
        if let Err(error) = health {
            shutdown.cancel();
            let _ = tokio::time::timeout(Duration::from_secs(10), task).await;
            return Err(error);
        }

        Ok(Self {
            base_url,
            shutdown,
            task,
            _model_dir: model_dir,
        })
    }

    async fn shutdown(self) -> Result<()> {
        self.shutdown.cancel();
        tokio::time::timeout(Duration::from_secs(10), self.task)
            .await
            .context("timed out waiting for sim frontend shutdown")?
            .context("sim frontend task panicked")?
    }
}

/// Model dir at a path containing `qwen`. `ParserSelection::Auto` resolves the
/// parser from the *model path string* (`config.model`) by case-insensitive
/// substring match against a registry (`qwen` -> `qwen3_xml`) — a random temp
/// dir has no model-family name and would fail with `ParserUnavailableForModel`.
/// The `qwen-` prefix makes `Auto` select the qwen3 XML parser exactly as it
/// would for `models/Qwen3-4B` in production.
fn qwen_model_dir() -> Result<TempDir> {
    let dir = tempfile::Builder::new()
        .prefix("qwen-sim-toolcall-")
        .tempdir()
        .context("failed to create qwen-prefixed temp model dir")?;

    fs::write(dir.path().join("tokenizer.json"), fuse_tokenizer_json()?)
        .context("failed to write tool-call tokenizer.json")?;
    fs::write(
        dir.path().join("tokenizer_config.json"),
        serde_json::to_string_pretty(&json!({
            "unk_token": "<unk>",
            "eos_token": "<|endoftext|>",
            "tokenizer_class": "PreTrainedTokenizerFast",
            // Minimal template so /v1/chat/completions can render messages.
            "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}"
        }))?,
    )
    .context("failed to write tokenizer_config.json")?;
    fs::write(
        dir.path().join("config.json"),
        serde_json::to_string_pretty(&json!({
            "model_type": "pegainfer_sim",
            "max_position_embeddings": 128,
            "eos_token_id": 6,
            "vocab_size": 7
        }))?,
    )
    .context("failed to write config.json")?;

    Ok(dir)
}

/// WordLevel tokenizer + `Fuse` decoder whose ids 1/2/3 detokenize to the three
/// tool-call chunks. The sim emits token *ids*, detokenized by this served
/// `tokenizer.json`; a vocab of real words cannot spell a tool call, so the
/// vocab entries are the raw chunks and `Fuse` concatenates them byte-for-byte.
fn fuse_tokenizer_json() -> Result<String> {
    let mut vocab = Map::new();
    vocab.insert("<unk>".to_string(), json!(0));
    vocab.insert(CHUNK_1.to_string(), json!(1));
    vocab.insert(CHUNK_2.to_string(), json!(2));
    vocab.insert(CHUNK_3.to_string(), json!(3));
    // The frontend also runs `reasoning_parser: Auto`, which on a `qwen` path
    // selects the qwen3 reasoning parser and requires `<think>`/`</think>`
    // delimiter tokens to exist (real Qwen3 tokenizers carry them). Without
    // these the chat request 500s before ever reaching the tool parser.
    vocab.insert("<think>".to_string(), json!(4));
    vocab.insert("</think>".to_string(), json!(5));
    // EOS id padded onto the script in `ToolCallSimServer::spawn` (see there for why).
    vocab.insert("<|endoftext|>".to_string(), json!(6));

    let special = |id: u32, content: &str| {
        json!({
            "id": id,
            "content": content,
            "single_word": false,
            "lstrip": false,
            "rstrip": false,
            "normalized": false,
            "special": true
        })
    };
    let tokenizer = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            special(0, "<unk>"),
            special(4, "<think>"),
            special(5, "</think>"),
            special(6, "<|endoftext|>")
        ],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        // Fuse concatenates decoded tokens with no separator, so the chunks
        // reassemble byte-for-byte.
        "decoder": { "type": "Fuse" },
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": "<unk>"
        }
    });
    serde_json::to_string_pretty(&tokenizer).context("failed to serialize fuse tokenizer")
}

fn test_client() -> Result<Client> {
    Client::builder()
        .timeout(HTTP_TIMEOUT)
        .build()
        .context("failed to build HTTP test client")
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
        .context("failed to reserve loopback port for tool-call e2e test")?;
    Ok(listener.local_addr()?.port())
}

#[test]
fn scripted_chunks_concatenate_to_expected_payload() {
    let joined = format!("{CHUNK_1}{CHUNK_2}{CHUNK_3}");
    assert_eq!(
        joined,
        "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"location\": \"SF\"}}\n</tool_call>"
    );
    // The inner JSON object must be valid on its own.
    let inner = joined
        .trim_start_matches("<tool_call>")
        .trim()
        .trim_end_matches("</tool_call>")
        .trim();
    let parsed: Value = serde_json::from_str(inner).expect("inner tool-call JSON is valid");
    assert_eq!(parsed["name"], json!("get_weather"));
    assert_eq!(parsed["arguments"]["location"], json!("SF"));
}
