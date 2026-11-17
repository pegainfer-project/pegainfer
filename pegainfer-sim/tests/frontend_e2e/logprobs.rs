use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use reqwest::Client;
use serde_json::Value;
use serde_json::json;

use super::SimServer;
use super::assert_event_stream_content_type;
use super::completion_body;
use super::parse_terminal_sse_chunks;
use super::test_client;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicit_prompt_logprobs_returns_scored_prompt_positions() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;
    // The previously-500 case: an explicit multi-token prompt_logprobs request
    // must come back with one map per prompt position instead of an error.
    let mut body = completion_body(&server.model_name, false);
    body["prompt"] = json!([1, 2, 1]);
    body["prompt_logprobs"] = json!(2);
    body["return_tokens_as_token_ids"] = json!(true);

    let completion = post_completion_body(&client, &server.base_url, &body).await?;
    let choice = &completion["choices"][0];
    let prompt_logprobs = choice["prompt_logprobs"]
        .as_array()
        .ok_or_else(|| anyhow!("missing prompt_logprobs: {completion}"))?;
    assert_eq!(
        prompt_logprobs.len(),
        3,
        "one entry per prompt token: {completion}"
    );
    assert!(
        prompt_logprobs[0].is_null(),
        "leading position: {completion}"
    );
    for (index, position) in prompt_logprobs.iter().enumerate().skip(1) {
        let map = position
            .as_object()
            .ok_or_else(|| anyhow!("position {index} must be a map: {completion}"))?;
        let scored = [1, 2, 1][index].to_string();
        assert!(
            map.contains_key(&scored),
            "position {index} must contain its scored token {scored}: {completion}"
        );
        assert_eq!(
            map[&scored]["rank"], 3,
            "prompt token is behind two candidates"
        );
    }
    // Completion logprobs were not requested.
    assert!(
        choice.get("logprobs").is_none_or(Value::is_null),
        "no completion logprobs without a logprobs request: {completion}"
    );

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicit_zero_logprobs_returns_scored_token_only() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;
    // logprobs=0 is a real request (scored token, no alternatives) — not the
    // disabled value. Response assembly used to expect a logprob the engine
    // never emitted.
    let mut body = completion_body(&server.model_name, false);
    body["logprobs"] = json!(0);
    body["return_tokens_as_token_ids"] = json!(true);

    let completion = post_completion_body(&client, &server.base_url, &body).await?;
    let logprobs = &completion["choices"][0]["logprobs"];
    let positions = assert_scored_positions(logprobs)?;
    assert_eq!(
        positions as u64,
        completion["usage"]["completion_tokens"].as_u64().unwrap()
    );

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicit_zero_prompt_logprobs_returns_scored_token_only() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;
    let mut body = completion_body(&server.model_name, false);
    body["prompt_logprobs"] = json!(0);
    body["return_tokens_as_token_ids"] = json!(true);

    let completion = post_completion_body(&client, &server.base_url, &body).await?;
    let prompt_logprobs = completion["choices"][0]["prompt_logprobs"]
        .as_array()
        .ok_or_else(|| anyhow!("prompt_logprobs=0 missing payload: {completion}"))?;
    assert_eq!(
        prompt_logprobs.len(),
        body["prompt"].as_array().unwrap().len()
    );
    assert!(prompt_logprobs[0].is_null());
    let scored = body["prompt"][1].to_string();
    assert_eq!(prompt_logprobs[1][&scored]["rank"], 3);
    assert!(
        prompt_logprobs[1]
            .as_object()
            .unwrap()
            .contains_key(&scored)
    );

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn single_token_prompt_logprobs_answers_from_the_request() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;
    let mut body = completion_body(&server.model_name, false);
    body["prompt"] = json!([1]);
    body["prompt_logprobs"] = json!(1);
    body["return_tokens_as_token_ids"] = json!(true);

    let completion = post_completion_body(&client, &server.base_url, &body).await?;
    let prompt_logprobs = completion["choices"][0]["prompt_logprobs"]
        .as_array()
        .ok_or_else(|| anyhow!("single-token prompt_logprobs missing: {completion}"))?;
    assert_eq!(prompt_logprobs.len(), 1);
    assert!(prompt_logprobs[0].is_null());

    server.shutdown().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn streaming_zero_logprobs_returns_aligned_scores_and_finishes() -> Result<()> {
    let server = SimServer::spawn().await?;
    let client = test_client()?;
    let mut body = completion_body(&server.model_name, true);
    body["logprobs"] = json!(0);
    body["return_tokens_as_token_ids"] = json!(true);
    let response = client
        .post(format!("{}/v1/completions", server.base_url))
        .json(&body)
        .send()
        .await?
        .error_for_status()?;
    assert_event_stream_content_type(&response)?;
    let chunks = parse_terminal_sse_chunks(&response.text().await?)?;
    let mut scored_positions = 0;
    let mut finished = false;
    for chunk in &chunks {
        assert!(chunk.get("error").is_none(), "stream error: {chunk}");
        let choice = &chunk["choices"][0];
        if let Some(logprobs) = choice.get("logprobs").filter(|value| !value.is_null()) {
            scored_positions += assert_scored_positions(logprobs)?;
        }
        finished |= choice["finish_reason"] == "length";
    }
    assert!(finished, "missing length finish: {chunks:?}");
    assert_eq!(
        scored_positions as u64,
        body["max_tokens"].as_u64().unwrap()
    );
    server.shutdown().await
}

fn assert_scored_positions(logprobs: &Value) -> Result<usize> {
    let tokens = logprobs["tokens"].as_array().context("missing tokens")?;
    let scores = logprobs["token_logprobs"]
        .as_array()
        .context("missing scores")?;
    let alternatives = logprobs["top_logprobs"]
        .as_array()
        .context("missing alternatives")?;
    assert_eq!(tokens.len(), scores.len());
    assert_eq!(tokens.len(), alternatives.len());
    for (index, token) in tokens.iter().enumerate() {
        let scored = token.as_str().context("token must be a string")?;
        assert!(scores[index].is_number(), "position {index}: {logprobs}");
        let map = alternatives[index]
            .as_object()
            .context("missing scored token map")?;
        assert_eq!(
            map.get(scored),
            Some(&scores[index]),
            "position {index}: {logprobs}"
        );
    }
    Ok(tokens.len())
}

async fn post_completion_body(client: &Client, base_url: &str, body: &Value) -> Result<Value> {
    client
        .post(format!("{base_url}/v1/completions"))
        .header(reqwest::header::CONTENT_TYPE, "application/json")
        .body(body.to_string())
        .send()
        .await?
        .error_for_status()?
        .json()
        .await
        .context("failed to parse completion response")
}
