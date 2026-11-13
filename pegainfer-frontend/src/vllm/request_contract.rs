use anyhow::Context as _;
use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::body::to_bytes;
use axum::extract::Request;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::post;
use serde_json::json;
use tower::ServiceExt as _;

const REQUEST_BODY_LIMIT: usize = 16 * 1024 * 1024;

/// HTTP contract for an engine that emits greedy text without score/transfer
/// support. Reject before the stream opens so clients receive actionable 400s.
pub fn greedy_text_routes(vllm_router: Router) -> Router {
    Router::new()
        .route("/v1/completions", post(validate_greedy_text))
        .route("/v1/chat/completions", post(validate_greedy_text))
        .with_state(vllm_router.clone())
        .fallback_service(vllm_router)
}

async fn validate_greedy_text(
    axum::extract::State(router): axum::extract::State<Router>,
    request: Request,
) -> Response {
    let result = async {
        let (mut parts, body) = request.into_parts();
        let body = to_bytes(body, REQUEST_BODY_LIMIT)
            .await
            .context("read request body")?;
        let value: serde_json::Value =
            serde_json::from_slice(&body).context("parse request JSON")?;
        let greedy = value
            .get("temperature")
            .and_then(serde_json::Value::as_f64)
            .is_some_and(|t| {
                // Match the engine's f32 sampling boundary, including decimal
                // values just below epsilon that round up when lowered.
                let temperature = t as f32;
                temperature.is_finite() && (0.0..1e-5).contains(&temperature)
            })
            || value.get("top_k").and_then(serde_json::Value::as_i64) == Some(1);
        anyhow::ensure!(
            greedy,
            "this engine requires greedy sampling: set temperature=0 or top_k=1"
        );
        for field in ["logprobs", "prompt_logprobs", "top_logprobs"] {
            anyhow::ensure!(
                value
                    .get(field)
                    .is_none_or(|v| v.is_null() || v.as_bool() == Some(false)),
                "this engine does not support {field}"
            );
        }
        anyhow::ensure!(
            value.get("echo").and_then(serde_json::Value::as_bool) != Some(true),
            "this engine does not support prompt echo scoring"
        );
        anyhow::ensure!(
            value
                .pointer("/vllm_xargs/kv_transfer_params")
                .is_none_or(serde_json::Value::is_null),
            "this engine does not support KV transfer"
        );
        parts.headers.insert(
            axum::http::header::CONTENT_LENGTH,
            axum::http::HeaderValue::from_str(&body.len().to_string())
                .context("invalid buffered request length")?,
        );
        router
            .oneshot(Request::from_parts(parts, Body::from(body)))
            .await
            .context("dispatch request")
    }
    .await;
    match result {
        Ok(response) => response,
        Err(error) => (StatusCode::BAD_REQUEST, Json(json!({"error": {
            "message": format!("{error:#}"), "type": "invalid_request_error", "code": "invalid_request_error"
        }}))).into_response(),
    }
}

pub(crate) fn prefill_only_routes(vllm_router: Router) -> Router {
    Router::new()
        .route("/v1/completions", post(validate_prefill_only))
        .route("/v1/chat/completions", post(validate_prefill_only))
        .with_state(vllm_router.clone())
        .fallback_service(vllm_router)
}

async fn validate_prefill_only(
    axum::extract::State(vllm_router): axum::extract::State<Router>,
    request: Request,
) -> Response {
    match validate_prefill_only_inner(vllm_router, request).await {
        Ok(response) => response,
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": {
                    "message": format!("{error:#}"),
                    "type": "invalid_request_error",
                    "code": "invalid_request_error"
                }
            })),
        )
            .into_response(),
    }
}

async fn validate_prefill_only_inner(
    vllm_router: Router,
    request: Request,
) -> anyhow::Result<Response> {
    let (mut parts, body) = request.into_parts();
    let body = to_bytes(body, REQUEST_BODY_LIMIT)
        .await
        .context("failed to read OpenAI request body")?;
    let value: serde_json::Value =
        serde_json::from_slice(&body).context("failed to parse OpenAI request JSON")?;
    let max_tokens = value.get("max_tokens").and_then(serde_json::Value::as_u64);
    let max_completion_tokens = value
        .get("max_completion_tokens")
        .and_then(serde_json::Value::as_u64);
    let is_chat = parts.uri.path() == "/v1/chat/completions";
    if is_chat
        && let (Some(max_tokens), Some(max_completion_tokens)) = (max_tokens, max_completion_tokens)
    {
        anyhow::ensure!(
            max_tokens == max_completion_tokens,
            "max_tokens ({max_tokens}) conflicts with max_completion_tokens \
             ({max_completion_tokens})"
        );
    }
    let effective_max_tokens = if is_chat {
        max_completion_tokens.or(max_tokens)
    } else {
        max_tokens
    };
    anyhow::ensure!(
        effective_max_tokens == Some(1),
        "prefill only routes require max_tokens=1, got {}",
        effective_max_tokens.map_or_else(|| "omitted".to_owned(), |value| value.to_string())
    );
    parts.headers.insert(
        axum::http::header::CONTENT_LENGTH,
        axum::http::HeaderValue::from_str(&body.len().to_string())
            .context("request body length is invalid")?,
    );
    vllm_router
        .oneshot(Request::from_parts(parts, Body::from(body)))
        .await
        .context("vLLM router failed to handle prefill-only request")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(path: &str, body: &'static str) -> Request {
        Request::post(path)
            .header("content-type", "application/json")
            .body(Body::from(body))
            .expect("request")
    }

    #[tokio::test]
    async fn greedy_text_accepts_both_greedy_paths_and_preserves_body() {
        async fn echo(request: Request) -> Response {
            let (parts, body) = request.into_parts();
            let bytes = to_bytes(body, REQUEST_BODY_LIMIT).await.unwrap();
            assert_eq!(parts.headers["content-length"], bytes.len().to_string());
            (StatusCode::OK, bytes).into_response()
        }
        let downstream = Router::new()
            .route("/v1/completions", post(echo))
            .route("/v1/chat/completions", post(echo));
        for path in ["/v1/completions", "/v1/chat/completions"] {
            for body in [
                r#"{"temperature":0,"prompt":"test"}"#,
                r#"{"temperature":0.000009999,"max_tokens":2}"#,
                r#"{"temperature":1,"top_k":1}"#,
                r#"{"top_k":1}"#,
                r#"{"temperature":0,"logprobs":null,"echo":false}"#,
                r#"{"temperature":0,"logprobs":false,"prompt_logprobs":null,"top_logprobs":null,"vllm_xargs":{"kv_transfer_params":null}}"#,
            ] {
                let response = greedy_text_routes(downstream.clone())
                    .oneshot(request(path, body))
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::OK, "{path} {body}");
                assert_eq!(
                    to_bytes(response.into_body(), REQUEST_BODY_LIMIT)
                        .await
                        .unwrap(),
                    body
                );
            }
        }
    }

    #[tokio::test]
    async fn greedy_text_rejects_sampling_boundary_and_scoring_with_reasons() {
        async fn unexpected_request() -> StatusCode {
            panic!("invalid request reached engine");
        }
        let downstream = Router::new()
            .route("/v1/completions", post(unexpected_request))
            .route("/v1/chat/completions", post(unexpected_request));
        for path in ["/v1/completions", "/v1/chat/completions"] {
            for (body, reason) in [
                ("{}", "greedy sampling"),
                (r#"{"temperature":0.00001}"#, "greedy sampling"),
                (r#"{"temperature":0.0000099999999999}"#, "greedy sampling"),
                (r#"{"temperature":0.5}"#, "greedy sampling"),
                (r#"{"temperature":-1}"#, "greedy sampling"),
                (r#"{"temperature":0,"logprobs":0}"#, "logprobs"),
                (r#"{"temperature":0,"logprobs":true}"#, "logprobs"),
                (
                    r#"{"temperature":0,"prompt_logprobs":0}"#,
                    "prompt_logprobs",
                ),
                (r#"{"temperature":0,"top_logprobs":1}"#, "top_logprobs"),
                (r#"{"temperature":0,"echo":true}"#, "echo"),
                (
                    r#"{"temperature":0,"vllm_xargs":{"kv_transfer_params":{}}}"#,
                    "KV transfer",
                ),
                ("invalid JSON", "parse request JSON"),
            ] {
                let response = greedy_text_routes(downstream.clone())
                    .oneshot(request(path, body))
                    .await
                    .unwrap();
                assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
                let error: serde_json::Value = serde_json::from_slice(
                    &to_bytes(response.into_body(), REQUEST_BODY_LIMIT)
                        .await
                        .unwrap(),
                )
                .unwrap();
                assert_eq!(error["error"]["type"], "invalid_request_error");
                assert!(
                    error["error"]["message"].as_str().unwrap().contains(reason),
                    "{error}"
                );
            }
        }
    }

    #[tokio::test]
    async fn greedy_text_preserves_fallback_routes() {
        let downstream = Router::new().route(
            "/health",
            axum::routing::get(|| async { StatusCode::NO_CONTENT }),
        );
        let response = greedy_text_routes(downstream)
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);
    }

    #[tokio::test]
    async fn prefill_only_contract_accepts_chat_token_aliases() {
        let downstream =
            Router::new().route("/v1/chat/completions", post(|| async { StatusCode::OK }));
        for body in [
            r#"{"max_tokens":1}"#,
            r#"{"max_completion_tokens":1}"#,
            r#"{"max_tokens":1,"max_completion_tokens":1}"#,
        ] {
            let response = prefill_only_routes(downstream.clone())
                .oneshot(request("/v1/chat/completions", body))
                .await
                .expect("response");
            assert_eq!(response.status(), StatusCode::OK, "{body}");
        }
    }

    #[tokio::test]
    async fn prefill_only_contract_rejects_invalid_or_conflicting_limits() {
        let downstream = Router::new()
            .route("/v1/chat/completions", post(|| async { StatusCode::OK }))
            .route("/v1/completions", post(|| async { StatusCode::OK }));
        for (path, body) in [
            ("/v1/chat/completions", r#"{"max_tokens":2}"#),
            (
                "/v1/chat/completions",
                r#"{"max_tokens":1,"max_completion_tokens":2}"#,
            ),
            ("/v1/completions", r#"{"max_completion_tokens":1}"#),
        ] {
            let response = prefill_only_routes(downstream.clone())
                .oneshot(request(path, body))
                .await
                .expect("response");
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{body}");
        }
    }
}
