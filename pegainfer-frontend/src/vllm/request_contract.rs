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
