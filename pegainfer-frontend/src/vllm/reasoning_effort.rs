//! `reasoning_effort` normalization for the OpenAI chat route.
//!
//! The OpenAI field accepts `none|minimal|low|medium|high|xhigh|max`, but a
//! chat template may know a narrower vocabulary: Qwen3.8's accepts only
//! `xhigh|medium|low` and *raises* for anything else, and the pinned frontend
//! turns that raise into a 500 rather than a 400
//! (`docs/models/qwen35/support-qwen38.md`). For those checkpoints — and only
//! those — map the OpenAI-only extreme values onto the template's own words
//! before the upstream renderer sees the request:
//!
//! - `high` and `max` mean "most reasoning" → `xhigh`
//! - `minimal` means "least reasoning above none" → `low`
//!
//! The mapping is gated on the model because other lines must keep receiving
//! the caller's value verbatim: their templates accept what OpenAI accepts,
//! and rewriting it would break them (Kimi K3's renderer, for one, supports
//! `high` and `max` natively and rejects the `xhigh` rewrite). The gate is
//! the save-time `output_gate_type` config marker — the field a Qwen3.8 save
//! carries and a Qwen3.5 save omits, the same one the golden dumper keys on.
//!
//! `none`, `low`, `medium` and `xhigh` pass through untouched, as does a
//! request that carries no top-level field. An explicit
//! `chat_template_kwargs.reasoning_effort` is deliberately **not** rewritten:
//! that is the caller addressing the template directly, and it keeps full
//! control (at the cost of the template's own rejection if the value is
//! outside its vocabulary).

use std::path::Path;

use anyhow::Result;
use axum::Router;
use axum::body::Body;
use axum::body::Bytes;
use axum::body::to_bytes;
use axum::extract::Request;
use axum::http::HeaderValue;
use axum::http::Method;
use axum::http::StatusCode;
use axum::http::header::CONTENT_LENGTH;
use axum::middleware::Next;
use axum::response::Response;
use log::info;

/// Only the chat route reads `reasoning_effort`.
const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

/// Matches [`crate::vllm::lora`]'s route limit: the body is buffered to be
/// rewritten, and an oversized request must not be silently truncated.
const BODY_LIMIT: usize = 128 * 1024 * 1024;

const EFFORT_ALIASES: &[(&str, &str)] = &[("high", "xhigh"), ("max", "xhigh"), ("minimal", "low")];

/// Wrap a router so chat requests are normalized before the upstream handler
/// converts them, when the served checkpoint's template needs the mapping.
/// Outermost on purpose: every serving path (plain, prefill-only, LoRA)
/// funnels through the router this wraps.
pub(crate) fn normalize_chat_requests(router: Router, enabled: bool) -> Router {
    if enabled {
        router.layer(axum::middleware::from_fn(normalize_request))
    } else {
        router
    }
}

/// Whether the served checkpoint's template rejects the OpenAI-only effort
/// values and therefore needs the mapping. Only a Qwen3.8-class save records
/// the `output_gate_type` marker in its `config.json`; anything else — other
/// lines, or a path with no readable local config (e.g. a HuggingFace id) —
/// keeps the mapping off.
pub(crate) fn mapping_enabled(model_path: &str) -> bool {
    let Ok(content) = std::fs::read_to_string(Path::new(model_path).join("config.json")) else {
        return false;
    };
    let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) else {
        return false;
    };
    let text = config.get("text_config").unwrap_or(&config);
    let enabled = text.get("output_gate_type").is_some();
    if enabled {
        info!("reasoning_effort mapping enabled for {model_path}");
    }
    enabled
}

async fn normalize_request(request: Request, next: Next) -> Result<Response, (StatusCode, String)> {
    if request.method() != Method::POST || request.uri().path() != CHAT_COMPLETIONS_PATH {
        return Ok(next.run(request).await);
    }

    let (mut parts, body) = request.into_parts();
    let mut bytes = to_bytes(body, BODY_LIMIT).await.map_err(|error| {
        (
            StatusCode::PAYLOAD_TOO_LARGE,
            format!("failed to read the chat request body: {error}"),
        )
    })?;

    if let Some((from, to)) = rewrite_reasoning_effort(&mut bytes) {
        info!("reasoning_effort {from} -> {to}");
        if let Ok(length) = HeaderValue::from_str(&bytes.len().to_string()) {
            parts.headers.insert(CONTENT_LENGTH, length);
        }
    }

    Ok(next
        .run(Request::from_parts(parts, Body::from(bytes)))
        .await)
}

/// Rewrite the top-level `reasoning_effort` value in place, returning the
/// `(from, to)` pair when it changed. A body that is not JSON, or whose field
/// is absent or not a string, is left exactly as it arrived.
pub(crate) fn rewrite_reasoning_effort(body: &mut Bytes) -> Option<(String, &'static str)> {
    let mut value: serde_json::Value = serde_json::from_slice(body).ok()?;
    let effort = value.get("reasoning_effort")?.as_str()?.to_string();
    let to = EFFORT_ALIASES
        .iter()
        .find(|(from, _)| *from == effort)
        .map(|(_, to)| *to)?;
    value["reasoning_effort"] = serde_json::Value::String(to.to_string());
    let rewritten: Result<Vec<u8>, _> = serde_json::to_vec(&value);
    *body = Bytes::from(rewritten.ok()?);
    Some((effort, to))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rewrite(json: &str) -> (Option<(String, &'static str)>, serde_json::Value) {
        let mut body = Bytes::from(json.to_string());
        let changed = rewrite_reasoning_effort(&mut body);
        let value = serde_json::from_slice(&body).expect("body stays valid JSON");
        (changed, value)
    }

    #[test]
    fn maps_the_openai_only_extremes_onto_the_templates_vocabulary() {
        for (from, to) in [("high", "xhigh"), ("max", "xhigh"), ("minimal", "low")] {
            let (changed, value) =
                rewrite(&format!(r#"{{"reasoning_effort":"{from}","messages":[]}}"#));
            assert_eq!(changed, Some((from.to_string(), to)), "{from}");
            assert_eq!(value["reasoning_effort"], serde_json::json!(to), "{from}");
        }
    }

    #[test]
    fn leaves_supported_and_unrelated_values_untouched() {
        for effort in ["none", "low", "medium", "xhigh"] {
            let (changed, value) = rewrite(&format!(r#"{{"reasoning_effort":"{effort}"}}"#));
            assert_eq!(changed, None, "{effort}");
            assert_eq!(
                value["reasoning_effort"],
                serde_json::json!(effort),
                "{effort}"
            );
        }
    }

    #[test]
    fn bodies_without_a_rewritable_top_level_field_are_returned_verbatim() {
        let (changed, _) = rewrite(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(changed, None, "no field");

        let (changed, value) = rewrite(r#"{"reasoning_effort":7}"#);
        assert_eq!(changed, None, "non-string field");
        assert_eq!(value["reasoning_effort"], serde_json::json!(7));

        let mut body = Bytes::from_static(b"not json at all");
        assert_eq!(rewrite_reasoning_effort(&mut body), None);
        assert_eq!(&body[..], b"not json at all");

        let (changed, value) = rewrite(r#"{"chat_template_kwargs":{"reasoning_effort":"high"}}"#);
        assert_eq!(changed, None, "kwargs-only effort stays the caller's");
        assert_eq!(
            value["chat_template_kwargs"]["reasoning_effort"],
            serde_json::json!("high")
        );

        let source = r#"{"messages":[{"role":"user","content":"set reasoning_effort to high"}]}"#;
        let mut body = Bytes::from_static(source.as_bytes());
        assert_eq!(rewrite_reasoning_effort(&mut body), None);
        assert_eq!(&body[..], source.as_bytes());
    }

    #[test]
    fn mapping_gates_on_the_save_time_output_gate_marker() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().to_str().expect("utf-8 temp path");

        assert!(!mapping_enabled(path), "no readable config");

        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type":"qwen3_5"}"#,
        )
        .expect("write config");
        assert!(!mapping_enabled(path), "qwen3_5 save without the marker");

        std::fs::write(
            dir.path().join("config.json"),
            r#"{"text_config":{"output_gate_type":"swish"}}"#,
        )
        .expect("write config");
        assert!(mapping_enabled(path), "the marker turns the mapping on");
    }

    #[test]
    fn an_effort_in_both_places_rewrites_only_the_top_level_field() {
        let (changed, value) = rewrite(
            r#"{"reasoning_effort":"high","chat_template_kwargs":{"reasoning_effort":"high"}}"#,
        );
        assert_eq!(changed, Some(("high".to_string(), "xhigh")));
        assert_eq!(
            value["chat_template_kwargs"]["reasoning_effort"],
            serde_json::json!("high"),
            "template kwargs stay the caller's to set"
        );
    }
}
