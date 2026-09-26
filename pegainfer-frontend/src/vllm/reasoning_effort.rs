//! `reasoning_effort` normalization for the OpenAI chat route.
//!
//! The OpenAI field accepts `none|minimal|low|medium|high|xhigh|max`, but a
//! chat template may know a narrower vocabulary: Qwen3.8's accepts only
//! `xhigh|medium|low` and *raises* for anything else, and the pinned frontend
//! turns that raise into a 500 rather than a 400 (`docs/models/qwen35/support-qwen38.md`).
//! Map the OpenAI-only extreme values onto the template's own words before the
//! upstream renderer sees the request:
//!
//! - `high` and `max` mean "most reasoning" → `xhigh`
//! - `minimal` means "least reasoning above none" → `low`
//!
//! `none`, `low`, `medium` and `xhigh` pass through untouched, as does a request
//! that carries no top-level field. An explicit
//! `chat_template_kwargs.reasoning_effort` is deliberately **not** rewritten:
//! that is the caller addressing the template directly, and it keeps full
//! control (at the cost of the template's own rejection if the value is outside
//! its vocabulary).

use anyhow::Result;
use axum::Router;
use axum::body::Body;
use axum::body::Bytes;
use axum::body::to_bytes;
use axum::extract::Request;
use axum::http::HeaderValue;
use axum::http::Method;
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
/// converts them. Outermost on purpose: every serving path (plain, prefill-only,
/// LoRA) funnels through the router this wraps.
pub(crate) fn normalize_chat_requests(router: Router) -> Router {
    router.layer(axum::middleware::from_fn(normalize_request))
}

async fn normalize_request(request: Request, next: Next) -> Response {
    if request.method() != Method::POST || request.uri().path() != CHAT_COMPLETIONS_PATH {
        return next.run(request).await;
    }

    let (mut parts, body) = request.into_parts();
    let Ok(mut bytes) = to_bytes(body, BODY_LIMIT).await else {
        // An unreadable body is the upstream route's error to report.
        return next.run(Request::from_parts(parts, Body::empty())).await;
    };

    if let Some((from, to)) = rewrite_reasoning_effort(&mut bytes) {
        info!("reasoning_effort {from} -> {to}");
        if let Ok(length) = HeaderValue::from_str(&bytes.len().to_string()) {
            parts.headers.insert(CONTENT_LENGTH, length);
        }
    }

    next.run(Request::from_parts(parts, Body::from(bytes)))
        .await
}

/// Rewrite the top-level `reasoning_effort` value in place, returning the
/// `(from, to)` pair when it changed. A body that is not JSON, or whose field is
/// absent or not a string, is left exactly as it arrived.
pub(crate) fn rewrite_reasoning_effort(body: &mut Bytes) -> Option<(String, &'static str)> {
    // The field is absent from almost every request, and a body can be large, so
    // skip the parse unless the key's bytes are actually present. A mention that
    // only appears inside message content costs a parse and still changes
    // nothing, because only the top-level field is ever read.
    const KEY: &[u8] = b"\"reasoning_effort\"";
    if body.len() < KEY.len() || !body.windows(KEY.len()).any(|window| window == KEY) {
        return None;
    }

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
    fn ignores_absent_non_string_and_non_json_fields() {
        let (changed, _) = rewrite(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(changed, None, "no field");

        let (changed, value) = rewrite(r#"{"reasoning_effort":7}"#);
        assert_eq!(changed, None, "non-string field");
        assert_eq!(value["reasoning_effort"], serde_json::json!(7));

        let mut body = Bytes::from_static(b"not json at all");
        assert_eq!(rewrite_reasoning_effort(&mut body), None);
        assert_eq!(&body[..], b"not json at all");

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

    #[test]
    fn template_kwargs_only_effort_is_not_rewritten() {
        let (changed, value) = rewrite(r#"{"chat_template_kwargs":{"reasoning_effort":"high"}}"#);
        assert_eq!(changed, None);
        assert_eq!(
            value["chat_template_kwargs"]["reasoning_effort"],
            serde_json::json!("high")
        );
    }

    #[test]
    fn a_mention_inside_message_content_is_not_the_field() {
        let source = r#"{"messages":[{"role":"user","content":"set reasoning_effort to high"}]}"#;
        let mut body = Bytes::from_static(source.as_bytes());
        assert_eq!(rewrite_reasoning_effort(&mut body), None);
        assert_eq!(&body[..], source.as_bytes());
    }
}
