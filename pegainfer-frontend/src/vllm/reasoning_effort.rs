//! `reasoning_effort` normalization for the OpenAI chat route.
//!
//! The OpenAI field accepts `none|minimal|low|medium|high|xhigh|max`, but the
//! served checkpoint's chat template may know a narrower vocabulary: Qwen3.8's
//! accepts only `xhigh|medium|low` and *raises* for anything else, and the
//! pinned frontend turns that raise into a 500 rather than a 400
//! (`docs/models/qwen35/support-qwen38.md`).
//!
//! The active rewrite set is derived from the served template itself, once at
//! startup: a probe renders a minimal request per candidate value through the
//! same vendored renderer the server will use, and a `from -> to` alias is
//! enabled only when the template rejects `from` and accepts `to`
//! (`high`/`max` → `xhigh`, `minimal` → `low`). A checkpoint whose template
//! accepts the OpenAI values — every non-Qwen3.8 line today, Kimi K3 among
//! them — gets no rewrite at all, so the caller's values reach it verbatim.
//! The probe runs only for saves carrying the `output_gate_type` config
//! marker, so other lines pay no startup cost.
//!
//! `none`, `low`, `medium` and `xhigh` pass through untouched, as does a
//! request that carries no top-level field. Two caller-supplied template
//! controls are deliberately **not** rewritten: an explicit
//! `chat_template_kwargs.reasoning_effort`, which is the caller addressing the
//! template directly, and a per-request `chat_template` override, whose
//! vocabulary the startup probe never measured. Both keep full control, at the
//! cost of their own rejection if the value is outside the template they
//! address.

use std::path::Path;
use std::sync::Arc;

use axum::Error as AxumError;
use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::body::Bytes;
use axum::body::to_bytes;
use axum::extract::Request;
use axum::extract::State;
use axum::http::HeaderValue;
use axum::http::Method;
use axum::http::StatusCode;
use axum::http::header::CONTENT_LENGTH;
use axum::middleware;
use axum::middleware::Next;
use axum::response::IntoResponse;
use axum::response::Response;
use http_body_util::LengthLimitError;
use log::debug;
use log::info;
use log::warn;
use vllm_chat::ChatMessage;
use vllm_chat::ChatOptions;
use vllm_chat::ChatRequest;
use vllm_chat::ChatRole;
use vllm_chat::EffortValue;
use vllm_chat::LoadModelBackendsOptions;
use vllm_chat::ResolvedToolContext;
use vllm_chat::SamplingParams;
use vllm_chat::load_model_backends;
use vllm_text::TextDecodeOptions;

/// Only the chat route reads `reasoning_effort`.
const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

/// Matches [`crate::vllm::lora`]'s route limit: the body is buffered to be
/// rewritten, and an oversized request must not be silently truncated.
const BODY_LIMIT: usize = 128 * 1024 * 1024;

/// Semantic candidate rewrites between the OpenAI vocabulary and a template's
/// own words. Which of them are active is decided per deployment by
/// [`probe_effort_aliases`], never assumed from the config alone.
const CANDIDATE_ALIASES: &[(&str, &str)] =
    &[("high", "xhigh"), ("max", "xhigh"), ("minimal", "low")];

/// Values rendered against the loaded template at startup: every alias source
/// plus every alias target, so a candidate is armed only on its own verdict.
fn probe_values() -> Vec<&'static str> {
    let mut values: Vec<&'static str> = Vec::with_capacity(CANDIDATE_ALIASES.len() * 2);
    for (from, to) in CANDIDATE_ALIASES {
        for value in [from, to] {
            if !values.contains(value) {
                values.push(value);
            }
        }
    }
    values
}

/// The deployment's active `from -> to` rewrites; empty means "never touch
/// the request body".
pub(crate) type EffortAliases = Arc<[(&'static str, &'static str)]>;

/// Derive the active alias table from the served checkpoint. Only a
/// Qwen3.8-class save records the `output_gate_type` marker in its
/// `config.json`; for anything else — other lines, or a path with no local
/// config at all (e.g. a HuggingFace id) — the probe is skipped entirely and
/// the mapping stays off.
pub(crate) async fn probe_effort_aliases(model_path: &str) -> EffortAliases {
    if !probe_warranted(model_path) {
        return Vec::new().into();
    }
    let aliases: Vec<_> = match probe_template(model_path).await {
        Ok(aliases) => aliases,
        Err(error) => {
            // Fail safe: without a readable template verdict there is no
            // justification to rewrite anything. The server loads its own
            // backends afterwards and reports its own error if the model is
            // genuinely broken.
            warn!(
                "reasoning_effort mapping disabled: template probe failed for {model_path}: {error}"
            );
            Vec::new()
        }
    };
    if !aliases.is_empty() {
        info!("reasoning_effort mapping for {model_path}: {aliases:?}");
    }
    aliases.into()
}

/// The config-marker prefilter: does this checkpoint's save belong to the
/// generation whose template is known to restrict `reasoning_effort`? A
/// missing directory means the model id is not a local path; a directory
/// whose config cannot be read or parsed is an anomaly worth surfacing.
fn probe_warranted(model_path: &str) -> bool {
    let path = Path::new(model_path);
    let config_path = path.join("config.json");
    let content = match std::fs::read_to_string(&config_path) {
        Ok(content) => content,
        Err(error) if path.is_dir() => {
            warn!(
                "reasoning_effort probe skipped: local model directory {} has no readable config.json: {error}",
                path.display()
            );
            return false;
        }
        Err(_) => return false,
    };
    let config: serde_json::Value = match serde_json::from_str(&content) {
        Ok(config) => config,
        Err(error) => {
            warn!(
                "reasoning_effort probe skipped: {} is not valid JSON: {error}",
                config_path.display()
            );
            return false;
        }
    };
    let text = config.get("text_config").unwrap_or(&config);
    text.get("output_gate_type").is_some()
}

/// Render one minimal request per candidate value through the same vendored
/// stack the server uses, and keep the aliases whose source the template
/// rejects while its target is accepted.
async fn probe_template(model_path: &str) -> vllm_chat::Result<Vec<(&'static str, &'static str)>> {
    let backends = load_model_backends(
        model_path,
        LoadModelBackendsOptions {
            language_model_only: true,
            ..Default::default()
        },
    )
    .await?;
    let renderer = backends.chat_backend.chat_renderer();
    let values = probe_values();
    let mut accepted: Vec<(&str, bool)> = Vec::with_capacity(values.len());
    for value in values {
        accepted.push((value, renderer.render(&probe_request(value)).is_ok()));
    }
    let accepts = |value: &str| accepted.iter().any(|&(probed, ok)| probed == value && ok);
    Ok(CANDIDATE_ALIASES
        .iter()
        .filter(|(from, to)| !accepts(from) && accepts(to))
        .copied()
        .collect())
}

/// The smallest request that carries one `reasoning_effort` value: a single
/// user turn, every other field at the value the vendored routes resolve for a
/// plain text chat. The probe renders it and throws the prompt away.
fn probe_request(effort: &str) -> ChatRequest {
    ChatRequest {
        request_id: "reasoning-effort-probe".to_string(),
        messages: vec![ChatMessage::text(ChatRole::User, "probe")],
        sampling_params: SamplingParams::default(),
        chat_options: ChatOptions {
            reasoning_effort: Some(EffortValue::String(effort.to_string())),
            ..ChatOptions::default()
        },
        tool_context: ResolvedToolContext::default(),
        decode_options: TextDecodeOptions::default(),
        intermediate: false,
        prompt_truncation: None,
        priority: 0,
        documents: None,
        cache_salt: None,
        add_special_tokens: false,
        data_parallel_rank: None,
        session_id: None,
        lora_request: None,
    }
}

/// Wrap a router so chat requests are normalized before the upstream handler
/// converts them. Outermost on purpose: every serving path (plain,
/// prefill-only, LoRA) funnels through the router this wraps.
pub(crate) fn normalize_chat_requests(router: Router, aliases: EffortAliases) -> Router {
    if aliases.is_empty() {
        return router;
    }
    router.layer(middleware::from_fn_with_state(aliases, normalize_request))
}

async fn normalize_request(
    State(aliases): State<EffortAliases>,
    request: Request,
    next: Next,
) -> Response {
    if request.method() != Method::POST || request.uri().path() != CHAT_COMPLETIONS_PATH {
        return next.run(request).await;
    }

    let (mut parts, body) = request.into_parts();
    let mut bytes = match to_bytes(body, BODY_LIMIT).await {
        Ok(bytes) => bytes,
        Err(error) => return body_read_error(error),
    };

    if let Some((from, to)) = rewrite_reasoning_effort(&mut bytes, &aliases) {
        debug!("reasoning_effort {from} -> {to}");
        parts
            .headers
            .insert(CONTENT_LENGTH, HeaderValue::from(bytes.len() as u64));
    }

    next.run(Request::from_parts(parts, Body::from(bytes)))
        .await
}

/// The route's error contract: the same OpenAI-style JSON envelope the
/// upstream routes answer with (`vllm_server`'s `ApiError::to_error_response`,
/// which is not exported), keeping both the status category and the cause.
/// 413 belongs to the length-limit rejection alone; any other read failure
/// keeps the 400 this route reported before the layer existed.
fn body_read_error(error: AxumError) -> Response {
    let message = format!("failed to read the chat request body: {error}");
    let status = if error.into_inner().is::<LengthLimitError>() {
        StatusCode::PAYLOAD_TOO_LARGE
    } else {
        StatusCode::BAD_REQUEST
    };
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": null,
                "code": "invalid_request_error",
            }
        })),
    )
        .into_response()
}

/// Rewrite the top-level `reasoning_effort` value in place, returning the
/// `(from, to)` pair when it changed. A body that is not JSON, or whose field
/// is absent, not a string, or outside the deployment's active aliases, is
/// left exactly as it arrived — as is any request carrying its own
/// `chat_template`, which renders with a template the startup probe never saw.
pub(crate) fn rewrite_reasoning_effort(
    body: &mut Bytes,
    aliases: &[(&'static str, &'static str)],
) -> Option<(String, &'static str)> {
    let mut value: serde_json::Value = serde_json::from_slice(body).ok()?;
    // The aliases are the *default* template's verdict. A per-request override
    // is a different template with its own vocabulary, so rewriting on the
    // default's authority can turn a value the override accepts into one it
    // rejects. Leave the caller's value for the override to judge.
    if value
        .get("chat_template")
        .is_some_and(serde_json::Value::is_string)
    {
        return None;
    }
    let effort = value.get("reasoning_effort")?.as_str()?.to_string();
    let to = aliases
        .iter()
        .find(|(from, _)| *from == effort)
        .map(|(_, to)| *to)?;
    value["reasoning_effort"] = serde_json::Value::String(to.to_string());
    // Serializing a `Value` into a `Vec` cannot fail: every key is already a
    // string, and a non-finite number degrades to `null`.
    let rewritten = serde_json::to_vec(&value).expect("a serde_json::Value serializes");
    *body = Bytes::from(rewritten);
    Some((effort, to))
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_ALIASES: &[(&str, &str)] = CANDIDATE_ALIASES;

    fn rewrite(json: &str) -> (Option<(String, &'static str)>, serde_json::Value) {
        let mut body = Bytes::from(json.to_string());
        let changed = rewrite_reasoning_effort(&mut body, ALL_ALIASES);
        let value = serde_json::from_slice(&body).expect("body stays valid JSON");
        (changed, value)
    }

    #[test]
    fn bodies_without_a_rewritable_top_level_field_are_returned_verbatim() {
        let (changed, _) = rewrite(r#"{"messages":[{"role":"user","content":"hi"}]}"#);
        assert_eq!(changed, None, "no field");

        let (changed, value) = rewrite(r#"{"reasoning_effort":7}"#);
        assert_eq!(changed, None, "non-string field");
        assert_eq!(value["reasoning_effort"], serde_json::json!(7));

        let mut body = Bytes::from_static(b"not json at all");
        assert_eq!(rewrite_reasoning_effort(&mut body, ALL_ALIASES), None);
        assert_eq!(&body[..], b"not json at all");

        let (changed, value) = rewrite(r#"{"chat_template_kwargs":{"reasoning_effort":"high"}}"#);
        assert_eq!(changed, None, "kwargs-only effort stays the caller's");
        assert_eq!(
            value["chat_template_kwargs"]["reasoning_effort"],
            serde_json::json!("high")
        );
    }

    #[test]
    fn a_per_request_template_override_keeps_the_callers_effort_value() {
        // The alias table is the *default* template's verdict. An override can
        // accept what the default rejects, so guessing for it can break a
        // request that would otherwise render.
        let (changed, value) = rewrite(r#"{"reasoning_effort":"high","chat_template":"{{ '' }}"}"#);
        assert_eq!(
            changed, None,
            "the override's vocabulary is not ours to know"
        );
        assert_eq!(value["reasoning_effort"], serde_json::json!("high"));

        // No override — an absent or null field renders the default template the
        // probe measured, so the mapping applies.
        let (changed, _) = rewrite(r#"{"reasoning_effort":"high","chat_template":null}"#);
        assert_eq!(changed, Some(("high".to_string(), "xhigh")));
    }

    // ---- startup probe -------------------------------------------------

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
      "pre_tokenizer": { "type": "Whitespace" },
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": { "<unk>": 0, "probe": 1 },
        "unk_token": "<unk>"
      }
    }"#;

    /// Qwen3.8's stock guard shape: the vocabulary check sits inside the
    /// thinking branch and admits only `xhigh|medium|low`.
    const GUARDED_TEMPLATE: &str = r"{%- if enable_thinking is undefined or enable_thinking is true %}{%- set resolved = reasoning_effort|default('xhigh') %}{%- if resolved not in ('xhigh', 'medium', 'low') %}{{ raise_exception('Unexpected reasoning effort ' ~ reasoning_effort) }}{%- endif %}{%- endif %}{% for message in messages %}{{ message.content }}{% endfor %}";

    /// The review counterexample: a Qwen3.8-marked config whose custom
    /// template accepts the OpenAI extremes (`low|high|max`, K3-style). The
    /// probe must not arm `high`/`max` rewrites for it.
    const ACCEPTING_TEMPLATE: &str = r"{%- set resolved = reasoning_effort|default('high') %}{%- if resolved not in ('low', 'high', 'max') %}{{ raise_exception('Unexpected reasoning effort ' ~ reasoning_effort) }}{%- endif %}{% for message in messages %}{{ message.content }}{% endfor %}";

    fn probe_dir(template: &str, marker: bool) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("tokenizer.json"), TINY_TOKENIZER_JSON)
            .expect("write tokenizer.json");
        let tokenizer_config = serde_json::json!({
            "unk_token": "<unk>",
            "tokenizer_class": "PreTrainedTokenizerFast",
            "chat_template": template,
        });
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            serde_json::to_string(&tokenizer_config).expect("tokenizer_config serializes"),
        )
        .expect("write tokenizer_config.json");
        let mut config = serde_json::json!({
            "model_type": "pegainfer_sim",
            "max_position_embeddings": 128,
            "vocab_size": 16,
        });
        if marker {
            config["text_config"] = serde_json::json!({ "output_gate_type": "swish" });
        }
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&config).expect("config serializes"),
        )
        .expect("write config.json");
        dir
    }

    fn path_of(dir: &tempfile::TempDir) -> &str {
        dir.path().to_str().expect("utf-8 temp path")
    }

    #[tokio::test]
    async fn the_probe_arms_every_alias_for_the_stock_guarded_template() {
        let dir = probe_dir(GUARDED_TEMPLATE, true);
        let aliases = probe_template(path_of(&dir))
            .await
            .expect("the probe renders the tiny fixture");
        assert_eq!(
            aliases.as_slice(),
            [("high", "xhigh"), ("max", "xhigh"), ("minimal", "low")]
        );
    }

    #[tokio::test]
    async fn the_probe_never_arms_a_rewrite_the_template_does_not_need() {
        // Same marked config, custom template that accepts high/max: only
        // minimal (rejected, with low accepted) may be rewritten.
        let dir = probe_dir(ACCEPTING_TEMPLATE, true);
        let aliases = probe_template(path_of(&dir))
            .await
            .expect("the probe renders the tiny fixture");
        assert_eq!(aliases.as_slice(), [("minimal", "low")]);
    }

    #[tokio::test]
    async fn the_probe_is_skipped_without_the_marker_or_a_local_config() {
        let unmarked = probe_dir(GUARDED_TEMPLATE, false);
        assert!(
            probe_effort_aliases(path_of(&unmarked)).await.is_empty(),
            "an unmarked save keeps the caller's values verbatim"
        );

        let missing = "/nonexistent/hf-model-id";
        assert!(
            probe_effort_aliases(missing).await.is_empty(),
            "a path with no local config (an HF id) is not an error"
        );
    }

    // ---- middleware over the real router -------------------------------

    fn guarded_app(aliases: EffortAliases) -> Router {
        normalize_chat_requests(
            Router::new().route(
                CHAT_COMPLETIONS_PATH,
                axum::routing::post(|| async { "ok" }),
            ),
            aliases,
        )
    }

    /// Drive the real middleware and return what the caller would see.
    async fn chat_post(body: Body) -> (StatusCode, serde_json::Value) {
        use tower::ServiceExt as _;

        let request = Request::builder()
            .method(Method::POST)
            .uri(CHAT_COMPLETIONS_PATH)
            .body(body)
            .expect("request builds");
        let response = guarded_app(ALL_ALIASES.to_vec().into())
            .oneshot(request)
            .await
            .expect("the middleware always answers");
        let status = response.status();
        let bytes = to_bytes(response.into_body(), 64 * 1024)
            .await
            .expect("response body reads");
        let body: serde_json::Value =
            serde_json::from_slice(&bytes).expect("the error body is the OpenAI JSON envelope");
        (status, body)
    }

    fn assert_error_envelope(body: &serde_json::Value, cause: &str) {
        assert_eq!(
            body["error"]["type"],
            serde_json::json!("invalid_request_error"),
            "{body}"
        );
        assert!(
            body["error"]["message"]
                .as_str()
                .is_some_and(|m| m.contains(cause)),
            "the cause must survive in the envelope: {body}"
        );
    }

    #[tokio::test]
    async fn an_oversized_chat_body_is_rejected_as_payload_too_large() {
        use futures::StreamExt as _;

        let chunk = Bytes::from(vec![0u8; 1024 * 1024]);
        let chunks = BODY_LIMIT / chunk.len() + 2;
        let body = Body::from_stream(
            futures::stream::repeat_with(move || Ok::<Bytes, std::io::Error>(chunk.clone()))
                .take(chunks),
        );
        let (status, body) = chat_post(body).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
        assert_error_envelope(&body, "length limit exceeded");
    }

    #[tokio::test]
    async fn a_truncated_chat_body_keeps_the_bad_request_category_and_the_cause() {
        let frames = vec![
            Ok::<Bytes, std::io::Error>(Bytes::from_static(br#"{"reasoning_effort":"high","#)),
            Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "connection closed before the body completed",
            )),
        ];
        let (status, body) = chat_post(Body::from_stream(futures::stream::iter(frames))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_error_envelope(&body, "connection closed before the body completed");
    }
}
