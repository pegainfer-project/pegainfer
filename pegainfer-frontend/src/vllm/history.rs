//! The prompt's history boundary for the OpenAI chat route.
//!
//! A chat prompt is the rendered conversation followed by the template's
//! generation prompt (`<|im_start|>assistant\n<think>\n` for Qwen3.8). The next
//! turn of the same conversation repeats the rendered conversation byte for
//! byte and then renders the reply the model gave *as history*, which is not
//! how the generation prompt reads. So the longest prefix two turns share ends
//! where the last message ends, and a scheduler whose state cannot roll back
//! (a recurrent line) serves the next turn only from a snapshot taken there.
//! Nothing behind the vendored chat route sees the request text, so this layer
//! measures the boundary in front of the route and the bridge carries it to
//! the scheduler as [`crate::engine::Request::history_tokens`].
//!
//! The measure is per request and local to the last message: the message is
//! rendered alone, with and without the generation prompt, through the same
//! renderer and tokenizer the route uses, and the generation prompt's token
//! count `T` is the tail the two tokenizations do not share. A token that
//! straddles the boundary counts towards `T`, so the boundary moves one token
//! earlier rather than into the generation prompt. `T` travels as
//! `vllm_xargs.generation_prompt_tokens`; the bridge, which sees the whole
//! prompt's token ids, sets `history_tokens = prompt_len - T`. Rendering only
//! the last message is exact for a template that appends the generation prompt
//! after the messages, which is every template the frontend serves, and costs
//! one message's rendering instead of the conversation's.
//!
//! `add_generation_prompt=false` and `continue_final_message=true` render no
//! generation prompt, so `T = 0` and the whole prompt is history. A body the
//! measure cannot read (not JSON, no messages, a last message the renderer
//! rejects) passes through untouched and the scheduler sees no boundary; the
//! route reports its own error for it.

use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::body::Bytes;
use axum::body::to_bytes;
use axum::extract::Request;
use axum::extract::State;
use axum::http::HeaderValue;
use axum::http::Method;
use axum::http::header::CONTENT_LENGTH;
use axum::middleware;
use axum::middleware::Next;
use axum::response::Response;
use log::debug;
use serde::de::DeserializeOwned;
use serde_json::Value;
use serde_json::json;
use vllm_chat::ChatContent;
use vllm_chat::ChatMessage;
use vllm_chat::ChatOptions;
use vllm_chat::ChatRequest;
use vllm_chat::DynChatRenderer;
use vllm_chat::GenerationPromptMode;
use vllm_chat::LoadedModelBackends;
use vllm_chat::ResolvedToolContext;
use vllm_chat::SamplingParams;
use vllm_text::Prompt;
use vllm_text::TextDecodeOptions;
use vllm_text::tokenizer::DynTokenizer;

use super::reasoning_effort::BODY_LIMIT;
use super::reasoning_effort::CHAT_COMPLETIONS_PATH;
use super::reasoning_effort::body_read_error;

/// The `vllm_xargs` key carrying the generation prompt's token count from
/// this layer to the bridge.
pub(crate) const GENERATION_PROMPT_TOKENS: &str = "generation_prompt_tokens";

/// The route's renderer and tokenizer, held to measure one message at a time.
pub(crate) struct Boundary {
    renderer: DynChatRenderer,
    tokenizer: DynTokenizer,
}

impl Boundary {
    pub(crate) fn new(backends: &LoadedModelBackends) -> Self {
        Self {
            renderer: backends.chat_backend.chat_renderer(),
            tokenizer: backends.text_backend.tokenizer(),
        }
    }

    /// The generation prompt's token count for one chat body, or `None` when
    /// the body does not render.
    fn generation_prompt_tokens(&self, body: &Value) -> Option<usize> {
        let options = chat_options(body);
        if !options.add_generation_prompt() {
            return Some(0);
        }
        let message = last_message(body)?;
        let with = self.tokens(&message, options.clone())?;
        let without = self.tokens(
            &message,
            ChatOptions {
                generation_prompt_mode: GenerationPromptMode::NoGenerationPrompt,
                ..options
            },
        )?;
        let shared = with
            .iter()
            .zip(&without)
            .take_while(|(a, b)| a == b)
            .count();
        Some(with.len() - shared)
    }

    fn tokens(&self, message: &ChatMessage, chat_options: ChatOptions) -> Option<Vec<u32>> {
        let rendered = self
            .renderer
            .render(&probe_request(message.clone(), chat_options))
            .ok()?;
        match rendered.prompt {
            Prompt::TokenIds(ids) => Some(ids),
            Prompt::Text(text) => self.tokenizer.encode(&text, false).ok(),
        }
    }
}

/// The template controls the route itself resolves from the body, so the
/// probe renders under the same mode, template and kwargs as the prompt.
fn chat_options(body: &Value) -> ChatOptions {
    let add_generation_prompt = body.get("add_generation_prompt").and_then(Value::as_bool);
    let continue_final_message = body
        .get("continue_final_message")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let generation_prompt_mode = match (add_generation_prompt, continue_final_message) {
        (_, true) => GenerationPromptMode::ContinueFinalAssistant,
        (Some(false), false) => GenerationPromptMode::NoGenerationPrompt,
        (None | Some(true), false) => GenerationPromptMode::StartNewAssistant,
    };
    ChatOptions {
        generation_prompt_mode,
        chat_template: field(body, "chat_template"),
        reasoning_effort: field(body, "reasoning_effort"),
        response_format: None,
        template_kwargs: field(body, "chat_template_kwargs").unwrap_or_default(),
    }
}

fn field<T: DeserializeOwned>(body: &Value, key: &str) -> Option<T> {
    serde_json::from_value(body.get(key)?.clone()).ok()
}

/// The last message in the route's own vocabulary. An assistant message keeps
/// only its text: the generation prompt's shape does not depend on its tool
/// calls, and the route rejects what this layer cannot read.
fn last_message(body: &Value) -> Option<ChatMessage> {
    let message = body.get("messages")?.as_array()?.last()?;
    let role = message.get("role")?.as_str()?;
    let content = match message.get("content") {
        None | Some(Value::Null) => ChatContent::Text(String::new()),
        Some(Value::String(text)) => ChatContent::Text(text.clone()),
        Some(parts) => serde_json::from_value(parts.clone()).ok()?,
    };
    Some(match role {
        "system" => ChatMessage::system(content),
        "developer" => ChatMessage::developer(content, None),
        "user" => ChatMessage::user(content),
        "assistant" => ChatMessage::assistant_text(content.try_flatten_to_text().ok()?),
        "tool" => ChatMessage::tool_response(
            content,
            message
                .get("tool_call_id")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        ),
        custom => ChatMessage::custom(custom, content),
    })
}

/// One message under the request's template controls, every other field at
/// the value the route resolves for a plain text chat.
fn probe_request(message: ChatMessage, chat_options: ChatOptions) -> ChatRequest {
    ChatRequest {
        request_id: "history-boundary-probe".to_string(),
        messages: vec![message],
        sampling_params: SamplingParams::default(),
        chat_options,
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

/// Wrap a router so every chat request carries its generation prompt's token
/// count into `vllm_xargs`; a server whose backends did not load serves the
/// router as is.
pub(crate) fn mark_history(router: Router, boundary: Option<Arc<Boundary>>) -> Router {
    match boundary {
        Some(boundary) => router.layer(middleware::from_fn_with_state(boundary, mark_request)),
        None => router,
    }
}

async fn mark_request(
    State(boundary): State<Arc<Boundary>>,
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

    if let Some(tokens) = mark_generation_prompt(&mut bytes, &boundary) {
        debug!("generation prompt is {tokens} tokens");
        parts
            .headers
            .insert(CONTENT_LENGTH, HeaderValue::from(bytes.len() as u64));
    }

    next.run(Request::from_parts(parts, Body::from(bytes)))
        .await
}

/// Write the measured count into the body's `vllm_xargs`, returning it when
/// the body changed. A caller's own value for the key is replaced: the count
/// is this server's measure of its own template, not a request parameter.
fn mark_generation_prompt(body: &mut Bytes, boundary: &Boundary) -> Option<usize> {
    let mut value: Value = serde_json::from_slice(body).ok()?;
    let tokens = boundary.generation_prompt_tokens(&value)?;
    value
        .as_object_mut()?
        .entry("vllm_xargs")
        .or_insert_with(|| json!({}))
        .as_object_mut()?
        .insert(GENERATION_PROMPT_TOKENS.to_string(), json!(tokens));
    // Serializing a `Value` into a `Vec` cannot fail: every key is already a
    // string, and a non-finite number degrades to `null`.
    *body = Bytes::from(serde_json::to_vec(&value).expect("a serde_json::Value serializes"));
    Some(tokens)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fmt::Write;

    use vllm_chat::ChatRenderer;
    use vllm_chat::RenderedPrompt;
    use vllm_text::tokenizer::Tokenizer;

    use super::*;

    /// A template that wraps each message in its role and appends the
    /// generation prompt as separate words, except for the tail of a user
    /// message ending in `~`, which it glues to the generation prompt so one
    /// token straddles the boundary.
    struct Template;

    impl ChatRenderer for Template {
        fn render(&self, request: &ChatRequest) -> vllm_chat::Result<RenderedPrompt> {
            let mut text = String::new();
            for message in &request.messages {
                let role = serde_json::to_value(message.role()).unwrap();
                let content = message.text_content()?;
                write!(text, "<{}> {content} <end>", role.as_str().unwrap()).unwrap();
            }
            if request.chat_options.add_generation_prompt() {
                if text.ends_with("~ <end>") {
                    text.push_str("<assistant><think>");
                } else {
                    text.push_str(" <assistant> <think>");
                }
            }
            Ok(RenderedPrompt {
                prompt: Prompt::Text(text),
                media_order: None,
                effective_template_kwargs: HashMap::default(),
            })
        }
    }

    /// Whitespace words are tokens; the id is the word's length.
    struct Words;

    impl Tokenizer for Words {
        fn encode(&self, text: &str, _: bool) -> vllm_text::tokenizer::Result<Vec<u32>> {
            Ok(text.split_whitespace().map(|w| w.len() as u32).collect())
        }
        fn encode_ordinary(&self, text: &str) -> vllm_text::tokenizer::Result<Vec<u32>> {
            self.encode(text, false)
        }
        fn decode(&self, _: &[u32], _: bool) -> vllm_text::tokenizer::Result<String> {
            unimplemented!()
        }
        fn token_to_id(&self, _: &str) -> Option<u32> {
            None
        }
        fn id_to_token(&self, _: u32) -> Option<String> {
            None
        }
    }

    fn boundary() -> Boundary {
        Boundary {
            renderer: Arc::new(Template),
            tokenizer: Arc::new(Words),
        }
    }

    fn mark(json: &str) -> (Option<usize>, Value) {
        let mut body = Bytes::from(json.to_string());
        let changed = mark_generation_prompt(&mut body, &boundary());
        let value = serde_json::from_slice(&body).expect("body stays valid JSON");
        (changed, value)
    }

    #[test]
    fn the_generation_prompt_is_the_tail_the_history_rendering_lacks() {
        let (changed, value) = mark(r#"{"messages":[{"role":"user","content":"hi there"}]}"#);
        assert_eq!(changed, Some(2), "<assistant> <think>");
        assert_eq!(value["vllm_xargs"][GENERATION_PROMPT_TOKENS], json!(2));

        let (changed, _) = mark(
            r#"{"messages":[{"role":"system","content":"s"},{"role":"tool","content":"out","tool_call_id":"c1"}]}"#,
        );
        assert_eq!(changed, Some(2), "the last message alone decides");
    }

    #[test]
    fn a_token_straddling_the_boundary_is_counted_as_generation_prompt() {
        let (changed, _) = mark(r#"{"messages":[{"role":"user","content":"hi ~"}]}"#);
        assert_eq!(
            changed,
            Some(1),
            "`<end><assistant><think>` replaced `<end>`"
        );
    }

    #[test]
    fn a_request_without_a_generation_prompt_is_all_history() {
        let (changed, _) =
            mark(r#"{"messages":[{"role":"user","content":"hi"}],"add_generation_prompt":false}"#);
        assert_eq!(changed, Some(0));

        let (changed, _) = mark(
            r#"{"messages":[{"role":"assistant","content":"so"}],"continue_final_message":true}"#,
        );
        assert_eq!(changed, Some(0));
    }

    #[test]
    fn the_mark_joins_the_callers_xargs_and_replaces_its_own_key() {
        let (changed, value) = mark(
            r#"{"messages":[{"role":"user","content":"hi"}],"vllm_xargs":{"kv_transfer_params":1,"generation_prompt_tokens":99}}"#,
        );
        assert_eq!(changed, Some(2));
        assert_eq!(
            value["vllm_xargs"],
            json!({"kv_transfer_params": 1, GENERATION_PROMPT_TOKENS: 2})
        );
    }

    #[test]
    fn a_body_the_measure_cannot_read_passes_through_verbatim() {
        for json in [
            r#"{"prompt":"not a chat"}"#,
            r#"{"messages":[]}"#,
            r#"{"messages":[{"content":"no role"}]}"#,
            r#"{"messages":[{"role":"user","content":"hi"}],"vllm_xargs":7}"#,
        ] {
            let (changed, value) = mark(json);
            assert_eq!(changed, None, "{json}");
            assert_eq!(
                value,
                serde_json::from_str::<Value>(json).unwrap(),
                "{json}"
            );
        }
        let mut body = Bytes::from_static(b"not json at all");
        assert_eq!(mark_generation_prompt(&mut body, &boundary()), None);
        assert_eq!(&body[..], b"not json at all");
    }
}
