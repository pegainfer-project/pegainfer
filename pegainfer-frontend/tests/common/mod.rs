//! Shared machinery for the chat-render parity gates. Each model line keeps a
//! thin `tests/<line>_chat_template_parity.rs` that names its golden and its
//! content format; what is common is how a case from the reference JSON becomes
//! a `ChatRequest` and how the renderer's output is compared.
//!
//! `#![allow(dead_code)]`: this module is compiled into every integration
//! binary that includes it, and each uses a subset.
#![allow(dead_code)]

use serde_json::Value;
use sha2::Digest;
use sha2::Sha256;
use vllm_chat::ChatMessage;
use vllm_chat::ChatOptions;
use vllm_chat::ChatRequest;
use vllm_chat::ChatRole;
use vllm_chat::GenerationPromptMode;
use vllm_chat::ReasoningEffort;
use vllm_text::Prompt;

/// The committed golden named by `env_key`, or `default_rel_path` under
/// `test_data/` when the variable is unset.
pub(crate) fn golden(env_key: &str, default_path: &str) -> Value {
    let path = std::env::var(env_key).unwrap_or_else(|_| default_path.to_string());
    let raw =
        std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("failed to read {path}: {err}"));
    serde_json::from_str(&raw).unwrap_or_else(|err| panic!("failed to parse {path}: {err}"))
}

/// Guards every parity case: a reference render only means something against
/// the exact checkpoint files it was dumped from.
pub(crate) fn assert_checkpoint_matches_reference(golden: &Value, model_path: &str) {
    let dir = std::path::Path::new(model_path);
    let expected = golden["file_sha256"]
        .as_object()
        .expect("golden file_sha256 must be an object");
    for (name, digest) in expected {
        let path = dir.join(name);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        let actual =
            Sha256::digest(&bytes)
                .iter()
                .fold(String::with_capacity(64), |mut hex, byte| {
                    use std::fmt::Write;
                    let _ = write!(hex, "{byte:02x}");
                    hex
                });
        let expected = digest.as_str().expect("sha256 must be a string");
        assert_eq!(
            actual, expected,
            "{name} does not match the pinned checkpoint the fixture was dumped from \
             ({actual} vs {expected}); this suite runs against that checkpoint only"
        );
    }
}

fn chat_role(name: &str) -> ChatRole {
    match name {
        "system" => ChatRole::System,
        "user" => ChatRole::User,
        "assistant" => ChatRole::Assistant,
        other => panic!("unexpected role {other}"),
    }
}

fn reasoning_effort(name: &str) -> ReasoningEffort {
    match name {
        "none" => ReasoningEffort::None,
        "minimal" => ReasoningEffort::Minimal,
        "low" => ReasoningEffort::Low,
        "medium" => ReasoningEffort::Medium,
        "high" => ReasoningEffort::High,
        "xhigh" => ReasoningEffort::XHigh,
        "max" => ReasoningEffort::Max,
        other => panic!(
            "the golden asks for reasoning_effort {other:?}, which the renderer cannot \
             express; extend this match arm"
        ),
    }
}

/// Rebuild the request the reference rendered. `template_kwargs` is what the
/// reference dump passed to `apply_chat_template`; `reasoning_effort` travels on
/// its own option because that is how the serving API exposes it.
pub(crate) fn chat_request(case: &Value) -> ChatRequest {
    let messages = case["messages"]
        .as_array()
        .expect("chat case messages")
        .iter()
        .map(|message| {
            ChatMessage::text(
                chat_role(message["role"].as_str().expect("message role")),
                message["content"].as_str().expect("message content"),
            )
        })
        .collect();
    let mut template_kwargs = serde_json::Map::new();
    let mut effort = None;
    if let Some(kwargs) = case.get("template_kwargs").and_then(Value::as_object) {
        for (key, value) in kwargs {
            if key == "reasoning_effort" {
                effort = Some(reasoning_effort(
                    value.as_str().expect("reasoning_effort must be a string"),
                ));
            } else {
                template_kwargs.insert(key.clone(), value.clone());
            }
        }
    }
    let generation_prompt_mode = if case["add_generation_prompt"]
        .as_bool()
        .expect("add_generation_prompt")
    {
        GenerationPromptMode::StartNewAssistant
    } else {
        GenerationPromptMode::NoGenerationPrompt
    };

    ChatRequest {
        messages,
        chat_options: ChatOptions {
            generation_prompt_mode,
            reasoning_effort: effort,
            template_kwargs: template_kwargs.into_iter().collect(),
            ..ChatOptions::default()
        },
        ..ChatRequest::for_test()
    }
}

/// Render every case in `golden["chat_templates"]` and collect the disagreements
/// so one run reports every divergence, not just the first.
pub(crate) fn render_mismatches<R>(golden: &Value, render: R) -> Vec<String>
where
    R: Fn(&ChatRequest) -> Result<Prompt, String>,
{
    let mut mismatches = Vec::new();
    let Value::Array(cases) = &golden["chat_templates"] else {
        panic!("golden chat_templates must be an array");
    };
    for case in cases {
        let name = case["name"].as_str().expect("chat case name");
        let expected = case["rendered"].as_str().expect("chat case rendered");
        match render(&chat_request(case)) {
            Ok(Prompt::Text(actual)) if actual == expected => {}
            Ok(Prompt::Text(actual)) => {
                mismatches.push(format!(
                    "{name}:\n  expected {expected:?}\n  got      {actual:?}"
                ));
            }
            Ok(Prompt::TokenIds(ids)) => {
                mismatches.push(format!("{name}: renderer returned token ids {ids:?}"));
            }
            Err(err) => mismatches.push(format!("{name}: render failed: {err}")),
        }
    }
    mismatches
}

pub(crate) fn assert_no_mismatches(mismatches: &[String], golden: &Value) {
    let cases = golden["chat_templates"]
        .as_array()
        .expect("golden chat_templates")
        .len();
    assert!(
        mismatches.is_empty(),
        "{} of {cases} chat renders disagree with the reference:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

/// Render every golden case through the checkpoint's own template in the string
/// content form. `Auto` can resolve a template to the parts form, whose system
/// turns render differently, so the comparison pins the string form.
pub(crate) fn string_form_mismatches(model_path: &str, golden: &Value) -> Vec<String> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let backends = runtime
        .block_on(vllm_chat::load_model_backends(
            model_path,
            vllm_chat::LoadModelBackendsOptions {
                language_model_only: true,
                chat_template_content_format: vllm_chat::ChatTemplateContentFormatOption::String,
                ..vllm_chat::LoadModelBackendsOptions::default()
            },
        ))
        .expect("failed to load chat backends");
    let renderer = backends.chat_backend.chat_renderer();
    render_mismatches(golden, |request| {
        renderer
            .render(request)
            .map(|rendered| rendered.prompt)
            .map_err(|err| err.to_string())
    })
}
