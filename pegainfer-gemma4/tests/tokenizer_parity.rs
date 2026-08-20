//! Parity against a Hugging Face reference dumped by
//! `tools/accuracy/dump_gemma4_tokenizer_golden.py`. Point
//! `PEGAINFER_TEST_MODEL_PATH` at the pinned 12B checkpoint the reference was
//! dumped from and run with `--ignored`; the file-hash guard binds the suite to
//! exactly that checkpoint.
//!
//! Both sides tokenize with the same `tokenizers` crate, so the token-id tests
//! gate the Python wrapper's behaviour and version skew, not the tokenization
//! algorithm. The render test compares two different Jinja implementations and
//! carries the weight here.

mod common;

use serde_json::Value;
use sha2::Digest;
use sha2::Sha256;
use vllm_chat::ChatMessage;
use vllm_chat::ChatOptions;
use vllm_chat::ChatRequest;
use vllm_chat::ChatRole;
use vllm_chat::ChatTemplateContentFormatOption;
use vllm_chat::GenerationPromptMode;
use vllm_chat::LoadModelBackendsOptions;
use vllm_chat::load_model_backends;
use vllm_text::Prompt;

const GOLDEN_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../test_data/gemma4-tokenizer-golden.json"
);

fn golden() -> Value {
    let raw = std::fs::read_to_string(GOLDEN_PATH)
        .unwrap_or_else(|err| panic!("failed to read {GOLDEN_PATH}: {err}"));
    serde_json::from_str(&raw).unwrap_or_else(|err| panic!("failed to parse {GOLDEN_PATH}: {err}"))
}

fn model_path() -> String {
    std::env::var("PEGAINFER_TEST_MODEL_PATH").expect(
        "PEGAINFER_TEST_MODEL_PATH must point at the pinned 12B Gemma 4 checkpoint \
         the reference was dumped from",
    )
}

/// Guards every parity test: the fixture only means something against the exact
/// checkpoint it was dumped from, which is the pinned 12B one.
fn assert_checkpoint_matches_reference(golden: &Value) {
    let dir = model_path();
    let expected = golden["file_sha256"]
        .as_object()
        .expect("golden file_sha256");
    for (name, digest) in expected {
        let path = std::path::Path::new(&dir).join(name);
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
            "{name} does not match the pinned 12B reference the fixture was dumped from \
             ({actual} vs {expected}); this suite runs against that checkpoint only"
        );
    }
}

fn expected_ids(value: &Value, key: &str) -> Vec<u32> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("golden entry missing {key}"))
        .iter()
        .map(|id| {
            u32::try_from(id.as_u64().expect("token id must be a number"))
                .expect("token id must fit u32")
        })
        .collect()
}

#[test]
#[ignore = "requires the pinned 12B checkpoint"]
fn probe_ids_match_hf_reference() {
    let golden = golden();
    assert_checkpoint_matches_reference(&golden);
    let tokenizer = common::load_tokenizer(&model_path());
    let mut mismatches = Vec::new();

    for probe in golden["probes"].as_array().expect("golden probes") {
        let name = probe["name"].as_str().expect("probe name");
        let text = probe["text"].as_str().expect("probe text");
        for (key, add_specials) in [("ids_plain", false), ("ids_with_specials", true)] {
            let expected = expected_ids(probe, key);
            let actual = tokenizer
                .encode(text, add_specials)
                .unwrap_or_else(|err| panic!("encode failed for {name}/{key}: {err}"));
            if actual != expected {
                mismatches.push(format!(
                    "{name}/{key}: expected {expected:?}, got {actual:?}"
                ));
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} of {} probe encodings disagree with the reference:\n{}",
        mismatches.len(),
        golden["probes"].as_array().map_or(0, Vec::len) * 2,
        mismatches.join("\n")
    );
}

#[test]
#[ignore = "requires the pinned 12B checkpoint"]
fn special_token_ids_match_hf_reference() {
    let golden = golden();
    assert_checkpoint_matches_reference(&golden);
    let tokenizer = common::load_tokenizer(&model_path());
    let specials = golden["special_tokens"]
        .as_object()
        .expect("golden special_tokens");
    let mut mismatches = Vec::new();

    for (name, entry) in specials {
        let token = entry["token"].as_str().expect("special token text");
        let expected = u32::try_from(entry["id"].as_u64().expect("special token id"))
            .expect("special token id must fit u32");
        let actual = tokenizer
            .encode(token, false)
            .unwrap_or_else(|err| panic!("encode failed for {name}: {err}"));
        if actual != vec![expected] {
            mismatches.push(format!(
                "{name} ({token}): expected [{expected}], got {actual:?}"
            ));
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} of {} special tokens disagree with the reference:\n{}",
        mismatches.len(),
        specials.len(),
        mismatches.join("\n")
    );
}

fn chat_request(case: &Value) -> ChatRequest {
    let messages = case["messages"]
        .as_array()
        .expect("chat case messages")
        .iter()
        .map(|message| {
            let role = match message["role"].as_str().expect("message role") {
                "system" => ChatRole::System,
                "user" => ChatRole::User,
                "assistant" => ChatRole::Assistant,
                other => panic!("unexpected role {other}"),
            };
            ChatMessage::text(role, message["content"].as_str().expect("message content"))
        })
        .collect();
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
            ..ChatOptions::default()
        },
        ..ChatRequest::for_test()
    }
}

/// Covers the string content form only. The frontend's default `Auto` format
/// selects the parts form for this template, which renders system turns
/// differently — see docs/models/gemma4/tokenizer.md.
#[test]
#[ignore = "requires the pinned 12B checkpoint"]
fn string_form_chat_renders_match_hf_reference() {
    let golden = golden();
    assert_checkpoint_matches_reference(&golden);
    let cases = golden["chat_templates"]
        .as_array()
        .expect("golden chat_templates");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");
    let backends = runtime
        .block_on(load_model_backends(
            &model_path(),
            LoadModelBackendsOptions {
                language_model_only: true,
                chat_template_content_format: ChatTemplateContentFormatOption::String,
                ..LoadModelBackendsOptions::default()
            },
        ))
        .expect("failed to load chat backends");
    let renderer = backends.chat_backend.chat_renderer();

    let mut mismatches = Vec::new();
    for case in cases {
        let name = case["name"].as_str().expect("chat case name");
        let expected = case["rendered"].as_str().expect("chat case rendered");
        let rendered = renderer
            .render(&chat_request(case))
            .unwrap_or_else(|err| panic!("render failed for {name}: {err}"));
        match rendered.prompt {
            Prompt::Text(actual) if actual == expected => {}
            Prompt::Text(actual) => {
                mismatches.push(format!("{name}: expected {expected:?}, got {actual:?}"));
            }
            Prompt::TokenIds(ids) => {
                mismatches.push(format!("{name}: renderer returned token ids {ids:?}"));
            }
        }
    }

    assert!(
        mismatches.is_empty(),
        "{} of {} chat renders disagree with the reference:\n{}",
        mismatches.len(),
        cases.len(),
        mismatches.join("\n")
    );
}
