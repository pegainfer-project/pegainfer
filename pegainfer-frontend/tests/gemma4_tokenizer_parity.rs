//! Chat-render parity for Gemma 4 against a Hugging Face reference dumped by
//! `tools/accuracy/dump_gemma4_tokenizer_golden.py`. Owned by the frontend
//! crate because the contract under test is the chat-render path (the
//! vendored vllm renderer against the reference's Jinja2); the fixture is
//! shared test data and the checkpoint comes from the environment. The
//! maintainer runner executes it; directly, point `PEGAINFER_TEST_MODEL_PATH`
//! at the pinned 12B checkpoint the reference was dumped from and run
//! `cargo test -p pegainfer-frontend --test gemma4_tokenizer_parity -- --ignored`;
//! the file-hash guard binds it to exactly that checkpoint.
//!
//! Only the render comparison lives here, because only it compares two
//! implementations: minijinja against the reference's Jinja2. The token-id
//! comparisons that used to sit alongside it ran the same `tokenizers` crate on
//! both sides, so they gated the Python wrapper's version skew rather than
//! anything this repository decides.

mod common;

use serde_json::Value;

/// The committed reference, unless the variable points at one dumped from
/// another checkpoint.
fn golden() -> Value {
    common::golden(
        "PEGAINFER_GEMMA4_CHAT_GOLDEN",
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../test_data/gemma4-tokenizer-golden.json"
        ),
    )
}

fn model_path() -> String {
    std::env::var("PEGAINFER_TEST_MODEL_PATH").expect(
        "PEGAINFER_TEST_MODEL_PATH must point at the Gemma 4 checkpoint the \
         reference was dumped from",
    )
}

/// Covers the string content form only. The frontend's default `Auto` format
/// selects the parts form for this template, which renders system turns
/// differently — see docs/models/gemma4/tokenizer.md.
#[test]
#[ignore = "requires the pinned 12B checkpoint"]
fn string_form_chat_renders_match_hf_reference() {
    let golden = golden();
    let dir = model_path();
    common::assert_checkpoint_matches_reference(&golden, &dir);
    let mismatches = common::string_form_mismatches(&dir, &golden);
    common::assert_no_mismatches(&mismatches, &golden);
}
