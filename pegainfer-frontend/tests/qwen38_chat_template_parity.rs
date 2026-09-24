//! Chat-render parity for the Qwen3.8 line against the Hugging Face reference
//! dumped by `tools/accuracy/dump_chat_template_golden.py qwen38`.
//!
//! Qwen3.8's text tower is shape-identical to Qwen3.5's
//! (`docs/models/qwen35/support-qwen38.md`), so the model side needs no new
//! behaviour. Its chat template is where the two diverge: it reads
//! `reasoning_effort` (default `xhigh`, restricted to `xhigh|medium|low`), gates
//! the reasoning instructions on `enable_thinking`, and honours
//! `preserve_thinking`. Those knobs reach the template through the renderer's
//! `reasoning_effort` option and `template_kwargs`, so each combination is a case
//! here. Frontend-owned for the same reason as `gemma4_tokenizer_parity.rs`:
//! only the render compares two implementations — minijinja against the
//! reference's Jinja2.
//!
//!     PEGAINFER_TEST_MODEL_PATH=/path/to/Qwen3.8-27B \
//!       cargo test -r -p pegainfer-frontend --test qwen38_chat_template_parity \
//!       -- --ignored
//! The file-hash guard binds the run to the exact checkpoint the reference was
//! dumped from.

mod common;

use serde_json::Value;

fn golden() -> Value {
    common::golden(
        "PEGAINFER_QWEN38_CHAT_GOLDEN",
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../test_data/qwen38-chat-golden.json"
        ),
    )
}

#[test]
#[ignore = "requires the pinned Qwen3.8-27B checkpoint"]
fn chat_renders_match_hf_reference() {
    let golden = golden();
    let dir = common::model_path("Qwen3.8");
    common::assert_checkpoint_matches_reference(&golden, &dir);
    let mismatches = common::string_form_mismatches(&dir, &golden);
    common::assert_no_mismatches(&mismatches, &golden);
}
