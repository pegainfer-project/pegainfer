//! Qwen3.5 configuration: validated model geometry, tensor-parallel local
//! geometry, and the tokenizer schema.
//!
//! Ownership model (directional boundaries):
//! - [`model`] deserializes raw config and validates into a TP-agnostic
//!   [`Config35`];
//! - [`tp`] depends on [`model`] and produces the validated
//!   [`LocalGeometry`] that downstream code accepts;
//! - [`tokenizer`] owns the frontend tokenizer schema and decodable-vocab width;
//! - [`error`] owns the typed [`ConfigError`] variants.

use anyhow::Result;

mod error;
mod model;
mod tokenizer;
mod tp;

pub(crate) use model::Config35;
pub(crate) use model::GDN_AOT_KEY_HEAD_DIM;
pub(crate) use model::GDN_AOT_VALUE_HEAD_DIM;
pub(crate) use model::LINEAR_CONV_MAX_KERNEL_DIM;
pub(crate) use model::LayerType;
pub(crate) use tokenizer::tokenizer_effective_vocab;
pub(crate) use tp::LocalGeometry;
pub(crate) use tp::TensorParallelConfig;

/// Identity check that `json` is a Qwen3.5 config; size and shape validation
/// belong to the config loader.
pub(crate) fn probe_config_json(json: &serde_json::Value) -> Result<()> {
    let model_type = json
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if model_type != "qwen3_5" {
        anyhow::bail!("not a Qwen3.5 config: model_type={model_type}");
    }
    let architectures: Vec<&str> = json
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .map(|arr| arr.iter().filter_map(serde_json::Value::as_str).collect())
        .unwrap_or_default();
    anyhow::ensure!(
        architectures.contains(&"Qwen3_5ForConditionalGeneration"),
        "Qwen3.5 architectures must contain Qwen3_5ForConditionalGeneration"
    );
    let text_model_type = json
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    anyhow::ensure!(
        text_model_type == "qwen3_5_text",
        "Qwen3.5 text_config.model_type must be qwen3_5_text, got {text_model_type}"
    );
    Ok(())
}

#[cfg(test)]
mod tokenizer_tests {
    use std::fs;

    use super::tokenizer_effective_vocab;

    fn dir_with(json: &str, config: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("tokenizer.json"), json).unwrap();
        fs::write(dir.path().join("tokenizer_config.json"), config).unwrap();
        dir
    }

    #[test]
    fn effective_vocab_is_the_dense_decodable_width() {
        let dir = dir_with(
            r#"{ "model": { "vocab": { "a": 0, "b": 1, "c": 2 } }, "added_tokens": [ { "id": 3, "content": "<x>" } ] }"#,
            r#"{ "added_tokens_decoder": { "4": { "content": "<z>" }, "5": { "content": "<w>" }, "x": { "content": "<bad-key>" } } }"#,
        );
        assert_eq!(
            tokenizer_effective_vocab(dir.path().to_str().unwrap()).unwrap(),
            6
        );
    }

    #[test]
    fn effective_vocab_fails_on_a_sparse_id_space() {
        let dir = dir_with(
            r#"{ "model": { "vocab": { "a": 0, "b": 1 } } }"#,
            r#"{ "added_tokens_decoder": { "5": { "content": "<z>" } } }"#,
        );
        assert!(tokenizer_effective_vocab(dir.path().to_str().unwrap()).is_err());
    }

    #[test]
    fn one_invalid_decoder_entry_drops_all_decoder_tokens() {
        let dir = dir_with(
            r#"{ "model": { "vocab": { "a": 0, "b": 1 } } }"#,
            r#"{ "added_tokens_decoder": { "2": { "content": "<z>" }, "3": { "content": "<w>", "special": "not-a-bool" } } }"#,
        );
        assert_eq!(
            tokenizer_effective_vocab(dir.path().to_str().unwrap()).unwrap(),
            2
        );
    }
}
