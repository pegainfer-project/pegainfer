//! Qwen3.5 configuration: validated model geometry, tensor-parallel local
//! geometry, and model-line identity checks.
//!
//! Ownership model (directional boundaries):
//! - [`model`] deserializes raw config and validates into a TP-agnostic
//!   [`Config35`];
//! - [`tp`] depends on [`model`] and produces the validated
//!   [`LocalGeometry`] that downstream code accepts;
//! - [`error`] owns the typed [`ConfigError`] variants.

use anyhow::Result;

mod error;
mod model;
mod tp;

pub(crate) use model::Config35;
pub(crate) use model::GDN_AOT_KEY_HEAD_DIM;
pub(crate) use model::GDN_AOT_VALUE_HEAD_DIM;
pub(crate) use model::LINEAR_CONV_MAX_KERNEL_DIM;
pub(crate) use model::LayerType;
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
