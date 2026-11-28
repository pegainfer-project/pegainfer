//! The Qwen3 checkpoint fields this backend reads, parsed the way the CUDA line
//! parses them (`pegainfer-qwen3/src/config.rs`).

use std::fs;

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct Config {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab_size: usize,
    /// Token-selection width: `vocab_size` bounded to the frontend-decodable vocab.
    #[serde(skip)]
    pub(crate) decodable_vocab: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    pub(crate) max_position_embeddings: usize,
    pub(crate) tie_word_embeddings: bool,
    #[serde(skip)]
    pub(crate) stop_token_ids: Vec<u32>,
}

fn default_max_position_embeddings() -> usize {
    40960
}

impl Config {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{model_path}/config.json");
        let content = fs::read_to_string(&config_path)?;
        let mut config: Config = serde_json::from_str(&content)?;
        config.decodable_vocab = config.vocab_size;
        ensure!(
            config.num_key_value_heads > 0
                && config
                    .num_attention_heads
                    .is_multiple_of(config.num_key_value_heads),
            "num_attention_heads ({}) must be a positive multiple of num_key_value_heads ({})",
            config.num_attention_heads,
            config.num_key_value_heads,
        );
        config.stop_token_ids =
            pegainfer_frontend::eos::eos_token_ids(model_path, config.vocab_size)?;
        Ok(config)
    }
}

/// Identity check that `json` is a Qwen3 config; size and shape validation belong to the config loader.
pub(crate) fn probe_config_json(json: &Value) -> Result<()> {
    let model_type = json.get("model_type").and_then(Value::as_str).unwrap_or("");
    if model_type != "qwen3" {
        bail!("not a Qwen3 config: model_type={model_type}");
    }
    let architectures: Vec<&str> = json
        .get("architectures")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    ensure!(
        architectures.contains(&"Qwen3ForCausalLM"),
        "Qwen3 architectures must contain Qwen3ForCausalLM"
    );
    Ok(())
}
