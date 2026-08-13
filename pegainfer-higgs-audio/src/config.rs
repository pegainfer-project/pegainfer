use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use serde_json::Value;

use crate::one_step_golden::CODEBOOK_VOCAB_SIZE;
use crate::one_step_golden::HIDDEN_SIZE;
use crate::one_step_golden::NUM_CODEBOOKS;

pub const EXPECTED_MODEL_TYPE: &str = "higgs_multimodal_qwen3";
pub const EXPECTED_ARCHITECTURE: &str = "HiggsMultimodalQwen3ForConditionalGeneration";
pub const EXPECTED_AUDIO_ENCODER_TYPE: &str = "discrete";
pub const EXPECTED_NUM_LAYERS: usize = 36;
pub const EXPECTED_NUM_ATTENTION_HEADS: usize = 32;
pub const EXPECTED_NUM_KV_HEADS: usize = 8;
pub const EXPECTED_HEAD_DIM: usize = 128;
pub const EXPECTED_INTERMEDIATE_SIZE: usize = 9728;
pub const EXPECTED_TEXT_VOCAB_SIZE: usize = 151_936;
pub const EXPECTED_ROPE_THETA: u64 = 1_000_000;
pub const EXPECTED_MODEL_CARD_CONTEXT: usize = 8192;

#[derive(Debug, Clone, PartialEq)]
pub struct HiggsConfig {
    pub model_type: String,
    pub architecture: String,
    pub audio_token_id: i64,
    pub text: TextConfig,
    pub audio: AudioEncoderConfig,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub eos_token_id: u32,
    pub tie_word_embeddings: bool,
    pub rope_theta: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioEncoderConfig {
    pub encoder_type: String,
    pub num_codebooks: usize,
    pub vocab_size: usize,
    pub out_dim: usize,
    pub tie_word_embeddings: bool,
    pub use_delay_pattern: bool,
}

impl HiggsConfig {
    pub fn from_model_dir(model_dir: impl AsRef<Path>) -> Result<Self> {
        let path = model_dir.as_ref().join("config.json");
        let value: Value = serde_json::from_slice(
            &std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
        )
        .with_context(|| format!("parse {}", path.display()))?;
        Self::from_json(&value)
    }

    pub fn from_json(value: &Value) -> Result<Self> {
        let text = value
            .get("text_config")
            .context("Higgs config missing text_config")?;
        let audio = value
            .get("audio_encoder_config")
            .context("Higgs config missing audio_encoder_config")?;
        let config = Self {
            model_type: str_field(value, "model_type")?.to_string(),
            architecture: first_architecture(value)?,
            audio_token_id: i64_field(value, "audio_token_id")?,
            text: TextConfig {
                hidden_size: usize_field(text, "hidden_size")?,
                intermediate_size: usize_field(text, "intermediate_size")?,
                num_hidden_layers: usize_field(text, "num_hidden_layers")?,
                num_attention_heads: usize_field(text, "num_attention_heads")?,
                num_key_value_heads: usize_field(text, "num_key_value_heads")?,
                head_dim: usize_field(text, "head_dim")?,
                vocab_size: usize_field(text, "vocab_size")?,
                rms_norm_eps: f32_field(text, "rms_norm_eps")?,
                max_position_embeddings: usize_field(text, "max_position_embeddings")?,
                eos_token_id: u32_field(text, "eos_token_id")?,
                tie_word_embeddings: bool_field(text, "tie_word_embeddings")?,
                rope_theta: text
                    .get("rope_parameters")
                    .and_then(|rope| rope.get("rope_theta"))
                    .and_then(Value::as_u64)
                    .context("text_config.rope_parameters.rope_theta missing or not u64")?,
            },
            audio: AudioEncoderConfig {
                encoder_type: str_field(audio, "encoder_type")?.to_string(),
                num_codebooks: usize_field(audio, "num_codebooks")?,
                vocab_size: usize_field(audio, "vocab_size")?,
                out_dim: usize_field(audio, "out_dim")?,
                tie_word_embeddings: bool_field(audio, "tie_word_embeddings")?,
                use_delay_pattern: bool_field(audio, "use_delay_pattern")?,
            },
        };
        config.validate_current_contract()?;
        Ok(config)
    }

    pub fn validate_current_contract(&self) -> Result<()> {
        if self.model_type != EXPECTED_MODEL_TYPE {
            bail!("unexpected Higgs model_type {}", self.model_type);
        }
        if self.architecture != EXPECTED_ARCHITECTURE {
            bail!("unexpected Higgs architecture {}", self.architecture);
        }
        if self.audio_token_id != -100 {
            bail!(
                "Higgs audio_token_id must be -100, got {}",
                self.audio_token_id
            );
        }
        if self.text.hidden_size != HIDDEN_SIZE
            || self.text.intermediate_size != EXPECTED_INTERMEDIATE_SIZE
            || self.text.num_hidden_layers != EXPECTED_NUM_LAYERS
            || self.text.num_attention_heads != EXPECTED_NUM_ATTENTION_HEADS
            || self.text.num_key_value_heads != EXPECTED_NUM_KV_HEADS
            || self.text.head_dim != EXPECTED_HEAD_DIM
            || self.text.vocab_size != EXPECTED_TEXT_VOCAB_SIZE
            || self.text.rope_theta != EXPECTED_ROPE_THETA
            || !self.text.tie_word_embeddings
        {
            bail!("Higgs text_config does not match the pinned one-step contract: {self:?}");
        }
        if self.audio.encoder_type != EXPECTED_AUDIO_ENCODER_TYPE
            || self.audio.num_codebooks != NUM_CODEBOOKS
            || self.audio.vocab_size != CODEBOOK_VOCAB_SIZE
            || self.audio.out_dim != HIDDEN_SIZE
            || !self.audio.tie_word_embeddings
            || !self.audio.use_delay_pattern
        {
            bail!(
                "Higgs audio_encoder_config does not match the pinned one-step contract: {self:?}"
            );
        }
        Ok(())
    }

    pub fn kv_bytes_per_position_bf16(&self) -> usize {
        self.text.num_hidden_layers
            * 2
            * self.text.num_key_value_heads
            * self.text.head_dim
            * std::mem::size_of::<half::bf16>()
    }

    pub fn kv_bytes_for_positions_bf16(&self, positions: usize) -> usize {
        self.kv_bytes_per_position_bf16() * positions
    }
}

fn first_architecture(value: &Value) -> Result<String> {
    value
        .get("architectures")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
        .and_then(Value::as_str)
        .map(str::to_string)
        .context("Higgs config missing architectures[0]")
}

fn str_field<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .with_context(|| format!("missing string field {key}"))
}

fn usize_field(value: &Value, key: &str) -> Result<usize> {
    let raw = value
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("missing usize field {key}"))?;
    usize::try_from(raw).with_context(|| format!("{key} does not fit usize"))
}

fn u32_field(value: &Value, key: &str) -> Result<u32> {
    let raw = value
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("missing u32 field {key}"))?;
    u32::try_from(raw).with_context(|| format!("{key} does not fit u32"))
}

fn i64_field(value: &Value, key: &str) -> Result<i64> {
    value
        .get(key)
        .and_then(Value::as_i64)
        .with_context(|| format!("missing i64 field {key}"))
}

fn f32_field(value: &Value, key: &str) -> Result<f32> {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map(|value| value as f32)
        .with_context(|| format!("missing f32 field {key}"))
}

fn bool_field(value: &Value, key: &str) -> Result<bool> {
    value
        .get(key)
        .and_then(Value::as_bool)
        .with_context(|| format!("missing bool field {key}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_config() -> serde_json::Value {
        serde_json::json!({
            "architectures": [EXPECTED_ARCHITECTURE],
            "audio_token_id": -100,
            "model_type": EXPECTED_MODEL_TYPE,
            "text_config": {
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": EXPECTED_INTERMEDIATE_SIZE,
                "num_hidden_layers": EXPECTED_NUM_LAYERS,
                "num_attention_heads": EXPECTED_NUM_ATTENTION_HEADS,
                "num_key_value_heads": EXPECTED_NUM_KV_HEADS,
                "head_dim": EXPECTED_HEAD_DIM,
                "vocab_size": EXPECTED_TEXT_VOCAB_SIZE,
                "rms_norm_eps": 1e-6,
                "max_position_embeddings": 32768,
                "eos_token_id": 151643,
                "tie_word_embeddings": true,
                "rope_parameters": {"rope_theta": EXPECTED_ROPE_THETA}
            },
            "audio_encoder_config": {
                "encoder_type": EXPECTED_AUDIO_ENCODER_TYPE,
                "num_codebooks": NUM_CODEBOOKS,
                "vocab_size": CODEBOOK_VOCAB_SIZE,
                "out_dim": HIDDEN_SIZE,
                "tie_word_embeddings": true,
                "use_delay_pattern": true
            }
        })
    }

    #[test]
    fn parses_pinned_higgs_config_shape_contract() {
        let config = HiggsConfig::from_json(&minimal_config()).unwrap();
        assert_eq!(config.kv_bytes_per_position_bf16(), 144 * 1024);
        assert_eq!(
            config.kv_bytes_for_positions_bf16(EXPECTED_MODEL_CARD_CONTEXT),
            1152 * 1024 * 1024
        );
    }

    #[test]
    fn rejects_non_discrete_audio_encoder() {
        let mut value = minimal_config();
        value["audio_encoder_config"]["encoder_type"] = serde_json::json!("whisper");
        let err = HiggsConfig::from_json(&value).unwrap_err().to_string();
        assert!(err.contains("audio_encoder_config"));
    }
}
