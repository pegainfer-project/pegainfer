//! Validated Qwen3.5 model config.
//!
//! This module owns the *model* geometry: deserialized-from-disk [`RawConfig`]
//! is converted once into a validated [`Config35`] via [`TryFrom`]. It has no
//! knowledge of tensor parallelism — sharded/local dimensions, shard ranges and
//! the TP assignment live in [`super::tp`] as a separate directional boundary.

use std::fs;

use anyhow::Result;
use serde::Deserialize;

use super::error::ConfigError;

/// Which attention variant a layer uses. Model-level (TP-agnostic) layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerType {
    FullAttention,
    LinearAttention,
}

/// RoPE block inside `text_config`. Only these two keys are read.
#[derive(Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f64,
    partial_rotary_factor: f64,
}

/// Raw `text_config` block, exactly as it appears on disk.
#[derive(Debug, Deserialize)]
pub(crate) struct TextConfig {
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    rms_norm_eps: f64,
    layer_types: Vec<String>,
    linear_conv_kernel_dim: usize,
    linear_key_head_dim: usize,
    linear_num_key_heads: usize,
    linear_num_value_heads: usize,
    linear_value_head_dim: usize,
    rope_parameters: RopeParameters,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,
    eos_token_id: u32,
}

/// Raw Qwen3.5 config as deserialized from `config.json`. Held only long enough
/// to be validated into a [`Config35`].
#[derive(Debug, Deserialize)]
pub(crate) struct RawConfig {
    text_config: TextConfig,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,
}

/// Head dims baked into the kernels; head counts are runtime parameters.
pub(crate) const GDN_AOT_KEY_HEAD_DIM: usize = 128;
pub(crate) const GDN_AOT_VALUE_HEAD_DIM: usize = 128;
pub(crate) const LINEAR_CONV_MAX_KERNEL_DIM: usize = 4;
const FULL_ATTN_HEAD_DIM: usize = 256;

/// Validated Qwen3.5 model configuration (text-only).
///
/// Every cross-field and kernel-AOT rule is enforced here, at the single
/// [`Self::try_from`]/[`Self::from_file`] boundary, so downstream code can only
/// observe a model that is known to be loadable. Tensor-parallel sharding is
/// never a property of this type.
#[derive(Debug)]
pub(crate) struct Config35 {
    // Common
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f32,
    pub(crate) eos_token_id: u32,

    // Full attention params
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,

    // Linear attention params
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_key_head_dim: usize,
    pub(crate) linear_num_value_heads: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) linear_conv_kernel_dim: usize,

    // RoPE
    pub(crate) rope_theta: f32,
    pub(crate) rotary_dim: usize,
    pub(crate) max_position_embeddings: usize,

    // Layer layout
    pub(crate) layer_types: Vec<LayerType>,

    /// `false` requires a top-level `lm_head.weight`; `true` reuses `embed_tokens`.
    pub(crate) tie_word_embeddings: bool,

    /// Token-selection width: `vocab_size` bounded to the frontend-decodable vocab.
    pub(crate) selection_vocab: usize,
}

impl Config35 {
    /// Load and validate `config.json` from a model directory.
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        let raw: RawConfig = serde_json::from_str(&content)?;
        let config = Self::try_from(raw)?;
        Ok(config)
    }

    /// Number of full attention layers in the model.
    pub(crate) fn num_full_attention_layers(&self) -> usize {
        self.layer_types
            .iter()
            .filter(|&&t| t == LayerType::FullAttention)
            .count()
    }

    /// Q dimension for full attention (without gate).
    pub(crate) fn full_attn_q_dim(&self) -> usize {
        self.num_attention_heads * self.head_dim
    }

    /// KV dimension for full attention.
    pub(crate) fn full_attn_kv_dim(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub(crate) fn decode_group_is_compiled(&self) -> bool {
        // Uncompiled GQA groups use the batched hybrid eager fallback.
        pegainfer_core::ops::SUPPORTED_GQA_GROUP_SIZES
            .contains(&(self.num_attention_heads / self.num_key_value_heads))
    }

    /// QKV projection output dimension for linear attention.
    pub(crate) fn linear_attn_qkv_dim(&self) -> usize {
        let q_dim = self.linear_num_key_heads * self.linear_key_head_dim;
        let k_dim = q_dim;
        let v_dim = self.linear_num_value_heads * self.linear_value_head_dim;
        q_dim + k_dim + v_dim
    }

    /// Z projection output dimension for linear attention.
    pub(crate) fn linear_attn_z_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Bound the output-selection width to the frontend-decodable vocab.
    ///
    /// The frontend decodes a dense prefix of the vocab; the checkpoint may pad
    /// beyond it. Refusing a tokenizer wider than the checkpoint is the
    /// fail-closed rule, and it is checked here at the validation boundary
    /// rather than scattered through the loader.
    pub(crate) fn bound_selection_vocab(
        &mut self,
        effective_vocab: usize,
    ) -> Result<(), ConfigError> {
        if effective_vocab > self.vocab_size {
            return Err(ConfigError::EffectiveVocabExceedsCheckpoint {
                used: effective_vocab,
                vocab_size: self.vocab_size,
            });
        }
        self.selection_vocab = effective_vocab;
        Ok(())
    }
}

/// Validate a raw Qwen3.5 config into a [`Config35`].
///
/// Every check here is a static, load-time invariant: an invalid model file is
/// rejected before any weight is mapped or any CUDA buffer is allocated. The
/// kernel head dims are compile-time AOT constants, so a config that disagrees
/// cannot be served and is rejected here rather than at first kernel launch.
impl TryFrom<RawConfig> for Config35 {
    type Error = ConfigError;

    fn try_from(raw: RawConfig) -> std::result::Result<Self, Self::Error> {
        let root_max_position_embeddings = raw.max_position_embeddings;
        let root_tie_word_embeddings = raw.tie_word_embeddings;
        let t = raw.text_config;

        let tie_word_embeddings = t
            .tie_word_embeddings
            .or(root_tie_word_embeddings)
            .ok_or(ConfigError::MissingTieWordEmbeddings)?;

        let layer_types: Vec<LayerType> = t
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" => Ok(LayerType::FullAttention),
                "linear_attention" => Ok(LayerType::LinearAttention),
                other => Err(ConfigError::UnknownLayerType(other.to_string())),
            })
            .collect::<std::result::Result<_, _>>()?;

        if layer_types.len() != t.num_hidden_layers {
            return Err(ConfigError::LayerTypeCountMismatch {
                actual: layer_types.len(),
                expected: t.num_hidden_layers,
            });
        }

        let rotary_dim = (t.head_dim as f64 * t.rope_parameters.partial_rotary_factor) as usize;
        if rotary_dim == 0 {
            return Err(ConfigError::ZeroRotaryDim);
        }

        let max_position_embeddings = t
            .max_position_embeddings
            .or(root_max_position_embeddings)
            .ok_or(ConfigError::MissingMaxPositionEmbeddings)?;
        if max_position_embeddings == 0 {
            return Err(ConfigError::NonPositiveMaxPositionEmbeddings);
        }

        if t.linear_key_head_dim != GDN_AOT_KEY_HEAD_DIM
            || t.linear_value_head_dim != GDN_AOT_VALUE_HEAD_DIM
        {
            return Err(ConfigError::GdnAotHeadDimMismatch {
                key: t.linear_key_head_dim,
                value: t.linear_value_head_dim,
                expected_key: GDN_AOT_KEY_HEAD_DIM,
                expected_value: GDN_AOT_VALUE_HEAD_DIM,
            });
        }
        if t.head_dim != FULL_ATTN_HEAD_DIM {
            return Err(ConfigError::FullAttnHeadDimMismatch {
                expected: FULL_ATTN_HEAD_DIM,
                actual: t.head_dim,
            });
        }
        if !(1..=LINEAR_CONV_MAX_KERNEL_DIM).contains(&t.linear_conv_kernel_dim) {
            return Err(ConfigError::LinearConvKernelDim {
                max: LINEAR_CONV_MAX_KERNEL_DIM,
                actual: t.linear_conv_kernel_dim,
            });
        }
        if t.linear_num_key_heads == 0
            || !t
                .linear_num_value_heads
                .is_multiple_of(t.linear_num_key_heads)
        {
            return Err(ConfigError::LinearHeadDivisibility {
                key_heads: t.linear_num_key_heads,
                value_heads: t.linear_num_value_heads,
            });
        }
        if t.num_key_value_heads == 0
            || !t.num_attention_heads.is_multiple_of(t.num_key_value_heads)
        {
            return Err(ConfigError::AttentionHeadDivisibility {
                attention_heads: t.num_attention_heads,
                key_value_heads: t.num_key_value_heads,
            });
        }

        Ok(Self {
            hidden_size: t.hidden_size,
            intermediate_size: t.intermediate_size,
            num_hidden_layers: t.num_hidden_layers,
            vocab_size: t.vocab_size,
            rms_norm_eps: t.rms_norm_eps as f32,
            eos_token_id: t.eos_token_id,
            num_attention_heads: t.num_attention_heads,
            num_key_value_heads: t.num_key_value_heads,
            head_dim: t.head_dim,
            linear_num_key_heads: t.linear_num_key_heads,
            linear_key_head_dim: t.linear_key_head_dim,
            linear_num_value_heads: t.linear_num_value_heads,
            linear_value_head_dim: t.linear_value_head_dim,
            linear_conv_kernel_dim: t.linear_conv_kernel_dim,
            rope_theta: t.rope_parameters.rope_theta as f32,
            rotary_dim,
            max_position_embeddings,
            layer_types,
            tie_word_embeddings,
            selection_vocab: t.vocab_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_RAW: &str = r#"{
  "max_position_embeddings": 4096,
  "tie_word_embeddings": true,
  "text_config": {
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "vocab_size": 1000,
    "rms_norm_eps": 1e-6,
    "layer_types": ["linear_attention", "full_attention"],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
    "rope_parameters": { "rope_theta": 10000.0, "partial_rotary_factor": 0.25 },
    "eos_token_id": 0
  }
}"#;

    fn parse(raw: &str) -> RawConfig {
        serde_json::from_str(raw).expect("fixture must deserialize")
    }

    fn config(raw: &str) -> Config35 {
        Config35::try_from(parse(raw)).expect("fixture must validate")
    }

    #[test]
    fn valid_fixture_loads() {
        let config = config(VALID_RAW);
        assert_eq!(config.num_full_attention_layers(), 1);
        assert_eq!(config.vocab_size, 1000);
    }

    #[test]
    fn missing_tie_word_embeddings_is_typed() {
        let json = VALID_RAW.replace("\"tie_word_embeddings\": true,", "");
        let err = Config35::try_from(parse(&json)).unwrap_err();
        assert_eq!(err, ConfigError::MissingTieWordEmbeddings);
    }

    #[test]
    fn unknown_layer_type_is_typed() {
        let json = VALID_RAW.replace("linear_attention", "window_attention");
        let err = Config35::try_from(parse(&json)).unwrap_err();
        assert_eq!(
            err,
            ConfigError::UnknownLayerType("window_attention".to_string())
        );
    }

    #[test]
    fn layer_type_count_mismatch_is_typed() {
        let json = VALID_RAW.replace(
            "[\"linear_attention\", \"full_attention\"]",
            "[\"linear_attention\"]",
        );
        let err = Config35::try_from(parse(&json)).unwrap_err();
        assert_eq!(
            err,
            ConfigError::LayerTypeCountMismatch {
                actual: 1,
                expected: 2,
            }
        );
    }

    #[test]
    fn missing_max_position_embeddings_is_typed() {
        let json = VALID_RAW.replace("\"max_position_embeddings\": 4096,", "");
        let err = Config35::try_from(parse(&json)).unwrap_err();
        assert_eq!(err, ConfigError::MissingMaxPositionEmbeddings);
    }

    #[test]
    fn wide_linear_conv_kernel_is_typed() {
        let json = VALID_RAW.replace(
            "\"linear_conv_kernel_dim\": 4",
            "\"linear_conv_kernel_dim\": 5",
        );
        let err = Config35::try_from(parse(&json)).unwrap_err();
        assert_eq!(
            err,
            ConfigError::LinearConvKernelDim {
                max: LINEAR_CONV_MAX_KERNEL_DIM,
                actual: 5,
            }
        );
    }

    #[test]
    fn acceptance_of_48_value_heads() {
        // Regression guard: 48 value heads with 16 key heads must load.
        config(VALID_RAW);
    }
}
