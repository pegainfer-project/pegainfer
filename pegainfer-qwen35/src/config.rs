use std::collections::HashSet;
use std::fs;

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use log::warn;
use serde::Deserialize;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TensorParallelConfig {
    pub(crate) rank: usize,
    pub(crate) world_size: usize,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerType {
    FullAttention,
    LinearAttention,
}

#[derive(Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f64,
    partial_rotary_factor: f64,
}

#[derive(Debug, Deserialize)]
struct TextConfig {
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

#[derive(Debug, Deserialize)]
struct RawConfig {
    text_config: TextConfig,
    max_position_embeddings: Option<usize>,
    tie_word_embeddings: Option<bool>,
}

/// Qwen3.5 model configuration (text-only).
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

/// Head dims baked into the kernels; head counts are runtime parameters.
pub(crate) const GDN_AOT_KEY_HEAD_DIM: usize = 128;
pub(crate) const GDN_AOT_VALUE_HEAD_DIM: usize = 128;
pub(crate) const LINEAR_CONV_MAX_KERNEL_DIM: usize = 4;
const FULL_ATTN_HEAD_DIM: usize = 256;

impl Config35 {
    pub(crate) fn from_file(model_path: &str) -> Result<Self> {
        let config_path = format!("{}/config.json", model_path);
        let content = fs::read_to_string(&config_path)?;
        let raw: RawConfig = serde_json::from_str(&content)?;
        let root_max_position_embeddings = raw.max_position_embeddings;
        let root_tie_word_embeddings = raw.tie_word_embeddings;
        let t = raw.text_config;

        let tie_word_embeddings = t
            .tie_word_embeddings
            .or(root_tie_word_embeddings)
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 config missing tie_word_embeddings"))?;

        let layer_types: Vec<LayerType> = t
            .layer_types
            .iter()
            .map(|s| match s.as_str() {
                "full_attention" => Ok(LayerType::FullAttention),
                "linear_attention" => Ok(LayerType::LinearAttention),
                other => Err(anyhow::anyhow!("Unknown layer type: {}", other)),
            })
            .collect::<Result<_>>()?;

        anyhow::ensure!(
            layer_types.len() == t.num_hidden_layers,
            "layer_types length {} != num_hidden_layers {}",
            layer_types.len(),
            t.num_hidden_layers
        );

        let rotary_dim = (t.head_dim as f64 * t.rope_parameters.partial_rotary_factor) as usize;
        anyhow::ensure!(rotary_dim > 0, "Qwen3.5 rotary_dim must be positive");
        let max_position_embeddings = t
            .max_position_embeddings
            .or(root_max_position_embeddings)
            .ok_or_else(|| anyhow::anyhow!("Qwen3.5 config missing max_position_embeddings"))?;
        anyhow::ensure!(
            max_position_embeddings > 0,
            "Qwen3.5 max_position_embeddings must be positive"
        );

        anyhow::ensure!(
            t.linear_key_head_dim == GDN_AOT_KEY_HEAD_DIM
                && t.linear_value_head_dim == GDN_AOT_VALUE_HEAD_DIM,
            "Qwen3.5 GDN Triton-AOT kernels are baked for key/value head dim {}/{}; \
             config has {}/{} (dims are baked into the AOT signatures in pegainfer-kernels/build.rs).",
            GDN_AOT_KEY_HEAD_DIM,
            GDN_AOT_VALUE_HEAD_DIM,
            t.linear_key_head_dim,
            t.linear_value_head_dim,
        );
        anyhow::ensure!(
            t.head_dim == FULL_ATTN_HEAD_DIM,
            "Qwen3.5 full-attention kernels are baked for head_dim {}; config has {}.",
            FULL_ATTN_HEAD_DIM,
            t.head_dim,
        );
        anyhow::ensure!(
            (1..=LINEAR_CONV_MAX_KERNEL_DIM).contains(&t.linear_conv_kernel_dim),
            "Qwen3.5 linear conv decode kernels support kernel_dim in 1..={}; config has {}.",
            LINEAR_CONV_MAX_KERNEL_DIM,
            t.linear_conv_kernel_dim,
        );
        anyhow::ensure!(
            t.linear_num_key_heads > 0
                && t.linear_num_value_heads
                    .is_multiple_of(t.linear_num_key_heads),
            "Qwen3.5 GDN kernels require linear_num_value_heads ({}) divisible by \
             linear_num_key_heads ({})",
            t.linear_num_value_heads,
            t.linear_num_key_heads,
        );
        anyhow::ensure!(
            t.num_key_value_heads > 0
                && t.num_attention_heads.is_multiple_of(t.num_key_value_heads),
            "Qwen3.5 num_attention_heads ({}) must be a positive multiple of \
             num_key_value_heads ({})",
            t.num_attention_heads,
            t.num_key_value_heads,
        );

        let config = Self {
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
        };
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

    pub(crate) fn local_num_attention_heads(&self, tp: TensorParallelConfig) -> usize {
        self.num_attention_heads / tp.world_size
    }

    pub(crate) fn local_num_key_value_heads(&self, tp: TensorParallelConfig) -> usize {
        self.num_key_value_heads / tp.world_size
    }

    pub(crate) fn local_intermediate_size(&self, tp: TensorParallelConfig) -> usize {
        self.intermediate_size / tp.world_size
    }

    pub(crate) fn local_full_attn_q_dim(&self, tp: TensorParallelConfig) -> usize {
        self.local_num_attention_heads(tp) * self.head_dim
    }

    pub(crate) fn local_full_attn_kv_dim(&self, tp: TensorParallelConfig) -> usize {
        self.local_num_key_value_heads(tp) * self.head_dim
    }

    /// Local gated full-attention q projection output dimension.
    pub(crate) fn local_full_attn_gated_q_dim(&self, tp: TensorParallelConfig) -> usize {
        self.local_full_attn_q_dim(tp) * 2
    }
}

impl TensorParallelConfig {
    pub(crate) fn validate_for(self, config: &Config35, enable_cuda_graph: bool) -> Result<()> {
        if self.world_size == 0 {
            return Err(anyhow::anyhow!("tensor_parallel.world_size must be >= 1"));
        }
        if self.rank >= self.world_size {
            return Err(anyhow::anyhow!(
                "tensor_parallel.rank {} must be < world_size {}",
                self.rank,
                self.world_size
            ));
        }
        if self.is_sharded() && enable_cuda_graph {
            return Err(anyhow::anyhow!(
                "Qwen3.5 tensor parallelism is eager-only in Phase 1; disable CUDA Graph for tp world_size={}",
                self.world_size
            ));
        }
        if !config.num_attention_heads.is_multiple_of(self.world_size) {
            return Err(anyhow::anyhow!(
                "num_attention_heads={} not divisible by tp world_size={}",
                config.num_attention_heads,
                self.world_size
            ));
        }
        if !config.num_key_value_heads.is_multiple_of(self.world_size) {
            return Err(anyhow::anyhow!(
                "num_key_value_heads={} not divisible by tp world_size={}",
                config.num_key_value_heads,
                self.world_size
            ));
        }
        if !config.intermediate_size.is_multiple_of(self.world_size) {
            return Err(anyhow::anyhow!(
                "intermediate_size={} not divisible by tp world_size={}",
                config.intermediate_size,
                self.world_size
            ));
        }
        Ok(())
    }

    pub(crate) fn shard_range(self, total: usize) -> (usize, usize) {
        let shard_len = total / self.world_size;
        (self.rank * shard_len, shard_len)
    }

    pub(crate) fn is_sharded(self) -> bool {
        self.world_size > 1
    }
}

#[cfg(test)]
mod tp_tests {
    use super::*;

    fn test_config() -> Config35 {
        Config35 {
            hidden_size: 2560,
            intermediate_size: 9216,
            num_hidden_layers: 32,
            vocab_size: 248_320,
            selection_vocab: 248_320,
            rms_norm_eps: 1e-6,
            eos_token_id: 151_645,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            linear_num_key_heads: 16,
            linear_key_head_dim: 128,
            linear_num_value_heads: 32,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            rope_theta: 10_000.0,
            rotary_dim: 64,
            max_position_embeddings: 262_144,
            tie_word_embeddings: true,
            layer_types: vec![LayerType::LinearAttention; 32],
        }
    }

    #[test]
    fn default_tensor_parallel_is_tp1() {
        let config = test_config();
        let tp = TensorParallelConfig::default();

        tp.validate_for(&config, true).unwrap();
        assert!(!tp.is_sharded());
        assert_eq!(tp.shard_range(config.full_attn_q_dim()), (0, 4096));
        assert_eq!(config.local_num_attention_heads(tp), 16);
        assert_eq!(config.local_num_key_value_heads(tp), 4);
        assert_eq!(config.local_intermediate_size(tp), 9216);
        assert_eq!(config.local_full_attn_q_dim(tp), 4096);
        assert_eq!(config.local_full_attn_kv_dim(tp), 1024);
        assert_eq!(config.local_full_attn_gated_q_dim(tp), 8192);
    }

    #[test]
    fn computes_tp2_dense_local_dimensions() {
        let config = test_config();
        let tp = TensorParallelConfig {
            rank: 1,
            world_size: 2,
        };

        tp.validate_for(&config, false).unwrap();
        assert!(tp.is_sharded());
        assert_eq!(tp.shard_range(config.full_attn_q_dim()), (2048, 2048));
        assert_eq!(config.local_num_attention_heads(tp), 8);
        assert_eq!(config.local_num_key_value_heads(tp), 2);
        assert_eq!(config.local_intermediate_size(tp), 4608);
        assert_eq!(config.local_full_attn_q_dim(tp), 2048);
        assert_eq!(config.local_full_attn_kv_dim(tp), 512);
        assert_eq!(config.local_full_attn_gated_q_dim(tp), 4096);
    }

    #[test]
    fn rejects_invalid_world_size_and_rank() {
        let config = test_config();

        let err = TensorParallelConfig {
            rank: 0,
            world_size: 0,
        }
        .validate_for(&config, false)
        .unwrap_err()
        .to_string();
        assert!(err.contains("world_size must be >= 1"));

        let err = TensorParallelConfig {
            rank: 2,
            world_size: 2,
        }
        .validate_for(&config, false)
        .unwrap_err()
        .to_string();
        assert!(err.contains("rank 2 must be < world_size 2"));
    }

    #[test]
    fn rejects_indivisible_dense_dimensions() {
        let tp = TensorParallelConfig {
            rank: 0,
            world_size: 3,
        };

        let mut config = test_config();
        let err = tp.validate_for(&config, false).unwrap_err().to_string();
        assert!(err.contains("num_attention_heads=16 not divisible"));

        config.num_attention_heads = 15;
        config.num_key_value_heads = 4;
        let err = tp.validate_for(&config, false).unwrap_err().to_string();
        assert!(err.contains("num_key_value_heads=4 not divisible"));

        config.num_key_value_heads = 3;
        config.intermediate_size = 9217;
        let err = tp.validate_for(&config, false).unwrap_err().to_string();
        assert!(err.contains("intermediate_size=9217 not divisible"));
    }

    #[test]
    fn rejects_tensor_parallel_cuda_graph_phase1() {
        let config = test_config();
        let tp = TensorParallelConfig {
            rank: 0,
            world_size: 2,
        };

        let err = tp.validate_for(&config, true).unwrap_err().to_string();
        assert!(err.contains("eager-only in Phase 1"));
    }

    #[test]
    fn phase1_does_not_require_linear_attention_divisibility() {
        let mut config = test_config();
        config.linear_num_key_heads = 17;
        config.linear_num_value_heads = 31;
        let tp = TensorParallelConfig {
            rank: 1,
            world_size: 2,
        };

        tp.validate_for(&config, false).unwrap();
    }
}

/// Schema kept identical to the pinned vLLM frontend; unread fields exist for
/// payload type-checking.
#[allow(dead_code)]
// The tokenizer_config schema is bool-heavy by design.
#[allow(clippy::struct_excessive_bools)]
#[derive(Deserialize)]
struct AddedTokenConfig {
    #[serde(default)]
    id: Option<u32>,
    content: String,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    special: bool,
}

#[derive(Deserialize)]
struct TokenizerJsonIds {
    model: TokenizerModelIds,
    #[serde(default)]
    added_tokens: Vec<AddedTokenConfig>,
}

#[derive(Deserialize)]
struct TokenizerModelIds {
    vocab: std::collections::HashMap<String, u32>,
}

#[derive(Deserialize)]
struct TokenizerConfigIds {
    #[serde(default)]
    added_tokens_decoder: std::collections::HashMap<String, AddedTokenConfig>,
}

/// Width of the frontend-decodable id space, mirroring the pinned frontend's
/// merge: `tokenizer.json` vocab and added_tokens (fatal on parse failure -
/// the frontend cannot serve without it) plus `tokenizer_config.json`
/// added_tokens_decoder (whole-file typed parse; failure drops all decoder
/// tokens with a warning, unparsable keys are skipped per entry). The ids
/// must form a dense prefix - a row-range selection bound cannot mask holes -
/// so a sparse id space fails the load instead of silently truncating the
/// output space.
pub(crate) fn tokenizer_effective_vocab(model_path: &str) -> Result<usize> {
    let path = format!("{}/tokenizer.json", model_path);
    let content =
        fs::read_to_string(&path).map_err(|e| anyhow::anyhow!("cannot read {path}: {e}"))?;
    let tj: TokenizerJsonIds =
        serde_json::from_str(&content).map_err(|e| anyhow::anyhow!("cannot parse {path}: {e}"))?;
    anyhow::ensure!(!tj.model.vocab.is_empty(), "{path} model.vocab is empty");
    let mut ids: HashSet<u32> = tj.model.vocab.into_values().collect();
    ids.extend(tj.added_tokens.iter().filter_map(|t| t.id));

    let config_path = format!("{}/tokenizer_config.json", model_path);
    if let Ok(text) = fs::read_to_string(&config_path) {
        match serde_json::from_str::<TokenizerConfigIds>(&text) {
            Ok(cfg) => ids.extend(
                cfg.added_tokens_decoder
                    .keys()
                    .filter_map(|k| k.parse::<u32>().ok()),
            ),
            Err(e) => warn!(
                "cannot parse {config_path}: {e}; skipping its added tokens like the frontend does"
            ),
        }
    }

    let width = ids.len();
    let max_id = *ids.iter().max().expect("vocab checked non-empty") as usize;
    anyhow::ensure!(
        max_id + 1 == width,
        "tokenizer id space is not dense (max id {max_id}, {width} distinct ids); \
         a row-range selection bound cannot mask holes"
    );
    Ok(width)
}

/// Identity check that `json` is a Qwen3.5 config; size and shape validation belong to the config loader.
pub(crate) fn probe_config_json(json: &Value) -> Result<()> {
    let model_type = json.get("model_type").and_then(Value::as_str).unwrap_or("");
    if model_type != "qwen3_5" {
        bail!("not a Qwen3.5 config: model_type={model_type}");
    }
    let architectures: Vec<&str> = json
        .get("architectures")
        .and_then(Value::as_array)
        .map(|arr| arr.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    ensure!(
        architectures.contains(&"Qwen3_5ForConditionalGeneration"),
        "Qwen3.5 architectures must contain Qwen3_5ForConditionalGeneration"
    );
    let text_model_type = json
        .get("text_config")
        .and_then(|tc| tc.get("model_type"))
        .and_then(Value::as_str)
        .unwrap_or("");
    ensure!(
        text_model_type == "qwen3_5_text",
        "Qwen3.5 text_config.model_type must be qwen3_5_text, got {text_model_type}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Config35;

    #[test]
    fn guard_accepts_48_value_heads() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{
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
        std::fs::write(dir.path().join("config.json"), json).unwrap();
        Config35::from_file(dir.path().to_str().unwrap()).expect("48 value heads must load");
    }

    #[test]
    fn guard_rejects_wide_linear_conv_decode_kernel() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{
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
    "linear_conv_kernel_dim": 5,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 48,
    "linear_value_head_dim": 128,
    "rope_parameters": { "rope_theta": 10000.0, "partial_rotary_factor": 0.25 },
    "eos_token_id": 0
  }
}"#;
        std::fs::write(dir.path().join("config.json"), json).unwrap();

        let err = Config35::from_file(dir.path().to_str().unwrap())
            .expect_err("wide conv decode kernels must be rejected");
        assert!(
            err.to_string().contains("linear conv decode kernels"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn effective_vocab_is_the_dense_decodable_width() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{
  "model": { "vocab": { "a": 0, "b": 1, "c": 2 } },
  "added_tokens": [ { "id": 3, "content": "<x>" } ]
}"#;
        std::fs::write(dir.path().join("tokenizer.json"), json).unwrap();
        let cfg = r#"{ "added_tokens_decoder": { "4": { "content": "<z>" }, "5": { "content": "<w>" }, "x": { "content": "<bad-key>" } } }"#;
        std::fs::write(dir.path().join("tokenizer_config.json"), cfg).unwrap();
        assert_eq!(
            super::tokenizer_effective_vocab(dir.path().to_str().unwrap()).unwrap(),
            6
        );
    }

    #[test]
    fn effective_vocab_fails_on_a_sparse_id_space() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{ "model": { "vocab": { "a": 0, "b": 1 } } }"#;
        std::fs::write(dir.path().join("tokenizer.json"), json).unwrap();
        let cfg = r#"{ "added_tokens_decoder": { "5": { "content": "<z>" } } }"#;
        std::fs::write(dir.path().join("tokenizer_config.json"), cfg).unwrap();
        assert!(super::tokenizer_effective_vocab(dir.path().to_str().unwrap()).is_err());
    }

    #[test]
    fn one_invalid_decoder_entry_drops_all_decoder_tokens() {
        let dir = tempfile::tempdir().unwrap();
        let json = r#"{ "model": { "vocab": { "a": 0, "b": 1 } } }"#;
        std::fs::write(dir.path().join("tokenizer.json"), json).unwrap();
        let cfg = r#"{ "added_tokens_decoder": { "2": { "content": "<z>" }, "3": { "content": "<w>", "special": "not-a-bool" } } }"#;
        std::fs::write(dir.path().join("tokenizer_config.json"), cfg).unwrap();
        assert_eq!(
            super::tokenizer_effective_vocab(dir.path().to_str().unwrap()).unwrap(),
            2
        );
    }
}
