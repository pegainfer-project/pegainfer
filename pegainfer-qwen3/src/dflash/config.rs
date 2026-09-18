//! The native, full-vocabulary DFlash2 checkpoint contract.

use std::collections::BTreeMap;
use std::fs;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use serde::Deserialize;
use serde_json::Value;

/// Validated metadata for the released BF16, anchor-drop selector profile.
/// Inspection and selector execution do not enable the native backbone server.
#[derive(Clone, Debug)]
pub struct NativeDFlash2Config {
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) head_dim: usize,
    pub(crate) vocab_size: usize,
    pub(crate) num_target_layers: usize,
    pub(crate) target_layer_ids: Vec<usize>,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rope_theta: f64,
    pub(crate) block_size: usize,
    pub(crate) selector_rank: usize,
    pub(crate) selector_top_k: usize,
    pub(crate) conv_group_size: usize,
    pub(crate) conv_kernel_size: usize,
    pub(crate) sliding_window: Option<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DraftConfig {
    block_size: usize,
    conv_group_size: usize,
    conv_kernel_size: usize,
    mask_token_id: u32,
    selector_rank: usize,
    selector_top_k: usize,
    target_layer_ids: Vec<usize>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RopeConfig {
    rope_theta: f64,
    rope_type: String,
}

#[derive(Deserialize)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "DTO mirrors the published checkpoint schema"
)]
struct RawConfig {
    architectures: Vec<String>,
    model_type: String,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    vocab_size: usize,
    num_target_layers: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    hidden_act: String,
    attention_bias: bool,
    attention_dropout: f64,
    is_causal: bool,
    dtype: String,
    tie_word_embeddings: bool,
    dflash_config: DraftConfig,
    rope_parameters: RopeConfig,
    use_sliding_window: bool,
    sliding_window: Option<usize>,
    layer_types: Vec<String>,
    max_window_layers: usize,
}

/// Caller-supplied target/fixture identity and geometry. This is metadata
/// validation, not a proof of training provenance or a GPU head binding.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeTargetMetadata {
    pub model_id: String,
    pub revision: String,
    pub tokenizer_id: String,
    pub tokenizer_revision: String,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
}

impl NativeDFlash2Config {
    /// Recognize native evidence before the permissive legacy parser can drop it.
    pub(crate) fn is_native(json: &Value) -> bool {
        json.get("architectures")
            .and_then(Value::as_array)
            .is_some_and(|a| a.iter().any(|v| v == "DFlash2DraftModel"))
            || json.get("dflash_config").is_some_and(|d| {
                [
                    "selector_rank",
                    "selector_top_k",
                    "conv_group_size",
                    "conv_kernel_size",
                ]
                .iter()
                .any(|key| d.get(key).is_some())
            })
            || ["selector_rank", "selector_top_k", "speculators_config"]
                .iter()
                .any(|key| json.get(key).is_some())
    }

    pub fn from_file(model_path: &str) -> Result<Self> {
        let path = std::path::Path::new(model_path).join("config.json");
        let content = fs::read_to_string(&path)
            .with_context(|| format!("native DFlash2 inspect: reading {}", path.display()))?;
        let json = serde_json::from_str(&content).context("native DFlash2 config.json")?;
        Self::from_json(&json)
    }

    pub fn from_json(json: &Value) -> Result<Self> {
        // Permit ordinary HF provenance/training metadata, but never silently
        // discard a new field that could change the computation or token space.
        const FIELDS: &[&str] = &[
            "architectures",
            "model_type",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "num_target_layers",
            "max_position_embeddings",
            "rms_norm_eps",
            "hidden_act",
            "attention_bias",
            "attention_dropout",
            "is_causal",
            "dtype",
            "tie_word_embeddings",
            "dflash_config",
            "rope_parameters",
            "use_sliding_window",
            "sliding_window",
            "layer_types",
            "max_window_layers",
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
            "initializer_range",
            "transformers_version",
            "use_cache",
            "_name_or_path",
            "_commit_hash",
        ];

        let object = json
            .as_object()
            .context("native DFlash2 config must be an object")?;
        for name in object.keys() {
            ensure!(
                FIELDS.contains(&name.as_str()),
                "native DFlash2 config: unsupported field {name:?}"
            );
        }

        let raw: RawConfig =
            serde_json::from_value(json.clone()).context("native DFlash2 config schema")?;
        ensure!(
            raw.architectures == ["DFlash2DraftModel"] && raw.model_type == "qwen3",
            "native DFlash2 config: expected architectures=[DFlash2DraftModel], model_type=qwen3"
        );
        ensure!(
            raw.dtype == "bfloat16",
            "native DFlash2 dtype: expected bfloat16, got {}",
            raw.dtype
        );
        ensure!(
            raw.hidden_act == "silu"
                && !raw.attention_bias
                && raw.attention_dropout == 0.0
                && !raw.is_causal,
            "native DFlash2 requires silu, no attention bias/dropout, and is_causal=false"
        );
        ensure!(
            raw.rms_norm_eps.is_finite() && raw.rms_norm_eps > 0.0,
            "native DFlash2 rms_norm_eps must be finite and positive"
        );
        ensure!(
            raw.rope_parameters.rope_type == "default"
                && raw.rope_parameters.rope_theta.is_finite()
                && raw.rope_parameters.rope_theta > 0.0,
            "native DFlash2 rope_parameters requires default RoPE and finite positive rope_theta"
        );

        for (name, value) in [
            ("hidden_size", raw.hidden_size),
            ("intermediate_size", raw.intermediate_size),
            ("num_hidden_layers", raw.num_hidden_layers),
            ("num_attention_heads", raw.num_attention_heads),
            ("num_key_value_heads", raw.num_key_value_heads),
            ("head_dim", raw.head_dim),
            ("vocab_size", raw.vocab_size),
            ("num_target_layers", raw.num_target_layers),
            ("max_position_embeddings", raw.max_position_embeddings),
            ("selector_rank", raw.dflash_config.selector_rank),
            ("conv_group_size", raw.dflash_config.conv_group_size),
            ("conv_kernel_size", raw.dflash_config.conv_kernel_size),
        ] {
            ensure!(
                value > 0 && i32::try_from(value).is_ok(),
                "native DFlash2 {name}: expected 1..=i32::MAX, got {value}"
            );
        }

        ensure!(
            raw.num_attention_heads
                .is_multiple_of(raw.num_key_value_heads),
            "native DFlash2 num_attention_heads must be divisible by num_key_value_heads"
        );
        ensure!(
            raw.head_dim.is_multiple_of(2),
            "native DFlash2 head_dim must be even for full-head default RoPE"
        );
        ensure!(
            raw.hidden_size
                .is_multiple_of(raw.dflash_config.conv_group_size),
            "native DFlash2 hidden_size must be divisible by conv_group_size"
        );

        ensure!(
            raw.dflash_config.block_size >= 2
                && i32::try_from(raw.dflash_config.block_size).is_ok(),
            "native DFlash2 block_size must be in 2..=i32::MAX"
        );
        ensure!(
            raw.dflash_config.selector_top_k == 16 && raw.vocab_size >= 16,
            "native DFlash2 selector_top_k must be 16 and <= vocab_size"
        );
        ensure!(
            (raw.dflash_config.mask_token_id as usize) < raw.vocab_size,
            "native DFlash2 mask_token_id is outside vocab_size"
        );

        ensure!(
            !raw.dflash_config.target_layer_ids.is_empty()
                && raw
                    .dflash_config
                    .target_layer_ids
                    .iter()
                    .all(|&i| i < raw.num_target_layers)
                && raw
                    .dflash_config
                    .target_layer_ids
                    .windows(2)
                    .all(|w| w[0] < w[1]),
            "native DFlash2 target_layer_ids must be nonempty, strictly increasing, and inside num_target_layers"
        );

        ensure!(
            raw.layer_types.len() == raw.num_hidden_layers,
            "native DFlash2 layer_types length must equal num_hidden_layers"
        );
        ensure!(
            raw.max_window_layers <= raw.num_hidden_layers,
            "native DFlash2 max_window_layers exceeds num_hidden_layers"
        );
        for (index, kind) in raw.layer_types.iter().enumerate() {
            let expected = if raw.use_sliding_window && index < raw.max_window_layers {
                "sliding_attention"
            } else {
                "full_attention"
            };
            ensure!(
                kind == expected,
                "native DFlash2 layer_types[{index}]: expected {expected}, got {kind}"
            );
        }

        let sliding_window = if raw.use_sliding_window {
            let window = raw
                .sliding_window
                .context("native DFlash2 sliding_window is required")?;
            ensure!(
                window > 0 && window <= raw.max_position_embeddings,
                "native DFlash2 sliding_window must be positive and <= max_position_embeddings"
            );
            Some(window)
        } else {
            ensure!(
                raw.sliding_window.is_none(),
                "native DFlash2 sliding_window must be null when use_sliding_window=false"
            );
            None
        };

        // Both values are legal: this profile shares the target's two owners,
        // independently of whether the target internally ties those owners.
        let _ = raw.tie_word_embeddings;

        let config = Self {
            hidden_size: raw.hidden_size,
            intermediate_size: raw.intermediate_size,
            num_hidden_layers: raw.num_hidden_layers,
            num_attention_heads: raw.num_attention_heads,
            num_key_value_heads: raw.num_key_value_heads,
            head_dim: raw.head_dim,
            vocab_size: raw.vocab_size,
            num_target_layers: raw.num_target_layers,
            target_layer_ids: raw.dflash_config.target_layer_ids,
            max_position_embeddings: raw.max_position_embeddings,
            rope_theta: raw.rope_parameters.rope_theta,
            block_size: raw.dflash_config.block_size,
            selector_rank: raw.dflash_config.selector_rank,
            selector_top_k: raw.dflash_config.selector_top_k,
            conv_group_size: raw.dflash_config.conv_group_size,
            conv_kernel_size: raw.dflash_config.conv_kernel_size,
            sliding_window,
        };
        config.weight_bytes()?;

        Ok(config)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn selector_rank(&self) -> usize {
        self.selector_rank
    }

    pub fn candidate_count(&self) -> usize {
        self.selector_top_k
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_theta
    }

    /// A component checkpoint must never enter the legacy serving lane.
    pub fn validate_serving(&self) -> Result<()> {
        bail!(
            "native DFlash2 backbone execution is not supported: requires dynamic convolution (kernel {}, group {}), {} and native draft/verify integration; selector component inspection/execution only",
            self.conv_kernel_size,
            self.conv_group_size,
            if self.sliding_window.is_some() {
                "sliding-window attention"
            } else {
                "native attention"
            }
        )
    }

    pub fn validate_target(&self, target: &NativeTargetMetadata) -> Result<()> {
        for (name, value) in [
            ("model_id", &target.model_id),
            ("revision", &target.revision),
            ("tokenizer_id", &target.tokenizer_id),
            ("tokenizer_revision", &target.tokenizer_revision),
        ] {
            ensure!(
                !value.trim().is_empty() && value != "main" && value != "latest",
                "native DFlash2 target metadata: {name} must identify a pinned source"
            );
        }

        for (name, expected, actual) in [
            ("hidden_size", self.hidden_size, target.hidden_size),
            ("vocab_size", self.vocab_size, target.vocab_size),
            (
                "num_hidden_layers",
                self.num_target_layers,
                target.num_hidden_layers,
            ),
        ] {
            ensure!(
                expected == actual,
                "native DFlash2 target {name}: expected {expected}, got {actual}"
            );
        }

        // The released target is hybrid Qwen3.5-family (24/4 heads, dim 256),
        // while its draft uses Qwen3 attention (32/8 heads, dim 128). Shared
        // hidden/head ownership requires D/V compatibility, not equal backbones.
        ensure!(
            target.num_key_value_heads > 0
                && target.num_attention_heads > 0
                && target
                    .num_attention_heads
                    .is_multiple_of(target.num_key_value_heads)
                && target.head_dim > 0,
            "native DFlash2 target attention geometry must be positive with an integral GQA ratio"
        );
        ensure!(
            target.rope_theta.is_finite() && target.rope_theta > 0.0,
            "native DFlash2 target rope_theta must be finite and positive"
        );
        ensure!(
            target.max_position_embeddings > 0
                && target.max_position_embeddings <= self.max_position_embeddings,
            "native DFlash2 target max_position_embeddings must be in 1..={}",
            self.max_position_embeddings
        );

        Ok(())
    }

    fn root_tensors(&self) -> Result<BTreeMap<&'static str, Vec<usize>>> {
        let d = self.hidden_size;
        let r = self.selector_rank;

        Ok(BTreeMap::from([
            ("candidate_selector.hidden_projection.weight", vec![r, d]),
            (
                "candidate_selector.predecessor_codebook",
                vec![self.vocab_size, r],
            ),
            (
                "candidate_selector.successor_codebook",
                vec![self.vocab_size, r],
            ),
            (
                "fc.weight",
                vec![d, checked_product(&[self.target_layer_ids.len(), d])?],
            ),
            ("hidden_norm.weight", vec![d]),
            ("norm.weight", vec![d]),
        ]))
    }

    fn layer_tensors(&self) -> Result<BTreeMap<&'static str, Vec<usize>>> {
        let d = self.hidden_size;
        let q = checked_product(&[self.num_attention_heads, self.head_dim])?;
        let kv = checked_product(&[self.num_key_value_heads, self.head_dim])?;
        let conv = checked_product(&[2, self.conv_kernel_size, d / self.conv_group_size])?;

        Ok(BTreeMap::from([
            ("self_attn.q_proj.weight", vec![q, d]),
            ("self_attn.k_proj.weight", vec![kv, d]),
            ("self_attn.v_proj.weight", vec![kv, d]),
            ("self_attn.o_proj.weight", vec![d, q]),
            ("self_attn.q_norm.weight", vec![self.head_dim]),
            ("self_attn.k_norm.weight", vec![self.head_dim]),
            ("mlp.gate_proj.weight", vec![self.intermediate_size, d]),
            ("mlp.up_proj.weight", vec![self.intermediate_size, d]),
            ("mlp.down_proj.weight", vec![d, self.intermediate_size]),
            ("input_layernorm.weight", vec![d]),
            ("post_attention_layernorm.weight", vec![d]),
            (
                "attention_conv.base_kernel",
                vec![2, self.conv_kernel_size, d],
            ),
            ("attention_conv.kernel_projection.weight", vec![conv, d]),
            ("mlp_conv.base_kernel", vec![2, self.conv_kernel_size, d]),
            ("mlp_conv.kernel_projection.weight", vec![conv, d]),
        ]))
    }

    pub(crate) fn expected_tensors(&self) -> Result<BTreeMap<String, Vec<usize>>> {
        let mut tensors: BTreeMap<_, _> = self
            .root_tensors()?
            .into_iter()
            .map(|(name, shape)| (name.to_owned(), shape))
            .collect();

        let layer = self.layer_tensors()?;
        for i in 0..self.num_hidden_layers {
            for (suffix, shape) in &layer {
                tensors.insert(format!("layers.{i}.{suffix}"), shape.clone());
            }
        }

        Ok(tensors)
    }

    pub fn selector_weight_bytes(&self) -> Result<usize> {
        let tensors = self.root_tensors()?;
        weight_bytes(
            tensors
                .iter()
                .filter(|(name, _)| name.starts_with("candidate_selector."))
                .map(|(_, shape)| shape),
        )
    }

    /// Complete checkpoint payload; excludes the separately owned target head.
    pub fn weight_bytes(&self) -> Result<usize> {
        let root_bytes = weight_bytes(self.root_tensors()?.values())?;
        let layer_bytes = weight_bytes(self.layer_tensors()?.values())?;
        checked_product(&[self.num_hidden_layers, layer_bytes])?
            .checked_add(root_bytes)
            .context("native DFlash2 checkpoint byte size overflow")
    }
}

fn checked_product(dimensions: &[usize]) -> Result<usize> {
    dimensions.iter().try_fold(1usize, |product, &dim| {
        product
            .checked_mul(dim)
            .context("native DFlash2 tensor shape overflow")
    })
}

fn weight_bytes<'a>(mut shapes: impl Iterator<Item = &'a Vec<usize>>) -> Result<usize> {
    shapes.try_fold(0usize, |sum, shape| {
        let bytes = checked_product(shape)?
            .checked_mul(2)
            .context("native DFlash2 tensor byte size overflow")?;

        sum.checked_add(bytes)
            .context("native DFlash2 checkpoint byte size overflow")
    })
}
