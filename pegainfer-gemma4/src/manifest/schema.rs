//! The tensor contract a config implies.

use std::collections::HashMap;
use std::fmt;

use anyhow::Result;

use super::EXPECTED_DTYPE;
use crate::config::Gemma4Config;
use crate::config::LayerKind;

/// Every text tensor lives here, which is what keeps the modality skip list
/// from shadowing a required one.
const TEXT_PREFIX: &str = "model.language_model.";

pub(crate) struct Matrix2d {
    pub(crate) name: String,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

pub(crate) struct Vector1d {
    pub(crate) name: String,
    pub(crate) len: usize,
}

pub(crate) struct AttentionTensors {
    pub(crate) q_proj: Matrix2d,
    pub(crate) k_proj: Matrix2d,
    /// Absent on global layers, which the checkpoint ships without one.
    pub(crate) v_proj: Option<Matrix2d>,
    pub(crate) o_proj: Matrix2d,
    pub(crate) q_norm: Vector1d,
    pub(crate) k_norm: Vector1d,
}

pub(crate) struct MlpTensors {
    pub(crate) gate: Matrix2d,
    pub(crate) up: Matrix2d,
    pub(crate) down: Matrix2d,
}

pub(crate) struct LayerTensors {
    pub(crate) input_layernorm: Vector1d,
    pub(crate) post_attention_layernorm: Vector1d,
    pub(crate) pre_feedforward_layernorm: Vector1d,
    pub(crate) post_feedforward_layernorm: Vector1d,
    pub(crate) layer_scalar: Vector1d,
    pub(crate) attention: AttentionTensors,
    pub(crate) mlp: MlpTensors,
}

pub(crate) struct Manifest {
    pub(crate) embed_tokens: Matrix2d,
    pub(crate) norm: Vector1d,
    pub(crate) layers: Vec<LayerTensors>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ExpectedShape {
    Matrix { rows: usize, cols: usize },
    Vector { len: usize },
}

impl ExpectedShape {
    pub(super) fn matches(self, shape: &[usize]) -> bool {
        match self {
            Self::Matrix { rows, cols } => shape == [rows, cols],
            Self::Vector { len } => shape == [len],
        }
    }

    fn checked_bytes(self) -> Option<usize> {
        let elements = match self {
            Self::Matrix { rows, cols } => rows.checked_mul(cols)?,
            Self::Vector { len } => len,
        };
        elements.checked_mul(EXPECTED_DTYPE.bitsize() / 8)
    }
}

impl fmt::Display for ExpectedShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix { rows, cols } => write!(f, "[{rows}, {cols}]"),
            Self::Vector { len } => write!(f, "[{len}]"),
        }
    }
}

impl Matrix2d {
    fn new(name: String, rows: usize, cols: usize) -> Self {
        Self { name, rows, cols }
    }

    fn expected(&self) -> ExpectedShape {
        ExpectedShape::Matrix {
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl Vector1d {
    fn new(name: String, len: usize) -> Self {
        Self { name, len }
    }

    fn expected(&self) -> ExpectedShape {
        ExpectedShape::Vector { len: self.len }
    }
}

impl Manifest {
    pub(crate) fn from_config(config: &Gemma4Config) -> Result<Self> {
        anyhow::ensure!(
            !config.moe_enabled,
            "Gemma 4: this checkpoint enables the MoE block. Its decoder layers keep their dense \
             MLP and add routed experts, a router and three more norm sites on top; the loader \
             describes only the dense schema, so it cannot name what those layers also carry"
        );
        anyhow::ensure!(
            config.tie_word_embeddings,
            "Gemma 4: this checkpoint unties the LM head, which the loader has no tensor name for; \
             every published size ties it to the input embedding"
        );
        let hidden = config.hidden_size;
        let layers = config
            .layer_types
            .iter()
            .enumerate()
            .map(|(index, &kind)| LayerTensors::new(config, index, kind))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            embed_tokens: Matrix2d::new(
                format!("{TEXT_PREFIX}embed_tokens.weight"),
                config.vocab_size,
                hidden,
            ),
            norm: Vector1d::new(format!("{TEXT_PREFIX}norm.weight"), hidden),
            layers,
        })
    }

    pub(super) fn expected_shapes(&self) -> HashMap<&str, ExpectedShape> {
        fn matrix(m: &Matrix2d) -> (&str, ExpectedShape) {
            (m.name.as_str(), m.expected())
        }
        fn vector(v: &Vector1d) -> (&str, ExpectedShape) {
            (v.name.as_str(), v.expected())
        }
        let mut out = HashMap::new();
        out.extend([matrix(&self.embed_tokens), vector(&self.norm)]);
        for layer in &self.layers {
            out.extend([
                vector(&layer.input_layernorm),
                vector(&layer.post_attention_layernorm),
                vector(&layer.pre_feedforward_layernorm),
                vector(&layer.post_feedforward_layernorm),
                vector(&layer.layer_scalar),
                vector(&layer.attention.q_norm),
                vector(&layer.attention.k_norm),
                matrix(&layer.attention.q_proj),
                matrix(&layer.attention.k_proj),
                matrix(&layer.attention.o_proj),
                matrix(&layer.mlp.gate),
                matrix(&layer.mlp.up),
                matrix(&layer.mlp.down),
            ]);
            if let Some(v_proj) = &layer.attention.v_proj {
                out.insert(v_proj.name.as_str(), v_proj.expected());
            }
        }
        out
    }

    /// Before any allocator rounding.
    pub(crate) fn weight_bytes(&self) -> Result<usize> {
        self.expected_shapes()
            .values()
            .try_fold(0usize, |total, shape| {
                shape
                    .checked_bytes()
                    .and_then(|bytes| total.checked_add(bytes))
            })
            .ok_or_else(|| anyhow::anyhow!("Gemma 4: the manifest's total size overflows usize"))
    }
}

impl LayerTensors {
    fn new(config: &Gemma4Config, index: usize, kind: LayerKind) -> Result<Self> {
        let hidden = config.hidden_size;
        let prefix = format!("{TEXT_PREFIX}layers.{index}.");
        let (head_dim, kv_heads) = match kind {
            LayerKind::Sliding => (config.head_dim, config.num_key_value_heads),
            LayerKind::Global => (config.global_head_dim, config.num_global_key_value_heads),
        };
        // External input, and the probe pins only the head dims.
        let projection_dim = |heads: usize, what: &str| {
            heads.checked_mul(head_dim).ok_or_else(|| {
                anyhow::anyhow!(
                    "Gemma 4 layer {index}: {what} dimension {heads} x {head_dim} overflows usize"
                )
            })
        };
        let q_dim = projection_dim(config.num_attention_heads, "query")?;
        let kv_dim = projection_dim(kv_heads, "key/value")?;
        Ok(Self {
            input_layernorm: Vector1d::new(format!("{prefix}input_layernorm.weight"), hidden),
            post_attention_layernorm: Vector1d::new(
                format!("{prefix}post_attention_layernorm.weight"),
                hidden,
            ),
            pre_feedforward_layernorm: Vector1d::new(
                format!("{prefix}pre_feedforward_layernorm.weight"),
                hidden,
            ),
            post_feedforward_layernorm: Vector1d::new(
                format!("{prefix}post_feedforward_layernorm.weight"),
                hidden,
            ),
            layer_scalar: Vector1d::new(format!("{prefix}layer_scalar"), 1),
            attention: AttentionTensors {
                q_proj: Matrix2d::new(format!("{prefix}self_attn.q_proj.weight"), q_dim, hidden),
                k_proj: Matrix2d::new(format!("{prefix}self_attn.k_proj.weight"), kv_dim, hidden),
                v_proj: match kind {
                    LayerKind::Sliding => Some(Matrix2d::new(
                        format!("{prefix}self_attn.v_proj.weight"),
                        kv_dim,
                        hidden,
                    )),
                    LayerKind::Global => None,
                },
                o_proj: Matrix2d::new(format!("{prefix}self_attn.o_proj.weight"), hidden, q_dim),
                q_norm: Vector1d::new(format!("{prefix}self_attn.q_norm.weight"), head_dim),
                k_norm: Vector1d::new(format!("{prefix}self_attn.k_norm.weight"), head_dim),
            },
            mlp: MlpTensors {
                gate: Matrix2d::new(
                    format!("{prefix}mlp.gate_proj.weight"),
                    config.intermediate_size,
                    hidden,
                ),
                up: Matrix2d::new(
                    format!("{prefix}mlp.up_proj.weight"),
                    config.intermediate_size,
                    hidden,
                ),
                down: Matrix2d::new(
                    format!("{prefix}mlp.down_proj.weight"),
                    hidden,
                    config.intermediate_size,
                ),
            },
        })
    }
}

/// Two sliding layers and one global, at the published 12B dimensions.
#[cfg(test)]
pub(super) fn sample_config() -> Gemma4Config {
    Gemma4Config {
        hidden_size: 3840,
        intermediate_size: 15360,
        vocab_size: 262_144,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        num_global_key_value_heads: 1,
        head_dim: 256,
        global_head_dim: 512,
        layer_types: vec![LayerKind::Sliding, LayerKind::Sliding, LayerKind::Global],
        tie_word_embeddings: true,
        moe_enabled: false,
        rms_norm_eps: 1e-6,
        sliding_rope_theta: 10_000.0,
        sliding_window: 1024,
        max_position_embeddings: 262_144,
        global_rope_theta: 1_000_000.0,
        global_rotary_dim: 128,
        final_logit_softcapping: 30.0,
    }
}

#[cfg(test)]
mod tests {
    use super::sample_config as config;
    use super::*;

    fn rejection(config: &Gemma4Config) -> String {
        match Manifest::from_config(config) {
            Ok(_) => panic!("expected this config to be rejected"),
            Err(e) => e.to_string(),
        }
    }

    #[test]
    fn global_layers_have_no_v_proj_and_wider_heads() {
        let config = config();
        let manifest = Manifest::from_config(&config).unwrap();
        let sliding = &manifest.layers[0].attention;
        let global = &manifest.layers[2].attention;
        assert_eq!(sliding.v_proj.as_ref().unwrap().rows, 8 * 256);
        assert!(global.v_proj.is_none());
        assert_eq!(sliding.q_proj.rows, 16 * 256);
        assert_eq!(global.q_proj.rows, 16 * 512);
        assert_eq!(global.k_proj.rows, 512);
        assert_eq!(global.q_norm.len, 512);
    }

    #[test]
    fn moe_and_untied_configs_are_rejected_before_any_tensor_is_named() {
        let mut moe = config();
        moe.moe_enabled = true;
        let err = rejection(&moe);
        assert!(err.contains("MoE block"), "{err}");

        let mut untied = config();
        untied.tie_word_embeddings = false;
        let err = rejection(&untied);
        assert!(err.contains("unties the LM head"), "{err}");
    }

    #[test]
    fn dimensions_that_overflow_are_rejected_rather_than_wrapped() {
        let mut wide = config();
        wide.num_attention_heads = usize::MAX;
        let err = rejection(&wide);
        assert!(err.contains("overflows usize"), "{err}");

        let mut deep = config();
        deep.vocab_size = usize::MAX;
        let err = Manifest::from_config(&deep)
            .expect("per-tensor shapes are fine; only the total overflows")
            .weight_bytes()
            .expect_err("a manifest whose total overflows usize was accepted")
            .to_string();
        assert!(err.contains("overflows usize"), "{err}");
    }
}
