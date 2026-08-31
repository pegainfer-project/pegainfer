//! The tensor contract a config implies.

use std::collections::HashMap;
use std::fmt;

use anyhow::Result;
use safetensors::Dtype;

use super::EXPECTED_DTYPE;
use crate::config::Gemma4Config;
use crate::config::LayerKind;
use crate::config::MoeConfig;

/// Two NVFP4 values share a byte.
const FP4_PER_BYTE: usize = 2;
/// One block scale per this many values along the reduction axis.
const FP4_GROUP: usize = 16;

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

/// A tensor the checkpoint stores in something other than the model dtype.
/// The dense schema implies bf16 everywhere; the routed experts do not, so
/// they carry their own.
pub(crate) struct TypedTensor {
    pub(crate) name: String,
    pub(crate) shape: ExpectedShape,
    pub(crate) dtype: Dtype,
}

/// One projection of one expert, stored as NVFP4: packed values, an e4m3
/// block scale per [`FP4_GROUP`] of them along the reduction axis, a
/// tensor-level scale, and the activation scale a W4A4 kernel would consume.
pub(crate) struct QuantMatrix {
    pub(crate) weight: TypedTensor,
    pub(crate) weight_scale: TypedTensor,
    pub(crate) weight_scale_2: TypedTensor,
    pub(crate) input_scale: TypedTensor,
}

pub(crate) struct ExpertTensors {
    pub(crate) gate: QuantMatrix,
    pub(crate) up: QuantMatrix,
    pub(crate) down: QuantMatrix,
}

pub(crate) struct RouterTensors {
    pub(crate) proj: Matrix2d,
    pub(crate) scale: Vector1d,
    pub(crate) per_expert_scale: Vector1d,
}

/// What a routed layer carries beyond the dense schema. The dense MLP stays
/// where it is: this size keeps it and adds these alongside.
pub(crate) struct MoeTensors {
    pub(crate) pre_feedforward_layernorm_2: Vector1d,
    pub(crate) post_feedforward_layernorm_1: Vector1d,
    pub(crate) post_feedforward_layernorm_2: Vector1d,
    pub(crate) router: RouterTensors,
    pub(crate) experts: Vec<ExpertTensors>,
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
    /// Present only on the routing size.
    pub(crate) moe: Option<MoeTensors>,
}

pub(crate) struct Manifest {
    pub(crate) embed_tokens: Matrix2d,
    pub(crate) norm: Vector1d,
    pub(crate) layers: Vec<LayerTensors>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ExpectedShape {
    Matrix {
        rows: usize,
        cols: usize,
    },
    Vector {
        len: usize,
    },
    /// Rank zero, which the quantization scales are stored as.
    Scalar,
}

impl ExpectedShape {
    pub(crate) fn matches(self, shape: &[usize]) -> bool {
        match self {
            Self::Matrix { rows, cols } => shape == [rows, cols],
            Self::Vector { len } => shape == [len],
            Self::Scalar => shape.is_empty(),
        }
    }

    fn checked_bytes(self, dtype: Dtype) -> Option<usize> {
        let elements = match self {
            Self::Matrix { rows, cols } => rows.checked_mul(cols)?,
            Self::Vector { len } => len,
            Self::Scalar => 1,
        };
        elements.checked_mul(dtype.bitsize() / 8)
    }
}

impl fmt::Display for ExpectedShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix { rows, cols } => write!(f, "[{rows}, {cols}]"),
            Self::Vector { len } => write!(f, "[{len}]"),
            Self::Scalar => write!(f, "[]"),
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

    pub(super) fn expected_tensors(&self) -> HashMap<&str, (ExpectedShape, Dtype)> {
        fn matrix(m: &Matrix2d) -> (&str, (ExpectedShape, Dtype)) {
            (m.name.as_str(), (m.expected(), EXPECTED_DTYPE))
        }
        fn vector(v: &Vector1d) -> (&str, (ExpectedShape, Dtype)) {
            (v.name.as_str(), (v.expected(), EXPECTED_DTYPE))
        }
        fn typed(t: &TypedTensor) -> (&str, (ExpectedShape, Dtype)) {
            (t.name.as_str(), (t.shape, t.dtype))
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
                out.extend([matrix(v_proj)]);
            }
            let Some(moe) = &layer.moe else {
                continue;
            };
            out.extend([
                vector(&moe.pre_feedforward_layernorm_2),
                vector(&moe.post_feedforward_layernorm_1),
                vector(&moe.post_feedforward_layernorm_2),
                matrix(&moe.router.proj),
                vector(&moe.router.scale),
                vector(&moe.router.per_expert_scale),
            ]);
            for expert in &moe.experts {
                for projection in [&expert.gate, &expert.up, &expert.down] {
                    out.extend(projection.tensors().map(typed));
                }
            }
        }
        out
    }

    /// Before any allocator rounding.
    pub(crate) fn weight_bytes(&self) -> Result<usize> {
        self.expected_tensors()
            .values()
            .try_fold(0usize, |total, (shape, dtype)| {
                shape
                    .checked_bytes(*dtype)
                    .and_then(|bytes| total.checked_add(bytes))
            })
            .ok_or_else(|| anyhow::anyhow!("Gemma 4: the manifest's total size overflows usize"))
    }
}

impl QuantMatrix {
    /// `reduction` is the axis the block scales run along, in logical values.
    fn new(prefix: &str, rows: usize, reduction: usize) -> Result<Self> {
        anyhow::ensure!(
            reduction.is_multiple_of(FP4_GROUP),
            "Gemma 4: '{prefix}' reduces over {reduction} values, which is not a whole number of \
             {FP4_GROUP}-value scale blocks, so the checkpoint's scale shape cannot be derived"
        );
        let packed = ExpectedShape::Matrix {
            rows,
            cols: reduction / FP4_PER_BYTE,
        };
        let scales = ExpectedShape::Matrix {
            rows,
            cols: reduction / FP4_GROUP,
        };
        Ok(Self {
            weight: TypedTensor {
                name: format!("{prefix}.weight"),
                shape: packed,
                dtype: Dtype::U8,
            },
            weight_scale: TypedTensor {
                name: format!("{prefix}.weight_scale"),
                shape: scales,
                dtype: Dtype::F8_E4M3,
            },
            weight_scale_2: TypedTensor {
                name: format!("{prefix}.weight_scale_2"),
                shape: ExpectedShape::Scalar,
                dtype: Dtype::F32,
            },
            input_scale: TypedTensor {
                name: format!("{prefix}.input_scale"),
                shape: ExpectedShape::Scalar,
                dtype: Dtype::F32,
            },
        })
    }

    /// Rows, and the logical values each row holds before packing.
    pub(crate) fn geometry(&self) -> Result<(usize, usize)> {
        match self.weight.shape {
            ExpectedShape::Matrix { rows, cols } => Ok((rows, cols * FP4_PER_BYTE)),
            other => anyhow::bail!(
                "Gemma 4: '{}' is {other}, which is not a packed matrix",
                self.weight.name
            ),
        }
    }

    fn tensors(&self) -> [&TypedTensor; 4] {
        [
            &self.weight,
            &self.weight_scale,
            &self.weight_scale_2,
            &self.input_scale,
        ]
    }
}

impl MoeTensors {
    fn new(prefix: &str, hidden: usize, moe: &MoeConfig) -> Result<Self> {
        let width = moe.intermediate_size;
        let experts = (0..moe.num_experts)
            .map(|expert| {
                let at = format!("{prefix}experts.{expert}");
                Ok(ExpertTensors {
                    gate: QuantMatrix::new(&format!("{at}.gate_proj"), width, hidden)?,
                    up: QuantMatrix::new(&format!("{at}.up_proj"), width, hidden)?,
                    down: QuantMatrix::new(&format!("{at}.down_proj"), hidden, width)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            pre_feedforward_layernorm_2: Vector1d::new(
                format!("{prefix}pre_feedforward_layernorm_2.weight"),
                hidden,
            ),
            post_feedforward_layernorm_1: Vector1d::new(
                format!("{prefix}post_feedforward_layernorm_1.weight"),
                hidden,
            ),
            post_feedforward_layernorm_2: Vector1d::new(
                format!("{prefix}post_feedforward_layernorm_2.weight"),
                hidden,
            ),
            router: RouterTensors {
                proj: Matrix2d::new(
                    format!("{prefix}router.proj.weight"),
                    moe.num_experts,
                    hidden,
                ),
                scale: Vector1d::new(format!("{prefix}router.scale"), hidden),
                per_expert_scale: Vector1d::new(
                    format!("{prefix}router.per_expert_scale"),
                    moe.num_experts,
                ),
            },
            experts,
        })
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
            moe: config
                .moe
                .map(|moe| MoeTensors::new(&prefix, hidden, &moe))
                .transpose()?,
        })
    }
}

/// Two sliding layers and one global, at the published 12B dimensions.
#[cfg(test)]
pub(crate) fn sample_config() -> Gemma4Config {
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
        moe: None,
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

    /// The published 26B dimensions, so the shapes below are the ones its
    /// checkpoint actually carries.
    fn moe_config() -> Gemma4Config {
        let mut config = config();
        config.hidden_size = 2816;
        config.intermediate_size = 2112;
        config.moe = Some(MoeConfig {
            num_experts: 128,
            top_k: 8,
            intermediate_size: 704,
        });
        config
    }

    #[test]
    fn routed_layers_name_their_experts_at_the_packed_shapes() {
        let manifest = Manifest::from_config(&moe_config()).unwrap();
        let moe = manifest.layers[0].moe.as_ref().expect("layer 0 routes");
        assert_eq!(moe.experts.len(), 128);
        assert_eq!(
            moe.router.proj.name,
            "model.language_model.layers.0.router.proj.weight"
        );
        assert_eq!((moe.router.proj.rows, moe.router.proj.cols), (128, 2816));
        assert_eq!(moe.router.per_expert_scale.len, 128);
        assert_eq!(moe.router.scale.len, 2816);

        let gate = &moe.experts[0].gate;
        assert_eq!(
            gate.weight.name,
            "model.language_model.layers.0.experts.0.gate_proj.weight"
        );
        // Two values to a byte, one e4m3 scale to sixteen.
        assert_eq!(
            gate.weight.shape,
            ExpectedShape::Matrix {
                rows: 704,
                cols: 1408
            }
        );
        assert_eq!(gate.weight.dtype, Dtype::U8);
        assert_eq!(
            gate.weight_scale.shape,
            ExpectedShape::Matrix {
                rows: 704,
                cols: 176
            }
        );
        assert_eq!(gate.weight_scale.dtype, Dtype::F8_E4M3);
        assert_eq!(gate.weight_scale_2.shape, ExpectedShape::Scalar);
        assert_eq!(gate.weight_scale_2.dtype, Dtype::F32);

        // down reduces over the expert width instead of the hidden size.
        let down = &moe.experts[0].down;
        assert_eq!(
            down.weight.shape,
            ExpectedShape::Matrix {
                rows: 2816,
                cols: 352
            }
        );
        assert_eq!(
            down.weight_scale.shape,
            ExpectedShape::Matrix {
                rows: 2816,
                cols: 44
            }
        );

        // The dense MLP stays beside the experts, unquantized.
        let dense = &manifest.layers[0].mlp;
        assert_eq!((dense.gate.rows, dense.gate.cols), (2112, 2816));
    }

    #[test]
    fn a_reduction_that_is_not_whole_scale_blocks_is_refused() {
        let mut ragged = moe_config();
        ragged.hidden_size = 2810;
        let err = rejection(&ragged);
        assert!(err.contains("scale blocks"), "{err}");
    }

    #[test]
    fn an_untied_config_is_rejected_before_any_tensor_is_named() {
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
