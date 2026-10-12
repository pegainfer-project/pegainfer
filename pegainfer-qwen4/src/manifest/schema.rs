//! The expected tensor set, derived from the validated config.

use std::collections::BTreeMap;
use std::fmt;

use safetensors::Dtype;

use super::EXPECTED_DTYPE;
use super::INTEGER_INDEX_DTYPE;
use crate::config::Config;
use crate::config::LayerKind;

/// Every text tensor carries this prefix — except `lm_head.weight`, which the
/// checkpoint stores at the top level. The prefix is what makes the skip set
/// safe: a skipped tensor can never be a required one.
const TEXT_PREFIX: &str = "model.language_model.";

/// The expected shape of one tensor. Rank matters, so the depthwise conv and the
/// fused expert stacks get their own variants rather than being flattened.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpectedShape {
    Matrix {
        rows: usize,
        cols: usize,
    },
    Vector {
        len: usize,
    },
    /// Depthwise conv weight, stored as `[channels, 1, kernel]`.
    Conv1d {
        channels: usize,
        kernel: usize,
    },
    /// Fused expert stack, stored as `[experts, rows, cols]` and — unlike every
    /// other tensor here — carrying **no `.weight` suffix**.
    ExpertStack {
        experts: usize,
        rows: usize,
        cols: usize,
    },
}

impl ExpectedShape {
    pub(crate) fn matches(self, shape: &[usize]) -> bool {
        match self {
            Self::Matrix { rows, cols } => shape == [rows, cols],
            Self::Vector { len } => shape == [len],
            Self::Conv1d { channels, kernel } => shape == [channels, 1, kernel],
            Self::ExpertStack {
                experts,
                rows,
                cols,
            } => shape == [experts, rows, cols],
        }
    }

    pub(crate) fn elements(self) -> usize {
        match self {
            Self::Matrix { rows, cols } => rows * cols,
            Self::Vector { len } => len,
            Self::Conv1d { channels, kernel } => channels * kernel,
            Self::ExpertStack {
                experts,
                rows,
                cols,
            } => experts * rows * cols,
        }
    }
}

impl fmt::Display for ExpectedShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matrix { rows, cols } => write!(f, "[{rows}, {cols}]"),
            Self::Vector { len } => write!(f, "[{len}]"),
            Self::Conv1d { channels, kernel } => write!(f, "[{channels}, 1, {kernel}]"),
            Self::ExpertStack {
                experts,
                rows,
                cols,
            } => write!(f, "[{experts}, {rows}, {cols}]"),
        }
    }
}

/// The tensor contract: every required text tensor, with its shape and dtype.
#[derive(Debug)]
pub(crate) struct Manifest {
    expected: BTreeMap<String, (ExpectedShape, Dtype)>,
}

impl Manifest {
    pub(crate) fn from_config(config: &Config) -> Self {
        let mut expected: BTreeMap<String, (ExpectedShape, Dtype)> = BTreeMap::new();
        let mut put = |name: String, shape: ExpectedShape, dtype: Dtype| {
            let previous = expected.insert(name.clone(), (shape, dtype));
            debug_assert!(previous.is_none(), "{name} derived twice");
        };

        let hidden = config.hidden_size();
        let vocab = config.text.vocab_size;

        // ---- bookend -------------------------------------------------------
        // Untied: both ends are real tensors. `lm_head.weight` sits at the top
        // level, outside the text prefix.
        put(
            "lm_head.weight".to_string(),
            ExpectedShape::Matrix {
                rows: vocab,
                cols: hidden,
            },
            EXPECTED_DTYPE,
        );
        put(
            format!("{TEXT_PREFIX}embed_tokens.weight"),
            ExpectedShape::Matrix {
                rows: vocab,
                cols: hidden,
            },
            EXPECTED_DTYPE,
        );
        // There is no `model.language_model.norm.weight`: the readout is the
        // hyper-connection mixer, which has no injection weight.
        hyper_connection(
            &mut put,
            &format!("{TEXT_PREFIX}hyper_connection_mixer"),
            config,
            false,
        );

        // ---- per layer -----------------------------------------------------
        // Exactly one PLE layer is modelled (`Config::ple_layer`, already
        // converted from the 1-indexed `ple_layer_ids`).
        let ple_layer = config.ple_layer;
        for (index, kind) in config.layer_kinds.iter().enumerate() {
            let layer = format!("{TEXT_PREFIX}layers.{index}");
            hyper_connection(
                &mut put,
                &format!("{layer}.attn_hyper_connection"),
                config,
                true,
            );
            hyper_connection(
                &mut put,
                &format!("{layer}.mlp_hyper_connection"),
                config,
                true,
            );
            moe(&mut put, &layer, config);
            match kind {
                LayerKind::LinearAttention => gated_delta_net(&mut put, &layer, config),
                LayerKind::FullAttention => sparse_attention(&mut put, &layer, config),
            }
            if index == ple_layer {
                ple(&mut put, &layer, config);
            }
        }

        Self { expected }
    }
    pub(crate) fn expected_tensors(&self) -> &BTreeMap<String, (ExpectedShape, Dtype)> {
        &self.expected
    }

    /// Stored bytes of the required text tensors, at each tensor's own dtype.
    pub(crate) fn weight_bytes(&self) -> u128 {
        self.expected
            .values()
            .map(|(shape, dtype)| shape.elements() as u128 * dtype_size(*dtype) as u128)
            .sum()
    }
}

fn dtype_size(dtype: Dtype) -> usize {
    match dtype {
        Dtype::BF16 | Dtype::F16 => 2,
        Dtype::F32 | Dtype::I32 => 4,
        Dtype::F64 | Dtype::I64 => 8,
        Dtype::U8 | Dtype::I8 | Dtype::BOOL => 1,
        _ => 0,
    }
}

/// One hyper-connection instance. `with_injection` distinguishes the two
/// per-layer instances (which scatter a sublayer's output back into the branches)
/// from the top-level mixer, which is the readout and carries no injection weight.
fn hyper_connection(
    put: &mut impl FnMut(String, ExpectedShape, Dtype),
    prefix: &str,
    config: &Config,
    with_injection: bool,
) {
    let hc_hidden = config.hc_hidden_size();
    let lowrank = config.text.hc_lowrank;
    put(
        format!("{prefix}.hc_norm.weight"),
        ExpectedShape::Vector { len: hc_hidden },
        EXPECTED_DTYPE,
    );
    put(
        format!("{prefix}.input_mix_weight_down.weight"),
        ExpectedShape::Matrix {
            rows: lowrank,
            cols: hc_hidden,
        },
        EXPECTED_DTYPE,
    );
    put(
        format!("{prefix}.input_mix_weight_up.weight"),
        ExpectedShape::Matrix {
            rows: hc_hidden,
            cols: lowrank,
        },
        EXPECTED_DTYPE,
    );
    if with_injection {
        put(
            format!("{prefix}.block_inject_weight.weight"),
            ExpectedShape::Matrix {
                rows: config.text.hc_count,
                cols: hc_hidden,
            },
            EXPECTED_DTYPE,
        );
    }
}

/// Every layer is MoE; there is no dense variant to defer behind.
fn moe(put: &mut impl FnMut(String, ExpectedShape, Dtype), layer: &str, config: &Config) {
    let text = &config.text;
    let experts = text.num_experts;
    let hidden = text.hidden_size;
    let intermediate = text.moe_intermediate_size;
    let shared = text.shared_expert_intermediate_size;
    // gate and up are fused into one stack, hence 2x the expert width.
    put(
        format!("{layer}.mlp.experts.gate_up_proj"),
        ExpectedShape::ExpertStack {
            experts,
            rows: 2 * intermediate,
            cols: hidden,
        },
        EXPECTED_DTYPE,
    );
    put(
        format!("{layer}.mlp.experts.down_proj"),
        ExpectedShape::ExpertStack {
            experts,
            rows: hidden,
            cols: intermediate,
        },
        EXPECTED_DTYPE,
    );
    put(
        format!("{layer}.mlp.gate.weight"),
        ExpectedShape::Matrix {
            rows: experts,
            cols: hidden,
        },
        EXPECTED_DTYPE,
    );
    for name in ["gate_proj", "up_proj"] {
        put(
            format!("{layer}.mlp.shared_expert.{name}.weight"),
            ExpectedShape::Matrix {
                rows: shared,
                cols: hidden,
            },
            EXPECTED_DTYPE,
        );
    }
    put(
        format!("{layer}.mlp.shared_expert.down_proj.weight"),
        ExpectedShape::Matrix {
            rows: hidden,
            cols: shared,
        },
        EXPECTED_DTYPE,
    );
    put(
        format!("{layer}.mlp.shared_expert_gate.weight"),
        ExpectedShape::Matrix {
            rows: 1,
            cols: hidden,
        },
        EXPECTED_DTYPE,
    );
}

fn gated_delta_net(
    put: &mut impl FnMut(String, ExpectedShape, Dtype),
    layer: &str,
    config: &Config,
) {
    let text = &config.text;
    let prefix = format!("{layer}.linear_attn");
    let hidden = text.hidden_size;
    let value_heads = text.linear_num_value_heads;
    // Fused Q|K|V: key_dim + key_dim + value_dim.
    put(
        format!("{prefix}.in_proj_qkv.weight"),
        ExpectedShape::Matrix {
            rows: config.gdn_conv_dim(),
            cols: hidden,
        },
        EXPECTED_DTYPE,
    );
    put(
        format!("{prefix}.in_proj_z.weight"),
        ExpectedShape::Matrix {
            rows: config.gdn_value_dim(),
            cols: hidden,
        },
        EXPECTED_DTYPE,
    );
    for name in ["a", "b"] {
        put(
            format!("{prefix}.in_proj_{name}.weight"),
            ExpectedShape::Matrix {
                rows: value_heads,
                cols: hidden,
            },
            EXPECTED_DTYPE,
        );
    }
    put(
        format!("{prefix}.conv1d.weight"),
        ExpectedShape::Conv1d {
            channels: config.gdn_conv_dim(),
            kernel: text.linear_conv_kernel_dim,
        },
        EXPECTED_DTYPE,
    );
    for name in ["A_log", "dt_bias"] {
        put(
            format!("{prefix}.{name}"),
            ExpectedShape::Vector { len: value_heads },
            EXPECTED_DTYPE,
        );
    }
    put(
        format!("{prefix}.norm.weight"),
        ExpectedShape::Vector {
            len: text.linear_value_head_dim,
        },
        EXPECTED_DTYPE,
    );
    put(
        format!("{prefix}.out_proj.weight"),
        ExpectedShape::Matrix {
            rows: hidden,
            cols: config.gdn_value_dim(),
        },
        EXPECTED_DTYPE,
    );
}

fn sparse_attention(
    put: &mut impl FnMut(String, ExpectedShape, Dtype),
    layer: &str,
    config: &Config,
) {
    let text = &config.text;
    let prefix = format!("{layer}.self_attn");
    let hidden = text.hidden_size;
    let head_dim = text.head_dim;
    let query_dim = text.num_attention_heads * head_dim;
    let kv_dim = text.num_key_value_heads * head_dim;
    // Q carries the output gate fused, so it is twice the query width.
    put(
        format!("{prefix}.q_proj.weight"),
        ExpectedShape::Matrix {
            rows: 2 * query_dim,
            cols: hidden,
        },
        EXPECTED_DTYPE,
    );
    for name in ["k_proj", "v_proj"] {
        put(
            format!("{prefix}.{name}.weight"),
            ExpectedShape::Matrix {
                rows: kv_dim,
                cols: hidden,
            },
            EXPECTED_DTYPE,
        );
    }
    put(
        format!("{prefix}.o_proj.weight"),
        ExpectedShape::Matrix {
            rows: hidden,
            cols: query_dim,
        },
        EXPECTED_DTYPE,
    );
    for name in ["q_norm", "k_norm"] {
        put(
            format!("{prefix}.{name}.weight"),
            ExpectedShape::Vector { len: head_dim },
            EXPECTED_DTYPE,
        );
    }
    // The indexer is a second, differently-shaped attention with its own fused
    // Q|K projection and its own layernorms.
    let indexer_dim = (text.indexer_n_heads + text.indexer_kv_heads) * text.indexer_head_dim;
    put(
        format!("{prefix}.indexer.index_qk_proj.weight"),
        ExpectedShape::Matrix {
            rows: indexer_dim,
            cols: hidden,
        },
        EXPECTED_DTYPE,
    );
    for name in ["q_layernorm", "k_layernorm"] {
        put(
            format!("{prefix}.indexer.{name}.weight"),
            ExpectedShape::Vector {
                len: text.indexer_head_dim,
            },
            EXPECTED_DTYPE,
        );
    }
}

/// The n-gram block. Mounted on exactly the layers `ple_layer_ids` names,
/// converted from 1-indexed by [`Config::ple_tensor_layers`].
fn ple(put: &mut impl FnMut(String, ExpectedShape, Dtype), layer: &str, config: &Config) {
    let text = &config.text;
    let ngram = &config.ngram;
    let prefix = format!("{layer}.ple");
    let hc_hidden = config.hc_hidden_size();
    // The block queries the hyper-connection stream, so its key side is hc-wide
    // while its value side is only hidden-wide.
    put(
        format!("{prefix}.key_proj.weight"),
        ExpectedShape::Matrix {
            rows: hc_hidden,
            cols: text.ple_embed_dim,
        },
        EXPECTED_DTYPE,
    );
    put(
        format!("{prefix}.value_proj.weight"),
        ExpectedShape::Matrix {
            rows: text.hidden_size,
            cols: text.ple_embed_dim,
        },
        EXPECTED_DTYPE,
    );
    // Dilated by `ngram_size` at runtime, but the weight tensor is rank-3 with the
    // same layout as the GDN conv — the dilation is not visible in the shape.
    put(
        format!("{prefix}.conv1d.weight"),
        ExpectedShape::Conv1d {
            channels: hc_hidden,
            kernel: text.ple_conv_kernel_size,
        },
        EXPECTED_DTYPE,
    );
    for name in ["norm_key", "norm_query", "norm_conv"] {
        put(
            format!("{prefix}.{name}.weight"),
            ExpectedShape::Vector { len: hc_hidden },
            EXPECTED_DTYPE,
        );
    }
    let embedding = format!("{prefix}.ple_embedding");
    put(
        format!("{embedding}.layer_multipliers"),
        ExpectedShape::Vector {
            len: text.ngram_size,
        },
        INTEGER_INDEX_DTYPE,
    );
    for name in ["ngram_heads_offsets", "ngram_heads_vocab_sizes"] {
        put(
            format!("{embedding}.{name}"),
            ExpectedShape::Vector { len: ngram.heads },
            INTEGER_INDEX_DTYPE,
        );
    }
    for shard in 0..ngram.shards {
        put(
            format!("{embedding}.ngram_embedding.shard_{shard}.weight"),
            ExpectedShape::Matrix {
                rows: ngram.rows_per_shard,
                cols: ngram.head_dim,
            },
            EXPECTED_DTYPE,
        );
    }
}
