//! The expected tensor set, derived from the validated config.

use std::collections::BTreeMap;
use std::fmt;

use safetensors::Dtype;

use super::EXPECTED_DTYPE;
use super::INTEGER_INDEX_DTYPE;
use crate::config::Config;
use crate::config::LayerKind;

/// Every text tensor carries this prefix, except `lm_head.weight` at the top level. The
/// prefix is what makes the skip set safe: a skipped tensor can never be a required one.
const TEXT_PREFIX: &str = "model.language_model.";

/// The expected shape of one tensor, rank included.
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
    /// Fused expert stack, stored as `[experts, rows, cols]`, with **no `.weight` suffix**.
    ExpertStack {
        experts: usize,
        rows: usize,
        cols: usize,
    },
}

impl ExpectedShape {
    /// The dimensions a shard header must state.
    pub(crate) fn dims(self) -> Vec<usize> {
        match self {
            Self::Matrix { rows, cols } => vec![rows, cols],
            Self::Vector { len } => vec![len],
            Self::Conv1d { channels, kernel } => vec![channels, 1, kernel],
            Self::ExpertStack {
                experts,
                rows,
                cols,
            } => vec![experts, rows, cols],
        }
    }
}

impl fmt::Display for ExpectedShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.dims())
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
            assert!(previous.is_none(), "{name} derived twice");
        };

        let hidden = config.text.hidden_size;
        let vocab = config.text.vocab_size;

        // Untied: `lm_head.weight` sits at the top level, outside the text prefix.
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
        // No `model.language_model.norm.weight`: the readout is the hyper-connection mixer.
        hyper_connection(
            &mut put,
            &format!("{TEXT_PREFIX}hyper_connection_mixer"),
            config,
            false,
        );

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
            if index == config.ple_layer {
                ple(&mut put, &layer, config);
            }
        }

        Self { expected }
    }
    pub(crate) fn expected_tensors(&self) -> &BTreeMap<String, (ExpectedShape, Dtype)> {
        &self.expected
    }
}

/// Width of the hyper-connection residual stream.
fn hc_hidden(config: &Config) -> usize {
    config.text.hc_count * config.text.hidden_size
}

/// One hyper-connection instance. `with_injection` marks the per-layer instances, which
/// scatter a sublayer's output back into the branches; the top-level mixer has none.
fn hyper_connection(
    put: &mut impl FnMut(String, ExpectedShape, Dtype),
    prefix: &str,
    config: &Config,
    with_injection: bool,
) {
    let hc_hidden = hc_hidden(config);
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

/// Every layer is MoE.
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
    let key_dim = text.linear_num_key_heads * text.linear_key_head_dim;
    let value_dim = value_heads * text.linear_value_head_dim;
    // Fused Q|K|V, which the conv then runs over channel by channel.
    let conv_dim = 2 * key_dim + value_dim;
    put(
        format!("{prefix}.in_proj_qkv.weight"),
        ExpectedShape::Matrix {
            rows: conv_dim,
            cols: hidden,
        },
        EXPECTED_DTYPE,
    );
    put(
        format!("{prefix}.in_proj_z.weight"),
        ExpectedShape::Matrix {
            rows: value_dim,
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
            channels: conv_dim,
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
            cols: value_dim,
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
    // The indexer has its own fused Q|K projection and layernorms.
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

/// The n-gram block, mounted on the one layer `Config::ple_layer` names.
fn ple(put: &mut impl FnMut(String, ExpectedShape, Dtype), layer: &str, config: &Config) {
    let text = &config.text;
    let ngram = &config.ngram;
    let prefix = format!("{layer}.ple");
    let hc_hidden = hc_hidden(config);
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
    // Dilated by `ngram_size` at runtime, which the shape does not show.
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
