//! Validated geometry for a `qwen4_exp` checkpoint, and the derivations the
//! tensor contract depends on.
//!
//! The probe checks identity; every other rule is here, behind one typed parse,
//! so detection and loading cannot disagree about what is servable. Following
//! `pegainfer-gemma4`, this pins **structure rather than widths**: the expert
//! count, hidden size, layer count, head counts and `hc_count` stay free, so a
//! smaller `qwen4_exp` checkpoint passes the same rules. A value pinned to one
//! constant is one whose alternative needs a kernel or a serving path this line
//! does not have.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use anyhow::ensure;
use serde::Deserialize;

use crate::probe::probe_config_json;

/// The GDN and full-attention kernel limits, mirrored from
/// `pegainfer-qwen35/src/config/model.rs`, whose GDN wrappers own them. They
/// move to the shared owner when slice B lifts those wrappers out of qwen35;
/// until then this crate stays free of CUDA.
const GDN_AOT_KEY_HEAD_DIM: usize = 128;
const GDN_AOT_VALUE_HEAD_DIM: usize = 128;
const LINEAR_CONV_MAX_KERNEL_DIM: usize = 4;
const FULL_ATTN_HEAD_DIM: usize = 256;

#[derive(Deserialize)]
pub(crate) struct RawConfig {
    tie_word_embeddings: bool,
    quantization_config: Option<serde_json::Value>,
    text_config: TextConfig,
}

/// The nested rope object, where the reference reads every rope value from
/// (`modeling_qwen4_exp.py` indexes `config.rope_parameters["rope_theta"]`).
#[derive(Debug, Deserialize)]
pub(crate) struct RopeParameters {
    pub(crate) rope_type: String,
    pub(crate) rope_theta: f64,
    pub(crate) partial_rotary_factor: f64,
}

#[derive(Debug, Deserialize)]
pub(crate) struct TextConfig {
    pub(crate) dtype: String,
    pub(crate) quantization_config: Option<serde_json::Value>,
    pub(crate) mamba_ssm_dtype: String,
    pub(crate) tie_word_embeddings: bool,

    pub(crate) num_hidden_layers: usize,
    pub(crate) hidden_size: usize,
    pub(crate) vocab_size: usize,
    pub(crate) layer_types: Vec<String>,
    pub(crate) full_attention_interval: usize,

    pub(crate) head_dim: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) rope_parameters: RopeParameters,
    /// A legacy copy nothing upstream reads; it must agree with the nested one.
    pub(crate) partial_rotary_factor: Option<f64>,

    pub(crate) linear_key_head_dim: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_num_value_heads: usize,
    pub(crate) linear_conv_kernel_dim: usize,
    pub(crate) output_gate_type: String,

    pub(crate) indexer_n_heads: usize,
    pub(crate) indexer_kv_heads: usize,
    pub(crate) indexer_head_dim: usize,
    pub(crate) indexer_budget: usize,
    pub(crate) indexer_compress_ratio: usize,

    pub(crate) num_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) shared_expert_intermediate_size: usize,
    #[serde(default = "default_norm_topk_prob")]
    #[cfg_attr(not(test), allow(dead_code, reason = "the router is the first reader"))]
    pub(crate) norm_topk_prob: bool,

    pub(crate) hc_count: usize,
    pub(crate) hc_lowrank: usize,

    pub(crate) ple_layer_ids: Vec<usize>,
    pub(crate) ple_conv_kernel_size: usize,
    pub(crate) ngram_size: usize,
    pub(crate) heads_per_ngram: usize,
    pub(crate) ple_embed_dim: usize,
    pub(crate) ngram_vocab_size_base: usize,
    pub(crate) split_ngram_parts: usize,
    pub(crate) make_ngram_vocab_size_divisible_by: usize,
}

/// `norm_topk_prob` is **absent** from the frozen checkpoint's `config.json` and
/// comes from the upstream default. It decides whether the router renormalizes
/// its top-k weights, so a port that reads only `config.json` silently drops the
/// renormalization and gets plausible, wrong routing.
fn default_norm_topk_prob() -> bool {
    true
}

/// Validated geometry plus the derivations the tensor contract needs.
#[derive(Debug)]
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "the tensor contract is the first reader")
)]
pub(crate) struct Config {
    pub(crate) text: TextConfig,
    pub(crate) layer_kinds: Vec<LayerKind>,
    pub(crate) rotary_dim: usize,
    /// The decoder layer carrying the n-gram block. `ple_layer_ids` is
    /// **1-indexed**: the frozen `[2]` puts its tensors at `layers.1`, and reading
    /// it as a 0-indexed subscript injects into the wrong layer and still produces
    /// fluent, wrong text.
    pub(crate) ple_layer: usize,
    pub(crate) ngram: NgramTable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LayerKind {
    LinearAttention,
    FullAttention,
}

impl Config {
    /// Checks identity first, so only a `qwen4_exp` config reaches the typed
    /// parse.
    pub(crate) fn from_json(json: &serde_json::Value) -> Result<Self> {
        probe_config_json(json)?;
        let raw = RawConfig::deserialize(json).map_err(|error| anyhow!("qwen4_exp: {error}"))?;
        Self::try_from(raw)
    }
}

impl TryFrom<RawConfig> for Config {
    type Error = anyhow::Error;

    fn try_from(raw: RawConfig) -> Result<Self> {
        let t = raw.text_config;
        // The quantized siblings (the official FP8, nvidia's NVFP4) keep
        // `dtype: bfloat16` and differ from this checkpoint only by this block.
        ensure!(
            raw.quantization_config.is_none() && t.quantization_config.is_none(),
            "qwen4_exp: quantized checkpoints are not served (quantization_config is present)"
        );
        // The loader must materialize an embedding *and* an LM head; a tied
        // variant would leave one of them missing from the checkpoint.
        ensure!(
            !raw.tie_word_embeddings && !t.tie_word_embeddings,
            "qwen4_exp: tie_word_embeddings must be false at both levels; this line serves the \
             untied checkpoint"
        );
        ensure!(
            t.dtype == "bfloat16",
            "qwen4_exp: text_config.dtype must be bfloat16, got {}",
            t.dtype
        );
        // The GDN recurrent state is f32 in the in-tree kernels.
        ensure!(
            t.mamba_ssm_dtype == "float32",
            "qwen4_exp: mamba_ssm_dtype must be float32, got {}",
            t.mamba_ssm_dtype
        );
        ensure!(
            t.hidden_size > 0 && t.vocab_size > 0,
            "qwen4_exp: hidden_size and vocab_size must be positive"
        );
        let layer_kinds = layer_kinds(&t)?;
        attention(&t)?;
        gated_delta_net(&t)?;
        indexer(&t)?;
        moe(&t)?;
        ensure!(
            t.hc_count > 1,
            "qwen4_exp: hc_count must be greater than 1 (the reference refuses a single \
             residual branch), got {}",
            t.hc_count
        );
        ensure!(t.hc_lowrank > 0, "qwen4_exp: hc_lowrank must be positive");
        let rotary_dim = rotary_dim(&t)?;
        let ple_layer = ple_layer(&t, &layer_kinds)?;
        let ngram = NgramTable::derive(&t)?;
        Ok(Self {
            text: t,
            layer_kinds,
            rotary_dim,
            ple_layer,
            ngram,
        })
    }
}

/// Every `full_attention_interval`-th layer is full attention, the rest Gated
/// DeltaNet. The rule is the interval, not a count of 12 full layers.
fn layer_kinds(t: &TextConfig) -> Result<Vec<LayerKind>> {
    let interval = t.full_attention_interval;
    let count = t.layer_types.len();
    ensure!(
        interval > 0,
        "qwen4_exp: full_attention_interval must be positive"
    );
    ensure!(count > 0, "qwen4_exp: layer_types is empty");
    ensure!(
        count.is_multiple_of(interval),
        "qwen4_exp: layer_types has {count} entries, not a multiple of full_attention_interval \
         {interval}"
    );
    // `num_hidden_layers` is what the per-layer tensor names count by, while
    // `layer_types` says which kind each layer is.
    ensure!(
        t.num_hidden_layers == count,
        "qwen4_exp: num_hidden_layers is {} but layer_types has {count} entries",
        t.num_hidden_layers
    );
    t.layer_types
        .iter()
        .enumerate()
        .map(|(i, kind)| {
            let expected = if (i + 1).is_multiple_of(interval) {
                "full_attention"
            } else {
                "linear_attention"
            };
            ensure!(
                kind == expected,
                "qwen4_exp: layer_types[{i}] is {kind:?}, expected {expected}"
            );
            Ok(if kind == "full_attention" {
                LayerKind::FullAttention
            } else {
                LayerKind::LinearAttention
            })
        })
        .collect()
}

fn attention(t: &TextConfig) -> Result<()> {
    ensure!(
        t.head_dim == FULL_ATTN_HEAD_DIM,
        "qwen4_exp: head_dim must be {FULL_ATTN_HEAD_DIM}, got {}",
        t.head_dim
    );
    // Grouped attention maps each KV head over a whole group of query heads.
    ensure!(
        t.num_attention_heads > 0
            && t.num_key_value_heads > 0
            && t.num_attention_heads.is_multiple_of(t.num_key_value_heads),
        "qwen4_exp: num_attention_heads {} must be a positive multiple of num_key_value_heads {}",
        t.num_attention_heads,
        t.num_key_value_heads
    );
    Ok(())
}

fn gated_delta_net(t: &TextConfig) -> Result<()> {
    ensure!(
        t.linear_key_head_dim == GDN_AOT_KEY_HEAD_DIM
            && t.linear_value_head_dim == GDN_AOT_VALUE_HEAD_DIM,
        "qwen4_exp: linear_key_head_dim and linear_value_head_dim must be \
         {GDN_AOT_KEY_HEAD_DIM} (the Triton AOT keys are specialized on them), got {} and {}",
        t.linear_key_head_dim,
        t.linear_value_head_dim
    );
    // Gated DeltaNet maps value heads in whole groups over key heads.
    ensure!(
        t.linear_num_key_heads > 0
            && t.linear_num_value_heads > 0
            && t.linear_num_value_heads
                .is_multiple_of(t.linear_num_key_heads),
        "qwen4_exp: linear_num_value_heads {} must be a positive multiple of \
         linear_num_key_heads {}",
        t.linear_num_value_heads,
        t.linear_num_key_heads
    );
    ensure!(
        (1..=LINEAR_CONV_MAX_KERNEL_DIM).contains(&t.linear_conv_kernel_dim),
        "qwen4_exp: linear_conv_kernel_dim must be in 1..={LINEAR_CONV_MAX_KERNEL_DIM} (the \
         in-tree conv1d caps its state), got {}",
        t.linear_conv_kernel_dim
    );
    // Live on this architecture: upstream passes it to the GDN gated norm. The
    // frozen checkpoint is sigmoid, and no other value has been validated here.
    ensure!(
        t.output_gate_type == "sigmoid",
        "qwen4_exp: output_gate_type must be sigmoid, got {}",
        t.output_gate_type
    );
    Ok(())
}

/// The QSA indexer rules of the reference's `validate_architecture`
/// (`configuration_qwen4_exp.py:213-224`).
fn indexer(t: &TextConfig) -> Result<()> {
    ensure!(
        t.indexer_n_heads > 0
            && t.indexer_head_dim > 0
            && t.indexer_budget > 0
            && t.indexer_compress_ratio > 0,
        "qwen4_exp: indexer_n_heads, indexer_head_dim, indexer_budget and \
         indexer_compress_ratio must be positive"
    );
    ensure!(
        t.indexer_kv_heads == 1,
        "qwen4_exp: indexer_kv_heads must be 1 (the reference QSA requires it), got {}",
        t.indexer_kv_heads
    );
    ensure!(
        t.indexer_budget.is_multiple_of(t.indexer_compress_ratio),
        "qwen4_exp: indexer_budget {} must be divisible by indexer_compress_ratio {}",
        t.indexer_budget,
        t.indexer_compress_ratio
    );
    Ok(())
}

fn moe(t: &TextConfig) -> Result<()> {
    ensure!(
        (1..=t.num_experts).contains(&t.num_experts_per_tok),
        "qwen4_exp: num_experts_per_tok {} must be in 1..=num_experts {}",
        t.num_experts_per_tok,
        t.num_experts
    );
    ensure!(
        t.moe_intermediate_size > 0 && t.shared_expert_intermediate_size > 0,
        "qwen4_exp: moe_intermediate_size and shared_expert_intermediate_size must be positive"
    );
    Ok(())
}

fn rotary_dim(t: &TextConfig) -> Result<usize> {
    let rope = &t.rope_parameters;
    // 262,144 is native under default RoPE; this line has no YaRN path.
    ensure!(
        rope.rope_type == "default",
        "qwen4_exp: rope_parameters.rope_type must be default, got {}",
        rope.rope_type
    );
    ensure!(
        rope.rope_theta.is_finite() && rope.rope_theta > 0.0,
        "qwen4_exp: rope_parameters.rope_theta must be a positive number, got {}",
        rope.rope_theta
    );
    let partial = rope.partial_rotary_factor;
    ensure!(
        partial > 0.0 && partial <= 1.0,
        "qwen4_exp: rope_parameters.partial_rotary_factor must be in (0, 1], got {partial}"
    );
    if let Some(legacy) = t.partial_rotary_factor {
        // Exact equality on purpose: both are JSON literals, and a tolerance would
        // hide a checkpoint that writes 0.3 in one place and 0.25 in the other.
        #[allow(clippy::float_cmp)]
        let agrees = legacy == partial;
        ensure!(
            agrees,
            "qwen4_exp: partial_rotary_factor is ambiguous: text_config says {legacy}, \
             text_config.rope_parameters says {partial}, and only the nested copy is read"
        );
    }
    let exact = t.head_dim as f64 * partial;
    ensure!(
        exact.fract() == 0.0,
        "qwen4_exp: partial_rotary_factor {partial} x head_dim {} is not a whole number of \
         rotary dimensions",
        t.head_dim
    );
    let rotary_dim = exact as usize;
    // The reference builds its frequencies with `arange(0, rotary_dim, 2)` and
    // rotates `cos.shape[-1]` dims, which is one more than an odd width.
    ensure!(
        rotary_dim.is_multiple_of(2),
        "qwen4_exp: rotary_dim {rotary_dim} must be even (the reference rotates in pairs)"
    );
    // The reference requires the rotary width to fit the QSA index head
    // (`configuration_qwen4_exp.py:227`).
    ensure!(
        rotary_dim <= t.indexer_head_dim,
        "qwen4_exp: rotary_dim {rotary_dim} exceeds indexer_head_dim {}",
        t.indexer_head_dim
    );
    Ok(rotary_dim)
}

fn ple_layer(t: &TextConfig, layer_kinds: &[LayerKind]) -> Result<usize> {
    // Each PLE layer has its own n-gram table, its prime run offset by the layer's
    // position in the list (`modeling_qwen4_exp.py:1037-1039`); one table is
    // derived here.
    let [id] = t.ple_layer_ids[..] else {
        bail!(
            "qwen4_exp: exactly one ple_layer_ids entry is modelled, got {:?}",
            t.ple_layer_ids
        );
    };
    ensure!(
        (1..=layer_kinds.len()).contains(&id),
        "qwen4_exp: ple_layer_ids entry {id} is outside 1..={} (the field is 1-indexed)",
        layer_kinds.len()
    );
    // The reference supports PLE only on linear_attention layers
    // (`configuration_qwen4_exp.py:248-254`).
    ensure!(
        layer_kinds[id - 1] == LayerKind::LinearAttention,
        "qwen4_exp: ple_layer_ids entry {id} sits on a full_attention layer, but PLE is only \
         supported on linear_attention layers"
    );
    ensure!(
        t.ple_conv_kernel_size > 0,
        "qwen4_exp: ple_conv_kernel_size must be positive"
    );
    Ok(id - 1)
}

/// The n-gram table geometry, **derived from config alone**. The checkpoint
/// also stores `ngram_heads_vocab_sizes` and `ngram_heads_offsets`, so a loader
/// can cross-check this against the stored vectors before any bytes are
/// uploaded.
#[derive(Debug)]
#[cfg_attr(
    not(test),
    allow(dead_code, reason = "the tensor contract is the first reader")
)]
pub(crate) struct NgramTable {
    /// `(ngram_size - 1) * heads_per_ngram`, one head per (order, head) pair.
    pub(crate) heads: usize,
    pub(crate) head_dim: usize,
    /// Per-head vocabulary size; each is a distinct prime.
    pub(crate) head_vocab_sizes: Vec<usize>,
    /// Per-head row offset into the flat table.
    pub(crate) head_offsets: Vec<usize>,
    /// Sum of the per-head vocabularies, before padding.
    pub(crate) total_vocab_size: usize,
    /// Row count of the stored embedding, padded up to a multiple of
    /// `make_ngram_vocab_size_divisible_by`.
    pub(crate) padded_vocab_size: usize,
    /// The table ships as this many equal named shards.
    pub(crate) shards: usize,
    pub(crate) rows_per_shard: usize,
}

impl NgramTable {
    fn derive(t: &TextConfig) -> Result<Self> {
        ensure!(
            t.ngram_size >= 2,
            "qwen4_exp: ngram_size must be at least 2 (an n-gram needs a context token), got {}",
            t.ngram_size
        );
        let heads = (t.ngram_size - 1) * t.heads_per_ngram;
        ensure!(
            heads > 0 && t.ple_embed_dim > 0 && t.ple_embed_dim.is_multiple_of(heads),
            "qwen4_exp: ple_embed_dim {} must split evenly across the {heads} derived n-gram heads",
            t.ple_embed_dim
        );
        ensure!(
            t.ngram_vocab_size_base > 0
                && t.make_ngram_vocab_size_divisible_by > 0
                && t.split_ngram_parts > 0,
            "qwen4_exp: ngram_vocab_size_base, make_ngram_vocab_size_divisible_by and \
             split_ngram_parts must be positive"
        );

        // Head `i` takes the (i+1)-th prime after `ngram_vocab_size_base - 1`, and
        // the heads are laid out contiguously (`modeling_qwen4_exp.py:1037-1042`).
        let mut head_vocab_sizes = Vec::with_capacity(heads);
        let mut head_offsets = Vec::with_capacity(heads);
        let mut total = 0usize;
        for head in 0..heads {
            let size = nth_prime_after(t.ngram_vocab_size_base - 1, head + 1);
            head_offsets.push(total);
            total += size;
            head_vocab_sizes.push(size);
        }

        let divisor = t.make_ngram_vocab_size_divisible_by;
        let padded = total.div_ceil(divisor) * divisor;
        ensure!(
            padded.is_multiple_of(t.split_ngram_parts),
            "qwen4_exp: padded n-gram vocabulary {padded} is not divisible into {} equal shards",
            t.split_ngram_parts
        );
        Ok(Self {
            heads,
            head_dim: t.ple_embed_dim / heads,
            head_vocab_sizes,
            head_offsets,
            total_vocab_size: total,
            padded_vocab_size: padded,
            shards: t.split_ngram_parts,
            rows_per_shard: padded / t.split_ngram_parts,
        })
    }
}

/// The reference's `_find_nth_prime_after`: the `count`-th prime strictly
/// greater than `start`.
fn nth_prime_after(start: usize, count: usize) -> usize {
    let mut prime = start;
    for _ in 0..count {
        prime += 1;
        while !is_prime(prime) {
            prime += 1;
        }
    }
    prime
}

/// The reference's `_is_prime`: trial division by odd numbers up to the square
/// root.
fn is_prime(value: usize) -> bool {
    if value < 2 {
        return false;
    }
    if value.is_multiple_of(2) {
        return value == 2;
    }
    let mut divisor = 3;
    while divisor * divisor <= value {
        if value.is_multiple_of(divisor) {
            return false;
        }
        divisor += 2;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// sha256 of the frozen revision's `config.json`. The fixture is committed
    /// verbatim, so an edit to it has to come with a new revision and a new hash.
    const FROZEN_CONFIG_SHA256: &str =
        "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b";

    /// The frozen revision's real `config.json`. Testing against it rather than
    /// a transcription is the point: a hand-written fixture can drift from the
    /// checkpoint silently.
    fn frozen() -> serde_json::Value {
        serde_json::from_str(include_str!("../tests/frozen_config.json")).unwrap()
    }

    fn config() -> Config {
        Config::from_json(&frozen()).unwrap()
    }

    fn err(mutate: impl FnOnce(&mut serde_json::Value)) -> String {
        let mut cfg = frozen();
        mutate(&mut cfg);
        Config::from_json(&cfg).unwrap_err().to_string()
    }

    fn text(cfg: &mut serde_json::Value, field: &str, value: serde_json::Value) {
        cfg["text_config"][field] = value;
    }

    #[test]
    fn the_committed_fixture_is_the_frozen_revision() {
        use sha2::Digest;
        let raw = include_str!("../tests/frozen_config.json");
        let digest = sha2::Sha256::digest(raw.as_bytes());
        let hex = digest
            .iter()
            .fold(String::with_capacity(64), |mut hex, byte| {
                use std::fmt::Write as _;
                let _ = write!(hex, "{byte:02x}");
                hex
            });
        assert_eq!(
            hex, FROZEN_CONFIG_SHA256,
            "fixture drifted from the frozen revision"
        );
    }

    #[test]
    fn the_frozen_revision_derives_the_measured_geometry() {
        let cfg = config();
        let full = cfg
            .layer_kinds
            .iter()
            .filter(|kind| **kind == LayerKind::FullAttention)
            .count();
        assert_eq!((full, cfg.layer_kinds.len() - full), (12, 36));
        assert_eq!(cfg.rotary_dim, 64);
    }

    /// `ple_layer_ids: [2]` must land the PLE tensors on `layers.1`, not
    /// `layers.2`.
    #[test]
    fn ple_layer_ids_are_one_indexed() {
        assert_eq!(config().ple_layer, 1);
    }

    /// The table totals, which neither the stored vectors nor the shard shapes state.
    #[test]
    fn the_ngram_table_derivation_matches_the_checkpoint() {
        let ngram = config().ngram;
        assert_eq!(ngram.total_vocab_size, 320_001_446);
        assert_eq!(ngram.padded_vocab_size, 320_001_536);
    }

    #[test]
    fn the_absent_router_field_takes_its_upstream_default() {
        // Absent from the frozen config.json; dropping the default would silently
        // skip the router's top-k renormalization.
        assert!(
            !frozen()["text_config"]
                .as_object()
                .unwrap()
                .contains_key("norm_topk_prob")
        );
        assert!(config().text.norm_topk_prob);
    }

    #[test]
    fn nth_prime_after_matches_the_reference() {
        assert_eq!(nth_prime_after(19_999_999, 1), 20_000_003);
        assert_eq!(nth_prime_after(19_999_999, 2), 20_000_023);
        assert_eq!(nth_prime_after(19_999_999, 16), 20_000_171);
        assert_eq!(nth_prime_after(1, 1), 2);
        assert_eq!(nth_prime_after(2, 3), 7);
        assert_eq!(nth_prime_after(0, 1), 2);
        assert_eq!(nth_prime_after(0, 2), 3);
    }

    /// Widths a smaller `qwen4_exp` checkpoint would change must not be pinned.
    #[test]
    fn smaller_widths_still_pass() {
        let mut cfg = frozen();
        text(&mut cfg, "num_experts", serde_json::json!(64));
        text(&mut cfg, "num_experts_per_tok", serde_json::json!(4));
        text(&mut cfg, "moe_intermediate_size", serde_json::json!(512));
        text(
            &mut cfg,
            "shared_expert_intermediate_size",
            serde_json::json!(512),
        );
        text(&mut cfg, "hidden_size", serde_json::json!(1280));
        text(&mut cfg, "hc_count", serde_json::json!(2));
        text(&mut cfg, "hc_lowrank", serde_json::json!(160));
        text(&mut cfg, "linear_num_key_heads", serde_json::json!(8));
        text(&mut cfg, "linear_num_value_heads", serde_json::json!(24));
        text(&mut cfg, "num_attention_heads", serde_json::json!(12));
        Config::from_json(&cfg).unwrap();
    }

    /// The real `Qwen/Qwen3.8-Flash-Next-FP8` config differs from the frozen one
    /// only by this block (its 943-entry `modules_to_not_convert` trimmed), and
    /// its `text_config.dtype` is still `bfloat16`.
    #[test]
    fn the_fp8_sibling_fails_closed() {
        let err = err(|c| {
            c["quantization_config"] = serde_json::json!({
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
                "weight_per_tensor": false,
                "act_per_tensor": false,
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["lm_head", "model.language_model.embed_tokens"],
                "modules_to_convert": ["ple.ple_embedding.ngram_embedding"]
            });
        });
        assert!(err.contains("quantization_config"), "{err}");
    }

    #[test]
    fn a_non_bf16_dtype_bails() {
        let err = err(|c| text(c, "dtype", serde_json::json!("float16")));
        assert!(err.contains("bfloat16"), "{err}");
    }

    #[test]
    fn a_tied_checkpoint_fails_closed() {
        let err = err(|c| text(c, "tie_word_embeddings", serde_json::json!(true)));
        assert!(err.contains("tie_word_embeddings"), "{err}");
    }

    #[test]
    fn a_gdn_head_dim_off_the_aot_key_bails() {
        let err = err(|c| text(c, "linear_value_head_dim", serde_json::json!(64)));
        assert!(err.contains("linear_value_head_dim"), "{err}");
        assert!(err.contains("AOT"), "{err}");
    }

    #[test]
    fn a_conv_kernel_past_the_in_tree_cap_bails() {
        let err = err(|c| text(c, "linear_conv_kernel_dim", serde_json::json!(5)));
        assert!(err.contains("linear_conv_kernel_dim"), "{err}");
    }

    #[test]
    fn a_non_integral_gqa_group_bails() {
        let err = err(|c| {
            text(c, "num_attention_heads", serde_json::json!(10));
            text(c, "num_key_value_heads", serde_json::json!(3));
        });
        assert!(err.contains("num_key_value_heads"), "{err}");
    }

    #[test]
    fn a_non_integral_gdn_head_group_bails() {
        let err = err(|c| {
            text(c, "linear_num_key_heads", serde_json::json!(10));
            text(c, "linear_num_value_heads", serde_json::json!(48));
        });
        assert!(err.contains("linear_num_key_heads"), "{err}");
    }

    #[test]
    fn a_non_sigmoid_output_gate_bails() {
        // The sibling Qwen3.8-27B checkpoint carries `swish` here, where the field
        // is dead. On this architecture it is live and selects the GDN gated-norm
        // activation, so a different value is an unvalidated variant.
        let err = err(|c| text(c, "output_gate_type", serde_json::json!("swish")));
        assert!(err.contains("output_gate_type"), "{err}");
    }

    #[test]
    fn a_non_f32_recurrent_state_bails() {
        let err = err(|c| text(c, "mamba_ssm_dtype", serde_json::json!("bfloat16")));
        assert!(err.contains("mamba_ssm_dtype"), "{err}");
    }

    #[test]
    fn a_yarn_rope_bails() {
        let err = err(|c| {
            c["text_config"]["rope_parameters"]["rope_type"] = serde_json::json!("yarn");
        });
        assert!(err.contains("rope_type"), "{err}");
    }

    #[test]
    fn a_partial_rotary_factor_that_is_not_whole_bails() {
        let err = err(|c| {
            c["text_config"]["rope_parameters"]["partial_rotary_factor"] = serde_json::json!(0.3);
            text(c, "partial_rotary_factor", serde_json::json!(0.3));
        });
        assert!(err.contains("whole number"), "{err}");
    }

    #[test]
    fn an_odd_rotary_width_bails() {
        // 63/256 is exact in binary, so 256 x factor is the whole number 63.
        let err = err(|c| {
            c["text_config"]["rope_parameters"]["partial_rotary_factor"] =
                serde_json::json!(0.246_093_75);
            text(c, "partial_rotary_factor", serde_json::json!(0.246_093_75));
        });
        assert!(err.contains("even"), "{err}");
    }

    #[test]
    fn an_ambiguous_partial_rotary_factor_bails() {
        let err = err(|c| text(c, "partial_rotary_factor", serde_json::json!(0.5)));
        assert!(err.contains("ambiguous"), "{err}");
    }

    #[test]
    fn a_rotary_width_past_the_indexer_head_bails() {
        // The frozen 64 rotary dims (0.25 x 256) fit the 128-wide index head; a
        // 32-wide one does not.
        let err = err(|c| text(c, "indexer_head_dim", serde_json::json!(32)));
        assert!(err.contains("indexer_head_dim"), "{err}");
    }

    #[test]
    fn the_full_attention_interval_pattern_is_enforced() {
        let err = err(|c| c["text_config"]["layer_types"][1] = serde_json::json!("full_attention"));
        assert!(err.contains("layer_types[1]"), "{err}");
    }

    #[test]
    fn a_layer_count_not_divisible_by_the_interval_bails() {
        let err = err(|c| {
            text(
                c,
                "layer_types",
                serde_json::json!(["linear_attention", "linear_attention", "full_attention"]),
            );
        });
        assert!(err.contains("multiple of full_attention_interval"), "{err}");
    }

    #[test]
    fn a_layer_count_that_disagrees_with_layer_types_bails() {
        let err = err(|c| text(c, "num_hidden_layers", serde_json::json!(47)));
        assert!(err.contains("num_hidden_layers"), "{err}");
    }

    #[test]
    fn top_k_above_the_expert_count_bails() {
        let err = err(|c| text(c, "num_experts_per_tok", serde_json::json!(513)));
        assert!(err.contains("num_experts"), "{err}");
    }

    #[test]
    fn a_single_branch_residual_stream_bails() {
        let err = err(|c| text(c, "hc_count", serde_json::json!(1)));
        assert!(err.contains("hc_count"), "{err}");
    }

    #[test]
    fn a_zero_residual_width_bails() {
        let err = err(|c| text(c, "hidden_size", serde_json::json!(0)));
        assert!(err.contains("hidden_size"), "{err}");
    }

    #[test]
    fn an_indexer_kv_heads_above_one_bails() {
        let err = err(|c| text(c, "indexer_kv_heads", serde_json::json!(2)));
        assert!(err.contains("indexer_kv_heads"), "{err}");
    }

    #[test]
    fn an_indexer_budget_not_divisible_by_the_compress_ratio_bails() {
        let err = err(|c| text(c, "indexer_budget", serde_json::json!(2049)));
        assert!(err.contains("indexer_budget"), "{err}");
    }

    #[test]
    fn ple_layer_ids_are_range_checked() {
        let zero = err(|c| text(c, "ple_layer_ids", serde_json::json!([0])));
        assert!(zero.contains("1-indexed"), "{zero}");
        let past = err(|c| text(c, "ple_layer_ids", serde_json::json!([49])));
        assert!(past.contains("outside 1..=48"), "{past}");
    }

    #[test]
    fn a_multi_layer_ple_bails() {
        let err = err(|c| text(c, "ple_layer_ids", serde_json::json!([2, 6])));
        assert!(err.contains("ple_layer_ids"), "{err}");
    }

    #[test]
    fn ple_on_a_full_attention_layer_bails() {
        // Under interval 4, 1-indexed layer 4 is full attention.
        let err = err(|c| text(c, "ple_layer_ids", serde_json::json!([4])));
        assert!(err.contains("linear_attention"), "{err}");
    }

    #[test]
    fn a_unigram_only_ngram_bails() {
        let err = err(|c| text(c, "ngram_size", serde_json::json!(1)));
        assert!(err.contains("ngram_size"), "{err}");
    }

    #[test]
    fn a_zero_ngram_base_bails() {
        let err = err(|c| text(c, "ngram_vocab_size_base", serde_json::json!(0)));
        assert!(err.contains("ngram_vocab_size_base"), "{err}");
    }

    #[test]
    fn a_table_that_cannot_shard_evenly_is_refused() {
        // 3 parts cannot divide a 128-padded row count.
        let err = err(|c| text(c, "split_ngram_parts", serde_json::json!(3)));
        assert!(err.contains("equal shards"), "{err}");
    }

    #[test]
    fn an_embed_dim_that_cannot_split_across_heads_is_refused() {
        let err = err(|c| text(c, "ple_embed_dim", serde_json::json!(2561)));
        assert!(err.contains("ple_embed_dim"), "{err}");
    }

    #[test]
    fn the_ngram_partition_equals_the_stored_integer_vectors() {
        let stored = crate::manifest::frozen_tensors().values;
        let ngram = config().ngram;
        let embedding = "model.language_model.layers.1.ple.ple_embedding";
        assert_eq!(
            stored[&format!("{embedding}.ngram_heads_vocab_sizes")],
            ngram.head_vocab_sizes
        );
        assert_eq!(
            stored[&format!("{embedding}.ngram_heads_offsets")],
            ngram.head_offsets
        );
    }
}
