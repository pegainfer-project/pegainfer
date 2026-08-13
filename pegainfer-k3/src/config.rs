//! K3 architecture constants + config.json validation (glm52 style:
//! constants are the source of truth, config.json is checked against them).
//!
//! The checkpoint is a multimodal wrapper (`KimiK3ForConditionalGeneration`)
//! whose text tower lives under `text_config` and whose text weights all carry
//! a `language_model.` name prefix. We serve TEXT ONLY: `probe_config_json`
//! validates the wrapper identity and then every text-tower number; the vision
//! tower is deliberately not modelled here (the weight manifest excludes it).
//!
//! Exactly two facts are allowed to vary between published checkpoints: the
//! routed-expert count (`num_experts`) and the EP topology the fleet is
//! launched with. Both live in `K3RoutedExperts` / `K3MoeTopo` rather than in
//! a config struct.

// Bring-up: the constants below are the K3 architecture contract, consumed
// piecewise by the loader, the model builder and the executor as those land.
#![allow(dead_code)]

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use serde_json::Value;

pub(crate) const K3_HIDDEN: usize = 7168;
pub(crate) const K3_VOCAB: usize = 163_840;
pub(crate) const K3_LAYERS: usize = 93;
/// `first_k_dense_replace` — layer 0 carries a plain dense MLP, layers 1..93
/// are latent-MoE.
pub(crate) const K3_DENSE_LAYERS: usize = 1;
/// The checkpoint's `max_position_embeddings` — `probe_config_json` pins the
/// config to exactly this, so it doubles as the architecture ceiling any
/// launch-time `max_model_len` must respect.
pub(crate) const K3_MAX_CONTEXT: usize = 1_048_576;

pub(crate) const K3_HEADS: usize = 96;
const K3_KV_HEADS: usize = 96;
/// Per-head width shared by the KDA recurrent state and the MLA value head.
pub(crate) const K3_HEAD_DIM: usize = 128;
/// `heads * head_dim` — the width of every KDA q/k/v projection, of the
/// per-layer output gate `g_proj`, and of the attention output before `o_proj`.
pub(crate) const K3_ATTN_INNER: usize = K3_HEADS * K3_HEAD_DIM;

// ---- MLA (full-attention) layers -----------------------------------------
pub(crate) const K3_Q_LORA_RANK: usize = 1536;
pub(crate) const K3_KV_LORA_RANK: usize = 512;
pub(crate) const K3_QK_NOPE_HEAD_DIM: usize = 128;
pub(crate) const K3_QK_ROPE_HEAD_DIM: usize = 64;
pub(crate) const K3_QK_HEAD_DIM: usize = K3_QK_NOPE_HEAD_DIM + K3_QK_ROPE_HEAD_DIM;
pub(crate) const K3_V_HEAD_DIM: usize = 128;
pub(crate) const K3_Q_B_OUT: usize = K3_HEADS * K3_QK_HEAD_DIM;
pub(crate) const K3_KV_A_OUT: usize = K3_KV_LORA_RANK + K3_QK_ROPE_HEAD_DIM;
pub(crate) const K3_KV_B_OUT: usize = K3_HEADS * (K3_QK_NOPE_HEAD_DIM + K3_V_HEAD_DIM);
pub(crate) const K3_O_PROJ_IN: usize = K3_HEADS * K3_V_HEAD_DIM;

// ---- KDA (linear-attention) layers ---------------------------------------
/// Short depthwise conv window in front of the KDA q/k/v streams.
pub(crate) const K3_KDA_CONV_KERNEL: usize = 4;
/// Low-rank width of the KDA forget gate (`f_a_proj` out / `f_b_proj` in).
pub(crate) const K3_KDA_GATE_RANK: usize = K3_HEAD_DIM;
pub(crate) const K3_KDA_GATE_LOWER_BOUND: f64 = -5.0;

// ---- MoE -----------------------------------------------------------------
pub(crate) const K3_DENSE_INTERMEDIATE: usize = 33_792;
pub(crate) const K3_EXPERT_INTERMEDIATE: usize = 3072;
/// Latent width the routed experts operate in: hidden 7168 is projected down
/// to 3584 before dispatch and back up after combine.
pub(crate) const K3_ROUTED_EXPERT_HIDDEN: usize = 3584;
pub(crate) const K3_TOPK: usize = 16;
pub(crate) const K3_SHARED_EXPERTS: usize = 2;
/// The two shared experts are stored fused into one MLP.
pub(crate) const K3_SHARED_INTERMEDIATE: usize = K3_SHARED_EXPERTS * K3_EXPERT_INTERMEDIATE;
pub(crate) const K3_ROUTED_SCALING_FACTOR: f64 = 1.0;
/// MXFP4 group size: one e8m0 scale byte per 32 packed values.
pub(crate) const K3_MXFP4_GROUP: usize = 32;
/// MXFP4 packs two 4-bit values per byte.
pub(crate) const K3_MXFP4_PACK: usize = 2;

// ---- Residual / activation ------------------------------------------------
/// Attention-residual stream block size.
pub(crate) const K3_ATTN_RES_BLOCK: usize = 12;
pub(crate) const K3_SITU_BETA: f64 = 4.0;
pub(crate) const K3_SITU_LINEAR_BETA: f64 = 25.0;

const K3_RMS_NORM_EPS: f64 = 1.0e-5;
/// The f32 the GPU norm kernels consume (every RMSNorm in the model shares the
/// one checkpoint eps that `probe_config_json` validates).
pub(crate) const K3_RMS_EPS: f32 = K3_RMS_NORM_EPS as f32;

/// Routed-expert counts this loader accepts. The two published K3 text towers
/// differ in this number alone — every other architecture constant, tensor
/// name and tensor shape is identical, so one loader serves both.
pub(crate) const K3_SUPPORTED_ROUTED_EXPERTS: [usize; 2] = [224, 896];

/// Layer kind. `linear_attn_config` lists the split explicitly; the predicate
/// below is the derived form, and `probe_config_json` proves the two agree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum K3LayerKind {
    /// Kimi Delta Attention: fixed-size recurrent state + short conv window.
    Kda,
    /// Multi-head Latent Attention over paged KV.
    Mla,
}

/// Full attention lands on every 4th layer and on the last layer:
/// `{3, 7, …, 91} ∪ {92}` = 24 of 93; the remaining 69 are KDA.
///
/// The config's `linear_attn_config` lists are ONE-based (`full_attn_layers`
/// ends at 93 for a 93-layer model); this predicate takes a zero-based
/// checkpoint layer index.
pub(crate) fn k3_layer_kind(layer: usize) -> K3LayerKind {
    if layer + 1 == K3_LAYERS || layer % 4 == 3 {
        K3LayerKind::Mla
    } else {
        K3LayerKind::Kda
    }
}

pub(crate) fn k3_layer_is_moe(layer: usize) -> bool {
    layer >= K3_DENSE_LAYERS
}

/// The checkpoint's routed-expert count, validated against
/// `K3_SUPPORTED_ROUTED_EXPERTS`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct K3RoutedExperts(usize);

impl K3RoutedExperts {
    pub(crate) fn new(count: usize) -> Result<Self> {
        ensure!(
            K3_SUPPORTED_ROUTED_EXPERTS.contains(&count),
            "K3 num_experts must be one of {K3_SUPPORTED_ROUTED_EXPERTS:?}, got {count}"
        );
        Ok(Self(count))
    }

    /// Read `text_config.num_experts` out of a K3 config.json.
    pub(crate) fn from_config_json(json: &Value) -> Result<Self> {
        Self::new(usize_field(text_config(json)?, "num_experts")?)
    }

    pub(crate) fn count(self) -> usize {
        self.0
    }
}

/// Expert-parallel partition of the routed experts. Rank `r` owns the
/// contiguous block `r*local .. (r+1)*local`; `ep_size == 1` keeps every
/// expert on one device for single-GPU bring-up.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct K3MoeTopo {
    routed_experts: K3RoutedExperts,
    ep_size: usize,
}

impl K3MoeTopo {
    pub(crate) fn new(routed_experts: K3RoutedExperts, ep_size: usize) -> Result<Self> {
        ensure!(ep_size >= 1, "K3 ep_size must be at least 1, got {ep_size}");
        ensure!(
            routed_experts.count().is_multiple_of(ep_size),
            "K3 ep_size {ep_size} does not divide {} routed experts",
            routed_experts.count()
        );
        Ok(Self {
            routed_experts,
            ep_size,
        })
    }

    pub(crate) fn routed_experts(self) -> K3RoutedExperts {
        self.routed_experts
    }

    pub(crate) fn device_count(self) -> usize {
        self.ep_size
    }

    pub(crate) fn local_experts(self) -> usize {
        self.routed_experts.count() / self.ep_size
    }

    pub(crate) fn rank_expert_range(self, rank: usize) -> Result<std::ops::Range<usize>> {
        ensure!(
            rank < self.ep_size,
            "K3 rank must be in 0..{}, got {rank}",
            self.ep_size
        );
        let local = self.local_experts();
        Ok(rank * local..(rank + 1) * local)
    }
}

/// Validate a K3 config.json (the multimodal wrapper) against the constants
/// above. Everything except `text_config.num_experts` is pinned exactly.
pub(crate) fn probe_config_json(json: &Value) -> Result<()> {
    let model_type = string_field(json, "model_type")?;
    if model_type != "kimi_k3" {
        bail!("not a K3 config: model_type={model_type}");
    }
    ensure!(
        string_array_field(json, "architectures")?
            .iter()
            .any(|value| value == "KimiK3ForConditionalGeneration"),
        "K3 architectures must contain KimiK3ForConditionalGeneration"
    );
    ensure!(
        string_field(json, "dtype")? == "bfloat16",
        "K3 dtype must be bfloat16"
    );
    ensure!(
        !bool_field(json, "tie_word_embeddings")?,
        "K3 tie_word_embeddings must be false"
    );

    let text = text_config(json)?;
    ensure!(
        string_field(text, "model_type")? == "kimi_linear",
        "K3 text_config.model_type must be kimi_linear"
    );
    ensure!(
        string_field(text, "hidden_act")? == "situ",
        "K3 hidden_act must be situ"
    );

    ensure_eq_usize(text, "hidden_size", K3_HIDDEN)?;
    ensure_eq_usize(text, "vocab_size", K3_VOCAB)?;
    ensure_eq_usize(text, "num_hidden_layers", K3_LAYERS)?;
    ensure_eq_usize(text, "first_k_dense_replace", K3_DENSE_LAYERS)?;
    ensure_eq_usize(text, "max_position_embeddings", K3_MAX_CONTEXT)?;
    ensure_eq_usize(text, "intermediate_size", K3_DENSE_INTERMEDIATE)?;
    ensure_eq_usize(text, "moe_intermediate_size", K3_EXPERT_INTERMEDIATE)?;
    ensure_eq_usize(text, "routed_expert_hidden_size", K3_ROUTED_EXPERT_HIDDEN)?;
    ensure_eq_usize(text, "attn_res_block_size", K3_ATTN_RES_BLOCK)?;

    ensure_eq_usize(text, "num_attention_heads", K3_HEADS)?;
    ensure_eq_usize(text, "num_key_value_heads", K3_KV_HEADS)?;
    ensure_eq_usize(text, "q_lora_rank", K3_Q_LORA_RANK)?;
    ensure_eq_usize(text, "kv_lora_rank", K3_KV_LORA_RANK)?;
    ensure_eq_usize(text, "qk_nope_head_dim", K3_QK_NOPE_HEAD_DIM)?;
    ensure_eq_usize(text, "qk_rope_head_dim", K3_QK_ROPE_HEAD_DIM)?;
    ensure_eq_usize(text, "v_head_dim", K3_V_HEAD_DIM)?;
    ensure!(
        bool_field(text, "mla_use_nope")?,
        "K3 mla_use_nope must be true"
    );
    ensure!(
        bool_field(text, "mla_use_output_gate")?,
        "K3 mla_use_output_gate must be true"
    );

    // Routed-expert count is the one number allowed to vary.
    K3RoutedExperts::from_config_json(json)?;
    ensure_eq_usize(text, "num_experts_per_token", K3_TOPK)?;
    ensure_eq_usize(text, "num_shared_experts", K3_SHARED_EXPERTS)?;
    ensure_eq_usize(text, "num_expert_group", 1)?;
    ensure_eq_usize(text, "topk_group", 1)?;
    ensure_eq_usize(text, "moe_layer_freq", 1)?;
    ensure_eq_usize(text, "num_nextn_predict_layers", 0)?;
    ensure!(
        string_field(text, "topk_method")? == "noaux_tc",
        "K3 topk_method must be noaux_tc"
    );
    ensure!(
        string_field(text, "moe_router_activation_func")? == "sigmoid",
        "K3 moe_router_activation_func must be sigmoid"
    );
    ensure!(
        bool_field(text, "moe_renormalize")?,
        "K3 moe_renormalize must be true"
    );
    ensure!(
        bool_field(text, "use_grouped_topk")?,
        "K3 use_grouped_topk must be true"
    );
    ensure!(
        bool_field(text, "latent_moe_use_norm")?,
        "K3 latent_moe_use_norm must be true"
    );
    ensure_float_close(
        number_field(text, "routed_scaling_factor")?,
        K3_ROUTED_SCALING_FACTOR,
        1.0e-12,
        "routed_scaling_factor",
    )?;
    ensure_float_close(
        number_field(text, "rms_norm_eps")?,
        K3_RMS_NORM_EPS,
        1.0e-12,
        "rms_norm_eps",
    )?;
    ensure_float_close(
        number_field(text, "activation_situ_beta")?,
        K3_SITU_BETA,
        1.0e-12,
        "activation_situ_beta",
    )?;
    ensure_float_close(
        number_field(text, "activation_situ_linear_beta")?,
        K3_SITU_LINEAR_BETA,
        1.0e-12,
        "activation_situ_linear_beta",
    )?;

    ensure_linear_attn_config(text)?;
    ensure_mxfp4_quantization(text)?;
    Ok(())
}

fn text_config(json: &Value) -> Result<&Value> {
    json.get("text_config")
        .ok_or_else(|| anyhow::anyhow!("K3 config missing text_config"))
}

/// `linear_attn_config` carries the KDA/full-attention split as two ONE-based
/// layer lists. Both are checked against `k3_layer_kind`, so a checkpoint that
/// moves the split fails the probe instead of silently mis-executing.
fn ensure_linear_attn_config(text: &Value) -> Result<()> {
    let cfg = text
        .get("linear_attn_config")
        .ok_or_else(|| anyhow::anyhow!("K3 text_config missing linear_attn_config"))?;
    ensure_eq_usize(cfg, "num_heads", K3_HEADS)?;
    ensure_eq_usize(cfg, "head_dim", K3_HEAD_DIM)?;
    ensure_eq_usize(cfg, "short_conv_kernel_size", K3_KDA_CONV_KERNEL)?;
    ensure!(
        bool_field(cfg, "use_full_rank_gate")?,
        "K3 linear_attn_config.use_full_rank_gate must be true"
    );
    ensure_float_close(
        number_field(cfg, "gate_lower_bound")?,
        K3_KDA_GATE_LOWER_BOUND,
        1.0e-12,
        "linear_attn_config.gate_lower_bound",
    )?;

    let mut seen = vec![None::<K3LayerKind>; K3_LAYERS];
    for (key, kind) in [
        ("kda_layers", K3LayerKind::Kda),
        ("full_attn_layers", K3LayerKind::Mla),
    ] {
        for one_based in usize_array_field(cfg, key)? {
            let layer = one_based
                .checked_sub(1)
                .ok_or_else(|| anyhow::anyhow!("K3 linear_attn_config.{key} contains 0"))?;
            ensure!(
                layer < K3_LAYERS,
                "K3 linear_attn_config.{key} entry {one_based} is out of range"
            );
            ensure!(
                seen[layer].is_none(),
                "K3 linear_attn_config lists layer {one_based} twice"
            );
            seen[layer] = Some(kind);
        }
    }
    for (layer, kind) in seen.into_iter().enumerate() {
        let kind = kind.ok_or_else(|| {
            anyhow::anyhow!("K3 linear_attn_config does not cover layer index {layer}")
        })?;
        let expected = k3_layer_kind(layer);
        ensure!(
            kind == expected,
            "K3 layer {layer} kind mismatch: config says {kind:?}, architecture says {expected:?}"
        );
    }
    Ok(())
}

/// Routed experts are compressed-tensors MXFP4; everything else (attention,
/// shared experts, dense MLP, lm_head) must stay outside the quantized set.
fn ensure_mxfp4_quantization(text: &Value) -> Result<()> {
    let quant = text
        .get("quantization_config")
        .ok_or_else(|| anyhow::anyhow!("K3 text_config missing quantization_config"))?;
    ensure!(
        string_field(quant, "quant_method")? == "compressed-tensors",
        "K3 quantization_config.quant_method must be compressed-tensors"
    );
    ensure!(
        string_field(quant, "format")? == "mxfp4-pack-quantized",
        "K3 quantization_config.format must be mxfp4-pack-quantized"
    );
    let group = quant
        .get("config_groups")
        .and_then(|groups| groups.get("group_0"))
        .ok_or_else(|| anyhow::anyhow!("K3 quantization_config missing config_groups.group_0"))?;
    let weights = group
        .get("weights")
        .ok_or_else(|| anyhow::anyhow!("K3 quantization_config group_0 missing weights"))?;
    ensure_eq_usize(weights, "num_bits", 4)?;
    ensure_eq_usize(weights, "group_size", K3_MXFP4_GROUP)?;
    ensure!(
        string_field(weights, "strategy")? == "group",
        "K3 MXFP4 strategy must be group"
    );
    ensure!(
        bool_field(weights, "symmetric")?,
        "K3 MXFP4 weights must be symmetric"
    );
    ensure!(
        string_field(weights, "scale_dtype")? == "torch.uint8",
        "K3 MXFP4 scale_dtype must be torch.uint8 (e8m0)"
    );
    let ignore = string_array_field(quant, "ignore")?;
    for required in [
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        "re:.*lm_head.*",
    ] {
        ensure!(
            ignore.iter().any(|entry| entry == required),
            "K3 quantization_config.ignore must contain {required}"
        );
    }
    Ok(())
}

fn string_field(json: &Value, key: &str) -> Result<String> {
    json.get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow::anyhow!("missing string field {key}"))
}

fn string_array_field(json: &Value, key: &str) -> Result<Vec<String>> {
    let values = json
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("missing string array field {key}"))?;
    values
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(ToOwned::to_owned)
                .ok_or_else(|| anyhow::anyhow!("field {key} contains a non-string entry"))
        })
        .collect()
}

fn usize_array_field(json: &Value, key: &str) -> Result<Vec<usize>> {
    let values = json
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("missing integer array field {key}"))?;
    values
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|raw| usize::try_from(raw).ok())
                .ok_or_else(|| anyhow::anyhow!("field {key} contains a non-index entry"))
        })
        .collect()
}

fn usize_field(json: &Value, key: &str) -> Result<usize> {
    let value = json
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow::anyhow!("missing unsigned integer field {key}"))?;
    usize::try_from(value).with_context(|| format!("field {key} does not fit usize"))
}

fn bool_field(json: &Value, key: &str) -> Result<bool> {
    json.get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| anyhow::anyhow!("missing bool field {key}"))
}

fn number_field(json: &Value, key: &str) -> Result<f64> {
    json.get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow::anyhow!("missing numeric field {key}"))
}

fn ensure_eq_usize(json: &Value, key: &str, expected: usize) -> Result<()> {
    let actual = usize_field(json, key)?;
    ensure!(
        actual == expected,
        "{key} mismatch: got {actual}, expected {expected}"
    );
    Ok(())
}

fn ensure_float_close(actual: f64, expected: f64, tolerance: f64, label: &str) -> Result<()> {
    ensure!(
        (actual - expected).abs() <= tolerance,
        "{label} mismatch: got {actual}, expected {expected}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_kinds_match_the_checkpoint_split() {
        let mla = (0..K3_LAYERS)
            .filter(|layer| k3_layer_kind(*layer) == K3LayerKind::Mla)
            .collect::<Vec<_>>();
        assert_eq!(mla.len(), 24);
        assert_eq!(&mla[..3], &[3, 7, 11]);
        assert_eq!(&mla[mla.len() - 2..], &[91, 92]);
        assert_eq!(
            (0..K3_LAYERS)
                .filter(|layer| k3_layer_kind(*layer) == K3LayerKind::Kda)
                .count(),
            69
        );
        // Layer 0 is the one dense-MLP layer and it is a KDA layer.
        assert_eq!(k3_layer_kind(0), K3LayerKind::Kda);
        assert!(!k3_layer_is_moe(0));
        assert!(k3_layer_is_moe(1));
    }

    #[test]
    fn moe_topologies_partition_the_routed_experts() {
        for count in K3_SUPPORTED_ROUTED_EXPERTS {
            let experts = K3RoutedExperts::new(count).unwrap();
            for ep_size in [1usize, 2, 4, 8, 16] {
                let topo = K3MoeTopo::new(experts, ep_size).unwrap();
                assert_eq!(topo.local_experts() * ep_size, count);
                let mut covered = 0usize;
                let mut cursor = 0usize;
                for rank in 0..topo.device_count() {
                    let range = topo.rank_expert_range(rank).unwrap();
                    assert_eq!(range.start, cursor, "rank {rank} range is not contiguous");
                    cursor = range.end;
                    covered += range.len();
                }
                assert_eq!(covered, count);
                assert_eq!(cursor, count);
                assert!(topo.rank_expert_range(ep_size).is_err());
            }
        }
        // 224 experts at EP4 and 896 at EP16 are the same 56-expert shard.
        assert_eq!(
            K3MoeTopo::new(K3RoutedExperts::new(224).unwrap(), 4)
                .unwrap()
                .local_experts(),
            56
        );
        assert_eq!(
            K3MoeTopo::new(K3RoutedExperts::new(896).unwrap(), 16)
                .unwrap()
                .local_experts(),
            56
        );
        assert!(K3RoutedExperts::new(256).is_err());
        assert!(K3MoeTopo::new(K3RoutedExperts::new(224).unwrap(), 3).is_err());
    }
}
