//! GLM5.2 constants and config probing.

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use serde_json::Value;

pub(crate) const GLM52_HIDDEN: usize = 6144;
pub(crate) const GLM52_VOCAB: usize = 154_880;
pub(crate) const GLM52_LAYERS: usize = 78;
/// Checkpoint layer containing the native multi-token prediction decoder.
pub(crate) const GLM52_MTP_LAYER: usize = GLM52_LAYERS;
pub(crate) const GLM52_DENSE_LAYERS: usize = 3;
/// The checkpoint's `max_position_embeddings` — `probe_config_json` pins the
/// config to exactly this, so it doubles as the architecture ceiling any
/// launch-time `max_model_len` must respect.
pub(crate) const GLM52_MAX_CONTEXT: usize = 1_048_576;

pub(crate) const GLM52_HEADS: usize = 64;
const GLM52_KV_HEADS: usize = 64;
const GLM52_HEAD_DIM: usize = 192;
pub(crate) const GLM52_Q_LORA_RANK: usize = 2048;
pub(crate) const GLM52_KV_LORA_RANK: usize = 512;
pub(crate) const GLM52_QK_NOPE_HEAD_DIM: usize = 192;
pub(crate) const GLM52_QK_ROPE_HEAD_DIM: usize = 64;
pub(crate) const GLM52_QK_HEAD_DIM: usize = GLM52_QK_NOPE_HEAD_DIM + GLM52_QK_ROPE_HEAD_DIM;
pub(crate) const GLM52_V_HEAD_DIM: usize = 256;
pub(crate) const GLM52_Q_B_OUT: usize = GLM52_HEADS * GLM52_QK_HEAD_DIM;
pub(crate) const GLM52_KV_A_OUT: usize = GLM52_KV_LORA_RANK + GLM52_QK_ROPE_HEAD_DIM;
pub(crate) const GLM52_KV_B_OUT: usize = GLM52_HEADS * (GLM52_QK_NOPE_HEAD_DIM + GLM52_V_HEAD_DIM);
pub(crate) const GLM52_O_PROJ_IN: usize = GLM52_HEADS * GLM52_V_HEAD_DIM;

/// Half of the rotary head dim — the rotary table width shared by MLA and
/// indexer rope.
pub(crate) const GLM52_ROPE_HALF: usize = GLM52_QK_ROPE_HEAD_DIM / 2;
/// `1/sqrt(GLM52_QK_HEAD_DIM)` = 1/sqrt(256) = 0.0625 — the MLA softmax scale
/// (rope_type "default" means no YaRN mscale correction).
pub(crate) const GLM52_SM_SCALE: f32 = 0.0625;

pub(crate) const GLM52_DENSE_INTERMEDIATE: usize = 12_288;
pub(crate) const GLM52_EXPERT_INTERMEDIATE: usize = 2048;
pub(crate) const GLM52_ROUTED_EXPERTS: usize = 256;
pub(crate) const GLM52_TOPK: usize = 8;
const GLM52_SHARED_EXPERTS: usize = 1;
pub(crate) const GLM52_ROUTED_SCALING_FACTOR: f64 = 2.5;
const GLM52_RMS_NORM_EPS: f64 = 1.0e-5;
/// The f32 the GPU norm kernels consume (every RMSNorm in the model shares
/// the one checkpoint eps that `probe_config_json` validates).
pub(crate) const GLM52_RMS_EPS: f32 = GLM52_RMS_NORM_EPS as f32;

pub(crate) const GLM52_INDEX_TOPK: usize = 2048;
const GLM52_INDEX_TOPK_FREQ: usize = 4;
pub(crate) const GLM52_INDEX_HEAD_DIM: usize = 128;
pub(crate) const GLM52_INDEX_HEADS: usize = 32;
const GLM52_INDEX_SKIP_TOPK_OFFSET: usize = 3;
const GLM52_NEXTN_LAYERS: usize = 1;

pub(crate) const GLM52_ROPE_THETA: f64 = 8_000_000.0;

/// `indexer_types[layer]` per the transformers derivation
/// (`index_topk_freq=4`, `index_skip_topk_offset=3`): full iff
/// `max(layer-(offset-1), 0) % freq == 0` → {0,1,2} ∪ {6,10,…,74}, 21 of 78 layers.
pub(crate) fn glm52_layer_has_full_indexer(layer: usize) -> bool {
    layer
        .saturating_sub(GLM52_INDEX_SKIP_TOPK_OFFSET - 1)
        .is_multiple_of(GLM52_INDEX_TOPK_FREQ)
}

pub(crate) fn probe_config_json(json: &Value) -> Result<()> {
    let model_type = string_field(json, "model_type")?;
    if model_type != "glm_moe_dsa" {
        bail!("not a GLM5.2 config: model_type={model_type}");
    }
    ensure!(
        string_array_field(json, "architectures")?
            .iter()
            .any(|value| value == "GlmMoeDsaForCausalLM"),
        "GLM5.2 architectures must contain GlmMoeDsaForCausalLM"
    );
    ensure!(
        string_field(json, "dtype")? == "bfloat16",
        "GLM5.2 dtype must be bfloat16"
    );
    ensure!(
        !bool_field(json, "attention_bias")?,
        "GLM5.2 attention_bias must be false"
    );
    ensure!(
        string_field(json, "hidden_act")? == "silu",
        "GLM5.2 hidden_act must be silu"
    );

    ensure_eq_usize(json, "hidden_size", GLM52_HIDDEN)?;
    ensure_eq_usize(json, "vocab_size", GLM52_VOCAB)?;
    ensure_eq_usize(json, "num_hidden_layers", GLM52_LAYERS)?;
    ensure_eq_usize(json, "first_k_dense_replace", GLM52_DENSE_LAYERS)?;
    ensure_eq_usize(json, "max_position_embeddings", GLM52_MAX_CONTEXT)?;
    ensure_eq_usize(json, "intermediate_size", GLM52_DENSE_INTERMEDIATE)?;
    ensure_eq_usize(json, "moe_intermediate_size", GLM52_EXPERT_INTERMEDIATE)?;

    ensure_eq_usize(json, "num_attention_heads", GLM52_HEADS)?;
    ensure_eq_usize(json, "num_key_value_heads", GLM52_KV_HEADS)?;
    ensure_eq_usize(json, "head_dim", GLM52_HEAD_DIM)?;
    ensure_eq_usize(json, "q_lora_rank", GLM52_Q_LORA_RANK)?;
    ensure_eq_usize(json, "kv_lora_rank", GLM52_KV_LORA_RANK)?;
    ensure_eq_usize(json, "qk_nope_head_dim", GLM52_QK_NOPE_HEAD_DIM)?;
    ensure_eq_usize(json, "qk_rope_head_dim", GLM52_QK_ROPE_HEAD_DIM)?;
    ensure_eq_usize(json, "qk_head_dim", GLM52_QK_HEAD_DIM)?;
    ensure_eq_usize(json, "v_head_dim", GLM52_V_HEAD_DIM)?;
    ensure!(
        bool_field(json, "rope_interleave")?,
        "GLM5.2 rope_interleave must be true"
    );
    let rope = json
        .get("rope_parameters")
        .ok_or_else(|| anyhow::anyhow!("GLM5.2 config missing rope_parameters"))?;
    ensure!(
        string_field(rope, "rope_type")? == "default",
        "GLM5.2 rope_parameters.rope_type must be default"
    );
    ensure_float_close(
        number_field(rope, "rope_theta")?,
        GLM52_ROPE_THETA,
        1.0e-6,
        "rope_parameters.rope_theta",
    )?;

    ensure_eq_usize(json, "n_routed_experts", GLM52_ROUTED_EXPERTS)?;
    ensure_eq_usize(json, "num_experts_per_tok", GLM52_TOPK)?;
    ensure_eq_usize(json, "n_shared_experts", GLM52_SHARED_EXPERTS)?;
    ensure_eq_usize(json, "n_group", 1)?;
    ensure_eq_usize(json, "topk_group", 1)?;
    ensure!(
        string_field(json, "topk_method")? == "noaux_tc",
        "GLM5.2 topk_method must be noaux_tc"
    );
    ensure!(
        string_field(json, "scoring_func")? == "sigmoid",
        "GLM5.2 scoring_func must be sigmoid"
    );
    ensure!(
        bool_field(json, "norm_topk_prob")?,
        "GLM5.2 norm_topk_prob must be true"
    );
    ensure_float_close(
        number_field(json, "routed_scaling_factor")?,
        GLM52_ROUTED_SCALING_FACTOR,
        1.0e-12,
        "routed_scaling_factor",
    )?;
    ensure_float_close(
        number_field(json, "rms_norm_eps")?,
        GLM52_RMS_NORM_EPS,
        1.0e-12,
        "rms_norm_eps",
    )?;

    ensure_eq_usize(json, "index_topk", GLM52_INDEX_TOPK)?;
    ensure_eq_usize(json, "index_topk_freq", GLM52_INDEX_TOPK_FREQ)?;
    ensure_eq_usize(json, "index_head_dim", GLM52_INDEX_HEAD_DIM)?;
    ensure_eq_usize(json, "index_n_heads", GLM52_INDEX_HEADS)?;
    ensure_eq_usize(json, "index_skip_topk_offset", GLM52_INDEX_SKIP_TOPK_OFFSET)?;
    ensure!(
        bool_field(json, "indexer_rope_interleave")?,
        "GLM5.2 indexer_rope_interleave must be true"
    );

    ensure_eq_usize(json, "num_nextn_predict_layers", GLM52_NEXTN_LAYERS)?;
    ensure!(
        bool_field(json, "index_share_for_mtp_iteration")?,
        "GLM5.2 index_share_for_mtp_iteration must be true"
    );
    ensure!(
        !bool_field(json, "tie_word_embeddings")?,
        "GLM5.2 tie_word_embeddings must be false"
    );
    ensure_mlp_layer_types(json)?;
    ensure_indexer_types(json)?;
    ensure_fp8_quantization(json)?;

    Ok(())
}

fn ensure_mlp_layer_types(json: &Value) -> Result<()> {
    let types = string_array_field(json, "mlp_layer_types")?;
    ensure!(
        types.len() == GLM52_LAYERS,
        "GLM5.2 mlp_layer_types length mismatch: got {}, expected {GLM52_LAYERS}",
        types.len()
    );
    for (idx, kind) in types.iter().enumerate() {
        let expected = if idx < GLM52_DENSE_LAYERS {
            "dense"
        } else {
            "sparse"
        };
        ensure!(
            kind == expected,
            "GLM5.2 mlp_layer_types[{idx}] mismatch: got {kind}, expected {expected}"
        );
    }
    Ok(())
}

fn ensure_indexer_types(json: &Value) -> Result<()> {
    let types = string_array_field(json, "indexer_types")?;
    ensure!(
        types.len() == GLM52_LAYERS,
        "GLM5.2 indexer_types length mismatch: got {}, expected {GLM52_LAYERS}",
        types.len()
    );
    ensure!(
        types.iter().all(|kind| kind == "full" || kind == "shared"),
        "GLM5.2 indexer_types must contain only full/shared entries"
    );
    Ok(())
}

fn ensure_fp8_quantization(json: &Value) -> Result<()> {
    let quant = json
        .get("quantization_config")
        .ok_or_else(|| anyhow::anyhow!("GLM5.2 config missing quantization_config"))?;
    ensure!(
        string_field(quant, "quant_method")? == "fp8",
        "GLM5.2 quantization_config.quant_method must be fp8"
    );
    ensure!(
        string_field(quant, "fmt")? == "e4m3",
        "GLM5.2 quantization_config.fmt must be e4m3"
    );
    ensure!(
        string_field(quant, "activation_scheme")? == "dynamic",
        "GLM5.2 quantization_config.activation_scheme must be dynamic"
    );
    let block = quant
        .get("weight_block_size")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("GLM5.2 quantization_config.weight_block_size missing"))?;
    ensure!(
        block.len() == 2 && block[0].as_u64() == Some(128) && block[1].as_u64() == Some(128),
        "GLM5.2 weight_block_size must be [128, 128]"
    );
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
