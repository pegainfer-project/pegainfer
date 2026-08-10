//! Kimi-K2.6 text-only constants, config probing, and derived shapes.

// Most consumers live behind the kimi-k2 feature; the constants and shape
// helpers are intentionally kept compiling in feature-off builds.
#![cfg_attr(not(feature = "kimi-k2"), allow(dead_code))]

use std::fs;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use serde_json::Value;

pub(crate) const KIMI_K2_HIDDEN: usize = 7168;
pub(crate) const KIMI_K2_VOCAB: usize = 163_840;
pub const KIMI_K2_LAYERS: usize = 61;
pub(crate) const KIMI_K2_DENSE_LAYERS: usize = 1;
pub(crate) const KIMI_K2_MOE_LAYERS: usize = 60;
const KIMI_K2_MAX_CONTEXT: usize = 262_144;

pub(crate) const KIMI_K2_HEADS: usize = 64;
pub(crate) const KIMI_K2_Q_LORA_RANK: usize = 1536;
const KIMI_K2_KV_LORA_RANK: usize = 512;
pub(crate) const KIMI_K2_QK_NOPE_HEAD_DIM: usize = 128;
pub(crate) const KIMI_K2_QK_ROPE_HEAD_DIM: usize = 64;
pub(crate) const KIMI_K2_Q_HEAD_DIM: usize = KIMI_K2_QK_NOPE_HEAD_DIM + KIMI_K2_QK_ROPE_HEAD_DIM;
pub(crate) const KIMI_K2_V_HEAD_DIM: usize = 128;

pub(crate) const KIMI_K2_DENSE_INTERMEDIATE: usize = 18_432;
pub(crate) const KIMI_K2_EXPERT_INTERMEDIATE: usize = 2048;
pub(crate) const KIMI_K2_ROUTED_EXPERTS: usize = 384;
pub(crate) const KIMI_K2_TOPK: usize = 8;
const KIMI_K2_SHARED_EXPERTS: usize = 1;
pub(crate) const KIMI_K2_INT4_GROUP_SIZE: usize = 32;

pub(crate) const KIMI_K2_ROPE_THETA: f32 = 50_000.0;
pub(crate) const KIMI_K2_YARN_FACTOR: f32 = 64.0;
pub(crate) const KIMI_K2_YARN_ORIGINAL_MAX_POS: usize = 4096;
pub(crate) const KIMI_K2_YARN_BETA_FAST: f32 = 32.0;
pub(crate) const KIMI_K2_YARN_BETA_SLOW: f32 = 1.0;
const KIMI_K2_ROUTED_SCALING_FACTOR: f32 = 2.827;
pub(crate) const KIMI_K2_RMS_NORM_EPS: f32 = 1.0e-5;

/// Validate that `json` is a Kimi-K2.6 text config and that every dimension
/// matches the constants this crate is compiled against. This is a pure gate:
/// callers only care whether it succeeds, so nothing is materialized.
pub(crate) fn probe_config_json(json: &Value) -> Result<()> {
    let outer_model_type = string_field(json, "model_type")?;
    let text = if outer_model_type == "kimi_k25" {
        json.get("text_config")
            .ok_or_else(|| anyhow::anyhow!("Kimi outer config missing text_config"))?
    } else if outer_model_type == "kimi_k2" {
        json
    } else {
        bail!("not a Kimi-K2 config: model_type={outer_model_type}");
    };

    let text_model_type = string_field(text, "model_type")?;
    ensure!(
        text_model_type == "kimi_k2",
        "Kimi text_config.model_type must be kimi_k2, got {text_model_type}"
    );

    let hidden_size = usize_field(text, "hidden_size")?;
    let vocab_size = usize_field(text, "vocab_size")?;
    let num_hidden_layers = usize_field(text, "num_hidden_layers")?;
    let first_k_dense_replace = usize_field(text, "first_k_dense_replace")?;
    let max_position_embeddings = usize_field(text, "max_position_embeddings")?;
    let num_attention_heads = usize_field(text, "num_attention_heads")?;
    let q_lora_rank = usize_field(text, "q_lora_rank")?;
    let kv_lora_rank = usize_field(text, "kv_lora_rank")?;
    let qk_nope_head_dim = usize_field(text, "qk_nope_head_dim")?;
    let qk_rope_head_dim = usize_field(text, "qk_rope_head_dim")?;
    let v_head_dim = usize_field(text, "v_head_dim")?;
    let n_routed_experts = usize_field(text, "n_routed_experts")?;
    let num_experts_per_tok = usize_field(text, "num_experts_per_tok")?;
    let n_shared_experts = usize_field(text, "n_shared_experts")?;
    let moe_intermediate_size = usize_field(text, "moe_intermediate_size")?;
    let dense_intermediate_size = usize_field(text, "intermediate_size")?;
    let routed_scaling_factor = number_field(text, "routed_scaling_factor")?;
    let rms_norm_eps = number_field(text, "rms_norm_eps")?;
    let rope_theta = number_field(text, "rope_theta")?;
    ensure_float_close(
        routed_scaling_factor,
        f64::from(KIMI_K2_ROUTED_SCALING_FACTOR),
        1.0e-6,
        "routed_scaling_factor",
    )?;
    ensure_float_close(
        rope_theta,
        f64::from(KIMI_K2_ROPE_THETA),
        1.0e-6,
        "rope_theta",
    )?;
    ensure!(
        string_field(text, "topk_method")? == "noaux_tc",
        "Kimi topk_method must be noaux_tc"
    );
    ensure!(
        string_field(text, "scoring_func")? == "sigmoid",
        "Kimi scoring_func must be sigmoid"
    );
    ensure!(
        bool_field(text, "norm_topk_prob")?,
        "Kimi norm_topk_prob must be true"
    );
    ensure!(usize_field(text, "n_group")? == 1, "Kimi n_group must be 1");
    ensure!(
        usize_field(text, "topk_group")? == 1,
        "Kimi topk_group must be 1"
    );

    ensure!(
        hidden_size == KIMI_K2_HIDDEN,
        "hidden_size mismatch: {hidden_size}"
    );
    ensure!(
        vocab_size == KIMI_K2_VOCAB,
        "vocab_size mismatch: {vocab_size}"
    );
    ensure!(
        num_hidden_layers == KIMI_K2_LAYERS,
        "num_hidden_layers mismatch: {num_hidden_layers}"
    );
    ensure!(
        first_k_dense_replace == KIMI_K2_DENSE_LAYERS,
        "first_k_dense_replace mismatch: {first_k_dense_replace}"
    );
    ensure!(
        max_position_embeddings == KIMI_K2_MAX_CONTEXT,
        "max_position_embeddings mismatch: {max_position_embeddings}"
    );
    ensure!(
        num_attention_heads == KIMI_K2_HEADS,
        "num_attention_heads mismatch: {num_attention_heads}"
    );
    ensure!(
        q_lora_rank == KIMI_K2_Q_LORA_RANK,
        "q_lora_rank mismatch: {q_lora_rank}"
    );
    ensure!(
        kv_lora_rank == KIMI_K2_KV_LORA_RANK,
        "kv_lora_rank mismatch: {kv_lora_rank}"
    );
    ensure!(
        qk_nope_head_dim == KIMI_K2_QK_NOPE_HEAD_DIM,
        "qk_nope_head_dim mismatch: {qk_nope_head_dim}"
    );
    ensure!(
        qk_rope_head_dim == KIMI_K2_QK_ROPE_HEAD_DIM,
        "qk_rope_head_dim mismatch: {qk_rope_head_dim}"
    );
    ensure!(
        v_head_dim == KIMI_K2_V_HEAD_DIM,
        "v_head_dim mismatch: {v_head_dim}"
    );
    ensure!(
        n_routed_experts == KIMI_K2_ROUTED_EXPERTS,
        "n_routed_experts mismatch: {n_routed_experts}"
    );
    ensure!(
        num_experts_per_tok == KIMI_K2_TOPK,
        "num_experts_per_tok mismatch: {num_experts_per_tok}"
    );
    ensure!(
        n_shared_experts == KIMI_K2_SHARED_EXPERTS,
        "n_shared_experts mismatch: {n_shared_experts}"
    );
    ensure!(
        moe_intermediate_size == KIMI_K2_EXPERT_INTERMEDIATE,
        "moe_intermediate_size mismatch: {moe_intermediate_size}"
    );
    ensure!(
        dense_intermediate_size == KIMI_K2_DENSE_INTERMEDIATE,
        "intermediate_size mismatch: {dense_intermediate_size}"
    );
    ensure!(
        (rms_norm_eps - f64::from(KIMI_K2_RMS_NORM_EPS)).abs() < 1.0e-12,
        "rms_norm_eps mismatch: {rms_norm_eps}"
    );

    let rope_scaling = text
        .get("rope_scaling")
        .ok_or_else(|| anyhow::anyhow!("Kimi config missing rope_scaling"))?;
    let rope_scaling_type = rope_scaling
        .get("type")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    ensure!(
        rope_scaling_type.as_deref() == Some("yarn"),
        "Kimi rope_scaling.type must be yarn, got {:?}",
        rope_scaling_type
    );
    ensure_float_close(
        number_field(rope_scaling, "factor")?,
        f64::from(KIMI_K2_YARN_FACTOR),
        1.0e-6,
        "rope_scaling.factor",
    )?;
    ensure!(
        usize_field(rope_scaling, "original_max_position_embeddings")?
            == KIMI_K2_YARN_ORIGINAL_MAX_POS,
        "Kimi rope_scaling.original_max_position_embeddings mismatch"
    );
    ensure_float_close(
        number_field(rope_scaling, "beta_fast")?,
        f64::from(KIMI_K2_YARN_BETA_FAST),
        1.0e-6,
        "rope_scaling.beta_fast",
    )?;
    ensure_float_close(
        number_field(rope_scaling, "beta_slow")?,
        f64::from(KIMI_K2_YARN_BETA_SLOW),
        1.0e-6,
        "rope_scaling.beta_slow",
    )?;
    ensure_float_close(
        number_field(rope_scaling, "mscale")?,
        1.0,
        1.0e-12,
        "rope_scaling.mscale",
    )?;
    ensure_float_close(
        number_field(rope_scaling, "mscale_all_dim")?,
        1.0,
        1.0e-12,
        "rope_scaling.mscale_all_dim",
    )?;

    let quantization_config = text.get("quantization_config");
    let quant_method = quantization_config
        .and_then(|value| value.get("quant_method"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let quant_format = quantization_config
        .and_then(|value| value.get("format"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    ensure!(
        quant_method.as_deref() == Some("compressed-tensors"),
        "Kimi quantization_config.quant_method must be compressed-tensors, got {:?}",
        quant_method
    );
    ensure!(
        quant_format.as_deref() == Some("pack-quantized"),
        "Kimi quantization_config.format must be pack-quantized, got {:?}",
        quant_format
    );

    Ok(())
}

/// Stop-token IDs for EOS detection. `generation_config.json` is authoritative
/// (it may list several); `config.json`'s `eos_token_id` is the fallback. A
/// chat model without either cannot signal end-of-turn, so this fails instead
/// of letting every request silently run to `max_tokens`.
pub(crate) fn load_stop_token_ids(model_path: &Path) -> Result<Vec<u32>> {
    let stop_token_ids = read_stop_token_ids(model_path)?;
    log::info!("stop tokens: {stop_token_ids:?}");
    Ok(stop_token_ids)
}

fn read_stop_token_ids(model_path: &Path) -> Result<Vec<u32>> {
    let generation_config_path = model_path.join("generation_config.json");
    match fs::read_to_string(&generation_config_path) {
        Ok(content) => {
            let json: Value = serde_json::from_str(&content)
                .with_context(|| format!("parse {}", generation_config_path.display()))?;
            eos_token_ids(&json).ok_or_else(|| {
                anyhow::anyhow!(
                    "{} has no usable eos_token_id",
                    generation_config_path.display()
                )
            })
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            let config_path = model_path.join("config.json");
            let content = fs::read_to_string(&config_path)
                .with_context(|| format!("read {}", config_path.display()))?;
            let json: Value = serde_json::from_str(&content)
                .with_context(|| format!("parse {}", config_path.display()))?;
            // kimi_k25 wraps the text model config in text_config.
            eos_token_ids(&json)
                .or_else(|| json.get("text_config").and_then(eos_token_ids))
                .ok_or_else(|| {
                    anyhow::anyhow!("{} has no usable eos_token_id", config_path.display())
                })
        }
        Err(err) => Err(err).with_context(|| format!("read {}", generation_config_path.display())),
    }
}

fn eos_token_ids(json: &Value) -> Option<Vec<u32>> {
    let to_u32 = |value: &Value| value.as_u64().and_then(|id| u32::try_from(id).ok());
    match json.get("eos_token_id")? {
        Value::Array(ids) => {
            let mut ids = ids.iter().map(to_u32).collect::<Option<Vec<_>>>()?;
            ids.dedup();
            (!ids.is_empty()).then_some(ids)
        }
        id => to_u32(id).map(|id| vec![id]),
    }
}

fn string_field(json: &Value, key: &str) -> Result<String> {
    json.get(key)
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow::anyhow!("missing string field {key}"))
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

fn ensure_float_close(actual: f64, expected: f64, tolerance: f64, label: &str) -> Result<()> {
    ensure!(
        (actual - expected).abs() <= tolerance,
        "{label} mismatch: got {actual}, expected {expected}"
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct KimiK2ParallelShape {
    pub(crate) tp_world: usize,
    pub(crate) dp_world: usize,
    pub(crate) ep_world: usize,
    pub(crate) heads_per_tp: usize,
    pub(crate) local_experts: usize,
    pub(crate) vocab_per_tp: usize,
}

/// TP-local tensor dimensions derived from `KimiK2ParallelShape`.
/// All fields scale with `heads_per_tp` or `tp_world`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct KimiLocalDims {
    pub(crate) local_heads: usize,
    pub(crate) q_proj_out: usize,
    pub(crate) o_proj_in: usize,
    pub(crate) q_nope_out: usize,
    pub(crate) q_pe_out: usize,
    pub(crate) abs_q_out: usize,
    pub(crate) dense_gate_up: usize,
    pub(crate) dense_activated: usize,
    pub(crate) shared_gate_up: usize,
    pub(crate) shared_activated: usize,
}

impl KimiK2ParallelShape {
    #[must_use]
    pub(crate) fn tp8_ep8() -> Self {
        Self::new(8, 1)
    }

    #[must_use]
    pub(crate) fn tp1_dp8() -> Self {
        Self::new(1, 8)
    }

    #[must_use]
    fn new(tp_world: usize, dp_world: usize) -> Self {
        let ep_world = tp_world * dp_world;
        Self {
            tp_world,
            dp_world,
            ep_world,
            heads_per_tp: KIMI_K2_HEADS / tp_world,
            local_experts: KIMI_K2_ROUTED_EXPERTS / ep_world,
            vocab_per_tp: KIMI_K2_VOCAB / tp_world,
        }
    }

    #[must_use]
    pub(crate) fn local_dims(&self) -> KimiLocalDims {
        let h = self.heads_per_tp;
        KimiLocalDims {
            local_heads: h,
            q_proj_out: h * KIMI_K2_Q_HEAD_DIM,
            o_proj_in: h * KIMI_K2_V_HEAD_DIM,
            q_nope_out: h * KIMI_K2_QK_NOPE_HEAD_DIM,
            q_pe_out: h * KIMI_K2_QK_ROPE_HEAD_DIM,
            abs_q_out: h * KIMI_K2_KV_LORA_RANK,
            dense_gate_up: 2 * KIMI_K2_DENSE_INTERMEDIATE / self.tp_world,
            dense_activated: KIMI_K2_DENSE_INTERMEDIATE / self.tp_world,
            shared_gate_up: 2 * KIMI_K2_EXPERT_INTERMEDIATE / self.tp_world,
            shared_activated: KIMI_K2_EXPERT_INTERMEDIATE / self.tp_world,
        }
    }

    #[must_use]
    pub(crate) fn parallel_config(&self) -> pegainfer_frontend::parallel::ParallelConfig {
        pegainfer_frontend::parallel::ParallelConfig::new(self.tp_world, self.dp_world)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::eos_token_ids;

    #[test]
    fn eos_token_ids_accepts_single_and_array() {
        assert_eq!(
            eos_token_ids(&json!({"eos_token_id": 163_586})),
            Some(vec![163_586])
        );
        assert_eq!(
            eos_token_ids(&json!({"eos_token_id": [163_585, 163_586, 163_586]})),
            Some(vec![163_585, 163_586])
        );
    }

    #[test]
    fn eos_token_ids_rejects_missing_or_malformed() {
        assert_eq!(eos_token_ids(&json!({})), None);
        assert_eq!(eos_token_ids(&json!({"eos_token_id": []})), None);
        assert_eq!(eos_token_ids(&json!({"eos_token_id": "eos"})), None);
        assert_eq!(eos_token_ids(&json!({"eos_token_id": [163_585, -1]})), None);
    }
}
