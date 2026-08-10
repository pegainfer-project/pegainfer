// Fail-closed config probe: identity plus the structural facts shared by every
// published size; no size pinning beyond the single published MoE configuration.
use anyhow::Result;
use anyhow::bail;

const GEMMA4_LOCAL_HEAD_DIM: u64 = 256;
const GEMMA4_GLOBAL_HEAD_DIM: u64 = 512;
const GEMMA4_SLIDING_WINDOW: u64 = 1024;
const GEMMA4_GLOBAL_LAYER_PERIOD: usize = 6;
const GEMMA4_MOE_NUM_EXPERTS: u64 = 128;
const GEMMA4_MOE_TOP_K: u64 = 8;
const GEMMA4_MOE_INTERMEDIATE: u64 = 704;

pub(crate) fn probe_config_json(json: &serde_json::Value) -> Result<()> {
    let (family, text_config) = probe_identity(json)?;
    probe_common_text(family, text_config)?;
    probe_layer_types(family, text_config)?;
    probe_moe(family, text_config)
}

fn probe_identity(json: &serde_json::Value) -> Result<(&'static str, &serde_json::Value)> {
    let outer_type = json
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let (family, text_model_type, arch_class) = match outer_type {
        "gemma4" => ("gemma4", "gemma4_text", "Gemma4ForConditionalGeneration"),
        "gemma4_unified" => (
            "gemma4_unified",
            "gemma4_unified_text",
            "Gemma4UnifiedForConditionalGeneration",
        ),
        unknown => bail!("not a Gemma 4 config: model_type={unknown}"),
    };

    let architectures = json
        .get("architectures")
        .and_then(serde_json::Value::as_array);
    let has_arch = architectures.is_some_and(|arr| {
        arr.iter()
            .any(|v| v.as_str().is_some_and(|s| s == arch_class))
    });
    if !has_arch {
        bail!("Gemma 4 {family}: architectures must contain {arch_class}");
    }

    let text_config = json
        .get("text_config")
        .ok_or_else(|| anyhow::anyhow!("Gemma 4 {family}: missing text_config"))?;

    let text_type = text_config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if text_type != text_model_type {
        bail!(
            "Gemma 4 {family}: text_config.model_type is {text_type}, expected {text_model_type} — \
             cross-family mismatch"
        );
    }
    Ok((family, text_config))
}

fn probe_common_text(family: &str, text_config: &serde_json::Value) -> Result<()> {
    let e4b_window = text_config
        .get("sliding_window")
        .and_then(serde_json::Value::as_u64)
        == Some(512);
    let e4b_k_neq_v = text_config
        .get("attention_k_eq_v")
        .and_then(serde_json::Value::as_bool)
        == Some(false);
    if e4b_window && e4b_k_neq_v {
        bail!(
            "Gemma 4 {family}: this is an E4B edge-series configuration (sliding_window 512, \
             attention_k_eq_v false, shared KV layers), which this model line does not support; \
             supported configurations are the 12B/26B/31B ones"
        );
    }

    let head_dim = text_config
        .get("head_dim")
        .and_then(serde_json::Value::as_u64);
    if head_dim != Some(GEMMA4_LOCAL_HEAD_DIM) {
        bail!("Gemma 4 {family}: head_dim must be {GEMMA4_LOCAL_HEAD_DIM}, got {head_dim:?}");
    }
    let global_head_dim = text_config
        .get("global_head_dim")
        .and_then(serde_json::Value::as_u64);
    if global_head_dim != Some(GEMMA4_GLOBAL_HEAD_DIM) {
        bail!(
            "Gemma 4 {family}: global_head_dim must be {GEMMA4_GLOBAL_HEAD_DIM}, got {global_head_dim:?}"
        );
    }
    let sliding_window = text_config
        .get("sliding_window")
        .and_then(serde_json::Value::as_u64);
    if sliding_window != Some(GEMMA4_SLIDING_WINDOW) {
        bail!(
            "Gemma 4 {family}: sliding_window must be {GEMMA4_SLIDING_WINDOW}, got {sliding_window:?}"
        );
    }
    let attn_k_eq_v = text_config
        .get("attention_k_eq_v")
        .and_then(serde_json::Value::as_bool);
    if attn_k_eq_v != Some(true) {
        bail!("Gemma 4 {family}: attention_k_eq_v must be true, got {attn_k_eq_v:?}");
    }
    let num_kv_shared = text_config
        .get("num_kv_shared_layers")
        .and_then(serde_json::Value::as_u64);
    if num_kv_shared != Some(0) {
        bail!("Gemma 4 {family}: num_kv_shared_layers must be 0, got {num_kv_shared:?}");
    }
    let hidden_activation = text_config
        .get("hidden_activation")
        .and_then(serde_json::Value::as_str);
    if hidden_activation != Some("gelu_pytorch_tanh") {
        bail!(
            "Gemma 4 {family}: hidden_activation must be gelu_pytorch_tanh, got {hidden_activation:?}"
        );
    }
    Ok(())
}

fn probe_layer_types(family: &str, text_config: &serde_json::Value) -> Result<()> {
    let layer_types = text_config
        .get("layer_types")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("Gemma 4 {family}: missing layer_types"))?;
    if layer_types.is_empty() {
        bail!("Gemma 4 {family}: layer_types is empty");
    }
    let last_idx = layer_types.len() - 1;
    for (i, entry) in layer_types.iter().enumerate() {
        let s = entry.as_str().unwrap_or("");
        if s != "sliding_attention" && s != "full_attention" {
            bail!(
                "Gemma 4 {family}: layer_types[{i}] is {s:?}, must be sliding_attention or full_attention"
            );
        }
        let is_full = s == "full_attention";
        let expected_full =
            i % GEMMA4_GLOBAL_LAYER_PERIOD == GEMMA4_GLOBAL_LAYER_PERIOD - 1 || i == last_idx;
        if is_full != expected_full {
            bail!(
                "Gemma 4 {family}: layer_types[{i}] is {s:?}, expected {}",
                if expected_full {
                    "full_attention"
                } else {
                    "sliding_attention"
                }
            );
        }
    }
    Ok(())
}

fn probe_moe(family: &str, text_config: &serde_json::Value) -> Result<()> {
    let enable_moe = text_config.get("enable_moe_block");
    let Some(enable_moe) = enable_moe.and_then(serde_json::Value::as_bool) else {
        bail!("Gemma 4 {family}: enable_moe_block must be an explicit boolean, got {enable_moe:?}");
    };
    if enable_moe {
        let num_experts = text_config
            .get("num_experts")
            .and_then(serde_json::Value::as_u64);
        if num_experts != Some(GEMMA4_MOE_NUM_EXPERTS) {
            bail!(
                "Gemma 4 {family}: MoE enabled but num_experts is {num_experts:?}, expected {GEMMA4_MOE_NUM_EXPERTS}"
            );
        }
        let top_k = text_config
            .get("top_k_experts")
            .and_then(serde_json::Value::as_u64);
        if top_k != Some(GEMMA4_MOE_TOP_K) {
            bail!(
                "Gemma 4 {family}: MoE enabled but top_k_experts is {top_k:?}, expected {GEMMA4_MOE_TOP_K}"
            );
        }
        let moe_intermediate = text_config
            .get("moe_intermediate_size")
            .and_then(serde_json::Value::as_u64);
        if moe_intermediate != Some(GEMMA4_MOE_INTERMEDIATE) {
            bail!(
                "Gemma 4 {family}: MoE enabled but moe_intermediate_size is {moe_intermediate:?}, expected {GEMMA4_MOE_INTERMEDIATE}"
            );
        }
    } else {
        for field in ["num_experts", "top_k_experts", "moe_intermediate_size"] {
            if text_config.get(field).is_some_and(|v| !v.is_null()) {
                bail!("Gemma 4 {family}: MoE disabled but {field} is present and non-null");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good_12b_config() -> serde_json::Value {
        serde_json::json!({
            "model_type": "gemma4_unified",
            "architectures": ["Gemma4UnifiedForConditionalGeneration"],
            "text_config": {
                "model_type": "gemma4_unified_text",
                "head_dim": 256,
                "global_head_dim": 512,
                "sliding_window": 1024,
                "attention_k_eq_v": true,
                "num_kv_shared_layers": 0,
                "hidden_activation": "gelu_pytorch_tanh",
                "enable_moe_block": false,
                "layer_types": [
                    "sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "sliding_attention", "full_attention",
                    "sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "sliding_attention", "full_attention"
                ]
            }
        })
    }

    fn good_26b_config() -> serde_json::Value {
        let mut cfg = good_12b_config();
        cfg["model_type"] = serde_json::json!("gemma4");
        cfg["architectures"] = serde_json::json!(["Gemma4ForConditionalGeneration"]);
        let tc = &mut cfg["text_config"];
        tc["model_type"] = serde_json::json!("gemma4_text");
        tc["enable_moe_block"] = serde_json::json!(true);
        tc["num_experts"] = serde_json::json!(128);
        tc["top_k_experts"] = serde_json::json!(8);
        tc["moe_intermediate_size"] = serde_json::json!(704);
        tc["intermediate_size"] = serde_json::json!(2112);
        cfg
    }

    #[test]
    fn good_12b_passes() {
        probe_config_json(&good_12b_config()).unwrap();
    }

    #[test]
    fn good_26b_passes() {
        probe_config_json(&good_26b_config()).unwrap();
    }

    #[test]
    fn moe_width_2112_bails() {
        let mut cfg = good_26b_config();
        cfg["text_config"]["moe_intermediate_size"] = serde_json::json!(2112);
        let err = probe_config_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("moe_intermediate_size"), "{err}");
    }

    #[test]
    fn missing_enable_moe_block_bails() {
        let mut cfg = good_12b_config();
        cfg["text_config"]
            .as_object_mut()
            .unwrap()
            .remove("enable_moe_block");
        let err = probe_config_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("enable_moe_block"), "{err}");
    }

    #[test]
    fn moe_disabled_with_num_experts_bails() {
        let mut cfg = good_12b_config();
        cfg["text_config"]["num_experts"] = serde_json::json!(128);
        let err = probe_config_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("num_experts"), "{err}");
    }

    #[test]
    fn moe_top_k_as_string_bails() {
        let mut cfg = good_26b_config();
        cfg["text_config"]["top_k_experts"] = serde_json::json!("8");
        let err = probe_config_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("top_k_experts"), "{err}");
    }

    #[test]
    fn e4b_edge_series_rejected_by_name() {
        let mut cfg = good_12b_config();
        cfg["model_type"] = serde_json::json!("gemma4");
        cfg["architectures"] = serde_json::json!(["Gemma4ForConditionalGeneration"]);
        let tc = &mut cfg["text_config"];
        tc["model_type"] = serde_json::json!("gemma4_text");
        tc["sliding_window"] = serde_json::json!(512);
        tc["attention_k_eq_v"] = serde_json::json!(false);
        tc["num_kv_shared_layers"] = serde_json::json!(18);
        let err = probe_config_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("E4B"), "{err}");
    }

    #[test]
    fn wrong_layer_at_non_6th_index_bails() {
        let mut cfg = good_12b_config();
        cfg["text_config"]["layer_types"][3] = serde_json::json!("full_attention");
        let err = probe_config_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("layer_types[3]"), "{err}");
    }

    #[test]
    fn cross_family_mismatch_bails() {
        let mut cfg = good_12b_config();
        cfg["architectures"] = serde_json::json!(["Gemma4ForConditionalGeneration"]);
        let err = probe_config_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("architectures"), "{err}");
    }

    #[test]
    fn moe_enabled_without_num_experts_bails() {
        let mut cfg = good_12b_config();
        cfg["text_config"]["enable_moe_block"] = serde_json::json!(true);
        let err = probe_config_json(&cfg).unwrap_err().to_string();
        assert!(err.contains("num_experts"), "{err}");
    }
}
