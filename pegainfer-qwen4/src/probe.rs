//! Identity check: the outer `model_type`, `architectures` and the inner
//! `text_config.model_type` must all name `qwen4_exp`. Every structural rule
//! lives in [`crate::config`], behind one typed parse.

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;

/// The identity this line claims: the outer `model_type` and the inner
/// `text_config.model_type` the probe requires together.
///
/// Exposed because `pegainfer-server`'s `feature_gate_hint` hand-mirrors each
/// line's identities in a table that must compile when this crate does not; a
/// test under this feature binds the mirror to these constants so the two cannot
/// drift apart silently.
pub const MODEL_TYPE: &str = "qwen4_exp";
pub const TEXT_MODEL_TYPE: &str = "qwen4_exp_text";

const ARCHITECTURE: &str = "Qwen4ExpForConditionalGeneration";

pub(crate) fn probe_config_json(json: &serde_json::Value) -> Result<()> {
    let model_type = json
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    if model_type != MODEL_TYPE {
        bail!("not a qwen4_exp config: model_type={model_type}");
    }
    let has_arch = json
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|arr| arr.iter().any(|v| v.as_str() == Some(ARCHITECTURE)));
    ensure!(
        has_arch,
        "qwen4_exp: architectures must contain {ARCHITECTURE}"
    );
    let text_type = json
        .get("text_config")
        .and_then(|text| text.get("model_type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    ensure!(
        text_type == TEXT_MODEL_TYPE,
        "qwen4_exp: text_config.model_type is {text_type}, expected {TEXT_MODEL_TYPE} — \
         cross-family mismatch"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frozen() -> serde_json::Value {
        serde_json::from_str(include_str!("../tests/frozen_config.json")).unwrap()
    }

    fn err(mutate: impl FnOnce(&mut serde_json::Value)) -> String {
        let mut cfg = frozen();
        mutate(&mut cfg);
        probe_config_json(&cfg).unwrap_err().to_string()
    }

    /// The registry picks a *unique* claimant, so this line accepting another
    /// family's identity would be a routing regression in that family.
    #[test]
    fn no_other_lines_identity_is_accepted() {
        for model_type in [
            "qwen3",
            "qwen3_5",
            "qwen4",
            "gemma3",
            "gemma4",
            "gemma4_unified",
            "glm_moe_dsa",
            "kimi_k2",
            "kimi_k25",
            "kimi_k3",
            "deepseek_v2",
            // The inner identity alone must not be enough: a config that carries
            // the text type but a foreign outer type is a cross-family mismatch.
            "qwen4_exp_text",
            "",
        ] {
            let cfg = serde_json::json!({
                "model_type": model_type,
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "text_config": {"model_type": "qwen4_exp_text"}
            });
            let err = match probe_config_json(&cfg) {
                Ok(()) => panic!("{model_type} must not be claimed by this line"),
                Err(error) => error.to_string(),
            };
            assert!(
                err.contains("not a qwen4_exp config"),
                "{model_type}: {err}"
            );
        }
    }

    #[test]
    fn a_cross_family_text_config_bails() {
        let err = err(|c| c["text_config"]["model_type"] = serde_json::json!("qwen3_5_text"));
        assert!(err.contains("cross-family mismatch"), "{err}");
    }

    #[test]
    fn a_missing_architecture_bails() {
        let err = err(|c| c["architectures"] = serde_json::json!(["Qwen4ExpForCausalLM"]));
        assert!(err.contains("architectures"), "{err}");
    }
}
