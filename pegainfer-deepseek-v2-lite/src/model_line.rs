//! DeepSeek-V2-Lite's [`ModelLine`] implementation.

use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;

pub static MODEL_LINE: DeepSeekV2LiteLine = DeepSeekV2LiteLine;

pub struct DeepSeekV2LiteLine;

impl ModelLine for DeepSeekV2LiteLine {
    fn name(&self) -> &'static str {
        "DeepSeek-V2-Lite"
    }

    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        match crate::probe_config_json(config) {
            Ok(true) => Ok(()),
            Ok(false) => Err(format!(
                "model_type {:?} is not \"deepseek_v2\"",
                config.get("model_type").and_then(serde_json::Value::as_str)
            )),
            Err(error) => Err(error.to_string()),
        }
    }

    fn consumed_shared_args(&self) -> &'static [&'static str] {
        &["cuda_graph"]
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        crate::launch(ctx.model_path, ctx.shared.cuda_graph).map(LaunchedEngine::Handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_rejects_foreign_model_type() {
        let config = serde_json::json!({"model_type": "qwen3"});
        let reason = MODEL_LINE.probe(&config).unwrap_err();
        assert!(reason.contains("deepseek_v2"), "{reason}");
    }

    #[test]
    fn probe_rejects_non_lite_shape_with_the_real_reason() {
        // model_type matches but the shape gate fails: the reason must name
        // the shape mismatch, not claim a foreign model_type.
        let config = serde_json::json!({
            "model_type": "deepseek_v2",
            "n_routed_experts": 1,
            "hidden_size": 1
        });
        let reason = MODEL_LINE.probe(&config).unwrap_err();
        assert!(
            reason.contains("unsupported DeepSeek-V2 config"),
            "{reason}"
        );
    }
}
