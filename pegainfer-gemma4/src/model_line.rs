//! Gemma 4's [`ModelLine`] implementation.

use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;

pub static MODEL_LINE: Gemma4Line = Gemma4Line;

pub struct Gemma4Line;

impl ModelLine for Gemma4Line {
    fn name(&self) -> &'static str {
        "Gemma 4"
    }

    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        crate::probe_config_json(config).map_err(|error| error.to_string())
    }

    fn consumed_shared_args(&self) -> &'static [&'static str] {
        &["device_ordinal", "cuda_graph"]
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        crate::start_engine(
            ctx.model_path,
            &EngineLoadOptions {
                enable_cuda_graph: ctx.shared.cuda_graph,
                device_ordinals: vec![ctx.shared.device_ordinal],
                ..EngineLoadOptions::default()
            },
        )
        .map(LaunchedEngine::Stepped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_accepts_12b_unified_identity() {
        let config: serde_json::Value = serde_json::from_str(
            r#"{"model_type":"gemma4_unified","architectures":["Gemma4UnifiedForConditionalGeneration"],"text_config":{"model_type":"gemma4_unified_text","head_dim":256,"global_head_dim":512,"sliding_window":1024,"attention_k_eq_v":true,"num_kv_shared_layers":0,"hidden_activation":"gelu_pytorch_tanh","enable_moe_block":false,"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"]}}"#,
        )
        .expect("fixture json");
        MODEL_LINE.probe(&config).expect("12B unified should probe");
    }

    #[test]
    fn probe_rejects_previous_generation_gemma3() {
        let config = serde_json::json!({
            "model_type": "gemma3",
            "architectures": ["Gemma3ForConditionalGeneration"],
            "text_config": {"model_type": "gemma3_text"}
        });
        let reason = MODEL_LINE.probe(&config).unwrap_err();
        assert!(reason.contains("gemma3"), "{reason}");
    }
}
