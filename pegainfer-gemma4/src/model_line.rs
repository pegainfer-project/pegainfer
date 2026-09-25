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
        crate::probe::probe_config_json(config).map_err(|error| error.to_string())
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
