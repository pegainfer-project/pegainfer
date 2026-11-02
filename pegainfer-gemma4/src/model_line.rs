//! Gemma 4's [`ModelLine`] implementation.

use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;
use pegainfer_frontend::parallel::ParallelConfig;

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
        &["device_ordinal", "cuda_graph", "tp_size"]
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        // One tensor-parallel rank per device: `--tp-size=2` on two L20s is
        // the whole device list.
        let tp = ctx.shared.tp_size;
        let device_ordinals: Vec<usize> = if tp <= 1 {
            vec![ctx.shared.device_ordinal]
        } else {
            (0..tp).collect()
        };
        let parallel_config = (tp > 1).then(|| ParallelConfig::new(tp, 1));
        crate::start_engine(
            ctx.model_path,
            &EngineLoadOptions {
                enable_cuda_graph: ctx.shared.cuda_graph,
                device_ordinals,
                parallel_config,
                ..EngineLoadOptions::default()
            },
        )
        .map(LaunchedEngine::Stepped)
    }
}
