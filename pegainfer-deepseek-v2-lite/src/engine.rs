use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use log::info;
use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::EngineLoadOptions;
use tokio::sync::mpsc;

use crate::runtime::DeepSeekV2LiteEp2Generator;
use crate::scheduler::MixedRequestScheduler;

pub(crate) fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    let started = Instant::now();
    info!("starting DeepSeek-V2-Lite EP2 engine");
    let generator = DeepSeekV2LiteEp2Generator::load(model_path, options)?;
    let servable_len = generator.config().supported_plain_rope_context() as u32;
    let (submit_tx, submit_rx) = mpsc::unbounded_channel();

    let join_handle = std::thread::Builder::new()
        .name("deepseek-v2-lite-ep2".to_string())
        .spawn(move || MixedRequestScheduler::new(generator, submit_rx).run())
        .context("spawn DeepSeek-V2-Lite EP=2 engine thread")?;

    info!(
        "DeepSeek-V2-Lite EP2 engine started cost {:.2}s",
        started.elapsed().as_secs_f64()
    );
    Ok(EngineHandle::new_with_join_handle(submit_tx, join_handle).with_servable_len(servable_len))
}
