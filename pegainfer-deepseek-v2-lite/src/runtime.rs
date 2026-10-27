mod backend;
mod generation;
mod graph_probe;
mod helpers;
mod layers;
mod moe;
mod readiness;
mod routing;
#[cfg(test)]
mod tests;
mod types;

use std::env;
use std::sync::Mutex;

use anyhow::Result;
use backend::EpBackendRuntime;
pub use types::BatchedGenerationResult;
pub use types::DecodeGraphReadinessReport;
pub use types::GenerationResult;
pub use types::GenerationStats;

use crate::Config;
use crate::device_kv::DeviceKvPool;
use crate::model::DriverRankModel;
use crate::model::ExpertRankModel;

pub struct DeepSeekV2LiteEp2Generator {
    device_ordinals: Vec<usize>,
    config: Config,
    device_routed_moe: Option<Mutex<moe::DeviceRoutedMoeRuntime>>,
    device_kv: Option<Mutex<DeviceKvPool>>,
    rank0: DriverRankModel,
    rank1: ExpertRankModel,
    backend: EpBackendRuntime,
}

// SAFETY: The generator is driven by exactly one worker thread after load. It
// switches CUDA devices explicitly before every rank-local op and recreates the
// thread-local cuBLAS handle when the active device changes.
unsafe impl Send for DeepSeekV2LiteEp2Generator {}

fn parse_rollback_value(
    name: &str,
    raw: Option<&str>,
    optimized: &str,
    rollback: &str,
) -> std::result::Result<bool, String> {
    match raw.map(str::trim) {
        None | Some("") => Ok(false),
        Some(value) if value == optimized => Ok(false),
        Some(value) if value == rollback => Ok(true),
        Some(value) => Err(format!(
            "{name} must be `{optimized}` or `{rollback}`, got `{value}`"
        )),
    }
}

fn load_rollback_value(
    name: &str,
    optimized: &str,
    rollback: &str,
) -> std::result::Result<bool, String> {
    match env::var(name) {
        Ok(value) => parse_rollback_value(name, Some(&value), optimized, rollback),
        Err(env::VarError::NotPresent) => Ok(false),
        Err(env::VarError::NotUnicode(_)) => Err(format!("{name} must contain valid Unicode text")),
    }
}

fn rollback_enabled(value: &std::result::Result<bool, String>) -> Result<bool> {
    value
        .as_ref()
        .copied()
        .map_err(|message| anyhow::anyhow!("{message}"))
}
