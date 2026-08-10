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

use backend::EpBackendRuntime;
pub use types::BatchedGenerationResult;
pub use types::DecodeGraphReadinessReport;
pub use types::GenerationResult;
pub use types::GenerationStats;

use crate::Config;
use crate::model::DriverRankModel;
use crate::model::ExpertRankModel;

pub struct DeepSeekV2LiteEp2Generator {
    device_ordinals: Vec<usize>,
    config: Config,
    rank0: DriverRankModel,
    rank1: ExpertRankModel,
    backend: EpBackendRuntime,
}

// SAFETY: The generator is driven by exactly one worker thread after load. It
// switches CUDA devices explicitly before every rank-local op and recreates the
// thread-local cuBLAS handle when the active device changes.
unsafe impl Send for DeepSeekV2LiteEp2Generator {}
