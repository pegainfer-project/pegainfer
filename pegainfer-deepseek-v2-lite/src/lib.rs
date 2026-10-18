#[cfg(feature = "deepseek-v2-lite")]
mod attribution;
mod config;
#[cfg(feature = "deepseek-v2-lite")]
mod device;
#[cfg(feature = "deepseek-v2-lite")]
mod engine;
mod ep;
#[cfg(feature = "deepseek-v2-lite")]
mod host_ops;
#[cfg(feature = "deepseek-v2-lite")]
mod model;
pub mod model_line;
#[cfg(feature = "deepseek-v2-lite")]
mod nccl_backend;
#[cfg(feature = "deepseek-v2-lite")]
mod runtime;
#[cfg(feature = "deepseek-v2-lite")]
mod scheduler;
#[cfg(feature = "deepseek-v2-lite")]
mod weights;

use std::path::Path;

use anyhow::Result;
#[cfg(feature = "deepseek-v2-lite")]
pub use attribution::CallSiteRollup;
#[cfg(feature = "deepseek-v2-lite")]
pub use attribution::DecodeAttributionProfile;
#[cfg(feature = "deepseek-v2-lite")]
pub use attribution::SectionRollup;
#[cfg(feature = "deepseek-v2-lite")]
pub use attribution::SectionSample;
pub use config::Config;
use config::SUPPORTED_HIDDEN_SIZE;
use ep::SUPPORTED_ROUTED_EXPERTS;
use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::EngineLoadOptions;
#[cfg(feature = "deepseek-v2-lite")]
use pegainfer_frontend::engine::EpBackend;
#[cfg(feature = "deepseek-v2-lite")]
pub use runtime::BatchedGenerationResult;
#[cfg(feature = "deepseek-v2-lite")]
pub use runtime::DecodeGraphReadinessReport;
#[cfg(feature = "deepseek-v2-lite")]
pub use runtime::DeepSeekV2LiteEp2Generator;
#[cfg(feature = "deepseek-v2-lite")]
pub use runtime::GenerationResult;
#[cfg(feature = "deepseek-v2-lite")]
pub use runtime::GenerationStats;

fn probe_config_json(json: &serde_json::Value) -> Result<bool> {
    let Some(model_type) = json.get("model_type").and_then(serde_json::Value::as_str) else {
        return Ok(false);
    };
    if model_type != "deepseek_v2" {
        return Ok(false);
    }
    let n_routed_experts = json
        .get("n_routed_experts")
        .and_then(serde_json::Value::as_u64);
    let hidden_size = json.get("hidden_size").and_then(serde_json::Value::as_u64);
    let is_lite = n_routed_experts.is_some_and(|value| value == SUPPORTED_ROUTED_EXPERTS as u64)
        && hidden_size.is_some_and(|value| value == SUPPORTED_HIDDEN_SIZE as u64);
    if !is_lite {
        anyhow::bail!(
            "unsupported DeepSeek-V2 config: DeepSeek-V2-Lite first gate expects hidden_size={} and n_routed_experts={}, got hidden_size={:?}, n_routed_experts={:?}",
            SUPPORTED_HIDDEN_SIZE,
            SUPPORTED_ROUTED_EXPERTS,
            hidden_size,
            n_routed_experts
        );
    }
    Ok(true)
}

#[cfg(feature = "deepseek-v2-lite")]
pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    engine::start_engine(model_path, options)
}

#[cfg(not(feature = "deepseek-v2-lite"))]
pub fn start_engine(_model_path: &Path, _options: EngineLoadOptions) -> Result<EngineHandle> {
    anyhow::bail!(
        "DeepSeek-V2-Lite runtime is feature-gated; rebuild with --features deepseek-v2-lite"
    )
}

/// Start the DeepSeek-V2-Lite engine for the server. The binary forwards the
/// user's `cuda_graph` request uniformly; whether to honor it is the model's
/// call. The server EP=2 path does not enable CUDA Graph capture, so it ignores
/// the request (warning if one came in). The diagnostic decode graph probe lives
/// in the attribution gate. The EP=2 topology (devices `0..1`) is fixed by the
/// model.
#[cfg(feature = "deepseek-v2-lite")]
fn launch(model_path: &Path, cuda_graph: bool) -> Result<EngineHandle> {
    if cuda_graph {
        log::warn!("DeepSeek V2 Lite does not support CUDA Graph; ignoring --cuda-graph=true");
    }
    engine::start_engine(
        model_path,
        EngineLoadOptions {
            enable_cuda_graph: false,
            device_ordinals: vec![0, 1],
            parallel_config: None,
            ep_backend: EpBackend::Nccl,
            seed: 42,
        },
    )
}

#[cfg(not(feature = "deepseek-v2-lite"))]
pub fn launch(_model_path: &Path, _cuda_graph: bool) -> Result<EngineHandle> {
    anyhow::bail!(
        "DeepSeek-V2-Lite runtime is feature-gated; rebuild with --features deepseek-v2-lite"
    )
}
