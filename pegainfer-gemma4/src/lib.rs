// The config and the manifest are CUDA-free so they test without a device, but
// only the loader consumes them, so a gate-off library does not carry them.
#[cfg(any(feature = "gemma4", test))]
mod config;
#[cfg(any(feature = "gemma4", test))]
mod manifest;
pub mod model_line;
mod probe;
// Dead until an executor calls in; test targets do use them, so an
// `expect(dead_code)` cannot hold in every build.
#[cfg(feature = "gemma4")]
#[allow(dead_code)]
mod layer;
#[cfg(all(feature = "gemma4", test))]
mod testkit;
#[cfg(feature = "gemma4")]
#[allow(dead_code)]
mod weights;

use std::path::Path;

use anyhow::Result;
use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::EngineLoadOptions;
pub(crate) use probe::probe_config_json;

#[cfg(feature = "gemma4")]
fn start_engine(_model_path: &Path, _options: EngineLoadOptions) -> Result<EngineHandle> {
    anyhow::bail!("Gemma 4 engine is not implemented yet (registration only)")
}

#[cfg(not(feature = "gemma4"))]
pub fn start_engine(_model_path: &Path, _options: EngineLoadOptions) -> Result<EngineHandle> {
    anyhow::bail!(
        "Gemma 4 support is feature-gated; rebuild pegainfer-server with --features gemma4"
    )
}
