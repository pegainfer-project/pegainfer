// The config and the manifest are CUDA-free so they test without a device, but
// only the loader consumes them, so a gate-off library does not carry them.
#[cfg(any(feature = "gemma4", test))]
mod config;
#[cfg(any(feature = "gemma4", test))]
mod manifest;
pub mod model_line;
#[cfg(any(feature = "gemma4", test))]
mod nvfp4;
mod probe;
// The engine is the live consumer; the oracles reach the rest, so an
// `expect(dead_code)` cannot hold in every build.
#[cfg(feature = "gemma4")]
mod engine;
#[cfg(feature = "gemma4")]
mod forward;
#[cfg(feature = "gemma4")]
mod green_ctx;
#[cfg(feature = "gemma4")]
mod kv;
#[cfg(feature = "gemma4")]
mod layer;
#[cfg(feature = "gemma4")]
mod moe;
#[cfg(feature = "gemma4")]
mod prefix_cache;
#[cfg(feature = "gemma4")]
mod serve;
#[cfg(all(test, feature = "gemma4"))]
mod testkit;
#[cfg(feature = "gemma4")]
mod weights;

use std::path::Path;

use anyhow::Result;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineLoadOptions;
pub(crate) use probe::probe_config_json;

#[cfg(feature = "gemma4")]
fn start_engine(model_path: &Path, options: &EngineLoadOptions) -> Result<Engine> {
    engine::start(model_path, options)
}

#[cfg(not(feature = "gemma4"))]
fn start_engine(_model_path: &Path, _options: &EngineLoadOptions) -> Result<Engine> {
    anyhow::bail!(
        "Gemma 4 support is feature-gated; rebuild pegainfer-server with --features gemma4"
    )
}
