//! Text-only Kimi-K2.6 model crate.
//!
//! The current crate stage owns the compile-checked operator API surface and
//! text-only config probing. CUDA/runtime bodies land behind these headers.

#![allow(incomplete_features)]
// `use super::*` is the flat-module-layout idiom for these tightly-coupled
// submodules (weights.rs + weights/*, runner/worker.rs + worker/*), and the
// safetensors-mirroring field names (gate_proj, weight_packed, …) read clearer
// with their shared affix than without. Both are pedantic-only lints.
#![allow(clippy::wildcard_imports, clippy::struct_field_names)]
#![feature(generic_const_exprs)]

use std::path::Path;

use anyhow::Result;
use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::EngineLoadOptions;
#[cfg(feature = "kimi-k2")]
use pegainfer_frontend::engine::EpBackend;

#[cfg(feature = "kimi-k2")]
pub mod batch_decode_trace;
pub(crate) mod config;
#[cfg(feature = "kernel-report")]
pub mod kernel_report;
#[cfg(feature = "kimi-k2")]
pub mod model_line;
#[cfg(feature = "kimi-k2")]
mod runner;
#[cfg(feature = "kimi-k2")]
mod typed_scratch;
#[cfg(feature = "kimi-k2")]
mod weights;

pub use config::KIMI_K2_LAYERS;

#[cfg(feature = "kimi-k2")]
#[allow(clippy::needless_pass_by_value)]
pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<EngineHandle> {
    runner::start_engine(model_path, &options)
}

#[cfg(not(feature = "kimi-k2"))]
pub fn start_engine(_model_path: &Path, _options: EngineLoadOptions) -> Result<EngineHandle> {
    anyhow::bail!("Kimi-K2 runtime is feature-gated; rebuild with --features kimi-k2")
}

/// Server-facing launch knobs for Kimi-K2. The binary passes raw CLI values;
/// [`launch`] owns the EP topology policy — validating TP/DP, deriving the EP
/// world and its device ordinals — so the server never hardcodes Kimi's
/// parallel layout.
#[cfg(feature = "kimi-k2")]
#[derive(Clone, Copy, Debug)]
pub struct KimiLaunchOptions {
    tp_size: usize,
    dp_size: usize,
    ep_backend: EpBackend,
    cuda_graph: bool,
}

/// Start the Kimi-K2 engine from server-facing [`KimiLaunchOptions`].
#[cfg(feature = "kimi-k2")]
fn launch(model_path: &Path, options: KimiLaunchOptions) -> Result<EngineHandle> {
    use log::info;
    use pegainfer_frontend::parallel::ParallelConfig;

    anyhow::ensure!(
        options.tp_size > 0 && options.dp_size > 0,
        "Kimi-K2 --tp-size and --dp-size must be positive"
    );
    let parallel = ParallelConfig::new(options.tp_size, options.dp_size);
    info!(
        "Kimi-K2 EP options: tp_size={}, dp_size={}, ep_world={}, ep_backend={:?}",
        options.tp_size,
        options.dp_size,
        parallel.ep_world(),
        options.ep_backend
    );
    runner::start_engine(
        model_path,
        &EngineLoadOptions {
            enable_cuda_graph: options.cuda_graph,
            device_ordinals: (0..parallel.ep_world()).collect(),
            parallel_config: Some(parallel),
            ep_backend: options.ep_backend,
            seed: 42,
        },
    )
}
