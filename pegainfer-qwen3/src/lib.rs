mod batch_decode;
mod batch_decode_buffers;
mod batch_decode_dag;
pub mod batch_decode_trace;
mod config;
mod dflash;
mod dspark;
mod eagle3;
mod executor;
mod frontend_adapter;
pub(crate) mod green_ctx;
pub mod kernel_bench;
mod lora;
pub mod model_line;
#[cfg(any(test, feature = "test-fixtures"))]
pub use lora::fixtures as lora_fixtures;
mod prefill;
mod projection_fusion;
mod scheduler;
mod speculative;
mod split_kv;
mod unified_forward;
mod verify_graph;
mod weights;

use std::path::Path;
use std::path::PathBuf;

use anyhow::Result;
pub(crate) use config::probe_config_json;
use log::info;
use log::warn;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::EpBackend;
pub use scheduler::DEFAULT_MAX_PREFILL_TOKENS;
pub(crate) use weights::DEFAULT_GPU_MEMORY_UTILIZATION;
pub use weights::DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES;
pub use weights::DEFAULT_KV_PAGE_SIZE;
pub use weights::Qwen3MemoryOptions;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen3LoraOptions {
    max_loras: usize,
    max_lora_rank: usize,
}

impl Qwen3LoraOptions {
    const DEFAULT_MAX_LORAS: usize = 1;
    const DEFAULT_MAX_LORA_RANK: usize = 64;
    const SUPPORTED_MAX_LORA_RANKS: [usize; 9] = [1, 8, 16, 32, 64, 128, 256, 320, 512];

    fn validate(self) -> Result<Self> {
        anyhow::ensure!(self.max_loras > 0, "max_loras must be >= 1");
        anyhow::ensure!(
            Self::is_supported_max_lora_rank(self.max_lora_rank),
            "max_lora_rank must be one of: {}",
            Self::supported_max_lora_ranks_display()
        );
        Ok(self)
    }

    fn is_supported_max_lora_rank(rank: usize) -> bool {
        Self::SUPPORTED_MAX_LORA_RANKS.contains(&rank)
    }

    fn supported_max_lora_ranks_display() -> String {
        Self::SUPPORTED_MAX_LORA_RANKS
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

impl Default for Qwen3LoraOptions {
    fn default() -> Self {
        Self {
            max_loras: Self::DEFAULT_MAX_LORAS,
            max_lora_rank: Self::DEFAULT_MAX_LORA_RANK,
        }
    }
}

/// Prefill/decode GPU-sharing mode (`--decode-overlap`). Defined alongside the
/// stream plumbing in [`green_ctx`].
pub use green_ctx::DecodeOverlap;

/// KV-offload (pegaflow) opt-in for the single-GPU Qwen3 path.
///
/// Disabled by default — the existing GPU-only prefix cache is unchanged.
/// When enabled, the executor saves sealed KV blocks to pegaflow's host tier
/// and prefetches CPU-resident prefixes back into HBM before prefill, so a
/// prompt that has fallen out of the GPU cache still skips recompute. Only the
/// single-GPU topology is supported (tensor parallel shards KV per rank).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3OffloadOptions {
    enabled: bool,
    /// Host pinned-memory pool size (the CPU KV-tier capacity), in bytes.
    pinned_pool_bytes: usize,
    /// Back the pool with 2 MiB hugepages (the box must hold a reservation).
    use_hugepages: bool,
    /// `Some` joins the cross-instance P2P mesh: block hashes register with a
    /// MetaServer, peers pull missing prefixes over RDMA, and this engine
    /// serves theirs. The P/D disaggregation data plane.
    p2p: Option<Qwen3P2pOptions>,
    /// `Some` when the P/D prefill peer is vLLM (pegaflow connector): offload
    /// query keys switch from kvbm lineage hashes to vLLM's prefix-cache hash
    /// scheme so this decode node can find the blocks vLLM registered.
    vllm_compat: Option<Qwen3VllmCompatOptions>,
}

/// Cross-instance P2P KV sharing (see `pegainfer_kv_offload::P2pConfig`).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3P2pOptions {
    /// MetaServer gRPC address, e.g. `http://127.0.0.1:50056`.
    metaserver_addr: String,
    /// This engine's routable `IP:port` (literal socket address; also the
    /// P2P gRPC listen address, so hostnames are rejected at startup). Peers
    /// dial it for RDMA handshakes and block queries.
    advertise_addr: String,
    /// RDMA NIC device names (e.g. `mlx5_0`).
    rdma_nics: Vec<String>,
    /// Barrier a request's KV saves (host tier + MetaServer registration)
    /// before its `Finished` event is emitted. The prefill role in a P/D
    /// deployment turns this on so its HTTP response *is* the KV-ready signal;
    /// costs one write-pipeline + registration drain per finishing step, so
    /// leave it off on decode/serving instances.
    flush_on_finish: bool,
}

/// Decode-node settings for a P/D deployment whose prefill node is vLLM with
/// the pegaflow connector. vLLM registers KV under its own prefix-cache block
/// hashes (`xxh3_128` over canonical-CBOR chained tuples — see
/// `pegainfer_kv_offload::VllmBlockHasher`); with this set, cold-request
/// offload queries derive those keys instead of kvbm lineage hashes, and a
/// zero hit waits out the producer's save/registration tail instead of
/// immediately prefilling from scratch.
///
/// Requires on every vLLM prefill process: `--prefix-caching-hash-algo
/// xxhash_cbor` and `PYTHONHASHSEED` set to `python_hash_seed` (unset, vLLM's
/// chain root is `os.urandom` — unreproducible across processes).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Qwen3VllmCompatOptions {
    /// The `PYTHONHASHSEED` value shared with every vLLM prefill process.
    python_hash_seed: String,
    /// The P side's pegaflow-connector namespace: an 8-hex digest the
    /// connector derives from vLLM config and logs at startup
    /// (`namespace=...`). Both sides must address the same content domain.
    namespace: String,
    /// How long a cold request keeps re-querying a zero hit before giving up
    /// on the expected remote KV and prefilling locally. Covers the P side's
    /// post-response save + MetaServer-registration tail (tens of ms).
    miss_wait: std::time::Duration,
}

impl Qwen3OffloadOptions {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            pinned_pool_bytes: 0,
            use_hugepages: false,
            p2p: None,
            vllm_compat: None,
        }
    }

    pub fn enabled(pinned_pool_bytes: usize) -> Self {
        Self {
            enabled: true,
            pinned_pool_bytes,
            use_hugepages: false,
            p2p: None,
            vllm_compat: None,
        }
    }

    #[must_use]
    fn with_p2p(mut self, p2p: Qwen3P2pOptions) -> Self {
        self.p2p = Some(p2p);
        self
    }

    #[must_use]
    fn with_vllm_compat(mut self, compat: Qwen3VllmCompatOptions) -> Self {
        self.vllm_compat = Some(compat);
        self
    }
}

impl Default for Qwen3OffloadOptions {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Low-level Qwen3 execution interface.
///
/// This is the production phase boundary used by the Qwen3 scheduler and by
/// model-local benchmarks. The root server should use `start_engine` instead.
pub mod runtime {
    pub use crate::batch_decode_buffers::split_chunk_size_for;
    pub use crate::executor::DecodePlan;
    pub use crate::executor::DecodeRequestResult;
    pub use crate::executor::DecodeResult;
    pub use crate::executor::DecodeStepItem;
    pub use crate::executor::PrefillPlan;
    pub use crate::executor::PrefillRequestResult;
    pub use crate::executor::PrefillResult;
    pub use crate::executor::PrefillStepItem;
    pub use crate::executor::Qwen3Executor;
    pub use crate::executor::RequestId;
    pub use crate::executor::UnifiedPlan;
    pub use crate::executor::UnifiedResult;
}

/// Server-facing launch knobs for the Qwen3 engine.
///
/// The binary maps raw CLI flags into this struct; [`launch`] then owns the
/// Qwen3 startup policy — the TP→device mapping and the LoRA↔CUDA-Graph
/// exclusion — and dispatches to the right low-level entry.
/// That policy lives with the model instead of leaking into the server.
#[derive(Clone, Debug)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "these launch flags are independent server options, not states"
)]
pub struct Qwen3LaunchOptions {
    /// CUDA device for single-GPU loads (ignored when `tp_size > 1`).
    pub device_ordinal: usize,
    /// Tensor-parallel world size; `> 1` uses devices `0..tp_size`.
    pub tp_size: usize,
    /// Whether the user requested CUDA Graph. LoRA serving forces it off;
    /// under tensor parallelism every decode graph is pre-captured at startup.
    pub cuda_graph: bool,
    /// Export the live rank-0, batch-1 SplitKv decode graph during startup.
    /// The requested PNG gets a detailed sibling `.dot` for LLM inspection.
    pub dump_graph_png: Option<PathBuf>,
    pub offload: Qwen3OffloadOptions,
    pub no_prefix_cache: bool,
    pub max_prefill_tokens: usize,
    pub memory: Qwen3MemoryOptions,
    /// `Some` switches on LoRA serving (and disables CUDA Graph).
    pub lora: Option<Qwen3LoraOptions>,
    /// How prefill and decode share the GPU (`--decode-overlap`).
    pub decode_overlap: DecodeOverlap,
    pub batch_invariant: bool,
    /// `Some` enables DFlash speculative decoding with this drafter model.
    /// Single-GPU only and mutually exclusive with LoRA and KV offload.
    pub dflash_draft_model_path: Option<PathBuf>,
}

/// Start the Qwen3 engine from server-facing [`Qwen3LaunchOptions`].
#[allow(
    clippy::needless_pass_by_value,
    reason = "launch is a one-shot ownership boundary used by external worker threads"
)]
pub fn launch(model_path: &Path, options: Qwen3LaunchOptions) -> Result<Engine> {
    let device_ordinals = if options.tp_size == 1 {
        vec![options.device_ordinal]
    } else {
        (0..options.tp_size).collect()
    };
    // LoRA serving repoints adapter weights between steps, which a captured
    // decode graph bakes in.
    let enable_cuda_graph = if options.lora.is_some() {
        if options.cuda_graph {
            warn!("Qwen3: CUDA Graph is disabled while LoRA serving is enabled");
        }
        false
    } else {
        options.cuda_graph
    };
    if let Some(path) = &options.dump_graph_png {
        anyhow::ensure!(
            enable_cuda_graph,
            "Qwen3 graph export requires CUDA Graph enabled"
        );
        anyhow::ensure!(
            options.lora.is_none(),
            "Qwen3 graph export is not supported with LoRA serving"
        );
        pegainfer_core::cuda_graph::validate_graph_dump_request(path)?;
    }
    let engine = EngineLoadOptions {
        enable_cuda_graph,
        device_ordinals,
        parallel_config: None,
        ep_backend: EpBackend::Nccl,
        seed: 42,
    };
    if options.offload.enabled {
        info!(
            "Qwen3 KV offload enabled: host tier {:.1} GiB, no_prefix_cache={}",
            options.offload.pinned_pool_bytes as f64 / f64::from(1u32 << 30),
            options.no_prefix_cache
        );
    }
    // Also rejected at enable_decode_overlap (the load-bearing guard); failing
    // here saves the full TP model load + graph pre-capture before the error.
    anyhow::ensure!(
        options.tp_size == 1 || matches!(options.decode_overlap, DecodeOverlap::Off),
        "decode-overlap is unsupported under tensor parallelism"
    );
    anyhow::ensure!(
        !(options.dflash_draft_model_path.is_some() && options.lora.is_some()),
        "DFlash speculative decoding cannot be combined with LoRA serving"
    );
    anyhow::ensure!(
        options.dflash_draft_model_path.is_none()
            || matches!(options.decode_overlap, DecodeOverlap::Off),
        "DFlash speculative decoding cannot be combined with decode overlap \
         (--decode-overlap): the speculative path never takes the unified overlap \
         route, so the overlap streams would only waste VRAM the drafter needs"
    );
    match options.lora {
        Some(lora) => {
            info!(
                "Starting Qwen3 engine with LoRA control; max_loras={}, max_lora_rank={}",
                lora.max_loras, lora.max_lora_rank
            );
            start_engine_with_lora_control(
                model_path,
                engine,
                lora,
                options.offload,
                options.no_prefix_cache,
                options.max_prefill_tokens,
                options.memory,
                options.decode_overlap,
                options.batch_invariant,
            )
        }
        None => start_engine_with_offload_inner(
            model_path,
            engine,
            options.offload,
            options.no_prefix_cache,
            options.max_prefill_tokens,
            options.memory,
            options.decode_overlap,
            options.batch_invariant,
            options.dflash_draft_model_path.as_deref(),
            options.dump_graph_png.as_deref(),
        ),
    }
}

pub fn start_engine(model_path: &Path, options: EngineLoadOptions) -> Result<Engine> {
    start_engine_with_offload(
        model_path,
        options,
        Qwen3OffloadOptions::disabled(),
        false,
        DEFAULT_MAX_PREFILL_TOKENS,
        Qwen3MemoryOptions::default(),
        DecodeOverlap::Off,
        false,
        None,
    )
}

/// Like [`start_engine`] but with pegaflow KV offload (single-GPU only). The
/// host tier persists sealed KV blocks and serves CPU-resident prefixes back
/// into HBM before prefill.
///
/// `no_prefix_cache` is the vLLM-style switch (see
/// [`Qwen3Executor::set_no_prefix_cache`](runtime::Qwen3Executor::set_no_prefix_cache)):
/// without offload it disables prefix matching outright; with offload it keeps
/// the host tier but stops cross-request HBM reuse, so every prefix is served
/// from L2 — the pure-L2 benchmark mode.
///
/// `max_prefill_tokens` caps the total prompt tokens batch-prefilled in one
/// scheduler step (see [`DEFAULT_MAX_PREFILL_TOKENS`]).
#[allow(clippy::too_many_arguments)]
pub fn start_engine_with_offload(
    model_path: &Path,
    options: EngineLoadOptions,
    offload_options: Qwen3OffloadOptions,
    no_prefix_cache: bool,
    max_prefill_tokens: usize,
    memory_options: Qwen3MemoryOptions,
    decode_overlap: DecodeOverlap,
    batch_invariant: bool,
    dflash_draft_model_path: Option<&Path>,
) -> Result<Engine> {
    start_engine_with_offload_inner(
        model_path,
        options,
        offload_options,
        no_prefix_cache,
        max_prefill_tokens,
        memory_options,
        decode_overlap,
        batch_invariant,
        dflash_draft_model_path,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn start_engine_with_offload_inner(
    model_path: &Path,
    options: EngineLoadOptions,
    offload_options: Qwen3OffloadOptions,
    no_prefix_cache: bool,
    max_prefill_tokens: usize,
    memory_options: Qwen3MemoryOptions,
    decode_overlap: DecodeOverlap,
    batch_invariant: bool,
    dflash_draft_model_path: Option<&Path>,
    dump_graph_png: Option<&Path>,
) -> Result<Engine> {
    let EngineLoadOptions {
        enable_cuda_graph,
        device_ordinals,
        seed,
        ..
    } = options;
    let model_path = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path must be valid UTF-8"))?;
    if batch_invariant {
        ensure_batch_invariant_supported(
            decode_overlap,
            no_prefix_cache,
            offload_options.enabled,
            dflash_draft_model_path.is_some(),
            device_ordinals.len(),
        )?;
    }
    apply_batch_invariant_policy(batch_invariant);
    let dflash_draft_model_path = dflash_draft_model_path
        .map(|path| {
            path.to_str()
                .ok_or_else(|| anyhow::anyhow!("DFlash draft model path must be valid UTF-8"))
        })
        .transpose()?;
    frontend_adapter::start_qwen3(
        model_path,
        enable_cuda_graph,
        &device_ordinals,
        seed,
        offload_options,
        no_prefix_cache,
        max_prefill_tokens,
        memory_options,
        decode_overlap,
        dflash_draft_model_path,
        dump_graph_png,
    )
}

pub fn start_engine_with_lora_control(
    model_path: &Path,
    options: EngineLoadOptions,
    lora_options: Qwen3LoraOptions,
    offload_options: Qwen3OffloadOptions,
    no_prefix_cache: bool,
    max_prefill_tokens: usize,
    memory_options: Qwen3MemoryOptions,
    decode_overlap: DecodeOverlap,
    batch_invariant: bool,
) -> Result<Engine> {
    let EngineLoadOptions {
        enable_cuda_graph,
        device_ordinals,
        seed,
        ..
    } = options;
    let model_path = model_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("model path must be valid UTF-8"))?;
    if batch_invariant {
        anyhow::bail!(
            "--batch-invariant is not supported with LoRA: the decode-GEMM pin warms and self-checks \
             only the base projections, not the per-adapter rank-K shapes, so the combination is \
             unverified"
        );
    }
    anyhow::ensure!(
        matches!(decode_overlap, DecodeOverlap::Off),
        "LoRA serving does not support --decode-overlap"
    );
    apply_batch_invariant_policy(batch_invariant);
    frontend_adapter::start_qwen3_with_lora_control(
        model_path,
        enable_cuda_graph,
        &device_ordinals,
        seed,
        lora_options.validate()?,
        offload_options,
        no_prefix_cache,
        max_prefill_tokens,
        memory_options,
    )
}

fn apply_batch_invariant_policy(batch_invariant: bool) {
    use pegainfer_kernels::ops::NumericPolicy;
    use pegainfer_kernels::ops::set_numeric_policy;
    let policy = if batch_invariant {
        NumericPolicy::Pin
    } else {
        NumericPolicy::Tuned
    };
    info!("Qwen3 numeric policy: {policy:?} (--batch-invariant={batch_invariant})");
    set_numeric_policy(policy);
}

fn ensure_batch_invariant_supported(
    decode_overlap: DecodeOverlap,
    no_prefix_cache: bool,
    offload: bool,
    dflash: bool,
    world_size: usize,
) -> Result<()> {
    if !matches!(decode_overlap, DecodeOverlap::Off) {
        anyhow::bail!(
            "--batch-invariant is not compatible with --decode-overlap; the stream override would force the pinned GEMM to bail at runtime"
        );
    }
    if offload {
        anyhow::bail!(
            "--batch-invariant is not supported with KV offload: offload keeps prefix matching on \
             (--no-prefix-cache only disables HBM retention there), and a host-tier prefix hit \
             shifts a prompt's chunk boundaries off the request-local grid"
        );
    }
    if world_size > 1 {
        anyhow::bail!(
            "--batch-invariant is not supported with tensor parallelism: world_size={world_size} has unverified cross-rank reduction order"
        );
    }
    if dflash {
        anyhow::bail!(
            "--batch-invariant is not supported with DFlash speculative decoding: the decode-GEMM \
             pin warms and self-checks only the base projections, so the drafter's fc/MLP GEMMs \
             are not pinned and would bail at the GEMM boundary under Pin"
        );
    }
    if !no_prefix_cache {
        anyhow::bail!(
            "--batch-invariant requires --no-prefix-cache; prefix-cache hits move a prompt's chunk \
             boundaries off the request-local grid, so batch-invariant prefill cannot be provided"
        );
    }
    Ok(())
}
