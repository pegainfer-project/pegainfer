//! GLM5.2 engine surface (TP4 replicated, EP4..EP64 free-running).
//!
//! Startup validates the official GLM5.2 FP8 checkpoint layout, loads rank
//! slices to GPU memory (the non-expert stack replicated to every rank,
//! experts placed into their packed layout at H2D time), builds the resident
//! models, and serves generation with one autonomous engine per logical DP
//! rank: every engine steps its rank unconditionally (idle ranks enter with
//! padding rows) and the per-MoE-layer DeepEP collectives pair by entry
//! count under the conservative protocol-max bound.

mod bookend;
mod config;
mod dense;
mod dspark;
#[cfg(test)]
mod dspark_smoke;
mod fp8;
#[cfg(test)]
mod freerun_probe;
mod indexer;
#[cfg(test)]
mod indexer_smoke;
mod layer;
mod mla_decode;
mod mla_front;
mod model;
pub mod model_line;
mod moe_decode;
mod moe_ep;
mod moe_ep8;
mod moe_tp;
mod mtp;
#[cfg(test)]
mod oracle;
mod prefill_tp;
mod rendezvous;
mod rows;
mod runner;
mod scheduler;
mod scratch;
mod weights;

use std::ops::Range;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use bytesize::ByteSize;
pub(crate) use config::GLM52_LAYERS;
pub(crate) use config::GLM52_ROUTED_EXPERTS;
pub(crate) use config::probe_config_json;
use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::KvCapacity;
use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_kv_store::ArenaSpec;
use pegainfer_kv_store::BlockPool;
use pegainfer_kv_store::KvStore;
use pegainfer_kv_store::KvStoreBuilder;
use pegainfer_kv_store::OffloadMirror;
use pegainfer_kv_store::OffloadRankSpec;
use pegainfer_kv_store::P2pConfig;
use pegainfer_kv_store::PegaflowHost;
use runner::Glm52PrefillBatch;
use runner::Glm52RankPlacement;
use runner::Glm52RankWorker;
use runner::Glm52Worker;
use scheduler::Glm52EngineSpec;
use tokio::sync::mpsc;
use tokio::sync::watch;
use weights::GLM52_EP_RANKS;
use weights::Glm52RankLoadBundle;
use weights::Glm52WeightManifest;

use crate::config::GLM52_MAX_CONTEXT;
use crate::model::GLM52_MODEL_LEN_ALIGN;
use crate::model::glm52_arena_bytes;
use crate::model::glm52_pool_blocks;

const GLM52_PREFILL_CHUNK_ALIGN: usize = GLM52_MODEL_LEN_ALIGN;
#[cfg(test)]
const GLM52_DEFAULT_PREFILL_CHUNK_SIZE: usize = 16_384;

/// Optional speculative decoder used by the GLM5.2 engine.
#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Glm52Drafter {
    None,
    /// External DSpark checkpoint.
    Dspark(PathBuf),
    /// Checkpoint-native layer-78 multi-token prediction decoder.
    NativeMtp,
}

impl Glm52Drafter {
    fn enabled(&self) -> bool {
        !matches!(self, Self::None)
    }

    fn is_dspark(&self) -> bool {
        matches!(self, Self::Dspark(_))
    }

    fn is_mtp(&self) -> bool {
        matches!(self, Self::NativeMtp)
    }

    fn dspark_path(&self) -> Option<&Path> {
        match self {
            Self::Dspark(path) => Some(path),
            Self::None | Self::NativeMtp => None,
        }
    }
}

/// Parse a `--glm52-ranks` value (`start..end`, e.g. `4..8`) into the global
/// DP rank range a process hosts.
fn parse_rank_range(spec: &str) -> Result<Range<usize>> {
    let (start, end) = spec
        .split_once("..")
        .with_context(|| format!("rank range `{spec}` must be start..end (e.g. 4..8)"))?;
    let start: usize = start
        .parse()
        .with_context(|| format!("rank range `{spec}` has a non-numeric start"))?;
    let end: usize = end
        .parse()
        .with_context(|| format!("rank range `{spec}` has a non-numeric end"))?;
    ensure!(start < end, "rank range `{spec}` must satisfy start < end");
    Ok(start..end)
}

/// TP4 prefill-only configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm52PrefillOnlyOptions {
    chunk_size: usize,
}

/// GLM5.2 parallel shape. EP8 is the production layout today; TP4 is the
/// GB300 bring-up target.
#[derive(Clone, Debug)]
pub struct Glm52LaunchOptions {
    tp_size: usize,
    dp_size: usize,
    /// Optional speculative decoder. DSpark enables lossless speculative
    /// sampling from an external checkpoint; native MTP uses the checkpoint's
    /// layer-78 decoder and supports EP decode (single- or multi-process —
    /// the MTP round rides the same fixed chain and DeepEP context as the
    /// target steps) and TP4 prefill-only.
    drafter: Glm52Drafter,
    /// Per-request context cap (`prompt + max_tokens - 1 <= max_model_len`).
    /// `None` sizes it from the post-weight-load free VRAM (fleet minimum);
    /// an explicit value is still validated against that budget so an
    /// impossible cap fails at launch, not at the first long request.
    max_model_len: Option<usize>,
    /// Enables TP4 prefill-only serving.
    prefill_only: Option<Glm52PrefillOnlyOptions>,
    /// vLLM-style kill switch: disable prefix matching outright (every
    /// prefill recomputes the full prompt). Prefix caching is also forced
    /// off while a speculative decoder is on: DSpark needs aux-hidden
    /// captures for every prefix row, while native MTP needs target hidden
    /// states and uninterrupted MTP KV continuity.
    no_prefix_cache: bool,
    /// `Some` adds the pegaflow host tier under the prefix cache: sealed KV
    /// blocks flow to one shared pinned pool on request release, and a
    /// prompt whose prefix fell out of HBM restores from it at admission.
    /// Requires the prefix cache (rejected at launch alongside any
    /// speculative decoder or `no_prefix_cache`).
    kv_offload: Option<Glm52KvOffloadOptions>,
    /// Launch-time MoE sharding topology. `Ep8` (default) is the
    /// high-throughput configuration: 32 whole experts per rank, DeepEP
    /// dispatch/combine, buckets 1-8. `Tp4` is GB300 **prefill-only**
    /// tensor parallel (NCCL all-reduce; requires `--glm52-prefill-only`).
    moe_topo: Glm52MoeTopo,
    /// Stage checkpoint bytes through pinned double buffers. Intended for
    /// warm page-cache starts; cold network filesystems should leave it off.
    weight_staging: bool,
    /// Export rank 0's already pre-captured whole-step decode graph during
    /// startup. EP and TP4 export bucket 1. The requested PNG gets a
    /// complete sibling `.dot` for machine inspection.
    dump_graph_png: Option<PathBuf>,
    /// The global DP ranks THIS process hosts (default: the whole topology
    /// — the single-node engine). A partial range is the multi-process
    /// cross-node shape: every node runs the same binary over its own ranks,
    /// and the collective DeepEP communicator is the only coupling
    /// (`docs/models/glm52/free-running-dp.md` §3).
    ranks: Option<Range<usize>>,
    /// Bootstrap rendezvous address, required exactly when `ranks` is a
    /// partial range. The process hosting rank 0 binds it and serves the
    /// DeepEP unique id; every other process connects to fetch it. A
    /// one-time handshake — there is no runtime control plane.
    rendezvous: Option<String>,
}

/// Launch-time MoE sharding topology (the expert slab is repacked during
/// H2D load, so this is a boot choice — the two layouts never co-reside).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Glm52MoeTopo {
    #[default]
    Ep8,
    /// Four-GPU expert-parallel layout (DP4/EP4, 64 whole routed experts per
    /// rank — the GB300 high-throughput target). Same DeepEP protocol as EP8
    /// with its own shim instantiation; routed experts use the SM100
    /// DeepGEMM masked grouped chain.
    Ep4,
    /// Cross-tray expert-parallel widths on GB300 NVL72 (4 GPUs per tray;
    /// every tray's process hosts its own ranks behind `--glm52-ranks`).
    /// Same DeepEP protocol with one
    /// shim instantiation per width; all run the SM100 DeepGEMM chain.
    Ep16,
    Ep32,
    Ep64,
    Tp4,
}

impl Glm52MoeTopo {
    #[must_use]
    fn default_dp_size(self) -> usize {
        match self {
            Self::Tp4 => 1,
            _ => self.device_count(),
        }
    }

    #[must_use]
    fn device_count(self) -> usize {
        match self {
            Self::Ep8 => GLM52_EP_RANKS,
            Self::Ep4 | Self::Tp4 => 4,
            Self::Ep16 => 16,
            Self::Ep32 => 32,
            Self::Ep64 => 64,
        }
    }

    /// Number of independently scheduled request partitions. Tensor-
    /// replicated workers execute one mirrored partition in lock-step.
    #[must_use]
    fn logical_rank_count(self) -> usize {
        if self.uses_tensor_replicated_moe() {
            1
        } else {
            self.device_count()
        }
    }

    /// The `--tp-size` this topology requires (server validation mirrors the
    /// launch-time ensure).
    #[must_use]
    fn expected_tp_size(self) -> usize {
        match self {
            Self::Tp4 => 4,
            _ => 1,
        }
    }

    #[must_use]
    fn expected_ep_size(self) -> usize {
        match self {
            Self::Tp4 => 1,
            _ => self.device_count(),
        }
    }

    #[must_use]
    fn uses_ep_expert_bundles(self) -> bool {
        !self.uses_tensor_replicated_moe()
    }

    /// Whole routed experts per rank of an expert-bundle topology (EP8 → 32,
    /// EP4 → 64). Meaningless for the tensor-replicated topologies.
    #[must_use]
    fn ep_local_experts(self) -> usize {
        debug_assert!(self.uses_ep_expert_bundles());
        GLM52_ROUTED_EXPERTS / self.expected_ep_size()
    }

    /// Whether this topology mirrors one logical rank across all workers
    /// (TP4) — the server needs it to size the frontend partition count
    /// for a hosted rank range.
    #[must_use]
    fn uses_tensor_replicated_moe(self) -> bool {
        matches!(self, Self::Tp4)
    }
}

impl std::str::FromStr for Glm52MoeTopo {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s {
            "ep4" => Ok(Self::Ep4),
            "ep8" => Ok(Self::Ep8),
            "ep16" => Ok(Self::Ep16),
            "ep32" => Ok(Self::Ep32),
            "ep64" => Ok(Self::Ep64),
            "tp4" => Ok(Self::Tp4),
            other => {
                anyhow::bail!(
                    "GLM5.2 MoE topology must be ep4, ep8, ep16, ep32, ep64, or tp4, \
                     got {other}"
                )
            }
        }
    }
}

#[cfg(test)]
mod topology_tests {
    use super::*;

    #[test]
    fn tp4_topology_shape_is_four_rank_replicated_tp() {
        assert_eq!(Glm52MoeTopo::Tp4.default_dp_size(), 1);
        assert_eq!(Glm52MoeTopo::Tp4.device_count(), 4);
        assert_eq!(Glm52MoeTopo::Tp4.logical_rank_count(), 1);
        assert_eq!(Glm52MoeTopo::Tp4.expected_tp_size(), 4);
        assert_eq!(Glm52MoeTopo::Tp4.expected_ep_size(), 1);
        assert!(!Glm52MoeTopo::Tp4.uses_ep_expert_bundles());
        assert!(Glm52MoeTopo::Tp4.uses_tensor_replicated_moe());
    }

    #[test]
    fn ep8_shapes_remain_unchanged() {
        let topo = Glm52MoeTopo::Ep8;
        assert_eq!(topo.default_dp_size(), GLM52_EP_RANKS);
        assert_eq!(topo.device_count(), GLM52_EP_RANKS);
        assert_eq!(topo.expected_tp_size(), 1);
        assert_eq!(topo.expected_ep_size(), GLM52_EP_RANKS);
        assert_eq!(topo.logical_rank_count(), GLM52_EP_RANKS);
        assert!(topo.uses_ep_expert_bundles());
        assert!(!topo.uses_tensor_replicated_moe());
        assert_eq!(topo.ep_local_experts(), 32);
    }

    #[test]
    fn ep4_topology_shape_is_four_rank_expert_parallel() {
        assert_eq!(Glm52MoeTopo::Ep4.default_dp_size(), 4);
        assert_eq!(Glm52MoeTopo::Ep4.device_count(), 4);
        assert_eq!(Glm52MoeTopo::Ep4.logical_rank_count(), 4);
        assert_eq!(Glm52MoeTopo::Ep4.expected_tp_size(), 1);
        assert_eq!(Glm52MoeTopo::Ep4.expected_ep_size(), 4);
        assert!(Glm52MoeTopo::Ep4.uses_ep_expert_bundles());
        assert!(!Glm52MoeTopo::Ep4.uses_tensor_replicated_moe());
        assert_eq!(Glm52MoeTopo::Ep4.ep_local_experts(), 64);
        assert_eq!("ep4".parse::<Glm52MoeTopo>().unwrap(), Glm52MoeTopo::Ep4);
    }

    #[test]
    fn cross_tray_ep_widths_shard_all_routed_experts() {
        for (topo, ranks, local) in [
            (Glm52MoeTopo::Ep16, 16, 16),
            (Glm52MoeTopo::Ep32, 32, 8),
            (Glm52MoeTopo::Ep64, 64, 4),
        ] {
            assert_eq!(topo.default_dp_size(), ranks);
            assert_eq!(topo.device_count(), ranks);
            assert_eq!(topo.logical_rank_count(), ranks);
            assert_eq!(topo.expected_tp_size(), 1);
            assert_eq!(topo.expected_ep_size(), ranks);
            assert!(topo.uses_ep_expert_bundles());
            assert!(!topo.uses_tensor_replicated_moe());
            assert_eq!(topo.ep_local_experts(), local);
            assert_eq!(format!("ep{ranks}").parse::<Glm52MoeTopo>().unwrap(), topo);
        }
    }
}

/// Host-tier KV offload knobs. One `PegaEngine` (one pinned pool) backs all
/// 8 DP ranks under a single namespace: the MLA latent has no TP sharding
/// and the non-expert weights are replicated, so any rank's KV for a token
/// prefix is as good as any other's — the same tolerance as reusing a
/// rank's own prefix cache (FP reduction order may differ across the batch
/// shapes that computed it, never the semantics). Any rank restores what
/// any rank saved.
#[derive(Clone, Debug)]
pub struct Glm52KvOffloadOptions {
    /// Host pinned-memory pool size in bytes, shared by all ranks.
    pinned_pool_bytes: usize,
    /// Back the pool with hugepages (the box must hold a reservation —
    /// check `HugePages_Total`).
    use_hugepages: bool,
    /// `Some` joins the cross-instance P2P mesh: saved block hashes register
    /// with the MetaServer and missing prefixes are pulled from peer
    /// instances over RDMA — the P/D disaggregation data plane.
    p2p: Option<Glm52P2pOptions>,
}

/// Cross-instance P2P KV sharing (see `pegainfer_kv_store::P2pConfig`).
#[derive(Clone, Debug)]
pub struct Glm52P2pOptions {
    /// MetaServer gRPC address, e.g. `http://10.0.0.100:50056`.
    metaserver_addr: String,
    /// This engine's routable `IP:port` (doubles as the embedded transfer
    /// service's bind address). Must be reachable by every peer.
    advertise_addr: String,
    /// RDMA NIC device names to register the pinned pool on.
    rdma_nics: Vec<String>,
}

/// GLM5.2 kernels (FlashMLA SM100, DeepGEMM MQA and routed experts, …)
/// no longer ship a Hopper path. Refuse SM9x and older at launch.
///
/// `local_gpus` is the number of GPUs **this process** hosts (mapped to local
/// ordinals `0..local_gpus`), not the global EP width — multi-process
/// EP16/32/64 shards may own only four devices on a tray.
fn ensure_blackwell_devices(local_gpus: usize) -> Result<()> {
    ensure!(local_gpus > 0, "GLM5.2 needs at least one GPU");
    for ordinal in 0..local_gpus {
        let ctx = pegainfer_kernels::tensor::DeviceContext::new_with_device(ordinal)
            .with_context(|| format!("GLM5.2 open device {ordinal} for arch check"))?;
        let major = ctx.ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )?;
        let minor = ctx.ctx.attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )?;
        ensure!(
            major >= 10,
            "GLM5.2 requires Blackwell (compute capability ≥ 10.0); device {ordinal} is \
             SM{major}.{minor}. Hopper (SM9x) support was removed."
        );
    }
    Ok(())
}

fn launch(model_path: &Path, options: Glm52LaunchOptions) -> Result<EngineHandle> {
    let Glm52LaunchOptions {
        tp_size,
        dp_size,
        drafter,
        max_model_len,
        prefill_only,
        no_prefix_cache,
        kv_offload,
        moe_topo,
        weight_staging,
        dump_graph_png,
        ranks,
        rendezvous,
    } = options;
    let device_count = moe_topo.device_count();
    let ranks = ranks.unwrap_or(0..device_count);
    ensure!(
        ranks.start < ranks.end && ranks.end <= device_count,
        "GLM5.2 --glm52-ranks {}..{} is outside 0..{device_count} or empty",
        ranks.start,
        ranks.end
    );
    // The (slots, drafts) startup pair must fit the step-row budget —
    // an over-committed pair would silently cap verify spans under full
    // occupancy and collapse speculation (#812's original failure mode).
    // Only native MTP rides verify spans; without a drafter every slot is a
    // single decode row and any slot count within the ceiling fits.
    let slots = model::glm52_decode_slots();
    let draft_len = if drafter.is_mtp() {
        crate::mtp::glm52_mtp_draft_len()
    } else {
        0
    };
    ensure!(
        slots * (1 + draft_len) <= model::GLM52_MAX_STEP_ROWS,
        "GLM5.2 GLM52_DECODE_SLOTS={slots} x (1 + GLM52_MTP_DRAFTS={draft_len}) exceeds the \
         {}-row step budget; 32 slots need GLM52_MTP_DRAFTS=2",
        model::GLM52_MAX_STEP_ROWS
    );
    log::info!(
        "GLM5.2 decode profile: slots={slots} mtp_draft_span={draft_len} ({} of {} step rows)",
        slots * (1 + draft_len),
        model::GLM52_MAX_STEP_ROWS
    );
    // Probe only the GPUs this process hosts (local ordinals 0..N), not the
    // global EP width — EP16 on one 4-GPU tray would otherwise try device 4+.
    ensure_blackwell_devices(ranks.end - ranks.start)?;
    let multi_process = ranks != (0..device_count);
    if moe_topo.uses_tensor_replicated_moe() {
        ensure!(
            !multi_process,
            "GLM5.2 {moe_topo:?} is a single-process topology (its workers rendezvous \
             device pointers in-process); --glm52-ranks must cover 0..{device_count}"
        );
    }
    ensure!(
        !multi_process || rendezvous.is_some(),
        "GLM5.2 hosting ranks {}..{} of {device_count} requires --glm52-rendezvous \
         (the rank-0 process binds it, everyone else connects)",
        ranks.start,
        ranks.end
    );
    if drafter.is_mtp() {
        ensure!(
            moe_topo.uses_ep_expert_bundles()
                || (moe_topo == Glm52MoeTopo::Tp4 && prefill_only.is_some()),
            "GLM5.2 native MTP requires EP decode or TP4 prefill-only"
        );
    }
    if let Some(path) = &dump_graph_png {
        ensure!(
            ranks.start == 0,
            "GLM5.2 --dump-graph-png only works on the process hosting rank 0"
        );
        pegainfer_core::cuda_graph::validate_graph_dump_request(path)?;
    }
    match moe_topo {
        Glm52MoeTopo::Tp4 => {
            ensure!(
                tp_size == 4,
                "GLM5.2 TP4 requires --tp-size=4, got {tp_size}"
            );
            ensure!(
                dp_size == 1,
                "GLM5.2 TP4 requires --dp-size=1 (or omitted), got {dp_size}"
            );
            ensure!(
                prefill_only.is_some(),
                "GLM5.2 TP4 is prefill-only: pass --glm52-prefill-only \
                 (decode TP / LL packet path was removed)"
            );
        }
        _ => {
            ensure!(
                tp_size == 1,
                "GLM5.2 {moe_topo:?} requires --tp-size=1, got {tp_size}"
            );
            let expected_dp = moe_topo.default_dp_size();
            ensure!(
                dp_size == expected_dp,
                "GLM5.2 {moe_topo:?} requires --dp-size={expected_dp} (or omitted), got {dp_size}"
            );
        }
    }
    if let Some(prefill) = prefill_only {
        ensure!(
            moe_topo == Glm52MoeTopo::Tp4,
            "GLM5.2 prefill-only mode requires the TP4 topology"
        );
        ensure!(
            prefill.chunk_size > 0 && prefill.chunk_size.is_multiple_of(GLM52_PREFILL_CHUNK_ALIGN),
            "GLM5.2 prefill chunk size {} must be a positive multiple of {}",
            prefill.chunk_size,
            GLM52_PREFILL_CHUNK_ALIGN,
        );
        ensure!(
            !drafter.is_dspark(),
            "GLM5.2 prefill-only mode is incompatible with DSpark"
        );
        ensure!(
            !no_prefix_cache,
            "GLM5.2 prefill-only mode requires prefix caching"
        );
        ensure!(
            kv_offload.is_none() || drafter.is_mtp(),
            "GLM5.2 prefill-only KV offload requires native MTP's 101-arena contract"
        );
        ensure!(
            dump_graph_png.is_none(),
            "GLM5.2 prefill-only mode does not expose a decode CUDA graph"
        );
    }
    // The offload tier extends the prefix cache (restored blocks surface as
    // matched prefix), so a config that disables prefix matching while asking
    // for offload is contradictory — fail loud instead of silently idling an
    // allocated multi-GiB pinned pool.
    ensure!(
        kv_offload.is_none() || ((!drafter.enabled() || drafter.is_mtp()) && !no_prefix_cache),
        "GLM5.2 --kv-offload requires the prefix cache: drop --no-prefix-cache and the \
         DSpark drafter"
    );
    // The TP topology mirrors KV on every rank; the host tier's restore leg
    // H2Ds into ONE rank's arena, which would silently desync the other 7.
    ensure!(
        kv_offload.is_none()
            || moe_topo.uses_ep_expert_bundles()
            || (moe_topo == Glm52MoeTopo::Tp4 && prefill_only.is_some()),
        "GLM5.2 --kv-offload requires EP decode or TP4 prefill-only"
    );
    start_engine(
        model_path,
        &Glm52LoadOptions {
            ranks,
            rendezvous,
            tp_size,
            dp_size,
            ep_size: moe_topo.expected_ep_size(),
        },
        drafter,
        max_model_len,
        prefill_only,
        no_prefix_cache,
        kv_offload,
        moe_topo,
        weight_staging,
        dump_graph_png,
    )
}

/// Free VRAM held back from the context-cap budget on every rank, covering
/// the post-probe allocations the exact arena ledger does not model: the
/// MLA W_UK/W_UV bf16 dequant during build (~1.1 GiB net over the freed fp8
/// kv_b), DeepEP collective buffers, the 8 whole-step graph instantiations,
/// cuBLAS workspaces, and allocator fragmentation. The SM100 DeepGEMM delta
/// over the retired weight-only chain is charged exactly and separately.
/// Measured on 8×H200
/// (jz-38, 2026-07-06): the worst rank's non-arena post-probe allocations
/// came to ~3.05 GiB, so 5 GiB leaves ~2 GiB of post-build headroom over
/// the [`GLM52_POST_BUILD_MIN_FREE_BYTES`] floor; the post-build re-probe
/// below turns any drift into a launch failure instead of a mid-serving
/// OOM.
const GLM52_VRAM_RESERVE_BYTES: usize = 5 << 30;

/// Extra reserve when the DSpark drafter is enabled: the replicated draft
/// weights (~3.8 GiB bf16) plus its dense forward scratch, which load after
/// the probe. The drafter's cap-scaled buffers are in the exact ledger
/// (`glm52_dspark_arena_bytes`), not here.
const GLM52_DSPARK_VRAM_RESERVE_BYTES: usize = 5 << 30;

/// Fixed prefill workspace reserve: the row-block-bounded MoE scratch
/// (gathered fp8 routes + grouped W2 output), the attention/dense sub-tile
/// buffers, the unpacked bf16 KV pool, and the GEMM workspaces.
const GLM52_PREFILL_FIXED_SCRATCH_BYTES: usize = 3 << 30;

/// Estimated scratch bytes per token row (chunk-scale activations, MLA
/// front/query buffers, indexer carry, router logits).
const GLM52_PREFILL_SCRATCH_BYTES_PER_TOKEN: usize = 160 << 10;

/// The smallest cap worth serving with (the pre-refactor bring-up value);
/// a budget below this is a misconfiguration, not a working engine.
const GLM52_MIN_MODEL_LEN: usize = 4096;

/// Free VRAM every rank must still have AFTER the model, DeepEP contexts,
/// and the optional drafter are fully resident — headroom for the whole-step
/// graph instantiations (captured lazily by the engine) and allocator
/// fragmentation. The post-build re-probe fails launch below this, so a
/// ledger/reserve drift crashes at startup, not mid-serving.
const GLM52_POST_BUILD_MIN_FREE_BYTES: usize = 1 << 30;

/// The launch-time context-cap decision and the numbers behind it — the log
/// line and the tests consume the same values the decision used, so they
/// cannot drift apart. The pool block count is NOT decided here anymore:
/// the ledger only provides the fill floor, and the authoritative count
/// comes from the measured two-phase build (`build_rank_models`).
#[derive(Clone, Copy, Debug)]
struct Glm52ContextBudget {
    max_model_len: usize,
    /// Exact bytes the cap costs a rank at the floor pool (build arenas +
    /// drafter lane) — an initial estimate for the startup log.
    arena_bytes: usize,
    reserve_bytes: usize,
    budget_bytes: usize,
    /// The measured fill's floor: the legacy nominal `slots x cap` pool when
    /// the ledger says it fits, else a one-request pool (the
    /// explicit-big-cap shape the pool decoupling exists for).
    floor_blocks: usize,
}

/// Exact cap-scaled bytes a rank allocates for a candidate cap: the build
/// arenas plus the selected speculative lane.
fn glm52_cap_bytes(
    max_model_len: usize,
    pool_blocks: usize,
    drafter: &Glm52Drafter,
    prefill_only: bool,
    moe_topo: Glm52MoeTopo,
) -> Result<usize> {
    Ok(glm52_arena_bytes(max_model_len, pool_blocks, prefill_only)?
        + if drafter.is_dspark() {
            crate::dspark::glm52_dspark_arena_bytes(max_model_len)
        } else if drafter.is_mtp() {
            crate::mtp::glm52_mtp_arena_bytes(max_model_len, pool_blocks, moe_topo)?
        } else {
            0
        })
}

/// The smallest useful pool: one max-length request's pages plus the
/// padding block and one spare.
fn glm52_one_request_pool_blocks(max_model_len: usize) -> usize {
    (max_model_len + 1).div_ceil(GLM52_MODEL_LEN_ALIGN) + 2
}

/// Free VRAM held back from the measured KV fill on every rank — the ONLY
/// remaining estimate in the pool sizing. Everything the fill can measure is
/// already resident when the workers report free VRAM (the fixed build), and
/// the exactly-known post-finish charges (DeepGEMM buffers, the DSpark lane)
/// are subtracted separately; this covers what allocates after FinishKv and
/// has no exact ledger — the whole-step CUDA graph instantiations (captured
/// lazily by the engines), cuBLAS/FlashInfer workspaces, and allocator
/// fragmentation. `GLM52_GRAPH_RESERVE_GIB` overrides.
fn glm52_graph_reserve_bytes() -> usize {
    (std::env::var("GLM52_GRAPH_RESERVE_GIB")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(3))
        << 30
}

/// The measured KV fill (#818, two-phase build): after every rank built its
/// fixed half, the fleet-minimum free VRAM — minus the graph reserve and the
/// exactly-known post-finish charges — is spent on pool blocks at the
/// ledger's exact per-block slab cost. The floor is never reduced (the
/// nominal pool where affordable, else one max-length request).
fn glm52_measured_pool_blocks(
    min_free_bytes: usize,
    post_finish_bytes: usize,
    slab_bytes_per_block: usize,
    floor_blocks: usize,
) -> usize {
    let usable = min_free_bytes.saturating_sub(glm52_graph_reserve_bytes() + post_finish_bytes);
    if slab_bytes_per_block == 0 {
        return floor_blocks;
    }
    (usable / slab_bytes_per_block).max(floor_blocks)
}

/// The legacy `slots x cap` pool sizing — the cap-derivation NOMINAL (the
/// binary search finds the largest cap whose nominal-pool cost fits), and
/// the floor the budget fill starts from.
fn glm52_nominal_pool_blocks(max_model_len: usize) -> usize {
    glm52_pool_blocks(max_model_len, model::glm52_decode_slots())
}

fn glm52_prefill_scratch_reservation(
    prefill_only: Option<Glm52PrefillOnlyOptions>,
) -> Result<usize> {
    let Some(prefill) = prefill_only else {
        return Ok(0);
    };
    prefill
        .chunk_size
        .checked_mul(GLM52_PREFILL_SCRATCH_BYTES_PER_TOKEN)
        .and_then(|bytes| bytes.checked_add(GLM52_PREFILL_FIXED_SCRATCH_BYTES))
        .context("GLM5.2 prefill scratch reservation overflow")
}

/// Decide the per-request context cap from the post-weight-load VRAM budget.
/// Every slot's cache region is sized `max_model_len` tokens at build, so a
/// candidate cap's cost is exact arithmetic ([`glm52_cap_bytes`]) over the
/// fleet-minimum free bytes — kept free of CUDA so the policy is
/// unit-testable. Auto mode binary-searches the largest aligned cap that
/// fits; an explicit cap must be aligned and fit, or launch fails.
fn derive_max_model_len(
    requested: Option<usize>,
    min_free_vram_bytes: usize,
    drafter: &Glm52Drafter,
    prefill_scratch_bytes: usize,
    prefill_only: bool,
    moe_topo: Glm52MoeTopo,
) -> Result<Glm52ContextBudget> {
    let reserve_bytes = GLM52_VRAM_RESERVE_BYTES
        + if drafter.is_dspark() {
            GLM52_DSPARK_VRAM_RESERVE_BYTES
        } else {
            0
        }
        + prefill_scratch_bytes;
    let budget_bytes = min_free_vram_bytes.saturating_sub(reserve_bytes);
    let max_model_len = if let Some(requested) = requested {
        ensure!(
            requested >= GLM52_MIN_MODEL_LEN,
            "GLM5.2 --max-model-len {requested} is below the minimum {GLM52_MIN_MODEL_LEN}"
        );
        ensure!(
            requested <= GLM52_MAX_CONTEXT,
            "GLM5.2 --max-model-len {requested} exceeds the checkpoint's \
             max_position_embeddings {GLM52_MAX_CONTEXT}"
        );
        ensure!(
            requested.is_multiple_of(GLM52_MODEL_LEN_ALIGN),
            "GLM5.2 --max-model-len {requested} must be a multiple of {GLM52_MODEL_LEN_ALIGN} \
             (the FlashMLA page size); nearest valid values are {} and {}",
            requested / GLM52_MODEL_LEN_ALIGN * GLM52_MODEL_LEN_ALIGN,
            requested.next_multiple_of(GLM52_MODEL_LEN_ALIGN),
        );
        // The pool no longer multiplies the cap by the slot count (#818):
        // an explicit cap only has to fit the fixed arenas plus a pool that
        // can hold ONE max-length request — admission's lifetime reservation
        // guards each request, concurrency comes from whatever pool the
        // budget fill provides.
        let required = glm52_cap_bytes(
            requested,
            glm52_one_request_pool_blocks(requested),
            drafter,
            prefill_only,
            moe_topo,
        )?;
        // The graph reserve must survive the fill: without it a cap in the
        // band (budget - reserve, budget] passes here, the measured fill
        // floors at the one-request pool anyway, and FinishKv or the lazy
        // graph capture OOMs raw instead of failing at this door.
        ensure!(
            required + glm52_graph_reserve_bytes() <= budget_bytes,
            "GLM5.2 --max-model-len {requested} needs {} of cache per rank (fixed arenas + a \
             one-request pool + {} graph reserve) but only {} fits (min rank free VRAM {} - \
             reserve {}); lower it or free VRAM",
            ByteSize(required as u64),
            ByteSize(glm52_graph_reserve_bytes() as u64),
            ByteSize(budget_bytes as u64),
            ByteSize(min_free_vram_bytes as u64),
            ByteSize(reserve_bytes as u64),
        );
        requested
    } else {
        // Largest aligned cap whose exact cost fits the budget: the cost is
        // monotone in the cap, so binary search over the aligned candidates.
        let (mut lo, mut hi) = (0, GLM52_MAX_CONTEXT / GLM52_MODEL_LEN_ALIGN);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if glm52_cap_bytes(
                mid * GLM52_MODEL_LEN_ALIGN,
                glm52_nominal_pool_blocks(mid * GLM52_MODEL_LEN_ALIGN),
                drafter,
                prefill_only,
                moe_topo,
            )? <= budget_bytes
            {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        let derived = lo * GLM52_MODEL_LEN_ALIGN;
        ensure!(
            derived >= GLM52_MIN_MODEL_LEN,
            "GLM5.2 free VRAM leaves a context cap of {derived} (< {GLM52_MIN_MODEL_LEN}): \
             budget {} (min rank free VRAM {} - reserve {})",
            ByteSize(budget_bytes as u64),
            ByteSize(min_free_vram_bytes as u64),
            ByteSize(reserve_bytes as u64),
        );
        derived
    };
    // Fill floor for the measured pool sizing (#818): the legacy nominal
    // (`slots x cap`) where the ledger says it still fits — identical
    // defaults — otherwise a pool holding one max-length request (the
    // explicit-big-cap shape the decoupling exists for). The ledger's job
    // ends here: the authoritative block count comes from the measured
    // two-phase fill between the workers' build phases.
    let nominal = glm52_nominal_pool_blocks(max_model_len);
    let floor_blocks = if glm52_cap_bytes(max_model_len, nominal, drafter, prefill_only, moe_topo)?
        <= budget_bytes
    {
        nominal
    } else {
        glm52_one_request_pool_blocks(max_model_len)
    };
    Ok(Glm52ContextBudget {
        max_model_len,
        floor_blocks,
        arena_bytes: glm52_cap_bytes(max_model_len, floor_blocks, drafter, prefill_only, moe_topo)?,
        reserve_bytes,
        budget_bytes,
    })
}

#[derive(Clone, Debug)]
struct Glm52LoadOptions {
    /// The global DP ranks this process hosts; its local GPUs are device
    /// ordinals `0..ranks.len()`.
    ranks: Range<usize>,
    /// Bootstrap rendezvous address for a multi-process fleet (see
    /// [`Glm52LaunchOptions::rendezvous`]).
    rendezvous: Option<String>,
    tp_size: usize,
    dp_size: usize,
    ep_size: usize,
}

#[derive(Debug)]
struct StartupValidation {
    /// The global DP ranks this process hosts (`device_ordinal i` serves
    /// `ranks.start + i`).
    ranks: Range<usize>,
    rendezvous: Option<String>,
    rank_bundles: Vec<Glm52RankLoadBundle>,
    rank_tensor_counts: Vec<usize>,
    rank_expert_ranges: Vec<std::ops::Range<usize>>,
}

#[derive(Debug)]
/// Per-rank facts gathered while the weights landed (index = rank).
struct GpuWeightLoadReport {
    tensor_counts: Vec<usize>,
    bytes: Vec<usize>,
    free_vram_bytes: Vec<usize>,
}

struct LoadedGlm52Runtime {
    workers: Vec<Glm52Worker>,
    report: GpuWeightLoadReport,
}

fn start_engine(
    model_path: &Path,
    options: &Glm52LoadOptions,
    drafter: Glm52Drafter,
    requested_max_model_len: Option<usize>,
    prefill_only: Option<Glm52PrefillOnlyOptions>,
    no_prefix_cache: bool,
    kv_offload: Option<Glm52KvOffloadOptions>,
    moe_topo: Glm52MoeTopo,
    weight_staging: bool,
    dump_graph_png: Option<PathBuf>,
) -> Result<EngineHandle> {
    let startup = validate_startup(model_path, options, moe_topo, drafter.is_mtp())?;
    let loaded = load_rank_weights_to_gpu(model_path, &startup, moe_topo, weight_staging)?;
    log::info!(
        "GLM5.2 load-weight startup complete: hosted_ranks={:?}, rank_plan_tensors={:?}, rank_gpu_tensors={:?}, rank_gpu_bytes={:?}",
        startup.ranks,
        startup.rank_tensor_counts,
        loaded.report.tensor_counts,
        format_bytes(&loaded.report.bytes),
    );

    let min_free_vram_bytes = loaded
        .report
        .free_vram_bytes
        .iter()
        .copied()
        .min()
        .expect("at least one rank loaded");
    // These model-build allocations land after the weight probe. Charge
    // exact bytes here so auto mode cannot spend them on KV and then fail the
    // post-build graph-capture headroom gate.
    let qa_kva_twin_bytes = mla_front::glm52_qa_kva_twin_bytes()?;
    let deepgemm_vram_charge_bytes =
        moe_ep::glm52_deepgemm_vram_charge_bytes(moe_topo, drafter.is_mtp())?;
    let post_weight_charge_bytes = qa_kva_twin_bytes
        .checked_add(deepgemm_vram_charge_bytes)
        .context("GLM5.2 post-weight VRAM charge overflow")?;
    let budget = derive_max_model_len(
        requested_max_model_len,
        min_free_vram_bytes.saturating_sub(post_weight_charge_bytes),
        &drafter,
        glm52_prefill_scratch_reservation(prefill_only)?,
        prefill_only.is_some(),
        moe_topo,
    )?;
    if let Some(prefill) = prefill_only {
        ensure!(
            prefill.chunk_size <= budget.max_model_len,
            "GLM5.2 prefill chunk size {} exceeds max_model_len {}; lower \
             --glm52-prefill-chunk-size or raise --max-model-len",
            prefill.chunk_size,
            budget.max_model_len,
        );
    }
    let max_model_len = budget.max_model_len;
    log::info!(
        "GLM5.2 max_model_len={max_model_len} ({}): min rank free VRAM {} after weights \
         (qa|kv_a twins {} + DeepGEMM post-weight delta {} charged), floor-pool arenas {} \
         across {} slots{}, reserve {}, budget {}{}",
        if requested_max_model_len.is_some() {
            "--max-model-len"
        } else {
            "VRAM-derived"
        },
        ByteSize(min_free_vram_bytes as u64),
        ByteSize(qa_kva_twin_bytes as u64),
        ByteSize(deepgemm_vram_charge_bytes as u64),
        ByteSize(budget.arena_bytes as u64),
        model::glm52_decode_slots(),
        if drafter.enabled() {
            " (draft lane included)"
        } else {
            ""
        },
        ByteSize(budget.reserve_bytes as u64),
        ByteSize(budget.budget_bytes as u64),
        prefill_only.map_or_else(String::new, |prefill| format!(
            ", prefill-only chunk {} (scratch reservation {})",
            prefill.chunk_size,
            ByteSize(
                glm52_prefill_scratch_reservation(Some(prefill))
                    .expect("validated prefill scratch reservation") as u64
            ),
        )),
    );

    let eos_token_ids = read_eos_token_ids(model_path)?;
    // Exactly-known VRAM that lands AFTER the workers' free-VRAM measurement
    // (post-FinishKv): the DeepEP/DeepGEMM buffers created in SetupComm, and
    // — when DSpark is on — the draft weights reserve plus its cap-scaled
    // states loaded after comm setup. The measured KV fill holds these back
    // alongside the graph reserve.
    let post_finish_reserve_bytes = deepgemm_vram_charge_bytes
        + if drafter.is_dspark() {
            GLM52_DSPARK_VRAM_RESERVE_BYTES + crate::dspark::glm52_dspark_arena_bytes(max_model_len)
        } else {
            0
        };
    // build_rank_models sends SetupComm, so from inside it the DeepEP
    // contexts exist and their destruction is COLLECTIVE: any startup failure
    // from here on must broadcast Shutdown to every rank BEFORE the workers'
    // sequential Drop joins them one by one (the same teardown contract as
    // the engine exit) — otherwise the first dropped worker blocks in
    // the destroy barrier waiting for ranks that were never told to shut
    // down, and the launch error surfaces only after the ~100 s DeepEP
    // device timeout. The TP LL rendezvous rejecting a topology (poison
    // pill, NVLink probe) is a real failure landing exactly in this window.
    let (rank_arenas, pool_blocks) = match build_rank_models(
        &loaded.workers,
        max_model_len,
        moe_topo,
        &drafter,
        prefill_only.map(|options| options.chunk_size),
        &startup.ranks,
        startup.rendezvous.as_deref(),
        budget.floor_blocks,
        post_finish_reserve_bytes,
    ) {
        Ok(rank_arenas) => rank_arenas,
        Err(err) => {
            for worker in &loaded.workers {
                let _ = worker.request_shutdown();
            }
            return Err(err);
        }
    };
    let mirrored_early = moe_topo.uses_tensor_replicated_moe();
    let local_ranks_early = if mirrored_early {
        1
    } else {
        loaded.workers.len()
    };
    // The rank pools are built BEFORE the engines spawn: the store's rank
    // table freezes at build, and BlockPool is a pure CPU object with no
    // thread affinity. Each engine and the store share the same Arc.
    let pools: Vec<Arc<BlockPool>> = (0..local_ranks_early)
        .map(|_| Arc::new(BlockPool::new(GLM52_MODEL_LEN_ALIGN, pool_blocks)))
        .collect();
    let store_runtime = store_runtime_handle();
    let post_comm_startup = || -> Result<Arc<KvStore>> {
        if prefill_only.is_some() {
            preflight_prefill_kernels(&loaded.workers)?;
        }
        if let Some(dspark_path) = drafter.dspark_path() {
            load_dspark_drafters(&loaded.workers, dspark_path)?;
        }
        ensure_post_build_headroom(&loaded.workers)?;
        let device_ordinals: Vec<usize> = (0..startup.ranks.len()).collect();
        // A mirrored (TP) topology is one logical rank: worker 0's arena is
        // the primary (the save side) and every other worker's is a mirror —
        // MLA KV is tensor-replicated, so a tier load must land on ALL of
        // them or ranks 1.. attend over never-written pages and the o_proj
        // all-reduce merges the garbage identically on every rank (#847).
        let (store_arenas, store_mirrors) = if mirrored_early {
            let mut workers = rank_arenas.into_iter();
            let primary = workers.next().context("TP arenas")?;
            let mirrors = workers
                .enumerate()
                .map(|(index, arenas)| OffloadMirror {
                    device_id: device_ordinals[1 + index] as i32,
                    arenas,
                })
                .collect();
            (vec![primary], vec![mirrors])
        } else {
            let ranks = rank_arenas.len();
            (rank_arenas, (0..ranks).map(|_| Vec::new()).collect())
        };
        build_kv_store(
            kv_offload.as_ref(),
            store_arenas,
            store_mirrors,
            drafter.is_mtp(),
            &device_ordinals[..local_ranks_early],
            &pools,
            if mirrored_early {
                0
            } else {
                startup.ranks.start
            },
            &store_runtime,
        )
    };
    let store = match post_comm_startup() {
        Ok(store) => store,
        Err(err) => {
            for worker in &loaded.workers {
                let _ = worker.request_shutdown();
            }
            return Err(err);
        }
    };
    let kv_total_blocks = pool_blocks - 1;
    // One autonomous engine per LOCAL logical rank (a mirrored topology
    // collapses to a single engine driving every worker). Each engine owns
    // its submit queue and load feed, so the frontend sees one scheduler
    // partition per hosted rank.
    let mirrored = moe_topo.uses_tensor_replicated_moe();
    let local_ranks = if mirrored { 1 } else { loaded.workers.len() };
    let (load_txs, load_rxs): (Vec<_>, Vec<_>) = (0..local_ranks)
        .map(|_| {
            watch::channel(SchedulerMetrics {
                kv_total_blocks: kv_total_blocks as u64,
                ..SchedulerMetrics::default()
            })
        })
        .unzip();
    let (mut graph_dump_request, graph_dump_response) =
        match (&dump_graph_png, startup.ranks.start == 0) {
            (Some(path), true) => {
                let (response_tx, response_rx) = crossbeam_channel::bounded(1);
                (Some((path.clone(), response_tx)), Some(response_rx))
            }
            _ => (None, None),
        };
    let worker_groups: Vec<Vec<Glm52Worker>> = if mirrored {
        vec![loaded.workers]
    } else {
        loaded
            .workers
            .into_iter()
            .map(|worker| vec![worker])
            .collect()
    };
    let mut submit_txs = Vec::with_capacity(local_ranks);
    let mut startup_rxs = Vec::with_capacity(local_ranks);
    let mut join_handles = Vec::with_capacity(local_ranks);
    for (engine_index, engine_workers) in worker_groups.into_iter().enumerate() {
        let rank = if mirrored {
            0
        } else {
            startup.ranks.start + engine_index
        };
        let (submit_tx, submit_rx) = mpsc::unbounded_channel();
        let (startup_tx, startup_rx) = crossbeam_channel::bounded(1);
        let spec = Glm52EngineSpec {
            rank,
            submit_rx,
            workers: engine_workers,
            eos_token_ids: eos_token_ids.clone(),
            drafter: drafter.clone(),
            prefill_chunk_size: prefill_only.map(|prefill| prefill.chunk_size),
            pool: Arc::clone(&pools[engine_index]),
            kv_offload: kv_offload.is_some(),
            store: Arc::clone(&store),
            runtime: store_runtime.clone(),
            max_model_len,
            no_prefix_cache,
            moe_topo,
            load_tx: load_txs[engine_index].clone(),
            graph_dump_request: if engine_index == 0 {
                graph_dump_request.take()
            } else {
                None
            },
            startup_tx,
        };
        match scheduler::Glm52Engine::spawn(spec) {
            Ok(handle) => {
                submit_txs.push(submit_tx);
                startup_rxs.push(startup_rx);
                join_handles.push(handle);
            }
            Err(err) => {
                drop(submit_txs);
                for handle in join_handles {
                    let _ = handle.join();
                }
                return Err(anyhow::anyhow!(
                    "failed to spawn GLM5.2 engine for rank {rank}: {err}"
                ));
            }
        }
    }
    // Bootstrap barrier: every engine pre-captures its bucket graphs (the
    // fixed capture sequence pairs the collectives fleet-wide — that IS the
    // rendezvous) and reports once. On any failure close every queue so the
    // healthy engines exit their loops and shut their workers down
    // concurrently, then join them.
    let abort_engines =
        |submit_txs: Vec<mpsc::UnboundedSender<pegainfer_frontend::engine::SubmittedRequest>>,
         join_handles: Vec<std::thread::JoinHandle<()>>| {
            drop(submit_txs);
            for handle in join_handles {
                let _ = handle.join();
            }
        };
    for (engine_index, startup_rx) in startup_rxs.iter().enumerate() {
        let report = startup_rx.recv();
        let failed = match report {
            Ok(Ok(())) => None,
            Ok(Err(err)) => Some(err),
            Err(_) => Some(anyhow::anyhow!(
                "GLM5.2 engine {engine_index} exited before reporting bootstrap"
            )),
        };
        if let Some(err) = failed {
            abort_engines(submit_txs, join_handles);
            return Err(err.context("GLM5.2 engine bootstrap failed"));
        }
    }
    if let Some(response) = graph_dump_response {
        let summary = response
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank 0 exited before reporting graph export"))
            .and_then(|result| result);
        let summary = match summary {
            Ok(summary) => summary,
            Err(err) => {
                abort_engines(submit_txs, join_handles);
                return Err(err.context("GLM5.2 CUDA Graph export failed"));
            }
        };
        log::info!(
            "GLM5.2 decode CUDA Graph exported: nodes={}, kernels={}, edges={}, dot={}, png={}",
            summary.nodes,
            summary.kernels,
            summary.edges,
            summary.dot_path.display(),
            summary.png_path.display()
        );
    }
    // Publish the launch-time cap so the frontend clamps its config.json
    // max_position_embeddings (1M) at the API boundary instead of admitting
    // requests the scheduler would reject (same contract as qwen3/dsv2-lite).
    let servable_len = u32::try_from(max_model_len)
        .expect("max_model_len is bounded by GLM52_MAX_CONTEXT and fits u32");
    Ok(
        EngineHandle::new_with_join_handles(submit_txs, join_handles)
            .with_servable_len(servable_len)
            .with_kv_capacity(KvCapacity {
                total_blocks: kv_total_blocks,
                block_size: GLM52_MODEL_LEN_ALIGN,
            })
            .with_metrics_watches(load_rxs),
    )
}

fn preflight_prefill_kernels(workers: &[Glm52Worker]) -> Result<()> {
    let started = Instant::now();
    let responses = workers
        .iter()
        .map(|worker| {
            worker.prefill_chunk_async(Glm52PrefillBatch {
                token_ids: vec![0],
                positions: vec![0],
                request_indptr: vec![0, 1],
                block_indptr: vec![0, 1],
                block_ids: vec![0],
                request_slots: vec![0],
                padding_block: 1,
                slot_mapping: vec![0],
                mtp_next_tokens: vec![Some(0)],
                output_rows: Vec::new(),
                sampling: Vec::new(),
                seed: 0,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    for (rank, response) in responses.into_iter().enumerate() {
        response.recv().map_err(|_| {
            anyhow::anyhow!("GLM5.2 rank {rank} dropped its prefill preflight response")
        })??;
    }
    log::info!(
        "GLM5.2 TP4 prefill kernel preflight completed on all ranks in {:.3}s",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Load the DSpark drafter on every rank (rank-local, ~3.8 GB bf16 each —
/// the draft's embed/lm_head reuse the target's, so they are never loaded).
fn load_dspark_drafters(workers: &[Glm52Worker], dspark_path: &Path) -> Result<()> {
    let started = Instant::now();
    let responses = workers
        .iter()
        .map(|worker| worker.load_dspark_async(dspark_path))
        .collect::<Result<Vec<_>>>()?;
    for (rank, response) in responses.into_iter().enumerate() {
        response.recv().map_err(|_| {
            anyhow::anyhow!("GLM5.2 rank {rank} dropped its dspark-load response")
        })??;
    }
    log::info!(
        "GLM5.2 DSpark drafter loaded on all ranks in {:.2}s (speculative decoding: verify \
         spans ride the decode buckets, accept stats logged per request)",
        started.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Re-probe every rank once everything the reserve constants stand in for is
/// resident (model arenas, dequanted MLA weights, DeepEP contexts, optional
/// drafter): if any rank is left with less headroom than the whole-step
/// graph instantiations and allocator slack need, fail the launch with the
/// numbers — a reserve/ledger drift must crash here, not as a mid-serving
/// OOM that tears the collective group down.
fn ensure_post_build_headroom(workers: &[Glm52Worker]) -> Result<()> {
    let responses = workers
        .iter()
        .map(Glm52Worker::free_vram_async)
        .collect::<Result<Vec<_>>>()?;
    let mut per_rank = Vec::with_capacity(responses.len());
    for (rank, response) in responses.into_iter().enumerate() {
        let free = response
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped its VRAM-probe response"))??;
        ensure!(
            free >= GLM52_POST_BUILD_MIN_FREE_BYTES,
            "GLM5.2 rank {rank} has only {} free VRAM after build (< {} headroom for graph \
             capture); lower --max-model-len or free device memory",
            ByteSize(free as u64),
            ByteSize(GLM52_POST_BUILD_MIN_FREE_BYTES as u64),
        );
        per_rank.push(free);
    }
    log::info!(
        "GLM5.2 post-build free VRAM per rank: {:?}",
        format_bytes(&per_rank)
    );
    Ok(())
}

/// Build every rank's resident model in the measured two phases, then create
/// the collective contexts. Phases on purpose: the build is per-rank and can
/// fail (OOM, packaging drift) — every rank must report success BEFORE
/// anyone enters context creation, or a single failure strands peer ranks in
/// a collective init with no useful error. Between the phases the fleet's
/// measured free-VRAM minimum decides the KV pool block count, published
/// process-wide (the schedulers' `BlockPool` reads it) before any FinishKv
/// dispatch. TP4 currently stops after the per-rank build, before entering
/// any EP/TP collective setup.
#[allow(clippy::too_many_arguments)]
fn build_rank_models(
    workers: &[Glm52Worker],
    max_model_len: usize,
    moe_topo: Glm52MoeTopo,
    drafter: &Glm52Drafter,
    prefill_chunk_size: Option<usize>,
    ranks: &Range<usize>,
    rendezvous: Option<&str>,
    floor_blocks: usize,
    post_finish_reserve_bytes: usize,
) -> Result<(Vec<Vec<ArenaSpec>>, usize)> {
    let build_started = Instant::now();
    let prefill_only = prefill_chunk_size.is_some();
    let responses = workers
        .iter()
        .map(|worker| {
            worker.build_fixed_async(max_model_len, moe_topo, drafter.clone(), prefill_chunk_size)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut rank_free_bytes = Vec::with_capacity(responses.len());
    for (local_index, response) in responses.into_iter().enumerate() {
        // Label with the global rank, not the local worker slot — on a
        // multi-process fleet every process would otherwise call its first
        // worker "rank 0".
        let rank = ranks.start + local_index;
        rank_free_bytes.push(
            response.recv().map_err(|_| {
                anyhow::anyhow!("GLM5.2 rank {rank} dropped its fixed-build response")
            })?? as usize,
        );
    }
    let min_free_bytes = rank_free_bytes
        .iter()
        .copied()
        .min()
        .expect("at least one rank built");
    // The exact per-block slab cost, taken from the SAME ledger the
    // allocations use (linear in blocks, so one difference is the marginal):
    // 78 MLA pages + the full-indexer index-K blocks + the MTP mirror and
    // the prefill unpacked page, whichever of those this launch carries.
    let slab_bytes_per_block = glm52_cap_bytes(
        max_model_len,
        floor_blocks + 1,
        drafter,
        prefill_only,
        moe_topo,
    )?
    .saturating_sub(glm52_cap_bytes(
        max_model_len,
        floor_blocks,
        drafter,
        prefill_only,
        moe_topo,
    )?);
    let pool_blocks = match std::env::var("GLM52_KV_POOL_BLOCKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(pinned) => {
            // A pin below the one-request floor would advertise a cap no
            // request can ever fit — admission then parks the impossible
            // request at the FIFO head forever. And a pin above the measured
            // affordable count would either fail slab allocation outright or
            // eat into the graph reserve and die at pre-capture — the A/B
            // knob must stay inside both fences. Refuse loudly either way.
            let one_request = glm52_one_request_pool_blocks(max_model_len);
            ensure!(
                pinned >= one_request,
                "GLM52_KV_POOL_BLOCKS={pinned} is below the {one_request}-block one-request \
                 floor for max_model_len {max_model_len}"
            );
            let affordable = glm52_measured_pool_blocks(
                min_free_bytes,
                post_finish_reserve_bytes,
                slab_bytes_per_block,
                one_request,
            );
            ensure!(
                pinned <= affordable,
                "GLM52_KV_POOL_BLOCKS={pinned} exceeds the {affordable}-block measured \
                 affordable count (min rank free {} - reserves at {} per block)",
                ByteSize(min_free_bytes as u64),
                ByteSize(slab_bytes_per_block as u64),
            );
            pinned
        }
        None => glm52_measured_pool_blocks(
            min_free_bytes,
            post_finish_reserve_bytes,
            slab_bytes_per_block,
            floor_blocks,
        ),
    };
    log::info!(
        "GLM5.2 KV pool: {} blocks ({} tokens, {} past the {}-slot nominal) — the measured \
         free-VRAM fill (min rank free {} after the fixed build, graph reserve {}, post-finish \
         charges {}, {}/block) decouples pool capacity from the {max_model_len}-token cap (#818)",
        pool_blocks,
        pool_blocks * GLM52_MODEL_LEN_ALIGN,
        pool_blocks.saturating_sub(glm52_nominal_pool_blocks(max_model_len)),
        model::glm52_decode_slots(),
        ByteSize(min_free_bytes as u64),
        ByteSize(glm52_graph_reserve_bytes() as u64),
        ByteSize(post_finish_reserve_bytes as u64),
        ByteSize(slab_bytes_per_block as u64),
    );
    let responses = workers
        .iter()
        .map(|worker| worker.finish_kv_async(pool_blocks))
        .collect::<Result<Vec<_>>>()?;
    let mut rank_arenas = Vec::with_capacity(responses.len());
    for (local_index, response) in responses.into_iter().enumerate() {
        let rank = ranks.start + local_index;
        rank_arenas.push(
            response
                .recv()
                .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped its build response"))??,
        );
    }
    let unique_id = if moe_topo.uses_ep_expert_bundles() {
        // One-time bootstrap rendezvous: the rank-0-hosting process generates
        // and serves the id, every other process fetches it (single-process
        // fleets generate in-process and never touch the network).
        rendezvous::unique_id(moe_topo.expected_ep_size(), ranks, rendezvous)?
    } else {
        // TP allreduce bootstrap just needs one NCCL unique id; ride the EP8
        // shim's generator.
        pegainfer_kernels::ops::glm52_ep_deepep_unique_id(8)?
    };
    let tp_exchange = moe_topo
        .uses_tensor_replicated_moe()
        .then(|| std::sync::Arc::new(crate::moe_tp::Glm52TpExchange::new(moe_topo.device_count())));
    if tp_exchange.is_some() {
        // The prefill-only NCCL all-reduce moves ~200 MB per layer; NCCL's
        // default channel count on this single-host NVLink topology comes up
        // as 2 and caps the ring at ~46 GB/s (measured 8.7 ms per 16K-row
        // all-reduce). 16..32 channels restores ~5x (measured 1.6 ms).
        // Set BEFORE any worker thread initializes its communicator; user
        // overrides win. Single-threaded here, so set_var is race-free.
        for (key, value) in [("NCCL_MIN_NCHANNELS", "16"), ("NCCL_MAX_NCHANNELS", "32")] {
            if std::env::var_os(key).is_none() {
                unsafe { std::env::set_var(key, value) };
            }
        }
    }
    let responses = workers
        .iter()
        .map(|worker| worker.setup_comm_async(unique_id, moe_topo, tp_exchange.clone()))
        .collect::<Result<Vec<_>>>()?;
    for (local_index, response) in responses.into_iter().enumerate() {
        let rank = ranks.start + local_index;
        response
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} dropped its comm-setup response"))??;
    }
    log::info!(
        "GLM5.2 rank models built in {:.2}s (weights adopted in place + {:?} contexts up)",
        build_started.elapsed().as_secs_f64(),
        moe_topo
    );
    Ok((rank_arenas, pool_blocks))
}

/// One shared pegaflow host (one pinned pool) with each rank's arenas
/// registered as its own instance under a single namespace — replicated
/// non-expert weights make DP ranks' KV interchangeable, so any rank
/// restores what any rank saved.
/// The namespace folds the layout facts that make blocks interchange-safe
/// (per-token packing, page size, layer count); pool capacity deliberately
/// stays out (a block's bytes don't depend on it).
/// Build the process-wide [`KvStore`]: every logical rank's pool, and — with
/// offload configured — one shared [`PegaflowHost`] plus a per-rank pegaflow
/// instance registration. All DP ranks share one namespace (replicated
/// non-expert weights make their KV interchangeable), folding the layout
/// facts that make blocks interchange-safe.
fn build_kv_store(
    opts: Option<&Glm52KvOffloadOptions>,
    rank_arenas: Vec<Vec<ArenaSpec>>,
    rank_mirrors: Vec<Vec<OffloadMirror>>,
    native_mtp: bool,
    device_ordinals: &[usize],
    pools: &[Arc<BlockPool>],
    ranks_start: usize,
    runtime: &tokio::runtime::Handle,
) -> Result<Arc<KvStore>> {
    // The store's rank table is keyed by GLOBAL logical rank — the engines
    // query it with `ranks.start + local_index` on a multi-process fleet.
    let mut builder = KvStoreBuilder::new(runtime.clone());
    let Some(opts) = opts else {
        for (local, pool) in pools.iter().enumerate() {
            builder = builder.rank(ranks_start + local, Arc::clone(pool));
        }
        return Ok(Arc::new(builder.build()));
    };

    // One page-granular arena per rank; the page layout constants ARE the
    // block byte layout, so the registration only cross-checks that every
    // rank's arena carries them.
    ensure!(
        rank_arenas.iter().all(|arenas| {
            arenas.len() == 1
                && arenas[0].name == "glm52.page"
                && arenas[0].segment_bytes == crate::model::GLM52_KV_PAGE_CONTENT_BYTES
                && arenas[0].block_stride_bytes == crate::model::GLM52_KV_PAGE_STRIDE
        }),
        "GLM5.2 KV offload expects one page-first arena per rank \
         (segment {} B, stride {} B)",
        crate::model::GLM52_KV_PAGE_CONTENT_BYTES,
        crate::model::GLM52_KV_PAGE_STRIDE,
    );
    let mut host_builder = PegaflowHost::builder(opts.pinned_pool_bytes)
        .use_hugepages(opts.use_hugepages)
        .runtime_threads(2);
    if let Some(p2p) = &opts.p2p {
        host_builder = host_builder.p2p(P2pConfig {
            metaserver_addr: p2p.metaserver_addr.clone(),
            advertise_addr: p2p.advertise_addr.clone(),
            rdma_nics: p2p.rdma_nics.clone(),
        });
    }
    let host = host_builder
        .build()
        .map_err(|err| anyhow::anyhow!("GLM5.2 KV offload host: {err}"))?;
    // The stride is the layout identity: one page carries every layer's MLA
    // + index-K slices and the L78 MTP mirror slices (drafter or not), so
    // agreeing on (layer count, token page, stride) is agreeing on every
    // byte of a block's SHAPE. The mtp flag is the content half: a
    // drafterless producer reserves the L78 slices but never writes them,
    // and a native-MTP consumer restoring such a page would propose from
    // zeros — the two configs must not see each other's blocks.
    let namespace = format!(
        "pegainfer-glm52-l{GLM52_LAYERS}-p{}-page{}-mtp{}",
        pegainfer_kernels::ops::GLM52_FLASHMLA_SPARSE_PAGE_SIZE,
        crate::model::GLM52_KV_PAGE_STRIDE,
        usize::from(native_mtp),
    );
    ensure!(
        rank_arenas.len() == pools.len() && device_ordinals.len() == pools.len(),
        "GLM5.2 KV offload: {} arena sets, {} pools, {} devices must agree",
        rank_arenas.len(),
        pools.len(),
        device_ordinals.len()
    );
    ensure!(
        rank_mirrors.len() == pools.len(),
        "GLM5.2 KV offload: {} mirror sets for {} pools",
        rank_mirrors.len(),
        pools.len()
    );
    let ranks = rank_arenas.len();
    let arenas_per_rank = rank_arenas.first().map_or(0, Vec::len);
    let mirrors_per_rank = rank_mirrors.first().map_or(0, Vec::len);
    for (local, ((arenas, mirrors), &device_ordinal)) in rank_arenas
        .into_iter()
        .zip(rank_mirrors)
        .zip(device_ordinals)
        .enumerate()
    {
        let rank = ranks_start + local;
        builder = builder
            .rank_with_offload(
                rank,
                Arc::clone(&pools[local]),
                &host,
                OffloadRankSpec {
                    instance_id: format!("glm52-rank{rank}"),
                    namespace: namespace.clone(),
                    device_id: device_ordinal as i32,
                    arenas,
                    page_first: false,
                    mirrors,
                },
            )
            .map_err(|err| {
                anyhow::anyhow!("GLM5.2 KV offload rank {rank} registration: {err:#}")
            })?;
    }
    log::info!(
        "GLM5.2 KV offload up: {} pinned host pool (hugepages: {}), namespace {namespace}, \
         {ranks} rank instances x {arenas_per_rank} arenas ({mirrors_per_rank} mirrors each)",
        ByteSize(opts.pinned_pool_bytes as u64),
        opts.use_hugepages,
    );
    Ok(Arc::new(builder.build()))
}

/// The store's watcher/resolver runtime: the ambient tokio runtime when the
/// loader runs under one (pegainfer-server's `spawn_blocking` keeps the
/// context), else a small dedicated runtime living for the process.
fn store_runtime_handle() -> tokio::runtime::Handle {
    if let Ok(handle) = tokio::runtime::Handle::try_current() {
        return handle;
    }
    static FALLBACK: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    FALLBACK
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .expect("kv-store fallback runtime")
        })
        .handle()
        .clone()
}

/// EOS ids from the checkpoint's generation_config.json (`eos_token_id` is a
/// number or an array of numbers).
fn read_eos_token_ids(model_path: &Path) -> Result<Vec<u32>> {
    let path = model_path.join("generation_config.json");
    let content = std::fs::read_to_string(&path)
        .map_err(|err| anyhow::anyhow!("read {}: {err}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|err| anyhow::anyhow!("parse {}: {err}", path.display()))?;
    let field = json
        .get("eos_token_id")
        .ok_or_else(|| anyhow::anyhow!("{} missing eos_token_id", path.display()))?;
    let as_u32 = |value: &serde_json::Value| -> Result<u32> {
        value
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| anyhow::anyhow!("eos_token_id entry {value} is not a u32"))
    };
    let ids = match field {
        serde_json::Value::Array(entries) => {
            entries.iter().map(as_u32).collect::<Result<Vec<_>>>()?
        }
        other => vec![as_u32(other)?],
    };
    ensure!(!ids.is_empty(), "eos_token_id list is empty");
    Ok(ids)
}

fn validate_startup(
    model_path: &Path,
    options: &Glm52LoadOptions,
    moe_topo: Glm52MoeTopo,
    native_mtp: bool,
) -> Result<StartupValidation> {
    let config_path = model_path.join("config.json");
    let content = std::fs::read_to_string(&config_path)
        .map_err(|err| anyhow::anyhow!("read {}: {err}", config_path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&content)
        .map_err(|err| anyhow::anyhow!("parse {}: {err}", config_path.display()))?;
    probe_config_json(&json)?;

    let expected_devices = moe_topo.device_count();
    ensure!(
        options.ranks.end <= expected_devices && options.ranks.start < options.ranks.end,
        "GLM5.2 {moe_topo:?} load requires ranks within 0..{expected_devices}, got {}..{}",
        options.ranks.start,
        options.ranks.end
    );
    ensure!(
        options.tp_size == moe_topo.expected_tp_size()
            && options.dp_size == moe_topo.default_dp_size()
            && options.ep_size == moe_topo.expected_ep_size(),
        "GLM5.2 {moe_topo:?} requires TP{}/DP{}/EP{}, got TP{} DP{} EP{}",
        moe_topo.expected_tp_size(),
        moe_topo.default_dp_size(),
        moe_topo.expected_ep_size(),
        options.tp_size,
        options.dp_size,
        options.ep_size
    );

    let manifest = Glm52WeightManifest::from_model_dir(model_path)?;
    let rank_bundles = manifest.all_rank_load_bundles(moe_topo, native_mtp)?;
    let mut rank_tensor_counts = Vec::with_capacity(options.ranks.len());
    let mut rank_expert_ranges = Vec::with_capacity(options.ranks.len());
    for bundle in &rank_bundles[options.ranks.clone()] {
        rank_tensor_counts.push(bundle.plan.tensor_count);
        rank_expert_ranges.push(bundle.plan.expert_range.clone());
    }

    log::info!(
        "GLM5.2 load-weight startup validated: model_path={}, fleet_ranks={}, hosted_ranks={:?}, logical_parallel=TP{} DP{} EP{}, rank_expert_ranges={:?}, rank_plan_tensors={:?}",
        model_path.display(),
        rank_bundles.len(),
        options.ranks,
        options.tp_size,
        options.dp_size,
        options.ep_size,
        rank_expert_ranges,
        rank_tensor_counts,
    );

    Ok(StartupValidation {
        ranks: options.ranks.clone(),
        rendezvous: options.rendezvous.clone(),
        rank_bundles,
        rank_tensor_counts,
        rank_expert_ranges,
    })
}

fn load_rank_weights_to_gpu(
    model_path: &Path,
    startup: &StartupValidation,
    moe_topo: Glm52MoeTopo,
    weight_staging: bool,
) -> Result<LoadedGlm52Runtime> {
    let spawn_started = Instant::now();
    log::info!(
        "start spawn GLM5.2 rank workers: hosted_ranks={:?}",
        startup.ranks,
    );
    let mut workers = Vec::with_capacity(startup.ranks.len());
    for (device_ordinal, rank) in startup.ranks.clone().enumerate() {
        let placement = Glm52RankPlacement {
            rank,
            device_ordinal,
        };
        workers.push(Glm52RankWorker::spawn(
            placement,
            startup.rank_bundles[rank].clone(),
        )?);
    }
    log::info!(
        "spawn GLM5.2 rank workers cost {:.2}s: ranks={}",
        spawn_started.elapsed().as_secs_f64(),
        workers.len()
    );

    let load_started = Instant::now();
    log::info!(
        "start load GLM5.2 rank weights: ranks={}, rank_expert_ranges={:?}",
        workers.len(),
        startup.rank_expert_ranges,
    );
    let load_results = workers
        .iter()
        .map(|worker| worker.load_weights_async(model_path, moe_topo, weight_staging))
        .collect::<Result<Vec<_>>>()?;
    let mut reports = Vec::with_capacity(load_results.len());
    for (local_index, rx) in load_results.into_iter().enumerate() {
        // `report.rank` is the global fleet rank; the enumerate index is only
        // this process's worker slot, so recompute the rank from the hosted
        // range before validating (matters when ranks.start != 0).
        let rank = startup.ranks.start + local_index;
        let report = rx
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank {rank} worker dropped load response"))??;
        ensure!(
            report.rank == rank && report.loaded_to_gpu,
            "GLM5.2 rank {rank} invalid weight-load report: {:?}",
            report
        );
        reports.push(report);
    }
    let rank_tensor_counts = reports
        .iter()
        .map(|report| report.loaded_tensor_count)
        .collect::<Vec<_>>();
    let rank_bytes = reports
        .iter()
        .map(|report| report.loaded_total_bytes)
        .collect::<Vec<_>>();
    let rank_free_vram_bytes = reports
        .iter()
        .map(|report| report.free_vram_bytes)
        .collect::<Vec<_>>();
    log::info!(
        "GLM5.2 rank weight load cost {:.2}s: ranks={}, tensors={:?}, resident_bytes={:?}",
        load_started.elapsed().as_secs_f64(),
        reports.len(),
        rank_tensor_counts,
        format_bytes(&rank_bytes),
    );

    Ok(LoadedGlm52Runtime {
        workers,
        report: GpuWeightLoadReport {
            tensor_counts: rank_tensor_counts,
            bytes: rank_bytes,
            free_vram_bytes: rank_free_vram_bytes,
        },
    })
}

fn format_bytes(values: &[usize]) -> Vec<String> {
    values
        .iter()
        .map(|&value| ByteSize(value as u64).to_string())
        .collect()
}

#[cfg(test)]
mod max_model_len_tests {
    use super::*;

    const TEST_TOPO: Glm52MoeTopo = Glm52MoeTopo::Ep8;

    /// Free VRAM that budgets exactly a `cap`-token context (exact ledger +
    /// reserve) — inverted through the same `glm52_cap_bytes` the derivation
    /// uses, so the tests exercise the policy, not a parallel formula.
    fn free_for(cap: usize, drafter: &Glm52Drafter, prefill_scratch_bytes: usize) -> usize {
        let reserve = GLM52_VRAM_RESERVE_BYTES
            + if drafter.is_dspark() {
                GLM52_DSPARK_VRAM_RESERVE_BYTES
            } else {
                0
            }
            + prefill_scratch_bytes;
        reserve
            + glm52_cap_bytes(
                cap,
                glm52_nominal_pool_blocks(cap),
                drafter,
                false,
                TEST_TOPO,
            )
            .expect("cap bytes")
    }

    #[test]
    fn derived_cap_is_aligned_and_scales_with_free_vram() {
        let cap = derive_max_model_len(
            None,
            free_for(10_048, &Glm52Drafter::None, 0),
            &Glm52Drafter::None,
            0,
            false,
            TEST_TOPO,
        )
        .expect("derive")
        .max_model_len;
        assert_eq!(cap, 10_048, "exact budget for an aligned cap derives it");
        assert!(cap.is_multiple_of(GLM52_MODEL_LEN_ALIGN));
        let larger = derive_max_model_len(
            None,
            free_for(50_048, &Glm52Drafter::None, 0),
            &Glm52Drafter::None,
            0,
            false,
            TEST_TOPO,
        )
        .expect("derive")
        .max_model_len;
        assert!(larger > cap);
    }

    #[test]
    fn floor_is_the_nominal_pool_when_the_ledger_affords_it() {
        // Identical defaults to the pre-measured fill: a budget that covers
        // the legacy `slots x cap` pool floors the measured fill there.
        let drafter = Glm52Drafter::NativeMtp;
        let free = free_for(50_048, &drafter, 0) + (64 << 20);
        let budget = derive_max_model_len(Some(50_048), free, &drafter, 0, false, TEST_TOPO)
            .expect("derive");
        assert_eq!(budget.floor_blocks, glm52_nominal_pool_blocks(50_048));
    }

    #[test]
    fn floor_falls_back_to_a_one_request_pool_for_an_explicit_big_cap() {
        // The decoupling's raison d'être: an explicit cap whose nominal
        // `slots x cap` pool overflows the budget still launches, floored at
        // a pool holding ONE max-length request.
        let cap = 99_968;
        let one_request = glm52_one_request_pool_blocks(cap);
        let free = GLM52_VRAM_RESERVE_BYTES
            + glm52_cap_bytes(cap, one_request, &Glm52Drafter::None, false, TEST_TOPO)
                .expect("cap bytes")
            + glm52_graph_reserve_bytes()
            + (64 << 20);
        let budget =
            derive_max_model_len(Some(cap), free, &Glm52Drafter::None, 0, false, TEST_TOPO)
                .expect("derive");
        assert_eq!(budget.floor_blocks, one_request);
        assert!(one_request < glm52_nominal_pool_blocks(cap));
    }

    #[test]
    fn measured_fill_spends_free_vram_at_the_exact_per_block_cost() {
        let floor = 100;
        let slab = 1 << 20;
        let reserve = glm52_graph_reserve_bytes();
        // Nothing past the reserve: the floor is never reduced.
        assert_eq!(glm52_measured_pool_blocks(reserve, 0, slab, floor), floor);
        // Every byte past the reserve buys blocks at the exact marginal.
        assert_eq!(
            glm52_measured_pool_blocks(reserve + 512 * slab, 0, slab, floor),
            512
        );
        // Exactly-known post-finish charges are held back alongside it.
        assert_eq!(
            glm52_measured_pool_blocks(reserve + 512 * slab, 12 * slab, slab, floor),
            500
        );
    }

    #[test]
    fn dspark_lane_shrinks_the_derived_cap() {
        let free = free_for(50_048, &Glm52Drafter::None, 0);
        let plain = derive_max_model_len(None, free, &Glm52Drafter::None, 0, false, TEST_TOPO)
            .expect("derive");
        let dspark_drafter = Glm52Drafter::Dspark(PathBuf::from("draft"));
        let dspark =
            derive_max_model_len(None, free, &dspark_drafter, 0, false, TEST_TOPO).expect("derive");
        assert!(
            dspark.max_model_len < plain.max_model_len,
            "dspark cap-scaled cost must shrink the cap"
        );
    }

    #[test]
    fn native_mtp_lane_shrinks_the_derived_cap() {
        let free = free_for(50_048, &Glm52Drafter::None, 0);
        let plain = derive_max_model_len(None, free, &Glm52Drafter::None, 0, false, TEST_TOPO)
            .expect("derive");
        let native_mtp =
            derive_max_model_len(None, free, &Glm52Drafter::NativeMtp, 0, false, TEST_TOPO)
                .expect("derive");
        assert!(
            native_mtp.max_model_len < plain.max_model_len,
            "native MTP cap-scaled KV must shrink the target context cap"
        );
        assert!(
            glm52_cap_bytes(
                50_048,
                glm52_nominal_pool_blocks(50_048),
                &Glm52Drafter::NativeMtp,
                false,
                TEST_TOPO
            )
            .expect("MTP cap bytes")
                > glm52_cap_bytes(
                    50_048,
                    glm52_nominal_pool_blocks(50_048),
                    &Glm52Drafter::None,
                    false,
                    TEST_TOPO
                )
                .expect("plain cap bytes"),
            "native MTP must be represented in the exact memory ledger"
        );
    }

    #[test]
    fn tp4_native_mtp_charges_the_execution_and_wire_caches() {
        let cap = 16_384;
        let tp4 = glm52_cap_bytes(
            cap,
            glm52_nominal_pool_blocks(cap),
            &Glm52Drafter::NativeMtp,
            true,
            Glm52MoeTopo::Tp4,
        )
        .expect("TP4 cap bytes");
        let ep4 = glm52_cap_bytes(
            cap,
            glm52_nominal_pool_blocks(cap),
            &Glm52Drafter::NativeMtp,
            true,
            Glm52MoeTopo::Ep4,
        )
        .expect("EP4 cap bytes");
        assert!(
            tp4 > ep4,
            "TP4 must charge its additional execution-layout cache"
        );
    }

    #[test]
    fn derived_cap_never_exceeds_the_checkpoint_ceiling() {
        let budget = derive_max_model_len(
            None,
            usize::MAX / 2,
            &Glm52Drafter::None,
            0,
            false,
            TEST_TOPO,
        )
        .expect("derive");
        assert_eq!(budget.max_model_len, GLM52_MAX_CONTEXT);
    }

    #[test]
    fn too_little_vram_fails_instead_of_serving_a_toy_cap() {
        let err = derive_max_model_len(
            None,
            free_for(1024, &Glm52Drafter::None, 0),
            &Glm52Drafter::None,
            0,
            false,
            TEST_TOPO,
        )
        .expect_err("sub-minimum cap must fail");
        assert!(err.to_string().contains("context cap"), "{err}");
    }

    #[test]
    fn unaligned_requested_cap_is_rejected_with_the_nearest_valid_values() {
        let err = derive_max_model_len(
            Some(5000),
            free_for(100_032, &Glm52Drafter::None, 0),
            &Glm52Drafter::None,
            0,
            false,
            TEST_TOPO,
        )
        .expect_err("unaligned cap must fail, not silently round");
        let message = err.to_string();
        assert!(
            message.contains("4992") && message.contains("5056"),
            "{message}"
        );
    }

    #[test]
    fn requested_cap_beyond_the_budget_fails_at_launch() {
        // Post-decoupling an explicit cap only has to afford the fixed
        // arenas plus a ONE-request pool (concurrency comes from the
        // measured fill), so the rejection boundary is a budget too small
        // for even that — profile-independent by construction.
        let one_request = glm52_cap_bytes(
            99_968,
            glm52_one_request_pool_blocks(99_968),
            &Glm52Drafter::None,
            false,
            TEST_TOPO,
        )
        .expect("cap bytes");
        let reserve = GLM52_VRAM_RESERVE_BYTES;
        let err = derive_max_model_len(
            Some(99_968),
            reserve + one_request - (1 << 20),
            &Glm52Drafter::None,
            0,
            false,
            TEST_TOPO,
        )
        .expect_err("a budget below the one-request pool must fail");
        assert!(err.to_string().contains("--max-model-len"), "{err}");
    }

    #[test]
    fn requested_cap_below_the_minimum_fails() {
        derive_max_model_len(
            Some(1024),
            free_for(100_032, &Glm52Drafter::None, 0),
            &Glm52Drafter::None,
            0,
            false,
            TEST_TOPO,
        )
        .expect_err("sub-minimum cap must fail");
    }

    #[test]
    fn prefill_pool_budgets_the_full_slot_count_plus_scratch() {
        let prefill = Glm52PrefillOnlyOptions {
            chunk_size: GLM52_DEFAULT_PREFILL_CHUNK_SIZE,
        };
        let scratch =
            glm52_prefill_scratch_reservation(Some(prefill)).expect("prefill reservation");
        let free = free_for(100_032, &Glm52Drafter::None, 0);
        let decode = derive_max_model_len(None, free, &Glm52Drafter::None, 0, false, TEST_TOPO)
            .expect("decode budget");
        let prefill =
            derive_max_model_len(None, free, &Glm52Drafter::None, scratch, true, TEST_TOPO)
                .expect("prefill budget");
        // Prefill-only budgets the full slot count too (the prefix cache
        // retains released prefixes in the pool headroom), so its cap can
        // only trail the decode cap — by exactly the scratch reservation.
        assert!(
            prefill.max_model_len <= decode.max_model_len,
            "prefill cap {} must not exceed the decode cap {} once both \
             budget the full slot count",
            prefill.max_model_len,
            decode.max_model_len,
        );
        assert_eq!(prefill.reserve_bytes, GLM52_VRAM_RESERVE_BYTES + scratch);
    }

    #[test]
    fn decode_pool_rounds_each_slots_dangling_token_page() {
        let cap = 4096usize;
        let pages_per_slot = (cap + 1).div_ceil(GLM52_MODEL_LEN_ALIGN);
        assert_eq!(
            glm52_pool_blocks(cap, model::GLM52_MAX_BATCH_PER_RANK),
            model::GLM52_MAX_BATCH_PER_RANK * pages_per_slot + 1
        );
    }

    #[test]
    fn default_prefill_chunk_reservation_is_stable() {
        let bytes = glm52_prefill_scratch_reservation(Some(Glm52PrefillOnlyOptions {
            chunk_size: GLM52_DEFAULT_PREFILL_CHUNK_SIZE,
        }))
        .expect("prefill reservation");
        // 3 GiB fixed (row-block MoE scratch + sub-tile buffers) plus
        // 160 KiB x 16384 chunk rows of chunk-scale activations.
        assert_eq!(bytes, 5_905_580_032);
    }
}
