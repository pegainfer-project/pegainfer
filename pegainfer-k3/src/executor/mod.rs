//! The GPU decode executor: one rank's model, its slot state and the step that
//! turns a batch of last tokens into a batch of next tokens.
//!
//! ## Slots
//!
//! A slot *is* a batch row: slot `i` owns row `i` of every state slab, and a
//! step runs the compiled bucket that covers the highest occupied slot. Rows a
//! step does not own are computed and discarded — the batched kernels are
//! row-independent, so their presence cannot move an owned row's result. There
//! is no compaction yet: a sparse slot set costs a wider bucket, not a wrong
//! answer.
//!
//! What a padding row cannot do is stay still. It is fed a placeholder token
//! and its recurrent state advances like everyone else's, so **a seat's state
//! is only meaningful while the seat is in every batch**. That is the contract
//! the scheduler already keeps — a running request is decoded every step — and
//! [`K3Executor::prefill`] resets the seat at admission, so a sequence never
//! inherits what the padding steps left behind. A caller that parks a live
//! sequence for a step and comes back to it gets the parked row destroyed.
//!
//! ## Prefill
//!
//! Prefill runs the batched step over **chunks of one sequence**: the bucket's
//! rows carry up to `chunk_tokens` consecutive prompt tokens (default: the
//! 4224-row MegaMoE protocol maximum, clamped to `max_ctx`), so every
//! row-independent stage (norms, projections, MoE) digests the whole chunk
//! in one launch, the MLA layers attend `[context | chunk]` through one
//! dense FlashMLA FMHA call per layer over kv_b-expanded scratch, and the
//! KDA recurrence crosses the chunk as one chunkwise FlashKDA forward per
//! layer ([`step::k3_prefill_chunk_step`]). Chunk steps skip the batched
//! epilogue; the boundary token is sampled once after the final chunk
//! ([`step::k3_prefill_boundary_sample`]). It runs on a
//! **separate state pool**
//! rather than on the sequence's own slot, because a batched step advances
//! every row of its bucket: prefilling in place would step the sequences
//! already decoding. The pool keeps one row of KDA/conv state (the recurrence
//! is sequential anyway) but a full bucket of attention-residual snapshots and
//! block-table rows. When the prompt is consumed the pool's state is copied
//! into the slot, and the slot joins the batch.
//!
//! ## Graphs
//!
//! Decode is captured per (bucket, parity) and replayed; the H2D feed of the
//! step's inputs and the D2H of the sampled ids stay outside the capture, as in
//! the reference engine. `PEGAINFER_K3_CUDA_GRAPH=0` forces the eager path.
//!
//! `ep_size > 1` runs eagerly for now — capture works on the single-rank fused
//! path, but a captured cross-rank launch has never been replayed here and the
//! kernel's device barriers pair the world inside it, so that is its own piece
//! of work rather than a default.
//!
//! ## Expert parallelism
//!
//! Ranks are free-running: each has its own scheduler thread and its own
//! requests, and the only runtime coupling is the fused MoE launch inside the
//! step ([`ep`]). Two consequences show up here:
//!
//! * an **empty batch is a real step**, not an early return — the rank owes its
//!   peers a launch per MoE layer either way, so it takes one with every row
//!   padding;
//! * **any error is fatal**. A rank that skips a step leaves every peer inside a
//!   device barrier it will never reach, so the rank exits the process instead
//!   of returning into the scheduler's keep-serving path.

mod buffers;
pub mod ep;
mod gemm;
mod paged_kv;
mod step;

use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::sys::CUdevice_attribute;
use log::info;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_kernels::ops::K3_BATCH_BUCKETS;
use pegainfer_kernels::ops::K3_DEEPGEMM_SM100_GROUPS;
use pegainfer_kernels::ops::K3_MAX_BATCH;
use pegainfer_kernels::ops::K3_MAX_CHUNK;
use pegainfer_kernels::ops::k3_batch_bucket;
use pegainfer_kernels::ops::k3_chunk_bucket;
use pegainfer_kernels::tensor::DeviceContext;

use self::buffers::K3MegaGeometry;
use self::buffers::K3MegaScratch;
use self::buffers::K3Scratch;
use self::buffers::K3StatePool;
use self::ep::K3EpRendezvous;
use self::ep::K3EpRuntime;
use self::ep::ep_fatal;
use self::step::K3StepShape;
use self::step::k3_decode_step;
use self::step::k3_prefill_boundary_sample;
use self::step::k3_prefill_chunk_step;
use crate::config::K3_DENSE_LAYERS;
use crate::config::K3_LAYERS;
use crate::config::K3MoeTopo;
use crate::config::probe_config_json;
use crate::model::K3ExpertBankForm;
use crate::model::K3RankModel;
use crate::scheduler::DecodeSlot;
use crate::scheduler::SlotId;
use crate::scheduler::StepExecutor;
use crate::weights::K3RankGpuContext;
use crate::weights::K3WeightManifest;
use crate::weights::load_rank_weights_to_gpu;

/// Rows the masked expert layout reserves per expert. The router picks
/// distinct experts per token, so one bucket can contribute at most one row per
/// expert and the largest bucket exactly fills this. Masked chain only — the
/// fused kernel has no such layout.
const K3_MASKED_CAP: usize = K3_MAX_BATCH;

/// Layer count override for bring-up, honoured by [`K3Executor::load`].
/// Unset (or `0`) builds the whole model.
const K3_LAYERS_ENV: &str = "PEGAINFER_K3_LAYERS";
/// Set to `0` to run every step eagerly instead of through a captured graph.
const K3_CUDA_GRAPH_ENV: &str = "PEGAINFER_K3_CUDA_GRAPH";
/// Concurrent slots per rank; rounded up to a compiled bucket, capped at the
/// widest one.
const K3_MAX_BATCH_ENV: &str = "PEGAINFER_K3_MAX_BATCH";
/// Context ceiling per slot (tokens); the paged pool is sized from it when
/// `kv_pages` is not set explicitly.
const K3_MAX_CTX_ENV: &str = "PEGAINFER_K3_MAX_CTX";
/// Default per-slot context ceiling. Free to raise: the cost is pool pages
/// (27.6 KB per token across the 24 MLA layers), not compiled kernels.
const K3_DEFAULT_MAX_CTX: usize = 4096;
/// The only SM count the fused MegaMoE kernel is AOT-instantiated for. Its
/// grid sync spans the whole grid, so the launch geometry is baked in.
const K3_MEGA_SMS: usize = 152;
/// Expert-parallel widths the fused MegaMoE kernel is AOT-instantiated for.
/// The rank count is a template parameter (it sets the ring capacities and the
/// experts-per-rank divisor), so this is not a runtime dimension.
const K3_MEGA_EP_SIZES: [usize; 2] = [1, 4];

/// Slots per rank an expert-parallel launch takes when nothing says otherwise.
///
/// The fused kernel's protocol maximum is 4224 rows per rank (sized for
/// chunked prefill), so the compiled
/// bucket ceiling (128) is the target once the backbone goes FP8. Today the
/// binding constraint is the KDA state slab: ~929 MB per slot (f32 recurrent
/// x2 parity + conv windows across 69 layers), so 64 slots cost ~58 GiB —
/// what fits next to the 224-expert rank's weights with room left for the
/// paged MLA pool. An explicit `PEGAINFER_K3_MAX_BATCH` still wins.
const K3_EP_DEFAULT_MAX_BATCH: usize = 64;

/// What a launch decides about an executor before its weights are read.
#[derive(Clone, Copy, Debug)]
pub struct K3ExecutorConfig {
    /// Concurrent slots, i.e. the row capacity of every state slab. Rounded up
    /// to a compiled bucket.
    pub max_batch: usize,
    /// Context ceiling per slot, in tokens. A runtime number: the paged
    /// attention kernel walks block tables, so nothing is compiled per
    /// capacity.
    pub max_ctx: usize,
    /// Pages in the MLA latent KV pool (64 tokens per page, all MLA layers'
    /// slices inside one page). `0` derives full coverage — every slot can
    /// reach `max_ctx` — so allocation can only fail when this is set lower
    /// (oversubscription is the caller's explicit choice).
    pub kv_pages: usize,
    /// Layers to build; `K3_LAYERS` for the whole model.
    pub num_layers: usize,
    /// Prefill chunk cap in tokens. `0` derives the widest the transport
    /// carries: the MegaMoE protocol maximum (4224, clamped to `max_ctx`)
    /// under the fused kernel, `max_batch` under the masked chain (whose
    /// layout reserves at most [`K3_MASKED_CAP`] rows per expert).
    pub chunk_tokens: usize,
    /// Capture and replay the step, rather than launching it eagerly.
    pub cuda_graph: bool,
    /// Which kernel runs the routed experts. Production is always
    /// [`K3MoeTransport::MEGA`]; see that type for why the alternative exists
    /// and why it is not selectable from a serving configuration.
    pub moe_transport: K3MoeTransport,
}

/// Which kernel runs the routed experts.
///
/// There is one production value, [`K3MoeTransport::MEGA`]. The masked
/// grouped-GEMM chain it replaced survives as the *numerics anchor* — it is
/// what `k3_moe_chain_gate` checks against an f32 reference, and what
/// `golden_decode` A/Bs the fused kernel against — and the two are deliberately
/// not bit-equivalent: MegaMoE applies the routing weights before the down
/// projection and mid-quantizes per 32 elements rather than per 128.
///
/// It is a newtype rather than a plain enum, and the chain is reachable only
/// through a `_for_tests` constructor, so nothing in a deployed configuration —
/// least of all a stray environment variable — can flip which arithmetic serves
/// a request. The chain is also single-rank only: expert parallelism has
/// exactly one transport.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct K3MoeTransport(MoeTransport);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MoeTransport {
    Mega,
    MaskedChain,
}

impl K3MoeTransport {
    /// DeepGEMM's fused MegaMoE(situ) kernel: one launch per MoE layer covering
    /// dispatch, both FP8xFP4 GEMMs, the activation, the mid-quantization and
    /// the combine — across the expert-parallel world when there is one.
    pub const MEGA: Self = Self(MoeTransport::Mega);

    /// The masked grouped-GEMM chain, for gates that need the anchor. Not a
    /// serving configuration: single rank only, and slower.
    #[doc(hidden)]
    #[must_use]
    pub const fn masked_chain_for_tests() -> Self {
        Self(MoeTransport::MaskedChain)
    }

    pub(crate) const fn is_mega(self) -> bool {
        matches!(self.0, MoeTransport::Mega)
    }
}

impl Default for K3ExecutorConfig {
    fn default() -> Self {
        Self {
            max_batch: K3_MAX_BATCH,
            max_ctx: K3_DEFAULT_MAX_CTX,
            kv_pages: 0,
            num_layers: K3_LAYERS,
            chunk_tokens: 0,
            cuda_graph: true,
            moe_transport: K3MoeTransport::MEGA,
        }
    }
}

impl K3ExecutorConfig {
    /// Apply the bring-up environment overrides.
    #[must_use]
    pub fn from_env(mut self) -> Self {
        if let Ok(raw) = std::env::var(K3_LAYERS_ENV)
            && let Ok(layers) = raw.parse::<usize>()
            && (1..=K3_LAYERS).contains(&layers)
        {
            self.num_layers = layers;
        }
        if std::env::var(K3_CUDA_GRAPH_ENV).as_deref() == Ok("0") {
            self.cuda_graph = false;
        }
        if let Ok(raw) = std::env::var(K3_MAX_BATCH_ENV)
            && let Ok(slots) = raw.parse::<usize>()
            && (1..=K3_MAX_BATCH).contains(&slots)
        {
            self.max_batch = slots;
        }
        if let Ok(raw) = std::env::var(K3_MAX_CTX_ENV)
            && let Ok(tokens) = raw.parse::<usize>()
            && (1..=crate::config::K3_MAX_CONTEXT).contains(&tokens)
        {
            self.max_ctx = tokens;
        }
        self
    }

    /// Apply what an EP width decides on its own: a narrower default slot count
    /// (an explicit `PEGAINFER_K3_MAX_BATCH` still wins) and, for now, no graph
    /// capture — capture is proven on the single-rank fused path but a captured
    /// cross-rank launch has not been replayed here.
    #[must_use]
    pub fn for_ep(mut self, ep_size: usize) -> Self {
        if ep_size > 1 {
            if std::env::var(K3_MAX_BATCH_ENV).is_err() {
                self.max_batch = K3_EP_DEFAULT_MAX_BATCH;
            }
            self.cuda_graph = false;
        }
        self
    }
}

pub struct K3Executor {
    gpu: K3RankGpuContext,
    ctx: DeviceContext,
    model: K3RankModel,
    /// Slot state, one row per slot.
    decode_state: K3StatePool,
    /// The one-row pool a prompt is consumed on.
    prefill_state: K3StatePool,
    scratch: K3Scratch,
    max_batch: usize,
    max_ctx: usize,
    /// Prefill chunk cap in tokens (see [`K3ExecutorConfig::chunk_tokens`]).
    chunk_tokens: usize,
    groups: usize,
    num_sms: usize,
    /// Which half of the ping-pong state slabs the next decode step reads.
    parity: usize,
    cuda_graph: bool,
    /// Routed experts run through the fused MegaMoE kernel.
    mega: bool,
    /// Mega launches this step must make: one per MoE layer. Zero single-rank,
    /// where there are no peers to fall out of step with.
    mega_launches_per_step: usize,
    /// One graph per (bucket, parity). Prefill chunks run eagerly.
    decode_graphs: Vec<CudaGraphState>,
    /// Step inputs, staged on the host and copied in before every step.
    token_host: Vec<u32>,
    context_len_host: Vec<i32>,
    kv_row_host: Vec<i32>,
    sampled_host: Vec<i32>,
    thread_bound: bool,
    /// Present exactly when `ep_size > 1`: this rank's slab handshake with its
    /// peers. It issues nothing per step.
    ep: Option<K3EpRuntime>,
}

// SAFETY: the executor owns one rank's context, stream and device buffers, and
// binds them to whichever thread steps it (see `bind_thread`). Exactly one
// scheduler thread ever steps one executor.
unsafe impl Send for K3Executor {}

impl K3Executor {
    /// Load one rank of a K3 checkpoint and build its decode executor.
    ///
    /// Single-rank only: an expert-parallel group's ranks have to share one
    /// rendezvous, so they go through [`Self::load_ep`].
    pub fn load(
        model_path: &Path,
        device_ordinal: usize,
        rank: usize,
        ep_size: usize,
        config: K3ExecutorConfig,
    ) -> Result<Self> {
        ensure!(
            ep_size == 1,
            "K3Executor::load builds a single-rank executor; an ep_size of {ep_size} needs \
             K3Executor::load_ep with the group's shared rendezvous"
        );
        Self::load_inner(model_path, device_ordinal, rank, ep_size, config, None)
    }

    /// Load one rank of an expert-parallel group.
    ///
    /// `rendezvous` is shared by every rank of the group and carries its width.
    /// This rank publishes its symmetric slab here, but reads the world's table
    /// back lazily, on the worker thread that steps it — so a rank that fails to
    /// load cannot leave its peers blocked waiting for a slab that will never
    /// arrive.
    pub fn load_ep(
        model_path: &Path,
        device_ordinal: usize,
        rank: usize,
        config: K3ExecutorConfig,
        rendezvous: Arc<K3EpRendezvous>,
    ) -> Result<Self> {
        let ep_size = rendezvous.ranks();
        ensure!(
            ep_size > 1,
            "K3Executor::load_ep needs an EP group wider than one rank; use K3Executor::load"
        );
        Self::load_inner(
            model_path,
            device_ordinal,
            rank,
            ep_size,
            config,
            Some(rendezvous),
        )
    }

    fn load_inner(
        model_path: &Path,
        device_ordinal: usize,
        rank: usize,
        ep_size: usize,
        config: K3ExecutorConfig,
        rendezvous: Option<Arc<K3EpRendezvous>>,
    ) -> Result<Self> {
        probe_config_json(&read_config(model_path)?)
            .with_context(|| format!("validate the K3 config at {}", model_path.display()))?;
        let manifest = K3WeightManifest::from_model_dir(model_path)?;
        let topo = K3MoeTopo::new(manifest.routed_experts(), ep_size)?;
        ensure!(
            config.moe_transport.is_mega()
                || K3_DEEPGEMM_SM100_GROUPS.contains(&topo.local_experts()),
            "K3 rank holds {} experts, but the masked grouped GEMM is instantiated for \
             {K3_DEEPGEMM_SM100_GROUPS:?}",
            topo.local_experts()
        );
        // Plan only the layers this executor builds: a truncated build frees
        // the rest, but not before they are resident, and at low expert
        // parallelism a whole rank does not fit on one device.
        let bundle = manifest.rank_load_bundle_for_layers(rank, topo, config.num_layers)?;
        let gpu = K3RankGpuContext::new(device_ordinal)?;
        let ctx = gpu.device_context()?;
        let loaded = load_rank_weights_to_gpu(&gpu, model_path, &bundle, false)?;
        let form = if config.moe_transport.is_mega() {
            K3ExpertBankForm::Mega
        } else {
            K3ExpertBankForm::MaskedChain
        };
        let model = K3RankModel::build(&ctx, loaded.weights, topo, rank, config.num_layers, form)?;
        Self::new(gpu, ctx, model, config, rendezvous)
    }

    /// Wrap an already-built rank model.
    pub(crate) fn new(
        gpu: K3RankGpuContext,
        ctx: DeviceContext,
        model: K3RankModel,
        config: K3ExecutorConfig,
        rendezvous: Option<Arc<K3EpRendezvous>>,
    ) -> Result<Self> {
        let max_batch = k3_batch_bucket(config.max_batch)?;
        ensure!(
            (1..=crate::config::K3_MAX_CONTEXT).contains(&config.max_ctx),
            "K3 max_ctx {} is outside 1..={}",
            config.max_ctx,
            crate::config::K3_MAX_CONTEXT
        );
        let num_sms = ctx
            .ctx
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|error| anyhow::anyhow!("query the K3 device SM count: {error}"))?
            as usize;
        ensure!(
            matches!(num_sms, 148 | 152),
            "K3 routed experts need a B200 (148 SM) or GB300 (152 SM) device; this one has {num_sms}"
        );
        let groups = model.topo.local_experts();
        let routed_experts = model.topo.routed_experts().count();
        let num_layers = model.layers.len();
        let blocks = model.blocks;
        let ep_size = rendezvous.as_ref().map_or(1, |r| r.ranks());
        let mega = config.moe_transport.is_mega();
        let mut cuda_graph = config.cuda_graph;
        if ep_size > 1 && cuda_graph {
            // Capture is proven on the single-rank fused path, but a captured
            // cross-rank launch has never been replayed here and the kernel's
            // device barriers pair the world inside it. Eager until that is its
            // own piece of work.
            info!(
                "K3 rank {} runs eagerly: CUDA-graph capture is off under expert parallelism \
                 (ep_size={ep_size})",
                model.rank
            );
            cuda_graph = false;
        }

        let slot_pages = config.max_ctx.div_ceil(paged_kv::K3_KV_PAGE_TOKENS);
        let kv_pages = if config.kv_pages == 0 {
            max_batch * slot_pages
        } else {
            config.kv_pages
        };
        let decode_state = K3StatePool::new(
            &ctx,
            max_batch,
            max_batch,
            config.max_ctx,
            num_layers,
            blocks,
            kv_pages,
        )?;
        // The prefill chunk cap: the MegaMoE protocol maximum under the fused
        // kernel (clamped to the context — a chunk can never exceed the
        // prompt), the decode row capacity under the masked chain, whose
        // layout caps rows per expert.
        let chunk_tokens = if config.chunk_tokens > 0 {
            config.chunk_tokens
        } else if mega {
            K3_MAX_CHUNK.min(config.max_ctx)
        } else {
            max_batch
        };
        ensure!(
            mega || chunk_tokens <= K3_MASKED_CAP,
            "K3 masked chain caps prefill chunks at {K3_MASKED_CAP} tokens, got {chunk_tokens}"
        );
        let chunk_bucket = k3_chunk_bucket(chunk_tokens)?;
        // Every per-layer scratch buffer spans the widest bucket any step
        // runs; the epilogue buffers stay at the decode rows (a prefill chunk
        // samples its boundary token through a one-row pass instead).
        let scratch_rows = max_batch.max(chunk_bucket);
        // The prefill pool holds ONE sequence (one row of KDA state, one page
        // chain — full coverage is one slot's pages) but steps it a chunk at
        // a time, so its snapshot slab and block table span the chunk bucket.
        let prefill_state = K3StatePool::new(
            &ctx,
            1,
            chunk_bucket,
            config.max_ctx,
            num_layers,
            blocks,
            slot_pages,
        )?;
        if mega {
            ensure!(
                K3_MEGA_EP_SIZES.contains(&ep_size),
                "K3 MegaMoE is AOT-instantiated for ep_size {K3_MEGA_EP_SIZES:?} only, not \
                 {ep_size}"
            );
            // The fused kernel's grid sync spans exactly its instantiation's SM
            // count, so a mismatched launch grid would hang rather than
            // misbehave. Fail here, before a whole rank's experts are rebuilt
            // into the mega layout.
            ensure!(
                num_sms == K3_MEGA_SMS,
                "K3 MegaMoE is AOT-instantiated for the GB300 {K3_MEGA_SMS}-SM grid only; \
                 this device has {num_sms} SMs"
            );
        } else {
            // The masked chain is the anchor, not a transport: nothing shards it.
            ensure!(
                ep_size == 1,
                "the K3 masked chain is single-rank only; expert parallelism runs MegaMoE"
            );
            // Worst case one expert claims every token, so the batch has to fit
            // the masked layout's per-expert rows. The route metadata's device
            // trap is the backstop; this turns it into a launch-time message.
            ensure!(
                max_batch <= K3_MASKED_CAP,
                "K3 masked chain got {max_batch} slots but the layout reserves {K3_MASKED_CAP} \
                 rows per expert — lower {K3_MAX_BATCH_ENV}"
            );
        }
        let mut scratch = K3Scratch::new(
            &ctx,
            scratch_rows,
            max_batch,
            config.max_ctx,
            routed_experts,
            groups,
            K3_MASKED_CAP,
            mega.then_some(K3MegaGeometry {
                num_sms,
                num_ranks: ep_size,
                rank_idx: model.rank,
            }),
        )?;
        // Every allocation this rank will ever hand a peer has to be live and
        // zeroed before its base pointer is published, hence the sync first.
        gpu.sync()?;
        let ep = rendezvous
            .map(|rendezvous| {
                let mega = scratch
                    .mega
                    .as_mut()
                    .context("K3 EP rank built without its symmetric buffer")?;
                K3EpRuntime::new(rendezvous, model.rank, mega.base(), gpu.device_ordinal())
            })
            .transpose()?;

        let bucket_count = K3_BATCH_BUCKETS.iter().filter(|b| **b <= max_batch).count();
        info!(
            "K3 rank {} executor ready: slots={max_batch}, max_ctx={}, layers={num_layers}, \
             blocks={blocks}, local_experts={groups}, sms={num_sms}, cuda_graph={cuda_graph}, \
             ep_size={ep_size}, moe={}",
            model.rank,
            config.max_ctx,
            if mega {
                "mega"
            } else {
                "masked-chain (anchor)"
            },
        );
        Ok(Self {
            gpu,
            ctx,
            model,
            decode_state,
            prefill_state,
            scratch,
            max_batch,
            max_ctx: config.max_ctx,
            groups,
            num_sms,
            parity: 0,
            cuda_graph,
            mega,
            mega_launches_per_step: if mega && ep_size > 1 {
                num_layers.saturating_sub(K3_DENSE_LAYERS)
            } else {
                0
            },
            chunk_tokens,
            decode_graphs: (0..2 * bucket_count)
                .map(|_| CudaGraphState::new())
                .collect(),
            token_host: vec![0; scratch_rows],
            context_len_host: vec![1; scratch_rows],
            kv_row_host: vec![-1; scratch_rows],
            sampled_host: vec![0; max_batch],
            thread_bound: false,
            ep,
        })
    }

    fn bind_thread(&mut self) -> Result<()> {
        if !self.thread_bound {
            self.gpu.set_current()?;
            // The cuBLAS handle is thread-local per device.
            unsafe {
                pegainfer_kernels::ffi::cublas_init();
            }
            self.thread_bound = true;
        }
        Ok(())
    }

    /// Bind the thread and, the first time through, resolve the group's slab
    /// table on it.
    ///
    /// Every path that can reach a step goes through here, and nothing else
    /// does: the rendezvous blocks until the last peer has published, so a rank
    /// must not enter it from a housekeeping call its peers are not making.
    fn enter_step(&mut self) -> Result<()> {
        self.bind_thread()?;
        if let Some(ep) = self.ep.as_mut()
            && let Some(ptrs) = ep.ensure_ready()?
        {
            self.scratch
                .mega
                .as_mut()
                .context("K3 EP step ran without its symmetric buffer")?
                .set_peers(ptrs)?;
        }
        Ok(())
    }

    /// Is this executor one rank of an expert-parallel group?
    fn is_expert_parallel(&self) -> bool {
        self.ep.is_some()
    }

    /// `live_rows` is how many leading rows of the bucket this step actually
    /// owns — zero on an expert-parallel padding step. Only the MegaMoE path at
    /// `ep_size > 1` reads it: everything else runs the whole bucket and throws
    /// the padding rows away, but a mega rank's rows travel to its peers, so it
    /// sends only the rows it has.
    fn shape(&self, bucket: usize, parity: usize, live_rows: usize) -> K3StepShape {
        K3StepShape {
            bucket,
            live_rows,
            parity,
            chunk_start: 0,
            groups: self.groups,
            masked_cap: K3_MASKED_CAP,
            num_sms: self.num_sms,
            mega: self.mega,
        }
    }

    /// Copy the staged step inputs to the device.
    fn feed(&mut self) -> Result<()> {
        let stream = &self.ctx.stream;
        stream
            .memcpy_htod(&self.token_host, &mut self.scratch.token_ids)
            .map_err(|error| anyhow::anyhow!("K3 token feed failed: {error}"))?;
        stream
            .memcpy_htod(&self.context_len_host, &mut self.scratch.context_len)
            .map_err(|error| anyhow::anyhow!("K3 context-length feed failed: {error}"))?;
        stream
            .memcpy_htod(&self.kv_row_host, &mut self.scratch.kv_row)
            .map_err(|error| anyhow::anyhow!("K3 KV-row feed failed: {error}"))
    }

    /// Run one decode step, through its graph when graphs are on.
    fn run_step(&mut self, bucket: usize, parity: usize, live_rows: usize) -> Result<()> {
        let shape = self.shape(bucket, parity, live_rows);
        let bucket_index = K3_BATCH_BUCKETS
            .iter()
            .position(|candidate| *candidate == bucket)
            .expect("bucket comes from k3_batch_bucket");
        let graph_index = 2 * bucket_index + parity;
        let pool = &mut self.decode_state;
        let ctx = &self.ctx;
        let model = &self.model;
        let scratch = &mut self.scratch;
        // The block table rides outside capture with the rest of the step
        // inputs; the captured kernels read the device table by pointer.
        pool.kv.sync_table(ctx)?;
        if !self.cuda_graph {
            let launches = self.mega_launches_per_step;
            if let Some(mega) = scratch.mega.as_mut() {
                mega.begin_step(launches);
            }
            k3_decode_step(ctx, model, shape, pool, scratch)?;
            return scratch
                .mega
                .as_ref()
                .map_or(Ok(()), K3MegaScratch::end_step);
        }
        let mut graph = std::mem::take(&mut self.decode_graphs[graph_index]);
        // Capture is off above one rank, so a captured body is always a
        // single-rank step with nobody to fall out of phase with.
        let result = graph.run_or_capture(ctx, || k3_decode_step(ctx, model, shape, pool, scratch));
        self.decode_graphs[graph_index] = graph;
        result
    }

    /// Read this step's sampled ids back.
    fn sampled(&mut self, rows: usize) -> Result<&[i32]> {
        self.ctx
            .stream
            .memcpy_dtoh(&self.scratch.argmax_indices, &mut self.sampled_host)
            .map_err(|error| anyhow::anyhow!("K3 sampled-token readback failed: {error}"))?;
        self.gpu.sync()?;
        Ok(&self.sampled_host[..rows])
    }

    /// Greedy continuation of one slot, straight through the decode path.
    ///
    /// Feeds `prompt` one token per step, then feeds each sampled token back
    /// for `steps` more, and returns the argmax after every step — the protocol
    /// the golden fixture records. The slot's state is reset first; the bucket
    /// is whatever `slot` forces, so a low slot exercises a narrow bucket and a
    /// high one exercises a wide bucket with a single owned row.
    pub fn greedy_replay(
        &mut self,
        slot: SlotId,
        prompt: &[u32],
        steps: usize,
    ) -> Result<Vec<u32>> {
        ensure!(slot < self.max_batch, "K3 slot {slot} is out of range");
        ensure!(!prompt.is_empty(), "K3 replay needs at least one token");
        self.bind_thread()?;
        self.decode_state.reset_row(&self.ctx, slot)?;
        let mut sampled = Vec::with_capacity(prompt.len() + steps);
        for index in 0..prompt.len() + steps {
            let last_token = if index < prompt.len() {
                prompt[index]
            } else {
                sampled[index - 1]
            };
            let step = self.decode(&[DecodeSlot { slot, last_token }])?;
            sampled.push(step[0]);
        }
        Ok(sampled)
    }

    /// Replay a fixed token sequence on `slot` and return the argmax after
    /// every step.
    ///
    /// Unlike [`Self::greedy_replay`] the feed is given rather than fed back,
    /// so a step's inputs do not depend on the previous step's sample. That
    /// separates the two ways a replay can leave a reference trajectory: an
    /// argmax that genuinely differs, and an argmax that agrees until one
    /// near-tie sends the two continuations apart.
    pub fn forced_replay(&mut self, slot: SlotId, feed: &[u32]) -> Result<Vec<u32>> {
        ensure!(slot < self.max_batch, "K3 slot {slot} is out of range");
        self.bind_thread()?;
        self.decode_state.reset_row(&self.ctx, slot)?;
        let mut sampled = Vec::with_capacity(feed.len());
        for last_token in feed.iter().copied() {
            sampled.push(self.decode(&[DecodeSlot { slot, last_token }])?[0]);
        }
        Ok(sampled)
    }

    /// Test hook: reverse the decode pool's free page list, so the next
    /// sequence's pages land at different physical ids in a different order.
    /// The paged cache's core gate (`tests/paged_kv.rs`) is that no page
    /// permutation can move a single logit bit.
    #[doc(hidden)]
    pub fn scramble_kv_pages(&mut self) {
        self.decode_state.kv.reverse_free_list();
    }

    /// Bring-up diagnostics: the logit row the most recent step left for
    /// `row`, widened to f32. Costs a device round trip; not a serving path.
    pub fn last_logits(&mut self, row: usize) -> Result<Vec<f32>> {
        ensure!(row < self.max_batch, "K3 row {row} is out of range");
        let width = crate::config::K3_VOCAB;
        let window = self.scratch.logits.slice(row * width..(row + 1) * width);
        let logits = self
            .ctx
            .stream
            .clone_dtoh(&window)
            .map_err(|error| anyhow::anyhow!("K3 logit readback failed: {error}"))?;
        self.gpu.sync()?;
        Ok(logits.into_iter().map(f32::from).collect())
    }
}

fn read_config(model_path: &Path) -> Result<serde_json::Value> {
    let raw = std::fs::read_to_string(model_path.join("config.json"))
        .with_context(|| format!("read {}/config.json", model_path.display()))?;
    serde_json::from_str(&raw).context("parse the K3 config.json")
}

impl K3Executor {
    /// The prefill chunk cap this executor runs, in tokens.
    pub fn chunk_tokens(&self) -> usize {
        self.chunk_tokens
    }
}

impl StepExecutor for K3Executor {
    fn max_batch(&self) -> usize {
        self.max_batch
    }

    fn max_context_tokens(&self) -> usize {
        self.max_ctx
    }

    fn prefill(&mut self, slot: SlotId, prompt: &[u32], params: &SamplingParams) -> Result<u32> {
        let rank = self.model.rank;
        match self.prefill_inner(slot, prompt, params) {
            Err(error) if self.is_expert_parallel() => ep_fatal(rank, "prefill", &error),
            other => other,
        }
    }

    fn decode(&mut self, batch: &[DecodeSlot]) -> Result<Vec<u32>> {
        let rank = self.model.rank;
        match self.decode_inner(batch) {
            Err(error) if self.is_expert_parallel() => ep_fatal(rank, "decode", &error),
            other => other,
        }
    }

    fn release(&mut self, slot: SlotId) {
        if self.bind_thread().is_ok()
            && let Err(error) = self.decode_state.reset_row(&self.ctx, slot)
        {
            log::warn!("K3 slot {slot} release did not clear its state: {error:#}");
        }
    }
}

impl K3Executor {
    fn prefill_inner(
        &mut self,
        slot: SlotId,
        prompt: &[u32],
        _params: &SamplingParams,
    ) -> Result<u32> {
        ensure!(slot < self.max_batch, "K3 slot {slot} is out of range");
        ensure!(!prompt.is_empty(), "K3 prefill needs at least one token");
        ensure!(
            prompt.len() <= self.max_ctx,
            "K3 prompt of {} tokens exceeds the {} slot context",
            prompt.len(),
            self.max_ctx
        );
        self.enter_step()?;
        self.prefill_state.reset_row(&self.ctx, 0)?;

        // Walk the prompt in chunks of up to `max_batch` tokens; each chunk is
        // one batched step whose rows are the chunk's consecutive tokens.
        // Prefill always runs eagerly — the chunk's KDA loop makes its launch
        // count depend on the token count, so there is no fixed body to
        // capture per bucket.
        let mut parity = 0usize;
        let mut consumed = 0usize;
        let mut last_tokens = 1usize;
        while consumed < prompt.len() {
            let tokens = self.chunk_tokens.min(prompt.len() - consumed);
            let bucket = k3_chunk_bucket(tokens)?;
            for (row, token) in prompt[consumed..consumed + tokens].iter().enumerate() {
                let position = consumed + row;
                self.token_host[row] = *token;
                self.context_len_host[row] = i32::try_from(position + 1)?;
                self.prefill_state
                    .kv
                    .ensure_mapped(&self.ctx, 0, position)?;
                self.kv_row_host[row] = self.prefill_state.kv.write_index(0, position)?;
            }
            for row in tokens..bucket {
                self.token_host[row] = 0;
                self.context_len_host[row] = 1;
                self.kv_row_host[row] = -1;
            }
            // Every row of the bucket reads the one sequence's pages; a padded
            // row sees context length 1 and its result is discarded.
            self.prefill_state.kv.mirror_row_table(0, bucket)?;
            self.feed()?;
            self.prefill_state.kv.sync_table(&self.ctx)?;
            let mut shape = self.shape(bucket, parity, tokens);
            shape.chunk_start = consumed;
            let launches = self.mega_launches_per_step;
            if let Some(mega) = self.scratch.mega.as_mut() {
                mega.begin_step(launches);
            }
            k3_prefill_chunk_step(
                &self.ctx,
                &self.model,
                shape,
                &mut self.prefill_state,
                &mut self.scratch,
            )?;
            if let Some(mega) = self.scratch.mega.as_ref() {
                mega.end_step()?;
            }
            // Under the chunkwise KDA kernel parity is a per-chunk double
            // buffer: every chunk reads one slab and lands in the other.
            parity ^= 1;
            consumed += tokens;
            last_tokens = tokens;
            self.prefill_state.positions[0] = consumed;
        }
        // The chunk steps skipped the batched epilogue; sample the boundary
        // token once, at one row, over the final chunk's last live token.
        // The snapshot collapse it needs is the same one `adopt_row` wants —
        // the final token's snapshots move to row 0, the handover row.
        self.prefill_state
            .collapse_snapshots(&self.ctx, last_tokens - 1)?;
        k3_prefill_boundary_sample(
            &self.ctx,
            &self.model,
            last_tokens - 1,
            &self.prefill_state.blocks,
            &mut self.scratch,
        )?;
        let sampled = self.sampled(1)?[0] as u32;
        self.decode_state.reset_row(&self.ctx, slot)?;
        let target_parity = self.parity;
        self.decode_state.adopt_row(
            &self.ctx,
            &self.prefill_state,
            0,
            parity,
            slot,
            target_parity,
        )?;
        self.gpu.sync()?;
        Ok(sampled)
    }

    /// One decode step.
    ///
    /// An empty batch is an early return single-rank — there is nothing to
    /// compute and nobody to answer to. Under expert parallelism it is a
    /// *padding step*: this rank owes its peers the same per-layer sequence
    /// whether or not it has work, so it runs the narrowest bucket with every
    /// row padding and returns no tokens. That is the whole free-running story
    /// — a rank never negotiates, it just keeps walking the chain.
    ///
    /// Under MegaMoE the same step still launches every MoE layer, at zero
    /// local tokens: the kernel serves this rank's experts for its peers'
    /// tokens and joins every device barrier regardless of what it holds
    /// itself.
    fn decode_inner(&mut self, batch: &[DecodeSlot]) -> Result<Vec<u32>> {
        if batch.is_empty() && !self.is_expert_parallel() {
            return Ok(Vec::new());
        }
        self.enter_step()?;
        let rows = batch.iter().map(|entry| entry.slot + 1).max().unwrap_or(0);
        ensure!(
            rows <= self.max_batch,
            "K3 decode reached slot {} above the {} configured slots",
            rows.saturating_sub(1),
            self.max_batch
        );
        let bucket = k3_batch_bucket(rows.max(1))?;

        for row in 0..self.max_batch {
            self.token_host[row] = 0;
            self.context_len_host[row] = 1;
            self.kv_row_host[row] = -1;
        }
        for entry in batch {
            let position = self.decode_state.positions[entry.slot];
            ensure!(
                position < self.max_ctx,
                "K3 slot {} reached its {} token context",
                entry.slot,
                self.max_ctx
            );
            self.token_host[entry.slot] = entry.last_token;
            self.context_len_host[entry.slot] = i32::try_from(position + 1)?;
            self.decode_state
                .kv
                .ensure_mapped(&self.ctx, entry.slot, position)?;
            self.kv_row_host[entry.slot] =
                self.decode_state.kv.write_index(entry.slot, position)?;
        }

        self.feed()?;
        let parity = self.parity;
        self.run_step(bucket, parity, rows)?;
        self.parity ^= 1;
        for entry in batch {
            self.decode_state.positions[entry.slot] += 1;
        }

        let sampled = self.sampled(self.max_batch)?;
        Ok(batch
            .iter()
            .map(|entry| sampled[entry.slot] as u32)
            .collect())
    }
}
