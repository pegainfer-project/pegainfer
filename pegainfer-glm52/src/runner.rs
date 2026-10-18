use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use crossbeam_channel::Receiver;
use crossbeam_channel::Sender;
use crossbeam_channel::bounded;
use crossbeam_channel::unbounded;
use pegainfer_core::cuda_graph::CudaGraphDumpSummary;
use pegainfer_kv_store::ArenaSpec;

use crate::dspark::GLM52_DSPARK_DRAFTS;
use crate::dspark::Glm52DsparkModel;
use crate::dspark::Glm52DsparkScratch;
use crate::dspark::Glm52DsparkSlotState;
use crate::model::GLM52_MAX_BATCH_PER_RANK;
use crate::model::GLM52_MAX_STEP_ROWS;
use crate::model::Glm52RankModel;
use crate::model::Glm52RankModelFixed;
use crate::model::Glm52StepKv;
use crate::model::Glm52StepShape;
use crate::moe_ep::Glm52MoeEpRankState;
use crate::moe_ep::Glm52MoeEpState;
use crate::moe_tp::Glm52MoeTpRank;
use crate::moe_tp::Glm52MoeTpSliceBank;
use crate::moe_tp::Glm52MoeTpState;
use crate::moe_tp::Glm52TpExchange;
use crate::moe_tp::load_tp_slice_layer;
use crate::prefill_tp::Glm52PrefillOutput;
use crate::weights::Glm52RankGpuContext;
use crate::weights::Glm52RankGpuWeights;
use crate::weights::Glm52RankLoadBundle;
use crate::weights::Glm52WeightManifest;
use crate::weights::load_rank_weights_to_gpu;

/// Global rank + local CUDA device of one worker. Rank bounds are enforced
/// where placements are built, against the launch topology's real width —
/// there is no per-width invariant here.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Glm52RankPlacement {
    pub(crate) rank: usize,
    pub(crate) device_ordinal: usize,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Glm52RankWeightLoadReport {
    pub(crate) rank: usize,
    pub(crate) loaded_tensor_count: usize,
    pub(crate) loaded_total_bytes: usize,
    resident_raw_bytes: usize,
    pub(crate) loaded_to_gpu: bool,
    /// This rank's device free VRAM right after the weights landed — the
    /// launch-time context-cap probe takes the fleet minimum.
    pub(crate) free_vram_bytes: usize,
}

/// The engine's launch-ahead directives for one step. Both directions are
/// rank-local: an engine leases its OWN next step speculatively and always
/// consumes its own speculation (the slot set is frozen while a lease is
/// outstanding — see `plan::lease_flags`), so the consume/lease pairing
/// needs no cross-rank agreement.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Glm52StepFlags {
    /// This step IS the speculative replay this engine enqueued last step.
    pub(crate) consume: bool,
    /// The next step is guaranteed to repeat this shape with each row
    /// advanced by its own argmax — the engine MUST enqueue that next
    /// replay launch-ahead (see `Glm52RankModel::decode_step`).
    pub(crate) lease: bool,
    /// Bypass CUDA graph capture and replay.
    pub(crate) eager: bool,
}

impl Glm52StepFlags {
    /// No speculation in either direction (warm pre-capture, tests).
    pub(crate) fn plain() -> Self {
        Self {
            consume: false,
            lease: false,
            eager: false,
        }
    }
}

/// One step row whose committed token is sampled instead of taking the fused
/// greedy argmax: a non-greedy request's decode span row (every row of a
/// verify span — anchor and draft prefix alike), or the last row of its
/// prompt-completing span. `step` is the request-local decode step the row's
/// token lands at — a seeded request's philox seed mixes it, so its tokens
/// replay independently of batch composition AND of how many rows rode each
/// speculative round (spec and plain produce the same seeded stream).
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Glm52RowSample {
    pub(crate) row: usize,
    pub(crate) params: pegainfer_sample::SamplingParams,
    pub(crate) step: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct Glm52PrefillBatch {
    pub(crate) token_ids: Vec<u32>,
    pub(crate) positions: Vec<u32>,
    pub(crate) request_indptr: Vec<u32>,
    pub(crate) block_indptr: Vec<u32>,
    pub(crate) block_ids: Vec<i32>,
    /// Scheduler slot owning each request range. Needed to address the
    /// per-slot native-MTP speculative scratch pages after the boundary.
    pub(crate) request_slots: Vec<usize>,
    pub(crate) padding_block: i32,
    pub(crate) slot_mapping: Vec<i64>,
    /// Shifted-token carry for MTP layer 78, one per request range. A prompt
    /// token continues a mid-prompt chunk; `None` means the target boundary
    /// output becomes the carry after sampling.
    pub(crate) mtp_next_tokens: Vec<Option<u32>>,
    pub(crate) output_rows: Vec<u32>,
    pub(crate) sampling: Vec<Glm52RowSample>,
    pub(crate) seed: u64,
}

impl Glm52PrefillBatch {
    pub(crate) fn validate(&self) -> Result<()> {
        let rows = self.token_ids.len();
        ensure!(
            rows > 0 && self.positions.len() == rows && self.slot_mapping.len() == rows,
            "GLM5.2 prefill batch row buffers disagree"
        );
        ensure!(
            self.request_indptr.first() == Some(&0)
                && self.request_indptr.last().copied() == Some(rows as u32)
                && self.request_indptr.windows(2).all(|w| w[0] < w[1]),
            "GLM5.2 prefill request row ranges are invalid"
        );
        let requests = self.request_indptr.len() - 1;
        ensure!(
            self.mtp_next_tokens.len() == requests && self.request_slots.len() == requests,
            "GLM5.2 prefill request metadata differs from {requests} request ranges"
        );
        ensure!(
            self.block_indptr.len() == requests + 1
                && self.block_indptr.first() == Some(&0)
                && self.block_indptr.last().copied() == Some(self.block_ids.len() as u32)
                && self.block_indptr.windows(2).all(|w| w[0] <= w[1]),
            "GLM5.2 prefill block ranges are invalid"
        );
        ensure!(
            self.padding_block >= 0,
            "GLM5.2 prefill padding block is invalid"
        );
        ensure!(
            self.output_rows.windows(2).all(|w| w[0] < w[1])
                && self.output_rows.iter().all(|&row| (row as usize) < rows),
            "GLM5.2 prefill output rows are invalid"
        );
        ensure!(
            self.sampling.windows(2).all(|w| w[0].row < w[1].row)
                && self
                    .sampling
                    .iter()
                    .all(|sample| sample.row < self.output_rows.len()),
            "GLM5.2 prefill sampling rows are invalid"
        );
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Glm52MtpAppend {
    /// Row in the target step's retained final-normalized hidden buffer.
    pub(crate) target_row: usize,
    pub(crate) slot: usize,
    /// Sequence token shifted one place to the left, matching vLLM's MTP
    /// first-pass input construction.
    pub(crate) input_token: u32,
    pub(crate) position: usize,
    /// Target BlockPool pages covering the committed sequence after this
    /// target step. Layer 78 must address these same physical page IDs.
    pub(crate) pages: Vec<i32>,
}

/// One rank's work in one native-MTP round. Native MTP is an EP collective
/// (layer 78 is a MoE layer), and its collective chain is FIXED: every rank
/// runs one context forward plus `GLM52_MTP_DRAFTS - 1` proposal forwards
/// every round — rank-local buckets, deterministic padding rows for work it
/// doesn't have — so the per-step collective count never depends on host
/// state (the free-running fixed-chain discipline,
/// `docs/models/glm52/free-running-dp.md` §4).
#[derive(Debug)]
pub(crate) struct Glm52MtpRound {
    /// Bucket of the target step this rank just ran — selects the retained
    /// final-normalized hidden buffer the appends read from.
    pub(crate) source_bucket: usize,
    pub(crate) context_bucket: usize,
    pub(crate) draft_bucket: usize,
    pub(crate) resets: Vec<usize>,
    pub(crate) appends: Vec<Glm52MtpAppend>,
    pub(crate) proposal_slots: Vec<usize>,
}

enum Glm52RankCommand {
    LoadWeights {
        model_path: PathBuf,
        moe_topo: crate::Glm52MoeTopo,
        weight_staging: bool,
        resp: Sender<Result<Glm52RankWeightLoadReport>>,
    },
    /// Non-collective, phase 1 of the two-phase build: adopt the resident
    /// weights into everything EXCEPT the pool-scaled KV slabs, then reply
    /// with this rank's measured free device VRAM — launch min-reduces the
    /// fleet's replies to size the KV pool from real headroom instead of a
    /// ledger estimate.
    BuildFixed {
        max_model_len: usize,
        moe_topo: crate::Glm52MoeTopo,
        drafter: crate::Glm52Drafter,
        prefill_chunk_size: Option<usize>,
        resp: Sender<Result<u64>>,
    },
    /// Non-collective, phase 2: allocate the pool-scaled slabs for the
    /// launch-decided block count and assemble the rank runtime. Every rank
    /// must report success BEFORE anyone enters SetupComm — a build failure
    /// on one rank must never strand the others in NCCL init. Replies with
    /// the rank's cache arena descriptors so launch can register them with
    /// the shared KV offload host.
    FinishKv {
        pool_blocks: usize,
        resp: Sender<Result<Vec<ArenaSpec>>>,
    },
    /// Collective: create the DeepEP context (barriers across ranks). Issued
    /// to every rank concurrently, only after all builds succeeded. Under the
    /// TP topology, also allocates the LL buffers and rendezvouses peer
    /// pointers through `tp_exchange`.
    SetupComm {
        unique_id: Box<[u8; 128]>,
        moe_topo: crate::Glm52MoeTopo,
        tp_exchange: Option<Arc<Glm52TpExchange>>,
        resp: Sender<Result<()>>,
    },
    /// One full-model step for one rank (75 MoE collectives inside): feed
    /// `inputs[row]` per forwarded row (a slot's span rows walk consecutive
    /// positions), reply with the next token per ROW (greedy argmax, or a
    /// sampling pass for the rows in `sampling`). The rank's engine sends
    /// this unconditionally every step — idle ranks enter with padding rows
    /// and their outputs are discarded; `shape.bucket` is rank-local (the
    /// MoE collectives take rank-local row counts under the conservative
    /// protocol-max bound).
    Step {
        inputs: Box<[(u32, usize); GLM52_MAX_STEP_ROWS]>,
        shape: Glm52StepShape,
        kv: Box<Glm52StepKv>,
        flags: Glm52StepFlags,
        /// Rows sampled instead of argmaxed (non-greedy requests; empty on
        /// the all-greedy fast path), and the step's philox seed for their
        /// unseeded members.
        sampling: Vec<Glm52RowSample>,
        seed: u64,
        resp: Sender<Result<[u32; GLM52_MAX_STEP_ROWS]>>,
    },
    PrefillChunk {
        batch: Glm52PrefillBatch,
        resp: Sender<Result<Glm52PrefillOutput>>,
    },
    /// Non-collective: load the DSpark draft model onto this rank. Issued to
    /// every rank after FinishKv (the draft reuses the target's
    /// embed/lm_head at forward time).
    LoadDspark {
        path: PathBuf,
        resp: Sender<Result<()>>,
    },
    /// Non-collective: report this rank's current free device VRAM (the
    /// post-build headroom check).
    FreeVram {
        resp: Sender<Result<usize>>,
    },
    /// Rank-local draft round (no collectives; runs between this rank's
    /// steps).
    /// `resets` clear slot draft states (request left / new admission),
    /// `appends` feed step rows of the LAST Step's `bucket` capture buffer to
    /// slot pending contexts, `proposals` ask for a 7-token draft span per
    /// slot from `(slot, anchor_token, anchor_pos)`. Ordered: resets, then
    /// appends, then proposals.
    Draft {
        bucket: usize,
        resets: Vec<usize>,
        appends: Vec<(usize, usize)>,
        proposals: Vec<(usize, u32, usize)>,
        resp: Sender<Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>>>,
    },
    /// Collective native-MTP round. Every EP rank's engine issues one command
    /// per step, including ranks with no live proposal (padding rows enter the
    /// fixed chain), with the rank's OWN context/draft buckets for the
    /// layer-78 MoE collectives — no cross-rank round-kind negotiation.
    MtpDraft {
        round: Glm52MtpRound,
        resp: Sender<Result<Vec<[u32; crate::mtp::GLM52_MTP_DRAFTS]>>>,
    },
    /// Rank-local inspection of an already pre-captured whole-step graph.
    /// This command stays on the worker so CUDA graph handles never cross the
    /// thread/context ownership boundary.
    DumpDecodeGraph {
        bucket: usize,
        png_path: PathBuf,
        title: String,
        resp: Sender<Result<CudaGraphDumpSummary>>,
    },
    Shutdown,
}

pub(crate) struct Glm52RankWorker {
    tx: Sender<Glm52RankCommand>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Glm52RankWorker {
    pub(crate) fn spawn(
        placement: Glm52RankPlacement,
        bundle: Glm52RankLoadBundle,
    ) -> Result<Self> {
        ensure!(
            bundle.plan.rank == placement.rank,
            "GLM5.2 rank load plan {} does not match placement {}",
            bundle.plan.rank,
            placement.rank
        );
        let (tx, rx) = unbounded();
        let (startup_tx, startup_rx) = bounded::<Result<()>>(1);
        let handle = thread::Builder::new()
            .name(format!("glm52-rank-{}", placement.rank))
            .spawn(move || {
                let ctx = match Glm52RankGpuContext::new(placement.device_ordinal) {
                    Ok(ctx) => ctx,
                    Err(err) => {
                        let _ = startup_tx.send(Err(err));
                        return;
                    }
                };
                let _ = startup_tx.send(Ok(()));
                rank_worker_loop(&rx, Glm52RankThreadState::new(placement, ctx, bundle));
            })
            .map_err(|err| anyhow::anyhow!("failed to spawn GLM5.2 rank worker: {err}"))?;
        startup_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker exited during startup"))??;
        Ok(Self {
            tx,
            handle: Some(handle),
        })
    }

    pub(crate) fn load_weights_async(
        &self,
        model_path: &Path,
        moe_topo: crate::Glm52MoeTopo,
        weight_staging: bool,
    ) -> Result<Receiver<Result<Glm52RankWeightLoadReport>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::LoadWeights {
                model_path: model_path.to_path_buf(),
                moe_topo,
                weight_staging,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn build_fixed_async(
        &self,
        max_model_len: usize,
        moe_topo: crate::Glm52MoeTopo,
        drafter: crate::Glm52Drafter,
        prefill_chunk_size: Option<usize>,
    ) -> Result<Receiver<Result<u64>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::BuildFixed {
                max_model_len,
                moe_topo,
                drafter,
                prefill_chunk_size,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn finish_kv_async(
        &self,
        pool_blocks: usize,
    ) -> Result<Receiver<Result<Vec<ArenaSpec>>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::FinishKv {
                pool_blocks,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn setup_comm_async(
        &self,
        unique_id: [u8; 128],
        moe_topo: crate::Glm52MoeTopo,
        tp_exchange: Option<Arc<Glm52TpExchange>>,
    ) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::SetupComm {
                unique_id: Box::new(unique_id),
                moe_topo,
                tp_exchange,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn step_async(
        &self,
        inputs: [(u32, usize); GLM52_MAX_STEP_ROWS],
        shape: Glm52StepShape,
        kv: Glm52StepKv,
        flags: Glm52StepFlags,
        sampling: Vec<Glm52RowSample>,
        seed: u64,
    ) -> Result<Receiver<Result<[u32; GLM52_MAX_STEP_ROWS]>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::Step {
                inputs: Box::new(inputs),
                shape,
                kv: Box::new(kv),
                flags,
                sampling,
                seed,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn prefill_chunk_async(
        &self,
        batch: Glm52PrefillBatch,
    ) -> Result<Receiver<Result<Glm52PrefillOutput>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::PrefillChunk {
                batch,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn load_dspark_async(&self, path: &Path) -> Result<Receiver<Result<()>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::LoadDspark {
                path: path.to_path_buf(),
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn free_vram_async(&self) -> Result<Receiver<Result<usize>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::FreeVram { resp: resp_tx })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn draft_async(
        &self,
        bucket: usize,
        resets: Vec<usize>,
        appends: Vec<(usize, usize)>,
        proposals: Vec<(usize, u32, usize)>,
    ) -> Result<Receiver<Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::Draft {
                bucket,
                resets,
                appends,
                proposals,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn mtp_draft_async(
        &self,
        round: Glm52MtpRound,
    ) -> Result<Receiver<Result<Vec<[u32; crate::mtp::GLM52_MTP_DRAFTS]>>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::MtpDraft {
                round,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    pub(crate) fn dump_decode_graph_async(
        &self,
        bucket: usize,
        png_path: PathBuf,
        title: String,
    ) -> Result<Receiver<Result<CudaGraphDumpSummary>>> {
        let (resp_tx, resp_rx) = bounded(1);
        self.tx
            .send(Glm52RankCommand::DumpDecodeGraph {
                bucket,
                png_path,
                title,
                resp: resp_tx,
            })
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(resp_rx)
    }

    fn shutdown(&mut self) -> Result<()> {
        self.request_shutdown()?;
        self.join()
    }

    pub(crate) fn request_shutdown(&self) -> Result<()> {
        self.tx
            .send(Glm52RankCommand::Shutdown)
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker channel closed"))?;
        Ok(())
    }

    fn join(&mut self) -> Result<()> {
        let Some(handle) = self.handle.take() else {
            return Ok(());
        };
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("GLM5.2 rank worker panicked"))?;
        Ok(())
    }
}

impl Drop for Glm52RankWorker {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

/// One rank executor as an engine sees it. Engines drive only LOCAL workers:
/// a cross-node fleet is one process per node joining the same DeepEP
/// communicator — the TCP rank-host (`remote.rs`) is retired
/// (`docs/models/glm52/free-running-dp.md` §9).
pub(crate) type Glm52Worker = Glm52RankWorker;

struct Glm52RankRuntime {
    model: Box<Glm52RankModel>,
    /// Second stream on the same device: the shared expert overlaps the MoE
    /// collectives on it (fork/join via events inside the decode graph).
    aux_ctx: pegainfer_kernels::tensor::DeviceContext,
    /// Populated by SetupComm (collective), after every rank's build succeeded.
    ep8: Option<Glm52MoeEpState>,
    /// Populated by SetupComm when TP is on (LL rendezvous is
    /// collective too): the runtime state plus the slice banks loaded in
    /// LoadWeights.
    tp: Option<Glm52MoeTpRank>,
    /// Kept from SetupComm for the shutdown-side teardown rendezvous: the
    /// TP LL buffers are peer-mapped on every device, so no rank may unmap
    /// (drop `tp`) until every rank has retired its GPU work — see
    /// `teardown_tp`. Stored on ALL ranks (even if this rank's TP setup
    /// failed) so the barrier's participant count is always the full fleet.
    tp_exchange: Option<Arc<Glm52TpExchange>>,
    /// Populated by LoadDspark when the drafter is enabled.
    dspark: Option<Glm52DsparkRank>,
}

/// This rank's DSpark lane: the replicated draft model, the shared forward
/// scratch, and one draft state per slot — every buffer preallocated to the
/// launch-time cap at load (the VRAM probe's ledger charged exactly this
/// footprint), because a mid-serving draft round must never hit the
/// allocator: a transient OOM there would tear the whole engine down.
struct Glm52DsparkRank {
    model: Glm52DsparkModel,
    scratch: Glm52DsparkScratch,
    // Sized to the RUNTIME slot count (admission never fills past it) — the
    // cache-length-scaled states are exactly what the dspark arena ledger
    // charges, and a ceiling-sized array would allocate unledgered GiBs.
    slots: Vec<Glm52DsparkSlotState>,
}

struct Glm52RankThreadState {
    placement: Glm52RankPlacement,
    ctx: Glm52RankGpuContext,
    bundle: Glm52RankLoadBundle,
    loaded: Option<Glm52RankGpuWeights>,
    /// TP-topology slice banks (loaded with the weights), waiting for the
    /// SetupComm rendezvous to assemble the runtime `Glm52MoeTpRank`.
    tp_slices: BTreeMap<usize, Glm52MoeTpSliceBank>,
    /// The fixed half of the two-phase model build, parked between
    /// BuildFixed's VRAM measurement and FinishKv's pool allocation.
    fixed: Option<Box<Glm52RankModelFixed>>,
    runtime: Option<Glm52RankRuntime>,
}

impl Glm52RankThreadState {
    fn new(
        placement: Glm52RankPlacement,
        ctx: Glm52RankGpuContext,
        bundle: Glm52RankLoadBundle,
    ) -> Self {
        Self {
            placement,
            ctx,
            bundle,
            loaded: None,
            tp_slices: BTreeMap::new(),
            fixed: None,
            runtime: None,
        }
    }

    fn load_weights(
        &mut self,
        model_path: &Path,
        moe_topo: crate::Glm52MoeTopo,
        weight_staging: bool,
    ) -> Result<Glm52RankWeightLoadReport> {
        let rank_load_started = Instant::now();
        let loaded = load_rank_weights_to_gpu(&self.ctx, model_path, &self.bundle, weight_staging)?;
        let resident_load_secs = rank_load_started.elapsed().as_secs_f64();
        if moe_topo.uses_tensor_replicated_moe() {
            // Tensor-replicated topology: the bundle carried no routed experts;
            // gather this rank's 1/TP intermediate slice of ALL experts for
            // every MoE layer instead.
            let manifest = Glm52WeightManifest::from_model_dir(model_path)?;
            let dev_ctx = self.ctx.device_context()?;
            let tp_slices_started = Instant::now();
            let mut slowest_layer = None::<(usize, f64)>;
            let end = if self.bundle.plan.native_mtp {
                crate::config::GLM52_MTP_LAYER + 1
            } else {
                crate::config::GLM52_LAYERS
            };
            for layer in crate::config::GLM52_DENSE_LAYERS..end {
                let layer_started = Instant::now();
                let bank = load_tp_slice_layer(
                    &dev_ctx,
                    model_path,
                    &manifest,
                    self.placement.rank,
                    moe_topo.device_count(),
                    layer,
                )?;
                self.tp_slices.insert(layer, bank);
                let elapsed = layer_started.elapsed().as_secs_f64();
                if slowest_layer.is_none_or(|(_, slowest)| elapsed > slowest) {
                    slowest_layer = Some((layer, elapsed));
                }
            }
            let (slowest_layer, slowest_secs) =
                slowest_layer.expect("tensor-replicated topology has MoE layers");
            log::info!(
                "GLM5.2 rank {} TP slice-bank load profile: layers={}, total={:.2}s, \
                 mean={:.2}s/layer, slowest_layer={} {:.2}s; this second pass re-reads and \
                 host-gathers every routed expert independently on each TP rank",
                self.placement.rank,
                self.tp_slices.len(),
                tp_slices_started.elapsed().as_secs_f64(),
                tp_slices_started.elapsed().as_secs_f64() / self.tp_slices.len() as f64,
                slowest_layer,
                slowest_secs,
            );
        }
        ensure!(
            loaded.loaded_total_bytes == loaded.weights.total_bytes,
            "GLM5.2 rank {} loaded bytes {} differ from resident raw bytes {}",
            self.placement.rank,
            loaded.loaded_total_bytes,
            loaded.weights.total_bytes
        );
        let free_vram_bytes = self.ctx.free_vram_bytes()?;
        let report = Glm52RankWeightLoadReport {
            rank: self.placement.rank,
            loaded_tensor_count: loaded.loaded_tensor_count,
            loaded_total_bytes: loaded.loaded_total_bytes,
            resident_raw_bytes: loaded.weights.total_bytes,
            loaded_to_gpu: true,
            free_vram_bytes,
        };
        self.loaded = Some(loaded.weights);
        log::info!(
            "GLM5.2 rank {} complete weight residency: base_plan={resident_load_secs:.2}s, \
             tp_slice_second_pass={:.2}s, total={:.2}s",
            self.placement.rank,
            (rank_load_started.elapsed().as_secs_f64() - resident_load_secs).max(0.0),
            rank_load_started.elapsed().as_secs_f64(),
        );
        Ok(report)
    }

    /// Phase 1: build everything except the pool-scaled KV slabs, then
    /// report this device's measured free VRAM — the number launch sizes the
    /// fleet's pool from (same probe as the FreeVram command).
    fn build_fixed(
        &mut self,
        max_model_len: usize,
        moe_topo: crate::Glm52MoeTopo,
        drafter: crate::Glm52Drafter,
        prefill_chunk_size: Option<usize>,
    ) -> Result<u64> {
        let mut weights = self
            .loaded
            .take()
            .context("GLM5.2 build_fixed called before weights were loaded")?;
        let dev_ctx = self.ctx.device_context()?;
        self.fixed = Some(Box::new(Glm52RankModel::build_fixed(
            &dev_ctx,
            &mut weights,
            max_model_len,
            moe_topo,
            moe_topo
                .uses_tensor_replicated_moe()
                .then_some(self.placement.rank),
            &drafter,
            prefill_chunk_size,
        )?));
        Ok(self.ctx.free_vram_bytes()? as u64)
    }

    /// Phase 2: allocate the pool-scaled slabs for the launch-decided count
    /// and assemble the rank runtime.
    fn finish_kv(&mut self, pool_blocks: usize) -> Result<Vec<ArenaSpec>> {
        let fixed = self
            .fixed
            .take()
            .context("GLM5.2 finish_kv called before build_fixed")?;
        let dev_ctx = self.ctx.device_context()?;
        let model = Box::new(Glm52RankModel::finish_kv(&dev_ctx, *fixed, pool_blocks)?);
        let arenas = model.kv_arenas(&dev_ctx.stream)?;
        let aux_ctx = self.ctx.auxiliary_device_context("decode aux")?;
        self.runtime = Some(Glm52RankRuntime {
            model,
            aux_ctx,
            ep8: None,
            tp: None,
            tp_exchange: None,
            dspark: None,
        });
        Ok(arenas)
    }

    fn load_dspark(&mut self, path: &Path) -> Result<()> {
        let dev_ctx = self.ctx.device_context()?;
        let runtime = self
            .runtime
            .as_mut()
            .context("GLM5.2 load_dspark before build_model")?;
        ensure!(
            runtime.dspark.is_none(),
            "GLM5.2 rank {} DSpark drafter already loaded",
            self.placement.rank
        );
        let model = Glm52DsparkModel::load(&dev_ctx, path, runtime.model.max_model_len())?;
        let scratch = Glm52DsparkScratch::new(&dev_ctx, model.cache_len())?;
        let slot_count = crate::model::glm52_decode_slots();
        let mut slots = Vec::with_capacity(slot_count);
        for _ in 0..slot_count {
            slots.push(Glm52DsparkSlotState::new(&dev_ctx, model.cache_len())?);
        }
        runtime.dspark = Some(Glm52DsparkRank {
            model,
            scratch,
            slots,
        });
        Ok(())
    }

    fn draft(
        &mut self,
        bucket: usize,
        resets: &[usize],
        appends: &[(usize, usize)],
        proposals: &[(usize, u32, usize)],
    ) -> Result<Vec<[u32; GLM52_DSPARK_DRAFTS]>> {
        let dev_ctx = self.ctx.device_context()?;
        let runtime = self
            .runtime
            .as_mut()
            .context("GLM5.2 draft before build_model")?;
        let Glm52RankRuntime { model, dspark, .. } = runtime;
        let dspark = dspark
            .as_mut()
            .context("GLM5.2 draft command without a loaded DSpark drafter")?;

        for &slot in resets {
            ensure!(slot < GLM52_MAX_BATCH_PER_RANK, "dspark reset slot {slot}");
            dspark.slots[slot].reset();
        }
        if !appends.is_empty() {
            let captured = model.captured(bucket)?;
            for &(row, slot) in appends {
                ensure!(
                    row < bucket && slot < GLM52_MAX_BATCH_PER_RANK,
                    "dspark append row {row} (bucket {bucket}) slot {slot}"
                );
                dspark.slots[slot].append_captured_row(&dev_ctx, captured, row)?;
            }
        }
        if proposals.is_empty() {
            return Ok(Vec::new());
        }
        // Proposals must arrive slot-ascending so the reply order is the
        // request order (and slots are trivially unique).
        ensure!(
            proposals.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "dspark proposals must be sorted by slot: {proposals:?}"
        );
        let mut states = Vec::with_capacity(proposals.len());
        let mut anchors = Vec::with_capacity(proposals.len());
        let mut wanted = proposals.iter().peekable();
        for (slot, state) in dspark.slots.iter_mut().enumerate() {
            let Some(&&(want_slot, anchor, anchor_pos)) = wanted.peek() else {
                break;
            };
            if want_slot != slot {
                continue;
            }
            wanted.next();
            states.push(state);
            anchors.push((anchor, anchor_pos));
        }
        ensure!(
            wanted.peek().is_none(),
            "dspark propose slot out of range in {proposals:?}"
        );
        dspark.model.propose(
            &dev_ctx,
            model.embed(),
            model.lm_head(),
            &mut states,
            &anchors,
            &mut dspark.scratch,
        )
    }

    fn mtp_draft(
        &mut self,
        round: &Glm52MtpRound,
    ) -> Result<Vec<[u32; crate::mtp::GLM52_MTP_DRAFTS]>> {
        let dev_ctx = self.ctx.device_context()?;
        let runtime = self
            .runtime
            .as_mut()
            .context("GLM5.2 native MTP draft before build_model")?;
        runtime.model.mtp_propose(
            &dev_ctx,
            &runtime.aux_ctx,
            runtime.ep8.as_mut(),
            runtime.tp.as_mut(),
            round,
        )
    }

    fn setup_comm(
        &mut self,
        unique_id: &[u8; 128],
        moe_topo: crate::Glm52MoeTopo,
        tp_exchange: Option<&Arc<Glm52TpExchange>>,
    ) -> Result<()> {
        let dev_ctx = self.ctx.device_context()?;
        let runtime = self
            .runtime
            .as_mut()
            .context("GLM5.2 setup_comm before build_model")?;
        ensure!(
            runtime.ep8.is_none(),
            "GLM5.2 rank {} DeepEP context already created",
            self.placement.rank
        );
        // Before anything fallible: every rank must hold the exchange so the
        // shutdown teardown rendezvous counts the full fleet even when this
        // rank's TP setup fails below.
        runtime.tp_exchange = tp_exchange.cloned();
        if moe_topo.uses_ep_expert_bundles() {
            // Collective: every EP rank calls this concurrently. Topology
            // selects the DeepEP shim; every width runs the SM100 DeepGEMM
            // masked expert chain.
            let ranks = moe_topo.expected_ep_size();
            let rank = self.placement.rank;
            runtime.ep8 = Some(match moe_topo {
                crate::Glm52MoeTopo::Ep8 => Glm52MoeEpState::Ep8(Box::new(
                    Glm52MoeEpRankState::new(&dev_ctx, unique_id, ranks, rank)?,
                )),
                crate::Glm52MoeTopo::Ep4 => Glm52MoeEpState::Ep4(Box::new(
                    Glm52MoeEpRankState::new(&dev_ctx, unique_id, ranks, rank)?,
                )),
                crate::Glm52MoeTopo::Ep16 => Glm52MoeEpState::Ep16(Box::new(
                    Glm52MoeEpRankState::new(&dev_ctx, unique_id, ranks, rank)?,
                )),
                crate::Glm52MoeTopo::Ep32 => Glm52MoeEpState::Ep32(Box::new(
                    Glm52MoeEpRankState::new(&dev_ctx, unique_id, ranks, rank)?,
                )),
                crate::Glm52MoeTopo::Ep64 => Glm52MoeEpState::Ep64(Box::new(
                    Glm52MoeEpRankState::new(&dev_ctx, unique_id, ranks, rank)?,
                )),
                other => anyhow::bail!("GLM5.2 {other:?} is not an expert-bundle topology"),
            });
        }
        if let Some(exchange) = tp_exchange {
            ensure!(
                !self.tp_slices.is_empty(),
                "GLM5.2 rank {} TP rendezvous without slice banks — load/setup drifted",
                self.placement.rank
            );
            let topology = match moe_topo {
                crate::Glm52MoeTopo::Tp4 => pegainfer_kernels::ops::Glm52TpTopology::Tp4,
                _ => anyhow::bail!("GLM5.2 {moe_topo:?} setup received a TP exchange"),
            };
            ensure!(
                runtime.model.is_prefill_only(),
                "GLM5.2 TP4 is prefill-only; launch with --glm52-prefill-only"
            );
            let mut state = Glm52MoeTpState::new(topology, self.placement.rank)?;
            // Collective: every TP rank reaches NCCL init concurrently.
            state.init_prefill_nccl(&dev_ctx, exchange)?;
            runtime.tp = Some(Glm52MoeTpRank {
                state,
                slices: std::mem::take(&mut self.tp_slices),
            });
        }
        Ok(())
    }

    /// Shutdown path only: retire this rank's in-flight GPU work, then block
    /// in the fleet-wide teardown rendezvous before dropping the TP state —
    /// unmapping an LL buffer pulls the mapping from under EVERY device, so
    /// it is only safe once no rank can still be replaying a step.
    fn teardown_tp(&mut self) {
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };
        let Some(exchange) = runtime.tp_exchange.take() else {
            return;
        };
        match self.ctx.device_context() {
            Ok(dev_ctx) => {
                if let Err(err) = dev_ctx.stream.synchronize() {
                    log::warn!(
                        "GLM5.2 rank {} TP teardown stream sync failed: {err:#}",
                        self.placement.rank
                    );
                }
            }
            Err(err) => log::warn!(
                "GLM5.2 rank {} TP teardown could not reach its device: {err:#}",
                self.placement.rank
            ),
        }
        exchange.teardown_rendezvous(self.placement.rank);
        runtime.tp = None;
    }

    fn step(
        &mut self,
        inputs: &[(u32, usize); GLM52_MAX_STEP_ROWS],
        shape: Glm52StepShape,
        kv: &Glm52StepKv,
        flags: Glm52StepFlags,
        sampling: &[Glm52RowSample],
        seed: u64,
    ) -> Result<[u32; GLM52_MAX_STEP_ROWS]> {
        let dev_ctx = self.ctx.device_context()?;
        let runtime = self
            .runtime
            .as_mut()
            .context("GLM5.2 step before build_model")?;
        let outputs = runtime.model.decode_step(
            &dev_ctx,
            &runtime.aux_ctx,
            runtime.ep8.as_mut(),
            runtime.tp.as_mut(),
            inputs,
            shape,
            kv,
            flags,
            sampling,
            seed,
        )?;
        #[cfg(test)]
        if crate::freerun_probe::enabled() {
            let (topk_idx, topk_weight_bits) =
                runtime.model.probe_step_route(&dev_ctx, shape.bucket)?;
            crate::freerun_probe::record_step_route(crate::freerun_probe::StepRouteRecord {
                rank: self.placement.rank,
                bucket: shape.bucket,
                active_rows: shape.active_rows,
                topk_idx,
                topk_weight_bits,
            });
        }
        Ok(outputs)
    }

    fn prefill_chunk(&mut self, batch: &Glm52PrefillBatch) -> Result<Glm52PrefillOutput> {
        batch.validate()?;
        let dev_ctx = self.ctx.device_context()?;
        let runtime = self
            .runtime
            .as_mut()
            .context("GLM5.2 prefill chunk before build_model")?;
        runtime
            .model
            .prefill_chunk(&dev_ctx, &runtime.aux_ctx, batch, runtime.tp.as_mut())
    }

    fn dump_decode_graph(
        &self,
        bucket: usize,
        png_path: &Path,
        title: &str,
    ) -> Result<CudaGraphDumpSummary> {
        let runtime = self
            .runtime
            .as_ref()
            .context("GLM5.2 graph dump before build_model")?;
        runtime.model.dump_decode_graph_png(bucket, png_path, title)
    }
}

fn rank_worker_loop(rx: &Receiver<Glm52RankCommand>, mut state: Glm52RankThreadState) {
    while let Ok(cmd) = rx.recv() {
        match cmd {
            Glm52RankCommand::LoadWeights {
                model_path,
                moe_topo,
                weight_staging,
                resp,
            } => {
                let _ = resp.send(state.load_weights(&model_path, moe_topo, weight_staging));
            }
            Glm52RankCommand::BuildFixed {
                max_model_len,
                moe_topo,
                drafter,
                prefill_chunk_size,
                resp,
            } => {
                let _ = resp.send(state.build_fixed(
                    max_model_len,
                    moe_topo,
                    drafter,
                    prefill_chunk_size,
                ));
            }
            Glm52RankCommand::FinishKv { pool_blocks, resp } => {
                let _ = resp.send(state.finish_kv(pool_blocks));
            }
            Glm52RankCommand::SetupComm {
                unique_id,
                moe_topo,
                tp_exchange,
                resp,
            } => {
                let _ = resp.send(state.setup_comm(&unique_id, moe_topo, tp_exchange.as_ref()));
            }
            Glm52RankCommand::Step {
                inputs,
                shape,
                kv,
                flags,
                sampling,
                seed,
                resp,
            } => {
                let _ = resp.send(state.step(&inputs, shape, &kv, flags, &sampling, seed));
            }
            Glm52RankCommand::PrefillChunk { batch, resp } => {
                let _ = resp.send(state.prefill_chunk(&batch));
            }
            Glm52RankCommand::DumpDecodeGraph {
                bucket,
                png_path,
                title,
                resp,
            } => {
                let _ = resp.send(state.dump_decode_graph(bucket, &png_path, &title));
            }
            Glm52RankCommand::LoadDspark { path, resp } => {
                let _ = resp.send(state.load_dspark(&path));
            }
            Glm52RankCommand::FreeVram { resp } => {
                let _ = resp.send(state.ctx.free_vram_bytes());
            }
            Glm52RankCommand::Draft {
                bucket,
                resets,
                appends,
                proposals,
                resp,
            } => {
                let _ = resp.send(state.draft(bucket, &resets, &appends, &proposals));
            }
            Glm52RankCommand::MtpDraft { round, resp } => {
                let _ = resp.send(state.mtp_draft(&round));
            }
            Glm52RankCommand::Shutdown => break,
        }
    }
    // Worker exit is load-bearing for the fleet (a silently gone rank hangs
    // every peer in the next collective) — always say why the loop ended.
    log::info!(
        "GLM5.2 rank {} worker loop exiting ({})",
        state.placement.rank,
        if rx.is_empty() {
            "shutdown or channel closed"
        } else {
            "channel closed with queued commands"
        }
    );
    // TP LL buffers are peer-mapped on every device, and a speculative
    // launch-ahead replay can still be in flight when Shutdown lands (harvest
    // only blocks on the argmax D2H). Quiesce this rank, then rendezvous with
    // the fleet before any rank unmaps — must run before the collective
    // DeepEP drop below, and keeps LL teardown independent of
    // `Glm52RankRuntime` field order.
    state.teardown_tp();
    // The DeepEP context drop is collective — it runs here as every rank's
    // worker exits its loop after its engine broadcast Shutdown.
    drop(state);
}
