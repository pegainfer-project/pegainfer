//! Expert parallelism: the fixed per-layer collective chain, and the state one
//! rank needs to walk it.
//!
//! ## Free-running ranks
//!
//! Every rank is an autonomous engine with its own scheduler thread, its own
//! slots and its own requests. The **only** runtime coupling is the collective
//! chain inside a step, and that chain is a compile-time constant: every rank
//! walks the same sequence, at the same shapes, on every step it takes. There
//! is no coordinator, no cross-rank host protocol, and nothing to negotiate —
//! a rank that has nothing to serve takes a padding step, which is a real step
//! with every row marked padding.
//!
//! NCCL pairs collectives **by entry order**, so the chain length has to be
//! structurally constant rather than merely usually equal:
//! [`K3EpRuntime::end_step`] counts the launches and refuses a step that did
//! not issue exactly [`K3EpRuntime::collectives_per_step`] of them. Plain NCCL
//! has no device-side timeout — a mispair is a silent wrong answer or a hang,
//! never an error — so this counter is the only cheap guard there is.
//!
//! ## What the chain does, and why it is exact
//!
//! Per MoE layer:
//!
//! 1. pack this rank's fixed-shape contribution (latent rows + the two router
//!    arrays, padding rows constructively determined), then **allgather** each
//!    of the three so every rank holds the global batch;
//! 2. run the ordinary masked chain over the global batch through this rank's
//!    expert window (`local_expert_base .. + groups`);
//! 3. scatter the masked W2 rows into an entry-major staging buffer — a dense
//!    full-cover pass, so entries this rank does not own become exact zeros;
//! 4. **all-reduce(sum, bf16)** that buffer in place. Every global entry is
//!    owned by exactly one rank, so the reduction only ever adds a value to
//!    zeros: exact in any reduction order and any float format. (A general
//!    bf16 bulk all-reduce is *not* safe — see
//!    `docs/lessons/kimi-bringup-numerics.md`. Disjoint support is the whole
//!    reason this one is, and the bitwise gate is what proves it. Do not
//!    "optimize" this into an overlapping-support reduction.)
//! 5. combine this rank's own token rows out of the reduced staging buffer.
//!
//! Four collectives per MoE layer, constant shapes, constant count.
//!
//! ## Rules this module exists to keep
//!
//! * **Comms are created on the rank's own worker thread.** The launcher builds
//!   executors on the main thread; the comm is minted lazily on the first step,
//!   which runs on the scheduler thread, after `bind_thread`. Creating comms on
//!   a controller thread and moving them into workers is a recorded way to get
//!   invalid-handle symptoms and hangs.
//! * **Every rank finishes loading before any rank calls into NCCL.** The
//!   launcher loads all ranks' weights before it spawns a single scheduler
//!   thread, and comm init is lazy on the first step, so it is ordered after
//!   every load by construction. One rank OOMing during load must not strand
//!   its peers inside `ncclCommInitRank`, which never times out.
//! * **Protocol-max shapes.** Every collective buffer is allocated once at the
//!   worst case and is pointer-stable; a rank's live batch never reaches the
//!   wire.

use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::time::Duration;

use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::ops::K3_ROUTER_TOPK;
use pegainfer_kernels::tensor::DeviceContext;

use crate::config::K3_ROUTED_EXPERT_HIDDEN;

/// How long a peer waits for rank 0 to publish the NCCL unique id before it
/// gives up. Rank 0 mints on its own first step, so this only ever expires
/// when rank 0 died on the way there.
const ID_RENDEZVOUS_TIMEOUT: Duration = Duration::from_secs(300);

/// Collectives one MoE layer issues: three allgathers (latent, expert ids,
/// router weights) and one all-reduce over the entry staging buffer.
const COLLECTIVES_PER_MOE_LAYER: usize = 4;

/// The in-process handshake that gives every rank of one EP group the same
/// NCCL unique id. Rank 0 mints it on its first step; peers wait, with a
/// timeout that errors rather than hanging forever.
///
/// Deliberately in-process: an EP group is one process with one thread per
/// rank. A multi-node group would replace this with a real out-of-band
/// exchange and nothing else about the chain would move.
#[derive(Debug)]
pub struct K3EpRendezvous {
    ranks: usize,
    id: Mutex<Option<[::core::ffi::c_char; 128]>>,
    ready: Condvar,
}

impl K3EpRendezvous {
    /// A rendezvous for an `ranks`-wide EP group.
    #[must_use]
    pub fn new(ranks: usize) -> Arc<Self> {
        Arc::new(Self {
            ranks,
            id: Mutex::new(None),
            ready: Condvar::new(),
        })
    }

    pub(crate) fn ranks(&self) -> usize {
        self.ranks
    }

    fn nccl_id(&self, rank: usize) -> Result<cudarc::nccl::Id> {
        let mut slot = self.id.lock().expect("K3 EP rendezvous poisoned");
        if rank == 0 {
            ensure!(slot.is_none(), "K3 EP NCCL id minted twice");
            let id = cudarc::nccl::Id::new()
                .map_err(|err| anyhow::anyhow!("K3 EP NCCL unique id creation failed: {err:?}"))?;
            *slot = Some(*id.internal());
            self.ready.notify_all();
            return Ok(id);
        }
        while slot.is_none() {
            let (guard, timeout) = self
                .ready
                .wait_timeout(slot, ID_RENDEZVOUS_TIMEOUT)
                .expect("K3 EP rendezvous poisoned");
            slot = guard;
            if timeout.timed_out() && slot.is_none() {
                bail!(
                    "K3 EP rank {rank} waited {}s for the NCCL unique id and rank 0 never \
                     published it",
                    ID_RENDEZVOUS_TIMEOUT.as_secs()
                );
            }
        }
        Ok(cudarc::nccl::Id::uninit(
            *slot.as_ref().expect("checked above"),
        ))
    }
}

/// cudarc's `Comm` holds a raw `ncclComm_t` and is `!Send`. This one is minted
/// on, and used from, exactly one worker thread — the scheduler thread that
/// steps this rank's executor — and is never touched concurrently. The
/// containing executor is moved to that thread before the comm exists.
struct K3EpComm(cudarc::nccl::Comm);

// SAFETY: see the type's documentation — one thread creates it, the same one
// thread uses it, and it is dropped there too.
unsafe impl Send for K3EpComm {}

/// The collective buffers, allocated once at protocol-max shape and never
/// resized, so their device pointers are stable for the process's life.
pub(crate) struct K3EpBuffers {
    /// This rank's `[max_batch, latent]` bf16 contribution.
    pub(crate) latent_send: CudaSlice<bf16>,
    /// The gathered `[ep_size * max_batch, latent]` bf16 global batch.
    pub(crate) latent_recv: CudaSlice<bf16>,
    /// This rank's `[max_batch * topk]` global expert ids, `-1` for padding.
    pub(crate) idx_send: CudaSlice<i32>,
    pub(crate) idx_recv: CudaSlice<i32>,
    /// This rank's `[max_batch * topk]` router weights, `0` for padding.
    pub(crate) weight_send: CudaSlice<f32>,
    pub(crate) weight_recv: CudaSlice<f32>,
    /// Entry-major `[ep_size * max_batch * topk, latent]` bf16 staging: the
    /// all-reduce's operand, and the entry combine's input.
    pub(crate) stage: CudaSlice<bf16>,
}

impl K3EpBuffers {
    fn new(ctx: &DeviceContext, max_batch: usize, ep_size: usize) -> Result<Self> {
        let stream = &ctx.stream;
        let latent = K3_ROUTED_EXPERT_HIDDEN;
        let topk = K3_ROUTER_TOPK;
        let global_rows = ep_size * max_batch;
        Ok(Self {
            latent_send: stream.alloc_zeros(max_batch * latent)?,
            latent_recv: stream.alloc_zeros(global_rows * latent)?,
            idx_send: stream.alloc_zeros(max_batch * topk)?,
            idx_recv: stream.alloc_zeros(global_rows * topk)?,
            weight_send: stream.alloc_zeros(max_batch * topk)?,
            weight_recv: stream.alloc_zeros(global_rows * topk)?,
            stage: stream.alloc_zeros(global_rows * topk * latent)?,
        })
    }
}

/// One rank's expert-parallel runtime.
pub(crate) struct K3EpRuntime {
    rank: usize,
    ep_size: usize,
    /// Rows this rank contributes to every allgather — the protocol-max row
    /// count, never a live bucket.
    max_batch: usize,
    /// The rank's first global expert id: the masked chain's window.
    local_expert_base: usize,
    /// Global routed-expert count, i.e. the range that marks an entry active.
    routed_experts: usize,
    rendezvous: Arc<K3EpRendezvous>,
    comm: Option<K3EpComm>,
    pub(crate) buffers: K3EpBuffers,
    /// Collective launches issued since [`Self::begin_step`].
    issued: usize,
    /// What that count must be when the step ends.
    collectives_per_step: usize,
}

impl K3EpRuntime {
    pub(crate) fn new(
        ctx: &DeviceContext,
        rendezvous: Arc<K3EpRendezvous>,
        rank: usize,
        max_batch: usize,
        local_expert_base: usize,
        routed_experts: usize,
        moe_layers: usize,
    ) -> Result<Self> {
        let ep_size = rendezvous.ranks();
        ensure!(
            ep_size > 1 && rank < ep_size,
            "K3 EP runtime needs an ep_size above 1 and a rank inside it, got rank {rank} of \
             {ep_size}"
        );
        let buffers = K3EpBuffers::new(ctx, max_batch, ep_size)?;
        Ok(Self {
            rank,
            ep_size,
            max_batch,
            local_expert_base,
            routed_experts,
            rendezvous,
            comm: None,
            buffers,
            issued: 0,
            collectives_per_step: moe_layers * COLLECTIVES_PER_MOE_LAYER,
        })
    }

    pub(crate) fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Token rows the masked chain runs over: the whole fleet's batch.
    pub(crate) fn chain_tokens(&self) -> usize {
        self.ep_size * self.max_batch
    }

    /// This rank's first row in the gathered global batch.
    pub(crate) fn token_base(&self) -> usize {
        self.rank * self.max_batch
    }

    pub(crate) fn local_expert_base(&self) -> usize {
        self.local_expert_base
    }

    pub(crate) fn routed_experts(&self) -> usize {
        self.routed_experts
    }

    pub(crate) fn collectives_per_step(&self) -> usize {
        self.collectives_per_step
    }

    /// Mint this rank's communicator, on the thread that will use it.
    ///
    /// Collective: every rank must reach it. It is called from the first step,
    /// which is ordered after every rank's weight load by construction — the
    /// launcher loads all ranks before it spawns any scheduler thread, so one
    /// rank running out of memory during load cannot strand its peers inside
    /// `ncclCommInitRank`, which has no timeout.
    pub(crate) fn ensure_comm(&mut self, ctx: &DeviceContext) -> Result<()> {
        if self.comm.is_some() {
            return Ok(());
        }
        let id = self.rendezvous.nccl_id(self.rank)?;
        let comm = cudarc::nccl::Comm::from_rank(ctx.stream.clone(), self.rank, self.ep_size, id)
            .map_err(|err| anyhow::anyhow!("K3 EP NCCL comm init failed: {err:?}"))?;
        log::info!(
            "K3 EP rank {} of {} joined the NCCL group: {} collectives per step",
            self.rank,
            self.ep_size,
            self.collectives_per_step
        );
        self.comm = Some(K3EpComm(comm));
        Ok(())
    }

    /// Start a step's collective ledger.
    pub(crate) fn begin_step(&mut self) {
        self.issued = 0;
    }

    /// Close the ledger. A step that issued anything other than the constant
    /// has left this rank out of phase with its peers, and every collective
    /// from here on pairs against the wrong step.
    pub(crate) fn end_step(&self) -> Result<()> {
        debug_assert_eq!(
            self.issued, self.collectives_per_step,
            "K3 EP collective chain length is a compile-time constant"
        );
        ensure!(
            self.issued == self.collectives_per_step,
            "K3 EP rank {} issued {} collectives this step but the chain is a fixed {} — this \
             rank is now out of phase with its peers",
            self.rank,
            self.issued,
            self.collectives_per_step
        );
        Ok(())
    }

    /// Gather the three dispatch slabs. Shapes are validated host-side because
    /// NCCL will not: a receive buffer one element short is a memory stomp.
    pub(crate) fn all_gather_dispatch(&mut self) -> Result<()> {
        let latent = K3_ROUTED_EXPERT_HIDDEN;
        let topk = K3_ROUTER_TOPK;
        let send_rows = self.max_batch;
        let recv_rows = self.ep_size * self.max_batch;
        ensure!(
            self.buffers.latent_send.len() == send_rows * latent
                && self.buffers.latent_recv.len() == recv_rows * latent
                && self.buffers.idx_send.len() == send_rows * topk
                && self.buffers.idx_recv.len() == recv_rows * topk
                && self.buffers.weight_send.len() == send_rows * topk
                && self.buffers.weight_recv.len() == recv_rows * topk,
            "K3 EP dispatch buffers are not at protocol-max shape for {} ranks x {send_rows} rows",
            self.ep_size
        );
        // Split the borrow by hand: the communicator and the buffers are
        // disjoint fields, and NCCL wants both at once.
        let comm = &self
            .comm
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("K3 EP rank {} has no communicator", self.rank))?
            .0;
        let buffers = &mut self.buffers;
        comm.all_gather(&buffers.latent_send, &mut buffers.latent_recv)
            .map_err(|err| anyhow::anyhow!("K3 EP latent all-gather failed: {err:?}"))?;
        comm.all_gather(&buffers.idx_send, &mut buffers.idx_recv)
            .map_err(|err| anyhow::anyhow!("K3 EP expert-id all-gather failed: {err:?}"))?;
        comm.all_gather(&buffers.weight_send, &mut buffers.weight_recv)
            .map_err(|err| anyhow::anyhow!("K3 EP router-weight all-gather failed: {err:?}"))?;
        self.issued += 3;
        Ok(())
    }

    /// Merge the fleet's expert outputs. Disjoint support makes this exact:
    /// each entry's row comes from its expert's home rank and every other rank
    /// contributed zeros there.
    pub(crate) fn all_reduce_stage(&mut self) -> Result<()> {
        let entries = self.ep_size * self.max_batch * K3_ROUTER_TOPK;
        let want = entries * K3_ROUTED_EXPERT_HIDDEN;
        ensure!(
            self.buffers.stage.len() == want,
            "K3 EP staging buffer is {} elements, the chain's fixed shape is {want}",
            self.buffers.stage.len()
        );
        let comm = &self
            .comm
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("K3 EP rank {} has no communicator", self.rank))?
            .0;
        let buffers = &mut self.buffers;
        comm.all_reduce_in_place(&mut buffers.stage, &cudarc::nccl::ReduceOp::Sum)
            .map_err(|err| anyhow::anyhow!("K3 EP expert-output all-reduce failed: {err:?}"))?;
        self.issued += 1;
        Ok(())
    }
}

/// A step that fails under expert parallelism has already left the group out
/// of phase: the survivors' pending collectives pair against the wrong step,
/// which is deterministic garbage rather than an error, and plain NCCL never
/// times out. There is nothing to recover, so the rank takes the process down
/// instead of returning into the scheduler's fail-the-batch-and-keep-serving
/// path (which stays, for single-rank).
// Exiting the process is the point, not a shortcut: there is no state from
// which this group can serve a correct next token.
#[allow(clippy::exit)]
pub(crate) fn ep_fatal(rank: usize, phase: &str, error: &anyhow::Error) -> ! {
    log::error!(
        "K3 EP rank {rank} failed during {phase}: {error:#}. The EP group cannot recover from a \
         missed step — every peer's next collective would pair against the wrong one — so this \
         process is exiting."
    );
    std::process::exit(1);
}
