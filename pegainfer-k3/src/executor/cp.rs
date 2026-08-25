//! Context-parallel (CP) prefill: one long prompt split contiguously across
//! the expert-parallel ranks, each rank walking its own segment through the
//! same chunk step, with three per-layer exchanges stitching the segments
//! back into one sequence (see `docs/models/k3/cp-lane-design.md`):
//!
//! * **KDA (KCP)**: the recurrence's per-token update is affine in the state
//!   (`S' = A·S + b` with `A`, `b` functions of the inputs alone), so a whole
//!   segment collapses to `S_out = M_seg·S_in + D_seg`. Each rank exports its
//!   `(M, D)` package in one fused FlashKDA pass (`k3_flash_kda_fwd_md` —
//!   kernel 1 once, a dual-state kernel 2 carrying `M` from an in-kernel
//!   identity seed with `v = 0` and `D` from a zero seed with real `v`) —
//!   then every rank folds its upstream packages into its true input state
//!   with per-head fp32 GEMMs and runs the real forward once. Rank 0's real
//!   forward *is* its package (`D`, with a known zero input), and the last
//!   rank exports nothing.
//! * **conv halo**: a segment's first `K3_CONV_STATE` convolution windows
//!   reach into the previous segment. The upstream rank publishes its last
//!   `K3_CONV_STATE` *normed* rows; the receiver projects them through the
//!   q/k/v bands itself and lands them as its carried window — the batched
//!   conv kernel then runs unchanged.
//! * **MLA**: each rank publishes its segment's post-norm latent and rope
//!   rows; a rank assembles rows `0..seg_end` straight into the dense-FMHA
//!   context scratch (the paged gather is bypassed) and the existing
//!   bottom-right-aligned causal FMHA serves its queries at
//!   `t_kv = seg_start + seg_len`.
//!
//! The exchange transport is in-process ranks and plain peer-access
//! device-to-device copies, with every ordering edge expressed **on-device**
//! through CUDA events — the host never syncs a stream inside a superstep.
//! Each of the roughly `2 × 69 + 24` windows per superstep runs the same
//! four-beat protocol on every rank:
//!
//! 1. record my *publish* event (all my publish writes are enqueued), then
//!    announce it through my `published` counter;
//! 2. for each rank I read from this window, spin (host, enqueue-side only)
//!    until it announced, then make my stream wait on its publish event;
//! 3. enqueue my peer reads, record my *consume* event, announce it;
//! 4. for each rank that reads *my* buffers this window, spin until it
//!    announced, then make my stream wait on its consume event — so my next
//!    window's publish writes cannot overwrite what a peer is still reading.
//!
//! The host spins wait only for peer *enqueue* progress (announces follow
//! enqueue-only work), so enqueue threads run the whole superstep ahead of
//! the GPUs and every real dependency resolves stream-to-stream. One event
//! pair per rank suffices: a wait can only ever attach to the intended
//! record or a *later* one on the same stream (the counter handshake orders
//! record before wait at enqueue time), which is at worst conservative,
//! never early. Counters are monotonic across supersteps — every rank
//! passes the same collective window count, so the slots agree at each
//! superstep boundary even as CP ranks rotate.

use std::ops::Range;
use std::sync::Arc;
use std::sync::Barrier;
use std::sync::Mutex;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys as cu_sys;
use half::bf16;
use pegainfer_kernels::ops::K3_KDA_HEAD_DIM;
use pegainfer_kernels::ops::K3_KDA_HEADS;
use pegainfer_kernels::ops::gemm_strided_batched_f32;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::active_cu_stream;

use super::buffers::K3_CONV_STATE;
use super::buffers::K3_KDA_STATE;
use super::buffers::copy_rows;
use super::whale_gang::K3WhaleGang;
use crate::config::K3_ATTN_INNER;
use crate::config::K3_HIDDEN;
use crate::config::K3_KV_LORA_RANK;
use crate::config::K3_QK_ROPE_HEAD_DIM;

/// Raw device base pointers one CP rank publishes for its peers to read.
/// Valid only while the owning rank's [`K3CpScratch`] is alive and the group
/// is inside one superstep's exchange protocol.
#[derive(Clone, Copy, Default)]
pub(crate) struct K3CpPeerPtrs {
    pub(crate) normed_tail: u64,
    pub(crate) kda_m: u64,
    pub(crate) kda_d: u64,
    pub(crate) mla_latent: u64,
    pub(crate) mla_rope: u64,
    /// Raw `CUevent` handles for the owning rank's publish/consume events —
    /// peers wait on these cross-device (the safe wrapper refuses foreign
    /// contexts); the owning [`K3CpScratch`] keeps the events alive.
    pub(crate) publish_event: u64,
    pub(crate) consume_event: u64,
}

/// Which ranks a window couples. `Halo` moves the conv carry one hop down
/// the chain; `Upstream` fans every upstream rank's publication down to all
/// of its successors (KDA packages, MLA latents).
#[derive(Clone, Copy)]
pub(crate) enum K3CpWindowKind {
    Halo,
    Upstream,
}

impl K3CpWindowKind {
    /// Ranks whose publications `me` reads this window. CP ranks — the fleet
    /// gang maps them through its member table.
    pub(crate) fn reads_from(self, me: usize) -> Range<usize> {
        match self {
            Self::Halo => me.saturating_sub(1)..me,
            Self::Upstream => 0..me,
        }
    }

    /// Ranks that read `me`'s publications this window.
    pub(crate) fn read_by(self, me: usize, cp_size: usize) -> Range<usize> {
        match self {
            Self::Halo => (me + 1).min(cp_size)..(me + 2).min(cp_size),
            Self::Upstream => me + 1..cp_size,
        }
    }
}

/// The coordination substrate one CP scratch runs its exchange windows over:
/// the in-process gang (peer access + CUDA events) or the fleet whale gang
/// (fabric slabs + doorbells). The forward path is agnostic — it snapshots a
/// [`K3CpWindowSync`] and calls [`K3CpSyncHandle::exchange`], and the window
/// protocol semantics are identical either way.
#[derive(Clone)]
pub(crate) enum K3CpSyncHandle {
    Local(Arc<K3CpGroup>),
    Fleet(Arc<K3WhaleGang>),
}

impl K3CpSyncHandle {
    pub(crate) fn exchange(
        &self,
        ctx: &DeviceContext,
        sync: &K3CpWindowSync,
        consume: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        match (self, sync) {
            (Self::Local(group), K3CpWindowSync::Local { .. }) => {
                group.exchange(ctx, sync, consume)
            }
            (
                Self::Fleet(gang),
                K3CpWindowSync::Fleet {
                    cp_rank,
                    kind,
                    gang: members,
                    value,
                },
            ) => gang.exchange(ctx, *cp_rank, *kind, members, *value, consume),
            _ => anyhow::bail!("K3 CP window sync does not match its coordination substrate"),
        }
    }
}

/// The borrow-free snapshot one exchange window needs: taken from the
/// scratch *before* the `consume` closure mutably captures it.
pub(crate) enum K3CpWindowSync {
    Local {
        cp_rank: usize,
        kind: K3CpWindowKind,
        /// `(publish_event, consume_event)` raw handles per CP rank.
        events: Vec<(u64, u64)>,
    },
    Fleet {
        cp_rank: usize,
        kind: K3CpWindowKind,
        /// The whale's global ranks in CP order.
        gang: Vec<usize>,
        /// This window's doorbell value.
        value: u64,
    },
}

/// The in-process CP gang: a barrier and a published pointer table shared by
/// every rank of one context-parallel prefill. One group serves any number of
/// consecutive supersteps; the pointer table is republished at each entry.
pub struct K3CpGroup {
    cp_size: usize,
    barrier: Barrier,
    ptrs: Mutex<Vec<Option<K3CpPeerPtrs>>>,
    /// Per-rank monotonic count of windows whose publish event is recorded.
    /// Announces enqueue-side progress only — never device completion.
    published: Vec<AtomicU64>,
    /// Same for the consume events.
    consumed: Vec<AtomicU64>,
}

/// A peer that stops announcing mid-superstep died mid-protocol (a gang
/// prefill failure is engine-fatal); give slow enqueue threads — which can
/// legitimately block on launch-queue backpressure for a superstep — ample
/// slack before declaring it.
const K3_CP_EXCHANGE_DEADLINE: Duration = Duration::from_secs(60);

impl K3CpGroup {
    pub fn new(cp_size: usize) -> Result<Arc<Self>> {
        ensure!(
            cp_size >= 2,
            "a K3 CP group needs at least two ranks, got {cp_size}"
        );
        Ok(Arc::new(Self {
            cp_size,
            barrier: Barrier::new(cp_size),
            ptrs: Mutex::new(vec![None; cp_size]),
            published: (0..cp_size).map(|_| AtomicU64::new(0)).collect(),
            consumed: (0..cp_size).map(|_| AtomicU64::new(0)).collect(),
        }))
    }

    pub fn cp_size(&self) -> usize {
        self.cp_size
    }

    /// Publish this rank's buffer pointers and return the whole gang's table.
    /// Collective: every rank must call it once per superstep entry.
    pub(crate) fn publish_and_snapshot(
        &self,
        cp_rank: usize,
        mine: K3CpPeerPtrs,
    ) -> Result<Vec<K3CpPeerPtrs>> {
        {
            let mut table = self.ptrs.lock().expect("K3 CP pointer table poisoned");
            table[cp_rank] = Some(mine);
        }
        self.barrier.wait();
        let snapshot = {
            let table = self.ptrs.lock().expect("K3 CP pointer table poisoned");
            table
                .iter()
                .map(|entry| entry.context("a K3 CP rank never published its buffers"))
                .collect::<Result<Vec<_>>>()?
        };
        // Every rank ran the same collective window count last superstep, so
        // the counters must agree here — a skew means a rank skipped a
        // window, which the event protocol cannot survive.
        let base = self.published[0].load(Ordering::Acquire);
        for rank in 0..self.cp_size {
            let published = self.published[rank].load(Ordering::Acquire);
            let consumed = self.consumed[rank].load(Ordering::Acquire);
            ensure!(
                published == base && consumed == base,
                "K3 CP window counters skewed at superstep entry: \
                 rank {rank} at publish {published} / consume {consumed}, expected {base}"
            );
        }
        // Hold everyone until the slowest reader has its snapshot, so a rank
        // entering the NEXT superstep cannot republish underneath it.
        self.barrier.wait();
        Ok(snapshot)
    }

    /// One exchange window: everything this rank published is on-device
    /// before its peers read, and everything it read is on-device before its
    /// peers overwrite — both edges expressed as stream waits on peer
    /// events, never as host syncs (see the module doc for the four-beat
    /// protocol and its safety argument). `consume` issues this rank's reads
    /// of peer buffers on its own stream. Collective — every rank must reach
    /// every window, in the same order, with the same window kind.
    pub(crate) fn exchange(
        &self,
        ctx: &DeviceContext,
        sync: &K3CpWindowSync,
        consume: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        let K3CpWindowSync::Local {
            cp_rank: me,
            kind,
            events,
        } = sync
        else {
            anyhow::bail!("an in-process K3 CP group was handed a fleet window sync");
        };
        let me = *me;
        ensure!(
            events.len() == self.cp_size,
            "K3 CP window sync of {} ranks in a {}-rank gang",
            events.len(),
            self.cp_size
        );
        // Only this thread advances its own slot, so Relaxed reads it back.
        let window = self.published[me].load(Ordering::Relaxed) + 1;
        record_event(ctx, events[me].0).context("K3 CP publish event record failed")?;
        self.published[me].store(window, Ordering::Release);
        for rank in kind.reads_from(me) {
            self.await_announce(&self.published[rank], window, rank, "publish")?;
            wait_event(ctx, events[rank].0).context("K3 CP publish event wait failed")?;
        }
        consume()?;
        record_event(ctx, events[me].1).context("K3 CP consume event record failed")?;
        self.consumed[me].store(window, Ordering::Release);
        for rank in kind.read_by(me, self.cp_size) {
            self.await_announce(&self.consumed[rank], window, rank, "consume")?;
            wait_event(ctx, events[rank].1).context("K3 CP consume event wait failed")?;
        }
        Ok(())
    }

    /// Spin until `counter` reaches `window` — waiting on a peer *thread*'s
    /// enqueue progress, never on a device. A peer that stops announcing has
    /// died mid-protocol; time out instead of hanging the gang.
    fn await_announce(
        &self,
        counter: &AtomicU64,
        window: u64,
        rank: usize,
        stage: &str,
    ) -> Result<()> {
        if counter.load(Ordering::Acquire) >= window {
            return Ok(());
        }
        let deadline = Instant::now() + K3_CP_EXCHANGE_DEADLINE;
        let mut lap = 0u32;
        while counter.load(Ordering::Acquire) < window {
            lap = lap.wrapping_add(1);
            if lap % 1024 == 0 {
                ensure!(
                    Instant::now() < deadline,
                    "K3 CP rank {rank} never announced its {stage} for exchange window \
                     {window} — a gang member died mid-protocol"
                );
                std::thread::yield_now();
            } else {
                std::hint::spin_loop();
            }
        }
        Ok(())
    }
}

fn new_event(ctx: &DeviceContext) -> Result<CudaEvent> {
    ctx.stream
        .context()
        .new_event(None)
        .map_err(|error| anyhow::anyhow!("{error}"))
}

fn record_event(ctx: &DeviceContext, event: u64) -> Result<()> {
    // SAFETY: the handle is a live event published through the gang table —
    // its owning scratch outlives the gang's supersteps — recorded on this
    // rank's active stream.
    unsafe {
        cudarc::driver::result::event::record(event as cu_sys::CUevent, active_cu_stream(ctx))
    }
    .map_err(|error| anyhow::anyhow!("{error}"))
}

fn wait_event(ctx: &DeviceContext, event: u64) -> Result<()> {
    // SAFETY: as above; cross-device stream waits are exactly what
    // `cuStreamWaitEvent` supports (the safe wrapper just refuses foreign
    // contexts). The counter handshake ordered the peer's record before this
    // call, so the wait attaches to the intended record or a later one.
    unsafe {
        cudarc::driver::result::stream::wait_event(
            active_cu_stream(ctx),
            event as cu_sys::CUevent,
            cu_sys::CUevent_wait_flags::CU_EVENT_WAIT_DEFAULT,
        )
    }
    .map_err(|error| anyhow::anyhow!("{error}"))
}

/// The per-rank segment floor for a *fleet* whale gang, in tokens: below
/// ~2k-token segments FlashKDA falls off its throughput plateau (the 2026-08
/// segment sweep: 4224-token segments run at 96% of peak, 1056 at 87%, 264 at
/// 58%), so a wider gang stops paying for itself. The in-process lane's floor
/// stays [`K3_CONV_STATE`]-based ([`k3_cp_admits`]) — its width is fixed at
/// arm time, not chosen per prompt.
pub const K3_CP_SEGMENT_FLOOR: usize = 2048;

/// Whether a prompt of `total` tokens is CP-eligible: every segment must
/// outspan the conv window (the halo exchange publishes exactly
/// [`K3_CONV_STATE`] rows) and fit one chunk step (M0: one superstep). This
/// is the one place the admission window lives — the scheduler asks it, and
/// [`k3_cp_segments`]' consumers re-check it defensively.
pub fn k3_cp_admits(total: usize, cp_size: usize, chunk_tokens: usize) -> bool {
    let shortest = total / cp_size;
    let longest = shortest + usize::from(!total.is_multiple_of(cp_size));
    shortest > K3_CONV_STATE && longest <= chunk_tokens
}

/// Split `total` prompt tokens into `cp_size` contiguous segments, earlier
/// ranks taking the remainder — the later segments carry the taller MLA
/// context triangle, so they get the shorter extends.
pub(crate) fn k3_cp_segments(total: usize, cp_size: usize) -> Vec<(usize, usize)> {
    let base = total / cp_size;
    let extra = total % cp_size;
    let mut segments = Vec::with_capacity(cp_size);
    let mut start = 0usize;
    for rank in 0..cp_size {
        let len = base + usize::from(rank < extra);
        segments.push((start, len));
        start += len;
    }
    segments
}

/// One rank's CP working set: the buffers it publishes to its peers, the
/// receive arenas it folds their packages in, and the gang handle. Allocated
/// exactly once per executor — the pool grants to the gang's devices must
/// precede every allocation they cover — and re-armed per superstep with
/// that superstep's split and CP rank (the rank is a function of the job's
/// poster, so it rotates job to job while the buffers stay put).
pub(crate) struct K3CpScratch {
    pub(crate) sync: K3CpSyncHandle,
    pub(crate) cp_rank: usize,
    pub(crate) cp_size: usize,
    /// `(start, len)` of every rank's segment, re-derived per superstep.
    pub(crate) segments: Vec<(usize, usize)>,
    /// The gang's published pointer table, snapshotted per superstep.
    pub(crate) peers: Vec<K3CpPeerPtrs>,
    // Publish buffers — peers read these inside exchange windows.
    /// Last `K3_CONV_STATE` normed rows of this segment, `[3, hidden]` bf16.
    pub(crate) normed_tail: CudaSlice<bf16>,
    /// This segment's KDA transition `M`, `[heads, 128, 128]` f32.
    pub(crate) kda_m: CudaSlice<f32>,
    /// This segment's KDA zero-state output `D` (rank 0: its final state).
    pub(crate) kda_d: CudaSlice<f32>,
    /// This segment's post-norm MLA latents, `[seg_cap, 512]` bf16.
    pub(crate) mla_latent_pub: CudaSlice<bf16>,
    /// This segment's shared rope halves, `[seg_cap, 64]` bf16.
    pub(crate) mla_rope_pub: CudaSlice<bf16>,
    // Local working buffers.
    /// Received upstream normed tail, `[4, hidden]` (row 3 is bucket padding).
    pub(crate) halo_normed: CudaSlice<bf16>,
    /// Band-GEMM partial for the halo projection, `[4, inner]` f32.
    pub(crate) halo_partial: CudaSlice<f32>,
    /// Landed halo inputs, `[4, inner]` bf16.
    pub(crate) halo_xs: CudaSlice<bf16>,
    /// Received packages, `[cp_size, K3_KDA_STATE]` each.
    pub(crate) recv_m: CudaSlice<f32>,
    pub(crate) recv_d: CudaSlice<f32>,
    merge_a: CudaSlice<f32>,
    merge_b: CudaSlice<f32>,
    /// Pool write indices for the upstream rows `0..upstream_len` — set on
    /// the owner (last CP rank) so the MLA exchange persists the received
    /// context into the owner's paged pool for decode; 0 elsewhere.
    pub(crate) upstream_kv_rows: CudaSlice<i32>,
    pub(crate) upstream_len: usize,
    seg_cap: usize,
    /// This rank's exchange events; peers wait on their raw handles via the
    /// gang table, so they must live exactly as long as the scratch. Present
    /// exactly on the in-process substrate — the fleet orders with doorbells.
    publish_event: Option<CudaEvent>,
    consume_event: Option<CudaEvent>,
    /// The committed whale sequence the current fleet superstep serves —
    /// the doorbell value base. Meaningless on the in-process substrate.
    whale_seq: u64,
    /// Exchange windows this fleet superstep has opened so far.
    whale_window: u64,
    /// The current whale's global ranks in CP order (fleet only).
    whale_gang: Vec<usize>,
}

impl K3CpScratch {
    pub(crate) fn new(ctx: &DeviceContext, group: Arc<K3CpGroup>, seg_cap: usize) -> Result<Self> {
        let cp_size = group.cp_size();
        let stream = &ctx.stream;
        Ok(Self {
            // Meaningful only between an `arm` and the superstep it opened.
            cp_rank: 0,
            cp_size,
            segments: Vec::new(),
            peers: Vec::new(),
            normed_tail: stream.alloc_zeros(K3_CONV_STATE * K3_HIDDEN)?,
            kda_m: stream
                .alloc_zeros(K3_KDA_STATE)
                .context("alloc K3 CP transition package")?,
            kda_d: stream.alloc_zeros(K3_KDA_STATE)?,
            mla_latent_pub: stream
                .alloc_zeros(seg_cap * K3_KV_LORA_RANK)
                .context("alloc K3 CP latent publish buffer")?,
            mla_rope_pub: stream.alloc_zeros(seg_cap * K3_QK_ROPE_HEAD_DIM)?,
            halo_normed: stream.alloc_zeros((K3_CONV_STATE + 1) * K3_HIDDEN)?,
            halo_partial: stream.alloc_zeros((K3_CONV_STATE + 1) * K3_ATTN_INNER)?,
            halo_xs: stream.alloc_zeros((K3_CONV_STATE + 1) * K3_ATTN_INNER)?,
            recv_m: stream
                .alloc_zeros(cp_size * K3_KDA_STATE)
                .context("alloc K3 CP package arena")?,
            recv_d: stream.alloc_zeros(cp_size * K3_KDA_STATE)?,
            merge_a: stream.alloc_zeros(K3_KDA_STATE)?,
            merge_b: stream.alloc_zeros(K3_KDA_STATE)?,
            upstream_kv_rows: stream
                .alloc_zeros(seg_cap.saturating_mul(cp_size.saturating_sub(1)).max(1))
                .context("alloc K3 CP upstream index buffer")?,
            upstream_len: 0,
            seg_cap,
            publish_event: Some(new_event(ctx).context("create K3 CP publish event")?),
            consume_event: Some(new_event(ctx).context("create K3 CP consume event")?),
            whale_seq: 0,
            whale_window: 0,
            whale_gang: Vec::new(),
            sync: K3CpSyncHandle::Local(group),
        })
    }

    /// The fleet variant: publish buffers are carved out of this rank's
    /// fabric slab (so peers across processes can read them), local working
    /// buffers stay pool allocations, and the recv arenas are sized for the
    /// widest gang the world can seat. The slab arrives zeroed from its
    /// allocation, matching the local constructor's `alloc_zeros`.
    pub(crate) fn new_fleet(
        ctx: &DeviceContext,
        gang: Arc<K3WhaleGang>,
        seg_cap: usize,
    ) -> Result<Self> {
        let world = gang.world();
        let mine = gang.peer_ptrs(gang.rank())?;
        // SAFETY: each pointer addresses a live region of this rank's own
        // fabric slab, disjoint by layout, sized exactly as claimed, and the
        // slab is never freed (fleet slabs are process-lifetime). The
        // wrapper's drop will try a pool free on a VMM pointer, which the
        // context records and ignores — same story as the mega slab.
        let carve_bf16 =
            |base: u64, len: usize| unsafe { ctx.stream.upgrade_device_ptr::<bf16>(base, len) };
        let carve_f32 =
            |base: u64, len: usize| unsafe { ctx.stream.upgrade_device_ptr::<f32>(base, len) };
        let stream = &ctx.stream;
        Ok(Self {
            cp_rank: 0,
            cp_size: world,
            segments: Vec::new(),
            peers: Vec::new(),
            normed_tail: carve_bf16(mine.normed_tail, K3_CONV_STATE * K3_HIDDEN),
            kda_m: carve_f32(mine.kda_m, K3_KDA_STATE),
            kda_d: carve_f32(mine.kda_d, K3_KDA_STATE),
            mla_latent_pub: carve_bf16(mine.mla_latent, seg_cap * K3_KV_LORA_RANK),
            mla_rope_pub: carve_bf16(mine.mla_rope, seg_cap * K3_QK_ROPE_HEAD_DIM),
            halo_normed: stream.alloc_zeros((K3_CONV_STATE + 1) * K3_HIDDEN)?,
            halo_partial: stream.alloc_zeros((K3_CONV_STATE + 1) * K3_ATTN_INNER)?,
            halo_xs: stream.alloc_zeros((K3_CONV_STATE + 1) * K3_ATTN_INNER)?,
            recv_m: stream
                .alloc_zeros(world * K3_KDA_STATE)
                .context("alloc K3 whale package arena")?,
            recv_d: stream.alloc_zeros(world * K3_KDA_STATE)?,
            merge_a: stream.alloc_zeros(K3_KDA_STATE)?,
            merge_b: stream.alloc_zeros(K3_KDA_STATE)?,
            upstream_kv_rows: stream
                .alloc_zeros(seg_cap.saturating_mul(world.saturating_sub(1)).max(1))
                .context("alloc K3 whale upstream index buffer")?,
            upstream_len: 0,
            seg_cap,
            publish_event: None,
            consume_event: None,
            whale_seq: 0,
            whale_window: 0,
            whale_gang: Vec::new(),
            sync: K3CpSyncHandle::Fleet(gang),
        })
    }

    /// Arm the upstream persist: `indices` are the owner's pool write indices
    /// for global positions `0..indices.len()` (pages already mapped).
    pub(crate) fn set_upstream_rows(&mut self, ctx: &DeviceContext, indices: &[i32]) -> Result<()> {
        ensure!(
            indices.len() <= self.upstream_kv_rows.len(),
            "K3 CP upstream span of {} rows exceeds the {} buffer",
            indices.len(),
            self.upstream_kv_rows.len()
        );
        if !indices.is_empty() {
            let mut window = self.upstream_kv_rows.slice_mut(0..indices.len());
            ctx.stream
                .memcpy_htod(indices, &mut window)
                .map_err(|error| anyhow::anyhow!("K3 CP upstream index feed failed: {error}"))?;
        }
        self.upstream_len = indices.len();
        Ok(())
    }

    pub(crate) fn clear_upstream_rows(&mut self) {
        self.upstream_len = 0;
    }

    /// Enter one superstep: take this superstep's CP rank, adopt the split,
    /// and swap pointer tables with the gang. Collective. In-process only —
    /// a fleet scratch arms with [`K3CpScratch::arm_fleet`].
    pub(crate) fn arm(
        &mut self,
        ctx: &DeviceContext,
        cp_rank: usize,
        segments: Vec<(usize, usize)>,
    ) -> Result<()> {
        let K3CpSyncHandle::Local(group) = self.sync.clone() else {
            anyhow::bail!("a fleet K3 CP scratch cannot arm an in-process superstep");
        };
        ensure!(
            cp_rank < self.cp_size,
            "K3 CP rank {cp_rank} of {}",
            self.cp_size
        );
        self.cp_rank = cp_rank;
        ensure!(
            segments.len() == self.cp_size,
            "K3 CP split of {} segments for a {}-rank gang",
            segments.len(),
            self.cp_size
        );
        ensure!(
            segments[self.cp_rank].1 <= self.seg_cap,
            "K3 CP segment of {} tokens exceeds the {}-row publish buffers",
            segments[self.cp_rank].1,
            self.seg_cap
        );
        self.segments = segments;
        let ptr = |slice: &CudaSlice<f32>| {
            let (p, _guard) = slice.device_ptr(&ctx.stream);
            p
        };
        let ptr_bf = |slice: &CudaSlice<bf16>| {
            let (p, _guard) = slice.device_ptr(&ctx.stream);
            p
        };
        let mine = K3CpPeerPtrs {
            normed_tail: ptr_bf(&self.normed_tail),
            kda_m: ptr(&self.kda_m),
            kda_d: ptr(&self.kda_d),
            mla_latent: ptr_bf(&self.mla_latent_pub),
            mla_rope: ptr_bf(&self.mla_rope_pub),
            publish_event: self
                .publish_event
                .as_ref()
                .expect("a local scratch owns its events")
                .cu_event() as u64,
            consume_event: self
                .consume_event
                .as_ref()
                .expect("a local scratch owns its events")
                .cu_event() as u64,
        };
        self.peers = group.publish_and_snapshot(self.cp_rank, mine)?;
        Ok(())
    }

    /// Enter one whale superstep: adopt the committed descriptor's gang, CP
    /// rank and split, and point the peer table at the pre-imported fabric
    /// slabs. Not collective on the host — the whale rendezvous already
    /// committed every member to this exact superstep, and the pointer table
    /// is static after the import — so there is nothing to wait for.
    pub(crate) fn arm_fleet(
        &mut self,
        seq: u64,
        cp_rank: usize,
        gang_ranks: &[usize],
        segments: Vec<(usize, usize)>,
    ) -> Result<()> {
        let K3CpSyncHandle::Fleet(gang) = self.sync.clone() else {
            anyhow::bail!("an in-process K3 CP scratch cannot arm a whale superstep");
        };
        let width = gang_ranks.len();
        ensure!(
            (2..=gang.world()).contains(&width),
            "K3 whale gang of {width} ranks in a {}-rank world",
            gang.world()
        );
        ensure!(cp_rank < width, "K3 whale CP rank {cp_rank} of {width}");
        ensure!(
            gang_ranks[cp_rank] == gang.rank(),
            "K3 whale gang seats rank {} at CP position {cp_rank}, but this executor is rank {}",
            gang_ranks[cp_rank],
            gang.rank()
        );
        ensure!(
            segments.len() == width,
            "K3 whale split of {} segments for a {width}-rank gang",
            segments.len()
        );
        ensure!(
            segments[cp_rank].1 <= self.seg_cap,
            "K3 whale segment of {} tokens exceeds the {}-row publish buffers",
            segments[cp_rank].1,
            self.seg_cap
        );
        self.cp_rank = cp_rank;
        self.cp_size = width;
        self.segments = segments;
        self.peers = gang_ranks
            .iter()
            .map(|&member| gang.peer_ptrs(member))
            .collect::<Result<_>>()?;
        self.whale_seq = seq;
        self.whale_window = 0;
        self.whale_gang = gang_ranks.to_vec();
        Ok(())
    }

    /// Snapshot what one exchange window needs — plain copied data, so the
    /// `consume` closure is free to capture the scratch. On the fleet
    /// substrate this also claims the window's doorbell value, so every
    /// snapshot must be spent on exactly one exchange.
    pub(crate) fn window_sync(&mut self, kind: K3CpWindowKind) -> Result<K3CpWindowSync> {
        Ok(match &self.sync {
            K3CpSyncHandle::Local(_) => K3CpWindowSync::Local {
                cp_rank: self.cp_rank,
                kind,
                events: self
                    .peers
                    .iter()
                    .map(|peer| (peer.publish_event, peer.consume_event))
                    .collect(),
            },
            K3CpSyncHandle::Fleet(_) => {
                let value = K3WhaleGang::window_value(self.whale_seq, self.whale_window)?;
                self.whale_window += 1;
                K3CpWindowSync::Fleet {
                    cp_rank: self.cp_rank,
                    kind,
                    gang: self.whale_gang.clone(),
                    value,
                }
            }
        })
    }

    /// Fold the received upstream packages into this rank's true KDA input
    /// state for the current layer: `S = D_0; for j in 1..cp_rank: S = S·M_j
    /// + D_j`, per head in fp32. Returns the merged `[heads, 128, 128]` state.
    pub(crate) fn merge_upstream(&mut self, ctx: &DeviceContext) -> Result<&CudaSlice<f32>> {
        ensure!(self.cp_rank >= 1, "K3 CP rank 0 has nothing to merge");
        let d = K3_KDA_HEAD_DIM;
        let per_head = d * d;
        copy_rows(ctx, &self.recv_d, 0, &mut self.merge_a, 0, 1, K3_KDA_STATE)?;
        for j in 1..self.cp_rank {
            copy_rows(ctx, &self.recv_d, j, &mut self.merge_b, 0, 1, K3_KDA_STATE)?;
            let m_j = self.recv_m.slice(j * K3_KDA_STATE..(j + 1) * K3_KDA_STATE);
            // The state slab is `[head, v, k]` and the KDA transition acts on
            // the k axis, so the segment map applies from the right: row-major
            // per-head `S' = S·M + D` through the column-major swap (first
            // operand = M, second = S), C pre-filled with D and accumulated.
            gemm_strided_batched_f32(
                ctx,
                false,
                false,
                d,
                d,
                d,
                &m_j,
                d,
                per_head,
                &self.merge_a,
                d,
                per_head,
                true,
                &mut self.merge_b,
                d,
                per_head,
                K3_KDA_HEADS,
            )?;
            std::mem::swap(&mut self.merge_a, &mut self.merge_b);
        }
        Ok(&self.merge_a)
    }
}

/// Stream-ordered device copy from a peer's published buffer (raw base
/// pointer) into a local slice. Peer access must already be open — the
/// MegaMoE scratch construction opened it before any CP buffer existed.
pub(crate) fn k3_cp_copy_in<T: cudarc::driver::DeviceRepr>(
    ctx: &DeviceContext,
    src_base: u64,
    src_elem_offset: usize,
    dst: &mut CudaSlice<T>,
    dst_elem_offset: usize,
    elems: usize,
) -> Result<()> {
    if elems == 0 {
        return Ok(());
    }
    ensure!(
        dst_elem_offset + elems <= dst.len(),
        "K3 CP peer copy of {elems} elements at {dst_elem_offset} overflows the {} destination",
        dst.len()
    );
    let element = size_of::<T>();
    let (dst_ptr, _guard) = dst.device_ptr_mut(&ctx.stream);
    // SAFETY: the destination range was bounds-checked; the source is a live
    // peer publish buffer inside an exchange window — this copy runs after
    // the enqueued wait on its owner's publish event, and the owner's next
    // overwrite waits on this rank's consume event recorded behind it.
    unsafe {
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            dst_ptr + (dst_elem_offset * element) as u64,
            src_base + (src_elem_offset * element) as u64,
            elems * element,
            pegainfer_kernels::tensor::active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|error| anyhow::anyhow!("K3 CP peer copy failed: {error}"))
}
