//! Context-parallel (CP) prefill: one long prompt split contiguously across
//! the expert-parallel ranks, each rank walking its own segment through the
//! same chunk step, with three per-layer exchanges stitching the segments
//! back into one sequence (see `docs/models/k3/cp-lane-design.md`):
//!
//! * **KDA (KCP)**: the recurrence's per-token update is affine in the state
//!   (`S' = A·S + b` with `A`, `b` functions of the inputs alone), so a whole
//!   segment collapses to `S_out = M_seg·S_in + D_seg`. Each rank exports its
//!   `(M, D)` package by running the vendored FlashKDA forward twice with
//!   doctored operands — `v = 0` from an identity state yields `M`, real `v`
//!   from a zero state yields `D` — then every rank folds its upstream
//!   packages into its true input state with per-head fp32 GEMMs and runs the
//!   real forward once. Rank 0's real forward *is* its package (`D`, with a
//!   known zero input), and the last rank exports nothing.
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
//! The exchange transport is M0-grade: in-process ranks, plain peer-access
//! device-to-device copies, and a host [`Barrier`] bracketing each window
//! (sync own stream → barrier → copy from peers → sync → barrier). Roughly
//! `2 × 69 + 24` windows per superstep — measurable host overhead that an M1
//! pass moves onto events, but free of ordering hazards.

use std::sync::Arc;
use std::sync::Barrier;
use std::sync::Mutex;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;
use pegainfer_kernels::ops::K3_KDA_HEAD_DIM;
use pegainfer_kernels::ops::K3_KDA_HEADS;
use pegainfer_kernels::ops::gemm_strided_batched_f32;
use pegainfer_kernels::tensor::DeviceContext;

use super::buffers::K3_CONV_STATE;
use super::buffers::K3_KDA_STATE;
use super::buffers::copy_rows;
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
}

/// The in-process CP gang: a barrier and a published pointer table shared by
/// every rank of one context-parallel prefill. One group serves any number of
/// consecutive supersteps; the pointer table is republished at each entry.
pub struct K3CpGroup {
    cp_size: usize,
    barrier: Barrier,
    ptrs: Mutex<Vec<Option<K3CpPeerPtrs>>>,
}

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
        // Hold everyone until the slowest reader has its snapshot, so a rank
        // entering the NEXT superstep cannot republish underneath it.
        self.barrier.wait();
        Ok(snapshot)
    }

    /// One exchange window: everything this rank published is on-device
    /// before its peers read, and everything it read is on-device before its
    /// peers overwrite. `consume` issues this rank's reads of peer buffers on
    /// its own stream. Collective — every rank must reach every window.
    pub(crate) fn exchange(
        &self,
        ctx: &DeviceContext,
        consume: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        sync_stream(ctx)?;
        self.barrier.wait();
        consume()?;
        sync_stream(ctx)?;
        self.barrier.wait();
        Ok(())
    }
}

fn sync_stream(ctx: &DeviceContext) -> Result<()> {
    ctx.stream
        .synchronize()
        .map_err(|error| anyhow::anyhow!("K3 CP stream sync failed: {error}"))
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
/// once per executor (after the MegaMoE peer-access grants, so the pool
/// covers every buffer) and re-armed per superstep.
pub(crate) struct K3CpScratch {
    pub(crate) group: Arc<K3CpGroup>,
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
    /// All-zero `v` operand for the transition (`M`) forward.
    pub(crate) zero_v: CudaSlice<bf16>,
    /// All-zero input state for the `D` forward.
    pub(crate) zero_state: CudaSlice<f32>,
    /// Per-head identity input state for the `M` forward.
    pub(crate) identity: CudaSlice<f32>,
    /// Received packages, `[cp_size, K3_KDA_STATE]` each.
    pub(crate) recv_m: CudaSlice<f32>,
    pub(crate) recv_d: CudaSlice<f32>,
    merge_a: CudaSlice<f32>,
    merge_b: CudaSlice<f32>,
    seg_cap: usize,
}

impl K3CpScratch {
    pub(crate) fn new(
        ctx: &DeviceContext,
        group: Arc<K3CpGroup>,
        cp_rank: usize,
        seg_cap: usize,
    ) -> Result<Self> {
        let cp_size = group.cp_size();
        ensure!(cp_rank < cp_size, "K3 CP rank {cp_rank} of {cp_size}");
        let stream = &ctx.stream;
        let d = K3_KDA_HEAD_DIM;
        let mut identity_host = vec![0f32; K3_KDA_STATE];
        for head in 0..K3_KDA_HEADS {
            for i in 0..d {
                identity_host[head * d * d + i * d + i] = 1.0;
            }
        }
        Ok(Self {
            cp_rank,
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
            zero_v: stream
                .alloc_zeros(seg_cap * K3_ATTN_INNER)
                .context("alloc K3 CP zero-v operand")?,
            zero_state: stream.alloc_zeros(K3_KDA_STATE)?,
            identity: stream.clone_htod(&identity_host)?,
            recv_m: stream
                .alloc_zeros(cp_size * K3_KDA_STATE)
                .context("alloc K3 CP package arena")?,
            recv_d: stream.alloc_zeros(cp_size * K3_KDA_STATE)?,
            merge_a: stream.alloc_zeros(K3_KDA_STATE)?,
            merge_b: stream.alloc_zeros(K3_KDA_STATE)?,
            seg_cap,
            group,
        })
    }

    pub(crate) fn seg_cap(&self) -> usize {
        self.seg_cap
    }

    /// Enter one superstep: adopt the split and swap pointer tables with the
    /// gang. Collective.
    pub(crate) fn arm(&mut self, ctx: &DeviceContext, segments: Vec<(usize, usize)>) -> Result<()> {
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
        };
        self.peers = self.group.publish_and_snapshot(self.cp_rank, mine)?;
        Ok(())
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
    // peer publish buffer inside an exchange window (its owner is parked at
    // the window's closing barrier until this copy's stream sync).
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
