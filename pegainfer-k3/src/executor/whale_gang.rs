//! The fleet whale gang's data plane: the cross-process counterpart of the
//! in-process CP gang in [`super::cp`].
//!
//! An in-process gang exchanges segment publications through peer-access
//! device copies ordered by CUDA events. Neither survives a process boundary:
//! peers in another process cannot dereference a pool pointer, and a CUDA
//! event handle is meaningless outside the context that created it. The fleet
//! replaces both with the same substrate the MegaMoE kernel already runs on —
//! NVLink-fabric memory (NVL72 + IMEX):
//!
//! * **Buffers**: every rank's whole CP publish surface (conv halo tail, KDA
//!   `(M, D)` packages, MLA latent/rope rows) lives in one
//!   `CU_MEM_HANDLE_TYPE_FABRIC` slab, allocated at startup and imported once
//!   by every process in the world. After that import, a peer's publication
//!   is an ordinary device pointer — [`super::cp::k3_cp_copy_in`] works
//!   unchanged.
//! * **Ordering**: CUDA events give way to **doorbells** — `u64` flag arrays
//!   in the same slabs, rung by a one-thread SM kernel
//!   (`k3_whale_doorbell_ring`) and waited on with `cuStreamWaitValue64`.
//!   Writes go to the *remote* rank's slab and waits watch the rank's *own*
//!   slab, so every wait is stream-memops on local memory and every remote
//!   touch is a plain NVLink store — the direction the fabric is built for.
//!   The split is forced, not stylistic: the stream memops engine rejects
//!   fabric-imported mappings outright (`CUDA_ERROR_INVALID_VALUE` on GB300,
//!   two-process probe), while SM stores go through them on every MegaMoE
//!   step. The four-beat window protocol is the in-process one verbatim
//!   (publish, await publications, consume, await consumers); only the
//!   primitive changed.
//!
//! Doorbell values must be agreed without any host negotiation, across gangs
//! that rotate membership whale to whale. They are derived entirely from the
//! whale rendezvous ([`crate::scheduler::whale`]): window `w` of whale `seq`
//! carries the value `(seq + 1) * K3_WHALE_WINDOW_STRIDE + w + 1`. The
//! sequencer hands out `seq` strictly increasing and every superstep runs the
//! same fixed window schedule, so the values a rank writes are strictly
//! monotonic even across whales it sits out — which is exactly what the
//! `GEQ` wait condition needs. Flag arrays are indexed by *global* rank, so
//! rotation never aliases a slot.

use std::sync::Arc;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::sys as cu_sys;
use half::bf16;
use pegainfer_kernels::ops::K3_MEGA_FABRIC_HANDLE_BYTES;
use pegainfer_kernels::ops::k3_chunk_bucket;
use pegainfer_kernels::ops::k3_mega_fabric_slab_alloc;
use pegainfer_kernels::ops::k3_mega_fabric_slab_import;
use pegainfer_kernels::ops::k3_mega_fabric_supported;
use pegainfer_kernels::ops::k3_whale_doorbell_ring;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::active_cu_stream;

use super::buffers::K3_CONV_STATE;
use super::buffers::K3_KDA_STATE;
use super::cp::K3CpPeerPtrs;
use super::cp::K3CpScratch;
use super::cp::K3CpSyncHandle;
use super::cp::K3CpWindowKind;
use crate::config::K3_HIDDEN;
use crate::config::K3_KV_LORA_RANK;
use crate::config::K3_QK_ROPE_HEAD_DIM;

/// Doorbell values per whale: window `w` of whale `seq` rings
/// `(seq + 1) * STRIDE + w + 1`. The stride bounds the windows one superstep
/// may run (the real schedule is ~`2 x layers + mix layers`, far below it),
/// and `window_value` asserts the bound instead of silently aliasing the
/// next whale's values.
pub(crate) const K3_WHALE_WINDOW_STRIDE: u64 = 4096;

/// Byte offsets of one rank's regions inside its whale slab. Every rank in
/// the world derives the identical layout from the same `(world, seg_cap)`,
/// which is what lets a peer address a remote region with nothing but the
/// imported base.
#[derive(Clone, Copy, Debug)]
pub(crate) struct K3WhaleSlabLayout {
    /// `[world] u64` — inbox: rank `r` announces its window publication here.
    publish_inbox: usize,
    /// `[world] u64` — inbox: rank `r` acknowledges consuming my publication.
    consume_inbox: usize,
    normed_tail: usize,
    kda_m: usize,
    kda_d: usize,
    mla_latent: usize,
    mla_rope: usize,
    pub(crate) num_bytes: usize,
}

/// Every region starts on a fresh cache line; the doorbell flags get one
/// line per slot so a remote store never false-shares with a neighbour.
const K3_WHALE_ALIGN: usize = 128;

fn aligned(offset: usize) -> usize {
    offset.next_multiple_of(K3_WHALE_ALIGN)
}

impl K3WhaleSlabLayout {
    pub(crate) fn new(world: usize, seg_cap: usize) -> Self {
        let mut offset = 0usize;
        let mut region = |bytes: usize| -> usize {
            let base = aligned(offset);
            offset = base + bytes;
            base
        };
        let publish_inbox = region(world * K3_WHALE_ALIGN);
        let consume_inbox = region(world * K3_WHALE_ALIGN);
        let normed_tail = region(K3_CONV_STATE * K3_HIDDEN * size_of::<bf16>());
        let kda_m = region(K3_KDA_STATE * size_of::<f32>());
        let kda_d = region(K3_KDA_STATE * size_of::<f32>());
        let mla_latent = region(seg_cap * K3_KV_LORA_RANK * size_of::<bf16>());
        let mla_rope = region(seg_cap * K3_QK_ROPE_HEAD_DIM * size_of::<bf16>());
        Self {
            publish_inbox,
            consume_inbox,
            normed_tail,
            kda_m,
            kda_d,
            mla_latent,
            mla_rope,
            num_bytes: aligned(offset),
        }
    }

    fn publish_flag(&self, base: u64, from: usize) -> u64 {
        base + (self.publish_inbox + from * K3_WHALE_ALIGN) as u64
    }

    fn consume_flag(&self, base: u64, from: usize) -> u64 {
        base + (self.consume_inbox + from * K3_WHALE_ALIGN) as u64
    }
}

/// One rank's slab identity on the wire: what the whale hub's startup
/// exchange moves between processes.
#[derive(Clone, Copy, Debug)]
pub struct K3WhaleSlabWire {
    pub handle: [u8; K3_MEGA_FABRIC_HANDLE_BYTES],
    pub num_bytes: usize,
}

/// Allocate this rank's whale slab on `device_ordinal`: zeroed, mapped for
/// every local device, fabric-exportable. Returns the local base pointer and
/// the wire identity peers import.
pub(crate) fn k3_whale_slab_alloc(
    device_ordinal: usize,
    layout: &K3WhaleSlabLayout,
) -> Result<(u64, K3WhaleSlabWire)> {
    ensure!(
        k3_mega_fabric_supported(device_ordinal).unwrap_or(false),
        "K3 whale rank on device {device_ordinal} cannot allocate NVLink-fabric memory; a \
         cross-machine whale gang needs the IMEX daemon and a fabric-capable driver"
    );
    let (ptr, handle) = k3_mega_fabric_slab_alloc(device_ordinal, layout.num_bytes)
        .context("alloc K3 whale CP fabric slab")?;
    Ok((
        u64::try_from(ptr).context("K3 whale slab base pointer")?,
        K3WhaleSlabWire {
            handle,
            num_bytes: layout.num_bytes,
        },
    ))
}

/// The world's whale data plane as seen from one rank: every rank's slab
/// mapped into this process, plus the shared layout. Built once after the
/// hub's slab exchange; process-lifetime, like the mega slabs (a whale fleet
/// dies together).
pub(crate) struct K3WhaleGang {
    /// This executor's global rank — where peers ring my doorbells.
    rank: usize,
    layout: K3WhaleSlabLayout,
    /// Per world rank: the slab base as addressed from this process (my own
    /// entry is the local allocation).
    bases: Vec<u64>,
}

/// Map the whole world's whale slabs into this process, once: `local` maps
/// the ranks this process hosts to their locally allocated bases (already
/// mapped for every local device by their allocation); every other rank's
/// handle is imported through `device_ordinal`, which likewise maps it for
/// all local devices. Every local executor then builds its own
/// [`K3WhaleGang`] over one clone of the returned table.
pub(crate) fn k3_whale_import_world(
    world_slabs: &[K3WhaleSlabWire],
    local: &[(usize, u64)],
    layout: &K3WhaleSlabLayout,
    device_ordinal: usize,
) -> Result<Vec<u64>> {
    let world = world_slabs.len();
    let mut bases = vec![0u64; world];
    let mut have = vec![false; world];
    for &(local_rank, base) in local {
        ensure!(
            local_rank < world,
            "K3 whale local rank {local_rank} outside the {world} world"
        );
        bases[local_rank] = base;
        have[local_rank] = true;
    }
    for (peer, wire) in world_slabs.iter().enumerate() {
        if have[peer] {
            continue;
        }
        ensure!(
            wire.num_bytes == layout.num_bytes,
            "K3 whale rank {peer} published a {}-byte slab, expected {} — the fleet disagrees on \
             the slab layout",
            wire.num_bytes,
            layout.num_bytes
        );
        let ptr = k3_mega_fabric_slab_import(&wire.handle, wire.num_bytes, device_ordinal)
            .with_context(|| format!("import K3 whale rank {peer}'s slab"))?;
        bases[peer] = u64::try_from(ptr).context("K3 whale imported base")?;
    }
    Ok(bases)
}

impl K3WhaleGang {
    /// One rank's view over the process-wide base table from
    /// [`k3_whale_import_world`].
    pub(crate) fn new(bases: Vec<u64>, rank: usize, layout: K3WhaleSlabLayout) -> Result<Self> {
        ensure!(
            rank < bases.len(),
            "K3 whale rank {rank} outside the {}-slab table",
            bases.len()
        );
        ensure!(
            bases.iter().all(|&base| base != 0),
            "K3 whale slab table has an unmapped rank"
        );
        Ok(Self {
            rank,
            layout,
            bases,
        })
    }

    pub(crate) fn world(&self) -> usize {
        self.bases.len()
    }

    pub(crate) fn rank(&self) -> usize {
        self.rank
    }

    /// This rank's own slab base — where its publish buffers are carved.
    fn my_base(&self) -> u64 {
        self.bases[self.rank]
    }

    /// The publish-surface pointers of `rank`'s slab, as addressed from this
    /// process. The event fields stay zero: fleet ordering runs on
    /// doorbells, never on events.
    pub(crate) fn peer_ptrs(&self, rank: usize) -> Result<K3CpPeerPtrs> {
        ensure!(
            rank < self.bases.len(),
            "K3 whale peer rank {rank} outside the {} world",
            self.bases.len()
        );
        let base = self.bases[rank];
        Ok(K3CpPeerPtrs {
            normed_tail: base + self.layout.normed_tail as u64,
            kda_m: base + self.layout.kda_m as u64,
            kda_d: base + self.layout.kda_d as u64,
            mla_latent: base + self.layout.mla_latent as u64,
            mla_rope: base + self.layout.mla_rope as u64,
            publish_event: 0,
            consume_event: 0,
        })
    }

    /// The doorbell value window `window` of whale `seq` rings.
    pub(crate) fn window_value(seq: u64, window: u64) -> Result<u64> {
        ensure!(
            window + 1 < K3_WHALE_WINDOW_STRIDE,
            "K3 whale superstep ran {window} exchange windows, past the {K3_WHALE_WINDOW_STRIDE} \
             doorbell stride"
        );
        Ok((seq + 1) * K3_WHALE_WINDOW_STRIDE + window + 1)
    }

    /// One exchange window over the fabric — the four-beat protocol of
    /// [`super::cp::K3CpGroup::exchange`] with doorbells for events. `gang`
    /// is the whale's global ranks in CP order and `cp_rank` this rank's
    /// position in it (both validated when the superstep armed); `consume`
    /// issues this rank's reads of peer buffers on its own stream. Entirely
    /// stream-ordered: the host enqueues and returns, and every wait is a
    /// `cuStreamWaitValue64` on this rank's own slab.
    pub(crate) fn exchange(
        &self,
        ctx: &DeviceContext,
        cp_rank: usize,
        kind: K3CpWindowKind,
        gang: &[usize],
        doorbell: u64,
        consume: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        let cp_size = gang.len();
        let my_base = self.my_base();
        // Beat 1: announce my publication to every rank that reads it, in one
        // ring. Stream order puts those bytes complete before the ring kernel
        // launches.
        let publish_flags: Vec<u64> = kind
            .read_by(cp_rank, cp_size)
            .map(|reader| {
                self.layout
                    .publish_flag(self.bases[gang[reader]], self.rank)
            })
            .collect();
        if !publish_flags.is_empty() {
            k3_whale_doorbell_ring(&publish_flags, doorbell, active_cu_stream(ctx))
                .context("K3 whale publish doorbell ring")?;
        }
        // Beat 2: wait for every publication I read this window.
        for source in kind.reads_from(cp_rank) {
            let flag = self.layout.publish_flag(my_base, gang[source]);
            stream_wait_value(ctx, flag, doorbell).context("K3 whale publish doorbell wait")?;
        }
        // Beat 3: enqueue my reads, then acknowledge them to their owners.
        consume()?;
        let consume_flags: Vec<u64> = kind
            .reads_from(cp_rank)
            .map(|source| {
                self.layout
                    .consume_flag(self.bases[gang[source]], self.rank)
            })
            .collect();
        if !consume_flags.is_empty() {
            k3_whale_doorbell_ring(&consume_flags, doorbell, active_cu_stream(ctx))
                .context("K3 whale consume doorbell ring")?;
        }
        // Beat 4: wait for my readers, so my next window's publish writes
        // cannot overwrite what a peer is still reading.
        for reader in kind.read_by(cp_rank, cp_size) {
            let flag = self.layout.consume_flag(my_base, gang[reader]);
            stream_wait_value(ctx, flag, doorbell).context("K3 whale consume doorbell wait")?;
        }
        Ok(())
    }
}

/// The whale lane's startup dance on one executor, in call order: allocate
/// the slab, exchange handles through the hub (the caller's job), import the
/// world, install the gang. Serving needs nothing further — a committed
/// whale re-arms the scratch per superstep.
impl super::K3Executor {
    /// Allocate this rank's whale slab — its fleet CP publish surface plus
    /// doorbells — before the fleet's handle exchange. `world` is the whale
    /// world size; every rank must derive the identical layout, which the
    /// import checks byte-for-byte. Once per executor. Returns the local base
    /// (for the process-wide import table) and the wire identity peers import.
    pub fn arm_whale_slab(&mut self, world: usize) -> Result<(u64, K3WhaleSlabWire)> {
        ensure!(
            self.whale_slab.is_none(),
            "K3 rank {} armed its whale slab twice",
            self.model.rank
        );
        // Arming runs on the engine-launch thread, which under parallel
        // (staged) loading has never held a CUDA context — the VMM calls
        // behind the slab alloc need one current.
        self.bind_thread()?;
        let layout = K3WhaleSlabLayout::new(world, k3_chunk_bucket(self.chunk_tokens)?);
        let (base, wire) = k3_whale_slab_alloc(self.ctx.device_ordinal, &layout)?;
        self.whale_slab = Some((base, layout));
        Ok((base, wire))
    }

    /// Map the exchanged world table into this process — once, on any one
    /// local executor; every local rank then installs one clone of the
    /// result. `local` maps this process's ranks to their slab bases.
    pub fn import_whale_world(
        &self,
        table: &[K3WhaleSlabWire],
        local: &[(usize, u64)],
    ) -> Result<Vec<u64>> {
        let (_, layout) = self
            .whale_slab
            .context("K3 whale import before arm_whale_slab")?;
        k3_whale_import_world(table, local, &layout, self.ctx.device_ordinal)
    }

    /// Install the imported world table ([`k3_whale_import_world`]'s result)
    /// and stand the whale data plane up for this rank.
    pub fn install_whale_gang(&mut self, bases: Vec<u64>) -> Result<()> {
        let (base, layout) = self
            .whale_slab
            .context("K3 whale gang install before arm_whale_slab")?;
        let rank = self.model.rank;
        ensure!(
            bases.get(rank).copied() == Some(base),
            "K3 whale slab table does not carry rank {rank}'s own base"
        );
        self.whale_gang = Some(Arc::new(K3WhaleGang::new(bases, rank, layout)?));
        Ok(())
    }

    /// Build (or re-arm) the fleet CP working set for this whale superstep.
    /// Unlike the in-process path this opens no peer access — the fabric
    /// mappings carry their own grants — and arms without any collective.
    pub(crate) fn ensure_whale_scratch(
        &mut self,
        seq: u64,
        cp_rank: usize,
        gang_ranks: &[usize],
        segments: Vec<(usize, usize)>,
    ) -> Result<()> {
        let gang = self
            .whale_gang
            .clone()
            .context("K3 whale prefill before the gang was installed")?;
        if let Some(scratch) = self.cp_scratch.as_ref() {
            ensure!(
                matches!(&scratch.sync, K3CpSyncHandle::Fleet(mine) if Arc::ptr_eq(mine, &gang)),
                "K3 CP scratch was built for a different substrate than the whale gang"
            );
        } else {
            self.cp_scratch = Some(Box::new(K3CpScratch::new_fleet(
                &self.ctx,
                gang,
                k3_chunk_bucket(self.chunk_tokens)?,
            )?));
            // Local arenas live and zeroed before the superstep touches them.
            self.gpu.sync()?;
        }
        self.cp_scratch
            .as_mut()
            .expect("built above")
            .arm_fleet(seq, cp_rank, gang_ranks, segments)
    }
}

fn stream_wait_value(ctx: &DeviceContext, addr: u64, value: u64) -> Result<()> {
    // SAFETY: `addr` is inside this rank's own live fabric slab (locally
    // allocated — the memops engine accepts local fabric memory, unlike
    // imported mappings); GEQ is what monotonic doorbell values need — a rank
    // that sat out intermediate whales left its flags behind, and any later
    // ring satisfies every earlier wait.
    unsafe {
        cu_sys::cuStreamWaitValue64_v2(
            active_cu_stream(ctx),
            addr,
            value,
            cu_sys::CUstreamWaitValue_flags::CU_STREAM_WAIT_VALUE_GEQ as u32,
        )
    }
    .result()
    .map_err(|error| anyhow::anyhow!("cuStreamWaitValue64 failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_regions_are_disjoint_aligned_and_world_scaled() {
        let world = 16;
        let seg_cap = 16896;
        let layout = K3WhaleSlabLayout::new(world, seg_cap);
        let regions = [
            (layout.publish_inbox, world * K3_WHALE_ALIGN),
            (layout.consume_inbox, world * K3_WHALE_ALIGN),
            (
                layout.normed_tail,
                K3_CONV_STATE * K3_HIDDEN * size_of::<bf16>(),
            ),
            (layout.kda_m, K3_KDA_STATE * size_of::<f32>()),
            (layout.kda_d, K3_KDA_STATE * size_of::<f32>()),
            (
                layout.mla_latent,
                seg_cap * K3_KV_LORA_RANK * size_of::<bf16>(),
            ),
            (
                layout.mla_rope,
                seg_cap * K3_QK_ROPE_HEAD_DIM * size_of::<bf16>(),
            ),
        ];
        for (offset, _) in regions {
            assert_eq!(offset % K3_WHALE_ALIGN, 0, "unaligned region at {offset}");
        }
        let mut sorted = regions;
        sorted.sort_by_key(|&(offset, _)| offset);
        for pair in sorted.windows(2) {
            assert!(
                pair[0].0 + pair[0].1 <= pair[1].0,
                "regions overlap: {pair:?}"
            );
        }
        let (last_offset, last_bytes) = sorted[sorted.len() - 1];
        assert!(last_offset + last_bytes <= layout.num_bytes);
    }

    #[test]
    fn doorbell_values_are_strictly_monotonic_across_whales_and_windows() {
        let mut previous = 0u64;
        for seq in [0u64, 1, 2, 7, 8] {
            for window in 0..4u64 {
                let value = K3WhaleGang::window_value(seq, window).unwrap();
                assert!(value > previous, "seq {seq} window {window}");
                previous = value;
            }
        }
        assert!(K3WhaleGang::window_value(0, K3_WHALE_WINDOW_STRIDE).is_err());
    }

    /// GPU: ring a doorbell on a freshly allocated whale slab through the
    /// exact production call chain (fabric alloc -> `k3_whale_doorbell_ring`
    /// -> `stream_wait_value`). Needs one fabric-capable GPU. Cross-process
    /// import coverage is the fleet smoke's job — a single process cannot
    /// import its own handle.
    #[test]
    #[ignore = "needs a fabric-capable GPU"]
    fn doorbell_rings_on_a_real_fabric_slab() {
        use pegainfer_kernels::tensor::DeviceContext;
        let ctx = DeviceContext::new_with_device(0).expect("device 0");
        let layout = K3WhaleSlabLayout::new(8, 16896);
        let (base, _wire) = k3_whale_slab_alloc(0, &layout).expect("fabric slab");
        let value = K3WhaleGang::window_value(0, 0).unwrap();
        let flags: Vec<u64> = (0..8).map(|from| layout.publish_flag(base, from)).collect();
        k3_whale_doorbell_ring(&flags, value, active_cu_stream(&ctx)).expect("doorbell ring");
        for (from, &flag) in flags.iter().enumerate() {
            stream_wait_value(&ctx, flag, value)
                .unwrap_or_else(|e| panic!("publish wait from={from}: {e:#}"));
        }
        ctx.stream.synchronize().expect("stream sync");
    }

    #[test]
    fn flag_slots_are_aligned_and_one_line_apart() {
        let layout = K3WhaleSlabLayout::new(4, 4224);
        let base = 1 << 20;
        let flags: Vec<u64> = (0..4)
            .flat_map(|from| {
                [
                    layout.publish_flag(base, from),
                    layout.consume_flag(base, from),
                ]
            })
            .collect();
        for &flag in &flags {
            assert_eq!(flag % 8, 0, "a doorbell store at {flag:#x} would tear");
        }
        let mut sorted = flags;
        sorted.sort_unstable();
        for pair in sorted.windows(2) {
            assert!(
                pair[1] - pair[0] >= K3_WHALE_ALIGN as u64,
                "flag slots share a cache line: {pair:x?}"
            );
        }
    }
}
