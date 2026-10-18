//! The host-tier seam: what [`crate::KvStore`] needs from the storage layer
//! below it, expressed as a dyn-compatible trait, plus the sole backend
//! behind the seam — the pegaflow one ([`crate::PegaflowHost`]).
//!
//! Types cross the seam concretely (`QueryLeaseId`, `KvBlockGuard`): with
//! one crate-private implementation, type erasure would be defending against
//! an alternative backend that does not exist. The seam's value is
//! organizational — orchestration above (`store.rs`), engine bridging
//! below — and one engine protocol deliberately passes through:
//! [`TierQuery::Loading`], "a deeper tier is fetching; ask again later". The
//! store owns that re-query loop for now; having the engine signal readiness
//! instead of being polled is on the table (design doc, open questions).

use std::future::Future;
use std::pin::Pin;

use pegaflow_core::QueryLeaseId;

pub(crate) type TierFuture<T> = Pin<Box<dyn Future<Output = T> + Send>>;

/// Leading blocks of a prefix the host tier has resident, pinned by `lease`
/// until [`HostTier::load`] consumes it or [`HostTier::release`] declines it.
///
/// One lease covers every replica device: pegaflow mints it with a consumer
/// budget of the instance's sealed `world_size`, and each per-device load
/// consumes one unit — the host pins drop when the last device's copy lands.
/// This is the same contract the vLLM MLA-TP connector runs in production.
pub(crate) struct LeasedBlocks {
    pub blocks: usize,
    pub(crate) lease: QueryLeaseId,
}

/// One query's outcome against the host tier.
pub(crate) enum TierQuery {
    /// Nothing resident and nothing in flight.
    Miss,
    /// A deeper tier (SSD / remote peer) is fetching; re-query after a pause.
    Loading,
    Hit(LeasedBlocks),
}

/// The storage layer below the store. All returned futures observe operations
/// already submitted to the tier's own runtime — dropping one detaches the
/// observer, never the I/O (the concrete engine's handle contract).
pub(crate) trait HostTier: Send + Sync {
    /// How long a prefix of `block_hashes` is host-resident. `req_id` scopes
    /// an in-flight deeper-tier fetch across re-queries. `wait_full` makes
    /// the fetch all-or-nothing: stay `Loading` until the entire missing
    /// prefix is resident instead of answering with a partial hit — for
    /// callers that cannot recompute a miss (the decode side of a
    /// disaggregated prefill/decode pair).
    fn query(
        &self,
        req_id: &str,
        block_hashes: Vec<Vec<u8>>,
        wait_full: bool,
    ) -> TierFuture<anyhow::Result<TierQuery>>;

    /// Copy the hit's blocks into the GPU pages `dst_page_ids` (one
    /// destination per hit block, across all registered layers). Consumes the
    /// lease whether it succeeds or fails.
    fn load(&self, hit: LeasedBlocks, dst_page_ids: Vec<i32>) -> TierFuture<anyhow::Result<()>>;

    /// Decline a hit without loading it, releasing the host-side pins now
    /// instead of waiting out the lease TTL.
    fn release(&self, hit: LeasedBlocks);

    /// Submit an async GPU→host save of sealed blocks. The returned future
    /// observes an already-submitted operation (same contract as the
    /// siblings); the source blocks' `guards` are dropped only once the D2H
    /// lands (the reuse contract — before that the blocks must not be
    /// overwritten).
    fn save(
        &self,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
        guards: Vec<KvBlockGuard>,
    ) -> TierFuture<anyhow::Result<()>>;

    /// Write-pipeline visibility barrier: once the returned future resolves,
    /// every save submitted *before this call* is query-visible from this
    /// tier (and, with P2P configured, registered with the MetaServer — the
    /// P/D "KV ready" signal). Dropping the future detaches the observer,
    /// never the barrier — the same contract as the sibling futures. This is
    /// deliberately a weaker barrier than persistence: spilling to an SSD
    /// tier is not covered (see [`crate::PegaflowHost::flush_all`]).
    fn flush(&self) -> TierFuture<anyhow::Result<()>>;
}

// ── The pegaflow-backed tier ────────────────────────────────────────────────

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Mutex;

use pegaflow_core::EngineError;
use pegaflow_core::LayerSave;
use pegaflow_core::PrefetchStatus;
use pegaflow_core::TransferMode;
use tokio::task::JoinHandle;

use crate::builder::ArenaSpec;
use crate::builder::OffloadRankSpec;
use crate::host::PegaflowHost;
use crate::pool::KvBlockGuard;

/// Single-executor topology: each store rank registers its arenas as its own
/// pegaflow instance (a model executor that shards one GPU's KV registers one
/// rank per shard of its own). Tensor-replicated mirrors join the SAME
/// instance as extra world members — pegaflow's replica contract, not a TP
/// split — so `world_size` is derived per registration, while TP stays 1.
const TP_RANK: usize = 0;
const PP_RANK: usize = 0;
const TP_SIZE: usize = 1;

/// The concrete pegaflow-backed [`HostTier`]: one pegaflow *instance* (the
/// registration of [`OffloadRankSpec`]'s arenas) over a shared
/// [`PegaflowHost`]. Constructed by
/// [`crate::KvStoreBuilder::rank_with_offload`].
pub(crate) struct PegaflowTier {
    host: Arc<PegaflowHost>,
    instance_id: String,
    device_id: i32,
    /// Tensor-replicated mirror devices; every load fans out to each of them
    /// (pegaflow replica contract: the primary saves, each device loads into
    /// its own copy).
    mirror_devices: Vec<i32>,
    /// Owned per-layer names; load borrows them as `&[&str]`.
    layer_names: Vec<String>,
    /// Handles of in-flight save tasks, drained by `flush`: a save task's
    /// completion means its batch is queued for the insert worker, so the
    /// pipeline flush that follows covers it — a racing `flush` can never
    /// report visibility for a save still mid-D2H. One lock keeps
    /// snapshot-and-drain atomic against concurrent `save` submissions.
    pending_saves: Mutex<Vec<JoinHandle<()>>>,
}

impl PegaflowTier {
    /// Validate the rank's arena geometry and register it with the host
    /// engine as one pegaflow instance: the primary device first (pegaflow's
    /// first owner is the save side), then every mirror under the same
    /// instance — identical layer sets collapse to one replica shard, and
    /// the topology seals once `1 + mirrors` devices have joined.
    pub(crate) fn register(
        host: &Arc<PegaflowHost>,
        spec: OffloadRankSpec,
    ) -> Result<Self, EngineError> {
        validate_arenas(&spec.arenas)?;
        for mirror in &spec.mirrors {
            validate_arenas(&mirror.arenas)?;
            // A name-set mismatch would not fail registration — pegaflow
            // would read disjoint sets as a layer SPLIT and route loads to
            // exactly one device again, silently.
            let matches = mirror.arenas.len() == spec.arenas.len()
                && mirror.arenas.iter().zip(&spec.arenas).all(|(m, p)| {
                    m.name == p.name
                        && m.num_blocks == p.num_blocks
                        && m.segment_bytes == p.segment_bytes
                        && m.segments == p.segments
                        && m.kv_stride_bytes == p.kv_stride_bytes
                        && m.block_stride_bytes == p.block_stride_bytes
                });
            if !matches {
                return Err(EngineError::InvalidArgument(format!(
                    "mirror on device {} does not replicate the primary's \
                     arena names/geometry",
                    mirror.device_id
                )));
            }
        }

        let world_size = 1 + spec.mirrors.len();
        let devices = std::iter::once((spec.device_id, &spec.arenas))
            .chain(spec.mirrors.iter().map(|m| (m.device_id, &m.arenas)));
        for (device_id, arenas) in devices {
            let n = arenas.len();
            let mut layer_names = Vec::with_capacity(n);
            let mut data_ptrs = Vec::with_capacity(n);
            let mut size_bytes = Vec::with_capacity(n);
            let mut num_blocks = Vec::with_capacity(n);
            let mut segment_bytes = Vec::with_capacity(n);
            let mut kv_stride_bytes = Vec::with_capacity(n);
            let mut segments = Vec::with_capacity(n);
            let mut block_stride_bytes = Vec::with_capacity(n);
            for arena in arenas {
                layer_names.push(arena.name.clone());
                data_ptrs.push(arena.base_device_ptr);
                size_bytes.push(arena.size_bytes);
                num_blocks.push(arena.num_blocks);
                // pegaflow's `bytes_per_block` argument is the per-SEGMENT
                // byte count, not the whole block's (see
                // ArenaSpec::segment_bytes).
                segment_bytes.push(arena.segment_bytes);
                kv_stride_bytes.push(arena.kv_stride_bytes);
                segments.push(arena.segments);
                block_stride_bytes.push(arena.block_stride_bytes);
            }
            host.engine().register_context_layer_batch_strided(
                &spec.instance_id,
                &spec.namespace,
                device_id,
                TP_RANK,
                PP_RANK,
                TP_SIZE,
                world_size,
                &layer_names,
                &data_ptrs,
                &size_bytes,
                &num_blocks,
                &segment_bytes,
                &kv_stride_bytes,
                &segments,
                Some(block_stride_bytes.as_slice()),
                transfer_mode(),
                spec.page_first,
            )?;
        }

        Ok(Self {
            host: Arc::clone(host),
            instance_id: spec.instance_id,
            device_id: spec.device_id,
            mirror_devices: spec.mirrors.iter().map(|m| m.device_id).collect(),
            layer_names: spec.arenas.iter().map(|a| a.name.clone()).collect(),
            pending_saves: Mutex::new(Vec::new()),
        })
    }
}

/// The registration-time geometry checks, with pegaflow's historical traps
/// turned into explicit errors at this boundary (the engine itself accepts
/// some of these and misbehaves later):
///
/// - the register call's `bytes_per_block` is per-SEGMENT bytes, so a
///   two-segment (K/V-split) arena must place its V copy beyond K:
///   `kv_stride_bytes >= segment_bytes`;
/// - consecutive blocks sit `block_stride_bytes` apart, so the stride must
///   fit one block's whole strided extent
///   (`(segments - 1) * kv_stride_bytes + segment_bytes`);
/// - `size_bytes` must cover the strided reach of the last block
///   (`(num_blocks - 1) * block_stride_bytes + extent`) — pegaflow validates
///   copies against this bound;
/// - every arena shares the pool's block-id space, so `num_blocks` must
///   match across arenas and names must be unique.
fn validate_arenas(arenas: &[ArenaSpec]) -> Result<(), EngineError> {
    let invalid = |msg: String| EngineError::InvalidArgument(msg);
    if arenas.is_empty() {
        return Err(invalid(
            "rank_with_offload requires at least one arena".into(),
        ));
    }
    let num_blocks = arenas[0].num_blocks;
    let mut names = HashSet::with_capacity(arenas.len());
    for arena in arenas {
        if !names.insert(&arena.name) {
            return Err(invalid(format!(
                "duplicate arena name {:?} in one rank's registration",
                arena.name
            )));
        }
        if arena.num_blocks != num_blocks {
            return Err(invalid(format!(
                "arena {:?} has num_blocks={} but the rank's first arena has {}: \
                 all arenas are indexed by the same pool block ids",
                arena.name, arena.num_blocks, num_blocks
            )));
        }
        if arena.num_blocks == 0 || arena.segment_bytes == 0 || arena.segments == 0 {
            return Err(invalid(format!(
                "arena {:?}: num_blocks, segment_bytes and segments must all be non-zero",
                arena.name
            )));
        }
        if arena.segments == 2 && arena.kv_stride_bytes < arena.segment_bytes {
            return Err(invalid(format!(
                "arena {:?}: two-segment (K/V-split) layout needs \
                 kv_stride_bytes ({}) >= segment_bytes ({}) — V copies start \
                 kv_stride_bytes into the block",
                arena.name, arena.kv_stride_bytes, arena.segment_bytes
            )));
        }
        let extent = (arena.segments - 1) * arena.kv_stride_bytes + arena.segment_bytes;
        if arena.block_stride_bytes < extent {
            return Err(invalid(format!(
                "arena {:?}: block_stride_bytes ({}) must fit one block's extent ({})",
                arena.name, arena.block_stride_bytes, extent
            )));
        }
        let last_block_reach = (arena.num_blocks - 1) * arena.block_stride_bytes + extent;
        if arena.size_bytes < last_block_reach {
            return Err(invalid(format!(
                "arena {:?}: size_bytes ({}) does not cover the last block's \
                 strided reach ({})",
                arena.name, arena.size_bytes, last_block_reach
            )));
        }
    }
    Ok(())
}

/// H2D/D2H backend for the instance's GPU worker pools.
fn transfer_mode() -> TransferMode {
    // Direct (cuMemcpyAsync on the DMA engines) by default: the Kernel
    // backend was A/B'd for fragmented bulk-restore batches (#704) and
    // measured WORSE for co-resident decode (its grid-strided copy kernels
    // compete for SMs with decode kernels). On a prefill-only rank there is
    // no decode to protect, and Direct's per-fragment cuMemcpyAsync
    // serializes badly on host-restore storms — the env var selects per
    // deployment.
    match std::env::var("PEGAINFER_KV_TRANSFER_MODE").as_deref() {
        Ok("kernel") => TransferMode::Kernel,
        _ => TransferMode::Direct,
    }
}

/// `EngineError` into the trait's anyhow channel. A named fn (not a closure
/// or `Error::new` path) so every `map_err` site pins the error type.
fn engine_err(e: EngineError) -> anyhow::Error {
    anyhow::Error::new(e)
}

impl HostTier for PegaflowTier {
    fn query(
        &self,
        req_id: &str,
        block_hashes: Vec<Vec<u8>>,
        wait_full: bool,
    ) -> TierFuture<anyhow::Result<TierQuery>> {
        let engine = Arc::clone(self.host.engine());
        let instance_id = self.instance_id.clone();
        let req_id = req_id.to_string();
        let join = self.host.runtime().spawn(async move {
            let status = engine
                .count_prefix_hit_blocks_with_prefetch(
                    &instance_id,
                    &req_id,
                    &block_hashes,
                    wait_full,
                )
                .await?;
            match status {
                // A deeper tier (SSD / remote peer) is fetching; the store owns
                // the re-query loop.
                PrefetchStatus::Loading => Ok(TierQuery::Loading),
                // Empty `blocks` is the terminal cold-miss; it must not reach
                // `create_query_lease` (which rejects empty sets).
                PrefetchStatus::Ready { blocks, .. } if blocks.is_empty() => Ok(TierQuery::Miss),
                PrefetchStatus::Ready { blocks, .. } => {
                    let blocks_len = blocks.len();
                    let lease = engine.create_query_lease(&instance_id, blocks)?;
                    Ok(TierQuery::Hit(LeasedBlocks {
                        blocks: blocks_len,
                        lease,
                    }))
                }
            }
        });
        Box::pin(async move {
            join.await
                .map_err(|e| anyhow::anyhow!("pegaflow query task aborted: {e}"))?
                .map_err(engine_err)
        })
    }

    fn load(&self, hit: LeasedBlocks, dst_page_ids: Vec<i32>) -> TierFuture<anyhow::Result<()>> {
        // One destination per leased block — minted by `query` and counted
        // here by the reserving `BlockPool`, so the lengths agree by
        // construction (the engine rejects a mismatch, consuming the lease
        // either way).
        debug_assert_eq!(dst_page_ids.len(), hit.blocks);
        // pegaflow indexes GPU blocks by `usize`; pegainfer carries them as
        // `i32` (its kvbm/CUDA convention). Block ids are slot indices,
        // always non-negative.
        let dst: Vec<usize> = dst_page_ids.into_iter().map(|id| id as usize).collect();
        let layer_refs: Vec<&str> = self.layer_names.iter().map(String::as_str).collect();
        // Fan the load out to the primary and every mirror device — the same
        // host blocks land in each device's replica (pegaflow routes by
        // device_id; block ids are shared across replicas). Every submission
        // consumes one unit of the shared lease's `world_size` budget; the
        // receivers resolve when their H2D copies land. Success requires
        // every device — a partial landing would leave a mirror attending
        // over stale pages, the exact corruption this fan-out exists to
        // prevent.
        let mut receivers = Vec::with_capacity(1 + self.mirror_devices.len());
        let mut submit_err: Option<anyhow::Error> = None;
        let devices = std::iter::once(self.device_id).chain(self.mirror_devices.iter().copied());
        for device_id in devices {
            let loads = [(hit.lease, dst.clone())];
            match self.host.engine().batch_load_kv_blocks_multi_layer_inproc(
                &self.instance_id,
                TP_RANK,
                device_id,
                &layer_refs,
                &loads,
            ) {
                // Stop submitting — commit only follows full success.
                Err(e) => {
                    submit_err = Some(anyhow::Error::new(e));
                    break;
                }
                Ok(rx) => receivers.push(rx),
            }
        }
        let engine = Arc::clone(self.host.engine());
        let lease = hit.lease;
        Box::pin(async move {
            // Drain every submitted copy before surfacing any error: on error
            // the caller treats the load as settled and drops its
            // reservation, so a still-in-flight copy would write into pages
            // the pool may have already re-issued. Only a receiver that has
            // resolved is known to be done writing.
            let mut first_err = submit_err;
            for rx in receivers {
                let landed = match rx.await {
                    Ok(done) => done.map_err(engine_err),
                    Err(_) => Err(anyhow::anyhow!(
                        "pegaflow load worker dropped the completion signal"
                    )),
                };
                if let Err(e) = landed
                    && first_err.is_none()
                {
                    first_err = Some(e);
                }
            }
            match first_err {
                None => Ok(()),
                Some(e) => {
                    // A short submission left lease budget unconsumed; return
                    // the host pins now instead of waiting out the TTL sweep.
                    engine.release_query_lease(&lease);
                    Err(e)
                }
            }
        })
    }

    fn release(&self, hit: LeasedBlocks) {
        self.host.engine().release_query_lease(&hit.lease);
    }

    fn save(
        &self,
        block_ids: Vec<i32>,
        block_hashes: Vec<Vec<u8>>,
        guards: Vec<KvBlockGuard>,
    ) -> TierFuture<anyhow::Result<()>> {
        debug_assert_eq!(block_ids.len(), block_hashes.len());
        if block_ids.is_empty() {
            drop(guards);
            return Box::pin(std::future::ready(Ok(())));
        }
        let ids: Vec<usize> = block_ids.iter().map(|&id| id as usize).collect();
        // Fan one (block_id, hash) list across every layer — the device data
        // differs per layer, the ids and hashes don't.
        let saves: Vec<LayerSave> = self
            .layer_names
            .iter()
            .map(|name| LayerSave {
                layer_name: name.clone(),
                block_ids: ids.clone(),
                block_hashes: block_hashes.clone(),
            })
            .collect();
        let engine = Arc::clone(self.host.engine());
        let instance_id = self.instance_id.clone();
        let device_id = self.device_id;
        let (tx, rx) = tokio::sync::oneshot::channel();
        let handle = self.host.runtime().spawn(async move {
            let result = engine
                .batch_save_kv_blocks_from_ipc(&instance_id, TP_RANK, PP_RANK, device_id, saves)
                .await;
            // A dropped observer is the fire-and-forget path: the failure is
            // ours to log, nobody else will see it.
            if let Err(unobserved) = tx.send(result)
                && let Err(e) = unobserved
            {
                log::warn!("pegaflow save failed (best-effort): {e}");
            }
            // Source-block pins release only now the D2H has landed (the
            // REUSE contract — before this point the blocks must not be
            // overwritten by a new request).
            drop(guards);
        });
        let mut pending = self.pending_saves.lock().expect("pending_saves poisoned");
        pending.retain(|h| !h.is_finished());
        pending.push(handle);
        Box::pin(async move {
            rx.await
                .map_err(|_| anyhow::anyhow!("pegaflow save task dropped its result"))?
                .map_err(engine_err)
        })
    }

    fn flush(&self) -> TierFuture<anyhow::Result<()>> {
        let handles: Vec<JoinHandle<()>> = {
            let mut pending = self.pending_saves.lock().expect("pending_saves poisoned");
            pending.retain(|h| !h.is_finished());
            std::mem::take(&mut pending)
        };
        let engine = Arc::clone(self.host.engine());
        let with_registrations = self.host.has_p2p();
        Box::pin(async move {
            for handle in handles {
                // A panicked save task still terminates the barrier — its
                // failure already surfaced through the save's own observer.
                let _ = handle.await;
            }
            // With P2P the barrier extends to MetaServer registration, so a
            // peer's content-hash discovery sees what local queries see.
            if with_registrations {
                engine.flush_saves_and_registrations().await;
            } else {
                engine.flush_saves().await;
            }
            Ok(())
        })
    }
}
