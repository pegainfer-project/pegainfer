//! [`OffloadEngine`]: the in-process connector that moves KV blocks between
//! pegainfer's GPU paged cache and pegaflow's host/SSD tiers.
//!
//! It owns a pegaflow instance registration over a shared [`OffloadHost`] and
//! translates pegainfer's page-first KV layout into pegaflow's per-layer
//! strided registration. Block content hashes are opaque `Vec<u8>` here — the
//! caller (scheduler) derives them from kvbm sequence hashes, so this layer
//! never depends on the logical-cache hashing scheme.
//!
//! Every operation is submitted to pegaflow's async runtime and observed
//! through a pollable handle ([`QueryHandle`] / [`SaveHandle`] /
//! [`LoadHandle`]); the blocking entry points are thin submit-then-wait
//! wrappers for tests and non-pipelined callers.

use std::sync::Arc;
use std::sync::Mutex;

use cudarc::driver::CudaStream;
use pegaflow_core::EngineError;
use pegaflow_core::LayerSave;
use pegaflow_core::PrefetchStatus;
use pegaflow_core::QueryLeaseId;
use pegainfer_kv_cache::KvBuffer;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

use crate::config::OffloadConfig;
use crate::handle::LoadHandle;
use crate::handle::QueryHandle;
use crate::handle::QueryHit;
use crate::handle::QueryOutcome;
use crate::handle::SaveHandle;
use crate::host::OffloadHost;
use crate::layout::KvArena;
use crate::layout::Registration;

/// Single-GPU, single-rank topology. The dense Qwen3 path runs one offload
/// engine per executor rank, each owning one GPU's KV buffer.
const TP_RANK: usize = 0;
const PP_RANK: usize = 0;
const TP_SIZE: usize = 1;
const WORLD_SIZE: usize = 1;

/// Upper bound on the [`OffloadEngine::flush_saves_then`] barrier. Generous
/// for the normal case (D2H drain + a few local RPCs complete in
/// milliseconds); the cap only bites when the MetaServer connection stalls
/// mid-RPC, where the alternative is withholding finished requests' responses
/// for the TCP keepalive window.
const FLUSH_DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

/// Guard the blocking entry points: tokio panics with an opaque message if
/// you block on a runtime from within any runtime. These methods are meant for
/// the synchronous scheduler thread — fail loud and specific if that's violated.
fn assert_outside_runtime(op: &str) {
    debug_assert!(
        tokio::runtime::Handle::try_current().is_err(),
        "OffloadEngine::{op} blocks on the offload runtime and must be \
         called from a synchronous thread, never from within a tokio runtime"
    );
}

/// In-process bridge from one rank's GPU KV cache to pegaflow's offload
/// tiers, over a shared or private [`OffloadHost`].
///
/// Save is best-effort fire-and-forget (a lost save only forfeits a future
/// hit, never inference correctness); saves that must survive a handoff
/// (eviction) use [`Self::submit_save`] / [`Self::save_blocking`] and observe
/// the result. Runtime and P2P lifetime live on the host — see
/// [`OffloadHost`] for drop semantics.
pub struct OffloadEngine {
    host: Arc<OffloadHost>,
    instance_id: String,
    device_id: i32,
    /// Owned per-layer names; load borrows these as `&[&str]`.
    layer_names: Vec<String>,
    /// In-flight fire-and-forget save tasks plus the completion signal of the
    /// latest flush barrier. One lock so a barrier's "drain handles + chain
    /// behind the previous barrier" is atomic — two racing barriers can never
    /// each take half the coverage (see [`Self::flush_saves_then`]).
    write_barrier: Mutex<WriteBarrierState>,
}

/// Save handles and barrier chain behind [`OffloadEngine::write_barrier`].
struct WriteBarrierState {
    /// In-flight save tasks; finished handles are pruned on each
    /// [`OffloadEngine::submit_save`].
    pending_saves: Vec<JoinHandle<()>>,
    /// Completion signal of the latest spawned flush barrier (fires on
    /// success and deadline alike). `None` before the first barrier.
    prev_flush_done: Option<oneshot::Receiver<()>>,
}

impl OffloadEngine {
    /// Build a private host and register `buffer` as the GPU side of the
    /// offload.
    ///
    /// `stream` must be the stream that owns `buffer` (used only to read its
    /// base device address). pegaflow attaches the device's primary CUDA
    /// context for its own worker transfers — the same context pegainfer runs
    /// on — so the registered pointers are valid across both.
    pub fn new(
        config: OffloadConfig,
        buffer: &KvBuffer,
        stream: &CudaStream,
    ) -> Result<Self, EngineError> {
        let reg = Registration::from_buffer(buffer, stream);
        let host = OffloadHost::new(config.host())?;
        Self::register(
            host,
            config.instance_id,
            &config.namespace,
            config.device_id,
            reg,
            false,
        )
    }

    /// Build the engine over explicit arenas (instead of one fused
    /// [`KvBuffer`]) onto an existing shared host: this rank becomes
    /// one more pegaflow instance over the host's single pool. Ranks that
    /// should see each other's blocks pass the same `namespace`. Every
    /// arena's device allocation must stay live and pointer-stable for the
    /// engine's lifetime (the registration bakes raw device addresses), and
    /// all arenas must be indexed by the same pool block ids.
    /// `page_first` must match how the namespace's writer stores blocks: the
    /// vLLM connector stores MLA-model blocks page-first (all layers of a
    /// block concatenated into one host page, offsets by lexicographic layer
    /// name), so joining a vLLM MLA namespace requires `true` — with layer
    /// names and per-layer block bytes identical to the writer's.
    pub fn with_arenas_on(
        host: Arc<OffloadHost>,
        instance_id: impl Into<String>,
        namespace: &str,
        device_id: i32,
        arenas: &[KvArena],
        page_first: bool,
    ) -> Result<Self, EngineError> {
        Self::register(
            host,
            instance_id.into(),
            namespace,
            device_id,
            Registration::from_arenas(arenas),
            page_first,
        )
    }

    fn register(
        host: Arc<OffloadHost>,
        instance_id: String,
        namespace: &str,
        device_id: i32,
        reg: Registration,
        page_first: bool,
    ) -> Result<Self, EngineError> {
        host.engine.register_context_layer_batch_strided(
            &instance_id,
            namespace,
            device_id,
            TP_RANK,
            PP_RANK,
            TP_SIZE,
            WORLD_SIZE,
            &reg.layer_names,
            &reg.data_ptrs,
            &reg.size_bytes,
            &reg.num_blocks,
            &reg.bytes_per_block,
            &reg.kv_stride_bytes,
            &reg.segments,
            Some(reg.block_stride_bytes.as_slice()),
            // Direct (cuMemcpyAsync on the DMA engines) by default: the
            // Kernel backend was A/B'd for the fragmented bulk-restore
            // batches (#704) and measured WORSE for co-resident decode (its
            // grid-strided copy kernels compete for SMs with decode
            // kernels). On a prefill-only rank there is no decode to
            // protect, and Direct's per-fragment cuMemcpyAsync serializes
            // badly on host-restore storms — PEGAINFER_KV_TRANSFER_MODE
            // selects per deployment.
            match std::env::var("PEGAINFER_KV_TRANSFER_MODE").as_deref() {
                Ok("kernel") => pegaflow_core::TransferMode::Kernel,
                _ => pegaflow_core::TransferMode::Direct,
            },
            // Layer-first (false): one pegaflow layer per model layer, the
            // page-interleaved gap expressed via `block_stride_bytes` — the
            // native pegainfer layout. Page-first (true) instead stores each
            // block as one host page holding every layer at its
            // name-sorted offset; used only to join a namespace whose writer
            // (the vLLM connector on MLA models) stores blocks that way.
            page_first,
        )?;

        Ok(Self {
            host,
            instance_id,
            device_id,
            layer_names: reg.layer_names,
            write_barrier: Mutex::new(WriteBarrierState {
                pending_saves: Vec::new(),
                prev_flush_done: None,
            }),
        })
    }

    /// Fan one (block_id, hash) list across every layer — the device data
    /// differs per layer, the ids and hashes don't.
    fn build_saves(&self, block_ids: &[i32], block_hashes: &[Vec<u8>]) -> Vec<LayerSave> {
        // pegaflow indexes GPU blocks by `usize`; pegainfer carries them as
        // `i32` (its kvbm/CUDA convention). Convert once at this boundary —
        // block ids are slot indices, always non-negative.
        let block_ids: Vec<usize> = block_ids.iter().map(|&id| id as usize).collect();
        self.layer_names
            .iter()
            .map(|name| LayerSave {
                layer_name: name.clone(),
                block_ids: block_ids.clone(),
                block_hashes: block_hashes.to_vec(),
            })
            .collect()
    }

    /// Submit a GPU→CPU save of the named blocks; the [`SaveHandle`] resolves
    /// once the host tier has captured the data (the insert may still be in
    /// flight; pair with [`Self::flush_saves_then`] for cache visibility).
    /// A dropped handle degrades to fire-and-forget: the save still runs and
    /// failures are logged instead of surfaced.
    ///
    /// `block_hashes[i]` is the content hash of `block_ids[i]`; all layers
    /// share the same (block_id, hash) pairing — only the device data differs.
    ///
    /// ORDERING CONTRACT: pegaflow's D2H runs on *its own* stream, with no
    /// dependency on pegainfer's compute stream. The caller must therefore only
    /// save blocks whose KV writes are already complete — i.e. call this after
    /// the producing forward step has synchronized (block-seal time, which is
    /// post-step-sync in the executor). Saving a block whose attention write is
    /// still in flight reads torn data. This connector cannot enforce the
    /// invariant (it does not own the compute stream); the wiring must uphold it.
    ///
    /// REUSE CONTRACT: the copy reads the GPU block asynchronously *after* this
    /// returns, so the block must stay stable until the copy lands. `keep_alive`
    /// is an opaque payload (e.g. the source blocks' allocator guards) held for
    /// the lifetime of the spawned save and dropped only once it finishes — so
    /// the caller's blocks cannot be evicted and overwritten under the in-flight
    /// D2H (which would snapshot the wrong KV and persist it under the old hash).
    /// Pass `()` only when the blocks are owned elsewhere for the whole save.
    fn submit_save<G: Send + 'static>(
        &self,
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
        keep_alive: G,
    ) -> SaveHandle {
        debug_assert_eq!(block_ids.len(), block_hashes.len());
        if block_ids.is_empty() {
            return SaveHandle::settled(Ok(()));
        }
        let saves = self.build_saves(block_ids, block_hashes);
        let engine = Arc::clone(&self.host.engine);
        let instance_id = self.instance_id.clone();
        let device_id = self.device_id;
        let (tx, rx) = oneshot::channel();
        let handle = self.host.runtime.spawn(async move {
            let result = engine
                .batch_save_kv_blocks_from_ipc(&instance_id, TP_RANK, PP_RANK, device_id, saves)
                .await;
            // A dropped receiver is the fire-and-forget path: the failure is
            // ours to log, nobody else will see it.
            if let Err(unobserved) = tx.send(result)
                && let Err(e) = unobserved
            {
                log::warn!("pegaflow save failed (best-effort): {e}");
            }
            // Release the source-block pins only now the D2H has landed; before
            // this point the blocks must not be reused (see REUSE CONTRACT).
            drop(keep_alive);
        });
        // Track for the flush barrier; prune the ones that already settled so
        // the list stays bounded by the genuinely in-flight saves.
        let mut barrier = self.write_barrier.lock().expect("write_barrier poisoned");
        barrier.pending_saves.retain(|h| !h.is_finished());
        barrier.pending_saves.push(handle);
        SaveHandle::from_rx(rx)
    }

    /// Fire-and-forget form of [`Self::submit_save`]: any failure (pinned pool
    /// full, copy error) is logged, never surfaced. Both contracts of
    /// `submit_save` apply.
    pub fn save<G: Send + 'static>(
        &self,
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
        keep_alive: G,
    ) {
        drop(self.submit_save(block_ids, block_hashes, keep_alive));
    }

    /// Submit-then-wait form of [`Self::submit_save`], for synchronous
    /// callers: the GPU block can be reused the moment this returns, and
    /// errors surface.
    pub fn save_blocking(
        &self,
        block_ids: &[i32],
        block_hashes: &[Vec<u8>],
    ) -> Result<(), EngineError> {
        assert_outside_runtime("save_blocking");
        self.submit_save(block_ids, block_hashes, ()).wait()
    }

    /// Submit a lookup for how long a prefix of `block_hashes` is resident in
    /// the CPU tier.
    ///
    /// The [`QueryHandle`] resolves to [`QueryOutcome::Ready`] with the
    /// hit-block count and a lease owning those blocks (pass the lease to
    /// [`Self::load`] to copy them to GPU), or [`QueryOutcome::Loading`] when
    /// pegaflow is fetching the missing prefix from a remote peer / SSD in the
    /// background — re-submit with the same `req_id` to poll. `req_id` must be
    /// non-empty and unique enough to scope an in-flight prefetch (the request
    /// id works).
    ///
    /// `wait_for_full_prefix` makes the fetch all-or-nothing: the query stays
    /// `Loading` until the *entire* missing prefix is host-resident, instead
    /// of resolving `Ready` with a partial hit. Use it when the caller cannot
    /// recompute the miss (the native P/D handoff); leave it off when a
    /// partial prefix is still a win (plain host restore).
    ///
    /// A handle dropped before it settles can strand a `Ready` outcome's
    /// lease until its TTL expires — poll every submitted query to settlement
    /// (the same discipline as an abandoned [`LoadHandle`]).
    fn submit_query(
        &self,
        req_id: &str,
        block_hashes: &[Vec<u8>],
        wait_for_full_prefix: bool,
    ) -> QueryHandle {
        if block_hashes.is_empty() {
            return QueryHandle::settled(Ok(QueryOutcome::Ready(QueryHit {
                lease: None,
                num_blocks: 0,
            })));
        }
        let engine = Arc::clone(&self.host.engine);
        let instance_id = self.instance_id.clone();
        let req_id = req_id.to_string();
        let block_hashes = block_hashes.to_vec();
        let (tx, rx) = oneshot::channel();
        self.host.runtime.spawn(async move {
            let result = async {
                let status = engine
                    .count_prefix_hit_blocks_with_prefetch(
                        &instance_id,
                        &req_id,
                        &block_hashes,
                        wait_for_full_prefix,
                    )
                    .await?;
                match status {
                    PrefetchStatus::Loading => Ok(QueryOutcome::Loading),
                    PrefetchStatus::Ready { blocks, .. } => {
                        if blocks.is_empty() {
                            return Ok(QueryOutcome::Ready(QueryHit {
                                lease: None,
                                num_blocks: 0,
                            }));
                        }
                        let num_blocks = blocks.len();
                        let lease = engine.create_query_lease(&instance_id, blocks)?;
                        Ok(QueryOutcome::Ready(QueryHit {
                            lease: Some(lease),
                            num_blocks,
                        }))
                    }
                }
            }
            .await;
            let _ = tx.send(result);
        });
        QueryHandle::from_rx(rx)
    }

    /// Submit-then-wait form of [`Self::submit_query`], for synchronous
    /// callers.
    pub fn query(
        &self,
        req_id: &str,
        block_hashes: &[Vec<u8>],
        wait_for_full_prefix: bool,
    ) -> Result<QueryOutcome, EngineError> {
        assert_outside_runtime("query");
        self.submit_query(req_id, block_hashes, wait_for_full_prefix)
            .wait()
    }

    /// Copy the leased CPU blocks into the GPU blocks named by `dst_block_ids`,
    /// across every registered layer. Returns a non-blocking [`LoadHandle`].
    ///
    /// `dst_block_ids.len()` must equal the lease's block count (the
    /// `num_blocks` from [`Self::submit_query`]); pegaflow maps the i-th
    /// leased block onto `dst_block_ids[i]` for each layer.
    pub fn load(
        &self,
        lease: QueryLeaseId,
        dst_block_ids: Vec<i32>,
    ) -> Result<LoadHandle, EngineError> {
        let layer_refs: Vec<&str> = self.layer_names.iter().map(String::as_str).collect();
        // pegaflow indexes GPU blocks by `usize` (see `build_saves`).
        let dst_block_ids: Vec<usize> = dst_block_ids.into_iter().map(|id| id as usize).collect();
        let loads = [(lease, dst_block_ids)];
        let rx = self.host.engine.batch_load_kv_blocks_multi_layer_inproc(
            &self.instance_id,
            TP_RANK,
            self.device_id,
            &layer_refs,
            &loads,
        )?;
        Ok(LoadHandle::from_rx(rx))
    }

    /// Release a query lease without loading it.
    ///
    /// A query pins its hit blocks behind a lease until [`Self::load`]
    /// consumes it. When the caller decides not to load (e.g. no GPU
    /// destination blocks are free), it must release the lease here — a dropped
    /// [`QueryLeaseId`] is an inert token, so without this the pinned host
    /// blocks would sit unevictable until the lease's TTL expires. A no-op if
    /// the lease was already consumed by a `load`.
    pub fn release_query_lease(&self, lease: QueryLeaseId) {
        self.host.engine.release_query_lease(&lease);
    }

    /// Blocking form of [`Self::flush_saves_then`], for tests and eviction
    /// handoff on synchronous threads. Bounded by the same [`FLUSH_DEADLINE`]
    /// chain.
    pub fn flush_saves(&self) {
        assert_outside_runtime("flush_saves");
        let (tx, rx) = oneshot::channel();
        self.flush_saves_then(move || {
            let _ = tx.send(());
        });
        // block_on (not a bare channel wait) keeps the call-from-a-runtime
        // misuse a loud tokio panic instead of a silently deadlocked worker.
        let _ = self.host.runtime.block_on(rx);
    }

    /// Barrier the save pipeline, then call `then` — without blocking the
    /// caller. Once `then` runs, a following query (local or from a
    /// P2P peer) observes every block saved before this call: this is the P/D
    /// KV-ready signal, where the prefill node withholds a request's
    /// `Finished` event until its KV is peer-visible.
    ///
    /// The barrier first awaits every in-flight [`Self::submit_save`] (their
    /// D2H copy + write-pipeline submit), then drains the write
    /// pipeline, then waits for the queued MetaServer registrations to be
    /// delivered (or dropped after a failed attempt — registration stays
    /// best-effort, the barrier only bounds *when* delivery is attempted,
    /// never *whether* it succeeds; a peer that misses a registration
    /// degrades to recompute). Without P2P the last step is a no-op.
    ///
    /// Barriers chain: each first awaits the previous barrier's completion,
    /// so — as long as no barrier in the chain hit its deadline — it
    /// transitively covers every save submitted before its own call,
    /// including handles an earlier barrier drained whose D2H had not yet
    /// submitted into the write pipeline (e.g. a chunked prefill's early
    /// chunks flushed by another request's finish). Without the chain, the
    /// pipeline drain cannot see such saves and the barrier would falsely
    /// report them visible. A predecessor that timed out may leave its
    /// drained handles permanently uncovered; that is the same accepted
    /// degradation as the deadline itself — peers recompute.
    ///
    /// Each barrier is capped at [`FLUSH_DEADLINE`] (the wait on the
    /// predecessor counts against it, and the predecessor is itself capped,
    /// so delays never accumulate): a stalled MetaServer connection degrades
    /// to "registrations still in flight" — semantically the same as a
    /// dropped registration — and `then` still runs.
    pub fn flush_saves_then(&self, then: impl FnOnce() + Send + 'static) {
        let (done_tx, done_rx) = oneshot::channel();
        let (handles, prev_done) = {
            let mut barrier = self.write_barrier.lock().expect("write_barrier poisoned");
            let handles = std::mem::take(&mut barrier.pending_saves);
            let prev_done = barrier.prev_flush_done.replace(done_rx);
            (handles, prev_done)
        };
        let engine = Arc::clone(&self.host.engine);
        self.host.runtime.spawn(async move {
            let flushed = tokio::time::timeout(FLUSH_DEADLINE, async {
                if let Some(prev) = prev_done {
                    // A cancelled predecessor (runtime teardown) resolves as
                    // an error immediately — don't let it stall the chain.
                    let _ = prev.await;
                }
                for handle in handles {
                    let _ = handle.await;
                }
                engine.flush_saves_and_registrations().await;
            })
            .await;
            if flushed.is_err() {
                log::warn!(
                    "KV offload flush timed out after {FLUSH_DEADLINE:?}; \
                     saves/registrations still in flight (peers may recompute)"
                );
            }
            let _ = done_tx.send(());
            then();
        });
    }
}
