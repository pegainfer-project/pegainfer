//! [`PegaflowHost`]: the process-level shared host tier — one pinned-memory
//! [`PegaEngine`] plus the multi-thread tokio runtime that drives its async
//! save/query, and the optional P2P serving lifecycle.
//!
//! One host serves any number of rank-level registrations (each
//! [`crate::KvStoreBuilder::rank_with_offload`] becomes one pegaflow
//! *instance* against it). Blocks land in the one host cache keyed by
//! `(namespace, hash)`: ranks that share a namespace restore each other's
//! saves. Share a namespace only when the KV bytes are interchangeable across
//! those instances — for replicated-weight DP ranks that holds to the same
//! tolerance as reusing a rank's own prefix cache (FP reduction order may
//! differ across batch shapes, exactly like two local recomputations of the
//! same prefix).
//!
//! Dropping the last handle drops the runtime, abandoning in-flight
//! fire-and-forget saves (acceptable — the host tier is a cache) and stopping
//! P2P serving; peers degrade to their own local prefill.

use std::path::PathBuf;
use std::sync::Arc;

use pegaflow_core::EngineError;
use pegaflow_core::P2pTransferService;
use pegaflow_core::PegaEngine;
use pegaflow_core::SsdCacheConfig;
use pegaflow_core::StorageConfig;
use tokio::runtime::Runtime;
use tokio::sync::oneshot;

mod runtime;

use runtime::HostRuntime;

/// Cross-node P2P sharing over pegaflow's MetaServer + RDMA data plane.
///
/// With this set, the host (a) registers saved block hashes with the
/// MetaServer, (b) serves peer RDMA fetches on `advertise_addr`, and (c) on a
/// local miss discovers and pulls the prefix from whichever peer owns it
/// (one-sided RDMA READ into the local pinned pool, then a normal H2D load).
/// This is the P/D disaggregation data plane: a decode node finds the prefill
/// node's KV by content hash — no handle protocol.
#[derive(Clone, Debug)]
pub struct P2pConfig {
    /// MetaServer gRPC address, e.g. `http://10.0.0.100:50056`.
    pub metaserver_addr: String,
    /// This node's routable `IP:port` (a literal socket address — it doubles
    /// as the embedded transfer service's bind address, so hostnames are
    /// rejected at startup). Peers dial it for RDMA handshakes and block
    /// queries, and the MetaServer records it as the block owner. Must not be
    /// 0.0.0.0/127.0.0.1 for cross-node use.
    pub advertise_addr: String,
    /// RDMA NIC device names to register the pinned pool on (e.g. `mlx5_0`).
    pub rdma_nics: Vec<String>,
}

/// The shared host side of the store: one [`PegaEngine`], the runtime that
/// drives it, and the P2P serving tasks. Build via [`Self::builder`].
pub struct PegaflowHost {
    // Request runtime shutdown before releasing the engine.
    runtime: HostRuntime,
    engine: Arc<PegaEngine>,
    /// `Some` when P2P is on: resolving the sender shuts the gRPC service
    /// down (dropping the last `PegaflowHost` fires it).
    #[allow(dead_code)]
    p2p_shutdown: Option<oneshot::Sender<()>>,
    has_p2p: bool,
}

impl PegaflowHost {
    /// A host with `pinned_pool_bytes` of pinned host memory (the CPU KV tier
    /// capacity). Further knobs on [`PegaflowHostBuilder`].
    pub fn builder(pinned_pool_bytes: usize) -> PegaflowHostBuilder {
        PegaflowHostBuilder {
            pinned_pool_bytes,
            use_hugepages: false,
            runtime_threads: Self::DEFAULT_RUNTIME_THREADS,
            p2p: None,
            ssd: None,
        }
    }

    const DEFAULT_RUNTIME_THREADS: usize = 8;

    /// Drain both the write pipeline and the SSD writer: on return, every
    /// save submitted before this call is cache-visible *and* persisted to
    /// the SSD tier (when one is configured). The store-side barrier
    /// ([`crate::KvStore::flush_saves`]) deliberately stops at visibility;
    /// this is the checkpoint/test barrier for the deeper tier.
    pub async fn flush_all(&self) {
        self.engine.flush_all().await;
    }

    pub(crate) fn engine(&self) -> &Arc<PegaEngine> {
        &self.engine
    }

    pub(crate) fn runtime(&self) -> &Runtime {
        self.runtime.get()
    }

    pub(crate) fn has_p2p(&self) -> bool {
        self.has_p2p
    }
}

/// Builder for [`PegaflowHost`] — see [`PegaflowHost::builder`].
pub struct PegaflowHostBuilder {
    pinned_pool_bytes: usize,
    use_hugepages: bool,
    runtime_threads: usize,
    p2p: Option<P2pConfig>,
    ssd: Option<(Vec<PathBuf>, u64)>,
}

impl PegaflowHostBuilder {
    /// Back the pinned pool with hugepages (pegaflow supports it natively).
    /// Default false. Verify the box actually holds a reservation
    /// (`HugePages_Total`) — some cluster platforms re-claim it across
    /// reboots.
    #[must_use]
    pub fn use_hugepages(mut self, yes: bool) -> Self {
        self.use_hugepages = yes;
        self
    }

    /// Worker threads for the runtime driving pegaflow's async save/query.
    /// Default 8; the work is control-plane (fire-and-forget saves, brief
    /// cache lookups, P2P serving when enabled), so these threads mostly sit
    /// idle-parked — headroom for concurrency, not compute.
    #[must_use]
    pub fn runtime_threads(mut self, threads: usize) -> Self {
        self.runtime_threads = threads;
        self
    }

    /// Join the cross-node P2P mesh (see [`P2pConfig`]).
    #[must_use]
    pub fn p2p(mut self, config: P2pConfig) -> Self {
        self.p2p = Some(config);
        self
    }

    /// Add an SSD cache tier below the pinned pool: sealed blocks are
    /// persisted to `cache_paths` (capacity `capacity_bytes` across them) and
    /// a local miss prefetches from there, surfacing on the store as the
    /// `Loading`/re-query cycle.
    #[must_use]
    pub fn ssd_cache(mut self, cache_paths: Vec<PathBuf>, capacity_bytes: u64) -> Self {
        self.ssd = Some((cache_paths, capacity_bytes));
        self
    }

    fn storage_config(&self) -> Result<StorageConfig, EngineError> {
        let mut storage_config = StorageConfig::default();
        if let Some(p2p) = &self.p2p {
            if p2p.rdma_nics.is_empty() {
                return Err(EngineError::InvalidArgument(
                    "P2P requires at least one RDMA NIC".into(),
                ));
            }
            storage_config.rdma_nic_names = Some(p2p.rdma_nics.clone());
            storage_config.metaserver_addr = Some(p2p.metaserver_addr.clone());
            storage_config.advertise_addr = Some(p2p.advertise_addr.clone());
        }
        if let Some((paths, capacity_bytes)) = &self.ssd {
            if paths.is_empty() {
                return Err(EngineError::InvalidArgument(
                    "ssd_cache requires at least one cache path".into(),
                ));
            }
            if *capacity_bytes == 0 {
                return Err(EngineError::InvalidArgument(
                    "ssd_cache capacity must be non-zero".into(),
                ));
            }
            storage_config.ssd_cache_config = Some(SsdCacheConfig {
                cache_paths: paths.clone(),
                capacity_bytes: *capacity_bytes,
                ..SsdCacheConfig::default()
            });
        }
        Ok(storage_config)
    }

    pub fn build(self) -> Result<Arc<PegaflowHost>, EngineError> {
        let storage_config = self.storage_config()?;
        // TODO(kv-store): converge on the process-global tokio runtime. The
        // store already borrows the caller's Handle, and this private pool is
        // a second set of threads doing the same kind of work. Kept private
        // for now: engine construction needs a runtime context at build time
        // (MetaServerClient spawns at construction), and the ownership/Drop
        // story (shutdown_background) is simplest while we own it.
        let runtime = HostRuntime::new(self.runtime_threads)?;

        // pegaflow's MetaServerClient spawns its background registration loop
        // with tokio::spawn, so the engine must be built inside our runtime.
        let engine = {
            let _guard = runtime.get().enter();
            Arc::new(PegaEngine::new_with_config(
                self.pinned_pool_bytes,
                self.use_hugepages,
                storage_config,
            )?)
        };

        // P2P serving side: peers discovered us via the MetaServer and dial
        // `advertise_addr` for the RDMA handshake + block queries. Same
        // lifecycle as the engine — shut down (via the oneshot) on drop.
        let p2p_shutdown = match self.p2p {
            Some(p2p) => Some(Self::start_p2p(runtime.get(), &engine, &p2p)?),
            None => None,
        };
        let has_p2p = p2p_shutdown.is_some();

        Ok(Arc::new(PegaflowHost {
            runtime,
            engine,
            p2p_shutdown,
            has_p2p,
        }))
    }

    /// Spawn the P2P gRPC transfer service plus its two background GC sweeps,
    /// returning the shutdown sender. Startup is fail-loud: a taken bind port
    /// errors the build instead of silently never serving.
    fn start_p2p(
        runtime: &Runtime,
        engine: &Arc<PegaEngine>,
        p2p: &P2pConfig,
    ) -> Result<oneshot::Sender<()>, EngineError> {
        let listen: std::net::SocketAddr = p2p.advertise_addr.parse().map_err(|e| {
            EngineError::InvalidArgument(format!(
                "P2P advertise_addr {:?} is not a socket address: {e}",
                p2p.advertise_addr
            ))
        })?;
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
        let serve_engine = Arc::clone(engine);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
        runtime.spawn(async move {
            // Bind eagerly: a taken address must fail startup, not defer to
            // the first peer connection.
            let listener = match tokio::net::TcpListener::bind(listen).await {
                Ok(l) => {
                    let _ = ready_tx.send(Ok(()));
                    l
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(format!("bind {listen}: {e}")));
                    return;
                }
            };
            let incoming = tokio_stream::wrappers::TcpListenerStream::new(listener);
            if let Err(e) = P2pTransferService::serve_with_incoming(serve_engine, incoming, async {
                let _ = shutdown_rx.await;
            })
            .await
            {
                log::error!("P2P transfer service exited: {e}");
            }
        });
        ready_rx
            .recv()
            .map_err(|_| EngineError::Storage("P2P serve task died at startup".into()))?
            .map_err(EngineError::Storage)?;

        // Background GC, mirroring pegaflow-server's task. Two sweeps:
        // expired transfer locks (a crashed peer must not pin our blocks past
        // the lock timeout) and stale prefetch state — an abandoned remote
        // fetch (request dropped mid-RemoteFetch, or the executor's re-query
        // deadline fired) leaves an orphaned entry whose completed task pins
        // its fetched blocks in the pinned pool until this sweep drops it.
        let gc_engine = Arc::clone(engine);
        runtime.spawn(async move {
            const STALE_MAX_AGE: std::time::Duration = std::time::Duration::from_mins(5);
            let mut tick = tokio::time::interval(std::time::Duration::from_mins(1));
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                tick.tick().await;
                let expired = gc_engine.gc_expired_transfer_locks();
                if expired > 0 {
                    log::warn!("P2P GC released {expired} expired transfer locks");
                }
                let (stale, failed) = gc_engine
                    .gc_stale_inflight(STALE_MAX_AGE, STALE_MAX_AGE)
                    .await;
                if stale > 0 || failed > 0 {
                    log::info!(
                        "P2P GC dropped {stale} stale prefetch entries, \
                         {failed} failed-remote markers"
                    );
                }
            }
        });
        log::info!(
            "KV store P2P enabled: serving on {listen}, metaserver={}",
            p2p.metaserver_addr
        );
        Ok(shutdown_tx)
    }
}

// The host is shared across scheduler/store threads and its engine/runtime
// handle is what the per-rank tiers spawn onto; fail at compile time.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PegaflowHost>();
};
