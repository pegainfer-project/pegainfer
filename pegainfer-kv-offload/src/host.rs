//! The shared host side of the offload: one [`PegaEngine`], the tokio runtime
//! that drives it, and the optional P2P serving lifecycle.

use std::sync::Arc;

use pegaflow_core::EngineError;
use pegaflow_core::P2pTransferService;
use pegaflow_core::PegaEngine;
use pegaflow_core::StorageConfig;
use tokio::runtime::Runtime;
use tokio::sync::oneshot;

use crate::config::HostConfig;

/// One host serves any number of rank-level `OffloadEngine`s. That is the
/// DP-rank sharing model: each rank registers its own GPU arenas as its own
/// pegaflow *instance*, but blocks land in the one host tier keyed by
/// `(namespace, hash)` — with a shared namespace, any rank restores what any
/// rank saved. Callers share a namespace only when their KV is
/// interchangeable across instances: for replicated-weight DP ranks that
/// holds to the same tolerance as reusing a rank's own prefix cache (the
/// bytes may differ by FP reduction order across batch shapes, exactly like
/// two local recomputations of the same prefix would).
///
/// Dropping the last handle drops the [`Runtime`], which abandons any
/// in-flight fire-and-forget saves (acceptable — the host tier is a cache)
/// and stops the P2P serving tasks; peers degrade to their own local
/// prefill. In-flight `OffloadEngine::flush_saves_then` barriers are
/// cancelled too, dropping their `then` callbacks unrun.
pub struct OffloadHost {
    pub(crate) engine: Arc<PegaEngine>,
    pub(crate) runtime: Runtime,
    /// `Some` when P2P is on: resolves the P2P serving tasks (gRPC transfer
    /// service + transfer-lock GC) on drop.
    #[allow(dead_code)]
    p2p_shutdown: Option<oneshot::Sender<()>>,
}

impl OffloadHost {
    pub(crate) fn new(config: HostConfig) -> Result<Arc<Self>, EngineError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(config.runtime_threads.max(1))
            .enable_all()
            .build()
            .map_err(|e| EngineError::Storage(format!("offload runtime build: {e}")))?;

        let mut storage_config = StorageConfig::default();
        if let Some(p2p) = &config.p2p {
            if p2p.rdma_nics.is_empty() {
                return Err(EngineError::InvalidArgument(
                    "P2P requires at least one RDMA NIC".into(),
                ));
            }
            storage_config.rdma_nic_names = Some(p2p.rdma_nics.clone());
            storage_config.metaserver_addr = Some(p2p.metaserver_addr.clone());
            storage_config.advertise_addr = Some(p2p.advertise_addr.clone());
        }
        // pegaflow's MetaServerClient spawns its background registration loop
        // with tokio::spawn, so the engine must be built inside our runtime.
        let engine = {
            let _guard = runtime.enter();
            Arc::new(PegaEngine::new_with_config(
                config.pinned_pool_bytes,
                config.use_hugepages,
                storage_config,
            )?)
        };

        // P2P serving side: peers discovered us via the MetaServer and dial
        // `advertise_addr` for the RDMA handshake + block queries. Same
        // lifecycle as the engine — shut down (via the oneshot) on drop.
        let p2p_shutdown = match config.p2p {
            Some(p2p) => {
                let listen: std::net::SocketAddr = p2p.advertise_addr.parse().map_err(|e| {
                    EngineError::InvalidArgument(format!(
                        "P2P advertise_addr {:?} is not a socket address: {e}",
                        p2p.advertise_addr
                    ))
                })?;
                let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();
                let serve_engine = Arc::clone(&engine);
                let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<(), String>>();
                runtime.spawn(async move {
                    // Bind eagerly so startup fails loud on a taken port
                    // instead of P2P silently never serving.
                    let bound = tokio::net::TcpListener::bind(listen).await;
                    let listener = match bound {
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
                    if let Err(e) =
                        P2pTransferService::serve_with_incoming(serve_engine, incoming, async {
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
                // expired transfer locks (a crashed peer must not pin our
                // blocks past the lock timeout) and stale prefetch state — an
                // abandoned remote fetch (request dropped mid-RemoteFetch, or
                // the executor's re-query deadline fired) leaves an orphaned
                // entry whose completed task pins its fetched blocks in the
                // pinned pool until this sweep drops it.
                let gc_engine = Arc::clone(&engine);
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
                    "KV offload P2P enabled: serving on {listen}, metaserver={}",
                    p2p.metaserver_addr
                );
                Some(shutdown_tx)
            }
            None => None,
        };

        Ok(Arc::new(Self {
            engine,
            runtime,
            p2p_shutdown,
        }))
    }
}
