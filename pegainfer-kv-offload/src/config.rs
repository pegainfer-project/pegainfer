//! Configuration for offload engines and their shared host.

/// Cross-instance P2P sharing over pegaflow's MetaServer + RDMA data plane.
///
/// With this set, the engine (a) registers saved block hashes with the
/// MetaServer, (b) serves peer RDMA fetches on `listen_addr`, and (c) on a
/// local host-tier miss, discovers and pulls the missing prefix from whichever
/// peer owns it (one-sided RDMA READ into the local pinned pool, then a normal
/// H2D load). This is the P/D disaggregation data plane: a decode node finds
/// the prefill node's KV by content hash — no handle protocol.
#[derive(Clone, Debug)]
pub struct P2pConfig {
    /// MetaServer gRPC address, e.g. `http://10.0.0.100:50056`.
    pub metaserver_addr: String,
    /// This engine's routable `IP:port` (a literal socket address — it doubles
    /// as the embedded transfer service's bind address, so hostnames are
    /// rejected at startup). Peers dial it for RDMA handshakes and block
    /// queries, and the MetaServer records it as the block owner. Must not be
    /// 0.0.0.0/127.0.0.1 for cross-node use.
    pub advertise_addr: String,
    /// RDMA NIC device names to register the pinned pool on (e.g. `mlx5_0`).
    pub rdma_nics: Vec<String>,
}

/// Tuning knobs for a new `OffloadEngine`.
pub struct OffloadConfig {
    /// Stable identifier shared across this engine's lifetime so prefix blocks
    /// saved by one request are query-visible to the next.
    pub(crate) instance_id: String,
    /// Content-addressing domain shared with P2P peers: two engines see each
    /// other's blocks iff their namespaces match. Callers derive it from
    /// whatever makes KV layouts interchange-safe (model, dtype, block
    /// geometry). Single-node offload can use any constant.
    pub(crate) namespace: String,
    /// CUDA device ordinal whose KV buffer this engine offloads.
    pub(crate) device_id: i32,
    /// Host pinned-memory pool size in bytes (the CPU KV tier capacity).
    pinned_pool_bytes: usize,
    /// Back the pinned pool with hugepages (see [`HostConfig::use_hugepages`]).
    pub use_hugepages: bool,
    /// Worker threads for the embedded runtime that drives pegaflow's async
    /// save/query. Two is plenty: save is fire-and-forget, query is a brief
    /// memory-cache lookup.
    runtime_threads: usize,
    /// `Some` joins the cross-instance P2P mesh (see [`P2pConfig`]).
    p2p: Option<P2pConfig>,
}

impl OffloadConfig {
    pub fn new(instance_id: impl Into<String>, device_id: i32, pinned_pool_bytes: usize) -> Self {
        Self {
            instance_id: instance_id.into(),
            namespace: "pegainfer".to_string(),
            device_id,
            pinned_pool_bytes,
            use_hugepages: false,
            runtime_threads: 2,
            p2p: None,
        }
    }

    /// The host-tier half of this config (private-host constructors split it
    /// off before consuming the instance fields).
    pub(crate) fn host(&self) -> HostConfig {
        HostConfig {
            pinned_pool_bytes: self.pinned_pool_bytes,
            use_hugepages: self.use_hugepages,
            runtime_threads: self.runtime_threads,
            p2p: self.p2p.clone(),
        }
    }

    #[must_use]
    pub fn with_namespace(mut self, namespace: impl Into<String>) -> Self {
        self.namespace = namespace.into();
        self
    }

    #[must_use]
    pub fn with_p2p(mut self, p2p: P2pConfig) -> Self {
        self.p2p = Some(p2p);
        self
    }
}

/// Host-tier knobs for a shared `OffloadHost`.
pub(crate) struct HostConfig {
    /// Host pinned-memory pool size in bytes (the CPU KV tier capacity).
    pub(crate) pinned_pool_bytes: usize,
    /// Back the pinned pool with hugepages (pegaflow supports it natively).
    /// Verify the box actually holds a reservation (`HugePages_Total`) —
    /// some cluster platforms re-claim it across reboots.
    pub(crate) use_hugepages: bool,
    /// Worker threads for the runtime that drives pegaflow's async save/query.
    pub(crate) runtime_threads: usize,
    /// `Some` joins the cross-instance P2P mesh (see [`P2pConfig`]).
    pub(crate) p2p: Option<P2pConfig>,
}
