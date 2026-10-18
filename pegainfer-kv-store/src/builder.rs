//! Construction: chainable knobs, rank declaration, and the freeze at
//! [`KvStoreBuilder::build`].

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::time::Duration;

use crate::BlockPool;
use crate::KvStore;
use crate::KvStoreStats;
use crate::host::PegaflowHost;
use crate::store::RankState;
use crate::tier::HostTier;
use crate::tier::PegaflowTier;

/// One declared rank, held until [`KvStoreBuilder::build`] resolves the
/// derived values.
type RankDecl = (usize, Arc<BlockPool>, Option<Arc<dyn HostTier>>);

/// A healthy host restore settles within a handful of polls, so this
/// interval adds no visible latency to one — it only paces how often the
/// tier (and the pool) is re-asked while waiting.
const DEFAULT_REQUERY_INTERVAL: Duration = Duration::from_millis(5);
/// A safety net for a hung tier, not a latency target: set far above any
/// healthy restore, so the failure mode is a degraded resolve instead of a
/// request stranded on the tier.
const DEFAULT_RESOLVE_DEADLINE: Duration = Duration::from_secs(15);

/// Builds a [`KvStore`]. The rank table freezes at [`Self::build`] — the
/// store's read paths take no lock. A `BlockPool` is a pure CPU object with
/// no thread affinity: construct pools before engine threads spawn.
pub struct KvStoreBuilder {
    runtime: tokio::runtime::Handle,
    requery_interval: Duration,
    resolve_deadline: Duration,
    ranks: Vec<RankDecl>,
}

impl KvStoreBuilder {
    #[must_use]
    pub fn new(runtime: tokio::runtime::Handle) -> Self {
        Self {
            runtime,
            requery_interval: DEFAULT_REQUERY_INTERVAL,
            resolve_deadline: DEFAULT_RESOLVE_DEADLINE,
            ranks: Vec::new(),
        }
    }

    /// Pause between host-tier re-queries while a deeper tier is fetching —
    /// also the retry cadence while waiting out pool pressure.
    #[must_use]
    pub fn with_requery_interval(mut self, interval: Duration) -> Self {
        self.requery_interval = interval;
        self
    }

    /// Ceiling on one resolve's host-tier wait — bounds every tier await
    /// (query and load alike) and the pool-pressure wait, so a hung storage
    /// worker degrades the resolve instead of stranding the request outside
    /// the scheduler.
    #[must_use]
    pub fn with_resolve_deadline(mut self, deadline: Duration) -> Self {
        self.resolve_deadline = deadline;
        self
    }

    /// Declare a rank with only its GPU pool: resolves serve radix hits and
    /// seals are no-ops — the tier-less mode behind the same API.
    #[must_use]
    pub fn rank(mut self, rank: usize, pool: Arc<BlockPool>) -> Self {
        self.ranks.push((rank, pool, None));
        self
    }

    /// Declare a rank backed by a real pegaflow offload: registers `spec`'s
    /// GPU arenas with `host`'s shared engine as one pegaflow *instance* and
    /// wires the rank's tier to it. Fails on invalid arena geometry or a
    /// registration error — a half-registered rank must never reach the
    /// frozen rank table.
    ///
    /// Every arena's device allocation must stay live and pointer-stable for
    /// the host's lifetime (the registration bakes raw device addresses), and
    /// all arenas are indexed by the same pool block ids — keep `pool`'s
    /// block count within `num_blocks`.
    pub fn rank_with_offload(
        mut self,
        rank: usize,
        pool: Arc<BlockPool>,
        host: &Arc<PegaflowHost>,
        spec: OffloadRankSpec,
    ) -> anyhow::Result<Self> {
        let tier = Arc::new(PegaflowTier::register(host, spec).map_err(anyhow::Error::new)?);
        self.ranks.push((rank, pool, Some(tier)));
        Ok(self)
    }

    #[must_use]
    pub fn build(self) -> KvStore {
        let ranks = self
            .ranks
            .into_iter()
            .map(|(rank, pool, tier)| {
                (
                    rank,
                    RankState {
                        pool,
                        tier,
                        pinned: Arc::new(AtomicUsize::new(0)),
                        loads: tokio_util::task::TaskTracker::new(),
                    },
                )
            })
            .collect();
        KvStore {
            runtime: self.runtime,
            requery_interval: self.requery_interval,
            resolve_deadline: self.resolve_deadline,
            ranks,
            stats: Arc::new(KvStoreStats::default()),
        }
    }
}

/// One strided GPU arena registered with pegaflow as one "layer" of a rank.
///
/// A fused KV buffer (qwen3-style) contributes one arena per model layer; a
/// model with sidecar caches (GLM5.2: MLA latent + index-K per layer, two
/// separate allocations sharing pool block ids) contributes several arenas
/// per model layer — pegaflow moves whatever arenas are registered under one
/// block id together, keeping sidecars in lockstep with their main cache.
///
/// The registration parameter names are pegaflow's historical traps; read
/// these field docs before filling them in:
///
/// - pegaflow's `bytes_per_block` argument is the **per-segment** byte count,
///   so [`segment_bytes`](Self::segment_bytes) is one segment's span, not the
///   whole block's;
/// - with `segments == 2` (K/V split), the V copy starts `kv_stride_bytes`
///   into the block, so `kv_stride_bytes >= segment_bytes` must hold;
/// - `block_stride_bytes >= (segments - 1) * kv_stride_bytes + segment_bytes`
///   (one block's whole strided extent), and
///   `size_bytes >= (num_blocks - 1) * block_stride_bytes + extent` (the last
///   block's reach is what pegaflow validates copies against).
///
/// All of these are checked by [`KvStoreBuilder::rank_with_offload`] before
/// the engine sees them.
pub struct ArenaSpec {
    /// Keys the arena for the engine's lifetime (save/load fan out across
    /// every registered name); unique within the rank's registration.
    pub name: String,
    /// Base device address of the arena. The allocation behind it must stay
    /// live and pointer-stable for the host's lifetime.
    pub base_device_ptr: u64,
    /// Total bytes of the arena allocation (must cover the last block's
    /// strided reach — see the struct docs).
    pub size_bytes: usize,
    /// Copy units in this arena; identical across all of a rank's arenas
    /// (they share the pool's block-id space).
    pub num_blocks: usize,
    /// Bytes of one segment of one block — this is what pegaflow's register
    /// call misleadingly names `bytes_per_block`.
    pub segment_bytes: usize,
    /// Segments per block: 1 for a contiguous copy unit, 2 for a K/V split.
    pub segments: usize,
    /// Byte offset of segment k+1 from segment k within one block. 0 for
    /// single-segment arenas.
    pub kv_stride_bytes: usize,
    /// Byte distance between consecutive blocks in the arena (≥ one block's
    /// extent; larger expresses page-interleaved layouts like qwen3's).
    pub block_stride_bytes: usize,
}

/// The per-rank half of a pegaflow offload registration (the process-wide
/// half is [`PegaflowHost`]).
pub struct OffloadRankSpec {
    /// Stable identifier of this rank's pegaflow instance for the host's
    /// lifetime, so prefix blocks saved by one request are query-visible to
    /// the next.
    pub instance_id: String,
    /// Content-addressing domain shared with peers: two instances see each
    /// other's blocks iff their namespaces match. Derive it from whatever
    /// makes KV layouts interchange-safe (model, dtype, block geometry).
    /// Single-node offload can use any constant.
    pub namespace: String,
    /// CUDA device ordinal whose arenas these are.
    pub device_id: i32,
    /// The rank's GPU arenas, one pegaflow "layer" each (see [`ArenaSpec`]).
    pub arenas: Vec<ArenaSpec>,
    /// `false` = layer-first (one pegaflow layer per arena, interleave in
    /// `block_stride_bytes`) — the native pegainfer layout. `true` =
    /// page-first: each block stored as one host page holding every layer at
    /// its name-sorted offset; only to join a namespace whose writer (the
    /// vLLM connector on MLA models) stores blocks that way — with layer
    /// names and per-layer block bytes identical to the writer's.
    pub page_first: bool,
    /// Tensor-replicated mirrors of this rank's arenas on other devices.
    /// Each mirror registers the same layer set from its own device under the
    /// same instance, entering pegaflow's replica contract: the primary
    /// (`device_id`) saves, and every tier load lands on the primary AND
    /// every mirror — a load that reached only one device would leave the
    /// other TP workers attending over stale pages (openinfer#849, the #847
    /// review finding). Empty for single-device ranks.
    pub mirrors: Vec<OffloadMirror>,
}

/// One tensor-replicated mirror of a rank's arenas: the same layer names and
/// geometry as the primary, backed by another device's memory.
pub struct OffloadMirror {
    /// CUDA device ordinal whose arenas these are.
    pub device_id: i32,
    /// The mirror's arenas; names and geometry must match the primary's
    /// (checked at registration).
    pub arenas: Vec<ArenaSpec>,
}
