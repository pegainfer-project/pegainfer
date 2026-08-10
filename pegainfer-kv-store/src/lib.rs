//! `pegainfer-kv-store`: the shared KV read/write orchestration layer.
//!
//! One `KvStore` per process orchestrates every rank's prefix-cache reads
//! (GPU radix + host tier) and checkpoint writes over the primitives the
//! models already use — the logical [`BlockPool`] for GPU pages, and the
//! pegaflow-backed host tier below it (wired via [`PegaflowHost`] +
//! [`KvStoreBuilder::rank_with_offload`]). Design rationale and migration
//! plan: `docs/subsystems/kv-cache/design.md`.
//!
//! # Wiring it up
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use pegainfer_kv_store::BlockPool;
//! use pegainfer_kv_store::CacheScope;
//! use pegainfer_kv_store::KvStoreBuilder;
//! use pegainfer_kv_store::NeverCancelled;
//! use pegainfer_kv_store::ResolvePolicy;
//!
//! # async fn wire() -> anyhow::Result<()> {
//! // 1. Build once at launch, one entry per rank. The pool is the SAME
//! //    logical pool that rank's scheduler allocates from. A rank declared
//! //    with `rank_with_offload(rank, pool, &host, spec)` gets host offload;
//! //    a bare `rank(rank, pool)` still serves GPU radix hits.
//! let pool = Arc::new(BlockPool::new(64, 1 << 14));
//! let store = Arc::new(
//!     KvStoreBuilder::new(tokio::runtime::Handle::current())
//!         .rank(0, Arc::clone(&pool))
//!         .build(),
//! );
//!
//! // 2. Read path — called from the dispatch task, BEFORE the request
//! //    reaches the scheduler. Bind the request to `rank` first: the
//! //    returned prefix pins blocks on that rank and routes by it
//! //    (`EngineHandle::submit_resolved(req, prefix)` moves both into the
//! //    rank's inbox). Pass the request's `TokenSink` as the cancel probe.
//! let prompt_tokens = [1u32, 2, 3];
//! let prefix = store
//!     .resolve_prefix(
//!         0,
//!         "req-1",
//!         &prompt_tokens,
//!         CacheScope::default(), // .cache_salt(..) / .lora(..) as the model requires
//!         ResolvePolicy::default(), // .wait_for_full_hit() on the disaggregated-decode side
//!         &NeverCancelled,
//!     )
//!     .await;
//! # let _ = prefix;
//! # Ok(())
//! # }
//! ```
//!
//! The scheduler side stays synchronous, four touchpoints per request:
//!
//! 1. **Intake**: recv `(req, kv_prefix)`; after `match_and_add_prefix`
//!    reports the cached tokens, `drop(kv_prefix)` — its hold is the
//!    anti-eviction pin for the resolved blocks, nothing else.
//! 2. **Admission budget**: subtract [`KvStore::pinned_blocks`] from the
//!    usable pool.
//! 3. **Checkpoint boundaries**: [`KvStore::seal`] with a [`SaveCursor`] kept
//!    next to the `RequestKv` (policy decides when; see the design doc).
//! 4. **Retirement**: [`KvStore::retire`] — the final seal plus release,
//!    parking the KV with any must-complete saves.
//!
//! Executable end-to-end sequences live in `tests/` (the real pegaflow
//! engine over real GPU arenas; needs a GPU, and for the SSD cases in
//! `tests/ssd.rs` an io_uring-capable kernel).
//!
//! # Contracts
//!
//! - [`KvStore::resolve_prefix`] is the whole read path (probe → host query
//!   with re-query/deadline → reserve → load → radix commit).
//!   One terminal type: [`KvPrefix`] — degraded outcomes surface as a
//!   smaller hit plus a stats event, never a distinct variant; pool
//!   pressure is waited out under the deadline, not degraded on.
//! - [`KvStore::seal`] / [`KvStore::retire`] are the write path. Guards pin
//!   source pages across the async D2H — those pins count into
//!   [`KvStore::pinned_blocks`], which admission subtracts from its budget;
//!   [`SaveClass::Handoff`] must complete and parks the KV.
//! - Cancellation is the request's existing abort state ([`CancelProbe`]
//!   over `TokenSink::is_closed`) observed between operations; a submitted
//!   DMA is an uncancellable section.

mod builder;
mod host;
mod policy;
pub mod pool;
mod stats;
mod store;
mod tier;

pub use pegainfer_frontend::engine::KvPrefix;

pub use crate::builder::ArenaSpec;
pub use crate::builder::KvStoreBuilder;
pub use crate::builder::OffloadMirror;
pub use crate::builder::OffloadRankSpec;
pub use crate::host::P2pConfig;
pub use crate::host::PegaflowHost;
pub use crate::host::PegaflowHostBuilder;
pub use crate::policy::CacheScope;
pub use crate::policy::CancelProbe;
pub use crate::policy::NeverCancelled;
pub use crate::policy::ResolvePolicy;
pub use crate::policy::SaveClass;
pub use crate::policy::SaveCursor;
pub use crate::pool::BlockPool;
pub use crate::pool::KvBlockGuard;
pub use crate::pool::LoadReservation;
pub use crate::pool::PAD_TOKEN_ID;
pub(crate) use crate::pool::PrefixProbe;
pub use crate::pool::RequestKv;
pub use crate::pool::resolved_page_ids;
pub(crate) use crate::stats::DegradeReason;
pub use crate::stats::KvStoreStats;
pub use crate::store::KvStore;
