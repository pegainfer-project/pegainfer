//! [`KvStore`] itself: the resolve/seal/retire orchestration over the
//! per-rank surfaces frozen by [`crate::KvStoreBuilder`].

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;

use pegainfer_frontend::engine::KvPrefix;
use tokio::sync::oneshot;
use tokio::time::Instant;
use tokio_util::task::TaskTracker;

use crate::BlockPool;
use crate::CacheScope;
use crate::CancelProbe;
use crate::DegradeReason;
use crate::KvStoreStats;
use crate::PrefixProbe;
use crate::RequestKv;
use crate::ResolvePolicy;
use crate::SaveClass;
use crate::SaveCursor;
use crate::tier::HostTier;
use crate::tier::LeasedBlocks;
use crate::tier::TierQuery;

/// One rank's KV surfaces. `pool` is the same logical pool the rank's
/// scheduler allocates from; `BlockPool` is internally synchronized, so the
/// resolve task may allocate concurrently with the scheduler without a lock.
pub(crate) struct RankState {
    pub(crate) pool: Arc<BlockPool>,
    pub(crate) tier: Option<Arc<dyn HostTier>>,
    /// Blocks pinned by in-flight saves (both classes) — physically
    /// unallocatable until their D2H lands. Admission subtracts this from
    /// its budget. `Arc`: the per-save watcher tasks decrement it after the
    /// store's borrow ends.
    pub(crate) pinned: Arc<AtomicUsize>,
    /// In-flight restore H2Ds, including deadline-abandoned ones. Drained
    /// by [`KvStore::flush_loads`] before the GPU arenas are freed.
    pub(crate) loads: TaskTracker,
}

/// A tier hit whose lease releases on drop unless the caller takes it. The
/// detached query task returns its hit through this guard, which carries the
/// invariant: a query, once submitted, always settles and always releases
/// any lease the caller will never consume — a wait abandoned at the
/// deadline drops the task's output unread, and the drop releases.
struct LeaseGuard {
    tier: Arc<dyn HostTier>,
    hit: Option<LeasedBlocks>,
}

impl LeaseGuard {
    fn blocks(&self) -> usize {
        self.hit
            .as_ref()
            .expect("guard holds its hit until taken")
            .blocks
    }

    /// Consume the guard, taking over the lease (no release on drop).
    fn take(mut self) -> LeasedBlocks {
        self.hit.take().expect("guard holds its hit until taken")
    }
}

impl Drop for LeaseGuard {
    fn drop(&mut self) {
        if let Some(hit) = self.hit.take() {
            self.tier.release(hit);
        }
    }
}

/// One tier query's outcome, flat: the wait's three failure layers
/// (deadline, task panic, tier error) fold into `TimedOut`/`Failed`.
enum QueryOutcome {
    Miss,
    Loading,
    Hit(LeaseGuard),
    TimedOut,
    Failed(anyhow::Error),
}

/// The process-wide KV store: one instance, `Arc`-shared. Knows token
/// prefixes, pools, and tiers — not engines, inboxes, or requests. Built by
/// [`crate::KvStoreBuilder`]; the rank table is immutable after build.
pub struct KvStore {
    pub(crate) runtime: tokio::runtime::Handle,
    pub(crate) requery_interval: Duration,
    pub(crate) resolve_deadline: Duration,
    pub(crate) ranks: HashMap<usize, RankState>,
    pub(crate) stats: Arc<KvStoreStats>,
}

impl KvStore {
    /// Blocks pinned by `rank`'s in-flight saves; admission subtracts this
    /// from its usable budget, so it never admits a request against pages a
    /// pending D2H copy still needs.
    pub fn pinned_blocks(&self, rank: usize) -> usize {
        self.rank(rank).pinned.load(Ordering::Acquire)
    }

    pub fn stats(&self) -> &KvStoreStats {
        &self.stats
    }

    /// One tier query, run to settlement in a detached task so an abandoned
    /// wait leaves the settlement — and a late hit's lease release — to the
    /// task ([`LeaseGuard`] drops unread). Never await `tier.query` directly:
    /// dropping its future detaches only the observer, not the tier's work.
    async fn guarded_query(
        &self,
        tier: &Arc<dyn HostTier>,
        req_id: &str,
        block_hashes: Vec<Vec<u8>>,
        wait_full: bool,
        deadline: Instant,
    ) -> QueryOutcome {
        let tier = Arc::clone(tier);
        let req_id = req_id.to_string();
        let join = self.runtime.spawn(async move {
            match tier.query(&req_id, block_hashes, wait_full).await {
                Ok(TierQuery::Miss) => QueryOutcome::Miss,
                Ok(TierQuery::Loading) => QueryOutcome::Loading,
                Ok(TierQuery::Hit(hit)) => QueryOutcome::Hit(LeaseGuard {
                    tier,
                    hit: Some(hit),
                }),
                Err(err) => QueryOutcome::Failed(err),
            }
        });
        match tokio::time::timeout_at(deadline, join).await {
            Err(_elapsed) => QueryOutcome::TimedOut,
            Ok(Err(join_err)) => {
                QueryOutcome::Failed(anyhow::anyhow!("tier query task failed: {join_err}"))
            }
            Ok(Ok(outcome)) => outcome,
        }
    }

    /// The whole read path. Resolves `prompt_tokens`' cached prefix on
    /// `rank`: GPU radix probe, host-tier query (re-query while a deeper tier
    /// fetches, bounded by the resolve deadline), page reservation, load, and
    /// registration into the radix — so the eventual `match_and_add_prefix`
    /// on the scheduler thread reuses the full prefix.
    ///
    /// One terminal type: a [`KvPrefix`] whose hold keeps the resolved blocks
    /// resident until the scheduler's match consumes them. Degraded outcomes
    /// (deadline, tier error, pool pressure) return the GPU hit alone and
    /// report to stats; a cancelled request returns [`KvPrefix::none`] — it
    /// dies at admission's existing closed-sink check.
    pub async fn resolve_prefix(
        &self,
        rank: usize,
        req_id: &str,
        prompt_tokens: &[u32],
        scope: CacheScope<'_>,
        policy: ResolvePolicy,
        cancel: &dyn CancelProbe,
    ) -> KvPrefix {
        self.stats.resolves.fetch_add(1, Ordering::Relaxed);
        let state = self.rank(rank);
        let pool = &state.pool;
        let block_size = pool.block_size();

        if cancel.is_cancelled() {
            self.stats.record_degrade(req_id, DegradeReason::Cancelled);
            return KvPrefix::none();
        }
        let mut probe = pool.probe_prefix_with_cache_salt(
            prompt_tokens.to_vec(),
            scope.cache_salt,
            scope.lora_name,
        );
        if policy.full_pages {
            probe.include_final_block();
        }
        let cacheable_blocks = probe.cacheable_blocks();

        let finish = |mut probe: PrefixProbe| {
            let hit_blocks = probe.held_blocks().min(cacheable_blocks);
            if hit_blocks == 0 {
                return KvPrefix::none();
            }
            // Credit/pin parity: admission credits the hold with exactly
            // `hit_tokens / block_size` blocks, so the hold must pin exactly
            // that many. A page-aligned full match holds one block past the
            // cacheable cap; keeping it pinned would let a front at exact
            // pool capacity wait forever on the very block its own hold pins.
            probe.truncate_held(hit_blocks);
            self.stats.resolve_hits.fetch_add(1, Ordering::Relaxed);
            KvPrefix::resolved(hit_blocks * block_size, rank, Box::new(probe))
        };

        let host_hashes = probe.cpu_query_hashes();
        let tier = match state.tier.as_ref() {
            Some(tier) if !host_hashes.is_empty() => tier,
            _ => return finish(probe),
        };

        // Host-tier query, re-queried until the hit is BOTH host-ready and
        // pool-admittable, bounded by one deadline. Waiting on pool pressure
        // (instead of instantly degrading) is deliberate: if the pool cannot
        // spare the prefix's blocks now, admission could not take the request
        // now either — it waits regardless, and waiting here buys the cheap
        // prefill while degrading would buy a full recompute for the same
        // wait. The lease is released before every pause (host pins + TTL
        // must not ride out a pool wait; a re-query re-establishes it
        // instantly from the host tier). Each query runs detached (see
        // `spawn_guarded_query`): a wait abandoned at the deadline leaves
        // the settlement — and the release of a late hit's lease — to the
        // detached task.
        let deadline = Instant::now() + self.resolve_deadline;
        let wait_start = Instant::now();
        let mut polls = 0u32;
        let (hit, reservation) = loop {
            polls += 1;
            if cancel.is_cancelled() {
                self.stats.record_degrade(req_id, DegradeReason::Cancelled);
                return KvPrefix::none();
            }
            let outcome = self
                .guarded_query(
                    tier,
                    req_id,
                    host_hashes.clone(),
                    policy.wait_for_full_hit,
                    deadline,
                )
                .await;
            match outcome {
                QueryOutcome::TimedOut => {
                    self.stats
                        .record_degrade(req_id, DegradeReason::DeadlineExceeded);
                    return finish(probe);
                }
                QueryOutcome::Failed(err) => {
                    log::warn!("kv-store resolve {req_id}: tier query failed: {err:#}");
                    self.stats.record_degrade(req_id, DegradeReason::TierError);
                    return finish(probe);
                }
                QueryOutcome::Miss => {
                    // Plain restore: a miss is a cold cache, conclude. Full-
                    // hit mode: the producer's registration may not have
                    // landed yet — keep waiting under the deadline.
                    if !policy.wait_for_full_hit {
                        return finish(probe);
                    }
                    if Instant::now() >= deadline {
                        self.stats
                            .record_degrade(req_id, DegradeReason::DeadlineExceeded);
                        return finish(probe);
                    }
                    tokio::time::sleep(self.requery_interval).await;
                }
                QueryOutcome::Loading => {
                    if Instant::now() >= deadline {
                        self.stats
                            .record_degrade(req_id, DegradeReason::DeadlineExceeded);
                        return finish(probe);
                    }
                    tokio::time::sleep(self.requery_interval).await;
                }
                QueryOutcome::Hit(guard) => {
                    // The lease is all-or-nothing, so a hit that doesn't fit
                    // the pool right now is declined (the guard's drop
                    // releases, not TTL-parked) and retried after a pause.
                    if let Some(reservation) = pool.reserve_loaded_blocks(guard.blocks()) {
                        break (guard.take(), reservation);
                    }
                    drop(guard);
                    if Instant::now() >= deadline {
                        self.stats
                            .record_degrade(req_id, DegradeReason::PoolPressure);
                        return finish(probe);
                    }
                    tokio::time::sleep(self.requery_interval).await;
                }
            }
        };
        if cancel.is_cancelled() {
            tier.release(hit);
            self.stats.record_degrade(req_id, DegradeReason::Cancelled);
            return KvPrefix::none();
        }

        // The DMA is uncancellable: the tracked task owns the reservation
        // until the tier settles; the deadline abandons only the wait.
        let wait_ms = wait_start.elapsed().as_millis();
        let loaded = hit.blocks;
        let page_ids = reservation.page_ids();
        let load = tier.load(hit, page_ids);
        let join = self.runtime.spawn(state.loads.track_future(async move {
            let result = load.await;
            (result, reservation)
        }));
        let load_start = Instant::now();
        match tokio::time::timeout_at(deadline, join).await {
            Ok(Ok((Ok(()), reservation))) => {
                let load_ms = load_start.elapsed().as_millis();
                let commit_start = Instant::now();
                pool.commit_loaded_blocks(&mut probe, reservation);
                // Stage split for the slow tail: `wait` is producer latency
                // (seal/registration not landed, or pool pressure), `load` is
                // the host->device move, `commit` is radix insertion — three
                // different owners, unreadable from the total alone.
                let commit_ms = commit_start.elapsed().as_millis();
                if wait_ms + load_ms + commit_ms > 50 {
                    log::info!(
                        "kv-store resolve timing: req={req_id} rank={rank} blocks={loaded} \
                         wait={wait_ms}ms polls={polls} load={load_ms}ms commit={commit_ms}ms"
                    );
                }
                self.stats
                    .resolve_loaded_blocks
                    .fetch_add(loaded as u64, Ordering::Relaxed);
                finish(probe)
            }
            Ok(Ok((Err(err), _reservation))) => {
                log::warn!("kv-store resolve {req_id}: tier load failed: {err:#}");
                self.stats.record_degrade(req_id, DegradeReason::TierError);
                // The load settled; dropping the reservation here returns the
                // destination blocks untouched by any registered hash.
                finish(probe)
            }
            Ok(Err(join_err)) => {
                log::warn!("kv-store resolve {req_id}: load task failed: {join_err}");
                self.stats.record_degrade(req_id, DegradeReason::TierError);
                finish(probe)
            }
            Err(_elapsed) => {
                // The reservation now lives with the detached task until the
                // DMA settles; if it never does, those blocks are gone for
                // good — count it so pool-drain from hung DMAs is visible.
                self.stats.loads_abandoned.fetch_add(1, Ordering::Relaxed);
                self.stats
                    .record_degrade(req_id, DegradeReason::DeadlineExceeded);
                finish(probe)
            }
        }
    }

    /// Save the request's freshly-sealed blocks past the cursor. Call after
    /// the step that sealed them has synchronized (the tier reads the GPU
    /// asynchronously — the ordering contract passes through). Guards pin the
    /// source pages until the D2H lands; the pinned pages are visible to
    /// admission via [`Self::pinned_blocks`].
    pub fn seal(&self, rank: usize, kv: &RequestKv, cursor: &mut SaveCursor, class: SaveClass) {
        let state = self.rank(rank);
        let Some(tier) = state.tier.as_ref() else {
            return;
        };
        if cursor.saved_blocks == 0 {
            // Prefix-hit blocks were stored by whoever first sealed them.
            cursor.saved_blocks = kv.prefix_matched_blocks();
        }
        let pairs = kv.assigned_block_hashes();
        if pairs.len() <= cursor.saved_blocks {
            return;
        }
        let count = pairs.len() - cursor.saved_blocks;

        // Guards align 1:1 with `assigned_block_hashes`.
        let guards: Vec<_> = kv
            .assigned_block_guards()
            .into_iter()
            .skip(cursor.saved_blocks)
            .collect();
        let (ids, hashes): (Vec<i32>, Vec<Vec<u8>>) = pairs[cursor.saved_blocks..]
            .iter()
            .map(|(id, hash)| (*id, hash.to_vec()))
            .unzip();
        cursor.saved_blocks = pairs.len();

        state.pinned.fetch_add(count, Ordering::AcqRel);
        self.stats.saves_submitted.fetch_add(1, Ordering::Relaxed);
        let save = tier.save(ids, hashes, guards);

        let done_tx = if class == SaveClass::Handoff {
            let (tx, rx) = oneshot::channel();
            cursor.pending.push(rx);
            Some(tx)
        } else {
            None
        };
        let pinned = Arc::clone(&state.pinned);
        let stats = Arc::clone(&self.stats);
        self.runtime.spawn(async move {
            let result = save.await;
            pinned.fetch_sub(count, Ordering::AcqRel);
            let outcome = match result {
                Ok(()) => Ok(()),
                Err(err) => {
                    stats.saves_failed.fetch_add(1, Ordering::Relaxed);
                    log::warn!("kv-store save of {count} blocks failed: {err:#}");
                    Err(format!("{err:#}"))
                }
            };
            if let Some(tx) = done_tx {
                let _ = tx.send(outcome);
            }
        });
    }

    /// Final seal + release. With no must-complete saves outstanding the KV
    /// releases immediately; fire-and-forget D2H stays safe because the save
    /// guards pin the source pages independently of the release. With
    /// `Handoff` saves pending, the KV instead stays parked — registered and
    /// resident — until they settle. Never blocks the caller.
    pub fn retire(&self, rank: usize, mut kv: RequestKv, mut cursor: SaveCursor, class: SaveClass) {
        self.seal(rank, &kv, &mut cursor, class);
        if cursor.pending.is_empty() {
            release_logged(&mut kv);
            return;
        }
        self.stats.retires_parked.fetch_add(1, Ordering::Relaxed);
        let stats = Arc::clone(&self.stats);
        self.runtime.spawn(async move {
            let mut failed = false;
            for rx in cursor.pending {
                if !matches!(rx.await, Ok(Ok(()))) {
                    failed = true;
                }
            }
            if failed {
                // The checkpoint the consuming peer expects is missing; it
                // sees this handoff as a short hit and rejects it. The
                // producing scheduler should eventually withhold its
                // KV-ready response until these saves confirm — that wiring
                // is not in place yet, so for now the failure is counted and
                // logged loudly.
                stats.handoff_failed.fetch_add(1, Ordering::Relaxed);
                log::error!("kv-store retire: handoff save failed; peer will miss this checkpoint");
            }
            release_logged(&mut kv);
        });
    }

    /// Write-pipeline visibility barrier for `rank`: on return, every
    /// [`Self::seal`]/[`Self::retire`] save submitted before this call is
    /// query-visible to this tier (and, with P2P, MetaServer-registered for
    /// peers). The `Handoff` save completion callback answers "data landed";
    /// this barrier answers "findable by hash". A rank without a tier has
    /// nothing to flush.
    pub async fn flush_saves(&self, rank: usize) -> anyhow::Result<()> {
        match self.rank(rank).tier.as_ref() {
            Some(tier) => tier.flush().await,
            None => Ok(()),
        }
    }

    /// Terminal barrier: resolves once every restore H2D on `rank` has
    /// settled. Call before freeing the GPU arenas; bound with a caller-side
    /// timeout (a hung tier never settles). Closes the rank's tracker.
    pub async fn flush_loads(&self, rank: usize) {
        let loads = &self.rank(rank).loads;
        loads.close();
        loads.wait().await;
    }

    /// The rank table is frozen at build, so an unknown rank is a wiring
    /// bug — masking it (e.g. silently no-op'ing a Handoff seal) would
    /// violate the no-silent-drop contract. Fail fast instead.
    fn rank(&self, rank: usize) -> &RankState {
        self.ranks.get(&rank).unwrap_or_else(|| {
            panic!("kv-store: rank {rank} not registered (rank table is frozen at build)")
        })
    }
}

fn release_logged(kv: &mut RequestKv) {
    if let Err(err) = kv.release() {
        // Blocks still return via assignment RAII when the KV drops; the
        // explicit release only failed to run from a clean state.
        log::warn!("kv-store retire: KV release failed (blocks return via RAII): {err:#}");
    }
}

// The store's whole concurrency story rests on these being thread-safe;
// fail at compile time, not in a review.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<KvStore>();
};

#[cfg(test)]
mod tests {
    use pegaflow_core::QueryLeaseId;
    use tokio::sync::Notify;

    use super::*;
    use crate::BlockPool;
    use crate::KvBlockGuard;
    use crate::NeverCancelled;
    use crate::tier::HostTier;
    use crate::tier::LeasedBlocks;
    use crate::tier::TierFuture;
    use crate::tier::TierQuery;

    /// A tier whose query always hits but whose load stalls until the test
    /// fires `release_load` — the hung-storage-worker shape the load
    /// deadline exists for.
    struct StallLoadTier {
        release_load: Arc<Notify>,
    }

    impl HostTier for StallLoadTier {
        fn query(
            &self,
            _req_id: &str,
            block_hashes: Vec<Vec<u8>>,
            _wait_full: bool,
        ) -> TierFuture<anyhow::Result<TierQuery>> {
            let blocks = block_hashes.len();
            Box::pin(std::future::ready(Ok(TierQuery::Hit(LeasedBlocks {
                blocks,
                lease: QueryLeaseId::fresh(),
            }))))
        }

        fn load(
            &self,
            _hit: LeasedBlocks,
            _dst_page_ids: Vec<i32>,
        ) -> TierFuture<anyhow::Result<()>> {
            let release = Arc::clone(&self.release_load);
            Box::pin(async move {
                release.notified().await;
                Ok(())
            })
        }

        fn release(&self, _hit: LeasedBlocks) {}

        fn save(
            &self,
            _block_ids: Vec<i32>,
            _block_hashes: Vec<Vec<u8>>,
            _guards: Vec<KvBlockGuard>,
        ) -> TierFuture<anyhow::Result<()>> {
            Box::pin(std::future::ready(Ok(())))
        }

        fn flush(&self) -> TierFuture<anyhow::Result<()>> {
            Box::pin(std::future::ready(Ok(())))
        }
    }

    /// A tier whose query stalls until the test fires `release_query`, then
    /// answers with a lease-carrying hit — the shape where the caller's
    /// deadline passes while the tier still owes a settlement. `released`
    /// counts every lease handed back.
    struct StallQueryTier {
        release_query: Arc<Notify>,
        released: Arc<AtomicUsize>,
    }

    impl HostTier for StallQueryTier {
        fn query(
            &self,
            _req_id: &str,
            block_hashes: Vec<Vec<u8>>,
            _wait_full: bool,
        ) -> TierFuture<anyhow::Result<TierQuery>> {
            let release = Arc::clone(&self.release_query);
            let blocks = block_hashes.len();
            Box::pin(async move {
                release.notified().await;
                Ok(TierQuery::Hit(LeasedBlocks {
                    blocks,
                    lease: QueryLeaseId::fresh(),
                }))
            })
        }

        fn load(
            &self,
            _hit: LeasedBlocks,
            _dst_page_ids: Vec<i32>,
        ) -> TierFuture<anyhow::Result<()>> {
            Box::pin(std::future::ready(Ok(())))
        }

        fn release(&self, _hit: LeasedBlocks) {
            self.released.fetch_add(1, Ordering::AcqRel);
        }

        fn save(
            &self,
            _block_ids: Vec<i32>,
            _block_hashes: Vec<Vec<u8>>,
            _guards: Vec<KvBlockGuard>,
        ) -> TierFuture<anyhow::Result<()>> {
            Box::pin(std::future::ready(Ok(())))
        }

        fn flush(&self) -> TierFuture<anyhow::Result<()>> {
            Box::pin(std::future::ready(Ok(())))
        }
    }

    fn tier_store_with_deadline(
        runtime: tokio::runtime::Handle,
        tier: Arc<dyn HostTier>,
        resolve_deadline: Duration,
    ) -> KvStore {
        let mut ranks = HashMap::new();
        ranks.insert(
            0,
            RankState {
                pool: Arc::new(BlockPool::new(16, 8)),
                tier: Some(tier),
                pinned: Arc::new(AtomicUsize::new(0)),
                loads: TaskTracker::new(),
            },
        );
        KvStore {
            runtime,
            requery_interval: Duration::from_millis(1),
            resolve_deadline,
            ranks,
            stats: Arc::new(KvStoreStats::default()),
        }
    }

    fn tier_store(runtime: tokio::runtime::Handle, tier: Arc<dyn HostTier>) -> KvStore {
        tier_store_with_deadline(runtime, tier, Duration::from_millis(50))
    }

    fn stalled_store(runtime: tokio::runtime::Handle, release_load: Arc<Notify>) -> KvStore {
        tier_store(runtime, Arc::new(StallLoadTier { release_load }))
    }

    /// The plain-path query invariant: `resolve_prefix`'s
    /// host query abandoned at the deadline leaves settlement to the
    /// detached task, and a late hit's lease is released — a storage stall
    /// never pins host blocks until the lease TTL.
    #[test]
    fn plain_query_timeout_releases_the_late_hits_lease() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        let release_query = Arc::new(Notify::new());
        let released = Arc::new(AtomicUsize::new(0));
        let store = tier_store(
            rt.handle().clone(),
            Arc::new(StallQueryTier {
                release_query: Arc::clone(&release_query),
                released: Arc::clone(&released),
            }),
        );

        // One cacheable block past the (empty) GPU hit, so the plain resolve
        // reaches the host-tier query.
        let prompt: Vec<u32> = (0..17).collect();
        let prefix = rt.block_on(store.resolve_prefix(
            0,
            "plain",
            &prompt,
            crate::CacheScope::default(),
            crate::ResolvePolicy::default(),
            &NeverCancelled,
        ));
        assert_eq!(prefix.hit_tokens(), 0, "a stalled query degrades to no hit");
        assert_eq!(store.stats.resolve_degraded.load(Ordering::Relaxed), 1);
        assert_eq!(
            released.load(Ordering::Acquire),
            0,
            "nothing to release while the query is unsettled"
        );

        release_query.notify_one();
        rt.block_on(async {
            tokio::time::timeout(Duration::from_secs(5), async {
                while released.load(Ordering::Acquire) == 0 {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            })
            .await
        })
        .expect("the settled query releases the unconsumed lease");
        assert_eq!(released.load(Ordering::Acquire), 1);
    }

    /// The plain-path load invariant: a stalled load abandons only the WAIT
    /// at the deadline (the resolve degrades to no hit; the resolver task is
    /// never wedged), while the detached task keeps owning the load — and
    /// its destination reservation — until the tier settles, visible through
    /// `loads_pending` and drained by `flush_loads`.
    #[test]
    fn plain_load_timeout_detaches_and_flush_loads_drains() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("test runtime");
        let release_load = Arc::new(Notify::new());
        let store = stalled_store(rt.handle().clone(), Arc::clone(&release_load));

        // One cacheable block past the (empty) GPU hit, so the plain resolve
        // reaches the tier: the query hits instantly, the load stalls.
        let prompt: Vec<u32> = (0..17).collect();
        let prefix = rt.block_on(store.resolve_prefix(
            0,
            "plain",
            &prompt,
            crate::CacheScope::default(),
            crate::ResolvePolicy::default(),
            &NeverCancelled,
        ));
        assert_eq!(prefix.hit_tokens(), 0, "a stalled load degrades to no hit");
        assert_eq!(store.stats.resolve_degraded.load(Ordering::Relaxed), 1);
        assert_eq!(store.stats.loads_abandoned.load(Ordering::Relaxed), 1);
        assert_eq!(
            store.rank(0).loads.len(),
            1,
            "the detached task owns the load"
        );

        release_load.notify_one();
        rt.block_on(async {
            tokio::time::timeout(Duration::from_secs(5), store.flush_loads(0)).await
        })
        .expect("the load settles once the tier releases it");
        assert_eq!(store.rank(0).loads.len(), 0, "flush_loads drained the load");
    }
}
