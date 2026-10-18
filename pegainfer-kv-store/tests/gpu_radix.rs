//! GPU-radix-only service and resolve cancellation.
//!
//! No host tier involved except as the cancelled resolve's target: a rank
//! without offload still serves GPU radix hits, and a cancelled resolve must
//! consume nothing on its way out.

mod common;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use common::BLOCK_TOKENS;
use common::HOST_POOL_BYTES;
use common::NUM_BLOCKS;
use common::RANK;
use common::Rig;
use common::degraded;
use common::gpu_lock;
use common::loaded_blocks;
use common::prefill;
use common::prompt;
use pegainfer_kv_store::BlockPool;
use pegainfer_kv_store::CacheScope;
use pegainfer_kv_store::CancelProbe;
use pegainfer_kv_store::KvStoreBuilder;
use pegainfer_kv_store::NeverCancelled;
use pegainfer_kv_store::PegaflowHost;
use pegainfer_kv_store::ResolvePolicy;
use pegainfer_kv_store::SaveClass;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn no_tier_resolves_the_gpu_radix_only() {
    // A rank declared without offload still serves GPU radix hits; the hold
    // alone keeps them resident past eviction until the scheduler consumes.
    let pool = Arc::new(BlockPool::new(BLOCK_TOKENS, NUM_BLOCKS));
    let prompt = prompt(4);
    prefill(&pool, &prompt).release().expect("release");
    let store = KvStoreBuilder::new(tokio::runtime::Handle::current())
        .rank(RANK, Arc::clone(&pool))
        .build();

    let prefix = store
        .resolve_prefix(
            RANK,
            "r1",
            &prompt,
            CacheScope::default(),
            ResolvePolicy::default(),
            &NeverCancelled,
        )
        .await;
    assert_eq!(prefix.hit_tokens(), 4 * BLOCK_TOKENS);
    assert!(prefix.has_hold());
    assert_eq!(store.stats().resolve_degraded.load(Ordering::Relaxed), 0);

    pool.evict_inactive();
    let mut req = pool.new_request(prompt.clone(), 4, None);
    assert_eq!(
        req.match_and_add_prefix(&pool).expect("match"),
        4 * BLOCK_TOKENS,
        "the hold kept evict_inactive from claiming the hit blocks"
    );
    drop(prefix);
    req.release().expect("release");
}

struct AlwaysCancelled;

impl CancelProbe for AlwaysCancelled {
    fn is_cancelled(&self) -> bool {
        true
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_resolve_skips_io_and_leaves_no_state_behind() {
    let _gpu = gpu_lock().lock().await;
    let host = PegaflowHost::builder(HOST_POOL_BYTES)
        .build()
        .expect("host");
    let rig = Rig::new("cancelled_resolve", host, None);
    let prompt = prompt(4);

    // Make the prefix genuinely hittable, then drop all local evidence of it.
    rig.run_and_retire(&prompt, SaveClass::Cacheable);
    rig.store.flush_saves(RANK).await.expect("flush");
    rig.pool.evict_inactive();

    let prefix = rig
        .store
        .resolve_prefix(
            RANK,
            "r-cancelled",
            &prompt,
            CacheScope::default(),
            ResolvePolicy::default(),
            &AlwaysCancelled,
        )
        .await;
    assert_eq!(prefix.hit_tokens(), 0);
    assert!(!prefix.has_hold());
    assert_eq!(degraded(&rig), 1, "the cancel is the recorded degrade");
    assert_eq!(loaded_blocks(&rig), 0);

    // Nothing leaked or got consumed by mistake: the next resolve sees the
    // full hit as if the cancelled one never ran.
    let prefix = rig
        .store
        .resolve_prefix(
            RANK,
            "r-after",
            &prompt,
            CacheScope::default(),
            ResolvePolicy::default(),
            &NeverCancelled,
        )
        .await;
    assert_eq!(prefix.hit_tokens(), 4 * BLOCK_TOKENS);
    assert_eq!(loaded_blocks(&rig), 4);
    assert_eq!(degraded(&rig), 1);
    drop(prefix);
}
