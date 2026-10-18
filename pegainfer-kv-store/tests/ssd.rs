//! The SSD tier: spill/reclaim/prefetch against a real io_uring-backed cache
//! file, cold-miss conclusion through the `Loading` re-query, and the
//! disaggregated prefill/decode handoff for real.

mod common;

use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use common::BLOCK_TOKENS;
use common::HOST_POOL_BYTES;
use common::NUM_LAYERS;
use common::RANK;
use common::Rig;
use common::SEGMENT_BYTES;
use common::degraded;
use common::gpu_lock;
use common::io_uring_available;
use common::loaded_blocks;
use common::prefill;
use common::prompt;
use common::prompt_salted;
use common::wait_until;
use pegainfer_kv_store::CacheScope;
use pegainfer_kv_store::NeverCancelled;
use pegainfer_kv_store::PegaflowHost;
use pegainfer_kv_store::ResolvePolicy;
use pegainfer_kv_store::SaveClass;
use pegainfer_kv_store::SaveCursor;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ssd_spill_prefetch_roundtrip() {
    const SAVE_FOOTPRINT: usize = NUM_LAYERS * 4 * SEGMENT_BYTES;

    let _gpu = gpu_lock().lock().await;
    if !io_uring_available() {
        eprintln!("skipping: io_uring is not available in this environment");
        return;
    }

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let cache_path = temp_dir.path().join("cache.bin");
    // Pinned pool sized for exactly one save's footprint (4 layers x 4 blocks
    // x 512 B, one batched 2 KiB pinned alloc per layer) plus 1 KiB slack —
    // too little for even one layer of a second save, so saving B must
    // reclaim A out of the read cache; A then survives only on SSD.
    let host = PegaflowHost::builder(SAVE_FOOTPRINT + 1024)
        .ssd_cache(vec![cache_path], 64 << 20)
        .build()
        .expect("host with ssd cache");
    let rig = Rig::new("ssd_spill_prefetch", host, None);

    let prompt_a = prompt_salted(4, 100_000);
    let prompt_b = prompt_salted(4, 200_000);

    rig.run_and_retire(&prompt_a, SaveClass::Cacheable);
    rig.store
        .flush_saves(RANK)
        .await
        .expect("flush A (visibility)");
    // Also drain the SSD writer: after this, A persists on disk regardless of
    // what happens to its memory-tier copy.
    rig.host.flush_all().await;

    rig.run_and_retire(&prompt_b, SaveClass::Cacheable);
    // B's save settling implies its pinned allocations succeeded, which
    // required reclaiming A from the read cache.
    wait_until("B's save settled (A reclaimed to SSD)", || {
        rig.store.pinned_blocks(RANK) == 0
    })
    .await;
    assert_eq!(rig.store.stats().saves_failed.load(Ordering::Relaxed), 0);

    rig.pool.evict_inactive();

    // A is resident neither on the GPU nor in host RAM: the first tier query
    // spawns pegaflow's SSD prefetch (Loading) and the store's re-query loop
    // rides it to Ready, then loads. One resolve, full hit.
    let prefix = rig
        .store
        .resolve_prefix(
            RANK,
            "r-ssd",
            &prompt_a,
            CacheScope::default(),
            ResolvePolicy::default(),
            &NeverCancelled,
        )
        .await;
    assert_eq!(prefix.hit_tokens(), 4 * BLOCK_TOKENS);
    assert_eq!(loaded_blocks(&rig), 4);
    assert_eq!(degraded(&rig), 0);
    drop(prefix);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ssd_cold_miss_terminates_without_degrade() {
    let _gpu = gpu_lock().lock().await;
    if !io_uring_available() {
        eprintln!("skipping: io_uring is not available in this environment");
        return;
    }

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let cache_path = temp_dir.path().join("cache.bin");
    let host = PegaflowHost::builder(HOST_POOL_BYTES)
        .ssd_cache(vec![cache_path], 64 << 20)
        .build()
        .expect("host with ssd cache");
    let rig = Rig::new("ssd_cold_miss", host, None);

    // Only the 2-block head of this 6-block prompt lives anywhere (GPU radix;
    // never saved). Under an SSD-backed host a tier query first answers
    // Loading while the SSD prefetch runs, and once that proves empty the
    // answer is Ready(nothing) — a cold cache, concluded: the resolve returns
    // the GPU hit at once and counts NO degrade.
    let head = prompt(2);
    prefill(&rig.pool, &head).release().expect("release");
    let prompt = prompt(6);

    let prefix = rig
        .store
        .resolve_prefix(
            RANK,
            "r-cold",
            &prompt,
            CacheScope::default(),
            ResolvePolicy::default(),
            &NeverCancelled,
        )
        .await;
    assert_eq!(prefix.hit_tokens(), 2 * BLOCK_TOKENS);
    assert_eq!(degraded(&rig), 0, "Loading -> empty Ready is not a degrade");
    assert_eq!(loaded_blocks(&rig), 0);
    drop(prefix);
}

/// The disaggregated handoff shape, for real: the decode side's resolve is
/// already waiting when the prefill side's checkpoint lands — a miss under
/// the all-or-nothing policy means "not yet", so the re-query loop rides out
/// the producer's save latency and still completes with the full hit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wait_full_hit_waits_for_producer_registration() {
    let _gpu = gpu_lock().lock().await;
    if !io_uring_available() {
        eprintln!("skipping: io_uring is not available in this environment");
        return;
    }

    let temp_dir = tempfile::tempdir().expect("temp dir");
    let cache_path = temp_dir.path().join("cache.bin");
    let host = PegaflowHost::builder(HOST_POOL_BYTES)
        .ssd_cache(vec![cache_path], 64 << 20)
        .build()
        .expect("host with ssd cache");
    let rig = Rig::new("wait_full_registration", host, Some(Duration::from_secs(1)));
    let prompt = prompt(4);

    let store = Arc::clone(&rig.store);
    let pool = Arc::clone(&rig.pool);
    let produced = prompt.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(40)).await;
        let kv = prefill(&pool, &produced);
        store.retire(RANK, kv, SaveCursor::new(), SaveClass::Cacheable);
        store.flush_saves(RANK).await.expect("flush");
    });
    let prefix = rig
        .store
        .resolve_prefix(
            RANK,
            "r-decode",
            &prompt,
            CacheScope::default(),
            ResolvePolicy::default().wait_for_full_hit(),
            &NeverCancelled,
        )
        .await;
    assert_eq!(prefix.hit_tokens(), 4 * BLOCK_TOKENS);
    assert_eq!(loaded_blocks(&rig), 4);
    assert_eq!(degraded(&rig), 0);
    drop(prefix);
}
