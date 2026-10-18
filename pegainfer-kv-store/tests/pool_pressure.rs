//! Pool pressure under a real tier: a resolve that finds its hit but not
//! enough free GPU pages waits for the pool to drain, and degrades only when
//! the deadline actually expires. The pressure is made the honest way —
//! real pool reservations held out of circulation — never by watermark
//! bookkeeping.

mod common;

use std::sync::atomic::Ordering;
use std::time::Duration;

use common::BLOCK_TOKENS;
use common::HOST_POOL_BYTES;
use common::RANK;
use common::Rig;
use common::degraded;
use common::gpu_lock;
use common::loaded_blocks;
use common::prompt;
use pegainfer_kv_store::CacheScope;
use pegainfer_kv_store::NeverCancelled;
use pegainfer_kv_store::PegaflowHost;
use pegainfer_kv_store::ResolvePolicy;
use pegainfer_kv_store::SaveClass;

/// A warm host hit with the GPU radix empty, ready for pressure games.
fn warm_host_hit(rig: &Rig, full_blocks: usize) -> Vec<u32> {
    let prompt = prompt(full_blocks);
    rig.run_and_retire(&prompt, SaveClass::Cacheable);
    prompt
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pool_pressure_waits_then_completes() {
    let _gpu = gpu_lock().lock().await;
    let host = PegaflowHost::builder(HOST_POOL_BYTES)
        .build()
        .expect("host");
    let rig = Rig::new("pool_pressure_waits", host, None);
    let prompt = warm_host_hit(&rig, 5);
    rig.store.flush_saves(RANK).await.expect("flush");
    rig.pool.evict_inactive();

    // Hold real pool pages so that one block short of the hit's need stays
    // free: the resolve's own reservation cannot fit and it waits. The
    // ballast is dropped after a pause, draining the pressure for real.
    let needed = 5;
    let ballast = rig
        .pool
        .reserve_loaded_blocks(rig.pool.available_blocks() - (needed - 1))
        .expect("ballast reservation");
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(15)).await;
        drop(ballast);
    });

    let prefix = rig
        .store
        .resolve_prefix(
            RANK,
            "r1",
            &prompt,
            CacheScope::default(),
            ResolvePolicy::default(),
            &NeverCancelled,
        )
        .await;
    // Pressure that clears within the deadline costs latency, never the hit —
    // even though every re-query round had to release its real engine lease
    // before pausing.
    assert_eq!(prefix.hit_tokens(), needed * BLOCK_TOKENS);
    assert_eq!(loaded_blocks(&rig), needed as u64);
    assert_eq!(degraded(&rig), 0);
    drop(prefix);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pool_pressure_degrades_at_deadline() {
    let _gpu = gpu_lock().lock().await;
    let host = PegaflowHost::builder(HOST_POOL_BYTES)
        .build()
        .expect("host");
    let rig = Rig::new(
        "pool_pressure_degrades",
        host,
        Some(Duration::from_millis(30)),
    );
    let prompt = warm_host_hit(&rig, 5);
    rig.store.flush_saves(RANK).await.expect("flush");
    rig.pool.evict_inactive();

    // Same four-free-blocks pressure, held for the whole resolve: it degrades
    // at the deadline, having declined (released, not stranded) every lease
    // it took while waiting.
    let _ballast = rig
        .pool
        .reserve_loaded_blocks(rig.pool.available_blocks() - 4)
        .expect("ballast reservation");

    let prefix = rig
        .store
        .resolve_prefix(
            RANK,
            "r1",
            &prompt,
            CacheScope::default(),
            ResolvePolicy::default(),
            &NeverCancelled,
        )
        .await;
    assert_eq!(prefix.hit_tokens(), 0);
    assert_eq!(degraded(&rig), 1);
    assert_eq!(loaded_blocks(&rig), 0);
    assert_eq!(
        rig.store.stats().loads_abandoned.load(Ordering::Relaxed),
        0,
        "no DMA was ever submitted, so no reservation can be abandoned"
    );
    drop(prefix);
}
