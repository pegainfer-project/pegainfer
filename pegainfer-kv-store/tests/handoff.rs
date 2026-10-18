//! Retire/handoff: the must-complete save parks the retiring KV until the
//! D2H settles, and nothing returns to the pool early.

mod common;

use std::sync::atomic::Ordering;

use common::HOST_POOL_BYTES;
use common::RANK;
use common::Rig;
use common::gpu_lock;
use common::prompt;
use common::wait_until;
use pegainfer_kv_store::PegaflowHost;
use pegainfer_kv_store::SaveClass;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn retire_handoff_parks_until_save_settles() {
    let _gpu = gpu_lock().lock().await;
    let host = PegaflowHost::builder(HOST_POOL_BYTES)
        .build()
        .expect("host");
    let rig = Rig::new("retire_handoff_parks", host, None);
    let prompt = prompt(3);

    let available_before = rig.pool.available_blocks();
    rig.run_and_retire(&prompt, SaveClass::Handoff);

    assert_eq!(
        rig.store.stats().retires_parked.load(Ordering::Relaxed),
        1,
        "a Handoff retire with a save in flight parks the KV"
    );
    // While the must-complete save is in flight the parked KV must stay out
    // of the pool. A real engine's 1.5 KiB D2H can settle before the first
    // sample, so this observation is conditional; the recovery below is not.
    if rig.store.pinned_blocks(RANK) > 0 {
        assert!(
            rig.pool.available_blocks() < available_before,
            "the parked KV must not return while its save is in flight"
        );
    }
    wait_until("handoff save settled", || {
        rig.store.pinned_blocks(RANK) == 0
    })
    .await;
    wait_until("parked KV returned to the pool", || {
        // Released blocks land in the inactive prefix cache, which still
        // counts as available capacity.
        rig.pool.available_blocks() >= available_before
    })
    .await;
    assert_eq!(
        rig.store.stats().handoff_failed.load(Ordering::Relaxed),
        0,
        "a healthy engine never fails a handoff checkpoint"
    );
}
