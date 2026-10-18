//! Restore-commit path: reserved blocks staged + registered under the probe's
//! continuation hashes must be re-matchable as one contiguous prefix. Pure
//! logical-pool tests, no GPU. Also serves as the registration-cost regression
//! anchor for #704 (a 1024-block commit is ~1.6ms after the registry PRT fix;
//! it was 36ms before).

use pegainfer_kv_cache::BlockPool;

const BLOCK_SIZE: usize = 16;

#[test]
fn commit_absorbs_full_prefix() {
    let pool = BlockPool::new(BLOCK_SIZE, 64).expect("pool");
    // 33 full blocks + 1 forwarded token → cacheable = 33.
    let prompt: Vec<u32> = (0..=(33 * BLOCK_SIZE as u32)).map(|i| i % 97).collect();

    let mut probe = pool.probe_prefix(prompt.clone(), None);
    assert_eq!(probe.gpu_hit_blocks(), 0);
    assert_eq!(probe.cpu_query_window(), 33);

    let reservation = pool.reserve_loaded_blocks(33).expect("reserve");
    pool.commit_loaded_blocks(&mut probe, reservation);
    assert_eq!(probe.held_blocks(), 33);

    // A request over the same prompt reuses the whole restored prefix.
    let mut req = pool.new_request(prompt, 4, None);
    let matched = req.match_and_add_prefix(&pool).expect("match");
    assert_eq!(matched, 33 * BLOCK_SIZE);
}

#[test]
fn commit_extends_partial_gpu_hit() {
    let pool = BlockPool::new(BLOCK_SIZE, 64).expect("pool");
    let prompt: Vec<u32> = (0..=(8 * BLOCK_SIZE as u32)).map(|i| i % 89).collect();

    // Seed the GPU cache with the first 3 blocks via a real request.
    let mut seed = pool.new_request(prompt[..3 * BLOCK_SIZE].to_vec(), 1, None);
    seed.schedule_prefill(3 * BLOCK_SIZE, &pool)
        .expect("seed schedule");
    seed.apply_prefill(1, &pool).expect("seed apply");
    seed.release().expect("seed release");

    // Probe sees the 3-block GPU hit; the restore covers the remaining 5 and
    // must register them under the continuation hashes past the hit.
    let mut probe = pool.probe_prefix(prompt.clone(), None);
    assert_eq!(probe.gpu_hit_blocks(), 3);
    assert_eq!(probe.cpu_query_window(), 5);

    let reservation = pool.reserve_loaded_blocks(5).expect("reserve");
    pool.commit_loaded_blocks(&mut probe, reservation);
    assert_eq!(probe.held_blocks(), 8);

    let mut req = pool.new_request(prompt, 4, None);
    let matched = req.match_and_add_prefix(&pool).expect("match");
    assert_eq!(matched, 8 * BLOCK_SIZE);
}
