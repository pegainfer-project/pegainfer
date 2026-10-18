//! Save → resolve roundtrips over the real host tier: byte-exact D2H→H2D
//! restoration, the all-or-nothing wait policy's cold/warm behaviour, and
//! prefix-skip seal semantics against real save visibility.

mod common;

use std::sync::atomic::Ordering;
use std::time::Duration;

use common::BLOCK_TOKENS;
use common::HOST_POOL_BYTES;
use common::NUM_LAYERS;
use common::RANK;
use common::Rig;
use common::SEGMENT_BYTES;
use common::block_pattern;
use common::degraded;
use common::gpu_lock;
use common::loaded_blocks;
use common::prefill;
use common::prompt;
use common::prompt_aligned;
use pegainfer_kv_store::CacheScope;
use pegainfer_kv_store::NeverCancelled;
use pegainfer_kv_store::PAD_TOKEN_ID;
use pegainfer_kv_store::PegaflowHost;
use pegainfer_kv_store::ResolvePolicy;
use pegainfer_kv_store::SaveClass;
use pegainfer_kv_store::SaveCursor;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seal_flush_resolve_roundtrip() {
    roundtrip(false).await;
}

/// Same roundtrip in the vLLM-connector host packing (`page_first`): each
/// block is stored as one host page holding every layer at its name-sorted
/// offset. glm52's MLA layout interop with a vLLM writer relies on this
/// packing agreeing byte-for-byte with the native one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seal_flush_resolve_roundtrip_page_first() {
    roundtrip(true).await;
}

async fn roundtrip(page_first: bool) {
    let _gpu = gpu_lock().lock().await;
    let host = PegaflowHost::builder(HOST_POOL_BYTES)
        .build()
        .expect("host");
    // Distinct namespaces per layout: the packing is part of the content
    // domain (two instances exchange blocks only if they agree on it).
    let mut rig = Rig::new_with_layout(
        if page_first {
            "roundtrip_page_first"
        } else {
            "roundtrip_native"
        },
        host,
        None,
        page_first,
    );
    let prompt = prompt(4);

    // Producer side: prefill seals the 4 full blocks; stage recognizable
    // content into them first (the arenas are the KV the "forward pass" wrote).
    let kv = prefill(&rig.pool, &prompt);
    let saved_ids: Vec<i32> = kv
        .assigned_block_hashes()
        .iter()
        .map(|&(id, _)| id)
        .collect();
    assert_eq!(
        saved_ids.len(),
        4,
        "65-token prompt seals its 4 full blocks"
    );
    rig.stage_block_patterns(&saved_ids);
    rig.stream.synchronize().expect("stage sync");
    rig.store
        .retire(RANK, kv, SaveCursor::new(), SaveClass::Cacheable);
    rig.store.flush_saves(RANK).await.expect("flush");

    // Drop the produced KV from HBM and the radix alike: the resolve below
    // must restore both the radix entry and the block bytes from the host.
    rig.zero_arenas();
    rig.pool.evict_inactive();

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
    assert_eq!(prefix.hit_tokens(), 4 * BLOCK_TOKENS);
    assert_eq!(loaded_blocks(&rig), 4);
    assert_eq!(degraded(&rig), 0);
    assert_eq!(
        rig.store.stats().saves_failed.load(Ordering::Relaxed),
        0,
        "no save failure on a healthy engine"
    );

    // Byte-exact check across the whole D2H -> H2D cycle: the i-th hit block's
    // content must equal what was staged in the i-th saved block, per layer.
    let mut req = rig.pool.new_request(prompt.clone(), 4, None);
    assert_eq!(
        req.match_and_add_prefix(&rig.pool).expect("match"),
        4 * BLOCK_TOKENS,
        "the loaded blocks were committed under the continuation hashes"
    );
    let dst_ids = req.current_page_indices();
    for layer in 0..NUM_LAYERS {
        let bytes = rig.arena_bytes(layer);
        for (pos, (&src, &dst)) in saved_ids.iter().zip(dst_ids.iter()).enumerate() {
            let begin = dst as usize * SEGMENT_BYTES;
            assert_eq!(
                &bytes[begin..begin + SEGMENT_BYTES],
                block_pattern(layer, src as usize).as_slice(),
                "layer {layer} hit position {pos} (dst block {dst})"
            );
        }
    }
    drop(prefix);
    req.release().expect("release");
}

/// The whole pad-and-seal contract: P pads its partial page to the boundary
/// and seals; the decode side rebuilds the naming chain (prompt + anchor +
/// pads) from the handoff metadata alone and resolves every page, boundary
/// included, to a full hit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pad_and_seal_roundtrips_via_the_padded_chain() {
    let _gpu = gpu_lock().lock().await;
    let host = PegaflowHost::builder(HOST_POOL_BYTES)
        .build()
        .expect("host");
    let rig = Rig::new("pad_and_seal", host, None);

    // P side: 37 committed rows (2 full pages + 5), a sampled anchor, pads.
    let prompt: Vec<u32> = (0..37).map(|i| i % 251).collect();
    let anchor = 251u32;
    let mut kv = rig.pool.new_request(prompt.clone(), 4, None);
    kv.schedule_prefill(prompt.len(), &rig.pool)
        .expect("schedule");
    kv.apply_prefill(anchor, &rig.pool).expect("apply");
    let pads = kv.pad_to_boundary(&rig.pool).expect("pad");
    assert_eq!(pads, 10, "37 rows + anchor + 10 pads reach the 48 boundary");
    assert_eq!(
        kv.assigned_block_hashes().len(),
        3,
        "the padded page sealed alongside the full pages"
    );
    rig.store
        .retire(RANK, kv, SaveCursor::new(), SaveClass::Handoff);
    rig.store.flush_saves(RANK).await.expect("flush");
    rig.pool.evict_inactive();

    // D side: the chain derives from committed_len + anchor, nothing else.
    let mut chain = prompt;
    chain.push(anchor);
    chain.resize(3 * BLOCK_TOKENS, PAD_TOKEN_ID);
    let prefix = rig
        .store
        .resolve_prefix(
            RANK,
            "d-restore",
            &chain,
            CacheScope::default(),
            ResolvePolicy::default().wait_for_full_hit().full_pages(),
            &NeverCancelled,
        )
        .await;
    assert_eq!(prefix.hit_tokens(), 3 * BLOCK_TOKENS);
    assert_eq!(loaded_blocks(&rig), 3);
    assert_eq!(degraded(&rig), 0);
    drop(prefix);
}

/// The boundary block of a pad-aligned chain: the default policy stops one
/// block short of the chain end, `full_pages` resolves it too — the decode
/// side of a handoff never prefills, so its boundary block must be part of
/// the hit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_pages_resolves_the_boundary_block() {
    let _gpu = gpu_lock().lock().await;
    let host = PegaflowHost::builder(HOST_POOL_BYTES)
        .build()
        .expect("host");
    let rig = Rig::new("full_pages_boundary", host, None);
    let chain = prompt_aligned(4);

    rig.run_and_retire(&chain, SaveClass::Cacheable);
    rig.store.flush_saves(RANK).await.expect("flush");
    rig.pool.evict_inactive();

    let partial = rig
        .store
        .resolve_prefix(
            RANK,
            "r-default",
            &chain,
            CacheScope::default(),
            ResolvePolicy::default(),
            &NeverCancelled,
        )
        .await;
    assert_eq!(
        partial.hit_tokens(),
        3 * BLOCK_TOKENS,
        "default cap excludes the boundary block"
    );
    drop(partial);

    let full = rig
        .store
        .resolve_prefix(
            RANK,
            "r-full-pages",
            &chain,
            CacheScope::default(),
            ResolvePolicy::default().wait_for_full_hit().full_pages(),
            &NeverCancelled,
        )
        .await;
    assert_eq!(full.hit_tokens(), 4 * BLOCK_TOKENS);
    assert_eq!(loaded_blocks(&rig), 3 + 1, "the boundary block loads too");
    assert_eq!(degraded(&rig), 0);
    drop(full);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wait_for_full_hit_degrades_when_cold_then_succeeds_after_save() {
    let _gpu = gpu_lock().lock().await;
    let host = PegaflowHost::builder(HOST_POOL_BYTES)
        .build()
        .expect("host");
    let rig = Rig::new("wait_full_hit", host, Some(Duration::from_millis(25)));
    let prompt = prompt(4);

    // Cold cache under the all-or-nothing protocol: a Miss reads as "the
    // producing prefill's registration has not landed yet", so the resolve
    // waits out the 25 ms deadline instead of concluding empty-handed.
    let cold = rig
        .store
        .resolve_prefix(
            RANK,
            "r-cold",
            &prompt,
            CacheScope::default(),
            ResolvePolicy::default().wait_for_full_hit(),
            &NeverCancelled,
        )
        .await;
    assert_eq!(cold.hit_tokens(), 0);
    assert!(!cold.has_hold());
    assert_eq!(degraded(&rig), 1, "deadline-exceeded degrade");
    assert_eq!(loaded_blocks(&rig), 0);

    // Once the producer's save + flush lands, the same policy hits fully.
    rig.run_and_retire(&prompt, SaveClass::Cacheable);
    rig.store.flush_saves(RANK).await.expect("flush");
    rig.pool.evict_inactive();

    let warm = rig
        .store
        .resolve_prefix(
            RANK,
            "r-warm",
            &prompt,
            CacheScope::default(),
            ResolvePolicy::default().wait_for_full_hit(),
            &NeverCancelled,
        )
        .await;
    assert_eq!(warm.hit_tokens(), 4 * BLOCK_TOKENS);
    assert_eq!(loaded_blocks(&rig), 4);
    assert_eq!(degraded(&rig), 1, "the cold degrade stands, nothing new");
    drop(warm);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn seal_skips_prefix_hit_blocks() {
    let _gpu = gpu_lock().lock().await;
    let host = PegaflowHost::builder(HOST_POOL_BYTES)
        .build()
        .expect("host");
    let rig = Rig::new("seal_skips_prefix_hit_blocks", host, None);
    let seed_prompt = prompt(3);
    let long_prompt = prompt(6); // shares its first 3 blocks with seed_prompt

    rig.run_and_retire(&seed_prompt, SaveClass::Cacheable);
    rig.store.flush_saves(RANK).await.expect("flush seed");
    rig.pool.evict_inactive();

    // Only the seeded 3 blocks are query-visible anywhere.
    let prefix = rig
        .store
        .resolve_prefix(
            RANK,
            "r-seeded",
            &long_prompt,
            CacheScope::default(),
            ResolvePolicy::default(),
            &NeverCancelled,
        )
        .await;
    assert_eq!(prefix.hit_tokens(), 3 * BLOCK_TOKENS);
    assert_eq!(loaded_blocks(&rig), 3);
    drop(prefix);

    // A request on the long prompt GPU-matches those 3 (the resolve left them
    // in the radix) and seals; its cursor must skip the matched prefix — the
    // new queryable blocks afterwards are exactly the continuation.
    let mut kv = rig.pool.new_request(long_prompt.clone(), 4, None);
    assert_eq!(
        kv.match_and_add_prefix(&rig.pool).expect("match"),
        3 * BLOCK_TOKENS
    );
    let remaining = long_prompt.len() - 3 * BLOCK_TOKENS;
    kv.schedule_prefill(remaining, &rig.pool).expect("schedule");
    kv.apply_prefill(1, &rig.pool).expect("apply");
    let mut cursor = SaveCursor::new();
    rig.store.seal(RANK, &kv, &mut cursor, SaveClass::Cacheable);
    rig.store
        .flush_saves(RANK)
        .await
        .expect("flush continuation");
    assert_eq!(
        rig.store.stats().saves_submitted.load(Ordering::Relaxed),
        2,
        "one seal by the seed's retire, one by this request"
    );
    rig.store.retire(RANK, kv, cursor, SaveClass::Cacheable);

    rig.pool.evict_inactive();
    let prefix = rig
        .store
        .resolve_prefix(
            RANK,
            "r-full",
            &long_prompt,
            CacheScope::default(),
            ResolvePolicy::default(),
            &NeverCancelled,
        )
        .await;
    // 3 (seeded) + 3 (continuation) — anything the cursor mis-saved or
    // double-counted shows up here as the wrong total.
    assert_eq!(prefix.hit_tokens(), 6 * BLOCK_TOKENS);
    assert_eq!(loaded_blocks(&rig), 3 + 6);
    assert_eq!(degraded(&rig), 0);
    assert_eq!(rig.store.stats().saves_failed.load(Ordering::Relaxed), 0);
    drop(prefix);
}
