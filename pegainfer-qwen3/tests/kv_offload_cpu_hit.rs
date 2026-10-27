//! Live GPU+CPU prefix-hit gate for the pegaflow KV-offload integration.
//!
//! Drives a real Qwen3-4B [`Qwen3Executor`] with offload enabled to prove the
//! end-to-end wiring on actual model weights:
//!   * a cold prefill SAVEs its sealed KV blocks to pegaflow's host tier;
//!   * after the GPU prefix cache is flushed, a second identical request finds
//!     the prefix only on the CPU tier (a genuine CPU-only hit) and the async
//!     prefetch RESTOREs it into HBM;
//!   * the restored KV reproduces the original first-token logits.
//!
//! This is the one test that exercises save → host-tier persistence → query →
//! async load → register → prefill-rematch through the executor, not a unit
//! harness. `tests/cpu_roundtrip.rs` (in `pegainfer-kv-offload`) covers the raw
//! byte path; this covers the live executor wiring. If the load landed in the
//! wrong layer/segment/block the warm logits would be whole nats off.
//!
//! Requires a CUDA GPU and Qwen3-4B weights; skips cleanly when absent
//! (point `PEGAINFER_TEST_MODEL_PATH` at the weights to run it).

use std::collections::HashMap;
use std::path::Path;

use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen3::Qwen3LoraOptions;
use pegainfer_qwen3::Qwen3OffloadOptions;
use pegainfer_qwen3::runtime::ConnectorLookupCounters;
use pegainfer_qwen3::runtime::PrefillPlan;
use pegainfer_qwen3::runtime::PrefillStepItem;
use pegainfer_qwen3::runtime::Qwen3Executor;
use pegainfer_qwen3::runtime::RequestId;

const MODEL_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3-4B");
const BLOCK: usize = 16;
const LOGPROBS: usize = 16;
const MAX_OUTPUT: usize = 8;
/// 512 MiB host tier — comfortably more than the handful of dense Qwen3-4B
/// blocks this test offloads (~2.25 MiB/block).
const HOST_TIER_BYTES: usize = 512 << 20;

/// Warm-vs-cold bounds, following the prefix-cache methodology: the CPU-restored
/// KV is byte-identical to the original GPU compute, so the only legitimate
/// drift is the prefill GEMM shrinking to the uncached tail (bf16 reduction
/// order). The warm argmax must sit within `REGRET_TOL` of cold; the mean head
/// delta must stay at the bf16 floor.
const REGRET_TOL: f32 = 0.20;
const MEAN_TOL: f32 = 0.06;

fn model_path_or_skip() -> Option<String> {
    match std::env::var("PEGAINFER_TEST_MODEL_PATH") {
        Ok(path) => Some(path),
        Err(_) if Path::new(MODEL_PATH).join("config.json").exists() => {
            Some(MODEL_PATH.to_string())
        }
        Err(_) => {
            eprintln!(
                "skipping qwen3 kv_offload_cpu_hit: {MODEL_PATH}/config.json is missing; \
                 set PEGAINFER_TEST_MODEL_PATH to run it"
            );
            None
        }
    }
}

/// Deterministic synthetic prompt; different seeds share no prefix.
fn prompt(seed: usize, len: usize) -> Vec<u32> {
    (0..len)
        .map(|i| ((seed * 100_003 + i * 17) % 50_000 + 1_000) as u32)
        .collect()
}

fn prefill_item(id: u64, prompt: &[u32]) -> PrefillStepItem {
    PrefillStepItem::new(
        RequestId::new(id),
        prompt.to_vec(),
        MAX_OUTPUT,
        SamplingParams::default(),
        Some(LOGPROBS),
        None,
    )
}

/// The connector lookups settled since `before`, as `(queries, hits)` tokens.
fn lookups_since(ex: &Qwen3Executor, before: ConnectorLookupCounters) -> (u64, u64) {
    let now = ex.connector_lookup_counters();
    (now.queries - before.queries, now.hits - before.hits)
}

fn first_token_top(pr: &pegainfer_qwen3::runtime::PrefillResult) -> Vec<(u32, f32)> {
    pr.requests[0]
        .first_token_logprob
        .as_ref()
        .expect("logprobs requested but none returned")
        .top_logprobs
        .clone()
}

/// The warm (CPU-restored) first-token logits must agree with the cold compute
/// up to bf16 reduction noise: warm argmax within `REGRET_TOL` of cold, mean
/// head-token delta under `MEAN_TOL`.
fn assert_close(cold: &[(u32, f32)], warm: &[(u32, f32)]) {
    let cold_map: HashMap<u32, f32> = cold.iter().copied().collect();
    let cold_top = cold[0].1;
    match cold_map.get(&warm[0].0) {
        None => panic!(
            "warm argmax {} absent from cold top-{}",
            warm[0].0,
            cold.len()
        ),
        Some(&clp) => assert!(
            cold_top - clp <= REGRET_TOL,
            "warm argmax {} sits {:.4} nat below cold argmax",
            warm[0].0,
            cold_top - clp
        ),
    }
    let deltas: Vec<f32> = warm
        .iter()
        .take(8)
        .filter_map(|&(token, wlp)| cold_map.get(&token).map(|&clp| (wlp - clp).abs()))
        .collect();
    assert!(!deltas.is_empty(), "no head-token overlap");
    let mean = deltas.iter().sum::<f32>() / deltas.len() as f32;
    let max = deltas.iter().copied().fold(0.0f32, f32::max);
    eprintln!(
        "kv_offload_cpu_hit: {} head deltas — mean {mean:.4} max {max:.4}",
        deltas.len()
    );
    assert!(
        mean <= MEAN_TOL,
        "mean head logprob delta {mean:.4} > {MEAN_TOL} — restored KV drifted past bf16 noise"
    );
}

/// One executor, five scenarios, run sequentially. cargo runs `#[test]`
/// functions on parallel threads; two Qwen3-4B executors sharing device 0 and
/// the same pegaflow instance id ("qwen3-dev0") would collide on the host
/// tier. Production wires exactly one executor per model, so the realistic
/// shape is one executor servicing every prefix. The scenarios use disjoint
/// prompt seeds, so they share no prefix and cannot cross-contaminate.
#[test]
fn live_gpu_and_cpu_prefix_hits() {
    let Some(model_path) = model_path_or_skip() else {
        return;
    };
    let mut ex = Qwen3Executor::from_runtime_with_lora_options(
        &model_path,
        false,
        &[0],
        Qwen3LoraOptions::default(),
        Qwen3OffloadOptions::enabled(HOST_TIER_BYTES),
        pegainfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS,
        None,
        pegainfer_qwen3::Qwen3MemoryOptions::default(),
    )
    .expect("build offload executor");
    assert!(ex.offload_enabled(), "offload must be active");

    cpu_tier_restores_evicted_prefix(&mut ex);
    gpu_and_cpu_combined_hit(&mut ex);
    connector_miss_is_counted(&mut ex);
    local_growth_after_a_partial_load_stays_local(&mut ex);
    cancelled_after_load_still_counts_its_lookup(&mut ex);
}

/// A prefix that is evicted from HBM and restored entirely from the CPU tier
/// (`gpu_hit == 0`): the baseline CPU round-trip through the live executor.
fn cpu_tier_restores_evicted_prefix(ex: &mut Qwen3Executor) {
    let p = prompt(7, 50); // 3 full blocks (48 tok) + 2-token tail

    // ── Cold: first sight of P. Computes all of P on GPU and offloads the 3
    // sealed blocks to the host tier. ──
    let cold = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(1, &p)],
        })
        .expect("cold prefill");
    assert_eq!(
        cold.requests[0].cached_tokens,
        Some(0),
        "first sight of P is cold: a lookup ran and missed"
    );
    let cold_first = first_token_top(&cold);
    ex.drop_request(RequestId::new(1)).expect("drop req1");

    // ── Persist the saves, then evict P from HBM so it lives only on CPU. ──
    ex.flush_offload_saves();
    ex.evict_cached_blocks();

    // ── A GPU miss now: the prefetch must restore P from the CPU tier. ──
    let hit = ex.begin_kv_prefetch(RequestId::new(2), &p, None, 0);
    assert!(hit, "P must hit the CPU tier after GPU eviction");
    let ready = ex.wait_ready_prefetch(0);
    assert!(
        ready.contains(&RequestId::new(2)),
        "prefetch load must settle ready, got {ready:?}"
    );

    // ── Warm: the restored CPU prefix is matched, only the 2-token tail
    // recomputes (the full-block cap keeps the 3rd block's last token off the
    // match the same way the GPU prefix cache does). ──
    let warm = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(2, &p)],
        })
        .expect("warm prefill");
    assert_eq!(
        warm.requests[0].cached_tokens,
        Some(3 * BLOCK),
        "CPU-restored prefix: 3 blocks matched, tail recomputed"
    );
    // All of the match came back from the host tier, so the external leg owns
    // all of it — the split this PR reports separately.
    assert_eq!(
        warm.requests[0].external_hit_tokens,
        Some(3 * BLOCK),
        "restored blocks are attributed to the external side, not local KV"
    );
    let warm_first = first_token_top(&warm);
    ex.drop_request(RequestId::new(2)).expect("drop req2");

    // ── The restored KV must reproduce the original GPU first-token logits. ──
    assert_close(&cold_first, &warm_first);
}

/// A connector query that finds nothing is still a query. `begin_kv_prefetch`
/// answers `false` for a miss and leaves no prefetch state behind, so the
/// external counters can only separate "asked and missed" from "never asked"
/// by recording the consultation itself. Without that record every miss
/// disappears from the counters and the external hit rate reads 100%.
///
/// The missed request then stays deferred while another request registers
/// the same prefix in local KV. Its external query width must still be the
/// query that was sent (the whole prompt), not one re-derived from the
/// larger admission-time match.
fn connector_miss_is_counted(ex: &mut Qwen3Executor) {
    let unseen = prompt(11, 50); // disjoint seed: never saved to the host tier
    let id = RequestId::new(3);
    let before = ex.connector_lookup_counters();

    assert!(
        !ex.begin_kv_prefetch(id, &unseen, None, 0),
        "an unseen prefix must not park on a load"
    );
    assert!(
        ex.wait_ready_prefetch(0).is_empty(),
        "a miss leaves no load to settle"
    );

    // While request 3 waits, request 4 computes the same prompt and leaves
    // its 3 full blocks in the local prefix cache. It never asked the
    // connector.
    let sibling = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(4, &unseen)],
        })
        .expect("sibling prefill");
    assert_eq!(sibling.requests[0].external_hit_tokens, None);
    ex.drop_request(RequestId::new(4)).expect("drop req4");

    let missed = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(3, &unseen)],
        })
        .expect("miss prefill");
    assert_eq!(
        missed.requests[0].cached_tokens,
        Some(3 * BLOCK),
        "the sibling's blocks are matched locally at admission"
    );
    // The point of this case: `Some(0)`, not `None`. `None` would mean the
    // connector was never consulted, which is what dropped every miss from the
    // external counters before this was recorded explicitly.
    assert_eq!(
        missed.requests[0].external_hit_tokens,
        Some(0),
        "the connector was queried and restored nothing; a miss must be \
         counted, not dropped as 'never asked'"
    );
    assert_eq!(
        lookups_since(ex, before),
        (unseen.len() as u64, 0),
        "the connector was asked about the whole prompt, before the sibling \
         made part of it local; the sibling never asked"
    );
    ex.drop_request(id).expect("drop req3");
}

/// A single prefix that is part GPU-resident, part CPU-only: the prefetch must
/// stack the CPU continuation onto the GPU hit and the re-match must see one
/// contiguous prefix. This is the case that catches an off-by-`gpu_hit` bug in
/// the query/commit offset math — the pure-CPU test (`gpu_hit == 0`) cannot.
fn gpu_and_cpu_combined_hit(ex: &mut Qwen3Executor) {
    let full = prompt(9, 100); // 6 full blocks (96 tok) + 4-token tail
    let short = full[..50].to_vec(); // a 3-block prefix of `full`

    // ── Cold-compute `full`, saving all 6 blocks to the host tier. ──
    let cold = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(1, &full)],
        })
        .expect("cold full prefill");
    assert_eq!(
        cold.requests[0].cached_tokens,
        Some(0),
        "first sight of full is cold: a lookup ran and missed"
    );
    let cold_first = first_token_top(&cold);
    ex.drop_request(RequestId::new(1)).expect("drop req1");
    ex.flush_offload_saves();

    // ── Drop the whole prefix from HBM (CPU keeps all 6 blocks), then
    // re-establish ONLY the first 3 blocks in HBM by cold-prefilling `short`.
    // GPU now holds blocks 0..3; CPU holds blocks 0..6. ──
    ex.evict_cached_blocks();
    let s = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(2, &short)],
        })
        .expect("short prefill");
    assert_eq!(
        s.requests[0].cached_tokens,
        Some(0),
        "short re-warms blocks 0..3 cold: a lookup ran and missed"
    );
    ex.drop_request(RequestId::new(2)).expect("drop req2");

    // ── Prefetch `full`: GPU hits blocks 0..3, the host tier must supply the
    // continuation 3..6. A pure GPU hit would not start a load. ──
    let before = ex.connector_lookup_counters();
    let hit = ex.begin_kv_prefetch(RequestId::new(3), &full, None, 0);
    assert!(
        hit,
        "blocks 3..6 must be fetched from the CPU tier beyond the GPU hit"
    );
    let ready = ex.wait_ready_prefetch(0);
    assert!(
        ready.contains(&RequestId::new(3)),
        "prefetch must settle, got {ready:?}"
    );

    // ── Warm prefill `full`: all 6 blocks match (3 GPU + 3 CPU). Without the
    // CPU continuation this would be 3. ──
    let warm = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(3, &full)],
        })
        .expect("warm full prefill");
    assert_eq!(
        warm.requests[0].cached_tokens,
        Some(6 * BLOCK),
        "combined hit: 3 GPU-resident + 3 CPU-restored blocks match as one prefix"
    );
    // Only the second half crossed the connector: the first 3 blocks were
    // already in HBM, so they belong to the local family.
    assert_eq!(
        warm.requests[0].external_hit_tokens,
        Some(3 * BLOCK),
        "the GPU-resident half is local, the restored half is external"
    );
    assert_eq!(
        lookups_since(ex, before),
        ((full.len() - 3 * BLOCK) as u64, (3 * BLOCK) as u64),
        "the connector was asked about everything past the GPU-resident half \
         and restored the second half"
    );
    let warm_first = first_token_top(&warm);
    ex.drop_request(RequestId::new(3)).expect("drop req3");

    assert_close(&cold_first, &warm_first);
}

/// The connector restores only part of the prompt, and while the request is
/// parked on that load another request computes the whole prompt locally.
/// The admission-time match then spans the loaded blocks and the sibling's
/// longer prefix; only the loaded blocks crossed the connector, so only they
/// are external hits.
fn local_growth_after_a_partial_load_stays_local(ex: &mut Qwen3Executor) {
    let full = prompt(13, 100); // 6 full blocks (96 tok) + 4-token tail
    let short = full[..50].to_vec(); // 3-block prefix of `full`

    // ── Only `short`'s 3 blocks reach the host tier; HBM keeps none. ──
    ex.execute_prefill(PrefillPlan {
        sample_seed: 0,
        requests: &[prefill_item(1, &short)],
    })
    .expect("cold short prefill");
    ex.drop_request(RequestId::new(1)).expect("drop req1");
    ex.flush_offload_saves();
    ex.evict_cached_blocks();

    // ── Request 2 asks the connector about all of `full` (nothing local) and
    // loads the 3 blocks it has. ──
    let id = RequestId::new(2);
    let before = ex.connector_lookup_counters();
    assert!(
        ex.begin_kv_prefetch(id, &full, None, 0),
        "the host tier holds the first 3 blocks"
    );
    let ready = ex.wait_ready_prefetch(0);
    assert!(ready.contains(&id), "prefetch must settle, got {ready:?}");

    // ── While request 2 is parked, request 3 computes `full` and registers
    // blocks 3..6 locally. ──
    ex.execute_prefill(PrefillPlan {
        sample_seed: 0,
        requests: &[prefill_item(3, &full)],
    })
    .expect("sibling full prefill");
    ex.drop_request(RequestId::new(3)).expect("drop req3");

    let admitted = ex
        .execute_prefill(PrefillPlan {
            sample_seed: 0,
            requests: &[prefill_item(2, &full)],
        })
        .expect("admitted full prefill");
    assert_eq!(
        admitted.requests[0].cached_tokens,
        Some(6 * BLOCK),
        "3 loaded blocks + the sibling's 3 local blocks match as one prefix"
    );
    assert_eq!(
        admitted.requests[0].external_hit_tokens,
        Some(3 * BLOCK),
        "only the loaded blocks are external; the sibling's growth is local"
    );
    assert_eq!(
        lookups_since(ex, before),
        (full.len() as u64, (3 * BLOCK) as u64),
        "the connector was asked about the whole prompt and restored 3 blocks"
    );
    ex.drop_request(id).expect("drop req2");
}

/// A request whose load settled and which is then cancelled before its first
/// prefill (the scheduler's abort and reject paths both end in
/// `drop_request`) still asked the connector. Its lookup must reach the
/// counters as a query with no hits, exactly once.
fn cancelled_after_load_still_counts_its_lookup(ex: &mut Qwen3Executor) {
    let p = prompt(17, 50); // 3 full blocks (48 tok) + 2-token tail

    ex.execute_prefill(PrefillPlan {
        sample_seed: 0,
        requests: &[prefill_item(1, &p)],
    })
    .expect("cold prefill");
    ex.drop_request(RequestId::new(1)).expect("drop req1");
    ex.flush_offload_saves();
    ex.evict_cached_blocks();

    let id = RequestId::new(2);
    let before = ex.connector_lookup_counters();
    assert!(
        ex.begin_kv_prefetch(id, &p, None, 0),
        "P must hit the CPU tier after GPU eviction"
    );
    let ready = ex.wait_ready_prefetch(0);
    assert!(ready.contains(&id), "prefetch must settle, got {ready:?}");
    assert_eq!(
        lookups_since(ex, before),
        (0, 0),
        "a lookup is settled when the request leaves, not when it is sent"
    );

    // Cancelled while parked: no prefill ever runs for it.
    ex.drop_request(id).expect("drop req2");
    assert_eq!(
        lookups_since(ex, before),
        (p.len() as u64, 0),
        "a cancelled request still reports the query it sent, with no hits"
    );
    ex.drop_request(id).expect("drop req2 again");
    assert_eq!(
        lookups_since(ex, before),
        (p.len() as u64, 0),
        "a lookup is settled once"
    );
}
