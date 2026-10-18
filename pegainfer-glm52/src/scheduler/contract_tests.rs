//! Offline replica of the engine's engine-fatal KV contract: the exact
//! schedule/apply sequence of the submit walk, driven end to end against a
//! real [`BlockPool`] — a schedule failure in serving is engine-fatal, so
//! the full-lifetime reservation must be proven tight here.

use std::collections::VecDeque;
use std::sync::Arc;

use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::KvPrefix;
use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_frontend::engine::TokenSink;
use pegainfer_kv_store::BlockPool;
use pegainfer_kv_store::CacheScope;
use pegainfer_kv_store::KvStore;
use pegainfer_kv_store::KvStoreBuilder;
use pegainfer_kv_store::NeverCancelled;
use pegainfer_kv_store::ResolvePolicy;
use pegainfer_kv_store::SaveCursor;
use pegainfer_sample::SamplingParams;

use super::ActiveRequest;
use super::PAGE;
use super::RankSlots;
use super::admission::lifetime_blocks;
use super::admit_from_queue;
use super::graph::graph_dump_bucket;
use super::offload::NativeMtpHandoff;
use super::offload::Resolved;
use super::offload::native_pd_resolve;
use super::publish_load;
use super::slot::GLM52_DSPARK_EP8_SPAN_DRAFTS;
use super::slot::Glm52SlotState;
use super::slot::Glm52StepOutcome;
use super::testkit::EOS;
use super::testkit::request;
use crate::model::GLM52_MAX_BATCH_PER_RANK;

/// A tier-less store over `pool`, on a private runtime: admission consults
/// only `pinned_blocks` (0 here), so any live handle serves.
fn test_store(pool: &Arc<BlockPool>) -> (KvStore, tokio::runtime::Runtime) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    let store = KvStoreBuilder::new(rt.handle().clone())
        .rank(0, Arc::clone(pool))
        .build();
    (store, rt)
}

fn plain(req: GenerateRequest) -> Resolved {
    Resolved::Plain {
        req,
        prefix: KvPrefix::none(),
    }
}

#[test]
fn graph_dump_uses_bucket_one() {
    assert_eq!(graph_dump_bucket(), 1, "EP and TP4 export bucket-1 graphs");
}

#[test]
fn load_snapshot_reports_the_ranks_own_state() {
    let pool = Arc::new(BlockPool::new(PAGE, 8));
    let mut slots: RankSlots = std::array::from_fn(|_| None);

    let req = request(vec![10, 11], SamplingParams::default(), 4);
    let state = Glm52SlotState::new(req.prompt_tokens.clone(), req.max_tokens, true, 0);
    let mut kv = pool.new_request(req.prompt_tokens.clone(), req.max_tokens, None);
    kv.schedule_prefill(1, &pool).expect("one live KV block");
    slots[0] = Some(ActiveRequest {
        req,
        state,
        client_prompt_tokens: 2,
        kv,
        save_cursor: SaveCursor::new(),
        boundary_copy: None,
    });

    let mut pending = VecDeque::new();
    pending.push_back(plain(request(vec![20], SamplingParams::default(), 4)));
    pending.push_back(plain(request(vec![21], SamplingParams::default(), 4)));

    let (load_tx, load_rx) = tokio::sync::watch::channel(SchedulerMetrics::default());
    publish_load(&load_tx, &pool, &slots, &pending, 0);

    let snapshot = *load_rx.borrow();
    assert_eq!(snapshot.num_running_reqs, 1);
    assert_eq!(snapshot.num_waiting_reqs, 2);
    assert_eq!(snapshot.kv_total_blocks, 7);
    assert_eq!(snapshot.kv_used_blocks, 1);
}

#[test]
fn load_snapshot_counts_in_flight_resolves_as_waiting() {
    // The waiting count covers the whole intake-to-slot window: a request
    // resolving off-thread (up to the full resolve deadline) is admitted
    // load, and a snapshot that omitted it would let the frontend route a
    // burst at a rank that is anything but idle.
    let pool = Arc::new(BlockPool::new(PAGE, 8));
    let slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::new();
    let (load_tx, load_rx) = tokio::sync::watch::channel(SchedulerMetrics::default());

    // Intake's side: the counter rises before the resolver task spawns,
    // with the deque still empty.
    let inflight = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    inflight.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    publish_load(
        &load_tx,
        &pool,
        &slots,
        &pending,
        inflight.load(std::sync::atomic::Ordering::Acquire),
    );
    assert_eq!(
        load_rx.borrow().num_waiting_reqs,
        1,
        "a mid-resolve request is waiting load, not invisible"
    );

    // The drain's side: the decrement and the deque push happen together,
    // so the request re-homes without ever double-counting.
    inflight.fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
    pending.push_back(plain(request(vec![20], SamplingParams::default(), 4)));
    publish_load(
        &load_tx,
        &pool,
        &slots,
        &pending,
        inflight.load(std::sync::atomic::Ordering::Acquire),
    );
    assert_eq!(
        load_rx.borrow().num_waiting_reqs,
        1,
        "moving to the deque keeps the count at one"
    );
}

#[test]
fn admission_fills_free_slots_from_the_local_queue() {
    let pool = Arc::new(BlockPool::new(PAGE, 8));
    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::new();
    let mut req = request(vec![10], SamplingParams::default(), 4);
    req.data_parallel_rank = Some(0);
    let (token_tx, _token_rx) = TokenSink::standalone();
    req.token_tx = token_tx;
    pending.push_back(plain(req));
    let mut pending_resets = Vec::new();

    let (store, _rt) = test_store(&pool);
    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        7,
        &store,
        false,
        false,
        false,
        &mut pending_resets,
    )
    .expect("admission");

    assert!(slots[0].is_some());
    assert!(pending.is_empty());
    let active = slots[0].as_ref().expect("admitted");
    assert_eq!(
        pool.entitled_blocks(),
        active.kv.lifetime_blocks() - active.kv.resident_blocks(),
        "admission entitles the un-drawn lifetime remainder"
    );
}

#[test]
fn prefill_only_admits_multiple_requests_within_pool_capacity() {
    let pool = Arc::new(BlockPool::new(PAGE, 16));
    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::new();
    let mut token_receivers = Vec::new();
    for token in [10, 20] {
        let mut req = request(vec![token], SamplingParams::default(), 1);
        let (token_tx, token_rx) = TokenSink::standalone();
        req.token_tx = token_tx;
        token_receivers.push(token_rx);
        pending.push_back(plain(req));
    }
    let mut pending_resets = Vec::new();

    let (store, _rt) = test_store(&pool);
    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        15,
        &store,
        true,
        false,
        true,
        &mut pending_resets,
    )
    .expect("prefill-only admission");

    assert_eq!(slots.iter().flatten().count(), 2);
    assert!(pending.is_empty());
}

#[test]
fn admission_defers_while_physical_pages_are_temporarily_held() {
    let pool = Arc::new(BlockPool::new(PAGE, 6));
    let mut held = pool.new_request(vec![1; 2 * PAGE], 1, None);
    held.schedule_prefill(2 * PAGE, &pool)
        .expect("temporarily hold two physical pages");

    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::new();
    let mut req = request(vec![2; 3 * PAGE], SamplingParams::default(), 1);
    let (token_tx, _token_rx) = TokenSink::standalone();
    req.token_tx = token_tx;
    pending.push_back(plain(req));
    let mut pending_resets = Vec::new();

    let (store, _rt) = test_store(&pool);
    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        5,
        &store,
        true,
        false,
        true,
        &mut pending_resets,
    )
    .expect("temporary pressure is not an admission error");

    assert_eq!(pending.len(), 1, "request stays queued");
    assert!(slots.iter().all(Option::is_none));

    held.revert_schedule().expect("release temporary pages");
    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        5,
        &store,
        true,
        false,
        true,
        &mut pending_resets,
    )
    .expect("admit after pressure clears");

    assert!(pending.is_empty());
    assert_eq!(slots.iter().flatten().count(), 1);
}

#[test]
fn admission_sheds_rear_prefix_holds_when_the_front_cannot_fit() {
    // The stall shape: the front's lifetime budget cannot fit because pages
    // are pinned by a hold owned by a request QUEUED BEHIND it — a hold that
    // only releases at its owner's admission, which the front blocks. The
    // budget defer must shed that rear hold within the same call, not park
    // the queue forever.
    let pool = Arc::new(BlockPool::new(PAGE, 6));
    let held = {
        let mut kv = pool.new_request(vec![1; 2 * PAGE], 1, None);
        kv.schedule_prefill(2 * PAGE, &pool)
            .expect("rear hold pins two physical pages");
        kv
    };

    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::new();
    let mut front = request(vec![2; 3 * PAGE], SamplingParams::default(), 1);
    let (front_tx, _front_rx) = TokenSink::standalone();
    front.token_tx = front_tx;
    pending.push_back(plain(front));
    let mut rear = request(vec![9], SamplingParams::default(), 1);
    let (rear_tx, _rear_rx) = TokenSink::standalone();
    rear.token_tx = rear_tx;
    pending.push_back(Resolved::Plain {
        req: rear,
        prefix: KvPrefix::resolved(2 * PAGE, 0, Box::new(held)),
    });
    let mut pending_resets = Vec::new();

    let (store, _rt) = test_store(&pool);
    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        5,
        &store,
        true,
        false,
        true,
        &mut pending_resets,
    )
    .expect("shedding the rear hold is not an admission error");

    assert!(
        slots
            .iter()
            .flatten()
            .any(|active| active.req.prompt_tokens.len() == 3 * PAGE),
        "front admits once the rear hold is shed"
    );
    for entry in &pending {
        if let Resolved::Plain { prefix, .. } = entry {
            assert_eq!(prefix.hit_tokens(), 0, "queued holds were shed, not kept");
        }
    }
}

#[test]
fn admission_admits_a_page_aligned_full_hit_at_exact_pool_capacity() {
    // A page-aligned prompt fully cached: the GPU probe matches every prompt
    // block, but the resolve caps the hit one block short (the final chunk
    // must recompute to emit the first token). The returned hold must pin
    // exactly hit_tokens/PAGE blocks — a hold that kept the capped block
    // pinned would, at exact pool capacity, make the front wait forever on
    // the very block its own hold withholds from the budget.
    let pool = Arc::new(BlockPool::new(PAGE, 4));
    let (store, rt) = test_store(&pool);

    let prompt: Vec<u32> = (0..2 * PAGE as u32).map(|t| 40_000 + t).collect();
    let mut producer = pool.new_request(prompt.clone(), 1, None);
    producer
        .schedule_prefill(2 * PAGE, &pool)
        .expect("producer prefill");
    producer
        .apply_prefill(50_000, &pool)
        .expect("producer apply");
    producer.release().expect("producer release");

    let prefix = rt.block_on(store.resolve_prefix(
        0,
        "aligned",
        &prompt,
        CacheScope::default(),
        ResolvePolicy::default(),
        &NeverCancelled,
    ));
    assert_eq!(
        prefix.hit_tokens(),
        PAGE,
        "the cap leaves the final block uncached"
    );
    assert_eq!(
        pool.available_blocks(),
        2,
        "the hold pins exactly hit_tokens/PAGE blocks"
    );

    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::new();
    let mut req = request(prompt, SamplingParams::default(), 1);
    let (token_tx, _token_rx) = TokenSink::standalone();
    req.token_tx = token_tx;
    pending.push_back(Resolved::Plain { req, prefix });
    let mut pending_resets = Vec::new();

    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        3,
        &store,
        true,
        false,
        false,
        &mut pending_resets,
    )
    .expect("a fully-cached page-aligned prompt at exact capacity must admit");

    assert!(pending.is_empty(), "front admitted, not deferred");
    assert_eq!(slots.iter().flatten().count(), 1);
}

/// The P side of the handoff against `pool`'s radix, tier-less: prefill
/// `committed`, emit `anchor`, pad-and-name the boundary page, release —
/// then resolve the padded chain exactly as intake does. Returns the
/// admission-ready triple plus the client's event stream.
fn resolved_native(
    pool: &Arc<BlockPool>,
    store: &KvStore,
    rt: &tokio::runtime::Runtime,
    committed: Vec<u32>,
    anchor: u32,
    max_tokens: usize,
) -> (
    GenerateRequest,
    KvPrefix,
    NativeMtpHandoff,
    pegainfer_frontend::engine::TokenStreamReceiver,
) {
    let salt = super::native_mtp_cache_salt();
    let mut producer = pool.new_request_with_cache_salt(committed.clone(), 2, Some(salt), None);
    producer
        .schedule_prefill(committed.len(), pool)
        .expect("producer prefill");
    producer
        .apply_prefill(anchor, pool)
        .expect("producer apply");
    producer.pad_to_boundary(pool).expect("producer pad");
    producer.release().expect("producer release");

    let handoff = NativeMtpHandoff {
        fingerprint: super::offload::handoff_fingerprint(),
        committed_len: committed.len(),
        anchor_token_id: Some(anchor),
        draft_tokens: vec![anchor; crate::mtp::GLM52_MTP_DRAFTS],
    };
    let mut req = request(committed, SamplingParams::default(), max_tokens);
    let (tx, rx) = TokenSink::standalone();
    req.token_tx = tx;
    let prefix = rt
        .block_on(native_pd_resolve(
            store,
            0,
            &req,
            &handoff,
            anchor,
            &NeverCancelled,
        ))
        .expect("the sealed padded chain resolves fully");
    (req, prefix, handoff, rx)
}

#[test]
fn native_admission_rebuilds_the_padded_chain() {
    let pool = Arc::new(BlockPool::new(PAGE, 8));
    let (store, rt) = test_store(&pool);
    let committed: Vec<u32> = (0..(PAGE + 7) as u32).map(|t| 80_000 + t).collect();
    let (req, prefix, handoff, mut rx) = resolved_native(&pool, &store, &rt, committed, 70_001, 8);

    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::from([Resolved::Native {
        req,
        prefix,
        handoff,
    }]);
    let mut pending_resets = Vec::new();
    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        7,
        &store,
        true,
        false,
        false,
        &mut pending_resets,
    )
    .expect("native admission");

    assert!(pending.is_empty());
    let active = slots.iter().flatten().next().expect("slot formed");
    assert_eq!(active.client_prompt_tokens, PAGE + 7);
    assert_eq!(
        active.kv.kv_position(),
        PAGE + 7,
        "full pages matched, boundary rows scheduled, anchor dangling"
    );
    let copy = active
        .boundary_copy
        .as_ref()
        .expect("an unaligned restore carries the boundary copy");
    assert_ne!(copy.src_page, copy.dst_page);
    assert!(matches!(
        rx.try_recv(),
        Ok((_, pegainfer_frontend::engine::TokenEvent::Scheduled { .. }))
    ));
    assert!(matches!(
        rx.try_recv(),
        Ok((
            _,
            pegainfer_frontend::engine::TokenEvent::Token { id: 70_001, .. }
        ))
    ));
}

#[test]
fn aligned_native_front_admits_against_its_own_hold() {
    // A page-aligned commit has no boundary page (the anchor stays unnamed)
    // and the restore's pinned pages are the front's own hold: at exact
    // physical capacity, admission must credit them or the restore starves
    // on the very pages it brought.
    let pool = Arc::new(BlockPool::new(PAGE, 5));
    let (store, rt) = test_store(&pool);
    let committed: Vec<u32> = (0..(2 * PAGE) as u32).map(|t| 80_000 + t).collect();
    let (req, prefix, handoff, _rx) =
        resolved_native(&pool, &store, &rt, committed, 70_001, 2 * PAGE - 1);

    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::from([Resolved::Native {
        req,
        prefix,
        handoff,
    }]);
    let mut pending_resets = Vec::new();
    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        4,
        &store,
        true,
        false,
        false,
        &mut pending_resets,
    )
    .expect("admission at exact capacity");

    assert!(pending.is_empty());
    let active = slots.iter().flatten().next().expect("slot formed");
    assert_eq!(active.kv.kv_position(), 2 * PAGE);
    assert!(
        active.boundary_copy.is_none(),
        "an aligned commit restores whole pages; the anchor's page is private"
    );
}

#[test]
fn unaligned_native_credits_full_pages_only() {
    // An unaligned hold pins full pages PLUS the padded boundary page, but
    // only the full pages fold into the resident set — the boundary page
    // stays a pinned copy source and its destination is a fresh private
    // allocation. Crediting the boundary page would admit into a pool that
    // cannot back the destination and die at the boundary schedule.
    let committed: Vec<u32> = (0..(PAGE + 1) as u32).map(|t| 90_000 + t).collect();

    // Exact capacity: the hold owns both resolved pages, nothing is free.
    // need = 2, credit = 1 full page, physical = 0 + 1 → defer, not fatal.
    let pool = Arc::new(BlockPool::new(PAGE, 3));
    let (store, rt) = test_store(&pool);
    let (req, prefix, handoff, _rx) =
        resolved_native(&pool, &store, &rt, committed.clone(), 70_001, 8);
    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::from([Resolved::Native {
        req,
        prefix,
        handoff,
    }]);
    let mut pending_resets = Vec::new();
    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        100,
        &store,
        true,
        false,
        false,
        &mut pending_resets,
    )
    .expect("a boundary destination the pool cannot back defers, never fails");
    assert_eq!(pending.len(), 1, "the front waits for a free page");
    assert_eq!(slots.iter().flatten().count(), 0);

    // One spare page backs the boundary destination: the same shape admits.
    let pool = Arc::new(BlockPool::new(PAGE, 4));
    let (store, rt) = test_store(&pool);
    let (req, prefix, handoff, _rx) = resolved_native(&pool, &store, &rt, committed, 70_001, 8);
    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::from([Resolved::Native {
        req,
        prefix,
        handoff,
    }]);
    let mut pending_resets = Vec::new();
    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        100,
        &store,
        true,
        false,
        false,
        &mut pending_resets,
    )
    .expect("one spare page admits the boundary copy");
    assert!(pending.is_empty());
    let active = slots.iter().flatten().next().expect("slot formed");
    assert_eq!(active.kv.kv_position(), PAGE + 1);
    assert!(
        active.boundary_copy.is_some(),
        "an unaligned commit restores through a boundary copy"
    );
}

#[test]
fn suppressed_eos_finishes_at_admission_without_a_slot() {
    let pool = Arc::new(BlockPool::new(PAGE, 4));
    let (store, _rt) = test_store(&pool);
    let handoff = NativeMtpHandoff {
        fingerprint: super::offload::handoff_fingerprint(),
        committed_len: 3,
        anchor_token_id: None,
        draft_tokens: Vec::new(),
    };
    let mut req = request(vec![10, 11, 12], SamplingParams::default(), 8);
    let (tx, mut rx) = TokenSink::standalone();
    req.token_tx = tx;

    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::from([Resolved::Native {
        req,
        prefix: KvPrefix::none(),
        handoff,
    }]);
    let mut pending_resets = Vec::new();
    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        3,
        &store,
        true,
        false,
        false,
        &mut pending_resets,
    )
    .expect("EOS handoff admission");

    assert!(pending.is_empty());
    assert_eq!(slots.iter().flatten().count(), 0, "no slot forms");
    assert!(matches!(
        rx.try_recv(),
        Ok((_, pegainfer_frontend::engine::TokenEvent::Scheduled { .. }))
    ));
    assert!(matches!(
        rx.try_recv(),
        Ok((
            _,
            pegainfer_frontend::engine::TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                completion_tokens: 1,
                ..
            }
        ))
    ));
}

#[test]
fn suppressed_eos_finishes_behind_a_budget_stalled_front() {
    // An EOS-only handoff needs no slot and no KV: the intake sweep must
    // finish it even when the FIFO front is budget-stalled and would
    // otherwise park the whole queue.
    let pool = Arc::new(BlockPool::new(PAGE, 4));
    let (store, _rt) = test_store(&pool);
    let mut stalled = request(vec![20, 21, 22], SamplingParams::default(), PAGE * 8);
    let (stalled_tx, _stalled_rx) = TokenSink::standalone();
    stalled.token_tx = stalled_tx;
    let handoff = NativeMtpHandoff {
        fingerprint: super::offload::handoff_fingerprint(),
        committed_len: 3,
        anchor_token_id: None,
        draft_tokens: Vec::new(),
    };
    let mut req = request(vec![10, 11, 12], SamplingParams::default(), 8);
    let (tx, mut rx) = TokenSink::standalone();
    req.token_tx = tx;

    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::from([
        Resolved::Plain {
            req: stalled,
            prefix: KvPrefix::none(),
        },
        Resolved::Native {
            req,
            prefix: KvPrefix::none(),
            handoff,
        },
    ]);
    let mut pending_resets = Vec::new();
    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        1,
        &store,
        true,
        false,
        false,
        &mut pending_resets,
    )
    .expect("EOS sweep behind stalled front");

    assert_eq!(pending.len(), 1, "the stalled front stays queued");
    assert!(matches!(pending.front(), Some(Resolved::Plain { .. })));
    assert_eq!(slots.iter().flatten().count(), 0, "no slot forms");
    assert!(matches!(
        rx.try_recv(),
        Ok((_, pegainfer_frontend::engine::TokenEvent::Scheduled { .. }))
    ));
    assert!(matches!(
        rx.try_recv(),
        Ok((
            _,
            pegainfer_frontend::engine::TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                completion_tokens: 1,
                ..
            }
        ))
    ));
}

#[test]
fn anchor_exhausting_max_tokens_finishes_as_length() {
    // The replayed anchor is the request's whole budget: finish before any
    // restore work — the prefix hold just drops.
    let pool = Arc::new(BlockPool::new(PAGE, 4));
    let (store, _rt) = test_store(&pool);
    let handoff = NativeMtpHandoff {
        fingerprint: super::offload::handoff_fingerprint(),
        committed_len: 3,
        anchor_token_id: Some(70_001),
        draft_tokens: Vec::new(),
    };
    let mut req = request(vec![10, 11, 12], SamplingParams::default(), 1);
    let (tx, mut rx) = TokenSink::standalone();
    req.token_tx = tx;

    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::from([Resolved::Native {
        req,
        prefix: KvPrefix::none(),
        handoff,
    }]);
    let mut pending_resets = Vec::new();
    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        3,
        &store,
        true,
        false,
        false,
        &mut pending_resets,
    )
    .expect("anchored-length admission");

    assert_eq!(slots.iter().flatten().count(), 0, "no slot forms");
    assert!(matches!(
        rx.try_recv(),
        Ok((_, pegainfer_frontend::engine::TokenEvent::Scheduled { .. }))
    ));
    assert!(matches!(
        rx.try_recv(),
        Ok((
            _,
            pegainfer_frontend::engine::TokenEvent::Token { id: 70_001, .. }
        ))
    ));
    assert!(matches!(
        rx.try_recv(),
        Ok((
            _,
            pegainfer_frontend::engine::TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                completion_tokens: 1,
                ..
            }
        ))
    ));
}

#[test]
fn budget_stalled_front_lets_a_fitting_native_bypass() {
    // The FIFO stall shape: the front's lifetime cannot fit the budget, and
    // a native queued behind it pins its restored pages until its own
    // admission — which the stuck front blocks. Admitting the fitting
    // native out of order is the only transition that ever releases them.
    let pool = Arc::new(BlockPool::new(PAGE, 8));
    let (store, rt) = test_store(&pool);

    let mut front = request(vec![2; 6 * PAGE], SamplingParams::default(), 1);
    let (front_tx, _front_rx) = TokenSink::standalone();
    front.token_tx = front_tx;

    let committed: Vec<u32> = (0..(PAGE + 7) as u32).map(|t| 80_000 + t).collect();
    let (req, prefix, handoff, _rx) = resolved_native(&pool, &store, &rt, committed, 70_001, 8);

    let mut slots: RankSlots = std::array::from_fn(|_| None);
    let mut pending = VecDeque::from([
        plain(front),
        Resolved::Native {
            req,
            prefix,
            handoff,
        },
    ]);
    let mut pending_resets = Vec::new();
    admit_from_queue(
        0,
        &mut pending,
        &mut slots,
        &pool,
        5,
        &store,
        true,
        false,
        false,
        &mut pending_resets,
    )
    .expect("bypassing a budget-stalled front is not an admission error");

    assert_eq!(slots.iter().flatten().count(), 1, "the native admitted");
    assert!(
        slots
            .iter()
            .flatten()
            .any(|active| active.client_prompt_tokens == PAGE + 7),
        "the admitted slot is the native, not the front"
    );
    assert_eq!(pending.len(), 1, "the too-big front stays queued in order");
}

/// Drive one request end to end through the engine's exact schedule/apply
/// sequence against `pool` — the offline replica of the two engine-fatal
/// submit-walk assertions (span start == `kv_position`, schedule never fails
/// under the admission reservation). Verify spans fully accept their drafts,
/// maximizing the KV draw per round. Returns the first schedule failure (the
/// tight-budget control asserts one).
fn drive_request(
    pool: &BlockPool,
    prompt_len: usize,
    max_tokens: usize,
    with_drafts: bool,
) -> Result<(), String> {
    let prompt: Vec<u32> = (0..prompt_len as u32).map(|t| 10_000 + t).collect();
    let mut state = Glm52SlotState::new(prompt.clone(), max_tokens, true, 0);
    let mut kv = pool.new_request(prompt, max_tokens, None);
    let mut fresh = 60_000u32;
    loop {
        if with_drafts && state.wants_drafts() {
            state.set_drafts(
                vec![70_001, 70_002, 70_003, 70_004, 70_005, 70_006, 70_007],
                GLM52_DSPARK_EP8_SPAN_DRAFTS,
            );
        }
        let span = state.feed_want().min(GLM52_MAX_BATCH_PER_RANK);
        assert_eq!(
            state.next_input_at(0).position,
            kv.kv_position(),
            "span start drifted from the pool's kv_position"
        );
        let mid_prefill = state.mid_prefill();
        if mid_prefill {
            kv.schedule_prefill(span, pool)
                .map_err(|e| format!("schedule_prefill: {e}"))?;
        } else if span == 1 {
            kv.schedule_decode(pool)
                .map_err(|e| format!("schedule_decode: {e}"))?;
        } else {
            kv.schedule_speculative(span, pool)
                .map_err(|e| format!("schedule_speculative: {e}"))?;
        }
        // The prologue's page-row coverage, offline: the exact page row
        // must cover every fed position.
        let pages = kv.step_page_indices(span);
        let last_position = state.next_input_at(span - 1).position;
        assert!(
            pages.len() * PAGE > last_position,
            "page row misses a fed position"
        );
        fresh += 1;
        // Rows 1.. echo the fed tokens (a verify span fully accepts its
        // drafts), the last row emits a fresh token.
        let outputs: Vec<u32> = (1..span)
            .map(|offset| state.next_input_at(offset).token)
            .chain(std::iter::once(fresh))
            .collect();
        match state.advance_span(&outputs, &[]) {
            Glm52StepOutcome::Prefilling => {
                kv.apply_prefill_chunk(pool).expect("apply_prefill_chunk");
            }
            Glm52StepOutcome::Commit {
                committed, finish, ..
            } => {
                if mid_prefill {
                    kv.apply_prefill(committed[0], pool).expect("apply_prefill");
                } else if span == 1 {
                    kv.apply_decode(committed[0], pool).expect("apply_decode");
                } else {
                    kv.apply_speculative(&committed, pool)
                        .expect("apply_speculative");
                }
                if finish.is_some() {
                    break;
                }
            }
        }
    }
    kv.release().map_err(|e| format!("release: {e}"))?;
    Ok(())
}

#[test]
fn full_lifetime_reservation_covers_kvbm_peak_draw() {
    // The submit walk turns any schedule failure into an engine
    // exit; this is that contract's offline test. A pool sized
    // exactly `lifetime_blocks + 1` (padding) must carry every shape end
    // to end — and one block less must NOT, or the reservation is merely
    // sufficient by accident, not tight.
    for &(prompt_len, max_tokens) in &[
        (64usize, 64usize),
        (64, 65),
        (63, 65),
        (1, 128),
        (127, 2),
        (192, 3),
        (65, 1),
    ] {
        for with_drafts in [false, true] {
            let lifetime = lifetime_blocks(prompt_len, max_tokens);
            let pool = Arc::new(BlockPool::new(PAGE, lifetime + 1));
            drive_request(&pool, prompt_len, max_tokens, with_drafts).unwrap_or_else(|e| {
                panic!("({prompt_len},{max_tokens},drafts={with_drafts}): {e}")
            });
            let tight = Arc::new(BlockPool::new(PAGE, lifetime));
            assert!(
                drive_request(&tight, prompt_len, max_tokens, with_drafts).is_err(),
                "({prompt_len},{max_tokens},drafts={with_drafts}): a budget below the \
                 lifetime must fail somewhere"
            );
        }
    }
}

#[test]
fn eos_truncated_speculative_apply_stays_in_contract() {
    // EOS mid-verify-span truncates `committed` (the suppressed EOS is
    // its last entry); `apply_speculative` with the truncated run and
    // the release must both stay clean.
    let pool = Arc::new(BlockPool::new(PAGE, 16));
    let prompt: Vec<u32> = (0..70).collect();
    let mut state = Glm52SlotState::new(prompt.clone(), 32, false, 0);
    let mut kv = pool.new_request(prompt, 32, None);
    loop {
        if !state.mid_prefill() {
            break;
        }
        let span = state.feed_want().min(GLM52_MAX_BATCH_PER_RANK);
        assert_eq!(state.next_input_at(0).position, kv.kv_position());
        kv.schedule_prefill(span, &pool).expect("schedule_prefill");
        match state.advance_span(&vec![50u32; span], EOS) {
            Glm52StepOutcome::Prefilling => {
                kv.apply_prefill_chunk(&pool).expect("apply_prefill_chunk");
            }
            Glm52StepOutcome::Commit { committed, .. } => {
                kv.apply_prefill(committed[0], &pool)
                    .expect("apply_prefill");
            }
        }
    }
    state.set_drafts(vec![21, 7, 23], GLM52_DSPARK_EP8_SPAN_DRAFTS);
    let span = state.feed_want();
    assert_eq!(span, 4, "anchor + 3 drafts");
    assert_eq!(state.next_input_at(0).position, kv.kv_position());
    kv.schedule_speculative(span, &pool)
        .expect("schedule_speculative");
    let outcome = state.advance_span(&[21, 7, 23, 99], EOS);
    let Glm52StepOutcome::Commit {
        committed,
        emit,
        finish,
        ..
    } = outcome
    else {
        panic!("verify span must commit");
    };
    assert_eq!(committed, vec![21, 7], "truncated to the consumed run");
    assert_eq!(emit, 1, "the suppressed EOS is consumed, not emitted");
    assert_eq!(finish, Some(FinishReason::Stop));
    kv.apply_speculative(&committed, &pool)
        .expect("apply_speculative with the truncated run");
    kv.release().expect("release");
}
