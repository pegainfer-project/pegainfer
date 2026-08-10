//! Per-step planning for one rank: its own bucket and row list
//! ([`plan_step_shape`]), the rank-local launch-ahead decision
//! ([`lease_flags`]), and the rows the sampler owns instead of the fused
//! argmax ([`collect_sampling_rows`]) — pure functions over the rank's
//! occupancy and feed wants. Buckets are rank-local (the collectives take
//! rank-local row counts and the conservative protocol-max bound; see
//! `docs/models/glm52/free-running-dp.md` §2), and the launch-ahead lease is
//! rank-local too: a free-running engine always consumes its own
//! speculation, so no cross-rank agreement exists (see [`lease_flags`]).

use pegainfer_sample::SamplingParams;

use super::PAGE;
use super::RankSlots;
use super::slot::Glm52SlotState;
use crate::config::GLM52_VOCAB;
use crate::model::GLM52_DECODE_BUCKETS;
use crate::model::GLM52_MAX_BATCH_PER_RANK;
use crate::model::GLM52_MAX_STEP_ROWS;
use crate::model::Glm52StepKv;
use crate::model::Glm52StepShape;
use crate::runner::Glm52RowSample;
use crate::runner::Glm52StepFlags;

/// Build the all-padding step KV used while pre-capturing graph buckets.
pub(super) fn padding_step_kv(
    bucket: usize,
    table_width: usize,
    padding_page: i32,
    inputs: &[(u32, usize); GLM52_MAX_STEP_ROWS],
) -> Glm52StepKv {
    let pages = vec![padding_page; bucket * table_width].into_boxed_slice();
    let mut slot_mapping = [0i64; GLM52_MAX_STEP_ROWS];
    for (row, slot) in slot_mapping.iter_mut().enumerate().take(bucket) {
        *slot = padding_page as i64 * PAGE as i64 + (inputs[row].1 % PAGE) as i64;
    }
    Glm52StepKv {
        pages,
        slot_mapping,
        boundary_copies: Vec::new(),
    }
}

/// One rank's forward shape for one step, decided from its feed-want
/// snapshot (`wants[slot]` = rows that slot can usefully fill: 0 free, 1
/// decode, remaining-prompt while mid-prefill).
///
/// The bucket is the smallest [`GLM52_DECODE_BUCKETS`] member covering the
/// rank's OWN row demand (Σ wants, capped at the max bucket; never smaller
/// than its active count — a smaller bucket would silently drop rows).
/// Buckets are rank-local: the MoE collectives take each rank's real row
/// count and the conservative protocol-max GEMM bound
/// (`ep_ranks × GLM52_MAX_BATCH_PER_RANK`), so no rank pays compute for
/// another rank's demand (the free-running gates measured the conservative
/// bound at zero cost — `docs/models/glm52/free-running-dp.md` §8).
/// Every active slot first gets one row (liveness), then the leftover bucket
/// capacity extends mid-prefill slots into *spans* (consecutive prompt
/// positions batched through one step), round-robin across the hungry slots
/// so co-resident prefills drain in parallel; rows past `active_rows` are
/// padding (their slot ids are insignificant — see the body). Span rows are
/// emitted as one contiguous run per slot — the [`Glm52StepShape`] contract.
pub(super) fn plan_step_shape(wants: &[usize; GLM52_MAX_BATCH_PER_RANK]) -> Glm52StepShape {
    let demand = wants.iter().sum::<usize>().min(GLM52_MAX_STEP_ROWS);
    let bucket = *GLM52_DECODE_BUCKETS
        .iter()
        .find(|&&rows| rows >= demand.max(1))
        .expect("the largest bucket covers every demand by construction");
    let spans = plan_prefill_spans(wants, bucket);
    let mut slots: [u8; GLM52_MAX_STEP_ROWS] = [0; GLM52_MAX_STEP_ROWS];
    let mut dst = 0usize;
    for (slot, &span) in spans.iter().enumerate() {
        for _ in 0..span {
            slots[dst] = slot as u8;
            dst += 1;
        }
    }
    // Rows `active_rows..bucket` are padding. Their slot ids are
    // insignificant — a bucket may hold more rows than there are slots
    // (verify spans, #812), so padding no longer maps onto distinct free
    // slots. Every consumer bounds its semantic walk by `active_rows`;
    // padding rows keep deterministic defaults (padding token/position,
    // padding-page KV) regardless of the id here.
    let active_rows = dst;
    Glm52StepShape {
        bucket,
        slots,
        active_rows,
    }
}

/// The rank-local launch-ahead flag decision — pure so the rules are
/// testable. A free-running engine ALWAYS consumes its own speculation:
/// leasing freezes the slot set for exactly one step (newcomers wait in the
/// pending queue, finishes defer their physical release to the consume
/// step), so `consume` is a structural guarantee the caller passes through,
/// not a decision made here. The coordinator-era stale-replay re-run — a
/// second collective chain that only paired if every rank speculated
/// together — does not exist anymore.
///
/// `lease`: enqueue the next step speculatively — pure single-token GREEDY
/// decode everywhere (the speculation feeds each row's argmax token, so a
/// sampled row would replay the wrong input) with model-length headroom, off
/// every 64-token page boundary (the feed kernel's `slot_mapping += 1` only
/// stays valid inside the current page, and the advanced step's page must
/// already be in the uploaded block table; breaking the streak at every
/// active row's boundary also bounds padding rows — reset to position 0 by
/// each full prologue — inside the padding page), nothing queued, no draft
/// round, and no deferred releases pending (a dead slot's rows leave the
/// shape after the consume step, so it must not be leased into the next
/// one). An idle rank must NOT lease: with zero active rows the per-row
/// boundary break never fires, and chained replays would walk the padding
/// row's slot_mapping out of the padding page.
///
/// `offload_enabled` kills the lease outright: a leased replay keeps writing
/// KV on the rank stream for ~a step after the engine joined its argmax D2H,
/// and the offload restore leg H2Ds into freshly-reallocated pool pages on
/// pegaflow's OWN stream at the very next admission — the two are unordered,
/// so a replay row landing after the restore would silently poison a
/// content-addressed block for every later match. Without leases, the joined
/// D2H is the last thing on the rank stream and admission truly is a quiet
/// boundary. Costs ~0.7 ms/step, offload deployments only.
pub(super) fn lease_flags(
    consume: bool,
    pending_empty: bool,
    drafter_enabled: bool,
    offload_enabled: bool,
    deferred_pending: bool,
    slots: &RankSlots,
    max_model_len: usize,
) -> Glm52StepFlags {
    let lease = pending_empty
        && !drafter_enabled
        && !offload_enabled
        && !deferred_pending
        && slots.iter().any(Option::is_some)
        && slots.iter().flatten().all(|active| {
            takes_argmax(&active.req.params) && lease_ok(&active.state, max_model_len)
        });
    // `GLM52_EAGER_DECODE=1` runs every decode step through the eager path
    // (same step body, no graph capture/replay) — the stage-3 measurement
    // switch for the graph-vs-padding trade at wide buckets. Eager steps
    // cannot ride launch-ahead replays, so the lease is withheld too.
    let eager = eager_decode_forced();
    Glm52StepFlags {
        consume,
        lease: lease && !eager,
        eager,
    }
}

/// Whether `GLM52_EAGER_DECODE=1` forces the eager (graph-free) decode path.
fn eager_decode_forced() -> bool {
    static EAGER: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *EAGER.get_or_init(|| std::env::var("GLM52_EAGER_DECODE").as_deref() == Ok("1"))
}

/// Whether a request's committed rows take the fused argmax — the shared
/// effectively-greedy predicate over the GLM vocab (a `top_p <= 1/vocab`
/// nucleus holds only the argmax token; routing it to the sampler would make
/// bf16-tied maxima stochastic, diverging from `select_batch`'s semantics).
/// The SAME predicate gates lease-granting and sampling-row collection, which
/// is what keeps "sampled row never rides a launch-ahead step" structural.
pub(super) fn takes_argmax(params: &SamplingParams) -> bool {
    pegainfer_sample::effectively_greedy(params, GLM52_VOCAB)
}

/// Whether one active request's KV position permits leasing the next step: a
/// pure single-token decode row with model-length headroom whose advanced
/// position stays inside its current 64-token page (see [`lease_flags`] for
/// why the page boundary breaks the streak).
fn lease_ok(state: &Glm52SlotState, max_model_len: usize) -> bool {
    let position = state.next_input_at(0).position;
    state.feed_want() == 1 && position + 1 < max_model_len && !(position + 1).is_multiple_of(PAGE)
}

/// The step rows a rank samples instead of argmaxes: walk the shape's
/// contiguous per-slot runs and mark each non-greedy slot's committable rows
/// (see [`Glm52SlotState::sampling_rows`]) with their request params and
/// request-local decode steps. Rows come out strictly ascending — the runs
/// are disjoint and walked in order, offsets ascend within a run — which
/// `sample_rows_into` re-checks.
pub(super) fn collect_sampling_rows(
    shape: &Glm52StepShape,
    rank_slots: &RankSlots,
) -> Vec<Glm52RowSample> {
    let mut sampling = Vec::new();
    let mut row = 0usize;
    while row < shape.active_rows {
        let slot = shape.slots[row] as usize;
        let mut end = row + 1;
        while end < shape.active_rows && shape.slots[end] as usize == slot {
            end += 1;
        }
        if let Some(active) = &rank_slots[slot]
            && !takes_argmax(&active.req.params)
        {
            for (offset, step) in active.state.sampling_rows(end - row) {
                sampling.push(Glm52RowSample {
                    row: row + offset,
                    params: active.req.params,
                    step,
                });
            }
        }
        row = end;
    }
    sampling
}

pub(super) fn feed_wants(slots: &RankSlots) -> [usize; GLM52_MAX_BATCH_PER_RANK] {
    std::array::from_fn(|slot| {
        slots[slot]
            .as_ref()
            .map_or(0, |active| active.state.feed_want())
    })
}

/// Split one prefill launch across active requests without exceeding the
/// large-M row budget. Every active request gets one row before remaining
/// rows are distributed round-robin.
pub(super) fn plan_prefill_spans(
    wants: &[usize; GLM52_MAX_BATCH_PER_RANK],
    max_rows: usize,
) -> [usize; GLM52_MAX_BATCH_PER_RANK] {
    assert!(max_rows > 0, "prefill row budget must be positive");
    let mut spans = [0; GLM52_MAX_BATCH_PER_RANK];
    let mut used = 0;
    for (slot, &want) in wants.iter().enumerate() {
        if want > 0 && used < max_rows {
            spans[slot] = 1;
            used += 1;
        }
    }
    while used < max_rows {
        let mut advanced = false;
        for (slot, &want) in wants.iter().enumerate() {
            if used == max_rows {
                break;
            }
            if spans[slot] > 0 && spans[slot] < want {
                spans[slot] += 1;
                used += 1;
                advanced = true;
            }
        }
        if !advanced {
            break;
        }
    }
    spans
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::ActiveRequest;
    use crate::scheduler::slot::Glm52StepOutcome;
    use crate::scheduler::testkit::EOS;
    use crate::scheduler::testkit::commit;
    use crate::scheduler::testkit::request;
    use crate::scheduler::testkit::sampled;
    use crate::scheduler::testkit::state;
    use crate::scheduler::testkit::test_kv;

    #[test]
    fn consume_is_a_structural_passthrough() {
        // The engine freezes the slot set while leased, so consume is never
        // a decision: the flag returned equals the flag handed in.
        let greedy = decoding_rank(pegainfer_sample::SamplingParams::default());
        let flags = lease_flags(true, true, false, false, false, &greedy, 4096);
        assert!(flags.consume);
        let flags = lease_flags(false, true, false, false, false, &greedy, 4096);
        assert!(!flags.consume);
    }

    #[test]
    fn idle_rank_never_leases() {
        // Zero actives vacuously satisfy the per-row lease rules; without
        // the explicit guard a chained lease would walk the padding row's
        // slot_mapping out of the padding page.
        let slots: RankSlots = std::array::from_fn(|_| None);
        let flags = lease_flags(false, true, false, false, false, &slots, 4096);
        assert!(!flags.lease && !flags.consume);
    }

    #[test]
    fn deferred_releases_break_the_lease_chain() {
        // A finish under a lease defers its physical release to the consume
        // step; that step must not re-lease — the dead slot's rows leave the
        // shape right after it.
        let greedy = decoding_rank(pegainfer_sample::SamplingParams::default());
        let flags = lease_flags(true, true, false, false, true, &greedy, 4096);
        assert!(flags.consume && !flags.lease);
    }

    #[test]
    fn no_lease_without_an_empty_queue() {
        let greedy = decoding_rank(pegainfer_sample::SamplingParams::default());
        let flags = lease_flags(false, false, false, false, false, &greedy, 4096);
        assert!(!flags.lease && !flags.consume);
    }

    /// One rank holding a single decoding request with the given params (its
    /// prompt token is already fed, so `feed_want() == 1`).
    fn decoding_rank(params: pegainfer_sample::SamplingParams) -> RankSlots {
        let req = request(vec![10], params, 8);
        let mut state = Glm52SlotState::new(req.prompt_tokens.clone(), req.max_tokens, false, 0);
        assert!(matches!(
            state.advance_span(&[20], &[]),
            Glm52StepOutcome::Commit { .. }
        ));
        let kv = test_kv(req.prompt_tokens.clone(), req.max_tokens);
        let mut slots: RankSlots = std::array::from_fn(|_| None);
        slots[0] = Some(ActiveRequest {
            req,
            state,
            client_prompt_tokens: 1,
            kv,
            save_cursor: pegainfer_kv_store::SaveCursor::new(),
            boundary_copy: None,
        });
        slots
    }

    #[test]
    fn offload_blocks_the_lease() {
        // A leased replay keeps writing KV on the rank stream after the
        // join; the offload restore H2Ds on pegaflow's stream, unordered
        // against it. Offload on ⇒ never lease.
        let greedy = decoding_rank(pegainfer_sample::SamplingParams::default());
        assert!(!lease_flags(false, true, false, true, false, &greedy, 4096).lease);
    }

    #[test]
    fn drafter_blocks_the_lease() {
        let greedy = decoding_rank(pegainfer_sample::SamplingParams::default());
        assert!(!lease_flags(false, true, true, false, false, &greedy, 4096).lease);
    }

    #[test]
    fn non_greedy_request_blocks_the_lease() {
        // The speculation feeds each row's argmax token; a sampled row would
        // replay the wrong input, so any non-greedy active blocks the lease.
        let greedy = decoding_rank(pegainfer_sample::SamplingParams::default());
        assert!(lease_flags(false, true, false, false, false, &greedy, 4096).lease);

        let sampled = decoding_rank(pegainfer_sample::SamplingParams {
            temperature: 0.7,
            ..Default::default()
        });
        assert!(!lease_flags(false, true, false, false, false, &sampled, 4096).lease);

        // An effectively-greedy request (top_p nucleus <= 1/vocab holds only
        // the argmax token) takes the argmax path, so it may ride the lease.
        let tiny_top_p = decoding_rank(pegainfer_sample::SamplingParams {
            temperature: 0.7,
            top_p: 0.5 / GLM52_VOCAB as f32,
            ..Default::default()
        });
        assert!(lease_flags(false, true, false, false, false, &tiny_top_p, 4096).lease);
    }

    #[test]
    fn collect_sampling_rows_marks_each_spans_committable_rows() {
        // Bucket 8: slot 0 runs a 2-row verify span (non-greedy, drafts
        // installed), slot 1 finishes its prompt with a 3-row span
        // (non-greedy), slot 3 is mid-prompt (non-greedy, span does NOT
        // complete), slot 2 decodes greedily, row 7 pads.
        let mut slots = [0u8; GLM52_MAX_STEP_ROWS];
        slots[..8].copy_from_slice(&[0, 0, 1, 1, 1, 3, 2, 4]);
        let shape = Glm52StepShape {
            bucket: 8,
            slots,
            active_rows: 7,
        };
        let mut rank_slots: RankSlots = std::array::from_fn(|_| None);

        let mut decode_state = state(vec![10], 8, false);
        assert_eq!(
            decode_state.advance_span(&[20], EOS),
            commit(&[20], 1, None, 1)
        );
        decode_state.set_drafts(
            vec![50, 51, 52],
            crate::scheduler::slot::GLM52_DSPARK_EP8_SPAN_DRAFTS,
        );
        rank_slots[0] = Some(ActiveRequest {
            req: request(vec![10], sampled(0.8), 8),
            state: decode_state,
            client_prompt_tokens: 1,
            kv: test_kv(vec![10], 8),
            save_cursor: pegainfer_kv_store::SaveCursor::new(),
            boundary_copy: None,
        });

        let mut boundary_state = state(vec![10, 11, 12, 13, 14], 8, false);
        assert_eq!(
            boundary_state.advance_span(&[99, 98], EOS),
            Glm52StepOutcome::Prefilling
        );
        rank_slots[1] = Some(ActiveRequest {
            req: request(vec![10, 11, 12, 13, 14], sampled(0.8), 8),
            state: boundary_state,
            client_prompt_tokens: 5,
            kv: test_kv(vec![10, 11, 12, 13, 14], 8),
            save_cursor: pegainfer_kv_store::SaveCursor::new(),
            boundary_copy: None,
        });

        let mut greedy_state = state(vec![10], 8, false);
        assert_eq!(
            greedy_state.advance_span(&[20], EOS),
            commit(&[20], 1, None, 1)
        );
        rank_slots[2] = Some(ActiveRequest {
            req: request(vec![10], pegainfer_sample::SamplingParams::default(), 8),
            state: greedy_state,
            client_prompt_tokens: 1,
            kv: test_kv(vec![10], 8),
            save_cursor: pegainfer_kv_store::SaveCursor::new(),
            boundary_copy: None,
        });

        rank_slots[3] = Some(ActiveRequest {
            req: request(vec![30; 10], sampled(0.8), 8),
            state: state(vec![30; 10], 8, false),
            client_prompt_tokens: 10,
            kv: test_kv(vec![30; 10], 8),
            save_cursor: pegainfer_kv_store::SaveCursor::new(),
            boundary_copy: None,
        });

        let rows = collect_sampling_rows(&shape, &rank_slots);
        let picked: Vec<(usize, u64)> = rows.iter().map(|s| (s.row, s.step)).collect();
        // Slot 0's verify span samples BOTH rows (anchor row 0 at step 1,
        // draft row 1 at step 2 — the planner granted 2 of its 4 wanted
        // rows); slot 1's boundary span commits its LAST row (row 2 +
        // offset 2 = 4, first generated token → step 0). Slot 3's
        // mid-prompt span and slot 2's greedy row contribute nothing.
        assert_eq!(picked, vec![(0, 1), (1, 2), (4, 0)]);
    }

    #[test]
    fn effectively_greedy_rows_take_the_argmax_path() {
        // temperature > 0 but the top_p nucleus (<= 1/vocab) holds only the
        // argmax token: the row must NOT be collected for the sampler — the
        // FlashInfer pass could pick a different bf16-tied maximum, whereas
        // `select_batch` pins this case to the deterministic argmax.
        let shape = Glm52StepShape {
            bucket: 1,
            slots: [0; GLM52_MAX_STEP_ROWS],
            active_rows: 1,
        };
        let mut rank_slots: RankSlots = std::array::from_fn(|_| None);
        let mut state = state(vec![10], 8, false);
        assert_eq!(state.advance_span(&[20], EOS), commit(&[20], 1, None, 1));
        rank_slots[0] = Some(ActiveRequest {
            req: request(
                vec![10],
                pegainfer_sample::SamplingParams {
                    top_p: 0.5 / GLM52_VOCAB as f32,
                    ..sampled(0.8)
                },
                8,
            ),
            state,
            client_prompt_tokens: 1,
            kv: test_kv(vec![10], 8),
            save_cursor: pegainfer_kv_store::SaveCursor::new(),
            boundary_copy: None,
        });
        assert!(collect_sampling_rows(&shape, &rank_slots).is_empty());
    }

    #[test]
    fn lease_breaks_at_the_page_boundary() {
        // Anchor at position 62 → the next position 63 stays in page 0:
        // lease ok. Anchor at position 63 → position 64 opens page 1: the
        // feed kernel's `slot_mapping += 1` would leave the page — no lease.
        let mut s = state((0..63).collect(), 8, false);
        let mut outputs = vec![99u32; 63];
        *outputs.last_mut().unwrap() = 42;
        assert_eq!(s.advance_span(&outputs, EOS), commit(&[42], 1, None, 63));
        assert_eq!(s.next_input_at(0).position, 63);
        assert!(!lease_ok(&s, 4096), "position 63 -> 64 crosses the page");
        assert_eq!(s.advance_span(&[43], EOS), commit(&[43], 1, None, 1));
        assert_eq!(s.next_input_at(0).position, 64);
        assert!(lease_ok(&s, 4096), "position 64 -> 65 stays inside page 1");
        // Model-length headroom still gates.
        assert!(!lease_ok(&s, 65));
    }

    /// `counts` decode-phase requests per rank (each wants one row).
    fn decode_wants(count: usize) -> [usize; GLM52_MAX_BATCH_PER_RANK] {
        std::array::from_fn(|slot| usize::from(slot < count))
    }

    /// The observable part of a shape: the bucket and the forwarded rows'
    /// slots (trailing entries beyond the bucket are never read).
    fn forwarded(shape: &Glm52StepShape) -> (usize, Vec<u8>) {
        (shape.bucket, shape.slots[..shape.bucket].to_vec())
    }

    #[test]
    fn bucket_is_the_smallest_covering_the_ranks_own_demand() {
        assert_eq!(forwarded(&plan_step_shape(&decode_wants(0))), (1, vec![0]));
        assert_eq!(forwarded(&plan_step_shape(&decode_wants(1))), (1, vec![0]));
        assert_eq!(
            forwarded(&plan_step_shape(&decode_wants(2))),
            (2, vec![0, 1])
        );
        // The padding row reuses slot id 0 (#812) — insignificant past
        // `active_rows`.
        assert_eq!(
            forwarded(&plan_step_shape(&decode_wants(3))),
            (4, vec![0, 1, 2, 0])
        );
        assert_eq!(plan_step_shape(&decode_wants(3)).active_rows, 3);
        // Past the 4-row bucket the eight-row bucket takes over.
        assert_eq!(plan_step_shape(&decode_wants(5)).bucket, 8);
    }

    #[test]
    fn partial_buckets_pack_actives_first() {
        // A rank holding slots {1, 5} forwards them in rows 0..2 in its own
        // bucket-2 shape; a rank with 5 actives rides a bucket-8 shape whose
        // padding rows carry the insignificant slot id 0 (#812).
        let mut holey = decode_wants(0);
        holey[1] = 1;
        holey[5] = 1;
        assert_eq!(forwarded(&plan_step_shape(&holey)), (2, vec![1, 5]));
        let mut deep = decode_wants(5);
        deep[0] = 0;
        deep[7] = 1;
        assert_eq!(
            forwarded(&plan_step_shape(&deep)),
            (8, vec![1, 2, 3, 4, 7, 0, 0, 0])
        );
        assert_eq!(plan_step_shape(&deep).active_rows, 5);
    }

    #[test]
    fn prefill_want_extends_one_slot_into_a_span() {
        // A lone mid-prefill request with plenty of prompt left fills the
        // whole max bucket with its span — 48 rows since #812.
        let mut wants = decode_wants(0);
        wants[2] = 3000;
        assert_eq!(
            forwarded(&plan_step_shape(&wants)),
            (GLM52_MAX_STEP_ROWS, vec![2; GLM52_MAX_STEP_ROWS]),
            "one hungry slot owns every row of the max bucket"
        );

        // A short prompt remainder only lifts the bucket as far as needed.
        let mut wants = decode_wants(0);
        wants[0] = 3;
        assert_eq!(forwarded(&plan_step_shape(&wants)), (4, vec![0, 0, 0, 0]));
    }

    #[test]
    fn spans_share_the_bucket_with_decode_slots_actives_first() {
        // Slot 0 decodes (1 row), slot 1 is mid-prefill: liveness rows first,
        // then the leftover capacity extends the prefill span — one
        // contiguous run per slot.
        let mut wants = decode_wants(0);
        wants[0] = 1;
        wants[1] = 100;
        let mut expect = vec![1u8; GLM52_MAX_STEP_ROWS];
        expect[0] = 0;
        assert_eq!(
            forwarded(&plan_step_shape(&wants)),
            (GLM52_MAX_STEP_ROWS, expect)
        );

        // Two mid-prefill slots with small wants: both met, remaining rows
        // are padding (slot id 0, insignificant).
        let mut wants = decode_wants(0);
        wants[0] = 3;
        wants[1] = 2;
        assert_eq!(
            forwarded(&plan_step_shape(&wants)),
            (8, vec![0, 0, 0, 1, 1, 0, 0, 0]),
            "wants met, remaining rows are padding"
        );
        let shape = plan_step_shape(&wants);
        assert_eq!(shape.active_rows, 5);
    }

    #[test]
    fn two_long_prefills_split_the_leftover_round_robin() {
        // Two co-resident long prefills split the bucket evenly — neither
        // starves at a single liveness row while the other eats the leftover.
        let mut wants = decode_wants(0);
        wants[2] = 3000;
        wants[5] = 3000;
        let half = GLM52_MAX_STEP_ROWS / 2;
        let mut expect = vec![2u8; half];
        expect.extend(std::iter::repeat_n(5u8, half));
        assert_eq!(
            forwarded(&plan_step_shape(&wants)),
            (GLM52_MAX_STEP_ROWS, expect)
        );

        // A decode slot in the mix keeps its single row; the prefills split
        // what remains (47 rows -> 24 + 23 by round-robin order).
        let mut wants = decode_wants(0);
        wants[0] = 1;
        wants[3] = 3000;
        wants[6] = 3000;
        let mut expect = vec![0u8];
        expect.extend(std::iter::repeat_n(3u8, 48));
        expect.extend(std::iter::repeat_n(6u8, 47));
        assert_eq!(
            forwarded(&plan_step_shape(&wants)),
            (GLM52_MAX_STEP_ROWS, expect)
        );
    }

    /// The #812 headline: eight verifying slots each want a full native-MTP
    /// span (1 anchor + 5 drafts) and ALL of them get it — pre-#812 the
    /// demand cap at the slot count collapsed every span to one row
    /// (measured as TPOT == ITL at full occupancy).
    #[test]
    fn full_occupancy_verify_spans_fit_the_max_bucket() {
        // The latency profile: 16 slots riding the full 5-draft span fill
        // the max bucket exactly (8 slots land on the 48 bucket unchanged).
        let mut wants = [0usize; GLM52_MAX_BATCH_PER_RANK];
        wants[..16].fill(6);
        let shape = plan_step_shape(&wants);
        assert_eq!(shape.bucket, GLM52_MAX_STEP_ROWS);
        assert_eq!(shape.active_rows, GLM52_MAX_STEP_ROWS);
        let mut expect = Vec::new();
        for slot in 0..16 {
            expect.extend(std::iter::repeat_n(slot as u8, 6));
        }
        assert_eq!(shape.slots[..GLM52_MAX_STEP_ROWS].to_vec(), expect);
        let mut eight = [0usize; GLM52_MAX_BATCH_PER_RANK];
        eight[..8].fill(6);
        assert_eq!(plan_step_shape(&eight).bucket, 48);
    }

    #[test]
    fn wide_occupancy_two_draft_spans_fit_the_max_bucket() {
        // The throughput profile: 16 slots riding a 2-draft span — the
        // ceiling slot count saturates the same 48-row budget exactly.
        let wants = [3usize; GLM52_MAX_BATCH_PER_RANK];
        let shape = plan_step_shape(&wants);
        assert_eq!(shape.bucket, GLM52_MAX_STEP_ROWS);
        assert_eq!(shape.active_rows, GLM52_MAX_STEP_ROWS);
        let mut expect = Vec::new();
        for slot in 0..GLM52_MAX_BATCH_PER_RANK {
            expect.extend(std::iter::repeat_n(slot as u8, 3));
        }
        assert_eq!(shape.slots[..GLM52_MAX_STEP_ROWS].to_vec(), expect);
    }

    #[test]
    fn over_committed_wants_shrink_round_robin_not_collapse() {
        // A mis-paired (slots, drafts) config (launch() rejects it, but the
        // planner must still be safe): 16 slots each wanting the 6-row span
        // get round-robined down to 3 rows apiece, never to a bare anchor.
        let wants = [6usize; GLM52_MAX_BATCH_PER_RANK];
        let shape = plan_step_shape(&wants);
        assert_eq!(shape.bucket, GLM52_MAX_STEP_ROWS);
        assert_eq!(shape.active_rows, GLM52_MAX_STEP_ROWS);
        for slot in 0..GLM52_MAX_BATCH_PER_RANK {
            let rows = shape.slots[..GLM52_MAX_STEP_ROWS]
                .iter()
                .filter(|&&s| s as usize == slot)
                .count();
            assert_eq!(rows, 3, "slot {slot} should get an even 3-row share");
        }
    }

    #[test]
    fn large_m_prefill_budget_is_shared_across_requests() {
        let mut wants = [0; GLM52_MAX_BATCH_PER_RANK];
        wants[0] = 20_000;
        wants[3] = 20_000;
        wants[6] = 4;
        let spans = plan_prefill_spans(&wants, 16_384);
        assert_eq!(spans.iter().sum::<usize>(), 16_384);
        assert_eq!(spans[6], 4);
        assert!(
            spans[0].abs_diff(spans[3]) <= 1,
            "long requests must share the remaining rows: {spans:?}"
        );
        let mut tight = [0; GLM52_MAX_BATCH_PER_RANK];
        (tight[0], tight[3]) = (1, 1);
        assert_eq!(plan_prefill_spans(&wants, 2), tight);
    }
}
