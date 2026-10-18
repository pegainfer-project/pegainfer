//! Per-request slot state machine: pure data transitions deciding what a
//! slot feeds each step ([`Glm52SlotState::feed_want`] /
//! [`Glm52SlotState::next_input_at`]) and what a step's span of outputs means
//! ([`Glm52SlotState::advance_span`]).

use pegainfer_frontend::engine::FinishReason;

use crate::dspark::GLM52_DSPARK_DRAFTS;
use crate::dspark::accept_prefix_match;

#[cfg(test)]
pub(crate) const MTP_PRODUCTION_GATE_REQUEST_ID: &str = "native-mtp-hidden-boundary-gate";
#[cfg(test)]
pub(crate) const MTP_SLOT_REUSE_GATE_REQUEST_ID: &str = "native-mtp-slot-reuse-gate";

#[cfg(test)]
#[derive(Clone, Debug, Default)]
pub(crate) struct MtpProductionGateStats {
    pub(crate) first_proposal: Option<Vec<u32>>,
    pub(crate) rounds: u64,
    pub(crate) accepted_drafts: u64,
    pub(crate) reuse_first_proposal: Option<Vec<u32>>,
    pub(crate) reuse_rounds: u64,
    pub(crate) reuse_accepted_drafts: u64,
}

#[cfg(test)]
static MTP_PRODUCTION_STATS: std::sync::LazyLock<std::sync::Mutex<MtpProductionGateStats>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(MtpProductionGateStats::default()));

#[cfg(test)]
pub(crate) fn reset_mtp_production_stats() {
    *MTP_PRODUCTION_STATS
        .lock()
        .expect("MTP production stats lock poisoned") = MtpProductionGateStats::default();
}

#[cfg(test)]
pub(super) fn record_mtp_proposal(request_id: Option<&str>, drafts: &[u32]) {
    if !matches!(
        request_id,
        Some(MTP_PRODUCTION_GATE_REQUEST_ID | MTP_SLOT_REUSE_GATE_REQUEST_ID)
    ) {
        return;
    }
    let mut stats = MTP_PRODUCTION_STATS
        .lock()
        .expect("MTP production stats lock poisoned");
    match request_id {
        Some(MTP_PRODUCTION_GATE_REQUEST_ID) if stats.first_proposal.is_none() => {
            stats.first_proposal = Some(drafts.to_vec());
        }
        Some(MTP_SLOT_REUSE_GATE_REQUEST_ID) if stats.reuse_first_proposal.is_none() => {
            stats.reuse_first_proposal = Some(drafts.to_vec());
        }
        _ => {}
    }
}

#[cfg(test)]
pub(crate) fn mtp_production_stats() -> MtpProductionGateStats {
    MTP_PRODUCTION_STATS
        .lock()
        .expect("MTP production stats lock poisoned")
        .clone()
}

/// What a rank forwards this step. Idle rows feed the padding input; their
/// KV/index-cache writes land in the pool's reserved padding page, which no
/// request is ever assigned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Glm52StepInput {
    pub(super) token: u32,
    pub(super) position: usize,
}

pub(super) const GLM52_PADDING_STEP: Glm52StepInput = Glm52StepInput {
    token: 0,
    position: 0,
};

/// The consequence of one step's span of outputs for one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum Glm52StepOutcome {
    /// Mid-prefill: the model's outputs are discarded, keep feeding the prompt.
    Prefilling,
    /// Commit the span's agreed tokens: `committed` is the consumed run
    /// (what advances the request's KV bookkeeping — a suppressed EOS is its
    /// last entry), of which the leading `emit` tokens are sent to the
    /// client, then finish if `finish` is set. Plain decode commits exactly
    /// one token; a verify span commits the accepted draft prefix plus the
    /// model's correction or bonus token (1..=span tokens).
    Commit {
        committed: Vec<u32>,
        emit: usize,
        finish: Option<FinishReason>,
        /// Leading span rows whose tokens are now committed context for the
        /// draft lane: all rows of a prompt span; anchor + accepted drafts of
        /// a verify span (rejected rows' captured hidden is dead).
        context_rows: usize,
    },
}

/// One rank's active request as a pure state machine: `feed_want` /
/// `next_input_at` decide what the rank forwards (a span of consecutive
/// prompt positions mid-prefill, one decode row after), `advance_span` folds
/// the step's span of outputs in.
#[derive(Debug)]
pub(super) struct Glm52SlotState {
    prompt: Vec<u32>,
    max_tokens: usize,
    ignore_eos: bool,
    /// Prompt tokens already fed to the model.
    fed: usize,
    /// Generated tokens (a suppressed EOS counts).
    completion: usize,
    /// Sampling steps already consumed outside this D request's client-visible
    /// completion budget. Manual native-P/D uses one: P sampled the forwarded
    /// anchor at step 0, while D still owes the full requested output count.
    sampling_offset: usize,
    /// The latest committed token; the next span's anchor once the prompt is
    /// fully fed.
    last_token: u32,
    /// The current spec-round proposal from the rank's draft lane, consumed
    /// (and cleared) by the next span. Empty = plain single-row decode — the
    /// drafter-off path is this same code with `drafts` never set.
    drafts: Vec<u32>,
    /// Accept telemetry across the request's verify rounds.
    spec: SpecStats,
}

/// Drafts fed per verify span under EP8: 3 drafts + anchor = a bucket-4
/// verify step. A/B-measured on 8×H200 (2026-07-04,
/// docs/models/glm52/dspark-mtp.md): the bucket-4 step costs ~32 ms vs
/// bucket-8's ~46, and that cheaper round beats span 8's extra accepted tail
/// on EVERY tested prompt class. The drafter still proposes 7; the tail is
/// simply not fed. Mirrored TP may feed more drafts per round via the
/// per-engine `span_drafts` argument (`GLM52_DSPARK_DRAFTS`); the cap is
/// a per-round argument, not this constant.
pub(super) const GLM52_DSPARK_EP8_SPAN_DRAFTS: usize = 3;

/// Accept histogram over a request's verify rounds (spans that actually fed
/// drafts; bonus-only single-row spans don't count).
#[derive(Debug, Default)]
struct SpecStats {
    rounds: u64,
    accepted_sum: u64,
    hist: [u64; GLM52_DSPARK_DRAFTS + 1],
}

impl Glm52SlotState {
    /// `cached_tokens` = the prefix-cache hit: those leading prompt tokens'
    /// KV is already resident in pool pages, so feeding starts past them
    /// (the matcher always leaves >= 1 prompt token uncached).
    pub(super) fn new(
        prompt: Vec<u32>,
        max_tokens: usize,
        ignore_eos: bool,
        cached_tokens: usize,
    ) -> Self {
        debug_assert!(cached_tokens < prompt.len());
        Self {
            fed: cached_tokens,
            prompt,
            max_tokens,
            ignore_eos,
            completion: 0,
            sampling_offset: 0,
            last_token: 0,
            drafts: Vec::new(),
            spec: SpecStats::default(),
        }
    }

    /// Resume a native P/D handoff at its forwarded target anchor.
    ///
    /// The restored KV covers every token before the anchor. The anchor is
    /// already generated by P but has not been fed to the target model, so it
    /// becomes decode row zero without counting as a D-side completion. Its
    /// sampling step was consumed on P, so D's first sampled row starts at 1.
    #[cfg(test)]
    fn seed_native_pd_anchor(&mut self) {
        debug_assert_eq!(self.fed + 1, self.prompt.len());
        self.fed = self.prompt.len();
        self.sampling_offset = 1;
        self.last_token = *self
            .prompt
            .last()
            .expect("native P/D prompt contains an anchor");
    }

    /// Resume a v2 router handoff after replaying P's anchor to the client.
    ///
    /// The anchor remains the first verifier input but already consumes one
    /// output-token budget position. Removing it from the state's prompt
    /// keeps that input at the original prompt length.
    pub(super) fn seed_native_pd_replayed_anchor(&mut self) {
        debug_assert_eq!(self.fed + 1, self.prompt.len());
        self.last_token = self
            .prompt
            .pop()
            .expect("native P/D prompt contains an anchor");
        debug_assert_eq!(self.fed, self.prompt.len());
        self.completion = 1;
    }

    #[cfg(test)]
    pub(super) fn record_mtp_production_gate(&self, request_id: Option<&str>) {
        if !matches!(
            request_id,
            Some(MTP_PRODUCTION_GATE_REQUEST_ID | MTP_SLOT_REUSE_GATE_REQUEST_ID)
        ) {
            return;
        }
        let mut stats = MTP_PRODUCTION_STATS
            .lock()
            .expect("MTP production stats lock poisoned");
        match request_id {
            Some(MTP_PRODUCTION_GATE_REQUEST_ID) => {
                stats.rounds = self.spec.rounds;
                stats.accepted_drafts = self.spec.accepted_sum;
            }
            Some(MTP_SLOT_REUSE_GATE_REQUEST_ID) => {
                stats.reuse_rounds = self.spec.rounds;
                stats.reuse_accepted_drafts = self.spec.accepted_sum;
            }
            _ => {}
        }
    }

    pub(super) fn completion_tokens(&self) -> usize {
        self.completion
    }

    /// Whether the prompt is still being fed (decode starts once it isn't).
    pub(super) fn mid_prefill(&self) -> bool {
        self.fed < self.prompt.len()
    }

    /// Prompt tokens not yet fed; 0 once the request is decoding.
    pub(super) fn remaining_prompt(&self) -> usize {
        self.prompt.len() - self.fed
    }

    /// Rows this request can usefully fill in one step: the whole remaining
    /// prompt while mid-prefill (the planner caps it to the bucket); the
    /// verify span (anchor + proposed drafts) in decode, capped so a round
    /// can never commit past `max_tokens` — which also keeps every fed
    /// position under the model-length cap, since `validate_request` pins
    /// `prompt + max_tokens - 1 <= max_model_len`.
    pub(super) fn feed_want(&self) -> usize {
        if self.mid_prefill() {
            self.remaining_prompt()
        } else {
            (1 + self.drafts.len()).min(self.max_tokens - self.completion)
        }
    }

    /// The `offset`-th row of this step's span: consecutive prompt positions
    /// while mid-prefill; the anchor (offset 0) then the draft prefix in
    /// decode. The planner may grant fewer rows than `feed_want` — the span
    /// is then a prefix (anchor + first drafts), and the un-fed drafts are
    /// discarded by `advance_span`.
    pub(super) fn next_input_at(&self, offset: usize) -> Glm52StepInput {
        if self.mid_prefill() {
            debug_assert!(self.fed + offset < self.prompt.len());
            Glm52StepInput {
                token: self.prompt[self.fed + offset],
                position: self.fed + offset,
            }
        } else {
            debug_assert!(offset <= self.drafts.len());
            let token = if offset == 0 {
                self.last_token
            } else {
                self.drafts[offset - 1]
            };
            Glm52StepInput {
                token,
                position: self.prompt.len() + self.completion - 1 + offset,
            }
        }
    }

    /// The span rows whose outputs `advance_span` may commit, each with the
    /// request-local decode step its token lands at — the sampler overwrites
    /// exactly these rows. Empty while the span is still mid-prompt (outputs
    /// discarded); a prompt-completing span's last row yields the first
    /// generated token. A decode span is a verify: EVERY row (anchor + draft
    /// prefix) samples, and row `k`'s step is
    /// `sampling_offset + completion + k` — the same step a plain decode
    /// would sample that token at, which is what makes a
    /// seeded request's speculative stream replay its plain stream
    /// token-exactly ([`accept_prefix_match`] then just prefix-matches the sampled
    /// tokens against the drafts; the zero-draft plain row is the same rule).
    pub(super) fn sampling_rows(&self, span_rows: usize) -> Vec<(usize, u64)> {
        debug_assert!(span_rows > 0);
        if self.mid_prefill() {
            if span_rows == self.remaining_prompt() {
                vec![(
                    span_rows - 1,
                    (self.sampling_offset + self.completion) as u64,
                )]
            } else {
                Vec::new()
            }
        } else {
            debug_assert!(span_rows <= 1 + self.drafts.len());
            (0..span_rows)
                .map(|k| (k, (self.sampling_offset + self.completion + k) as u64))
                .collect()
        }
    }

    /// The next spec round's anchor once the request is decoding: the latest
    /// committed token and the position it will be fed at. `None` mid-prefill
    /// (no token to extend yet).
    pub(super) fn decode_anchor(&self) -> Option<(u32, usize)> {
        (!self.mid_prefill() && self.completion > 0)
            .then(|| (self.last_token, self.prompt.len() + self.completion - 1))
    }

    /// Whether a fresh draft proposal is worth requesting: decoding, and at
    /// least two tokens of budget left (a one-token tail can only ever commit
    /// the anchor's own output — a plain row).
    pub(super) fn wants_drafts(&self) -> bool {
        !self.mid_prefill() && self.completion + 1 < self.max_tokens
    }

    /// Native MTP always executes its fixed five-token proposal chain. Do not
    /// start that chain in a shorter request tail: the verifier would discard
    /// most of it, and the unused speculative KV positions could cross the
    /// request's launch-time model-length cap.
    pub(super) fn wants_full_draft(&self, draft_tokens: usize) -> bool {
        !self.mid_prefill() && self.max_tokens - self.completion >= draft_tokens
    }

    /// Install the draft lane's proposal for the next verify span, truncated
    /// to the topology's span cap ([`GLM52_DSPARK_EP8_SPAN_DRAFTS`] under EP8,
    /// all of [`GLM52_DSPARK_DRAFTS`] under the mirrored TP span shape).
    /// Returns the effective draft count after truncating to this engine's
    /// verify span (envelopes may carry the full fixed-width proposal array).
    pub(super) fn set_drafts(&mut self, mut drafts: Vec<u32>, span_drafts: usize) -> usize {
        drafts.truncate(span_drafts);
        self.drafts = drafts;
        self.drafts.len()
    }

    /// Fold one step's span of outputs in.
    ///
    /// Mid-prompt rows' outputs are discarded; the row that fed the LAST
    /// prompt token yields the first generated token. In decode the span is
    /// a verify: `outputs[k]` is the target's committed token after span row
    /// `k` (anchor + fed draft prefix) — the fused argmax for greedy
    /// requests, the sampled token for non-greedy ones — and
    /// [`accept_prefix_match`] commits the agreed prefix plus one model token.
    /// With sampled outputs that prefix-match IS lossless speculative
    /// sampling for a deterministic draft: every committed token is a sample
    /// from the target distribution, acceptance only decides how many ride
    /// one step. The plain single-row step is the zero-draft special case of
    /// the same rule.
    pub(super) fn advance_span(
        &mut self,
        outputs: &[u32],
        eos_token_ids: &[u32],
    ) -> Glm52StepOutcome {
        debug_assert!(!outputs.is_empty());
        let (committed, context_rows) = if self.mid_prefill() {
            debug_assert!(outputs.len() <= self.remaining_prompt());
            self.fed += outputs.len();
            if self.mid_prefill() {
                return Glm52StepOutcome::Prefilling;
            }
            // Boundary: every span row is committed prompt context, and the
            // last row's output is the first generated token.
            let output = *outputs.last().expect("span outputs are non-empty");
            (vec![output], outputs.len())
        } else {
            let drafts_fed = outputs.len() - 1;
            debug_assert!(drafts_fed <= self.drafts.len());
            let committed = accept_prefix_match(&self.drafts[..drafts_fed], outputs);
            if drafts_fed > 0 {
                let accepted_drafts = committed.len() - 1;
                self.spec.record(accepted_drafts);
            }
            let context_rows = committed.len();
            (committed, context_rows)
        };
        self.drafts.clear();

        let mut committed = committed;
        let mut emit = 0usize;
        let mut finish = None;
        for &token in &committed {
            self.completion += 1;
            if !self.ignore_eos && eos_token_ids.contains(&token) {
                finish = Some(FinishReason::Stop);
                break;
            }
            emit += 1;
            self.last_token = token;
            if self.completion >= self.max_tokens {
                finish = Some(FinishReason::Length);
                break;
            }
        }
        // Truncate to the consumed run (a suppressed EOS is consumed but not
        // emitted) so the caller's KV bookkeeping advances by exactly the
        // tokens this state accounted for.
        let consumed = emit + usize::from(matches!(finish, Some(FinishReason::Stop)));
        committed.truncate(consumed);
        Glm52StepOutcome::Commit {
            committed,
            emit,
            finish,
            context_rows,
        }
    }

    /// Log the request's accept telemetry when it leaves its slot (only when
    /// it ran verify rounds — plain-decode requests stay silent).
    pub(super) fn log_spec_stats(&self, rank: usize, slot: usize) {
        let stats = &self.spec;
        if stats.rounds == 0 {
            return;
        }
        let mean_accepted = stats.accepted_sum as f64 / stats.rounds as f64;
        log::info!(
            "GLM5.2 speculative: rank={rank} slot={slot} rounds={} mean_accepted_drafts={mean_accepted:.3} \
             mean_accepted_incl_bonus={:.3} hist={:?}",
            stats.rounds,
            mean_accepted + 1.0,
            stats.hist,
        );
    }
}

impl SpecStats {
    fn record(&mut self, accepted_drafts: usize) {
        self.rounds += 1;
        self.accepted_sum += accepted_drafts as u64;
        self.hist[accepted_drafts.min(GLM52_DSPARK_DRAFTS)] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mtp::GLM52_MTP_DRAFTS;
    use crate::scheduler::testkit::EOS;
    use crate::scheduler::testkit::commit;
    use crate::scheduler::testkit::state;

    #[test]
    fn prefill_rides_decode_then_emits() {
        let mut state = state(vec![10, 11, 12], 4, false);

        assert_eq!(state.feed_want(), 3);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 10,
                position: 0
            }
        );
        assert_eq!(state.advance_span(&[99], EOS), Glm52StepOutcome::Prefilling);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 11,
                position: 1
            }
        );
        assert_eq!(state.advance_span(&[99], EOS), Glm52StepOutcome::Prefilling);

        // The last prompt token's step yields the first generated token.
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 12,
                position: 2
            }
        );
        assert_eq!(state.advance_span(&[42], EOS), commit(&[42], 1, None, 1));
        assert_eq!(state.completion_tokens(), 1);

        // Decode continues from the emitted token at the next position.
        assert_eq!(state.feed_want(), 1);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 42,
                position: 3
            }
        );
    }

    #[test]
    fn one_token_contract_finishes_on_the_prompt_tail() {
        let mut state = state(vec![10, 11, 12], 1, false);
        assert_eq!(
            state.advance_span(&[90, 91, 42], EOS),
            commit(&[42], 1, Some(FinishReason::Length), 3)
        );
        assert_eq!(state.completion_tokens(), 1);
        assert!(!state.mid_prefill());
    }

    #[test]
    fn prompt_span_feeds_consecutive_positions_and_keeps_only_the_last_output() {
        let mut state = state(vec![10, 11, 12, 13], 4, false);

        // One span covers three prompt tokens; mid-prompt outputs discarded.
        assert_eq!(state.feed_want(), 4);
        assert_eq!(
            (0..3).map(|i| state.next_input_at(i)).collect::<Vec<_>>(),
            vec![
                Glm52StepInput {
                    token: 10,
                    position: 0
                },
                Glm52StepInput {
                    token: 11,
                    position: 1
                },
                Glm52StepInput {
                    token: 12,
                    position: 2
                },
            ]
        );
        assert_eq!(
            state.advance_span(&[99, 98, 97], EOS),
            Glm52StepOutcome::Prefilling
        );

        // The next span finishes the prompt; its last output is the first
        // generated token.
        assert_eq!(state.feed_want(), 1);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 13,
                position: 3
            }
        );
        assert_eq!(state.advance_span(&[42], EOS), commit(&[42], 1, None, 1));
        assert_eq!(state.completion_tokens(), 1);
    }

    #[test]
    fn whole_prompt_in_one_span_emits_from_the_boundary_row() {
        let mut state = state(vec![10, 11, 12], 4, false);
        // All three span rows are committed prompt context.
        assert_eq!(
            state.advance_span(&[99, 98, 42], EOS),
            commit(&[42], 1, None, 3)
        );
        assert_eq!(state.completion_tokens(), 1);
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 42,
                position: 3
            }
        );
    }

    #[test]
    fn eos_is_suppressed_and_counts_toward_completion() {
        let mut state = state(vec![10], 4, false);
        assert_eq!(
            state.advance_span(&[7], EOS),
            commit(&[7], 0, Some(FinishReason::Stop), 1)
        );
        assert_eq!(state.completion_tokens(), 1);
    }

    #[test]
    fn ignore_eos_decodes_through_the_stop_token() {
        let mut state = state(vec![10], 4, true);
        assert_eq!(state.advance_span(&[7], EOS), commit(&[7], 1, None, 1));
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 7,
                position: 1
            }
        );
    }

    #[test]
    fn length_cap_emits_the_final_token() {
        let mut state = state(vec![10], 2, false);
        assert_eq!(state.advance_span(&[42], EOS), commit(&[42], 1, None, 1));
        assert_eq!(
            state.advance_span(&[43], EOS),
            commit(&[43], 1, Some(FinishReason::Length), 1)
        );
        assert_eq!(state.completion_tokens(), 2);
    }

    #[test]
    fn eos_outranks_the_length_cap() {
        let mut state = state(vec![10], 1, false);
        assert_eq!(
            state.advance_span(&[7], EOS),
            commit(&[7], 0, Some(FinishReason::Stop), 1)
        );
    }

    #[test]
    fn max_tokens_one_emits_then_finishes() {
        let mut state = state(vec![10, 11], 1, false);
        assert_eq!(state.advance_span(&[99], EOS), Glm52StepOutcome::Prefilling);
        assert_eq!(
            state.advance_span(&[42], EOS),
            commit(&[42], 1, Some(FinishReason::Length), 1)
        );
    }

    #[test]
    fn verify_span_commits_accepted_prefix_plus_correction() {
        let mut state = state(vec![10], 32, false);
        // Boundary emits the anchor t0 = 20.
        assert_eq!(state.advance_span(&[20], EOS), commit(&[20], 1, None, 1));
        state.set_drafts(
            vec![21, 22, 99, 98, 97, 96, 95],
            GLM52_DSPARK_EP8_SPAN_DRAFTS,
        );

        // The proposal is truncated to GLM52_DSPARK_EP8_SPAN_DRAFTS: a 4-row
        // verify span (anchor + 3 drafts) at consecutive positions.
        assert_eq!(state.feed_want(), 4);
        assert_eq!(
            (0..3).map(|i| state.next_input_at(i)).collect::<Vec<_>>(),
            vec![
                Glm52StepInput {
                    token: 20,
                    position: 1
                },
                Glm52StepInput {
                    token: 21,
                    position: 2
                },
                Glm52StepInput {
                    token: 22,
                    position: 3
                },
            ]
        );

        // Target agrees with drafts 21, 22, diverges at the third (30 != 99):
        // commit the accepted prefix + the correction, context = anchor + 2
        // accepted rows.
        let outputs = [21, 22, 30, 0];
        assert_eq!(
            state.advance_span(&outputs, EOS),
            commit(&[21, 22, 30], 3, None, 3)
        );
        assert_eq!(state.completion_tokens(), 4);
        // The correction is the next anchor; drafts were consumed.
        assert_eq!(state.decode_anchor(), Some((30, 4)));
        assert_eq!(state.feed_want(), 1);
    }

    #[test]
    fn native_pd_starts_by_verifying_the_forwarded_anchor() {
        let mut state = Glm52SlotState::new(vec![10, 11, 20], 8, false, 2);
        state.seed_native_pd_anchor();
        state.set_drafts(vec![21, 22, 99, 98, 97], GLM52_MTP_DRAFTS);

        assert!(!state.mid_prefill());
        assert_eq!(state.completion_tokens(), 0);
        assert_eq!(state.feed_want(), 6);
        assert_eq!(
            (0..3).map(|i| state.next_input_at(i)).collect::<Vec<_>>(),
            vec![
                Glm52StepInput {
                    token: 20,
                    position: 2
                },
                Glm52StepInput {
                    token: 21,
                    position: 3
                },
                Glm52StepInput {
                    token: 22,
                    position: 4
                },
            ]
        );

        assert_eq!(
            state.advance_span(&[21, 22, 30, 0, 0, 0], EOS),
            commit(&[21, 22, 30], 3, None, 3)
        );
        assert_eq!(state.decode_anchor(), Some((30, 5)));
    }

    #[test]
    fn manual_native_pd_advances_rng_without_spending_client_budget() {
        let mut state = Glm52SlotState::new(vec![10, 11, 12, 42], 4, true, 3);
        state.seed_native_pd_anchor();

        assert_eq!(
            state.completion_tokens(),
            0,
            "manual D output budget excludes the anchor already returned by P"
        );
        assert_eq!(
            state.sampling_rows(1),
            vec![(0, 1)],
            "P sampled the anchor at step 0, so D must continue at step 1"
        );

        assert_eq!(state.advance_span(&[43], EOS), commit(&[43], 1, None, 1));
        assert_eq!(state.sampling_rows(1), vec![(0, 2)]);
    }

    #[test]
    fn native_pd_replayed_anchor_counts_against_the_router_budget() {
        let mut state = Glm52SlotState::new(vec![10, 11, 20], 8, false, 2);
        state.seed_native_pd_replayed_anchor();
        state.set_drafts(vec![21, 22, 99, 98, 97], GLM52_MTP_DRAFTS);

        assert_eq!(state.completion_tokens(), 1);
        assert_eq!(
            state.sampling_rows(1),
            vec![(0, 1)],
            "router completion already accounts for P's sampling step"
        );
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 20,
                position: 2
            }
        );
        assert_eq!(
            state.advance_span(&[21, 22, 30, 0, 0, 0], EOS),
            commit(&[21, 22, 30], 3, None, 3)
        );
        assert_eq!(state.completion_tokens(), 4);
        assert_eq!(state.decode_anchor(), Some((30, 5)));
    }

    #[test]
    fn verify_span_truncated_by_the_planner_accepts_only_fed_drafts() {
        let mut state = state(vec![10], 32, false);
        assert_eq!(state.advance_span(&[20], EOS), commit(&[20], 1, None, 1));
        state.set_drafts(
            vec![21, 22, 23, 24, 25, 26, 27],
            GLM52_DSPARK_EP8_SPAN_DRAFTS,
        );

        // The planner granted only 3 of the 4 wanted rows: the span is the
        // anchor + first 2 drafts, and acceptance ranges over those 2 only.
        assert_eq!(state.feed_want(), 4);
        let outputs = [21, 22, 23];
        assert_eq!(
            state.advance_span(&outputs, EOS),
            commit(&[21, 22, 23], 3, None, 3)
        );
        assert_eq!(state.completion_tokens(), 4);
        assert_eq!(state.decode_anchor(), Some((23, 4)));
    }

    #[test]
    fn eos_inside_the_committed_run_truncates_and_finishes() {
        let mut state = state(vec![10], 32, false);
        assert_eq!(state.advance_span(&[20], EOS), commit(&[20], 1, None, 1));
        // Draft 2 is the EOS token (7): accepted, counted, suppressed; the
        // rest of the committed run is dropped.
        state.set_drafts(
            vec![21, 7, 23, 24, 25, 26, 27],
            GLM52_DSPARK_EP8_SPAN_DRAFTS,
        );
        let outputs = [21, 7, 23, 24];
        assert_eq!(
            state.advance_span(&outputs, EOS),
            commit(&[21, 7], 1, Some(FinishReason::Stop), 4)
        );
        assert_eq!(state.completion_tokens(), 3);
    }

    #[test]
    fn length_cap_truncates_the_verify_want_and_the_committed_run() {
        let mut state = state(vec![10], 4, false);
        assert_eq!(state.advance_span(&[20], EOS), commit(&[20], 1, None, 1));
        state.set_drafts(
            vec![21, 22, 23, 24, 25, 26, 27],
            GLM52_DSPARK_EP8_SPAN_DRAFTS,
        );
        // remaining = 3 -> the span may commit at most 3 more tokens.
        assert_eq!(state.feed_want(), 3);
        let outputs = [21, 22, 23];
        assert_eq!(
            state.advance_span(&outputs, EOS),
            commit(&[21, 22, 23], 3, Some(FinishReason::Length), 3)
        );
        assert_eq!(state.completion_tokens(), 4);
    }

    #[test]
    fn wants_drafts_only_with_two_tokens_of_budget() {
        let mut state = state(vec![10], 3, false);
        assert!(!state.wants_drafts(), "mid-prefill never wants drafts");
        assert_eq!(state.advance_span(&[20], EOS), commit(&[20], 1, None, 1));
        assert!(state.wants_drafts());
        assert_eq!(state.advance_span(&[21], EOS), commit(&[21], 1, None, 1));
        assert!(!state.wants_drafts(), "one-token tail is a plain row");
    }

    #[test]
    fn fixed_mtp_chain_stops_before_a_short_tail() {
        let mut state = state(vec![10], 6, false);
        assert!(!state.wants_full_draft(5), "mid-prefill never drafts");
        assert_eq!(state.advance_span(&[20], EOS), commit(&[20], 1, None, 1));
        assert!(state.wants_full_draft(5));
        assert_eq!(state.advance_span(&[21], EOS), commit(&[21], 1, None, 1));
        assert!(
            !state.wants_full_draft(5),
            "four-token tail must not launch a five-token MTP chain"
        );
    }

    #[test]
    fn sampling_rows_are_the_committable_rows_of_the_span() {
        let mut state = state(vec![10, 11, 12], 8, false);
        // Mid-prompt span: outputs discarded, nothing to sample.
        assert_eq!(state.sampling_rows(2), vec![]);
        // Prompt-completing span: the last row's output is the first
        // generated token, sampled at step 0.
        assert_eq!(state.sampling_rows(3), vec![(2, 0)]);
        assert_eq!(
            state.advance_span(&[99, 98, 42], EOS),
            commit(&[42], 1, None, 3)
        );
        // Plain decode: the single anchor row, at the request-local step.
        assert_eq!(state.sampling_rows(1), vec![(0, 1)]);
        // Verify span: every row samples, row k at step completion + k —
        // the steps a plain decode would sample those tokens at.
        state.set_drafts(vec![50, 51, 52], GLM52_DSPARK_EP8_SPAN_DRAFTS);
        assert_eq!(state.sampling_rows(4), vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
        // The planner may grant fewer rows: the prefix samples.
        assert_eq!(state.sampling_rows(2), vec![(0, 1), (1, 2)]);
    }

    #[test]
    fn sampling_steps_track_completion_across_rounds() {
        // The seed contract's invariant: a token's sample step equals the
        // completion index it lands at, regardless of how many rows rode
        // each round — a partial accept must not shift the next round's
        // steps (an off-by-one here silently breaks seeded replayability).
        let mut state = state(vec![10, 11, 12], 16, false);
        assert_eq!(
            state.advance_span(&[99, 98, 42], EOS),
            commit(&[42], 1, None, 3)
        );
        state.set_drafts(vec![50, 51, 52], GLM52_DSPARK_EP8_SPAN_DRAFTS);
        assert_eq!(state.sampling_rows(4), vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
        // Partial accept: sampled rows match d1, d2, reject d3 → commit
        // [d1, d2, correction] = 3 tokens.
        assert_eq!(
            state.advance_span(&[50, 51, 77, 88], EOS),
            commit(&[50, 51, 77], 3, None, 3)
        );
        // Next round resumes at completion = 4: plain decode would sample
        // its 5th token (index 4) at step 4 — so must the verify span.
        state.set_drafts(vec![60, 61, 62], GLM52_DSPARK_EP8_SPAN_DRAFTS);
        assert_eq!(state.sampling_rows(4), vec![(0, 4), (1, 5), (2, 6), (3, 7)]);
    }

    #[test]
    fn decode_anchor_is_the_latest_token_at_its_feed_position() {
        let mut state = state(vec![10, 11], 4, false);
        assert_eq!(state.decode_anchor(), None);
        assert_eq!(
            state.advance_span(&[99, 42], EOS),
            commit(&[42], 1, None, 2)
        );
        // The anchor is what the next decode row would feed.
        assert_eq!(state.decode_anchor(), Some((42, 2)));
        assert_eq!(
            state.next_input_at(0),
            Glm52StepInput {
                token: 42,
                position: 2
            }
        );
        assert_eq!(state.advance_span(&[43], EOS), commit(&[43], 1, None, 1));
        assert_eq!(state.decode_anchor(), Some((43, 3)));
    }

    #[test]
    fn cached_prefix_starts_feeding_at_the_suffix() {
        // 3 blocks of prompt with the first 2 cache-hit: feeding starts at
        // position 128 and only the suffix is ever fed.
        let prompt: Vec<u32> = (0..192).collect();
        let s = Glm52SlotState::new(prompt, 8, false, 128);
        assert_eq!(s.feed_want(), 64);
        assert_eq!(s.next_input_at(0).position, 128);
        assert_eq!(s.next_input_at(0).token, 128);
    }
}
