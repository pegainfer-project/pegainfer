use std::collections::VecDeque;
use std::fmt::Display;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;

use crate::profile::PrefillPolicy;
use crate::profile::SchedulerProfile;
use crate::profile::StepShape;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestPhase {
    Waiting,
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkerRequest<I> {
    pub id: I,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RequestRejection {
    ModelLengthExceeded {
        total_tokens: u64,
        max_model_len: u32,
    },
    WholePrefillExceedsStepBudget {
        prompt_tokens: u32,
        max_num_batched_tokens: u32,
    },
}

impl Display for RequestRejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelLengthExceeded {
                total_tokens,
                max_model_len,
            } => write!(
                formatter,
                "request total tokens {total_tokens} exceed max_model_len {max_model_len}"
            ),
            Self::WholePrefillExceedsStepBudget {
                prompt_tokens,
                max_num_batched_tokens,
            } => write!(
                formatter,
                "whole prefill has {prompt_tokens} tokens but max_num_batched_tokens is {max_num_batched_tokens}"
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubmissionResult {
    Queued,
    Finished,
    Rejected(RequestRejection),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CancelResult {
    Cancelled,
    Deferred,
    AlreadyRequested,
    NotFound,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RequestSnapshot {
    pub phase: RequestPhase,
    pub remaining_prefill_tokens: u32,
    pub generated_tokens: u32,
    /// Tokens whose KV has been computed before the next decode query.
    pub context_tokens: u32,
    pub output_tokens: u32,
    pub cancel_requested: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StepId(u64);

impl StepId {
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillWork<I> {
    pub request_id: I,
    pub tokens: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodeWork<I> {
    pub request_id: I,
    /// Existing KV length before this decode query is computed.
    pub context_tokens: u32,
}

#[derive(Debug, Eq, PartialEq)]
pub struct StepPlan<I> {
    id: StepId,
    shape: StepShape,
    admitted: Vec<I>,
    prefill: Vec<PrefillWork<I>>,
    decode: Vec<DecodeWork<I>>,
}

impl<I> StepPlan<I> {
    #[must_use]
    pub fn id(&self) -> StepId {
        self.id
    }

    #[must_use]
    pub fn shape(&self) -> StepShape {
        self.shape
    }

    #[must_use]
    pub fn admitted(&self) -> &[I] {
        &self.admitted
    }

    #[must_use]
    pub fn prefill(&self) -> &[PrefillWork<I>] {
        &self.prefill
    }

    #[must_use]
    pub fn decode(&self) -> &[DecodeWork<I>] {
        &self.decode
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrefillProgress<I> {
    pub request_id: I,
    pub processed_tokens: u32,
    pub remaining_tokens: u32,
    pub context_tokens: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeneratedToken<I> {
    pub request_id: I,
    pub token_index: u32,
}

#[derive(Debug, Eq, PartialEq)]
pub struct StepOutcome<I> {
    pub prefill: Vec<PrefillProgress<I>>,
    pub generated: Vec<GeneratedToken<I>>,
    pub finished: Vec<I>,
    pub cancelled: Vec<I>,
}

struct RequestState<I> {
    id: I,
    phase: RequestPhase,
    remaining_prefill_tokens: u32,
    generated_tokens: u32,
    context_tokens: u32,
    output_tokens: u32,
    cancel_requested: bool,
}

impl<I: Copy> RequestState<I> {
    fn new(request: WorkerRequest<I>) -> Self {
        Self {
            id: request.id,
            phase: RequestPhase::Waiting,
            remaining_prefill_tokens: request.prompt_tokens,
            generated_tokens: 0,
            context_tokens: 0,
            output_tokens: request.output_tokens,
            cancel_requested: false,
        }
    }

    fn snapshot(&self) -> RequestSnapshot {
        RequestSnapshot {
            phase: self.phase,
            remaining_prefill_tokens: self.remaining_prefill_tokens,
            generated_tokens: self.generated_tokens,
            context_tokens: self.context_tokens,
            output_tokens: self.output_tokens,
            cancel_requested: self.cancel_requested,
        }
    }
}

pub struct WorkerState<I> {
    scheduler: SchedulerProfile,
    waiting: VecDeque<RequestState<I>>,
    running: Vec<RequestState<I>>,
    in_flight: Option<StepPlan<I>>,
    next_step_id: u64,
}

impl<I> WorkerState<I>
where
    I: Copy + Eq,
{
    pub fn new(scheduler: SchedulerProfile) -> Result<Self> {
        scheduler.validate()?;
        Ok(Self {
            scheduler,
            waiting: VecDeque::new(),
            running: Vec::new(),
            in_flight: None,
            next_step_id: 0,
        })
    }

    #[must_use]
    pub fn scheduler(&self) -> &SchedulerProfile {
        &self.scheduler
    }

    pub(crate) fn preflight(
        &self,
        prompt_tokens: u32,
        output_tokens: u64,
    ) -> Option<RequestRejection> {
        let total_tokens = u64::from(prompt_tokens).saturating_add(output_tokens);
        if total_tokens > u64::from(self.scheduler.max_model_len) {
            return Some(RequestRejection::ModelLengthExceeded {
                total_tokens,
                max_model_len: self.scheduler.max_model_len,
            });
        }
        if output_tokens > 0
            && self.scheduler.prefill == PrefillPolicy::Whole
            && prompt_tokens > self.scheduler.max_num_batched_tokens
        {
            return Some(RequestRejection::WholePrefillExceedsStepBudget {
                prompt_tokens,
                max_num_batched_tokens: self.scheduler.max_num_batched_tokens,
            });
        }
        None
    }

    pub fn submit(&mut self, request: WorkerRequest<I>) -> Result<SubmissionResult> {
        ensure!(
            self.request(request.id).is_none(),
            "worker request id is already active"
        );
        if let Some(rejection) =
            self.preflight(request.prompt_tokens, u64::from(request.output_tokens))
        {
            return Ok(SubmissionResult::Rejected(rejection));
        }
        if request.output_tokens == 0 {
            return Ok(SubmissionResult::Finished);
        }
        self.waiting.push_back(RequestState::new(request));
        Ok(SubmissionResult::Queued)
    }

    #[must_use]
    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    #[must_use]
    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    #[must_use]
    pub fn is_idle(&self) -> bool {
        self.waiting.is_empty() && self.running.is_empty() && self.in_flight.is_none()
    }

    #[must_use]
    pub fn in_flight(&self) -> Option<&StepPlan<I>> {
        self.in_flight.as_ref()
    }

    #[must_use]
    pub fn request(&self, id: I) -> Option<RequestSnapshot> {
        self.running
            .iter()
            .chain(self.waiting.iter())
            .find(|request| request.id == id)
            .map(RequestState::snapshot)
    }

    pub fn cancel(&mut self, id: I) -> CancelResult {
        if let Some(index) = self.waiting.iter().position(|request| request.id == id) {
            self.waiting.remove(index);
            return CancelResult::Cancelled;
        }
        let Some(index) = self.running.iter().position(|request| request.id == id) else {
            return CancelResult::NotFound;
        };
        if self.running[index].cancel_requested {
            return CancelResult::AlreadyRequested;
        }
        if self.in_flight.is_some() {
            self.running[index].cancel_requested = true;
            CancelResult::Deferred
        } else {
            self.running.remove(index);
            CancelResult::Cancelled
        }
    }

    pub fn plan_step(&mut self) -> Result<Option<&StepPlan<I>>> {
        ensure!(
            self.in_flight.is_none(),
            "cannot plan a worker step while another step is in flight"
        );
        if self.running.is_empty() && self.waiting.is_empty() {
            return Ok(None);
        }

        let step_id = StepId(self.next_step_id);
        let next_step_id = self
            .next_step_id
            .checked_add(1)
            .context("worker step id overflow")?;
        let mut remaining_budget = self.scheduler.max_num_batched_tokens;
        let mut decode = Vec::new();
        let mut sum_decode_ctx_tokens = 0_u64;

        // vLLM V1 keeps every running decode scheduled before spending the
        // remaining token budget on prefill work.
        for request in &self.running {
            if request.phase != RequestPhase::Decode || request.cancel_requested {
                continue;
            }
            ensure!(remaining_budget > 0, "running decode exceeded token budget");
            remaining_budget -= 1;
            sum_decode_ctx_tokens = sum_decode_ctx_tokens
                .checked_add(u64::from(request.context_tokens))
                .context("decode context sum overflow")?;
            decode.push(DecodeWork {
                request_id: request.id,
                context_tokens: request.context_tokens,
            });
        }

        let mut prefill = Vec::new();
        let mut prefill_tokens_in_step = 0_u32;
        let mut prefill_blocked = false;
        for request in &self.running {
            if request.phase != RequestPhase::Prefill || request.cancel_requested {
                continue;
            }
            let tokens = scheduled_prefill_tokens(
                self.scheduler.prefill,
                request.remaining_prefill_tokens,
                remaining_budget,
            );
            if tokens == 0 {
                prefill_blocked = true;
                break;
            }
            remaining_budget -= tokens;
            prefill_tokens_in_step += tokens;
            prefill.push(PrefillWork {
                request_id: request.id,
                tokens,
            });
        }

        let mut admitted = Vec::new();
        while !prefill_blocked
            && remaining_budget > 0
            && self.running.len() < self.scheduler.max_num_seqs as usize
            && !self.waiting.is_empty()
        {
            let request = self
                .waiting
                .front()
                .expect("admission loop checked that a waiting request exists");
            let (phase, prefill_tokens) = if request.remaining_prefill_tokens == 0 {
                (RequestPhase::Decode, 0)
            } else {
                let tokens = scheduled_prefill_tokens(
                    self.scheduler.prefill,
                    request.remaining_prefill_tokens,
                    remaining_budget,
                );
                if tokens == 0 {
                    break;
                }
                (RequestPhase::Prefill, tokens)
            };

            let mut request = self
                .waiting
                .pop_front()
                .context("waiting request disappeared during admission")?;
            request.phase = phase;
            admitted.push(request.id);
            if phase == RequestPhase::Decode {
                remaining_budget -= 1;
                decode.push(DecodeWork {
                    request_id: request.id,
                    context_tokens: 0,
                });
            } else {
                remaining_budget -= prefill_tokens;
                prefill_tokens_in_step += prefill_tokens;
                prefill.push(PrefillWork {
                    request_id: request.id,
                    tokens: prefill_tokens,
                });
            }
            self.running.push(request);
        }

        let decode_reqs = u32::try_from(decode.len()).context("decode request count overflow")?;
        let shape = StepShape {
            decode_reqs,
            sum_decode_ctx_tokens,
            prefill_tokens_in_step,
        };
        ensure!(
            shape.decode_reqs > 0 || shape.prefill_tokens_in_step > 0,
            "worker with queued or running requests produced an empty step"
        );
        ensure!(
            shape.decode_reqs <= self.scheduler.max_num_seqs,
            "planned decode requests exceed sequence capacity"
        );
        ensure!(
            shape.decode_reqs + shape.prefill_tokens_in_step
                <= self.scheduler.max_num_batched_tokens,
            "planned step exceeds token capacity"
        );

        self.next_step_id = next_step_id;
        self.in_flight = Some(StepPlan {
            id: step_id,
            shape,
            admitted,
            prefill,
            decode,
        });
        Ok(self.in_flight.as_ref())
    }

    pub fn complete_step(&mut self, step_id: StepId) -> Result<StepOutcome<I>> {
        let current_id = self
            .in_flight
            .as_ref()
            .map(StepPlan::id)
            .context("cannot complete a worker step when no step is in flight")?;
        ensure!(
            current_id == step_id,
            "worker step id {} does not match in-flight step {}",
            step_id.get(),
            current_id.get()
        );
        let plan = self.in_flight.take().expect("in-flight step was checked");
        let mut outcome = StepOutcome {
            prefill: Vec::with_capacity(plan.prefill.len()),
            generated: Vec::with_capacity(plan.prefill.len() + plan.decode.len()),
            finished: Vec::new(),
            cancelled: Vec::new(),
        };

        for work in plan.prefill {
            let request = self
                .running
                .iter_mut()
                .find(|request| request.id == work.request_id)
                .context("planned prefill request is no longer running")?;
            if request.cancel_requested {
                continue;
            }
            ensure!(
                request.phase == RequestPhase::Prefill
                    && work.tokens > 0
                    && work.tokens <= request.remaining_prefill_tokens,
                "invalid prefill work in completed step"
            );
            request.remaining_prefill_tokens -= work.tokens;
            request.context_tokens = request
                .context_tokens
                .checked_add(work.tokens)
                .context("prefill context length overflow")?;
            outcome.prefill.push(PrefillProgress {
                request_id: request.id,
                processed_tokens: work.tokens,
                remaining_tokens: request.remaining_prefill_tokens,
                context_tokens: request.context_tokens,
            });
            if request.remaining_prefill_tokens == 0 {
                request.phase = RequestPhase::Decode;
                // The final prefill logits produce the first output token; a
                // separate decode step here would add a spurious TTFT step.
                request.generated_tokens = request
                    .generated_tokens
                    .checked_add(1)
                    .context("generated token count overflow")?;
                outcome.generated.push(GeneratedToken {
                    request_id: request.id,
                    token_index: request.generated_tokens,
                });
                if request.generated_tokens == request.output_tokens {
                    outcome.finished.push(request.id);
                }
            }
        }

        for work in plan.decode {
            let request = self
                .running
                .iter_mut()
                .find(|request| request.id == work.request_id)
                .context("planned decode request is no longer running")?;
            if request.cancel_requested {
                continue;
            }
            ensure!(
                request.phase == RequestPhase::Decode
                    && request.context_tokens == work.context_tokens
                    && request.generated_tokens < request.output_tokens,
                "invalid decode work in completed step"
            );
            request.context_tokens = request
                .context_tokens
                .checked_add(1)
                .context("decode context length overflow")?;
            request.generated_tokens = request
                .generated_tokens
                .checked_add(1)
                .context("generated token count overflow")?;
            outcome.generated.push(GeneratedToken {
                request_id: request.id,
                token_index: request.generated_tokens,
            });
            if request.generated_tokens == request.output_tokens {
                outcome.finished.push(request.id);
            }
        }

        outcome.cancelled.extend(
            self.running
                .iter()
                .filter(|request| request.cancel_requested)
                .map(|request| request.id),
        );
        self.running
            .retain(|request| !request.cancel_requested && !outcome.finished.contains(&request.id));
        Ok(outcome)
    }
}

fn scheduled_prefill_tokens(
    policy: PrefillPolicy,
    remaining_tokens: u32,
    remaining_budget: u32,
) -> u32 {
    match policy {
        PrefillPolicy::Whole => {
            if remaining_tokens <= remaining_budget {
                remaining_tokens
            } else {
                0
            }
        }
        PrefillPolicy::Chunked { max_chunk_tokens } => {
            remaining_tokens.min(max_chunk_tokens).min(remaining_budget)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::SchedulerPolicy;

    fn scheduler(
        max_num_seqs: u32,
        max_num_batched_tokens: u32,
        max_model_len: u32,
        prefill: PrefillPolicy,
    ) -> SchedulerProfile {
        SchedulerProfile {
            policy: SchedulerPolicy::VllmV1,
            max_num_seqs,
            max_num_batched_tokens,
            max_model_len,
            prefill,
        }
    }

    fn request(id: u32, prompt_tokens: u32, output_tokens: u32) -> WorkerRequest<u32> {
        WorkerRequest {
            id,
            prompt_tokens,
            output_tokens,
        }
    }

    #[test]
    fn impossible_and_duplicate_requests_are_rejected() {
        let mut whole = WorkerState::new(scheduler(2, 4, 8, PrefillPolicy::Whole)).unwrap();
        assert_eq!(
            whole.submit(request(1, 7, 2)).unwrap(),
            SubmissionResult::Rejected(RequestRejection::ModelLengthExceeded {
                total_tokens: 9,
                max_model_len: 8,
            })
        );
        assert_eq!(
            whole.submit(request(2, 5, 1)).unwrap(),
            SubmissionResult::Rejected(RequestRejection::WholePrefillExceedsStepBudget {
                prompt_tokens: 5,
                max_num_batched_tokens: 4,
            })
        );
        assert_eq!(
            whole.submit(request(3, 1, 1)).unwrap(),
            SubmissionResult::Queued
        );
        assert!(whole.submit(request(3, 1, 1)).is_err());
        assert_eq!(
            whole.submit(request(4, 4, 0)).unwrap(),
            SubmissionResult::Finished
        );
    }

    #[test]
    fn context_and_generated_progress_end_in_one_terminal_outcome() {
        let mut worker = WorkerState::new(scheduler(1, 4, 16, PrefillPolicy::Whole)).unwrap();
        worker.submit(request(1, 3, 3)).unwrap();

        let prefill_id = worker.plan_step().unwrap().unwrap().id();
        let prefill = worker.complete_step(prefill_id).unwrap();
        assert_eq!(
            prefill.generated,
            [GeneratedToken {
                request_id: 1,
                token_index: 1
            }]
        );
        assert_eq!(worker.request(1).unwrap().context_tokens, 3);

        let decode_one = worker.plan_step().unwrap().unwrap();
        assert_eq!(decode_one.shape().sum_decode_ctx_tokens, 3);
        let decode_one_id = decode_one.id();
        let outcome = worker.complete_step(decode_one_id).unwrap();
        assert_eq!(outcome.generated[0].token_index, 2);
        assert_eq!(worker.request(1).unwrap().context_tokens, 4);

        let decode_two = worker.plan_step().unwrap().unwrap();
        assert_eq!(decode_two.shape().sum_decode_ctx_tokens, 4);
        let decode_two_id = decode_two.id();
        let terminal = worker.complete_step(decode_two_id).unwrap();
        assert_eq!(terminal.finished, [1]);
        assert!(worker.request(1).is_none());
        assert!(worker.plan_step().unwrap().is_none());
    }

    #[test]
    fn cancellation_after_nonterminal_step_completion_removes_running_request() {
        let mut worker = WorkerState::new(scheduler(1, 4, 16, PrefillPolicy::Whole)).unwrap();
        worker.submit(request(1, 0, 3)).unwrap();

        let step_id = worker.plan_step().unwrap().unwrap().id();
        let outcome = worker.complete_step(step_id).unwrap();
        assert_eq!(
            outcome.generated,
            [GeneratedToken {
                request_id: 1,
                token_index: 1
            }]
        );
        assert!(outcome.finished.is_empty());
        assert!(worker.request(1).is_some());

        assert_eq!(worker.cancel(1), CancelResult::Cancelled);
        assert!(worker.request(1).is_none());
        assert!(worker.plan_step().unwrap().is_none());
    }

    #[test]
    fn generated_plans_never_exceed_sequence_or_token_capacity() {
        for max_num_seqs in 1..=4 {
            for max_num_batched_tokens in max_num_seqs..=6 {
                for max_chunk_tokens in 1..=max_num_batched_tokens {
                    let mut worker = WorkerState::new(scheduler(
                        max_num_seqs,
                        max_num_batched_tokens,
                        32,
                        PrefillPolicy::Chunked { max_chunk_tokens },
                    ))
                    .unwrap();
                    let mut terminal_requests = 0;
                    for id in 0..12 {
                        let prompt_tokens = id % 8;
                        let output_tokens = id % 3 + 1;
                        assert_eq!(
                            worker
                                .submit(request(id, prompt_tokens, output_tokens))
                                .unwrap(),
                            SubmissionResult::Queued
                        );
                    }

                    for _ in 0..256 {
                        let Some(plan) = worker.plan_step().unwrap() else {
                            break;
                        };
                        let shape = plan.shape();
                        assert!(shape.decode_reqs <= max_num_seqs);
                        assert!(
                            shape.decode_reqs + shape.prefill_tokens_in_step
                                <= max_num_batched_tokens
                        );
                        assert!(
                            plan.prefill()
                                .iter()
                                .all(|work| work.tokens <= max_chunk_tokens)
                        );
                        assert_eq!(
                            plan.decode()
                                .iter()
                                .map(|work| u64::from(work.context_tokens))
                                .sum::<u64>(),
                            shape.sum_decode_ctx_tokens
                        );
                        let step_id = plan.id();
                        assert!(worker.running_len() <= max_num_seqs as usize);
                        terminal_requests += worker.complete_step(step_id).unwrap().finished.len();
                    }
                    assert!(worker.is_idle());
                    assert_eq!(terminal_requests, 12);
                }
            }
        }
    }

    #[test]
    fn chunk_cap_changes_same_step_admission_without_changing_outputs() {
        #[derive(Debug)]
        struct ReplayStep {
            id: u64,
            shape: StepShape,
            admitted: Vec<u32>,
            prefill: Vec<(u32, u32)>,
            running: usize,
            waiting: usize,
            finished: Vec<u32>,
            generated: Vec<(u32, u32)>,
        }

        struct Replay {
            steps: Vec<ReplayStep>,
            finished: Vec<u32>,
            generated: Vec<(u32, u32)>,
        }

        fn record_step(worker: &mut WorkerState<u32>) -> Option<ReplayStep> {
            let (step_id, mut step) = {
                let plan = worker.plan_step().unwrap()?;
                (
                    plan.id(),
                    ReplayStep {
                        id: plan.id().get(),
                        shape: plan.shape(),
                        admitted: plan.admitted().to_vec(),
                        prefill: plan
                            .prefill()
                            .iter()
                            .map(|work| (work.request_id, work.tokens))
                            .collect(),
                        running: worker.running_len(),
                        waiting: worker.waiting_len(),
                        finished: Vec::new(),
                        generated: Vec::new(),
                    },
                )
            };
            let outcome = worker.complete_step(step_id).unwrap();
            step.finished = outcome.finished;
            step.generated = outcome
                .generated
                .into_iter()
                .map(|token| (token.request_id, token.token_index))
                .collect();
            Some(step)
        }

        fn replay(max_chunk_tokens: u32) -> Replay {
            let mut worker = WorkerState::new(scheduler(
                4,
                8,
                128,
                PrefillPolicy::Chunked { max_chunk_tokens },
            ))
            .unwrap();
            assert_eq!(
                worker.submit(request(1, 0, 3)).unwrap(),
                SubmissionResult::Queued
            );
            assert_eq!(
                worker.submit(request(2, 0, 3)).unwrap(),
                SubmissionResult::Queued
            );

            let mut steps = vec![record_step(&mut worker).unwrap()];
            assert_eq!(steps[0].shape.decode_reqs, 2);
            assert_eq!(steps[0].generated.len(), 2);

            assert_eq!(
                worker.submit(request(3, 100, 2)).unwrap(),
                SubmissionResult::Queued
            );
            assert_eq!(
                worker.submit(request(4, 1, 1)).unwrap(),
                SubmissionResult::Queued
            );

            for _ in 0..128 {
                let Some(step) = record_step(&mut worker) else {
                    break;
                };
                steps.push(step);
            }
            assert!(worker.is_idle());
            let mut finished = steps
                .iter()
                .flat_map(|step| step.finished.iter().copied())
                .collect::<Vec<_>>();
            finished.sort_unstable();
            let mut generated = steps
                .iter()
                .flat_map(|step| step.generated.iter().copied())
                .collect::<Vec<_>>();
            generated.sort_unstable();
            Replay {
                steps,
                finished,
                generated,
            }
        }

        let cap_six = replay(6);
        let cap_two = replay(2);

        assert_eq!(
            cap_six.steps[1].shape,
            StepShape {
                decode_reqs: 2,
                sum_decode_ctx_tokens: 2,
                prefill_tokens_in_step: 6,
            }
        );
        assert_eq!(cap_six.steps[1].admitted, [3]);
        assert_eq!(cap_six.steps[1].prefill, [(3, 6)]);
        assert_eq!((cap_six.steps[1].running, cap_six.steps[1].waiting), (3, 1));

        assert_eq!(
            cap_two.steps[1].shape,
            StepShape {
                decode_reqs: 2,
                sum_decode_ctx_tokens: 2,
                prefill_tokens_in_step: 3,
            }
        );
        assert_eq!(cap_two.steps[1].admitted, [3, 4]);
        assert_eq!(cap_two.steps[1].prefill, [(3, 2), (4, 1)]);
        assert_eq!((cap_two.steps[1].running, cap_two.steps[1].waiting), (4, 0));

        let cap_six_d_admission = cap_six
            .steps
            .iter()
            .find(|step| step.admitted.contains(&4))
            .map(|step| step.id);
        let cap_two_d_admission = cap_two
            .steps
            .iter()
            .find(|step| step.admitted.contains(&4))
            .map(|step| step.id);
        assert_eq!(cap_six_d_admission, Some(3));
        assert_eq!(cap_two_d_admission, Some(1));
        assert_eq!(
            cap_six
                .steps
                .iter()
                .find(|step| step.finished.contains(&3))
                .map(|step| step.id),
            Some(18)
        );
        assert_eq!(
            cap_two
                .steps
                .iter()
                .find(|step| step.finished.contains(&3))
                .map(|step| step.id),
            Some(51)
        );
        assert_eq!(cap_six.finished, [1, 2, 3, 4]);
        assert_eq!(cap_two.finished, [1, 2, 3, 4]);
        assert_eq!(
            cap_six.generated,
            [
                (1, 1),
                (1, 2),
                (1, 3),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 1),
                (3, 2),
                (4, 1),
            ]
        );
        assert_eq!(cap_two.generated, cap_six.generated);
        assert_eq!(cap_six.steps.len(), 19);
        assert_eq!(cap_two.steps.len(), 52);
    }
}
