//! Executor-side speculative-decode orchestration: the optimistic KV
//! transaction around a verify forward (schedule → forward → accept/commit or
//! roll back) and the thin draft dispatch.
//!
//! The forward itself runs on the worker lane (see [`super::dflash_lane`]); this
//! module owns only the KV bookkeeping the executor thread is responsible for.

use anyhow::Result;

use super::Qwen3Executor;
use super::RequestId;
use super::StepCommand;
use super::WorkerStepOutcome;
use crate::speculative::DraftPlan;
use crate::speculative::DraftResult;
use crate::speculative::VerifyPlan;
use crate::speculative::VerifyResult;

impl Qwen3Executor {
    pub(super) fn execute_speculative_verify_impl(
        &mut self,
        plan: VerifyPlan<'_>,
    ) -> Result<VerifyResult> {
        anyhow::ensure!(
            self.speculative.is_some(),
            "speculative verification requested but no draft model is loaded"
        );
        for req in plan.requests {
            anyhow::ensure!(
                !req.as_slice().is_empty(),
                "speculative verify request {:?} has an empty verify span",
                req.request_id
            );
            anyhow::ensure!(
                self.dflash_ready_requests.contains(&req.request_id),
                "speculative verification requested before DFlash state is ready for {:?}",
                req.request_id
            );
            anyhow::ensure!(
                self.request_kvs.contains_key(&req.request_id),
                "missing RequestKv for {:?}",
                req.request_id
            );
        }

        // Reserve KV slots for each request's full K+1 verify span. Roll back
        // every prior reservation if any single one fails — all-or-nothing.
        let mut scheduled = Vec::with_capacity(plan.requests.len());
        for req in plan.requests {
            let span_len = req.as_slice().len();
            let rkv = self
                .request_kvs
                .get_mut(&req.request_id)
                .expect("RequestKv was validated before speculative scheduling");
            if let Err(e) = rkv.schedule_speculative(span_len, self.kv_mgr.pool()) {
                self.revert_speculative_schedules(&scheduled);
                return Err(anyhow::anyhow!(
                    "schedule_speculative failed for {:?}: {e}",
                    req.request_id
                ));
            }
            scheduled.push(req.request_id);
        }

        let kv_views = plan
            .requests
            .iter()
            .map(|req| self.request_kvs[&req.request_id].speculative_view(req.as_slice().len()))
            .collect();

        let step = StepCommand::SpeculativeVerify {
            requests: plan.requests.to_vec(),
            kv_views,
            sample_seed: plan.sample_seed,
        };
        let outcome = match self.run_step(&step) {
            Ok(outcome) => outcome,
            Err(e) => {
                self.revert_speculative_schedules(&scheduled);
                return Err(e);
            }
        };
        let result = match outcome {
            WorkerStepOutcome::SpeculativeVerify(result) => result,
            other => {
                self.revert_speculative_schedules(&scheduled);
                return Err(anyhow::anyhow!(
                    "speculative verify returned unexpected: {}",
                    other.kind()
                ));
            }
        };
        if result.requests.len() != plan.requests.len() {
            self.revert_speculative_schedules(&scheduled);
            return Err(anyhow::anyhow!(
                "speculative verify returned {} request results for {} requests",
                result.requests.len(),
                plan.requests.len()
            ));
        }
        for (req, req_result) in plan.requests.iter().zip(&result.requests) {
            if req.request_id != req_result.request_id {
                self.revert_speculative_schedules(&scheduled);
                return Err(anyhow::anyhow!(
                    "speculative verify returned request {:?} for {:?}",
                    req_result.request_id,
                    req.request_id
                ));
            }
        }

        // Commit the accepted prefix of each request's KV and free the rest.
        // On a mid-loop failure, only the not-yet-applied requests roll back
        // (applied ones already committed and cannot be reverted).
        let mut applied = Vec::with_capacity(result.requests.len());
        for req_result in &result.requests {
            let rkv = self
                .request_kvs
                .get_mut(&req_result.request_id)
                .expect("request must exist after speculative verify");
            if let Err(e) = rkv.apply_speculative(&req_result.accepted_tokens, self.kv_mgr.pool()) {
                let unapplied = scheduled
                    .iter()
                    .copied()
                    .filter(|request_id| !applied.contains(request_id))
                    .collect::<Vec<_>>();
                self.revert_speculative_schedules(&unapplied);
                return Err(anyhow::anyhow!(
                    "apply_speculative failed for {:?}: {e}",
                    req_result.request_id
                ));
            }
            applied.push(req_result.request_id);
        }
        for req_result in &result.requests {
            self.save_sealed_blocks(req_result.request_id);
        }

        if let Some(counters) = self.spec_decode_counters.as_mut() {
            for (req, req_result) in plan.requests.iter().zip(&result.requests) {
                let num_draft_tokens = req.as_slice().len().saturating_sub(1);
                counters.observe_draft(num_draft_tokens, req_result.matched_draft_tokens);
            }
        }

        Ok(result)
    }

    pub(super) fn execute_speculative_draft_impl(
        &mut self,
        plan: DraftPlan<'_>,
    ) -> Result<DraftResult> {
        anyhow::ensure!(
            self.speculative.is_some(),
            "speculative draft requested but no draft model is loaded"
        );
        for req in plan.requests {
            anyhow::ensure!(
                self.dflash_ready_requests.contains(&req.request_id),
                "speculative draft requested before DFlash state is ready for {:?}",
                req.request_id
            );
        }
        let step = StepCommand::SpeculativeDraft {
            requests: plan.requests.to_vec(),
        };
        match self.run_step(&step)? {
            WorkerStepOutcome::SpeculativeDraft(result) => Ok(result),
            other => Err(anyhow::anyhow!(
                "speculative draft returned unexpected: {}",
                other.kind()
            )),
        }
    }

    /// Roll back speculative KV reservations. Each request reverts its own
    /// reservation independently (the LIFO block discipline is intra-sequence,
    /// via RAII); the reverse order here just mirrors schedule order and is
    /// cosmetic.
    fn revert_speculative_schedules(&mut self, request_ids: &[RequestId]) {
        for &request_id in request_ids.iter().rev() {
            let Some(rkv) = self.request_kvs.get_mut(&request_id) else {
                log::warn!(
                    "missing RequestKv while reverting speculative schedule for {request_id:?}"
                );
                continue;
            };
            if let Err(error) = rkv.revert_schedule() {
                log::warn!("failed to revert speculative schedule for {request_id:?}: {error}");
            }
        }
    }
}
