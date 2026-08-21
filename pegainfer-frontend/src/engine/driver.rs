//! The model-side runtime trait and the polling loop that drives it.
//!
//! The loop lives here — once, for every model line — so its conventions
//! (drain order, one metrics publish and one commit per iteration, shutdown
//! and fatal handling) are code, not per-crate discipline. A model line's
//! runtime obligation is exactly the three [`Scheduler`] methods. Anything
//! beyond them (e.g. LoRA control) is not the contract's business: a
//! scheduler that serves an extra capability captures its own channel before
//! spawn and drains it inside `step`.

use super::ledger::RequestLedger;
use super::metrics::SchedulerMetrics;
use super::step::QueuedRequest;
use super::wiring::LiveScheduler;
use super::wiring::SchedulerBackend;
use super::wiring::scheduler_pair;

/// One scheduler's runtime behavior. Runs on a dedicated OS thread spawned
/// by [`spawn_scheduler`]; `Send` suffices.
pub trait Scheduler: Send {
    /// Take ownership of one submitted request. Ownership transfer only —
    /// every verdict (admit/reject/retire) is written to the ledger from
    /// [`Self::step`], the single emission site. Nothing is lost by
    /// deferring: the driver commits once per iteration, so a verdict
    /// recorded in `step` lands in the same commit a submit-time verdict
    /// would have.
    fn submit(&mut self, request: QueuedRequest);

    /// Advance one step: admission, GPU work, and per-request ledger writes.
    /// The driver loops over this without pause — an idle scheduler returns
    /// quickly and gets polled again, so in-flight completions from the
    /// scheduler's own worker threads are picked up within a step. Recoverable
    /// execution failures are the scheduler's to absorb (fail the touched
    /// requests, keep serving); `Err` means the engine is beyond use. The
    /// driver then writes off every open account with the error and winds
    /// down.
    fn step(&mut self, ledger: &mut RequestLedger) -> anyhow::Result<()>;

    /// Metrics snapshot; the driver publishes it once per iteration.
    fn metrics(&self) -> SchedulerMetrics;
}

/// Spawn one scheduler: mint the wiring, start the driver thread, return the
/// frontend end. `name` names the OS thread (shows in `top`/gdb).
pub fn spawn_scheduler<S: Scheduler + 'static>(name: &str, scheduler: S) -> LiveScheduler {
    let (handle, backend) = scheduler_pair();
    let join = std::thread::Builder::new()
        .name(name.to_string())
        .spawn(move || drive(scheduler, backend))
        .expect("failed to spawn scheduler thread");
    LiveScheduler { handle, join }
}

/// The polling loop. Exits when the frontend is gone (submission channel
/// disconnected) and the scheduler reports itself drained, or when `step`
/// reports a fatal error. An idle iteration (nothing running or waiting) ends
/// in a [`std::hint::spin_loop`] before the next probe; busy iterations never
/// pause.
pub fn drive<S: Scheduler>(mut scheduler: S, backend: SchedulerBackend) {
    let SchedulerBackend {
        submissions,
        mut ledger,
        metrics,
    } = backend;
    let mut submissions_open = true;
    loop {
        loop {
            match submissions.try_recv() {
                Ok(envelope) => {
                    let queued = ledger.register(envelope);
                    scheduler.submit(queued);
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    submissions_open = false;
                    break;
                }
            }
        }
        let step = scheduler.step(&mut ledger);
        let snapshot = scheduler.metrics();
        metrics.publish(&snapshot);
        ledger.commit_step();
        if let Err(error) = step {
            // The ledger holds an account for every unanswered request, so
            // the write-off reaches them all — including any the scheduler
            // lost track of — with the real error attached.
            log::error!("scheduler fatal, engine winding down: {error:#}");
            ledger.fail_all(&format!("engine fatal: {error:#}"));
            ledger.commit_step();
            return;
        }
        if snapshot.num_running_reqs == 0 && snapshot.num_waiting_reqs == 0 {
            if !submissions_open {
                log::info!("scheduler drained after frontend shutdown, exiting");
                return;
            }
            // The iteration did no work; relax the core's issue slots before
            // probing again. Latency is untouched — busy iterations never
            // reach this hint.
            std::hint::spin_loop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::step::Request;
    use super::super::step::RequestId;
    use super::super::step::Terminal;
    use super::*;
    use crate::engine::FinishReason;

    /// Emits one token per step per request and finishes at `max_tokens`.
    #[derive(Default)]
    struct EchoScheduler {
        queued: Vec<(RequestId, usize)>,
        running: Vec<(RequestId, usize)>,
        fatal_next_step: bool,
    }

    impl Scheduler for EchoScheduler {
        fn submit(&mut self, request: QueuedRequest) {
            self.queued.push((request.id, request.request.max_tokens));
        }

        fn step(&mut self, ledger: &mut RequestLedger) -> anyhow::Result<()> {
            if self.fatal_next_step && !(self.queued.is_empty() && self.running.is_empty()) {
                anyhow::bail!("injected fatal");
            }
            for (id, max_tokens) in self.queued.drain(..) {
                ledger.admit(id);
                self.running.push((id, max_tokens));
            }
            let mut still_running = Vec::new();
            for (id, max_tokens) in self.running.drain(..) {
                if ledger.is_aborted(id) {
                    ledger.retire(id);
                    continue;
                }
                let next = ledger.completion_tokens(id) as u32;
                ledger.push_tokens(id, &[next], &[]);
                if ledger.completion_tokens(id) >= max_tokens {
                    ledger.finish(id, FinishReason::Length);
                } else {
                    still_running.push((id, max_tokens));
                }
            }
            self.running = still_running;
            Ok(())
        }

        fn metrics(&self) -> SchedulerMetrics {
            SchedulerMetrics {
                num_running_reqs: self.running.len() as u64,
                num_waiting_reqs: self.queued.len() as u64,
                ..SchedulerMetrics::default()
            }
        }
    }

    fn request(max_tokens: usize) -> Request {
        Request {
            prompt_tokens: vec![1, 2],
            params: crate::sampler::SamplingParams::default(),
            max_tokens,
            lora_adapter: None,
            kv_transfer_params: None,
            logprobs: 0,
            echo: false,
            trace_parent: None,
            client_label: None,
        }
    }

    #[test]
    fn driven_engine_streams_and_drains_on_shutdown() {
        let partition = spawn_scheduler("test-echo", EchoScheduler::default());
        let mut handle = partition.handle;
        let mut steps = handle.take_steps().expect("step stream");

        let _control = handle.submit(request(3));
        let mut tokens = Vec::new();
        let mut terminal = None;
        while terminal.is_none() {
            let step = steps.blocking_recv().expect("step message");
            for update in step.updates {
                assert!(update.scheduled.is_some() || !tokens.is_empty());
                tokens.extend(update.tokens);
                if let Some(t) = update.terminal {
                    terminal = Some(t);
                }
            }
        }
        assert_eq!(tokens, vec![0, 1, 2]);
        assert!(matches!(
            terminal,
            Some(Terminal::Finished {
                reason: FinishReason::Length,
                prompt_tokens: 2,
                completion_tokens: 3,
            })
        ));

        // Dropping the handle disconnects the submission channel; a drained
        // scheduler exits.
        drop(handle);
        partition.join.join().expect("driver thread exits cleanly");
    }

    #[test]
    fn fatal_step_fails_in_flight_requests_with_the_error() {
        let partition = spawn_scheduler(
            "test-fatal",
            EchoScheduler {
                fatal_next_step: true,
                ..EchoScheduler::default()
            },
        );
        let mut handle = partition.handle;
        let mut steps = handle.take_steps().expect("step stream");

        let _control = handle.submit(request(100));
        let mut failure_message = None;
        while let Some(step) = steps.blocking_recv() {
            for update in step.updates {
                if let Some(Terminal::Failed { message, .. }) = update.terminal {
                    failure_message = Some(message);
                }
            }
            if failure_message.is_some() {
                break;
            }
        }
        let message = failure_message.expect("in-flight request must be answered on fatal");
        assert!(
            message.contains("injected fatal"),
            "the write-off must carry the real error: {message}"
        );
        partition
            .join
            .join()
            .expect("driver thread exits after fatal");
    }
}
