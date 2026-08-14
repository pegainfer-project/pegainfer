//! The model-side runtime trait and the polling loop that drives it.
//!
//! The loop lives here — once, for every model line — so its conventions
//! (drain order, one load publish and one commit per iteration, shutdown and
//! fatal handling) are code, not per-crate discipline. A model line's runtime
//! obligation is exactly the three [`Scheduler`] methods. Anything beyond
//! them (e.g. LoRA control) is not the contract's business: a scheduler that
//! serves an extra capability captures its own channel before spawn and
//! drains it inside `step`.

use super::emitter::StepEmitter;
use super::handle::LoadSnapshot;
use super::request_lifecycle::QueuedRequest;
use super::wiring::LiveScheduler;
use super::wiring::SchedulerBackend;
use super::wiring::scheduler_pair;

/// One scheduler's runtime behavior. Runs on a dedicated OS thread spawned
/// by [`spawn_scheduler`]; `Send` suffices.
pub trait Scheduler: Send {
    /// Take ownership of one submitted request. Ownership transfer only —
    /// every verdict (admit/reject/retire) is emitted from [`Self::step`],
    /// the single emission site. Nothing is lost by deferring: the driver
    /// commits once per iteration, so a verdict buffered in `step` lands in
    /// the same commit a submit-time verdict would have.
    fn submit(&mut self, req: QueuedRequest);

    /// Advance one step: admission, GPU work, and per-request emission. The
    /// driver loops over this without pause — an idle scheduler returns
    /// quickly and gets polled again, so in-flight completions from the
    /// scheduler's own worker threads are picked up within a step. Recoverable
    /// execution failures are the scheduler's to absorb (fail the touched
    /// requests, keep serving); `Err` means the engine is beyond use. The
    /// driver then winds down, and every request still held by the scheduler
    /// is answered by its handle's drop bomb.
    fn step(&mut self, emitter: &mut StepEmitter) -> anyhow::Result<()>;

    /// Occupancy snapshot; the driver publishes it once per iteration.
    fn load(&self) -> LoadSnapshot;
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
/// disconnected) and the scheduler reports itself drained, or when `step` reports a fatal
/// error. An idle iteration (nothing running or waiting) ends in a
/// [`std::hint::spin_loop`] before the next probe; busy iterations never
/// pause.
pub fn drive<S: Scheduler>(mut scheduler: S, backend: SchedulerBackend) {
    let SchedulerBackend {
        submissions,
        mut emitter,
        load,
    } = backend;
    let mut submissions_open = true;
    loop {
        loop {
            match submissions.try_recv() {
                Ok(req) => scheduler.submit(req),
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    submissions_open = false;
                    break;
                }
            }
        }
        let step = scheduler.step(&mut emitter);
        let snapshot = scheduler.load();
        load.publish(snapshot);
        emitter.commit_step();
        if let Err(error) = step {
            // Dropping the scheduler drops its held request handles; their
            // drop bombs answer every in-flight request with `Failed`.
            log::error!("scheduler fatal, engine winding down: {error:#}");
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
    use super::super::request_lifecycle::ActiveRequest;
    use super::super::step::Request;
    use super::super::step::Terminal;
    use super::*;
    use crate::engine::FinishReason;

    /// Emits one token per step per request and finishes at `max_tokens`.
    #[derive(Default)]
    struct EchoScheduler {
        queued: Vec<QueuedRequest>,
        running: Vec<(ActiveRequest, usize)>,
        fatal_next_step: bool,
    }

    impl Scheduler for EchoScheduler {
        fn submit(&mut self, req: QueuedRequest) {
            self.queued.push(req);
        }

        fn step(&mut self, emitter: &mut StepEmitter) -> anyhow::Result<()> {
            if self.fatal_next_step {
                anyhow::bail!("injected fatal");
            }
            for req in self.queued.drain(..) {
                let max_tokens = req.request().max_tokens;
                let active = emitter.admit(req);
                self.running.push((active, max_tokens));
            }
            let mut still_running = Vec::new();
            for (mut active, max_tokens) in self.running.drain(..) {
                if active.is_aborted() {
                    emitter.retire(active);
                    continue;
                }
                let next = active.completion_tokens() as u32;
                emitter.push_tokens(&mut active, &[next], &[]);
                if active.completion_tokens() >= max_tokens {
                    emitter.finish(active, FinishReason::Length);
                } else {
                    still_running.push((active, max_tokens));
                }
            }
            self.running = still_running;
            Ok(())
        }

        fn load(&self) -> LoadSnapshot {
            LoadSnapshot {
                num_running_reqs: self.running.len() as u64,
                num_waiting_reqs: self.queued.len() as u64,
                ..LoadSnapshot::default()
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
    fn fatal_step_fails_in_flight_requests_via_drop_bombs() {
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
        let mut saw_failed = false;
        while let Some(step) = steps.blocking_recv() {
            for update in step.updates {
                if matches!(update.terminal, Some(Terminal::Failed { .. })) {
                    saw_failed = true;
                }
            }
            if saw_failed {
                break;
            }
        }
        assert!(saw_failed, "in-flight request must be answered on fatal");
        partition
            .join
            .join()
            .expect("driver thread exits after fatal");
    }
}
