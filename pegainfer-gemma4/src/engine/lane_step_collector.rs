use std::collections::HashMap;
use std::collections::VecDeque;

use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::RequestUpdate;
use pegainfer_frontend::engine::StepReceiver;
use pegainfer_frontend::engine::Terminal;

pub(super) struct Drained {
    pub(super) tokens: usize,
    pub(super) cached: usize,
    pub(super) scheduled: usize,
    pub(super) finish: FinishReason,
    pub(super) ids: Vec<u32>,
}

pub(super) struct StepCollector {
    steps: StepReceiver,
    buffered: HashMap<RequestId, VecDeque<RequestUpdate>>,
}

/// A request the scheduler refused or failed before scheduling sends no
/// further update, so waiting for its Scheduled would block until the
/// stream closes at shutdown, which the waiting test never reaches.
fn ended_before_scheduled(id: RequestId, terminal: &Terminal) -> ! {
    match terminal {
        Terminal::Rejected { reason, .. } => {
            panic!("request {id:?} was rejected before it was scheduled: {reason}")
        }
        Terminal::Failed { message, .. } => {
            panic!("request {id:?} failed before it was scheduled: {message}")
        }
        Terminal::Finished { reason, .. } => {
            panic!("request {id:?} finished before it was scheduled: {reason:?}")
        }
    }
}

impl StepCollector {
    pub(super) fn new(steps: StepReceiver) -> Self {
        Self {
            steps,
            buffered: HashMap::new(),
        }
    }

    fn ingest(&mut self, update: RequestUpdate) {
        self.buffered
            .entry(update.id)
            .or_default()
            .push_back(update);
    }

    fn next_for(&mut self, id: RequestId) -> RequestUpdate {
        loop {
            if let Some(update) = self.buffered.get_mut(&id).and_then(VecDeque::pop_front) {
                return update;
            }
            let step = self
                .steps
                .blocking_recv()
                .expect("step stream closed while awaiting an update");
            for update in step.updates {
                self.ingest(update);
            }
        }
    }

    pub(super) fn wait_scheduled(&mut self, id: RequestId) {
        let mut held = Vec::new();
        loop {
            let update = self.next_for(id);
            if let Some(terminal) = &update.terminal
                && update.scheduled.is_none()
            {
                ended_before_scheduled(id, terminal);
            }
            let scheduled = update.scheduled.is_some();
            held.push(update);
            if scheduled {
                let queue = self.buffered.entry(id).or_default();
                for update in held.into_iter().rev() {
                    queue.push_front(update);
                }
                return;
            }
        }
    }

    pub(super) fn wait_scheduled_together(&mut self, ids: &[RequestId]) {
        loop {
            let step = self
                .steps
                .blocking_recv()
                .expect("step stream closed before the admission burst");
            for update in &step.updates {
                if let Some(terminal) = &update.terminal
                    && update.scheduled.is_none()
                    && ids.contains(&update.id)
                {
                    ended_before_scheduled(update.id, terminal);
                }
            }
            let any = ids.iter().any(|id| {
                step.updates
                    .iter()
                    .any(|update| update.id == *id && update.scheduled.is_some())
            });
            let all = ids.iter().all(|id| {
                step.updates
                    .iter()
                    .any(|update| update.id == *id && update.scheduled.is_some())
            });
            for update in step.updates {
                self.ingest(update);
            }
            if any {
                assert!(all, "the admission cohort split across scheduler steps");
                return;
            }
        }
    }

    pub(super) fn wait_tokens(&mut self, id: RequestId, wanted: usize) {
        let mut seen = 0;
        let mut held = Vec::new();
        while seen < wanted {
            let update = self.next_for(id);
            seen += update.tokens.len();
            assert!(
                update.terminal.is_none(),
                "request finished before {wanted} tokens"
            );
            held.push(update);
        }
        let queue = self.buffered.entry(id).or_default();
        for update in held.into_iter().rev() {
            queue.push_front(update);
        }
    }

    pub(super) fn drain(&mut self, id: RequestId, name: &str) -> Drained {
        let mut tokens = 0;
        let mut cached = 0;
        let mut scheduled = 0;
        let mut ids = Vec::new();
        loop {
            let update = self.next_for(id);
            if update.scheduled.is_some() {
                scheduled += 1;
            }
            cached = update.cached_tokens.unwrap_or(cached);
            tokens += update.tokens.len();
            ids.extend(update.tokens);
            match update.terminal {
                Some(Terminal::Finished { reason, .. }) => {
                    return Drained {
                        tokens,
                        cached,
                        scheduled,
                        finish: reason,
                        ids,
                    };
                }
                Some(Terminal::Rejected { reason, .. }) => {
                    panic!("{name}: rejected: {reason}")
                }
                Some(Terminal::Failed { message, .. }) => panic!("{name}: failed: {message}"),
                None => {}
            }
        }
    }

    pub(super) fn terminal(&mut self, id: RequestId) -> Terminal {
        loop {
            if let Some(terminal) = self.next_for(id).terminal {
                return terminal;
            }
        }
    }

    fn pump_available(&mut self) {
        while let Ok(step) = self.steps.try_recv() {
            for update in step.updates {
                self.ingest(update);
            }
        }
    }

    pub(super) fn buffered_tokens(&mut self, id: RequestId) -> usize {
        self.pump_available();
        self.buffered.get(&id).map_or(0, |updates| {
            updates.iter().map(|update| update.tokens.len()).sum()
        })
    }

    pub(super) fn saw_scheduled(&mut self, id: RequestId) -> bool {
        self.pump_available();
        self.buffered
            .get(&id)
            .is_some_and(|updates| updates.iter().any(|update| update.scheduled.is_some()))
    }

    pub(super) fn terminals_after_close(&mut self, id: RequestId) -> Vec<Terminal> {
        let mut terminals: Vec<Terminal> = self
            .buffered
            .remove(&id)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|update| update.terminal)
            .collect();
        while let Some(step) = self.steps.blocking_recv() {
            terminals.extend(
                step.updates
                    .into_iter()
                    .filter(|update| update.id == id)
                    .filter_map(|update| update.terminal),
            );
        }
        terminals
    }
}
