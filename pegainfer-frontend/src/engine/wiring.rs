//! Wiring between one frontend and one scheduler, plus the engine bundle a
//! model line hands back from `launch`.
//!
//! Both ends of a scheduler are minted together by [`scheduler_pair`], so a
//! model crate cannot cross-wire or forget a line: the scheduler side arrives
//! as one [`SchedulerBackend`] value, the frontend side as one
//! [`SchedulerHandle`]. Channel choices per direction: the submit channel is
//! crossbeam (sync consumer on the scheduler thread; senders never block on
//! unbounded channels), the step stream is tokio (async consumer in the
//! protocol stack; the sync producer's send never blocks either), load is a
//! shared cell (read-only pull, deliberately unsubscribable — see
//! [`LoadPublisher`]).
//!
//! How many schedulers an engine runs and what each one means (DP replicas,
//! anything else) is the model line's decision; the contract carries the
//! collection and attaches no rank semantics to it. Placement across
//! schedulers is frontend policy.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;

use super::control::LoraClient;
use super::handle::LoadSnapshot;
use super::kv::KvCapacity;
use super::request_lifecycle::HandleCore;
use super::request_lifecycle::QueuedRequest;
use super::request_lifecycle::RequestControl;
use super::request_lifecycle::StepReceiver;
use super::step::Request;
use super::step::RequestId;

/// Everything the scheduler thread consumes and produces. The driver
/// ([`super::drive`]) destructures this; model code only ever sees the
/// emitter and, through trait arguments, the request handles.
pub struct SchedulerBackend {
    pub(crate) submissions: crossbeam_channel::Receiver<QueuedRequest>,
    pub(crate) emitter: super::emitter::StepEmitter,
    pub(crate) load: LoadPublisher,
}

/// Sole writer of a scheduler's load cell; the driver publishes once per
/// iteration from [`super::Scheduler::load`].
///
/// Deliberately a plain cell and not a `watch` channel: the driver busy-polls,
/// so a subscription edge (`changed()`) would fire per spin and turn any
/// subscriber into a message flood at idle. With only [`SchedulerHandle::load`]
/// to read it, "notify me on load change" is unrepresentable — consumers pull
/// the snapshot at the moment they need one. A `Mutex` (not per-field atomics)
/// so a reader never sees fields torn across two steps; both sides touch it
/// uncontended for nanoseconds.
pub struct LoadPublisher(Arc<Mutex<LoadSnapshot>>);

impl LoadPublisher {
    pub(crate) fn publish(&self, snapshot: LoadSnapshot) {
        *self.0.lock().expect("load cell poisoned") = snapshot;
    }
}

/// The frontend's end of one running scheduler.
pub struct SchedulerHandle {
    submit_tx: crossbeam_channel::Sender<QueuedRequest>,
    steps: Option<StepReceiver>,
    load: Arc<Mutex<LoadSnapshot>>,
    next_id: AtomicU64,
    /// Kept so requests minted after the scheduler thread exits still get
    /// their drop-bomb terminal delivered (the handle needs a live sender).
    step_tx: super::request_lifecycle::StepSender,
}

impl SchedulerHandle {
    /// Mint identity, queue timestamp, and abort flag, then hand the request
    /// to the scheduler. Never fails: if the scheduler is gone, the
    /// `QueuedRequest`'s drop bomb answers the request with a `Failed`
    /// terminal on the step stream, which the caller observes like any other
    /// terminal.
    pub fn submit(&self, request: Request) -> RequestControl {
        let id = RequestId::new(self.next_id.fetch_add(1, Ordering::Relaxed));
        let abort = Arc::new(AtomicBool::new(false));
        let control = RequestControl::new(id, Arc::clone(&abort));
        let request = QueuedRequest::new(
            HandleCore {
                id,
                abort,
                tx: self.step_tx.clone(),
            },
            request,
            Instant::now(),
        );
        if let Err(returned) = self.submit_tx.send(request) {
            drop(returned.into_inner());
        }
        control
    }

    /// The step stream, handed out once (there is one stream and one
    /// consumer — the protocol stack's translation loop).
    pub fn take_steps(&mut self) -> Option<StepReceiver> {
        self.steps.take()
    }

    /// The scheduler's most recent load snapshot. Pull-only by design (see
    /// [`LoadPublisher`]): read it at the moment you need one — routing a
    /// request, stamping stats onto an outgoing batch, serving a scrape.
    pub fn load(&self) -> LoadSnapshot {
        *self.load.lock().expect("load cell poisoned")
    }
}

/// Mint both ends of one scheduler's wiring.
#[must_use]
pub fn scheduler_pair() -> (SchedulerHandle, SchedulerBackend) {
    let (submit_tx, submit_rx) = crossbeam_channel::unbounded();
    let (step_tx, step_rx) = tokio::sync::mpsc::unbounded_channel();
    let load = Arc::new(Mutex::new(LoadSnapshot::default()));
    (
        SchedulerHandle {
            submit_tx,
            steps: Some(step_rx),
            load: Arc::clone(&load),
            next_id: AtomicU64::new(0),
            step_tx: step_tx.clone(),
        },
        SchedulerBackend {
            submissions: submit_rx,
            emitter: super::emitter::StepEmitter::new(step_tx),
            load: LoadPublisher(load),
        },
    )
}

/// What `ModelLine::launch` returns for a step-driven engine — a handoff
/// package the protocol stack dissolves at wiring time. Required fields are
/// the onboarding checklist: an engine that reports no capacity or no
/// servable length says so explicitly with `None`, it cannot just forget.
pub struct Engine {
    /// The engine's running schedulers, one handle each. How many and what
    /// they mean is the model line's business.
    pub schedulers: Vec<LiveScheduler>,
    pub info: EngineInfo,
    /// LoRA adapter control, for engines that serve it. The `Option` is the
    /// capability — no separate flag exists to disagree with it.
    pub lora: Option<LoraClient>,
}

pub struct LiveScheduler {
    pub handle: SchedulerHandle,
    /// The driver thread. Joined by the server at shutdown, after the handles
    /// (and with them the submit senders) are dropped.
    pub join: std::thread::JoinHandle<()>,
}

/// What `ModelLine::launch` returns during the contract migration: either a
/// legacy per-token engine or a step-driven one. Deleted (in favor of
/// [`Engine`] alone) once every model line is migrated.
pub enum LaunchedEngine {
    Handle(super::handle::EngineHandle),
    Stepped(Engine),
}

impl From<super::handle::EngineHandle> for LaunchedEngine {
    fn from(handle: super::handle::EngineHandle) -> Self {
        Self::Handle(handle)
    }
}

impl From<Engine> for LaunchedEngine {
    fn from(engine: Engine) -> Self {
        Self::Stepped(engine)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct EngineInfo {
    /// KV pool capacity, or an explicit `None` for engines that do not report
    /// one (the frontend then skips batch-fit checks).
    pub kv_capacity: Option<KvCapacity>,
    /// Longest servable request in tokens, or an explicit `None` to leave the
    /// protocol stack's max-length validation at the model context window.
    pub servable_len: Option<u32>,
}
