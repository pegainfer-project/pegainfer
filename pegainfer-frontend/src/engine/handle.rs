use std::any::Any;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread::JoinHandle;
use std::thread::{self};

use tokio::sync::mpsc;
use tokio::sync::oneshot;
use tokio::sync::watch;

use super::control::EngineCommand;
use super::control::EngineControlError;
use super::control::EngineControlRequest;
use super::control::EngineControlResult;
use super::control::LoadLoraAdapterRequest;
use super::control::UnloadLoraAdapterRequest;
use super::event::TokenEvent;
use super::event::unix_now_s;
use super::kv::KvBlockEvent;
use super::kv::KvCapacity;
use super::kv::KvPrefix;
use super::kv::SubmittedRequest;
use super::request::GenerateRequest;
use crate::parallel::ParallelConfig;

/// Launch-time options for building an engine: GPU placement, parallel
/// topology, collective backend, graph capture toggle, seed. Consumed once by
/// the model crate's `start_engine` / `launch`; [`EngineHandle`] is what the
/// frontend holds afterwards.
#[derive(Clone, Debug)]
pub struct EngineLoadOptions {
    pub enable_cuda_graph: bool,
    pub device_ordinals: Vec<usize>,
    pub parallel_config: Option<ParallelConfig>,
    pub ep_backend: EpBackend,
    pub seed: u64,
}

impl Default for EngineLoadOptions {
    fn default() -> Self {
        Self {
            enable_cuda_graph: true,
            device_ordinals: vec![0],
            parallel_config: None,
            ep_backend: EpBackend::Nccl,
            seed: 42,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum EpBackend {
    #[default]
    Nccl,
    DeepEp,
}

/// Live KV-cache occupancy the scheduler republishes after every step.
///
/// `kv_used_blocks` is the load signal an out-of-band consumer (e.g. a Dynamo
/// KV router) scores against; `kv_total_blocks` is the engine's whole-pool
/// capacity (the same number advertised as the servable ceiling), so the
/// consumer can derive fractional usage without a second query. Carried over a
/// [`watch`] channel: the scheduler is the sole writer and never blocks on a
/// reader, and a reader only ever sees the latest snapshot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LoadSnapshot {
    pub kv_used_blocks: u64,
    pub kv_total_blocks: u64,
    /// Requests currently occupying a decode/prefill slot.
    pub num_running_reqs: u64,
    /// Requests admitted but not yet running (KV pressure, prefetch wait).
    pub num_waiting_reqs: u64,
}

#[derive(Clone)]
pub struct EngineHandle {
    inner: Arc<EngineInner>,
    servable_len: Option<u32>,
    /// KV pool capacity in blocks + block size, or `None` if the engine did not
    /// report it. See [`KvCapacity`].
    kv_capacity: Option<KvCapacity>,
    /// One optional live-load feed per scheduler partition. A normal engine
    /// has one partition; a data-parallel engine exposes one per logical DP
    /// rank. Each handle clone owns cloned receivers and `watch` fans out.
    load_watches: Vec<Option<watch::Receiver<LoadSnapshot>>>,
    /// Block store/remove feed for a cache-aware router, or `None` if not wired.
    /// `mpsc` (every event matters — unlike the coalescing load feed), so the
    /// single receiver is handed out exactly once via [`Self::take_kv_events`];
    /// the shared cell lets all handle clones agree on who took it.
    kv_events: Option<Arc<Mutex<Option<mpsc::UnboundedReceiver<KvBlockEvent>>>>>,
}

struct EngineInner {
    /// One submit channel per scheduler partition (logical DP rank). A
    /// single-partition engine holds exactly one sender.
    submit_txs: Vec<mpsc::UnboundedSender<SubmittedRequest>>,
    command_tx: Option<mpsc::UnboundedSender<EngineCommand>>,
    join_handles: Vec<JoinHandle<()>>,
}

impl EngineHandle {
    pub fn new(submit_tx: mpsc::UnboundedSender<SubmittedRequest>) -> Self {
        Self::from_parts(vec![submit_tx], None, Vec::new())
    }

    #[cfg(test)]
    fn new_with_command_channel(command_tx: mpsc::UnboundedSender<EngineCommand>) -> Self {
        Self::from_parts(Vec::new(), Some(command_tx), Vec::new())
    }

    pub fn new_with_command_channel_and_join_handle(
        command_tx: mpsc::UnboundedSender<EngineCommand>,
        join_handle: JoinHandle<()>,
    ) -> Self {
        Self::from_parts(Vec::new(), Some(command_tx), vec![join_handle])
    }

    /// Construct a handle that owns the engine thread shutdown.
    ///
    /// Dropping the last handle clone closes the submit channel and then waits
    /// for the thread to return. That final drop may block until in-flight
    /// generation and backend teardown finish.
    pub fn new_with_join_handle(
        submit_tx: mpsc::UnboundedSender<SubmittedRequest>,
        join_handle: JoinHandle<()>,
    ) -> Self {
        Self::from_parts(vec![submit_tx], None, vec![join_handle])
    }

    /// Construct a multi-partition handle: one submit channel and one owned
    /// engine thread per logical scheduler partition (e.g. one autonomous
    /// engine per DP rank). [`Self::submit`] routes by the request's
    /// `data_parallel_rank`; unbound requests go to the least-loaded
    /// partition (waiting requests weigh 4x, the vLLM DP policy).
    ///
    /// Dropping the last handle clone closes every channel and joins every
    /// thread in partition order.
    pub fn new_with_join_handles(
        submit_txs: Vec<mpsc::UnboundedSender<SubmittedRequest>>,
        join_handles: Vec<JoinHandle<()>>,
    ) -> Self {
        assert!(
            !submit_txs.is_empty(),
            "an engine must expose at least one scheduler partition"
        );
        Self::from_parts(submit_txs, None, join_handles)
    }

    fn from_parts(
        submit_txs: Vec<mpsc::UnboundedSender<SubmittedRequest>>,
        command_tx: Option<mpsc::UnboundedSender<EngineCommand>>,
        join_handles: Vec<JoinHandle<()>>,
    ) -> Self {
        Self {
            inner: Arc::new(EngineInner {
                submit_txs,
                command_tx,
                join_handles,
            }),
            servable_len: None,
            kv_capacity: None,
            load_watches: vec![None],
            kv_events: None,
        }
    }

    #[must_use]
    pub fn with_servable_len(mut self, servable_len: u32) -> Self {
        self.servable_len = Some(servable_len);
        self
    }

    pub fn servable_len(&self) -> Option<u32> {
        self.servable_len
    }

    #[must_use]
    pub fn with_kv_capacity(mut self, kv_capacity: KvCapacity) -> Self {
        self.kv_capacity = Some(kv_capacity);
        self
    }

    /// KV pool capacity, if the engine reported it. A batch whose per-request
    /// block footprint exceeds [`KvCapacity::total_blocks`] cannot be resident
    /// at once.
    pub(crate) fn kv_capacity(&self) -> Option<KvCapacity> {
        self.kv_capacity
    }

    #[must_use]
    pub fn with_load_watch(mut self, load_watch: watch::Receiver<LoadSnapshot>) -> Self {
        self.load_watches = vec![Some(load_watch)];
        self
    }

    /// Attach one load feed per logical scheduler partition.
    ///
    /// The vector is also the engine's frontend-visible DP topology, so an
    /// empty vector is an invalid engine rather than a single partition with
    /// missing metrics.
    #[must_use]
    pub fn with_load_watches(mut self, load_watches: Vec<watch::Receiver<LoadSnapshot>>) -> Self {
        assert!(
            !load_watches.is_empty(),
            "an engine must expose at least one scheduler partition"
        );
        self.load_watches = load_watches.into_iter().map(Some).collect();
        self
    }

    /// Number of logical scheduler partitions exposed to the frontend.
    pub(crate) fn scheduler_partition_count(&self) -> usize {
        self.load_watches.len()
    }

    /// A receiver for one partition's live load, if it wired one. Awaiting
    /// [`watch::Receiver::changed`] wakes once per scheduler step under load and
    /// stays quiet when idle, so a consumer republishes on real change rather
    /// than polling. `None` if the partition does not report a load feed or the
    /// index is outside the engine topology.
    pub(crate) fn load_watch_for(&self, partition: usize) -> Option<watch::Receiver<LoadSnapshot>> {
        self.load_watches.get(partition)?.clone()
    }

    /// Single-partition compatibility accessor.
    pub fn load_watch(&self) -> Option<watch::Receiver<LoadSnapshot>> {
        self.load_watch_for(0)
    }

    #[must_use]
    pub fn with_kv_events(mut self, rx: mpsc::UnboundedReceiver<KvBlockEvent>) -> Self {
        self.kv_events = Some(Arc::new(Mutex::new(Some(rx))));
        self
    }

    /// Take the engine's KV block-event receiver. Returns the receiver on the
    /// first call and `None` thereafter (there is one stream and one consumer —
    /// the cache-aware router pump). `None` also if the engine wired no feed.
    pub fn take_kv_events(&self) -> Option<mpsc::UnboundedReceiver<KvBlockEvent>> {
        self.kv_events
            .as_ref()?
            .lock()
            .expect("kv-events cell poisoned")
            .take()
    }

    /// Submit an unresolved request: no KV-prefix resolution ran for it, so
    /// the scheduler receives [`KvPrefix::none`] and prefills from its own
    /// GPU prefix match alone. See [`Self::submit_resolved`].
    #[allow(clippy::result_large_err)]
    pub fn submit(
        &self,
        req: GenerateRequest,
    ) -> std::result::Result<(), mpsc::error::SendError<GenerateRequest>> {
        self.submit_resolved(req, KvPrefix::none())
    }

    /// Submit a request together with its KV-prefix resolution. Routing is by
    /// the request's `data_parallel_rank` (unbound requests go to the
    /// least-loaded partition), so the caller that resolved the prefix must
    /// have bound the request to the rank it resolved against.
    ///
    /// On the legacy command path (an engine wired with only a command
    /// channel) the prefix is dropped: those engines never resolve prefixes,
    /// and dropping releases the (necessarily absent) hold.
    #[allow(clippy::result_large_err)]
    fn submit_resolved(
        &self,
        req: GenerateRequest,
        kv_prefix: KvPrefix,
    ) -> std::result::Result<(), mpsc::error::SendError<GenerateRequest>> {
        if !self.inner.submit_txs.is_empty() {
            // A resolved prefix binds the request: its hold pins blocks on
            // the rank it resolved against, so that rank wins the routing.
            // A caller-bound rank that disagrees is a caller bug.
            let partition = match (kv_prefix.rank(), req.data_parallel_rank) {
                (Some(resolved), bound) => {
                    debug_assert!(
                        bound.is_none_or(|b| b == resolved),
                        "request bound to rank {bound:?} but its prefix resolved on rank {resolved}"
                    );
                    resolved
                }
                (None, Some(bound)) => bound,
                (None, None) => self.least_loaded_partition(),
            };
            if let Some(submit_tx) = self.inner.submit_txs.get(partition) {
                return submit_tx
                    .send((req, kv_prefix))
                    .map_err(|err| mpsc::error::SendError(err.0.0));
            }
            // An out-of-range rank is a caller error, not an engine
            // failure: answer the request with the standard
            // Scheduled → Rejected pair instead of failing the submit.
            reject_unroutable(&req, partition, self.inner.submit_txs.len());
            return Ok(());
        }
        match self.inner.command_tx.as_ref() {
            Some(command_tx) => command_tx
                .send(EngineCommand::Generate(Box::new(req)))
                .map_err(|err| match err.0 {
                    EngineCommand::Generate(req) => mpsc::error::SendError(*req),
                    EngineCommand::Control(_) => unreachable!("submitted generate command"),
                }),
            None => Err(mpsc::error::SendError(req)),
        }
    }

    /// Placement for an unbound request: the partition with the lowest
    /// waiting-weighted load (`running + 4 × waiting`, the same weight the
    /// vLLM DP load balancer uses), ties to the lowest index. Scores come
    /// from the partitions' load watches; a partition without a feed scores
    /// zero, so the single-partition degenerate always returns 0.
    fn least_loaded_partition(&self) -> usize {
        self.inner
            .submit_txs
            .iter()
            .enumerate()
            .min_by_key(|(partition, _)| {
                let score = self
                    .load_watches
                    .get(*partition)
                    .and_then(Option::as_ref)
                    .map_or(0, |watch| {
                        let snapshot = watch.borrow();
                        snapshot.num_running_reqs + 4 * snapshot.num_waiting_reqs
                    });
                (score, *partition)
            })
            .map_or(0, |(partition, _)| partition)
    }

    pub fn supports_lora_control(&self) -> bool {
        self.inner.command_tx.is_some()
    }

    pub async fn load_lora_adapter(
        &self,
        request: LoadLoraAdapterRequest,
    ) -> EngineControlResult<()> {
        match self.inner.command_tx.as_ref() {
            Some(command_tx) => {
                let (response_tx, response_rx) = oneshot::channel();
                command_tx
                    .send(EngineCommand::Control(
                        EngineControlRequest::LoadLoraAdapter {
                            request,
                            response_tx,
                        },
                    ))
                    .map_err(|_| EngineControlError::ChannelClosed)?;

                response_rx
                    .await
                    .map_err(|_| EngineControlError::ChannelClosed)?
                    .map_err(EngineControlError::OperationFailed)
            }
            None => Err(EngineControlError::Unsupported(
                "engine does not support dynamic LoRA adapter loading",
            )),
        }
    }

    pub async fn list_lora_adapters(&self) -> EngineControlResult<Vec<String>> {
        match self.inner.command_tx.as_ref() {
            Some(command_tx) => {
                let (response_tx, response_rx) = oneshot::channel();
                command_tx
                    .send(EngineCommand::Control(
                        EngineControlRequest::ListLoraAdapters { response_tx },
                    ))
                    .map_err(|_| EngineControlError::ChannelClosed)?;

                response_rx
                    .await
                    .map_err(|_| EngineControlError::ChannelClosed)?
                    .map_err(EngineControlError::OperationFailed)
            }
            None => Err(EngineControlError::Unsupported(
                "engine does not support dynamic LoRA adapter loading",
            )),
        }
    }

    pub async fn unload_lora_adapter(
        &self,
        request: UnloadLoraAdapterRequest,
    ) -> EngineControlResult<()> {
        match self.inner.command_tx.as_ref() {
            Some(command_tx) => {
                let (response_tx, response_rx) = oneshot::channel();
                command_tx
                    .send(EngineCommand::Control(
                        EngineControlRequest::UnloadLoraAdapter {
                            request,
                            response_tx,
                        },
                    ))
                    .map_err(|_| EngineControlError::ChannelClosed)?;

                response_rx
                    .await
                    .map_err(|_| EngineControlError::ChannelClosed)?
                    .map_err(EngineControlError::OperationFailed)
            }
            None => Err(EngineControlError::Unsupported(
                "engine does not support dynamic LoRA adapter loading",
            )),
        }
    }
}

pub fn panic_message(payload: &dyn Any) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("<non-string panic payload>")
}

/// Answer a request the handle cannot place (a `data_parallel_rank` outside
/// the engine's partition topology) with the standard Scheduled → Rejected
/// pair — the same surface a model scheduler's intake rejection produces.
fn reject_unroutable(req: &GenerateRequest, partition: usize, partitions: usize) {
    let queued_at_unix_s = req.queued_at_unix_s.unwrap_or_else(unix_now_s);
    let _ = req.token_tx.send(TokenEvent::Scheduled {
        queued_at_unix_s,
        scheduled_at_unix_s: unix_now_s(),
        prompt_tokens: req.prompt_tokens.len(),
        cached_tokens: 0,
    });
    let _ = req.token_tx.send(TokenEvent::Rejected {
        message: format!("data_parallel_rank {partition} is outside 0..{partitions}"),
        prompt_tokens: req.prompt_tokens.len(),
        completion_tokens: 0,
    });
}

impl Drop for EngineInner {
    fn drop(&mut self) {
        self.submit_txs.clear();
        let _ = self.command_tx.take();
        for join_handle in self.join_handles.drain(..) {
            if join_handle.thread().id() != thread::current().id() {
                if let Err(panic) = join_handle.join() {
                    log::warn!(
                        "engine thread panicked during shutdown: {}",
                        panic_message(panic.as_ref())
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::Ordering;
    use std::thread;

    use super::*;
    use crate::engine::TokenSink;
    use crate::engine::TokenStreamReceiver;
    use crate::sampler::SamplingParams;

    #[test]
    fn joins_owned_thread_after_last_handle_drop() {
        let (submit_tx, mut submit_rx) = mpsc::unbounded_channel::<SubmittedRequest>();
        let exited = Arc::new(AtomicBool::new(false));
        let thread_exited = Arc::clone(&exited);
        let join_handle = thread::spawn(move || {
            while submit_rx.blocking_recv().is_some() {}
            thread_exited.store(true, Ordering::SeqCst);
        });
        let handle = EngineHandle::new_with_join_handle(submit_tx, join_handle);
        let clone = handle.clone();

        drop(handle);
        assert!(!exited.load(Ordering::SeqCst));

        drop(clone);
        assert!(exited.load(Ordering::SeqCst));
    }

    fn routed_request(rank: Option<usize>) -> (GenerateRequest, TokenStreamReceiver) {
        let (token_tx, token_rx) = TokenSink::standalone();
        (
            GenerateRequest {
                request_id: None,
                queued_at_unix_s: None,
                trace_parent: None,
                data_parallel_rank: rank,
                prompt_tokens: vec![1],
                params: SamplingParams::default(),
                max_tokens: 1,
                lora_adapter: None,
                kv_transfer_params: None,
                token_tx,
                logprobs: 0,
                echo: false,
            },
            token_rx,
        )
    }

    #[test]
    fn multi_partition_submit_routes_by_bound_rank() {
        let (tx0, mut rx0) = mpsc::unbounded_channel::<SubmittedRequest>();
        let (tx1, mut rx1) = mpsc::unbounded_channel::<SubmittedRequest>();
        let handle = EngineHandle::new_with_join_handles(vec![tx0, tx1], Vec::new());

        let (req, _events) = routed_request(Some(1));
        handle.submit(req).expect("submit");
        assert!(rx0.try_recv().is_err());
        let (routed, kv_prefix) = rx1.try_recv().expect("routed to rank 1");
        assert_eq!(routed.prompt_tokens, [1]);
        assert_eq!(kv_prefix.hit_tokens(), 0);
    }

    #[test]
    fn multi_partition_submit_places_unbound_on_the_least_loaded() {
        let (tx0, mut rx0) = mpsc::unbounded_channel::<SubmittedRequest>();
        let (tx1, mut rx1) = mpsc::unbounded_channel::<SubmittedRequest>();
        let (load_tx0, load_rx0) = watch::channel(LoadSnapshot {
            num_running_reqs: 2,
            ..LoadSnapshot::default()
        });
        let (_load_tx1, load_rx1) = watch::channel(LoadSnapshot {
            num_waiting_reqs: 1,
            ..LoadSnapshot::default()
        });
        let handle = EngineHandle::new_with_join_handles(vec![tx0, tx1], Vec::new())
            .with_load_watches(vec![load_rx0, load_rx1]);

        // Scores are running + 4 × waiting: rank 0 scores 2, rank 1 scores 4.
        let (req, _events) = routed_request(None);
        handle.submit(req).expect("submit");
        assert!(rx0.try_recv().is_ok());
        assert!(rx1.try_recv().is_err());

        // Rank 0 rises to 6 running while rank 1 still waits 1: 4 < 6, so the
        // next unbound request tips to rank 1.
        load_tx0.send_replace(LoadSnapshot {
            num_running_reqs: 6,
            ..LoadSnapshot::default()
        });
        let (req, _events) = routed_request(None);
        handle.submit(req).expect("submit");
        assert!(rx1.try_recv().is_ok());
    }

    #[test]
    fn multi_partition_out_of_range_rank_is_rejected_not_dropped() {
        let (tx0, _rx0) = mpsc::unbounded_channel::<SubmittedRequest>();
        let handle = EngineHandle::new_with_join_handles(vec![tx0], Vec::new());

        let (req, mut events) = routed_request(Some(7));
        handle
            .submit(req)
            .expect("out-of-range is answered, not an error");
        assert!(matches!(
            events.try_recv().map(|(_, event)| event),
            Ok(TokenEvent::Scheduled { .. })
        ));
        assert!(matches!(
            events.try_recv().map(|(_, event)| event),
            Ok(TokenEvent::Rejected { ref message, .. }) if message.contains("outside 0..1")
        ));
    }

    #[test]
    fn multi_partition_drop_joins_every_thread() {
        let exited: Vec<_> = (0..3).map(|_| Arc::new(AtomicBool::new(false))).collect();
        let (txs, joins): (Vec<_>, Vec<_>) = exited
            .iter()
            .map(|flag| {
                let (tx, mut rx) = mpsc::unbounded_channel::<SubmittedRequest>();
                let flag = Arc::clone(flag);
                let join = thread::spawn(move || {
                    while rx.blocking_recv().is_some() {}
                    flag.store(true, Ordering::SeqCst);
                });
                (tx, join)
            })
            .unzip();
        let handle = EngineHandle::new_with_join_handles(txs, joins);
        drop(handle);
        for flag in &exited {
            assert!(flag.load(Ordering::SeqCst));
        }
    }

    #[test]
    fn lora_control_support_is_opt_in() {
        let (submit_tx, _submit_rx) = mpsc::unbounded_channel::<SubmittedRequest>();
        let handle = EngineHandle::new(submit_tx);
        assert!(!handle.supports_lora_control());

        let (command_tx, _command_rx) = mpsc::unbounded_channel::<EngineCommand>();
        let handle = EngineHandle::new_with_command_channel(command_tx);
        assert!(handle.supports_lora_control());
    }

    #[test]
    fn load_watches_define_scheduler_partitions() {
        let (submit_tx, _submit_rx) = mpsc::unbounded_channel::<SubmittedRequest>();
        let (_load_tx0, load_rx0) = watch::channel(LoadSnapshot {
            num_running_reqs: 1,
            ..LoadSnapshot::default()
        });
        let (_load_tx1, load_rx1) = watch::channel(LoadSnapshot {
            num_waiting_reqs: 2,
            ..LoadSnapshot::default()
        });
        let handle = EngineHandle::new(submit_tx).with_load_watches(vec![load_rx0, load_rx1]);

        assert_eq!(handle.scheduler_partition_count(), 2);
        assert_eq!(
            handle
                .load_watch_for(0)
                .expect("rank 0 watch")
                .borrow()
                .num_running_reqs,
            1
        );
        assert_eq!(
            handle
                .load_watch_for(1)
                .expect("rank 1 watch")
                .borrow()
                .num_waiting_reqs,
            2
        );
        assert!(handle.load_watch_for(2).is_none());
    }

    #[tokio::test]
    async fn load_lora_adapter_sends_control_command() {
        let (command_tx, mut command_rx) = mpsc::unbounded_channel::<EngineCommand>();
        let handle = EngineHandle::new_with_command_channel(command_tx);

        let request = LoadLoraAdapterRequest {
            lora_name: "adapter-a".to_string(),
            lora_path: PathBuf::from("/tmp/adapter-a"),
            load_inplace: false,
        };
        let load = tokio::spawn({
            let handle = handle.clone();
            let request = request.clone();
            async move { handle.load_lora_adapter(request).await }
        });

        let command = command_rx.recv().await.expect("control command");
        match command {
            EngineCommand::Control(EngineControlRequest::LoadLoraAdapter {
                request: actual,
                response_tx,
            }) => {
                assert_eq!(actual, request);
                response_tx.send(Ok(())).expect("send load result");
            }
            EngineCommand::Control(
                EngineControlRequest::UnloadLoraAdapter { .. }
                | EngineControlRequest::ListLoraAdapters { .. },
            ) => {
                panic!("expected LoRA load command")
            }
            EngineCommand::Generate(_) => panic!("expected LoRA control command"),
        }

        load.await.expect("join load task").expect("load succeeded");
    }

    #[tokio::test]
    async fn list_lora_adapters_sends_control_command() {
        let (command_tx, mut command_rx) = mpsc::unbounded_channel::<EngineCommand>();
        let handle = EngineHandle::new_with_command_channel(command_tx);

        let list = tokio::spawn({
            let handle = handle.clone();
            async move { handle.list_lora_adapters().await }
        });

        let command = command_rx.recv().await.expect("control command");
        match command {
            EngineCommand::Control(EngineControlRequest::ListLoraAdapters { response_tx }) => {
                response_tx
                    .send(Ok(vec!["adapter-a".to_string()]))
                    .expect("send list result");
            }
            EngineCommand::Control(
                EngineControlRequest::LoadLoraAdapter { .. }
                | EngineControlRequest::UnloadLoraAdapter { .. },
            ) => {
                panic!("expected LoRA list command")
            }
            EngineCommand::Generate(_) => panic!("expected LoRA control command"),
        }

        assert_eq!(
            list.await.expect("join load task").expect("list succeeded"),
            vec!["adapter-a"]
        );
    }

    #[tokio::test]
    async fn load_lora_adapter_reports_unsupported_without_control() {
        let (submit_tx, _submit_rx) = mpsc::unbounded_channel::<SubmittedRequest>();
        let handle = EngineHandle::new(submit_tx);
        let error = handle
            .load_lora_adapter(LoadLoraAdapterRequest {
                lora_name: "adapter-a".to_string(),
                lora_path: PathBuf::from("/tmp/adapter-a"),
                load_inplace: false,
            })
            .await
            .expect_err("control should be unsupported");
        assert_eq!(
            error,
            EngineControlError::Unsupported("engine does not support dynamic LoRA adapter loading")
        );
    }

    #[tokio::test]
    async fn unload_lora_adapter_sends_control_command() {
        let (command_tx, mut command_rx) = mpsc::unbounded_channel::<EngineCommand>();
        let handle = EngineHandle::new_with_command_channel(command_tx);

        let request = UnloadLoraAdapterRequest {
            lora_name: "adapter-a".to_string(),
            lora_int_id: None,
        };
        let unload = tokio::spawn({
            let handle = handle.clone();
            let request = request.clone();
            async move { handle.unload_lora_adapter(request).await }
        });

        let command = command_rx.recv().await.expect("control command");
        match command {
            EngineCommand::Control(EngineControlRequest::UnloadLoraAdapter {
                request: actual,
                response_tx,
            }) => {
                assert_eq!(actual, request);
                response_tx.send(Ok(())).expect("send unload result");
            }
            EngineCommand::Control(
                EngineControlRequest::LoadLoraAdapter { .. }
                | EngineControlRequest::ListLoraAdapters { .. },
            ) => {
                panic!("expected LoRA unload command")
            }
            EngineCommand::Generate(_) => panic!("expected LoRA control command"),
        }

        unload
            .await
            .expect("join unload task")
            .expect("unload succeeded");
    }
}
