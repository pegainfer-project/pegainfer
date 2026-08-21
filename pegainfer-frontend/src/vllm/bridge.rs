use std::collections::BTreeSet;
use std::collections::HashMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU8;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use fastrace::Span;
use fastrace::collector::SpanContext;
use log::info;
use log::warn;
use tokio::sync::mpsc;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::EngineId;
use vllm_engine_core_client::protocol::dtype::ModelDtype;
use vllm_engine_core_client::protocol::encode_msgpack;
use vllm_engine_core_client::protocol::handshake::EngineCoreReadyResponse;
use vllm_engine_core_client::protocol::logprobs::Logprobs;
use vllm_engine_core_client::protocol::logprobs::MaybeWireLogprobs;
use vllm_engine_core_client::protocol::logprobs::PositionLogprobs;
use vllm_engine_core_client::protocol::output::EngineCoreEvent;
use vllm_engine_core_client::protocol::output::EngineCoreEventType;
use vllm_engine_core_client::protocol::output::EngineCoreFinishReason;
use vllm_engine_core_client::protocol::output::EngineCoreOutput;
use vllm_engine_core_client::protocol::output::EngineCoreOutputs;
use vllm_engine_core_client::protocol::output::RequestBatchOutputs;
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_engine_core_client::protocol::output::UtilityCallOutput;
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::request::EngineCoreRequestType;
use vllm_engine_core_client::protocol::stats::PrefillStats;
use vllm_engine_core_client::protocol::stats::SchedulerStats;
use vllm_engine_core_client::protocol::stats::SpecDecodingStats;
use vllm_engine_core_client::protocol::utility::UtilityCallId;
use vllm_engine_core_client::protocol::utility::UtilityOutput;
use vllm_engine_core_client::protocol::utility::UtilityResultEnvelope;
use zeromq::DealerSocket;
use zeromq::PushSocket;
use zeromq::SocketOptions;
use zeromq::ZmqMessage;
use zeromq::prelude::Socket;
use zeromq::prelude::SocketRecv;
use zeromq::prelude::SocketSend;
use zeromq::util::PeerIdentity;

use crate::engine::EngineHandle;
use crate::engine::FinishReason;
use crate::engine::GenerateRequest;
use crate::engine::RequestAbortReason;
use crate::engine::RequestTag;
use crate::engine::SchedulerMetrics;
use crate::engine::SpecDecodeCounters;
use crate::engine::TokenEvent;
use crate::engine::TokenSink;
use crate::engine::TokenStreamReceiver;
use crate::vllm::wire::convert_finish_reason;
use crate::vllm::wire::convert_sampling;
use crate::vllm::wire::lora_adapter_from_sampling_params;
use crate::vllm::wire::requested_logprobs;
use crate::vllm::wire::to_wire_position_logprobs;

pub(crate) struct LocalEngineBridge {
    pub(crate) input_address: String,
    pub(crate) output_address: String,
    pub(crate) handle: EngineHandle,
    pub(crate) max_model_len: u32,
    pub(crate) engine_index: u32,
    pub(crate) data_parallel_size: u32,
    pub(crate) metrics_watch: Option<watch::Receiver<SchedulerMetrics>>,
}

impl LocalEngineBridge {
    pub(crate) async fn run(self, shutdown: CancellationToken) -> Result<()> {
        let BridgeLink {
            mut input,
            output_tx,
            mut child_tasks,
        } = connect_link(
            &self.input_address,
            &self.output_address,
            self.engine_index,
            self.data_parallel_size,
            self.max_model_len,
            self.handle.kv_capacity(),
            self.metrics_watch.clone(),
            &shutdown,
        )
        .await?;

        // One shared channel carries every request's token events, tagged by
        // request id; this loop is the sole consumer. Per-request state lives
        // in `streams`, keyed by the same tag, and holds the abort reason the
        // scheduler observes (via `TokenSink`) when an abort flips it.
        let (event_tx, mut event_rx) = mpsc::unbounded_channel();
        let mut streams: HashMap<RequestTag, RequestStreamState> = HashMap::new();

        info!(
            "local vLLM engine {} bridge connected: input={}, output={}, max_model_len={}",
            self.engine_index, self.input_address, self.output_address, self.max_model_len
        );

        let run_result = loop {
            tokio::select! {
                biased;
                () = shutdown.cancelled() => break Ok(()),
                joined = child_tasks.join_next(), if !child_tasks.is_empty() => {
                    if shutdown.is_cancelled() {
                        break Ok(());
                    }
                    break match joined {
                        Some(Ok((name, Ok(())))) => {
                            Err(anyhow::anyhow!("local engine {name} exited unexpectedly"))
                        }
                        Some(Ok((name, Err(error)))) => {
                            Err(error).with_context(|| format!("local engine {name} failed"))
                        }
                        Some(Err(error)) => {
                            Err(anyhow::anyhow!("local engine child task panicked: {error}"))
                        }
                        None => Err(anyhow::anyhow!("local engine child task set became empty")),
                    };
                }
                Some(first) = event_rx.recv() => {
                    if let Err(error) = dispatch_burst(
                        self.engine_index,
                        first,
                        &mut event_rx,
                        &mut streams,
                        &output_tx,
                    )
                    {
                        break Err(error).context("failed to dispatch local engine output");
                    }
                }
                recv = input.recv() => {
                    let message = match recv.context("failed to receive local engine request") {
                        Ok(message) => message,
                        Err(error) => break Err(error),
                    };
                    if let Err(error) = self.handle_message(
                        message,
                        &event_tx,
                        &output_tx,
                        &mut streams,
                    ) {
                        break Err(error).context("failed to handle local engine request");
                    }
                }
            }
        };

        // Cancel every in-flight request so the scheduler retires them on its
        // next emit instead of pushing into a channel no one drains.
        for state in streams.values() {
            state.abort(RequestAbortReason::Cancelled);
        }
        drop(output_tx);
        child_tasks.abort_all();
        while child_tasks.join_next().await.is_some() {}

        run_result
    }

    fn handle_message(
        &self,
        message: ZmqMessage,
        event_tx: &mpsc::UnboundedSender<(RequestTag, TokenEvent)>,
        output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
        streams: &mut HashMap<RequestTag, RequestStreamState>,
    ) -> Result<()> {
        let frames = message.into_vec();
        if frames.len() != 2 {
            bail!(
                "expected 2 local engine request frames, got {}",
                frames.len()
            );
        }

        match frames[0].as_ref() {
            ty if ty == EngineCoreRequestType::Add.to_frame().as_ref() => {
                let request: EngineCoreRequest =
                    vllm_engine_core_client::protocol::decode_msgpack(&frames[1])?;
                self.start_request(request, event_tx, output_tx, streams)
            }
            ty if ty == EngineCoreRequestType::Abort.to_frame().as_ref() => {
                let request_ids: Vec<String> =
                    vllm_engine_core_client::protocol::decode_msgpack(&frames[1])?;
                for request_id in request_ids {
                    // Drop our state first, then set the abort reason (so the
                    // scheduler's next emit fails and retires the request). The
                    // `Release` store orders the `streams.remove` before the
                    // reason the scheduler reads with `Acquire`; any token
                    // already in flight for this id is discarded by the demux
                    // when it finds no stream entry. An abort before the first
                    // token output reached the frontend is a disconnect; after
                    // that first token it is stream cancellation. Scheduled
                    // metadata can flush with the first engine output but is
                    // not enough to prove a client-visible token.
                    if let Some(state) = streams.remove(request_id.as_str()) {
                        let reason = if state.has_emitted_tokens {
                            RequestAbortReason::Cancelled
                        } else {
                            RequestAbortReason::Disconnected
                        };
                        state.abort(reason);
                    }
                }
                Ok(())
            }
            ty if ty == EngineCoreRequestType::Utility.to_frame().as_ref() => {
                let (_client_index, call_id, method_name, _args): (
                    u32,
                    UtilityCallId,
                    String,
                    rmpv::Value,
                ) = rmp_serde::from_slice(&frames[1])?;
                send_utility_response(self.engine_index, output_tx, call_id, &method_name)
            }
            other => bail!("unsupported local engine request type frame: {other:?}"),
        }
    }

    fn start_request(
        &self,
        request: EngineCoreRequest,
        event_tx: &mpsc::UnboundedSender<(RequestTag, TokenEvent)>,
        output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
        streams: &mut HashMap<RequestTag, RequestStreamState>,
    ) -> Result<()> {
        let EngineCoreRequest {
            request_id,
            prompt_token_ids,
            sampling_params,
            ..
        } = request;
        let Some(prompt_tokens) = prompt_token_ids else {
            warn!("request {request_id} dropped: missing prompt_token_ids");
            send_terminal_output(
                self.engine_index,
                output_tx,
                request_id,
                EngineCoreFinishReason::Error,
                None,
                None,
                None,
            )?;
            return Ok(());
        };
        let Some(sampling_params) = sampling_params else {
            warn!("request {request_id} dropped: missing sampling_params");
            send_terminal_output(
                self.engine_index,
                output_tx,
                request_id,
                EngineCoreFinishReason::Error,
                None,
                None,
                None,
            )?;
            return Ok(());
        };

        if let Some(unsupported) = crate::vllm::wire::unsupported_request_params(&sampling_params) {
            warn!("request {request_id} rejected: {unsupported}");
            send_terminal_output(
                self.engine_index,
                output_tx,
                request_id,
                EngineCoreFinishReason::Error,
                None,
                None,
                None,
            )?;
            return Ok(());
        }
        let lora_adapter = match lora_adapter_from_sampling_params(&sampling_params) {
            Ok(adapter) => adapter,
            Err(error) => {
                let message = error.to_string();
                warn!("request {request_id} rejected: {message}");
                send_terminal_output(
                    self.engine_index,
                    output_tx,
                    request_id,
                    EngineCoreFinishReason::Error,
                    Some(StopReason::Text(message)),
                    None,
                    None,
                )?;
                return Ok(());
            }
        };
        let kv_transfer_params = sampling_params
            .extra_args
            .as_ref()
            .and_then(|args| args.get("kv_transfer_params"))
            .cloned();
        let stop_sentinel_id = stop_sentinel_id(
            sampling_params.eos_token_id,
            &sampling_params.stop_token_ids,
        );

        let tag: RequestTag = Arc::from(request_id.as_str());
        let abort_reason = Arc::new(AtomicU8::new(RequestAbortReason::None as u8));
        let token_tx = TokenSink::new(tag.clone(), event_tx.clone(), Arc::clone(&abort_reason));
        // Open the request's root span here, before submit, so its context can
        // travel into the scheduler as the parent of the queue/prefill/decode
        // spans. When tracing is off we must build a *noop* span rather than a
        // real one: fastrace is compiled with `enable`, so `Span::root` with a
        // sampled context always allocates and reports. Gating on `is_enabled()`
        // keeps the default (tracing-off) path free of per-request span work,
        // and `from_span` on a noop span yields `None` so the scheduler skips
        // its span work too.
        let trace_root = if crate::tracing_state::is_enabled() {
            Span::root("request", SpanContext::random())
                .with_property(|| ("request_id", tag.to_string()))
        } else {
            Span::noop()
        };
        let trace_parent = SpanContext::from_span(&trace_root);
        self.handle
            .submit(GenerateRequest {
                trace_parent,
                request_id: Some(request_id),
                queued_at_unix_s: Some(request.arrival_time),
                data_parallel_rank: Some(self.engine_index as usize),
                prompt_tokens,
                params: convert_sampling(&sampling_params),
                max_tokens: sampling_params.max_tokens as usize,
                lora_adapter,
                kv_transfer_params,
                token_tx,
                logprobs: requested_logprobs(&sampling_params),
                echo: false,
            })
            .context("failed to submit request to scheduler")?;

        streams.insert(
            tag,
            RequestStreamState::new(abort_reason, trace_root, stop_sentinel_id),
        );
        Ok(())
    }
}

/// Per-request demux state held by the bridge loop, keyed by [`RequestTag`].
/// Replaces the former per-request task's locals; `first_token_*` flush onto
/// the request's first output, `abort_reason` is the state the scheduler's
/// [`TokenSink`] checks so an abort retires the request without closing the
/// shared channel.
struct RequestStreamState {
    first_token_events: Option<Vec<EngineCoreEvent>>,
    first_token_prefill_stats: Option<PrefillStats>,
    /// P/D handoff metadata can arrive in a burst with no token or terminal
    /// event, so retain it until the next output carries it to the router.
    kv_transfer_params: Option<serde_json::Value>,
    /// The vLLM text decoder removes the final token from a stop-finished
    /// output. Keep an EOS or explicit stop token as that removable sentinel.
    stop_sentinel_id: Option<u32>,
    abort_reason: Arc<AtomicU8>,
    has_emitted_tokens: bool,
    /// Request-lifetime root span (submit → finish). The scheduler opens
    /// queue/prefill/decode as children of this via the `SpanContext` passed in
    /// `GenerateRequest.trace_parent`, so the host-side phase breakdown is timed
    /// where the work actually happens (inside the scheduler), not inferred from
    /// event arrival at this downstream demux. `Span::noop()` when tracing is
    /// off. Held only for its `Drop`: dropping this state (on request completion
    /// or abort) ends the root span and closes the trace.
    #[allow(dead_code)]
    trace_root: Span,
}

impl RequestStreamState {
    fn new(abort_reason: Arc<AtomicU8>, trace_root: Span, stop_sentinel_id: Option<u32>) -> Self {
        Self {
            first_token_events: None,
            first_token_prefill_stats: None,
            kv_transfer_params: None,
            stop_sentinel_id,
            abort_reason,
            has_emitted_tokens: false,
            trace_root,
        }
    }

    fn abort(&self, reason: RequestAbortReason) {
        reason.store(&self.abort_reason);
    }
}

/// Drain the ready burst from the shared token channel (the `first` event plus
/// everything already queued), bucket it per request preserving event order,
/// fold each request's events into at most one `EngineCoreOutput`, and ship the
/// whole burst as a single `EngineCoreOutputs` — collapsing what used to be N
/// per-request ZMQ messages per step into one.
fn dispatch_burst(
    engine_index: u32,
    first: (RequestTag, TokenEvent),
    event_rx: &mut TokenStreamReceiver,
    streams: &mut HashMap<RequestTag, RequestStreamState>,
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
) -> Result<()> {
    // Bucket the burst by request, keeping first-seen order so outputs are
    // deterministic and each request's events stay in arrival order.
    let mut order: Vec<RequestTag> = Vec::new();
    let mut buckets: HashMap<RequestTag, Vec<TokenEvent>> = HashMap::new();
    let mut bucket = |tag: RequestTag, event: TokenEvent| {
        if let Some(events) = buckets.get_mut(&tag) {
            events.push(event);
        } else {
            order.push(Arc::clone(&tag));
            buckets.insert(tag, vec![event]);
        }
    };
    bucket(first.0, first.1);
    while let Ok((tag, event)) = event_rx.try_recv() {
        bucket(tag, event);
    }

    let mut outputs: Vec<EngineCoreOutput> = Vec::with_capacity(order.len());
    let mut finished_requests: BTreeSet<String> = BTreeSet::new();
    for tag in order {
        let events = buckets.remove(&tag).expect("bucket for ordered tag");
        // No stream entry means the request was aborted or already finished;
        // its late events are dropped.
        let Some(state) = streams.get_mut(&tag) else {
            continue;
        };
        let (output, terminated) = reduce_request(&tag, state, events);
        if let Some(output) = output {
            if !output.new_token_ids.is_empty() {
                state.has_emitted_tokens = true;
            }
            outputs.push(output);
        }
        if terminated {
            // Dropping the state ends both spans, closing the request trace.
            streams.remove(&tag);
            finished_requests.insert(tag.to_string());
        }
    }

    if outputs.is_empty() {
        return Ok(());
    }
    send_outputs(
        output_tx,
        RequestBatchOutputs {
            engine_index,
            outputs,
            finished_requests: (!finished_requests.is_empty()).then_some(finished_requests),
            timestamp: now_secs_f64(),
            ..Default::default()
        }
        .into(),
    )
}

/// Fold one request's events from a single burst into at most one output.
/// Tokens coalesce, and a trailing terminal rides the same output carrying its
/// finish reason; `first_token_events`/`prefill_stats` flush onto whichever
/// output goes first. A lone `Scheduled` (no token, no terminal) yields no
/// output — its metadata waits in `state` for the first real output. Returns
/// `(output, terminated)`.
fn reduce_request(
    request_id: &str,
    state: &mut RequestStreamState,
    events: Vec<TokenEvent>,
) -> (Option<EngineCoreOutput>, bool) {
    let mut token_ids: Vec<u32> = Vec::new();
    let mut positions: Vec<PositionLogprobs> = Vec::new();
    let mut has_logprobs = false;
    let mut finish_reason: Option<EngineCoreFinishReason> = None;
    let mut stop_reason: Option<StopReason> = None;
    let mut terminated = false;

    for event in events {
        match event {
            TokenEvent::Scheduled {
                queued_at_unix_s,
                scheduled_at_unix_s,
                prompt_tokens,
                cached_tokens,
            } => {
                state.first_token_events = Some(vec![
                    EngineCoreEvent {
                        r#type: EngineCoreEventType::Queued,
                        timestamp: queued_at_unix_s,
                    },
                    EngineCoreEvent {
                        r#type: EngineCoreEventType::Scheduled,
                        timestamp: scheduled_at_unix_s,
                    },
                ]);
                // Upstream invariant: computed (actual prefill work) +
                // cached (prefix-cache hit) == prompt; double-counting skews
                // the per-source prompt token metrics.
                state.first_token_prefill_stats = Some(PrefillStats {
                    num_prompt_tokens: prompt_tokens as u32,
                    num_computed_tokens: prompt_tokens.saturating_sub(cached_tokens) as u32,
                    num_cached_tokens: cached_tokens as u32,
                    num_local_cached_tokens: cached_tokens as u32,
                    num_external_cached_tokens: 0,
                });
            }
            TokenEvent::Token { id, logprob } => {
                token_ids.push(id);
                if let Some(position) = to_wire_position_logprobs(id, logprob) {
                    has_logprobs = true;
                    positions.push(position);
                } else {
                    positions.push(PositionLogprobs {
                        entries: Vec::new(),
                    });
                }
            }
            TokenEvent::PromptTokens { .. } => {
                // Prompt logprobs are intentionally deferred for this bridge.
            }
            TokenEvent::KvTransfer { params } => {
                state.kv_transfer_params = Some(params);
            }
            TokenEvent::Finished {
                finish_reason: fr, ..
            } => {
                // PegaInfer suppresses EOS before emitting TokenEvents, while
                // vLLM's text decoder expects the terminal Stop output to
                // contain EOS and unconditionally removes its final token.
                // Without this protocol token, a speculative step that commits
                // [visible token, EOS] loses the visible token at the frontend.
                if fr == FinishReason::Stop
                    && let Some(stop_sentinel_id) = state.stop_sentinel_id
                {
                    token_ids.push(stop_sentinel_id);
                    positions.push(PositionLogprobs {
                        entries: Vec::new(),
                    });
                }
                finish_reason = Some(convert_finish_reason(fr));
                terminated = true;
            }
            TokenEvent::Error { message, .. } => {
                warn!("request {request_id} failed: {message}");
                finish_reason = Some(EngineCoreFinishReason::Error);
                stop_reason = Some(StopReason::Text(message));
                terminated = true;
            }
            TokenEvent::Rejected { message, .. } => {
                // Rejected means the request could not be admitted, not that it
                // completed cleanly.
                warn!("request {request_id} rejected: {message}");
                finish_reason = Some(EngineCoreFinishReason::Error);
                stop_reason = Some(StopReason::Text(message));
                terminated = true;
            }
        }
    }

    if token_ids.is_empty() && !terminated {
        return (None, false);
    }

    let logprobs = has_logprobs.then_some(MaybeWireLogprobs::Direct(Logprobs { positions }));
    let mut output = engine_output(
        request_id.to_string(),
        token_ids,
        logprobs,
        finish_reason,
        stop_reason,
        state.first_token_events.take(),
        state.first_token_prefill_stats.take(),
    );
    output.kv_transfer_params = state.kv_transfer_params.take();
    (Some(output), terminated)
}

fn stop_sentinel_id(eos_token_id: Option<u32>, stop_token_ids: &[u32]) -> Option<u32> {
    eos_token_id.or_else(|| stop_token_ids.first().copied())
}

/// vLLM `SchedulerStats` view of a load snapshot — what the frontend's
/// Prometheus gauges (`scheduler_running`, `scheduler_waiting`,
/// `kv_cache_usage`) and DP load balancer consume.
pub(crate) fn scheduler_stats_from(snapshot: &SchedulerMetrics) -> SchedulerStats {
    SchedulerStats {
        num_running_reqs: snapshot.num_running_reqs,
        num_waiting_reqs: snapshot.num_waiting_reqs,
        kv_cache_usage: if snapshot.kv_total_blocks == 0 {
            0.0
        } else {
            snapshot.kv_used_blocks as f64 / snapshot.kv_total_blocks as f64
        },
        ..SchedulerStats::default()
    }
}

/// Per-interval spec-decode delta from two cumulative snapshots, in the wire
/// shape the frontend increments its `vllm:spec_decode_*_total` counters by (see
/// [`SpecDecodeCounters`] for why the transport carries totals and the wire
/// carries deltas).
fn spec_decode_delta(last: &SpecDecodeCounters, cur: &SpecDecodeCounters) -> SpecDecodingStats {
    let num_accepted_tokens_per_pos = cur
        .num_accepted_tokens_per_pos
        .iter()
        .zip(&last.num_accepted_tokens_per_pos)
        .map(|(cur_pos, last_pos)| cur_pos.saturating_sub(*last_pos))
        .take(cur.num_spec_tokens as usize)
        .collect();
    SpecDecodingStats {
        num_spec_tokens: cur.num_spec_tokens,
        num_drafts: cur.num_drafts.saturating_sub(last.num_drafts),
        num_draft_tokens: cur.num_draft_tokens.saturating_sub(last.num_draft_tokens),
        num_accepted_tokens: cur
            .num_accepted_tokens
            .saturating_sub(last.num_accepted_tokens),
        num_accepted_tokens_per_pos,
    }
}

/// Forward every scheduler load snapshot as a stats-only output batch; the
/// frontend records it into the shared Prometheus registry. Sends the current
/// snapshot up front so the gauges initialize before the first step, then one
/// message per watch change until shutdown. Losing either output or scheduler
/// load feed is fatal because keeping that engine registered would leave stale
/// metrics and a request-routing black hole.
async fn publish_scheduler_stats(
    engine_index: u32,
    mut load_rx: watch::Receiver<SchedulerMetrics>,
    output_tx: mpsc::UnboundedSender<EngineCoreOutputs>,
    shutdown: CancellationToken,
) -> Result<()> {
    let mut last_spec = SpecDecodeCounters::default();
    loop {
        let snapshot = *load_rx.borrow_and_update();
        let spec_decoding_stats = if let Some(cur) = &snapshot.spec_decode {
            let delta = spec_decode_delta(&last_spec, cur);
            last_spec = *cur;
            // A zero-draft interval would divide by zero in the frontend's
            // acceptance-rate log, and every counter moves only inside
            // `observe_draft`, so dropping it loses nothing.
            (delta.num_drafts > 0).then_some(delta)
        } else {
            // Reset so a drafter loaded later diffs from zero
            last_spec = SpecDecodeCounters::default();
            None
        };
        let mut stats = scheduler_stats_from(&snapshot);
        stats.spec_decoding_stats = spec_decoding_stats;
        let outputs = RequestBatchOutputs {
            engine_index,
            scheduler_stats: Some(Box::new(stats)),
            timestamp: now_secs_f64(),
            ..Default::default()
        }
        .into();
        send_outputs(&output_tx, outputs).context("failed to publish scheduler stats")?;
        tokio::select! {
            biased;
            () = shutdown.cancelled() => return Ok(()),
            changed = load_rx.changed() => {
                changed.context("scheduler load feed closed")?;
            }
        }
    }
}

/// The connected transport of one engine bridge: the vLLM EngineCore
/// handshake done, the output pump running, the stats publisher attached.
/// Shared by the legacy per-token bridge and the stepped bridge — everything
/// north of the engine data plane is identical between them.
struct BridgeLink {
    input: DealerSocket,
    output_tx: mpsc::UnboundedSender<EngineCoreOutputs>,
    child_tasks: tokio::task::JoinSet<(&'static str, Result<()>)>,
}

#[allow(clippy::too_many_arguments)]
async fn connect_link(
    input_address: &str,
    output_address: &str,
    engine_index: u32,
    data_parallel_size: u32,
    max_model_len: u32,
    kv_capacity: Option<crate::engine::KvCapacity>,
    metrics_watch: Option<watch::Receiver<SchedulerMetrics>>,
    shutdown: &CancellationToken,
) -> Result<BridgeLink> {
    wait_for_ipc_endpoint(input_address, shutdown).await?;
    wait_for_ipc_endpoint(output_address, shutdown).await?;

    let engine_id = EngineId::from_engine_index(engine_index);
    let mut socket_options = SocketOptions::default();
    socket_options.peer_identity(PeerIdentity::try_from(engine_id)?);

    let mut input = DealerSocket::with_options(socket_options);
    input
        .connect(input_address)
        .await
        .with_context(|| format!("failed to connect local engine input {input_address}"))?;

    let (num_gpu_blocks, block_size, kv_cache_size_tokens, kv_cache_max_concurrency) =
        match kv_capacity {
            Some(c) => {
                // vLLM single-group concurrency: blocks / ceil(max_len / block_size).
                let blocks_per_req = u64::from(max_model_len).div_ceil(c.block_size as u64);
                (
                    c.total_blocks as u64,
                    c.block_size as u64,
                    Some(c.total_tokens() as u64),
                    Some(c.total_blocks as f64 / blocks_per_req as f64),
                )
            }
            None => (0, 16, None, None),
        };
    let ready = EngineCoreReadyResponse {
        max_model_len: u64::from(max_model_len),
        num_gpu_blocks,
        block_size,
        dp_stats_address: None,
        dtype: ModelDtype::BFloat16,
        vllm_version: "pegainfer-local-bridge".to_string(),
        world_size: 1,
        data_parallel_size: u64::from(data_parallel_size),
        kv_cache_size_tokens,
        kv_cache_max_concurrency,
    };
    info!(
        "local engine {engine_index} KV capacity: {kv_capacity:?} -> \
         kv_cache_size_tokens={kv_cache_size_tokens:?} \
         kv_cache_max_concurrency={kv_cache_max_concurrency:?}"
    );
    input
        .send(ZmqMessage::from(encode_msgpack(&ready)?))
        .await
        .context("failed to send local engine ready response")?;

    let mut output = PushSocket::new();
    output
        .connect(output_address)
        .await
        .with_context(|| format!("failed to connect local engine output {output_address}"))?;

    let (output_tx, output_rx) = mpsc::unbounded_channel();
    let mut child_tasks = tokio::task::JoinSet::new();
    child_tasks.spawn(async move { ("output sender", output_loop(output, output_rx).await) });

    // Legacy handle engines republish load snapshots as stats-only output
    // batches: their scheduler loops park when idle, so the watch cadence is
    // bounded. The stepped bridge passes no watch here — its driver busy-polls
    // (every spin would become a message), so it reads the watch at send time
    // and stamps stats onto the batches it already emits instead.
    if let Some(load_rx) = metrics_watch {
        let stats_output_tx = output_tx.clone();
        let stats_shutdown = shutdown.clone();
        child_tasks.spawn(async move {
            (
                "scheduler stats publisher",
                publish_scheduler_stats(engine_index, load_rx, stats_output_tx, stats_shutdown)
                    .await,
            )
        });
    }

    Ok(BridgeLink {
        input,
        output_tx,
        child_tasks,
    })
}

async fn output_loop(
    mut output: PushSocket,
    mut output_rx: mpsc::UnboundedReceiver<EngineCoreOutputs>,
) -> Result<()> {
    while let Some(outputs) = output_rx.recv().await {
        output
            .send(ZmqMessage::from(encode_msgpack(&outputs)?))
            .await
            .context("failed to send local engine output")?;
    }
    Ok(())
}

fn send_terminal_output(
    engine_index: u32,
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    request_id: String,
    finish_reason: EngineCoreFinishReason,
    stop_reason: Option<StopReason>,
    events: Option<Vec<EngineCoreEvent>>,
    prefill_stats: Option<PrefillStats>,
) -> Result<()> {
    send_outputs(
        output_tx,
        RequestBatchOutputs {
            engine_index,
            outputs: vec![engine_output(
                request_id.clone(),
                Vec::new(),
                None,
                Some(finish_reason),
                stop_reason,
                events,
                prefill_stats,
            )],
            finished_requests: Some(BTreeSet::from([request_id])),
            timestamp: now_secs_f64(),
            ..Default::default()
        }
        .into(),
    )
}

fn send_utility_response(
    engine_index: u32,
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    call_id: UtilityCallId,
    method_name: &str,
) -> Result<()> {
    let result = match method_name {
        "is_sleeping" | "is_paused" | "reset_prefix_cache" => rmpv::ext::to_value(false)?,
        "sleep" | "wake_up" | "reset_mm_cache" | "reset_encoder_cache" | "collective_rpc" => {
            rmpv::Value::Nil
        }
        _ => rmpv::Value::Nil,
    };

    send_outputs(
        output_tx,
        UtilityCallOutput {
            engine_index,
            timestamp: now_secs_f64(),
            output: UtilityOutput {
                call_id,
                failure_message: None,
                result: Some(UtilityResultEnvelope::without_type_info(result)),
            },
        }
        .into(),
    )
}

fn send_outputs(
    output_tx: &mpsc::UnboundedSender<EngineCoreOutputs>,
    outputs: EngineCoreOutputs,
) -> Result<()> {
    output_tx
        .send(outputs)
        .map_err(|_| anyhow::anyhow!("local engine output channel closed"))
}

fn engine_output(
    request_id: String,
    new_token_ids: Vec<u32>,
    new_logprobs: Option<MaybeWireLogprobs>,
    finish_reason: Option<EngineCoreFinishReason>,
    stop_reason: Option<StopReason>,
    events: Option<Vec<EngineCoreEvent>>,
    prefill_stats: Option<PrefillStats>,
) -> EngineCoreOutput {
    EngineCoreOutput {
        request_id,
        new_token_ids,
        new_logprobs,
        new_prompt_logprobs_tensors: None,
        pooling_output: None,
        finish_reason,
        stop_reason,
        events,
        kv_transfer_params: None,
        ec_transfer_params: None,
        trace_headers: None,
        prefill_stats,
        routed_experts: None,
        num_nans_in_logits: 0,
    }
}

fn now_secs_f64() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_secs_f64()
}

pub(crate) fn local_ipc_namespace() -> Result<PathBuf> {
    let base_dir =
        std::env::var_os("PEGAINFER_IPC_DIR").map_or_else(|| PathBuf::from("/tmp"), PathBuf::from);
    let uuid = uuid::Uuid::new_v4().to_string();
    let path = base_dir.join(format!("pgi-{}-{}", std::process::id(), &uuid[..8]));
    std::fs::create_dir_all(&path)
        .with_context(|| format!("failed to create IPC namespace {}", path.display()))?;
    Ok(path)
}

pub(crate) fn ipc_endpoint(namespace: &Path, name: &str) -> String {
    format!("ipc://{}", namespace.join(name).to_string_lossy())
}

async fn wait_for_ipc_endpoint(address: &str, shutdown: &CancellationToken) -> Result<()> {
    let Some(path) = address.strip_prefix("ipc://") else {
        return Ok(());
    };
    let path = Path::new(path);
    loop {
        if path.exists() {
            return Ok(());
        }
        tokio::select! {
            () = shutdown.cancelled() => bail!("shutdown before IPC endpoint appeared"),
            () = tokio::time::sleep(Duration::from_millis(20)) => {}
        }
    }
}

mod stepped;

pub(crate) use stepped::SteppedEngineBridge;

#[cfg(test)]
mod tests;
