//! The vLLM EngineCore bridge for step-driven engines.
//!
//! Where the legacy bridge re-folds a per-token event stream into burst
//! outputs, this one translates each [`StepOutputs`] message 1:1 into one
//! `EngineCoreOutputs`: the scheduler already batched the step, and each
//! [`RequestUpdate`] is already the per-request fold. Wall-clock timestamps
//! for the wire are rendered from the contract's monotonic `Instant`s against
//! a per-bridge anchor.

use std::collections::BTreeSet;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use fastrace::Span;
use fastrace::collector::SpanContext;
use log::info;
use log::warn;
use tokio_util::sync::CancellationToken;
use vllm_engine_core_client::protocol::logprobs::Logprobs;
use vllm_engine_core_client::protocol::logprobs::MaybeWireLogprobs;
use vllm_engine_core_client::protocol::logprobs::PositionLogprobs;
use vllm_engine_core_client::protocol::output::EngineCoreEvent;
use vllm_engine_core_client::protocol::output::EngineCoreEventType;
use vllm_engine_core_client::protocol::output::EngineCoreFinishReason;
use vllm_engine_core_client::protocol::output::EngineCoreOutput;
use vllm_engine_core_client::protocol::output::RequestBatchOutputs;
use vllm_engine_core_client::protocol::output::StopReason;
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::request::EngineCoreRequestType;
use vllm_engine_core_client::protocol::stats::PrefillStats;
use vllm_engine_core_client::protocol::stats::SchedulerStats;
use vllm_engine_core_client::protocol::utility::UtilityCallId;
use zeromq::ZmqMessage;
use zeromq::prelude::SocketRecv;

use super::BridgeLink;
use super::SpecDecodeTracker;
use super::connect_link;
use super::engine_output;
use super::now_secs_f64;
use super::scheduler_stats_from;
use super::send_outputs;
use super::send_terminal_output;
use super::send_utility_response;
use super::stop_sentinel_id;
use crate::engine::FinishReason;
use crate::engine::KvCapacity;
use crate::engine::Request;
use crate::engine::RequestControl;
use crate::engine::RequestId;
use crate::engine::RequestUpdate;
use crate::engine::SchedulerHandle;
use crate::engine::StepOutputs;
use crate::engine::Terminal;
use crate::vllm::wire::convert_finish_reason;
use crate::vllm::wire::convert_sampling;
use crate::vllm::wire::lora_adapter_from_sampling_params;
use crate::vllm::wire::requested_logprobs;
use crate::vllm::wire::to_wire_position_logprobs;

pub(crate) struct SteppedEngineBridge {
    pub(crate) input_address: String,
    pub(crate) output_address: String,
    pub(crate) scheduler: SchedulerHandle,
    pub(crate) kv_capacity: Option<KvCapacity>,
    pub(crate) max_model_len: u32,
    pub(crate) engine_index: u32,
    pub(crate) data_parallel_size: u32,
}

impl SteppedEngineBridge {
    pub(crate) async fn run(mut self, shutdown: CancellationToken) -> Result<()> {
        let mut steps = self
            .scheduler
            .take_steps()
            .context("partition step stream already taken")?;
        let mut spec = SpecDecodeTracker::default();
        // Stats are pull-at-send: no push task, the load cell is read when a
        // batch goes out (and once here, so the frontend's gauges initialize
        // before any traffic). An idle engine publishes nothing.
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
            self.kv_capacity,
            None,
            &shutdown,
        )
        .await?;
        send_outputs(
            &output_tx,
            RequestBatchOutputs {
                engine_index: self.engine_index,
                scheduler_stats: Some(Box::new(self.stats(&mut spec))),
                timestamp: now_secs_f64(),
                ..Default::default()
            }
            .into(),
        )?;

        let anchor = UnixAnchor::now();
        let mut streams: HashMap<RequestId, SteppedStream> = HashMap::new();
        let mut names: HashMap<String, RequestId> = HashMap::new();

        info!(
            "local vLLM engine {} stepped bridge connected: input={}, output={}, max_model_len={}",
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
                Some(step) = steps.recv() => {
                    if let Err(error) = self.dispatch_step(
                        step,
                        &anchor,
                        &mut streams,
                        &mut names,
                        &mut spec,
                        &output_tx,
                    ) {
                        break Err(error).context("failed to dispatch local engine step");
                    }
                }
                recv = input.recv() => {
                    let message = match recv.context("failed to receive local engine request") {
                        Ok(message) => message,
                        Err(error) => break Err(error),
                    };
                    if let Err(error) = self.handle_message(
                        message,
                        &mut streams,
                        &mut names,
                        &output_tx,
                    ) {
                        break Err(error).context("failed to handle local engine request");
                    }
                }
            }
        };

        // Flip every in-flight request's abort flag so the scheduler retires
        // them on its next touch; dropping the partition handle afterwards
        // disconnects the submission channel and lets the driver drain out.
        for state in streams.values() {
            state.control.abort();
        }
        drop(output_tx);
        child_tasks.abort_all();
        while child_tasks.join_next().await.is_some() {}

        run_result
    }

    /// Stats for an outgoing batch; the spec delta runs from the last batch
    /// stamped, not the last step run.
    fn stats(&self, spec: &mut SpecDecodeTracker) -> SchedulerStats {
        let snapshot = self.scheduler.metrics();
        let mut stats = scheduler_stats_from(&snapshot);
        stats.spec_decoding_stats = spec.interval(&snapshot);
        stats
    }

    fn dispatch_step(
        &self,
        step: StepOutputs,
        anchor: &UnixAnchor,
        streams: &mut HashMap<RequestId, SteppedStream>,
        names: &mut HashMap<String, RequestId>,
        spec: &mut SpecDecodeTracker,
        output_tx: &tokio::sync::mpsc::UnboundedSender<
            vllm_engine_core_client::protocol::output::EngineCoreOutputs,
        >,
    ) -> Result<()> {
        let mut outputs: Vec<EngineCoreOutput> = Vec::with_capacity(step.updates.len());
        let mut finished_requests: BTreeSet<String> = BTreeSet::new();
        for update in step.updates {
            // No stream entry means the request was aborted or already
            // finished; late updates (including retire-window drop bombs) are
            // dropped.
            let Some(state) = streams.get_mut(&update.id) else {
                continue;
            };
            let id = update.id;
            let (output, terminated) = reduce_update(state, update, anchor);
            if let Some(output) = output {
                outputs.push(output);
            }
            if terminated {
                // Dropping the state ends the root span, closing the trace.
                let state = streams.remove(&id).expect("terminated stream present");
                names.remove(&state.request_id);
                finished_requests.insert(state.request_id);
            }
        }

        if outputs.is_empty() {
            // A drafted step with no batch to ride would strand its increment
            // until the next batch, which may never come.
            let stats = self.stats(spec);
            if stats.spec_decoding_stats.is_some() {
                send_outputs(
                    output_tx,
                    RequestBatchOutputs {
                        engine_index: self.engine_index,
                        scheduler_stats: Some(Box::new(stats)),
                        timestamp: now_secs_f64(),
                        ..Default::default()
                    }
                    .into(),
                )?;
            }
            return Ok(());
        }
        // The cell already holds this step's snapshot (the driver publishes
        // load before committing the step), so the batch carries stats that
        // match its own tokens — a finishing batch reports the drained state
        // and the gauges settle instead of freezing at the last busy value.
        send_outputs(
            output_tx,
            RequestBatchOutputs {
                engine_index: self.engine_index,
                outputs,
                finished_requests: (!finished_requests.is_empty()).then_some(finished_requests),
                scheduler_stats: Some(Box::new(self.stats(spec))),
                timestamp: now_secs_f64(),
            }
            .into(),
        )
    }

    fn handle_message(
        &self,
        message: ZmqMessage,
        streams: &mut HashMap<RequestId, SteppedStream>,
        names: &mut HashMap<String, RequestId>,
        output_tx: &tokio::sync::mpsc::UnboundedSender<
            vllm_engine_core_client::protocol::output::EngineCoreOutputs,
        >,
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
                self.start_request(request, streams, names, output_tx)
            }
            ty if ty == EngineCoreRequestType::Abort.to_frame().as_ref() => {
                let request_ids: Vec<String> =
                    vllm_engine_core_client::protocol::decode_msgpack(&frames[1])?;
                for request_id in request_ids {
                    // Drop our state first, then flip the abort flag (whose
                    // `Release` store orders after the removal); the
                    // scheduler's next touch retires the request, and any
                    // update already in flight finds no stream entry.
                    let Some(id) = names.remove(&request_id) else {
                        continue;
                    };
                    if let Some(state) = streams.remove(&id) {
                        state.control.abort();
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
        streams: &mut HashMap<RequestId, SteppedStream>,
        names: &mut HashMap<String, RequestId>,
        output_tx: &tokio::sync::mpsc::UnboundedSender<
            vllm_engine_core_client::protocol::output::EngineCoreOutputs,
        >,
    ) -> Result<()> {
        let EngineCoreRequest {
            request_id,
            prompt_token_ids,
            sampling_params,
            ..
        } = request;
        let Some(prompt_tokens) = prompt_token_ids else {
            warn!("request {request_id} dropped: missing prompt_token_ids");
            return send_terminal_output(
                self.engine_index,
                output_tx,
                request_id,
                EngineCoreFinishReason::Error,
                None,
                None,
                None,
            );
        };
        let Some(sampling_params) = sampling_params else {
            warn!("request {request_id} dropped: missing sampling_params");
            return send_terminal_output(
                self.engine_index,
                output_tx,
                request_id,
                EngineCoreFinishReason::Error,
                None,
                None,
                None,
            );
        };

        if let Some(unsupported) = crate::vllm::wire::unsupported_request_params(&sampling_params) {
            warn!("request {request_id} rejected: {unsupported}");
            return send_terminal_output(
                self.engine_index,
                output_tx,
                request_id,
                EngineCoreFinishReason::Error,
                None,
                None,
                None,
            );
        }
        let lora_adapter = match lora_adapter_from_sampling_params(&sampling_params) {
            Ok(adapter) => adapter,
            Err(error) => {
                let message = error.to_string();
                warn!("request {request_id} rejected: {message}");
                return send_terminal_output(
                    self.engine_index,
                    output_tx,
                    request_id,
                    EngineCoreFinishReason::Error,
                    Some(StopReason::Text(message)),
                    None,
                    None,
                );
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

        // Open the request's root span before submit so its context travels
        // into the scheduler as the parent of the queue/prefill/decode spans;
        // noop (no allocation, `from_span` yields `None`) when tracing is off.
        let trace_root = if crate::tracing_state::is_enabled() {
            Span::root("request", SpanContext::random())
                .with_property(|| ("request_id", request_id.clone()))
        } else {
            Span::noop()
        };
        let trace_parent = SpanContext::from_span(&trace_root);
        let control = self.scheduler.submit(Request {
            prompt_tokens,
            params: convert_sampling(&sampling_params),
            max_tokens: sampling_params.max_tokens as usize,
            lora_adapter,
            kv_transfer_params,
            logprobs: requested_logprobs(&sampling_params),
            echo: false,
            trace_parent,
            client_label: Some(Arc::from(request_id.as_str())),
        });

        names.insert(request_id.clone(), control.id());
        streams.insert(
            control.id(),
            SteppedStream::new(request_id, control, trace_root, stop_sentinel_id),
        );
        Ok(())
    }
}

/// Per-request demux state, keyed by the contract's [`RequestId`]; `names`
/// maps the vLLM wire's string request id back to it for aborts.
struct SteppedStream {
    request_id: String,
    control: RequestControl,
    /// Queued/Scheduled wire events, held until the request's first shipped
    /// output (a scheduled-only update ships nothing on its own).
    first_token_events: Option<Vec<EngineCoreEvent>>,
    /// Set by `Scheduled`; a request refused or failed while still queued
    /// never prefilled and reports no prefill stats.
    prompt_tokens: Option<usize>,
    cached_tokens: usize,
    prefill_stats_sent: bool,
    /// P/D handoff metadata can arrive in an update with no token or
    /// terminal, so retain it until the next output carries it to the router.
    kv_transfer_params: Option<serde_json::Value>,
    /// The vLLM text decoder removes the final token from a stop-finished
    /// output. Keep an EOS or explicit stop token as that removable sentinel.
    stop_sentinel_id: Option<u32>,
    /// Request-lifetime root span; held only for its `Drop`, which closes the
    /// trace when the stream state is removed.
    #[allow(dead_code)]
    trace_root: Span,
}

impl SteppedStream {
    fn new(
        request_id: String,
        control: RequestControl,
        trace_root: Span,
        stop_sentinel_id: Option<u32>,
    ) -> Self {
        Self {
            request_id,
            control,
            first_token_events: None,
            prompt_tokens: None,
            cached_tokens: 0,
            prefill_stats_sent: false,
            kv_transfer_params: None,
            stop_sentinel_id,
            trace_root,
        }
    }

    /// Prefill stats for the request's first shipped output. Built lazily so
    /// a `cached_tokens` fact arriving after `Scheduled` (chunked prefill
    /// reports it with the first chunk) still lands in the stats. Upstream
    /// invariant: computed + cached == prompt. `None` for a request that
    /// never reached `Scheduled`.
    fn take_prefill_stats(&mut self) -> Option<PrefillStats> {
        if self.prefill_stats_sent {
            return None;
        }
        let prompt_tokens = self.prompt_tokens?;
        self.prefill_stats_sent = true;
        Some(PrefillStats {
            num_prompt_tokens: prompt_tokens as u32,
            num_computed_tokens: prompt_tokens.saturating_sub(self.cached_tokens) as u32,
            num_cached_tokens: self.cached_tokens as u32,
            num_local_cached_tokens: self.cached_tokens as u32,
            num_external_cached_tokens: 0,
            num_cache_creation_tokens: 0,
        })
    }
}

/// Translate one request's step record into at most one wire output. A
/// record carrying only metadata (scheduled, cached, kv-transfer) ships
/// nothing; its facts wait in `state` for the first real output. Returns
/// `(output, terminated)`.
fn reduce_update(
    state: &mut SteppedStream,
    update: RequestUpdate,
    anchor: &UnixAnchor,
) -> (Option<EngineCoreOutput>, bool) {
    if let Some(scheduled) = update.scheduled {
        state.first_token_events = Some(vec![
            EngineCoreEvent {
                r#type: EngineCoreEventType::Queued,
                timestamp: anchor.unix(scheduled.queued_at),
            },
            EngineCoreEvent {
                r#type: EngineCoreEventType::Scheduled,
                timestamp: anchor.unix(scheduled.scheduled_at),
            },
        ]);
        state.prompt_tokens = Some(scheduled.prompt_tokens);
    }
    if let Some(cached) = update.cached_tokens {
        state.cached_tokens = cached;
    }
    if let Some(params) = update.kv_transfer {
        state.kv_transfer_params = Some(params);
    }
    // Prompt logprobs (update.prompt_echo) are dropped here: no vLLM-protocol
    // consumer requests echo yet (the HTTP layer rejects `echo` + `prompt_logprobs`
    // before submission), matching the legacy bridge. Wiring it up means mapping
    // PromptEcho into EngineCoreOutput's prompt_logprobs fields.

    let mut token_ids = update.tokens;
    let mut has_logprobs = false;
    let mut positions: Vec<PositionLogprobs> = Vec::with_capacity(token_ids.len());
    for (i, &id) in token_ids.iter().enumerate() {
        let logprob = update.logprobs.get(i).cloned().flatten();
        if let Some(position) = to_wire_position_logprobs(id, logprob) {
            has_logprobs = true;
            positions.push(position);
        } else {
            positions.push(PositionLogprobs {
                entries: Vec::new(),
            });
        }
    }

    let mut finish_reason: Option<EngineCoreFinishReason> = None;
    let mut stop_reason: Option<StopReason> = None;
    let mut terminated = false;
    match update.terminal {
        None => {}
        Some(Terminal::Finished { reason, .. }) => {
            // PegaInfer suppresses EOS before emitting tokens, while vLLM's
            // text decoder expects the terminal Stop output to contain EOS
            // and unconditionally removes its final token.
            if reason == FinishReason::Stop
                && let Some(stop_sentinel_id) = state.stop_sentinel_id
            {
                token_ids.push(stop_sentinel_id);
                positions.push(PositionLogprobs {
                    entries: Vec::new(),
                });
            }
            finish_reason = Some(convert_finish_reason(reason));
            terminated = true;
        }
        Some(Terminal::Rejected { reason, .. }) => {
            // The vLLM wire only carries a string; typed classification stops
            // here and the rendered message is what reaches the client.
            let message = reason.to_string();
            warn!("request {} rejected: {message}", state.request_id);
            finish_reason = Some(EngineCoreFinishReason::Error);
            stop_reason = Some(StopReason::Text(message));
            terminated = true;
        }
        Some(Terminal::Failed { message, .. }) => {
            warn!("request {} failed: {message}", state.request_id);
            finish_reason = Some(EngineCoreFinishReason::Error);
            stop_reason = Some(StopReason::Text(message));
            terminated = true;
        }
    }

    if token_ids.is_empty() && !terminated {
        return (None, false);
    }

    let logprobs = has_logprobs.then_some(MaybeWireLogprobs::Direct(Logprobs { positions }));
    let mut output = engine_output(
        state.request_id.clone(),
        token_ids,
        logprobs,
        finish_reason,
        stop_reason,
        state.first_token_events.take(),
        state.take_prefill_stats(),
    );
    output.kv_transfer_params = state.kv_transfer_params.take();
    (Some(output), terminated)
}

/// Renders the contract's monotonic timestamps as wall-clock floats for the
/// vLLM wire. One anchor per bridge: both readings taken together at startup,
/// so every rendered timestamp shares one clock base.
struct UnixAnchor {
    sys: f64,
    instant: Instant,
}

impl UnixAnchor {
    fn now() -> Self {
        Self {
            sys: now_secs_f64(),
            instant: Instant::now(),
        }
    }

    fn unix(&self, t: Instant) -> f64 {
        if t >= self.instant {
            self.sys + (t - self.instant).as_secs_f64()
        } else {
            self.sys - (self.instant - t).as_secs_f64()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::RejectReason;
    use crate::engine::scheduler_pair;

    fn request() -> Request {
        Request {
            prompt_tokens: vec![1, 2],
            params: crate::sampler::SamplingParams::default(),
            max_tokens: 1,
            lora_adapter: None,
            kv_transfer_params: None,
            logprobs: 0,
            echo: false,
            trace_parent: None,
            client_label: None,
        }
    }

    #[test]
    fn unsupported_rejection_is_rendered_for_vllm() {
        let (handle, _backend) = scheduler_pair();
        let control = handle.submit(request());
        let mut stream = SteppedStream::new("unsupported".into(), control, Span::noop(), None);
        let mut update = RequestUpdate::empty(stream.control.id());
        update.terminal = Some(Terminal::Rejected {
            reason: RejectReason::Unsupported {
                feature: "kv_transfer".into(),
            },
            prompt_tokens: 2,
        });

        let (output, terminated) = reduce_update(&mut stream, update, &UnixAnchor::now());
        let output = output.expect("rejection produces a wire output");
        assert!(terminated);
        assert_eq!(output.finish_reason, Some(EngineCoreFinishReason::Error));
        assert_eq!(
            output.stop_reason,
            Some(StopReason::Text(
                "this engine does not support kv_transfer".into()
            ))
        );
        assert!(
            output.prefill_stats.is_none(),
            "a request refused while queued did no prefill"
        );
    }
}
