//! The Gemma 4 engine: one owned thread with iteration-level scheduling.
//! Prefill runs whole at the step boundary; every active request then
//! advances one token per batched decode step, sharing each layer's weight
//! pass.

use std::collections::VecDeque;
use std::path::Path;

use anyhow::Context as AnyhowContext;
use anyhow::Result;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::engine::unix_now_s;
use pegainfer_sample::LogprobRequest;
use pegainfer_sample::SampleScratch;
use tokio::sync::mpsc;

use crate::forward::MULTIMODAL_PLACEHOLDER_IDS;
use crate::kv::GemmaKv;
use crate::kv::PAGE_SIZE;
use crate::kv::admit_tokens;
use crate::prefix_cache::PrefixCache;
use crate::serve::GemmaServe;
use crate::serve::StepArena;
use crate::weights::Gemma4Weights;

/// The default serving ceiling; `serving_context` tells the raising story.
const MAX_CONTEXT: usize = 8192;

/// Decode-batch ceiling: bounds the step buffers, the sampling scratch and
/// the pool budget. Admission beyond it queues at the step boundary rather
/// than rejecting.
const MAX_CONCURRENCY: usize = 16;
const ASYNC_PREFILL_ENV: &str = "PEGAINFER_ASYNC_PREFILL";
const PREFIX_CACHE_ENV: &str = "PEGAINFER_PREFIX_CACHE";
const MIX_CHUNK_TOKENS_ENV: &str = "PEGAINFER_MIX_CHUNK_TOKENS";
const MAX_CONTEXT_ENV: &str = "PEGAINFER_MAX_CONTEXT";
const DECODE_SLOTS_ENV: &str = "PEGAINFER_DECODE_SLOTS";
const MIN_CONTEXT: usize = 1024;
const MIN_CHUNK_TOKENS: usize = 64;
const CEILING_DOMAIN: usize = i32::MAX as usize;

/// A live-batch admission either shares all SMs or uses a capped Green
/// Context. Disabled is represented by `Option<LaneMode>`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LaneMode {
    Shared,
    Green(u32),
}

fn read_env(name: &str) -> Result<Option<String>> {
    normalize_env(name, std::env::var(name))
}

fn normalize_env(
    name: &str,
    value: std::result::Result<String, std::env::VarError>,
) -> Result<Option<String>> {
    match value {
        Ok(raw) => Ok(Some(raw)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => anyhow::bail!("{name} is not valid UTF-8"),
    }
}

fn async_prefill_mode() -> Result<Option<LaneMode>> {
    read_env(ASYNC_PREFILL_ENV)?.map_or(Ok(None), |raw| parse_async_prefill_mode(&raw))
}

fn parse_async_prefill_mode(raw: &str) -> Result<Option<LaneMode>> {
    let value = raw.trim().to_ascii_lowercase();
    match value.as_str() {
        "" | "0" | "false" | "off" => Ok(None),
        "shared" => Ok(Some(LaneMode::Shared)),
        other => match other
            .strip_prefix("green:")
            .and_then(|pct| pct.parse().ok())
        {
            Some(pct) if (1..=99).contains(&pct) => Ok(Some(LaneMode::Green(pct))),
            _ => anyhow::bail!(
                "{ASYNC_PREFILL_ENV}={raw:?} not recognized (off | shared | green:NN, 1..=99)"
            ),
        },
    }
}

/// The serving ceiling — prompt plus output per request, the pool budget
/// axis, and the published servable length.
fn serving_context(checkpoint_limit: usize) -> Result<usize> {
    read_env(MAX_CONTEXT_ENV)?.map_or(Ok(MAX_CONTEXT.min(checkpoint_limit)), |raw| {
        parse_serving_context(&raw, checkpoint_limit)
    })
}

fn parse_serving_context(raw: &str, checkpoint_limit: usize) -> Result<usize> {
    let limit = checkpoint_limit.min(CEILING_DOMAIN);
    match raw.trim().parse::<usize>() {
        Ok(value) if (MIN_CONTEXT..=limit).contains(&value) => Ok(value),
        _ => anyhow::bail!(
            "{MAX_CONTEXT_ENV}={raw:?} not recognized (N, {MIN_CONTEXT} <= N <= {limit}: the \
             checkpoint's limit inside the i32 metadata domain)"
        ),
    }
}

/// The decode-slot count the pools are budgeted for.
fn decode_slots() -> Result<usize> {
    read_env(DECODE_SLOTS_ENV)?.map_or(Ok(MAX_CONCURRENCY), |raw| parse_decode_slots(&raw))
}

fn parse_decode_slots(raw: &str) -> Result<usize> {
    match raw.trim().parse::<usize>() {
        Ok(value) if (1..=MAX_CONCURRENCY).contains(&value) => Ok(value),
        _ => anyhow::bail!(
            "{DECODE_SLOTS_ENV}={raw:?} not recognized (N, 1 <= N <= {MAX_CONCURRENCY})"
        ),
    }
}

/// Bounds the prompt rows computed by one chunked-walk step. The effective
/// step rounds down to whole 128-row tiles.
fn mix_chunk_tokens(max_context: usize) -> Result<Option<usize>> {
    read_env(MIX_CHUNK_TOKENS_ENV)?
        .map_or(Ok(None), |raw| parse_mix_chunk_tokens(&raw, max_context))
}

/// GEMM and attention tiles consume whole 128-row blocks, so a width that is
/// not a multiple of 128 pays for a tile it does not fill.
fn parse_mix_chunk_tokens(raw: &str, max_context: usize) -> Result<Option<usize>> {
    let value = raw.trim().to_ascii_lowercase();
    match value.as_str() {
        "" | "0" | "off" => Ok(None),
        other => match other.parse::<usize>() {
            Ok(chunk) if chunk >= MIN_CHUNK_TOKENS && chunk < max_context => {
                Ok(Some(if chunk < 128 {
                    chunk
                } else {
                    chunk - chunk % 128
                }))
            }
            _ => anyhow::bail!(
                "{MIX_CHUNK_TOKENS_ENV}={raw:?} not recognized \
                 (off | N, {MIN_CHUNK_TOKENS} <= N < {max_context})"
            ),
        },
    }
}

fn prefix_cache_cap() -> Result<Option<usize>> {
    read_env(PREFIX_CACHE_ENV)?.map_or(Ok(None), |raw| parse_prefix_cache_cap(&raw))
}

fn parse_prefix_cache_cap(raw: &str) -> Result<Option<usize>> {
    let value = raw.trim().to_ascii_lowercase();
    match value.as_str() {
        "" | "0" | "off" => Ok(None),
        other => match other.parse::<usize>() {
            Ok(cap) if cap > 0 => Ok(Some(cap)),
            _ => anyhow::bail!("{PREFIX_CACHE_ENV}={raw:?} not recognized (off | K, K > 0)"),
        },
    }
}

pub(crate) fn start(model_path: &Path, options: &EngineLoadOptions) -> Result<EngineHandle> {
    let dir = model_path
        .to_str()
        .context("model path is not valid UTF-8")?
        .to_string();
    anyhow::ensure!(
        options.device_ordinals.len() == 1,
        "gemma4 is single-device; got device_ordinals {:?}",
        options.device_ordinals
    );
    anyhow::ensure!(
        options.parallel_config.is_none(),
        "gemma4 has no parallel topology support yet"
    );
    let device = options.device_ordinals[0];
    let base_seed = options.seed;
    let graph_enabled = options.enable_cuda_graph;

    let policy = generation_policy(&dir)?;

    let (submit_tx, mut submit_rx) =
        mpsc::unbounded_channel::<pegainfer_frontend::engine::SubmittedRequest>();
    let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<usize>>();
    let join = std::thread::Builder::new()
        .name("gemma4-engine".into())
        .spawn(move || {
            let state = EngineState::load(&dir, device, policy, base_seed, graph_enabled);
            let mut state = match state {
                Ok(state) => {
                    let _ = ready_tx.send(Ok(state.max_context));
                    state
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            let mut pending: VecDeque<Submitted> = VecDeque::new();
            let mut active: Vec<Active> = Vec::new();
            let mut disconnected = false;
            'engine: loop {
                loop {
                    match submit_rx.try_recv() {
                        Ok(item) => pending.push_back(item),
                        Err(mpsc::error::TryRecvError::Empty) => break,
                        Err(mpsc::error::TryRecvError::Disconnected) => {
                            disconnected = true;
                            break;
                        }
                    }
                }
                // Join a finished overlapped prefill: poll while other work
                // exists, block on the lane when it is the only work left.
                let lane_ready = match state.lane.as_mut() {
                    Some(lane) if lane.inflight.is_some() => {
                        let lane_is_only_work = active.is_empty() && pending.is_empty();
                        if lane_is_only_work {
                            lane.drain_or_abort();
                        }
                        lane_is_only_work || lane.inflight_complete()
                    }
                    _ => false,
                };
                if lane_ready {
                    state.join_async_prefill(&mut active);
                }
                if active.is_empty()
                    && pending.is_empty()
                    && state
                        .lane
                        .as_ref()
                        .is_none_or(|lane| lane.inflight.is_none())
                {
                    if disconnected {
                        break 'engine;
                    }
                    match submit_rx.blocking_recv() {
                        Some(item) => pending.push_back(item),
                        None => break 'engine,
                    }
                }
                // A request that finishes inside its own prefill never takes
                // a slot, so slot occupancy alone does not bound this loop:
                // a backlog of one-token requests would prefill in full
                // before the next decode round and stall every stream in
                // flight for its whole length. Bound the attempts too, so a
                // burst costs the streams a bounded number of prefills per
                // token however deep it is.
                state.admit_from_queue(&mut pending, &mut active);
                if !active.is_empty() {
                    state.decode_round(&mut active);
                }
            }
        })
        .context("spawn gemma4 engine thread")?;
    let servable = ready_rx
        .recv()
        .context("gemma4 engine thread died during load")??;
    // Publishing the real ceiling is what lets the frontend refuse an
    // over-length request with its own message instead of forwarding one the
    // engine can only fail mid-stream.
    // The ceiling domain keeps the value inside i32, so this API-boundary
    // conversion cannot fail.
    let servable = u32::try_from(servable).expect("ceiling domain holds servable inside u32");
    Ok(EngineHandle::new_with_join_handle(submit_tx, join).with_servable_len(servable))
}

struct GenerationPolicy {
    eos: Vec<u32>,
    suppress: Vec<u32>,
}

fn generation_policy(dir: &str) -> Result<GenerationPolicy> {
    let path = format!("{dir}/generation_config.json");
    let json: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&path).with_context(|| format!("read {path}"))?,
    )?;
    let eos = token_ids(
        json.get("eos_token_id")
            .with_context(|| format!("{path} missing eos_token_id"))?,
    )
    .with_context(|| format!("{path} eos_token_id"))?;
    anyhow::ensure!(!eos.is_empty(), "{path} declares an empty eos_token_id");
    let mut suppress = match json.get("suppress_tokens") {
        Some(value) => token_ids(value).with_context(|| format!("{path} suppress_tokens"))?,
        None => Vec::new(),
    };
    suppress.extend(MULTIMODAL_PLACEHOLDER_IDS);
    suppress.sort_unstable();
    suppress.dedup();
    Ok(GenerationPolicy { eos, suppress })
}

impl GenerationPolicy {
    /// Whether this pick retires its row instead of being emitted. Every path
    /// that can stop a request asks here, so the rule cannot drift.
    fn stops(&self, id: u32, ignore_eos: bool) -> bool {
        !ignore_eos && self.eos.contains(&id)
    }

    /// Both sets index the vocabulary, so both are checked against it once
    /// the head is loaded: an out-of-range suppressed id would fail the first
    /// request, and an out-of-range stop id would never match at all and turn
    /// every request into a length stop.
    fn check_against_vocab(&self, vocab: usize) -> Result<()> {
        for (kind, ids) in [
            ("eos_token_id", &self.eos),
            ("effective suppression set", &self.suppress),
        ] {
            for &id in ids {
                anyhow::ensure!(
                    (id as usize) < vocab,
                    "{kind} lists {id}, outside the {vocab} the checkpoint's head spans"
                );
            }
        }
        Ok(())
    }
}

fn token_ids(value: &serde_json::Value) -> Result<Vec<u32>> {
    fn one(value: &serde_json::Value) -> Result<u32> {
        let raw = value
            .as_u64()
            .with_context(|| format!("{value} is not an unsigned integer"))?;
        u32::try_from(raw).with_context(|| format!("token id {raw} does not fit a u32"))
    }
    match value {
        serde_json::Value::Number(_) => Ok(vec![one(value)?]),
        serde_json::Value::Array(items) => items.iter().map(one).collect(),
        other => anyhow::bail!("unexpected token id shape: {other}"),
    }
}

type Submitted = pegainfer_frontend::engine::SubmittedRequest;

/// Overlapped admission: with the lane on, a prompt arriving into a live
/// decode batch prefills on its own stream while decode steps keep
/// replaying on `ctx.stream` — the admission costs the streams a slowdown
/// instead of a mixed step per prompt. `shared` lets the prefill grids
/// compete for every SM; `green:NN` pins the lane to NN% of them, which is
/// what actually protects decode ITL.
/// One in-flight overlapped prefill, parked until the lane's completion
/// event fires: the request, its KV, and the pass owning every device
/// buffer the in-flight kernels still read.
struct InflightPrefill {
    request: GenerateRequest,
    kv: GemmaKv,
    pass: crate::serve::PrefillPass,
    /// The cache entry this request resumed from, if any — its stale
    /// ancestor at capture time.
    resumed: Option<u64>,
}

/// The overlap lane: a dedicated prefill stream and a reusable completion
/// event. At most one prefill is in flight; while it runs, later arrivals
/// wait in the queue and decode keeps stepping — which is the point.
struct AsyncPrefillLane {
    stream: crate::green_ctx::PrefillLaneStream,
    event: cudarc::driver::CudaEvent,
    inflight: Option<InflightPrefill>,
}

impl AsyncPrefillLane {
    fn new(ctx: &DeviceContext, mode: LaneMode) -> Result<Self> {
        let stream = match mode {
            LaneMode::Shared => crate::green_ctx::PrefillLaneStream::shared()?,
            LaneMode::Green(pct) => {
                crate::green_ctx::PrefillLaneStream::green(ctx.device_ordinal, pct)?
            }
        };
        let event = ctx
            .ctx
            .new_event(None)
            .map_err(|e| anyhow::anyhow!("prefill completion event create failed: {e}"))?;
        Ok(Self {
            stream,
            event,
            inflight: None,
        })
    }

    /// True once the in-flight prefill's event has fired. An unexpected
    /// query error aborts: the pass's buffers may still be in use and no
    /// safe recovery exists.
    fn inflight_complete(&self) -> bool {
        debug_assert!(self.inflight.is_some());
        let query = unsafe { cudarc::driver::sys::cuEventQuery(self.event.cu_event()) };
        match query {
            cudarc::driver::sys::CUresult::CUDA_SUCCESS => true,
            cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => false,
            other => {
                log::error!("FATAL: cuEventQuery(prefill) failed ({other:?}); aborting");
                std::process::abort();
            }
        }
    }

    /// Block until the lane stream is drained; abort on failure rather
    /// than let the pass's buffers be reused under in-flight kernels.
    fn drain_or_abort(&self) {
        let sync = unsafe { cudarc::driver::sys::cuStreamSynchronize(self.stream.stream) };
        if sync != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            log::error!("FATAL: cuStreamSynchronize(prefill) failed ({sync:?}); aborting");
            std::process::abort();
        }
    }
}

impl Drop for AsyncPrefillLane {
    fn drop(&mut self) {
        self.drain_or_abort();
    }
}

/// The fail-closed request validation every admission path shares; `Err`
/// carries the rejection message. The contract refuses every capability
/// this engine does not implement — echo, a frontend-resolved prefix, P/D
/// transfer metadata, a multi-partition placement — loudly, not silently.
fn validate_request(
    request: &GenerateRequest,
    prefix_hit_tokens: usize,
    max_context: usize,
) -> Result<usize, String> {
    let prompt_tokens = request.prompt_tokens.len();
    if prompt_tokens == 0 {
        return Err("empty prompt".into());
    }
    if request.max_tokens == 0 {
        return Err("max_tokens must be positive".into());
    }
    let Some(context_len) = prompt_tokens
        .checked_add(request.max_tokens)
        .filter(|len| *len <= max_context)
    else {
        return Err(format!(
            "prompt {prompt_tokens} + max_tokens {} exceeds the serving ceiling {max_context}",
            request.max_tokens
        ));
    };
    if request.lora_adapter.is_some() {
        return Err("gemma4 has no LoRA support".into());
    }
    if request.echo {
        return Err("gemma4 does not echo the prompt".into());
    }
    if prefix_hit_tokens > 0 {
        return Err(format!(
            "gemma4 resolves its own prefix cache; refusing a frontend resolution claiming \
             {prefix_hit_tokens} cached tokens"
        ));
    }
    if request.kv_transfer_params.is_some() {
        return Err("gemma4 has no P/D transfer support; kv_transfer_params refused".into());
    }
    if let Some(rank) = request.data_parallel_rank {
        if rank != 0 {
            return Err(format!(
                "gemma4 is single-partition; data_parallel_rank {rank} refused"
            ));
        }
    }
    Ok(context_len)
}

/// Both pools' page budgets for one configuration; `None` when the
/// arithmetic overflows.
fn pool_pages(
    transient_pages: usize,
    window_pages: usize,
    context_pages: usize,
    slots: usize,
    cache_entries: usize,
    entry_global_pages: usize,
) -> Option<(usize, usize)> {
    let local = (slots - 1)
        .checked_mul(window_pages)?
        .checked_add(transient_pages)?
        .checked_add(1)?
        .checked_add(cache_entries.checked_mul(window_pages)?)?;
    let global = slots
        .checked_mul(context_pages)?
        .checked_add(1)?
        .checked_add(cache_entries.checked_mul(entry_global_pages)?)?;
    Some((local, global))
}

/// The global family never releases, so a request's whole account — its
/// validated context length, page-ceilinged — is what admission must see.
/// The pool provisions slots times the ceiling and validation caps every
/// request inside it, so a shortfall at this door is an accounting bug
/// surfacing before any segment runs, not a load signal.
fn global_account_pages(context_len: usize) -> usize {
    context_len.div_ceil(PAGE_SIZE)
}

/// How many prompts one mixed step may absorb: bounded well below the
/// sampler-row capacity so a burst still leaves decode rows headroom.
const MIX_MAX_PROMPTS: usize = 4;

/// The gathered step's prompt-row ceiling. Gathering amortizes only the
/// step floor (~27 ms at 12B) while every live stream's inter-token gap
/// pays the whole gathered step (~0.2 ms per row), so absorbing long
/// prompts into one step trades a large certain loss for a small fixed
/// win — measured at 16 coincident ~1900-token prompts, an unbounded
/// gather tripled the stream's p99 gap for a sub-2% wall saving. Short
/// bursts are where the floor dominates; the ceiling keeps the gather
/// there, and a long prompt keeps its own step.
const MIX_GATHER_ROWS: usize = 512;

/// One prompt mid-walk: its unseen suffix begins at `offset`, and `first`
/// holds the token its final segment sampled until the walker graduates.
struct Walker {
    request: GenerateRequest,
    kv: GemmaKv,
    resumed: Option<u64>,
    offset: usize,
    first: Option<(u32, Option<TokenLogprob>)>,
    failed: bool,
}

#[derive(Clone, Copy)]
enum AdmissionNeed {
    Tokens(usize),
    GlobalPages(usize),
}

enum ReservationDecision {
    Ready,
    Requeue,
    Refused(String),
}

/// One row of a step's sampler call. A mid-walk segment's row is sampled and
/// discarded, so it carries `ignore_eos` whatever its request asked and a
/// `logprobs` of 0: it never stops and is never scored.
#[derive(Clone, Copy)]
struct SampleRow<'a> {
    params: &'a pegainfer_frontend::sampler::SamplingParams,
    step: u64,
    logprobs: usize,
    ignore_eos: bool,
}

/// One sampler call's outcome, row-aligned with the logits it read.
struct SampledRows {
    picked: Vec<u32>,
    logprobs: Vec<Option<TokenLogprob>>,
    stops: Vec<bool>,
}

/// Suppress, sample and score one step's logits, with the failed stage on the
/// error's context chain. Nothing here sends an event or moves request state:
/// what a pick means for its row is the only thing the callers disagree about.
#[allow(clippy::too_many_arguments)]
fn sample_logits_rows(
    ctx: &DeviceContext,
    suppress_ids: &ops::SuppressIds,
    policy: &GenerationPolicy,
    scratch: &mut SampleScratch,
    base_seed: u64,
    sample_nonce: &mut u64,
    rows: &[SampleRow<'_>],
    logits: &mut HiddenStates,
) -> Result<SampledRows> {
    ops::suppress_logits_bf16_in_place(ctx, logits, suppress_ids).context("suppression")?;

    *sample_nonce = sample_nonce.wrapping_add(1);
    let call_seed = base_seed ^ sample_nonce.rotate_left(17);
    let picked = {
        let params: Vec<_> = rows.iter().map(|row| row.params).collect();
        let steps: Vec<u64> = rows.iter().map(|row| row.step).collect();
        pegainfer_sample::select_batch(ctx, logits, &params, &steps, call_seed, scratch)
            .context("sampling")?
    };
    let stops: Vec<bool> = rows
        .iter()
        .zip(&picked)
        .map(|(row, &id)| policy.stops(id, row.ignore_eos))
        .collect();
    let requests: Vec<LogprobRequest> = rows
        .iter()
        .enumerate()
        .filter(|(row, spec)| spec.logprobs > 0 && !stops[*row])
        .map(|(row, spec)| LogprobRequest {
            row,
            picked: picked[row],
            top_k: spec.logprobs,
        })
        .collect();
    let mut logprobs: Vec<Option<TokenLogprob>> = vec![None; rows.len()];
    if !requests.is_empty() {
        let scored =
            pegainfer_sample::token_logprobs_batch(ctx, logits, &requests).context("logprobs")?;
        for (request, logprob) in requests.iter().zip(scored) {
            logprobs[request.row] = Some(logprob);
        }
    }
    Ok(SampledRows {
        picked,
        logprobs,
        stops,
    })
}

/// Capture this prompt's tail into the prefix cache when one is configured and
/// the state qualifies; `resumed` is the entry it supersedes. The fields come
/// in apart rather than as `&mut self` because a caller still holding the
/// step's logits holds a borrow of the arena.
fn capture_prefix(
    ctx: &DeviceContext,
    serve: &GemmaServe,
    cache: &mut Option<PrefixCache>,
    kv: &GemmaKv,
    prompt: &[u32],
    resumed: Option<u64>,
) {
    if let Some(cache) = cache.as_mut() {
        if let Some(entry) = serve.capture_checkpoint(ctx, kv, prompt) {
            cache.insert(entry, resumed);
        }
    }
}

/// Retire the whole live batch: a step that could not run leaves no row a token.
fn fail_active_batch(active: &mut Vec<Active>, what: &str, err: &anyhow::Error) {
    log::error!("{what} failed: {err:#}");
    for entry in active.drain(..) {
        let _ = entry.request.token_tx.send(TokenEvent::Error {
            message: format!("{what} failed: {err:#}"),
            prompt_tokens: entry.prompt_tokens,
            completion_tokens: entry.emitted,
        });
    }
}

/// Sample one mixed step's logits — `head` describes rows `0..head.len()`, the
/// active rows follow — then deliver the active rows' events. `Err` carries the
/// failure message after the active batch has been failed.
#[allow(clippy::too_many_arguments)]
fn mixed_head_flow(
    ctx: &DeviceContext,
    suppress_ids: &ops::SuppressIds,
    policy: &GenerationPolicy,
    scratch: &mut SampleScratch,
    base_seed: u64,
    sample_nonce: &mut u64,
    head: &[SampleRow<'_>],
    active: &mut Vec<Active>,
    logits: &mut HiddenStates,
) -> Result<SampledRows, String> {
    let k = head.len();
    let sampled = {
        let rows: Vec<SampleRow<'_>> = head
            .iter()
            .copied()
            .chain(active.iter().map(Active::sample_row))
            .collect();
        sample_logits_rows(
            ctx,
            suppress_ids,
            policy,
            scratch,
            base_seed,
            sample_nonce,
            &rows,
            logits,
        )
    };
    let mut sampled = match sampled {
        Ok(sampled) => sampled,
        Err(err) => {
            fail_active_batch(active, "mixed step", &err);
            return Err(format!("mixed step failed: {err:#}"));
        }
    };
    // Active rows: the decode-round event flow, `k` logits rows up.
    emit_decode_rows(active, &mut sampled, k);
    Ok(sampled)
}

/// One in-flight request between decode steps: its KV, the token that feeds
/// the next step, and its progress counters.
struct Active {
    request: GenerateRequest,
    kv: GemmaKv,
    next: u32,
    emitted: usize,
    prompt_tokens: usize,
}

impl Active {
    fn sample_row(&self) -> SampleRow<'_> {
        SampleRow {
            params: &self.request.params,
            step: self.emitted as u64,
            logprobs: self.request.logprobs,
            ignore_eos: self.request.params.ignore_eos,
        }
    }
}

enum Admitted {
    Active(Box<Active>),
    /// Finished, refused, cancelled or failed: nothing carries forward.
    Done,
    /// The pools cannot hold it right now; retry once pages return.
    Requeue(Box<Submitted>),
}

fn send_scheduled(request: &GenerateRequest, prompt_tokens: usize, cached_tokens: usize) -> bool {
    request
        .token_tx
        .send(TokenEvent::Scheduled {
            queued_at_unix_s: request.queued_at_unix_s.unwrap_or_else(unix_now_s),
            scheduled_at_unix_s: unix_now_s(),
            prompt_tokens,
            cached_tokens,
        })
        .is_ok()
}

/// Everything the engine thread owns for the life of the process. CUDA state
/// is thread-affine, so it is built here rather than handed in: a context or
/// cuBLAS handle minted on the caller thread fails with invalid-handle on the
/// first GEMM.
struct EngineState {
    ctx: DeviceContext,
    serve: GemmaServe,
    arena: StepArena,
    scratch: SampleScratch,
    /// Conversation-tail prefix cache; `None` unless
    /// `PEGAINFER_PREFIX_CACHE=K` opted in at startup.
    prefix_cache: Option<PrefixCache>,
    policy: GenerationPolicy,
    /// Validated once against the head, then retained on-device for one mask
    /// launch per logits batch.
    suppress_ids: ops::SuppressIds,
    base_seed: u64,
    /// Seedless sampling variety across requests comes from this counter
    /// mixed into the per-call seed; a request's own `params.seed` replays
    /// via (seed, step) regardless of it.
    sample_nonce: u64,
    /// The overlap lane; `None` unless `PEGAINFER_ASYNC_PREFILL` opted in
    /// at startup.
    lane: Option<AsyncPrefillLane>,
    /// The chunked-walk segment span; `None` unless
    /// `PEGAINFER_MIX_CHUNK_TOKENS` opted in at startup.
    mix_chunk: Option<usize>,
    /// The serving ceiling this process was started with; the pools are
    /// budgeted against it.
    max_context: usize,
    /// The decode-slot count the pools are budgeted for; requests past it
    /// queue.
    slots: usize,
}

impl EngineState {
    fn reserve_with_eviction(
        &mut self,
        kv: &mut GemmaKv,
        need: AdmissionNeed,
        can_wait: bool,
    ) -> ReservationDecision {
        loop {
            let refusal = match need {
                AdmissionNeed::Tokens(tokens) => {
                    admit_tokens(&self.serve.local_pool, &self.serve.global_pool, kv, tokens)
                        .err()
                        .map(|err| format!("admission refused: {err:#}"))
                }
                AdmissionNeed::GlobalPages(pages)
                    if pages
                        > kv.global.held_pages() + self.serve.global_pool.available_pages() =>
                {
                    Some(format!(
                        "the global family cannot hold this request's {pages} pages"
                    ))
                }
                AdmissionNeed::GlobalPages(_) => None,
            };
            let Some(message) = refusal else {
                return ReservationDecision::Ready;
            };
            if self
                .prefix_cache
                .as_mut()
                .is_some_and(PrefixCache::evict_lru)
            {
                continue;
            }
            return if can_wait {
                ReservationDecision::Requeue
            } else {
                ReservationDecision::Refused(message)
            };
        }
    }

    fn load(
        dir: &str,
        device: usize,
        policy: GenerationPolicy,
        base_seed: u64,
        graph_enabled: bool,
    ) -> Result<Self> {
        // Refuse an unservable global GQA shape, a bad lane mode or a bad
        // ceiling before the multi-GiB load.
        let config = crate::config::Gemma4Config::from_file(dir)?;
        let global_split = crate::serve::global_split_factor(&config)?;
        let max_context = serving_context(config.max_position_embeddings)?;
        let lane_mode = async_prefill_mode()?;
        let mix_chunk = mix_chunk_tokens(max_context)?;
        let slots = decode_slots()?;
        if max_context > MAX_CONTEXT {
            anyhow::ensure!(
                mix_chunk.is_some(),
                "PEGAINFER_MAX_CONTEXT={max_context} needs PEGAINFER_MIX_CHUNK_TOKENS: a whole \
                 scan would hold the full context in sliding pages"
            );
            anyhow::ensure!(
                lane_mode.is_none(),
                "the overlap lane prefills whole; PEGAINFER_ASYNC_PREFILL is unsupported over \
                 the default {MAX_CONTEXT} ceiling"
            );
        }
        let (weights, _) = Gemma4Weights::from_safetensors(dir, device, config)?;
        let ctx = DeviceContext::new_with_device(device)?;
        let vocab = weights.embed_tokens.rows;
        policy.check_against_vocab(vocab)?;
        // Pool budget for a batch. Whole-prompt admissions hold every page
        // of their prompt until the step releases, so without the chunk knob
        // the local pool carries one full-context transient on top of the
        // window-capped steady footprint of the other active requests; with
        // it, every scan is bounded and the transient shrinks to window plus
        // segment. The global family never releases, so it stays linear in
        // context for each request's whole lifetime. Both pools add the
        // padding page they reserve.
        let context_pages = max_context.div_ceil(PAGE_SIZE);
        let window_pages = weights.config.sliding_window.div_ceil(PAGE_SIZE) + 1;
        // The cache brings its own page budget so cached entries never eat
        // serving headroom.
        let cache_cap = prefix_cache_cap()?;
        let cache_entries = cache_cap.unwrap_or(0);
        let sliding_window = weights.config.sliding_window;
        // With the chunk knob set every scan is bounded by window plus
        // segment — except the lane's, which prefills whole and keeps the
        // full transient.
        let transient_pages = match mix_chunk {
            Some(chunk) if lane_mode.is_none() => {
                // A round's rows split across walkers, and every walker's
                // reservation rounds up to its own page — so the budget
                // carries one page of rounding per extra walker.
                window_pages + chunk.div_ceil(PAGE_SIZE) + (MIX_MAX_PROMPTS - 1)
            }
            _ => context_pages,
        };
        let (local_pages, global_pages) = pool_pages(
            transient_pages,
            window_pages,
            context_pages,
            slots,
            cache_entries,
            crate::prefix_cache::entry_global_pages(max_context),
        )
        .ok_or_else(|| {
            anyhow::anyhow!(
                "the pool budget arithmetic overflows for a {max_context} token ceiling with \
                 {slots} slots and {cache_entries} cache entries"
            )
        })?;
        // The arena pads steps to power-of-two buckets.
        let arena_rows = slots.next_power_of_two();
        // Page ids and mixed-step row metadata are i32 downstream, and the
        // ceiling guard alone does not bound what slots and cache multiply
        // out to — the derived counts answer here, before any allocation.
        anyhow::ensure!(
            i32::try_from(local_pages).is_ok()
                && global_pages
                    .checked_mul(global_split)
                    .is_some_and(|expanded| i32::try_from(expanded).is_ok())
                && max_context
                    .checked_add(arena_rows)
                    .is_some_and(|cap| i32::try_from(cap).is_ok()),
            "a {max_context} token ceiling with {slots} slots and {cache_entries} cache entries \
             derives page or row counts past the i32 metadata domain (the global family's pseudo \
             tables carry {global_split} copies of every page)"
        );
        let serve = GemmaServe::new(&ctx, weights, max_context, local_pages, global_pages)
            .map_err(|err| {
                err.context(format!(
                    "a {max_context} token ceiling, {slots} decode slots and {cache_entries} \
                     cache entries sized the pools to {local_pages} local / {global_pages} \
                     global pages"
                ))
            })?;
        let prefix_cache = cache_cap.map(|k| PrefixCache::new(k, sliding_window));
        let scratch = SampleScratch::new(&ctx, vocab, arena_rows)?;
        let mut arena = serve.alloc_step_arena(&ctx, arena_rows, graph_enabled)?;
        serve.precapture_decode_graphs(&ctx, &mut arena)?;
        let suppress_ids = ops::SuppressIds::upload(&ctx, &policy.suppress, vocab)?;
        let lane = lane_mode
            .map(|mode| AsyncPrefillLane::new(&ctx, mode))
            .transpose()?;
        Ok(Self {
            ctx,
            serve,
            arena,
            scratch,
            prefix_cache,
            policy,
            suppress_ids,
            base_seed,
            sample_nonce: 0,
            lane,
            mix_chunk,
            max_context,
            slots,
        })
    }

    /// One turn's intake at the roster edge: admit from the queue head
    /// until the slots are full, the queue empties, the lane is busy, or a
    /// request has to wait for pages. Attempts are bounded so a burst
    /// costs the streams a bounded number of prefills per token however
    /// deep the queue is.
    fn admit_from_queue(&mut self, pending: &mut VecDeque<Submitted>, active: &mut Vec<Active>) {
        let mut attempts = 0;
        while attempts < self.slots && active.len() < self.slots {
            // With the lane busy, arrivals wait in `pending` while decode
            // keeps stepping.
            if self
                .lane
                .as_ref()
                .is_some_and(|lane| lane.inflight.is_some())
            {
                break;
            }
            let Some(item) = pending.pop_front() else {
                break;
            };
            attempts += 1;
            let can_wait = !active.is_empty();
            match self.admit_and_prefill(item, can_wait, active, pending, &mut attempts) {
                Admitted::Active(request) => active.push(*request),
                Admitted::Done => {}
                Admitted::Requeue(item) => {
                    pending.push_front(*item);
                    break;
                }
            }
        }
    }

    /// Validate, admit and prefill one request whole, emit its first token,
    /// and hand it to the decode batch. An admission refusal is a refusal to
    /// the client only when no active request could free the pages it needs;
    /// otherwise the request waits at the queue head.
    fn admit_and_prefill(
        &mut self,
        item: Submitted,
        can_wait: bool,
        active: &mut Vec<Active>,
        pending: &mut VecDeque<Submitted>,
        attempts: &mut usize,
    ) -> Admitted {
        let (request, prefix) = item;
        let sink = request.token_tx.clone();
        if sink.is_closed() {
            return Admitted::Done;
        }
        let prompt_tokens = request.prompt_tokens.len();
        // Scheduled is paired with whatever ends the request, so a refusal
        // emits it first rather than leaving the client with no lifecycle.
        let reject = |message: String| {
            if send_scheduled(&request, prompt_tokens, 0) {
                let _ = sink.send(TokenEvent::Rejected {
                    message,
                    prompt_tokens,
                    completion_tokens: 0,
                });
            }
            Admitted::Done
        };
        let context_len = match validate_request(&request, prefix.hit_tokens(), self.max_context) {
            Ok(len) => len,
            Err(message) => return reject(message),
        };

        let mut resumed = None;
        let mut kv = match self
            .prefix_cache
            .as_mut()
            .and_then(|cache| cache.resolve(&request.prompt_tokens))
        {
            Some((entry, t)) => match self.serve.restore_from_checkpoint(&self.ctx, entry, t) {
                Ok(kv) => {
                    resumed = Some(entry.id);
                    kv
                }
                Err(err) => {
                    log::warn!("gemma4 prefix-cache restore failed (falling back): {err:#}");
                    self.serve.alloc_kv()
                }
            },
            None => self.serve.alloc_kv(),
        };
        // The lane prefills whole on its own stream, so a lane-bound
        // admission still reserves everything up front. A chunked
        // admission reserves nothing here: every segment admits its own
        // pages right before it is written, so no walker parks a quantum
        // — parked first segments across several walkers would exhaust
        // the one shared segment transient the pool provisions.
        let lane_takes = self.lane.is_some() && !active.is_empty();
        let need = if self.mix_chunk.is_none() || lane_takes {
            AdmissionNeed::Tokens(prompt_tokens - kv.local.seq_len())
        } else {
            // A chunked admission reserves per segment; the whole-account
            // door (see `global_account_pages`) still answers up front.
            AdmissionNeed::GlobalPages(global_account_pages(context_len))
        };
        match self.reserve_with_eviction(&mut kv, need, can_wait) {
            ReservationDecision::Ready => {}
            ReservationDecision::Requeue => {
                return Admitted::Requeue(Box::new((request, prefix)));
            }
            ReservationDecision::Refused(message) => return reject(message),
        }
        // A restored prefix is what the bridge reports as cached: the
        // resumed KV's frontier is exactly the token count served from it.
        if !send_scheduled(&request, prompt_tokens, kv.local.seq_len()) {
            return Admitted::Done;
        }

        // Overlapped admission: the prefill launches onto the lane stream
        // and this call returns immediately — decode steps continue while it
        // runs. A prompt arriving with nothing active stays on the sync
        // path: there is nothing to protect, and full-SM speed wins the head
        // of every refill burst.
        if self.lane.is_some() && !active.is_empty() {
            return self.launch_async_prefill(request, kv, resumed);
        }

        // Mixed admission: with a live decode batch, prompts ride its
        // weight scan — one step prefills every gathered newcomer and
        // advances every active row.
        if !active.is_empty() {
            self.ready_decode_rows(active);
            if !active.is_empty() {
                // Gather more admissible prompts into the same step. A
                // pool-refused or over-budget candidate returns to the queue
                // head and stops the gather — the engine loop's
                // head-of-line-waits semantics — and an invalid one is
                // rejected in place. Every popped candidate consumes the
                // turn's shared admission budget, gathered or not, so a
                // queue of dead submissions cannot stall the decode round.
                // The row pricing runs after the prefix-cache resolve: a
                // warm candidate costs the step only its unseen suffix.
                let mut newcomers: Vec<(GenerateRequest, GemmaKv, Option<u64>)> =
                    vec![(request, kv, resumed)];
                let mut rows_budget = {
                    let (_, kv, _) = &newcomers[0];
                    prompt_tokens - kv.local.seq_len()
                };
                while newcomers.len() < MIX_MAX_PROMPTS
                    && (self.mix_chunk.is_some() || rows_budget < MIX_GATHER_ROWS)
                    && newcomers.len() + active.len() < self.slots
                    && *attempts < self.slots
                {
                    let Some((cand, cand_prefix)) = pending.pop_front() else {
                        break;
                    };
                    *attempts += 1;
                    let cand_sink = cand.token_tx.clone();
                    if cand_sink.is_closed() {
                        continue;
                    }
                    let cand_context_len =
                        match validate_request(&cand, cand_prefix.hit_tokens(), self.max_context) {
                            Ok(len) => len,
                            Err(message) => {
                                let n = cand.prompt_tokens.len();
                                if send_scheduled(&cand, n, 0) {
                                    let _ = cand_sink.send(TokenEvent::Rejected {
                                        message,
                                        prompt_tokens: n,
                                        completion_tokens: 0,
                                    });
                                }
                                continue;
                            }
                        };
                    let cand_len = cand.prompt_tokens.len();
                    let mut cand_resumed = None;
                    let mut cand_kv = match self
                        .prefix_cache
                        .as_mut()
                        .and_then(|cache| cache.resolve(&cand.prompt_tokens))
                    {
                        Some((entry, t)) => {
                            match self.serve.restore_from_checkpoint(&self.ctx, entry, t) {
                                Ok(kv) => {
                                    cand_resumed = Some(entry.id);
                                    kv
                                }
                                Err(err) => {
                                    log::warn!(
                                        "gemma4 prefix-cache restore failed (falling back): {err:#}"
                                    );
                                    self.serve.alloc_kv()
                                }
                            }
                        }
                        None => self.serve.alloc_kv(),
                    };
                    let new_tokens = cand_len - cand_kv.local.seq_len();
                    if self.mix_chunk.is_some() {
                        // A chunked candidate reserves nothing locally: its
                        // segments admit their own pages inside the walk;
                        // the whole-account door still answers here.
                        let cand_global_want = global_account_pages(cand_context_len);
                        if cand_global_want
                            > cand_kv.global.held_pages() + self.serve.global_pool.available_pages()
                        {
                            pending.push_front((cand, cand_prefix));
                            break;
                        }
                        if !send_scheduled(&cand, cand_len, cand_kv.local.seq_len()) {
                            continue;
                        }
                        rows_budget += new_tokens;
                        newcomers.push((cand, cand_kv, cand_resumed));
                        continue;
                    }
                    if rows_budget + new_tokens > MIX_GATHER_ROWS {
                        pending.push_front((cand, cand_prefix));
                        break;
                    }
                    if admit_tokens(
                        &self.serve.local_pool,
                        &self.serve.global_pool,
                        &mut cand_kv,
                        new_tokens,
                    )
                    .is_err()
                    {
                        pending.push_front((cand, cand_prefix));
                        break;
                    }
                    if !send_scheduled(&cand, cand_len, cand_kv.local.seq_len()) {
                        continue;
                    }
                    rows_budget += new_tokens;
                    newcomers.push((cand, cand_kv, cand_resumed));
                }
                return self.mixed_admission(newcomers, active);
            }
        }

        let fail = |message: String| {
            let _ = sink.send(TokenEvent::Error {
                message,
                prompt_tokens,
                completion_tokens: 0,
            });
            Admitted::Done
        };
        // Under the chunk knob a solo prompt walks its own segments too:
        // residency stays window plus segment whatever the prompt length.
        let stepped = if let Some(chunk) = self.mix_chunk {
            self.walk_plain_prompt(&mut kv, &request.prompt_tokens, chunk)
        } else {
            let resume = kv.local.seq_len();
            self.serve
                .step(&self.ctx, &mut kv, &request.prompt_tokens[resume..])
        };
        let mut logits = match stepped {
            Ok(logits) => logits,
            Err(err) => return fail(format!("{err:#}")),
        };
        capture_prefix(
            &self.ctx,
            &self.serve,
            &mut self.prefix_cache,
            &kv,
            &request.prompt_tokens,
            resumed,
        );
        self.first_token_flow(request, kv, &mut logits)
    }

    /// Sample and settle a prefill's first token from logits row 0 — the
    /// shared tail of a sync admission and an overlapped-prefill join.
    fn first_token_flow(
        &mut self,
        request: GenerateRequest,
        kv: GemmaKv,
        logits: &mut HiddenStates,
    ) -> Admitted {
        let sampled = {
            let rows = [SampleRow {
                params: &request.params,
                step: 0,
                logprobs: request.logprobs,
                ignore_eos: request.params.ignore_eos,
            }];
            sample_logits_rows(
                &self.ctx,
                &self.suppress_ids,
                &self.policy,
                &mut self.scratch,
                self.base_seed,
                &mut self.sample_nonce,
                &rows,
                logits,
            )
        };
        let mut sampled = match sampled {
            Ok(sampled) => sampled,
            Err(err) => {
                let _ = request.token_tx.send(TokenEvent::Error {
                    message: format!("{err:#}"),
                    prompt_tokens: request.prompt_tokens.len(),
                    completion_tokens: 0,
                });
                return Admitted::Done;
            }
        };
        match settle_first_token(
            &self.policy,
            request,
            kv,
            sampled.picked[0],
            sampled.logprobs[0].take(),
        ) {
            Some(entry) => Admitted::Active(Box::new(entry)),
            None => Admitted::Done,
        }
    }

    /// Launch one whole-prompt prefill onto the lane stream and record the
    /// completion event. On a launch error the lane stream is drained
    /// before the KV reservation drops, so no returned page can still be
    /// written by a stale kernel.
    fn launch_async_prefill(
        &mut self,
        request: GenerateRequest,
        mut kv: GemmaKv,
        resumed: Option<u64>,
    ) -> Admitted {
        let lane = self.lane.as_mut().expect("gated by the caller");
        debug_assert!(lane.inflight.is_none());
        // A restored prefix is already in the KV: the lane prefills only the
        // unseen suffix, exactly like the sync and mixed paths.
        let resume = kv.local.seq_len();
        let launched = {
            let _guard = unsafe {
                pegainfer_core::tensor::StreamOverrideGuard::activate(lane.stream.stream)
            };
            self.serve
                .prefill_into_logits(&self.ctx, &mut kv, &request.prompt_tokens[resume..])
        };
        let recorded = launched.and_then(|pass| {
            lane.stream
                .record_event(lane.event.cu_event())
                .map(|()| pass)
        });
        match recorded {
            Ok(pass) => {
                lane.inflight = Some(InflightPrefill {
                    request,
                    kv,
                    pass,
                    resumed,
                });
                Admitted::Done
            }
            Err(err) => {
                log::error!("gemma4 async prefill launch failed: {err:#}");
                lane.drain_or_abort();
                let _ = request.token_tx.send(TokenEvent::Error {
                    message: format!("prefill failed: {err:#}"),
                    prompt_tokens: request.prompt_tokens.len(),
                    completion_tokens: 0,
                });
                Admitted::Done
            }
        }
    }

    /// Join a completed overlapped prefill: run the deferred window
    /// release, capture into the prefix cache, and take the first-token
    /// flow the sync path uses.
    fn join_async_prefill(&mut self, active: &mut Vec<Active>) {
        let Some(lane) = self.lane.as_mut() else {
            return;
        };
        let Some(inflight) = lane.inflight.take() else {
            return;
        };
        let InflightPrefill {
            request,
            mut kv,
            mut pass,
            resumed,
        } = inflight;
        // The frontier after any prefill equals the prompt length; a lane
        // pass that processed the wrong suffix cannot pass this gate.
        if kv.local.seq_len() != request.prompt_tokens.len() {
            log::error!(
                "gemma4 async prefill frontier {} != prompt {}",
                kv.local.seq_len(),
                request.prompt_tokens.len()
            );
            let _ = request.token_tx.send(TokenEvent::Error {
                message: "async prefill frontier mismatch".into(),
                prompt_tokens: request.prompt_tokens.len(),
                completion_tokens: 0,
            });
            return;
        }
        if let Err(err) = self.serve.release_prefill_window(&mut kv) {
            let _ = request.token_tx.send(TokenEvent::Error {
                message: format!("prefill window release failed: {err:#}"),
                prompt_tokens: request.prompt_tokens.len(),
                completion_tokens: 0,
            });
            return;
        }
        capture_prefix(
            &self.ctx,
            &self.serve,
            &mut self.prefix_cache,
            &kv,
            &request.prompt_tokens,
            resumed,
        );
        if let Admitted::Active(entry) = self.first_token_flow(request, kv, &mut pass.logits) {
            active.push(*entry);
        }
    }

    /// The chunked walk behind `PEGAINFER_MIX_CHUNK_TOKENS`: every
    /// gathered prompt walks the same segment schedule, one shared mixed
    /// step per round, packing up to `chunk` unseen prompt rows across
    /// walkers in admission order on top of the live decode batch — the
    /// streams advance one token per round instead of waiting out whole
    /// prompts. Each round samples every segment's last row; only a
    /// walker's final segment's row is kept as its first token, and that
    /// walker graduates into the decode batch at the round boundary. A
    /// drained roster finishes the remaining tails on the plain path,
    /// segment by segment.
    fn walk_plain_prompt(
        &self,
        kv: &mut GemmaKv,
        prompt: &[u32],
        chunk: usize,
    ) -> Result<HiddenStates> {
        while prompt.len() - kv.local.seq_len() > chunk {
            let offset = kv.local.seq_len();
            self.step_plain_segment(kv, &prompt[offset..offset + chunk])?;
        }
        let offset = kv.local.seq_len();
        self.step_plain_segment(kv, &prompt[offset..])
    }

    fn step_plain_segment(&self, kv: &mut GemmaKv, tokens: &[u32]) -> Result<HiddenStates> {
        admit_tokens(
            &self.serve.local_pool,
            &self.serve.global_pool,
            kv,
            tokens.len(),
        )?;
        self.serve.step(&self.ctx, kv, tokens)
    }

    fn finish_plain_walker(&mut self, walker: &mut Walker, chunk: usize, active: &mut Vec<Active>) {
        let mut logits =
            match self.walk_plain_prompt(&mut walker.kv, &walker.request.prompt_tokens, chunk) {
                Ok(logits) => logits,
                Err(err) => {
                    let _ = walker.request.token_tx.send(TokenEvent::Error {
                        message: format!("walk tail failed: {err:#}"),
                        prompt_tokens: walker.request.prompt_tokens.len(),
                        completion_tokens: 0,
                    });
                    walker.failed = true;
                    return;
                }
            };
        walker.offset = walker.request.prompt_tokens.len();
        let head = [SampleRow {
            params: &walker.request.params,
            step: 0,
            logprobs: walker.request.logprobs,
            ignore_eos: walker.request.params.ignore_eos,
        }];
        match mixed_head_flow(
            &self.ctx,
            &self.suppress_ids,
            &self.policy,
            &mut self.scratch,
            self.base_seed,
            &mut self.sample_nonce,
            &head,
            active,
            &mut logits,
        ) {
            Ok(mut sampled) => {
                walker.first = Some((sampled.picked[0], sampled.logprobs[0].take()));
            }
            Err(message) => {
                let _ = walker.request.token_tx.send(TokenEvent::Error {
                    message,
                    prompt_tokens: walker.request.prompt_tokens.len(),
                    completion_tokens: 0,
                });
                walker.failed = true;
            }
        }
    }

    fn mixed_walk(
        &mut self,
        chunk: usize,
        newcomers: Vec<(GenerateRequest, GemmaKv, Option<u64>)>,
        active: &mut Vec<Active>,
    ) -> Admitted {
        let mut walkers: Vec<Walker> = newcomers
            .into_iter()
            .map(|(request, kv, resumed)| Walker {
                offset: kv.local.seq_len(),
                request,
                kv,
                resumed,
                first: None,
                failed: false,
            })
            .collect();

        let mut first_round = true;
        loop {
            // A client that disconnected mid-walk stops consuming rounds;
            // its pages return with the walker.
            for w in &mut walkers {
                if !w.failed && w.request.token_tx.is_closed() {
                    w.failed = true;
                }
            }
            // A finished walker graduates at the round boundary instead of
            // idling out the other walkers.
            let mut i = 0;
            while i < walkers.len() {
                if !walkers[i].failed && walkers[i].first.is_some() {
                    let w = walkers.remove(i);
                    self.graduate_walker(w, active);
                } else {
                    i += 1;
                }
            }
            if !walkers
                .iter()
                .any(|w| !w.failed && w.offset < w.request.prompt_tokens.len())
            {
                break;
            }
            if !first_round {
                self.ready_decode_rows(active);
            }
            first_round = false;

            if active.is_empty() {
                // The batch drained mid-walk: finish each remaining tail
                // on the plain path, one segment at a time — the KV
                // frontier carries the position.
                for w in &mut walkers {
                    if w.failed || w.offset >= w.request.prompt_tokens.len() {
                        continue;
                    }
                    self.finish_plain_walker(w, chunk, active);
                }
                continue;
            }

            // This round's slate, in walker order: rows to take per walker
            // and whether that take completes its prompt.
            let mut budget = chunk;
            let mut takes: Vec<Option<(usize, bool)>> = vec![None; walkers.len()];
            for (wi, w) in walkers.iter_mut().enumerate() {
                if w.failed {
                    continue;
                }
                let rest = w.request.prompt_tokens.len() - w.offset;
                if rest == 0 || budget == 0 {
                    continue;
                }
                let take = rest.min(budget);
                // Reserve what this round writes; a walker holding these
                // pages already skips the call. A refusal means the pool's
                // provision failed — fail the walker loud rather than
                // stall the round.
                if let Err(err) = admit_tokens(
                    &self.serve.local_pool,
                    &self.serve.global_pool,
                    &mut w.kv,
                    take,
                ) {
                    let _ = w.request.token_tx.send(TokenEvent::Error {
                        message: format!("walk segment admission failed: {err:#}"),
                        prompt_tokens: w.request.prompt_tokens.len(),
                        completion_tokens: 0,
                    });
                    w.failed = true;
                    continue;
                }
                takes[wi] = Some((take, take == rest));
                budget -= take;
            }

            let decode_tokens: Vec<u32> = active.iter().map(|entry| entry.next).collect();
            let stepped = {
                let mut kvs: Vec<&mut GemmaKv> =
                    active.iter_mut().map(|entry| &mut entry.kv).collect();
                let mut prefills: Vec<(&mut GemmaKv, &[u32])> = Vec::new();
                for (w, t) in walkers.iter_mut().zip(&takes) {
                    if let Some((take, _)) = *t {
                        let seg = &w.request.prompt_tokens[w.offset..w.offset + take];
                        prefills.push((&mut w.kv, seg));
                    }
                }
                self.serve.mixed_prefill_decode_step(
                    &self.ctx,
                    &mut self.arena,
                    &mut prefills,
                    &mut kvs,
                    &decode_tokens,
                )
            };
            let logits = match stepped {
                Ok(logits) => logits,
                Err(err) => {
                    fail_active_batch(active, "walk step", &err);
                    for w in walkers.drain(..) {
                        let _ = w.request.token_tx.send(TokenEvent::Error {
                            message: format!("walk step failed: {err:#}"),
                            prompt_tokens: w.request.prompt_tokens.len(),
                            completion_tokens: 0,
                        });
                    }
                    return Admitted::Done;
                }
            };

            let flow = {
                let head: Vec<SampleRow<'_>> = walkers
                    .iter()
                    .zip(&takes)
                    .filter(|(_, t)| t.is_some())
                    .map(|(w, t)| {
                        let (_, last) = t.expect("filtered");
                        SampleRow {
                            params: &w.request.params,
                            step: 0,
                            logprobs: if last { w.request.logprobs } else { 0 },
                            ignore_eos: if last {
                                w.request.params.ignore_eos
                            } else {
                                true
                            },
                        }
                    })
                    .collect();
                mixed_head_flow(
                    &self.ctx,
                    &self.suppress_ids,
                    &self.policy,
                    &mut self.scratch,
                    self.base_seed,
                    &mut self.sample_nonce,
                    &head,
                    active,
                    logits,
                )
            };
            match flow {
                Ok(mut sampled) => {
                    let mut si = 0usize;
                    for (w, t) in walkers.iter_mut().zip(&takes) {
                        if let Some((take, last)) = *t {
                            w.offset += take;
                            if last {
                                w.first = Some((sampled.picked[si], sampled.logprobs[si].take()));
                            }
                            si += 1;
                        }
                    }
                }
                Err(message) => {
                    for w in walkers.drain(..) {
                        let _ = w.request.token_tx.send(TokenEvent::Error {
                            message: message.clone(),
                            prompt_tokens: w.request.prompt_tokens.len(),
                            completion_tokens: 0,
                        });
                    }
                    return Admitted::Done;
                }
            }
        }

        Admitted::Done
    }

    /// One finished walker joins the batch at its round boundary: capture
    /// its prompt state, then emit or retire its first token exactly like
    /// the whole-prompt form.
    fn graduate_walker(&mut self, w: Walker, active: &mut Vec<Active>) {
        let Walker {
            request,
            kv,
            resumed,
            first,
            ..
        } = w;
        let (next, logprob) = first.expect("graduation follows a final segment");
        capture_prefix(
            &self.ctx,
            &self.serve,
            &mut self.prefix_cache,
            &kv,
            &request.prompt_tokens,
            resumed,
        );
        if let Some(entry) = settle_first_token(&self.policy, request, kv, next, logprob) {
            active.push(entry);
        }
    }

    /// Retire every active request the next step cannot serve — a closed
    /// sink, or a pool that cannot grow its KV by one token — and admit the
    /// step's token for the rest.
    fn ready_decode_rows(&self, active: &mut Vec<Active>) {
        let mut row = 0;
        while row < active.len() {
            let entry = &mut active[row];
            if entry.request.token_tx.is_closed() {
                active.swap_remove(row);
                continue;
            }
            if let Err(err) = admit_tokens(
                &self.serve.local_pool,
                &self.serve.global_pool,
                &mut entry.kv,
                1,
            ) {
                let _ = entry.request.token_tx.send(TokenEvent::Error {
                    message: format!("{err:#}"),
                    prompt_tokens: entry.prompt_tokens,
                    completion_tokens: entry.emitted,
                });
                active.swap_remove(row);
                continue;
            }
            row += 1;
        }
    }

    /// The mixed-admission tail of [`Self::admit_and_prefill`]: every
    /// gathered prompt and the live decode batch share one step, then one
    /// sampler call covers the newcomers' first tokens (logits rows `0..k`)
    /// and every active row after them. Finished newcomers emit in place
    /// and the rest join `active` directly, so the caller always receives
    /// `Done`.
    fn mixed_admission(
        &mut self,
        mut newcomers: Vec<(GenerateRequest, GemmaKv, Option<u64>)>,
        active: &mut Vec<Active>,
    ) -> Admitted {
        if let Some(chunk) = self.mix_chunk {
            return self.mixed_walk(chunk, newcomers, active);
        }
        let fail_newcomers = |newcomers: &mut Vec<(GenerateRequest, GemmaKv, Option<u64>)>,
                              message: &str| {
            for (request, _, _) in newcomers.drain(..) {
                let _ = request.token_tx.send(TokenEvent::Error {
                    message: message.to_string(),
                    prompt_tokens: request.prompt_tokens.len(),
                    completion_tokens: 0,
                });
            }
            Admitted::Done
        };

        let decode_tokens: Vec<u32> = active.iter().map(|entry| entry.next).collect();
        let logits = {
            let mut kvs: Vec<&mut GemmaKv> = active.iter_mut().map(|entry| &mut entry.kv).collect();
            let mut prefills: Vec<(&mut GemmaKv, &[u32])> = newcomers
                .iter_mut()
                .map(|(request, kv, _)| {
                    let resume = kv.local.seq_len();
                    (kv, &request.prompt_tokens[resume..])
                })
                .collect();
            match self.serve.mixed_prefill_decode_step(
                &self.ctx,
                &mut self.arena,
                &mut prefills,
                &mut kvs,
                &decode_tokens,
            ) {
                Ok(logits) => logits,
                Err(err) => {
                    fail_active_batch(active, "mixed step", &err);
                    return fail_newcomers(&mut newcomers, &format!("mixed step failed: {err:#}"));
                }
            }
        };
        for (request, kv, resumed) in &newcomers {
            capture_prefix(
                &self.ctx,
                &self.serve,
                &mut self.prefix_cache,
                kv,
                &request.prompt_tokens,
                *resumed,
            );
        }
        let mut sampled = {
            let head: Vec<SampleRow<'_>> = newcomers
                .iter()
                .map(|(request, _, _)| SampleRow {
                    params: &request.params,
                    step: 0,
                    logprobs: request.logprobs,
                    ignore_eos: request.params.ignore_eos,
                })
                .collect();
            match mixed_head_flow(
                &self.ctx,
                &self.suppress_ids,
                &self.policy,
                &mut self.scratch,
                self.base_seed,
                &mut self.sample_nonce,
                &head,
                active,
                logits,
            ) {
                Ok(sampled) => sampled,
                Err(message) => return fail_newcomers(&mut newcomers, &message),
            }
        };

        // The newcomers: their first tokens are logits rows `0..k`.
        for (j, (request, kv, _)) in newcomers.into_iter().enumerate() {
            if let Some(entry) = settle_first_token(
                &self.policy,
                request,
                kv,
                sampled.picked[j],
                sampled.logprobs[j].take(),
            ) {
                active.push(entry);
            }
        }
        Admitted::Done
    }

    /// One batched decode step: every active request advances a token,
    /// sharing each layer's weight pass. A cancelled request and a request
    /// the pools cannot grow for retire before the batch is built; a finished
    /// one retires after its token lands.
    fn decode_round(&mut self, active: &mut Vec<Active>) {
        self.ready_decode_rows(active);
        if active.is_empty() {
            return;
        }

        let tokens: Vec<u32> = active.iter().map(|entry| entry.next).collect();
        let logits = {
            let mut kvs: Vec<&mut GemmaKv> = active.iter_mut().map(|entry| &mut entry.kv).collect();
            match self
                .serve
                .decode_batch_step(&self.ctx, &mut self.arena, &mut kvs, &tokens)
            {
                Ok(logits) => logits,
                Err(err) => return fail_active_batch(active, "batched decode", &err),
            }
        };
        let sampled = {
            let rows: Vec<SampleRow<'_>> = active.iter().map(Active::sample_row).collect();
            sample_logits_rows(
                &self.ctx,
                &self.suppress_ids,
                &self.policy,
                &mut self.scratch,
                self.base_seed,
                &mut self.sample_nonce,
                &rows,
                logits,
            )
        };
        match sampled {
            Ok(mut sampled) => emit_decode_rows(active, &mut sampled, 0),
            Err(err) => fail_active_batch(active, "batched decode", &err),
        }
    }
}

/// Deliver one decode step's outcome to every active row and retire the
/// finished ones — the event flow both the pure decode round and the mixed
/// admission share; `row_base` is the row's offset into the step's logits
/// (the mixed step's row 0 is the newcomer). A stop token retires the
/// request without being emitted; a send failure retires a cancelled one.
fn emit_decode_rows(active: &mut Vec<Active>, sampled: &mut SampledRows, row_base: usize) {
    let mut retire: Vec<usize> = Vec::new();
    for (row, entry) in active.iter_mut().enumerate() {
        if sampled.stops[row + row_base] {
            let _ = entry.request.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Stop,
                prompt_tokens: entry.prompt_tokens,
                completion_tokens: entry.emitted,
            });
            retire.push(row);
            continue;
        }
        let token = sampled.picked[row + row_base];
        entry.emitted += 1;
        if entry
            .request
            .token_tx
            .send(TokenEvent::Token {
                id: token,
                logprob: sampled.logprobs[row + row_base].take(),
            })
            .is_err()
        {
            retire.push(row);
            continue;
        }
        if entry.emitted >= entry.request.max_tokens {
            let _ = entry.request.token_tx.send(TokenEvent::Finished {
                finish_reason: FinishReason::Length,
                prompt_tokens: entry.prompt_tokens,
                completion_tokens: entry.emitted,
            });
            retire.push(row);
            continue;
        }
        entry.next = token;
    }
    for row in retire.into_iter().rev() {
        active.swap_remove(row);
    }
}

/// Deliver one admission's first token and decide whether the request joins the
/// decode batch. The stop token retires the request without being emitted: the
/// frontend appends its own sentinel for a terminal Stop and drops the last id,
/// so an engine that emits EOS costs the client its final visible token.
fn settle_first_token(
    policy: &GenerationPolicy,
    request: GenerateRequest,
    kv: GemmaKv,
    next: u32,
    logprob: Option<TokenLogprob>,
) -> Option<Active> {
    let prompt_tokens = request.prompt_tokens.len();
    let finish = |reason: FinishReason, completion_tokens: usize| -> Option<Active> {
        let _ = request.token_tx.send(TokenEvent::Finished {
            finish_reason: reason,
            prompt_tokens,
            completion_tokens,
        });
        None
    };
    if policy.stops(next, request.params.ignore_eos) {
        return finish(FinishReason::Stop, 0);
    }
    if request
        .token_tx
        .send(TokenEvent::Token { id: next, logprob })
        .is_err()
    {
        return None;
    }
    if request.max_tokens <= 1 {
        return finish(FinishReason::Length, 1);
    }
    Some(Active {
        request,
        kv,
        next,
        emitted: 1,
        prompt_tokens,
    })
}

#[cfg(test)]
mod knob_tests {
    use super::*;

    #[test]
    fn environment_boundary_distinguishes_missing_and_malformed() {
        assert_eq!(
            normalize_env(PREFIX_CACHE_ENV, Err(std::env::VarError::NotPresent)).unwrap(),
            None
        );
        let invalid = std::ffi::OsString::from("invalid");
        let error = normalize_env(
            ASYNC_PREFILL_ENV,
            Err(std::env::VarError::NotUnicode(invalid)),
        )
        .expect_err("non-UTF-8 must refuse");
        assert!(error.to_string().contains("not valid UTF-8"));
    }

    #[test]
    fn async_prefill_parses_or_refuses() {
        for off in ["", "0", "false", "off", " OFF "] {
            assert_eq!(parse_async_prefill_mode(off).unwrap(), None);
        }
        assert_eq!(
            parse_async_prefill_mode("shared").unwrap(),
            Some(LaneMode::Shared)
        );
        assert_eq!(
            parse_async_prefill_mode("green:35").unwrap(),
            Some(LaneMode::Green(35))
        );
        for bad in [
            "1",
            "true",
            "on",
            "green",
            "green:0",
            "green:100",
            "green:x",
        ] {
            assert!(
                parse_async_prefill_mode(bad).is_err(),
                "{bad:?} must refuse"
            );
        }
    }

    #[test]
    fn numeric_knobs_parse_or_refuse() {
        assert_eq!(parse_serving_context("32768", 262_144).unwrap(), 32_768);
        assert!(parse_serving_context("1023", 262_144).is_err());
        assert!(parse_serving_context("262145", 262_144).is_err());
        assert!(parse_serving_context("4294967296", usize::MAX).is_err());
        assert_eq!(parse_decode_slots("16").unwrap(), 16);
        assert!(parse_decode_slots("0").is_err());
        assert!(parse_decode_slots("17").is_err());
        assert_eq!(parse_prefix_cache_cap("4").unwrap(), Some(4));
        for off in ["", "0", "off", " OFF "] {
            assert_eq!(
                parse_prefix_cache_cap(off).unwrap(),
                None,
                "{off:?} is how both opt in knobs spell off"
            );
        }
        for bad in ["many", "-1", "4k"] {
            assert!(parse_prefix_cache_cap(bad).is_err(), "{bad:?} must refuse");
        }
    }

    #[test]
    fn chunk_mode_parses_or_refuses() {
        for off in ["", "0", "off", " OFF "] {
            assert_eq!(parse_mix_chunk_tokens(off, 8192).unwrap(), None);
        }
        assert_eq!(parse_mix_chunk_tokens("64", 8192).unwrap(), Some(64));
        assert_eq!(parse_mix_chunk_tokens("8192", 262_144).unwrap(), Some(8192));
        assert_eq!(
            parse_mix_chunk_tokens("2496", 262_144).unwrap(),
            Some(2432),
            "a step aligns down to whole tiles"
        );
        assert_eq!(
            parse_mix_chunk_tokens("127", 8192).unwrap(),
            Some(127),
            "below one tile the width is never rounded to zero"
        );
        assert_eq!(parse_mix_chunk_tokens("128", 8192).unwrap(), Some(128));
        for bad in ["63", "8192", "on", "2k", "-1"] {
            assert!(
                parse_mix_chunk_tokens(bad, 8192).is_err(),
                "{bad:?} must refuse"
            );
        }
    }
}

#[cfg(test)]
mod gate {
    use super::*;

    #[test]
    fn the_suppression_mask_writes_only_the_ids_it_is_given() {
        let ctx = DeviceContext::new().expect("GPU required");
        let (vocab, rows) = (8usize, 2usize);
        let mut logits = HiddenStates::zeros(&ctx, vocab, rows).expect("logits");
        let suppress_ids = ops::SuppressIds::upload(&ctx, &[3u32, 5], vocab).expect("ids");
        ops::suppress_logits_bf16_in_place(&ctx, &mut logits, &suppress_ids).expect("suppress");

        let host = logits.to_host(&ctx).expect("D2H");
        for row in 0..rows {
            for id in 0..vocab {
                let value = host[row * vocab + id];
                if id == 3 || id == 5 {
                    assert!(
                        value == f32::NEG_INFINITY,
                        "row {row} id {id} is {value}, not suppressed"
                    );
                } else {
                    assert!(value == 0.0, "row {row} id {id} moved to {value}");
                }
            }
        }

        // The bound is structural: an id the head does not span cannot reach
        // the kernel, and neither can ids checked against a different head.
        let past_the_head = ops::SuppressIds::upload(&ctx, &[vocab as u32], vocab);
        assert!(
            past_the_head.is_err(),
            "an id at the head's width must be refused at upload"
        );
        let other_head = ops::SuppressIds::upload(&ctx, &[1u32], vocab + 1).expect("ids");
        assert!(
            ops::suppress_logits_bf16_in_place(&ctx, &mut logits, &other_head).is_err(),
            "ids checked against a wider head must not be applied to these logits"
        );
    }

    #[test]
    fn the_generation_policy_refuses_ids_outside_the_head() {
        let policy = GenerationPolicy {
            eos: vec![1],
            suppress: vec![8],
        };
        let err = policy
            .check_against_vocab(8)
            .expect_err("an id past the row must be refused");
        assert!(
            err.to_string()
                .contains("effective suppression set lists 8"),
            "wrong refusal: {err:#}"
        );
    }
}

#[cfg(test)]
mod lane_tests {
    use std::path::Path;

    use pegainfer_frontend::engine::EngineLoadOptions;
    use pegainfer_frontend::engine::FinishReason;
    use pegainfer_frontend::engine::GenerateRequest;
    use pegainfer_frontend::engine::TokenEvent;
    use pegainfer_frontend::engine::TokenSink;
    use pegainfer_frontend::engine::TokenStreamReceiver;

    fn submit(
        handle: &pegainfer_frontend::engine::EngineHandle,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> TokenStreamReceiver {
        let (request, token_rx) = walk_request(prompt_tokens, max_tokens);
        handle.submit(request).expect("submit");
        token_rx
    }

    struct Drained {
        tokens: usize,
        cached: usize,
        finish: FinishReason,
    }

    fn drain(rx: &mut TokenStreamReceiver, name: &str) -> Drained {
        let mut tokens = 0;
        let mut cached = 0;
        loop {
            match rx.blocking_recv().map(|(_, event)| event) {
                Some(TokenEvent::Token { .. }) => tokens += 1,
                Some(TokenEvent::Scheduled { cached_tokens, .. }) => cached = cached_tokens,
                Some(TokenEvent::PromptTokens { .. } | TokenEvent::KvTransfer { .. }) => {}
                Some(TokenEvent::Finished { finish_reason, .. }) => {
                    return Drained {
                        tokens,
                        cached,
                        finish: finish_reason,
                    };
                }
                Some(TokenEvent::Error { message, .. }) => panic!("{name}: error: {message}"),
                Some(TokenEvent::Rejected { message, .. }) => panic!("{name}: rejected: {message}"),
                None => panic!("{name}: channel closed without Finished"),
            }
        }
    }

    fn ids(len: usize, salt: u32) -> Vec<u32> {
        (0..len as u32)
            .map(|i| 1000 + (i * 37 + salt) % 50000)
            .collect()
    }

    fn pin_live_stream(handle: &pegainfer_frontend::engine::EngineHandle) -> TokenStreamReceiver {
        let mut streamer = submit(handle, ids(40, 0), 1024);
        let mut seen = 0;
        while seen < 2 {
            match streamer.blocking_recv().map(|(_, event)| event) {
                Some(TokenEvent::Token { .. }) => seen += 1,
                Some(TokenEvent::Error { message, .. }) => panic!("streamer: {message}"),
                Some(_) => {}
                None => panic!("streamer closed early"),
            }
        }
        streamer
    }

    fn warm_prompt(prefix: &[u32]) -> Vec<u32> {
        let mut prompt = prefix.to_vec();
        prompt.extend(ids(60, 11));
        prompt
    }

    fn assert_warm_result(rx: &mut TokenStreamReceiver, cached: usize, label: &str) {
        let warm = drain(rx, label);
        assert_eq!((warm.tokens, warm.finish), (4, FinishReason::Length));
        assert_eq!(warm.cached, cached, "warm admission resume frontier");
    }

    /// The whole async lifecycle against one live engine, with every
    /// interleaving pinned by construction: the streamer's budget outlasts
    /// the script and its sink is dropped only at the end, so the decode
    /// batch is never empty and every admission below takes the lane. A
    /// long prompt rides the lane while a second arrival waits out the busy
    /// lane; a warm prefix-cache hit launches with only its unseen suffix
    /// (the join gate refuses a pass that processed the wrong one); a
    /// request cancelled after `Scheduled` — past every sync-path exit, so
    /// the drop lands while its pass is in flight — exits through the join;
    /// an out-of-vocab prompt fails the launch itself and the lane drains;
    /// and the engine still serves after all of it.
    /// Serialize the serving-knob environment across every gate and put it
    /// back afterwards — including on panic — so no gate can poison another
    /// or be poisoned by the operator's shell. Values round-trip as OsString.
    static ENV_LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();

    struct EnvGuard {
        saved: Vec<(&'static str, Option<std::ffi::OsString>)>,
        _lock: std::sync::MutexGuard<'static, ()>,
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: the guard's lock serializes all environment access
            // among gates, and the ignored suites run single-threaded.
            unsafe {
                for (k, v) in self.saved.drain(..) {
                    match v {
                        Some(v) => std::env::set_var(k, v),
                        None => std::env::remove_var(k),
                    }
                }
            }
        }
    }

    const SERVING_KNOBS: [&str; 5] = [
        "PEGAINFER_ASYNC_PREFILL",
        "PEGAINFER_PREFIX_CACHE",
        "PEGAINFER_MIX_CHUNK_TOKENS",
        "PEGAINFER_MAX_CONTEXT",
        "PEGAINFER_DECODE_SLOTS",
    ];

    /// Clear every serving knob, set `overrides`, and hand back the guard
    /// that restores the environment when it drops.
    fn scoped_engine_env(overrides: &[(&str, &str)]) -> EnvGuard {
        let lock = ENV_LOCK
            .get_or_init(|| std::sync::Mutex::new(()))
            .lock()
            .expect("env lock");
        let saved = SERVING_KNOBS
            .iter()
            .map(|k| (*k, std::env::var_os(k)))
            .collect();
        // SAFETY: as in Drop — the lock serializes environment access.
        unsafe {
            for k in SERVING_KNOBS {
                std::env::remove_var(k);
            }
            for (k, v) in overrides {
                std::env::set_var(k, v);
            }
        }
        EnvGuard { saved, _lock: lock }
    }

    /// The raise has to arrive where a client can see it. `servable_len` is
    /// the frontend's only view of the ceiling, and it travels a path no
    /// state-level assertion touches: load, the ready channel, the u32
    /// narrowing, and `EngineHandle::with_servable_len`.
    #[test]
    #[ignore = "requires the pinned 12B checkpoint, a GPU, and --test-threads=1"]
    fn the_raise_reaches_the_frontend() {
        let dir = crate::testkit::model_path();
        let _env = scoped_engine_env(&[
            ("PEGAINFER_MAX_CONTEXT", "32768"),
            ("PEGAINFER_MIX_CHUNK_TOKENS", "2048"),
            ("PEGAINFER_DECODE_SLOTS", "2"),
        ]);
        let handle =
            super::start(Path::new(&dir), &EngineLoadOptions::default()).expect("engine start");
        assert_eq!(
            handle.servable_len(),
            Some(32768),
            "the raised ceiling must reach the frontend, not just the engine state"
        );
        let mut rx = submit(&handle, ids(40, 5), 4);
        let served = drain(&mut rx, "raised ceiling");
        assert_eq!((served.tokens, served.finish), (4, FinishReason::Length));
    }

    fn lane_lifecycle_script(mode: &str) {
        let dir = crate::testkit::model_path();
        let _env = scoped_engine_env(&[
            ("PEGAINFER_ASYNC_PREFILL", mode),
            ("PEGAINFER_PREFIX_CACHE", "4"),
        ]);
        let handle =
            super::start(Path::new(&dir), &EngineLoadOptions::default()).expect("engine start");

        // The streamer pins the decode batch non-empty for the whole
        // script: its budget is far past the script's round count, and its
        // sink drops only after the last assertion.
        let streamer = pin_live_stream(&handle);

        let long_prompt = ids(1500, 7);
        let mut lane_rx = submit(&handle, long_prompt.clone(), 4);
        let mut queued_rx = submit(&handle, ids(60, 3), 4);
        let lane = drain(&mut lane_rx, "lane prefill");
        assert_eq!((lane.tokens, lane.finish), (4, FinishReason::Length));
        let queued = drain(&mut queued_rx, "queued behind the lane");
        assert_eq!((queued.tokens, queued.finish), (4, FinishReason::Length));

        // Warm hit: the long prompt plus a new turn resumes at the captured
        // frontier and rides the lane with only the suffix.
        let mut warm_rx = submit(&handle, warm_prompt(&long_prompt), 4);
        assert_warm_result(&mut warm_rx, 1500, "warm suffix on the lane");

        // In-flight cancellation: `Scheduled` is sent past every sync-path
        // exit, so a sink dropped after it lands while the lane pass runs.
        let mut cancel_rx = submit(&handle, ids(1500, 23), 8);
        loop {
            match cancel_rx.blocking_recv().map(|(_, event)| event) {
                Some(TokenEvent::Scheduled { .. }) => break,
                Some(TokenEvent::Error { message, .. }) => panic!("cancel arm: {message}"),
                Some(_) => {}
                None => panic!("cancel arm closed before Scheduled"),
            }
        }
        drop(cancel_rx);

        // Launch failure: an out-of-vocab id passes admission and fails
        // inside the lane pass; the lane drains and reports, loudly.
        let mut bad = ids(200, 31);
        bad[100] = 300_000;
        let mut bad_rx = submit(&handle, bad, 4);
        loop {
            match bad_rx.blocking_recv().map(|(_, event)| event) {
                Some(TokenEvent::Error { message, .. }) => {
                    assert!(
                        message.contains("prefill failed"),
                        "launch failure must name the prefill: {message}"
                    );
                    break;
                }
                Some(TokenEvent::Token { .. } | TokenEvent::Finished { .. }) => {
                    panic!("an out-of-vocab prompt must not produce tokens")
                }
                Some(_) => {}
                None => panic!("launch failure lost its error event"),
            }
        }

        let mut after_rx = submit(&handle, ids(40, 29), 4);
        let after = drain(&mut after_rx, "after the cancelled and failed lanes");
        assert_eq!((after.tokens, after.finish), (4, FinishReason::Length));
        drop(streamer);
    }

    fn walk_request(prompt: Vec<u32>, max_tokens: usize) -> (GenerateRequest, TokenStreamReceiver) {
        let (token_tx, token_rx) = TokenSink::standalone();
        (
            GenerateRequest {
                trace_parent: None,
                request_id: None,
                queued_at_unix_s: None,
                data_parallel_rank: None,
                prompt_tokens: prompt,
                params: pegainfer_frontend::sampler::SamplingParams {
                    ignore_eos: true,
                    ..pegainfer_frontend::sampler::SamplingParams::default()
                },
                max_tokens,
                lora_adapter: None,
                kv_transfer_params: None,
                token_tx,
                logprobs: 0,
                echo: false,
            },
            token_rx,
        )
    }

    fn walk_drain(rx: &mut TokenStreamReceiver, name: &str, into: &mut Vec<u32>) {
        loop {
            match rx.blocking_recv().map(|(_, event)| event) {
                Some(TokenEvent::Token { id, .. }) => into.push(id),
                Some(TokenEvent::Finished { .. }) => return,
                Some(TokenEvent::Error { message, .. } | TokenEvent::Rejected { message, .. }) => {
                    panic!("{name}: {message}")
                }
                Some(_) => {}
                None => panic!("{name}: channel closed without Finished"),
            }
        }
    }

    /// A single-request episode through the plain admission path — the
    /// engine's own serial reference: same suppression, same sampler.
    fn walk_serial(state: &mut super::EngineState, prompt: &[u32], budget: usize) -> Vec<u32> {
        let (request, mut rx) = walk_request(prompt.to_vec(), budget);
        let mut active = Vec::new();
        let mut pending = std::collections::VecDeque::new();
        let mut attempts = 0usize;
        match state.admit_and_prefill(
            (request, pegainfer_frontend::engine::KvPrefix::none()),
            false,
            &mut active,
            &mut pending,
            &mut attempts,
        ) {
            super::Admitted::Active(entry) => active.push(*entry),
            super::Admitted::Done => {}
            super::Admitted::Requeue(_) => panic!("serial episode requeued"),
        }
        while !active.is_empty() {
            state.decode_round(&mut active);
        }
        let mut out = Vec::new();
        walk_drain(&mut rx, "serial episode", &mut out);
        out
    }

    fn walk_prompts() -> Vec<Vec<u32>> {
        crate::testkit::generate_fixture_prompts()
    }

    fn headroom_admit(
        state: &mut super::EngineState,
        prompt: Vec<u32>,
        max_tokens: usize,
        active: &mut Vec<super::Active>,
        pending: &mut std::collections::VecDeque<pegainfer_frontend::engine::SubmittedRequest>,
    ) -> TokenStreamReceiver {
        let (request, rx) = walk_request(prompt, max_tokens);
        let mut attempts = 0usize;
        match state.admit_and_prefill(
            (request, pegainfer_frontend::engine::KvPrefix::none()),
            false,
            active,
            pending,
            &mut attempts,
        ) {
            super::Admitted::Active(entry) => active.push(*entry),
            super::Admitted::Done => {}
            super::Admitted::Requeue(_) => panic!("headroom admission requeued"),
        }
        rx
    }

    /// Load an engine with the chunk knob set ahead of the load — the pool
    /// is sized by it — while the other serving knobs are cleared for the
    /// duration, so the outside environment cannot change the pool or the
    /// admission route under the test.
    fn walk_test_state(chunk: &str) -> super::EngineState {
        let dir = crate::testkit::model_path();
        let policy = super::generation_policy(&dir).expect("policy");
        let _env = scoped_engine_env(&[("PEGAINFER_MIX_CHUNK_TOKENS", chunk)]);
        super::EngineState::load(&dir, 0, policy, 0x5EED, true).expect("engine state")
    }

    /// A raise without the chunk knob, and a raise with the overlap lane,
    /// both refuse before the multi-GiB load — the startup policy the
    /// serving doc promises.
    #[test]
    #[ignore = "requires the pinned 12B checkpoint and --test-threads=1"]
    fn the_raise_refuses_without_its_prerequisites() {
        let dir = crate::testkit::model_path();
        let load = |overrides: &[(&str, &str)]| {
            let policy = super::generation_policy(&dir).expect("policy");
            let _env = scoped_engine_env(overrides);
            super::EngineState::load(&dir, 0, policy, 0x5EED, true)
        };
        let err = load(&[("PEGAINFER_MAX_CONTEXT", "32768")])
            .err()
            .expect("a raise without the chunk knob must refuse");
        assert!(
            format!("{err:#}").contains("needs PEGAINFER_MIX_CHUNK_TOKENS"),
            "unexpected refusal: {err:#}"
        );
        let err = load(&[
            ("PEGAINFER_MAX_CONTEXT", "32768"),
            ("PEGAINFER_MIX_CHUNK_TOKENS", "2048"),
            ("PEGAINFER_ASYNC_PREFILL", "green:35"),
        ])
        .err()
        .expect("the lane over the default ceiling must refuse");
        assert!(
            format!("{err:#}").contains("unsupported over"),
            "unexpected refusal: {err:#}"
        );
    }

    /// The slots boundary, driven at the roster edge the engine loop owns
    /// — the same intake method, single-threaded, no clocks, no
    /// thresholds: with both slots held by live requests an intake pass
    /// leaves the third queued with no Scheduled; once an incumbent
    /// retires, the same intake admits it and it runs to its budget.
    #[test]
    #[ignore = "requires the pinned 12B checkpoint, a GPU, and --test-threads=1"]
    fn the_raised_ceiling_and_slots_hold_at_the_roster_edge() {
        let dir = crate::testkit::model_path();
        let policy = super::generation_policy(&dir).expect("policy");
        let _env = scoped_engine_env(&[
            ("PEGAINFER_MAX_CONTEXT", "32768"),
            ("PEGAINFER_MIX_CHUNK_TOKENS", "2048"),
            ("PEGAINFER_DECODE_SLOTS", "2"),
        ]);
        let mut state =
            super::EngineState::load(&dir, 0, policy, 0x5EED, true).expect("engine state");
        assert_eq!(
            state.max_context, 32768,
            "the engine follows the raised ceiling"
        );
        let prompts = walk_prompts();
        let long: Vec<u32> = prompts[0].iter().cycle().copied().take(12000).collect();
        let (req_a, mut rx_a) = walk_request(long, 4);
        let (req_b, mut rx_b) = walk_request(prompts[1].clone(), 40);
        let (req_c, mut rx_c) = walk_request(prompts[2].clone(), 6);

        let mut pending = std::collections::VecDeque::new();
        let mut active: Vec<super::Active> = Vec::new();
        pending.push_back((req_a, pegainfer_frontend::engine::KvPrefix::none()));
        pending.push_back((req_b, pegainfer_frontend::engine::KvPrefix::none()));
        state.admit_from_queue(&mut pending, &mut active);
        assert_eq!(active.len(), 2, "both slots are held by live requests");
        assert!(pending.is_empty(), "nothing waits yet");

        pending.push_back((req_c, pegainfer_frontend::engine::KvPrefix::none()));
        state.admit_from_queue(&mut pending, &mut active);
        assert_eq!(
            pending.len(),
            1,
            "the intake leaves the third request queued at full slots"
        );
        assert!(
            rx_c.try_recv().is_err(),
            "no Scheduled while both slots are held"
        );

        while active.len() == 2 {
            state.decode_round(&mut active);
        }
        state.admit_from_queue(&mut pending, &mut active);
        assert_eq!(active.len(), 2, "the freed slot admits the third request");
        assert!(pending.is_empty(), "the queue drained");
        while !active.is_empty() {
            state.decode_round(&mut active);
        }
        assert_eq!(drain(&mut rx_a, "incumbent a").tokens, 4);
        assert_eq!(drain(&mut rx_b, "incumbent b").tokens, 40);
        assert_eq!(drain(&mut rx_c, "queued third").tokens, 6);
    }

    /// The chunked pool provisions one shared segment transient, so no
    /// walker may park pages ahead of its rounds: with the knob set before
    /// load — the reduced production pool, asserted against the provision
    /// arithmetic — twelve streams at full window plus three near-context
    /// prompts entering one gather must all finish. Parked first quanta
    /// across the walkers exhausted this pool.
    #[test]
    #[ignore = "requires the pinned 12B checkpoint, a GPU, and --test-threads=1"]
    fn the_gathered_transient_leaves_headroom() {
        let dir = crate::testkit::model_path();
        let mut state = walk_test_state("2048");
        assert_eq!(state.mix_chunk, Some(2048), "the knob preceded the load");
        assert_eq!(state.max_context, 8192, "the ceiling default held");
        assert_eq!(state.slots, 16, "the slots default held");
        let window = crate::config::Gemma4Config::from_file(&dir)
            .expect("config")
            .sliding_window;
        let window_pages = window.div_ceil(crate::kv::PAGE_SIZE) + 1;
        let provisioned = window_pages
            + 2048usize.div_ceil(crate::kv::PAGE_SIZE)
            + (super::MIX_MAX_PROMPTS - 1)
            + (super::MAX_CONCURRENCY - 1) * window_pages;
        assert_eq!(
            state.serve.local_pool.available_pages(),
            provisioned,
            "the pool was sized by the reduced chunked provision"
        );

        let prompts = walk_prompts();
        let stream_prompt: Vec<u32> = prompts[0].iter().cycle().copied().take(1500).collect();
        let long: Vec<u32> = prompts[0].iter().cycle().copied().take(5900).collect();

        let mut active: Vec<super::Active> = Vec::new();
        let mut pending = std::collections::VecDeque::new();
        let mut stream_rx: Vec<TokenStreamReceiver> = (0..12)
            .map(|_| {
                headroom_admit(
                    &mut state,
                    stream_prompt.clone(),
                    60,
                    &mut active,
                    &mut pending,
                )
            })
            .collect();
        assert_eq!(active.len(), 12, "every stream holds a decode slot");

        let mut long_rx: Vec<TokenStreamReceiver> = Vec::new();
        let (first_long, first_rx) = walk_request(long.clone(), 2);
        long_rx.push(first_rx);
        for _ in 0..2 {
            let (request, rx) = walk_request(long.clone(), 2);
            pending.push_back((request, pegainfer_frontend::engine::KvPrefix::none()));
            long_rx.push(rx);
        }
        let mut attempts = 0usize;
        match state.admit_and_prefill(
            (first_long, pegainfer_frontend::engine::KvPrefix::none()),
            false,
            &mut active,
            &mut pending,
            &mut attempts,
        ) {
            super::Admitted::Done => {}
            _ => panic!("the gathered walk must land every walker in the batch"),
        }
        assert!(
            pending.is_empty(),
            "all three newcomers must enter one gather"
        );
        // Walkers graduate at their final segment's round and may retire
        // through their two-token budgets before the walk returns; the
        // streams must all still be decoding, and the drains below hold
        // every request to its full budget.
        assert!(
            active.len() >= 12,
            "a stream lost its slot during the walk: {} active",
            active.len()
        );
        while !active.is_empty() {
            state.decode_round(&mut active);
        }
        for (i, rx) in long_rx.iter_mut().enumerate() {
            let mut produced = Vec::new();
            walk_drain(rx, &format!("long prompt {i}"), &mut produced);
            assert_eq!(produced.len(), 2, "long prompt {i} reached its budget");
        }
        for (i, rx) in stream_rx.iter_mut().enumerate() {
            let mut produced = Vec::new();
            walk_drain(rx, &format!("stream {i}"), &mut produced);
            assert_eq!(produced.len(), 60, "stream {i} reached its budget");
        }
    }

    /// One shared walk against the engine's own serial path, on the reduced
    /// production pool with the 64-row production floor: three fixture
    /// prompts get their reference sequences from single-request episodes
    /// through the production admission path, then a rider decodes while
    /// the other two enter one gathered walk, both walk again from an idle
    /// roster (the segment-by-segment tail path) at a 24-row span so
    /// boundary rounds carry a final and a non-final segment, and a walker
    /// cancelled before the first round is dropped without touching its
    /// partner or the rider. Every surviving request's greedy sequence
    /// must match its reference whole. The direct `mixed_walk` calls in
    /// the later phases stage what the production loop cannot pin
    /// deterministically — a drained roster, a mid-walk disconnect, a
    /// span below the knob floor — with the production zero-ahead
    /// reservation shape: nothing is admitted before the walk. Those
    /// phases are algorithm oracles for `mixed_walk` itself, not serving
    /// evidence: the public floor rejects spans under 64.
    #[test]
    #[ignore = "requires the pinned 12B checkpoint, a GPU, and --test-threads=1"]
    fn the_gathered_walk_matches_the_serial_path() {
        let mut state = walk_test_state("64");
        assert_eq!(state.mix_chunk, Some(64), "the knob preceded the load");
        let prompts = walk_prompts();
        let cases = ["a", "b", "c"];
        let budgets = [24usize, 17, 21];

        let serial: Vec<Vec<u32>> = prompts
            .iter()
            .zip(budgets)
            .map(|(prompt, budget)| walk_serial(&mut state, prompt, budget))
            .collect();
        for (i, sequence) in serial.iter().enumerate() {
            assert_eq!(
                sequence.len(),
                budgets[i],
                "serial episode {i} reached its budget"
            );
        }

        // Rider live, walkers b and c entering one production gather.
        let mut active: Vec<super::Active> = Vec::new();
        let mut pending = std::collections::VecDeque::new();
        let mut rx_a = headroom_admit(
            &mut state,
            prompts[0].clone(),
            budgets[0],
            &mut active,
            &mut pending,
        );
        assert_eq!(active.len(), 1, "the rider holds a decode slot");

        let (req_b, mut rx_b) = walk_request(prompts[1].clone(), budgets[1]);
        let (req_c, mut rx_c) = walk_request(prompts[2].clone(), budgets[2]);
        pending.push_back((req_c, pegainfer_frontend::engine::KvPrefix::none()));
        let mut attempts = 0usize;
        match state.admit_and_prefill(
            (req_b, pegainfer_frontend::engine::KvPrefix::none()),
            false,
            &mut active,
            &mut pending,
            &mut attempts,
        ) {
            super::Admitted::Done => {}
            _ => panic!("the gathered walk must land every walker in the batch"),
        }
        assert!(pending.is_empty(), "both walkers must enter one gather");
        assert_eq!(active.len(), 3, "rider plus both walkers keep decoding");
        while !active.is_empty() {
            state.decode_round(&mut active);
        }
        let mut produced: Vec<Vec<u32>> = vec![Vec::new(); 3];
        walk_drain(&mut rx_a, "rider", &mut produced[0]);
        walk_drain(&mut rx_b, "walker b", &mut produced[1]);
        walk_drain(&mut rx_c, "walker c", &mut produced[2]);
        for (i, case) in cases.iter().enumerate() {
            assert_eq!(
                produced[i], serial[i],
                "case {case}: the gathered walk diverged from the serial path"
            );
            eprintln!("case {case}: {} tokens gathered == serial", serial[i].len());
        }

        // The same walk from an idle roster — the tails feed segment by
        // segment through the drained-roster path, nothing admitted ahead.
        let (req_a2, mut rx_a2) = walk_request(prompts[0].clone(), budgets[0]);
        let (req_b2, mut rx_b2) = walk_request(prompts[1].clone(), budgets[1]);
        let kv_a2 = state.serve.alloc_kv();
        let kv_b2 = state.serve.alloc_kv();
        let mut active2: Vec<super::Active> = Vec::new();
        let admitted2 = state.mixed_walk(
            24,
            vec![(req_a2, kv_a2, None), (req_b2, kv_b2, None)],
            &mut active2,
        );
        assert!(
            matches!(admitted2, super::Admitted::Done),
            "the idle walk returns Done"
        );
        assert_eq!(active2.len(), 2, "both idle walkers keep decoding");
        while !active2.is_empty() {
            state.decode_round(&mut active2);
        }
        let mut produced2: Vec<Vec<u32>> = vec![Vec::new(); 2];
        walk_drain(&mut rx_a2, "idle walker a", &mut produced2[0]);
        walk_drain(&mut rx_b2, "idle walker b", &mut produced2[1]);
        for i in 0..2 {
            assert_eq!(
                produced2[i], serial[i],
                "idle case {}: the walk diverged from the serial path",
                cases[i]
            );
        }

        // A walker whose client disconnected before the first round is
        // dropped without consuming rounds — while a rider decodes through
        // them — and the survivors still match the serial path.
        let (req_c3, mut rx_c3) = walk_request(prompts[2].clone(), budgets[2]);
        let mut pending3 = std::collections::VecDeque::new();
        let mut attempts3 = 0usize;
        let mut active3: Vec<super::Active> = Vec::new();
        match state.admit_and_prefill(
            (req_c3, pegainfer_frontend::engine::KvPrefix::none()),
            false,
            &mut active3,
            &mut pending3,
            &mut attempts3,
        ) {
            super::Admitted::Active(entry) => active3.push(*entry),
            _ => panic!("phase three rider admission must hand back an active lane"),
        }
        let (req_a3, rx_a3) = walk_request(prompts[0].clone(), budgets[0]);
        let (req_b3, mut rx_b3) = walk_request(prompts[1].clone(), budgets[1]);
        drop(rx_a3);
        let kv_a3 = state.serve.alloc_kv();
        let kv_b3 = state.serve.alloc_kv();
        state.ready_decode_rows(&mut active3);
        let admitted3 = state.mixed_walk(
            24,
            vec![(req_a3, kv_a3, None), (req_b3, kv_b3, None)],
            &mut active3,
        );
        assert!(
            matches!(admitted3, super::Admitted::Done),
            "the walk with a cancelled walker returns Done"
        );
        assert_eq!(
            active3.len(),
            2,
            "the rider and the surviving walker keep decoding"
        );
        while !active3.is_empty() {
            state.decode_round(&mut active3);
        }
        let mut produced3 = Vec::new();
        walk_drain(&mut rx_b3, "surviving walker", &mut produced3);
        assert_eq!(
            produced3, serial[1],
            "the surviving walker diverged from the serial path"
        );
        let mut produced_rider3 = Vec::new();
        walk_drain(&mut rx_c3, "phase three rider", &mut produced_rider3);
        assert_eq!(
            produced_rider3, serial[2],
            "the phase three rider diverged from the serial path"
        );
    }

    /// The production gather against one live engine, interleavings pinned:
    /// the streamer's budget outlasts the script and its sink drops last,
    /// so the decode batch is never empty and every admission takes the
    /// gather path. Dead and invalid submissions queued ahead of a valid
    /// prompt drain within the shared admission budget without wedging the
    /// engine; a warm prefix-cache hit whose rendered prompt exceeds the
    /// gather ceiling but whose unseen suffix does not still resolves and
    /// completes; and every survivor finishes its full budget.
    fn gather_lifecycle_script() {
        let dir = crate::testkit::model_path();
        let _env = scoped_engine_env(&[("PEGAINFER_PREFIX_CACHE", "4")]);
        let handle =
            super::start(Path::new(&dir), &EngineLoadOptions::default()).expect("engine start");

        let streamer = pin_live_stream(&handle);

        // Dead and invalid submissions ahead of a valid prompt: the gather
        // drains them against the shared budget and the valid one lands.
        drop(submit(&handle, ids(50, 1), 4));
        drop(submit(&handle, ids(50, 2), 4));
        let mut invalid_rx = submit(&handle, Vec::new(), 4);
        let mut valid_rx = submit(&handle, ids(60, 3), 4);
        loop {
            match invalid_rx.blocking_recv().map(|(_, event)| event) {
                Some(TokenEvent::Rejected { message, .. }) => {
                    assert!(message.contains("empty prompt"), "unexpected: {message}");
                    break;
                }
                Some(TokenEvent::Token { .. } | TokenEvent::Finished { .. }) => {
                    panic!("an empty prompt must be rejected")
                }
                Some(_) => {}
                None => panic!("invalid submission lost its rejection"),
            }
        }
        let valid = drain(&mut valid_rx, "valid after the corpses");
        assert_eq!((valid.tokens, valid.finish), (4, FinishReason::Length));

        // Warm gather: a captured long conversation resumes with a short
        // suffix while a short primary sits ahead in the same intake — the
        // rendered prompt is far past the gather ceiling, its unseen suffix
        // far under it.
        let long_prompt = ids(1600, 7);
        let mut long_rx = submit(&handle, long_prompt.clone(), 4);
        let long_done = drain(&mut long_rx, "long prefill");
        assert_eq!(
            (long_done.tokens, long_done.finish),
            (4, FinishReason::Length)
        );
        let mut head_rx = submit(&handle, ids(60, 13), 4);
        let mut warm_rx = submit(&handle, warm_prompt(&long_prompt), 4);
        let head = drain(&mut head_rx, "gather head");
        assert_eq!((head.tokens, head.finish), (4, FinishReason::Length));
        assert_warm_result(&mut warm_rx, 1600, "warm gathered suffix");

        drop(streamer);
    }

    #[test]
    #[ignore = "requires the pinned 12B checkpoint, a GPU, and --test-threads=1"]
    fn the_engine_lifecycle_variants_complete() {
        lane_lifecycle_script("shared");
        lane_lifecycle_script("green:35");
        gather_lifecycle_script();
    }

    #[test]
    fn pool_pages_follow_the_knobs() {
        // The documented defaults: 8192 ceiling, 16 slots, no cache.
        assert_eq!(
            super::pool_pages(512, 65, 512, 16, 0, 256),
            Some((1488, 8193))
        );
        // Raised 32768 x 2 slots with a 2048 chunk: transient = W + 128 + 3.
        assert_eq!(
            super::pool_pages(196, 65, 2048, 2, 0, 1024),
            Some((262, 4097))
        );
        assert_eq!(super::pool_pages(usize::MAX, 65, 512, 16, 0, 256), None);
    }

    #[test]
    fn the_global_door_is_defensive() {
        // Validation caps every context inside the ceiling and the pool
        // provisions slots times the ceiling's pages, so no request's whole
        // account can exceed its slot's share: the door only ever fires on
        // an accounting bug, which is exactly its job.
        for ceiling in [8192usize, 32768, 262_144] {
            let provision_per_slot = ceiling.div_ceil(super::PAGE_SIZE);
            for context_len in [1usize, 17, ceiling / 2, ceiling] {
                assert!(super::global_account_pages(context_len) <= provision_per_slot);
            }
        }
    }
}
