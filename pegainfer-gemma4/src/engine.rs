//! The Gemma 4 engine: one contract-driven scheduler with iteration-level
//! scheduling.
//! Prefill runs at the step boundary, whole unless the chunk knob splits it
//! (the overlap lane always prefills whole);
//! every active request then
//! advances one token per batched decode step, sharing each layer's weight
//! pass.

use std::collections::VecDeque;
use std::path::Path;

use anyhow::Context as AnyhowContext;
use anyhow::Result;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_core::kv_pool::KvStorage;
use pegainfer_core::ops;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::engine::Engine;
use pegainfer_frontend::engine::EngineInfo;
use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_frontend::engine::FinishReason;
use pegainfer_frontend::engine::PromptEcho;
use pegainfer_frontend::engine::QueuedRequest;
use pegainfer_frontend::engine::RejectReason;
use pegainfer_frontend::engine::Request;
use pegainfer_frontend::engine::RequestId;
use pegainfer_frontend::engine::RequestLedger;
use pegainfer_frontend::engine::Scheduler;
use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_frontend::engine::spawn_scheduler;
use pegainfer_sample::LogprobRequest;
use pegainfer_sample::SampleScratch;

use crate::config::TensorParallelConfig;
use crate::forward::MULTIMODAL_PLACEHOLDER_IDS;
use crate::kv::GLOBAL_PAGE_SIZE;
use crate::kv::GemmaKv;
use crate::kv::LOCAL_PAGE_SIZE;
use crate::kv::RankKv;
use crate::kv::admit_tokens;
use crate::layer::LayerGeometry;
use crate::prefix_cache::PrefixCache;
use crate::serve::GemmaServe;
use crate::serve::GlobalAttn;
use crate::serve::PrecapturePhase;
use crate::serve::StepArena;
use crate::weights::Gemma4Weights;

/// The default serving ceiling.
const MAX_CONTEXT: usize = 8192;

/// Decode-batch ceiling: bounds the step buffers, the sampling scratch and
/// the pool budget. Admission beyond it queues at the step boundary rather
/// than rejecting.
const MAX_CONCURRENCY: usize = 16;
const ASYNC_PREFILL_ENV: &str = "PEGAINFER_ASYNC_PREFILL";
const PREFIX_CACHE_ENV: &str = "PEGAINFER_PREFIX_CACHE";
const MIX_CHUNK_TOKENS_ENV: &str = "PEGAINFER_MIX_CHUNK_TOKENS";
const MIX_GATHER_ROWS_ENV: &str = "PEGAINFER_MIX_GATHER_ROWS";
const MIX_MAX_PROMPTS_ENV: &str = "PEGAINFER_MIX_MAX_PROMPTS";
const MAX_CONTEXT_ENV: &str = "PEGAINFER_MAX_CONTEXT";
const DECODE_SLOTS_ENV: &str = "PEGAINFER_DECODE_SLOTS";
const KV_FP8_ENV: &str = "PEGAINFER_KV_FP8";
const ADMIT_COALESCE_ENV: &str = "PEGAINFER_ADMIT_COALESCE_MS";
const GLOBAL_ATTN_ENV: &str = "PEGAINFER_GLOBAL_ATTN";
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

pub(crate) fn read_env(name: &str) -> Result<Option<String>> {
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

fn parse_decode_slots(raw: &str) -> Result<usize> {
    match raw.trim().parse::<usize>() {
        Ok(value) if (1..=MAX_CONCURRENCY).contains(&value) => Ok(value),
        _ => anyhow::bail!(
            "{DECODE_SLOTS_ENV}={raw:?} not recognized (N, 1 <= N <= {MAX_CONCURRENCY})"
        ),
    }
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

/// Refuse a checkpoint the generated bodies have no kernel for: the launcher
/// answers `cudaErrorInvalidValue` for another geometry, and it would answer
/// on the first global prefill.
pub(crate) fn tilelang_geometry_refusal(config: &crate::config::Gemma4Config) -> Result<()> {
    if !pegainfer_kernels::ops::gemma4_hd512_prefill_is_built() {
        return Ok(());
    }
    let (heads, kv_heads, head_dim, page) = pegainfer_kernels::ops::gemma4_hd512_prefill_geometry()
        .context(
            "the build carries generated kernels but does not state the geometry they were \
             compiled for; regenerate the TileLang directory with the current generator",
        )?;
    let theirs = (
        config.num_attention_heads,
        config.num_global_key_value_heads,
        config.global_head_dim,
        crate::kv::GLOBAL_PAGE_SIZE,
    );
    anyhow::ensure!(
        theirs == (heads, kv_heads, head_dim, page),
        "{GLOBAL_ATTN_ENV} asks for kernels compiled for {heads} query heads over \
         {kv_heads} KV heads at head dim {head_dim} on {page}-row pages, but this \
         checkpoint's global family is {} over {} at {} on {}-row pages; serve it \
         through the incumbent kernel",
        theirs.0,
        theirs.1,
        theirs.2,
        theirs.3
    );
    Ok(())
}

/// Refuse a device the generated bodies cannot run on: generation targets one
/// arch, whose accelerated target runs on that capability alone, and a block
/// opts into more shared memory than some architectures of the same number
/// grant.
fn ensure_tilelang_device(device: usize) -> Result<()> {
    if !pegainfer_kernels::ops::gemma4_hd512_prefill_is_built() {
        return Ok(());
    }
    let arch = pegainfer_kernels::ops::gemma4_hd512_prefill_arch().context(
        "the build carries generated kernels but does not state the arch they were built \
         for; regenerate the TileLang directory with the current generator",
    )?;
    let built: u32 = arch
        .trim_start_matches("sm_")
        .trim_end_matches(|c: char| !c.is_ascii_digit())
        .parse()
        .with_context(|| format!("the build reported an unreadable TileLang arch {arch:?}"))?;
    let ctx = DeviceContext::new_with_device(device)
        .with_context(|| format!("open device {device} for the TileLang arch check"))?;
    let major = ctx.ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    )?;
    let minor = ctx.ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
    )?;
    let running = u32::try_from(major * 10 + minor).context("compute capability fits u32")?;
    anyhow::ensure!(
        running == built,
        "{GLOBAL_ATTN_ENV} asks for kernels built for {arch}, but device {device} is \
         SM{major}.{minor}: the generated bodies carry an image for one arch. Build with \
         PEGAINFER_CUDA_SM={running}, or serve this device through the incumbent kernel"
    );
    let wanted = pegainfer_kernels::ops::gemma4_hd512_prefill_smem().context(
        "the build carries generated kernels but does not state the shared memory they opt \
         into; regenerate the TileLang directory with the current generator",
    )?;
    let granted = ctx.ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
    )?;
    let granted = usize::try_from(granted).context("shared-memory limit fits usize")?;
    anyhow::ensure!(
        wanted <= granted,
        "{GLOBAL_ATTN_ENV} asks for kernels whose block opts into {wanted} B of shared \
         memory, and device {device} grants {granted} B per block; serve it through the \
         incumbent kernel"
    );
    Ok(())
}

fn parse_global_attn(raw: &str) -> Result<GlobalAttn> {
    let value = raw.trim().to_ascii_lowercase();
    match value.as_str() {
        "" | "0" | "off" => Ok(GlobalAttn::Incumbent),
        "tilelang" => Ok(GlobalAttn::TileLang),
        "tilelang640" => Ok(GlobalAttn::TileLangFolded),
        _ => {
            anyhow::bail!("{GLOBAL_ATTN_ENV}={raw:?} not recognized (off | tilelang | tilelang640)")
        }
    }
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

pub(crate) fn local_kv_storage(
    lookup: &dyn Fn(&str) -> Result<Option<String>>,
) -> Result<KvStorage> {
    lookup(KV_FP8_ENV)?.map_or(Ok(KvStorage::Bf16), |raw| parse_kv_fp8(&raw))
}

fn parse_kv_fp8(raw: &str) -> Result<KvStorage> {
    let value = raw.trim().to_ascii_lowercase();
    match value.as_str() {
        "" | "0" | "off" => Ok(KvStorage::Bf16),
        "local" => Ok(KvStorage::E4m3),
        _ => anyhow::bail!("{KV_FP8_ENV}={raw:?} not recognized (off | local)"),
    }
}

fn parse_admit_coalesce_ms(raw: &str) -> Result<Option<std::time::Duration>> {
    let value = raw.trim().to_ascii_lowercase();
    match value.as_str() {
        "" | "0" | "off" => Ok(None),
        other => match other.parse::<u64>() {
            Ok(ms) if (1..=2000).contains(&ms) => Ok(Some(std::time::Duration::from_millis(ms))),
            _ => anyhow::bail!(
                "{ADMIT_COALESCE_ENV}={raw:?} not recognized (off | N ms, 1 <= N <= 2000)"
            ),
        },
    }
}

/// Holds arrivals that would invade a live decode batch so one window's
/// arrivals land as a back-to-back burst of admissions: the stream's tail
/// gap prices the number of interruptions. One mixed step merges extra
/// prompts only with chunking or while the gathered rows stay under the
/// gather bound. The cohort is the prompt bound capped by free slots, not a
/// batch across completions; idle engines admit on sight and shallow batches
/// skip.
struct CoalesceDoor {
    window: std::time::Duration,
    max_prompts: usize,
    since: Option<std::time::Instant>,
}

impl CoalesceDoor {
    fn new(window: std::time::Duration, max_prompts: usize) -> Self {
        Self {
            window,
            max_prompts,
            since: None,
        }
    }

    fn opens(
        &mut self,
        pending: usize,
        active: usize,
        slots: usize,
        now: std::time::Instant,
    ) -> bool {
        if pending == 0 || active == 0 || (active + pending) * 2 < slots {
            self.since = None;
            return true;
        }
        let cohort = self.max_prompts.min(slots.saturating_sub(active)).max(1);
        let since = *self.since.get_or_insert(now);
        let open = pending >= cohort || now.duration_since(since) >= self.window;
        if open {
            self.since = None;
        }
        open
    }
}

pub(crate) fn start(model_path: &Path, options: &EngineLoadOptions) -> Result<Engine> {
    start_with_knobs(model_path, options, &read_env)
}

fn start_with_knobs(
    model_path: &Path,
    options: &EngineLoadOptions,
    lookup: &dyn Fn(&str) -> Result<Option<String>>,
) -> Result<Engine> {
    let dir = model_path
        .to_str()
        .context("model path is not valid UTF-8")?
        .to_string();
    let ordinals = options.device_ordinals.clone();
    anyhow::ensure!(
        !ordinals.is_empty(),
        "gemma4 needs at least one device ordinal"
    );
    let world = ordinals.len();
    if let Some(parallel) = options.parallel_config.as_ref() {
        anyhow::ensure!(
            parallel.tp_world() == world
                && parallel.dp_world() == 1
                && parallel.ep_world() == world,
            "gemma4 tensor parallelism takes one rank per device (tp_size = ep_size = {world}, \
             dp_size = 1); got tp {} dp {} ep {}",
            parallel.tp_world(),
            parallel.dp_world(),
            parallel.ep_world()
        );
    }
    let base_seed = options.seed;
    let graph_enabled = options.enable_cuda_graph;

    let config = crate::config::Gemma4Config::from_file(&dir)?;
    let knobs = ServingKnobs::resolve(lookup, &config)?;
    let policy = generation_policy(&dir)?;
    if world > 1 {
        anyhow::ensure!(
            knobs.lane_mode.is_none(),
            "{ASYNC_PREFILL_ENV} is unsupported under tensor parallelism: its lane stream cannot \
             be lock-stepped across ranks"
        );
        anyhow::ensure!(
            !knobs.global_attn.tilelang(),
            "{GLOBAL_ATTN_ENV} is unsupported under tensor parallelism: the generated kernels are \
             compiled for the whole global family"
        );
        anyhow::ensure!(
            knobs.prefix_cache.is_none(),
            "{PREFIX_CACHE_ENV} is unsupported under tensor parallelism"
        );
        // The ranks take ordinals `0..world` of what this process can see, so a
        // world size past the visible devices can only fail later, less clearly.
        let visible = cudarc::driver::result::device::get_count().map_err(|e| {
            anyhow::anyhow!("a {world}-rank launch needs to count CUDA devices: {e:?}")
        })?;
        anyhow::ensure!(
            visible as usize >= world,
            "a {world}-rank launch needs {world} CUDA devices, this process sees {visible}"
        );
        // `docs/models/gemma4/tp.md` records that this container's stock NCCL
        // corrupts memory under the default `LL`/`SIMPLE` protocols: the
        // all-reduce kernel writes far past its buffer, and the symptom is the
        // primary's collective stalling forever rather than an error. Nothing in
        // the stack can detect that, so the configuration in force is said out
        // loud rather than assumed.
        let proto = std::env::var("NCCL_PROTO").unwrap_or_else(|_| "unset".to_string());
        let version = cudarc::nccl::result::get_nccl_version().map_or_else(
            |_| "unknown".to_string(),
            |v| format!("{}.{}.{}", v / 10000, (v / 100) % 100, v % 100),
        );
        if proto.to_ascii_uppercase().contains("LL128") {
            log::info!("tensor parallel: {world} ranks, NCCL {version}, NCCL_PROTO={proto}");
        } else {
            log::warn!(
                "tensor parallel: {world} ranks, NCCL {version}, NCCL_PROTO={proto} — this \
                 environment's stock NCCL corrupts the all-reduce buffer unless the protocol \
                 is LL128 (see docs/models/gemma4/tp.md)"
            );
        }
    }

    let state = EngineState::load(
        &dir,
        &config,
        knobs,
        &ordinals,
        policy,
        base_seed,
        graph_enabled,
    )?;
    let servable = state.max_context;
    // Publishing the real ceiling is what lets the frontend refuse an
    // over-length request with its own message instead of forwarding one the
    // engine can only fail mid-stream.
    // The ceiling domain keeps the value inside i32, so this API-boundary
    // conversion cannot fail.
    let servable = u32::try_from(servable).expect("ceiling domain holds servable inside u32");
    let scheduler = Gemma4Scheduler::new(state);
    Ok(Engine {
        schedulers: vec![spawn_scheduler("gemma4-engine", scheduler)],
        info: EngineInfo {
            kv_capacity: None,
            servable_len: Some(servable),
        },
        lora: None,
    })
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

/// One in-flight overlapped prefill, parked until the lane's completion
/// event fires: the request, its KV, and the pass owning every device
/// buffer the in-flight kernels still read.
struct InflightPrefill {
    request: QueuedRequest,
    kv: GemmaKv,
    pass: crate::serve::PrefillPass,
    /// The cache entry this request resumed from, if any — its stale
    /// ancestor at capture time.
    resumed: Option<u64>,
}

/// Overlapped admission: with the lane on, a prompt arriving into a live
/// decode batch prefills on its own stream while decode steps keep
/// replaying on `ctx.stream` — the admission costs the streams a slowdown
/// instead of a mixed step per prompt. `shared` lets the prefill grids
/// compete for every SM; `green:NN` pins the lane to NN% of them, which is
/// what actually protects decode ITL.
///
/// The lane itself: a dedicated prefill stream and a reusable completion
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
    /// query error is engine-fatal: the driver writes off every request and
    /// exits instead of guessing whether the pass is safe to join.
    fn inflight_complete(&self) -> Result<bool> {
        debug_assert!(self.inflight.is_some());
        let query = unsafe { cudarc::driver::sys::cuEventQuery(self.event.cu_event()) };
        match query {
            cudarc::driver::sys::CUresult::CUDA_SUCCESS => Ok(true),
            cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => Ok(false),
            other => anyhow::bail!("cuEventQuery(prefill) failed ({other:?})"),
        }
    }

    /// Block until the lane stream is drained. Failure is returned to
    /// `Scheduler::step` as an engine-fatal error.
    fn drain(&self) -> Result<()> {
        let sync = unsafe { cudarc::driver::sys::cuStreamSynchronize(self.stream.stream) };
        anyhow::ensure!(
            sync == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "cuStreamSynchronize(prefill) failed ({sync:?})"
        );
        Ok(())
    }
}

impl Drop for AsyncPrefillLane {
    fn drop(&mut self) {
        if let Err(error) = self.drain() {
            log::error!("prefill lane teardown failed: {error:#}");
        }
    }
}

/// The fail-closed request validation every admission path shares; `Err`
/// carries the typed refusal. Refuse every unsupported capability carried by
/// the stepped `Request` (LoRA and P/D transfer metadata) rather than
/// silently ignoring it, and a scored prompt longer than `score_ceiling`.
fn validate_request(
    request: &Request,
    max_context: usize,
    score_ceiling: usize,
) -> Result<usize, RejectReason> {
    let prompt_tokens = request.prompt_tokens.len();
    if prompt_tokens == 0 {
        return Err(RejectReason::Unsupported {
            feature: "empty prompts".into(),
        });
    }
    if request.max_tokens == 0 {
        return Err(RejectReason::Unsupported {
            feature: "zero max_tokens".into(),
        });
    }
    let Some(context_len) = prompt_tokens
        .checked_add(request.max_tokens)
        .filter(|len| *len <= max_context)
    else {
        return Err(RejectReason::ContextLength {
            prompt_tokens,
            max_tokens: request.max_tokens,
            limit: max_context,
        });
    };
    if request.lora_adapter.is_some() {
        return Err(RejectReason::Unsupported {
            feature: "LoRA".into(),
        });
    }
    if request.prompt_logprobs.is_some() && prompt_tokens > score_ceiling {
        return Err(RejectReason::EchoPrefillTokens {
            prompt_tokens,
            limit: score_ceiling,
        });
    }
    if request.kv_transfer_params.is_some() {
        return Err(RejectReason::Unsupported {
            feature: "kv_transfer".into(),
        });
    }
    Ok(context_len)
}

/// Both pools' page budgets for one configuration; `None` when the
/// arithmetic overflows. The first three counts are not interchangeable:
/// the transient and the window are local pages, the context is global
/// ones, and the two families size their pages differently.
fn pool_pages(
    transient_pages: usize,
    window_pages: usize,
    global_context_pages: usize,
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
        .checked_mul(global_context_pages)?
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
    context_len.div_ceil(GLOBAL_PAGE_SIZE)
}

/// How many prompts one mixed step may absorb: bounded well below the
/// sampler-row capacity so a burst still leaves decode rows headroom.
const MIX_MAX_PROMPTS: usize = 4;

/// The unchunked follower-gather budget, not a step row ceiling: the
/// leader's unseen suffix counts against it but the leader itself is never
/// bounded (a long leader still rides the live decode batch, alone), and
/// the chunked walk ignores it — the chunk knob prices rows per step
/// instead. The trade it bounds: gathering amortizes only the step floor
/// while every live stream's inter-token gap pays the whole gathered step,
/// so absorbing long prompts trades a large certain loss for a small fixed
/// win; short bursts are where the floor dominates, and the budget keeps
/// the gather there. Calibration measurements live in the benchmark
/// records, not here.
const MIX_GATHER_ROWS: usize = 512;

fn parse_mix_max_prompts(raw: &str, slots: usize) -> Result<usize> {
    let prompts: usize = raw
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("{MIX_MAX_PROMPTS_ENV} must be a count: {raw:?}"))?;
    anyhow::ensure!(
        prompts > 0 && prompts <= slots,
        "{MIX_MAX_PROMPTS_ENV} must be in 1..={slots}"
    );
    Ok(prompts)
}

/// A step's rows live in metadata the ceiling sizes, so a budget past it
/// buys a step that cannot be built; refusing at start-up beats the same
/// refusal arriving as a failed step mid-run.
fn parse_mix_gather_rows(raw: &str, max_context: usize) -> Result<usize> {
    let rows: usize = raw
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("{MIX_GATHER_ROWS_ENV} must be a row count: {raw:?}"))?;
    anyhow::ensure!(
        rows > 0 && rows <= max_context,
        "{MIX_GATHER_ROWS_ENV} must be in 1..={max_context}, the serving ceiling"
    );
    Ok(rows)
}

/// Every serving knob, read through one lookup and held against the others
/// before the weights load.
#[derive(Clone, Copy, Debug)]
struct ServingKnobs {
    /// The serving ceiling: prompt plus output per request, the pool budget
    /// axis, and the published servable length.
    max_context: usize,
    lane_mode: Option<LaneMode>,
    /// The chunked-walk segment span. The effective step rounds down to whole
    /// 128-row tiles.
    mix_chunk: Option<usize>,
    mix_gather: usize,
    mix_max_prompts: usize,
    admit_coalesce: Option<std::time::Duration>,
    slots: usize,
    local_kv_storage: KvStorage,
    global_attn: GlobalAttn,
    prefix_cache: Option<usize>,
}

impl ServingKnobs {
    fn resolve(
        lookup: &dyn Fn(&str) -> Result<Option<String>>,
        config: &crate::config::Gemma4Config,
    ) -> Result<Self> {
        let checkpoint_limit = config.max_position_embeddings;
        let max_context = lookup(MAX_CONTEXT_ENV)?
            .map_or(Ok(MAX_CONTEXT.min(checkpoint_limit)), |raw| {
                parse_serving_context(&raw, checkpoint_limit)
            })?;
        let lane_mode =
            lookup(ASYNC_PREFILL_ENV)?.map_or(Ok(None), |raw| parse_async_prefill_mode(&raw))?;
        let mix_chunk = lookup(MIX_CHUNK_TOKENS_ENV)?
            .map_or(Ok(None), |raw| parse_mix_chunk_tokens(&raw, max_context))?;
        let mix_gather = lookup(MIX_GATHER_ROWS_ENV)?.map_or(Ok(MIX_GATHER_ROWS), |raw| {
            parse_mix_gather_rows(&raw, max_context)
        })?;
        let admit_coalesce =
            lookup(ADMIT_COALESCE_ENV)?.map_or(Ok(None), |raw| parse_admit_coalesce_ms(&raw))?;
        let slots = lookup(DECODE_SLOTS_ENV)?
            .map_or(Ok(MAX_CONCURRENCY), |raw| parse_decode_slots(&raw))?;
        let mix_max_prompts = lookup(MIX_MAX_PROMPTS_ENV)?.map_or(Ok(MIX_MAX_PROMPTS), |raw| {
            parse_mix_max_prompts(&raw, slots)
        })?;
        let local_kv_storage = local_kv_storage(lookup)?;
        let global_attn = lookup(GLOBAL_ATTN_ENV)?
            .map_or(Ok(GlobalAttn::Incumbent), |raw| parse_global_attn(&raw))?;
        let prefix_cache =
            lookup(PREFIX_CACHE_ENV)?.map_or(Ok(None), |raw| parse_prefix_cache_cap(&raw))?;

        // The stub tier links under the same name and refuses at launch, so
        // without this the answer would arrive after the weights are loaded
        // and on the first prompt rather than here.
        anyhow::ensure!(
            !global_attn.tilelang() || pegainfer_kernels::ops::gemma4_hd512_prefill_is_built(),
            "{GLOBAL_ATTN_ENV} asks for the generated kernel, which needs a build that \
             carries it; this one fell back to the stub tier, so pegainfer-kernels was \
             compiled without TileLang and without a pre-generated directory"
        );
        if global_attn.tilelang() {
            tilelang_geometry_refusal(config)?;
        }
        anyhow::ensure!(
            !config.w4a16 || lane_mode.is_none(),
            "{ASYNC_PREFILL_ENV} cannot serve a W4A16 checkpoint: its decode GEMMs finish split \
             tiles in-kernel and need every CTA resident across the whole device, which a lane \
             prefill beside the decode stream does not leave them"
        );
        anyhow::ensure!(
            admit_coalesce.is_none() || lane_mode.is_none(),
            "{ADMIT_COALESCE_ENV} and {ASYNC_PREFILL_ENV} cannot combine: the lane flies one \
             prefill at a time, so the door could only delay it"
        );
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
        anyhow::ensure!(
            local_kv_storage != KvStorage::E4m3 || prefix_cache.is_none(),
            "{KV_FP8_ENV} and {PREFIX_CACHE_ENV} cannot combine: the prefix cache copies pool \
             pages in bf16 element units"
        );
        Ok(Self {
            max_context,
            lane_mode,
            mix_chunk,
            mix_gather,
            mix_max_prompts,
            admit_coalesce,
            slots,
            local_kv_storage,
            global_attn,
            prefix_cache,
        })
    }
}

/// One prompt mid-walk: its unseen suffix begins at `offset`, and `first`
/// holds the token its final segment sampled until the walker graduates.
struct Walker {
    request: QueuedRequest,
    kv: GemmaKv,
    resumed: Option<u64>,
    offset: usize,
    first: Option<SampledToken>,
    failed: bool,
}

/// The chunked walk behind `PEGAINFER_MIX_CHUNK_TOKENS`: every gathered
/// prompt walks the same segment schedule, one shared mixed step per round,
/// packing up to `chunk` unseen prompt rows across walkers in admission order
/// on top of the live decode batch — the streams advance one token per round
/// instead of waiting out whole prompts. Each round samples every segment's
/// last row; only a walker's final segment's row is kept as its first token,
/// and that walker graduates into the decode batch at the round boundary. A
/// drained roster finishes the remaining tails on the plain path, segment by
/// segment.
///
/// One scheduler step advances one round:
/// the contract driver commits the ledger once per step, so running the whole
/// walk inside one call would withhold every live stream's tokens until all
/// prompt suffixes had completed.
struct Walk {
    walkers: Vec<Walker>,
    chunk: usize,
    first_round: bool,
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

type Newcomer = (QueuedRequest, GemmaKv, Option<u64>);

#[derive(Clone, Copy)]
struct NewcomerOptions {
    reserve_whole: bool,
    evict_cache: bool,
    can_wait: bool,
    max_new_tokens: Option<usize>,
}

enum PreparedNewcomer {
    Ready(Newcomer, usize),
    Done,
    Requeue(QueuedRequest),
}

/// One row of a step's sampler call.
#[derive(Clone, Copy)]
struct SampleRow<'a> {
    params: &'a pegainfer_frontend::sampler::SamplingParams,
    step: u64,
    logprobs: Option<usize>,
    ignore_eos: bool,
}

impl<'a> SampleRow<'a> {
    /// The row that samples `request`'s completion token number `step`.
    fn of(request: &'a Request, step: u64) -> Self {
        Self {
            params: &request.params,
            step,
            logprobs: request.logprobs,
            ignore_eos: request.params.ignore_eos,
        }
    }

    /// A mid-walk segment's row, sampled and discarded: it never stops and is
    /// never scored.
    fn discarded(request: &'a Request) -> Self {
        Self {
            logprobs: None,
            ignore_eos: true,
            ..Self::of(request, 0)
        }
    }
}

/// The per-call sampler seed: the engine's base seed mixed with a counter
/// every sampler call advances, the staged greedy ones that take no seed
/// included. Seedless sampling variety across requests comes from it; a
/// request's own `params.seed` replays via (seed, step) regardless of it.
struct SampleSeed {
    base: u64,
    nonce: u64,
}

impl SampleSeed {
    fn next_call(&mut self) -> u64 {
        self.nonce = self.nonce.wrapping_add(1);
        self.base ^ self.nonce.rotate_left(17)
    }
}

/// One sampler call's outcome, row-aligned with the logits it read.
struct SampledRows {
    picked: Vec<u32>,
    logprobs: Vec<Option<TokenLogprob>>,
    stops: Vec<bool>,
}

impl SampledRows {
    fn token(&mut self, row: usize) -> SampledToken {
        SampledToken {
            id: self.picked[row],
            logprob: self.logprobs[row].take(),
            stop: self.stops[row],
        }
    }
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
    seed: &mut SampleSeed,
    rows: &[SampleRow<'_>],
    logits: &mut HiddenStates,
) -> Result<SampledRows> {
    ops::suppress_logits_bf16_in_place(ctx, logits, suppress_ids).context("suppression")?;

    let call_seed = seed.next_call();
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
        .filter_map(|(row, spec)| {
            spec.logprobs
                .filter(|_| !stops[row])
                .map(|top_k| LogprobRequest {
                    row,
                    picked: picked[row],
                    top_k,
                })
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
fn fail_active_batch(
    active: &mut Vec<Active>,
    what: &str,
    err: &anyhow::Error,
    ledger: &mut RequestLedger,
) {
    log::error!("{what} failed: {err:#}");
    fail_requests(
        active.drain(..).map(|entry| entry.request.id),
        what,
        err,
        ledger,
    );
}

fn fail_requests(
    ids: impl IntoIterator<Item = RequestId>,
    what: &str,
    err: &anyhow::Error,
    ledger: &mut RequestLedger,
) {
    for id in ids {
        if ledger.is_active(id) {
            ledger.fail(id, format!("{what} failed: {err:#}"));
        }
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
    seed: &mut SampleSeed,
    head: &[SampleRow<'_>],
    active: &mut Vec<Active>,
    logits: &mut HiddenStates,
    ledger: &mut RequestLedger,
) -> Result<SampledRows> {
    let k = head.len();
    let sampled = {
        let rows: Vec<SampleRow<'_>> = head
            .iter()
            .copied()
            .chain(active.iter().map(|entry| entry.sample_row(ledger)))
            .collect();
        sample_logits_rows(ctx, suppress_ids, policy, scratch, seed, &rows, logits)
    };
    let mut sampled = match sampled {
        Ok(sampled) => sampled,
        Err(err) => {
            fail_active_batch(active, "mixed step", &err, ledger);
            return Err(err.context("mixed step"));
        }
    };
    // Active rows: the decode-round event flow, `k` logits rows up.
    emit_decode_rows(active, &mut sampled, k, ledger);
    Ok(sampled)
}

/// One in-flight request between decode steps: its KV, the token that feeds
/// the next step, and its progress counters.
struct Active {
    request: QueuedRequest,
    kv: GemmaKv,
    next: u32,
    /// The row has finished while a speculative step over its slot is still
    /// in flight and retires when that step drains.
    stopping: bool,
}

impl Active {
    fn sample_row(&self, ledger: &RequestLedger) -> SampleRow<'_> {
        SampleRow::of(
            &self.request.request,
            ledger.completion_tokens(self.request.id) as u64,
        )
    }

    /// A staged greedy pick never went through [`sample_logits_rows`], so its
    /// stop is decided here, by the same rule.
    fn settle_staged(&mut self, policy: &GenerationPolicy, token: u32, ledger: &mut RequestLedger) {
        if self.stopping {
            return;
        }
        let stop = policy.stops(token, self.request.request.params.ignore_eos);
        self.stopping = settle_token(
            self,
            SampledToken {
                id: token,
                logprob: None,
                stop,
            },
            ledger,
        );
    }
}

const DECODE_PIPELINE_DEPTH: usize = 2;

/// One staged decode step whose readback has not yet been collected.
struct PendingDecode {
    rows: usize,
    slot: usize,
}

enum Admitted {
    Active(Box<Active>),
    /// Finished, refused, cancelled or failed: nothing carries forward.
    Done,
    /// The pools cannot hold it right now; retry once pages return.
    Requeue(Box<QueuedRequest>),
}

fn send_scheduled(request: &QueuedRequest, cached_tokens: usize, ledger: &mut RequestLedger) {
    ledger.admit(request.id);
    if cached_tokens > 0 {
        ledger.set_cached_tokens(request.id, cached_tokens);
    }
}

/// Everything the contract-owned scheduler thread owns for the life of the
/// engine. Loading completes before the driver thread is spawned, so launch
/// failures return synchronously to the caller.
/// One tensor-parallel rank's device state. Rank 0's lives in `EngineState`'s
/// own fields; ranks 1.. world size ride in `EngineState::more`.
struct RankState {
    ctx: DeviceContext,
    serve: GemmaServe,
    arena: StepArena,
}

/// Make `ctx`'s device current on this thread. The thread-local cuBLAS handles
/// are keyed by device, so switching the current device is enough once every
/// rank has been bound at the start of a step.
fn activate_rank(ctx: &DeviceContext) -> Result<()> {
    select_device(ctx)?;
    // `cublas_init` is the only call that can create a device's handle pair on
    // this thread, and it also selects them, so it both covers the first use
    // of a device and the switch back to one already used.
    unsafe { pegainfer_core::ffi::cublas_init() };
    let err = unsafe { pegainfer_core::ffi::cublas_activate_device_handles() };
    anyhow::ensure!(
        err == 0,
        "cuBLAS handle activation on device {} failed: cudaError={err}",
        ctx.device_ordinal
    );
    Ok(())
}

/// Set `ctx`'s device current and bind its primary context to this thread.
fn select_device(ctx: &DeviceContext) -> Result<()> {
    let err = unsafe { pegainfer_core::ffi::cuda_set_device(ctx.device_ordinal as i32) };
    anyhow::ensure!(
        err == 0,
        "cudaSetDevice({}) on the scheduler thread failed: cudaError={err}",
        ctx.device_ordinal
    );
    ctx.ctx.bind_to_thread().map_err(|e| {
        anyhow::anyhow!(
            "bind device {} to the scheduler thread: {e}",
            ctx.device_ordinal
        )
    })?;
    Ok(())
}

/// The rank-`rank` KV of every row, in row order.
fn rank_kvs(rows: &mut [Active], rank: usize) -> Vec<&mut RankKv> {
    rows.iter_mut()
        .map(|entry| entry.kv.core_mut(rank))
        .collect()
}

struct EngineState {
    ctx: DeviceContext,
    serve: GemmaServe,
    arena: StepArena,
    /// Ranks 1.. world size, empty at world size 1. Same shape as rank 0's
    /// fields above.
    more: Vec<RankState>,
    scratch: SampleScratch,
    /// Conversation-tail prefix cache; `None` unless
    /// `PEGAINFER_PREFIX_CACHE=K` opted in at startup.
    prefix_cache: Option<PrefixCache>,
    policy: GenerationPolicy,
    /// Validated once against the head, then retained on-device for one mask
    /// launch per logits batch.
    suppress_ids: ops::SuppressIds,
    seed: SampleSeed,
    /// Present only while the active row order is frozen.
    pipeline: Option<PendingDecode>,
    /// Captured suppression, argmax and id-copy chain per decode bucket.
    sampler_graphs: Vec<CudaGraphState>,
    /// The overlap lane; `None` unless `PEGAINFER_ASYNC_PREFILL` opted in
    /// at startup.
    lane: Option<AsyncPrefillLane>,
    /// The chunked-walk segment span; `None` unless
    /// `PEGAINFER_MIX_CHUNK_TOKENS` opted in at startup.
    mix_chunk: Option<usize>,
    mix_gather: usize,
    mix_max_prompts: usize,
    /// The serving ceiling this process was started with; the pools are
    /// budgeted against it.
    max_context: usize,
    /// The longest prompt a scored request may carry. It prefills whole, so
    /// it is held to the default ceiling and to the local pages an idle
    /// server has.
    score_ceiling: usize,
    /// The decode-slot count the pools are budgeted for; requests past it
    /// queue.
    slots: usize,
    /// The admission coalesce window; `None` unless
    /// `PEGAINFER_ADMIT_COALESCE_MS` opted in at startup.
    admit_coalesce: Option<std::time::Duration>,
}

/// Captured collective graphs bake in NCCL kernel launches, and NCCL's
/// communicator abort wedges while a graph that references them is still
/// alive — so every rank releases its graphs here, on its own device, before
/// the fields (and with them the communicators) drop.
impl Drop for EngineState {
    fn drop(&mut self) {
        let _ = select_device(&self.ctx);
        self.arena.release_graphs();
        for state in &mut self.more {
            let _ = select_device(&state.ctx);
            state.arena.release_graphs();
        }
    }
}

impl EngineState {
    /// A fresh per-request KV across every rank: rank 0 from the primary
    /// serve, each other rank from its own serve. Each entry is bound to its
    /// rank's pools, which is what `admit_tokens`' `belongs_to` check needs.
    fn alloc_kv(&self) -> GemmaKv {
        GemmaKv::multi(
            self.serve.alloc_rank_kv(),
            self.more
                .iter()
                .map(|rank| rank.serve.alloc_rank_kv())
                .collect(),
        )
    }

    /// Every rank must stand on rank 0's frontier. The page-id contract —
    /// identical budgets and an identical admission sequence keep the ranks'
    /// page ids in step — is checked nowhere else, so a divergence caught here
    /// is the last chance before it becomes a silent read of another rank's
    /// page. The free-page counts carry the same signal but take the pool locks,
    /// so they are compared in a debug build only.
    fn assert_ranks_in_step(&self, kv: &GemmaKv) -> Result<()> {
        let local = kv.local.seq_len();
        let global = kv.global.seq_len();
        for rank in 1..=self.more.len() {
            let theirs = kv.core(rank);
            anyhow::ensure!(
                theirs.local.seq_len() == local && theirs.global.seq_len() == global,
                "rank {rank} stands on frontier {}/{} where rank 0 stands on {local}/{global}",
                theirs.local.seq_len(),
                theirs.global.seq_len()
            );
        }
        #[cfg(debug_assertions)]
        for (index, state) in self.more.iter().enumerate() {
            let rank = index + 1;
            anyhow::ensure!(
                state.serve.local_pool.available_pages() == self.serve.local_pool.available_pages(),
                "rank {rank}'s local pool holds different free pages than rank 0's"
            );
            anyhow::ensure!(
                state.serve.global_pool.available_pages()
                    == self.serve.global_pool.available_pages(),
                "rank {rank}'s global pool holds different free pages than rank 0's"
            );
        }
        Ok(())
    }

    /// Run `per_rank` on every extra rank, drain each stream, and require that
    /// they all end on rank 0's frontier.
    ///
    /// Call this *after* rank 0's own step has been started, unconditionally:
    /// the ranks share one collective sequence, so a rank-0 failure that skipped
    /// the peer would leave the peer's matching calls unpaired, and the pairing
    /// then shifts for every later step — reducing one layer's partials against
    /// another's — or wedges the peer at its drain. A failure on an extra rank
    /// is fatal here, because the ranks' frontiers must not drift apart, while
    /// rank 0's own failure stays the caller's to report. The drain is what
    /// turns a device fault on a non-primary rank into a named error instead of
    /// the primary's collective stalling forever.
    fn step_extra_ranks<F>(&self, kv: &mut GemmaKv, per_rank: F) -> Result<()>
    where
        F: Fn(&RankState, &mut RankKv) -> Result<()>,
    {
        for (rank, state) in self.more.iter().enumerate() {
            activate_rank(&state.ctx)?;
            log::debug!("tp: rank {} prefill entering", rank + 1);
            per_rank(state, kv.core_mut(rank + 1)).context("tensor-parallel prefill")?;
            log::debug!("tp: rank {} prefill launched", rank + 1);
            state
                .ctx
                .sync()
                .with_context(|| format!("drain rank {}", rank + 1))?;
            log::debug!("tp: rank {} drained", rank + 1);
        }
        self.assert_ranks_in_step(kv)?;
        // The caller's next device work is rank 0's.
        activate_rank(&self.ctx)
    }

    /// One plain prefill step on every extra rank; their logits are discarded.
    fn prefill_extra_ranks(&self, kv: &mut GemmaKv, tokens: &[u32]) -> Result<()> {
        self.step_extra_ranks(kv, |state, rank_kv| {
            state.serve.step(&state.ctx, rank_kv, tokens).map(|_| ())
        })
    }

    /// Reserve on every extra rank's pools, so their fronts stay in step with
    /// rank 0's. Purely host-side, so no activation is needed.
    fn admit_extra_ranks(&self, kv: &mut GemmaKv, tokens: usize) -> Result<()> {
        for (rank, state) in self.more.iter().enumerate() {
            admit_tokens(
                &state.serve.local_pool,
                &state.serve.global_pool,
                kv.core_mut(rank + 1),
                tokens,
            )?;
        }
        // A rank that reserved different pages here has already lost the page-id
        // race, and nothing after this can repair it.
        #[cfg(debug_assertions)]
        {
            let free = (
                self.serve.local_pool.available_pages(),
                self.serve.global_pool.available_pages(),
            );
            for (rank, state) in self.more.iter().enumerate() {
                debug_assert_eq!(
                    free,
                    (
                        state.serve.local_pool.available_pages(),
                        state.serve.global_pool.available_pages()
                    ),
                    "rank {}'s pools reserved different pages than rank 0's",
                    rank + 1
                );
            }
        }
        Ok(())
    }
}

/// The scheduler thread is not the thread that loaded the engine: the
/// primary context must be made current there and the thread-local cuBLAS
/// handles created, or the first eager GEMM fails with an invalid handle.
/// The returned guard tears the handles down when the scheduler drops on
/// that thread.
fn bind_engine_thread(ctx: &DeviceContext) -> Result<CublasThreadGuard> {
    let err = unsafe { pegainfer_core::ffi::cuda_set_device(ctx.device_ordinal as i32) };
    anyhow::ensure!(
        err == 0,
        "cudaSetDevice({}) on the scheduler thread failed: cudaError={err}",
        ctx.device_ordinal
    );
    ctx.ctx
        .bind_to_thread()
        .map_err(|e| anyhow::anyhow!("bind the CUDA context to the scheduler thread: {e}"))?;
    unsafe { pegainfer_core::ffi::cublas_init() };
    Ok(CublasThreadGuard)
}

/// Destroys the thread-local cuBLAS handles [`bind_engine_thread`] created.
/// The scheduler holds it so the drop lands on the driver thread that owns
/// the handles, once the engine state has released its device work.
struct CublasThreadGuard;

impl Drop for CublasThreadGuard {
    fn drop(&mut self) {
        unsafe { pegainfer_core::ffi::cublas_destroy() };
    }
}

struct Gemma4Scheduler {
    state: EngineState,
    pending: VecDeque<QueuedRequest>,
    active: Vec<Active>,
    door: Option<CoalesceDoor>,
    walk: Option<Walk>,
    /// `Some` once the driver thread has bound the context; declared last so
    /// the handles outlive the engine state's teardown.
    cublas: Option<CublasThreadGuard>,
}

impl Gemma4Scheduler {
    fn new(state: EngineState) -> Self {
        let door = state.coalesce_door();
        Self {
            state,
            pending: VecDeque::new(),
            active: Vec::new(),
            door,
            walk: None,
            cublas: None,
        }
    }
}

impl EngineState {
    fn coalesce_door(&self) -> Option<CoalesceDoor> {
        self.admit_coalesce
            .map(|window| CoalesceDoor::new(window, self.mix_max_prompts))
    }

    fn intake_turn(
        &mut self,
        door: &mut Option<CoalesceDoor>,
        pending: &mut VecDeque<QueuedRequest>,
        active: &mut Vec<Active>,
        walk: &mut Option<Walk>,
        now: std::time::Instant,
        ledger: &mut RequestLedger,
    ) -> Result<bool> {
        let open = door
            .as_mut()
            .is_none_or(|door| door.opens(pending.len(), active.len(), self.slots, now));
        if open {
            self.admit_from_queue(pending, active, walk, ledger)?;
        }
        Ok(open)
    }

    fn reserve_with_eviction(
        &mut self,
        kv: &mut GemmaKv,
        need: AdmissionNeed,
        evict_cache: bool,
        can_wait: bool,
    ) -> ReservationDecision {
        loop {
            let refusal = match need {
                AdmissionNeed::Tokens(tokens) => {
                    // Every rank's pools reserve the same pages, so the fronts
                    // stay in step; only then does the admission succeed.
                    admit_tokens(&self.serve.local_pool, &self.serve.global_pool, kv, tokens)
                        .and_then(|()| self.admit_extra_ranks(kv, tokens))
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
            if evict_cache
                && self
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

    /// A prompt that asks for its scores never resumes: every position has
    /// to go through this prefill's head.
    fn resolve_newcomer_kv(&mut self, request: &Request) -> (GemmaKv, Option<u64>) {
        match self
            .prefix_cache
            .as_mut()
            .filter(|_| request.prompt_logprobs.is_none())
            .and_then(|cache| cache.resolve(&request.prompt_tokens))
        {
            Some((entry, t)) => match self.serve.restore_from_checkpoint(&self.ctx, entry, t) {
                Ok(kv) => (kv, Some(entry.id)),
                Err(err) => {
                    log::warn!("prefix-cache restore failed (falling back): {err:#}");
                    (self.alloc_kv(), None)
                }
            },
            None => (self.alloc_kv(), None),
        }
    }

    fn prepare_newcomer(
        &mut self,
        request: QueuedRequest,
        options: NewcomerOptions,
        ledger: &mut RequestLedger,
    ) -> PreparedNewcomer {
        if ledger.is_aborted(request.id) {
            ledger.retire(request.id);
            return PreparedNewcomer::Done;
        }
        let context_len =
            match validate_request(&request.request, self.max_context, self.score_ceiling) {
                Ok(len) => len,
                Err(reason) => {
                    ledger.reject(request.id, reason);
                    return PreparedNewcomer::Done;
                }
            };
        let (mut kv, resumed) = self.resolve_newcomer_kv(&request.request);
        let new_tokens = request.request.prompt_tokens.len() - kv.local.seq_len();
        if options
            .max_new_tokens
            .is_some_and(|limit| new_tokens > limit)
        {
            return PreparedNewcomer::Requeue(request);
        }
        let need = if options.reserve_whole {
            AdmissionNeed::Tokens(new_tokens)
        } else {
            AdmissionNeed::GlobalPages(global_account_pages(context_len))
        };
        match self.reserve_with_eviction(&mut kv, need, options.evict_cache, options.can_wait) {
            ReservationDecision::Ready => {}
            ReservationDecision::Requeue => {
                return PreparedNewcomer::Requeue(request);
            }
            ReservationDecision::Refused(message) => {
                log::warn!("KV admission refused {}: {message}", request.id);
                ledger.reject(
                    request.id,
                    RejectReason::KvBudget {
                        prompt_tokens: request.request.prompt_tokens.len(),
                        worst_case_tokens: context_len,
                    },
                );
                return PreparedNewcomer::Done;
            }
        }
        send_scheduled(&request, kv.local.seq_len(), ledger);
        PreparedNewcomer::Ready((request, kv, resumed), new_tokens)
    }

    fn load(
        dir: &str,
        config: &crate::config::Gemma4Config,
        knobs: ServingKnobs,
        ordinals: &[usize],
        policy: GenerationPolicy,
        base_seed: u64,
        graph_enabled: bool,
    ) -> Result<Self> {
        anyhow::ensure!(
            !ordinals.is_empty(),
            "a gemma4 engine needs at least one device ordinal"
        );
        let world = ordinals.len();
        let device = ordinals[0];
        let tp = TensorParallelConfig::new(0, world);
        // Refuse an unservable global GQA shape or device before the
        // multi-GiB load.
        tp.validate_for(config)?;
        let global_split =
            crate::serve::global_split_factor(&LayerGeometry::global_of(config, tp)?)?;
        let ServingKnobs {
            max_context,
            lane_mode,
            mix_chunk,
            mix_gather,
            mix_max_prompts,
            admit_coalesce,
            slots,
            local_kv_storage,
            global_attn,
            prefix_cache: cache_cap,
        } = knobs;
        if global_attn.tilelang() {
            ensure_tilelang_device(device)?;
        }
        let weights = Gemma4Weights::from_safetensors(dir, device, config.clone(), tp)?;
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
        // The families page at different granularities, so each budget below
        // names the one it counts: the window and the local transient are
        // local pages, the global account is global pages. One ceiling in
        // local pages is not the same number in global pages.
        let local_context_pages = max_context.div_ceil(LOCAL_PAGE_SIZE);
        let global_context_pages = max_context.div_ceil(GLOBAL_PAGE_SIZE);
        let window_pages = weights.config.sliding_window.div_ceil(LOCAL_PAGE_SIZE) + 1;
        // The cache brings its own page budget so cached entries never eat
        // serving headroom.
        let cache_entries = cache_cap.unwrap_or(0);
        let sliding_window = weights.config.sliding_window;
        // With the chunk knob set every scan is bounded by window plus
        // segment — except the lane's, which prefills whole and keeps the
        // full transient, and a scored prompt's, which prefills whole and
        // is refused past what the pool holds.
        let transient_pages = match mix_chunk {
            Some(chunk) if lane_mode.is_none() => {
                // A round's rows split across walkers, and every walker's
                // reservation rounds up to its own page — so the budget
                // carries one page of rounding per extra walker.
                window_pages + chunk.div_ceil(LOCAL_PAGE_SIZE) + (mix_max_prompts - 1)
            }
            _ => local_context_pages,
        };
        let (local_pages, global_pages) = pool_pages(
            transient_pages,
            window_pages,
            global_context_pages,
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
        // Every local page but the padding one, an idle server's whole pool.
        let score_ceiling = MAX_CONTEXT.min((local_pages - 1) * LOCAL_PAGE_SIZE);
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
        let mut serve = GemmaServe::new(
            &ctx,
            weights,
            tp,
            max_context,
            local_kv_storage,
            local_pages,
            global_pages,
            global_attn,
        )
        .map_err(|err| {
            err.context(format!(
                "a {max_context} token ceiling, {slots} decode slots and {cache_entries} \
                     cache entries sized the pools to {local_pages} local / {global_pages} \
                     global pages"
            ))
        })?;
        // The page count is the budget's; what a page costs is the format's.
        // Said once here, so a pool that came out a different size than the
        // format promised is visible at start-up rather than at the OOM.
        {
            let layout = serve.global_pool.layout();
            let pages = serve.global_pool.capacity_pages();
            let page_bytes = layout.page_stride * layout.storage.elem_bytes();
            log::info!(
                "global KV pool: {pages} pages x {page_bytes} B ({:?}, {} columns per \
                 head) = {:.2} GiB",
                layout.format,
                layout.format.row_width(layout.head_dim),
                (pages * page_bytes) as f64 / (1u64 << 30) as f64
            );
        }
        let prefix_cache = cache_cap.map(|k| PrefixCache::new(k, sliding_window));
        let mut scratch = SampleScratch::new(&ctx, vocab, arena_rows)?;
        let mut arena = serve.alloc_step_arena(&ctx, arena_rows, graph_enabled)?;
        // One rank sweeps here; more than one sweeps in the interleaved driver
        // below, which owns the buckets for every rank.
        if world == 1 {
            serve.precapture_decode_graphs(&ctx, &mut arena)?;
        }
        let suppress_ids = ops::SuppressIds::upload(&ctx, &policy.suppress, vocab)?;
        let mut sampler_graphs = Vec::new();
        if graph_enabled {
            let (logits, ids) = arena.logits_and_ids();
            // The warm pass lands lazy module loads outside capture.
            logits.seq_len = arena_rows;
            ops::suppress_logits_bf16_in_place(&ctx, logits, &suppress_ids)?;
            pegainfer_sample::greedy_argmax_ids(&ctx, logits, arena_rows, ids, &mut scratch)?;
            let mut bucket = 1usize;
            while bucket <= arena_rows {
                logits.seq_len = bucket;
                let mut graph = CudaGraphState::new();
                graph.capture_only(&ctx, || {
                    ops::suppress_logits_bf16_in_place(&ctx, logits, &suppress_ids)?;
                    pegainfer_sample::greedy_argmax_ids(&ctx, logits, bucket, ids, &mut scratch)
                })?;
                sampler_graphs.push(graph);
                bucket *= 2;
            }
        }
        // The KV pools, arena and warm passes were enqueued on this stream. The lane
        // stream and everything after it must see them complete, graphs or not.
        ctx.sync()?;
        let lane = lane_mode
            .map(|mode| AsyncPrefillLane::new(&ctx, mode))
            .transpose()?;
        // The other ranks: their own sharded weights, context, pools and
        // arena. The pool budget is rank-independent (identical pools with
        // identical counts), so it is computed once above and handed to each.
        let mut more = Vec::with_capacity(world - 1);
        for (rank, ordinal) in ordinals.iter().copied().enumerate().skip(1) {
            let rank_tp = TensorParallelConfig::new(rank, world);
            let weights = Gemma4Weights::from_safetensors(dir, ordinal, config.clone(), rank_tp)?;
            let rank_ctx = DeviceContext::new_with_device(ordinal)?;
            let rank_serve = GemmaServe::new(
                &rank_ctx,
                weights,
                rank_tp,
                max_context,
                local_kv_storage,
                local_pages,
                global_pages,
                global_attn,
            )
            .map_err(|err| {
                err.context(format!(
                    "rank {rank} (device {ordinal}) sized its pools to {local_pages} local / \
                     {global_pages} global pages"
                ))
            })?;
            let rank_arena = rank_serve.alloc_step_arena(&rank_ctx, arena_rows, graph_enabled)?;
            more.push(RankState {
                ctx: rank_ctx,
                serve: rank_serve,
                arena: rank_arena,
            });
        }
        // Materialize every rank's lazily-loaded cuBLAS/cublasLt kernels before
        // the communicator exists. Under the default CUDA_MODULE_LOADING=LAZY
        // a fresh process enters the driver's module loader on the first GEMM
        // of each shape; NCCL's proxy threads enter that same loader while the
        // communicator comes up, and the two deadlock — the eager prefill then
        // never returns while the engine spin-waits. One tower pass per rank
        // here, with no comm live, pays those loads single-threaded. The row
        // counts span cublasLt's kernel-selection regions: a single row takes
        // a GEMV-like kernel, the rest tile GEMMs whose choice shifts at small
        // row counts. It also warms the decode capture sweep that used to run
        // below under graphs.
        if world > 1 {
            let warm_rows = [1usize, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
                .into_iter()
                .filter(|rows| *rows <= max_context);
            let warm = |serve: &GemmaServe, ctx: &DeviceContext, rows: usize| -> Result<()> {
                let mut kv = serve.alloc_kv();
                admit_tokens(&serve.local_pool, &serve.global_pool, &mut kv, rows)?;
                activate_rank(ctx)?;
                serve.step(ctx, &mut kv, &vec![0u32; rows])?;
                Ok(())
            };
            for rows in warm_rows.clone() {
                warm(&serve, &ctx, rows)?;
            }
            for state in &more {
                for rows in warm_rows.clone() {
                    warm(&state.serve, &state.ctx, rows)?;
                }
            }
            activate_rank(&ctx)?;
        }
        // One communicator per rank, built on the stream the decode graph would
        // run on, so an all-reduce lands inside that graph.
        if world > 1 {
            let mut streams = Vec::with_capacity(world);
            streams.push(ctx.stream.clone());
            for state in &more {
                streams.push(state.ctx.stream.clone());
            }
            let mut comms = cudarc::nccl::safe::Comm::from_devices(streams)
                .map_err(|e| anyhow::anyhow!("failed to initialize NCCL comms: {e:?}"))?
                .into_iter();
            serve.attach_tp_comm(comms.next().expect("one comm per rank"));
            for state in &mut more {
                state
                    .serve
                    .attach_tp_comm(comms.next().expect("one comm per rank"));
            }
        }
        // With graphs on, every rank captures its decode graphs, phase by phase:
        // `Warm` and `Launch` execute and enqueue the all-reduce, whose peer
        // call must be in flight, so the phases interleave across ranks instead
        // of one rank finishing its whole sweep. `Capture` only records, and a
        // recorded collective replays when its peer replays. Each rank sweeps
        // its own arena and its own single-rank dummy.
        if graph_enabled && world > 1 {
            let mut dummies: Vec<GemmaKv> = Vec::with_capacity(world);
            let mut primary = serve.alloc_kv();
            admit_tokens(&serve.local_pool, &serve.global_pool, &mut primary, 1)?;
            dummies.push(primary);
            for state in &more {
                let mut dummy = state.serve.alloc_kv();
                admit_tokens(
                    &state.serve.local_pool,
                    &state.serve.global_pool,
                    &mut dummy,
                    1,
                )?;
                dummies.push(dummy);
            }
            let mut bucket = 1usize;
            while bucket <= arena.bucket_ceiling() {
                for phase in PrecapturePhase::ALL {
                    activate_rank(&ctx)?;
                    serve.precapture_bucket(&ctx, &mut arena, &mut dummies[0], bucket, phase)?;
                    for (rank, state) in more.iter_mut().enumerate() {
                        activate_rank(&state.ctx)?;
                        state.serve.precapture_bucket(
                            &state.ctx,
                            &mut state.arena,
                            &mut dummies[rank + 1],
                            bucket,
                            phase,
                        )?;
                    }
                }
                bucket *= 2;
            }
            arena.reset_min_bucket();
            activate_rank(&ctx)?;
            ctx.sync()?;
        }
        Ok(Self {
            ctx,
            serve,
            arena,
            more,
            scratch,
            prefix_cache,
            policy,
            suppress_ids,
            seed: SampleSeed {
                base: base_seed,
                nonce: 0,
            },
            pipeline: None,
            sampler_graphs,
            lane,
            mix_chunk,
            mix_gather,
            mix_max_prompts,
            max_context,
            score_ceiling,
            slots,
            admit_coalesce,
        })
    }

    /// One turn's intake at the roster edge: admit from the queue head
    /// until the slots are full, the queue empties, the lane is busy, or a
    /// request has to wait for pages. Attempts are bounded so a burst
    /// costs the streams a bounded number of prefills per token however
    /// deep the queue is.
    fn admit_from_queue(
        &mut self,
        pending: &mut VecDeque<QueuedRequest>,
        active: &mut Vec<Active>,
        walk: &mut Option<Walk>,
        ledger: &mut RequestLedger,
    ) -> Result<()> {
        if pending.is_empty() {
            return Ok(());
        }
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
            match self.admit_and_prefill(
                item,
                can_wait,
                active,
                pending,
                walk,
                &mut attempts,
                ledger,
            )? {
                Admitted::Active(request) => active.push(*request),
                Admitted::Done => {}
                Admitted::Requeue(item) => {
                    pending.push_front(*item);
                    break;
                }
            }
            if walk.is_some() {
                break;
            }
        }
        Ok(())
    }

    /// Validate and admit one request. The synchronous arms prefill it, emit
    /// its first token, and hand it to the decode batch; the lane arm only
    /// launches the in-flight prefill, which the join later settles. An
    /// admission refusal is a refusal to the client only when no active
    /// request could free the pages it needs; otherwise the request waits at
    /// the queue head.
    fn admit_and_prefill(
        &mut self,
        item: QueuedRequest,
        can_wait: bool,
        active: &mut Vec<Active>,
        pending: &mut VecDeque<QueuedRequest>,
        walk: &mut Option<Walk>,
        attempts: &mut usize,
        ledger: &mut RequestLedger,
    ) -> Result<Admitted> {
        // The lane prefills whole on its own stream, so a lane-bound
        // admission still reserves everything up front. A chunked
        // admission reserves nothing here: every segment admits its own
        // pages right before it is written, so no walker parks a quantum
        // — parked first segments across several walkers would exhaust
        // the one shared segment transient the pool provisions.
        // A prompt that asks for its scores takes the solo whole-prompt
        // prefill even beside a live batch: only that pass holds every
        // prompt row's final hidden state at once.
        let scored = item.request.prompt_logprobs.is_some();
        let lane_takes = self.lane.is_some() && !active.is_empty() && !scored;
        let options = NewcomerOptions {
            reserve_whole: self.mix_chunk.is_none() || lane_takes || scored,
            evict_cache: true,
            can_wait,
            max_new_tokens: None,
        };
        let (request, mut kv, resumed) = match self.prepare_newcomer(item, options, ledger) {
            PreparedNewcomer::Ready(newcomer, _) => newcomer,
            PreparedNewcomer::Done => return Ok(Admitted::Done),
            PreparedNewcomer::Requeue(item) => return Ok(Admitted::Requeue(Box::new(item))),
        };
        let prompt_tokens = request.request.prompt_tokens.len();

        // Overlapped admission: the prefill launches onto the lane stream
        // and this call returns immediately — decode steps continue while it
        // runs. A prompt arriving with nothing active stays on the sync
        // path: there is nothing to protect, and full-SM speed wins the head
        // of every refill burst.
        if lane_takes {
            return self.launch_async_prefill(request, kv, resumed, ledger);
        }

        // Mixed admission: with a live decode batch, prompts ride its
        // weight scan — one step prefills every gathered newcomer and
        // advances every active row.
        if !active.is_empty() {
            self.drain_pipeline(active, ledger)?;
        }
        if !active.is_empty() && !scored {
            self.ready_decode_rows(active, ledger);
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
                let mut newcomers: Vec<Newcomer> = vec![(request, kv, resumed)];
                let mut rows_budget = {
                    let (_, kv, _) = &newcomers[0];
                    prompt_tokens - kv.local.seq_len()
                };
                while newcomers.len() < self.mix_max_prompts
                    && (self.mix_chunk.is_some() || rows_budget < self.mix_gather)
                    && newcomers.len() + active.len() < self.slots
                    && *attempts < self.slots
                {
                    let Some(candidate) = pending.pop_front() else {
                        break;
                    };
                    if candidate.request.prompt_logprobs.is_some() {
                        pending.push_front(candidate);
                        break;
                    }
                    *attempts += 1;
                    let options = NewcomerOptions {
                        reserve_whole: self.mix_chunk.is_none(),
                        evict_cache: false,
                        can_wait: true,
                        // Lazy: a chunked gather's budget can exceed the
                        // gather rows, and the bound is unused there.
                        max_new_tokens: self
                            .mix_chunk
                            .is_none()
                            .then(|| self.mix_gather.saturating_sub(rows_budget)),
                    };
                    match self.prepare_newcomer(candidate, options, ledger) {
                        PreparedNewcomer::Ready(newcomer, new_tokens) => {
                            rows_budget += new_tokens;
                            newcomers.push(newcomer);
                        }
                        PreparedNewcomer::Done => {}
                        PreparedNewcomer::Requeue(candidate) => {
                            pending.push_front(candidate);
                            break;
                        }
                    }
                }
                return self.mixed_admission(newcomers, active, walk, ledger);
            }
        }

        let mut echo = None;
        let stepped = if let Some(top_k) = request.request.prompt_logprobs {
            let prompt = &request.request.prompt_tokens;
            let mut scores: Vec<Option<TokenLogprob>> = vec![None];
            let (ctx, suppress_ids) = (&self.ctx, &self.suppress_ids);
            let mut score = |logits: &mut HiddenStates, start: usize| -> Result<()> {
                // The same logits a sampled token is scored on: softcapped,
                // then suppressed.
                ops::suppress_logits_bf16_in_place(ctx, logits, suppress_ids)
                    .context("suppression")?;
                let requests: Vec<LogprobRequest> = (0..logits.seq_len)
                    .map(|row| LogprobRequest {
                        row,
                        picked: prompt[start + row + 1],
                        top_k,
                    })
                    .collect();
                let scored = pegainfer_sample::token_logprobs_batch(ctx, logits, &requests)
                    .context("prompt logprobs")?;
                scores.extend(scored.into_iter().map(Some));
                Ok(())
            };
            activate_rank(&self.ctx)?;
            let stepped = self
                .serve
                .step_scoring(ctx, &mut kv, prompt, Some(&mut score));
            // The other ranks run the same prefill to write their KV shards;
            // scoring is rank 0's alone. They are driven after rank 0 and
            // unconditionally, and a failure on them is fatal.
            self.step_extra_ranks(&mut kv, |state, rank_kv| {
                state
                    .serve
                    .step_scoring(&state.ctx, rank_kv, prompt, None)
                    .map(|_| ())
            })?;
            echo = Some(PromptEcho {
                ids: prompt.clone(),
                logprobs: scores,
            });
            stepped
        } else if let Some(chunk) = self.mix_chunk {
            // Under the chunk knob a solo prompt walks its own segments too:
            // residency stays window plus segment whatever the prompt length.
            self.walk_plain_prompt(&mut kv, &request.request.prompt_tokens, chunk)
        } else {
            let resume = kv.local.seq_len();
            activate_rank(&self.ctx)?;
            let result =
                self.serve
                    .step(&self.ctx, &mut kv, &request.request.prompt_tokens[resume..]);
            // Started after rank 0 and unconditionally: a rank-0 failure here
            // must not skip the peer (see `step_extra_ranks`).
            self.prefill_extra_ranks(&mut kv, &request.request.prompt_tokens[resume..])?;
            result
        };
        let mut logits = match stepped {
            Ok(logits) => logits,
            Err(err) => {
                // This prompt's prefill failed; its pages return with `kv`
                // and the engine keeps serving.
                log::error!("solo prefill failed: {err:#}");
                ledger.fail(request.id, format!("prefill failed: {err:#}"));
                return Ok(Admitted::Done);
            }
        };
        capture_prefix(
            &self.ctx,
            &self.serve,
            &mut self.prefix_cache,
            &kv,
            &request.request.prompt_tokens,
            resumed,
        );
        Ok(self.first_token_flow(request, kv, &mut logits, echo, ledger))
    }

    /// Sample and settle a prefill's first token from logits row 0 — the
    /// shared tail of a sync admission and an overlapped-prefill join.
    fn first_token_flow(
        &mut self,
        request: QueuedRequest,
        kv: GemmaKv,
        logits: &mut HiddenStates,
        echo: Option<PromptEcho>,
        ledger: &mut RequestLedger,
    ) -> Admitted {
        let sampled = sample_logits_rows(
            &self.ctx,
            &self.suppress_ids,
            &self.policy,
            &mut self.scratch,
            &mut self.seed,
            &[SampleRow::of(&request.request, 0)],
            logits,
        );
        let mut sampled = match sampled {
            Ok(sampled) => sampled,
            Err(err) => {
                log::error!("first-token sampling failed: {err:#}");
                ledger.fail(request.id, format!("first-token sampling failed: {err:#}"));
                return Admitted::Done;
            }
        };
        match settle_first_token(request, kv, sampled.token(0), echo, ledger) {
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
        request: QueuedRequest,
        mut kv: GemmaKv,
        resumed: Option<u64>,
        ledger: &mut RequestLedger,
    ) -> Result<Admitted> {
        let lane = self.lane.as_mut().expect("gated by the caller");
        debug_assert!(lane.inflight.is_none());
        // A restored prefix is already in the KV: the lane prefills only the
        // unseen suffix, exactly like the sync and mixed paths.
        let resume = kv.local.seq_len();
        let launched = {
            let _guard = unsafe {
                pegainfer_core::tensor::StreamOverrideGuard::activate(lane.stream.stream)
            };
            self.serve.prefill_into_logits(
                &self.ctx,
                &mut kv,
                &request.request.prompt_tokens[resume..],
            )
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
                Ok(Admitted::Done)
            }
            Err(err) => {
                // This prompt's launch failed. Drain the lane so no stale
                // kernel can write the pages `kv` returns, fail the request,
                // keep serving; only a failed drain is engine-fatal.
                log::error!("async prefill launch failed: {err:#}");
                lane.drain()?;
                ledger.fail(request.id, format!("prefill failed: {err:#}"));
                Ok(Admitted::Done)
            }
        }
    }

    /// Join a completed overlapped prefill: run the deferred window
    /// release, capture into the prefix cache, and take the first-token
    /// flow the sync path uses.
    fn join_async_prefill(
        &mut self,
        active: &mut Vec<Active>,
        ledger: &mut RequestLedger,
    ) -> Result<()> {
        if self
            .lane
            .as_ref()
            .is_some_and(|lane| lane.inflight.is_some())
        {
            self.drain_pipeline(active, ledger)?;
        }
        let Some(lane) = self.lane.as_mut() else {
            return Ok(());
        };
        let Some(inflight) = lane.inflight.take() else {
            return Ok(());
        };
        let InflightPrefill {
            request,
            mut kv,
            mut pass,
            resumed,
        } = inflight;
        if ledger.is_aborted(request.id) {
            ledger.retire(request.id);
            return Ok(());
        }
        // The frontier after any prefill equals the prompt length; a lane
        // pass that processed the wrong suffix cannot pass this gate.
        if kv.local.seq_len() != request.request.prompt_tokens.len() {
            let message = format!(
                "gemma4 async prefill frontier {} != prompt {}",
                kv.local.seq_len(),
                request.request.prompt_tokens.len()
            );
            log::error!("{message}");
            ledger.fail(request.id, message);
            return Ok(());
        }
        if let Err(err) = self.serve.release_prefill_window(&mut kv) {
            log::error!("async prefill window release failed: {err:#}");
            ledger.fail(request.id, format!("prefill failed: {err:#}"));
            return Ok(());
        }
        capture_prefix(
            &self.ctx,
            &self.serve,
            &mut self.prefix_cache,
            &kv,
            &request.request.prompt_tokens,
            resumed,
        );
        if let Admitted::Active(entry) =
            self.first_token_flow(request, kv, &mut pass.logits, None, ledger)
        {
            active.push(*entry);
        }
        Ok(())
    }

    /// A prompt's unseen suffix on the plain path, one `chunk` segment at a
    /// time.
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
        self.admit_extra_ranks(kv, tokens.len())?;
        activate_rank(&self.ctx)?;
        // Rank 0's verdict is held rather than propagated: the extra ranks have
        // to be driven either way (see `step_extra_ranks`), and their failure
        // wins.
        let logits = self.serve.step(&self.ctx, kv, tokens);
        self.prefill_extra_ranks(kv, tokens)?;
        logits
    }

    fn finish_plain_walker(
        &mut self,
        walker: &mut Walker,
        chunk: usize,
        active: &mut Vec<Active>,
        ledger: &mut RequestLedger,
    ) -> Result<()> {
        let mut logits = match self.walk_plain_prompt(
            &mut walker.kv,
            &walker.request.request.prompt_tokens,
            chunk,
        ) {
            Ok(logits) => logits,
            Err(err) => {
                ledger.fail(walker.request.id, format!("walk tail failed: {err:#}"));
                walker.failed = true;
                return Ok(());
            }
        };
        walker.offset = walker.request.request.prompt_tokens.len();
        let mut sampled = mixed_head_flow(
            &self.ctx,
            &self.suppress_ids,
            &self.policy,
            &mut self.scratch,
            &mut self.seed,
            &[SampleRow::of(&walker.request.request, 0)],
            active,
            &mut logits,
            ledger,
        )?;
        walker.first = Some(sampled.token(0));
        Ok(())
    }

    fn start_mixed_walk(chunk: usize, newcomers: Vec<Newcomer>) -> Walk {
        let walkers = newcomers
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
        Walk {
            walkers,
            chunk,
            first_round: true,
        }
    }

    /// Advance exactly one chunked-walk round. Returns `true` when no walker
    /// remains. The caller drops the scheduler's `walk` only after this
    /// boundary so the driver can commit this round's live tokens first.
    fn advance_walk_round(
        &mut self,
        walk: &mut Walk,
        active: &mut Vec<Active>,
        ledger: &mut RequestLedger,
    ) -> Result<bool> {
        for walker in &mut walk.walkers {
            if !walker.failed && ledger.is_aborted(walker.request.id) {
                ledger.retire(walker.request.id);
                walker.failed = true;
            }
        }
        self.graduate_ready_walkers(&mut walk.walkers, active, ledger);
        if !walk
            .walkers
            .iter()
            .any(|w| !w.failed && w.offset < w.request.request.prompt_tokens.len())
        {
            return Ok(true);
        }
        if !walk.first_round {
            self.ready_decode_rows(active, ledger);
        }
        walk.first_round = false;

        if active.is_empty() {
            for walker in &mut walk.walkers {
                if walker.failed || walker.offset >= walker.request.request.prompt_tokens.len() {
                    continue;
                }
                self.finish_plain_walker(walker, walk.chunk, active, ledger)?;
            }
            self.graduate_ready_walkers(&mut walk.walkers, active, ledger);
            return Ok(true);
        }

        let mut budget = walk.chunk;
        let mut takes: Vec<Option<(usize, bool)>> = vec![None; walk.walkers.len()];
        for (index, walker) in walk.walkers.iter_mut().enumerate() {
            if walker.failed {
                continue;
            }
            let rest = walker.request.request.prompt_tokens.len() - walker.offset;
            if rest == 0 || budget == 0 {
                continue;
            }
            let take = rest.min(budget);
            if let Err(err) = admit_tokens(
                &self.serve.local_pool,
                &self.serve.global_pool,
                &mut walker.kv,
                take,
            )
            .and_then(|()| self.admit_extra_ranks(&mut walker.kv, take))
            {
                ledger.fail(
                    walker.request.id,
                    format!("walk segment admission failed: {err:#}"),
                );
                walker.failed = true;
                continue;
            }
            takes[index] = Some((take, take == rest));
            budget -= take;
        }

        let sampled = {
            let mut prefills: Vec<(&mut GemmaKv, &[u32])> = Vec::new();
            let mut head: Vec<SampleRow<'_>> = Vec::new();
            for (walker, take) in walk.walkers.iter_mut().zip(&takes) {
                let Some((take, last)) = *take else {
                    continue;
                };
                let request = &walker.request.request;
                prefills.push((
                    &mut walker.kv,
                    &request.prompt_tokens[walker.offset..walker.offset + take],
                ));
                head.push(if last {
                    SampleRow::of(request, 0)
                } else {
                    SampleRow::discarded(request)
                });
            }
            self.mixed_step(&mut prefills, &head, active, ledger)
        };
        let mut sampled = match sampled {
            Ok(sampled) => sampled,
            Err(err) => {
                let walkers = walk.walkers.drain(..).map(|walker| walker.request.id);
                fail_requests(walkers, "walk step", &err, ledger);
                return Err(err);
            }
        };
        let mut sampled_index = 0usize;
        for (walker, take) in walk.walkers.iter_mut().zip(&takes) {
            if let Some((take, last)) = *take {
                walker.offset += take;
                if last {
                    walker.first = Some(sampled.token(sampled_index));
                }
                sampled_index += 1;
            }
        }
        self.graduate_ready_walkers(&mut walk.walkers, active, ledger);
        Ok(!walk
            .walkers
            .iter()
            .any(|walker| !walker.failed && walker.first.is_none()))
    }

    fn graduate_ready_walkers(
        &mut self,
        walkers: &mut Vec<Walker>,
        active: &mut Vec<Active>,
        ledger: &mut RequestLedger,
    ) {
        let mut index = 0;
        while index < walkers.len() {
            if !walkers[index].failed && walkers[index].first.is_some() {
                let walker = walkers.remove(index);
                self.graduate_walker(walker, active, ledger);
            } else {
                index += 1;
            }
        }
    }

    /// One finished walker joins the batch at its round boundary: capture
    /// its prompt state, then emit or retire its first token exactly like
    /// the whole-prompt form.
    fn graduate_walker(&mut self, w: Walker, active: &mut Vec<Active>, ledger: &mut RequestLedger) {
        let Walker {
            request,
            kv,
            resumed,
            first,
            ..
        } = w;
        let token = first.expect("graduation follows a final segment");
        capture_prefix(
            &self.ctx,
            &self.serve,
            &mut self.prefix_cache,
            &kv,
            &request.request.prompt_tokens,
            resumed,
        );
        if let Some(entry) = settle_first_token(request, kv, token, None, ledger) {
            active.push(entry);
        }
    }

    /// Retire every aborted row and fail a request whose KV cannot grow by
    /// one token; admit the step's token for the rest.
    fn ready_decode_rows(&self, active: &mut Vec<Active>, ledger: &mut RequestLedger) {
        let mut row = 0;
        while row < active.len() {
            let entry = &mut active[row];
            if entry.stopping {
                active.swap_remove(row);
                continue;
            }
            if ledger.is_aborted(entry.request.id) {
                ledger.retire(entry.request.id);
                active.swap_remove(row);
                continue;
            }
            if let Err(err) = admit_tokens(
                &self.serve.local_pool,
                &self.serve.global_pool,
                &mut entry.kv,
                1,
            )
            .and_then(|()| self.admit_extra_ranks(&mut entry.kv, 1))
            {
                ledger.fail(
                    entry.request.id,
                    format!("decode KV admission failed: {err:#}"),
                );
                active.swap_remove(row);
                continue;
            }
            row += 1;
        }
    }

    fn pipeline_eligible(&self, active: &[Active], ledger: &RequestLedger) -> bool {
        !active.is_empty()
            && active.len() <= self.scratch.max_rows()
            && active.iter().all(|entry| {
                !entry.stopping
                    && entry.request.request.logprobs.is_none()
                    && entry
                        .request
                        .request
                        .max_tokens
                        .saturating_sub(ledger.completion_tokens(entry.request.id))
                        >= DECODE_PIPELINE_DEPTH
                    && pegainfer_sample::effectively_greedy(
                        &entry.request.request.params,
                        self.scratch.vocab(),
                    )
            })
    }

    /// Reserve the next token without retiring or reordering a row.
    fn ready_rows_pinned(&self, active: &mut [Active], ledger: &RequestLedger) -> bool {
        active.iter_mut().all(|entry| {
            !ledger.is_aborted(entry.request.id)
                && admit_tokens(
                    &self.serve.local_pool,
                    &self.serve.global_pool,
                    &mut entry.kv,
                    1,
                )
                .and_then(|()| self.admit_extra_ranks(&mut entry.kv, 1))
                .is_ok()
        })
    }

    fn fence(&self) -> Result<()> {
        let sync = unsafe { cudarc::driver::sys::cuStreamSynchronize(self.ctx.stream.cu_stream()) };
        anyhow::ensure!(
            sync == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "cuStreamSynchronize(decode) failed ({sync:?})"
        );
        Ok(())
    }

    /// Queue one decode and stage its greedy picks into the next embedding's
    /// id buffer and one pinned readback slot.
    fn launch_staged(
        &mut self,
        active: &mut [Active],
        resident: bool,
        slot: usize,
    ) -> Result<usize> {
        let rows = active.len();
        // The staged pipeline hands rank 1 no ids of its own (it has no
        // sampler), so under tensor parallelism every rank takes the
        // explicit-token path.
        let resident = resident && self.more.is_empty();
        let tokens = (!resident).then(|| active.iter().map(|entry| entry.next).collect::<Vec<_>>());
        {
            activate_rank(&self.ctx)?;
            let mut kvs = rank_kvs(active, 0);
            // Rank 0's verdict is held, not propagated: the other ranks must be
            // driven either way (see `step_extra_ranks`).
            let rank0 = if let Some(tokens) = tokens.as_deref() {
                self.serve
                    .decode_batch_step(&self.ctx, &mut self.arena, &mut kvs, tokens)
                    .map(|_| ())
            } else {
                self.serve
                    .decode_batch_step_resident(&self.ctx, &mut self.arena, &mut kvs)
                    .map(|_| ())
            };
            // The other ranks carry the same tokens and page ids; only rank 0
            // samples, so their logits are discarded. A failure on them is fatal
            // — the ranks' frontiers must not drift apart.
            if let Some(tokens) = tokens.as_deref() {
                for (rank, state) in self.more.iter_mut().enumerate() {
                    activate_rank(&state.ctx)?;
                    let mut kvs = rank_kvs(active, rank + 1);
                    state.serve.decode_batch_step(
                        &state.ctx,
                        &mut state.arena,
                        &mut kvs,
                        tokens,
                    )?;
                }
                // The sampler below reads rank 0's arena.
                activate_rank(&self.ctx)?;
            }
            rank0?;
        }
        let graph_slot = crate::serve::decode_bucket_slot(rows);
        if let Some(graph) = self.sampler_graphs.get_mut(graph_slot) {
            graph
                .launch_captured(&self.ctx)
                .context("launch sampler graph")?;
            self.seed.next_call();
            pegainfer_sample::greedy_stage_readback(&self.ctx, slot, &mut self.scratch)
                .context("stage greedy readback")?;
        } else {
            let (logits, ids) = self.arena.logits_and_ids();
            ops::suppress_logits_bf16_in_place(&self.ctx, logits, &self.suppress_ids)
                .context("suppression")?;
            self.seed.next_call();
            pegainfer_sample::greedy_stage_resident(
                &self.ctx,
                logits,
                rows,
                ids,
                slot,
                &mut self.scratch,
            )
            .context("stage greedy picks")?;
        }
        Ok(rows)
    }

    /// Deliver a staged step without changing the row order. Finished rows
    /// stay pinned until the speculative successor drains.
    fn collect_pending(
        &mut self,
        active: &mut [Active],
        pending: &PendingDecode,
        ledger: &mut RequestLedger,
    ) -> Result<()> {
        anyhow::ensure!(
            pending.rows == active.len(),
            "pipeline collected {} rows against a batch of {}",
            pending.rows,
            active.len()
        );
        let picked = pegainfer_sample::greedy_collect_resident(
            pending.rows,
            pending.slot,
            &mut self.scratch,
        )?;
        for (entry, token) in active.iter_mut().zip(picked) {
            entry.settle_staged(&self.policy, token, ledger);
        }
        Ok(())
    }

    fn drain_pipeline(
        &mut self,
        active: &mut Vec<Active>,
        ledger: &mut RequestLedger,
    ) -> Result<()> {
        let Some(pending) = self.pipeline.take() else {
            return Ok(());
        };
        if let Err(err) = self.collect_pending(active, &pending, ledger) {
            self.fence()?;
            fail_active_batch(active, "pipelined decode drain", &err, ledger);
            return Err(err.context("pipelined decode drain"));
        }
        active.retain(|entry| !entry.stopping);
        Ok(())
    }

    /// One step that prefills `prefills` alongside every active row's next
    /// token. `head` samples the prefill rows; the active rows follow and get
    /// their events here. On `Err` the active batch has been failed.
    fn mixed_step(
        &mut self,
        prefills: &mut [(&mut GemmaKv, &[u32])],
        head: &[SampleRow<'_>],
        active: &mut Vec<Active>,
        ledger: &mut RequestLedger,
    ) -> Result<SampledRows> {
        let decode_tokens: Vec<u32> = active.iter().map(|entry| entry.next).collect();
        let stepped = {
            activate_rank(&self.ctx)?;
            let mut kvs = rank_kvs(active, 0);
            let mut rank0_prefills: Vec<(&mut RankKv, &[u32])> = prefills
                .iter_mut()
                .map(|(kv, tokens)| (&mut ***kv, *tokens))
                .collect();
            self.serve.mixed_prefill_decode_step(
                &self.ctx,
                &mut self.arena,
                &mut rank0_prefills,
                &mut kvs,
                &decode_tokens,
            )
        };
        // The other ranks run the same mixed step on their own families and
        // arena; their logits are discarded.
        for (rank, state) in self.more.iter_mut().enumerate() {
            activate_rank(&state.ctx)?;
            let mut extra_prefills: Vec<(&mut RankKv, &[u32])> = prefills
                .iter_mut()
                .map(|(kv, tokens)| (kv.core_mut(rank + 1), *tokens))
                .collect();
            let mut kvs = rank_kvs(active, rank + 1);
            state.serve.mixed_prefill_decode_step(
                &state.ctx,
                &mut state.arena,
                &mut extra_prefills,
                &mut kvs,
                &decode_tokens,
            )?;
        }
        activate_rank(&self.ctx)?;
        let logits = match stepped {
            Ok(logits) => logits,
            Err(err) => {
                fail_active_batch(active, "mixed step", &err, ledger);
                return Err(err.context("gemma4 mixed step"));
            }
        };
        mixed_head_flow(
            &self.ctx,
            &self.suppress_ids,
            &self.policy,
            &mut self.scratch,
            &mut self.seed,
            head,
            active,
            logits,
            ledger,
        )
    }

    /// The mixed-admission tail of [`Self::admit_and_prefill`]: every
    /// gathered prompt and the live decode batch share one step, then one
    /// sampler call covers the newcomers' first tokens (logits rows `0..k`)
    /// and every active row after them. Finished newcomers emit in place
    /// and the rest join `active` directly, so the caller always receives
    /// `Done`.
    fn mixed_admission(
        &mut self,
        mut newcomers: Vec<Newcomer>,
        active: &mut Vec<Active>,
        walk: &mut Option<Walk>,
        ledger: &mut RequestLedger,
    ) -> Result<Admitted> {
        if let Some(chunk) = self.mix_chunk {
            debug_assert!(walk.is_none());
            *walk = Some(Self::start_mixed_walk(chunk, newcomers));
            return Ok(Admitted::Done);
        }

        let sampled = {
            let mut prefills: Vec<(&mut GemmaKv, &[u32])> = Vec::with_capacity(newcomers.len());
            let mut head: Vec<SampleRow<'_>> = Vec::with_capacity(newcomers.len());
            for (request, kv, _) in &mut newcomers {
                let resume = kv.local.seq_len();
                let request = &request.request;
                prefills.push((kv, &request.prompt_tokens[resume..]));
                head.push(SampleRow::of(request, 0));
            }
            self.mixed_step(&mut prefills, &head, active, ledger)
        };
        let mut sampled = match sampled {
            Ok(sampled) => sampled,
            Err(err) => {
                let newcomers = newcomers.drain(..).map(|(request, _, _)| request.id);
                fail_requests(newcomers, "mixed step", &err, ledger);
                return Err(err);
            }
        };
        for (request, kv, resumed) in &newcomers {
            capture_prefix(
                &self.ctx,
                &self.serve,
                &mut self.prefix_cache,
                kv,
                &request.request.prompt_tokens,
                *resumed,
            );
        }

        // The newcomers: their first tokens are logits rows `0..k`.
        for (j, (request, kv, _)) in newcomers.into_iter().enumerate() {
            if let Some(entry) = settle_first_token(request, kv, sampled.token(j), None, ledger) {
                active.push(entry);
            }
        }
        Ok(Admitted::Done)
    }

    fn decode_round_collect(
        &mut self,
        active: &mut Vec<Active>,
        ledger: &mut RequestLedger,
    ) -> Result<()> {
        let tokens: Vec<u32> = active.iter().map(|entry| entry.next).collect();
        activate_rank(&self.ctx)?;
        // Rank 0's verdict is held rather than propagated: the extra ranks have
        // to be driven either way (see `step_extra_ranks`).
        let rank0 = {
            let mut kvs = rank_kvs(active, 0);
            self.serve
                .decode_batch_step(&self.ctx, &mut self.arena, &mut kvs, &tokens)
                .map(|_| ())
        };
        // The other ranks carry the same tokens and page ids; only rank 0
        // samples. A failure on them is fatal — the ranks' frontiers must not
        // drift apart.
        for (rank, state) in self.more.iter_mut().enumerate() {
            activate_rank(&state.ctx)?;
            let mut kvs = rank_kvs(active, rank + 1);
            state
                .serve
                .decode_batch_step(&state.ctx, &mut state.arena, &mut kvs, &tokens)?;
        }
        activate_rank(&self.ctx)?;
        if let Err(err) = rank0 {
            self.fence()?;
            fail_active_batch(active, "batched decode", &err, ledger);
            return Err(err.context("batched decode"));
        }
        let (logits, _) = self.arena.logits_and_ids();
        let sampled = {
            let rows: Vec<SampleRow<'_>> = active
                .iter()
                .map(|entry| entry.sample_row(ledger))
                .collect();
            sample_logits_rows(
                &self.ctx,
                &self.suppress_ids,
                &self.policy,
                &mut self.scratch,
                &mut self.seed,
                &rows,
                logits,
            )
        };
        match sampled {
            Ok(mut sampled) => emit_decode_rows(active, &mut sampled, 0, ledger),
            Err(err) => {
                self.fence()?;
                fail_active_batch(active, "batched decode", &err, ledger);
                return Err(err.context("batched decode sampling"));
            }
        }
        Ok(())
    }

    /// One batched decode step. Eligible greedy batches keep one speculative
    /// successor in flight and drain before any row-order change.
    fn decode_round(&mut self, active: &mut Vec<Active>, ledger: &mut RequestLedger) -> Result<()> {
        if let Some(pending) = self.pipeline.take() {
            if self.pipeline_eligible(active, ledger) && self.ready_rows_pinned(active, ledger) {
                let next_slot = (pending.slot + 1) % DECODE_PIPELINE_DEPTH;
                match self.launch_staged(active, true, next_slot) {
                    Ok(rows) => {
                        if let Err(err) = self.collect_pending(active, &pending, ledger) {
                            self.fence()?;
                            fail_active_batch(active, "pipelined decode collect", &err, ledger);
                            return Err(err.context("pipelined decode collect"));
                        }
                        self.pipeline = Some(PendingDecode {
                            rows,
                            slot: next_slot,
                        });
                    }
                    Err(err) => {
                        if let Err(collect_err) = self.collect_pending(active, &pending, ledger) {
                            log::error!("collect during launch failure failed: {collect_err:#}");
                        }
                        self.fence()?;
                        fail_active_batch(active, "pipelined decode launch", &err, ledger);
                        return Err(err.context("pipelined decode launch"));
                    }
                }
                return Ok(());
            }
            self.pipeline = Some(pending);
            self.drain_pipeline(active, ledger)?;
            if active.is_empty() {
                return Ok(());
            }
        }

        self.ready_decode_rows(active, ledger);
        if active.is_empty() {
            return Ok(());
        }
        if self.pipeline_eligible(active, ledger) {
            match self.launch_staged(active, false, 0) {
                Ok(rows) => self.pipeline = Some(PendingDecode { rows, slot: 0 }),
                Err(err) => {
                    self.fence()?;
                    fail_active_batch(active, "batched decode", &err, ledger);
                    return Err(err.context("batched decode launch"));
                }
            }
            return Ok(());
        }
        self.decode_round_collect(active, ledger)
    }
}

impl Gemma4Scheduler {
    fn advance_walk(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        let Some(mut walk) = self.walk.take() else {
            return Ok(());
        };
        let done = self
            .state
            .advance_walk_round(&mut walk, &mut self.active, ledger)?;
        if !done {
            self.walk = Some(walk);
        }
        Ok(())
    }
}

impl Scheduler for Gemma4Scheduler {
    fn submit(&mut self, request: QueuedRequest) {
        self.pending.push_back(request);
    }

    fn step(&mut self, ledger: &mut RequestLedger) -> Result<()> {
        if self.cublas.is_none() {
            // Rank 0's handles, and the guard that tears every device's handles
            // down on this thread. Each other rank's handles are created the
            // first time `activate_rank` reaches it.
            self.cublas = Some(bind_engine_thread(&self.state.ctx)?);
            activate_rank(&self.state.ctx)?;
        }
        if self.walk.is_some() {
            return self.advance_walk(ledger);
        }

        let lane_ready = match self.state.lane.as_ref() {
            Some(lane) if lane.inflight.is_some() => {
                let lane_is_only_work = self.active.is_empty() && self.pending.is_empty();
                if lane_is_only_work {
                    lane.drain()?;
                }
                lane_is_only_work || lane.inflight_complete()?
            }
            _ => false,
        };
        if lane_ready {
            self.state.join_async_prefill(&mut self.active, ledger)?;
        }

        self.state.intake_turn(
            &mut self.door,
            &mut self.pending,
            &mut self.active,
            &mut self.walk,
            std::time::Instant::now(),
            ledger,
        )?;
        if self.walk.is_some() {
            return self.advance_walk(ledger);
        }
        if !self.active.is_empty() {
            self.state.decode_round(&mut self.active, ledger)?;
        }
        Ok(())
    }

    fn metrics(&self) -> SchedulerMetrics {
        let local_total = self.state.serve.local_pool.capacity_pages();
        let global_total = self.state.serve.global_pool.capacity_pages();
        let local_used = local_total.saturating_sub(self.state.serve.local_pool.available_pages());
        let global_used =
            global_total.saturating_sub(self.state.serve.global_pool.available_pages());
        let walkers = self.walk.as_ref().map_or(0, |walk| {
            walk.walkers.iter().filter(|walker| !walker.failed).count()
        });
        let lane_inflight = self
            .state
            .lane
            .as_ref()
            .is_some_and(|lane| lane.inflight.is_some()) as usize;
        SchedulerMetrics {
            kv_used_blocks: (local_used + global_used) as u64,
            kv_total_blocks: (local_total + global_total) as u64,
            num_running_reqs: (self.active.len() + walkers + lane_inflight) as u64,
            num_waiting_reqs: self.pending.len() as u64,
            spec_decode: None,
        }
    }
}

/// One sampled pick, with the stop the sampler decided for it.
struct SampledToken {
    id: u32,
    logprob: Option<TokenLogprob>,
    stop: bool,
}

/// Settle one pick for its request, first token or later: an aborted request
/// retires with no event, and a stop token retires it without being emitted
/// (the frontend appends its own sentinel for a terminal Stop and drops the
/// last id, so an engine that emits EOS costs the client its final visible
/// token). Any other token is emitted and finishes the request at
/// `max_tokens`. Returns whether the request is done.
fn settle_token(entry: &mut Active, token: SampledToken, ledger: &mut RequestLedger) -> bool {
    let id = entry.request.id;
    if ledger.is_aborted(id) {
        ledger.retire(id);
        return true;
    }
    if token.stop {
        ledger.finish(id, FinishReason::Stop);
        return true;
    }
    ledger.push_tokens(id, &[token.id], &[token.logprob]);
    if ledger.completion_tokens(id) >= entry.request.request.max_tokens {
        ledger.finish(id, FinishReason::Length);
        return true;
    }
    entry.next = token.id;
    false
}

/// Settle one decode step's picks for every active row and retire the
/// finished ones — the flow both the pure decode round and the mixed step
/// share; `row_base` is the rows' offset into the step's logits (a mixed
/// step's first `row_base` rows are its prompts).
fn emit_decode_rows(
    active: &mut Vec<Active>,
    sampled: &mut SampledRows,
    row_base: usize,
    ledger: &mut RequestLedger,
) {
    let mut retire: Vec<usize> = Vec::new();
    for (row, entry) in active.iter_mut().enumerate() {
        if settle_token(entry, sampled.token(row + row_base), ledger) {
            retire.push(row);
        }
    }
    for row in retire.into_iter().rev() {
        active.swap_remove(row);
    }
}

/// Settle one admission's first token, its prompt echo ahead of it; a
/// request that is not done joins the decode batch.
fn settle_first_token(
    request: QueuedRequest,
    kv: GemmaKv,
    token: SampledToken,
    echo: Option<PromptEcho>,
    ledger: &mut RequestLedger,
) -> Option<Active> {
    if let Some(echo) = echo.filter(|_| !ledger.is_aborted(request.id)) {
        ledger.echo_prompt(request.id, echo);
    }
    let mut entry = Active {
        request,
        kv,
        next: token.id,
        stopping: false,
    };
    (!settle_token(&mut entry, token, ledger)).then_some(entry)
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
    fn fp8_knob_parses_or_refuses() {
        for off in ["off", "", "0"] {
            assert_eq!(parse_kv_fp8(off).unwrap(), KvStorage::Bf16);
        }
        assert_eq!(parse_kv_fp8(" LOCAL ").unwrap(), KvStorage::E4m3);
        assert!(parse_kv_fp8("global").is_err());
    }

    #[test]
    fn global_attn_parses_or_refuses() {
        for off in ["off", "", "0"] {
            assert_eq!(
                parse_global_attn(off).expect("off parses"),
                GlobalAttn::Incumbent
            );
        }
        assert_eq!(
            parse_global_attn(" TileLang ").expect("trimmed and cased"),
            GlobalAttn::TileLang
        );
        assert_eq!(
            parse_global_attn("tilelang640").expect("folded parses"),
            GlobalAttn::TileLangFolded
        );
        for bad in [
            "on",
            "1",
            "flashinfer",
            "tile",
            "tilelang:1",
            "640",
            "tilelang1024",
        ] {
            assert!(parse_global_attn(bad).is_err(), "{bad:?} must refuse");
        }
    }

    #[test]
    fn admit_coalesce_parses_or_refuses() {
        for off in ["off", "0", ""] {
            assert_eq!(parse_admit_coalesce_ms(off).unwrap(), None);
        }
        assert_eq!(
            parse_admit_coalesce_ms("300").unwrap(),
            Some(std::time::Duration::from_millis(300))
        );
        assert_eq!(
            parse_admit_coalesce_ms("1").unwrap(),
            Some(std::time::Duration::from_millis(1))
        );
        assert_eq!(
            parse_admit_coalesce_ms("2000").unwrap(),
            Some(std::time::Duration::from_millis(2000))
        );
        for bad in ["0x", "2001", "abc"] {
            assert!(parse_admit_coalesce_ms(bad).is_err(), "{bad:?} must refuse");
        }
    }

    fn test_door() -> CoalesceDoor {
        CoalesceDoor::new(std::time::Duration::from_millis(100), MIX_MAX_PROMPTS)
    }

    #[test]
    fn coalesce_door_idle_opens_and_clears_the_timer() {
        let now = std::time::Instant::now();
        let mut door = test_door();
        door.since = Some(now);
        assert!(door.opens(1, 0, 8, now));
        assert_eq!(door.since, None);
    }

    #[test]
    fn coalesce_door_empty_queue_opens() {
        let now = std::time::Instant::now();
        let mut door = test_door();
        assert!(door.opens(0, 4, 8, now));
        assert_eq!(door.since, None);
    }

    #[test]
    fn coalesce_door_shallow_batch_opens() {
        let now = std::time::Instant::now();
        let mut door = test_door();
        assert!(door.opens(1, 1, 8, now));
        assert_eq!(door.since, None);
    }

    #[test]
    fn coalesce_door_deep_under_cohort_batch_closes_and_pins_the_timer() {
        let now = std::time::Instant::now();
        let mut door = test_door();
        assert!(!door.opens(1, 3, 8, now));
        assert_eq!(door.since, Some(now));
        assert!(!door.opens(1, 3, 8, now + std::time::Duration::from_millis(1)));
        assert_eq!(door.since, Some(now));
    }

    #[test]
    fn coalesce_door_full_cohort_opens_and_clears() {
        let now = std::time::Instant::now();
        let mut door = test_door();
        door.since = Some(now);
        assert!(door.opens(4, 4, 8, now));
        assert_eq!(door.since, None);
    }

    #[test]
    fn coalesce_door_elapsed_window_opens() {
        let now = std::time::Instant::now();
        let mut door = test_door();
        let window = door.window;
        assert!(!door.opens(1, 3, 8, now));
        assert!(door.opens(1, 3, 8, now + window));
        assert_eq!(door.since, None);
    }

    #[test]
    fn coalesce_door_full_roster_opens_for_one_capacity_bounded_arrival() {
        let now = std::time::Instant::now();
        let mut door = test_door();
        assert!(door.opens(1, 8, 8, now));
        assert_eq!(door.since, None);
    }

    #[test]
    fn coalesce_door_freed_slots_rederive_the_cohort() {
        let now = std::time::Instant::now();
        let mut door = test_door();
        assert!(door.opens(2, 6, 8, now));
        assert!(!door.opens(2, 5, 8, now));
        assert_eq!(door.since, Some(now));
    }

    #[test]
    fn coalesce_door_cohort_follows_the_prompt_bound() {
        let now = std::time::Instant::now();
        let mut door = CoalesceDoor::new(std::time::Duration::from_millis(100), 8);
        assert!(!door.opens(4, 8, 16, now));
        assert!(door.opens(8, 8, 16, now));
    }

    #[test]
    fn gather_bounds_parse_or_refuse() {
        assert_eq!(parse_mix_gather_rows("8192", 8192).unwrap(), 8192);
        assert_eq!(parse_mix_gather_rows(" 512 ", 8192).unwrap(), 512);
        for bad in ["0", "8193", "off", "-1", "4k"] {
            assert!(
                parse_mix_gather_rows(bad, 8192).is_err(),
                "{bad:?} must refuse: a step past the ceiling fails mid-run, where a \
                 client reads the error-finished stream as served"
            );
        }
        assert_eq!(parse_mix_max_prompts("16", 16).unwrap(), 16);
        for bad in ["0", "17", "all"] {
            assert!(
                parse_mix_max_prompts(bad, 16).is_err(),
                "{bad:?} must refuse"
            );
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
#[path = "engine/lane_step_collector.rs"]
mod lane_step_collector;

#[cfg(test)]
#[path = "engine/lane_tests.rs"]
mod lane_tests;

#[cfg(test)]
#[path = "engine/lane_gates_lifecycle.rs"]
mod lane_gates_lifecycle;

#[cfg(test)]
#[path = "engine/lane_gates_roster.rs"]
mod lane_gates_roster;

#[cfg(test)]
#[path = "engine/lane_gates_walk.rs"]
mod lane_gates_walk;

#[cfg(test)]
#[path = "engine/lane_gates_logprobs.rs"]
mod lane_gates_logprobs;

#[cfg(test)]
#[path = "engine/lane_gates_tp.rs"]
mod lane_gates_tp;
