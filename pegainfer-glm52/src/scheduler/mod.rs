//! Free-running per-rank engines. Every logical DP rank is an autonomous
//! engine: its own request queue, slots, KV pool, offload/P-D state, and
//! load feed, driven by its own thread. There is no coordinator — the only
//! coupling between engines is the fixed-cadence DeepEP collective chain
//! itself (75 MoE layers per step plus the fixed MTP round): every engine
//! steps unconditionally, idle ranks enter with padding rows, and the
//! collective's back-pressure is the synchronization. The invariants that
//! make that safe are structural, not negotiated: a fixed chain (no
//! conditional collectives), conservative protocol-max shapes, and
//! deterministic padding rows (`docs/models/glm52/free-running-dp.md` §4).
//!
//! Each engine admits up to `GLM52_MAX_BATCH_PER_RANK` requests from its own
//! queue (the `EngineHandle` routes frontend requests to rank queues by
//! `data_parallel_rank`). KV pages come from the rank's [`BlockPool`]
//! (64-token pages, content-hashed blocks): admission reserves a request's
//! full-lifetime page count up front (honor-or-reject — a request that can
//! never fit is rejected, one that can't fit *now* stays queued), so decode
//! can never run out of pages mid-request, and released requests' sealed
//! blocks stay matchable as the prefix cache.
//!
//! Every step the engine forwards its OWN batch bucket — each active slot's
//! *span* of next tokens (mid-prefill slots batch up to a bucket of
//! consecutive prompt positions through one step; decode slots feed one
//! row), idle slots feed a padding row whose output is discarded. The
//! bucket is the smallest member of `GLM52_DECODE_BUCKETS` covering the
//! rank's own row demand. The TP4 replicated topology is the N=1
//! special case: ONE logical rank's engine drives every mirrored worker
//! with the identical step, and the join asserts bit-identical results (the
//! replicated-activations contract); as the sole issuer of its collectives
//! it may block while fully idle instead of free-running.
//!
//! The per-request decisions (what to feed next, what a step's output means)
//! live in [`Glm52SlotState`] as pure data transitions, and the
//! admission/step-shape decisions in [`admission`] / [`plan`] as pure
//! functions over the occupancy and feed wants.

mod admission;
#[cfg(test)]
mod contract_tests;
mod graph;
mod load;
mod mtp;
mod offload;
mod plan;
mod slot;
#[cfg(test)]
mod testkit;

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use admission::admit_from_queue;
use anyhow::Context as _;
use graph::GraphDumpRequest;
use graph::dump_rank0_decode_graph;
use graph::precapture_step_graphs;
use load::publish_load;
use mtp::run_mtp_round;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::KvPrefix;
use pegainfer_frontend::engine::SchedulerMetrics;
use pegainfer_frontend::engine::SubmittedRequest;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_kv_store::BlockPool;
use pegainfer_kv_store::KvStore;
use pegainfer_kv_store::RequestKv;
use pegainfer_kv_store::SaveClass;
use pegainfer_kv_store::SaveCursor;
use pegainfer_sample::mix_seed;
use plan::collect_sampling_rows;
use plan::feed_wants;
use plan::lease_flags;
use plan::plan_prefill_spans;
use plan::plan_step_shape;
use plan::takes_argmax;
use slot::GLM52_PADDING_STEP;
use slot::Glm52SlotState;
use slot::Glm52StepOutcome;
#[cfg(test)]
pub(crate) use slot::MTP_PRODUCTION_GATE_REQUEST_ID;
#[cfg(test)]
pub(crate) use slot::MTP_SLOT_REUSE_GATE_REQUEST_ID;
#[cfg(test)]
pub(crate) use slot::mtp_production_stats;
#[cfg(test)]
pub(crate) use slot::reset_mtp_production_stats;
use tokio::sync::mpsc;
use tokio::sync::watch;

use crate::model::GLM52_MAX_BATCH_PER_RANK;
use crate::model::GLM52_MAX_STEP_ROWS;
use crate::model::GLM52_MODEL_LEN_ALIGN;
use crate::model::Glm52StepKv;
use crate::model::Glm52StepShape;
use crate::model::glm52_table_width;
use crate::runner::Glm52MtpAppend;
use crate::runner::Glm52PrefillBatch;
use crate::runner::Glm52StepFlags;
use crate::runner::Glm52Worker;

/// The KV page size (== the FlashMLA page / index-K block / model-len
/// alignment — one 64 everywhere).
const PAGE: usize = GLM52_MODEL_LEN_ALIGN;

/// Engine-level philox seed for unseeded non-greedy rows (the Kimi
/// convention: unseeded requests need no replay guarantee, so a fixed engine
/// seed suffices; per-request `seed` params replay through `mix_seed`).
const GLM52_SAMPLE_SEED: u64 = 42;

fn prefix_cache_enabled(drafter: &crate::Glm52Drafter, no_prefix_cache: bool) -> bool {
    match drafter {
        // DSpark consumes aux-hidden captures produced by exactly the target
        // forwards a prefix hit skips — missing state, not stale state (#590).
        crate::Glm52Drafter::Dspark(_) => false,
        // Native MTP's layer-78 KV rides the same pool page ids as the main
        // cache, so a GPU radix hit reuses it for free: a hit means the pages
        // were never recycled, and their L78 rows are exactly the ones written
        // when the main KV was. See `native_mtp_cache_salt` for the
        // shifted-token page-boundary caveat this accepts.
        crate::Glm52Drafter::NativeMtp | crate::Glm52Drafter::None => !no_prefix_cache,
    }
}

#[cfg(test)]
mod prefix_cache_policy_tests {
    use super::*;

    #[test]
    fn native_mtp_matches_prefixes_dspark_never_does() {
        // Native MTP page identity is prefix-stable (constant salt) and its
        // L78 KV lives at the matched pages themselves, so radix hits are
        // sound; DSpark stays excluded — its aux-hidden captures cannot be
        // recovered from a hit at all.
        assert!(prefix_cache_enabled(&crate::Glm52Drafter::NativeMtp, false));
        assert!(!prefix_cache_enabled(&crate::Glm52Drafter::NativeMtp, true));
        assert!(!prefix_cache_enabled(
            &crate::Glm52Drafter::Dspark(std::path::PathBuf::from("draft")),
            false
        ));
        assert!(prefix_cache_enabled(&crate::Glm52Drafter::None, false));
    }
}

struct ActiveRequest {
    req: GenerateRequest,
    state: Glm52SlotState,
    /// Prompt length from the client request. Native P/D appends P's anchor
    /// internally, but OpenAI usage must still report the original.
    client_prompt_tokens: usize,
    /// The request's page assignments in the rank's pool. Block RAII: blocks
    /// return to the pool (registered ones as matchable prefix-cache entries)
    /// when this drops or `release()`s.
    kv: RequestKv,
    /// Save bookkeeping for the store's seal/retire verbs, kept next to the
    /// KV it tracks.
    save_cursor: SaveCursor,
    /// Native P/D only: pending first-step D2D of the restored boundary page
    /// into the request's private page. Taken by the worker prologue before
    /// the request's first kernel runs.
    boundary_copy: Option<BoundaryCopy>,
}

/// One whole-page D2D within the KV slab, from the restored padded-name
/// page to the request's own page (every layer's slices move together by
/// construction). `prefix` pins the source until the copy retires.
struct BoundaryCopy {
    src_page: i32,
    dst_page: i32,
    _prefix: KvPrefix,
}

/// Per-rank slot occupancy: `slots[slot]`.
type RankSlots = [Option<ActiveRequest>; GLM52_MAX_BATCH_PER_RANK];

/// What one slot's span asked kvbm for this step — decides which `apply_*`
/// commits the outputs (schedule and apply must pair exactly).
#[derive(Clone, Copy, Debug)]
enum SpanKind {
    /// Prompt span that does NOT finish the prompt: KV advances, no token.
    PrefillChunk,
    /// Prompt span whose last row feeds the final prompt token: its output
    /// is the first generated token.
    PrefillBoundary,
    /// Single decode row (the zero-draft case).
    Decode,
    /// Verify span: anchor + fed drafts, committing the accepted prefix.
    Speculative,
}

/// Everything one engine needs at spawn: the per-rank pieces (queue,
/// workers, load feed) plus the shared launch configuration. Construction
/// happens inside the engine thread ([`Glm52Engine::spawn`]) so a startup
/// failure tears this rank's workers down concurrently with its siblings'.
pub(crate) struct Glm52EngineSpec {
    pub(crate) rank: usize,
    pub(crate) submit_rx: mpsc::UnboundedReceiver<SubmittedRequest>,
    /// This rank's executors: exactly one under EP, every mirrored worker
    /// under the tensor-replicated topologies.
    pub(crate) workers: Vec<Glm52Worker>,
    /// The rank's logical pool, shared with the process-wide [`KvStore`]
    /// (built before spawn — the store's rank table freezes at build).
    pub(crate) pool: Arc<BlockPool>,
    /// Whether the rank registered a host tier with the store.
    pub(crate) kv_offload: bool,
    /// The process-wide store: resolve/seal/retire and the pinned-pages
    /// admission debit all go through it.
    pub(crate) store: Arc<KvStore>,
    /// Runtime the resolver tasks spawn onto (the same handle the store
    /// drives its watchers with).
    pub(crate) runtime: tokio::runtime::Handle,
    pub(crate) eos_token_ids: Vec<u32>,
    pub(crate) drafter: crate::Glm52Drafter,
    pub(crate) prefill_chunk_size: Option<usize>,
    pub(crate) max_model_len: usize,
    pub(crate) no_prefix_cache: bool,
    pub(crate) moe_topo: crate::Glm52MoeTopo,
    pub(crate) load_tx: watch::Sender<SchedulerMetrics>,
    pub(crate) graph_dump_request: Option<GraphDumpRequest>,
    /// Bootstrap report: the engine sends once after graph pre-capture (and
    /// the rank-0 graph dump), or once with the failure that killed it.
    pub(crate) startup_tx: crossbeam_channel::Sender<anyhow::Result<()>>,
}

/// One logical DP rank's autonomous engine. Owns its workers — and its
/// offload engines: they hold the shared pegaflow host, which must outlive
/// every in-flight save and dies with the engine.
pub(crate) struct Glm52Engine {
    rank: usize,
    submit_rx: mpsc::UnboundedReceiver<SubmittedRequest>,
    workers: Vec<Glm52Worker>,
    eos_token_ids: Vec<u32>,
    drafter: crate::Glm52Drafter,
    prefill_chunk_size: Option<usize>,
    max_model_len: usize,
    prefix_cache: bool,
    /// Whether this rank's store registration carries a host tier (drives
    /// the step-flag hint the planner uses).
    kv_offload: bool,
    store: Arc<KvStore>,
    runtime: tokio::runtime::Handle,
    /// Resolver channel: intake spawns per-request resolution on the store's
    /// runtime; completed intakes come back here — `pending` holds only
    /// scheduler-ready requests.
    ready_tx: mpsc::UnboundedSender<offload::Resolved>,
    ready_rx: mpsc::UnboundedReceiver<offload::Resolved>,
    /// Resolves spawned but not yet drained from `ready_rx` — the engine may
    /// not exit (or block solely on `submit_rx`) while any are in flight.
    resolves_inflight: Arc<std::sync::atomic::AtomicUsize>,
    moe_topo: crate::Glm52MoeTopo,
    load_tx: watch::Sender<SchedulerMetrics>,
    graph_dump_request: Option<GraphDumpRequest>,
    startup_tx: crossbeam_channel::Sender<anyhow::Result<()>>,
    /// Tensor-replicated topology: this rank drives mirrored executors with
    /// identical steps (bit-identical outputs asserted at the join).
    mirrored: bool,
    /// Verify-span draft budget: EP feeds 3 (the measured bucket-4 optimum);
    /// TP4 mirrored topology feeds the drafter's full proposal.
    span_drafts: usize,
    pool: Arc<BlockPool>,
    table_width: usize,
    /// Pool pages available to requests (total minus the padding page) —
    /// constant for the engine's lifetime.
    usable_blocks: usize,
    slots: RankSlots,
    pending: VecDeque<offload::Resolved>,
    /// Slot draft states to clear on the next draft round (request left the
    /// slot, or a new one was admitted into it). Flushed with each step's
    /// Draft commands; the handler is idempotent, so duplicates are harmless.
    pending_resets: Vec<usize>,
    /// The shape this engine leased the NEXT step as: the device already
    /// holds that step's speculative replay, so the next step is pinned to
    /// this shape (see [`plan::lease_flags`]).
    leased_shape: Option<Glm52StepShape>,
    /// Slots whose requests finished while a lease was outstanding: their
    /// rows ride the leased replay (outputs discarded), so their physical
    /// release waits for the consume step — freeing the pages earlier would
    /// let admission hand them to another request while the replay still
    /// writes them.
    deferred_releases: Vec<usize>,
    /// Rank-local step counter driving the non-greedy rows' philox seeds (a
    /// fresh well-mixed seed per (step, rank); ranks never compare seeds).
    sample_step: u64,
    channel_open: bool,
}

impl Glm52Engine {
    /// Spawn the engine's thread. The KV pool is built inside the thread:
    /// its sizing arithmetic is the one fallible construction step, and a
    /// failure must still report on `startup_tx` and let the spec's drop
    /// shut this rank's workers down (each worker's own Drop sends Shutdown
    /// and joins — the destroy barrier pairs once the launcher tears the
    /// fleet down on the failed report).
    pub(crate) fn spawn(spec: Glm52EngineSpec) -> std::io::Result<std::thread::JoinHandle<()>> {
        let rank = spec.rank;
        std::thread::Builder::new()
            .name(format!("glm52-engine-{rank}"))
            .spawn(move || {
                let startup_tx = spec.startup_tx.clone();
                match Glm52Engine::new(spec) {
                    Ok(engine) => engine.run(),
                    Err(err) => {
                        let _ = startup_tx.send(Err(
                            err.context(format!("GLM5.2 rank {rank} KV pool construction"))
                        ));
                    }
                }
            })
    }

    fn new(spec: Glm52EngineSpec) -> anyhow::Result<Self> {
        let mirrored = spec.moe_topo.uses_tensor_replicated_moe();
        debug_assert_eq!(
            spec.workers.len(),
            if mirrored {
                spec.moe_topo.device_count()
            } else {
                1
            },
            "one executor per EP rank; every mirrored worker under TP"
        );
        // The pool arrives pre-built (shared with the process-wide KvStore,
        // whose rank table froze at build); block ids index the rank's
        // page-first KV slab pages directly. Under mirrored TP the single
        // pool drives every executor — the mirrored steps write the
        // identical block ids on every mirror's slab.
        let pool = spec.pool;
        let table_width = glm52_table_width(spec.max_model_len);
        // Prefix matching policy lives in `prefix_cache_enabled`: DSpark is
        // the only drafter that forces it off (aux-hidden captures cannot be
        // recovered from a hit).
        let prefix_cache = prefix_cache_enabled(&spec.drafter, spec.no_prefix_cache);
        if spec.drafter.enabled() && !prefix_cache && !spec.no_prefix_cache {
            log::info!("GLM5.2 prefix cache disabled: DSpark drafting is on");
        }
        // The prefill-only engine keeps host-tier restore off: a restore
        // recovers the 656-byte wire arenas but neither the 576-byte
        // FlashInfer proposal cache (unregistered) nor the mirrors on the
        // other three ranks (the H2D lands on one rank's arena) — GPU radix
        // hits have neither problem, so plain prefix matching stays on.
        let (ready_tx, ready_rx) = mpsc::unbounded_channel();
        Ok(Self {
            rank: spec.rank,
            submit_rx: spec.submit_rx,
            workers: spec.workers,
            eos_token_ids: spec.eos_token_ids,
            drafter: spec.drafter,
            prefill_chunk_size: spec.prefill_chunk_size,
            max_model_len: spec.max_model_len,
            prefix_cache,
            kv_offload: spec.kv_offload,
            store: spec.store,
            runtime: spec.runtime,
            ready_tx,
            ready_rx,
            resolves_inflight: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            moe_topo: spec.moe_topo,
            load_tx: spec.load_tx,
            graph_dump_request: spec.graph_dump_request,
            startup_tx: spec.startup_tx,
            mirrored,
            span_drafts: if mirrored {
                crate::dspark::GLM52_DSPARK_DRAFTS
            } else {
                slot::GLM52_DSPARK_EP8_SPAN_DRAFTS
            },
            usable_blocks: pool.total_blocks() - 1,
            table_width,
            pool,
            slots: std::array::from_fn(|_| None),
            pending: VecDeque::new(),
            pending_resets: Vec::new(),
            leased_shape: None,
            deferred_releases: Vec::new(),
            sample_step: 0,
            channel_open: true,
        })
    }

    fn run(mut self) {
        if let Err(err) = self.bootstrap() {
            let _ = self.startup_tx.send(Err(err));
            self.shutdown_workers();
            return;
        }
        let _ = self.startup_tx.send(Ok(()));
        self.serve_loop();
        self.teardown();
    }

    /// Pre-capture every bucket graph, then (rank 0 only) service the graph
    /// dump. Every engine captures the same fixed bucket sequence, so the
    /// collectives inside capture pair across the fleet without any
    /// coordinator — the pre-capture IS the bootstrap rendezvous.
    fn bootstrap(&mut self) -> anyhow::Result<()> {
        if self.prefill_chunk_size.is_none() {
            precapture_step_graphs(
                &self.workers,
                std::slice::from_ref(self.pool.as_ref()),
                self.table_width,
                self.mirrored,
            )?;
        }
        if let Some((png_path, response)) = self.graph_dump_request.take() {
            match dump_rank0_decode_graph(&self.workers, self.moe_topo, png_path) {
                Ok(summary) => {
                    let _ = response.send(Ok(summary));
                }
                Err(err) => {
                    log::error!("GLM5.2 CUDA Graph export failed: {err:#}");
                    let _ = response.send(Err(anyhow::anyhow!("{err:#}")));
                    return Err(err.context("GLM5.2 CUDA Graph export"));
                }
            }
        }
        Ok(())
    }

    fn serve_loop(&mut self) {
        'serve: loop {
            // Intake: EP engines never block — an idle EP rank runs padding
            // steps at full speed so its peers never wait on it in a
            // collective (the free-running contract). A mirrored engine is
            // the sole issuer of its collectives, so it may block while
            // fully idle instead of burning the machine on padding.
            if self.channel_open
                && self.all_idle()
                && self.pending.is_empty()
                && self.resolves_inflight() == 0
            {
                self.publish();
                if self.mirrored {
                    // Sole issuer of its collectives: may block while fully
                    // idle. With zero resolves in flight the only wake-up
                    // source is the submit channel.
                    match self.submit_rx.blocking_recv() {
                        Some((req, _kv_prefix)) => self.intake(req),
                        None => self.channel_open = false,
                    }
                }
            }
            while self.channel_open {
                match self.submit_rx.try_recv() {
                    Ok((req, _kv_prefix)) => self.intake(req),
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => self.channel_open = false,
                }
            }
            // Drain completed resolutions: the inbox holds only
            // scheduler-ready requests (no polling, no queue-front parking).
            while let Ok(resolved) = self.ready_rx.try_recv() {
                self.resolves_inflight
                    .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
                self.pending.push_back(resolved);
            }
            if !self.channel_open
                && self.all_idle()
                && self.pending.is_empty()
                && self.resolves_inflight() == 0
            {
                break;
            }

            // Admission freezes while a speculation is outstanding: the
            // lease pins the next step's shape, so newcomers wait one step.
            let t_admit = std::time::Instant::now();
            if self.leased_shape.is_none()
                && let Err(err) = self.admit()
            {
                self.fatal(&err);
            }
            let admit_ms = t_admit.elapsed().as_millis();
            self.publish();

            // A mirrored engine with nothing to run has nothing to pace: its
            // collectives are all intra-process, so skipping the step changes
            // nothing observable (and an empty prefill batch is invalid). EP
            // engines step unconditionally — their peers may be busy.
            if self.mirrored && self.all_idle() {
                continue 'serve;
            }

            if let Some(max_rows) = self.prefill_chunk_size {
                if self
                    .slots
                    .iter()
                    .flatten()
                    .any(|active| !active.state.mid_prefill())
                {
                    self.fatal(&anyhow::anyhow!(
                        "GLM5.2 prefill-only invariant failed: a request reached decode"
                    ));
                }
                self.sample_step += 1;
                if let Err(err) = self.prefill_step(max_rows) {
                    self.fatal(&err);
                }
                continue 'serve;
            }

            // One step: this rank's own bucket — each active slot's span of
            // consecutive next tokens, padding rows on the free slots. The
            // shape comes from the lease if one is outstanding (the device
            // already holds that step's speculative replay — the shape is
            // pinned, which is why admission froze), else from the rank's
            // own feed wants.
            let consume = self.leased_shape.is_some();
            let shape = match self.leased_shape.take() {
                Some(leased) => leased,
                None => plan_step_shape(&feed_wants(&self.slots)),
            };
            let flags = lease_flags(
                consume,
                self.pending.is_empty(),
                self.drafter.enabled(),
                self.kv_offload,
                !self.deferred_releases.is_empty(),
                &self.slots,
                self.max_model_len,
            );
            self.leased_shape = flags.lease.then_some(shape);
            self.sample_step += 1;
            let t_step = std::time::Instant::now();
            let (outputs, span_kinds, step_inputs) = match self.submit_and_join_step(&shape, flags)
            {
                Ok(step) => step,
                Err(err) => self.fatal(&err),
            };
            let step_ms = t_step.elapsed().as_millis();
            let t_apply = std::time::Instant::now();
            let (rank_appends, mtp_appends, mut rank_proposals) =
                match self.apply_step_outputs(&outputs, &shape, span_kinds, &step_inputs) {
                    Ok(walked) => walked,
                    Err(err) => self.fatal(&err),
                };
            // Serve-iteration stall forensics: the EP free-running contract
            // means any phase here that overruns a round period starves the
            // whole fleet's dispatch — name the phase, don't infer it from
            // peers' spin time.
            let apply_ms = t_apply.elapsed().as_millis();
            if step_ms > 300 || admit_ms > 25 || apply_ms > 25 {
                log::warn!(
                    "GLM5.2 slow serve iter: rank={} admit={admit_ms}ms step={step_ms}ms \
                     apply={apply_ms}ms active_slots={} pending={} resolves_inflight={}",
                    self.rank,
                    self.slots.iter().flatten().count(),
                    self.pending.len(),
                    self.resolves_inflight(),
                );
            }
            // Deferred releases complete ONLY at the end of the consume
            // step: the speculation they wait on was enqueued by the lease
            // step and replays during this one — freeing the pages any
            // earlier would hand them to admission while the replay still
            // writes them. (A lease step may ADD deferrals; it never
            // completes them.)
            if consume {
                self.release_deferred();
            }

            // Mirrored-TP speculative policy: draft only when the rank is
            // solo — a concurrent batch's bucket rows go to liveness first.
            // Suppress the proposals (appends and resets still flow, so the
            // drafter's shadow KV stays fresh and proposals resume the round
            // after the batch drains back to solo). Drafts already installed
            // on the solo slot are deliberately left to drain.
            if self.mirrored && self.slots.iter().flatten().count() != 1 {
                rank_proposals.clear();
            }

            let draft_result = if self.drafter.is_dspark() {
                self.run_draft_round(shape.bucket, rank_appends, rank_proposals)
            } else if self.drafter.is_mtp() {
                run_mtp_round(
                    self.rank,
                    &self.workers[0],
                    &mut self.slots,
                    shape.bucket,
                    &mut self.pending_resets,
                    mtp_appends,
                    rank_proposals,
                )
            } else {
                Ok(())
            };
            if let Err(err) = draft_result {
                self.fatal(&err);
            }
        }
    }

    fn all_idle(&self) -> bool {
        self.slots.iter().all(Option::is_none)
    }

    fn intake(&mut self, req: GenerateRequest) {
        if let Err(message) = admission::validate_request(
            &req,
            self.max_model_len,
            self.prefill_chunk_size.is_some(),
            self.prefill_chunk_size.is_some() && self.drafter.is_mtp(),
        ) {
            admission::reject(&req, message);
            return;
        }
        debug_assert!(
            req.data_parallel_rank.is_none_or(|rank| rank == self.rank),
            "GLM5.2 rank {} received a request bound for rank {:?}",
            self.rank,
            req.data_parallel_rank
        );

        // A bad handoff envelope is an intake rejection, same as a bad
        // sampling param — it must not occupy a resolver task.
        let handoff = match offload::native_mtp_handoff(&req) {
            Ok(handoff) => handoff,
            Err(err) => {
                admission::reject(&req, format!("{err:#}"));
                return;
            }
        };
        // A handoff this engine cannot restore is an intake rejection, not a
        // silent downgrade: resolving it as Plain would generate from the
        // client prompt without the transferred KV or the anchor replay — a
        // successful-but-wrong continuation that defeats the router's
        // fallback. Only an MTP decode engine restores native handoffs.
        let native = match native_handoff_disposition(
            handoff.is_some(),
            self.prefill_only(),
            self.drafter.is_mtp(),
        ) {
            NativeHandoffDisposition::Restore => handoff,
            NativeHandoffDisposition::Plain => None,
            NativeHandoffDisposition::Reject => {
                admission::reject(
                    &req,
                    "GLM5.2 native P/D handoff requires an MTP decode engine; \
                     this role cannot restore it"
                        .to_owned(),
                );
                return;
            }
        };

        // Requests with nothing to resolve (prefix cache off, and no native
        // handoff) go straight to the inbox; everything else resolves
        // off-thread and arrives via `ready_rx` — the engine loop never
        // waits on storage. The prefill-only role resolves Plain prefixes
        // like any other role: it never restores native handoffs (already
        // dispositioned above), but its multi-turn traffic re-reads the
        // prefixes it sealed to the host tier — and its peers' via the P2P
        // mesh — instead of recomputing them from row one.
        let wants_resolve = native.is_some() || self.prefix_cache;
        if !wants_resolve {
            self.pending.push_back(offload::Resolved::Plain {
                req,
                prefix: KvPrefix::none(),
            });
            return;
        }
        let store = Arc::clone(&self.store);
        let rank = self.rank;
        let ready_tx = self.ready_tx.clone();
        let inflight = Arc::clone(&self.resolves_inflight);
        inflight.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        let salt = (self.prefill_chunk_size.is_some() && self.drafter.is_mtp())
            .then(native_mtp_cache_salt);
        self.runtime.spawn(async move {
            let resolved = match native {
                Some(handoff) => match handoff.anchor_token_id {
                    // P consumed EOS: nothing to restore or decode — the
                    // anchored finish happens at admission, no resolve runs.
                    None => offload::Resolved::Native {
                        req,
                        prefix: KvPrefix::none(),
                        handoff,
                    },
                    Some(anchor) => {
                        match offload::native_pd_resolve(
                            &store,
                            rank,
                            &req,
                            &handoff,
                            anchor,
                            &req.token_tx,
                        )
                        .await
                        {
                            Ok(prefix) => offload::Resolved::Native {
                                req,
                                prefix,
                                handoff,
                            },
                            Err(err) => offload::Resolved::Failed {
                                req,
                                message: format!("native P/D resolve: {err:#}"),
                            },
                        }
                    }
                },
                None => {
                    let mut scope = pegainfer_kv_store::CacheScope::default();
                    if let Some(salt) = salt {
                        scope = scope.cache_salt(salt);
                    }
                    let req_id_owned;
                    let req_id = match req.request_id.as_deref() {
                        Some(id) => id,
                        None => {
                            req_id_owned = anon_resolve_key("glm52", rank);
                            &req_id_owned
                        }
                    };
                    let prefix = store
                        .resolve_prefix(
                            rank,
                            req_id,
                            &req.prompt_tokens,
                            scope,
                            pegainfer_kv_store::ResolvePolicy::default(),
                            &req.token_tx,
                        )
                        .await;
                    offload::Resolved::Plain { req, prefix }
                }
            };
            // The engine counts this send via `resolves_inflight`; if the
            // receiver is gone the engine already exited and the request's
            // sink closes with it.
            let _ = ready_tx.send(resolved);
        });
    }

    fn prefill_only(&self) -> bool {
        self.prefill_chunk_size.is_some()
    }

    fn admit(&mut self) -> anyhow::Result<()> {
        admit_from_queue(
            self.rank,
            &mut self.pending,
            &mut self.slots,
            &self.pool,
            self.usable_blocks,
            &self.store,
            self.prefix_cache,
            self.drafter.enabled(),
            self.prefill_chunk_size.is_some() && self.drafter.is_mtp(),
            &mut self.pending_resets,
        )
    }

    fn resolves_inflight(&self) -> usize {
        self.resolves_inflight
            .load(std::sync::atomic::Ordering::Acquire)
    }

    fn publish(&self) {
        publish_load(
            &self.load_tx,
            &self.pool,
            &self.slots,
            &self.pending,
            self.resolves_inflight(),
        );
    }

    /// One step: submit — schedule each active span's KV (full-lifetime
    /// reservation makes every schedule succeed; a failure is an accounting
    /// bug and is engine-fatal), build the row inputs, page rows and write
    /// slots, collect the step's sampling rows, and fire — then join ALL
    /// executors before failing: the executor recv'd first often reports the
    /// ~100 s DeepEP device-timeout trap, not the root cause. Returns the
    /// rank's outputs plus what the submit phase scheduled per slot
    /// (`span_kinds[slot]`), which the output walk pairs exactly.
    #[allow(clippy::type_complexity)]
    fn submit_and_join_step(
        &mut self,
        shape: &Glm52StepShape,
        flags: Glm52StepFlags,
    ) -> anyhow::Result<(
        [u32; GLM52_MAX_STEP_ROWS],
        [Option<SpanKind>; GLM52_MAX_BATCH_PER_RANK],
        [(u32, usize); GLM52_MAX_STEP_ROWS],
    )> {
        let pool = &self.pool;
        let padding_page = pool.padding_block_id();
        let sampling = collect_sampling_rows(shape, &self.slots);
        let seed = mix_seed(
            mix_seed(GLM52_SAMPLE_SEED, self.sample_step),
            self.rank as u64,
        );
        let mut span_kinds = [None; GLM52_MAX_BATCH_PER_RANK];
        let mut inputs =
            [(GLM52_PADDING_STEP.token, GLM52_PADDING_STEP.position); GLM52_MAX_STEP_ROWS];
        // A consumed speculation replays with device-advanced inputs and
        // never reads the step KV — skip building the page rows (the whole
        // point of launch-ahead is keeping this host path off the hot step
        // boundary). KV *scheduling* still runs: kvbm's bookkeeping must
        // advance every step.
        let mut pages = if flags.consume {
            Vec::new()
        } else {
            vec![padding_page; shape.bucket * self.table_width]
        };
        let mut slot_mapping = [padding_page as i64 * PAGE as i64; GLM52_MAX_STEP_ROWS];
        let mut boundary_copies = Vec::new();
        // Walk the shape's contiguous per-slot runs. Real rows end at
        // `active_rows`; the padding tail keeps the padding defaults (its
        // slot ids are insignificant, #812).
        let mut row = 0usize;
        while row < shape.active_rows {
            let slot_id = shape.slots[row] as usize;
            let mut end = row + 1;
            while end < shape.active_rows && shape.slots[end] as usize == slot_id {
                end += 1;
            }
            let span = end - row;
            if self.deferred_releases.contains(&slot_id) {
                // A dead slot's rows ride the replay with the padding
                // defaults; its KV bookkeeping stopped at the finish.
                row = end;
                continue;
            }
            let Some(active) = self.slots[slot_id].as_mut() else {
                // Padding rows keep the padding-page defaults.
                row = end;
                continue;
            };
            for (offset, r) in (row..end).enumerate() {
                let step = active.state.next_input_at(offset);
                inputs[r] = (step.token, step.position);
            }
            // The span must extend kvbm's view exactly: its first row's
            // position is the next KV slot to write. Drift between the
            // slot state's position math and the pool's bookkeeping
            // writes KV into the wrong page — fail the step instead.
            if inputs[row].1 != active.kv.kv_position() {
                return Err(anyhow::anyhow!(
                    "GLM5.2 rank {} slot {slot_id} span starts at position {} but the \
                     KV pool is at {}",
                    self.rank,
                    inputs[row].1,
                    active.kv.kv_position()
                ));
            }
            let mid_prefill = active.state.mid_prefill();
            let (kind, scheduled) = if mid_prefill {
                let kind = if active.state.remaining_prompt() == span {
                    SpanKind::PrefillBoundary
                } else {
                    SpanKind::PrefillChunk
                };
                (kind, active.kv.schedule_prefill(span, pool))
            } else if span == 1 {
                (SpanKind::Decode, active.kv.schedule_decode(pool))
            } else {
                (
                    SpanKind::Speculative,
                    active.kv.schedule_speculative(span, pool),
                )
            };
            if let Err(err) = scheduled {
                return Err(anyhow::anyhow!(
                    "GLM5.2 rank {} slot {slot_id} violated its full-lifetime KV \
                     reservation ({kind:?}, span {span}): {err}",
                    self.rank
                ));
            }
            span_kinds[slot_id] = Some(kind);
            if let Some(copy) = active.boundary_copy.as_ref() {
                // A consumed lease pre-enqueued its kernels last step — the
                // copy would order after them. Admission changes the slot
                // set, which forbids the lease, so this cannot fire.
                anyhow::ensure!(
                    !flags.consume,
                    "GLM5.2 rank {} slot {slot_id} boundary copy cannot ride a leased replay",
                    self.rank
                );
                boundary_copies.push((copy.src_page, copy.dst_page));
            }
            if !flags.consume {
                let row_pages = active.kv.step_page_indices(span);
                for r in row..end {
                    pages[r * self.table_width..r * self.table_width + row_pages.len()]
                        .copy_from_slice(&row_pages);
                    let position = inputs[r].1;
                    slot_mapping[r] =
                        row_pages[position / PAGE] as i64 * PAGE as i64 + (position % PAGE) as i64;
                }
            }
            row = end;
        }
        let kv = Glm52StepKv {
            pages: pages.into_boxed_slice(),
            slot_mapping,
            boundary_copies,
        };
        // Logical-to-executor mapping: 1:1 under EP, or this rank's step
        // mirrored onto every worker under the replicated topology
        // (identical inputs/KV/seed, bit-identical outputs asserted at the
        // join).
        let mut responses = Vec::with_capacity(self.workers.len());
        for worker in &self.workers {
            responses.push(worker.step_async(
                inputs,
                *shape,
                kv.clone(),
                flags,
                sampling.clone(),
                seed,
            )?);
        }
        let mut outputs = Vec::with_capacity(responses.len());
        let mut step_err: Option<anyhow::Error> = None;
        for (executor, resp) in responses.into_iter().enumerate() {
            let result = resp.recv().map_err(|_| {
                anyhow::anyhow!(
                    "GLM5.2 rank {} executor {executor} dropped its step response",
                    self.rank
                )
            });
            match result {
                Ok(Ok(step_tokens)) => outputs.push(step_tokens),
                Ok(Err(err)) | Err(err) => {
                    let err = err.context(format!(
                        "GLM5.2 rank {} executor {executor} step",
                        self.rank
                    ));
                    log::error!(
                        "GLM5.2 rank {} executor {executor} step failed: {err:#}",
                        self.rank
                    );
                    step_err.get_or_insert(err);
                    outputs.push([0; GLM52_MAX_STEP_ROWS]);
                }
            }
        }
        if let Some(err) = step_err {
            return Err(err);
        }
        if self.mirrored {
            // The replicated contract: every executor computed the identical
            // step, so any divergence means the redundant compute desynced —
            // serving on it would emit rank-dependent garbage. Crash early.
            for (executor, out) in outputs.iter().enumerate().skip(1) {
                anyhow::ensure!(
                    out == &outputs[0],
                    "GLM5.2 mirrored executor {executor} step outputs diverge from executor 0 \
                     (the replicated bit-identity contract broke)"
                );
            }
            outputs.truncate(1);
        }
        // The step replied after its output D2H, so the boundary copies —
        // enqueued ahead of the kernels on the same stream — have landed;
        // the holds pinning their source pages can release.
        for active in self.slots.iter_mut().flatten() {
            active.boundary_copy = None;
        }
        Ok((outputs[0], span_kinds, inputs))
    }

    /// Fold this rank's span of outputs into its slot states, commit the
    /// span's KV bookkeeping under the exact kind the submit phase scheduled
    /// (a mispairing is an engine bug and is fatal), emit tokens and
    /// finish/disconnect releases, and collect the draft lane's context
    /// appends and next-round proposals.
    #[allow(clippy::type_complexity)]
    fn apply_step_outputs(
        &mut self,
        outputs: &[u32; GLM52_MAX_STEP_ROWS],
        shape: &Glm52StepShape,
        span_kinds: [Option<SpanKind>; GLM52_MAX_BATCH_PER_RANK],
        step_inputs: &[(u32, usize); GLM52_MAX_STEP_ROWS],
    ) -> anyhow::Result<(
        Vec<(usize, usize)>,
        Vec<Glm52MtpAppend>,
        Vec<(usize, u32, usize)>,
    )> {
        let mut rank_appends = Vec::new();
        let mut mtp_appends = Vec::new();
        let mut rank_proposals = Vec::new();
        // Walk the shape's contiguous per-slot runs; each active slot folds
        // its whole span of row outputs in at once. Padding rows (past
        // `active_rows`) carry no outputs anyone reads (#812).
        let mut row = 0usize;
        while row < shape.active_rows {
            let slot_id = shape.slots[row] as usize;
            let mut end = row + 1;
            while end < shape.active_rows && shape.slots[end] as usize == slot_id {
                end += 1;
            }
            let span_rows = row..end;
            let span_outputs = &outputs[span_rows.clone()];
            row = end;
            if self.deferred_releases.contains(&slot_id) {
                // The replay row's output is discarded; the release was
                // handled at the finish and completes in `release_deferred`.
                continue;
            }
            let slot = &mut self.slots[slot_id];
            let Some(active) = slot.as_mut() else {
                continue;
            };
            let prompt_tokens = active.client_prompt_tokens;
            let outcome = active.state.advance_span(span_outputs, &self.eos_token_ids);
            // Commit the span's KV bookkeeping under the exact kind the
            // submit phase scheduled — a mispairing is an engine bug and
            // is fatal.
            let pool = &self.pool;
            let applied = match (&outcome, span_kinds[slot_id]) {
                (Glm52StepOutcome::Prefilling, Some(SpanKind::PrefillChunk)) => {
                    active.kv.apply_prefill_chunk(pool)
                }
                (Glm52StepOutcome::Commit { committed, .. }, Some(SpanKind::PrefillBoundary)) => {
                    active.kv.apply_prefill(committed[0], pool)
                }
                (Glm52StepOutcome::Commit { committed, .. }, Some(SpanKind::Decode)) => {
                    active.kv.apply_decode(committed[0], pool).map(|_| ())
                }
                (Glm52StepOutcome::Commit { committed, .. }, Some(SpanKind::Speculative)) => {
                    active.kv.apply_speculative(committed, pool).map(|_| ())
                }
                (outcome, kind) => Err(anyhow::anyhow!(
                    "GLM5.2 rank {} slot {slot_id} outcome {outcome:?} does not pair \
                     with scheduled span kind {kind:?}",
                    self.rank
                )),
            };
            if let Err(err) = applied {
                return Err(
                    err.context(format!("GLM5.2 rank {} slot {slot_id} KV apply", self.rank))
                );
            }
            let (freed, context_rows) = match outcome {
                Glm52StepOutcome::Prefilling => {
                    // Prefill never sends, so a disconnect is only
                    // visible through the sink probe — without it a
                    // long prompt zombies the slot until prefill
                    // completes. Every prompt row is committed
                    // context.
                    (active.req.token_tx.is_closed(), span_outputs.len())
                }
                Glm52StepOutcome::Commit {
                    committed,
                    emit,
                    finish,
                    context_rows,
                } => {
                    // A dropped receiver (client disconnect) frees the
                    // slot; its pool pages release with the request
                    // (sealed blocks stay matchable as prefix cache).
                    let mut freed = false;
                    for &token in &committed[..emit] {
                        if active
                            .req
                            .token_tx
                            .send(TokenEvent::Token {
                                id: token,
                                logprob: None,
                            })
                            .is_err()
                        {
                            freed = true;
                            break;
                        }
                    }
                    if let Some(finish_reason) = finish
                        && !freed
                    {
                        let _ = active.req.token_tx.send(TokenEvent::Finished {
                            finish_reason,
                            prompt_tokens,
                            completion_tokens: active.state.completion_tokens(),
                        });
                        freed = true;
                    }
                    (freed, context_rows)
                }
            };
            if freed {
                #[cfg(test)]
                active
                    .state
                    .record_mtp_production_gate(active.req.request_id.as_deref());
                active.state.log_spec_stats(self.rank, slot_id);
                // Seal the freshly-registered blocks BEFORE the release:
                // the hashes and guards come off the still-assigned request
                // state, and the guards keep the pages pinned through the
                // async D2H copy.
                self.store.seal(
                    self.rank,
                    &active.kv,
                    &mut active.save_cursor,
                    SaveClass::Cacheable,
                );
                if self.leased_shape.is_some() {
                    // A speculation for the next step is already on the
                    // device: the slot's row rides the replay (its output
                    // is discarded), so the physical release waits for the
                    // consume step — freeing the pages now would let
                    // admission hand them to another request while the
                    // replay still writes them.
                    self.deferred_releases.push(slot_id);
                } else {
                    let finished = slot.take().expect("freed slot was active");
                    self.store.retire(
                        self.rank,
                        finished.kv,
                        finished.save_cursor,
                        SaveClass::Cacheable,
                    );
                    if self.drafter.enabled() {
                        self.pending_resets.push(slot_id);
                    }
                    *slot = None;
                }
            } else if self.drafter.enabled() {
                if self.drafter.is_dspark() {
                    rank_appends.extend(span_rows.clone().take(context_rows).map(|r| (r, slot_id)));
                } else {
                    for (offset, target_row) in span_rows.clone().take(context_rows).enumerate() {
                        let input_token = if offset + 1 < context_rows {
                            step_inputs[target_row + 1].0
                        } else {
                            active.state.next_input_at(0).token
                        };
                        mtp_appends.push(Glm52MtpAppend {
                            target_row,
                            slot: slot_id,
                            input_token,
                            position: step_inputs[target_row].1,
                            pages: active.kv.current_page_indices(),
                        });
                    }
                }
                let wants_drafts = if self.drafter.is_mtp() {
                    active
                        .state
                        .wants_full_draft(crate::mtp::glm52_mtp_draft_len())
                } else {
                    active.state.wants_drafts()
                };
                if wants_drafts && let Some((anchor, anchor_pos)) = active.state.decode_anchor() {
                    rank_proposals.push((slot_id, anchor, anchor_pos));
                }
            }
        }
        Ok((rank_appends, mtp_appends, rank_proposals))
    }

    /// Physically release the slots whose finishes were deferred by the
    /// lease this step consumed. Their replay rows were skipped by both the
    /// submit and the apply walk; their pages stayed mapped through the
    /// replay, and the client events and offload saves already happened at
    /// the finish.
    fn release_deferred(&mut self) {
        for slot_id in self.deferred_releases.drain(..) {
            let Some(active) = self.slots[slot_id].take() else {
                continue;
            };
            self.store.retire(
                self.rank,
                active.kv,
                active.save_cursor,
                SaveClass::Cacheable,
            );
        }
    }

    /// DSpark draft round (rank-local, no collectives): resets, context
    /// appends from THIS step's capture buffer, and new proposals for the
    /// next verify span. FIFO per-worker channels order it before the next
    /// step; the blocking join keeps the round cadence (draft sits between
    /// verify steps, ~2 ms against a 22-46 ms step).
    fn run_draft_round(
        &mut self,
        bucket: usize,
        appends: Vec<(usize, usize)>,
        proposals: Vec<(usize, u32, usize)>,
    ) -> anyhow::Result<()> {
        let resets = std::mem::take(&mut self.pending_resets);
        if resets.is_empty() && appends.is_empty() && proposals.is_empty() {
            return Ok(());
        }
        let proposal_slots: Vec<usize> = proposals.iter().map(|&(slot, _, _)| slot).collect();
        // Same logical-to-executor mapping as the step submit: under the
        // mirrored topology every worker drafts from its own (identical)
        // capture buffer and must propose the identical spans.
        let (last_worker, fanned) = self.workers.split_last().expect("one executor per rank");
        let mut rxs = Vec::with_capacity(self.workers.len());
        for worker in fanned {
            rxs.push(worker.draft_async(
                bucket,
                resets.clone(),
                appends.clone(),
                proposals.clone(),
            )?);
        }
        // The payloads are fanned out to every executor but the last; the
        // last takes ownership instead of another clone.
        rxs.push(last_worker.draft_async(bucket, resets, appends, proposals)?);
        let mut all_spans = Vec::with_capacity(rxs.len());
        for (executor, rx) in rxs.into_iter().enumerate() {
            let result = rx
                .recv()
                .map_err(|_| {
                    anyhow::anyhow!(
                        "GLM5.2 rank {} executor {executor} dropped its draft response",
                        self.rank
                    )
                })
                .and_then(|r| r);
            match result {
                Ok(spans) => all_spans.push(spans),
                // A draft failure is rank-local, but it means the drafter's
                // invariants broke — crash early rather than silently degrade
                // to plain decode.
                Err(err) => {
                    return Err(err.context(format!(
                        "GLM5.2 rank {} executor {executor} draft",
                        self.rank
                    )));
                }
            }
        }
        for (executor, spans) in all_spans.iter().enumerate().skip(1) {
            anyhow::ensure!(
                spans == &all_spans[0],
                "GLM5.2 mirrored executor {executor} draft spans diverge from executor 0 \
                 (the replicated bit-identity contract broke)"
            );
        }
        let spans = all_spans.swap_remove(0);
        anyhow::ensure!(
            spans.len() == proposal_slots.len(),
            "GLM5.2 rank {} draft returned {} spans for {} proposals",
            self.rank,
            spans.len(),
            proposal_slots.len()
        );
        for (slot_id, span) in proposal_slots.into_iter().zip(spans) {
            if let Some(active) = self.slots[slot_id].as_mut() {
                active.state.set_drafts(span.to_vec(), self.span_drafts);
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn prefill_step(&mut self, max_rows: usize) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.workers.len() > 1,
            "GLM5.2 native prefill requires one logical rank mirrored across local TP workers"
        );
        let wants = feed_wants(&self.slots);
        let spans = plan_prefill_spans(&wants, max_rows);
        let pool = &self.pool;
        let mut batch = Glm52PrefillBatch {
            token_ids: Vec::new(),
            positions: Vec::new(),
            request_indptr: vec![0],
            block_indptr: vec![0],
            block_ids: Vec::new(),
            request_slots: Vec::new(),
            padding_block: pool.padding_block_id(),
            slot_mapping: Vec::new(),
            mtp_next_tokens: Vec::new(),
            output_rows: Vec::new(),
            sampling: Vec::new(),
            seed: mix_seed(GLM52_SAMPLE_SEED, self.sample_step),
        };
        let mut scheduled = Vec::new();
        for (slot_id, &span) in spans.iter().enumerate() {
            if span == 0 {
                continue;
            }
            let active = self.slots[slot_id]
                .as_mut()
                .expect("prefill planner assigns only active slots");
            anyhow::ensure!(
                active.state.mid_prefill() && span <= active.state.remaining_prompt(),
                "GLM5.2 prefill planner produced an invalid span"
            );
            anyhow::ensure!(
                active.state.next_input_at(0).position == active.kv.kv_position(),
                "GLM5.2 prefill slot {slot_id} position drift"
            );
            active
                .kv
                .schedule_prefill(span, pool)
                .map_err(|err| anyhow::anyhow!("GLM5.2 prefill slot {slot_id} schedule: {err}"))?;
            let pages = active.kv.step_page_indices(span);
            for offset in 0..span {
                let input = active.state.next_input_at(offset);
                batch.token_ids.push(input.token);
                batch.positions.push(input.position as u32);
                let page = pages[input.position / PAGE];
                batch
                    .slot_mapping
                    .push(page as i64 * PAGE as i64 + (input.position % PAGE) as i64);
            }
            batch.block_ids.extend_from_slice(&pages);
            batch.request_slots.push(slot_id);
            batch.request_indptr.push(batch.token_ids.len() as u32);
            batch.block_indptr.push(batch.block_ids.len() as u32);
            let boundary = span == active.state.remaining_prompt();
            batch
                .mtp_next_tokens
                .push((!boundary).then(|| active.state.next_input_at(span).token));
            if boundary {
                batch.output_rows.push((batch.token_ids.len() - 1) as u32);
                if !takes_argmax(&active.req.params) {
                    batch.sampling.push(crate::runner::Glm52RowSample {
                        row: batch.output_rows.len() - 1,
                        params: active.req.params,
                        step: active.state.completion_tokens() as u64,
                    });
                }
            }
            scheduled.push((slot_id, span, boundary));
        }
        batch.validate()?;

        let responses = self
            .workers
            .iter()
            .map(|worker| worker.prefill_chunk_async(batch.clone()))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut outputs = Vec::with_capacity(responses.len());
        for (executor, response) in responses.into_iter().enumerate() {
            outputs.push(
                response
                    .recv()
                    .map_err(|_| {
                        anyhow::anyhow!(
                            "GLM5.2 rank {} executor {executor} dropped prefill response",
                            self.rank
                        )
                    })?
                    .with_context(|| {
                        format!("GLM5.2 rank {} executor {executor} prefill", self.rank)
                    })?,
            );
        }
        for (executor, output) in outputs.iter().enumerate().skip(1) {
            anyhow::ensure!(
                output == &outputs[0],
                "GLM5.2 TP prefill output diverged on executor {executor}: \
                 executor0={:?}, executor{executor}={output:?}",
                outputs[0],
            );
        }
        anyhow::ensure!(
            outputs[0].target_tokens.len() == batch.output_rows.len(),
            "GLM5.2 prefill returned {} boundary outputs, expected {}",
            outputs[0].target_tokens.len(),
            batch.output_rows.len()
        );
        anyhow::ensure!(
            outputs[0].mtp_draft1.is_empty()
                || outputs[0].mtp_draft1.len() == batch.output_rows.len(),
            "GLM5.2 prefill returned {} MTP draft-1 tokens, expected zero or {}",
            outputs[0].mtp_draft1.len(),
            batch.output_rows.len()
        );
        anyhow::ensure!(
            outputs[0].mtp_drafts.is_empty()
                || outputs[0].mtp_drafts.len() == batch.output_rows.len(),
            "GLM5.2 prefill returned {} complete MTP proposals, expected zero or {}",
            outputs[0].mtp_drafts.len(),
            batch.output_rows.len()
        );

        let mut boundary_output = outputs[0].target_tokens.iter();
        let mut boundary_drafts = outputs[0].mtp_drafts.iter();
        for (slot_id, span, boundary) in scheduled {
            // Proposals are positional batch outputs: consume one for every
            // boundary even if that request's client disconnected. Whether the
            // proposal is published must not shift later requests' mapping.
            let drafts = take_boundary_drafts(boundary, &mut boundary_drafts);
            let slot = &mut self.slots[slot_id];
            let active = slot
                .as_mut()
                .expect("scheduled prefill slot remains active");
            let mut span_outputs = vec![0; span];
            if boundary {
                span_outputs[span - 1] = *boundary_output.next().expect("validated output count");
            }
            let prompt_tokens = active.client_prompt_tokens;
            let outcome = active
                .state
                .advance_span(&span_outputs, &self.eos_token_ids);
            let freed = match outcome {
                Glm52StepOutcome::Prefilling => {
                    active.kv.apply_prefill_chunk(pool)?;
                    active.req.token_tx.is_closed()
                }
                Glm52StepOutcome::Commit {
                    committed,
                    emit,
                    finish,
                    ..
                } => {
                    active.kv.apply_prefill(committed[0], pool)?;
                    let mut freed = false;
                    for &token in &committed[..emit] {
                        if active
                            .req
                            .token_tx
                            .send(TokenEvent::Token {
                                id: token,
                                logprob: None,
                            })
                            .is_err()
                        {
                            freed = true;
                            break;
                        }
                    }
                    if !freed && let Some(drafts) = drafts {
                        let committed_len = active.kv.kv_position();
                        // Pad the partial page to the boundary and register
                        // it under the padded name (no-op on an aligned
                        // commit): sealed and guarded like any full page —
                        // the keyed tail path and its parking are gone.
                        if let Err(err) = active.kv.pad_to_boundary(&self.pool) {
                            let message =
                                format!("GLM5.2 native P/D pad-to-boundary failed: {err:#}");
                            log::warn!("{message}");
                            let _ = active.req.token_tx.send(TokenEvent::Error {
                                message,
                                prompt_tokens,
                                completion_tokens: active.state.completion_tokens(),
                            });
                            freed = true;
                        }
                        if !freed {
                            let handoff = offload::NativeMtpHandoff {
                                fingerprint: offload::handoff_fingerprint(),
                                committed_len,
                                anchor_token_id: (emit == 1).then(|| committed[0]),
                                draft_tokens: drafts.to_vec(),
                            };
                            let _ = active.req.token_tx.send(TokenEvent::KvTransfer {
                                params: serde_json::to_value(offload::PegaInferPdEnvelope {
                                    pegainfer_pd: handoff,
                                })
                                .expect("handoff envelope serializes"),
                            });
                        }
                    }
                    if let Some(finish_reason) = finish
                        && !freed
                    {
                        let _ = active.req.token_tx.send(TokenEvent::Finished {
                            finish_reason,
                            prompt_tokens,
                            completion_tokens: active.state.completion_tokens(),
                        });
                        freed = true;
                    }
                    freed
                }
            };
            if freed {
                // Every sealed page — the padded boundary page included — is
                // guarded: KvBlockGuards pin the pages through the D2H
                // independently of the release, so nothing parks. A lost
                // save surfaces as the consuming D's hit shortfall, which
                // rejects pre-slot and the router retries against P.
                let mut finished = slot.take().expect("freed slot was active");
                self.store.seal(
                    self.rank,
                    &finished.kv,
                    &mut finished.save_cursor,
                    SaveClass::Cacheable,
                );
                self.store.retire(
                    self.rank,
                    finished.kv,
                    finished.save_cursor,
                    SaveClass::Cacheable,
                );
            }
        }
        Ok(())
    }

    /// A failed step leaves the ranks permanently out of lockstep: whichever
    /// collective the survivors are spinning in would pair with the NEXT
    /// step's first dispatch and every layer after it would run against the
    /// wrong expert bank — byte-deterministic garbage, no crash. The fleet
    /// cannot be re-synced; fail this rank's requests and exit the process.
    /// The peers fail-stop on their own collective errors/timeouts, and the
    /// router pulls the traffic (`docs/models/glm52/free-running-dp.md` §6).
    fn fatal(&mut self, err: &anyhow::Error) -> ! {
        log::error!(
            "GLM5.2 rank {} fatal; the engine process exits \
             (the EP collective group cannot recover): {err:#}",
            self.rank
        );
        for slot in &mut self.slots {
            let Some(active) = slot.take() else {
                continue;
            };
            let _ = active.req.token_tx.send(TokenEvent::Error {
                message: format!("{err:#}"),
                prompt_tokens: active.client_prompt_tokens,
                completion_tokens: active.state.completion_tokens(),
            });
        }
        for resolved in self.pending.drain(..) {
            let req = resolved.into_request();
            let _ = req.token_tx.send(TokenEvent::Error {
                message: format!("{err:#}"),
                prompt_tokens: req.prompt_tokens.len(),
                completion_tokens: 0,
            });
        }
        std::process::exit(1);
    }

    /// Graceful teardown (the submit channel closed and every request
    /// drained): fail whatever never got a slot, flush and drop the offload
    /// engines BEFORE the workers drop the models, then shut the workers
    /// down. Every engine reaches here together (its channel closed with
    /// the others), so the collective DeepEP destroy barrier pairs across
    /// the fleet.
    fn teardown(mut self) {
        for resolved in self.pending.drain(..) {
            let req = resolved.into_request();
            let _ = req.token_tx.send(TokenEvent::Error {
                message: "GLM5.2 engine shut down before the request was scheduled".to_owned(),
                prompt_tokens: req.prompt_tokens.len(),
                completion_tokens: 0,
            });
        }
        // Drain in-flight saves BEFORE the workers drop the models: the
        // registered arenas' device memory must outlive every D2H copy, and
        // pegaflow's save worker cannot cancel a copy already handed to it.
        // Bounded: a stuck host tier cannot hang teardown. In-flight
        // resolves settle inside the store (their holds and reservations
        // ride detached tasks); the store outlives this engine via its Arc.
        // Loads first: a resolve abandoned at its deadline leaves a detached
        // H2D still writing arena memory; the workers must not free the
        // arenas under it. Then saves (D2H reads the same arenas). Both
        // barriers deadline-bounded so a hung tier cannot hang teardown.
        let rank = self.rank;
        let drained = self.runtime.block_on(async {
            let loads =
                tokio::time::timeout(Duration::from_secs(5), self.store.flush_loads(rank)).await;
            let saves =
                tokio::time::timeout(Duration::from_secs(5), self.store.flush_saves(rank)).await;
            (loads.is_ok(), saves.is_ok())
        });
        if !drained.0 {
            log::warn!("GLM5.2 rank {rank} teardown: restore-load drain exceeded its deadline");
        }
        if !drained.1 {
            log::warn!("GLM5.2 rank {rank} teardown: save flush exceeded its deadline");
        }
        match teardown_disposition(drained.0, drained.1) {
            TeardownDisposition::FreeWorkers => self.shutdown_workers(),
            TeardownDisposition::LeakWorkers => {
                log::error!(
                    "GLM5.2 rank {rank} teardown: a detached DMA may still target this \
                     rank's registered GPU arenas; leaking the workers (and the model \
                     arenas they own) for the remainder of the process lifetime instead \
                     of freeing them under the copy"
                );
                std::mem::forget(std::mem::take(&mut self.workers));
            }
        }
    }

    /// The DeepEP context drop is collective: broadcast Shutdown to this
    /// rank's workers BEFORE their Drop joins them — a sequential
    /// shutdown-then-join would leave a worker spinning in the destroy
    /// barrier for ranks that never got the command (until the ~100 s
    /// device timeout). Dropping a worker frees its device memory: the
    /// worker thread owns the rank's model, whose arenas are the registered
    /// DMA targets for every offload load/save — call this only once both
    /// teardown drains settled.
    fn shutdown_workers(self) {
        for worker in &self.workers {
            let _ = worker.request_shutdown();
        }
        drop(self.workers);
    }
}

/// How intake treats a request's native P/D handoff on this engine role.
///
/// INVARIANT: a handoff never degrades silently. An engine that cannot
/// restore it (prefill-only, or a decode role without the MTP drafter) must
/// reject at intake so the router's fallback fires — queuing the request as
/// Plain would generate from the client prompt without the transferred KV or
/// the anchor replay, returning a successful but incorrect continuation.
#[derive(Debug, Eq, PartialEq)]
enum NativeHandoffDisposition {
    Restore,
    Plain,
    Reject,
}

fn native_handoff_disposition(
    has_handoff: bool,
    prefill_only: bool,
    drafter_is_mtp: bool,
) -> NativeHandoffDisposition {
    match (has_handoff, prefill_only || !drafter_is_mtp) {
        (false, _) => NativeHandoffDisposition::Plain,
        (true, false) => NativeHandoffDisposition::Restore,
        (true, true) => NativeHandoffDisposition::Reject,
    }
}

#[cfg(test)]
mod native_handoff_disposition_tests {
    use super::NativeHandoffDisposition;
    use super::native_handoff_disposition;

    #[test]
    fn an_unrestorable_handoff_rejects_instead_of_degrading_to_plain() {
        // prefill-only and non-MTP decode both lack the restore path.
        assert_eq!(
            native_handoff_disposition(true, true, true),
            NativeHandoffDisposition::Reject
        );
        assert_eq!(
            native_handoff_disposition(true, false, false),
            NativeHandoffDisposition::Reject
        );
        assert_eq!(
            native_handoff_disposition(true, false, true),
            NativeHandoffDisposition::Restore
        );
        // No handoff: role capability is irrelevant, the request is plain.
        assert_eq!(
            native_handoff_disposition(false, true, false),
            NativeHandoffDisposition::Plain
        );
    }
}

/// What teardown may do with this rank's workers once the bounded DMA
/// drains have reported.
///
/// INVARIANT: an arena a detached DMA may still touch is never returned to
/// the allocator. The worker threads own the models and their registered
/// GPU arenas, so dropping a worker frees device memory an unsettled
/// restore H2D still writes (or a save D2H still reads) — a use-after-free
/// at teardown. When either drain timed out, the only safe terminal state
/// short of aborting the process is leaking the workers for the remainder
/// of the process lifetime: their threads stay parked on their command
/// channels and the DeepEP destroy barrier is forfeited (peers fail-stop on
/// their own device timeouts, as they do for any lost rank).
#[derive(Debug, Eq, PartialEq)]
enum TeardownDisposition {
    FreeWorkers,
    LeakWorkers,
}

fn teardown_disposition(loads_drained: bool, saves_drained: bool) -> TeardownDisposition {
    if loads_drained && saves_drained {
        TeardownDisposition::FreeWorkers
    } else {
        TeardownDisposition::LeakWorkers
    }
}

#[cfg(test)]
mod teardown_disposition_tests {
    use super::TeardownDisposition;
    use super::teardown_disposition;

    /// Driving the real drain-timeout path needs a GPU engine (the workers
    /// wrap CUDA executors), so the decision is factored pure and pinned
    /// here: any unsettled DMA drain must forfeit the worker teardown —
    /// freeing the arenas under a live copy is never an option.
    #[test]
    fn any_unsettled_dma_drain_leaks_the_workers() {
        assert_eq!(
            teardown_disposition(true, true),
            TeardownDisposition::FreeWorkers
        );
        assert_eq!(
            teardown_disposition(false, true),
            TeardownDisposition::LeakWorkers
        );
        assert_eq!(
            teardown_disposition(true, false),
            TeardownDisposition::LeakWorkers
        );
        assert_eq!(
            teardown_disposition(false, false),
            TeardownDisposition::LeakWorkers
        );
    }
}

fn take_boundary_drafts<'a>(
    boundary: bool,
    drafts: &mut std::slice::Iter<'a, [u32; crate::mtp::GLM52_MTP_DRAFTS]>,
) -> Option<&'a [u32; crate::mtp::GLM52_MTP_DRAFTS]> {
    if boundary { drafts.next() } else { None }
}

/// Fallback prefetch key for a resolve whose request carries no id —
/// `prefix` names the resolve path (plain restore, native P/D). The host
/// tier scopes in-flight prefetch state by this key, so it must be unique per
/// resolve — resolvers run concurrently, and two sharing one key would poll
/// each other's fetch state. One process-wide sequence keeps every anonymous
/// resolve distinct across paths, ranks, and lifetimes.
fn anon_resolve_key(prefix: &str, rank: usize) -> String {
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    format!("{prefix}-r{rank}-{seq}")
}

#[cfg(test)]
mod anon_resolve_key_tests {
    use super::anon_resolve_key;

    #[test]
    fn consecutive_fallback_keys_never_collide() {
        let first = anon_resolve_key("glm52", 2);
        let second = anon_resolve_key("glm52", 2);
        assert!(
            first.starts_with("glm52-r2-"),
            "the key embeds the path and rank: {first}"
        );
        assert_ne!(
            first, second,
            "the tier keys in-flight prefetch state by this string; \
             two resolvers must never share one"
        );
        // The native P/D path draws from the same sequence: concurrent
        // anonymous handoffs must not poll each other's prefetch state
        // either.
        let native_first = anon_resolve_key("native-pd", 2);
        let native_second = anon_resolve_key("native-pd", 2);
        assert!(
            native_first.starts_with("native-pd-r2-"),
            "the key embeds the path and rank: {native_first}"
        );
        assert_ne!(native_first, native_second);
    }
}

/// Partition lineage-hashed native-MTP pages from plain (drafterless) pages:
/// layer-78 rows ride the same pool pages, so an MTP request must never match
/// a block whose L78 arenas were never written — and vice versa.
///
/// The salt is CONSTANT, which makes page identity prefix-stable across
/// requests and turns (v1 hashed the full prompt into the salt, which made
/// every continuation its own cache universe and killed all cross-turn
/// reuse). The accepted trade: layer 78 consumes shifted tokens, so the last
/// MTP row of page k depends on the first token of page k+1. Two prompts
/// that share pages and diverge EXACTLY at a page boundary therefore share
/// one L78 row whose shifted input came from the other continuation. That
/// row only feeds draft proposals — target verification rejects any draft it
/// misleads, so output text is never affected. Same-conversation multi-turn
/// extensions agree on every shared token and never alias at all. The
/// anchor-dependent final MTP row rides the padded boundary page, whose name
/// binds the anchor when the commit is unaligned; the aligned case accepts a
/// draft-quality-only collision (resolver-ownership.md §2.2).
fn native_mtp_cache_salt() -> &'static str {
    "pegainfer-glm52-native-mtp-pages-v2"
}

#[cfg(test)]
mod tp_prefill_output_tests {
    use super::native_mtp_cache_salt;
    use super::take_boundary_drafts;

    #[test]
    fn disconnected_boundary_does_not_shift_the_next_requests_drafts() {
        let proposals = [
            [11; crate::mtp::GLM52_MTP_DRAFTS],
            [22; crate::mtp::GLM52_MTP_DRAFTS],
        ];
        let mut drafts = proposals.iter();

        let _discarded_after_disconnect = take_boundary_drafts(true, &mut drafts);
        assert_eq!(take_boundary_drafts(false, &mut drafts), None);
        assert_eq!(take_boundary_drafts(true, &mut drafts), Some(&proposals[1]));
    }

    #[test]
    fn native_mtp_page_identity_is_prefix_stable() {
        // The salt is constant: a turn that extends its predecessor derives
        // identical block hashes for the shared prefix, so radix / host-tier
        // reuse works across turns and requests. The v1 full-prompt salt made
        // every continuation its own cache universe (zero reuse); the
        // page-boundary shifted-token alias this accepts is draft-quality
        // only — see `native_mtp_cache_salt`.
        assert_eq!(native_mtp_cache_salt(), native_mtp_cache_salt());
        assert!(
            !native_mtp_cache_salt().is_empty(),
            "the constant salt must still partition MTP pages from plain \
             (drafterless) pages, whose L78 arenas were never written"
        );
    }
}
