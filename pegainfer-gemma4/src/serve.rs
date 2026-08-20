//! KV-backed serving forward: two KV families, prefill and decode, and
//! atomic dual-pool admission.
//!
//! The layer forwards take the step's plans and prep metadata and the
//! pools this module owns, never a request's state. Both coordinate systems
//! (absolute positions for RoPE, cache-relative slots for the paged
//! scatter) coincide below the
//! sliding window; past it the local family releases its front, and
//! `origin_pages` is what converts between the two.
//!
//! Batched decode reads the local family through the windowed prefill
//! entry at seq_len 1 and the global family through its native split-KV
//! decode entry. Attention reads are read-only (the prep kernels own the
//! pool writes) with sm_scale 1.0 — Gemma 4 runs unscaled attention.

use anyhow::Context as AnyhowContext;
use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::cuda_graph::CudaGraphState;
use pegainfer_core::kv_pool::KvPool;
use pegainfer_core::ops;
use pegainfer_core::ops::PrefillPagedPlan;
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;

use crate::config::Gemma4Config;
use crate::config::LayerKind;
use crate::forward::embed_scale_bf16;
use crate::forward::logits_tail;
use crate::forward::logits_tail_into;
use crate::forward::validate_tokens;
use crate::kv::GemmaKv;
use crate::kv::PAGE_SIZE;
use crate::kv::SlidingLocalKv;
use crate::kv::admit_tokens;
use crate::layer::EpilogueScratch;
use crate::layer::LayerGeometry;
use crate::layer::attention_epilogue_into;
use crate::layer::build_proportional_rope_tables;
use crate::weights::Gemma4Layer;
use crate::weights::Gemma4Weights;

/// How a step feeds the preps. The tower above them is the same either way;
/// only the metadata a prep reads changes.
#[derive(Clone, Copy)]
enum PrepRef<'a> {
    Single {
        start_pos: usize,
        /// Absolute page the local resident row starts at. The preps shift
        /// only the row index by it; RoPE stays on absolute positions.
        local_page_origin: usize,
        global_plan: &'a PrefillPagedPlan,
    },
    /// One row per request. Each row's local page window rides the local
    /// plan; the global family's window and positions live in the step's
    /// uploaded tables. What neither carries is the sliding family's
    /// released front, one page index per row.
    Batched {
        local_origins: &'a CudaSlice<i32>,
        global_tables: &'a GlobalTables,
    },
    /// A mixed step's rows: the admitted prompts occupy the row prefix as
    /// segments and the live decode batch the suffix. One per-token prep
    /// launch per family covers every row through the step's [`MixRows`]
    /// tables; for attention, the local plan the caller passes is the
    /// ragged one covering every row, and the global family reads its
    /// prefix through `global_prefill_plan` and its suffix through the
    /// step tables.
    Mixed {
        prefill_len: usize,
        mix_positions: &'a CudaSlice<i32>,
        mix_local_pages: &'a CudaSlice<i32>,
        mix_local_origins: &'a CudaSlice<i32>,
        mix_global_pages: &'a CudaSlice<i32>,
        mix_global_origins: &'a CudaSlice<i32>,
        mix_indptr: &'a CudaSlice<i32>,
        global_tables: &'a GlobalTables,
        global_prefill_plan: &'a PrefillPagedPlan,
    },
}

/// One prompt segment's slice of a mixed step's stacked rows. Positions,
/// pages and origins ride the per-row [`MixRows`] arrays; the segment
/// keeps only what the head gather needs — where its rows sit and how
/// many there are.
struct MixSeg {
    row_offset: usize,
    rows: usize,
}

/// Host metadata shared by pure-decode and mixed-step decode rows.
struct DecodeRow {
    position: usize,
    local_last: usize,
    local_start: usize,
    local_origin: usize,
    global_last: usize,
}

/// Per-row prep metadata for the unified per-token prep launches: each
/// row's absolute position and, per family, the single page holding it
/// with that page's index as the row's window origin — so the kernels'
/// `pos / page_size - origin` lands on slot 0 of a one-page window. One
/// launch per family then covers prefill and decode rows uniformly,
/// whatever the step's segment split.
struct MixRows {
    local_ps: usize,
    global_ps: usize,
    positions: Vec<i32>,
    local_pages: Vec<i32>,
    local_origins: Vec<i32>,
    global_pages: Vec<i32>,
    global_origins: Vec<i32>,
}

impl MixRows {
    fn new(local_ps: usize, global_ps: usize) -> Self {
        Self {
            local_ps,
            global_ps,
            positions: Vec::new(),
            local_pages: Vec::new(),
            local_origins: Vec::new(),
            global_pages: Vec::new(),
            global_origins: Vec::new(),
        }
    }

    fn push(
        &mut self,
        pos: usize,
        local_window: &[i32],
        local_origin_pages: usize,
        global_table: &[i32],
    ) -> Result<()> {
        let local_slot = pos / self.local_ps - local_origin_pages;
        let global_slot = pos / self.global_ps;
        self.positions
            .push(i32::try_from(pos).context("mix position fits i32")?);
        self.local_pages.push(
            *local_window
                .get(local_slot)
                .with_context(|| format!("row pos {pos} outside its local window"))?,
        );
        self.local_origins
            .push(i32::try_from(pos / self.local_ps).context("mix local origin fits i32")?);
        self.global_pages.push(
            *global_table
                .get(global_slot)
                .with_context(|| format!("row pos {pos} outside its global table"))?,
        );
        self.global_origins
            .push(i32::try_from(global_slot).context("mix global origin fits i32")?);
        Ok(())
    }

    fn reset(&mut self, rows: usize) {
        for values in [
            &mut self.positions,
            &mut self.local_pages,
            &mut self.local_origins,
            &mut self.global_pages,
            &mut self.global_origins,
        ] {
            values.clear();
            values.reserve(rows);
        }
    }
}

struct HostPlanningScratch {
    local_rows: Vec<Vec<i32>>,
    local_last: Vec<usize>,
    local_start: Vec<usize>,
    seq_lens: Vec<usize>,
    global_rows: Vec<Vec<i32>>,
    global_last: Vec<usize>,
    global_start: Vec<usize>,
    local_origins: Vec<i32>,
    global_pages_cat: Vec<i32>,
    global_indptr: Vec<i32>,
    positions: Vec<i32>,
    pseudo_pages: Vec<i32>,
    pseudo_indptr: Vec<i32>,
    pseudo_last: Vec<i32>,
    pseudo_kv_lens: Vec<usize>,
    ones: Vec<usize>,
    prefill_global_rows: Vec<Vec<i32>>,
    prefill_global_last: Vec<usize>,
    prefill_global_start: Vec<usize>,
    segs: Vec<MixSeg>,
    mix_rows: MixRows,
    ids: Vec<u32>,
}

impl HostPlanningScratch {
    fn new(local_ps: usize, global_ps: usize) -> Self {
        Self {
            local_rows: Vec::new(),
            local_last: Vec::new(),
            local_start: Vec::new(),
            seq_lens: Vec::new(),
            global_rows: Vec::new(),
            global_last: Vec::new(),
            global_start: Vec::new(),
            local_origins: Vec::new(),
            global_pages_cat: Vec::new(),
            global_indptr: Vec::new(),
            positions: Vec::new(),
            pseudo_pages: Vec::new(),
            pseudo_indptr: Vec::new(),
            pseudo_last: Vec::new(),
            pseudo_kv_lens: Vec::new(),
            ones: Vec::new(),
            prefill_global_rows: Vec::new(),
            prefill_global_last: Vec::new(),
            prefill_global_start: Vec::new(),
            segs: Vec::new(),
            mix_rows: MixRows::new(local_ps, global_ps),
            ids: Vec::new(),
        }
    }

    fn reset_decode(&mut self, rows: usize) {
        reuse_rows(&mut self.local_rows, rows);
        reuse_rows(&mut self.global_rows, rows);
        self.local_last.clear();
        self.local_start.clear();
        self.global_last.clear();
        self.global_start.clear();
        self.local_origins.clear();
        self.global_pages_cat.clear();
        self.positions.clear();
        self.pseudo_pages.clear();
        self.pseudo_last.clear();
        self.pseudo_kv_lens.clear();
        self.ids.clear();
        self.ones.clear();
        self.ones.resize(rows, 1);
        reset_indptr(&mut self.global_indptr);
        reset_indptr(&mut self.pseudo_indptr);
    }

    fn reset_mixed(&mut self, prompts: usize, batch: usize, rows: usize) {
        reuse_rows(&mut self.local_rows, prompts + batch);
        reuse_rows(&mut self.global_rows, batch);
        reuse_rows(&mut self.prefill_global_rows, prompts);
        self.local_last.clear();
        self.local_start.clear();
        self.seq_lens.clear();
        self.global_last.clear();
        self.prefill_global_last.clear();
        self.prefill_global_start.clear();
        self.pseudo_pages.clear();
        self.pseudo_last.clear();
        self.pseudo_kv_lens.clear();
        self.segs.clear();
        self.ids.clear();
        reset_indptr(&mut self.pseudo_indptr);
        self.mix_rows.reset(rows);
    }
}

fn reuse_rows(rows: &mut Vec<Vec<i32>>, needed: usize) {
    let retained = needed.max(rows.len());
    rows.resize_with(retained, Vec::new);
    for row in rows.iter_mut().take(needed) {
        row.clear();
    }
}

fn reset_indptr(indptr: &mut Vec<i32>) {
    indptr.clear();
    indptr.push(0);
}

struct GemmaStepPlan {
    start_pos: usize,
    local_page_origin: usize,
    local_plan: PrefillPagedPlan,
    global_plan: PrefillPagedPlan,
}

fn upload_prefix<T: cudarc::driver::DeviceRepr>(
    ctx: &DeviceContext,
    slot: &mut CudaSlice<T>,
    values: &[T],
) -> Result<()> {
    anyhow::ensure!(
        values.len() <= slot.len(),
        "step metadata of {} entries overruns its {} entry slot",
        values.len(),
        slot.len()
    );
    let mut view = slot.slice_mut(..values.len());
    ctx.stream
        .memcpy_htod(values, &mut view)
        .map_err(|err| anyhow::anyhow!("step metadata H2D failed: {err}"))
}

/// Enqueue device-to-device copies of whole pool pages (`src[i] -> dst[i]`)
/// on `ctx.stream`. A page is one contiguous `page_stride`-element span;
/// source and destination live in the same buffer, which is why this
/// reaches for the raw driver copy instead of cudarc's safe API.
fn copy_pool_pages(
    ctx: &DeviceContext,
    buffer: &CudaSlice<bf16>,
    layout: &pegainfer_core::kv_pool::KvLayout,
    src: &[i32],
    dst: &[i32],
) -> Result<()> {
    use cudarc::driver::DevicePtr;
    anyhow::ensure!(
        src.len() == dst.len(),
        "page copy list mismatch: {} src vs {} dst",
        src.len(),
        dst.len()
    );
    let page_bytes = layout.page_stride * std::mem::size_of::<bf16>();
    let (base, _guard) = buffer.device_ptr(&ctx.stream);
    for (&s, &d) in src.iter().zip(dst) {
        let src_ptr = base + s as u64 * page_bytes as u64;
        let dst_ptr = base + d as u64 * page_bytes as u64;
        let rc = unsafe {
            cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                dst_ptr,
                src_ptr,
                page_bytes,
                ctx.stream.cu_stream(),
            )
        };
        anyhow::ensure!(
            rc == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "page D2D copy failed: {rc:?}"
        );
    }
    Ok(())
}

/// The attention path's working set. Both families run the same tower over
/// the same rows, so one set serves both: each buffer is allocated at the
/// wider family's width and reshaped per layer.
struct AttnScratch {
    normed_x: HiddenStates,
    q_states: HiddenStates,
    k_states: HiddenStates,
    v_states: HiddenStates,
    q_prep: HiddenStates,
    attn: HiddenStates,
}

impl AttnScratch {
    fn new(
        ctx: &DeviceContext,
        local: &LayerGeometry,
        global: &LayerGeometry,
        max_rows: usize,
    ) -> Result<Self> {
        let q_dim = |geom: &LayerGeometry| geom.num_q_heads * geom.head_dim;
        let kv_dim = |geom: &LayerGeometry| geom.num_kv_heads * geom.head_dim;
        let q_max = q_dim(local).max(q_dim(global));
        let kv_max = kv_dim(local).max(kv_dim(global));
        Ok(Self {
            normed_x: HiddenStates::zeros(ctx, local.hidden_size, max_rows)?,
            q_states: HiddenStates::zeros(ctx, q_max, max_rows)?,
            k_states: HiddenStates::zeros(ctx, kv_max, max_rows)?,
            v_states: HiddenStates::zeros(ctx, kv_max, max_rows)?,
            q_prep: HiddenStates::zeros(ctx, q_max, max_rows)?,
            attn: HiddenStates::zeros(ctx, q_max, max_rows)?,
        })
    }

    fn set(&mut self, geom: &LayerGeometry, seq_len: usize) {
        let q_dim = geom.num_q_heads * geom.head_dim;
        let kv_dim = geom.num_kv_heads * geom.head_dim;
        for (buf, hidden_dim) in [
            (&mut self.normed_x, geom.hidden_size),
            (&mut self.q_states, q_dim),
            (&mut self.k_states, kv_dim),
            (&mut self.v_states, kv_dim),
            (&mut self.q_prep, q_dim),
            (&mut self.attn, q_dim),
        ] {
            buf.hidden_dim = hidden_dim;
            buf.seq_len = seq_len;
        }
    }
}

/// The tower's whole working set for one step: attention buffers, epilogue
/// buffers, and the hidden pair the layers alternate between so no layer
/// writes the buffer it is reading.
/// Order `ctx.stream` producers (plan uploads, token-id H2D, buffer
/// allocations) before the override stream consumes them. No-op without an
/// override.
fn fence_producers_before_override(ctx: &DeviceContext) -> Result<()> {
    use cudarc::driver::sys;
    if !pegainfer_core::tensor::has_stream_override() {
        return Ok(());
    }
    let override_stream = pegainfer_core::tensor::active_cu_stream(ctx);
    let producer = ctx.stream.cu_stream();
    unsafe {
        let mut event: sys::CUevent = std::ptr::null_mut();
        let create = sys::cuEventCreate(
            &raw mut event,
            sys::CUevent_flags_enum::CU_EVENT_DISABLE_TIMING as u32,
        );
        anyhow::ensure!(
            create == sys::CUresult::CUDA_SUCCESS,
            "cuEventCreate (producer fence) failed: {create:?}"
        );
        let record = sys::cuEventRecord(event, producer);
        let wait = if record == sys::CUresult::CUDA_SUCCESS {
            sys::cuStreamWaitEvent(override_stream, event, 0)
        } else {
            record
        };
        let destroy = sys::cuEventDestroy_v2(event);
        anyhow::ensure!(
            record == sys::CUresult::CUDA_SUCCESS,
            "cuEventRecord (producer fence) failed: {record:?}"
        );
        anyhow::ensure!(
            wait == sys::CUresult::CUDA_SUCCESS,
            "cuStreamWaitEvent (producer fence) failed: {wait:?}"
        );
        anyhow::ensure!(
            destroy == sys::CUresult::CUDA_SUCCESS,
            "cuEventDestroy (producer fence) failed: {destroy:?}"
        );
    }
    Ok(())
}

struct TowerScratch {
    attn: AttnScratch,
    epilogue: EpilogueScratch,
    hidden: [HiddenStates; 2],
}

impl TowerScratch {
    fn new(
        ctx: &DeviceContext,
        local: &LayerGeometry,
        global: &LayerGeometry,
        max_rows: usize,
    ) -> Result<Self> {
        Ok(Self {
            attn: AttnScratch::new(ctx, local, global, max_rows)?,
            epilogue: EpilogueScratch::new(ctx, local, max_rows)?,
            hidden: [
                HiddenStates::zeros(ctx, local.hidden_size, max_rows)?,
                HiddenStates::zeros(ctx, local.hidden_size, max_rows)?,
            ],
        })
    }

    fn open(&mut self, seq_len: usize) -> Result<()> {
        self.epilogue.set_rows(seq_len)?;
        for buf in &mut self.hidden {
            buf.seq_len = seq_len;
        }
        Ok(())
    }
}

fn bucket_slot(bucket: usize) -> usize {
    bucket.trailing_zeros() as usize
}

fn hidden_pair(hidden: &mut [HiddenStates; 2], src: usize) -> (&HiddenStates, &mut HiddenStates) {
    let (first, second) = hidden.split_at_mut(1);
    if src == 0 {
        (&first[0], &mut second[0])
    } else {
        (&second[0], &mut first[0])
    }
}

const GLOBAL_SPLIT_CHUNK_TOKENS: usize = 256;

/// The global family's decode tables, uploaded per step: the per-request
/// half feeds the prep, the factor-repeated half feeds the split-KV
/// attention read over the pseudo-requests (see [`global_split_factor`]).
struct GlobalTables {
    pages: CudaSlice<i32>,
    indptr: CudaSlice<i32>,
    /// Per-row window-start pages for the prep. Pure decode rows read their
    /// whole row from page 0, so this stays zero-filled; a per-token prep
    /// may compress a row's window to the single page holding its position.
    origins: CudaSlice<i32>,
    positions: CudaSlice<i32>,
    pseudo_pages: CudaSlice<i32>,
    pseudo_indptr: CudaSlice<i32>,
    pseudo_last: CudaSlice<i32>,
}

/// The split-KV plan, refilled per step at graph-stable padded shapes; the
/// chunk size is written to its device slot once at alloc.
struct SplitKvState {
    request_indices_d: CudaSlice<i32>,
    kv_tile_indices_d: CudaSlice<i32>,
    chunk_size_d: CudaSlice<i32>,
    o_indptr_d: CudaSlice<i32>,
    valid_mask_d: CudaSlice<u8>,
    tmp_v: CudaSlice<bf16>,
    tmp_s: CudaSlice<f32>,
    /// Chunk-count bound per pseudo-request; a step's padded slot count is
    /// the split factor times its bucket times this.
    cap: usize,
}

pub(crate) struct StepArena {
    tower: TowerScratch,
    host: HostPlanningScratch,
    local_plan: PrefillPagedPlan,
    global_tables: GlobalTables,
    global_split: SplitKvState,
    local_origins: CudaSlice<i32>,
    ids: CudaSlice<u32>,
    /// Mixed-step per-row prep metadata at step-stable pointers, covering
    /// prefill and decode rows uniformly: absolute position and, per
    /// family, the single page holding it with that page's index as the
    /// row's window origin. `mix_indptr` is the identity ramp written once
    /// at alloc — with one page per row the ramp serves both families. One
    /// per-token prep launch per family then replaces the per-segment
    /// walks, whatever the step's segment split.
    mix_positions: CudaSlice<i32>,
    mix_local_pages: CudaSlice<i32>,
    mix_local_origins: CudaSlice<i32>,
    mix_global_pages: CudaSlice<i32>,
    mix_global_origins: CudaSlice<i32>,
    mix_indptr: CudaSlice<i32>,
    head_normed: HiddenStates,
    logits: HiddenStates,
    /// One graph per power-of-two bucket, at index `log2(bucket)`, captured
    /// by the startup sweep. The captured kernels read and write only
    /// step-stable pointers; per-step change rides the plan and metadata
    /// contents uploaded before launch.
    graphs: Vec<CudaGraphState>,
    graph_enabled: bool,
    /// Floor for the padded bucket. Only the pre-capture sweep raises it, to
    /// drive every bucket from a single dummy request.
    min_bucket: usize,
    max_rows: usize,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

impl StepArena {
    /// Settle the step's preconditions — the allocation stream and a row
    /// count the buffers can hold — since neither can be safely deferred
    /// until the epilogue, by which point the step has already written KV.
    fn open(&mut self, ctx: &DeviceContext, rows: usize) -> Result<()> {
        anyhow::ensure!(
            std::sync::Arc::ptr_eq(&ctx.stream, &self.stream),
            "a step arena must be used on the stream it was allocated on"
        );
        anyhow::ensure!(
            rows <= self.max_rows,
            "step of {rows} rows exceeds the arena's {} row ceiling",
            self.max_rows
        );
        self.tower.open(rows)
    }
}

/// How many pseudo-requests the global decode read presents each request
/// as. FlashInfer's decode dispatcher compiles GQA groups {1,2,3,4,8}: a
/// dispatchable group passes through whole, and a non-dispatchable group
/// over one KV head halves into pseudo-requests — an exact memory identity
/// only because MQA gives every query head the same KV head (the 12B
/// global family's 16 over 1). Anything else fails loud.
pub(crate) fn global_split_factor(config: &Gemma4Config) -> Result<usize> {
    const DISPATCHABLE: [usize; 5] = [1, 2, 3, 4, 8];
    let q = config.num_attention_heads;
    let kv = config.num_global_key_value_heads;
    anyhow::ensure!(
        kv > 0 && q.is_multiple_of(kv),
        "global family of {q} query heads over {kv} KV heads is not a whole GQA group"
    );
    let group = q / kv;
    if DISPATCHABLE.contains(&group) {
        return Ok(1);
    }
    if kv == 1 && group.is_multiple_of(2) && DISPATCHABLE.contains(&(group / 2)) {
        return Ok(2);
    }
    anyhow::bail!(
        "the global decode read has no dispatch for {q} query heads over {kv} KV heads \
         (GQA group {group}): supported are groups 1,2,3,4,8 whole, or twice one of \
         those over a single KV head"
    )
}

/// Everything a serving step needs that outlives requests.
pub(crate) struct GemmaServe {
    /// The weights these pools, rope tables and layer numbering were built
    /// for. Holding them is what makes a step's model identity structural:
    /// KV pages written under one checkpoint are meaningless under another.
    weights: Gemma4Weights,
    pub(crate) local_pool: KvPool,
    pub(crate) global_pool: KvPool,
    local_geom: LayerGeometry,
    global_geom: LayerGeometry,
    sliding_window: usize,
    global_split_factor: usize,
    final_logit_softcapping: f32,
    release_enabled: bool,
    sliding_cos: DeviceVec,
    sliding_sin: DeviceVec,
    global_cos: DeviceVec,
    global_sin: DeviceVec,
    cos_max_pos: usize,
    /// Model layer index -> index within its family's pool layer axis.
    family_index: Vec<usize>,
}

impl GemmaServe {
    fn decode_row(
        &self,
        kv: &GemmaKv,
        local_pages: &mut Vec<i32>,
        global_pages: &mut Vec<i32>,
    ) -> Result<DecodeRow> {
        let position = kv.local.seq_len();
        let kv_len = position + 1;
        self.check_step_bounds(kv, kv_len)?;
        let page = self.local_pool.layout().page_size;
        let local_origin = kv.local.origin_pages();
        let origin_tokens = local_origin * page;
        let relative_len = kv_len
            .checked_sub(origin_tokens)
            .context("the resident window starts past the step's frontier")?;
        let local_start = position
            .checked_sub(origin_tokens)
            .context("the step starts before the resident window")?;
        kv.local.extend_page_row(local_pages);
        anyhow::ensure!(
            local_pages.len() == relative_len.div_ceil(page),
            "local resident row of {} pages against {relative_len} tokens",
            local_pages.len()
        );
        let remainder = relative_len % page;
        let local_last = if remainder == 0 { page } else { remainder };
        let global = kv.global.desc_for_len(kv_len)?;
        kv.global.extend_page_indices_i32(global_pages);
        Ok(DecodeRow {
            position,
            local_last,
            local_start,
            local_origin,
            global_last: global.last_page_len(),
        })
    }

    pub(crate) fn new(
        ctx: &DeviceContext,
        weights: Gemma4Weights,
        max_context: usize,
        local_pages: usize,
        global_pages: usize,
    ) -> Result<Self> {
        // One source of truth for geometry, rope tables and layer numbering.
        let config = &weights.config;
        let device_ordinal = ctx.device_ordinal;
        // One tensor answers for the set: the loader materializes every
        // weight through a single context.
        let weights_ordinal = weights.embed_tokens.data.ordinal();
        anyhow::ensure!(
            weights_ordinal == device_ordinal,
            "weights live on device {weights_ordinal} but this context \
             allocates on device {device_ordinal}"
        );
        let global_split_factor = global_split_factor(config)?;
        let (mut locals, mut globals) = (0usize, 0usize);
        let family_index = config
            .layer_types
            .iter()
            .map(|kind| match kind {
                LayerKind::Sliding => {
                    locals += 1;
                    locals - 1
                }
                LayerKind::Global => {
                    globals += 1;
                    globals - 1
                }
            })
            .collect();
        let local_pool = KvPool::new(
            ctx,
            locals,
            config.num_key_value_heads,
            config.head_dim,
            PAGE_SIZE,
            local_pages,
        )?;
        let global_pool = KvPool::new(
            ctx,
            globals,
            config.num_global_key_value_heads,
            config.global_head_dim,
            PAGE_SIZE,
            global_pages,
        )?;
        let local_geom = LayerGeometry::local_of(config);
        let global_geom = LayerGeometry::global_of(config);
        let (sliding_cos, sliding_sin) = pegainfer_core::rope::precompute_rope(
            ctx,
            &RopeTableSpec {
                rotary_dim: local_geom.head_dim,
                frequency_dim: local_geom.head_dim,
                max_seq_len: max_context,
                theta: config.sliding_rope_theta,
            },
        )?;
        let (global_cos, global_sin) = build_proportional_rope_tables(
            ctx,
            config.global_rope_theta,
            global_geom.head_dim,
            config.global_rotary_dim,
            max_context,
        )?;
        let (sliding_window, final_logit_softcapping) =
            (config.sliding_window, config.final_logit_softcapping);
        Ok(Self {
            weights,
            local_pool,
            global_pool,
            local_geom,
            global_geom,
            sliding_window,
            global_split_factor,
            final_logit_softcapping,
            release_enabled: true,
            sliding_cos,
            sliding_sin,
            global_cos,
            global_sin,
            cos_max_pos: max_context,
            family_index,
        })
    }

    /// One arena per engine thread, sized for the decode step; a prompt
    /// builds a [`TowerScratch`] for its own width instead. A request's
    /// tiles are `ceil(rows * group / cta_tile_q)` with a positive
    /// `cta_tile_q`, so `rows * group` bounds each plan.
    pub(crate) fn alloc_step_arena(
        &self,
        ctx: &DeviceContext,
        max_rows: usize,
        graph_enabled: bool,
    ) -> Result<StepArena> {
        anyhow::ensure!(
            max_rows.is_power_of_two(),
            "the arena's {max_rows} rows must be a power of two: steps pad to buckets"
        );
        let group = |geom: &LayerGeometry| geom.num_q_heads / geom.num_kv_heads;
        let alloc = |err: &'static str| {
            move |e: cudarc::driver::DriverError| anyhow::anyhow!("{err} alloc failed: {e}")
        };
        let factor = self.global_split_factor;
        let global_split_cap = self.cos_max_pos.div_ceil(GLOBAL_SPLIT_CHUNK_TOKENS);
        let global_split_slots = factor * max_rows * global_split_cap;
        let global_split_heads = self.global_geom.num_q_heads / factor;
        let mut global_chunk = ctx
            .stream
            .alloc_zeros(1)
            .map_err(alloc("global chunk size"))?;
        ctx.stream
            .memcpy_htod(&[GLOBAL_SPLIT_CHUNK_TOKENS as i32], &mut global_chunk)
            .map_err(|e| anyhow::anyhow!("global chunk-size upload failed: {e}"))?;
        // A mixed step's rows are bounded by the serving ceiling's prompt
        // rows plus a full decode bucket; the indptr ramp is written once.
        let mix_rows_cap = self.cos_max_pos + max_rows;
        let mut mix_indptr = ctx
            .stream
            .alloc_zeros(mix_rows_cap + 1)
            .map_err(alloc("mix indptr"))?;
        let ramp: Vec<i32> =
            (0..=i32::try_from(mix_rows_cap).context("mix rows cap fits i32")?).collect();
        ctx.stream
            .memcpy_htod(&ramp, &mut mix_indptr)
            .map_err(|e| anyhow::anyhow!("mix indptr ramp upload failed: {e}"))?;
        Ok(StepArena {
            tower: TowerScratch::new(ctx, &self.local_geom, &self.global_geom, max_rows)?,
            host: HostPlanningScratch::new(
                self.local_pool.layout().page_size,
                self.global_pool.layout().page_size,
            ),
            local_plan: PrefillPagedPlan::new_preallocated(
                ctx,
                max_rows,
                self.local_pool.capacity_pages(),
                max_rows,
                max_rows * group(&self.local_geom),
            )?,
            global_tables: GlobalTables {
                pages: ctx
                    .stream
                    .alloc_zeros(self.global_pool.capacity_pages())
                    .map_err(alloc("global pages"))?,
                indptr: ctx
                    .stream
                    .alloc_zeros(max_rows + 1)
                    .map_err(alloc("global indptr"))?,
                origins: ctx
                    .stream
                    .alloc_zeros(max_rows)
                    .map_err(alloc("global origins"))?,
                positions: ctx
                    .stream
                    .alloc_zeros(max_rows)
                    .map_err(alloc("global positions"))?,
                pseudo_pages: ctx
                    .stream
                    .alloc_zeros(factor * self.global_pool.capacity_pages())
                    .map_err(alloc("global pseudo pages"))?,
                pseudo_indptr: ctx
                    .stream
                    .alloc_zeros(factor * max_rows + 1)
                    .map_err(alloc("global pseudo indptr"))?,
                pseudo_last: ctx
                    .stream
                    .alloc_zeros(factor * max_rows)
                    .map_err(alloc("global pseudo last-page lens"))?,
            },
            global_split: SplitKvState {
                request_indices_d: ctx
                    .stream
                    .alloc_zeros(global_split_slots)
                    .map_err(alloc("global split request indices"))?,
                kv_tile_indices_d: ctx
                    .stream
                    .alloc_zeros(global_split_slots)
                    .map_err(alloc("global split tile indices"))?,
                chunk_size_d: global_chunk,
                o_indptr_d: ctx
                    .stream
                    .alloc_zeros(factor * max_rows + 1)
                    .map_err(alloc("global split o_indptr"))?,
                valid_mask_d: ctx
                    .stream
                    .alloc_zeros(global_split_slots)
                    .map_err(alloc("global split valid mask"))?,
                tmp_v: ctx
                    .stream
                    .alloc_zeros(
                        global_split_slots * global_split_heads * self.global_geom.head_dim,
                    )
                    .map_err(alloc("global split tmp_v"))?,
                tmp_s: ctx
                    .stream
                    .alloc_zeros(global_split_slots * global_split_heads)
                    .map_err(alloc("global split tmp_s"))?,
                cap: global_split_cap,
            },
            local_origins: ctx.stream.alloc_zeros(max_rows).map_err(alloc("origins"))?,
            ids: ctx.stream.alloc_zeros(max_rows).map_err(alloc("ids"))?,
            mix_positions: ctx
                .stream
                .alloc_zeros(mix_rows_cap)
                .map_err(alloc("mix positions"))?,
            mix_local_pages: ctx
                .stream
                .alloc_zeros(mix_rows_cap)
                .map_err(alloc("mix local pages"))?,
            mix_local_origins: ctx
                .stream
                .alloc_zeros(mix_rows_cap)
                .map_err(alloc("mix local origins"))?,
            mix_global_pages: ctx
                .stream
                .alloc_zeros(mix_rows_cap)
                .map_err(alloc("mix global pages"))?,
            mix_global_origins: ctx
                .stream
                .alloc_zeros(mix_rows_cap)
                .map_err(alloc("mix global origins"))?,
            mix_indptr,
            head_normed: HiddenStates::zeros(ctx, self.local_geom.hidden_size, max_rows)?,
            logits: HiddenStates::zeros(ctx, self.weights.embed_tokens.rows, max_rows)?,
            graphs: (0..=bucket_slot(max_rows))
                .map(|_| CudaGraphState::new())
                .collect(),
            graph_enabled,
            min_bucket: 1,
            max_rows,
            stream: ctx.stream.clone(),
        })
    }

    pub(crate) fn alloc_kv(&self) -> GemmaKv {
        GemmaKv {
            local: SlidingLocalKv::new(self.local_pool.clone()),
            global: self.global_pool.alloc(),
        }
    }

    /// Copy a request's post-prefill KV into cache-owned pages — the
    /// capture half of the conversation-tail prefix cache. Returns `None`
    /// when either pool cannot spare the pages: capture is strictly
    /// best-effort and never competes with an admission. Stream order makes
    /// the copies safe — the prompt's KV writes were enqueued on
    /// `ctx.stream` before this call, and the copies enqueue after them on
    /// the same stream.
    pub(crate) fn capture_checkpoint(
        &self,
        ctx: &DeviceContext,
        kv: &GemmaKv,
        token_ids: &[u32],
    ) -> Option<crate::prefix_cache::CachedKv> {
        // Slack guard: cache pages must never push either pool to the edge,
        // or the saved prefill time resurfaces as refused admissions.
        const POOL_SLACK_PAGES: usize = 128;
        let global_src = kv.global.page_indices_i32();
        if global_src.len() > crate::prefix_cache::entry_global_pages(self.cos_max_pos) {
            return None;
        }
        let (local_avail, global_avail) = (
            self.local_pool.available_pages(),
            self.global_pool.available_pages(),
        );
        if local_avail < kv.local.held_pages() + POOL_SLACK_PAGES
            || global_avail < global_src.len() + POOL_SLACK_PAGES
        {
            return None;
        }
        let global_pages = self.global_pool.try_reserve(global_src.len())?;
        let mut local_pages = Vec::with_capacity(kv.local.held_pages());
        for _ in 0..kv.local.held_pages() {
            local_pages.push(self.local_pool.try_reserve(1)?);
        }
        let mut global_dst = Vec::with_capacity(global_src.len());
        global_pages.extend_page_indices_i32(&mut global_dst);
        let local_src = kv.local.page_row();
        let mut local_dst = Vec::with_capacity(local_pages.len());
        for reservation in &local_pages {
            reservation.extend_page_indices_i32(&mut local_dst);
        }
        let copy = copy_pool_pages(
            ctx,
            self.global_pool.buffer(),
            self.global_pool.layout(),
            &global_src,
            &global_dst,
        )
        .and_then(|()| {
            copy_pool_pages(
                ctx,
                self.local_pool.buffer(),
                self.local_pool.layout(),
                &local_src,
                &local_dst,
            )
        });
        if let Err(err) = copy {
            log::warn!("gemma4 prefix-cache capture failed: {err:#}");
            return None;
        }
        Some(crate::prefix_cache::CachedKv::new(
            token_ids.to_vec(),
            global_pages,
            local_pages,
            kv.local.origin_pages(),
        ))
    }

    /// Rebuild a request KV from a cached conversation tail — the restore
    /// half. The result is shaped exactly like a request that prefilled
    /// `[0, t)` itself and already released its out-of-window pages, so
    /// every downstream path (suffix prefill from `seq_len`, decode,
    /// release) proceeds unchanged. Any failure aborts the whole restore —
    /// both families or neither — and the caller falls back to a full
    /// prefill.
    pub(crate) fn restore_from_checkpoint(
        &self,
        ctx: &DeviceContext,
        entry: &crate::prefix_cache::CachedKv,
        t: usize,
    ) -> Result<GemmaKv> {
        anyhow::ensure!(
            t > 0 && t <= entry.token_ids.len(),
            "restore point {t} outside entry of {} tokens",
            entry.token_ids.len()
        );
        let page = self.local_pool.layout().page_size;
        // The release law's origin at frontier `t`; resolve guaranteed it
        // does not precede the captured window's origin.
        let origin_t = if t > self.sliding_window {
            (t - self.sliding_window) / page
        } else {
            0
        };
        anyhow::ensure!(
            origin_t >= entry.local_origin,
            "restore point {t} precedes the captured window (origin {origin_t} vs {})",
            entry.local_origin
        );
        let t_pages = t.div_ceil(page);
        let local_take = t_pages - origin_t;
        let local_skip = origin_t - entry.local_origin;
        anyhow::ensure!(
            local_skip + local_take <= entry.local_pages.len(),
            "restore window slice [{local_skip}, {}) outside {} cached pages",
            local_skip + local_take,
            entry.local_pages.len()
        );

        let mut global = self.global_pool.alloc();
        global.ensure_capacity(t)?;
        let global_dst = global.page_indices_i32();
        let mut global_src = Vec::new();
        entry.global_pages.extend_page_indices_i32(&mut global_src);
        global_src.truncate(global_dst.len());
        anyhow::ensure!(
            global_dst.len() == global_src.len(),
            "restore page count mismatch: {} cached vs {} allocated",
            global_src.len(),
            global_dst.len()
        );
        let mut resident = Vec::with_capacity(local_take);
        for _ in 0..local_take {
            resident.push(
                self.local_pool
                    .try_reserve(1)
                    .context("restore: local pool out of pages")?,
            );
        }
        let mut local_src = Vec::with_capacity(local_take);
        for reservation in entry.local_pages.iter().skip(local_skip).take(local_take) {
            reservation.extend_page_indices_i32(&mut local_src);
        }
        let mut local_dst = Vec::with_capacity(local_take);
        for reservation in &resident {
            reservation.extend_page_indices_i32(&mut local_dst);
        }
        copy_pool_pages(
            ctx,
            self.global_pool.buffer(),
            self.global_pool.layout(),
            &global_src,
            &global_dst,
        )?;
        copy_pool_pages(
            ctx,
            self.local_pool.buffer(),
            self.local_pool.layout(),
            &local_src,
            &local_dst,
        )?;
        global.advance(t);
        Ok(GemmaKv {
            local: SlidingLocalKv::restore(self.local_pool.clone(), resident, origin_t, t),
            global,
        })
    }

    /// The eviction gate runs the same request twice, once with the front
    /// held resident, to show what release does and does not change.
    #[cfg(test)]
    pub(crate) fn set_release_for_test(&mut self, on: bool) {
        self.release_enabled = on;
    }

    fn advance_local(&self, kv: &mut GemmaKv, tokens: usize) -> Result<()> {
        if self.release_enabled {
            kv.local.advance_and_release(tokens, self.sliding_window)
        } else {
            kv.local.advance(tokens);
            Ok(())
        }
    }

    fn check_step_bounds(&self, kv: &GemmaKv, kv_len: usize) -> Result<()> {
        anyhow::ensure!(
            kv.local.seq_len() == kv.global.seq_len(),
            "the two families' frontiers diverged: local {} global {}",
            kv.local.seq_len(),
            kv.global.seq_len()
        );
        anyhow::ensure!(
            kv.local.belongs_to(&self.local_pool) && kv.global.belongs_to(&self.global_pool),
            "a KV state came from another pool; its page ids do not address \
             this one's buffer"
        );
        anyhow::ensure!(
            kv_len <= self.cos_max_pos,
            "kv_len {kv_len} exceeds rope tables' {} rows",
            self.cos_max_pos
        );
        Ok(())
    }

    fn plan_step(
        &self,
        ctx: &DeviceContext,
        kv: &GemmaKv,
        start_pos: usize,
        seq_len: usize,
    ) -> Result<GemmaStepPlan> {
        let kv_len = start_pos + seq_len;
        let global_desc = kv.global.desc_for_len(kv_len)?;
        // The local plan lives in cache-relative coordinates: the resident row
        // starts `origin_pages` pages into the sequence, and window_left masks
        // whatever sub-window prefix the first page still carries, so a
        // resident start that is not window-aligned loses nothing.
        let page = kv.local.layout().page_size;
        let origin_tokens = kv.local.origin_pages() * page;
        let rel_kv_len = kv_len
            .checked_sub(origin_tokens)
            .context("the resident window starts past the step's frontier")?;
        let rel_start = start_pos
            .checked_sub(origin_tokens)
            .context("the step starts before the resident window")?;
        let row = kv.local.page_row();
        anyhow::ensure!(
            row.len() == rel_kv_len.div_ceil(page),
            "local resident row of {} pages against {rel_kv_len} tokens",
            row.len()
        );
        let rel_last_page = if rel_kv_len.is_multiple_of(page) {
            page
        } else {
            rel_kv_len % page
        };
        let local_plan = PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
            ctx,
            &[row],
            &[rel_last_page],
            &[rel_start],
            &[seq_len],
            self.local_geom.num_q_heads,
            self.local_geom.num_kv_heads,
            self.local_geom.head_dim,
            0,
        )?;
        let global_plan = PrefillPagedPlan::new(
            ctx,
            &global_desc,
            start_pos,
            seq_len,
            self.global_geom.num_q_heads,
            self.global_geom.num_kv_heads,
            self.global_geom.head_dim,
        )?;
        Ok(GemmaStepPlan {
            start_pos,
            local_page_origin: kv.local.origin_pages(),
            local_plan,
            global_plan,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn local_layer_serve(
        &self,
        ctx: &DeviceContext,
        tower: &mut TowerScratch,
        layer: &Gemma4Layer,
        family_layer: usize,
        seq_len: usize,
        prep: PrepRef<'_>,
        local_plan: &PrefillPagedPlan,
        src: usize,
    ) -> Result<()> {
        let geom = &self.local_geom;
        let q_dim = geom.num_q_heads * geom.head_dim;
        let kv_dim = geom.num_kv_heads * geom.head_dim;
        let v_proj = layer
            .attention
            .v_proj
            .as_ref()
            .context("local layer requires v_proj")?;
        let TowerScratch {
            attn: scratch,
            epilogue,
            hidden,
        } = tower;
        scratch.set(geom, seq_len);
        let (x, out) = hidden_pair(hidden, src);

        ops::rms_norm_batch_into(
            ctx,
            x,
            &layer.input_layernorm,
            geom.rms_norm_eps,
            &mut scratch.normed_x,
        );
        ops::gemm_rows_into_checked(
            ctx,
            &layer.attention.q_proj,
            0,
            q_dim,
            &scratch.normed_x,
            &mut scratch.q_states,
        )?;
        ops::gemm_rows_into_checked(
            ctx,
            &layer.attention.k_proj,
            0,
            kv_dim,
            &scratch.normed_x,
            &mut scratch.k_states,
        )?;
        ops::gemm_rows_into_checked(
            ctx,
            v_proj,
            0,
            kv_dim,
            &scratch.normed_x,
            &mut scratch.v_states,
        )?;

        match prep {
            PrepRef::Single {
                start_pos,
                local_page_origin,
                ..
            } => {
                ops::qkv_norm_rope_paged_prefill_hd256_plain_into(
                    ctx,
                    &scratch.q_states,
                    &scratch.k_states,
                    &scratch.v_states,
                    &mut scratch.q_prep,
                    0,
                    self.local_pool.buffer(),
                    &self.local_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.sliding_cos,
                    &self.sliding_sin,
                    family_layer,
                    local_plan.page_indices_d(),
                    0,
                    local_page_origin,
                    start_pos,
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
            }
            PrepRef::Batched {
                local_origins,
                global_tables,
            } => {
                ops::qkv_norm_rope_paged_decode_hd256_plain_into(
                    ctx,
                    &scratch.q_states,
                    &scratch.k_states,
                    &scratch.v_states,
                    &mut scratch.q_prep,
                    0,
                    self.local_pool.buffer(),
                    &self.local_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.sliding_cos,
                    &self.sliding_sin,
                    family_layer,
                    local_plan.page_indices_d(),
                    local_plan.page_indptr_d(),
                    local_origins,
                    &global_tables.positions,
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
            }
            PrepRef::Mixed {
                mix_positions,
                mix_local_pages,
                mix_local_origins,
                mix_indptr,
                ..
            } => {
                // One per-token launch covers every row — each row writes
                // its position's slot in its own one-page window.
                ops::qkv_norm_rope_paged_decode_hd256_plain_into(
                    ctx,
                    &scratch.q_states,
                    &scratch.k_states,
                    &scratch.v_states,
                    &mut scratch.q_prep,
                    0,
                    self.local_pool.buffer(),
                    &self.local_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.sliding_cos,
                    &self.sliding_sin,
                    family_layer,
                    mix_local_pages,
                    mix_indptr,
                    mix_local_origins,
                    mix_positions,
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
            }
        }

        let window_left = i32::try_from(self.sliding_window - 1).expect("window fits i32");
        ops::batch_prefill_paged_window_hd256_into(
            ctx,
            &scratch.q_prep,
            self.local_pool.buffer(),
            &self.local_pool.layout().kernel_layout(),
            family_layer,
            local_plan,
            &mut scratch.attn,
            geom.num_q_heads,
            1.0,
            window_left,
        )?;
        attention_epilogue_into(ctx, layer, geom, x, &scratch.attn, epilogue, out)
    }

    #[allow(clippy::too_many_arguments)]
    fn global_layer_serve(
        &self,
        ctx: &DeviceContext,
        tower: &mut TowerScratch,
        layer: &Gemma4Layer,
        family_layer: usize,
        seq_len: usize,
        prep: PrepRef<'_>,
        split: Option<&mut SplitKvState>,
        src: usize,
    ) -> Result<()> {
        let geom = &self.global_geom;
        let q_dim = geom.num_q_heads * geom.head_dim;
        let kv_dim = geom.num_kv_heads * geom.head_dim;
        anyhow::ensure!(
            layer.attention.v_proj.is_none(),
            "global layer must not carry a v_proj; V is the k_proj fork"
        );
        let TowerScratch {
            attn: scratch,
            epilogue,
            hidden,
        } = tower;
        scratch.set(geom, seq_len);
        let (x, out) = hidden_pair(hidden, src);

        ops::rms_norm_batch_into(
            ctx,
            x,
            &layer.input_layernorm,
            geom.rms_norm_eps,
            &mut scratch.normed_x,
        );
        ops::gemm_rows_into_checked(
            ctx,
            &layer.attention.q_proj,
            0,
            q_dim,
            &scratch.normed_x,
            &mut scratch.q_states,
        )?;
        ops::gemm_rows_into_checked(
            ctx,
            &layer.attention.k_proj,
            0,
            kv_dim,
            &scratch.normed_x,
            &mut scratch.k_states,
        )?;

        // The prep writes both K and the weightless-normed V fork from the
        // one raw K read — no D2D fork copy on the serving path.
        match prep {
            PrepRef::Single {
                start_pos,
                global_plan,
                ..
            } => {
                ops::qk_norm_partial_rope_paged_prefill_hd512_into(
                    ctx,
                    &scratch.q_states,
                    &scratch.k_states,
                    &mut scratch.q_prep,
                    0,
                    self.global_pool.buffer(),
                    &self.global_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.global_cos,
                    &self.global_sin,
                    family_layer,
                    global_plan.page_indices_d(),
                    0,
                    start_pos,
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
                ops::batch_prefill_paged_hd512_into(
                    ctx,
                    &scratch.q_prep,
                    self.global_pool.buffer(),
                    &self.global_pool.layout().kernel_layout(),
                    family_layer,
                    global_plan,
                    &mut scratch.attn,
                    geom.num_q_heads,
                    1.0,
                )?;
            }
            PrepRef::Batched { global_tables, .. } => {
                let split = split.context("batched global decode needs the split-KV state")?;
                ops::qk_norm_partial_rope_paged_decode_hd512_into(
                    ctx,
                    &scratch.q_states,
                    &scratch.k_states,
                    &mut scratch.q_prep,
                    0,
                    self.global_pool.buffer(),
                    &self.global_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.global_cos,
                    &self.global_sin,
                    family_layer,
                    &global_tables.pages,
                    &global_tables.indptr,
                    &global_tables.origins,
                    &global_tables.positions,
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
                // A pure reshape: `[rows, q·512]` and `[factor·rows,
                // (q/factor)·512]` are the same memory.
                let factor = self.global_split_factor;
                scratch.q_prep.hidden_dim = q_dim / factor;
                scratch.q_prep.seq_len = factor * seq_len;
                scratch.attn.hidden_dim = q_dim / factor;
                scratch.attn.seq_len = factor * seq_len;
                let meta = ops::Hd512DecodeMetadata::new(
                    &global_tables.pseudo_pages,
                    &global_tables.pseudo_indptr,
                    &global_tables.pseudo_last,
                    &split.request_indices_d,
                    &split.kv_tile_indices_d,
                    &split.chunk_size_d,
                );
                ops::paged_attention_batch_decode_split_kv_hd512_into(
                    ctx,
                    &scratch.q_prep,
                    0,
                    self.global_pool.buffer(),
                    &self.global_pool.layout().kernel_layout(),
                    family_layer,
                    &meta,
                    &split.o_indptr_d,
                    &split.valid_mask_d,
                    &mut split.tmp_v,
                    &mut split.tmp_s,
                    factor * seq_len * split.cap,
                    &mut scratch.attn,
                    geom.num_q_heads / factor,
                    1.0,
                )?;
                scratch.q_prep.hidden_dim = q_dim;
                scratch.q_prep.seq_len = seq_len;
                scratch.attn.hidden_dim = q_dim;
                scratch.attn.seq_len = seq_len;
            }
            PrepRef::Mixed {
                prefill_len,
                mix_positions,
                mix_global_pages,
                mix_global_origins,
                mix_indptr,
                global_tables,
                global_prefill_plan,
                ..
            } => {
                let split = split.context("mixed global decode needs the split-KV state")?;
                // One per-token launch covers every row — each row writes
                // its position's slot in its own one-page window.
                ops::qk_norm_partial_rope_paged_decode_hd512_into(
                    ctx,
                    &scratch.q_states,
                    &scratch.k_states,
                    &mut scratch.q_prep,
                    0,
                    self.global_pool.buffer(),
                    &self.global_pool.layout().kernel_layout(),
                    &layer.attention.q_norm,
                    &layer.attention.k_norm,
                    &self.global_cos,
                    &self.global_sin,
                    family_layer,
                    mix_global_pages,
                    mix_indptr,
                    mix_global_origins,
                    mix_positions,
                    self.cos_max_pos,
                    geom.num_q_heads,
                    geom.num_kv_heads,
                    geom.head_dim,
                    geom.rms_norm_eps,
                )?;
                // The prompt rows read through the ragged prefill plan —
                // one entry per segment — and the suffix through the native
                // split entry at the pseudo-row offset, the same kernel and
                // chunk plan as a pure decode step.
                scratch.q_prep.seq_len = prefill_len;
                scratch.attn.seq_len = prefill_len;
                ops::batch_prefill_paged_hd512_into(
                    ctx,
                    &scratch.q_prep,
                    self.global_pool.buffer(),
                    &self.global_pool.layout().kernel_layout(),
                    family_layer,
                    global_prefill_plan,
                    &mut scratch.attn,
                    geom.num_q_heads,
                    1.0,
                )?;
                let batch = seq_len - prefill_len;
                let factor = self.global_split_factor;
                scratch.q_prep.hidden_dim = q_dim / factor;
                scratch.q_prep.seq_len = factor * seq_len;
                scratch.attn.hidden_dim = q_dim / factor;
                scratch.attn.seq_len = factor * seq_len;
                let meta = ops::Hd512DecodeMetadata::new(
                    &global_tables.pseudo_pages,
                    &global_tables.pseudo_indptr,
                    &global_tables.pseudo_last,
                    &split.request_indices_d,
                    &split.kv_tile_indices_d,
                    &split.chunk_size_d,
                );
                ops::paged_attention_batch_decode_split_kv_hd512_into(
                    ctx,
                    &scratch.q_prep,
                    factor * prefill_len,
                    self.global_pool.buffer(),
                    &self.global_pool.layout().kernel_layout(),
                    family_layer,
                    &meta,
                    &split.o_indptr_d,
                    &split.valid_mask_d,
                    &mut split.tmp_v,
                    &mut split.tmp_s,
                    factor * batch * split.cap,
                    &mut scratch.attn,
                    geom.num_q_heads / factor,
                    1.0,
                )?;
                scratch.q_prep.hidden_dim = q_dim;
                scratch.q_prep.seq_len = seq_len;
                scratch.attn.hidden_dim = q_dim;
                scratch.attn.seq_len = seq_len;
            }
        }
        attention_epilogue_into(ctx, layer, geom, x, &scratch.attn, epilogue, out)
    }

    fn plan_decode_batch(
        &self,
        ctx: &DeviceContext,
        arena: &mut StepArena,
        kvs: &[&mut GemmaKv],
        padded: usize,
    ) -> Result<()> {
        let batch = kvs.len();
        let StepArena {
            host,
            local_plan,
            global_tables,
            global_split,
            local_origins: origins_slot,
            ..
        } = arena;
        host.reset_decode(padded);
        let HostPlanningScratch {
            local_rows,
            local_last,
            local_start,
            global_rows,
            global_last,
            global_start,
            local_origins,
            global_pages_cat,
            global_indptr,
            positions,
            pseudo_pages,
            pseudo_indptr,
            pseudo_last,
            pseudo_kv_lens,
            ones,
            ..
        } = host;

        for (r, kv) in kvs.iter().enumerate() {
            let row = self.decode_row(kv, &mut local_rows[r], &mut global_rows[r])?;
            local_origins.push(i32::try_from(row.local_origin).context("origin fits i32")?);
            local_last.push(row.local_last);
            local_start.push(row.local_start);
            global_last.push(row.global_last);
            global_start.push(row.position);
        }

        // Pad rows write each pool's reserved padding page — never a real
        // request's KV — at position 0 with a one-token window.
        for r in batch..padded {
            local_origins.push(0);
            local_rows[r].push(self.local_pool.padding_page_id());
            local_last.push(1);
            local_start.push(0);
            global_rows[r].push(self.global_pool.padding_page_id());
            global_last.push(1);
            global_start.push(0);
        }
        local_plan.update_batch_with_cta_tile_q(
            ctx,
            &local_rows[..padded],
            local_last,
            local_start,
            ones,
            self.local_geom.num_q_heads,
            self.local_geom.num_kv_heads,
            self.local_geom.head_dim,
            0,
        )?;
        let global_page = self.global_pool.layout().page_size;
        let factor = self.global_split_factor;
        for r in 0..padded {
            let row = &global_rows[r];
            let row_len = i32::try_from(row.len()).context("global pages fit i32")?;
            let last = i32::try_from(global_last[r]).context("global last-page len fits i32")?;
            let kv_len = (row.len() - 1) * global_page + global_last[r];
            positions.push(i32::try_from(global_start[r]).context("position fits i32")?);
            global_indptr.push(global_indptr.last().unwrap() + row_len);
            global_pages_cat.extend_from_slice(row);
            for _ in 0..factor {
                pseudo_pages.extend_from_slice(row);
                pseudo_indptr.push(pseudo_indptr.last().unwrap() + row_len);
                pseudo_last.push(last);
                pseudo_kv_lens.push(kv_len);
            }
        }
        let global_csr = ops::build_split_kv_csr(
            GLOBAL_SPLIT_CHUNK_TOKENS,
            global_split.cap,
            pseudo_kv_lens,
            factor * padded,
        )?;
        upload_prefix(ctx, &mut global_tables.pages, global_pages_cat)?;
        upload_prefix(ctx, &mut global_tables.indptr, global_indptr)?;
        upload_prefix(ctx, &mut global_tables.positions, positions)?;
        upload_prefix(ctx, &mut global_tables.pseudo_pages, pseudo_pages)?;
        upload_prefix(ctx, &mut global_tables.pseudo_indptr, pseudo_indptr)?;
        upload_prefix(ctx, &mut global_tables.pseudo_last, pseudo_last)?;
        upload_prefix(
            ctx,
            &mut global_split.request_indices_d,
            &global_csr.request_indices,
        )?;
        upload_prefix(
            ctx,
            &mut global_split.kv_tile_indices_d,
            &global_csr.kv_tile_indices,
        )?;
        upload_prefix(ctx, &mut global_split.o_indptr_d, &global_csr.o_indptr)?;
        upload_prefix(
            ctx,
            &mut global_split.valid_mask_d,
            &global_csr.block_valid_mask,
        )?;
        upload_prefix(ctx, origins_slot, local_origins)
    }

    /// The stream that built the pools is the one that ordered every page
    /// write already in them, so a step on any other stream has no ordering
    /// against the KV it is about to read.
    fn check_stream(&self, ctx: &DeviceContext) -> Result<()> {
        anyhow::ensure!(
            std::sync::Arc::ptr_eq(&ctx.stream, self.local_pool.buffer().stream()),
            "a step must use the DeviceContext stream that constructed this GemmaServe"
        );
        Ok(())
    }

    /// Returns the hidden slot holding the tower's output.
    #[allow(clippy::too_many_arguments)]
    fn run_tower(
        &self,
        ctx: &DeviceContext,
        tower: &mut TowerScratch,
        ids: &CudaSlice<u32>,
        seq_len: usize,
        local_plan: &PrefillPagedPlan,
        prep: PrepRef<'_>,
        mut global_split: Option<&mut SplitKvState>,
    ) -> Result<usize> {
        let weights = &self.weights;
        ops::embedding_batch(ctx, &weights.embed_tokens, ids, &mut tower.hidden[0])?;
        ops::scale_bf16_in_place(
            ctx,
            &mut tower.hidden[0],
            embed_scale_bf16(self.local_geom.hidden_size),
        )?;
        let mut src = 0usize;
        for (index, kind) in weights.config.layer_types.iter().enumerate() {
            let layer = &weights.layers[index];
            let family_layer = self.family_index[index];
            match kind {
                LayerKind::Sliding => {
                    self.local_layer_serve(
                        ctx,
                        tower,
                        layer,
                        family_layer,
                        seq_len,
                        prep,
                        local_plan,
                        src,
                    )?;
                }
                LayerKind::Global => {
                    self.global_layer_serve(
                        ctx,
                        tower,
                        layer,
                        family_layer,
                        seq_len,
                        prep,
                        global_split.as_deref_mut(),
                        src,
                    )?;
                }
            }
            src ^= 1;
        }
        Ok(src)
    }

    /// One batched decode step: each request advances a single token and the
    /// batch shares every layer's weight pass. Row `r` of the returned logits
    /// is request `r`'s next-token distribution; they live in the arena and
    /// are valid until its next step.
    pub(crate) fn decode_batch_step<'a>(
        &self,
        ctx: &DeviceContext,
        arena: &'a mut StepArena,
        kvs: &mut [&mut GemmaKv],
        tokens: &[u32],
    ) -> Result<&'a mut HiddenStates> {
        let batch = kvs.len();
        let padded = self.prepare_decode_step(ctx, arena, kvs, tokens)?;
        let StepArena {
            tower,
            local_plan,
            global_tables,
            global_split,
            local_origins,
            ids,
            head_normed,
            logits,
            graphs,
            graph_enabled,
            ..
        } = arena;
        if *graph_enabled {
            let graph = &mut graphs[bucket_slot(padded)];
            anyhow::ensure!(
                graph.is_captured(),
                "no captured graph for bucket {padded}; the pre-capture sweep must cover \
                 every bucket before serving"
            );
            graph.launch_captured(ctx)?;
        } else {
            self.decode_gpu_body(
                ctx,
                tower,
                ids,
                padded,
                local_plan,
                global_tables,
                global_split,
                local_origins,
                head_normed,
                logits,
            )?;
        }
        logits.seq_len = batch;
        // Append-then-attend, per request: the batch shared a step, but each
        // request owns its own frontier and its own released front.
        for kv in kvs.iter_mut() {
            self.advance_local(kv, 1)?;
            kv.global.advance(1);
        }
        Ok(logits)
    }

    /// The decode step's GPU body — embedding through the LM head. A pure
    /// kernel sequence over step-stable pointers: no allocation, no
    /// synchronization, no pool bookkeeping, which is what makes it
    /// capturable.
    #[allow(clippy::too_many_arguments)]
    fn decode_gpu_body(
        &self,
        ctx: &DeviceContext,
        tower: &mut TowerScratch,
        ids: &CudaSlice<u32>,
        rows: usize,
        local_plan: &PrefillPagedPlan,
        global_tables: &GlobalTables,
        global_split: &mut SplitKvState,
        local_origins: &CudaSlice<i32>,
        head_normed: &mut HiddenStates,
        logits: &mut HiddenStates,
    ) -> Result<()> {
        let src = self.run_tower(
            ctx,
            tower,
            ids,
            rows,
            local_plan,
            PrepRef::Batched {
                local_origins,
                global_tables,
            },
            Some(global_split),
        )?;
        logits_tail_into(
            ctx,
            &self.weights,
            &tower.hidden[src],
            self.local_geom.rms_norm_eps,
            self.final_logit_softcapping,
            head_normed,
            logits,
        )
    }

    /// One mixed step: a whole admitted prompt (rows `[0..prompt_len)`)
    /// rides the same weight scan as the live decode batch (the row
    /// suffix). Always eager — the prompt length varies per admission, so
    /// this shape never rides a graph; the pure-decode steps around it keep
    /// their bucketed replays. The returned logits hold `batch + 1` rows:
    /// row 0 the prompt's next-token distribution, rows 1.. the decode
    /// batch in order.
    pub(crate) fn mixed_prefill_decode_step<'a>(
        &self,
        ctx: &DeviceContext,
        arena: &'a mut StepArena,
        prefills: &mut [(&mut GemmaKv, &[u32])],
        decode_kvs: &mut [&mut GemmaKv],
        decode_tokens: &[u32],
    ) -> Result<&'a mut HiddenStates> {
        let prompts = prefills.len();
        let prefill_len: usize = prefills.iter().map(|(_, p)| p.len()).sum();
        let batch = decode_kvs.len();
        anyhow::ensure!(
            prompts > 0 && prefill_len > 0 && batch > 0,
            "mixed step needs prompt rows ({prompts} prompts, {prefill_len} tokens) \
             and a live decode batch ({batch})"
        );
        anyhow::ensure!(
            decode_tokens.len() == batch,
            "mixed step has {batch} decode requests but {} tokens",
            decode_tokens.len()
        );
        anyhow::ensure!(
            batch + prompts <= arena.max_rows,
            "mixed step logits rows {} exceed the arena's {} row ceiling",
            batch + prompts,
            arena.max_rows
        );
        self.check_stream(ctx)?;
        for (_, prompt) in prefills.iter() {
            validate_tokens(&self.weights, self.local_geom.hidden_size, prompt)?;
        }
        validate_tokens(&self.weights, self.local_geom.hidden_size, decode_tokens)?;
        let rows = prefill_len + batch;
        let page = self.local_pool.layout().page_size;
        let global_page = self.global_pool.layout().page_size;
        let StepArena {
            host,
            global_tables,
            global_split,
            mix_positions,
            mix_local_pages,
            mix_local_origins,
            mix_global_pages,
            mix_global_origins,
            mix_indptr,
            head_normed,
            logits,
            ..
        } = arena;
        host.reset_mixed(prompts, batch, rows);
        let HostPlanningScratch {
            local_rows,
            local_last,
            local_start,
            seq_lens,
            global_rows,
            global_last,
            pseudo_pages,
            pseudo_indptr,
            pseudo_last,
            pseudo_kv_lens,
            prefill_global_rows,
            prefill_global_last,
            prefill_global_start,
            segs,
            mix_rows,
            ids: ids_host,
            ..
        } = host;

        // Prompt entries first, then the decode rows — the ragged shape the
        // plans, the row buffers and the segment table share. All entries
        // derive from pre-advance state, the same way both pure steps plan.
        let mut row_cursor = 0usize;
        for (r, (kv, prompt)) in prefills.iter().enumerate() {
            let start = kv.local.seq_len();
            let kv_len = start + prompt.len();
            self.check_step_bounds(kv, kv_len)?;
            let origin_tokens = kv.local.origin_pages() * page;
            let rel_kv_len = kv_len
                .checked_sub(origin_tokens)
                .context("the resident window starts past the step's frontier")?;
            let rel_start = start
                .checked_sub(origin_tokens)
                .context("the step starts before the resident window")?;
            // A walking prompt parks every page up front, so its resident
            // row over-covers a mid-walk entry; the attention derives its
            // kv length from the row, so each entry's row is truncated to
            // exactly its coverage — an identity for whole-prompt steps —
            // after asserting the row reaches that far.
            let row = &mut local_rows[r];
            kv.local.extend_page_row(row);
            anyhow::ensure!(
                row.len() >= rel_kv_len.div_ceil(page),
                "local resident row of {} pages cannot cover {rel_kv_len} tokens",
                row.len()
            );
            row.truncate(rel_kv_len.div_ceil(page));
            let rel_last = if rel_kv_len.is_multiple_of(page) {
                page
            } else {
                rel_kv_len % page
            };
            let global_row = &mut prefill_global_rows[r];
            kv.global.extend_page_indices_i32(global_row);
            anyhow::ensure!(
                global_row.len() >= kv_len.div_ceil(global_page),
                "global resident row of {} pages cannot cover {kv_len} tokens",
                global_row.len()
            );
            global_row.truncate(kv_len.div_ceil(global_page));
            let global_last = if kv_len.is_multiple_of(global_page) {
                global_page
            } else {
                kv_len % global_page
            };
            for i in 0..prompt.len() {
                mix_rows.push(start + i, row, kv.local.origin_pages(), global_row)?;
            }
            segs.push(MixSeg {
                row_offset: row_cursor,
                rows: prompt.len(),
            });
            row_cursor += prompt.len();
            prefill_global_last.push(global_last);
            prefill_global_start.push(start);
            local_last.push(rel_last);
            local_start.push(rel_start);
            seq_lens.push(prompt.len());
        }
        for (r, kv) in decode_kvs.iter().enumerate() {
            let local_row = &mut local_rows[prompts + r];
            let global_row = &mut global_rows[r];
            let row = self.decode_row(kv, local_row, global_row)?;
            mix_rows.push(row.position, local_row, row.local_origin, global_row)?;
            local_last.push(row.local_last);
            local_start.push(row.local_start);
            seq_lens.push(1);
            global_last.push(row.global_last);
        }
        // One ragged local plan covers every row's windowed attention read.
        let local_plan = PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
            ctx,
            &local_rows[..prompts + batch],
            local_last,
            local_start,
            seq_lens,
            self.local_geom.num_q_heads,
            self.local_geom.num_kv_heads,
            self.local_geom.head_dim,
            0,
        )?;
        let global_prefill_plan = PrefillPagedPlan::from_raw_batch_with_cta_tile_q(
            ctx,
            &prefill_global_rows[..prompts],
            prefill_global_last,
            prefill_global_start,
            &seq_lens[..prompts],
            self.global_geom.num_q_heads,
            self.global_geom.num_kv_heads,
            self.global_geom.head_dim,
            0,
        )?;
        // The pseudo tables for the split-KV attention read, over the
        // decode rows only.
        let factor = self.global_split_factor;
        for r in 0..batch {
            let row = &global_rows[r];
            let row_len = i32::try_from(row.len()).context("global pages fit i32")?;
            let last = i32::try_from(global_last[r]).context("global last-page len fits i32")?;
            let kv_len = (row.len() - 1) * global_page + global_last[r];
            for _ in 0..factor {
                pseudo_pages.extend_from_slice(row);
                pseudo_indptr.push(pseudo_indptr.last().unwrap() + row_len);
                pseudo_last.push(last);
                pseudo_kv_lens.push(kv_len);
            }
        }
        let global_csr = ops::build_split_kv_csr(
            GLOBAL_SPLIT_CHUNK_TOKENS,
            global_split.cap,
            pseudo_kv_lens,
            factor * batch,
        )?;
        upload_prefix(ctx, mix_positions, &mix_rows.positions)?;
        upload_prefix(ctx, mix_local_pages, &mix_rows.local_pages)?;
        upload_prefix(ctx, mix_local_origins, &mix_rows.local_origins)?;
        upload_prefix(ctx, mix_global_pages, &mix_rows.global_pages)?;
        upload_prefix(ctx, mix_global_origins, &mix_rows.global_origins)?;
        upload_prefix(ctx, &mut global_tables.pseudo_pages, pseudo_pages)?;
        upload_prefix(ctx, &mut global_tables.pseudo_indptr, pseudo_indptr)?;
        upload_prefix(ctx, &mut global_tables.pseudo_last, pseudo_last)?;
        upload_prefix(
            ctx,
            &mut global_split.request_indices_d,
            &global_csr.request_indices,
        )?;
        upload_prefix(
            ctx,
            &mut global_split.kv_tile_indices_d,
            &global_csr.kv_tile_indices,
        )?;
        upload_prefix(ctx, &mut global_split.o_indptr_d, &global_csr.o_indptr)?;
        upload_prefix(
            ctx,
            &mut global_split.valid_mask_d,
            &global_csr.block_valid_mask,
        )?;

        for (_, prompt) in prefills.iter() {
            ids_host.extend_from_slice(prompt);
        }
        ids_host.extend_from_slice(decode_tokens);
        let ids = ctx
            .stream
            .clone_htod(ids_host)
            .map_err(|e| anyhow::anyhow!("mixed step ids H2D failed: {e}"))?;
        let mut tower = TowerScratch::new(ctx, &self.local_geom, &self.global_geom, rows)?;
        tower.open(rows)?;
        let src = self.run_tower(
            ctx,
            &mut tower,
            &ids,
            rows,
            &local_plan,
            PrepRef::Mixed {
                prefill_len,
                mix_positions: &*mix_positions,
                mix_local_pages: &*mix_local_pages,
                mix_local_origins: &*mix_local_origins,
                mix_global_pages: &*mix_global_pages,
                mix_global_origins: &*mix_global_origins,
                mix_indptr: &*mix_indptr,
                global_tables: &*global_tables,
                global_prefill_plan: &global_prefill_plan,
            },
            Some(global_split),
        )?;
        // Compact the sampled rows into the free ping-pong slot — each
        // segment's last row, then the decode suffix as one range — and run
        // the batch + prompts rows through the LM head.
        let (x, staging) = hidden_pair(&mut tower.hidden, src);
        staging.seq_len = batch + prompts;
        for (j, seg) in segs.iter().enumerate() {
            ops::copy_hidden_token_range_into(
                ctx,
                x,
                seg.row_offset + seg.rows - 1,
                staging,
                j,
                1,
            )?;
        }
        ops::copy_hidden_token_range_into(ctx, x, prefill_len, staging, prompts, batch)?;
        logits_tail_into(
            ctx,
            &self.weights,
            staging,
            self.local_geom.rms_norm_eps,
            self.final_logit_softcapping,
            head_normed,
            logits,
        )?;
        logits.seq_len = batch + prompts;
        // Append-then-attend, per request, the same way both pure steps
        // settle their frontiers.
        for (kv, prompt) in prefills.iter_mut() {
            self.advance_local(kv, prompt.len())?;
            kv.global.advance(prompt.len());
        }
        for kv in decode_kvs.iter_mut() {
            self.advance_local(kv, 1)?;
            kv.global.advance(1);
        }
        Ok(logits)
    }

    /// Host-side preparation a decode step and its capture share: checks,
    /// the bucket, the arena open, the plan refill and the metadata uploads.
    fn prepare_decode_step(
        &self,
        ctx: &DeviceContext,
        arena: &mut StepArena,
        kvs: &[&mut GemmaKv],
        tokens: &[u32],
    ) -> Result<usize> {
        let batch = kvs.len();
        anyhow::ensure!(batch > 0, "a decode batch needs at least one request");
        anyhow::ensure!(
            tokens.len() == batch,
            "decode batch has {batch} requests but {} tokens",
            tokens.len()
        );
        anyhow::ensure!(
            batch <= arena.max_rows,
            "decode batch of {batch} exceeds the arena's {} row ceiling",
            arena.max_rows
        );
        self.check_stream(ctx)?;
        validate_tokens(&self.weights, self.local_geom.hidden_size, tokens)?;

        // A step computes at its power-of-two bucket whether or not graphs
        // are on: padding is part of the numeric contract, which is what
        // keeps a captured replay and the eager escape hatch the same
        // arithmetic at every batch size.
        let padded = batch.next_power_of_two().max(arena.min_bucket);
        arena.open(ctx, padded)?;
        self.plan_decode_batch(ctx, arena, kvs, padded)?;
        arena.host.ids.extend_from_slice(tokens);
        arena.host.ids.resize(padded, 0);
        upload_prefix(ctx, &mut arena.ids, &arena.host.ids)?;
        Ok(padded)
    }

    /// Capture every power-of-two decode graph before serving, then
    /// synchronize, so capture cost and any capture error land here rather
    /// than on the first requests. Each bucket warms one eager pass first —
    /// forcing lazy CUDA and cuBLAS initialization outside capture — records
    /// the body without executing it, then drives the same step through the
    /// serving path, which replays the fresh graph and advances the dummy.
    pub(crate) fn precapture_decode_graphs(
        &self,
        ctx: &DeviceContext,
        arena: &mut StepArena,
    ) -> Result<()> {
        if !arena.graph_enabled {
            return Ok(());
        }
        let mut kv = self.alloc_kv();
        admit_tokens(&self.local_pool, &self.global_pool, &mut kv, 1)?;
        self.step(ctx, &mut kv, &[0])?;
        let mut bucket = 1usize;
        while bucket <= arena.max_rows {
            arena.min_bucket = bucket;
            admit_tokens(&self.local_pool, &self.global_pool, &mut kv, 1)?;
            {
                let mut kvs: [&mut GemmaKv; 1] = [&mut kv];
                let padded = self.prepare_decode_step(ctx, arena, &kvs, &[0])?;
                let StepArena {
                    tower,
                    local_plan,
                    global_tables,
                    global_split,
                    local_origins,
                    ids,
                    head_normed,
                    logits,
                    graphs,
                    ..
                } = arena;
                self.decode_gpu_body(
                    ctx,
                    tower,
                    ids,
                    padded,
                    local_plan,
                    global_tables,
                    global_split,
                    local_origins,
                    head_normed,
                    logits,
                )?;
                graphs[bucket_slot(padded)].capture_only(ctx, || {
                    self.decode_gpu_body(
                        ctx,
                        tower,
                        ids,
                        padded,
                        local_plan,
                        global_tables,
                        global_split,
                        local_origins,
                        head_normed,
                        logits,
                    )
                })?;
                self.decode_batch_step(ctx, arena, &mut kvs, &[0])?;
            }
            bucket *= 2;
        }
        arena.min_bucket = 1;
        ctx.sync()
    }

    pub(crate) fn step(
        &self,
        ctx: &DeviceContext,
        kv: &mut GemmaKv,
        tokens: &[u32],
    ) -> Result<HiddenStates> {
        let seq_len = tokens.len();
        anyhow::ensure!(seq_len > 0, "step needs at least one token");
        self.check_stream(ctx)?;
        let weights = &self.weights;
        validate_tokens(weights, self.local_geom.hidden_size, tokens)?;
        let start_pos = kv.local.seq_len();
        self.check_step_bounds(kv, start_pos + seq_len)?;
        let plan = self.plan_step(ctx, kv, start_pos, seq_len)?;
        log::debug!(
            "gemma4 step: start_pos {start_pos} seq_len {seq_len} pages local {} global {}",
            kv.local.held_pages(),
            kv.global.held_pages()
        );

        let mut tower = TowerScratch::new(ctx, &self.local_geom, &self.global_geom, seq_len)?;
        let ids = ctx
            .stream
            .clone_htod(tokens)
            .map_err(|e| anyhow::anyhow!("token ids H2D failed: {e}"))?;
        let src = self.run_tower(
            ctx,
            &mut tower,
            &ids,
            seq_len,
            &plan.local_plan,
            PrepRef::Single {
                start_pos: plan.start_pos,
                local_page_origin: plan.local_page_origin,
                global_plan: &plan.global_plan,
            },
            None,
        )?;
        let hidden = &tower.hidden[src];
        let mut last = HiddenStates::zeros(ctx, hidden.hidden_dim, 1)?;
        ops::copy_hidden_token_range_into(ctx, hidden, seq_len - 1, &mut last, 0, 1)?;
        let logits = logits_tail(
            ctx,
            weights,
            &last,
            self.local_geom.rms_norm_eps,
            self.final_logit_softcapping,
        )?;
        // Append-then-attend: the old window and the new tokens were both
        // resident through the layers above, so only now is the front safe to
        // release. The local move is settled first because it is the only
        // fallible one — the global advance cannot fail, so nothing here can
        // leave one family ahead of the other.
        self.advance_local(kv, seq_len)?;
        kv.global.advance(seq_len);
        Ok(logits)
    }

    /// Whole-prompt prefill left in flight on the active stream — the
    /// overlap-safe form of [`GemmaServe::step`].
    /// Every host-side producer (plan upload, token-id H2D, buffer
    /// allocation) runs on `ctx.stream` before one producer fence;
    /// everything after is kernel work over buffers the returned pass owns,
    /// so the caller may launch it under a stream override and leave it in
    /// flight while decode steps continue on `ctx.stream`. The
    /// append-then-attend window release is deferred to
    /// [`GemmaServe::release_prefill_window`] at join time — a page
    /// released here could be re-allocated to a decoding request's frontier
    /// and written on `ctx.stream` while the in-flight layers still read
    /// it.
    pub(crate) fn prefill_into_logits(
        &self,
        ctx: &DeviceContext,
        kv: &mut GemmaKv,
        tokens: &[u32],
    ) -> Result<PrefillPass> {
        let seq_len = tokens.len();
        anyhow::ensure!(seq_len > 0, "prefill needs at least one token");
        self.check_stream(ctx)?;
        let weights = &self.weights;
        validate_tokens(weights, self.local_geom.hidden_size, tokens)?;
        let start_pos = kv.local.seq_len();
        self.check_step_bounds(kv, start_pos + seq_len)?;
        let plan = self.plan_step(ctx, kv, start_pos, seq_len)?;
        let mut tower = TowerScratch::new(ctx, &self.local_geom, &self.global_geom, seq_len)?;
        let ids = ctx
            .stream
            .clone_htod(tokens)
            .map_err(|e| anyhow::anyhow!("token ids H2D failed: {e}"))?;
        let mut last = HiddenStates::zeros(ctx, self.local_geom.hidden_size, 1)?;
        let mut normed = HiddenStates::zeros(ctx, self.local_geom.hidden_size, 1)?;
        let mut logits = HiddenStates::zeros(ctx, weights.embed_tokens.rows, 1)?;
        fence_producers_before_override(ctx)?;
        let src = self.run_tower(
            ctx,
            &mut tower,
            &ids,
            seq_len,
            &plan.local_plan,
            PrepRef::Single {
                start_pos: plan.start_pos,
                local_page_origin: plan.local_page_origin,
                global_plan: &plan.global_plan,
            },
            None,
        )?;
        let hidden = &tower.hidden[src];
        ops::copy_hidden_token_range_into(ctx, hidden, seq_len - 1, &mut last, 0, 1)?;
        logits_tail_into(
            ctx,
            weights,
            &last,
            self.local_geom.rms_norm_eps,
            self.final_logit_softcapping,
            &mut normed,
            &mut logits,
        )?;
        kv.local.advance(seq_len);
        kv.global.advance(seq_len);
        Ok(PrefillPass {
            logits,
            _tower: tower,
            _plan: plan,
            _ids: ids,
            _last: last,
            _normed: normed,
        })
    }

    /// The deferred append-then-attend release for an overlapped prefill:
    /// call once its completion event has fired, before the request joins
    /// the decode batch.
    pub(crate) fn release_prefill_window(&self, kv: &mut GemmaKv) -> Result<()> {
        self.advance_local(kv, 0)
    }
}

/// Everything an overlapped prefill keeps alive while its kernels are in
/// flight on the lane stream: dropping any of it before the completion
/// event fires would free device memory the pass still reads.
pub(crate) struct PrefillPass {
    pub(crate) logits: HiddenStates,
    _tower: TowerScratch,
    _plan: GemmaStepPlan,
    _ids: CudaSlice<u32>,
    _last: HiddenStates,
    _normed: HiddenStates,
}

#[path = "serve_oracle.rs"]
#[cfg(test)]
pub(crate) mod oracle;
