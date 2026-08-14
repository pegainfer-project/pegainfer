use std::ffi::c_void;
use std::mem::size_of;
use std::time::Duration;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys;
use half::bf16;
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::rope::precompute_rope;
use pegainfer_kernels::ffi;
use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::prefill_attention_paged_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;
use serde::Deserialize;
use serde::Serialize;

const NUM_LAYERS: usize = 1;
pub const NUM_QO_HEADS: usize = 32;
pub const NUM_KV_HEADS: usize = 8;
pub const HEAD_DIM: usize = 128;
pub const PAGE_SIZE: usize = 16;
pub const REPORT_ITERS: u64 = 128;
// Mirror the default Tuned decode path (SPLIT_KV_TUNED_MAX_CHUNKS), not the opt-in
// --batch-invariant Pin width (SPLIT_KV_MAX_CHUNKS_PER_REQUEST).
const DEFAULT_SPLIT_KV_CHUNK_TOKENS: usize = crate::batch_decode_buffers::SPLIT_KV_CHUNK_TOKENS;
const DEFAULT_SPLIT_KV_MAX_CHUNKS_PER_REQUEST: usize =
    crate::batch_decode_buffers::SPLIT_KV_TUNED_MAX_CHUNKS;
const MEMORY_TRANSFERS_PER_CLOCK: f64 = 2.0;
const CACHE_CLEAR_L2_MULTIPLIER: usize = 2;
const CACHE_CLEAR_MIN_BYTES: usize = 128 * 1024 * 1024;

pub use crate::batch_decode_buffers::SplitKvCsr;
pub use crate::batch_decode_buffers::build_split_kv_csr;
pub use crate::split_kv::SplitKvConfig;

const DEFAULT_SPLIT_KV_CONFIG: SplitKvConfig = SplitKvConfig::new(
    DEFAULT_SPLIT_KV_CHUNK_TOKENS,
    DEFAULT_SPLIT_KV_MAX_CHUNKS_PER_REQUEST,
);

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum AttentionKernelVariant {
    NonPartition,
    SplitKv(SplitKvConfig),
}

impl AttentionKernelVariant {
    pub fn label(self) -> String {
        match self {
            Self::NonPartition => "non_partition".to_string(),
            Self::SplitKv(config) => config.label(),
        }
    }

    pub fn decode_path(self) -> DecodePath {
        match self {
            Self::NonPartition => DecodePath::NonPartition,
            Self::SplitKv(_) => DecodePath::SplitK,
        }
    }

    fn split_config(self) -> SplitKvConfig {
        match self {
            Self::NonPartition => DEFAULT_SPLIT_KV_CONFIG,
            Self::SplitKv(config) => config,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum DecodePath {
    NonPartition,
    SplitK,
}

impl DecodePath {
    pub fn name(self, split_config: SplitKvConfig) -> String {
        match self {
            Self::NonPartition => "non_partition".to_string(),
            Self::SplitK => split_config.label(),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct AttentionKernelShape {
    pub batch_size: usize,
    pub kv_len: usize,
}

impl AttentionKernelShape {
    pub const fn new(batch_size: usize, kv_len: usize) -> Self {
        Self { batch_size, kv_len }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionKernelSpec {
    pub shape: AttentionKernelShape,
    pub variant: AttentionKernelVariant,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct PrefillAttentionShape {
    pub batch_size: usize,
    pub seq_len: usize,
}

impl PrefillAttentionShape {
    pub const fn new(batch_size: usize, seq_len: usize) -> Self {
        Self {
            batch_size,
            seq_len,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct PrefillAttentionSpec {
    pub shape: PrefillAttentionShape,
    pub variant: PrefillAttentionVariant,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PrefillAttentionVariant {
    Default,
    CtaTileQ(usize),
}

impl PrefillAttentionVariant {
    pub fn label(self) -> String {
        match self {
            Self::Default => "default".to_string(),
            Self::CtaTileQ(tile_q) => format!("cta_q{tile_q}"),
        }
    }

    pub fn range_label(self) -> String {
        match self {
            Self::Default => "auto".to_string(),
            Self::CtaTileQ(tile_q) => format!("q{tile_q}"),
        }
    }

    fn cta_tile_q_override(self) -> i32 {
        match self {
            Self::Default => 0,
            Self::CtaTileQ(tile_q) => tile_q as i32,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum PrefillStage {
    Full,
    QkNormRope,
    KvScatter,
    AttentionCore,
}

impl PrefillStage {
    pub fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::QkNormRope => "qk_norm_rope",
            Self::KvScatter => "kv_scatter",
            Self::AttentionCore => "attention_core",
        }
    }

    pub fn range_label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::QkNormRope => "qk",
            Self::KvScatter => "kv",
            Self::AttentionCore => "attn",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct DevicePeakBandwidth {
    pub memory_clock_khz: i32,
    pub memory_bus_width_bits: i32,
    peak_bytes_per_sec: f64,
}

impl DevicePeakBandwidth {
    pub fn query(ctx: &DeviceContext) -> Result<Self> {
        let memory_clock_khz = ctx
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
            .map_err(|e| anyhow!("failed to query memory clock: {e}"))?;
        let memory_bus_width_bits = ctx
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
            .map_err(|e| anyhow!("failed to query memory bus width: {e}"))?;
        let peak_bytes_per_sec = f64::from(memory_clock_khz)
            * 1_000.0
            * (f64::from(memory_bus_width_bits) / 8.0)
            * MEMORY_TRANSFERS_PER_CLOCK;

        Ok(Self {
            memory_clock_khz,
            memory_bus_width_bits,
            peak_bytes_per_sec,
        })
    }

    pub fn peak_gb_per_sec(&self) -> f64 {
        self.peak_bytes_per_sec / 1.0e9
    }
}

pub struct L2CacheClear {
    a: CudaSlice<bf16>,
    b: CudaSlice<bf16>,
    out: CudaSlice<bf16>,
    len: usize,
}

impl L2CacheClear {
    pub fn new(ctx: &DeviceContext) -> Result<Self> {
        let l2_bytes =
            ctx.ctx
                .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
                .map_err(|e| anyhow!("failed to query L2 cache size: {e}"))? as usize;
        let clear_bytes = cache_clear_bytes(l2_bytes);
        let len = clear_bytes.div_ceil(size_of::<bf16>());

        Ok(Self {
            a: ctx.stream.alloc_zeros(len)?,
            b: ctx.stream.alloc_zeros(len)?,
            out: ctx.stream.alloc_zeros(len)?,
            len,
        })
    }

    pub fn clear(&mut self, ctx: &DeviceContext) -> Result<()> {
        // CUDA's reset-persisting-L2 APIs do not evict normal cache lines, so
        // benchmarks use a large streaming kernel to push prior data out of L2.
        let (a_ptr, _a_guard) = self.a.device_ptr(&ctx.stream);
        let (b_ptr, _b_guard) = self.b.device_ptr(&ctx.stream);
        let (out_ptr, _out_guard) = self.out.device_ptr_mut(&ctx.stream);
        let result = unsafe {
            ffi::add_cuda(
                a_ptr as *const ffi::Half,
                b_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                self.len as i32,
                ctx.stream.cu_stream(),
            )
        };
        result.result()?;
        Ok(())
    }
}

pub fn cache_clear_bytes(l2_bytes: usize) -> usize {
    (l2_bytes * CACHE_CLEAR_L2_MULTIPLIER).max(CACHE_CLEAR_MIN_BYTES)
}

pub struct AttentionDecodeCase {
    pub ctx: DeviceContext,
    layout: PagedKvLayout,
    q: HiddenStates,
    output: HiddenStates,
    kv_buffer: CudaSlice<bf16>,
    page_indices_d: CudaSlice<i32>,
    page_indptr_d: CudaSlice<i32>,
    last_page_len_d: CudaSlice<i32>,
    request_indices_d: CudaSlice<i32>,
    kv_tile_indices_d: CudaSlice<i32>,
    kv_chunk_size_d: CudaSlice<i32>,
    split_request_indices_d: CudaSlice<i32>,
    split_kv_tile_indices_d: CudaSlice<i32>,
    split_kv_chunk_size_d: CudaSlice<i32>,
    split_o_indptr_d: CudaSlice<i32>,
    split_block_valid_mask_d: CudaSlice<u8>,
    split_tmp_v: CudaSlice<bf16>,
    split_tmp_s: CudaSlice<f32>,
    split_padded_slots: usize,
    split_config: SplitKvConfig,
    start: CudaEvent,
    end: CudaEvent,
    batch_size: usize,
    kv_len: usize,
}

impl AttentionDecodeCase {
    pub fn for_spec(spec: AttentionKernelSpec) -> Result<Self> {
        Self::new_with_split_config(
            spec.shape.batch_size,
            spec.shape.kv_len,
            spec.variant.split_config(),
        )
    }

    fn new_with_split_config(
        batch_size: usize,
        kv_len: usize,
        split_config: SplitKvConfig,
    ) -> Result<Self> {
        let ctx = DeviceContext::new()?;
        let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE);
        let q_dim = NUM_QO_HEADS * HEAD_DIM;
        let pages_per_request = kv_len.div_ceil(PAGE_SIZE);
        let total_pages = pages_per_request * batch_size;

        let q_host = patterned_bf16(q_dim * batch_size, 0.01);
        let kv_host = patterned_bf16(total_pages * layout.page_stride, 0.001);

        let q = HiddenStates {
            data: ctx.stream.clone_htod(&q_host)?,
            hidden_dim: q_dim,
            seq_len: batch_size,
        };
        let output = HiddenStates::zeros(&ctx, q_dim, batch_size)?;
        let kv_buffer = ctx.stream.clone_htod(&kv_host)?;

        let mut page_indices = Vec::with_capacity(total_pages);
        let mut page_indptr = Vec::with_capacity(batch_size + 1);
        page_indptr.push(0);
        for request_idx in 0..batch_size {
            for page_offset in 0..pages_per_request {
                page_indices.push((request_idx * pages_per_request + page_offset) as i32);
            }
            page_indptr.push(page_indices.len() as i32);
        }

        let last_page_len = match kv_len % PAGE_SIZE {
            0 => PAGE_SIZE,
            rem => rem,
        };
        let last_page_lens = vec![last_page_len as i32; batch_size];
        let request_indices: Vec<i32> = (0..batch_size as i32).collect();
        let kv_tile_indices = vec![0i32; batch_size];
        let kv_chunk_sizes = vec![kv_len as i32; batch_size];
        let split_chunk_size = split_config.actual_chunk_size(kv_len);
        let split_padded_slots = batch_size * split_config.max_chunks_per_request;
        let split_csr = build_split_kv_csr(
            split_chunk_size,
            split_config.max_chunks_per_request,
            &vec![kv_len; batch_size],
            batch_size,
        )?;
        let split_kv_chunk_sizes = [split_chunk_size as i32];

        let start = ctx
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = ctx
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;

        let page_indices_d = ctx.stream.clone_htod(&page_indices)?;
        let page_indptr_d = ctx.stream.clone_htod(&page_indptr)?;
        let last_page_len_d = ctx.stream.clone_htod(&last_page_lens)?;
        let request_indices_d = ctx.stream.clone_htod(&request_indices)?;
        let kv_tile_indices_d = ctx.stream.clone_htod(&kv_tile_indices)?;
        let kv_chunk_size_d = ctx.stream.clone_htod(&kv_chunk_sizes)?;
        let split_request_indices_d = ctx.stream.clone_htod(&split_csr.request_indices)?;
        let split_kv_tile_indices_d = ctx.stream.clone_htod(&split_csr.kv_tile_indices)?;
        let split_kv_chunk_size_d = ctx.stream.clone_htod(&split_kv_chunk_sizes)?;
        let split_o_indptr_d = ctx.stream.clone_htod(&split_csr.o_indptr)?;
        let split_block_valid_mask_d = ctx.stream.clone_htod(&split_csr.block_valid_mask)?;
        let split_tmp_v = ctx.stream.alloc_zeros(split_padded_slots * q_dim)?;
        let split_tmp_s = ctx.stream.alloc_zeros(split_padded_slots * NUM_QO_HEADS)?;

        let case = Self {
            ctx,
            layout,
            q,
            output,
            kv_buffer,
            page_indices_d,
            page_indptr_d,
            last_page_len_d,
            request_indices_d,
            kv_tile_indices_d,
            kv_chunk_size_d,
            split_request_indices_d,
            split_kv_tile_indices_d,
            split_kv_chunk_size_d,
            split_o_indptr_d,
            split_block_valid_mask_d,
            split_tmp_v,
            split_tmp_s,
            split_padded_slots,
            split_config,
            start,
            end,
            batch_size,
            kv_len,
        };
        case.ctx.sync()?;
        Ok(case)
    }

    pub fn shape(&self) -> AttentionKernelShape {
        AttentionKernelShape::new(self.batch_size, self.kv_len)
    }

    pub fn split_config(&self) -> SplitKvConfig {
        self.split_config
    }

    pub fn cu_context_ptr(&self) -> *mut c_void {
        self.ctx.ctx.cu_ctx().cast::<c_void>()
    }

    pub fn launch_once(&mut self, path: DecodePath) -> Result<()> {
        self.launch_inner(path)?;
        Ok(())
    }

    fn launch_inner(&mut self, path: DecodePath) -> Result<i32> {
        let (q_ptr, _q_guard) = self.q.data.device_ptr(&self.ctx.stream);
        let (out_ptr, _out_guard) = self.output.data.device_ptr_mut(&self.ctx.stream);
        let (kv_ptr, _kv_guard) = self.kv_buffer.device_ptr(&self.ctx.stream);
        let (page_indices_ptr, _page_indices_guard) =
            self.page_indices_d.device_ptr(&self.ctx.stream);
        let (page_indptr_ptr, _page_indptr_guard) = self.page_indptr_d.device_ptr(&self.ctx.stream);
        let (last_page_len_ptr, _last_page_len_guard) =
            self.last_page_len_d.device_ptr(&self.ctx.stream);
        let (request_indices_ptr, _request_indices_guard) =
            self.request_indices_d.device_ptr(&self.ctx.stream);
        let (kv_tile_indices_ptr, _kv_tile_indices_guard) =
            self.kv_tile_indices_d.device_ptr(&self.ctx.stream);
        let (kv_chunk_size_ptr, _kv_chunk_size_guard) =
            self.kv_chunk_size_d.device_ptr(&self.ctx.stream);
        let (split_request_indices_ptr, _split_request_indices_guard) =
            self.split_request_indices_d.device_ptr(&self.ctx.stream);
        let (split_kv_tile_indices_ptr, _split_kv_tile_indices_guard) =
            self.split_kv_tile_indices_d.device_ptr(&self.ctx.stream);
        let (split_kv_chunk_size_ptr, _split_kv_chunk_size_guard) =
            self.split_kv_chunk_size_d.device_ptr(&self.ctx.stream);
        let (split_o_indptr_ptr, _split_o_indptr_guard) =
            self.split_o_indptr_d.device_ptr(&self.ctx.stream);
        let (split_block_valid_mask_ptr, _split_block_valid_mask_guard) =
            self.split_block_valid_mask_d.device_ptr(&self.ctx.stream);
        let (split_tmp_v_ptr, _split_tmp_v_guard) =
            self.split_tmp_v.device_ptr_mut(&self.ctx.stream);
        let (split_tmp_s_ptr, _split_tmp_s_guard) =
            self.split_tmp_s.device_ptr_mut(&self.ctx.stream);

        let k_offset_elems = 0i64;
        let v_offset_elems = self.layout.kv_block_len as i64;
        let stride_page = self.layout.page_stride as i64;
        let sm_scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let stream = self.ctx.stream.cu_stream();
        let result = match path {
            DecodePath::NonPartition => unsafe {
                ffi::paged_attention_decode_cuda(
                    q_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    kv_ptr as *const ffi::Half,
                    k_offset_elems,
                    v_offset_elems,
                    page_indices_ptr as *const i32,
                    page_indptr_ptr as *const i32,
                    last_page_len_ptr as *const i32,
                    request_indices_ptr as *const i32,
                    kv_tile_indices_ptr as *const i32,
                    kv_chunk_size_ptr as *const i32,
                    NUM_QO_HEADS as i32,
                    NUM_KV_HEADS as i32,
                    HEAD_DIM as i32,
                    PAGE_SIZE as i32,
                    self.batch_size as i32,
                    stride_page,
                    sm_scale,
                    stream,
                )
            },
            DecodePath::SplitK => unsafe {
                ffi::paged_attention_decode_split_kv_cuda(
                    q_ptr as *const ffi::Half,
                    out_ptr as *mut ffi::Half,
                    kv_ptr as *const ffi::Half,
                    k_offset_elems,
                    v_offset_elems,
                    page_indices_ptr as *const i32,
                    page_indptr_ptr as *const i32,
                    last_page_len_ptr as *const i32,
                    split_request_indices_ptr as *const i32,
                    split_kv_tile_indices_ptr as *const i32,
                    split_kv_chunk_size_ptr as *const i32,
                    split_o_indptr_ptr as *const i32,
                    split_block_valid_mask_ptr as *const u8,
                    split_tmp_v_ptr as *mut ffi::Half,
                    split_tmp_s_ptr as *mut f32,
                    NUM_QO_HEADS as i32,
                    NUM_KV_HEADS as i32,
                    HEAD_DIM as i32,
                    PAGE_SIZE as i32,
                    self.batch_size as i32,
                    self.split_padded_slots as i32,
                    stride_page,
                    sm_scale,
                    stream,
                )
            },
        };
        if result != 0 {
            bail!(
                "{} paged attention failed with error {result}{}",
                path.name(self.split_config),
                pegainfer_kernels::ops::ffi_exception_message(result)
            );
        }
        Ok(result)
    }

    pub fn measure_decode_only_cold_l2(
        &mut self,
        criterion_iters: u64,
        path: DecodePath,
        cache_clear: &mut L2CacheClear,
    ) -> Result<Duration> {
        let mut elapsed_ms = 0.0f64;

        for _ in 0..criterion_iters {
            cache_clear.clear(&self.ctx)?;
            self.start.record(&self.ctx.stream)?;
            self.launch_once(path)?;
            self.end.record(&self.ctx.stream)?;
            elapsed_ms += f64::from(self.start.elapsed_ms(&self.end)?);
        }

        Ok(Duration::from_secs_f64(elapsed_ms / 1_000.0))
    }
}

pub struct AttentionPrefillCase {
    pub ctx: DeviceContext,
    layout: PagedKvLayout,
    q: HiddenStates,
    k: HiddenStates,
    v: HiddenStates,
    output: HiddenStates,
    q_norm: DeviceVec,
    k_norm: DeviceVec,
    cos_cache: DeviceVec,
    sin_cache: DeviceVec,
    kv_buffer: CudaSlice<bf16>,
    plan: PrefillPagedPlan,
    start: CudaEvent,
    end: CudaEvent,
    batch_size: usize,
    seq_len: usize,
    variant: PrefillAttentionVariant,
}

impl AttentionPrefillCase {
    pub fn for_spec(spec: PrefillAttentionSpec) -> Result<Self> {
        Self::new(spec.shape.batch_size, spec.shape.seq_len, spec.variant)
    }

    fn new(batch_size: usize, seq_len: usize, variant: PrefillAttentionVariant) -> Result<Self> {
        anyhow::ensure!(
            batch_size > 0,
            "prefill batch_size must be greater than zero"
        );
        anyhow::ensure!(seq_len > 0, "prefill seq_len must be greater than zero");

        let ctx = DeviceContext::new()?;
        let layout = PagedKvLayout::new(NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE);
        let q_dim = NUM_QO_HEADS * HEAD_DIM;
        let kv_dim = NUM_KV_HEADS * HEAD_DIM;
        let pages_per_request = seq_len.div_ceil(PAGE_SIZE);
        let total_pages = pages_per_request * batch_size;

        let q = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&patterned_bf16(q_dim * batch_size * seq_len, 0.01))?,
            hidden_dim: q_dim,
            seq_len: batch_size * seq_len,
        };
        let k = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&patterned_bf16(kv_dim * batch_size * seq_len, 0.001))?,
            hidden_dim: kv_dim,
            seq_len: batch_size * seq_len,
        };
        let v = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&patterned_bf16(kv_dim * batch_size * seq_len, 0.002))?,
            hidden_dim: kv_dim,
            seq_len: batch_size * seq_len,
        };
        let output = HiddenStates::zeros(&ctx, q_dim, batch_size * seq_len)?;
        let q_norm = DeviceVec::from_host(&ctx, &vec![bf16::from_f32(1.0); HEAD_DIM])?;
        let k_norm = DeviceVec::from_host(&ctx, &vec![bf16::from_f32(1.0); HEAD_DIM])?;
        let (cos_cache, sin_cache) = precompute_rope(
            &ctx,
            &RopeTableSpec {
                rotary_dim: HEAD_DIM,
                frequency_dim: HEAD_DIM,
                max_seq_len: seq_len,
                theta: 1e6,
            },
        )?;
        let kv_buffer = ctx
            .stream
            .clone_htod(&patterned_bf16(total_pages * layout.page_stride, 0.001))?;

        let last_page_len = match seq_len % PAGE_SIZE {
            0 => PAGE_SIZE,
            rem => rem,
        };
        let page_indices: Vec<Vec<i32>> = (0..batch_size)
            .map(|request_idx| {
                (0..pages_per_request)
                    .map(|page_offset| (request_idx * pages_per_request + page_offset) as i32)
                    .collect()
            })
            .collect();
        let last_page_lens = vec![last_page_len; batch_size];
        let start_positions = vec![0usize; batch_size];
        let seq_lens = vec![seq_len; batch_size];
        let cta_tile_q_override = variant.cta_tile_q_override();
        let plan = if batch_size == 1 {
            PrefillPagedPlan::new_with_cta_tile_q(
                &ctx,
                &page_indices[0],
                last_page_len,
                0,
                seq_len,
                NUM_QO_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                cta_tile_q_override,
            )?
        } else {
            PrefillPagedPlan::new_batch_with_cta_tile_q(
                &ctx,
                &page_indices,
                &last_page_lens,
                &start_positions,
                &seq_lens,
                NUM_QO_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                cta_tile_q_override,
            )?
        };

        let start = ctx
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = ctx
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;

        let case = Self {
            ctx,
            layout,
            q,
            k,
            v,
            output,
            q_norm,
            k_norm,
            cos_cache,
            sin_cache,
            kv_buffer,
            plan,
            start,
            end,
            batch_size,
            seq_len,
            variant,
        };
        case.ctx.sync()?;
        Ok(case)
    }

    pub fn shape(&self) -> PrefillAttentionShape {
        PrefillAttentionShape::new(self.batch_size, self.seq_len)
    }

    fn total_tokens(&self) -> usize {
        self.batch_size * self.seq_len
    }

    pub fn cu_context_ptr(&self) -> *mut c_void {
        self.ctx.ctx.cu_ctx().cast::<c_void>()
    }

    fn launch_once(&mut self) -> Result<()> {
        prefill_attention_paged_into(
            &self.ctx,
            &mut self.q,
            &mut self.k,
            &self.v,
            &self.q_norm,
            &self.k_norm,
            &self.cos_cache,
            &self.sin_cache,
            &self.kv_buffer,
            &self.layout,
            0,
            &self.plan,
            &mut self.output,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            1.0e-6,
        )
    }

    pub fn prepare_stage(&mut self, stage: PrefillStage) -> Result<()> {
        match stage {
            PrefillStage::Full | PrefillStage::QkNormRope => Ok(()),
            PrefillStage::KvScatter => {
                self.launch_qk_norm_rope();
                Ok(())
            }
            PrefillStage::AttentionCore => {
                self.launch_qk_norm_rope();
                self.launch_kv_scatter()
            }
        }
    }

    pub fn pre_measure_stage(&mut self, stage: PrefillStage) -> Result<()> {
        self.prepare_stage(stage)?;
        self.launch_stage(stage)?;
        self.ctx.sync()
    }

    pub fn launch_stage(&mut self, stage: PrefillStage) -> Result<()> {
        match stage {
            PrefillStage::Full => self.launch_once(),
            PrefillStage::QkNormRope => {
                self.launch_qk_norm_rope();
                Ok(())
            }
            PrefillStage::KvScatter => self.launch_kv_scatter(),
            PrefillStage::AttentionCore => self.launch_attention_core(),
        }
    }

    fn launch_qk_norm_rope(&mut self) {
        let total_tokens = self.total_tokens();
        let (q_ptr, _q_guard) = self.q.data.device_ptr_mut(&self.ctx.stream);
        let (k_ptr, _k_guard) = self.k.data.device_ptr_mut(&self.ctx.stream);
        let (qn_ptr, _qn_guard) = self.q_norm.data.device_ptr(&self.ctx.stream);
        let (kn_ptr, _kn_guard) = self.k_norm.data.device_ptr(&self.ctx.stream);
        let (cos_ptr, _cos_guard) = self.cos_cache.data.device_ptr(&self.ctx.stream);
        let (sin_ptr, _sin_guard) = self.sin_cache.data.device_ptr(&self.ctx.stream);

        let (positions_ptr, _positions_guard) =
            self.plan.positions_d().device_ptr(&self.ctx.stream);
        unsafe {
            ffi::qk_norm_rope_batched_decode_cuda(
                q_ptr as *mut ffi::Half,
                k_ptr as *mut ffi::Half,
                qn_ptr as *const ffi::Half,
                kn_ptr as *const ffi::Half,
                cos_ptr as *const ffi::Half,
                sin_ptr as *const ffi::Half,
                positions_ptr as *const i32,
                NUM_QO_HEADS as i32,
                NUM_KV_HEADS as i32,
                HEAD_DIM as i32,
                total_tokens as i32,
                1.0e-6,
                (self.cos_cache.data.len() / HEAD_DIM) as i32,
                self.ctx.stream.cu_stream(),
            );
        }
    }

    fn launch_kv_scatter(&mut self) -> Result<()> {
        let (kv_ptr, _kv_guard) = self.kv_buffer.device_ptr(&self.ctx.stream);
        let (k_ptr, _k_guard) = self.k.data.device_ptr(&self.ctx.stream);
        let (v_ptr, _v_guard) = self.v.data.device_ptr(&self.ctx.stream);
        let (page_indices_ptr, _page_indices_guard) =
            self.plan.page_indices_d().device_ptr(&self.ctx.stream);
        let (page_indptr_ptr, _page_indptr_guard) =
            self.plan.page_indptr_d().device_ptr(&self.ctx.stream);
        let (last_page_len_ptr, _last_page_len_guard) =
            self.plan.last_page_len_d().device_ptr(&self.ctx.stream);
        let (batch_indices_ptr, _batch_indices_guard) =
            self.plan.batch_indices_d().device_ptr(&self.ctx.stream);
        let (positions_ptr, _positions_guard) =
            self.plan.positions_d().device_ptr(&self.ctx.stream);

        let kv_dim = NUM_KV_HEADS * HEAD_DIM;
        let k_offset = 0i64;
        let v_offset = self.layout.kv_block_len as i64;
        let stride_page = self.layout.page_stride as i64;
        let result = unsafe {
            ffi::paged_kv_scatter_cuda(
                kv_ptr as *const ffi::Half,
                k_offset,
                v_offset,
                page_indices_ptr as *const i32,
                page_indptr_ptr as *const i32,
                last_page_len_ptr as *const i32,
                k_ptr as *const ffi::Half,
                v_ptr as *const ffi::Half,
                batch_indices_ptr as *const i32,
                positions_ptr as *const i32,
                self.total_tokens() as i32,
                NUM_KV_HEADS as i32,
                HEAD_DIM as i32,
                PAGE_SIZE as i32,
                stride_page,
                kv_dim as i64,
                HEAD_DIM as i64,
                self.ctx.stream.cu_stream(),
            )
        };
        if result != 0 {
            bail!(
                "segmented paged_kv_scatter_cuda failed with error {result}{}",
                pegainfer_kernels::ops::ffi_exception_message(result)
            );
        }
        Ok(())
    }

    fn launch_attention_core(&mut self) -> Result<()> {
        let total_tokens = self.total_tokens();
        let (q_ptr, _q_guard) = self.q.data.device_ptr(&self.ctx.stream);
        let (out_ptr, _out_guard) = self.output.data.device_ptr_mut(&self.ctx.stream);
        let (kv_ptr, _kv_guard) = self.kv_buffer.device_ptr(&self.ctx.stream);
        let (page_indices_ptr, _page_indices_guard) =
            self.plan.page_indices_d().device_ptr(&self.ctx.stream);
        let (page_indptr_ptr, _page_indptr_guard) =
            self.plan.page_indptr_d().device_ptr(&self.ctx.stream);
        let (last_page_len_ptr, _last_page_len_guard) =
            self.plan.last_page_len_d().device_ptr(&self.ctx.stream);
        let (q_indptr_ptr, _q_indptr_guard) = self.plan.q_indptr_d().device_ptr(&self.ctx.stream);
        let (request_indices_ptr, _request_indices_guard) =
            self.plan.request_indices_d().device_ptr(&self.ctx.stream);
        let (qo_tile_indices_ptr, _qo_tile_indices_guard) =
            self.plan.qo_tile_indices_d().device_ptr(&self.ctx.stream);
        let (kv_tile_indices_ptr, _kv_tile_indices_guard) =
            self.plan.kv_tile_indices_d().device_ptr(&self.ctx.stream);
        let (kv_chunk_size_ptr, _kv_chunk_size_guard) =
            self.plan.kv_chunk_size_d().device_ptr(&self.ctx.stream);
        let (total_num_rows_ptr, _total_num_rows_guard) =
            self.plan.total_num_rows_d().device_ptr(&self.ctx.stream);

        let k_offset = 0i64;
        let v_offset = self.layout.kv_block_len as i64;
        let stride_page = self.layout.page_stride as i64;
        let sm_scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let result = unsafe {
            ffi::batch_prefill_paged_cuda_with_cta_tile_q(
                q_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                kv_ptr as *const ffi::Half,
                k_offset,
                v_offset,
                page_indices_ptr as *const i32,
                page_indptr_ptr as *const i32,
                last_page_len_ptr as *const i32,
                q_indptr_ptr as *const i32,
                request_indices_ptr as *const i32,
                qo_tile_indices_ptr as *const i32,
                kv_tile_indices_ptr as *const i32,
                kv_chunk_size_ptr as *const i32,
                total_num_rows_ptr as *const u32,
                NUM_QO_HEADS as i32,
                NUM_KV_HEADS as i32,
                HEAD_DIM as i32,
                PAGE_SIZE as i32,
                total_tokens as i32,
                self.plan.batch_size(),
                self.plan.num_tiles(),
                stride_page,
                sm_scale,
                self.variant.cta_tile_q_override(),
                self.ctx.stream.cu_stream(),
            )
        };
        if result != 0 {
            bail!(
                "segmented batch_prefill_paged_cuda failed with error {result}{}",
                pegainfer_kernels::ops::ffi_exception_message(result)
            );
        }
        Ok(())
    }

    pub fn measure_stage_cold_l2(
        &mut self,
        criterion_iters: u64,
        stage: PrefillStage,
        cache_clear: &mut L2CacheClear,
    ) -> Result<Duration> {
        let mut elapsed_ms = 0.0f64;

        for _ in 0..criterion_iters {
            self.prepare_stage(stage)?;
            cache_clear.clear(&self.ctx)?;
            self.start.record(&self.ctx.stream)?;
            self.launch_stage(stage)?;
            self.end.record(&self.ctx.stream)?;
            elapsed_ms += f64::from(self.start.elapsed_ms(&self.end)?);
        }

        Ok(Duration::from_secs_f64(elapsed_ms / 1_000.0))
    }
}

pub struct SinglePrefillCase {
    pub ctx: DeviceContext,
    q: HiddenStates,
    output: HiddenStates,
    k_cache: CudaSlice<bf16>,
    v_cache: CudaSlice<bf16>,
    start: CudaEvent,
    end: CudaEvent,
    seq_len: usize,
}

impl SinglePrefillCase {
    pub fn for_spec(spec: PrefillAttentionSpec) -> Result<Self> {
        anyhow::ensure!(
            spec.shape.batch_size == 1,
            "single prefill bench only supports batch_size=1"
        );
        Self::new(spec.shape.seq_len)
    }

    fn new(seq_len: usize) -> Result<Self> {
        anyhow::ensure!(
            seq_len > 0,
            "single prefill seq_len must be greater than zero"
        );
        let ctx = DeviceContext::new()?;
        let q_dim = NUM_QO_HEADS * HEAD_DIM;
        let kv_dim = NUM_KV_HEADS * HEAD_DIM;
        let q = HiddenStates {
            data: ctx
                .stream
                .clone_htod(&patterned_bf16(q_dim * seq_len, 0.01))?,
            hidden_dim: q_dim,
            seq_len,
        };
        let output = HiddenStates::zeros(&ctx, q_dim, seq_len)?;
        let k_cache = ctx
            .stream
            .clone_htod(&patterned_bf16(kv_dim * seq_len, 0.001))?;
        let v_cache = ctx
            .stream
            .clone_htod(&patterned_bf16(kv_dim * seq_len, 0.002))?;
        let start = ctx
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = ctx
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let case = Self {
            ctx,
            q,
            output,
            k_cache,
            v_cache,
            start,
            end,
            seq_len,
        };
        case.ctx.sync()?;
        Ok(case)
    }

    pub fn shape(&self) -> PrefillAttentionShape {
        PrefillAttentionShape::new(1, self.seq_len)
    }

    pub fn cu_context_ptr(&self) -> *mut c_void {
        self.ctx.ctx.cu_ctx().cast::<c_void>()
    }

    pub fn pre_measure(&mut self) -> Result<()> {
        self.launch_once()?;
        self.ctx.sync()
    }

    pub fn launch_once(&mut self) -> Result<()> {
        let (q_ptr, _q_guard) = self.q.data.device_ptr(&self.ctx.stream);
        let (out_ptr, _out_guard) = self.output.data.device_ptr_mut(&self.ctx.stream);
        let (k_ptr, _k_guard) = self.k_cache.device_ptr(&self.ctx.stream);
        let (v_ptr, _v_guard) = self.v_cache.device_ptr(&self.ctx.stream);
        let sm_scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let result = unsafe {
            ffi::single_prefill_cuda(
                q_ptr as *const ffi::Half,
                out_ptr as *mut ffi::Half,
                k_ptr as *const ffi::Half,
                v_ptr as *const ffi::Half,
                NUM_QO_HEADS as i32,
                NUM_KV_HEADS as i32,
                HEAD_DIM as i32,
                self.seq_len as i32,
                self.seq_len as i32,
                self.seq_len as i32,
                sm_scale,
                self.ctx.stream.cu_stream(),
            )
        };
        if result != 0 {
            bail!(
                "single_prefill_cuda failed with error {result}{}",
                pegainfer_kernels::ops::ffi_exception_message(result)
            );
        }
        Ok(())
    }

    pub fn measure_cold_l2(
        &mut self,
        criterion_iters: u64,
        cache_clear: &mut L2CacheClear,
    ) -> Result<Duration> {
        let mut elapsed_ms = 0.0f64;

        for _ in 0..criterion_iters {
            cache_clear.clear(&self.ctx)?;
            self.start.record(&self.ctx.stream)?;
            self.launch_once()?;
            self.end.record(&self.ctx.stream)?;
            elapsed_ms += f64::from(self.start.elapsed_ms(&self.end)?);
        }

        Ok(Duration::from_secs_f64(elapsed_ms / 1_000.0))
    }
}

/// Qwen3-4B dense-op dimensions. The dense benches are weight-free (synthetic
/// buffers at production shapes), so the model facts live here as constants —
/// same convention as the attention constants above.
pub const HIDDEN_SIZE: usize = 2560;
pub const INTERMEDIATE_SIZE: usize = 9728;
pub const VOCAB_SIZE: usize = 151_936;
const Q_DIM: usize = NUM_QO_HEADS * HEAD_DIM;
const KV_DIM: usize = NUM_KV_HEADS * HEAD_DIM;
/// Position span for the decode qk-norm-rope bench: mid-context decode is the
/// common case, and the cache read is position-indexed, so the span only has
/// to be large enough that positions don't all hit one cache line.
const DENSE_ROPE_CACHE_TOKENS: usize = 8192;
/// Model fact (config.json `rms_norm_eps`), mirrored here like the head
/// counts so the weight-free benches launch the production epsilon.
const RMS_NORM_EPS: f32 = 1.0e-6;
/// Device-memory cap for the `gemm_lt_tune` weight-rotation copies of a
/// projection-GEMM dense case; the actual copy count is derived from the L2
/// sweep size so the tuner stays DRAM-cold, and this cap only protects
/// small-VRAM cards from the lm_head shape.
const TUNE_ROTATION_BUDGET_BYTES: usize = 2 * (1 << 30);

/// The projection GEMM (out_dim, in_dim) shapes production launches — the same
/// set `decode_projection_pin_shapes` warms for the Pin policy. Gate and up
/// share a shape, so one variant covers both.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GemmProjection {
    QProj,
    KvProj,
    OProj,
    GateUpHalf,
    DownProj,
    LmHead,
}

impl GemmProjection {
    pub fn parse(raw: &str) -> Option<Self> {
        Some(match raw {
            "q_proj" => Self::QProj,
            "kv_proj" => Self::KvProj,
            "o_proj" => Self::OProj,
            "gate_up_half" => Self::GateUpHalf,
            "down_proj" => Self::DownProj,
            "lm_head" => Self::LmHead,
            _ => return None,
        })
    }

    fn label(self) -> &'static str {
        match self {
            Self::QProj => "q_proj",
            Self::KvProj => "kv_proj",
            Self::OProj => "o_proj",
            Self::GateUpHalf => "gate_up_half",
            Self::DownProj => "down_proj",
            Self::LmHead => "lm_head",
        }
    }

    /// (out_dim, in_dim) — cuBLAS N is the token/row count of the step.
    pub const fn out_in(self) -> (usize, usize) {
        match self {
            Self::QProj => (Q_DIM, HIDDEN_SIZE),
            Self::KvProj => (KV_DIM, HIDDEN_SIZE),
            Self::OProj => (HIDDEN_SIZE, Q_DIM),
            Self::GateUpHalf => (INTERMEDIATE_SIZE, HIDDEN_SIZE),
            Self::DownProj => (HIDDEN_SIZE, INTERMEDIATE_SIZE),
            Self::LmHead => (VOCAB_SIZE, HIDDEN_SIZE),
        }
    }
}

/// One dense (non-attention) forward op at production shape. `rows` is the
/// step's token/row count: decode batch size, or prefill token count.
#[derive(Clone, Copy, Debug)]
pub enum DenseKernelKind {
    ProjectionGemm(GemmProjection),
    RmsNorm,
    FusedAddRmsNorm,
    QkNormRopeDecode,
    SiluMul,
    Embedding,
    Sampling { greedy: bool },
}

impl DenseKernelKind {
    pub fn label(self) -> String {
        match self {
            Self::ProjectionGemm(projection) => projection.label().to_string(),
            Self::Sampling { greedy: true } => "argmax".to_string(),
            Self::Sampling { greedy: false } => "sampling".to_string(),
            Self::RmsNorm
            | Self::FusedAddRmsNorm
            | Self::QkNormRopeDecode
            | Self::SiluMul
            | Self::Embedding => "default".to_string(),
        }
    }
}

/// The buffers a dense case owns, one variant per kind — which buffers exist
/// for which op is a type-level fact, not a runtime assertion. One instance
/// per case, never stored in collections, so the variant size spread is
/// irrelevant and boxing the large ones would only add indirection.
#[allow(clippy::large_enum_variant)]
enum DenseBuffers {
    Gemm {
        weight: DeviceMatrix,
        x: HiddenStates,
        out: HiddenStates,
    },
    Norm {
        weight: DeviceVec,
        x: HiddenStates,
        out: HiddenStates,
    },
    FusedAddNorm {
        weight: DeviceVec,
        hidden: HiddenStates,
        residual: HiddenStates,
        out: HiddenStates,
    },
    QkRope {
        q: HiddenStates,
        k: HiddenStates,
        q_norm: DeviceVec,
        k_norm: DeviceVec,
        cos_cache: DeviceVec,
        sin_cache: DeviceVec,
        positions: CudaSlice<i32>,
    },
    SiluMul {
        gate: HiddenStates,
        up: HiddenStates,
        out: HiddenStates,
    },
    Embedding {
        table: DeviceMatrix,
        token_ids: CudaSlice<u32>,
        out: HiddenStates,
    },
    Sampling {
        logits: HiddenStates,
        scratch: pegainfer_sample::SampleScratch,
        params: Vec<pegainfer_frontend::sampler::SamplingParams>,
        seed: u64,
    },
}

/// Bench harness for the dense forward ops, mirroring the attention cases:
/// synthetic buffers at production shapes, one launch per measured iteration,
/// cold L2 via the streaming sweep. Launches go through the same
/// `pegainfer_kernels::ops` entry points as `BatchDecodeDag` / the prefill
/// path, so cuBLAS algo selection matches production steady state after the
/// pre-measure launch.
pub struct DenseCase {
    pub ctx: DeviceContext,
    buffers: DenseBuffers,
    start: CudaEvent,
    end: CudaEvent,
}

fn zeros_matrix(ctx: &DeviceContext, rows: usize, cols: usize) -> Result<DeviceMatrix> {
    Ok(DeviceMatrix {
        data: ctx.stream.alloc_zeros(rows * cols)?,
        rows,
        cols,
    })
}

fn ones_vec(ctx: &DeviceContext, len: usize) -> Result<DeviceVec> {
    DeviceVec::from_host(ctx, &vec![bf16::ONE; len])
}

/// Sampling-case logits: production distributions are sharply peaked, and the
/// FlashInfer rejection sampler's round count depends on that peakedness — a
/// flat synthetic vocabulary would overstate its cost. Each row gets a few
/// dominant logits (top-1 mass ~0.5 after softmax) over a low-noise floor, at
/// row-varying positions.
fn peaked_logits(ctx: &DeviceContext, rows: usize) -> Result<HiddenStates> {
    let mut host = patterned_bf16(VOCAB_SIZE * rows, 0.001);
    for row in 0..rows {
        for peak in 0..8 {
            let token = (row * 48_271 + peak * 15_485_863) % VOCAB_SIZE;
            host[row * VOCAB_SIZE + token] = bf16::from_f32(10.0 - peak as f32);
        }
    }
    Ok(HiddenStates {
        data: ctx.stream.clone_htod(&host)?,
        hidden_dim: VOCAB_SIZE,
        seq_len: rows,
    })
}

/// Build the projection weight and tune its cuBLASLt plan the way the
/// executor does. Production decode GEMMs at N <= GEMM_LT_MAX_N run the algo
/// `gemm_lt_tune` selected at startup over every layer's weights — an L2-cold
/// rotation — and an untuned context falls back to GemmEx, mis-ranking the
/// small-N projections. The rotation here is sized off the L2 sweep size, so
/// the tuner times DRAM-cold candidates even for the small kv_proj weight;
/// the copies are dropped afterwards (the tuned plan is keyed by shape, not
/// pointer).
fn gemm_weight_tuned(
    ctx: &DeviceContext,
    out_dim: usize,
    in_dim: usize,
    rows: usize,
) -> Result<DeviceMatrix> {
    // Zero weights: cuBLAS HMMA does no zero-skipping, and the lm_head table
    // is too large to build patterned on the host.
    let weight = zeros_matrix(ctx, out_dim, in_dim)?;
    if rows <= pegainfer_kernels::ops::GEMM_LT_MAX_N {
        let weight_bytes = out_dim * in_dim * size_of::<bf16>();
        let l2_bytes = ctx
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?
            as usize;
        let cold_copies = cache_clear_bytes(l2_bytes).div_ceil(weight_bytes).max(1);
        let budget_copies = (TUNE_ROTATION_BUDGET_BYTES / weight_bytes).max(1);
        let extra_copies = cold_copies.min(budget_copies) - 1;
        let rotation: Vec<DeviceMatrix> = (0..extra_copies)
            .map(|_| zeros_matrix(ctx, out_dim, in_dim))
            .collect::<Result<_>>()?;
        let samples: Vec<(&DeviceMatrix, usize)> = std::iter::once((&weight, 0))
            .chain(rotation.iter().map(|weight| (weight, 0)))
            .collect();
        pegainfer_kernels::ops::gemm_lt_tune(ctx, &samples, out_dim, rows)?;
    }
    Ok(weight)
}

impl DenseCase {
    pub fn new(kind: DenseKernelKind, rows: usize) -> Result<Self> {
        anyhow::ensure!(rows > 0, "dense case rows must be greater than zero");
        let ctx = DeviceContext::new()?;

        let buffers = match kind {
            DenseKernelKind::ProjectionGemm(projection) => {
                let (out_dim, in_dim) = projection.out_in();
                DenseBuffers::Gemm {
                    weight: gemm_weight_tuned(&ctx, out_dim, in_dim, rows)?,
                    x: hidden_of(&ctx, in_dim, rows, 0.01)?,
                    out: HiddenStates::zeros(&ctx, out_dim, rows)?,
                }
            }
            DenseKernelKind::RmsNorm => DenseBuffers::Norm {
                weight: ones_vec(&ctx, HIDDEN_SIZE)?,
                x: hidden_of(&ctx, HIDDEN_SIZE, rows, 0.01)?,
                out: HiddenStates::zeros(&ctx, HIDDEN_SIZE, rows)?,
            },
            DenseKernelKind::FusedAddRmsNorm => DenseBuffers::FusedAddNorm {
                weight: ones_vec(&ctx, HIDDEN_SIZE)?,
                hidden: hidden_of(&ctx, HIDDEN_SIZE, rows, 0.01)?,
                residual: hidden_of(&ctx, HIDDEN_SIZE, rows, 0.02)?,
                out: HiddenStates::zeros(&ctx, HIDDEN_SIZE, rows)?,
            },
            DenseKernelKind::QkNormRopeDecode => {
                let positions: Vec<i32> = (0..rows)
                    .map(|i| ((i * 997) % DENSE_ROPE_CACHE_TOKENS) as i32)
                    .collect();
                let (cos_cache, sin_cache) = precompute_rope(
                    &ctx,
                    &RopeTableSpec {
                        rotary_dim: HEAD_DIM,
                        frequency_dim: HEAD_DIM,
                        max_seq_len: DENSE_ROPE_CACHE_TOKENS,
                        theta: 1e6,
                    },
                )?;
                DenseBuffers::QkRope {
                    q: hidden_of(&ctx, Q_DIM, rows, 0.01)?,
                    k: hidden_of(&ctx, KV_DIM, rows, 0.01)?,
                    q_norm: ones_vec(&ctx, HEAD_DIM)?,
                    k_norm: ones_vec(&ctx, HEAD_DIM)?,
                    cos_cache,
                    sin_cache,
                    positions: ctx.stream.clone_htod(&positions)?,
                }
            }
            DenseKernelKind::SiluMul => DenseBuffers::SiluMul {
                gate: hidden_of(&ctx, INTERMEDIATE_SIZE, rows, 0.01)?,
                up: hidden_of(&ctx, INTERMEDIATE_SIZE, rows, 0.02)?,
                out: HiddenStates::zeros(&ctx, INTERMEDIATE_SIZE, rows)?,
            },
            DenseKernelKind::Embedding => {
                let token_ids: Vec<u32> = (0..rows)
                    .map(|i| ((i * 7919) % VOCAB_SIZE) as u32)
                    .collect();
                DenseBuffers::Embedding {
                    table: zeros_matrix(&ctx, VOCAB_SIZE, HIDDEN_SIZE)?,
                    token_ids: ctx.stream.clone_htod(&token_ids)?,
                    out: HiddenStates::zeros(&ctx, HIDDEN_SIZE, rows)?,
                }
            }
            DenseKernelKind::Sampling { greedy } => {
                let params = if greedy {
                    pegainfer_frontend::sampler::SamplingParams::default()
                } else {
                    pegainfer_frontend::sampler::SamplingParams {
                        temperature: 0.8,
                        top_k: 50,
                        top_p: 0.9,
                        min_p: 0.0,
                        seed: None,
                        ignore_eos: true,
                    }
                };
                DenseBuffers::Sampling {
                    logits: peaked_logits(&ctx, rows)?,
                    scratch: pegainfer_sample::SampleScratch::new(&ctx, VOCAB_SIZE, rows)?,
                    params: vec![params; rows],
                    seed: 0x5eed,
                }
            }
        };

        let start = ctx
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let end = ctx
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let case = Self {
            ctx,
            buffers,
            start,
            end,
        };
        case.ctx.sync()?;
        Ok(case)
    }

    pub fn cu_context_ptr(&self) -> *mut c_void {
        self.ctx.ctx.cu_ctx().cast::<c_void>()
    }

    pub fn pre_measure(&mut self) -> Result<()> {
        self.launch_once()?;
        self.ctx.sync()
    }

    pub fn launch_once(&mut self) -> Result<()> {
        use pegainfer_kernels::ops as kops;
        match &mut self.buffers {
            DenseBuffers::Gemm { weight, x, out } => {
                kops::gemm_into(&self.ctx, weight, x, out);
                Ok(())
            }
            DenseBuffers::Norm { weight, x, out } => {
                kops::rms_norm_batch_into(&self.ctx, x, weight, RMS_NORM_EPS, out);
                Ok(())
            }
            DenseBuffers::FusedAddNorm {
                weight,
                hidden,
                residual,
                out,
            } => kops::fused_add_rms_norm_round_batch_into(
                &self.ctx,
                hidden,
                residual,
                weight,
                RMS_NORM_EPS,
                out,
            ),
            DenseBuffers::QkRope {
                q,
                k,
                q_norm,
                k_norm,
                cos_cache,
                sin_cache,
                positions,
            } => {
                kops::qk_norm_rope_batch_decode_into(
                    &self.ctx,
                    q,
                    k,
                    0,
                    q.seq_len,
                    q_norm,
                    k_norm,
                    cos_cache,
                    sin_cache,
                    positions,
                    NUM_QO_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    RMS_NORM_EPS,
                )?;
                Ok(())
            }
            DenseBuffers::SiluMul { gate, up, out } => {
                kops::silu_mul_batch_into(&self.ctx, gate, up, out)
            }
            DenseBuffers::Embedding {
                table,
                token_ids,
                out,
            } => kops::embedding_batch(&self.ctx, table, token_ids, out),
            DenseBuffers::Sampling {
                logits,
                scratch,
                params,
                seed,
            } => {
                let param_refs: Vec<&pegainfer_frontend::sampler::SamplingParams> =
                    params.iter().collect();
                let steps = vec![0u64; param_refs.len()];
                *seed = seed.wrapping_add(1);
                pegainfer_sample::select_batch(
                    &self.ctx,
                    logits,
                    &param_refs,
                    &steps,
                    *seed,
                    scratch,
                )?;
                Ok(())
            }
        }
    }

    /// Cold-L2 latency, same protocol as the attention cases. The sampling
    /// case's measured span includes its device-to-host token readback and
    /// stream sync — that is the production step-tail cost, not overhead.
    pub fn measure_cold_l2(
        &mut self,
        criterion_iters: u64,
        cache_clear: &mut L2CacheClear,
    ) -> Result<Duration> {
        let mut elapsed_ms = 0.0f64;
        for _ in 0..criterion_iters {
            cache_clear.clear(&self.ctx)?;
            self.start.record(&self.ctx.stream)?;
            self.launch_once()?;
            self.end.record(&self.ctx.stream)?;
            elapsed_ms += f64::from(self.start.elapsed_ms(&self.end)?);
        }
        Ok(Duration::from_secs_f64(elapsed_ms / 1_000.0))
    }
}

fn hidden_of(ctx: &DeviceContext, dim: usize, rows: usize, scale: f32) -> Result<HiddenStates> {
    Ok(HiddenStates {
        data: ctx.stream.clone_htod(&patterned_bf16(dim * rows, scale))?,
        hidden_dim: dim,
        seq_len: rows,
    })
}

fn patterned_bf16(len: usize, scale: f32) -> Vec<bf16> {
    (0..len)
        .map(|i| bf16::from_f32((((i % 251) as f32) - 125.0) * scale))
        .collect()
}
