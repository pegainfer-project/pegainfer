//! Decode (paged) attention kernel bench: non-partitioned and split-KV
//! decode paths at production shapes.

use std::ffi::c_void;
use std::time::Duration;

use anyhow::Result;
use anyhow::bail;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys;
use half::bf16;
use pegainfer_core::ops::build_split_kv_csr;
use pegainfer_kernels::ffi;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;
use pegainfer_qwen3::runtime::SPLIT_KV_CHUNK_TOKENS;
use pegainfer_qwen3::runtime::SPLIT_KV_TUNED_MAX_CHUNKS;
use pegainfer_qwen3::runtime::SplitKvConfig;
use serde::Deserialize;
use serde::Serialize;

use super::common::HEAD_DIM;
use super::common::L2CacheClear;
use super::common::NUM_KV_HEADS;
use super::common::NUM_LAYERS;
use super::common::NUM_QO_HEADS;
use super::common::PAGE_SIZE;
use super::common::patterned_bf16;

/// The production Tuned decode-path split width, taken from the runtime constants
/// rather than restated — a retune there must move the report's `split_tmp_*` sizing
/// with it. Deliberately not the opt-in `--batch-invariant` Pin width.
const DEFAULT_SPLIT_KV_CONFIG: SplitKvConfig =
    SplitKvConfig::new(SPLIT_KV_CHUNK_TOKENS, SPLIT_KV_TUNED_MAX_CHUNKS);

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) enum AttentionKernelVariant {
    NonPartition,
    SplitKv(SplitKvConfig),
}

impl AttentionKernelVariant {
    pub(crate) fn label(self) -> String {
        match self {
            Self::NonPartition => "non_partition".to_string(),
            Self::SplitKv(config) => config.label(),
        }
    }

    pub(crate) fn decode_path(self) -> DecodePath {
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
pub(crate) enum DecodePath {
    NonPartition,
    SplitK,
}

impl DecodePath {
    pub(crate) fn name(self, split_config: SplitKvConfig) -> String {
        match self {
            Self::NonPartition => "non_partition".to_string(),
            Self::SplitK => split_config.label(),
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub(crate) struct AttentionKernelShape {
    pub(crate) batch_size: usize,
    pub(crate) kv_len: usize,
}

impl AttentionKernelShape {
    pub(crate) const fn new(batch_size: usize, kv_len: usize) -> Self {
        Self { batch_size, kv_len }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AttentionKernelSpec {
    pub(crate) shape: AttentionKernelShape,
    pub(crate) variant: AttentionKernelVariant,
}

pub(crate) struct AttentionDecodeCase {
    pub(crate) ctx: DeviceContext,
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
    pub(crate) fn for_spec(spec: AttentionKernelSpec) -> Result<Self> {
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

    pub(crate) fn shape(&self) -> AttentionKernelShape {
        AttentionKernelShape::new(self.batch_size, self.kv_len)
    }

    pub(crate) fn split_config(&self) -> SplitKvConfig {
        self.split_config
    }

    pub(crate) fn cu_context_ptr(&self) -> *mut c_void {
        self.ctx.ctx.cu_ctx().cast::<c_void>()
    }

    pub(crate) fn launch_once(&mut self, path: DecodePath) -> Result<()> {
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

    pub(crate) fn measure_decode_only_cold_l2(
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
