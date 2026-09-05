//! Batched paged-prefill attention bench, staged into qk-norm-rope / kv-scatter
//! / attention-core sub-phases so each can be measured in isolation.

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
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::rope::precompute_rope;
use pegainfer_kernels::ffi;
use pegainfer_kernels::ops::PrefillPagedPlan;
use pegainfer_kernels::ops::prefill_attention_paged_into;
use pegainfer_kernels::paged_kv::PagedKvLayout;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;
use serde::Deserialize;
use serde::Serialize;

use super::common::HEAD_DIM;
use super::common::L2CacheClear;
use super::common::NUM_KV_HEADS;
use super::common::NUM_LAYERS;
use super::common::NUM_QO_HEADS;
use super::common::PAGE_SIZE;
use super::common::patterned_bf16;

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub(crate) struct PrefillAttentionShape {
    pub(crate) batch_size: usize,
    pub(crate) seq_len: usize,
}

impl PrefillAttentionShape {
    pub(crate) const fn new(batch_size: usize, seq_len: usize) -> Self {
        Self {
            batch_size,
            seq_len,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefillAttentionSpec {
    pub(crate) shape: PrefillAttentionShape,
    pub(crate) variant: PrefillAttentionVariant,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) enum PrefillAttentionVariant {
    Default,
    CtaTileQ(usize),
}

impl PrefillAttentionVariant {
    pub(crate) fn label(self) -> String {
        match self {
            Self::Default => "default".to_string(),
            Self::CtaTileQ(tile_q) => format!("cta_q{tile_q}"),
        }
    }

    pub(crate) fn range_label(self) -> String {
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
pub(crate) enum PrefillStage {
    Full,
    QkNormRope,
    KvScatter,
    AttentionCore,
}

impl PrefillStage {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::QkNormRope => "qk_norm_rope",
            Self::KvScatter => "kv_scatter",
            Self::AttentionCore => "attention_core",
        }
    }

    pub(crate) fn range_label(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::QkNormRope => "qk",
            Self::KvScatter => "kv",
            Self::AttentionCore => "attn",
        }
    }
}

pub(crate) struct AttentionPrefillCase {
    pub(crate) ctx: DeviceContext,
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
    pub(crate) fn for_spec(spec: PrefillAttentionSpec) -> Result<Self> {
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

    pub(crate) fn shape(&self) -> PrefillAttentionShape {
        PrefillAttentionShape::new(self.batch_size, self.seq_len)
    }

    fn total_tokens(&self) -> usize {
        self.batch_size * self.seq_len
    }

    pub(crate) fn cu_context_ptr(&self) -> *mut c_void {
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

    pub(crate) fn prepare_stage(&mut self, stage: PrefillStage) -> Result<()> {
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

    pub(crate) fn pre_measure_stage(&mut self, stage: PrefillStage) -> Result<()> {
        self.prepare_stage(stage)?;
        self.launch_stage(stage)?;
        self.ctx.sync()
    }

    pub(crate) fn launch_stage(&mut self, stage: PrefillStage) -> Result<()> {
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

    pub(crate) fn measure_stage_cold_l2(
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
