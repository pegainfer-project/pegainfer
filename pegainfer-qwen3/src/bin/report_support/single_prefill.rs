//! Single-request (batch_size=1) unpaged prefill attention bench.

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
use pegainfer_kernels::ffi;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

use super::common::HEAD_DIM;
use super::common::L2CacheClear;
use super::common::NUM_KV_HEADS;
use super::common::NUM_QO_HEADS;
use super::common::patterned_bf16;
use super::prefill_attention::PrefillAttentionShape;
use super::prefill_attention::PrefillAttentionSpec;

pub(crate) struct SinglePrefillCase {
    pub(crate) ctx: DeviceContext,
    q: HiddenStates,
    output: HiddenStates,
    k_cache: CudaSlice<bf16>,
    v_cache: CudaSlice<bf16>,
    start: CudaEvent,
    end: CudaEvent,
    seq_len: usize,
}

impl SinglePrefillCase {
    pub(crate) fn for_spec(spec: PrefillAttentionSpec) -> Result<Self> {
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

    pub(crate) fn shape(&self) -> PrefillAttentionShape {
        PrefillAttentionShape::new(1, self.seq_len)
    }

    pub(crate) fn cu_context_ptr(&self) -> *mut c_void {
        self.ctx.ctx.cu_ctx().cast::<c_void>()
    }

    pub(crate) fn pre_measure(&mut self) -> Result<()> {
        self.launch_once()?;
        self.ctx.sync()
    }

    pub(crate) fn launch_once(&mut self) -> Result<()> {
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

    pub(crate) fn measure_cold_l2(
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
