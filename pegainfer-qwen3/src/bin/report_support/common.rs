//! Report-harness primitives shared across every benchmark domain: device
//! geometry constants, L2 cache clearing, device bandwidth query, and
//! synthetic buffer generation.

use std::mem::size_of;

use anyhow::Result;
use anyhow::anyhow;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys;
use half::bf16;
use pegainfer_kernels::ffi;
use pegainfer_kernels::tensor::DeviceContext;
use serde::Serialize;

pub(crate) const NUM_LAYERS: usize = 1;
pub(crate) const NUM_QO_HEADS: usize = 32;
pub(crate) const NUM_KV_HEADS: usize = 8;
pub(crate) const HEAD_DIM: usize = 128;
pub(crate) const PAGE_SIZE: usize = 16;
pub(crate) const REPORT_ITERS: u64 = 128;
const MEMORY_TRANSFERS_PER_CLOCK: f64 = 2.0;
const CACHE_CLEAR_L2_MULTIPLIER: usize = 2;
const CACHE_CLEAR_MIN_BYTES: usize = 128 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct DevicePeakBandwidth {
    pub(crate) memory_clock_khz: i32,
    pub(crate) memory_bus_width_bits: i32,
    peak_bytes_per_sec: f64,
}

impl DevicePeakBandwidth {
    pub(crate) fn query(ctx: &DeviceContext) -> Result<Self> {
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

    pub(crate) fn peak_gb_per_sec(&self) -> f64 {
        self.peak_bytes_per_sec / 1.0e9
    }
}

pub(crate) struct L2CacheClear {
    a: CudaSlice<bf16>,
    b: CudaSlice<bf16>,
    out: CudaSlice<bf16>,
    len: usize,
}

impl L2CacheClear {
    pub(crate) fn new(ctx: &DeviceContext) -> Result<Self> {
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

    pub(crate) fn clear(&mut self, ctx: &DeviceContext) -> Result<()> {
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

pub(crate) fn cache_clear_bytes(l2_bytes: usize) -> usize {
    (l2_bytes * CACHE_CLEAR_L2_MULTIPLIER).max(CACHE_CLEAR_MIN_BYTES)
}

pub(crate) fn patterned_bf16(len: usize, scale: f32) -> Vec<bf16> {
    (0..len)
        .map(|i| bf16::from_f32((((i % 251) as f32) - 125.0) * scale))
        .collect()
}
