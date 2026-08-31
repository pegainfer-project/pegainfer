//! Device tensor types and CUDA context.

use std::cell::Cell;
use std::sync::Arc;

use anyhow::Result;
use anyhow::anyhow;
use cudarc::driver::CudaContext;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::sys::CUstream;
use half::bf16;
use serde::Deserialize;
use serde::Serialize;

use crate::ffi;

// ─── Thread-local stream override for Green Context SM partitioning ────────

thread_local! {
    static STREAM_OVERRIDE: Cell<Option<CUstream>> = const { Cell::new(None) };
    static PREFILL_STREAM_OVERRIDE: Cell<bool> = const { Cell::new(false) };
}

/// Scoped stream override for [`active_cu_stream`]; drop restores the previous value.
pub struct StreamOverrideGuard {
    previous: Option<CUstream>,
    previous_prefill: bool,
}

impl StreamOverrideGuard {
    /// # Safety
    /// The stream must be valid for the guard's lifetime and belong to the
    /// same CUDA device as the `DeviceContext`.
    pub unsafe fn activate(stream: CUstream) -> Self {
        Self::activate_inner(stream, false)
    }

    /// # Safety
    /// The stream must be valid for the guard's lifetime and belong to the
    /// same CUDA device as the `DeviceContext`.
    pub unsafe fn activate_prefill(stream: CUstream) -> Self {
        Self::activate_inner(stream, true)
    }

    fn activate_inner(stream: CUstream, prefill: bool) -> Self {
        let previous = STREAM_OVERRIDE.with(|c| c.replace(Some(stream)));
        let previous_prefill = PREFILL_STREAM_OVERRIDE.with(|c| c.replace(prefill));
        Self {
            previous,
            previous_prefill,
        }
    }
}

impl Drop for StreamOverrideGuard {
    fn drop(&mut self) {
        STREAM_OVERRIDE.with(|c| c.set(self.previous));
        PREFILL_STREAM_OVERRIDE.with(|c| c.set(self.previous_prefill));
    }
}

/// Returns true if a stream override is currently active on this thread.
pub fn has_stream_override() -> bool {
    STREAM_OVERRIDE.with(|c| c.get().is_some())
}

/// Returns true while split-concurrent prefill owns the stream override.
pub fn has_prefill_stream_override() -> bool {
    PREFILL_STREAM_OVERRIDE.with(Cell::get)
}

/// Returns the effective CUDA stream: the thread-local override if set,
/// otherwise the context's own stream.
#[inline]
pub fn active_cu_stream(ctx: &DeviceContext) -> CUstream {
    STREAM_OVERRIDE
        .with(Cell::get)
        .unwrap_or_else(|| ctx.stream.cu_stream())
}

/// Poll the active stream up to a fixed cap, then synchronize that stream.
const STREAM_SPIN_WAIT_CAP: std::time::Duration = std::time::Duration::from_millis(5);

pub fn stream_spin_wait(ctx: &DeviceContext) -> anyhow::Result<()> {
    let stream = active_cu_stream(ctx);
    let cap = std::time::Instant::now() + STREAM_SPIN_WAIT_CAP;
    loop {
        match unsafe { cudarc::driver::sys::cuStreamQuery(stream) } {
            cudarc::driver::sys::CUresult::CUDA_SUCCESS => return Ok(()),
            cudarc::driver::sys::CUresult::CUDA_ERROR_NOT_READY => {
                if std::time::Instant::now() >= cap {
                    let result = unsafe { cudarc::driver::sys::cuStreamSynchronize(stream) };
                    anyhow::ensure!(
                        result == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                        "stream sync after spin cap failed: {result:?}"
                    );
                    return Ok(());
                }
                std::hint::spin_loop();
            }
            err => return Err(anyhow::anyhow!("cuStreamQuery failed: {err:?}")),
        }
    }
}

/// Copy token ids from an i32 argmax buffer into a u32 embedding buffer.
pub fn memcpy_dtod_u32_from_i32(
    ctx: &DeviceContext,
    src: &CudaSlice<i32>,
    dst: &mut CudaSlice<u32>,
    count: usize,
) -> anyhow::Result<()> {
    use cudarc::driver::DevicePtr;
    use cudarc::driver::DevicePtrMut;

    anyhow::ensure!(
        !has_stream_override(),
        "dtod i32->u32 copy runs on the base stream only"
    );
    anyhow::ensure!(
        count <= src.len() && count <= dst.len(),
        "dtod i32->u32 copy of {count} exceeds src {} or dst {}",
        src.len(),
        dst.len()
    );
    if count == 0 {
        return Ok(());
    }
    let (src_ptr, _src_guard) = src.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = dst.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
            dst_ptr,
            src_ptr,
            count * std::mem::size_of::<i32>(),
            ctx.stream.cu_stream(),
        )
    };
    anyhow::ensure!(
        result == cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "cuMemcpyDtoDAsync i32->u32 failed: {result:?}"
    );
    Ok(())
}

/// Marker trait for tensor metadata tags.
pub trait NamedTag {
    const NAME: &'static str;
}

/// Marker trait for tensor element type vocabulary.
pub trait DTypeTag: NamedTag {}

/// Marker trait for tensor layout vocabulary.
pub trait LayoutTag: NamedTag {}

/// Marker trait for tensor axis vocabulary.
pub trait AxisTag: NamedTag {}

macro_rules! named_tag {
    ($name:ident, $value:literal, $trait_name:ident) => {
        #[derive(Clone, Copy, Debug, Default)]
        pub struct $name;

        impl NamedTag for $name {
            const NAME: &'static str = $value;
        }

        impl $trait_name for $name {}
    };
}

named_tag!(Bf16, "bf16", DTypeTag);
named_tag!(F32, "f32", DTypeTag);
named_tag!(I32, "i32", DTypeTag);
named_tag!(U32, "u32", DTypeTag);
named_tag!(U8, "u8", DTypeTag);

named_tag!(Contiguous1D, "contiguous_1d", LayoutTag);
named_tag!(RowMajor2D, "row_major_2d", LayoutTag);
named_tag!(HiddenStatesLayout, "hidden_states", LayoutTag);
named_tag!(PagedKvPageFirst, "paged_kv_page_first", LayoutTag);

named_tag!(Batch, "batch", AxisTag);
named_tag!(BatchPlusOne, "batch_plus_1", AxisTag);
named_tag!(HeadDim, "head_dim", AxisTag);
named_tag!(Hidden, "hidden", AxisTag);
named_tag!(InDim, "in", AxisTag);
named_tag!(Intermediate, "intermediate", AxisTag);
named_tag!(Inter2, "inter2", AxisTag);
named_tag!(Kv, "kv", AxisTag);
named_tag!(KvDim, "kv_dim", AxisTag);
named_tag!(KvHead, "kv_head", AxisTag);
named_tag!(Layer, "layer", AxisTag);
named_tag!(OutDim, "out", AxisTag);
named_tag!(OutTotal, "out_total", AxisTag);
named_tag!(Page, "page", AxisTag);
named_tag!(PageSlot, "page_slot", AxisTag);
named_tag!(PosInPage, "pos_in_page", AxisTag);
named_tag!(QDim, "q_dim", AxisTag);
named_tag!(RopeDim, "rope_dim", AxisTag);
named_tag!(Seq, "seq", AxisTag);
named_tag!(Tile, "tile", AxisTag);
named_tag!(Token, "token", AxisTag);
named_tag!(Vocab, "vocab", AxisTag);

/// One named axis in an erased tensor metadata description.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct AxisSpec {
    pub name: String,
    pub size: usize,
}

impl AxisSpec {
    pub fn new<A: AxisTag>(size: usize) -> Self {
        Self {
            name: A::NAME.to_string(),
            size,
        }
    }

    pub fn named(name: impl Into<String>, size: usize) -> Self {
        Self {
            name: name.into(),
            size,
        }
    }
}

/// Erased tensor metadata for schedules, reports, and future instrumentation.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct TensorSpec {
    pub(crate) dtype: String,
    pub(crate) layout: String,
    pub axes: Vec<AxisSpec>,
}

impl TensorSpec {
    pub fn new<D: DTypeTag, L: LayoutTag>(axes: impl IntoIterator<Item = AxisSpec>) -> Self {
        Self {
            dtype: D::NAME.to_string(),
            layout: L::NAME.to_string(),
            axes: axes.into_iter().collect(),
        }
    }

    pub fn named(
        dtype: impl Into<String>,
        layout: impl Into<String>,
        axes: impl IntoIterator<Item = AxisSpec>,
    ) -> Self {
        Self {
            dtype: dtype.into(),
            layout: layout.into(),
            axes: axes.into_iter().collect(),
        }
    }

    pub fn compact(&self) -> String {
        let axes = self
            .axes
            .iter()
            .map(|axis| format!("{}={}", axis.name, axis.size))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}[{}] layout={}", self.dtype, axes, self.layout)
    }
}

/// A named kernel argument carrying an erased tensor description.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct TensorArg {
    pub name: String,
    pub spec: TensorSpec,
}

impl TensorArg {
    fn new(name: impl Into<String>, spec: TensorSpec) -> Self {
        Self {
            name: name.into(),
            spec,
        }
    }

    pub fn compact(&self) -> String {
        format!("{}: {}", self.name, self.spec.compact())
    }
}

/// String-valued non-tensor kernel metadata.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct AttrSpec {
    pub name: String,
    pub value: String,
}

impl AttrSpec {
    fn new(name: impl Into<String>, value: String) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

/// Erased logical kernel call IR shared by static schedules and future traces.
#[derive(Clone, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct KernelCall {
    pub op: String,
    pub label: String,
    pub inputs: Vec<TensorArg>,
    pub outputs: Vec<TensorArg>,
    pub attrs: Vec<AttrSpec>,
}

impl KernelCall {
    #[must_use]
    pub fn new(op: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            op: op.into(),
            label: label.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            attrs: Vec::new(),
        }
    }

    #[must_use]
    pub fn input(mut self, name: impl Into<String>, spec: TensorSpec) -> Self {
        self.inputs.push(TensorArg::new(name, spec));
        self
    }

    #[must_use]
    pub fn output(mut self, name: impl Into<String>, spec: TensorSpec) -> Self {
        self.outputs.push(TensorArg::new(name, spec));
        self
    }

    #[must_use]
    pub fn attr(mut self, name: impl Into<String>, value: String) -> Self {
        self.attrs.push(AttrSpec::new(name, value));
        self
    }
}

/// CUDA device context holding context and stream.
#[derive(Clone)]
pub struct DeviceContext {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub device_ordinal: usize,
}

impl DeviceContext {
    pub fn new() -> Result<Self> {
        Self::new_with_device(0)
    }

    pub fn new_with_device(device_ordinal: usize) -> Result<Self> {
        unsafe {
            let err = ffi::cuda_set_device(device_ordinal as i32);
            if err != 0 {
                return Err(anyhow!(
                    "Failed to set CUDA device {}: cudaError={}",
                    device_ordinal,
                    err
                ));
            }
        }
        let ctx = CudaContext::new(device_ordinal)
            .map_err(|e| anyhow!("Failed to create CUDA context: {}", e))?;

        // Disable multi-stream event tracking before creating streams.
        // We use a single compute stream, so no cross-stream synchronization is needed.
        // This avoids stream.wait(event) calls that break CUDA Graph capture.
        // SAFETY: We only use one stream for all GPU work.
        unsafe {
            ctx.disable_event_tracking();
        }

        let stream = ctx
            .new_stream()
            .map_err(|e| anyhow!("Failed to create CUDA stream: {}", e))?;

        // Initialize cuBLAS handle
        unsafe {
            ffi::cublas_init();
        }

        Ok(Self {
            ctx,
            stream,
            device_ordinal,
        })
    }

    /// Synchronize stream
    pub fn sync(&self) -> Result<()> {
        self.stream
            .synchronize()
            .map_err(|e| anyhow!("Sync failed: {}", e))
    }
}

/// 1D device tensor (vector) — stored as bf16.
pub struct DeviceVec {
    pub data: CudaSlice<bf16>,
    pub len: usize,
}

impl DeviceVec {
    /// Create from host data (bf16)
    pub fn from_host(ctx: &DeviceContext, data: &[bf16]) -> Result<Self> {
        let gpu_data = ctx
            .stream
            .clone_htod(data)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            len: data.len(),
        })
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn from_safetensors(ctx: &DeviceContext, data: &[u8]) -> Result<Self> {
        if !data.len().is_multiple_of(2) {
            return Err(anyhow!(
                "Data length must be even for bf16: got {} bytes",
                data.len()
            ));
        }
        let len = data.len() / 2;
        // NOTE: This assumes a little-endian host. Safetensors are little-endian.
        // On a big-endian machine, this will be incorrect. A full solution would
        // involve byte-swapping.
        let slice = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<bf16>(), len) };
        Self::from_host(ctx, slice)
    }

    /// Create zeroed tensor
    pub fn zeros(ctx: &DeviceContext, len: usize) -> Result<Self> {
        let gpu_data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(len)
            .map_err(|e| anyhow!("Alloc failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            len,
        })
    }

    /// Copy to host as f32.
    pub fn to_host(&self, ctx: &DeviceContext) -> Result<Vec<f32>> {
        let host_f16 = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
        ctx.sync()?;
        Ok(host_f16.iter().map(|x| x.to_f32()).collect())
    }
}

impl Clone for DeviceVec {
    fn clone(&self) -> Self {
        Self {
            data: self.data.try_clone().unwrap(),
            len: self.len,
        }
    }
}

/// 2D device tensor (matrix) — stored in row-major order as bf16.
pub struct DeviceMatrix {
    pub data: CudaSlice<bf16>,
    pub rows: usize,
    pub cols: usize,
}

impl DeviceMatrix {
    /// Vertically stack matrices (same cols, concatenate rows). GPU D2D copy.
    pub fn vstack(ctx: &DeviceContext, matrices: &[&DeviceMatrix]) -> Result<Self> {
        assert!(!matrices.is_empty());
        let cols = matrices[0].cols;
        for m in matrices {
            assert_eq!(m.cols, cols, "vstack: all matrices must have same cols");
        }
        let total_rows: usize = matrices.iter().map(|m| m.rows).sum();
        let mut data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(total_rows * cols)
            .map_err(|e| anyhow!("vstack alloc failed: {}", e))?;
        let mut offset = 0;
        for m in matrices {
            let n = m.rows * m.cols;
            let src = m.data.slice(..n);
            let mut dst = data.slice_mut(offset..offset + n);
            ctx.stream
                .memcpy_dtod(&src, &mut dst)
                .map_err(|e| anyhow!("vstack D2D copy failed: {}", e))?;
            offset += n;
        }
        Ok(Self {
            data,
            rows: total_rows,
            cols,
        })
    }

    /// Create from host data (row-major, bf16)
    pub fn from_host(ctx: &DeviceContext, data: &[bf16], rows: usize, cols: usize) -> Result<Self> {
        assert_eq!(data.len(), rows * cols);
        let gpu_data = ctx
            .stream
            .clone_htod(data)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            rows,
            cols,
        })
    }

    #[allow(clippy::cast_ptr_alignment)]
    pub fn from_safetensors(
        ctx: &DeviceContext,
        data: &[u8],
        rows: usize,
        cols: usize,
    ) -> Result<Self> {
        if data.len() != rows * cols * std::mem::size_of::<bf16>() {
            return Err(anyhow!(
                "Data length mismatch: expected {} bytes, got {} bytes",
                rows * cols * std::mem::size_of::<bf16>(),
                data.len()
            ));
        }
        // NOTE: This assumes a little-endian host. Safetensors are little-endian.
        // On a big-endian machine, this will be incorrect. A full solution would
        // involve byte-swapping.
        let slice =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<bf16>(), rows * cols) };
        Self::from_host(ctx, slice, rows, cols)
    }
}

/// Batched hidden states: seq_len vectors of dim hidden_dim, stored contiguously.
/// Memory layout: [hidden_dim * seq_len] elements, token i at offset i * hidden_dim.
/// cuBLAS interprets as [hidden_dim, seq_len] column-major.
pub struct HiddenStates {
    pub data: CudaSlice<bf16>,
    pub hidden_dim: usize,
    pub seq_len: usize,
}

impl HiddenStates {
    pub fn as_ref(&self) -> HiddenStatesRef<'_> {
        HiddenStatesRef {
            data: &self.data,
            hidden_dim: self.hidden_dim,
            seq_len: self.seq_len,
        }
    }

    /// Logical extent `hidden_dim * seq_len`, checked to fit the backing
    /// allocation (`>=`, not `==`: buffers allocate at a max and rewrite
    /// `seq_len` to the active size per step). Fields are public, so a safe
    /// caller can shape a logical size past the backing; launch wrappers
    /// call this to reject that before it reaches a kernel.
    pub(crate) fn checked_extent(&self, what: &str) -> Result<usize> {
        let extent = self
            .hidden_dim
            .checked_mul(self.seq_len)
            .ok_or_else(|| anyhow!("{what} logical extent overflow"))?;
        if self.data.len() < extent {
            return Err(anyhow!(
                "{what} backing len {} < hidden_dim {} * seq_len {}",
                self.data.len(),
                self.hidden_dim,
                self.seq_len
            ));
        }
        Ok(extent)
    }

    /// Create zeroed batch
    pub fn zeros(ctx: &DeviceContext, hidden_dim: usize, seq_len: usize) -> Result<Self> {
        let data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(hidden_dim * seq_len)
            .map_err(|e| anyhow!("Alloc failed: {}", e))?;
        Ok(Self {
            data,
            hidden_dim,
            seq_len,
        })
    }

    /// Create from host data: `hidden_dim * seq_len` bf16, token `i` at `i * hidden_dim`.
    pub fn from_host(
        ctx: &DeviceContext,
        data: &[bf16],
        hidden_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        assert_eq!(data.len(), hidden_dim * seq_len);
        let gpu_data = ctx
            .stream
            .clone_htod(data)
            .map_err(|e| anyhow!("H2D copy failed: {}", e))?;
        Ok(Self {
            data: gpu_data,
            hidden_dim,
            seq_len,
        })
    }

    /// Copy to host as f32. bf16 → f32 is lossless, so f32 equality is bitwise.
    pub fn to_host(&self, ctx: &DeviceContext) -> Result<Vec<f32>> {
        let host = ctx
            .stream
            .clone_dtoh(&self.data)
            .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
        ctx.sync()?;
        Ok(host.iter().map(|x| x.to_f32()).collect())
    }
}

// ── Typed tensor layer ───────────────────────────────────────────────
//
// `GpuTensor<DIM>` encodes the hidden dimension in the type. The seq_len
// (batch) axis stays runtime because it changes per step. Weight matrices
// carry both dimensions: `GpuWeight<OUT, IN>`.
//
// These are additive — existing `HiddenStates`/`DeviceMatrix` stay untouched.
// Model crates migrate one at a time.

/// Batched bf16 activation tensor with compile-time hidden dimension.
///
/// Memory layout: `[DIM * seq_len]` contiguous bf16, token `i` at `i * DIM`.
/// cuBLAS sees `[DIM, seq_len]` column-major.
pub struct GpuTensor<const DIM: usize> {
    pub data: CudaSlice<bf16>,
    pub seq_len: usize,
}

impl<const DIM: usize> GpuTensor<DIM> {
    pub fn zeros(ctx: &DeviceContext, seq_len: usize) -> Result<Self> {
        let data: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(DIM * seq_len)
            .map_err(|e| anyhow!("GpuTensor<{}>::zeros alloc failed: {}", DIM, e))?;
        Ok(Self { data, seq_len })
    }

    pub fn from_device_matrix_rows(m: DeviceMatrix) -> Result<Self> {
        anyhow::ensure!(
            m.cols == DIM,
            "GpuTensor<{}>::from_device_matrix_rows col mismatch: got {}",
            DIM,
            m.cols,
        );
        Ok(Self {
            data: m.data,
            seq_len: m.rows,
        })
    }
}

/// bf16 weight matrix with compile-time dimensions: `[OUT, IN]` row-major.
pub struct GpuWeight<const OUT: usize, const IN: usize> {
    pub(crate) data: CudaSlice<bf16>,
}

impl<const OUT: usize, const IN: usize> GpuWeight<OUT, IN> {
    pub fn from_device_matrix(m: DeviceMatrix) -> Result<Self> {
        anyhow::ensure!(
            m.rows == OUT && m.cols == IN,
            "GpuWeight<{}, {}>::from_device_matrix shape mismatch: got [{}, {}]",
            OUT,
            IN,
            m.rows,
            m.cols,
        );
        Ok(Self { data: m.data })
    }
}

/// bf16 RMSNorm weight vector with compile-time dimension.
pub struct NormWeight<const DIM: usize> {
    pub data: CudaSlice<bf16>,
}

impl<const DIM: usize> NormWeight<DIM> {
    pub fn from_device_vec(v: DeviceVec) -> Result<Self> {
        anyhow::ensure!(
            v.len == DIM,
            "NormWeight<{}>::from_device_vec len mismatch: got {}",
            DIM,
            v.len,
        );
        Ok(Self { data: v.data })
    }
}

/// f32 raw buffer with compile-time element count per batch entry.
pub struct GpuRawSlice<const ELEMS: usize> {
    pub data: CudaSlice<f32>,
    pub batch_size: usize,
}

impl<const ELEMS: usize> GpuRawSlice<ELEMS> {
    pub fn zeros(ctx: &DeviceContext, batch_size: usize) -> Result<Self> {
        let data: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(ELEMS * batch_size)
            .map_err(|e| anyhow!("GpuRawSlice<{}>::zeros alloc failed: {}", ELEMS, e))?;
        Ok(Self { data, batch_size })
    }
}

/// i32 raw buffer with compile-time element count per batch entry.
pub struct GpuRawSliceI32<const ELEMS: usize> {
    pub data: CudaSlice<i32>,
    pub batch_size: usize,
}

impl<const ELEMS: usize> GpuRawSliceI32<ELEMS> {
    pub fn zeros(ctx: &DeviceContext, batch_size: usize) -> Result<Self> {
        let data: CudaSlice<i32> = ctx
            .stream
            .alloc_zeros(ELEMS * batch_size)
            .map_err(|e| anyhow!("GpuRawSliceI32<{}>::zeros alloc failed: {}", ELEMS, e))?;
        Ok(Self { data, batch_size })
    }
}

/// Non-owning reference to `HiddenStates`-shaped data (bridge to untyped ops).
#[derive(Clone, Copy)]
pub struct HiddenStatesRef<'a> {
    pub data: &'a CudaSlice<bf16>,
    pub hidden_dim: usize,
    pub seq_len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_override_guard_restores_on_drop() {
        let outer = 0x10usize as CUstream;
        assert!(!has_stream_override());
        assert!(!has_prefill_stream_override());
        {
            let _outer_guard = unsafe { StreamOverrideGuard::activate_prefill(outer) };
            assert!(has_prefill_stream_override());
            {
                let _inner_guard = unsafe { StreamOverrideGuard::activate(0x20usize as CUstream) };
                assert!(!has_prefill_stream_override());
            }
            assert_eq!(STREAM_OVERRIDE.with(Cell::get), Some(outer));
            assert!(has_prefill_stream_override());
        }
        assert!(!has_stream_override());
        assert!(!has_prefill_stream_override());
    }

    fn copy_matrix_to_host(ctx: &DeviceContext, matrix: &DeviceMatrix) -> Vec<bf16> {
        let host = ctx
            .stream
            .clone_dtoh(&matrix.data)
            .expect("D2H copy failed");
        ctx.sync().expect("CUDA sync failed");
        host
    }

    #[test]
    fn test_device_matrix_from_safetensors_matches_from_host() {
        let ctx = DeviceContext::new().expect("Failed to create CUDA context");
        let rows = 3;
        let cols = 2;
        let host = vec![
            bf16::from_f32(-8.0),
            bf16::from_f32(-0.25),
            bf16::from_f32(1.0),
            bf16::from_f32(3.5),
            bf16::from_f32(9.0),
            bf16::from_f32(10.75),
        ];
        let safetensor_bytes: Vec<u8> = host
            .iter()
            .flat_map(|v| v.to_bits().to_le_bytes())
            .collect();

        let from_host =
            DeviceMatrix::from_host(&ctx, &host, rows, cols).expect("from_host should succeed");
        let from_safetensors = DeviceMatrix::from_safetensors(&ctx, &safetensor_bytes, rows, cols)
            .expect("from_safetensors should succeed");

        assert_eq!(from_safetensors.rows, from_host.rows);
        assert_eq!(from_safetensors.cols, from_host.cols);

        let host_out = copy_matrix_to_host(&ctx, &from_host);
        let safetensors_out = copy_matrix_to_host(&ctx, &from_safetensors);
        assert_eq!(host_out.len(), safetensors_out.len());
        for (idx, (a, b)) in host_out.iter().zip(safetensors_out.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "from_safetensors/from_host mismatch at index {}",
                idx
            );
        }
    }
}
