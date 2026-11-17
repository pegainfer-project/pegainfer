//! Gemma 4 QAT W4A16 linears: the checkpoint's compressed-tensors weights in
//! the TileLang decode GEMMs' fragment layout, those GEMMs for up to sixteen
//! rows, and a bf16 dequantization for wider steps, which then run the dense
//! GEMM. The gate|up linear's GEMMs write gelu(gate) * up directly.

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys::CUdevice_attribute;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceMatrix;
use crate::tensor::HiddenStates;

const GROUP: usize = 32;
const BUCKETS: [usize; 5] = [1, 2, 4, 8, 16];

/// `(ctas, block_n, block_k, warps)` the generated GEMMs were compiled for,
/// `None` in a stub build.
pub fn gemma4_w4a16_geometry() -> Option<(usize, usize, usize, usize)> {
    let stated = option_env!("PEGAINFER_GEMMA4_W4A16_TILELANG_GEOMETRY")?;
    let mut parts = stated.split(',').map(str::trim).map(str::parse::<usize>);
    let mut next = || parts.next()?.ok();
    Some((next()?, next()?, next()?, next()?))
}

/// The per-CTA list of stream-K contributors whose parts of a split tile the
/// CTA finishes, flattened with offsets; `w4a16_defs.plan` in Python.
fn fixup_plan(nb: usize, kb: usize, ctas: usize) -> (Vec<i32>, Vec<i32>) {
    let units = nb * kb;
    let lo = |c: usize| c * units / ctas;
    let hi = |c: usize| (c + 1) * units / ctas;
    let mut offsets = vec![0i32];
    let mut flat = Vec::new();
    for c in 0..ctas {
        if lo(c) < hi(c) {
            let t = (hi(c) - 1) / kb;
            if t * kb >= lo(c) && (t + 1) * kb > hi(c) {
                let mut c2 = c + 1;
                while c2 < ctas && lo(c2) < (t + 1) * kb {
                    if lo(c2) < hi(c2) {
                        flat.push(c2 as i32);
                    }
                    c2 += 1;
                }
            }
        }
        offsets.push(flat.len() as i32);
    }
    (offsets, flat)
}

/// A W4A16 linear `rows x cols` in the GEMMs' fragment layout, with its
/// stream-K fix-up table.
pub struct W4a16Matrix {
    wq: CudaSlice<u32>,
    sq: CudaSlice<u32>,
    fin_off: CudaSlice<i32>,
    fin_list: CudaSlice<i32>,
    pub rows: usize,
    pub cols: usize,
    /// Gate then up, `rows / 2` each: the layout interleaves them so its
    /// GEMMs write gelu(gate) * up, `rows / 2` wide.
    pub gelu_mul: bool,
}

impl W4a16Matrix {
    /// From the checkpoint's `[rows, cols / 8]` packed int32 words and
    /// `[rows, cols / 32]` bf16 scales, both as the bytes the file stores;
    /// `gelu_mul` marks a gate|up stack. Refuses a build without the
    /// generated GEMMs or a device they were not compiled for, before the
    /// weight is rewritten.
    pub fn from_checkpoint(
        ctx: &DeviceContext,
        packed: &CudaSlice<u8>,
        scales: &CudaSlice<u8>,
        rows: usize,
        cols: usize,
        gelu_mul: bool,
    ) -> Result<Self> {
        let (ctas, block_n, block_k, _) = gemma4_w4a16_geometry().context(
            "this build carries no W4A16 GEMMs: build with the gemma4 feature where TileLang is \
             installed, and with the serving device visible or PEGAINFER_GEMMA4_W4A16_SMS set to \
             its SM count",
        )?;
        check_device(ctx, ctas, rows, cols)?;
        ensure!(
            rows.is_multiple_of(block_n) && cols.is_multiple_of(block_k),
            "W4A16 {rows} x {cols} is not whole {block_n} x {block_k} tiles"
        );
        ensure!(
            packed.len() == rows * cols / 2 && scales.len() == rows * cols / GROUP * 2,
            "W4A16 {rows} x {cols} carries {} packed and {} scale bytes",
            packed.len(),
            scales.len()
        );
        let mut wq = ctx.stream.alloc_zeros::<u32>(rows * cols / 8)?;
        let mut sq = ctx
            .stream
            .alloc_zeros::<u32>(rows / 16 * (cols / GROUP) * 8)?;
        {
            let (packed, _g0) = packed.device_ptr(&ctx.stream);
            let (scales, _g1) = scales.device_ptr(&ctx.stream);
            let (wq_ptr, _g2) = wq.device_ptr_mut(&ctx.stream);
            let (sq_ptr, _g3) = sq.device_ptr_mut(&ctx.stream);
            unsafe {
                ffi::gemma4_w4a16_pack_cuda(
                    packed as *const u32,
                    scales as *const u16,
                    wq_ptr as *mut u32,
                    sq_ptr as *mut u32,
                    i32::try_from(rows)?,
                    i32::try_from(cols)?,
                    split(rows, gelu_mul)?,
                    crate::tensor::active_cu_stream(ctx),
                )
            }
            .result()?;
        }
        let (offsets, flat) = fixup_plan(rows / block_n, cols / block_k, ctas);
        let fin_off = ctx.stream.clone_htod(&offsets)?;
        let fin_list = ctx.stream.clone_htod(if flat.is_empty() {
            &[0i32][..]
        } else {
            &flat[..]
        })?;
        Ok(Self {
            wq,
            sq,
            fin_off,
            fin_list,
            rows,
            cols,
            gelu_mul,
        })
    }

    /// The weight as a bf16 `[rows, cols]` matrix in checkpoint row order,
    /// into `out`'s storage.
    pub fn dequant_into(&self, ctx: &DeviceContext, out: &mut DeviceMatrix) -> Result<()> {
        ensure!(
            out.data.len() >= self.rows * self.cols,
            "dequant scratch holds {} values, not {} x {}",
            out.data.len(),
            self.rows,
            self.cols
        );
        out.rows = self.rows;
        out.cols = self.cols;
        let (wq, _g0) = self.wq.device_ptr(&ctx.stream);
        let (sq, _g1) = self.sq.device_ptr(&ctx.stream);
        let (dst, _g2) = out.data.device_ptr_mut(&ctx.stream);
        unsafe {
            ffi::gemma4_w4a16_dequant_cuda(
                wq as *const u32,
                sq as *const u32,
                dst as *mut ffi::Half,
                i32::try_from(self.rows)?,
                i32::try_from(self.cols)?,
                split(self.rows, self.gelu_mul)?,
                crate::tensor::active_cu_stream(ctx),
            )
        }
        .result()?;
        Ok(())
    }

    /// Whether a step of `rows` runs the TileLang GEMM, as opposed to the
    /// dequantized dense one.
    pub fn runs_tilelang(rows: usize) -> bool {
        rows <= BUCKETS[BUCKETS.len() - 1]
    }
}

/// The layout's gate|up interleave argument: the half height, zero for none.
fn split(rows: usize, gelu_mul: bool) -> Result<i32> {
    Ok(if gelu_mul {
        i32::try_from(rows / 2)?
    } else {
        0
    })
}

/// The in-kernel fix-up waits on other CTAs, so every CTA has to be resident
/// at once: the device must be the one the CTA count was derived from, and
/// hold two blocks per SM of each bucket's kernel.
fn check_device(ctx: &DeviceContext, ctas: usize, rows: usize, cols: usize) -> Result<()> {
    let sms = ctx
        .ctx
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
    let sms = usize::try_from(sms)?;
    ensure!(
        ctas == 2 * sms,
        "the W4A16 GEMMs were compiled for {} SMs and this device has {sms}; rebuild with \
         PEGAINFER_GEMMA4_W4A16_SMS={sms}",
        ctas / 2
    );
    for bucket in BUCKETS {
        let mut blocks = 0i32;
        let status = unsafe {
            ffi::gemma4_w4a16_occupancy(
                i32::try_from(rows)?,
                i32::try_from(cols)?,
                bucket as i32,
                &raw mut blocks,
                crate::tensor::active_cu_stream(ctx),
            )
        };
        ensure!(
            status == 0,
            "no W4A16 GEMM was built for {rows} x {cols} at {bucket} rows (cudaError {status})"
        );
        ensure!(
            blocks >= 2,
            "the {rows} x {cols} W4A16 GEMM at {bucket} rows fits {blocks} block(s) per SM; its \
             in-kernel fix-up needs two"
        );
    }
    Ok(())
}

/// Stream-K partials and flags for the decode GEMMs, and the bf16 matrix the
/// wide steps dequantize into. One per stream: the GEMMs leave the flags
/// zeroed, so calls in stream order share them.
pub struct W4a16Scratch {
    part: CudaSlice<f32>,
    flags: CudaSlice<i32>,
    dense: DeviceMatrix,
}

impl W4a16Scratch {
    /// `max_values` is the largest linear's `rows * cols`.
    pub fn new(ctx: &DeviceContext, max_values: usize) -> Result<Self> {
        let (ctas, block_n, _, _) =
            gemma4_w4a16_geometry().context("this build carries no W4A16 GEMMs")?;
        Ok(Self {
            part: ctx.stream.alloc_zeros(ctas * 16 * block_n)?,
            flags: ctx.stream.alloc_zeros(ctas)?,
            dense: DeviceMatrix {
                data: ctx.stream.alloc_zeros(max_values)?,
                rows: 0,
                cols: 0,
            },
        })
    }
}

/// `out = x @ weight^T`. Up to sixteen rows run the TileLang GEMM at the
/// bucket that holds them, which reads and writes the bucket's padding rows,
/// so both buffers must have room for it; wider steps dequantize the weight
/// and run the dense GEMM. A gate|up weight's TileLang GEMM writes
/// gelu(gate) * up, `rows / 2` wide; its dense GEMM writes both, stacked.
pub fn gemma4_w4a16_gemm_into(
    ctx: &DeviceContext,
    weight: &W4a16Matrix,
    x: &HiddenStates,
    scratch: &mut W4a16Scratch,
    out: &mut HiddenStates,
) -> Result<()> {
    let rows = x.seq_len;
    let width = if weight.gelu_mul && W4a16Matrix::runs_tilelang(rows) {
        weight.rows / 2
    } else {
        weight.rows
    };
    ensure!(
        x.hidden_dim == weight.cols && out.hidden_dim == width && out.seq_len == x.seq_len,
        "W4A16 {} x {} cannot map {} x {} into {} x {}",
        weight.rows,
        weight.cols,
        x.seq_len,
        x.hidden_dim,
        out.seq_len,
        out.hidden_dim
    );
    if rows == 0 {
        return Ok(());
    }
    let Some(&bucket) = BUCKETS.iter().find(|&&b| b >= rows) else {
        weight.dequant_into(ctx, &mut scratch.dense)?;
        return crate::ops::gemm_rows_into_checked(ctx, &scratch.dense, 0, weight.rows, x, out);
    };
    ensure!(
        x.data.len() >= bucket * weight.cols && out.data.len() >= bucket * width,
        "W4A16 at {rows} rows runs the {bucket}-row GEMM, which needs {} input and {} output \
         values; the buffers hold {} and {}",
        bucket * weight.cols,
        bucket * width,
        x.data.len(),
        out.data.len()
    );
    let (xp, _g0) = x.data.device_ptr(&ctx.stream);
    let (wq, _g1) = weight.wq.device_ptr(&ctx.stream);
    let (sq, _g2) = weight.sq.device_ptr(&ctx.stream);
    let (y, _g3) = out.data.device_ptr_mut(&ctx.stream);
    let (part, _g4) = scratch.part.device_ptr_mut(&ctx.stream);
    let (flags, _g5) = scratch.flags.device_ptr_mut(&ctx.stream);
    let (fin_off, _g6) = weight.fin_off.device_ptr(&ctx.stream);
    let (fin_list, _g7) = weight.fin_list.device_ptr(&ctx.stream);
    let status = unsafe {
        ffi::gemma4_w4a16_gemm(
            xp as *mut core::ffi::c_void,
            wq as *mut i32,
            sq as *mut i32,
            y as *mut core::ffi::c_void,
            part as *mut f32,
            flags as *mut i32,
            fin_off as *mut i32,
            fin_list as *mut i32,
            i32::try_from(weight.rows)?,
            i32::try_from(weight.cols)?,
            bucket as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    ensure!(
        status == 0,
        "W4A16 GEMM {} x {} at {bucket} rows failed (cudaError {status})",
        weight.rows,
        weight.cols
    );
    Ok(())
}
