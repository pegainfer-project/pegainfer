//! BF16 eager operators for the first Ascend Qwen3 execution path.
//!
//! Every public operator takes the device lock exactly once (`Device::run`) and
//! submits on that stream. Executor ownership follows CANN: the non-repeatable
//! executor returned by `*GetWorkspaceSize` is released by CANN after the op
//! runs and must not be destroyed (`aclDestroyAclOpExecutor` double-frees on
//! CANN 9.2.1). The GEMM-family ops cache a *repeatable* executor instead; CANN
//! does not reclaim those, so `Device` destroys them on drop.
use std::collections::hash_map::DefaultHasher;
use std::ffi::c_void;
use std::hash::Hash;
use std::hash::Hasher;

use anyhow::Result;
use anyhow::ensure;
use half::bf16;

use crate::device::Buffer;
use crate::device::Device;
use crate::device::DeviceRepr;
use crate::sys;

fn dims(values: &[usize]) -> Result<usize> {
    ensure!(
        values.iter().all(|&v| v > 0 && i32::try_from(v).is_ok()),
        "invalid Ascend operator shape {values:?}"
    );
    values.iter().try_fold(1usize, |n, &v| {
        n.checked_mul(v)
            .ok_or_else(|| anyhow::anyhow!("shape overflow"))
    })
}

fn check<T: DeviceRepr>(device: &Device, buffer: &Buffer<T>, len: usize) -> Result<()> {
    ensure!(
        device.owns(buffer),
        "operator received a buffer from another device/stream"
    );
    ensure!(
        buffer.len() >= len,
        "operator buffer too short: {} < {len}",
        buffer.len()
    );
    Ok(())
}

fn key(op: &str, shapes: &[i64], ptrs: &[usize]) -> u64 {
    let mut h = DefaultHasher::new();
    op.hash(&mut h);
    shapes.hash(&mut h);
    for p in ptrs {
        p.hash(&mut h);
    }
    h.finish()
}

fn strides(shape: &[i64]) -> Vec<i64> {
    let mut out = vec![1; shape.len()];
    let mut acc = 1i64;
    for i in (0..shape.len()).rev() {
        out[i] = acc;
        acc *= shape[i];
    }
    out
}

/// ACLNN keeps pointers to the shape/stride arrays, so they must outlive the
/// tensor descriptor. `Tensors` owns both for the duration of one call.
struct Tensors {
    handles: Vec<*mut sys::aclTensor>,
    arrays: Vec<Vec<i64>>,
}

impl Tensors {
    fn new() -> Self {
        Self {
            handles: Vec::new(),
            arrays: Vec::new(),
        }
    }
    fn mk(
        &mut self,
        shape: &[i64],
        dtype: u32,
        ptr: *mut c_void,
        what: &str,
    ) -> Result<*mut sys::aclTensor> {
        let stride = strides(shape);
        self.mk_strided(shape, &stride, dtype, ptr, what)
    }

    fn mk_strided(
        &mut self,
        shape: &[i64],
        stride: &[i64],
        dtype: u32,
        ptr: *mut c_void,
        what: &str,
    ) -> Result<*mut sys::aclTensor> {
        ensure!(shape.len() == stride.len(), "stride rank mismatch");
        self.arrays.push(shape.to_vec());
        self.arrays.push(stride.to_vec());
        let n = self.arrays.len();
        let dims = self.arrays[n - 2].as_ptr();
        let st = self.arrays[n - 1].as_ptr();
        let t = {
            sys::aclCreateTensor(
                dims,
                shape.len(),
                dtype,
                st,
                0,
                sys::ACL_FORMAT_ND,
                dims,
                shape.len(),
                ptr,
            )
        };
        ensure!(!t.is_null(), "aclCreateTensor returned null for {what}");
        self.handles.push(t);
        Ok(t)
    }
    fn mk_at(
        &mut self,
        shape: &[i64],
        dtype: u32,
        base: *mut c_void,
        offset_elems: usize,
        elem_size: usize,
        what: &str,
    ) -> Result<*mut sys::aclTensor> {
        let ptr = unsafe { base.cast::<u8>().add(offset_elems * elem_size) }.cast();
        self.mk(shape, dtype, ptr, what)
    }
}

impl Drop for Tensors {
    fn drop(&mut self) {
        for &t in &self.handles {
            {
                sys::aclDestroyTensor(t);
            }
        }
    }
}

impl Tensors {
    /// Move the descriptors out so they can be dropped after the stream is
    /// synchronized, instead of destroying them immediately after enqueue.
    fn release(mut self) -> (Vec<usize>, Vec<Vec<i64>>) {
        let handles = self.handles.drain(..).map(|t| t as usize).collect();
        let arrays = self.arrays.drain(..).collect();
        (handles, arrays)
    }
}

/// Descriptors for cached executors are destroyed with the device; their
/// shape/stride arrays are leaked deliberately (bounded, one per shape).
fn mk_tensor_owned(shape: &[i64], dtype: u32, ptr: *mut c_void) -> *mut sys::aclTensor {
    let dims: &'static [i64] = Box::leak(shape.to_vec().into_boxed_slice());
    let st: &'static [i64] = Box::leak(strides(shape).into_boxed_slice());
    {
        sys::aclCreateTensor(
            dims.as_ptr(),
            shape.len(),
            dtype,
            st.as_ptr(),
            0,
            sys::ACL_FORMAT_ND,
            dims.as_ptr(),
            shape.len(),
            ptr,
        )
    }
}

fn mk_tensor_owned_nz(
    view: &[i64],
    storage: &[i64],
    dtype: u32,
    ptr: *mut c_void,
) -> *mut sys::aclTensor {
    let view_dims: &'static [i64] = Box::leak(view.to_vec().into_boxed_slice());
    let storage_dims: &'static [i64] = Box::leak(storage.to_vec().into_boxed_slice());
    {
        sys::aclCreateTensor(
            view_dims.as_ptr(),
            view.len(),
            dtype,
            std::ptr::null(),
            0,
            sys::ACL_FORMAT_FRACTAL_NZ,
            storage_dims.as_ptr(),
            storage.len(),
            ptr,
        )
    }
}

struct Scalar(*mut sys::aclScalar);
impl Drop for Scalar {
    fn drop(&mut self) {
        {
            sys::aclDestroyScalar(self.0);
        }
    }
}

/// Run a cached executor on an already-bound stream.
fn run_cached(
    d: &Device,
    stream: *mut c_void,
    cache_key: u64,
    create: impl FnOnce() -> Result<(usize, u64)>,
    launch: impl FnOnce(*mut c_void, u64, *mut sys::aclOpExecutor, *mut c_void) -> i32,
) -> Result<()> {
    let (executor, ws) = d.cached_gemm(cache_key, create)?;
    let workspace = d.workspace_locked(ws)?;
    let status = launch(workspace, ws, executor as *mut sys::aclOpExecutor, stream);
    crate::device::check(status)
}

/// Run a one-shot ACLNN op. CANN owns the non-repeatable executor returned by
/// `*GetWorkspaceSize` and releases it after execution; destroying it here
/// double-frees on CANN 9.2.1 (reproduced for MatMul, AddRmsNorm and IFA in
/// `aclnn-smoke/smoke_exec_lifecycle.cpp`). A 1000-iteration loop keeps RSS
/// flat, so not destroying it leaks nothing. The caller owns and frees the
/// workspace.
fn run_oneshot(
    d: &Device,
    stream: *mut c_void,
    prepare: impl FnOnce(&mut u64, &mut *mut sys::aclOpExecutor) -> i32,
    launch: impl FnOnce(*mut c_void, u64, *mut sys::aclOpExecutor, *mut c_void) -> i32,
) -> Result<()> {
    let mut ws = 0u64;
    let mut executor = std::ptr::null_mut();
    crate::device::check(prepare(&mut ws, &mut executor))?;
    let workspace = d.workspace_locked(ws)?;
    let status = launch(workspace, ws, executor, stream);
    crate::device::check(status)
}

/// Gather rows. Every index must be below `vocab`.
pub fn embedding(
    d: &Device,
    w: &Buffer<bf16>,
    ids: &Buffer<u32>,
    out: &mut Buffer<bf16>,
    rows: usize,
    h: usize,
    vocab: usize,
) -> Result<()> {
    check(d, w, dims(&[vocab, h])?)?;
    check(d, ids, rows)?;
    check(d, out, dims(&[rows, h])?)?;
    let cache_key = key(
        "embedding",
        &[rows as i64, h as i64, vocab as i64],
        &[
            w.as_ptr() as usize,
            ids.as_ptr() as usize,
            out.as_mut_ptr() as usize,
        ],
    );
    d.run(|stream| {
        run_cached(
            d,
            stream,
            cache_key,
            || {
                let tw = mk_tensor_owned(
                    &[vocab as i64, h as i64],
                    sys::ACL_BF16,
                    w.as_ptr().cast_mut().cast(),
                );
                let ti = mk_tensor_owned(
                    &[rows as i64],
                    sys::ACL_INT32,
                    ids.as_ptr().cast_mut().cast(),
                );
                let to = mk_tensor_owned(
                    &[rows as i64, h as i64],
                    sys::ACL_BF16,
                    out.as_mut_ptr().cast(),
                );
                ensure!(
                    !tw.is_null() && !ti.is_null() && !to.is_null(),
                    "embedding tensor"
                );
                let mut ws = 0u64;
                let mut ex = std::ptr::null_mut();
                crate::device::check(sys::aclnnEmbeddingGetWorkspaceSize(
                    tw, ti, to, &mut ws, &mut ex,
                ))?;
                crate::device::check(sys::aclSetAclOpExecutorRepeatable(ex))?;
                d.keep_tensors(vec![tw, ti, to]);
                Ok((ex as usize, ws))
            },
            |p, n, ex, st| sys::aclnnEmbedding(p, n, ex, st),
        )
    })
}

/// `out = x @ w` with `x:[rows,k]` and a pre-transposed `w:[k,n]`.
pub fn gemm(
    d: &Device,
    w: &Buffer<bf16>,
    x: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    rows: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    check(d, w, dims(&[k, n])?)?;
    check(d, x, dims(&[rows, k])?)?;
    check(d, out, dims(&[rows, n])?)?;
    let cache_key = key(
        "mm",
        &[rows as i64, n as i64, k as i64],
        &[
            w.as_ptr() as usize,
            x.as_ptr() as usize,
            out.as_ptr() as usize,
        ],
    );
    d.run(|stream| {
        let (executor, ws) = d.cached_gemm(cache_key, || {
            let tx = mk_tensor_owned(
                &[rows as i64, k as i64],
                sys::ACL_BF16,
                x.as_ptr().cast_mut().cast(),
            );
            let tw = mk_tensor_owned(
                &[k as i64, n as i64],
                sys::ACL_BF16,
                w.as_ptr().cast_mut().cast(),
            );
            let to = mk_tensor_owned(
                &[rows as i64, n as i64],
                sys::ACL_BF16,
                out.as_mut_ptr().cast(),
            );
            ensure!(
                !tx.is_null() && !tw.is_null() && !to.is_null(),
                "gemm tensor"
            );
            let mut ws = 0u64;
            let mut ex = std::ptr::null_mut();
            crate::device::check(sys::aclnnMatmulGetWorkspaceSize(
                tx, tw, to, 0, &mut ws, &mut ex,
            ))?;
            crate::device::check(sys::aclSetAclOpExecutorRepeatable(ex))?;
            // The repeatable executor keeps references to these descriptors;
            // `Device::drop` destroys the executor first, then the tensors.
            d.keep_tensors(vec![tx, tw, to]);
            Ok((ex as usize, ws))
        })?;
        let workspace = d.workspace_locked(ws)?;
        let ex = executor as *mut sys::aclOpExecutor;
        let status = { sys::aclnnMatmul(workspace, ws, ex, stream) };
        crate::device::check(status)
    })
}

pub fn gemm_nz(
    d: &Device,
    w: &Buffer<bf16>,
    x: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    rows: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let nblocks = n.div_ceil(16);
    let kpad = k.div_ceil(16) * 16;
    check(d, w, nblocks * kpad * 16)?;
    check(d, x, dims(&[rows, k])?)?;
    check(d, out, dims(&[rows, n])?)?;
    let cache_key = key(
        "mm",
        &[rows as i64, n as i64, k as i64],
        &[
            w.as_ptr() as usize,
            x.as_ptr() as usize,
            out.as_ptr() as usize,
        ],
    );
    d.run(|stream| {
        let (executor, ws) = d.cached_gemm(cache_key, || {
            let tx = mk_tensor_owned(
                &[rows as i64, k as i64],
                sys::ACL_BF16,
                x.as_ptr().cast_mut().cast(),
            );
            let tw = mk_tensor_owned_nz(
                &[k as i64, n as i64],
                &[1, nblocks as i64, kpad as i64, 16],
                sys::ACL_BF16,
                w.as_ptr().cast_mut().cast(),
            );
            let to = mk_tensor_owned(
                &[rows as i64, n as i64],
                sys::ACL_BF16,
                out.as_mut_ptr().cast(),
            );
            ensure!(
                !tx.is_null() && !tw.is_null() && !to.is_null(),
                "gemm tensor"
            );
            let mut ws = 0u64;
            let mut ex = std::ptr::null_mut();
            crate::device::check(sys::aclnnMatmulWeightNzGetWorkspaceSize(
                tx, tw, to, 0, &mut ws, &mut ex,
            ))?;
            crate::device::check(sys::aclSetAclOpExecutorRepeatable(ex))?;
            // The repeatable executor keeps references to these descriptors;
            // `Device::drop` destroys the executor first, then the tensors.
            d.keep_tensors(vec![tx, tw, to]);
            Ok((ex as usize, ws))
        })?;
        let workspace = d.workspace_locked(ws)?;
        let ex = executor as *mut sys::aclOpExecutor;
        let status = { sys::aclnnMatmulWeightNz(workspace, ws, ex, stream) };
        crate::device::check(status)
    })
}

pub fn norm(
    d: &Device,
    x: &Buffer<bf16>,
    w: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    rows: usize,
    h: usize,
    eps: f32,
) -> Result<()> {
    let n = dims(&[rows, h])?;
    check(d, x, n)?;
    check(d, w, h)?;
    check(d, out, n)?;
    ensure!(eps.is_finite() && eps > 0., "invalid norm epsilon");
    d.run(|stream| {
        let rstd = d.scratch_locked(rows * 4)?;
        let cache_key = key(
            "norm",
            &[rows as i64, h as i64],
            &[
                x.as_ptr() as usize,
                w.as_ptr() as usize,
                out.as_mut_ptr() as usize,
                rstd as usize,
                eps.to_bits() as usize,
            ],
        );
        run_cached(
            d,
            stream,
            cache_key,
            || {
                let tx = mk_tensor_owned(
                    &[rows as i64, h as i64],
                    sys::ACL_BF16,
                    x.as_ptr().cast_mut().cast(),
                );
                let tw = mk_tensor_owned(&[h as i64], sys::ACL_BF16, w.as_ptr().cast_mut().cast());
                let ty = mk_tensor_owned(
                    &[rows as i64, h as i64],
                    sys::ACL_BF16,
                    out.as_mut_ptr().cast(),
                );
                let tr = mk_tensor_owned(&[rows as i64, 1], sys::ACL_FLOAT, rstd);
                ensure!(
                    !tx.is_null() && !tw.is_null() && !ty.is_null() && !tr.is_null(),
                    "norm tensor"
                );
                let mut ws = 0u64;
                let mut ex = std::ptr::null_mut();
                crate::device::check(sys::aclnnRmsNormGetWorkspaceSize(
                    tx,
                    tw,
                    f64::from(eps),
                    ty,
                    tr,
                    &mut ws,
                    &mut ex,
                ))?;
                crate::device::check(sys::aclSetAclOpExecutorRepeatable(ex))?;
                d.keep_tensors(vec![tx, tw, ty, tr]);
                Ok((ex as usize, ws))
            },
            |p, n, ex, st| sys::aclnnRmsNorm(p, n, ex, st),
        )
    })
}

/// `x = a + b` and `y = rms_norm(x) * gamma` in one ACLNN op.
/// The AddRmsNorm executor is not destroyed (CANN 9.2.1 double-frees on that
/// path); its descriptors are deferred like the other one-shot ops.
#[allow(clippy::too_many_arguments)]
pub fn add_norm(
    d: &Device,
    a: &Buffer<bf16>,
    b: &Buffer<bf16>,
    gamma: &Buffer<bf16>,
    x_out: &mut Buffer<bf16>,
    y_out: &mut Buffer<bf16>,
    rows: usize,
    h: usize,
    eps: f32,
) -> Result<()> {
    let n = dims(&[rows, h])?;
    check(d, a, n)?;
    check(d, b, n)?;
    check(d, gamma, h)?;
    check(d, x_out, n)?;
    check(d, y_out, n)?;
    ensure!(eps.is_finite() && eps > 0., "invalid norm epsilon");
    d.run(|stream| {
        let mut t = Tensors::new();
        let rstd = d.scratch_locked(rows * 4)?;
        let ta = t.mk(
            &[rows as i64, h as i64],
            sys::ACL_BF16,
            a.as_ptr().cast_mut().cast(),
            "add_norm a",
        )?;
        let tb = t.mk(
            &[rows as i64, h as i64],
            sys::ACL_BF16,
            b.as_ptr().cast_mut().cast(),
            "add_norm b",
        )?;
        let tg = t.mk(
            &[h as i64],
            sys::ACL_BF16,
            gamma.as_ptr().cast_mut().cast(),
            "add_norm gamma",
        )?;
        let tx = t.mk(
            &[rows as i64, h as i64],
            sys::ACL_BF16,
            x_out.as_mut_ptr().cast(),
            "add_norm x out",
        )?;
        let ty = t.mk(
            &[rows as i64, h as i64],
            sys::ACL_BF16,
            y_out.as_mut_ptr().cast(),
            "add_norm y out",
        )?;
        let tr = t.mk(&[rows as i64, 1], sys::ACL_FLOAT, rstd, "add_norm rstd")?;
        let result = run_oneshot(
            d,
            stream,
            |ws, ex| {
                sys::aclnnAddRmsNormGetWorkspaceSize(ta, tb, tg, f64::from(eps), ty, tr, tx, ws, ex)
            },
            |p, n, ex, st| sys::aclnnAddRmsNorm(p, n, ex, st),
        );
        let (handles, arrays) = t.release();
        d.defer_tensors(handles, arrays);
        result
    })
}

pub fn add(
    d: &Device,
    a: &Buffer<bf16>,
    b: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    n: usize,
) -> Result<()> {
    check(d, a, n)?;
    check(d, b, n)?;
    check(d, out, n)?;
    ensure!(n > 0, "empty add");
    let mut one = 1.0f32;
    let alpha = Scalar(sys::aclCreateScalar((&raw mut one).cast(), sys::ACL_FLOAT));
    ensure!(!alpha.0.is_null(), "aclCreateScalar returned null");
    let cache_key = key(
        "add",
        &[n as i64],
        &[
            a.as_ptr() as usize,
            b.as_ptr() as usize,
            out.as_ptr() as usize,
        ],
    );
    d.run(|stream| {
        run_cached(
            d,
            stream,
            cache_key,
            || {
                let ta = mk_tensor_owned(&[n as i64], sys::ACL_BF16, a.as_ptr().cast_mut().cast());
                let tb = mk_tensor_owned(&[n as i64], sys::ACL_BF16, b.as_ptr().cast_mut().cast());
                let to = mk_tensor_owned(&[n as i64], sys::ACL_BF16, out.as_mut_ptr().cast());
                ensure!(
                    !ta.is_null() && !tb.is_null() && !to.is_null(),
                    "add tensor"
                );
                let mut ws = 0u64;
                let mut ex = std::ptr::null_mut();
                crate::device::check(sys::aclnnAddGetWorkspaceSize(
                    ta, tb, alpha.0, to, &mut ws, &mut ex,
                ))?;
                crate::device::check(sys::aclSetAclOpExecutorRepeatable(ex))?;
                d.keep_tensors(vec![ta, tb, to]);
                Ok((ex as usize, ws))
            },
            |p, ws, ex, s| sys::aclnnAdd(p, ws, ex, s),
        )
    })
}

/// Cached `out = a * b`. The repeatable executor and its descriptors are
/// destroyed with the device; all addresses must stay stable.
fn mul_locked(
    d: &Device,
    stream: *mut c_void,
    a: *mut c_void,
    b: *mut c_void,
    out: *mut c_void,
    n: usize,
) -> Result<()> {
    let cache_key = key("mul", &[n as i64], &[a as usize, b as usize, out as usize]);
    let (executor, ws) = d.cached_gemm(cache_key, || {
        let ta = mk_tensor_owned(&[n as i64], sys::ACL_BF16, a);
        let tb = mk_tensor_owned(&[n as i64], sys::ACL_BF16, b);
        let to = mk_tensor_owned(&[n as i64], sys::ACL_BF16, out);
        ensure!(
            !ta.is_null() && !tb.is_null() && !to.is_null(),
            "mul tensor"
        );
        let mut ws = 0u64;
        let mut ex = std::ptr::null_mut();
        crate::device::check(sys::aclnnMulGetWorkspaceSize(ta, tb, to, &mut ws, &mut ex))?;
        crate::device::check(sys::aclSetAclOpExecutorRepeatable(ex))?;
        d.keep_tensors(vec![ta, tb, to]);
        Ok((ex as usize, ws))
    })?;
    let workspace = d.workspace_locked(ws)?;
    let ex = executor as *mut sys::aclOpExecutor;
    let status = sys::aclnnMul(workspace, ws, ex, stream);
    crate::device::check(status)
}

/// Clipped SwiGLU over a concatenated `[rows, 2i]` gate/up matrix.
pub fn swiglu(
    d: &Device,
    x: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    rows: usize,
    i: usize,
) -> Result<()> {
    check(d, x, dims(&[rows, 2 * i])?)?;
    check(d, out, dims(&[rows, i])?)?;
    d.run(|stream| {
        let mut t = Tensors::new();
        let tx = t.mk(
            &[rows as i64, (2 * i) as i64],
            sys::ACL_BF16,
            x.as_ptr().cast_mut().cast(),
            "swiglu in",
        )?;
        let to = t.mk(
            &[rows as i64, i as i64],
            sys::ACL_BF16,
            out.as_mut_ptr().cast(),
            "swiglu out",
        )?;
        let result = run_oneshot(
            d,
            stream,
            |ws, ex| {
                sys::aclnnClippedSwigluGetWorkspaceSize(
                    tx,
                    std::ptr::null(),
                    1,
                    1.0,
                    1e10,
                    0.0,
                    false,
                    to,
                    ws,
                    ex,
                )
            },
            |p, n, ex, st| sys::aclnnClippedSwiglu(p, n, ex, st),
        );
        let (handles, arrays) = t.release();
        d.defer_tensors(handles, arrays);
        result
    })
}

/// `out = silu(a) * b`, used for the SwiGLU MLP.
pub fn silu(
    d: &Device,
    a: &Buffer<bf16>,
    b: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    n: usize,
) -> Result<()> {
    check(d, a, n)?;
    check(d, b, n)?;
    check(d, out, n)?;
    ensure!(n > 0, "empty activation");
    let a_ptr = a.as_ptr().cast_mut().cast();
    let b_ptr = b.as_ptr().cast_mut().cast();
    let out_ptr = out.as_mut_ptr().cast();
    d.run(|stream| {
        // The activation lands in device-lifetime scratch so the cached
        // executors see stable addresses.
        let activated = d.scratch_locked(n * 2)?;
        let cache_key = key(
            "silu",
            &[n as i64],
            &[a_ptr as usize, activated as usize, out_ptr as usize],
        );
        run_cached(
            d,
            stream,
            cache_key,
            || {
                let ta = mk_tensor_owned(&[n as i64], sys::ACL_BF16, a_ptr);
                let tmp = mk_tensor_owned(&[n as i64], sys::ACL_BF16, activated);
                ensure!(!ta.is_null() && !tmp.is_null(), "silu tensor");
                let mut ws = 0u64;
                let mut ex = std::ptr::null_mut();
                crate::device::check(sys::aclnnSiluGetWorkspaceSize(ta, tmp, &mut ws, &mut ex))?;
                crate::device::check(sys::aclSetAclOpExecutorRepeatable(ex))?;
                d.keep_tensors(vec![ta, tmp]);
                Ok((ex as usize, ws))
            },
            |p, ws, ex, s| sys::aclnnSilu(p, ws, ex, s),
        )?;
        mul_locked(d, stream, activated, b_ptr, out_ptr, n)
    })
}

/// Apply RoPE to Q and K in one call. `cos`/`sin` are `[context, dim]` tables;
/// only the rows named by `positions` are gathered.
#[allow(clippy::too_many_arguments)]
pub fn rope(
    d: &Device,
    q: &mut Buffer<bf16>,
    k: &mut Buffer<bf16>,
    positions: &Buffer<i32>,
    cos: &Buffer<bf16>,
    sin: &Buffer<bf16>,
    rows: usize,
    q_heads: usize,
    kv_heads: usize,
    dim: usize,
    context: usize,
) -> Result<()> {
    check(d, q, dims(&[rows, q_heads, dim])?)?;
    check(d, k, dims(&[rows, kv_heads, dim])?)?;
    check(d, positions, rows)?;
    check(d, cos, dims(&[context, dim])?)?;
    check(d, sin, dims(&[context, dim])?)?;
    ensure!(dim.is_multiple_of(2), "invalid rotary geometry");
    d.run(|stream| {
        let mut t = Tensors::new();
        // cos and sin have identical sizes; a single scratch block split in two
        // prevents the size-keyed scratch cache from handing back the same
        // address for both (which would make RoPE read sin as cos).
        let pair = d.scratch_locked(rows * dim * 4)?;
        let cos_rows = pair;
        let sin_rows = unsafe { pair.cast::<u8>().add(rows * dim * 2) }.cast();
        let tcos = t.mk(
            &[context as i64, dim as i64],
            sys::ACL_BF16,
            cos.as_ptr().cast_mut().cast(),
            "rope cos table",
        )?;
        let tsin = t.mk(
            &[context as i64, dim as i64],
            sys::ACL_BF16,
            sin.as_ptr().cast_mut().cast(),
            "rope sin table",
        )?;
        let tp = t.mk(
            &[rows as i64],
            sys::ACL_INT32,
            positions.as_ptr().cast_mut().cast(),
            "rope positions",
        )?;
        let crows = t.mk(
            &[rows as i64, dim as i64],
            sys::ACL_BF16,
            cos_rows,
            "rope cos rows",
        )?;
        let srows = t.mk(
            &[rows as i64, dim as i64],
            sys::ACL_BF16,
            sin_rows,
            "rope sin rows",
        )?;
        run_oneshot(
            d,
            stream,
            |ws, ex| sys::aclnnEmbeddingGetWorkspaceSize(tcos, tp, crows, ws, ex),
            |p, n, ex, s| sys::aclnnEmbedding(p, n, ex, s),
        )?;
        run_oneshot(
            d,
            stream,
            |ws, ex| sys::aclnnEmbeddingGetWorkspaceSize(tsin, tp, srows, ws, ex),
            |p, n, ex, s| sys::aclnnEmbedding(p, n, ex, s),
        )?;
        let tq = t.mk(
            &[1, rows as i64, q_heads as i64, dim as i64],
            sys::ACL_BF16,
            q.as_mut_ptr().cast(),
            "rope q",
        )?;
        let tk = t.mk(
            &[1, rows as i64, kv_heads as i64, dim as i64],
            sys::ACL_BF16,
            k.as_mut_ptr().cast(),
            "rope k",
        )?;
        let tc = t.mk(
            &[1, rows as i64, 1, dim as i64],
            sys::ACL_BF16,
            cos_rows,
            "rope cos view",
        )?;
        let ts = t.mk(
            &[1, rows as i64, 1, dim as i64],
            sys::ACL_BF16,
            sin_rows,
            "rope sin view",
        )?;
        let mode = c"half";
        let result = run_oneshot(
            d,
            stream,
            |ws, ex| {
                sys::aclnnApplyRotaryPosEmbV2GetWorkspaceSize(
                    tq,
                    tk,
                    tc,
                    ts,
                    1,
                    mode.as_ptr().cast_mut(),
                    ws,
                    ex,
                )
            },
            |p, n, ex, s| sys::aclnnApplyRotaryPosEmbV2(p, n, ex, s),
        );
        let (handles, arrays) = t.release();
        d.defer_tensors(handles, arrays);
        result
    })
}

#[derive(Clone, Copy)]
pub struct AttentionShape {
    pub rows: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub dim: usize,
    pub context: usize,
    pub capacity: usize,
}

/// Append KV rows. `slots` is the host view of the same rows as `k`.
/// Cache layout is `[slot, kv_head, context, dim]` so a slot view is directly
/// usable as an IFA `[1, kv_head, context, dim]` cache.
#[allow(clippy::too_many_arguments)]
pub fn cache(
    d: &Device,
    k: &Buffer<bf16>,
    v: &Buffer<bf16>,
    keys: &mut Buffer<bf16>,
    values: &mut Buffer<bf16>,
    slots: &[i32],
    positions: &Buffer<i32>,
    s: AttentionShape,
) -> Result<()> {
    let n = dims(&[s.rows, s.kv_heads, s.dim])?;
    let cache = dims(&[s.capacity, s.kv_heads, s.context, s.dim])?;
    for b in [k, v] {
        check(d, b, n)?;
    }
    check(d, keys, cache)?;
    check(d, values, cache)?;
    check(d, positions, s.rows)?;
    ensure!(slots.len() == s.rows, "slot/row mismatch");
    let row_elems = s.kv_heads * s.dim;
    let slot_elems = s.kv_heads * s.context * s.dim;
    for row in 0..s.rows {
        let slot = usize::try_from(slots[row])?;
        ensure!(slot < s.capacity, "slot out of range");
        d.run(|stream| {
            let mut t = Tensors::new();
            let kc = t.mk_at(
                &[s.kv_heads as i64, s.context as i64, s.dim as i64],
                sys::ACL_BF16,
                keys.as_mut_ptr().cast(),
                slot * slot_elems,
                2,
                "cache k",
            )?;
            let vc = t.mk_at(
                &[s.kv_heads as i64, s.context as i64, s.dim as i64],
                sys::ACL_BF16,
                values.as_mut_ptr().cast(),
                slot * slot_elems,
                2,
                "cache v",
            )?;
            let src_k = t.mk_at(
                &[s.kv_heads as i64, 1, s.dim as i64],
                sys::ACL_BF16,
                k.as_ptr().cast_mut().cast(),
                row * row_elems,
                2,
                "cache src k",
            )?;
            let src_v = t.mk_at(
                &[s.kv_heads as i64, 1, s.dim as i64],
                sys::ACL_BF16,
                v.as_ptr().cast_mut().cast(),
                row * row_elems,
                2,
                "cache src v",
            )?;
            let index = t.mk_at(
                &[1],
                sys::ACL_INT32,
                positions.as_ptr().cast_mut().cast(),
                row,
                4,
                "cache index",
            )?;
            let kc_out = t.mk_at(
                &[s.kv_heads as i64, s.context as i64, s.dim as i64],
                sys::ACL_BF16,
                keys.as_mut_ptr().cast(),
                slot * slot_elems,
                2,
                "cache k out",
            )?;
            let vc_out = t.mk_at(
                &[s.kv_heads as i64, s.context as i64, s.dim as i64],
                sys::ACL_BF16,
                values.as_mut_ptr().cast(),
                slot * slot_elems,
                2,
                "cache v out",
            )?;
            run_oneshot(
                d,
                stream,
                |ws, ex| sys::aclnnIndexCopyGetWorkspaceSize(kc, 1, index, src_k, kc_out, ws, ex),
                |p, n, ex, s| sys::aclnnIndexCopy(p, n, ex, s),
            )?;
            let result = run_oneshot(
                d,
                stream,
                |ws, ex| sys::aclnnIndexCopyGetWorkspaceSize(vc, 1, index, src_v, vc_out, ws, ex),
                |p, n, ex, s| sys::aclnnIndexCopy(p, n, ex, s),
            );
            let (handles, arrays) = t.release();
            d.defer_tensors(handles, arrays);
            result
        })?;
    }
    Ok(())
}

/// One row of decode-style IFA against its slot's KV prefix.
#[allow(clippy::too_many_arguments)]
fn ifa_row(
    d: &Device,
    q: &Buffer<bf16>,
    keys: &Buffer<bf16>,
    values: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    slot: usize,
    pos: usize,
    s: AttentionShape,
    row: usize,
) -> Result<()> {
    let row_elems = s.heads * s.dim;
    let kv_row_elems = s.kv_heads * s.dim;
    let slot_elems = s.context * kv_row_elems;
    let scale = 1.0 / (s.dim as f64).sqrt();
    let seq_len = i64::try_from(pos + 1)?;
    let q_ptr = q.as_ptr() as usize + row * row_elems * 2;
    let out_ptr = out.as_mut_ptr() as usize + row * row_elems * 2;
    let keys_ptr = keys.as_ptr() as usize + slot * slot_elems * 2;
    let values_ptr = values.as_ptr() as usize + slot * slot_elems * 2;
    let cache_key = key(
        "ifa",
        &[
            s.heads as i64,
            s.kv_heads as i64,
            s.dim as i64,
            s.context as i64,
            slot as i64,
            row as i64,
        ],
        &[q_ptr, out_ptr, keys_ptr, values_ptr],
    );
    let descs = d.cached_descs(cache_key, || {
        let tq = mk_tensor_owned(
            &[1, s.heads as i64, 1, s.dim as i64],
            sys::ACL_BF16,
            q_ptr as *mut c_void,
        );
        let tc = mk_tensor_owned(
            &[1, s.kv_heads as i64, s.context as i64, s.dim as i64],
            sys::ACL_BF16,
            keys_ptr as *mut c_void,
        );
        let tv = mk_tensor_owned(
            &[1, s.kv_heads as i64, s.context as i64, s.dim as i64],
            sys::ACL_BF16,
            values_ptr as *mut c_void,
        );
        let to = mk_tensor_owned(
            &[1, s.heads as i64, 1, s.dim as i64],
            sys::ACL_BF16,
            out_ptr as *mut c_void,
        );
        ensure!(
            !tq.is_null() && !tc.is_null() && !tv.is_null() && !to.is_null(),
            "attention tensor"
        );
        let karr = [tc as *const sys::aclTensor];
        let varr = [tv as *const sys::aclTensor];
        let klist = sys::aclCreateTensorList(karr.as_ptr(), 1);
        let vlist = sys::aclCreateTensorList(varr.as_ptr(), 1);
        ensure!(!klist.is_null() && !vlist.is_null(), "tensor list");
        // CANN 9.2.1 double-frees when a tensor list and a tensor it
        // contains are both destroyed, so `tensors` keeps only the
        // standalone query/output descriptors; the lists own K/V.
        Ok(crate::device::Descs {
            tensors: vec![tq as usize, to as usize],
            lists: vec![klist as usize, vlist as usize],
        })
    })?;
    let tq = descs.tensors[0] as *mut sys::aclTensor;
    let to = descs.tensors[1] as *mut sys::aclTensor;
    let klist = descs.lists[0] as *mut sys::aclTensorList;
    let vlist = descs.lists[1] as *mut sys::aclTensorList;
    let actual = d.int_array(seq_len)? as *mut sys::aclIntArray;
    d.run(|stream| {
        let layout = c"BNSD";
        run_oneshot(
            d,
            stream,
            |ws, ex| {
                sys::aclnnIncreFlashAttentionV4GetWorkspaceSize(
                    tq,
                    klist,
                    vlist,
                    std::ptr::null(),
                    std::ptr::null(),
                    actual,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    s.heads as i64,
                    scale,
                    layout.as_ptr().cast_mut(),
                    s.kv_heads as i64,
                    0,
                    1,
                    to,
                    ws,
                    ex,
                )
            },
            |p, n, ex, st| sys::aclnnIncreFlashAttentionV4(p, n, ex, st),
        )
    })
}

/// Causal attention for each row against its slot's KV prefix.
#[allow(clippy::too_many_arguments)]
pub fn attention(
    d: &Device,
    q: &Buffer<bf16>,
    keys: &Buffer<bf16>,
    values: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    slots: &[i32],
    positions: &[i32],
    s: AttentionShape,
) -> Result<()> {
    let n = dims(&[s.rows, s.heads, s.dim])?;
    let cache = dims(&[s.capacity, s.context, s.kv_heads, s.dim])?;
    ensure!(
        s.heads.is_multiple_of(s.kv_heads) && slots.len() == s.rows && positions.len() == s.rows,
        "unsupported attention geometry"
    );
    check(d, q, n)?;
    check(d, out, n)?;
    check(d, keys, cache)?;
    check(d, values, cache)?;
    for row in 0..s.rows {
        let slot = usize::try_from(slots[row])?;
        let pos = usize::try_from(positions[row])?;
        ensure!(slot < s.capacity && pos < s.context, "row out of range");
        ifa_row(d, q, keys, values, out, slot, pos, s, row)?;
    }
    Ok(())
}

/// Group consecutive rows that belong to one slot and run a multi-row prefill
/// chunk through PromptFlashAttentionV3. Single-row groups stay on IFA.
#[allow(clippy::too_many_arguments)]
pub fn attention_grouped(
    d: &Device,
    q: &Buffer<bf16>,
    keys: &Buffer<bf16>,
    values: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    slots: &[i32],
    positions: &[i32],
    s: AttentionShape,
    mask: &Buffer<i8>,
) -> Result<()> {
    let n = dims(&[s.rows, s.heads, s.dim])?;
    let cache = dims(&[s.capacity, s.context, s.kv_heads, s.dim])?;
    ensure!(
        s.heads.is_multiple_of(s.kv_heads) && slots.len() == s.rows && positions.len() == s.rows,
        "unsupported attention geometry"
    );
    check(d, q, n)?;
    check(d, out, n)?;
    check(d, keys, cache)?;
    check(d, values, cache)?;
    let mut start = 0;
    while start < s.rows {
        let slot = usize::try_from(slots[start])?;
        let mut end = start + 1;
        while end < s.rows && slots[end] == slots[start] && positions[end] == positions[end - 1] + 1
        {
            end += 1;
        }
        let rows = end - start;
        let pos = usize::try_from(positions[start])?;
        ensure!(
            slot < s.capacity && pos + rows <= s.context,
            "row out of range"
        );
        if rows == 1 {
            ifa_row(d, q, keys, values, out, slot, pos, s, start)?;
        } else {
            prefill_rows(d, q, keys, values, out, slot, start, rows, pos, s, mask)?;
        }
        start = end;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn prefill_rows(
    d: &Device,
    q: &Buffer<bf16>,
    keys: &Buffer<bf16>,
    values: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    slot: usize,
    row_start: usize,
    rows: usize,
    pos0: usize,
    s: AttentionShape,
    mask: &Buffer<i8>,
) -> Result<()> {
    let mask_dim = (mask.len() as f64).sqrt() as usize;
    ensure!(
        mask_dim * mask_dim == mask.len() && mask_dim >= s.context,
        "attention mask geometry"
    );
    let row_elems = s.heads * s.dim;
    let slot_elems = s.context * s.kv_heads * s.dim;
    let kv_len = pos0 + rows;
    ensure!(
        row_start + rows <= s.rows && kv_len <= s.context && slot < s.capacity,
        "prefill row out of range"
    );
    let q_ptr = q.as_ptr() as usize + row_start * row_elems * 2;
    let out_ptr = out.as_mut_ptr() as usize + row_start * row_elems * 2;
    let keys_ptr = keys.as_ptr() as usize + slot * slot_elems * 2;
    let values_ptr = values.as_ptr() as usize + slot * slot_elems * 2;
    let scale = 1.0 / (s.dim as f64).sqrt();
    d.run(|stream| {
        let mut t = Tensors::new();
        // Q/out are row-major [rows, heads, dim]; present them as BNSD
        // [1, heads, rows, dim] through strides [heads*rows*dim, dim,
        // heads*dim, 1]. PFA V3 honours these strides (validated in
        // smoke_pfa_layout.cpp).
        let q_stride = [
            (s.heads * rows * s.dim) as i64,
            s.dim as i64,
            (s.heads * s.dim) as i64,
            1,
        ];
        let tq = t.mk_strided(
            &[1, s.heads as i64, rows as i64, s.dim as i64],
            &q_stride,
            sys::ACL_BF16,
            q_ptr as *mut c_void,
            "pfa q",
        )?;
        let tk = t.mk(
            &[1, s.kv_heads as i64, s.context as i64, s.dim as i64],
            sys::ACL_BF16,
            keys_ptr as *mut c_void,
            "pfa k",
        )?;
        let tv = t.mk(
            &[1, s.kv_heads as i64, s.context as i64, s.dim as i64],
            sys::ACL_BF16,
            values_ptr as *mut c_void,
            "pfa v",
        )?;
        let to = t.mk_strided(
            &[1, s.heads as i64, rows as i64, s.dim as i64],
            &q_stride,
            sys::ACL_BF16,
            out_ptr as *mut c_void,
            "pfa out",
        )?;
        let tm = t.mk(
            &[mask_dim as i64, mask_dim as i64],
            sys::ACL_INT8,
            mask.as_ptr().cast_mut().cast(),
            "pfa mask",
        )?;
        let actual_q = d.int_array(rows as i64)? as *const sys::aclIntArray;
        let actual_kv = d.int_array(kv_len as i64)? as *const sys::aclIntArray;
        let layout = c"BNSD";
        let result = run_oneshot(
            d,
            stream,
            |ws, ex| {
                sys::aclnnPromptFlashAttentionV3GetWorkspaceSize(
                    tq,
                    tk,
                    tv,
                    std::ptr::null(),
                    tm,
                    actual_q,
                    actual_kv,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    s.heads as i64,
                    scale,
                    i64::from(i32::MAX),
                    0,
                    layout.as_ptr().cast_mut(),
                    s.kv_heads as i64,
                    3,
                    1,
                    to,
                    ws,
                    ex,
                )
            },
            |p, n, ex, st| sys::aclnnPromptFlashAttentionV3(p, n, ex, st),
        );
        let (handles, arrays) = t.release();
        d.defer_tensors(handles, arrays);
        result
    })
}

pub fn top1(
    d: &Device,
    x: &Buffer<bf16>,
    ids: &mut Buffer<u32>,
    rows: usize,
    vocab: usize,
) -> Result<()> {
    check(d, x, dims(&[rows, vocab])?)?;
    check(d, ids, rows)?;
    let cache_key = key(
        "argmax",
        &[rows as i64, vocab as i64],
        &[x.as_ptr() as usize, ids.as_mut_ptr() as usize],
    );
    d.run(|stream| {
        run_cached(
            d,
            stream,
            cache_key,
            || {
                let tx = mk_tensor_owned(
                    &[rows as i64, vocab as i64],
                    sys::ACL_BF16,
                    x.as_ptr().cast_mut().cast(),
                );
                let to = mk_tensor_owned(&[rows as i64], sys::ACL_INT32, ids.as_mut_ptr().cast());
                ensure!(!tx.is_null() && !to.is_null(), "argmax tensor");
                let mut ws = 0u64;
                let mut ex = std::ptr::null_mut();
                crate::device::check(sys::aclnnArgMaxGetWorkspaceSize(
                    tx, 1, false, to, &mut ws, &mut ex,
                ))?;
                crate::device::check(sys::aclSetAclOpExecutorRepeatable(ex))?;
                d.keep_tensors(vec![tx, to]);
                Ok((ex as usize, ws))
            },
            |p, n, ex, st| sys::aclnnArgMax(p, n, ex, st),
        )
    })
}

#[cfg(test)]
mod validation_tests {
    use super::*;

    #[test]
    fn dimensions_reject_zero_and_overflow() {
        assert!(dims(&[1, 128]).is_ok());
        assert!(dims(&[0, 128]).is_err());
        assert!(dims(&[i32::MAX as usize + 1]).is_err());
        assert!(dims(&[i32::MAX as usize; 3]).is_err());
    }
}
