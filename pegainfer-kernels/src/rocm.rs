//! BF16 eager operators for the first ROCm Qwen3 execution path.
use std::ffi::c_void;

use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use pegainfer_device::Buffer;
use pegainfer_device::Device;
use pegainfer_device::DeviceRepr;
unsafe extern "C" {
    fn pgi_embedding(
        d: *mut c_void,
        w: *const bf16,
        ids: *const u32,
        out: *mut bf16,
        rows: i32,
        h: i32,
        vocab: i32,
    ) -> i32;
    fn pgi_norm(
        d: *mut c_void,
        x: *const bf16,
        w: *const bf16,
        out: *mut bf16,
        rows: i32,
        h: i32,
        eps: f32,
    ) -> i32;
    fn pgi_add(d: *mut c_void, a: *const bf16, b: *const bf16, out: *mut bf16, n: usize) -> i32;
    fn pgi_silu(d: *mut c_void, a: *const bf16, b: *const bf16, out: *mut bf16, n: usize) -> i32;
    fn pgi_rope(
        d: *mut c_void,
        q: *mut bf16,
        p: *const i32,
        rows: i32,
        heads: i32,
        dim: i32,
        theta: f32,
    ) -> i32;
    fn pgi_cache(
        d: *mut c_void,
        k: *const bf16,
        v: *const bf16,
        keys: *mut bf16,
        values: *mut bf16,
        slots: *const i32,
        pos: *const i32,
        rows: i32,
        heads: i32,
        dim: i32,
        context: i32,
        capacity: i32,
    ) -> i32;
    fn pgi_attention(
        d: *mut c_void,
        q: *const bf16,
        keys: *const bf16,
        values: *const bf16,
        out: *mut bf16,
        slots: *const i32,
        pos: *const i32,
        rows: i32,
        heads: i32,
        kv_heads: i32,
        dim: i32,
        context: i32,
        capacity: i32,
    ) -> i32;
    fn pgi_top1(d: *mut c_void, x: *const bf16, ids: *mut u32, rows: i32, vocab: i32) -> i32;
    fn pgi_gemm(
        d: *mut c_void,
        w: *const bf16,
        x: *const bf16,
        out: *mut bf16,
        rows: i32,
        n: i32,
        k: i32,
    ) -> i32;
}
fn dims(values: &[usize]) -> Result<usize> {
    ensure!(
        values.iter().all(|&v| v > 0 && i32::try_from(v).is_ok()),
        "invalid HIP operator shape {values:?}"
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
/// Gather rows. Every device-side ID must be below `vocab`; violations trap
/// and surface at submission or stream synchronization. No zero-row fallback.
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
    unsafe {
        d.submit(|dev| {
            pgi_embedding(
                dev,
                w.as_ptr(),
                ids.as_ptr(),
                out.as_mut_ptr(),
                rows as _,
                h as _,
                vocab as _,
            )
        })
    }
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
    unsafe {
        d.submit(|dev| {
            pgi_norm(
                dev,
                x.as_ptr(),
                w.as_ptr(),
                out.as_mut_ptr(),
                rows as _,
                h as _,
                eps,
            )
        })
    }
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
    unsafe { d.submit(|dev| pgi_add(dev, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n)) }
}
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
    unsafe { d.submit(|dev| pgi_silu(dev, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n)) }
}
pub fn gemm(
    d: &Device,
    w: &Buffer<bf16>,
    x: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    rows: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    check(d, w, dims(&[n, k])?)?;
    check(d, x, dims(&[rows, k])?)?;
    check(d, out, dims(&[rows, n])?)?;
    unsafe {
        d.submit(|dev| {
            pgi_gemm(
                dev,
                w.as_ptr(),
                x.as_ptr(),
                out.as_mut_ptr(),
                rows as _,
                n as _,
                k as _,
            )
        })
    }
}
pub fn rope(
    d: &Device,
    q: &mut Buffer<bf16>,
    p: &Buffer<i32>,
    rows: usize,
    heads: usize,
    dim: usize,
    theta: f32,
) -> Result<()> {
    check(d, q, dims(&[rows, heads, dim])?)?;
    check(d, p, rows)?;
    ensure!(
        dim.is_multiple_of(2) && theta.is_finite() && theta > 0.,
        "invalid rotary geometry"
    );
    unsafe {
        d.submit(|dev| {
            pgi_rope(
                dev,
                q.as_mut_ptr(),
                p.as_ptr(),
                rows as _,
                heads as _,
                dim as _,
                theta,
            )
        })
    }
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
/// Append KV rows. Device slots/positions must be in `[0, capacity)` and
/// `[0, context)` and writes must be unique. Bounds violations trap.
pub fn cache(
    d: &Device,
    k: &Buffer<bf16>,
    v: &Buffer<bf16>,
    keys: &mut Buffer<bf16>,
    values: &mut Buffer<bf16>,
    slots: &Buffer<i32>,
    pos: &Buffer<i32>,
    s: AttentionShape,
) -> Result<()> {
    let n = dims(&[s.rows, s.kv_heads, s.dim])?;
    let cache = dims(&[s.capacity, s.context, s.kv_heads, s.dim])?;
    for b in [k, v] {
        check(d, b, n)?;
    }
    check(d, keys, cache)?;
    check(d, values, cache)?;
    check(d, slots, s.rows)?;
    check(d, pos, s.rows)?;
    unsafe {
        d.submit(|dev| {
            pgi_cache(
                dev,
                k.as_ptr(),
                v.as_ptr(),
                keys.as_mut_ptr(),
                values.as_mut_ptr(),
                slots.as_ptr(),
                pos.as_ptr(),
                s.rows as _,
                s.kv_heads as _,
                s.dim as _,
                s.context as _,
                s.capacity as _,
            )
        })
    }
}
/// Attend to each row's causal prefix. Slot/position bounds are checked on
/// device and trap on violation, rather than returning stale output.
pub fn attention(
    d: &Device,
    q: &Buffer<bf16>,
    keys: &Buffer<bf16>,
    values: &Buffer<bf16>,
    out: &mut Buffer<bf16>,
    slots: &Buffer<i32>,
    pos: &Buffer<i32>,
    s: AttentionShape,
) -> Result<()> {
    let n = dims(&[s.rows, s.heads, s.dim])?;
    let cache = dims(&[s.capacity, s.context, s.kv_heads, s.dim])?;
    ensure!(
        s.heads.is_multiple_of(s.kv_heads) && s.context <= 8192,
        "unsupported attention geometry"
    );
    check(d, q, n)?;
    check(d, out, n)?;
    check(d, keys, cache)?;
    check(d, values, cache)?;
    check(d, slots, s.rows)?;
    check(d, pos, s.rows)?;
    unsafe {
        d.submit(|dev| {
            pgi_attention(
                dev,
                q.as_ptr(),
                keys.as_ptr(),
                values.as_ptr(),
                out.as_mut_ptr(),
                slots.as_ptr(),
                pos.as_ptr(),
                s.rows as _,
                s.heads as _,
                s.kv_heads as _,
                s.dim as _,
                s.context as _,
                s.capacity as _,
            )
        })
    }
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
    unsafe { d.submit(|dev| pgi_top1(dev, x.as_ptr(), ids.as_mut_ptr(), rows as _, vocab as _)) }
}

/// Exact targets compiled into this artifact.
pub fn built_arches() -> &'static str {
    env!("PEGAINFER_BUILT_ROCM_ARCH")
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    #[test]
    fn dimensions_reject_zero_abi_overflow_and_product_overflow() {
        assert!(dims(&[1, 128]).is_ok());
        assert!(dims(&[0, 128]).is_err());
        assert!(dims(&[i32::MAX as usize + 1]).is_err());
        assert!(dims(&[i32::MAX as usize; 3]).is_err());
    }
}
