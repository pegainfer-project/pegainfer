//! Scatter/gather whole rows between the drafter's dense K/V mirrors and the
//! paged draft pool. The page table decides each row's slot, and every buffer
//! a call names shares the same rows, so one launch covers all of them.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::HiddenStates;

/// Buffers one launch can copy; the pair table travels in the kernel
/// parameters.
const MAX_PAIRS: usize = 32;

fn validate(
    op: &str,
    pairs: &[(&HiddenStates, &mut HiddenStates)],
    locs: &CudaSlice<i32>,
    rows: usize,
    gather: bool,
) -> usize {
    assert!(!pairs.is_empty(), "{op} needs at least one buffer pair");
    assert!(
        pairs.len() <= MAX_PAIRS,
        "{op} got {} buffer pairs, the kernel takes {MAX_PAIRS}",
        pairs.len()
    );
    let hidden = pairs[0].0.hidden_dim;
    for (src, dst) in pairs {
        assert_eq!(
            src.hidden_dim, dst.hidden_dim,
            "{op} hidden_dim mismatch: src {}, dst {}",
            src.hidden_dim, dst.hidden_dim
        );
        assert_eq!(
            hidden, src.hidden_dim,
            "{op} buffer pairs disagree on hidden_dim"
        );
        let rows_bound = if gather { dst.seq_len } else { src.seq_len };
        assert!(
            rows <= rows_bound,
            "{op} source rows {rows} exceed seq_len {rows_bound}"
        );
    }
    assert!(
        locs.len() >= rows,
        "{op} locations has {} entries for {rows} rows",
        locs.len()
    );
    hidden
}

fn launch(
    ctx: &DeviceContext,
    pairs: &mut [(&HiddenStates, &mut HiddenStates)],
    locs: &CudaSlice<i32>,
    rows: usize,
    gather: bool,
) -> Result<()> {
    if rows == 0 {
        return Ok(());
    }
    let op = if gather {
        "draft_kv_gather_into"
    } else {
        "draft_kv_scatter_into"
    };
    let hidden = validate(op, pairs, locs, rows, gather);
    let pair_count = pairs.len();

    let mut srcs = [std::ptr::null::<ffi::Half>(); MAX_PAIRS];
    let mut dsts = [std::ptr::null_mut::<ffi::Half>(); MAX_PAIRS];
    let mut guards = Vec::with_capacity(2 * pair_count);
    for (i, (src, dst)) in pairs.iter_mut().enumerate() {
        let (src_ptr, src_guard) = src.data.device_ptr(&ctx.stream);
        let (dst_ptr, dst_guard) = dst.data.device_ptr_mut(&ctx.stream);
        srcs[i] = src_ptr as *const ffi::Half;
        dsts[i] = dst_ptr as *mut ffi::Half;
        guards.push(src_guard);
        guards.push(dst_guard);
    }

    let (locs_ptr, locs_guard) = locs.device_ptr(&ctx.stream);
    let result = unsafe {
        if gather {
            ffi::draft_kv_gather_cuda(
                srcs.as_ptr(),
                dsts.as_ptr(),
                pair_count as i32,
                locs_ptr as *const i32,
                rows as i32,
                hidden as i32,
                crate::tensor::active_cu_stream(ctx),
            )
        } else {
            ffi::draft_kv_scatter_cuda(
                srcs.as_ptr(),
                dsts.as_ptr(),
                pair_count as i32,
                locs_ptr as *const i32,
                rows as i32,
                hidden as i32,
                crate::tensor::active_cu_stream(ctx),
            )
        }
    };
    result.result()?;
    drop(guards);
    drop(locs_guard);
    Ok(())
}

/// Write `rows` rows of each source into the slots `locs[0..rows]` names in
/// the matching destination. Every location must index a row of every
/// destination; the page table that produces them is validated by the caller.
pub fn draft_kv_scatter_into(
    ctx: &DeviceContext,
    pairs: &mut [(&HiddenStates, &mut HiddenStates)],
    locs: &CudaSlice<i32>,
    rows: usize,
) -> Result<()> {
    launch(ctx, pairs, locs, rows, false)
}

/// Read the slots `locs[0..rows]` of each source into the first `rows` rows of
/// the matching destination.
pub fn draft_kv_gather_into(
    ctx: &DeviceContext,
    pairs: &mut [(&HiddenStates, &mut HiddenStates)],
    locs: &CudaSlice<i32>,
    rows: usize,
) -> Result<()> {
    launch(ctx, pairs, locs, rows, true)
}
