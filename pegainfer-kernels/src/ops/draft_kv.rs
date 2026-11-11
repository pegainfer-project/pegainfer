//! Gather/scatter whole rows between a dense draft scratch and the paged
//! draft KV pool. The page table decides each row's destination, so one
//! launch copies every row instead of one launch per token.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::HiddenStates;

pub fn draft_kv_scatter_into(
    ctx: &DeviceContext,
    src: &HiddenStates,
    dst: &mut HiddenStates,
    locs: &CudaSlice<i32>,
    rows: usize,
) -> Result<()> {
    assert_eq!(
        src.hidden_dim, dst.hidden_dim,
        "draft_kv_scatter_into hidden_dim mismatch: src {}, dst {}",
        src.hidden_dim, dst.hidden_dim
    );
    assert!(
        rows <= src.seq_len,
        "draft_kv_scatter source rows {rows} exceed seq_len {}",
        src.seq_len
    );
    assert!(
        locs.len() >= rows,
        "draft_kv_scatter locations has {} entries for {rows} rows",
        locs.len()
    );
    if rows == 0 {
        return Ok(());
    }
    let (src_ptr, _gs) = src.data.device_ptr(&ctx.stream);
    let (dst_ptr, _gd) = dst.data.device_ptr_mut(&ctx.stream);
    let (locs_ptr, _gl) = locs.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::draft_kv_scatter_cuda(
            src_ptr as *const ffi::Half,
            dst_ptr as *mut ffi::Half,
            locs_ptr as *const i32,
            rows as i32,
            src.hidden_dim as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}

pub fn draft_kv_gather_into(
    ctx: &DeviceContext,
    src: &HiddenStates,
    dst: &mut HiddenStates,
    locs: &CudaSlice<i32>,
    rows: usize,
) -> Result<()> {
    assert_eq!(
        src.hidden_dim, dst.hidden_dim,
        "draft_kv_gather_into hidden_dim mismatch: src {}, dst {}",
        src.hidden_dim, dst.hidden_dim
    );
    assert!(
        rows <= dst.seq_len,
        "draft_kv_gather destination rows {rows} exceed seq_len {}",
        dst.seq_len
    );
    assert!(
        locs.len() >= rows,
        "draft_kv_gather locations has {} entries for {rows} rows",
        locs.len()
    );
    if rows == 0 {
        return Ok(());
    }
    let (src_ptr, _gs) = src.data.device_ptr(&ctx.stream);
    let (dst_ptr, _gd) = dst.data.device_ptr_mut(&ctx.stream);
    let (locs_ptr, _gl) = locs.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::draft_kv_gather_cuda(
            src_ptr as *const ffi::Half,
            dst_ptr as *mut ffi::Half,
            locs_ptr as *const i32,
            rows as i32,
            src.hidden_dim as i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}
