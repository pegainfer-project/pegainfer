//! CUDA-backed DFlash2 candidate selection.
//!
//! Runs deterministic top-k extraction and request-local path walks.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceMatrix;
use crate::tensor::HiddenStates;

const DFLASH2_SELECTOR_TOP_K: usize = 16;

/// Persistent buffers for the two selector launches.
pub struct DFlash2SelectorScratch {
    pub candidate_ids: CudaSlice<u32>,
    pub candidate_scores: CudaSlice<f32>,
    pub selected: CudaSlice<u32>,
    rows: usize,
}

impl DFlash2SelectorScratch {
    pub fn new(ctx: &DeviceContext, rows: usize) -> Result<Self> {
        ensure!(rows > 0, "DFlash2 selector rows must be positive");
        let candidate_rows = rows
            .checked_mul(DFLASH2_SELECTOR_TOP_K)
            .ok_or_else(|| anyhow!("DFlash2 selector scratch size overflow"))?;
        Ok(Self {
            candidate_ids: ctx.stream.alloc_zeros(candidate_rows)?,
            candidate_scores: ctx.stream.alloc_zeros(candidate_rows)?,
            selected: ctx.stream.alloc_zeros(rows)?,
            rows,
        })
    }
}

/// Run top-16 extraction and request-local path walks.
///
/// Inputs are anchor-inclusive request-major rows; output rows are compact.
/// Ties are resolved by score descending, then token id ascending.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_selector_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    projected_hidden: &HiddenStates,
    predecessor: &DeviceMatrix,
    successor: &DeviceMatrix,
    anchors: &CudaSlice<u32>,
    input_block_size: usize,
    position_offset: usize,
    positions_per_request: usize,
    scratch: &mut DFlash2SelectorScratch,
) -> Result<()> {
    ensure!(
        input_block_size > 0,
        "DFlash2 selector input block size must be positive"
    );
    ensure!(
        positions_per_request > 0,
        "DFlash2 selector positions must be positive"
    );
    ensure!(position_offset <= input_block_size);
    ensure!(
        positions_per_request <= input_block_size - position_offset,
        "DFlash2 selector positions exceed the input block"
    );
    ensure!(logits.seq_len == projected_hidden.seq_len);
    ensure!(logits.seq_len.is_multiple_of(input_block_size));
    let requests = logits.seq_len / input_block_size;
    let compact_rows = requests
        .checked_mul(positions_per_request)
        .ok_or_else(|| anyhow!("DFlash2 selector compact row count overflow"))?;
    ensure!(
        anchors.len() >= requests,
        "selector anchor buffer is too small"
    );
    ensure!(
        scratch.rows >= compact_rows,
        "selector scratch is too small"
    );
    ensure!(projected_hidden.hidden_dim == predecessor.cols);
    ensure!(predecessor.cols == successor.cols);
    ensure!(predecessor.rows == successor.rows);
    ensure!(logits.hidden_dim >= DFLASH2_SELECTOR_TOP_K);
    ensure!(predecessor.rows == logits.hidden_dim);
    let rows_i32 = i32::try_from(compact_rows)
        .map_err(|_| anyhow!("DFlash2 selector row count exceeds i32"))?;
    let requests_i32 = i32::try_from(requests)
        .map_err(|_| anyhow!("DFlash2 selector request count exceeds i32"))?;
    let input_block_size_i32 = i32::try_from(input_block_size)
        .map_err(|_| anyhow!("DFlash2 selector input block size exceeds i32"))?;
    let position_offset_i32 = i32::try_from(position_offset)
        .map_err(|_| anyhow!("DFlash2 selector position offset exceeds i32"))?;
    let positions_per_request_i32 = i32::try_from(positions_per_request)
        .map_err(|_| anyhow!("DFlash2 selector position count exceeds i32"))?;
    let vocab_i32 = i32::try_from(logits.hidden_dim)
        .map_err(|_| anyhow!("DFlash2 selector vocabulary size exceeds i32"))?;
    let rank_i32 = i32::try_from(predecessor.cols)
        .map_err(|_| anyhow!("DFlash2 selector rank exceeds i32"))?;

    let (logits_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
    let (candidate_ids_ptr, _gi) = scratch.candidate_ids.device_ptr_mut(&ctx.stream);
    let (candidate_scores_ptr, _gs) = scratch.candidate_scores.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::dflash2_selector_topk_cuda(
            logits_ptr as *const ffi::Half,
            candidate_ids_ptr as *mut u32,
            candidate_scores_ptr as *mut f32,
            rows_i32,
            input_block_size_i32,
            position_offset_i32,
            positions_per_request_i32,
            vocab_i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    ensure!(
        status == 0,
        "DFlash2 selector top-k launch failed: {status}"
    );

    let (hidden_ptr, _gh) = projected_hidden.data.device_ptr(&ctx.stream);
    let (predecessor_ptr, _gp) = predecessor.data.device_ptr(&ctx.stream);
    let (successor_ptr, _gs) = successor.data.device_ptr(&ctx.stream);
    let (anchors_ptr, _ga) = anchors.device_ptr(&ctx.stream);
    let (selected_ptr, _go) = scratch.selected.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::dflash2_selector_walk_cuda(
            hidden_ptr as *const ffi::Half,
            predecessor_ptr as *const ffi::Half,
            successor_ptr as *const ffi::Half,
            anchors_ptr as *const u32,
            candidate_ids_ptr as *const u32,
            candidate_scores_ptr as *const f32,
            selected_ptr as *mut u32,
            requests_i32,
            input_block_size_i32,
            position_offset_i32,
            positions_per_request_i32,
            vocab_i32,
            rank_i32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    ensure!(status == 0, "DFlash2 selector path launch failed: {status}");
    Ok(())
}

/// Copy selected token ids to host after the caller synchronizes.
pub fn dflash2_selector_selected_host(
    ctx: &DeviceContext,
    scratch: &DFlash2SelectorScratch,
    rows: usize,
) -> Result<Vec<u32>> {
    ensure!(rows <= scratch.rows, "selector output rows exceed scratch");
    ctx.stream
        .clone_dtoh(&scratch.selected.slice(..rows))
        .map_err(|e| anyhow!("DFlash2 selector D2H failed: {e}"))
}

/// Bytes required by [`DFlash2SelectorScratch`] for `rows` active rows.
pub const fn dflash2_selector_scratch_bytes(rows: usize) -> usize {
    rows * (DFLASH2_SELECTOR_TOP_K * std::mem::size_of::<u32>()
        + DFLASH2_SELECTOR_TOP_K * std::mem::size_of::<f32>()
        + std::mem::size_of::<u32>())
}
