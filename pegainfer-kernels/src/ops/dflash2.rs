//! DFlash2's strict candidate lattice; request sampling remains in pegainfer-sample.

use std::ffi::CStr;
use std::sync::Arc;

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
use crate::tensor::active_cu_stream;
use crate::tensor::has_stream_override;

pub const DFLASH2_CANDIDATE_K: usize = 16;

/// Fixed-capacity, stream-local scratch. Keys are packed in fixed-size row chunks;
/// CUB selects and orders only 16 candidates per row. Partial chunks have
/// deterministic padding and never read input row 0.
/// All allocation and CUB workspace queries occur in `new`, outside capture.
///
/// The output accessors expose capacity-sized buffers. Only `active_batch *
/// (block_size - 1)` rows belong to the latest selection. An error flag must be
/// collected with the path before any token is submitted to a verifier.
pub struct DFlash2Scratch {
    max_batch: usize,
    block_size: usize,
    vocab: usize,
    rank: usize,
    rows_per_chunk: usize,

    keys_in: CudaSlice<u64>,
    keys_out: CudaSlice<u64>,
    topk_workspace: CudaSlice<u8>,

    candidate_ids: CudaSlice<u32>,
    unary_scores: CudaSlice<f32>,
    gated_predecessors: CudaSlice<f32>,
    successors: CudaSlice<f32>,
    edge_scores: CudaSlice<f32>,
    selected_ids: CudaSlice<u32>,
    error_flag: CudaSlice<u32>,

    total_bytes: usize,
}

fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |product, &value| {
        product
            .checked_mul(value)
            .filter(|&size| isize::try_from(size).is_ok())
            .ok_or_else(|| anyhow!("DFlash2 allocation/shape overflow: {values:?}"))
    })
}

fn check_ffi(status: i32, stage: &str) -> Result<()> {
    if status != 0 {
        // Guarded FFI calls clear the thread-local error on entry and retain it
        // until the next guarded call; read it immediately on this thread.
        let message = unsafe {
            let pointer = ffi::pegainfer_kernels_last_error();
            if pointer.is_null() {
                String::new()
            } else {
                CStr::from_ptr(pointer).to_string_lossy().into_owned()
            }
        };
        return Err(anyhow!("DFlash2 {stage} failed ({status}): {message}"));
    }

    Ok(())
}

impl DFlash2Scratch {
    pub fn new(
        ctx: &DeviceContext,
        max_batch: usize,
        block_size: usize,
        vocab: usize,
        rank: usize,
        rows_per_chunk: usize,
    ) -> Result<Self> {
        ensure!(
            !has_stream_override(),
            "DFlash2 scratch requires the base stream"
        );
        ensure!(
            max_batch > 0 && block_size >= 2 && rank > 0,
            "DFlash2 requires max_batch > 0, block_size >= 2, rank > 0"
        );
        ensure!(
            vocab >= DFLASH2_CANDIDATE_K && i32::try_from(vocab).is_ok(),
            "DFlash2 vocab must be in 16..=i32::MAX, got {vocab}"
        );
        ensure!(
            i32::try_from(rank).is_ok(),
            "DFlash2 rank exceeds cuBLAS int32"
        );

        let input_rows = product(&[max_batch, block_size])?;
        ensure!(
            i32::try_from(input_rows).is_ok(),
            "DFlash2 input rows exceed int32"
        );

        let rows = product(&[max_batch, block_size - 1])?;
        ensure!(
            rows_per_chunk > 0 && rows_per_chunk <= rows,
            "DFlash2 packing chunk must be in 1..={rows}, got {rows_per_chunk}"
        );

        let keys = product(&[rows_per_chunk, vocab])?;
        let chunk_candidates = product(&[rows_per_chunk, DFLASH2_CANDIDATE_K])?;
        let candidates = product(&[rows, DFLASH2_CANDIDATE_K])?;
        let gathered = product(&[candidates, rank])?;
        let edges = product(&[candidates, DFLASH2_CANDIDATE_K])?;

        let mut workspace_bytes = 0;
        let status = unsafe {
            ffi::dflash2_topk_workspace_bytes_cuda(
                vocab as i32,
                &raw mut workspace_bytes,
                active_cu_stream(ctx),
            )
        };
        check_ffi(status, "CUB workspace query")?;

        // TopK reuses one row's library scratch; only the packed input retains
        // full vocabulary width. Caller separately owns H/W/A/B.
        let sizes = [
            product(&[keys, 8])?,
            product(&[chunk_candidates, 8])?,
            workspace_bytes.max(1),
            product(&[candidates, 8])?,
            product(&[gathered, 8])?,
            product(&[edges, 4])?,
            product(&[rows, 4])?,
            4,
        ];
        let total_bytes = sizes.iter().try_fold(0usize, |sum, &bytes| {
            sum.checked_add(bytes)
                .filter(|&size| isize::try_from(size).is_ok())
                .ok_or_else(|| anyhow!("DFlash2 scratch byte total overflow"))
        })?;

        Ok(Self {
            max_batch,
            block_size,
            vocab,
            rank,
            rows_per_chunk,
            keys_in: ctx.stream.alloc_zeros(keys)?,
            keys_out: ctx.stream.alloc_zeros(chunk_candidates)?,
            topk_workspace: ctx.stream.alloc_zeros(workspace_bytes.max(1))?,
            candidate_ids: ctx.stream.alloc_zeros(candidates)?,
            unary_scores: ctx.stream.alloc_zeros(candidates)?,
            gated_predecessors: ctx.stream.alloc_zeros(gathered)?,
            successors: ctx.stream.alloc_zeros(gathered)?,
            edge_scores: ctx.stream.alloc_zeros(edges)?,
            selected_ids: ctx.stream.alloc_zeros(rows)?,
            error_flag: ctx.stream.alloc_zeros(1)?,
            total_bytes,
        })
    }

    pub fn candidate_ids(&self) -> &CudaSlice<u32> {
        &self.candidate_ids
    }

    pub fn unary_scores(&self) -> &CudaSlice<f32> {
        &self.unary_scores
    }

    /// Row-major `[request, position, predecessor_candidate, successor_candidate]`.
    pub fn edge_scores(&self) -> &CudaSlice<f32> {
        &self.edge_scores
    }

    pub fn selected_ids(&self) -> &CudaSlice<u32> {
        &self.selected_ids
    }

    /// Bit mask: 1 = NaN/+Inf unary, 2 = fewer than 16 finite candidates,
    /// 4 = invalid token ID, 8 = nonfinite gate/edge. Zero means valid.
    pub fn error_flag(&self) -> &CudaSlice<u32> {
        &self.error_flag
    }

    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }
}

fn check_stream<T>(ctx: &DeviceContext, buffer: &CudaSlice<T>, name: &str) -> Result<()> {
    ensure!(
        Arc::ptr_eq(buffer.stream(), &ctx.stream),
        "DFlash2 {name} must belong to the scratch base stream"
    );

    Ok(())
}

/// Enqueue native anchor-drop selection using BF16 unary logits, BF16 projected
/// hidden states and BF16 codebooks. No allocation, host transfer or sync occurs
/// here. Warm up shared cuBLAS before graph capture, as for existing GEMM ops.
///
/// Buffers must use the same base stream: DeviceContext disables cudarc's event
/// tracking, so a stream override would leave buffer ownership unsynchronized.
/// An empty batch is a no-op; its caller must not collect stale scratch outputs.
#[allow(clippy::too_many_arguments)]
pub fn dflash2_select_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    projected_hidden: &HiddenStates,
    predecessor: &DeviceMatrix,
    successor: &DeviceMatrix,
    anchors: &CudaSlice<u32>,
    active_batch: usize,
    scratch: &mut DFlash2Scratch,
) -> Result<()> {
    ensure!(
        !has_stream_override(),
        "DFlash2 selection requires the base stream"
    );
    ensure!(
        active_batch <= scratch.max_batch,
        "DFlash2 active batch {active_batch} exceeds capacity {}",
        scratch.max_batch
    );
    if active_batch == 0 {
        return Ok(());
    }

    let input_rows = product(&[active_batch, scratch.block_size])?;
    logits.checked_extent("DFlash2 unary logits")?;
    projected_hidden.checked_extent("DFlash2 projected hidden")?;
    ensure!(
        logits.hidden_dim == scratch.vocab && logits.seq_len >= input_rows,
        "DFlash2 unary expected at least [{input_rows}, {}], got [{}, {}]",
        scratch.vocab,
        logits.seq_len,
        logits.hidden_dim
    );
    ensure!(
        projected_hidden.hidden_dim == scratch.rank && projected_hidden.seq_len >= input_rows,
        "DFlash2 projected hidden expected at least [{input_rows}, {}], got [{}, {}]",
        scratch.rank,
        projected_hidden.seq_len,
        projected_hidden.hidden_dim
    );

    let codebook_elements = product(&[scratch.vocab, scratch.rank])?;
    for (name, matrix) in [("predecessor", predecessor), ("successor", successor)] {
        ensure!(
            matrix.rows == scratch.vocab
                && matrix.cols == scratch.rank
                && matrix.data.len() >= codebook_elements,
            "DFlash2 {name} expected [{}, {}] with sufficient backing, got [{}, {}], len={}",
            scratch.vocab,
            scratch.rank,
            matrix.rows,
            matrix.cols,
            matrix.data.len()
        );
        check_stream(ctx, &matrix.data, name)?;
    }

    ensure!(
        anchors.len() >= active_batch,
        "DFlash2 anchors shorter than active batch"
    );
    check_stream(ctx, &logits.data, "logits")?;
    check_stream(ctx, &projected_hidden.data, "projected hidden")?;
    check_stream(ctx, anchors, "anchors")?;
    check_stream(ctx, &scratch.error_flag, "scratch")?;

    {
        let (logits_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (hidden_ptr, _gh) = projected_hidden.data.device_ptr(&ctx.stream);
        let (predecessor_ptr, _gp) = predecessor.data.device_ptr(&ctx.stream);
        let (successor_ptr, _gs) = successor.data.device_ptr(&ctx.stream);
        let (anchors_ptr, _ga) = anchors.device_ptr(&ctx.stream);
        let (ids_ptr, _gi) = scratch.candidate_ids.device_ptr_mut(&ctx.stream);
        let (unary_ptr, _gu) = scratch.unary_scores.device_ptr_mut(&ctx.stream);
        let (gated_ptr, _gg) = scratch.gated_predecessors.device_ptr_mut(&ctx.stream);
        let (successors_ptr, _gss) = scratch.successors.device_ptr_mut(&ctx.stream);
        let (error_ptr, _ge) = scratch.error_flag.device_ptr_mut(&ctx.stream);
        let (keys_in_ptr, _gki) = scratch.keys_in.device_ptr_mut(&ctx.stream);
        let (keys_out_ptr, _gko) = scratch.keys_out.device_ptr_mut(&ctx.stream);
        let workspace_bytes = scratch.topk_workspace.len();
        let (workspace_ptr, _gw) = scratch.topk_workspace.device_ptr_mut(&ctx.stream);

        let status = unsafe {
            ffi::dflash2_prepare_cuda(
                logits_ptr as *const ffi::Half,
                hidden_ptr as *const ffi::Half,
                predecessor_ptr as *const ffi::Half,
                successor_ptr as *const ffi::Half,
                anchors_ptr as *const u32,
                ids_ptr as *mut u32,
                unary_ptr as *mut f32,
                gated_ptr as *mut f32,
                successors_ptr as *mut f32,
                error_ptr as *mut u32,
                keys_in_ptr as *mut u64,
                keys_out_ptr as *mut u64,
                workspace_ptr as *mut std::ffi::c_void,
                workspace_bytes,
                active_batch as i32,
                scratch.block_size as i32,
                scratch.vocab as i32,
                scratch.rank as i32,
                scratch.rows_per_chunk as i32,
                active_cu_stream(ctx),
            )
        };
        check_ffi(status, "candidate selection/gather")?;
    }

    let rows = active_batch * (scratch.block_size - 1);
    let candidate_stride = DFLASH2_CANDIDATE_K * scratch.rank;

    // Row-major E[p,c] = gated[p,:] dot successor[c,:]. cuBLAS sees the
    // transposed column-major result, hence successor^T is its first operand.
    super::linear::gemm_strided_batched_f32(
        ctx,
        true,
        false,
        DFLASH2_CANDIDATE_K,
        DFLASH2_CANDIDATE_K,
        scratch.rank,
        &scratch.successors,
        scratch.rank,
        candidate_stride,
        &scratch.gated_predecessors,
        scratch.rank,
        candidate_stride,
        false,
        &mut scratch.edge_scores,
        DFLASH2_CANDIDATE_K,
        DFLASH2_CANDIDATE_K * DFLASH2_CANDIDATE_K,
        rows,
    )?;

    {
        let (edges_ptr, _ge) = scratch.edge_scores.device_ptr_mut(&ctx.stream);
        let (unary_ptr, _gu) = scratch.unary_scores.device_ptr(&ctx.stream);
        let (ids_ptr, _gi) = scratch.candidate_ids.device_ptr(&ctx.stream);
        let (selected_ptr, _gs) = scratch.selected_ids.device_ptr_mut(&ctx.stream);
        let (error_ptr, _gf) = scratch.error_flag.device_ptr_mut(&ctx.stream);

        let status = unsafe {
            ffi::dflash2_finish_cuda(
                edges_ptr as *mut f32,
                unary_ptr as *const f32,
                ids_ptr as *const u32,
                selected_ptr as *mut u32,
                error_ptr as *mut u32,
                active_batch as i32,
                (scratch.block_size - 1) as i32,
                active_cu_stream(ctx),
            )
        };
        check_ffi(status, "edge scoring/walk")?;
    }

    Ok(())
}
