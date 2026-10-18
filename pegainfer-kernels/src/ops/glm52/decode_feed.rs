use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use super::moe_tp::GLM52_TP_HIDDEN;
use super::moe_tp::GLM52_TP_MAX_RANKS;
use crate::ffi;
use crate::tensor::DeviceContext;

const VOCAB_CANDIDATE_FIELDS: usize = 4;

/// Pack one rank-local `(top value, global token id)` candidate per row into
/// rank-unique positions of a hidden-width vector. The existing TP attention
/// all-reduce can then gather every rank's candidate without introducing a
/// second communication protocol into the captured decode graph.
#[allow(clippy::too_many_arguments)]
pub fn glm52_vocab_parallel_pack_launch(
    ctx: &DeviceContext,
    local_values: &CudaSlice<half::bf16>,
    local_indices: &CudaSlice<i32>,
    partial: &mut CudaSlice<half::bf16>,
    rows: usize,
    rank: usize,
    vocab_start: usize,
) -> Result<()> {
    ensure!(rows > 0, "GLM5.2 vocab pack needs at least one row");
    ensure!(
        rank < GLM52_TP_MAX_RANKS,
        "GLM5.2 vocab pack rank {rank} out of range"
    );
    ensure!(
        (rank + 1) * VOCAB_CANDIDATE_FIELDS <= GLM52_TP_HIDDEN,
        "GLM5.2 vocab pack fields exceed the hidden vector"
    );
    ensure!(
        local_values.len() >= rows && local_indices.len() >= rows,
        "GLM5.2 vocab pack candidate buffers are too small"
    );
    ensure!(
        partial.len() >= rows * GLM52_TP_HIDDEN,
        "GLM5.2 vocab pack hidden buffer is too small"
    );
    let vocab_start_i32 = i32::try_from(vocab_start)
        .map_err(|_| anyhow!("GLM5.2 vocab shard start {vocab_start} exceeds i32"))?;

    let (values_ptr, _gv) = local_values.device_ptr(&ctx.stream);
    let (indices_ptr, _gi) = local_indices.device_ptr(&ctx.stream);
    let (partial_ptr, _gp) = partial.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_vocab_parallel_pack_cuda(
            values_ptr as *const ffi::Half,
            indices_ptr as *const i32,
            partial_ptr as *mut ffi::Half,
            rows as i32,
            rank as i32,
            vocab_start_i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 vocab candidate pack failed: {err}"))
}

/// Select the global top-1 from the rank candidates gathered by the TP
/// all-reduce. Ties retain the lowest global token id, matching the local
/// argmax contract.
pub fn glm52_vocab_parallel_unpack_launch(
    ctx: &DeviceContext,
    gathered: &CudaSlice<half::bf16>,
    values: &mut CudaSlice<half::bf16>,
    indices: &mut CudaSlice<i32>,
    rows: usize,
    ranks: usize,
) -> Result<()> {
    ensure!(rows > 0, "GLM5.2 vocab unpack needs at least one row");
    ensure!(
        (1..=GLM52_TP_MAX_RANKS).contains(&ranks),
        "GLM5.2 vocab unpack rank count {ranks} out of range"
    );
    ensure!(
        gathered.len() >= rows * GLM52_TP_HIDDEN,
        "GLM5.2 vocab unpack hidden buffer is too small"
    );
    ensure!(
        values.len() >= rows && indices.len() >= rows,
        "GLM5.2 vocab unpack output buffers are too small"
    );

    let (gathered_ptr, _gg) = gathered.device_ptr(&ctx.stream);
    let (values_ptr, _gv) = values.device_ptr_mut(&ctx.stream);
    let (indices_ptr, _gi) = indices.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_vocab_parallel_unpack_cuda(
            gathered_ptr as *const ffi::Half,
            values_ptr as *mut ffi::Half,
            indices_ptr as *mut i32,
            rows as i32,
            ranks as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 global vocab candidate select failed: {err}"))
}

/// Advance the decode-step device inputs in place for a self-fed next step:
/// `token_ids[row] = argmax_indices[row]`, and `positions`/`slot_mapping`/
/// `seq_lens` each move one position forward. Enqueued between two whole-step
/// graph replays; a host prologue that follows instead simply overwrites all
/// four buffers.
pub fn glm52_decode_feed_launch(
    ctx: &DeviceContext,
    argmax_indices: &CudaSlice<i32>,
    token_ids: &mut CudaSlice<u32>,
    positions: &mut CudaSlice<u32>,
    slot_mapping: &mut CudaSlice<i64>,
    seq_lens: &mut CudaSlice<i32>,
    rows: usize,
) -> Result<()> {
    ensure!(rows > 0, "GLM5.2 decode feed needs at least one row");
    for (len, name) in [
        (argmax_indices.len(), "argmax_indices"),
        (token_ids.len(), "token_ids"),
        (positions.len(), "positions"),
        (slot_mapping.len(), "slot_mapping"),
        (seq_lens.len(), "seq_lens"),
    ] {
        ensure!(
            len >= rows,
            "GLM5.2 decode feed buffer {name} holds {len} rows, step needs {rows}"
        );
    }

    let (argmax_ptr, _argmax_guard) = argmax_indices.device_ptr(&ctx.stream);
    let (token_ptr, _token_guard) = token_ids.device_ptr_mut(&ctx.stream);
    let (pos_ptr, _pos_guard) = positions.device_ptr_mut(&ctx.stream);
    let (slot_ptr, _slot_guard) = slot_mapping.device_ptr_mut(&ctx.stream);
    let (seq_ptr, _seq_guard) = seq_lens.device_ptr_mut(&ctx.stream);

    let result = unsafe {
        ffi::glm52_decode_feed_launch_cuda(
            argmax_ptr as *const i32,
            token_ptr as *mut u32,
            pos_ptr as *mut u32,
            slot_ptr as *mut i64,
            seq_ptr as *mut i32,
            rows as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 decode feed CUDA launch failed: {err}"))
}
