use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;
use crate::tensor::HiddenStatesRef;

/// One non-greedy row of a batched sampling call.
///
/// `temperature` must be > 0 and `top_p` in (0, 1] — greedy rows
/// (`temperature <= 0` or `top_k == 1`) belong on the argmax path.
/// `top_k <= 0` means disabled. `min_p` in [0, 1); `0.0` means disabled —
/// `gpu_sample_batch_into` partitions min_p rows into their own pass, so
/// callers may mix freely.
#[derive(Clone, Copy, Debug)]
pub struct BatchSamplingRow {
    /// Row index into the logits arena.
    pub row: usize,
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
}

/// Device buffers for `gpu_sample_batch_into`, sized for `max_rows` x `vocab`.
pub struct BatchSamplingScratch {
    probs: CudaSlice<f32>,
    row_indices: CudaSlice<i32>,
    temperature: CudaSlice<f32>,
    top_k: CudaSlice<i32>,
    top_p: CudaSlice<f32>,
    min_p: CudaSlice<f32>,
    topk_row_states: CudaSlice<u8>,
    valid: CudaSlice<u8>,
    out: CudaSlice<i32>,
    softmax_workspace: CudaSlice<u8>,
    max_rows: usize,
    vocab: usize,
}

impl BatchSamplingScratch {
    pub fn new(ctx: &DeviceContext, max_rows: usize, vocab: usize) -> Result<Self> {
        ensure!(
            max_rows > 0 && vocab > 0,
            "batch sampling scratch requires max_rows > 0 and vocab > 0"
        );
        // OnlineSoftmax vocab-splitting path: batch x ceil(vocab / 8192)
        // partials of {f32 max, f32 denominator}, plus alignment slack.
        let softmax_workspace_bytes = max_rows * vocab.div_ceil(8192) * 8 + 256;
        let topk_row_states_bytes = unsafe { ffi::gpu_sample_topk_renorm_row_states_bytes_cuda() };
        let alloc = |n: usize| -> Result<CudaSlice<f32>> {
            ctx.stream
                .alloc_zeros(n)
                .map_err(|e| anyhow!("batch sampling scratch alloc failed: {e}"))
        };
        Ok(Self {
            probs: alloc(max_rows * vocab)?,
            row_indices: ctx.stream.alloc_zeros(max_rows)?,
            temperature: alloc(max_rows)?,
            top_k: ctx.stream.alloc_zeros(max_rows)?,
            top_p: alloc(max_rows)?,
            min_p: alloc(max_rows)?,
            topk_row_states: ctx.stream.alloc_zeros(topk_row_states_bytes)?,
            valid: ctx.stream.alloc_zeros(max_rows)?,
            out: ctx.stream.alloc_zeros(max_rows)?,
            softmax_workspace: ctx.stream.alloc_zeros(softmax_workspace_bytes)?,
            max_rows,
            vocab,
        })
    }
}

/// Batched temperature/top-k/top-p sampling: gathers the requested bf16 arena
/// rows, then runs FlashInfer's batched softmax + sampling — three kernel
/// launches, one sync, and one D2H for the whole batch.
///
/// `seed` must be fresh per decode step (one philox seed per call; rows
/// decorrelate through the philox subsequence). Returns one token per row, in
/// `rows` order.
///
/// min_p rows run as their own pass: if they shared a call, every row would
/// ride the min_p kernel, whose u-scaling (`u * q`) and survivor predicate
/// differ from the fused fast path — a min_p == 0 row could then sample a
/// different token than it would alone. Partitioning here (not in callers)
/// keeps "min_p == 0 rows take the original path" true for every caller, at
/// the cost of a second full pass (gather + softmax + sample, own sync) only
/// when a batch actually mixes.
pub fn gpu_sample_batch_into(
    ctx: &DeviceContext,
    logits: HiddenStatesRef<'_>,
    rows: &[BatchSamplingRow],
    seed: u64,
    scratch: &mut BatchSamplingScratch,
) -> Result<Vec<u32>> {
    if rows.iter().all(|r| r.min_p > 0.0) || rows.iter().all(|r| r.min_p <= 0.0) {
        return sample_uniform_batch_into(ctx, logits, rows, seed, scratch);
    }
    let (minp, plain): (Vec<BatchSamplingRow>, Vec<BatchSamplingRow>) =
        rows.iter().copied().partition(|r| r.min_p > 0.0);
    let plain_tokens = sample_uniform_batch_into(ctx, logits, &plain, seed, scratch)?;
    // Distinct philox key for the second pass: both passes restart their
    // subsequences at 0, so reusing `seed` would hand minp row i the same
    // uniform stream as plain row i and correlate their tokens.
    let minp_seed = seed ^ 0x9E37_79B9_7F4A_7C15;
    let minp_tokens = sample_uniform_batch_into(ctx, logits, &minp, minp_seed, scratch)?;
    let mut plain_it = plain_tokens.into_iter();
    let mut minp_it = minp_tokens.into_iter();
    Ok(rows
        .iter()
        .map(|r| {
            if r.min_p > 0.0 {
                minp_it.next().expect("minp token per minp row")
            } else {
                plain_it.next().expect("plain token per plain row")
            }
        })
        .collect())
}

/// One homogeneous FlashInfer pass (rows are all-min_p or all-plain).
fn sample_uniform_batch_into(
    ctx: &DeviceContext,
    logits: HiddenStatesRef<'_>,
    rows: &[BatchSamplingRow],
    seed: u64,
    scratch: &mut BatchSamplingScratch,
) -> Result<Vec<u32>> {
    let n = rows.len();
    ensure!(n > 0, "batch sampling requires at least one row");
    ensure!(
        n <= scratch.max_rows,
        "batch sampling scratch too small: {n} rows > capacity {}",
        scratch.max_rows
    );
    ensure!(
        logits.hidden_dim == scratch.vocab,
        "batch sampling vocab mismatch: logits {} vs scratch {}",
        logits.hidden_dim,
        scratch.vocab
    );

    let mut row_indices = Vec::with_capacity(n);
    let mut temperature = Vec::with_capacity(n);
    let mut top_k = Vec::with_capacity(n);
    let mut top_p = Vec::with_capacity(n);
    let mut min_p = Vec::with_capacity(n);
    let mut has_top_k_filter = false;
    let mut has_top_p_filter = false;
    let mut has_min_p_filter = false;
    for r in rows {
        ensure!(
            r.row < logits.seq_len,
            "batch sampling row {} out of arena range {}",
            r.row,
            logits.seq_len
        );
        ensure!(
            r.temperature > 0.0 && r.temperature.is_finite(),
            "batch sampling temperature {} must be finite and > 0 (greedy rows take the argmax path)",
            r.temperature
        );
        ensure!(
            r.top_p > 0.0 && r.top_p <= 1.0,
            "batch sampling top_p {} must be in (0, 1]",
            r.top_p
        );
        ensure!(
            (0.0..1.0).contains(&r.min_p) && r.min_p.is_finite(),
            "batch sampling min_p {} must be in [0, 1)",
            r.min_p
        );
        row_indices.push(i32::try_from(r.row)?);
        temperature.push(r.temperature);
        // FlashInfer reads top_k as u32; "disabled" is any k >= vocab.
        let vocab = i32::try_from(scratch.vocab)?;
        let clamped_top_k = if r.top_k > 0 && r.top_k < vocab {
            has_top_k_filter = true;
            r.top_k
        } else {
            vocab
        };
        top_k.push(clamped_top_k);
        if r.top_p < 1.0 {
            has_top_p_filter = true;
        }
        top_p.push(r.top_p);
        if r.min_p > 0.0 {
            has_min_p_filter = true;
        }
        min_p.push(r.min_p);
    }
    ctx.stream
        .memcpy_htod(&row_indices, &mut scratch.row_indices)?;
    ctx.stream
        .memcpy_htod(&temperature, &mut scratch.temperature)?;
    ctx.stream.memcpy_htod(&top_k, &mut scratch.top_k)?;
    ctx.stream.memcpy_htod(&top_p, &mut scratch.top_p)?;
    if has_min_p_filter {
        ctx.stream.memcpy_htod(&min_p, &mut scratch.min_p)?;
    }

    {
        let softmax_workspace_bytes = scratch.softmax_workspace.len();
        let (logits_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
        let (indices_ptr, _gi) = scratch.row_indices.device_ptr(&ctx.stream);
        let (probs_ptr, _gp) = scratch.probs.device_ptr_mut(&ctx.stream);
        let (temp_ptr, _gt) = scratch.temperature.device_ptr(&ctx.stream);
        let (top_k_ptr, _gk) = scratch.top_k.device_ptr(&ctx.stream);
        let (top_p_ptr, _gtp) = scratch.top_p.device_ptr(&ctx.stream);
        let (min_p_ptr, _gmp) = scratch.min_p.device_ptr(&ctx.stream);
        // topk_row_states is only read on the min_p pipeline; the fast path
        // hands the kernel a null instead of borrowing the buffer.
        let row_states = if has_min_p_filter {
            Some(scratch.topk_row_states.device_ptr_mut(&ctx.stream))
        } else {
            None
        };
        let row_states_ptr = row_states.as_ref().map_or(0, |(ptr, _guard)| *ptr);
        let (valid_ptr, _gv) = scratch.valid.device_ptr_mut(&ctx.stream);
        let (out_ptr, _go) = scratch.out.device_ptr_mut(&ctx.stream);
        let (ws_ptr, _gw) = scratch.softmax_workspace.device_ptr_mut(&ctx.stream);

        let err = unsafe {
            ffi::gpu_sample_batch_flashinfer_cuda(
                logits_ptr as *const ffi::Half,
                indices_ptr as *const i32,
                probs_ptr as *mut f32,
                temp_ptr as *const f32,
                top_k_ptr as *const i32,
                top_p_ptr as *const f32,
                if has_min_p_filter {
                    min_p_ptr as *const f32
                } else {
                    std::ptr::null()
                },
                row_states_ptr as *mut u8,
                valid_ptr as *mut u8,
                out_ptr as *mut i32,
                ws_ptr as *mut u8,
                softmax_workspace_bytes,
                n as i32,
                scratch.vocab as i32,
                i32::from(has_top_k_filter),
                i32::from(has_top_p_filter),
                seed,
                0,
                crate::tensor::active_cu_stream(ctx),
            )
        };
        ensure!(
            err == 0,
            "batch sampling kernel failed with error {err}{}",
            crate::ops::ffi_exception_message(err)
        );
    }

    let out = ctx
        .stream
        .clone_dtoh(&scratch.out)
        .map_err(|e| anyhow!("D2H batch sample read failed: {e}"))?;
    let valid = ctx
        .stream
        .clone_dtoh(&scratch.valid)
        .map_err(|e| anyhow!("D2H batch sample valid read failed: {e}"))?;
    ctx.sync()?;

    let mut tokens = Vec::with_capacity(n);
    for (i, r) in rows.iter().enumerate() {
        ensure!(
            valid[i] != 0,
            "batch sampling produced no valid token for arena row {} (probs failed to cover u)",
            r.row
        );
        ensure!(
            out[i] >= 0 && (out[i] as usize) < scratch.vocab,
            "batch sampling token {} for arena row {} out of vocab range {}",
            out[i],
            r.row,
            scratch.vocab
        );
        tokens.push(out[i] as u32);
    }
    Ok(tokens)
}

/// Argmax — returns the index of the maximum element.
///
/// Allocates a temporary output buffer. Model decode paths use batched argmax
/// through `pegainfer-sample`'s `select_batch`.
pub fn argmax(ctx: &DeviceContext, x: &DeviceVec) -> Result<u32> {
    let mut out_gpu: CudaSlice<i32> = ctx
        .stream
        .alloc_zeros(1)
        .map_err(|e| anyhow!("Alloc failed: {}", e))?;

    {
        let (x_ptr, _gx) = x.data.device_ptr(&ctx.stream);
        let (out_ptr, _go) = out_gpu.device_ptr_mut(&ctx.stream);

        unsafe {
            ffi::argmax_cuda(
                x_ptr as *const ffi::Half,
                out_ptr as *mut i32,
                x.len as i32,
                crate::tensor::active_cu_stream(ctx),
            );
        }
    }

    let result = ctx
        .stream
        .clone_dtoh(&out_gpu)
        .map_err(|e| anyhow!("D2H copy failed: {}", e))?;
    ctx.sync()?;

    Ok(result[0] as u32)
}

pub fn argmax_batch_bf16_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    values: &mut CudaSlice<half::bf16>,
    out: &mut CudaSlice<i32>,
) -> Result<()> {
    let rows = logits.seq_len;
    if rows == 0 {
        return Err(anyhow!("argmax batch requires at least one row"));
    }
    if values.len() < rows {
        return Err(anyhow!(
            "argmax batch values scratch too small: have {}, need {}",
            values.len(),
            rows
        ));
    }
    if out.len() < rows {
        return Err(anyhow!(
            "argmax batch output too small: have {}, need {}",
            out.len(),
            rows
        ));
    }

    let (logits_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
    let (values_ptr, _gv) = values.device_ptr_mut(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::argmax_batch_bf16_cuda(
            logits_ptr as *const ffi::Half,
            values_ptr as *mut ffi::Half,
            out_ptr as *mut i32,
            rows as i32,
            logits.hidden_dim as i32,
            crate::tensor::active_cu_stream(ctx),
        );
    }

    Ok(())
}

pub fn argmax_batch_bf16_split_partials_len(rows: usize, vocab: usize) -> usize {
    const TILE_ELEMS: usize = 4096;
    rows * vocab.div_ceil(TILE_ELEMS)
}

/// Partial-buffer length for the Markov-step argmax, whose tiles are 1024
/// elements: its read is one logits row + one bias row, so it needs 4x the
/// blocks of the full-row batched argmax to not be latency-bound (see
/// `MARKOV_STEP_TILE_ELEMS` in `argmax.cu`).
pub fn markov_step_argmax_partials_len(rows: usize, vocab: usize) -> usize {
    const TILE_ELEMS: usize = 1024;
    // Saturating: an overflowing request then fails every buffer-length check
    // downstream instead of wrapping into a small (falsely satisfiable) need.
    rows.saturating_mul(vocab.div_ceil(TILE_ELEMS))
}

/// Row-wise two-stage bf16 argmax over `rows` rows of `n`: tile-parallel
/// partials then one finalize block per row. Same per-row total order as the
/// retired single-row `argmax_bf16_into` (lowest GLOBAL index wins ties, NaN
/// never wins) — the
/// partials carry global indices, so each row's result is bit-identical to
/// the single-block scan (and independent of the other rows) while each vocab
/// row spreads over ~n/4096 CTAs instead of one. `partial_*` must hold
/// `argmax_batch_bf16_split_partials_len(rows, n)` elements; `values`/`indices`
/// hold one element per row.
#[allow(clippy::too_many_arguments)]
pub fn argmax_bf16_split_into(
    ctx: &DeviceContext,
    logits: &CudaSlice<half::bf16>,
    rows: usize,
    n: usize,
    partial_values: &mut CudaSlice<f32>,
    partial_indices: &mut CudaSlice<i32>,
    values: &mut CudaSlice<half::bf16>,
    indices: &mut CudaSlice<i32>,
) -> Result<()> {
    if rows == 0 || n == 0 || logits.len() < rows * n {
        return Err(anyhow!(
            "argmax_bf16_split_into logits too small: have {}, need {}",
            logits.len(),
            rows * n
        ));
    }
    if values.len() < rows || indices.len() < rows {
        return Err(anyhow!(
            "argmax_bf16_split_into outputs must hold {rows} elements: have {}/{}",
            values.len(),
            indices.len()
        ));
    }
    let needed = argmax_batch_bf16_split_partials_len(rows, n);
    if partial_values.len() < needed || partial_indices.len() < needed {
        return Err(anyhow!(
            "argmax_bf16_split_into partials too small: {}/{} need {needed}",
            partial_values.len(),
            partial_indices.len()
        ));
    }
    let (x_ptr, _gx) = logits.device_ptr(&ctx.stream);
    let (pv_ptr, _gpv) = partial_values.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = partial_indices.device_ptr_mut(&ctx.stream);
    let (v_ptr, _gv) = values.device_ptr_mut(&ctx.stream);
    let (i_ptr, _gi) = indices.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::argmax_batch_bf16_split_cuda(
            x_ptr as *const ffi::Half,
            v_ptr as *mut ffi::Half,
            i_ptr as *mut i32,
            pv_ptr as *mut f32,
            pi_ptr as *mut i32,
            rows as i32,
            n as i32,
            crate::tensor::active_cu_stream(ctx),
        );
    }
    Ok(())
}

/// Two-stage indexed batched argmax: tile-parallel partials then a per-row
/// finalize. Lowest index wins ties; each vocab row spreads over many CTAs
/// instead of one.
#[allow(clippy::too_many_arguments)]
pub fn argmax_batch_bf16_split_indexed_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    row_indices: &CudaSlice<i32>,
    rows: usize,
    partial_values: &mut CudaSlice<f32>,
    partial_indices: &mut CudaSlice<i32>,
    values: &mut CudaSlice<half::bf16>,
    out: &mut CudaSlice<i32>,
) -> Result<()> {
    if rows == 0 {
        return Err(anyhow!(
            "argmax split indexed batch requires at least one row"
        ));
    }
    if row_indices.len() < rows {
        return Err(anyhow!(
            "argmax split indexed row scratch too small: have {}, need {}",
            row_indices.len(),
            rows
        ));
    }
    let needed_partials = argmax_batch_bf16_split_partials_len(rows, logits.hidden_dim);
    if partial_values.len() < needed_partials || partial_indices.len() < needed_partials {
        return Err(anyhow!(
            "argmax split indexed partials scratch too small: have {}/{}, need {}",
            partial_values.len(),
            partial_indices.len(),
            needed_partials
        ));
    }
    if values.len() < rows || out.len() < rows {
        return Err(anyhow!(
            "argmax split indexed outputs too small: have {}/{}, need {}",
            values.len(),
            out.len(),
            rows
        ));
    }

    let (logits_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
    let (row_indices_ptr, _gr) = row_indices.device_ptr(&ctx.stream);
    let (pv_ptr, _gpv) = partial_values.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = partial_indices.device_ptr_mut(&ctx.stream);
    let (values_ptr, _gv) = values.device_ptr_mut(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::argmax_batch_bf16_split_indexed_cuda(
            logits_ptr as *const ffi::Half,
            row_indices_ptr as *const i32,
            values_ptr as *mut ffi::Half,
            out_ptr as *mut i32,
            pv_ptr as *mut f32,
            pi_ptr as *mut i32,
            rows as i32,
            logits.hidden_dim as i32,
            crate::tensor::active_cu_stream(ctx),
        );
    }

    Ok(())
}

/// Launch the per-row logprob reduction into pre-allocated buffers laid out
/// `[rows]` / `[rows * k_max]` row-major; each row writes `top_k[i] <= k_max`
/// entries and leaves the tail untouched.
///
/// # Safety
///
/// The device-resident contents of `row_indices`, `picked`, and `top_k`
/// cannot be validated here: for every `i < rows`, `row_indices[i]` must lie
/// in `[0, logits.seq_len)`, `picked[i]` in `[0, logits.hidden_dim)`, and
/// `top_k[i]` in `[0, min(k_max, logits.hidden_dim)]`. The launch goes to
/// `active_cu_stream(ctx)`: input buffers must be populated before the
/// launch and outputs read back after it, in that stream's order.
#[allow(clippy::too_many_arguments)]
pub unsafe fn logprob_topk_batch_bf16_into(
    ctx: &DeviceContext,
    logits: HiddenStatesRef<'_>,
    row_indices: &CudaSlice<i32>,
    picked: &CudaSlice<i32>,
    top_k: &CudaSlice<i32>,
    rows: usize,
    k_max: usize,
    out_picked_lp: &mut CudaSlice<f32>,
    out_topk_vals: &mut CudaSlice<f32>,
    out_topk_ids: &mut CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        rows > 0 && logits.hidden_dim > 0,
        "logprob_topk requires rows > 0 and a non-empty vocab"
    );
    let backing = logits
        .hidden_dim
        .checked_mul(logits.seq_len)
        .ok_or_else(|| anyhow!("logprob_topk arena extent overflows"))?;
    ensure!(
        logits.data.len() >= backing,
        "logprob_topk arena backing too small: have {}, need {backing}",
        logits.data.len()
    );
    ensure!(
        row_indices.len() >= rows && picked.len() >= rows && top_k.len() >= rows,
        "logprob_topk inputs must hold {rows} elements: have {}/{}/{}",
        row_indices.len(),
        picked.len(),
        top_k.len()
    );
    ensure!(
        out_picked_lp.len() >= rows,
        "logprob_topk picked output too small: have {}, need {rows}",
        out_picked_lp.len()
    );
    let topk_needed = rows
        .checked_mul(k_max)
        .ok_or_else(|| anyhow!("logprob_topk rows * k_max overflows"))?;
    ensure!(
        out_topk_vals.len() >= topk_needed && out_topk_ids.len() >= topk_needed,
        "logprob_topk top-k outputs too small: have {}/{}, need {topk_needed}",
        out_topk_vals.len(),
        out_topk_ids.len()
    );
    let rows_i32 = i32::try_from(rows)?;
    let vocab_i32 = i32::try_from(logits.hidden_dim)?;
    let k_max_i32 = i32::try_from(k_max)?;

    let (x_ptr, _gx) = logits.data.device_ptr(&ctx.stream);
    let (rows_ptr, _gr) = row_indices.device_ptr(&ctx.stream);
    let (picked_ptr, _gp) = picked.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = top_k.device_ptr(&ctx.stream);
    let (lp_ptr, _gl) = out_picked_lp.device_ptr_mut(&ctx.stream);
    let (vals_ptr, _gv) = out_topk_vals.device_ptr_mut(&ctx.stream);
    let (ids_ptr, _gi) = out_topk_ids.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::logprob_topk_batch_bf16_cuda(
            x_ptr as *const ffi::Half,
            rows_ptr as *const i32,
            picked_ptr as *const i32,
            k_ptr as *const i32,
            lp_ptr as *mut f32,
            vals_ptr as *mut f32,
            ids_ptr as *mut i32,
            rows_i32,
            vocab_i32,
            k_max_i32,
            crate::tensor::active_cu_stream(ctx),
        );
    }

    Ok(())
}

/// DSpark Markov-head step argmax. For each request `row`, argmax over
/// `base[row*block_size + step] + bias[row]` and write the chosen token id as
/// u32 (so it feeds straight back as the next step's prev-token lookup).
/// `base` is the request-major block logits `[rows*block_size, vocab]`; `bias`
/// is the per-request Markov logit bias `[rows, vocab]` for this step. `partial_*`
/// must hold [`markov_step_argmax_partials_len`] elements — this path tiles
/// finer than the batched argmax, so its partials buffer is NOT the same size.
/// `sampled_tokens` receives the request-major block token at
/// `row * block_size + step`, allowing callers to D2H the finished block once.
#[allow(clippy::too_many_arguments)]
pub fn markov_step_argmax_into(
    ctx: &DeviceContext,
    base: &HiddenStates,
    bias: &HiddenStates,
    block_size: usize,
    step: usize,
    rows: usize,
    partial_values: &mut CudaSlice<f32>,
    partial_indices: &mut CudaSlice<i32>,
    out_tokens: &mut CudaSlice<u32>,
    sampled_tokens: &mut CudaSlice<u32>,
) -> Result<()> {
    markov_step_argmax_impl(
        ctx,
        base,
        bias,
        block_size,
        step,
        1,
        None,
        rows,
        partial_values,
        partial_indices,
        out_tokens,
        sampled_tokens,
    )
}

/// [`markov_step_argmax_into`] over `rows = groups * chains` chain-rows with a
/// per-chain-group request map: chain-row `r` belongs to group `r / chains`
/// and reads base block `req_map[r / chains]`, so a hedged subset that is not
/// a batch prefix still addresses its own requests' logits.
///
/// # Safety
///
/// The map lives on device, so its CONTENTS cannot be validated here. The
/// caller must guarantee every `req_map[0..rows/chains]` value is in
/// `0..base.seq_len / block_size` — the kernel uses them directly as base
/// block indices, and an out-of-range or negative value reads out of bounds
/// on the GPU. Validate on the host before uploading.
#[allow(clippy::too_many_arguments)]
pub unsafe fn markov_step_argmax_mapped_into(
    ctx: &DeviceContext,
    base: &HiddenStates,
    bias: &HiddenStates,
    block_size: usize,
    step: usize,
    chains: usize,
    req_map: &CudaSlice<i32>,
    rows: usize,
    partial_values: &mut CudaSlice<f32>,
    partial_indices: &mut CudaSlice<i32>,
    out_tokens: &mut CudaSlice<u32>,
    sampled_tokens: &mut CudaSlice<u32>,
) -> Result<()> {
    markov_step_argmax_impl(
        ctx,
        base,
        bias,
        block_size,
        step,
        chains,
        Some(req_map),
        rows,
        partial_values,
        partial_indices,
        out_tokens,
        sampled_tokens,
    )
}

#[allow(clippy::too_many_arguments)]
fn markov_step_argmax_impl(
    ctx: &DeviceContext,
    base: &HiddenStates,
    bias: &HiddenStates,
    block_size: usize,
    step: usize,
    chains: usize,
    req_map: Option<&CudaSlice<i32>>,
    rows: usize,
    partial_values: &mut CudaSlice<f32>,
    partial_indices: &mut CudaSlice<i32>,
    out_tokens: &mut CudaSlice<u32>,
    sampled_tokens: &mut CudaSlice<u32>,
) -> Result<()> {
    if rows == 0 {
        return Err(anyhow!("markov step argmax requires at least one row"));
    }
    if base.hidden_dim == 0 {
        return Err(anyhow!(
            "markov step argmax requires a non-zero vocab (zero tiles would launch an empty grid)"
        ));
    }
    if step >= block_size {
        return Err(anyhow!(
            "markov step step {step} >= block_size {block_size}"
        ));
    }
    base.checked_extent("markov step base")?;
    bias.checked_extent("markov step bias")?;
    // The kernels derive their indices in signed int, so the products need
    // bounding, not just each factor.
    super::checked_i32(base.seq_len, "markov step base rows")?;
    if rows > 65_535 {
        return Err(anyhow!("markov step rows {rows} exceeds grid-y limit"));
    }
    if base.hidden_dim > (i32::MAX as usize) - 1024 {
        return Err(anyhow!("markov step vocab {} too large", base.hidden_dim));
    }
    let vocab = base.hidden_dim;
    if bias.hidden_dim != vocab {
        return Err(anyhow!(
            "markov step bias vocab {} != base vocab {}",
            bias.hidden_dim,
            vocab
        ));
    }
    if chains == 0 || !rows.is_multiple_of(chains) {
        return Err(anyhow!(
            "markov step rows {rows} not divisible by chains {chains}"
        ));
    }
    let base_needed = (rows / chains)
        .checked_mul(block_size)
        .ok_or_else(|| anyhow!("markov step (rows/chains)*block_size overflow"))?;
    if req_map.is_none() && base.seq_len < base_needed {
        return Err(anyhow!(
            "markov step base rows {} < (rows/chains)*block_size {}",
            base.seq_len,
            base_needed
        ));
    }
    if let Some(map) = req_map {
        if map.len() < rows / chains {
            return Err(anyhow!(
                "markov step req_map {} < chain groups {}",
                map.len(),
                rows / chains
            ));
        }
    }
    if bias.seq_len < rows {
        return Err(anyhow!("markov step bias rows {} < {}", bias.seq_len, rows));
    }
    if out_tokens.len() < rows {
        return Err(anyhow!(
            "markov step out too small: {} < {}",
            out_tokens.len(),
            rows
        ));
    }
    let sampled_needed = rows
        .checked_mul(block_size)
        .ok_or_else(|| anyhow!("markov step rows*block_size overflow"))?;
    if sampled_tokens.len() < sampled_needed {
        return Err(anyhow!(
            "markov sampled-token scratch too small: {} < {}",
            sampled_tokens.len(),
            sampled_needed
        ));
    }
    let needed = markov_step_argmax_partials_len(rows, vocab);
    if partial_values.len() < needed || partial_indices.len() < needed {
        return Err(anyhow!(
            "markov step partials too small: {}/{} need {}",
            partial_values.len(),
            partial_indices.len(),
            needed
        ));
    }
    if needed > i32::MAX as usize || sampled_needed > i32::MAX as usize {
        return Err(anyhow!(
            "markov step index space too large: partials {needed}, sampled {sampled_needed}"
        ));
    }

    let (base_ptr, _gb) = base.data.device_ptr(&ctx.stream);
    let (bias_ptr, _gbi) = bias.data.device_ptr(&ctx.stream);
    let (pv_ptr, _gpv) = partial_values.device_ptr_mut(&ctx.stream);
    let (pi_ptr, _gpi) = partial_indices.device_ptr_mut(&ctx.stream);
    let (out_ptr, _go) = out_tokens.device_ptr_mut(&ctx.stream);
    let (sampled_ptr, _gs) = sampled_tokens.device_ptr_mut(&ctx.stream);
    let map_guard = req_map.map(|map| map.device_ptr(&ctx.stream));
    let map_ptr = map_guard
        .as_ref()
        .map_or(std::ptr::null(), |(ptr, _)| *ptr as *const i32);

    let rc = unsafe {
        ffi::markov_step_argmax_cuda(
            base_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            super::checked_i32(block_size, "markov step block_size")?,
            super::checked_i32(step, "markov step step")?,
            super::checked_i32(chains, "markov step chains")?,
            map_ptr,
            super::checked_i32(rows, "markov step rows")?,
            super::checked_i32(vocab, "markov step vocab")?,
            pv_ptr as *mut f32,
            pi_ptr as *mut i32,
            out_ptr as *mut u32,
            sampled_ptr as *mut u32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if rc != 0 {
        return Err(anyhow!("markov step argmax launch failed: cuda error {rc}"));
    }

    Ok(())
}

/// Top-2 of `base[(row*block+step), v] + bias[row, v]` per row (exact for
/// rows with two or more finite candidates; see the sentinel below). The
/// top-1 is written exactly like [`markov_step_argmax_into`]'s output (the
/// ping-pong `output` plus the `sampled_tokens` stamp), so a branch step needs
/// no separate argmax pass; the runner-up lands in `out_top2` — the
/// speculative hedge's branch source.
/// A row with fewer than two finite logits returns index 0 as the runner-up
/// sentinel (possibly duplicating top-1); callers must deduplicate — the
/// hedge expansion drops chains identical to the greedy one.
#[allow(clippy::too_many_arguments)]
pub fn markov_step_top2_into(
    ctx: &DeviceContext,
    base: &HiddenStates,
    bias: &HiddenStates,
    block_size: usize,
    step: usize,
    rows: usize,
    partial_v1: &mut CudaSlice<f32>,
    partial_i1: &mut CudaSlice<i32>,
    partial_v2: &mut CudaSlice<f32>,
    partial_i2: &mut CudaSlice<i32>,
    output: &mut CudaSlice<u32>,
    sampled_tokens: &mut CudaSlice<u32>,
    out_top2: &mut CudaSlice<u32>,
) -> Result<()> {
    if rows == 0 {
        return Err(anyhow!("markov step top2 requires at least one row"));
    }
    if base.hidden_dim == 0 {
        return Err(anyhow!(
            "markov step top2 requires a non-zero vocab (zero tiles would launch an empty grid)"
        ));
    }
    if step >= block_size {
        return Err(anyhow!(
            "markov top2 step {step} >= block_size {block_size}"
        ));
    }
    base.checked_extent("markov top2 base")?;
    bias.checked_extent("markov top2 bias")?;
    super::checked_i32(base.seq_len, "markov top2 base rows")?;
    if rows > 65_535 {
        return Err(anyhow!("markov top2 rows {rows} exceeds grid-y limit"));
    }
    if base.hidden_dim > (i32::MAX as usize) - 1024 {
        return Err(anyhow!("markov top2 vocab {} too large", base.hidden_dim));
    }
    let sampled_needed = rows
        .checked_mul(block_size)
        .ok_or_else(|| anyhow!("markov top2 rows*block_size overflow"))?;
    if sampled_tokens.len() < sampled_needed {
        return Err(anyhow!(
            "markov top2 sampled buffer {} < rows*block_size {}",
            sampled_tokens.len(),
            sampled_needed
        ));
    }
    let vocab = base.hidden_dim;
    if bias.hidden_dim != vocab {
        return Err(anyhow!(
            "markov top2 bias vocab {} != base vocab {}",
            bias.hidden_dim,
            vocab
        ));
    }
    let base_needed = rows
        .checked_mul(block_size)
        .ok_or_else(|| anyhow!("markov top2 rows*block_size overflow"))?;
    if base.seq_len < base_needed {
        return Err(anyhow!(
            "markov top2 base rows {} < rows*block_size {}",
            base.seq_len,
            base_needed
        ));
    }
    if bias.seq_len < rows {
        return Err(anyhow!("markov top2 bias rows {} < {}", bias.seq_len, rows));
    }
    if output.len() < rows || out_top2.len() < rows {
        return Err(anyhow!(
            "markov top2 out too small: {}/{} < {}",
            output.len(),
            out_top2.len(),
            rows
        ));
    }
    let needed = markov_step_argmax_partials_len(rows, vocab);
    if partial_v1.len() < needed
        || partial_i1.len() < needed
        || partial_v2.len() < needed
        || partial_i2.len() < needed
    {
        return Err(anyhow!(
            "markov top2 partials too small: need {needed} per pair"
        ));
    }
    if needed > i32::MAX as usize || sampled_needed > i32::MAX as usize {
        return Err(anyhow!(
            "markov top2 index space too large: partials {needed}, sampled {sampled_needed}"
        ));
    }

    let (base_ptr, _gb) = base.data.device_ptr(&ctx.stream);
    let (bias_ptr, _gbi) = bias.data.device_ptr(&ctx.stream);
    let (pv1_ptr, _g1) = partial_v1.device_ptr_mut(&ctx.stream);
    let (pi1_ptr, _g2) = partial_i1.device_ptr_mut(&ctx.stream);
    let (pv2_ptr, _g3) = partial_v2.device_ptr_mut(&ctx.stream);
    let (pi2_ptr, _g4) = partial_i2.device_ptr_mut(&ctx.stream);
    let (out_ptr, _g5) = output.device_ptr_mut(&ctx.stream);
    let (samp_ptr, _g6) = sampled_tokens.device_ptr_mut(&ctx.stream);
    let (o2_ptr, _g7) = out_top2.device_ptr_mut(&ctx.stream);

    let block_i = super::checked_i32(block_size, "markov top2 block_size")?;
    let step_i = super::checked_i32(step, "markov top2 step")?;
    // One chain-row per request: the ladder's multi-chain rows go through the
    // argmax path, which has the mapped variant.
    let chains_i = 1i32;
    let rows_i = super::checked_i32(rows, "markov top2 rows")?;
    let vocab_i = super::checked_i32(vocab, "markov top2 vocab")?;
    let rc = unsafe {
        ffi::markov_step_top2_cuda(
            base_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            block_i,
            step_i,
            chains_i,
            rows_i,
            vocab_i,
            pv1_ptr as *mut f32,
            pi1_ptr as *mut i32,
            pv2_ptr as *mut f32,
            pi2_ptr as *mut i32,
            out_ptr as *mut u32,
            samp_ptr as *mut u32,
            o2_ptr as *mut u32,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if rc != 0 {
        return Err(anyhow!("markov top2 launch failed: cuda error {rc}"));
    }

    Ok(())
}

/// Hedge-ladder branch force (see `hedge_ladder_force_cuda`): stamps chain
/// `(i, j)`'s device-resident runner token into the prev buffer and the
/// sampled block at `step`, for all `n` requests.
#[allow(clippy::too_many_arguments)]
///
/// # Safety
///
/// `req_map` lives on device and cannot be validated here: every
/// `req_map[0..n]` value must be in `0..runner_stride`, or the runner gather
/// reads out of bounds on the GPU. Validate on the host before uploading.
pub unsafe fn hedge_ladder_force_into(
    ctx: &DeviceContext,
    prev_tokens: &mut CudaSlice<u32>,
    sampled_tokens: &mut CudaSlice<u32>,
    runner_tokens: &CudaSlice<u32>,
    req_map: &CudaSlice<i32>,
    n: usize,
    c: usize,
    j: usize,
    runner_stride: usize,
    block_size: usize,
    step: usize,
) -> Result<()> {
    if n == 0 || c == 0 || j >= c {
        return Err(anyhow!("hedge force bad shape: n={n} c={c} j={j}"));
    }
    let rows = n
        .checked_mul(c)
        .ok_or_else(|| anyhow!("hedge force n*c overflow"))?;
    let sampled_needed = rows
        .checked_mul(block_size)
        .ok_or_else(|| anyhow!("hedge force n*c*block_size overflow"))?;
    // Mapped indices may address anywhere in `0..runner_stride`, so the WHOLE
    // stripe must be backed, not just its first `n` entries.
    let runner_needed = j
        .checked_add(1)
        .and_then(|v| v.checked_mul(runner_stride))
        .ok_or_else(|| anyhow!("hedge force runner stripe overflow"))?;
    if prev_tokens.len() < rows
        || sampled_tokens.len() < sampled_needed
        || runner_tokens.len() < runner_needed
    {
        return Err(anyhow!("hedge force buffers too small"));
    }
    if step >= block_size {
        return Err(anyhow!(
            "hedge force step {step} >= block_size {block_size}"
        ));
    }
    if runner_stride < n {
        return Err(anyhow!("hedge force runner stride {runner_stride} < n {n}"));
    }
    if req_map.len() < n {
        return Err(anyhow!("hedge force req_map {} < n {n}", req_map.len()));
    }
    if n > (i32::MAX as usize) - 128 {
        return Err(anyhow!("hedge force n {n} too large"));
    }
    if sampled_needed > i32::MAX as usize || runner_needed > i32::MAX as usize {
        return Err(anyhow!(
            "hedge force index space too large: sampled {sampled_needed}, runner {runner_needed}"
        ));
    }
    let n_i = super::checked_i32(n, "hedge force n")?;
    let c_i = super::checked_i32(c, "hedge force c")?;
    let j_i = super::checked_i32(j, "hedge force j")?;
    let stride_i = super::checked_i32(runner_stride, "hedge force runner_stride")?;
    let block_i = super::checked_i32(block_size, "hedge force block_size")?;
    let step_i = super::checked_i32(step, "hedge force step")?;
    let (prev_ptr, _g1) = prev_tokens.device_ptr_mut(&ctx.stream);
    let (samp_ptr, _g2) = sampled_tokens.device_ptr_mut(&ctx.stream);
    let (run_ptr, _g3) = runner_tokens.device_ptr(&ctx.stream);
    let (map_ptr, _g4) = req_map.device_ptr(&ctx.stream);
    let rc = unsafe {
        ffi::hedge_ladder_force_cuda(
            prev_ptr as *mut u32,
            samp_ptr as *mut u32,
            run_ptr as *const u32,
            map_ptr as *const i32,
            n_i,
            c_i,
            j_i,
            stride_i,
            block_i,
            step_i,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    if rc != 0 {
        return Err(anyhow!("hedge ladder force launch failed: cuda error {rc}"));
    }
    Ok(())
}

pub fn flashinfer_top1_row_states_bytes() -> usize {
    unsafe { ffi::flashinfer_top1_row_states_bytes_cuda() }
}

pub fn flashinfer_top1_batch_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    top1_values: &mut CudaSlice<half::bf16>,
    row_states_scratch: &mut CudaSlice<u8>,
    out: &mut CudaSlice<i32>,
) -> Result<()> {
    let rows = logits.seq_len;
    if top1_values.len() < rows {
        return Err(anyhow!(
            "top1 values scratch too small: have {}, need {}",
            top1_values.len(),
            rows
        ));
    }
    if out.len() < rows {
        return Err(anyhow!(
            "top1 output too small: have {}, need {}",
            out.len(),
            rows
        ));
    }
    let row_states_bytes = flashinfer_top1_row_states_bytes();
    if row_states_scratch.len() < row_states_bytes {
        return Err(anyhow!(
            "top1 row states scratch too small: have {}, need {}",
            row_states_scratch.len(),
            row_states_bytes
        ));
    }

    let (l_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = top1_values.device_ptr_mut(&ctx.stream);
    let (r_ptr, _gr) = row_states_scratch.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = out.device_ptr_mut(&ctx.stream);

    unsafe {
        ffi::flashinfer_top1_batch_cuda(
            l_ptr as *const ffi::Half,
            v_ptr as *mut ffi::Half,
            r_ptr as *mut u8,
            o_ptr as *mut i32,
            rows as i32,
            logits.hidden_dim as i32,
            crate::tensor::active_cu_stream(ctx),
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;

    const VOCAB: usize = 32768; // >= 24576 so OnlineSoftmax takes the vocab-splitting path
    const ARENA_ROWS: usize = 8;

    /// Arena where every row not under test is poisoned with a dominant logit
    /// at `POISON_TOKEN` — a broken row gather makes every assertion fail.
    const POISON_TOKEN: usize = 7777;

    fn arena_with_rows(ctx: &DeviceContext, rows: &[(usize, Vec<f32>)]) -> HiddenStates {
        let mut host = vec![bf16::from_f32(0.0); ARENA_ROWS * VOCAB];
        for r in 0..ARENA_ROWS {
            host[r * VOCAB + POISON_TOKEN] = bf16::from_f32(20.0);
        }
        for (row, values) in rows {
            assert_eq!(values.len(), VOCAB);
            for (i, v) in values.iter().enumerate() {
                host[row * VOCAB + i] = bf16::from_f32(*v);
            }
        }
        let data = ctx.stream.clone_htod(&host).expect("htod logits");
        HiddenStates {
            data,
            hidden_dim: VOCAB,
            seq_len: ARENA_ROWS,
        }
    }

    fn flat_row(fill: f32) -> Vec<f32> {
        vec![fill; VOCAB]
    }

    #[test]
    fn batch_sampling_honors_top_k_top_p_and_gathers_rows() {
        let ctx = DeviceContext::new().expect("create CUDA context");

        // Row 1: top_k=5 — five high tokens; the unmasked tail would win ~83%
        // of draws (32k tokens at e^2 vs five at e^8..e^10), so a missing
        // top-k mask fails immediately.
        let top5: Vec<usize> = vec![11, 503, 1024, 9000, 32000];
        let mut row_k = flat_row(2.0);
        for (i, &t) in top5.iter().enumerate() {
            row_k[t] = 10.0 - 0.5 * i as f32;
        }

        // Row 4: top_p=0.5 with one token holding ~83% of the mass — the
        // nucleus is exactly that token, so every draw must return it.
        let mut row_p = flat_row(0.0);
        row_p[222] = 12.0;

        // Row 6: near-zero temperature sharpens to argmax.
        let mut row_t = flat_row(0.0);
        row_t[31999] = 4.0;

        let logits = arena_with_rows(&ctx, &[(1, row_k), (4, row_p), (6, row_t)]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [
            BatchSamplingRow {
                row: 1,
                temperature: 1.0,
                top_k: 5,
                top_p: 1.0,
                min_p: 0.0,
            },
            BatchSamplingRow {
                row: 4,
                temperature: 1.0,
                top_k: -1,
                top_p: 0.5,
                min_p: 0.0,
            },
            BatchSamplingRow {
                row: 6,
                temperature: 0.05,
                top_k: -1,
                top_p: 1.0,
                min_p: 0.0,
            },
        ];

        for seed in 0..64u64 {
            let tokens = gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, seed, &mut scratch)
                .expect("sample");
            assert!(
                top5.contains(&(tokens[0] as usize)),
                "seed {seed}: top_k=5 row sampled {} outside the top-5 set",
                tokens[0]
            );
            assert_eq!(
                tokens[1], 222,
                "seed {seed}: top_p=0.5 row escaped the single-token nucleus"
            );
            assert_eq!(
                tokens[2], 31999,
                "seed {seed}: near-zero temperature row missed the argmax"
            );
        }
    }

    #[test]
    fn batch_sampling_top_p_only_small_nucleus_collapses_to_argmax() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        let mut row = flat_row(0.0);
        row[123] = 2.0;
        row[456] = 1.5;
        row[789] = 1.0;
        let logits = arena_with_rows(&ctx, &[(2, row)]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [BatchSamplingRow {
            row: 2,
            temperature: 1.0,
            top_k: -1,
            top_p: 1e-6,
            min_p: 0.0,
        }];
        let tokens =
            gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, 17, &mut scratch).expect("sample");
        assert_eq!(tokens, vec![123], "tiny top_p should collapse to argmax");
    }

    #[test]
    fn batch_sampling_same_seed_is_deterministic() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        // Two flat rows: uniform over 32768 tokens, so different seeds
        // colliding on both rows is ~1e-9.
        let logits = arena_with_rows(&ctx, &[(2, flat_row(0.0)), (5, flat_row(0.0))]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [
            BatchSamplingRow {
                row: 2,
                temperature: 1.0,
                top_k: -1,
                top_p: 1.0,
                min_p: 0.0,
            },
            BatchSamplingRow {
                row: 5,
                temperature: 1.0,
                top_k: -1,
                top_p: 1.0,
                min_p: 0.0,
            },
        ];

        let a =
            gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, 42, &mut scratch).expect("sample");
        let b =
            gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, 42, &mut scratch).expect("sample");
        let c =
            gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, 43, &mut scratch).expect("sample");
        assert_eq!(a, b, "same seed must reproduce the same tokens");
        assert_ne!(a, c, "different seeds must diverge on flat rows");
    }

    #[test]
    fn batch_sampling_applies_per_row_temperature() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        // Effective 2-token distribution: logit ln(3) vs 0, everything else
        // at -120 so the 32766-token tail stays negligible even after the
        // temperature=4 flattening (e^-30 x 32766 ≈ 3e-9). P(token 100) =
        // 0.75 at temperature 1, 3^(1/4)/(3^(1/4)+1) ≈ 0.568 at temperature
        // 4. Fixed seed sequence + deterministic kernel make the observed
        // counts reproducible.
        let mut row = flat_row(-120.0);
        row[100] = 3.0f32.ln();
        row[200] = 0.0;
        let logits = arena_with_rows(&ctx, &[(3, row.clone()), (7, row)]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [
            BatchSamplingRow {
                row: 3,
                temperature: 1.0,
                top_k: -1,
                top_p: 1.0,
                min_p: 0.0,
            },
            BatchSamplingRow {
                row: 7,
                temperature: 4.0,
                top_k: -1,
                top_p: 1.0,
                min_p: 0.0,
            },
        ];

        let draws = 300;
        let mut hits = [0u32; 2];
        for seed in 0..draws {
            let tokens =
                gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, seed as u64, &mut scratch)
                    .expect("sample");
            for (i, &t) in tokens.iter().enumerate() {
                assert!(
                    t == 100 || t == 200,
                    "row {i} sampled {t}, outside the 2-token support"
                );
                if t == 100 {
                    hits[i] += 1;
                }
            }
        }
        let freq_t1 = f64::from(hits[0]) / f64::from(draws);
        let freq_t4 = f64::from(hits[1]) / f64::from(draws);
        assert!(
            (0.65..=0.85).contains(&freq_t1),
            "temperature=1 row frequency {freq_t1} outside [0.65, 0.85] (expected 0.75)"
        );
        assert!(
            (0.47..=0.67).contains(&freq_t4),
            "temperature=4 row frequency {freq_t4} outside [0.47, 0.67] (expected 0.568)"
        );
    }

    #[test]
    fn batch_sampling_min_p_masks_below_threshold() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        // Two-token support: P(100) ≈ 0.7, P(200) ≈ 0.28, tail ≈ 0 (logits
        // ln(0.7/0.28) apart, rest at -120). min_p thresholds against the max
        // prob: 0.5 * 0.7 = 0.35 keeps only token 100; 0.2 * 0.7 = 0.14 keeps
        // both.
        let mut row = flat_row(-120.0);
        row[100] = (0.7f32 / 0.28).ln();
        row[200] = 0.0;
        let logits = arena_with_rows(&ctx, &[(1, row.clone()), (3, row)]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");

        let strict = [BatchSamplingRow {
            row: 1,
            temperature: 1.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.5,
        }];
        for seed in 0..64u64 {
            let tokens = gpu_sample_batch_into(&ctx, logits.as_ref(), &strict, seed, &mut scratch)
                .expect("sample");
            assert_eq!(
                tokens[0], 100,
                "seed {seed}: min_p=0.5 must mask the 0.28-prob token"
            );
        }

        let loose = [BatchSamplingRow {
            row: 3,
            temperature: 1.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.2,
        }];
        let mut saw_minor = false;
        for seed in 0..128u64 {
            let tokens = gpu_sample_batch_into(&ctx, logits.as_ref(), &loose, seed, &mut scratch)
                .expect("sample");
            assert!(
                tokens[0] == 100 || tokens[0] == 200,
                "seed {seed}: min_p=0.2 sampled {} outside the surviving pair",
                tokens[0]
            );
            saw_minor |= tokens[0] == 200;
        }
        assert!(
            saw_minor,
            "min_p=0.2 never sampled the 0.28-prob token in 128 draws (~1e-19 if unmasked)"
        );
    }

    #[test]
    fn batch_sampling_min_p_composes_with_top_k_and_top_p() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        // Five spaced tokens; top_k=3 keeps {11, 503, 1024}, then the top-p /
        // min_p stages cut deeper. With min_p=0.6 after top-k renorm the
        // survivor set is exactly the argmax.
        let picks: Vec<usize> = vec![11, 503, 1024, 9000, 32000];
        let mut row = flat_row(-120.0);
        for (i, &t) in picks.iter().enumerate() {
            row[t] = 8.0 - 1.0 * i as f32;
        }
        let logits = arena_with_rows(&ctx, &[(2, row)]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [BatchSamplingRow {
            row: 2,
            temperature: 1.0,
            top_k: 3,
            top_p: 0.99,
            min_p: 0.6,
        }];
        for seed in 0..64u64 {
            let tokens = gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, seed, &mut scratch)
                .expect("sample");
            assert_eq!(
                tokens[0], 11,
                "seed {seed}: top_k=3 + min_p=0.6 must collapse to the argmax"
            );
        }
    }

    #[test]
    fn batch_sampling_rejects_greedy_rows() {
        let ctx = DeviceContext::new().expect("create CUDA context");
        let logits = arena_with_rows(&ctx, &[]);
        let mut scratch = BatchSamplingScratch::new(&ctx, ARENA_ROWS, VOCAB).expect("scratch");
        let rows = [BatchSamplingRow {
            row: 0,
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            min_p: 0.0,
        }];
        assert!(
            gpu_sample_batch_into(&ctx, logits.as_ref(), &rows, 1, &mut scratch).is_err(),
            "temperature=0 must be rejected — greedy rows take the argmax path"
        );
    }
}
