//! Probability-aware verification of a top-16 draft chain.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use super::sampling::BatchSamplingRow;
use super::sampling::BatchSamplingScratch;
use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::HiddenStatesRef;
use crate::tensor::active_cu_stream;

/// A retained verify span and its row in the preceding batched proposal.
#[derive(Clone, Copy)]
pub struct SparseVerifyRequest {
    /// First entry in the supplied sampling rows, followed by `num_drafts` rows.
    pub logits_offset: usize,
    pub draft_row: usize,
    pub num_drafts: usize,
}

/// The probability arena belongs to the existing sampler. This scratch holds
/// only request mappings, random inputs and the accepted-count/posterior pairs.
pub struct SparseVerifyScratch {
    requests: CudaSlice<i32>,
    uniforms: CudaSlice<f32>,
    output: CudaSlice<u32>,
    max_requests: usize,
    block_size: usize,
}

impl SparseVerifyScratch {
    pub fn new(ctx: &DeviceContext, max_requests: usize, block_size: usize) -> Result<Self> {
        ensure!(
            max_requests > 0 && block_size >= 2,
            "invalid sparse verify capacity"
        );
        Ok(Self {
            requests: ctx.stream.alloc_zeros(max_requests * 3)?,
            uniforms: ctx.stream.alloc_zeros(max_requests * block_size)?,
            output: ctx.stream.alloc_zeros(max_requests * 2)?,
            max_requests,
            block_size,
        })
    }

    pub fn reservation_bytes(max_requests: usize, block_size: usize) -> usize {
        max_requests * (block_size + 5) * 4
    }
}

/// Accept a sampled prefix and emit one correction or bonus token per request.
///
/// Candidate IDs include anchor rows (`[draft_batch, block, 16]`); q and draft
/// tokens omit them. Uniforms are `[requests, block]`: acceptance draws first,
/// then the posterior draw in the last column. All draws must be in `[0, 1)`.
/// The sampler parameters in each request's span must be identical. Results
/// retain request order even when min-p requests require a separate pass.
pub fn sparse_speculative_sample(
    ctx: &DeviceContext,
    logits: HiddenStatesRef<'_>,
    rows: &[BatchSamplingRow],
    requests: &[SparseVerifyRequest],
    candidate_ids: &CudaSlice<u32>,
    draft_probs: &CudaSlice<f32>,
    draft_ids: &CudaSlice<u32>,
    uniforms: &[f32],
    scratch: &mut SparseVerifyScratch,
    sampling: &mut BatchSamplingScratch,
) -> Result<Vec<(usize, u32)>> {
    let block = scratch.block_size;
    ensure!(
        requests.len() <= scratch.max_requests,
        "sparse verify request capacity exceeded"
    );
    ensure!(
        uniforms.len() == requests.len() * block,
        "sparse verify random input shape mismatch"
    );
    ensure!(
        logits.hidden_dim == sampling.vocab,
        "sparse verify vocabulary mismatch"
    );
    for req in requests {
        ensure!(
            req.num_drafts < block && req.logits_offset + req.num_drafts < rows.len(),
            "sparse verify span out of range"
        );
        ensure!(
            (req.draft_row + 1) * block * 16 <= candidate_ids.len()
                && (req.draft_row + 1) * (block - 1) * 16 <= draft_probs.len()
                && (req.draft_row + 1) * (block - 1) <= draft_ids.len(),
            "sparse verify proposal row out of range"
        );
    }

    let mut result = vec![(0, 0); requests.len()];
    for min_p in [false, true] {
        let indices: Vec<_> = requests
            .iter()
            .enumerate()
            .filter(|(_, req)| (rows[req.logits_offset].min_p > 0.0) == min_p)
            .map(|(i, _)| i)
            .collect();
        if indices.is_empty() {
            continue;
        }

        let mut pass_rows = Vec::new();
        let mut mapping = Vec::with_capacity(indices.len() * 3);
        let mut random = Vec::with_capacity(indices.len() * block);
        for &i in &indices {
            let req = requests[i];
            mapping.extend([
                i32::try_from(pass_rows.len())?,
                i32::try_from(req.draft_row)?,
                i32::try_from(req.num_drafts)?,
            ]);
            pass_rows
                .extend_from_slice(&rows[req.logits_offset..=req.logits_offset + req.num_drafts]);
            random.extend_from_slice(&uniforms[i * block..(i + 1) * block]);
        }
        prepare_probabilities(ctx, logits, &pass_rows, min_p, sampling)?;
        ctx.stream.memcpy_htod(&mapping, &mut scratch.requests)?;
        ctx.stream.memcpy_htod(&random, &mut scratch.uniforms)?;

        {
            let (probs, _gp) = sampling.probs.device_ptr_mut(&ctx.stream);
            let (minp, _gm) = sampling.min_p.device_ptr(&ctx.stream);
            let (metadata, _gr) = scratch.requests.device_ptr(&ctx.stream);
            let (ids, _gi) = candidate_ids.device_ptr(&ctx.stream);
            let (q, _gq) = draft_probs.device_ptr(&ctx.stream);
            let (drafts, _gd) = draft_ids.device_ptr(&ctx.stream);
            let (random, _gu) = scratch.uniforms.device_ptr(&ctx.stream);
            let (output, _go) = scratch.output.device_ptr_mut(&ctx.stream);
            let err = unsafe {
                ffi::sparse_speculative_sample_cuda(
                    probs as *mut f32,
                    if min_p {
                        minp as *const f32
                    } else {
                        std::ptr::null()
                    },
                    metadata as *const i32,
                    ids as *const u32,
                    q as *const f32,
                    drafts as *const u32,
                    random as *const f32,
                    output as *mut u32,
                    pass_rows.len() as i32,
                    indices.len() as i32,
                    block as i32,
                    sampling.vocab as i32,
                    active_cu_stream(ctx),
                )
            };
            ensure!(
                err == 0,
                "sparse speculative sampling failed: {err}{}",
                super::ffi_exception_message(err)
            );
        }
        let output = ctx
            .stream
            .clone_dtoh(&scratch.output.slice(..indices.len() * 2))?;
        for (&i, pair) in indices.iter().zip(output.as_chunks::<2>().0) {
            ensure!(
                pair[1] < sampling.vocab as u32,
                "sparse speculative sampling produced an invalid posterior"
            );
            result[i] = (pair[0] as usize, pair[1]);
        }
    }
    Ok(result)
}

fn prepare_probabilities(
    ctx: &DeviceContext,
    logits: HiddenStatesRef<'_>,
    rows: &[BatchSamplingRow],
    min_p: bool,
    scratch: &mut BatchSamplingScratch,
) -> Result<()> {
    ensure!(
        rows.len() <= scratch.max_rows,
        "sparse verify probability capacity exceeded"
    );
    let vocab = i32::try_from(scratch.vocab)?;
    let mut indices = Vec::with_capacity(rows.len());
    let mut temperatures = Vec::with_capacity(rows.len());
    let mut top_k = Vec::with_capacity(rows.len());
    let mut top_p = Vec::with_capacity(rows.len());
    let mut min_ps = Vec::with_capacity(rows.len());
    for row in rows {
        ensure!(
            row.row < logits.seq_len,
            "sparse verify logits row out of range"
        );
        ensure!(
            row.temperature > 0.0 && row.temperature.is_finite(),
            "sparse verify requires a positive temperature"
        );
        indices.push(i32::try_from(row.row)?);
        temperatures.push(row.temperature);
        top_k.push(if row.top_k > 0 {
            row.top_k.min(vocab)
        } else {
            vocab
        });
        top_p.push(row.top_p);
        min_ps.push(row.min_p);
    }
    let has_top_k = top_k.iter().any(|&k| k < vocab);
    let has_top_p = top_p.iter().any(|&p| p < 1.0);
    ctx.stream.memcpy_htod(&indices, &mut scratch.row_indices)?;
    ctx.stream
        .memcpy_htod(&temperatures, &mut scratch.temperature)?;
    ctx.stream.memcpy_htod(&top_k, &mut scratch.top_k)?;
    ctx.stream.memcpy_htod(&top_p, &mut scratch.top_p)?;
    if min_p {
        ctx.stream.memcpy_htod(&min_ps, &mut scratch.min_p)?;
    }
    let workspace_bytes = scratch.softmax_workspace.len();
    let (input, _gl) = logits.data.device_ptr(&ctx.stream);
    let (rows_ptr, _gr) = scratch.row_indices.device_ptr(&ctx.stream);
    let (probs, _gp) = scratch.probs.device_ptr_mut(&ctx.stream);
    let (temperatures, _gt) = scratch.temperature.device_ptr(&ctx.stream);
    let (top_k, _gk) = scratch.top_k.device_ptr(&ctx.stream);
    let (top_p, _gtp) = scratch.top_p.device_ptr(&ctx.stream);
    let (states, _gs) = scratch.topk_row_states.device_ptr_mut(&ctx.stream);
    let (workspace, _gw) = scratch.softmax_workspace.device_ptr_mut(&ctx.stream);
    let err = unsafe {
        ffi::gpu_sample_probs_flashinfer_cuda(
            input as *const ffi::Half,
            rows_ptr as *const i32,
            probs as *mut f32,
            temperatures as *const f32,
            top_k as *const i32,
            top_p as *const f32,
            states as *mut u8,
            workspace as *mut u8,
            workspace_bytes,
            rows.len() as i32,
            vocab,
            i32::from(has_top_k),
            i32::from(has_top_p),
            i32::from(min_p),
            active_cu_stream(ctx),
        )
    };
    ensure!(
        err == 0,
        "sparse verify probability preparation failed: {err}{}",
        super::ffi_exception_message(err)
    );
    Ok(())
}
