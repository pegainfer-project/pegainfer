//! DSpine bridge kernels: read normalization, code packing, feature writes and
//! the greedy serial last-write refinement.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys::CUresult;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceMatrix;
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;
use crate::tensor::active_cu_stream;

fn check(status: CUresult, what: &str) -> Result<()> {
    ensure!(
        status == CUresult::CUDA_SUCCESS,
        "{what} failed: {status:?}"
    );
    Ok(())
}

/// `raw` -> `normalized = raw / (rms + eps)` plus the per-row fp32 `rms`.
pub fn prepare_read_into(
    ctx: &DeviceContext,
    raw: &HiddenStates,
    normalized: &mut HiddenStates,
    rms: &mut CudaSlice<f32>,
    eps: f32,
) -> Result<()> {
    let rows = raw.seq_len;
    let hidden = raw.hidden_dim;
    ensure!(
        rows > 0 && hidden > 0,
        "DSpine prepare_read requires a non-empty batch"
    );
    ensure!(
        normalized.seq_len == rows && normalized.hidden_dim == hidden,
        "DSpine prepare_read output shape mismatch"
    );
    ensure!(
        rms.len() >= rows,
        "DSpine rms buffer holds {} < {rows} rows",
        rms.len()
    );
    raw.checked_extent("DSpine raw")?;
    normalized.checked_extent("DSpine normalized")?;
    let (raw_ptr, _gr) = raw.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = normalized.data.device_ptr_mut(&ctx.stream);
    let (rms_ptr, _gm) = rms.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::dspine_prepare_read_cuda(
            raw_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            rms_ptr as *mut f32,
            rows as i32,
            hidden as i32,
            eps,
            active_cu_stream(ctx),
        )
    };
    check(status, "dspine_prepare_read")
}

/// Replace the normalized read with `sqrt(C) * codebook[anchor]` on block rows.
pub fn normalize_codes_into(
    ctx: &DeviceContext,
    projected: &HiddenStates,
    codebook: &DeviceMatrix,
    input_ids: &CudaSlice<u32>,
    codes: &mut HiddenStates,
    block_size: usize,
    eps: f32,
) -> Result<()> {
    let rows = projected.seq_len;
    let code_size = projected.hidden_dim;
    ensure!(
        rows > 0 && code_size > 0 && block_size > 0,
        "DSpine normalize_codes requires non-empty dimensions"
    );
    ensure!(
        rows.is_multiple_of(block_size),
        "DSpine normalize_codes rows {rows} are not whole blocks of {block_size}"
    );
    ensure!(
        projected.data.len() >= rows * code_size,
        "DSpine projected backing too small"
    );
    ensure!(
        codes.seq_len == rows && codes.hidden_dim == code_size,
        "DSpine codes shape mismatch"
    );
    ensure!(
        codebook.cols == code_size,
        "DSpine codebook width {} != code size {code_size}",
        codebook.cols
    );
    ensure!(
        input_ids.len() >= rows,
        "DSpine token ids hold {} < {rows} rows",
        input_ids.len()
    );
    let (proj_ptr, _gp) = projected.data.device_ptr(&ctx.stream);
    let (book_ptr, _gb) = codebook.data.device_ptr(&ctx.stream);
    let (ids_ptr, _gi) = input_ids.device_ptr(&ctx.stream);
    let (codes_ptr, _gc) = codes.data.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::dspine_normalize_codes_cuda(
            proj_ptr as *const ffi::Half,
            book_ptr as *const ffi::Half,
            ids_ptr as *const u32,
            codes_ptr as *mut ffi::Half,
            rows as i32,
            code_size as i32,
            block_size as i32,
            eps,
            active_cu_stream(ctx),
        )
    };
    check(status, "dspine_normalize_codes")
}

/// `message = sigmoid(Q[r] + S[r-1] + bias) * V[r-1]`, zero on block anchors.
pub fn feature_message_into(
    ctx: &DeviceContext,
    packed: &HiddenStates,
    gate_bias: &DeviceVec,
    message: &mut HiddenStates,
    block_size: usize,
) -> Result<()> {
    let rows = packed.seq_len;
    let rank = message.hidden_dim;
    ensure!(
        rows > 0 && rank > 0 && block_size > 0,
        "DSpine feature_message requires non-empty dimensions"
    );
    ensure!(
        packed.hidden_dim == 3 * rank,
        "DSpine packed width {} != 3R {rank}",
        packed.hidden_dim
    );
    ensure!(
        gate_bias.len == rank,
        "DSpine gate bias length {} != R {rank}",
        gate_bias.len
    );
    ensure!(
        message.seq_len == rows,
        "DSpine message rows {} != packed rows {rows}",
        message.seq_len
    );
    packed.checked_extent("DSpine packed")?;
    message.checked_extent("DSpine message")?;
    let (packed_ptr, _gp) = packed.data.device_ptr(&ctx.stream);
    let (bias_ptr, _gb) = gate_bias.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = message.data.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::dspine_feature_message_cuda(
            packed_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            rows as i32,
            rank as i32,
            block_size as i32,
            active_cu_stream(ctx),
        )
    };
    check(status, "dspine_feature_message")
}

/// `out = raw + rms * delta` on non-anchor rows, `raw` on anchors.
pub fn feature_finish_into(
    ctx: &DeviceContext,
    raw: &HiddenStates,
    delta: &HiddenStates,
    rms: &CudaSlice<f32>,
    out: &mut HiddenStates,
    block_size: usize,
) -> Result<()> {
    let rows = raw.seq_len;
    let hidden = raw.hidden_dim;
    ensure!(
        rows > 0 && hidden > 0 && block_size > 0,
        "DSpine feature_finish requires non-empty dimensions"
    );
    ensure!(
        delta.seq_len == rows && delta.hidden_dim == hidden,
        "DSpine delta shape mismatch"
    );
    ensure!(
        out.seq_len == rows && out.hidden_dim == hidden,
        "DSpine finish output shape mismatch"
    );
    ensure!(
        rms.len() >= rows,
        "DSpine rms buffer holds {} < {rows} rows",
        rms.len()
    );
    raw.checked_extent("DSpine raw")?;
    delta.checked_extent("DSpine delta")?;
    out.checked_extent("DSpine finish output")?;
    let (raw_ptr, _gr) = raw.data.device_ptr(&ctx.stream);
    let (delta_ptr, _gd) = delta.data.device_ptr(&ctx.stream);
    let (rms_ptr, _gm) = rms.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.data.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::dspine_feature_finish_cuda(
            raw_ptr as *const ffi::Half,
            delta_ptr as *const ffi::Half,
            rms_ptr as *const f32,
            out_ptr as *mut ffi::Half,
            rows as i32,
            hidden as i32,
            block_size as i32,
            active_cu_stream(ctx),
        )
    };
    check(status, "dspine_feature_finish")
}

/// Top-`k` token ids for every proposal row (block rows 1..block), ascending.
pub fn topk_into(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    batch: usize,
    block_size: usize,
    k: usize,
    candidate_ids: &mut CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        batch > 0 && block_size >= 2 && (1..=64).contains(&k) && k.is_power_of_two(),
        "DSpine topk invalid batch/block/k: {batch}/{block_size}/{k}"
    );
    ensure!(
        logits.seq_len == batch * block_size,
        "DSpine topk logits rows {} != batch*block {}",
        logits.seq_len,
        batch * block_size
    );
    let proposals = batch * (block_size - 1);
    ensure!(
        candidate_ids.len() >= proposals * k,
        "DSpine candidate buffer holds {} < {}",
        candidate_ids.len(),
        proposals * k
    );
    logits.checked_extent("DSpine logits")?;
    let (logits_ptr, _gl) = logits.data.device_ptr(&ctx.stream);
    let (ids_ptr, _gi) = candidate_ids.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::dspine_topk_cuda(
            logits_ptr as *const ffi::Half,
            ids_ptr as *mut i32,
            batch as i32,
            block_size as i32,
            logits.hidden_dim as i32,
            k as i32,
            active_cu_stream(ctx),
        )
    };
    check(status, "dspine_topk")
}

/// Build the `[proposal, lane] -> receiver lane` table for the serial chain.
#[allow(clippy::too_many_arguments)]
pub fn serial_table_into(
    ctx: &DeviceContext,
    packed: &HiddenStates,
    rms: &CudaSlice<f32>,
    raw: &HiddenStates,
    lm_head: &DeviceMatrix,
    norm_weight: &DeviceVec,
    tg: &DeviceMatrix,
    tsv: &DeviceMatrix,
    gate_bias: &DeviceVec,
    candidates: &CudaSlice<i32>,
    input_ids: &CudaSlice<u32>,
    idx: &mut CudaSlice<i32>,
    batch: usize,
    block_size: usize,
    k: usize,
) -> Result<()> {
    let rows = batch * block_size;
    let proposals = batch * (block_size - 1);
    let hidden = raw.hidden_dim;
    let rank = gate_bias.len;
    ensure!(
        batch > 0 && block_size >= 2 && hidden > 0 && rank > 0 && k >= 1,
        "DSpine serial_table invalid dimensions"
    );
    ensure!(
        packed.seq_len == rows && raw.seq_len == rows && rms.len() >= rows,
        "DSpine serial_table row mismatch"
    );
    ensure!(
        packed.hidden_dim == 3 * rank,
        "DSpine packed width {} != 3R {rank}",
        packed.hidden_dim
    );
    ensure!(
        lm_head.cols == hidden && norm_weight.len == hidden,
        "DSpine serial_table head/norm width mismatch"
    );
    ensure!(
        tg.cols == rank && tsv.cols == 2 * rank,
        "DSpine table widths mismatch: TG={} TSV={}",
        tg.cols,
        tsv.cols
    );
    ensure!(
        candidates.len() >= proposals * k && idx.len() >= proposals * k,
        "DSpine candidate/idx buffers too small"
    );
    ensure!(
        input_ids.len() >= rows,
        "DSpine token ids hold {} < {rows} rows",
        input_ids.len()
    );
    packed.checked_extent("DSpine packed")?;
    raw.checked_extent("DSpine raw")?;
    let (packed_ptr, _gp) = packed.data.device_ptr(&ctx.stream);
    let (rms_ptr, _gm) = rms.device_ptr(&ctx.stream);
    let (raw_ptr, _gr) = raw.data.device_ptr(&ctx.stream);
    let (head_ptr, _gh) = lm_head.data.device_ptr(&ctx.stream);
    let (norm_ptr, _gn) = norm_weight.data.device_ptr(&ctx.stream);
    let (tg_ptr, _gt) = tg.data.device_ptr(&ctx.stream);
    let (tsv_ptr, _gv) = tsv.data.device_ptr(&ctx.stream);
    let (bias_ptr, _gb) = gate_bias.data.device_ptr(&ctx.stream);
    let (cand_ptr, _gc) = candidates.device_ptr(&ctx.stream);
    let (ids_ptr, _gi) = input_ids.device_ptr(&ctx.stream);
    let (idx_ptr, _gx) = idx.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::dspine_serial_table_cuda(
            packed_ptr as *const ffi::Half,
            rms_ptr as *const f32,
            raw_ptr as *const ffi::Half,
            head_ptr as *const ffi::Half,
            norm_ptr as *const ffi::Half,
            tg_ptr as *const ffi::Half,
            tsv_ptr as *const ffi::Half,
            bias_ptr as *const ffi::Half,
            cand_ptr as *const i32,
            ids_ptr as *const u32,
            idx_ptr as *mut i32,
            batch as i32,
            block_size as i32,
            hidden as i32,
            rank as i32,
            k as i32,
            active_cu_stream(ctx),
        )
    };
    check(status, "dspine_serial_table")
}

/// Walk the lane table and emit `[anchor, drafts...]` per request.
pub fn serial_chain_into(
    ctx: &DeviceContext,
    idx: &CudaSlice<i32>,
    candidates: &CudaSlice<i32>,
    input_ids: &CudaSlice<u32>,
    out: &mut CudaSlice<u32>,
    batch: usize,
    block_size: usize,
    k: usize,
) -> Result<()> {
    let rows = batch * block_size;
    let proposals = batch * (block_size - 1);
    ensure!(
        batch > 0 && block_size >= 2 && k >= 1,
        "DSpine serial_chain invalid dimensions"
    );
    ensure!(
        idx.len() >= proposals * k && candidates.len() >= proposals * k,
        "DSpine chain tables too small"
    );
    ensure!(
        input_ids.len() >= rows && out.len() >= rows,
        "DSpine chain token buffers too small"
    );
    let (idx_ptr, _gi) = idx.device_ptr(&ctx.stream);
    let (cand_ptr, _gc) = candidates.device_ptr(&ctx.stream);
    let (ids_ptr, _gn) = input_ids.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::dspine_serial_chain_cuda(
            idx_ptr as *const i32,
            cand_ptr as *const i32,
            ids_ptr as *const u32,
            out_ptr as *mut u32,
            batch as i32,
            block_size as i32,
            k as i32,
            active_cu_stream(ctx),
        )
    };
    check(status, "dspine_serial_chain")
}

/// `out = bf16(fp32(input) * fp32(weight[col]))` for the TG table.
pub fn scale_cols_into(
    ctx: &DeviceContext,
    input: &DeviceMatrix,
    weight: &DeviceVec,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(
        input.cols == weight.len,
        "DSpine scale_cols width {} != weight {}",
        input.cols,
        weight.len
    );
    let count = input.rows * input.cols;
    ensure!(
        out.len() >= count,
        "DSpine scale_cols output holds {} < {count}",
        out.len()
    );
    let (in_ptr, _gi) = input.data.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = weight.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::dspine_scale_cols_cuda(
            in_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            input.rows as i64,
            input.cols as i32,
            active_cu_stream(ctx),
        )
    };
    check(status, "dspine_scale_cols")
}

/// `out = bf16(fp32(input) * factor)` for the TSV table.
pub fn scale_const_into(
    ctx: &DeviceContext,
    input: &DeviceMatrix,
    out: &mut CudaSlice<bf16>,
    factor: f32,
) -> Result<()> {
    let count = input.rows * input.cols;
    ensure!(
        out.len() >= count,
        "DSpine scale_const output holds {} < {count}",
        out.len()
    );
    let (in_ptr, _gi) = input.data.device_ptr(&ctx.stream);
    let (out_ptr, _go) = out.device_ptr_mut(&ctx.stream);
    let status = unsafe {
        ffi::dspine_scale_const_cuda(
            in_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            count as i64,
            factor,
            active_cu_stream(ctx),
        )
    };
    check(status, "dspine_scale_const")
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;

    fn hidden(ctx: &DeviceContext, data: &[bf16], dim: usize, seq: usize) -> HiddenStates {
        HiddenStates {
            data: ctx.stream.clone_htod(data).expect("H2D"),
            hidden_dim: dim,
            seq_len: seq,
        }
    }

    fn matrix(ctx: &DeviceContext, data: &[bf16], rows: usize, cols: usize) -> DeviceMatrix {
        DeviceMatrix {
            data: ctx.stream.clone_htod(data).expect("H2D"),
            rows,
            cols,
        }
    }

    fn vector(ctx: &DeviceContext, data: &[bf16]) -> DeviceVec {
        DeviceVec {
            data: ctx.stream.clone_htod(data).expect("H2D"),
            len: data.len(),
        }
    }

    fn host(ctx: &DeviceContext, tensor: &HiddenStates) -> Vec<f32> {
        let values: Vec<bf16> = ctx.stream.clone_dtoh(&tensor.data).expect("D2H");
        ctx.sync().expect("sync");
        values.iter().map(|v| v.to_f32()).collect()
    }

    fn sigmoid(x: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn read_and_finish_match_the_reference() {
        let ctx = DeviceContext::new().expect("GPU required");
        let rows = 4usize;
        let dim = 8usize;
        let block = 2usize;
        let raw: Vec<bf16> = (0..rows * dim)
            .map(|i| bf16::from_f32((i as f32 * 0.037).sin() * 1.5))
            .collect();
        let delta: Vec<bf16> = (0..rows * dim)
            .map(|i| bf16::from_f32((i as f32 * 0.011).cos()))
            .collect();
        let raw_d = hidden(&ctx, &raw, dim, rows);
        let delta_d = hidden(&ctx, &delta, dim, rows);
        let mut normalized = HiddenStates::zeros(&ctx, dim, rows).unwrap();
        let mut rms = ctx.stream.alloc_zeros(rows).unwrap();
        prepare_read_into(&ctx, &raw_d, &mut normalized, &mut rms, 1e-6).expect("prepare");
        let rms_host: Vec<f32> = ctx.stream.clone_dtoh(&rms).unwrap();
        let norm_host = host(&ctx, &normalized);
        let mut expected_rms = Vec::with_capacity(rows);
        for row in 0..rows {
            let mean = raw[row * dim..(row + 1) * dim]
                .iter()
                .map(|v| {
                    let x = v.to_f32();
                    x * x
                })
                .sum::<f32>()
                / dim as f32;
            let r = mean.max(1.1754943508222875e-38).sqrt();
            expected_rms.push(r);
            for col in 0..dim {
                let want = raw[row * dim + col].to_f32() / (r + 1e-6);
                let want = bf16::from_f32(want).to_f32();
                assert!((norm_host[row * dim + col] - want).abs() < 1e-3);
            }
        }
        for (got, want) in rms_host.iter().zip(&expected_rms) {
            assert!((got - want).abs() < 1e-4, "rms {got} vs {want}");
        }

        let mut out = HiddenStates::zeros(&ctx, dim, rows).unwrap();
        feature_finish_into(&ctx, &raw_d, &delta_d, &rms, &mut out, block).expect("finish");
        let out_host = host(&ctx, &out);
        for row in 0..rows {
            for col in 0..dim {
                let index = row * dim + col;
                let want = if row % block == 0 {
                    raw[index].to_f32()
                } else {
                    raw[index].to_f32() + expected_rms[row] * delta[index].to_f32()
                };
                let want = bf16::from_f32(want).to_f32();
                assert!((out_host[index] - want).abs() < 2e-3);
            }
        }
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn codes_and_messages_match_the_reference() {
        let ctx = DeviceContext::new().expect("GPU required");
        let rows = 4usize;
        let block = 2usize;
        let code = 4usize;
        let rank = 3usize;
        let projected: Vec<bf16> = (0..rows * code)
            .map(|i| bf16::from_f32((i as f32 * 0.13).sin() + 0.1))
            .collect();
        let codebook: Vec<bf16> = (0..2 * code)
            .map(|i| bf16::from_f32((i as f32 * 0.07).cos()))
            .collect();
        let ids = vec![1u32, 0, 1, 0];
        let projected_d = hidden(&ctx, &projected, code, rows);
        let codebook_d = matrix(&ctx, &codebook, 2, code);
        let ids_d = ctx.stream.clone_htod(&ids).unwrap();
        let mut codes = HiddenStates::zeros(&ctx, code, rows).unwrap();
        normalize_codes_into(
            &ctx,
            &projected_d,
            &codebook_d,
            &ids_d,
            &mut codes,
            block,
            1e-6,
        )
        .expect("codes");
        let codes_host = host(&ctx, &codes);
        let scale = (code as f32).sqrt();
        for row in 0..rows {
            if row % block == 0 {
                let anchor = ids[row] as usize;
                for col in 0..code {
                    let want =
                        bf16::from_f32(scale * codebook[anchor * code + col].to_f32()).to_f32();
                    assert!((codes_host[row * code + col] - want).abs() < 1e-3);
                }
            } else {
                let norm = projected[row * code..(row + 1) * code]
                    .iter()
                    .map(|v| {
                        let x = v.to_f32();
                        x * x
                    })
                    .sum::<f32>()
                    .sqrt();
                for col in 0..code {
                    let want = bf16::from_f32(
                        scale * projected[row * code + col].to_f32() / (norm + 1e-6),
                    )
                    .to_f32();
                    assert!((codes_host[row * code + col] - want).abs() < 1e-3);
                }
            }
        }

        let packed: Vec<bf16> = (0..rows * 3 * rank)
            .map(|i| bf16::from_f32((i as f32 * 0.05).sin() * 0.7))
            .collect();
        let bias: Vec<bf16> = (0..rank).map(|i| bf16::from_f32(i as f32 * 0.01)).collect();
        let packed_d = hidden(&ctx, &packed, 3 * rank, rows);
        let bias_d = vector(&ctx, &bias);
        let mut message = HiddenStates::zeros(&ctx, rank, rows).unwrap();
        feature_message_into(&ctx, &packed_d, &bias_d, &mut message, block).expect("message");
        let message_host = host(&ctx, &message);
        for row in 0..rows {
            let pred = if row % block == 0 { row } else { row - 1 };
            for d in 0..rank {
                let want = if row % block == 0 {
                    0.0
                } else {
                    let q = packed[row * 3 * rank + d].to_f32();
                    let s = packed[pred * 3 * rank + rank + d].to_f32();
                    let v = packed[pred * 3 * rank + 2 * rank + d].to_f32();
                    bf16::from_f32(sigmoid(q + s + bias[d].to_f32()) * v).to_f32()
                };
                assert!((message_host[row * rank + d] - want).abs() < 1e-3);
            }
        }
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn topk_and_serial_chain_match_the_reference() {
        let ctx = DeviceContext::new().expect("GPU required");
        let batch = 2usize;
        let block = 3usize;
        let hidden_dim = 8usize;
        let rank = 4usize;
        let k = 2usize;
        let vocab = 128usize;
        let rows = batch * block;
        let proposals = batch * (block - 1);

        let logits: Vec<bf16> = (0..rows * vocab)
            .map(|i| bf16::from_f32((i % vocab) as f32 * 0.01 + (i / vocab) as f32 * 0.001))
            .collect();
        let logits_d = hidden(&ctx, &logits, vocab, rows);
        let mut candidates = ctx.stream.alloc_zeros(proposals * k).unwrap();
        topk_into(&ctx, &logits_d, batch, block, k, &mut candidates).expect("topk");
        let candidates_host: Vec<i32> = ctx.stream.clone_dtoh(&candidates).unwrap();
        for n in 0..proposals {
            let b = n / (block - 1);
            let i = n % (block - 1);
            let row = b * block + 1 + i;
            let mut ids: Vec<i32> = (0..vocab as i32).collect();
            ids.sort_by_key(|&id| -((logits[row * vocab + id as usize].to_f32() * 1e6) as i64));
            let mut want = ids[..k].to_vec();
            want.sort_unstable();
            let mut got = candidates_host[n * k..(n + 1) * k].to_vec();
            got.sort_unstable();
            assert_eq!(got, want, "row {row}");
        }

        let raw: Vec<bf16> = (0..rows * hidden_dim)
            .map(|i| bf16::from_f32((i as f32 * 0.021).sin()))
            .collect();
        let packed: Vec<bf16> = (0..rows * 3 * rank)
            .map(|i| bf16::from_f32((i as f32 * 0.033).cos()))
            .collect();
        let rms: Vec<f32> = (0..rows).map(|i| 0.5 + i as f32 * 0.1).collect();
        let lm_head: Vec<bf16> = (0..vocab * hidden_dim)
            .map(|i| bf16::from_f32((i as f32 * 0.017).sin()))
            .collect();
        let norm: Vec<bf16> = (0..hidden_dim)
            .map(|i| bf16::from_f32(1.0 + i as f32 * 0.01))
            .collect();
        let tg: Vec<bf16> = (0..vocab * rank)
            .map(|i| bf16::from_f32((i as f32 * 0.019).cos()))
            .collect();
        let tsv: Vec<bf16> = (0..vocab * 2 * rank)
            .map(|i| bf16::from_f32((i as f32 * 0.023).sin()))
            .collect();
        let bias: Vec<bf16> = (0..rank).map(|i| bf16::from_f32(i as f32 * 0.02)).collect();
        let ids = vec![3u32, 0, 0, 5, 0, 0];
        let packed_d = hidden(&ctx, &packed, 3 * rank, rows);
        let raw_d = hidden(&ctx, &raw, hidden_dim, rows);
        let lm_d = matrix(&ctx, &lm_head, vocab, hidden_dim);
        let norm_d = vector(&ctx, &norm);
        let tg_d = matrix(&ctx, &tg, vocab, rank);
        let tsv_d = matrix(&ctx, &tsv, vocab, 2 * rank);
        let bias_d = vector(&ctx, &bias);
        let ids_d = ctx.stream.clone_htod(&ids).unwrap();
        let rms_d = ctx.stream.clone_htod(&rms).unwrap();
        let mut idx = ctx.stream.alloc_zeros(proposals * k).unwrap();
        serial_table_into(
            &ctx,
            &packed_d,
            &rms_d,
            &raw_d,
            &lm_d,
            &norm_d,
            &tg_d,
            &tsv_d,
            &bias_d,
            &candidates,
            &ids_d,
            &mut idx,
            batch,
            block,
            k,
        )
        .expect("serial table");
        let idx_host: Vec<i32> = ctx.stream.clone_dtoh(&idx).unwrap();

        let mut expected = vec![0i32; proposals * k];
        for n in 0..proposals {
            let b = n / (block - 1);
            let i = n % (block - 1);
            let row = b * block + 1 + i;
            let base: Vec<f32> = (0..k)
                .map(|j| {
                    let token = candidates_host[n * k + j] as usize;
                    (0..hidden_dim)
                        .map(|h| {
                            raw[row * hidden_dim + h].to_f32()
                                * norm[h].to_f32()
                                * lm_head[token * hidden_dim + h].to_f32()
                        })
                        .sum()
                })
                .collect();
            for lane in 0..k {
                let sender = if i == 0 {
                    ids[b * block] as usize
                } else {
                    candidates_host[(n - 1) * k + lane] as usize
                };
                let mut best = f32::NEG_INFINITY;
                let mut best_lane = 0usize;
                for j in 0..k {
                    let token = candidates_host[n * k + j] as usize;
                    let mut dot = 0.0f32;
                    for d in 0..rank {
                        let q = packed[row * 3 * rank + d].to_f32();
                        let s = tsv[sender * 2 * rank + d].to_f32();
                        let v = tsv[sender * 2 * rank + rank + d].to_f32();
                        let m = sigmoid(q + s + bias[d].to_f32()) * v;
                        dot += m * tg[token * rank + d].to_f32();
                    }
                    let score = base[j] + rms[row] * dot;
                    if score > best || (score == best && j < best_lane) {
                        best = score;
                        best_lane = j;
                    }
                }
                expected[n * k + lane] = best_lane as i32;
            }
        }
        assert_eq!(idx_host, expected, "serial table");

        let mut tokens = ctx.stream.alloc_zeros(rows).unwrap();
        serial_chain_into(
            &ctx,
            &idx,
            &candidates,
            &ids_d,
            &mut tokens,
            batch,
            block,
            k,
        )
        .expect("serial chain");
        let tokens_host: Vec<u32> = ctx.stream.clone_dtoh(&tokens).unwrap();
        let mut expected_tokens = vec![0u32; rows];
        for b in 0..batch {
            let n0 = b * (block - 1);
            let mut lane = idx_host[n0 * k] as usize;
            expected_tokens[b * block] = ids[b * block];
            for i in 0..block - 1 {
                expected_tokens[b * block + 1 + i] = candidates_host[(n0 + i) * k + lane] as u32;
                if i + 1 < block - 1 {
                    lane = idx_host[(n0 + i + 1) * k + lane] as usize;
                }
            }
        }
        assert_eq!(tokens_host, expected_tokens);
    }

    #[test]
    #[ignore = "requires a GPU"]
    fn scale_kernels_match_the_reference() {
        let ctx = DeviceContext::new().expect("GPU required");
        let rows = 3usize;
        let cols = 4usize;
        let input: Vec<bf16> = (0..rows * cols)
            .map(|i| bf16::from_f32(i as f32 * 0.25 - 0.5))
            .collect();
        let weight: Vec<bf16> = (0..cols)
            .map(|i| bf16::from_f32(0.5 + i as f32 * 0.25))
            .collect();
        let input_d = matrix(&ctx, &input, rows, cols);
        let weight_d = vector(&ctx, &weight);
        let mut out = ctx.stream.alloc_zeros(rows * cols).unwrap();
        scale_cols_into(&ctx, &input_d, &weight_d, &mut out).expect("scale cols");
        let out_host: Vec<bf16> = ctx.stream.clone_dtoh(&out).unwrap();
        for row in 0..rows {
            for col in 0..cols {
                let want = bf16::from_f32(input[row * cols + col].to_f32() * weight[col].to_f32());
                assert_eq!(out_host[row * cols + col].to_bits(), want.to_bits());
            }
        }

        let mut out = ctx.stream.alloc_zeros(rows * cols).unwrap();
        scale_const_into(&ctx, &input_d, &mut out, 3.0).expect("scale const");
        let out_host: Vec<bf16> = ctx.stream.clone_dtoh(&out).unwrap();
        for (got, value) in out_host.iter().zip(&input) {
            assert_eq!(
                got.to_bits(),
                bf16::from_f32(value.to_f32() * 3.0).to_bits()
            );
        }
    }
}
