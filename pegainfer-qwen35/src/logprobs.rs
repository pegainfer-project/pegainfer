use anyhow::Result;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;

pub(crate) type LogprobSnapshot = (Vec<f32>, usize);

pub(crate) fn snapshot_requested_logprobs(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    requested_top_k: &[Option<usize>],
) -> Result<Vec<Option<LogprobSnapshot>>> {
    anyhow::ensure!(
        requested_top_k.len() <= logits.seq_len,
        "Qwen3.5 logprobs request/logits row mismatch: requested={}, logits_rows={}",
        requested_top_k.len(),
        logits.seq_len
    );
    requested_top_k
        .iter()
        .enumerate()
        .map(|(i, top_k)| {
            top_k
                .map(|top_k| {
                    let row = crate::ops::extract_vec(ctx, logits, i)?;
                    Ok((row.to_host(ctx)?, top_k))
                })
                .transpose()
        })
        .collect()
}
