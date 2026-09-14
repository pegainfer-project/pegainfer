use anyhow::Result;
use anyhow::ensure;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_frontend::engine::TokenLogprob;
use pegainfer_sample::LogprobRequest;

use super::DecodeStepItem;
use super::PrefillRequestResult;
use super::PrefillStepItem;

pub(super) struct DecodeRows<'a> {
    pub requests: &'a [DecodeStepItem],
    pub logits: &'a HiddenStates,
    pub tokens: &'a [u32],
    pub row_offset: usize,
}

pub(super) fn gather_decode_logprobs(
    ctx: &DeviceContext,
    rows: &DecodeRows<'_>,
) -> Result<Vec<Option<TokenLogprob>>> {
    let requests: Vec<_> = rows
        .requests
        .iter()
        .enumerate()
        .map(|(i, req)| {
            req.logprobs.map(|top_k| LogprobRequest {
                row: rows.row_offset + i,
                picked: rows.tokens[rows.row_offset + i],
                top_k,
            })
        })
        .collect();
    score_rows(ctx, rows.logits, &requests)
}

pub(super) struct PrefillRows<'a> {
    pub requests: &'a [PrefillStepItem],
    pub logits: &'a HiddenStates,
    pub tokens: &'a [u32],
    pub all_position_logits: Option<&'a HiddenStates>,
}

pub(super) fn build_prefill_request_results(
    ctx: &DeviceContext,
    rows: &PrefillRows<'_>,
) -> Result<Vec<PrefillRequestResult>> {
    let wanted: Vec<_> = rows
        .requests
        .iter()
        .enumerate()
        .map(|(i, req)| {
            req.logprobs
                .filter(|_| req.is_final_chunk())
                .map(|top_k| LogprobRequest {
                    row: i,
                    picked: rows.tokens[i],
                    top_k,
                })
        })
        .collect();
    let first = score_rows(ctx, rows.logits, &wanted)?;
    let prompts = match rows.all_position_logits {
        Some(logits) => score_prompts(ctx, rows.requests, logits)?,
        None => vec![None; rows.requests.len()],
    };
    Ok(rows
        .requests
        .iter()
        .zip(first)
        .zip(prompts)
        .enumerate()
        .map(
            |(i, ((req, first_token_logprob), prompt_logprobs))| PrefillRequestResult {
                request_id: req.request_id,
                first_token: rows.tokens[i],
                first_token_logprob,
                prompt_logprobs,
                cached_tokens: req.cached_tokens,
                completed: req.is_final_chunk(),
                prefill_pos: req.chunk_start + req.chunk_tokens,
            },
        )
        .collect())
}

fn score_rows(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    rows: &[Option<LogprobRequest>],
) -> Result<Vec<Option<TokenLogprob>>> {
    let wanted: Vec<_> = rows.iter().flatten().copied().collect();
    let results = pegainfer_sample::token_logprobs_batch(ctx, logits, &wanted)?;
    let mut results = results.into_iter();
    Ok(rows
        .iter()
        .map(|row| row.and_then(|_| results.next()))
        .collect())
}

fn score_prompts(
    ctx: &DeviceContext,
    requests: &[PrefillStepItem],
    all_logits: &HiddenStates,
) -> Result<Vec<Option<Vec<Option<TokenLogprob>>>>> {
    let rows = prompt_rows(requests)?;
    let scores = pegainfer_sample::token_logprobs_batch(ctx, all_logits, &rows)?;
    let mut scores = scores.into_iter();
    Ok(requests
        .iter()
        .map(|req| {
            req.prompt_logprobs.map(|_| {
                std::iter::once(None)
                    .chain(scores.by_ref().take(req.prompt_tokens.len() - 1).map(Some))
                    .collect()
            })
        })
        .collect())
}

fn prompt_rows(requests: &[PrefillStepItem]) -> Result<Vec<LogprobRequest>> {
    let mut rows = Vec::new();
    let mut token_offset = 0;
    for req in requests {
        if let Some(top_k) = req.prompt_logprobs {
            ensure!(
                !req.prompt_tokens.is_empty()
                    && req.chunk_start == 0
                    && req.chunk_tokens == req.prompt_tokens.len()
                    && req.cached_tokens == 0,
                "prompt logprobs require a whole uncached prompt"
            );
            rows.extend(
                req.prompt_tokens
                    .iter()
                    .enumerate()
                    .skip(1)
                    .map(|(j, &picked)| LogprobRequest {
                        row: token_offset + j - 1,
                        picked,
                        top_k,
                    }),
            );
        }
        token_offset += req.chunk_tokens;
    }
    Ok(rows)
}
