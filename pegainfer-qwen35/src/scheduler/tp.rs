//! Scheduler TP-shard helpers: build TP step items and align/decode artifacts.
//! Split out of scheduler.rs to keep TP bookkeeping out of the God module.

use super::*;

pub(super) fn tp_prefill_items(chunk: &ScheduledChunk) -> Result<Vec<TpPrefillChunkItem>> {
    let ScheduledChunkBackendState::Tp { request_ids } = &chunk.backend_state else {
        anyhow::bail!("TP prefill received single-GPU chunk state");
    };
    anyhow::ensure!(
        chunk.reqs.len() == request_ids.len()
            && chunk.reqs.len() == chunk.windows.len()
            && chunk.reqs.len() == chunk.ends.len(),
        "Qwen3.5 TP scheduled prefill vectors are misaligned"
    );
    Ok(chunk
        .reqs
        .iter()
        .zip(request_ids)
        .zip(&chunk.windows)
        .zip(&chunk.ends)
        .map(|(((req, request_id), window), end)| {
            TpPrefillChunkItem::new_with_sampling(
                *request_id,
                window.clone(),
                req.logprobs,
                req.params,
                *end == req.prompt_tokens.len(),
            )
        })
        .collect())
}

pub(super) fn tp_decode_items(active: &[ActiveRequest35]) -> Result<Vec<TpDecodeStepItem>> {
    active
        .iter()
        .map(|req| {
            let ActiveBackendState::Tp { request_id } = &req.backend_state else {
                anyhow::bail!("TP decode received single-GPU active state");
            };
            Ok(TpDecodeStepItem::new(
                *request_id,
                req.last_token,
                req.logprobs,
                req.params,
            ))
        })
        .collect()
}

pub(super) fn align_prefill_results(
    chunk: &ScheduledChunk,
    result: &PrefillResult,
) -> Result<Vec<Option<PrefillArtifact>>> {
    let ScheduledChunkBackendState::Tp { request_ids } = &chunk.backend_state else {
        anyhow::bail!("align_prefill_results requires TP chunk state");
    };
    anyhow::ensure!(
        request_ids.len() == chunk.reqs.len() && chunk.ends.len() == chunk.reqs.len(),
        "Qwen3.5 TP prefill alignment vectors are misaligned"
    );
    let expected: HashSet<RequestId> = request_ids
        .iter()
        .zip(&chunk.reqs)
        .zip(&chunk.ends)
        .filter_map(|((&request_id, req), &end)| {
            (end == req.prompt_tokens.len()).then_some(request_id)
        })
        .collect();
    let mut by_id = HashMap::with_capacity(result.requests.len());
    for PrefillRequestResult {
        request_id,
        first_token,
        first_token_logprob,
    } in &result.requests
    {
        anyhow::ensure!(
            expected.contains(request_id),
            "Qwen3.5 TP prefill returned unknown or non-final request id {}",
            request_id.get()
        );
        let artifact = PrefillArtifact {
            token: *first_token,
            logprob: first_token_logprob.clone(),
        };
        anyhow::ensure!(
            by_id.insert(*request_id, artifact).is_none(),
            "Qwen3.5 TP prefill returned duplicate request id {}",
            request_id.get()
        );
    }
    anyhow::ensure!(
        by_id.len() == expected.len(),
        "Qwen3.5 TP prefill result is missing final request IDs"
    );

    request_ids
        .iter()
        .zip(&chunk.reqs)
        .zip(&chunk.ends)
        .map(|((&request_id, req), &end)| {
            if end == req.prompt_tokens.len() {
                by_id.remove(&request_id).map(Some).ok_or_else(|| {
                    anyhow::anyhow!(
                        "Qwen3.5 TP prefill result is missing final request id {}",
                        request_id.get()
                    )
                })
            } else {
                Ok(None)
            }
        })
        .collect()
}

pub(super) fn align_decode_results(
    active: &[ActiveRequest35],
    result: &DecodeResult,
) -> Result<Vec<DecodeArtifact>> {
    let expected: Vec<RequestId> = active
        .iter()
        .map(|active_req| {
            let ActiveBackendState::Tp { request_id } = active_req.backend_state else {
                anyhow::bail!("align_decode_results requires TP active state");
            };
            Ok(request_id)
        })
        .collect::<Result<_>>()?;
    let expected_set: HashSet<_> = expected.iter().copied().collect();
    anyhow::ensure!(
        expected_set.len() == expected.len(),
        "Qwen3.5 TP active decode IDs contain duplicates"
    );
    let mut by_id = HashMap::with_capacity(result.requests.len());
    for DecodeRequestResult {
        request_id,
        token,
        logprob,
    } in &result.requests
    {
        anyhow::ensure!(
            expected_set.contains(request_id),
            "Qwen3.5 TP decode returned unknown request id {}",
            request_id.get()
        );
        let artifact = DecodeArtifact {
            token: *token,
            logprob: logprob.clone(),
        };
        anyhow::ensure!(
            by_id.insert(*request_id, artifact).is_none(),
            "Qwen3.5 TP decode returned duplicate request id {}",
            request_id.get()
        );
    }
    expected
        .into_iter()
        .map(|request_id| {
            by_id.remove(&request_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "Qwen3.5 TP decode result is missing request id {}",
                    request_id.get()
                )
            })
        })
        .collect()
}

pub(super) fn split_decode_artifacts(
    artifacts: &[DecodeArtifact],
) -> (Vec<u32>, Vec<Option<TokenLogprob>>) {
    artifacts
        .iter()
        .map(|artifact| (artifact.token, artifact.logprob.clone()))
        .unzip()
}
