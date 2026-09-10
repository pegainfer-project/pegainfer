//! Tensor-parallel response contract: the worker reply envelope, bounded
//! response collection, and per-command response validation.

use super::*;

#[derive(Debug)]
pub(super) enum TpWorkerReply {
    Ack,
    DropAck {
        existed: bool,
    },
    Prefill(PrefillResult),
    Decode(DecodeResult),
    Unified(TpUnifiedResult),
    #[cfg(test)]
    Snapshot(WorkerStateSnapshot),
}

#[derive(Debug)]
pub(super) struct TpWorkerResponse {
    pub(super) rank: usize,
    pub(super) result: Result<TpWorkerReply>,
}

const TP_RUNTIME_STEP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

pub(super) fn recv_runtime_responses(
    responses: &mpsc::Receiver<TpWorkerResponse>,
    expected: usize,
    operation: &'static str,
    poison: &TpRuntimePoison,
) -> Result<Vec<TpWorkerResponse>> {
    collect_runtime_responses(expected, operation, poison, || {
        recv_runtime_response(responses, operation, poison)
    })
}

fn collect_runtime_responses(
    expected: usize,
    operation: &'static str,
    poison: &TpRuntimePoison,
    mut recv_next: impl FnMut() -> Result<TpWorkerResponse>,
) -> Result<Vec<TpWorkerResponse>> {
    let mut collected = Vec::with_capacity(expected);
    for _ in 0..expected {
        let response = recv_next()?;
        if let Err(err) = &response.result {
            // A failed rank may leave peers blocked in a collective, so response-set
            // completeness is no longer recoverable or useful.
            let reason = poison.poison(format!(
                "rank {} failed during {operation}: {err:#}",
                response.rank
            ));
            return Err(anyhow::anyhow!(reason));
        }
        collected.push(response);
    }
    Ok(collected)
}

pub(super) fn validate_dispatched_responses<T>(
    result: Result<T>,
    operation: &'static str,
    poison: &TpRuntimePoison,
) -> Result<T> {
    result.map_err(|err| {
        let reason = poison.poison(format!(
            "invalid Qwen3.5 TP {operation} response set: {err:#}"
        ));
        anyhow::anyhow!(reason)
    })
}

fn validate_exact_rank_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
    operation: &'static str,
) -> Result<Vec<(usize, TpWorkerReply)>> {
    anyhow::ensure!(
        responses.len() == world_size,
        "{operation} expected {world_size} responses, got {}",
        responses.len()
    );
    let mut seen_ranks = HashSet::with_capacity(world_size);
    let mut replies = responses
        .into_iter()
        .map(|response| {
            anyhow::ensure!(
                response.rank < world_size,
                "{operation} returned out-of-range rank {} for world size {world_size}",
                response.rank
            );
            anyhow::ensure!(
                seen_ranks.insert(response.rank),
                "{operation} returned duplicate rank {}",
                response.rank
            );
            Ok((response.rank, response.result?))
        })
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(
        (0..world_size).all(|rank| seen_ranks.contains(&rank)),
        "{operation} response set did not contain every rank"
    );
    replies.sort_unstable_by_key(|(rank, _)| *rank);
    Ok(replies)
}

pub(super) fn validate_ack_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
    operation: &'static str,
) -> Result<()> {
    for (rank, reply) in validate_exact_rank_responses(responses, world_size, operation)? {
        anyhow::ensure!(
            matches!(reply, TpWorkerReply::Ack),
            "{operation} rank {rank} returned {} instead of acknowledgement",
            reply_name(&reply)
        );
    }
    Ok(())
}

pub(super) fn validate_drop_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
    expectation: DropExpectation,
) -> Result<()> {
    let expected = expectation == DropExpectation::MustExist;
    let existence = validate_exact_rank_responses(responses, world_size, "drop request")?
        .into_iter()
        .map(|(rank, reply)| match reply {
            TpWorkerReply::DropAck { existed } => Ok((rank, existed)),
            reply => anyhow::bail!(
                "drop request rank {rank} returned {} instead of drop acknowledgement",
                reply_name(&reply)
            ),
        })
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(
        existence.iter().all(|(_, existed)| *existed == expected),
        "drop request expected {expectation:?}, got rank existence {existence:?}"
    );
    Ok(())
}

/// Reduce one exact-rank reply set to the rank-0 primary payload; every
/// non-primary rank must acknowledge. `operation` names the command in error
/// messages and `noun` names the expected payload.
fn validate_primary_responses<R>(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
    operation: &'static str,
    noun: &'static str,
    unwrap_payload: fn(TpWorkerReply) -> Option<R>,
) -> Result<R> {
    let mut primary = None;
    for (rank, reply) in validate_exact_rank_responses(responses, world_size, operation)? {
        match (rank, reply) {
            (0, reply) => {
                let wrong_payload = reply_name(&reply);
                primary = Some(unwrap_payload(reply).ok_or_else(|| {
                    anyhow::anyhow!(
                        "{operation} rank 0 returned {wrong_payload} instead of primary {noun} result",
                    )
                })?);
            }
            (_, reply) => anyhow::ensure!(
                matches!(reply, TpWorkerReply::Ack),
                "{operation} non-primary rank {rank} returned {} instead of acknowledgement",
                reply_name(&reply)
            ),
        }
    }
    primary.ok_or_else(|| anyhow::anyhow!("{operation} returned no primary result"))
}

fn prefill_payload(reply: TpWorkerReply) -> Option<PrefillResult> {
    match reply {
        TpWorkerReply::Prefill(result) => Some(result),
        _ => None,
    }
}

fn decode_payload(reply: TpWorkerReply) -> Option<DecodeResult> {
    match reply {
        TpWorkerReply::Decode(result) => Some(result),
        _ => None,
    }
}

fn unified_payload(reply: TpWorkerReply) -> Option<TpUnifiedResult> {
    match reply {
        TpWorkerReply::Unified(result) => Some(result),
        _ => None,
    }
}

pub(super) fn validate_prefill_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
) -> Result<PrefillResult> {
    validate_primary_responses(responses, world_size, "prefill", "prefill", prefill_payload)
}

pub(super) fn validate_decode_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
) -> Result<DecodeResult> {
    validate_primary_responses(responses, world_size, "decode", "decode", decode_payload)
}

pub(super) fn validate_unified_responses(
    responses: Vec<TpWorkerResponse>,
    world_size: usize,
) -> Result<TpUnifiedResult> {
    validate_primary_responses(
        responses,
        world_size,
        "unified step",
        "unified",
        unified_payload,
    )
}

fn reply_name(reply: &TpWorkerReply) -> &'static str {
    match reply {
        TpWorkerReply::Ack => "acknowledgement",
        TpWorkerReply::DropAck { .. } => "drop acknowledgement",
        TpWorkerReply::Prefill(_) => "prefill result",
        TpWorkerReply::Decode(_) => "decode result",
        TpWorkerReply::Unified(_) => "unified result",
        #[cfg(test)]
        TpWorkerReply::Snapshot(_) => "worker snapshot",
    }
}

#[cfg(test)]
pub(super) fn wait_for_worker_snapshots(
    responses: &mpsc::Receiver<TpWorkerResponse>,
    world_size: usize,
    poison: &TpRuntimePoison,
) -> Result<Vec<WorkerStateSnapshot>> {
    let mut seen_ranks = HashSet::with_capacity(world_size);
    let mut snapshots = Vec::with_capacity(world_size);
    for _ in 0..world_size {
        let response = recv_runtime_response(responses, "snapshot state", poison)?;
        anyhow::ensure!(
            response.rank < world_size,
            "Qwen3.5 TP snapshot returned out-of-range rank {} for world size {world_size}",
            response.rank
        );
        anyhow::ensure!(
            seen_ranks.insert(response.rank),
            "Qwen3.5 TP snapshot returned duplicate rank {}",
            response.rank
        );
        match response.result? {
            TpWorkerReply::Snapshot(snapshot) => {
                anyhow::ensure!(
                    snapshot.rank == response.rank,
                    "Qwen3.5 TP snapshot payload rank {} does not match response rank {}",
                    snapshot.rank,
                    response.rank
                );
                anyhow::ensure!(
                    snapshot.request_count == snapshot.requests.len(),
                    "Qwen3.5 TP rank {} snapshot count {} does not match {} request entries",
                    snapshot.rank,
                    snapshot.request_count,
                    snapshot.requests.len()
                );
                snapshots.push(snapshot);
            }
            TpWorkerReply::Ack => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned acknowledgement")
            }
            TpWorkerReply::DropAck { .. } => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned drop acknowledgement")
            }
            TpWorkerReply::Prefill(_) => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned prefill result")
            }
            TpWorkerReply::Decode(_) => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned decode result")
            }
            TpWorkerReply::Unified(_) => {
                anyhow::bail!("Qwen3.5 TP snapshot unexpectedly returned unified result")
            }
        }
    }
    anyhow::ensure!(
        (0..world_size).all(|rank| seen_ranks.contains(&rank)),
        "Qwen3.5 TP snapshot response set did not contain every rank"
    );
    snapshots.sort_unstable_by_key(|snapshot| snapshot.rank);
    Ok(snapshots)
}

fn recv_runtime_response(
    responses: &mpsc::Receiver<TpWorkerResponse>,
    operation: &'static str,
    poison: &TpRuntimePoison,
) -> Result<TpWorkerResponse> {
    match responses.recv_timeout(TP_RUNTIME_STEP_TIMEOUT) {
        Ok(response) => Ok(response),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            let reason = poison.poison(format!("response channel disconnected during {operation}"));
            Err(anyhow::anyhow!(reason))
        }
        Err(mpsc::RecvTimeoutError::Timeout) => fatal_tp_abort(&format!(
            "Qwen3.5 TP {operation} did not complete within {}s",
            TP_RUNTIME_STEP_TIMEOUT.as_secs()
        )),
    }
}

pub(super) fn fatal_tp_abort(message: &str) -> ! {
    eprintln!("{message}; aborting");
    log::error!("{message}; aborting");
    std::process::abort();
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_response_failure_poisons_executor() {
        let poison = TpRuntimePoison::default();
        let responses = vec![
            reply(0, TpWorkerReply::Ack),
            TpWorkerResponse {
                rank: 1,
                result: Err(anyhow::anyhow!("rank 1 failed")),
            },
        ];

        let err = validate_dispatched_responses(
            validate_ack_responses(responses, 2, "test"),
            "test",
            &poison,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("rank 1 failed"));
        assert!(poison.ensure_healthy().is_err());
    }

    #[test]
    fn runtime_response_collection_fails_fast_when_peer_never_responds() {
        let poison = TpRuntimePoison::default();
        let (tx, rx) = mpsc::channel();
        tx.send(TpWorkerResponse {
            rank: 0,
            result: Err(anyhow::anyhow!("rank 0 failed")),
        })
        .unwrap();
        let _keep_peer_channel_connected = tx;
        let mut receive_attempts = 0;

        let err = collect_runtime_responses(2, "test", &poison, || {
            receive_attempts += 1;
            rx.recv_timeout(std::time::Duration::from_millis(50))
                .map_err(|err| anyhow::anyhow!("waited for nonresponding rank: {err}"))
        })
        .unwrap_err()
        .to_string();

        assert_eq!(receive_attempts, 1, "collector waited for the missing rank");
        assert!(err.contains("rank 0 failed"));
        assert!(!err.contains("waited for nonresponding rank"));
        assert!(poison.ensure_healthy().is_err());
    }

    #[test]
    fn disconnected_runtime_response_poisons_executor() {
        let poison = TpRuntimePoison::default();
        let (tx, rx) = mpsc::channel();
        drop(tx);

        let err = recv_runtime_response(&rx, "test", &poison)
            .unwrap_err()
            .to_string();
        assert!(err.contains("response channel disconnected during test"));
        assert!(poison.ensure_healthy().is_err());
    }

    fn reply(rank: usize, reply: TpWorkerReply) -> TpWorkerResponse {
        TpWorkerResponse {
            rank,
            result: Ok(reply),
        }
    }
}
