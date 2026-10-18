use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;

use super::DeepSeekV2LiteEp2Generator;
use super::backend::EpBackendKind;
use super::backend::EpBackendRuntime;
use super::types::DecodeGraphBlocker;
use super::types::DecodeGraphReadinessMetrics;
use super::types::DecodeGraphReadinessReport;
use super::types::FullDecodeGraphProbeReport;
use super::types::GenerationStats;

#[cfg(test)]
mod tests;

impl DeepSeekV2LiteEp2Generator {
    pub fn decode_graph_readiness_report(
        &mut self,
        stats: &GenerationStats,
        batch_size: usize,
        run_nccl_graph_smoke: bool,
        run_full_decode_graph_probe: bool,
        full_decode_probe_prompt_tokens: Option<&[u32]>,
        full_decode_probe_output_len: usize,
    ) -> Result<DecodeGraphReadinessReport> {
        let backend = self.backend.kind();
        ensure!(
            stats.ep_backend == backend.as_str(),
            "DeepSeek-V2-Lite graph readiness stats backend mismatch: stats={}, runtime={}",
            stats.ep_backend,
            backend.as_str()
        );
        ensure!(
            !run_full_decode_graph_probe || batch_size == 1,
            "DeepSeek-V2-Lite full decode graph probe is scoped to batch_size=1, got {batch_size}"
        );
        let nccl_graph_smoke = if run_nccl_graph_smoke {
            match &self.backend {
                EpBackendRuntime::Nccl(nccl) => {
                    let report = nccl.graph_smoke_all_reduce_f32(&self.rank0.ctx, &self.rank1.ctx);
                    ensure!(
                        report.verified(),
                        "DeepSeek-V2-Lite --nccl-graph-smoke failed: {}",
                        report.failure_summary()
                    );
                    Some(report)
                }
                EpBackendRuntime::HostStaged => bail!(
                    "DeepSeek-V2-Lite --nccl-graph-smoke requires PEGAINFER_DSV2_LITE_EP_BACKEND=nccl"
                ),
            }
        } else {
            None
        };
        let blockers = decode_graph_blockers(backend);
        let full_decode_graph_probe = if blockers.is_empty() {
            self.full_decode_graph_probe_report(
                run_full_decode_graph_probe,
                full_decode_probe_prompt_tokens,
                full_decode_probe_output_len,
            )?
        } else {
            full_decode_graph_probe_report(backend, run_full_decode_graph_probe, &blockers)?
        };
        let full_decode_capture_ready = full_decode_graph_probe.ready();
        let status = decode_graph_readiness_status(
            backend,
            full_decode_capture_ready,
            full_decode_graph_probe.requested,
        );
        Ok(DecodeGraphReadinessReport {
            schema: 2,
            backend: stats.ep_backend.clone(),
            batch_size,
            full_decode_capture_ready,
            status,
            blockers,
            metrics: DecodeGraphReadinessMetrics {
                host_dispatch_calls: stats.host_dispatch_calls,
                host_combine_calls: stats.host_combine_calls,
                host_dispatch_elements: stats.host_dispatch_elements,
                host_combine_elements: stats.host_combine_elements,
                nccl_dense_exchange_calls: stats.nccl_dense_exchange_calls,
                nccl_combine_calls: stats.nccl_combine_calls,
                nccl_dense_exchange_elements: stats.nccl_dense_exchange_elements,
                nccl_combine_elements: stats.nccl_combine_elements,
                nccl_dispatch_local_routes: stats.nccl_dispatch_local_routes,
                nccl_dispatch_remote_routes: stats.nccl_dispatch_remote_routes,
                nccl_combine_routes: stats.nccl_combine_routes,
            },
            nccl_graph_smoke_requested: run_nccl_graph_smoke,
            nccl_graph_smoke,
            full_decode_graph_probe,
            claim_boundary: "This is a graph-readiness diagnostic for the covered DeepSeek-V2-Lite EP2 decode attribution gate. A successful NCCL f32 smoke proves only basic preallocated collective capture/replay on this runtime. Full decode CUDA Graph readiness is claimed only when full_decode_graph_probe captures, instantiates, replays, and verifies the covered shape.",
        })
    }
}

fn decode_graph_readiness_status(
    backend: EpBackendKind,
    full_decode_capture_ready: bool,
    full_decode_graph_probe_requested: bool,
) -> &'static str {
    match backend {
        EpBackendKind::HostStaged => "not_applicable_host_staged_backend",
        EpBackendKind::Nccl if full_decode_capture_ready => "full_decode_capture_ready",
        EpBackendKind::Nccl if !full_decode_graph_probe_requested => {
            "full_decode_probe_not_requested"
        }
        EpBackendKind::Nccl => "blocked_full_decode_path",
    }
}

fn full_decode_graph_probe_report(
    backend: EpBackendKind,
    requested: bool,
    blockers: &[DecodeGraphBlocker],
) -> Result<FullDecodeGraphProbeReport> {
    if requested && backend != EpBackendKind::Nccl {
        bail!(
            "DeepSeek-V2-Lite --full-decode-graph-probe requires PEGAINFER_DSV2_LITE_EP_BACKEND=nccl"
        );
    }
    if !requested {
        return Ok(FullDecodeGraphProbeReport {
            requested: false,
            captured: false,
            instantiated: false,
            replayed: false,
            verified: false,
            replay_count: 0,
            verified_replay_count: 0,
            failure_stage: "not_requested",
            failure_summary: None,
            blockers: Vec::new(),
            capture_mode: "thread_local",
        });
    }
    if !blockers.is_empty() {
        let blocker_ids = blockers
            .iter()
            .map(|blocker| blocker.id)
            .collect::<Vec<_>>()
            .join(", ");
        return Ok(FullDecodeGraphProbeReport {
            requested: true,
            captured: false,
            instantiated: false,
            replayed: false,
            verified: false,
            replay_count: 0,
            verified_replay_count: 0,
            failure_stage: "preflight_blocked",
            failure_summary: Some(format!(
                "full decode graph probe skipped before CUDA stream capture because the current NCCL decode path still has capture blockers: {blocker_ids}"
            )),
            blockers: blockers.to_vec(),
            capture_mode: "thread_local",
        });
    }

    Ok(FullDecodeGraphProbeReport {
        requested: true,
        captured: false,
        instantiated: false,
        replayed: false,
        verified: false,
        replay_count: 0,
        verified_replay_count: 0,
        failure_stage: "probe_not_wired",
        failure_summary: Some(
            "no static blockers were reported, but the full decode graph capture executor is not wired"
                .to_string(),
        ),
        blockers: Vec::new(),
        capture_mode: "thread_local",
    })
}

fn decode_graph_blockers(backend: EpBackendKind) -> Vec<DecodeGraphBlocker> {
    match backend {
        EpBackendKind::HostStaged => vec![
            DecodeGraphBlocker {
                id: "host_staged_route_and_dispatch_on_host",
                source: "runtime/moe.rs::moe_forward_host_staged",
                reason: "routing, per-route expert dispatch, and contribution accumulation are intentionally host-staged",
            },
            DecodeGraphBlocker {
                id: "host_staged_hidden_d2h_and_h2d",
                source: "host_ops.rs::hidden_to_bf16 / hidden_from_f32_host",
                reason: "the baseline path copies hidden states through host memory and synchronizes around those copies",
            },
        ],
        EpBackendKind::Nccl => Vec::new(),
    }
}
