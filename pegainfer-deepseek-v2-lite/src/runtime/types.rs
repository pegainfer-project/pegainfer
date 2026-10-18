use pegainfer_frontend::engine::FinishReason;
use serde::Serialize;

use super::backend::EpBackendKind;
use super::backend::NCCL_BACKEND;
use crate::nccl_backend::NcclGraphSmokeReport;

#[derive(Clone, Debug, Default)]
pub struct GenerationStats {
    pub device_ordinals: Vec<usize>,
    pub ep_backend: String,
    pub ep_size: usize,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub host_dispatch_calls: usize,
    pub host_dispatch_elements: usize,
    pub host_combine_calls: usize,
    pub host_combine_elements: usize,
    pub host_dispatch_local_routes: usize,
    pub host_dispatch_remote_routes: usize,
    pub nccl_dispatch_local_routes: usize,
    pub nccl_dispatch_remote_routes: usize,
    pub nccl_combine_routes: usize,
    pub nccl_dense_exchange_calls: usize,
    pub nccl_combine_calls: usize,
    pub nccl_dense_exchange_elements: usize,
    pub nccl_combine_elements: usize,
    pub output_token_sha256: String,
}

#[derive(Clone, Debug)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub finish_reason: FinishReason,
    pub stats: GenerationStats,
}

#[derive(Clone, Debug)]
pub struct BatchedGenerationResult {
    pub tokens: Vec<Vec<u32>>,
    pub prefill_next_token_us: Vec<u64>,
    pub per_token_decode_us: Vec<u64>,
    pub total_generation_us: u64,
    pub stats: GenerationStats,
}

#[derive(Clone, Debug, Serialize)]
pub struct DecodeGraphReadinessReport {
    pub(super) schema: u32,
    pub(super) backend: String,
    pub(super) batch_size: usize,
    pub(super) full_decode_capture_ready: bool,
    pub(super) status: &'static str,
    pub(super) blockers: Vec<DecodeGraphBlocker>,
    pub(super) metrics: DecodeGraphReadinessMetrics,
    pub(super) nccl_graph_smoke_requested: bool,
    pub(super) nccl_graph_smoke: Option<NcclGraphSmokeReport>,
    pub(super) full_decode_graph_probe: FullDecodeGraphProbeReport,
    pub(super) claim_boundary: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct DecodeGraphBlocker {
    pub(super) id: &'static str,
    pub(super) source: &'static str,
    pub(super) reason: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct FullDecodeGraphProbeReport {
    pub(super) requested: bool,
    pub(super) captured: bool,
    pub(super) instantiated: bool,
    pub(super) replayed: bool,
    pub(super) verified: bool,
    pub(super) replay_count: usize,
    pub(super) verified_replay_count: usize,
    pub(super) failure_stage: &'static str,
    pub(super) failure_summary: Option<String>,
    pub(super) blockers: Vec<DecodeGraphBlocker>,
    pub(super) capture_mode: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct DecodeGraphReadinessMetrics {
    pub(super) host_dispatch_calls: usize,
    pub(super) host_combine_calls: usize,
    pub(super) host_dispatch_elements: usize,
    pub(super) host_combine_elements: usize,
    pub(super) nccl_dense_exchange_calls: usize,
    pub(super) nccl_combine_calls: usize,
    pub(super) nccl_dense_exchange_elements: usize,
    pub(super) nccl_combine_elements: usize,
    pub(super) nccl_dispatch_local_routes: usize,
    pub(super) nccl_dispatch_remote_routes: usize,
    pub(super) nccl_combine_routes: usize,
}

impl DecodeGraphReadinessReport {
    pub fn full_decode_capture_ready(&self) -> bool {
        self.full_decode_capture_ready
    }

    pub fn blocker_count(&self) -> usize {
        if self.full_decode_graph_probe.blockers.is_empty() {
            self.blockers.len()
        } else {
            self.full_decode_graph_probe.blockers.len()
        }
    }

    pub fn nccl_graph_smoke_status(&self) -> &'static str {
        if self.backend != NCCL_BACKEND {
            return "not_applicable";
        }
        if !self.nccl_graph_smoke_requested {
            return "not_run";
        }
        self.nccl_graph_smoke
            .as_ref()
            .map_or("not_run", NcclGraphSmokeReport::coverage_status)
    }

    pub fn full_decode_graph_probe_status(&self) -> &'static str {
        self.full_decode_graph_probe.coverage_status()
    }
}

impl FullDecodeGraphProbeReport {
    pub(super) fn ready(&self) -> bool {
        self.captured
            && self.instantiated
            && self.replayed
            && self.verified
            && self.replay_count > 0
            && self.verified_replay_count == self.replay_count
    }

    pub(super) fn coverage_status(&self) -> &'static str {
        if self.ready() {
            "captured_replayed_verified"
        } else if !self.requested {
            "not_requested"
        } else if !self.blockers.is_empty() {
            "blocked_preflight"
        } else if self.verified {
            "verified_but_incomplete"
        } else if self.replayed {
            "replayed_but_not_verified"
        } else if self.instantiated {
            "instantiated_but_not_replayed"
        } else if self.captured {
            "captured_but_not_instantiated"
        } else {
            "failed"
        }
    }
}

impl GenerationStats {
    pub(super) fn record_routes(
        &mut self,
        backend: EpBackendKind,
        local_routes: usize,
        remote_routes: usize,
    ) {
        match backend {
            EpBackendKind::HostStaged => {
                self.host_dispatch_local_routes += local_routes;
                self.host_dispatch_remote_routes += remote_routes;
            }
            EpBackendKind::Nccl => {
                self.nccl_dispatch_local_routes += local_routes;
                self.nccl_dispatch_remote_routes += remote_routes;
                self.nccl_combine_routes += local_routes + remote_routes;
            }
        }
    }

    pub(super) fn record_host_staged_moe(&mut self, hidden_dim: usize, route_count: usize) {
        let elements = hidden_dim * route_count;
        self.host_dispatch_calls += 1;
        self.host_combine_calls += 1;
        self.host_dispatch_elements += elements;
        self.host_combine_elements += elements;
    }

    pub(super) fn record_nccl_moe_collectives(&mut self, hidden_dim: usize, seq_len: usize) {
        let elements = hidden_dim * seq_len;
        self.nccl_dense_exchange_calls += 1;
        self.nccl_combine_calls += 1;
        self.nccl_dense_exchange_elements += elements;
        self.nccl_combine_elements += elements;
    }
}
