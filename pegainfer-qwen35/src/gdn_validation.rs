#![cfg(feature = "gdn-validation")]

//! Non-default runtime evidence for the required SM120 GDN validation gates.

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use anyhow::Result;

use crate::weights::Qwen35Model;

#[derive(Debug, Default)]
struct GdnValidationEvidenceCounters {
    successful_launches: AtomicU64,
    graph_captures: AtomicU64,
    graph_replays: AtomicU64,
    graph_eager_fallbacks: AtomicU64,
    state_slot_copies: AtomicU64,
    state_slot_reuses: AtomicU64,
    slot_compactions: AtomicU64,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct GdnValidationEvidenceHandle {
    counters: Arc<GdnValidationEvidenceCounters>,
}

impl GdnValidationEvidenceHandle {
    pub(crate) fn record_successful_launch(&self) {
        self.counters
            .successful_launches
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_graph_capture(&self) {
        self.counters.graph_captures.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_graph_replay(&self) {
        self.counters.graph_replays.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_graph_eager_fallback(&self) {
        self.counters
            .graph_eager_fallbacks
            .fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn record_state_slot_copy(&self, reused: bool) {
        self.counters
            .state_slot_copies
            .fetch_add(1, Ordering::Relaxed);
        if reused {
            self.counters
                .state_slot_reuses
                .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(crate) fn record_slot_compaction(&self) {
        self.counters
            .slot_compactions
            .fetch_add(1, Ordering::Relaxed);
    }
}

/// Runtime proof for production dispatch and same-path validation gates.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GdnPrefillRuntimeEvidence {
    pub selected_backend: String,
    pub artifact_sha256: String,
    pub successful_launches: u64,
    pub graph_captures: u64,
    pub graph_replays: u64,
    pub graph_eager_fallbacks: u64,
    pub state_slot_copies: u64,
    pub state_slot_reuses: u64,
    pub slot_compactions: u64,
}

#[derive(Clone, Debug)]
pub struct GdnPrefillRuntimeEvidenceHandle {
    selected_backend: &'static str,
    artifact_sha256: String,
    validation: GdnValidationEvidenceHandle,
}

impl GdnPrefillRuntimeEvidenceHandle {
    pub fn snapshot(&self) -> GdnPrefillRuntimeEvidence {
        let counters = &self.validation.counters;
        GdnPrefillRuntimeEvidence {
            selected_backend: self.selected_backend.to_owned(),
            artifact_sha256: self.artifact_sha256.clone(),
            successful_launches: counters.successful_launches.load(Ordering::Relaxed),
            graph_captures: counters.graph_captures.load(Ordering::Relaxed),
            graph_replays: counters.graph_replays.load(Ordering::Relaxed),
            graph_eager_fallbacks: counters.graph_eager_fallbacks.load(Ordering::Relaxed),
            state_slot_copies: counters.state_slot_copies.load(Ordering::Relaxed),
            state_slot_reuses: counters.state_slot_reuses.load(Ordering::Relaxed),
            slot_compactions: counters.slot_compactions.load(Ordering::Relaxed),
        }
    }
}

impl Qwen35Model {
    pub fn flashinfer_gdn_runtime_evidence(&self) -> Result<GdnPrefillRuntimeEvidence> {
        Ok(self.flashinfer_gdn_runtime_evidence_handle()?.snapshot())
    }

    pub fn flashinfer_gdn_runtime_evidence_handle(
        &self,
    ) -> Result<GdnPrefillRuntimeEvidenceHandle> {
        let backend = self.flashinfer_gdn()?;
        Ok(GdnPrefillRuntimeEvidenceHandle {
            selected_backend: "flashinfer",
            artifact_sha256: backend.artifact_sha256().to_owned(),
            validation: self.gdn_validation_evidence.clone(),
        })
    }
}
