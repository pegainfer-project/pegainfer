use super::DecodeGraphBlocker;
use super::EpBackendKind;
use super::FullDecodeGraphProbeReport;
use super::decode_graph_blockers;
use super::decode_graph_readiness_status;
use super::full_decode_graph_probe_report;

#[test]
fn nccl_readiness_has_no_static_blockers_after_probe_wiring() {
    let blockers = decode_graph_blockers(EpBackendKind::Nccl);
    let ids: Vec<_> = blockers.iter().map(|blocker| blocker.id).collect();

    assert!(ids.is_empty(), "unexpected NCCL blockers: {ids:?}");
}

#[test]
fn full_decode_probe_is_not_requested_by_default() {
    let report = full_decode_graph_probe_report(EpBackendKind::Nccl, false, &[]).unwrap();

    assert!(!report.requested);
    assert!(!report.ready());
    assert_eq!(report.coverage_status(), "not_requested");
    assert_eq!(report.failure_stage, "not_requested");
    assert!(report.failure_summary.is_none());
    assert!(report.blockers.is_empty());
}

#[test]
fn full_decode_probe_preflight_blockers_fail_closed_and_serialize() {
    let blockers = vec![synthetic_blocker("synthetic_probe_blocker")];
    let report = full_decode_graph_probe_report(EpBackendKind::Nccl, true, &blockers).unwrap();
    let json = serde_json::to_value(&report).unwrap();

    assert_requested_but_not_captured(&report);
    assert_eq!(report.failure_stage, "preflight_blocked");
    assert_eq!(report.coverage_status(), "blocked_preflight");
    assert_eq!(report.blockers.len(), 1);
    assert!(
        report
            .failure_summary
            .as_deref()
            .is_some_and(|summary| summary.contains("synthetic_probe_blocker"))
    );
    assert_eq!(json["requested"], true);
    assert_eq!(json["replay_count"], 0);
    assert_eq!(json["verified_replay_count"], 0);
    assert_eq!(json["failure_stage"], "preflight_blocked");
    assert_eq!(json["capture_mode"], "thread_local");
    assert_eq!(json["blockers"][0]["id"], "synthetic_probe_blocker");
}

#[test]
fn full_decode_probe_rejects_host_staged_backend() {
    let blockers = decode_graph_blockers(EpBackendKind::HostStaged);
    let err =
        full_decode_graph_probe_report(EpBackendKind::HostStaged, true, &blockers).unwrap_err();

    assert!(
        err.to_string()
            .contains("--full-decode-graph-probe requires")
    );
}

#[test]
fn readiness_status_tracks_request_and_success() {
    assert_eq!(
        decode_graph_readiness_status(EpBackendKind::HostStaged, false, true),
        "not_applicable_host_staged_backend"
    );
    assert_eq!(
        decode_graph_readiness_status(EpBackendKind::Nccl, false, false),
        "full_decode_probe_not_requested"
    );
    assert_eq!(
        decode_graph_readiness_status(EpBackendKind::Nccl, false, true),
        "blocked_full_decode_path"
    );
    assert_eq!(
        decode_graph_readiness_status(EpBackendKind::Nccl, true, true),
        "full_decode_capture_ready"
    );
}

#[test]
fn full_decode_probe_ready_requires_complete_replay_verification() {
    let complete = complete_probe_report();
    assert!(complete.ready());
    assert_eq!(complete.coverage_status(), "captured_replayed_verified");

    let missing_instantiation = FullDecodeGraphProbeReport {
        instantiated: false,
        ..complete.clone()
    };
    assert!(!missing_instantiation.ready());
    assert_eq!(
        missing_instantiation.coverage_status(),
        "verified_but_incomplete"
    );

    let no_replay = FullDecodeGraphProbeReport {
        replay_count: 0,
        verified_replay_count: 0,
        ..complete.clone()
    };
    assert!(!no_replay.ready());

    let partial_replay = FullDecodeGraphProbeReport {
        verified_replay_count: complete.replay_count - 1,
        ..complete
    };
    assert!(!partial_replay.ready());
    assert_eq!(partial_replay.coverage_status(), "verified_but_incomplete");
}

#[test]
fn full_decode_probe_shape_limit_blocker_is_serializable() {
    let report = FullDecodeGraphProbeReport {
        requested: true,
        captured: false,
        instantiated: false,
        replayed: false,
        verified: false,
        replay_count: 0,
        verified_replay_count: 0,
        failure_stage: "preflight_blocked",
        failure_summary: Some(
            "max_seq_len=4097 exceeds the fixed-topology probe kernel limit 4096".to_string(),
        ),
        blockers: vec![DecodeGraphBlocker {
            id: "decode_graph_probe_shape_limit",
            source: "runtime/graph_probe.rs::full_decode_graph_probe_inner",
            reason: "test-only shape blocker",
        }],
        capture_mode: "thread_local",
    };
    let json = serde_json::to_value(&report).unwrap();

    assert_requested_but_not_captured(&report);
    assert_eq!(report.coverage_status(), "blocked_preflight");
    assert_eq!(json["failure_stage"], "preflight_blocked");
    assert_eq!(json["blockers"][0]["id"], "decode_graph_probe_shape_limit");
}

fn complete_probe_report() -> FullDecodeGraphProbeReport {
    FullDecodeGraphProbeReport {
        requested: true,
        captured: true,
        instantiated: true,
        replayed: true,
        verified: true,
        replay_count: 8,
        verified_replay_count: 8,
        failure_stage: "none",
        failure_summary: None,
        blockers: Vec::new(),
        capture_mode: "thread_local",
    }
}

fn synthetic_blocker(id: &'static str) -> DecodeGraphBlocker {
    DecodeGraphBlocker {
        id,
        source: "runtime/readiness/tests.rs",
        reason: "test-only blocker",
    }
}

fn assert_requested_but_not_captured(report: &FullDecodeGraphProbeReport) {
    assert!(report.requested);
    assert!(!report.captured);
    assert!(!report.instantiated);
    assert!(!report.replayed);
    assert!(!report.verified);
    assert_eq!(report.replay_count, 0);
    assert_eq!(report.verified_replay_count, 0);
    assert!(!report.ready());
}
