//! vLLM HTTP mapping: /health, /ready, runtime metrics, and the watchdog.
//!
//! The state machine lives in `engine::health`; this module only translates state into HTTP and owns watchdog exit.

use std::sync::Arc;
use std::sync::RwLock;
use std::time::Duration;

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::Request;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use log::warn;
use serde_json::json;
use tokio_util::sync::CancellationToken;
use tower::ServiceExt;

use crate::engine::EngineHealth;
use crate::engine::EnginePhase;
use crate::vllm::admission::Admission;

pub(crate) type HealthSlot = Arc<RwLock<Option<Vec<EngineHealth>>>>;

#[derive(Clone)]
struct HttpState {
    health: HealthSlot,
    admission: Arc<Admission>,
    /// Models without managed health still forward vLLM's own /health.
    fallback: Router,
}

pub(crate) fn new_health_slot() -> HealthSlot {
    Arc::new(RwLock::new(None))
}

pub(crate) fn publish_health(slot: &HealthSlot, healths: Vec<EngineHealth>) {
    *slot.write().expect("health slot poisoned") = Some(healths);
}

/// Worst phase across every scheduler, the first failure, and the oldest beat.
struct Aggregate {
    phase: EnginePhase,
    failure: Option<String>,
    heartbeat_age: Duration,
}

fn aggregate(healths: &[EngineHealth]) -> Aggregate {
    let mut aggregate = Aggregate {
        phase: EnginePhase::Ready,
        failure: None,
        heartbeat_age: Duration::ZERO,
    };
    for health in healths {
        let phase = health.phase();
        aggregate.phase = match (aggregate.phase, phase) {
            (EnginePhase::Failed, _) | (_, EnginePhase::Failed) => EnginePhase::Failed,
            (EnginePhase::Loading, _) | (_, EnginePhase::Loading) => EnginePhase::Loading,
            (EnginePhase::Draining, _) | (_, EnginePhase::Draining) => EnginePhase::Draining,
            _ => EnginePhase::Ready,
        };
        if phase == EnginePhase::Failed && aggregate.failure.is_none() {
            aggregate.failure = health.failure();
        }
        aggregate.heartbeat_age = aggregate.heartbeat_age.max(health.heartbeat_age());
    }
    aggregate
}

pub(crate) fn routes(health: HealthSlot, admission: Arc<Admission>, vllm_router: Router) -> Router {
    let state = HttpState {
        health,
        admission: admission.clone(),
        fallback: vllm_router.clone(),
    };
    Router::new()
        .route("/health", get(health_handler))
        .route("/ready", get(ready_handler))
        .route("/pegainfer/metrics", get(metrics_handler))
        .fallback_service(vllm_router)
        .layer(axum::middleware::from_fn_with_state(
            admission,
            crate::vllm::admission::admit,
        ))
        .with_state(state)
}

async fn health_handler(State(state): State<HttpState>) -> Response {
    let healths = state.health.read().expect("health slot poisoned").clone();
    let Some(healths) = healths else {
        return forward_health(&state.fallback).await;
    };
    let aggregate = aggregate(&healths);
    let status = match aggregate.phase {
        EnginePhase::Ready | EnginePhase::Draining => StatusCode::OK,
        EnginePhase::Loading | EnginePhase::Failed => StatusCode::SERVICE_UNAVAILABLE,
    };
    (
        status,
        axum::Json(json!({
            "status": if aggregate.phase == EnginePhase::Failed { "failed" } else { "ok" },
            "phase": aggregate.phase.as_str(),
            "heartbeat_age_ms": aggregate.heartbeat_age.as_millis() as u64,
            "failure": aggregate.failure,
        })),
    )
        .into_response()
}

async fn ready_handler(State(state): State<HttpState>) -> Response {
    let healths = state.health.read().expect("health slot poisoned").clone();
    if let Some(healths) = healths {
        let aggregate = aggregate(&healths);
        let ready = aggregate.phase == EnginePhase::Ready && state.admission.is_accepting();
        let status = if ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        };
        return (
            status,
            axum::Json(json!({
                "ready": ready,
                "phase": aggregate.phase.as_str(),
            })),
        )
            .into_response();
    }
    // The legacy handle contract has no EngineHealth. The listener only
    // binds after the engine registers, so transport health is the
    // readiness signal on those paths.
    let accepting = state.admission.is_accepting();
    let healthy = forward_health(&state.fallback).await.status().is_success();
    let ready = healthy && accepting;
    let phase = if !accepting {
        "draining"
    } else if healthy {
        "ready"
    } else {
        "loading"
    };
    let status = if ready {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    };
    (
        status,
        axum::Json(json!({
            "ready": ready,
            "phase": phase,
        })),
    )
        .into_response()
}

async fn metrics_handler(State(state): State<HttpState>) -> Response {
    let healths = state.health.read().expect("health slot poisoned").clone();
    let (phase, heartbeat_age_ms, failure) = if let Some(healths) = healths {
        let aggregate = aggregate(&healths);
        (
            aggregate.phase.as_str(),
            aggregate.heartbeat_age.as_millis() as u64,
            aggregate.failure,
        )
    } else {
        // Legacy handles keep their health in the transport layer. Derive the
        // serving phase from that signal so metrics agree with /ready.
        let healthy = forward_health(&state.fallback).await.status().is_success();
        let phase = if healthy {
            if state.admission.is_accepting() {
                "ready"
            } else {
                "draining"
            }
        } else {
            "loading"
        };
        (phase, 0, None)
    };
    axum::Json(json!({
        "phase": phase,
        "heartbeat_age_ms": heartbeat_age_ms,
        "failure": failure,
        "queue_depth": state.admission.waiting(),
        "active_requests": state.admission.active(),
        "admission_rejected_full": state.admission.rejected_full(),
        "admission_rejected_engine": state.admission.rejected_engine(),
        "admission_rejected_deadline": state.admission.rejected_deadline(),
    }))
    .into_response()
}

async fn forward_health(fallback: &Router) -> Response {
    let request = Request::builder()
        .method("GET")
        .uri("/health")
        .body(Body::empty())
        .expect("static health request must build");
    match fallback.clone().oneshot(request).await {
        Ok(response) => response,
        Err(error) => (
            StatusCode::SERVICE_UNAVAILABLE,
            axum::Json(json!({"status": "error", "message": error.to_string()})),
        )
            .into_response(),
    }
}

/// Standalone watchdog: after a 30s heartbeat stall, switch /health to 503 first, then terminate the process.
// Terminating the process is the watchdog's job; the supervisor is responsible for restarting it.
#[allow(clippy::exit)]
pub(crate) fn start_watchdog(
    slot: HealthSlot,
    admission: Arc<Admission>,
    server_shutdown: CancellationToken,
) {
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(Duration::from_secs(1));
        loop {
            tick.tick().await;
            // A server that shut down cleanly no longer has a heartbeat to
            // watch; without this, an old watchdog outlives its server and
            // eventually kills an unrelated process owner.
            if server_shutdown.is_cancelled() {
                return;
            }
            let healths = slot.read().expect("health slot poisoned").clone();
            let Some(healths) = healths else {
                continue;
            };
            let Some(reason) = healths.iter().find_map(EngineHealth::watchdog_failure) else {
                continue;
            };
            warn!("engine watchdog failed: {reason}");
            // Heartbeat stalled: close admission immediately so no new request reaches a scheduler that is not advancing.
            admission.stop_accepting();
            // Keep a scrapeable 503 window, then exit; the supervisor restarts the process.
            tokio::time::sleep(Duration::from_secs(2)).await;
            server_shutdown.cancel();
            tokio::time::sleep(Duration::from_secs(2)).await;
            std::process::exit(1);
        }
    });
}

/// On SIGINT/SIGTERM, stop accepting and wait for in-flight requests for at most 22s;
/// keep a 2s /ready 503 observation window, then cancel the HTTP server. The
/// process-level deadline for the remaining engine cleanup is owned by the
/// binary, so library users and tests never kill an unrelated process.
pub(crate) fn drain_on_shutdown(
    shutdown: CancellationToken,
    slot: HealthSlot,
    admission: Arc<Admission>,
    server_shutdown: CancellationToken,
) {
    tokio::spawn(async move {
        shutdown.cancelled().await;
        let healths = slot.read().expect("health slot poisoned").clone();
        if let Some(healths) = healths {
            for health in &healths {
                health.mark_draining();
            }
        }
        admission.stop_accepting();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(22);
        while admission.active() > 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        // Keep a /ready 503 observation window before closing the listener.
        tokio::time::sleep(Duration::from_secs(2)).await;
        server_shutdown.cancel();
    });
}

#[cfg(test)]
mod tests {
    use axum::body::to_bytes;
    use tower::ServiceExt;

    use super::*;

    #[tokio::test]
    async fn health_and_ready_follow_engine_phase() {
        let health = new_health_slot();
        let admission = Admission::new();
        let engine = EngineHealth::new();
        publish_health(&health, vec![engine.clone()]);
        let app = routes(health, admission, Router::new());

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);

        engine.mark_ready();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/ready")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        engine.mark_draining();
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/ready")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn ready_falls_back_to_transport_health() {
        let fallback = Router::new().route("/health", get(|| async { StatusCode::OK }));
        let app = routes(new_health_slot(), Admission::new(), fallback);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/ready")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["ready"], true);
    }

    #[tokio::test]
    async fn ready_reports_draining_for_legacy_handle_after_shutdown() {
        let fallback = Router::new().route("/health", get(|| async { StatusCode::OK }));
        let admission = Admission::new();
        admission.stop_accepting();
        let app = routes(new_health_slot(), admission, fallback);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/ready")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["phase"], "draining");
    }

    #[tokio::test]
    async fn metrics_falls_back_to_transport_health() {
        let fallback = Router::new().route("/health", get(|| async { StatusCode::OK }));
        let app = routes(new_health_slot(), Admission::new(), fallback);
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/pegainfer/metrics")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["phase"], "ready");
    }

    #[tokio::test]
    async fn health_reports_failure() {
        let health = new_health_slot();
        let admission = Admission::new();
        let engine = EngineHealth::new();
        engine.mark_ready();
        engine.mark_failed("rank 0 prefill failed");
        publish_health(&health, vec![engine]);
        let app = routes(health, admission, Router::new());

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["status"], "failed");
        assert_eq!(body["failure"], "rank 0 prefill failed");
    }

    #[tokio::test]
    async fn any_scheduler_failure_fails_the_engine() {
        let health = new_health_slot();
        let first = EngineHealth::new();
        first.mark_ready();
        let second = EngineHealth::new();
        second.mark_ready();
        second.mark_failed("partition 1 failed");
        publish_health(&health, vec![first, second]);
        let app = routes(health, Admission::new(), Router::new());

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        let body: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(body["failure"], "partition 1 failed");

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/ready")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
