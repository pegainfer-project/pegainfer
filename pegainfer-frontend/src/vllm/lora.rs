use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use axum::Json;
use axum::Router;
use axum::body::Body;
use axum::body::Bytes;
use axum::body::to_bytes;
use axum::extract::Request;
use axum::extract::State;
use axum::http::HeaderValue;
use axum::http::StatusCode;
use axum::http::header::CONTENT_LENGTH;
use axum::response::IntoResponse;
use axum::response::Response;
use axum::routing::get;
use axum::routing::post;
use serde::Deserialize;
use serde::Serialize;
use tokio::sync::RwLock;
use tower::ServiceExt;

use crate::engine::LoadLoraAdapterRequest;
use crate::engine::LoraClient;
use crate::engine::LoraControlError;
use crate::engine::UnloadLoraAdapterRequest;
use crate::vllm::wire::LORA_ADAPTER_XARG;

const LORA_ROUTE_BODY_LIMIT: usize = 128 * 1024 * 1024;

#[derive(Clone)]
struct LoraRouteState {
    control: LoraClient,
    adapter_names: Arc<RwLock<HashSet<String>>>,
}

#[derive(Clone)]
struct LoraOpenAiState {
    vllm_router: Router,
    base_model_name: String,
    served_model_names: Vec<String>,
    adapter_names: Arc<RwLock<HashSet<String>>>,
}

#[derive(Debug, Deserialize)]
struct LoadLoraAdapterHttpRequest {
    lora_name: String,
    lora_path: PathBuf,
    #[serde(default)]
    load_inplace: bool,
    #[serde(default)]
    is_3d_lora_weight: bool,
}

#[derive(Debug, Deserialize)]
struct UnloadLoraAdapterHttpRequest {
    lora_name: String,
    #[serde(default)]
    lora_int_id: Option<i64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LoraModule {
    name: String,
    path: PathBuf,
}

/// CLI parser for `--lora-modules`: vLLM-style `name=path`, a JSON object,
/// or a single-entry JSON list with `name` and `path`.
pub fn parse_lora_modules_arg(value: &str) -> Result<LoraModule, String> {
    if let Some((name, path)) = value.split_once('=') {
        return parse_lora_module_fields(name, path);
    }
    let json: serde_json::Value =
        serde_json::from_str(value).map_err(|error| format!("invalid --lora-modules: {error}"))?;
    match json {
        serde_json::Value::Object(map) => parse_lora_module_json(&map),
        serde_json::Value::Array(entries) if entries.len() == 1 => {
            let Some(serde_json::Value::Object(map)) = entries.first() else {
                return Err("--lora-modules JSON list entries must be objects".to_string());
            };
            parse_lora_module_json(map)
        }
        serde_json::Value::Array(_) => Err(
            "pass multiple --lora-modules values instead of one JSON list with multiple entries"
                .to_string(),
        ),
        _ => Err(
            "--lora-modules must be `name=path`, a JSON object, or a single-entry JSON list"
                .to_string(),
        ),
    }
}

fn parse_lora_module_json(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Result<LoraModule, String> {
    let name = map
        .get("name")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "--lora-modules JSON object requires string field `name`".to_string())?;
    let path = map
        .get("path")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| "--lora-modules JSON object requires string field `path`".to_string())?;
    parse_lora_module_fields(name, path)
}

fn parse_lora_module_fields(name: &str, path: &str) -> Result<LoraModule, String> {
    if name.is_empty() {
        return Err("--lora-modules name must not be empty".to_string());
    }
    if path.is_empty() {
        return Err("--lora-modules path must not be empty".to_string());
    }
    Ok(LoraModule {
        name: name.to_string(),
        path: PathBuf::from(path),
    })
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    error: String,
}

#[derive(Debug, Serialize)]
struct ModelListBody {
    object: &'static str,
    data: Vec<ModelCardBody>,
}

#[derive(Debug, Serialize)]
struct ModelCardBody {
    id: String,
    object: &'static str,
    created: i64,
    owned_by: &'static str,
}

pub(crate) fn lora_routes(
    control: LoraClient,
    adapter_names: Arc<RwLock<HashSet<String>>>,
) -> Router {
    Router::new()
        .route("/v1/load_lora_adapter", post(load_lora_adapter))
        .route("/v1/unload_lora_adapter", post(unload_lora_adapter))
        .with_state(LoraRouteState {
            control,
            adapter_names,
        })
}

pub(crate) fn lora_openai_routes(
    vllm_router: Router,
    base_model_name: String,
    served_model_names: Vec<String>,
    adapter_names: Arc<RwLock<HashSet<String>>>,
) -> Router {
    Router::new()
        .route("/v1/models", get(lora_models))
        .route("/v1/completions", post(forward_lora_openai_request))
        .route("/v1/chat/completions", post(forward_lora_openai_request))
        .with_state(LoraOpenAiState {
            vllm_router,
            base_model_name,
            served_model_names,
            adapter_names,
        })
}

async fn load_lora_adapter(
    axum::extract::State(state): axum::extract::State<LoraRouteState>,
    Json(request): Json<LoadLoraAdapterHttpRequest>,
) -> Response {
    if request.lora_name.is_empty() {
        return bad_request("lora_name must not be empty");
    }
    if request.lora_path.as_os_str().is_empty() {
        return bad_request("lora_path must not be empty");
    }
    if request.is_3d_lora_weight {
        return bad_request("is_3d_lora_weight=true is not supported by Qwen3 LoRA PR1");
    }

    let lora_name = request.lora_name.clone();
    match state
        .control
        .load(LoadLoraAdapterRequest {
            lora_name: request.lora_name,
            lora_path: request.lora_path,
            load_inplace: request.load_inplace,
        })
        .await
    {
        Ok(()) => {
            state.adapter_names.write().await.insert(lora_name.clone());
            (
                StatusCode::OK,
                format!("Success: LoRA adapter '{lora_name}' added successfully."),
            )
                .into_response()
        }
        Err(error @ LoraControlError::EngineGone) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorBody {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(LoraControlError::Failed(message)) => {
            (StatusCode::BAD_REQUEST, Json(ErrorBody { error: message })).into_response()
        }
    }
}

async fn unload_lora_adapter(
    axum::extract::State(state): axum::extract::State<LoraRouteState>,
    Json(request): Json<UnloadLoraAdapterHttpRequest>,
) -> Response {
    if request.lora_name.is_empty() {
        return bad_request("lora_name must not be empty");
    }

    let lora_name = request.lora_name.clone();
    match state
        .control
        .unload(UnloadLoraAdapterRequest {
            lora_name: request.lora_name,
            lora_int_id: request.lora_int_id,
        })
        .await
    {
        Ok(()) => {
            state.adapter_names.write().await.remove(&lora_name);
            (
                StatusCode::OK,
                format!("Success: LoRA adapter '{lora_name}' removed successfully."),
            )
                .into_response()
        }
        Err(error @ LoraControlError::EngineGone) => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(ErrorBody {
                error: error.to_string(),
            }),
        )
            .into_response(),
        Err(LoraControlError::Failed(message)) => {
            (StatusCode::BAD_REQUEST, Json(ErrorBody { error: message })).into_response()
        }
    }
}

fn bad_request(message: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(ErrorBody {
            error: message.into(),
        }),
    )
        .into_response()
}

pub(crate) async fn load_startup_lora_modules(
    control: &LoraClient,
    adapter_names: &Arc<RwLock<HashSet<String>>>,
    lora_modules: &[LoraModule],
) -> Result<()> {
    for module in lora_modules {
        control
            .load(LoadLoraAdapterRequest {
                lora_name: module.name.clone(),
                lora_path: module.path.clone(),
                load_inplace: false,
            })
            .await
            .with_context(|| {
                format!(
                    "failed to load startup LoRA module {} from {}",
                    module.name,
                    module.path.display()
                )
            })?;
        adapter_names.write().await.insert(module.name.clone());
    }
    Ok(())
}

async fn lora_models(State(state): State<LoraOpenAiState>) -> Response {
    lora_models_response(
        &state.served_model_names,
        &state.base_model_name,
        &state.adapter_names,
    )
    .await
}

async fn forward_lora_openai_request(
    State(state): State<LoraOpenAiState>,
    request: Request,
) -> Response {
    match forward_lora_openai_request_inner(state, request).await {
        Ok(response) => response,
        Err(error) => (
            StatusCode::BAD_REQUEST,
            Json(ErrorBody {
                error: format!("failed to prepare LoRA request: {error:#}"),
            }),
        )
            .into_response(),
    }
}

async fn forward_lora_openai_request_inner(
    state: LoraOpenAiState,
    request: Request,
) -> Result<Response> {
    let (mut parts, body) = request.into_parts();
    let mut body = to_bytes(body, LORA_ROUTE_BODY_LIMIT)
        .await
        .context("failed to read OpenAI request body")?;
    let _ =
        rewrite_lora_request_body(&mut body, &state.base_model_name, &state.adapter_names).await?;
    parts.headers.insert(
        CONTENT_LENGTH,
        HeaderValue::from_str(&body.len().to_string())
            .context("rewritten body length is invalid")?,
    );

    state
        .vllm_router
        .oneshot(Request::from_parts(parts, Body::from(body)))
        .await
        .context("vLLM router failed to handle LoRA request")
}

async fn rewrite_lora_request_body(
    body: &mut Bytes,
    base_model_name: &str,
    adapter_names: &Arc<RwLock<HashSet<String>>>,
) -> Result<Option<String>> {
    let mut value: serde_json::Value =
        serde_json::from_slice(body).context("failed to parse OpenAI request JSON")?;
    let Some(model) = value.get("model").and_then(serde_json::Value::as_str) else {
        return Ok(None);
    };
    if model == base_model_name {
        return Ok(None);
    }
    if !adapter_names.read().await.contains(model) {
        return Ok(None);
    }
    let adapter = model.to_string();
    value["model"] = serde_json::Value::String(base_model_name.to_string());
    let Some(map) = value.as_object_mut() else {
        return Ok(None);
    };
    let xargs = map
        .entry("vllm_xargs")
        .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
    if !xargs.is_object() {
        *xargs = serde_json::Value::Object(serde_json::Map::new());
    }
    xargs
        .as_object_mut()
        .expect("vllm_xargs must be object")
        .insert(
            LORA_ADAPTER_XARG.to_string(),
            serde_json::Value::String(adapter.clone()),
        );
    *body = Bytes::from(serde_json::to_vec(&value)?);
    Ok(Some(adapter))
}

async fn lora_models_response(
    served_model_names: &[String],
    base_model_name: &str,
    adapter_names: &Arc<RwLock<HashSet<String>>>,
) -> Response {
    let mut ids: Vec<String> = if served_model_names.is_empty() {
        vec![base_model_name.to_string()]
    } else {
        served_model_names.to_vec()
    };
    ids.extend(adapter_names.read().await.iter().cloned());
    ids.sort();
    ids.dedup();
    Json(ModelListBody {
        object: "list",
        data: ids
            .into_iter()
            .map(|id| ModelCardBody {
                id,
                object: "model",
                created: 0,
                owned_by: "vllm-frontend-rs",
            })
            .collect(),
    })
    .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A client whose engine is already gone: the receiver is dropped, so
    /// every roundtrip answers `EngineGone`. (An engine without LoRA never
    /// mounts these routes at all — the `Option` on `Engine::lora` is the
    /// capability — so "unsupported" is not a reachable route state.)
    fn gone_engine_client() -> LoraClient {
        let (client, _rx) = LoraClient::channel();
        client
    }

    fn route_state(control: LoraClient) -> LoraRouteState {
        LoraRouteState {
            control,
            adapter_names: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    #[tokio::test]
    async fn load_lora_adapter_route_reports_gone_engine() {
        let state = route_state(gone_engine_client());
        let response = load_lora_adapter(
            axum::extract::State(state),
            Json(LoadLoraAdapterHttpRequest {
                lora_name: "adapter-a".to_string(),
                lora_path: PathBuf::from("/tmp/adapter-a"),
                load_inplace: false,
                is_3d_lora_weight: false,
            }),
        )
        .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn load_lora_adapter_route_rejects_pr1_unsupported_fields() {
        let state = route_state(gone_engine_client());
        let response = load_lora_adapter(
            axum::extract::State(state),
            Json(LoadLoraAdapterHttpRequest {
                lora_name: "adapter-a".to_string(),
                lora_path: PathBuf::from("/tmp/adapter-a"),
                load_inplace: false,
                is_3d_lora_weight: true,
            }),
        )
        .await;

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn unload_lora_adapter_route_reports_gone_engine() {
        let state = route_state(gone_engine_client());
        let response = unload_lora_adapter(
            axum::extract::State(state),
            Json(UnloadLoraAdapterHttpRequest {
                lora_name: "adapter-a".to_string(),
                lora_int_id: None,
            }),
        )
        .await;

        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
    }

    #[tokio::test]
    async fn rewrite_lora_request_body_maps_adapter_model_to_base_and_xarg() {
        let adapter_names = Arc::new(RwLock::new(HashSet::from(["adapter-a".to_string()])));
        let mut body = Bytes::from(
            serde_json::json!({
                "model": "adapter-a",
                "prompt": "hello"
            })
            .to_string(),
        );

        let selected = rewrite_lora_request_body(&mut body, "base-model", &adapter_names)
            .await
            .expect("rewrite request");

        assert_eq!(selected.as_deref(), Some("adapter-a"));
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json body");
        assert_eq!(value["model"], "base-model");
        assert_eq!(value["prompt"], "hello");
        assert_eq!(value["vllm_xargs"][LORA_ADAPTER_XARG], "adapter-a");
    }

    #[tokio::test]
    async fn rewrite_lora_request_body_leaves_base_and_unknown_models_untouched() {
        let adapter_names = Arc::new(RwLock::new(HashSet::from(["adapter-a".to_string()])));
        let mut base_body = Bytes::from(r#"{"model":"base-model","prompt":"hello"}"#);
        let selected = rewrite_lora_request_body(&mut base_body, "base-model", &adapter_names)
            .await
            .expect("base request");
        assert_eq!(selected, None);
        assert_eq!(
            &base_body[..],
            br#"{"model":"base-model","prompt":"hello"}"#
        );

        let mut unknown_body = Bytes::from(r#"{"model":"missing-adapter","prompt":"hello"}"#);
        let selected = rewrite_lora_request_body(&mut unknown_body, "base-model", &adapter_names)
            .await
            .expect("unknown adapter request");
        assert_eq!(selected, None);
        assert_eq!(
            &unknown_body[..],
            br#"{"model":"missing-adapter","prompt":"hello"}"#
        );
    }

    #[tokio::test]
    async fn lora_openai_forwarder_rewrites_then_calls_vllm_router() {
        let adapter_names = Arc::new(RwLock::new(HashSet::from(["adapter-a".to_string()])));
        let vllm_router = Router::new().route(
            "/v1/completions",
            post(|headers: axum::http::HeaderMap, body: Bytes| async move {
                let content_length = headers
                    .get(CONTENT_LENGTH)
                    .and_then(|value| value.to_str().ok())
                    .and_then(|value| value.parse::<usize>().ok())
                    .expect("content-length header");
                assert_eq!(content_length, body.len());
                Json(serde_json::from_slice::<serde_json::Value>(&body).expect("json body"))
            }),
        );
        let state = LoraOpenAiState {
            vllm_router,
            base_model_name: "base-model".to_string(),
            served_model_names: vec!["base-model".to_string()],
            adapter_names,
        };
        let request = Request::builder()
            .method("POST")
            .uri("/v1/completions")
            .body(Body::from(
                serde_json::json!({
                    "model": "adapter-a",
                    "prompt": "hello"
                })
                .to_string(),
            ))
            .expect("request");

        let response = forward_lora_openai_request_inner(state, request)
            .await
            .expect("forward request");

        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), LORA_ROUTE_BODY_LIMIT)
            .await
            .expect("read body");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("json response");
        assert_eq!(value["model"], "base-model");
        assert_eq!(value["prompt"], "hello");
        assert_eq!(value["vllm_xargs"][LORA_ADAPTER_XARG], "adapter-a");
    }

    #[tokio::test]
    async fn lora_models_response_includes_base_and_loaded_adapters() {
        let adapter_names = Arc::new(RwLock::new(HashSet::from([
            "adapter-b".to_string(),
            "adapter-a".to_string(),
        ])));

        let response =
            lora_models_response(&["served-base".to_string()], "model-path", &adapter_names).await;
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body(), LORA_ROUTE_BODY_LIMIT)
            .await
            .expect("read body");
        let value: serde_json::Value = serde_json::from_slice(&body).expect("models JSON");
        let ids = value["data"]
            .as_array()
            .expect("data array")
            .iter()
            .map(|entry| entry["id"].as_str().expect("id string"))
            .collect::<Vec<_>>();
        assert_eq!(ids, vec!["adapter-a", "adapter-b", "served-base"]);
    }
}
