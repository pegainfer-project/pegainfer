//! The pegainfer server binary: pure dispatch over the compiled-in
//! [`ModelLine`]s. Every model-specific flag, rule, and option type lives in
//! its model crate; this file only wires detection, CLI validation, and the
//! serve path selection together.

use std::time::Instant;

use anyhow::Context;
use clap::FromArgMatches;
use log::info;
use pegainfer_core::logging;
use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::model_line::DetectError;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;
use pegainfer_frontend::model_line::ModelLineRegistry;
use pegainfer_frontend::model_line::SharedArgs;
use pegainfer_frontend::model_line::provided_args;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

fn model_lines() -> Vec<&'static dyn ModelLine> {
    vec![
        #[cfg(feature = "deepseek-v2-lite")]
        &pegainfer_deepseek_v2_lite::model_line::MODEL_LINE,
        #[cfg(feature = "gemma4")]
        &pegainfer_gemma4::model_line::MODEL_LINE,
        #[cfg(feature = "glm52")]
        &pegainfer_glm52::model_line::MODEL_LINE,
        #[cfg(feature = "kimi-k2")]
        &pegainfer_kimi_k2::model_line::MODEL_LINE,
        #[cfg(feature = "qwen3")]
        &pegainfer_qwen3::model_line::MODEL_LINE,
        #[cfg(feature = "qwen35")]
        &pegainfer_qwen35::model_line::MODEL_LINE,
    ]
}

/// When no compiled-in line claims the config but the identity belongs to a
/// known family, tell the user which feature to rebuild with.
///
/// The identity strings here must stay in sync with each line's `probe`
/// gate: this table exists precisely because the line (and its probe) is
/// compiled out, so the duplication cannot be derived away.
fn feature_gate_hint(config: &serde_json::Value) -> Option<String> {
    struct Family {
        feature: &'static str,
        model_types: &'static [&'static str],
        text_model_types: &'static [&'static str],
        compiled: bool,
    }
    let model_type = config
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let text_model_type = config
        .get("text_config")
        .and_then(|text| text.get("model_type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or("");
    let families = [
        Family {
            feature: "deepseek-v2-lite",
            model_types: &["deepseek_v2"],
            text_model_types: &[],
            compiled: cfg!(feature = "deepseek-v2-lite"),
        },
        Family {
            feature: "gemma4",
            model_types: &["gemma4", "gemma4_unified"],
            text_model_types: &["gemma4_text", "gemma4_unified_text"],
            compiled: cfg!(feature = "gemma4"),
        },
        Family {
            feature: "glm52",
            model_types: &["glm_moe_dsa"],
            text_model_types: &[],
            compiled: cfg!(feature = "glm52"),
        },
        Family {
            feature: "kimi-k2",
            model_types: &["kimi_k25", "kimi_k2"],
            text_model_types: &["kimi_k2"],
            compiled: cfg!(feature = "kimi-k2"),
        },
        Family {
            feature: "qwen3",
            model_types: &["qwen3"],
            text_model_types: &[],
            compiled: cfg!(feature = "qwen3"),
        },
        Family {
            feature: "qwen35",
            model_types: &["qwen3_5"],
            text_model_types: &["qwen3_5_text"],
            compiled: cfg!(feature = "qwen35"),
        },
    ];
    families.iter().find_map(|family| {
        (!family.compiled
            && (family.model_types.contains(&model_type)
                || family.text_model_types.contains(&text_model_type)))
        .then(|| {
            format!(
                "this looks like a {} model; rebuild pegainfer-server with --features {}",
                family.feature, family.feature
            )
        })
    })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    logging::init_default();
    pegainfer_core::tracing::init();

    let registry = ModelLineRegistry::new(model_lines());
    let cmd = registry
        .build_command(clap::Command::new("pegainfer").about("PegaInfer GPU inference server"));
    let matches = cmd.clone().get_matches();
    let shared = SharedArgs::from_arg_matches(&matches)
        .map_err(|error| anyhow::anyhow!("invalid CLI args: {error}"))?;
    let provided = provided_args(&matches, &cmd);

    let config_path = shared.model_path.join("config.json");
    let content = std::fs::read_to_string(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;

    let line = registry.detect(&config).map_err(|error| {
        if matches!(error, DetectError::NoMatch { .. }) {
            if let Some(hint) = feature_gate_hint(&config) {
                return anyhow::anyhow!("{error}; {hint}");
            }
        }
        anyhow::Error::from(error)
    })?;

    registry.validate_provided(line, &provided, &cmd)?;
    shared.validate(&provided)?;
    let ctx = LaunchContext {
        model_path: &shared.model_path,
        config: &config,
        shared: &shared,
        matches: &matches,
    };
    line.validate(&ctx, &provided)?;
    let plan = line.serve_plan(&ctx)?;

    info!("=== pegainfer - {} (GPU) ===", line.name());
    info!("Loading engine...");
    let start = Instant::now();
    info!(
        "Runtime: model_path={}, user-set flags {provided:?}",
        shared.model_path.display(),
    );

    let model_path = shared.model_path.clone();
    let served_model_name = shared.served_model_name.clone();
    let port = shared.port;

    // Engine load (weights → GPU) runs on a blocking thread so the HTTP
    // frontend (tokenizer, chat templates) loads concurrently. The frontend
    // binds only after the engine registers, so readiness is unchanged.
    let engine_load = tokio::task::spawn_blocking(move || -> anyhow::Result<LaunchedEngine> {
        let ctx = LaunchContext {
            model_path: &shared.model_path,
            config: &config,
            shared: &shared,
            matches: &matches,
        };
        line.launch(&ctx)
            .with_context(|| format!("failed to start {} engine", line.name()))
    });

    let serve_result = if let Some(lora_modules) = plan.lora_modules {
        // LoRA routes need the engine's control plane when the router is
        // built, so this path stays sequential.
        let launched = engine_load
            .await
            .context("engine loader thread panicked")??;
        let LaunchedEngine::Stepped(engine) = launched else {
            anyhow::bail!("LoRA serving requires a step-driven engine");
        };
        info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());
        let max_model_len =
            pegainfer_frontend::vllm::load_max_model_len(&model_path).unwrap_or(4096);
        pegainfer_frontend::vllm::serve_model_with_lora_routes(
            engine,
            model_path.to_string_lossy().into_owned(),
            served_model_name.into_iter().collect(),
            lora_modules,
            port,
            max_model_len,
            pegainfer_frontend::vllm::shutdown_token_from_ctrl_c(),
        )
        .await
    } else {
        let shutdown = tokio_util::sync::CancellationToken::new();
        let engine = {
            let shutdown = shutdown.clone();
            async move {
                let handle = engine_load
                    .await
                    .context("engine loader thread panicked")??;
                info!("Engine loaded: elapsed_ms={}", start.elapsed().as_millis());
                // The blocking load can't be cancelled, so SIGINT keeps its
                // default kill behavior until the engine is up; only then
                // switch to graceful shutdown.
                pegainfer_frontend::vllm::cancel_token_on_ctrl_c(&shutdown);
                anyhow::Ok(handle)
            }
        };
        if plan.prefill_only {
            pegainfer_frontend::vllm::serve_prefill_only_with_engine_count(
                engine,
                &model_path,
                served_model_name.into_iter().collect(),
                port,
                None,
                plan.scheduler_partition_count,
                shutdown,
            )
            .await
        } else {
            pegainfer_frontend::vllm::serve_with_engine_count(
                engine,
                &model_path,
                served_model_name.into_iter().collect(),
                port,
                None,
                plan.scheduler_partition_count,
                shutdown,
            )
            .await
        }
    }
    .context("vLLM frontend server failed");

    // Export the final batch of request spans before the runtime tears down.
    // Flush before propagating an error too — a failed server is exactly where
    // the last buffered spans matter. No-op when tracing was never enabled.
    pegainfer_core::tracing::flush();
    serve_result?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "glm52"))]
    #[test]
    fn hint_names_the_feature_for_a_known_but_uncompiled_family() {
        let config = serde_json::json!({"model_type": "glm_moe_dsa"});
        let hint = feature_gate_hint(&config).expect("glm52 identity should hint");
        assert!(hint.contains("--features glm52"), "{hint}");
    }

    #[cfg(not(feature = "gemma4"))]
    #[test]
    fn hint_matches_text_config_identities() {
        let config = serde_json::json!({"text_config": {"model_type": "gemma4_unified_text"}});
        let hint = feature_gate_hint(&config).expect("gemma4 text identity should hint");
        assert!(hint.contains("--features gemma4"), "{hint}");
    }

    #[cfg(feature = "qwen3")]
    #[test]
    fn hint_is_silent_for_a_compiled_family() {
        let config = serde_json::json!({"model_type": "qwen3"});
        assert!(feature_gate_hint(&config).is_none());
    }

    #[test]
    fn hint_is_silent_for_an_unknown_family() {
        let config = serde_json::json!({"model_type": "frobnicate_lm"});
        assert!(feature_gate_hint(&config).is_none());
    }
}
