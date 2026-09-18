use std::path::PathBuf;
use std::sync::Once;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use clap::Parser;
use pegainfer_sim::SimulatedEngineConfig;
use pegainfer_sim::profile::EngineProfile;
use pegainfer_sim::profile::LoadedEngineProfile;
use pegainfer_sim::start_engine;

const DEFAULT_MODEL_ID: &str = "Qwen/Qwen3-0.6B";
const DEFAULT_MAX_MODEL_LEN: u32 = 8192;
const DEFAULT_BASE_TTFT_MS: f64 = 5.0;
const DEFAULT_PREFILL_TOKENS_PER_MS: f64 = 100.0;
const DEFAULT_TPOT_MS: f64 = 12.0;
const DEFAULT_FALLBACK_TOKEN_ID: u32 = 0;

static LOGGING_INIT: Once = Once::new();

#[derive(Parser, Debug)]
#[command(
    name = "pegainfer-sim",
    about = "CPU-only simulated inference server for OpenAI/vLLM serving benchmarks"
)]
struct Args {
    /// Model identity. In legacy mode it also remains the metadata path; with
    /// --profile it must match the profile's target model id.
    #[arg(long)]
    model_id: Option<String>,

    /// Local tokenizer/model metadata directory used by the vLLM frontend.
    /// With --profile, this can differ from the profile's target model id.
    #[arg(long, value_name = "PATH")]
    model_path: Option<PathBuf>,

    /// Port to listen on.
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Max context length reported to the vLLM frontend.
    #[arg(long)]
    max_model_len: Option<u32>,

    /// Fixed TTFT floor before the first fake token.
    #[arg(long)]
    base_ttft_ms: Option<f64>,

    /// Simulated prefill throughput used as prompt_len / throughput.
    #[arg(long)]
    prefill_tokens_per_ms: Option<f64>,

    /// Fixed delay between generated fake tokens.
    #[arg(long)]
    tpot_ms: Option<f64>,

    /// Token id used when a request has an empty prompt-token list.
    #[arg(long, default_value_t = DEFAULT_FALLBACK_TOKEN_ID)]
    fallback_token_id: u32,

    /// Versioned timing and scheduler profile generated from a target engine.
    #[arg(long, value_name = "FILE")]
    profile: Option<PathBuf>,
}

#[derive(Debug)]
struct RuntimeConfig {
    engine: SimulatedEngineConfig,
    model_path: PathBuf,
    served_model_name: Vec<String>,
    max_model_len: u32,
    profile: Option<LoadedEngineProfile>,
}

fn build_runtime(args: &Args) -> Result<RuntimeConfig> {
    if let Some(path) = &args.profile {
        ensure_legacy_timing_flags_are_absent(args)?;
        let profile = EngineProfile::load_from_path(path)
            .with_context(|| format!("failed to load engine profile {}", path.display()))?;
        let model_id = profile.model_id().to_string();
        if let Some(requested) = &args.model_id {
            ensure!(
                requested == profile.model_id(),
                "--model-id '{}' conflicts with profile model_id '{}'",
                requested,
                profile.model_id()
            );
        }
        if let Some(requested) = args.max_model_len {
            ensure!(
                requested == profile.scheduler.max_model_len,
                "--max-model-len {} conflicts with profile max_model_len {}",
                requested,
                profile.scheduler.max_model_len
            );
        }
        let engine = SimulatedEngineConfig::default()
            .with_fallback_token_id(args.fallback_token_id)
            .with_engine_profile(profile.clone())?;
        let model_path = args.model_path.clone().unwrap_or_else(|| {
            args.model_id
                .as_deref()
                .map_or_else(|| PathBuf::from(&model_id), PathBuf::from)
        });
        return Ok(RuntimeConfig {
            engine,
            model_path,
            served_model_name: vec![model_id],
            max_model_len: profile.scheduler.max_model_len,
            profile: Some(profile),
        });
    }
    let model_id = args
        .model_id
        .clone()
        .or_else(|| {
            args.model_path
                .as_deref()
                .map(|path| path.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| DEFAULT_MODEL_ID.to_string());
    let model_path = args
        .model_path
        .clone()
        .unwrap_or_else(|| PathBuf::from(&model_id));
    let max_model_len = args.max_model_len.unwrap_or(DEFAULT_MAX_MODEL_LEN);
    ensure!(max_model_len > 0, "max_model_len must be positive");
    let base_ttft_ms = args.base_ttft_ms.unwrap_or(DEFAULT_BASE_TTFT_MS);
    let prefill_tokens_per_ms = args
        .prefill_tokens_per_ms
        .unwrap_or(DEFAULT_PREFILL_TOKENS_PER_MS);
    let tpot_ms = args.tpot_ms.unwrap_or(DEFAULT_TPOT_MS);
    let engine = SimulatedEngineConfig::new(
        base_ttft_ms,
        prefill_tokens_per_ms,
        tpot_ms,
        args.fallback_token_id,
    )?;
    Ok(RuntimeConfig {
        engine,
        model_path,
        served_model_name: if args.model_path.is_some() && args.model_id.is_some() {
            vec![model_id]
        } else {
            Vec::new()
        },
        max_model_len,
        profile: None,
    })
}

fn ensure_legacy_timing_flags_are_absent(args: &Args) -> Result<()> {
    let provided = [
        ("--base-ttft-ms", args.base_ttft_ms.is_some()),
        (
            "--prefill-tokens-per-ms",
            args.prefill_tokens_per_ms.is_some(),
        ),
        ("--tpot-ms", args.tpot_ms.is_some()),
    ];
    if let Some((name, true)) = provided.into_iter().find(|(_, present)| *present) {
        bail!("{name} cannot be combined with --profile; timing comes from the profile");
    }
    Ok(())
}

fn report_profile(runtime: &RuntimeConfig) {
    let Some(profile) = &runtime.profile else {
        eprintln!("active engine profile: legacy fixed TTFT/TPOT scheduler");
        return;
    };
    let scheduler = &profile.scheduler;
    eprintln!(
        "active engine profile: path={} manifest={} manifest_sha256={} target={} version={} model={} revision={} gpu={} scheduler={:?} max_num_seqs={} max_num_batched_tokens={} max_model_len={} predictor_coverage={:?}",
        profile.profile_path.display(),
        profile.manifest_path.display(),
        profile.calibration.sha256,
        profile.manifest.target_engine,
        profile.manifest.engine_version,
        profile.manifest.model_id,
        profile.manifest.model_revision,
        profile.manifest.gpu,
        scheduler.policy,
        scheduler.max_num_seqs,
        scheduler.max_num_batched_tokens,
        scheduler.max_model_len,
        profile.predictor.coverage.domain(),
    );
}

fn init_logging() {
    LOGGING_INIT.call_once(|| {
        let filter_spec = std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string());
        let filter = logforth::filter::env_filter::EnvFilterBuilder::from_spec(filter_spec).build();
        logforth::starter_log::builder()
            .dispatch(|dispatch| {
                dispatch
                    .filter(filter)
                    .append(logforth::append::Stderr::default())
            })
            .apply();
    });
}

#[tokio::main]
async fn main() -> Result<()> {
    init_logging();
    let args = Args::parse();
    let runtime = build_runtime(&args)?;
    report_profile(&runtime);
    let engine = start_engine(&runtime.engine);

    pegainfer_frontend::vllm::serve(
        std::future::ready(Ok(engine.into())),
        &runtime.model_path,
        runtime.served_model_name,
        args.port,
        Some(runtime.max_model_len),
        pegainfer_frontend::vllm::shutdown_token_from_ctrl_c(),
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn legacy_args() -> Args {
        Args {
            model_id: Some("test-model".to_string()),
            model_path: None,
            port: 8000,
            max_model_len: None,
            base_ttft_ms: None,
            prefill_tokens_per_ms: None,
            tpot_ms: None,
            fallback_token_id: DEFAULT_FALLBACK_TOKEN_ID,
            profile: None,
        }
    }

    #[test]
    fn legacy_cli_keeps_fixed_timing_scheduler() {
        let runtime = build_runtime(&legacy_args()).expect("legacy runtime should build");

        assert!(runtime.profile.is_none());
    }

    #[test]
    fn sim_logger_accepts_warn_records() {
        init_logging();

        assert!(log::log_enabled!(target: "pegainfer_sim::profile", log::Level::Warn));
    }
}
