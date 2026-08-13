//! Qwen3.5's [`ModelLine`] implementation.

use std::collections::BTreeSet;

use clap::Args as ClapArgs;
use clap::FromArgMatches;
use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::model_line::CliError;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;

use crate::Qwen35LaunchOptions;
use crate::Qwen35SchedulerPolicy;

pub static MODEL_LINE: Qwen35Line = Qwen35Line;

pub struct Qwen35Line;

// Qwen3.5-exclusive CLI flags.
#[derive(ClapArgs)]
struct Qwen35Cli {
    /// Decode-batch capacity, 1..=64. Qwen3.5 internally rounds allocation to
    /// the next graph bucket but admits only this many scheduler slots; defaults
    /// to 64.
    #[arg(long)]
    max_batch: Option<usize>,

    /// Qwen3.5 prefill/decode scheduler policy. Defaults to `off`; `auto` is
    /// opt-in and currently single-GPU only.
    #[arg(long, value_enum, default_value_t = CliQwen35SchedulerPolicy::Off)]
    qwen35_scheduler_policy: CliQwen35SchedulerPolicy,
}

/// CLI selector for the Qwen3.5 adaptive scheduler policy.
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum CliQwen35SchedulerPolicy {
    /// Fixed chunked-prefill behavior.
    #[default]
    Off,
    /// Runtime-state adaptive policy.
    Auto,
}

impl CliQwen35SchedulerPolicy {
    fn resolve(self) -> Qwen35SchedulerPolicy {
        match self {
            Self::Off => Qwen35SchedulerPolicy::Off,
            Self::Auto => Qwen35SchedulerPolicy::Auto,
        }
    }
}

fn cli(ctx: &LaunchContext<'_>) -> Qwen35Cli {
    Qwen35Cli::from_arg_matches(ctx.matches).expect("Qwen35Cli parses from the merged command")
}

impl ModelLine for Qwen35Line {
    fn name(&self) -> &'static str {
        "Qwen3.5"
    }

    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
        let text_model_type = config
            .get("text_config")
            .and_then(|text| text.get("model_type"))
            .and_then(serde_json::Value::as_str);
        if model_type != Some("qwen3_5") && text_model_type != Some("qwen3_5_text") {
            return Err(format!("model_type {model_type:?} is not \"qwen3_5\""));
        }
        crate::probe_config_json(config).map_err(|error| error.to_string())
    }

    fn augment_cli(&self, cmd: clap::Command) -> clap::Command {
        Qwen35Cli::augment_args(cmd)
    }

    fn consumed_shared_args(&self) -> &'static [&'static str] {
        &[
            "device_ordinal",
            "tp_size",
            "cuda_graph",
            "max_prefill_tokens",
        ]
    }

    fn validate(
        &self,
        ctx: &LaunchContext<'_>,
        _provided: &BTreeSet<String>,
    ) -> Result<(), CliError> {
        let cli = cli(ctx);
        if let Some(max_batch) = cli.max_batch {
            if !(1..=crate::MAX_DECODE_BATCH).contains(&max_batch) {
                return Err(CliError::rule(format!(
                    "--max-batch must be in 1..={} for Qwen3.5, got {max_batch}",
                    crate::MAX_DECODE_BATCH
                )));
            }
        }
        if ctx.shared.tp_size > 1
            && matches!(cli.qwen35_scheduler_policy, CliQwen35SchedulerPolicy::Auto)
        {
            return Err(CliError::rule(
                "--qwen35-scheduler-policy=auto is single-GPU only; Qwen3.5 TP uses the fixed off policy",
            ));
        }
        Ok(())
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        let cli = cli(ctx);
        crate::launch_with_options_and_policy(
            ctx.model_path,
            Qwen35LaunchOptions {
                device_ordinal: ctx.shared.device_ordinal,
                tp_size: ctx.shared.tp_size,
                cuda_graph: ctx.shared.cuda_graph,
                max_batch: cli.max_batch.unwrap_or(crate::MAX_DECODE_BATCH),
                max_prefill_tokens: ctx
                    .shared
                    .max_prefill_tokens
                    .unwrap_or(crate::DEFAULT_MAX_PREFILL_TOKENS),
            },
            cli.qwen35_scheduler_policy.resolve(),
        )
        .map(LaunchedEngine::Handle)
    }
}

#[cfg(test)]
mod tests {
    use pegainfer_frontend::model_line::parse_for_line;

    use super::*;

    fn validate_argv(argv: &[&str]) -> Result<(), CliError> {
        let (shared, matches, provided) =
            parse_for_line(&MODEL_LINE, argv).map_err(|error| CliError::rule(error.to_string()))?;
        let config = serde_json::json!({});
        let ctx = LaunchContext {
            model_path: std::path::Path::new("unused"),
            config: &config,
            shared: &shared,
            matches: &matches,
        };
        MODEL_LINE.validate(&ctx, &provided)
    }

    #[test]
    fn probe_accepts_qwen35_identity() {
        let json = serde_json::json!({
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "text_config": {"model_type": "qwen3_5_text"}
        });
        MODEL_LINE
            .probe(&json)
            .expect("qwen3_5 config should probe");
    }

    #[test]
    fn accepts_tp_size() {
        validate_argv(&["pegainfer", "--tp-size", "2", "--cuda-graph=false"])
            .expect("Qwen3.5 should accept --tp-size for eager TP startup");
    }

    #[test]
    fn accepts_scheduler_policy_off() {
        validate_argv(&["pegainfer", "--qwen35-scheduler-policy", "off"])
            .expect("Qwen3.5 should accept explicit scheduler-policy off");
    }

    #[test]
    fn rejects_tp_auto_scheduler_policy() {
        let error = validate_argv(&[
            "pegainfer",
            "--tp-size",
            "2",
            "--cuda-graph=false",
            "--qwen35-scheduler-policy",
            "auto",
        ])
        .expect_err("Qwen3.5 TP should reject auto scheduler-policy");
        assert!(error.to_string().contains("single-GPU only"));
    }

    #[test]
    fn accepts_non_bucket_scheduler_max_batch() {
        validate_argv(&["pegainfer", "--max-batch", "5"])
            .expect("Qwen3.5 should accept scheduler max_batch between decode buckets");
    }

    #[test]
    fn rejects_zero_scheduler_max_batch() {
        let error = validate_argv(&["pegainfer", "--max-batch", "0"])
            .expect_err("Qwen3.5 should reject zero scheduler max_batch");
        assert!(
            error.to_string().contains("--max-batch must be in 1..="),
            "unexpected error: {error}"
        );
    }
}
