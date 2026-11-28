//! Qwen3.5's [`ModelLine`] implementation.

use std::collections::BTreeSet;

use clap::Args as ClapArgs;
use clap::FromArgMatches;
use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::model_line::CliDecodeOverlap;
use pegainfer_frontend::model_line::CliError;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;
use pegainfer_frontend::model_line::ServePlan;
use pegainfer_frontend::vllm::ParserSelection;

use crate::Qwen35DecodeOverlap;
use crate::Qwen35GdnBackend;
use crate::Qwen35LaunchOptions;
use crate::StepPolicy;

pub static MODEL_LINE: Qwen35Line = Qwen35Line;

pub struct Qwen35Line;

// Qwen3.5-exclusive CLI flags.
#[derive(ClapArgs)]
struct Qwen35Cli {
    /// Per-rank GPU budget in MiB for joint prefix snapshots; zero disables reuse.
    #[arg(long, default_value_t = 0)]
    qwen35_prefix_cache_mib: usize,
    /// Decode-batch capacity, 1..=64. Qwen3.5 internally rounds allocation to
    /// the next graph bucket but admits only this many scheduler slots; defaults
    /// to 64.
    #[arg(long)]
    max_batch: Option<usize>,

    /// GDN prefill backend. The FlashInfer candidate requires a build-linked
    /// artifact and SM120/Hv32/TP1; Triton remains the default.
    #[arg(long, value_enum, default_value_t = Qwen35GdnBackend::Triton)]
    qwen35_gdn_backend: Qwen35GdnBackend,

    /// Bound mixed prefill+decode steps that have active decoders by a fitted
    /// stall target in milliseconds. The built-in coefficients are fitted on a
    /// single-card H20 with Qwen3.5-4B and charge a per-prompt launch cost plus
    /// the attention cost of the context already resident in KV, so a packed
    /// burst of short prompts and a deep continuation are both priced. Pure
    /// prefill steps still let the head prompt use the full
    /// --max-prefill-tokens budget. Unset keeps the fixed --max-prefill-tokens
    /// budget (1024 rows by default); requires --decode-overlap=off.
    #[arg(long)]
    qwen35_prefill_stall_ms: Option<u64>,
}

fn resolve_step_policy(cli: &Qwen35Cli) -> StepPolicy {
    cli.qwen35_prefill_stall_ms
        .map_or(StepPolicy::Fixed, StepPolicy::stall_bounded_ms)
}

fn cli(ctx: &LaunchContext<'_>) -> Qwen35Cli {
    Qwen35Cli::from_arg_matches(ctx.matches).expect("Qwen35Cli parses from the merged command")
}

pub(crate) fn launch_options(ctx: &LaunchContext<'_>) -> Qwen35LaunchOptions {
    let cli = cli(ctx);
    Qwen35LaunchOptions {
        device_ordinal: ctx.shared.device_ordinal,
        tp_size: ctx.shared.tp_size,
        cuda_graph: ctx.shared.cuda_graph,
        gdn_backend: cli.qwen35_gdn_backend,
        prefix_cache_mib: cli.qwen35_prefix_cache_mib,
        max_batch: cli.max_batch.unwrap_or(crate::MAX_DECODE_BATCH),
        max_prefill_tokens: ctx
            .shared
            .max_prefill_tokens
            .unwrap_or(crate::DEFAULT_MAX_PREFILL_TOKENS),
        step_policy: resolve_step_policy(&cli),
    }
}

fn resolve_decode_overlap(overlap: CliDecodeOverlap) -> Result<Qwen35DecodeOverlap, CliError> {
    match overlap {
        CliDecodeOverlap::Off => Ok(Qwen35DecodeOverlap::Off),
        CliDecodeOverlap::Stream => Ok(Qwen35DecodeOverlap::SharedSm),
        CliDecodeOverlap::GreenCtx => Err(CliError::rule(
            "Qwen3.5 supports --decode-overlap=off|stream only",
        )),
    }
}

impl ModelLine for Qwen35Line {
    fn name(&self) -> &'static str {
        "Qwen3.5"
    }

    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
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
            "no_prefix_cache",
            "max_prefill_tokens",
            "decode_overlap",
            "decode_sm_pct",
        ]
    }

    fn validate(
        &self,
        ctx: &LaunchContext<'_>,
        _provided: &BTreeSet<String>,
    ) -> Result<(), CliError> {
        let cli = cli(ctx);
        let decode_overlap = resolve_decode_overlap(ctx.shared.decode_overlap)?;
        if ctx.shared.tp_size > 1 && cli.qwen35_gdn_backend != Qwen35GdnBackend::Triton {
            return Err(CliError::rule(
                "Qwen3.5 --qwen35-gdn-backend=flashinfer-candidate requires TP world_size=1",
            ));
        }
        if cli.qwen35_prefix_cache_mib > 0 && ctx.shared.no_prefix_cache {
            return Err(CliError::rule(
                "--qwen35-prefix-cache-mib and --no-prefix-cache are contradictory",
            ));
        }
        crate::check_launch(
            cli.max_batch.unwrap_or(crate::MAX_DECODE_BATCH),
            ctx.shared
                .max_prefill_tokens
                .unwrap_or(crate::DEFAULT_MAX_PREFILL_TOKENS),
            ctx.shared.tp_size > 1,
            decode_overlap,
            resolve_step_policy(&cli),
        )
        .map_err(|error| CliError::rule(error.to_string()))
    }

    fn serve_plan(&self, _ctx: &LaunchContext<'_>) -> Result<ServePlan, CliError> {
        Ok(ServePlan {
            // Both generations' templates request the Qwen Coder tool syntax,
            // but upstream `Auto` matches the model *path* by substring: a
            // `Qwen3.8-27B` directory resolves to the JSON `qwen3_json` parser
            // (`docs/models/qwen35/support-qwen38.md`). The line knows better
            // than the directory name.
            auto_tool_call_parser: Some(ParserSelection::Explicit("qwen3_coder".to_string())),
            ..Default::default()
        })
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        crate::launch_with_options(
            ctx.model_path,
            launch_options(ctx),
            resolve_decode_overlap(ctx.shared.decode_overlap)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?,
        )
        .map(LaunchedEngine::Stepped)
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

    fn resolved_launch_options(argv: &[&str]) -> Qwen35LaunchOptions {
        let (shared, matches, _) =
            parse_for_line(&MODEL_LINE, argv).expect("the shared flags parse");
        let config = serde_json::json!({});
        let ctx = LaunchContext {
            model_path: std::path::Path::new("unused"),
            config: &config,
            shared: &shared,
            matches: &matches,
        };
        launch_options(&ctx)
    }

    /// The launch default is one standard chunk on both paths; an explicit
    /// `--max-prefill-tokens` still wins.
    #[test]
    fn default_budget_is_one_standard_chunk() {
        assert_eq!(
            resolved_launch_options(&["pegainfer"]).max_prefill_tokens,
            crate::DEFAULT_MAX_PREFILL_TOKENS
        );
        assert_eq!(
            resolved_launch_options(&["pegainfer", "--tp-size", "2", "--cuda-graph=false"])
                .max_prefill_tokens,
            crate::DEFAULT_MAX_PREFILL_TOKENS
        );
        assert_eq!(
            resolved_launch_options(&[
                "pegainfer",
                "--tp-size",
                "2",
                "--cuda-graph=false",
                "--max-prefill-tokens",
                "4096",
            ])
            .max_prefill_tokens,
            4096
        );
    }

    #[test]
    fn default_step_policy_is_fixed_single_and_tp() {
        assert_eq!(
            resolved_launch_options(&["pegainfer"]).step_policy,
            StepPolicy::Fixed
        );
        assert_eq!(
            resolved_launch_options(&["pegainfer", "--tp-size", "2", "--cuda-graph=false"])
                .step_policy,
            StepPolicy::Fixed
        );
    }

    #[test]
    fn parses_prefill_stall_ms_into_stall_bounded_policy() {
        let policy =
            resolved_launch_options(&["pegainfer", "--qwen35-prefill-stall-ms", "60"]).step_policy;
        assert_eq!(
            policy,
            StepPolicy::StallBounded {
                max_stall_us: 60_000,
                cost_model: crate::StepCostModel::H20_QWEN35_4B,
            }
        );
    }

    #[test]
    fn rejects_zero_prefill_stall_ms() {
        let error = validate_argv(&["pegainfer", "--qwen35-prefill-stall-ms", "0"])
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("--qwen35-prefill-stall-ms must be at least 1, got 0"),
            "{error}"
        );
    }

    #[test]
    fn rejects_unreachable_prefill_stall_ms() {
        for stall_ms in ["1", "11"] {
            let error = validate_argv(&["pegainfer", "--qwen35-prefill-stall-ms", stall_ms])
                .unwrap_err()
                .to_string();
            assert!(error.contains("at least 12 ms"), "{error}");
        }
    }

    #[test]
    fn rejects_stall_bound_with_stream_overlap() {
        let error = validate_argv(&[
            "pegainfer",
            "--qwen35-prefill-stall-ms",
            "60",
            "--decode-overlap",
            "stream",
            "--max-batch",
            "32",
        ])
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("cannot be combined with --decode-overlap=stream"),
            "{error}"
        );
    }

    #[test]
    fn rejects_contradictory_prefix_cache_flags() {
        let error = validate_argv(&[
            "pegainfer",
            "--qwen35-prefix-cache-mib",
            "128",
            "--no-prefix-cache",
        ])
        .unwrap_err()
        .to_string();
        assert!(error.contains("contradictory"), "{error}");
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
        validate_argv(&[
            "pegainfer",
            "--tp-size",
            "2",
            "--cuda-graph=false",
            "--qwen35-prefix-cache-mib",
            "128",
        ])
        .expect("Qwen3.5 should accept prefix caching with eager TP startup");
    }

    #[test]
    fn accepts_shared_stream_overlap() {
        let max_batch = crate::MAX_SHARED_SM_DECODE_BATCH.to_string();
        validate_argv(&[
            "pegainfer",
            "--decode-overlap",
            "stream",
            "--max-batch",
            &max_batch,
        ])
        .expect("Qwen3.5 should accept shared-stream overlap");
    }

    #[test]
    fn rejects_stream_overlap_default_max_batch() {
        let error = validate_argv(&["pegainfer", "--decode-overlap", "stream"])
            .expect_err("Qwen3.5 should reject stream overlap at the default max_batch");
        assert!(error.to_string().contains("max-batch"));
        assert!(
            error
                .to_string()
                .contains(&crate::MAX_SHARED_SM_DECODE_BATCH.to_string())
        );
    }

    #[test]
    fn rejects_green_context_overlap() {
        let error = validate_argv(&["pegainfer", "--decode-overlap", "green-ctx"])
            .expect_err("Qwen3.5 should reject green-context overlap");
        assert!(error.to_string().contains("off|stream"));
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

    /// The server resolves `--tool-call-parser` through this plan, so the
    /// line's answer is the parser the served grammar gets: `auto` must become
    /// the Coder parser both generations' templates emit, and an explicit
    /// choice must survive untouched.
    #[test]
    fn auto_tool_call_parser_resolves_to_the_coder_parser() {
        let (shared, matches, _) =
            parse_for_line(&MODEL_LINE, &["pegainfer"]).expect("the shared flags parse");
        let config = serde_json::json!({});
        let ctx = LaunchContext {
            model_path: std::path::Path::new("unused"),
            config: &config,
            shared: &shared,
            matches: &matches,
        };
        let plan = MODEL_LINE.serve_plan(&ctx).expect("the plan builds");

        assert_eq!(
            plan.resolve_tool_call_parser(ParserSelection::Auto),
            ParserSelection::Explicit("qwen3_coder".to_string())
        );
        let explicit = ParserSelection::Explicit("qwen3_xml".to_string());
        assert_eq!(
            plan.resolve_tool_call_parser(explicit.clone()),
            explicit,
            "an explicit name is the caller's to make"
        );
    }
}
