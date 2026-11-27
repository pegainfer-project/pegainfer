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

pub static MODEL_LINE: Qwen35Line = Qwen35Line;

pub struct Qwen35Line;

// Qwen3.5-exclusive CLI flags.
#[derive(ClapArgs)]
struct Qwen35Cli {
    /// Per-rank GPU budget in MiB for joint prefix snapshots; zero disables reuse.
    #[arg(long, default_value_t = 0)]
    qwen35_prefix_cache_mib: usize,
    /// Per-step decode width limit (1..=64, default 64).
    #[arg(long)]
    decode_width: Option<usize>,
    /// Max resident requests per instance (single-GPU default 128 = 2 x decode_width; TP default 64, single cohort only).
    #[arg(long)]
    max_resident: Option<usize>,

    /// GDN prefill backend. The FlashInfer candidate requires a build-linked
    /// artifact and SM120/Hv32/TP1; Triton remains the default.
    #[arg(long, value_enum, default_value_t = Qwen35GdnBackend::Triton)]
    qwen35_gdn_backend: Qwen35GdnBackend,
}

fn cli(ctx: &LaunchContext<'_>) -> Qwen35Cli {
    Qwen35Cli::from_arg_matches(ctx.matches).expect("Qwen35Cli parses from the merged command")
}

fn resolve_capacity_args(cli: &Qwen35Cli, tp_size: usize) -> (usize, usize) {
    let decode_width = cli.decode_width.unwrap_or(crate::DEFAULT_DECODE_WIDTH);
    let max_resident = cli
        .max_resident
        .unwrap_or_else(|| crate::ServingCapacity::default_max_resident(decode_width, tp_size));
    (decode_width, max_resident)
}

pub(crate) fn launch_options(ctx: &LaunchContext<'_>) -> Qwen35LaunchOptions {
    let cli = cli(ctx);
    let (decode_width, max_resident) = resolve_capacity_args(&cli, ctx.shared.tp_size);
    Qwen35LaunchOptions {
        device_ordinal: ctx.shared.device_ordinal,
        tp_size: ctx.shared.tp_size,
        cuda_graph: ctx.shared.cuda_graph,
        gdn_backend: cli.qwen35_gdn_backend,
        prefix_cache_mib: cli.qwen35_prefix_cache_mib,
        decode_width,
        max_resident,
        max_prefill_tokens: ctx
            .shared
            .max_prefill_tokens
            .unwrap_or(crate::DEFAULT_MAX_PREFILL_TOKENS),
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
        let (decode_width, max_resident) = resolve_capacity_args(&cli, ctx.shared.tp_size);
        crate::check_launch(
            decode_width,
            max_resident,
            ctx.shared
                .max_prefill_tokens
                .unwrap_or(crate::DEFAULT_MAX_PREFILL_TOKENS),
            ctx.shared.tp_size,
            decode_overlap,
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
    fn default_capacity_is_128_single_gpu_and_64_tp2() {
        let single = resolved_launch_options(&["pegainfer"]);
        assert_eq!(single.decode_width, 64);
        assert_eq!(single.max_resident, 128);

        let tp2 = resolved_launch_options(&["pegainfer", "--tp-size", "2", "--cuda-graph=false"]);
        assert_eq!(tp2.decode_width, 64);
        assert_eq!(tp2.max_resident, 64);
    }

    #[test]
    fn accepts_shared_stream_overlap() {
        let width = crate::MAX_SHARED_SM_DECODE_BATCH.to_string();
        validate_argv(&[
            "pegainfer",
            "--decode-overlap",
            "stream",
            "--decode-width",
            &width,
        ])
        .expect("Qwen3.5 should accept shared-stream overlap");
    }

    #[test]
    fn rejects_stream_overlap_default_decode_width() {
        let error = validate_argv(&["pegainfer", "--decode-overlap", "stream"])
            .expect_err("Qwen3.5 should reject stream overlap at the default decode_width");
        assert!(error.to_string().contains("decode-width"));
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
    fn accepts_non_bucket_decode_width_and_resident_override() {
        validate_argv(&["pegainfer", "--decode-width", "5", "--max-resident", "10"])
            .expect("Qwen3.5 should accept decode_width between buckets with 2 cohorts");
    }

    #[test]
    fn rejects_zero_decode_width_and_tp2_dual_cohort() {
        let error = validate_argv(&["pegainfer", "--decode-width", "0"])
            .expect_err("Qwen3.5 should reject zero decode_width");
        assert!(
            error.to_string().contains("decode_width must be in 1..="),
            "unexpected error: {error}"
        );

        let tp2_err = validate_argv(&[
            "pegainfer",
            "--tp-size",
            "2",
            "--cuda-graph=false",
            "--max-resident",
            "128",
        ])
        .expect_err("TP2 must reject max_resident > decode_width");
        assert!(
            tp2_err.to_string().contains("single cohort"),
            "unexpected error: {tp2_err}"
        );
    }

    #[test]
    fn rejects_legacy_max_batch_flag() {
        assert!(
            validate_argv(&["pegainfer", "--max-batch", "64"]).is_err(),
            "legacy --max-batch flag must no longer be accepted"
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
