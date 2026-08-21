//! K3's [`ModelLine`] implementation: the config probe, the K3-exclusive CLI
//! section, the frontend serve plan (one scheduler partition per EP rank),
//! and launch.

use std::collections::BTreeSet;

use anyhow::Context as _;
use clap::Args as ClapArgs;
use clap::FromArgMatches;
use log::info;
use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::model_line::CliError;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;
use pegainfer_frontend::model_line::ServePlan;
use serde_json::Value;

use crate::config::K3_HIDDEN;
use crate::config::K3_LAYERS;
use crate::config::K3_SUPPORTED_ROUTED_EXPERTS;
use crate::executor::K3Executor;
use crate::executor::K3ExecutorConfig;
use crate::executor::ep::K3EpRendezvous;
use crate::scheduler::K3SchedulerConfig;

pub static MODEL_LINE: K3Line = K3Line;

pub struct K3Line;

/// Identity strings from the checkpoint's `config.json`. The wrapper config is
/// multimodal, so the language-model fields sit one level down under
/// `text_config` and carry their own (`kimi_linear`) identity.
const MODEL_TYPE: &str = "kimi_k3";
const ARCHITECTURE: &str = "KimiK3ForConditionalGeneration";
const TEXT_MODEL_TYPE: &str = "kimi_linear";

/// EP world sizes the decode topology is defined for. Experts shard whole
/// across ranks, and the fused MegaMoE kernel that carries the routed experts
/// is AOT-instantiated per (expert count, width) world — so this list is the
/// union of widths any checkpoint supports, not a set of arithmetically valid
/// shardings; the exact world is validated at launch against the kernel's own
/// matrix (`k3_mega_world_supported`). Kept independent of the checkpoint on
/// purpose: the frontend needs the partition count from the CLI alone, before
/// any config or weights are read.
const EP_SIZES: &[usize] = &[1, 4, 8, 16, 32, 64];

// K3-exclusive CLI flags.
#[derive(ClapArgs)]
struct K3Cli {
    /// K3 expert-parallel world size: one rank (and one scheduler partition)
    /// per GPU, routed experts split whole across ranks. One of 1, 4, 8, 16,
    /// 32 or 64; widths above 4 span machines and shard per checkpoint (the
    /// 224-expert dev checkpoint stops at 16, the full 896-expert model needs
    /// at least 8).
    #[arg(long, default_value_t = 1)]
    k3_ep_size: usize,

    /// K3 global EP ranks this process hosts, `start..end` (e.g. `4..8`).
    /// Default: the whole world (single-process, one machine). A partial
    /// range is the multi-process cross-machine shape: every machine runs the
    /// same binary over its own ranks, and requires `--k3-rendezvous`.
    #[arg(long)]
    k3_ranks: Option<String>,

    /// K3 bootstrap rendezvous address (`host:port`): the process hosting
    /// rank 0 binds it, collects every rank's NVLink-fabric slab handle, and
    /// serves the world's table back. A one-time handshake — after it the
    /// engines never talk again (the MegaMoE kernel pairs the ranks itself,
    /// over the rack-wide NVLink domain).
    #[arg(long)]
    k3_rendezvous: Option<String>,
}

/// Parse a `start..end` rank range.
fn parse_rank_range(spec: &str) -> Result<std::ops::Range<usize>, String> {
    let (start, end) = spec
        .split_once("..")
        .ok_or_else(|| format!("expected start..end, got {spec:?}"))?;
    let parse = |raw: &str| {
        raw.trim()
            .parse::<usize>()
            .map_err(|_| format!("expected start..end with integer bounds, got {spec:?}"))
    };
    let range = parse(start)?..parse(end)?;
    if range.is_empty() {
        return Err(format!("rank range {spec:?} is empty"));
    }
    Ok(range)
}

fn cli(ctx: &LaunchContext<'_>) -> K3Cli {
    K3Cli::from_arg_matches(ctx.matches).expect("K3Cli parses from the merged command")
}

/// Validate `--k3-ep-size` where it is read, so an unsupported width fails
/// with the real problem instead of surfacing later as a topology mismatch.
fn ep_size(cli: &K3Cli) -> Result<usize, CliError> {
    if EP_SIZES.contains(&cli.k3_ep_size) {
        Ok(cli.k3_ep_size)
    } else {
        Err(CliError::rule(format!(
            "--k3-ep-size must be one of {EP_SIZES:?}, got {}",
            cli.k3_ep_size
        )))
    }
}

/// Read a language-model field: `text_config` first (where this family keeps
/// them), then the top level, so a text-only export probes the same way.
fn text_field<'a>(config: &'a Value, key: &str) -> Option<&'a Value> {
    config
        .get("text_config")
        .and_then(|text| text.get(key))
        .or_else(|| config.get(key))
}

fn expect_usize(config: &Value, key: &str, expected: usize) -> Result<(), String> {
    match text_field(config, key).and_then(Value::as_u64) {
        Some(value) if value == expected as u64 => Ok(()),
        other => Err(format!("{key} {other:?} is not {expected}")),
    }
}

fn claims_architecture(config: &Value) -> bool {
    config
        .get("architectures")
        .and_then(Value::as_array)
        .is_some_and(|list| {
            list.iter()
                .any(|entry| entry.as_str() == Some(ARCHITECTURE))
        })
}

/// End-of-sequence ids the scheduler stops on. The field is a scalar in this
/// family's config; arrays appear in other exports of the same architecture,
/// so accept both.
fn eos_token_ids(config: &Value) -> Vec<u32> {
    let field = config
        .get("eos_token_id")
        .or_else(|| text_field(config, "eos_token_id"));
    match field {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_u64)
            .map(|id| id as u32)
            .collect(),
        Some(value) => value.as_u64().map(|id| id as u32).into_iter().collect(),
        None => Vec::new(),
    }
}

impl ModelLine for K3Line {
    fn name(&self) -> &'static str {
        "K3"
    }

    fn probe(&self, config: &Value) -> Result<(), String> {
        let model_type = config.get("model_type").and_then(Value::as_str);
        if model_type != Some(MODEL_TYPE) {
            return Err(format!("model_type {model_type:?} is not {MODEL_TYPE:?}"));
        }
        if !claims_architecture(config) {
            return Err(format!(
                "architectures {:?} does not contain {ARCHITECTURE:?}",
                config.get("architectures")
            ));
        }
        let text_model_type = text_field(config, "model_type").and_then(Value::as_str);
        if text_model_type != Some(TEXT_MODEL_TYPE) {
            return Err(format!(
                "text_config.model_type {text_model_type:?} is not {TEXT_MODEL_TYPE:?}"
            ));
        }
        // Cheap shape gates against the architecture constants — enough to
        // catch a mislabelled directory. The exhaustive check every field
        // gets is the loader's `config::probe_config_json`, at launch.
        expect_usize(config, "hidden_size", K3_HIDDEN)?;
        expect_usize(config, "num_hidden_layers", K3_LAYERS)?;
        match text_field(config, "num_experts").and_then(Value::as_u64) {
            Some(experts) if K3_SUPPORTED_ROUTED_EXPERTS.contains(&(experts as usize)) => Ok(()),
            other => Err(format!(
                "num_experts {other:?} is not one of {K3_SUPPORTED_ROUTED_EXPERTS:?}"
            )),
        }
    }

    fn augment_cli(&self, cmd: clap::Command) -> clap::Command {
        K3Cli::augment_args(cmd)
    }

    fn consumed_shared_args(&self) -> &'static [&'static str] {
        // `model_path`, `served_model_name` and `port` are core flags every
        // line accepts; listed here too so this is the whole accepted set at
        // a glance. `no_prefix_cache` is accepted and inert: K3's KDA
        // recurrent state is not reconstructible from tokens, so prefix
        // caching is already off.
        &[
            "model_path",
            "served_model_name",
            "port",
            "device_ordinal",
            "no_prefix_cache",
            "dflash_draft_model_path",
        ]
    }

    fn validate(
        &self,
        ctx: &LaunchContext<'_>,
        provided: &BTreeSet<String>,
    ) -> Result<(), CliError> {
        let cli = cli(ctx);
        let ep_size = ep_size(&cli)?;
        if ep_size > 1 && provided.contains("device_ordinal") {
            return Err(CliError::rule(
                "--device-ordinal applies to single-rank K3 only; --k3-ep-size>1 uses devices 0..local_ranks",
            ));
        }
        let ranks = local_ranks(&cli, ep_size)?;
        if ranks.len() < ep_size && cli.k3_rendezvous.is_none() {
            return Err(CliError::rule(
                "--k3-ranks hosts a slice of the EP world, which needs --k3-rendezvous for the \
                 fleet's fabric-handle exchange",
            ));
        }
        if ranks.len() == ep_size && cli.k3_rendezvous.is_some() {
            return Err(CliError::rule(
                "--k3-rendezvous is for a multi-process fleet; this process hosts the whole EP \
                 world already",
            ));
        }
        Ok(())
    }

    fn serve_plan(&self, ctx: &LaunchContext<'_>) -> Result<ServePlan, CliError> {
        let cli = cli(ctx);
        let ep_size = ep_size(&cli)?;
        Ok(ServePlan {
            // One partition per EP rank THIS PROCESS hosts, and CLI-derivable:
            // the frontend registers engine identities before the weights are
            // loaded.
            scheduler_partition_count: local_ranks(&cli, ep_size)?.len(),
            prefill_only: false,
            lora_modules: None,
        })
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        let cli = cli(ctx);
        let ep_size = ep_size(&cli)?;
        let ranks = local_ranks(&cli, ep_size)?;
        let eos_token_ids = eos_token_ids(ctx.config);
        let config = K3ExecutorConfig::default().from_env().for_ep(ep_size);
        info!(
            "K3 engine starting: ep_size={ep_size}, ranks={ranks:?}, \
             eos_token_ids={eos_token_ids:?}, slots={}, ctx={}, layers={}, cuda_graph={}, \
             moe=mega",
            config.max_batch, config.max_ctx, config.num_layers, config.cuda_graph
        );
        // One executor per EP rank this process hosts, on devices
        // 0..ranks.len(). A single-rank run honours --device-ordinal, which
        // `validate` already restricted.
        //
        // Every local rank's weights are resident before any rank is stepped:
        // a rank publishes its symmetric slab at load, but reads the world's
        // table back on its own scheduler thread at its first step — which is
        // after `start_with_executors` below, and so after every load. That
        // ordering is the point: the table read blocks, so one rank running
        // out of memory mid-load must not be able to strand its peers waiting.
        // Across a fleet the same story holds per process, and the bootstrap's
        // long timeouts absorb the machines' load skew.
        let rendezvous = match (&cli.k3_rendezvous, ep_size > 1) {
            (_, false) => None,
            (None, true) => Some(K3EpRendezvous::new(ep_size)),
            (Some(addr), true) => {
                Some(K3EpRendezvous::fleet(ep_size, ranks.clone(), addr.clone())?)
            }
        };
        let mut executors = Vec::with_capacity(ranks.len());
        for (device, rank) in ranks.clone().enumerate() {
            let device = if ep_size == 1 {
                ctx.shared.device_ordinal
            } else {
                device
            };
            let executor = match rendezvous.clone() {
                Some(rendezvous) => {
                    K3Executor::load_ep(ctx.model_path, device, rank, config, rendezvous)
                }
                None => K3Executor::load(ctx.model_path, device, rank, ep_size, config),
            };
            let mut executor = executor.with_context(|| {
                format!("loading K3 rank {rank} of {ep_size} onto device {device}")
            })?;
            // The DSpark draft lane is rank-local and collective-free, so
            // arming it per rank adds no cross-rank coupling.
            if let Some(draft_path) = &ctx.shared.dflash_draft_model_path {
                executor
                    .load_dspark(draft_path)
                    .with_context(|| format!("arming the K3 dspark draft lane on rank {rank}"))?;
            }
            executors.push(executor);
        }
        Ok(LaunchedEngine::Stepped(
            crate::scheduler::start_with_executors(
                executors,
                &K3SchedulerConfig {
                    eos_token_ids,
                    kv_capacity: None,
                },
            ),
        ))
    }
}

/// The global ranks this process hosts: `--k3-ranks` when given, the whole
/// world otherwise. Validated where it is read, like the EP width.
fn local_ranks(cli: &K3Cli, ep_size: usize) -> Result<std::ops::Range<usize>, CliError> {
    let Some(spec) = &cli.k3_ranks else {
        return Ok(0..ep_size);
    };
    if ep_size == 1 {
        return Err(CliError::rule(
            "--k3-ranks partitions an EP world; it needs --k3-ep-size > 1",
        ));
    }
    let ranks =
        parse_rank_range(spec).map_err(|err| CliError::rule(format!("--k3-ranks: {err}")))?;
    if ranks.end > ep_size {
        return Err(CliError::rule(format!(
            "--k3-ranks {spec} does not fit an EP world of {ep_size}"
        )));
    }
    Ok(ranks)
}

#[cfg(test)]
mod tests {
    use pegainfer_frontend::model_line::parse_for_line;

    use super::*;

    /// A config shaped like the checkpoint's: multimodal wrapper on the
    /// outside, language-model fields under `text_config`.
    fn reference_config(experts: usize) -> Value {
        serde_json::json!({
            "model_type": MODEL_TYPE,
            "architectures": [ARCHITECTURE],
            "eos_token_id": 163_586,
            "text_config": {
                "model_type": TEXT_MODEL_TYPE,
                "architectures": ["KimiLinearForCausalLM"],
                "hidden_size": K3_HIDDEN,
                "num_hidden_layers": K3_LAYERS,
                "num_experts": experts,
                "num_experts_per_token": 16,
            },
        })
    }

    fn plan_for(argv: &[&str]) -> Result<ServePlan, CliError> {
        let (shared, matches, provided) =
            parse_for_line(&MODEL_LINE, argv).map_err(|error| CliError::rule(error.to_string()))?;
        let config = reference_config(224);
        let ctx = LaunchContext {
            model_path: std::path::Path::new("unused"),
            config: &config,
            shared: &shared,
            matches: &matches,
        };
        MODEL_LINE.validate(&ctx, &provided)?;
        MODEL_LINE.serve_plan(&ctx)
    }

    /// The scheduler partition count a CLI produces, or the rule it broke.
    /// (`ServePlan` is not `Debug`, so error assertions go through this.)
    fn partitions_for(argv: &[&str]) -> Result<usize, CliError> {
        plan_for(argv).map(|plan| plan.scheduler_partition_count)
    }

    #[test]
    fn probe_accepts_both_expert_counts() {
        for experts in K3_SUPPORTED_ROUTED_EXPERTS {
            MODEL_LINE
                .probe(&reference_config(experts))
                .unwrap_or_else(|error| {
                    panic!("K3 should claim a {experts}-expert config: {error}")
                });
        }
    }

    #[test]
    fn probe_rejects_another_family() {
        let error = MODEL_LINE
            .probe(&serde_json::json!({"model_type": "kimi_k2"}))
            .expect_err("K3 must not claim another family's config");
        assert!(error.contains("kimi_k3"), "{error}");
    }

    #[test]
    fn probe_rejects_a_mismatched_shape() {
        let mut config = reference_config(224);
        config["text_config"]["hidden_size"] = serde_json::json!(4096);
        let error = MODEL_LINE
            .probe(&config)
            .expect_err("a K3-labelled config with the wrong hidden size must be refused");
        assert!(error.contains("hidden_size"), "{error}");

        let mut config = reference_config(224);
        config["text_config"]["num_experts"] = serde_json::json!(64);
        let error = MODEL_LINE
            .probe(&config)
            .expect_err("an unsupported expert count must be refused");
        assert!(error.contains("num_experts"), "{error}");
    }

    #[test]
    fn probe_rejects_a_missing_architecture_claim() {
        let mut config = reference_config(224);
        config["architectures"] = serde_json::json!(["SomethingElseForCausalLM"]);
        let error = MODEL_LINE
            .probe(&config)
            .expect_err("the architecture claim is part of the identity");
        assert!(error.contains("architectures"), "{error}");
    }

    #[test]
    fn eos_ids_come_from_either_spelling() {
        assert_eq!(eos_token_ids(&reference_config(224)), vec![163_586]);
        assert_eq!(
            eos_token_ids(&serde_json::json!({"eos_token_id": [1, 2]})),
            vec![1, 2]
        );
        assert!(eos_token_ids(&serde_json::json!({})).is_empty());
    }

    #[test]
    fn serve_plan_defaults_to_one_partition() {
        let plan = plan_for(&["pegainfer"]).expect("defaults should validate");
        assert_eq!(plan.scheduler_partition_count, 1);
        assert!(!plan.prefill_only && plan.lora_modules.is_none());
    }

    #[test]
    fn serve_plan_counts_one_partition_per_ep_rank() {
        for ep_size in EP_SIZES {
            let partitions = partitions_for(&["pegainfer", "--k3-ep-size", &ep_size.to_string()])
                .expect("a supported EP width should validate");
            assert_eq!(partitions, *ep_size);
        }
    }

    #[test]
    fn rejects_an_unsupported_ep_size() {
        let error = partitions_for(&["pegainfer", "--k3-ep-size", "6"])
            .expect_err("EP6 has no MegaMoE instantiation for any checkpoint");
        assert!(
            error.to_string().contains("--k3-ep-size must be one of"),
            "{error}"
        );
    }

    #[test]
    fn a_partial_fleet_counts_its_own_ranks() {
        let partitions = partitions_for(&[
            "pegainfer",
            "--k3-ep-size",
            "16",
            "--k3-ranks",
            "4..8",
            "--k3-rendezvous",
            "10.0.0.1:19300",
        ])
        .expect("a partial fleet slice should validate");
        assert_eq!(partitions, 4);
    }

    #[test]
    fn a_partial_fleet_requires_the_rendezvous() {
        let error = partitions_for(&["pegainfer", "--k3-ep-size", "16", "--k3-ranks", "4..8"])
            .expect_err("a fleet slice cannot pair without the bootstrap");
        assert!(error.to_string().contains("--k3-rendezvous"), "{error}");
    }

    #[test]
    fn a_whole_world_refuses_a_rendezvous() {
        let error = partitions_for(&[
            "pegainfer",
            "--k3-ep-size",
            "4",
            "--k3-rendezvous",
            "10.0.0.1:19300",
        ])
        .expect_err("an in-process world has nobody to exchange handles with");
        assert!(error.to_string().contains("--k3-rendezvous"), "{error}");
    }

    #[test]
    fn rank_ranges_are_validated() {
        let error = partitions_for(&["pegainfer", "--k3-ranks", "0..2"])
            .expect_err("--k3-ranks without a wider world is meaningless");
        assert!(error.to_string().contains("--k3-ep-size"), "{error}");

        let error = partitions_for(&[
            "pegainfer",
            "--k3-ep-size",
            "8",
            "--k3-ranks",
            "6..10",
            "--k3-rendezvous",
            "10.0.0.1:19300",
        ])
        .expect_err("ranks past the world must be refused");
        assert!(error.to_string().contains("does not fit"), "{error}");

        let error = partitions_for(&[
            "pegainfer",
            "--k3-ep-size",
            "8",
            "--k3-ranks",
            "4",
            "--k3-rendezvous",
            "10.0.0.1:19300",
        ])
        .expect_err("a bound without .. must be refused");
        assert!(error.to_string().contains("start..end"), "{error}");
    }

    #[test]
    fn accepts_a_device_ordinal_only_for_a_single_rank() {
        partitions_for(&["pegainfer", "--device-ordinal", "2"])
            .expect("a single-rank K3 run picks its GPU");
        let error = partitions_for(&["pegainfer", "--device-ordinal", "2", "--k3-ep-size", "4"])
            .expect_err("--device-ordinal is meaningless across EP ranks");
        assert!(error.to_string().contains("--device-ordinal"), "{error}");
    }

    #[test]
    fn rejects_a_shared_flag_this_line_does_not_read() {
        let error = partitions_for(&["pegainfer", "--kv-offload"])
            .expect_err("K3 serves no KV offload tier");
        assert!(error.to_string().contains("--kv-offload"), "{error}");
    }
}
