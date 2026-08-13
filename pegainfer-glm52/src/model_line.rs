//! GLM5.2's [`ModelLine`] implementation: topology-aware validation, the
//! GLM5.2-exclusive CLI section, the frontend serve plan (partition count,
//! prefill-only), and launch-option assembly.

use std::collections::BTreeSet;

use clap::Args as ClapArgs;
use clap::FromArgMatches;
use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::model_line::CliError;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;
use pegainfer_frontend::model_line::ServePlan;

use crate::Glm52Drafter;
use crate::Glm52KvOffloadOptions;
use crate::Glm52LaunchOptions;
use crate::Glm52MoeTopo;
use crate::Glm52P2pOptions;
use crate::Glm52PrefillOnlyOptions;

pub static MODEL_LINE: Glm52Line = Glm52Line;

pub struct Glm52Line;

// GLM5.2-exclusive CLI flags.
#[derive(ClapArgs)]
struct Glm52Cli {
    /// Per-request context cap: prompt + max_tokens - 1 must fit. When
    /// omitted, GLM5.2 sizes it from post-weight-load free VRAM.
    #[arg(long)]
    max_model_len: Option<usize>,

    /// Run GLM5.2 TP4 with prefix caching and no decode.
    #[arg(long, default_value_t = false)]
    glm52_prefill_only: bool,

    /// Token rows per prefill chunk. Must be a multiple of 64.
    #[arg(long, default_value_t = 16_384)]
    glm52_prefill_chunk_size: usize,

    /// GLM5.2 launch-time MoE sharding topology: `ep8` (default) is the
    /// high-throughput configuration (32 whole experts per rank, DeepEP
    /// dispatch/combine, buckets 1-8); `ep4` is its four-GPU counterpart
    /// (64 whole experts per rank, weight-only expert GEMMs — the GB300
    /// high-throughput topology); `tp4` is GB300 **prefill-only** tensor
    /// parallel (requires `--glm52-prefill-only`; NCCL all-reduce). Hopper
    /// (SM9x) and decode TP/LL paths are not supported.
    #[arg(long, default_value = "ep8")]
    moe_topo: String,

    /// Use GLM5.2's checkpoint-native MTP layer as the speculative drafter.
    #[arg(long = "glm52-native-mtp")]
    glm52_native_mtp: bool,

    /// Stage GLM5.2 checkpoint bytes through pinned double buffers. This can
    /// substantially accelerate warm-page-cache loads; leave off for cold
    /// network-filesystem starts.
    #[arg(long)]
    glm52_weight_staging: bool,

    /// GLM5.2 global DP ranks this process hosts, `start..end` (e.g. `4..8`).
    /// Default: the whole topology (single-node). A partial range is the
    /// multi-process cross-node shape: every node runs the same binary over
    /// its own ranks, and requires `--glm52-rendezvous`.
    #[arg(long)]
    glm52_ranks: Option<String>,

    /// GLM5.2 bootstrap rendezvous address (`host:port`): the process hosting
    /// rank 0 binds it and serves the DeepEP unique id; every other process
    /// connects to fetch it. A one-time handshake — no runtime control plane.
    #[arg(long)]
    glm52_rendezvous: Option<String>,
}

fn cli(ctx: &LaunchContext<'_>) -> Glm52Cli {
    Glm52Cli::from_arg_matches(ctx.matches).expect("Glm52Cli parses from the merged command")
}

/// Parse `--moe-topo` with the accepted strings owned by the model crate, so
/// an invalid value fails with the real problem instead of a misleading
/// dp/tp complaint.
fn moe_topo(cli: &Glm52Cli) -> Result<Glm52MoeTopo, CliError> {
    cli.moe_topo
        .parse()
        .map_err(|err| CliError::rule(format!("--moe-topo: {err}")))
}

impl ModelLine for Glm52Line {
    fn name(&self) -> &'static str {
        "GLM5.2"
    }

    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
        if model_type != Some("glm_moe_dsa") {
            return Err(format!("model_type {model_type:?} is not \"glm_moe_dsa\""));
        }
        crate::probe_config_json(config).map_err(|error| error.to_string())
    }

    fn augment_cli(&self, cmd: clap::Command) -> clap::Command {
        Glm52Cli::augment_args(cmd)
    }

    fn consumed_shared_args(&self) -> &'static [&'static str] {
        &[
            "tp_size",
            "dp_size",
            "dflash_draft_model_path",
            "no_prefix_cache",
            "kv_offload",
            "kv_offload_host_gib",
            "kv_offload_hugepages",
            "kv_p2p_metaserver_addr",
            "kv_p2p_advertise_addr",
            "kv_p2p_nics",
            "dump_graph_png",
        ]
    }

    fn validate(
        &self,
        ctx: &LaunchContext<'_>,
        provided: &BTreeSet<String>,
    ) -> Result<(), CliError> {
        let cli = cli(ctx);
        let shared = ctx.shared;
        let moe_topo = moe_topo(&cli)?;
        if let Some(dp_size) = shared.dp_size {
            let expected_dp_size = moe_topo.default_dp_size();
            if dp_size != expected_dp_size {
                return Err(CliError::rule(format!(
                    "GLM5.2 --moe-topo={} requires --dp-size={expected_dp_size} when provided; omit --dp-size to use the topology default",
                    cli.moe_topo
                )));
            }
        }
        let expected_tp_size = moe_topo.expected_tp_size();
        if shared.tp_size != expected_tp_size {
            return Err(CliError::rule(format!(
                "GLM5.2 --moe-topo={} requires --tp-size={expected_tp_size}, got {}",
                cli.moe_topo, shared.tp_size
            )));
        }
        if cli.glm52_prefill_only {
            if !matches!(moe_topo, Glm52MoeTopo::Tp4) {
                return Err(CliError::rule(
                    "--glm52-prefill-only requires --moe-topo=tp4",
                ));
            }
            if shared.no_prefix_cache {
                return Err(CliError::rule(
                    "--glm52-prefill-only requires prefix caching; drop --no-prefix-cache",
                ));
            }
            if shared.dflash_draft_model_path.is_some() {
                return Err(CliError::rule(
                    "--glm52-prefill-only is incompatible with the DSpark drafter",
                ));
            }
            if shared.kv_offload && !cli.glm52_native_mtp {
                return Err(CliError::rule(
                    "--glm52-prefill-only KV offload requires --glm52-native-mtp",
                ));
            }
            if shared.dump_graph_png.is_some() {
                return Err(CliError::rule(
                    "--glm52-prefill-only does not expose a decode CUDA graph",
                ));
            }
        } else if provided.contains("glm52_prefill_chunk_size") {
            return Err(CliError::rule(
                "--glm52-prefill-chunk-size requires --glm52-prefill-only",
            ));
        }
        if cli.glm52_prefill_chunk_size == 0
            || !cli
                .glm52_prefill_chunk_size
                .is_multiple_of(crate::GLM52_PREFILL_CHUNK_ALIGN)
        {
            return Err(CliError::rule(format!(
                "--glm52-prefill-chunk-size must be a positive multiple of {}, got {}",
                crate::GLM52_PREFILL_CHUNK_ALIGN,
                cli.glm52_prefill_chunk_size
            )));
        }
        if cli.glm52_native_mtp && shared.dflash_draft_model_path.is_some() {
            return Err(CliError::rule(
                "--glm52-native-mtp and --dflash-draft-model-path are mutually exclusive",
            ));
        }
        if cli.glm52_native_mtp
            && !matches!(
                moe_topo,
                Glm52MoeTopo::Ep4
                    | Glm52MoeTopo::Ep8
                    | Glm52MoeTopo::Ep16
                    | Glm52MoeTopo::Ep32
                    | Glm52MoeTopo::Ep64
            )
            && !(cli.glm52_prefill_only && matches!(moe_topo, Glm52MoeTopo::Tp4))
        {
            return Err(CliError::rule(
                "--glm52-native-mtp requires EP decode (--moe-topo=ep4 or ep8 and up) or TP4 prefill-only",
            ));
        }
        Ok(())
    }

    fn serve_plan(&self, ctx: &LaunchContext<'_>) -> Result<ServePlan, CliError> {
        let cli = cli(ctx);
        let moe_topo = moe_topo(&cli)?;
        let scheduler_partition_count = match &cli.glm52_ranks {
            // A partial fleet hosts only its own ranks; a mirrored topology
            // always collapses to one logical rank.
            Some(spec) if !moe_topo.uses_tensor_replicated_moe() => crate::parse_rank_range(spec)
                .map_err(|err| CliError::rule(format!("--glm52-ranks: {err}")))?
                .len(),
            _ => moe_topo.logical_rank_count(),
        };
        Ok(ServePlan {
            scheduler_partition_count,
            prefill_only: cli.glm52_prefill_only,
            lora_modules: None,
        })
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        use anyhow::Context;
        let cli = cli(ctx);
        let shared = ctx.shared;
        let moe_topo = moe_topo(&cli)?;
        let drafter = if cli.glm52_native_mtp {
            Glm52Drafter::NativeMtp
        } else if let Some(path) = &shared.dflash_draft_model_path {
            Glm52Drafter::Dspark(path.clone())
        } else {
            Glm52Drafter::None
        };
        crate::launch(
            ctx.model_path,
            Glm52LaunchOptions {
                tp_size: shared.tp_size,
                dp_size: shared.dp_size.unwrap_or_else(|| moe_topo.default_dp_size()),
                drafter,
                max_model_len: cli.max_model_len,
                prefill_only: cli.glm52_prefill_only.then_some(Glm52PrefillOnlyOptions {
                    chunk_size: cli.glm52_prefill_chunk_size,
                }),
                no_prefix_cache: shared.no_prefix_cache,
                kv_offload: shared.kv_offload.then(|| Glm52KvOffloadOptions {
                    pinned_pool_bytes: (shared.kv_offload_host_gib * f64::from(1u32 << 30))
                        as usize,
                    use_hugepages: shared.kv_offload_hugepages,
                    p2p: match (
                        shared.kv_p2p_metaserver_addr.clone(),
                        shared.kv_p2p_advertise_addr.clone(),
                    ) {
                        (Some(metaserver_addr), Some(advertise_addr)) => Some(Glm52P2pOptions {
                            metaserver_addr,
                            advertise_addr,
                            rdma_nics: shared.kv_p2p_nics.clone(),
                        }),
                        _ => None,
                    },
                }),
                moe_topo,
                weight_staging: cli.glm52_weight_staging,
                dump_graph_png: shared.dump_graph_png.clone(),
                ranks: cli
                    .glm52_ranks
                    .as_deref()
                    .map(crate::parse_rank_range)
                    .transpose()
                    .context("--glm52-ranks")?,
                rendezvous: cli.glm52_rendezvous.clone(),
            },
        )
        .context("failed to start GLM5.2 engine")
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
    fn accepts_graph_png_dump() {
        validate_argv(&["pegainfer", "--dump-graph-png", "decode.png"])
            .expect("GLM5.2 should accept a graph PNG dump");
    }

    #[test]
    fn accepts_omitted_dp_size() {
        validate_argv(&["pegainfer"])
            .expect("GLM5.2 should default to DP8/EP8 when --dp-size is omitted");
    }

    #[test]
    fn accepts_native_mtp_on_ep8() {
        validate_argv(&["pegainfer", "--glm52-native-mtp"])
            .expect("native MTP should validate on the default EP8 topology");
    }

    #[test]
    fn accepts_native_mtp_on_ep4() {
        validate_argv(&["pegainfer", "--glm52-native-mtp", "--moe-topo", "ep4"])
            .expect("native MTP should validate on EP4");
    }

    #[test]
    fn native_mtp_rejects_a_second_drafter() {
        let error = validate_argv(&[
            "pegainfer",
            "--glm52-native-mtp",
            "--dflash-draft-model-path",
            "/tmp/dspark",
        ])
        .expect_err("native MTP and DSpark must be mutually exclusive");
        assert!(error.to_string().contains("mutually exclusive"), "{error}");
    }

    #[test]
    fn native_mtp_rejects_tp_topology() {
        let error = validate_argv(&[
            "pegainfer",
            "--glm52-native-mtp",
            "--moe-topo",
            "tp4",
            "--tp-size",
            "4",
        ])
        .expect_err("native MTP currently requires EP decode");
        assert!(
            error.to_string().contains("--moe-topo=ep4 or ep8"),
            "{error}"
        );
    }

    #[test]
    fn rejects_non_dp8_for_ep8() {
        let error = validate_argv(&["pegainfer", "--dp-size", "1"])
            .expect_err("GLM5.2 should reject explicit non-DP8");
        assert!(error.to_string().contains("--dp-size=8"));
    }

    #[test]
    fn accepts_tp4_dp1() {
        validate_argv(&["pegainfer", "--moe-topo", "tp4", "--tp-size", "4"])
            .expect("GLM5.2 TP4 should default to DP1");
    }

    #[test]
    fn tp4_rejects_non_dp1() {
        let error = validate_argv(&[
            "pegainfer",
            "--moe-topo",
            "tp4",
            "--tp-size",
            "4",
            "--dp-size",
            "8",
        ])
        .expect_err("GLM5.2 TP4 should reject explicit non-DP1");
        assert!(error.to_string().contains("--dp-size=1"));
    }

    #[test]
    fn tp4_rejects_omitted_tp_size() {
        let error = validate_argv(&["pegainfer", "--moe-topo", "tp4"])
            .expect_err("GLM5.2 TP4 should reject the default --tp-size=1");
        assert!(error.to_string().contains("--tp-size=4"));
    }

    #[test]
    fn ep8_rejects_tp4_tp_size() {
        let error = validate_argv(&["pegainfer", "--tp-size", "4"])
            .expect_err("GLM5.2 EP8 should reject --tp-size=4");
        assert!(error.to_string().contains("--tp-size=1"));
    }

    #[test]
    fn rejects_unknown_moe_topo() {
        let error = validate_argv(&["pegainfer", "--moe-topo", "tp2"])
            .expect_err("GLM5.2 should reject an unknown topology string");
        assert!(
            error
                .to_string()
                .contains("ep4, ep8, ep16, ep32, ep64, or tp4")
        );
    }

    #[test]
    fn accepts_ep4_default_dp4() {
        validate_argv(&["pegainfer", "--moe-topo", "ep4"])
            .expect("GLM5.2 EP4 should default to DP4 with --tp-size=1");
    }

    #[test]
    fn ep4_rejects_non_dp4() {
        let error = validate_argv(&["pegainfer", "--moe-topo", "ep4", "--dp-size", "8"])
            .expect_err("GLM5.2 EP4 should reject explicit non-DP4");
        assert!(error.to_string().contains("--dp-size=4"));
    }

    #[test]
    fn prefill_only_accepts_tp4_defaults() {
        validate_argv(&[
            "pegainfer",
            "--moe-topo",
            "tp4",
            "--tp-size",
            "4",
            "--glm52-prefill-only",
        ])
        .expect("TP4 prefill-only defaults should validate");
    }

    #[test]
    fn prefill_only_rejects_decode_features() {
        for extra in [
            vec!["--no-prefix-cache"],
            vec!["--dflash-draft-model-path", "/tmp/dspark"],
            vec!["--dump-graph-png", "/tmp/decode.png"],
        ] {
            let mut argv = vec![
                "pegainfer",
                "--moe-topo",
                "tp4",
                "--tp-size",
                "4",
                "--glm52-prefill-only",
            ];
            argv.extend(extra);
            validate_argv(&argv).expect_err("prefill-only must reject decode-only features");
        }
    }

    #[test]
    fn prefill_chunk_requires_mode_and_page_alignment() {
        let error = validate_argv(&["pegainfer", "--glm52-prefill-chunk-size", "16384"])
            .expect_err("an inert chunk size must be rejected");
        assert!(error.to_string().contains("requires --glm52-prefill-only"));

        let error = validate_argv(&[
            "pegainfer",
            "--moe-topo",
            "tp4",
            "--tp-size",
            "4",
            "--glm52-prefill-only",
            "--glm52-prefill-chunk-size",
            "16001",
        ])
        .expect_err("unaligned chunk must be rejected");
        assert!(error.to_string().contains("positive multiple of 64"));
    }

    #[test]
    fn serve_plan_counts_partial_fleet_ranks() {
        let (shared, matches, _provided) = parse_for_line(
            &MODEL_LINE,
            &[
                "pegainfer",
                "--moe-topo",
                "ep16",
                "--glm52-ranks",
                "4..8",
                "--glm52-rendezvous",
                "host:1",
            ],
        )
        .expect("partial-fleet flags should parse");
        let config = serde_json::json!({});
        let ctx = LaunchContext {
            model_path: std::path::Path::new("unused"),
            config: &config,
            shared: &shared,
            matches: &matches,
        };
        let plan = MODEL_LINE.serve_plan(&ctx).expect("serve plan");
        assert_eq!(plan.scheduler_partition_count, 4);
        assert!(!plan.prefill_only);
    }
}
