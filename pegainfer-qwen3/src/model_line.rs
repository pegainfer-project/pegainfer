//! Qwen3's [`ModelLine`] implementation: config probing, the Qwen3-exclusive
//! CLI section, cross-flag rules, and launch-option assembly. The server
//! binary only sees `&MODEL_LINE`.

use std::collections::BTreeSet;

use clap::Args as ClapArgs;
use clap::FromArgMatches;
use pegainfer_frontend::engine::LaunchedEngine;
use pegainfer_frontend::model_line::ArgRequirement;
use pegainfer_frontend::model_line::CliError;
use pegainfer_frontend::model_line::LaunchContext;
use pegainfer_frontend::model_line::ModelLine;
use pegainfer_frontend::model_line::ServePlan;
use pegainfer_frontend::vllm::LoraModule;
use pegainfer_frontend::vllm::parse_lora_modules_arg;

use crate::DecodeOverlap;
use crate::Qwen3LaunchOptions;
use crate::Qwen3LoraOptions;
use crate::Qwen3MemoryOptions;
use crate::Qwen3OffloadOptions;
use crate::Qwen3P2pOptions;
use crate::Qwen3VllmCompatOptions;

pub static MODEL_LINE: Qwen3Line = Qwen3Line;

pub struct Qwen3Line;

// Qwen3-exclusive CLI flags. Shared flags (`--tp-size`, `--kv-offload`, …)
// live in `SharedArgs`. (Regular comment: a doc comment would override the
// command about via clap's augment.)
#[derive(ClapArgs)]
#[allow(clippy::struct_excessive_bools)] // independent CLI flags, not a state machine
struct Qwen3Cli {
    /// Enable Qwen3 LoRA serving mode.
    #[arg(long, default_value_t = false)]
    enable_lora: bool,

    /// LoRA modules to load at startup. Accepts vLLM-style `name=path`, JSON
    /// object, or JSON list object entries with `name` and `path`.
    #[arg(long = "lora-modules", value_parser = parse_lora_modules_arg)]
    lora_modules: Vec<LoraModule>,

    /// Maximum number of resident LoRA adapters in Qwen3 LoRA mode.
    #[arg(long = "max-loras", default_value_t = Qwen3LoraOptions::DEFAULT_MAX_LORAS)]
    max_loras: usize,

    /// Maximum supported LoRA rank in Qwen3 LoRA mode.
    #[arg(long = "max-lora-rank", default_value_t = Qwen3LoraOptions::DEFAULT_MAX_LORA_RANK, value_parser = parse_max_lora_rank_arg)]
    max_lora_rank: usize,

    /// P/D prefill role: barrier each request's KV saves (host tier +
    /// MetaServer registration) before its final token event, so this
    /// instance's HTTP response doubles as the KV-ready signal a router can
    /// act on. Leave off on decode instances.
    #[arg(long, default_value_t = false)]
    kv_p2p_flush_on_finish: bool,

    /// P/D decode role with a vLLM prefill peer: the shared PYTHONHASHSEED
    /// value set on every vLLM prefill process. Switches offload query keys to
    /// vLLM's prefix-cache hash scheme (requires the P side to run
    /// --prefix-caching-hash-algo xxhash_cbor) and makes a cold request wait
    /// out the producer's registration tail instead of prefilling locally.
    /// Requires --kv-pd-vllm-namespace and the P2P mesh flags.
    #[arg(long, value_parser = parse_pythonhashseed)]
    kv_pd_vllm_seed: Option<String>,

    /// The vLLM prefill peer's pegaflow-connector namespace (an 8-hex digest
    /// the connector logs at startup as `namespace=...`). Both sides must
    /// address the same content domain. The digest carries no model identity:
    /// pointing a decode node at a different model's namespace (same
    /// tokenizer, same geometry class) silently cross-loads foreign KV.
    #[arg(long, value_parser = parse_pegaflow_namespace)]
    kv_pd_vllm_namespace: Option<String>,

    /// Zero-hit wait window for --kv-pd-vllm-seed mode, in milliseconds: how
    /// long a cold request keeps re-querying before giving up on the expected
    /// remote KV and prefilling locally. Must stay below the executor's 15s
    /// remote-fetch deadline (enforced at startup).
    #[arg(long, default_value_t = 5000)]
    kv_pd_miss_wait_ms: u64,

    /// Fraction of total GPU memory the Qwen3 instance may use. The KV cache is
    /// sized from this budget after startup profiling accounts for weights,
    /// runtime buffers, activation peak, margin, and (single-GPU only) CUDA-graph
    /// capture; tensor parallelism runs decode eagerly (no graph).
    #[arg(long, default_value_t = crate::DEFAULT_GPU_MEMORY_UTILIZATION)]
    gpu_memory_utilization: f64,

    /// Additional Qwen3 GPU memory to hold back after profile-based KV sizing,
    /// in MiB. Covers allocator fragmentation and small unprofiled drift.
    #[arg(long, default_value_t = (crate::DEFAULT_KV_CACHE_MEMORY_MARGIN_BYTES >> 20) as usize)]
    kv_cache_memory_margin_mib: usize,

    /// KV cache page (block) size in tokens. FlashInfer's paged attention only
    /// accepts a restricted set; 16 (default) or 64. Larger pages cut block
    /// bookkeeping overhead at the cost of coarser-grained allocation.
    #[arg(long, default_value_t = crate::DEFAULT_KV_PAGE_SIZE)]
    kv_page_size: usize,

    /// How prefill and decode share the GPU (single-GPU Qwen3 only).
    /// `off` serializes them on one stream (lowest TTFT); `stream` overlaps on
    /// two streams sharing all SMs; `green-ctx` pins each to a disjoint Green
    /// Context SM partition (lower decode ITL p99, higher TTFT).
    #[arg(long, value_enum, default_value_t = CliDecodeOverlap::Off)]
    decode_overlap: CliDecodeOverlap,

    /// Percent of SMs pinned to decode in `--decode-overlap green-ctx` (the rest
    /// go to prefill); rejected if set in any other mode.
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(1..=99))]
    decode_sm_pct: u32,

    /// Enable single-GPU Qwen3 batch-invariant serving by pinning the numeric paths and cutting
    /// each prompt's prefill chunks on its own grid. Off by default. Requires `--no-prefix-cache`;
    /// incompatible with `--kv-offload`, which keeps prefix matching on regardless.
    #[arg(long, default_value_t = false)]
    batch_invariant: bool,
}

/// CLI selector for prefill/decode overlap. Mapped to [`DecodeOverlap`]
/// together with `--decode-sm-pct`.
#[derive(Clone, Copy, Debug, clap::ValueEnum)]
enum CliDecodeOverlap {
    /// One stream; prefill and decode serialize.
    Off,
    /// Two CUDA streams sharing all SMs.
    Stream,
    /// Green Context SM partition (SM-pinned streams).
    #[value(name = "green-ctx")]
    GreenCtx,
}

impl CliDecodeOverlap {
    fn resolve(self, decode_sm_pct: u32) -> DecodeOverlap {
        match self {
            Self::Off => DecodeOverlap::Off,
            Self::Stream => DecodeOverlap::SharedSm,
            Self::GreenCtx => DecodeOverlap::GreenCtx {
                decode_pct: decode_sm_pct,
            },
        }
    }
}

fn cli(ctx: &LaunchContext<'_>) -> Qwen3Cli {
    Qwen3Cli::from_arg_matches(ctx.matches).expect("Qwen3Cli parses from the merged command")
}

impl ModelLine for Qwen3Line {
    fn name(&self) -> &'static str {
        "Qwen3"
    }

    fn probe(&self, config: &serde_json::Value) -> Result<(), String> {
        let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
        if model_type != Some("qwen3") {
            return Err(format!("model_type {model_type:?} is not \"qwen3\""));
        }
        crate::probe_config_json(config).map_err(|error| error.to_string())
    }

    fn augment_cli(&self, cmd: clap::Command) -> clap::Command {
        Qwen3Cli::augment_args(cmd)
    }

    fn consumed_shared_args(&self) -> &'static [&'static str] {
        &[
            "cuda_graph",
            "dump_graph_png",
            "device_ordinal",
            "tp_size",
            "kv_offload",
            "kv_offload_host_gib",
            "kv_offload_hugepages",
            "kv_p2p_metaserver_addr",
            "kv_p2p_advertise_addr",
            "kv_p2p_nics",
            "no_prefix_cache",
            "max_prefill_tokens",
            "dflash_draft_model_path",
        ]
    }

    fn arg_requirements(&self) -> &'static [ArgRequirement] {
        &[
            ("kv_p2p_flush_on_finish", &["kv_p2p_metaserver_addr"]),
            (
                "kv_pd_vllm_seed",
                &["kv_p2p_metaserver_addr", "kv_pd_vllm_namespace"],
            ),
            ("kv_pd_vllm_namespace", &["kv_pd_vllm_seed"]),
            ("kv_pd_miss_wait_ms", &["kv_pd_vllm_seed"]),
        ]
    }

    fn validate(
        &self,
        ctx: &LaunchContext<'_>,
        provided: &BTreeSet<String>,
    ) -> Result<(), CliError> {
        let cli = cli(ctx);
        let shared = ctx.shared;
        if !cli.enable_lora && !cli.lora_modules.is_empty() {
            return Err(CliError::rule("--lora-modules requires --enable-lora"));
        }
        if !cli.enable_lora
            && (provided.contains("max_loras") || provided.contains("max_lora_rank"))
        {
            return Err(CliError::rule(
                "--max-loras and --max-lora-rank require --enable-lora",
            ));
        }
        if shared.dump_graph_png.is_some() && cli.enable_lora {
            return Err(CliError::rule(
                "--dump-graph-png is not supported with --enable-lora (LoRA disables CUDA Graph)",
            ));
        }
        if provided.contains("decode_sm_pct")
            && !matches!(cli.decode_overlap, CliDecodeOverlap::GreenCtx)
        {
            return Err(CliError::rule(
                "--decode-sm-pct only applies with --decode-overlap=green-ctx",
            ));
        }
        if !matches!(cli.decode_overlap, CliDecodeOverlap::Off) && shared.tp_size > 1 {
            return Err(CliError::rule(
                "--decode-overlap is single-GPU only; tp_size>1 has no prefill/decode overlap",
            ));
        }
        if cli.batch_invariant {
            if cli.enable_lora {
                return Err(CliError::rule(
                    "--batch-invariant is not supported with --enable-lora; enable one at a time",
                ));
            }
            if !matches!(cli.decode_overlap, CliDecodeOverlap::Off) {
                return Err(CliError::rule(
                    "--batch-invariant is not compatible with --decode-overlap; the stream override would force the pinned GEMM to bail at runtime",
                ));
            }
            if shared.kv_offload {
                return Err(CliError::rule(
                    "--batch-invariant is not supported with --kv-offload: offload keeps prefix matching \
                     on (--no-prefix-cache only disables HBM retention there), and a host-tier prefix hit \
                     shifts a prompt's chunk boundaries off the request-local grid",
                ));
            }
            if shared.dflash_draft_model_path.is_some() {
                return Err(CliError::rule(
                    "--batch-invariant is not supported with DFlash speculative decoding; enable one at a time",
                ));
            }
            if shared.tp_size > 1 {
                return Err(CliError::rule(
                    "--batch-invariant is not supported with --tp-size > 1; enable one at a time",
                ));
            }
            if !shared.no_prefix_cache {
                return Err(CliError::rule(
                    "--batch-invariant requires --no-prefix-cache; prefix-cache hits move a prompt's chunk \
                     boundaries off the request-local grid, so batch-invariant prefill cannot be provided",
                ));
            }
        }
        if shared.dflash_draft_model_path.is_some() {
            if cli.enable_lora {
                return Err(CliError::rule(
                    "--dflash-draft-model-path is not supported with --enable-lora",
                ));
            }
            if shared.kv_offload {
                return Err(CliError::rule(
                    "--dflash-draft-model-path is not supported with --kv-offload",
                ));
            }
            if shared.tp_size != 1 {
                return Err(CliError::rule(
                    "--dflash-draft-model-path currently requires --tp-size=1",
                ));
            }
        }
        Ok(())
    }

    fn serve_plan(&self, ctx: &LaunchContext<'_>) -> Result<ServePlan, CliError> {
        let cli = cli(ctx);
        Ok(ServePlan {
            lora_modules: cli.enable_lora.then(|| cli.lora_modules.clone()),
            ..ServePlan::default()
        })
    }

    fn launch(&self, ctx: &LaunchContext<'_>) -> anyhow::Result<LaunchedEngine> {
        let cli = cli(ctx);
        let shared = ctx.shared;
        let offload = if shared.kv_offload {
            let bytes = (shared.kv_offload_host_gib * f64::from(1u32 << 30)) as usize;
            let mut offload = Qwen3OffloadOptions::enabled(bytes);
            offload.use_hugepages = shared.kv_offload_hugepages;
            if let (Some(metaserver_addr), Some(advertise_addr)) = (
                shared.kv_p2p_metaserver_addr.clone(),
                shared.kv_p2p_advertise_addr.clone(),
            ) {
                offload = offload.with_p2p(Qwen3P2pOptions {
                    metaserver_addr,
                    advertise_addr,
                    rdma_nics: shared.kv_p2p_nics.clone(),
                    flush_on_finish: cli.kv_p2p_flush_on_finish,
                });
            }
            if let Some(seed) = cli.kv_pd_vllm_seed.clone() {
                offload = offload.with_vllm_compat(Qwen3VllmCompatOptions {
                    python_hash_seed: seed,
                    namespace: cli
                        .kv_pd_vllm_namespace
                        .clone()
                        .expect("clap requires kv_pd_vllm_namespace"),
                    miss_wait: std::time::Duration::from_millis(cli.kv_pd_miss_wait_ms),
                });
            }
            offload
        } else {
            Qwen3OffloadOptions::disabled()
        };
        let lora = cli.enable_lora.then_some(Qwen3LoraOptions {
            max_loras: cli.max_loras,
            max_lora_rank: cli.max_lora_rank,
        });
        let kv_cache_memory_margin_bytes = cli
            .kv_cache_memory_margin_mib
            .checked_mul(1 << 20)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "--kv-cache-memory-margin-mib is too large: {}",
                    cli.kv_cache_memory_margin_mib
                )
            })?;
        crate::launch(
            ctx.model_path,
            Qwen3LaunchOptions {
                device_ordinal: shared.device_ordinal,
                tp_size: shared.tp_size,
                cuda_graph: shared.cuda_graph,
                dump_graph_png: shared.dump_graph_png.clone(),
                offload,
                no_prefix_cache: shared.no_prefix_cache,
                max_prefill_tokens: shared
                    .max_prefill_tokens
                    .unwrap_or(crate::DEFAULT_MAX_PREFILL_TOKENS),
                memory: Qwen3MemoryOptions::new(
                    cli.gpu_memory_utilization,
                    kv_cache_memory_margin_bytes,
                    cli.kv_page_size,
                )
                .validate()?,
                lora,
                decode_overlap: cli.decode_overlap.resolve(cli.decode_sm_pct),
                batch_invariant: cli.batch_invariant,
                dflash_draft_model_path: shared.dflash_draft_model_path.clone(),
            },
        )
        .map(LaunchedEngine::Stepped)
    }
}

fn parse_max_lora_rank_arg(value: &str) -> Result<usize, String> {
    let rank = value
        .parse::<usize>()
        .map_err(|error| format!("invalid --max-lora-rank: {error}"))?;
    if Qwen3LoraOptions::is_supported_max_lora_rank(rank) {
        Ok(rank)
    } else {
        Err(format!(
            "--max-lora-rank must be one of: {}",
            Qwen3LoraOptions::supported_max_lora_ranks_display()
        ))
    }
}

/// PYTHONHASHSEED as vLLM accepts it: a decimal integer in [0, 4294967295].
/// An empty or malformed seed would derive a well-formed key space that can
/// never match the peer — a config error must fail here, not as slow requests.
fn parse_pythonhashseed(s: &str) -> Result<String, String> {
    if s.parse::<u32>().is_err() || s.starts_with('+') {
        return Err(format!(
            "PYTHONHASHSEED must be a decimal integer in [0, 4294967295], got {s:?}"
        ));
    }
    Ok(s.to_string())
}

/// A pegaflow namespace digest: exactly 8 lowercase hex chars.
fn parse_pegaflow_namespace(s: &str) -> Result<String, String> {
    if s.len() != 8
        || !s
            .bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
    {
        return Err(format!(
            "namespace must be an 8-char lowercase hex digest, got {s:?}"
        ));
    }
    Ok(s.to_string())
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
    fn probe_accepts_qwen3_identity() {
        let json = serde_json::json!({"model_type":"qwen3","architectures":["Qwen3ForCausalLM"]});
        MODEL_LINE.probe(&json).expect("qwen3 config should probe");
    }

    #[test]
    fn probe_rejects_bad_architectures() {
        let json = serde_json::json!({"model_type":"qwen3","architectures":["SomethingElse"]});
        MODEL_LINE.probe(&json).unwrap_err();
    }

    #[test]
    fn probe_rejects_foreign_model_type() {
        let json = serde_json::json!({"model_type":"glm_moe_dsa"});
        let reason = MODEL_LINE.probe(&json).unwrap_err();
        assert!(reason.contains("qwen3"), "{reason}");
    }

    #[test]
    fn accepts_graph_png_dump() {
        validate_argv(&["pegainfer", "--dump-graph-png", "decode.png"])
            .expect("graph PNG dump with CUDA Graph enabled should validate");
    }

    #[test]
    fn graph_png_dump_requires_cuda_graph() {
        let error = validate_argv(&[
            "pegainfer",
            "--dump-graph-png",
            "decode.png",
            "--cuda-graph=false",
        ])
        .expect_err("graph dump without CUDA Graph should be rejected");
        assert!(error.to_string().contains("requires --cuda-graph=true"));
    }

    #[test]
    fn graph_png_dump_rejects_lora() {
        let error = validate_argv(&[
            "pegainfer",
            "--dump-graph-png",
            "decode.png",
            "--enable-lora",
        ])
        .expect_err("graph dump with LoRA should be rejected");
        assert!(
            error
                .to_string()
                .contains("not supported with --enable-lora")
        );
    }

    #[test]
    fn rejects_other_lines_flags() {
        let error =
            parse_for_line(&MODEL_LINE, &["pegainfer", "--dp-size", "8"]).expect_err("dp_size");
        assert!(
            error.to_string().contains("is not used by Qwen3"),
            "{error}"
        );
    }

    #[test]
    fn parses_supported_max_lora_rank() {
        assert_eq!(parse_max_lora_rank_arg("16").expect("parse rank"), 16);
        assert_eq!(parse_max_lora_rank_arg("320").expect("parse rank"), 320);
    }

    #[test]
    fn rejects_unsupported_max_lora_rank() {
        let error = parse_max_lora_rank_arg("7").expect_err("rank should be unsupported");
        assert!(error.contains("--max-lora-rank must be one of"));
        assert!(error.contains("16"));
    }

    #[test]
    fn lora_default_rank_is_64() {
        assert_eq!(Qwen3LoraOptions::default().max_lora_rank, 64);
    }
}
