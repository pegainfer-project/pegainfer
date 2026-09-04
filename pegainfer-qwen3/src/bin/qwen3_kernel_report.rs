use std::cell::RefCell;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fs::File;
use std::fs::{self};
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use clap::Args;
use clap::Parser;
use clap::Subcommand;
use cudarc::driver::sys;
use pegainfer_cupti::profile_range_with_prepare;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_qwen3::runtime::SplitKvConfig;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value;
use serde_json::json;
use sha2::Digest;
use sha2::Sha256;

#[path = "report_support/mod.rs"]
mod report_support;

use report_support::common::DevicePeakBandwidth;
use report_support::common::HEAD_DIM;
use report_support::common::L2CacheClear;
use report_support::common::NUM_KV_HEADS;
use report_support::common::NUM_QO_HEADS;
use report_support::common::PAGE_SIZE;
use report_support::common::REPORT_ITERS;
use report_support::common::cache_clear_bytes;
use report_support::decode_attention::AttentionDecodeCase;
use report_support::decode_attention::AttentionKernelShape;
use report_support::decode_attention::AttentionKernelSpec;
use report_support::decode_attention::AttentionKernelVariant;
use report_support::decode_attention::DecodePath;
use report_support::dense::DenseCase;
use report_support::dense::DenseKernelKind;
use report_support::dense::GemmProjection;
use report_support::dense::HIDDEN_SIZE;
use report_support::dense::INTERMEDIATE_SIZE;
use report_support::dense::VOCAB_SIZE;
use report_support::prefill_attention::AttentionPrefillCase;
use report_support::prefill_attention::PrefillAttentionShape;
use report_support::prefill_attention::PrefillAttentionSpec;
use report_support::prefill_attention::PrefillAttentionVariant;
use report_support::prefill_attention::PrefillStage;
use report_support::single_prefill::SinglePrefillCase;

const DEFAULT_MANIFEST: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/kernel_manifests/qwen3.toml");
const DEFAULT_OP: &str = "paged_decode_attention";
const PREFILL_OP: &str = "paged_prefill_attention";
const PREFILL_QK_NORM_ROPE_OP: &str = "prefill_qk_norm_rope";
const PREFILL_KV_SCATTER_OP: &str = "prefill_kv_scatter";
const PREFILL_ATTENTION_CORE_OP: &str = "prefill_attention_core";
const SINGLE_PREFILL_ATTENTION_CORE_OP: &str = "single_prefill_attention_core";

/// Dense (non-attention) forward ops. `decode_*` sweeps batch_size, `prefill_*`
/// sweeps seq_len at batch_size=1 — `rows` is the step's token/row count either
/// way. lm_head is decode-only: production gathers last-token columns before
/// the logits GEMM, so its N is the request count, never the prefill token count.
#[derive(Clone, Copy, Eq, PartialEq)]
enum DensePhase {
    Decode,
    Prefill,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum DenseOpFamily {
    ProjectionGemm,
    RmsNorm,
    FusedAddRmsNorm,
    QkNormRope,
    SiluMul,
    Embedding,
    Sampling,
}

fn dense_op_for(op_name: &str) -> Option<(DenseOpFamily, DensePhase)> {
    let (phase, rest) = if let Some(rest) = op_name.strip_prefix("decode_") {
        (DensePhase::Decode, rest)
    } else {
        (DensePhase::Prefill, op_name.strip_prefix("prefill_")?)
    };
    let family = match rest {
        "projection_gemm" => DenseOpFamily::ProjectionGemm,
        "rms_norm" => DenseOpFamily::RmsNorm,
        "fused_add_rms_norm" => DenseOpFamily::FusedAddRmsNorm,
        "qk_norm_rope" if phase == DensePhase::Decode => DenseOpFamily::QkNormRope,
        "silu_mul" => DenseOpFamily::SiluMul,
        "embedding" => DenseOpFamily::Embedding,
        "sampling" if phase == DensePhase::Decode => DenseOpFamily::Sampling,
        _ => return None,
    };
    Some((family, phase))
}

fn parse_dense_variant(
    family: DenseOpFamily,
    phase: DensePhase,
    raw: &str,
) -> Result<DenseKernelKind> {
    match family {
        DenseOpFamily::ProjectionGemm => {
            let projection = GemmProjection::parse(raw)
                .ok_or_else(|| anyhow!("unknown projection gemm variant `{raw}`"))?;
            anyhow::ensure!(
                !(phase == DensePhase::Prefill && projection == GemmProjection::LmHead),
                "lm_head is decode-only: prefill gathers last-token columns first, \
                 so its logits GEMM never sees the prefill token count"
            );
            Ok(DenseKernelKind::ProjectionGemm(projection))
        }
        DenseOpFamily::Sampling => match raw {
            "argmax" => Ok(DenseKernelKind::Sampling { greedy: true }),
            "sampling" => Ok(DenseKernelKind::Sampling { greedy: false }),
            _ => bail!("sampling variants are `argmax` or `sampling`; got `{raw}`"),
        },
        _ => {
            anyhow::ensure!(
                raw == "default",
                "this dense op only supports variant `default`; got `{raw}`"
            );
            Ok(match family {
                DenseOpFamily::RmsNorm => DenseKernelKind::RmsNorm,
                DenseOpFamily::FusedAddRmsNorm => DenseKernelKind::FusedAddRmsNorm,
                DenseOpFamily::QkNormRope => DenseKernelKind::QkNormRopeDecode,
                DenseOpFamily::SiluMul => DenseKernelKind::SiluMul,
                DenseOpFamily::Embedding => DenseKernelKind::Embedding,
                DenseOpFamily::ProjectionGemm | DenseOpFamily::Sampling => unreachable!(),
            })
        }
    }
}
const DEFAULT_COMPOSITION: &str = "decode_attention_only";
const PARALLEL_STRATEGY: &str = "tp1_pp1";
const SHAPE_SOURCE: &str = "static";
const QWEN3_LAYER_COUNT: u64 = 36;
const COMPOSITION_COVERAGE_NOTE: &str = "Only measured paged decode attention is included; linear, MLP, norm, embedding, and sampling reports are not covered yet.";
const CUPTI_METRICS: &[&str] = &[
    "gpu__time_duration.sum",
    "sm__cycles_elapsed.avg.per_second",
    "dram__bytes.sum",
    "dram__bytes_op_read.sum",
    "dram__bytes_op_write.sum",
    "lts__t_bytes.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__warps_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_subpipe_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed",
    "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.per_second",
];
const LATENCY_WARN_PCT: f64 = 5.0;
const LATENCY_FAIL_PCT: f64 = 10.0;

#[derive(Parser)]
#[command(about = "Manifest-driven Qwen3 kernel per-op report runner")]
struct Cli {
    #[arg(long, default_value = DEFAULT_MANIFEST)]
    manifest: PathBuf,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run a per-op kernel report.
    Run(RunCli),
    /// Compare two per-op reports.
    Compare(CompareArgs),
    /// Compose one or more per-op reports into a phase contribution report.
    Compose(ComposeArgs),
    /// Rank op-report snapshots against their roofline: analytic
    /// speed-of-light (minimum traffic / FLOPs at the op's semantics vs device
    /// peaks) per selected variant, sorted most-headroom-first. Raw CUPTI
    /// metrics, when present in the snapshot, add achieved-DRAM and
    /// tensor-pipe columns for diagnosis. Interpretation lives here, not in
    /// `run` — snapshots stay raw.
    Rank(RankArgs),
}

#[derive(Args)]
struct RunCli {
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_OP)]
    op: String,
    #[arg(long, action = clap::ArgAction::SetTrue)]
    cupti: bool,
    #[arg(long, action = clap::ArgAction::SetTrue)]
    no_cupti: bool,
    #[arg(long)]
    iters: Option<u64>,
    #[arg(long, value_delimiter = ',')]
    contexts: Vec<usize>,
    #[arg(long = "seq-lens", value_delimiter = ',')]
    seq_lens: Vec<usize>,
    #[arg(long = "batch-sizes", value_delimiter = ',')]
    batch_sizes: Vec<usize>,
    #[arg(long, value_delimiter = ',')]
    variants: Vec<String>,
}

impl Default for RunCli {
    fn default() -> Self {
        Self {
            out: None,
            op: DEFAULT_OP.to_string(),
            cupti: false,
            no_cupti: false,
            iters: None,
            contexts: Vec::new(),
            seq_lens: Vec::new(),
            batch_sizes: Vec::new(),
            variants: Vec::new(),
        }
    }
}

#[derive(Args)]
struct CompareArgs {
    #[arg(long)]
    base: PathBuf,
    #[arg(long)]
    new: PathBuf,
}

#[derive(Args)]
struct RankArgs {
    /// Op-report snapshot JSONs (any mix of ops); selection winners are
    /// ranked. Takes several paths per flag, so a shell glob works:
    /// `rank --input reports/*.json`.
    #[arg(long = "input", required = true, num_args = 1..)]
    inputs: Vec<PathBuf>,
    /// Peak dense bf16 tensor TFLOPS of the GPU, for the compute-side bound.
    /// Deliberately an explicit input — there is no reliable CUDA query for it
    /// and a baked-in per-arch table would rot. Without it, compute-bound ops
    /// only get the DRAM bound plus their arithmetic intensity for context.
    #[arg(long = "peak-bf16-tflops")]
    peak_bf16_tflops: Option<f64>,
    /// Only print kernels below this % of their speed-of-light bound.
    #[arg(long)]
    below: Option<f64>,
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Args)]
struct ComposeArgs {
    #[arg(long = "input", required = true)]
    inputs: Vec<PathBuf>,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value = DEFAULT_COMPOSITION)]
    composition: String,
    #[arg(long = "batch-size", default_value_t = 1)]
    batch_size: usize,
    #[arg(long = "context", default_value_t = 1024)]
    kv_len: usize,
}

struct RunArgs {
    out: Option<PathBuf>,
    op: String,
    cupti: bool,
    iters: u64,
    contexts: Vec<usize>,
    seq_lens: Vec<usize>,
    batch_sizes: Vec<usize>,
    variants: Vec<String>,
}

struct ExternalProvenance {
    git: GitProvenance,
    build: BuildProvenance,
    driver_version: Option<String>,
    cuda_toolkit: Option<String>,
}

struct LoadedManifest {
    path: PathBuf,
    manifest: KernelManifest,
    hash: String,
}

#[derive(Deserialize)]
struct KernelManifest {
    model: String,
    ops: Vec<OpManifest>,
}

#[derive(Deserialize)]
struct OpManifest {
    name: String,
    #[serde(default)]
    phase: String,
    #[serde(default)]
    batch_size: Vec<usize>,
    #[serde(default)]
    kv_len: Vec<usize>,
    #[serde(default)]
    seq_len: Vec<usize>,
    #[serde(default)]
    variants: Vec<String>,
}

#[derive(Clone, Copy)]
enum KernelSpec {
    Decode(AttentionKernelSpec),
    Prefill {
        spec: PrefillAttentionSpec,
        stage: PrefillStage,
    },
    SinglePrefill(PrefillAttentionSpec),
    Dense {
        kind: DenseKernelKind,
        rows: usize,
        phase: DensePhase,
    },
}

#[derive(Clone, Copy, Deserialize, Serialize)]
struct RegressionThresholds {
    #[serde(default = "default_latency_warn_pct")]
    latency_warn_pct: f64,
    #[serde(default = "default_latency_fail_pct")]
    latency_fail_pct: f64,
}

impl Default for RegressionThresholds {
    fn default() -> Self {
        Self {
            latency_warn_pct: LATENCY_WARN_PCT,
            latency_fail_pct: LATENCY_FAIL_PCT,
        }
    }
}

#[derive(Deserialize, Serialize)]
struct KernelSnapshot {
    schema: u32,
    #[serde(default = "default_report_type")]
    report_type: String,
    model: String,
    #[serde(default)]
    parallel_strategy: String,
    op: String,
    #[serde(default)]
    phase: String,
    created_at_unix_secs: u64,
    #[serde(default)]
    manifest_path: Option<String>,
    #[serde(default)]
    manifest_hash: Option<String>,
    #[serde(default)]
    hardware_class: Vec<String>,
    git: GitProvenance,
    hardware: HardwareProvenance,
    build: BuildProvenance,
    measurement: MeasurementConfig,
    #[serde(default)]
    thresholds: RegressionThresholds,
    cases: Vec<CaseResult>,
    #[serde(default)]
    selections: Vec<VariantSelection>,
}

#[derive(Deserialize, Serialize)]
struct GitProvenance {
    commit: Option<String>,
    dirty: Option<bool>,
}

#[derive(Deserialize, Serialize)]
struct HardwareProvenance {
    gpu_name: String,
    device_ordinal: usize,
    compute_capability: String,
    driver_version: Option<String>,
    cuda_toolkit: Option<String>,
    memory_clock_khz: i32,
    memory_bus_width_bits: i32,
    peak_gb_s: f64,
    l2_bytes: usize,
    cache_clear_bytes: usize,
}

#[derive(Deserialize, Serialize)]
struct BuildProvenance {
    target_sm_env: Option<String>,
    flashinfer_commit: Option<String>,
    kernel_archive: Option<String>,
    kernel_archive_fnv1a64: Option<String>,
}

#[derive(Deserialize, Serialize)]
struct MeasurementConfig {
    iters: u64,
    cache_state: String,
    #[serde(default)]
    pre_measure_launches: u64,
    inner_launches: u64,
    cupti_enabled: bool,
    cupti_metrics: Vec<String>,
}

#[derive(Clone, Deserialize, Serialize)]
struct CaseResult {
    #[serde(default)]
    case_id: String,
    op: String,
    variant: String,
    #[serde(default)]
    selector_key: Value,
    #[serde(default)]
    shape_source: String,
    shape: CaseShape,
    params: CaseParams,
    latency_us: Option<f64>,
    cupti: Option<BTreeMap<String, f64>>,
    error: Option<String>,
}

#[derive(Clone, Deserialize, Serialize)]
struct CaseShape {
    batch_size: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv_len: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    seq_len: Option<usize>,
    num_qo_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    page_size: usize,
    dtype: String,
}

#[derive(Clone, Deserialize, Serialize)]
struct CaseParams {
    chunk_tokens: Option<usize>,
    max_chunks_per_request: Option<usize>,
    actual_chunk_size: Option<usize>,
    active_chunks_per_request: Option<usize>,
    padded_slots: Option<usize>,
    cta_tile_q: Option<usize>,
    /// Projection GEMM (out_dim, in_dim); the `rank` pass reads these to build
    /// the analytic traffic/FLOP model. Absent on non-GEMM cases.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    gemm_out_dim: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    gemm_in_dim: Option<usize>,
}

#[derive(Clone, Deserialize, Serialize)]
struct VariantSelection {
    case_id: String,
    op: String,
    selector_key: Value,
    shape: CaseShape,
    selected_variant: String,
    latency_us: Option<f64>,
}

#[derive(Serialize)]
struct CompositionReport {
    schema: u32,
    report_type: String,
    model: String,
    parallel_strategy: String,
    phase: String,
    composition: String,
    selector_key: Value,
    source_op_reports: Vec<String>,
    total_latency_us: Option<f64>,
    ops: Vec<ComposedOpResult>,
    coverage_note: Option<String>,
}

#[derive(Serialize)]
struct ComposedOpResult {
    op: String,
    selected_variant: Option<String>,
    selector_key: Value,
    repeat_count: u64,
    single_latency_us: Option<f64>,
    total_latency_us: Option<f64>,
    share_pct: Option<f64>,
    error: Option<String>,
}

fn default_latency_warn_pct() -> f64 {
    LATENCY_WARN_PCT
}

fn default_latency_fail_pct() -> f64 {
    LATENCY_FAIL_PCT
}

fn default_report_type() -> String {
    "op_report".to_string()
}

fn load_manifest(path: &Path) -> Result<LoadedManifest> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read manifest {}", path.display()))?;
    let manifest: KernelManifest = toml::from_str(&content)
        .with_context(|| format!("failed to parse manifest {}", path.display()))?;
    anyhow::ensure!(
        !manifest.model.is_empty(),
        "manifest model must not be empty"
    );
    Ok(LoadedManifest {
        path: path.to_path_buf(),
        manifest,
        hash: sha256_short(content.as_bytes()),
    })
}

fn op_manifest<'a>(manifest: &'a KernelManifest, op_name: &str) -> Result<&'a OpManifest> {
    manifest
        .ops
        .iter()
        .find(|op| op.name == op_name)
        .ok_or_else(|| anyhow!("manifest does not define op `{op_name}`"))
}

fn prefill_stage_for_op(op_name: &str) -> Option<PrefillStage> {
    match op_name {
        PREFILL_OP => Some(PrefillStage::Full),
        PREFILL_QK_NORM_ROPE_OP => Some(PrefillStage::QkNormRope),
        PREFILL_KV_SCATTER_OP => Some(PrefillStage::KvScatter),
        PREFILL_ATTENTION_CORE_OP => Some(PrefillStage::AttentionCore),
        _ => None,
    }
}

fn is_single_prefill_op(op_name: &str) -> bool {
    op_name == SINGLE_PREFILL_ATTENTION_CORE_OP
}

fn build_run_args(manifest: &KernelManifest, cli: RunCli) -> Result<RunArgs> {
    let op = op_manifest(manifest, &cli.op)?;
    validate_op_manifest(op)?;
    let variants = if cli.variants.is_empty() {
        op.variants.clone()
    } else {
        cli.variants.clone()
    };
    validate_variants(op, &variants)?;
    if let Some((_, phase)) = dense_op_for(&op.name) {
        let rows = match phase {
            DensePhase::Decode => {
                anyhow::ensure!(
                    cli.seq_lens.is_empty() && cli.contexts.is_empty(),
                    "decode dense ops sweep --batch-sizes only"
                );
                if cli.batch_sizes.is_empty() {
                    op.batch_size.clone()
                } else {
                    ensure_positive_list("--batch-sizes", &cli.batch_sizes)?;
                    cli.batch_sizes.clone()
                }
            }
            DensePhase::Prefill => {
                anyhow::ensure!(
                    cli.batch_sizes.is_empty() && cli.contexts.is_empty(),
                    "prefill dense ops sweep --seq-lens only"
                );
                if cli.seq_lens.is_empty() {
                    op.seq_len.clone()
                } else {
                    ensure_positive_list("--seq-lens", &cli.seq_lens)?;
                    cli.seq_lens.clone()
                }
            }
        };
        let iters = cli.iters.unwrap_or(REPORT_ITERS);
        anyhow::ensure!(iters > 0, "--iters must be greater than zero");
        let (batch_sizes, seq_lens) = match phase {
            DensePhase::Decode => (rows, Vec::new()),
            DensePhase::Prefill => (vec![1], rows),
        };
        return Ok(RunArgs {
            out: cli.out,
            op: cli.op,
            cupti: cli.cupti || !cli.no_cupti,
            iters,
            contexts: Vec::new(),
            seq_lens,
            batch_sizes,
            variants,
        });
    }
    let contexts = if op.name == DEFAULT_OP {
        if cli.contexts.is_empty() {
            op.kv_len.clone()
        } else {
            ensure_positive_list("--contexts", &cli.contexts)?;
            cli.contexts.clone()
        }
    } else {
        Vec::new()
    };
    let seq_lens = if prefill_stage_for_op(&op.name).is_some() || is_single_prefill_op(&op.name) {
        if !cli.seq_lens.is_empty() {
            ensure_positive_list("--seq-lens", &cli.seq_lens)?;
            cli.seq_lens.clone()
        } else if !cli.contexts.is_empty() {
            ensure_positive_list("--contexts", &cli.contexts)?;
            cli.contexts.clone()
        } else {
            op.seq_len.clone()
        }
    } else {
        anyhow::ensure!(
            cli.seq_lens.is_empty(),
            "--seq-lens is only valid for prefill ops"
        );
        Vec::new()
    };
    let batch_sizes = if cli.batch_sizes.is_empty() {
        op.batch_size.clone()
    } else {
        ensure_positive_list("--batch-sizes", &cli.batch_sizes)?;
        cli.batch_sizes.clone()
    };
    let cupti = cli.cupti || !cli.no_cupti;
    let iters = cli.iters.unwrap_or(REPORT_ITERS);
    anyhow::ensure!(iters > 0, "--iters must be greater than zero");
    anyhow::ensure!(!variants.is_empty(), "at least one variant is required");
    Ok(RunArgs {
        out: cli.out,
        op: cli.op,
        cupti,
        iters,
        contexts,
        seq_lens,
        batch_sizes,
        variants,
    })
}

fn ensure_positive_list(name: &str, values: &[usize]) -> Result<()> {
    anyhow::ensure!(!values.is_empty(), "{name} must not be empty");
    anyhow::ensure!(
        values.iter().all(|value| *value > 0),
        "{name} values must be greater than zero"
    );
    Ok(())
}

fn validate_op_manifest(op: &OpManifest) -> Result<()> {
    if let Some((_, phase)) = dense_op_for(&op.name) {
        match phase {
            DensePhase::Decode => anyhow::ensure!(
                !op.batch_size.is_empty(),
                "`{}` needs non-empty batch_size",
                op.name
            ),
            DensePhase::Prefill => anyhow::ensure!(
                !op.seq_len.is_empty(),
                "`{}` needs non-empty seq_len",
                op.name
            ),
        }
        anyhow::ensure!(!op.variants.is_empty(), "`{}` needs variants", op.name);
        return Ok(());
    }
    anyhow::ensure!(
        op.name == DEFAULT_OP
            || prefill_stage_for_op(&op.name).is_some()
            || is_single_prefill_op(&op.name),
        "no provider for op `{}` in this report tool",
        op.name
    );
    anyhow::ensure!(
        !op.batch_size.is_empty(),
        "`{}` needs non-empty batch_size",
        op.name
    );
    if op.name == DEFAULT_OP {
        anyhow::ensure!(
            !op.kv_len.is_empty(),
            "`{DEFAULT_OP}` needs non-empty kv_len"
        );
    } else {
        anyhow::ensure!(
            !op.seq_len.is_empty(),
            "`{}` needs non-empty seq_len",
            op.name
        );
    }
    anyhow::ensure!(!op.variants.is_empty(), "`{}` needs variants", op.name);
    Ok(())
}

fn validate_variants(op: &OpManifest, variants: &[String]) -> Result<()> {
    anyhow::ensure!(!variants.is_empty(), "at least one variant is required");
    if let Some((family, phase)) = dense_op_for(&op.name) {
        for variant in variants {
            parse_dense_variant(family, phase, variant)?;
        }
        return Ok(());
    }
    match op.name.as_str() {
        DEFAULT_OP => {
            for variant in variants {
                parse_decode_variant(variant)?;
            }
        }
        _ if prefill_stage_for_op(&op.name).is_some() || is_single_prefill_op(&op.name) => {
            for variant in variants {
                if op.name == PREFILL_OP || op.name == PREFILL_ATTENTION_CORE_OP {
                    parse_prefill_variant(variant)?;
                } else {
                    anyhow::ensure!(
                        variant == "default",
                        "`{}` only supports variant `default`; got `{variant}`",
                        op.name
                    );
                }
            }
        }
        _ => unreachable!("op manifest validation should reject unknown providers"),
    }
    Ok(())
}

fn parse_decode_variant(raw: &str) -> Result<AttentionKernelVariant> {
    if raw == "non_partition" {
        return Ok(AttentionKernelVariant::NonPartition);
    }
    match SplitKvConfig::parse(raw) {
        Some(config) => Ok(AttentionKernelVariant::SplitKv(config)),
        None => bail!("variant must be non_partition or split_kv_{{chunk}}x{{max}}, got `{raw}`"),
    }
}

fn parse_prefill_variant(raw: &str) -> Result<PrefillAttentionVariant> {
    if raw == "default" {
        return Ok(PrefillAttentionVariant::Default);
    }
    let Some(tile_q) = raw.strip_prefix("cta_q") else {
        bail!("unknown prefill variant `{raw}`");
    };
    let tile_q = tile_q
        .parse::<usize>()
        .with_context(|| format!("invalid CTA tile Q in `{raw}`"))?;
    anyhow::ensure!(
        matches!(tile_q, 16 | 64 | 128),
        "prefill CTA tile Q must be one of 16, 64, or 128; got {tile_q}"
    );
    Ok(PrefillAttentionVariant::CtaTileQ(tile_q))
}

fn selected_specs(args: &RunArgs) -> Result<Vec<KernelSpec>> {
    if let Some((family, phase)) = dense_op_for(&args.op) {
        let kinds = args
            .variants
            .iter()
            .map(|variant| parse_dense_variant(family, phase, variant))
            .collect::<Result<Vec<_>>>()?;
        let rows: &[usize] = match phase {
            DensePhase::Decode => &args.batch_sizes,
            DensePhase::Prefill => &args.seq_lens,
        };
        return Ok(kinds
            .iter()
            .copied()
            .flat_map(|kind| {
                rows.iter()
                    .copied()
                    .map(move |rows| KernelSpec::Dense { kind, rows, phase })
            })
            .collect());
    }
    match args.op.as_str() {
        DEFAULT_OP => {
            let variants = args
                .variants
                .iter()
                .map(String::as_str)
                .map(parse_decode_variant)
                .collect::<Result<Vec<_>>>()?;
            Ok(args
                .batch_sizes
                .iter()
                .copied()
                .flat_map(|batch_size| {
                    variants.iter().copied().flat_map(move |variant| {
                        args.contexts.iter().copied().map(move |kv_len| {
                            KernelSpec::Decode(AttentionKernelSpec {
                                shape: AttentionKernelShape::new(batch_size, kv_len),
                                variant,
                            })
                        })
                    })
                })
                .collect())
        }
        _ => {
            if is_single_prefill_op(&args.op) {
                Ok(args
                    .batch_sizes
                    .iter()
                    .copied()
                    .flat_map(|batch_size| {
                        args.seq_lens.iter().copied().map(move |seq_len| {
                            KernelSpec::SinglePrefill(PrefillAttentionSpec {
                                shape: PrefillAttentionShape::new(batch_size, seq_len),
                                variant: PrefillAttentionVariant::Default,
                            })
                        })
                    })
                    .collect())
            } else {
                let Some(stage) = prefill_stage_for_op(&args.op) else {
                    unreachable!("run args should reject unknown providers");
                };
                let variants = if args.op == PREFILL_OP || args.op == PREFILL_ATTENTION_CORE_OP {
                    args.variants
                        .iter()
                        .map(String::as_str)
                        .map(parse_prefill_variant)
                        .collect::<Result<Vec<_>>>()?
                } else {
                    vec![PrefillAttentionVariant::Default]
                };
                Ok(args
                    .batch_sizes
                    .iter()
                    .copied()
                    .flat_map(|batch_size| {
                        variants.iter().copied().flat_map(move |variant| {
                            args.seq_lens
                                .iter()
                                .copied()
                                .map(move |seq_len| KernelSpec::Prefill {
                                    spec: PrefillAttentionSpec {
                                        shape: PrefillAttentionShape::new(batch_size, seq_len),
                                        variant,
                                    },
                                    stage,
                                })
                        })
                    })
                    .collect())
            }
        }
    }
}

fn run_snapshot(loaded: &LoadedManifest, args: &RunArgs) -> Result<KernelSnapshot> {
    let op = op_manifest(&loaded.manifest, &args.op)?;
    validate_op_manifest(op)?;
    let external = query_external_provenance();
    let probe = DeviceContext::new()?;
    let peak = DevicePeakBandwidth::query(&probe)?;
    drop(probe);

    let specs = selected_specs(args)?;
    eprintln!(
        "running {} {} {} kernel cases",
        specs.len(),
        loaded.manifest.model,
        args.op
    );
    let mut cases = Vec::with_capacity(specs.len());
    for spec in specs {
        match spec {
            KernelSpec::Decode(spec) => eprintln!(
                "case op={} variant={} bs={} ctx={}",
                args.op,
                spec.variant.label(),
                spec.shape.batch_size,
                spec.shape.kv_len
            ),
            KernelSpec::Prefill { spec, stage } => eprintln!(
                "case op={} stage={} variant={} bs={} seq={}",
                args.op,
                stage.label(),
                spec.variant.label(),
                spec.shape.batch_size,
                spec.shape.seq_len
            ),
            KernelSpec::SinglePrefill(spec) => eprintln!(
                "case op={} variant=default bs={} seq={}",
                args.op, spec.shape.batch_size, spec.shape.seq_len
            ),
            KernelSpec::Dense { kind, rows, phase } => eprintln!(
                "case op={} variant={} {}={rows}",
                args.op,
                kind.label(),
                match phase {
                    DensePhase::Decode => "bs",
                    DensePhase::Prefill => "seq",
                },
            ),
        }
        cases.push(measure_case(spec, args, &loaded.manifest, op));
    }
    let selections = build_selections(&cases);

    let probe = DeviceContext::new()?;
    let hardware = query_hardware(
        &probe,
        &peak,
        external.driver_version.clone(),
        external.cuda_toolkit.clone(),
    )?;

    Ok(KernelSnapshot {
        schema: 4,
        report_type: "op_report".to_string(),
        model: loaded.manifest.model.clone(),
        parallel_strategy: PARALLEL_STRATEGY.to_string(),
        op: args.op.clone(),
        phase: op.phase.clone(),
        created_at_unix_secs: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        manifest_path: Some(loaded.path.display().to_string()),
        manifest_hash: Some(loaded.hash.clone()),
        hardware_class: Vec::new(),
        git: external.git,
        hardware,
        build: external.build,
        measurement: MeasurementConfig {
            iters: args.iters,
            cache_state: "l2_cleared_by_sweep".to_string(),
            pre_measure_launches: 1,
            inner_launches: 1,
            cupti_enabled: args.cupti,
            cupti_metrics: if args.cupti {
                CUPTI_METRICS
                    .iter()
                    .map(|metric| (*metric).to_string())
                    .collect()
            } else {
                Vec::new()
            },
        },
        thresholds: RegressionThresholds::default(),
        cases,
        selections,
    })
}

fn measure_case(
    spec: KernelSpec,
    args: &RunArgs,
    manifest: &KernelManifest,
    op: &OpManifest,
) -> CaseResult {
    match spec {
        KernelSpec::Decode(spec) => measure_decode_case(spec, args, manifest, op),
        KernelSpec::Prefill { spec, stage } => {
            measure_prefill_case(spec, stage, args, manifest, op)
        }
        KernelSpec::SinglePrefill(spec) => measure_single_prefill_case(spec, args, manifest, op),
        KernelSpec::Dense { kind, rows, phase } => {
            measure_dense_case(kind, rows, phase, args, manifest, op)
        }
    }
}

fn measure_dense_case(
    kind: DenseKernelKind,
    rows: usize,
    phase: DensePhase,
    args: &RunArgs,
    manifest: &KernelManifest,
    op: &OpManifest,
) -> CaseResult {
    let mut result = empty_dense_case_result(kind, rows, phase, manifest, op);
    let case_result = (|| -> Result<()> {
        if args.cupti {
            result.cupti = Some(measure_cupti_dense(kind, rows, op)?);
        }
        {
            let mut case = DenseCase::new(kind, rows)?;
            case.pre_measure()?;
            let mut cache_clear = L2CacheClear::new(&case.ctx)?;
            let elapsed = case.measure_cold_l2(args.iters, &mut cache_clear)?;
            result.latency_us = Some(elapsed.as_secs_f64() * 1.0e6 / args.iters as f64);
        }
        Ok(())
    })();

    if let Err(err) = case_result {
        result.error = Some(format!("{err:#}"));
        eprintln!(
            "case failed op={} variant={} rows={rows}: {err:#}",
            op.name,
            kind.label()
        );
    }
    result
}

fn empty_dense_case_result(
    kind: DenseKernelKind,
    rows: usize,
    phase: DensePhase,
    manifest: &KernelManifest,
    op: &OpManifest,
) -> CaseResult {
    let (batch_size, seq_len) = match phase {
        DensePhase::Decode => (rows, None),
        DensePhase::Prefill => (1, Some(rows)),
    };
    let variant = kind.label();
    // Unlike attention, projection/sampling variants are distinct kernels
    // (different shapes / different samplers), not competing implementations
    // of one workload — fold them into the selector so `build_selections`
    // never picks the cheapest shape as "the winner" and drops the rest.
    let mut selector = match phase {
        DensePhase::Decode => json!({ "dtype": "bf16", "batch_size": batch_size }),
        DensePhase::Prefill => prefill_selector_key("bf16", 1, rows),
    };
    if matches!(
        kind,
        DenseKernelKind::ProjectionGemm(_) | DenseKernelKind::Sampling { .. }
    ) {
        selector["kernel"] = json!(variant);
    }
    let selector_key = selector;
    let case_id_payload = format!(
        "{}|{}|{}|{}|{}",
        manifest.model, PARALLEL_STRATEGY, op.name, selector_key, variant
    );
    let (gemm_out_dim, gemm_in_dim) = match kind {
        DenseKernelKind::ProjectionGemm(projection) => {
            let (out_dim, in_dim) = projection.out_in();
            (Some(out_dim), Some(in_dim))
        }
        _ => (None, None),
    };
    CaseResult {
        case_id: format!("sha256:{}", sha256_short(case_id_payload.as_bytes())),
        op: op.name.clone(),
        variant,
        selector_key,
        shape_source: SHAPE_SOURCE.to_string(),
        shape: CaseShape {
            batch_size,
            kv_len: None,
            seq_len,
            num_qo_heads: NUM_QO_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
            page_size: PAGE_SIZE,
            dtype: "bf16".to_string(),
        },
        params: CaseParams {
            chunk_tokens: None,
            max_chunks_per_request: None,
            actual_chunk_size: None,
            active_chunks_per_request: None,
            padded_slots: None,
            cta_tile_q: None,
            gemm_out_dim,
            gemm_in_dim,
        },
        latency_us: None,
        cupti: None,
        error: None,
    }
}

fn measure_cupti_dense(
    kind: DenseKernelKind,
    rows: usize,
    op: &OpManifest,
) -> Result<BTreeMap<String, f64>> {
    let case = RefCell::new(DenseCase::new(kind, rows)?);
    case.borrow_mut().pre_measure()?;

    let mut cache_clear = L2CacheClear::new(&case.borrow().ctx)?;
    let context = case.borrow().cu_context_ptr();
    let device_ordinal = case.borrow().ctx.device_ordinal;
    let clear_ctx = case.borrow().ctx.clone();
    let range_name = format!("qd/{}-{}/r{rows}", op.name, kind.label());

    let mut prepare = || {
        cache_clear
            .clear(&clear_ctx)
            .map_err(|err| format!("{err:#}"))
    };
    let mut launch = || {
        case.borrow_mut()
            .launch_once()
            .map_err(|err| format!("{err:#}"))
    };
    let values = unsafe {
        profile_range_with_prepare(
            context,
            device_ordinal,
            &range_name,
            CUPTI_METRICS,
            Some(&mut prepare),
            &mut launch,
        )?
    };
    Ok(cupti_metric_map(&values))
}

fn measure_decode_case(
    spec: AttentionKernelSpec,
    args: &RunArgs,
    manifest: &KernelManifest,
    op: &OpManifest,
) -> CaseResult {
    let variant = spec.variant.label();
    let path = spec.variant.decode_path();
    let mut result = empty_decode_case_result(spec, manifest, op);
    let case_result = (|| -> Result<()> {
        if args.cupti {
            result.cupti = Some(measure_cupti_decode(spec, path)?);
        }

        {
            let mut case = AttentionDecodeCase::for_spec(spec)?;
            case.launch_once(path)?;
            case.ctx.sync()?;
            let mut cache_clear = L2CacheClear::new(&case.ctx)?;
            let elapsed = case.measure_decode_only_cold_l2(args.iters, path, &mut cache_clear)?;
            result.latency_us = Some(elapsed.as_secs_f64() * 1.0e6 / args.iters as f64);
        }
        Ok(())
    })();

    if let Err(err) = case_result {
        result.error = Some(format!("{err:#}"));
        eprintln!(
            "case failed variant={} bs={} ctx={}: {err:#}",
            variant, spec.shape.batch_size, spec.shape.kv_len
        );
    }
    result
}

fn measure_prefill_case(
    spec: PrefillAttentionSpec,
    stage: PrefillStage,
    args: &RunArgs,
    manifest: &KernelManifest,
    op: &OpManifest,
) -> CaseResult {
    let mut result = empty_prefill_case_result(spec, manifest, op);
    let case_result = (|| -> Result<()> {
        if args.cupti {
            result.cupti = Some(measure_cupti_prefill(spec, stage)?);
        }

        {
            let mut case = AttentionPrefillCase::for_spec(spec)?;
            case.pre_measure_stage(stage)?;
            let mut cache_clear = L2CacheClear::new(&case.ctx)?;
            let elapsed = case.measure_stage_cold_l2(args.iters, stage, &mut cache_clear)?;
            result.latency_us = Some(elapsed.as_secs_f64() * 1.0e6 / args.iters as f64);
        }
        Ok(())
    })();

    if let Err(err) = case_result {
        result.error = Some(format!("{err:#}"));
        eprintln!(
            "case failed stage={} variant={} bs={} seq={}: {err:#}",
            stage.label(),
            spec.variant.label(),
            spec.shape.batch_size,
            spec.shape.seq_len
        );
    }
    result
}

fn measure_single_prefill_case(
    spec: PrefillAttentionSpec,
    args: &RunArgs,
    manifest: &KernelManifest,
    op: &OpManifest,
) -> CaseResult {
    let mut result = empty_prefill_case_result(spec, manifest, op);
    let case_result = (|| -> Result<()> {
        if args.cupti {
            result.cupti = Some(measure_cupti_single_prefill(spec)?);
        }

        {
            let mut case = SinglePrefillCase::for_spec(spec)?;
            case.pre_measure()?;
            let mut cache_clear = L2CacheClear::new(&case.ctx)?;
            let elapsed = case.measure_cold_l2(args.iters, &mut cache_clear)?;
            result.latency_us = Some(elapsed.as_secs_f64() * 1.0e6 / args.iters as f64);
        }
        Ok(())
    })();

    if let Err(err) = case_result {
        result.error = Some(format!("{err:#}"));
        eprintln!(
            "case failed op={} variant=default bs={} seq={}: {err:#}",
            op.name, spec.shape.batch_size, spec.shape.seq_len
        );
    }
    result
}

fn empty_decode_case_result(
    spec: AttentionKernelSpec,
    manifest: &KernelManifest,
    op: &OpManifest,
) -> CaseResult {
    let params = match spec.variant {
        AttentionKernelVariant::NonPartition => CaseParams {
            chunk_tokens: None,
            max_chunks_per_request: None,
            actual_chunk_size: None,
            active_chunks_per_request: None,
            padded_slots: None,
            cta_tile_q: None,
            gemm_out_dim: None,
            gemm_in_dim: None,
        },
        AttentionKernelVariant::SplitKv(config) => CaseParams {
            chunk_tokens: Some(config.chunk_tokens),
            max_chunks_per_request: Some(config.max_chunks_per_request),
            actual_chunk_size: Some(config.actual_chunk_size(spec.shape.kv_len)),
            active_chunks_per_request: Some(config.active_chunks(spec.shape.kv_len)),
            padded_slots: Some(spec.shape.batch_size * config.max_chunks_per_request),
            cta_tile_q: None,
            gemm_out_dim: None,
            gemm_in_dim: None,
        },
    };
    let selector_key = decode_selector_key("bf16", spec.shape.batch_size, spec.shape.kv_len);
    let variant = spec.variant.label();
    let case_id_payload = format!(
        "{}|{}|{}|{}|{}",
        manifest.model, PARALLEL_STRATEGY, op.name, selector_key, variant
    );
    CaseResult {
        case_id: format!("sha256:{}", sha256_short(case_id_payload.as_bytes())),
        op: op.name.clone(),
        variant,
        selector_key,
        shape_source: SHAPE_SOURCE.to_string(),
        shape: CaseShape {
            batch_size: spec.shape.batch_size,
            kv_len: Some(spec.shape.kv_len),
            seq_len: None,
            num_qo_heads: NUM_QO_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
            page_size: PAGE_SIZE,
            dtype: "bf16".to_string(),
        },
        params,
        latency_us: None,
        cupti: None,
        error: None,
    }
}

fn empty_prefill_case_result(
    spec: PrefillAttentionSpec,
    manifest: &KernelManifest,
    op: &OpManifest,
) -> CaseResult {
    let selector_key = prefill_selector_key("bf16", spec.shape.batch_size, spec.shape.seq_len);
    let variant = spec.variant.label();
    let case_id_payload = format!(
        "{}|{}|{}|{}|{}",
        manifest.model, PARALLEL_STRATEGY, op.name, selector_key, variant
    );
    CaseResult {
        case_id: format!("sha256:{}", sha256_short(case_id_payload.as_bytes())),
        op: op.name.clone(),
        variant,
        selector_key,
        shape_source: SHAPE_SOURCE.to_string(),
        shape: CaseShape {
            batch_size: spec.shape.batch_size,
            kv_len: None,
            seq_len: Some(spec.shape.seq_len),
            num_qo_heads: NUM_QO_HEADS,
            num_kv_heads: NUM_KV_HEADS,
            head_dim: HEAD_DIM,
            page_size: PAGE_SIZE,
            dtype: "bf16".to_string(),
        },
        params: CaseParams {
            chunk_tokens: None,
            max_chunks_per_request: None,
            actual_chunk_size: None,
            active_chunks_per_request: None,
            padded_slots: None,
            cta_tile_q: match spec.variant {
                PrefillAttentionVariant::Default => None,
                PrefillAttentionVariant::CtaTileQ(tile_q) => Some(tile_q),
            },
            gemm_out_dim: None,
            gemm_in_dim: None,
        },
        latency_us: None,
        cupti: None,
        error: None,
    }
}

fn decode_selector_key(dtype: &str, batch_size: usize, kv_len: usize) -> Value {
    json!({
        "dtype": dtype,
        "batch_size": batch_size,
        "kv_len": kv_len,
    })
}

fn prefill_selector_key(dtype: &str, batch_size: usize, seq_len: usize) -> Value {
    json!({
        "dtype": dtype,
        "batch_size": batch_size,
        "seq_len": seq_len,
    })
}

fn build_selections(cases: &[CaseResult]) -> Vec<VariantSelection> {
    let mut grouped: BTreeMap<String, Vec<&CaseResult>> = BTreeMap::new();
    for case in cases {
        let key = format!("{}|{}", case.op, selector_key_string(&case.selector_key));
        grouped.entry(key).or_default().push(case);
    }

    let mut selections = Vec::with_capacity(grouped.len());
    for cases in grouped.values() {
        let best = cases
            .iter()
            .copied()
            .filter(|case| case.error.is_none())
            .filter_map(|case| case.latency_us.map(|latency| (latency, case)))
            .min_by(|(left, _), (right, _)| left.total_cmp(right))
            .map(|(_, case)| case)
            .or_else(|| cases.first().copied());
        if let Some(best) = best {
            selections.push(VariantSelection {
                case_id: best.case_id.clone(),
                op: best.op.clone(),
                selector_key: best.selector_key.clone(),
                shape: best.shape.clone(),
                selected_variant: best.variant.clone(),
                latency_us: best.latency_us,
            });
        }
    }
    selections
}

fn measure_cupti_decode(
    spec: AttentionKernelSpec,
    path: DecodePath,
) -> Result<BTreeMap<String, f64>> {
    let mut case = AttentionDecodeCase::for_spec(spec)?;
    case.launch_once(path)?;
    case.ctx.sync()?;

    let mut cache_clear = L2CacheClear::new(&case.ctx)?;
    let context = case.cu_context_ptr();
    let device_ordinal = case.ctx.device_ordinal;
    let clear_ctx = case.ctx.clone();
    let range_name = format!(
        "qk/{}/b{}/k{}",
        path.name(case.split_config()),
        case.shape().batch_size,
        case.shape().kv_len
    );

    let mut prepare = || {
        cache_clear
            .clear(&clear_ctx)
            .map_err(|err| format!("{err:#}"))
    };
    let mut launch = || case.launch_once(path).map_err(|err| format!("{err:#}"));
    let values = unsafe {
        profile_range_with_prepare(
            context,
            device_ordinal,
            &range_name,
            CUPTI_METRICS,
            Some(&mut prepare),
            &mut launch,
        )?
    };

    Ok(cupti_metric_map(&values))
}

fn measure_cupti_prefill(
    spec: PrefillAttentionSpec,
    stage: PrefillStage,
) -> Result<BTreeMap<String, f64>> {
    let case = RefCell::new(AttentionPrefillCase::for_spec(spec)?);
    case.borrow_mut().pre_measure_stage(stage)?;

    let mut cache_clear = L2CacheClear::new(&case.borrow().ctx)?;
    let context = case.borrow().cu_context_ptr();
    let device_ordinal = case.borrow().ctx.device_ordinal;
    let clear_ctx = case.borrow().ctx.clone();
    let range_name = format!(
        "qpf/{}-{}/b{}/s{}",
        stage.range_label(),
        spec.variant.range_label(),
        case.borrow().shape().batch_size,
        case.borrow().shape().seq_len
    );

    let mut prepare = || {
        case.borrow_mut()
            .prepare_stage(stage)
            .map_err(|err| format!("{err:#}"))?;
        cache_clear
            .clear(&clear_ctx)
            .map_err(|err| format!("{err:#}"))
    };
    let mut launch = || {
        case.borrow_mut()
            .launch_stage(stage)
            .map_err(|err| format!("{err:#}"))
    };
    let values = unsafe {
        profile_range_with_prepare(
            context,
            device_ordinal,
            &range_name,
            CUPTI_METRICS,
            Some(&mut prepare),
            &mut launch,
        )?
    };

    Ok(cupti_metric_map(&values))
}

fn measure_cupti_single_prefill(spec: PrefillAttentionSpec) -> Result<BTreeMap<String, f64>> {
    let case = RefCell::new(SinglePrefillCase::for_spec(spec)?);
    case.borrow_mut().pre_measure()?;

    let mut cache_clear = L2CacheClear::new(&case.borrow().ctx)?;
    let context = case.borrow().cu_context_ptr();
    let device_ordinal = case.borrow().ctx.device_ordinal;
    let clear_ctx = case.borrow().ctx.clone();
    let range_name = format!(
        "qpf/single/b{}/s{}",
        case.borrow().shape().batch_size,
        case.borrow().shape().seq_len
    );

    let mut prepare = || {
        cache_clear
            .clear(&clear_ctx)
            .map_err(|err| format!("{err:#}"))
    };
    let mut launch = || {
        case.borrow_mut()
            .launch_once()
            .map_err(|err| format!("{err:#}"))
    };
    let values = unsafe {
        profile_range_with_prepare(
            context,
            device_ordinal,
            &range_name,
            CUPTI_METRICS,
            Some(&mut prepare),
            &mut launch,
        )?
    };

    Ok(cupti_metric_map(&values))
}

fn cupti_metric_map(values: &[f64]) -> BTreeMap<String, f64> {
    CUPTI_METRICS
        .iter()
        .copied()
        .zip(values.iter().copied())
        .map(|(metric, value)| (metric.to_string(), value))
        .collect()
}

fn query_external_provenance() -> ExternalProvenance {
    ExternalProvenance {
        git: GitProvenance {
            commit: None,
            dirty: None,
        },
        build: BuildProvenance {
            target_sm_env: std::env::var("PEGAINFER_CUDA_SM").ok(),
            flashinfer_commit: None,
            kernel_archive: None,
            kernel_archive_fnv1a64: None,
        },
        driver_version: None,
        cuda_toolkit: None,
    }
}

fn query_hardware(
    ctx: &DeviceContext,
    peak: &DevicePeakBandwidth,
    driver_version: Option<String>,
    cuda_toolkit: Option<String>,
) -> Result<HardwareProvenance> {
    let l2_bytes = ctx
        .ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)?
        as usize;
    let (cc_major, cc_minor) = ctx.ctx.compute_capability()?;
    Ok(HardwareProvenance {
        gpu_name: ctx.ctx.name()?,
        device_ordinal: ctx.device_ordinal,
        compute_capability: format!("{cc_major}.{cc_minor}"),
        driver_version,
        cuda_toolkit,
        memory_clock_khz: peak.memory_clock_khz,
        memory_bus_width_bits: peak.memory_bus_width_bits,
        peak_gb_s: peak.peak_gb_per_sec(),
        l2_bytes,
        cache_clear_bytes: cache_clear_bytes(l2_bytes),
    })
}

fn read_snapshot(path: &Path) -> Result<KernelSnapshot> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read snapshot {}", path.display()))?;
    serde_json::from_str(&content).with_context(|| format!("failed to parse {}", path.display()))
}

fn compare_snapshots(args: &CompareArgs) -> Result<()> {
    let base = read_snapshot(&args.base)?;
    let new = read_snapshot(&args.new)?;
    let thresholds = new.thresholds;
    let base_cases: HashMap<_, _> = base
        .cases
        .iter()
        .map(|case| (case_key(case), case))
        .collect();
    let mut warnings = 0usize;
    let mut failures = 0usize;

    for case in &new.cases {
        let key = case_key(case);
        let Some(base_case) = base_cases.get(&key) else {
            println!("WARN new case {key}");
            warnings += 1;
            continue;
        };
        compare_latency(
            &key,
            "latency_us",
            base_case.latency_us,
            case.latency_us,
            thresholds,
            &mut warnings,
            &mut failures,
        );
    }

    println!("kernel report compare complete: warnings={warnings} failures={failures}");
    if failures > 0 {
        bail!("{failures} kernel snapshot comparison failures");
    }
    Ok(())
}

fn case_key(case: &CaseResult) -> String {
    if case.selector_key.is_null() {
        let length = case
            .shape
            .kv_len
            .or(case.shape.seq_len)
            .map_or_else(|| "unknown".to_string(), |value| value.to_string());
        return format!(
            "{}|{}|bs{}|len{}",
            case.op, case.variant, case.shape.batch_size, length
        );
    }
    format!(
        "{}|{}|{}",
        case.op,
        case.variant,
        selector_key_string(&case.selector_key)
    )
}

fn selector_key_string(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "null".to_string())
}

fn compare_latency(
    key: &str,
    metric: &str,
    base: Option<f64>,
    new: Option<f64>,
    thresholds: RegressionThresholds,
    warnings: &mut usize,
    failures: &mut usize,
) {
    let (Some(base), Some(new)) = (base, new) else {
        println!("WARN missing {metric} {key}");
        *warnings += 1;
        return;
    };
    if base <= 0.0 {
        return;
    }
    let pct = (new / base - 1.0) * 100.0;
    if pct > thresholds.latency_fail_pct {
        println!("FAIL {metric} {key}: {base:.3} -> {new:.3} ({pct:+.2}%)");
        *failures += 1;
    } else if pct > thresholds.latency_warn_pct {
        println!("WARN {metric} {key}: {base:.3} -> {new:.3} ({pct:+.2}%)");
        *warnings += 1;
    }
}

fn compose_report(loaded: &LoadedManifest, args: &ComposeArgs) -> Result<CompositionReport> {
    anyhow::ensure!(
        args.composition == DEFAULT_COMPOSITION,
        "only `{DEFAULT_COMPOSITION}` composition is currently supported"
    );
    let op = op_manifest(&loaded.manifest, DEFAULT_OP)?;
    let snapshots = args
        .inputs
        .iter()
        .map(|path| read_snapshot(path).map(|snapshot| (path, snapshot)))
        .collect::<Result<Vec<_>>>()?;
    let mut by_op: HashMap<String, &KernelSnapshot> = HashMap::new();
    for (_, snapshot) in &snapshots {
        by_op.insert(snapshot.op.clone(), snapshot);
    }

    let selector = decode_selector_key("bf16", args.batch_size, args.kv_len);
    let repeat_count = QWEN3_LAYER_COUNT;
    let composed_op = if let Some(snapshot) = by_op.get(DEFAULT_OP) {
        let selections = if snapshot.selections.is_empty() {
            build_selections(&snapshot.cases)
        } else {
            snapshot.selections.clone()
        };
        let selection = selections
            .iter()
            .find(|selection| selection.op == DEFAULT_OP && selection.selector_key == selector);
        if let Some(selection) = selection {
            ComposedOpResult {
                op: op.name.clone(),
                selected_variant: Some(selection.selected_variant.clone()),
                selector_key: selector.clone(),
                repeat_count,
                single_latency_us: selection.latency_us,
                total_latency_us: selection
                    .latency_us
                    .map(|latency| latency * repeat_count as f64),
                share_pct: None,
                error: None,
            }
        } else {
            ComposedOpResult {
                op: op.name.clone(),
                selected_variant: None,
                selector_key: selector.clone(),
                repeat_count,
                single_latency_us: None,
                total_latency_us: None,
                share_pct: None,
                error: Some("selector_key not found in op report".to_string()),
            }
        }
    } else {
        ComposedOpResult {
            op: op.name.clone(),
            selected_variant: None,
            selector_key: selector,
            repeat_count,
            single_latency_us: None,
            total_latency_us: None,
            share_pct: None,
            error: Some("missing input op report".to_string()),
        }
    };
    let mut composed_ops = vec![composed_op];

    let total_latency_us =
        sum_optional(composed_ops.iter().map(|op| op.total_latency_us).collect());
    if let Some(total) = total_latency_us {
        for op in &mut composed_ops {
            op.share_pct = op.total_latency_us.map(|latency| latency / total * 100.0);
        }
    }

    Ok(CompositionReport {
        schema: 3,
        report_type: "composition_report".to_string(),
        model: loaded.manifest.model.clone(),
        parallel_strategy: PARALLEL_STRATEGY.to_string(),
        phase: "decode".to_string(),
        composition: args.composition.clone(),
        selector_key: decode_selector_key("bf16", args.batch_size, args.kv_len),
        source_op_reports: snapshots
            .iter()
            .map(|(path, _)| path.display().to_string())
            .collect(),
        total_latency_us,
        ops: composed_ops,
        coverage_note: Some(COMPOSITION_COVERAGE_NOTE.to_string()),
    })
}

const BF16_BYTES: f64 = 2.0;

/// Minimum DRAM traffic and FLOPs the op must perform at its semantics —
/// the classic roofline inputs. Any real kernel moves at least these bytes,
/// so `sol_pct` is a true upper bound on remaining headroom regardless of how
/// the kernel is written.
struct AnalyticModel {
    min_bytes: f64,
    flops: f64,
}

fn analytic_model(case: &CaseResult) -> Option<AnalyticModel> {
    let shape = &case.shape;
    let q_dim = (shape.num_qo_heads * shape.head_dim) as f64;
    let kv_dim = (shape.num_kv_heads * shape.head_dim) as f64;
    let head_dim = shape.head_dim as f64;
    let bs = shape.batch_size as f64;
    let hidden = HIDDEN_SIZE as f64;
    let inter = INTERMEDIATE_SIZE as f64;
    let vocab = VOCAB_SIZE as f64;

    if let Some((family, phase)) = dense_op_for(&case.op) {
        let rows = match phase {
            DensePhase::Decode => bs,
            DensePhase::Prefill => shape.seq_len? as f64,
        };
        let model = match family {
            DenseOpFamily::ProjectionGemm => {
                let m = case.params.gemm_out_dim? as f64;
                let k = case.params.gemm_in_dim? as f64;
                AnalyticModel {
                    min_bytes: BF16_BYTES * (m * k + k * rows + m * rows),
                    flops: 2.0 * m * k * rows,
                }
            }
            DenseOpFamily::RmsNorm => AnalyticModel {
                min_bytes: BF16_BYTES * (2.0 * hidden * rows + hidden),
                flops: 0.0,
            },
            // reads hidden + residual, writes the rounded residual sum back
            // into hidden and the normed result into out
            DenseOpFamily::FusedAddRmsNorm => AnalyticModel {
                min_bytes: BF16_BYTES * (4.0 * hidden * rows + hidden),
                flops: 0.0,
            },
            // q and k are read and written in place; cos/sin read per row.
            // The 2*head_dim cache term reflects the duplicated-half rope
            // cache production ships; a compact cache could halve it, so this
            // bound is ~2.5% above the theoretical floor for this op.
            DenseOpFamily::QkNormRope => AnalyticModel {
                min_bytes: BF16_BYTES * rows * (2.0 * (q_dim + kv_dim) + 2.0 * head_dim),
                flops: 0.0,
            },
            DenseOpFamily::SiluMul => AnalyticModel {
                min_bytes: BF16_BYTES * 3.0 * inter * rows,
                flops: 0.0,
            },
            DenseOpFamily::Embedding => AnalyticModel {
                min_bytes: BF16_BYTES * 2.0 * hidden * rows + 4.0 * rows,
                flops: 0.0,
            },
            DenseOpFamily::Sampling => AnalyticModel {
                min_bytes: BF16_BYTES * vocab * rows,
                flops: 0.0,
            },
        };
        return Some(model);
    }

    match case.op.as_str() {
        DEFAULT_OP => {
            let kv_len = shape.kv_len? as f64;
            Some(AnalyticModel {
                // q read + out write, plus the K and V history each request scans
                min_bytes: BF16_BYTES * bs * (2.0 * q_dim + 2.0 * kv_len * kv_dim),
                flops: 4.0 * q_dim * kv_len * bs,
            })
        }
        PREFILL_QK_NORM_ROPE_OP => {
            let tokens = bs * shape.seq_len? as f64;
            Some(AnalyticModel {
                min_bytes: BF16_BYTES * tokens * (2.0 * (q_dim + kv_dim) + 2.0 * head_dim),
                flops: 0.0,
            })
        }
        PREFILL_KV_SCATTER_OP => {
            let tokens = bs * shape.seq_len? as f64;
            Some(AnalyticModel {
                min_bytes: BF16_BYTES * 4.0 * kv_dim * tokens,
                flops: 0.0,
            })
        }
        PREFILL_OP | PREFILL_ATTENTION_CORE_OP | SINGLE_PREFILL_ATTENTION_CORE_OP => {
            let seq = shape.seq_len? as f64;
            let tokens = bs * seq;
            Some(AnalyticModel {
                min_bytes: BF16_BYTES * tokens * 2.0 * (q_dim + kv_dim),
                // causal QK^T + PV: 2 GEMM-like passes over ~seq/2 keys per query
                flops: 2.0 * q_dim * seq * tokens,
            })
        }
        _ => None,
    }
}

#[derive(Serialize)]
struct RankedKernel {
    op: String,
    variant: String,
    selector_key: Value,
    latency_us: f64,
    min_gb: f64,
    gflops: f64,
    arithmetic_intensity: Option<f64>,
    sol_time_us: f64,
    sol_pct: f64,
    binding_bound: String,
    achieved_dram_gb_s: Option<f64>,
    achieved_dram_pct: Option<f64>,
    tensor_pipe_pct: Option<f64>,
}

#[derive(Serialize)]
struct RankReport {
    schema: u32,
    report_type: String,
    peak_gb_s: f64,
    peak_bf16_tflops: Option<f64>,
    source_op_reports: Vec<String>,
    kernels: Vec<RankedKernel>,
    skipped: Vec<String>,
}

fn rank_snapshots(args: &RankArgs) -> Result<RankReport> {
    let snapshots = args
        .inputs
        .iter()
        .map(|path| read_snapshot(path).map(|snapshot| (path, snapshot)))
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(!snapshots.is_empty(), "rank needs at least one snapshot");
    let peak_gb_s = snapshots[0].1.hardware.peak_gb_s;
    let gpu_name = snapshots[0].1.hardware.gpu_name.as_str();
    for (path, snapshot) in &snapshots {
        anyhow::ensure!(
            snapshot.hardware.gpu_name == gpu_name,
            "snapshot {} was taken on different hardware ({} vs {})",
            path.display(),
            snapshot.hardware.gpu_name,
            gpu_name
        );
        // Old snapshots predate case_id/selector_key; ranking them would
        // silently collapse whole sweeps into one bogus winner.
        for case in &snapshot.cases {
            anyhow::ensure!(
                !case.case_id.is_empty() && !case.selector_key.is_null(),
                "snapshot {} has cases without case_id/selector_key — \
                 regenerate it with the current report tool before ranking",
                path.display()
            );
        }
    }

    let mut kernels = Vec::new();
    let mut skipped = Vec::new();
    for (path, snapshot) in &snapshots {
        let selections = if snapshot.selections.is_empty() {
            build_selections(&snapshot.cases)
        } else {
            snapshot.selections.clone()
        };
        let by_id: HashMap<&str, &CaseResult> = snapshot
            .cases
            .iter()
            .map(|case| (case.case_id.as_str(), case))
            .collect();
        for selection in &selections {
            let Some(case) = by_id.get(selection.case_id.as_str()) else {
                bail!(
                    "{}: selection {} has no matching case — corrupt snapshot",
                    path.display(),
                    selection.case_id
                );
            };
            let Some(latency_us) = case.latency_us else {
                skipped.push(format!("{}: no latency", case_key(case)));
                continue;
            };
            let Some(model) = analytic_model(case) else {
                skipped.push(format!(
                    "{}: no analytic model for op `{}` ({})",
                    case_key(case),
                    case.op,
                    path.display()
                ));
                continue;
            };
            let dram_time_s = model.min_bytes / (peak_gb_s * 1.0e9);
            let flop_time_s = args
                .peak_bf16_tflops
                .filter(|_| model.flops > 0.0)
                .map(|tflops| model.flops / (tflops * 1.0e12));
            let sol_time_s = flop_time_s.map_or(dram_time_s, |flop| dram_time_s.max(flop));
            let sol_time_us = sol_time_s * 1.0e6;
            let sol_pct = sol_time_us / latency_us * 100.0;
            let binding_bound = match flop_time_s {
                Some(flop) if flop > dram_time_s => "tensor".to_string(),
                None if model.flops > 0.0 => {
                    "dram (compute bound unknown: pass --peak-bf16-tflops)".to_string()
                }
                _ => "dram".to_string(),
            };

            let cupti = case.cupti.as_ref();
            let duration_s = cupti
                .and_then(|m| m.get("gpu__time_duration.sum"))
                .map(|ns| ns * 1.0e-9)
                .filter(|s| *s > 0.0);
            let achieved_dram_gb_s = cupti
                .and_then(|m| m.get("dram__bytes.sum"))
                .zip(duration_s)
                .map(|(bytes, s)| bytes / s / 1.0e9);
            let achieved_dram_pct = achieved_dram_gb_s.map(|gb| gb / peak_gb_s * 100.0);
            let tensor_pipe_pct = cupti
                .and_then(|m| {
                    m.get(
                        "sm__ops_path_tensor_op_hmma_src_bf16_dst_fp32_sparsity_off.sum.pct_of_peak_sustained_elapsed",
                    )
                })
                .copied();

            kernels.push(RankedKernel {
                op: case.op.clone(),
                variant: case.variant.clone(),
                selector_key: case.selector_key.clone(),
                latency_us,
                min_gb: model.min_bytes / 1.0e9,
                gflops: model.flops / 1.0e9,
                arithmetic_intensity: (model.min_bytes > 0.0 && model.flops > 0.0)
                    .then(|| model.flops / model.min_bytes),
                sol_time_us,
                sol_pct,
                binding_bound,
                achieved_dram_gb_s,
                achieved_dram_pct,
                tensor_pipe_pct,
            });
        }
    }

    kernels.sort_by(|left, right| left.sol_pct.total_cmp(&right.sol_pct));
    if let Some(below) = args.below {
        kernels.retain(|kernel| kernel.sol_pct < below);
    }

    Ok(RankReport {
        schema: 1,
        report_type: "rank_report".to_string(),
        peak_gb_s,
        peak_bf16_tflops: args.peak_bf16_tflops,
        source_op_reports: snapshots
            .iter()
            .map(|(path, _)| path.display().to_string())
            .collect(),
        kernels,
        skipped,
    })
}

fn print_rank_table(report: &RankReport) {
    println!(
        "{:<28} {:<14} {:<30} {:>11} {:>10} {:>8} {:>9} {:>9} {:>8}  bound",
        "op", "variant", "selector", "latency_us", "sol_us", "sol%", "dram_gb/s", "dram%", "hmma%"
    );
    for kernel in &report.kernels {
        let fmt_opt = |value: Option<f64>| {
            value.map_or_else(|| "-".to_string(), |value| format!("{value:.1}"))
        };
        println!(
            "{:<28} {:<14} {:<30} {:>11.2} {:>10.2} {:>7.1}% {:>9} {:>9} {:>8}  {}",
            kernel.op,
            kernel.variant,
            selector_key_string(&kernel.selector_key),
            kernel.latency_us,
            kernel.sol_time_us,
            kernel.sol_pct,
            fmt_opt(kernel.achieved_dram_gb_s),
            fmt_opt(kernel.achieved_dram_pct),
            fmt_opt(kernel.tensor_pipe_pct),
            kernel.binding_bound,
        );
    }
    if !report.skipped.is_empty() {
        eprintln!("skipped {} entries:", report.skipped.len());
        for entry in &report.skipped {
            eprintln!("  {entry}");
        }
    }
}

fn sum_optional(values: Vec<Option<f64>>) -> Option<f64> {
    values
        .into_iter()
        .try_fold(0.0, |total, value| value.map(|latency| total + latency))
}

fn write_json<T: Serialize>(value: &T, out: Option<&Path>) -> Result<()> {
    let json = serde_json::to_string_pretty(value)?;
    match out {
        Some(path) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = File::create(path)?;
            file.write_all(json.as_bytes())?;
            file.write_all(b"\n")?;
            eprintln!("wrote {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}

fn sha256_short(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    hex::encode(&digest[..16])
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let loaded = load_manifest(&cli.manifest)?;
    match cli
        .command
        .unwrap_or_else(|| Command::Run(RunCli::default()))
    {
        Command::Run(run_cli) => {
            let args = build_run_args(&loaded.manifest, run_cli)?;
            let snapshot = run_snapshot(&loaded, &args)?;
            write_json(&snapshot, args.out.as_deref())?;
            let failed = snapshot.cases.iter().any(|case| case.error.is_some());
            if failed {
                bail!("one or more kernel cases failed");
            }
        }
        Command::Compare(args) => compare_snapshots(&args)?,
        Command::Compose(args) => {
            let report = compose_report(&loaded, &args)?;
            write_json(&report, args.out.as_deref())?;
        }
        Command::Rank(args) => {
            let report = rank_snapshots(&args)?;
            print_rank_table(&report);
            if args.out.is_some() {
                write_json(&report, args.out.as_deref())?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense_case(op: &str, variant: &str, batch: usize, seq: Option<usize>) -> CaseResult {
        let (gemm_out_dim, gemm_in_dim) = match variant {
            "lm_head" => (Some(VOCAB_SIZE), Some(HIDDEN_SIZE)),
            _ => (None, None),
        };
        CaseResult {
            case_id: String::new(),
            op: op.to_string(),
            variant: variant.to_string(),
            selector_key: Value::Null,
            shape_source: String::new(),
            shape: CaseShape {
                batch_size: batch,
                kv_len: None,
                seq_len: seq,
                num_qo_heads: NUM_QO_HEADS,
                num_kv_heads: NUM_KV_HEADS,
                head_dim: HEAD_DIM,
                page_size: PAGE_SIZE,
                dtype: "bf16".to_string(),
            },
            params: CaseParams {
                chunk_tokens: None,
                max_chunks_per_request: None,
                actual_chunk_size: None,
                active_chunks_per_request: None,
                padded_slots: None,
                cta_tile_q: None,
                gemm_out_dim,
                gemm_in_dim,
            },
            latency_us: Some(100.0),
            cupti: None,
            error: None,
        }
    }

    #[test]
    fn analytic_model_matches_hand_math() {
        // lm_head at bs=1: weights dominate — 2 * V * H bytes plus the
        // activation column in and logits column out.
        let case = dense_case("decode_projection_gemm", "lm_head", 1, None);
        let model = analytic_model(&case).expect("gemm model");
        let v = VOCAB_SIZE as f64;
        let h = HIDDEN_SIZE as f64;
        assert!((model.min_bytes - 2.0 * (v * h + h + v)).abs() < 1.0);
        assert!((model.flops - 2.0 * v * h).abs() < 1.0);

        // Decode attention at bs=2, kv=1024: the K and V histories dominate.
        let mut attention = dense_case(DEFAULT_OP, "non_partition", 2, None);
        attention.shape.kv_len = Some(1024);
        let model = analytic_model(&attention).expect("attention model");
        let q_dim = (NUM_QO_HEADS * HEAD_DIM) as f64;
        let kv_dim = (NUM_KV_HEADS * HEAD_DIM) as f64;
        let expected = 2.0 * 2.0 * (2.0 * q_dim + 2.0 * 1024.0 * kv_dim);
        assert!((model.min_bytes - expected).abs() < 1.0);
    }

    #[test]
    fn dense_op_parsing_rejects_prefill_lm_head() {
        let (family, phase) = dense_op_for("prefill_projection_gemm").expect("dense op");
        assert!(parse_dense_variant(family, phase, "lm_head").is_err());
        assert!(parse_dense_variant(family, phase, "q_proj").is_ok());
        assert!(dense_op_for("prefill_sampling").is_none());
        assert!(dense_op_for("decode_qk_norm_rope").is_some());
    }
}
