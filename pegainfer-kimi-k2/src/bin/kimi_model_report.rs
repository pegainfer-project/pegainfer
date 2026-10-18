use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use clap::ValueEnum;
use pegainfer_bench::Accum;
use pegainfer_bench::CallSiteRow;
use pegainfer_bench::RollupRow;
use pegainfer_bench::accumulate;
use pegainfer_bench::call_site_row;
use pegainfer_bench::rollup_row;
use pegainfer_kernels::ops::KIMI_K2_EP_WORLD;
use pegainfer_kernels::tensor::KernelCall;
use pegainfer_kimi_k2::KIMI_K2_LAYERS;
use pegainfer_kimi_k2::batch_decode_trace::MODEL;
use pegainfer_kimi_k2::batch_decode_trace::PHASE_DECODE;
use pegainfer_kimi_k2::batch_decode_trace::TP_WORLD_SIZE;
use pegainfer_kimi_k2::batch_decode_trace::normalize_call_site;
use pegainfer_kimi_k2::batch_decode_trace::trace_decode_kernel_calls;
use pegainfer_kimi_k2::batch_decode_trace::trace_runtime_decode_kernel_calls;
use pegainfer_kimi_k2::kernel_report::LatencyStats;
use pegainfer_kimi_k2::kernel_report::MeasuredCall;
use pegainfer_kimi_k2::kernel_report::bench_key;
use pegainfer_kimi_k2::kernel_report::measure_call;
use serde::Serialize;

const DEFAULT_ITERS: u64 = 16;

#[derive(Parser)]
#[command(about = "Kimi-K2 model-level operator report")]
struct Cli {
    command: String,
    #[arg(long = "batch-size")]
    batch_size: usize,
    #[arg(long = "kv-len")]
    kv_len: usize,
    #[arg(long, default_value = "text")]
    format: String,
    #[arg(long)]
    out: Option<PathBuf>,
    #[arg(long, default_value_t = DEFAULT_ITERS)]
    iters: u64,
    #[arg(long, default_value = "models/Kimi-K2.5")]
    model_path: String,
    #[arg(long, value_enum, default_value_t = TraceSource::Runtime)]
    source: TraceSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum TraceSource {
    Runtime,
    Static,
}

#[derive(Serialize)]
struct ModelReport {
    schema: u32,
    report_type: String,
    model: String,
    phase: String,
    rank_scope: String,
    config: ReportConfig,
    schedule_source: String,
    total_schedule_calls: usize,
    measured_schedule_calls: usize,
    missing_schedule_calls: usize,
    total_measured_us: f64,
    total_p99_us: f64,
    schedule: Vec<KernelCall>,
    by_op: Vec<RollupRow>,
    by_call_site: Vec<CallSiteRow>,
    missing_by_op: Vec<MissingOpRow>,
    coverage: Vec<CoverageRow>,
}

#[derive(Serialize)]
struct ReportConfig {
    batch_size: usize,
    kv_len: usize,
    layers: usize,
    tp_world_size: usize,
    ep_world_size: usize,
    iters: u64,
}

#[derive(Serialize)]
struct CoverageRow {
    call_site: String,
    op: String,
    status: String,
    calls: usize,
    latency: Option<LatencyStats>,
    key: Option<String>,
    reason: Option<String>,
}

#[derive(Serialize)]
struct MissingOpRow {
    op: String,
    calls: usize,
    call_sites: usize,
    reason: String,
}

struct BenchEntry {
    key: String,
    measured: MeasuredCall,
}

#[derive(Default)]
struct MissingAccum {
    calls: usize,
    call_sites: BTreeSet<String>,
    reason: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.command != PHASE_DECODE {
        bail!("only `decode` is implemented; got `{}`", cli.command);
    }
    if cli.batch_size == 0 || cli.kv_len == 0 || cli.iters == 0 {
        bail!("--batch-size, --kv-len, and --iters must be greater than zero");
    }
    if cli.format != "text" && cli.format != "json" {
        bail!("--format must be `text` or `json`");
    }

    let schedule = load_schedule(cli.source, &cli.model_path, cli.batch_size, cli.kv_len)?;
    let catalog = measure_catalog(&schedule, cli.iters)?;
    let report = compose_report(
        cli.batch_size,
        cli.kv_len,
        cli.iters,
        cli.source,
        schedule,
        &catalog,
    )?;
    let out = cli.out.unwrap_or_else(|| {
        PathBuf::from(format!(
            "target/model_reports/{MODEL}/decode-rank0-bs{}-kv{}.json",
            cli.batch_size, cli.kv_len
        ))
    });
    write_json_report(&out, &report)?;
    match cli.format.as_str() {
        "json" => println!("{}", serde_json::to_string_pretty(&report)?),
        "text" => print_text_report(&report, &out),
        _ => unreachable!("format validated"),
    }
    Ok(())
}

fn measure_catalog(calls: &[KernelCall], iters: u64) -> Result<HashMap<String, BenchEntry>> {
    let mut catalog = HashMap::new();
    for call in calls {
        let key = bench_key(call)?;
        if catalog.contains_key(&key) {
            continue;
        }
        let measured = measure_call(call, iters)
            .with_context(|| format!("failed to measure {} ({})", call.label, call.op))?;
        catalog.insert(key.clone(), BenchEntry { key, measured });
    }
    Ok(catalog)
}

fn compose_report(
    batch_size: usize,
    kv_len: usize,
    iters: u64,
    source: TraceSource,
    schedule: Vec<KernelCall>,
    catalog: &HashMap<String, BenchEntry>,
) -> Result<ModelReport> {
    let mut by_op: BTreeMap<String, Accum> = BTreeMap::new();
    let mut by_site: BTreeMap<String, (String, Accum)> = BTreeMap::new();
    let mut missing_by_op: BTreeMap<String, MissingAccum> = BTreeMap::new();
    let mut coverage: BTreeMap<String, CoverageRow> = BTreeMap::new();

    for call in &schedule {
        let key = bench_key(call)?;
        let entry = catalog
            .get(&key)
            .ok_or_else(|| anyhow::anyhow!("missing measured catalog entry for {}", call.label))?;
        let site = normalize_call_site(&call.label);
        let cov_key = format!("{site}::{}", call.op);
        if let Some(stats) = &entry.measured.stats {
            accumulate(by_op.entry(call.op.clone()).or_default(), stats);
            let (_, site_accum) = by_site
                .entry(site.clone())
                .or_insert_with(|| (call.op.clone(), Accum::default()));
            accumulate(site_accum, stats);
            coverage
                .entry(cov_key)
                .or_insert_with(|| CoverageRow {
                    call_site: site,
                    op: call.op.clone(),
                    status: "measured".to_string(),
                    calls: 0,
                    latency: Some(stats.clone()),
                    key: Some(entry.key.clone()),
                    reason: None,
                })
                .calls += 1;
        } else {
            let missing = missing_by_op.entry(call.op.clone()).or_default();
            missing.calls += 1;
            missing.call_sites.insert(site.clone());
            if missing.reason.is_none() {
                missing.reason.clone_from(&entry.measured.reason);
            }
            coverage
                .entry(cov_key)
                .or_insert_with(|| CoverageRow {
                    call_site: site,
                    op: call.op.clone(),
                    status: "missing_provider".to_string(),
                    calls: 0,
                    latency: None,
                    key: Some(entry.key.clone()),
                    reason: entry.measured.reason.clone(),
                })
                .calls += 1;
        }
    }

    let total = by_op.values().map(|row| row.total_us).sum::<f64>();
    let total_p99 = by_op.values().map(|row| row.total_p99_us).sum::<f64>();
    let measured_schedule_calls = by_op.values().map(|row| row.calls).sum::<usize>();
    let missing_schedule_calls = missing_by_op.values().map(|row| row.calls).sum::<usize>();
    let by_op = by_op
        .into_iter()
        .map(|(op, accum)| rollup_row(op, accum, total))
        .collect::<Vec<_>>();
    let by_call_site = by_site
        .into_iter()
        .map(|(call_site, (op, accum))| call_site_row(call_site, op, accum, total))
        .collect::<Vec<_>>();
    let missing_by_op = missing_by_op
        .into_iter()
        .map(|(op, accum)| MissingOpRow {
            op,
            calls: accum.calls,
            call_sites: accum.call_sites.len(),
            reason: accum
                .reason
                .unwrap_or_else(|| "provider missing".to_string()),
        })
        .collect::<Vec<_>>();

    Ok(ModelReport {
        schema: 2,
        report_type: "kimi_model_decode_report".to_string(),
        model: MODEL.to_string(),
        phase: PHASE_DECODE.to_string(),
        rank_scope: "rank0 local compute plus collective placeholders; MoE EP imbalance requires per-rank extension".to_string(),
        config: ReportConfig {
            batch_size,
            kv_len,
            layers: KIMI_K2_LAYERS,
            tp_world_size: TP_WORLD_SIZE,
            ep_world_size: KIMI_K2_EP_WORLD,
            iters,
        },
        schedule_source: match source {
            TraceSource::Runtime => {
                "Kimi runner decode trace via EngineHandle/worker; no HTTP".to_string()
            }
            TraceSource::Static => {
                "Kimi runner worker decode DAG mirror; no HTTP, no prompt/prefill window"
                    .to_string()
            }
        },
        total_schedule_calls: schedule.len(),
        measured_schedule_calls,
        missing_schedule_calls,
        total_measured_us: total,
        total_p99_us: total_p99,
        schedule,
        by_op,
        by_call_site,
        missing_by_op,
        coverage: coverage.into_values().collect(),
    })
}

fn load_schedule(
    source: TraceSource,
    model_path: &str,
    batch_size: usize,
    kv_len: usize,
) -> Result<Vec<KernelCall>> {
    match source {
        TraceSource::Runtime => trace_runtime_decode_kernel_calls(model_path, batch_size, kv_len),
        TraceSource::Static => trace_decode_kernel_calls(model_path, batch_size, kv_len),
    }
}

fn write_json_report(path: &Path, report: &ModelReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    fs::write(path, serde_json::to_string_pretty(report)?)
        .with_context(|| format!("failed to write {}", path.display()))
}

fn print_text_report(report: &ModelReport, out: &Path) {
    println!(
        "Kimi decode rank0 report bs={} kv={} measured_total={:.3}ms measured_p99_sum={:.3}ms",
        report.config.batch_size,
        report.config.kv_len,
        report.total_measured_us / 1000.0,
        report.total_p99_us / 1000.0
    );
    println!("wrote {}", out.display());
    println!("rank_scope: {}", report.rank_scope);
    println!(
        "coverage: measured_calls={} missing_calls={} total_calls={}",
        report.measured_schedule_calls, report.missing_schedule_calls, report.total_schedule_calls
    );
    println!("\nby_op:");
    for row in &report.by_op {
        println!(
            "  {:34} calls={:4} total={:9.3}us per={:8.3}us p99={:8.3}us pct={:6.2}%",
            row.op, row.calls, row.total_us, row.per_call_us, row.p99_us, row.pct
        );
    }
    let missing = report
        .coverage
        .iter()
        .filter(|row| row.status != "measured")
        .count();
    println!("\ncoverage_missing_call_sites={missing}");
    if !report.missing_by_op.is_empty() {
        println!("missing_by_op:");
        for row in &report.missing_by_op {
            println!(
                "  {:34} calls={:4} call_sites={:3} reason={}",
                row.op, row.calls, row.call_sites, row.reason
            );
        }
    }
}
