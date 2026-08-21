use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use clap::Parser;
use comfy_table::Cell;
use comfy_table::Color;
use comfy_table::ContentArrangement;
use comfy_table::Table;
use comfy_table::presets::UTF8_FULL;
use cudarc::driver::CudaSlice;
use cudarc::driver::sys;
use half::bf16;
use pegainfer_bench::Accum;
use pegainfer_bench::CallSiteRow;
use pegainfer_bench::LatencyStats;
use pegainfer_bench::RollupRow;
use pegainfer_bench::accumulate;
use pegainfer_bench::attr_usize;
use pegainfer_bench::axis;
use pegainfer_bench::call_site_row;
use pegainfer_bench::input;
use pegainfer_bench::output;
use pegainfer_bench::rollup_row;
use pegainfer_bench::zero_matrix;
use pegainfer_core::kv_pool::KvLayout;
use pegainfer_core::ops;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::per_token_served;
use pegainfer_kernels::ops::pin_served;
use pegainfer_kernels::ops::reset_numeric_policy_counters;
use pegainfer_kernels::ops::set_numeric_policy;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceVec;
use pegainfer_kernels::tensor::HiddenStates;
use pegainfer_kernels::tensor::KernelCall;
use pegainfer_kernels::tensor::TensorSpec;
use pegainfer_qwen3::batch_decode_trace::HEAD_DIM_VALUE;
use pegainfer_qwen3::batch_decode_trace::KV_DIM_VALUE;
use pegainfer_qwen3::batch_decode_trace::MODEL;
use pegainfer_qwen3::batch_decode_trace::NUM_KV_HEADS;
use pegainfer_qwen3::batch_decode_trace::NUM_LAYERS;
use pegainfer_qwen3::batch_decode_trace::NUM_Q_HEADS;
use pegainfer_qwen3::batch_decode_trace::PHASE_DECODE;
use pegainfer_qwen3::batch_decode_trace::RMS_NORM_EPS;
use pegainfer_qwen3::batch_decode_trace::normalize_call_site;
use pegainfer_qwen3::batch_decode_trace::trace_decode_kernel_calls;
use pegainfer_qwen3::kernel_bench::L2CacheClear;
use pegainfer_qwen3::kernel_bench::build_split_kv_csr;
use serde::Serialize;

const DEFAULT_ITERS: u64 = 32;

#[derive(Parser)]
#[command(about = "Qwen3-4B model-level operator report")]
struct Cli {
    /// Only `decode` is implemented in the MVP.
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
    #[arg(long, default_value = "models/Qwen3-4B")]
    model_path: String,
    /// Numeric policy to trace + measure under, matching production: `tuned` | `pin` | `per-token`.
    #[arg(long, default_value = "tuned")]
    policy: String,
}

#[derive(Clone, Serialize)]
struct ModelReport {
    schema: u32,
    report_type: String,
    model: String,
    phase: String,
    config: ReportConfig,
    schedule_source: String,
    total_measured_us: f64,
    total_p99_us: f64,
    /// True when `total_measured_us` excludes an op (GEMM under Tuned): a partial total.
    total_is_partial: bool,
    /// Projection GEMMs served by the pinned cuBLASLt algo this run: >0 under Pin (every GEMM
    /// serves or the run bails), 0 under Tuned/PerToken.
    pin_served: u64,
    /// Projection GEMMs served by the per-token oracle this run: >0 under PerToken, 0 otherwise.
    per_token_served: u64,
    schedule: Vec<KernelCall>,
    by_op: Vec<RollupRow>,
    by_call_site: Vec<CallSiteRow>,
    coverage: Vec<CoverageRow>,
}

#[derive(Clone, Serialize)]
struct ReportConfig {
    batch_size: usize,
    kv_len: usize,
    layers: usize,
    tp_world_size: usize,
    iters: u64,
    numeric_policy: String,
}

#[derive(Clone, Serialize)]
struct CoverageRow {
    call_site: String,
    op: String,
    status: String,
    calls: usize,
    latency: Option<LatencyStats>,
    key: Option<String>,
}

struct BenchEntry {
    key: String,
    measure: Measure,
}

/// A measured op under this run's policy: `Faithful` carries the latency that feeds the rollup;
/// `Excluded` carries only the reason, so no latency exists to leak into the totals or coverage.
enum Measure {
    Faithful(LatencyStats),
    Excluded(ExcludeReason),
}

#[derive(Clone, Copy)]
enum ExcludeReason {
    /// Tuned projection GEMM: the report's untuned context would run plain GemmEx (not the
    /// production gemm_lt-tuned algo) if measured, so it is classified excluded and skipped.
    UnfaithfulGemmEx,
}

impl ExcludeReason {
    fn status(self) -> &'static str {
        match self {
            ExcludeReason::UnfaithfulGemmEx => "unfaithful_gemmex",
        }
    }
}

fn classify(policy: NumericPolicy, op: &str) -> Option<ExcludeReason> {
    if matches!(op, "gemm" | "gemm_rows") && matches!(policy, NumericPolicy::Tuned) {
        Some(ExcludeReason::UnfaithfulGemmEx)
    } else {
        None
    }
}

fn parse_policy(s: &str) -> Result<NumericPolicy> {
    match s {
        "tuned" => Ok(NumericPolicy::Tuned),
        "pin" => Ok(NumericPolicy::Pin),
        "per-token" => Ok(NumericPolicy::PerToken),
        other => bail!("--policy must be tuned | pin | per-token, got `{other}`"),
    }
}

fn policy_name(p: NumericPolicy) -> &'static str {
    match p {
        NumericPolicy::Tuned => "tuned",
        NumericPolicy::Pin => "pin",
        NumericPolicy::PerToken => "per-token",
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.command != PHASE_DECODE {
        bail!("only `decode` is implemented; got `{}`", cli.command);
    }
    if cli.batch_size == 0 {
        bail!("--batch-size must be greater than zero");
    }
    if cli.kv_len == 0 {
        bail!("--kv-len must be greater than zero");
    }
    if cli.iters == 0 {
        bail!("--iters must be greater than zero");
    }
    if cli.format != "text" && cli.format != "json" {
        bail!("--format must be `text` or `json`");
    }

    let policy = parse_policy(&cli.policy)?;
    set_numeric_policy(policy);

    let schedule = trace_decode_kernel_calls(&cli.model_path, cli.batch_size, cli.kv_len)?;
    reset_numeric_policy_counters();
    let catalog = measure_catalog(&schedule, cli.iters, policy)
        .with_context(|| "failed to build strict measured catalog")?;
    // launch_gemm_pin bails on an un-warmed shape, with no silent fallback.
    let pin_served_count = pin_served();
    if matches!(policy, NumericPolicy::Pin) {
        anyhow::ensure!(
            pin_served_count > 0,
            "no projection GEMM served through the Pin path (served=0); policy didn't take effect or the pins weren't warmed"
        );
    }
    // PerToken must route through the per-token oracle, not silently fall back to GemmEx
    // (which would still report `measured`).
    let per_token_served_count = per_token_served();
    if matches!(policy, NumericPolicy::PerToken) {
        anyhow::ensure!(
            per_token_served_count > 0,
            "no projection GEMM served through the PerToken oracle (served=0); policy didn't take effect"
        );
    }
    let report = compose_report(
        cli.batch_size,
        cli.kv_len,
        cli.iters,
        schedule,
        &catalog,
        policy,
        pin_served_count,
        per_token_served_count,
    )?;
    let out = cli.out.unwrap_or_else(|| {
        // Key the default path on the policy too: pin/tuned/per-token runs differ, so without it a
        // later run silently overwrites an earlier one's JSON/DOT.
        PathBuf::from(format!(
            "target/model_reports/{MODEL}/decode-{}-bs{}-kv{}.json",
            policy_name(policy),
            cli.batch_size,
            cli.kv_len
        ))
    });
    write_json_report(&out, &report)?;
    let dot_out = out.with_extension("dot");
    write_dot_report(&dot_out, &report)?;

    match cli.format.as_str() {
        "text" => print_text_report(&report, &out, &dot_out),
        "json" => {
            println!("{}", serde_json::to_string_pretty(&report)?);
            eprintln!("wrote {}", out.display());
            eprintln!("wrote {}", dot_out.display());
        }
        _ => unreachable!("format validated above"),
    }

    Ok(())
}

fn write_json_report(path: &Path, report: &ModelReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(report)?;
    fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))
}

fn write_dot_report(path: &Path, report: &ModelReport) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut site_order = Vec::new();
    for call in &report.schedule {
        let site = normalize_call_site(&call.label);
        if !site_order.contains(&site) {
            site_order.push(site);
        }
    }
    let rows = report
        .by_call_site
        .iter()
        .map(|row| (row.call_site.as_str(), row))
        .collect::<HashMap<_, _>>();
    // Drop sites with no rollup row (e.g. GEMM excluded under Tuned) so the edge chain skips them
    // instead of leaving Graphviz to auto-create phantom nodes for the dangling endpoints.
    site_order.retain(|site| rows.contains_key(site.as_str()));
    let mut dot = String::new();
    dot.push_str("digraph qwen3_decode_operator_report {\n");
    dot.push_str("  rankdir=LR;\n");
    dot.push_str("  graph [fontname=\"Helvetica\"];\n");
    dot.push_str("  node [shape=box, style=\"rounded,filled\", fontname=\"Helvetica\", fillcolor=\"#f8fafc\", color=\"#94a3b8\"];\n");
    dot.push_str("  edge [color=\"#94a3b8\"];\n");
    for site in &site_order {
        let Some(row) = rows.get(site.as_str()) else {
            continue;
        };
        let fill = if row.pct >= 25.0 {
            "#fee2e2"
        } else if row.pct >= 10.0 {
            "#ffedd5"
        } else if row.pct >= 2.0 {
            "#fef9c3"
        } else {
            "#f8fafc"
        };
        writeln!(
            dot,
            "  \"{}\" [fillcolor=\"{}\", label=\"{}\\n{}\\nmean {} · p99/call {} · {:.1}%\"];",
            dot_escape(site),
            fill,
            dot_escape(site),
            dot_escape(&row.op),
            format_us(row.total_us),
            format_us(row.p99_us),
            row.pct
        )?;
    }
    for window in site_order.windows(2) {
        writeln!(
            dot,
            "  \"{}\" -> \"{}\";",
            dot_escape(&window[0]),
            dot_escape(&window[1])
        )?;
    }
    dot.push_str("}\n");
    fs::write(path, dot).with_context(|| format!("failed to write {}", path.display()))
}

fn compose_report(
    batch_size: usize,
    kv_len: usize,
    iters: u64,
    schedule: Vec<KernelCall>,
    catalog: &HashMap<String, BenchEntry>,
    policy: NumericPolicy,
    pin_served_count: u64,
    per_token_served_count: u64,
) -> Result<ModelReport> {
    let mut op_rows: BTreeMap<String, Accum> = BTreeMap::new();
    let mut site_rows: BTreeMap<String, (String, Accum)> = BTreeMap::new();
    let mut coverage_rows: BTreeMap<(String, String), CoverageRow> = BTreeMap::new();
    let mut missing = Vec::new();

    for call in &schedule {
        let site = normalize_call_site(&call.label);
        if is_noop_all_reduce(call) {
            // Counted toward call totals but contributes zero time on a single rank.
            let zero = LatencyStats::zero(iters);
            accumulate(op_rows.entry(call.op.clone()).or_default(), &zero);
            let (_, site_accum) = site_rows
                .entry(site.clone())
                .or_insert_with(|| (call.op.clone(), Accum::default()));
            accumulate(site_accum, &zero);
            coverage_rows
                .entry((site.clone(), call.op.clone()))
                .and_modify(|row| row.calls += 1)
                .or_insert(CoverageRow {
                    call_site: site,
                    op: call.op.clone(),
                    status: "no_op".to_string(),
                    calls: 1,
                    latency: Some(zero),
                    key: None,
                });
            continue;
        }

        let key = bench_key(call)?;
        let Some(entry) = catalog.get(&key) else {
            missing.push(format!(
                "op={} label={} key={key}\n{}",
                call.op,
                call.label,
                describe_call(call)
            ));
            continue;
        };

        if let Measure::Faithful(stats) = &entry.measure {
            accumulate(op_rows.entry(call.op.clone()).or_default(), stats);
            let (_, site_accum) = site_rows
                .entry(site.clone())
                .or_insert_with(|| (call.op.clone(), Accum::default()));
            accumulate(site_accum, stats);
        }
        coverage_rows
            .entry((site.clone(), call.op.clone()))
            .and_modify(|row| row.calls += 1)
            .or_insert(CoverageRow {
                call_site: site,
                op: call.op.clone(),
                status: match &entry.measure {
                    Measure::Faithful(_) => "measured",
                    Measure::Excluded(reason) => reason.status(),
                }
                .to_string(),
                calls: 1,
                latency: match &entry.measure {
                    Measure::Faithful(stats) => Some(stats.clone()),
                    Measure::Excluded(_) => None,
                },
                key: Some(entry.key.clone()),
            });
    }

    if !missing.is_empty() {
        bail!("missing bench result:\n{}", missing.join("\n\n"));
    }

    // Re-derived from the per-op accumulators (no-op all-reduce contributes 0),
    // so totals are summed in op order rather than schedule order, a sub-ULP
    // reshuffle that only touches this untracked target/ report.
    let total = op_rows.values().map(|accum| accum.total_us).sum::<f64>();
    let total_p99 = op_rows
        .values()
        .map(|accum| accum.total_p99_us)
        .sum::<f64>();

    let mut by_op: Vec<_> = op_rows
        .into_iter()
        .map(|(op, accum)| rollup_row(op, accum, total))
        .collect();
    by_op.sort_by(|a, b| b.total_us.total_cmp(&a.total_us).then(a.op.cmp(&b.op)));

    let mut by_call_site: Vec<_> = site_rows
        .into_iter()
        .map(|(call_site, (op, accum))| call_site_row(call_site, op, accum, total))
        .collect();
    by_call_site.sort_by(|a, b| {
        b.total_us
            .total_cmp(&a.total_us)
            .then(a.call_site.cmp(&b.call_site))
    });

    let total_is_partial = catalog
        .values()
        .any(|entry| matches!(entry.measure, Measure::Excluded(_)));

    Ok(ModelReport {
        schema: 3,
        report_type: "model_operator_report".to_string(),
        model: MODEL.to_string(),
        phase: PHASE_DECODE.to_string(),
        config: ReportConfig {
            batch_size,
            kv_len,
            layers: NUM_LAYERS,
            tp_world_size: 1,
            iters,
            numeric_policy: policy_name(policy).to_string(),
        },
        schedule_source:
            "runtime trace: Qwen3Model::batch_decode eager DAG with CUDA Graph disabled".to_string(),
        total_measured_us: total,
        total_p99_us: total_p99,
        total_is_partial,
        pin_served: pin_served_count,
        per_token_served: per_token_served_count,
        schedule,
        by_op,
        by_call_site,
        coverage: coverage_rows.into_values().collect(),
    })
}

fn measure_catalog(
    calls: &[KernelCall],
    iters: u64,
    policy: NumericPolicy,
) -> Result<HashMap<String, BenchEntry>> {
    let mut catalog = HashMap::new();
    for call in calls {
        if is_noop_all_reduce(call) {
            continue;
        }
        let key = bench_key(call)?;
        if catalog.contains_key(&key) {
            continue;
        }
        let measure = match classify(policy, &call.op) {
            Some(reason) => Measure::Excluded(reason),
            None => {
                let stats = measure_call(call, iters).with_context(|| {
                    format!("failed to measure {}\n{}", call.label, describe_call(call))
                })?;
                Measure::Faithful(stats)
            }
        };
        catalog.insert(key.clone(), BenchEntry { key, measure });
    }
    Ok(catalog)
}

fn measure_call(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    if call.op == "paged_decode_attention" {
        return measure_paged_decode_attention(call, iters);
    }
    match call.op.as_str() {
        "embedding_batch" => measure_embedding(call, iters),
        "rms_norm_batch" => measure_rms_norm_batch(call, iters),
        "gemm_rows" => measure_gemm_rows(call, iters),
        "qk_norm_rope_batch_decode" => measure_qk_norm_rope(call, iters),
        "gemm" => measure_gemm(call, iters),
        "fused_add_rms_norm_batch" => measure_fused_add_rms_norm(call, iters),
        "silu_mul_fused_batch" => measure_silu_mul_fused(call, iters),
        other => bail!("no benchmark provider for op `{other}`"),
    }
}

fn measure_embedding(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let weight = input(call, "weight")?;
    let out = output(call, "out")?;
    let vocab = axis(weight, "vocab")?;
    let hidden = axis(weight, "hidden")?;
    let batch = axis(out, "batch")?;
    let ctx = DeviceContext::new()?;
    let embed = zero_matrix(&ctx, vocab, hidden)?;
    let token_ids: CudaSlice<u32> = ctx.stream.alloc_zeros(batch)?;
    let mut out = HiddenStates::zeros(&ctx, hidden, batch)?;
    measure_loop(&ctx, iters, || {
        ops::embedding_batch(&ctx, &embed, &token_ids, &mut out)?;
        Ok(())
    })
}

fn measure_rms_norm_batch(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let x = input(call, "x")?;
    let hidden = axis(x, "hidden")?;
    let batch = axis(x, "batch")?;
    let ctx = DeviceContext::new()?;
    let x = HiddenStates::zeros(&ctx, hidden, batch)?;
    let weight = DeviceVec::zeros(&ctx, hidden)?;
    let mut out = HiddenStates::zeros(&ctx, hidden, batch)?;
    measure_loop(&ctx, iters, || {
        ops::rms_norm_batch_into(&ctx, &x, &weight, RMS_NORM_EPS, &mut out);
        Ok(())
    })
}

fn measure_gemm_rows(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let weight = input(call, "weight")?;
    let x = input(call, "x")?;
    let out = output(call, "out")?;
    let out_total = axis(weight, "out_total")?;
    let in_dim = axis(weight, "in")?;
    let rows = attr_usize(call, "rows")?;
    let row_offset = attr_usize(call, "row_offset")?;
    let batch = axis(x, "batch")?;
    let out_dim = first_axis_size(out)?;
    anyhow::ensure!(out_dim == rows, "gemm_rows output dim must match rows");
    let ctx = DeviceContext::new()?;
    let weight = zero_matrix(&ctx, out_total, in_dim)?;
    let x = HiddenStates::zeros(&ctx, in_dim, batch)?;
    let mut out = HiddenStates::zeros(&ctx, rows, batch)?;
    measure_loop(&ctx, iters, || {
        ops::gemm_rows_into(&ctx, &weight, row_offset, rows, &x, &mut out);
        Ok(())
    })
}

fn measure_gemm(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let weight = input(call, "weight")?;
    let x = input(call, "x")?;
    let out_dim = axis(weight, "out")?;
    let in_dim = axis(weight, "in")?;
    let batch = axis(x, "batch")?;
    let ctx = DeviceContext::new()?;
    let weight = zero_matrix(&ctx, out_dim, in_dim)?;
    let x = HiddenStates::zeros(&ctx, in_dim, batch)?;
    let mut out = HiddenStates::zeros(&ctx, out_dim, batch)?;
    measure_loop(&ctx, iters, || {
        ops::gemm_into(&ctx, &weight, &x, &mut out);
        Ok(())
    })
}

fn measure_fused_add_rms_norm(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let hidden_spec = input(call, "hidden")?;
    let hidden_dim = axis(hidden_spec, "hidden")?;
    let batch = axis(hidden_spec, "batch")?;
    let ctx = DeviceContext::new()?;
    let mut hidden = HiddenStates::zeros(&ctx, hidden_dim, batch)?;
    let residual = HiddenStates::zeros(&ctx, hidden_dim, batch)?;
    let weight = DeviceVec::zeros(&ctx, hidden_dim)?;
    let mut out = HiddenStates::zeros(&ctx, hidden_dim, batch)?;
    measure_loop(&ctx, iters, || {
        ops::fused_add_rms_norm_batch_into(
            &ctx,
            &mut hidden,
            &residual,
            &weight,
            RMS_NORM_EPS,
            &mut out,
        );
        Ok(())
    })
}

fn measure_silu_mul_fused(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let gate_up = input(call, "gate_up")?;
    let out = output(call, "out")?;
    let inter2 = axis(gate_up, "inter2")?;
    let inter = axis(out, "intermediate")?;
    let batch = axis(gate_up, "batch")?;
    anyhow::ensure!(inter2 == 2 * inter, "gate_up axis must be 2 * intermediate");
    let ctx = DeviceContext::new()?;
    let gate_up = HiddenStates::zeros(&ctx, inter2, batch)?;
    let mut out = HiddenStates::zeros(&ctx, inter, batch)?;
    measure_loop(&ctx, iters, || {
        ops::silu_mul_fused_batch_into(&ctx, &gate_up, &mut out)?;
        Ok(())
    })
}

fn measure_qk_norm_rope(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let q = input(call, "q")?;
    let k = input(call, "k")?;
    let q_dim = axis(q, "q_dim")?;
    let kv_dim = axis(k, "kv_dim")?;
    let batch = axis(q, "batch")?;
    let seq = axis(input(call, "cos_cache")?, "seq")?;
    let ctx = DeviceContext::new()?;
    let mut q = HiddenStates::zeros(&ctx, q_dim, batch)?;
    let mut k = HiddenStates::zeros(&ctx, kv_dim, batch)?;
    let q_norm = DeviceVec::zeros(&ctx, HEAD_DIM_VALUE)?;
    let k_norm = DeviceVec::zeros(&ctx, HEAD_DIM_VALUE)?;
    let cos_cache = DeviceVec::zeros(&ctx, seq * HEAD_DIM_VALUE)?;
    let sin_cache = DeviceVec::zeros(&ctx, seq * HEAD_DIM_VALUE)?;
    let positions_host = vec![seq.saturating_sub(1) as i32; batch];
    let positions = ctx.stream.clone_htod(&positions_host)?;
    measure_loop(&ctx, iters, || {
        ops::qk_norm_rope_batch_decode_into(
            &ctx,
            &mut q,
            &mut k,
            0,
            batch,
            &q_norm,
            &k_norm,
            &cos_cache,
            &sin_cache,
            &positions,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM_VALUE,
            RMS_NORM_EPS,
        )?;
        Ok(())
    })
}

fn measure_paged_decode_attention(call: &KernelCall, iters: u64) -> Result<LatencyStats> {
    let q = input(call, "q")?;
    let kv = input(call, "kv_buffer")?;
    let batch = axis(q, "batch")?;
    let q_dim = axis(q, "q_dim")?;
    let total_pages = axis(kv, "page")?;
    let num_layers = axis(kv, "layer")?;
    let kv_heads = axis(kv, "kv_head")?;
    let head_dim = axis(kv, "head_dim")?;
    let page_size = axis(kv, "pos_in_page")?;
    let kv_len = attr_usize(call, "kv_len")?;
    let pages_per_request = kv_len.div_ceil(page_size);
    let layout =
        KvLayout::new(num_layers, kv_heads, head_dim, page_size).expect("kv layout geometry");
    let ctx = DeviceContext::new()?;
    let q = HiddenStates::zeros(&ctx, q_dim, batch)?;
    let k = HiddenStates::zeros(&ctx, KV_DIM_VALUE, batch)?;
    let v = HiddenStates::zeros(&ctx, KV_DIM_VALUE, batch)?;
    let mut out = HiddenStates::zeros(&ctx, q_dim, batch)?;
    let kv_buffer: CudaSlice<bf16> = ctx.stream.alloc_zeros(total_pages * layout.page_stride)?;
    let (page_indices, page_indptr) = page_tables(batch, pages_per_request);
    let last_page_len_value = match kv_len % page_size {
        0 => page_size,
        rem => rem,
    };
    let last_page_len = vec![last_page_len_value as i32; batch];
    let positions = vec![kv_len.saturating_sub(1) as i32; batch];
    let request_indices: Vec<i32> = (0..batch as i32).collect();
    let page_indices_d = ctx.stream.clone_htod(&page_indices)?;
    let page_indptr_d = ctx.stream.clone_htod(&page_indptr)?;
    let last_page_len_d = ctx.stream.clone_htod(&last_page_len)?;
    let positions_d = ctx.stream.clone_htod(&positions)?;
    let request_indices_d = ctx.stream.clone_htod(&request_indices)?;
    let variant = attr_string(call, "variant")?;

    // Replay the split-KV scalars from the recorded attrs; the Pin chunk is fixed off max_context,
    // not the step's kv_len.
    if variant == "split_kv" {
        let split_chunk_size = attr_usize(call, "split_chunk_size")?;
        let cap = attr_usize(call, "split_max_chunks")?;
        let padded_slots = batch * cap;
        let split_csr = build_split_kv_csr(split_chunk_size, cap, &vec![kv_len; batch], batch)?;
        let split_kv_chunk_size = [split_chunk_size as i32];
        let split_request_indices_d = ctx.stream.clone_htod(&split_csr.request_indices)?;
        let split_kv_tile_indices_d = ctx.stream.clone_htod(&split_csr.kv_tile_indices)?;
        let split_kv_chunk_size_d = ctx.stream.clone_htod(&split_kv_chunk_size)?;
        let split_o_indptr_d = ctx.stream.clone_htod(&split_csr.o_indptr)?;
        let split_block_valid_mask_d = ctx.stream.clone_htod(&split_csr.block_valid_mask)?;
        let mut split_tmp_v = ctx.stream.alloc_zeros(padded_slots * q_dim)?;
        let mut split_tmp_s = ctx.stream.alloc_zeros(padded_slots * NUM_Q_HEADS)?;
        let stats = measure_loop(&ctx, iters, || {
            ops::paged_attention_batch_decode_split_kv_into(
                &ctx,
                &q,
                &k,
                &v,
                0,
                &kv_buffer,
                &layout,
                0,
                &page_indices_d,
                &page_indptr_d,
                &last_page_len_d,
                &positions_d,
                &request_indices_d,
                &split_request_indices_d,
                &split_kv_tile_indices_d,
                &split_kv_chunk_size_d,
                &split_o_indptr_d,
                &split_block_valid_mask_d,
                &mut split_tmp_v,
                &mut split_tmp_s,
                padded_slots,
                &mut out,
                NUM_Q_HEADS,
                batch,
            )
        })?;
        Ok(stats)
    } else if variant == "non_partition" {
        let kv_tile_indices = vec![0_i32; batch];
        let kv_chunk_size = vec![kv_len as i32; batch];
        let kv_tile_indices_d = ctx.stream.clone_htod(&kv_tile_indices)?;
        let kv_chunk_size_d = ctx.stream.clone_htod(&kv_chunk_size)?;
        let stats = measure_loop(&ctx, iters, || {
            ops::paged_attention_batch_decode_into(
                &ctx,
                &q,
                &k,
                &v,
                &kv_buffer,
                &layout,
                0,
                &page_indices_d,
                &page_indptr_d,
                &last_page_len_d,
                &positions_d,
                &request_indices_d,
                &kv_tile_indices_d,
                &kv_chunk_size_d,
                &mut out,
                NUM_Q_HEADS,
                batch,
            )
        })?;
        Ok(stats)
    } else {
        bail!("unsupported decode attention variant `{variant}`")
    }
}

fn page_tables(batch: usize, pages_per_request: usize) -> (Vec<i32>, Vec<i32>) {
    let mut page_indices = Vec::with_capacity(batch * pages_per_request);
    let mut page_indptr = Vec::with_capacity(batch + 1);
    page_indptr.push(0);
    for request_idx in 0..batch {
        for page_offset in 0..pages_per_request {
            page_indices.push((request_idx * pages_per_request + page_offset) as i32);
        }
        page_indptr.push(page_indices.len() as i32);
    }
    (page_indices, page_indptr)
}

fn measure_loop(
    ctx: &DeviceContext,
    iters: u64,
    mut launch: impl FnMut() -> Result<()>,
) -> Result<LatencyStats> {
    let mut cache_clear = L2CacheClear::new(ctx)?;
    let start = ctx
        .ctx
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = ctx
        .ctx
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    launch()?;
    ctx.sync()?;
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        cache_clear.clear(ctx)?;
        start.record(&ctx.stream)?;
        launch()?;
        end.record(&ctx.stream)?;
        samples.push(f64::from(start.elapsed_ms(&end)?) * 1.0e3);
    }
    LatencyStats::from_samples(iters, samples)
}

fn first_axis_size(spec: &TensorSpec) -> Result<usize> {
    spec.axes
        .first()
        .map(|axis| axis.size)
        .ok_or_else(|| anyhow!("{} has no axes", spec.compact()))
}

fn attr_string(call: &KernelCall, name: &str) -> Result<String> {
    call.attrs
        .iter()
        .find(|attr| attr.name == name)
        .map(|attr| attr.value.clone())
        .ok_or_else(|| anyhow!("call `{}` missing attr `{name}`", call.label))
}

fn is_noop_all_reduce(call: &KernelCall) -> bool {
    call.op == "all_reduce_hidden"
        && call
            .attrs
            .iter()
            .any(|attr| attr.name == "no_op" && attr.value == "true")
}

fn bench_key(call: &KernelCall) -> Result<String> {
    #[derive(Serialize)]
    struct Key<'a> {
        op: &'a str,
        inputs: &'a [pegainfer_kernels::tensor::TensorArg],
        outputs: &'a [pegainfer_kernels::tensor::TensorArg],
        attrs: BTreeMap<&'a str, &'a str>,
    }

    let attrs = call
        .attrs
        .iter()
        .filter(|attr| attr.name != "no_op" && attr.name != "tp_world_size")
        .map(|attr| (attr.name.as_str(), attr.value.as_str()))
        .collect();
    serde_json::to_string(&Key {
        op: &call.op,
        inputs: &call.inputs,
        outputs: &call.outputs,
        attrs,
    })
    .with_context(|| format!("failed to serialize bench key for {}", call.label))
}

fn describe_call(call: &KernelCall) -> String {
    let inputs = call
        .inputs
        .iter()
        .map(|arg| format!("  in  {}", arg.compact()))
        .collect::<Vec<_>>()
        .join("\n");
    let outputs = call
        .outputs
        .iter()
        .map(|arg| format!("  out {}", arg.compact()))
        .collect::<Vec<_>>()
        .join("\n");
    format!("{inputs}\n{outputs}")
}

fn print_text_report(report: &ModelReport, out: &Path, dot_out: &Path) {
    println!("{MODEL} decode operator report");
    println!(
        "config: bs={} kv_len={} layers={} tp={} iters={} policy={}",
        report.config.batch_size,
        report.config.kv_len,
        report.config.layers,
        report.config.tp_world_size,
        report.config.iters,
        report.config.numeric_policy
    );
    println!("json: {}", out.display());
    println!("dot:  {}", dot_out.display());
    println!("source: {}", report.schedule_source);
    let gemm_excluded = report.total_is_partial;
    println!(
        "total measured: {}{}",
        format_us(report.total_measured_us),
        if gemm_excluded {
            " (PARTIAL: projection GEMM excluded)"
        } else {
            ""
        }
    );
    if gemm_excluded {
        println!(
            "warning: {} policy excludes projection GEMM; totals and %tot are partial. \
             Run --policy pin for a faithful GEMM latency.",
            report.config.numeric_policy
        );
    }
    println!();

    println!("By op");
    let mut by_op = table();
    by_op.set_header(vec![
        "op",
        "calls",
        "mean total",
        "%tot",
        "avg mean",
        "avg std",
        "avg p99",
    ]);
    for row in &report.by_op {
        by_op.add_row(vec![
            Cell::new(&row.op).fg(op_color(row.pct)),
            Cell::new(row.calls),
            Cell::new(format_us(row.total_us)),
            Cell::new(format_pct(row.pct)),
            Cell::new(format_us(row.per_call_us)),
            Cell::new(format_us(row.stddev_us)),
            Cell::new(format_us(row.p99_us)),
        ]);
    }
    println!("{by_op}");

    println!("By call site");
    let mut by_site = table();
    by_site.set_header(vec![
        "call site",
        "op",
        "calls",
        "mean/call",
        "std",
        "p99/call",
        "mean total",
        "%tot",
    ]);
    for row in report.by_call_site.iter().take(16) {
        by_site.add_row(vec![
            Cell::new(&row.call_site).fg(op_color(row.pct)),
            Cell::new(&row.op),
            Cell::new(row.calls),
            Cell::new(format_us(row.per_call_us)),
            Cell::new(format_us(row.stddev_us)),
            Cell::new(format_us(row.p99_us)),
            Cell::new(format_us(row.total_us)),
            Cell::new(format_pct(row.pct)),
        ]);
    }
    println!("{by_site}");

    println!("Coverage");
    let mut coverage = table();
    coverage.set_header(vec!["call site", "op", "status", "calls", "mean", "p99"]);
    for row in &report.coverage {
        let mean = row
            .latency
            .as_ref()
            .map_or_else(|| "-".to_string(), |value| format_us(value.mean_us));
        let p99 = row
            .latency
            .as_ref()
            .map_or_else(|| "-".to_string(), |value| format_us(value.p99_us));
        coverage.add_row(vec![
            Cell::new(&row.call_site),
            Cell::new(&row.op),
            Cell::new(&row.status).fg(status_color(&row.status)),
            Cell::new(row.calls),
            Cell::new(mean),
            Cell::new(p99),
        ]);
    }
    println!("{coverage}");

    println!("Schedule preview");
    let mut preview = table();
    preview.set_header(vec!["label", "op", "args", "first input"]);
    for call in report.schedule.iter().take(14) {
        let first_input = call
            .inputs
            .first()
            .map(pegainfer_core::tensor::TensorArg::compact)
            .map_or_else(|| "-".to_string(), |input| truncate(&input, 86));
        preview.add_row(vec![
            Cell::new(&call.label),
            Cell::new(&call.op),
            Cell::new(format!(
                "{} in / {} out",
                call.inputs.len(),
                call.outputs.len()
            )),
            Cell::new(first_input),
        ]);
    }
    println!("{preview}");
}

fn table() -> Table {
    let mut table = Table::new();
    table
        .load_preset(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table
}

fn op_color(pct: f64) -> Color {
    if pct >= 25.0 {
        Color::Red
    } else if pct >= 10.0 {
        Color::Yellow
    } else {
        Color::White
    }
}

fn status_color(status: &str) -> Color {
    match status {
        "measured" => Color::Green,
        "no_op" => Color::DarkGrey,
        _ => Color::Yellow,
    }
}

fn format_us(value: f64) -> String {
    if value >= 1_000.0 {
        format!("{:.3} ms", value / 1_000.0)
    } else {
        format!("{value:.3} us")
    }
}

fn format_pct(value: f64) -> String {
    format!("{value:.1}%")
}

fn truncate(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut truncated = value
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    truncated.push_str("...");
    truncated
}

fn dot_escape(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
