//! Model-agnostic kernel benchmarking harness.
//!
//! Every model crate's kernel-report tooling needs the same building blocks: a
//! CUDA-event timing loop, latency statistics, accessors over the
//! [`KernelCall`] schedule that the report bins serialize, and the rollup that
//! folds per-call latencies into per-op / per-call-site report rows. Those
//! pieces carry no model knowledge, so they live here and each model crate
//! keeps only its own `measure_*` providers (which call into [`measure_loop`])
//! and its own schedule walk (which feeds [`accumulate`]).

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use cudarc::driver::sys;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::GpuWeight;
use pegainfer_kernels::tensor::KernelCall;
use pegainfer_kernels::tensor::TensorSpec;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct LatencyStats {
    iters: u64,
    pub mean_us: f64,
    stddev_us: f64,
    min_us: f64,
    p50_us: f64,
    p95_us: f64,
    pub p99_us: f64,
    max_us: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct MeasuredCall {
    pub supported: bool,
    pub reason: Option<String>,
    pub stats: Option<LatencyStats>,
}

impl LatencyStats {
    /// All-zero stats for calls that are counted but deliberately not timed
    /// (e.g. a no-op collective on a single rank).
    pub fn zero(iters: u64) -> Self {
        Self {
            iters,
            mean_us: 0.0,
            stddev_us: 0.0,
            min_us: 0.0,
            p50_us: 0.0,
            p95_us: 0.0,
            p99_us: 0.0,
            max_us: 0.0,
        }
    }

    pub fn from_samples(iters: u64, mut samples: Vec<f64>) -> Result<Self> {
        if samples.is_empty() {
            bail!("latency sample set is empty");
        }
        samples.sort_by(f64::total_cmp);
        let mean_us = samples.iter().sum::<f64>() / samples.len() as f64;
        let stddev_us = if samples.len() > 1 {
            let variance = samples
                .iter()
                .map(|sample| {
                    let delta = sample - mean_us;
                    delta * delta
                })
                .sum::<f64>()
                / (samples.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };
        Ok(Self {
            iters,
            mean_us,
            stddev_us,
            min_us: samples[0],
            p50_us: percentile(&samples, 0.50),
            p95_us: percentile(&samples, 0.95),
            p99_us: percentile(&samples, 0.99),
            max_us: samples[samples.len() - 1],
        })
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let idx = ((sorted.len() as f64 - 1.0) * quantile).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Time `launch` over `iters` CUDA-event-bracketed iterations after a 3-shot warmup.
pub fn measure_loop(
    ctx: &DeviceContext,
    iters: u64,
    mut launch: impl FnMut() -> Result<()>,
) -> Result<LatencyStats> {
    if iters == 0 {
        bail!("iters must be greater than zero");
    }
    for _ in 0..3 {
        launch()?;
    }
    ctx.sync()?;
    let start = ctx
        .ctx
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let end = ctx
        .ctx
        .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let mut samples = Vec::with_capacity(iters as usize);
    for _ in 0..iters {
        start.record(&ctx.stream)?;
        launch()?;
        end.record(&ctx.stream)?;
        samples.push(f64::from(start.elapsed_ms(&end)?) * 1_000.0);
    }
    ctx.sync()?;
    LatencyStats::from_samples(iters, samples)
}

/// Stable JSON identity of a [`KernelCall`] — op name plus its input/output/attr shapes.
pub fn bench_key(call: &KernelCall) -> Result<String> {
    Ok(serde_json::to_string(&serde_json::json!({
        "op": call.op,
        "inputs": call.inputs,
        "outputs": call.outputs,
        "attrs": call.attrs,
    }))?)
}

pub fn axis(spec: &TensorSpec, name: &str) -> Result<usize> {
    spec.axes
        .iter()
        .find(|axis| axis.name == name)
        .map(|axis| axis.size)
        .ok_or_else(|| anyhow!("missing axis `{name}` in {}", spec.compact()))
}

pub fn input<'a>(call: &'a KernelCall, name: &str) -> Result<&'a TensorSpec> {
    call.inputs
        .iter()
        .find(|arg| arg.name == name)
        .map(|arg| &arg.spec)
        .ok_or_else(|| anyhow!("{} missing input `{name}`", call.label))
}

pub fn output<'a>(call: &'a KernelCall, name: &str) -> Result<&'a TensorSpec> {
    call.outputs
        .iter()
        .find(|arg| arg.name == name)
        .map(|arg| &arg.spec)
        .ok_or_else(|| anyhow!("{} missing output `{name}`", call.label))
}

pub fn attr_usize(call: &KernelCall, name: &str) -> Result<usize> {
    call.attrs
        .iter()
        .find(|attr| attr.name == name)
        .ok_or_else(|| anyhow!("{} missing attr `{name}`", call.label))?
        .value
        .parse()
        .map_err(|err| anyhow!("{} invalid attr `{name}`: {err}", call.label))
}

/// Zero-initialized `rows × cols` device matrix, sized straight from a kernel shape.
pub fn zero_matrix(ctx: &DeviceContext, rows: usize, cols: usize) -> Result<DeviceMatrix> {
    Ok(DeviceMatrix {
        data: ctx.stream.alloc_zeros(rows * cols)?,
        rows,
        cols,
    })
}

pub fn zero_weight<const OUT: usize, const IN: usize>(
    ctx: &DeviceContext,
) -> Result<GpuWeight<OUT, IN>> {
    GpuWeight::from_device_matrix(zero_matrix(ctx, OUT, IN)?)
}

// ── Model-report rollup ─────────────────────────────────────────────────
//
// A model report folds per-call latencies into per-op and per-call-site rows,
// each carrying its share of the measured total. The aggregation math is
// identical across models; only the schedule walk that feeds it (how a model
// handles no-op collectives or ops without a provider) is model-specific, so
// that loop stays in each report bin.

/// Running aggregate of one op's or call-site's per-call [`LatencyStats`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Accum {
    pub calls: usize,
    pub total_us: f64,
    pub total_p99_us: f64,
    sum_stddev_us: f64,
    sum_p99_us: f64,
}

/// Fold one call's stats into an accumulator.
pub fn accumulate(accum: &mut Accum, stats: &LatencyStats) {
    accum.calls += 1;
    accum.total_us += stats.mean_us;
    accum.total_p99_us += stats.p99_us;
    accum.sum_stddev_us += stats.stddev_us;
    accum.sum_p99_us += stats.p99_us;
}

/// One per-op row of a model report.
#[derive(Clone, Debug, Serialize)]
pub struct RollupRow {
    pub op: String,
    pub calls: usize,
    pub total_us: f64,
    total_p99_us: f64,
    pub per_call_us: f64,
    pub stddev_us: f64,
    pub p99_us: f64,
    pub pct: f64,
}

/// One per-call-site row of a model report.
#[derive(Clone, Debug, Serialize)]
pub struct CallSiteRow {
    pub call_site: String,
    pub op: String,
    pub calls: usize,
    pub per_call_us: f64,
    pub stddev_us: f64,
    pub p99_us: f64,
    pub total_us: f64,
    total_p99_us: f64,
    pub pct: f64,
}

pub fn rollup_row(op: String, accum: Accum, total: f64) -> RollupRow {
    let calls = accum.calls.max(1) as f64;
    RollupRow {
        op,
        calls: accum.calls,
        total_us: accum.total_us,
        total_p99_us: accum.total_p99_us,
        per_call_us: accum.total_us / calls,
        stddev_us: accum.sum_stddev_us / calls,
        p99_us: accum.sum_p99_us / calls,
        pct: pct(accum.total_us, total),
    }
}

pub fn call_site_row(call_site: String, op: String, accum: Accum, total: f64) -> CallSiteRow {
    let calls = accum.calls.max(1) as f64;
    CallSiteRow {
        call_site,
        op,
        calls: accum.calls,
        per_call_us: accum.total_us / calls,
        stddev_us: accum.sum_stddev_us / calls,
        p99_us: accum.sum_p99_us / calls,
        total_us: accum.total_us,
        total_p99_us: accum.total_p99_us,
        pct: pct(accum.total_us, total),
    }
}

/// Percentage share of `value` within `total`, guarding against divide-by-zero.
fn pct(value: f64, total: f64) -> f64 {
    if total == 0.0 {
        0.0
    } else {
        value / total * 100.0
    }
}
