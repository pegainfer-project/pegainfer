//! Stage 9 single-variable Qwen3.5 GDN backend benchmark.
//!
//! This binary uses the same scheduler and production dispatch as serving. The
//! backend seam is crate-internal and exists only to run matched A/B evidence.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use pegainfer_frontend::engine::EngineHandle;
use pegainfer_frontend::engine::GenerateRequest;
use pegainfer_frontend::engine::TokenEvent;
use pegainfer_frontend::engine::TokenSink;
use pegainfer_frontend::engine::TokenStreamReceiver;
use pegainfer_frontend::sampler::SamplingParams;
use pegainfer_qwen35::runtime::GdnPrefillRuntimeEvidence;
use pegainfer_qwen35::runtime_ops::GdrChunkwiseScratch35;
use serde::Serialize;

const H_Q: usize = 16;
const H_K: usize = 16;
const H_V: usize = 32;
const HEAD_DIM: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Triton,
    FlashInfer,
}

impl Backend {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "triton" => Ok(Self::Triton),
            "flashinfer" => Ok(Self::FlashInfer),
            _ => bail!("--backend must be triton or flashinfer, got {value}"),
        }
    }
}

#[derive(Debug)]
struct Args {
    backend: Backend,
    model_path: PathBuf,
    output: Option<PathBuf>,
    prompt_len: usize,
    concurrency: usize,
    warmup: usize,
    iterations: usize,
    max_new_tokens: usize,
    max_prefill_tokens: usize,
    device: usize,
    run_label: String,
}

#[derive(Debug, Serialize)]
struct Stats {
    count: usize,
    mean_ms: f64,
    stddev_ms: f64,
    p50_ms: f64,
    p95_ms: f64,
    p99_ms: f64,
    max_ms: f64,
}

#[derive(Debug, Serialize)]
struct RateStats {
    count: usize,
    mean: f64,
    stddev: f64,
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
}

#[derive(Debug, Serialize)]
struct ScratchReport {
    scope: &'static str,
    geometry: &'static str,
    tokens: usize,
    triton_operator_bytes: usize,
    flashinfer_operator_bytes_excluding_workspace: usize,
    flashinfer_operator_bytes_including_runtime_workspace: Option<usize>,
    flashinfer_runtime_workspace_bytes: Option<u64>,
    artifact_size_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct EvidenceReport {
    selected_backend: String,
    artifact_sha256: String,
    artifact_size_bytes: u64,
    runtime_workspace_bytes: u64,
    successful_launches: u64,
}

#[derive(Debug, Serialize)]
struct Report {
    schema_version: u32,
    surface: &'static str,
    run_label: String,
    backend: Backend,
    model_path: String,
    artifact_manifest_sha256: Option<String>,
    code_commit: Option<String>,
    gpu_label: Option<String>,
    cuda_label: Option<String>,
    prompt_len: usize,
    concurrency: usize,
    warmup: usize,
    iterations: usize,
    max_new_tokens: usize,
    engine_startup_ms: f64,
    ttft: Stats,
    tpot: Stats,
    request_e2e: Stats,
    batch_throughput_tokens_per_second: RateStats,
    completion_tokens: usize,
    scratch: ScratchReport,
    flashinfer_evidence: Option<EvidenceReport>,
}

#[derive(Debug)]
struct RequestTiming {
    ttft_ms: f64,
    e2e_ms: f64,
    tpot_ms: Vec<f64>,
    completion_tokens: usize,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    ensure!(
        args.prompt_len > 0,
        "--prompt-len must be greater than zero"
    );
    ensure!(
        args.concurrency > 0,
        "--concurrency must be greater than zero"
    );
    ensure!(
        args.iterations > 0,
        "--iterations must be greater than zero"
    );
    ensure!(
        args.max_new_tokens >= 2,
        "--max-new-tokens must be at least two so TPOT has samples"
    );
    ensure!(
        args.concurrency <= pegainfer_qwen35::MAX_DECODE_BATCH,
        "--concurrency exceeds Qwen3.5 MAX_DECODE_BATCH={} ",
        pegainfer_qwen35::MAX_DECODE_BATCH
    );

    let startup_started = Instant::now();
    let (handle, evidence_handle) = match args.backend {
        Backend::Triton => (
            pegainfer_qwen35::start_engine_with_triton_gdn_for_accuracy(
                &args.model_path,
                args.device,
                args.concurrency,
                args.max_prefill_tokens,
            )?,
            None,
        ),
        Backend::FlashInfer => {
            let (handle, evidence) =
                pegainfer_qwen35::start_engine_with_flashinfer_gdn_for_accuracy(
                    &args.model_path,
                    args.device,
                    args.concurrency,
                    args.max_prefill_tokens,
                )?;
            (handle, Some(evidence))
        }
    };
    let engine_startup_ms = duration_ms(startup_started.elapsed());

    for warmup_index in 0..args.warmup {
        run_batch(&handle, &args, warmup_index, true)?;
    }

    let mut ttft = Vec::with_capacity(args.iterations * args.concurrency);
    let mut tpot = Vec::new();
    let mut request_e2e = Vec::with_capacity(args.iterations * args.concurrency);
    let mut throughput = Vec::with_capacity(args.iterations);
    let mut completion_tokens = 0usize;

    let measurement_range = nvtx::range!("qwen35.gdn_stage9.measure.{:?}", args.backend);
    for iteration in 0..args.iterations {
        let batch_started = Instant::now();
        let timings = run_batch(&handle, &args, iteration, false)?;
        let batch_seconds = batch_started.elapsed().as_secs_f64();
        let batch_tokens = timings
            .iter()
            .map(|timing| timing.completion_tokens)
            .sum::<usize>();
        ensure!(batch_seconds > 0.0, "benchmark batch duration is zero");
        throughput.push(batch_tokens as f64 / batch_seconds);
        completion_tokens += batch_tokens;
        for timing in timings {
            ttft.push(timing.ttft_ms);
            request_e2e.push(timing.e2e_ms);
            tpot.extend(timing.tpot_ms);
        }
    }
    drop(measurement_range);

    let flashinfer_evidence = if let Some(evidence_handle) = evidence_handle {
        let evidence = evidence_handle.snapshot();
        ensure!(
            evidence.successful_launches > 0,
            "FlashInfer benchmark completed without a successful candidate launch"
        );
        Some(evidence_report(&evidence))
    } else {
        None
    };
    let workspace_bytes = flashinfer_evidence
        .as_ref()
        .map_or(0, |evidence| evidence.runtime_workspace_bytes);
    let artifact_size_bytes = flashinfer_evidence
        .as_ref()
        .map_or(0, |evidence| evidence.artifact_size_bytes);

    let report = Report {
        schema_version: 2,
        surface: "qwen35_engine_handle_no_http_transport",
        run_label: args.run_label,
        backend: args.backend,
        model_path: args.model_path.display().to_string(),
        artifact_manifest_sha256: env::var("PEGAINFER_STAGE9_MANIFEST_SHA256").ok(),
        code_commit: env::var("PEGAINFER_STAGE9_COMMIT").ok(),
        gpu_label: env::var("PEGAINFER_STAGE9_GPU").ok(),
        cuda_label: env::var("PEGAINFER_STAGE9_CUDA").ok(),
        prompt_len: args.prompt_len,
        concurrency: args.concurrency,
        warmup: args.warmup,
        iterations: args.iterations,
        max_new_tokens: args.max_new_tokens,
        engine_startup_ms,
        ttft: stats(&mut ttft)?,
        tpot: stats(&mut tpot)?,
        request_e2e: stats(&mut request_e2e)?,
        batch_throughput_tokens_per_second: rate_stats(&mut throughput)?,
        completion_tokens,
        scratch: scratch_report(args.prompt_len, workspace_bytes, artifact_size_bytes),
        flashinfer_evidence,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = args.output {
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .with_context(|| format!("create output directory {}", parent.display()))?;
        }
        fs::write(&path, &json).with_context(|| format!("write {}", path.display()))?;
    }
    println!("{json}");
    Ok(())
}

fn run_batch(
    handle: &EngineHandle,
    args: &Args,
    iteration: usize,
    warmup: bool,
) -> Result<Vec<RequestTiming>> {
    let mut workers = Vec::with_capacity(args.concurrency);
    let mut submissions = Vec::with_capacity(args.concurrency);
    for request_index in 0..args.concurrency {
        let (token_tx, token_rx) = TokenSink::standalone();
        let (start_tx, start_rx) = mpsc::sync_channel(1);
        workers.push(thread::spawn(move || collect_timing(token_rx, start_rx)));
        submissions.push((request_index, token_tx, start_tx));
    }

    for (request_index, token_tx, start_tx) in submissions {
        let prompt_tokens = deterministic_prompt(args.prompt_len, request_index);
        let started = Instant::now();
        handle.submit(GenerateRequest {
            trace_parent: None,
            request_id: Some(format!(
                "stage9-{}-{iteration}-{request_index}",
                if warmup { "warmup" } else { "measure" }
            )),
            queued_at_unix_s: None,
            data_parallel_rank: None,
            prompt_tokens,
            params: SamplingParams {
                ignore_eos: true,
                ..SamplingParams::default()
            },
            max_tokens: args.max_new_tokens,
            lora_adapter: None,
            kv_transfer_params: None,
            token_tx,
            logprobs: 0,
            echo: false,
        })?;
        start_tx
            .send(started)
            .context("send request start time to collector")?;
    }

    workers
        .into_iter()
        .map(|worker| {
            worker
                .join()
                .map_err(|_| anyhow::anyhow!("Stage 9 collector thread panicked"))?
        })
        .collect()
}

fn collect_timing(
    mut receiver: TokenStreamReceiver,
    start_rx: mpsc::Receiver<Instant>,
) -> Result<RequestTiming> {
    let started = start_rx.recv().context("receive request start time")?;
    let mut first_token_at = None;
    let mut previous_token_at = None;
    let mut tpot_ms = Vec::new();
    let mut completion_tokens = 0usize;
    loop {
        let (_, event) = receiver
            .blocking_recv()
            .context("scheduler channel closed before Finished")?;
        let now = Instant::now();
        match event {
            TokenEvent::Token { .. } => {
                if let Some(previous) = previous_token_at {
                    tpot_ms.push(duration_ms(now.duration_since(previous)));
                } else {
                    first_token_at = Some(now);
                }
                previous_token_at = Some(now);
                completion_tokens += 1;
            }
            TokenEvent::Finished { .. } => {
                let first = first_token_at.context(
                    "request produced no token; use a different deterministic prompt for timing",
                )?;
                ensure!(
                    !tpot_ms.is_empty(),
                    "request produced fewer than two tokens; TPOT is undefined"
                );
                return Ok(RequestTiming {
                    ttft_ms: duration_ms(first.duration_since(started)),
                    e2e_ms: duration_ms(now.duration_since(started)),
                    tpot_ms,
                    completion_tokens,
                });
            }
            TokenEvent::Error { message, .. } | TokenEvent::Rejected { message, .. } => {
                bail!("scheduler request failed: {message}");
            }
            TokenEvent::Scheduled { .. }
            | TokenEvent::PromptTokens { .. }
            | TokenEvent::KvTransfer { .. } => {}
        }
    }
}

fn deterministic_prompt(prompt_len: usize, request_index: usize) -> Vec<u32> {
    (0..prompt_len)
        .map(|index| 100 + ((index + request_index * 17) % 30_000) as u32)
        .collect()
}

fn scratch_report(
    tokens: usize,
    flashinfer_workspace_bytes: u64,
    artifact_size_bytes: u64,
) -> ScratchReport {
    let triton_operator_bytes =
        GdrChunkwiseScratch35::operator_scratch_bytes_from_dims(H_V, HEAD_DIM, HEAD_DIM, tokens);
    let bf16_elements = tokens * (H_Q * HEAD_DIM + H_K * HEAD_DIM + H_V * HEAD_DIM * 2);
    let f32_elements = tokens * H_V * 2;
    let flashinfer_operator_bytes_excluding_workspace = bf16_elements * size_of::<half::bf16>()
        + f32_elements * size_of::<f32>()
        + size_of::<u32>()
        + 2 * size_of::<i64>();
    let runtime_workspace = (flashinfer_workspace_bytes > 0).then_some(flashinfer_workspace_bytes);
    let flashinfer_operator_bytes_including_runtime_workspace =
        runtime_workspace.map(|workspace| {
            flashinfer_operator_bytes_excluding_workspace
                + usize::try_from(workspace).expect("validated runtime workspace fits usize")
        });
    ScratchReport {
        scope: "backend-owned device allocations only; recurrent state and common model temporaries excluded",
        geometry: "Hq=16,Hk=16,Hv=32,D=128",
        tokens,
        triton_operator_bytes,
        flashinfer_operator_bytes_excluding_workspace,
        flashinfer_operator_bytes_including_runtime_workspace,
        flashinfer_runtime_workspace_bytes: runtime_workspace,
        artifact_size_bytes: (artifact_size_bytes > 0).then_some(artifact_size_bytes),
    }
}

fn evidence_report(evidence: &GdnPrefillRuntimeEvidence) -> EvidenceReport {
    EvidenceReport {
        selected_backend: evidence.selected_backend.clone(),
        artifact_sha256: evidence.artifact_sha256.clone(),
        artifact_size_bytes: evidence.artifact_size_bytes,
        runtime_workspace_bytes: evidence.runtime_workspace_bytes,
        successful_launches: evidence.successful_launches,
    }
}

fn stats(values: &mut [f64]) -> Result<Stats> {
    ensure!(!values.is_empty(), "timing sample set is empty");
    values.sort_by(f64::total_cmp);
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    Ok(Stats {
        count: values.len(),
        mean_ms: mean,
        stddev_ms: variance.sqrt(),
        p50_ms: percentile(values, 0.50),
        p95_ms: percentile(values, 0.95),
        p99_ms: percentile(values, 0.99),
        max_ms: *values.last().expect("non-empty timing samples"),
    })
}

fn rate_stats(values: &mut [f64]) -> Result<RateStats> {
    let stats = stats(values)?;
    Ok(RateStats {
        count: stats.count,
        mean: stats.mean_ms,
        stddev: stats.stddev_ms,
        p50: stats.p50_ms,
        p95: stats.p95_ms,
        p99: stats.p99_ms,
        max: stats.max_ms,
    })
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index]
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn parse_args() -> Result<Args> {
    let mut backend = None;
    let mut model_path = None;
    let mut output = None;
    let mut prompt_len = 128usize;
    let mut concurrency = 1usize;
    let mut warmup = 2usize;
    let mut iterations = 10usize;
    let mut max_new_tokens = 8usize;
    let mut max_prefill_tokens = 20_000usize;
    let mut device = 0usize;
    let mut run_label = "stage9".to_string();
    let mut args = env::args().skip(1);
    while let Some(flag) = args.next() {
        if flag == "--help" || flag == "-h" {
            print_help();
            std::process::exit(0);
        }
        let value = args
            .next()
            .with_context(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--backend" => backend = Some(Backend::parse(&value)?),
            "--model-path" => model_path = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--prompt-len" => prompt_len = parse_usize(&flag, &value)?,
            "--concurrency" => concurrency = parse_usize(&flag, &value)?,
            "--warmup" => warmup = parse_usize(&flag, &value)?,
            "--iterations" => iterations = parse_usize(&flag, &value)?,
            "--max-new-tokens" => max_new_tokens = parse_usize(&flag, &value)?,
            "--max-prefill-tokens" => max_prefill_tokens = parse_usize(&flag, &value)?,
            "--device" => device = parse_usize(&flag, &value)?,
            "--run-label" => run_label = value,
            _ => bail!("unknown argument {flag}; run with --help"),
        }
    }
    Ok(Args {
        backend: backend.context("--backend is required")?,
        model_path: model_path.context("--model-path is required")?,
        output,
        prompt_len,
        concurrency,
        warmup,
        iterations,
        max_new_tokens,
        max_prefill_tokens,
        device,
        run_label,
    })
}

fn parse_usize(flag: &str, value: &str) -> Result<usize> {
    value
        .parse::<usize>()
        .with_context(|| format!("{flag} must be an unsigned integer, got {value}"))
}

fn print_help() {
    println!(
        "Usage: gdn_stage9_bench --backend triton|flashinfer --model-path PATH [options]\n\
         \nFlashInfer must be linked at build time with PEGAINFER_QWEN35_GDN_AOT_BUNDLE.\n\
         \nOptions:\n  --prompt-len N           default 128\n  --concurrency N          default 1\n  --warmup N              default 2\n  --iterations N          default 10\n  --max-new-tokens N      default 8\n  --max-prefill-tokens N  default 20000\n  --device N              default 0\n  --run-label TEXT        default stage9\n  --output PATH           also write JSON to PATH"
    );
}
