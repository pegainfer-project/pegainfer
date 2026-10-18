//! Direct DeepSeek-V2-Lite EP2 decode diagnostic benchmark.
//!
//! This binary records in-process attribution and correctness metadata. It does
//! not exercise HTTP serving or establish serving SLO or production readiness.

use std::env;
use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use pegainfer_deepseek_v2_lite::DecodeGraphReadinessReport;
use pegainfer_deepseek_v2_lite::DeepSeekV2LiteEp2Generator;
use pegainfer_frontend::engine::EngineLoadOptions;
use serde_json::Value;
use serde_json::json;
use sha2::Digest;
use sha2::Sha256;
use vllm_text::tokenizer::HuggingFaceTokenizer;
use vllm_text::tokenizer::Tokenizer;

const PROMPT: &str = "Hello";
const OUTPUT_LEN: usize = 16;
const MAX_BATCH_SIZE: usize = 8;
const EXPECTED_OUTPUT_SHA256_PAIRS: &[(&str, &str, &str)] = &[
    (
        "4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225",
        "0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347",
        "DeepSeek-V2-Lite snapshot 604d5664 on 2x RTX 5090, torch 2.7.0, transformers 4.40.2",
    ),
    (
        "d05a7b0f0ac6435fb51040582a337d8b6d72844dd61194daa1b3090fa0e16ce8",
        "4aaafbe4b3a46bc5b9ab5ea8d09d5fad71225006c2e234e87a928e3265b387c6",
        "DeepSeek-V2-Lite snapshot 604d5664 on 2x A800-SXM4-80GB, torch 2.7.0, transformers 4.40.2",
    ),
];

struct Cli {
    model_path: String,
    commit: Option<String>,
    batch_size: usize,
    nccl_graph_smoke: bool,
    full_decode_graph_probe: bool,
    out: Option<PathBuf>,
}

fn main() -> Result<()> {
    let Some(cli) = parse_cli()? else {
        return Ok(());
    };
    let model_path = resolve_model_path(&cli.model_path);
    ensure!(
        model_path.join("config.json").exists(),
        "missing config.json under {}",
        model_path.display()
    );

    let tokenizer_path = model_path.join("tokenizer.json");
    let tokenizer = HuggingFaceTokenizer::new(&tokenizer_path).map_err(|err| {
        anyhow::anyhow!(
            "failed to load tokenizer {}: {err:?}",
            tokenizer_path.display()
        )
    })?;
    let prompt_tokens = tokenizer
        .encode(PROMPT, false)
        .map_err(|err| anyhow::anyhow!("encode prompt failed: {err:?}"))?;
    ensure!(!prompt_tokens.is_empty(), "tokenizer returned empty prompt");

    trace_stage("load generator");
    let mut generator = DeepSeekV2LiteEp2Generator::load(
        &model_path,
        EngineLoadOptions {
            enable_cuda_graph: false,
            device_ordinals: vec![0, 1],
            seed: 42,
            ..EngineLoadOptions::default()
        },
    )?;
    trace_stage("generator loaded");
    let (mut report, probe_status, probe_ready) = if cli.batch_size == 1 {
        trace_stage("run batch=1 attribution oracle");
        let (result, attribution) =
            generator.generate_greedy_with_attribution(&prompt_tokens, OUTPUT_LEN, false)?;
        trace_stage("batch=1 attribution oracle complete");
        trace_stage("run graph readiness report");
        let graph_readiness = generator.decode_graph_readiness_report(
            &result.stats,
            1,
            cli.nccl_graph_smoke,
            cli.full_decode_graph_probe,
            Some(&prompt_tokens),
            OUTPUT_LEN,
        )?;
        trace_stage("graph readiness report complete");
        let probe_status = graph_readiness.full_decode_graph_probe_status();
        let probe_ready = graph_readiness.full_decode_capture_ready();
        let report = single_report(
            &tokenizer,
            &prompt_tokens,
            &result,
            &attribution,
            &graph_readiness,
        )?;
        (report, probe_status, probe_ready)
    } else {
        trace_stage("run batch attribution oracle");
        let (result, attribution) = generator.generate_greedy_batch_same_prompt_with_attribution(
            &prompt_tokens,
            cli.batch_size,
            OUTPUT_LEN,
            true,
        )?;
        trace_stage("batch attribution oracle complete");
        trace_stage("run graph readiness report");
        let graph_readiness = generator.decode_graph_readiness_report(
            &result.stats,
            cli.batch_size,
            cli.nccl_graph_smoke,
            cli.full_decode_graph_probe,
            None,
            OUTPUT_LEN,
        )?;
        trace_stage("graph readiness report complete");
        let probe_status = graph_readiness.full_decode_graph_probe_status();
        let probe_ready = graph_readiness.full_decode_capture_ready();
        let report = batch_report(
            &tokenizer,
            &prompt_tokens,
            &result,
            &attribution,
            &graph_readiness,
        )?;
        (report, probe_status, probe_ready)
    };
    attach_report_metadata(&mut report, &cli, &model_path)?;

    let text = serde_json::to_string_pretty(&report)?;
    let output_path = cli.out.clone().map(resolve_workspace_path);
    if let Some(path) = &output_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        fs::write(path, format!("{text}\n"))
            .with_context(|| format!("write {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    println!("{text}");
    ensure_full_decode_graph_probe_succeeded(
        cli.full_decode_graph_probe,
        probe_ready,
        probe_status,
        output_path.as_deref(),
    )?;
    Ok(())
}

fn ensure_full_decode_graph_probe_succeeded(
    requested: bool,
    ready: bool,
    status: &str,
    output_path: Option<&Path>,
) -> Result<()> {
    ensure!(
        !requested || ready,
        "DeepSeek-V2-Lite --full-decode-graph-probe failed with status={status}; diagnostic JSON was {}",
        graph_probe_diagnostic_target(output_path)
    );
    Ok(())
}

fn graph_probe_diagnostic_target(output_path: Option<&Path>) -> String {
    output_path.map_or_else(
        || "printed to stdout".to_string(),
        |path| format!("written to {} and printed to stdout", path.display()),
    )
}

fn trace_stage(stage: &str) {
    if env::var_os("PEGAINFER_DSV2_LITE_GRAPH_PROBE_TRACE").is_some() {
        eprintln!("[dsv2-lite-attribution] {stage}");
    }
}

fn single_report(
    tokenizer: &HuggingFaceTokenizer,
    prompt_tokens: &[u32],
    result: &pegainfer_deepseek_v2_lite::GenerationResult,
    attribution: &pegainfer_deepseek_v2_lite::DecodeAttributionProfile,
    graph_readiness: &DecodeGraphReadinessReport,
) -> Result<Value> {
    let generated_text = tokenizer
        .decode(&result.tokens, false)
        .map_err(|err| anyhow::anyhow!("decode output failed: {err:?}"))?;
    let text_sha256 = sha256_text(&generated_text);
    let (expected_token_sha256, expected_text_sha256, matched_oracle) =
        expect_output_oracle(&result.stats.output_token_sha256, &text_sha256)?;

    let by_section = attribution.by_section();
    let by_gpu_section = attribution.by_gpu_section();
    let by_gpu_call_site = attribution.by_gpu_call_site();
    let by_op = by_op_rows(&by_section);
    let mut per_output_token_us = Vec::with_capacity(OUTPUT_LEN);
    if let Some(prefill_next_token_us) = attribution.prefill_next_token_us() {
        per_output_token_us.push(prefill_next_token_us);
    }
    per_output_token_us.extend_from_slice(attribution.per_token_decode_us());

    Ok(json!({
        "schema": 1,
        "report_type": "deepseek-v2-lite-ep2-decode-attribution",
        "backend": &result.stats.ep_backend,
        "workload": {
            "batch_size": 1,
            "prompt": PROMPT,
            "prompt_token_ids": &prompt_tokens,
            "output_len": OUTPUT_LEN,
            "ignore_eos": false,
            "ep_size": result.stats.ep_size,
            "device_ordinals": &result.stats.device_ordinals,
            "cuda_graph": false,
        },
        "config": {
            "batch_size": 1,
            "prompt": PROMPT,
            "prompt_token_ids": &prompt_tokens,
            "output_len": OUTPUT_LEN,
            "ep_size": result.stats.ep_size,
            "device_ordinals": &result.stats.device_ordinals,
        },
        "accuracy": {
            "generated_token_ids": &result.tokens,
            "generated_text": &generated_text,
            "token_sha256": &result.stats.output_token_sha256,
            "text_sha256": &text_sha256,
            "expected_token_sha256": expected_token_sha256,
            "expected_text_sha256": expected_text_sha256,
            "matched_output_oracle": matched_oracle,
            "known_output_sha256_pairs": expected_output_sha256_pairs(),
            "token_hash_exact": true,
            "text_hash_exact": true,
        },
        "timing": {
            "source": "CPU-side wall-clock sections around the existing DeepSeek-V2-Lite EP2 greedy path",
            "total_generation_us": attribution.total_generation_us(),
            "prefill_next_token_us": attribution.prefill_next_token_us(),
            "per_output_token_us": per_output_token_us,
            "decode_token_count": attribution.per_token_decode_us().len(),
            "per_token_decode_us": attribution.per_token_decode_us(),
            "per_token_decode_stats": latency_stats(attribution.per_token_decode_us()),
        },
        "metrics": {
            "total_generation_us": attribution.total_generation_us(),
            "prefill_next_token_us": attribution.prefill_next_token_us(),
            "decode_token_count": attribution.per_token_decode_us().len(),
            "per_token_decode_stats": latency_stats(attribution.per_token_decode_us()),
            "gpu_sample_count": attribution.gpu_sample_count(),
            "gpu_timing_failure_count": attribution.gpu_timing_failure_count(),
        },
        "attribution_source": "DeepSeekV2LiteEp2Generator::generate_greedy_with_attribution; CPU sections use host wall-clock timers, and selected GPU/NCCL sections also carry CUDA event timing plus optional NVTX ranges.",
        "gpu_timing": {
            "source": "CUDA event timing around selected DeepSeek-V2-Lite EP2 GPU/NCCL stream sections in the explicit attribution path",
            "sample_count": attribution.gpu_sample_count(),
            "failure_count": attribution.gpu_timing_failure_count(),
            "nvtx_enabled": attribution.nvtx_enabled(),
            "nvtx_range_count": attribution.nvtx_range_count(),
            "scope": "selected GPU/NCCL sections only; host decode work, eager route planning, and the mixed attention_host_path remain CPU-side attribution rows; GPU timing failures do not replace the token/text hash oracle; NVTX range wall time is only a profiler correlation marker and may include host/event overhead",
        },
        "schedule_source": "fixed DeepSeek-V2-Lite EP2 greedy gate: prompt=Hello, output_len=16, cuda_graph=false, device_ordinals=[0,1]",
        "by_section": by_section,
        "by_op": by_op,
        "by_call_site": attribution.by_call_site(),
        "by_gpu_section": by_gpu_section,
        "by_gpu_call_site": by_gpu_call_site,
        "coverage": coverage_rows(&result.stats.ep_backend, 1, attribution, graph_readiness),
        "ep": ep_report(&result.stats),
        "cuda_graph_readiness": graph_readiness,
        "claim_boundary": "Attribution only for the covered EP2 Hello/16 decode gate. CPU-side section timing, selected CUDA event timing, NVTX ranges, and route/collective counts are not a throughput, sparse-dispatch, multi-node, or production EP readiness claim.",
    }))
}

fn batch_report(
    tokenizer: &HuggingFaceTokenizer,
    prompt_tokens: &[u32],
    result: &pegainfer_deepseek_v2_lite::BatchedGenerationResult,
    attribution: &pegainfer_deepseek_v2_lite::DecodeAttributionProfile,
    graph_readiness: &DecodeGraphReadinessReport,
) -> Result<Value> {
    ensure!(
        result.per_token_decode_us == attribution.per_token_decode_us(),
        "batch result decode timings differ from attribution decode timings"
    );
    let mut generated_text_by_row = Vec::with_capacity(result.tokens.len());
    let mut token_sha256_by_row = Vec::with_capacity(result.tokens.len());
    let mut text_sha256_by_row = Vec::with_capacity(result.tokens.len());
    for row in &result.tokens {
        let generated_text = tokenizer
            .decode(row, false)
            .map_err(|err| anyhow::anyhow!("decode batch output failed: {err:?}"))?;
        token_sha256_by_row.push(sha256_tokens(row));
        text_sha256_by_row.push(sha256_text(&generated_text));
        generated_text_by_row.push(generated_text);
    }
    ensure!(
        token_sha256_by_row
            .iter()
            .all(|hash| hash == &token_sha256_by_row[0])
            && text_sha256_by_row
                .iter()
                .all(|hash| hash == &text_sha256_by_row[0]),
        "DeepSeek-V2-Lite batch attribution rows are not hash-identical"
    );
    let (expected_token_sha256, expected_text_sha256, matched_oracle) =
        expect_output_oracle(&token_sha256_by_row[0], &text_sha256_by_row[0])?;

    let by_section = attribution.by_section();
    let by_gpu_section = attribution.by_gpu_section();
    let by_gpu_call_site = attribution.by_gpu_call_site();
    let by_op = by_op_rows(&by_section);

    Ok(json!({
        "schema": 1,
        "report_type": "deepseek-v2-lite-ep2-decode-attribution",
        "backend": &result.stats.ep_backend,
        "workload": {
            "batch_size": result.tokens.len(),
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "output_len": OUTPUT_LEN,
            "ignore_eos": true,
            "ep_size": result.stats.ep_size,
            "device_ordinals": &result.stats.device_ordinals,
            "cuda_graph": false,
        },
        "config": {
            "batch_size": result.tokens.len(),
            "prompt": PROMPT,
            "prompt_token_ids": prompt_tokens,
            "output_len": OUTPUT_LEN,
            "ignore_eos": true,
            "ep_size": result.stats.ep_size,
            "device_ordinals": &result.stats.device_ordinals,
        },
        "accuracy": {
            "generated_token_ids": &result.tokens[0],
            "generated_text": &generated_text_by_row[0],
            "token_sha256": &token_sha256_by_row[0],
            "text_sha256": &text_sha256_by_row[0],
            "generated_token_ids_by_row": &result.tokens,
            "generated_text_by_row": &generated_text_by_row,
            "token_sha256_by_row": &token_sha256_by_row,
            "text_sha256_by_row": &text_sha256_by_row,
            "same_prompt_rows_exact": true,
            "expected_token_sha256": expected_token_sha256,
            "expected_text_sha256": expected_text_sha256,
            "matched_output_oracle": matched_oracle,
            "known_output_sha256_pairs": expected_output_sha256_pairs(),
            "token_hash_exact": true,
            "text_hash_exact": true,
        },
        "timing": {
            "source": "CPU-side wall-clock sections around the existing DeepSeek-V2-Lite EP2 same-prompt batched greedy path",
            "total_generation_us": result.total_generation_us,
            "prefill_next_token_us_by_row": &result.prefill_next_token_us,
            "per_shared_decode_step_us": &result.per_token_decode_us,
            "decode_step_count": result.per_token_decode_us.len(),
            "per_token_decode_us": attribution.per_token_decode_us(),
            "per_token_decode_stats": latency_stats(attribution.per_token_decode_us()),
        },
        "metrics": {
            "total_generation_us": result.total_generation_us,
            "prefill_next_token_us_by_row": &result.prefill_next_token_us,
            "decode_step_count": result.per_token_decode_us.len(),
            "per_token_decode_stats": latency_stats(attribution.per_token_decode_us()),
            "gpu_sample_count": attribution.gpu_sample_count(),
            "gpu_timing_failure_count": attribution.gpu_timing_failure_count(),
        },
        "attribution_source": "DeepSeekV2LiteEp2Generator::generate_greedy_batch_same_prompt_with_attribution; CPU sections use host wall-clock timers, and selected GPU/NCCL sections also carry CUDA event timing plus optional NVTX ranges.",
        "gpu_timing": {
            "source": "CUDA event timing around selected DeepSeek-V2-Lite EP2 GPU/NCCL stream sections in the explicit batched attribution path",
            "sample_count": attribution.gpu_sample_count(),
            "failure_count": attribution.gpu_timing_failure_count(),
            "nvtx_enabled": attribution.nvtx_enabled(),
            "nvtx_range_count": attribution.nvtx_range_count(),
            "scope": "selected GPU/NCCL sections only; host decode work, eager route planning, and the mixed attention_host_path remain CPU-side attribution rows; GPU timing failures do not replace the token/text hash oracle; NVTX range wall time is only a profiler correlation marker and may include host/event overhead",
        },
        "schedule_source": format!(
            "fixed DeepSeek-V2-Lite EP2 greedy gate: batch_size={}, prompt=Hello, output_len=16, cuda_graph=false, device_ordinals=[0,1]",
            result.tokens.len()
        ),
        "by_section": by_section,
        "by_op": by_op,
        "by_call_site": attribution.by_call_site(),
        "by_gpu_section": by_gpu_section,
        "by_gpu_call_site": by_gpu_call_site,
        "coverage": coverage_rows(&result.stats.ep_backend, result.tokens.len(), attribution, graph_readiness),
        "ep": ep_report(&result.stats),
        "cuda_graph_readiness": graph_readiness,
        "claim_boundary": "Attribution only for the covered EP2 Hello/16 same-prompt batched decode gate. CPU-side section timing, selected CUDA event timing, NVTX ranges, and route/collective counts are not a throughput, sparse-dispatch, multi-node, or production EP readiness claim.",
    }))
}

fn parse_cli() -> Result<Option<Cli>> {
    let mut model_path = "models/DeepSeek-V2-Lite".to_string();
    let mut commit = None;
    let mut batch_size = 1;
    let mut nccl_graph_smoke = false;
    let mut full_decode_graph_probe = false;
    let mut out = None;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model-path" => {
                model_path = args
                    .next()
                    .context("--model-path requires a path argument")?;
            }
            "--commit" => {
                commit = Some(args.next().context("--commit requires a value")?);
            }
            "--batch-size" => {
                batch_size = args
                    .next()
                    .context("--batch-size requires an integer argument")?
                    .parse()
                    .context("--batch-size must be an integer")?;
                ensure!(
                    (1..=MAX_BATCH_SIZE).contains(&batch_size),
                    "--batch-size must be in 1..={MAX_BATCH_SIZE}, got {batch_size}"
                );
            }
            "--out" => {
                out = Some(PathBuf::from(
                    args.next().context("--out requires a path argument")?,
                ));
            }
            "--nccl-graph-smoke" => {
                nccl_graph_smoke = true;
            }
            "--full-decode-graph-probe" => {
                full_decode_graph_probe = true;
            }
            "-h" | "--help" => {
                println!(
                    "DeepSeek-V2-Lite EP2 decode attribution gate\n\nUSAGE:\n  dsv2_lite_ep2_decode_attribution [--model-path PATH] [--commit COMMIT] [--batch-size N] [--nccl-graph-smoke] [--full-decode-graph-probe] [--out PATH]\n\nThe gate is intentionally fixed to prompt=Hello, output_len=16, with batch-size in 1..=8. Select NCCL with PEGAINFER_DSV2_LITE_EP_BACKEND=nccl. Use --nccl-graph-smoke to run a preallocated f32 NCCL all-reduce CUDA Graph capture/replay smoke after attribution. Use --full-decode-graph-probe to request the fail-closed full decode graph readiness probe for the retained batch-size=1 NCCL shape."
                );
                return Ok(None);
            }
            other => bail!(
                "unsupported argument `{other}`; supported flags: --model-path PATH, --commit COMMIT, --batch-size N, --nccl-graph-smoke, --full-decode-graph-probe, --out PATH"
            ),
        }
    }
    ensure!(
        !full_decode_graph_probe || batch_size == 1,
        "--full-decode-graph-probe is currently scoped to --batch-size 1, got {batch_size}"
    );
    Ok(Some(Cli {
        model_path,
        commit,
        batch_size,
        nccl_graph_smoke,
        full_decode_graph_probe,
        out,
    }))
}

fn ep_report(stats: &pegainfer_deepseek_v2_lite::GenerationStats) -> Value {
    let (local_route_count, remote_route_count) = match stats.ep_backend.as_str() {
        "host-staged" => (
            stats.host_dispatch_local_routes,
            stats.host_dispatch_remote_routes,
        ),
        "nccl" => (
            stats.nccl_dispatch_local_routes,
            stats.nccl_dispatch_remote_routes,
        ),
        _ => (0, 0),
    };
    json!({
        "dispatch_calls": stats.host_dispatch_calls,
        "dispatch_elements": stats.host_dispatch_elements,
        "combine_calls": stats.host_combine_calls,
        "combine_elements": stats.host_combine_elements,
        "nccl_exchange_calls": stats.nccl_dense_exchange_calls,
        "nccl_exchange_elements": stats.nccl_dense_exchange_elements,
        "nccl_combine_calls": stats.nccl_combine_calls,
        "nccl_combine_elements": stats.nccl_combine_elements,
        "local_route_count": local_route_count,
        "remote_route_count": remote_route_count,
        "host_dispatch_local_routes": stats.host_dispatch_local_routes,
        "host_dispatch_remote_routes": stats.host_dispatch_remote_routes,
        "nccl_dispatch_local_routes": stats.nccl_dispatch_local_routes,
        "nccl_dispatch_remote_routes": stats.nccl_dispatch_remote_routes,
        "nccl_combine_routes": stats.nccl_combine_routes,
    })
}

fn by_op_rows(rows: &[pegainfer_deepseek_v2_lite::SectionRollup]) -> Vec<Value> {
    rows.iter()
        .map(|row| {
            json!({
                "op": row.section,
                "calls": row.calls,
                "total_us": row.total_us,
                "mean_us": row.mean_us,
                "min_us": row.min_us,
                "p50_us": row.p50_us,
                "p95_us": row.p95_us,
                "p99_us": row.p99_us,
                "max_us": row.max_us,
                "pct": row.pct,
            })
        })
        .collect()
}

fn coverage_rows(
    backend: &str,
    batch_size: usize,
    attribution: &pegainfer_deepseek_v2_lite::DecodeAttributionProfile,
    graph_readiness: &DecodeGraphReadinessReport,
) -> Vec<Value> {
    vec![
        json!({
            "item": "accuracy.token_text_hash",
            "status": "passed",
            "source": "same token/text hash oracle as the DeepSeek-V2-Lite EP2 HF accuracy gate",
        }),
        json!({
            "item": "cpu_side_sections",
            "status": "measured",
            "source": "Instant-based host-side section timers in the explicit attribution path",
        }),
        json!({
            "item": "ep_route_and_transfer_counts",
            "status": "measured",
            "source": format!("{backend} EP counters recorded by the DeepSeek-V2-Lite EP2 runtime"),
        }),
        json!({
            "item": "gpu_event_timing",
            "status": gpu_timing_status(attribution),
            "source": "CUDA event timing around selected GPU/NCCL stream sections; pure host sections and mixed attention_host_path are not represented as GPU event rows; failures are counted separately from the accuracy oracle",
        }),
        json!({
            "item": "nvtx_ranges",
            "status": if attribution.nvtx_enabled() { "emitted" } else { "available_when_enabled" },
            "source": "set PEGAINFER_DSV2_LITE_NVTX=1 to emit NVTX ranges for the same selected GPU/NCCL attribution sections",
        }),
        json!({
            "item": "throughput_or_production_ep_readiness",
            "status": "not_claimed",
            "source": format!("batch={batch_size} prompt=Hello output_len=16 diagnostic gate only"),
        }),
        json!({
            "item": "full_decode_cuda_graph_capture",
            "status": graph_readiness.full_decode_graph_probe_status(),
            "source": format!("{} blocker(s) reported by the DeepSeek-V2-Lite EP2 graph-readiness diagnostic; readiness requires full_decode_graph_probe to capture, instantiate, replay, and verify", graph_readiness.blocker_count()),
        }),
        json!({
            "item": "nccl_cuda_graph_smoke",
            "status": graph_readiness.nccl_graph_smoke_status(),
            "source": "optional preallocated f32 NCCL all-reduce capture/replay smoke; this is not full decode graph coverage",
        }),
    ]
}

fn gpu_timing_status(attribution: &pegainfer_deepseek_v2_lite::DecodeAttributionProfile) -> &str {
    match (
        attribution.gpu_sample_count(),
        attribution.gpu_timing_failure_count(),
    ) {
        (0, 0) => "not_covered",
        (0, _) => "timing_failed",
        (_, 0) => "measured",
        (_, _) => "measured_with_failures",
    }
}

fn expect_output_oracle(
    token_sha256: &str,
    text_sha256: &str,
) -> Result<(&'static str, &'static str, &'static str)> {
    matched_expected_output_oracle(token_sha256, text_sha256).with_context(|| {
        format!(
            "DeepSeek-V2-Lite attribution hash drift: got token_sha256={token_sha256} text_sha256={text_sha256}, expected one HF-confirmed pair from {:?}",
            EXPECTED_OUTPUT_SHA256_PAIRS
        )
    })
}

fn matched_expected_output_oracle(
    token_sha256: &str,
    text_sha256: &str,
) -> Option<(&'static str, &'static str, &'static str)> {
    EXPECTED_OUTPUT_SHA256_PAIRS
        .iter()
        .find(|(expected_token, expected_text, _)| {
            token_sha256 == *expected_token && text_sha256 == *expected_text
        })
        .copied()
}

fn expected_output_sha256_pairs() -> Vec<Value> {
    EXPECTED_OUTPUT_SHA256_PAIRS
        .iter()
        .map(|(token_sha256, text_sha256, source)| {
            json!({
                "token_sha256": token_sha256,
                "text_sha256": text_sha256,
                "source": source,
            })
        })
        .collect()
}

fn latency_stats(samples: &[u64]) -> Value {
    if samples.is_empty() {
        return json!({
            "count": 0,
            "total_us": 0,
            "mean_us": 0.0,
            "min_us": 0,
            "p50_us": 0,
            "p95_us": 0,
            "p99_us": 0,
            "max_us": 0,
        });
    }
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let total_us: u64 = sorted.iter().sum();
    json!({
        "count": sorted.len(),
        "total_us": total_us,
        "mean_us": total_us as f64 / sorted.len() as f64,
        "min_us": sorted[0],
        "p50_us": percentile(&sorted, 0.50),
        "p95_us": percentile(&sorted, 0.95),
        "p99_us": percentile(&sorted, 0.99),
        "max_us": sorted[sorted.len() - 1],
    })
}

fn percentile(sorted: &[u64], quantile: f64) -> u64 {
    let idx = ((sorted.len() as f64 - 1.0) * quantile).ceil() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn sha256_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
}

fn sha256_tokens(tokens: &[u32]) -> String {
    let mut hasher = Sha256::new();
    for token in tokens {
        hasher.update(token.to_le_bytes());
    }
    hex::encode(hasher.finalize())
}

fn current_commit() -> Option<String> {
    command_output(
        Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(workspace_root()),
    )
}

fn resolve_commit(requested: Option<&str>) -> Result<Option<String>> {
    if let Some(requested) = requested {
        ensure!(
            matches!(requested.len(), 12 | 40)
                && requested.bytes().all(|byte| byte.is_ascii_hexdigit())
                && requested.bytes().all(|byte| !byte.is_ascii_uppercase()),
            "--commit must be a 12- or 40-character lowercase Git object id"
        );
    }
    let current = current_commit();
    ensure!(
        requested.is_none() || current.is_some(),
        "--commit requires a Git worktree so the requested revision can be verified"
    );
    if let (Some(requested), Some(current)) = (requested, current.as_deref()) {
        let matches_head = if requested.len() == 12 {
            current.get(..12) == Some(requested)
        } else {
            requested == current
        };
        ensure!(
            matches_head,
            "--commit {requested} does not match current HEAD {current}"
        );
    }
    Ok(requested.map(str::to_owned).or(current))
}

fn hardware_toolchain_metadata() -> Value {
    json!({
        "gpu": command_output(Command::new("nvidia-smi").args([
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ])).map(|value| value.lines().map(str::to_owned).collect::<Vec<_>>()).unwrap_or_default(),
        "cuda_visible_devices": env::var("CUDA_VISIBLE_DEVICES").ok(),
        "nvcc_version": command_output(Command::new("nvcc").arg("--version")),
        "rustc_version": command_output(Command::new("rustc").arg("--version")),
        "cargo_version": command_output(Command::new("cargo").arg("--version")),
    })
}

fn command_output(command: &mut Command) -> Option<String> {
    let output = command.output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn attach_report_metadata(report: &mut Value, cli: &Cli, model_path: &Path) -> Result<()> {
    let commit = resolve_commit(cli.commit.as_deref())?;
    let object = report
        .as_object_mut()
        .context("decode attribution report must be a JSON object")?;
    object.insert("schema_version".to_string(), json!(1));
    object.insert(
        "kind".to_string(),
        json!("deepseek_v2_lite_direct_decode_attribution"),
    );
    object.insert("model".to_string(), json!("DeepSeek-V2-Lite"));
    object.insert("phase".to_string(), json!("decode"));
    object.insert(
        "report_intent".to_string(),
        json!("direct_decode_diagnostic"),
    );
    object.insert("commit".to_string(), json!(commit));
    object.insert(
        "hardware_toolchain".to_string(),
        hardware_toolchain_metadata(),
    );
    object.insert(
        "metadata".to_string(),
        json!({
            "commit": commit,
            "model_path": model_path.display().to_string(),
            "command": env::args().collect::<Vec<_>>(),
        }),
    );
    Ok(())
}

fn resolve_model_path(raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.join("config.json").exists() {
        return path;
    }
    let workspace_path = resolve_workspace_path(path.clone());
    if workspace_path.join("config.json").exists() {
        return workspace_path;
    }
    path
}

fn resolve_workspace_path(path: PathBuf) -> PathBuf {
    if path.is_absolute() {
        return path;
    }
    workspace_root().join(path)
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("model crate must live under the workspace root")
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_commit_must_match_current_head() {
        let error = resolve_commit(Some("000000000000"))
            .expect_err("a retained direct report must reject a forged commit");
        assert!(error.to_string().contains("does not match current HEAD"));
    }

    #[test]
    fn full_commit_cannot_forge_the_current_short_prefix() {
        let current = current_commit().expect("tests run from a Git worktree");
        let replacement = if current.ends_with('0') { '1' } else { '0' };
        let forged = format!("{}{replacement}", &current[..current.len() - 1]);
        let error = resolve_commit(Some(&forged))
            .expect_err("a 40-character commit must match the complete HEAD object id");
        assert!(error.to_string().contains("does not match current HEAD"));
    }

    #[test]
    fn full_decode_graph_probe_gate_allows_not_requested() {
        ensure_full_decode_graph_probe_succeeded(
            false,
            false,
            "full_decode_probe_not_requested",
            None,
        )
        .unwrap();
    }

    #[test]
    fn full_decode_graph_probe_gate_allows_ready_probe() {
        ensure_full_decode_graph_probe_succeeded(true, true, "captured_replayed_verified", None)
            .unwrap();
    }

    #[test]
    fn full_decode_graph_probe_gate_fails_requested_unready_probe() {
        let err = ensure_full_decode_graph_probe_succeeded(
            true,
            false,
            "blocked_preflight",
            Some(Path::new("target/probe.json")),
        )
        .unwrap_err();
        let message = err.to_string();

        assert!(message.contains("--full-decode-graph-probe failed"));
        assert!(message.contains("status=blocked_preflight"));
        assert!(message.contains("target/probe.json"));
    }
}
