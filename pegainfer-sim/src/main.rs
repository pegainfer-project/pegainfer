use std::path::Path;

use anyhow::Result;
use clap::Parser;
use pegainfer_sim::SimulatedEngineConfig;
use pegainfer_sim::start_engine;

const DEFAULT_MODEL_ID: &str = "Qwen/Qwen3-0.6B";

#[derive(Parser, Debug)]
#[command(
    name = "pegainfer-sim",
    about = "CPU-only simulated inference server for OpenAI/vLLM serving benchmarks"
)]
struct Args {
    /// Tokenizer/model metadata id used by the vLLM frontend. No weights are loaded.
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    model_id: String,

    /// Port to listen on.
    #[arg(long, default_value_t = 8000)]
    port: u16,

    /// Max context length reported to the vLLM frontend.
    #[arg(long, default_value_t = 8192)]
    max_model_len: u32,

    /// Fixed TTFT floor before the first fake token.
    #[arg(long, default_value_t = 5.0)]
    base_ttft_ms: f64,

    /// Simulated prefill throughput used as prompt_len / throughput.
    #[arg(long, default_value_t = 100.0)]
    prefill_tokens_per_ms: f64,

    /// Fixed delay between generated fake tokens.
    #[arg(long, default_value_t = 12.0)]
    tpot_ms: f64,

    /// Token id used when a request has an empty prompt-token list.
    #[arg(long, default_value_t = 0)]
    fallback_token_id: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let config = SimulatedEngineConfig::new(
        args.base_ttft_ms,
        args.prefill_tokens_per_ms,
        args.tpot_ms,
        args.fallback_token_id,
    )?;
    let handle = start_engine(config);

    pegainfer_frontend::vllm::serve(
        std::future::ready(Ok(handle.into())),
        Path::new(&args.model_id),
        Vec::new(),
        args.port,
        Some(args.max_model_len),
        pegainfer_frontend::vllm::shutdown_token_from_ctrl_c(),
    )
    .await
}
