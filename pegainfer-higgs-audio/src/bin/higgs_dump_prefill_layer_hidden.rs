use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use pegainfer_higgs_audio::layer_dump::write_layer_hidden_dump;
use pegainfer_higgs_audio::one_step_actual::load_prompt_from_golden;
use pegainfer_qwen3::runtime::Qwen3Executor;

#[derive(Parser)]
#[command(about = "Dump Higgs/Qwen3 per-layer prefill hidden snapshots for one golden prompt")]
struct Args {
    /// Qwen3-compatible body view produced by higgs_materialize_qwen3_body.
    #[arg(long)]
    qwen3_body_dir: PathBuf,
    /// Golden safetensors fixture; prompt tensors are copied from this file.
    #[arg(long)]
    golden: PathBuf,
    /// Output layer hidden safetensors path.
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 0)]
    device_ordinal: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let prompt = load_prompt_from_golden(&args.golden)?;
    let prompt_ids = prompt.prompt_ids()?;
    let qwen3_body_dir = args
        .qwen3_body_dir
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("qwen3 body dir must be valid UTF-8"))?;
    let mut executor = Qwen3Executor::from_runtime(qwen3_body_dir, false, &[args.device_ordinal])?;
    let hidden = executor.prefill_layer_hidden_bf16(prompt_ids)?;
    let summary = write_layer_hidden_dump(
        &args.out,
        &prompt,
        &hidden.embedding_hidden_bf16,
        &hidden.layer_hidden_bf16,
        &hidden.final_normed_bf16,
    )?;

    println!("higgs prefill layer hidden dump: ok");
    println!("  out: {}", summary.output_path.display());
    println!("  prompt_tokens: {}", summary.prompt_tokens);
    println!("  layers: {}", summary.layers);
    println!(
        "  hidden_values_per_layer: {}",
        summary.hidden_values_per_layer
    );
    Ok(())
}
