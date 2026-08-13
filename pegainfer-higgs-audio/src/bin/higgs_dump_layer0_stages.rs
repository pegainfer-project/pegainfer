use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use pegainfer_higgs_audio::layer_dump::write_stage_dump;
use pegainfer_higgs_audio::one_step_actual::load_prompt_from_golden;
use pegainfer_qwen3::runtime::Qwen3Executor;

#[derive(Parser)]
#[command(
    about = "Dump Higgs/Qwen3 prefill stage snapshots for one selected layer and one golden prompt"
)]
struct Args {
    /// Qwen3-compatible body view produced by higgs_materialize_qwen3_body.
    #[arg(long)]
    qwen3_body_dir: PathBuf,
    /// Golden safetensors fixture; prompt tensors are copied from this file.
    #[arg(long)]
    golden: PathBuf,
    /// Output layer stage safetensors path.
    #[arg(long)]
    out: PathBuf,
    /// Zero-based decoder layer index to stage-dump.
    #[arg(long, default_value_t = 0)]
    layer_idx: usize,
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
    let stages = executor.prefill_layer_stages_bf16(args.layer_idx, prompt_ids)?;
    let summary = write_stage_dump(&args.out, &prompt, &stages.stages)?;

    println!("higgs layer stage dump: ok");
    println!("  out: {}", summary.output_path.display());
    println!("  layer_idx: {}", args.layer_idx);
    println!("  prompt_tokens: {}", summary.prompt_tokens);
    println!("  stages: {}", summary.stages);
    println!("  values: {}", summary.values);
    Ok(())
}
