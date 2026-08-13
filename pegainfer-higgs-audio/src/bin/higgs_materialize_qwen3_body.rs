use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use pegainfer_higgs_audio::config::HiggsConfig;
use pegainfer_higgs_audio::load_plan::HiggsRuntimeLoadPlan;
use pegainfer_higgs_audio::materialize_qwen3::materialize_qwen3_body_view;
use pegainfer_higgs_audio::materialize_qwen3::write_qwen3_config_view;
use pegainfer_higgs_audio::weights::HiggsWeightManifest;
use pegainfer_higgs_audio::weights::validate_checkpoint_headers;

#[derive(Parser)]
#[command(about = "Materialize a Qwen3-compatible view of the Higgs text/body checkpoint")]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    out_dir: PathBuf,
    /// Write only Qwen3 config files plus a tensor-alias manifest; do not copy weight payloads.
    #[arg(long)]
    metadata_only: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let config = HiggsConfig::from_model_dir(&args.model_dir)?;
    let manifest = HiggsWeightManifest::from_model_dir(&args.model_dir)?;
    let plan = HiggsRuntimeLoadPlan::from_manifest(&config, &manifest)?;
    validate_checkpoint_headers(&args.model_dir, &config, &manifest)?;
    if args.metadata_only {
        let summary = write_qwen3_config_view(&args.out_dir, &config, &plan)?;
        println!("higgs qwen3 config view materialized: ok");
        println!("  out_dir: {}", summary.output_dir.display());
        println!("  alias_manifest: {}", summary.alias_manifest.display());
        println!("  aliases: {}", summary.aliases);
        return Ok(());
    }

    let summary = materialize_qwen3_body_view(&args.model_dir, &args.out_dir, &config, &plan)?;
    println!("higgs qwen3 body view materialized: ok");
    println!("  out_dir: {}", summary.output_dir.display());
    println!("  tensors: {}", summary.tensors);
    println!("  payload_mib: {}", summary.payload_bytes / 1024 / 1024);
    Ok(())
}
