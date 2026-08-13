use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use clap::Parser;
use pegainfer_higgs_audio::config::EXPECTED_MODEL_CARD_CONTEXT;
use pegainfer_higgs_audio::config::HiggsConfig;
use pegainfer_higgs_audio::load_plan::HiggsRuntimeLoadPlan;
use pegainfer_higgs_audio::one_step_golden::REQUIRED_TENSORS;
use pegainfer_higgs_audio::one_step_golden::{self};
use pegainfer_higgs_audio::weights::HiggsWeightManifest;
use pegainfer_higgs_audio::weights::fused_modality_shape;
use pegainfer_higgs_audio::weights::validate_checkpoint_headers;
use sha2::Digest;
use sha2::Sha256;

#[derive(Parser)]
#[command(about = "Validate the Higgs Audio model artifacts against the one-step golden contract")]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    golden: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let config = HiggsConfig::from_model_dir(&args.model_dir)?;
    let manifest = HiggsWeightManifest::from_model_dir(&args.model_dir)?;
    let summary = manifest.validate_for_config(&config)?;
    let load_plan = HiggsRuntimeLoadPlan::from_manifest(&config, &manifest)?;
    let load_summary = load_plan.summary();
    let header_summary = validate_checkpoint_headers(&args.model_dir, &config, &manifest)?;
    let golden = one_step_golden::load_and_validate(&args.golden)?;

    check_metadata_hash(
        &golden,
        "config_sha256",
        &args.model_dir.join("config.json"),
    )?;
    check_metadata_hash(
        &golden,
        "tokenizer_json_sha256",
        &args.model_dir.join("tokenizer.json"),
    )?;
    check_metadata_hash(
        &golden,
        "model_index_sha256",
        &args.model_dir.join("model.safetensors.index.json"),
    )?;
    check_optional_model_size(&golden, &args.model_dir.join("model.safetensors"))?;

    println!("higgs artifact check: ok");
    println!("  model_dir: {}", args.model_dir.display());
    println!("  golden: {} bytes sha256={}", golden.bytes, golden.sha256);
    println!(
        "  golden tensors: {} required tensor specs validated",
        REQUIRED_TENSORS.len()
    );
    println!(
        "  config: layers={} hidden={} q_heads={} kv_heads={} head_dim={}",
        config.text.num_hidden_layers,
        config.text.hidden_size,
        config.text.num_attention_heads,
        config.text.num_key_value_heads,
        config.text.head_dim
    );
    println!(
        "  kv bf16: {} bytes/position, {} MiB @ {} positions",
        config.kv_bytes_per_position_bf16(),
        config.kv_bytes_for_positions_bf16(EXPECTED_MODEL_CARD_CONTEXT) / 1024 / 1024,
        EXPECTED_MODEL_CARD_CONTEXT
    );
    println!(
        "  manifest: total={} body={} decoder_only={} fused_modality_shape={:?}",
        summary.total_tensors,
        summary.body_tensors,
        summary.decoder_only_tensors,
        fused_modality_shape()
    );
    println!(
        "  checkpoint headers: files={} tensors={} bf16={}",
        header_summary.files_checked,
        header_summary.tensors_checked,
        header_summary.bf16_tensors_checked
    );
    println!(
        "  runtime load plan: tensors={} shard_files={} bf16_mib={} qwen3_backbone={} higgs_head={}",
        load_summary.tensors,
        load_summary.shard_files,
        load_summary.bf16_bytes / 1024 / 1024,
        load_summary.qwen3_backbone_tensors,
        load_summary.higgs_head_tensors
    );

    Ok(())
}

fn check_metadata_hash(
    golden: &one_step_golden::GoldenContract,
    metadata_key: &str,
    path: &Path,
) -> Result<()> {
    let expected = golden
        .metadata
        .get(metadata_key)
        .with_context(|| format!("golden metadata missing {metadata_key}"))?;
    let actual = sha256_file(path)?;
    if &actual != expected {
        bail!(
            "{metadata_key} mismatch for {}: expected {expected}, got {actual}",
            path.display()
        );
    }
    Ok(())
}

fn check_optional_model_size(
    golden: &one_step_golden::GoldenContract,
    model_safetensors: &Path,
) -> Result<()> {
    let Some(expected) = golden.metadata.get("model_safetensors_size") else {
        return Ok(());
    };
    if !model_safetensors.exists() {
        println!(
            "  note: skipping model.safetensors size check because {} is absent",
            model_safetensors.display()
        );
        return Ok(());
    }
    let actual = std::fs::metadata(model_safetensors)
        .with_context(|| format!("stat {}", model_safetensors.display()))?
        .len()
        .to_string();
    if &actual != expected {
        bail!(
            "model_safetensors_size mismatch for {}: expected {expected}, got {actual}",
            model_safetensors.display()
        );
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let mut digest = Sha256::new();
    digest.update(bytes);
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
