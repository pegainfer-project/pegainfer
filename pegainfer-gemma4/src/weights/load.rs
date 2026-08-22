//! Checkpoint loading: classify the whole manifest, then upload.

use std::time::Instant;

use anyhow::Result;
use log::info;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::weight_loader::SlotId;
use pegainfer_core::weight_loader::StagedWeightLoader;
use pegainfer_core::weight_loader::VecSlotId;
use pegainfer_core::weight_loader::WeightPrefetch;
use pegainfer_core::weight_loader::deserialize_shards;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_core::weight_loader::mmap_shards;
use safetensors::Dtype;
use safetensors::SafeTensors;

use super::Gemma4Attention;
use super::Gemma4Layer;
use super::Gemma4Mlp;
use super::Gemma4Weights;
use crate::config::Gemma4Config;
use crate::manifest::schema::Manifest;
use crate::manifest::schema::Matrix2d;
use crate::manifest::schema::Vector1d;
use crate::manifest::validate::ObservedTensor;

/// The wall figures are probes, not a partition: context creation, slot
/// redemption, prefetch join and unmap fall between them, and the allocations
/// submitted under `record_api_wall_ms` execute under
/// `execute_and_drain_wall_ms`. Only `elapsed_ms` is a total.
pub(crate) struct LoadStats {
    /// Every required tensor at its dtype.
    manifest_bytes: usize,
    /// Free-before minus free-after. Signed: this measures the device, not the
    /// process, so anything else running on it moves the number too.
    device_bytes: i64,
    device_free_bytes: usize,
    /// Config, headers, classification. No device. The advisory prefetch
    /// workers start inside this window and keep running past it.
    validate_wall_ms: f64,
    /// Submitting one allocation and one staging plan per tensor. Wider than
    /// the shared loader's `alloc_api_wall`, which times the allocs alone.
    record_api_wall_ms: f64,
    /// The checkpoint is consumed here, so this carries the source read as
    /// well as the transfer.
    execute_and_drain_wall_ms: f64,
    /// The whole call, unmap included.
    elapsed_ms: f64,
    skipped_modality_tensors: usize,
}

struct LayerSlots {
    input_layernorm: VecSlotId,
    post_attention_layernorm: VecSlotId,
    pre_feedforward_layernorm: VecSlotId,
    post_feedforward_layernorm: VecSlotId,
    layer_scalar: f32,
    q_proj: SlotId,
    k_proj: SlotId,
    v_proj: Option<SlotId>,
    o_proj: SlotId,
    q_norm: VecSlotId,
    k_norm: VecSlotId,
    gate: SlotId,
    up: SlotId,
    down: SlotId,
}

struct RecordedPlan {
    embed_tokens: SlotId,
    norm: VecSlotId,
    layers: Vec<LayerSlots>,
}

fn record_matrix(loader: &mut StagedWeightLoader, tensor: &Matrix2d) -> Result<SlotId> {
    loader.matrix(&tensor.name, tensor.rows, tensor.cols)
}

fn record_vector(loader: &mut StagedWeightLoader, tensor: &Vector1d) -> Result<VecSlotId> {
    loader.vector(&tensor.name, tensor.len)
}

fn read_headers(shards: &[SafeTensors]) -> Result<Vec<(String, Dtype, Vec<usize>)>> {
    let mut headers = Vec::new();
    for shard in shards {
        for name in shard.names() {
            let view = shard
                .tensor(name)
                .map_err(|e| anyhow::anyhow!("Gemma 4: cannot read header of '{name}': {e}"))?;
            headers.push((name.to_string(), view.dtype(), view.shape().to_vec()));
        }
    }
    Ok(headers)
}

fn classify_checkpoint(manifest: &Manifest, shards: &[SafeTensors]) -> Result<usize> {
    let headers = read_headers(shards)?;
    let observed: Vec<ObservedTensor> = headers
        .iter()
        .map(|(name, dtype, shape)| ObservedTensor {
            name,
            dtype: *dtype,
            shape,
        })
        .collect();
    let report = manifest.classify(&observed);
    report.check()?;
    let skipped = report.skipped_modality.len();
    info!(
        "Gemma 4 manifest: {} text tensors, {skipped} modality tensors skipped",
        observed.len() - skipped
    );
    Ok(skipped)
}

fn read_scalar_bf16(shards: &[SafeTensors], name: &str) -> Result<f32> {
    for shard in shards {
        if let Ok(view) = shard.tensor(name) {
            anyhow::ensure!(
                view.dtype() == Dtype::BF16 && view.data().len() == 2,
                "Gemma 4: '{name}' must be a single bf16 scalar"
            );
            let bits = u16::from_le_bytes([view.data()[0], view.data()[1]]);
            let value = half::bf16::from_bits(bits).to_f32();
            anyhow::ensure!(
                value.is_finite(),
                "Gemma 4: '{name}' = {value} is not finite"
            );
            return Ok(value);
        }
    }
    anyhow::bail!("Gemma 4: tensor '{name}' missing from every shard")
}

fn record_plan(
    loader: &mut StagedWeightLoader,
    shards: &[SafeTensors],
    manifest: &Manifest,
) -> Result<RecordedPlan> {
    let embed_tokens = record_matrix(loader, &manifest.embed_tokens)?;
    let norm = record_vector(loader, &manifest.norm)?;
    let mut layers = Vec::with_capacity(manifest.layers.len());
    for layer in &manifest.layers {
        let attention = &layer.attention;
        layers.push(LayerSlots {
            input_layernorm: record_vector(loader, &layer.input_layernorm)?,
            post_attention_layernorm: record_vector(loader, &layer.post_attention_layernorm)?,
            pre_feedforward_layernorm: record_vector(loader, &layer.pre_feedforward_layernorm)?,
            post_feedforward_layernorm: record_vector(loader, &layer.post_feedforward_layernorm)?,
            layer_scalar: read_scalar_bf16(shards, &layer.layer_scalar.name)?,
            q_proj: record_matrix(loader, &attention.q_proj)?,
            k_proj: record_matrix(loader, &attention.k_proj)?,
            v_proj: attention
                .v_proj
                .as_ref()
                .map(|v_proj| record_matrix(loader, v_proj))
                .transpose()?,
            o_proj: record_matrix(loader, &attention.o_proj)?,
            q_norm: record_vector(loader, &attention.q_norm)?,
            k_norm: record_vector(loader, &attention.k_norm)?,
            gate: record_matrix(loader, &layer.mlp.gate)?,
            up: record_matrix(loader, &layer.mlp.up)?,
            down: record_matrix(loader, &layer.mlp.down)?,
        });
    }
    Ok(RecordedPlan {
        embed_tokens,
        norm,
        layers,
    })
}

/// Only valid after a successful `finish`.
fn materialize(
    loader: &mut StagedWeightLoader,
    plan: RecordedPlan,
    config: Gemma4Config,
) -> Result<Gemma4Weights> {
    Ok(Gemma4Weights {
        embed_tokens: loader.take(plan.embed_tokens),
        norm: loader.take_vec(plan.norm),
        layers: plan
            .layers
            .into_iter()
            .map(|slots| -> Result<Gemma4Layer> {
                Ok(Gemma4Layer {
                    input_layernorm: loader.take_vec(slots.input_layernorm),
                    post_attention_layernorm: loader.take_vec(slots.post_attention_layernorm),
                    pre_feedforward_layernorm: loader.take_vec(slots.pre_feedforward_layernorm),
                    post_feedforward_layernorm: loader.take_vec(slots.post_feedforward_layernorm),
                    layer_scalar: slots.layer_scalar,
                    attention: Gemma4Attention {
                        q_proj: loader.take(slots.q_proj),
                        k_proj: loader.take(slots.k_proj),
                        v_proj: slots.v_proj.map(|slot| loader.take(slot)),
                        o_proj: loader.take(slots.o_proj),
                        q_norm: loader.take_vec(slots.q_norm),
                        k_norm: loader.take_vec(slots.k_norm),
                    },
                    mlp: Gemma4Mlp {
                        gate: loader.take(slots.gate),
                        up: loader.take(slots.up),
                        down: loader.take(slots.down),
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?,
        config,
    })
}

fn free_device_bytes() -> Result<usize> {
    let (free, _total) = cudarc::driver::result::mem_get_info()
        .map_err(|e| anyhow::anyhow!("Gemma 4: cuMemGetInfo failed: {e}"))?;
    Ok(free)
}

fn elapsed_ms(since: Instant) -> f64 {
    since.elapsed().as_secs_f64() * 1e3
}

fn gib(bytes: i64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

impl Gemma4Weights {
    /// Loads the text tower onto one device. Config, manifest and headers are
    /// all checked before a device context exists, so a checkpoint that does
    /// not match its config costs no GPU.
    /// The caller parses and validates the config before the multi-GiB
    /// load; taking it here keeps that validated copy the one the weights
    /// are built from.
    pub(crate) fn from_safetensors(
        model_path: &str,
        device_ordinal: usize,
        config: Gemma4Config,
    ) -> Result<(Self, LoadStats)> {
        let started = Instant::now();
        let manifest = Manifest::from_config(&config)?;
        let manifest_bytes = manifest.weight_bytes()?;

        let (shard_paths, weight_map) = load_shard_info(model_path)?;
        let prefetch = WeightPrefetch::spawn(&shard_paths);
        let mmaps = mmap_shards(&shard_paths)?;
        let shards = deserialize_shards(&mmaps)?;
        let skipped_modality_tensors = classify_checkpoint(&manifest, &shards)?;
        let validate_wall_ms = elapsed_ms(started);

        let ctx = DeviceContext::new_with_device(device_ordinal)?;
        let free_before = free_device_bytes()?;
        let mut loader = StagedWeightLoader::new(&ctx, &shards, &weight_map)?;

        let recording = Instant::now();
        let plan = record_plan(&mut loader, &shards, &manifest)?;
        let record_api_wall_ms = elapsed_ms(recording);

        let uploading = Instant::now();
        loader.finish()?;
        let execute_and_drain_wall_ms = elapsed_ms(uploading);

        let weights = materialize(&mut loader, plan, config)?;
        drop(loader);
        drop(prefetch);
        let device_free_bytes = free_device_bytes()?;
        drop(shards);
        // A few hundred ms at this size. Qwen3 backgrounds it to protect its
        // ready time; kept synchronous here so the reported total is the whole
        // cost. Lift that spawn into core once an executor wants it too.
        drop(mmaps);

        let stats = LoadStats {
            manifest_bytes,
            device_bytes: free_before as i64 - device_free_bytes as i64,
            device_free_bytes,
            validate_wall_ms,
            record_api_wall_ms,
            execute_and_drain_wall_ms,
            elapsed_ms: elapsed_ms(started),
            skipped_modality_tensors,
        };
        info!(
            "Gemma 4 weights resident: {:.2} GiB manifest, {:.2} GiB device, {:.2} GiB free, \
             {} modality tensors skipped; \
             {:.0} ms total, of which {:.0} validate, {:.0} record-api, {:.0} execute-and-drain",
            gib(stats.manifest_bytes as i64),
            gib(stats.device_bytes),
            gib(stats.device_free_bytes as i64),
            stats.skipped_modality_tensors,
            stats.elapsed_ms,
            stats.validate_wall_ms,
            stats.record_api_wall_ms,
            stats.execute_and_drain_wall_ms
        );
        Ok((weights, stats))
    }
}

#[cfg(test)]
mod tests {
    use anyhow::Context;

    use super::*;
    use crate::config::LayerKind;

    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

    /// Out of range on any host, so opening a device becomes a visible failure.
    /// Not `usize::MAX`, whose cast to the driver's `int` is -1 by accident.
    const UNOPENABLE_DEVICE: usize = 1024;

    fn model_path() -> Result<String> {
        std::env::var("PEGAINFER_TEST_MODEL_PATH")
            .context("PEGAINFER_TEST_MODEL_PATH must point to a Gemma 4 checkpoint directory")
    }

    /// No forward pass; residency only.
    ///
    /// ```text
    /// PEGAINFER_TEST_MODEL_PATH=/path/to/gemma-4-12B-it cargo test --release \
    ///   -p pegainfer-gemma4 --features gemma4 --lib \
    ///   weights::load::tests::loads_the_text_tower_and_reports_residency -- \
    ///   --exact --ignored --nocapture --test-threads=1
    /// ```
    #[test]
    #[ignore = "requires the pinned 12B checkpoint and a GPU"]
    fn loads_the_text_tower_and_reports_residency() -> Result<()> {
        let path = model_path()?;
        let config = Gemma4Config::from_file(&path)?;
        let (weights, stats) = Gemma4Weights::from_safetensors(&path, 0, config)?;

        let config = &weights.config;
        assert_eq!(weights.layers.len(), config.layer_types.len());
        for (layer, &kind) in weights.layers.iter().zip(&config.layer_types) {
            let attention = &layer.attention;
            match kind {
                LayerKind::Sliding => {
                    assert!(attention.v_proj.is_some(), "sliding layer without a v_proj");
                    assert_eq!(attention.q_norm.len, config.head_dim);
                }
                LayerKind::Global => {
                    assert!(attention.v_proj.is_none(), "global layer with a v_proj");
                    assert_eq!(attention.q_norm.len, config.global_head_dim);
                }
            }
            assert_eq!(attention.q_proj.cols, config.hidden_size);
            assert_eq!(attention.o_proj.rows, config.hidden_size);
        }
        assert_eq!(weights.embed_tokens.rows, config.vocab_size);
        assert_eq!(weights.embed_tokens.cols, config.hidden_size);

        let globals = config
            .layer_types
            .iter()
            .filter(|&&kind| kind == LayerKind::Global)
            .count();
        println!(
            "layers {} ({globals} global), {} modality tensors skipped\n\
             manifest {:.2} GiB, device {:.2} GiB, free {:.2} GiB\n\
             {:.0} ms total, of which {:.0} validate, {:.0} record-api, {:.0} execute-and-drain",
            weights.layers.len(),
            stats.skipped_modality_tensors,
            stats.manifest_bytes as f64 / GIB,
            stats.device_bytes as f64 / GIB,
            stats.device_free_bytes as f64 / GIB,
            stats.elapsed_ms,
            stats.validate_wall_ms,
            stats.record_api_wall_ms,
            stats.execute_and_drain_wall_ms,
        );
        Ok(())
    }

    /// Every faulty tensor named, before a device exists — hence the unopenable
    /// ordinal: a load that created its context first would fail with a driver
    /// error instead of the manifest's.
    #[test]
    #[ignore = "requires the pinned 12B checkpoint"]
    fn a_disagreeing_config_names_every_faulty_tensor() -> Result<()> {
        let path = model_path()?;
        let staged = tempfile::tempdir()?;
        for entry in std::fs::read_dir(&path)? {
            let entry = entry?;
            // Original name, resolved path: a Hugging Face snapshot points its
            // entries at `blobs/<hash>`, whose name is not the tensor file's.
            let target = entry.path().canonicalize()?;
            std::os::unix::fs::symlink(&target, staged.path().join(entry.file_name()))?;
        }

        let config_path = staged.path().join("config.json");
        let text = std::fs::read_to_string(format!("{path}/config.json"))?;
        let mut config: serde_json::Value = serde_json::from_str(&text)?;
        let width = config["text_config"]["intermediate_size"]
            .as_u64()
            .context("config.json has no text_config.intermediate_size")?;
        let hidden = config["text_config"]["hidden_size"]
            .as_u64()
            .context("config.json has no text_config.hidden_size")?;
        config["text_config"]["intermediate_size"] = serde_json::json!(width - 1);
        // Drop the symlink first: writing through it would edit the checkpoint.
        std::fs::remove_file(&config_path)?;
        std::fs::write(&config_path, serde_json::to_string(&config)?)?;

        let staged_path = staged.path().to_str().context("temp path is not UTF-8")?;
        let parsed = Gemma4Config::from_file(staged_path).expect("config");
        let err = Gemma4Weights::from_safetensors(staged_path, UNOPENABLE_DEVICE, parsed)
            .err()
            .context("a config that disagrees with the checkpoint was accepted")?
            .to_string();

        let layers = config["text_config"]["layer_types"]
            .as_array()
            .context("config.json has no text_config.layer_types")?
            .len();
        assert!(
            err.contains(&format!("{} fault(s)", layers * 3)),
            "expected one fault per MLP tensor per layer: {err}"
        );
        let named = format!(
            "model.language_model.layers.0.mlp.down_proj.weight: checkpoint has [{hidden}, \
             {width}], config implies [{hidden}, {}]",
            width - 1
        );
        assert!(err.contains(&named), "expected `{named}` in:\n{err}");
        Ok(())
    }
}
