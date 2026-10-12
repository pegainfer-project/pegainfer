//! Turning a checkpoint directory into the vocabulary [`super::validate`] classifies.
//!
//! Two things a whole-checkpoint loader does not have to cope with, but this does:
//!
//! - **A partial download is the normal case here.** The frozen revision is 360 GB
//!   across 131 shards, so verification runs against whichever shards were
//!   fetched. A scan therefore reports the shards it never opened, and `check`
//!   refuses to call an unopened shard's tensors "missing".
//! - **The index can disagree with the shards.** `model.safetensors.index.json`
//!   is generated, so a re-sharded or truncated checkpoint leaves it stale. Each
//!   opened shard's header is compared against the names the index attributes to
//!   it, which turns a stale index into a named fault instead of a misleading
//!   wall of "missing".
//!
//! Headers are read through mmap + `SafeTensors::deserialize`, the path every
//! weights loader in this repo takes. That is not just consistency: `deserialize`
//! validates that the data offsets are contiguous and end exactly at the file's
//! end, so a truncated shard is rejected here rather than halfway through a
//! device copy, and no page of weight data is faulted in while only headers are
//! wanted.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::fs::File;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use memmap2::Mmap;
use safetensors::Dtype;
use safetensors::SafeTensors;

use super::schema::Manifest;
use super::validate::ManifestReport;
use super::validate::ObservedTensor;

pub(crate) const INDEX_FILE: &str = "model.safetensors.index.json";
const SHARD_SUFFIX: &str = ".safetensors";
/// A fault names its first few entries: a stale index can be wrong about a
/// thousand tensors at once and the first few are enough to diagnose it.
const NAMED_ENTRIES: usize = 3;

/// One tensor as its shard's header states it.
pub(crate) struct HeaderTensor {
    pub(crate) name: String,
    pub(crate) dtype: Dtype,
    pub(crate) shape: Vec<usize>,
    pub(crate) shard: String,
}

pub(crate) struct CheckpointScan {
    /// Shards the index names that were opened.
    pub(crate) present_shards: Vec<String>,
    /// Shards the index names that are not on disk.
    pub(crate) absent_shards: Vec<String>,
    /// `*.safetensors` files on disk that the index does not name.
    pub(crate) stray_shards: Vec<String>,
    /// Opened shards whose header and the index disagree about their contents.
    pub(crate) index_disagreement: Vec<String>,
    tensors: Vec<HeaderTensor>,
}

impl CheckpointScan {
    pub(crate) fn read(dir: &Path) -> Result<Self> {
        let index = read_index(dir)?;
        let mut scan = Self {
            present_shards: Vec::new(),
            absent_shards: Vec::new(),
            stray_shards: stray_shards(dir, &index)?,
            index_disagreement: Vec::new(),
            tensors: Vec::new(),
        };
        for (shard, indexed) in &index {
            let path = dir.join(shard);
            if !path.is_file() {
                scan.absent_shards.push(shard.clone());
                continue;
            }
            scan.present_shards.push(shard.clone());
            let file =
                File::open(&path).with_context(|| format!("cannot open {}", path.display()))?;
            // SAFETY: nothing else holds a write handle to a checkpoint shard.
            let mmap = unsafe { Mmap::map(&file) }
                .with_context(|| format!("cannot map {}", path.display()))?;
            let header = SafeTensors::deserialize(mmap.as_ref()).with_context(|| {
                format!("{} is not a readable safetensors file", path.display())
            })?;
            let mut carried = BTreeSet::new();
            for (name, view) in header.iter() {
                carried.insert(name.to_string());
                scan.tensors.push(HeaderTensor {
                    name: name.to_string(),
                    dtype: view.dtype(),
                    shape: view.shape().to_vec(),
                    shard: shard.clone(),
                });
            }
            if let Some(fault) = disagreement(shard, indexed, &carried) {
                scan.index_disagreement.push(fault);
            }
        }
        Ok(scan)
    }

    /// Every tensor the opened shards carry, in shard order.
    pub(crate) fn tensors(&self) -> &[HeaderTensor] {
        &self.tensors
    }

    pub(crate) fn report(&self, manifest: &Manifest) -> ManifestReport {
        let observed: Vec<ObservedTensor<'_>> = self
            .tensors
            .iter()
            .map(|tensor| ObservedTensor {
                name: &tensor.name,
                dtype: tensor.dtype,
                shape: &tensor.shape,
            })
            .collect();
        manifest.classify(&observed)
    }

    /// Every fault this scan can actually speak to. A partial scan still has to
    /// be right about the tensors it did open; it just cannot testify about the
    /// shards it never saw.
    pub(crate) fn check(&self, manifest: &Manifest) -> Result<()> {
        let mut report = self.report(manifest);
        if !self.absent_shards.is_empty() {
            report.forget_missing();
        }
        report.check_with(&[
            (
                "index disagrees with shard",
                self.index_disagreement.clone(),
            ),
            ("shard not named by the index", self.stray_shards.clone()),
        ])
    }
}

fn read_index(dir: &Path) -> Result<BTreeMap<String, BTreeSet<String>>> {
    let path = dir.join(INDEX_FILE);
    let raw = std::fs::read_to_string(&path)
        .with_context(|| format!("cannot read {}", path.display()))?;
    let json: serde_json::Value = serde_json::from_str(&raw)
        .with_context(|| format!("{} is not valid JSON", path.display()))?;
    let weight_map = json["weight_map"]
        .as_object()
        .with_context(|| format!("{} carries no weight_map object", path.display()))?;
    let mut index: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for (name, shard) in weight_map {
        let shard = shard
            .as_str()
            .with_context(|| format!("{INDEX_FILE} maps {name} to a non-string shard"))?;
        index
            .entry(shard.to_string())
            .or_default()
            .insert(name.clone());
    }
    Ok(index)
}

fn stray_shards(dir: &Path, index: &BTreeMap<String, BTreeSet<String>>) -> Result<Vec<String>> {
    let entries =
        std::fs::read_dir(dir).with_context(|| format!("cannot list {}", dir.display()))?;
    let mut stray = Vec::new();
    for entry in entries {
        let name = entry?.file_name().to_string_lossy().into_owned();
        if name.ends_with(SHARD_SUFFIX) && !index.contains_key(&name) {
            stray.push(name);
        }
    }
    stray.sort();
    Ok(stray)
}

fn disagreement(
    shard: &str,
    indexed: &BTreeSet<String>,
    carried: &BTreeSet<String>,
) -> Option<String> {
    let unattributed: Vec<&str> = carried.difference(indexed).map(String::as_str).collect();
    let absent: Vec<&str> = indexed.difference(carried).map(String::as_str).collect();
    if unattributed.is_empty() && absent.is_empty() {
        return None;
    }
    Some(format!(
        "{shard}: header carries {} the index does not attribute to it, index attributes {} the \
         header does not carry",
        summarize(&unattributed),
        summarize(&absent)
    ))
}

fn summarize(names: &[&str]) -> String {
    if names.is_empty() {
        return "nothing".to_string();
    }
    let head: Vec<&str> = names.iter().copied().take(NAMED_ENTRIES).collect();
    match names.len() - head.len() {
        0 => format!("{head:?}"),
        rest => format!("{head:?} and {rest} more"),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use safetensors::tensor::TensorView;
    use tempfile::TempDir;

    use super::*;
    use crate::manifest::INTEGER_INDEX_DTYPE;
    use crate::manifest::dtype_size;
    use crate::manifest::schema::ExpectedShape;

    /// The reader is contract-agnostic, so these tests run it against a config
    /// shrunk to the point where a whole checkpoint fits in a temp dir. The
    /// frozen revision's own shards run 800 MB to 3 GB and its n-gram table alone
    /// is 102 GB, so a faithful fixture of it could not be written at all. The
    /// real shards are covered by
    /// `the_derived_contract_holds_against_the_real_shards` below.
    fn small_config() -> serde_json::Value {
        let mut json: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/frozen_config.json")).unwrap();
        for (key, value) in [
            ("image_token_id", 2),
            ("video_token_id", 3),
            ("vision_start_token_id", 4),
            ("vision_end_token_id", 5),
        ] {
            json[key] = serde_json::json!(value);
        }
        let text = json["text_config"].as_object_mut().unwrap();
        for (key, value) in [
            ("bos_token_id", serde_json::json!(1)),
            ("eos_token_id", serde_json::json!(1)),
            ("vocab_size", serde_json::json!(64)),
            ("hidden_size", serde_json::json!(32)),
            ("num_hidden_layers", serde_json::json!(4)),
            (
                "layer_types",
                serde_json::json!([
                    "linear_attention",
                    "linear_attention",
                    "linear_attention",
                    "full_attention"
                ]),
            ),
            ("num_attention_heads", serde_json::json!(2)),
            ("num_key_value_heads", serde_json::json!(1)),
            ("linear_num_key_heads", serde_json::json!(1)),
            ("linear_num_value_heads", serde_json::json!(2)),
            ("num_experts", serde_json::json!(4)),
            ("num_experts_per_tok", serde_json::json!(2)),
            ("moe_intermediate_size", serde_json::json!(8)),
            ("shared_expert_intermediate_size", serde_json::json!(8)),
            ("hc_count", serde_json::json!(2)),
            ("hc_lowrank", serde_json::json!(4)),
            ("heads_per_ngram", serde_json::json!(2)),
            ("ple_embed_dim", serde_json::json!(32)),
            // Four primes above 99 pad to 424 rows, which split into four shards
            // of 106 rows — the whole table then costs a few kilobytes instead of
            // 102 GB.
            ("ngram_vocab_size_base", serde_json::json!(100)),
            ("make_ngram_vocab_size_divisible_by", serde_json::json!(8)),
            ("split_ngram_parts", serde_json::json!(4)),
        ] {
            text.insert(key.to_string(), value);
        }
        json
    }

    fn manifest_of(json: &serde_json::Value) -> Manifest {
        Manifest::from_config(&crate::config::Config::from_json(json).unwrap())
    }

    fn small_manifest() -> Manifest {
        manifest_of(&small_config())
    }

    fn frozen_manifest() -> Manifest {
        manifest_of(&serde_json::from_str(include_str!("../../tests/frozen_config.json")).unwrap())
    }

    fn dims(shape: ExpectedShape) -> Vec<usize> {
        match shape {
            ExpectedShape::Matrix { rows, cols } => vec![rows, cols],
            ExpectedShape::Vector { len } => vec![len],
            ExpectedShape::Conv1d { channels, kernel } => vec![channels, 1, kernel],
            ExpectedShape::ExpertStack {
                experts,
                rows,
                cols,
            } => vec![experts, rows, cols],
        }
    }

    /// The contract's own tensors, in the sorted order the shard split uses, so a
    /// checkpoint built from this is faultless by construction and these tests
    /// exercise the reader rather than the schema.
    fn contract(manifest: &Manifest) -> Vec<(String, Dtype, Vec<usize>)> {
        manifest
            .expected_tensors()
            .iter()
            .map(|(name, (shape, dtype))| (name.clone(), *dtype, dims(*shape)))
            .collect()
    }

    /// Writes a shard holding `tensors`, zero-filled. Only the header is ever read
    /// back, so the bytes are padding.
    fn write_shard(dir: &Path, name: &str, tensors: &[(String, Dtype, Vec<usize>)]) {
        let data: Vec<Vec<u8>> = tensors
            .iter()
            .map(|(_, dtype, shape)| {
                vec![0u8; shape.iter().product::<usize>() * dtype_size(*dtype)]
            })
            .collect();
        let views: Vec<(String, TensorView<'_>)> = tensors
            .iter()
            .zip(&data)
            .map(|((name, dtype, shape), bytes)| {
                (
                    name.clone(),
                    TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            })
            .collect();
        safetensors::tensor::serialize_to_file(views, None, &dir.join(name)).unwrap();
    }

    fn write_index(dir: &Path, weight_map: &BTreeMap<String, String>) {
        let index = serde_json::json!({ "weight_map": weight_map });
        std::fs::write(dir.join(INDEX_FILE), index.to_string()).unwrap();
    }

    fn shard_name(chunk: usize, total: usize) -> String {
        format!("model-{chunk:05}-of-{total:05}.safetensors")
    }

    /// A faithful checkpoint spread over `shards` shards, plus its index. Returns
    /// the shard count it actually used, which `div_ceil` can round down from.
    fn faithful_checkpoint(shards: usize) -> (TempDir, Manifest, usize) {
        let manifest = small_manifest();
        let tensors = contract(&manifest);
        let per_shard = tensors.len().div_ceil(shards);
        let total = tensors.len().div_ceil(per_shard);
        let dir = TempDir::new().unwrap();
        let mut weight_map = BTreeMap::new();
        for chunk in 0..total {
            let slice = &tensors[chunk * per_shard..((chunk + 1) * per_shard).min(tensors.len())];
            let name = shard_name(chunk, total);
            for (tensor, ..) in slice {
                weight_map.insert(tensor.clone(), name.clone());
            }
            write_shard(dir.path(), &name, slice);
        }
        write_index(dir.path(), &weight_map);
        (dir, manifest, total)
    }

    /// Deletes every shard but the first `keep` and leaves the index untouched —
    /// which is exactly what a partial download looks like.
    fn keep_only_shards(dir: &Path, keep: usize) {
        let mut shards: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(SHARD_SUFFIX))
            .collect();
        shards.sort();
        for name in shards.into_iter().skip(keep) {
            std::fs::remove_file(dir.join(name)).unwrap();
        }
    }

    // ---- the happy paths ---------------------------------------------------

    #[test]
    fn a_full_scan_of_a_faithful_checkpoint_is_clean() {
        let (dir, manifest, total) = faithful_checkpoint(3);
        let scan = CheckpointScan::read(dir.path()).unwrap();
        scan.check(&manifest).unwrap();
        assert!(scan.absent_shards.is_empty());
        assert!(scan.stray_shards.is_empty());
        assert!(scan.index_disagreement.is_empty());
        assert_eq!(scan.present_shards.len(), total);
        assert_eq!(scan.tensors().len(), manifest.expected_tensors().len());
    }

    #[test]
    fn a_partial_scan_names_the_shards_it_never_opened_and_stays_clean() {
        let (dir, manifest, total) = faithful_checkpoint(3);
        keep_only_shards(dir.path(), 1);
        let scan = CheckpointScan::read(dir.path()).unwrap();
        // It cannot testify about what it did not open, so "missing" is not a
        // fault — but nothing else may be one either.
        scan.check(&manifest).unwrap();
        assert_eq!(scan.present_shards.len(), 1);
        assert_eq!(
            scan.absent_shards,
            (1..total)
                .map(|chunk| shard_name(chunk, total))
                .collect::<Vec<_>>()
        );
        assert!(
            !scan.report(&manifest).is_clean(),
            "the raw report must still name the unopened tensors as missing, so a caller that \
             believes it holds the whole checkpoint cannot be quietly excused"
        );
    }

    // ---- the faults --------------------------------------------------------

    #[test]
    fn a_real_shape_fault_survives_a_partial_scan() {
        let (dir, manifest, total) = faithful_checkpoint(3);
        keep_only_shards(dir.path(), 1);
        let tensors = contract(&manifest);
        let per_shard = tensors.len().div_ceil(total);
        let mut first = tensors[..per_shard].to_vec();
        // A fused expert stack stored one expert short.
        let entry = first
            .iter_mut()
            .find(|(name, ..)| name.ends_with(".mlp.experts.gate_up_proj"))
            .expect("layer 0's expert stack sorts into the first shard");
        entry.2[0] -= 1;
        let (bent, reported) = (entry.0.clone(), format!("{:?}", entry.2));
        write_shard(dir.path(), &shard_name(0, total), &first);

        let scan = CheckpointScan::read(dir.path()).unwrap();
        let err = scan.check(&manifest).unwrap_err().to_string();
        assert!(err.contains("shape mismatch"), "{err}");
        assert!(err.contains(&bent), "{err}");
        assert!(err.contains(&reported), "{err}");
        // Still excused from the shards it never opened.
        assert!(!err.contains("missing required text tensor"), "{err}");
    }

    #[test]
    fn a_real_dtype_fault_survives_a_partial_scan() {
        let (dir, manifest, total) = faithful_checkpoint(3);
        keep_only_shards(dir.path(), 1);
        let tensors = contract(&manifest);
        let per_shard = tensors.len().div_ceil(total);
        let mut first = tensors[..per_shard].to_vec();
        // The recast a quantized sibling ships: f32 where the contract says bf16.
        let recast = "model.language_model.embed_tokens.weight";
        first
            .iter_mut()
            .find(|(name, ..)| name == recast)
            .expect("the embedding sorts into the first shard")
            .1 = Dtype::F32;
        write_shard(dir.path(), &shard_name(0, total), &first);

        let scan = CheckpointScan::read(dir.path()).unwrap();
        let err = scan.check(&manifest).unwrap_err().to_string();
        assert!(err.contains("dtype mismatch"), "{err}");
        assert!(err.contains(recast), "{err}");
        assert!(err.contains("F32"), "{err}");
    }

    #[test]
    fn a_stale_index_is_named_in_both_directions() {
        let (dir, manifest, total) = faithful_checkpoint(3);
        let tensors = contract(&manifest);
        // Re-point the first tensor at a later shard and the last tensor at the
        // first: one shard then carries a tensor the index denies, and another is
        // credited with a tensor it does not have.
        let unattributed = tensors.first().unwrap().0.clone();
        let absent = tensors.last().unwrap().0.clone();
        let raw = std::fs::read_to_string(dir.path().join(INDEX_FILE)).unwrap();
        let mut index: serde_json::Value = serde_json::from_str(&raw).unwrap();
        index["weight_map"][unattributed.as_str()] = serde_json::json!(shard_name(1, total));
        index["weight_map"][absent.as_str()] = serde_json::json!(shard_name(0, total));
        std::fs::write(dir.path().join(INDEX_FILE), index.to_string()).unwrap();

        let scan = CheckpointScan::read(dir.path()).unwrap();
        let err = scan.check(&manifest).unwrap_err().to_string();
        assert!(err.contains("index disagrees with shard"), "{err}");
        assert!(err.contains(&unattributed), "{err}");
        assert!(err.contains(&absent), "{err}");
    }

    #[test]
    fn a_shard_the_index_does_not_name_is_refused() {
        let (dir, manifest, total) = faithful_checkpoint(3);
        let stray = format!("model-00009-of-{total:05}.safetensors");
        std::fs::write(dir.path().join(&stray), b"junk").unwrap();
        let scan = CheckpointScan::read(dir.path()).unwrap();
        assert_eq!(scan.stray_shards, vec![stray.clone()]);
        let err = scan.check(&manifest).unwrap_err().to_string();
        assert!(err.contains("shard not named by the index"), "{err}");
        assert!(err.contains(&stray), "{err}");
    }

    #[test]
    fn a_truncated_shard_is_refused_before_any_weight_is_read() {
        let (dir, manifest, total) = faithful_checkpoint(3);
        let shard = dir.path().join(shard_name(1, total));
        let len = std::fs::metadata(&shard).unwrap().len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&shard)
            .unwrap()
            .set_len(len / 2)
            .unwrap();
        let err = CheckpointScan::read(dir.path())
            .err()
            .expect("a truncated shard must be refused")
            .to_string();
        assert!(err.contains("not a readable safetensors file"), "{err}");
        assert!(err.contains(&shard_name(1, total)), "{err}");
        assert!(!manifest.expected_tensors().is_empty());
    }

    #[test]
    fn a_missing_index_is_refused_with_its_path() {
        let dir = TempDir::new().unwrap();
        let err = CheckpointScan::read(dir.path())
            .err()
            .expect("a directory with no index must be refused")
            .to_string();
        assert!(err.contains(INDEX_FILE), "{err}");
        assert!(err.contains(&dir.path().display().to_string()), "{err}");
    }

    // ---- what the reader actually reads ------------------------------------

    #[test]
    fn the_reader_carries_shapes_and_dtypes_not_just_names() {
        let (dir, manifest, _) = faithful_checkpoint(3);
        let scan = CheckpointScan::read(dir.path()).unwrap();
        let expected = manifest.expected_tensors();
        for tensor in scan.tensors() {
            let (shape, dtype) = expected
                .get(tensor.name.as_str())
                .unwrap_or_else(|| panic!("{} is not in the contract", tensor.name));
            assert_eq!(tensor.dtype, *dtype, "{}", tensor.name);
            assert_eq!(tensor.shape, dims(*shape), "{}", tensor.name);
        }
        // Rank is part of what comes back: the depthwise conv weight and the fused
        // expert stacks are rank 3, and a reader that flattened shapes to element
        // counts would still pass every assertion above.
        let ranks: Vec<usize> = scan
            .tensors()
            .iter()
            .map(|tensor| tensor.shape.len())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        assert_eq!(ranks, vec![1, 2, 3]);
        assert_eq!(
            scan.tensors()
                .iter()
                .filter(|tensor| tensor.dtype == INTEGER_INDEX_DTYPE)
                .count(),
            3,
            "layer_multipliers plus the two per-head n-gram index vectors"
        );
    }

    #[test]
    fn every_tensor_is_read_from_the_shard_the_index_points_at() {
        let (dir, manifest, _) = faithful_checkpoint(3);
        let raw = std::fs::read_to_string(dir.path().join(INDEX_FILE)).unwrap();
        let index: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let weight_map = index["weight_map"].as_object().unwrap();
        let scan = CheckpointScan::read(dir.path()).unwrap();
        let mut seen: HashMap<&str, usize> = HashMap::new();
        for tensor in scan.tensors() {
            assert_eq!(
                weight_map[tensor.name.as_str()].as_str().unwrap(),
                tensor.shard,
                "{} was read from a shard the index does not point at",
                tensor.name
            );
            *seen.entry(tensor.name.as_str()).or_default() += 1;
        }
        assert_eq!(seen.len(), manifest.expected_tensors().len());
        assert!(seen.values().all(|count| *count == 1));
    }

    #[test]
    fn the_index_file_name_is_the_published_one() {
        assert_eq!(INDEX_FILE, "model.safetensors.index.json");
        assert!(PathBuf::from(INDEX_FILE).is_relative());
    }

    /// Runs against a real checkpoint directory: `PEGAINFER_TEST_MODEL_PATH`
    /// pointing at one that carries `model.safetensors.index.json` and at least
    /// one shard. A partial directory is accepted — that is the point of the scan
    /// — but every shard it does open has to agree with the derived contract on
    /// names, shapes and dtypes, and with the index about its own contents.
    #[test]
    #[ignore = "needs real Qwen3.8-Flash-Next shards on disk"]
    fn the_derived_contract_holds_against_the_real_shards() {
        let dir = std::env::var("PEGAINFER_TEST_MODEL_PATH")
            .expect("PEGAINFER_TEST_MODEL_PATH must point at a checkpoint directory");
        let manifest = frozen_manifest();
        let scan = CheckpointScan::read(Path::new(&dir))
            .unwrap_or_else(|e| panic!("cannot scan {dir}: {e:#}"));
        assert!(
            !scan.present_shards.is_empty(),
            "no shard to check in {dir}"
        );
        scan.check(&manifest)
            .unwrap_or_else(|e| panic!("real shards disagree with the derived contract: {e:#}"));
        // #1104 asks that a proof record which shards it used, so a reader knows
        // whether the gate ever saw the whole checkpoint. Name them rather than
        // just counting: "3 of 131" does not say which three. The required/skipped
        // split is printed too, so the "446 opened / 111 required" figure the
        // bring-up doc quotes is this gate's own output, not a side count.
        let report = scan.report(&manifest);
        eprintln!(
            "opened {} of {} indexed shard(s): {} tensor(s) checked, {} required text tensors, {} \
             skipped optional; shards: {}",
            scan.present_shards.len(),
            scan.present_shards.len() + scan.absent_shards.len(),
            scan.tensors().len(),
            scan.tensors().len() - report.skipped_optional.len(),
            report.skipped_optional.len(),
            scan.present_shards.join(" "),
        );
    }
}
