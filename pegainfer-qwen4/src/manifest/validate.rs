//! The contract held up against what a checkpoint carries.
//!
//! Lifted nearly verbatim from `pegainfer-gemma4/src/manifest/validate.rs`; only
//! the skip prefixes and the fault message name this line. The classifier is the
//! part worth not rewriting: one pass over every header in every shard, all
//! faults aggregated into a single report, and skipped-optional tensors counted
//! rather than faulted.

use anyhow::Result;
use safetensors::Dtype;

use super::schema::Manifest;

/// Tensors this text-only line deliberately does not load. Skipping is mandatory
/// rather than optional here: the vision blocks are interleaved into the same
/// shards as the text layers, so a loader that reads whole shards must classify
/// per tensor. `no_skip_prefix_can_shadow_a_text_tensor` is what keeps this list
/// from ever hiding a required tensor.
const SKIPPED_OPTIONAL_PREFIXES: &[&str] = &["model.visual.", "mtp."];

pub(crate) struct ObservedTensor<'a> {
    pub(crate) name: &'a str,
    pub(crate) dtype: Dtype,
    pub(crate) shape: &'a [usize],
}

impl Manifest {
    pub(crate) fn classify(&self, observed: &[ObservedTensor]) -> ManifestReport {
        let mut expected = self.expected_tensors().clone();
        let mut report = ManifestReport::default();
        for tensor in observed {
            let Some((shape, dtype)) = expected.remove(tensor.name) else {
                if SKIPPED_OPTIONAL_PREFIXES
                    .iter()
                    .any(|prefix| tensor.name.starts_with(prefix))
                {
                    report.skipped_optional.push(tensor.name.to_string());
                } else {
                    report.unexpected.push(tensor.name.to_string());
                }
                continue;
            };
            if !shape.matches(tensor.shape) {
                report.shape_mismatch.push(format!(
                    "{}: checkpoint has {:?}, config implies {shape}",
                    tensor.name, tensor.shape
                ));
            }
            if tensor.dtype != dtype {
                report.dtype_mismatch.push(format!(
                    "{}: checkpoint has {:?}, expected {dtype:?}",
                    tensor.name, tensor.dtype
                ));
            }
        }
        report.missing = expected.into_keys().collect();
        report.sort();
        report
    }
}

#[derive(Default)]
pub(crate) struct ManifestReport {
    missing: Vec<String>,
    pub(crate) skipped_optional: Vec<String>,
    unexpected: Vec<String>,
    shape_mismatch: Vec<String>,
    dtype_mismatch: Vec<String>,
}

impl ManifestReport {
    fn sort(&mut self) {
        self.missing.sort();
        self.skipped_optional.sort();
        self.unexpected.sort();
        self.shape_mismatch.sort();
        self.dtype_mismatch.sort();
    }

    pub(crate) fn is_clean(&self) -> bool {
        self.missing.is_empty()
            && self.unexpected.is_empty()
            && self.shape_mismatch.is_empty()
            && self.dtype_mismatch.is_empty()
    }

    /// A caller that never opened every shard cannot say the checkpoint lacks the
    /// tensors it never looked for.
    pub(crate) fn forget_missing(&mut self) {
        self.missing.clear();
    }

    pub(crate) fn check(&self) -> Result<()> {
        self.check_with(&[])
    }

    /// Skipped optional tensors are not faults. `extra` adds fault classes that
    /// are not about the config contract at all — a stale index, a stray shard —
    /// so one pass names everything wrong with the checkpoint.
    pub(crate) fn check_with(&self, extra: &[(&str, Vec<String>)]) -> Result<()> {
        let faults: Vec<String> = [
            ("missing required text tensor", &self.missing),
            ("unexpected tensor", &self.unexpected),
            ("shape mismatch", &self.shape_mismatch),
            ("dtype mismatch", &self.dtype_mismatch),
        ]
        .into_iter()
        .flat_map(|(label, entries)| {
            entries
                .iter()
                .map(move |entry| format!("  {label}: {entry}"))
        })
        .chain(extra.iter().flat_map(|(label, entries)| {
            entries
                .iter()
                .map(move |entry| format!("  {label}: {entry}"))
        }))
        .collect();
        if faults.is_empty() {
            return Ok(());
        }
        anyhow::bail!(
            "{} checkpoint does not match the config, {} fault(s):\n{}",
            crate::MODEL_TYPE,
            faults.len(),
            faults.join("\n")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::schema::ExpectedShape;

    fn frozen_config() -> crate::config::Config {
        let json: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/frozen_config.json")).unwrap();
        crate::config::Config::from_json(&json).unwrap()
    }

    fn frozen_manifest() -> Manifest {
        Manifest::from_config(&frozen_config())
    }

    /// What a faultless checkpoint's headers look like, derived from the manifest
    /// itself — so these tests exercise the classifier, not the schema.
    fn faultless(manifest: &Manifest) -> Vec<(String, Dtype, Vec<usize>)> {
        manifest
            .expected_tensors()
            .iter()
            .map(|(name, (shape, dtype))| (name.clone(), *dtype, dims(*shape)))
            .collect()
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

    fn observe(tensors: &[(String, Dtype, Vec<usize>)]) -> Vec<ObservedTensor<'_>> {
        tensors
            .iter()
            .map(|(name, dtype, shape)| ObservedTensor {
                name,
                dtype: *dtype,
                shape,
            })
            .collect()
    }

    fn error_for(manifest: &Manifest, tensors: &[(String, Dtype, Vec<usize>)]) -> String {
        manifest
            .classify(&observe(tensors))
            .check()
            .expect_err("expected the manifest to reject this checkpoint")
            .to_string()
    }

    fn bend<'a>(
        tensors: &'a mut [(String, Dtype, Vec<usize>)],
        name: &str,
    ) -> &'a mut (String, Dtype, Vec<usize>) {
        tensors
            .iter_mut()
            .find(|(candidate, ..)| candidate == name)
            .unwrap_or_else(|| panic!("{name} is not in the derived manifest"))
    }

    // ---- the derived contract itself --------------------------------------

    /// Cross-checked against the frozen revision's `model.safetensors.index.json`:
    /// 1658 keys, of which 333 are `model.visual.*` and 31 are `mtp.*`, leaving
    /// exactly 1294 text tensors — and `lm_head.weight` is the only one of those
    /// outside the text prefix.
    #[test]
    fn the_derived_contract_is_the_frozen_checkpoint_text_set() {
        let manifest = frozen_manifest();
        assert_eq!(manifest.expected_tensors().len(), 1658 - 333 - 31);
        assert_eq!(
            manifest
                .expected_tensors()
                .keys()
                .filter(|name| !name.starts_with("model.language_model."))
                .collect::<Vec<_>>(),
            vec!["lm_head.weight"]
        );
        // 128 shards x 2,500,012 rows x 160 columns x 2 bytes; the ~51 GB the
        // ecosystem quotes for this table is the fp8 footprint, not this one.
        assert_eq!(
            frozen_config().ngram.bf16_bytes(),
            128 * 2_500_012 * 160 * 2
        );
        // The published repo is 360.0 GB across 131 shards, of which the skipped
        // vision and MTP tensors are a few GB; the text set must land under it.
        let bytes = manifest.weight_bytes();
        assert!(
            bytes > 350_000_000_000 && bytes < 360_000_000_000,
            "derived text-tensor bytes {bytes} are not consistent with the published 360.0 GB repo"
        );
    }

    #[test]
    fn every_layer_family_is_mounted_where_the_config_says() {
        let manifest = frozen_manifest();
        let names = manifest.expected_tensors();
        // 36 linear-attention layers, 12 full-attention layers, every 4th full.
        for index in 0..48 {
            let linear = names.contains_key(&format!(
                "model.language_model.layers.{index}.linear_attn.in_proj_qkv.weight"
            ));
            let full = names.contains_key(&format!(
                "model.language_model.layers.{index}.self_attn.q_proj.weight"
            ));
            assert_ne!(linear, full, "layer {index} must be exactly one kind");
            assert_eq!(full, (index + 1) % 4 == 0, "layer {index} kind");
        }
        // The PLE block is on `layers.1`, not `layers.2`: `ple_layer_ids` is
        // 1-indexed, and getting this wrong still produces fluent, wrong text.
        assert!(names.contains_key("model.language_model.layers.1.ple.key_proj.weight"));
        assert!(!names.contains_key("model.language_model.layers.2.ple.key_proj.weight"));
    }

    #[test]
    fn the_untied_bookend_requires_both_ends() {
        let manifest = frozen_manifest();
        let names = manifest.expected_tensors();
        // `lm_head.weight` is the one required tensor outside the text prefix.
        assert!(names.contains_key("lm_head.weight"));
        assert!(names.contains_key("model.language_model.embed_tokens.weight"));
        // There is no final norm: the readout is the hyper-connection mixer, and
        // it carries no injection weight.
        assert!(!names.contains_key("model.language_model.norm.weight"));
        assert!(names.contains_key("model.language_model.hyper_connection_mixer.hc_norm.weight"));
        assert!(!names.contains_key(
            "model.language_model.hyper_connection_mixer.block_inject_weight.weight"
        ));
    }

    #[test]
    fn the_ngram_table_is_128_named_shards_plus_three_integer_vectors() {
        let manifest = frozen_manifest();
        let tensors = manifest.expected_tensors();
        let shards = tensors
            .iter()
            .filter(|(name, _)| name.contains(".ngram_embedding.shard_"))
            .collect::<Vec<_>>();
        assert_eq!(shards.len(), 128);
        for (_, (shape, dtype)) in &shards {
            assert_eq!(
                *shape,
                ExpectedShape::Matrix {
                    rows: 2_500_012,
                    cols: 160
                }
            );
            assert_eq!(*dtype, Dtype::BF16);
        }
        for (name, len) in [
            ("layer_multipliers", 3usize),
            ("ngram_heads_offsets", 16),
            ("ngram_heads_vocab_sizes", 16),
        ] {
            let key = format!("model.language_model.layers.1.ple.ple_embedding.{name}");
            assert_eq!(
                tensors.get(&key).copied(),
                Some((ExpectedShape::Vector { len }, Dtype::I64)),
                "{key}"
            );
        }
    }

    /// The shared loader takes a 1-D tensor's length on trust and reinterprets its
    /// bytes, so a manifest that only checked matrices would leave every scalar
    /// vector unverified. These lengths are semantics, not decoration.
    #[test]
    fn one_dimensional_lengths_are_part_of_the_contract() {
        let manifest = frozen_manifest();
        let tensors = manifest.expected_tensors();
        let vectors = [
            ("model.language_model.layers.0.linear_attn.A_log", 48usize),
            ("model.language_model.layers.0.linear_attn.dt_bias", 48),
            ("model.language_model.layers.0.linear_attn.norm.weight", 128),
            ("model.language_model.layers.3.self_attn.q_norm.weight", 256),
            ("model.language_model.layers.3.self_attn.k_norm.weight", 256),
            (
                "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight",
                128,
            ),
            (
                "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight",
                10240,
            ),
        ];
        for (name, len) in vectors {
            assert_eq!(
                tensors.get(name).map(|(shape, _)| *shape),
                Some(ExpectedShape::Vector { len }),
                "{name}"
            );
        }
        // 2 hyper-connection norms per layer (48), 3 GDN scalars per linear layer
        // (36), 4 attention/indexer norms per full layer (12), the readout mixer's
        // norm, and the PLE block's 3 norms + 3 integer vectors.
        assert_eq!(
            tensors
                .iter()
                .filter(|(_, (shape, _))| matches!(shape, ExpectedShape::Vector { .. }))
                .count(),
            2 * 48 + 3 * 36 + 4 * 12 + 1 + 6,
            "every 1-D tensor must be length-checked, not taken on trust"
        );
    }

    #[test]
    fn rank_three_shapes_are_distinguished_from_matrices() {
        let manifest = frozen_manifest();
        let tensors = manifest.expected_tensors();
        assert_eq!(
            tensors
                .get("model.language_model.layers.0.linear_attn.conv1d.weight")
                .map(|(shape, _)| *shape),
            Some(ExpectedShape::Conv1d {
                channels: 10240,
                kernel: 4
            })
        );
        // The GDN conv and the PLE conv have the same stored shape but different
        // runtime dilation; the manifest can only pin the shape, so this asserts
        // both are present rather than conflated.
        assert_eq!(
            tensors
                .get("model.language_model.layers.1.ple.conv1d.weight")
                .map(|(shape, _)| *shape),
            Some(ExpectedShape::Conv1d {
                channels: 10240,
                kernel: 4
            })
        );
        assert_eq!(
            tensors
                .get("model.language_model.layers.0.mlp.experts.gate_up_proj")
                .map(|(shape, _)| *shape),
            Some(ExpectedShape::ExpertStack {
                experts: 512,
                rows: 1280,
                cols: 2560
            })
        );
        // Fused expert stacks carry no `.weight` suffix; asserting the suffixed
        // name is absent is what stops a silent rename from passing.
        assert!(
            !tensors.contains_key("model.language_model.layers.0.mlp.experts.gate_up_proj.weight")
        );
    }

    // ---- the five fault classes -------------------------------------------

    #[test]
    fn optional_tensors_are_skipped_not_faulted() {
        let manifest = frozen_manifest();
        let mut tensors = faultless(&manifest);
        for name in [
            "model.visual.patch_embed.proj.weight",
            "model.visual.blocks.0.attn.qkv.bias",
            "model.visual.merger.norm.weight",
            "mtp.fc_hidden.weight",
            "mtp.layers.0.self_attn.q_proj.weight",
        ] {
            tensors.push((name.to_string(), Dtype::BF16, vec![1152, 1152]));
        }
        let report = manifest.classify(&observe(&tensors));
        report.check().unwrap();
        assert_eq!(report.skipped_optional.len(), 5);
    }

    #[test]
    fn no_skip_prefix_can_shadow_a_text_tensor() {
        let manifest = frozen_manifest();
        for name in manifest.expected_tensors().keys() {
            for prefix in SKIPPED_OPTIONAL_PREFIXES {
                assert!(
                    !name.starts_with(prefix),
                    "required tensor {name} would be skipped as optional prefix {prefix}"
                );
            }
        }
    }

    #[test]
    fn a_missing_tensor_is_named() {
        let manifest = frozen_manifest();
        let dropped = "model.language_model.layers.3.self_attn.indexer.k_layernorm.weight";
        let tensors: Vec<_> = faultless(&manifest)
            .into_iter()
            .filter(|(name, ..)| name != dropped)
            .collect();
        let err = error_for(&manifest, &tensors);
        assert!(err.contains("missing required text tensor"), "{err}");
        assert!(err.contains(dropped), "{err}");
    }

    #[test]
    fn a_missing_ngram_shard_is_named() {
        let manifest = frozen_manifest();
        let dropped =
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_77.weight";
        let tensors: Vec<_> = faultless(&manifest)
            .into_iter()
            .filter(|(name, ..)| name != dropped)
            .collect();
        let err = error_for(&manifest, &tensors);
        assert!(err.contains("shard_77"), "{err}");
    }

    #[test]
    fn an_unexpected_tensor_is_named() {
        let manifest = frozen_manifest();
        let mut tensors = faultless(&manifest);
        let extra = "model.language_model.layers.0.linear_attn.in_proj_qkvz.weight";
        tensors.push((extra.to_string(), Dtype::BF16, vec![10240, 2560]));
        let err = error_for(&manifest, &tensors);
        assert!(err.contains("unexpected tensor"), "{err}");
        assert!(err.contains(extra), "{err}");
    }

    #[test]
    fn a_shape_mismatch_reports_both_shapes() {
        let manifest = frozen_manifest();
        let bent = "model.language_model.layers.0.mlp.experts.down_proj";
        let mut tensors = faultless(&manifest);
        bend(&mut tensors, bent).2 = vec![512, 2560, 639];
        let err = error_for(&manifest, &tensors);
        assert!(err.contains("shape mismatch"), "{err}");
        assert!(err.contains(bent), "{err}");
        assert!(err.contains("[512, 2560, 639]"), "{err}");
        assert!(err.contains("[512, 2560, 640]"), "{err}");
    }

    #[test]
    fn a_vector_of_the_wrong_length_is_a_shape_mismatch() {
        let manifest = frozen_manifest();
        let bent = "model.language_model.layers.0.linear_attn.A_log";
        let mut tensors = faultless(&manifest);
        bend(&mut tensors, bent).2 = vec![16];
        let err = error_for(&manifest, &tensors);
        assert!(err.contains(bent), "{err}");
        assert!(err.contains("[16]"), "{err}");
        assert!(err.contains("[48]"), "{err}");
    }

    #[test]
    fn a_dtype_mismatch_is_named() {
        let manifest = frozen_manifest();
        let recast = "model.language_model.layers.0.linear_attn.A_log";
        let mut tensors = faultless(&manifest);
        bend(&mut tensors, recast).1 = Dtype::F32;
        let err = error_for(&manifest, &tensors);
        assert!(err.contains("dtype mismatch"), "{err}");
        assert!(err.contains(recast), "{err}");
        assert!(err.contains("F32"), "{err}");
    }

    /// The n-gram partition metadata is integer, so a checkpoint that stored it
    /// as a weight dtype is a real fault rather than a harmless widening.
    #[test]
    fn integer_index_vectors_reject_a_weight_dtype() {
        let manifest = frozen_manifest();
        let recast = "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets";
        let mut tensors = faultless(&manifest);
        bend(&mut tensors, recast).1 = Dtype::BF16;
        let err = error_for(&manifest, &tensors);
        assert!(err.contains(recast), "{err}");
        assert!(err.contains("I64"), "{err}");
    }

    /// Requires the frozen checkpoint's `model.safetensors.index.json`: set
    /// `PEGAINFER_TEST_MODEL_PATH` to a directory containing it. Proves the
    /// derived contract *is* the checkpoint's text tensor set — not merely the
    /// same size — and that every key this line does not require falls under a
    /// declared skip prefix. This is the K3 `manifest_covers_the_checkpoint_index`
    /// pattern.
    ///
    /// The index is 170 KB and lists all 1658 keys, so this is the only pass that
    /// sees the *whole* checkpoint; `super::read`'s companion test is the one that
    /// checks real shapes and dtypes, but only over the shards on disk.
    #[test]
    #[ignore = "needs the frozen checkpoint's model.safetensors.index.json"]
    fn the_derived_contract_equals_the_real_checkpoint_index() {
        let dir = std::env::var("PEGAINFER_TEST_MODEL_PATH")
            .expect("PEGAINFER_TEST_MODEL_PATH must point at a checkpoint directory");
        let raw = std::fs::read_to_string(
            std::path::Path::new(&dir).join(crate::manifest::read::INDEX_FILE),
        )
        .expect("checkpoint directory has no model.safetensors.index.json");
        let index: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let weight_map = index["weight_map"].as_object().expect("weight_map");

        let manifest = frozen_manifest();
        let derived = manifest.expected_tensors();
        let report = manifest.classify(&observe(
            &weight_map
                .keys()
                // The index carries no dtypes or shapes, so this pass can only
                // check the *set*: the shape and dtype here are placeholders, and
                // the assertions below are on the missing / unexpected / skipped
                // classes alone.
                .map(|name| (name.clone(), Dtype::BF16, vec![0]))
                .collect::<Vec<_>>(),
        ));

        assert!(
            report.missing.is_empty(),
            "contract requires {} tensor(s) the checkpoint does not carry, e.g. {:?}",
            report.missing.len(),
            &report.missing[..report.missing.len().min(5)]
        );
        assert!(
            report.unexpected.is_empty(),
            "checkpoint carries {} tensor(s) the contract does not expect and no skip prefix \
             covers, e.g. {:?}",
            report.unexpected.len(),
            &report.unexpected[..report.unexpected.len().min(5)]
        );
        // Set equality, both directions already established above.
        assert_eq!(
            report.skipped_optional.len() + derived.len(),
            weight_map.len()
        );
    }

    #[test]
    fn every_fault_is_reported_in_one_pass() {
        let manifest = frozen_manifest();
        let dropped = "model.language_model.layers.1.ple.value_proj.weight";
        let mut tensors: Vec<_> = faultless(&manifest)
            .into_iter()
            .filter(|(name, ..)| name != dropped)
            .collect();
        tensors.push((
            "model.language_model.oops".to_string(),
            Dtype::BF16,
            vec![1],
        ));
        let bent = "model.language_model.layers.0.mlp.shared_expert.up_proj.weight";
        let entry = bend(&mut tensors, bent);
        entry.2 = vec![640, 2559];
        entry.1 = Dtype::F32;
        let err = error_for(&manifest, &tensors);
        assert!(err.contains("4 fault(s)"), "{err}");
        for expected in [dropped, "model.language_model.oops", bent] {
            assert!(err.contains(expected), "{err}");
        }
    }
}
