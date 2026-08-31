//! The contract held up against what a checkpoint carries.

use anyhow::Result;
use safetensors::Dtype;

use super::schema::Manifest;

/// Spans every published size; a checkpoint matches only its own.
const OPTIONAL_MODALITY_PREFIXES: &[&str] = &[
    "model.embed_audio.",
    "model.embed_vision.",
    "model.vision_embedder.",
    "model.vision_tower.",
];

pub(crate) struct ObservedTensor<'a> {
    pub(crate) name: &'a str,
    pub(crate) dtype: Dtype,
    pub(crate) shape: &'a [usize],
}

impl Manifest {
    pub(crate) fn classify(&self, observed: &[ObservedTensor]) -> ManifestReport {
        let mut expected = self.expected_tensors();
        let mut report = ManifestReport::default();
        for tensor in observed {
            let Some((shape, dtype)) = expected.remove(tensor.name) else {
                if OPTIONAL_MODALITY_PREFIXES
                    .iter()
                    .any(|prefix| tensor.name.starts_with(prefix))
                {
                    report.skipped_modality.push(tensor.name.to_string());
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
        report.missing = expected.into_keys().map(str::to_string).collect();
        report.sort();
        report
    }
}

#[derive(Default)]
pub(crate) struct ManifestReport {
    missing: Vec<String>,
    pub(crate) skipped_modality: Vec<String>,
    unexpected: Vec<String>,
    shape_mismatch: Vec<String>,
    dtype_mismatch: Vec<String>,
}

impl ManifestReport {
    fn sort(&mut self) {
        self.missing.sort();
        self.skipped_modality.sort();
        self.unexpected.sort();
        self.shape_mismatch.sort();
        self.dtype_mismatch.sort();
    }

    /// Skipped modality tensors are not faults.
    pub(crate) fn check(&self) -> Result<()> {
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
        .collect();
        if faults.is_empty() {
            return Ok(());
        }
        anyhow::bail!(
            "Gemma 4 checkpoint does not match the config, {} fault(s):\n{}",
            faults.len(),
            faults.join("\n")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::schema::ExpectedShape;
    use crate::manifest::schema::sample_config as config;

    fn faultless(manifest: &Manifest) -> Vec<(String, Dtype, Vec<usize>)> {
        manifest
            .expected_tensors()
            .into_iter()
            .map(|(name, (shape, dtype))| {
                let dims = match shape {
                    ExpectedShape::Matrix { rows, cols } => vec![rows, cols],
                    ExpectedShape::Vector { len } => vec![len],
                    ExpectedShape::Scalar => vec![],
                };
                (name.to_string(), dtype, dims)
            })
            .collect()
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

    #[test]
    fn faultless_checkpoint_is_accepted() {
        let manifest = Manifest::from_config(&config()).unwrap();
        let tensors = faultless(&manifest);
        let report = manifest.classify(&observe(&tensors));
        report.check().unwrap();
        assert!(report.skipped_modality.is_empty());
    }

    #[test]
    fn optional_modality_tensors_are_skipped_not_faulted() {
        let manifest = Manifest::from_config(&config()).unwrap();
        let mut tensors = faultless(&manifest);
        for name in [
            "model.embed_audio.embedding_projection.weight",
            "model.embed_vision.embedding_projection.weight",
            "model.vision_embedder.patch_dense.weight",
            "model.vision_tower.encoder.layers.0.mlp.fc1.weight",
        ] {
            tensors.push((name.to_string(), Dtype::BF16, vec![3840, 3840]));
        }
        let report = manifest.classify(&observe(&tensors));
        report.check().unwrap();
        assert_eq!(report.skipped_modality.len(), 4);
    }

    #[test]
    fn no_modality_prefix_can_shadow_a_text_tensor() {
        let manifest = Manifest::from_config(&config()).unwrap();
        for name in manifest.expected_tensors().keys() {
            for prefix in OPTIONAL_MODALITY_PREFIXES {
                assert!(
                    !name.starts_with(prefix),
                    "required tensor {name} would be skipped as modality prefix {prefix}"
                );
            }
        }
    }

    #[test]
    fn missing_text_tensor_is_named() {
        let manifest = Manifest::from_config(&config()).unwrap();
        let dropped = "model.language_model.layers.2.self_attn.k_norm.weight";
        let tensors: Vec<_> = faultless(&manifest)
            .into_iter()
            .filter(|(name, ..)| name != dropped)
            .collect();
        let err = error_for(&manifest, &tensors);
        assert!(err.contains("missing required text tensor"), "{err}");
        assert!(err.contains(dropped), "{err}");
    }

    #[test]
    fn unexpected_tensor_is_named() {
        let manifest = Manifest::from_config(&config()).unwrap();
        let mut tensors = faultless(&manifest);
        let extra = "model.language_model.layers.0.self_attn.v_bias";
        tensors.push((extra.to_string(), Dtype::BF16, vec![2048]));
        let err = error_for(&manifest, &tensors);
        assert!(err.contains("unexpected tensor"), "{err}");
        assert!(err.contains(extra), "{err}");
    }

    #[test]
    fn shape_mismatch_reports_both_shapes() {
        let manifest = Manifest::from_config(&config()).unwrap();
        let bent = "model.language_model.layers.0.mlp.down_proj.weight";
        let mut tensors = faultless(&manifest);
        let entry = tensors.iter_mut().find(|(name, ..)| name == bent).unwrap();
        entry.2 = vec![3840, 15359];
        let err = error_for(&manifest, &tensors);
        assert!(err.contains("shape mismatch"), "{err}");
        assert!(err.contains(bent), "{err}");
        assert!(err.contains("[3840, 15359]"), "{err}");
        assert!(err.contains("[3840, 15360]"), "{err}");
    }

    #[test]
    fn dtype_mismatch_is_named() {
        let manifest = Manifest::from_config(&config()).unwrap();
        let recast = "model.language_model.norm.weight";
        let mut tensors = faultless(&manifest);
        let entry = tensors
            .iter_mut()
            .find(|(name, ..)| name == recast)
            .unwrap();
        entry.1 = Dtype::F32;
        let err = error_for(&manifest, &tensors);
        assert!(err.contains("dtype mismatch"), "{err}");
        assert!(err.contains(recast), "{err}");
        assert!(err.contains("F32"), "{err}");
    }

    #[test]
    fn every_fault_is_reported_in_one_pass() {
        let manifest = Manifest::from_config(&config()).unwrap();
        let dropped = "model.language_model.layers.1.self_attn.v_proj.weight";
        let mut tensors: Vec<_> = faultless(&manifest)
            .into_iter()
            .filter(|(name, ..)| name != dropped)
            .collect();
        tensors.push((
            "model.language_model.oops".to_string(),
            Dtype::BF16,
            vec![1],
        ));
        let bent = "model.language_model.layers.0.mlp.up_proj.weight";
        let entry = tensors.iter_mut().find(|(name, ..)| name == bent).unwrap();
        entry.2 = vec![15360, 3839];
        entry.1 = Dtype::F32;
        let err = error_for(&manifest, &tensors);
        assert!(err.contains("4 fault(s)"), "{err}");
        for expected in [dropped, "model.language_model.oops", bent] {
            assert!(err.contains(expected), "{err}");
        }
    }
}
