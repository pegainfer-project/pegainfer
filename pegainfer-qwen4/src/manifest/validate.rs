//! Holds a checkpoint's tensors against the contract in one pass and reports every
//! fault at once. Lifted from `pegainfer-gemma4/src/manifest/validate.rs` with this
//! line's skip set and message.

use anyhow::Result;
use safetensors::Dtype;

use super::schema::Manifest;

/// Tensors this text-only line does not load. They share shards with text tensors, so a
/// loader classifies per tensor.
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
            if tensor.shape != shape.dims() {
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

    /// Skipped optional tensors are not faults.
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

    fn frozen_manifest() -> Manifest {
        let json: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/frozen_config.json")).unwrap();
        Manifest::from_config(&crate::config::Config::from_json(&json).unwrap())
    }

    /// Headers that match the manifest exactly, so these tests exercise the classifier
    /// alone.
    fn faultless(manifest: &Manifest) -> Vec<(String, Dtype, Vec<usize>)> {
        manifest
            .expected_tensors()
            .iter()
            .map(|(name, (shape, dtype))| (name.clone(), *dtype, shape.dims()))
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

    fn bend<'a>(
        tensors: &'a mut [(String, Dtype, Vec<usize>)],
        name: &str,
    ) -> &'a mut (String, Dtype, Vec<usize>) {
        tensors
            .iter_mut()
            .find(|(candidate, ..)| candidate == name)
            .unwrap_or_else(|| panic!("{name} is not in the derived manifest"))
    }

    /// All 1658 tensors of the frozen revision: the 1294 text tensors match the contract,
    /// and the skip set covers the 333 vision and 31 MTP tensors.
    #[test]
    fn the_contract_is_the_frozen_checkpoint() {
        let frozen = crate::manifest::frozen_tensors();
        let observed: Vec<ObservedTensor<'_>> = frozen
            .tensors
            .iter()
            .map(|(name, (dtype, shape))| ObservedTensor {
                name,
                dtype: *dtype,
                shape,
            })
            .collect();
        let report = frozen_manifest().classify(&observed);
        report.check().unwrap();
        assert_eq!(report.skipped_optional.len(), 333 + 31);
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
