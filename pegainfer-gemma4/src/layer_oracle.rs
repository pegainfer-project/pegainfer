use super::*;
use crate::config::Gemma4Config;
use crate::config::LayerKind;
use crate::config::first_last;
use crate::testkit::GOLDEN_PATH;
use crate::testkit::METADATA_KEY;
use crate::testkit::assert_checkpoint_matches;
use crate::testkit::bf16_tensor;
use crate::testkit::model_path;
use crate::weights::Gemma4Weights;

/// Declared against the measured error structure on the pinned 12B
/// checkpoint (sm_89): the one-token case is bitwise exact for both
/// probe layers, and the nine-token case shows scattered rounding noise
/// only — worst element 0.25 absolute, token 0 exact, errors appearing
/// once softmax runs over multiple keys. That is the unscaled-attention
/// signature (logits carry no rsqrt damping, so one-ulp bf16 GEMM
/// differences shift softmax weights), not a layout defect, which would
/// scramble whole head blocks by O(1). ABS_TOL leaves 1.6x headroom over
/// the measured worst; wiring-class bugs (a 16x scale error, swapped
/// weights) blow past it by orders of magnitude.
const ABS_TOL: f32 = 0.4;
const REL_TOL: f32 = 2e-2;

/// Reports the full error structure before asserting, so a failure shows
/// whether it is scattered rounding noise or a structural pattern (whole
/// tokens or channel blocks off — the signature of a layout bug).
fn compare(got: &[f32], expected: &[bf16], hidden_size: usize, what: &str) -> usize {
    assert_eq!(got.len(), expected.len(), "{what} length");
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut worst_idx = 0usize;
    let mut violations = 0usize;
    let mut per_token_max: Vec<f32> = vec![0.0; got.len().div_ceil(hidden_size)];
    for (i, (&g, &e)) in got.iter().zip(expected).enumerate() {
        let e = e.to_f32();
        if !g.is_finite() || !e.is_finite() {
            violations += 1;
            per_token_max[i / hidden_size] = f32::INFINITY;
            continue;
        }
        let abs = (g - e).abs();
        let rel = if e.abs() > 1e-3 { abs / e.abs() } else { 0.0 };
        if abs > ABS_TOL + REL_TOL * e.abs() {
            violations += 1;
        }
        if abs > max_abs {
            max_abs = abs;
            worst_idx = i;
        }
        max_rel = max_rel.max(rel);
        per_token_max[i / hidden_size] = per_token_max[i / hidden_size].max(abs);
    }
    eprintln!(
        "{what}: max_abs {max_abs} at (token {}, ch {}), max_rel {max_rel}, \
         {violations}/{} over tolerance, per-token max_abs {per_token_max:?}",
        worst_idx / hidden_size,
        worst_idx % hidden_size,
        got.len()
    );
    violations
}

#[test]
#[ignore = "requires the pinned 12B checkpoint via PEGAINFER_TEST_MODEL_PATH and a GPU"]
fn layers_match_hf_probes() {
    let dir = model_path();
    let fixture_bytes = std::fs::read(GOLDEN_PATH).expect("read fixture");
    let (_, meta) =
        safetensors::SafeTensors::read_metadata(&fixture_bytes).expect("fixture metadata");
    let manifest: serde_json::Value = serde_json::from_str(
        meta.metadata()
            .as_ref()
            .expect("fixture metadata map")
            .get(METADATA_KEY)
            .expect("gemma4_golden metadata key"),
    )
    .expect("parse fixture manifest");
    assert_checkpoint_matches(&manifest, &dir);
    let fixture = safetensors::SafeTensors::deserialize(&fixture_bytes).expect("parse fixture");

    // The layer indices come from parsing the layer map; the fixture
    // metadata records the dumper's own parse, so the two independent
    // derivations must agree before anything numeric is asserted.
    let config = Gemma4Config::from_file(&dir).expect("config");
    let (first, last) =
        first_last(&config.layer_types, LayerKind::Sliding).expect("12B carries sliding layers");
    let (gfirst, glast) =
        first_last(&config.layer_types, LayerKind::Global).expect("12B carries global layers");
    let probe_layers = &manifest["probe_layers"];
    assert_eq!(
        first as u64,
        probe_layers["sliding_first"]
            .as_u64()
            .expect("sliding_first"),
        "first sliding layer disagrees with the fixture's parse"
    );
    assert_eq!(
        last as u64,
        probe_layers["sliding_last"].as_u64().expect("sliding_last"),
        "last sliding layer disagrees with the fixture's parse"
    );
    assert_eq!(
        gfirst as u64,
        probe_layers["global_first"].as_u64().expect("global_first"),
        "first global layer disagrees with the fixture's parse"
    );
    assert_eq!(
        glast as u64,
        probe_layers["global_last"].as_u64().expect("global_last"),
        "last global layer disagrees with the fixture's parse"
    );
    assert_eq!(
        glast,
        last + 1,
        "global_last must sit directly after sliding_last for the \
         dedup label below to be its input"
    );

    let cut_labels: Vec<String> = manifest["cut_labels"]
        .as_array()
        .expect("cut_labels")
        .iter()
        .map(|v| v.as_str().expect("cut label").to_string())
        .collect();
    let cut_index = |label: &str| {
        cut_labels
            .iter()
            .position(|l| l == label)
            .unwrap_or_else(|| panic!("cut {label} missing from fixture"))
    };

    let (weights, _) =
        Gemma4Weights::from_safetensors(&dir, 0, config.clone()).expect("load 12B weights");
    let ctx = DeviceContext::new_with_device(0).expect("device context");

    let geom = LayerGeometry::local_of(&config);
    let global_geom = LayerGeometry::global_of(&config);
    let cos_max_pos = 16;
    let (cos_cache, sin_cache) = pegainfer_core::rope::precompute_rope(
        &ctx,
        &pegainfer_core::rope::RopeTableSpec {
            rotary_dim: geom.head_dim,
            frequency_dim: geom.head_dim,
            max_seq_len: cos_max_pos,
            theta: config.sliding_rope_theta,
        },
    )
    .expect("rope tables");
    let (gcos_cache, gsin_cache) = build_proportional_rope_tables(
        &ctx,
        config.global_rope_theta,
        global_geom.head_dim,
        config.global_rotary_dim,
        cos_max_pos,
    )
    .expect("proportional rope tables");

    let mut over_tolerance: Vec<String> = Vec::new();
    for case in ["single", "short"] {
        let (shape, hidden) = bf16_tensor(&fixture, &format!("{case}_hidden"));
        assert_eq!(shape.len(), 3, "{case}_hidden rank");
        assert_eq!(shape[0], cut_labels.len(), "{case}_hidden cut count");
        let (seq_len, hidden_size) = (shape[1], shape[2]);
        assert_eq!(hidden_size, geom.hidden_size, "{case}_hidden width");
        let cut = |label: &str| {
            let i = cut_index(label);
            &hidden[i * seq_len * hidden_size..(i + 1) * seq_len * hidden_size]
        };

        for (name, index) in [("sliding_first", first), ("sliding_last", last)] {
            let x = HiddenStates::from_host(&ctx, cut(&format!("{name}_in")), hidden_size, seq_len)
                .expect("x H2D");
            let out = local_layer_forward(
                &ctx,
                &weights.layers[index],
                &geom,
                &x,
                0,
                config.sliding_window,
                &cos_cache,
                &sin_cache,
                cos_max_pos,
            )
            .expect("layer forward");
            let got = out.to_host(&ctx).expect("out D2H");
            let expected = cut(&format!("{name}_out"));
            let violations = compare(&got, expected, hidden_size, &format!("{case}/{name}"));
            if violations > 0 {
                over_tolerance.push(format!("{case}/{name} (layer {index})"));
            }
        }

        for (name, index, in_label) in [
            ("global_first", gfirst, "global_first_in"),
            ("global_last", glast, "sliding_last_out"),
        ] {
            let x =
                HiddenStates::from_host(&ctx, cut(in_label), hidden_size, seq_len).expect("x H2D");
            let out = global_layer_forward(
                &ctx,
                &weights.layers[index],
                &global_geom,
                &x,
                0,
                &gcos_cache,
                &gsin_cache,
                cos_max_pos,
            )
            .expect("global layer forward");
            let got = out.to_host(&ctx).expect("out D2H");
            let expected = cut(&format!("{name}_out"));
            let violations = compare(&got, expected, hidden_size, &format!("{case}/{name}"));
            if violations > 0 {
                over_tolerance.push(format!("{case}/{name} (layer {index})"));
            }
        }
    }
    assert!(
        over_tolerance.is_empty(),
        "comparisons over tolerance: {over_tolerance:?}"
    );
}
