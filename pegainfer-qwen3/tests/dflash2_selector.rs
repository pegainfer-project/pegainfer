//! Public native-selector acceptance against pinned real forward evidence.
//! This gate does not execute the native backbone or enable serving.

#![allow(
    clippy::float_cmp,
    reason = "Exact comparisons pin BF16 outputs and representable numerical fixtures"
)]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::io::Read;
use std::path::Component;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::sys::CUgraphInstantiate_flags;
use cudarc::driver::sys::CUstreamCaptureMode;
use half::bf16;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_core::weight_loader::load_shard_info;
use pegainfer_qwen3::DFlash2Sampler;
use pegainfer_qwen3::NativeArtifact;
use pegainfer_qwen3::NativeTargetMetadata;
use safetensors::Dtype;
use safetensors::SafeTensors;
use serde_json::Value;
use sha2::Digest;
use sha2::Sha256;

fn bf16_values(data: &[u8]) -> Vec<bf16> {
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|&v| bf16::from_bits(u16::from_le_bytes(v)))
        .collect()
}

fn verify_hash(path: &Path, expected: &Value) -> Result<()> {
    let mut file = fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0u8; 1024 * 1024];
    loop {
        let size = file.read(&mut buffer)?;
        if size == 0 {
            break;
        }
        digest.update(&buffer[..size]);
    }

    let mut actual = String::with_capacity(64);
    for byte in digest.finalize() {
        write!(actual, "{byte:02x}")?;
    }
    ensure!(
        actual == expected.as_str().context("missing SHA256")?,
        "hash mismatch: {}",
        path.display()
    );

    Ok(())
}

fn relative_shard_path(path: &Path) -> Result<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            Component::CurDir => {}
            _ => anyhow::bail!("unsafe fixture shard path: {}", path.display()),
        }
    }
    ensure!(
        !normalized.as_os_str().is_empty(),
        "empty fixture shard path"
    );
    Ok(normalized)
}

fn verify_checkpoint_hashes(checkpoint: &Path, hashes: &Value) -> Result<()> {
    let hashes = hashes.as_object().context("checkpoint hashes")?;
    let mut normalized = BTreeMap::new();
    for (file, digest) in hashes {
        let path = relative_shard_path(Path::new(file))?;
        ensure!(
            normalized.insert(path, digest).is_none(),
            "duplicate fixture shard path: {file}"
        );
    }
    let (shards, _) = load_shard_info(checkpoint.to_str().context("checkpoint path UTF-8")?)?;
    ensure!(
        !normalized.is_empty() && normalized.len() == shards.len(),
        "complete checkpoint hash coverage is required"
    );
    for shard in shards {
        let relative = relative_shard_path(Path::new(&shard).strip_prefix(checkpoint)?)?;
        let digest = normalized
            .get(&relative)
            .with_context(|| format!("missing hash for shard {}", relative.display()))?;
        verify_hash(&checkpoint.join(relative), digest)?;
    }

    Ok(())
}

fn flatten(value: &Value, out: &mut Vec<f32>) {
    if let Some(array) = value.as_array() {
        for item in array {
            flatten(item, out);
        }
    } else {
        out.push(value.as_f64().expect("numeric golden") as f32);
    }
}

fn golden(value: &Value) -> Vec<f32> {
    let mut output = Vec::new();
    flatten(value, &mut output);
    output
}

fn golden_ids(value: &Value) -> Result<Vec<u32>> {
    if let Some(values) = value.as_array() {
        let mut ids = Vec::new();
        for value in values {
            ids.extend(golden_ids(value)?);
        }
        Ok(ids)
    } else {
        Ok(vec![u32::try_from(
            value.as_u64().context("nonnegative golden token ID")?,
        )?])
    }
}

/// Explicit acceptance gate: invoking this test without real evidence FAILS.
#[test]
#[ignore = "requires PEGAINFER_DFLASH2_CHECKPOINT and PEGAINFER_DFLASH2_FIXTURE from real pinned forwards"]
fn selector_real_checkpoint_golden() -> Result<()> {
    let checkpoint = std::env::var("PEGAINFER_DFLASH2_CHECKPOINT")
        .context("real native checkpoint is required")?;
    let fixture =
        std::env::var("PEGAINFER_DFLASH2_FIXTURE").context("real captured fixture is required")?;
    let checkpoint = Path::new(&checkpoint);
    let fixture = Path::new(&fixture);
    let manifest: Value = serde_json::from_slice(&fs::read(fixture.join("manifest.json"))?)?;
    ensure!(
        manifest["format_version"] == 1,
        "unsupported fixture version"
    );
    ensure!(
        manifest["provenance"]["input_origin"] == "pinned_target_and_native_draft_forward",
        "real forward provenance is required"
    );
    for field in [
        "target_model",
        "target_revision",
        "draft_model",
        "draft_revision",
        "reference_revision",
        "reference_source_sha256",
        "capture_source_sha256",
        "draft_checkpoint_sha256",
    ] {
        ensure!(
            !manifest["provenance"][field]
                .as_str()
                .context(field)?
                .is_empty(),
            "missing provenance {field}"
        );
    }

    verify_hash(&checkpoint.join("config.json"), &manifest["config_sha256"])?;
    verify_checkpoint_hashes(checkpoint, &manifest["checkpoint_files"])?;
    ensure!(
        manifest["checkpoint_files"]
            .as_object()
            .context("checkpoint hashes")?
            .values()
            .any(|digest| digest == &manifest["provenance"]["draft_checkpoint_sha256"]),
        "capture and selector checkpoint hashes differ"
    );
    let index = checkpoint.join("model.safetensors.index.json");
    if index.exists() {
        verify_hash(&index, &manifest["checkpoint_index_sha256"])?;
    }
    verify_hash(
        &fixture.join("inputs.safetensors"),
        &manifest["inputs_sha256"],
    )?;
    verify_hash(
        &fixture.join("reference.json"),
        &manifest["reference_sha256"],
    )?;
    verify_hash(
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../scripts/generate_dflash2_selector_fixture.py"
        )),
        &manifest["generator_sha256"],
    )?;

    let reference: Value = serde_json::from_slice(&fs::read(fixture.join("reference.json"))?)?;
    ensure!(
        reference["format_version"] == 1
            && reference["candidate_k"] == 16
            && reference["projected_hidden_includes_anchor"] == true,
        "golden layout contract"
    );
    for (field, expected) in [
        ("projection_output_dtype", "BF16"),
        ("edge_accumulation_dtype", "F32"),
        ("candidate_ties", "logit_desc_token_id_asc"),
        ("path_ties", "score_desc_token_id_asc"),
    ] {
        ensure!(
            reference[field] == expected,
            "golden numerical contract {field}"
        );
    }

    let bytes = fs::read(fixture.join("inputs.safetensors"))?;
    let inputs = SafeTensors::deserialize(&bytes)?;
    let h = inputs.tensor("hidden")?;
    let u = inputs.tensor("logits")?;
    let anchor = inputs.tensor("anchors")?;
    ensure!(
        h.dtype() == Dtype::BF16 && u.dtype() == Dtype::BF16 && anchor.dtype() == Dtype::I64,
        "fixture dtypes"
    );
    ensure!(
        h.shape().len() == 3 && u.shape().len() == 3 && anchor.shape().len() == 1,
        "fixture ranks"
    );
    let (n, block, width) = (h.shape()[0], h.shape()[1], h.shape()[2]);
    ensure!(
        n > 1 && u.shape()[..2] == [n, block] && anchor.shape() == [n],
        "fixture must include multiple actual requests"
    );

    let artifact = NativeArtifact::inspect(checkpoint.to_str().unwrap())?;
    let target: NativeTargetMetadata =
        serde_json::from_value(manifest["provenance"]["target_metadata"].clone())
            .context("real target metadata")?;
    ensure!(
        manifest["provenance"]["target_model"] == target.model_id
            && manifest["provenance"]["target_revision"] == target.revision,
        "target provenance mismatch"
    );
    artifact.config().validate_target(&target)?;

    let ctx = DeviceContext::new()?;
    let mut sampler = DFlash2Sampler::from_safetensors(&ctx, checkpoint.to_str().unwrap(), n, 2)?;
    ensure!(
        block == sampler.config().block_size()
            && width == sampler.config().hidden_size()
            && u.shape()[2] == sampler.config().vocab_size(),
        "fixture geometry"
    );

    let hidden_cpu = bf16_values(h.data());
    let logits_cpu = bf16_values(u.data());
    let vocab = u.shape()[2];
    let h = HiddenStates::from_host(&ctx, &hidden_cpu, width, n * block)?;
    let u = HiddenStates::from_host(&ctx, &logits_cpu, vocab, n * block)?;
    let anchors = anchor
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|&v| u32::try_from(i64::from_le_bytes(v)))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let anchor_cpu = anchors;
    let anchors = ctx.stream.clone_htod(&anchor_cpu)?;

    let start = std::time::Instant::now();
    let pending = sampler.enqueue(&h, &u, &anchors, n)?;
    let projected = ctx.stream.clone_dtoh(&pending.projected_hidden().data)?;
    let candidates = ctx.stream.clone_dtoh(pending.candidate_ids())?;
    let unary = ctx.stream.clone_dtoh(pending.unary_scores())?;
    let edges = ctx.stream.clone_dtoh(pending.edge_scores())?;
    let paths = pending.collect()?;

    let expected_projection = golden(&reference["projected_hidden"]);
    ensure!(
        projected.len() == expected_projection.len(),
        "projection golden extent"
    );
    let mut max_projection_steps = 0;
    let mut projection_difference_count = 0;
    for (index, (&actual, &expected)) in projected.iter().zip(&expected_projection).enumerate() {
        // One BF16 ULP permits a different FP32 reduction before narrowing.
        let ordered = |value: bf16| {
            let bits = value.to_bits();
            if bits & 0x8000 != 0 {
                !bits
            } else {
                bits ^ 0x8000
            }
        };
        let steps = ordered(actual).abs_diff(ordered(bf16::from_f32(expected)));
        max_projection_steps = max_projection_steps.max(steps);
        projection_difference_count += usize::from(actual.to_f32() != expected);
        ensure!(
            actual.is_finite()
                && expected.is_finite()
                && (actual.to_f32() == expected || steps <= 1),
            "projection mismatch {index}: {actual:?} vs {expected}, {steps} BF16 steps"
        );
    }

    assert_eq!(candidates, golden_ids(&reference["candidate_ids"])?);
    assert_eq!(unary, golden(&reference["candidate_values"]));

    let expected_edges = golden(&reference["edge_scores"]);
    ensure!(edges.len() == expected_edges.len(), "edge golden extent");
    let mut max_error = 0f32;
    for (index, (&actual, &expected)) in edges.iter().zip(&expected_edges).enumerate() {
        max_error = max_error.max((actual - expected).abs());
        ensure!(
            actual.is_finite()
                && expected.is_finite()
                && (actual - expected).abs() <= 1e-4 + 1e-4 * expected.abs(),
            "edge mismatch {index}: {actual} vs {expected}"
        );
    }

    assert_eq!(paths, golden_ids(&reference["path_ids"])?);
    eprintln!(
        "real DFlash2 component: {n} requests, projection differences={projection_difference_count}/{}, max BF16 steps={max_projection_steps}, max edge error={max_error}, fixture margin={}, eager diagnostic pass={:?}, device bytes={}, pinned bytes={}",
        projected.len(),
        reference["minimum_selected_edge_margin"],
        start.elapsed(),
        sampler.device_bytes(),
        sampler.pinned_host_bytes()
    );
    drop(sampler);

    // Reuse real captured requests to cover capacity changes and measure the
    // component at serving-like vocabulary sizes. This is not a serving speedup.
    let mut sampler = DFlash2Sampler::from_safetensors(&ctx, checkpoint.to_str().unwrap(), 16, 2)?;
    let mut repeated_h = Vec::new();
    let mut repeated_u = Vec::new();
    let mut repeated_anchors = Vec::new();
    let mut repeated_path = Vec::new();
    for request in 0..16 {
        let origin = request % n;
        repeated_h
            .extend_from_slice(&hidden_cpu[origin * block * width..(origin + 1) * block * width]);
        repeated_u
            .extend_from_slice(&logits_cpu[origin * block * vocab..(origin + 1) * block * vocab]);
        repeated_anchors.push(anchor_cpu[origin]);
        repeated_path.extend_from_slice(&paths[origin * (block - 1)..(origin + 1) * (block - 1)]);
    }

    let mut h = HiddenStates::from_host(&ctx, &repeated_h, width, 16 * block)?;
    let mut u = HiddenStates::from_host(&ctx, &repeated_u, vocab, 16 * block)?;
    let anchors = ctx.stream.clone_htod(&repeated_anchors)?;

    for requests in [16, 1, 8, 16] {
        h.seq_len = requests * block;
        u.seq_len = requests * block;
        let expected = &repeated_path[..requests * (block - 1)];
        assert_eq!(
            sampler.enqueue(&h, &u, &anchors, requests)?.collect()?,
            expected
        );

        ctx.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let pending = sampler.enqueue(&h, &u, &anchors, requests)?;
        let graph = ctx
            .stream
            .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?
            .context("empty real selector graph")?;
        graph.launch()?;
        ctx.stream.synchronize()?;
        let start = std::time::Instant::now();
        for _ in 0..20 {
            graph.launch()?;
        }
        ctx.stream.synchronize()?;
        let replay_us = start.elapsed().as_secs_f64() * 1e6 / 20.0;
        assert_eq!(pending.collect()?, expected);
        drop(graph);
        eprintln!(
            "real selector graph N={requests} B={block} V={vocab} chunk=2: {replay_us:.1} us/replay, sampler={} bytes, caller inputs={} bytes",
            sampler.device_bytes(),
            (repeated_h.len() + repeated_u.len()) * 2 + 16 * 4
        );
    }

    Ok(())
}
