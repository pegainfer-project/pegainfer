//! Tiny-checkpoint projection, collection and startup-boundary checks.

#![allow(
    clippy::float_cmp,
    reason = "Exact comparisons pin BF16 outputs and representable numerical fixtures"
)]

use std::collections::BTreeMap;
use std::fs;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::sys::CUgraphInstantiate_flags;
use cudarc::driver::sys::CUstreamCaptureMode;
use half::bf16;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use safetensors::SafeTensors;
use serde_json::Value;
use serde_json::json;

use super::DFlash2Sampler;
use crate::dflash::config::NativeDFlash2Config;

fn bf16_values(data: &[u8]) -> Vec<bf16> {
    data.as_chunks::<2>()
        .0
        .iter()
        .map(|&v| bf16::from_bits(u16::from_le_bytes(v)))
        .collect()
}

fn tiny_checkpoint() -> Result<tempfile::TempDir> {
    let dir = tempfile::tempdir()?;
    let mut value: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/dflash2/native_config.json"
    ))?;
    for (name, size) in [
        ("hidden_size", 8),
        ("intermediate_size", 12),
        ("num_hidden_layers", 1),
        ("num_attention_heads", 2),
        ("num_key_value_heads", 1),
        ("head_dim", 4),
        ("vocab_size", 32),
        ("num_target_layers", 3),
        ("max_window_layers", 1),
    ] {
        value[name] = json!(size);
    }
    value["layer_types"] = json!(["sliding_attention"]);
    value["dflash_config"] = json!({
        "block_size": 3,
        "conv_group_size": 4,
        "conv_kernel_size": 2,
        "mask_token_id": 31,
        "selector_rank": 5,
        "selector_top_k": 16,
        "target_layer_ids": [0, 2]
    });
    fs::write(dir.path().join("config.json"), serde_json::to_vec(&value)?)?;

    let config = NativeDFlash2Config::from_json(&value)?;
    let mut header = BTreeMap::new();
    let mut payload = Vec::new();

    // Unused backbone weights only satisfy the complete loader contract. The
    // numerical assertions below concern the three nonzero selector tensors.
    for (name, shape) in config.expected_tensors()? {
        let start = payload.len();
        for index in 0..shape.iter().product::<usize>() {
            let number = if name.contains("hidden_projection") {
                ((index % 7) as f32 - 3.0) / 8.0
            } else if name.ends_with("predecessor_codebook") {
                ((index % 11) as f32 - 5.0) / 4.0
            } else if name.ends_with("successor_codebook") {
                ((index % 13) as f32 - 6.0) / 8.0
            } else {
                0.0
            };
            payload.extend(bf16::from_f32(number).to_bits().to_le_bytes());
        }
        header.insert(
            name,
            json!({"dtype": "BF16", "shape": shape, "data_offsets": [start, payload.len()]}),
        );
    }

    let mut header = serde_json::to_vec(&header)?;
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(payload);
    fs::write(dir.path().join("model.safetensors"), file)?;
    Ok(dir)
}

#[test]
fn native_startup_rejects_before_target_or_device_loading() -> Result<()> {
    let checkpoint = tiny_checkpoint()?;

    // Both target path and device are unusable. The native capability error
    // must win, proving the actual executor entry rejects before touching them.
    let result = crate::executor::Qwen3Executor::from_runtime_with_lora_options(
        "/missing-target-for-native-preflight",
        true,
        &[usize::MAX],
        crate::Qwen3LoraOptions::default(),
        crate::Qwen3OffloadOptions::disabled(),
        crate::DEFAULT_MAX_PREFILL_TOKENS,
        Some(checkpoint.path().to_str().unwrap()),
        crate::Qwen3MemoryOptions::default(),
    );
    let Err(error) = result else {
        anyhow::bail!("native serving unexpectedly started")
    };
    ensure!(
        format!("{error:#}").contains("native DFlash2 backbone execution is not supported"),
        "unexpected startup error: {error:#}"
    );

    Ok(())
}

#[test]
fn selector_checkpoint_projection_and_collection() -> Result<()> {
    let checkpoint = tiny_checkpoint()?;
    let ctx = DeviceContext::new()?;
    let mut sampler =
        DFlash2Sampler::from_safetensors(&ctx, checkpoint.path().to_str().unwrap(), 2, 1)?;

    let hidden_cpu: Vec<_> = (0..6 * 8)
        .map(|i| bf16::from_f32((i % 9) as f32 / 4.0 - 1.0))
        .collect();
    let logits_cpu: Vec<_> = (0..6 * 32)
        .map(|i| bf16::from_f32(((i * 7) % 29) as f32 / 4.0))
        .collect();
    let mut hidden = HiddenStates::from_host(&ctx, &hidden_cpu, 8, 6)?;
    let mut logits = HiddenStates::from_host(&ctx, &logits_cpu, 32, 6)?;
    let anchors = ctx.stream.clone_htod(&[31u32, 3])?;

    let input = fs::read(checkpoint.path().join("model.safetensors"))?;
    let weights = SafeTensors::deserialize(&input)?;
    let w = bf16_values(
        weights
            .tensor("candidate_selector.hidden_projection.weight")?
            .data(),
    );
    let a = bf16_values(
        weights
            .tensor("candidate_selector.predecessor_codebook")?
            .data(),
    );
    let b = bf16_values(
        weights
            .tensor("candidate_selector.successor_codebook")?
            .data(),
    );

    let projection: Vec<_> = (0..6 * 5)
        .map(|i| {
            bf16::from_f32(
                (0..8)
                    .map(|d| hidden_cpu[(i / 5) * 8 + d].to_f32() * w[(i % 5) * 8 + d].to_f32())
                    .sum(),
            )
            .to_f32()
        })
        .collect();

    let mut expected = Vec::new();
    for (request, anchor) in [31usize, 3].into_iter().enumerate() {
        let mut predecessor = anchor;
        for slot in 1..3 {
            let row = request * 3 + slot;
            let mut ids: Vec<_> = (0..32).collect();
            ids.sort_by(|&left, &right| {
                logits_cpu[row * 32 + right]
                    .to_f32()
                    .total_cmp(&logits_cpu[row * 32 + left].to_f32())
                    .then(left.cmp(&right))
            });
            ids.truncate(16);
            let score = |id: usize| {
                logits_cpu[row * 32 + id].to_f32()
                    + (0..5)
                        .map(|r| {
                            a[predecessor * 5 + r].to_f32()
                                * projection[row * 5 + r]
                                * b[id * 5 + r].to_f32()
                        })
                        .sum::<f32>()
            };
            ids.sort_by(|&left, &right| {
                score(right).total_cmp(&score(left)).then(left.cmp(&right))
            });
            predecessor = ids[0];
            expected.push(predecessor as u32);
        }
    }

    // This nonzero asymmetric case differs from both swapped-codebook and
    // gate-disabled paths, so shape-correct semantic regressions are observable.
    assert_eq!(expected, [30, 31, 25, 18]);

    for requests in [2, 1, 2] {
        hidden.seq_len = requests * 3;
        logits.seq_len = requests * 3;
        let pending = sampler.enqueue(&hidden, &logits, &anchors, requests)?;
        let actual_projection = ctx.stream.clone_dtoh(&pending.projected_hidden().data)?;
        for (index, actual) in actual_projection.iter().take(requests * 3 * 5).enumerate() {
            assert_eq!(actual.to_f32(), projection[index], "projection {index}");
        }
        assert_eq!(pending.collect()?, expected[..requests * 2]);
    }

    // Keep all allocations alive while a captured production enqueue replays.
    // Changing the hidden buffer to zero makes the expected path unary-only;
    // this catches graphs that accidentally retain an old projection/output.
    ctx.stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
    let pending = sampler.enqueue(&hidden, &logits, &anchors, 2)?;
    let graph = ctx
        .stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)?
        .context("empty selector graph")?;
    graph.launch()?;
    assert_eq!(ctx.stream.clone_dtoh(pending.selected_ids())?, expected);
    ctx.stream
        .memcpy_htod(&vec![bf16::ZERO; hidden_cpu.len()], &mut hidden.data)?;
    graph.launch()?;
    let unary_path: Vec<_> = [1, 2, 4, 5]
        .map(|row| {
            (0..32)
                .max_by(|&left, &right| {
                    logits_cpu[row * 32 + left]
                        .to_f32()
                        .total_cmp(&logits_cpu[row * 32 + right].to_f32())
                        .then(right.cmp(&left))
                })
                .unwrap() as u32
        })
        .into();
    assert_eq!(pending.collect()?, unary_path);
    drop(graph);

    ctx.stream.memcpy_htod(&hidden_cpu, &mut hidden.data)?;
    let invalid = ctx.stream.clone_htod(&[32u32, 3])?;
    assert!(
        sampler
            .enqueue(&hidden, &logits, &invalid, 2)?
            .collect()
            .is_err()
    );
    assert_eq!(
        sampler.enqueue(&hidden, &logits, &anchors, 2)?.collect()?,
        expected
    );

    hidden.seq_len = 0;
    logits.seq_len = 0;
    assert!(
        sampler
            .enqueue(&hidden, &logits, &anchors, 0)?
            .collect()?
            .is_empty()
    );

    Ok(())
}
