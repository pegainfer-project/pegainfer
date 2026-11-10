//! Holds one layer forward against the reference recorded by
//! `tools/accuracy/dump_qwen4_layer_golden.py` on the checkpoint's own weights.
//! Needs `PEGAINFER_QWEN4_MODEL_PATH` (the checkpoint, or the shards of that layer
//! plus the index) and `PEGAINFER_QWEN4_LAYER_GOLDEN` (the recorded file).

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;
use safetensors::Dtype;
use safetensors::SafeTensors;

use super::linear_layer_prefill;
use crate::weights::Geometry;
use crate::weights::load_config;
use crate::weights::load_linear_layer;

const METADATA_KEY: &str = "qwen4_layer_golden";
/// A routing difference counts as a tie when the reference's own logits separate
/// the two experts by at most this many bf16 steps.
const TIE_STEPS: f32 = 2.0;

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn floats(golden: &SafeTensors, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let view = golden
        .tensor(name)
        .with_context(|| format!("golden lacks {name}"))?;
    ensure!(
        view.dtype() == Dtype::BF16,
        "{name} is {:?}, expected BF16",
        view.dtype()
    );
    let values = view
        .data()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| bf16::from_bits(u16::from_le_bytes(*pair)).to_f32())
        .collect();
    Ok((view.shape().to_vec(), values))
}

fn ints(golden: &SafeTensors, name: &str) -> Result<Vec<i64>> {
    let view = golden
        .tensor(name)
        .with_context(|| format!("golden lacks {name}"))?;
    ensure!(
        view.dtype() == Dtype::I64,
        "{name} is {:?}, expected I64",
        view.dtype()
    );
    Ok(view
        .data()
        .as_chunks::<8>()
        .0
        .iter()
        .map(|bytes| i64::from_le_bytes(*bytes))
        .collect())
}

struct Diff {
    max_abs: f32,
    mean_abs: f32,
    max_ref: f32,
}

fn diff(ours: &[f32], reference: &[f32]) -> Diff {
    assert_eq!(ours.len(), reference.len());
    let mut max_abs = 0f32;
    let mut sum = 0f64;
    let mut max_ref = 0f32;
    for (a, b) in ours.iter().zip(reference) {
        let d = (a - b).abs();
        assert!(
            d.is_finite(),
            "non-finite difference: ours {a}, reference {b}"
        );
        max_abs = max_abs.max(d);
        sum += f64::from(d);
        max_ref = max_ref.max(b.abs());
    }
    Diff {
        max_abs,
        mean_abs: (sum / ours.len() as f64) as f32,
        max_ref,
    }
}

/// The distance to the next bf16 value above `x`.
fn bf16_step(x: f32) -> f32 {
    let at = bf16::from_f32(x.abs());
    bf16::from_bits(at.to_bits() + 1).to_f32() - at.to_f32()
}

#[test]
fn layer_matches_the_reference() -> Result<()> {
    let (Some(model), Some(golden_path)) = (
        env_path("PEGAINFER_QWEN4_MODEL_PATH"),
        env_path("PEGAINFER_QWEN4_LAYER_GOLDEN"),
    ) else {
        eprintln!("skipped: set PEGAINFER_QWEN4_MODEL_PATH and PEGAINFER_QWEN4_LAYER_GOLDEN");
        return Ok(());
    };
    let bytes = std::fs::read(&golden_path)?;
    let golden = SafeTensors::deserialize(&bytes)?;
    let (_, metadata) = SafeTensors::read_metadata(&bytes)?;
    let manifest: serde_json::Value = serde_json::from_str(
        metadata
            .metadata()
            .as_ref()
            .and_then(|m| m.get(METADATA_KEY))
            .context("golden carries no manifest")?,
    )?;
    let layer = manifest["layer"]
        .as_u64()
        .context("manifest has no layer")? as usize;

    let config = load_config(&model)?;
    let g = Geometry::from_config(&config);
    let ctx = DeviceContext::new()?;
    let weights = load_linear_layer(&ctx, &model, &config, layer)?;

    let (shape, stream_in) = floats(&golden, "stream_in")?;
    ensure!(
        shape.len() == 2 && shape[1] == g.stream_width(),
        "stream_in is {shape:?}"
    );
    let rows = shape[0];
    let host: Vec<bf16> = stream_in.iter().map(|&x| bf16::from_f32(x)).collect();
    let input = HiddenStates::from_host(&ctx, &host, g.stream_width(), rows)?;
    let trace = linear_layer_prefill(&ctx, &g, &weights, &input)?;
    ctx.sync()?;

    let stages: [(&str, &HiddenStates); 18] = [
        ("attn_mixed", &trace.attn_mixed),
        ("attn_inject", &trace.attn_inject),
        ("gdn_qkv", &trace.gdn_qkv),
        ("gdn_z", &trace.gdn_z),
        ("gdn_b", &trace.gdn_b),
        ("gdn_a", &trace.gdn_a),
        ("gdn_core", &trace.gdn_core),
        ("gdn_normed", &trace.gdn_normed),
        ("attn_out", &trace.attn_out),
        ("stream_after_attn", &trace.stream_after_attn),
        ("mlp_mixed", &trace.mlp_mixed),
        ("mlp_inject", &trace.mlp_inject),
        ("router_logits", &trace.moe.router_logits),
        ("shared_gate_logit", &trace.moe.shared_gate_logit),
        ("shared_out", &trace.moe.shared_out),
        ("expert_out", &trace.moe.expert_out),
        ("moe_out", &trace.moe.out),
        ("stream_out", &trace.stream_out),
    ];
    eprintln!("qwen4 layer {layer} gate: {rows} tokens, max |ours - reference| per stage");
    let mut measured = Vec::new();
    for (name, ours) in stages {
        let (_, reference) = floats(&golden, name)?;
        let d = diff(&ours.to_host(&ctx)?, &reference);
        eprintln!(
            "  {name:>18}: max {:.3e}  mean {:.3e}  (reference max {:.3e})",
            d.max_abs, d.mean_abs, d.max_ref
        );
        measured.push((name, d));
    }

    // Picks may differ where the reference's own logits tie; anything wider is a
    // routing fault.
    let (_, logits_ref) = floats(&golden, "router_logits")?;
    let picks = ints(&golden, "selected_experts")?;
    let ours: Vec<i32> = ctx.stream.clone_dtoh(&trace.moe.selected_experts)?;
    ctx.sync()?;
    let mut differing = 0usize;
    let mut widest = 0f32;
    for token in 0..rows {
        let pairs = token * g.top_k..(token + 1) * g.top_k;
        let reference: BTreeSet<i64> = picks[pairs.clone()].iter().copied().collect();
        let mine: BTreeSet<i64> = ours[pairs].iter().map(|&e| i64::from(e)).collect();
        if reference == mine {
            continue;
        }
        differing += 1;
        let logits = &logits_ref[token * g.experts..(token + 1) * g.experts];
        let kept = reference
            .iter()
            .map(|&e| logits[e as usize])
            .fold(f32::INFINITY, f32::min);
        let swapped_in = mine
            .difference(&reference)
            .map(|&e| logits[e as usize])
            .fold(f32::NEG_INFINITY, f32::max);
        let steps = (kept - swapped_in) / bf16_step(kept.abs().max(swapped_in.abs()));
        widest = widest.max(steps);
        eprintln!(
            "  token {token}: picks differ, the reference's logits are {steps:.2} bf16 steps apart"
        );
    }
    let (_, weights_ref) = floats(&golden, "routing_weights")?;
    let weights_ours: Vec<f32> = ctx
        .stream
        .clone_dtoh(&trace.moe.routing_weights)?
        .iter()
        .map(|w: &bf16| w.to_f32())
        .collect();
    ctx.sync()?;
    let weight_diff = diff(&weights_ours, &weights_ref);
    eprintln!(
        "  routing: {differing}/{rows} tokens pick a different expert set, weights max {:.3e}",
        weight_diff.max_abs
    );

    // The recorded reference ran once on the accelerator and once on the CPU; that
    // spread is the floor a bf16 port cannot be held below.
    for name in ["attn_out", "stream_after_attn", "moe_out", "stream_out"] {
        let (_, accel) = floats(&golden, name)?;
        let (_, cpu) = floats(&golden, &format!("cpu_{name}"))?;
        let floor = diff(&accel, &cpu).max_abs;
        let ours = &measured
            .iter()
            .find(|(stage, _)| *stage == name)
            .expect("measured")
            .1;
        eprintln!("  {name:>18}: reference accelerator vs cpu floor {floor:.3e}");
        ensure!(
            ours.max_abs <= 4.0 * floor.max(1e-3),
            "{name}: max |ours - reference| {:.3e} exceeds four times the reference's own floor {floor:.3e}",
            ours.max_abs
        );
    }
    ensure!(
        widest <= TIE_STEPS,
        "a routing difference is not a tie: the reference's logits are {widest:.2} bf16 steps apart"
    );
    Ok(())
}
