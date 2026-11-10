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

use super::AttentionTrace;
use super::MixerTrace;
use super::Rope;
use super::layer_prefill;
use crate::weights::Geometry;
use crate::weights::load_config;
use crate::weights::load_layer;

const METADATA_KEY: &str = "qwen4_layer_golden";

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

/// One stage as the engine computed it and as the reference recorded it.
fn stage<'a>(
    ctx: &DeviceContext,
    golden: &SafeTensors,
    name: &'a str,
    ours: &HiddenStates,
) -> Result<(&'a str, Vec<f32>, Vec<f32>)> {
    Ok((name, ours.to_host(ctx)?, floats(golden, name)?.1))
}

/// The dims the partial RoPE leaves alone, out of `[rows * heads, head_dim]`.
fn unrotated(values: &[f32], g: &Geometry) -> Vec<f32> {
    values
        .chunks_exact(g.head_dim)
        .flat_map(|head| head[g.rotary_dim..].iter().copied())
        .collect()
}

/// K as the prep kernel wrote it into the scratch pool, back in `[rows * kv_heads, head_dim]`.
fn pool_k(ctx: &DeviceContext, t: &AttentionTrace, g: &Geometry, rows: usize) -> Result<Vec<f32>> {
    let pool: Vec<bf16> = ctx.stream.clone_dtoh(&t.kv)?;
    ctx.sync()?;
    let width = g.kv_width();
    let mut k = Vec::with_capacity(rows * width);
    for pos in 0..rows {
        let row =
            (pos / t.layout.page_size) * t.layout.page_stride + (pos % t.layout.page_size) * width;
        k.extend(pool[row..row + width].iter().map(|x| x.to_f32()));
    }
    Ok(k)
}

/// The reference's indexer mask is additive, zero where a key is kept. Dense
/// attention stands in for it only while every causal key is kept.
fn ensure_dense(golden: &SafeTensors, rows: usize) -> Result<()> {
    let (shape, mask) = floats(golden, "attn_indexer_mask")?;
    ensure!(shape == [rows, rows], "attn_indexer_mask is {shape:?}");
    let dropped = (0..rows)
        .flat_map(|i| (0..=i).map(move |j| i * rows + j))
        .filter(|&at| mask[at] != 0.0)
        .count();
    ensure!(
        dropped == 0,
        "the reference's indexer dropped {dropped} causal keys, so dense attention is not the reference for this prompt"
    );
    Ok(())
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
    let weights = load_layer(&ctx, &model, &config, layer)?;

    let (shape, stream_in) = floats(&golden, "stream_in")?;
    ensure!(
        shape.len() == 2 && shape[1] == g.stream_width(),
        "stream_in is {shape:?}"
    );
    let rows = shape[0];
    let host: Vec<bf16> = stream_in.iter().map(|&x| bf16::from_f32(x)).collect();
    let input = HiddenStates::from_host(&ctx, &host, g.stream_width(), rows)?;
    let rope = Rope::new(&ctx, &g, rows)?;
    let trace = layer_prefill(&ctx, &g, &weights, &rope, &input)?;
    ctx.sync()?;

    let mut stages = vec![
        stage(&ctx, &golden, "attn_mixed", &trace.attn_mixed)?,
        stage(&ctx, &golden, "attn_inject", &trace.attn_inject)?,
    ];
    match &trace.mixer {
        MixerTrace::Linear(t) => stages.extend([
            stage(&ctx, &golden, "gdn_qkv", &t.qkv)?,
            stage(&ctx, &golden, "gdn_z", &t.z)?,
            stage(&ctx, &golden, "gdn_b", &t.b)?,
            stage(&ctx, &golden, "gdn_a", &t.a)?,
            stage(&ctx, &golden, "gdn_core", &t.core)?,
            stage(&ctx, &golden, "gdn_normed", &t.normed)?,
        ]),
        MixerTrace::Full(t) => {
            ensure_dense(&golden, rows)?;
            stages.extend([
                stage(&ctx, &golden, "attn_q_proj", &t.q_proj)?,
                stage(&ctx, &golden, "attn_k_proj", &t.k_proj)?,
                stage(&ctx, &golden, "attn_v_proj", &t.v_proj)?,
                (
                    "q_norm_unrotated",
                    unrotated(&t.q_prepped.to_host(&ctx)?, &g),
                    unrotated(&floats(&golden, "attn_q_normed")?.1, &g),
                ),
                (
                    "k_norm_unrotated",
                    unrotated(&pool_k(&ctx, t, &g, rows)?, &g),
                    unrotated(&floats(&golden, "attn_k_normed")?.1, &g),
                ),
                stage(&ctx, &golden, "attn_gated", &t.gated)?,
            ]);
        }
    }
    stages.extend([
        stage(&ctx, &golden, "attn_out", &trace.attn_out)?,
        stage(&ctx, &golden, "stream_after_attn", &trace.stream_after_attn)?,
        stage(&ctx, &golden, "mlp_mixed", &trace.mlp_mixed)?,
        stage(&ctx, &golden, "mlp_inject", &trace.mlp_inject)?,
        stage(&ctx, &golden, "router_logits", &trace.moe.router_logits)?,
        stage(
            &ctx,
            &golden,
            "shared_gate_logit",
            &trace.moe.shared_gate_logit,
        )?,
        stage(&ctx, &golden, "shared_out", &trace.moe.shared_out)?,
        stage(&ctx, &golden, "expert_out", &trace.moe.expert_out)?,
        stage(&ctx, &golden, "moe_out", &trace.moe.out)?,
        stage(&ctx, &golden, "stream_out", &trace.stream_out)?,
    ]);
    eprintln!("qwen4 layer {layer} gate: {rows} tokens, max |ours - reference| per stage");
    let mut measured = Vec::new();
    for (name, ours, reference) in &stages {
        let d = diff(ours, reference);
        eprintln!(
            "  {name:>18}: max {:.3e}  mean {:.3e}  (reference max {:.3e})",
            d.max_abs, d.mean_abs, d.max_ref
        );
        measured.push((*name, d));
    }

    // The picks must be a top-k of this engine's own logits, and the logits are
    // held to the reference's floor below, so a pick that differs from the
    // reference's can only sit where the reference's own gap is inside that
    // floor. The gaps are printed for reading.
    let (_, logits_ref) = floats(&golden, "router_logits")?;
    let picks_ref = ints(&golden, "selected_experts")?;
    let logits_ours = trace.moe.router_logits.to_host(&ctx)?;
    let picks_ours: Vec<i32> = ctx.stream.clone_dtoh(&trace.moe.selected_experts)?;
    ctx.sync()?;
    let mut differing = 0usize;
    for token in 0..rows {
        let pairs = token * g.top_k..(token + 1) * g.top_k;
        let experts = token * g.experts..(token + 1) * g.experts;
        let mine: BTreeSet<usize> = picks_ours[pairs.clone()]
            .iter()
            .map(|&e| e as usize)
            .collect();
        ensure!(
            mine.len() == g.top_k,
            "token {token}: the picks repeat an expert"
        );
        let logits = &logits_ours[experts.clone()];
        let lowest_kept = mine
            .iter()
            .map(|&e| logits[e])
            .fold(f32::INFINITY, f32::min);
        let highest_dropped = (0..g.experts)
            .filter(|e| !mine.contains(e))
            .map(|e| logits[e])
            .fold(f32::NEG_INFINITY, f32::max);
        ensure!(
            lowest_kept >= highest_dropped,
            "token {token}: the picks are not a top {} of the engine's own logits (kept {lowest_kept}, dropped {highest_dropped})",
            g.top_k
        );
        let reference: BTreeSet<usize> = picks_ref[pairs].iter().map(|&e| e as usize).collect();
        if reference == mine {
            continue;
        }
        differing += 1;
        let logits = &logits_ref[experts];
        let kept = reference
            .iter()
            .map(|&e| logits[e])
            .fold(f32::INFINITY, f32::min);
        let swapped_in = mine
            .difference(&reference)
            .map(|&e| logits[e])
            .fold(f32::NEG_INFINITY, f32::max);
        let steps = (kept - swapped_in) / bf16_step(kept.abs().max(swapped_in.abs()));
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
    for name in [
        "attn_out",
        "stream_after_attn",
        "router_logits",
        "moe_out",
        "stream_out",
    ] {
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
    Ok(())
}
