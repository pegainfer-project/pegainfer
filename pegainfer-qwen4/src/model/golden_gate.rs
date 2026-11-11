//! Holds a chain of layers with the bookends around them against the reference
//! recorded by `tools/accuracy/dump_qwen4_layer_golden.py` on the checkpoint's
//! own weights: the prompt in prefill and, when the golden carries them, decode
//! steps on the carried state. Needs `PEGAINFER_QWEN4_MODEL_PATH` (the
//! checkpoint, or the shards of those layers and the bookends plus the index)
//! and `PEGAINFER_QWEN4_LAYER_GOLDEN`.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use half::bf16;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;
use safetensors::Dtype;
use safetensors::SafeTensors;

use super::Forward;
use super::Model;
use super::ModelState;
use crate::layer::KvScratch;
use crate::layer::LayerTrace;
use crate::layer::MixerState;
use crate::layer::MixerTrace;
use crate::weights::Geometry;
use crate::weights::load_config;

const METADATA_KEY: &str = "qwen4_layer_golden";

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn floats(golden: &SafeTensors, name: &str) -> Result<(Vec<usize>, Vec<f32>)> {
    let view = golden
        .tensor(name)
        .with_context(|| format!("golden lacks {name}"))?;
    let values = match view.dtype() {
        Dtype::BF16 => view
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| bf16::from_bits(u16::from_le_bytes(*pair)).to_f32())
            .collect(),
        Dtype::F32 => view
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect(),
        other => bail!("{name} is {other:?}, expected BF16 or F32"),
    };
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

fn print_diff(name: &str, d: &Diff) {
    eprintln!(
        "  {name:>30}: max {:.3e}  mean {:.3e}  (reference max {:.3e})",
        d.max_abs, d.mean_abs, d.max_ref
    );
}

/// The stages the reference also recorded on the CPU are held to four times its
/// own accelerator versus CPU spread, the floor a bf16 port cannot sit below.
fn hold_floor(golden: &SafeTensors, name: &str, ours: &Diff) -> Result<()> {
    let (_, accel) = floats(golden, name)?;
    let (_, cpu) = floats(golden, &format!("cpu_{name}"))?;
    let floor = diff(&accel, &cpu).max_abs;
    eprintln!("  {name:>30}: reference accelerator vs cpu floor {floor:.3e}");
    ensure!(
        ours.max_abs <= 4.0 * floor.max(1e-3),
        "{name}: max |ours - reference| {:.3e} exceeds four times the reference's own floor {floor:.3e}",
        ours.max_abs
    );
    Ok(())
}

/// One stage as the engine computed it and as the reference recorded it.
fn stage(
    ctx: &DeviceContext,
    golden: &SafeTensors,
    name: String,
    ours: &HiddenStates,
) -> Result<(String, Vec<f32>, Vec<f32>)> {
    let reference = floats(golden, &name)?.1;
    Ok((name, ours.to_host(ctx)?, reference))
}

/// The dims the partial RoPE leaves alone, out of `[rows * heads, head_dim]`.
fn unrotated(values: &[f32], g: &Geometry) -> Vec<f32> {
    values
        .chunks_exact(g.head_dim)
        .flat_map(|head| head[g.rotary_dim..].iter().copied())
        .collect()
}

/// K of the last `rows` positions as the prep kernel wrote it into the scratch
/// pool, back in `[rows * kv_heads, head_dim]`.
fn pool_k(ctx: &DeviceContext, kv: &KvScratch, g: &Geometry, rows: usize) -> Result<Vec<f32>> {
    let pool: Vec<bf16> = ctx.stream.clone_dtoh(&kv.pool)?;
    ctx.sync()?;
    let width = g.kv_width();
    let mut k = Vec::with_capacity(rows * width);
    for pos in kv.len - rows..kv.len {
        let row = (pos / kv.layout.page_size) * kv.layout.page_stride
            + (pos % kv.layout.page_size) * width;
        k.extend(pool[row..row + width].iter().map(|x| x.to_f32()));
    }
    Ok(k)
}

/// The reference's indexer mask is additive, zero where a key is kept, one row
/// per query over every key so far. Dense attention stands in for it only while
/// every causal key is kept.
fn ensure_dense(golden: &SafeTensors, name: &str, rows: usize, kv_len: usize) -> Result<()> {
    let (shape, mask) = floats(golden, name)?;
    ensure!(shape == [rows, kv_len], "{name} is {shape:?}");
    let first = kv_len - rows;
    let dropped = (0..rows)
        .flat_map(|i| (0..=first + i).map(move |j| i * kv_len + j))
        .filter(|&at| mask[at] != 0.0)
        .count();
    ensure!(
        dropped == 0,
        "the reference's indexer dropped {dropped} causal keys, so dense attention is not the reference for this prompt"
    );
    Ok(())
}

/// One layer's forward against the reference's stages under `prefix`: every
/// stage is printed, the picks must be a top-k of the engine's own logits, and
/// the floored stages are held.
fn hold_layer(
    ctx: &DeviceContext,
    golden: &SafeTensors,
    g: &Geometry,
    prefix: &str,
    trace: &LayerTrace,
    state: &MixerState,
) -> Result<()> {
    let name = |stage: &str| format!("{prefix}{stage}");
    let rows = trace.attn_mixed.seq_len;
    let mut stages = vec![
        stage(ctx, golden, name("attn_mixed"), &trace.attn_mixed)?,
        stage(ctx, golden, name("attn_inject"), &trace.attn_inject)?,
    ];
    let mut held = vec![
        "attn_out",
        "stream_after_attn",
        "router_logits",
        "moe_out",
        "stream_out",
    ];
    match (&trace.mixer, state) {
        (MixerTrace::Linear(t), MixerState::Linear(s)) => {
            stages.extend([
                stage(ctx, golden, name("gdn_qkv"), &t.qkv)?,
                stage(ctx, golden, name("gdn_z"), &t.z)?,
                stage(ctx, golden, name("gdn_b"), &t.b)?,
                stage(ctx, golden, name("gdn_a"), &t.a)?,
                stage(ctx, golden, name("gdn_core"), &t.core)?,
                stage(ctx, golden, name("gdn_normed"), &t.normed)?,
            ]);
            let recurrent: Vec<f32> = ctx.stream.clone_dtoh(&s.recurrent)?;
            ctx.sync()?;
            let reference = floats(golden, &name("gdn_state"))?.1;
            stages.push((name("gdn_state"), recurrent, reference));
            held.push("gdn_state");
        }
        (MixerTrace::Full(t), MixerState::Full(kv)) => {
            ensure_dense(golden, &name("attn_indexer_mask"), rows, kv.len)?;
            stages.extend([
                stage(ctx, golden, name("attn_q_proj"), &t.q_proj)?,
                stage(ctx, golden, name("attn_k_proj"), &t.k_proj)?,
                stage(ctx, golden, name("attn_v_proj"), &t.v_proj)?,
                (
                    name("q_norm_unrotated"),
                    unrotated(&t.q_prepped.to_host(ctx)?, g),
                    unrotated(&floats(golden, &name("attn_q_normed"))?.1, g),
                ),
                (
                    name("k_norm_unrotated"),
                    unrotated(&pool_k(ctx, kv, g, rows)?, g),
                    unrotated(&floats(golden, &name("attn_k_normed"))?.1, g),
                ),
                stage(ctx, golden, name("attn_gated"), &t.gated)?,
            ]);
        }
        _ => bail!("the trace and the state disagree on the layer kind"),
    }
    stages.extend([
        stage(ctx, golden, name("attn_out"), &trace.attn_out)?,
        stage(
            ctx,
            golden,
            name("stream_after_attn"),
            &trace.stream_after_attn,
        )?,
        stage(ctx, golden, name("mlp_mixed"), &trace.mlp_mixed)?,
        stage(ctx, golden, name("mlp_inject"), &trace.mlp_inject)?,
        stage(ctx, golden, name("router_logits"), &trace.moe.router_logits)?,
        stage(
            ctx,
            golden,
            name("shared_gate_logit"),
            &trace.moe.shared_gate_logit,
        )?,
        stage(ctx, golden, name("shared_out"), &trace.moe.shared_out)?,
        stage(ctx, golden, name("expert_out"), &trace.moe.expert_out)?,
        stage(ctx, golden, name("moe_out"), &trace.moe.out)?,
        stage(ctx, golden, name("stream_out"), &trace.stream_out)?,
    ]);
    let mut measured = Vec::new();
    for (stage, ours, reference) in &stages {
        let d = diff(ours, reference);
        print_diff(stage, &d);
        measured.push((stage.as_str(), d));
    }

    // The picks must be a top-k of this engine's own logits, and the logits are
    // held to the reference's floor below, so a pick that differs from the
    // reference's can only sit where the reference's own gap is inside that
    // floor. The gaps are printed for reading.
    let (_, logits_ref) = floats(golden, &name("router_logits"))?;
    let picks_ref = ints(golden, &name("selected_experts"))?;
    let logits_ours = trace.moe.router_logits.to_host(ctx)?;
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
    let (_, weights_ref) = floats(golden, &name("routing_weights"))?;
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

    for stage in held {
        let full = name(stage);
        let ours = &measured
            .iter()
            .find(|(measured, _)| *measured == full)
            .expect("measured")
            .1;
        hold_floor(golden, &full, ours)?;
    }
    Ok(())
}

/// One pass of the model against the reference under `prefix`: the embedding
/// stream must match exactly, every layer is held, and the logits the head
/// produces are held to the reference's own floor. The argmax agreement is
/// printed with the reference's own gap where it disagrees.
fn hold_pass(
    ctx: &DeviceContext,
    golden: &SafeTensors,
    model: &Model,
    layers: &[usize],
    prefix: &str,
    forward: &Forward,
    state: &ModelState,
) -> Result<()> {
    let g = &model.g;
    let (_, stream_ref) = floats(golden, &format!("{prefix}stream_in"))?;
    let stream_diff = diff(&forward.stream_in.to_host(ctx)?, &stream_ref);
    print_diff(&format!("{prefix}stream_in"), &stream_diff);
    ensure!(
        stream_diff.max_abs == 0.0,
        "the embedding stream differs from the reference by {:.3e}",
        stream_diff.max_abs
    );
    for ((layer, trace), mixer) in layers.iter().zip(&forward.layers).zip(&state.layers) {
        hold_layer(
            ctx,
            golden,
            g,
            &format!("{prefix}layer{layer}_"),
            trace,
            mixer,
        )?;
    }
    let (_, mixed_ref) = floats(golden, &format!("{prefix}mixed"))?;
    print_diff(
        &format!("{prefix}mixed"),
        &diff(&forward.mixed.to_host(ctx)?, &mixed_ref),
    );
    let name = format!("{prefix}logits");
    let (_, logits_ref) = floats(golden, &name)?;
    let logits_ours = forward.logits.to_host(ctx)?;
    let logits_diff = diff(&logits_ours, &logits_ref);
    print_diff(&name, &logits_diff);
    let rows = forward.logits.seq_len;
    let mut agree = 0usize;
    for token in 0..rows {
        let span = token * g.vocab..(token + 1) * g.vocab;
        let top = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, &v)| {
                    if v > best.1 { (i, v) } else { best }
                })
        };
        let (ours, _) = top(&logits_ours[span.clone()]);
        let (theirs, best) = top(&logits_ref[span.clone()]);
        if ours == theirs {
            agree += 1;
        } else {
            let gap = (best - logits_ref[span.start + ours]) / bf16_step(best.abs());
            eprintln!(
                "  token {token}: argmax {ours} against the reference's {theirs}, whose logits are {gap:.2} bf16 steps apart"
            );
        }
    }
    eprintln!("  argmax agrees on {agree}/{rows} tokens");
    hold_floor(golden, &name, &logits_diff)
}

#[test]
fn model_matches_the_reference() -> Result<()> {
    let (Some(model_dir), Some(golden_path)) = (
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
    let layers: Vec<usize> = manifest["layers"]
        .as_array()
        .context("manifest has no layers")?
        .iter()
        .map(|v| v.as_u64().map(|v| v as usize).context("layer index"))
        .collect::<Result<_>>()?;
    let decode_steps = manifest["decode_steps"].as_u64().unwrap_or(0) as usize;
    let ids: Vec<u32> = ints(&golden, "input_ids")?
        .iter()
        .map(|&id| u32::try_from(id).context("token id"))
        .collect::<Result<_>>()?;
    let prefill = ids.len() - decode_steps;

    let config = load_config(&model_dir)?;
    let ctx = DeviceContext::new()?;
    let model = Model::load(&ctx, &model_dir, &config, &layers, ids.len())?;
    let mut state = model.new_state(&ctx, ids.len())?;

    let forward = model.prefill(&ctx, &ids[..prefill], &mut state)?;
    ctx.sync()?;
    eprintln!(
        "qwen4 layers {layers:?} gate: {prefill} prompt tokens, max |ours - reference| per stage"
    );
    hold_pass(&ctx, &golden, &model, &layers, "", &forward, &state)?;
    for (step, &id) in ids[prefill..].iter().enumerate() {
        let forward = model.decode(&ctx, id, &mut state)?;
        ctx.sync()?;
        eprintln!(
            "qwen4 layers {layers:?} gate: decode step {step} at position {}",
            prefill + step
        );
        hold_pass(
            &ctx,
            &golden,
            &model,
            &layers,
            &format!("decode{step}_"),
            &forward,
            &state,
        )?;
    }
    Ok(())
}
