//! Holds a chain of layers with the bookends around them against the reference
//! recorded by `tools/accuracy/dump_qwen4_layer_golden.py` on the checkpoint's
//! own weights: the prompt in prefill and, when the golden carries them, decode
//! steps on the carried state. Every layer is held on the reference's own
//! input for it and the head on the reference's last stream; the engine's own
//! chain through the loaded layers is printed beside them. Needs
//! `PEGAINFER_QWEN4_MODEL_PATH` (the checkpoint, or the shards of those layers
//! and the bookends plus the index; the n-gram layer also needs the table's
//! shards) and `PEGAINFER_QWEN4_LAYER_GOLDEN`. With
//! `PEGAINFER_QWEN4_GATE_STREAMED` set, one layer is resident at a time and
//! every pass goes through it before the next loads, which is how all 48
//! layers go through one card.

use std::collections::BTreeSet;
use std::path::Path;
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

use super::Model;
use crate::config::Config;
use crate::layer::KvScratch;
use crate::layer::LayerState;
use crate::layer::LayerTrace;
use crate::layer::MixerState;
use crate::layer::MixerTrace;
use crate::layer::layer_decode;
use crate::layer::layer_prefill;
use crate::weights::DecoderLayer;
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

/// `diff` over the rows `keep` marks, of `[keep.len(), width]` values.
fn diff_rows(ours: &[f32], reference: &[f32], keep: &[bool]) -> Diff {
    let width = ours.len() / keep.len();
    let kept = |values: &[f32]| -> Vec<f32> {
        values
            .chunks(width)
            .zip(keep)
            .filter(|(_, keep)| **keep)
            .flat_map(|(row, _)| row.iter().copied())
            .collect()
    };
    diff(&kept(ours), &kept(reference))
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
    state: &LayerState,
) -> Result<()> {
    let name = |stage: &str| format!("{prefix}{stage}");
    let rows = trace.attn_mixed.seq_len;
    let mut stages = Vec::new();
    let mut held = vec![
        "attn_out",
        "stream_after_attn",
        "router_logits",
        "moe_out",
        "stream_out",
    ];
    if let Some(ple) = &trace.ple {
        // The rows are gathered from the same table, so any difference is in
        // the hash.
        let (embeds, ours, reference) = stage(ctx, golden, name("ple_embeds"), &ple.embeds)?;
        let d = diff(&ours, &reference);
        print_diff(&embeds, &d);
        ensure!(
            d.max_abs == 0.0,
            "{embeds}: the gathered n-gram rows differ from the reference's by {:.3e}",
            d.max_abs
        );
        stages.extend([
            stage(ctx, golden, name("ple_key"), &ple.key)?,
            stage(ctx, golden, name("ple_value"), &ple.value)?,
            stage(ctx, golden, name("ple_out"), &ple.out)?,
            stage(ctx, golden, name("stream_ple"), &ple.stream)?,
        ]);
        held.push("ple_out");
    }
    stages.extend([
        stage(ctx, golden, name("attn_mixed"), &trace.attn_mixed)?,
        stage(ctx, golden, name("attn_inject"), &trace.attn_inject)?,
    ]);
    match (&trace.mixer, &state.mixer) {
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
    let mut agree = vec![true; rows];
    for (token, agrees) in agree.iter_mut().enumerate() {
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
        *agrees = false;
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
    // A token that picks another expert set runs other experts, so its routed
    // output has nothing to match; the flip itself is bounded by the logits
    // hold, as above. The routed output and the stream it lands in are held on
    // the tokens whose sets agree.
    if differing > 0 {
        for stage in ["moe_out", "stream_out"] {
            let full = name(stage);
            let (_, ours, reference) = stages
                .iter()
                .find(|(staged, _, _)| *staged == full)
                .expect("staged");
            let d = diff_rows(ours, reference, &agree);
            eprintln!(
                "  {full:>30}: max {:.3e} on the {} tokens whose expert sets agree",
                d.max_abs,
                rows - differing
            );
            measured
                .iter_mut()
                .find(|(measured, _)| *measured == full)
                .expect("measured")
                .1 = d;
        }
    }

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

/// The embedding stream of a pass must match the reference exactly.
fn hold_stream_in(
    ctx: &DeviceContext,
    golden: &SafeTensors,
    prefix: &str,
    stream: &HiddenStates,
) -> Result<()> {
    let name = format!("{prefix}stream_in");
    let (_, reference) = floats(golden, &name)?;
    let d = diff(&stream.to_host(ctx)?, &reference);
    print_diff(&name, &d);
    ensure!(
        d.max_abs == 0.0,
        "{name}: the embedding stream differs from the reference by {:.3e}",
        d.max_abs
    );
    Ok(())
}

/// A pass's readout against the reference's: the mixed row and the logits are
/// printed, with the argmax agreement and the reference's own gap where it
/// disagrees. Returns the logits' distance.
fn compare_readout(
    ctx: &DeviceContext,
    golden: &SafeTensors,
    g: &Geometry,
    prefix: &str,
    mixed: &HiddenStates,
    logits: &HiddenStates,
) -> Result<Diff> {
    let (_, mixed_ref) = floats(golden, &format!("{prefix}mixed"))?;
    print_diff(
        &format!("{prefix}mixed"),
        &diff(&mixed.to_host(ctx)?, &mixed_ref),
    );
    let name = format!("{prefix}logits");
    let (_, logits_ref) = floats(golden, &name)?;
    let logits_ours = logits.to_host(ctx)?;
    let logits_diff = diff(&logits_ours, &logits_ref);
    print_diff(&name, &logits_diff);
    let rows = logits.seq_len;
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
    Ok(logits_diff)
}

/// The reference's stream under `name`, uploaded as the engine's.
fn reference_stream(
    ctx: &DeviceContext,
    golden: &SafeTensors,
    g: &Geometry,
    name: &str,
    rows: usize,
) -> Result<HiddenStates> {
    // The stream is recorded in bf16, so the round trip through f32 is exact.
    let values: Vec<bf16> = floats(golden, name)?
        .1
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    ensure!(
        values.len() == rows * g.stream_width(),
        "{name} holds {} values, not {rows} rows of the stream",
        values.len()
    );
    HiddenStates::from_host(ctx, &values, g.stream_width(), rows)
}

/// What each loaded layer reads from the reference: the embedding stream for
/// the first, the previous loaded layer's output for the rest.
fn layer_inputs(layers: &[usize]) -> Vec<String> {
    std::iter::once("stream_in".to_string())
        .chain(
            layers
                .iter()
                .map(|layer| format!("layer{layer}_stream_out")),
        )
        .take(layers.len())
        .collect()
}

/// One layer held on the reference's own input: every pass reads the stream
/// the reference fed this layer, so each held stage measures this layer alone,
/// against the reference's spread for this layer alone (the CPU pass of the
/// reference is fed the same way). Down a chain both distances from the
/// accelerator compound, and one pass of one stage is a single draw of that.
fn hold_layer_on_reference(
    ctx: &DeviceContext,
    golden: &SafeTensors,
    model: &Model,
    w: &DecoderLayer,
    layer: usize,
    input: &str,
    passes: &[(String, &[u32])],
) -> Result<()> {
    let g = &model.g;
    let total = passes.iter().map(|(_, ids)| ids.len()).sum();
    let mut state = LayerState::new(ctx, g, w, total)?;
    for (pass, (prefix, ids)) in passes.iter().enumerate() {
        let stream = reference_stream(ctx, golden, g, &format!("{prefix}{input}"), ids.len())?;
        let trace = if pass == 0 {
            layer_prefill(ctx, g, w, &model.rope, &mut state, ids, &stream)?
        } else {
            layer_decode(ctx, g, w, &model.rope, &mut state, ids[0], &stream)?
        };
        ctx.sync()?;
        hold_layer(
            ctx,
            golden,
            g,
            &format!("{prefix}layer{layer}_"),
            &trace,
            &state,
        )?;
    }
    Ok(())
}

/// The readouts of every pass: the head on the reference's last stream, held
/// to the reference's own floor for the head, then the engine's own chain
/// through the loaded layers, printed only, since down a chain the distance is
/// compounded rounding with no floor of its own.
fn hold_readouts(
    ctx: &DeviceContext,
    golden: &SafeTensors,
    model: &Model,
    last: usize,
    passes: &[(String, &[u32])],
    chained: &[(HiddenStates, HiddenStates)],
) -> Result<()> {
    let g = &model.g;
    for ((prefix, ids), (chain_mixed, chain_logits)) in passes.iter().zip(chained) {
        let stream = reference_stream(
            ctx,
            golden,
            g,
            &format!("{prefix}layer{last}_stream_out"),
            ids.len(),
        )?;
        let (mixed, logits) = model.readout(ctx, &stream)?;
        ctx.sync()?;
        let d = compare_readout(ctx, golden, g, prefix, &mixed, &logits)?;
        hold_floor(golden, &format!("{prefix}logits"), &d)?;
        eprintln!("  the engine's own chain:");
        compare_readout(ctx, golden, g, prefix, chain_mixed, chain_logits)?;
    }
    Ok(())
}

/// The passes a golden records, each with the prefix its tensors carry: the
/// prompt's prefill, then one decode step per remaining token.
fn split_passes(ids: &[u32], decode_steps: usize) -> Vec<(String, &[u32])> {
    let prefill = ids.len() - decode_steps;
    std::iter::once((String::new(), &ids[..prefill]))
        .chain((0..decode_steps).map(|step| {
            let at = prefill + step;
            (format!("decode{step}_"), &ids[at..=at])
        }))
        .collect()
}

/// Every layer resident: each is held on the reference's input, and the
/// engine's own chain is the forward `Model::prefill` and `Model::decode` run.
fn hold_resident(
    ctx: &DeviceContext,
    golden: &SafeTensors,
    model: &Model,
    passes: &[(String, &[u32])],
) -> Result<()> {
    let total = passes.iter().map(|(_, ids)| ids.len()).sum();
    let mut state = model.new_state(ctx, total)?;
    let mut chained = Vec::with_capacity(passes.len());
    for (pass, (prefix, ids)) in passes.iter().enumerate() {
        let forward = if pass == 0 {
            model.prefill(ctx, ids, &mut state)?
        } else {
            model.decode(ctx, ids[0], &mut state)?
        };
        ctx.sync()?;
        hold_stream_in(ctx, golden, prefix, &forward.stream_in)?;
        chained.push((forward.mixed, forward.logits));
    }
    let layers: Vec<usize> = model.layers.iter().map(|(layer, _)| *layer).collect();
    for ((layer, w), input) in model.layers.iter().zip(layer_inputs(&layers)) {
        hold_layer_on_reference(ctx, golden, model, w, *layer, &input, passes)?;
    }
    let last = *layers.last().context("the golden names no layer")?;
    hold_readouts(ctx, golden, model, last, passes, &chained)
}

/// One layer resident at a time, every pass through it before the next layer
/// loads, held on the reference's input and then run on the engine's own chain.
/// A layer's pass reads only its own state and the previous layer's output for
/// that pass, so this order computes the numbers the resident forward computes
/// with one layer's weights on the card, which is what lets all 48 layers go
/// through one card against the reference. `model` carries the bookends and the
/// RoPE tables; the layers come and go.
fn hold_streamed(
    ctx: &DeviceContext,
    golden: &SafeTensors,
    model_dir: &Path,
    config: &Config,
    model: &Model,
    layers: &[usize],
    passes: &[(String, &[u32])],
) -> Result<()> {
    let g = &model.g;
    let total = passes.iter().map(|(_, ids)| ids.len()).sum();
    let mut streams = Vec::with_capacity(passes.len());
    for (prefix, ids) in passes {
        let stream = model.embed(ctx, ids)?;
        hold_stream_in(ctx, golden, prefix, &stream)?;
        streams.push(stream);
    }
    for (&layer, input) in layers.iter().zip(layer_inputs(layers)) {
        let w = load_layer(ctx, model_dir, config, layer)?;
        eprintln!("qwen4 streamed gate: layer {layer} resident");
        hold_layer_on_reference(ctx, golden, model, &w, layer, &input, passes)?;
        let mut state = LayerState::new(ctx, g, &w, total)?;
        for (pass, ((_, ids), stream)) in passes.iter().zip(&mut streams).enumerate() {
            let trace = if pass == 0 {
                layer_prefill(ctx, g, &w, &model.rope, &mut state, ids, stream)?
            } else {
                layer_decode(ctx, g, &w, &model.rope, &mut state, ids[0], stream)?
            };
            *stream = trace.stream_out;
        }
    }
    let chained = streams
        .iter()
        .map(|stream| model.readout(ctx, stream))
        .collect::<Result<Vec<_>>>()?;
    ctx.sync()?;
    let last = *layers.last().context("the golden names no layer")?;
    hold_readouts(ctx, golden, model, last, passes, &chained)
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
    ensure!(
        manifest["format_version"].as_u64() == Some(5),
        "the golden is format {}, and the gate reads format 5, whose floors are each layer's own",
        manifest["format_version"]
    );
    let decode_steps = manifest["decode_steps"].as_u64().unwrap_or(0) as usize;
    let ids: Vec<u32> = ints(&golden, "input_ids")?
        .iter()
        .map(|&id| u32::try_from(id).context("token id"))
        .collect::<Result<_>>()?;
    let prefill = ids.len() - decode_steps;

    let config = load_config(&model_dir)?;
    let ctx = DeviceContext::new()?;
    let passes = split_passes(&ids, decode_steps);
    eprintln!(
        "qwen4 layers {layers:?} gate: {prefill} prompt tokens then {decode_steps} decode steps, max |ours - reference| per stage"
    );
    if std::env::var_os("PEGAINFER_QWEN4_GATE_STREAMED").is_some() {
        let model = Model::load(&ctx, &model_dir, &config, &[], ids.len())?;
        hold_streamed(&ctx, &golden, &model_dir, &config, &model, &layers, &passes)
    } else {
        let model = Model::load(&ctx, &model_dir, &config, &layers, ids.len())?;
        hold_resident(&ctx, &golden, &model, &passes)
    }
}
