//! Holds a chain of layers with the bookends around them against the reference
//! recorded by `tools/accuracy/dump_qwen4_layer_golden.py` on the checkpoint's
//! own weights: the prompt in prefill and, when the golden carries them, decode
//! steps on the carried state. Every layer is held on the reference's own
//! input for it and the head on the reference's last stream; the engine's own
//! chain through the loaded layers is printed beside them. Needs
//! `PEGAINFER_QWEN4_MODEL_PATH` (the checkpoint, or the shards of those layers
//! and the bookends plus the index; the n-gram layer also needs the table's
//! shards) and `PEGAINFER_QWEN4_LAYER_GOLDEN`, one golden or several joined
//! like `PATH`. Each golden is one sequence in its own slot: the prompts are
//! prefilled one at a time, and each decode step runs every sequence that still
//! has one as one batch, each row held against its own golden. With
//! `PEGAINFER_QWEN4_GATE_STREAMED` set, one layer is resident at a time and
//! every pass goes through it before the next loads, which is how all 48
//! layers go through one card. `PEGAINFER_QWEN4_GATE_DEVICES` (comma separated
//! ordinals) does the same with the engine's own chain run across those cards.

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::ops::Range;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context as _;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use half::bf16;
use pegainfer_kernels::ops::copy_hidden_token_range_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;
use safetensors::Dtype;
use safetensors::SafeTensors;

use super::Model;
use super::split::Output;
use super::split::Readout;
use super::split::SplitModel;
use crate::config::Config;
use crate::layer::KvPool;
use crate::layer::LayerState;
use crate::layer::LayerTrace;
use crate::layer::MixerState;
use crate::layer::MixerTrace;
use crate::layer::layer_decode;
use crate::layer::layer_prefill;
use crate::weights::DecoderLayer;
use crate::weights::Geometry;
use crate::weights::Mixer;
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

/// One pass's floor for a stage the reference also recorded on the CPU: its
/// own accelerator versus CPU spread, which a bf16 port cannot sit below. A
/// bf16 stage is rounded value by value, so two faithful runs can sit one bf16
/// step apart at its largest value, and a spread under that step is a pair of
/// runs that happened to round alike: such a stage's floor is at least that
/// step. Every floor is at least 1e-3.
fn pass_floor(golden: &SafeTensors, name: &str) -> Result<f32> {
    let (_, accel) = floats(golden, name)?;
    let (_, cpu) = floats(golden, &format!("cpu_{name}"))?;
    let spread = diff(&cpu, &accel);
    let step = match golden.tensor(name)?.dtype() {
        Dtype::BF16 => bf16_step(spread.max_ref),
        _ => 0.0,
    };
    Ok(spread.max_abs.max(step).max(1e-3))
}

/// A stage is held to four times the largest floor its passes show over the
/// sequence, which is the same as holding our largest distance over the
/// sequence to four times the reference's. One pass's spread is a single draw
/// of how far two faithful runs sit apart, and a draw can come out near
/// nothing: a one token prompt's layer 3 router logits agreed exactly, and the
/// recurrent state's spread shrinks with the decay while the distance it is
/// held against carries every earlier step (a one token prompt's layer 0 state
/// sat 2.4e-2 apart at decode step 3 and 3.4e-3 apart at step 5).
fn sequence_floor(seq: &Sequence, stage: &str) -> Result<f32> {
    seq.passes().iter().try_fold(0f32, |floor, (prefix, _)| {
        Ok(floor.max(pass_floor(&seq.golden, &format!("{prefix}{stage}"))?))
    })
}

fn hold(name: &str, ours: &Diff, floor: f32) -> Result<()> {
    let line = 4.0 * floor;
    eprintln!("  {name:>30}: reference floor over the sequence {floor:.3e}, line {line:.3e}");
    ensure!(
        ours.max_abs <= line,
        "{name}: max |ours - reference| {:.3e} exceeds {line:.3e}, four times the reference's own floor over the sequence",
        ours.max_abs
    );
    Ok(())
}

/// The rows `rows` of a batch, on the host.
fn host_rows(ctx: &DeviceContext, ours: &HiddenStates, rows: &Range<usize>) -> Result<Vec<f32>> {
    let width = ours.hidden_dim;
    Ok(ours.to_host(ctx)?[rows.start * width..rows.end * width].to_vec())
}

/// One stage of one sequence, the rows `rows` of the batch, as the engine
/// computed it and as the reference recorded it.
fn stage(
    ctx: &DeviceContext,
    golden: &SafeTensors,
    name: String,
    ours: &HiddenStates,
    rows: &Range<usize>,
) -> Result<(String, Vec<f32>, Vec<f32>)> {
    let reference = floats(golden, &name)?.1;
    Ok((name, host_rows(ctx, ours, rows)?, reference))
}

/// Several passes' rows stacked into one batch, in order.
fn stack(ctx: &DeviceContext, parts: &[&HiddenStates]) -> Result<HiddenStates> {
    let width = parts.first().context("nothing to stack")?.hidden_dim;
    let mut batch = HiddenStates::zeros(ctx, width, parts.iter().map(|p| p.seq_len).sum())?;
    let mut at = 0;
    for part in parts {
        copy_hidden_token_range_into(ctx, part, 0, &mut batch, at, part.seq_len)?;
        at += part.seq_len;
    }
    Ok(batch)
}

/// Row `row` of a batch on its own.
fn row_of(ctx: &DeviceContext, batch: &HiddenStates, row: usize) -> Result<HiddenStates> {
    let mut one = HiddenStates::zeros(ctx, batch.hidden_dim, 1)?;
    copy_hidden_token_range_into(ctx, batch, row, &mut one, 0, 1)?;
    Ok(one)
}

/// The dims the partial RoPE leaves alone, out of `[rows * heads, head_dim]`.
fn unrotated(values: &[f32], g: &Geometry) -> Vec<f32> {
    values
        .chunks_exact(g.head_dim)
        .flat_map(|head| head[g.rotary_dim..].iter().copied())
        .collect()
}

/// K of the last `rows` positions of `slot` as the prep kernel wrote it into
/// the pool, back in `[rows * kv_heads, head_dim]`.
fn pool_k(
    ctx: &DeviceContext,
    kv: &KvPool,
    g: &Geometry,
    slot: usize,
    rows: usize,
) -> Result<Vec<f32>> {
    let pool: Vec<bf16> = ctx.stream.clone_dtoh(&kv.pool)?;
    ctx.sync()?;
    let width = g.kv_width();
    let (first, len) = (kv.first_page(slot), kv.lens[slot]);
    let mut k = Vec::with_capacity(rows * width);
    for pos in len - rows..len {
        let row = (first + pos / kv.layout.page_size) * kv.layout.page_stride
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

/// One sequence's rows `rows` of a layer's forward, in `slot`, against the
/// reference's stages under `prefix`: every stage is printed, the picks must be
/// a top-k of the engine's own logits, and the floored stages are held to the
/// sequence's `floors`.
#[allow(clippy::too_many_arguments)]
fn hold_layer(
    ctx: &DeviceContext,
    golden: &SafeTensors,
    g: &Geometry,
    prefix: &str,
    trace: &LayerTrace,
    rows: Range<usize>,
    state: &LayerState,
    slot: usize,
    floors: &BTreeMap<&str, f32>,
) -> Result<()> {
    let name = |stage: &str| format!("{prefix}{stage}");
    let count = rows.len();
    let mut stages = Vec::new();
    if let Some(ple) = &trace.ple {
        // The rows are gathered from the same table, so any difference is in
        // the hash.
        let (embeds, ours, reference) = stage(ctx, golden, name("ple_embeds"), &ple.embeds, &rows)?;
        let d = diff(&ours, &reference);
        print_diff(&embeds, &d);
        ensure!(
            d.max_abs == 0.0,
            "{embeds}: the gathered n-gram rows differ from the reference's by {:.3e}",
            d.max_abs
        );
        stages.extend([
            stage(ctx, golden, name("ple_key"), &ple.key, &rows)?,
            stage(ctx, golden, name("ple_value"), &ple.value, &rows)?,
            stage(ctx, golden, name("ple_out"), &ple.out, &rows)?,
            stage(ctx, golden, name("stream_ple"), &ple.stream, &rows)?,
        ]);
    }
    stages.extend([
        stage(ctx, golden, name("attn_mixed"), &trace.attn_mixed, &rows)?,
        stage(ctx, golden, name("attn_inject"), &trace.attn_inject, &rows)?,
    ]);
    match (&trace.mixer, &state.mixer) {
        (MixerTrace::Linear(t), MixerState::Linear(states)) => {
            stages.extend([
                stage(ctx, golden, name("gdn_qkv"), &t.qkv, &rows)?,
                stage(ctx, golden, name("gdn_z"), &t.z, &rows)?,
                stage(ctx, golden, name("gdn_b"), &t.b, &rows)?,
                stage(ctx, golden, name("gdn_a"), &t.a, &rows)?,
                stage(ctx, golden, name("gdn_core"), &t.core, &rows)?,
                stage(ctx, golden, name("gdn_normed"), &t.normed, &rows)?,
            ]);
            let recurrent: Vec<f32> = ctx.stream.clone_dtoh(&states[slot].recurrent)?;
            ctx.sync()?;
            let reference = floats(golden, &name("gdn_state"))?.1;
            stages.push((name("gdn_state"), recurrent, reference));
        }
        (MixerTrace::Full(t), MixerState::Full(kv)) => {
            ensure_dense(golden, &name("attn_indexer_mask"), count, kv.lens[slot])?;
            stages.extend([
                stage(ctx, golden, name("attn_q_proj"), &t.q_proj, &rows)?,
                stage(ctx, golden, name("attn_k_proj"), &t.k_proj, &rows)?,
                stage(ctx, golden, name("attn_v_proj"), &t.v_proj, &rows)?,
                (
                    name("q_norm_unrotated"),
                    unrotated(&host_rows(ctx, &t.q_prepped, &rows)?, g),
                    unrotated(&floats(golden, &name("attn_q_normed"))?.1, g),
                ),
                (
                    name("k_norm_unrotated"),
                    unrotated(&pool_k(ctx, kv, g, slot, count)?, g),
                    unrotated(&floats(golden, &name("attn_k_normed"))?.1, g),
                ),
                stage(ctx, golden, name("attn_gated"), &t.gated, &rows)?,
            ]);
        }
        _ => bail!("the trace and the state disagree on the layer kind"),
    }
    stages.extend([
        stage(ctx, golden, name("attn_out"), &trace.attn_out, &rows)?,
        stage(
            ctx,
            golden,
            name("stream_after_attn"),
            &trace.stream_after_attn,
            &rows,
        )?,
        stage(ctx, golden, name("mlp_mixed"), &trace.mlp_mixed, &rows)?,
        stage(ctx, golden, name("mlp_inject"), &trace.mlp_inject, &rows)?,
        stage(
            ctx,
            golden,
            name("router_logits"),
            &trace.moe.router_logits,
            &rows,
        )?,
        stage(
            ctx,
            golden,
            name("shared_gate_logit"),
            &trace.moe.shared_gate_logit,
            &rows,
        )?,
        stage(
            ctx,
            golden,
            name("shared_out"),
            &trace.moe.shared_out,
            &rows,
        )?,
        stage(
            ctx,
            golden,
            name("expert_out"),
            &trace.moe.expert_out,
            &rows,
        )?,
        stage(ctx, golden, name("moe_out"), &trace.moe.out, &rows)?,
        stage(ctx, golden, name("stream_out"), &trace.stream_out, &rows)?,
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
    let logits_ours = host_rows(ctx, &trace.moe.router_logits, &rows)?;
    let pairs_of = rows.start * g.top_k..rows.end * g.top_k;
    let picks_ours: Vec<i32> = ctx.stream.clone_dtoh(&trace.moe.selected_experts)?;
    ctx.sync()?;
    let picks_ours = &picks_ours[pairs_of.clone()];
    let mut differing = 0usize;
    let mut agree = vec![true; count];
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
    let weights_ours: Vec<f32> = ctx.stream.clone_dtoh(&trace.moe.routing_weights)?[pairs_of]
        .iter()
        .map(|w: &bf16| w.to_f32())
        .collect();
    ctx.sync()?;
    let weight_diff = diff(&weights_ours, &weights_ref);
    eprintln!(
        "  routing: {differing}/{count} tokens pick a different expert set, weights max {:.3e}",
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
                count - differing
            );
            measured
                .iter_mut()
                .find(|(measured, _)| *measured == full)
                .expect("measured")
                .1 = d;
        }
    }

    for (stage, floor) in floors {
        let full = name(stage);
        let ours = &measured
            .iter()
            .find(|(measured, _)| *measured == full)
            .with_context(|| format!("{full} is held but was not measured"))?
            .1;
        hold(&full, ours, *floor)?;
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

/// One golden: a sequence's prompt in prefill, then one decode step per
/// remaining token.
struct Sequence<'a> {
    golden: SafeTensors<'a>,
    layers: Vec<usize>,
    ids: Vec<u32>,
    decode_steps: usize,
}

impl<'a> Sequence<'a> {
    fn read(bytes: &'a [u8]) -> Result<Self> {
        let golden = SafeTensors::deserialize(bytes)?;
        let (_, metadata) = SafeTensors::read_metadata(bytes)?;
        let manifest: serde_json::Value = serde_json::from_str(
            metadata
                .metadata()
                .as_ref()
                .and_then(|m| m.get(METADATA_KEY))
                .context("golden carries no manifest")?,
        )?;
        ensure!(
            manifest["format_version"].as_u64() == Some(5),
            "the golden is format {}, and the gate reads format 5, whose floors are each layer's own",
            manifest["format_version"]
        );
        let layers = manifest["layers"]
            .as_array()
            .context("manifest has no layers")?
            .iter()
            .map(|v| v.as_u64().map(|v| v as usize).context("layer index"))
            .collect::<Result<_>>()?;
        let decode_steps = manifest["decode_steps"].as_u64().unwrap_or(0) as usize;
        let ids = ints(&golden, "input_ids")?
            .iter()
            .map(|&id| u32::try_from(id).context("token id"))
            .collect::<Result<_>>()?;
        Ok(Self {
            golden,
            layers,
            ids,
            decode_steps,
        })
    }

    fn prompt(&self) -> &[u32] {
        &self.ids[..self.ids.len() - self.decode_steps]
    }

    /// The token decode step `step` feeds.
    fn token(&self, step: usize) -> u32 {
        self.ids[self.ids.len() - self.decode_steps + step]
    }

    /// The passes it records, each with the prefix its tensors carry.
    fn passes(&self) -> Vec<(String, &[u32])> {
        let prefill = self.prompt().len();
        std::iter::once((String::new(), self.prompt()))
            .chain((0..self.decode_steps).map(|step| {
                let at = prefill + step;
                (format!("decode{step}_"), &self.ids[at..=at])
            }))
            .collect()
    }
}

fn decode_steps(sequences: &[Sequence]) -> usize {
    sequences.iter().map(|s| s.decode_steps).max().unwrap_or(0)
}

/// Tokens the longest sequence reaches, what each slot has to hold.
fn capacity(sequences: &[Sequence]) -> usize {
    sequences.iter().map(|s| s.ids.len()).max().unwrap_or(0)
}

/// The slots decode step `step` batches, every sequence that has one, and the
/// token each feeds.
fn batch_at(sequences: &[Sequence], step: usize) -> (Vec<usize>, Vec<u32>) {
    sequences
        .iter()
        .enumerate()
        .filter(|(_, s)| step < s.decode_steps)
        .map(|(slot, s)| (slot, s.token(step)))
        .unzip()
}

/// One layer held on the reference's own input: every pass reads the stream
/// the reference fed this layer, so each held stage measures this layer alone,
/// against the reference's spread for this layer alone (the CPU pass of the
/// reference is fed the same way). Down a chain both distances from the
/// accelerator compound, and one pass of one stage is a single draw of that.
/// Each prompt runs in its own slot, and each decode step stacks the
/// reference's rows of every sequence that has one into one batch.
fn hold_layer_on_reference(
    ctx: &DeviceContext,
    sequences: &[Sequence],
    model: &Model,
    w: &DecoderLayer,
    layer: usize,
    input: &str,
) -> Result<()> {
    let g = &model.g;
    let mut held = vec![
        "attn_out",
        "stream_after_attn",
        "router_logits",
        "moe_out",
        "stream_out",
    ];
    if w.ple.is_some() {
        held.push("ple_out");
    }
    if let Mixer::Linear(_) = w.mixer {
        held.push("gdn_state");
    }
    let floors = sequences
        .iter()
        .map(|seq| {
            held.iter()
                .map(|&stage| {
                    Ok((
                        stage,
                        sequence_floor(seq, &format!("layer{layer}_{stage}"))?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    let mut state = LayerState::new(ctx, g, w, sequences.len(), capacity(sequences))?;
    for (slot, seq) in sequences.iter().enumerate() {
        let prompt = seq.prompt();
        let stream = reference_stream(ctx, &seq.golden, g, input, prompt.len())?;
        let trace = layer_prefill(ctx, g, w, &model.rope, &mut state, slot, prompt, &stream)?;
        ctx.sync()?;
        eprintln!("sequence {slot}:");
        let prefix = format!("layer{layer}_");
        hold_layer(
            ctx,
            &seq.golden,
            g,
            &prefix,
            &trace,
            0..prompt.len(),
            &state,
            slot,
            &floors[slot],
        )?;
    }
    for step in 0..decode_steps(sequences) {
        let (slots, ids) = batch_at(sequences, step);
        let prefix = format!("decode{step}_");
        let rows = slots
            .iter()
            .map(|&slot| {
                let name = format!("{prefix}{input}");
                reference_stream(ctx, &sequences[slot].golden, g, &name, 1)
            })
            .collect::<Result<Vec<_>>>()?;
        let stream = stack(ctx, &rows.iter().collect::<Vec<_>>())?;
        let trace = layer_decode(ctx, g, w, &model.rope, &mut state, &slots, &ids, &stream)?;
        ctx.sync()?;
        for (row, &slot) in slots.iter().enumerate() {
            eprintln!("sequence {slot}:");
            hold_layer(
                ctx,
                &sequences[slot].golden,
                g,
                &format!("{prefix}layer{layer}_"),
                &trace,
                row..row + 1,
                &state,
                slot,
                &floors[slot],
            )?;
        }
    }
    Ok(())
}

/// The readouts of every pass of every sequence: the head on the reference's
/// last stream, held to the reference's own floor for the head, then the
/// engine's own chain through the loaded layers, printed only, since down a
/// chain the distance is compounded rounding with no floor of its own.
fn hold_readouts(
    ctx: &DeviceContext,
    sequences: &[Sequence],
    model: &Model,
    last: usize,
    chained: &[Vec<(HiddenStates, HiddenStates)>],
) -> Result<()> {
    let g = &model.g;
    for ((slot, seq), chain) in sequences.iter().enumerate().zip(chained) {
        let logits_floor = sequence_floor(seq, "logits")?;
        for ((prefix, ids), (chain_mixed, chain_logits)) in seq.passes().iter().zip(chain) {
            let stream = reference_stream(
                ctx,
                &seq.golden,
                g,
                &format!("{prefix}layer{last}_stream_out"),
                ids.len(),
            )?;
            let (mixed, logits) = model.readout(ctx, &stream)?;
            ctx.sync()?;
            eprintln!("sequence {slot}:");
            let d = compare_readout(ctx, &seq.golden, g, prefix, &mixed, &logits)?;
            hold(&format!("{prefix}logits"), &d, logits_floor)?;
            eprintln!("  the engine's own chain:");
            compare_readout(ctx, &seq.golden, g, prefix, chain_mixed, chain_logits)?;
        }
    }
    Ok(())
}

/// Every layer resident: each is held on the reference's input, and the
/// engine's own chain is the forward `Model::prefill` and `Model::decode` run,
/// each prompt in its own slot and each decode step as one batch.
fn hold_resident(ctx: &DeviceContext, sequences: &[Sequence], model: &Model) -> Result<()> {
    let mut state = model.new_state(ctx, sequences.len(), capacity(sequences))?;
    let mut chained = Vec::with_capacity(sequences.len());
    for (slot, seq) in sequences.iter().enumerate() {
        let forward = model.prefill(ctx, slot, seq.prompt(), &mut state)?;
        ctx.sync()?;
        hold_stream_in(ctx, &seq.golden, "", &forward.stream_in)?;
        chained.push(vec![(forward.mixed, forward.logits)]);
    }
    for step in 0..decode_steps(sequences) {
        let (slots, ids) = batch_at(sequences, step);
        let forward = model.decode(ctx, &slots, &ids, &mut state)?;
        ctx.sync()?;
        let prefix = format!("decode{step}_");
        for (row, &slot) in slots.iter().enumerate() {
            let stream_in = row_of(ctx, &forward.stream_in, row)?;
            hold_stream_in(ctx, &sequences[slot].golden, &prefix, &stream_in)?;
            chained[slot].push((
                row_of(ctx, &forward.mixed, row)?,
                row_of(ctx, &forward.logits, row)?,
            ));
        }
    }
    let layers: Vec<usize> = model.layers.iter().map(|(layer, _)| *layer).collect();
    for ((layer, w), input) in model.layers.iter().zip(layer_inputs(&layers)) {
        hold_layer_on_reference(ctx, sequences, model, w, *layer, &input)?;
    }
    let last = *layers.last().context("the golden names no layer")?;
    hold_readouts(ctx, sequences, model, last, &chained)
}

/// One layer resident at a time, every pass through it before the next layer
/// loads, held on the reference's input and then run on the engine's own chain.
/// A layer's pass reads only its own state and the previous layer's output for
/// that pass, so this order computes the numbers the resident forward computes
/// with one layer's weights on the card, which is what lets all 48 layers go
/// through one card against the reference. The batches are the resident
/// forward's too: each prompt alone, each decode step's rows stacked, and the
/// readouts the same way. `model` carries the bookends and the RoPE tables;
/// the layers come and go. With `split` devices, the engine's own chain runs
/// through a [`SplitModel`] over them instead, each card holding a run of the
/// layers, and the readouts read the streams it hands back.
fn hold_streamed(
    ctx: &DeviceContext,
    sequences: &[Sequence],
    model_dir: &Path,
    config: &Config,
    model: &Model,
    layers: &[usize],
    split: Option<&[usize]>,
) -> Result<()> {
    let g = &model.g;
    // Every pass of every sequence, `streams[slot][pass]`.
    let mut streams = Vec::with_capacity(sequences.len());
    for seq in sequences {
        let mut passes = Vec::new();
        for (prefix, ids) in seq.passes() {
            let stream = model.embed(ctx, ids)?;
            hold_stream_in(ctx, &seq.golden, &prefix, &stream)?;
            passes.push(stream);
        }
        streams.push(passes);
    }
    let decode_stack = |streams: &[Vec<HiddenStates>], slots: &[usize], step: usize| {
        let rows: Vec<&HiddenStates> = slots.iter().map(|&s| &streams[s][step + 1]).collect();
        stack(ctx, &rows)
    };
    for (&layer, input) in layers.iter().zip(layer_inputs(layers)) {
        let w = load_layer(ctx, model_dir, config, layer)?;
        eprintln!("qwen4 streamed gate: layer {layer} resident");
        hold_layer_on_reference(ctx, sequences, model, &w, layer, &input)?;
        if split.is_some() {
            continue;
        }
        let mut state = LayerState::new(ctx, g, &w, sequences.len(), capacity(sequences))?;
        for (slot, seq) in sequences.iter().enumerate() {
            let prompt = &streams[slot][0];
            let trace = layer_prefill(
                ctx,
                g,
                &w,
                &model.rope,
                &mut state,
                slot,
                seq.prompt(),
                prompt,
            )?;
            streams[slot][0] = trace.stream_out;
        }
        for step in 0..decode_steps(sequences) {
            let (slots, ids) = batch_at(sequences, step);
            let input = decode_stack(&streams, &slots, step)?;
            let trace = layer_decode(ctx, g, &w, &model.rope, &mut state, &slots, &ids, &input)?;
            for (row, &slot) in slots.iter().enumerate() {
                streams[slot][step + 1] = row_of(ctx, &trace.stream_out, row)?;
            }
        }
    }
    if let Some(devices) = split {
        let cards = SplitModel::load(
            model_dir,
            devices,
            layers,
            sequences.len(),
            capacity(sequences),
        )?;
        let upload = |output: Output, rows: usize| match output {
            Output::Stream(host) => HiddenStates::from_host(ctx, &host, g.stream_width(), rows),
            Output::Logits(_) => bail!("asked the cards for the stream"),
        };
        for (slot, seq) in sequences.iter().enumerate() {
            let prompt = seq.prompt();
            let out = cards.prefill(slot, prompt, Readout::Stream)?;
            streams[slot][0] = upload(out, prompt.len())?;
        }
        for step in 0..decode_steps(sequences) {
            let (slots, ids) = batch_at(sequences, step);
            let out = upload(cards.decode(&slots, &ids, Readout::Stream)?, slots.len())?;
            for (row, &slot) in slots.iter().enumerate() {
                streams[slot][step + 1] = row_of(ctx, &out, row)?;
            }
        }
    }
    let mut chained = Vec::with_capacity(sequences.len());
    for passes in &streams {
        chained.push(vec![model.readout(ctx, &passes[0])?]);
    }
    for step in 0..decode_steps(sequences) {
        let (slots, _) = batch_at(sequences, step);
        let (mixed, logits) = model.readout(ctx, &decode_stack(&streams, &slots, step)?)?;
        for (row, &slot) in slots.iter().enumerate() {
            chained[slot].push((row_of(ctx, &mixed, row)?, row_of(ctx, &logits, row)?));
        }
    }
    ctx.sync()?;
    let last = *layers.last().context("the golden names no layer")?;
    hold_readouts(ctx, sequences, model, last, &chained)
}

#[test]
fn model_matches_the_reference() -> Result<()> {
    let (Some(model_dir), Some(goldens)) = (
        env_path("PEGAINFER_QWEN4_MODEL_PATH"),
        std::env::var_os("PEGAINFER_QWEN4_LAYER_GOLDEN"),
    ) else {
        eprintln!("skipped: set PEGAINFER_QWEN4_MODEL_PATH and PEGAINFER_QWEN4_LAYER_GOLDEN");
        return Ok(());
    };
    let bytes = std::env::split_paths(&goldens)
        .map(std::fs::read)
        .collect::<std::io::Result<Vec<_>>>()?;
    let sequences = bytes
        .iter()
        .map(|bytes| Sequence::read(bytes))
        .collect::<Result<Vec<_>>>()?;
    let layers = sequences.first().context("no golden named")?.layers.clone();
    ensure!(
        sequences.iter().all(|s| s.layers == layers),
        "the goldens record different layers"
    );

    let config = load_config(&model_dir)?;
    let ctx = DeviceContext::new()?;
    eprintln!("qwen4 layers {layers:?} gate, max |ours - reference| per stage");
    for (slot, seq) in sequences.iter().enumerate() {
        eprintln!(
            "sequence {slot}: {} prompt tokens then {} decode steps",
            seq.prompt().len(),
            seq.decode_steps
        );
    }
    let positions = capacity(&sequences);
    let split = std::env::var("PEGAINFER_QWEN4_GATE_DEVICES")
        .ok()
        .map(|list| {
            list.split(',')
                .map(|device| device.trim().parse::<usize>().context("a device ordinal"))
                .collect::<Result<Vec<_>>>()
        })
        .transpose()?;
    if split.is_some() || std::env::var_os("PEGAINFER_QWEN4_GATE_STREAMED").is_some() {
        let model = Model::load(&ctx, &model_dir, &config, &[], positions, true)?;
        hold_streamed(
            &ctx,
            &sequences,
            &model_dir,
            &config,
            &model,
            &layers,
            split.as_deref(),
        )
    } else {
        let model = Model::load(&ctx, &model_dir, &config, &layers, positions, true)?;
        hold_resident(&ctx, &sequences, &model)
    }
}
