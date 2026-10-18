//! Device-free build plan: how one rank's loaded checkpoint tensors become
//! kernel-ready weight slots.
//!
//! Every slot below is one entry of the certified decode engine's per-layer
//! weight struct, under the same name, so the executor reads the same things
//! the reference step reads. The plan is pure (names, shapes, dtypes, byte
//! counts) so the routing can be proven on the host: [`k3_model_slots`] must
//! consume exactly the non-expert tensor set the loader makes resident, each
//! name once, each slot's byte count agreeing with the loader's contract.
//!
//! Layout contract:
//!
//! * **Dense projections stay `[out, in]` row-major**, exactly as the
//!   checkpoint stores them. The executor's cuBLASLt bf16 GEMMs consume that
//!   orientation natively with `OP_T`, so no transpose is needed and every
//!   unfused projection is a zero-copy adopt. (The reference engine's `_t()`
//!   is an artifact of its own GEMV's K-major B operand; the TileLang `gemv`
//!   instantiations are not part of the batched executor path.) The useful
//!   consequence is that the reference's `cat(..., dim=1)` fusions of
//!   transposed matrices are exactly row concatenations of the untransposed
//!   ones — plain contiguous byte copies.
//! * **Routed experts stay in the loader's packing**: the DeepGEMM masked
//!   FP8xFP4 grouped GEMM takes its B operand as K-major packed
//!   `[experts, n rows, k / 2 bytes]`, which is byte-identical to the
//!   checkpoint's N-row-major storage, and its `n` for the gate|up call is the
//!   fused `[gate; up]` the loader already writes. Pure view, zero repack; only
//!   the e8m0 scale bytes are relaid out, on-device, at model build.
//! * **The KDA conv taps are the one genuine transpose.** `conv_silu` (a
//!   TileLang kernel that IS on the executor path) reads `[taps, inner]`, the
//!   reference's `squeeze(1).T`, while the checkpoint stores
//!   `[inner, 1, taps]`. These are ~196 KiB f32 tensors, so the build does the
//!   transpose on a host round trip.

use crate::config::K3_ATTN_INNER;
use crate::config::K3_ATTN_RES_BLOCK;
use crate::config::K3_DENSE_INTERMEDIATE;
use crate::config::K3_HEAD_DIM;
use crate::config::K3_HEADS;
use crate::config::K3_HIDDEN;
use crate::config::K3_KDA_CONV_KERNEL;
use crate::config::K3_KDA_GATE_RANK;
use crate::config::K3_KV_A_OUT;
use crate::config::K3_KV_B_OUT;
use crate::config::K3_KV_LORA_RANK;
use crate::config::K3_O_PROJ_IN;
use crate::config::K3_Q_B_OUT;
use crate::config::K3_Q_LORA_RANK;
use crate::config::K3_QK_HEAD_DIM;
use crate::config::K3_ROUTED_EXPERT_HIDDEN;
use crate::config::K3_ROUTED_SCALING_FACTOR;
use crate::config::K3_SHARED_INTERMEDIATE;
use crate::config::K3_VOCAB;
use crate::config::K3LayerKind;
use crate::config::K3RoutedExperts;
use crate::config::k3_layer_is_moe;
use crate::config::k3_layer_kind;

/// Padded row count of the KDA `wsm` projection. `b_proj` (one row per head)
/// and `f_a_proj` (the low-rank forget gate) are driven by one GEMV whose
/// output width is rounded up to a 64-lane multiple; the pad rows are zero, so
/// the padded lanes read as zero exactly as in the reference.
pub(crate) const K3_KDA_WSM_ROWS: usize = (K3_HEADS + K3_KDA_GATE_RANK).next_multiple_of(64);

/// Fused width of the KDA q/k/v/g GEMV.
pub(crate) const K3_KDA_BIG_ROWS: usize = 4 * K3_ATTN_INNER;

/// Fused width of the MLA q_a / kv_a / gate GEMV.
pub(crate) const K3_MLA_FUSED_ROWS: usize = K3_Q_LORA_RANK + K3_KV_A_OUT + K3_ATTN_INNER;

/// MLA softmax scale: `qk_head_dim ** -0.5`, the one value the reference keeps
/// as a device scalar rather than a kernel constant.
pub(crate) fn k3_mla_scale() -> f64 {
    (K3_QK_HEAD_DIM as f64).powf(-0.5)
}

/// Element type of a built slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum K3SlotDtype {
    Bf16,
    F32,
}

impl K3SlotDtype {
    pub(crate) fn bytes(self) -> usize {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

/// How a slot's device buffer is produced from its checkpoint sources.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum K3SlotBuild {
    /// Take the loader's buffer unchanged — retype only, no copy. The
    /// reference leaves these tensors alone too.
    Adopt,
    /// Concatenate the sources' rows into one buffer, in source order. The
    /// sources contribute `source_rows` rows; any rows past them stay zero
    /// (the KDA `wsm` pad).
    StackRows { source_rows: usize },
    /// Fold an RMSNorm gamma into the single projection row it always
    /// multiplies: `f32(gamma) * f32(proj)`. The attention-residual scoring
    /// vectors are the only place the reference precomputes a product.
    FoldNormIntoProj,
    /// Transpose an f32 source stored as `[cols, rows]` into the slot's
    /// `[rows, cols]`. Element count is unchanged, so nothing downstream can
    /// detect a missed transpose by size — the build performs it, on the host,
    /// and only for the small conv-tap tensors.
    TransposeF32,
    /// A launch constant the reference materializes as a device scalar.
    Constant(f64),
}

/// One destination buffer of the built model.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct K3SlotPlan {
    /// Field name in the built weight struct — the reference engine's key.
    pub(crate) slot: &'static str,
    pub(crate) build: K3SlotBuild,
    pub(crate) dtype: K3SlotDtype,
    /// Row-major extent of the destination. Vectors carry `cols == 1`.
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    /// Stripped checkpoint names, in fuse order.
    pub(crate) sources: Vec<String>,
}

impl K3SlotPlan {
    pub(crate) fn bytes(&self) -> usize {
        self.rows * self.cols * self.dtype.bytes()
    }
}

fn slot(
    name: &'static str,
    build: K3SlotBuild,
    dtype: K3SlotDtype,
    rows: usize,
    cols: usize,
    sources: &[String],
) -> K3SlotPlan {
    K3SlotPlan {
        slot: name,
        build,
        dtype,
        rows,
        cols,
        sources: sources.to_vec(),
    }
}

/// `[out, in]` bf16 matrix taken straight from the checkpoint.
fn adopt_matrix(name: &'static str, rows: usize, cols: usize, source: String) -> K3SlotPlan {
    slot(
        name,
        K3SlotBuild::Adopt,
        K3SlotDtype::Bf16,
        rows,
        cols,
        &[source],
    )
}

/// bf16 vector (an RMSNorm gamma) taken straight from the checkpoint.
fn adopt_vector(name: &'static str, len: usize, source: String) -> K3SlotPlan {
    slot(
        name,
        K3SlotBuild::Adopt,
        K3SlotDtype::Bf16,
        len,
        1,
        &[source],
    )
}

/// f32 tensor taken straight from the checkpoint. The reference casts these
/// with `.float()`, which is a no-op on an f32 checkpoint tensor.
fn adopt_f32(name: &'static str, rows: usize, cols: usize, source: String) -> K3SlotPlan {
    slot(
        name,
        K3SlotBuild::Adopt,
        K3SlotDtype::F32,
        rows,
        cols,
        &[source],
    )
}

fn stack_rows(
    name: &'static str,
    rows: usize,
    source_rows: usize,
    cols: usize,
    sources: &[String],
) -> K3SlotPlan {
    slot(
        name,
        K3SlotBuild::StackRows { source_rows },
        K3SlotDtype::Bf16,
        rows,
        cols,
        sources,
    )
}

/// The attention-residual scoring vector of one tap: `gamma * proj`, f32.
fn scoring_vector(name: &'static str, prefix: &str) -> K3SlotPlan {
    slot(
        name,
        K3SlotBuild::FoldNormIntoProj,
        K3SlotDtype::F32,
        K3_HIDDEN,
        1,
        &[
            format!("{prefix}norm.weight"),
            format!("{prefix}proj.weight"),
        ],
    )
}

fn constant(name: &'static str, value: f64) -> K3SlotPlan {
    slot(
        name,
        K3SlotBuild::Constant(value),
        K3SlotDtype::Bf16,
        1,
        1,
        &[],
    )
}

/// Model-level slots: token embedding, the final norm, the output-side
/// attention-residual scoring vector and the LM head.
pub(crate) fn k3_bookend_slots() -> Vec<K3SlotPlan> {
    vec![
        adopt_matrix(
            "embed",
            K3_VOCAB,
            K3_HIDDEN,
            "model.embed_tokens.weight".to_owned(),
        ),
        adopt_vector("gamma_final", K3_HIDDEN, "model.norm.weight".to_owned()),
        scoring_vector("sw_out", "model.output_attn_res_"),
        adopt_matrix("w_lm", K3_VOCAB, K3_HIDDEN, "lm_head.weight".to_owned()),
    ]
}

/// Every slot of one decoder layer: the two norms, the two residual-stream
/// scoring vectors, the attention block (KDA or MLA) and the MLP block (dense
/// or latent MoE).
pub(crate) fn k3_layer_slots(layer: usize, routed_experts: K3RoutedExperts) -> Vec<K3SlotPlan> {
    let p = format!("model.layers.{layer}.");
    let a = format!("{p}self_attn.");
    let mut slots = vec![
        adopt_vector("gamma_in", K3_HIDDEN, format!("{p}input_layernorm.weight")),
        adopt_vector(
            "gamma_post",
            K3_HIDDEN,
            format!("{p}post_attention_layernorm.weight"),
        ),
        scoring_vector("sw_attn", &format!("{p}self_attention_res_")),
        scoring_vector("sw_mlp", &format!("{p}mlp_res_")),
    ];
    match k3_layer_kind(layer) {
        K3LayerKind::Kda => push_kda_slots(&mut slots, &a),
        K3LayerKind::Mla => push_mla_slots(&mut slots, &a),
    }
    if k3_layer_is_moe(layer) {
        push_moe_slots(&mut slots, &p, routed_experts);
    } else {
        push_dense_slots(&mut slots, &p);
    }
    slots
}

fn push_kda_slots(slots: &mut Vec<K3SlotPlan>, a: &str) {
    slots.push(stack_rows(
        "wbig",
        K3_KDA_BIG_ROWS,
        K3_KDA_BIG_ROWS,
        K3_HIDDEN,
        &[
            format!("{a}q_proj.weight"),
            format!("{a}k_proj.weight"),
            format!("{a}v_proj.weight"),
            format!("{a}g_proj.weight"),
        ],
    ));
    // The rows past b_proj + f_a_proj are the 64-lane pad and stay zero.
    slots.push(stack_rows(
        "wsm",
        K3_KDA_WSM_ROWS,
        K3_HEADS + K3_KDA_GATE_RANK,
        K3_HIDDEN,
        &[format!("{a}b_proj.weight"), format!("{a}f_a_proj.weight")],
    ));
    slots.push(adopt_matrix(
        "w_f_b",
        K3_ATTN_INNER,
        K3_KDA_GATE_RANK,
        format!("{a}f_b_proj.weight"),
    ));
    // The conv weights are [inner, 1, taps] in the checkpoint; the reference
    // drops the singleton channel axis (a reshape) and transposes to
    // [taps, inner], which is what `conv_silu` reads.
    for (name, stem) in [("cw_q", "q"), ("cw_k", "k"), ("cw_v", "v")] {
        slots.push(slot(
            name,
            K3SlotBuild::TransposeF32,
            K3SlotDtype::F32,
            K3_KDA_CONV_KERNEL,
            K3_ATTN_INNER,
            &[format!("{a}{stem}_conv1d.weight")],
        ));
    }
    slots.push(adopt_f32(
        "dt_bias",
        K3_ATTN_INNER,
        1,
        format!("{a}dt_bias"),
    ));
    // The loader already trimmed A_log's checkpoint padding to one value per
    // head, so the resident buffer is the whole slot.
    slots.push(adopt_f32("a_log", K3_HEADS, 1, format!("{a}A_log")));
    slots.push(adopt_f32(
        "gamma_o",
        K3_HEAD_DIM,
        1,
        format!("{a}o_norm.weight"),
    ));
    slots.push(adopt_matrix(
        "w_o",
        K3_HIDDEN,
        K3_O_PROJ_IN,
        format!("{a}o_proj.weight"),
    ));
}

fn push_mla_slots(slots: &mut Vec<K3SlotPlan>, a: &str) {
    slots.push(stack_rows(
        "wfu",
        K3_MLA_FUSED_ROWS,
        K3_MLA_FUSED_ROWS,
        K3_HIDDEN,
        &[
            format!("{a}q_a_proj.weight"),
            format!("{a}kv_a_proj_with_mqa.weight"),
            format!("{a}g_proj.weight"),
        ],
    ));
    slots.push(adopt_vector(
        "gamma_q_a",
        K3_Q_LORA_RANK,
        format!("{a}q_a_layernorm.weight"),
    ));
    slots.push(adopt_vector(
        "gamma_kv_a",
        K3_KV_LORA_RANK,
        format!("{a}kv_a_layernorm.weight"),
    ));
    slots.push(adopt_matrix(
        "w_q_b",
        K3_Q_B_OUT,
        K3_Q_LORA_RANK,
        format!("{a}q_b_proj.weight"),
    ));
    slots.push(adopt_matrix(
        "w_kv_b",
        K3_KV_B_OUT,
        K3_KV_LORA_RANK,
        format!("{a}kv_b_proj.weight"),
    ));
    slots.push(adopt_matrix(
        "w_o",
        K3_HIDDEN,
        K3_O_PROJ_IN,
        format!("{a}o_proj.weight"),
    ));
    slots.push(constant("scale", k3_mla_scale()));
}

fn push_moe_slots(slots: &mut Vec<K3SlotPlan>, p: &str, routed_experts: K3RoutedExperts) {
    let m = format!("{p}block_sparse_moe.");
    let experts = routed_experts.count();
    slots.push(adopt_matrix(
        "w_router",
        experts,
        K3_HIDDEN,
        format!("{m}gate.weight"),
    ));
    slots.push(adopt_f32(
        "bias",
        experts,
        1,
        format!("{m}gate.e_score_correction_bias"),
    ));
    slots.push(constant("rs", K3_ROUTED_SCALING_FACTOR));
    slots.push(adopt_matrix(
        "w_lat_down",
        K3_ROUTED_EXPERT_HIDDEN,
        K3_HIDDEN,
        format!("{m}routed_expert_down_proj.weight"),
    ));
    slots.push(adopt_matrix(
        "w_lat_up",
        K3_HIDDEN,
        K3_ROUTED_EXPERT_HIDDEN,
        format!("{m}routed_expert_up_proj.weight"),
    ));
    slots.push(adopt_vector(
        "gamma_lat",
        K3_ROUTED_EXPERT_HIDDEN,
        format!("{m}routed_expert_norm.weight"),
    ));
    slots.push(stack_rows(
        "wsh",
        2 * K3_SHARED_INTERMEDIATE,
        2 * K3_SHARED_INTERMEDIATE,
        K3_HIDDEN,
        &[
            format!("{m}shared_experts.gate_proj.weight"),
            format!("{m}shared_experts.up_proj.weight"),
        ],
    ));
    slots.push(adopt_matrix(
        "sh_down",
        K3_HIDDEN,
        K3_SHARED_INTERMEDIATE,
        format!("{m}shared_experts.down_proj.weight"),
    ));
}

fn push_dense_slots(slots: &mut Vec<K3SlotPlan>, p: &str) {
    slots.push(stack_rows(
        "wgu",
        2 * K3_DENSE_INTERMEDIATE,
        2 * K3_DENSE_INTERMEDIATE,
        K3_HIDDEN,
        &[
            format!("{p}mlp.gate_proj.weight"),
            format!("{p}mlp.up_proj.weight"),
        ],
    ));
    slots.push(adopt_matrix(
        "w_dn",
        K3_HIDDEN,
        K3_DENSE_INTERMEDIATE,
        format!("{p}mlp.down_proj.weight"),
    ));
}

/// Every slot of a `num_layers`-deep model, bookends first.
pub(crate) fn k3_model_slots(
    num_layers: usize,
    routed_experts: K3RoutedExperts,
) -> Vec<K3SlotPlan> {
    let mut slots = k3_bookend_slots();
    for layer in 0..num_layers {
        slots.extend(k3_layer_slots(layer, routed_experts));
    }
    slots
}

/// One layer's place in the attention-residual stream. A snapshot layer
/// (`layer % attn_res_block_size == 0`) stores the incoming prefix sum as a new
/// block before attention and starts a fresh prefix, so the entry mix sees
/// `nb_in` blocks and the exit mix `nb_mlp`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct K3LayerGeometry {
    pub(crate) layer: usize,
    pub(crate) snapshot: bool,
    /// Blocks visible to the pre-attention mix.
    pub(crate) nb_in: usize,
    /// Blocks visible to the pre-MLP mix.
    pub(crate) nb_mlp: usize,
}

/// The residual-stream geometry of a `num_layers`-deep model, plus the total
/// block count the output mix sees.
pub(crate) fn k3_layer_geometry(num_layers: usize) -> (Vec<K3LayerGeometry>, usize) {
    let mut blocks = 0usize;
    let mut out = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let snapshot = layer.is_multiple_of(K3_ATTN_RES_BLOCK);
        let nb_in = blocks;
        if snapshot {
            blocks += 1;
        }
        out.push(K3LayerGeometry {
            layer,
            snapshot,
            nb_in,
            nb_mlp: blocks,
        });
    }
    (out, blocks)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::config::K3_LAYERS;
    use crate::weights::expected_tensor_contract;

    fn experts224() -> K3RoutedExperts {
        K3RoutedExperts::new(224).unwrap()
    }

    fn source_bytes(name: &str, routed_experts: K3RoutedExperts) -> usize {
        expected_tensor_contract(name, routed_experts)
            .unwrap_or_else(|error| panic!("{name} must have a loader contract: {error:#}"))
            .resident_byte_len()
            .unwrap()
    }

    /// The plan must consume exactly the non-expert tensors the loader makes
    /// resident: 2460 names per rank (the count `weights.rs` pins against the
    /// real checkpoint index), each exactly once, each with a contract.
    #[test]
    fn model_slots_consume_every_resident_non_expert_tensor_once() {
        let slots = k3_model_slots(K3_LAYERS, experts224());
        let mut seen = BTreeSet::new();
        for plan in &slots {
            for name in &plan.sources {
                assert!(seen.insert(name.clone()), "{name} is consumed twice");
                // Resolving the contract proves the name is one the loader
                // plans, with the dtype/shape this slot was sized from.
                source_bytes(name, experts224());
            }
        }
        assert_eq!(seen.len(), 2460);
        // No slot name repeats inside one layer's plan (the builder keys its
        // materialized buffers by slot name).
        for layer in 0..K3_LAYERS {
            let layer_slots = k3_layer_slots(layer, experts224());
            let names = layer_slots
                .iter()
                .map(|plan| plan.slot)
                .collect::<BTreeSet<_>>();
            assert_eq!(names.len(), layer_slots.len(), "layer {layer}");
        }
    }

    /// Slot byte counts must agree with the loader's contract: adopted slots
    /// exactly, stacked slots exactly except for the KDA `wsm` pad.
    #[test]
    fn slot_bytes_agree_with_the_loader_contract() {
        for plan in k3_model_slots(K3_LAYERS, experts224()) {
            let src: usize = plan
                .sources
                .iter()
                .map(|name| source_bytes(name, experts224()))
                .sum();
            match plan.build {
                // A transpose moves the same elements: same byte count as an
                // adopt, which is exactly why it cannot be caught downstream.
                K3SlotBuild::Adopt | K3SlotBuild::TransposeF32 => {
                    assert_eq!(plan.bytes(), src, "{}", plan.slot);
                }
                K3SlotBuild::StackRows { source_rows } => {
                    assert_eq!(src, source_rows * plan.cols * 2, "{}", plan.slot);
                    assert!(source_rows <= plan.rows, "{}", plan.slot);
                    // `wsm` is the one padded slot: b_proj + f_a_proj rows
                    // rounded up to 64 lanes.
                    assert_eq!(source_rows < plan.rows, plan.slot == "wsm", "{}", plan.slot);
                }
                // gamma [H] + proj [1, H] bf16 fold into one f32 [H].
                K3SlotBuild::FoldNormIntoProj => {
                    assert_eq!(plan.sources.len(), 2);
                    assert_eq!(src, 2 * K3_HIDDEN * 2);
                    assert_eq!(plan.bytes(), K3_HIDDEN * 4);
                }
                K3SlotBuild::Constant(_) => {
                    assert!(plan.sources.is_empty());
                    assert_eq!(plan.bytes(), 2);
                }
            }
        }
    }

    #[test]
    fn layer_slots_route_by_attention_kind_and_mlp_kind() {
        let names = |layer: usize| {
            k3_layer_slots(layer, experts224())
                .into_iter()
                .map(|plan| plan.slot)
                .collect::<BTreeSet<_>>()
        };
        // Layer 0: KDA attention, the model's only dense MLP.
        let l0 = names(0);
        assert!(l0.contains("wbig") && l0.contains("cw_v") && l0.contains("a_log"));
        assert!(l0.contains("wgu") && l0.contains("w_dn"));
        assert!(!l0.contains("wfu") && !l0.contains("w_router"));
        // Layer 3: MLA attention + latent MoE.
        let l3 = names(3);
        assert!(l3.contains("wfu") && l3.contains("w_kv_b") && l3.contains("scale"));
        assert!(l3.contains("w_router") && l3.contains("wsh") && l3.contains("gamma_lat"));
        assert!(!l3.contains("wbig") && !l3.contains("wgu"));
        // Layer 1: KDA attention + MoE.
        let l1 = names(1);
        assert!(l1.contains("wbig") && l1.contains("w_router"));
        // Both kinds own the shared output projection and the two scoring taps.
        for layer in [0usize, 1, 3, 92] {
            let n = names(layer);
            assert!(n.contains("w_o") && n.contains("sw_attn") && n.contains("sw_mlp"));
        }
    }

    #[test]
    fn fused_slots_stack_their_sources_in_reference_order() {
        let by_slot = |layer: usize, want: &str| {
            k3_layer_slots(layer, experts224())
                .into_iter()
                .find(|plan| plan.slot == want)
                .unwrap()
        };
        assert_eq!(
            by_slot(0, "wbig").sources,
            [
                "model.layers.0.self_attn.q_proj.weight",
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
                "model.layers.0.self_attn.g_proj.weight",
            ]
        );
        assert_eq!(
            by_slot(3, "wfu").sources,
            [
                "model.layers.3.self_attn.q_a_proj.weight",
                "model.layers.3.self_attn.kv_a_proj_with_mqa.weight",
                "model.layers.3.self_attn.g_proj.weight",
            ]
        );
        assert_eq!(by_slot(3, "wfu").rows, 1536 + 576 + 12_288);
        assert_eq!(
            by_slot(0, "wsm").sources,
            [
                "model.layers.0.self_attn.b_proj.weight",
                "model.layers.0.self_attn.f_a_proj.weight",
            ]
        );
        assert_eq!(K3_KDA_WSM_ROWS, 256);
        // The conv taps land transposed: checkpoint [inner, 1, taps] becomes
        // the [taps, inner] `conv_silu` reads.
        for name in ["cw_q", "cw_k", "cw_v"] {
            let cw = by_slot(0, name);
            assert_eq!(cw.build, K3SlotBuild::TransposeF32);
            assert_eq!((cw.rows, cw.cols), (K3_KDA_CONV_KERNEL, K3_ATTN_INNER));
        }
        assert_eq!(
            by_slot(0, "sw_attn").sources,
            [
                "model.layers.0.self_attention_res_norm.weight",
                "model.layers.0.self_attention_res_proj.weight",
            ]
        );
        assert_eq!(
            by_slot(1, "wsh").sources,
            [
                "model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight",
                "model.layers.1.block_sparse_moe.shared_experts.up_proj.weight",
            ]
        );
    }

    /// The router is the one slot whose width tracks the checkpoint's expert
    /// count; everything else is identical across the two published towers.
    #[test]
    fn only_the_router_tracks_the_expert_count() {
        let width_free_bytes = |count: usize| {
            let experts = K3RoutedExperts::new(count).unwrap();
            let slots = k3_layer_slots(1, experts);
            let router = slots.iter().find(|plan| plan.slot == "w_router").unwrap();
            assert_eq!(router.rows, count);
            let bias = slots.iter().find(|plan| plan.slot == "bias").unwrap();
            assert_eq!(bias.rows, count);
            slots
                .iter()
                .filter(|plan| plan.slot != "w_router" && plan.slot != "bias")
                .map(K3SlotPlan::bytes)
                .sum::<usize>()
        };
        assert_eq!(width_free_bytes(224), width_free_bytes(896));
        // Layer 1 is KDA attention + latent MoE: everything but the router and
        // its bias is the same buffer set in both published towers.
        assert_eq!(width_free_bytes(224), 1_255_354_242);
    }

    #[test]
    fn residual_geometry_matches_the_reference_snapshot_rule() {
        let (geom, blocks) = k3_layer_geometry(K3_LAYERS);
        assert_eq!(blocks, 8);
        assert_eq!(geom.len(), K3_LAYERS);
        assert_eq!(
            geom[0],
            K3LayerGeometry {
                layer: 0,
                snapshot: true,
                nb_in: 0,
                nb_mlp: 1
            }
        );
        assert_eq!(
            geom[11],
            K3LayerGeometry {
                layer: 11,
                snapshot: false,
                nb_in: 1,
                nb_mlp: 1
            }
        );
        assert_eq!(
            geom[12],
            K3LayerGeometry {
                layer: 12,
                snapshot: true,
                nb_in: 1,
                nb_mlp: 2
            }
        );
        assert_eq!(geom[92].nb_mlp, 8);
        // Truncation keeps the prefix geometry it would have in the full model.
        let (short, short_blocks) = k3_layer_geometry(4);
        assert_eq!(short_blocks, 1);
        assert_eq!(short.as_slice(), &geom[..4]);
    }
}
