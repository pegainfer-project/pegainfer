//! Qwen3.8-Flash-Next MoE block: router routing, the shared-expert branch and
//! the block combine.
//!
//! The device half of the committed MoE golden's consumer — the host half is
//! `pegainfer-qwen4/tests/moe_block_replay.rs`, and
//! `tests/qwen38fn_moe_block_parity.rs` drives this wrapper against the same
//! fixture. Frozen semantics: transformers 5.16.1 `qwen4_exp`
//! `Qwen4ExpTextSparseMoeBlock`; the kernel header
//! (`csrc/qwen38fn/qwen38fn_moe_route.cu`) carries the chain and its rounding
//! points.
//!
//! Composition, deliberately thin: the router GEMM, the shared branch's three
//! GEMMs and its `silu(gate)·up` are the production ops every other line uses
//! (`gemm_into_checked`, `silu_mul_batch_into`); this module owns only what has
//! no production fit — the routing kernel (fp32 softmax over 512 → top-10 →
//! `norm_topk_prob` renorm → bf16 weights; surveyed against gemma4's
//! 128-expert `__expf` router, the glm52/kimi sigmoid routers and FlashInfer's
//! f32 top-k, none of which carries this contract) and the shared expert's
//! scalar-sigmoid-gate combine. The combine is the one near-duplicate: its
//! two-rounding sum is `csrc/shared/elementwise.cu`'s `add_scaled_bf16_kernel`
//! arithmetic, which takes its scale as a single host `f32` and so cannot carry
//! a per-token device gate.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::ops::checked_i32;
use crate::ops::gemm_into_checked;
use crate::ops::silu_mul_batch_into;
use crate::tensor::DeviceContext;
use crate::tensor::DeviceMatrix;
use crate::tensor::HiddenStates;
use crate::tensor::active_cu_stream;

/// The checkpoint's MoE block geometry. Frozen with the line: the routing
/// kernel is compiled for the expert count and top-k, and the shared branch's
/// shapes are asserted against these widths before any launch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38fnMoeGeometry {
    /// `text_config.hidden_size`.
    pub hidden: usize,
    /// `text_config.num_experts`. The routing kernel is compiled for this.
    pub experts: usize,
    /// `text_config.num_experts_per_tok`. The routing kernel is compiled for
    /// this.
    pub top_k: usize,
    /// `text_config.shared_expert_intermediate_size`.
    pub shared_intermediate: usize,
}

impl Qwen38fnMoeGeometry {
    /// The frozen Qwen3.8-Flash-Next text config's MoE block.
    pub fn qwen38_flash_next() -> Self {
        Self {
            hidden: 2560,
            experts: 512,
            top_k: 10,
            shared_intermediate: 640,
        }
    }

    /// Refuse a geometry the kernels were not compiled for. Called by every
    /// wrapper here before it forms a pointer, and public so a caller deriving
    /// geometry from a checkpoint's config can check it without a launch.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.experts == 512 && self.top_k == 10,
            "MoE geometry {self:?} is not the geometry \
             csrc/qwen38fn/qwen38fn_moe_route.cu was compiled for (512 experts, top-10)"
        );
        ensure!(
            self.hidden > 0 && self.shared_intermediate > 0,
            "MoE geometry {self:?} carries a zero width"
        );
        Ok(())
    }
}

/// The routing contract's two semantic switches.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen38fnRouteSemantics {
    /// The checkpoint's `norm_topk_prob`: renormalize the selected weights so
    /// they sum to 1. Absent from `config.json` and defaulting to `true`
    /// upstream — the trap the golden exists for.
    pub renorm: bool,
    /// Carry the softmax probabilities in bf16 instead of the reference's fp32
    /// upcast. **Not a production choice**: this is the port mistake the parity
    /// gate's negative control prices, exposed so that control runs through the
    /// production kernel. Production callers use [`Self::norm_topk_prob`],
    /// which leaves it false.
    pub softmax_bf16: bool,
}

impl Qwen38fnRouteSemantics {
    /// Production semantics: fp32 softmax, renormalization per the checkpoint's
    /// (defaulted) `norm_topk_prob`.
    pub fn norm_topk_prob(renorm: bool) -> Self {
        Self {
            renorm,
            softmax_bf16: false,
        }
    }
}

/// Route one batch of bf16 router logits.
///
/// `logits` is the router GEMM's stored bf16 output, `[seq_len, experts]` —
/// the reference's softmax reads exactly those bits, so the GEMM's rounding is
/// part of the routing contract (the gate's chained check pins it).
/// `weights` receives `[seq_len, top_k]` bf16 renormalized weights and `ids`
/// the matching i32 expert ids, in the canonical order: descending
/// probability, ascending expert index inside a tie.
pub fn qwen38fn_moe_route_into(
    ctx: &DeviceContext,
    geometry: &Qwen38fnMoeGeometry,
    semantics: &Qwen38fnRouteSemantics,
    logits: &HiddenStates,
    weights: &mut CudaSlice<bf16>,
    ids: &mut CudaSlice<i32>,
) -> Result<()> {
    geometry.validate()?;
    ensure!(
        logits.hidden_dim == geometry.experts,
        "MoE route: logits carry {} columns, the geometry has {} experts",
        logits.hidden_dim,
        geometry.experts
    );
    // `HiddenStates`'s dims are labels; the kernel reads the allocation with
    // them as strides, so the backing is what gets checked.
    logits.checked_extent("MoE route logits")?;
    let rows = logits.seq_len;
    let picks = rows.checked_mul(geometry.top_k).ok_or_else(|| {
        anyhow!(
            "MoE route: {rows} tokens x top-{} overflows usize",
            geometry.top_k
        )
    })?;
    ensure!(
        weights.len() >= picks,
        "MoE route: weights holds {} bf16, needs {picks} for {rows} tokens x top-{}",
        weights.len(),
        geometry.top_k
    );
    ensure!(
        ids.len() >= picks,
        "MoE route: ids holds {} i32, needs {picks}",
        ids.len()
    );
    if rows == 0 {
        return Ok(());
    }
    let rows = checked_i32(rows, "MoE route rows")?;
    let experts = checked_i32(geometry.experts, "MoE route experts")?;
    let top_k = checked_i32(geometry.top_k, "MoE route top_k")?;

    let (logits_ptr, _logits_guard) = logits.data.device_ptr(&ctx.stream);
    let (weights_ptr, _weights_guard) = weights.device_ptr_mut(&ctx.stream);
    let (ids_ptr, _ids_guard) = ids.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::qwen38fn_moe_route_cuda(
            logits_ptr as *const ffi::Half,
            weights_ptr as *mut ffi::Half,
            ids_ptr as *mut i32,
            rows,
            experts,
            top_k,
            i32::from(semantics.renorm),
            i32::from(semantics.softmax_bf16),
            active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| anyhow!("Qwen3.8-FN MoE route launch failed: {err}"))
}

/// Caller-owned scratch for [`qwen38fn_moe_shared_branch_into`]: the two
/// projections and the activated intermediate, one `[tokens,
/// shared_intermediate]` buffer each.
pub struct Qwen38fnSharedBranchScratch {
    gate: HiddenStates,
    up: HiddenStates,
    intermediate: HiddenStates,
}

impl Qwen38fnSharedBranchScratch {
    pub fn new(ctx: &DeviceContext, geometry: &Qwen38fnMoeGeometry, tokens: usize) -> Result<Self> {
        geometry.validate()?;
        let alloc = |name: &str| {
            HiddenStates::zeros(ctx, geometry.shared_intermediate, tokens)
                .map_err(|err| anyhow!("{name} scratch: {err}"))
        };
        Ok(Self {
            gate: alloc("shared gate")?,
            up: alloc("shared up")?,
            intermediate: alloc("shared intermediate")?,
        })
    }

    /// The token count this scratch was allocated for.
    pub fn tokens(&self) -> usize {
        self.gate.seq_len
    }
}

/// The shared expert: `down(silu(gate(x)) · up(x))`, composed of the
/// production GEMM and SiLU-mul ops with the reference's bf16 rounding at each
/// boundary (both projections, the activation, the product, the down
/// projection). The learned scalar gate is *not* applied here — the reference
/// captures `shared_out` before it, and the combine below folds it in.
pub fn qwen38fn_moe_shared_branch_into(
    ctx: &DeviceContext,
    geometry: &Qwen38fnMoeGeometry,
    hidden: &HiddenStates,
    gate_proj: &DeviceMatrix,
    up_proj: &DeviceMatrix,
    down_proj: &DeviceMatrix,
    scratch: &mut Qwen38fnSharedBranchScratch,
    shared_out: &mut HiddenStates,
) -> Result<()> {
    geometry.validate()?;
    ensure!(
        hidden.hidden_dim == geometry.hidden,
        "MoE shared branch: hidden carries {} columns, the geometry has {}",
        hidden.hidden_dim,
        geometry.hidden
    );
    for (name, proj, rows, cols) in [
        (
            "gate_proj",
            gate_proj,
            geometry.shared_intermediate,
            geometry.hidden,
        ),
        (
            "up_proj",
            up_proj,
            geometry.shared_intermediate,
            geometry.hidden,
        ),
        (
            "down_proj",
            down_proj,
            geometry.hidden,
            geometry.shared_intermediate,
        ),
    ] {
        ensure!(
            proj.rows == rows && proj.cols == cols,
            "MoE shared branch: {name} is {}x{}, the geometry needs {rows}x{cols}",
            proj.rows,
            proj.cols
        );
    }
    let tokens = hidden.seq_len;
    ensure!(
        scratch.tokens() == tokens,
        "MoE shared branch: scratch holds {} tokens, hidden carries {tokens}",
        scratch.tokens()
    );
    ensure!(
        shared_out.hidden_dim == geometry.hidden && shared_out.seq_len == tokens,
        "MoE shared branch: shared_out is {}x{}, needs {tokens}x{}",
        shared_out.seq_len,
        shared_out.hidden_dim,
        geometry.hidden
    );
    // The GEMM and SiLU ops below assert the labels only, so the two buffers
    // whose shape this wrapper owns are held against their allocations here —
    // a caller-shaped logical size past the backing would otherwise reach the
    // kernels as an out-of-range stride.
    hidden.checked_extent("MoE shared branch hidden")?;
    shared_out.checked_extent("MoE shared branch shared_out")?;

    gemm_into_checked(ctx, gate_proj, hidden, &mut scratch.gate)?;
    gemm_into_checked(ctx, up_proj, hidden, &mut scratch.up)?;
    silu_mul_batch_into(ctx, &scratch.gate, &scratch.up, &mut scratch.intermediate)?;
    gemm_into_checked(ctx, down_proj, &scratch.intermediate, shared_out)
}

/// The block sum: `out = expert_out + bf16(sigmoid(gate_logit)) · shared_out`,
/// every operator bf16-rounded where the reference rounds.
///
/// `gate_logit` is the `[1, hidden]` gate linear's stored bf16 output, one
/// scalar per token. The routed-expert half (`expert_out`) is an input: the
/// fused 512-expert stack is #1107's, and until it lands the golden's recorded
/// activation is what this reads.
pub fn qwen38fn_moe_combine_into(
    ctx: &DeviceContext,
    geometry: &Qwen38fnMoeGeometry,
    gate_logit: &CudaSlice<bf16>,
    expert_out: &HiddenStates,
    shared_out: &HiddenStates,
    out: &mut HiddenStates,
) -> Result<()> {
    geometry.validate()?;
    let tokens = expert_out.seq_len;
    for (name, mat) in [("expert_out", expert_out), ("shared_out", shared_out)] {
        ensure!(
            mat.hidden_dim == geometry.hidden,
            "MoE combine: {name} carries {} columns, the geometry has {}",
            mat.hidden_dim,
            geometry.hidden
        );
    }
    ensure!(
        shared_out.seq_len == tokens && out.seq_len == tokens && out.hidden_dim == geometry.hidden,
        "MoE combine: shape mismatch — expert_out {tokens}x{}, shared_out {}x{}, out {}x{}",
        expert_out.hidden_dim,
        shared_out.seq_len,
        shared_out.hidden_dim,
        out.seq_len,
        out.hidden_dim
    );
    // The kernel strides all three matrices by `hidden`, so what is checked is
    // the allocation behind the labels, not the labels.
    expert_out.checked_extent("MoE combine expert_out")?;
    shared_out.checked_extent("MoE combine shared_out")?;
    out.checked_extent("MoE combine out")?;
    ensure!(
        gate_logit.len() >= tokens,
        "MoE combine: gate_logit holds {} bf16, needs one per token ({tokens})",
        gate_logit.len()
    );
    if tokens == 0 {
        return Ok(());
    }
    let tokens = checked_i32(tokens, "MoE combine tokens")?;
    let hidden = checked_i32(geometry.hidden, "MoE combine hidden")?;

    let (gate_ptr, _gate_guard) = gate_logit.device_ptr(&ctx.stream);
    let (expert_ptr, _expert_guard) = expert_out.data.device_ptr(&ctx.stream);
    let (shared_ptr, _shared_guard) = shared_out.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::qwen38fn_moe_shared_combine_cuda(
            gate_ptr as *const ffi::Half,
            expert_ptr as *const ffi::Half,
            shared_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            tokens,
            hidden,
            active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| anyhow!("Qwen3.8-FN MoE shared combine launch failed: {err}"))
}

#[cfg(test)]
mod tests {
    use super::Qwen38fnMoeGeometry;
    use super::Qwen38fnRouteSemantics;

    /// The geometry is the hand-off with the kernel's compiled constants, so a
    /// mismatch has to be refused before a pointer is formed. `hidden` and
    /// `shared_intermediate` are not compiled into the route kernel — the
    /// wrappers hold them against the caller's shapes — but a zero width must
    /// still fail here, device-free.
    #[test]
    fn only_the_compiled_geometry_passes() {
        let compiled = Qwen38fnMoeGeometry::qwen38_flash_next();
        compiled.validate().expect("the frozen geometry");
        for (field, mutated) in [
            (
                "experts",
                Qwen38fnMoeGeometry {
                    experts: 128,
                    ..compiled
                },
            ),
            (
                "top_k",
                Qwen38fnMoeGeometry {
                    top_k: 8,
                    ..compiled
                },
            ),
            (
                "hidden",
                Qwen38fnMoeGeometry {
                    hidden: 0,
                    ..compiled
                },
            ),
            (
                "shared_intermediate",
                Qwen38fnMoeGeometry {
                    shared_intermediate: 0,
                    ..compiled
                },
            ),
        ] {
            assert!(
                mutated.validate().is_err(),
                "a geometry with another {field} must be refused"
            );
        }
    }

    /// Production semantics never carry the control's bf16 softmax.
    #[test]
    fn production_semantics_upcast_the_softmax() {
        for renorm in [true, false] {
            let semantics = Qwen38fnRouteSemantics::norm_topk_prob(renorm);
            assert_eq!(semantics.renorm, renorm);
            assert!(!semantics.softmax_bf16);
        }
    }
}
