//! Qwen3.8-Flash-Next QSA token-indexer block scoring.
//!
//! Turns the fused `index_qk_proj` output into one score per (query row,
//! visible micro-block); selection — top-k plus the unconditional tail — is a
//! later stage, deliberately not here. Frozen semantics live in
//! `csrc/qwen4/qwen4_qsa_indexer_score.cu` (transformers 5.16.1 `qwen4_exp`
//! `Qwen4ExpTextQSAIndexer.forward`); the kernel is compiled for one geometry
//! and this wrapper sizes its buffers from it. The checkpoint's own indexer
//! fields are checked where the model crate reads its config, once that caller
//! lands — this stage takes only `rms_eps` from the caller.
//!
//! The MoE block below is the device half of the committed MoE golden's
//! consumer — the host half is `pegainfer-qwen4/tests/moe_block_replay.rs`.
//! It owns only what has no production fit: the routing kernel and the
//! shared expert's per-token gate combine; the router GEMM, the shared
//! branch's projections and its `silu(gate)·up` go through the same ops every
//! other line uses. Frozen semantics: `Qwen4ExpTextSparseMoeBlock` in
//! `csrc/qwen4/qwen4_moe_route.cu`.

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
use crate::tensor::DeviceVec;
use crate::tensor::HiddenStates;
use crate::tensor::active_cu_stream;

/// The geometry the kernel is compiled for, used to size this wrapper's buffer
/// checks. The kernel reads with the same numbers: four query heads and one
/// shared key head at 128 (a 640-wide fused row), four tokens per micro-block,
/// and 64 rotated dims — also the shared attention table's row stride.
const FUSED_WIDTH: usize = 640;
const HEAD_DIM: usize = 128;
const ROTARY_DIM: usize = 64;
const COMPRESS_RATIO: usize = 4;

/// Score every complete micro-block of `fused_qk` for every query row.
///
/// `fused_qk` is the whole fused `index_qk_proj` output, one row per token:
/// `FUSED_WIDTH` columns, the query heads first and the raw key head last.
/// The key array must start at token 0 — the micro-block grid starts at the
/// first visible token and this kernel reads it as token 0 (see the kernel's
/// grid contract).
///
/// `rows[j]` is each query row's position (the RoPE position and the
/// `fused_qk` row its query heads take) and `visible_len[j]` how many tokens
/// it can see, counted from token 0 and at most `rows[j] + 1`, which is what
/// causal text passes. Row `j` gets `visible_len[j] / COMPRESS_RATIO` scores at
/// stride `score_cols`;
/// entries past that are left untouched, as the reference leaves them NaN — a
/// caller that wants to tell an unwritten entry from a real one has to poison
/// the buffer itself (the parity gate pre-fills a sentinel). Both arrays are
/// checked on the device: a row outside the key array, or a visible length past
/// the row's own token, traps, so a broken caller fails visibly instead of
/// getting a plausible score for blocks it should not see.
///
/// `pooled_keys` is the workspace: `tokens / COMPRESS_RATIO` rows of
/// `HEAD_DIM`, reused across calls.
///
/// `cos`/`sin` are the shared attention RoPE tables (`pegainfer_core::rope`,
/// `[max_pos, ROTARY_DIM]`, position-major): the indexer rotates with the same
/// position embeddings the attention uses, so this stage computes no trig. They
/// must be built with `rotary_dim` and `frequency_dim` both `ROTARY_DIM`, the
/// frequencies the reference's partial RoPE uses, and cover every position this
/// call can read, `0..tokens`.
pub fn qsa_indexer_block_scores_into(
    ctx: &DeviceContext,
    fused_qk: &CudaSlice<bf16>,
    q_norm_weight: &CudaSlice<bf16>,
    k_norm_weight: &CudaSlice<bf16>,
    cos: &DeviceVec,
    sin: &DeviceVec,
    rows: &CudaSlice<i32>,
    visible_len: &CudaSlice<i32>,
    tokens: usize,
    scores: &mut CudaSlice<f32>,
    score_cols: usize,
    pooled_keys: &mut CudaSlice<bf16>,
    rms_eps: f32,
) -> Result<()> {
    // NaN fails the comparison, as it does in the kernel's own guard.
    ensure!(
        rms_eps >= 0.0,
        "QSA indexer rms_norm_eps {rms_eps} must be non-negative"
    );
    let blocks = tokens / COMPRESS_RATIO;
    ensure!(
        score_cols >= blocks,
        "QSA indexer score_cols {score_cols} cannot hold {blocks} blocks"
    );
    let query_rows = rows.len();
    ensure!(
        visible_len.len() >= query_rows,
        "QSA indexer visible_len {} shorter than the {query_rows} query rows",
        visible_len.len()
    );
    ensure!(
        fused_qk.len() >= tokens.saturating_mul(FUSED_WIDTH),
        "QSA indexer fused_qk holds {} bf16, needs {} for {tokens} tokens",
        fused_qk.len(),
        tokens.saturating_mul(FUSED_WIDTH)
    );
    ensure!(
        q_norm_weight.len() >= HEAD_DIM && k_norm_weight.len() >= HEAD_DIM,
        "QSA indexer norm weights must hold {HEAD_DIM} bf16"
    );
    let table_len = tokens.saturating_mul(ROTARY_DIM);
    ensure!(
        cos.len >= table_len && sin.len >= table_len,
        "QSA indexer cos/sin tables hold {}/{} bf16, need {table_len} for {tokens} positions",
        cos.len,
        sin.len
    );
    ensure!(
        scores.len() >= query_rows.saturating_mul(score_cols),
        "QSA indexer scores holds {} f32, needs {} for {query_rows} x {score_cols}",
        scores.len(),
        query_rows.saturating_mul(score_cols)
    );
    ensure!(
        pooled_keys.len() >= blocks.saturating_mul(HEAD_DIM),
        "QSA indexer pooled_keys workspace holds {} bf16, needs {}",
        pooled_keys.len(),
        blocks.saturating_mul(HEAD_DIM)
    );
    if query_rows == 0 || tokens == 0 {
        return Ok(());
    }
    // The kernel takes extents as i32; checked before any pointer is formed.
    let query_rows = checked_i32(query_rows, "QSA indexer query rows")?;
    let tokens = checked_i32(tokens, "QSA indexer tokens")?;
    let score_cols = checked_i32(score_cols, "QSA indexer score columns")?;

    let (fused_ptr, _fused_guard) = fused_qk.device_ptr(&ctx.stream);
    let (q_norm_ptr, _q_norm_guard) = q_norm_weight.device_ptr(&ctx.stream);
    let (k_norm_ptr, _k_norm_guard) = k_norm_weight.device_ptr(&ctx.stream);
    let (cos_ptr, _cos_guard) = cos.data.device_ptr(&ctx.stream);
    let (sin_ptr, _sin_guard) = sin.data.device_ptr(&ctx.stream);
    let (rows_ptr, _rows_guard) = rows.device_ptr(&ctx.stream);
    let (visible_ptr, _visible_guard) = visible_len.device_ptr(&ctx.stream);
    let (scores_ptr, _scores_guard) = scores.device_ptr_mut(&ctx.stream);
    let (pooled_ptr, _pooled_guard) = pooled_keys.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::qwen4_qsa_indexer_block_scores_cuda(
            fused_ptr as *const ffi::Half,
            q_norm_ptr as *const ffi::Half,
            k_norm_ptr as *const ffi::Half,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            rows_ptr as *const i32,
            visible_ptr as *const i32,
            scores_ptr as *mut f32,
            pooled_ptr as *mut ffi::Half,
            query_rows,
            tokens,
            score_cols,
            rms_eps,
            active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| anyhow::anyhow!("QSA indexer block score launch failed: {err}"))
}

/// The checkpoint's MoE block geometry. Frozen with the line: the routing
/// kernel is compiled for the expert count and top-k, and the shared branch's
/// shapes are asserted against these widths before any launch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen4MoeGeometry {
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

impl Qwen4MoeGeometry {
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
             csrc/qwen4/qwen4_moe_route.cu was compiled for (512 experts, top-10)"
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
pub struct Qwen4RouteSemantics {
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

impl Qwen4RouteSemantics {
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
pub fn qwen4_moe_route_into(
    ctx: &DeviceContext,
    geometry: &Qwen4MoeGeometry,
    semantics: &Qwen4RouteSemantics,
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
        ffi::qwen4_moe_route_cuda(
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

/// Caller-owned scratch for [`qwen4_moe_shared_branch_into`]: the two
/// projections and the activated intermediate, one `[tokens,
/// shared_intermediate]` buffer each.
pub struct Qwen4SharedBranchScratch {
    gate: HiddenStates,
    up: HiddenStates,
    intermediate: HiddenStates,
}

impl Qwen4SharedBranchScratch {
    pub fn new(ctx: &DeviceContext, geometry: &Qwen4MoeGeometry, tokens: usize) -> Result<Self> {
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
pub fn qwen4_moe_shared_branch_into(
    ctx: &DeviceContext,
    geometry: &Qwen4MoeGeometry,
    hidden: &HiddenStates,
    gate_proj: &DeviceMatrix,
    up_proj: &DeviceMatrix,
    down_proj: &DeviceMatrix,
    scratch: &mut Qwen4SharedBranchScratch,
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
pub fn qwen4_moe_combine_into(
    ctx: &DeviceContext,
    geometry: &Qwen4MoeGeometry,
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
        ffi::qwen4_moe_shared_combine_cuda(
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
    use super::Qwen4MoeGeometry;
    use super::Qwen4RouteSemantics;

    /// The geometry is the hand-off with the kernel's compiled constants, so a
    /// mismatch has to be refused before a pointer is formed. `hidden` and
    /// `shared_intermediate` are not compiled into the route kernel — the
    /// wrappers hold them against the caller's shapes — but a zero width must
    /// still fail here, device-free.
    #[test]
    fn only_the_compiled_geometry_passes() {
        let compiled = Qwen4MoeGeometry::qwen38_flash_next();
        compiled.validate().expect("the frozen geometry");
        for (field, mutated) in [
            (
                "experts",
                Qwen4MoeGeometry {
                    experts: 128,
                    ..compiled
                },
            ),
            (
                "top_k",
                Qwen4MoeGeometry {
                    top_k: 8,
                    ..compiled
                },
            ),
            (
                "hidden",
                Qwen4MoeGeometry {
                    hidden: 0,
                    ..compiled
                },
            ),
            (
                "shared_intermediate",
                Qwen4MoeGeometry {
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
            let semantics = Qwen4RouteSemantics::norm_topk_prob(renorm);
            assert_eq!(semantics.renorm, renorm);
            assert!(!semantics.softmax_bf16);
        }
    }
}
