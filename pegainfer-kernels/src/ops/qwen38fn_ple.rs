//! Qwen3.8-Flash-Next PLE block (per-layer embedding), device side.
//!
//! The block one decoder layer injects the hashed n-gram embedding through. Its
//! inputs are the two recorded projections and the residual stream; its output
//! is the same `hc_count * hidden_size` wide stream. The n-gram table gather
//! that produces those projections is not here — this starts where the golden
//! starts, at the projections' outputs.
//!
//! Frozen semantics live in `csrc/qwen38fn/qwen38fn_ple_block.cu`
//! (transformers 5.16.1 `qwen4_exp` `Qwen4ExpTextPLELayer.forward`), including
//! every bf16 rounding point of the reference: the gate's *double* rounding
//! around the `1 / sqrt(hidden_size)` divide, the signed sqrt's `clamp_min(1e-6)`
//! floor, and the conv's rounding before its silu. The geometry this wrapper
//! accepts is the checkpoint's own, and every field is held against the compiled
//! constants before a launch.
//!
//! The three grouped `(1 + w)` RMSNorms — `norm_key` and `norm_query` over the
//! two input streams, `norm_conv` over the gated one — are **not** new kernels.
//! Each is four calls to the existing batched `(1 + w)` entry point,
//! [`ffi::rms_norm_batched_offset_cuda`], one per branch: it normalizes
//! `seq_len` contiguous rows of `hidden_dim`, and this stream's branches are
//! interleaved row-major, so the first launch below splits each stream into
//! branch-major scratch and the norms then run on contiguous groups. `ops::norm`
//! wraps that entry point for a whole `HiddenStates` owner and cannot express a
//! branch offset, which is why the calls are made here against the same FFI it
//! calls. A single-launch grouped entry point is #1158; if and when it lands,
//! these twelve calls collapse into three and the branch-major split goes with
//! them.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::sys::CUstream;
use half::bf16;

use crate::ffi;
use crate::ops::checked_i32;
use crate::tensor::DeviceContext;
use crate::tensor::active_cu_stream;

/// The checkpoint's PLE block geometry. Frozen with the line: `pegainfer-qwen4`
/// serves one `qwen4_exp` text geometry, and the kernels are compiled for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PleBlockGeometry {
    /// `hidden_size`, which the block's `ple_embed_dim` equals: one branch's
    /// width, and the group a norm reduces over.
    pub hidden: usize,
    /// `hc_count`: hyper-connection branches in the stream.
    pub hc_count: usize,
    /// `ple_conv_kernel_size`: taps of the depthwise conv.
    pub conv_kernel: usize,
    /// `ngram_size`: the conv's dilation, so its state is `(conv_kernel - 1) *
    /// dilation` rows and its receptive field one more.
    pub dilation: usize,
    /// `rms_norm_eps`, shared by all three `(1 + w)` norms.
    pub rms_eps: f32,
}

impl PleBlockGeometry {
    /// The frozen Qwen3.8-Flash-Next text config's PLE block.
    pub fn qwen38_flash_next() -> Self {
        Self {
            hidden: 2560,
            hc_count: 4,
            conv_kernel: 4,
            dilation: 3,
            rms_eps: 1.0e-6,
        }
    }

    /// Columns of the stream the block reads and writes.
    pub fn width(&self) -> usize {
        self.hc_count * self.hidden
    }

    /// `sqrt(hidden_size)` in the f32 the reference's opmath divides by.
    ///
    /// Computed the way the reference does — an f64 square root narrowed to f32 —
    /// and passed to the kernel rather than compiled into it, because the gate
    /// *divides* by this: multiplying by its reciprocal rounds differently, and
    /// the difference lands on every gate.
    pub fn sqrt_hidden(&self) -> f32 {
        (self.hidden as f64).sqrt() as f32
    }

    /// Rows the conv reads before the sequence starts: its zero initial state.
    pub fn conv_state(&self) -> usize {
        self.conv_kernel.saturating_sub(1) * self.dilation
    }

    /// Scratch [`ple_block_into`] needs for `seq_len` rows, in bf16 elements.
    ///
    /// Six regions of `seq_len * width()`, one per intermediate: the two
    /// branch-major splits, the two grouped norms over them, the gated stream
    /// and the conv's normed input. Every region is a whole stream, because each
    /// is written by one launch and read by the next.
    pub fn workspace_len(&self, seq_len: usize) -> usize {
        6 * seq_len * self.width()
    }

    /// Refuse a geometry the kernels were not compiled for. Called by
    /// [`ple_block_into`] before it forms a pointer, and public so a caller
    /// deriving geometry from a checkpoint's config can check it without a
    /// launch.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.hidden == 2560 && self.hc_count == 4 && self.conv_kernel == 4,
            "PLE block geometry {self:?} is not the geometry \
             csrc/qwen38fn/qwen38fn_ple_block.cu was compiled for"
        );
        // The dilation is a runtime argument (the state it implies is a bound
        // inside the kernel, not a buffer), but only `ngram_size` is the frozen
        // semantics: a different one convolves the wrong rows and still produces
        // a plausible stream.
        ensure!(
            self.dilation == 3,
            "PLE block dilation {} is not the compiled ngram_size 3",
            self.dilation
        );
        // Bit equality, not a tolerance: the norms' epsilon enters an rsqrt, so
        // a drifted value changes every element by less than the comparison can
        // see and more than the block should tolerate.
        ensure!(
            self.rms_eps.to_bits() == 1.0e-6f32.to_bits(),
            "PLE block rms_norm_eps {} is not the compiled 1e-6",
            self.rms_eps
        );
        Ok(())
    }
}

/// Which arithmetic the block computes.
///
/// [`Reference`](PleBlockVariant::Reference) is the frozen
/// `Qwen4ExpTextPLELayer` semantics and the only variant
/// [`ple_block_into`] passes. The other three are the parity gate's negative
/// controls, on the device side: each turns exactly one knob and each must move
/// the output past the gate's bound, which is what proves the *device* path is
/// pinned and not only the host oracle that mirrors it.
///
/// They live here rather than in a second copy of the kernels because a control
/// that runs different code from the thing it controls proves nothing: the same
/// launches, the same buffers and the same entry points run with one scalar
/// changed, so a control that passed would mean the gate cannot see that fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PleBlockVariant {
    /// The frozen semantics.
    Reference,
    /// Control: drop the sign restore after the sqrt, so a negative gate reaches
    /// the sigmoid as its magnitude and inverts that branch's contribution.
    UnsignedSqrtGate,
    /// Control: one dot over the whole `hc_count * hidden` stream instead of one
    /// per branch, so all four branches share a gate scaled by four.
    FullStreamGate,
    /// Control: conv dilation 1 instead of `ngram_size`, so three of the four
    /// taps read rows the reference never looks at.
    Dilation1Conv,
}

impl PleBlockVariant {
    /// Channels one gate's dot reduces over.
    fn gate_span(self, geometry: &PleBlockGeometry) -> usize {
        match self {
            Self::FullStreamGate => geometry.width(),
            _ => geometry.hidden,
        }
    }

    /// Whether the sqrt's sign is restored.
    fn signed_sqrt(self) -> i32 {
        i32::from(self != Self::UnsignedSqrtGate)
    }

    /// The conv's dilation.
    fn dilation(self, geometry: &PleBlockGeometry) -> usize {
        match self {
            Self::Dilation1Conv => 1,
            _ => geometry.dilation,
        }
    }
}

/// Run one PLE block over `seq_len` rows.
///
/// `hidden`, `key` and `out` are row-major `[seq_len, width()]` — `hidden` is
/// the residual stream the query norm reads, `key` the projection the key norm
/// reads. `value` is `[seq_len, hidden]` and is broadcast over the branches. The
/// three norm weights are one per channel of the stream, `[width()]`, and
/// `conv_weight` is the depthwise `[width(), 1, conv_kernel]` flattened.
///
/// `workspace` is caller-owned scratch of [`PleBlockGeometry::workspace_len`]
/// elements; see that method for the layout. Nothing is allocated here, so the
/// call is capturable.
///
/// This is the frozen semantics; see [`ple_block_variant_into`] for the gate's
/// control surface.
pub fn ple_block_into(
    ctx: &DeviceContext,
    geometry: &PleBlockGeometry,
    hidden: &CudaSlice<bf16>,
    key: &CudaSlice<bf16>,
    value: &CudaSlice<bf16>,
    norm_key: &CudaSlice<bf16>,
    norm_query: &CudaSlice<bf16>,
    norm_conv: &CudaSlice<bf16>,
    conv_weight: &CudaSlice<bf16>,
    seq_len: usize,
    out: &mut CudaSlice<bf16>,
    workspace: &mut CudaSlice<bf16>,
) -> Result<()> {
    ple_block_variant_into(
        ctx,
        geometry,
        PleBlockVariant::Reference,
        hidden,
        key,
        value,
        norm_key,
        norm_query,
        norm_conv,
        conv_weight,
        seq_len,
        out,
        workspace,
    )
}

/// [`ple_block_into`] with one knob turned. Only the parity gate's negative
/// controls call this with a variant other than
/// [`Reference`](PleBlockVariant::Reference); see that type for why the knobs
/// are here and not in a second copy of the kernels.
pub fn ple_block_variant_into(
    ctx: &DeviceContext,
    geometry: &PleBlockGeometry,
    variant: PleBlockVariant,
    hidden: &CudaSlice<bf16>,
    key: &CudaSlice<bf16>,
    value: &CudaSlice<bf16>,
    norm_key: &CudaSlice<bf16>,
    norm_query: &CudaSlice<bf16>,
    norm_conv: &CudaSlice<bf16>,
    conv_weight: &CudaSlice<bf16>,
    seq_len: usize,
    out: &mut CudaSlice<bf16>,
    workspace: &mut CudaSlice<bf16>,
) -> Result<()> {
    geometry.validate()?;
    let width = geometry.width();
    let group = geometry.hidden;
    let stream_elements = seq_len
        .checked_mul(width)
        .ok_or_else(|| anyhow!("PLE block extent overflow at {seq_len} rows"))?;
    ensure!(
        hidden.len() >= stream_elements && key.len() >= stream_elements,
        "PLE block inputs hold {} and {} bf16, need {stream_elements} each for {seq_len} rows of \
         {width}",
        hidden.len(),
        key.len()
    );
    ensure!(
        value.len() >= seq_len.saturating_mul(group),
        "PLE block value holds {} bf16, needs {} for {seq_len} rows of {group}",
        value.len(),
        seq_len.saturating_mul(group)
    );
    ensure!(
        norm_key.len() >= width && norm_query.len() >= width && norm_conv.len() >= width,
        "PLE block norm weights must hold {width} bf16 each, got {}, {} and {}",
        norm_key.len(),
        norm_query.len(),
        norm_conv.len()
    );
    ensure!(
        conv_weight.len() >= width.saturating_mul(geometry.conv_kernel),
        "PLE block conv weight holds {} bf16, needs {} for {width} channels of {} taps",
        conv_weight.len(),
        width.saturating_mul(geometry.conv_kernel),
        geometry.conv_kernel
    );
    ensure!(
        out.len() >= stream_elements,
        "PLE block out holds {} bf16, needs {stream_elements}",
        out.len()
    );
    let needed = geometry.workspace_len(seq_len);
    ensure!(
        workspace.len() >= needed,
        "PLE block workspace holds {} bf16, needs {needed} for {seq_len} rows",
        workspace.len()
    );
    if seq_len == 0 {
        return Ok(());
    }
    // The entry points take extents as i32; checked before any pointer is formed.
    let rows = checked_i32(seq_len, "PLE block rows")?;
    let width_i = checked_i32(width, "PLE block stream width")?;
    let group_i = checked_i32(group, "PLE block group width")?;
    let branches_i = checked_i32(geometry.hc_count, "PLE block branch count")?;
    let conv_kernel_i = checked_i32(geometry.conv_kernel, "PLE block conv kernel")?;
    let dilation = variant.dilation(geometry);
    let dilation_i = checked_i32(dilation, "PLE block conv dilation")?;
    let gate_span_i = checked_i32(variant.gate_span(geometry), "PLE block gate span")?;

    let (hidden_ptr, _hidden_guard) = hidden.device_ptr(&ctx.stream);
    let (key_ptr, _key_guard) = key.device_ptr(&ctx.stream);
    let (value_ptr, _value_guard) = value.device_ptr(&ctx.stream);
    let (norm_key_ptr, _norm_key_guard) = norm_key.device_ptr(&ctx.stream);
    let (norm_query_ptr, _norm_query_guard) = norm_query.device_ptr(&ctx.stream);
    let (norm_conv_ptr, _norm_conv_guard) = norm_conv.device_ptr(&ctx.stream);
    let (conv_weight_ptr, _conv_weight_guard) = conv_weight.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let (workspace_ptr, _workspace_guard) = workspace.device_ptr_mut(&ctx.stream);
    let stream = active_cu_stream(ctx);

    // Workspace regions, one whole stream each, addressed in bytes.
    let element_bytes = size_of::<bf16>() as u64;
    let region = stream_elements as u64 * element_bytes;
    let query_split = workspace_ptr;
    let key_split = workspace_ptr + region;
    let query_normed = workspace_ptr + 2 * region;
    let key_normed = workspace_ptr + 3 * region;
    let gated = workspace_ptr + 4 * region;
    let conv_in = workspace_ptr + 5 * region;

    // 1. Row-major stream -> branch-major, so the grouped norms see contiguous
    //    groups of `hidden`.
    split_branches(
        hidden_ptr,
        query_split,
        rows,
        width_i,
        group_i,
        branches_i,
        stream,
        "query",
    )?;
    split_branches(
        key_ptr, key_split, rows, width_i, group_i, branches_i, stream, "key",
    )?;

    // 2. The two input norms, one launch per branch. The weight for branch `b`
    //    is that branch's slice of the stream-wide vector.
    let branch_stride = (seq_len * group) as u64 * element_bytes;
    let weight_stride = group as u64 * element_bytes;
    for branch in 0..geometry.hc_count as u64 {
        grouped_norm(
            key_split + branch * branch_stride,
            norm_key_ptr + branch * weight_stride,
            key_normed + branch * branch_stride,
            group_i,
            rows,
            geometry.rms_eps,
            stream,
        );
        grouped_norm(
            query_split + branch * branch_stride,
            norm_query_ptr + branch * weight_stride,
            query_normed + branch * branch_stride,
            group_i,
            rows,
            geometry.rms_eps,
            stream,
        );
    }

    // 3. Per-branch gate dot, signed sqrt, sigmoid, and the value broadcast.
    let status = unsafe {
        ffi::qwen38fn_ple_gate_cuda(
            key_normed as *const ffi::Half,
            query_normed as *const ffi::Half,
            value_ptr as *const ffi::Half,
            gated as *mut ffi::Half,
            rows,
            group_i,
            branches_i,
            gate_span_i,
            geometry.sqrt_hidden(),
            variant.signed_sqrt(),
            stream,
        )
    };
    status
        .result()
        .map_err(|err| anyhow!("PLE block gate launch failed: {err}"))?;

    // 4. The conv's own grouped norm over the gated stream.
    for branch in 0..geometry.hc_count as u64 {
        grouped_norm(
            gated + branch * branch_stride,
            norm_conv_ptr + branch * weight_stride,
            conv_in + branch * branch_stride,
            group_i,
            rows,
            geometry.rms_eps,
            stream,
        );
    }

    // 5. Dilated causal depthwise conv, silu, and the residual add back onto the
    //    gated stream, landing row-major in the caller's output.
    let status = unsafe {
        ffi::qwen38fn_ple_conv_residual_cuda(
            conv_in as *const ffi::Half,
            gated as *const ffi::Half,
            conv_weight_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            rows,
            group_i,
            branches_i,
            conv_kernel_i,
            dilation_i,
            stream,
        )
    };
    status
        .result()
        .map_err(|err| anyhow!("PLE block conv launch failed: {err}"))?;
    Ok(())
}

fn split_branches(
    row_major: u64,
    branch_major: u64,
    rows: i32,
    width: i32,
    group: i32,
    branches: i32,
    stream: CUstream,
    what: &str,
) -> Result<()> {
    let status = unsafe {
        ffi::qwen38fn_ple_split_branches_cuda(
            row_major as *const ffi::Half,
            branch_major as *mut ffi::Half,
            rows,
            width,
            group,
            branches,
            stream,
        )
    };
    status
        .result()
        .map_err(|err| anyhow!("PLE block {what} branch split failed: {err}"))
}

/// One branch of a grouped `(1 + w)` RMSNorm, through the same entry point
/// `ops::norm::rms_norm_batch_offset_into` wraps.
///
/// That wrapper returns no status because the C++ side is `void`, and this call
/// inherits that: a failure here surfaces as the next launch's sticky error,
/// which is why every launch in this module checks its own result.
fn grouped_norm(
    x: u64,
    weight: u64,
    out: u64,
    hidden_dim: i32,
    seq_len: i32,
    eps: f32,
    stream: CUstream,
) {
    unsafe {
        ffi::rms_norm_batched_offset_cuda(
            x as *const ffi::Half,
            weight as *const ffi::Half,
            out as *mut ffi::Half,
            hidden_dim,
            seq_len,
            eps,
            stream,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::PleBlockGeometry;
    use super::PleBlockVariant;

    /// The geometry is the one hand-off this block has with the kernels'
    /// compiled constants, so a mismatch has to be refused before a pointer is
    /// formed. Device-free: no launch is involved.
    #[test]
    fn only_the_compiled_geometry_passes() {
        let compiled = PleBlockGeometry::qwen38_flash_next();
        compiled.validate().expect("the frozen geometry");
        assert_eq!(compiled.width(), 10_240);
        assert_eq!(compiled.conv_state(), 9);
        // `sqrt(2560)` narrowed from f64, the divisor the gate uses.
        // Bit equality: the gate divides by this, so a last-place difference is
        // a different divisor and not a rounding of the same one.
        assert_eq!(
            compiled.sqrt_hidden().to_bits(),
            ((2560f64).sqrt() as f32).to_bits()
        );
        assert_eq!(compiled.workspace_len(2), 6 * 2 * 10_240);
        for (field, mutated) in [
            (
                "hidden",
                PleBlockGeometry {
                    hidden: 1280,
                    ..compiled
                },
            ),
            (
                "hc_count",
                PleBlockGeometry {
                    hc_count: 2,
                    ..compiled
                },
            ),
            (
                "conv_kernel",
                PleBlockGeometry {
                    conv_kernel: 2,
                    ..compiled
                },
            ),
            (
                "dilation",
                PleBlockGeometry {
                    dilation: 1,
                    ..compiled
                },
            ),
            (
                "rms_eps",
                PleBlockGeometry {
                    rms_eps: 1.0e-5,
                    ..compiled
                },
            ),
            (
                // One f32 ulp: the geometry's own bit-equality check, and what a
                // config drifting in the last place hits.
                "rms_eps one ulp",
                PleBlockGeometry {
                    rms_eps: f32::from_bits(1.0e-6f32.to_bits() + 1),
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

    /// Each control turns exactly one knob, and the reference turns none. A
    /// control that changed two would not say which one the gate can see.
    #[test]
    fn every_control_turns_one_knob() {
        let geometry = PleBlockGeometry::qwen38_flash_next();
        let reference = PleBlockVariant::Reference;
        assert_eq!(reference.gate_span(&geometry), geometry.hidden);
        assert_eq!(reference.signed_sqrt(), 1);
        assert_eq!(reference.dilation(&geometry), geometry.dilation);
        for variant in [
            PleBlockVariant::UnsignedSqrtGate,
            PleBlockVariant::FullStreamGate,
            PleBlockVariant::Dilation1Conv,
        ] {
            let knobs = [
                variant.gate_span(&geometry) != reference.gate_span(&geometry),
                variant.signed_sqrt() != reference.signed_sqrt(),
                variant.dilation(&geometry) != reference.dilation(&geometry),
            ];
            assert_eq!(
                knobs.iter().filter(|turned| **turned).count(),
                1,
                "{variant:?} must differ from the reference in exactly one knob, got {knobs:?}"
            );
        }
        assert_eq!(
            PleBlockVariant::FullStreamGate.gate_span(&geometry),
            geometry.width()
        );
        assert_eq!(PleBlockVariant::UnsignedSqrtGate.signed_sqrt(), 0);
        assert_eq!(PleBlockVariant::Dilation1Conv.dilation(&geometry), 1);
    }
}
