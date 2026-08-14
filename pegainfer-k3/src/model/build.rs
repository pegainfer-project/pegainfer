//! Slot materialization: turn [`K3SlotPlan`]s into device buffers, then hand
//! them out under the plan's slot names.
//!
//! Every buffer here is either the loader's own allocation adopted unchanged
//! (retyped, never copied), a device-to-device row concatenation, or one of the
//! two tiny derived values the reference engine precomputes at init. No math
//! kernel runs at build time.

use std::collections::BTreeMap;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;

use super::plan::K3SlotBuild;
use super::plan::K3SlotDtype;
use super::plan::K3SlotPlan;
use crate::weights::K3RankGpuWeights;
use crate::weights::retype_owned;

enum K3SlotData {
    Bf16(CudaSlice<bf16>),
    F32(CudaSlice<f32>),
}

struct K3SlotBuffer {
    data: K3SlotData,
    rows: usize,
    cols: usize,
}

/// One weight struct's materialized slots, keyed by slot name. The struct is
/// assembled by taking each slot out; [`K3SlotBuffers::ensure_drained`] then
/// proves the plan and the struct agree — a slot the struct never took would
/// otherwise be silently uploaded and dropped.
pub(super) struct K3SlotBuffers {
    slots: BTreeMap<&'static str, K3SlotBuffer>,
    /// Device bytes these slots hold.
    pub(super) bytes: usize,
}

impl K3SlotBuffers {
    pub(super) fn materialize(
        ctx: &DeviceContext,
        weights: &mut K3RankGpuWeights,
        plans: &[K3SlotPlan],
    ) -> Result<Self> {
        let mut slots = BTreeMap::new();
        let mut bytes = 0usize;
        for plan in plans {
            let buffer = materialize_slot(ctx, weights, plan)
                .with_context(|| format!("build K3 weight slot {}", plan.slot))?;
            bytes += plan.bytes();
            ensure!(
                slots.insert(plan.slot, buffer).is_none(),
                "K3 build plan names slot {} twice",
                plan.slot
            );
        }
        Ok(Self { slots, bytes })
    }

    fn take(&mut self, slot: &'static str, rows: usize, cols: usize) -> Result<K3SlotBuffer> {
        let buffer = self
            .slots
            .remove(slot)
            .ok_or_else(|| anyhow::anyhow!("K3 build plan has no slot {slot}"))?;
        ensure!(
            buffer.rows == rows && buffer.cols == cols,
            "K3 slot {slot} is [{}, {}], the model wants [{rows}, {cols}]",
            buffer.rows,
            buffer.cols
        );
        Ok(buffer)
    }

    /// A `[rows, cols]` bf16 matrix in the checkpoint's `[out, in]` order.
    pub(super) fn matrix(
        &mut self,
        slot: &'static str,
        rows: usize,
        cols: usize,
    ) -> Result<DeviceMatrix> {
        let buffer = self.take(slot, rows, cols)?;
        match buffer.data {
            K3SlotData::Bf16(data) => Ok(DeviceMatrix { data, rows, cols }),
            K3SlotData::F32(_) => anyhow::bail!("K3 slot {slot} is f32, the model wants bf16"),
        }
    }

    pub(super) fn vector(&mut self, slot: &'static str, len: usize) -> Result<DeviceVec> {
        let buffer = self.take(slot, len, 1)?;
        match buffer.data {
            K3SlotData::Bf16(data) => Ok(DeviceVec { data, len }),
            K3SlotData::F32(_) => anyhow::bail!("K3 slot {slot} is f32, the model wants bf16"),
        }
    }

    /// An f32 buffer of `rows * cols` elements (`cols == 1` for vectors).
    pub(super) fn f32(
        &mut self,
        slot: &'static str,
        rows: usize,
        cols: usize,
    ) -> Result<CudaSlice<f32>> {
        let buffer = self.take(slot, rows, cols)?;
        match buffer.data {
            K3SlotData::F32(data) => Ok(data),
            K3SlotData::Bf16(_) => anyhow::bail!("K3 slot {slot} is bf16, the model wants f32"),
        }
    }

    pub(super) fn ensure_drained(&self, what: &str) -> Result<()> {
        ensure!(
            self.slots.is_empty(),
            "K3 {what} built {} slots the weight struct never took: {:?}",
            self.slots.len(),
            self.slots.keys().collect::<Vec<_>>()
        );
        Ok(())
    }
}

fn materialize_slot(
    ctx: &DeviceContext,
    weights: &mut K3RankGpuWeights,
    plan: &K3SlotPlan,
) -> Result<K3SlotBuffer> {
    let bytes = plan.bytes();
    let raw = match plan.build {
        K3SlotBuild::Adopt => {
            ensure!(
                plan.sources.len() == 1,
                "K3 adopted slot must have exactly one source, got {}",
                plan.sources.len()
            );
            let raw = weights.take_tensor(&plan.sources[0])?;
            ensure!(
                raw.len() == bytes,
                "K3 tensor {} is {} bytes, slot wants {bytes}",
                plan.sources[0],
                raw.len()
            );
            raw
        }
        K3SlotBuild::StackRows { source_rows } => {
            stack_rows(ctx, weights, plan, source_rows, bytes)?
        }
        K3SlotBuild::FoldNormIntoProj => fold_norm_into_proj(ctx, weights, plan, bytes)?,
        K3SlotBuild::TransposeF32 => transpose_f32(ctx, weights, plan, bytes)?,
        K3SlotBuild::Constant(value) => {
            ensure!(
                plan.sources.is_empty() && plan.dtype == K3SlotDtype::Bf16 && bytes == 2,
                "K3 constant slot {} must be a lone bf16 scalar",
                plan.slot
            );
            ctx.stream
                .clone_htod(&bf16::from_f64(value).to_le_bytes())
                .context("K3 constant slot upload")?
        }
    };
    let data = match plan.dtype {
        K3SlotDtype::Bf16 => K3SlotData::Bf16(retype_owned::<bf16>(&ctx.stream, raw)?),
        K3SlotDtype::F32 => K3SlotData::F32(retype_owned::<f32>(&ctx.stream, raw)?),
    };
    Ok(K3SlotBuffer {
        data,
        rows: plan.rows,
        cols: plan.cols,
    })
}

/// Row-concatenate the sources into one buffer. Rows past the sources keep the
/// allocation's zeros — the destination is zeroed before any copy, so a padded
/// slot never carries stale device bytes.
fn stack_rows(
    ctx: &DeviceContext,
    weights: &mut K3RankGpuWeights,
    plan: &K3SlotPlan,
    source_rows: usize,
    bytes: usize,
) -> Result<CudaSlice<u8>> {
    let row_bytes = plan.cols * plan.dtype.bytes();
    let mut dst = ctx
        .stream
        .alloc_zeros::<u8>(bytes)
        .context("K3 fused slot alloc")?;
    let mut offset = 0usize;
    for name in &plan.sources {
        let src = weights.take_tensor(name)?;
        ensure!(
            src.len().is_multiple_of(row_bytes) && offset + src.len() <= bytes,
            "K3 tensor {name} is {} bytes, which does not fit slot {} at offset {offset}",
            src.len(),
            plan.slot
        );
        let mut view = dst.slice_mut(offset..offset + src.len());
        ctx.stream
            .memcpy_dtod(&src, &mut view)
            .with_context(|| format!("K3 fuse {name} into slot {}", plan.slot))?;
        offset += src.len();
    }
    ensure!(
        offset == source_rows * row_bytes,
        "K3 slot {} fused {offset} bytes, the plan promises {} source rows",
        plan.slot,
        source_rows
    );
    Ok(dst)
}

/// Transpose an f32 source stored `[cols, rows]` into the slot's `[rows, cols]`
/// on a host round trip. Only the KDA conv taps take this path: three
/// `[inner, taps]` f32 tensors per KDA layer, ~196 KiB each, which the
/// reference spells `squeeze(1).T` and `conv_silu` reads as `[taps, inner]`.
/// A device transpose would be an element gather, i.e. a kernel; at this size
/// the round trip is the cheaper answer.
fn transpose_f32(
    ctx: &DeviceContext,
    weights: &mut K3RankGpuWeights,
    plan: &K3SlotPlan,
    bytes: usize,
) -> Result<CudaSlice<u8>> {
    ensure!(
        plan.sources.len() == 1 && plan.dtype == K3SlotDtype::F32,
        "K3 transposed slot {} must have one f32 source",
        plan.slot
    );
    let raw = weights.take_tensor(&plan.sources[0])?;
    ensure!(
        raw.len() == bytes,
        "K3 tensor {} is {} bytes, slot {} wants {bytes}",
        plan.sources[0],
        raw.len(),
        plan.slot
    );
    let typed = retype_owned::<f32>(&ctx.stream, raw)?;
    let src = ctx
        .stream
        .clone_dtoh(&typed)
        .with_context(|| format!("K3 read back {} for slot {}", plan.sources[0], plan.slot))?;
    ctx.sync()?;
    drop(typed);

    let host_bytes = transposed(&src, plan.rows, plan.cols)
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<u8>>();
    ctx.stream
        .clone_htod(&host_bytes)
        .with_context(|| format!("K3 upload slot {}", plan.slot))
}

/// `src` is `[cols, rows]` row-major; the result is `[rows, cols]`, i.e.
/// `out[r, c] == src[c, r]`.
fn transposed(src: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            out.push(src[c * rows + r]);
        }
    }
    out
}

/// Fold an RMSNorm gamma into the projection row it always multiplies. Both
/// operands are bf16 `[hidden]`; the product is computed in f32, matching the
/// reference's `gamma.float() * w.float()` exactly (bf16 widens exactly, and
/// the multiply is one IEEE f32 op either way).
fn fold_norm_into_proj(
    ctx: &DeviceContext,
    weights: &mut K3RankGpuWeights,
    plan: &K3SlotPlan,
    bytes: usize,
) -> Result<CudaSlice<u8>> {
    ensure!(
        plan.sources.len() == 2 && plan.cols == 1 && plan.dtype == K3SlotDtype::F32,
        "K3 scoring slot {} must fold two bf16 vectors into one f32 vector",
        plan.slot
    );
    let mut folded = Vec::with_capacity(plan.rows);
    let mut operands = Vec::with_capacity(2);
    for name in &plan.sources {
        let raw = weights.take_tensor(name)?;
        ensure!(
            raw.len() == plan.rows * 2,
            "K3 tensor {name} is {} bytes, slot {} wants a bf16 [{}]",
            raw.len(),
            plan.slot,
            plan.rows
        );
        let typed = retype_owned::<bf16>(&ctx.stream, raw)?;
        let host = ctx
            .stream
            .clone_dtoh(&typed)
            .with_context(|| format!("K3 read back {name} for slot {}", plan.slot))?;
        ctx.sync()?;
        operands.push(host);
    }
    for (gamma, proj) in operands[0].iter().zip(&operands[1]) {
        folded.push(gamma.to_f32() * proj.to_f32());
    }
    let bytes_out = folded
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<u8>>();
    ensure!(bytes_out.len() == bytes, "K3 scoring slot size mismatch");
    ctx.stream
        .clone_htod(&bytes_out)
        .with_context(|| format!("K3 upload slot {}", plan.slot))
}

#[cfg(test)]
mod tests {
    use super::transposed;

    /// The conv-tap transpose is not size-detectable, so pin its direction:
    /// the checkpoint's `[inner, taps]` becomes the `[taps, inner]` the
    /// convolution reads, one tap per contiguous row.
    #[test]
    fn transposed_maps_checkpoint_rows_to_tap_rows() {
        // Three "inner" lanes, two taps: [[0, 1], [2, 3], [4, 5]].
        let src = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(transposed(&src, 2, 3), vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0]);
        // Transposing back is the identity.
        let round_trip = transposed(&transposed(&src, 2, 3), 3, 2);
        assert_eq!(round_trip, src.to_vec());
    }
}
