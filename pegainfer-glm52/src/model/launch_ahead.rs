//! Launch-ahead speculation state machine: the speculative next-step replay
//! (feed kernel + rope re-gather + graph launch enqueued while the current
//! step executes) and its harvest. Split from `model/mod.rs`; see
//! `decode_step` there for the consume/lease protocol.

use anyhow::Result;
use anyhow::ensure;
use pegainfer_kernels::ops::embedding_rows_into;
use pegainfer_kernels::ops::glm52_decode_feed_launch;
use pegainfer_kernels::tensor::DeviceContext;

use super::GLM52_MAX_BATCH_PER_RANK;
use super::GLM52_MAX_STEP_ROWS;
use super::GLM52_VOCAB;
use super::Glm52RankModel;
use super::Glm52StepShape;

/// A speculative next-step whole-graph replay already enqueued on the decode
/// stream: the feed kernel advanced the device input buffers in place and
/// the bucket graph was launched ahead, hiding the ~0.7 ms host-side
/// `cuGraphLaunch` under the current step's execution (measured on jz-38:
/// 5.4 µs intra-step replay gap vs 810 µs at an unhidden step boundary).
/// Consumed only when the engine's next step matches `expect` exactly;
/// on any mismatch the full host prologue overwrites the advanced inputs and
/// the stale replay degrades to a recompute whose rows never influence the
/// following real replay (per-row math is row-independent).
pub(super) struct Glm52SpeculatedStep {
    pub(super) bucket: usize,
    pub(super) active_rows: usize,
    pub(super) slots: [u8; GLM52_MAX_STEP_ROWS],
    /// The engine inputs this speculation assumed: active rows carry
    /// (this step's argmax, position + 1); padding rows echo this step's
    /// padding input verbatim (their device rows keep self-feeding, which
    /// is harmless — pad outputs are never read and rows are isolated).
    pub(super) expect: [(u32, usize); GLM52_MAX_STEP_ROWS],
}

impl Glm52RankModel {
    /// Post-replay tail shared by both step paths: enqueue the argmax D2H,
    /// then — with the copy still in flight — optionally speculate the next
    /// step (feed kernel + rope-row gathers + launch-ahead replay), and only
    /// then block for this step's result. The speculative `cuGraphLaunch`
    /// thereby overlaps this step's execution instead of idling the GPU at
    /// the next step boundary.
    pub(super) fn decode_step_harvest(
        &mut self,
        ctx: &DeviceContext,
        inputs: &[(u32, usize); GLM52_MAX_STEP_ROWS],
        shape: Glm52StepShape,
        lease: bool,
    ) -> Result<[u32; GLM52_MAX_STEP_ROWS]> {
        let batch = shape.bucket;
        let bucket = self
            .buckets
            .iter_mut()
            .find(|bucket| bucket.rows == batch)
            .expect("decode_step validated the bucket");
        ctx.stream.memcpy_dtoh(
            &bucket.scratch.argmax_values,
            &mut bucket.argmax_values_host,
        )?;
        ctx.stream.memcpy_dtoh(
            &bucket.scratch.argmax_indices,
            &mut bucket.argmax_indices_host,
        )?;

        // Speculation must be all-ranks-or-none: the speculative replay is a
        // full set of collectives, so every precondition here is a GLOBAL
        // invariant the engine's lease already guarantees (single-token
        // rows from the shape, position headroom from its slot bookkeeping —
        // pad rows never outrun active rows — and captured graphs from the
        // startup pre-capture). A rank that silently skipped would desync
        // the collective pairing; crash early instead.
        let mut speculated = false;
        if lease {
            // Only real rows carry slot semantics; the padding tail reuses
            // slot id 0 (#812) and self-feeds harmlessly.
            let mut seen = [false; GLM52_MAX_BATCH_PER_RANK];
            for &slot in &shape.slots[..shape.active_rows] {
                let slot = slot as usize;
                ensure!(
                    slot < GLM52_MAX_BATCH_PER_RANK && !std::mem::replace(&mut seen[slot], true),
                    "GLM5.2 launch-ahead lease granted for a step with repeated slot {slot} \
                     (span rows are never leased)"
                );
            }
            for &position in &self.device_positions[..batch] {
                ensure!(
                    position + 1 < self.max_model_len,
                    "GLM5.2 launch-ahead lease granted with a row at position {position} — the \
                     advanced step would breach the model-length cap"
                );
                // The feed kernel advances slot_mapping by +1, which only
                // stays inside the row's current 64-token page off a page
                // boundary; the next page is not even in the uploaded block
                // table. The engine's lease gate breaks the streak at
                // every ACTIVE row's boundary, which also bounds every
                // padding row (reset to position 0 by each full prologue)
                // inside the padding page — a violation here is a gate bug.
                ensure!(
                    !(position + 1).is_multiple_of(super::GLM52_MODEL_LEN_ALIGN),
                    "GLM5.2 launch-ahead lease granted with a row crossing the 64-token page \
                     boundary at position {position} — the advanced slot mapping would leave \
                     its page"
                );
            }
            ensure!(
                bucket.graph.is_captured(),
                "GLM5.2 launch-ahead lease granted before the bucket {batch} graph was \
                 captured — the startup pre-capture must cover every shape"
            );
            // From the first enqueue below onward, a host-side failure
            // (launch error) leaves the other ranks' speculative collectives
            // half-paired — teardown then eats the ~100 s device timeout.
            // Every fallible CHECK is above for exactly that reason; do not
            // insert anything that can fail between these launches.
            glm52_decode_feed_launch(
                ctx,
                &bucket.scratch.argmax_indices,
                &mut self.token_ids,
                &mut self.positions,
                &mut self.slot_mapping,
                &mut self.seq_lens,
                batch,
            )?;
            embedding_rows_into(ctx, &self.cos_table, &self.positions, batch, &mut self.cos)?;
            embedding_rows_into(ctx, &self.sin_table, &self.positions, batch, &mut self.sin)?;
            bucket.graph.launch_captured(ctx)?;
            for position in &mut self.device_positions[..batch] {
                *position += 1;
            }
            speculated = true;
        }

        // Block on the D2H (the pinned slices synchronize on their own copy
        // events) and unpack — identical semantics to the old blocking
        // readback.
        let top_values = bucket
            .argmax_values_host
            .as_slice()
            .map_err(|err| anyhow::anyhow!("GLM5.2 argmax values D2H sync failed: {err}"))?;
        let top_indices = bucket
            .argmax_indices_host
            .as_slice()
            .map_err(|err| anyhow::anyhow!("GLM5.2 argmax indices D2H sync failed: {err}"))?;
        let mut outputs = [0u32; GLM52_MAX_STEP_ROWS];
        for row in 0..batch {
            outputs[row] = top_indices[row].max(0) as u32;
        }
        // Validate ACTIVE rows only: padding rows self-feed their own argmax
        // through lease streaks, and nobody consumes their outputs — an
        // engine-fatal assertion has no business reading them.
        for row in 0..shape.active_rows {
            let slot = shape.slots[row] as usize;
            let top_value = top_values[row].to_f32();
            let top_index = top_indices[row];
            ensure!(
                top_value.is_finite(),
                "GLM5.2 slot {slot} greedy argmax found no finite logit (top = {top_value})"
            );
            ensure!(
                (0..GLM52_VOCAB as i32).contains(&top_index),
                "GLM5.2 slot {slot} greedy argmax index {top_index} outside the vocab"
            );
        }

        if speculated {
            let mut expect = *inputs;
            for row in 0..shape.active_rows {
                expect[row] = (outputs[row], inputs[row].1 + 1);
            }
            self.speculated = Some(Glm52SpeculatedStep {
                bucket: batch,
                active_rows: shape.active_rows,
                slots: shape.slots,
                expect,
            });
        }
        Ok(outputs)
    }
}
