//! The model-execution seam the K3 scheduler drives.
//!
//! North of this trait is protocol — admission, the slot budget, aborts,
//! finish reasons — and it is what [`super::K3Scheduler`] implements against
//! the engine contract. South of it is the model: KDA recurrent state, the
//! paged MLA KV pool, latent MoE. Keeping the boundary this narrow means the
//! whole protocol is exercised today against a fake executor, and the real
//! model drops in later without a line of protocol code moving.
//!
//! The executor owns a fixed table of execution slots. A slot is one
//! request's seat in the batch — its paged-KV entry for the MLA layers plus
//! its KDA state row — and the scheduler hands slots out from a free list it
//! sizes from [`StepExecutor::max_batch`], returning each slot on every
//! terminal path (finish, fail, abort).

use std::sync::Arc;

use anyhow::Result;
use anyhow::bail;
use pegainfer_frontend::sampler::SamplingParams;

use super::whale::CommittedWhale;
use crate::executor::cp::K3CpGroup;

/// Index of one execution slot, in `0..max_batch`.
pub type SlotId = usize;

/// One occupied slot's input for a decode step. The executor advances that
/// slot's state by exactly one token and samples one token from it.
#[derive(Clone, Copy, Debug)]
pub struct DecodeSlot {
    pub slot: SlotId,
    /// The request's most recently committed token — this step's input.
    pub last_token: u32,
}

/// One K3 model replica, stepped by exactly one scheduler thread.
///
/// Errors are per-request, not per-engine: the scheduler answers a failing
/// prefill or decode by failing the requests that step touched and carries
/// on serving, so an executor reports trouble with `Err` rather than
/// poisoning itself.
pub trait StepExecutor: Send {
    /// Slots this executor can hold concurrently. Fixed for the executor's
    /// lifetime — the scheduler sizes its free list from it once.
    fn max_batch(&self) -> usize;

    /// Longest request this executor can ever serve, in tokens
    /// (`prompt + max_tokens`). Requests above it are rejected at admission
    /// instead of queueing forever.
    fn max_context_tokens(&self) -> usize;

    /// Ingest a prompt into `slot` and sample the request's first token.
    /// `params` belong to the slot from here until it is released: decode
    /// steps carry only the input token, so the executor keeps the per-slot
    /// sampling state itself.
    fn prefill(&mut self, slot: SlotId, prompt: &[u32], params: &SamplingParams) -> Result<u32>;

    /// Advance every slot in `batch` by one token. The returned tokens are
    /// parallel to `batch`.
    ///
    /// The scheduler calls this every step, **including with an empty
    /// batch**: an EP rank with nothing to serve must still launch the
    /// step's fixed per-layer MoE kernels (padding rows in place of live
    /// ones) so the device-side barriers inside them pair against the right
    /// peer step. Executors with no cross-rank obligations answer an empty
    /// batch with an empty vec and no device work.
    fn decode(&mut self, batch: &[DecodeSlot]) -> Result<Vec<u32>>;

    /// Advance every slot in `batch` by one *round*, committing one or more
    /// tokens per slot (parallel to `batch`, each list non-empty). This is
    /// what the scheduler drives every step; the default is plain decode —
    /// one token per slot — and an executor with a speculative path
    /// overrides it to return each round's accepted span.
    fn decode_many(&mut self, batch: &[DecodeSlot]) -> Result<Vec<Vec<u32>>> {
        Ok(self
            .decode(batch)?
            .into_iter()
            .map(|token| vec![token])
            .collect())
    }

    /// Context-parallel prefill: run this executor's segment of `prompt` as
    /// CP rank `cp_rank` of the gang in `group`, in lockstep with the whole
    /// gang. The last CP rank owns the sequence — it ingests the result into
    /// `slot` and returns the boundary token; other ranks ignore `slot` and
    /// return `None`. Default: this executor serves no CP.
    fn prefill_cp(
        &mut self,
        slot: SlotId,
        prompt: &[u32],
        group: &Arc<K3CpGroup>,
        cp_rank: usize,
    ) -> Result<Option<u32>> {
        let _ = (slot, prompt, group, cp_rank);
        bail!("this executor does not serve context-parallel prefill")
    }

    /// Enter a committed whale's CP superstep — the fleet-wide counterpart of
    /// [`StepExecutor::prefill_cp`]. Called exactly at the whale's committed
    /// launch, on every gang member, by the whale serving lane
    /// ([`super::whale`]); the descriptor names the gang and the segments are
    /// a deterministic function of it. `slot` is set on the owner (the
    /// poster, always the last CP rank), which ingests the result and returns
    /// the boundary token; helpers pass `None` and return `None`. One
    /// superstep is one launch. Default: this executor serves no whale lane.
    fn prefill_whale(
        &mut self,
        whale: &CommittedWhale,
        slot: Option<SlotId>,
    ) -> Result<Option<u32>> {
        let _ = (whale, slot);
        bail!("this executor does not serve whale prefill")
    }

    /// Run one padding step and wait for it. A scheduler thread waiting at a
    /// CP-gang rendezvous calls this so a peer blocked inside its own step's
    /// sync keeps receiving the mega launches that pair against its queued
    /// ones, while the wait keeps this rank's launch count within a step of
    /// the world's. Executors with no cross-rank obligations do nothing.
    fn pump_step(&mut self) -> Result<()> {
        Ok(())
    }

    /// Steps this executor has launched, of every kind. On an EP rank the
    /// mega launches pair across ranks by absolute index, so a CP gang
    /// equalizes the world's counts (pumping the laggards) before a step
    /// whose mid-step stream sync would otherwise wait on peer launches that
    /// can no longer come. Executors with no cross-rank obligations report 0.
    ///
    /// The gang's leveling is sound only if every stepping method returns
    /// with its steps fully launched — a method that kept launching after
    /// returning would advance this count under a peer already leveled
    /// against it.
    fn step_count(&self) -> u64 {
        0
    }

    /// Drop the slot's state. Called on every terminal path, including the
    /// silent one (abort), and always before the slot is handed out again.
    fn release(&mut self, slot: SlotId);
}
