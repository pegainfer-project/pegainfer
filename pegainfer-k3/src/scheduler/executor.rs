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

use anyhow::Result;
use pegainfer_frontend::sampler::SamplingParams;

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

    /// Drop the slot's state. Called on every terminal path, including the
    /// silent one (abort), and always before the slot is handed out again.
    fn release(&mut self, slot: SlotId);
}
