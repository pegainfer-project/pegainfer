//! The model's forward pass, behind a narrow host contract.
//!
//! Everything below this module is the certified launch sequence of the K3
//! forward pass — the part of the executor that is fixed by the model
//! architecture and gated by the golden fixtures, not framework code. Open it
//! when chasing numerics; for everything else, this contract is the whole
//! story:
//!
//! * **Host → device, before a step.** The caller stages three small arrays
//!   and copies them in: `scratch.token_ids` (the rows' input tokens),
//!   `scratch.context_len` (per-row attention context) and `scratch.kv_row`
//!   (per-row KV append position). That is the only H2D traffic a step
//!   depends on.
//! * **Device → host, after a step.** The sampled token ids sit in
//!   `scratch.argmax_indices`; reading them back is the caller's only D2H.
//!   Nothing *inside* a step reads device memory back to the host.
//! * **Capturable.** No launch varies its geometry with device state — the
//!   geometry is a function of [`K3StepShape`] alone — so the whole sequence
//!   can be captured into a CUDA graph and replayed against restaged inputs.
//! * **Expert parallelism.** Every rank issues the same launches in the same
//!   order every step, batch contents (or emptiness) notwithstanding; only
//!   the fused MegaMoE launch touches peers. That is what lets an EP group
//!   run without a coordinator.
//!
//! Three entry points, two of them the same sequence:
//!
//! * [`k3_decode_step`] — advance every row of the bucket by one token.
//! * [`k3_prefill_chunk_step`] — the identical sequence, with the bucket's
//!   rows carrying consecutive tokens of one sequence. Rows past
//!   `live_rows` are padding whose results are discarded. Chunk steps skip
//!   the epilogue (lm_head + sampling)...
//! * [`k3_prefill_boundary_sample`] — ...which runs here instead, once after
//!   the final chunk, at `b = 1` over the boundary token.

mod decode;
mod gemm;
mod prefill;
mod step;

pub(crate) use decode::k3_decode_step;
pub(crate) use prefill::k3_prefill_boundary_sample;
pub(crate) use prefill::k3_prefill_chunk_step;
pub(crate) use step::K3StepShape;
