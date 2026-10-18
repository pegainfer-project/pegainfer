mod tp1_dp8;
mod tp8_dp1;

use anyhow::Result;
pub(super) use tp1_dp8::Tp1Dp8ForwardExecutor;
pub(super) use tp8_dp1::Tp8Dp1ForwardExecutor;

use super::worker::KimiKvStepPages;
use super::worker::KimiOneTokenForwardReport;
use super::worker::KimiRowOptions;

pub(super) const DP_MAX_BATCH_PER_RANK: usize = 8;

pub(super) trait ForwardExecutor {
    /// Ensure `slot < decode_batch_size` is valid for following prefill/decode calls.
    fn ensure_decode_batch(&self, decode_batch_size: usize) -> Result<()>;

    /// Forward one prompt into `slot` inside a stable arena of `decode_batch_size` rows.
    /// `input_ids` is the uncached suffix; `cached_tokens` counts the prefix
    /// already present in the slot's pool pages (prefix cache hit), so
    /// `kv_pages` covers `cached_tokens + input_ids.len()` tokens in one CSR
    /// row. `row` selects the next token (greedy argmax vs sampling, driven
    /// by `seed`) and requests `logprobs` for it in the report.
    #[allow(clippy::too_many_arguments)]
    fn forward_prefill(
        &self,
        input_ids: &[u32],
        slot: usize,
        decode_batch_size: usize,
        cached_tokens: usize,
        ep_max_seq_len: usize,
        kv_pages: &KimiKvStepPages,
        row: KimiRowOptions,
        seed: u64,
    ) -> Result<KimiOneTokenForwardReport>;

    /// Return exactly one report per input row, in the same order.
    /// `kv_pages` carries one CSR row per input row, covering that request's
    /// KV including this step's append.
    #[allow(clippy::too_many_arguments)]
    fn forward_decode_batch(
        &self,
        token_ids: &[u32],
        append_positions: &[usize],
        slots: &[usize],
        decode_batch_size: usize,
        kv_pages: &KimiKvStepPages,
        rows: &[KimiRowOptions],
        seed: u64,
    ) -> Result<Vec<KimiOneTokenForwardReport>>;

    fn worker_count(&self) -> usize;

    fn gpu_weight_ready_count(&self) -> usize;
}
