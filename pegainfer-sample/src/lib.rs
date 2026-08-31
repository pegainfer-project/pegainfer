//! Batched token selection and host logprob formatting — the shared sampling
//! layer every model crate routes through.
//!
//! Two model-agnostic jobs live here:
//!
//! * [`select_batch`] — turn one logits arena into one next-token id per row.
//!   Greedy rows take a batched indexed argmax; the rest take batched
//!   FlashInfer temperature/top-k/top-p/min_p passes (min_p rows partitioned
//!   off the fused fast path, seeded rows as single-row replayable calls).
//!   There is no per-row escape hatch, so a caller cannot regress to
//!   `for i { sample(i) }`.
//! * [`token_logprobs_batch`] — batched device logprob extraction;
//!   [`token_logprob_from_row`] is its host single-row reference.
//!
//! Layering: the `.cu`/FFI and the low-level batch primitives live in
//! `pegainfer-kernels` (the CUDA build owner); this crate owns the policy
//! (greedy/non-greedy routing, logprob math) and the reusable scratch.
//! `SamplingParams`/`TokenLogprob` stay in `pegainfer-engine` (the CUDA-free
//! contract crate) and are reachable from here.
//!
//! Kimi-K2 keeps its own greedy path — a sharded-vocab local argmax whose top-1
//! logit feeds a cross-rank DP reduction (#236/#237), which [`select_batch`]'s
//! whole-vocab assumption cannot express — but routes its non-greedy rows
//! through the re-exported [`gpu_sample_batch_into`] and its logprobs through
//! [`token_logprob_from_row`].

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::PinnedHostSlice;
use pegainfer_frontend::engine::TokenLogprob;
pub use pegainfer_frontend::sampler::SamplingParams;
/// Low-level batched sampling, re-exported so a model that must drive its own
/// greedy path still reaches the single sampler entry rather than dipping into
/// `pegainfer-kernels` directly — e.g. Kimi-K2 (see the module docs).
pub use pegainfer_kernels::ops::BatchSamplingRow;
/// Low-level batched sampling, re-exported so a model that must drive its own
/// greedy path still reaches the single sampler entry rather than dipping into
/// `pegainfer-kernels` directly — e.g. Kimi-K2 (see the module docs).
pub use pegainfer_kernels::ops::BatchSamplingScratch;
use pegainfer_kernels::ops::argmax_batch_bf16_split_indexed_into;
use pegainfer_kernels::ops::argmax_batch_bf16_split_partials_len;
/// Low-level batched sampling, re-exported so a model that must drive its own
/// greedy path still reaches the single sampler entry rather than dipping into
/// `pegainfer-kernels` directly — e.g. Kimi-K2 (see the module docs).
pub use pegainfer_kernels::ops::gpu_sample_batch_into;
use pegainfer_kernels::ops::logprob_topk_batch_bf16_into;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;
use pegainfer_kernels::tensor::has_stream_override;
use pegainfer_kernels::tensor::stream_spin_wait;

const STAGED_READBACK_SLOTS: usize = 2;

/// Allocate-once device buffers for [`select_batch`], sized for `max_rows` × `vocab`.
///
/// Reused across decode steps — the decode path needs pointer-stable buffers, so
/// never reallocate per step. Greedy and non-greedy rows use disjoint buffers and
/// run sequentially, so a single `SampleScratch` covers a full mixed batch.
pub struct SampleScratch {
    /// Greedy row indices into the logits arena (indexed argmax input).
    row_indices: CudaSlice<i32>,
    argmax_partial_values: CudaSlice<f32>,
    argmax_partial_indices: CudaSlice<i32>,
    /// Top-1 logit value per greedy row — a required out-param of the argmax
    /// kernel, not read here.
    top1_values: CudaSlice<half::bf16>,
    /// One token id per greedy row, in `row_indices` order.
    argmax_out: CudaSlice<i32>,
    /// Pinned host landing buffer for `argmax_out`. A pageable readback here
    /// has synchronous-copy semantics, so the step thread queues behind any
    /// concurrent bulk copy traffic — a P/D KV-restore flood turned this
    /// sub-ms call into a flat 23.6 ms and froze token delivery for every
    /// active stream (#704). Pinned keeps the D2H async; the reader blocks
    /// only on the copy's own event.
    argmax_host: PinnedHostSlice<i32>,
    /// Alternate landing slot used while the prior readback is collected.
    argmax_host_alt: PinnedHostSlice<i32>,
    /// Identity row map for the all-greedy staged path.
    identity_rows: CudaSlice<i32>,
    sampling: BatchSamplingScratch,
    /// Vocab width every buffer above was sized for; `select_batch` rejects a
    /// logits arena whose `hidden_dim` differs, since the sizes are baked in.
    vocab: usize,
    max_rows: usize,
}

impl SampleScratch {
    pub fn new(ctx: &DeviceContext, vocab: usize, max_rows: usize) -> Result<Self> {
        ensure!(
            vocab > 0 && max_rows > 0,
            "SampleScratch requires vocab > 0 and max_rows > 0"
        );
        let partials = argmax_batch_bf16_split_partials_len(max_rows, vocab);
        let alloc_i32 = |n: usize| -> Result<CudaSlice<i32>> {
            ctx.stream
                .alloc_zeros(n)
                .map_err(|e| anyhow!("SampleScratch alloc failed: {e}"))
        };
        Ok(Self {
            row_indices: alloc_i32(max_rows)?,
            argmax_partial_values: ctx
                .stream
                .alloc_zeros(partials)
                .map_err(|e| anyhow!("SampleScratch alloc failed: {e}"))?,
            argmax_partial_indices: alloc_i32(partials)?,
            top1_values: ctx
                .stream
                .alloc_zeros(max_rows)
                .map_err(|e| anyhow!("SampleScratch alloc failed: {e}"))?,
            argmax_out: alloc_i32(max_rows)?,
            // Read only after a D2H lands in it (write-combined pages start
            // uninitialized). cudarc's alloc_pinned hardcodes write-combined
            // memory, whose CPU reads are uncached — fine for max_rows i32s,
            // but don't grow this buffer into anything read in a hot loop.
            argmax_host: unsafe { ctx.ctx.alloc_pinned::<i32>(max_rows) }
                .map_err(|e| anyhow!("SampleScratch pinned alloc failed: {e}"))?,
            argmax_host_alt: unsafe { ctx.ctx.alloc_pinned::<i32>(max_rows) }
                .map_err(|e| anyhow!("SampleScratch pinned alloc failed: {e}"))?,
            identity_rows: ctx
                .stream
                .clone_htod(&(0..max_rows as i32).collect::<Vec<_>>())
                .map_err(|e| anyhow!("SampleScratch identity upload failed: {e}"))?,
            sampling: BatchSamplingScratch::new(ctx, max_rows, vocab)?,
            vocab,
            max_rows,
        })
    }

    pub fn max_rows(&self) -> usize {
        self.max_rows
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }
}

/// Pick the next token for every row of a logits arena.
///
/// `params[i]` governs arena row `i`. Argmax rows are resolved together with a
/// batched indexed argmax; the remaining rows are resolved together with one
/// FlashInfer temperature/top-k/top-p pass seeded by `seed` (min_p rows are
/// partitioned into their own pass inside `gpu_sample_batch_into`, so
/// min_p-free rows always take the fused fast path). Returns one token id per
/// row, in row order.
///
/// A row takes the argmax path when [`effectively_greedy`] holds: explicit
/// greedy params, or a `top_p` nucleus so tight (`<= 1/vocab`) that only the
/// argmax survives. Routing those through argmax keeps an effectively-greedy
/// request deterministic — the rejection sampler would otherwise pick an
/// arbitrary member of a bf16-tied top — and skips a softmax it does not need.
///
/// `seed` must be fresh per decode step (one engine seed at startup, advanced
/// per step); unseeded rows decorrelate through the philox subsequence.
///
/// `steps[i]` is row `i`'s request-local decode step. It only matters for
/// rows whose params carry a `seed`: a seeded row is sampled as its own
/// single-row call with `mix_seed(request_seed, step)` as the philox seed, so
/// its tokens are a pure function of (seed, step, distribution) — FlashInfer's
/// kernels fold the batch position into the philox subsequence, so seeded rows
/// cannot ride the batched call without their stream changing with batch
/// composition.
pub fn select_batch(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    params: &[&SamplingParams],
    steps: &[u64],
    seed: u64,
    scratch: &mut SampleScratch,
) -> Result<Vec<u32>> {
    let n = params.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    ensure!(
        steps.len() == n,
        "select_batch: {} steps for {n} rows",
        steps.len()
    );
    ensure!(
        n <= scratch.max_rows,
        "select_batch: {n} rows exceeds scratch capacity {}",
        scratch.max_rows
    );
    ensure!(
        logits.seq_len >= n,
        "select_batch: logits arena has {} rows, need {n}",
        logits.seq_len
    );

    let vocab = logits.hidden_dim;
    ensure!(
        vocab == scratch.vocab,
        "select_batch: logits vocab {vocab} != scratch vocab {}",
        scratch.vocab
    );
    let is_argmax = |p: &&SamplingParams| effectively_greedy(p, vocab);
    let mut tokens = vec![0u32; n];

    // Argmax rows -> one batched indexed argmax.
    let greedy: Vec<i32> = params
        .iter()
        .enumerate()
        .filter_map(|(i, p)| is_argmax(p).then_some(i as i32))
        .collect();
    if !greedy.is_empty() {
        ctx.stream
            .memcpy_htod(&greedy, &mut scratch.row_indices)
            .map_err(|e| anyhow!("select_batch H2D greedy rows failed: {e}"))?;
        argmax_batch_bf16_split_indexed_into(
            ctx,
            logits,
            &scratch.row_indices,
            greedy.len(),
            &mut scratch.argmax_partial_values,
            &mut scratch.argmax_partial_indices,
            &mut scratch.top1_values,
            &mut scratch.argmax_out,
        )?;
        ctx.stream
            .memcpy_dtoh(&scratch.argmax_out, &mut scratch.argmax_host)
            .map_err(|e| anyhow!("select_batch D2H greedy tokens failed: {e}"))?;
        // Blocks on this copy's own event — which transitively covers the
        // argmax kernel queued before it on the same stream, so the wait is
        // equivalent to the old full-stream sync for this path. The spin
        // ahead of it takes the scheduler wake-up out of the step loop.
        stream_spin_wait(ctx)?;
        let out = scratch
            .argmax_host
            .as_slice()
            .map_err(|e| anyhow!("select_batch greedy D2H sync failed: {e}"))?;
        for (k, &row) in greedy.iter().enumerate() {
            tokens[row as usize] = out[k] as u32;
        }
    }

    // Unseeded sampling rows -> batched FlashInfer sampling. min_p rows may
    // mix freely: `gpu_sample_batch_into` partitions them into their own pass
    // so min_p-free rows always ride the fused fast path.
    let sampling_rows: Vec<BatchSamplingRow> = params
        .iter()
        .enumerate()
        .filter(|(_, p)| !is_argmax(p) && p.seed.is_none())
        .map(|(i, p)| BatchSamplingRow {
            row: i,
            temperature: p.temperature,
            top_k: p.top_k,
            top_p: p.top_p,
            min_p: p.min_p,
        })
        .collect();
    if !sampling_rows.is_empty() {
        let sampled = gpu_sample_batch_into(
            ctx,
            logits.as_ref(),
            &sampling_rows,
            seed,
            &mut scratch.sampling,
        )?;
        for (r, token) in sampling_rows.iter().zip(&sampled) {
            tokens[r.row] = *token;
        }
    }

    // Seeded rows -> one single-row call each, philox seed mixed from the
    // request seed and step so replay is independent of batch composition
    // (blockIdx is always 0 in an n=1 call).
    for (i, p) in params.iter().enumerate() {
        let Some(request_seed) = p.seed else {
            continue;
        };
        if is_argmax(p) {
            continue;
        }
        let row = [BatchSamplingRow {
            row: i,
            temperature: p.temperature,
            top_k: p.top_k,
            top_p: p.top_p,
            min_p: p.min_p,
        }];
        let sampled = gpu_sample_batch_into(
            ctx,
            logits.as_ref(),
            &row,
            mix_seed(request_seed, steps[i]),
            &mut scratch.sampling,
        )?;
        tokens[i] = sampled[0];
    }

    Ok(tokens)
}

fn validate_greedy_shape(
    logits: &HiddenStates,
    rows: usize,
    scratch: &SampleScratch,
) -> Result<()> {
    ensure!(rows > 0, "greedy_argmax_ids: empty batch");
    ensure!(
        rows <= scratch.max_rows,
        "greedy_argmax_ids: {rows} rows exceeds scratch capacity {}",
        scratch.max_rows
    );
    ensure!(
        logits.seq_len >= rows && logits.hidden_dim == scratch.vocab,
        "greedy_argmax_ids: logits shape {}x{} cannot serve {rows} rows x vocab {}",
        logits.seq_len,
        logits.hidden_dim,
        scratch.vocab
    );
    Ok(())
}

/// Capturable base-stream-only argmax and device copy into the next embedding's id buffer.
pub fn greedy_argmax_ids(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    rows: usize,
    ids_out: &mut CudaSlice<u32>,
    scratch: &mut SampleScratch,
) -> Result<()> {
    ensure!(
        !has_stream_override(),
        "the staged sampler path runs on the base stream only"
    );
    validate_greedy_shape(logits, rows, scratch)?;
    argmax_batch_bf16_split_indexed_into(
        ctx,
        logits,
        &scratch.identity_rows,
        rows,
        &mut scratch.argmax_partial_values,
        &mut scratch.argmax_partial_indices,
        &mut scratch.top1_values,
        &mut scratch.argmax_out,
    )?;
    pegainfer_kernels::tensor::memcpy_dtod_u32_from_i32(ctx, &scratch.argmax_out, ids_out, rows)
}

/// Queue the base-stream-only pinned readback outside the sampler graph.
pub fn greedy_stage_readback(
    ctx: &DeviceContext,
    slot: usize,
    scratch: &mut SampleScratch,
) -> Result<()> {
    ensure!(
        !has_stream_override(),
        "the staged sampler path runs on the base stream only"
    );
    ensure!(
        slot < STAGED_READBACK_SLOTS,
        "greedy_stage_readback: slot {slot} out of range"
    );
    let host = if slot == 0 {
        &mut scratch.argmax_host
    } else {
        &mut scratch.argmax_host_alt
    };
    ctx.stream
        .memcpy_dtoh(&scratch.argmax_out, host)
        .map_err(|e| anyhow!("greedy_stage_readback D2H stage failed: {e}"))
}

/// On the base stream, argmax every row, leave the picks in the next embedding's
/// id buffer, and queue their pinned readback without waiting for it.
pub fn greedy_stage_resident(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    rows: usize,
    ids_out: &mut CudaSlice<u32>,
    slot: usize,
    scratch: &mut SampleScratch,
) -> Result<()> {
    ensure!(
        !has_stream_override(),
        "the staged sampler path runs on the base stream only"
    );
    greedy_argmax_ids(ctx, logits, rows, ids_out, scratch)?;
    greedy_stage_readback(ctx, slot, scratch)
}

/// Collect a readback staged into one of the two pinned slots.
pub fn greedy_collect_resident(
    rows: usize,
    slot: usize,
    scratch: &mut SampleScratch,
) -> Result<Vec<u32>> {
    ensure!(
        slot < STAGED_READBACK_SLOTS,
        "greedy_collect_resident: slot {slot} out of range"
    );
    ensure!(
        rows <= scratch.max_rows,
        "greedy_collect_resident: {rows} rows exceeds scratch capacity {}",
        scratch.max_rows
    );
    let landed = if slot == 0 {
        scratch.argmax_host.as_slice()
    } else {
        scratch.argmax_host_alt.as_slice()
    }
    .map_err(|e| anyhow!("greedy_collect_resident D2H sync failed: {e}"))?;
    Ok(landed[..rows].iter().map(|&token| token as u32).collect())
}

/// SplitMix64 over (seed, step): a distinct, well-mixed philox seed per
/// request step, deterministic across runs and batch layouts. Public for the
/// models that drive their own greedy path (see the module docs) and must
/// reproduce [`select_batch`]'s seeded-row semantics: mix the request seed
/// with the request-local step, one single-row call per seeded row.
pub fn mix_seed(seed: u64, step: u64) -> u64 {
    let mut z = seed ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Whether a row can take the argmax path without changing sampling semantics.
///
/// Besides explicit greedy params, a `top_p` at or below `1/vocab` leaves only
/// the argmax token in the nucleus: the softmax maximum is always `>= 1/vocab`.
/// Public for the models that drive their own greedy path (see the module
/// docs): routing these rows to the sampler instead would make them stochastic
/// on bf16-tied maxima, diverging from [`select_batch`]'s semantics.
pub fn effectively_greedy(params: &SamplingParams, vocab_size: usize) -> bool {
    params.is_greedy()
        || (vocab_size > 0
            && params.top_p.is_finite()
            && params.top_p > 0.0
            && params.top_p <= 1.0 / vocab_size as f32)
}

/// Host log-softmax of `picked` plus the top-`top_k`, from one full-vocab logits
/// row. One O(V) pass; runs only when a request asked for logprobs.
///
/// Generic over the row element so callers feed `f32` (Qwen, already on host as
/// f32) or `bf16` (Kimi, straight from the device arena) without a widening
/// copy. Exponentials accumulate in f64 — summing a 160k-wide vocab in f32 loses
/// precision. Returns `None` for an empty row or an out-of-range `picked`.
///
/// The row must be the unsharded global vocab: a shard-local logsumexp is not
/// the global one, so sharded callers must merge across ranks first (#236).
pub fn token_logprob_from_row<T>(row: &[T], picked: u32, top_k: usize) -> Option<TokenLogprob>
where
    T: Copy + Into<f32>,
{
    let picked = picked as usize;
    if row.is_empty() || picked >= row.len() {
        return None;
    }

    let mut max = f32::NEG_INFINITY;
    for &v in row {
        max = max.max(v.into());
    }
    let mut sum = 0f64;
    for &v in row {
        let x: f32 = v.into();
        sum += f64::from(x - max).exp();
    }
    let log_sum_exp = max + sum.ln() as f32;

    // Rank before subtracting LSE so f32 rounding cannot change raw-logit order.
    let k = top_k.min(row.len());
    let mut top: Vec<(u32, f32)> = Vec::with_capacity(k + 1);
    if k > 0 {
        for (id, &v) in row.iter().enumerate() {
            let val: f32 = v.into();
            if top.len() == k && val <= top[k - 1].1 {
                continue;
            }
            let pos = top.partition_point(|&(_, kept)| kept >= val);
            top.insert(pos, (id as u32, val));
            top.truncate(k);
        }
        for entry in &mut top {
            entry.1 -= log_sum_exp;
        }
    }

    let picked_val: f32 = row[picked].into();
    Some(TokenLogprob {
        logprob: picked_val - log_sum_exp,
        top_logprobs: top,
    })
}

/// One row of a [`token_logprobs_batch`] call: the logprob of `picked` plus
/// the arena row's `top_k` entries.
#[derive(Clone, Copy, Debug)]
pub struct LogprobRequest {
    pub row: usize,
    pub picked: u32,
    pub top_k: usize,
}

/// Batched device twin of [`token_logprob_from_row`]: one launch and a
/// compact O(rows x (top_k + 1)) readback per call.
pub fn token_logprobs_batch(
    ctx: &DeviceContext,
    logits: &HiddenStates,
    requests: &[LogprobRequest],
) -> Result<Vec<TokenLogprob>> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }
    ensure!(
        !has_stream_override(),
        "token_logprobs_batch: cannot run under a stream override — buffer \
         traffic is ordered on the primary stream"
    );
    let vocab = logits.hidden_dim;
    ensure!(vocab > 0, "token_logprobs_batch: empty vocab");
    let mut rows = Vec::with_capacity(requests.len());
    let mut picked = Vec::with_capacity(requests.len());
    let mut ks = Vec::with_capacity(requests.len());
    let mut k_max = 0usize;
    for r in requests {
        ensure!(
            r.row < logits.seq_len,
            "token_logprobs_batch: row {} out of bounds for arena of {} rows",
            r.row,
            logits.seq_len
        );
        ensure!(
            (r.picked as usize) < vocab,
            "token_logprobs_batch: picked token {} out of bounds for vocab {vocab}",
            r.picked
        );
        let k = r.top_k.min(vocab);
        k_max = k_max.max(k);
        rows.push(i32::try_from(r.row)?);
        picked.push(i32::try_from(r.picked)?);
        ks.push(i32::try_from(k)?);
    }

    let htod = |data: &[i32]| -> Result<CudaSlice<i32>> {
        ctx.stream
            .clone_htod(data)
            .map_err(|e| anyhow!("token_logprobs_batch H2D failed: {e}"))
    };
    let rows_gpu = htod(&rows)?;
    let picked_gpu = htod(&picked)?;
    let ks_gpu = htod(&ks)?;
    let mut picked_lp_gpu: CudaSlice<f32> = ctx
        .stream
        .alloc_zeros(requests.len())
        .map_err(|e| anyhow!("token_logprobs_batch alloc failed: {e}"))?;
    let topk_len = requests
        .len()
        .checked_mul(k_max)
        .ok_or_else(|| anyhow!("token_logprobs_batch: rows * k_max overflows"))?
        .max(1);
    let mut vals_gpu: CudaSlice<f32> = ctx
        .stream
        .alloc_zeros(topk_len)
        .map_err(|e| anyhow!("token_logprobs_batch alloc failed: {e}"))?;
    let mut ids_gpu: CudaSlice<i32> = ctx
        .stream
        .alloc_zeros(topk_len)
        .map_err(|e| anyhow!("token_logprobs_batch alloc failed: {e}"))?;

    // Indices validated and stream override rejected above.
    unsafe {
        logprob_topk_batch_bf16_into(
            ctx,
            logits.as_ref(),
            &rows_gpu,
            &picked_gpu,
            &ks_gpu,
            requests.len(),
            k_max,
            &mut picked_lp_gpu,
            &mut vals_gpu,
            &mut ids_gpu,
        )?;
    }

    let picked_lp = ctx
        .stream
        .clone_dtoh(&picked_lp_gpu)
        .map_err(|e| anyhow!("token_logprobs_batch D2H failed: {e}"))?;
    let vals = ctx
        .stream
        .clone_dtoh(&vals_gpu)
        .map_err(|e| anyhow!("token_logprobs_batch D2H failed: {e}"))?;
    let ids = ctx
        .stream
        .clone_dtoh(&ids_gpu)
        .map_err(|e| anyhow!("token_logprobs_batch D2H failed: {e}"))?;
    ctx.sync()?;

    Ok((0..requests.len())
        .map(|i| {
            let k = ks[i] as usize;
            let base = i * k_max;
            TokenLogprob {
                logprob: picked_lp[i],
                top_logprobs: (0..k)
                    .map(|j| (ids[base + j] as u32, vals[base + j]))
                    .collect(),
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::token_logprob_from_row;

    #[test]
    fn token_logprob_matches_exact_log_softmax() {
        // bf16-exact inputs so the expected values are analytic.
        let row: Vec<bf16> = [1.0f32, 3.0, 2.0, 0.0, 3.0]
            .iter()
            .map(|&v| bf16::from_f32(v))
            .collect();
        let lse = (1f64.exp() + 3f64.exp() + 2f64.exp() + 1.0 + 3f64.exp()).ln() as f32;

        let out = token_logprob_from_row(&row, 2, 3).unwrap();

        assert!((out.logprob - (2.0 - lse)).abs() < 1e-6);
        // Top-3 sorted descending; tied logits keep ascending token-id order.
        let ids: Vec<u32> = out.top_logprobs.iter().map(|&(id, _)| id).collect();
        assert_eq!(ids, vec![1, 4, 2]);
        for &(id, lp) in &out.top_logprobs {
            assert!((lp - (row[id as usize].to_f32() - lse)).abs() < 1e-6);
        }
    }

    #[test]
    fn token_logprob_k_larger_than_vocab() {
        let row: Vec<bf16> = [0.5f32, -1.0].iter().map(|&v| bf16::from_f32(v)).collect();
        let out = token_logprob_from_row(&row, 0, 32).unwrap();
        assert_eq!(out.top_logprobs.len(), 2);
        assert_eq!(out.top_logprobs[0].0, 0);
        // log-softmax sums to 1 in probability space.
        let total: f64 = out
            .top_logprobs
            .iter()
            .map(|&(_, lp)| f64::from(lp).exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-6);
    }

    #[test]
    fn token_logprob_f32_input_and_guards() {
        // The Qwen f32 path plus the empty/out-of-range guards.
        let row = [0.0f32, 1.0, 2.0];
        let out = token_logprob_from_row(&row, 2, 2).unwrap();
        assert_eq!(out.top_logprobs[0].0, 2);
        assert!(token_logprob_from_row::<f32>(&[], 0, 1).is_none());
        assert!(token_logprob_from_row(&row, 9, 1).is_none());
    }
}
