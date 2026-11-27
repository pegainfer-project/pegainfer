//! Paged DSpark draft K/V addressed by the target KV's page ids (#590).
//!
//! A prefix hit is a suffix-only prefill: the drafter's context rows for the
//! cached prefix are never projected. Persisting the projected draft K/V under
//! the target's page ids turns a hit into a gather instead of a cold start,
//! which is what lets DSpark and prefix caching coexist.
//!
//! The draft row is `5 layers x 2 x 4096 bf16 = 81,920 B/token`, versus a
//! 656 B/token MLA target row (~125x), so this pool can never mirror the target
//! pool. It is bounded by a startup budget (a fraction of the KV-eligible VRAM
//! by default, `GLM52_DSPARK_DRAFT_POOL_BYTES` to pin it); blocks that do not
//! fit are simply not tagged, and a hit whose blocks are untagged keeps its
//! target KV and plain-decodes.

use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::HiddenStates;

use crate::dspark::DSPARK_LAYERS;
use crate::dspark::DSPARK_QKV_DIM;
use crate::model::GLM52_MODEL_LEN_ALIGN;

/// Draft page size: the same 64-token page the target pool uses, so a draft
/// row is addressed by the target's page id directly.
pub(crate) const DRAFT_POOL_PAGE: usize = GLM52_MODEL_LEN_ALIGN;

/// One draft K/V row: every draft layer's K and V, bf16.
pub(crate) const DRAFT_BYTES_PER_TOKEN: usize =
    DSPARK_LAYERS * 2 * DSPARK_QKV_DIM * size_of::<bf16>();

pub(crate) const DRAFT_BYTES_PER_BLOCK: usize = DRAFT_BYTES_PER_TOKEN * DRAFT_POOL_PAGE;

/// Default share of the KV-eligible VRAM the draft pool may take.
pub(crate) const DRAFT_POOL_DEFAULT_RATIO: f64 = 0.10;
const DRAFT_POOL_MAX_RATIO: f64 = 0.5;

pub(crate) const ENV_RATIO: &str = "GLM52_DSPARK_DRAFT_POOL_RATIO";
pub(crate) const ENV_BYTES: &str = "GLM52_DSPARK_DRAFT_POOL_BYTES";

/// Resolve the draft-pool byte budget from the startup knobs. `pool_base_bytes`
/// is the VRAM the KV pool and its arenas may use; the draft pool is carved out
/// of it before the context cap is derived, so enabling persistence always
/// costs context (never silent overcommit).
pub(crate) fn resolve_budget_bytes(pool_base_bytes: usize) -> Result<usize> {
    let ratio = match std::env::var_os(ENV_RATIO) {
        None => DRAFT_POOL_DEFAULT_RATIO,
        Some(raw) => {
            let raw = raw.to_string_lossy();
            match raw.trim().parse::<f64>() {
                Ok(value) if (0.0..=DRAFT_POOL_MAX_RATIO).contains(&value) => value,
                _ => anyhow::bail!(
                    "{ENV_RATIO}={raw:?} not recognized (a fraction, 0 <= N <= {DRAFT_POOL_MAX_RATIO})"
                ),
            }
        }
    };
    let budget = match std::env::var_os(ENV_BYTES) {
        None => (pool_base_bytes as f64 * ratio) as usize,
        Some(raw) => {
            let raw = raw.to_string_lossy();
            match raw.trim().parse::<usize>() {
                Ok(value) => value,
                Err(_) => anyhow::bail!("{ENV_BYTES}={raw:?} not recognized (bytes, N >= 0)"),
            }
        }
    };
    Ok(budget)
}

/// Whole pages the budget covers, clamped by the target pool's block count.
/// Zero disables the pool (and therefore DSpark prefix caching).
pub(crate) fn pool_blocks(budget_bytes: usize, target_blocks: usize) -> usize {
    (budget_bytes / DRAFT_BYTES_PER_BLOCK).min(target_blocks)
}

/// Bytes one allocated pool of `blocks` pages holds (charged at launch).
pub(crate) fn pool_bytes(blocks: usize) -> usize {
    blocks * DRAFT_BYTES_PER_BLOCK
}

/// Pool slot of absolute `position` under target `page`. One definition for
/// the admission restore list and the round's scatter list.
pub(crate) fn pool_slot(page: i32, position: usize) -> i32 {
    page * DRAFT_POOL_PAGE as i32 + (position % DRAFT_POOL_PAGE) as i32
}

/// Rows `[start, materialized)` that the target can address *and* the draft
/// pool owns. The pool may be smaller than the target pool; stopping before
/// the first foreign page keeps every scatter write inside the tensor.
pub(crate) fn scatter_cover(
    pages: &[i32],
    start: usize,
    materialized: usize,
    pool_blocks: usize,
) -> usize {
    let limit = materialized.min(pages.len() * DRAFT_POOL_PAGE);
    for position in start..limit {
        let page = pages[position / DRAFT_POOL_PAGE];
        if page < 0 || page as usize >= pool_blocks {
            return position;
        }
    }
    limit
}

/// One `[kv_dim, blocks * page]` tensor per draft layer per K/V.
pub(crate) struct Glm52DsparkKvPool {
    pub(crate) k: Vec<HiddenStates>,
    pub(crate) v: Vec<HiddenStates>,
    total_blocks: usize,
}

impl Glm52DsparkKvPool {
    pub(crate) fn new(ctx: &DeviceContext, total_blocks: usize) -> Result<Self> {
        ensure!(
            total_blocks > 0,
            "DSpark draft KV pool needs at least one block"
        );
        let slots = total_blocks
            .checked_mul(DRAFT_POOL_PAGE)
            .ok_or_else(|| anyhow::anyhow!("DSpark draft KV pool slot count overflow"))?;
        let mut k = Vec::with_capacity(DSPARK_LAYERS);
        let mut v = Vec::with_capacity(DSPARK_LAYERS);
        for _ in 0..DSPARK_LAYERS {
            k.push(HiddenStates::zeros(ctx, DSPARK_QKV_DIM, slots)?);
            v.push(HiddenStates::zeros(ctx, DSPARK_QKV_DIM, slots)?);
        }
        Ok(Self { k, v, total_blocks })
    }

    /// Rows the pool owns (the worker's scatter locations must stay below it).
    pub(crate) fn slots(&self) -> usize {
        self.total_blocks * DRAFT_POOL_PAGE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_size_is_the_dspark_geometry() {
        // 5 layers x (K+V) x 4096 x bf16.
        assert_eq!(DRAFT_BYTES_PER_TOKEN, 81_920);
        assert_eq!(DRAFT_BYTES_PER_BLOCK, 81_920 * 64);
    }

    #[test]
    fn budget_converts_to_whole_pages_and_clamps_to_the_target_pool() {
        assert_eq!(pool_blocks(0, 100), 0);
        assert_eq!(pool_blocks(DRAFT_BYTES_PER_BLOCK - 1, 100), 0);
        assert_eq!(pool_blocks(DRAFT_BYTES_PER_BLOCK, 100), 1);
        assert_eq!(pool_blocks(DRAFT_BYTES_PER_BLOCK * 7 + 5, 100), 7);
        assert_eq!(pool_blocks(DRAFT_BYTES_PER_BLOCK * 7, 3), 3);
        assert_eq!(pool_bytes(7), 7 * DRAFT_BYTES_PER_BLOCK);
    }

    #[test]
    fn scatter_cover_stops_at_materialized_pages_and_foreign_pages() {
        // Pages 1 and 9 belong to the pool; page 20 does not.
        let pages = [1i32, 9, 20];
        assert_eq!(
            scatter_cover(&pages, 0, 129, 10),
            128,
            "stop before page 20"
        );
        assert_eq!(scatter_cover(&pages, 0, 100, 10), 100, "materialized bound");
        assert_eq!(
            scatter_cover(&pages, 64, 200, 10),
            128,
            "start inside page 1"
        );
        assert_eq!(
            scatter_cover(&pages, 128, 200, 10),
            128,
            "already at the edge"
        );
        assert_eq!(scatter_cover(&pages, 0, 200, 30), 192, "page list bound");
        assert_eq!(scatter_cover(&[], 0, 10, 10), 0, "no pages");
    }

    #[test]
    fn slot_math_matches_the_target_page_table() {
        // Page 9, position 130 -> 9 * 64 + 2.
        assert_eq!(pool_slot(1, 0), 64);
        assert_eq!(pool_slot(9, 130), 9 * 64 + 2);
        assert_eq!(pool_slot(3, 64), 3 * 64);
    }
}
