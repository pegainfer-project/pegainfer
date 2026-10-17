//! Qwen3.8-Flash-Next QSA token-indexer block scoring.
//!
//! Turns the fused `index_qk_proj` output into one score per (query row,
//! visible micro-block); selection — top-k plus the unconditional tail — is a
//! later stage, deliberately not here. Frozen semantics live in
//! `csrc/qwen4/qwen4_qsa_indexer_score.cu` (transformers 5.16.1 `qwen4_exp`
//! `Qwen4ExpTextQSAIndexer.forward`); the geometry this wrapper accepts is the
//! checkpoint's own, held against the compiled constants before a launch.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::ops::checked_i32;
use crate::tensor::DeviceContext;
use crate::tensor::active_cu_stream;

/// The checkpoint's QSA indexer geometry. Frozen with the line: `pegainfer-qwen4`
/// serves one `qwen4_exp` text geometry, and the kernel is compiled for it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QsaIndexerGeometry {
    /// `indexer_n_heads`: query heads the indexer scores with.
    pub heads: usize,
    /// `indexer_kv_heads`. The reference requires exactly one, so the four
    /// query heads all read the same pooled key.
    pub kv_heads: usize,
    /// `indexer_head_dim`.
    pub head_dim: usize,
    /// `indexer_compress_ratio`: tokens per micro-block.
    pub compress_ratio: usize,
    /// Dims the partial RoPE rotates: `head_dim * partial_rotary_factor` of the
    /// *attention* head (256 * 0.25 = 64), which is under the indexer's 128.
    pub rotary_dim: usize,
    /// `rope_theta` of the checkpoint's text config.
    pub rope_theta: f32,
    /// `rms_norm_eps`, shared by both `(1 + w)` norms.
    pub rms_eps: f32,
}

impl QsaIndexerGeometry {
    /// The frozen Qwen3.8-Flash-Next text config's indexer.
    pub fn qwen38_flash_next() -> Self {
        Self {
            heads: 4,
            kv_heads: 1,
            head_dim: 128,
            compress_ratio: 4,
            rotary_dim: 64,
            rope_theta: 1.0e7,
            rms_eps: 1.0e-6,
        }
    }

    /// Columns of one fused projection row: the four query heads then the raw
    /// key head.
    pub fn fused_width(&self) -> usize {
        (self.heads + self.kv_heads) * self.head_dim
    }

    /// Blocks `pooled_keys` must hold for `tokens` tokens.
    pub fn max_blocks(&self, tokens: usize) -> usize {
        tokens / self.compress_ratio
    }

    /// Refuse a geometry the kernel was not compiled for. Called by
    /// [`qsa_indexer_block_scores_into`] before it forms a pointer, and public so
    /// a caller deriving geometry from a checkpoint's config can check it
    /// without a launch.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.heads == 4
                && self.kv_heads == 1
                && self.head_dim == 128
                && self.compress_ratio == 4
                && self.rotary_dim == 64,
            "QSA indexer geometry {self:?} is not the geometry \
             csrc/qwen4/qwen4_qsa_indexer_score.cu was compiled for"
        );
        // Bit equality, not a tolerance: the entry point refuses any other
        // theta, so a geometry carrying one must not pass as if it were compiled.
        ensure!(
            self.rope_theta.to_bits() == 1.0e7f32.to_bits(),
            "QSA indexer rope_theta {} is not the compiled 1e7",
            self.rope_theta
        );
        ensure!(
            self.rms_eps >= 0.0,
            "QSA indexer rms_norm_eps {} must be non-negative",
            self.rms_eps
        );
        Ok(())
    }
}

/// Score every complete micro-block of `fused_qk` for every query row.
///
/// `fused_qk` is the whole fused `index_qk_proj` output, one row per token:
/// `fused_width()` columns, the query heads first and the raw key head last.
/// The key array must start at token 0 — the micro-block grid starts at the
/// first visible token and this kernel reads it as token 0 (see the kernel's
/// grid contract).
///
/// `rows[j]` is each query row's position (the RoPE position and the
/// `fused_qk` row its query heads take) and `visible_len[j]` how many tokens
/// it can see, counted from token 0; causal text passes `rows[j] + 1`. Row `j`
/// gets `visible_len[j] / compress_ratio` scores at stride `score_cols`;
/// entries past that are left untouched, as the reference leaves them NaN — a
/// caller that wants to tell an unwritten entry from a real one has to poison
/// the buffer itself (the parity gate pre-fills a sentinel). Both arrays are
/// checked on the device: an out-of-range entry traps, so a broken caller fails
/// visibly instead of getting a plausible score for a row it never asked about.
///
/// `pooled_keys` is the workspace: `max_blocks(tokens)` rows of `head_dim`,
/// reused across calls of the same geometry.
pub fn qsa_indexer_block_scores_into(
    ctx: &DeviceContext,
    geometry: &QsaIndexerGeometry,
    fused_qk: &CudaSlice<bf16>,
    q_norm_weight: &CudaSlice<bf16>,
    k_norm_weight: &CudaSlice<bf16>,
    rows: &CudaSlice<i32>,
    visible_len: &CudaSlice<i32>,
    tokens: usize,
    scores: &mut CudaSlice<f32>,
    score_cols: usize,
    pooled_keys: &mut CudaSlice<bf16>,
) -> Result<()> {
    geometry.validate()?;
    ensure!(
        score_cols >= geometry.max_blocks(tokens),
        "QSA indexer score_cols {score_cols} cannot hold {} blocks",
        geometry.max_blocks(tokens)
    );
    let query_rows = rows.len();
    ensure!(
        visible_len.len() >= query_rows,
        "QSA indexer visible_len {} shorter than the {query_rows} query rows",
        visible_len.len()
    );
    ensure!(
        fused_qk.len() >= tokens.saturating_mul(geometry.fused_width()),
        "QSA indexer fused_qk holds {} bf16, needs {} for {tokens} tokens",
        fused_qk.len(),
        tokens.saturating_mul(geometry.fused_width())
    );
    ensure!(
        q_norm_weight.len() >= geometry.head_dim && k_norm_weight.len() >= geometry.head_dim,
        "QSA indexer norm weights must hold {} bf16",
        geometry.head_dim
    );
    ensure!(
        scores.len() >= query_rows.saturating_mul(score_cols),
        "QSA indexer scores holds {} f32, needs {} for {query_rows} x {score_cols}",
        scores.len(),
        query_rows.saturating_mul(score_cols)
    );
    ensure!(
        pooled_keys.len()
            >= geometry
                .max_blocks(tokens)
                .saturating_mul(geometry.head_dim),
        "QSA indexer pooled_keys workspace holds {} bf16, needs {}",
        pooled_keys.len(),
        geometry
            .max_blocks(tokens)
            .saturating_mul(geometry.head_dim)
    );
    if query_rows == 0 || tokens == 0 {
        return Ok(());
    }
    // The kernel takes extents as i32; checked before any pointer is formed.
    let query_rows = checked_i32(query_rows, "QSA indexer query rows")?;
    let tokens = checked_i32(tokens, "QSA indexer tokens")?;
    let score_cols = checked_i32(score_cols, "QSA indexer score columns")?;
    let fused_width = checked_i32(geometry.fused_width(), "QSA indexer fused width")?;
    let rotary_dim = checked_i32(geometry.rotary_dim, "QSA indexer rotary dim")?;

    let (fused_ptr, _fused_guard) = fused_qk.device_ptr(&ctx.stream);
    let (q_norm_ptr, _q_norm_guard) = q_norm_weight.device_ptr(&ctx.stream);
    let (k_norm_ptr, _k_norm_guard) = k_norm_weight.device_ptr(&ctx.stream);
    let (rows_ptr, _rows_guard) = rows.device_ptr(&ctx.stream);
    let (visible_ptr, _visible_guard) = visible_len.device_ptr(&ctx.stream);
    let (scores_ptr, _scores_guard) = scores.device_ptr_mut(&ctx.stream);
    let (pooled_ptr, _pooled_guard) = pooled_keys.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::qwen4_qsa_indexer_block_scores_cuda(
            fused_ptr as *const ffi::Half,
            q_norm_ptr as *const ffi::Half,
            k_norm_ptr as *const ffi::Half,
            rows_ptr as *const i32,
            visible_ptr as *const i32,
            scores_ptr as *mut f32,
            pooled_ptr as *mut ffi::Half,
            query_rows,
            tokens,
            score_cols,
            fused_width,
            rotary_dim,
            geometry.rope_theta,
            geometry.rms_eps,
            active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| anyhow::anyhow!("QSA indexer block score launch failed: {err}"))
}

#[cfg(test)]
mod tests {
    use super::QsaIndexerGeometry;

    /// The geometry is the one hand-off this stage has with the kernel's compiled
    /// constants, so a mismatch has to be refused before a pointer is formed.
    #[test]
    fn only_the_compiled_geometry_passes() {
        let compiled = QsaIndexerGeometry::qwen38_flash_next();
        compiled.validate().expect("the frozen geometry");
        assert_eq!(compiled.fused_width(), 640);
        assert_eq!(compiled.max_blocks(2055), 513);
        for (field, mutated) in [
            (
                "heads",
                QsaIndexerGeometry {
                    heads: 8,
                    ..compiled
                },
            ),
            (
                "kv_heads",
                QsaIndexerGeometry {
                    kv_heads: 2,
                    ..compiled
                },
            ),
            (
                "head_dim",
                QsaIndexerGeometry {
                    head_dim: 256,
                    ..compiled
                },
            ),
            (
                "compress_ratio",
                QsaIndexerGeometry {
                    compress_ratio: 2,
                    ..compiled
                },
            ),
            (
                "rotary_dim",
                QsaIndexerGeometry {
                    rotary_dim: 32,
                    ..compiled
                },
            ),
            (
                "rope_theta",
                QsaIndexerGeometry {
                    rope_theta: 1.0e6,
                    ..compiled
                },
            ),
            (
                "rms_eps",
                QsaIndexerGeometry {
                    rms_eps: -1.0e-6,
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
}
