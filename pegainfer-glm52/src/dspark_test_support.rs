//! Test-only `Glm52DsparkModel` constructors: the zero-weight synthetic model
//! (perf smokes) and deterministic pseudo-random weights (parity/isolation
//! gates). A child module of `dspark` so the private fields stay private.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use pegainfer_core::rope::RopeTableSpec;
use pegainfer_core::rope::precompute_rope;
use pegainfer_kernels::tensor::DeviceContext;
use pegainfer_kernels::tensor::DeviceMatrix;
use pegainfer_kernels::tensor::DeviceVec;

use super::*;

impl Glm52DsparkModel {
    /// Zero-weight model at the checkpoint's exact geometry, for perf smoke
    /// tests without the checkpoint — GPU kernel time is value-independent.
    #[cfg(test)]
    pub(crate) fn synthetic(ctx: &DeviceContext, cache_len: usize) -> Result<Self> {
        let mat = |rows: usize, cols: usize| -> Result<DeviceMatrix> {
            Ok(DeviceMatrix {
                data: ctx.stream.alloc_zeros(rows * cols)?,
                rows,
                cols,
            })
        };
        let mut layers = Vec::with_capacity(DSPARK_LAYERS);
        for _ in 0..DSPARK_LAYERS {
            layers.push(DsparkLayer {
                input_ln: DeviceVec::zeros(ctx, GLM52_HIDDEN)?,
                qkv: mat(3 * DSPARK_QKV_DIM, GLM52_HIDDEN)?,
                o_proj: mat(GLM52_HIDDEN, DSPARK_QKV_DIM)?,
                q_norm: DeviceVec::zeros(ctx, DSPARK_HEAD_DIM)?,
                k_norm: DeviceVec::zeros(ctx, DSPARK_HEAD_DIM)?,
                post_ln: DeviceVec::zeros(ctx, GLM52_HIDDEN)?,
                gate_up: mat(2 * DSPARK_INTER, GLM52_HIDDEN)?,
                down: mat(GLM52_HIDDEN, DSPARK_INTER)?,
            });
        }
        let (cos_cache, sin_cache) = precompute_rope(
            ctx,
            &RopeTableSpec {
                rotary_dim: DSPARK_HEAD_DIM,
                frequency_dim: DSPARK_HEAD_DIM,
                max_seq_len: cache_len,
                theta: DSPARK_ROPE_THETA,
            },
        )?;
        Ok(Self {
            layers,
            norm: DeviceVec::zeros(ctx, GLM52_HIDDEN)?,
            hidden_norm: DeviceVec::zeros(ctx, GLM52_HIDDEN)?,
            fc: mat(GLM52_HIDDEN, GLM52_DSPARK_CONTEXT_DIM)?,
            markov_w1: mat(GLM52_VOCAB, DSPARK_MARKOV_RANK)?,
            markov_w2: mat(GLM52_VOCAB, DSPARK_MARKOV_RANK)?,
            cos_cache,
            sin_cache,
            cache_len,
        })
    }

    /// Deterministic pseudo-random weights over the synthetic model, for
    /// graph-vs-eager parity tests: kernel *timing* is value-independent, but
    /// draft-token parity needs non-degenerate logits (all-zero weights make
    /// every argmax a trivial index-0 tie).
    #[cfg(test)]
    pub(crate) fn randomize_for_test(&mut self, ctx: &DeviceContext) -> Result<()> {
        fn fill(
            ctx: &DeviceContext,
            buf: &mut CudaSlice<half::bf16>,
            seed: u32,
            scale: f32,
            offset: f32,
        ) -> Result<()> {
            let mut state = seed | 1;
            let host: Vec<half::bf16> = (0..buf.len())
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    half::bf16::from_f32(
                        ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * scale + offset,
                    )
                })
                .collect();
            ctx.stream.memcpy_htod(&host, buf)?;
            Ok(())
        }
        let mut seed = 0x5eed_u32;
        let mut next = || {
            seed = seed.wrapping_add(0x9e37_79b9);
            seed
        };
        for layer in &mut self.layers {
            fill(ctx, &mut layer.qkv.data, next(), 0.02, 0.0)?;
            fill(ctx, &mut layer.o_proj.data, next(), 0.02, 0.0)?;
            fill(ctx, &mut layer.gate_up.data, next(), 0.02, 0.0)?;
            fill(ctx, &mut layer.down.data, next(), 0.02, 0.0)?;
            fill(ctx, &mut layer.input_ln.data, next(), 0.1, 1.0)?;
            fill(ctx, &mut layer.post_ln.data, next(), 0.1, 1.0)?;
            fill(ctx, &mut layer.q_norm.data, next(), 0.1, 1.0)?;
            fill(ctx, &mut layer.k_norm.data, next(), 0.1, 1.0)?;
        }
        fill(ctx, &mut self.norm.data, next(), 0.1, 1.0)?;
        fill(ctx, &mut self.hidden_norm.data, next(), 0.1, 1.0)?;
        fill(ctx, &mut self.fc.data, next(), 0.02, 0.0)?;
        fill(ctx, &mut self.markov_w1.data, next(), 0.5, 0.0)?;
        fill(ctx, &mut self.markov_w2.data, next(), 0.5, 0.0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::accept_prefix_match;

    #[test]
    fn accepts_full_run_plus_bonus() {
        assert_eq!(
            accept_prefix_match(&[10, 11, 12], &[10, 11, 12, 13]),
            vec![10, 11, 12, 13]
        );
    }

    #[test]
    fn accepts_prefix_then_correction() {
        assert_eq!(
            accept_prefix_match(&[10, 11, 99], &[10, 11, 22, 33]),
            vec![10, 11, 22]
        );
    }

    #[test]
    fn rejects_first_draft_commits_the_correction() {
        assert_eq!(accept_prefix_match(&[10, 11, 12], &[7, 8, 9, 10]), vec![7]);
    }

    #[test]
    fn empty_proposal_commits_the_model_token() {
        assert_eq!(accept_prefix_match(&[], &[42]), vec![42]);
    }
}
