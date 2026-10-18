use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use super::indexer::GLM52_INDEXER_HEAD_DIM;
use crate::ffi;
use crate::tensor::DeviceContext;

const GLM52_INDEXER_ROPE_HALF: usize = 32;

/// Non-interleaved (half-split / NeoX-style) RoPE for the DSA indexer
/// q `[n_heads, head_dim]` and k `[head_dim]` (in-place). Applies RoPE
/// to the first `GLM52_INDEXER_ROPE_DIM` (=64) elements of each q head and
/// of k; the remaining 64 pass-through dimensions are left unchanged.
/// `cos`/`sin` are `[32]` (rope_dim / 2).
///
/// Aligned to the transformers reference (GlmMoeDsaIndexer.forward) which
/// uses `apply_rotary_pos_emb` (half-split / rotate_half convention). The
/// config flag `indexer_rope_interleave=true` is consumed by vllm but NOT
/// by transformers; the oracle runs on transformers, so we match its
/// non-interleaved convention.
pub fn glm52_indexer_rope_launch(
    ctx: &DeviceContext,
    q: &mut CudaSlice<bf16>,
    k: &mut CudaSlice<bf16>,
    n_heads: usize,
    tokens: usize,
    cos: &CudaSlice<bf16>,
    sin: &CudaSlice<bf16>,
) -> Result<()> {
    ensure!(n_heads > 0, "GLM5.2 indexer RoPE n_heads must be positive");
    ensure!(tokens > 0, "GLM5.2 indexer RoPE tokens must be positive");
    ensure!(
        q.len() >= tokens * n_heads * GLM52_INDEXER_HEAD_DIM,
        "GLM5.2 indexer RoPE q too small: have {}, need {}",
        q.len(),
        tokens * n_heads * GLM52_INDEXER_HEAD_DIM
    );
    ensure!(
        k.len() >= tokens * GLM52_INDEXER_HEAD_DIM,
        "GLM5.2 indexer RoPE k too small: have {}, need {}",
        k.len(),
        tokens * GLM52_INDEXER_HEAD_DIM
    );
    ensure!(
        cos.len() >= tokens * GLM52_INDEXER_ROPE_HALF
            && sin.len() >= tokens * GLM52_INDEXER_ROPE_HALF,
        "GLM5.2 indexer RoPE cos/sin must be >= tokens * {GLM52_INDEXER_ROPE_HALF}"
    );

    let (q_ptr, _q_guard) = q.device_ptr_mut(&ctx.stream);
    let (k_ptr, _k_guard) = k.device_ptr_mut(&ctx.stream);
    let (cos_ptr, _cos_guard) = cos.device_ptr(&ctx.stream);
    let (sin_ptr, _sin_guard) = sin.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::glm52_indexer_rope_cuda(
            q_ptr as *mut ffi::Half,
            k_ptr as *mut ffi::Half,
            n_heads as i32,
            tokens as i32,
            cos_ptr as *const ffi::Half,
            sin_ptr as *const ffi::Half,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 indexer RoPE launch failed: {err}"))
}
