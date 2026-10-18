//! Smoke test for the DSA indexer forward — verifies kernel launches and
//! output shape, not correctness (that requires the oracle gate + checkpoint).
//!
//! Run (H200 + DeepGEMM env):
//! ```text
//! PEGAINFER_DEEPGEMM_ROOT=pegainfer-kernels/third_party/DeepGEMM/deep_gemm \
//! CUDA_HOME=/usr/local/cuda \
//!   cargo test --release -p pegainfer-glm52 --features glm52 --lib indexer_smoke -- --nocapture
//! ```

use anyhow::Result;
use half::bf16;
use pegainfer_kernels::ops::GLM52_INDEXER_TOPK;
use pegainfer_kernels::ops::Glm52IndexerCacheLayout;
use pegainfer_kernels::tensor::DeviceContext;

use crate::config::GLM52_HIDDEN;
use crate::config::GLM52_INDEX_HEAD_DIM;
use crate::config::GLM52_Q_LORA_RANK;
use crate::config::GLM52_ROPE_HALF;
use crate::fp8::FP8_BLOCK;
use crate::fp8::Glm52ProjBytes;
use crate::indexer::Glm52IndexerLayerWeights;
use crate::indexer::glm52_indexer_forward;
use crate::model::INDEX_CACHE_BLOCK;
use crate::model::NUM_SMS;
use crate::rows::Rows;

const HIDDEN: usize = GLM52_HIDDEN;
const Q_LORA: usize = GLM52_Q_LORA_RANK;
const INDEX_HEADS: usize = 32;
const INDEX_HEAD_DIM: usize = GLM52_INDEX_HEAD_DIM;
const CACHE_BLOCKS: usize = 4;
const MAX_MODEL_LEN: usize = 512;

fn deepgemm_env_ready() -> bool {
    std::env::var("PEGAINFER_DEEPGEMM_ROOT").is_ok() && std::env::var("CUDA_HOME").is_ok()
}

fn synthetic_proj(n: usize, k: usize) -> Glm52ProjBytes<'static> {
    fn zeroed_static(len: usize) -> &'static [u8] {
        Box::leak(vec![0u8; len].into_boxed_slice())
    }
    let weight = zeroed_static(n * k);
    let scale_len = n.div_ceil(FP8_BLOCK) * k.div_ceil(FP8_BLOCK);
    let scale: &'static [u8] = Box::leak(
        (0..scale_len)
            .flat_map(|_| 1.0f32.to_le_bytes())
            .collect::<Vec<u8>>()
            .into_boxed_slice(),
    );
    Glm52ProjBytes {
        weight,
        scale,
        n,
        k,
    }
}

#[test]
#[ignore = "requires H200 (SM90) — TRTLLM FP8 blockscale GEMM is SM90-only"]
fn indexer_smoke() -> Result<()> {
    if !deepgemm_env_ready() {
        eprintln!("skip: set PEGAINFER_DEEPGEMM_ROOT + CUDA_HOME to run indexer smoke test");
        return Ok(());
    }

    let ctx = DeviceContext::new()?;

    // ---- synthetic weights (all zeros, scale=1.0) ----
    let wq_b = synthetic_proj(INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA);
    let wk = synthetic_proj(INDEX_HEAD_DIM, HIDDEN);
    // weights_proj is bf16 [32, 6144] (not fp8 in checkpoint)
    let weights_proj_bf16 = vec![0u8; INDEX_HEADS * HIDDEN * 2];
    let k_norm_w = vec![0u8; INDEX_HEAD_DIM * 2]; // bf16
    let k_norm_b = vec![0u8; INDEX_HEAD_DIM * 2]; // bf16

    let w = Glm52IndexerLayerWeights::from_host(
        &ctx,
        &wq_b,
        &wk,
        &weights_proj_bf16,
        &k_norm_w,
        &k_norm_b,
    )?;

    // ---- inputs ----
    let hidden: Rows<HIDDEN> = Rows::zeros(&ctx, 1)?;
    let q_resid: Rows<Q_LORA> = Rows::zeros(&ctx, 1)?;
    let cos = ctx.stream.alloc_zeros::<bf16>(GLM52_ROPE_HALF)?;
    let sin = ctx.stream.alloc_zeros::<bf16>(GLM52_ROPE_HALF)?;

    // ---- cache setup ----
    let cache_layout = Glm52IndexerCacheLayout {
        cache_blocks: CACHE_BLOCKS,
        cache_block_size: INDEX_CACHE_BLOCK,
        cache_layer_offset_bytes: 0,
        cache_block_stride_bytes: INDEX_CACHE_BLOCK * (GLM52_INDEX_HEAD_DIM + 4),
    };
    let cache_bytes = cache_layout.min_cache_bytes()?;
    let mut index_k_cache = ctx.stream.alloc_zeros::<u8>(cache_bytes)?;

    let slot_mapping = ctx.stream.alloc_zeros::<i64>(1)?;
    let block_table_host: Vec<i32> = (0..CACHE_BLOCKS as i32).collect();
    let mut block_table = ctx.stream.alloc_zeros::<i32>(CACHE_BLOCKS)?;
    ctx.stream
        .memcpy_htod(&block_table_host, &mut block_table)?;
    let seq_lens_host = vec![128i32];
    let mut seq_lens = ctx.stream.alloc_zeros::<i32>(1)?;
    ctx.stream.memcpy_htod(&seq_lens_host, &mut seq_lens)?;

    // ---- forward ----
    let topk = glm52_indexer_forward(
        &ctx,
        &w,
        &hidden,
        &q_resid,
        &cos,
        &sin,
        &mut index_k_cache,
        cache_layout,
        &slot_mapping,
        &block_table,
        &seq_lens,
        NUM_SMS,
        MAX_MODEL_LEN,
    )?;

    // ---- verify shape ----
    assert_eq!(
        topk.len(),
        GLM52_INDEXER_TOPK,
        "topk_indices must be [2048]"
    );

    let topk_host = ctx.stream.clone_dtoh(&topk)?;
    for &v in &topk_host {
        assert!(v >= -1, "topk slot {v} < -1 (corrupted)");
        assert!(
            v < (CACHE_BLOCKS * INDEX_CACHE_BLOCK) as i32,
            "topk slot {v} out of range"
        );
    }

    eprintln!(
        "indexer smoke: topk[0..8] = {:?}",
        &topk_host[..8.min(topk_host.len())]
    );
    Ok(())
}
