//! Pre-allocated scratch buffers for Qwen3.5 prefill-only chunk-wise operators.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::GdrChunkwiseScratch35;

use super::config::Config35;
use super::config::LocalGeometry;

/// Outputs of the native, non-expanded GDN prepare kernel.
///
/// This buffer is intentionally separate from `GdrChunkwiseScratch35`: the
/// production Triton path below still requires value-head-expanded Q/K, while
/// the FlashInfer candidate consumes native Hq/Hk tensors directly.
pub(crate) struct GdnPrepareScratch35 {
    /// Normalized native Q, bf16 token-major `[T,Hq,D]`.
    pub(crate) q: HiddenStates,
    /// Normalized native K, bf16 token-major `[T,Hk,D]`.
    pub(crate) k: HiddenStates,
    /// Raw V, bf16 token-major `[T,Hv,D]`.
    pub(crate) v: HiddenStates,
    /// Per-token decay multiplier, fp32 `[T,Hv]` (not log/cumulative alpha).
    pub(crate) alpha: CudaSlice<f32>,
    /// Per-token beta, fp32 `[T,Hv]`.
    pub(crate) beta: CudaSlice<f32>,
}

impl GdnPrepareScratch35 {
    pub(crate) fn new(ctx: &DeviceContext, seq_len: usize) -> Result<Self> {
        anyhow::ensure!(seq_len > 0, "native GDN prepare requires T>=1");

        let geometry = pegainfer_kernels::ops::Qwen35GdnGeometry::PRODUCTION;

        Ok(Self {
            q: HiddenStates::zeros(ctx, geometry.h_q * geometry.head_dim, seq_len)?,
            k: HiddenStates::zeros(ctx, geometry.h_k * geometry.head_dim, seq_len)?,
            v: HiddenStates::zeros(ctx, geometry.h_v * geometry.head_dim, seq_len)?,
            alpha: ctx
                .stream
                .alloc_zeros(seq_len * geometry.h_v)
                .map_err(|e| anyhow::anyhow!("Alloc native GDN alpha failed: {e}"))?,
            beta: ctx
                .stream
                .alloc_zeros(seq_len * geometry.h_v)
                .map_err(|e| anyhow::anyhow!("Alloc native GDN beta failed: {e}"))?,
        })
    }
}

/// The chunk-wise GDR prefill scratch at this rank's local Qwen3.5 geometry.
pub(crate) fn gdr_chunkwise_scratch(
    ctx: &DeviceContext,
    config: &Config35,
    geometry: LocalGeometry,
    seq_len: usize,
) -> Result<GdrChunkwiseScratch35> {
    GdrChunkwiseScratch35::from_dims(
        ctx,
        geometry.local_linear_num_value_heads(),
        config.linear_key_head_dim,
        config.linear_value_head_dim,
        seq_len,
    )
}

/// Estimate peak GPU memory (bytes) for prefill scratch at a given seq_len.
///
/// Accounts for:
/// 1. GDR chunkwise scratch (persists across all linear attention layers)
/// 2. Per-layer transient peak — max of full-attention or MLP intermediates,
///    plus shared hidden-state buffers (temporaries freed between layers)
///
/// Direct-paged prefill writes full-attention K/V into the paged pool, so
/// HND KVCache staging buffers are no longer part of the prefill scratch.
pub(crate) fn prefill_scratch_bytes(
    config: &Config35,
    geometry: LocalGeometry,
    max_seq_len: usize,
) -> usize {
    let num_vh = geometry.local_linear_num_value_heads();
    let key_dim = config.linear_key_head_dim;
    let val_dim = config.linear_value_head_dim;
    let chunk_sz = GdrChunkwiseScratch35::CHUNK_SIZE;
    let num_chunks = max_seq_len.div_ceil(chunk_sz);
    let seq = max_seq_len;

    let kv_hidden = num_vh * key_dim;
    let vv_hidden = num_vh * val_dim;

    // 1. GDR scratch (bf16 = 2 bytes, f32 = 4 bytes)
    let gdr_bytes = {
        let f32_elems = seq * num_vh                            // g_cumsum
            + seq * num_vh                                      // beta
            + seq * num_vh * chunk_sz                           // a_tril
            + num_chunks * num_vh * val_dim * key_dim; // chunk_state
        let bf16_elems = seq * num_vh * chunk_sz                // a_inv
            + kv_hidden * seq                                   // q_expanded
            + kv_hidden * seq                                   // k_expanded
            + vv_hidden * seq                                   // v_raw
            + kv_hidden * seq                                   // w
            + vv_hidden * seq                                   // u
            + vv_hidden * seq; // v_new
        f32_elems * 4 + bf16_elems * 2
    };

    // 2. Per-layer transient peak (all bf16 = 2 bytes).
    //    Attention and MLP temps don't coexist — MLP runs after attention.
    let hidden_dim = config.hidden_size;
    let intermediate = geometry.local_intermediate_size();

    // Shared: hidden_batch + normed + hidden_plus_attn + normed_for_mlp
    let shared_layer = hidden_dim * seq * 4;

    // Full attention: q_full(with gate) + k + v + attn_out + q_prepped
    let full_qkv = geometry.local_full_attn_gated_q_dim();
    let full_kv = geometry.local_full_attn_kv_dim();
    let full_out = geometry.local_full_attn_q_dim();
    let full_attn_temps = (full_qkv + full_kv * 2 + full_out * 2) * seq;

    // MLP: gate_up_out + act_out (same peak footprint as separate gate/up)
    let mlp_temps = intermediate * seq * 3;

    let peak_layer = shared_layer + full_attn_temps.max(mlp_temps);
    let per_layer_bytes = peak_layer * 2; // bf16

    gdr_bytes + per_layer_bytes
}
