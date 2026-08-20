//! Pre-allocated scratch buffers for Qwen3.5 prefill-only chunk-wise operators.

use anyhow::Result;
use cudarc::driver::CudaSlice;
use half::bf16;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::HiddenStates;

use super::config::Config35;

/// Scratch buffers for a single Qwen3.5 linear-attention chunk-wise GDR prefill call.
///
/// The first implementation target is intentionally narrow:
/// - batch size = 1
/// - fixed Qwen3.5 linear-attention shapes
/// - forward-only
/// - chunk_size = 64
///
/// Buffers are explicit because the chunk-wise path is naturally a multi-stage
/// pipeline rather than one opaque kernel launch.
pub struct GdrChunkwiseScratch35 {
    /// Chunk-local cumulative gate, fp32: [seq_len, num_value_heads]
    pub(crate) g_cumsum: CudaSlice<f32>,
    /// Beta values, fp32: [seq_len, num_value_heads]
    pub(crate) beta: CudaSlice<f32>,

    /// Expanded + normalized q in token-major layout: [seq_len, num_value_heads * key_dim]
    pub(crate) q_expanded: HiddenStates,
    /// Expanded + normalized k in token-major layout: [seq_len, num_value_heads * key_dim]
    pub(crate) k_expanded: HiddenStates,
    /// Raw v in token-major layout: [seq_len, num_value_heads * value_dim]
    pub(crate) v_raw: HiddenStates,

    /// Chunk attention matrix storage, fp32: [seq_len, num_value_heads, chunk_size]
    pub(crate) a_tril: CudaSlice<f32>,
    /// Inverse (I + A)^-1 in bf16: [seq_len, num_value_heads, chunk_size]
    pub(crate) a_inv: CudaSlice<bf16>,

    /// Prepared W tensor in token-major layout: [seq_len, num_value_heads * key_dim]
    pub(crate) w: HiddenStates,
    /// Prepared U tensor in token-major layout: [seq_len, num_value_heads * value_dim]
    pub(crate) u: HiddenStates,
    /// New value tensor consumed by chunk output stage: [seq_len, num_value_heads * value_dim]
    pub(crate) v_new: HiddenStates,

    /// Per-chunk recurrent state snapshots, fp32: [num_chunks, num_value_heads, key_dim, value_dim]
    pub(crate) chunk_state: CudaSlice<f32>,
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "crate-private verifier reuses this scratch before serving wiring"
        )
    )]
    max_seq_len: usize,
}

impl GdrChunkwiseScratch35 {
    pub(crate) const CHUNK_SIZE: usize = 64;

    pub(crate) fn new(ctx: &DeviceContext, config: &Config35, seq_len: usize) -> Result<Self> {
        Self::from_dims(
            ctx,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            seq_len,
        )
    }

    pub fn from_dims(
        ctx: &DeviceContext,
        num_value_heads: usize,
        key_dim: usize,
        value_dim: usize,
        seq_len: usize,
    ) -> Result<Self> {
        let kv_hidden_dim = num_value_heads * key_dim;
        let vv_hidden_dim = num_value_heads * value_dim;
        let num_chunks = seq_len.div_ceil(Self::CHUNK_SIZE);

        let g_cumsum: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads)
            .map_err(|e| anyhow::anyhow!("Alloc g_cumsum failed: {}", e))?;
        let beta: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads)
            .map_err(|e| anyhow::anyhow!("Alloc beta failed: {}", e))?;
        let a_tril: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads * Self::CHUNK_SIZE)
            .map_err(|e| anyhow::anyhow!("Alloc a_tril failed: {}", e))?;
        let a_inv: CudaSlice<bf16> = ctx
            .stream
            .alloc_zeros(seq_len * num_value_heads * Self::CHUNK_SIZE)
            .map_err(|e| anyhow::anyhow!("Alloc a_inv failed: {}", e))?;
        let chunk_state: CudaSlice<f32> = ctx
            .stream
            .alloc_zeros(num_chunks * num_value_heads * value_dim * key_dim)
            .map_err(|e| anyhow::anyhow!("Alloc chunk_state failed: {}", e))?;

        Ok(Self {
            g_cumsum,
            beta,
            q_expanded: HiddenStates::zeros(ctx, kv_hidden_dim, seq_len)?,
            k_expanded: HiddenStates::zeros(ctx, kv_hidden_dim, seq_len)?,
            v_raw: HiddenStates::zeros(ctx, vv_hidden_dim, seq_len)?,
            a_tril,
            a_inv,
            w: HiddenStates::zeros(ctx, kv_hidden_dim, seq_len)?,
            u: HiddenStates::zeros(ctx, vv_hidden_dim, seq_len)?,
            v_new: HiddenStates::zeros(ctx, vv_hidden_dim, seq_len)?,
            chunk_state,
            max_seq_len: seq_len,
        })
    }

    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "crate-private verifier reuses this scratch before serving wiring"
        )
    )]
    pub(crate) fn set_rows(&mut self, seq_len: usize) {
        assert!(
            seq_len <= self.max_seq_len,
            "Qwen3.5 GDR scratch rows {seq_len} exceeds capacity {}",
            self.max_seq_len
        );
        self.q_expanded.seq_len = seq_len;
        self.k_expanded.seq_len = seq_len;
        self.v_raw.seq_len = seq_len;
        self.w.seq_len = seq_len;
        self.u.seq_len = seq_len;
        self.v_new.seq_len = seq_len;
    }

    pub(crate) fn num_chunks(seq_len: usize) -> usize {
        seq_len.div_ceil(Self::CHUNK_SIZE)
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
    pub(crate) fn estimate_bytes(config: &Config35, max_seq_len: usize) -> usize {
        let num_vh = config.linear_num_value_heads;
        let key_dim = config.linear_key_head_dim;
        let val_dim = config.linear_value_head_dim;
        let chunk_sz = Self::CHUNK_SIZE;
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
        let intermediate = config.intermediate_size;

        // Shared: hidden_batch + normed + hidden_plus_attn + normed_for_mlp
        let shared_layer = hidden_dim * seq * 4;

        // Full attention: q_full(with gate) + k + v + attn_out + q_prepped
        let full_qkv = config.num_attention_heads * config.head_dim * 2;
        let full_kv = config.num_key_value_heads * config.head_dim;
        let full_out = config.num_attention_heads * config.head_dim;
        let full_attn_temps = (full_qkv + full_kv * 2 + full_out * 2) * seq;

        // MLP: gate_up_out + act_out (same peak footprint as separate gate/up)
        let mlp_temps = intermediate * seq * 3;

        let peak_layer = shared_layer + full_attn_temps.max(mlp_temps);
        let per_layer_bytes = peak_layer * 2; // bf16

        gdr_bytes + per_layer_bytes
    }
}
