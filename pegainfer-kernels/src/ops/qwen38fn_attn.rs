//! Qwen3.8-Flash-Next QSA sparse full-attention.
//!
//! The forward this wraps is the middle of the three units the QSA bring-up plan
//! splits the layer into: the indexer scores and selects, this consumes the
//! selection as a visibility mask and produces the gated attention output that
//! `o_proj` reads. The mask is therefore an input, never derived here.

use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

/// The attention geometry the QSA kernels are compiled for. The kernel body
/// carries these as constants and refuses a call whose rotary geometry disagrees
/// with them, so this struct exists to hold the same numbers on the Rust side:
/// callers size their buffers with it, and the parity gate holds it against the
/// frozen config the goldens were dumped from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QsaAttnGeometry {
    /// `num_attention_heads`.
    pub heads: usize,
    /// `num_key_value_heads`; head `h` reads KV head `h / group()`.
    pub kv_heads: usize,
    /// `head_dim`, and the width the norms and the RoPE split work on.
    pub head_dim: usize,
    /// `head_dim * partial_rotary_factor`: the rotated prefix of a head.
    pub rotary_dim: usize,
    /// `rope_parameters.rope_theta`.
    pub rope_theta: f32,
    /// `rms_norm_eps`.
    pub rms_eps: f32,
}

impl QsaAttnGeometry {
    /// The longest key array one call accepts: a row of float32 scores lives in
    /// dynamic shared memory, and a longer context needs the streaming softmax
    /// rather than a longer row.
    ///
    /// The kernel carries the same budget as a compiled constant and refuses past
    /// it, so a drift between the two fails closed either way — and the parity
    /// gate holds them together, probing the kernel's copy with this constant and
    /// running one call at exactly it.
    pub const MAX_TOKENS: usize = 8192;

    /// The frozen Qwen3.8-Flash-Next text geometry.
    pub const fn qwen38_flash_next() -> Self {
        Self {
            heads: 24,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            rope_theta: 1.0e7,
            rms_eps: 1.0e-6,
        }
    }

    /// Query heads per KV head (`num_key_value_groups`).
    pub const fn group(&self) -> usize {
        self.heads / self.kv_heads
    }

    /// One fused `q_proj` row: every head's query dims followed by that head's
    /// output-gate dims.
    pub const fn query_width(&self) -> usize {
        self.heads * self.head_dim * 2
    }

    /// One `k_proj` or `v_proj` row.
    pub const fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// One `o_proj` input row, head-major.
    pub const fn out_width(&self) -> usize {
        self.heads * self.head_dim
    }
}

/// Replay the QSA sparse-attention forward over one layer's captured
/// activations and write both of the reference's observable outputs.
///
/// `queries` holds `query_rows` fused `q_proj` rows ([`QsaAttnGeometry::query_width`]
/// wide); `rows` gives the token each of those rows belongs to and `positions`
/// the position of every one of the `tokens` tokens, so the evaluated rows may be
/// any subset in any order. `mask` is the caller's visibility, one byte per
/// (query row, token), non-zero where visible — the indexer's selection AND
/// causality, which this call never re-derives. `normed_keys` is the key pass's
/// workspace and must hold `tokens * geometry.kv_width()` bf16.
///
/// Every element of `attn_weights` (`heads * query_rows * tokens`, head-major as
/// the reference returns them) and of `attn_out` (`query_rows * out_width`, the
/// input to `o_proj`) is written.
pub fn qsa_attn_forward_into(
    ctx: &DeviceContext,
    geometry: &QsaAttnGeometry,
    queries: &CudaSlice<bf16>,
    keys: &CudaSlice<bf16>,
    values: &CudaSlice<bf16>,
    mask: &CudaSlice<u8>,
    rows: &CudaSlice<i32>,
    positions: &CudaSlice<i32>,
    q_norm_weight: &CudaSlice<bf16>,
    k_norm_weight: &CudaSlice<bf16>,
    attn_weights: &mut CudaSlice<bf16>,
    attn_out: &mut CudaSlice<bf16>,
    normed_keys: &mut CudaSlice<bf16>,
) -> Result<()> {
    let query_rows = rows.len();
    let tokens = positions.len();
    let query_width = geometry.query_width();
    let kv_width = geometry.kv_width();
    ensure!(
        queries.len() >= query_rows * query_width,
        "qsa_attn_forward_into queries hold {} bf16, need {query_rows} rows of {query_width}",
        queries.len()
    );
    ensure!(
        keys.len() >= tokens * kv_width && values.len() >= tokens * kv_width,
        "qsa_attn_forward_into keys {} / values {} bf16, need {tokens} rows of {kv_width}",
        keys.len(),
        values.len()
    );
    ensure!(
        mask.len() >= query_rows * tokens,
        "qsa_attn_forward_into mask holds {} bytes, need {query_rows}x{tokens}",
        mask.len()
    );
    ensure!(
        q_norm_weight.len() >= geometry.head_dim && k_norm_weight.len() >= geometry.head_dim,
        "qsa_attn_forward_into norm weights {} / {} bf16, need {}",
        q_norm_weight.len(),
        k_norm_weight.len(),
        geometry.head_dim
    );
    ensure!(
        attn_weights.len() >= geometry.heads * query_rows * tokens,
        "qsa_attn_forward_into attn_weights holds {} bf16, need {}x{query_rows}x{tokens}",
        attn_weights.len(),
        geometry.heads
    );
    ensure!(
        attn_out.len() >= query_rows * geometry.out_width(),
        "qsa_attn_forward_into attn_out holds {} bf16, need {query_rows} rows of {}",
        attn_out.len(),
        geometry.out_width()
    );
    ensure!(
        normed_keys.len() >= tokens * kv_width,
        "qsa_attn_forward_into normed_keys workspace holds {} bf16, need {tokens} rows of {kv_width}",
        normed_keys.len()
    );
    ensure!(
        tokens <= QsaAttnGeometry::MAX_TOKENS,
        "qsa_attn_forward_into: {tokens} tokens exceed the {} the score row's \
         shared-memory budget allows; a longer context needs the streaming softmax",
        QsaAttnGeometry::MAX_TOKENS
    );

    let (q_ptr, _gq) = queries.device_ptr(&ctx.stream);
    let (k_ptr, _gk) = keys.device_ptr(&ctx.stream);
    let (v_ptr, _gv) = values.device_ptr(&ctx.stream);
    let (m_ptr, _gm) = mask.device_ptr(&ctx.stream);
    let (r_ptr, _gr) = rows.device_ptr(&ctx.stream);
    let (p_ptr, _gp) = positions.device_ptr(&ctx.stream);
    let (qn_ptr, _gqn) = q_norm_weight.device_ptr(&ctx.stream);
    let (kn_ptr, _gkn) = k_norm_weight.device_ptr(&ctx.stream);
    let (w_ptr, _gw) = attn_weights.device_ptr_mut(&ctx.stream);
    let (o_ptr, _go) = attn_out.device_ptr_mut(&ctx.stream);
    let (nk_ptr, _gnk) = normed_keys.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::qwen38fn_qsa_attn_forward_cuda(
            q_ptr as *const ffi::Half,
            k_ptr as *const ffi::Half,
            v_ptr as *const ffi::Half,
            m_ptr as *const u8,
            r_ptr as *const i32,
            p_ptr as *const i32,
            qn_ptr as *const ffi::Half,
            kn_ptr as *const ffi::Half,
            w_ptr as *mut ffi::Half,
            o_ptr as *mut ffi::Half,
            nk_ptr as *mut ffi::Half,
            query_rows as i32,
            tokens as i32,
            geometry.rotary_dim as i32,
            geometry.rope_theta,
            geometry.rms_eps,
            crate::tensor::active_cu_stream(ctx),
        )
    };
    result.result()?;
    Ok(())
}
