//! Kimi-K3 chunked-prefill MLA attention over FlashMLA's SM100 dense FMHA.
//!
//! The recipe is vLLM's chunked-prefill MLA path with the workspace covering
//! the whole context: gather the cached latent rows (the chunk's own tokens
//! are already appended), expand them through `kv_b` into per-head K/V, and
//! one bottom-right-aligned causal call attends the chunk's queries over
//! `[context | chunk]`. The paged latent stays the only persistent storage;
//! everything expanded here lives in per-chunk scratch.
//!
//! Layout conventions are baked into these wrappers: q/k are `[t, heads, 192]`
//! bf16 contiguous (per-head `nope | rope`, never rotated — K3 is NoPE), the
//! kv_b expansion is `[t, heads, 256]` (per-head `nope | value`) and V is
//! handed to the kernel as a strided view into it, out is `[t, heads, 128]`
//! bf16 contiguous.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::ffi::Half;
use crate::tensor::DeviceContext;

/// The cached latent row: post-norm kv latent | shared per-token rope half.
pub const K3_MLA_PREFILL_LATENT: usize = 512;
/// The shared per-token rope width.
pub const K3_MLA_PREFILL_ROPE: usize = 64;
/// One cached row, elements.
pub const K3_MLA_PREFILL_ROW: usize = K3_MLA_PREFILL_LATENT + K3_MLA_PREFILL_ROPE;
/// Per-head QK width (`nope | rope`).
pub const K3_MLA_PREFILL_QK: usize = 192;
/// Per-head V width, and the nope half of QK.
pub const K3_MLA_PREFILL_V: usize = 128;
/// Per-head kv_b expansion width (`nope | value`).
pub const K3_MLA_PREFILL_NV: usize = K3_MLA_PREFILL_V + K3_MLA_PREFILL_V;

/// Gather one sequence's cached latent rows into dense `[t, 512]` latent and
/// `[t, 64]` rope buffers. `table` is the sequence's block-table row;
/// `page_stride` and `layer_offset` are the slab's page walk stride and this
/// MLA layer's row shift inside a page, both in elements.
#[allow(clippy::too_many_arguments)]
pub fn k3_mla_prefill_gather_launch(
    ctx: &DeviceContext,
    t_total: usize,
    slab: &CudaSlice<bf16>,
    table: &CudaSlice<i32>,
    page_stride: usize,
    layer_offset: usize,
    latent_out: &mut CudaSlice<bf16>,
    rope_out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(t_total > 0, "K3 MLA prefill gather got an empty span");
    ensure!(
        latent_out.len() >= t_total * K3_MLA_PREFILL_LATENT
            && rope_out.len() >= t_total * K3_MLA_PREFILL_ROPE,
        "K3 MLA prefill gather outputs too small for t={t_total}: latent {}, rope {}",
        latent_out.len(),
        rope_out.len()
    );
    ensure!(
        table.len() >= t_total.div_ceil(64),
        "K3 MLA prefill gather table row of {} pages cannot span {t_total} tokens",
        table.len()
    );
    let (slab_ptr, _slab_guard) = slab.device_ptr(&ctx.stream);
    let (table_ptr, _table_guard) = table.device_ptr(&ctx.stream);
    let (latent_ptr, _latent_guard) = latent_out.device_ptr_mut(&ctx.stream);
    let (rope_ptr, _rope_guard) = rope_out.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_mla_prefill_gather(
            slab_ptr as *const Half,
            table_ptr as *const i32,
            page_stride as i64,
            layer_offset as i64,
            t_total as i32,
            latent_ptr as *mut Half,
            rope_ptr as *mut Half,
            ctx.stream.cu_stream(),
        )
    };
    rc.result()
        .map_err(|error| anyhow!("K3 MLA prefill gather (t={t_total}): {error}"))
}

/// Assemble the per-head K rows `[t, heads, 192]` from the kv_b expansion's
/// nope halves and the shared rope broadcast across heads.
pub fn k3_mla_prefill_expand_k_launch(
    ctx: &DeviceContext,
    t_total: usize,
    heads: usize,
    nope_v: &CudaSlice<bf16>,
    rope: &CudaSlice<bf16>,
    k_out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(t_total > 0, "K3 MLA prefill expand got an empty span");
    ensure!(
        nope_v.len() >= t_total * heads * K3_MLA_PREFILL_NV
            && rope.len() >= t_total * K3_MLA_PREFILL_ROPE
            && k_out.len() >= t_total * heads * K3_MLA_PREFILL_QK,
        "K3 MLA prefill expand buffers too small for t={t_total}, heads={heads}"
    );
    let (nv_ptr, _nv_guard) = nope_v.device_ptr(&ctx.stream);
    let (rope_ptr, _rope_guard) = rope.device_ptr(&ctx.stream);
    let (k_ptr, _k_guard) = k_out.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_mla_prefill_expand_k(
            nv_ptr as *const Half,
            rope_ptr as *const Half,
            k_ptr as *mut Half,
            t_total as i32,
            heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    rc.result()
        .map_err(|error| anyhow!("K3 MLA prefill expand_k (t={t_total}, heads={heads}): {error}"))
}

/// One chunk's MLA attention: `t_q` queries over `t_kv >= t_q` keys with the
/// bottom-right-aligned causal mask (query row `i` sees `t_kv - t_q + i + 1`
/// keys). `v` is read as a strided view into the kv_b expansion `nope_v`.
#[allow(clippy::too_many_arguments)]
pub fn k3_flash_mla_prefill_fwd_launch(
    ctx: &DeviceContext,
    t_q: usize,
    t_kv: usize,
    heads: usize,
    q: &CudaSlice<bf16>,
    k: &CudaSlice<bf16>,
    nope_v: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    scale: f32,
) -> Result<()> {
    ensure!(
        0 < t_q && t_q <= t_kv,
        "K3 FlashMLA prefill needs 0 < t_q <= t_kv, got {t_q}/{t_kv}"
    );
    ensure!(
        q.len() >= t_q * heads * K3_MLA_PREFILL_QK
            && k.len() >= t_kv * heads * K3_MLA_PREFILL_QK
            && nope_v.len() >= t_kv * heads * K3_MLA_PREFILL_NV
            && out.len() >= t_q * heads * K3_MLA_PREFILL_V,
        "K3 FlashMLA prefill buffers too small for t_q={t_q}, t_kv={t_kv}, heads={heads}"
    );
    let (q_ptr, _q_guard) = q.device_ptr(&ctx.stream);
    let (k_ptr, _k_guard) = k.device_ptr(&ctx.stream);
    let (nv_ptr, _nv_guard) = nope_v.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    // V lives at the value half of each expanded head row.
    let v_ptr = nv_ptr + (K3_MLA_PREFILL_V * size_of::<bf16>()) as u64;
    let rc = unsafe {
        ffi::k3_flash_mla_prefill_fwd(
            q_ptr as *const Half,
            (heads * K3_MLA_PREFILL_QK) as i64,
            K3_MLA_PREFILL_QK as i64,
            k_ptr as *const Half,
            (heads * K3_MLA_PREFILL_QK) as i64,
            K3_MLA_PREFILL_QK as i64,
            v_ptr as *const Half,
            (heads * K3_MLA_PREFILL_NV) as i64,
            K3_MLA_PREFILL_NV as i64,
            out_ptr as *mut Half,
            (heads * K3_MLA_PREFILL_V) as i64,
            K3_MLA_PREFILL_V as i64,
            t_q as i32,
            t_kv as i32,
            heads as i32,
            scale,
            ctx.stream.cu_stream(),
        )
    };
    rc.result().map_err(|error| {
        anyhow!("K3 FlashMLA prefill forward (t_q={t_q}, t_kv={t_kv}, heads={heads}) failed: {error} (NOT_SUPPORTED = built without an sm_100f target)")
    })
}
