//! Kimi-K3 absorbed-MLA decode over the paged latent KV cache.
//!
//! The cache holds one 576-wide bf16 latent row per token per MLA layer (the
//! post-norm kv latent | the shared rope half — NoPE, nothing rotated), in a
//! page-first slab of 64-token pages addressed through a per-row block table.
//! The kernel absorbs the query into latent space against `w_kv_b`'s W_UK
//! rows, attends MQA-style over the shared rows, and expands the attended
//! latent with the W_UV rows — see `csrc/k3/k3_mla_paged_attn.cu` for the
//! documented rounding chain.
//!
//! There is no compile-time context cap: the page walk is a runtime loop and
//! the per-row length `N` lives on the device, so the step needs no host sync.
//! Physical page ids never enter the arithmetic — the walk is by logical
//! position — so any page permutation is bit-identical. Batch is a plain
//! launch dimension here (no per-bucket instantiation), but callers still run
//! the compiled buckets: every other kernel in the step is bucket-shaped.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use super::super::k3_tilelang::K3_QK_DIM;
use super::super::k3_tilelang::K3_V_DIM;
use crate::ffi;
use crate::tensor::DeviceContext;

/// Tokens per MLA KV page — the kernel's only compile-time context term.
pub const K3_KV_PAGE_TOKENS: usize = 64;
/// kv_lora_rank, the latent width the absorption works in.
const K3_KV_LORA: usize = 512;
/// Cached latent row width: kv_lora_rank 512 | rope 64.
pub const K3_KV_LATENT_ROW: usize = K3_KV_LORA + 64;

/// Absorbed-MLA decode: `q [b, heads * 192]`, `w_kv_b [heads * 256, 512]`
/// (checkpoint orientation — per head `[128 nope | 128 value] x 512`), the
/// pool `slab` with this layer's slice at `layer_offset` elements and pages
/// `page_stride` elements apart, the device block `table [rows, max_pages]`
/// (`-1` = unmapped, read as zero latent), per-row device lengths `n [b]`,
/// the shared bf16 softmax `scale [1]`, out `o [b, heads * 128]`.
#[allow(clippy::too_many_arguments)]
pub fn k3_mla_paged_attn_launch(
    ctx: &DeviceContext,
    b: usize,
    num_heads: usize,
    q: &CudaSlice<bf16>,
    w_kv_b: &CudaSlice<bf16>,
    slab: &CudaSlice<bf16>,
    layer_offset: usize,
    page_stride: usize,
    table: &CudaSlice<i32>,
    max_pages: usize,
    n: &CudaSlice<i32>,
    scale: &CudaSlice<bf16>,
    o: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(b > 0 && num_heads > 0, "K3 paged MLA needs rows and heads");
    ensure!(
        page_stride >= K3_KV_PAGE_TOKENS * K3_KV_LATENT_ROW
            && layer_offset + K3_KV_PAGE_TOKENS * K3_KV_LATENT_ROW <= page_stride,
        "K3 paged MLA layer slice [{layer_offset}..) does not fit the {page_stride}-element page"
    );
    ensure!(
        q.len() >= b * num_heads * K3_QK_DIM
            && w_kv_b.len() >= num_heads * 2 * K3_V_DIM * K3_KV_LORA
            && table.len() >= b * max_pages
            && n.len() >= b
            && !scale.is_empty()
            && o.len() >= b * num_heads * K3_V_DIM,
        "K3 paged MLA buffers too small for b={b}, heads={num_heads}, max_pages={max_pages}: \
         q {}, w_kv_b {}, table {}, n {}, o {}",
        q.len(),
        w_kv_b.len(),
        table.len(),
        n.len(),
        o.len()
    );
    let (q_ptr, _q_guard) = q.device_ptr(&ctx.stream);
    let (w_ptr, _w_guard) = w_kv_b.device_ptr(&ctx.stream);
    let (slab_ptr, _slab_guard) = slab.device_ptr(&ctx.stream);
    let (table_ptr, _table_guard) = table.device_ptr(&ctx.stream);
    let (n_ptr, _n_guard) = n.device_ptr(&ctx.stream);
    let (scale_ptr, _scale_guard) = scale.device_ptr(&ctx.stream);
    let (o_ptr, _o_guard) = o.device_ptr_mut(&ctx.stream);
    unsafe {
        ffi::k3_mla_paged_attn_cuda(
            q_ptr as *const ffi::Half,
            w_ptr as *const ffi::Half,
            slab_ptr as *const ffi::Half,
            i64::try_from(layer_offset)?,
            i64::try_from(page_stride)?,
            table_ptr as *const i32,
            i32::try_from(max_pages)?,
            n_ptr as *const i32,
            scale_ptr as *const ffi::Half,
            o_ptr as *mut ffi::Half,
            i32::try_from(b)?,
            i32::try_from(num_heads)?,
            i32::try_from(K3_QK_DIM)?,
            i32::try_from(K3_V_DIM)?,
            crate::tensor::active_cu_stream(ctx),
        )
    }
    .result()
    .map_err(|err| anyhow!("K3 paged MLA attention launch failed: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_agrees_with_the_cache_row() {
        // The cached row is [kv_lora 512 | rope 64]; the query per head is
        // [nope 128 | rope 64]; the value head is 128 wide.
        assert_eq!(K3_KV_LATENT_ROW, K3_KV_LORA + (K3_QK_DIM - 128));
        assert_eq!(K3_V_DIM, 128);
        assert_eq!(K3_KV_PAGE_TOKENS, 64);
    }
}
