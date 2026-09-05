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
/// nope halves and the shared rope broadcast across heads. `rope_row0` shifts
/// the rope read window: the expansion may cover a context window whose rope
/// rows sit deep inside the full-context rope buffer.
pub fn k3_mla_prefill_expand_k_launch(
    ctx: &DeviceContext,
    t_total: usize,
    heads: usize,
    nope_v: &CudaSlice<bf16>,
    rope: &CudaSlice<bf16>,
    rope_row0: usize,
    k_out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(t_total > 0, "K3 MLA prefill expand got an empty span");
    ensure!(
        nope_v.len() >= t_total * heads * K3_MLA_PREFILL_NV
            && rope.len() >= (rope_row0 + t_total) * K3_MLA_PREFILL_ROPE
            && k_out.len() >= t_total * heads * K3_MLA_PREFILL_QK,
        "K3 MLA prefill expand buffers too small for t={t_total}, heads={heads}, \
         rope_row0={rope_row0}"
    );
    let (nv_ptr, _nv_guard) = nope_v.device_ptr(&ctx.stream);
    let (rope_ptr, _rope_guard) = rope.device_ptr(&ctx.stream);
    let (k_ptr, _k_guard) = k_out.device_ptr_mut(&ctx.stream);
    let rope_win = rope_ptr + (rope_row0 * K3_MLA_PREFILL_ROPE * size_of::<bf16>()) as u64;
    let rc = unsafe {
        ffi::k3_mla_prefill_expand_k(
            nv_ptr as *const Half,
            rope_win as *const Half,
            k_ptr as *mut Half,
            t_total as i32,
            heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    rc.result()
        .map_err(|error| anyhow!("K3 MLA prefill expand_k (t={t_total}, heads={heads}): {error}"))
}

/// Fold one window's FMHA output + LSE into the f32 accumulator pair
/// (`o_acc` `[t_q, heads, 128]`, `lse_acc` `[heads, t_q]`) via the
/// log-sum-exp identity. `reset` starts a fresh accumulation.
pub fn k3_mla_prefill_lse_merge_launch(
    ctx: &DeviceContext,
    t_q: usize,
    heads: usize,
    o_win: &CudaSlice<bf16>,
    lse_win: &CudaSlice<f32>,
    o_acc: &mut CudaSlice<f32>,
    lse_acc: &mut CudaSlice<f32>,
    reset: bool,
) -> Result<()> {
    ensure!(t_q > 0, "K3 MLA prefill LSE merge got an empty span");
    ensure!(
        o_win.len() >= t_q * heads * K3_MLA_PREFILL_V
            && lse_win.len() >= heads * t_q
            && o_acc.len() >= t_q * heads * K3_MLA_PREFILL_V
            && lse_acc.len() >= heads * t_q,
        "K3 MLA prefill LSE merge buffers too small for t_q={t_q}, heads={heads}"
    );
    let (ow_ptr, _ow_guard) = o_win.device_ptr(&ctx.stream);
    let (lw_ptr, _lw_guard) = lse_win.device_ptr(&ctx.stream);
    let (oa_ptr, _oa_guard) = o_acc.device_ptr_mut(&ctx.stream);
    let (la_ptr, _la_guard) = lse_acc.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_mla_prefill_lse_merge(
            ow_ptr as *const Half,
            lw_ptr as *const f32,
            oa_ptr as *mut f32,
            la_ptr as *mut f32,
            t_q as i32,
            heads as i32,
            i32::from(reset),
            ctx.stream.cu_stream(),
        )
    };
    rc.result()
        .map_err(|error| anyhow!("K3 MLA prefill LSE merge (t_q={t_q}, heads={heads}): {error}"))
}

/// Convert the merged f32 accumulator back to the bf16 attention output.
pub fn k3_mla_prefill_o_finalize_launch(
    ctx: &DeviceContext,
    t_q: usize,
    heads: usize,
    o_acc: &CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    ensure!(t_q > 0, "K3 MLA prefill finalize got an empty span");
    ensure!(
        o_acc.len() >= t_q * heads * K3_MLA_PREFILL_V
            && out.len() >= t_q * heads * K3_MLA_PREFILL_V,
        "K3 MLA prefill finalize buffers too small for t_q={t_q}, heads={heads}"
    );
    let (oa_ptr, _oa_guard) = o_acc.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_mla_prefill_o_finalize(
            oa_ptr as *const f32,
            out_ptr as *mut Half,
            t_q as i32,
            heads as i32,
            ctx.stream.cu_stream(),
        )
    };
    rc.result()
        .map_err(|error| anyhow!("K3 MLA prefill finalize (t_q={t_q}, heads={heads}): {error}"))
}

/// The two FMHA entries share one FFI signature; the wrapper core takes the
/// entry as a value.
type FmhaEntry = unsafe extern "C" fn(
    *const Half,
    i64,
    i64,
    *const Half,
    i64,
    i64,
    *const Half,
    i64,
    i64,
    *mut Half,
    i64,
    i64,
    *mut f32,
    i32,
    i32,
    i32,
    f32,
    cudarc::driver::sys::CUstream,
) -> cudarc::driver::sys::CUresult;

#[allow(clippy::too_many_arguments)]
fn k3_flash_mla_prefill_fmha(
    ctx: &DeviceContext,
    entry: FmhaEntry,
    kind: &str,
    t_q: usize,
    t_kv: usize,
    heads: usize,
    q: &CudaSlice<bf16>,
    k: &CudaSlice<bf16>,
    nope_v: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    lse_out: Option<&mut CudaSlice<f32>>,
    scale: f32,
) -> Result<()> {
    ensure!(
        t_q > 0 && t_kv > 0,
        "K3 FlashMLA prefill got an empty span: t_q={t_q}, t_kv={t_kv}"
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
    let lse_ptr = match lse_out {
        Some(lse) => {
            ensure!(
                lse.len() >= heads * t_q,
                "K3 FlashMLA prefill LSE buffer too small for t_q={t_q}, heads={heads}"
            );
            let (ptr, _guard) = lse.device_ptr_mut(&ctx.stream);
            ptr as *mut f32
        }
        None => std::ptr::null_mut(),
    };
    // V lives at the value half of each expanded head row.
    let v_ptr = nv_ptr + (K3_MLA_PREFILL_V * size_of::<bf16>()) as u64;
    let rc = unsafe {
        entry(
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
            lse_ptr,
            t_q as i32,
            t_kv as i32,
            heads as i32,
            scale,
            ctx.stream.cu_stream(),
        )
    };
    rc.result().map_err(|error| {
        anyhow!("K3 FlashMLA prefill {kind} (t_q={t_q}, t_kv={t_kv}, heads={heads}) failed: {error} (NOT_SUPPORTED = built without an sm_100f target)")
    })
}

/// One chunk's MLA attention: `t_q` queries over `t_kv >= t_q` keys with the
/// bottom-right-aligned causal mask (query row `i` sees `t_kv - t_q + i + 1`
/// keys). `v` is read as a strided view into the kv_b expansion `nope_v`.
/// `lse_out` optionally receives the per-row log-sum-exp as `[heads, t_q]` f32
/// (natural log, scale absorbed) for cross-window merging.
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
    lse_out: Option<&mut CudaSlice<f32>>,
    scale: f32,
) -> Result<()> {
    ensure!(
        t_q <= t_kv,
        "K3 FlashMLA causal prefill needs t_q <= t_kv, got {t_q}/{t_kv}"
    );
    k3_flash_mla_prefill_fmha(
        ctx,
        ffi::k3_flash_mla_prefill_fwd,
        "causal",
        t_q,
        t_kv,
        heads,
        q,
        k,
        nope_v,
        out,
        lse_out,
        scale,
    )
}

/// Dense (unmasked) variant: every query row attends all `t_kv` keys, with no
/// relation required between `t_q` and `t_kv`. Used for the context windows of
/// the W-chunked walk, where the window's keys all precede the chunk's queries.
#[allow(clippy::too_many_arguments)]
pub fn k3_flash_mla_prefill_fwd_dense_launch(
    ctx: &DeviceContext,
    t_q: usize,
    t_kv: usize,
    heads: usize,
    q: &CudaSlice<bf16>,
    k: &CudaSlice<bf16>,
    nope_v: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    lse_out: Option<&mut CudaSlice<f32>>,
    scale: f32,
) -> Result<()> {
    k3_flash_mla_prefill_fmha(
        ctx,
        ffi::k3_flash_mla_prefill_fwd_dense,
        "dense",
        t_q,
        t_kv,
        heads,
        q,
        k,
        nope_v,
        out,
        lse_out,
        scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random bf16 in roughly [-0.5, 0.5].
    fn ripple(seed: u64, i: usize) -> bf16 {
        let mut x = seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 33;
        x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
        x ^= x >> 33;
        bf16::from_f32(((x % 1000) as f32) / 1000.0 - 0.5)
    }

    /// The windowed walk must reproduce the single causal call: dense windows
    /// over the past + one causal `t_q x t_q` tail, merged through the LSE,
    /// against one bottom-right-aligned causal FMHA over the full context.
    /// Ragged tail included (1280 = 512 + 512 + 256).
    #[test]
    #[ignore = "needs an sm_100f GPU"]
    fn windowed_walk_matches_single_causal_call() {
        use crate::tensor::DeviceContext;
        let ctx = DeviceContext::new_with_device(0).expect("device 0");
        let (t_q, t_kv, heads, win) = (192usize, 1472usize, 96usize, 512usize);
        let ctx_len = t_kv - t_q;
        let scale = 0.072f32;

        let q_host: Vec<bf16> = (0..t_q * heads * K3_MLA_PREFILL_QK)
            .map(|i| ripple(1, i))
            .collect();
        let k_host: Vec<bf16> = (0..t_kv * heads * K3_MLA_PREFILL_QK)
            .map(|i| ripple(2, i))
            .collect();
        let nv_host: Vec<bf16> = (0..t_kv * heads * K3_MLA_PREFILL_NV)
            .map(|i| ripple(3, i))
            .collect();
        let q = ctx.stream.clone_htod(&q_host).expect("q");
        let k_full = ctx.stream.clone_htod(&k_host).expect("k");
        let nv_full = ctx.stream.clone_htod(&nv_host).expect("nope_v");

        // Reference: one causal call over the whole context.
        let mut out_ref = ctx
            .stream
            .alloc_zeros::<bf16>(t_q * heads * K3_MLA_PREFILL_V)
            .expect("out_ref");
        k3_flash_mla_prefill_fwd_launch(
            &ctx,
            t_q,
            t_kv,
            heads,
            &q,
            &k_full,
            &nv_full,
            &mut out_ref,
            None,
            scale,
        )
        .expect("reference forward");
        let host_ref = ctx.stream.clone_dtoh(&out_ref).expect("ref to host");

        // Windowed: dense windows over the past, causal tail, LSE merge.
        let mut o_win = ctx
            .stream
            .alloc_zeros::<bf16>(t_q * heads * K3_MLA_PREFILL_V)
            .expect("o_win");
        let mut lse_win = ctx.stream.alloc_zeros::<f32>(heads * t_q).expect("lse_win");
        let mut o_acc = ctx
            .stream
            .alloc_zeros::<f32>(t_q * heads * K3_MLA_PREFILL_V)
            .expect("o_acc");
        let mut lse_acc = ctx.stream.alloc_zeros::<f32>(heads * t_q).expect("lse_acc");
        let mut row = 0usize;
        while row < ctx_len {
            let len = win.min(ctx_len - row);
            let k_w = ctx
                .stream
                .clone_htod(
                    &k_host[row * heads * K3_MLA_PREFILL_QK..][..len * heads * K3_MLA_PREFILL_QK],
                )
                .expect("k window");
            let nv_w = ctx
                .stream
                .clone_htod(
                    &nv_host[row * heads * K3_MLA_PREFILL_NV..][..len * heads * K3_MLA_PREFILL_NV],
                )
                .expect("nv window");
            k3_flash_mla_prefill_fwd_dense_launch(
                &ctx,
                t_q,
                len,
                heads,
                &q,
                &k_w,
                &nv_w,
                &mut o_win,
                Some(&mut lse_win),
                scale,
            )
            .expect("dense window");
            k3_mla_prefill_lse_merge_launch(
                &ctx,
                t_q,
                heads,
                &o_win,
                &lse_win,
                &mut o_acc,
                &mut lse_acc,
                row == 0,
            )
            .expect("merge window");
            row += len;
        }
        let k_tail = ctx
            .stream
            .clone_htod(&k_host[ctx_len * heads * K3_MLA_PREFILL_QK..])
            .expect("k tail");
        let nv_tail = ctx
            .stream
            .clone_htod(&nv_host[ctx_len * heads * K3_MLA_PREFILL_NV..])
            .expect("nv tail");
        k3_flash_mla_prefill_fwd_launch(
            &ctx,
            t_q,
            t_q,
            heads,
            &q,
            &k_tail,
            &nv_tail,
            &mut o_win,
            Some(&mut lse_win),
            scale,
        )
        .expect("causal tail");
        k3_mla_prefill_lse_merge_launch(
            &ctx,
            t_q,
            heads,
            &o_win,
            &lse_win,
            &mut o_acc,
            &mut lse_acc,
            false,
        )
        .expect("merge tail");
        let mut out_w = ctx
            .stream
            .alloc_zeros::<bf16>(t_q * heads * K3_MLA_PREFILL_V)
            .expect("out_w");
        k3_mla_prefill_o_finalize_launch(&ctx, t_q, heads, &o_acc, &mut out_w).expect("finalize");
        let host_win = ctx.stream.clone_dtoh(&out_w).expect("win to host");
        ctx.sync().expect("sync");

        let mut worst = 0f32;
        for (i, (&r, &w)) in host_ref.iter().zip(host_win.iter()).enumerate() {
            let (r, w) = (r.to_f32(), w.to_f32());
            assert!(
                r.is_finite() && w.is_finite(),
                "non-finite at {i}: {r} vs {w}"
            );
            worst = worst.max((r - w).abs());
        }
        // bf16 outputs of the same f32-accumulated softmax: the merge only
        // reorders the sum, so the paths agree to bf16 rounding.
        assert!(
            worst < 2e-2,
            "windowed walk diverged: worst |diff| = {worst}"
        );
    }

    /// Drives the kernel past the int32 batch-stride ceiling
    /// (`t_kv * heads * 192`). Identical K rows make every causal softmax
    /// uniform, and V rows that vary only per head make the expected output
    /// exactly the per-head value, so any high-offset addressing error shows
    /// up as a wrong or non-finite element.
    #[test]
    #[ignore = "needs an sm_100f GPU with ~23 GiB free and ~13 GiB host RAM"]
    fn deep_context_attention_addresses_past_int32() {
        use crate::tensor::DeviceContext;
        let ctx = DeviceContext::new_with_device(0).expect("device 0");
        let (t_q, t_kv, heads) = (256usize, 260_096usize, 96usize);
        assert!(
            t_kv * heads * K3_MLA_PREFILL_QK > i32::MAX as usize,
            "depth no longer covers the i32 ceiling"
        );
        let q = ctx
            .stream
            .clone_htod(&vec![bf16::from_f32(0.05); t_q * heads * K3_MLA_PREFILL_QK])
            .expect("q");
        let k = ctx
            .stream
            .clone_htod(&vec![
                bf16::from_f32(0.03);
                t_kv * heads * K3_MLA_PREFILL_QK
            ])
            .expect("k");
        let row: Vec<bf16> = (0..heads * K3_MLA_PREFILL_NV)
            .map(|i| bf16::from_f32((i / K3_MLA_PREFILL_NV) as f32 / 128.0))
            .collect();
        let mut nope_v_host = vec![bf16::ZERO; t_kv * heads * K3_MLA_PREFILL_NV];
        for chunk in nope_v_host.chunks_mut(row.len()) {
            chunk.copy_from_slice(&row);
        }
        let nope_v = ctx.stream.clone_htod(&nope_v_host).expect("nope_v");
        drop(nope_v_host);
        let mut out = ctx
            .stream
            .alloc_zeros::<bf16>(t_q * heads * K3_MLA_PREFILL_V)
            .expect("out");
        k3_flash_mla_prefill_fwd_launch(
            &ctx, t_q, t_kv, heads, &q, &k, &nope_v, &mut out, None, 0.072,
        )
        .expect("deep-context forward");
        let host_out = ctx.stream.clone_dtoh(&out).expect("out to host");
        ctx.sync().expect("sync");
        for (i, &value) in host_out.iter().enumerate() {
            let head = (i / K3_MLA_PREFILL_V) % heads;
            let expected = head as f32 / 128.0;
            let got = value.to_f32();
            assert!(
                (got - expected).abs() < 1e-2,
                "out[{i}] (head {head}): got {got}, expected {expected}"
            );
        }
    }
}
