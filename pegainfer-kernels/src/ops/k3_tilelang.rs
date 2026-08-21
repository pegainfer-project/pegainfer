//! K3 TileLang batched decode kernels: safe wrappers over the AOT dispatch
//! launchers in `ffi::k3_tilelang`.
//!
//! The set covers one whole K3 decode step that is not a GEMM or attention —
//! norms and the bf16 landings of the framework GEMMs, the KDA convolution and
//! delta rule, the expert combine, the situ activation and the
//! attention-residual mix. Dense projections are served by cuBLASLt, the
//! routed experts by the DeepGEMM masked grouped-GEMM chain, MLA decode by
//! the hand-written absorbed paged kernel (`ops::k3::mla_paged`), and the MoE
//! router top-k by the hand-written parallel-argmax kernel
//! (`ops::k3::router_topk`), so neither a GEMV nor an attention family lives
//! here. The wrappers keep the certified kernels' operand names, so an
//! executor written against them reads like the Python engine's launch
//! sequence.
//!
//! Every kernel is compiled per static shape tuple, batch size included. A
//! configuration that was not instantiated fails at the launcher with
//! `cudaErrorInvalidValue` rather than silently running the wrong shape, so
//! the checks here cover buffer capacity and the invariants a wrong-sized
//! launch would not catch.
//!
//! The batch dimension is a compile dimension: a step with `rows` live rows
//! runs the next bucket up ([`K3_BATCH_BUCKETS`], via [`k3_batch_bucket`]) and
//! discards the tail rows, so the caller must hand in buffers sized for the
//! *bucket*, not for `rows`. Allocating every buffer at [`K3_MAX_BATCH`] once
//! and reusing it is the intended shape — it also keeps the pointers stable
//! for CUDA Graph capture. `b = 1` is a first-class bucket whose per-row
//! spelling is the certified single-row kernel, so single-stream decode needs
//! no separate kernel set.

use core::ffi::c_void;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

/// Batch buckets the generator instantiates, ascending.
pub const K3_BATCH_BUCKETS: [usize; 10] = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128];

/// Prefill chunk buckets: the chunked-prefill step runs the same batched
/// families at chunk width, whose ceiling is the MegaMoE protocol maximum
/// (4224 rows). Compiled for every family except `kda_core` — chunks cross
/// the KDA recurrence through FlashKDA, so the fused core never sees a
/// chunk-sized bucket.
pub const K3_PREFILL_BUCKETS: [usize; 5] = [256, 512, 1024, 2048, 4224];

/// The largest prefill chunk any configuration can run — the MegaMoE
/// protocol maximum (`k3_mega_max_tokens_per_rank`).
pub const K3_MAX_CHUNK: usize = 4224;
/// Largest bucket, i.e. the row capacity every reusable buffer should have.
pub const K3_MAX_BATCH: usize = 128;

/// Model hidden size, baked into the norm and attention-residual shapes.
pub const K3_HIDDEN: usize = 7168;
/// Routed-latent width, the working width of the MoE expert path.
pub const K3_LATENT: usize = 3584;
/// Routed-expert counts the router is instantiated for: the full table and the
/// per-rank shard of a 4-way expert-parallel deployment.
pub const K3_ROUTER_EXPERTS: [usize; 2] = [896, 224];
/// Route width baked into the router and combine instantiations.
pub const K3_ROUTER_TOPK: usize = 16;
/// Largest attention-residual candidate count (the snapshot history grows one
/// entry per `attn_res_block_size` layers).
pub const K3_ATTNRES_MAX_BLOCKS: usize = 8;

/// KDA head count and per-head width; `K3_KDA_DIM` is the flat projection
/// width the convolution and the output gate work on.
pub const K3_KDA_HEADS: usize = 96;
pub const K3_KDA_HEAD_DIM: usize = 128;
pub const K3_KDA_DIM: usize = K3_KDA_HEADS * K3_KDA_HEAD_DIM;
/// Short-convolution window; the carried state is `K3_CONV_WIDTH - 1` slots.
pub const K3_CONV_WIDTH: usize = 4;

/// MLA head count and the query/key and value widths per head. MLA decode
/// itself is not a TileLang family — it is the hand-written absorbed paged
/// kernel in `ops::k3::mla_paged` — but its head geometry is shared with the
/// landings here.
pub const K3_MLA_HEADS: usize = 96;
pub const K3_QK_DIM: usize = 192;
pub const K3_V_DIM: usize = 128;

/// Round a live row count up to the bucket that will run it.
///
/// The extra rows are computed and discarded, so the caller must still size
/// its buffers for the returned bucket.
pub fn k3_batch_bucket(rows: usize) -> Result<usize> {
    ensure!(rows > 0, "K3 decode needs at least one row");
    K3_BATCH_BUCKETS
        .into_iter()
        .find(|bucket| *bucket >= rows)
        .ok_or_else(|| anyhow!("K3 decode batch {rows} exceeds the largest bucket {K3_MAX_BATCH}"))
}

/// Round a prefill chunk's token count up to its compiled bucket — the decode
/// ladder extended by [`K3_PREFILL_BUCKETS`] up to the [`K3_MAX_CHUNK`]
/// protocol ceiling.
pub fn k3_chunk_bucket(rows: usize) -> Result<usize> {
    ensure!(rows > 0, "K3 chunk needs at least one row");
    K3_BATCH_BUCKETS
        .into_iter()
        .chain(K3_PREFILL_BUCKETS)
        .find(|bucket| *bucket >= rows)
        .ok_or_else(|| {
            anyhow!("K3 chunk of {rows} tokens exceeds the largest bucket {K3_MAX_CHUNK}")
        })
}

fn check_bucket(b: usize) -> Result<()> {
    ensure!(
        K3_BATCH_BUCKETS.contains(&b) || K3_PREFILL_BUCKETS.contains(&b),
        "K3 batch {b} is not a compiled bucket; round with k3_batch_bucket / k3_chunk_bucket \
         (buckets: {K3_BATCH_BUCKETS:?} + {K3_PREFILL_BUCKETS:?})"
    );
    Ok(())
}

fn check(rc: i32, what: &str) -> Result<()> {
    ensure!(
        rc == 0,
        "{what} failed: cudaError={rc} (1 = the configuration was never instantiated)"
    );
    Ok(())
}

/// KimiRMSNorm with round-before-scale semantics: the normalized value lands
/// in bf16 first and only then multiplies gamma.
pub fn k3_rms_norm_rbs_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    h: usize,
    x: &CudaSlice<bf16>,
    g: &CudaSlice<bf16>,
    o: &mut CudaSlice<bf16>,
) -> Result<()> {
    check_bucket(b)?;
    ensure!(
        x.len() >= b * h && g.len() >= h && o.len() >= b * h,
        "K3 rms_norm_rbs buffers too small for b={b}, h={h}: x {}, g {}, o {}",
        x.len(),
        g.len(),
        o.len()
    );
    let (x_ptr, _x_guard) = x.device_ptr(&ctx.stream);
    let (g_ptr, _g_guard) = g.device_ptr(&ctx.stream);
    let (o_ptr, _o_guard) = o.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_rms_norm_rbs_batched(
            x_ptr as *const c_void,
            g_ptr as *const c_void,
            o_ptr as *mut c_void,
            b as i32,
            h as i32,
            ctx.stream.cu_stream(),
        )
    };
    check(rc, &format!("K3 rms_norm_rbs_batched (B={b}, H={h})"))
}

/// Merge the column span `[off, off + n)` of each row's `[split_k, nt]`
/// partials and land bf16 once — the landing of every matmul in the certified
/// spelling.
#[allow(clippy::too_many_arguments)]
pub fn k3_land_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    nt: usize,
    n: usize,
    off: usize,
    split_k: usize,
    p: &CudaSlice<f32>,
    o: &mut CudaSlice<bf16>,
) -> Result<()> {
    check_bucket(b)?;
    ensure!(
        off + n <= nt,
        "K3 land span [{off}, {off}+{n}) does not fit the partial width {nt}"
    );
    ensure!(
        p.len() >= b * split_k * nt && o.len() >= b * n,
        "K3 land buffers too small for b={b}, nt={nt}, n={n}, split_k={split_k}: p {}, o {}",
        p.len(),
        o.len()
    );
    let (p_ptr, _p_guard) = p.device_ptr(&ctx.stream);
    let (o_ptr, _o_guard) = o.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_land_batched(
            p_ptr as *const f32,
            o_ptr as *mut c_void,
            b as i32,
            nt as i32,
            n as i32,
            off as i32,
            split_k as i32,
            ctx.stream.cu_stream(),
        )
    };
    check(
        rc,
        &format!("K3 land_batched (B={b}, NT={nt}, N={n}, OFF={off}, SK={split_k})"),
    )
}

/// [`k3_land_batched_launch`] fused with the round-before-scale norm — MLA's
/// `q_a`, the one place the engine fuses a merge and a norm.
#[allow(clippy::too_many_arguments)]
pub fn k3_land_rms_norm_rbs_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    nt: usize,
    n: usize,
    off: usize,
    split_k: usize,
    p: &CudaSlice<f32>,
    g: &CudaSlice<bf16>,
    o: &mut CudaSlice<bf16>,
) -> Result<()> {
    check_bucket(b)?;
    ensure!(
        off + n <= nt,
        "K3 land_rms_norm_rbs span [{off}, {off}+{n}) does not fit the partial width {nt}"
    );
    ensure!(
        p.len() >= b * split_k * nt && g.len() >= n && o.len() >= b * n,
        "K3 land_rms_norm_rbs buffers too small for b={b}, nt={nt}, n={n}: p {}, g {}, o {}",
        p.len(),
        g.len(),
        o.len()
    );
    let (p_ptr, _p_guard) = p.device_ptr(&ctx.stream);
    let (g_ptr, _g_guard) = g.device_ptr(&ctx.stream);
    let (o_ptr, _o_guard) = o.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_land_rms_norm_rbs_batched(
            p_ptr as *const f32,
            g_ptr as *const c_void,
            o_ptr as *mut c_void,
            b as i32,
            nt as i32,
            n as i32,
            off as i32,
            split_k as i32,
            ctx.stream.cu_stream(),
        )
    };
    check(
        rc,
        &format!("K3 land_rms_norm_rbs_batched (B={b}, NT={nt}, N={n}, OFF={off}, SK={split_k})"),
    )
}

/// The two `(A, Bt) -> O` elementwise kernels share a signature; only the
/// launcher symbol and the error label differ.
type BinaryLauncher = unsafe extern "C" fn(
    *const c_void,
    *const c_void,
    *mut c_void,
    i32,
    i32,
    cudarc::driver::sys::CUstream,
) -> i32;

fn k3_binary_batched_launch(
    ctx: &DeviceContext,
    launcher: BinaryLauncher,
    what: &str,
    b: usize,
    n: usize,
    a: &CudaSlice<bf16>,
    bt: &CudaSlice<bf16>,
    o: &mut CudaSlice<bf16>,
) -> Result<()> {
    check_bucket(b)?;
    ensure!(
        a.len() >= b * n && bt.len() >= b * n && o.len() >= b * n,
        "K3 {what} buffers too small for b={b}, n={n}: a {}, bt {}, o {}",
        a.len(),
        bt.len(),
        o.len()
    );
    let (a_ptr, _a_guard) = a.device_ptr(&ctx.stream);
    let (bt_ptr, _bt_guard) = bt.device_ptr(&ctx.stream);
    let (o_ptr, _o_guard) = o.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        launcher(
            a_ptr as *const c_void,
            bt_ptr as *const c_void,
            o_ptr as *mut c_void,
            b as i32,
            n as i32,
            ctx.stream.cu_stream(),
        )
    };
    check(rc, &format!("K3 {what} (B={b}, N={n})"))
}

/// `o = a + bt`, added in bf16 — the residual adds, and routed + shared.
pub fn k3_add2_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    n: usize,
    a: &CudaSlice<bf16>,
    bt: &CudaSlice<bf16>,
    o: &mut CudaSlice<bf16>,
) -> Result<()> {
    k3_binary_batched_launch(ctx, ffi::k3_add2_batched, "add2_batched", b, n, a, bt, o)
}

/// `o = a * bf16(sigmoid(bt))`, the MLA sigmoid output gate. The sigmoid is
/// taken in f32 and lands in bf16 before the product.
pub fn k3_mul_sigmoid_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    n: usize,
    a: &CudaSlice<bf16>,
    bt: &CudaSlice<bf16>,
    o: &mut CudaSlice<bf16>,
) -> Result<()> {
    k3_binary_batched_launch(
        ctx,
        ffi::k3_mul_sigmoid_batched,
        "mul_sigmoid_batched",
        b,
        n,
        a,
        bt,
        o,
    )
}

/// K3's situ activation: `4*tanh(g/4)*sigmoid(g) * 25*tanh(u/25)` in f32,
/// landing bf16 once. The two betas are compiled in.
pub fn k3_situ_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    n: usize,
    g: &CudaSlice<bf16>,
    u: &CudaSlice<bf16>,
    o: &mut CudaSlice<bf16>,
) -> Result<()> {
    check_bucket(b)?;
    ensure!(
        g.len() >= b * n && u.len() >= b * n && o.len() >= b * n,
        "K3 situ buffers too small for b={b}, n={n}: g {}, u {}, o {}",
        g.len(),
        u.len(),
        o.len()
    );
    let (g_ptr, _g_guard) = g.device_ptr(&ctx.stream);
    let (u_ptr, _u_guard) = u.device_ptr(&ctx.stream);
    let (o_ptr, _o_guard) = o.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_situ_batched(
            g_ptr as *const c_void,
            u_ptr as *const c_void,
            o_ptr as *mut c_void,
            b as i32,
            n as i32,
            ctx.stream.cu_stream(),
        )
    };
    check(rc, &format!("K3 situ_batched (B={b}, N={n})"))
}

/// KDA short convolution plus silu, one token per row.
///
/// `p` is the projection's f32 partial; its bf16 landing is written to `x` and
/// is also the newest window slot. `cs` is the carried window and `sn` its
/// successor, both `[b, width - 1, kp]` — the caller swaps or copies them.
/// Conv weights `cw` are f32 and shared by every row.
#[allow(clippy::too_many_arguments)]
pub fn k3_conv_silu_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    kp: usize,
    width: usize,
    split_k: usize,
    p: &CudaSlice<f32>,
    cw: &CudaSlice<f32>,
    cs: &CudaSlice<bf16>,
    x: &mut CudaSlice<bf16>,
    y: &mut CudaSlice<bf16>,
    sn: &mut CudaSlice<bf16>,
) -> Result<()> {
    check_bucket(b)?;
    ensure!(
        width >= 2,
        "K3 conv_silu needs a window of at least two slots"
    );
    let state = b * (width - 1) * kp;
    ensure!(
        p.len() >= b * split_k * kp
            && cw.len() >= width * kp
            && cs.len() >= state
            && x.len() >= b * kp
            && y.len() >= b * kp
            && sn.len() >= state,
        "K3 conv_silu buffers too small for b={b}, kp={kp}, width={width}: \
         p {}, cw {}, cs {}, x {}, y {}, sn {}",
        p.len(),
        cw.len(),
        cs.len(),
        x.len(),
        y.len(),
        sn.len()
    );
    let (p_ptr, _p_guard) = p.device_ptr(&ctx.stream);
    let (cw_ptr, _cw_guard) = cw.device_ptr(&ctx.stream);
    let (cs_ptr, _cs_guard) = cs.device_ptr(&ctx.stream);
    let (x_ptr, _x_guard) = x.device_ptr_mut(&ctx.stream);
    let (y_ptr, _y_guard) = y.device_ptr_mut(&ctx.stream);
    let (sn_ptr, _sn_guard) = sn.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_conv_silu_batched(
            p_ptr as *const f32,
            cw_ptr as *const f32,
            cs_ptr as *const c_void,
            x_ptr as *mut c_void,
            y_ptr as *mut c_void,
            sn_ptr as *mut c_void,
            b as i32,
            kp as i32,
            width as i32,
            split_k as i32,
            ctx.stream.cu_stream(),
        )
    };
    check(
        rc,
        &format!("K3 conv_silu_batched (B={b}, KP={kp}, W={width}, SK={split_k})"),
    )
}

/// One KDA delta-rule step per row: `state_n = state * decay + delta ⊗ k`,
/// then the attention read, its RMS norm and the bf16 sigmoid gate.
///
/// `state` and `state_n` are `[b, num_heads, head_dim, head_dim]` f32 laid out
/// `[head, v_dim, k_dim]` per row with decay along `k_dim`, and must be
/// distinct buffers. `dt`, `alog` and `go` are weights and carry no batch axis.
#[allow(clippy::too_many_arguments)]
pub fn k3_kda_core_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    num_heads: usize,
    head_dim: usize,
    split_k_gate: usize,
    q: &CudaSlice<bf16>,
    k: &CudaSlice<bf16>,
    v: &CudaSlice<bf16>,
    gp: &CudaSlice<f32>,
    dt: &CudaSlice<f32>,
    alog: &CudaSlice<f32>,
    bt: &CudaSlice<bf16>,
    g2: &CudaSlice<bf16>,
    go: &CudaSlice<f32>,
    state: &CudaSlice<f32>,
    state_n: &mut CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    check_bucket(b)?;
    let kp = num_heads * head_dim;
    let recurrent = b * num_heads * head_dim * head_dim;
    ensure!(
        q.len() >= b * kp
            && k.len() >= b * kp
            && v.len() >= b * kp
            && g2.len() >= b * kp
            && out.len() >= b * kp,
        "K3 kda_core projection buffers too small for b={b}, kp={kp}: \
         q {}, k {}, v {}, g2 {}, out {}",
        q.len(),
        k.len(),
        v.len(),
        g2.len(),
        out.len()
    );
    ensure!(
        gp.len() >= b * split_k_gate * kp
            && dt.len() >= kp
            && alog.len() >= num_heads
            && bt.len() >= b * num_heads
            && go.len() >= head_dim,
        "K3 kda_core gate buffers too small for b={b}, kp={kp}: gp {}, dt {}, alog {}, bt {}, go {}",
        gp.len(),
        dt.len(),
        alog.len(),
        bt.len(),
        go.len()
    );
    ensure!(
        state.len() >= recurrent && state_n.len() >= recurrent,
        "K3 kda_core state buffers too small for b={b}: state {}, state_n {} (need {recurrent})",
        state.len(),
        state_n.len()
    );
    let (q_ptr, _q_guard) = q.device_ptr(&ctx.stream);
    let (k_ptr, _k_guard) = k.device_ptr(&ctx.stream);
    let (v_ptr, _v_guard) = v.device_ptr(&ctx.stream);
    let (gp_ptr, _gp_guard) = gp.device_ptr(&ctx.stream);
    let (dt_ptr, _dt_guard) = dt.device_ptr(&ctx.stream);
    let (alog_ptr, _alog_guard) = alog.device_ptr(&ctx.stream);
    let (bt_ptr, _bt_guard) = bt.device_ptr(&ctx.stream);
    let (g2_ptr, _g2_guard) = g2.device_ptr(&ctx.stream);
    let (go_ptr, _go_guard) = go.device_ptr(&ctx.stream);
    let (state_ptr, _state_guard) = state.device_ptr(&ctx.stream);
    let (state_n_ptr, _state_n_guard) = state_n.device_ptr_mut(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_kda_core_batched(
            q_ptr as *const c_void,
            k_ptr as *const c_void,
            v_ptr as *const c_void,
            gp_ptr as *const f32,
            dt_ptr as *const f32,
            alog_ptr as *const f32,
            bt_ptr as *const c_void,
            g2_ptr as *const c_void,
            go_ptr as *const f32,
            state_ptr as *const f32,
            state_n_ptr as *mut f32,
            out_ptr as *mut c_void,
            b as i32,
            num_heads as i32,
            head_dim as i32,
            split_k_gate as i32,
            ctx.stream.cu_stream(),
        )
    };
    check(
        rc,
        &format!("K3 kda_core_batched (B={b}, KH={num_heads}, KD={head_dim})"),
    )
}

/// `kda_core`'s tail on its own: per (row, head) the f32 rms_norm of the bf16
/// attention landing `x` times the o_norm gamma `go [head_dim]`, landed once,
/// times the bf16 sigmoid of the output gate `g2` — word-for-word the batched
/// kernel's last loop. Chunked prefill computes the attention through FlashKDA
/// and finishes its rows here.
pub fn k3_o_norm_gate_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    num_heads: usize,
    head_dim: usize,
    x: &CudaSlice<bf16>,
    g2: &CudaSlice<bf16>,
    go: &CudaSlice<f32>,
    out: &mut CudaSlice<bf16>,
) -> Result<()> {
    check_bucket(b)?;
    let kp = num_heads * head_dim;
    ensure!(
        x.len() >= b * kp && g2.len() >= b * kp && out.len() >= b * kp && go.len() >= head_dim,
        "K3 o_norm_gate buffers too small for b={b}, kp={kp}: x {}, g2 {}, go {}, out {}",
        x.len(),
        g2.len(),
        go.len(),
        out.len()
    );
    let (x_ptr, _x_guard) = x.device_ptr(&ctx.stream);
    let (g2_ptr, _g2_guard) = g2.device_ptr(&ctx.stream);
    let (go_ptr, _go_guard) = go.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_o_norm_gate_batched(
            x_ptr as *const c_void,
            g2_ptr as *const c_void,
            go_ptr as *const f32,
            out_ptr as *mut c_void,
            b as i32,
            num_heads as i32,
            head_dim as i32,
            ctx.stream.cu_stream(),
        )
    };
    check(
        rc,
        &format!("K3 o_norm_gate_batched (B={b}, KH={num_heads}, KD={head_dim})"),
    )
}

/// Score the `blocks + 1` attention-residual candidates of every row: a
/// weightless RMS normalization then a dot with the fused f32 scoring vector.
#[allow(clippy::too_many_arguments)]
pub fn k3_attnres_scores_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    blocks: usize,
    h: usize,
    ps: &CudaSlice<bf16>,
    bl: &CudaSlice<bf16>,
    sw: &CudaSlice<f32>,
    sc: &mut CudaSlice<f32>,
) -> Result<()> {
    check_bucket(b)?;
    ensure!(
        (1..=K3_ATTNRES_MAX_BLOCKS).contains(&blocks),
        "K3 attnres blocks={blocks} outside the instantiated 1..={K3_ATTNRES_MAX_BLOCKS}"
    );
    ensure!(
        ps.len() >= b * h
            && bl.len() >= b * K3_ATTNRES_MAX_BLOCKS * h
            && sw.len() >= h
            && sc.len() >= b * (blocks + 1),
        "K3 attnres_scores buffers too small for b={b}, blocks={blocks}, h={h}: \
         ps {}, bl {}, sw {}, sc {}",
        ps.len(),
        bl.len(),
        sw.len(),
        sc.len()
    );
    let (ps_ptr, _ps_guard) = ps.device_ptr(&ctx.stream);
    let (bl_ptr, _bl_guard) = bl.device_ptr(&ctx.stream);
    let (sw_ptr, _sw_guard) = sw.device_ptr(&ctx.stream);
    let (sc_ptr, _sc_guard) = sc.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_attnres_scores_batched(
            ps_ptr as *const c_void,
            bl_ptr as *const c_void,
            sw_ptr as *const f32,
            sc_ptr as *mut f32,
            b as i32,
            blocks as i32,
            h as i32,
            ctx.stream.cu_stream(),
        )
    };
    check(
        rc,
        &format!("K3 attnres_scores_batched (B={b}, NB={blocks}, H={h})"),
    )
}

/// Softmax each row's scores and mix the *un-normalized* candidates by
/// probability, landing bf16 once.
#[allow(clippy::too_many_arguments)]
pub fn k3_attnres_mix_batched_launch(
    ctx: &DeviceContext,
    b: usize,
    blocks: usize,
    h: usize,
    ps: &CudaSlice<bf16>,
    bl: &CudaSlice<bf16>,
    sc: &CudaSlice<f32>,
    o: &mut CudaSlice<bf16>,
) -> Result<()> {
    check_bucket(b)?;
    ensure!(
        (1..=K3_ATTNRES_MAX_BLOCKS).contains(&blocks),
        "K3 attnres blocks={blocks} outside the instantiated 1..={K3_ATTNRES_MAX_BLOCKS}"
    );
    ensure!(
        ps.len() >= b * h
            && bl.len() >= b * K3_ATTNRES_MAX_BLOCKS * h
            && sc.len() >= b * (blocks + 1)
            && o.len() >= b * h,
        "K3 attnres_mix buffers too small for b={b}, blocks={blocks}, h={h}: \
         ps {}, bl {}, sc {}, o {}",
        ps.len(),
        bl.len(),
        sc.len(),
        o.len()
    );
    let (ps_ptr, _ps_guard) = ps.device_ptr(&ctx.stream);
    let (bl_ptr, _bl_guard) = bl.device_ptr(&ctx.stream);
    let (sc_ptr, _sc_guard) = sc.device_ptr(&ctx.stream);
    let (o_ptr, _o_guard) = o.device_ptr_mut(&ctx.stream);
    let rc = unsafe {
        ffi::k3_attnres_mix_batched(
            ps_ptr as *const c_void,
            bl_ptr as *const c_void,
            sc_ptr as *const f32,
            o_ptr as *mut c_void,
            b as i32,
            blocks as i32,
            h as i32,
            ctx.stream.cu_stream(),
        )
    };
    check(
        rc,
        &format!("K3 attnres_mix_batched (B={b}, NB={blocks}, H={h})"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_bucket_rounds_up_and_rejects_overflow() {
        assert_eq!(k3_batch_bucket(1).unwrap(), 1);
        assert_eq!(k3_batch_bucket(3).unwrap(), 4);
        assert_eq!(k3_batch_bucket(33).unwrap(), 48);
        assert_eq!(k3_batch_bucket(K3_MAX_BATCH).unwrap(), K3_MAX_BATCH);
        assert!(k3_batch_bucket(0).is_err());
        assert!(k3_batch_bucket(K3_MAX_BATCH + 1).is_err());
    }

    #[test]
    fn batch_buckets_are_ascending_and_end_at_the_maximum() {
        assert!(K3_BATCH_BUCKETS.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(*K3_BATCH_BUCKETS.last().unwrap(), K3_MAX_BATCH);
        // Every bucket must round-trip, or a caller could size buffers for a
        // bucket the generator never instantiated.
        for bucket in K3_BATCH_BUCKETS {
            assert_eq!(k3_batch_bucket(bucket).unwrap(), bucket);
            assert!(check_bucket(bucket).is_ok());
        }
        assert!(check_bucket(3).is_err());
    }

    #[test]
    fn chunk_buckets_extend_the_decode_ladder_to_the_protocol_max() {
        assert!(K3_PREFILL_BUCKETS.windows(2).all(|w| w[0] < w[1]));
        assert!(K3_PREFILL_BUCKETS[0] > K3_MAX_BATCH);
        assert_eq!(*K3_PREFILL_BUCKETS.last().unwrap(), K3_MAX_CHUNK);
        for bucket in K3_BATCH_BUCKETS.into_iter().chain(K3_PREFILL_BUCKETS) {
            assert_eq!(k3_chunk_bucket(bucket).unwrap(), bucket);
            assert!(check_bucket(bucket).is_ok());
        }
        assert_eq!(k3_chunk_bucket(K3_MAX_BATCH + 1).unwrap(), 256);
        assert_eq!(k3_chunk_bucket(4096).unwrap(), K3_MAX_CHUNK);
        assert!(k3_chunk_bucket(K3_MAX_CHUNK + 1).is_err());
    }

    #[test]
    fn model_dimensions_agree_with_the_generator() {
        // These mirror `pegainfer-k3/kernels/generate.py`; a silent divergence
        // would show up as `cudaErrorInvalidValue` from every launcher.
        assert_eq!(K3_KDA_DIM, 12288);
        assert_eq!(K3_QK_DIM, 192);
        assert_eq!(K3_MLA_HEADS, K3_KDA_HEADS);
        assert_eq!(K3_HIDDEN % 256, 0);
        assert_eq!(K3_LATENT % 256, 0);
        assert_eq!(K3_KDA_DIM % 256, 0);
        // The attention-residual history grows one entry per 12 layers of 93.
        assert_eq!(K3_ATTNRES_MAX_BLOCKS, 93_usize.div_ceil(12));
        // The 4-way expert-parallel shard of the full table.
        assert_eq!(K3_ROUTER_EXPERTS[0] / 4, K3_ROUTER_EXPERTS[1]);
    }
}
