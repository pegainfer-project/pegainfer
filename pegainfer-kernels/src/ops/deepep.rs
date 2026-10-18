//! Safe wrapper over the DeepEP elastic shim (csrc/deepep/).
//!
//! The shim is instantiated once per baked model config (Kimi-K2 `deepep_*`
//! symbols, GLM5.2 `glm52_deepep_*`); [`DeepEpAbi`] maps one instantiation's C
//! table into the shared wrapper [`DeepEpBase`]. One context per rank, used
//! from that rank's worker thread. All kernel launches are stream-ordered on
//! the caller's [`DeviceContext`] stream; the decode path has fixed worst-case
//! shapes and is CUDA-graph capturable.
//!
//! Layout contract (expanded mode): `recv.x` rows are per-(expert, token)
//! slots; each local expert's segment is contiguous and aligned to
//! [`DeepEpInfo::expert_alignment`]; `psum_expert` holds the running aligned
//! offsets (exclusive form). Combine never applies router weights — the
//! caller weights expert outputs before `decode_combine`.

use std::ffi::CStr;
use std::ffi::c_char;
use std::ffi::c_int;
use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi::DeepEpInfo;
use crate::ffi::{self};
use crate::tensor::DeviceContext;

/// The C table of one shim instantiation. Every method forwards to the
/// matching `<prefix>_deepep_*` symbol; see csrc/deepep/deepep.h for the
/// contract.
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub trait DeepEpAbi {
    type RawCtx;

    unsafe fn last_error() -> *const c_char;
    fn info() -> DeepEpInfo;
    unsafe fn unique_id(out: *mut u8) -> c_int;
    unsafe fn ctx_create(
        unique_id: *const u8,
        num_ranks: i32,
        rank_idx: i32,
        out: *mut *mut Self::RawCtx,
    ) -> c_int;
    unsafe fn ctx_destroy(ctx: *mut Self::RawCtx) -> c_int;
    unsafe fn decode_dispatch(
        ctx: *mut Self::RawCtx,
        stream: *mut c_void,
        x: *const c_void,
        topk_idx: *const i32,
        topk_weights: *const f32,
        num_tokens: i32,
        rank_count_scratch: *mut i32,
        dst_slot_scratch: *mut i32,
        psum_rank: *mut i32,
        psum_expert: *mut i32,
        recv_x: *mut c_void,
        recv_topk_weights: *mut f32,
        recv_src_metadata: *mut i32,
    ) -> c_int;
    unsafe fn decode_combine(
        ctx: *mut Self::RawCtx,
        stream: *mut c_void,
        x: *const c_void,
        src_metadata: *const i32,
        psum_rank: *const i32,
        combined_topk_idx: *const i32,
        num_tokens: i32,
        combined_x: *mut c_void,
    ) -> c_int;
}

/// Optional CPU-synchronized prefill capability. GLM5.2 deliberately does
/// not implement this trait: its scheduler sends prompt rows through decode.
#[allow(clippy::missing_safety_doc, clippy::too_many_arguments)]
pub trait DeepEpPrefillAbi: DeepEpAbi {
    unsafe fn prefill_dispatch_send(
        ctx: *mut Self::RawCtx,
        stream: *mut c_void,
        x: *const c_void,
        topk_idx: *const i32,
        topk_weights: *const f32,
        num_tokens: i32,
        rank_count_scratch: *mut i32,
        dst_slot_scratch: *mut i32,
        psum_rank: *mut i32,
        psum_expert: *mut i32,
    ) -> c_int;
    unsafe fn prefill_wait_counts(
        ctx: *mut Self::RawCtx,
        num_recv_tokens: *mut i32,
        num_expanded_tokens: *mut i32,
    ) -> c_int;
    unsafe fn prefill_dispatch_recv(
        ctx: *mut Self::RawCtx,
        stream: *mut c_void,
        num_recv_tokens: i32,
        psum_rank: *const i32,
        psum_expert: *const i32,
        recv_x: *mut c_void,
        recv_topk_weights: *mut f32,
        recv_src_metadata: *mut i32,
    ) -> c_int;
    unsafe fn prefill_combine(
        ctx: *mut Self::RawCtx,
        stream: *mut c_void,
        x: *const c_void,
        src_metadata: *const i32,
        psum_rank: *const i32,
        num_recv_tokens: i32,
        combined_topk_idx: *const i32,
        num_tokens: i32,
        combined_x: *mut c_void,
    ) -> c_int;
}

/// Stamps a [`DeepEpAbi`] impl by delegating each trait method to the given
/// ffi path (no ident pasting in expression position, so the mapping is
/// spelled out per method).
macro_rules! deepep_abi {
    (
        $abi:ident, $ctx:ty,
        last_error => $last_error:path,
        info => $info:path,
        unique_id => $unique_id:path,
        ctx_create => $ctx_create:path,
        ctx_destroy => $ctx_destroy:path,
        decode_dispatch => $decode_dispatch:path,
        decode_combine => $decode_combine:path,
    ) => {
        pub struct $abi;

        impl DeepEpAbi for $abi {
            type RawCtx = $ctx;

            unsafe fn last_error() -> *const c_char {
                unsafe { $last_error() }
            }
            fn info() -> DeepEpInfo {
                let mut info = DeepEpInfo::default();
                // Safety: plain struct fill, no preconditions.
                unsafe { $info(&raw mut info) };
                info
            }
            unsafe fn unique_id(out: *mut u8) -> c_int {
                unsafe { $unique_id(out) }
            }
            unsafe fn ctx_create(
                unique_id: *const u8,
                num_ranks: i32,
                rank_idx: i32,
                out: *mut *mut Self::RawCtx,
            ) -> c_int {
                unsafe { $ctx_create(unique_id, num_ranks, rank_idx, out) }
            }
            unsafe fn ctx_destroy(ctx: *mut Self::RawCtx) -> c_int {
                unsafe { $ctx_destroy(ctx) }
            }
            unsafe fn decode_dispatch(
                ctx: *mut Self::RawCtx,
                stream: *mut c_void,
                x: *const c_void,
                topk_idx: *const i32,
                topk_weights: *const f32,
                num_tokens: i32,
                rank_count_scratch: *mut i32,
                dst_slot_scratch: *mut i32,
                psum_rank: *mut i32,
                psum_expert: *mut i32,
                recv_x: *mut c_void,
                recv_topk_weights: *mut f32,
                recv_src_metadata: *mut i32,
            ) -> c_int {
                unsafe {
                    $decode_dispatch(
                        ctx,
                        stream,
                        x,
                        topk_idx,
                        topk_weights,
                        num_tokens,
                        rank_count_scratch,
                        dst_slot_scratch,
                        psum_rank,
                        psum_expert,
                        recv_x,
                        recv_topk_weights,
                        recv_src_metadata,
                    )
                }
            }
            unsafe fn decode_combine(
                ctx: *mut Self::RawCtx,
                stream: *mut c_void,
                x: *const c_void,
                src_metadata: *const i32,
                psum_rank: *const i32,
                combined_topk_idx: *const i32,
                num_tokens: i32,
                combined_x: *mut c_void,
            ) -> c_int {
                unsafe {
                    $decode_combine(
                        ctx,
                        stream,
                        x,
                        src_metadata,
                        psum_rank,
                        combined_topk_idx,
                        num_tokens,
                        combined_x,
                    )
                }
            }
        }
    };
}

deepep_abi!(
    KimiDeepEpAbi,
    ffi::DeepEpCtx,
    last_error => ffi::deepep_last_error,
    info => ffi::deepep_info,
    unique_id => ffi::deepep_unique_id,
    ctx_create => ffi::deepep_ctx_create,
    ctx_destroy => ffi::deepep_ctx_destroy,
    decode_dispatch => ffi::deepep_decode_dispatch,
    decode_combine => ffi::deepep_decode_combine,
);

impl DeepEpPrefillAbi for KimiDeepEpAbi {
    unsafe fn prefill_dispatch_send(
        ctx: *mut Self::RawCtx,
        stream: *mut c_void,
        x: *const c_void,
        topk_idx: *const i32,
        topk_weights: *const f32,
        num_tokens: i32,
        rank_count_scratch: *mut i32,
        dst_slot_scratch: *mut i32,
        psum_rank: *mut i32,
        psum_expert: *mut i32,
    ) -> c_int {
        unsafe {
            ffi::deepep_prefill_dispatch_send(
                ctx,
                stream,
                x,
                topk_idx,
                topk_weights,
                num_tokens,
                rank_count_scratch,
                dst_slot_scratch,
                psum_rank,
                psum_expert,
            )
        }
    }

    unsafe fn prefill_wait_counts(
        ctx: *mut Self::RawCtx,
        num_recv_tokens: *mut i32,
        num_expanded_tokens: *mut i32,
    ) -> c_int {
        unsafe { ffi::deepep_prefill_wait_counts(ctx, num_recv_tokens, num_expanded_tokens) }
    }

    unsafe fn prefill_dispatch_recv(
        ctx: *mut Self::RawCtx,
        stream: *mut c_void,
        num_recv_tokens: i32,
        psum_rank: *const i32,
        psum_expert: *const i32,
        recv_x: *mut c_void,
        recv_topk_weights: *mut f32,
        recv_src_metadata: *mut i32,
    ) -> c_int {
        unsafe {
            ffi::deepep_prefill_dispatch_recv(
                ctx,
                stream,
                num_recv_tokens,
                psum_rank,
                psum_expert,
                recv_x,
                recv_topk_weights,
                recv_src_metadata,
            )
        }
    }

    unsafe fn prefill_combine(
        ctx: *mut Self::RawCtx,
        stream: *mut c_void,
        x: *const c_void,
        src_metadata: *const i32,
        psum_rank: *const i32,
        num_recv_tokens: i32,
        combined_topk_idx: *const i32,
        num_tokens: i32,
        combined_x: *mut c_void,
    ) -> c_int {
        unsafe {
            ffi::deepep_prefill_combine(
                ctx,
                stream,
                x,
                src_metadata,
                psum_rank,
                num_recv_tokens,
                combined_topk_idx,
                num_tokens,
                combined_x,
            )
        }
    }
}

#[cfg(feature = "glm52")]
deepep_abi!(
    Glm52DeepEpAbi,
    ffi::Glm52DeepEpCtx,
    last_error => ffi::glm52_deepep_last_error,
    info => ffi::glm52_deepep_info,
    unique_id => ffi::glm52_deepep_unique_id,
    ctx_create => ffi::glm52_deepep_ctx_create,
    ctx_destroy => ffi::glm52_deepep_ctx_destroy,
    decode_dispatch => ffi::glm52_deepep_decode_dispatch,
    decode_combine => ffi::glm52_deepep_decode_combine,
);

#[cfg(feature = "glm52")]
deepep_abi!(
    Glm52Ep4DeepEpAbi,
    ffi::Glm52Ep4DeepEpCtx,
    last_error => ffi::glm52_ep4_deepep_last_error,
    info => ffi::glm52_ep4_deepep_info,
    unique_id => ffi::glm52_ep4_deepep_unique_id,
    ctx_create => ffi::glm52_ep4_deepep_ctx_create,
    ctx_destroy => ffi::glm52_ep4_deepep_ctx_destroy,
    decode_dispatch => ffi::glm52_ep4_deepep_decode_dispatch,
    decode_combine => ffi::glm52_ep4_deepep_decode_combine,
);

#[cfg(feature = "glm52")]
deepep_abi!(
    Glm52Ep16DeepEpAbi,
    ffi::Glm52Ep16DeepEpCtx,
    last_error => ffi::glm52_ep16_deepep_last_error,
    info => ffi::glm52_ep16_deepep_info,
    unique_id => ffi::glm52_ep16_deepep_unique_id,
    ctx_create => ffi::glm52_ep16_deepep_ctx_create,
    ctx_destroy => ffi::glm52_ep16_deepep_ctx_destroy,
    decode_dispatch => ffi::glm52_ep16_deepep_decode_dispatch,
    decode_combine => ffi::glm52_ep16_deepep_decode_combine,
);

#[cfg(feature = "glm52")]
deepep_abi!(
    Glm52Ep32DeepEpAbi,
    ffi::Glm52Ep32DeepEpCtx,
    last_error => ffi::glm52_ep32_deepep_last_error,
    info => ffi::glm52_ep32_deepep_info,
    unique_id => ffi::glm52_ep32_deepep_unique_id,
    ctx_create => ffi::glm52_ep32_deepep_ctx_create,
    ctx_destroy => ffi::glm52_ep32_deepep_ctx_destroy,
    decode_dispatch => ffi::glm52_ep32_deepep_decode_dispatch,
    decode_combine => ffi::glm52_ep32_deepep_decode_combine,
);

#[cfg(feature = "glm52")]
deepep_abi!(
    Glm52Ep64DeepEpAbi,
    ffi::Glm52Ep64DeepEpCtx,
    last_error => ffi::glm52_ep64_deepep_last_error,
    info => ffi::glm52_ep64_deepep_info,
    unique_id => ffi::glm52_ep64_deepep_unique_id,
    ctx_create => ffi::glm52_ep64_deepep_ctx_create,
    ctx_destroy => ffi::glm52_ep64_deepep_ctx_destroy,
    decode_dispatch => ffi::glm52_ep64_deepep_decode_dispatch,
    decode_combine => ffi::glm52_ep64_deepep_decode_combine,
);

fn shim_error<A: DeepEpAbi>(what: &str) -> anyhow::Error {
    // Safety: last_error returns a valid thread-local C string.
    let message = unsafe { CStr::from_ptr(A::last_error()) };
    anyhow!("{what}: {}", message.to_string_lossy())
}

macro_rules! shim_call {
    ($abi:ty, $what:literal, $call:expr) => {{
        let rc = $call;
        ensure!(rc == 0, shim_error::<$abi>($what));
    }};
}

/// Baked shim capacities of the Kimi-K2 single-node 8-rank config.
pub fn deepep_info() -> DeepEpInfo {
    KimiDeepEpAbi::info()
}

/// Baked shim capacities of the GLM5.2 DP1/EP8 config.
#[cfg(feature = "glm52")]
pub fn glm52_deepep_info() -> DeepEpInfo {
    Glm52DeepEpAbi::info()
}

/// NCCL unique id, generated on rank 0 and shared with all ranks.
pub fn deepep_unique_id() -> Result<[u8; 128]> {
    deepep_unique_id_for::<KimiDeepEpAbi>()
}

/// NCCL unique id via the GLM5.2 EP shim instantiation of the given width.
#[cfg(feature = "glm52")]
pub fn glm52_ep_deepep_unique_id(ranks: usize) -> Result<[u8; 128]> {
    match ranks {
        4 => deepep_unique_id_for::<Glm52Ep4DeepEpAbi>(),
        8 => deepep_unique_id_for::<Glm52DeepEpAbi>(),
        16 => deepep_unique_id_for::<Glm52Ep16DeepEpAbi>(),
        32 => deepep_unique_id_for::<Glm52Ep32DeepEpAbi>(),
        64 => deepep_unique_id_for::<Glm52Ep64DeepEpAbi>(),
        other => anyhow::bail!("GLM5.2 DeepEP has no shim instantiation for EP{other}"),
    }
}

fn deepep_unique_id_for<A: DeepEpAbi>() -> Result<[u8; 128]> {
    let mut id = [0u8; 128];
    // Safety: out buffer is 128 bytes as required.
    let rc = unsafe { A::unique_id(id.as_mut_ptr()) };
    ensure!(rc == 0, shim_error::<A>("deepep_unique_id"));
    Ok(id)
}

/// Dispatch-side scratch, allocated once per rank and reused every layer.
pub struct DeepEpDispatchScratch {
    /// Deterministic-prologue per-SM rank counters.
    rank_count: CudaSlice<i32>,
    /// Per-(token, topk) destination slot indices.
    dst_slot: CudaSlice<i32>,
    /// Received-token prefix sum per source rank (the combine handle).
    psum_rank: CudaSlice<i32>,
    /// Received-token aligned offsets per local expert (exclusive form,
    /// `num_local_experts + 1` entries; the kernel uses the first
    /// `num_local_experts` as atomic counters in expanded mode).
    pub psum_expert: CudaSlice<i32>,
}

impl DeepEpDispatchScratch {
    fn new(ctx: &DeviceContext, info: &DeepEpInfo, max_tokens: usize) -> Result<Self> {
        Ok(Self {
            rank_count: ctx
                .stream
                .alloc_zeros::<i32>(info.prologue_rank_count_len as usize)?,
            dst_slot: ctx
                .stream
                .alloc_zeros::<i32>(max_tokens * info.num_topk as usize)?,
            psum_rank: ctx.stream.alloc_zeros::<i32>(info.num_ranks as usize)?,
            psum_expert: ctx
                .stream
                .alloc_zeros::<i32>(info.num_local_experts as usize + 1)?,
        })
    }

    /// Decode-path scratch for the Kimi shim config.
    pub fn new_decode(ctx: &DeviceContext) -> Result<Self> {
        Self::new_decode_with(ctx, &deepep_info())
    }

    /// Prefill-path scratch for the Kimi shim config.
    pub fn new_prefill(ctx: &DeviceContext) -> Result<Self> {
        Self::new_prefill_with(ctx, &deepep_info())
    }

    /// Decode-path scratch sized from an explicit shim config.
    pub fn new_decode_with(ctx: &DeviceContext, info: &DeepEpInfo) -> Result<Self> {
        Self::new(ctx, info, info.decode_max_tokens_per_rank as usize)
    }

    /// Prefill-path scratch sized from an explicit shim config.
    fn new_prefill_with(ctx: &DeviceContext, info: &DeepEpInfo) -> Result<Self> {
        Self::new(ctx, info, info.prefill_max_tokens_per_rank as usize)
    }
}

/// Receive counts published by a prefill dispatch (CPU-synced).
#[derive(Clone, Copy, Debug)]
pub struct DeepEpPrefillCounts {
    /// Tokens received (src_metadata rows).
    pub num_recv_tokens: usize,
    /// Expanded slots (recv x rows); already segment-aligned.
    pub num_expanded_tokens: usize,
}

/// Per-rank DeepEP context over one shim instantiation. Creation and
/// destruction are collective — every rank's worker thread must call them
/// together, device set.
pub struct DeepEpBase<A: DeepEpAbi> {
    ctx: NonNull<A::RawCtx>,
    info: DeepEpInfo,
}

/// The Kimi-K2 shim instantiation (deepep_* symbols).
pub type DeepEp = DeepEpBase<KimiDeepEpAbi>;

/// The GLM5.2 EP8 shim instantiation (`glm52_deepep_*` symbols).
#[cfg(feature = "glm52")]
pub type Glm52DeepEp = DeepEpBase<Glm52DeepEpAbi>;

// One context per rank thread; the shim has no thread-affine state beyond
// the thread-local error string.
unsafe impl<A: DeepEpAbi> Send for DeepEpBase<A> {}

impl<A: DeepEpAbi> DeepEpBase<A> {
    /// Collective: all ranks must call concurrently with the same unique id.
    pub fn new(unique_id: &[u8; 128], num_ranks: usize, rank_idx: usize) -> Result<Self> {
        let mut ctx = std::ptr::null_mut();
        // Safety: unique_id is 128 bytes; out pointer is valid.
        let rc = unsafe {
            A::ctx_create(
                unique_id.as_ptr(),
                i32::try_from(num_ranks)?,
                i32::try_from(rank_idx)?,
                &raw mut ctx,
            )
        };
        ensure!(rc == 0, shim_error::<A>("deepep_ctx_create"));
        Ok(Self {
            ctx: NonNull::new(ctx).ok_or_else(|| anyhow!("deepep_ctx_create returned null"))?,
            info: A::info(),
        })
    }

    /// Decode dispatch: deterministic prologue + dispatch + copy epilogue in
    /// one stream-ordered call, fixed worst-case output shapes. The recv
    /// buffers must be sized at the published worst case
    /// (`decode_worst_expanded_tokens` / `decode_worst_recv_tokens`).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_dispatch(
        &self,
        ctx: &DeviceContext,
        x: &CudaSlice<bf16>,
        topk_idx: &CudaSlice<i32>,
        topk_weights: &CudaSlice<f32>,
        num_tokens: usize,
        scratch: &mut DeepEpDispatchScratch,
        recv_x: &mut CudaSlice<bf16>,
        recv_topk_weights: &mut CudaSlice<f32>,
        recv_src_metadata: &mut CudaSlice<i32>,
    ) -> Result<()> {
        let info = &self.info;
        ensure!(
            num_tokens <= info.decode_max_tokens_per_rank as usize,
            "decode dispatch: {num_tokens} tokens exceeds the baked cap {}",
            info.decode_max_tokens_per_rank
        );
        self.validate_dispatch_inputs(x, topk_idx, topk_weights, num_tokens)?;
        let expanded = info.decode_worst_expanded_tokens as usize;
        ensure!(
            recv_x.len() >= expanded * info.hidden as usize
                && recv_topk_weights.len() >= expanded
                && recv_src_metadata.len()
                    >= info.decode_worst_recv_tokens as usize * (info.num_topk as usize + 2),
            "decode dispatch: recv buffers below the published worst case"
        );

        let stream = &ctx.stream;
        let (x_ptr, _xg) = x.device_ptr(stream);
        let (idx_ptr, _ig) = topk_idx.device_ptr(stream);
        let (w_ptr, _wg) = topk_weights.device_ptr(stream);
        let (rc_ptr, _rcg) = scratch.rank_count.device_ptr_mut(stream);
        let (slot_ptr, _sg) = scratch.dst_slot.device_ptr_mut(stream);
        let (pr_ptr, _prg) = scratch.psum_rank.device_ptr_mut(stream);
        let (pe_ptr, _peg) = scratch.psum_expert.device_ptr_mut(stream);
        let (rx_ptr, _rxg) = recv_x.device_ptr_mut(stream);
        let (rw_ptr, _rwg) = recv_topk_weights.device_ptr_mut(stream);
        let (rm_ptr, _rmg) = recv_src_metadata.device_ptr_mut(stream);

        // Safety: all pointers are live device allocations sized per the
        // shim's published capacities (validated above / at construction).
        shim_call!(A, "deepep_decode_dispatch", unsafe {
            A::decode_dispatch(
                self.ctx.as_ptr(),
                stream.cu_stream().cast(),
                x_ptr as *const _,
                idx_ptr as *const i32,
                w_ptr as *const f32,
                num_tokens as i32,
                rc_ptr as *mut i32,
                slot_ptr as *mut i32,
                pr_ptr as *mut i32,
                pe_ptr as *mut i32,
                rx_ptr as *mut _,
                rw_ptr as *mut f32,
                rm_ptr as *mut i32,
            )
        });
        Ok(())
    }

    /// Decode combine + reduction. `x` carries the (already weighted) expert
    /// outputs in the expanded slots from the matching dispatch;
    /// `src_metadata` and `scratch` must come from that same dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_combine(
        &self,
        ctx: &DeviceContext,
        x: &CudaSlice<bf16>,
        scratch: &DeepEpDispatchScratch,
        src_metadata: &CudaSlice<i32>,
        topk_idx: &CudaSlice<i32>,
        num_tokens: usize,
        combined_x: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let info = &self.info;
        ensure!(
            num_tokens <= info.decode_max_tokens_per_rank as usize,
            "decode combine: {num_tokens} tokens exceeds the baked cap {}",
            info.decode_max_tokens_per_rank
        );
        ensure!(
            x.len() >= info.decode_worst_expanded_tokens as usize * info.hidden as usize,
            "decode combine: x below worst-case expanded capacity"
        );
        ensure!(
            combined_x.len() >= num_tokens * info.hidden as usize,
            "decode combine: combined_x too small"
        );

        let stream = &ctx.stream;
        let (x_ptr, _xg) = x.device_ptr(stream);
        let (sm_ptr, _sg) = src_metadata.device_ptr(stream);
        let (pr_ptr, _pg) = scratch.psum_rank.device_ptr(stream);
        let (idx_ptr, _ig) = topk_idx.device_ptr(stream);
        let (out_ptr, _og) = combined_x.device_ptr_mut(stream);

        // Safety: pointers are live; shapes validated above.
        shim_call!(A, "deepep_decode_combine", unsafe {
            A::decode_combine(
                self.ctx.as_ptr(),
                stream.cu_stream().cast(),
                x_ptr as *const _,
                sm_ptr as *const i32,
                pr_ptr as *const i32,
                idx_ptr as *const i32,
                num_tokens as i32,
                out_ptr as *mut _,
            )
        });
        Ok(())
    }

    fn validate_dispatch_inputs(
        &self,
        x: &CudaSlice<bf16>,
        topk_idx: &CudaSlice<i32>,
        topk_weights: &CudaSlice<f32>,
        num_tokens: usize,
    ) -> Result<()> {
        let info = &self.info;
        let topk = info.num_topk as usize;
        ensure!(
            x.len() >= num_tokens * info.hidden as usize,
            "dispatch: x smaller than num_tokens × hidden"
        );
        ensure!(
            topk_idx.len() >= num_tokens * topk && topk_weights.len() >= num_tokens * topk,
            "dispatch: topk arrays smaller than num_tokens × topk"
        );
        Ok(())
    }
}

impl<A: DeepEpPrefillAbi> DeepEpBase<A> {
    /// Prefill dispatch send: prologue + dispatch. Follow with
    /// [`Self::prefill_wait_counts`] and [`Self::prefill_dispatch_recv`].
    pub fn prefill_dispatch_send(
        &self,
        ctx: &DeviceContext,
        x: &CudaSlice<bf16>,
        topk_idx: &CudaSlice<i32>,
        topk_weights: &CudaSlice<f32>,
        num_tokens: usize,
        scratch: &mut DeepEpDispatchScratch,
    ) -> Result<()> {
        let info = &self.info;
        ensure!(
            num_tokens <= info.prefill_max_tokens_per_rank as usize,
            "prefill dispatch: {num_tokens} tokens exceeds the baked cap {}",
            info.prefill_max_tokens_per_rank
        );
        self.validate_dispatch_inputs(x, topk_idx, topk_weights, num_tokens)?;

        let stream = &ctx.stream;
        let (x_ptr, _xg) = x.device_ptr(stream);
        let (idx_ptr, _ig) = topk_idx.device_ptr(stream);
        let (w_ptr, _wg) = topk_weights.device_ptr(stream);
        let (rc_ptr, _rcg) = scratch.rank_count.device_ptr_mut(stream);
        let (slot_ptr, _sg) = scratch.dst_slot.device_ptr_mut(stream);
        let (pr_ptr, _prg) = scratch.psum_rank.device_ptr_mut(stream);
        let (pe_ptr, _peg) = scratch.psum_expert.device_ptr_mut(stream);

        // Safety: pointers are live; shapes validated above.
        shim_call!(A, "deepep_prefill_dispatch_send", unsafe {
            A::prefill_dispatch_send(
                self.ctx.as_ptr(),
                stream.cu_stream().cast(),
                x_ptr as *const _,
                idx_ptr as *const i32,
                w_ptr as *const f32,
                num_tokens as i32,
                rc_ptr as *mut i32,
                slot_ptr as *mut i32,
                pr_ptr as *mut i32,
                pe_ptr as *mut i32,
            )
        });
        Ok(())
    }

    /// Blocks the CPU on pinned counters until this rank's receive counts
    /// arrive.
    pub fn prefill_wait_counts(&self) -> Result<DeepEpPrefillCounts> {
        let mut num_recv_tokens = 0i32;
        let mut num_expanded_tokens = 0i32;
        // Safety: out pointers are valid.
        shim_call!(A, "deepep_prefill_wait_counts", unsafe {
            A::prefill_wait_counts(
                self.ctx.as_ptr(),
                &raw mut num_recv_tokens,
                &raw mut num_expanded_tokens,
            )
        });
        Ok(DeepEpPrefillCounts {
            num_recv_tokens: num_recv_tokens as usize,
            num_expanded_tokens: num_expanded_tokens as usize,
        })
    }

    /// Prefill copy epilogue into buffers sized from the synced counts.
    pub fn prefill_dispatch_recv(
        &self,
        ctx: &DeviceContext,
        counts: DeepEpPrefillCounts,
        scratch: &DeepEpDispatchScratch,
        recv_x: &mut CudaSlice<bf16>,
        recv_topk_weights: &mut CudaSlice<f32>,
        recv_src_metadata: &mut CudaSlice<i32>,
    ) -> Result<()> {
        let info = &self.info;
        ensure!(
            recv_x.len() >= counts.num_expanded_tokens * info.hidden as usize
                && recv_topk_weights.len() >= counts.num_expanded_tokens
                && recv_src_metadata.len() >= counts.num_recv_tokens * (info.num_topk as usize + 2),
            "prefill recv: buffers below the synced counts"
        );

        let stream = &ctx.stream;
        let (pr_ptr, _prg) = scratch.psum_rank.device_ptr(stream);
        let (pe_ptr, _peg) = scratch.psum_expert.device_ptr(stream);
        let (rx_ptr, _rxg) = recv_x.device_ptr_mut(stream);
        let (rw_ptr, _rwg) = recv_topk_weights.device_ptr_mut(stream);
        let (rm_ptr, _rmg) = recv_src_metadata.device_ptr_mut(stream);

        // Safety: pointers are live; capacities validated above.
        shim_call!(A, "deepep_prefill_dispatch_recv", unsafe {
            A::prefill_dispatch_recv(
                self.ctx.as_ptr(),
                stream.cu_stream().cast(),
                counts.num_recv_tokens as i32,
                pr_ptr as *const i32,
                pe_ptr as *const i32,
                rx_ptr as *mut _,
                rw_ptr as *mut f32,
                rm_ptr as *mut i32,
            )
        });
        Ok(())
    }

    /// Prefill combine + reduction over the synced counts.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_combine(
        &self,
        ctx: &DeviceContext,
        x: &CudaSlice<bf16>,
        scratch: &DeepEpDispatchScratch,
        src_metadata: &CudaSlice<i32>,
        counts: DeepEpPrefillCounts,
        topk_idx: &CudaSlice<i32>,
        num_tokens: usize,
        combined_x: &mut CudaSlice<bf16>,
    ) -> Result<()> {
        let info = &self.info;
        ensure!(
            num_tokens <= info.prefill_max_tokens_per_rank as usize,
            "prefill combine: {num_tokens} tokens exceeds the baked cap {}",
            info.prefill_max_tokens_per_rank
        );
        ensure!(
            x.len() >= counts.num_expanded_tokens * info.hidden as usize
                && combined_x.len() >= num_tokens * info.hidden as usize,
            "prefill combine: buffer below the synced counts"
        );

        let stream = &ctx.stream;
        let (x_ptr, _xg) = x.device_ptr(stream);
        let (sm_ptr, _sg) = src_metadata.device_ptr(stream);
        let (pr_ptr, _pg) = scratch.psum_rank.device_ptr(stream);
        let (idx_ptr, _ig) = topk_idx.device_ptr(stream);
        let (out_ptr, _og) = combined_x.device_ptr_mut(stream);

        // Safety: pointers are live; shapes validated above.
        shim_call!(A, "deepep_prefill_combine", unsafe {
            A::prefill_combine(
                self.ctx.as_ptr(),
                stream.cu_stream().cast(),
                x_ptr as *const _,
                sm_ptr as *const i32,
                pr_ptr as *const i32,
                counts.num_recv_tokens as i32,
                idx_ptr as *const i32,
                num_tokens as i32,
                out_ptr as *mut _,
            )
        });
        Ok(())
    }
}

impl<A: DeepEpAbi> Drop for DeepEpBase<A> {
    /// Collective: all ranks must drop together (the shim synchronizes the
    /// device and barriers before releasing NCCL resources).
    fn drop(&mut self) {
        // Safety: ctx is live and owned.
        let rc = unsafe { A::ctx_destroy(self.ctx.as_ptr()) };
        if rc != 0 {
            eprintln!("{}", shim_error::<A>("deepep_ctx_destroy"));
        }
    }
}
