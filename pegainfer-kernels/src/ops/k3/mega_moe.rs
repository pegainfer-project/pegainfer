//! Kimi-K3 routed experts through DeepGEMM's fused SM100 MegaMoE FP8 x FP4
//! kernel, AOT-instantiated (no JIT, no torch).
//!
//! Where the masked chain is seven launches (route metadata, gather+quant,
//! scale pack, W13 GEMM, situ+quant, scale pack, W2 GEMM, combine), MegaMoE is
//! one persistent grid that does dispatch, both GEMMs, the activation, the
//! mid-quantization and the combine. Everything it needs beyond the weights
//! lives in one flat "symmetric" byte slab: at `ep_size == 1` that is a plain
//! device allocation and the kernel's cross-rank barriers degrade to
//! grid-local synchronisation. Above one rank every rank allocates its own slab
//! on its own device and the whole table of base pointers is handed to every
//! launch; the kernel does its own NVLink pairing over those pointers, so the
//! host issues no collective of any kind.
//!
//! Two semantic differences from the masked chain are inherent to the fused
//! kernel and expected to move logits slightly:
//!
//! * routing weights multiply the activation *before* the W2 GEMM, whereas the
//!   chain applies them at combine time;
//! * the mid-quantization is per-32 e4m3 rather than the chain's per-128.
//!
//! See `csrc/k3/k3_mega_moe_sm100.cu` for the layout and instantiation
//! contract.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

/// Number of sub-buffer offsets the symmetric-buffer layout reports.
pub const K3_MEGA_NUM_SUB_BUFFERS: usize = 12;

/// Weight-row interleave granularity for the fused gate|up projection.
const K3_MEGA_INTERLEAVE_GRAN: usize = 8;
/// MXFP4 / activation scale-factor group size along K.
const K3_MEGA_SF_GROUP_K: usize = 32;
/// K elements covered by one packed i32 scale word.
const K3_MEGA_SF_WORD_K: usize = K3_MEGA_SF_GROUP_K * 4;

/// Which activation the fused kernel applies between the two GEMMs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum K3MegaActivation {
    /// `gate * sigmoid(gate) * up` — upstream's default, kept as a regression
    /// handle against the validated Python path.
    Swiglu,
    /// K3's `4 * tanh(gate / 4) * sigmoid(gate) * 25 * tanh(up / 25)`.
    Situ,
}

impl K3MegaActivation {
    const fn abi_kind(self) -> i32 {
        match self {
            Self::Swiglu => 0,
            Self::Situ => 1,
        }
    }
}

/// Byte offsets of the twelve sub-buffers inside the symmetric slab, in the
/// same order the upstream Python wrapper slices them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct K3MegaSymmLayout {
    /// Total slab size in bytes.
    pub num_bytes: usize,
    offsets: [u64; K3_MEGA_NUM_SUB_BUFFERS],
    /// Ring capacity in tokens (a kernel template parameter upstream).
    pub ring_tokens: usize,
    /// Scale-factor ring capacity in tokens.
    pub sf_ring_tokens: usize,
}

impl K3MegaSymmLayout {
    /// Offset of the FP8 activation region (`x`).
    #[must_use]
    pub const fn x(&self) -> u64 {
        self.offsets[0]
    }

    /// Offset of the packed activation scale-factor region (`x_sf`).
    #[must_use]
    pub const fn x_sf(&self) -> u64 {
        self.offsets[1]
    }

    /// Offset of the i64 top-k expert-id region.
    #[must_use]
    pub const fn topk_idx(&self) -> u64 {
        self.offsets[2]
    }

    /// Offset of the f32 top-k routing-weight region.
    #[must_use]
    pub const fn topk_weights(&self) -> u64 {
        self.offsets[3]
    }

    /// All twelve offsets, as the launch entry point expects them.
    #[must_use]
    pub const fn offsets(&self) -> &[u64; K3_MEGA_NUM_SUB_BUFFERS] {
        &self.offsets
    }
}

/// Token-count alignment the MegaMoE API enforces on `num_max_tokens_per_rank`.
#[must_use]
pub fn k3_mega_token_alignment() -> usize {
    // SAFETY: a pure constant getter with no arguments.
    (unsafe { ffi::k3_mega_token_alignment() }) as usize
}

/// Token capacity one rank's symmetric slab and the AOT kernels are built for
/// (`num_max_tokens_per_rank`): the chunked-prefill ceiling. The ring
/// capacities derived from it are kernel template parameters, so the launch
/// accepts exactly this value and a slab must be allocated at exactly this
/// size, whatever the executor's live batch is.
#[must_use]
pub fn k3_mega_max_tokens_per_rank() -> usize {
    // SAFETY: a pure constant getter with no arguments.
    (unsafe { ffi::k3_mega_max_tokens_per_rank() }) as usize
}

/// One rank's `CUmemFabricHandle`, as raw bytes: what a cross-machine EP
/// group's rendezvous actually exchanges.
pub const K3_MEGA_FABRIC_HANDLE_BYTES: usize = 64;

/// Whether the AOT matrix carries a MegaMoE kernel for this world (GLOBAL
/// expert count x rank count, situ activation). One source of truth — the
/// instantiation list lives in the CUDA TUs and this asks it.
#[must_use]
pub fn k3_mega_world_supported(num_experts: usize, num_ranks: usize) -> bool {
    let (Ok(experts), Ok(ranks)) = (i32::try_from(num_experts), i32::try_from(num_ranks)) else {
        return false;
    };
    // SAFETY: a pure constant predicate with no device state.
    (unsafe { ffi::k3_mega_world_supported(experts, ranks) }) != 0
}

/// Whether `device_ordinal` can allocate fabric-exportable memory at all
/// (driver + IMEX support): the preflight for a cross-machine EP fleet.
pub fn k3_mega_fabric_supported(device_ordinal: usize) -> Result<bool> {
    let mut supported = 0i32;
    // SAFETY: an ordinal-only attribute query.
    let result = unsafe {
        ffi::k3_mega_fabric_supported(i32::try_from(device_ordinal)?, &raw mut supported)
    };
    result
        .result()
        .map_err(|err| anyhow!("K3 fabric support query on device {device_ordinal}: {err}"))?;
    Ok(supported != 0)
}

/// Allocate one rank's fabric-exportable symmetric slab: `num_bytes` on
/// `device_ordinal`, mapped and access-granted for every local device, zeroed
/// and synchronized. Returns the device pointer and the fabric handle a peer
/// process imports. Process-lifetime — an EP group dies as a fleet, so
/// nothing ever frees a slab.
pub fn k3_mega_fabric_slab_alloc(
    device_ordinal: usize,
    num_bytes: usize,
) -> Result<(i64, [u8; K3_MEGA_FABRIC_HANDLE_BYTES])> {
    ensure!(num_bytes > 0, "K3 fabric slab needs a non-zero size");
    let mut ptr = 0i64;
    let mut handle = [0u8; K3_MEGA_FABRIC_HANDLE_BYTES];
    // SAFETY: out-pointers are valid for the writes the FFI contract names.
    let result = unsafe {
        ffi::k3_mega_fabric_slab_alloc(
            i32::try_from(device_ordinal)?,
            u64::try_from(num_bytes)?,
            &raw mut ptr,
            handle.as_mut_ptr(),
        )
    };
    result.result().map_err(|err| {
        anyhow!(
            "K3 fabric slab allocation of {num_bytes} bytes on device {device_ordinal} failed \
             (is the NVLink IMEX domain configured?): {err}"
        )
    })?;
    Ok((ptr, handle))
}

/// Import a peer rank's fabric handle and map its slab for every local
/// device. `num_bytes` must be the peer's slab size before granularity
/// rounding (every rank derives it from the same layout, so it is).
pub fn k3_mega_fabric_slab_import(
    handle: &[u8; K3_MEGA_FABRIC_HANDLE_BYTES],
    num_bytes: usize,
    device_ordinal: usize,
) -> Result<i64> {
    ensure!(num_bytes > 0, "K3 fabric slab import needs a non-zero size");
    let mut ptr = 0i64;
    // SAFETY: the handle buffer is 64 bytes by type; the out-pointer is valid.
    let result = unsafe {
        ffi::k3_mega_fabric_slab_import(
            handle.as_ptr(),
            u64::try_from(num_bytes)?,
            i32::try_from(device_ordinal)?,
            &raw mut ptr,
        )
    };
    result.result().map_err(|err| {
        anyhow!(
            "K3 fabric slab import of {num_bytes} bytes on device {device_ordinal} failed \
             (is the peer's node in this node's IMEX domain?): {err}"
        )
    })?;
    Ok(ptr)
}

/// Open the device pair `(self_ordinal, peer_ordinal)` for the fused kernel's
/// cross-rank addressing.
///
/// Two grants, not one: peer access so `self_ordinal`'s context can address
/// `peer_ordinal`'s memory, and a memory-pool access grant so `self_ordinal`'s
/// own stream-ordered allocations are addressable from `peer_ordinal`. Peer
/// access alone does not cover pool allocations, and everything here comes from
/// the stream-ordered allocator — that omission reads as an illegal address
/// inside the kernel, not as an error at setup.
///
/// The pool grant only reliably covers allocations made after it, so call this
/// for every device that will ever address this rank's slab BEFORE allocating
/// it. Idempotent, and a no-op for the self pair.
pub fn k3_mega_open_peer_access(self_ordinal: usize, peer_ordinal: usize) -> Result<()> {
    // SAFETY: an ordinal-only call into the CUDA runtime.
    let result = unsafe {
        ffi::k3_mega_open_peer_access(i32::try_from(self_ordinal)?, i32::try_from(peer_ordinal)?)
    };
    result.result().map_err(|err| {
        anyhow!("K3 MegaMoE peer access {self_ordinal} -> {peer_ordinal} failed: {err}")
    })
}

/// Symmetric-buffer sizing for one rank.
///
/// Pure host arithmetic over the shapes, the candidate block sizes and the SM
/// count — it allocates nothing and touches no device state, but the numbers
/// are template parameters of the AOT kernel, so the caller must allocate
/// exactly this and not a rounded-up variant.
#[allow(clippy::too_many_arguments)]
pub fn k3_mega_symm_buffer_layout(
    num_ranks: usize,
    num_experts: usize,
    num_max_tokens_per_rank: usize,
    num_topk: usize,
    hidden: usize,
    intermediate_hidden: usize,
    num_sms: usize,
) -> Result<K3MegaSymmLayout> {
    let alignment = k3_mega_token_alignment();
    ensure!(
        num_max_tokens_per_rank > 0 && num_max_tokens_per_rank.is_multiple_of(alignment),
        "K3 MegaMoE needs num_max_tokens_per_rank a multiple of {alignment}, got {num_max_tokens_per_rank}"
    );
    ensure!(
        num_ranks > 0 && num_experts > 0 && num_experts.is_multiple_of(num_ranks),
        "K3 MegaMoE needs num_experts ({num_experts}) divisible by num_ranks ({num_ranks})"
    );
    let mut num_bytes: u64 = 0;
    let mut offsets = [0u64; K3_MEGA_NUM_SUB_BUFFERS];
    let mut ring_tokens: i32 = 0;
    let mut sf_ring_tokens: i32 = 0;
    let result = unsafe {
        ffi::k3_mega_symm_buffer_layout_cuda(
            i32::try_from(num_ranks)?,
            i32::try_from(num_experts)?,
            i32::try_from(num_max_tokens_per_rank)?,
            i32::try_from(num_topk)?,
            i32::try_from(hidden)?,
            i32::try_from(intermediate_hidden)?,
            i32::try_from(num_sms)?,
            &raw mut num_bytes,
            offsets.as_mut_ptr(),
            &raw mut ring_tokens,
            &raw mut sf_ring_tokens,
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("K3 MegaMoE symmetric-buffer layout failed: {err}"))?;
    Ok(K3MegaSymmLayout {
        num_bytes: usize::try_from(num_bytes)?,
        offsets,
        ring_tokens: usize::try_from(ring_tokens)?,
        sf_ring_tokens: usize::try_from(sf_ring_tokens)?,
    })
}

/// Gate/up interleave over the packed-FP4 W13 bytes: `[groups, n, k / 2]` u8 in
/// split-half `[gate | up]` row order out into granularity-8 interleaved rows.
/// Loader-time helper.
pub fn k3_mega_prepare_l1_weights_launch(
    ctx: &DeviceContext,
    groups: usize,
    n: usize,
    k: usize,
    src: &CudaSlice<u8>,
    dst: &mut CudaSlice<u8>,
) -> Result<()> {
    ensure!(
        groups > 0
            && n > 0
            && k > 0
            && k.is_multiple_of(2)
            && n.is_multiple_of(2 * K3_MEGA_INTERLEAVE_GRAN),
        "K3 MegaMoE L1 weight interleave needs k%2=0 and n%16=0, got groups={groups}, n={n}, k={k}"
    );
    let bytes = groups * n * (k / 2);
    ensure!(
        src.len() >= bytes && dst.len() >= bytes,
        "K3 MegaMoE L1 weight interleave buffers too small ({bytes} bytes needed): src {}, dst {}",
        src.len(),
        dst.len()
    );
    let (src_ptr, _src_guard) = src.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = dst.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::k3_mega_prepare_l1_weights_cuda(
            src_ptr as *const u8,
            dst_ptr as *mut u8,
            i32::try_from(groups)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("K3 MegaMoE L1 weight interleave launch failed: {err}"))
}

/// Checkpoint UE8M0 scales (`[groups, n, k / 32]` u8, K-major) -> the MegaMoE
/// weight scale tensor (`[groups, k / 128, n]` i32, MN-major) with the UTCCP
/// row transpose, plus the gate/up interleave when `interleave` is set (W13).
/// Loader-time helper.
#[allow(clippy::too_many_arguments)]
pub fn k3_mega_prepare_sf_launch(
    ctx: &DeviceContext,
    groups: usize,
    n: usize,
    k: usize,
    interleave: bool,
    sf: &CudaSlice<u8>,
    packed: &mut CudaSlice<i32>,
) -> Result<()> {
    ensure!(
        groups > 0
            && n.is_multiple_of(128)
            && k > 0
            && k.is_multiple_of(K3_MEGA_SF_WORD_K)
            && (!interleave || n.is_multiple_of(2 * K3_MEGA_INTERLEAVE_GRAN)),
        "K3 MegaMoE SF prepare needs n%128=0 and k%128=0, got groups={groups}, n={n}, k={k}"
    );
    ensure!(
        sf.len() >= groups * n * (k / K3_MEGA_SF_GROUP_K)
            && packed.len() >= groups * (k / K3_MEGA_SF_WORD_K) * n,
        "K3 MegaMoE SF prepare buffers too small for {groups} groups (n={n}, k={k}): sf {}, packed {}",
        sf.len(),
        packed.len()
    );
    let (sf_ptr, _sf_guard) = sf.device_ptr(&ctx.stream);
    let (packed_ptr, _packed_guard) = packed.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::k3_mega_prepare_sf_cuda(
            sf_ptr as *const u8,
            packed_ptr as *mut i32,
            i32::try_from(groups)?,
            i32::try_from(n)?,
            i32::try_from(k)?,
            i32::from(interleave),
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("K3 MegaMoE SF prepare launch failed: {err}"))
}

/// Write one step's inputs into the symmetric slab: the bf16 latents quantized
/// to e4m3 plus packed group-32 UE8M0 scales, and the routing pair widened to
/// the kernel's i64 ids / f32 weights.
///
/// The slab is written in place through raw offsets rather than typed slices —
/// it is one allocation carrying twelve differently-typed regions.
#[allow(clippy::too_many_arguments)]
pub fn k3_mega_write_inputs_launch(
    ctx: &DeviceContext,
    layout: &K3MegaSymmLayout,
    symm: &mut CudaSlice<u8>,
    num_tokens: usize,
    hidden: usize,
    num_topk: usize,
    latent: &CudaSlice<bf16>,
    topk_idx: &CudaSlice<i32>,
    topk_weight: &CudaSlice<f32>,
) -> Result<()> {
    ensure!(
        hidden.is_multiple_of(K3_MEGA_SF_WORD_K),
        "K3 MegaMoE input quant needs hidden divisible by 128, got {hidden}"
    );
    ensure!(
        symm.len() >= layout.num_bytes,
        "K3 MegaMoE symmetric buffer too small: have {}, layout wants {}",
        symm.len(),
        layout.num_bytes
    );
    ensure!(
        latent.len() >= num_tokens * hidden
            && topk_idx.len() >= num_tokens * num_topk
            && topk_weight.len() >= num_tokens * num_topk,
        "K3 MegaMoE input buffers too small for {num_tokens} tokens: latent {}, topk_idx {}, topk_weight {}",
        latent.len(),
        topk_idx.len(),
        topk_weight.len()
    );
    let (symm_ptr, _symm_guard) = symm.device_ptr_mut(&ctx.stream);
    let (latent_ptr, _latent_guard) = latent.device_ptr(&ctx.stream);
    let (idx_ptr, _idx_guard) = topk_idx.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = topk_weight.device_ptr(&ctx.stream);

    let quant = unsafe {
        ffi::k3_mega_quant_x_cuda(
            latent_ptr as *const ffi::Half,
            (symm_ptr + layout.x()) as *mut u8,
            (symm_ptr + layout.x_sf()) as *mut i32,
            i32::try_from(num_tokens)?,
            i32::try_from(hidden)?,
            i32::try_from(hidden)?,
            i32::try_from(hidden / K3_MEGA_SF_WORD_K)?,
            ctx.stream.cu_stream(),
        )
    };
    quant
        .result()
        .map_err(|err| anyhow!("K3 MegaMoE activation quant launch failed: {err}"))?;

    let routing = unsafe {
        ffi::k3_mega_write_routing_cuda(
            idx_ptr as *const i32,
            weight_ptr as *const f32,
            (symm_ptr + layout.topk_idx()) as *mut i64,
            (symm_ptr + layout.topk_weights()) as *mut f32,
            i32::try_from(num_tokens)?,
            i32::try_from(num_topk)?,
            ctx.stream.cu_stream(),
        )
    };
    routing
        .result()
        .map_err(|err| anyhow!("K3 MegaMoE routing write launch failed: {err}"))
}

/// The fused MegaMoE launch.
///
/// `symm` must already carry this step's inputs (see
/// [`k3_mega_write_inputs_launch`]) and must have been zeroed once at
/// allocation time — the kernel's workspace counters live in the same slab and
/// are self-restoring across launches, but start from zero.
///
/// `symm_ptrs` is the world's base-pointer table: entry `r` is rank `r`'s slab
/// as addressed from this context, so every peer entry needs CUDA peer access
/// already enabled. Entry `shape.rank_idx` must be this rank's own `symm`.
#[allow(clippy::too_many_arguments)]
pub fn k3_mega_moe_launch(
    ctx: &DeviceContext,
    layout: &K3MegaSymmLayout,
    symm: &mut CudaSlice<u8>,
    symm_ptrs: &[i64],
    shape: K3MegaShape,
    activation: K3MegaActivation,
    l1_weights: &CudaSlice<u8>,
    l1_weights_sf: &CudaSlice<i32>,
    l2_weights: &CudaSlice<u8>,
    l2_weights_sf: &CudaSlice<i32>,
    output: &mut CudaSlice<bf16>,
) -> Result<()> {
    shape.validate()?;
    ensure!(
        symm.len() >= layout.num_bytes,
        "K3 MegaMoE symmetric buffer too small: have {}, layout wants {}",
        symm.len(),
        layout.num_bytes
    );
    // Weights are sharded: a rank holds only the experts it owns.
    let local_experts = shape.local_experts();
    ensure!(
        symm_ptrs.len() == shape.num_ranks,
        "K3 MegaMoE needs one base pointer per rank ({} ranks), got {}",
        shape.num_ranks,
        symm_ptrs.len()
    );
    ensure!(
        l1_weights.len() >= local_experts * (2 * shape.intermediate_hidden) * (shape.hidden / 2)
            && l1_weights_sf.len()
                >= local_experts
                    * (shape.hidden / K3_MEGA_SF_WORD_K)
                    * (2 * shape.intermediate_hidden)
            && l2_weights.len() >= local_experts * shape.hidden * (shape.intermediate_hidden / 2)
            && l2_weights_sf.len()
                >= local_experts * (shape.intermediate_hidden / K3_MEGA_SF_WORD_K) * shape.hidden,
        "K3 MegaMoE weight buffers too small for {shape:?}: l1 {}, l1_sf {}, l2 {}, l2_sf {}",
        l1_weights.len(),
        l1_weights_sf.len(),
        l2_weights.len(),
        l2_weights_sf.len()
    );
    ensure!(
        output.len() >= shape.num_tokens * shape.hidden,
        "K3 MegaMoE output too small for {shape:?}: {}",
        output.len()
    );

    let (symm_ptr, _symm_guard) = symm.device_ptr_mut(&ctx.stream);
    let (l1_ptr, _l1_guard) = l1_weights.device_ptr(&ctx.stream);
    let (l1_sf_ptr, _l1_sf_guard) = l1_weights_sf.device_ptr(&ctx.stream);
    let (l2_ptr, _l2_guard) = l2_weights.device_ptr(&ctx.stream);
    let (l2_sf_ptr, _l2_sf_guard) = l2_weights_sf.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = output.device_ptr_mut(&ctx.stream);

    ensure!(
        symm_ptrs[shape.rank_idx] == symm_ptr as i64,
        "K3 MegaMoE rank {} published base {:#x} but is launching on {:#x}",
        shape.rank_idx,
        symm_ptrs[shape.rank_idx],
        symm_ptr
    );
    let offsets = *layout.offsets();

    let result = unsafe {
        ffi::k3_mega_moe_launch_cuda(
            out_ptr as *mut ffi::Half,
            l1_ptr as *const u8,
            l1_sf_ptr as *const i32,
            l2_ptr as *const u8,
            l2_sf_ptr as *const i32,
            symm_ptrs.as_ptr(),
            offsets.as_ptr(),
            i32::try_from(shape.num_ranks)?,
            i32::try_from(shape.rank_idx)?,
            i32::try_from(shape.num_max_tokens_per_rank)?,
            i32::try_from(shape.num_tokens)?,
            i32::try_from(shape.num_experts)?,
            i32::try_from(shape.num_topk)?,
            i32::try_from(shape.hidden)?,
            i32::try_from(shape.intermediate_hidden)?,
            i32::try_from(shape.num_sms)?,
            activation.abi_kind(),
            std::ptr::null_mut(),
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("K3 MegaMoE launch failed: {err}"))
}

/// One MegaMoE call's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct K3MegaShape {
    /// Rows fed this step.
    pub num_tokens: usize,
    /// Row capacity the symmetric buffer and the kernel were built for.
    pub num_max_tokens_per_rank: usize,
    /// GLOBAL routed-expert count. Rank `r` owns the contiguous block
    /// `[r * num_experts / num_ranks, (r + 1) * ...)`, which is exactly how the
    /// kernel derives a token's destination rank from its expert id, so the
    /// routing ids fed to the slab must be global too.
    pub num_experts: usize,
    /// Expert-parallel world size. The instantiated widths depend on the
    /// expert count — ask [`k3_mega_world_supported`].
    pub num_ranks: usize,
    /// This rank's index in that world.
    pub rank_idx: usize,
    /// Router fan-out.
    pub num_topk: usize,
    /// Latent width the experts consume.
    pub hidden: usize,
    /// Per-expert intermediate width.
    pub intermediate_hidden: usize,
    /// Device SM count — the kernel's grid sync is over it, so the launch grid
    /// must match the instantiation exactly.
    pub num_sms: usize,
}

impl K3MegaShape {
    /// Experts this rank holds weights for.
    #[must_use]
    pub const fn local_experts(&self) -> usize {
        self.num_experts / self.num_ranks
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.num_ranks > 0
                && self.rank_idx < self.num_ranks
                && self.num_experts.is_multiple_of(self.num_ranks),
            "K3 MegaMoE rank {} of {} does not partition {} experts",
            self.rank_idx,
            self.num_ranks,
            self.num_experts
        );
        ensure!(
            self.num_tokens <= self.num_max_tokens_per_rank,
            "K3 MegaMoE fed {} tokens but the buffer holds {}",
            self.num_tokens,
            self.num_max_tokens_per_rank
        );
        ensure!(
            self.num_max_tokens_per_rank
                .is_multiple_of(k3_mega_token_alignment()),
            "K3 MegaMoE needs num_max_tokens_per_rank a multiple of {}, got {}",
            k3_mega_token_alignment(),
            self.num_max_tokens_per_rank
        );
        ensure!(
            self.hidden.is_multiple_of(128) && self.intermediate_hidden.is_multiple_of(128),
            "K3 MegaMoE needs hidden/intermediate divisible by 128, got {} and {}",
            self.hidden,
            self.intermediate_hidden
        );
        ensure!(
            self.num_sms == 152,
            "K3 MegaMoE is AOT-instantiated for the GB300 152-SM grid only, got {}",
            self.num_sms
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The protocol maximum must satisfy the upstream token alignment, and the
    /// symmetric-buffer layout must be computable at exactly that value for
    /// every instantiated world size. Host arithmetic only — prints the slab
    /// sizes so a protocol bump shows its memory cost. Skips on builds where
    /// the sm100 TU is stubbed out.
    #[test]
    fn protocol_max_layouts() {
        let max_tokens = k3_mega_max_tokens_per_rank();
        let alignment = k3_mega_token_alignment();
        assert_eq!(
            max_tokens % alignment,
            0,
            "protocol max {max_tokens} is not a multiple of the {alignment} alignment"
        );
        for (experts, ranks) in [
            (224usize, 1usize),
            (224, 4),
            (224, 8),
            (224, 16),
            (896, 8),
            (896, 16),
            (896, 32),
            (896, 64),
        ] {
            assert!(
                k3_mega_world_supported(experts, ranks),
                "world ({experts} experts, {ranks} ranks) must be in the AOT matrix"
            );
            match k3_mega_symm_buffer_layout(ranks, experts, max_tokens, 16, 3584, 3072, 152) {
                Ok(layout) => eprintln!(
                    "experts {experts} ranks {ranks}: slab {:.1} MiB, ring {} tokens, sf ring {} \
                     tokens",
                    layout.num_bytes as f64 / (1024.0 * 1024.0),
                    layout.ring_tokens,
                    layout.sf_ring_tokens,
                ),
                Err(error) => {
                    eprintln!(
                        "skipping experts {experts} ranks {ranks}: sm100 TU not built ({error:#})"
                    );
                }
            }
        }
        assert!(
            !k3_mega_world_supported(224, 64),
            "224 does not divide by 64"
        );
        assert!(
            !k3_mega_world_supported(896, 1),
            "the full model fits no single rank"
        );
    }
}
