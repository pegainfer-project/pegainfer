use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::ops::marlin_face::MarlinAlignBuffers;
use crate::ops::marlin_face::MarlinGemmBuffers;
use crate::ops::marlin_face::launch_marlin_align;
use crate::ops::marlin_face::launch_marlin_gemm;
#[cfg(test)]
use crate::tensor::AxisSpec;
use crate::tensor::DeviceContext;
use crate::tensor::GpuTensor;
#[cfg(test)]
#[cfg(test)]
use crate::tensor::TensorSpec;

pub const KIMI_K2_HIDDEN: usize = 7168;
pub const KIMI_K2_EXPERT_INTERMEDIATE: usize = 2048;
pub const KIMI_K2_SHARED_GATE_UP: usize = 2 * KIMI_K2_EXPERT_INTERMEDIATE;
const KIMI_K2_ROUTED_EXPERTS: usize = 384;
pub const KIMI_K2_EP_WORLD: usize = 8;
pub const KIMI_K2_LOCAL_EXPERTS: usize = KIMI_K2_ROUTED_EXPERTS / KIMI_K2_EP_WORLD;
pub const KIMI_K2_TOPK: usize = 8;
pub const KIMI_K2_INT4_GROUP_SIZE: usize = 32;

pub fn kimi_add_f32_bf16_to_bf16<const DIM: usize>(
    ctx: &DeviceContext,
    a: &CudaSlice<f32>,
    b: &GpuTensor<DIM>,
    out: &mut GpuTensor<DIM>,
) -> Result<()> {
    let elems = DIM * b.seq_len;
    ensure!(
        out.seq_len == b.seq_len,
        "Kimi f32 add seq_len mismatch: b={}, output={}",
        b.seq_len,
        out.seq_len
    );
    ensure!(
        a.len() >= elems,
        "Kimi f32 add input too small: have {}, need {}",
        a.len(),
        elems
    );

    let (a_ptr, _a_guard) = a.device_ptr(&ctx.stream);
    let (b_ptr, _b_guard) = b.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_add_f32_bf16_to_bf16_cuda(
            a_ptr as *const f32,
            b_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            elems as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_residual_add_scaled_f32<const DIM: usize>(
    ctx: &DeviceContext,
    hidden: &GpuTensor<DIM>,
    projected: &GpuTensor<DIM>,
    routed_f32: &CudaSlice<f32>,
    scale: f32,
    out: &mut GpuTensor<DIM>,
) -> Result<()> {
    let elems = DIM * hidden.seq_len;
    ensure!(
        projected.seq_len == hidden.seq_len && out.seq_len == hidden.seq_len,
        "Kimi residual_add_scaled_f32 seq_len mismatch: hidden={}, projected={}, out={}",
        hidden.seq_len,
        projected.seq_len,
        out.seq_len
    );
    ensure!(
        routed_f32.len() >= elems,
        "Kimi residual_add_scaled_f32 routed_f32 too small: have {}, need {}",
        routed_f32.len(),
        elems
    );

    let (hidden_ptr, _g0) = hidden.data.device_ptr(&ctx.stream);
    let (projected_ptr, _g1) = projected.data.device_ptr(&ctx.stream);
    let (routed_ptr, _g2) = routed_f32.device_ptr(&ctx.stream);
    let (out_ptr, _g3) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_residual_add_scaled_f32_cuda(
            hidden_ptr as *const ffi::Half,
            projected_ptr as *const ffi::Half,
            routed_ptr as *const f32,
            scale,
            out_ptr as *mut ffi::Half,
            elems as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

/// `out = bf16(hidden + projected) + scale * routed` with a bf16 routed
/// contribution (the DeepEP combine output dtype).
pub fn kimi_residual_add_scaled_bf16<const DIM: usize>(
    ctx: &DeviceContext,
    hidden: &GpuTensor<DIM>,
    projected: &GpuTensor<DIM>,
    routed: &GpuTensor<DIM>,
    scale: f32,
    out: &mut GpuTensor<DIM>,
) -> Result<()> {
    let elems = DIM * hidden.seq_len;
    ensure!(
        projected.seq_len == hidden.seq_len && out.seq_len == hidden.seq_len,
        "Kimi residual_add_scaled_bf16 seq_len mismatch: hidden={}, projected={}, out={}",
        hidden.seq_len,
        projected.seq_len,
        out.seq_len
    );
    ensure!(
        routed.seq_len >= hidden.seq_len,
        "Kimi residual_add_scaled_bf16 routed too small: have {}, need {}",
        routed.seq_len,
        hidden.seq_len
    );

    let (hidden_ptr, _g0) = hidden.data.device_ptr(&ctx.stream);
    let (projected_ptr, _g1) = projected.data.device_ptr(&ctx.stream);
    let (routed_ptr, _g2) = routed.data.device_ptr(&ctx.stream);
    let (out_ptr, _g3) = out.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_residual_add_scaled_bf16_cuda(
            hidden_ptr as *const ffi::Half,
            projected_ptr as *const ffi::Half,
            routed_ptr as *const ffi::Half,
            scale,
            out_ptr as *mut ffi::Half,
            elems as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KimiInt4ExpertRole {
    W1Gate,
    W3Up,
    W2Down,
}

impl KimiInt4ExpertRole {
    #[must_use]
    const fn expected_shape(self) -> KimiInt4LogicalShape {
        match self {
            Self::W1Gate | Self::W3Up => KimiInt4LogicalShape {
                out_dim: KIMI_K2_EXPERT_INTERMEDIATE,
                in_dim: KIMI_K2_HIDDEN,
            },
            Self::W2Down => KimiInt4LogicalShape {
                out_dim: KIMI_K2_HIDDEN,
                in_dim: KIMI_K2_EXPERT_INTERMEDIATE,
            },
        }
    }

    #[must_use]
    const fn label(self) -> &'static str {
        match self {
            Self::W1Gate => "w1_gate",
            Self::W3Up => "w3_up",
            Self::W2Down => "w2_down",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KimiInt4NibbleOrder {
    LowThenHigh,
    HighThenLow,
}

impl KimiInt4NibbleOrder {
    #[must_use]
    const fn label(self) -> &'static str {
        match self {
            Self::LowThenHigh => "low_then_high",
            Self::HighThenLow => "high_then_low",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KimiInt4Encoding {
    SignedSymmetric,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KimiInt4LogicalShape {
    pub out_dim: usize,
    pub in_dim: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KimiInt4TensorShape {
    experts: usize,
    rows: usize,
    cols: usize,
}

impl KimiInt4TensorShape {
    #[must_use]
    pub const fn elements(self) -> usize {
        self.experts * self.rows * self.cols
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KimiInt4WeightManifest {
    role: KimiInt4ExpertRole,
    global_experts: usize,
    pub local_experts: usize,
    local_expert_offset: usize,
    pub logical_shape: KimiInt4LogicalShape,
    pub packed_shape: KimiInt4TensorShape,
    pub scale_shape: KimiInt4TensorShape,
    group_size: usize,
    nibble_order: KimiInt4NibbleOrder,
    encoding: KimiInt4Encoding,
}

impl KimiInt4WeightManifest {
    #[must_use]
    pub fn ep8(
        role: KimiInt4ExpertRole,
        ep_rank: usize,
        nibble_order: KimiInt4NibbleOrder,
    ) -> Self {
        let logical_shape = role.expected_shape();
        let local_expert_offset = ep_rank * KIMI_K2_LOCAL_EXPERTS;
        Self {
            role,
            global_experts: KIMI_K2_ROUTED_EXPERTS,
            local_experts: KIMI_K2_LOCAL_EXPERTS,
            local_expert_offset,
            logical_shape,
            packed_shape: KimiInt4TensorShape {
                experts: KIMI_K2_LOCAL_EXPERTS,
                rows: logical_shape.out_dim,
                cols: packed_int4_cols(logical_shape.in_dim),
            },
            scale_shape: KimiInt4TensorShape {
                experts: KIMI_K2_LOCAL_EXPERTS,
                rows: logical_shape.out_dim,
                cols: logical_shape.in_dim / KIMI_K2_INT4_GROUP_SIZE,
            },
            group_size: KIMI_K2_INT4_GROUP_SIZE,
            nibble_order,
            encoding: KimiInt4Encoding::SignedSymmetric,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.global_experts == KIMI_K2_ROUTED_EXPERTS,
            "Kimi-K2 routed experts must be {}, got {}",
            KIMI_K2_ROUTED_EXPERTS,
            self.global_experts
        );
        ensure!(
            self.local_experts == KIMI_K2_LOCAL_EXPERTS,
            "Kimi-K2 EP8 rank must own {} local experts, got {}",
            KIMI_K2_LOCAL_EXPERTS,
            self.local_experts
        );
        ensure!(
            self.local_expert_offset + self.local_experts <= self.global_experts,
            "local expert range [{}..{}) exceeds {} global experts",
            self.local_expert_offset,
            self.local_expert_offset + self.local_experts,
            self.global_experts
        );
        ensure!(
            self.logical_shape == self.role.expected_shape(),
            "{} logical shape must be {:?}, got {:?}",
            self.role.label(),
            self.role.expected_shape(),
            self.logical_shape
        );
        ensure!(
            self.group_size == KIMI_K2_INT4_GROUP_SIZE,
            "Kimi-K2 compressed-tensors INT4 group size must be {}, got {}",
            KIMI_K2_INT4_GROUP_SIZE,
            self.group_size
        );
        ensure!(
            self.logical_shape.in_dim.is_multiple_of(self.group_size),
            "input dim {} must be divisible by group size {}",
            self.logical_shape.in_dim,
            self.group_size
        );

        let expected_packed = KimiInt4TensorShape {
            experts: self.local_experts,
            rows: self.logical_shape.out_dim,
            cols: packed_int4_cols(self.logical_shape.in_dim),
        };
        ensure!(
            self.packed_shape == expected_packed,
            "{} weight_packed shape must be {:?}, got {:?}",
            self.role.label(),
            expected_packed,
            self.packed_shape
        );

        let expected_scale = KimiInt4TensorShape {
            experts: self.local_experts,
            rows: self.logical_shape.out_dim,
            cols: self.logical_shape.in_dim / self.group_size,
        };
        ensure!(
            self.scale_shape == expected_scale,
            "{} weight_scale shape must be {:?}, got {:?}",
            self.role.label(),
            expected_scale,
            self.scale_shape
        );
        ensure!(
            self.encoding == KimiInt4Encoding::SignedSymmetric,
            "only signed symmetric INT4 is specified for Kimi-K2, got {:?}",
            self.encoding
        );

        Ok(())
    }

    #[must_use]
    #[cfg(test)]
    fn weight_packed_checkpoint_spec(&self) -> TensorSpec {
        TensorSpec::named(
            "u8",
            "expert_major_int4_packed_checkpoint_offset_binary",
            [
                AxisSpec::named("local_expert", self.packed_shape.experts),
                AxisSpec::named("out", self.packed_shape.rows),
                AxisSpec::named("packed_in_over_2", self.packed_shape.cols),
            ],
        )
    }

    #[must_use]
    fn marlin_packed_u32_elements(&self) -> usize {
        self.local_experts * (self.logical_shape.in_dim / 16) * (self.logical_shape.out_dim * 2)
    }

    #[must_use]
    #[cfg(test)]
    fn weight_packed_marlin_uint4b8_spec(&self) -> TensorSpec {
        TensorSpec::named(
            "u32",
            "expert_major_int4_packed_marlin_uint4b8_noact",
            [
                AxisSpec::named("local_expert", self.local_experts),
                AxisSpec::named("in_tile16", self.logical_shape.in_dim / 16),
                AxisSpec::named("out_x2", self.logical_shape.out_dim * 2),
            ],
        )
    }

    #[must_use]
    #[cfg(test)]
    fn weight_scale_checkpoint_spec(&self) -> TensorSpec {
        TensorSpec::named(
            "bf16",
            "expert_major_group_scale_checkpoint",
            [
                AxisSpec::named("local_expert", self.scale_shape.experts),
                AxisSpec::named("out", self.scale_shape.rows),
                AxisSpec::named("in_group", self.scale_shape.cols),
            ],
        )
    }

    #[must_use]
    #[cfg(test)]
    fn weight_scale_marlin_permuted_spec(&self) -> TensorSpec {
        TensorSpec::named(
            "bf16",
            "expert_major_group_scale_marlin_group_major_perm64",
            [
                AxisSpec::named("local_expert", self.scale_shape.experts),
                AxisSpec::named("in_group", self.scale_shape.cols),
                AxisSpec::named("out", self.scale_shape.rows),
            ],
        )
    }
}

pub struct KimiMarlinInt4Weight<'a> {
    pub manifest: KimiInt4WeightManifest,
    pub weight_packed_uint4b8: &'a CudaSlice<u8>,
    pub weight_scale_permuted: &'a CudaSlice<bf16>,
}

impl KimiMarlinInt4Weight<'_> {
    fn validate(&self) -> Result<()> {
        self.manifest.validate()?;
        ensure!(
            self.weight_packed_uint4b8.len() == self.manifest.packed_shape.elements(),
            "{} Marlin uint4b8 packed len must be {}, got {}",
            self.manifest.role.label(),
            self.manifest.packed_shape.elements(),
            self.weight_packed_uint4b8.len()
        );
        ensure!(
            self.weight_scale_permuted.len() == self.manifest.scale_shape.elements(),
            "{} Marlin permuted scale len must be {}, got {}",
            self.manifest.role.label(),
            self.manifest.scale_shape.elements(),
            self.weight_scale_permuted.len()
        );
        ensure!(
            self.manifest.nibble_order == KimiInt4NibbleOrder::LowThenHigh,
            "{} Marlin package expects low-then-high checkpoint nibbles before repack, got {}",
            self.manifest.role.label(),
            self.manifest.nibble_order.label()
        );
        Ok(())
    }
}

pub struct KimiMarlinFusedW13Int4Weight<'a> {
    pub local_experts: usize,
    pub in_dim: usize,
    pub intermediate_dim: usize,
    pub group_size: usize,
    pub weight_packed_uint4b8: &'a CudaSlice<u8>,
    pub weight_scale_permuted: &'a CudaSlice<bf16>,
}

impl KimiMarlinFusedW13Int4Weight<'_> {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.local_experts == KIMI_K2_LOCAL_EXPERTS,
            "Marlin fused W13 local_experts must be {}, got {}",
            KIMI_K2_LOCAL_EXPERTS,
            self.local_experts
        );
        ensure!(
            self.in_dim == KIMI_K2_HIDDEN,
            "Marlin fused W13 in_dim must be {}, got {}",
            KIMI_K2_HIDDEN,
            self.in_dim
        );
        ensure!(
            self.intermediate_dim == KIMI_K2_EXPERT_INTERMEDIATE,
            "Marlin fused W13 intermediate_dim must be {}, got {}",
            KIMI_K2_EXPERT_INTERMEDIATE,
            self.intermediate_dim
        );
        ensure!(
            self.group_size == KIMI_K2_INT4_GROUP_SIZE,
            "Marlin fused W13 group_size must be {}, got {}",
            KIMI_K2_INT4_GROUP_SIZE,
            self.group_size
        );
        let expected_packed = self.local_experts * (self.in_dim / 16) * (self.intermediate_dim * 4);
        ensure!(
            self.weight_packed_uint4b8.len() == expected_packed * std::mem::size_of::<u32>(),
            "Marlin fused W13 uint4b8 packed len must be {} bytes, got {}",
            expected_packed * std::mem::size_of::<u32>(),
            self.weight_packed_uint4b8.len()
        );
        let expected_scale =
            self.local_experts * (self.in_dim / self.group_size) * (2 * self.intermediate_dim);
        ensure!(
            self.weight_scale_permuted.len() == expected_scale,
            "Marlin fused W13 permuted scale len must be {}, got {}",
            expected_scale,
            self.weight_scale_permuted.len()
        );
        Ok(())
    }
}

pub struct KimiMarlinInt4ExpertWeights<'a> {
    pub w13: KimiMarlinFusedW13Int4Weight<'a>,
    pub w2_down: KimiMarlinInt4Weight<'a>,
}

impl KimiMarlinInt4ExpertWeights<'_> {
    pub fn validate(&self) -> Result<()> {
        self.w13.validate()?;
        self.w2_down.validate()?;
        ensure!(
            self.w2_down.manifest.role == KimiInt4ExpertRole::W2Down,
            "Marlin W2 role mismatch: got {:?}",
            self.w2_down.manifest.role
        );
        Ok(())
    }
}

pub struct KimiMarlinRouteWorkspace {
    sorted_token_ids: CudaSlice<i32>,
    expert_ids: CudaSlice<i32>,
    num_tokens_post_padded: CudaSlice<i32>,
    expert_offsets: CudaSlice<u32>,
    max_active_tokens: usize,
    max_padded_tokens: usize,
    pub max_m_blocks: usize,
    block_size: usize,
    topk: usize,
    local_experts: usize,
}

impl KimiMarlinRouteWorkspace {
    pub fn new(ctx: &DeviceContext, max_active_tokens: usize, block_size: usize) -> Result<Self> {
        ensure!(
            max_active_tokens > 0,
            "Kimi Marlin route max_active_tokens must be positive"
        );
        validate_marlin_block_size(block_size)?;
        let max_padded_tokens = marlin_padded_route_capacity(max_active_tokens, block_size)?;
        let max_m_blocks = max_padded_tokens.div_ceil(block_size);
        Ok(Self {
            sorted_token_ids: ctx.stream.alloc_zeros(max_padded_tokens)?,
            expert_ids: ctx.stream.alloc_zeros(max_m_blocks)?,
            num_tokens_post_padded: ctx.stream.alloc_zeros(1)?,
            expert_offsets: ctx.stream.alloc_zeros(KIMI_K2_LOCAL_EXPERTS + 1)?,
            max_active_tokens,
            max_padded_tokens,
            max_m_blocks,
            block_size,
            topk: KIMI_K2_TOPK,
            local_experts: KIMI_K2_LOCAL_EXPERTS,
        })
    }

    fn validate_for(&self, active_tokens: usize) -> Result<()> {
        validate_marlin_block_size(self.block_size)?;
        ensure!(
            active_tokens > 0,
            "Kimi Marlin route active_tokens must be positive"
        );
        ensure!(
            active_tokens <= self.max_active_tokens,
            "active_tokens {} exceeds Kimi Marlin route workspace capacity {}",
            active_tokens,
            self.max_active_tokens
        );
        let required_padded = marlin_padded_route_capacity(active_tokens, self.block_size)?;
        let required_blocks = required_padded.div_ceil(self.block_size);
        ensure!(
            self.max_padded_tokens >= required_padded
                && self.sorted_token_ids.len() >= self.max_padded_tokens,
            "Marlin sorted_token_ids capacity too small: have {} metadata/{} slice, need {}",
            self.max_padded_tokens,
            self.sorted_token_ids.len(),
            required_padded
        );
        ensure!(
            self.max_m_blocks >= required_blocks && self.expert_ids.len() >= self.max_m_blocks,
            "Marlin expert_ids capacity too small: have {} metadata/{} slice, need {}",
            self.max_m_blocks,
            self.expert_ids.len(),
            required_blocks
        );
        ensure!(
            self.num_tokens_post_padded.len() == 1,
            "num_tokens_post_padded len must be 1, got {}",
            self.num_tokens_post_padded.len()
        );
        ensure!(
            self.expert_offsets.len() == KIMI_K2_LOCAL_EXPERTS + 1,
            "expert_offsets len must be {}, got {}",
            KIMI_K2_LOCAL_EXPERTS + 1,
            self.expert_offsets.len()
        );
        ensure!(
            self.topk == KIMI_K2_TOPK && self.local_experts == KIMI_K2_LOCAL_EXPERTS,
            "Kimi Marlin route workspace constants must be topk={} local_experts={}",
            KIMI_K2_TOPK,
            KIMI_K2_LOCAL_EXPERTS
        );
        Ok(())
    }
}

pub struct KimiMarlinWna16Workspace {
    pub locks: CudaSlice<i32>,
    c_tmp: CudaSlice<f32>,
    max_m_blocks: usize,
    max_padded_tokens: usize,
    max_size_n: usize,
    block_size: usize,
}

impl KimiMarlinWna16Workspace {
    pub fn new(
        ctx: &DeviceContext,
        max_m_blocks: usize,
        max_size_n: usize,
        block_size: usize,
    ) -> Result<Self> {
        validate_marlin_block_size(block_size)?;
        ensure!(
            max_m_blocks > 0,
            "Kimi Marlin WNA16 max_m_blocks must be > 0"
        );
        ensure!(
            max_size_n >= KIMI_K2_EXPERT_INTERMEDIATE && max_size_n.is_multiple_of(64),
            "Kimi Marlin WNA16 max_size_n must be >= {} and divisible by 64, got {}",
            KIMI_K2_EXPERT_INTERMEDIATE,
            max_size_n
        );
        let lock_count = (max_size_n / 64)
            .checked_mul(max_m_blocks)
            .ok_or_else(|| anyhow::anyhow!("Kimi Marlin WNA16 workspace size overflow"))?
            .max(1);
        let max_padded_tokens = max_m_blocks
            .checked_mul(block_size)
            .ok_or_else(|| anyhow::anyhow!("Kimi Marlin WNA16 padded token capacity overflow"))?;
        let mut c_tmp_elements = max_size_n
            .checked_mul(max_padded_tokens)
            .ok_or_else(|| anyhow::anyhow!("Kimi Marlin WNA16 c_tmp capacity overflow"))?;
        if block_size == 8 {
            c_tmp_elements = c_tmp_elements
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("Kimi Marlin WNA16 c_tmp capacity overflow"))?;
        }
        Ok(Self {
            locks: ctx.stream.alloc_zeros(lock_count)?,
            c_tmp: ctx.stream.alloc_zeros(c_tmp_elements.max(1))?,
            max_m_blocks,
            max_padded_tokens,
            max_size_n,
            block_size,
        })
    }

    fn validate_for(&self, routing: &KimiMarlinRouting<'_>, size_n: usize) -> Result<()> {
        validate_marlin_block_size(self.block_size)?;
        ensure!(
            self.block_size == routing.block_size,
            "Kimi Marlin WNA16 workspace block_size {} must match routing {}",
            self.block_size,
            routing.block_size
        );
        ensure!(
            routing.max_m_blocks <= self.max_m_blocks,
            "Kimi Marlin WNA16 workspace max_m_blocks {} below routing {}",
            self.max_m_blocks,
            routing.max_m_blocks
        );
        ensure!(
            routing.max_padded_tokens <= self.max_padded_tokens,
            "Kimi Marlin WNA16 workspace max_padded_tokens {} below routing {}",
            self.max_padded_tokens,
            routing.max_padded_tokens
        );
        ensure!(
            size_n <= self.max_size_n && size_n.is_multiple_of(64),
            "Kimi Marlin WNA16 size_n {} exceeds workspace max {} or is not divisible by 64",
            size_n,
            self.max_size_n
        );
        let required = (size_n / 64)
            .checked_mul(routing.max_m_blocks)
            .ok_or_else(|| anyhow::anyhow!("Kimi Marlin WNA16 required workspace overflow"))?
            .max(1);
        ensure!(
            self.locks.len() >= required,
            "Kimi Marlin WNA16 workspace lock len must cover {}, got {}",
            required,
            self.locks.len()
        );
        let mut required_c_tmp = size_n
            .checked_mul(routing.max_padded_tokens)
            .ok_or_else(|| anyhow::anyhow!("Kimi Marlin WNA16 c_tmp required overflow"))?;
        if self.block_size == 8 {
            required_c_tmp = required_c_tmp
                .checked_mul(2)
                .ok_or_else(|| anyhow::anyhow!("Kimi Marlin WNA16 c_tmp required overflow"))?;
        }
        ensure!(
            self.c_tmp.len() >= required_c_tmp.max(1),
            "Kimi Marlin WNA16 c_tmp len must cover {}, got {}",
            required_c_tmp,
            self.c_tmp.len()
        );
        Ok(())
    }
}

pub struct KimiMarlinRouting<'a> {
    active_tokens: usize,
    pub route_elems: usize,
    block_size: usize,
    max_padded_tokens: usize,
    pub max_m_blocks: usize,
    sorted_token_ids: &'a CudaSlice<i32>,
    expert_ids: &'a CudaSlice<i32>,
    pub num_tokens_post_padded: &'a CudaSlice<i32>,
}

pub fn kimi_moe_marlin_align_block_size<'a>(
    ctx: &DeviceContext,
    workspace: &'a mut KimiMarlinRouteWorkspace,
    topk_idx: &CudaSlice<i32>,
    batch_size: usize,
    active_tokens: usize,
    global_expert_start: usize,
) -> Result<KimiMarlinRouting<'a>> {
    workspace.validate_for(active_tokens)?;
    validate_global_expert_start(global_expert_start)?;
    ensure!(batch_size > 0, "batch_size must be > 0");
    ensure!(
        active_tokens >= batch_size,
        "active_tokens {} must cover batch_size {} for bs>1 Marlin routing",
        active_tokens,
        batch_size
    );
    let route_elems = active_tokens
        .checked_mul(KIMI_K2_TOPK)
        .ok_or_else(|| anyhow::anyhow!("active_tokens * topk overflow"))?;
    ensure!(
        i32::try_from(route_elems).is_ok(),
        "route_elems {route_elems} exceeds i32::MAX"
    );
    ensure!(
        topk_idx.len() >= route_elems,
        "topk_idx len must cover active_tokens * topk: have {}, need {}",
        topk_idx.len(),
        route_elems
    );
    ensure!(
        i32::try_from(workspace.max_padded_tokens).is_ok(),
        "max_padded_tokens {} exceeds i32::MAX",
        workspace.max_padded_tokens
    );
    ensure!(
        i32::try_from(workspace.max_m_blocks).is_ok(),
        "max_m_blocks {} exceeds i32::MAX",
        workspace.max_m_blocks
    );

    {
        let buffers = MarlinAlignBuffers {
            topk_idx,
            sorted_token_ids: &mut workspace.sorted_token_ids,
            expert_ids: &mut workspace.expert_ids,
            num_tokens_post_padded: &mut workspace.num_tokens_post_padded,
            expert_offsets: &mut workspace.expert_offsets,
        };
        let result = launch_marlin_align(
            ctx,
            buffers,
            |topk_ptr, sorted_ptr, expert_ids_ptr, num_tokens_ptr, offsets_ptr| {
                Ok(unsafe {
                    ffi::kimi_moe_marlin_align_block_size_cuda(
                        topk_ptr,
                        sorted_ptr,
                        expert_ids_ptr,
                        num_tokens_ptr,
                        offsets_ptr,
                        std::ptr::null_mut(),
                        active_tokens as i32,
                        KIMI_K2_TOPK as i32,
                        global_expert_start as i32,
                        KIMI_K2_LOCAL_EXPERTS as i32,
                        workspace.block_size as i32,
                        workspace.max_padded_tokens as i32,
                        workspace.max_m_blocks as i32,
                        ctx.stream.cu_stream(),
                    )
                })
            },
        )?;
        result.result()?;
    }
    Ok(KimiMarlinRouting {
        active_tokens,
        route_elems,
        block_size: workspace.block_size,
        max_padded_tokens: workspace.max_padded_tokens,
        max_m_blocks: workspace.max_m_blocks,
        sorted_token_ids: &workspace.sorted_token_ids,
        expert_ids: &workspace.expert_ids,
        num_tokens_post_padded: &workspace.num_tokens_post_padded,
    })
}

pub fn kimi_marlin_int4_reorder_scale(
    ctx: &DeviceContext,
    weight_scale_checkpoint: &CudaSlice<bf16>,
    weight_scale_marlin: &mut CudaSlice<bf16>,
    manifest: &KimiInt4WeightManifest,
) -> Result<()> {
    manifest.validate()?;
    let expected_elements = manifest.scale_shape.elements();
    ensure!(
        weight_scale_checkpoint.len() == expected_elements,
        "{} checkpoint scale len must be {}, got {}",
        manifest.role.label(),
        expected_elements,
        weight_scale_checkpoint.len()
    );
    ensure!(
        weight_scale_marlin.len() == expected_elements,
        "{} Marlin scale len must be {}, got {}",
        manifest.role.label(),
        expected_elements,
        weight_scale_marlin.len()
    );
    ensure!(
        (expected_elements / KIMI_K2_LOCAL_EXPERTS).is_multiple_of(64),
        "{} Marlin scale elements per expert must be divisible by 64, got {}",
        manifest.role.label(),
        expected_elements / KIMI_K2_LOCAL_EXPERTS
    );
    let (src_ptr, _src_guard) = weight_scale_checkpoint.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = weight_scale_marlin.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_marlin_int4_reorder_scale_cuda(
            src_ptr as *const ffi::Half,
            dst_ptr as *mut ffi::Half,
            manifest.logical_shape.in_dim as i32,
            manifest.logical_shape.out_dim as i32,
            KIMI_K2_LOCAL_EXPERTS as i32,
            KIMI_K2_INT4_GROUP_SIZE as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_marlin_int4_reorder_weight(
    ctx: &DeviceContext,
    weight_packed_checkpoint_offset_binary: &CudaSlice<u8>,
    weight_packed_marlin: &mut CudaSlice<u8>,
    manifest: &KimiInt4WeightManifest,
) -> Result<()> {
    manifest.validate()?;
    let expected_bytes = manifest.packed_shape.elements();
    ensure!(
        weight_packed_checkpoint_offset_binary.len() == expected_bytes,
        "{} checkpoint packed len must be {}, got {}",
        manifest.role.label(),
        expected_bytes,
        weight_packed_checkpoint_offset_binary.len()
    );
    ensure!(
        weight_packed_marlin.len() == expected_bytes,
        "{} Marlin packed len must be {}, got {}",
        manifest.role.label(),
        expected_bytes,
        weight_packed_marlin.len()
    );
    ensure!(
        manifest.nibble_order == KimiInt4NibbleOrder::LowThenHigh,
        "{} Marlin repack expects low-then-high offset-binary INT4, got {}",
        manifest.role.label(),
        manifest.nibble_order.label()
    );
    ensure!(
        manifest.marlin_packed_u32_elements() * std::mem::size_of::<u32>() == expected_bytes,
        "{} Marlin packed u32 view must preserve checkpoint byte size",
        manifest.role.label()
    );
    ensure!(
        manifest.logical_shape.in_dim.is_multiple_of(16)
            && manifest.logical_shape.out_dim.is_multiple_of(64),
        "{} Marlin repack requires in_dim multiple of 16 and out_dim multiple of 64, got in={} out={}",
        manifest.role.label(),
        manifest.logical_shape.in_dim,
        manifest.logical_shape.out_dim
    );
    let (src_ptr, _src_guard) = weight_packed_checkpoint_offset_binary.device_ptr(&ctx.stream);
    let (dst_ptr, _dst_guard) = weight_packed_marlin.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_marlin_int4_reorder_weight_cuda(
            src_ptr as *const u8,
            dst_ptr as *mut u8,
            manifest.logical_shape.in_dim as i32,
            manifest.logical_shape.out_dim as i32,
            KIMI_K2_LOCAL_EXPERTS as i32,
            KIMI_K2_INT4_GROUP_SIZE as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_marlin_int4_fuse_w13(
    ctx: &DeviceContext,
    gate: &KimiMarlinInt4Weight<'_>,
    up: &KimiMarlinInt4Weight<'_>,
    weight_packed_w13: &mut CudaSlice<u8>,
    weight_scale_w13: &mut CudaSlice<bf16>,
) -> Result<()> {
    gate.validate()?;
    up.validate()?;
    ensure!(
        gate.manifest.role == KimiInt4ExpertRole::W1Gate,
        "Marlin W13 fuse gate role must be W1Gate, got {:?}",
        gate.manifest.role
    );
    ensure!(
        up.manifest.role == KimiInt4ExpertRole::W3Up,
        "Marlin W13 fuse up role must be W3Up, got {:?}",
        up.manifest.role
    );
    ensure!(
        gate.manifest.local_expert_offset == up.manifest.local_expert_offset,
        "Marlin W13 fuse requires matching expert ranges, got {} and {}",
        gate.manifest.local_expert_offset,
        up.manifest.local_expert_offset
    );
    ensure!(
        gate.manifest.logical_shape == up.manifest.logical_shape,
        "Marlin W13 fuse requires matching shapes, got {:?} and {:?}",
        gate.manifest.logical_shape,
        up.manifest.logical_shape
    );
    let expected_weight_len = gate.weight_packed_uint4b8.len() + up.weight_packed_uint4b8.len();
    ensure!(
        weight_packed_w13.len() == expected_weight_len,
        "Marlin fused W13 packed len must be {}, got {}",
        expected_weight_len,
        weight_packed_w13.len()
    );
    let expected_scale_len = gate.weight_scale_permuted.len() + up.weight_scale_permuted.len();
    ensure!(
        weight_scale_w13.len() == expected_scale_len,
        "Marlin fused W13 scale len must be {}, got {}",
        expected_scale_len,
        weight_scale_w13.len()
    );

    let (gate_weight_ptr, _gate_weight_guard) = gate.weight_packed_uint4b8.device_ptr(&ctx.stream);
    let (up_weight_ptr, _up_weight_guard) = up.weight_packed_uint4b8.device_ptr(&ctx.stream);
    let (w13_weight_ptr, _w13_weight_guard) = weight_packed_w13.device_ptr_mut(&ctx.stream);
    let (gate_scale_ptr, _gate_scale_guard) = gate.weight_scale_permuted.device_ptr(&ctx.stream);
    let (up_scale_ptr, _up_scale_guard) = up.weight_scale_permuted.device_ptr(&ctx.stream);
    let (w13_scale_ptr, _w13_scale_guard) = weight_scale_w13.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_marlin_int4_fuse_w13_cuda(
            gate_weight_ptr as *const u8,
            up_weight_ptr as *const u8,
            w13_weight_ptr as *mut u8,
            gate_scale_ptr as *const ffi::Half,
            up_scale_ptr as *const ffi::Half,
            w13_scale_ptr as *mut ffi::Half,
            gate.manifest.logical_shape.in_dim as i32,
            gate.manifest.logical_shape.out_dim as i32,
            KIMI_K2_LOCAL_EXPERTS as i32,
            KIMI_K2_INT4_GROUP_SIZE as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_marlin_wna16_w13_gemm<const IN: usize, const OUT: usize>(
    ctx: &DeviceContext,
    workspace: &mut KimiMarlinWna16Workspace,
    routing: &KimiMarlinRouting<'_>,
    input: &GpuTensor<IN>,
    weight: &KimiMarlinFusedW13Int4Weight<'_>,
    topk_weight: &CudaSlice<f32>,
    output_w13: &mut GpuTensor<OUT>,
) -> Result<()> {
    weight.validate()?;
    workspace.validate_for(routing, OUT)?;
    ensure!(
        IN == KIMI_K2_HIDDEN,
        "marlin_w13 input dim must be {}, got {}",
        KIMI_K2_HIDDEN,
        IN
    );
    ensure!(
        OUT == 2 * KIMI_K2_EXPERT_INTERMEDIATE,
        "marlin_w13 output dim must be {}, got {}",
        2 * KIMI_K2_EXPERT_INTERMEDIATE,
        OUT
    );
    ensure!(
        input.seq_len == routing.active_tokens,
        "marlin_w13 input seq_len must be {}, got {}",
        routing.active_tokens,
        input.seq_len
    );
    ensure!(
        output_w13.seq_len == routing.route_elems,
        "marlin_w13 output seq_len must be {}, got {}",
        routing.route_elems,
        output_w13.seq_len
    );
    ensure!(
        topk_weight.len() >= routing.route_elems,
        "topk_weight len must cover {}, got {}",
        routing.route_elems,
        topk_weight.len()
    );
    launch_marlin_wna16_gemm(
        ctx,
        workspace,
        routing,
        &input.data,
        weight.weight_packed_uint4b8,
        weight.weight_scale_permuted,
        topk_weight,
        &mut output_w13.data,
        KIMI_K2_TOPK,
        false,
        routing.active_tokens,
        OUT,
        IN,
    )
}

pub fn kimi_marlin_wna16_w2_gemm<const IN: usize, const OUT: usize>(
    ctx: &DeviceContext,
    workspace: &mut KimiMarlinWna16Workspace,
    routing: &KimiMarlinRouting<'_>,
    input: &GpuTensor<IN>,
    weight: &KimiMarlinInt4Weight<'_>,
    topk_weight: &CudaSlice<f32>,
    output: &mut GpuTensor<OUT>,
) -> Result<()> {
    weight.validate()?;
    ensure!(
        weight.manifest.role == KimiInt4ExpertRole::W2Down,
        "Marlin W2 role mismatch: got {:?}",
        weight.manifest.role
    );
    workspace.validate_for(routing, OUT)?;
    ensure!(
        IN == KIMI_K2_EXPERT_INTERMEDIATE,
        "marlin_w2 input dim must be {}, got {}",
        KIMI_K2_EXPERT_INTERMEDIATE,
        IN
    );
    ensure!(
        OUT == KIMI_K2_HIDDEN,
        "marlin_w2 output dim must be {}, got {}",
        KIMI_K2_HIDDEN,
        OUT
    );
    ensure!(
        topk_weight.len() >= routing.route_elems,
        "topk_weight len must cover {}, got {}",
        routing.route_elems,
        topk_weight.len()
    );
    launch_marlin_wna16_gemm(
        ctx,
        workspace,
        routing,
        &input.data,
        weight.weight_packed_uint4b8,
        weight.weight_scale_permuted,
        topk_weight,
        &mut output.data,
        1,
        true,
        routing.route_elems,
        OUT,
        IN,
    )
}

pub fn kimi_marlin_w13_swiglu<const INTER2: usize, const INTER: usize>(
    ctx: &DeviceContext,
    w13: &GpuTensor<INTER2>,
    output: &mut GpuTensor<INTER>,
) -> Result<()> {
    ensure!(
        INTER2 == 2 * INTER,
        "Kimi Marlin SwiGLU dim mismatch: input={}, output={}",
        INTER2,
        INTER
    );
    ensure!(
        w13.seq_len == output.seq_len,
        "Kimi Marlin SwiGLU seq_len mismatch: input={}, output={}",
        w13.seq_len,
        output.seq_len
    );
    let (w13_ptr, _w13_guard) = w13.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = output.data.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_marlin_w13_swiglu_cuda(
            w13_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            w13.seq_len as i32,
            INTER as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

/// SwiGLU over the DeepEP expanded layout: fixed occupancy-sized grid that
/// grid-strides up to `num_tokens_post_padded[0]` read on-device — cost tracks
/// the actual expanded rows, not the worst-case capacity, and no D2H is needed.
pub fn kimi_marlin_w13_swiglu_expanded<const INTER2: usize, const INTER: usize>(
    ctx: &DeviceContext,
    w13: &GpuTensor<INTER2>,
    num_tokens_post_padded: &CudaSlice<i32>,
    output: &mut GpuTensor<INTER>,
) -> Result<()> {
    ensure!(
        INTER2 == 2 * INTER,
        "Kimi expanded Marlin SwiGLU dim mismatch: input={}, output={}",
        INTER2,
        INTER
    );
    ensure!(
        w13.seq_len == output.seq_len,
        "Kimi expanded Marlin SwiGLU seq_len mismatch: input={}, output={}",
        w13.seq_len,
        output.seq_len
    );
    ensure!(
        !num_tokens_post_padded.is_empty(),
        "num_tokens_post_padded must have at least 1 element"
    );
    let (w13_ptr, _w13_guard) = w13.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = output.data.device_ptr_mut(&ctx.stream);
    let (ntp_ptr, _ntp_guard) = num_tokens_post_padded.device_ptr(&ctx.stream);
    let result = unsafe {
        ffi::kimi_marlin_w13_swiglu_expanded_cuda(
            w13_ptr as *const ffi::Half,
            out_ptr as *mut ffi::Half,
            ntp_ptr as *const i32,
            w13.seq_len as i32,
            INTER as i32,
            0,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

pub fn kimi_marlin_sum_topk_rows_f32<const DIM: usize>(
    ctx: &DeviceContext,
    route_output: &GpuTensor<DIM>,
    active_tokens: usize,
    out: &mut CudaSlice<f32>,
) -> Result<()> {
    ensure!(
        route_output.seq_len == active_tokens * KIMI_K2_TOPK,
        "marlin_sum_topk route_output seq_len must be {}, got {}",
        active_tokens * KIMI_K2_TOPK,
        route_output.seq_len
    );
    ensure!(
        out.len() >= active_tokens * DIM,
        "marlin_sum_topk output too small: have {}, need {}",
        out.len(),
        active_tokens * DIM
    );
    let (route_ptr, _route_guard) = route_output.data.device_ptr(&ctx.stream);
    let (out_ptr, _out_guard) = out.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::kimi_marlin_sum_topk_rows_f32_cuda(
            route_ptr as *const ffi::Half,
            out_ptr as *mut f32,
            active_tokens as i32,
            KIMI_K2_TOPK as i32,
            DIM as i32,
            ctx.stream.cu_stream(),
        )
    };
    result.result()?;
    Ok(())
}

/// Build Marlin routing metadata on-stream from a DeepEP post-epilogue
/// expert prefix sum (`psum_expert`, `num_local_experts + 1` entries).
///
/// The DeepEP expanded recv buffer is already expert-major with each expert
/// segment aligned to `expert_alignment`, so `sorted_token_ids` is identity
/// over real rows and sentinel over pad rows. A `<<<1,64>>>` kernel fills the
/// metadata and writes the actual padded total to `num_tokens_post_padded[0]`
/// — zero D2H; Marlin and `swiglu_w13_expanded` read it on-device.
pub fn kimi_deepep_build_marlin_routing_on_stream<'a>(
    ctx: &DeviceContext,
    workspace: &'a mut KimiMarlinRouteWorkspace,
    psum_expert: &CudaSlice<i32>,
    expert_alignment: usize,
    expanded_capacity: usize,
) -> Result<KimiMarlinRouting<'a>> {
    ensure!(
        expert_alignment > 0,
        "deepep expert_alignment must be positive"
    );
    ensure!(
        expert_alignment.is_multiple_of(workspace.block_size),
        "deepep expert_alignment {} must be a multiple of Marlin block_size {}",
        expert_alignment,
        workspace.block_size
    );
    ensure!(
        psum_expert.len() > KIMI_K2_LOCAL_EXPERTS,
        "psum_expert len must be >= {}, got {}",
        KIMI_K2_LOCAL_EXPERTS + 1,
        psum_expert.len()
    );
    ensure!(
        expanded_capacity <= workspace.max_padded_tokens,
        "deepep expanded_capacity {} exceeds workspace max_padded_tokens {}",
        expanded_capacity,
        workspace.max_padded_tokens
    );

    let block_size = workspace.block_size;

    // Use the full expanded capacity: any expert can receive tokens from
    // every EP rank. The GPU kernel writes the actual padded total to
    // num_tokens_post_padded[0]; Marlin skips sentinel-filled blocks.
    let tight_max = expanded_capacity;
    let tight_m_blocks = tight_max.div_ceil(block_size);

    {
        let (psum_ptr, _g0) = psum_expert.device_ptr(&ctx.stream);
        let (sorted_ptr, _g1) = workspace.sorted_token_ids.device_ptr_mut(&ctx.stream);
        let (expert_ptr, _g2) = workspace.expert_ids.device_ptr_mut(&ctx.stream);
        let (ntp_ptr, _g3) = workspace.num_tokens_post_padded.device_ptr_mut(&ctx.stream);

        let result = unsafe {
            ffi::kimi_deepep_build_marlin_routing_on_stream(
                psum_ptr as *const i32,
                sorted_ptr as *mut i32,
                expert_ptr as *mut i32,
                ntp_ptr as *mut i32,
                KIMI_K2_LOCAL_EXPERTS as i32,
                expert_alignment as i32,
                block_size as i32,
                tight_max as i32,
                tight_m_blocks as i32,
                ctx.stream.cu_stream(),
            )
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("kimi_deepep_build_marlin_routing_on_stream failed: {result:?}");
        }
    }

    Ok(KimiMarlinRouting {
        active_tokens: tight_max,
        route_elems: tight_max,
        block_size,
        max_padded_tokens: tight_max,
        max_m_blocks: tight_m_blocks,
        sorted_token_ids: &workspace.sorted_token_ids,
        expert_ids: &workspace.expert_ids,
        num_tokens_post_padded: &workspace.num_tokens_post_padded,
    })
}

/// W13 (gate+up) GEMM over the DeepEP expanded layout: top_k=1, no weight scaling.
pub fn kimi_marlin_wna16_expanded_w13_gemm<const IN: usize, const OUT: usize>(
    ctx: &DeviceContext,
    workspace: &mut KimiMarlinWna16Workspace,
    routing: &KimiMarlinRouting<'_>,
    input: &GpuTensor<IN>,
    weight: &KimiMarlinFusedW13Int4Weight<'_>,
    topk_weight: &CudaSlice<f32>,
    output_w13: &mut GpuTensor<OUT>,
) -> Result<()> {
    weight.validate()?;
    workspace.validate_for(routing, OUT)?;
    ensure!(
        topk_weight.len() >= routing.route_elems,
        "topk_weight len must cover {}, got {}",
        routing.route_elems,
        topk_weight.len()
    );
    launch_marlin_wna16_gemm(
        ctx,
        workspace,
        routing,
        &input.data,
        weight.weight_packed_uint4b8,
        weight.weight_scale_permuted,
        topk_weight,
        &mut output_w13.data,
        1,
        false,
        routing.active_tokens,
        OUT,
        IN,
    )
}

/// W2 (down) GEMM over the DeepEP expanded layout: top_k=1 with one top-k weight per
/// expert-major row. This matches the NCCL path's BF16 rounding boundary.
pub fn kimi_marlin_wna16_expanded_w2_gemm<const IN: usize, const OUT: usize>(
    ctx: &DeviceContext,
    workspace: &mut KimiMarlinWna16Workspace,
    routing: &KimiMarlinRouting<'_>,
    input: &GpuTensor<IN>,
    weight: &KimiMarlinInt4Weight<'_>,
    topk_weight: &CudaSlice<f32>,
    output: &mut GpuTensor<OUT>,
) -> Result<()> {
    weight.validate()?;
    ensure!(
        weight.manifest.role == KimiInt4ExpertRole::W2Down,
        "Marlin W2 role mismatch: got {:?}",
        weight.manifest.role
    );
    workspace.validate_for(routing, OUT)?;
    ensure!(
        topk_weight.len() >= routing.route_elems,
        "topk_weight len must cover {}, got {}",
        routing.route_elems,
        topk_weight.len()
    );
    launch_marlin_wna16_gemm(
        ctx,
        workspace,
        routing,
        &input.data,
        weight.weight_packed_uint4b8,
        weight.weight_scale_permuted,
        topk_weight,
        &mut output.data,
        1,
        true,
        routing.route_elems,
        OUT,
        IN,
    )
}

fn launch_marlin_wna16_gemm(
    ctx: &DeviceContext,
    workspace: &mut KimiMarlinWna16Workspace,
    routing: &KimiMarlinRouting<'_>,
    input: &CudaSlice<bf16>,
    weight_packed_uint4b8: &CudaSlice<u8>,
    weight_scale_permuted: &CudaSlice<bf16>,
    topk_weight: &CudaSlice<f32>,
    output: &mut CudaSlice<bf16>,
    top_k: usize,
    mul_topk_weights: bool,
    size_m: usize,
    size_n: usize,
    size_k: usize,
) -> Result<()> {
    ensure!(
        i32::try_from(size_m).is_ok()
            && i32::try_from(size_n).is_ok()
            && i32::try_from(size_k).is_ok(),
        "Kimi Marlin WNA16 MNK exceeds i32"
    );
    ensure!(
        i32::try_from(routing.max_padded_tokens).is_ok()
            && i32::try_from(workspace.locks.len()).is_ok(),
        "Kimi Marlin WNA16 metadata exceeds i32"
    );
    ensure!(
        !weight_packed_uint4b8.is_empty() && !weight_scale_permuted.is_empty(),
        "Kimi Marlin WNA16 weight package must be non-empty"
    );
    let lock_len = workspace.locks.len();
    let buffers = MarlinGemmBuffers {
        input,
        output,
        c_tmp: &mut workspace.c_tmp,
        qweight: weight_packed_uint4b8,
        scales: weight_scale_permuted,
        workspace: &mut workspace.locks,
        sorted_token_ids: routing.sorted_token_ids,
        expert_ids: routing.expert_ids,
        num_tokens_post_padded: routing.num_tokens_post_padded,
        topk_weights: topk_weight,
    };
    let result = launch_marlin_gemm(ctx, buffers, |ptrs| {
        Ok(unsafe {
            ffi::kimi_marlin_wna16_gemm_cuda(
                ptrs.input,
                ptrs.output,
                ptrs.c_tmp,
                ptrs.qweight,
                ptrs.scales.cast::<ffi::Half>(),
                ptrs.workspace,
                ptrs.sorted_token_ids,
                ptrs.expert_ids,
                ptrs.num_tokens_post_padded,
                ptrs.topk_weights,
                lock_len as i32,
                routing.max_padded_tokens as i32,
                routing.block_size as i32,
                top_k as i32,
                mul_topk_weights,
                size_m as i32,
                size_n as i32,
                size_k as i32,
                KIMI_K2_LOCAL_EXPERTS as i32,
                KIMI_K2_INT4_GROUP_SIZE as i32,
                0,
                ctx.stream.cu_stream(),
            )
        })
    })?;
    result.result()?;
    Ok(())
}

#[must_use]
const fn packed_int4_cols(cols: usize) -> usize {
    cols.div_ceil(2)
}

fn validate_marlin_block_size(block_size: usize) -> Result<()> {
    ensure!(
        block_size == 8 || ((16..=64).contains(&block_size) && block_size.is_multiple_of(16)),
        "Kimi Marlin block_size must be 8 or a multiple of 16 in [16, 64], got {}",
        block_size
    );
    Ok(())
}

fn validate_global_expert_start(global_expert_start: usize) -> Result<()> {
    ensure!(
        global_expert_start + KIMI_K2_LOCAL_EXPERTS <= KIMI_K2_ROUTED_EXPERTS,
        "global expert range [{}..{}) exceeds {} routed experts",
        global_expert_start,
        global_expert_start + KIMI_K2_LOCAL_EXPERTS,
        KIMI_K2_ROUTED_EXPERTS
    );
    Ok(())
}

fn marlin_padded_route_capacity(active_tokens: usize, block_size: usize) -> Result<usize> {
    validate_marlin_block_size(block_size)?;
    let route_elems = active_tokens
        .checked_mul(KIMI_K2_TOPK)
        .ok_or_else(|| anyhow::anyhow!("active_tokens * topk overflow"))?;
    let max_padding = KIMI_K2_LOCAL_EXPERTS
        .checked_mul(block_size - 1)
        .ok_or_else(|| anyhow::anyhow!("local_experts * (block_size - 1) overflow"))?;
    route_elems
        .checked_add(max_padding)
        .ok_or_else(|| anyhow::anyhow!("Marlin padded route capacity overflow"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ep8_w1_manifest_shapes_cover_compressed_tensors_metadata() {
        let manifest = KimiInt4WeightManifest::ep8(
            KimiInt4ExpertRole::W1Gate,
            3,
            KimiInt4NibbleOrder::LowThenHigh,
        );

        manifest.validate().expect("manifest should be valid");
        assert_eq!(manifest.local_experts, 48);
        assert_eq!(manifest.local_expert_offset, 144);
        assert_eq!(manifest.packed_shape.elements(), 48 * 2048 * (7168 / 2));
        assert_eq!(manifest.scale_shape.elements(), 48 * 2048 * (7168 / 32));
    }

    #[test]
    fn ep8_w2_manifest_uses_down_projection_shape() {
        let manifest = KimiInt4WeightManifest::ep8(
            KimiInt4ExpertRole::W2Down,
            0,
            KimiInt4NibbleOrder::LowThenHigh,
        );

        manifest.validate().expect("manifest should be valid");
        assert_eq!(
            manifest.logical_shape,
            KimiInt4LogicalShape {
                out_dim: KIMI_K2_HIDDEN,
                in_dim: KIMI_K2_EXPERT_INTERMEDIATE,
            }
        );
        assert_eq!(manifest.packed_shape.elements(), 48 * 7168 * (2048 / 2));
        assert_eq!(manifest.scale_shape.elements(), 48 * 7168 * (2048 / 32));
    }

    #[test]
    fn int4_scale_specs_distinguish_checkpoint_and_marlin_layouts() {
        let manifest = KimiInt4WeightManifest::ep8(
            KimiInt4ExpertRole::W1Gate,
            0,
            KimiInt4NibbleOrder::LowThenHigh,
        );

        let checkpoint = manifest.weight_scale_checkpoint_spec();
        assert_eq!(checkpoint.layout, "expert_major_group_scale_checkpoint");
        assert_eq!(checkpoint.axes[1].name, "out");
        assert_eq!(checkpoint.axes[2].name, "in_group");

        let marlin = manifest.weight_scale_marlin_permuted_spec();
        assert_eq!(
            marlin.layout,
            "expert_major_group_scale_marlin_group_major_perm64"
        );
        assert_eq!(marlin.axes[1].name, "in_group");
        assert_eq!(marlin.axes[2].name, "out");
    }

    #[test]
    fn int4_packed_specs_distinguish_checkpoint_and_marlin_layouts() {
        let manifest = KimiInt4WeightManifest::ep8(
            KimiInt4ExpertRole::W2Down,
            0,
            KimiInt4NibbleOrder::LowThenHigh,
        );

        let checkpoint = manifest.weight_packed_checkpoint_spec();
        assert_eq!(
            checkpoint.layout,
            "expert_major_int4_packed_checkpoint_offset_binary"
        );
        assert_eq!(checkpoint.axes[1].name, "out");
        assert_eq!(checkpoint.axes[2].name, "packed_in_over_2");

        let marlin = manifest.weight_packed_marlin_uint4b8_spec();
        assert_eq!(
            marlin.layout,
            "expert_major_int4_packed_marlin_uint4b8_noact"
        );
        assert_eq!(marlin.dtype, "u32");
        assert_eq!(marlin.axes[1].name, "in_tile16");
        assert_eq!(marlin.axes[1].size, KIMI_K2_EXPERT_INTERMEDIATE / 16);
        assert_eq!(marlin.axes[2].name, "out_x2");
        assert_eq!(marlin.axes[2].size, KIMI_K2_HIDDEN * 2);
        assert_eq!(
            manifest.marlin_packed_u32_elements() * std::mem::size_of::<u32>(),
            manifest.packed_shape.elements()
        );
    }

    #[test]
    fn marlin_route_capacity_matches_vllm_ignore_invalid_bound() {
        let active_tokens = 7;
        let block_size = 8;
        let route_elems = active_tokens * KIMI_K2_TOPK;
        let capacity = marlin_padded_route_capacity(active_tokens, block_size).expect("capacity");
        assert_eq!(
            capacity,
            route_elems + KIMI_K2_LOCAL_EXPERTS * (block_size - 1)
        );
        assert_eq!(capacity.div_ceil(block_size), 49);
    }

    #[test]
    #[ignore = "requires a CUDA GPU"]
    fn kimi_marlin_align_boundary_matches_vllm_contract() {
        const BATCH_SIZE: usize = 16;
        const BLOCK_SIZE: usize = 8;
        const GLOBAL_START: usize = 96;
        const SMALL_PATH_ROUTES: usize = 1016;
        const LARGE_PATH_ROUTES: usize = 1024;
        const LOCAL_GROUP_A: usize = 13;
        const LOCAL_GROUP_B: usize = 17;
        const LOCAL_GROUP_C: usize = KIMI_K2_LOCAL_EXPERTS - LOCAL_GROUP_A - LOCAL_GROUP_B;
        const BELOW_GROUP: usize = 32;
        const ABOVE_GROUP: usize = 64;
        const STRIDE_B: usize = 3;
        const STRIDE_C: usize = 5;
        const STRIDE_ABOVE: usize = 7;

        let ctx = crate::tensor::DeviceContext::new().expect("CUDA context");
        let mut workspace =
            KimiMarlinRouteWorkspace::new(&ctx, LARGE_PATH_ROUTES / KIMI_K2_TOPK, BLOCK_SIZE)
                .expect("workspace");
        let first_non_local = GLOBAL_START + KIMI_K2_LOCAL_EXPERTS;
        for route_elems in [SMALL_PATH_ROUTES, LARGE_PATH_ROUTES] {
            let active_tokens = route_elems / KIMI_K2_TOPK;
            let topk_host = (0..active_tokens)
                .flat_map(|token| {
                    [
                        GLOBAL_START + token % LOCAL_GROUP_A,
                        token % BELOW_GROUP,
                        GLOBAL_START + LOCAL_GROUP_A + (token * STRIDE_B) % LOCAL_GROUP_B,
                        first_non_local + token % ABOVE_GROUP,
                        GLOBAL_START
                            + LOCAL_GROUP_A
                            + LOCAL_GROUP_B
                            + (token * STRIDE_C) % LOCAL_GROUP_C,
                        BELOW_GROUP + (token * STRIDE_B) % BELOW_GROUP,
                        first_non_local + ABOVE_GROUP + (token * STRIDE_ABOVE) % ABOVE_GROUP,
                        2 * BELOW_GROUP + (token * STRIDE_C) % BELOW_GROUP,
                    ]
                    .map(|expert| i32::try_from(expert).expect("expert id"))
                })
                .collect::<Vec<_>>();
            assert_eq!(topk_host.len(), route_elems);

            let topk_dev = ctx.stream.clone_htod(&topk_host).expect("topk H2D");
            let (num_tokens, sorted, expert_ids) = {
                let routing = kimi_moe_marlin_align_block_size(
                    &ctx,
                    &mut workspace,
                    &topk_dev,
                    BATCH_SIZE,
                    active_tokens,
                    GLOBAL_START,
                )
                .expect("align");
                let num_tokens = ctx
                    .stream
                    .clone_dtoh(routing.num_tokens_post_padded)
                    .expect("num_tokens D2H");
                let sorted = ctx
                    .stream
                    .clone_dtoh(routing.sorted_token_ids)
                    .expect("sorted D2H");
                let expert_ids = ctx
                    .stream
                    .clone_dtoh(routing.expert_ids)
                    .expect("expert_ids D2H");
                (num_tokens, sorted, expert_ids)
            };
            let expert_offsets = ctx
                .stream
                .clone_dtoh(&workspace.expert_offsets)
                .expect("expert_offsets D2H");
            let total = usize::try_from(num_tokens[0]).expect("nonnegative padded tokens");
            assert!(total.is_multiple_of(BLOCK_SIZE));

            let mut expected_sorted = Vec::<i32>::new();
            let mut expected_expert_ids = Vec::<i32>::new();
            let mut expected_offsets = Vec::<u32>::with_capacity(KIMI_K2_LOCAL_EXPERTS + 1);
            let sentinel = i32::try_from(route_elems).expect("route sentinel");
            for local_expert in 0..KIMI_K2_LOCAL_EXPERTS {
                expected_offsets.push(u32::try_from(expected_sorted.len()).expect("expert offset"));
                let global_expert = GLOBAL_START + local_expert;
                let mut routes = topk_host
                    .iter()
                    .enumerate()
                    .filter(|&(_, &expert)| usize::try_from(expert).ok() == Some(global_expert))
                    .map(|(route_offset, _)| i32::try_from(route_offset).expect("route offset"))
                    .collect::<Vec<_>>();
                if routes.is_empty() {
                    continue;
                }
                let padded = routes.len().div_ceil(BLOCK_SIZE) * BLOCK_SIZE;
                expected_expert_ids.extend(std::iter::repeat_n(
                    i32::try_from(local_expert).expect("local expert"),
                    padded / BLOCK_SIZE,
                ));
                routes.extend(std::iter::repeat_n(sentinel, padded - routes.len()));
                expected_sorted.extend(routes);
            }
            expected_offsets.push(u32::try_from(expected_sorted.len()).expect("padded total"));

            assert_eq!(total, expected_sorted.len());
            assert_eq!(&sorted[..total], expected_sorted.as_slice());
            assert_eq!(
                &expert_ids[..expected_expert_ids.len()],
                expected_expert_ids.as_slice()
            );
            assert_eq!(expert_offsets, expected_offsets);
        }
    }

    #[test]
    #[ignore = "H20-only: verifies vLLM Marlin scale layout packer on device"]
    fn h20_kimi_marlin_scale_reorder_matches_vllm_permute() {
        use half::bf16;

        let ctx = crate::tensor::DeviceContext::new().expect("CUDA context");
        let local_experts = KIMI_K2_LOCAL_EXPERTS;
        let group_size = KIMI_K2_INT4_GROUP_SIZE;
        let in_dim = 64usize;
        let out_dim = 64usize;
        let scale_k = in_dim / group_size;
        let elements_per_expert = out_dim * scale_k;
        assert_eq!(elements_per_expert % 64, 0);

        let scale_value = |expert: usize, row: usize, group: usize| -> bf16 {
            bf16::from_f32(expert as f32 * 0.25 + row as f32 * 0.01 + group as f32 * 0.125)
        };
        let mut checkpoint = vec![bf16::ZERO; local_experts * elements_per_expert];
        for expert in 0..local_experts {
            for row in 0..out_dim {
                for group in 0..scale_k {
                    checkpoint[expert * elements_per_expert + row * scale_k + group] =
                        scale_value(expert, row, group);
                }
            }
        }

        let checkpoint_dev = ctx.stream.clone_htod(&checkpoint).expect("scale H2D");
        let mut marlin_dev = ctx
            .stream
            .alloc_zeros::<bf16>(checkpoint.len())
            .expect("marlin scale alloc");
        {
            let (src_ptr, _src_guard) = checkpoint_dev.device_ptr(&ctx.stream);
            let (dst_ptr, _dst_guard) = marlin_dev.device_ptr_mut(&ctx.stream);
            let result = unsafe {
                crate::ffi::kimi_marlin_int4_reorder_scale_cuda(
                    src_ptr as *const crate::ffi::Half,
                    dst_ptr as *mut crate::ffi::Half,
                    in_dim as i32,
                    out_dim as i32,
                    local_experts as i32,
                    group_size as i32,
                    ctx.stream.cu_stream(),
                )
            };
            result.result().expect("marlin scale reorder");
        }
        let got = ctx.stream.clone_dtoh(&marlin_dev).expect("scale D2H");
        ctx.sync().expect("sync");

        let marlin_scale_perm = |offset: usize| -> usize { offset / 8 + 8 * (offset % 8) };
        for expert in [0usize, 7, local_experts - 1] {
            for flat in 0..elements_per_expert {
                let source_flat = (flat / 64) * 64 + marlin_scale_perm(flat % 64);
                let group = source_flat / out_dim;
                let row = source_flat - group * out_dim;
                let idx = expert * elements_per_expert + flat;
                let expected = checkpoint[expert * elements_per_expert + row * scale_k + group];
                assert_eq!(
                    got[idx].to_bits(),
                    expected.to_bits(),
                    "expert={expert} flat={flat} row={row} group={group}"
                );
            }
        }
    }

    #[test]
    #[ignore = "H20-only: verifies vLLM no-actorder Marlin weight repack layout on device"]
    fn h20_kimi_marlin_weight_repack_matches_vllm_noact_layout() {
        let ctx = crate::tensor::DeviceContext::new().expect("CUDA context");
        let local_experts = KIMI_K2_LOCAL_EXPERTS;
        let group_size = KIMI_K2_INT4_GROUP_SIZE;
        let in_dim = 64usize;
        let out_dim = 64usize;
        let pack_factor = 8usize;
        let tile_k = 16usize;
        let tile_n = 64usize;
        let k_packed_cols = in_dim / pack_factor;
        let k_tiles = in_dim / tile_k;
        let n_tiles = out_dim / tile_n;
        let words_per_expert = out_dim * k_packed_cols;
        let marlin_words_per_expert = k_tiles * out_dim * 2;
        assert_eq!(words_per_expert, marlin_words_per_expert);

        let nibble = |expert: usize, row: usize, col: usize| -> u32 {
            ((expert * 3 + row * 5 + col * 7) & 0x0f) as u32
        };
        let mut checkpoint = vec![0u32; local_experts * words_per_expert];
        for expert in 0..local_experts {
            for row in 0..out_dim {
                for k_word in 0..k_packed_cols {
                    let mut word = 0u32;
                    for pos in 0..pack_factor {
                        word |= nibble(expert, row, k_word * pack_factor + pos) << (pos * 4);
                    }
                    checkpoint[expert * words_per_expert + row * k_packed_cols + k_word] = word;
                }
            }
        }

        let mut expected = vec![0u32; checkpoint.len()];
        let tc_offsets = [0usize, 1, 8, 9];
        let pack_idx = [0usize, 2, 4, 6, 1, 3, 5, 7];
        for expert in 0..local_experts {
            let checkpoint_base = expert * words_per_expert;
            let marlin_base = expert * marlin_words_per_expert;
            for k_tile in 0..k_tiles {
                for n_tile in 0..n_tiles {
                    let mut sh_stage = vec![0u32; tile_n * (tile_k / pack_factor)];
                    for k_id in 0..(tile_k / pack_factor) {
                        for n in 0..tile_n {
                            sh_stage[k_id * tile_n + n] = checkpoint[checkpoint_base
                                + (n_tile * tile_n + n) * k_packed_cols
                                + k_tile * (tile_k / pack_factor)
                                + k_id];
                        }
                    }
                    for warp_id in 0..4usize {
                        for th_id in 0..32usize {
                            let tc_col = th_id / 4;
                            let tc_row = (th_id % 4) * 2;
                            let cur_n = warp_id * 16 + tc_col;
                            let b1_vals = [sh_stage[cur_n], sh_stage[cur_n + tile_n]];
                            let b2_vals = [sh_stage[cur_n + 8], sh_stage[cur_n + 8 + tile_n]];

                            let mut vals = [0u32; 8];
                            for i in 0..4usize {
                                let cur_elem = tc_row + tc_offsets[i];
                                let cur_int = cur_elem / pack_factor;
                                let cur_pos = cur_elem % pack_factor;
                                vals[i] = (b1_vals[cur_int] >> (cur_pos * 4)) & 0x0f;
                                vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * 4)) & 0x0f;
                            }

                            let mut packed = 0u32;
                            for i in 0..8usize {
                                packed |= vals[pack_idx[i]] << (i * 4);
                            }
                            let tile_size = tile_k * tile_n / pack_factor;
                            let out_offset = (k_tile * n_tiles + n_tile) * tile_size;
                            expected[marlin_base + out_offset + th_id * 4 + warp_id] = packed;
                        }
                    }
                }
            }
        }

        let checkpoint_dev = ctx.stream.clone_htod(&checkpoint).expect("weight H2D");
        let mut marlin_dev = ctx
            .stream
            .alloc_zeros::<u32>(checkpoint.len())
            .expect("marlin weight alloc");
        {
            let (src_ptr, _src_guard) = checkpoint_dev.device_ptr(&ctx.stream);
            let (dst_ptr, _dst_guard) = marlin_dev.device_ptr_mut(&ctx.stream);
            let result = unsafe {
                crate::ffi::kimi_marlin_int4_reorder_weight_cuda(
                    src_ptr as *const u8,
                    dst_ptr as *mut u8,
                    in_dim as i32,
                    out_dim as i32,
                    local_experts as i32,
                    group_size as i32,
                    ctx.stream.cu_stream(),
                )
            };
            result.result().expect("marlin weight reorder");
        }
        let got = ctx.stream.clone_dtoh(&marlin_dev).expect("weight D2H");
        assert_eq!(got, expected);
    }
}
