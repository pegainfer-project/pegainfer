use std::sync::Once;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;
use half::bf16;

use crate::ffi;
use crate::tensor::DeviceContext;

static DSL_GEMM_INIT: Once = Once::new();
static DSL_GEMM_READY: AtomicBool = AtomicBool::new(false);

/// Whether the CuTe DSL AOT table is loaded and dispatching (after
/// [`glm52_fp8_dsl_preload`]). False on stub builds and under the kill switch.
pub fn glm52_fp8_dsl_gemm_ready() -> bool {
    DSL_GEMM_READY.load(Ordering::Acquire)
}

/// Load the CuTe DSL AOT GEMM modules (embedded cubins) onto every device.
/// Must run before any decode-bucket CUDA-graph capture — library load is
/// illegal mid-capture; the launch entries themselves are capture-safe.
/// No-op on stub builds (no PEGAINFER_CUTEDSL_PYTHON at compile time) and
/// under the `GLM52_FP8_DSL=0` kill switch.
pub fn glm52_fp8_dsl_preload() {
    DSL_GEMM_INIT.call_once(|| {
        if std::env::var("GLM52_FP8_DSL").is_ok_and(|v| v == "0") {
            log::info!("GLM5.2 CuTe DSL fp8 GEMM disabled by GLM52_FP8_DSL=0");
            return;
        }
        if unsafe { ffi::glm52_fp8_dsl_gemm_supported_cuda(64, 16384, 2048) } == 0 {
            return;
        }
        let rc = unsafe { ffi::glm52_fp8_dsl_gemm_load_cuda() };
        if rc == 0 {
            DSL_GEMM_READY.store(true, Ordering::Release);
            log::info!(
                "GLM5.2 CuTe DSL fp8 GEMM AOT loaded; wide-route buckets 16-96 dispatch to tcgen05"
            );
        } else {
            log::warn!("GLM5.2 CuTe DSL fp8 GEMM module load failed (rc={rc}); staying on CUTLASS");
        }
    });
}

pub fn glm52_fp8_groupwise_gemm_sm100_launch(
    ctx: &DeviceContext,
    m: usize,
    n: usize,
    k: usize,
    activation: &CudaSlice<u8>,
    activation_scale: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    weight_scale: &CudaSlice<u8>,
    output: &mut CudaSlice<bf16>,
    workspace: &mut CudaSlice<u8>,
) -> Result<()> {
    glm52_fp8_groupwise_gemm_sm100_offset_launch(
        ctx,
        m,
        n,
        k,
        activation,
        activation_scale,
        weight,
        0,
        weight_scale,
        0,
        output,
        workspace,
    )
}

#[allow(clippy::cast_ptr_alignment, clippy::too_many_arguments)]
pub fn glm52_fp8_groupwise_gemm_sm100_offset_launch(
    ctx: &DeviceContext,
    m: usize,
    n: usize,
    k: usize,
    activation: &CudaSlice<u8>,
    activation_scale: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    weight_offset: usize,
    weight_scale: &CudaSlice<u8>,
    weight_scale_offset: usize,
    output: &mut CudaSlice<bf16>,
    workspace: &mut CudaSlice<u8>,
) -> Result<()> {
    ensure!(
        m > 0 && m.is_multiple_of(4) && n > 0 && k.is_multiple_of(128),
        "GLM5.2 FP8 GEMM shape [{m}, {n}, {k}] requires m%4=0, n>0, and k%128=0"
    );
    ensure!(
        weight_scale_offset.is_multiple_of(align_of::<f32>()),
        "GLM5.2 FP8 weight scale offset {weight_scale_offset} is not f32-aligned"
    );
    ensure!(
        activation.len() >= m * k
            && activation_scale.len() >= m * k.div_ceil(128)
            && weight.len() >= weight_offset + n * k
            && weight_scale.len()
                >= weight_scale_offset + n.div_ceil(128) * k.div_ceil(128) * size_of::<f32>()
            && output.len() >= m * n
            && !workspace.is_empty(),
        "GLM5.2 FP8 GEMM buffers are too small for [{m}, {n}, {k}]"
    );
    let (activation_ptr, _activation_guard) = activation.device_ptr(&ctx.stream);
    let (activation_scale_ptr, _activation_scale_guard) = activation_scale.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (weight_scale_ptr, _weight_scale_guard) = weight_scale.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let workspace_bytes = workspace.len();
    let (workspace_ptr, _workspace_guard) = workspace.device_ptr_mut(&ctx.stream);
    // Exact-(m, n, k) table dispatch to the CuTe DSL tcgen05 AOT twin: only
    // the four wide-route projection shapes at decode buckets 16-96 hit it.
    // The branch is deterministic per bucket, so graph capture and replay
    // agree.
    if weight_offset == 0
        && weight_scale_offset == 0
        && DSL_GEMM_READY.load(Ordering::Acquire)
        && unsafe { ffi::glm52_fp8_dsl_gemm_supported_cuda(m as i32, n as i32, k as i32) } != 0
    {
        let rc = unsafe {
            ffi::glm52_fp8_dsl_gemm_cuda(
                activation_ptr as *const u8,
                activation_scale_ptr as *const f32,
                weight_ptr as *const u8,
                weight_scale_ptr as *const u8 as *const f32,
                output_ptr as *mut ffi::Half,
                workspace_ptr as *mut u8,
                workspace_bytes,
                m as i32,
                n as i32,
                k as i32,
                ctx.stream.cu_stream(),
            )
        };
        ensure!(
            rc == 0,
            "GLM5.2 CuTe DSL fp8 GEMM failed (rc={rc}) for [{m}, {n}, {k}]"
        );
        return Ok(());
    }
    let result = unsafe {
        ffi::glm52_fp8_groupwise_gemm_sm100_cuda(
            activation_ptr as *const u8,
            activation_scale_ptr as *const f32,
            (weight_ptr as *const u8).add(weight_offset),
            (weight_scale_ptr as *const u8)
                .add(weight_scale_offset)
                .cast::<f32>(),
            output_ptr as *mut ffi::Half,
            workspace_ptr as *mut u8,
            workspace_bytes,
            m as i32,
            n as i32,
            k as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 FP8 groupwise GEMM launch failed: {err}"))
}

#[allow(clippy::too_many_arguments)]
pub fn glm52_fp8_groupwise_gemm_sm100_bank_launch(
    ctx: &DeviceContext,
    m: usize,
    n: usize,
    k: usize,
    activation: &CudaSlice<u8>,
    activation_scale: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    weight_offset: usize,
    weight_scale: &CudaSlice<f32>,
    weight_scale_offset: usize,
    output: &mut CudaSlice<bf16>,
    workspace: &mut CudaSlice<u8>,
) -> Result<()> {
    ensure!(
        m > 0 && m.is_multiple_of(4) && n > 0 && k.is_multiple_of(128),
        "GLM5.2 bank FP8 GEMM shape [{m}, {n}, {k}] is invalid"
    );
    ensure!(
        activation.len() >= m * k
            && activation_scale.len() >= m * k.div_ceil(128)
            && weight.len() >= weight_offset + n * k
            && weight_scale.len() >= weight_scale_offset + n.div_ceil(128) * k.div_ceil(128)
            && output.len() >= m * n
            && !workspace.is_empty(),
        "GLM5.2 bank FP8 GEMM buffers are too small"
    );
    let (activation_ptr, _activation_guard) = activation.device_ptr(&ctx.stream);
    let (activation_scale_ptr, _activation_scale_guard) = activation_scale.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (weight_scale_ptr, _weight_scale_guard) = weight_scale.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let workspace_bytes = workspace.len();
    let (workspace_ptr, _workspace_guard) = workspace.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_fp8_groupwise_gemm_sm100_cuda(
            activation_ptr as *const u8,
            activation_scale_ptr as *const f32,
            (weight_ptr as *const u8).add(weight_offset),
            (weight_scale_ptr as *const f32).add(weight_scale_offset),
            output_ptr as *mut ffi::Half,
            workspace_ptr as *mut u8,
            workspace_bytes,
            m as i32,
            n as i32,
            k as i32,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 bank FP8 GEMM launch failed: {err}"))
}

/// Minimum workspace for the grouped GEMM: per-group pointer/stride/layout
/// arrays plus the CUTLASS grouped-kernel workspace.
pub const GLM52_FP8_GROUPED_GEMM_WORKSPACE_BYTES: usize = 16 << 20;

/// Grouped fp8 blockwise GEMM over `num_groups` expert slices (FlashInfer
/// `CutlassFP8GroupwiseScaledGroupGEMMSM100`, compiled from source — no
/// cubins): group `g` computes `out[m_indptr[g]..m_indptr[g+1], :n] =
/// deq(act rows) @ deq(weight[g])^T`. Activation rows are pre-gathered in
/// `m_indptr` order; `weight` is the routed slice bank base `[num_groups, n,
/// k]` fp8 with `weight_scale` `[num_groups, n/128, k/128]` f32 (checkpoint
/// layout as-is, same contract as the dense launch above). `m_indptr` is
/// device-resident — per-group row counts never touch the host.
#[allow(clippy::too_many_arguments)]
pub fn glm52_fp8_grouped_gemm_sm100_launch(
    ctx: &DeviceContext,
    max_m: usize,
    n: usize,
    k: usize,
    num_groups: usize,
    activation: &CudaSlice<u8>,
    activation_scale: &CudaSlice<f32>,
    weight: &CudaSlice<u8>,
    weight_scale: &CudaSlice<f32>,
    m_indptr: &CudaSlice<i32>,
    output: &mut CudaSlice<bf16>,
    workspace: &mut CudaSlice<u8>,
) -> Result<()> {
    ensure!(
        max_m > 0 && n > 0 && n.is_multiple_of(64) && k.is_multiple_of(128) && num_groups > 0,
        "GLM5.2 grouped FP8 GEMM shape [{max_m}, {n}, {k}] x {num_groups} is invalid"
    );
    ensure!(
        activation.len() >= max_m * k
            && activation_scale.len() >= max_m * k.div_ceil(128)
            && weight.len() >= num_groups * n * k
            && weight_scale.len() >= num_groups * n.div_ceil(128) * k.div_ceil(128)
            && m_indptr.len() > num_groups
            && output.len() >= max_m * n
            && workspace.len() >= GLM52_FP8_GROUPED_GEMM_WORKSPACE_BYTES,
        "GLM5.2 grouped FP8 GEMM buffers are too small for [{max_m}, {n}, {k}] x {num_groups}"
    );
    let (activation_ptr, _activation_guard) = activation.device_ptr(&ctx.stream);
    let (activation_scale_ptr, _activation_scale_guard) = activation_scale.device_ptr(&ctx.stream);
    let (weight_ptr, _weight_guard) = weight.device_ptr(&ctx.stream);
    let (weight_scale_ptr, _weight_scale_guard) = weight_scale.device_ptr(&ctx.stream);
    let (indptr_ptr, _indptr_guard) = m_indptr.device_ptr(&ctx.stream);
    let (output_ptr, _output_guard) = output.device_ptr_mut(&ctx.stream);
    let workspace_bytes = workspace.len();
    let (workspace_ptr, _workspace_guard) = workspace.device_ptr_mut(&ctx.stream);
    let result = unsafe {
        ffi::glm52_fp8_grouped_gemm_sm100_cuda(
            activation_ptr as *const u8,
            activation_scale_ptr as *const f32,
            weight_ptr as *const u8,
            weight_scale_ptr as *const f32,
            output_ptr as *mut ffi::Half,
            indptr_ptr as *const i32,
            max_m as i32,
            n as i32,
            k as i32,
            num_groups as i32,
            workspace_ptr as *mut u8,
            workspace_bytes,
            ctx.stream.cu_stream(),
        )
    };
    result
        .result()
        .map_err(|err| anyhow!("GLM5.2 grouped FP8 GEMM launch failed: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires an SM100/103 CUDA device"]
    fn fp8_groupwise_respects_row_major_weight() -> Result<()> {
        const M: usize = 4;
        const N: usize = 128;
        const K: usize = 128;
        const VALUES: [(u8, f32); 5] = [
            (0xc0, -2.0),
            (0xb8, -1.0),
            (0x00, 0.0),
            (0x38, 1.0),
            (0x40, 2.0),
        ];

        let ctx = DeviceContext::new()?;
        let mut activation_host = vec![0u8; M * K];
        for row in 0..M {
            for col in 0..K {
                activation_host[row * K + col] = VALUES[(row * 3 + col) % VALUES.len()].0;
            }
        }
        let mut weight_host = vec![0u8; N * K];
        for row in 0..N {
            weight_host[row * K + (row * 7 + 3) % K] =
                if row.is_multiple_of(2) { 0x38 } else { 0xb8 };
        }
        let activation = ctx.stream.clone_htod(&activation_host)?;
        let activation_scale = ctx.stream.clone_htod(&vec![1.0f32; M])?;
        let weight = ctx.stream.clone_htod(&weight_host)?;
        let weight_scale = ctx.stream.clone_htod(&1.0f32.to_ne_bytes())?;
        let mut output = ctx.stream.alloc_zeros::<bf16>(M * N)?;
        let mut workspace = ctx.stream.alloc_zeros::<u8>(32 << 20)?;

        glm52_fp8_groupwise_gemm_sm100_launch(
            &ctx,
            M,
            N,
            K,
            &activation,
            &activation_scale,
            &weight,
            &weight_scale,
            &mut output,
            &mut workspace,
        )?;
        let output = ctx.stream.clone_dtoh(&output)?;
        for row in 0..M {
            for col in 0..N {
                let source_col = (col * 7 + 3) % K;
                let source = VALUES[(row * 3 + source_col) % VALUES.len()].1;
                let want = if col.is_multiple_of(2) {
                    source
                } else {
                    -source
                };
                ensure!(
                    output[row * N + col].to_f32() == want,
                    "output[{row}, {col}]={} != {want}",
                    output[row * N + col].to_f32()
                );
            }
        }
        Ok(())
    }

    /// Grouped GEMM vs an exact hand-computed reference: 4 expert groups
    /// (one empty), one-hot ±1 weights, unit scales — every output element
    /// is an exact copy/negation of one activation value.
    #[test]
    #[ignore = "requires an SM100/103 CUDA device"]
    fn fp8_grouped_gemm_matches_reference() -> Result<()> {
        const GROUPS: usize = 4;
        const N: usize = 128;
        const K: usize = 256;
        const TOTAL_M: usize = 16;
        const INDPTR: [i32; GROUPS + 1] = [0, 3, 3, 8, 16];
        const VALUES: [(u8, f32); 5] = [
            (0xc0, -2.0),
            (0xb8, -1.0),
            (0x00, 0.0),
            (0x38, 1.0),
            (0x40, 2.0),
        ];

        let ctx = DeviceContext::new()?;
        let mut activation_host = vec![0u8; TOTAL_M * K];
        for row in 0..TOTAL_M {
            for col in 0..K {
                activation_host[row * K + col] = VALUES[(row * 3 + col) % VALUES.len()].0;
            }
        }
        let mut weight_host = vec![0u8; GROUPS * N * K];
        for group in 0..GROUPS {
            for row in 0..N {
                weight_host[group * N * K + row * K + (row * 7 + group * 3 + 3) % K] =
                    if row.is_multiple_of(2) { 0x38 } else { 0xb8 };
            }
        }
        let activation = ctx.stream.clone_htod(&activation_host)?;
        let activation_scale = ctx.stream.clone_htod(&vec![1.0f32; TOTAL_M * (K / 128)])?;
        let weight = ctx.stream.clone_htod(&weight_host)?;
        let weight_scale = ctx
            .stream
            .clone_htod(&vec![1.0f32; GROUPS * (N / 128) * (K / 128)])?;
        let m_indptr = ctx.stream.clone_htod(&INDPTR)?;
        let mut output = ctx.stream.alloc_zeros::<bf16>(TOTAL_M * N)?;
        let mut workspace = ctx
            .stream
            .alloc_zeros::<u8>(GLM52_FP8_GROUPED_GEMM_WORKSPACE_BYTES)?;

        glm52_fp8_grouped_gemm_sm100_launch(
            &ctx,
            TOTAL_M,
            N,
            K,
            GROUPS,
            &activation,
            &activation_scale,
            &weight,
            &weight_scale,
            &m_indptr,
            &mut output,
            &mut workspace,
        )?;
        let output = ctx.stream.clone_dtoh(&output)?;
        for group in 0..GROUPS {
            for row in INDPTR[group] as usize..INDPTR[group + 1] as usize {
                for col in 0..N {
                    let source_col = (col * 7 + group * 3 + 3) % K;
                    let source = VALUES[(row * 3 + source_col) % VALUES.len()].1;
                    let want = if col.is_multiple_of(2) {
                        source
                    } else {
                        -source
                    };
                    ensure!(
                        output[row * N + col].to_f32() == want,
                        "group {group} output[{row}, {col}]={} != {want}",
                        output[row * N + col].to_f32()
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires an SM100/103 CUDA device"]
    fn fp8_groupwise_handles_576_output_tail() -> Result<()> {
        const M: usize = 4;
        const N: usize = 576;
        const K: usize = 128;
        let ctx = DeviceContext::new()?;
        let activation = ctx.stream.clone_htod(&vec![0x38u8; M * K])?;
        let activation_scale = ctx.stream.clone_htod(&vec![1.0f32; M])?;
        let weight = ctx.stream.clone_htod(&vec![0x38u8; N * K])?;
        let scale_bytes: Vec<u8> = vec![1.0f32; N.div_ceil(128)]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect();
        let weight_scale = ctx.stream.clone_htod(&scale_bytes)?;
        let mut output = ctx.stream.alloc_zeros::<bf16>(M * N)?;
        let mut workspace = ctx.stream.alloc_zeros::<u8>(32 << 20)?;
        glm52_fp8_groupwise_gemm_sm100_launch(
            &ctx,
            M,
            N,
            K,
            &activation,
            &activation_scale,
            &weight,
            &weight_scale,
            &mut output,
            &mut workspace,
        )?;
        let output = ctx.stream.clone_dtoh(&output)?;
        ensure!(
            output.iter().all(|value| value.to_f32() == K as f32),
            "576-column tail GEMM produced an unexpected value"
        );
        Ok(())
    }
}
