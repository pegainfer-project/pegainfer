//! Stable Qwen3.5 GDN prefill boundary.
//!
//! Generated CuTe symbols, tensor wrappers, TMA descriptors, module lifetime,
//! and the low-level launch ABI stop below this module. Model crates see only
//! the semantic geometry and device buffers used by Gated DeltaNet prefill.

use std::ffi::CStr;
use std::ffi::c_void;
use std::ptr::NonNull;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::HiddenStates;

const QWEN35_GDN_ABI_VERSION: u32 = 1;
const STATUS_OK: i32 = 0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen35GdnGeometry {
    pub h_q: usize,
    pub h_k: usize,
    pub h_v: usize,
    pub head_dim: usize,
}

impl Qwen35GdnGeometry {
    pub const PRODUCTION: Self = Self {
        h_q: 16,
        h_k: 16,
        h_v: 32,
        head_dim: 128,
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen35GdnSupport {
    Supported,
    UnsupportedSm,
    UnsupportedGeometry,
}

fn qwen35_gdn_capability(sm: i32, geometry: Qwen35GdnGeometry) -> Qwen35GdnSupport {
    if sm != 120 {
        Qwen35GdnSupport::UnsupportedSm
    } else if geometry != Qwen35GdnGeometry::PRODUCTION {
        Qwen35GdnSupport::UnsupportedGeometry
    } else {
        Qwen35GdnSupport::Supported
    }
}

#[derive(Debug)]
pub struct Qwen35GdnAot {
    handle: NonNull<c_void>,
    device_ordinal: usize,
    geometry: Qwen35GdnGeometry,
    workspace_bytes: usize,
}

pub struct Qwen35GdnWorkspace {
    workspace: CudaSlice<u8>,
    cu_seqlens: CudaSlice<i64>,
    tokens: usize,
}

// The handle is bound to one CUDA device and all launches are issued by the
// owning model thread on its DeviceContext stream.
unsafe impl Send for Qwen35GdnAot {}

impl Qwen35GdnAot {
    pub fn load_for_production(
        ctx: &DeviceContext,
        geometry: Qwen35GdnGeometry,
    ) -> Result<Option<Self>> {
        let (major, minor) = ctx.ctx.compute_capability()?;
        let sm = major * 10 + minor;
        if qwen35_gdn_capability(sm, geometry) != Qwen35GdnSupport::Supported {
            return Ok(None);
        }
        ensure!(
            unsafe { ffi::pegainfer_qwen35_gdn_abi_version() } == QWEN35_GDN_ABI_VERSION,
            "Qwen3.5 GDN stable C ABI version mismatch"
        );
        ensure!(
            unsafe { ffi::pegainfer_qwen35_gdn_aot_available() } == 1,
            "SM120/Hv32 selects FlashInfer GDN, but the validated prebuilt AOT artifact was not linked; set PEGAINFER_QWEN35_GDN_AOT_BUNDLE at build time"
        );
        let mut raw = std::ptr::null_mut();
        let status =
            unsafe { ffi::pegainfer_qwen35_gdn_create(&raw mut raw, ctx.device_ordinal as i32) };
        ensure!(
            status == STATUS_OK,
            "Qwen3.5 GDN preload failed with stable ABI status {status}"
        );
        let handle = NonNull::new(raw).context("Qwen3.5 GDN preload returned a null handle")?;
        let mut workspace_bytes = 0;
        let status = unsafe {
            ffi::pegainfer_qwen35_gdn_workspace_bytes(handle.as_ptr(), &raw mut workspace_bytes)
        };
        if status != STATUS_OK {
            unsafe { ffi::pegainfer_qwen35_gdn_destroy(handle.as_ptr()) };
            anyhow::bail!("Qwen3.5 GDN workspace query failed with stable ABI status {status}");
        }
        Ok(Some(Self {
            handle,
            device_ordinal: ctx.device_ordinal,
            geometry,
            workspace_bytes,
        }))
    }

    pub fn artifact_sha256(&self) -> &'static str {
        let pointer = unsafe { ffi::pegainfer_qwen35_gdn_artifact_sha256() };
        if pointer.is_null() {
            return "unavailable";
        }
        unsafe { CStr::from_ptr(pointer) }
            .to_str()
            .unwrap_or("invalid-utf8")
    }

    pub fn allocate_workspace(
        &self,
        ctx: &DeviceContext,
        tokens: usize,
    ) -> Result<Qwen35GdnWorkspace> {
        ensure!(tokens > 0, "Qwen3.5 GDN workspace requires T>=1");
        let workspace = ctx
            .stream
            .alloc_zeros(self.workspace_bytes)
            .map_err(|error| anyhow::anyhow!("allocate Qwen3.5 GDN workspace: {error}"))?;
        let end = i64::try_from(tokens).context("Qwen3.5 GDN T exceeds i64")?;
        let cu_seqlens = ctx
            .stream
            .clone_htod(&[0_i64, end])
            .map_err(|error| anyhow::anyhow!("upload Qwen3.5 GDN sequence metadata: {error}"))?;
        Ok(Qwen35GdnWorkspace {
            workspace,
            cu_seqlens,
            tokens,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn launch_in_place(
        &self,
        ctx: &DeviceContext,
        q: &HiddenStates,
        k: &HiddenStates,
        v: &HiddenStates,
        alpha: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        output: &mut HiddenStates,
        launch_workspace: &mut Qwen35GdnWorkspace,
    ) -> Result<()> {
        let state_elements = self.geometry.h_v * self.geometry.head_dim * self.geometry.head_dim;
        ensure!(
            state.len() == state_elements,
            "Qwen3.5 GDN state length mismatch"
        );
        let (state_ptr, _state) = state.device_ptr_mut(&ctx.stream);
        self.launch_with_state_pointers(
            ctx,
            q,
            k,
            v,
            alpha,
            beta,
            state_ptr,
            state_ptr,
            output,
            launch_workspace,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn launch_with_state_pointers(
        &self,
        ctx: &DeviceContext,
        q: &HiddenStates,
        k: &HiddenStates,
        v: &HiddenStates,
        alpha: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        state_ptr: u64,
        initial_state_ptr: u64,
        output: &mut HiddenStates,
        launch_workspace: &mut Qwen35GdnWorkspace,
    ) -> Result<()> {
        let t = q.seq_len;
        let g = self.geometry;
        ensure!(
            ctx.device_ordinal == self.device_ordinal,
            "Qwen3.5 GDN device mismatch"
        );
        ensure!(
            t > 0 && k.seq_len == t && v.seq_len == t && output.seq_len == t,
            "Qwen3.5 GDN token extents do not match"
        );
        ensure!(
            q.hidden_dim == g.h_q * g.head_dim
                && k.hidden_dim == g.h_k * g.head_dim
                && v.hidden_dim == g.h_v * g.head_dim
                && output.hidden_dim == g.h_v * g.head_dim,
            "Qwen3.5 GDN tensor geometry mismatch"
        );
        ensure!(
            alpha.len() == t * g.h_v
                && beta.len() == t * g.h_v
                && launch_workspace.workspace.len() >= self.workspace_bytes
                && launch_workspace.cu_seqlens.len() == 2
                && launch_workspace.tokens == t,
            "Qwen3.5 GDN buffer contract mismatch"
        );

        let (q_ptr, _q) = q.data.device_ptr(&ctx.stream);
        let (k_ptr, _k) = k.data.device_ptr(&ctx.stream);
        let (v_ptr, _v) = v.data.device_ptr(&ctx.stream);
        let (alpha_ptr, _alpha) = alpha.device_ptr(&ctx.stream);
        let (beta_ptr, _beta) = beta.device_ptr(&ctx.stream);
        let (output_ptr, _output) = output.data.device_ptr_mut(&ctx.stream);
        let workspace_bytes = launch_workspace.workspace.len() as u64;
        let (workspace_ptr, _workspace) = launch_workspace.workspace.device_ptr_mut(&ctx.stream);
        let (cu_ptr, _cu) = launch_workspace.cu_seqlens.device_ptr(&ctx.stream);
        let args = ffi::FlashInferGdnPrefillArgs {
            abi_version: QWEN35_GDN_ABI_VERSION,
            struct_size: size_of::<ffi::FlashInferGdnPrefillArgs>() as u32,
            q: q_ptr,
            k: k_ptr,
            v: v_ptr,
            output: output_ptr,
            alpha: alpha_ptr,
            beta: beta_ptr,
            state: state_ptr,
            initial_state: initial_state_ptr,
            workspace: workspace_ptr,
            workspace_bytes,
            cu_seqlens: cu_ptr,
            tokens: t.try_into().context("Qwen3.5 GDN T exceeds u32")?,
            stream: ctx.stream.cu_stream(),
        };
        let status =
            unsafe { ffi::pegainfer_qwen35_gdn_launch(self.handle.as_ptr(), &raw const args) };
        ensure!(
            status == STATUS_OK,
            "Qwen3.5 GDN launch failed with stable ABI status {status}"
        );
        Ok(())
    }
}

impl Drop for Qwen35GdnAot {
    fn drop(&mut self) {
        unsafe { ffi::pegainfer_qwen35_gdn_destroy(self.handle.as_ptr()) };
    }
}

#[cfg(test)]
mod tests;
