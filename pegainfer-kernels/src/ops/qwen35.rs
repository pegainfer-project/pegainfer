//! Stable Qwen3.5 GDN prefill boundary.
//!
//! Generated CuTe symbols, tensor wrappers, TMA descriptors, module lifetime,
//! and the low-level launch ABI stop below this module. Model crates see only
//! the semantic geometry and device buffers used by Gated DeltaNet prefill.

use std::ffi::CStr;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtr;
use cudarc::driver::DevicePtrMut;

use crate::ffi;
use crate::tensor::DeviceContext;
use crate::tensor::HiddenStates;

pub const QWEN35_GDN_ABI_VERSION: u32 = 1;
const BF16_DTYPE: u32 = 1;
const F32_DTYPE: u32 = 2;
const HKV_V_CONTIGUOUS_LAYOUT: u32 = 1;
const STATUS_OK: i32 = 0;
const STATUS_NOT_SUPPORTED: i32 = 1;

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

    fn spec(self, sm: i32) -> Result<ffi::FlashInferGdnSpec> {
        Ok(ffi::FlashInferGdnSpec {
            abi_version: QWEN35_GDN_ABI_VERSION,
            struct_size: size_of::<ffi::FlashInferGdnSpec>() as u32,
            sm,
            h_q: self.h_q.try_into().context("GDN Hq exceeds u32")?,
            h_k: self.h_k.try_into().context("GDN Hk exceeds u32")?,
            h_v: self.h_v.try_into().context("GDN Hv exceeds u32")?,
            head_dim: self.head_dim.try_into().context("GDN D exceeds u32")?,
            qkv_dtype: BF16_DTYPE,
            state_dtype: F32_DTYPE,
            state_layout: HKV_V_CONTIGUOUS_LAYOUT,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Qwen35GdnSupport {
    Supported,
    UnsupportedSm,
    UnsupportedGeometry,
}

pub fn qwen35_gdn_capability(sm: i32, geometry: Qwen35GdnGeometry) -> Qwen35GdnSupport {
    if sm != 120 {
        Qwen35GdnSupport::UnsupportedSm
    } else if geometry != Qwen35GdnGeometry::PRODUCTION {
        Qwen35GdnSupport::UnsupportedGeometry
    } else {
        Qwen35GdnSupport::Supported
    }
}

fn linked_artifact_support(sm: i32, geometry: Qwen35GdnGeometry) -> Result<Qwen35GdnSupport> {
    let spec = geometry.spec(sm)?;
    let status = unsafe { ffi::pegainfer_qwen35_gdn_supported(&raw const spec) };
    match status {
        STATUS_OK => Ok(Qwen35GdnSupport::Supported),
        STATUS_NOT_SUPPORTED => Ok(if sm != 120 {
            Qwen35GdnSupport::UnsupportedSm
        } else {
            Qwen35GdnSupport::UnsupportedGeometry
        }),
        other => anyhow::bail!("Qwen3.5 GDN support query failed with stable ABI status {other}"),
    }
}

#[derive(Debug)]
pub struct Qwen35GdnAot {
    handle: NonNull<c_void>,
    device_ordinal: usize,
    geometry: Qwen35GdnGeometry,
    workspace_bytes: usize,
    successful_launches: Arc<AtomicU64>,
}

pub struct Qwen35GdnWorkspace {
    workspace: CudaSlice<u8>,
    cu_seqlens: CudaSlice<i64>,
    tokens: usize,
}

// The handle is bound to one CUDA device and all launches are issued by the
// owning model thread on its DeviceContext stream.
unsafe impl Send for Qwen35GdnAot {}
unsafe impl Sync for Qwen35GdnAot {}

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
        ensure!(
            linked_artifact_support(sm, geometry)? == Qwen35GdnSupport::Supported,
            "linked Qwen3.5 GDN artifact rejected its production specialization"
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
            successful_launches: Arc::new(AtomicU64::new(0)),
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

    pub fn artifact_size_bytes(&self) -> u64 {
        unsafe { ffi::pegainfer_qwen35_gdn_artifact_size_bytes() }
    }

    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }

    pub fn successful_launch_counter(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.successful_launches)
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

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    fn launch_separate_for_test(
        &self,
        ctx: &DeviceContext,
        q: &HiddenStates,
        k: &HiddenStates,
        v: &HiddenStates,
        alpha: &CudaSlice<f32>,
        beta: &CudaSlice<f32>,
        initial_state: &CudaSlice<f32>,
        state: &mut CudaSlice<f32>,
        output: &mut HiddenStates,
        launch_workspace: &mut Qwen35GdnWorkspace,
    ) -> Result<()> {
        let state_elements = self.geometry.h_v * self.geometry.head_dim * self.geometry.head_dim;
        ensure!(
            initial_state.len() == state_elements && state.len() == state_elements,
            "Qwen3.5 GDN separate-state length mismatch"
        );
        let (initial_state_ptr, _initial_state) = initial_state.device_ptr(&ctx.stream);
        let (state_ptr, _state) = state.device_ptr_mut(&ctx.stream);
        self.launch_with_state_pointers(
            ctx,
            q,
            k,
            v,
            alpha,
            beta,
            state_ptr,
            initial_state_ptr,
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
            cu_seqlens_len: 2,
            tokens: t.try_into().context("Qwen3.5 GDN T exceeds u32")?,
            h_q: g.h_q as u32,
            h_k: g.h_k as u32,
            h_v: g.h_v as u32,
            head_dim: g.head_dim as u32,
            stream: ctx.stream.cu_stream(),
        };
        let status =
            unsafe { ffi::pegainfer_qwen35_gdn_launch(self.handle.as_ptr(), &raw const args) };
        ensure!(
            status == STATUS_OK,
            "Qwen3.5 GDN launch failed with stable ABI status {status}"
        );
        self.successful_launches.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for Qwen35GdnAot {
    fn drop(&mut self) {
        unsafe { ffi::pegainfer_qwen35_gdn_destroy(self.handle.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;

    fn ensure_bitwise_f32(label: &str, expected: &[f32], actual: &[f32]) -> Result<()> {
        ensure!(
            expected.len() == actual.len(),
            "{label} length mismatch: expected {}, actual {}",
            expected.len(),
            actual.len()
        );
        if let Some(index) = expected
            .iter()
            .zip(actual)
            .position(|(expected, actual)| expected.to_bits() != actual.to_bits())
        {
            anyhow::bail!(
                "{label} first bitwise mismatch at {index}: expected={} actual={}",
                expected[index],
                actual[index]
            );
        }
        eprintln!("{label}: elements={} bitwise_mismatches=0", expected.len());
        Ok(())
    }

    #[test]
    fn stable_c_struct_layout_is_frozen() {
        macro_rules! assert_offsets {
            ($ty:ty, {$($field:ident: $offset:expr),+ $(,)?}) => {
                $(assert_eq!(std::mem::offset_of!($ty, $field), $offset);)+
            };
        }

        assert_eq!(size_of::<ffi::FlashInferGdnSpec>(), 40);
        assert_eq!(align_of::<ffi::FlashInferGdnSpec>(), 4);
        assert_offsets!(ffi::FlashInferGdnSpec, {
            abi_version: 0, struct_size: 4, sm: 8, h_q: 12, h_k: 16,
            h_v: 20, head_dim: 24, qkv_dtype: 28, state_dtype: 32,
            state_layout: 36,
        });

        assert_eq!(size_of::<ffi::FlashInferGdnPrefillArgs>(), 128);
        assert_eq!(align_of::<ffi::FlashInferGdnPrefillArgs>(), 8);
        assert_offsets!(ffi::FlashInferGdnPrefillArgs, {
            abi_version: 0, struct_size: 4, q: 8, k: 16, v: 24, output: 32,
            alpha: 40, beta: 48, state: 56, initial_state: 64, workspace: 72,
            workspace_bytes: 80, cu_seqlens: 88, cu_seqlens_len: 96,
            tokens: 100, h_q: 104, h_k: 108, h_v: 112, head_dim: 116,
            stream: 120,
        });
    }

    #[test]
    fn unsupported_geometry_is_explicit() {
        let hv48 = Qwen35GdnGeometry {
            h_v: 48,
            ..Qwen35GdnGeometry::PRODUCTION
        };
        assert_eq!(
            qwen35_gdn_capability(120, hv48),
            Qwen35GdnSupport::UnsupportedGeometry
        );
        assert_eq!(
            qwen35_gdn_capability(90, Qwen35GdnGeometry::PRODUCTION),
            Qwen35GdnSupport::UnsupportedSm
        );
        assert_eq!(
            qwen35_gdn_capability(120, Qwen35GdnGeometry::PRODUCTION),
            Qwen35GdnSupport::Supported
        );
    }

    #[test]
    #[ignore = "requires an SM120 GPU and a build-linked validated FlashInfer GDN AOT bundle"]
    fn sm120_stable_abi_alias_and_separate_state_are_bitwise_identical() -> Result<()> {
        let ctx = DeviceContext::new()?;
        let geometry = Qwen35GdnGeometry::PRODUCTION;
        let backend = Qwen35GdnAot::load_for_production(&ctx, geometry)?
            .context("validated FlashInfer GDN AOT bundle is not available on SM120")?;
        ensure!(
            backend.artifact_sha256() != "unavailable"
                && backend.artifact_sha256() != "invalid-utf8"
                && backend.artifact_sha256().len() == 64,
            "production boundary did not expose a linked object SHA-256"
        );
        ensure!(
            backend.artifact_size_bytes() > 0,
            "production boundary reported an empty linked object"
        );
        let launches_before = backend.successful_launch_counter().load(Ordering::Relaxed);

        let bf16_values = |elements: usize, modulus: usize, scale: f32| {
            (0..elements)
                .map(|index| {
                    let signed = (index % modulus) as i32 - (modulus / 2) as i32;
                    bf16::from_f32(signed as f32 * scale)
                })
                .collect::<Vec<_>>()
        };
        let state_elements = geometry.h_v * geometry.head_dim * geometry.head_dim;
        let initial_host = (0..geometry.h_v)
            .flat_map(|head| {
                (0..geometry.head_dim).flat_map(move |key| {
                    (0..geometry.head_dim)
                        .map(move |value| (head * 100_000 + key * 100 + value) as f32 * 1.0e-6)
                })
            })
            .collect::<Vec<_>>();
        ensure!(
            initial_host.len() == state_elements,
            "HKV fixture size mismatch"
        );

        for tokens in [1_usize, 63, 64, 65, 128] {
            let q = HiddenStates::from_host(
                &ctx,
                &bf16_values(tokens * geometry.h_q * geometry.head_dim, 127, 1.0 / 1024.0),
                geometry.h_q * geometry.head_dim,
                tokens,
            )?;
            let k = HiddenStates::from_host(
                &ctx,
                &bf16_values(tokens * geometry.h_k * geometry.head_dim, 113, 1.0 / 1024.0),
                geometry.h_k * geometry.head_dim,
                tokens,
            )?;
            let v = HiddenStates::from_host(
                &ctx,
                &bf16_values(tokens * geometry.h_v * geometry.head_dim, 97, 1.0 / 128.0),
                geometry.h_v * geometry.head_dim,
                tokens,
            )?;
            let alpha = ctx
                .stream
                .clone_htod(&vec![0.9921875_f32; tokens * geometry.h_v])?;
            let beta = ctx
                .stream
                .clone_htod(&vec![0.5_f32; tokens * geometry.h_v])?;

            let initial_state = ctx.stream.clone_htod(&initial_host)?;
            let mut separate_state: CudaSlice<f32> = ctx.stream.alloc_zeros(state_elements)?;
            let mut separate_output =
                HiddenStates::zeros(&ctx, geometry.h_v * geometry.head_dim, tokens)?;
            let mut separate_workspace = backend.allocate_workspace(&ctx, tokens)?;
            backend.launch_separate_for_test(
                &ctx,
                &q,
                &k,
                &v,
                &alpha,
                &beta,
                &initial_state,
                &mut separate_state,
                &mut separate_output,
                &mut separate_workspace,
            )?;

            let separate_output = separate_output.to_host(&ctx)?;
            let separate_state = ctx.stream.clone_dtoh(&separate_state)?;
            ctx.sync()?;

            ensure!(
                separate_output.iter().all(|value| value.is_finite()),
                "stable C ABI output contains a non-finite value at T={tokens}"
            );
            ensure!(
                separate_state.iter().all(|value| value.is_finite()),
                "stable C ABI final state contains a non-finite value at T={tokens}"
            );
            ensure!(
                separate_output.iter().any(|&value| value != 0.0),
                "stable C ABI output remained zero at T={tokens}"
            );
            ensure!(
                separate_state != initial_host,
                "stable C ABI recurrent state did not update at T={tokens}"
            );

            if tokens == 65 {
                let mut alias_state = ctx.stream.clone_htod(&initial_host)?;
                let mut alias_output =
                    HiddenStates::zeros(&ctx, geometry.h_v * geometry.head_dim, tokens)?;
                let mut alias_workspace = backend.allocate_workspace(&ctx, tokens)?;
                backend.launch_in_place(
                    &ctx,
                    &q,
                    &k,
                    &v,
                    &alpha,
                    &beta,
                    &mut alias_state,
                    &mut alias_output,
                    &mut alias_workspace,
                )?;
                let alias_output = alias_output.to_host(&ctx)?;
                let alias_state = ctx.stream.clone_dtoh(&alias_state)?;
                ctx.sync()?;
                ensure_bitwise_f32(
                    "stable C ABI alias/separate output [T=65,Hv=32,D=128,bf16]",
                    &separate_output,
                    &alias_output,
                )?;
                ensure_bitwise_f32(
                    "stable C ABI alias/separate final state [T=65,Hv=32,D=128,f32,HKV]",
                    &separate_state,
                    &alias_state,
                )?;
            }
        }

        let launches = backend.successful_launch_counter().load(Ordering::Relaxed);
        ensure!(
            launches - launches_before == 6,
            "stable C ABI launch counter expected five dynamic-T launches plus one alias launch, observed {}",
            launches - launches_before
        );
        Ok(())
    }
}
