use half::bf16;

use super::*;

impl Qwen35GdnAot {
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
}

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

fn assert_stable_c_struct_layout() {
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
fn stable_c_struct_layout_is_frozen() {
    assert_stable_c_struct_layout();
}

#[test]
#[ignore = "requires an SM120 GPU and a build-linked validated FlashInfer GDN AOT bundle"]
fn sm120_stable_abi_alias_and_separate_state_are_bitwise_identical() -> Result<()> {
    assert_stable_c_struct_layout();
    let ctx = DeviceContext::new()?;
    let unsupported = Qwen35GdnGeometry {
        h_v: 48,
        ..Qwen35GdnGeometry::PRODUCTION
    };
    ensure!(
        Qwen35GdnAot::load_for_production(&ctx, unsupported)?.is_none(),
        "production load boundary accepted unsupported Hv48 geometry on SM120"
    );

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

    Ok(())
}
