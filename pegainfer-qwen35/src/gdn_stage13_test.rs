//! Stage 13 real-SM120 correctness gate through the kernels-owned stable ABI.
//!
//! This test deliberately knows only the semantic `Qwen35GdnAot` surface. It
//! must not reconstruct generated CuTe symbols, TMA descriptors, or the raw C
//! launch argument layout owned by `pegainfer-kernels`.

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use cudarc::driver::CudaSlice;
use cudarc::driver::DevicePtrMut;
use half::bf16;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceVec;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::Qwen35GdnAot;
use pegainfer_kernels::ops::Qwen35GdnGeometry;

use crate::gdn_prepare_test_contract::Fixture;
use crate::gdn_prepare_test_contract::Prepared;
use crate::gdn_prepare_test_contract::bf16_to_f32;
use crate::gdn_prepare_test_contract::deterministic_fixture;
use crate::gdn_prepare_test_contract::prepare;
use crate::gdn_stage7_test_support::DifferenceStats;
use crate::gdn_stage7_test_support::PREPARE_GATE_TOLERANCE;
use crate::gdn_stage7_test_support::PREPARE_QK_TOLERANCE;
use crate::gdn_stage7_test_support::RECURRENCE_OUTPUT_TOLERANCE;
use crate::gdn_stage7_test_support::RECURRENCE_STATE_TOLERANCE;
use crate::gdn_stage7_test_support::asymmetric_hkv_state;
use crate::gdn_stage7_test_support::cpu_decode_from_raw;
use crate::gdn_stage7_test_support::cpu_stepwise;
use crate::gdn_stage7_test_support::transpose_kv_as_wrong_hvk;
use crate::prefill_buffers::GdnPrepareScratch35;
use crate::prefill_buffers::GdrChunkwiseScratch35;

struct DeviceFixture {
    qkv: HiddenStates,
    b: HiddenStates,
    a: HiddenStates,
    dt_bias: DeviceVec,
    a_log: CudaSlice<f32>,
}

fn bf16_from_bits(values: &[u16]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_bits).collect()
}

fn f32_from_bits(values: &[u16]) -> Vec<f32> {
    values.iter().copied().map(bf16_to_f32).collect()
}

fn upload_fixture(ctx: &DeviceContext, fixture: &Fixture) -> Result<DeviceFixture> {
    Ok(DeviceFixture {
        qkv: HiddenStates::from_host(
            ctx,
            &bf16_from_bits(&fixture.qkv),
            fixture.offsets.total,
            fixture.geometry.tokens,
        )?,
        b: HiddenStates::from_host(
            ctx,
            &bf16_from_bits(&fixture.b),
            fixture.geometry.h_v,
            fixture.geometry.tokens,
        )?,
        a: HiddenStates::from_host(
            ctx,
            &bf16_from_bits(&fixture.a),
            fixture.geometry.h_v,
            fixture.geometry.tokens,
        )?,
        dt_bias: DeviceVec::from_host(ctx, &bf16_from_bits(&fixture.dt_bias))?,
        a_log: ctx.stream.clone_htod(&fixture.a_log)?,
    })
}

fn log_and_gate(
    label: &str,
    reference: &[f32],
    candidate: &[f32],
    tolerance: crate::gdn_stage7_test_support::NumericTolerance,
) -> Result<DifferenceStats> {
    let stats =
        DifferenceStats::compare(reference, candidate, tolerance).map_err(anyhow::Error::msg)?;
    eprintln!("{label}: {stats:?}");
    stats.ensure_within(label).map_err(anyhow::Error::msg)?;
    Ok(stats)
}

fn validate_gpu_prepare(
    ctx: &DeviceContext,
    scratch: &GdnPrepareScratch35,
    expected: &Prepared,
    tokens: usize,
) -> Result<Prepared> {
    let status = ctx.stream.clone_dtoh(&scratch.non_finite_status)?;
    let q = ctx.stream.clone_dtoh(&scratch.q.data)?;
    let k = ctx.stream.clone_dtoh(&scratch.k.data)?;
    let v = ctx.stream.clone_dtoh(&scratch.v.data)?;
    let alpha = ctx.stream.clone_dtoh(&scratch.alpha)?;
    let beta = ctx.stream.clone_dtoh(&scratch.beta)?;
    ctx.sync()?;
    ensure!(
        status == [0],
        "native prepare rejected finite Stage 13 fixture"
    );

    let q_bits = q.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    let k_bits = k.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    let v_bits = v.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    let q_f32 = q.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
    let k_f32 = k.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
    log_and_gate(
        &format!("prepare.q Hv=32 T={tokens}"),
        &f32_from_bits(&expected.q),
        &q_f32,
        PREPARE_QK_TOLERANCE,
    )?;
    log_and_gate(
        &format!("prepare.k Hv=32 T={tokens}"),
        &f32_from_bits(&expected.k),
        &k_f32,
        PREPARE_QK_TOLERANCE,
    )?;
    ensure!(
        v_bits == expected.v,
        "prepare.v changed BF16 bits at Hv=32 T={tokens}"
    );
    log_and_gate(
        &format!("prepare.alpha Hv=32 T={tokens}"),
        &expected.alpha,
        &alpha,
        PREPARE_GATE_TOLERANCE,
    )?;
    log_and_gate(
        &format!("prepare.beta Hv=32 T={tokens}"),
        &expected.beta,
        &beta,
        PREPARE_GATE_TOLERANCE,
    )?;
    Ok(Prepared {
        q: q_bits,
        k: k_bits,
        v: v_bits,
        alpha,
        beta,
    })
}

fn gate_first_decode_handoff(
    ctx: &DeviceContext,
    cpu_prefill_state: &[f32],
    triton_state: &mut CudaSlice<f32>,
    flashinfer_state: &mut CudaSlice<f32>,
    tokens: usize,
) -> Result<()> {
    let fixture = deterministic_fixture(1, 32);
    let cpu = cpu_decode_from_raw(&fixture, cpu_prefill_state).map_err(anyhow::Error::msg)?;
    let repeat_twice = |values: &[u16]| values.iter().chain(values).copied().collect::<Vec<_>>();
    let qkv = HiddenStates::from_host(
        ctx,
        &bf16_from_bits(&repeat_twice(&fixture.qkv)),
        fixture.offsets.total,
        2,
    )?;
    let b = HiddenStates::from_host(ctx, &bf16_from_bits(&repeat_twice(&fixture.b)), 32, 2)?;
    let a = HiddenStates::from_host(ctx, &bf16_from_bits(&repeat_twice(&fixture.a)), 32, 2)?;
    let dt_bias = DeviceVec::from_host(ctx, &bf16_from_bits(&fixture.dt_bias))?;
    let a_log = ctx.stream.clone_htod(&fixture.a_log)?;
    let state_ptrs = {
        let (triton_ptr, _triton) = triton_state.device_ptr_mut(&ctx.stream);
        let (flashinfer_ptr, _flashinfer) = flashinfer_state.device_ptr_mut(&ctx.stream);
        ctx.stream.clone_htod(&[triton_ptr, flashinfer_ptr])?
    };
    let mut output = HiddenStates::zeros(ctx, 32 * 128, 2)?;
    crate::ops::gated_delta_rule_decode_batch_into(
        ctx,
        &qkv,
        &b,
        &a,
        &dt_bias,
        &a_log,
        &state_ptrs,
        &mut output,
        2,
        16,
        32,
        128,
        128,
    );

    let output = output.to_host(ctx)?;
    let triton_state = ctx.stream.clone_dtoh(triton_state)?;
    let flashinfer_state = ctx.stream.clone_dtoh(flashinfer_state)?;
    ctx.sync()?;
    let row = 32 * 128;
    for (label, reference, candidate, tolerance) in [
        (
            format!("first-decode CPU/Triton output Hv=32 after T={tokens}"),
            cpu.output.as_slice(),
            &output[..row],
            RECURRENCE_OUTPUT_TOLERANCE,
        ),
        (
            format!("first-decode CPU/FlashInfer output Hv=32 after T={tokens}"),
            cpu.output.as_slice(),
            &output[row..],
            RECURRENCE_OUTPUT_TOLERANCE,
        ),
        (
            format!("first-decode Triton/FlashInfer output Hv=32 after T={tokens}"),
            &output[..row],
            &output[row..],
            RECURRENCE_OUTPUT_TOLERANCE,
        ),
        (
            format!("first-decode CPU/Triton state Hv=32 after T={tokens}"),
            cpu.final_state.as_slice(),
            triton_state.as_slice(),
            RECURRENCE_STATE_TOLERANCE,
        ),
        (
            format!("first-decode CPU/FlashInfer state Hv=32 after T={tokens}"),
            cpu.final_state.as_slice(),
            flashinfer_state.as_slice(),
            RECURRENCE_STATE_TOLERANCE,
        ),
        (
            format!("first-decode Triton/FlashInfer state Hv=32 after T={tokens}"),
            triton_state.as_slice(),
            flashinfer_state.as_slice(),
            RECURRENCE_STATE_TOLERANCE,
        ),
    ] {
        log_and_gate(&label, reference, candidate, tolerance)?;
    }
    Ok(())
}

#[test]
#[ignore = "requires an SM120 GPU and a build-linked validated FlashInfer GDN AOT bundle"]
fn sm120_stable_abi_operator_gate_covers_hv32_dynamic_t_and_first_decode() -> Result<()> {
    let ctx = DeviceContext::new()?;
    let geometry = Qwen35GdnGeometry::PRODUCTION;
    let backend = Qwen35GdnAot::load_for_production(&ctx, geometry)?
        .context("validated FlashInfer GDN AOT bundle is not available on SM120")?;
    ensure!(
        backend.artifact_sha256() != "unavailable" && backend.artifact_size_bytes() > 0,
        "stable ABI did not expose linked artifact identity"
    );
    let launches_before = backend
        .successful_launch_counter()
        .load(std::sync::atomic::Ordering::Relaxed);

    for tokens in [1_usize, 2, 63, 64, 65, 127, 128] {
        let fixture = deterministic_fixture(tokens, 32);
        let expected_prepare = prepare(&fixture).map_err(anyhow::Error::msg)?;
        let device = upload_fixture(&ctx, &fixture)?;
        let mut prepared = GdnPrepareScratch35::from_dims(&ctx, 16, 16, 32, 128, tokens)?;
        crate::ops::gated_delta_rule_prefill_native_prepare_into(
            &ctx,
            &device.qkv,
            &device.b,
            &device.a,
            &device.dt_bias,
            &device.a_log,
            &mut prepared,
            16,
            16,
            32,
            128,
        )?;
        let actual_prepare = validate_gpu_prepare(&ctx, &prepared, &expected_prepare, tokens)?;
        let initial_host = asymmetric_hkv_state(fixture.geometry);
        let cpu = cpu_stepwise(fixture.geometry, &actual_prepare, &initial_host)
            .map_err(anyhow::Error::msg)?;

        if tokens == 1 {
            let wrong_hvk = transpose_kv_as_wrong_hvk(fixture.geometry, &initial_host);
            let wrong_cpu = cpu_stepwise(fixture.geometry, &actual_prepare, &wrong_hvk)
                .map_err(anyhow::Error::msg)?;
            let wrong_output = DifferenceStats::compare(
                &cpu.output,
                &wrong_cpu.output,
                RECURRENCE_OUTPUT_TOLERANCE,
            )
            .map_err(anyhow::Error::msg)?;
            let wrong_state = DifferenceStats::compare(
                &cpu.final_state,
                &wrong_cpu.final_state,
                RECURRENCE_STATE_TOLERANCE,
            )
            .map_err(anyhow::Error::msg)?;
            ensure!(
                wrong_output.violations > 0 || wrong_state.violations > 0,
                "wrong-HVK negative oracle was not detected"
            );
        }

        let mut triton_state = ctx.stream.clone_htod(&initial_host)?;
        let mut triton_scratch = GdrChunkwiseScratch35::from_dims(&ctx, 32, 128, 128, tokens)?;
        let mut triton_output = HiddenStates::zeros(&ctx, 32 * 128, tokens)?;
        crate::ops::gated_delta_rule_prefill_chunkwise_into(
            &ctx,
            &device.qkv,
            &device.b,
            &device.a,
            &device.dt_bias,
            &device.a_log,
            &mut triton_state,
            &mut triton_scratch,
            &mut triton_output,
            16,
            32,
            128,
            128,
        )?;

        let mut flashinfer_state = ctx.stream.clone_htod(&initial_host)?;
        ensure!(
            flashinfer_state.len() == 32 * 128 * 128,
            "Stage 13 recurrent-state allocation mismatch"
        );
        let mut flashinfer_output = HiddenStates::zeros(&ctx, 32 * 128, tokens)?;
        let mut workspace = backend.allocate_workspace(&ctx, tokens)?;
        backend.launch_in_place(
            &ctx,
            &prepared.q,
            &prepared.k,
            &prepared.v,
            &prepared.alpha,
            &prepared.beta,
            &mut flashinfer_state,
            &mut flashinfer_output,
            &mut workspace,
        )?;

        let triton_output_host = triton_output.to_host(&ctx)?;
        let flashinfer_output_host = flashinfer_output.to_host(&ctx)?;
        let triton_state_host = ctx.stream.clone_dtoh(&triton_state)?;
        let flashinfer_state_host = ctx.stream.clone_dtoh(&flashinfer_state)?;
        ctx.sync()?;
        ensure!(
            flashinfer_output_host.iter().all(|value| value.is_finite())
                && flashinfer_state_host.iter().all(|value| value.is_finite()),
            "FlashInfer stable ABI produced non-finite values at T={tokens}"
        );
        ensure!(
            flashinfer_output_host.iter().any(|&value| value != 0.0),
            "FlashInfer stable ABI output remained zero at T={tokens}"
        );
        ensure!(
            flashinfer_state_host != initial_host,
            "FlashInfer stable ABI state did not update at T={tokens}"
        );

        for (label, reference, candidate, tolerance) in [
            (
                format!("prefill CPU/Triton output Hv=32 T={tokens}"),
                cpu.output.as_slice(),
                triton_output_host.as_slice(),
                RECURRENCE_OUTPUT_TOLERANCE,
            ),
            (
                format!("prefill CPU/FlashInfer output Hv=32 T={tokens}"),
                cpu.output.as_slice(),
                flashinfer_output_host.as_slice(),
                RECURRENCE_OUTPUT_TOLERANCE,
            ),
            (
                format!("prefill Triton/FlashInfer output Hv=32 T={tokens}"),
                triton_output_host.as_slice(),
                flashinfer_output_host.as_slice(),
                RECURRENCE_OUTPUT_TOLERANCE,
            ),
            (
                format!("prefill CPU/Triton state Hv=32 T={tokens}"),
                cpu.final_state.as_slice(),
                triton_state_host.as_slice(),
                RECURRENCE_STATE_TOLERANCE,
            ),
            (
                format!("prefill CPU/FlashInfer state Hv=32 T={tokens}"),
                cpu.final_state.as_slice(),
                flashinfer_state_host.as_slice(),
                RECURRENCE_STATE_TOLERANCE,
            ),
            (
                format!("prefill Triton/FlashInfer state Hv=32 T={tokens}"),
                triton_state_host.as_slice(),
                flashinfer_state_host.as_slice(),
                RECURRENCE_STATE_TOLERANCE,
            ),
        ] {
            log_and_gate(&label, reference, candidate, tolerance)?;
        }

        gate_first_decode_handoff(
            &ctx,
            &cpu.final_state,
            &mut triton_state,
            &mut flashinfer_state,
            tokens,
        )?;
    }

    let launches = backend
        .successful_launch_counter()
        .load(std::sync::atomic::Ordering::Relaxed);
    ensure!(
        launches - launches_before == 7,
        "Stage 13 expected seven stable-ABI launches, observed {}",
        launches - launches_before
    );
    Ok(())
}
