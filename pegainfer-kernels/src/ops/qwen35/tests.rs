use std::path::Path;

use half::bf16;

use super::*;

fn read_tensor<T, const N: usize>(
    path: &Path,
    elements: usize,
    decode: impl Fn([u8; N]) -> T,
) -> Result<Vec<T>> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        bytes.len() == elements * N,
        "{}: expected {} bytes, got {}",
        path.display(),
        elements * N,
        bytes.len()
    );
    Ok(bytes
        .chunks_exact(N)
        .map(|chunk| decode(chunk.try_into().expect("exact tensor element width")))
        .collect())
}

fn compare_layout_result(label: &str, expected: &[f32], actual: &[f32]) -> Result<()> {
    ensure!(expected.len() == actual.len(), "{label}: length mismatch");
    let mut deltas = Vec::with_capacity(expected.len());
    let mut first_difference = None;
    for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
        let delta = (expected - actual).abs();
        deltas.push(delta);
        if first_difference.is_none()
            && (!expected.is_finite() || !actual.is_finite() || delta != 0.0)
        {
            first_difference = Some((index, expected, actual));
        }
    }
    deltas.sort_by(f32::total_cmp);
    let mean = deltas.iter().map(|&value| f64::from(value)).sum::<f64>() / deltas.len() as f64;
    let p99 = deltas[(deltas.len() - 1) * 99 / 100];
    let max = deltas[deltas.len() - 1];
    eprintln!(
        "{label}: elements={} first_difference={first_difference:?} mean_abs={mean:.9e} p99_abs={p99:.9e} max_abs={max:.9e} atol=0 rtol=0",
        expected.len()
    );
    // The HKV patch changes only state addresses; both artifacts execute the
    // same arithmetic with the same dtype. No model-logit tolerance applies.
    ensure!(
        first_difference.is_none(),
        "{label}: layout patch changed GDN results"
    );
    Ok(())
}

fn assert_stable_c_struct_layout() {
    macro_rules! assert_offsets {
        ($ty:ty, {$($field:ident: $offset:expr),+ $(,)?}) => {
            $(assert_eq!(std::mem::offset_of!($ty, $field), $offset);)+
        };
    }

    assert_eq!(size_of::<ffi::FlashInferGdnPrefillArgs>(), 104);
    assert_eq!(align_of::<ffi::FlashInferGdnPrefillArgs>(), 8);
    assert_offsets!(ffi::FlashInferGdnPrefillArgs, {
        abi_version: 0, struct_size: 4, q: 8, k: 16, v: 24, output: 32,
        alpha: 40, beta: 48, state: 56, workspace: 64,
        workspace_bytes: 72, cu_seqlens: 80, tokens: 88, stream: 96,
    });
}

#[test]
#[ignore = "requires SM120, a linked candidate and the generated upstream layout reference"]
fn sm120_stable_in_place_abi_matches_upstream_layout_reference() -> Result<()> {
    assert_stable_c_struct_layout();
    let directory = std::env::var_os("PEGAINFER_QWEN35_GDN_LAYOUT_REFERENCE").context(
        "PEGAINFER_QWEN35_GDN_LAYOUT_REFERENCE must name the generated upstream reference",
    )?;
    let directory = Path::new(&directory);
    let ctx = DeviceContext::new()?;
    let geometry = Qwen35GdnGeometry::PRODUCTION;
    let unsupported = Qwen35GdnAot::load_for_production(
        &ctx,
        Qwen35GdnGeometry {
            h_v: 48,
            ..geometry
        },
    )
    .expect_err("the production loader must reject unsupported Hv48 geometry");
    ensure!(
        unsupported
            .to_string()
            .contains("requires Hq=16/Hk=16/Hv=32/D=128"),
        "unsupported geometry failed for the wrong reason: {unsupported}"
    );
    let backend = Qwen35GdnAot::load_for_production(&ctx, geometry)?;
    let reference_object = std::fs::read_to_string(directory.join("patched-object.sha256"))?;
    ensure!(
        reference_object.trim().len() == 64 && reference_object.trim() == backend.artifact_sha256(),
        "layout reference must belong to the object linked by the production build"
    );
    let sm_count = ctx.ctx.attribute(
        cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
    )? as usize;
    ensure!(
        backend.workspace_bytes() == sm_count * 128,
        "GDN workspace query violates the per-SM contract"
    );
    let q_width = geometry.h_q * geometry.head_dim;
    let k_width = geometry.h_k * geometry.head_dim;
    let v_width = geometry.h_v * geometry.head_dim;
    let state_elements = v_width * geometry.head_dim;

    for (case, total_tokens, chunk_tokens) in [
        ("t1", 1, 1),
        ("t63", 63, 63),
        ("t64", 64, 64),
        ("t65", 65, 65),
        ("t128", 128, 128),
        ("resumed64", 128, 64),
    ] {
        let case_dir = directory.join(case);
        let read_bf16 = |name: &str, count| {
            read_tensor(&case_dir.join(name), count, |bytes| {
                bf16::from_bits(u16::from_le_bytes(bytes))
            })
        };
        let read_f32 =
            |name: &str, count| read_tensor(&case_dir.join(name), count, f32::from_le_bytes);
        let q_host = read_bf16("q.bf16", total_tokens * q_width)?;
        let k_host = read_bf16("k.bf16", total_tokens * k_width)?;
        let v_host = read_bf16("v.bf16", total_tokens * v_width)?;
        let alpha_host = read_f32("alpha.f32", total_tokens * geometry.h_v)?;
        let beta_host = read_f32("beta.f32", total_tokens * geometry.h_v)?;
        let initial = read_f32("initial-state.f32", state_elements)?;
        ensure!(
            initial.iter().all(|value| value.is_finite())
                && initial.iter().any(|&value| value != 0.0)
                && initial[1] != initial[geometry.head_dim],
            "{case}: initial HKV state must be finite, nonzero and asymmetric"
        );
        let mut state = ctx.stream.clone_htod(&initial)?;
        let mut actual_output = Vec::with_capacity(total_tokens * v_width);
        for start in (0..total_tokens).step_by(chunk_tokens) {
            let end = start + chunk_tokens;
            let q = HiddenStates::from_host(
                &ctx,
                &q_host[start * q_width..end * q_width],
                q_width,
                chunk_tokens,
            )?;
            let k = HiddenStates::from_host(
                &ctx,
                &k_host[start * k_width..end * k_width],
                k_width,
                chunk_tokens,
            )?;
            let v = HiddenStates::from_host(
                &ctx,
                &v_host[start * v_width..end * v_width],
                v_width,
                chunk_tokens,
            )?;
            let gates = start * geometry.h_v..end * geometry.h_v;
            let alpha = ctx.stream.clone_htod(&alpha_host[gates.clone()])?;
            let beta = ctx.stream.clone_htod(&beta_host[gates])?;
            let mut output = HiddenStates::zeros(&ctx, v_width, chunk_tokens)?;
            let mut workspace = backend.allocate_workspace(&ctx, chunk_tokens)?;
            let cu_seqlens = ctx.stream.clone_dtoh(&workspace.cu_seqlens)?;
            ctx.sync()?;
            ensure!(
                cu_seqlens == [0, chunk_tokens as i64],
                "{case}: sequence metadata does not match the launched extent"
            );
            if case == "t65" {
                let mut short_q = HiddenStates::from_host(&ctx, &q_host[..q_width], q_width, 1)?;
                short_q.seq_len = chunk_tokens;
                let error = backend
                    .launch_in_place(
                        &ctx,
                        &short_q,
                        &k,
                        &v,
                        &alpha,
                        &beta,
                        &mut state,
                        &mut output,
                        &mut workspace,
                    )
                    .expect_err("logical shape must not exceed the nonempty Q allocation");
                ensure!(
                    error.to_string().contains("Qwen3.5 GDN Q backing len"),
                    "short Q backing failed for the wrong reason: {error}"
                );
            }
            backend.launch_in_place(
                &ctx,
                &q,
                &k,
                &v,
                &alpha,
                &beta,
                &mut state,
                &mut output,
                &mut workspace,
            )?;
            actual_output.extend(output.to_host(&ctx)?);
            if chunk_tokens < total_tokens && start == 0 {
                let first_state = ctx.stream.clone_dtoh(&state)?;
                ctx.sync()?;
                ensure!(
                    first_state != initial,
                    "{case}: first chunk did not advance state"
                );
                compare_layout_result(
                    &format!("{case} first-state HKV"),
                    &read_f32("first-state.f32", state_elements)?,
                    &first_state,
                )?;
            }
        }
        let actual_state = ctx.stream.clone_dtoh(&state)?;
        ctx.sync()?;
        ensure!(
            actual_state != initial,
            "{case}: recurrent state did not update"
        );
        ensure!(
            actual_output.iter().any(|&value| value != 0.0),
            "{case}: output remained zero"
        );
        let expected_output = read_bf16("output.bf16", total_tokens * v_width)?
            .into_iter()
            .map(bf16::to_f32)
            .collect::<Vec<_>>();
        compare_layout_result(&format!("{case} output"), &expected_output, &actual_output)?;
        compare_layout_result(
            &format!("{case} final-state HKV"),
            &read_f32("state.f32", state_elements)?,
            &actual_state,
        )?;
    }
    Ok(())
}
