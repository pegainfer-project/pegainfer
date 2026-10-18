use half::bf16;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::HiddenStates;

use super::*;

#[test]
fn dense_mlp_preallocated_matches_per_token_path() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let hidden_dim = 4;
    let intermediate = 3;
    let seq_len = 2;
    let gate_up_host = bf16_vec(&[
        0.25, -0.5, 0.75, 1.0, -0.25, 0.5, 1.25, -1.5, 0.75, 0.25, -0.5, 0.5, 1.0, -0.75, 0.25,
        -0.25, -1.0, 0.5, 0.75, 0.25, 0.5, -1.25, 1.5, -0.5,
    ]);
    let down_host = bf16_vec(&[
        0.5, -0.25, 1.0, -0.75, 0.25, 0.5, -0.5, 1.25, 0.75, -1.0, 0.5, 0.25,
    ]);
    let input_host = bf16_vec(&[1.0, -0.5, 0.25, 2.0, -1.0, 0.75, 1.5, -0.25]);
    let mlp = DenseMlp {
        gate_up_proj: DeviceMatrix::from_host(&ctx, &gate_up_host, intermediate * 2, hidden_dim)
            .expect("gate_up H2D"),
        down_proj: DeviceMatrix::from_host(&ctx, &down_host, hidden_dim, intermediate)
            .expect("down H2D"),
    };
    let input = HiddenStates {
        data: ctx.stream.clone_htod(&input_host).expect("input H2D"),
        hidden_dim,
        seq_len,
    };

    let expected = dense_mlp_forward_per_token(&ctx, &mlp, &input).expect("per-token MLP");
    let mut scratch =
        DenseMlpForwardScratch::new(&ctx, &mlp, seq_len).expect("preallocated scratch");
    dense_mlp_forward_preallocated_into(&ctx, &mlp, &input, &mut scratch)
        .expect("preallocated MLP");

    let expected_host = ctx.stream.clone_dtoh(&expected.data).expect("expected D2H");
    let got_host = ctx.stream.clone_dtoh(&scratch.out.data).expect("got D2H");
    ctx.sync().expect("sync MLP outputs");
    assert_bf16_vec_close(&got_host, &expected_host, 2.0e-2);
}

fn bf16_vec(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

fn assert_bf16_vec_close(got: &[bf16], expected: &[bf16], tolerance: f32) {
    assert_eq!(got.len(), expected.len());
    for (idx, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        let got = got.to_f32();
        let expected = expected.to_f32();
        assert!(
            (got - expected).abs() <= tolerance,
            "value mismatch at {idx}: got {got}, expected {expected}"
        );
    }
}
