use half::bf16;
use pegainfer_core::tensor::DeviceContext;
use pegainfer_core::tensor::DeviceMatrix;
use pegainfer_core::tensor::HiddenStates;
use pegainfer_kernels::ops::Dsv2LiteAttentionConfig;
use pegainfer_kernels::ops::Dsv2LiteRouterOutput;
use pegainfer_kernels::ops::dsv2_lite_accumulate_fixed_expert_into;
use pegainfer_kernels::ops::dsv2_lite_accumulate_route_row_into;
use pegainfer_kernels::ops::dsv2_lite_decode_attention_into;
use pegainfer_kernels::ops::dsv2_lite_kv_norm_into;
use pegainfer_kernels::ops::dsv2_lite_router_logits_into;
use pegainfer_kernels::ops::dsv2_lite_router_softmax_topk_into;

use super::parse_rollback_value;
use crate::config::test_lite_config;
use crate::host_ops::gate_logits_host;
use crate::host_ops::topk_softmax_routes;

#[test]
fn rollback_policy_parser_rejects_unknown_values() {
    for (name, optimized, rollback) in [
        ("host expert batch", "batched", "serial"),
        ("NCCL expert batch", "grouped", "serial"),
        ("NCCL router", "device", "host"),
    ] {
        assert!(!parse_rollback_value(name, None, optimized, rollback).unwrap());
        assert!(!parse_rollback_value(name, Some(""), optimized, rollback).unwrap());
        assert!(!parse_rollback_value(name, Some(optimized), optimized, rollback).unwrap());
        assert!(parse_rollback_value(name, Some(rollback), optimized, rollback).unwrap());
        assert!(parse_rollback_value(name, Some("typo"), optimized, rollback).is_err());
    }
}

#[test]
fn device_router_matches_host_softmax_topk_rule() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let mut config = test_lite_config();
    config.hidden_size = 4;
    config.n_routed_experts = 8;
    config.num_experts_per_token = 6;

    let hidden_host = bf16_vec(&[1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.5]);
    let gate_host = router_gate_fixture();
    let hidden = HiddenStates {
        data: ctx.stream.clone_htod(&hidden_host).expect("hidden H2D"),
        hidden_dim: config.hidden_size,
        seq_len: 2,
    };
    let gate = DeviceMatrix::from_host(
        &ctx,
        &gate_host,
        config.n_routed_experts,
        config.hidden_size,
    )
    .expect("gate H2D");
    let mut topk_weight = ctx
        .stream
        .alloc_zeros::<f32>(hidden.seq_len * config.num_experts_per_token)
        .expect("topk weight");
    let mut topk_idx = ctx
        .stream
        .alloc_zeros::<i32>(hidden.seq_len * config.num_experts_per_token)
        .expect("topk idx");

    dsv2_lite_router_softmax_topk_into(
        &ctx,
        &hidden,
        &gate,
        config.num_experts_per_token,
        &mut Dsv2LiteRouterOutput {
            topk_weight: &mut topk_weight,
            topk_idx: &mut topk_idx,
        },
    )
    .expect("device router");
    let got_idx = ctx.stream.clone_dtoh(&topk_idx).expect("idx D2H");
    let got_weight = ctx.stream.clone_dtoh(&topk_weight).expect("weight D2H");
    ctx.sync().expect("sync router outputs");

    let gate_host_f32: Vec<_> = gate_host.iter().map(|value| value.to_f32()).collect();
    let logits = gate_logits_host(&config, &hidden_host, &gate_host_f32);
    let expected = topk_softmax_routes(&config, &logits, hidden.seq_len);

    for (token, expected_routes) in expected.iter().enumerate().take(hidden.seq_len) {
        for (route, &(expected_idx, expected_weight)) in expected_routes
            .iter()
            .enumerate()
            .take(config.num_experts_per_token)
        {
            let offset = token * config.num_experts_per_token + route;
            assert_eq!(got_idx[offset], expected_idx as i32);
            assert_close(got_weight[offset], expected_weight, 1.0e-6);
        }
    }
}

#[test]
fn device_router_logits_match_host_accumulation_bitwise() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let mut config = test_lite_config();
    config.hidden_size = 4;
    config.n_routed_experts = 8;

    let hidden_host = bf16_vec(&[1.0, -2.0, 0.5, 3.0, -1.0, 0.25, 2.0, -0.5]);
    let gate_host = router_gate_fixture();
    let hidden = HiddenStates {
        data: ctx.stream.clone_htod(&hidden_host).expect("hidden H2D"),
        hidden_dim: config.hidden_size,
        seq_len: 2,
    };
    let gate = DeviceMatrix::from_host(
        &ctx,
        &gate_host,
        config.n_routed_experts,
        config.hidden_size,
    )
    .expect("gate H2D");
    let mut logits = ctx
        .stream
        .alloc_zeros::<f32>(hidden.seq_len * config.n_routed_experts)
        .expect("router logits");

    dsv2_lite_router_logits_into(&ctx, hidden.as_ref(), &gate, &mut logits).expect("device logits");
    let got = ctx.stream.clone_dtoh(&logits).expect("logits D2H");
    ctx.sync().expect("sync logits");

    let gate_host_f32: Vec<_> = gate_host.iter().map(|value| value.to_f32()).collect();
    let expected = gate_logits_host(&config, &hidden_host, &gate_host_f32);
    assert_eq!(got.len(), expected.len());
    for (idx, (got, expected)) in got.iter().zip(expected).enumerate() {
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "router logits mismatch at index {idx}: got={got} expected={expected}"
        );
    }
}

#[test]
fn route_row_accumulation_selects_group_output_row() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let rows = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&[1.0, 2.0, 3.0, 4.0]))
            .expect("rows H2D"),
        hidden_dim: 2,
        seq_len: 2,
    };
    let mut out = ctx.stream.alloc_zeros::<f32>(4).expect("output");

    dsv2_lite_accumulate_route_row_into(&ctx, rows.as_ref(), 1, 0.5, 0, 2, &mut out)
        .expect("accumulate row");

    let actual = ctx.stream.clone_dtoh(&out).expect("output D2H");
    ctx.sync().expect("sync accumulation");
    assert_eq!(actual, vec![1.5, 2.0, 0.0, 0.0]);
}

#[test]
fn router_logits_rejects_inflated_hidden_metadata() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let hidden = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&[1.0, -2.0, 0.5, 3.0]))
            .expect("hidden H2D"),
        hidden_dim: 4,
        seq_len: 4096,
    };
    let gate = DeviceMatrix::from_host(&ctx, &router_gate_fixture(), 8, 4).expect("gate H2D");
    let mut logits = ctx
        .stream
        .alloc_zeros::<f32>(hidden.seq_len * gate.rows)
        .expect("logits");

    let err = dsv2_lite_router_logits_into(&ctx, hidden.as_ref(), &gate, &mut logits)
        .expect_err("inflated hidden metadata must fail before CUDA launch");
    assert_error_contains(&err, "router hidden backing buffer too small");
}

#[test]
fn router_logits_rejects_inflated_gate_metadata() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let hidden = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&[1.0, -2.0, 0.5, 3.0]))
            .expect("hidden H2D"),
        hidden_dim: 4,
        seq_len: 1,
    };
    let gate = DeviceMatrix {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&[0.25, 0.5, -0.25, 1.0]))
            .expect("gate H2D"),
        rows: 8,
        cols: 4,
    };
    let mut logits = ctx
        .stream
        .alloc_zeros::<f32>(hidden.seq_len * gate.rows)
        .expect("logits");

    let err = dsv2_lite_router_logits_into(&ctx, hidden.as_ref(), &gate, &mut logits)
        .expect_err("inflated gate metadata must fail before CUDA launch");
    assert_error_contains(&err, "router gate backing buffer too small");
}

#[test]
fn router_topk_rejects_inflated_hidden_metadata() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let hidden = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&[1.0, -2.0, 0.5, 3.0]))
            .expect("hidden H2D"),
        hidden_dim: 4,
        seq_len: 4096,
    };
    let gate = DeviceMatrix::from_host(&ctx, &router_gate_fixture(), 8, 4).expect("gate H2D");
    let topk = 6;
    let mut topk_weight = ctx
        .stream
        .alloc_zeros::<f32>(hidden.seq_len * topk)
        .expect("topk weight");
    let mut topk_idx = ctx
        .stream
        .alloc_zeros::<i32>(hidden.seq_len * topk)
        .expect("topk idx");

    let err = dsv2_lite_router_softmax_topk_into(
        &ctx,
        &hidden,
        &gate,
        topk,
        &mut Dsv2LiteRouterOutput {
            topk_weight: &mut topk_weight,
            topk_idx: &mut topk_idx,
        },
    )
    .expect_err("inflated hidden metadata must fail before CUDA launch");
    assert_error_contains(&err, "router hidden backing buffer too small");
}

#[test]
fn route_row_accumulation_rejects_inflated_rows_metadata() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let rows = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&[1.0, 2.0]))
            .expect("rows H2D"),
        hidden_dim: 2,
        seq_len: 4096,
    };
    let mut out = ctx
        .stream
        .alloc_zeros::<f32>(rows.hidden_dim * rows.seq_len)
        .expect("output");

    let err = dsv2_lite_accumulate_route_row_into(
        &ctx,
        rows.as_ref(),
        4095,
        0.5,
        0,
        rows.seq_len,
        &mut out,
    )
    .expect_err("inflated rows metadata must fail before CUDA launch");
    assert_error_contains(&err, "route rows backing buffer too small");
}

#[test]
fn fixed_expert_accumulate_rejects_inflated_output_metadata() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let expert_output = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&[1.0, 2.0, 3.0]))
            .expect("expert output H2D"),
        hidden_dim: 3,
        seq_len: 2,
    };
    let topk = 3;
    let topk_weight = ctx
        .stream
        .clone_htod(&[0.25f32, 0.5, 0.25, 0.1, 0.2, 0.7])
        .expect("route weights H2D");
    let topk_idx = ctx
        .stream
        .clone_htod(&[2i32, 5, 7, 1, 5, 6])
        .expect("route idx H2D");
    let mut accum = ctx
        .stream
        .alloc_zeros::<f32>(expert_output.hidden_dim * expert_output.seq_len)
        .expect("accum");

    let err = dsv2_lite_accumulate_fixed_expert_into(
        &ctx,
        &expert_output,
        &topk_weight,
        &topk_idx,
        5,
        topk,
        &mut accum,
    )
    .expect_err("inflated expert output metadata must fail before CUDA launch");
    assert_error_contains(&err, "fixed-expert output backing buffer too small");
}

#[test]
fn kv_norm_rejects_inflated_input_metadata() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let kv_a = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&[1.0, 2.0]))
            .expect("kv_a H2D"),
        hidden_dim: 4,
        seq_len: 2,
    };
    let norm_weight = ctx
        .stream
        .clone_htod(&bf16_vec(&[1.0, 1.0]))
        .expect("norm weight H2D");
    let mut compressed = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&[0.0, 0.0, 0.0, 0.0]))
            .expect("compressed H2D"),
        hidden_dim: 2,
        seq_len: 2,
    };

    let err = dsv2_lite_kv_norm_into(&ctx, &kv_a, &norm_weight, 2, 1.0e-6, &mut compressed)
        .expect_err("inflated kv_a metadata must fail before CUDA launch");
    assert_error_contains(&err, "kv norm kv_a backing buffer too small");
}

#[test]
fn kv_norm_rejects_inflated_output_metadata() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let kv_a = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&[1.0, 2.0, 3.0, 4.0]))
            .expect("kv_a H2D"),
        hidden_dim: 2,
        seq_len: 2,
    };
    let norm_weight = ctx
        .stream
        .clone_htod(&bf16_vec(&[1.0, 1.0]))
        .expect("norm weight H2D");
    let mut compressed = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&[0.0, 0.0]))
            .expect("compressed H2D"),
        hidden_dim: 2,
        seq_len: 2,
    };

    let err = dsv2_lite_kv_norm_into(&ctx, &kv_a, &norm_weight, 2, 1.0e-6, &mut compressed)
        .expect_err("inflated compressed metadata must fail before CUDA launch");
    assert_error_contains(&err, "kv norm compressed backing buffer too small");
}

#[test]
fn decode_attention_rejects_inflated_q_metadata() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let cfg = tiny_attention_config();
    let q = HiddenStates {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&[1.0, 2.0]))
            .expect("q H2D"),
        hidden_dim: 4,
        seq_len: 1,
    };
    let kv_a = hidden_fixture(&ctx, 3);
    let kv_b = hidden_fixture(&ctx, 4);
    let mut out = hidden_fixture(&ctx, 2);
    let mut key_cache = ctx
        .stream
        .alloc_zeros::<f32>(cfg.max_seq_len * cfg.num_heads * 4)
        .expect("key cache");
    let mut value_cache = ctx
        .stream
        .alloc_zeros::<f32>(cfg.max_seq_len * cfg.num_heads * cfg.v_head_dim)
        .expect("value cache");

    let err = dsv2_lite_decode_attention_into(
        &ctx,
        cfg,
        &q,
        &kv_a,
        &kv_b,
        0,
        &mut key_cache,
        &mut value_cache,
        &mut out,
    )
    .expect_err("inflated q metadata must fail before CUDA launch");
    assert_error_contains(&err, "attention q backing buffer too small");
}

#[test]
fn decode_attention_rejects_inflated_output_metadata() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let cfg = tiny_attention_config();
    let q = hidden_fixture(&ctx, 4);
    let kv_a = hidden_fixture(&ctx, 3);
    let kv_b = hidden_fixture(&ctx, 4);
    let mut out = HiddenStates {
        data: ctx.stream.clone_htod(&bf16_vec(&[0.0])).expect("out H2D"),
        hidden_dim: 2,
        seq_len: 1,
    };
    let mut key_cache = ctx
        .stream
        .alloc_zeros::<f32>(cfg.max_seq_len * cfg.num_heads * 4)
        .expect("key cache");
    let mut value_cache = ctx
        .stream
        .alloc_zeros::<f32>(cfg.max_seq_len * cfg.num_heads * cfg.v_head_dim)
        .expect("value cache");

    let err = dsv2_lite_decode_attention_into(
        &ctx,
        cfg,
        &q,
        &kv_a,
        &kv_b,
        0,
        &mut key_cache,
        &mut value_cache,
        &mut out,
    )
    .expect_err("inflated out metadata must fail before CUDA launch");
    assert_error_contains(&err, "attention out backing buffer too small");
}

#[test]
fn device_router_handles_single_zero_decode_row() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let mut config = test_lite_config();
    config.hidden_size = 4;
    config.n_routed_experts = 8;
    config.num_experts_per_token = 6;

    let hidden_host = bf16_vec(&[0.0, 0.0, 0.0, 0.0]);
    let gate_host = router_gate_fixture();
    let hidden = HiddenStates {
        data: ctx.stream.clone_htod(&hidden_host).expect("hidden H2D"),
        hidden_dim: config.hidden_size,
        seq_len: 1,
    };
    let gate = DeviceMatrix::from_host(
        &ctx,
        &gate_host,
        config.n_routed_experts,
        config.hidden_size,
    )
    .expect("gate H2D");
    let mut topk_weight = ctx
        .stream
        .alloc_zeros::<f32>(hidden.seq_len * config.num_experts_per_token)
        .expect("topk weight");
    let mut topk_idx = ctx
        .stream
        .alloc_zeros::<i32>(hidden.seq_len * config.num_experts_per_token)
        .expect("topk idx");

    dsv2_lite_router_softmax_topk_into(
        &ctx,
        &hidden,
        &gate,
        config.num_experts_per_token,
        &mut Dsv2LiteRouterOutput {
            topk_weight: &mut topk_weight,
            topk_idx: &mut topk_idx,
        },
    )
    .expect("device router");
    let got_idx = ctx.stream.clone_dtoh(&topk_idx).expect("idx D2H");
    let got_weight = ctx.stream.clone_dtoh(&topk_weight).expect("weight D2H");
    ctx.sync().expect("sync router outputs");

    let gate_host_f32: Vec<_> = gate_host.iter().map(|value| value.to_f32()).collect();
    let logits = gate_logits_host(&config, &hidden_host, &gate_host_f32);
    let expected = topk_softmax_routes(&config, &logits, hidden.seq_len);
    for route in 0..config.num_experts_per_token {
        let (expected_idx, expected_weight) = expected[0][route];
        assert_eq!(got_idx[route], expected_idx as i32);
        assert_close(got_weight[route], expected_weight, 1.0e-6);
    }
}

#[test]
fn fixed_expert_accumulate_masks_inactive_experts_and_accumulates_matches() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let hidden_dim = 3;
    let seq_len = 2;
    let topk = 3;
    let expert_output = expert_output_fixture(&ctx, hidden_dim, seq_len);
    let topk_weight = ctx
        .stream
        .clone_htod(&[0.25f32, 0.5, 0.25, 0.1, 0.2, 0.7])
        .expect("route weights H2D");
    let topk_idx = ctx
        .stream
        .clone_htod(&[2i32, 5, 7, 1, 5, 6])
        .expect("route idx H2D");
    let mut accum = ctx
        .stream
        .alloc_zeros::<f32>(hidden_dim * seq_len)
        .expect("accum");

    dsv2_lite_accumulate_fixed_expert_into(
        &ctx,
        &expert_output,
        &topk_weight,
        &topk_idx,
        3,
        topk,
        &mut accum,
    )
    .expect("inactive expert accumulate");
    let got = ctx.stream.clone_dtoh(&accum).expect("inactive D2H");
    ctx.sync().expect("sync inactive accumulate");
    assert_eq!(got, vec![0.0; hidden_dim * seq_len]);

    dsv2_lite_accumulate_fixed_expert_into(
        &ctx,
        &expert_output,
        &topk_weight,
        &topk_idx,
        5,
        topk,
        &mut accum,
    )
    .expect("active expert accumulate");
    let got = ctx.stream.clone_dtoh(&accum).expect("active D2H");
    ctx.sync().expect("sync active accumulate");
    assert_vec_close(&got, &[0.5, 1.0, 1.5, 0.8, 1.0, 1.2], 1.0e-6);

    dsv2_lite_accumulate_fixed_expert_into(
        &ctx,
        &expert_output,
        &topk_weight,
        &topk_idx,
        2,
        topk,
        &mut accum,
    )
    .expect("second active expert accumulate");
    let got = ctx.stream.clone_dtoh(&accum).expect("second active D2H");
    ctx.sync().expect("sync second active accumulate");
    assert_vec_close(&got, &[0.75, 1.5, 2.25, 0.8, 1.0, 1.2], 1.0e-6);
}

#[test]
fn fixed_expert_accumulate_ignores_zero_weight_routes() {
    let ctx = DeviceContext::new().expect("create CUDA context");
    let hidden_dim = 3;
    let seq_len = 2;
    let topk = 3;
    let expert_output = expert_output_fixture(&ctx, hidden_dim, seq_len);
    let topk_weight = ctx
        .stream
        .clone_htod(&[1.0f32, 0.0, 0.0, 0.0, 0.5, 0.5])
        .expect("route weights H2D");
    let topk_idx = ctx
        .stream
        .clone_htod(&[4i32, 1, 2, 4, 3, 2])
        .expect("route idx H2D");
    let mut accum = ctx
        .stream
        .alloc_zeros::<f32>(hidden_dim * seq_len)
        .expect("accum");

    dsv2_lite_accumulate_fixed_expert_into(
        &ctx,
        &expert_output,
        &topk_weight,
        &topk_idx,
        4,
        topk,
        &mut accum,
    )
    .expect("all-token fixed expert accumulate");
    let got = ctx.stream.clone_dtoh(&accum).expect("accum D2H");
    ctx.sync().expect("sync accumulate");
    assert_vec_close(&got, &[1.0, 2.0, 3.0, 0.0, 0.0, 0.0], 1.0e-6);
}

fn router_gate_fixture() -> Vec<bf16> {
    bf16_vec(&[
        0.25, 0.5, -0.25, 1.0, -0.5, 0.75, 0.25, -0.25, 1.0, -1.0, 0.5, 0.25, -0.75, 0.5, 1.0,
        -0.5, 0.5, 0.0, -1.0, 0.75, -1.25, 0.25, 0.75, 0.5, 0.0, -0.5, 1.25, -0.75, 0.75, 0.25,
        -0.25, 0.5,
    ])
}

fn expert_output_fixture(ctx: &DeviceContext, hidden_dim: usize, seq_len: usize) -> HiddenStates {
    HiddenStates {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
            .expect("expert output H2D"),
        hidden_dim,
        seq_len,
    }
}

fn hidden_fixture(ctx: &DeviceContext, hidden_dim: usize) -> HiddenStates {
    HiddenStates {
        data: ctx
            .stream
            .clone_htod(&bf16_vec(&vec![0.0; hidden_dim]))
            .expect("hidden H2D"),
        hidden_dim,
        seq_len: 1,
    }
}

fn tiny_attention_config() -> Dsv2LiteAttentionConfig {
    Dsv2LiteAttentionConfig {
        num_heads: 1,
        qk_nope_head_dim: 2,
        qk_rope_head_dim: 2,
        v_head_dim: 2,
        kv_lora_rank: 1,
        max_seq_len: 2,
        rms_norm_eps: 1.0e-6,
        rope_theta: 10_000.0,
        rope_scaling: None,
    }
}

fn bf16_vec(values: &[f32]) -> Vec<bf16> {
    values.iter().copied().map(bf16::from_f32).collect()
}

fn assert_vec_close(got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len());
    for (idx, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() <= tolerance,
            "value mismatch at {idx}: got {got}, expected {expected}"
        );
    }
}

fn assert_close(got: f32, expected: f32, tolerance: f32) {
    assert!(
        (got - expected).abs() <= tolerance,
        "got {got}, expected {expected}, tolerance {tolerance}"
    );
}

fn assert_error_contains(err: &anyhow::Error, needle: &str) {
    let message = err.to_string();
    assert!(
        message.contains(needle),
        "expected error containing {needle:?}, got {message:?}"
    );
}
