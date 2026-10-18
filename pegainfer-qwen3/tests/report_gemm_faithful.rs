#![cfg(feature = "kernel-report")]
//! Gate for `qwen3_model_report`: asserts the per-`numeric_policy()` runtime observations the type
//! system can't make — projection GEMMs serve through each policy's own kernel (pin_served /
//! per_token_served). Runs the bin as a subprocess; vacuous unless PEGAINFER_TEST_MODEL_PATH set.

use std::process::Command;

const KV_LEN: usize = 1024;

fn run_report(model_path: &str, policy: &str) -> serde_json::Value {
    let kv_len = KV_LEN.to_string();
    let out = Command::new(env!("CARGO_BIN_EXE_qwen3_model_report"))
        .args([
            "decode",
            "--batch-size",
            "1",
            "--kv-len",
            &kv_len,
            "--iters",
            "8",
            "--policy",
            policy,
            "--format",
            "json",
            "--model-path",
            model_path,
        ])
        .output()
        .expect("spawn qwen3_model_report");
    assert!(
        out.status.success(),
        "report (--policy {policy}) failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("parse report JSON")
}

#[test]
fn report_gemm_faithful_per_policy() {
    let Ok(model_path) = std::env::var("PEGAINFER_TEST_MODEL_PATH") else {
        eprintln!("skipping report_gemm_faithful: set PEGAINFER_TEST_MODEL_PATH to Qwen3-4B-base");
        return;
    };

    let pin = run_report(&model_path, "pin");
    assert!(
        pin["pin_served"].as_u64().expect("pin_served") > 0,
        "Pin must serve projection GEMMs through the pinned algo (pin_served > 0)"
    );

    let per_token = run_report(&model_path, "per-token");
    assert!(
        per_token["per_token_served"]
            .as_u64()
            .expect("per_token_served")
            > 0,
        "PerToken must serve projection GEMMs through the per-token oracle (per_token_served > 0)"
    );
}
