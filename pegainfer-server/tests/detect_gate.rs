//! The detection gate as the shipped binary runs it.
//!
//! The line's `probe` and the server's `feature_gate_hint` have unit tests, but
//! the wiring between them — `main` reading `config.json`, the registry's
//! `detect`, the hint, and `launch`'s refusal — is only exercised by running the
//! binary. This drives it against a model directory holding the frozen `qwen4_exp`
//! config and asserts the two shipped outcomes: a stock build names the feature to
//! rebuild with, an enabled build detects the line and then refuses to serve it.
//!
//! Neither path reaches CUDA. Detection is pure, and `launch` bails before the
//! engine is built, so this runs on a driver-less runner once the loader can
//! resolve the driver stub the way the CI step sets it up.

use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Output;

/// The real checkpoint config, committed by the line crate. The server has no
/// fixture of its own, and the gate is meant to see the real thing.
const FROZEN_CONFIG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../pegainfer-qwen38-flash-next/tests/frozen_config.json"
));

fn model_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "pegainfer-detect-gate-{tag}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), FROZEN_CONFIG).unwrap();
    dir
}

fn run_against(model_path: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_pegainfer"))
        .arg("--model-path")
        .arg(model_path)
        .output()
        .expect("spawn the pegainfer binary")
}

#[cfg(not(feature = "qwen38-flash-next"))]
#[test]
fn stock_build_names_the_flash_next_feature() {
    let dir = model_dir("stock");
    let out = run_against(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !out.status.success(),
        "the stock build must not claim a qwen4_exp config"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--features qwen38-flash-next"),
        "the hint did not name the feature:\n{stderr}"
    );
}

#[cfg(feature = "qwen38-flash-next")]
#[test]
fn enabled_build_detects_the_line_and_refuses_to_serve() {
    let dir = model_dir("enabled");
    let out = run_against(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !out.status.success(),
        "the line must refuse to serve until the text graph lands"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("Qwen3.8-Flash-Next"),
        "stderr did not name the detected line:\n{stderr}"
    );
    assert!(
        stderr.contains("text graph"),
        "stderr did not carry the refusal reason:\n{stderr}"
    );
}
