//! The detection gate as the shipped binary runs it.
//!
//! The line's `probe` and the server's `feature_gate_hint` have unit tests, but
//! the wiring between them — `main` reading `config.json`, the registry's
//! `detect`, the hint, and `launch`'s refusal — is only exercised by running the
//! binary. This drives it against a model directory holding the frozen `qwen4_exp`
//! config and asserts the two shipped outcomes: a stock build names the feature to
//! rebuild with, an enabled build detects the line and then refuses to serve it.
//!
//! Neither path reaches CUDA: detection is pure, and `launch` bails before the
//! engine is built.

use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::ExitStatus;
use std::process::Stdio;
use std::time::Duration;
use std::time::Instant;

/// The real checkpoint config, committed by the line crate. The server has no
/// fixture of its own, and the gate is meant to see the real thing.
const FROZEN_CONFIG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../pegainfer-qwen4/tests/frozen_config.json"
));

/// The repo's own minimal tokenizer (`pegainfer-frontend/examples/echo-model`,
/// the fixture `examples/echo-server.rs` serves). An enabled build reaches the
/// serve path before `launch` bails, and the frontend refuses to start without a
/// tokenizer — which would surface *that* error instead of the refusal. The
/// engine failure then cancels the server so the refusal is what reaches stderr.
const TOKENIZER: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../pegainfer-frontend/examples/echo-model/tokenizer.json"
));

fn model_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "pegainfer-detect-gate-{tag}-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), FROZEN_CONFIG).unwrap();
    std::fs::write(dir.join("tokenizer.json"), TOKENIZER).unwrap();
    dir
}

/// Both runs exit within a second once the binary is warm; this only has to
/// catch a startup or cancellation path that never returns.
const RUN_TIMEOUT: Duration = Duration::from_secs(60);

/// Runs the binary against `model_path` and returns its exit status and stderr.
/// Stderr goes to a file, so a full pipe cannot stall the child, and a run past
/// `RUN_TIMEOUT` is killed and fails with what it printed so far.
fn run_against(model_path: &Path) -> (ExitStatus, String) {
    let log = model_path.join("stderr.log");
    let mut child = Command::new(env!("CARGO_BIN_EXE_pegainfer"))
        .arg("--model-path")
        .arg(model_path)
        .stdout(Stdio::null())
        .stderr(File::create(&log).expect("create the stderr log"))
        .spawn()
        .expect("spawn the pegainfer binary");
    let deadline = Instant::now() + RUN_TIMEOUT;
    let status = loop {
        if let Some(status) = child.try_wait().expect("poll the pegainfer binary") {
            break status;
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "the binary did not exit within {RUN_TIMEOUT:?}:\n{}",
                std::fs::read_to_string(&log).unwrap_or_default()
            );
        }
        std::thread::sleep(Duration::from_millis(50));
    };
    let stderr = std::fs::read_to_string(&log).expect("read the stderr log");
    (status, stderr)
}

#[cfg(not(feature = "qwen4"))]
#[test]
fn stock_build_names_the_qwen4_feature() {
    let dir = model_dir("stock");
    let (status, stderr) = run_against(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !status.success(),
        "the stock build must not claim a qwen4_exp config"
    );
    assert!(
        stderr.contains("--features qwen4"),
        "the hint did not name the feature:\n{stderr}"
    );
}

#[cfg(feature = "qwen4")]
#[test]
fn enabled_build_detects_the_line_and_refuses_to_serve() {
    let dir = model_dir("enabled");
    let (status, stderr) = run_against(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !status.success(),
        "the line must refuse to serve until inference is implemented"
    );
    assert!(
        stderr.contains("Qwen4"),
        "stderr did not name the detected line:\n{stderr}"
    );
    assert!(
        stderr.contains("not implemented"),
        "stderr did not carry the refusal reason:\n{stderr}"
    );
}
