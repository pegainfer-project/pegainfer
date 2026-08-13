//! Builder-boundary tests for unsupported `--batch-invariant` combinations. They call the public
//! `start_engine_*` entry points, so they also cover guard wiring.

use std::path::Path;
use std::sync::Mutex;

use pegainfer_frontend::engine::EngineLoadOptions;
use pegainfer_kernels::ops::NumericPolicy;
use pegainfer_kernels::ops::numeric_policy;
use pegainfer_kernels::ops::set_numeric_policy;
use pegainfer_qwen3::DEFAULT_MAX_PREFILL_TOKENS;
use pegainfer_qwen3::DecodeOverlap;
use pegainfer_qwen3::Qwen3LoraOptions;
use pegainfer_qwen3::Qwen3MemoryOptions;
use pegainfer_qwen3::Qwen3OffloadOptions;
use pegainfer_qwen3::start_engine_with_lora_control;
use pegainfer_qwen3::start_engine_with_offload;

// Serialize the reject #[test]s — they share the process-global numeric policy.
static POLICY_LOCK: Mutex<()> = Mutex::new(());

#[test]
fn batch_invariant_rejects_decode_overlap() {
    let _g = POLICY_LOCK.lock().unwrap();
    set_numeric_policy(NumericPolicy::Tuned);
    let err = start_engine_with_offload(
        Path::new("/nonexistent-model"),
        EngineLoadOptions::default(),
        Qwen3OffloadOptions::disabled(),
        true,
        DEFAULT_MAX_PREFILL_TOKENS,
        Qwen3MemoryOptions::default(),
        DecodeOverlap::SharedSm,
        true,
        None,
    )
    .err()
    .expect("--batch-invariant + --decode-overlap must be rejected");
    assert!(
        format!("{err}").contains("decode-overlap"),
        "unexpected error: {err}"
    );
    assert_eq!(
        numeric_policy(),
        NumericPolicy::Tuned,
        "guard must reject before apply_batch_invariant_policy — global policy was polluted to Pin"
    );
}

#[test]
fn batch_invariant_rejects_dflash() {
    let _g = POLICY_LOCK.lock().unwrap();
    set_numeric_policy(NumericPolicy::Tuned);
    let err = start_engine_with_offload(
        Path::new("/nonexistent-model"),
        EngineLoadOptions::default(),
        Qwen3OffloadOptions::disabled(),
        true,
        DEFAULT_MAX_PREFILL_TOKENS,
        Qwen3MemoryOptions::default(),
        DecodeOverlap::Off,
        true,
        Some(Path::new("/nonexistent-draft")),
    )
    .err()
    .expect("--batch-invariant + DFlash must be rejected");
    assert!(
        format!("{err}").contains("DFlash"),
        "unexpected error: {err}"
    );
    assert_eq!(
        numeric_policy(),
        NumericPolicy::Tuned,
        "guard must reject before apply_batch_invariant_policy — global policy was polluted to Pin"
    );
}

#[test]
fn batch_invariant_rejects_prefix_cache() {
    let _g = POLICY_LOCK.lock().unwrap();
    set_numeric_policy(NumericPolicy::Tuned);
    let err = start_engine_with_offload(
        Path::new("/nonexistent-model"),
        EngineLoadOptions::default(),
        Qwen3OffloadOptions::disabled(),
        false,
        DEFAULT_MAX_PREFILL_TOKENS,
        Qwen3MemoryOptions::default(),
        DecodeOverlap::Off,
        true,
        None,
    )
    .err()
    .expect("--batch-invariant with the prefix cache on must be rejected");
    assert!(
        format!("{err}").contains("--no-prefix-cache"),
        "unexpected error: {err}"
    );
    assert_eq!(
        numeric_policy(),
        NumericPolicy::Tuned,
        "guard must reject before apply_batch_invariant_policy — global policy was polluted to Pin"
    );
}

#[test]
fn batch_invariant_rejects_kv_offload() {
    let _g = POLICY_LOCK.lock().unwrap();
    set_numeric_policy(NumericPolicy::Tuned);
    let err = start_engine_with_offload(
        Path::new("/nonexistent-model"),
        EngineLoadOptions::default(),
        Qwen3OffloadOptions::enabled(1 << 30),
        true,
        DEFAULT_MAX_PREFILL_TOKENS,
        Qwen3MemoryOptions::default(),
        DecodeOverlap::Off,
        true,
        None,
    )
    .err()
    .expect("--batch-invariant + KV offload must be rejected");
    assert!(
        format!("{err}").contains("KV offload"),
        "unexpected error: {err}"
    );
    assert_eq!(
        numeric_policy(),
        NumericPolicy::Tuned,
        "guard must reject before apply_batch_invariant_policy — global policy was polluted to Pin"
    );
}

#[test]
fn batch_invariant_rejects_lora() {
    let _g = POLICY_LOCK.lock().unwrap();
    set_numeric_policy(NumericPolicy::Tuned);
    let err = start_engine_with_lora_control(
        Path::new("/nonexistent-model"),
        EngineLoadOptions::default(),
        Qwen3LoraOptions::default(),
        Qwen3OffloadOptions::disabled(),
        true,
        DEFAULT_MAX_PREFILL_TOKENS,
        Qwen3MemoryOptions::default(),
        DecodeOverlap::Off,
        true,
    )
    .err()
    .expect("--batch-invariant + LoRA must be rejected");
    assert!(format!("{err}").contains("LoRA"), "unexpected error: {err}");
    assert_eq!(
        numeric_policy(),
        NumericPolicy::Tuned,
        "guard must reject before apply_batch_invariant_policy — global policy was polluted to Pin"
    );
}
