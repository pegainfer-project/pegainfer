use pegainfer_frontend::engine::FinishReason;

use super::backend::EpBackendKind;
use super::backend::parse_backend;
use super::backend::validate_backend_and_devices;
use super::helpers::append_generated_token;
use super::helpers::ensure_same_prompt_batch_rows_match;

#[test]
fn append_generated_token_handles_eos_stop_vs_ignore() {
    // EOS hit with ignore_eos=false: stop, do not append the EOS token.
    let mut generated = vec![10, 11];
    let finish_reason = append_generated_token(&mut generated, 100_001, &[100_001], false);
    assert_eq!(finish_reason, Some(FinishReason::Stop));
    assert_eq!(generated, vec![10, 11]);

    // EOS hit with ignore_eos=true: keep going, append the token.
    let mut generated = vec![10, 11];
    let finish_reason = append_generated_token(&mut generated, 100_001, &[100_001], true);
    assert_eq!(finish_reason, None);
    assert_eq!(generated, vec![10, 11, 100_001]);
}

#[test]
fn same_prompt_batch_rows_must_match() {
    ensure_same_prompt_batch_rows_match(&[
        vec![11, 304, 608],
        vec![11, 304, 608],
        vec![11, 304, 608],
    ])
    .unwrap();

    let err =
        ensure_same_prompt_batch_rows_match(&[vec![11, 304, 608], vec![11, 463, 608]]).unwrap_err();
    assert!(
        err.to_string().contains("generated token index 1"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn duplicate_device_ordinals_are_rejected() {
    let err = validate_backend_and_devices(&[0, 0]).unwrap_err();

    assert!(
        err.to_string()
            .contains("two distinct CUDA device ordinals"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn ep_backend_parsing() {
    assert_eq!(parse_backend(None).unwrap(), EpBackendKind::HostStaged);
    assert_eq!(parse_backend(Some("nccl")).unwrap(), EpBackendKind::Nccl);

    let err = parse_backend(Some("pplx")).unwrap_err();
    assert!(
        err.to_string()
            .contains("supported backends: host-staged, nccl"),
        "unexpected error: {err:#}"
    );
}
