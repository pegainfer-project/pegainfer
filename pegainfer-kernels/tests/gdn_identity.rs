//! Exercise the real candidate loader's identity rejection boundary.
//!
//! The production gate links only this test executable with GNU --wrap for
//! the identity query. Without that link option, the negative cases fail.
//! The existing layout gate then loads and launches the unmodified candidate.

use std::ffi::c_char;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering;

use anyhow::Result;
use pegainfer_kernels::ops::Qwen35GdnAot;
use pegainfer_kernels::ops::Qwen35GdnGeometry;
use pegainfer_kernels::tensor::DeviceContext;

static FAULT: AtomicU8 = AtomicU8::new(0);
const INVALID_UTF8: &[u8] = b"\xff\0";
const SHORT: [u8; 64] = {
    let mut bytes = [b'a'; 64];
    bytes[63] = 0;
    bytes
};
const LONG: [u8; 66] = {
    let mut bytes = [b'a'; 66];
    bytes[65] = 0;
    bytes
};
const NON_HEX: [u8; 65] = {
    let mut bytes = [b'a'; 65];
    bytes[0] = b'g';
    bytes[64] = 0;
    bytes
};

// This symbol belongs only to this test executable. Production builds retain
// the original C query and need neither an injection hook nor a second ABI.
#[unsafe(no_mangle)]
extern "C" fn __wrap_pegainfer_qwen35_gdn_artifact_sha256() -> *const c_char {
    match FAULT.load(Ordering::SeqCst) {
        1 => INVALID_UTF8.as_ptr().cast(),
        2 => SHORT.as_ptr().cast(),
        3 => LONG.as_ptr().cast(),
        4 => NON_HEX.as_ptr().cast(),
        _ => std::ptr::null(),
    }
}

#[test]
#[ignore = "requires SM120, a linked candidate, and the gate's identity-query --wrap link option"]
fn production_loader_rejects_invalid_artifact_identity() -> Result<()> {
    let ctx = DeviceContext::new()?;
    for (fault, label, expected) in [
        (0, "null", "artifact identity pointer is null"),
        (1, "invalid-utf8", "artifact identity is not valid UTF-8"),
        (
            2,
            "short",
            "artifact identity must contain exactly 64 hexadecimal characters",
        ),
        (
            3,
            "long",
            "artifact identity must contain exactly 64 hexadecimal characters",
        ),
        (
            4,
            "non-hex",
            "artifact identity contains non-hexadecimal characters",
        ),
    ] {
        FAULT.store(fault, Ordering::SeqCst);
        let error = Qwen35GdnAot::load_for_production(&ctx, Qwen35GdnGeometry::PRODUCTION)
            .expect_err("the real candidate loader must reject the injected identity");
        assert!(
            error.to_string().contains(expected),
            "{label} failed for the wrong reason: {error:#}; expected {expected}"
        );
        eprintln!("identity_rejection_passed={label}");
    }
    eprintln!("identity_rejections_passed=5");
    Ok(())
}
