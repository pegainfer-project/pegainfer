//! Real Rust-wrapper checks and isolated device-contract violations.
//! GPU tests are manual; CI builds these without creating a HIP device.
use anyhow::Result;
use half::bf16;
use pegainfer_device::Device;
use pegainfer_kernels::rocm::AttentionShape;
use pegainfer_kernels::rocm::{self as ops};

#[test]
#[ignore = "requires an explicitly selected ROCm GPU"]
fn wrappers_reject_invalid_buffers_geometry_and_scalars() -> Result<()> {
    let device = Device::new(0)?;
    let foreign = Device::new(0)?;
    let x = device.zeros::<bf16>(128)?;
    let weight = device.zeros::<bf16>(128)?;
    let mut out = device.zeros::<bf16>(128)?;
    let other = foreign.zeros::<bf16>(128)?;
    let short = device.zeros::<bf16>(127)?;
    for invalid in [&other, &short] {
        assert!(ops::norm(&device, invalid, &weight, &mut out, 1, 128, 1e-6).is_err());
    }
    for eps in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(ops::norm(&device, &x, &weight, &mut out, 1, 128, eps).is_err());
    }
    assert!(ops::gemm(&device, &weight, &x, &mut out, 2, 128, 128).is_err());
    let positions = device.from_slice(&[0i32])?;
    let mut query = device.zeros::<bf16>(128)?;
    assert!(ops::rope(&device, &mut query, &positions, 1, 1, 127, 1e6).is_err());
    for theta in [0., f32::NAN] {
        assert!(ops::rope(&device, &mut query, &positions, 1, 1, 128, theta).is_err());
    }
    let shape = AttentionShape {
        rows: 1,
        heads: 1,
        kv_heads: 2,
        dim: 128,
        context: 1,
        capacity: 1,
    };
    let cache = device.zeros::<bf16>(256)?;
    assert!(
        ops::attention(
            &device, &query, &cache, &cache, &mut out, &positions, &positions, shape
        )
        .is_err()
    );
    let shape = AttentionShape {
        context: 8193,
        kv_heads: 1,
        ..shape
    };
    let cache = device.zeros::<bf16>(8193 * 128)?;
    assert!(
        ops::attention(
            &device, &query, &cache, &cache, &mut out, &positions, &positions, shape
        )
        .is_err()
    );
    // Host validation must not launch a broken kernel or poison the stream.
    ops::norm(&device, &x, &weight, &mut out, 1, 128, 1e-6)?;
    device.synchronize()?;
    Ok(())
}

#[test]
#[ignore = "requires ROCm; launches a fresh process for each sticky device trap"]
fn invalid_device_indices_trap_in_isolated_processes() -> Result<()> {
    use std::os::unix::process::ExitStatusExt;
    for case in [
        "token",
        "cache_slot",
        "cache_position",
        "attention_slot",
        "attention_position",
    ] {
        let directory = tempfile::tempdir()?;
        let mut child = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "device_trap_child", "--ignored", "--nocapture"])
            .env("PEGAINFER_TRAP_CASE", case)
            .env("HSA_ENABLE_COREDUMP", "0")
            .current_dir(directory.path())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while child.try_wait()?.is_none() {
            if std::time::Instant::now() >= deadline {
                child.kill()?;
                let _ = child.wait();
                anyhow::bail!("{case}: device-trap child timed out");
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        let output = child.wait_with_output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stdout.contains("trap-armed"),
            "{case}: setup failed: {stdout} {stderr}"
        );
        assert!(
            (output.status.success() && stdout.contains("trap-observed"))
                || matches!(output.status.signal(), Some(4 | 6 | 11)),
            "{case}: device violation did not trap: {:?} {stdout} {stderr}",
            output.status
        );
    }
    Ok(())
}

#[test]
#[ignore = "subprocess entry point; not a standalone gate"]
#[allow(
    clippy::exit,
    reason = "a sticky GPU trap requires process teardown without HIP destructors"
)]
fn device_trap_child() -> Result<()> {
    use std::io::Write;
    let Ok(case) = std::env::var("PEGAINFER_TRAP_CASE") else {
        return Ok(());
    };
    let device = Device::new(0)?;
    let x = device.zeros::<bf16>(128)?;
    let mut output = device.zeros::<bf16>(128)?;
    let mut keys = device.zeros::<bf16>(128)?;
    let mut values = device.zeros::<bf16>(128)?;
    let ids = device.from_slice(&[1u32])?; // vocab is one, so this is invalid.
    let slots = device.from_slice(&[i32::from(case.ends_with("slot"))])?;
    let positions = device.from_slice(&[if case.ends_with("position") { -1i32 } else { 0 }])?;
    let shape = AttentionShape {
        rows: 1,
        heads: 1,
        kv_heads: 1,
        dim: 128,
        context: 1,
        capacity: 1,
    };
    println!("trap-armed");
    std::io::stdout().flush()?;
    let result = if case == "token" {
        ops::embedding(&device, &x, &ids, &mut output, 1, 128, 1)
    } else if case.starts_with("cache") {
        ops::cache(
            &device,
            &x,
            &x,
            &mut keys,
            &mut values,
            &slots,
            &positions,
            shape,
        )
    } else {
        ops::attention(
            &device,
            &x,
            &keys,
            &values,
            &mut output,
            &slots,
            &positions,
            shape,
        )
    };
    if result.and_then(|()| device.synchronize()).is_err() {
        println!("trap-observed");
        // Skip destruction against a poisoned context; process exit owns cleanup.
        std::process::exit(0);
    }
    panic!("invalid metadata completed without a device trap");
}
