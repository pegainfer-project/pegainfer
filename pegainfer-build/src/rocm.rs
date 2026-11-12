//! Native HIP compilation, deliberately independent of CUDA discovery.
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

pub fn compile(name: &str, sources: &[&str], includes: &[&str]) {
    for key in [
        "PEGAINFER_HIPCC",
        "PEGAINFER_ROCM_CORE",
        "PEGAINFER_ROCM_LIBRARIES",
        "PEGAINFER_ROCM_ARCH",
        "PEGAINFER_ROCM_DEV_INCLUDE",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let core = PathBuf::from(
        std::env::var("PEGAINFER_ROCM_CORE")
            .expect("set PEGAINFER_ROCM_CORE to the locked ROCm SDK"),
    );
    let libraries = PathBuf::from(
        std::env::var("PEGAINFER_ROCM_LIBRARIES")
            .expect("set PEGAINFER_ROCM_LIBRARIES to the locked ROCm libraries"),
    );
    let targets =
        std::env::var("PEGAINFER_ROCM_ARCH").expect("set PEGAINFER_ROCM_ARCH (gfx1100 or gfx1201)");
    println!("cargo:rustc-env=PEGAINFER_BUILT_ROCM_ARCH={targets}");
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let mut objects = Vec::new();
    for (i, source) in sources.iter().enumerate() {
        println!("cargo:rerun-if-changed={source}");
        let object = out.join(format!("{name}_{i}.o"));
        let mut cmd =
            Command::new(std::env::var("PEGAINFER_HIPCC").unwrap_or_else(|_| "hipcc".into()));
        cmd.args([
            "-x",
            "hip",
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-c",
            source,
            "-o",
        ])
        .arg(&object);
        cmd.arg(format!(
            "--rocm-device-lib-path={}",
            core.join("lib/llvm/amdgcn/bitcode").display()
        ));
        cmd.arg(format!("--hip-path={}", core.display()));
        cmd.arg("-I")
            .arg(core.join("include"))
            .arg("-I")
            .arg(libraries.join("include"));
        cmd.arg("-I").arg(
            std::env::var("PEGAINFER_ROCM_DEV_INCLUDE")
                .unwrap_or_else(|_| "/opt/rocm/core-10.1/include".into()),
        );
        for target in targets.split(',') {
            assert!(
                matches!(
                    target,
                    "gfx1100" | "gfx1101" | "gfx1102" | "gfx1200" | "gfx1201"
                ),
                "unsupported target {target}"
            );
            cmd.arg(format!("--offload-arch={target}"));
        }
        for include in includes {
            cmd.arg("-I").arg(include);
        }
        assert!(
            cmd.status().expect("launch hipcc").success(),
            "HIP compilation failed: {source}"
        );
        objects.push(object);
    }
    let archive = out.join(format!("lib{name}.a"));
    let _ = std::fs::remove_file(&archive);
    assert!(
        Command::new("ar")
            .arg("crs")
            .arg(&archive)
            .args(&objects)
            .status()
            .unwrap()
            .success()
    );
    // Runtime wheels carry SONAMEs but no development .so symlinks.
    for (name, source) in [
        ("libamdhip64.so", core.join("lib/libamdhip64.so.7")),
        ("librocblas.so", libraries.join("lib/librocblas.so.5")),
    ] {
        assert!(
            source.is_file(),
            "missing locked runtime {}",
            source.display()
        );
        let dest = out.join(name);
        let _ = std::fs::remove_file(&dest);
        std::os::unix::fs::symlink(source, dest).expect("create project-local linker alias");
    }
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static={name}");
    for root in [&libraries, &core] {
        let lib = root.join("lib");
        assert!(
            Path::new(&lib).is_dir(),
            "missing ROCm library directory {}",
            lib.display()
        );
        println!("cargo:rustc-link-search=native={}", lib.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    }
    for lib in ["amdhip64", "rocblas", "stdc++"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}
