//! Native HIP compilation, independent of CUDA discovery and host deployment.
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

fn library(root: &Path, name: &str) -> PathBuf {
    for subdir in ["lib", "lib64"] {
        let directory = root.join(subdir);
        let unversioned = directory.join(format!("lib{name}.so"));
        if unversioned.is_file() {
            return unversioned.canonicalize().expect("resolve ROCm library");
        }
        let mut versions = std::fs::read_dir(directory)
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_file()
                    && path
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                        .starts_with(&format!("lib{name}.so."))
            })
            .collect::<Vec<_>>();
        versions.sort();
        if let Some(path) = versions.pop() {
            return path.canonicalize().expect("resolve ROCm library");
        }
    }
    panic!(
        "missing lib{name}.so under {}; install ROCm libraries/devel or set PEGAINFER_ROCM_CORE/PEGAINFER_ROCM_LIBRARIES",
        root.display()
    );
}

pub fn compile(name: &str, sources: &[&str], includes: &[&str]) {
    for key in [
        "ROCM_PATH",
        "PEGAINFER_HIPCC",
        "PEGAINFER_ROCM_CORE",
        "PEGAINFER_ROCM_LIBRARIES",
        "PEGAINFER_ROCM_ARCH",
        "PEGAINFER_ROCM_DEV_INCLUDE",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let root = PathBuf::from(std::env::var_os("ROCM_PATH").unwrap_or_else(|| "/opt/rocm".into()));
    let core = std::env::var_os("PEGAINFER_ROCM_CORE").map_or_else(|| root.clone(), PathBuf::from);
    let libraries =
        std::env::var_os("PEGAINFER_ROCM_LIBRARIES").map_or_else(|| root.clone(), PathBuf::from);
    let development = std::env::var_os("PEGAINFER_ROCM_DEV_INCLUDE")
        .map_or_else(|| root.join("include"), PathBuf::from);
    let compiler = std::env::var_os("PEGAINFER_HIPCC")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let bundled = core.join("bin/hipcc");
            if bundled.is_file() {
                bundled
            } else {
                PathBuf::from("hipcc")
            }
        });
    let targets = std::env::var("PEGAINFER_ROCM_ARCH")
        .expect("set PEGAINFER_ROCM_ARCH, e.g. gfx1100 or gfx1201; cross-compilation needs no GPU");
    for target in targets.split(',') {
        assert!(
            matches!(
                target,
                "gfx1100" | "gfx1101" | "gfx1102" | "gfx1200" | "gfx1201"
            ),
            "unsupported target {target}"
        );
    }
    println!("cargo:rustc-env=PEGAINFER_BUILT_ROCM_ARCH={targets}");
    let bitcode = [
        core.join("lib/llvm/amdgcn/bitcode"),
        core.join("amdgcn/bitcode"),
        root.join("lib/llvm/amdgcn/bitcode"),
    ]
    .into_iter()
    .find(|path| path.is_dir());
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let mut objects = Vec::new();
    for (index, source) in sources.iter().enumerate() {
        println!("cargo:rerun-if-changed={source}");
        let object = out.join(format!("{name}_{index}.o"));
        let mut cmd = Command::new(&compiler);
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
        cmd.arg(format!("--hip-path={}", core.display()));
        if let Some(path) = &bitcode {
            cmd.arg(format!("--rocm-device-lib-path={}", path.display()));
        }
        for path in [
            core.join("include"),
            libraries.join("include"),
            development.clone(),
        ] {
            cmd.arg("-I").arg(path);
        }
        for target in targets.split(',') {
            cmd.arg(format!("--offload-arch={target}"));
        }
        for include in includes {
            cmd.arg("-I").arg(include);
        }
        assert!(
            cmd.status()
                .expect("launch hipcc (install ROCm devel)")
                .success(),
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
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static={name}");
    // Standard SDKs ship .so aliases; wheel runtime packages may only ship
    // SONAMEs. Keep linker aliases local to the build in either layout.
    for (name, source) in [
        ("amdhip64", library(&core, "amdhip64")),
        ("rocblas", library(&libraries, "rocblas")),
    ] {
        let dest = out.join(format!("lib{name}.so"));
        let _ = std::fs::remove_file(&dest);
        std::os::unix::fs::symlink(&source, dest).expect("create linker alias");
        println!(
            "cargo:rustc-link-search=native={}",
            source.parent().unwrap().display()
        );
        println!("cargo:rustc-link-lib=dylib={name}");
    }
    println!("cargo:rustc-link-lib=dylib=stdc++");
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resolves_standard_and_versioned_wheel_libraries() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir(temp.path().join("lib")).unwrap();
        let versioned = temp.path().join("lib/librocblas.so.5");
        std::fs::write(&versioned, b"library fixture").unwrap();
        assert_eq!(library(temp.path(), "rocblas"), versioned);
        std::os::unix::fs::symlink(&versioned, temp.path().join("lib/librocblas.so")).unwrap();
        assert_eq!(library(temp.path(), "rocblas"), versioned);
    }
    #[test]
    fn resolves_lib64_sdk_layout() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::create_dir(temp.path().join("lib64")).unwrap();
        let path = temp.path().join("lib64/libamdhip64.so");
        std::fs::write(&path, b"library fixture").unwrap();
        assert_eq!(library(temp.path(), "amdhip64"), path);
    }
}
