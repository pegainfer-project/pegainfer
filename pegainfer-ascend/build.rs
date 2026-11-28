//! Link the CANN runtime for `ascend` builds.
//!
//! The crate itself declares the FFI by hand, so `cargo check` works without a
//! CANN install. Link flags are emitted only when the feature is on and
//! `ASCEND_HOME_PATH` (or the default install path) exists.
fn main() {
    println!("cargo:rerun-if-env-changed=ASCEND_HOME_PATH");
    if std::env::var_os("CARGO_FEATURE_ASCEND").is_none() {
        return;
    }
    let home = std::env::var("ASCEND_HOME_PATH")
        .map(std::path::PathBuf::from)
        .ok()
        .filter(|p| p.join("aarch64-linux/lib64").is_dir())
        .or_else(|| {
            let default = std::path::PathBuf::from("/usr/local/Ascend/ascend-toolkit/latest");
            default
                .join("aarch64-linux/lib64")
                .is_dir()
                .then_some(default)
        });
    let Some(home) = home else {
        println!(
            "cargo:warning=ASCEND_HOME_PATH not set and no default CANN install found; \
             skipping link flags (cargo check still works)"
        );
        return;
    };
    let lib = home.join("aarch64-linux/lib64");
    println!("cargo:rustc-link-search=native={}", lib.display());
    for name in ["ascendcl", "opapi", "nnopbase"] {
        println!("cargo:rustc-link-lib=dylib={name}");
    }
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
}
