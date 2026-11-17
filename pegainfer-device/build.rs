fn main() {
    if std::env::var_os("CARGO_FEATURE_ROCM").is_none() {
        return;
    }
    println!("cargo:rerun-if-changed=csrc/runtime.h");
    pegainfer_build::rocm::compile("pegainfer_hip_device", &["csrc/runtime.cpp"], &["csrc"]);
}
