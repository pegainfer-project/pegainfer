use serde_json::Value;

use super::*;

// A real generated bundle is the shared fixture: passing this host test is
// contract evidence only. Canonical Gate 1 separately links and launches it.
#[test]
fn candidate_contract_rejects_mutations() {
    let kernel_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../pegainfer-kernels")
        .canonicalize()
        .unwrap();
    let original = PathBuf::from(
        std::env::var_os("PEGAINFER_QWEN35_GDN_AOT_BUNDLE")
            .expect("requires a real generated candidate"),
    );
    let accepted = validate_candidate(&kernel_root, &original)
        .unwrap_or_else(|error| panic!("real candidate fixture rejected: {error}"));
    let temp = tempfile::tempdir().unwrap();
    let bundle = temp.path().join("candidate");
    fs::create_dir(&bundle).unwrap();
    for (name, bytes) in &accepted.files {
        fs::write(bundle.join(name), bytes).unwrap();
    }
    let manifest_path = bundle.join("manifest.json");
    let manifest: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let write_manifest = |value: &Value| {
        fs::write(&manifest_path, serde_json::to_vec(value).unwrap()).unwrap();
    };
    let reject = |label: &str| {
        assert!(
            validate_candidate(&kernel_root, &bundle).is_err(),
            "accepted {label}"
        );
    };

    let cases = [
        ("/target/arch", Value::from("sm_90")),
        ("/geometry/h_v", Value::from(48)),
        ("/dtypes/state", Value::from("bfloat16")),
        ("/tokens/maximum", Value::from(1)),
        ("/abi/version", Value::from(0)),
        ("/abi/symbols", Value::Array(vec![])),
        ("/abi/q_view/stride/0", Value::from(1)),
        ("/abi/state_update", Value::from("out_of_place")),
        ("/workspace/bytes_per_sm", Value::from(0)),
        ("/toolchain/python", Value::from("untrusted")),
        ("/source/kernel_source_sha256", Value::from("untrusted")),
        ("/artifact/format", Value::from("ptx")),
        ("/artifact/object/sha256", Value::from("untrusted")),
    ];
    for (path, value) in &cases {
        let mut changed = manifest.clone();
        *changed.pointer_mut(path).expect("contract field exists") = value.clone();
        write_manifest(&changed);
        reject(path);
    }
    let mut missing = manifest.clone();
    missing.as_object_mut().unwrap().remove("abi");
    write_manifest(&missing);
    reject("missing ABI");
    let mut unknown = manifest.clone();
    unknown
        .as_object_mut()
        .unwrap()
        .insert("unexpected".into(), Value::Bool(true));
    write_manifest(&unknown);
    reject("unknown manifest field");
    write_manifest(&manifest);

    for (name, bytes) in &accepted.files {
        let path = bundle.join(name);
        // File kinds share read_regular; byte integrity is checked for each artifact.
        if *name == "kernel.o" {
            fs::remove_file(&path).unwrap();
            reject("missing kernel.o");
            fs::create_dir(&path).unwrap();
            reject("directory kernel.o");
            fs::remove_dir(&path).unwrap();
            #[cfg(unix)]
            {
                std::os::unix::fs::symlink(original.canonicalize().unwrap().join(name), &path)
                    .unwrap();
                reject("symlink kernel.o");
                fs::remove_file(&path).unwrap();
            }
        }
        if *name != "manifest.json" {
            let mut corrupted = bytes.clone();
            corrupted[0] ^= 1;
            fs::write(&path, &corrupted).unwrap();
            reject(&format!("same-size corruption {name}"));
        }
        // Remove content, not just a trailing newline that is optional in JSON.
        let truncated_len = bytes.trim_ascii_end().len() - 1;
        fs::write(&path, &bytes[..truncated_len]).unwrap();
        reject(&format!("truncated {name}"));
        fs::write(&path, bytes).unwrap();
    }
    assert!(validate_candidate(&kernel_root, &bundle.join("../candidate")).is_err());
    #[cfg(unix)]
    {
        let link = temp.path().join("linked");
        std::os::unix::fs::symlink(&bundle, &link).unwrap();
        assert!(validate_candidate(&kernel_root, &link).is_err());
    }
    validate_candidate(&kernel_root, &bundle).unwrap();
    println!(
        "validated real candidate and rejected {} metadata mutations plus file/path mutations",
        cases.len()
    );
}
