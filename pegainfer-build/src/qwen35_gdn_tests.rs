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

    // Exercise missing/unknown keys at every object boundary and a changed
    // value at every scalar leaf, including every shape and stride component.
    fn mutations(value: &Value, path: &str, result: &mut Vec<(String, Value)>, root: &Value) {
        let mut changed = root.clone();
        match value {
            Value::Object(fields) => {
                changed
                    .pointer_mut(path)
                    .unwrap()
                    .as_object_mut()
                    .unwrap()
                    .insert("unexpected".into(), Value::Bool(true));
                result.push((format!("unknown field at {path}"), changed));
                for (key, child) in fields {
                    let child_path = format!("{path}/{key}");
                    let mut missing = root.clone();
                    missing
                        .pointer_mut(path)
                        .unwrap()
                        .as_object_mut()
                        .unwrap()
                        .remove(key);
                    result.push((format!("missing {child_path}"), missing));
                    mutations(child, &child_path, result, root);
                }
            }
            Value::Array(values) => {
                for (index, child) in values.iter().enumerate() {
                    mutations(child, &format!("{path}/{index}"), result, root);
                }
            }
            _ => {
                *changed.pointer_mut(path).unwrap() = match value {
                    Value::String(text) => Value::String(format!("{text}-changed")),
                    Value::Number(number) => Value::from(number.as_u64().unwrap() + 1),
                    _ => panic!("unexpected manifest scalar: {path}"),
                };
                result.push((format!("changed {path}"), changed));
            }
        }
    }
    let mut cases = Vec::new();
    mutations(&manifest, "", &mut cases, &manifest);
    for (label, changed) in &cases {
        write_manifest(changed);
        reject(label);
    }
    write_manifest(&manifest);

    for (name, bytes) in &accepted.files {
        let path = bundle.join(name);
        fs::remove_file(&path).unwrap();
        reject(&format!("missing {name}"));
        fs::create_dir(&path).unwrap();
        reject(&format!("directory {name}"));
        fs::remove_dir(&path).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(original.canonicalize().unwrap().join(name), &path).unwrap();
            reject(&format!("symlink {name}"));
            fs::remove_file(&path).unwrap();
        }
        let mut corrupted = bytes.clone();
        corrupted[0] ^= 1;
        fs::write(&path, &corrupted).unwrap();
        reject(&format!("same-size corruption {name}"));
        fs::write(&path, &bytes[..bytes.len() - 1]).unwrap();
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
