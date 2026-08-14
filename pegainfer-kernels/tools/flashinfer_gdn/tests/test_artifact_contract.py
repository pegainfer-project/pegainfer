from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import artifact_contract as contract


def source_metadata() -> dict:
    return {
        "flashinfer_commit": contract.FROZEN_FLASHINFER_COMMIT,
        "kernel_source_sha256": "a" * 64,
        "generator_sha256": contract.sha256_file(contract.compiler_path()),
        "requirements_lock_sha256": contract.sha256_file(contract.requirements_lock_path()),
        "workspace": {
            "kind": "per_sm",
            "bytes_per_sm": 128,
            "alignment_bytes": 128,
            "formula": "sm_count * bytes_per_sm",
            "source": contract.WORKSPACE_SOURCE,
        },
        "target_arch": contract.TARGET_ARCH,
    }


def compile_metadata(variant: str, header: Path, obj: Path) -> dict:
    runtime = obj.parent / "libcuda_dialect_runtime_static.a"
    if not runtime.exists():
        runtime.write_bytes(b"!<arch>\n-test-static-runtime")
    return {
        **contract.expected_spec(variant),
        "flashinfer_commit": contract.FROZEN_FLASHINFER_COMMIT,
        "kernel_source_sha256": "a" * 64,
        "generator_sha256": contract.sha256_file(contract.compiler_path()),
        "requirements_lock_sha256": contract.sha256_file(contract.requirements_lock_path()),
        "workspace": source_metadata()["workspace"],
        "toolchain": dict(contract.PINNED_TOOLCHAIN),
        "aot": {
            "function_prefix": f"pegainfer_qwen35_gdn_{variant}",
            "header": header.name,
            "header_sha256": contract.sha256_file(header),
            "object": obj.name,
            "object_sha256": contract.sha256_file(obj),
            "object_size_bytes": obj.stat().st_size,
            "native_runtime": str(runtime),
            "native_runtime_sha256": contract.sha256_file(runtime),
            "native_runtime_size_bytes": runtime.stat().st_size,
        },
    }


class ArtifactContractTests(unittest.TestCase):
    def package(self, root: Path, variant: str) -> Path:
        raw = root / "raw" / variant
        raw.mkdir(parents=True)
        header = raw / f"pegainfer_qwen35_gdn_{variant}.h"
        obj = raw / f"pegainfer_qwen35_gdn_{variant}.o"
        header.write_text("/* generated test header */\n", encoding="utf-8")
        obj.write_bytes(b"\x7fELF-test-object")
        metadata = raw / "metadata.json"
        contract.write_json(metadata, compile_metadata(variant, header, obj))
        with mock.patch.object(contract, "verify_flashinfer_source", return_value=source_metadata()):
            return contract.package_variant(
                variant=variant,
                raw_aot_dir=raw,
                compile_metadata_path=metadata,
                output_dir=root / "bundle" / variant,
                flashinfer_dir=root,
            )

    def test_production_geometry_and_dynamic_t_package(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifests = [contract.read_json(self.package(root, variant)) for variant in contract.SUPPORTED_GEOMETRIES]
            self.assertEqual({m["geometry"]["h_v"] for m in manifests}, {32})
            self.assertTrue(all(m["tokens"] == {"extent": "dynamic", "minimum": 1, "divisibility": 1} for m in manifests))
            self.assertTrue(all(m["abi"]["geometry_binding"] == "stable_project_c_wrapper" for m in manifests))
            self.assertTrue(all(m["distribution"]["cute_runtime_linkage"] == "static" for m in manifests))
            self.assertTrue(all(not m["distribution"]["cuda_driver_jit_required"] for m in manifests))
            self.assertTrue(all(m["distribution"]["production_eligible"] for m in manifests))

    def test_compile_metadata_mismatches_fail(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            raw = Path(name)
            header = raw / "kernel.h"
            obj = raw / "kernel.o"
            header.write_bytes(b"header")
            obj.write_bytes(b"object")
            source = source_metadata()
            for label, key, value in (
                ("SHA", "flashinfer_commit", "0" * 40),
                ("SM", "target_arch", "sm_100a"),
                ("dtype", "dtypes", {**contract.DTYPES, "q": "float16"}),
                ("geometry", "geometry", {"h_q": 16, "h_k": 16, "h_v": 31, "head_dim": 128}),
            ):
                with self.subTest(label=label):
                    metadata = compile_metadata("qwen35_4b_candidate", header, obj)
                    metadata[key] = value
                    with self.assertRaises(contract.ContractError):
                        contract.validate_compile_metadata(metadata, "qwen35_4b_candidate", source)

    def test_manifest_patch_and_object_hash_mismatches_fail(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest_path = self.package(root, "qwen35_4b_candidate")
            original = contract.read_json(manifest_path)
            for label, mutate in (
                ("patch", lambda m: m["source"].__setitem__("patch_set_sha256", "0" * 64)),
                ("object hash", lambda m: m["artifact"]["object"].__setitem__("sha256", "0" * 64)),
            ):
                with self.subTest(label=label):
                    manifest = copy.deepcopy(original)
                    mutate(manifest)
                    contract.write_json(manifest_path, manifest)
                    with self.assertRaises(contract.ContractError):
                        contract.validate_manifest(manifest_path)

    def test_packaging_is_reproducible_across_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as first_name, tempfile.TemporaryDirectory() as second_name:
            first = Path(first_name)
            second = Path(second_name)
            self.assertEqual(self.package(first, "qwen35_4b_candidate").read_bytes(), self.package(second, "qwen35_4b_candidate").read_bytes())
            self.assertEqual((first / "bundle/qwen35_4b_candidate/kernel.o").read_bytes(), (second / "bundle/qwen35_4b_candidate/kernel.o").read_bytes())

    def test_source_lock_records_hkv_and_export_patches(self) -> None:
        lock, digest = contract.load_source_lock()
        self.assertTrue(lock["hkv_state_index_patch"]["applied"])
        self.assertEqual(lock["hkv_state_index_patch"]["ordered_layout"], [1, 0, 2, 3])
        self.assertEqual(lock["aot_export_patch"]["grid_x"], "cutlass.Int32")
        self.assertEqual(len(digest), 64)

    def test_bundle_index_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = root / "bundle"
            for variant in contract.SUPPORTED_GEOMETRIES:
                self.package(root, variant)
            index = {
                "schema_version": contract.SCHEMA_VERSION,
                "variants": {
                    variant: {
                        "manifest": f"{variant}/manifest.json",
                        "manifest_sha256": contract.sha256_file(bundle / variant / "manifest.json"),
                    }
                    for variant in sorted(contract.SUPPORTED_GEOMETRIES)
                },
            }
            contract.write_json(bundle / "bundle.json", index)
            with mock.patch.object(contract, "verify_flashinfer_source", return_value=source_metadata()):
                contract.validate_bundle(bundle, flashinfer_dir=root)
                index["variants"]["qwen35_4b_candidate"]["manifest_sha256"] = "0" * 64
                contract.write_json(bundle / "bundle.json", index)
                with self.assertRaisesRegex(contract.ContractError, "bundle manifest index"):
                    contract.validate_bundle(bundle, flashinfer_dir=root)


if __name__ == "__main__":
    unittest.main()
