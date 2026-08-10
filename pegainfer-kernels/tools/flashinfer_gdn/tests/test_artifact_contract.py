from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import artifact_contract as contract


PTX = """// Cuda compilation tools, release 12.9, V12.9.83
.version 8.8
.target sm_120a
.address_size 64
.visible .entry openinfer_gdn_test(
    .param .u64 q
)
{
    ret;
}
"""


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


def compile_metadata(variant: str) -> dict:
    return {
        **contract.expected_spec(variant),
        "flashinfer_commit": contract.FROZEN_FLASHINFER_COMMIT,
        "kernel_source_sha256": "a" * 64,
        "generator_sha256": contract.sha256_file(contract.compiler_path()),
        "requirements_lock_sha256": contract.sha256_file(contract.requirements_lock_path()),
        "workspace": source_metadata()["workspace"],
        "toolchain": {
            "python": "3.12.3",
            "host_cuda_toolkit": "12.8",
            "ptxas": "12.9",
            "ptx_compiler_release": "12.9",
            "ptx_compiler_version": "12.9.83",
            "ptx_isa": "8.8",
            "cutlass_dsl": "4.5.0",
            "cutlass_dsl_libs_base": "4.5.0",
            "cuda_nvcc_package": "12.9.86",
            "torch": "2.7.1",
            "cuda_python": "12.9.4",
            "cuda_bindings": "12.9.7",
        },
    }


class ArtifactContractTests(unittest.TestCase):
    def package(self, root: Path, variant: str) -> Path:
        raw = root / f"{variant}.ptx"
        metadata = root / f"{variant}.json"
        raw.write_text(PTX, encoding="utf-8")
        contract.write_json(metadata, compile_metadata(variant))
        with mock.patch.object(contract, "verify_flashinfer_source", return_value=source_metadata()):
            return contract.package_variant(
                variant=variant,
                raw_ptx_path=raw,
                compile_metadata_path=metadata,
                output_dir=root / "bundle" / variant,
                flashinfer_dir=root,
            )

    def test_both_geometries_and_dynamic_t_package(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifests = [contract.read_json(self.package(root, variant)) for variant in contract.SUPPORTED_GEOMETRIES]
            self.assertEqual({m["geometry"]["h_v"] for m in manifests}, {32, 48})
            self.assertTrue(all(m["tokens"] == {"extent": "dynamic", "minimum": 1, "divisibility": 1} for m in manifests))
            self.assertTrue(all(m["workspace"]["bytes_per_sm"] == 128 for m in manifests))
            self.assertTrue(all(not m["distribution"]["production_eligible"] for m in manifests))
            self.assertTrue(
                all(m["source"]["hkv_state_index_patch_applied"] for m in manifests)
            )
            self.assertTrue(
                all(m["abi"]["state_layout"] == "openinfer_hkv_v_contiguous" for m in manifests)
            )
            self.assertTrue(
                all(
                    m["abi"]["geometry_binding"]
                    == "manifest_guarded_runtime_head_parameters"
                    for m in manifests
                )
            )

    def test_normalization_removes_absolute_file_path(self) -> None:
        ptx = '.file 1 "/mnt/d/private/build/kernel.py"\n' + PTX
        normalized = contract.normalize_ptx(ptx)
        self.assertIn('.file 1 "kernel.py"', normalized)
        self.assertEqual(contract.leaked_absolute_paths(normalized), [])

    def test_path_leak_outside_file_directive_fails(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            raw = root / "bad.ptx"
            raw.write_text(PTX + "// /home/builder/secret/source.py\n", encoding="utf-8")
            metadata = root / "metadata.json"
            contract.write_json(metadata, compile_metadata("qwen35_4b_candidate"))
            with mock.patch.object(contract, "verify_flashinfer_source", return_value=source_metadata()):
                with self.assertRaisesRegex(contract.ContractError, "absolute path"):
                    contract.package_variant(
                        variant="qwen35_4b_candidate",
                        raw_ptx_path=raw,
                        compile_metadata_path=metadata,
                        output_dir=root / "out",
                        flashinfer_dir=root,
                    )

    def test_compile_metadata_mismatches_fail(self) -> None:
        source = source_metadata()
        cases = {
            "SHA": ("flashinfer_commit", "0" * 40),
            "SM": ("target_arch", "sm_100a"),
            "dtype": ("dtypes", {**contract.DTYPES, "q": "float16"}),
            "geometry": ("geometry", {"h_q": 16, "h_k": 16, "h_v": 31, "head_dim": 128}),
        }
        for label, (key, value) in cases.items():
            with self.subTest(label=label):
                metadata = compile_metadata("qwen35_4b_candidate")
                metadata[key] = value
                with self.assertRaises(contract.ContractError):
                    contract.validate_compile_metadata(metadata, "qwen35_4b_candidate", source)

    def test_manifest_patch_and_artifact_hash_mismatches_fail(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest_path = self.package(root, "qwen35_4b_candidate")
            original = contract.read_json(manifest_path)
            for label, mutate in (
                ("patch", lambda m: m["source"].__setitem__("patch_set_sha256", "0" * 64)),
                ("hash", lambda m: m["artifact"].__setitem__("sha256", "0" * 64)),
            ):
                with self.subTest(label=label):
                    manifest = copy.deepcopy(original)
                    mutate(manifest)
                    contract.write_json(manifest_path, manifest)
                    with self.assertRaises(contract.ContractError):
                        contract.validate_manifest(manifest_path)
            contract.write_json(manifest_path, original)

    def test_symbol_is_derived_from_ptx(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            manifest = contract.read_json(self.package(root, "operator_hv48"))
            self.assertEqual(manifest["abi"]["entry_symbol"], "openinfer_gdn_test")
            self.assertEqual(manifest["artifact"]["entry_symbols"], ["openinfer_gdn_test"])

    def test_packaging_is_reproducible_across_output_directories(self) -> None:
        with tempfile.TemporaryDirectory() as first_name, tempfile.TemporaryDirectory() as second_name:
            first = Path(first_name)
            second = Path(second_name)
            first_manifest = self.package(first, "qwen35_4b_candidate").read_bytes()
            second_manifest = self.package(second, "qwen35_4b_candidate").read_bytes()
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                (first / "bundle/qwen35_4b_candidate/kernel.ptx").read_bytes(),
                (second / "bundle/qwen35_4b_candidate/kernel.ptx").read_bytes(),
            )

    def test_source_lock_records_stage3_hkv_patch(self) -> None:
        lock, digest = contract.load_source_lock()
        self.assertEqual(len(lock["patches"]), 1)
        self.assertTrue(lock["hkv_state_index_patch"]["applied"])
        self.assertEqual(
            lock["hkv_state_index_patch"]["ordered_layout"], [1, 0, 2, 3]
        )
        self.assertEqual(
            lock["patches"][0]["sha256"],
            contract.sha256_file(
                contract.source_lock_path().parent / lock["patches"][0]["path"]
            ),
        )
        self.assertEqual(len(digest), 64)

    def test_bundle_index_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            bundle = root / "bundle"
            for variant in contract.SUPPORTED_GEOMETRIES:
                self.package(root, variant)
            index = {
                "schema_version": 1,
                "variants": {
                    variant: {
                        "manifest": f"{variant}/manifest.json",
                        "manifest_sha256": contract.sha256_file(
                            bundle / variant / "manifest.json"
                        ),
                    }
                    for variant in sorted(contract.SUPPORTED_GEOMETRIES)
                },
            }
            contract.write_json(bundle / "bundle.json", index)
            with mock.patch.object(
                contract, "verify_flashinfer_source", return_value=source_metadata()
            ):
                contract.validate_bundle(bundle, flashinfer_dir=root)
                index["variants"]["operator_hv48"]["manifest_sha256"] = "0" * 64
                contract.write_json(bundle / "bundle.json", index)
                with self.assertRaisesRegex(contract.ContractError, "bundle manifest index"):
                    contract.validate_bundle(bundle, flashinfer_dir=root)


if __name__ == "__main__":
    unittest.main()
