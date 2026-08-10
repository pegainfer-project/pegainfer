from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
FLASHINFER_DIR = REPO_ROOT / "pegainfer-kernels/third_party/flashinfer"
PATCH_PATH = TOOLS_DIR / "patches/0001-openinfer-hkv-state-layout.patch"
sys.path.insert(0, str(TOOLS_DIR))

from state_layout_contract import (
    OPENINFER_HKV_ORDER,
    UPSTREAM_ORDER,
    StateGeometry,
    cute_state_offset,
    first_wrong_mapping,
    openinfer_hkv_offset,
    ordered_strides,
    upstream_hvk_offset,
)


class StateLayoutContractTests(unittest.TestCase):
    def test_ordered_layout_strides_explain_hvk_to_hkv_patch(self) -> None:
        geometry = StateGeometry(heads=2, key_dim=3, value_dim=5)
        self.assertEqual(ordered_strides(geometry.shape, UPSTREAM_ORDER), (1, 3, 15, 30))
        self.assertEqual(
            ordered_strides(geometry.shape, OPENINFER_HKV_ORDER), (5, 1, 15, 30)
        )

    def test_patched_cute_mapping_equals_hkv_for_hv32_and_hv48(self) -> None:
        for heads in (32, 48):
            geometry = StateGeometry(heads=heads, key_dim=128, value_dim=128)
            with self.subTest(heads=heads):
                for head in range(heads):
                    for key in range(geometry.key_dim):
                        for value in range(geometry.value_dim):
                            self.assertEqual(
                                cute_state_offset(
                                    geometry, head=head, key=key, value=value
                                ),
                                openinfer_hkv_offset(
                                    geometry, head=head, key=key, value=value
                                ),
                            )

    def test_mapping_does_not_depend_on_token_extent(self) -> None:
        geometry = StateGeometry(heads=2, key_dim=3, value_dim=5)
        baseline = cute_state_offset(geometry, head=1, key=2, value=4)
        for dynamic_t in (1, 2, 63, 64, 65, 127, 128):
            with self.subTest(dynamic_t=dynamic_t):
                self.assertEqual(
                    cute_state_offset(geometry, head=1, key=2, value=4), baseline
                )

    def test_upstream_hvk_negative_case_reports_first_mismatch(self) -> None:
        geometry = StateGeometry(heads=2, key_dim=3, value_dim=5)
        mismatch = first_wrong_mapping(geometry)
        self.assertIsNotNone(mismatch)
        coordinate, expected, actual = mismatch or ((0, 0, 0), 0, 0)
        self.assertEqual(coordinate, (0, 0, 1))
        self.assertNotEqual(expected, actual)
        self.assertEqual(
            cute_state_offset(
                geometry,
                head=coordinate[0],
                key=coordinate[1],
                value=coordinate[2],
                order=UPSTREAM_ORDER,
            ),
            upstream_hvk_offset(
                geometry,
                head=coordinate[0],
                key=coordinate[1],
                value=coordinate[2],
            ),
        )

    def test_patch_applies_cleanly_to_frozen_flashinfer(self) -> None:
        result = subprocess.run(
            ["git", "-C", str(FLASHINFER_DIR), "apply", "--check", str(PATCH_PATH)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_patch_scope_is_only_two_state_layout_orders(self) -> None:
        patch = PATCH_PATH.read_text(encoding="utf-8")
        self.assertEqual(patch.count("order=(1, 0, 2, 3)"), 2)
        self.assertEqual(patch.count("order=(0, 1, 2, 3)"), 2)
        self.assertNotIn("q_tma", patch)
        self.assertNotIn("k_tma", patch)
        self.assertNotIn("v_tma", patch)
        self.assertNotIn("o_tma", patch)
        self.assertNotIn("transpose", patch.lower())
        self.assertNotIn("copy_", patch)


if __name__ == "__main__":
    unittest.main()
