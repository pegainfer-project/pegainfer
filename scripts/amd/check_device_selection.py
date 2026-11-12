#!/usr/bin/env python3
"""Read HIP metadata only, before the foundation verifier executes GPU work."""
import ctypes
import json
import os
from pathlib import Path

lock = json.loads(Path(__file__).with_name("environment.lock.json").read_text())
lib = ctypes.CDLL("/opt/venv/lib/python3.13/site-packages/_rocm_sdk_core/lib/libamdhip64.so.7")
lib.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
lib.hipDeviceGetPCIBusId.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
count = ctypes.c_int()
assert lib.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value == 1, "select exactly one HIP device"
bus = ctypes.create_string_buffer(64)
assert lib.hipDeviceGetPCIBusId(bus, len(bus), 0) == 0
actual = bus.value.decode()
assert actual not in lock["device_selection"]["excluded_pci_bus_ids"], f"excluded physical GPU selected: {actual}; physical GPU0 is HIP ordinal 2 on this host"
expected = os.environ["PEGAINFER_EXPECTED_PCI_BUS_ID"]
assert actual == expected, f"HIP selection maps to {actual}, expected {expected}"
print(json.dumps({"device_selection_passed": True, "hip_visible_devices": os.environ["HIP_VISIBLE_DEVICES"],
                  "logical_device": 0, "pci_bus_id": actual}), flush=True)
