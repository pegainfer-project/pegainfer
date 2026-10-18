"""Per-unit adapters. One module per bench unit; three functions each:

- make_inputs(shape, seed) -> dict[str, torch.Tensor]
- run(lib, tensors, shape, stream) — ctypes launch of the production symbol
- reference(tensors, shape) -> torch.Tensor (f32)

Module level must stay torch-free (registry/pytest import these on CPU boxes);
import torch inside functions via kernel_lab.loader.require_torch().
"""
