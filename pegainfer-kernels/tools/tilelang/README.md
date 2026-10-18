# TileLang Generators

This directory owns TileLang-based CUDA source generators used by
`pegainfer-kernels`.

Keep the technology boundary here and put model- or shape-family-specific
programs in subdirectories:

| Path | Role |
| --- | --- |
| `glm52/generate.py` | GLM5.2 sparse MLA decode kernel (Hopper wgmma, sm_90a). |

Generated CUDA is a build artifact under Cargo `OUT_DIR`; it should not be
checked into the repository.
