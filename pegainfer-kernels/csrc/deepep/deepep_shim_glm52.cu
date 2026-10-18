// DeepEP elastic shim, GLM5.2 instantiation (DP1/EP8, glm52_deepep_* symbols).
// Compiled only under the glm52 feature; the parameterized body lives in
// deepep_shim_impl.cuh.

#define DEEPEP_SHIM_HAS_PREFILL 0

#include "deepep_glm52.h"
#include "deepep_config_glm52.cuh"

#define DEEPEP_SHIM_CFG deepep_shim::cfg_glm52
#define DEEPEP_SHIM_CTX Glm52DeepEpCtx
#define DEEPEP_SHIM_FN(name) glm52_deepep_##name

#include "deepep_shim_impl.cuh"
