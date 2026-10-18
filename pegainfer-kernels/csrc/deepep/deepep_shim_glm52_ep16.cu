// DeepEP elastic shim, GLM5.2 EP16 instantiation (DP1/EP16,
// glm52_ep16_deepep_* symbols). Compiled only under the glm52 feature; the
// parameterized body lives in deepep_shim_impl.cuh.

#define DEEPEP_SHIM_HAS_PREFILL 0

#include "deepep_glm52_ep16.h"
#include "deepep_config_glm52_ep16.cuh"

#define DEEPEP_SHIM_CFG deepep_shim::cfg_glm52_ep16
#define DEEPEP_SHIM_CTX Glm52Ep16DeepEpCtx
#define DEEPEP_SHIM_FN(name) glm52_ep16_deepep_##name

#include "deepep_shim_impl.cuh"
