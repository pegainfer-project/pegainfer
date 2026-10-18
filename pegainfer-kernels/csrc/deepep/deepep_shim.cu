// DeepEP elastic shim, Kimi-K2 instantiation (TP1/DP8/EP8, deepep_* symbols).
// The parameterized body lives in deepep_shim_impl.cuh.

#define DEEPEP_SHIM_HAS_PREFILL 1

#include "deepep.h"
#include "deepep_config.cuh"

#define DEEPEP_SHIM_CFG deepep_shim::cfg
#define DEEPEP_SHIM_CTX DeepEpCtx
#define DEEPEP_SHIM_FN(name) deepep_##name

#include "deepep_shim_impl.cuh"
