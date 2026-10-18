#include "ffi_guard.cuh"

#include <string>

namespace {
thread_local std::string g_ffi_last_error;
}

void pegainfer_ffi_set_last_error(const char* what) {
  g_ffi_last_error = what == nullptr ? "" : what;
}

extern "C" const char* pegainfer_kernels_last_error() { return g_ffi_last_error.c_str(); }
