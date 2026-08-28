// CUPTI injection library that mines every CUDA module and kernel launch from
// a host process it is loaded into (vLLM, sglang, or PegaInfer itself), so a
// launched kernel's cubin, launch configuration, and staged parameter bytes
// can be lifted out for replay. Provider-agnostic: Triton, cuBLAS/cuBLASLt,
// CUTLASS, FlashInfer, and hand-written CUDA all surface as module-load events
// with real cubin bytes, not just Triton's on-disk cache.
//
// Load it with the CUDA driver's injection hook:
//
//   CUDA_INJECTION64_PATH=/path/to/libkernelcapture.so \
//   KERNEL_CAPTURE_DIR=/some/out \
//     python -m vllm.entrypoints.openai.api_server ...
//
// Output under KERNEL_CAPTURE_DIR (default ./kernel-capture):
//   module_<moduleId>.cubin   one file per distinct loaded module
//   launches.jsonl            one JSON object per captured kernel launch
//
// The driver calls InitializeInjection() once, before the first CUDA call, on
// any library named by CUDA_INJECTION64_PATH.

#define _GNU_SOURCE
#include <cupti.h>
#include <cuda.h>
// cuLaunchKernel_params / cuLaunchKernelEx_params come from
// generated_cuda_meta.h, which cupti.h already includes transitively.

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

// A kernel's parameter buffer is a few KiB at most; a walk past this many
// indices means cuFuncGetParamInfo is misbehaving, not that the kernel is huge.
#define MAX_PARAMS 4096

static CUpti_SubscriberHandle g_subscriber;
static char g_out_dir[4096];
static FILE *g_launches;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

// Bounded set of module ids already written, so a module loaded once but
// launched from thousands of times is dumped a single time.
static uint32_t g_seen_modules[65536];
static size_t g_seen_count;

static int module_already_seen(uint32_t module_id) {
  for (size_t i = 0; i < g_seen_count; i++) {
    if (g_seen_modules[i] == module_id) {
      return 1;
    }
  }
  if (g_seen_count < sizeof(g_seen_modules) / sizeof(g_seen_modules[0])) {
    g_seen_modules[g_seen_count++] = module_id;
  }
  return 0;
}

static void write_cubin(const CUpti_ModuleResourceData *module) {
  if (module_already_seen(module->moduleId)) {
    return;
  }
  char path[4200];
  snprintf(path, sizeof(path), "%s/module_%u.cubin", g_out_dir,
           module->moduleId);
  FILE *f = fopen(path, "wb");
  if (!f) {
    return;
  }
  fwrite(module->pCubin, 1, module->cubinSize, f);
  fclose(f);
}

static void write_hex(FILE *out, const unsigned char *bytes, size_t len) {
  static const char digits[] = "0123456789abcdef";
  for (size_t i = 0; i < len; i++) {
    fputc(digits[bytes[i] >> 4], out);
    fputc(digits[bytes[i] & 0xf], out);
  }
}

// Emit one launches.jsonl record: symbol, launch geometry, and every staged
// parameter value read through the function's own parameter layout.
static void record_launch(const char *symbol, CUfunction func,
                          unsigned int grid_x, unsigned int grid_y,
                          unsigned int grid_z, unsigned int block_x,
                          unsigned int block_y, unsigned int block_z,
                          unsigned int shared_bytes, void **kernel_params) {
  int num_regs = 0, static_shared = 0, const_bytes = 0, local_bytes = 0;
  int ptx_version = 0, binary_version = 0;
  cuFuncGetAttribute(&num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, func);
  cuFuncGetAttribute(&static_shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func);
  cuFuncGetAttribute(&const_bytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, func);
  cuFuncGetAttribute(&local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func);
  cuFuncGetAttribute(&ptx_version, CU_FUNC_ATTRIBUTE_PTX_VERSION, func);
  cuFuncGetAttribute(&binary_version, CU_FUNC_ATTRIBUTE_BINARY_VERSION, func);

  pthread_mutex_lock(&g_lock);
  fprintf(g_launches, "{\"symbol\":\"%s\",", symbol ? symbol : "");
  fprintf(g_launches, "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],",
          grid_x, grid_y, grid_z, block_x, block_y, block_z);
  fprintf(g_launches, "\"dynamic_shared_mem_bytes\":%u,", shared_bytes);
  fprintf(g_launches,
          "\"attributes\":{\"num_regs\":%d,\"static_shared_bytes\":%d,"
          "\"const_bytes\":%d,\"local_bytes\":%d,\"ptx_version\":%d,"
          "\"binary_version\":%d},",
          num_regs, static_shared, const_bytes, local_bytes, ptx_version,
          binary_version);

  // A null kernelParams array means the launch passed arguments through the
  // packed `extra` buffer (cuBLASLt nvjet kernels do this); the layout walk
  // does not apply, so the parameter list is explicitly null.
  if (!kernel_params) {
    fprintf(g_launches, "\"params\":null}\n");
    fflush(g_launches);
    pthread_mutex_unlock(&g_lock);
    return;
  }

  fprintf(g_launches, "\"params\":[");
  for (size_t index = 0; index < MAX_PARAMS; index++) {
    size_t offset = 0, size = 0;
    CUresult r = cuFuncGetParamInfo(func, index, &offset, &size);
    if (r == CUDA_ERROR_INVALID_VALUE) {
      break;
    }
    if (r != CUDA_SUCCESS) {
      break;
    }
    const unsigned char *staged = (const unsigned char *)kernel_params[index];
    if (index > 0) {
      fputc(',', g_launches);
    }
    fprintf(g_launches, "{\"offset\":%zu,\"size\":%zu,\"data\":\"", offset,
            size);
    if (staged) {
      write_hex(g_launches, staged, size);
    }
    fprintf(g_launches, "\"}");
  }
  fprintf(g_launches, "]}\n");
  fflush(g_launches);
  pthread_mutex_unlock(&g_lock);
}

static void CUPTIAPI callback(void *userdata, CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid, const void *cbdata) {
  (void)userdata;
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED) {
      const CUpti_ResourceData *resource = (const CUpti_ResourceData *)cbdata;
      write_cubin(
          (const CUpti_ModuleResourceData *)resource->resourceDescriptor);
    }
    return;
  }
  if (domain != CUPTI_CB_DOMAIN_DRIVER_API) {
    return;
  }
  const CUpti_CallbackData *data = (const CUpti_CallbackData *)cbdata;
  // Read arguments on the way in, while the caller's parameter staging is still
  // the live launch value.
  if (data->callbackSite != CUPTI_API_ENTER) {
    return;
  }
  if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel) {
    const cuLaunchKernel_params *p =
        (const cuLaunchKernel_params *)data->functionParams;
    record_launch(data->symbolName, p->f, p->gridDimX, p->gridDimY, p->gridDimZ,
                  p->blockDimX, p->blockDimY, p->blockDimZ, p->sharedMemBytes,
                  p->kernelParams);
  } else if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx) {
    const cuLaunchKernelEx_params *p =
        (const cuLaunchKernelEx_params *)data->functionParams;
    const CUlaunchConfig *cfg = p->config;
    record_launch(data->symbolName, p->f, cfg->gridDimX, cfg->gridDimY,
                  cfg->gridDimZ, cfg->blockDimX, cfg->blockDimY, cfg->blockDimZ,
                  cfg->sharedMemBytes, p->kernelParams);
  }
}

// The CUDA driver's injection entry point. Named exactly this; the driver
// dlsym's it out of every library on CUDA_INJECTION64_PATH.
int InitializeInjection(void) {
  const char *dir = getenv("KERNEL_CAPTURE_DIR");
  snprintf(g_out_dir, sizeof(g_out_dir), "%s",
           dir && *dir ? dir : "./kernel-capture");
  mkdir(g_out_dir, 0755);

  char path[4200];
  snprintf(path, sizeof(path), "%s/launches.jsonl", g_out_dir);
  g_launches = fopen(path, "w");
  if (!g_launches) {
    fprintf(stderr, "[kernel-capture] cannot open %s\n", path);
    return 0;
  }

  if (cuptiSubscribe(&g_subscriber, (CUpti_CallbackFunc)callback, NULL) !=
      CUPTI_SUCCESS) {
    fprintf(stderr, "[kernel-capture] cuptiSubscribe failed\n");
    return 0;
  }
  cuptiEnableDomain(1, g_subscriber, CUPTI_CB_DOMAIN_RESOURCE);
  cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel);
  cuptiEnableCallback(1, g_subscriber, CUPTI_CB_DOMAIN_DRIVER_API,
                      CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx);
  fprintf(stderr, "[kernel-capture] active, writing to %s\n", g_out_dir);
  return 1;
}
