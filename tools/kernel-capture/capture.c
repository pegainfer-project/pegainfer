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
// Output under KERNEL_CAPTURE_DIR (default ./kernel-capture), in a pid<N>/
// subdirectory per injected process so multi-rank engines (TP/EP workers)
// don't clobber each other:
//   pid<N>/module_<moduleId>.cubin   one file per distinct loaded module
//   pid<N>/launches.jsonl            one JSON object per captured kernel launch
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
#include <unistd.h>

// A kernel's parameter buffer is a few KiB at most; a walk past this many
// indices means cuFuncGetParamInfo is misbehaving, not that the kernel is huge.
#define MAX_PARAMS 4096

#define MAX_ABI_PARAMS 128
#define MAX_SYMBOL 640

static CUpti_SubscriberHandle g_subscriber;
static char g_out_dir[4096];
static FILE *g_launches;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

// Bounded set of module ids already written, so a module loaded once but
// launched from thousands of times is dumped a single time.
static uint32_t g_seen_modules[65536];
static size_t g_seen_count;

// One kernel's parameter layout, learned from the driver's own parse of the
// cubin at module-load time (see cache_module_abi). Used at launch to size the
// staged kernelParams array for kernels whose launch-time CUfunction does not
// support cuFuncGetParamInfo — every PyTorch/vLLM kernel launched through the
// runtime `<<<>>>` path.
typedef struct {
  int num_regs, static_shared, const_bytes, local_bytes;
  int ptx_version, binary_version;
} KernelAttrs;

typedef struct {
  char symbol[MAX_SYMBOL];
  int nparams;
  size_t offset[MAX_ABI_PARAMS];
  size_t size[MAX_ABI_PARAMS];
  KernelAttrs attrs;
} KernelAbi;

static KernelAttrs read_attrs(CUfunction func) {
  KernelAttrs a = {0};
  cuFuncGetAttribute(&a.num_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, func);
  cuFuncGetAttribute(&a.static_shared, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, func);
  cuFuncGetAttribute(&a.const_bytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, func);
  cuFuncGetAttribute(&a.local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, func);
  cuFuncGetAttribute(&a.ptx_version, CU_FUNC_ATTRIBUTE_PTX_VERSION, func);
  cuFuncGetAttribute(&a.binary_version, CU_FUNC_ATTRIBUTE_BINARY_VERSION, func);
  return a;
}

static KernelAbi *g_abi;
static size_t g_abi_count;
static size_t g_abi_cap;
// cuModuleLoadData below triggers another MODULE_LOADED callback on this
// thread; the guard keeps that recursion from re-entering the loader.
static __thread int g_in_self_load;

static const KernelAbi *abi_lookup(const char *symbol) {
  if (!symbol) {
    return NULL;
  }
  for (size_t i = 0; i < g_abi_count; i++) {
    if (strcmp(g_abi[i].symbol, symbol) == 0) {
      return &g_abi[i];
    }
  }
  return NULL;
}

static KernelAbi *abi_new(void) {
  if (g_abi_count == g_abi_cap) {
    size_t cap = g_abi_cap ? g_abi_cap * 2 : 1024;
    KernelAbi *grown = realloc(g_abi, cap * sizeof(*grown));
    if (!grown) {
      return NULL;
    }
    g_abi = grown;
    g_abi_cap = cap;
  }
  return &g_abi[g_abi_count++];
}

// Load a private copy of the just-loaded module and record every kernel's
// parameter layout. The driver parses the cubin for us, so this works for
// kernels the runtime registers, which the launch-time handle cannot answer.
static void cache_module_abi(const CUpti_ModuleResourceData *module) {
  g_in_self_load = 1;
  CUmodule mod = NULL;
  if (cuModuleLoadData(&mod, module->pCubin) != CUDA_SUCCESS) {
    g_in_self_load = 0;
    return;
  }
  unsigned int count = 0;
  if (cuModuleGetFunctionCount(&count, mod) != CUDA_SUCCESS || count == 0) {
    cuModuleUnload(mod);
    g_in_self_load = 0;
    return;
  }
  CUfunction *funcs = calloc(count, sizeof(*funcs));
  if (funcs && cuModuleEnumerateFunctions(funcs, count, mod) == CUDA_SUCCESS) {
    for (unsigned int i = 0; i < count; i++) {
      const char *name = NULL;
      if (cuFuncGetName(&name, funcs[i]) != CUDA_SUCCESS || !name) {
        continue;
      }
      KernelAbi *abi = abi_new();
      if (!abi) {
        break;
      }
      snprintf(abi->symbol, sizeof(abi->symbol), "%s", name);
      abi->attrs = read_attrs(funcs[i]);
      abi->nparams = 0;
      for (size_t p = 0; p < MAX_ABI_PARAMS; p++) {
        size_t offset = 0, size = 0;
        if (cuFuncGetParamInfo(funcs[i], p, &offset, &size) != CUDA_SUCCESS) {
          break;
        }
        abi->offset[p] = offset;
        abi->size[p] = size;
        abi->nparams++;
      }
    }
  }
  free(funcs);
  cuModuleUnload(mod);
  g_in_self_load = 0;
}

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

// Resolve an 8-byte parameter value against the CUDA allocation map and, if it
// is a live pointer, append a "pointer" object with its memory type and owning
// range — the signal that separates device/kv/weight pointers from scalars for
// downstream binding inference. Advisory: an integer that collides with an
// allocation also matches.
static void write_pointer_field(FILE *out, const unsigned char *bytes) {
  uint64_t value;
  memcpy(&value, bytes, sizeof(value));
  if (value == 0) {
    return;
  }
  unsigned int memory_type = 0;
  if (cuPointerGetAttribute(&memory_type, CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
                            (CUdeviceptr)value) != CUDA_SUCCESS) {
    return;
  }
  const char *type_name = memory_type == CU_MEMORYTYPE_HOST      ? "host"
                          : memory_type == CU_MEMORYTYPE_DEVICE  ? "device"
                          : memory_type == CU_MEMORYTYPE_ARRAY   ? "array"
                          : memory_type == CU_MEMORYTYPE_UNIFIED ? "unified"
                                                                 : "unknown";
  uint64_t range_start = 0;
  size_t range_size = 0;
  cuPointerGetAttribute(&range_start, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR,
                        (CUdeviceptr)value);
  cuPointerGetAttribute(&range_size, CU_POINTER_ATTRIBUTE_RANGE_SIZE,
                        (CUdeviceptr)value);
  fprintf(out,
          ",\"pointer\":{\"memory_type\":\"%s\",\"range_start\":\"0x%llx\","
          "\"range_size\":%zu}",
          type_name, (unsigned long long)range_start, range_size);
}

// Fill offset[]/size[] for a launch, preferring the live handle and falling
// back to the module-load ABI cache (which the runtime `<<<>>>` launch handle
// cannot answer). Returns the parameter count, or -1 if neither source knows.
static int resolve_layout(CUfunction func, const char *symbol, size_t *offset,
                          size_t *size) {
  for (int index = 0; index < MAX_PARAMS; index++) {
    size_t off = 0, sz = 0;
    if (cuFuncGetParamInfo(func, index, &off, &sz) != CUDA_SUCCESS) {
      if (index > 0) {
        return index;
      }
      break;
    }
    if ((size_t)index >= MAX_ABI_PARAMS) {
      return index;
    }
    offset[index] = off;
    size[index] = sz;
  }
  const KernelAbi *abi = abi_lookup(symbol);
  if (!abi) {
    return -1;
  }
  for (int index = 0; index < abi->nparams; index++) {
    offset[index] = abi->offset[index];
    size[index] = abi->size[index];
  }
  return abi->nparams;
}

// Emit one launches.jsonl record: symbol, launch geometry, function
// attributes, and every staged parameter value read through the resolved
// parameter layout.
static void record_launch(const char *symbol, CUfunction func,
                          unsigned int grid_x, unsigned int grid_y,
                          unsigned int grid_z, unsigned int block_x,
                          unsigned int block_y, unsigned int block_z,
                          unsigned int shared_bytes, void **kernel_params) {
  KernelAttrs attrs = read_attrs(func);

  pthread_mutex_lock(&g_lock);
  if (attrs.num_regs == 0) {
    const KernelAbi *abi = abi_lookup(symbol);
    if (abi) {
      attrs = abi->attrs;
    }
  }
  fprintf(g_launches, "{\"symbol\":\"%s\",", symbol ? symbol : "");
  fprintf(g_launches, "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],",
          grid_x, grid_y, grid_z, block_x, block_y, block_z);
  fprintf(g_launches, "\"dynamic_shared_mem_bytes\":%u,", shared_bytes);
  fprintf(g_launches,
          "\"attributes\":{\"num_regs\":%d,\"static_shared_bytes\":%d,"
          "\"const_bytes\":%d,\"local_bytes\":%d,\"ptx_version\":%d,"
          "\"binary_version\":%d},",
          attrs.num_regs, attrs.static_shared, attrs.const_bytes,
          attrs.local_bytes, attrs.ptx_version, attrs.binary_version);

  // A null kernelParams array means the launch passed arguments through the
  // packed `extra` buffer (cuBLASLt nvjet kernels do this); the layout walk
  // does not apply, so the parameter list is explicitly null.
  if (!kernel_params) {
    fprintf(g_launches, "\"params\":null}\n");
    fflush(g_launches);
    pthread_mutex_unlock(&g_lock);
    return;
  }

  size_t offset[MAX_ABI_PARAMS], size[MAX_ABI_PARAMS];
  int nparams = resolve_layout(func, symbol, offset, size);
  fprintf(g_launches, "\"params\":");
  if (nparams < 0) {
    // kernelParams is present but neither the handle nor the cache knows its
    // length; emitting a bounded blind walk would risk reading past the array.
    fprintf(g_launches, "\"unknown-layout\"}\n");
    fflush(g_launches);
    pthread_mutex_unlock(&g_lock);
    return;
  }
  fputc('[', g_launches);
  for (int index = 0; index < nparams; index++) {
    const unsigned char *staged = (const unsigned char *)kernel_params[index];
    if (index > 0) {
      fputc(',', g_launches);
    }
    fprintf(g_launches, "{\"offset\":%zu,\"size\":%zu,\"data\":\"", offset[index],
            size[index]);
    if (staged) {
      write_hex(g_launches, staged, size[index]);
    }
    fputc('"', g_launches);
    if (staged && size[index] == 8) {
      write_pointer_field(g_launches, staged);
    }
    fputc('}', g_launches);
  }
  fprintf(g_launches, "]}\n");
  fflush(g_launches);
  pthread_mutex_unlock(&g_lock);
}

static void CUPTIAPI callback(void *userdata, CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid, const void *cbdata) {
  (void)userdata;
  if (domain == CUPTI_CB_DOMAIN_RESOURCE) {
    // Our own cuModuleLoadData in cache_module_abi re-enters this callback;
    // skip that recursion.
    if (cbid == CUPTI_CBID_RESOURCE_MODULE_LOADED && !g_in_self_load) {
      const CUpti_ResourceData *resource = (const CUpti_ResourceData *)cbdata;
      const CUpti_ModuleResourceData *module =
          (const CUpti_ModuleResourceData *)resource->resourceDescriptor;
      // Multi-rank engines (one process, one thread per GPU) hit this path
      // concurrently on lazy module loads; g_seen_* and the g_abi table are
      // shared, so the whole module path holds the same lock as the readers.
      // The recursive MODULE_LOADED from our own cuModuleLoadData is filtered
      // by g_in_self_load above, before it could reach the lock.
      pthread_mutex_lock(&g_lock);
      write_cubin(module);
      cache_module_abi(module);
      pthread_mutex_unlock(&g_lock);
    }
    return;
  }
  if (domain != CUPTI_CB_DOMAIN_DRIVER_API) {
    return;
  }
  const CUpti_CallbackData *data = (const CUpti_CallbackData *)cbdata;
  // Read on the way *out*: under lazy module loading (the driver default since
  // CUDA 12.x) a kernel's module is materialized during the launch call, so on
  // the ENTER site cuFuncGetParamInfo/cuFuncGetAttribute return zeros. By EXIT
  // the function is loaded and queryable, and the caller's kernelParams array
  // — pointed at by functionParams — is still live within this callback.
  if (data->callbackSite != CUPTI_API_EXIT) {
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
  char base_dir[4096];
  snprintf(base_dir, sizeof(base_dir), "%s",
           dir && *dir ? dir : "./kernel-capture");
  mkdir(base_dir, 0755);
  snprintf(g_out_dir, sizeof(g_out_dir), "%.4080s/pid%d", base_dir,
           (int)getpid());
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
