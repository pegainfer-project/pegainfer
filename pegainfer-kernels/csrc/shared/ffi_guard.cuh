// Catch C++ exceptions (FlashInfer throws from host-side dispatch) before they
// cross the extern "C" boundary — a foreign exception reaching Rust aborts the
// process with no message.
#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <exception>

void pegainfer_ffi_set_last_error(const char* what);

// Entry points that call cuBLAS return this plus the cublasStatus_t on a
// cuBLAS failure, and the cudaError_t otherwise; Rust's
// `ops::ffi_status_detail` decodes the pair.
constexpr int PEGAINFER_CUBLAS_STATUS_BASE = 100000;

// The driver result an entry point that returns CUresult gives for a runtime
// error; every such conversion goes through here, so the same fault reads the
// same in Rust whichever kernel hit it. Each runtime error with a driver
// counterpart maps to it (the pairs share a value in the 12.8 and 13.1
// headers). The runtime-only errors have no driver code and report
// CUDA_ERROR_UNKNOWN, apart from the deprecated invalid device pointer. The
// guarded cases are newer than CUDA 12.3, the oldest supported toolkit.
inline CUresult map_cuda_error(cudaError_t err) {
  switch (err) {
    case cudaSuccess: return CUDA_SUCCESS;
    case cudaErrorInvalidValue: return CUDA_ERROR_INVALID_VALUE;
    case cudaErrorInvalidDevicePointer: return CUDA_ERROR_INVALID_VALUE;
    case cudaErrorMemoryAllocation: return CUDA_ERROR_OUT_OF_MEMORY;
    case cudaErrorInitializationError: return CUDA_ERROR_NOT_INITIALIZED;
    case cudaErrorCudartUnloading: return CUDA_ERROR_DEINITIALIZED;
    case cudaErrorProfilerDisabled: return CUDA_ERROR_PROFILER_DISABLED;
    case cudaErrorProfilerNotInitialized: return CUDA_ERROR_PROFILER_NOT_INITIALIZED;
    case cudaErrorProfilerAlreadyStarted: return CUDA_ERROR_PROFILER_ALREADY_STARTED;
    case cudaErrorProfilerAlreadyStopped: return CUDA_ERROR_PROFILER_ALREADY_STOPPED;
    case cudaErrorStubLibrary: return CUDA_ERROR_STUB_LIBRARY;
    case cudaErrorDevicesUnavailable: return CUDA_ERROR_DEVICE_UNAVAILABLE;
    case cudaErrorNoDevice: return CUDA_ERROR_NO_DEVICE;
    case cudaErrorInvalidDevice: return CUDA_ERROR_INVALID_DEVICE;
    case cudaErrorDeviceNotLicensed: return CUDA_ERROR_DEVICE_NOT_LICENSED;
    case cudaErrorInvalidKernelImage: return CUDA_ERROR_INVALID_IMAGE;
    case cudaErrorDeviceUninitialized: return CUDA_ERROR_INVALID_CONTEXT;
    case cudaErrorMapBufferObjectFailed: return CUDA_ERROR_MAP_FAILED;
    case cudaErrorUnmapBufferObjectFailed: return CUDA_ERROR_UNMAP_FAILED;
    case cudaErrorArrayIsMapped: return CUDA_ERROR_ARRAY_IS_MAPPED;
    case cudaErrorAlreadyMapped: return CUDA_ERROR_ALREADY_MAPPED;
    case cudaErrorNoKernelImageForDevice: return CUDA_ERROR_NO_BINARY_FOR_GPU;
    case cudaErrorAlreadyAcquired: return CUDA_ERROR_ALREADY_ACQUIRED;
    case cudaErrorNotMapped: return CUDA_ERROR_NOT_MAPPED;
    case cudaErrorNotMappedAsArray: return CUDA_ERROR_NOT_MAPPED_AS_ARRAY;
    case cudaErrorNotMappedAsPointer: return CUDA_ERROR_NOT_MAPPED_AS_POINTER;
    case cudaErrorECCUncorrectable: return CUDA_ERROR_ECC_UNCORRECTABLE;
    case cudaErrorUnsupportedLimit: return CUDA_ERROR_UNSUPPORTED_LIMIT;
    case cudaErrorDeviceAlreadyInUse: return CUDA_ERROR_CONTEXT_ALREADY_IN_USE;
    case cudaErrorPeerAccessUnsupported: return CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
    case cudaErrorInvalidPtx: return CUDA_ERROR_INVALID_PTX;
    case cudaErrorInvalidGraphicsContext: return CUDA_ERROR_INVALID_GRAPHICS_CONTEXT;
    case cudaErrorNvlinkUncorrectable: return CUDA_ERROR_NVLINK_UNCORRECTABLE;
    case cudaErrorJitCompilerNotFound: return CUDA_ERROR_JIT_COMPILER_NOT_FOUND;
    case cudaErrorUnsupportedPtxVersion: return CUDA_ERROR_UNSUPPORTED_PTX_VERSION;
    case cudaErrorJitCompilationDisabled: return CUDA_ERROR_JIT_COMPILATION_DISABLED;
    case cudaErrorUnsupportedExecAffinity: return CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY;
    case cudaErrorUnsupportedDevSideSync: return CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC;
    case cudaErrorInvalidSource: return CUDA_ERROR_INVALID_SOURCE;
    case cudaErrorFileNotFound: return CUDA_ERROR_FILE_NOT_FOUND;
    case cudaErrorSharedObjectSymbolNotFound: return CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND;
    case cudaErrorSharedObjectInitFailed: return CUDA_ERROR_SHARED_OBJECT_INIT_FAILED;
    case cudaErrorOperatingSystem: return CUDA_ERROR_OPERATING_SYSTEM;
    case cudaErrorInvalidResourceHandle: return CUDA_ERROR_INVALID_HANDLE;
    case cudaErrorIllegalState: return CUDA_ERROR_ILLEGAL_STATE;
    case cudaErrorLossyQuery: return CUDA_ERROR_LOSSY_QUERY;
    case cudaErrorSymbolNotFound: return CUDA_ERROR_NOT_FOUND;
    case cudaErrorNotReady: return CUDA_ERROR_NOT_READY;
    case cudaErrorIllegalAddress: return CUDA_ERROR_ILLEGAL_ADDRESS;
    case cudaErrorLaunchOutOfResources: return CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES;
    case cudaErrorLaunchTimeout: return CUDA_ERROR_LAUNCH_TIMEOUT;
    case cudaErrorLaunchIncompatibleTexturing: return CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING;
    case cudaErrorPeerAccessAlreadyEnabled: return CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED;
    case cudaErrorPeerAccessNotEnabled: return CUDA_ERROR_PEER_ACCESS_NOT_ENABLED;
    case cudaErrorSetOnActiveProcess: return CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE;
    case cudaErrorContextIsDestroyed: return CUDA_ERROR_CONTEXT_IS_DESTROYED;
    case cudaErrorAssert: return CUDA_ERROR_ASSERT;
    case cudaErrorTooManyPeers: return CUDA_ERROR_TOO_MANY_PEERS;
    case cudaErrorHostMemoryAlreadyRegistered: return CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED;
    case cudaErrorHostMemoryNotRegistered: return CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED;
    case cudaErrorHardwareStackError: return CUDA_ERROR_HARDWARE_STACK_ERROR;
    case cudaErrorIllegalInstruction: return CUDA_ERROR_ILLEGAL_INSTRUCTION;
    case cudaErrorMisalignedAddress: return CUDA_ERROR_MISALIGNED_ADDRESS;
    case cudaErrorInvalidAddressSpace: return CUDA_ERROR_INVALID_ADDRESS_SPACE;
    case cudaErrorInvalidPc: return CUDA_ERROR_INVALID_PC;
    case cudaErrorLaunchFailure: return CUDA_ERROR_LAUNCH_FAILED;
    case cudaErrorCooperativeLaunchTooLarge: return CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE;
    case cudaErrorNotPermitted: return CUDA_ERROR_NOT_PERMITTED;
    case cudaErrorNotSupported: return CUDA_ERROR_NOT_SUPPORTED;
    case cudaErrorSystemNotReady: return CUDA_ERROR_SYSTEM_NOT_READY;
    case cudaErrorSystemDriverMismatch: return CUDA_ERROR_SYSTEM_DRIVER_MISMATCH;
    case cudaErrorCompatNotSupportedOnDevice: return CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE;
    case cudaErrorMpsConnectionFailed: return CUDA_ERROR_MPS_CONNECTION_FAILED;
    case cudaErrorMpsRpcFailure: return CUDA_ERROR_MPS_RPC_FAILURE;
    case cudaErrorMpsServerNotReady: return CUDA_ERROR_MPS_SERVER_NOT_READY;
    case cudaErrorMpsMaxClientsReached: return CUDA_ERROR_MPS_MAX_CLIENTS_REACHED;
    case cudaErrorMpsMaxConnectionsReached: return CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED;
    case cudaErrorMpsClientTerminated: return CUDA_ERROR_MPS_CLIENT_TERMINATED;
    case cudaErrorCdpNotSupported: return CUDA_ERROR_CDP_NOT_SUPPORTED;
    case cudaErrorCdpVersionMismatch: return CUDA_ERROR_CDP_VERSION_MISMATCH;
    case cudaErrorStreamCaptureUnsupported: return CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED;
    case cudaErrorStreamCaptureInvalidated: return CUDA_ERROR_STREAM_CAPTURE_INVALIDATED;
    case cudaErrorStreamCaptureMerge: return CUDA_ERROR_STREAM_CAPTURE_MERGE;
    case cudaErrorStreamCaptureUnmatched: return CUDA_ERROR_STREAM_CAPTURE_UNMATCHED;
    case cudaErrorStreamCaptureUnjoined: return CUDA_ERROR_STREAM_CAPTURE_UNJOINED;
    case cudaErrorStreamCaptureIsolation: return CUDA_ERROR_STREAM_CAPTURE_ISOLATION;
    case cudaErrorStreamCaptureImplicit: return CUDA_ERROR_STREAM_CAPTURE_IMPLICIT;
    case cudaErrorCapturedEvent: return CUDA_ERROR_CAPTURED_EVENT;
    case cudaErrorStreamCaptureWrongThread: return CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD;
    case cudaErrorTimeout: return CUDA_ERROR_TIMEOUT;
    case cudaErrorGraphExecUpdateFailure: return CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE;
    case cudaErrorExternalDevice: return CUDA_ERROR_EXTERNAL_DEVICE;
    case cudaErrorInvalidClusterSize: return CUDA_ERROR_INVALID_CLUSTER_SIZE;
#if CUDART_VERSION >= 12060
    case cudaErrorFunctionNotLoaded: return CUDA_ERROR_FUNCTION_NOT_LOADED;
    case cudaErrorInvalidResourceType: return CUDA_ERROR_INVALID_RESOURCE_TYPE;
    case cudaErrorInvalidResourceConfiguration: return CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION;
#endif
#if CUDART_VERSION >= 12080
    case cudaErrorContained: return CUDA_ERROR_CONTAINED;
    case cudaErrorTensorMemoryLeak: return CUDA_ERROR_TENSOR_MEMORY_LEAK;
#endif
    default: return CUDA_ERROR_UNKNOWN;
  }
}

// Entering a guard clears the previous message so a -1 seen by Rust never
// reads a stale what() from an earlier call on the same thread.
#define PEGAINFER_FFI_GUARD_BEGIN \
  pegainfer_ffi_set_last_error(""); \
  try {
#define PEGAINFER_FFI_GUARD_END(ret_on_throw)              \
  }                                                        \
  catch (const std::exception& e) {                        \
    pegainfer_ffi_set_last_error(e.what());                \
    return ret_on_throw;                                   \
  }                                                        \
  catch (...) {                                            \
    pegainfer_ffi_set_last_error("unknown C++ exception"); \
    return ret_on_throw;                                   \
  }
