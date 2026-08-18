// Cross-machine symmetric slabs for the K3 MegaMoE kernel: NVLink-fabric
// (`CU_MEM_HANDLE_TYPE_FABRIC`) allocation, export and import.
//
// An in-process EP group shares slabs by plain peer access — every rank's
// stream-ordered allocation is dereferenceable from every local device once
// the pool grants are open (`k3_mega_open_peer_access`). A cross-machine
// group cannot: a peer's slab lives in another process on another tray, and
// the only thing that travels is a 64-byte fabric handle. So a fleet rank's
// slab comes from the VMM allocator instead of the pool:
//
//  * `k3_mega_fabric_slab_alloc` — cuMemCreate with the FABRIC handle type on
//    the rank's own device, mapped into this process, zeroed, exported. The
//    mapping is granted to EVERY local device up front (the group's local
//    membership is not known here, exactly like the pool-grant story), which
//    is what lets the other local ranks' kernels dereference it.
//  * `k3_mega_fabric_slab_import` — the receiving side: import a peer's
//    handle, reserve + map, grant every local device. The pointer that comes
//    back goes into the launch's base-pointer table like any other; the
//    kernel cannot tell a fabric mapping from a peer-access one.
//
// Requires the NVLink domain to span the machines (NVL72) and the IMEX
// daemon on every node; both are runtime facts, so failures surface here as
// CUDA errors rather than at build time. Mappings are process-lifetime: an
// EP group dies as a fleet (a missing rank strands every peer inside a
// device barrier), so teardown is process exit and nothing here unmaps.

#include "../shared/ffi_guard.cuh"

#include <cuda.h>
#include <cstring>

namespace {

// One CUmemFabricHandle, as raw bytes on the FFI boundary.
constexpr size_t kFabricHandleBytes = sizeof(CUmemFabricHandle);
static_assert(kFabricHandleBytes == 64, "CUmemFabricHandle is expected to be 64 bytes");

CUmemAllocationProp fabric_prop(int device_ordinal) {
  CUmemAllocationProp prop;
  std::memset(&prop, 0, sizeof(prop));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_ordinal;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;
  return prop;
}

// Round `num_bytes` up to the allocation granularity the fabric prop wants.
CUresult fabric_granular_size(int device_ordinal, unsigned long long num_bytes,
                              size_t* out_size) {
  const CUmemAllocationProp prop = fabric_prop(device_ordinal);
  size_t granularity = 0;
  const CUresult err = cuMemGetAllocationGranularity(&granularity, &prop,
                                                     CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
  if (err != CUDA_SUCCESS) return err;
  if (granularity == 0) return CUDA_ERROR_INVALID_VALUE;
  *out_size = ((size_t)num_bytes + granularity - 1) / granularity * granularity;
  return CUDA_SUCCESS;
}

// Map `handle` somewhere and grant read/write to every local device. The
// grant covers all of them because any local rank's kernel may dereference
// any slab, and which devices form the group is the rendezvous's knowledge,
// not this allocator's.
CUresult map_for_all_devices(CUmemGenericAllocationHandle handle, size_t size,
                             CUdeviceptr* out_ptr) {
  CUdeviceptr ptr = 0;
  CUresult err = cuMemAddressReserve(&ptr, size, 0, 0, 0);
  if (err != CUDA_SUCCESS) return err;
  err = cuMemMap(ptr, size, 0, handle, 0);
  if (err != CUDA_SUCCESS) {
    (void)cuMemAddressFree(ptr, size);
    return err;
  }
  int device_count = 0;
  err = cuDeviceGetCount(&device_count);
  if (err == CUDA_SUCCESS && device_count > 0) {
    for (int device = 0; device < device_count && err == CUDA_SUCCESS; ++device) {
      CUmemAccessDesc access;
      std::memset(&access, 0, sizeof(access));
      access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
      access.location.id = device;
      access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
      err = cuMemSetAccess(ptr, size, &access, 1);
    }
  }
  if (err != CUDA_SUCCESS) {
    (void)cuMemUnmap(ptr, size);
    (void)cuMemAddressFree(ptr, size);
    return err;
  }
  *out_ptr = ptr;
  return CUDA_SUCCESS;
}

}  // namespace

extern "C" {

// Whether `device_ordinal` can allocate fabric-exportable memory at all
// (driver + IMEX support). A cheap preflight so a fleet launch on a machine
// without IMEX fails with this name in the message instead of an opaque
// cuMemCreate error.
CUresult k3_mega_fabric_supported(int device_ordinal, int* out_supported) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (out_supported == nullptr || device_ordinal < 0) return CUDA_ERROR_INVALID_VALUE;
  CUdevice device;
  CUresult err = cuDeviceGet(&device, device_ordinal);
  if (err != CUDA_SUCCESS) return err;
  int supported = 0;
  err = cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                             device);
  if (err != CUDA_SUCCESS) return err;
  *out_supported = supported;
  return CUDA_SUCCESS;
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

// Allocate `num_bytes` (rounded up to the fabric granularity) on
// `device_ordinal`, map it for every local device, zero it, and export its
// fabric handle into `out_handle` (64 bytes). The zeroing is synchronized
// before return: a peer that receives the handle is entitled to assume the
// memory behind it is live and zeroed.
CUresult k3_mega_fabric_slab_alloc(int device_ordinal, unsigned long long num_bytes,
                                   long long* out_ptr, unsigned char* out_handle) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (out_ptr == nullptr || out_handle == nullptr || device_ordinal < 0 || num_bytes == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t size = 0;
  CUresult err = fabric_granular_size(device_ordinal, num_bytes, &size);
  if (err != CUDA_SUCCESS) return err;

  const CUmemAllocationProp prop = fabric_prop(device_ordinal);
  CUmemGenericAllocationHandle handle;
  err = cuMemCreate(&handle, size, &prop, 0);
  if (err != CUDA_SUCCESS) return err;

  CUmemFabricHandle fabric;
  std::memset(&fabric, 0, sizeof(fabric));
  err = cuMemExportToShareableHandle(&fabric, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
  if (err != CUDA_SUCCESS) {
    (void)cuMemRelease(handle);
    return err;
  }

  CUdeviceptr ptr = 0;
  err = map_for_all_devices(handle, size, &ptr);
  // The mapping holds its own reference; releasing the handle here makes the
  // mapping the allocation's lifetime, which is the process's.
  (void)cuMemRelease(handle);
  if (err != CUDA_SUCCESS) return err;

  err = cuMemsetD8(ptr, 0, size);
  if (err == CUDA_SUCCESS) err = cuCtxSynchronize();
  if (err != CUDA_SUCCESS) return err;

  std::memcpy(out_handle, &fabric, kFabricHandleBytes);
  *out_ptr = (long long)ptr;
  return CUDA_SUCCESS;
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

// Import a peer's 64-byte fabric handle and map it for every local device.
// `num_bytes` is the peer's slab size before granularity rounding; the
// mapping is rounded the same way the exporter's allocation was.
CUresult k3_mega_fabric_slab_import(const unsigned char* handle, unsigned long long num_bytes,
                                    int device_ordinal, long long* out_ptr) {
  PEGAINFER_FFI_GUARD_BEGIN
  if (handle == nullptr || out_ptr == nullptr || device_ordinal < 0 || num_bytes == 0) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  size_t size = 0;
  CUresult err = fabric_granular_size(device_ordinal, num_bytes, &size);
  if (err != CUDA_SUCCESS) return err;

  CUmemFabricHandle fabric;
  std::memcpy(&fabric, handle, kFabricHandleBytes);
  CUmemGenericAllocationHandle imported;
  err = cuMemImportFromShareableHandle(&imported, &fabric, CU_MEM_HANDLE_TYPE_FABRIC);
  if (err != CUDA_SUCCESS) return err;

  CUdeviceptr ptr = 0;
  err = map_for_all_devices(imported, size, &ptr);
  (void)cuMemRelease(imported);
  if (err != CUDA_SUCCESS) return err;

  *out_ptr = (long long)ptr;
  return CUDA_SUCCESS;
  PEGAINFER_FFI_GUARD_END(CUDA_ERROR_UNKNOWN)
}

}  // extern "C"
