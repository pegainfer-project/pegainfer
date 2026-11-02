"""AOT-compile the Gemma 4 TileLang attention kernels into one CUDA file.

`pegainfer-kernels/build.rs` runs this under the `gemma4` feature and hands
the result to nvcc. It prints the same `KEY=VALUE` manifest every TileLang
family prints, and mirrors it into `manifest.txt` so a build host can consume
a pre-generated directory without TileLang installed.

Four launchers come out: the global family's prefill over one lowered kernel
and its split-KV decode over two (a partial pass and a merge) behind one call,
and the sliding family's windowed prefill and decode. Each launcher is a spec
-- its C parameters, its refusals, and for each kernel it launches how the
lowered entry's parameters bind to those C arguments -- and the rendering is
one routine over the specs, so another kernel is a spec and not a copy.

What makes this family different from the K3 one is the lowering: the key and
query loads become bulk copies, so TileLang passes those two tensors as TMA
descriptors instead of pointers and adds a producer warpgroup to the block.
The descriptors are the launcher's to build, and every parameter it needs is
a constant in the lowered host stub — recovered here rather than assumed, so
a codegen change fails the build instead of encoding a stale descriptor.

The declared tensor extents are the serving arena's maxima. They reach the
device code only as bounds guards, so any smaller step passes them; the two
TMA tensors carry their real extents in the descriptors, which the launcher
builds per call from the arguments it is given.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import tilelang
import tilelang_defs as defs
from tilelang.env import CUTLASS_INCLUDE_DIR, TILELANG_TEMPLATE_PATH

ENTRY_SYMBOL = "main_kernel"
KERNEL_MARKER = 'extern "C" __global__ void'
LAUNCHER = "gemma4_hd512_prefill_varlen"
DECODE_LAUNCHER = "gemma4_hd512_decode_split_kv"
CU_STEM = "gemma4_hd512_prefill"
# TileLang names every entry point `main_kernel` with external C linkage, so
# each emitted body is renamed before it can collide with another's.
KERNEL_SYMBOL = f"{CU_STEM}_kernel"
DECODE_PARTIAL_SYMBOL = "gemma4_hd512_decode_partial_kernel"
DECODE_MERGE_GROUPS_SYMBOL = "gemma4_hd512_decode_merge_groups_kernel"
DECODE_MERGE_SYMBOL = "gemma4_hd512_decode_merge_kernel"
LOCAL_DECODE_LAUNCHER = "gemma4_hd256_decode_window"
LOCAL_PREFILL_LAUNCHER = "gemma4_hd256_prefill_window"
LOCAL_PREFILL_SYMBOL = "gemma4_hd256_prefill_window_kernel"
LOCAL_DECODE_PARTIAL_SYMBOL = "gemma4_hd256_decode_partial_kernel"
LOCAL_DECODE_MERGE_GROUPS_SYMBOL = "gemma4_hd256_decode_merge_groups_kernel"
LOCAL_DECODE_MERGE_SYMBOL = "gemma4_hd256_decode_merge_kernel"

# Each launcher's C parameters, without the trailing stream the consumer adds.
# The emitted definition and the manifest line both come from here, so the
# stub the build script writes when a kernel is absent cannot drift from the
# real one: C has no mangling, and a drifted pair would link silently.
LAUNCHER_PARAMS = [
    ("const void*", "q"),
    ("const void*", "kv"),
    ("const int*", "page_indices"),
    ("const int*", "page_indptr"),
    ("const int*", "q_indptr"),
    ("const int*", "host_q_indptr"),
    ("const int*", "last_page_len"),
    ("void*", "out"),
    ("int", "batch"),
    ("int", "q_rows"),
    ("int", "pool_rows"),
    ("int", "rows_per_page"),
    ("int", "layer_row"),
    ("int", "page_size"),
    ("int", "fold_rotary"),
    ("int", "num_qo_heads"),
    ("int", "num_kv_heads"),
    ("float", "sm_scale"),
]

# The decode's: the plan's per-slot arrays and workspace, the decode rows'
# offset into the step's buffers, the extents, pool geometry and row format,
# and the chunk the tile indices count in. The order is the stub tier's, in
# build.rs.
DECODE_LAUNCHER_PARAMS = [
    ("const void*", "q"),
    ("const void*", "kv"),
    ("const int*", "page_indices"),
    ("const int*", "page_indptr"),
    ("const int*", "last_page_len"),
    ("const int*", "request_indices"),
    ("const int*", "kv_tile_indices"),
    ("const unsigned char*", "valid_mask"),
    ("const int*", "o_indptr"),
    ("void*", "tmp_v"),
    ("float*", "tmp_s"),
    ("void*", "out"),
    ("int", "batch"),
    ("int", "padded_slots"),
    ("int", "row_offset"),
    ("int", "q_rows"),
    ("int", "pool_rows"),
    ("int", "rows_per_page"),
    ("int", "layer_row"),
    ("int", "page_size"),
    ("int", "fold_rotary"),
    ("int", "chunk_tokens"),
    ("int", "num_qo_heads"),
    ("int", "num_kv_heads"),
    ("float", "sm_scale"),
]

# The sliding family's prefill: the global prefill's parameters without a
# row format (its pool is split K|V) and with the window as a key distance.
LOCAL_PREFILL_LAUNCHER_PARAMS = [
    ("const void*", "q"),
    ("const void*", "kv"),
    ("const int*", "page_indices"),
    ("const int*", "page_indptr"),
    ("const int*", "q_indptr"),
    ("const int*", "host_q_indptr"),
    ("const int*", "last_page_len"),
    ("void*", "out"),
    ("int", "batch"),
    ("int", "q_rows"),
    ("int", "pool_rows"),
    ("int", "rows_per_page"),
    ("int", "layer_row"),
    ("int", "page_size"),
    ("int", "window_left"),
    ("int", "num_qo_heads"),
    ("int", "num_kv_heads"),
    ("float", "sm_scale"),
]

# The sliding family's decode: the same plan arrays and workspace as the
# global decode, no row format (its pool is split K|V), and the window as a
# key distance.
LOCAL_DECODE_LAUNCHER_PARAMS = [
    ("const void*", "q"),
    ("const void*", "kv"),
    ("const int*", "page_indices"),
    ("const int*", "page_indptr"),
    ("const int*", "last_page_len"),
    ("const int*", "request_indices"),
    ("const int*", "kv_tile_indices"),
    ("const unsigned char*", "valid_mask"),
    ("const int*", "o_indptr"),
    ("void*", "tmp_v"),
    ("float*", "tmp_s"),
    ("void*", "out"),
    ("int", "batch"),
    ("int", "padded_slots"),
    ("int", "row_offset"),
    ("int", "q_rows"),
    ("int", "pool_rows"),
    ("int", "rows_per_page"),
    ("int", "layer_row"),
    ("int", "page_size"),
    ("int", "chunk_tokens"),
    ("int", "window_left"),
    ("int", "num_qo_heads"),
    ("int", "num_kv_heads"),
    ("float", "sm_scale"),
]

# TileLang's `debug.h` *defines* `debug_print_msg` and the `uint16_t`
# specialization of `debug_print_buffer_value` with external linkage, so every
# translation unit that includes it exports the same two symbols. This kernel
# calls neither, and a binary that also links a K3 family would get duplicate
# definitions, so this unit is given privately named copies.
DEBUG_HEADER = "#include <tl_templates/cuda/debug.h>"
DEBUG_HELPERS = ("debug_print_msg", "debug_print_buffer_value")

# The model's global-attention geometry. Shapes are compile dimensions, so
# these are the kernel's identity, not run-time inputs.
HEADS = 32
GROUPS = 8
HEAD_DIM = 512
PAGE_SIZE = 64
# The global family's proportional RoPE rotates this many columns of the
# head, and the folded pool format keeps only these of K: a compile
# dimension of the folded bodies, like the head itself. The split bodies
# are lowered for a row format of zero.
ROTARY = 128
# The sliding family: 32 query heads over 16 key/value heads at head dim
# 256, paged at 64 rows -- the prefill's key tile, so its load is one copy
# -- and windowed at 1024 keys. Its decode reads 64-token chunks as one
# page a tile.
LOCAL_HEADS = 32
LOCAL_GROUPS = 2
LOCAL_HEAD_DIM = 256
LOCAL_PAGE_SIZE = 64
LOCAL_TILE_PAGES = 1
LOCAL_DECODE_CHUNK_TOKENS = 64
LOCAL_WINDOW = 1024

# The serving arena's maxima, in the units each tensor is indexed in. A step
# is always smaller; see the module docstring.
CEILING = 262144
SLOTS = 16
# A step's requests, the prompts being admitted and the rows decoding beside
# them, each hold a slot, so a prefill plan never names more than the slots.
MAX_BATCH = SLOTS
Q_ROWS = CEILING + defs.BLOCK_M
POOL_PAGES = SLOTS * (CEILING // PAGE_SIZE) + 1
PAGE_TABLE_LEN = SLOTS * (CEILING // PAGE_SIZE)
# The pool is addressed in rows of [kv_heads, head_dim]; one page holds every
# layer's K and V, and the deepest checkpoint this line serves sets the bound.
MAX_LAYERS = 64
POOL_ROWS = POOL_PAGES * MAX_LAYERS * 2 * PAGE_SIZE
# The decode's own maxima: its requests are the step's decode rows, which the
# split factor may double, and its slots are those times the chunk bound.
MAX_DECODE_ROWS = 2 * SLOTS
DECODE_CHUNK_TOKENS = 256
MAX_SLOTS = MAX_DECODE_ROWS * (CEILING // DECODE_CHUNK_TOKENS)
# The sliding family's slots: a resident window holds at most the window
# plus a page, and every decode row has one.
LOCAL_MAX_SLOTS = MAX_DECODE_ROWS * (
    (LOCAL_WINDOW + LOCAL_PAGE_SIZE + LOCAL_DECODE_CHUNK_TOKENS - 1) // LOCAL_DECODE_CHUNK_TOKENS
)
# The launchers take every row and slot count as C ints; a configuration that
# outgrew one would index past the guards rather than fail.
assert Q_ROWS < 2**31, Q_ROWS
assert POOL_ROWS < 2**31, POOL_ROWS
assert MAX_SLOTS < 2**31, MAX_SLOTS

# Past 48 KiB a kernel has to opt into its dynamic shared memory per symbol.
MAX_STATIC_SMEM = 48 * 1024
# SM90's per-block ceiling. A recovered size above it would launch-fail.
MAX_DYNAMIC_SMEM = 227 * 1024

TENSORMAP_BUILDER = "__tvm_tensormap_create_tiled"

# A pass config can reach nvcc's command line, which TileLang's JIT passes and
# an AOT build would not: without --use_fast_math the object sits half a ULP
# from the gated kernel. Every config is classified here or generation stops.
PASS_CONFIG_NVCC_FLAG = {tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: "--use_fast_math"}

# TVM's tensormap codes, mapped to the driver enums the launcher passes. Every
# code the stub can carry is listed; an unmapped one is a codegen change.
TENSORMAP_DTYPE = {9: "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16"}
TENSORMAP_INTERLEAVE = {0: "CU_TENSOR_MAP_INTERLEAVE_NONE"}
TENSORMAP_SWIZZLE = {
    0: "CU_TENSOR_MAP_SWIZZLE_NONE",
    1: "CU_TENSOR_MAP_SWIZZLE_32B",
    2: "CU_TENSOR_MAP_SWIZZLE_64B",
    3: "CU_TENSOR_MAP_SWIZZLE_128B",
}
TENSORMAP_L2 = {
    0: "CU_TENSOR_MAP_L2_PROMOTION_NONE",
    1: "CU_TENSOR_MAP_L2_PROMOTION_L2_64B",
    2: "CU_TENSOR_MAP_L2_PROMOTION_L2_128B",
    3: "CU_TENSOR_MAP_L2_PROMOTION_L2_256B",
}
TENSORMAP_OOB = {0: "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE"}

_SLOT_INT = re.compile(
    r"\(\(\(TVMFFIAny\*\)stack_ffi_any\)\[(\d+)\]\.v_int64\) = \(\(int64_t\)(-?\d+)\);"
)
_SLOT_PTR = re.compile(r"\(\(\(TVMFFIAny\*\)stack_ffi_any\)\[(\d+)\]\.v_ptr\) = (\w+);")
_PACKED_CALL = re.compile(
    r"TVMFFIFunctionCall\((\w+?)_packed, \(TVMFFIAny\*\) stack_ffi_any, (\d+),"
)


@dataclass(frozen=True)
class TensorMap:
    """One recovered `cuTensorMapEncodeTiled` call, still in TVM's spelling."""

    name: str
    dtype: int
    rank: int
    tensor: str
    dims: tuple[int, ...]
    strides: tuple[int, ...]
    box: tuple[int, ...]
    element_strides: tuple[int, ...]
    interleave: int
    swizzle: int
    l2_promotion: int
    oob_fill: int


def read_host_stub(kernel, needs_descriptors: bool = True) -> tuple[list[TensorMap], int, int]:
    """Recover the descriptor builds, the block width and the dynamic smem.

    TileLang bakes the launch into the host stub as a packed-call argument
    stack and exposes no accessor for it, hence the parse. Slots persist
    across calls, so only the values a call writes itself are its own; the
    entry call's grid and scalars are run-time expressions and are read from
    its arguments instead, which is why they never appear here.
    """
    slots: dict[int, object] = {}
    maps: list[TensorMap] = []
    launch: list[object] | None = None
    for line in kernel.get_host_source().splitlines():
        match = _SLOT_INT.search(line)
        if match:
            slots[int(match.group(1))] = int(match.group(2))
            continue
        match = _SLOT_PTR.search(line)
        if match:
            slots[int(match.group(1))] = match.group(2)
            continue
        match = _PACKED_CALL.search(line)
        if not match:
            continue
        callee, count = match.group(1), int(match.group(2))
        args = [slots.get(i) for i in range(count)]
        if callee == TENSORMAP_BUILDER:
            maps.append(parse_tensormap(args))
        elif callee == ENTRY_SYMBOL:
            launch = args

    if needs_descriptors and not maps:
        raise RuntimeError(
            "the lowering built no TMA descriptor; this launcher binds the "
            "kernel's loads as descriptors, so the parameter list is wrong now"
        )
    if not needs_descriptors and maps:
        raise RuntimeError(
            f"the lowering built {len(maps)} TMA descriptor(s) for a kernel the "
            "launcher binds by pointer alone; its loads changed"
        )
    if launch is None:
        raise RuntimeError("could not recover the entry call from the host stub")
    # The tail is block x/y/z then the dynamic smem, which TileLang omits when
    # it is zero: (block, 1, 1, smem) for a kernel with shared buffers and
    # (block, 1, 1) for one without. Anything else is a codegen change.
    if list(launch[-3:-1]) == [1, 1] and isinstance(launch[-1], int) and launch[-1] > 1:
        tail = list(launch[-4:])
        block, smem = tail[0], tail[3]
    elif list(launch[-2:]) == [1, 1]:
        tail = list(launch[-3:])
        block, smem = tail[0], 0
    else:
        raise RuntimeError(
            f"the launch tail {launch[-4:]} is neither (block, 1, 1, dynamic "
            "shared) nor (block, 1, 1); the lowering changed its geometry"
        )
    if not isinstance(block, int) or not isinstance(smem, int):
        raise TypeError(f"launch geometry has non-constant entries: {tail}")
    if smem > MAX_DYNAMIC_SMEM:
        raise RuntimeError(
            f"the lowering wants {smem} B of dynamic shared memory, past the "
            f"{MAX_DYNAMIC_SMEM} B a block can be given"
        )
    return maps, block, smem


def parse_tensormap(args: list) -> TensorMap:
    """Split one builder call's flat argument list by its recovered rank."""
    name, dtype, rank, tensor = args[0], args[1], args[2], args[3]
    if not isinstance(rank, int) or rank < 2:
        raise RuntimeError(f"descriptor {name} has a non-constant rank: {rank}")
    at = 4
    dims = tuple(args[at : at + rank])
    at += rank
    # The builder carries one stride per dimension, innermost first, and the
    # innermost one is the element size. The driver API takes the rest.
    strides = tuple(args[at : at + rank])
    at += rank
    box = tuple(args[at : at + rank])
    at += rank
    element_strides = tuple(args[at : at + rank])
    at += rank
    interleave, swizzle, l2_promotion, oob_fill = args[at : at + 4]
    values = (
        *dims,
        *strides,
        *box,
        *element_strides,
        interleave,
        swizzle,
        l2_promotion,
        oob_fill,
    )
    if any(not isinstance(value, int) for value in values):
        raise RuntimeError(f"descriptor {name} has non-constant parameters: {values}")
    return TensorMap(
        name=str(name),
        dtype=dtype,
        rank=rank,
        tensor=str(tensor),
        dims=dims,
        strides=strides,
        box=box,
        element_strides=element_strides,
        interleave=interleave,
        swizzle=swizzle,
        l2_promotion=l2_promotion,
        oob_fill=oob_fill,
    )


def enum_of(table: dict[int, str], code: int, what: str) -> str:
    if code not in table:
        raise RuntimeError(f"the lowering asked for an unmapped {what}: {code}")
    return table[code]


def runtime_dim(tmap: TensorMap, declared: int, label: str) -> int:
    """Which of a descriptor's dimensions the launcher fills in per call.

    The rows of a packed q buffer and of the pool are run-time quantities, and
    they enter the device code only through the descriptor, so the launcher
    substitutes them here. Matching on the declared value keeps the position
    tied to the lowering rather than to a hand-kept index.
    """
    hits = [i for i, dim in enumerate(tmap.dims) if dim == declared]
    if len(hits) != 1:
        raise RuntimeError(
            f"{label}: expected exactly one dimension equal to {declared} in "
            f"{tmap.dims}, found {len(hits)}"
        )
    return hits[0]


# The launcher's own name for each thing the stub names. A descriptor the
# lowering grew, or a tensor it renamed, has no binding here and fails
# generation rather than emitting a launcher that does not compile.
DESCRIPTOR_VAR = {"Q_desc": "q_desc", "KV_desc": "kv_desc"}
TENSOR_ARG = {"Q": "q", "KV": "kv"}


def render_descriptor(
    tmap: TensorMap,
    rows_expr: str,
    rows_at: int,
    descriptor_var: dict[str, str] = DESCRIPTOR_VAR,
    tensor_arg: dict[str, str] = TENSOR_ARG,
) -> str:
    """The launcher body that encodes one descriptor."""
    if tmap.name not in descriptor_var:
        raise RuntimeError(f"no launcher variable for descriptor {tmap.name}")
    if tmap.tensor not in tensor_arg:
        raise RuntimeError(f"no launcher argument for tensor {tmap.tensor}")
    dims = [str(dim) for dim in tmap.dims]
    dims[rows_at] = rows_expr
    return (
        f"  {{\n"
        f"    const cuuint64_t dims[{tmap.rank}] = {{{', '.join(dims)}}};\n"
        f"    const cuuint64_t strides[{tmap.rank - 1}] = "
        f"{{{', '.join(str(s) for s in tmap.strides[1:])}}};\n"
        f"    const cuuint32_t box[{tmap.rank}] = "
        f"{{{', '.join(str(b) for b in tmap.box)}}};\n"
        f"    const cuuint32_t element_strides[{tmap.rank}] = "
        f"{{{', '.join(str(e) for e in tmap.element_strides)}}};\n"
        f"    const CUresult encoded = encode(\n"
        f"        &{descriptor_var[tmap.name]}, "
        f"{enum_of(TENSORMAP_DTYPE, tmap.dtype, 'dtype')}, {tmap.rank},\n"
        f"        const_cast<void*>({tensor_arg[tmap.tensor]}), dims, strides, box, "
        f"element_strides,\n"
        f"        {enum_of(TENSORMAP_INTERLEAVE, tmap.interleave, 'interleave')},\n"
        f"        {enum_of(TENSORMAP_SWIZZLE, tmap.swizzle, 'swizzle')},\n"
        f"        {enum_of(TENSORMAP_L2, tmap.l2_promotion, 'L2 promotion')},\n"
        f"        {enum_of(TENSORMAP_OOB, tmap.oob_fill, 'out-of-bounds fill')});\n"
        f"    if (encoded != CUDA_SUCCESS) {{\n"
        f"      return static_cast<int>(cudaErrorInvalidValue);\n"
        f"    }}\n"
        f"  }}\n"
    )


ELEMENT_BYTES = {"CU_TENSOR_MAP_DATA_TYPE_BFLOAT16": 2}

LAUNCHER_HEAD = """
// Hand-written launcher. The key and query loads lower to bulk copies, so the
// entry takes TMA descriptors for those two and plain pointers for the rest;
// every descriptor parameter below is recovered from the lowered host stub.

namespace {

// cuTensorMapEncodeTiled is a driver entry point, resolved once and cached.
CUresult encode(CUtensorMap* map, CUtensorMapDataType dtype, cuuint32_t rank,
                void* tensor, const cuuint64_t* dims, const cuuint64_t* strides,
                const cuuint32_t* box, const cuuint32_t* element_strides,
                CUtensorMapInterleave interleave, CUtensorMapSwizzle swizzle,
                CUtensorMapL2promotion l2_promotion,
                CUtensorMapFloatOOBfill oob_fill) {
  using Fn = CUresult (*)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
                          const cuuint64_t*, const cuuint64_t*, const cuuint32_t*,
                          const cuuint32_t*, CUtensorMapInterleave,
                          CUtensorMapSwizzle, CUtensorMapL2promotion,
                          CUtensorMapFloatOOBfill);
  static Fn fn = [] {
    void* entry = nullptr;
    cudaDriverEntryPointQueryResult found;
    if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &entry,
                                cudaEnableDefault, &found) != cudaSuccess ||
        found != cudaDriverEntryPointSuccess) {
      return static_cast<Fn>(nullptr);
    }
    return reinterpret_cast<Fn>(entry);
  }();
  if (fn == nullptr) {
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  return fn(map, dtype, rank, tensor, dims, strides, box, element_strides,
            interleave, swizzle, l2_promotion, oob_fill);
}

}  // namespace
"""


@dataclass(frozen=True)
class Kernel:
    """One lowered prim_func and how a launcher binds it.

    `descriptor_var` names the C variable each lowered descriptor is encoded
    into and `tensor_arg` the C argument it is built over; `runtime_rows`
    says which declared extent of each descriptor the launcher replaces with
    a per-call count. `bind` maps every other entry parameter to a C
    expression. A parameter the lowering has that is bound nowhere fails
    generation, so the binding cannot silently fall behind the kernel.
    `when` is a C condition over the launcher's parameters under which this
    kernel's descriptors are built and it is launched; empty runs it on
    every call.
    """

    symbol: str
    build: Callable[[str], object]
    descriptor_var: dict[str, str]
    tensor_arg: dict[str, str]
    runtime_rows: dict[str, tuple[int, str]]
    bind: dict[str, str]
    grid: str
    opt_in_var: str
    when: str = ""


@dataclass(frozen=True)
class Launcher:
    """One `extern "C"` entry: its C parameters, its refusals, and the
    kernels it launches in order."""

    name: str
    params: list[tuple[str, str]]
    bounds: str
    prelude: str
    kernels: list[Kernel]


@dataclass(frozen=True)
class Lowered:
    kernel: Kernel
    maps: list[TensorMap]
    block: int
    smem: int
    order: list[str]
    preamble: str
    body: str


def lower(kernel: Kernel, arch: str) -> Lowered:
    """Compile one kernel and recover what its launcher needs from it."""
    compiled = kernel.build(arch)
    source = compiled.get_kernel_source()
    maps, block, smem = read_host_stub(compiled, bool(kernel.descriptor_var))
    check_strides(maps)
    if "decode_merge_groups" in kernel.symbol:
        check_in_place_merge_barrier(kernel.symbol, source)
    order = entry_parameter_order(source)
    preamble, body = split_source(source)
    if body.count(ENTRY_SYMBOL) != 2:
        raise RuntimeError(
            f"{kernel.symbol}: expected exactly two {ENTRY_SYMBOL} occurrences"
        )
    return Lowered(
        kernel=kernel,
        maps=maps,
        block=block,
        smem=smem,
        order=order,
        preamble=preamble,
        body=body.replace(ENTRY_SYMBOL, kernel.symbol),
    )


def render_launcher(launcher: Launcher, lowered: list[Lowered]) -> str:
    """The `extern "C"` entry `pegainfer-kernels` links against."""
    declarations = []
    stages = []
    for low in lowered:
        kernel = low.kernel
        bodies = []
        opt_ins = []
        by_name = {tmap.name: tmap for tmap in low.maps}
        if set(by_name) != set(kernel.descriptor_var):
            raise RuntimeError(
                f"{kernel.symbol}: expected descriptors "
                f"{sorted(kernel.descriptor_var)}, got {sorted(by_name)}"
            )
        declarations.extend(
            f"  alignas(64) CUtensorMap {var};\n"
            for var in kernel.descriptor_var.values()
        )
        for name, (declared, rows_expr) in kernel.runtime_rows.items():
            bodies.append(
                render_descriptor(
                    by_name[name],
                    rows_expr,
                    runtime_dim(by_name[name], declared, name),
                    kernel.descriptor_var,
                    kernel.tensor_arg,
                )
            )
        if low.smem > MAX_STATIC_SMEM:
            opt_ins.append(
                f"  // Past 48 KiB the kernel has to opt in; once per symbol.\n"
                f"  static const cudaError_t {kernel.opt_in_var} = cudaFuncSetAttribute(\n"
                f"      reinterpret_cast<const void*>({kernel.symbol}),\n"
                f"      cudaFuncAttributeMaxDynamicSharedMemorySize, {low.smem});\n"
                f"  if ({kernel.opt_in_var} != cudaSuccess) {{\n"
                f"    return static_cast<int>({kernel.opt_in_var});\n"
                f"  }}\n"
            )
        bound = {**kernel.descriptor_var, **kernel.bind}
        missing = [name for name in low.order if name not in bound]
        if missing:
            raise RuntimeError(
                f"{kernel.symbol}: the entry point grew parameters this launcher "
                f"does not bind: {missing}"
            )
        args = ",\n      ".join(bound[name] for name in low.order)
        launch = (
            f"  {kernel.symbol}<<<{kernel.grid}, dim3({low.block}), {low.smem}, stream>>>(\n"
            f"      {args});\n"
        )
        stage = "".join(bodies) + "".join(opt_ins) + launch
        if kernel.when:
            # A body lowered for one row format runs only on a call that
            # names it; the launcher's refusals hold the format to one of
            # the bodies it has.
            inner = "".join(f"  {line}\n" for line in stage.splitlines())
            stage = f"  if ({kernel.when}) {{\n{inner}  }}\n"
        stages.append(stage)

    signature = ", ".join(f"{kind} {name}" for kind, name in launcher.params)
    return (
        f'extern "C" int {launcher.name}(\n'
        f"    {signature},\n"
        f"    cudaStream_t stream) {{\n"
        f"{launcher.bounds}"
        f"{launcher.prelude}"
        f"{''.join(declarations)}"
        f"{''.join(stages)}"
        f"  return static_cast<int>(cudaGetLastError());\n"
        f"}}\n"
    )


def entry_parameter_order(source: str) -> list[str]:
    """Parameter names of the generated entry point, in its own order."""
    marker = source.index(f"{KERNEL_MARKER} {ENTRY_SYMBOL}(")
    open_at = source.index("(", marker)
    close_at = source.index(")", open_at)
    names = []
    for param in source[open_at + 1 : close_at].split(","):
        names.append(param.strip().split()[-1].lstrip("*"))
    return names


def split_source(source: str) -> tuple[str, str]:
    """Split `get_kernel_source()` into (include preamble, kernel bodies)."""
    marker = source.index(KERNEL_MARKER)
    return source[:marker], source[marker:]


def isolate_debug_helpers(preamble: str) -> str:
    """Rename `debug.h`'s externally linked helpers for this unit."""
    if DEBUG_HEADER not in preamble:
        return preamble
    renames = "".join(f"#define {name} {CU_STEM}_{name}\n" for name in DEBUG_HELPERS)
    restores = "".join(f"#undef {name}\n" for name in DEBUG_HELPERS)
    return preamble.replace(
        DEBUG_HEADER, f"{renames}{DEBUG_HEADER}\n{restores}".rstrip("\n")
    )


def vendor_includes(out_dir: Path) -> tuple[Path, Path]:
    """Copy the header roots in, so the directory stands on its own."""
    copied = []
    for source, name in (
        (TILELANG_TEMPLATE_PATH, "tilelang"),
        (CUTLASS_INCLUDE_DIR, "cutlass"),
    ):
        destination = out_dir / "include" / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
        copied.append(destination)
    return copied[0], copied[1]


def required_nvcc_flags() -> list[str]:
    """The flags TileLang's JIT would pass for this kernel's pass configs."""
    flags = []
    for key, enabled in defs.PASS_CONFIGS.items():
        if key not in PASS_CONFIG_NVCC_FLAG:
            raise RuntimeError(
                f"pass config {key} is not classified: say whether it reaches "
                "nvcc's command line, or the generated object will not match "
                "the kernel that was gated"
            )
        flag = PASS_CONFIG_NVCC_FLAG[key]
        if enabled and flag is not None:
            flags.append(flag)
    return flags


def build_prefill(arch: str, fold_rotary: int = 0):
    """Lower for the arch the objects will be assembled for.

    Generation must not depend on a GPU being visible to the build host --
    containers routinely have none, and TileLang then lowers for its own
    default, which nvcc rejects outright. So the arch is always passed.
    """
    return tilelang.compile(
        defs.prefill_varlen(
            HEADS,
            GROUPS,
            HEAD_DIM,
            PAGE_SIZE,
            MAX_BATCH,
            Q_ROWS,
            POOL_ROWS,
            PAGE_TABLE_LEN,
            fold_rotary=fold_rotary,
        ),
        target={"kind": "cuda", "arch": arch},
        pass_configs=defs.PASS_CONFIGS,
    )


def build_prefill_folded(arch: str):
    return build_prefill(arch, ROTARY)


def build_decode_partial(arch: str, fold_rotary: int = 0):
    return tilelang.compile(
        defs.decode_partial(
            HEADS,
            GROUPS,
            HEAD_DIM,
            PAGE_SIZE,
            Q_ROWS,
            MAX_DECODE_ROWS,
            MAX_SLOTS,
            POOL_ROWS,
            PAGE_TABLE_LEN,
            stages=defs.decode_stages(fold_rotary),
            fold_rotary=fold_rotary,
        ),
        target={"kind": "cuda", "arch": arch},
        pass_configs=defs.PASS_CONFIGS,
    )


def build_decode_merge_groups(arch: str):
    return tilelang.compile(
        defs.decode_merge_groups(HEADS, HEAD_DIM, MAX_DECODE_ROWS, MAX_SLOTS),
        target={"kind": "cuda", "arch": arch},
        pass_configs=defs.PASS_CONFIGS,
    )


def build_decode_partial_folded(arch: str):
    return build_decode_partial(arch, ROTARY)


def build_decode_merge(arch: str):
    return tilelang.compile(
        defs.decode_merge(HEADS, HEAD_DIM, Q_ROWS, MAX_DECODE_ROWS, MAX_SLOTS),
        target={"kind": "cuda", "arch": arch},
        pass_configs=defs.PASS_CONFIGS,
    )


def build_local_prefill(arch: str):
    return tilelang.compile(
        defs.prefill_window(
            LOCAL_HEADS,
            LOCAL_GROUPS,
            LOCAL_HEAD_DIM,
            LOCAL_PAGE_SIZE,
            MAX_BATCH,
            Q_ROWS,
            POOL_ROWS,
            PAGE_TABLE_LEN,
        ),
        target={"kind": "cuda", "arch": arch},
        pass_configs=defs.PASS_CONFIGS,
    )


def build_local_decode_partial(arch: str):
    return tilelang.compile(
        defs.decode_window_partial(
            LOCAL_HEADS,
            LOCAL_GROUPS,
            LOCAL_HEAD_DIM,
            LOCAL_PAGE_SIZE,
            LOCAL_TILE_PAGES,
            Q_ROWS,
            MAX_DECODE_ROWS,
            LOCAL_MAX_SLOTS,
            POOL_ROWS,
            PAGE_TABLE_LEN,
        ),
        target={"kind": "cuda", "arch": arch},
        pass_configs=defs.PASS_CONFIGS,
    )


def build_local_decode_merge_groups(arch: str):
    return tilelang.compile(
        defs.decode_merge_groups(LOCAL_HEADS, LOCAL_HEAD_DIM, MAX_DECODE_ROWS, LOCAL_MAX_SLOTS),
        target={"kind": "cuda", "arch": arch},
        pass_configs=defs.PASS_CONFIGS,
    )


def build_local_decode_merge(arch: str):
    return tilelang.compile(
        defs.decode_merge(LOCAL_HEADS, LOCAL_HEAD_DIM, Q_ROWS, MAX_DECODE_ROWS, LOCAL_MAX_SLOTS),
        target={"kind": "cuda", "arch": arch},
        pass_configs=defs.PASS_CONFIGS,
    )


# Past any of these the body drops a tail or reads the wrong rows and hands
# back plausible numbers, so the launcher refuses rather than truncates.
PREFILL_BOUNDS = (
    f"  if (q_rows > {Q_ROWS} || pool_rows > {POOL_ROWS} || batch > {MAX_BATCH}\n"
    f"      || page_size != {PAGE_SIZE} || num_qo_heads != {HEADS}\n"
    f"      || num_kv_heads != {HEADS // GROUPS}\n"
    f"      || (fold_rotary != 0 && fold_rotary != {ROTARY})) {{\n"
    f"    return static_cast<int>(cudaErrorInvalidValue);\n"
    f"  }}\n"
)
# The grid is computed here, from the sum the body re-walks to find its
# owner: a caller's own copy disagreeing is silent both ways, too large and
# CTAs spin up to exit, too small and a request's tail never runs.
PREFILL_GRID = (
    f"  int total_ctas = 0;\n"
    f"  for (int request = 0; request < batch; ++request) {{\n"
    f"    const int rows = host_q_indptr[request + 1] - host_q_indptr[request];\n"
    f"    const int tiles = (rows + {defs.BLOCK_M - 1}) / {defs.BLOCK_M};\n"
    f"    total_ctas += ((tiles + {defs.QBLK - 1}) / {defs.QBLK})"
    f" * {defs.QBLK} * {HEADS};\n"
    f"  }}\n"
    f"  if (total_ctas == 0) {{\n"
    f"    // A step with no prompt rows is a real state, not an error.\n"
    f"    return static_cast<int>(cudaSuccess);\n"
    f"  }}\n"
)

PREFILL_BIND = {
    "Output": "reinterpret_cast<bfloat16_t*>(out)",
    "PageIndices": "page_indices",
    "PageIndptr": "page_indptr",
    "QIndptr": "q_indptr",
    "LastPageLen": "last_page_len",
    "sm_scale": "sm_scale",
    "total_ctas": "total_ctas",
    "rows_per_page": "rows_per_page",
    "layer_row": "layer_row",
}
PREFILL_RUNTIME_ROWS = {
    "Q_desc": (Q_ROWS, "static_cast<cuuint64_t>(q_rows)"),
    "KV_desc": (POOL_ROWS, "static_cast<cuuint64_t>(pool_rows)"),
}

# One entry, two bodies: the split rows' and the folded rows', each lowered
# for its own row width and chosen by the format the call names. The
# folded body's descriptors are its own, since their inner extents differ.
PREFILL = Launcher(
    name=LAUNCHER,
    params=LAUNCHER_PARAMS,
    bounds=PREFILL_BOUNDS,
    prelude=PREFILL_GRID,
    kernels=[
        Kernel(
            symbol=KERNEL_SYMBOL,
            build=build_prefill,
            descriptor_var=DESCRIPTOR_VAR,
            tensor_arg=TENSOR_ARG,
            runtime_rows=PREFILL_RUNTIME_ROWS,
            bind=PREFILL_BIND,
            grid="dim3(total_ctas)",
            opt_in_var="opt_in",
            when="fold_rotary == 0",
        ),
        Kernel(
            symbol=f"{CU_STEM}_folded_kernel",
            build=build_prefill_folded,
            descriptor_var={"Q_desc": "q_desc_folded", "KV_desc": "kv_desc_folded"},
            tensor_arg=TENSOR_ARG,
            runtime_rows=PREFILL_RUNTIME_ROWS,
            bind=PREFILL_BIND,
            grid="dim3(total_ctas)",
            opt_in_var="opt_in_folded",
            when=f"fold_rotary == {ROTARY}",
        ),
    ],
)

# The decode's refusals: the same extents and shape as the prefill's, plus
# the slot count the plan arrays are declared at and the chunk, which the
# body walks as whole pages. The decode-row bound is the split factor times
# the deepest bucket, the same product the arena sizes its tables by.
DECODE_BOUNDS = (
    f"  if (q_rows > {Q_ROWS} || pool_rows > {POOL_ROWS} || batch > {MAX_DECODE_ROWS}\n"
    f"      || padded_slots > {MAX_SLOTS} || page_size != {PAGE_SIZE}\n"
    f"      || chunk_tokens <= 0 || chunk_tokens % page_size != 0\n"
    f"      || num_qo_heads != {HEADS} || num_kv_heads != {HEADS // GROUPS}\n"
    f"      || (fold_rotary != 0 && fold_rotary != {ROTARY})) {{\n"
    f"    return static_cast<int>(cudaErrorInvalidValue);\n"
    f"  }}\n"
)
DECODE_PRELUDE = (
    f"  const int chunk_pages = chunk_tokens / page_size;\n"
    f"  if (batch == 0 || padded_slots == 0) {{\n"
    f"    // A step with no decode rows is a real state, not an error.\n"
    f"    return static_cast<int>(cudaSuccess);\n"
    f"  }}\n"
    f"  // The merge's first level is gridded over the widest request's slots;\n"
    f"  // a group past a narrower request's does nothing.\n"
    f"  const int slots_per_request = (padded_slots + batch - 1) / batch;\n"
    f"  const int merge_groups = (slots_per_request + {defs.DECODE_MERGE_SPAN - 1})"
    f" / {defs.DECODE_MERGE_SPAN};\n"
)

DECODE_PARTIAL_BIND = {
    "Q": "reinterpret_cast<const bfloat16_t*>(q)",
    "PageIndices": "page_indices",
    "PageIndptr": "page_indptr",
    "LastPageLen": "last_page_len",
    "RequestIndices": "request_indices",
    "KvTileIndices": "kv_tile_indices",
    "ValidMask": "valid_mask",
    "sm_scale": "sm_scale",
    "rows_per_page": "rows_per_page",
    "layer_row": "layer_row",
    "row_offset": "row_offset",
    "chunk_pages": "chunk_pages",
    "padded_slots": "padded_slots",
    "TmpV": "reinterpret_cast<bfloat16_t*>(tmp_v)",
    "TmpS": "tmp_s",
}
DECODE_PARTIAL_RUNTIME_ROWS = {
    "KV_desc": (POOL_ROWS, "static_cast<cuuint64_t>(pool_rows)"),
}

DECODE = Launcher(
    name=DECODE_LAUNCHER,
    params=DECODE_LAUNCHER_PARAMS,
    bounds=DECODE_BOUNDS,
    prelude=DECODE_PRELUDE,
    kernels=[
        # The query tile is filled elementwise, so only the pool's page loads
        # lower to a descriptor here. The partial pass has a body per row
        # format; both store in head order, so the merge levels are shared.
        Kernel(
            symbol=DECODE_PARTIAL_SYMBOL,
            build=build_decode_partial,
            descriptor_var={"KV_desc": "kv_desc"},
            tensor_arg={"KV": "kv"},
            runtime_rows=DECODE_PARTIAL_RUNTIME_ROWS,
            bind=DECODE_PARTIAL_BIND,
            grid="dim3(padded_slots, num_kv_heads)",
            opt_in_var="opt_in_partial",
            when="fold_rotary == 0",
        ),
        Kernel(
            symbol=f"{DECODE_PARTIAL_SYMBOL[: -len('_kernel')]}_folded_kernel",
            build=build_decode_partial_folded,
            descriptor_var={"KV_desc": "kv_desc_folded"},
            tensor_arg={"KV": "kv"},
            runtime_rows=DECODE_PARTIAL_RUNTIME_ROWS,
            bind=DECODE_PARTIAL_BIND,
            grid="dim3(padded_slots, num_kv_heads)",
            opt_in_var="opt_in_partial_folded",
            when=f"fold_rotary == {ROTARY}",
        ),
        # The merge's two levels: groups of slots folded in place, then the
        # group heads into the output.
        Kernel(
            symbol=DECODE_MERGE_GROUPS_SYMBOL,
            build=build_decode_merge_groups,
            descriptor_var={},
            tensor_arg={},
            runtime_rows={},
            bind={
                "TmpV": "reinterpret_cast<bfloat16_t*>(tmp_v)",
                "TmpS": "tmp_s",
                "OIndptr": "o_indptr",
                "batch": "batch",
                "groups": "merge_groups",
            },
            grid="dim3(batch, num_qo_heads, merge_groups)",
            opt_in_var="opt_in_merge_groups",
        ),
        Kernel(
            symbol=DECODE_MERGE_SYMBOL,
            build=build_decode_merge,
            descriptor_var={},
            tensor_arg={},
            runtime_rows={},
            bind={
                "TmpV": "reinterpret_cast<bfloat16_t*>(tmp_v)",
                "TmpS": "tmp_s",
                "OIndptr": "o_indptr",
                "row_offset": "row_offset",
                "batch": "batch",
                "Output": "reinterpret_cast<bfloat16_t*>(out)",
            },
            grid="dim3(batch, num_qo_heads)",
            opt_in_var="opt_in_merge",
        ),
    ],
)

# The sliding family's decode refuses like the global one, plus a chunk
# that is not whole tiles and a window that is not a distance.
LOCAL_DECODE_BOUNDS = (
    f"  if (q_rows > {Q_ROWS} || pool_rows > {POOL_ROWS} || batch > {MAX_DECODE_ROWS}\n"
    f"      || padded_slots > {LOCAL_MAX_SLOTS} || page_size != {LOCAL_PAGE_SIZE}\n"
    f"      || chunk_tokens <= 0 || chunk_tokens % {LOCAL_TILE_PAGES * LOCAL_PAGE_SIZE} != 0\n"
    f"      || window_left < 0\n"
    f"      || num_qo_heads != {LOCAL_HEADS} || num_kv_heads != {LOCAL_HEADS // LOCAL_GROUPS}) {{\n"
    f"    return static_cast<int>(cudaErrorInvalidValue);\n"
    f"  }}\n"
)
LOCAL_DECODE_PRELUDE = (
    f"  const int chunk_pages = chunk_tokens / page_size;\n"
    f"  if (batch == 0 || padded_slots == 0) {{\n"
    f"    return static_cast<int>(cudaSuccess);\n"
    f"  }}\n"
    f"  const int slots_per_request = (padded_slots + batch - 1) / batch;\n"
    f"  const int merge_groups = (slots_per_request + {defs.DECODE_MERGE_SPAN - 1})"
    f" / {defs.DECODE_MERGE_SPAN};\n"
)

LOCAL_DECODE = Launcher(
    name=LOCAL_DECODE_LAUNCHER,
    params=LOCAL_DECODE_LAUNCHER_PARAMS,
    bounds=LOCAL_DECODE_BOUNDS,
    prelude=LOCAL_DECODE_PRELUDE,
    kernels=[
        # The page copies land in row slices of one shared tile, which the
        # lowering carries as plain copies rather than bulk ones, so the
        # pool is a plain pointer here and its declared rows are a guard.
        Kernel(
            symbol=LOCAL_DECODE_PARTIAL_SYMBOL,
            build=build_local_decode_partial,
            descriptor_var={},
            tensor_arg={},
            runtime_rows={},
            bind={
                **DECODE_PARTIAL_BIND,
                "KV": "reinterpret_cast<const bfloat16_t*>(kv)",
                "window_left": "window_left",
            },
            grid="dim3(padded_slots, num_kv_heads)",
            opt_in_var="opt_in_partial",
        ),
        Kernel(
            symbol=LOCAL_DECODE_MERGE_GROUPS_SYMBOL,
            build=build_local_decode_merge_groups,
            descriptor_var={},
            tensor_arg={},
            runtime_rows={},
            bind={
                "TmpV": "reinterpret_cast<bfloat16_t*>(tmp_v)",
                "TmpS": "tmp_s",
                "OIndptr": "o_indptr",
                "batch": "batch",
                "groups": "merge_groups",
            },
            grid="dim3(batch, num_qo_heads, merge_groups)",
            opt_in_var="opt_in_merge_groups",
        ),
        Kernel(
            symbol=LOCAL_DECODE_MERGE_SYMBOL,
            build=build_local_decode_merge,
            descriptor_var={},
            tensor_arg={},
            runtime_rows={},
            bind={
                "TmpV": "reinterpret_cast<bfloat16_t*>(tmp_v)",
                "TmpS": "tmp_s",
                "OIndptr": "o_indptr",
                "row_offset": "row_offset",
                "batch": "batch",
                "Output": "reinterpret_cast<bfloat16_t*>(out)",
            },
            grid="dim3(batch, num_qo_heads)",
            opt_in_var="opt_in_merge",
        ),
    ],
)

# The sliding family's prefill refuses like the global one, with the window
# as a distance in place of the row format; its grid is one CTA per (query
# tile, head) of every request, the sum the body re-walks.
LOCAL_PREFILL_BOUNDS = (
    f"  if (q_rows > {Q_ROWS} || pool_rows > {POOL_ROWS} || batch < 0 || batch > {MAX_BATCH}\n"
    f"      || page_size != {LOCAL_PAGE_SIZE} || window_left < 0\n"
    f"      || num_qo_heads != {LOCAL_HEADS} || num_kv_heads != {LOCAL_HEADS // LOCAL_GROUPS}) {{\n"
    f"    return static_cast<int>(cudaErrorInvalidValue);\n"
    f"  }}\n"
)
LOCAL_PREFILL_GRID = (
    f"  int total_ctas = 0;\n"
    f"  for (int request = 0; request < batch; ++request) {{\n"
    f"    const int rows = host_q_indptr[request + 1] - host_q_indptr[request];\n"
    f"    total_ctas += ((rows + {defs.LOCAL_BLOCK_M - 1}) / {defs.LOCAL_BLOCK_M}) * {LOCAL_HEADS};\n"
    f"  }}\n"
    f"  if (total_ctas == 0) {{\n"
    f"    return static_cast<int>(cudaSuccess);\n"
    f"  }}\n"
)
LOCAL_PREFILL = Launcher(
    name=LOCAL_PREFILL_LAUNCHER,
    params=LOCAL_PREFILL_LAUNCHER_PARAMS,
    bounds=LOCAL_PREFILL_BOUNDS,
    prelude=LOCAL_PREFILL_GRID,
    kernels=[
        Kernel(
            symbol=LOCAL_PREFILL_SYMBOL,
            build=build_local_prefill,
            descriptor_var=DESCRIPTOR_VAR,
            tensor_arg=TENSOR_ARG,
            runtime_rows=PREFILL_RUNTIME_ROWS,
            bind={**PREFILL_BIND, "batch": "batch", "window_left": "window_left"},
            grid="dim3(total_ctas)",
            opt_in_var="opt_in_local_prefill",
        ),
    ],
)

LAUNCHERS = [PREFILL, DECODE, LOCAL_DECODE, LOCAL_PREFILL]


def check_strides(maps: list[TensorMap]) -> None:
    """The builder's innermost stride is the element size; hold it to that."""
    for tmap in maps:
        dtype = enum_of(TENSORMAP_DTYPE, tmap.dtype, "dtype")
        if dtype not in ELEMENT_BYTES:
            raise RuntimeError(f"{dtype} has no element size to check against")
        expected = ELEMENT_BYTES[dtype]
        if tmap.strides[0] != expected:
            raise RuntimeError(
                f"{tmap.name}: innermost stride {tmap.strides[0]} is not the "
                f"{expected} B element of {dtype}"
            )


def check_in_place_merge_barrier(symbol: str, body: str) -> None:
    """A merge that folds a group into its own first slot has to read the
    whole group before any thread overwrites it.

    Every thread walks the group's `TmpS` and the store lands on the first
    slot, one of the rows just read, so without a block barrier between them
    a thread that finished the walk overwrites state another is still
    reading. That is a schedule away from wrong rather than wrong on any one
    run, which is why it is held on the emitted source: a comparison would
    need the losing schedule to show it.
    """
    store = re.search(r"TmpS\[[^;]*?\]\s*=", body)
    if store is None:
        raise RuntimeError(f"{symbol}: no store to TmpS to hold a barrier against")
    reads = [m.start() for m in re.finditer(r"TmpS\[", body) if m.start() < store.start()]
    if not reads:
        raise RuntimeError(f"{symbol}: no read of TmpS before the store")
    barrier = body.find("__syncthreads()", reads[-1], store.start())
    if barrier < 0:
        raise RuntimeError(
            f"{symbol}: the store to TmpS is not separated from the group's "
            "reads by __syncthreads(); a thread that finished the walk would "
            "overwrite what another is still reading"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--arch",
        required=True,
        help="arch to lower and assemble for, e.g. sm_90a. Required: without "
        "it TileLang lowers for whatever device it can see, or for its own "
        "default on a host with none.",
    )
    parser.add_argument(
        "--vendor-includes",
        action="store_true",
        help="copy the header roots into the output and point the manifest at "
        "the copies (self-contained pre-generated dir)",
    )
    args = parser.parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lowered = {
        kernel.symbol: lower(kernel, args.arch)
        for launcher in LAUNCHERS
        for kernel in launcher.kernels
    }
    # One preamble for the unit. The kernels share TileLang's headers; what
    # one may add over another is an include, appended after the first
    # preamble so its own conditional blocks stay balanced. Deduplicating
    # by line does not work: a repeated `#endif` is not a duplicate.
    first, *rest = lowered.values()
    base_lines = set(first.preamble.splitlines())
    extra: list[str] = []
    for low in rest:
        for line in low.preamble.splitlines():
            if line in base_lines or line in extra:
                continue
            if not line.startswith("#include"):
                raise RuntimeError(
                    f"{low.kernel.symbol}: its preamble differs from "
                    f"{first.kernel.symbol}'s beyond an include: {line!r}"
                )
            extra.append(line)
    preamble = first.preamble + "".join(f"{line}\n" for line in extra)

    cu_path = out_dir / f"{CU_STEM}.cu"
    cu_path.write_text(
        "// Generated by pegainfer-gemma4/kernels/generate.py. Do not edit.\n"
        + isolate_debug_helpers(preamble)
        + "".join(low.body for low in lowered.values())
        + LAUNCHER_HEAD
        + "".join(
            render_launcher(launcher, [lowered[k.symbol] for k in launcher.kernels])
            for launcher in LAUNCHERS
        )
    )

    if args.vendor_includes:
        template_include, cutlass_include = vendor_includes(out_dir)
    else:
        template_include = Path(TILELANG_TEMPLATE_PATH)
        cutlass_include = Path(CUTLASS_INCLUDE_DIR)

    # Relative where it can be, so a vendored directory survives being copied.
    def named(path: Path) -> str:
        try:
            return str(Path(path).relative_to(out_dir))
        except ValueError:
            return str(path)

    lines = [f"CU_PATH={named(cu_path)}"]
    lines.append(f"TILELANG_TEMPLATE_PATH={named(template_include)}")
    lines.append(f"CUTLASS_INCLUDE_DIR={named(cutlass_include)}")
    # Shapes are compile dimensions, so the consumer can refuse another.
    lines.append(f"GEOMETRY={HEADS},{HEADS // GROUPS},{HEAD_DIM},{PAGE_SIZE}")
    # The largest opt-in any block here makes: SM90 grants 227 KiB, SM120 only 99.
    lines.append(f"SMEM={max(low.smem for low in lowered.values())}")
    lines.extend(f"NVCC_FLAG={flag}" for flag in required_nvcc_flags())
    # One line per entry point, in the stub tier's order: build.rs checks
    # both the names and the parameter lists against its own table.
    lines.extend(
        f"LAUNCHER={launcher.name}|{', '.join(kind for kind, _ in launcher.params)}"
        for launcher in LAUNCHERS
    )
    # The body is lowered for exactly this arch and uses arch-conditional
    # instructions, so the consumer assembles it for that and not for the
    # generic SM list.
    lines.append(f"ARCH={args.arch}")
    # The manifest lets a build host consume a pre-generated directory without
    # re-running (or even having) TileLang; build.rs parses the same key=value
    # lines from either stdout or this file.
    (out_dir / "manifest.txt").write_text("\n".join(lines) + "\n")
    for line in lines:
        print(line)
    for symbol, low in lowered.items():
        print(
            f"# {symbol}: block {low.block} threads, {low.smem} B dynamic shared, "
            f"{len(low.maps)} descriptors"
        )


if __name__ == "__main__":
    main()
