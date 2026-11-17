#!/usr/bin/env python3
"""Dump vLLM's captured CUDA graphs to Graphviz .dot files.

vLLM never exposes its captured `cudaGraph_t`, but the full decode-step
kernel sequence (names, launch dims, dependency DAG) is the ground truth for
comparing engines kernel-by-kernel. This tool monkeypatches torch's graph
capture so every graph vLLM captures is retained and printed via
`cudaGraphDebugDotPrint` — no vLLM or torch source changes.

How it works (each step is load-bearing):
  - vLLM runs EngineCore and TP/DP workers in child processes, spawned (not
    forked) once CUDA is initialized in the parent — and a spawned
    interpreter has no monkeypatches. The CLI therefore writes a
    `sitecustomize.py` and prepends it to PYTHONPATH, so every child python
    process installs the hooks at interpreter startup.
  - torch >= 2.10 frees the underlying `cudaGraph_t` at capture_end unless
    `CUDAGraph(keep_graph=True)`. The pybind C++ object is constructed in
    `__init__`, so a subclass must override BOTH `__new__` and `__init__`;
    overriding `__new__` alone is silently ignored.
  - `CUDAGraph.capture_end` is overridden to dot-print the raw graph right
    after capture; hooking the `torch.cuda.graph` context manager instead
    misses captures that drive capture_begin/capture_end directly (vLLM's
    breakable piecewise wrapper). torch's own `debug_dump()` is a silent
    no-op here.

By default vLLM captures FULL_AND_PIECEWISE: one big graph per decode batch
size plus one small graph per compiled fragment. The CLI pins cudagraph_mode
to FULL_DECODE_ONLY so only the whole-step decode graphs are captured; pass
--cudagraph-mode FULL_AND_PIECEWISE to also dump the per-fragment graphs.
Render the output with scripts/profiling/cuda_graph_png.py.

Run inside a Python environment that has vLLM:

    uv run python scripts/profiling/vllm_graph_dump.py MODEL_PATH -o dots/

or import `install_graph_dump_hooks()` before any vLLM import in your own
script to attach the dumper to an arbitrary vLLM workload.
"""

import argparse
import ctypes
import os
from pathlib import Path

CUDA_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1 << 0


def _load_cudart():
    for lib in ("libcudart.so.13", "libcudart.so.12"):
        try:
            return ctypes.CDLL(lib)
        except OSError:
            continue
    raise RuntimeError("libcudart.so.12/.13 not found; is CUDA installed?")


def install_graph_dump_hooks(out_dir):
    """Patch torch so every CUDA graph captured after this call is dumped to
    `out_dir/gNNN_<nodes>n.dot`. Must run in the process that captures, before
    the capturing code executes; vLLM captures in spawned children, so call
    `prepare_child_injection` as well (the CLI does both)."""
    import torch
    import torch.cuda.graphs as tg

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cudart = _load_cudart()
    state = {"count": 0}

    def dump(graph):
        raw = ctypes.c_void_p(graph.raw_cuda_graph())
        num_nodes = ctypes.c_size_t()
        cudart.cudaGraphGetNodes(raw, None, ctypes.byref(num_nodes))
        idx = state["count"]
        state["count"] += 1
        # TP/DP workers are separate processes sharing out_dir; the pid keeps
        # per-rank dumps from overwriting each other
        path = out / f"g{idx:03d}_p{os.getpid()}_{num_nodes.value}n.dot"
        rc = cudart.cudaGraphDebugDotPrint(
            raw, str(path).encode(), ctypes.c_uint(CUDA_GRAPH_DEBUG_DOT_FLAGS_VERBOSE)
        )
        if rc != 0:
            raise RuntimeError(f"cudaGraphDebugDotPrint failed with cudaError {rc} for {path}")
        print(f"[graph-dump] #{idx}: {num_nodes.value} nodes -> {path}")

    # Dumping from capture_end covers every capture style: `with
    # torch.cuda.graph(...)` calls it from __exit__, and code that drives
    # capture_begin/capture_end by hand (e.g. vLLM's breakable piecewise
    # wrapper) never enters the context manager at all.
    class KeepGraph(tg.CUDAGraph):
        def __new__(cls, *args, **kwargs):
            return super().__new__(cls, keep_graph=True)

        def __init__(self, *args, **kwargs):
            super().__init__(keep_graph=True)

        def capture_end(self, *args, **kwargs):
            result = super().capture_end(*args, **kwargs)
            dump(self)
            return result

    torch.cuda.CUDAGraph = KeepGraph
    tg.CUDAGraph = KeepGraph


def prepare_child_injection(out_dir):
    """Spawned worker processes start from a fresh interpreter, so the hooks
    must be installed by sitecustomize at startup rather than inherited."""
    inject_dir = Path(out_dir).resolve() / ".inject"
    inject_dir.mkdir(parents=True, exist_ok=True)
    tool_dir = Path(__file__).resolve().parent
    (inject_dir / "sitecustomize.py").write_text(
        "import sys, traceback\n"
        f"sys.path.insert(0, {str(tool_dir)!r})\n"
        "try:\n"
        "    import vllm_graph_dump\n"
        f"    vllm_graph_dump.install_graph_dump_hooks({str(Path(out_dir).resolve())!r})\n"
        "except Exception:\n"
        "    traceback.print_exc()\n"
    )
    existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{inject_dir}:{existing}" if existing else str(inject_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", help="model path or HF id for vllm.LLM")
    ap.add_argument("-o", "--out-dir", default="vllm_graph_dots", help="output dir for .dot files")
    ap.add_argument("--capture-sizes", default="1", help="comma-separated cudagraph capture sizes")
    ap.add_argument("--cudagraph-mode", default="FULL_DECODE_ONLY",
                    help="vLLM cudagraph_mode; the default skips piecewise fragments entirely, "
                         "use FULL_AND_PIECEWISE to also dump per-fragment graphs")
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("-tp", "--tensor-parallel-size", type=int, default=1)
    ap.add_argument("-dp", "--data-parallel-size", type=int, default=1)
    ap.add_argument("--enable-expert-parallel", action="store_true")
    ap.add_argument("--load-format", default="auto",
                    help="use 'dummy' to skip weight loading; graph topology does not "
                         "depend on weight values")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-tokens", type=int, default=4)
    args = ap.parse_args()

    prepare_child_injection(args.out_dir)
    install_graph_dump_hooks(args.out_dir)

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        enable_expert_parallel=args.enable_expert_parallel,
        load_format=args.load_format,
        compilation_config={
            "cudagraph_mode": args.cudagraph_mode,
            "cudagraph_capture_sizes": [int(s) for s in args.capture_sizes.split(",")],
        },
    )
    out = llm.generate([args.prompt], SamplingParams(max_tokens=args.max_tokens))
    print("generated:", out[0].outputs[0].text)
    print(f"done; render with: uv run python scripts/profiling/cuda_graph_png.py {args.out_dir}/<file>.dot")


if __name__ == "__main__":
    main()
