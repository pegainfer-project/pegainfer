# Gemma 4 TileLang kernels

**TL;DR**: two TileLang families that `pegainfer-kernels/build.rs` generates
under the `gemma4` feature and hands to nvcc, each through three tiers
(generate, pre-generated, stub); the generated CUDA is a Cargo `OUT_DIR`
artifact that is never checked in. `generate.py` emits the attention family:
the global prefill, the global split-KV decode, and the sliding window's
prefill and decode. Unlike the K3 families it lowers to TMA, so the launchers
build the descriptors themselves from parameters recovered out of the lowered
host stub. `w4a16_generate.py` emits the W4A16 decode GEMMs, compiled for one
SM count.

## What lives here

| File | Role |
| --- | --- |
| `tilelang_defs.py` | The attention kernels, authored here. Upstream has nothing for these head dims on SM90. |
| `generate.py` | Lowers them, recovers the launch geometry and the TMA descriptor parameters, emits the `.cu` with its hand-written launchers. |
| `w4a16_defs.py` | The W4A16 decode GEMMs: stream-K over persistent CTAs, one kernel per linear shape and row bucket. |
| `w4a16_generate.py` | Lowers them through `generate.py`'s rendering behind one dispatching launcher, and states the CTA count the crate checks the device against. |

## Why one instantiation is enough

Every shape a TileLang kernel declares is a compile dimension, and a serving
step's packed query rows, page-table length and pool size are all run-time
quantities. They are nevertheless a single instantiation, because the declared
extents reach the generated code only as bounds guards — two lowerings that
differ solely in them differ in nothing else — so declaring the serving
arena's maxima lets every smaller step through. The real bounds are the walk's
own trip count and the predicated store.

The two tensors the lowering reads through TMA are separate: their extents
live in the descriptors, which the launcher builds per call from the arguments
it is given, so those are always the step's own.

## The descriptor parameters are recovered, not written down

TileLang exposes no accessor for the launch it baked into the host stub, so
`generate.py` parses the packed-call argument stack — the same technique the
K3 generator uses for its launch geometry, extended to the descriptor builds.
Each parameter is then mapped to its driver enum through a table with no
default, and the tensor and descriptor names are bound through tables with no
default either, so a codegen change fails generation instead of silently
encoding a stale descriptor.

The K3 generator refuses a TMA-lowered body outright, for exactly the reason
this file exists: its launchers bind plain pointers and the requested thread
count, and a warp-specialized kernel accepts neither.

## Gates

The attention kernels' numerics are gated against the serving path's own
reference, not against random tensors: paged output is bit-identical to the
contiguous form over scattered pages and partial final pages, and a ragged
batch matches an fp32 reference per request while leaving every row past the
batch untouched — those rows are the decode rows sharing a mixed step's output
buffer. The W4A16 GEMMs are gated against their definition on every 31B shape
(`pegainfer-kernels/tests/gemma4_w4a16_gemm.rs`): a row's bits are the same in
every bucket.
