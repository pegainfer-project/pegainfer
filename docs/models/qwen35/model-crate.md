# Qwen3.5 Model Crate

> **TL;DR:** `pegainfer-qwen35` owns the hybrid model runtime and scheduler; the server launches it through the shared step-contract `Engine`. Qwen3.8 uses the same model line.
>
> **Last touched:** 2026-10

## Ownership and entry points

The crate owns config and weight loading, recurrent/conv state, paged KV,
prefill/decode/unified execution, scheduler policy, TP workers and model-local
tests. Shared CUDA wrappers and kernel builds remain in `pegainfer-core` and
`pegainfer-kernels`; HTTP and the engine contract belong to
`pegainfer-frontend`.

`start_engine*` and `launch_with_options_policy_and_overlap` return
`Result<Engine>`. `Qwen35Line::launch` returns `LaunchedEngine::Stepped`,
exposing one logical scheduler for either a single GPU or a TP replica.
`EngineInfo` carries the servable context limit and logical KV capacity.

The low-level `runtime` module exposes executors and model state for
model-local tests, debugging and benchmarks. `runtime_ops` exposes the
operator wrappers needed by the crate's benches. The server uses the model
line's launch boundary rather than these execution internals.

Qwen3.8 shares the config identity and text geometry used by this line. Its
weight-loading, checkpoint-fixture and serving-template differences are
recorded in [Qwen3.8 support](support-qwen38.md).

## Scheduler boundary

The shared driver drains submissions, calls `Qwen35Scheduler::step`, publishes
metrics and commits ledger updates. `RequestLedger` owns request identity,
admission/terminal updates and emitted-token counts. The Qwen3.5 scheduler
retains admission, planning and backend-state ownership; its existing
backends retain chunked prefill, prefix caching, TP, CUDA Graph and single-GPU
overlap execution.

Single-GPU launch waits for CUDA context binding and thread-local cuBLAS
initialization on the scheduler thread before returning an engine. TP workers
initialize their own CUDA/cuBLAS/NCCL resources before the controller starts
its scheduler, so that scheduler needs no second startup handshake. See
[TP implementation](tp-implementation.md) for the worker protocol and
[scheduler metrics](load-snapshot.md) for publication, cancellation and idle
polling.

Qwen3.5 suppresses model EOS unless `ignore_eos` is set, and a stop finishes
without a typed cause. The bridge supplies its legacy sentinel so HTTP usage
still counts suppressed EOS; ledger counts cover only emitted tokens.
Request-scoped stop IDs and retained trigger/logprob metadata remain
follow-up work. Qwen3.5 rejects non-null `prompt_logprobs` before backend
allocation and does not provide prompt echo.

## Build and validation

The `qwen35` feature enables the crate and its Triton AOT GDR prefill kernels.
`PEGAINFER_TRITON_PYTHON` selects the build-time Python/Triton interpreter.
The default Qwen3 build does not enable this kernel path; the Qwen3.5 tests
and benches declare their required feature explicitly.

Use the model crate's scheduler, chunked-prefill, sampling, prefix-cache and
TP2 serving integration tests for request-flow changes. The HF logits gate
in [accuracy](accuracy.md) is the numerical oracle; an E2E test completing or
producing plausible text is not an accuracy measurement. The shared model
fixture uses `PEGAINFER_TEST_MODEL_PATH`; use an absolute path to avoid
package working-directory ambiguity. TP2 gates require two devices and must
run serially.
