# Runtime

> **TL;DR:** Runtime complexity grows fast as new model families come in. We control it by keeping a shared core (`pegainfer-core`) that owns the generation contract and orchestration, and pushing model-specific execution into per-model crates behind a single trait. The trait deliberately hides prefill vs decode and homogeneous vs hybrid attention from the caller.
>
> **Last touched:** 2026-05.

## Why this exists

Each new model family tempts you to grow a separate runtime — different prefill, different decode, different state. Doing that once per model means the serving path drifts and integration cost compounds.

The goal here is not abstraction for its own sake. It's that **new model-family work should start from a shared runtime foundation, not a fresh execution stack**. Framework debt should not grow faster than model coverage.

## The contract

```rust
trait GenerationState {
    fn logits(&self) -> &DeviceVec;
    fn reset(&mut self) -> Result<()>;
}

trait ModelForward {
    type State: GenerationState;

    fn create_state(&self) -> Result<Self::State>;
    fn forward(&self, tokens: &[u32], state: &mut Self::State) -> Result<()>;
    fn is_stop_token(&self, token_id: u32) -> bool;
}
```

Design points worth keeping in mind:

- **Single `forward` method.** `tokens.len() > 1` → batch prefill internally; `tokens.len() == 1` → decode. The caller (generation loop today, scheduler tomorrow) never sees the distinction.
- **`&self` is weights, `State` is per-request mutable.** Designed for batch > 1: N states share one set of weights.
- **Logits live on `State`.** Decode-path logits via `DecodeBuffers`, prefill-path logits via a separate field. The caller reads through `state.logits()` and doesn't care which path produced them.
- **Hybrid attention is a model-internal concern.** Qwen3.5's 24 linear + 8 full attention pattern, per-layer dispatch, recurrent state, and chunkwise prefill scratch all live behind the trait — the runtime sees a uniform `forward()`.

## What's been done

- Shared runtime/API entry extracted into `pegainfer-core`: sampler, page/KV pools, weight loading, CUDA Graph state, shared op adapters, `ModelForward` / `GenerationState`.
- Per-model crates (`pegainfer-qwen3`, `pegainfer-qwen35`) own their config, weights, prefill/decode execution, scheduler, and tests.
- Generation loop unified into `pegainfer-core` against the trait — replaces ~120 lines of duplicated orchestration per model.
- Internal modules (decode buffers, KV cache, recurrent state, FFI bindings, tokenizer streaming, weight-loader helpers) pulled back behind crate-local visibility. `unreachable_pub` is meaningful again.
- Trace machinery removed from the active public surface.

## Known tensions

- **Prefill/decode strategy diverges per model.** Qwen3 uses batched GEMM + `PrefillBuffers`; Qwen3.5 uses per-layer + `GdrChunkwiseScratch`. The trait does not constrain this, but it means model crates carry meaningful execution complexity.
- **State shape diverges per model.** Qwen3 = `DecodeBuffers + KVCache + CudaGraphState`. Qwen3.5 adds `RecurrentState`. Adding bs > 1 means each State must scale across requests, not just across layers.
- **CUDA Graph capture lives inside model executors.** That's intentional (pre-allocated buffers preserve pointer stability), but it couples graph state into the per-model crate and the contract has to stay friendly to that.

## Next

Two open directions, both downstream of the trait:

1. **bs > 1 in serving.** The `&self`/State split is the prerequisite, but the scheduler still feeds requests one at a time through the trait. Lifting the scheduler to multi-state batches is the next structural step.
2. **Cross-model orchestration cleanup.** When the third model family lands, anything still duplicated between Qwen3 and Qwen3.5 should be pulled up — not preemptively, but at integration time so the cost is visible.
