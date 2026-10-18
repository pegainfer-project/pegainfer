# DeepSeek-V2-Lite EP2 Decode Attribution Gate

> **TL;DR:** This is a direct diagnostic benchmark for a narrow DeepSeek-V2-Lite EP2 shape: `prompt="Hello"`, `output_len=16`, batch `1/4/8`, host-staged and NCCL. Its timing and graph-readiness fields do not carry an HTTP serving SLO or production-readiness claim.

Last touched: 2026-07

## Scope

Covered:

- Model: DeepSeek-V2-Lite.
- Shape: `prompt="Hello"`, prompt token ids `[17464]`, `output_len=16`.
- Attribution batches: `1`, `4`, and `8`.
- Full decode graph probe: NCCL backend, batch `1` only.
- Accuracy oracle: same-host HF / host-staged / NCCL token and text exactness.

Out of scope:

- production continuous batching;
- mixed-request serving;
- sparse dispatch;
- pegainfer-comm or multi-node EP;
- vLLM parity;
- performance improvement from CUDA Graph evidence alone.

## Commands

Run the HF / host-staged / NCCL comparison before trusting attribution:

```bash
mkdir -p target/accuracy/dsv2-lite-ep2

python tools/accuracy/hf_dump_dsv2_lite_ep2_greedy.py \
  --model-path models/DeepSeek-V2-Lite \
  --case-set-json test_data/deepseek-v2-lite-ep2-cases.json \
  --out target/accuracy/dsv2-lite-ep2/hf.json

PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite \
PEGAINFER_DSV2_LITE_E2E_CASE_SET=test_data/deepseek-v2-lite-ep2-cases.json \
PEGAINFER_DSV2_LITE_E2E_JSON_OUT=target/accuracy/dsv2-lite-ep2/host-staged.json \
  cargo test --release -p pegainfer-deepseek-v2-lite \
  --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite \
PEGAINFER_DSV2_LITE_E2E_CASE_SET=test_data/deepseek-v2-lite-ep2-cases.json \
PEGAINFER_DSV2_LITE_EP_BACKEND=nccl \
PEGAINFER_DSV2_LITE_E2E_JSON_OUT=target/accuracy/dsv2-lite-ep2/nccl.json \
  cargo test --release -p pegainfer-deepseek-v2-lite \
  --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

python tools/accuracy/compare_dsv2_lite_ep2_outputs.py \
  --hf target/accuracy/dsv2-lite-ep2/hf.json \
  --host-staged target/accuracy/dsv2-lite-ep2/host-staged.json \
  --nccl target/accuracy/dsv2-lite-ep2/nccl.json \
  --out target/accuracy/dsv2-lite-ep2/comparison.json \
  --require-all-exact
```

Collect batch-1 attribution:

```bash
cargo run --release -p pegainfer-deepseek-v2-lite \
  --features deepseek-v2-lite \
  --bin dsv2_lite_ep2_decode_attribution \
  -- --model-path models/DeepSeek-V2-Lite \
  --batch-size 1 \
  --out target/accuracy/dsv2-lite-ep2/host-staged-attribution.json

PEGAINFER_DSV2_LITE_EP_BACKEND=nccl \
  cargo run --release -p pegainfer-deepseek-v2-lite \
  --features deepseek-v2-lite \
  --bin dsv2_lite_ep2_decode_attribution \
  -- --model-path models/DeepSeek-V2-Lite \
  --batch-size 1 \
  --out target/accuracy/dsv2-lite-ep2/nccl-attribution.json
```

Run the full decode graph probe:

```bash
PEGAINFER_DSV2_LITE_EP_BACKEND=nccl \
  cargo run --release -p pegainfer-deepseek-v2-lite \
  --features deepseek-v2-lite \
  --bin dsv2_lite_ep2_decode_attribution \
  -- --model-path models/DeepSeek-V2-Lite \
  --batch-size 1 \
  --full-decode-graph-probe \
  --out target/accuracy/dsv2-lite-ep2/nccl-full-decode-graph-probe.json
```

Use `--batch-size 4` or `--batch-size 8` for attribution regression only. Those rows do not widen the graph probe claim.

Optional diagnostics:

- `--nccl-graph-smoke` runs a preallocated f32 NCCL all-reduce CUDA Graph smoke. It is collective-only evidence.
- `PEGAINFER_DSV2_LITE_NVTX=1` emits NVTX ranges for profiler correlation. JSON CUDA event rows remain the timing source.
- `PEGAINFER_NCCL_PYTHON` can point at a Python environment whose NCCL wheel is newer than the system `libnccl`.

## JSON Checks

The top-level report includes:

- `schema_version`, `kind`, model, backend, commit, and hardware/toolchain metadata;
- `workload` and `metrics`: stable direct-diagnostic contract fields;
- `accuracy`: generated tokens/text and exact hash status;
- `timing`: CPU-side section timing for the fixed attribution path;
- `gpu_timing` and `by_gpu_*`: selected CUDA event sections only;
- `ep`: host-staged or NCCL route/collective counters;
- `coverage`: claim status rows;
- `cuda_graph_readiness`: schema-2 graph readiness diagnostics.

For issue #278, the important readiness checks are:

```text
cuda_graph_readiness.schema == 2
cuda_graph_readiness.backend == "nccl"
cuda_graph_readiness.batch_size == 1
cuda_graph_readiness.full_decode_capture_ready == true
cuda_graph_readiness.full_decode_graph_probe.captured == true
cuda_graph_readiness.full_decode_graph_probe.instantiated == true
cuda_graph_readiness.full_decode_graph_probe.replayed == true
cuda_graph_readiness.full_decode_graph_probe.verified == true
cuda_graph_readiness.full_decode_graph_probe.replay_count == 8
cuda_graph_readiness.full_decode_graph_probe.verified_replay_count == 8
cuda_graph_readiness.full_decode_graph_probe.failure_stage == "none"
cuda_graph_readiness.full_decode_graph_probe.blockers == []
```

If any capture stage fails, the report must keep `full_decode_capture_ready=false` and record the exact `failure_stage` and blocker list.

## Latest Validation

Retained 2026-06-20 validation:

- Model snapshot: `604d5664dddd88a0433dbae533b7fe9472482de0`.
- Shape: `prompt="Hello"`, `output_len=16`.
- Runtime: 2 local GPUs, host-staged and NCCL EP2.
- HF / host-staged / NCCL comparison: `all_token_text_exact`.
- Token SHA256: `4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225`.
- Text SHA256: `0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347`.
- Generated text: `, I am a 19 year old girl from the UK. I am`.
- Full probe: `captured=true`, `instantiated=true`, `replayed=true`, `verified=true`, `8/8` replays verified, no blockers.

Current NCCL attribution snapshot:

| Batch | Token/text exact | GPU event samples | GPU failures | NCCL exchange/combine calls | Route counters | Graph probe |
| ---: | --- | ---: | ---: | --- | --- | --- |
| 1 | yes | 8384 | 0 | `416 / 416` | `local=1284`, `remote=1212`, `combine=2496` | `captured_replayed_verified`, `8/8` |
| 4 | yes | 23996 | 0 | `494 / 494` | `local=5136`, `remote=4848`, `combine=9984` | not requested |
| 8 | yes | 44812 | 0 | `598 / 598` | `local=10272`, `remote=9696`, `combine=19968` | not requested |

## Claim Boundary

This gate proves exact output for the retained HF / host-staged / NCCL comparison and graph capture/replay/verify for one NCCL decode step. It does not prove default serving graph coverage, multi-step graph replay, batch `4/8` graph coverage, HTTP batching, sparse dispatch, or a performance win.
