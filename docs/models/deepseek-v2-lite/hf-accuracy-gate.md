# DeepSeek-V2-Lite EP2 HF Accuracy Gate

> **TL;DR:** HF comparison gate for DeepSeek-V2-Lite EP2. The original `Hello` / 16-token shape remains covered, and issue #274 widens the same HF / host-staged / NCCL oracle to a small committed case set with multiple prompts plus diagnostic same-prompt batch sizes `4` and `8`.
>
> **Status:** Passing evidence must come from a same-host comparison against `test_data/deepseek-v2-lite-ep2-cases.json`. The Rust E2E may emit PegaInfer case outputs, but the HF JSON remains the accuracy oracle.

## Scope

In scope:

- HF truth: `AutoTokenizer` and `AutoModelForCausalLM` with `trust_remote_code=True`, `torch_dtype=torch.bfloat16`, `model.eval()`, and `torch.no_grad()`.
- Generation shapes: the committed cases in `test_data/deepseek-v2-lite-ep2-cases.json`: `Hello` at batch `1/4/8`, plus two additional batch-1 prompts, all with `output_len=16` and greedy argmax.
- Pegainfer paths: default host-staged EP2 backend and explicit `PEGAINFER_DSV2_LITE_EP_BACKEND=nccl`.
- Result comparison: per-case and per-row generated token ids, generated text, token sha256, text sha256, and first different generated-token index.

The committed case set uses one object per case:

- `id`: stable comparison key.
- `prompt`: prompt text passed to HF and PegaInfer.
- `output_len`: requested greedy output length.
- `batch_size`: PegaInfer same-prompt row count for the diagnostic batch cases.
- `ignore_eos`: when `true`, HF sets `eos_token_id=None` and PegaInfer ignores EOS so fixed-length batch rows can be compared. Batch cases must use `ignore_eos=true`; batch-1 cases may stop on EOS.

The current Rust same-prompt batch helper is capped at batch size `8`, so the committed diagnostic batch cases stop at `4` and `8`.

Out of scope:

- Performance claims.
- Sparse dispatch or production EP backend work.
- Generic EP topology, multi-node support, serving batch, or mixed-request continuous batching.
- Any NCCL runtime-path change when host-staged and NCCL still match each other.

## Issue #135 Coverage Map

| Issue / maintainer requirement | Covered by | Evidence |
| --- | --- | --- |
| DeepSeek-V2-Lite config loads via its own crate, sharing no config assumptions with other DeepSeek lines. | PR #149 | Dedicated `pegainfer-deepseek-v2-lite` config/weight/model crate. |
| Single-node `ep_size=2` validates rank, expert ownership, and local expert count. | PR #149 | EP layout is fixed to rank 0 experts `0..31` and rank 1 experts `32..63`, with load-time validation. |
| Each rank only loads its owned 32 routed experts. | PR #149 | Driver rank loads rank 0 experts; expert rank loads only rank 1 routed experts. |
| Unsupported backend/topology reports explicit errors. | PR #149 / #150 | Unsupported device count, duplicate devices, cuda_graph, and backend names fail closed. |
| Minimal dispatch/combine path exists for the first correctness gate. | PR #149 | Host-staged dispatch/combine path remains the default baseline. |
| Maintainer-requested naive NCCL backend exists before pegainfer-comm/NVLink work. | PR #150 | `PEGAINFER_DSV2_LITE_EP_BACKEND=nccl` path passes the same EP2 greedy E2E as host-staged. |
| HF ground-truth accuracy comparison exists. | This gate | HF `generate(use_cache=true)` greedy, host-staged EP2, and NCCL EP2 are token/text exact for the covered case set. |

Together with PR #149 and PR #150, this gate covers issue #135's correctness-first acceptance surface for the narrow EP=2 milestone. Follow-up work should be tracked separately for sparse/GPU dispatch, pegainfer-comm/NVLink integration, performance evidence, long context, and broader prompts/batches.

## Commands

Run all three outputs from the same model snapshot:

```bash
mkdir -p target/accuracy/dsv2-lite-ep2

python tools/accuracy/hf_dump_dsv2_lite_ep2_greedy.py \
  --model-path models/DeepSeek-V2-Lite \
  --case-set-json test_data/deepseek-v2-lite-ep2-cases.json \
  --out target/accuracy/dsv2-lite-ep2/hf.json

PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite \
PEGAINFER_DSV2_LITE_E2E_CASE_SET=test_data/deepseek-v2-lite-ep2-cases.json \
PEGAINFER_DSV2_LITE_E2E_JSON_OUT=target/accuracy/dsv2-lite-ep2/host-staged.json \
  cargo test --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite \
PEGAINFER_DSV2_LITE_E2E_CASE_SET=test_data/deepseek-v2-lite-ep2-cases.json \
PEGAINFER_DSV2_LITE_EP_BACKEND=nccl \
PEGAINFER_DSV2_LITE_E2E_JSON_OUT=target/accuracy/dsv2-lite-ep2/nccl.json \
  cargo test --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

python tools/accuracy/compare_dsv2_lite_ep2_outputs.py \
  --hf target/accuracy/dsv2-lite-ep2/hf.json \
  --host-staged target/accuracy/dsv2-lite-ep2/host-staged.json \
  --nccl target/accuracy/dsv2-lite-ep2/nccl.json \
  --out target/accuracy/dsv2-lite-ep2/comparison.json \
  --require-all-exact
```

Omit `--require-all-exact` only when intentionally collecting mismatch diagnostics.

On Blackwell-class GPUs, make sure the selected NCCL runtime supports the device. Older NCCL runtimes may fail communicator initialization before the model-level comparison runs.

## Interpretation

- `all_token_text_exact`: HF, host-staged, and NCCL agree on generated token ids and generated text.
- `pegainfer_baseline_accuracy_gap`: host-staged and NCCL match each other, but both differ from HF. Treat this as an PegaInfer baseline accuracy problem before touching NCCL transport.
- `nccl_transport_regression`: host-staged and NCCL differ. Debug the NCCL path before drawing any HF parity conclusion.

For batch cases, the HF output is the expected row output, and every host-staged / NCCL same-prompt row must match it. Host-staged and NCCL must also match each other row-by-row.

## Latest Evidence

2026-06-14, single-node 2x RTX 5090 validation with `test_data/deepseek-v2-lite-ep2-cases.json` on the same `models/DeepSeek-V2-Lite` snapshot for HF, host-staged, and NCCL. The HF truth source used `AutoModelForCausalLM.generate(..., do_sample=false, use_cache=true)` with `torch==2.7.0+cu128` and `transformers==4.40.2`. The Rust gate emitted schema-2 case-set JSON for host-staged and NCCL, including row-level outputs for the same-prompt batch cases.

Review artifacts from that run were written under `target/accuracy/dsv2-lite-ep2-review/`:

- `target/accuracy/dsv2-lite-ep2-review/hf.json`
- `target/accuracy/dsv2-lite-ep2-review/host-staged.json`
- `target/accuracy/dsv2-lite-ep2-review/nccl.json`
- `target/accuracy/dsv2-lite-ep2-review/comparison.json`

Comparison result:

- `case_count=5`.
- Classification: `all_token_text_exact`.
- Per-case classifications: `capital_16_bs1`, `code_16_bs1`, `hello_16_bs1`, `hello_16_bs4`, and `hello_16_bs8` all reported `all_token_text_exact`.
- Warnings: none.
- Batch rule: HF single-row output was broadcast as the expected row output; every host-staged and NCCL row for `hello_16_bs4` and `hello_16_bs8` matched HF and matched the other backend row-by-row.

2026-05-30, single-node 2 GPU validation with the same `models/DeepSeek-V2-Lite` snapshot for all three outputs. The model snapshot metadata recorded commit `604d5664dddd88a0433dbae533b7fe9472482de0`. The HF truth source used `AutoModelForCausalLM.generate(..., do_sample=false, use_cache=true)` with `torch==2.7.0+cu128` and `transformers==4.40.2` on 2x A800-SXM4-80GB:

The comparison gate must be run with an HF JSON dumped on the same model directory and runtime as the pegainfer outputs. The Rust E2E keeps known HF-confirmed hash pairs for this narrow `Hello`/16 shape because the same snapshot has produced different greedy text on RTX 5090 and A800 while still matching HF on each host. This does not claim a model-runtime improvement, a manual-loop root cause, or a transport issue.

| Source | Backend | Tokens | Token SHA256 | Text SHA256 | Text |
| --- | --- | ---: | --- | --- | --- |
| HF | `generate(use_cache=true)` | 16 | `d05a7b0f0ac6435fb51040582a337d8b6d72844dd61194daa1b3090fa0e16ce8` | `4aaafbe4b3a46bc5b9ab5ea8d09d5fad71225006c2e234e87a928e3265b387c6` | `, I am a 20 year old female and I have been having a` |
| pegainfer | host-staged | 16 | `d05a7b0f0ac6435fb51040582a337d8b6d72844dd61194daa1b3090fa0e16ce8` | `4aaafbe4b3a46bc5b9ab5ea8d09d5fad71225006c2e234e87a928e3265b387c6` | `, I am a 20 year old female and I have been having a` |
| pegainfer | NCCL | 16 | `d05a7b0f0ac6435fb51040582a337d8b6d72844dd61194daa1b3090fa0e16ce8` | `4aaafbe4b3a46bc5b9ab5ea8d09d5fad71225006c2e234e87a928e3265b387c6` | `, I am a 20 year old female and I have been having a` |

Known HF-confirmed static E2E pairs for snapshot `604d5664dddd88a0433dbae533b7fe9472482de0`:

| Host | Token SHA256 | Text SHA256 | Text |
| --- | --- | --- | --- |
| 2x RTX 5090, torch 2.7.0, transformers 4.40.2 | `4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225` | `0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347` | `, I am a 19 year old girl from the UK. I am` |
| 2x A800-SXM4-80GB, torch 2.7.0, transformers 4.40.2 | `d05a7b0f0ac6435fb51040582a337d8b6d72844dd61194daa1b3090fa0e16ce8` | `4aaafbe4b3a46bc5b9ab5ea8d09d5fad71225006c2e234e87a928e3265b387c6` | `, I am a 20 year old female and I have been having a` |

Classification: `all_token_text_exact`.

- HF vs host-staged: token-exact and text-exact; no first different token.
- HF vs NCCL: token-exact and text-exact; no first different token.
- Host-staged vs NCCL: token-exact and text-exact; this run does not show an NCCL transport regression.

Accuracy fixes covered by this gate:

- DeepSeek-V2 RoPE host path now matches HF's pair permutation and bf16 multiply/add materialization.
- YaRN inv-frequency and `mscale_all_dim` attention softmax scaling are applied in the host attention path.
- Host attention now rounds attention scores/probabilities through bf16 at the HF materialization points.
- DeepSeek-V2 RMSNorm now rounds the normalized hidden to bf16 before multiplying the bf16 norm weight, matching the HF module.
- MoE gate logits now use the HF fp32 gate projection, and selected experts are accumulated in deterministic expert-id order after top-k selection.
- MoE routed expert output is materialized before adding shared experts, matching HF's `moe_infer(...).to(bf16) + shared_experts(...)` structure.
- Fused `silu_mul` now matches the existing non-fused `silu_mul` bf16 behavior by rounding `SiLU(gate)` before multiplying by `up`.
