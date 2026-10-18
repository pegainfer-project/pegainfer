# DeepSeek-V2-Lite Device-Resident NCCL Combine

> **TL;DR:** Issue #275 moves the NCCL decode combine path to reusable device-resident f32 scratch buffers. The retained `Hello` / 16-token gate stays HF / host-staged / NCCL exact, and the readiness report no longer lists the old combine H2D/D2H/allocation/sync blockers. Current NCCL graph-readiness blockers live in `status.md`.

Last touched: 2026-06

## Preparation

- **Read**:
  - `docs/index.md` - current docs live under domain folders such as `docs/models/deepseek-v2-lite/`.
  - `docs/models/deepseek-v2-lite/status.md` - DeepSeek-V2-Lite is still a feature-gated EP2 correctness and attribution target; host-staged remains the oracle.
  - `docs/models/deepseek-v2-lite/decode-attribution-gate.md` - acceptance uses the `Hello` / 16-token HF / host-staged / NCCL gate plus graph-readiness blockers.
  - `docs/models/deepseek-v2-lite/hf-accuracy-gate.md` - same-host HF, host-staged, and NCCL token/text exactness is the correctness standard.
  - `docs/models/deepseek-v2-lite/source-layout.md` - runtime responsibilities are split, and issue #275 was intentionally left as follow-up work.
  - `pegainfer-deepseek-v2-lite/src/runtime/moe.rs` - the pre-#275 NCCL combine path accumulated routed expert outputs in host `Vec<f32>` buffers, then copied H2D for NCCL and D2H before final H2D conversion.
  - `pegainfer-deepseek-v2-lite/src/nccl_backend.rs` - the pre-#275 NCCL combine path allocated send/recv device buffers inside each call and synchronized both streams.
  - `pegainfer-deepseek-v2-lite/src/runtime/readiness.rs` - the pre-#275 readiness report listed combine H2D, allocation, sync, and D2H blockers.
  - `pegainfer-kernels/src/ops/elementwise.rs` and `pegainfer-kernels/csrc/shared/elementwise.cu` - existing device f32/bf16 conversion helpers could be reused, but there was no f32 accumulation helper for bf16 expert output.
- **Relevant history**:
  - `docs/models/deepseek-v2-lite/status.md` - NCCL plus CUDA Graph is the preferred direction, but the current gate must not be described as production EP.
  - `docs/models/deepseek-v2-lite/source-layout.md` - local macOS checks are not enough for this path; remote 2-GPU validation is the real acceptance path.
- **Implemented**:
  1. Add a shared CUDA helper that accumulates a bf16 single-token expert output into a f32 device contribution buffer at a selected token row.
  2. Re-export that helper through `pegainfer-core::ops`.
  3. Add reusable NCCL combine scratch buffers inside `NaiveNcclEp2Backend`, clear the f32 send scratch per MoE call, accumulate local/remote expert outputs on the owning device, all-reduce device buffers, and cast rank0 f32 result to bf16 on device.
  4. Update graph-readiness blockers and attribution wording so removed combine H2D/D2H/allocation/sync blockers are no longer claimed, while the remaining host routing and dense-exchange blockers stay explicit.
  5. Run formatting and local compile gates, then use the provided remote GPU host for the DeepSeek-V2-Lite EP2 exactness and attribution gates.
- **Risks / open questions**:
  - Device f32 accumulation must preserve the existing expert-id accumulation order before the final bf16 cast.
  - Dense exchange and host route selection still blocked full decode CUDA Graph capture at the time of issue #275; keep the current blocker list in `status.md`.
  - The provided SSH credential should stay local to the validation session and must not be echoed into docs or final output.

## Execution Log

Validated 2026-06-08 on the provided 2x RTX 5090 host with DeepSeek-V2-Lite snapshot `604d5664dddd88a0433dbae533b7fe9472482de0`, CUDA 12.8, Rust 1.96.0, Torch 2.7.0+cu128, Transformers 4.40.2, and NCCL from the Python CUDA wheel path.

Commands run:

```bash
cargo test --offline --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 --no-run

cargo clippy --offline --release -p pegainfer-deepseek-v2-lite \
  --features deepseek-v2-lite --bins --tests -- \
  -D warnings \
  -A clippy::option_option \
  -A clippy::manual_midpoint \
  -A clippy::needless_range_loop

python tools/accuracy/hf_dump_dsv2_lite_ep2_greedy.py \
  --model-path models/DeepSeek-V2-Lite \
  --prompt Hello \
  --output-len 16 \
  --out target/accuracy/dsv2-lite-ep2/hf.json

PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite \
PEGAINFER_DSV2_LITE_E2E_JSON_OUT=target/accuracy/dsv2-lite-ep2/host-staged.json \
  cargo test --offline --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

PEGAINFER_TEST_MODEL_PATH=models/DeepSeek-V2-Lite \
PEGAINFER_DSV2_LITE_EP_BACKEND=nccl \
PEGAINFER_DSV2_LITE_E2E_JSON_OUT=target/accuracy/dsv2-lite-ep2/nccl-after-decouple-cleanup.json \
  cargo test --offline --release -p pegainfer-deepseek-v2-lite --features deepseek-v2-lite --test e2e_ep2 -- --nocapture

python tools/accuracy/compare_dsv2_lite_ep2_outputs.py \
  --hf target/accuracy/dsv2-lite-ep2/hf.json \
  --host-staged target/accuracy/dsv2-lite-ep2/host-staged.json \
  --nccl target/accuracy/dsv2-lite-ep2/nccl-after-decouple-cleanup.json \
  --out target/accuracy/dsv2-lite-ep2/comparison-after-decouple-cleanup.json \
  --require-all-exact

PEGAINFER_DSV2_LITE_EP_BACKEND=nccl \
  cargo run --offline --release -p pegainfer-deepseek-v2-lite \
    --features deepseek-v2-lite \
    --bin dsv2_lite_ep2_decode_attribution \
    -- --model-path models/DeepSeek-V2-Lite \
    --batch-size 1 \
    --out target/accuracy/dsv2-lite-ep2/candidate-nccl-attribution.json
```

Results:

- HF / host-staged / NCCL comparison: `all_token_text_exact`.
- Generated text: `, I am a 19 year old girl from the UK. I am`.
- Token SHA256: `4fb4c8825fe4d2c4a1d966da25c259abdf675f4de4548daa5d41aea7dfe30225`.
- Text SHA256: `0eedf11429e9ac13bb799c31665c6e9f70a1ac4493a08a3f3da9ecf39c1ec347`.
- Candidate NCCL attribution: `gpu_timing.sample_count=8384`, `failure_count=0`.
- Initial remote cleanup gate: package `--bins --tests` clippy passed with only three explicit allows for then-existing lints (`pegainfer-core::logging` `option_option`, and two `host_ops` test lints).

Follow-up review gate on 2026-06-09 after fixing those lints:

```bash
cargo fmt --all --check

cargo clippy --release -p pegainfer-deepseek-v2-lite \
  --features deepseek-v2-lite --bins --tests -- -D warnings
```

Both commands passed on the same remote source copy after syncing the follow-up `host_ops.rs` and `logging.rs` fixes. The clippy command ran without `clippy::manual_midpoint`, `clippy::needless_range_loop`, or `clippy::option_option` allows.

Before/after readiness comparison for the same model snapshot and diagnostic shape at issue #275:

| Report | Readiness blockers |
| --- | --- |
| Baseline NCCL attribution | `nccl_dense_exchange_allocates_per_call`, `nccl_dense_exchange_syncs_rank_streams`, `nccl_route_iteration_on_host`, `nccl_contribution_accumulation_on_host`, `nccl_combine_h2d_contribution_copy`, `nccl_combine_allocates_per_call`, `nccl_combine_syncs_rank_streams`, `nccl_combine_d2h_result_copy` |
| Candidate NCCL attribution | `nccl_dense_exchange_allocates_per_call`, `nccl_dense_exchange_syncs_rank_streams`, `nccl_route_iteration_on_host`, `nccl_expert_accumulation_host_directed` |

Removed blockers:

- `nccl_contribution_accumulation_on_host`
- `nccl_combine_h2d_contribution_copy`
- `nccl_combine_allocates_per_call`
- `nccl_combine_syncs_rank_streams`
- `nccl_combine_d2h_result_copy`

The candidate report replaces the old `nccl_contribution_accumulate` and `nccl_combine_to_device` sections with `nccl_contribution_accumulate_device` and `nccl_combine_clear`, while keeping the final `nccl_combine` section for the f32 all-reduce plus rank0 bf16 cast.

## Debrief

The implementation keeps host-staged unchanged as the correctness oracle. The NCCL backend now owns reusable rank0/rank1 f32 send/recv scratch buffers behind `DeviceCombineScratch`; each MoE call clears the f32 send scratch on device, accumulates one-token expert outputs into the owning rank's send scratch with a CUDA helper, runs the f32 NCCL all-reduce, and casts rank0's f32 result back to bf16 on device.

The final bf16 `HiddenStates` returned to the model is still allocated per combine call. That allocation is outside the removed NCCL contribution/result round trip, so issue #275 did not claim full CUDA Graph readiness. The current blocker list should stay in `status.md`.
