# HD256 / HD512 QK-RoPE Smoke Oracles

**Created**: 2026-08-27
**Last touched**: 2026-08
**TL;DR**: The hd256/hd512 QK-RoPE device gates keep exactly three closed-form numerical anchors
(hd256 full-rotation and partial-tail, hd512 partial-rope) and derive every other expectation from
a GPU arm that an anchor already certifies. The chain is anchor → flat kernel → paged prefill →
paged decode: each link catches a failure the link above it cannot, and no link re-implements
RMSNorm, RoPE, or the paged address formula a second time. Adding a fourth dispatch variant means
extending the chain, not writing a new host oracle.

## Why the chain exists

Before #943 the two gate files carried seven closed-form tests and two independent copies of
`inv_rms`, `normed`, `expected_prep`, `expected_full`, `expected_pool`, and `assert_pool`. That is a
second implementation of the CUDA operator's substance, living in Rust, that has to evolve in
lockstep with the `.cu` contract. It fails in two ways that a passing test cannot show you:

- The host oracle drifts from the kernel and the gate goes red for a reason that is not a kernel bug.
- A specification mistake is made once and copied into both the kernel and its local oracle, so both
  are wrong and the gate is green.

The fix is not a shared oracle module — that relocates the second implementation instead of removing
it. The fix is to stop producing expected values on the host wherever a GPU arm that is already
certified can produce them.

## The chain

```
  closed-form anchor (host)          the only retained RMSNorm/RoPE formulas
        │ certifies
        ▼
  flat kernel                        hd256: qk_norm_rope_prefill_hd256_plain
  (contiguous, no pool)              hd512: qk_norm_partial_rope_batched_decode_hd512
        │ is the value oracle for
        ▼
  paged prefill <false>              the only retained address derivation
  (values from flat, addresses from
   PagedKvLayout, every other slot 0)
        │ is the bitwise oracle for
        ▼
  paged decode <true>                zero host math, zero addressing
  (equivalent metadata ⇒ bit-identical)
```

Each link is load-bearing against a different defect:

| Defect | Caught by |
| --- | --- |
| RMSNorm or RoPE pairing/tail formula wrong | the closed-form anchor |
| Paged address formula wrong (right value, wrong slot) | paged prefill vs flat, plus the exact-zero sentinels |
| Stray pool write outside the request's positions | the exact-zero sentinels on unreferenced pages and the other layer |
| Per-token metadata routing wrong (CSR window, per-token origin, `positions[]`) | paged decode vs paged prefill, bit for bit |

What makes the chain sound rather than circular: the flat kernel and the paged template share
`rms_norm_elem_*` and `apply_rope_pair_*` as device functions, so a bug in the shared math surfaces
at the anchor, which tests the flat kernel directly. The lower links then only have to answer
"did the certified value reach the right place, for the right row".

## Retained coverage

Three closed-form anchors, and nothing else on the host reconstructs operator math.

| Anchor | Contract it pins |
| --- | --- |
| `hd256 full_rotation_matches_closed_form` | `rotary_dim == HD`: the Gemma 4 local-layer production config, where the pass-through tail is empty and no thread takes the `d >= rotary_dim` branch |
| `hd256 partial_rotation_exercises_tail` | `rotary_dim < HD`: the only config that exercises the pass-through tail |
| `hd512 decode_prep_matches_closed_form` | hd512 rotary_dim is always 128 < 512, so one run covers rope-lo, rope-hi and tail together — a second hd512 anchor would be redundant |

Two host expressions survive outside the anchors, both one line, both the weightless V norm
(`x * inv_rms(x)`): hd256's V band reduces a separate `v_batch` input and hd512's V is the K=V fork,
so neither flat kernel emits a V the paged arm could be compared against. One multiply is not a
second implementation, and the alternative — driving a flat kernel with all-ones norm weights and an
identity RoPE row to synthesise a V oracle on device — couples the test to the period-3 cos/sin table
being unchanged, which is a worse trade for one multiply.

## Why the pool assertions are bitwise

`assert_pool_bits` compares bf16 bit patterns, not floats within a tolerance. Every non-zero
expectation is either a value another GPU arm produced from the same input at the same position — so
the two cannot differ at all — or the one-line V norm, which is reproducible exactly because three
things hold at once:

- `csrc/shared/*.cu` compiles without `--use_fast_math`. That flag is scoped to the vendored
  `k3_flash_kda` unit in `build.rs`, so `1.0f / sqrtf(x)` here is the correctly-rounded IEEE
  division and square root the host also computes, not the `rsqrt` approximation.
- Test inputs are constant across each head vector, so summing `HD` copies of `x²` in the kernel's
  reduction tree is exact in f32 and `total / HD` lands on `x²` with no rounding.
- Neither `x * inv_rms` nor `total / HD + eps` gives nvcc a multiply-add to contract into an FMA.

A pool failure that is off by one ulp therefore means one of those three stopped holding — a new
nvcc flag, a non-constant test input, a restructured reduction — not that the tolerance was too
tight. Loosening the assertion is the wrong fix; find which one changed.

The refusal gates (`rejects_bad_rotary_dim`, `rejects_position_beyond_cos_table`,
`prefill_rejects_undersized_kv_pool`) and the row-offset metamorphic gates
(`decode_prep_row_offset_serves_only_the_suffix`, `prefill_prep_row_offset_serves_only_the_suffix`,
`split_read_row_offset_serves_only_the_suffix`) carry no operator math and are untouched by this
discipline. Do not add operator math to them.

## What the model-level gate does and does not cover

`pegainfer-gemma4/src/layer_oracle.rs` replays the HF golden fixture's layer probes through the real
implementation on the real checkpoint, covering both layer types — hd256 sliding and hd512 global.
It is genuine external corroboration for the production configuration, and it is **not** a substitute
for the anchors: it is `#[ignore]`d, needs the pinned 12B checkpoint via `PEGAINFER_TEST_MODEL_PATH`
plus a GPU, and never runs by default. It also only ever sees the production `rotary_dim`, so the
partial-tail contract has no coverage there at all.

Retiring an anchor in favour of that gate would move a routinely-runnable check onto one that nobody
executes. See `docs/conventions/migration-defense.md`: a deleted defence needs a named successor that
actually runs.

## Running the gates

CI compiles these targets but never runs them — they need a device. Both test targets are
auto-discovered with no `required-features`, so the default feature set is enough:

```bash
PEGAINFER_REQUIRE_GPU=1 cargo test --release -p pegainfer-kernels --test hd256_qk_rope_plain_smoke --test hd512_qk_rope_smoke -- --nocapture
```

`PEGAINFER_REQUIRE_GPU=1` is not optional for a formal gate. Without it,
`tests/common/mod.rs::device_or_skip` treats a missing device as a skip and the whole suite passes
green without executing anything.

The trap binaries (`hd512_qk_rope_trap`, `hd512_qk_rope_trap_page`, `hd256_decode_csr_trap`) stay in
their own targets on purpose: `__trap()` poisons the CUDA context for whatever runs next in the same
binary.

## Negative controls

A gate built out of GPU-vs-GPU comparisons can pass because both arms are equally broken, so changing
one of these gates means re-earning confidence in it. Perturb the kernel, confirm red, revert. These
three were run on an A40 (sm_86, CUDA 12.8) against the gates as they stand:

| # | Perturbation | Turns red | How it reports |
| --- | --- | --- | --- |
| 1 | `+ 1` on the `kv_head * HD` term in `paged_kv_offset_hd256_plain` / `paged_kv_offset_hd512` | both `*_lands_*_at_layout_addresses` | `pool (a zero expectation is a slot the kernel must not touch)[59392]: got 0, expected 129` (hd256), `[14848]: got 0, expected 65` (hd512) |
| 2 | `page_origins[token]` → `0` in the `PER_TOKEN_META` branch | both `paged_decode_equals_paged_prefill_*` | `CUDA_ERROR_LAUNCH_FAILED` at the first D2H |
| 2b | `csr_page_row_checked(..., token, ...)` → `..., 0, ...` | both `paged_decode_equals_paged_prefill_*` | `per-token pool writes vs the whole-window run[57344]: got -129, expected 0` (hd256), `[14336]: got 1, expected 0` (hd512) |

Control 2 and 2b are both needed, and 2b is the one that matters. With the origin ignored, the last
row computes a row index past the end of its own window, so the kernel's own `__trap()` fires and the
launch dies before anything is compared — red, but it is the device guard doing the work, not the
assertion. Pinning every row to row 0's window instead stays in bounds, no guard fires, and the only
thing that can catch it is the bitwise pool comparison. A control that only ever trips a `__trap()`
proves the kernel guards itself, not that the gate would notice a wrong-but-in-bounds page.

Control 2b also shows why the per-token arm compresses each row's window to that row's own page,
giving the rows distinct non-zero origins (0, 1, 1, 2) over four different table spans. Handing every
row the full table at origin 0 would make the decode arm agree with the prefill arm whether or not
the per-token metadata is read at all, and the gate would pass vacuously. Keep the origins distinct
if this test is ever reworked.

**Pass `--no-fail-fast`.** `cargo test` stops after the first test binary that fails, so a control run
without it reports hd256 red and never executes hd512 at all — which looks exactly like a control
that passed on both. Filter to the single test under control as well (`-- --test-threads=1 <filter>`):
control 2 traps, and a trap poisons the CUDA context for every other test sharing that binary.

## The assumption to watch

The bottom link assumes paged prefill and paged decode remain two instantiations of one template
(`qkv_norm_rope_paged_prefill_hd256_plain_kernel<PER_TOKEN_META>`,
`qk_norm_partial_rope_paged_prefill_hd512_kernel<PER_TOKEN_META>`), differing only in how `pos`, the
page window, and the origin are fetched. If a future change forks decode into a standalone kernel for
performance, the bitwise comparison stops being "did the metadata route correctly" and becomes "do
two independent implementations happen to agree" — still useful, but no longer equivalent to the
closed-form coverage it replaced. Whoever forks it owns re-deciding what certifies the decode arm.

## Next step

Extend the chain rather than the host oracle. A new dispatch variant gets compared against the arm
one link above it; a genuinely new operator contract — a rotary regime with different control flow,
a new norm — gets its own small anchor, and says in its doc comment which control flow makes it
distinct from the existing three.
