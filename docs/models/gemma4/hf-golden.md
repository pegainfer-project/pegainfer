# Gemma 4 HF golden fixtures

**TL;DR:** three Hugging Face references, committed for Gemma 4 12B and dumpable for another size
under a tag. `test_data/gemma4-12b-hf-golden.safetensors` covers the window and everything below
it — layer-boundary activations at both ends of both layer types, plus top-64 logprobs, over a
single-token, a nine-token and a 1024-token case.
`test_data/gemma4-12b-hf-window-golden.safetensors` goes past it, recorded under both attention
backends. `test_data/gemma4-12b-hf-longctx-golden.safetensors` takes the same teacher-forced
comparison to 16384 and 32768 tokens for the raised serving ceiling. The `12b` in those names is
the fixture tag: `PEGAINFER_GEMMA4_FIXTURE_TAG` selects another set under the same three names, and
a reference whose tower does not fit on one card dumps with `--device auto`.

Last touched: 2026-10.

## What the base fixture contains

| case | tokens | probes | what it answers |
| --- | --- | --- | --- |
| `single` | 1 (BOS) | yes | softmax over one key is 1.0, so query, RoPE and mask drop out; V/O, the norms, the MLP and the softcap stay on the path |
| `short` | 9 | yes | the compact multi-token case — causal masking and non-zero positions are live from two tokens on |
| `edge` | 1024 | no | the widest prefill that evicts nothing, at exactly `sliding_window` |

No gate reads the probe activations any more: the layer-probe comparison ran a test-only second
implementation of the decoder layers, and it went when that implementation did. The probes stay in
the fixture and in the dumper, so a future per-layer gate needs no re-dump.

Global layers have no `v_proj` — the value is the `k_proj` output on its scale-free branch — so
`single` exercises `k_proj` too. The window edge is 1024 rather than 1023, measured rather than
read off the mask: changing token 0 still moves the last position of layer 0's output at 1023 and
at 1024, and stops at 1025. The window admits `sliding_window` keys inclusive of the current one.

Per case: `{case}_tokens` (int32), `{case}_topk_ids` (int32 `[P, 64]`), `{case}_topk_logprobs`
(fp32 `[P, 64]`, from `log_softmax` over fp32 logits). Probed cases add `{case}_hidden`, bf16
`[8, P, 3840]`, first axis being the cut list below — bf16 because that is the dtype the model
computes in, so widening would store a converted value rather than the reference one. `edge`
carries no probes: at window width they would dwarf the file and nothing reads them.

Sampled ids skip all 24 special and added tokens. Among them are the image and audio ids, which
are exactly the inputs text-only serving must reject, so a golden containing them would compare
against a request the engine will never serve.

## The eight cuts

Probe layers are read out of `layer_types`, not hardcoded — both ends of both types, which
exercises layer-type dispatch at each end. For 12B that is sliding 0/46 and global 5/47.

Cuts are layer *boundaries*: the input of layer `i` and the output of layer `i-1` are one
activation. Layers 46 and 47 are adjacent, which is why there are eight cuts and not nine —
`global_last_in` is the same tensor as `sliding_last_out`.

```
sliding_first_in   the scaled embedding, before any layer
sliding_first_out
global_first_in
global_first_out
sliding_last_in
sliding_last_out   also the input of global_last
global_last_out
final_norm_out     after the final RMSNorm, before the LM head
```

`final_norm_out` is what keeps the tail diagnosable: without it a logprob mismatch cannot be
attributed between the final norm, the tied LM head and the softcap.

## Facts this reference pins

**The embedding scale is 62.0, not `sqrt(3840)` = 61.9677.** The scale is a buffer cast to the
weight dtype *before* the multiply, so bf16 rounding is part of the reference — worth 5.2e-4
relative, far too large to write off as accumulation noise, and recorded in the metadata as
`embed_scale_bf16`.

**Text is causal.** `use_bidirectional_attention` is `"vision"` at 12B and the modelling code
reads it as `is_causal = value != "all"`.

**The layer output is scaled last.** Each decoder layer ends `hidden_states *= layer_scalar`,
after both residual adds — so that tensor applies to the layer's output, not to either branch.

**Logits are softcapped at 30.0** via `tanh(logits / 30) * 30`, after the LM head. The dumper
refuses to write a fixture whose logits exceed the cap.

## The window-crossing fixture

`gemma4-12b-hf-window-golden.safetensors` answers a different question: whether attention still
agrees with the reference once the oldest keys have aged out of the window. Four prompt lengths —
1023, 1024, 1025 and 4096 — each followed by eight teacher-forced continuation tokens from the
corpus, with the top-64 ids and logprobs recorded at the last prompt position and after each forced
token.

Both sdpa and eager are recorded because a token-for-token reference is not reachable at this
depth: the per-step top1-top2 margin collapses once the context passes ~1000 tokens, and the two
backends then continue the same prompt in different directions. The pair is what gives the gate a
floor — the gap the reference already has with itself.

Per case: `{case}_prompt` and `{case}_teacher` (int32), plus `{case}_sdpa_ids` /
`{case}_sdpa_logprobs` and `{case}_eager_ids` / `{case}_eager_logprobs` (`[9, 64]`, int32 and
fp32). The name follows `qwen35-*-hf-long-golden`: one model line, a second context régime.

## The long-context fixture

`gemma4-12b-hf-longctx-golden.safetensors` extends the window fixture's question to the raised
serving ceiling: 16384- and 32768-token prompts from the same corpus, each followed by eight
teacher-forced continuation tokens, with the top-64 ids and logprobs recorded at the last prompt
position and after each forced token — proportional RoPE and global attention far past the window
fixture's 4096.

At these depths eager does not fit next to the reference tower on the dump device, so both cases
record sdpa alone and the manifest names them in `eager_skipped`. A case without its own
dual-backend pair borrows the window fixture's deepest dual case (`w4096`) for its tolerance and
top-1 floor — the widest measured agreement bound available. That loan is only meaningful if both
fixtures were dumped under the same reference release, so the gate requires the two manifests to
name the same Transformers version, on top of the same checkpoint revision. The gate also runs the
widest case a second time in 2048-token chunks, the raised ceiling's production prefill shape.

Per case: `{case}_prompt` and `{case}_teacher` (int32), plus `{case}_sdpa_ids` /
`{case}_sdpa_logprobs` (`[9, 64]`, int32 and fp32).

## Regenerating

```bash
python tools/accuracy/dump_gemma4_hf_golden.py <checkpoint-dir> \
    test_data/gemma4-12b-hf-golden.safetensors \
    --source-repo google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

python tools/accuracy/dump_gemma4_window_golden.py <checkpoint-dir> \
    test_data/gemma4-12b-hf-window-golden.safetensors \
    --source-repo google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

python tools/accuracy/dump_gemma4_longctx_golden.py <checkpoint-dir> \
    test_data/gemma4-12b-hf-longctx-golden.safetensors \
    --source-repo google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
```

## Another size, under a tag

The three names carry a tag, `12b` for the committed set. Dumping the same three references from
another checkpoint under another tag, and pointing the runner at that tag, is what gates that size;
nothing about the fixtures' shape changes, only which checkpoint they answer to. Each fixture
records the checkpoint's own revision and weight-file digests, and every gate checks them before it
loads anything, so a set can only be run against the checkpoint it came from.

The reference tower is what limits this. `--device auto` shards it over every visible GPU, which is
how a 31B window or long-context reference is dumped: the prompts at 16384 and 32768 tokens do not
fit beside a 60 GiB tower on one card. The base fixture's prompts do fit, so it takes a single
device. `--device` defaults to `cuda:0`.

```bash
# the base fixture, one card
python tools/accuracy/dump_gemma4_hf_golden.py <31b-checkpoint-dir> \
    test_data/gemma4-31b-hf-golden.safetensors \
    --source-repo google/gemma-4-31B-it --revision <sha> --device cuda:0

# the two deep ones, tower sharded over the visible GPUs
python tools/accuracy/dump_gemma4_window_golden.py <31b-checkpoint-dir> \
    test_data/gemma4-31b-hf-window-golden.safetensors \
    --source-repo google/gemma-4-31B-it --revision <sha> --device auto

python tools/accuracy/dump_gemma4_longctx_golden.py <31b-checkpoint-dir> \
    test_data/gemma4-31b-hf-longctx-golden.safetensors \
    --source-repo google/gemma-4-31B-it --revision <sha> --device auto
```

The generate fixture the prompt-backed gates read follows the same naming
(`test_data/gemma4-<tag>-generate.safetensors`, `dump_gemma4_generate.py`), and so does the chat
reference `docs/models/gemma4/tokenizer.md` describes. A sharded checkpoint is fingerprinted by its
index plus each shard's header rather than by a single `model.safetensors`, so the provenance check
holds for both layouts. Only the 12B set is committed; another tag's files are local to the box that
dumped them, which is why the runner takes paths from the environment.

The generate fixture's three prompts do not carry over between sizes, because which continuations a
checkpoint is sure of depends on the checkpoint: the committed set holds for 12B and 31B and gives
1, 0 and 1 decisive tokens on the 26B routed one where the dumper needs twenty, while a counting
prompt holds the 26B tower for all fifty tokens and stops after two on 31B. The 26B set is
`tools/accuracy/gemma4-26b-prompts.json`, passed with `--prompts`; the fixture records the set it
was dumped with. The margins it was chosen on:

| case | prompt | decisive tokens | min margin |
| --- | --- | --- | --- |
| `a` | `1, 2, 3, ... 16,` | 50 | 9.94 |
| `b` | Apache license header | 50 | 6.38 |
| `c` | `2024-01-01` onwards, one per line | 50 | 3.50 |

A checkpoint HF cannot load needs a second directory. The 26B routed one is NVFP4, so the reference
tower is a bf16 copy of its weights, written by `tools/accuracy/dequantize_gemma4_nvfp4.py`
(ModelOpt unpacks the experts, the only quantized tensors in this checkpoint). The base fixture is
dumped from that copy with `--fingerprint-dir` pointing at the checkpoint the gates serve:
`file_sha256` pins the served checkpoint, and a `reference_tower` block records `--tower` (how the
copy was made) and the copy's own digests. The runner checks digests on the base fixture alone and
holds the other three to its revision, so the window, long-context and generate fixtures are dumped
from the same copy under the same `--source-repo` and `--revision`.

```bash
# transformers 5.16.1, torch 2.13.0, nvidia-modelopt 0.46.1
python tools/accuracy/dequantize_gemma4_nvfp4.py <26b-nvfp4-checkpoint> <26b-bf16-copy> \
    --device cuda:0

REV=a19cfe00be84568a6867111c9a68c9c44fdcffe6   # nvidia/Gemma-4-26B-A4B-NVFP4
python tools/accuracy/dump_gemma4_hf_golden.py <26b-bf16-copy> \
    test_data/gemma4-26b-hf-golden.safetensors \
    --source-repo nvidia/Gemma-4-26B-A4B-NVFP4 --revision $REV --device auto \
    --fingerprint-dir <26b-nvfp4-checkpoint> \
    --tower "dequantize_gemma4_nvfp4.py, nvidia-modelopt 0.46.1"
python tools/accuracy/dump_gemma4_generate.py <26b-bf16-copy> \
    test_data/gemma4-26b-generate.safetensors \
    --source-repo nvidia/Gemma-4-26B-A4B-NVFP4 --revision $REV --device auto \
    --prompts tools/accuracy/gemma4-26b-prompts.json
# the window and long-context dumpers take the same arguments as the 31B ones above,
# with the bf16 copy, this repo and this revision
```

The gates then compare our serving path in NVFP4 against HF in bf16 on the same weights. The
measured gap includes the quantization; the bound, twice the tower's own sdpa and eager spread,
does not.

Two runs against the same checkpoint produce the same bytes, so regeneration is checked with
`sha256sum` alone. The current fixtures are

| file | sha256 |
| --- | --- |
| `gemma4-12b-hf-golden.safetensors` | `c30a338d499512e6f0505bd12b184ebb5af9d7536f0b7fc9ea2bdfdb18b1a46d` |
| `gemma4-12b-hf-window-golden.safetensors` | `b72edd51a5977592f3d4b637152aaf794a33e356c9599ab14035dacbb9574c0e` |
| `gemma4-12b-hf-longctx-golden.safetensors` | `1c8442a51913f858af6bc7205bd85c5b67db91373d7bdbdc034bf1ce2e86889d` |

That only holds because the metadata is a **single sorted-JSON key**. safetensors serializes its
metadata map in randomized order, so a multi-key block makes two runs differ byte for byte while
carrying identical content — which is how a fixture that *is* reproducible can look like one that
is not.

Provenance is passed in, not inferred: a checkpoint directory carries no record of where it came
from. The base fixture's metadata records sha256 of `config.json`, `generation_config.json` and the
safetensors *header* — the header pins the tensor layout without reading 22 GiB of payload, the
revision pins the payload. The window and long-context fixtures record the source repo, the
revision and the Transformers release: their gates validate the running checkpoint against the base
fixture's hashes, then require every fixture to name the same revision, so one set of hashes covers
all three. The long-context gate additionally requires its Transformers release to match the window
fixture's, because it borrows that fixture's floor. **Transformers 5.11.0** is verified to load `gemma4_unified`; the
checkpoint declares `5.10.0.dev0`, a development build that was never released, so the pin is the
release that was tested rather than a guess at what that build became.

## Running the gates these fixtures serve

The gates that consume them are `#[ignore]`: they need the checkpoint and a device, so CI only
compiles them. `scripts/gemma4_gates.sh` runs them:

```bash
PEGAINFER_TEST_MODEL_PATH=<12b-checkpoint> \
  PEGAINFER_NVFP4_MODEL=<26b-checkpoint> \
  PEGAINFER_GATE_GPU=<index-or-UUID> scripts/gemma4_gates.sh [name-filter]
```

`PEGAINFER_GEMMA4_FIXTURE_TAG` selects the set, defaulting to `12b`; the runner exports each
fixture's path (`PEGAINFER_GEMMA4_GOLDEN`, `_WINDOW_GOLDEN`, `_LONGCTX_GOLDEN`, `_GENERATE`,
`_CHAT_GOLDEN`) so the gates read the tagged files rather than hard-coded ones, and it holds the
whole set against the checkpoint's digests before the first load. To run the suite at 31B:

```bash
PEGAINFER_GEMMA4_FIXTURE_TAG=31b \
  PEGAINFER_TEST_MODEL_PATH=<31b-checkpoint> \
  PEGAINFER_NVFP4_MODEL=<26b-checkpoint> \
  PEGAINFER_GATE_GPU=<index-or-UUID> scripts/gemma4_gates.sh
```

An unfiltered run owns both checkpoint-backed suites. The sync/lane parity, ragged graph/eager
parity and shared/green lifecycle gates run once with the dense checkpoint and once with the routed
checkpoint; the runner binds `PEGAINFER_TEST_MODEL_PATH` to the selected profile for each process.
Fixture-backed and raised-context gates remain dense-only. A filter requires the inputs declared by
all execution profiles selected for that gate.

The isolated routed-block diagnostic uses the 26B checkpoint separately:

```bash
PEGAINFER_NVFP4_MODEL=<26b-checkpoint> \
  PEGAINFER_GATE_GPU=<index-or-UUID> \
  scripts/gemma4_gates.sh the_routed_block_matches_the_reference_formulas
```

This diagnostic localizes router, expert GEMM, and combine errors; it is not a serving E2E.

It refuses to start when the checkpoint, a fixture, the pinned metadata or a device is missing,
holds the crate's ignored set against the gate list it carries — so a gate cannot leave the suite
unnoticed — and runs one gate per process. For a GPU-backed selection it resolves the requested
device to its stable UUID, exports that UUID as the sole `CUDA_VISIBLE_DEVICES` entry, and holds a
non-blocking cross-process lock on it until the suite exits. Lock contention refuses the run before
the first gate; when no selector is supplied the runner uses an existing single-device
`CUDA_VISIBLE_DEVICES`, then physical device 0. This ownership and one-process execution are not
stylistic choices: repeated checkpoint loads inside one test binary exhaust a 48 GiB device, while two
runners sharing a card can fail each other's gates on allocation rather than on any assertion.

## Why hooks rather than `output_hidden_states`

That argument works here — the class declares `_can_record_outputs["hidden_states"]` and it
returns 49 tensors. It is unusable because **its last entry is the final norm applied to the last
layer's output, not that output itself** (`hs[-1]` equals `norm(layer_47_out)` bitwise, and does
not equal the raw layer-47 output). The fixture needs both `global_last_out` and `final_norm_out`,
and that argument can supply only one.
