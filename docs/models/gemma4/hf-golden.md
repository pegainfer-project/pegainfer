# Gemma 4 HF golden fixture

**TL;DR:** `test_data/gemma4-12b-hf-golden.safetensors` is the Hugging Face reference for Gemma 4
12B — layer-boundary activations at both ends of both layer types, plus top-64 logprobs, over a
single-token, a nine-token and a 1024-token (exactly the sliding window) case. It is the reference
the in-crate golden gates replay their probes against.

Last touched: 2026-08.

## What the fixture contains

| case | tokens | probes | what it answers |
| --- | --- | --- | --- |
| `single` | 1 (BOS) | yes | softmax over one key is 1.0, so query, RoPE and mask drop out; V/O, the norms, the MLP and the softcap stay on the path |
| `short` | 9 | yes | the compact multi-token case — causal masking and non-zero positions are live from two tokens on |
| `edge` | 1024 | no | the widest prefill that evicts nothing, at exactly `sliding_window` |

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

## Regenerating

```bash
python tools/accuracy/dump_gemma4_hf_golden.py <checkpoint-dir> \
    test_data/gemma4-12b-hf-golden.safetensors \
    --source-repo google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
```

Two runs against the same checkpoint produce the same bytes, so regeneration is checked with
`sha256sum` alone. The current fixture is
`c30a338d499512e6f0505bd12b184ebb5af9d7536f0b7fc9ea2bdfdb18b1a46d`.

That only holds because the metadata is a **single sorted-JSON key**. safetensors serializes its
metadata map in randomized order, so a multi-key block makes two runs differ byte for byte while
carrying identical content — which is how a fixture that *is* reproducible can look like one that
is not.

Provenance is passed in, not inferred: a checkpoint directory carries no record of where it came
from. The metadata also records sha256 of `config.json`, `generation_config.json` and the
safetensors *header* — the header pins the tensor layout without reading 22 GiB of payload, the
revision pins the payload. **Transformers 5.11.0** is verified to load `gemma4_unified`; the
checkpoint declares `5.10.0.dev0`, a development build that was never released, so the pin is the
release that was tested rather than a guess at what that build became.

## Why hooks rather than `output_hidden_states`

That argument works here — the class declares `_can_record_outputs["hidden_states"]` and it
returns 49 tensors. It is unusable because **its last entry is the final norm applied to the last
layer's output, not that output itself** (`hs[-1]` equals `norm(layer_47_out)` bitwise, and does
not equal the raw layer-47 output). The fixture needs both `global_last_out` and `final_norm_out`,
and that argument can supply only one.
