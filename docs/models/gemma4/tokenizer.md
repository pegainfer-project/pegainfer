# Gemma 4 tokenizer and chat template

**TL;DR:** All five chat renders reproduce the Hugging Face reference when content is flattened to strings — gated by `pegainfer-frontend/tests/gemma4_tokenizer_parity.rs` (runner-owned) against the pinned 12B checkpoint, with the other two sizes covered by inspection rather than by running. The token-id probes are retired: both sides run the same `tokenizers` crate, so they gated the Python wrapper's version skew rather than behavior. One divergence is open: under the frontend's default content format the system turn gains a trailing space. Contracts the engine must honour: BOS comes only from the chat template, EOS is declared in three places with three different values, the published generation defaults are sampled rather than greedy, and text-only serving rejects modality tokens before embedding and suppresses them before sampling.

Last touched: 2026-09

## The gate runs on 12B; the result carries to the other sizes by inspection

The executable gate is bound to the pinned 12B checkpoint — the fixture records that checkpoint's
file hashes and every test checks them first, so pointing it at another size fails on the
`tokenizer_config.json` hash before asserting anything. That binding is deliberate: it keeps the
guard an exact-file provenance check rather than a set of semantic exceptions.

What carries to 26B-A4B and 31B is the conclusion, not the run, and it carries because the files
are the same bytes. Against the published repositories:

| file | 12B | 26B-A4B | 31B |
| --- | --- | --- | --- |
| `tokenizer.json` | `cc8d3a0c…` (32,169,626 B) | same | same |
| `chat_template.jinja` | `4741bf6e…` (18,683 B) | same | same |
| `tokenizer_config.json` | `6520cdc2…` | `6068e357…` | `6068e357…` |

Those are Hugging Face object ids — sha256 for the LFS-stored `tokenizer.json`, git blob hashes
for the two small files — while the fixture records sha256 throughout, and the parity tests check
the checkpoint against it before asserting anything.

`tokenizer.json` and `chat_template.jinja` are byte-identical across all three sizes, so anything
this suite establishes about tokenization or rendering holds for all of them. The
`tokenizer_config.json` difference is a single field — `processor_class` is
`Gemma4UnifiedProcessor` at 12B and `Gemma4Processor` at the other two — which names the
multimodal processor and affects neither. Gating the other two sizes for real would mean recording
their revisions and hash sets and running each; that is not what this suite does.

## Reference fixture

`test_data/gemma4-tokenizer-golden.json` is dumped by

```bash
python tools/accuracy/dump_gemma4_tokenizer_golden.py <model-dir> <out.json> \
  --source-repo google/gemma-4-12B-it --revision <sha>
```

Repository and revision are required arguments rather than inferred: a checkpoint directory
carries no reliable record of where it came from, and guessing from a renamed local directory
would write a wrong name into the golden. Everything in the fixture is asserted by the tests
except those two and the transformers version, which are provenance — they record where the
reference came from and cannot be checked from here. No local path is recorded.

Both sides tokenize with the same `tokenizers` crate — Python's fast tokenizer wraps it and
`vllm-tokenizer` calls it directly — which is why the token-id probes retired: they compared one
implementation with itself and could only gate the Python wrapper's version skew. What remains,
and what the gate asserts, is the chat render: minijinja against the reference's Jinja2, the one
comparison with two genuinely independent sides.

The parity test carries `#[ignore]` because it needs the checkpoint; the maintainer runner
executes it, or run it directly against the pinned 12B one:

```bash
PEGAINFER_TEST_MODEL_PATH=<pinned-12B-checkpoint-dir> \
  cargo test --release -p pegainfer-frontend --test gemma4_tokenizer_parity -- --ignored
```

## The chat template lives in its own file

Gemma 4 ships `chat_template.jinja` rather than embedding the template in
`tokenizer_config.json`. The renderer's precedence is: an explicit `chat_template` override, then
a dedicated template file, then the tokenizer config entry — so passing no override (what the
server does) picks up the `.jinja`.

Rendered structure, from the reference:

```
<bos><|turn>user\nWhat is 2 + 2?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>
```

Two consequences. Gemma 4 accepts a native `system` role — unlike Gemma 3, which folded system
prompts into the first user turn. And `add_generation_prompt` opens a **thought channel**
(`<|channel>thought\n<channel|>`) on the model turn, so the assistant's first emission is
reasoning content; anything parsing the stream has to expect the channel markers.

## Open divergence: content format changes the system turn

The renderer resolves `content` into one of two shapes, and this template branches on which it
gets. Given a string it emits `{{ content | trim }}`; given a list of parts it emits
`{{ item['text'] | trim + ' ' }}` per part, leaving a trailing space. The server's default
(`Auto`) inspects the template, finds a content-item loop, and therefore selects the parts form —
so a system prompt renders as `You are terse. <turn|>` where the reference produces
`You are terse.<turn|>`. User turns are unaffected; only the system branch adds the space.

The parity test pins the string form, which matches the reference exactly for all five cases and
establishes that template discovery, precedence and the Jinja engine are faithful. Choosing which
format the server should use for this line is a separate decision: the setting is global to the
frontend today, so changing it touches every model line and needs its own review.

## BOS comes from the template, not the tokenizer

`add_special_tokens` was established as a no-op for this tokenizer by the retired token-id
probes (every probe encoded identically with it on and off); the fixture no longer carries them.
The leading `<bos>` in a chat request comes from the template. An engine that adds BOS
itself would double it.

## EOS is declared three times

These three values are read from the checkpoint's config files; they are metadata observations,
not something the parity run exercises.

| source | value |
| --- | --- |
| `config.json` | `[1, 106]` |
| `text_config` | `1` |
| `generation_config.json` | `[1, 106, 50]` |

`generation_config.json` carries the generation semantics and is the one to honour; taking the
outer list silently drops token 50. Whichever precedence the loader picks must be explicit rather
than an artefact of which file it happens to read first.

## The published defaults are sampled, not greedy

Also read from the checkpoint rather than gated by a test.

`generation_config.json` sets `do_sample: true`, `temperature: 1.0`, `top_k: 64`, `top_p: 0.95`
at all three sizes. Any greedy comparison against a reference implementation has to state the
override it applies. 12B alone adds `suppress_tokens: [258883, 258882]` — the end-of-audio and
end-of-image ids; that checkpoint-specific list must not be applied to 26B or 31B, which do not
declare it. Independently, text-only serving suppresses all six modality placeholders at every
size because none can be fed into the next text-tower step.

## Modality tokens are reachable from plain text

| token | id |
| --- | --- |
| `<\|image\|>` | 258880 |
| `<\|audio\|>` | 258881 |
| `<\|image>` (begin image) | 255999 |
| `<image\|>` (end image) | 258882 |
| `<\|audio>` (begin audio) | 256000 |
| `<audio\|>` (end audio) | 258883 |

These encode as single ids straight from user text — `"before <|image|> after"` becomes
`[15849, 236743, 258880, 1308]`. A text-only engine has no image or audio embedder, so an
admitted request carrying one of these ids would reach the embedding table with a placeholder that
maps to nothing meaningful.

The engine uses one shared six-id set at both boundaries: host-token validation rejects a prompt or
decode input before embedding, and the effective sampling suppression set unions those ids with the
checkpoint's own `suppress_tokens`. A placeholder therefore cannot be embedded, emitted to the
client, or returned as the next decode input.
