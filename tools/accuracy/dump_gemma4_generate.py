"""Dump the Gemma 4 HF greedy-generation reference.

    python tools/accuracy/dump_gemma4_generate.py <model-dir> <out.safetensors> \
        --source-repo google/gemma-4-12B-it \
        --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

Three prompts, each continued greedily with the KV cache on — the reference a
paged serving path must reproduce token for token. Prompt lengths stay far
below the sliding window, so no eviction is involved.

What gets dumped is measured, not assumed. A case is admitted only when HF
agrees with itself (identical greedy tokens under sdpa and eager) and a replay
reproduces the dump. Each case is then truncated to its decisive prefix: the
first step where either backend's top1-top2 logit margin falls to MIN_MARGIN or
below ends the reference, because past that point the argmax is a coin toss
between implementations rather than a fact about the model — one backend's
margin alone cannot certify a step the other would refuse. MIN_MARGIN is
calibrated from behaviour observed on this checkpoint: cross-implementation
flips appear at margins up to 1.69, while every admitted step at 2.69 or above
has held. MIN_TOKENS then refuses a prefix too short to gate anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_NEW_TOKENS = 50
MIN_MARGIN = 2.25
MIN_TOKENS = 20
METADATA_KEY = "gemma4_generate"
PROMPTS = {
    "a": "def is_prime(n):\n    \"\"\"Return True if n is prime.\"\"\"\n",
    "b": "<!DOCTYPE html>\n<html>\n<head>\n",
    "c": "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("out")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def greedy_with_margins(model, input_ids: torch.Tensor) -> tuple[list[int], list[float]]:
    """Greedy tokens plus the per-step top1-top2 logit margins."""
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    tokens = out.sequences[0, input_ids.shape[1] :].tolist()
    margins = []
    for step in (s[0].float() for s in out.scores):
        if not torch.isfinite(step).all():
            raise SystemExit("non-finite logits in the reference scores, refusing to dump")
        top2 = step.topk(2).values
        margins.append((top2[0] - top2[1]).item())
    return tokens, margins


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    outs: dict[tuple[str, str], list[int]] = {}
    margins: dict[tuple[str, str], list[float]] = {}
    prompt_ids: dict[str, list[int]] = {}
    for impl in ("sdpa", "eager"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, device_map=args.device, attn_implementation=impl
        )
        model.eval()
        for case, text in PROMPTS.items():
            input_ids = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
            prompt_ids[case] = input_ids[0].tolist()
            generated, step_margins = greedy_with_margins(model, input_ids)
            if impl == "sdpa" and (generated, step_margins) != greedy_with_margins(
                model, input_ids
            ):
                raise SystemExit(f"case {case}: generate is not reproducible, refusing to dump")
            margins[(impl, case)] = step_margins
            outs[(impl, case)] = generated
        del model
        torch.cuda.empty_cache()

    tensors: dict[str, torch.Tensor] = {}
    manifest: dict[str, object] = {
        "source_repo": args.source_repo,
        "revision": args.revision,
        "transformers": __import__("transformers").__version__,
        "max_new_tokens": MAX_NEW_TOKENS,
        "prompts": PROMPTS,
        "backend_agreement": "sdpa == eager, verified at dump time",
        "min_margin_gate": MIN_MARGIN,
        "decisive_prefix": f"generation truncated at the first step with margin <= {MIN_MARGIN}",
    }
    for case in PROMPTS:
        sdpa_tokens = outs[("sdpa", case)]
        eager_tokens = outs[("eager", case)]
        both = [
            min(a, b)
            for a, b in zip(margins[("sdpa", case)], margins[("eager", case)])
        ]
        cut = next((i for i, m in enumerate(both) if m <= MIN_MARGIN), len(both))
        if cut < MIN_TOKENS:
            raise SystemExit(
                f"case {case}: both-backend decisive prefix is only {cut} tokens "
                f"(< {MIN_TOKENS}); pick a more decisive prompt"
            )
        if sdpa_tokens[:cut] != eager_tokens[:cut]:
            div = next(
                i for i, (x, y) in enumerate(zip(sdpa_tokens, eager_tokens)) if x != y
            )
            raise SystemExit(
                f"case {case}: sdpa and eager diverge at {div} inside the decisive "
                "prefix; pick a higher-confidence prompt"
            )
        generated = sdpa_tokens[:cut]
        print(
            f"case {case}: prompt {len(prompt_ids[case])} tokens -> {len(generated)} "
            f"decisive, min margin {min(both[:cut]):.2f}"
        )
        tensors[f"{case}_prompt"] = torch.tensor(prompt_ids[case], dtype=torch.int32)
        tensors[f"{case}_generated"] = torch.tensor(generated, dtype=torch.int32)

    payload = json.dumps(manifest, sort_keys=True)
    save_file(tensors, args.out, metadata={METADATA_KEY: payload})
    digest = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    print(f"wrote {args.out} sha256 {digest}")


if __name__ == "__main__":
    main()
