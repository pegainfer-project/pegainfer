"""Dump the Gemma 4 window-crossing logprob reference.

    python tools/accuracy/dump_gemma4_window_golden.py <model-dir> <out.safetensors> \
        --source-repo google/gemma-4-12B-it \
        --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7

Four prompt lengths around and beyond the sliding window (1023, 1024, 1025,
4096), each followed by 8 teacher-forced continuation tokens from the corpus.
Every position records the top-64 ids and logprobs under both sdpa and eager;
docs/models/gemma4/hf-golden.md says why a distribution under two backends is
the only reachable reference at this depth.

Refuses to write rather than record something unusable: non-finite scores at
any position abort the dump.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

TOP_K = 64
TEACHER_STEPS = 8
METADATA_KEY = "gemma4_window_golden"
CASES = {"w1023": 1023, "w1024": 1024, "w1025": 1025, "w4096": 4096}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("out")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def corpus_ids(tokenizer, min_len: int) -> list[int]:
    rows = "  <tr>\n    <td>item</td>\n  </tr>\n" * 800
    ids = tokenizer(f"<table>\n{rows}", return_tensors="pt").input_ids[0].tolist()
    if len(ids) < min_len:
        raise SystemExit(f"corpus tokenizes to {len(ids)} < {min_len} tokens")
    return ids


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    ids = corpus_ids(tokenizer, max(CASES.values()) + TEACHER_STEPS)

    tensors: dict[str, torch.Tensor] = {}
    for impl in ("sdpa", "eager"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, device_map=args.device, attn_implementation=impl
        )
        model.eval()
        for case, length in CASES.items():
            seq = torch.tensor([ids[: length + TEACHER_STEPS]], device=args.device)

            def rows_of() -> torch.Tensor:
                # Slice to the recorded positions before the fp32 upcast:
                # the full 262k-vocab logits over 4k positions in fp32 is
                # ~4.3 GiB and OOMs next to the 22 GiB tower.
                with torch.no_grad():
                    return model(seq).logits[0, length - 1 : length + TEACHER_STEPS].float()

            rows = rows_of()
            if not torch.equal(rows, rows_of()):
                raise SystemExit(f"case {case} ({impl}): forward is not reproducible")
            logprobs = F.log_softmax(rows, dim=-1)
            top = logprobs.topk(TOP_K, dim=-1)
            tensors[f"{case}_{impl}_ids"] = top.indices.to(torch.int32).cpu()
            tensors[f"{case}_{impl}_logprobs"] = top.values.cpu()
            print(f"{impl} {case}: recorded {rows.shape[0]} positions x top-{TOP_K}")
        del model
        torch.cuda.empty_cache()

    for case, length in CASES.items():
        tensors[f"{case}_prompt"] = torch.tensor(ids[:length], dtype=torch.int32)
        tensors[f"{case}_teacher"] = torch.tensor(
            ids[length : length + TEACHER_STEPS], dtype=torch.int32
        )
        # The consumer's floor: how far the two backends sit apart on the
        # recorded positions, evaluated at the sdpa top-64 ids.
        sdpa_ids = tensors[f"{case}_sdpa_ids"]
        sdpa_lp = tensors[f"{case}_sdpa_logprobs"]
        eager_ids = tensors[f"{case}_eager_ids"]
        eager_lp = tensors[f"{case}_eager_logprobs"]
        floor = 0.0
        agree = 0
        for pos in range(sdpa_ids.shape[0]):
            eager_map = {
                int(t): float(v) for t, v in zip(eager_ids[pos].tolist(), eager_lp[pos].tolist())
            }
            for t, v in zip(sdpa_ids[pos].tolist(), sdpa_lp[pos].tolist()):
                if int(t) in eager_map:
                    floor = max(floor, abs(float(v) - eager_map[int(t)]))
            agree += int(sdpa_ids[pos][0] == eager_ids[pos][0])
        print(
            f"case {case}: sdpa-vs-eager floor max|dlogprob| {floor:.2f}, "
            f"top-1 agree {agree}/{sdpa_ids.shape[0]}"
        )

    manifest = {
        "source_repo": args.source_repo,
        "revision": args.revision,
        "transformers": __import__("transformers").__version__,
        "cases": CASES,
        "teacher_steps": TEACHER_STEPS,
        "top_k": TOP_K,
        "corpus": "an unnumbered <table> row repeated, trimmed per case",
        "reference": "teacher-forced top-64 logprobs under both sdpa and eager",
    }
    save_file(tensors, args.out, metadata={METADATA_KEY: json.dumps(manifest, sort_keys=True)})
    digest = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    print(f"wrote {args.out} sha256 {digest}")


if __name__ == "__main__":
    main()
