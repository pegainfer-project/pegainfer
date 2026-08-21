"""Dump the Gemma 4 long-context logprob reference.

    python tools/accuracy/dump_gemma4_longctx_golden.py <model-dir> <out.safetensors> \
        --source-repo google/gemma-4-12B-it \
        --revision <same revision as the window fixture>

Two waypoints far past the sliding window (16384, 32768), each followed by 8
teacher-forced continuation tokens from the corpus. Every position records
the top-64 ids and logprobs under both sdpa and eager; eager materialises the
full attention matrix per layer, so a waypoint where it cannot fit is
recorded sdpa-only and named in the manifest — the consumer borrows the
widest dual-backend floor instead.

Refuses to write rather than record something unusable: non-finite scores at
any position abort the dump.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

TOP_K = 64
TEACHER_STEPS = 8
METADATA_KEY = "gemma4_longctx_golden"
CASES = {"w16384": 16384, "w32768": 32768}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("out")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def corpus_ids(tokenizer, min_len: int) -> list[int]:
    rows = "  <tr>\n    <td>item</td>\n  </tr>\n" * 7000
    ids = tokenizer(f"<table>\n{rows}", return_tensors="pt").input_ids[0].tolist()
    if len(ids) < min_len:
        raise SystemExit(f"corpus tokenizes to {len(ids)} < {min_len} tokens")
    return ids


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    ids = corpus_ids(tokenizer, max(CASES.values()) + TEACHER_STEPS)

    tensors: dict[str, torch.Tensor] = {}
    eager_skipped: list[str] = []
    for impl in ("sdpa", "eager"):
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, attn_implementation=impl
        ).to(args.device)
        model.eval()
        for case, length in CASES.items():
            seq = torch.tensor([ids[: length + TEACHER_STEPS]], device=args.device)

            def rows_of() -> torch.Tensor:
                # Keep only the recorded positions' logits: the full
                # 262k-vocab head over 32k positions is ~17 GiB in bf16 and
                # does not fit next to the 22 GiB tower.
                with torch.no_grad():
                    return model(seq, logits_to_keep=TEACHER_STEPS + 1).logits[0].float()

            try:
                rows = rows_of()
            except torch.cuda.OutOfMemoryError:
                if impl == "sdpa":
                    raise SystemExit(f"case {case}: sdpa itself cannot fit — no reference")
                eager_skipped.append(case)
                torch.cuda.empty_cache()
                print(f"eager {case}: does not fit, recorded sdpa-only")
                continue
            if not torch.equal(rows, rows_of()):
                raise SystemExit(f"case {case} ({impl}): forward is not reproducible")
            if not torch.isfinite(rows).all():
                raise SystemExit(f"case {case} ({impl}): non-finite logits")
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
        if case in eager_skipped:
            continue
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
        "eager_skipped": sorted(eager_skipped),
        "corpus": "an unnumbered <table> row repeated, trimmed per case",
        "reference": "teacher-forced top-64 logprobs under sdpa and, where it fits, eager",
    }
    save_file(tensors, args.out, metadata={METADATA_KEY: json.dumps(manifest, sort_keys=True)})


if __name__ == "__main__":
    main()
