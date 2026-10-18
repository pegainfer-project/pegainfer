#!/usr/bin/env python3
"""Compare HF, host-staged, and NCCL DeepSeek-V2-Lite EP=2 greedy outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def sha256_u32_le(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(int(value).to_bytes(4, byteorder="little", signed=False))
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json_or_stdout(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                obj, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
        raise


def list_of_ints(raw: Any) -> list[int]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"expected a list of token ids, got {type(raw)!r}")
    return [int(value) for value in raw]


def list_of_token_rows(raw: Any) -> list[list[int]] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError(f"expected generated_token_ids_by_row to be a list, got {type(raw)!r}")
    rows: list[list[int]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, list):
            raise ValueError(f"generated row {index} must be a list")
        rows.append([int(value) for value in item])
    return rows


@dataclass
class Output:
    name: str
    case_id: str
    row_index: int
    model_path: str | None
    backend: str | None
    prompt: str | None
    prompt_token_ids: list[int]
    batch_size: int | None
    output_len: int | None
    generation_mode: str | None
    generated_token_ids: list[int]
    generated_text: str
    reported_token_sha256: str | None
    reported_text_sha256: str | None
    token_sha256: str
    text_sha256: str
    raw: dict[str, Any]


def normalize_output(
    name: str,
    case_id: str,
    payload: dict[str, Any],
    *,
    row_index: int,
    token_ids: list[int],
    generated_text: str,
    reported_token_sha256: str | None,
    reported_text_sha256: str | None,
    fallback_model_path: str | None = None,
    fallback_backend: str | None = None,
    fallback_generation_mode: str | None = None,
) -> Output:
    prompt_token_ids = payload.get("prompt_token_ids") or []
    return Output(
        name=name,
        case_id=case_id,
        row_index=row_index,
        model_path=payload.get("model_path") or fallback_model_path,
        backend=payload.get("ep_backend") or payload.get("backend") or fallback_backend,
        prompt=payload.get("prompt"),
        prompt_token_ids=[int(value) for value in prompt_token_ids],
        batch_size=payload.get("batch_size"),
        output_len=payload.get("output_len") or payload.get("max_new_tokens"),
        generation_mode=payload.get("generation_mode") or fallback_generation_mode,
        generated_token_ids=token_ids,
        generated_text=generated_text,
        reported_token_sha256=reported_token_sha256,
        reported_text_sha256=reported_text_sha256,
        token_sha256=sha256_u32_le(token_ids),
        text_sha256=sha256_text(generated_text),
        raw=payload,
    )


def normalize_singleton(
    name: str,
    payload: dict[str, Any],
    *,
    case_id: str = "default",
    fallback_model_path: str | None = None,
    fallback_backend: str | None = None,
    fallback_generation_mode: str | None = None,
) -> list[Output]:
    rows = list_of_token_rows(payload.get("generated_token_ids_by_row"))
    texts_by_row = payload.get("generated_text_by_row")
    token_hashes_by_row = payload.get("token_sha256_by_row")
    text_hashes_by_row = payload.get("text_sha256_by_row")

    if rows is None:
        token_ids = payload.get("generated_token_ids")
        if token_ids is None:
            token_ids = payload.get("output_tokens")
        if token_ids is None:
            token_ids = payload.get("tokens")
        generated_text = payload.get("generated_text")
        if generated_text is None:
            generated_text = payload.get("output_text")
        if generated_text is None:
            generated_text = payload.get("output")
        return [
            normalize_output(
                name,
                case_id,
                payload,
                row_index=0,
                token_ids=list_of_ints(token_ids),
                generated_text=str(generated_text or ""),
                reported_token_sha256=payload.get("token_sha256")
                or payload.get("output_token_sha256"),
                reported_text_sha256=payload.get("text_sha256")
                or payload.get("output_text_sha256"),
                fallback_model_path=fallback_model_path,
                fallback_backend=fallback_backend,
                fallback_generation_mode=fallback_generation_mode,
            )
        ]

    outputs = []
    for row_index, token_ids in enumerate(rows):
        if isinstance(texts_by_row, list) and row_index < len(texts_by_row):
            generated_text = str(texts_by_row[row_index])
        else:
            generated_text = str(payload.get("generated_text") or payload.get("output_text") or "")
        reported_token_hash = (
            token_hashes_by_row[row_index]
            if isinstance(token_hashes_by_row, list) and row_index < len(token_hashes_by_row)
            else None
        )
        reported_text_hash = (
            text_hashes_by_row[row_index]
            if isinstance(text_hashes_by_row, list) and row_index < len(text_hashes_by_row)
            else None
        )
        outputs.append(
            normalize_output(
                name,
                case_id,
                payload,
                row_index=row_index,
                token_ids=token_ids,
                generated_text=generated_text,
                reported_token_sha256=reported_token_hash,
                reported_text_sha256=reported_text_hash,
                fallback_model_path=fallback_model_path,
                fallback_backend=fallback_backend,
                fallback_generation_mode=fallback_generation_mode,
            )
        )
    return outputs


def normalize_cases(name: str, payload: dict[str, Any]) -> dict[str, list[Output]]:
    if payload.get("schema") == 2 and isinstance(payload.get("cases"), list):
        cases: dict[str, list[Output]] = {}
        fallback_model_path = payload.get("model_path")
        fallback_backend = payload.get("ep_backend") or payload.get("backend")
        fallback_generation_mode = payload.get("generation_mode")
        for index, item in enumerate(payload["cases"]):
            if not isinstance(item, dict):
                raise ValueError(f"{name} case {index} must be an object")
            case_id = str(item.get("id") or item.get("case_id") or f"case_{index:03d}")
            if case_id in cases:
                raise ValueError(f"{name} contains duplicate case id {case_id!r}")
            cases[case_id] = normalize_singleton(
                name,
                item,
                case_id=case_id,
                fallback_model_path=fallback_model_path,
                fallback_backend=fallback_backend,
                fallback_generation_mode=fallback_generation_mode,
            )
        return cases
    return {"default": normalize_singleton(name, payload)}


def first_token_diff(left: Output, right: Output) -> dict[str, Any] | None:
    limit = min(len(left.generated_token_ids), len(right.generated_token_ids))
    for index in range(limit):
        left_token = left.generated_token_ids[index]
        right_token = right.generated_token_ids[index]
        if left_token != right_token:
            return {
                "row_index": right.row_index,
                "index": index,
                left.name: left_token,
                right.name: right_token,
                "reason": "token_mismatch",
            }
    if len(left.generated_token_ids) != len(right.generated_token_ids):
        return {
            "row_index": right.row_index,
            "index": limit,
            left.name: left.generated_token_ids[limit]
            if len(left.generated_token_ids) > limit
            else None,
            right.name: right.generated_token_ids[limit]
            if len(right.generated_token_ids) > limit
            else None,
            "reason": "length_mismatch",
        }
    return None


def pair_summary(
    left_rows: list[Output],
    right_rows: list[Output],
    *,
    broadcast_single: bool,
) -> dict[str, Any]:
    if not left_rows or not right_rows:
        return {
            "token_exact": False,
            "text_exact": False,
            "first_different_token": {"reason": "missing_outputs"},
        }
    if len(left_rows) == len(right_rows):
        pairs = list(zip(left_rows, right_rows, strict=True))
    elif broadcast_single and len(left_rows) == 1:
        # HF case-set dumps one expected row; PegaInfer emits every same-prompt
        # batch row, so only the left/HF side is allowed to broadcast.
        pairs = [(left_rows[0], right) for right in right_rows]
    else:
        return {
            "token_exact": False,
            "text_exact": False,
            "first_different_token": {
                "reason": "row_count_mismatch",
                left_rows[0].name: len(left_rows),
                right_rows[0].name: len(right_rows),
            },
        }

    token_exact = True
    text_exact = True
    first_diff = None
    for left, right in pairs:
        row_token_exact = left.generated_token_ids == right.generated_token_ids
        row_text_exact = left.generated_text == right.generated_text
        token_exact = token_exact and row_token_exact
        text_exact = text_exact and row_text_exact
        if first_diff is None and not row_token_exact:
            first_diff = first_token_diff(left, right)
        if first_diff is None and row_token_exact and not row_text_exact:
            first_diff = {"row_index": right.row_index, "reason": "text_mismatch"}

    return {
        "token_exact": token_exact,
        "text_exact": text_exact,
        "first_different_token": first_diff,
        "row_comparisons": len(pairs),
    }


def classify(pairs: dict[str, dict[str, Any]]) -> str:
    host_nccl_exact = pairs["host_staged_vs_nccl"]["token_exact"] and pairs[
        "host_staged_vs_nccl"
    ]["text_exact"]
    hf_host_exact = pairs["hf_vs_host_staged"]["token_exact"] and pairs[
        "hf_vs_host_staged"
    ]["text_exact"]
    hf_nccl_exact = pairs["hf_vs_nccl"]["token_exact"] and pairs["hf_vs_nccl"][
        "text_exact"
    ]
    if host_nccl_exact and hf_host_exact and hf_nccl_exact:
        return "all_token_text_exact"
    if not host_nccl_exact:
        return "nccl_transport_regression"
    return "pegainfer_baseline_accuracy_gap"


def overall_classification(case_results: list[dict[str, Any]]) -> str:
    labels = {case["classification"] for case in case_results}
    if labels == {"all_token_text_exact"}:
        return "all_token_text_exact"
    if "nccl_transport_regression" in labels:
        return "nccl_transport_regression"
    return "pegainfer_baseline_accuracy_gap"


def short(text: str, width: int = 72) -> str:
    one_line = text.replace("\n", "\\n")
    if len(one_line) <= width:
        return one_line
    return one_line[: width - 3] + "..."


def table(case_results: list[dict[str, Any]]) -> str:
    rows = [
        "| Case | Source | Row | Backend | Tokens | Token SHA256 | Text SHA256 | Text |",
        "| --- | --- | ---: | --- | ---: | --- | --- | --- |",
    ]
    for case in case_results:
        for output in case["_outputs_for_table"]:
            rows.append(
                "| {case_id} | {name} | {row} | {backend} | {tokens} | `{token_hash}` | `{text_hash}` | `{text}` |".format(
                    case_id=case["id"],
                    name=output.name,
                    row=output.row_index,
                    backend=output.backend or "-",
                    tokens=len(output.generated_token_ids),
                    token_hash=output.token_sha256,
                    text_hash=output.text_sha256,
                    text=short(output.generated_text),
                )
            )
    return "\n".join(rows)


def hash_warnings(outputs: list[Output]) -> list[str]:
    warnings = []
    for output in outputs:
        label = f"{output.name}:{output.case_id}:row{output.row_index}"
        if (
            output.reported_token_sha256
            and output.reported_token_sha256 != output.token_sha256
        ):
            warnings.append(
                f"{label}: reported token hash {output.reported_token_sha256} "
                f"does not match recomputed {output.token_sha256}"
            )
        if output.reported_text_sha256 and output.reported_text_sha256 != output.text_sha256:
            warnings.append(
                f"{label}: reported text hash {output.reported_text_sha256} "
                f"does not match recomputed {output.text_sha256}"
            )
    return warnings


def context_warnings(case_id: str, hf: list[Output], host: list[Output], nccl: list[Output]) -> list[str]:
    warnings = []
    outputs = hf + host + nccl

    prompts = {output.prompt for output in outputs if output.prompt is not None}
    if len(prompts) > 1:
        warnings.append(f"{case_id}: prompt mismatch across outputs: {sorted(prompts)!r}")

    prompt_token_ids = {
        tuple(output.prompt_token_ids)
        for output in outputs
        if output.prompt_token_ids
    }
    if len(prompt_token_ids) > 1:
        warnings.append(f"{case_id}: prompt_token_ids mismatch across outputs")

    model_paths = {output.model_path for output in outputs if output.model_path is not None}
    if len(model_paths) > 1:
        warnings.append(
            f"{case_id}: model_path labels differ; verify all outputs use the same snapshot: "
            f"{sorted(model_paths)!r}"
        )

    output_lens = {output.output_len for output in outputs if output.output_len is not None}
    if len(output_lens) > 1:
        warnings.append(f"{case_id}: output length labels differ: {sorted(output_lens)!r}")

    batch_sizes = {output.batch_size for output in outputs if output.batch_size is not None}
    if len(batch_sizes) > 1:
        warnings.append(f"{case_id}: batch_size labels differ: {sorted(batch_sizes)!r}")
    expected_rows = max(batch_sizes) if batch_sizes else None
    if expected_rows is not None:
        if len(host) != expected_rows:
            warnings.append(f"{case_id}: host-staged row count {len(host)} != batch_size {expected_rows}")
        if len(nccl) != expected_rows:
            warnings.append(f"{case_id}: NCCL row count {len(nccl)} != batch_size {expected_rows}")

    hf_generation_modes = {
        output.generation_mode
        for output in hf
        if output.generation_mode is not None
    }
    allowed_hf_modes = {"incremental_past_key_values", "transformers_generate_use_cache"}
    for mode in hf_generation_modes:
        if mode not in allowed_hf_modes:
            warnings.append(
                f"{case_id}: HF output generation_mode is not recognized: {mode}"
            )
    if any(output.backend not in (None, "host-staged") for output in host):
        warnings.append(f"{case_id}: host-staged file reports a non-host backend")
    if any(output.backend not in (None, "nccl") for output in nccl):
        warnings.append(f"{case_id}: NCCL file reports a non-NCCL backend")

    return warnings


def output_summary(output: Output) -> dict[str, Any]:
    return {
        "row_index": output.row_index,
        "model_path": output.model_path,
        "backend": output.backend,
        "prompt": output.prompt,
        "prompt_token_ids": output.prompt_token_ids,
        "batch_size": output.batch_size,
        "output_len": output.output_len,
        "generation_mode": output.generation_mode,
        "generated_token_ids": output.generated_token_ids,
        "generated_text": output.generated_text,
        "token_sha256": output.token_sha256,
        "text_sha256": output.text_sha256,
        "reported_token_sha256": output.reported_token_sha256,
        "reported_text_sha256": output.reported_text_sha256,
    }


def legacy_output_summary(output: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_path": output["model_path"],
        "backend": output["backend"],
        "prompt": output["prompt"],
        "prompt_token_ids": output["prompt_token_ids"],
        "generated_token_ids": output["generated_token_ids"],
        "generated_text": output["generated_text"],
        "token_sha256": output["token_sha256"],
        "text_sha256": output["text_sha256"],
        "reported_token_sha256": output["reported_token_sha256"],
        "reported_text_sha256": output["reported_text_sha256"],
    }


def legacy_pair_summary(pair: dict[str, Any]) -> dict[str, Any]:
    first_diff = pair["first_different_token"]
    if isinstance(first_diff, dict):
        first_diff = dict(first_diff)
        first_diff.pop("row_index", None)
    return {
        "token_exact": pair["token_exact"],
        "text_exact": pair["text_exact"],
        "first_different_token": first_diff,
    }


def compare_cases(
    hf_cases: dict[str, list[Output]],
    host_cases: dict[str, list[Output]],
    nccl_cases: dict[str, list[Output]],
) -> tuple[list[dict[str, Any]], list[str]]:
    case_ids = sorted(set(hf_cases) | set(host_cases) | set(nccl_cases))
    case_results: list[dict[str, Any]] = []
    warnings: list[str] = []
    for case_id in case_ids:
        hf = hf_cases.get(case_id, [])
        host = host_cases.get(case_id, [])
        nccl = nccl_cases.get(case_id, [])
        pairs = {
            "hf_vs_host_staged": pair_summary(hf, host, broadcast_single=True),
            "hf_vs_nccl": pair_summary(hf, nccl, broadcast_single=True),
            "host_staged_vs_nccl": pair_summary(host, nccl, broadcast_single=False),
        }
        classification = classify(pairs)
        all_outputs = hf + host + nccl
        warnings.extend(hash_warnings(all_outputs))
        warnings.extend(context_warnings(case_id, hf, host, nccl))
        case_results.append(
            {
                "id": case_id,
                "classification": classification,
                "outputs": {
                    "hf": [output_summary(output) for output in hf],
                    "host_staged": [output_summary(output) for output in host],
                    "nccl": [output_summary(output) for output in nccl],
                },
                "pairs": pairs,
                "_outputs_for_table": all_outputs,
            }
        )
    return case_results, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf", required=True, help="HF JSON output")
    parser.add_argument("--host-staged", required=True, help="host-staged pegainfer JSON output")
    parser.add_argument("--nccl", required=True, help="NCCL pegainfer JSON output")
    parser.add_argument("--out", help="Optional path for structured comparison JSON")
    parser.add_argument(
        "--require-all-exact",
        action="store_true",
        help="Exit nonzero unless every case is token/text exact across HF, host-staged, and NCCL",
    )
    args = parser.parse_args()

    hf_cases = normalize_cases("hf", load_json_or_stdout(Path(args.hf)))
    host_cases = normalize_cases("host_staged", load_json_or_stdout(Path(args.host_staged)))
    nccl_cases = normalize_cases("nccl", load_json_or_stdout(Path(args.nccl)))
    case_results, warnings = compare_cases(hf_cases, host_cases, nccl_cases)
    classification = overall_classification(case_results)

    print(table(case_results))
    print()
    print(f"Classification: {classification}")
    print(
        json.dumps(
            {
                "case_classifications": {
                    case["id"]: case["classification"] for case in case_results
                },
                "warnings": warnings,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    result_cases = []
    for case in case_results:
        public_case = dict(case)
        public_case.pop("_outputs_for_table", None)
        result_cases.append(public_case)
    if len(result_cases) == 1 and result_cases[0]["id"] == "default":
        default_case = result_cases[0]
        result = {
            "classification": classification,
            "outputs": {
                name: legacy_output_summary(outputs[0]) if outputs else {}
                for name, outputs in default_case["outputs"].items()
            },
            "pairs": {
                name: legacy_pair_summary(pair)
                for name, pair in default_case["pairs"].items()
            },
            "warnings": warnings,
        }
    else:
        result = {
            "schema": 2,
            "classification": classification,
            "case_count": len(result_cases),
            "cases": result_cases,
            "warnings": warnings,
        }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {out_path}")

    if args.require_all_exact and (
        classification != "all_token_text_exact" or warnings
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
