"""Dump a chat-template golden from a local checkpoint.

    python tools/accuracy/dump_chat_template_golden.py qwen38 \
        models/Qwen3.8-27B test_data/qwen38-chat-golden.json \
        --source-repo Qwen/Qwen3.8-27B --revision <sha>

One dumper with a per-model case table: a template's knobs are model behaviour
the frontend has to render identically, so each knob gets a case and the case
records the options the Rust renderer must set. Provenance is required rather
than inferred, and a template error aborts the dump instead of being recorded
as an expected result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import transformers
from transformers import AutoTokenizer

MESSAGES = [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Explain gravity in one sentence."},
]
MULTI_TURN = [
    {"role": "user", "content": "Is 4111 prime?"},
    {"role": "assistant", "content": "Yes."},
    {"role": "user", "content": "And 4112?"},
]
UNICODE = [{"role": "user", "content": "翻译：🙂 と こんにちは"}]

# (name, messages, add_generation_prompt, extra template kwargs)
CHAT_CASES: dict[str, list[tuple[str, list[dict], bool, dict]]] = {
    "gemma4": [
        (
            "single_user_with_generation_prompt",
            [{"role": "user", "content": "What is 2 + 2?"}],
            True,
            {},
        ),
        (
            "single_user_no_generation_prompt",
            [{"role": "user", "content": "What is 2 + 2?"}],
            False,
            {},
        ),
        (
            "multi_turn",
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there."},
                {"role": "user", "content": "And now?"},
            ],
            True,
            {},
        ),
        (
            "system_then_user",
            [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Explain gravity."},
            ],
            True,
            {},
        ),
        ("unicode_content", UNICODE, True, {}),
    ],
    # Qwen3.8's template reads `reasoning_effort` (default `xhigh`, restricted
    # to xhigh|medium|low) and gates the reasoning instructions on
    # `enable_thinking`.
    "qwen38": [
        ("plain_user_turn", MESSAGES, True, {}),
        ("plain_user_no_generation_prompt", MESSAGES, False, {}),
        ("multi_turn", MULTI_TURN, True, {}),
        ("unicode_content", UNICODE, True, {}),
        ("reasoning_effort_xhigh", MESSAGES, True, {"reasoning_effort": "xhigh"}),
        ("reasoning_effort_medium", MESSAGES, True, {"reasoning_effort": "medium"}),
        ("reasoning_effort_low", MESSAGES, True, {"reasoning_effort": "low"}),
        ("thinking_disabled", MESSAGES, True, {"enable_thinking": False}),
        (
            "thinking_disabled_with_effort",
            MESSAGES,
            True,
            {"enable_thinking": False, "reasoning_effort": "low"},
        ),
    ],
}

REQUIRED_FILES = ("tokenizer.json", "tokenizer_config.json")

# Models whose committed golden records "template_kwargs" even when empty.
# Keeps regeneration byte-identical for the already-committed fixtures.
ALWAYS_RECORD_KWARGS = {"qwen38"}


def dump_file_hashes(model_dir: Path) -> dict:
    hashes = {}
    for name in REQUIRED_FILES:
        path = model_dir / name
        if not path.exists():
            raise SystemExit(f"required file missing from the checkpoint: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    template = model_dir / "chat_template.jinja"
    if template.exists():
        hashes["chat_template.jinja"] = hashlib.sha256(template.read_bytes()).hexdigest()
    else:
        # Newer checkpoints carry the template inside tokenizer_config.json;
        # its digest is already covered above, so the guard still binds the
        # exact template bytes.
        config = json.loads((model_dir / "tokenizer_config.json").read_text())
        if config.get("chat_template") is None:
            raise SystemExit(
                "no chat template found in either chat_template.jinja or "
                "tokenizer_config.json; refusing to dump an unbound golden"
            )
    return hashes


def dump_chat_templates(
    tokenizer,
    cases: list[tuple[str, list[dict], bool, dict]],
    always_record_kwargs: bool,
) -> list:
    dumped = []
    for name, messages, add_generation_prompt, kwargs in cases:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        case = {
            "name": name,
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
        }
        if kwargs or always_record_kwargs:
            case["template_kwargs"] = kwargs
        case["rendered"] = rendered
        dumped.append(case)
    return dumped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(CHAT_CASES), help="per-model case table")
    parser.add_argument("model_dir")
    parser.add_argument("out")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))

    golden = {
        "source_repo": args.source_repo,
        "revision": args.revision,
        "transformers_version": transformers.__version__,
        "file_sha256": dump_file_hashes(model_dir),
        "chat_templates": dump_chat_templates(
            tokenizer, CHAT_CASES[args.model], args.model in ALWAYS_RECORD_KWARGS
        ),
    }

    Path(args.out).write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.out}: {len(golden['chat_templates'])} chat cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
