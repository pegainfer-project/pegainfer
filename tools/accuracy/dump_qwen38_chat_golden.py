"""Dump a Qwen3.8 chat-template golden from a local checkpoint.

    python tools/accuracy/dump_qwen38_chat_golden.py \
        /data/models/Qwen3.8-27B test_data/qwen38-chat-golden.json \
        --source-repo Qwen/Qwen3.8-27B --revision <sha>

Provenance is required rather than inferred, and a template error aborts the
dump instead of being recorded as an expected result -- same discipline as
`dump_gemma4_tokenizer_golden.py`, whose consumer this mirrors.

Qwen3.8's template is where this line differs from Qwen3.5's: it reads
`reasoning_effort` (default `xhigh`, restricted to `xhigh|medium|low`), gates the
reasoning instructions on `enable_thinking`, and honours `preserve_thinking`.
Those knobs are model behaviour the frontend has to render identically, so each
one gets a case, and the case records the options the Rust renderer must set.
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
CHAT_CASES: list[tuple[str, list[dict], bool, dict]] = [
    ("plain_user_turn", MESSAGES, True, {}),
    ("plain_user_no_generation_prompt", MESSAGES, False, {}),
    ("multi_turn", MULTI_TURN, True, {}),
    ("unicode_content", UNICODE, True, {}),
    ("reasoning_effort_xhigh", MESSAGES, True, {"reasoning_effort": "xhigh"}),
    ("reasoning_effort_medium", MESSAGES, True, {"reasoning_effort": "medium"}),
    ("reasoning_effort_low", MESSAGES, True, {"reasoning_effort": "low"}),
    ("thinking_disabled", MESSAGES, True, {"enable_thinking": False}),
    ("thinking_disabled_with_effort", MESSAGES, True, {"enable_thinking": False, "reasoning_effort": "low"}),
    ("preserve_thinking", MULTI_TURN, True, {"preserve_thinking": True}),
]

REQUIRED_FILES = ("tokenizer.json", "tokenizer_config.json")
TEMPLATE_FILES = ("chat_template.jinja",)


def dump_file_hashes(model_dir: Path) -> dict:
    hashes = {}
    for name in REQUIRED_FILES:
        path = model_dir / name
        if not path.exists():
            raise SystemExit(f"required file missing from the checkpoint: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in TEMPLATE_FILES:
        path = model_dir / name
        if path.exists():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if "chat_template.jinja" not in hashes:
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


def dump_chat_templates(tokenizer) -> list:
    cases = []
    for name, messages, add_generation_prompt, kwargs in CHAT_CASES:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        cases.append(
            {
                "name": name,
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
                "template_kwargs": kwargs,
                "rendered": rendered,
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
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
        "chat_templates": dump_chat_templates(tokenizer),
    }

    Path(args.out).write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.out}: {len(golden['chat_templates'])} chat cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())
