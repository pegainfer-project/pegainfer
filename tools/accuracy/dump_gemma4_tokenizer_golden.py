"""Dump a Gemma 4 tokenizer/chat-template golden from a local checkpoint.

    python tools/accuracy/dump_gemma4_tokenizer_golden.py <model-dir> <out.json> \
        --source-repo google/gemma-4-12B-it --revision <sha>

Provenance is required rather than inferred: a checkpoint directory carries no
reliable record of where it came from, and guessing from the path would write a
wrong repository name into the golden. Every chat case is expected to render; a
template error aborts the dump rather than being written into the golden.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import transformers
from transformers import AutoTokenizer


CHAT_CASES = [
    (
        "single_user_with_generation_prompt",
        [{"role": "user", "content": "What is 2 + 2?"}],
        True,
    ),
    (
        "single_user_no_generation_prompt",
        [{"role": "user", "content": "What is 2 + 2?"}],
        False,
    ),
    (
        "multi_turn",
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there."},
            {"role": "user", "content": "And now?"},
        ],
        True,
    ),
    (
        "system_then_user",
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": "Explain gravity."},
        ],
        True,
    ),
    (
        "unicode_content",
        [{"role": "user", "content": "翻译：🙂 と こんにちは"}],
        True,
    ),
]

HASHED_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")


def dump_file_hashes(model_dir: Path) -> dict:
    hashes = {}
    for name in HASHED_FILES:
        path = model_dir / name
        if not path.exists():
            raise SystemExit(f"required file missing from the checkpoint: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def dump_chat_templates(tokenizer) -> list:
    cases = []
    for name, messages, add_generation_prompt in CHAT_CASES:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        cases.append(
            {
                "name": name,
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
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
