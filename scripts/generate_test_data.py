#!/usr/bin/env python3
"""Generate greedy-decoded reference outputs from HuggingFace Transformers for e2e testing.

This script loads a model via HF Transformers and generates greedy (do_sample=False)
outputs for a set of test prompts. The results are saved as a single JSON file
that drives pegainfer's e2e tests.

Usage:
    python scripts/generate_test_data.py --model models/Qwen3-4B --name Qwen3-4B
    python scripts/generate_test_data.py --model models/Qwen3-4B --name Qwen3-4B --device cpu
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Test cases: must match the prompts and max_tokens used in e2e.rs
TEST_CASES = [
    {"name": "tell_story", "prompt": "Tell me a story", "max_new_tokens": 50},
    {"name": "my_name", "prompt": "My name is", "max_new_tokens": 50},
    {"name": "math", "prompt": "What is 2 + 2?", "max_new_tokens": 30},
    {"name": "chinese_weather", "prompt": "今天天气真好", "max_new_tokens": 50},
    {"name": "chinese_capital", "prompt": "请介绍一下中国的首都", "max_new_tokens": 50},
    {
        "name": "python_code",
        "prompt": "Write a Python function to reverse a string",
        "max_new_tokens": 50,
    },
]


def generate_test_data(
    model_path: str, model_name: str, output_dir: str, device: str
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    print(
        f"Model loaded: {model.config.model_type}, vocab_size={model.config.vocab_size}"
    )

    cases = []
    for tc in TEST_CASES:
        name = tc["name"]
        prompt = tc["prompt"]
        max_new_tokens = tc["max_new_tokens"]

        print(f'\n--- {name}: "{prompt}" (max_new_tokens={max_new_tokens}) ---')

        input_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        ).to(device)
        print(f"  prompt_tokens ({input_ids.shape[1]}): {input_ids[0].tolist()}")

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # greedy
                temperature=None,
                top_p=None,
            )

        # Extract only the generated tokens (exclude prompt)
        generated_ids = output_ids[0, input_ids.shape[1] :].tolist()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(
            f"  output_tokens ({len(generated_ids)}): {generated_ids[:20]}{'...' if len(generated_ids) > 20 else ''}"
        )
        print(
            f'  output: "{generated_text[:80]}{"..." if len(generated_text) > 80 else ""}"'
        )

        cases.append(
            {
                "name": name,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "output": generated_text,
                "output_tokens": generated_ids,
            }
        )

    result = {
        "model_name": model_name,
        "torch_version": torch.__version__,
        "device": device,
        "cases": cases,
    }

    output_file = output_path / f"{model_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {output_file} ({len(cases)} cases)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/Qwen3-4B",
        help="Path to HF model directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="Qwen3-4B",
        help="Model name (used in output filename)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_data",
        help="Output directory (default: test_data)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device (default: cpu)",
    )
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Error: Model path {args.model} does not exist", file=sys.stderr)
        sys.exit(1)

    generate_test_data(args.model, args.name, args.output, args.device)


if __name__ == "__main__":
    main()
