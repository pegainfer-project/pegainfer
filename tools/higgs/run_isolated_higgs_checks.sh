#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
tmp_root="${TMPDIR:-/tmp}/pegainfer-higgs-isolated"

rm -rf "$tmp_root"
mkdir -p "$tmp_root"
cp -R "$repo_root/pegainfer-higgs-audio" "$tmp_root/pegainfer-higgs-audio"
python3 - <<'PY' "$tmp_root/pegainfer-higgs-audio/Cargo.toml"
import re
import sys
from pathlib import Path
path = Path(sys.argv[1])
text = path.read_text()
text = re.sub(r'^pegainfer-core = .*\n', '', text, flags=re.MULTILINE)
text = re.sub(r'^pegainfer-qwen3 = .*\n', '', text, flags=re.MULTILINE)
text = re.sub(r'runtime-qwen3 = .*\n', 'runtime-qwen3 = []\n', text)
text = re.sub(r'\n\[\[bin\]\]\nname = "higgs_dump_one_step_actual"\npath = "src/bin/higgs_dump_one_step_actual.rs"\nrequired-features = \["runtime-qwen3"\]\n', '\n', text)
text = re.sub(r'\n\[\[bin\]\]\nname = "higgs_prefill_prompt_session_smoke"\npath = "src/bin/higgs_prefill_prompt_session_smoke.rs"\nrequired-features = \["runtime-qwen3"\]\n', '\n', text)
text = re.sub(r'\n\[\[bin\]\]\nname = "higgs_dump_prefill_layer_hidden"\npath = "src/bin/higgs_dump_prefill_layer_hidden.rs"\nrequired-features = \["runtime-qwen3"\]\n', '\n', text)
text = re.sub(r'\n\[\[bin\]\]\nname = "higgs_dump_layer0_stages"\npath = "src/bin/higgs_dump_layer0_stages.rs"\nrequired-features = \["runtime-qwen3"\]\n', '\n', text)
path.write_text(text)
PY
rm -f "$tmp_root/pegainfer-higgs-audio/src/bin/higgs_dump_one_step_actual.rs"
rm -f "$tmp_root/pegainfer-higgs-audio/src/bin/higgs_prefill_prompt_session_smoke.rs"
rm -f "$tmp_root/pegainfer-higgs-audio/src/bin/higgs_dump_prefill_layer_hidden.rs"
rm -f "$tmp_root/pegainfer-higgs-audio/src/bin/higgs_dump_layer0_stages.rs"
mkdir -p "$tmp_root/test_data"
cp "$repo_root/test_data/higgs-one-step-audio-logits.safetensors" "$tmp_root/test_data/"

python3 -m py_compile \
  "$repo_root/tools/accuracy/compare_higgs_trace_dump.py" \
  "$repo_root/tools/higgs/check_higgs_gate_summary.py" \
  "$repo_root/tools/higgs/check_higgs_sglang_omni_imports.py" \
  "$repo_root/tools/higgs/check_higgs_sglang_omni_runtime_readiness_summary.py" \
  "$repo_root/tools/higgs/check_higgs_sglang_omni_source_gate_summary.py"

summary_tmp="$tmp_root/summary-checks"
mkdir -p "$summary_tmp/auto-view"
printf '%s' 'x' >"$summary_tmp/golden.safetensors"
printf '%s' 'x' >"$summary_tmp/actual.safetensors"
printf '%s' 'x' >"$summary_tmp/session-actual.safetensors"
printf '%s\n' 'higgs one-step strict comparison: ok' >"$summary_tmp/source-compare.txt"
printf '%s\n' \
  'direct_higgs_imports=ok' \
  'full_higgs_model_import=missing_sglang' \
  >"$summary_tmp/readiness.txt"
printf '%s' 'x' >"$summary_tmp/session-smoke.txt"
printf '%s' 'x' >"$summary_tmp/semantic-compare.txt"
printf '%s' 'x' >"$summary_tmp/session-semantic-compare.txt"
printf '%s' 'x' >"$summary_tmp/auto-view/config.json"
printf '%s' 'x' >"$summary_tmp/auto-view/generation_config.json"
printf '%s' 'x' >"$summary_tmp/auto-view/higgs-qwen3-tensor-aliases.json"

printf '%s\n' \
  'status=ok' \
  "repo=$repo_root" \
  'commit=isolated' \
  'label=isolated' \
  'model_dir=/models/higgs' \
  "golden=$summary_tmp/golden.safetensors" \
  'sm=89' \
  'nvcc_jobs=8' \
  "actual=$summary_tmp/actual.safetensors" \
  "session_actual=$summary_tmp/session-actual.safetensors" \
  "compare_log=$summary_tmp/semantic-compare.txt" \
  "session_smoke_log=$summary_tmp/session-smoke.txt" \
  "session_compare_log=$summary_tmp/session-semantic-compare.txt" \
  "auto_view=$summary_tmp/auto-view" \
  'semantic_comparison=ok' \
  'session_semantic_comparison=ok' \
  'duplicate_request_id_guard=ok' \
  'artifacts_nonempty=ok' \
  >"$summary_tmp/cuda-gate.txt"
python3 "$repo_root/tools/higgs/check_higgs_gate_summary.py" "$summary_tmp/cuda-gate.txt" \
  --expected-label isolated \
  --expected-sm 89 \
  --expected-nvcc-jobs 8 \
  --expected-model-dir /models/higgs \
  --expected-golden "$summary_tmp/golden.safetensors" \
  --check-files

printf '%s\n' \
  'status=ok' \
  "repo=$repo_root" \
  'commit=isolated' \
  'label=isolated' \
  'model_dir=/models/higgs' \
  'sglang_omni_src=/src/sglang-omni' \
  'sglang_omni_commit=abc1234' \
  "golden=$summary_tmp/golden.safetensors" \
  "reference=$summary_tmp/actual.safetensors" \
  "compare_log=$summary_tmp/source-compare.txt" \
  "readiness_log=$summary_tmp/readiness.txt" \
  'sglang_omni_direct_imports=ok' \
  'sglang_omni_full_model_import=missing_sglang' \
  'source_reference_strict_comparison=ok' \
  'artifacts_nonempty=ok' \
  >"$summary_tmp/source-gate.txt"
python3 "$repo_root/tools/higgs/check_higgs_sglang_omni_source_gate_summary.py" "$summary_tmp/source-gate.txt" \
  --expected-label isolated \
  --expected-model-dir /models/higgs \
  --expected-sglang-omni-src /src/sglang-omni \
  --expected-sglang-omni-commit abc1234 \
  --expected-golden "$summary_tmp/golden.safetensors" \
  --check-files

printf '%s\n' \
  'python.version=3.12 isolated' \
  'pyproject.dependency.torch=torch==2.11.0' \
  'pyproject.dependency.sglang=sglang==0.5.16' \
  'package.torch.version=2.11.0' \
  'package.torch.cuda=13.0' \
  'package.torch.has_cuda_begin_allocate_current_thread_to_pool=ok' \
  'package.sglang.version=0.5.16' \
  'package.transformers.version=5.12.1' \
  'direct_higgs_imports=ok' \
  'full_higgs_model_import=ok' \
  >"$summary_tmp/runtime-readiness.txt"
printf '%s\n' \
  'status=ok' \
  "repo=$repo_root" \
  'commit=isolated' \
  'label=isolated' \
  'python=python3' \
  'python_version=3.12 isolated' \
  'sglang_omni_src=/src/sglang-omni' \
  'sglang_omni_commit=abc1234' \
  "readiness_log=$summary_tmp/runtime-readiness.txt" \
  'pyproject_torch=torch==2.11.0' \
  'pyproject_sglang=sglang==0.5.16' \
  'torch_version=2.11.0' \
  'torch_cuda=13.0' \
  'torch_has_cuda_pool_api=ok' \
  'sglang_version=0.5.16' \
  'transformers_version=5.12.1' \
  'sglang_omni_direct_imports=ok' \
  'sglang_omni_full_model_import=ok' \
  'runtime_ready=ok' \
  'artifacts_nonempty=ok' \
  >"$summary_tmp/runtime-readiness-gate.txt"
python3 "$repo_root/tools/higgs/check_higgs_sglang_omni_runtime_readiness_summary.py" \
  "$summary_tmp/runtime-readiness-gate.txt" \
  --expected-status ok \
  --expected-label isolated \
  --expected-python python3 \
  --expected-sglang-omni-src /src/sglang-omni \
  --expected-sglang-omni-commit abc1234 \
  --check-files

printf '%s\n' \
  'python.version=3.12 isolated' \
  'pyproject.dependency.torch=torch==2.11.0' \
  'pyproject.dependency.sglang=sglang==0.5.16' \
  'package.torch.version=2.6.0' \
  'package.torch.cuda=12.8' \
  'package.torch.has_cuda_begin_allocate_current_thread_to_pool=fail' \
  'package.sglang.version=0.5.16' \
  'package.transformers.version=5.12.1' \
  'direct_higgs_imports=ok' \
  'full_higgs_model_import=ImportError:torch stack mismatch' \
  >"$summary_tmp/runtime-readiness-fail.txt"
printf '%s\n' \
  'status=fail' \
  "repo=$repo_root" \
  'commit=isolated' \
  'label=isolated-fail' \
  'python=python3' \
  'python_version=3.12 isolated' \
  'sglang_omni_src=/src/sglang-omni' \
  'sglang_omni_commit=abc1234' \
  "readiness_log=$summary_tmp/runtime-readiness-fail.txt" \
  'pyproject_torch=torch==2.11.0' \
  'pyproject_sglang=sglang==0.5.16' \
  'torch_version=2.6.0' \
  'torch_cuda=12.8' \
  'torch_has_cuda_pool_api=fail' \
  'sglang_version=0.5.16' \
  'transformers_version=5.12.1' \
  'sglang_omni_direct_imports=ok' \
  'sglang_omni_full_model_import=ImportError:torch stack mismatch' \
  'runtime_ready=fail' \
  'artifacts_nonempty=ok' \
  >"$summary_tmp/runtime-readiness-fail-gate.txt"
python3 "$repo_root/tools/higgs/check_higgs_sglang_omni_runtime_readiness_summary.py" \
  "$summary_tmp/runtime-readiness-fail-gate.txt" \
  --expected-status fail \
  --expected-label isolated-fail \
  --expected-python python3 \
  --expected-sglang-omni-src /src/sglang-omni \
  --expected-sglang-omni-commit abc1234 \
  --check-files

cat >"$tmp_root/Cargo.toml" <<'TOML'
[workspace]
resolver = "3"
members = ["pegainfer-higgs-audio"]

[workspace.package]
version = "1.2.0"
edition = "2024"
license = "Apache-2.0"

[workspace.dependencies]
anyhow = "1.0"
clap = { version = "4.6.1", features = ["derive"] }
half = { version = "2.7", features = ["num-traits"] }
memmap2 = "0.9"
safetensors = "0.7"
serde_json = "1.0.149"
sha2 = "0.11"
tempfile = "3"

[workspace.lints.clippy]
pedantic = { level = "warn", priority = -2 }
cast_lossless = "allow"
cast_possible_truncation = "allow"
cast_possible_wrap = "allow"
cast_precision_loss = "allow"
cast_sign_loss = "allow"
collapsible_else_if = "allow"
collapsible_if = "allow"
doc_markdown = "allow"
implicit_hasher = "allow"
missing_errors_doc = "allow"
missing_panics_doc = "allow"
module_name_repetitions = "allow"
must_use_candidate = "allow"
similar_names = "allow"
too_many_arguments = "allow"
too_many_lines = "allow"
uninlined_format_args = "allow"
upper_case_acronyms = "allow"
redundant_clone = "warn"
unused_peekable = "warn"
dbg_macro = "warn"
exit = "warn"
get_unwrap = "warn"
print_stdout = "allow"
print_stderr = "allow"
rc_buffer = "warn"
rc_mutex = "warn"
rest_pat_in_fully_bound_structs = "warn"

[workspace.lints.rust]
unsafe_op_in_unsafe_fn = "warn"
unreachable_pub = "warn"
TOML

cd "$tmp_root"
cargo fmt --all --check
cargo test -p pegainfer-higgs-audio
cargo run -p pegainfer-higgs-audio --bin higgs_compare_one_step -- \
  --golden test_data/higgs-one-step-audio-logits.safetensors \
  --actual test_data/higgs-one-step-audio-logits.safetensors
