#!/usr/bin/env python3
"""Pinned original vllm-bench: one fixed batch per subprocess, logs persisted live."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import asyncio
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class VllmBenchError(RuntimeError):
    """Raised when vllm-bench cannot be used as a calibration driver."""


@dataclass(frozen=True)
class VllmBenchIdentity:
    """Stable executable identity recorded with raw campaign artifacts."""

    path: str
    sha256: str
    version: str
    help_sha256: str



def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise VllmBenchError(f"failed to read vllm-bench binary {path}: {exc}") from exc


def inspect_binary(binary: Path) -> VllmBenchIdentity:
    """Read version/help once and return an immutable executable identity."""

    binary = binary.expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise VllmBenchError(f"vllm-bench is not an executable file: {binary}")
    try:
        version_run = subprocess.run(
            [str(binary), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        help_run = subprocess.run(
            [str(binary), "--help"],
            check=False,
            capture_output=True,
            text=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VllmBenchError(f"failed to inspect vllm-bench {binary}: {exc}") from exc
    if version_run.returncode != 0:
        raise VllmBenchError(
            f"vllm-bench --version failed with {version_run.returncode}: "
            f"{version_run.stderr.strip()[:300]}"
        )
    if help_run.returncode != 0:
        raise VllmBenchError(
            f"vllm-bench --help failed with {help_run.returncode}: "
            f"{help_run.stderr.decode(errors='replace').strip()[:300]}"
        )
    version = (version_run.stdout or version_run.stderr).strip().splitlines()
    rendered_version = version[0].strip() if version else "unknown"
    return VllmBenchIdentity(
        path=str(binary),
        sha256=_sha256_file(binary),
        version=rendered_version,
        help_sha256=_sha256_bytes(help_run.stdout),
    )


def render_command(plan: dict, batch: dict, *, binary: Path, endpoint: str, model: str,
                   result_dir: Path, request_id_prefix: str, tokenizer: str) -> list[str]:
    """Render an explicitly registered batch; no cell/repeat policy lives here."""
    from request_plan import validate_plan
    validate_plan(plan)
    if batch not in plan['batches']:
        raise VllmBenchError('batch is absent from immutable plan')
    cached = plan['workload_kind'] == 'cached_prefill'
    if cached:
        from cached_prefix_workload import batch_payload
        dataset = result_dir / 'custom.jsonl'
        try:
            if dataset.read_bytes() != batch_payload(plan, batch):
                raise VllmBenchError('custom dataset differs from frozen batch content/row order')
        except OSError as exc:
            raise VllmBenchError(f'cannot read frozen custom dataset {dataset}: {exc}') from exc
    elif plan['generation']['prompt_representation'] != 'token_ids':
        raise VllmBenchError('cold/decode require token-ID prompts')
    argv = [str(binary), '--backend', 'openai', '--base-url', endpoint, '--model', model,
            '--tokenizer', tokenizer, '--dataset-name', 'custom' if cached else 'random',
            '--num-prompts', str(plan['concurrency']), '--max-concurrency', str(plan['concurrency'])]
    if cached:
        argv.extend(('--dataset-path', str(dataset), '--custom-output-len', '-1', '--no-oversample',
                     '--skip-chat-template'))
    else:
        argv.extend(('--random-input-len', str(plan['shape']['input_tokens']),
                     '--random-output-len', str(plan['shape']['output_tokens']), '--random-range-ratio', '0',
                     '--prompt-token-ids'))
    argv.extend(('--seed', str(batch['seed']), '--num-warmups', '0', '--ready-check-timeout-sec', '0',
            '--request-rate', 'inf', '--disable-shuffle', '--no-steady-state',
            '--save-result', '--save-detailed', '--result-dir', str(result_dir),
            '--result-filename', 'bench-result.json', '--request-id-prefix', request_id_prefix))
    sampling = plan['sampling']
    if sampling['ignore_eos']:
        argv.append('--ignore-eos')
    for key in ('temperature', 'top_p', 'top_k'):
        if key in sampling:
            argv.extend(('--' + key.replace('_', '-'), str(sampling[key])))
    argv.extend(('--extra-body', json.dumps({'return_token_ids': True, 'echo': False, 'ignore_eos': sampling['ignore_eos']}, sort_keys=True)))
    return argv


async def run_command(argv: list[str], *, root: Path, timeout: float, launcher: dict) -> dict[str, Any]:
    """Bound the whole process group; preserve stdout/stderr even on cancellation."""
    if _sha256_file(Path(launcher['path'])) != launcher['sha256']:
        raise VllmBenchError('frozen process launcher changed')
    invocation = [launcher['path'], '--pdeathsig', 'KILL', '--', *argv]
    root.mkdir(parents=True, exist_ok=True)
    process = None
    timed_out = False
    with (root / 'stdout.log').open('xb') as stdout, (root / 'stderr.log').open('xb') as stderr:
        try:
            env = {k: v for k, v in os.environ.items() if k.lower() not in
                   ('http_proxy', 'https_proxy', 'all_proxy', 'openai_api_key')}
            env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
            process = await asyncio.create_subprocess_exec(*invocation, stdout=stdout, stderr=stderr,
                                                           env=env, start_new_session=True)
            try:
                await asyncio.wait_for(process.wait(), timeout)
            except asyncio.TimeoutError:
                timed_out = True
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await process.wait()
            for stream in (stdout, stderr):
                stream.flush()
                os.fsync(stream.fileno())
    return {'invocation': invocation, 'returncode': process.returncode, 'timed_out': timed_out,
            'result_present': (root / 'bench-result.json').is_file()}
