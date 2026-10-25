"""Execute frozen cold/decode plans with original bench and one observing gateway."""
from __future__ import annotations

import asyncio
import math
import os
import signal
import socket
import struct
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

import aiohttp

from artifacts import (ExecutionStore, _read_json, _write_atomic,
                       checked_path, file_reference, load_compiled_bundle)
from baseline import resolve_reference, select_reference
from evidence_proxy import EvidenceError, EvidenceProxy, observer_identity
from manifest import canonical_scheduler_identity
from profile import verify_fitted_bundle
from probe import (probe_endpoint, profile_endpoint, profile_listener_matches, requires_cache_reset,
                   validate_profile_binding, validate_reset_receipt)
from steady_state import compute_statistics
from vllm_bench import render_command, run_command
from vllm_bench_collect import convert_evidence


@asynccontextmanager
async def profile_server(root: Path, variant: dict, *, binary: Path, model_path: Path,
                         timeout: float = 60):
    """Own one local profile server and retain its exact launch/exit evidence."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise EvidenceError('server startup timeout must be finite and positive')
    endpoint = profile_endpoint(variant['identity']['endpoint'])
    binary, model_path = binary.expanduser().resolve(), model_path.expanduser().resolve()
    if file_reference(model_path / 'config.json')['sha256'] != variant['identity']['model_config_sha256']:
        raise EvidenceError('server model metadata differs from frozen model identity')
    profile = checked_path(root, variant['profile_path'])
    flags = variant['server_flags']
    reserved = ('--profile', '--model-id', '--model-path', '--port')
    if any(flag.split('=', 1)[0] in reserved for flag in flags):
        raise EvidenceError('managed profile server flags override its bound endpoint/model/profile')
    argv = [str(binary), '--model-path', str(model_path), '--model-id', variant['identity']['model'],
            '--port', str(endpoint.port), '--profile', str(profile), *flags]
    directory = root / 'servers' / uuid.uuid4().hex
    directory.mkdir(parents=True)
    _write_atomic(directory / 'command.json', {'argv': argv, 'executable': file_reference(binary),
                                             'profile': file_reference(profile)})
    environment = {k: v for k, v in os.environ.items() if k.lower() not in ('http_proxy', 'https_proxy', 'all_proxy')}
    with (directory / 'server.log').open('wb') as log:
        child = await asyncio.create_subprocess_exec(*argv, stdout=log, stderr=asyncio.subprocess.STDOUT,
                                                     env=environment, start_new_session=True)
        try:
            _write_atomic(directory / 'started.json', {'pid': child.pid})
            deadline = time.monotonic() + timeout
            while True:
                if child.returncode is not None:
                    raise EvidenceError(f'profile server exited ({child.returncode}); see {directory}')
                try:
                    await asyncio.to_thread(probe_endpoint, variant['identity']['endpoint'],
                                            expected_model=variant['identity']['model'], timeout=min(1, timeout))
                    profile_binding(root, variant, child.pid)
                    break
                except (OSError, ValueError, RuntimeError) as exc:
                    if time.monotonic() >= deadline:
                        raise EvidenceError(f'profile server not ready; see {directory}: {exc}') from exc
                    await asyncio.sleep(0.1)
            yield child.pid
        finally:
            if child.returncode is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(child.wait(), timeout=10)
                except asyncio.TimeoutError:
                    os.killpg(child.pid, signal.SIGKILL)
                    await child.wait()
            _write_atomic(directory / 'stopped.json', {'pid': child.pid, 'returncode': child.returncode})


def profile_binding(root: Path, variant: dict, pid: int | None) -> dict:
    """Bind a local already-loaded profile to the listening process, without a generation probe."""
    if pid is None or pid <= 0:
        raise EvidenceError('profile-backed variants require --server-pid VARIANT=PID')
    endpoint = profile_endpoint(variant['identity']['endpoint'])
    process = Path('/proc') / str(pid)
    namespace = (process / 'ns/net').stat()
    own_namespace = Path('/proc/self/ns/net').stat()
    if (namespace.st_dev, namespace.st_ino) != (own_namespace.st_dev, own_namespace.st_ino):
        raise EvidenceError('profile PID is in a different network namespace')
    args = [x.decode() for x in (process / 'cmdline').read_bytes().split(b'\0') if x]
    def option(name):
        for index, arg in enumerate(args):
            if arg == name and index + 1 < len(args):
                return args[index+1]
            if arg.startswith(name + '='):
                return arg.split('=', 1)[1]
        return None
    configured = checked_path(root, variant['profile_path'])
    supplied = option('--profile')
    if not supplied:
        raise EvidenceError('server process was not started with --profile')
    actual = Path(supplied)
    if not actual.is_absolute():
        actual = (process / 'cwd').resolve() / actual
    if actual.resolve() != configured:
        raise EvidenceError('live server profile path differs from campaign binding')
    port = endpoint.port
    sockets = {os.readlink(fd) for fd in (process / 'fd').iterdir() if fd.exists() and fd.is_socket()}
    listeners = []
    for table in ('tcp', 'tcp6'):
        for line in (process / 'net' / table).read_text().splitlines()[1:]:
            fields = line.split()
            address, encoded_port = fields[1].split(':')
            if int(encoded_port, 16) != port or fields[3] != '0A':
                continue
            if table == 'tcp6':
                raise EvidenceError('IPv6/dual-stack listener at profile endpoint port cannot be attested')
            address = socket.inet_ntop(socket.AF_INET, struct.pack('=I', int(address, 16)))
            listener = {'family': 'IPv4', 'address': address, 'port': port, 'inode': fields[9]}
            if not profile_listener_matches(endpoint, listener):
                continue
            if f'socket:[{fields[9]}]' not in sockets:
                raise EvidenceError('profile endpoint has a listening socket not owned by PID')
            listeners.append(listener)
    binding = {'endpoint': {'address': endpoint.host, 'port': port},
               'listeners': sorted(listeners, key=lambda row: (row['address'], row['inode']))}
    validate_profile_binding(variant['identity']['endpoint'], binding)
    stat = (process / 'stat').read_text().rsplit(')', 1)[1].split()
    start_ticks = int(stat[19])
    started = time.time() - float(Path('/proc/uptime').read_text().split()[0]) + start_ticks / os.sysconf('SC_CLK_TCK')
    if configured.stat().st_mtime > started + 0.02:
        raise EvidenceError('profile changed after server start; restart server before sampling')
    profile = _read_json(configured)
    reference = profile.get('calibration', {})
    manifest_path = checked_path(configured.parent, reference.get('manifest', ''))
    if file_reference(manifest_path)['sha256'] != reference.get('sha256'):
        raise EvidenceError('profile calibration manifest digest mismatch')
    manifest = _read_json(manifest_path)
    if profile.get('schema_version') != 3 or manifest.get('schema_version') != 2:
        raise EvidenceError('unsupported runtime profile/calibration version')
    if manifest.get('model_id') != variant['identity']['model'] or manifest.get('model_config_sha256') != variant['identity']['model_config_sha256']:
        raise EvidenceError('profile/model identity mismatch')
    if profile.get('scheduler') != canonical_scheduler_identity(variant['runtime']['scheduler']):
        raise EvidenceError('profile scheduler differs from configured runtime')
    return {**binding, 'profile': file_reference(configured), 'manifest': file_reference(manifest_path),
            'server_pid': pid, 'process_start_ticks': start_ticks,
            'network_namespace': {'device': namespace.st_dev, 'inode': namespace.st_ino},
            'executable': file_reference((process / 'exe').resolve()), 'argv': args}


def verify_attempt_binding(campaign: dict, variant_name: str, root: Path, binding: dict, cell_id: str) -> None:
    if observer_identity() != campaign['client']['observer']:
        raise EvidenceError('frozen observer changed; recompile campaign into a new artifact root')
    if variant_name != 'target':
        variant = campaign['variants'][variant_name]
        if variant_name == 'zero_cost':
            variant = {**variant, **resolve_reference(root, campaign, binding['reference'], cell_id)}
        if variant_name == 'sim' and verify_fitted_bundle(root, campaign) != binding.get('selection'):
            raise EvidenceError('fitted profile selection changed')
        expected = {k: v for k, v in binding.items() if k not in ('variant_identity', 'observer_sha256', 'reference', 'selection')}
        if profile_binding(root, variant, binding['server_pid']) != expected:
            raise EvidenceError('profile server binding changed')


async def reset_cache(base: str, root: Path, timeout: float) -> None:
    """A false or unavailable reset cannot establish a cold target."""
    async with aiohttp.ClientSession(trust_env=False, timeout=aiohttp.ClientTimeout(total=timeout)) as session:
        async with session.post(base + '/reset_prefix_cache', allow_redirects=False) as response:
            body = await response.read()
            receipt = {'status': response.status, 'body': body.decode(errors='replace')}
            _write_atomic(root / 'reset.json', receipt)
            validate_reset_receipt({**receipt, 'body': body})


async def run_attempt(campaign: dict, cell: dict, plan: dict, variant_name: str,
                      store: ExecutionStore, *, timeout: float, binding: dict) -> dict:
    verify_attempt_binding(campaign, variant_name, store.root, binding, plan['cell_id'])
    previous = store.sealed(plan, variant_name, binding)
    if previous:
        return {**previous, 'skipped': True}
    directory, attempt = store.begin(plan, variant_name, binding)
    variant = campaign['variants'][variant_name]
    identity, client = variant['identity'], campaign['client']
    errors = []
    interrupted = None
    proxy = None
    try:
        _write_atomic(directory / 'observer.json', client['observer'])
        _write_atomic(directory / 'binding.json', binding)
        probe = await asyncio.to_thread(probe_endpoint, identity['endpoint'], expected_model=identity['model'],
                                        timeout=min(10, timeout), attempts=1)
        _write_atomic(directory / 'probe.json', probe)
        async with EvidenceProxy(identity['endpoint'], identity['model'], plan, directory, timeout=timeout) as proxy:
            for batch in plan['batches']:
                name = f"{batch['phase']}-{batch['repeat']}"
                batch_root = directory / 'batches' / name
                batch_root.mkdir(parents=True)
                if requires_cache_reset(plan, variant_name, variant['runtime'], batch):
                    await reset_cache(identity['endpoint'], batch_root, min(10, timeout))
                prefix = f"c2-{campaign['campaign_id'][:12]}-{variant_name}-{attempt}-{uuid.uuid4().hex}-{name}-"
                proxy.register(batch, prefix, batch_root)
                argv = render_command(plan, batch, binary=Path(client['binary']), endpoint=proxy.url,
                                      model=identity['model'], result_dir=batch_root,
                                      request_id_prefix=prefix, tokenizer=client['tokenizer'])
                _write_atomic(batch_root / 'command.json', {'argv': argv, 'target_identity': identity,
                                                           'request_id_prefix': prefix, 'batch': batch})
                outcome = await run_command(argv, root=batch_root, timeout=timeout, launcher=client['process_launcher'])
                _write_atomic(batch_root / 'process.json', outcome)
                await proxy.drain()
                if outcome['timed_out'] or outcome['returncode'] != 0 or not outcome['result_present']:
                    errors.append('bench failed, timed out, or did not write diagnostic result')
                if proxy.errors or errors:
                    break
    except (Exception, asyncio.CancelledError) as exc:
        errors.append(f'{type(exc).__name__}: {exc}')
        if isinstance(exc, asyncio.CancelledError):
            interrupted = exc
    if proxy:
        errors.extend(proxy.errors)
    try:
        verify_attempt_binding(campaign, variant_name, store.root, binding, plan['cell_id'])
    except Exception as exc:
        errors.append(f'attempt binding lost: {exc}')
    _write_atomic(directory / 'attempt.json', {'attempt': attempt, 'errors': errors,
                                               'cell_id': plan['cell_id'], 'variant': variant_name})
    try:
        raw_digest = store.raw_index(directory)
        envelope = convert_evidence(plan, variant_name, identity, directory, raw_index_sha256=raw_digest,
                                    client=client, batch_errors=errors)
        _write_atomic(directory / 'workload_evidence.json', envelope.pop('workload_evidence'))
        # workload_evidence is derived from raw, indexed separately to avoid a digest cycle.
        _write_atomic(directory / 'measurements.json', envelope)
        if plan['workload_kind'] == 'steady_decode':
            _write_atomic(directory / 'decode_statistics.json', compute_statistics(plan, envelope))
        verify_attempt_binding(campaign, variant_name, store.root, binding, plan['cell_id'])
        entry = store.finish(plan, variant_name, directory)
    except Exception as exc:
        _write_atomic(directory / 'conversion_failure.json', {'error': f'{type(exc).__name__}: {exc}'})
        store.raw_index(directory)
        entry = store.finish(plan, variant_name, directory, error=str(exc))
    if interrupted:
        raise interrupted
    return entry


async def run_campaign(root: Path, *, variants: list[str], selectors: list[str] | None = None,
                       timeout: float = 120, server_pids: dict[str, int] | None = None) -> list[dict]:
    if not math.isfinite(timeout) or timeout <= 0:
        raise EvidenceError('timeout must be finite and positive')
    root = root.expanduser().resolve()
    campaign = load_compiled_bundle(root)
    cells = campaign['workload']['cells']
    if selectors:
        chosen = []
        for selector in selectors:
            matches = [c for c in cells if selector in (c['cell_id'], c['name'])]
            if len(matches) != 1:
                raise EvidenceError(f'cell selector must match exactly one cell: {selector}')
            if matches[0] not in chosen:
                chosen.append(matches[0])
        cells = chosen
    if not cells or any(c['kind'] not in ('cold_prefill','steady_decode') for c in cells):
        raise EvidenceError('unsupported workload: C2 executes cold_prefill/steady_decode only; select cells explicitly')
    if not variants or len(variants) != len(set(variants)) or any(v not in ('target','zero_cost','sim') for v in variants):
        raise EvidenceError('select distinct target/zero_cost/sim variants')
    client = campaign['client']
    from campaign import _assets
    import copy
    checked = copy.deepcopy({'client': {k: v for k, v in client.items() if k != 'observer'}})
    assets = _assets(checked, root)
    for key in ('binary_sha256','binary_help_sha256','tokenizer_sha256','tokenizer_files','process_launcher'):
        if assets[key] != client[key]:
            raise EvidenceError(f'frozen client input changed: {key}')
    if assets['observer'] != client['observer']:
        raise EvidenceError('frozen observer changed; recompile campaign')
    results = []
    with ExecutionStore(root, campaign['campaign_id']) as store:
        for variant_name in variants:
            variant = campaign['variants'][variant_name]
            for cell in cells:
                binding = {'variant_identity': variant['identity'], 'observer_sha256': client['observer']['sha256']}
                configured = variant
                if variant_name == 'zero_cost':
                    binding['reference'] = select_reference(root, campaign, cell['cell_id'])
                    configured = {**variant, **resolve_reference(root, campaign, binding['reference'], cell['cell_id'])}
                if variant_name == 'sim':
                    binding['selection'] = verify_fitted_bundle(root, campaign)
                if variant_name != 'target':
                    binding.update(profile_binding(root, configured, (server_pids or {}).get(variant_name)))
                plan = _read_json(checked_path(root, cell['plan']['path']))
                result = await run_attempt(campaign, cell, plan, variant_name, store,
                                           timeout=timeout, binding=binding)
                results.append({'cell_id': cell['cell_id'], 'variant': variant_name, **result})
                if result['status'] != 'sealed':
                    return results
    return results
