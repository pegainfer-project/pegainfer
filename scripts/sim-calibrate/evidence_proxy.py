"""An asynchronous, at-most-once observing gateway for unmodified vllm-bench."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
from typing import Any

import aiohttp
from aiohttp import web

from manifest import sha256_json
from sse_evidence import EvidenceError, SSEParser


OBSERVER_CONFIG = {
    'implementation': 'aiohttp_streaming', 'clock': 'monotonic_ns',
    'request_start': 'proxy_upstream_dispatch', 'event_observation': 'proxy_upstream_read',
    'compression': 'identity', 'connection_reuse': False, 'http_version': '1.1',
    'redirects': False, 'upstream_retries': 0, 'trust_env': False,
    'bench_optional_metrics': 'disabled_at_proxy',
    'writer_open_files': 64, 'writer_queue_records': 4096, 'max_event_bytes': 4 * 1024 * 1024,
}
RETRY_CONTRACT = {'max_upstream_attempts_per_key': 1, 'duplicate_policy': 'fail_batch',
                  'client_retry_control': 'unavailable'}
TIMING_CONTRACT = {
    'clock': 'monotonic', 'first_token': 'first_output_token_ids_event',
    'last_token': 'last_output_token_ids_event', 'first_text': 'first_nonempty_output_text_event',
    'text_fallback': 'disabled', 'return_token_ids': True, 'echo': False,
    'decode_interval_tokens': 'right_event_output_token_count',
    'measurement_origin': 'http_sse_proxy', 'request_start': 'proxy_upstream_dispatch',
    'event_observation': 'proxy_upstream_read',
}
OBSERVER_SOURCES = (
    'evidence_proxy.py', 'sse_evidence.py', 'requirements-evidence-proxy.txt',
    'vllm_bench_collect.py', 'plan_driver.py', 'vllm_bench.py',
    'measurement_schema.py', 'request_plan.py', 'manifest.py', 'artifacts.py',
    'probe.py', 'campaign.py', 'profile.py', 'steady_state.py', 'baseline.py',
    'pairing.py', 'normalize.py', 'target_statistics.py',
)


def validate_request_body(body: dict, plan: dict, model: str) -> None:
    """The frozen wire contract, shared by acquisition and offline pairing."""
    shape, sampling = plan['shape'], plan['sampling']
    prompt = body.get('prompt')
    if not isinstance(prompt, list) or len(prompt) != shape['input_tokens'] or any(type(t) is not int or t < 0 for t in prompt):
        raise EvidenceError('request prompt IDs differ from planned shape')
    expected = {'model': model, 'max_tokens': shape['output_tokens'],
                'stream': True, 'return_token_ids': True, 'echo': False, **sampling}
    for key, value in expected.items():
        if body.get(key) != value or (isinstance(value, bool) and type(body.get(key)) is not bool):
            raise EvidenceError(f'request {key} differs from frozen contract')
    if body.get('n', 1) != 1 or body.get('best_of', 1) != 1:
        raise EvidenceError('multiple choices are unsupported')
    if body.get('stream_options', {}).get('include_usage') is not True:
        raise EvidenceError('request must include stream usage')


def observer_identity() -> dict[str, Any]:
    directory = Path(__file__).resolve().parent
    requirements = directory / 'requirements-evidence-proxy.txt'
    versions = {}
    for line in requirements.read_text().splitlines():
        if line and not line.startswith('#'):
            name, expected = line.split('==')
            actual = importlib.metadata.version(name)
            if actual != expected:
                raise EvidenceError(f'observer dependency mismatch: {name} {actual} != {expected}')
            versions[name] = actual
    sources = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
               for name in OBSERVER_SOURCES}
    return {'config': dict(OBSERVER_CONFIG), 'dependencies': versions, 'sources': sources,
            'sha256': sha256_json([OBSERVER_CONFIG, versions, sources])}


class EvidenceWriter:
    """One bounded queue; blocking writes/fsync run outside the observation loop."""

    def __init__(self):
        self.queue = asyncio.Queue(maxsize=OBSERVER_CONFIG['writer_queue_records'])
        self.error: BaseException | None = None
        self.task = asyncio.create_task(self._run())
        self.handles = OrderedDict()
        self.created: set[Path] = set()
        self.dirty: set[Path] = set()

    def put(self, path: Path, data: bytes) -> None:
        if self.error:
            raise EvidenceError(f'evidence writer failed: {self.error}')
        try:
            self.queue.put_nowait((path, data))
        except asyncio.QueueFull as exc:
            raise EvidenceError('evidence queue full; attempt invalid') from exc

    def json(self, path: Path, value: Any) -> None:
        self.put(path, (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode())

    def _write(self, path: Path, data: bytes) -> None:
        if path not in self.handles:
            path.parent.mkdir(parents=True, exist_ok=True)
            if len(self.handles) >= OBSERVER_CONFIG['writer_open_files']:
                _, old = self.handles.popitem(last=False)
                old.close()  # Flush to OS; barrier fsyncs each dirty file once.
            self.handles[path] = path.open('ab' if path in self.created else 'xb')
            self.created.add(path)
        self.handles.move_to_end(path)
        self.handles[path].write(data)
        self.dirty.add(path)

    async def _run(self):
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    return
                if not self.error:
                    await asyncio.to_thread(self._write, *item)
            except Exception as exc:
                self.error = exc
            finally:
                self.queue.task_done()

    def _flush(self):
        error = None
        for handle in self.handles.values():
            try:
                handle.close()
            except OSError as exc:
                error = error or exc
        self.handles.clear()
        for path in self.dirty:
            try:
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
            except OSError as exc:
                error = error or exc
        self.dirty.clear()
        if error:
            raise error

    async def barrier(self):
        await self.queue.join()
        await asyncio.to_thread(self._flush)
        if self.error:
            raise EvidenceError(f'evidence writer failed: {self.error}')

    async def close(self):
        try:
            if not self.task.done():
                await self.queue.put(None)
            await self.task
        finally:
            await asyncio.to_thread(self._flush)
        if self.error:
            raise EvidenceError(f'evidence writer failed: {self.error}')


class EvidenceProxy:
    """One observer/clock per attempt, with a closed key set per batch."""

    def __init__(self, upstream: str, model: str, plan: dict, root: Path, *, timeout: float):
        self.upstream = upstream.rstrip('/')
        self.model, self.plan, self.root, self.timeout = model, plan, root, timeout
        self.expected: dict[str, dict] = {}
        self.seen: set[str] = set()
        self.records: dict[str, dict] = {}
        self.errors: list[str] = []
        self.active: set[asyncio.Task] = set()
        self.batch_root: Path | None = None
        self.closing = False
        self.writer = None
        self.session = None
        self.runner = None

    async def __aenter__(self):
        self.writer = EvidenceWriter()
        # POST is non-idempotent in aiohttp; force_close also prevents stale-pool retries.
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=self.plan['concurrency'], force_close=True),
            timeout=aiohttp.ClientTimeout(total=self.timeout), trust_env=False,
            auto_decompress=False, skip_auto_headers={'Accept-Encoding'},
        )
        app = web.Application(client_max_size=32 * 1024 * 1024)
        app.router.add_route('*', '/{tail:.*}', self.handle)
        self.runner = web.AppRunner(app, shutdown_timeout=1)
        try:
            await self.runner.setup()
            server = web.TCPSite(self.runner, '127.0.0.1', 0)
            await server.start()
            self.url = 'http://127.0.0.1:' + str(server._server.sockets[0].getsockname()[1])
            return self
        except BaseException:
            await self.__aexit__(None, None, None)
            raise

    def register(self, batch: dict, prefix: str, root: Path) -> None:
        if self.active:
            raise EvidenceError('previous batch still has active upstream requests')
        self.batch_root = root
        by_key = {row['pairing_key']: row for row in self.plan['logical_requests']}
        self.expected = {f'{prefix}{i}': by_key[key] for i, key in enumerate(batch['request_keys'])}

    async def handle(self, incoming: web.Request):
        task = asyncio.current_task()
        self.active.add(task)
        record = None
        response = None
        try:
            # Original bench always tries /metrics for optional speculative stats.
            # This is not an experiment request and never reaches the target.
            if incoming.method == 'GET' and incoming.path == '/metrics' and not self.closing:
                self.writer.json((self.batch_root or self.root) / 'diagnostics.jsonl',
                                 {'method':'GET','path':'/metrics','status':404,'forwarded':False})
                return web.Response(status=404)
            if self.closing or incoming.method != 'POST' or incoming.path != '/v1/completions':
                raise EvidenceError('unexpected request outside completion batch')
            ids = incoming.headers.getall('x-request-id', [])
            request_id = ids[0] if len(ids) == 1 else None
            if request_id not in self.expected:
                raise EvidenceError('missing, ambiguous or unknown transport request ID')
            if request_id in self.seen:
                raise EvidenceError('duplicate transport request ID; batch invalid')
            self.seen.add(request_id)  # No await between check and reservation.
            logical = self.expected[request_id]
            instance = hashlib.sha256(request_id.encode()).hexdigest()
            directory = self.batch_root / 'requests' / instance
            record = {**logical, 'request_instance_id': instance, 'transport_request_id': request_id,
                      'proxy_ingress_ns': time.monotonic_ns(), 'upstream_dispatch_ns': None,
                      'upstream_attempts': 0, 'http_status': None, 'stream_end_ns': None,
                      'downstream_end_ns': None, 'eof': False, 'done': False, 'finished': False,
                      'error': None, 'directory': str(directory.relative_to(self.root))}
            self.records[logical['pairing_key']] = record
            raw = await incoming.read()
            self.writer.put(directory / 'request-body.bin', raw)
            body = json.loads(raw)
            if not isinstance(body, dict):
                raise EvidenceError('request body must be an object')
            self.writer.json(directory / 'request.json', body)
            validate_request_body(body, self.plan, self.model)
            # Only hop-by-hop/transport headers are removed. Never expose authorization in metadata.
            hop = {'host', 'connection', 'content-length', 'transfer-encoding', 'accept-encoding',
                   'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailer', 'upgrade'}
            hop.update(x.strip().lower() for x in incoming.headers.get('connection', '').split(','))
            headers = {k: v for k, v in incoming.headers.items() if k.lower() not in hop}
            headers['Accept-Encoding'] = 'identity'
            parser = SSEParser(OBSERVER_CONFIG['max_event_bytes'])
            record['upstream_attempts'] = 1
            record['upstream_dispatch_ns'] = time.monotonic_ns()
            async with self.session.post(self.upstream + '/v1/completions', data=raw, headers=headers,
                                         allow_redirects=False) as upstream:
                record['http_status'] = upstream.status
                record['response_headers'] = {k: v for k, v in upstream.headers.items()
                                              if k.lower() in ('content-type', 'content-encoding', 'x-request-id')}
                response = web.StreamResponse(status=upstream.status, headers={
                    'Content-Type': upstream.headers.get('Content-Type', 'application/octet-stream')})
                await response.prepare(incoming)
                offset = 0
                # Ensure empty responses have a raw file too.
                self.writer.put(directory / 'response-body.bin', b'')
                self.writer.put(directory / 'reads.jsonl', b'')
                self.writer.put(directory / 'events.jsonl', b'')
                async for chunk in upstream.content.iter_any():
                    timestamp = time.monotonic_ns()
                    self.writer.put(directory / 'response-body.bin', chunk)
                    self.writer.json(directory / 'reads.jsonl', {
                        'timestamp_ns': timestamp, 'byte_start': offset, 'byte_end': offset + len(chunk)})
                    offset += len(chunk)
                    for event in parser.feed(chunk, timestamp):
                        self.writer.json(directory / 'events.jsonl', event)
                    await response.write(chunk)
                    record['last_downstream_write_ns'] = time.monotonic_ns()
                record['stream_end_ns'] = time.monotonic_ns()
                record.update(eof=True, done=parser.done, finished=parser.finished)
                if upstream.status != 200 or upstream.headers.get('Content-Encoding', 'identity') != 'identity':
                    raise EvidenceError('upstream HTTP status or content encoding invalid')
                if 'text/event-stream' not in upstream.headers.get('Content-Type', ''):
                    raise EvidenceError('upstream response is not SSE')
                for event in parser.eof():
                    self.writer.json(directory / 'events.jsonl', event)
                record.update(done=parser.done, finished=parser.finished)
                await response.write_eof()
                record['downstream_end_ns'] = time.monotonic_ns()
                return response
        except (Exception, asyncio.CancelledError) as exc:
            error = f'{type(exc).__name__}: {exc}'
            self.errors.append(error)
            if record is not None:
                record['error'] = error
            if self.writer:
                self.writer.json((self.batch_root or self.root) / 'rejections.jsonl', {'error': error, 'method': incoming.method, 'path': incoming.path})
            if isinstance(exc, asyncio.CancelledError):
                raise
            if response is not None:
                response.force_close()
                return response
            return web.json_response({'error': 'evidence contract rejected request'}, status=409)
        finally:
            try:
                if record is not None:
                    self.writer.json(self.root / record['directory'] / 'terminal.json', record)
            finally:
                self.active.discard(task)

    async def drain(self):
        if self.active:
            await asyncio.wait_for(asyncio.gather(*list(self.active), return_exceptions=True), self.timeout)
        missing = set(self.expected) - self.seen
        if missing:
            self.errors.append(f'missing {len(missing)} registered request IDs')
        await self.writer.barrier()

    async def __aexit__(self, *_):
        self.closing = True
        try:
            if self.runner:
                await self.runner.cleanup()
            if self.active:
                tasks = list(self.active)
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            try:
                if self.session:
                    await self.session.close()
            finally:
                if self.writer:
                    await self.writer.close()
