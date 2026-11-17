"""Incremental SSE framing and output evidence; timestamps belong to upstream reads."""
from __future__ import annotations

import json
from typing import Any


class EvidenceError(ValueError):
    """A stream cannot establish complete, attributable output evidence."""


class SSEParser:
    """Keep wire offsets and use the read containing the event's closing boundary."""

    def __init__(self, max_event_bytes: int = 4 * 1024 * 1024):
        self.buffer = bytearray()
        self.offset = 0
        self.event_start = 0
        self.data: list[bytes] = []
        self.event_bytes = 0
        self.sequence = 0
        self.done = False
        self.finished = False
        self.last_read_ns = 0
        self.pending_cr_ns: int | None = None
        self.max_event_bytes = max_event_bytes

    def feed(self, chunk: bytes, timestamp_ns: int, *, _eof: bool = False) -> list[dict[str, Any]]:
        if chunk:
            self.last_read_ns = timestamp_ns
        self.buffer.extend(chunk)
        events = []
        while True:
            positions = [p for p in (self.buffer.find(b'\r'), self.buffer.find(b'\n')) if p >= 0]
            if not positions:
                break
            pos = min(positions)
            boundary_ns = timestamp_ns
            crlf = self.buffer[pos:pos + 2] == b'\r\n'
            if self.pending_cr_ns is not None:
                if not crlf:
                    boundary_ns = self.pending_cr_ns
                self.pending_cr_ns = None
            if self.buffer[pos] == 13 and pos + 1 == len(self.buffer) and not _eof:
                self.pending_cr_ns = boundary_ns
                break  # Resolve optional LF while retaining this CR's read time.
            size = pos + (2 if crlf else 1)
            line = bytes(self.buffer[:pos])
            if self.offset == 0:
                line = line.removeprefix(b'\xef\xbb\xbf')
            del self.buffer[:size]
            self.offset += size
            self.event_bytes += size
            if self.event_bytes > self.max_event_bytes:
                raise EvidenceError('SSE event exceeds byte limit')
            if line == b'':
                if self.data:
                    event = self._event(b'\n'.join(self.data), boundary_ns)
                    events.append(event)
                self.data = []
                self.event_bytes = 0
                self.event_start = self.offset
            elif line.startswith(b'data:'):
                value = line[5:]
                self.data.append(value[1:] if value.startswith(b' ') else value)
        if self.event_bytes + len(self.buffer) > self.max_event_bytes:
            raise EvidenceError('SSE event exceeds byte limit')
        return events

    def _event(self, data: bytes, timestamp_ns: int) -> dict[str, Any]:
        if self.done:
            raise EvidenceError('data after DONE')
        event = {'sequence': self.sequence, 'timestamp_ns': timestamp_ns,
                 'byte_start': self.event_start, 'byte_end': self.offset,
                 'output_token_ids': [], 'text': '', 'finish_reason': None, 'usage': None}
        self.sequence += 1
        if data == b'[DONE]':
            self.done = True
            return {**event, 'kind': 'done'}
        try:
            payload = json.loads(data.decode('utf-8'))
        except (UnicodeDecodeError, ValueError) as exc:
            raise EvidenceError('invalid UTF-8/JSON in SSE data') from exc
        if not isinstance(payload, dict) or payload.get('error') is not None:
            raise EvidenceError('non-object or error SSE payload')
        choices = payload.get('choices', [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise EvidenceError('only one completion choice is supported')
        if choices:
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get('index', 0) != 0:
                raise EvidenceError('invalid completion choice')
            delta = choice.get('delta', choice)
            if not isinstance(delta, dict):
                raise EvidenceError('invalid completion delta')
            ids = choice.get('token_ids', delta.get('token_ids'))
            if ids is not None:
                if not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids):
                    raise EvidenceError('invalid output token IDs')
                event['output_token_ids'] = ids
            text = delta.get('text', delta.get('content', ''))
            if text is not None and not isinstance(text, str):
                raise EvidenceError('invalid output text')
            event['text'] = text or ''
            finish = choice.get('finish_reason')
            if finish is not None and (not isinstance(finish, str) or not finish):
                raise EvidenceError('invalid finish reason')
            event['finish_reason'] = finish
            if self.finished and (event['output_token_ids'] or event['text'] or finish):
                raise EvidenceError('choice content after finish')
            if finish:
                self.finished = True
        event.update(kind='completion', usage=payload.get('usage'), payload=payload)
        return event

    def eof(self) -> list[dict[str, Any]]:
        events = self.feed(b'', self.last_read_ns, _eof=True)
        if self.buffer or self.data:
            raise EvidenceError('unframed partial SSE at EOF')
        if not self.done or not self.finished:
            raise EvidenceError('stream missing DONE or finish reason')
        return events
