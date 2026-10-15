"""SSE byte-boundary and classification regressions, independent of TCP chunking."""
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sse_evidence import SSEParser


def event(choice=None, **extra):
    return ('data: ' + json.dumps({'choices': [] if choice is None else [choice], **extra}, ensure_ascii=False) + '\r\n\r\n').encode()


class SSETests(unittest.TestCase):
    def test_every_byte_boundary_and_finish_with_content(self):
        wire = (b': keepalive\r\n\r\n' + event({'index':0, 'text':'', 'token_ids':[]}, prompt_token_ids=[9]) +
                event({'index':0, 'text':'', 'token_ids':[1,2]}) +
                event({'index':0, 'text':' \n中', 'token_ids':[3,4], 'finish_reason':'length'}) +
                event(usage={'completion_tokens':4}) + b'data: [DONE]\r\n\r\n')
        parser = SSEParser()
        result = []
        for index, value in enumerate(wire):
            result.extend(parser.feed(bytes([value]), index+1))
        parser.eof()
        self.assertEqual([e['output_token_ids'] for e in result], [[],[1,2],[3,4],[],[]])
        self.assertEqual(result[2]['text'], ' \n中')
        self.assertEqual(result[2]['timestamp_ns'], result[2]['byte_end'])
        self.assertEqual(result[-1]['byte_end'], len(wire))

    def test_cr_only_at_eof_and_bom_preserve_wire_bounds(self):
        wire = b'\xef\xbb\xbfdata: {"choices":[{"token_ids":[1],"finish_reason":"length"}]}\r\rdata: [DONE]\r\r'
        parser = SSEParser()
        result = parser.feed(wire, 42)
        result.extend(parser.eof())
        self.assertEqual(len(result), 2)
        self.assertEqual(result[-1]['byte_end'], len(wire))
        self.assertEqual(result[-1]['timestamp_ns'], 42)
