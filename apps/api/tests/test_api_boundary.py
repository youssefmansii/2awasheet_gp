"""Exercise the actual ASGI app without opening a port or touching real match files."""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from local_boundary import LocalAnalyticsBoundary
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


class ApiBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def request(self, method='GET', path='/api/health', client='127.0.0.1',
                      host='127.0.0.1', **kwargs):
        transport = httpx.ASGITransport(app=main.app, client=(client, 12345))
        async with httpx.AsyncClient(transport=transport, base_url=f'http://{host}:8000') as api:
            return await api.request(method, path, **kwargs)

    async def test_loopback_health_remains_available(self):
        self.assertEqual((await self.request()).status_code, 200)

    async def test_remote_client_cannot_access_api(self):
        self.assertEqual((await self.request(client='203.0.113.5')).status_code, 403)

    async def test_untrusted_host_cannot_rebind_to_local_api(self):
        self.assertEqual((await self.request(host='attacker.example')).status_code, 403)

    async def test_untrusted_origin_cannot_post_even_from_loopback(self):
        r = await self.request('POST', '/api/analytics/warmup', headers={'Origin': 'https://attacker.example'})
        self.assertEqual(r.status_code, 403)

    async def test_oversized_request_rejected_before_multipart_parsing(self):
        r = await self.request('POST', '/api/matches/upload', headers={'Content-Length': str(2**40)})
        self.assertEqual(r.status_code, 413)

    async def test_chunked_body_limit_cannot_be_bypassed_without_content_length(self):
        async def consume(request: Request):
            await request.body()
            return JSONResponse({'ok': True})

        app = LocalAnalyticsBoundary(Starlette(routes=[Route('/', consume, methods=['POST'])]), max_request_bytes=8)
        async def chunks():
            yield b'12345'
            yield b'67890'

        transport = httpx.ASGITransport(app=app, client=('127.0.0.1', 12345))
        async with httpx.AsyncClient(transport=transport, base_url='http://127.0.0.1') as client:
            r = await client.post('/', content=chunks())
            self.assertEqual(r.status_code, 413)

    async def test_forwarded_headers_cannot_turn_api_into_a_proxied_service(self):
        self.assertEqual((await self.request(headers={'X-Forwarded-For': '127.0.0.1'})).status_code, 403)

    async def test_metadata_only_update_preserves_tracking(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'MATCH_DATA_DIR': folder}):
            # Only a contract fixture is written, never the owner's capture.
            p = Path(folder) / 'tracking_12.jsonl'
            p.write_bytes(b'{"frameNum":1,"homePlayers":[]}\n')
            r = await self.request('POST', '/api/matches/upload', data={'match_id': '12'},
                                   files={'metadata': ('metadata.json', b'{"pitchLength":105}', 'application/json')})
            self.assertEqual(r.status_code, 200)
            self.assertEqual(p.read_bytes(), b'{"frameNum":1,"homePlayers":[]}\n')
            self.assertTrue((Path(folder) / 'metadata_12.json').exists())
