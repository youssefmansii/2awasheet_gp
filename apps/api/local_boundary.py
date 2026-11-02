"""Fail-closed boundary for the local, single-user analytics companion.

This is not authentication for an Internet service. Do not put this app behind
a proxy/tunnel: the deployed CV API is a separate, authenticated service.
"""
import re

from starlette.datastructures import Headers, URL
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

LOCAL_ORIGIN = re.compile(r'^http://(localhost|127\.0\.0\.1)(:\d+)?$')
MAX_REQUEST_BYTES = 1024 * 1024 * 1024  # 1 GiB, including multipart framing


class LocalAnalyticsBoundary:
    def __init__(self, app, max_request_bytes=MAX_REQUEST_BYTES):
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        client = scope.get('client')
        origin = headers.get('origin')
        trusted = (client is not None and client[0] in ('127.0.0.1', '::1')
                   and URL(scope=scope).hostname in ('localhost', '127.0.0.1', '::1')
                   and (origin is None or LOCAL_ORIGIN.fullmatch(origin))
                   and not any(h in headers for h in ('forwarded', 'x-forwarded-for', 'x-forwarded-host')))
        if not trusted:
            return await JSONResponse({'detail': 'Analytics API is local-only; remote and proxied access is disabled.'},
                                      status_code=403)(scope, receive, send)
        length = headers.get('content-length')
        if length is not None:
            if not length.isascii() or not length.isdigit():
                return await JSONResponse({'detail': 'Invalid Content-Length.'}, status_code=400)(scope, receive, send)
            if int(length) > self.max_request_bytes:
                return await JSONResponse({'detail': 'Request exceeds the analytics upload limit (1 GiB).'},
                                          status_code=413)(scope, receive, send)
        total = 0

        async def bounded_receive():
            nonlocal total
            message = await receive()
            if message['type'] == 'http.request':
                total += len(message.get('body', b''))
                if total > self.max_request_bytes:
                    raise HTTPException(413, 'Request exceeds the analytics upload limit (1 GiB).')
            return message

        await self.app(scope, bounded_receive, send)
