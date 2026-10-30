"""Optional tester gate: one shared access code, so strangers cannot spend the
model credit. Off unless LENS_ACCESS_CODE is set.

Open https://host/?code=CODE once: a cookie is set and the code leaves the URL.
The pairing QR carries the code, so a phone gets in by scanning it.
"""

from __future__ import annotations

import hmac
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, urlencode

COOKIE = "lens_access"
OPEN_PATHS = ("/api/health",)
LOCKED_PAGE = b"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Muna Lens</title><body style="font-family:system-ui;background:#f6f3ec;color:#1d2a3a;display:grid;
place-items:center;min-height:90vh;text-align:center;margin:0;padding:16px"><div><h1>Muna Lens</h1>
<p>Lens is in private testing. Open the tester link you were given.</p></div></body>"""


class AccessGate:
    def __init__(self, app, code: str | None):
        self.app, self.code = app, code

    def _ok(self, value: str | None) -> bool:
        return bool(value) and hmac.compare_digest(value, self.code)

    def _cookie(self, scope) -> str | None:
        for k, v in scope.get("headers", []):
            if k == b"cookie":
                c = SimpleCookie(v.decode("latin-1"))
                if COOKIE in c:
                    return c[COOKIE].value
        return None

    async def __call__(self, scope, receive, send):
        if not self.code or scope["type"] not in ("http", "websocket") or scope["path"] in OPEN_PATHS:
            return await self.app(scope, receive, send)
        if self._ok(self._cookie(scope)):
            return await self.app(scope, receive, send)
        query = parse_qs(scope.get("query_string", b"").decode())
        if scope["type"] == "http" and self._ok((query.pop("code", [None]))[0]):
            rest = urlencode({k: v[0] for k, v in query.items()})
            location = scope["path"] + (f"?{rest}" if rest else "")
            secure = "; Secure" if scope.get("scheme") == "https" else ""
            await send({"type": "http.response.start", "status": 303, "headers": [
                (b"location", location.encode()),
                (b"set-cookie", f"{COOKIE}={self.code}; Path=/; Max-Age=2592000; HttpOnly; SameSite=Lax{secure}"
                 .encode()),
            ]})
            return await send({"type": "http.response.body", "body": b""})
        if scope["type"] == "websocket":
            await receive()  # the connect event
            return await send({"type": "websocket.close", "code": 4401})
        api = scope["path"].startswith("/api/")
        body = b'{"detail":"tester access code required"}' if api else LOCKED_PAGE
        await send({"type": "http.response.start", "status": 401, "headers": [
            (b"content-type", b"application/json" if api else b"text/html; charset=utf-8")]})
        await send({"type": "http.response.body", "body": body})
