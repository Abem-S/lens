"""Optional tester gate: one shared access code, so strangers cannot spend the
model credit. Off unless LENS_ACCESS_CODE is set.

Open https://host/?code=CODE once (or type the code on the locked page): a signed
pass is set as a cookie (30 days) and the code leaves the URL.

Phone pairing never shows the code. The pairing QR carries a short-lived pairing
token (`/phone/<id>?pair=<token>`, valid 15 minutes, made only for a laptop that is
already let in); opening it gives the phone its own pass (12 hours).
"""

from __future__ import annotations

import hashlib
import hmac
import html
import time
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, urlencode

COOKIE = "lens_access"
OPEN_PATHS = ("/api/health",)
PASS_TTL_S = 30 * 24 * 3600  # a laptop that typed the code
PHONE_PASS_TTL_S = 12 * 3600  # a phone let in by a pairing QR
PAIR_TTL_S = 15 * 60  # how long a pairing QR can be scanned

LOCKED_PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Muna Lens</title><body style="font-family:system-ui;background:#f6f3ec;color:#1d2a3a;display:grid;
place-items:center;min-height:90vh;text-align:center;margin:0;padding:16px"><div style="max-width:28rem">
<h1>Muna Lens</h1><p>Lens is in private testing. Open the tester link you were given, or type your code.</p>
{note}<form method=get action="{path}" style="display:flex;gap:8px;justify-content:center;flex-wrap:wrap">
<label for=code style="position:absolute;left:-9999px">Tester code</label>
<input id=code name=code autocomplete=off placeholder="Tester code" required
 style="font:inherit;padding:10px 12px;border:1px solid #b9b2a3;border-radius:8px;min-width:12rem">
<button style="font:inherit;padding:10px 16px;border:0;border-radius:8px;background:#1d2a3a;color:#fff">
Open Lens</button></form></div></body>"""
WRONG_CODE = '<p role=alert style="color:#a33">That code did not work. Check it and try again.</p>'
EXPIRED_PAIR = ('<p role=alert style="color:#a33">This pairing code has expired. On your laptop, start phone mode '
                "again and scan the new code.</p>")


def _key(code: str) -> bytes:
    return hashlib.sha256(f"lens-access:{code}".encode()).digest()


def _sig(code: str, msg: str) -> str:
    return hmac.new(_key(code), msg.encode(), hashlib.sha256).hexdigest()[:32]


def make_pass(code: str, ttl_s: float = PASS_TTL_S, now: float | None = None) -> str:
    exp = int((now or time.time()) + ttl_s)
    return f"p.{exp}.{_sig(code, f'pass:{exp}')}"


def valid_pass(code: str, value: str | None, now: float | None = None) -> bool:
    if not value:
        return False
    if hmac.compare_digest(value, code):  # cookies set before passes were signed
        return True
    parts = value.split(".")
    if len(parts) != 3 or parts[0] != "p" or not parts[1].isdigit():
        return False
    return int(parts[1]) > (now or time.time()) and hmac.compare_digest(parts[2], _sig(code, f"pass:{parts[1]}"))


def pair_token(code: str, sid: str, now: float | None = None) -> str:
    exp = int((now or time.time()) + PAIR_TTL_S)
    return f"{exp}.{_sig(code, f'pair:{sid}:{exp}')}"


def valid_pair(code: str, sid: str, token: str | None, now: float | None = None) -> bool:
    if not token or "." not in token:
        return False
    exp, sig = token.split(".", 1)
    return exp.isdigit() and int(exp) > (now or time.time()) and hmac.compare_digest(sig, _sig(code, f"pair:{sid}:{exp}"))


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

    async def _let_in(self, scope, send, query: dict, ttl_s: float) -> None:
        rest = urlencode({k: v[0] for k, v in query.items()})
        location = scope["path"] + (f"?{rest}" if rest else "")
        secure = "; Secure" if scope.get("scheme") == "https" else ""
        cookie = f"{COOKIE}={make_pass(self.code, ttl_s)}; Path=/; Max-Age={int(ttl_s)}; HttpOnly; SameSite=Lax{secure}"
        await send({"type": "http.response.start", "status": 303, "headers": [
            (b"location", location.encode()), (b"set-cookie", cookie.encode()), (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": b""})

    async def __call__(self, scope, receive, send):
        if not self.code or scope["type"] not in ("http", "websocket") or scope["path"] in OPEN_PATHS:
            return await self.app(scope, receive, send)
        if valid_pass(self.code, self._cookie(scope)):
            return await self.app(scope, receive, send)
        query = parse_qs(scope.get("query_string", b"").decode())
        note = ""
        if scope["type"] == "http":
            code = query.pop("code", [None])[0]
            if code is not None:
                if self._ok(code.strip()):
                    return await self._let_in(scope, send, query, PASS_TTL_S)
                note = WRONG_CODE
            pair = query.pop("pair", [None])[0]
            if pair is not None:
                path = scope["path"]
                sid = path[len("/phone/"):].strip("/") if path.startswith("/phone/") else ""
                if sid and valid_pair(self.code, sid, pair):
                    return await self._let_in(scope, send, query, PHONE_PASS_TTL_S)
                note = EXPIRED_PAIR
        if scope["type"] == "websocket":
            await receive()  # the connect event
            return await send({"type": "websocket.close", "code": 4401})
        api = scope["path"].startswith("/api/")
        if api:
            body = b'{"detail":"Your tester access has expired. Open the tester link again."}'
        else:
            body = LOCKED_PAGE.format(note=note, path=html.escape(scope["path"], quote=True)).encode()
        await send({"type": "http.response.start", "status": 401, "headers": [
            (b"content-type", b"application/json" if api else b"text/html; charset=utf-8")]})
        await send({"type": "http.response.body", "body": body})
