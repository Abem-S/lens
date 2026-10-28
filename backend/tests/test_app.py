"""App plumbing: health, pairing sessions, and the phone -> screen relay (D1 backend)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from lens.app import create_app
from lens.config import Settings


def settings(**kw):
    base = {"baseten_key": None, "spark_base_url": "", "spark_model": "", "spark_price_in": None,
            "spark_price_out": None, "public_url": None, "cache_dir": Path("/tmp")}
    base.update(kw)
    return Settings(**base)


def test_health_reports_missing_key():
    c = TestClient(create_app(settings()))
    assert c.get("/api/health").json() == {"ok": True, "spark_configured": False}


def test_check_is_disabled_without_key():
    c = TestClient(create_app(settings()))
    r = c.post("/api/check", files={"image": ("p.jpg", b"x", "image/jpeg")})
    assert r.status_code == 503


def test_session_uses_public_url_for_phone():
    c = TestClient(create_app(settings(public_url="https://lens.example.com")))
    body = c.post("/api/sessions").json()
    assert body["phone_url"] == f"https://lens.example.com/phone/{body['id']}"


def test_phone_frames_reach_the_screen_and_commands_reach_the_phone():
    c = TestClient(create_app(settings()))
    sid = c.post("/api/sessions").json()["id"]
    with c.websocket_connect(f"/ws/{sid}/screen") as screen:
        with c.websocket_connect(f"/ws/{sid}/phone") as phone:
            assert screen.receive_json() == {"type": "phone_connected"}
            assert c.get(f"/api/sessions/{sid}").json()["phone_connected"] is True
            phone.send_bytes(b"\xff\xd8frame")
            assert screen.receive_bytes() == b"\xff\xd8frame"
            phone.send_json({"type": "ink", "reason": "new_ink"})
            assert screen.receive_json() == {"type": "ink", "reason": "new_ink"}
            screen.send_json({"type": "capture"})
            assert phone.receive_json() == {"type": "capture"}
        assert screen.receive_json() == {"type": "phone_disconnected"}


def test_unknown_session_is_refused():
    c = TestClient(create_app(settings()))
    with pytest.raises(WebSocketDisconnect) as exc, c.websocket_connect("/ws/nope/phone") as ws:
        ws.receive_text()
    assert exc.value.code == 4404
