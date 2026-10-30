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


def test_tutor_socket_refuses_without_key():
    c = TestClient(create_app(settings()))
    with c.websocket_connect("/ws/tutor") as ws:
        assert ws.receive_json()["type"] == "error"


def test_tutor_socket_runs_a_check_and_speaks_the_hint(monkeypatch):
    from lens.checker import check_page
    from lens.contract import Page, Step

    def fake_check(spark, image, problem, mask_for=None):
        page = Page(problem="Solve 2x + 5 = 11", steps=[Step(id="s1", latex="2x + 5 = 11"),
                                                         Step(id="s2", latex="2x = 16")])
        page, finding = check_page(page)
        return page, finding.model_copy(update={"hint": "What did you do with the 5?"})

    monkeypatch.setattr("lens.pipeline.check_image", fake_check)
    monkeypatch.setattr("lens.spark.SparkClient.from_settings", classmethod(lambda cls, s: object()))
    c = TestClient(create_app(settings(baseten_key="k")))
    with c.websocket_connect("/ws/tutor") as ws:
        ws.send_json({"type": "hello", "voice": False, "opening": False})
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["voice_available"] is False
        ws.send_json({"type": "check", "reason": "new_ink"})
        ws.send_bytes(b"jpeg")
        seen = []
        while not any(m.get("type") == "say" for m in seen):
            seen.append(ws.receive_json())
        result = next(m for m in seen if m["type"] == "result")
        assert result["finding"]["step_id"] == "s2"
        assert next(m for m in seen if m["type"] == "say")["text"] == "What did you do with the 5?"


def test_access_gate_locks_everything_but_health_until_the_code_is_used():
    c = TestClient(create_app(settings(access_code="s3cret")), follow_redirects=False)
    assert c.get("/api/health").status_code == 200
    assert c.post("/api/sessions").status_code == 401
    assert c.get("/").status_code == 401
    with pytest.raises(WebSocketDisconnect) as exc, c.websocket_connect("/ws/tutor") as ws:
        ws.receive_text()
    assert exc.value.code == 4401
    assert c.get("/?code=wrong").status_code == 401
    r = c.get("/?code=s3cret")
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert c.post("/api/sessions").status_code == 200  # cookie now set
    assert c.post("/api/sessions").json()["phone_url"].endswith("?code=s3cret")
