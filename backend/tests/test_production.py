"""Production readiness: rate limits, the daily cap, the access gate and pairing tokens,
GPU warm-keeping only while a student is active, stalled model calls, the cost tally,
and a swaying page that must not repeat a hint."""

import asyncio
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from test_spark import client_with, completion, sse_response
from test_tutor import FakeBrain, run, session, settle, wrong_page

from lens import access
from lens.app import create_app
from lens.config import Settings
from lens.limits import MESSAGES, BudgetExceeded, DailyBudget, Limiter, Limits, Slots, Window, client_ip
from lens.spark import CostTally, SparkClient, SparkStalled
from lens.warm import PING_EVERY_S, PRESENCE_FRESH_S, TRACK_PREWARM_S, Coverage, Warmer


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def settings(**kw):
    base = {"baseten_key": None, "spark_base_url": "", "spark_model": "", "spark_price_in": None,
            "spark_price_out": None, "public_url": None, "cache_dir": Path("/tmp")}
    base.update(kw)
    return Settings(**base)


# ---------- limits ----------

def test_window_allows_n_per_period_then_recovers():
    clock = Clock()
    w = Window(3, 60, clock)
    assert [w.allow() for _ in range(4)] == [True, True, True, False]
    clock.t += 61
    assert w.allow()


def test_zero_means_unlimited():
    assert all(Window(0, 60).allow() for _ in range(100))
    s = Slots(0)
    assert all(s.take() for _ in range(100))


def test_slots_cap_concurrent_sessions():
    s = Slots(2)
    assert s.take() and s.take() and not s.take()
    s.give()
    assert s.take()


def test_daily_budget_caps_calls_and_spend_and_resets_next_day():
    day = ["2026-10-31"]
    b = DailyBudget(max_calls=3, max_usd=1.0, day=lambda: day[0])
    for _ in range(3):
        assert b.allow()
        b.record(0.01)
    assert not b.allow()  # call cap
    day[0] = "2026-11-01"
    assert b.allow()
    b.record(1.5)
    assert not b.allow()  # spend cap


def test_limiter_counts_session_starts_per_ip():
    lim = Limiter(Limits(sessions_per_ip_hour=2))
    assert lim.session_start("1.1.1.1") and lim.session_start("1.1.1.1")
    assert not lim.session_start("1.1.1.1")
    assert lim.session_start("2.2.2.2")


def test_client_ip_takes_the_last_public_forwarded_address():
    assert client_ip({"x-forwarded-for": "6.6.6.6, 8.8.8.8, 10.0.0.2"}, "10.0.0.1") == "8.8.8.8"
    assert client_ip({}, "127.0.0.1") == "127.0.0.1"
    assert client_ip({"x-forwarded-for": "garbage"}, "1.2.3.4") == "1.2.3.4"


def test_limits_read_from_env(monkeypatch):
    monkeypatch.setenv("LENS_MAX_SESSIONS", "3")
    monkeypatch.setenv("LENS_DAILY_SPEND_USD", "2.5")
    monkeypatch.setenv("LENS_CHECKS_PER_MIN", "")
    lim = Limits.from_env()
    assert lim.max_sessions == 3 and lim.daily_spend_usd == 2.5 and lim.checks_per_min == Limits().checks_per_min


def test_price_defaults_to_the_documented_estimate(monkeypatch):
    from lens import config

    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("SPARK_PRICE_IN_PER_M", raising=False)
    monkeypatch.setenv("SPARK_PRICE_OUT_PER_M", "")
    s = Settings.from_env()
    assert (s.spark_price_in, s.spark_price_out) == (1.25, 4.25)


def _tutor_app(monkeypatch, **limits):
    monkeypatch.setattr("lens.spark.SparkClient.from_settings", classmethod(lambda cls, s: object()))
    return create_app(settings(baseten_key="k", limits=Limits(**limits)))


def _refused(c, path="/ws/tutor"):
    with pytest.raises(WebSocketDisconnect) as exc, c.websocket_connect(path) as ws:
        msg = ws.receive_json()
        ws.receive_json()
    return msg, exc.value.code


def test_tutor_refuses_when_too_many_sessions_are_open(monkeypatch):
    c = TestClient(_tutor_app(monkeypatch, max_sessions=1))
    with c.websocket_connect("/ws/tutor") as first:
        first.send_json({"type": "hello", "voice": False, "opening": False})
        assert first.receive_json()["type"] == "ready"
        msg, code = _refused(c)
        assert msg == {"type": "limit", "reason": "busy", "message": MESSAGES["busy"]} and code == 4429
    # the slot is given back when the first session closes
    with c.websocket_connect("/ws/tutor") as again:
        again.send_json({"type": "hello", "voice": False, "opening": False})
        assert again.receive_json()["type"] == "ready"


def test_tutor_refuses_an_ip_that_starts_too_many_sessions(monkeypatch):
    c = TestClient(_tutor_app(monkeypatch, sessions_per_ip_hour=1))
    with c.websocket_connect("/ws/tutor"):
        pass
    msg, code = _refused(c)
    assert msg["reason"] == "ip" and code == 4429


def test_checks_per_minute_are_capped_with_a_friendly_message(monkeypatch):
    from lens.checker import check_page
    from lens.contract import Page, Step

    def fake_check(spark, image, problem, mask_for=None, **live):
        return check_page(Page(steps=[Step(id="s1", latex="x = 1")]))

    monkeypatch.setattr("lens.pipeline.check_image", fake_check)
    c = TestClient(_tutor_app(monkeypatch, checks_per_min=1))
    with c.websocket_connect("/ws/tutor") as ws:
        ws.send_json({"type": "hello", "voice": False, "opening": False})
        ws.receive_json()
        for _ in range(2):
            ws.send_json({"type": "check", "reason": "new_ink"})
            ws.send_bytes(b"jpeg")
        seen = []
        while not (any(m["type"] == "result" for m in seen) and any(m["type"] == "limit" for m in seen)):
            seen.append(ws.receive_json())
    limit = next(m for m in seen if m["type"] == "limit")
    assert limit["reason"] == "checks" and "next step" in limit["message"]


def test_daily_cap_refuses_new_sessions(monkeypatch):
    app = _tutor_app(monkeypatch, daily_spark_calls=1)
    app.state.limiter.budget.record(0.0)
    msg, code = _refused(TestClient(app))
    assert msg["reason"] == "daily" and code == 4429


def test_spark_client_refuses_calls_past_the_daily_cap():
    c = client_with(lambda req: sse_response(completion("hi")), price_in=1.25, price_out=4.25)
    c.budget = DailyBudget(max_calls=1, max_usd=0)
    c.chat([{"role": "user", "content": "a"}])
    with pytest.raises(BudgetExceeded):
        c.chat([{"role": "user", "content": "b"}])


# ---------- access code + pairing ----------

def test_pairing_link_carries_a_short_lived_token_not_the_code():
    c = TestClient(create_app(settings(access_code="s3cret")), follow_redirects=False)
    assert c.get("/?code=s3cret").status_code == 303
    url = c.post("/api/sessions").json()["phone_url"]
    assert "s3cret" not in url and "?pair=" in url
    path = url.split("://", 1)[-1].split("/", 1)[1]
    phone = TestClient(c.app, follow_redirects=False)  # a different browser: no cookie
    assert phone.get("/" + path.split("?")[0]).status_code == 401
    r = phone.get("/" + path)
    assert r.status_code == 303 and "pair=" not in r.headers["location"]
    assert phone.get("/api/config").status_code == 200  # the phone is in now (its own pass)


def test_pairing_token_expires_and_is_bound_to_its_session():
    now = time.time()
    tok = access.pair_token("c", "abc", now=now)
    assert access.valid_pair("c", "abc", tok, now=now + 60)
    assert not access.valid_pair("c", "other", tok, now=now + 60)
    assert not access.valid_pair("c", "abc", tok, now=now + access.PAIR_TTL_S + 1)
    assert not access.valid_pair("other-code", "abc", tok, now=now)


def test_wrong_code_shows_the_form_with_a_note_and_pass_is_signed():
    c = TestClient(create_app(settings(access_code="s3cret")), follow_redirects=False)
    r = c.get("/?code=nope")
    assert r.status_code == 401 and "did not work" in r.text and "<form" in r.text
    assert c.get("/phone/x?pair=1.bad").text.count("expired") == 1
    r = c.get("/?code=s3cret")
    cookie = r.headers["set-cookie"]
    assert "s3cret" not in cookie and "lens_access=p." in cookie
    p = access.make_pass("s3cret", ttl_s=10, now=100)
    assert access.valid_pass("s3cret", p, now=105) and not access.valid_pass("s3cret", p, now=111)
    assert access.valid_pass("s3cret", "s3cret")  # cookies from before passes were signed


# ---------- warm only while active ----------

def _warmer(clock, live=None):
    calls = []
    w = Warmer({"voice": lambda: calls.append("voice"), "sam": lambda: None, "sam31": lambda: None,
                "missing": None}, tracking_live=live or (lambda: False), clock=clock)
    claim = w.tick

    def tick():  # as if each claimed ping answered at once
        out = claim()
        w.board.pinging.clear()
        return out

    w.tick = tick
    return w, calls


def test_one_ping_per_service_in_flight_across_sessions():
    clock = Clock()
    from lens.warm import PingBoard

    board = PingBoard()
    a = Warmer({"voice": lambda: None}, clock=clock, board=board)
    b = Warmer({"voice": lambda: None}, clock=clock, board=board)
    a.presence(True)
    b.presence(True)
    assert a.tick() == ["voice"]
    assert b.tick() == []  # a's ping is still waiting on a cold start
    clock.t += PING_EVERY_S * 3
    b.presence(True)
    assert b.tick() == []  # still waiting: a second request could start a second container
    board.pinging.clear()
    assert b.tick() == ["voice"]


def test_nothing_is_warmed_until_the_browser_says_someone_is_there():
    clock = Clock()
    w, _ = _warmer(clock)
    assert w.tick() == []
    w.presence(True)
    assert sorted(w.tick()) == ["sam", "sam31", "voice"]
    assert w.tick() == []  # not again before PING_EVERY_S
    clock.t += PING_EVERY_S
    assert sorted(w.tick()) == ["sam", "sam31", "voice"]


def test_warming_stops_when_idle_hidden_or_the_heartbeat_goes_quiet():
    clock = Clock()
    w, _ = _warmer(clock)
    w.presence(True)
    w.tick()
    w.presence(False)  # idle or hidden tab
    clock.t += PING_EVERY_S
    assert w.tick() == []
    w.presence(True)
    clock.t += PRESENCE_FRESH_S + 1  # a browser that stopped sending heartbeats (crashed, asleep)
    assert w.tick() == []


def test_tracker_is_prewarmed_only_for_a_while_and_not_while_tracking_is_live():
    clock = Clock()
    live = [False]
    w, _ = _warmer(clock, lambda: live[0])
    w.presence(True)
    assert "sam31" in w.tick()
    live[0] = True
    clock.t += PING_EVERY_S
    assert "sam31" not in w.tick()  # its own socket keeps it up
    live[0] = False
    for _ in range(int(TRACK_PREWARM_S // 30)):  # heartbeats every 30 s, as the browser sends them
        clock.t += 30
        w.presence(True)
    assert "sam31" not in w.tick()  # a visitor who never starts tracking stops warming it
    w.presence(False)
    clock.t += 10
    w.presence(True)  # coming back after being idle: warm it again
    assert "sam31" in w.tick()


def test_real_use_counts_as_warm_and_no_second_ping_while_one_is_waiting():
    clock = Clock()
    w, _ = _warmer(clock)
    w.presence(True)
    w.touch("voice")
    assert "voice" not in w.due()
    w.board.pinging.add("sam")
    assert "sam" not in w.due()


def test_warmer_pings_and_reports_ready_once():
    async def go():
        notes = []

        async def notify(svc, secs):
            notes.append(svc)

        calls = []
        w = Warmer({"voice": lambda: calls.append(1)}, notify=notify)
        w.presence(True)
        w.run_once()
        await asyncio.sleep(0.05)
        w.stop()
        return calls, notes

    calls, notes = run(go())
    assert calls == [1] and notes == ["voice"]


def test_coverage_counts_each_second_once():
    cov = Coverage()
    cov.extend("voice", 0)  # covers 0..90
    cov.extend("voice", 45)  # extends to 135
    cov.extend("sam31", 0)
    assert cov.gpu_seconds() == {"l4": 135.0, "l40s": 60.0}


def test_session_cost_summary_per_hour():
    t = CostTally()
    from lens.spark import CallLog

    t.add(CallLog("read", "low", 1000, 2000, cost_usd=0.01))
    t.gpu_s = {"l40s": 1800.0}
    s = t.summary(1800, {"l40s": 2.0})
    assert s["spark_usd"] == 0.01 and s["gpu_usd"] == 1.0 and s["usd_per_hour"] == pytest.approx(2.02)


def test_per_session_spark_copy_tallies_its_own_calls():
    base = client_with(lambda req: sse_response(completion("hi", p=1000, c=1000)), price_in=1.25, price_out=4.25)
    tally = CostTally()
    mine = base.for_session(tally)
    mine.chat([{"role": "user", "content": "a"}])
    assert tally.calls == 1 and tally.spark_usd == pytest.approx(0.0055)
    assert base.calls == []  # the shared client's log is untouched


def test_no_gpu_app_pins_a_container_up():
    src = Path(__file__).resolve().parents[1]
    for f in list((src / "modal_apps").glob("*.py")) + list((src / "src" / "lens").glob("*.py")):
        text = f.read_text()
        assert "min_containers" not in text.replace("no `min_containers`", ""), f
        assert "keep_warm" not in text and "update_autoscaler" not in text, f


# ---------- stalled model calls ----------

def test_hedged_call_gives_up_at_its_deadline_and_says_it_is_slow():
    import threading

    gate = threading.Event()

    def handler(req):
        gate.wait(3)  # every copy is stuck in the queue
        return sse_response(completion("late"))

    slow = []
    c = client_with(handler)
    c.hedge_deadline = 0.5
    c.on_slow = slow.append
    t = time.monotonic()
    with pytest.raises(SparkStalled):
        c.chat([{"role": "user", "content": "x"}], hedge_after=0.1, purpose="read_page")
    gate.set()
    assert time.monotonic() - t < 1.5 and slow == ["read_page"]


def test_slow_check_shows_a_note_and_a_stuck_check_ends_with_a_friendly_error(monkeypatch):
    monkeypatch.setattr("lens.tutor.CHECK_SLOW_S", 0.05)
    monkeypatch.setattr("lens.tutor.CHECK_CAP_S", 0.2)

    def stuck(jpeg, problem, **kw):
        time.sleep(0.5)
        return wrong_page()

    async def go():
        s, wire = session()
        s.services.check = stuck
        await s.on_check(b"jpeg")
        await settle(s)
        return wire

    wire = run(go())
    assert wire.of("note")[0]["text"].startswith("This is taking longer")
    err = wire.of("error")[0]["message"]
    assert "taking too long" in err and "Traceback" not in err
    assert wire.msgs[-1] == {"type": "state", "state": "listening"}


def test_check_error_never_shows_the_exception():
    async def go():
        s, wire = session()

        def boom(jpeg, problem, **kw):
            raise httpx.ConnectError("secret internals at 10.0.0.1")

        s.services.check = boom
        await s.on_check(b"jpeg")
        return wire

    msg = run(go()).of("error")[0]["message"]
    assert "10.0.0.1" not in msg and "try again" in msg


class StallingBrain:
    def __init__(self, delay):
        self.delay = delay

    def stream(self, messages, **kw):
        time.sleep(self.delay)
        yield "Late words. "


class FailingBrain:
    def stream(self, messages, **kw):
        raise BudgetExceeded()
        yield  # pragma: no cover


def test_slow_reply_says_one_moment_and_a_stalled_one_apologises(monkeypatch):
    monkeypatch.setattr("lens.tutor.REPLY_SLOW_S", 0.05)
    monkeypatch.setattr("lens.tutor.REPLY_CAP_S", 0.3)

    async def go(delay):
        s, wire = session(brain=StallingBrain(delay))
        s.page = wrong_page()[0]
        await s.on_utterance(b"what next")
        for _ in range(100):
            await settle(s)
            await asyncio.sleep(0.01)
        return wire.said()

    from lens.tutor import REPLY_FAILED, REPLY_SLOW_SAY

    assert run(go(0.15)) == [REPLY_SLOW_SAY, "Late words."]
    assert run(go(1.0)) == [REPLY_SLOW_SAY, REPLY_FAILED]


def test_reply_past_the_daily_cap_says_so():
    async def go():
        s, wire = session(brain=FailingBrain())
        s.page = wrong_page()[0]
        await s.on_utterance(b"hello")
        await settle(s)
        return wire.said()

    assert run(go()) == [MESSAGES["daily"]]


def test_no_nudges_while_the_student_is_away():
    async def go():
        s, wire = session(brain=FakeBrain("Hello there."))
        s.away = True
        s.live_state = lambda: {"phase": "paused", "present": True, "writing_ago_s": None}
        s._started = s.clock() - 1000
        await s.tick()
        await settle(s)
        return wire.said()

    assert run(go()) == []


# ---------- swaying page: the same mistake is not hinted twice ----------

def _swayed(page, finding, latex=None, dy=0.01):
    steps = [st.model_copy(update={"bbox": (st.bbox[0], st.bbox[1] + dy, st.bbox[2], st.bbox[3]),
                                   **({"latex": latex} if latex and st.id == finding.step_id else {})})
             for st in page.steps]
    return page.model_copy(update={"steps": steps}), finding


def test_swaying_page_reread_slightly_differently_does_not_repeat_the_hint():
    p, f = wrong_page()
    reread = _swayed(p, f, latex=r"2x = 11 +5 \,")  # same ink, read with different spacing marks
    misread = _swayed(p, f, latex="2x = 11 t 5")  # a symbol misread on the second look

    async def go():
        s, wire = session(checks=[(p, f), reread, misread])
        for frame in (b"a", b"b", b"c"):
            await s.on_check(frame, "new_ink")
            await settle(s)
        return wire

    wire = run(go())
    assert wire.said() == ["What happened to the sign of the 5?"]
    assert not any(m["type"] == "say" and m["text"] != wire.said()[0] for m in wire.msgs)


def test_a_new_wrong_attempt_with_different_numbers_is_hinted():
    from lens.tutor import same_step

    assert same_step("2x=11+5", (0.1, 0.35, 0.5, 0.08), "2x=11t5", (0.1, 0.36, 0.5, 0.08))
    assert not same_step("2x=16", (0.1, 0.35, 0.5, 0.08), "2x=17", (0.1, 0.35, 0.5, 0.08))
    assert not same_step("2x=11+5", (0.1, 0.35, 0.5, 0.08), "2x=11+5", (0.1, 0.75, 0.5, 0.08))


def test_tutor_rest_cost_is_logged_at_session_end(monkeypatch, caplog):
    import logging

    caplog.set_level(logging.INFO, logger="lens.app")
    c = TestClient(_tutor_app(monkeypatch))
    with c.websocket_connect("/ws/tutor") as ws:
        ws.send_json({"type": "hello", "voice": False, "opening": False})
        ws.receive_json()
        ws.send_json({"type": "presence", "active": False})
    time.sleep(0.1)
    assert any("session cost (estimate)" in r.getMessage() for r in caplog.records)


def test_spark_client_default_has_a_read_timeout():
    c = SparkClient(api_key="x")
    assert c._client.timeout.read == 75.0


def test_glimmer_is_warmed_only_when_it_is_the_brain():
    from lens.warm import glimmer_ping

    assert glimmer_ping(settings(baseten_key="k", spark_model="meta-models/muse-spark-open-1026")) is None
    assert callable(glimmer_ping(settings(baseten_key="k", spark_model="muse-glimmer",
                                          spark_base_url="https://x.modal.run/v1")))


def test_voice_stays_warm_only_once_it_is_on_after_the_first_minutes():
    clock = Clock()
    on = [False]
    w = Warmer({"voice": lambda: None}, voice_on=lambda: on[0], clock=clock)
    w.presence(True)
    assert w.tick() == ["voice"]  # prewarm: the student will likely press Talk
    w.board.pinging.clear()
    t_end = clock.t + TRACK_PREWARM_S + 1
    while clock.t < t_end:
        clock.t += 30
        w.presence(True)
    assert w.due() == []  # a silent session does not keep an L4 warm
    on[0] = True
    assert w.due() == ["voice"]
