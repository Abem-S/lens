"""FastAPI app: health, phone pairing + frame relay, page check, and the built frontend.

Run from the repo root: `make dev` (or `make serve` for one process).
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from lens.config import REPO_ROOT, Settings
from lens.limits import MESSAGES, Limiter, Window, client_ip
from lens.tracking import mount as mount_tracking
from lens.warm import PingBoard

log = logging.getLogger("lens.app")
SESSION_TTL_S = 6 * 3600
MAX_FRAME_BYTES = 4 * 1024 * 1024
WARM_DEBOUNCE_S = 30.0  # /api/warm pings the GPU apps at most this often, for everyone together


@dataclass
class PairSession:
    id: str
    created: float = field(default_factory=time.time)
    sockets: dict[str, set[WebSocket]] = field(default_factory=lambda: {"phone": set(), "screen": set()})


def lan_ip() -> str | None:
    """This machine's LAN address, so a phone on the same Wi-Fi can reach it."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent for UDP connect
            return s.getsockname()[0]
    except OSError:
        return None


class _Touched:
    """A GPU client whose real calls tell the session's Warmer the service is warm."""

    def __init__(self, inner, touch):
        self._inner, self._touch = inner, touch

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if not callable(attr) or name.startswith("_"):
            return attr

        def call(*a, **kw):
            self._touch()
            return attr(*a, **kw)

        return call


def _log_to_stderr() -> None:
    """Lens's own INFO lines (per-call cost, the per-session cost tally, latencies) reach the
    server log; without this only warnings would (uvicorn configures only its own loggers)."""
    import os

    lens_log = logging.getLogger("lens")
    if not lens_log.handlers and not logging.getLogger().handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        lens_log.addHandler(h)
    lens_log.setLevel(os.environ.get("LENS_LOG_LEVEL", "INFO").upper())


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    _log_to_stderr()
    app = FastAPI(title="Muna Lens")
    sessions: dict[str, PairSession] = {}
    state: dict = {}
    limiter = Limiter(settings.limits)
    # /ws/track: live SAM 3.1 tracking (lens/tracking.py), capped in sessions and length
    track_hub = mount_tracking(app, settings, limiter=limiter)
    warm_ips: dict[str, Window] = {}
    last_warm = [0.0]
    board = PingBoard()  # one ping per GPU service in flight, across sessions and /api/warm

    def spark():
        if "spark" not in state:
            from lens.spark import SparkClient

            client = SparkClient.from_settings(settings)
            if isinstance(client, SparkClient):
                client.budget = limiter.budget  # the daily cap covers every call
            state["spark"] = client
        return state["spark"]

    def gpu(name: str):
        """Modal clients, created on first use; None when Modal is not configured."""
        if name not in state:
            state[name] = None
            if settings.modal_configured:
                try:
                    if name == "sam":
                        from lens.sam import SamClient

                        state[name] = SamClient()
                    elif name == "voice":
                        from lens.voice import VoiceClient

                        state[name] = VoiceClient()
                    elif name == "sam31":
                        from lens.warm import tracker_ping

                        state[name] = tracker_ping()
                except Exception:
                    log.exception("could not connect %s on Modal", name)
        return state[name]

    def pings() -> dict:
        voice, sam = gpu("voice"), gpu("sam")
        if "glimmer" not in state:
            from lens.warm import glimmer_ping

            state["glimmer"] = glimmer_ping(settings)
        return {"voice": voice.ping if voice else None, "sam": sam.ping if sam else None, "sam31": gpu("sam31"),
                "glimmer": state["glimmer"]}

    def make_mask_for(sam):
        def mask_for(image: bytes, box):
            if sam is None:
                return None
            from lens.pointing import step_mask

            # SAM 3 finds every line like the boxed one: keep the one that is the boxed step.
            masks = sam.segment(image, boxes=[box], max_masks=5)
            return step_mask([m.mask for m in masks], box)

        return mask_for

    def make_checks(brain, sam):
        """Page and line checks bound to one session's Spark client (its cost tally) and SAM."""
        mask_for = make_mask_for(sam)

        def check_image(image: bytes, problem: str | None, **live):
            from lens.pipeline import check_image as run

            return run(brain, image, problem, mask_for=mask_for, **live)  # live: prior, on_read, timings

        def check_line(image: bytes, page, previous, box, problem: str | None, timings: dict | None = None):
            from lens.linecheck import check_line as run

            return run(brain, image, page, box, previous=previous, problem=problem, mask_for=mask_for,
                       timings=timings)

        def locate_fingertip(image: bytes):
            from lens.pointing import pointer_tip

            if sam is None:
                return None
            return pointer_tip(lambda img, text: [m.mask for m in sam.segment(img, text=text, max_masks=1)], image)

        return check_image, check_line, locate_fingertip

    def prune() -> None:
        now = time.time()
        for sid in [k for k, v in sessions.items() if now - v.created > SESSION_TTL_S]:
            sessions.pop(sid, None)

    def ip_of(conn: Request | WebSocket) -> str:
        return client_ip(dict(conn.headers), conn.client.host if conn.client else None)

    @app.get("/api/health")
    def health():
        return {"ok": True, "spark_configured": bool(settings.baseten_key)}

    @app.get("/api/config")
    def config():
        """What the browser needs to behave: when to rest the GPU services."""
        return {"idle_min": settings.limits.idle_min}

    @app.post("/api/warm")
    async def warm(request: Request):
        """The start page was opened and touched: wake the GPU apps once, so the first
        session does not wait for a cold start. One ping each keeps a container up only
        for its scaledown window (about a minute) if no session follows."""
        ip = ip_of(request)
        win = warm_ips.setdefault(ip, Window(1, 120))
        if len(warm_ips) > 10_000:
            warm_ips.clear()
        now = time.monotonic()
        if not settings.modal_configured or not win.allow() or now - last_warm[0] < WARM_DEBOUNCE_S:
            return {"warming": False}
        last_warm[0] = now

        async def ping_once(svc: str, ping) -> None:
            try:
                await asyncio.to_thread(ping)
            except Exception as e:
                log.warning("warm-up ping %s failed: %s", svc, e)
            finally:
                board.pinging.discard(svc)
                board.last[svc] = time.monotonic()

        for svc, ping in pings().items():
            if svc in ("voice", "sam31", "glimmer") and ping is not None and board.claim(svc, now):
                asyncio.create_task(ping_once(svc, ping))
        return {"warming": True}

    @app.post("/api/sessions")
    def create_session(request: Request):
        prune()
        if not limiter.session_start(ip_of(request)):
            raise HTTPException(429, MESSAGES["ip"])
        sid = secrets.token_urlsafe(6)
        sessions[sid] = PairSession(sid)
        # The page that asks may be on localhost; the phone must use an address it can reach.
        origin = request.headers.get("x-lens-origin") or str(request.base_url).rstrip("/")
        base = settings.public_url or origin
        if "localhost" in base or "127.0.0.1" in base:
            ip = lan_ip()
            if ip:
                base = base.replace("localhost", ip).replace("127.0.0.1", ip)
        url = f"{base.rstrip('/')}/phone/{sid}"
        if settings.access_code:  # the phone gets in by scanning: a short-lived pairing token, never the code
            from lens.access import pair_token

            url += f"?pair={pair_token(settings.access_code, sid)}"
        return {"id": sid, "phone_url": url}

    @app.get("/api/sessions/{sid}")
    def session_status(sid: str):
        s = sessions.get(sid)
        if not s:
            raise HTTPException(404, "no such session")
        return {"id": sid, "phone_connected": bool(s.sockets["phone"]), "screens": len(s.sockets["screen"])}

    @app.websocket("/ws/{sid}/{role}")
    async def relay(ws: WebSocket, sid: str, role: str):
        s = sessions.get(sid)
        if s is None or role not in ("phone", "screen"):
            await ws.close(code=4404)
            return
        await ws.accept()
        s.sockets[role].add(ws)
        other = "screen" if role == "phone" else "phone"

        async def send_all(targets: set[WebSocket], message: dict):
            dead = []
            for t in list(targets):
                try:
                    if "bytes" in message and message["bytes"] is not None:
                        await t.send_bytes(message["bytes"])
                    elif message.get("text") is not None:
                        await t.send_text(message["text"])
                except Exception:
                    dead.append(t)
            for t in dead:
                targets.discard(t)

        await send_all(s.sockets[other], {"text": f'{{"type":"{role}_connected"}}'})
        if role == "screen" and s.sockets["phone"]:
            await ws.send_text('{"type":"phone_connected"}')
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None and len(msg["bytes"]) > MAX_FRAME_BYTES:
                    continue
                # Phone frames and events go to screens; screen commands go to phones.
                await send_all(s.sockets[other], msg)
        except WebSocketDisconnect:
            pass
        finally:
            s.sockets[role].discard(ws)
            await send_all(s.sockets[other], {"text": f'{{"type":"{role}_disconnected"}}'})

    http_checks: dict[str, Window] = {}

    @app.post("/api/check")
    async def check(request: Request, image: UploadFile = File(...), problem: str | None = Form(None)):
        if not settings.baseten_key:
            raise HTTPException(503, "BASETEN is not set: page checks are disabled")
        if not limiter.budget.allow():
            raise HTTPException(429, MESSAGES["daily"])
        win = http_checks.setdefault(ip_of(request), limiter.checks())
        if len(http_checks) > 10_000:
            http_checks.clear()
        if not win.allow():
            raise HTTPException(429, MESSAGES["checks"])
        data = await image.read()
        if len(data) > MAX_FRAME_BYTES:
            raise HTTPException(413, "image too large")
        check_image, _, _ = make_checks(spark(), gpu("sam"))

        def work():
            page, finding = check_image(data, problem)
            return {"page": page.model_dump(), "finding": finding.model_dump() if finding else None}

        try:
            return await run_in_threadpool(work)
        except Exception as exc:
            from lens.tutor import check_error_message

            log.exception("http check failed")
            raise HTTPException(503, check_error_message(exc)) from exc

    @app.websocket("/ws/tutor")
    async def tutor(ws: WebSocket):
        """F1 + V2: one tutor session per Lens screen (protocol in lens/tutor.py)."""
        from lens.spark import CostTally, SparkClient
        from lens.tutor import Services, TutorSession
        from lens.warm import Warmer

        await ws.accept()
        if not settings.baseten_key:
            await ws.send_json({"type": "error", "message": "BASETEN is not set: the tutor is off"})
            await ws.close()
            return

        async def refuse(reason: str) -> None:
            await ws.send_json({"type": "limit", "reason": reason, "message": MESSAGES[reason]})
            await ws.close(code=4429)

        if not limiter.session_start(ip_of(ws)):
            return await refuse("ip")
        if not limiter.budget.allow():
            return await refuse("daily")
        if not limiter.sessions.take():
            return await refuse("busy")

        loop = asyncio.get_running_loop()
        opened_at = time.monotonic()
        tally = CostTally()
        session_ref: list[TutorSession] = []

        def on_slow(purpose: str) -> None:  # called from a worker thread by a hedged Spark call
            if session_ref:
                loop.call_soon_threadsafe(lambda: asyncio.ensure_future(session_ref[0].on_slow(purpose)))

        base = spark()
        brain = base.for_session(tally, on_slow) if isinstance(base, SparkClient) else base
        track_id: list[str | None] = [None]

        async def notify(service: str, seconds: float) -> None:
            await ws.send_json({"type": "warm", "service": service, "ready": True, "seconds": round(seconds, 1)})

        warmer = Warmer(pings(), tracking_live=lambda: track_hub.live(track_id[0]) is not None, notify=notify,
                        board=board, voice_on=lambda: bool(session_ref and session_ref[0].voice))
        voice_raw, sam_raw = gpu("voice"), gpu("sam")
        voice = _Touched(voice_raw, lambda: warmer.touch("voice")) if voice_raw is not None else None
        sam = _Touched(sam_raw, lambda: warmer.touch("sam")) if sam_raw is not None else None
        check_image, check_line, locate_fingertip = make_checks(brain, sam)
        from lens import notes as lens_notes

        services = Services(brain=brain, speech=voice, check=check_image, check_line=check_line,
                            locate_fingertip=locate_fingertip if settings.modal_configured else None,
                            notes=lambda n, talk, page, event: lens_notes.update(brain, n, talk, page, event))
        session = TutorSession(services, ws.send_json, ws.send_bytes)
        session_ref.append(session)
        session.start_ticker()  # speaks up by itself when the student seems stuck
        warmer.start()
        checks, utterances = limiter.checks(), limiter.utterances()
        pending: dict | None = None
        opened = False

        async def over(reason: str) -> bool:
            """True (and the student is told) when a limit stops this request."""
            if not limiter.budget.allow():
                reason = "daily"
            elif (reason == "checks" and checks.allow()) or (reason == "utterances" and utterances.allow()):
                return False
            await ws.send_json({"type": "limit", "reason": reason, "message": MESSAGES[reason]})
            return True

        def set_presence(active: bool) -> None:
            warmer.presence(active)
            session.away = not active

        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    data = json.loads(msg["text"])
                    kind = data.get("type")
                    if kind == "hello":
                        track_id[0] = data.get("track")
                        # A live tracking session already knows where the student points.
                        services.locate_fingertip = track_hub.locator(data.get("track"), services.locate_fingertip)
                        # ...and where the line just finished is, for the fast line check.
                        services.line_box = (lambda tid=data.get("track"): track_hub.last_line(tid))
                        # ...and what the student is doing right now (writing / paused / away).
                        services.live = (lambda tid=data.get("track"): track_hub.live(tid))
                        # ...and keeps the outlined step on the live page as it moves.
                        services.follow = (lambda jpeg, box, tid=data.get("track"): track_hub.follow(tid, jpeg, box))
                        session.voice = bool(data.get("voice", True)) and voice is not None
                        # Opening a session view is when to wake the GPU services (cold starts overlap
                        # with the camera starting or the phone pairing), as long as the student stays.
                        set_presence(bool(data.get("active", True)))
                        await ws.send_json({"type": "ready", "voice": session.voice, "voice_available": voice is not None,
                                            "pointing": services.locate_fingertip is not None,
                                            "idle_min": settings.limits.idle_min})
                        if data.get("opening", True):
                            opened = True
                            await session.start()
                    elif kind == "presence":  # the browser's heartbeat: active, or idle / hidden
                        set_presence(bool(data.get("active")))
                    elif kind == "voice":
                        session.voice = bool(data.get("on")) and voice is not None
                        if session.voice and not opened:
                            opened = True
                            await session.start()  # proactive: Lens speaks first
                        elif not session.voice:
                            await session.on_interrupt()
                    elif kind == "client_timing":  # what the student felt, measured in the browser
                        log.info("voice latency (browser) %s",
                                 {k: v for k, v in data.items() if k != "type" and isinstance(v, int | float)})
                    elif kind == "interrupt":
                        await session.on_interrupt()
                    elif kind == "speech_start":
                        await session.on_speech_start(bool(data.get("continues")))
                    elif kind == "speech_pause":
                        await session.on_speech_pause()
                    elif kind == "speech_resume":
                        await session.on_speech_resume()
                    elif kind == "speech_end":
                        if await over("utterances"):
                            await session.on_speech_cancel()
                        else:
                            await session.on_speech_end()
                    elif kind == "speech_cancel":
                        await session.on_speech_cancel()
                    elif kind == "activity":
                        await session.on_activity(str(data.get("phase")))
                    elif kind in ("text", "settings", "read_page", "end"):  # accessibility (lens/tutor.py)
                        await session.on_message(data)
                    elif kind in ("frame", "utterance", "check", "speech"):
                        pending = data  # its binary payload comes next
                elif msg.get("bytes") is not None and pending is not None:
                    payload, kind = msg["bytes"], pending["type"]
                    if len(payload) <= MAX_FRAME_BYTES:
                        if kind == "frame":
                            await session.on_frame(payload)
                        elif kind == "speech":
                            await session.on_speech_chunk(payload)
                        elif kind == "utterance":
                            if not await over("utterances"):
                                await session.on_utterance(payload)
                        elif kind == "check" and not await over("checks"):
                            asyncio.create_task(session.on_check(payload, pending.get("reason", "manual"),
                                                                 pending.get("box")))
                    pending = None
        except WebSocketDisconnect:
            pass
        finally:
            limiter.sessions.give()
            warmer.stop()
            track_hub.follow(track_id[0], None)  # nothing outlined once the session ends
            await session.close()
            tally.gpu_s = warmer.coverage.gpu_seconds()
            log.info("session cost (estimate) %s", tally.summary(time.monotonic() - opened_at, settings.gpu_rates))

    dist = REPO_ROOT / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            f = (dist / path).resolve()
            if path and f.is_file() and dist.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(Path(dist / "index.html"))

    app.state.sessions = sessions
    app.state.limiter = limiter
    if settings.access_code:
        from lens.access import AccessGate

        app.add_middleware(AccessGate, code=settings.access_code)
    return app
