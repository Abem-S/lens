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

log = logging.getLogger("lens.app")
SESSION_TTL_S = 6 * 3600
MAX_FRAME_BYTES = 4 * 1024 * 1024


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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="Muna Lens")
    sessions: dict[str, PairSession] = {}
    state: dict = {}

    def spark():
        if "spark" not in state:
            from lens.spark import SparkClient

            state["spark"] = SparkClient.from_settings(settings)
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
                    else:
                        from lens.voice import VoiceClient

                        state[name] = VoiceClient()
                except Exception:
                    log.exception("could not connect %s on Modal", name)
        return state[name]

    def mask_for(image: bytes, box):
        sam = gpu("sam")
        if sam is None:
            return None
        masks = sam.segment(image, boxes=[box], max_masks=1)
        return masks[0].mask if masks else None

    def check_image(image: bytes, problem: str | None):
        from lens.pipeline import check_image as run

        return run(spark(), image, problem, mask_for=mask_for)

    def locate_fingertip(image: bytes):
        from lens.pointing import fingertip_from_mask

        sam = gpu("sam")
        if sam is None:
            return None
        masks = sam.segment(image, text="fingertip", max_masks=1) or sam.segment(image, text="index finger",
                                                                                  max_masks=1)
        return fingertip_from_mask(masks[0].mask) if masks else None

    def prune() -> None:
        now = time.time()
        for sid in [k for k, v in sessions.items() if now - v.created > SESSION_TTL_S]:
            sessions.pop(sid, None)

    @app.get("/api/health")
    def health():
        return {"ok": True, "spark_configured": bool(settings.baseten_key)}

    @app.post("/api/sessions")
    def create_session(request: Request):
        prune()
        sid = secrets.token_urlsafe(6)
        sessions[sid] = PairSession(sid)
        # The page that asks may be on localhost; the phone must use an address it can reach.
        origin = request.headers.get("x-lens-origin") or str(request.base_url).rstrip("/")
        base = settings.public_url or origin
        if "localhost" in base or "127.0.0.1" in base:
            ip = lan_ip()
            if ip:
                base = base.replace("localhost", ip).replace("127.0.0.1", ip)
        return {"id": sid, "phone_url": f"{base.rstrip('/')}/phone/{sid}"}

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

    @app.post("/api/check")
    async def check(image: UploadFile = File(...), problem: str | None = Form(None)):
        if not settings.baseten_key:
            raise HTTPException(503, "BASETEN is not set: page checks are disabled")
        data = await image.read()
        if len(data) > MAX_FRAME_BYTES:
            raise HTTPException(413, "image too large")

        def work():
            page, finding = check_image(data, problem)
            return {"page": page.model_dump(), "finding": finding.model_dump() if finding else None}

        return await run_in_threadpool(work)

    @app.websocket("/ws/tutor")
    async def tutor(ws: WebSocket):
        """F1 + V2: one tutor session per Lens screen (protocol in lens/tutor.py)."""
        from lens.tutor import Services, TutorSession

        await ws.accept()
        if not settings.baseten_key:
            await ws.send_json({"type": "error", "message": "BASETEN is not set: the tutor is off"})
            await ws.close()
            return
        voice = gpu("voice")
        services = Services(brain=spark(), speech=voice, check=check_image,
                            locate_fingertip=locate_fingertip if settings.modal_configured else None)
        session = TutorSession(services, ws.send_json, ws.send_bytes)
        warm = [c for c in (voice, gpu("sam")) if c is not None]
        for c in warm:  # keep GPU containers up while a student is here (cold start ~1 min)
            await run_in_threadpool(c.keep_warm, True)
        pending: dict | None = None
        opened = False
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("text") is not None:
                    data = json.loads(msg["text"])
                    kind = data.get("type")
                    if kind == "hello":
                        session.voice = bool(data.get("voice", True)) and voice is not None
                        await ws.send_json({"type": "ready", "voice": session.voice, "voice_available": voice is not None,
                                            "pointing": services.locate_fingertip is not None})
                        if data.get("opening", True):
                            opened = True
                            await session.start()
                    elif kind == "voice":
                        session.voice = bool(data.get("on")) and voice is not None
                        if session.voice and not opened:
                            opened = True
                            await session.start()  # proactive: Lens speaks first
                        elif not session.voice:
                            await session.on_interrupt()
                    elif kind == "interrupt":
                        await session.on_interrupt()
                    elif kind in ("frame", "utterance", "check"):
                        pending = data  # its binary payload comes next
                elif msg.get("bytes") is not None and pending is not None:
                    payload, kind = msg["bytes"], pending["type"]
                    if len(payload) <= MAX_FRAME_BYTES:
                        if kind == "frame":
                            await session.on_frame(payload)
                        elif kind == "utterance":
                            await session.on_utterance(payload)
                        elif kind == "check":
                            asyncio.create_task(session.on_check(payload, pending.get("reason", "manual")))
                    pending = None
        except WebSocketDisconnect:
            pass
        finally:
            await session.close()
            for c in warm:
                await run_in_threadpool(c.keep_warm, False)

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
    return app
