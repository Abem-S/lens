"""Live tracking: SAM 3.1 tracks the hand and pen on a stream of frames; this turns
the tracks into events the tutor can act on.

Data flow: browser --frames (~5 fps, 640 px JPEG)--> app /ws/track --> Modal
`muna-lens-sam31` (one WebSocket per session) --tracked objects--> `TrackEvents`
here --events--> browser (which uploads a full frame for a check on `line_done`).

Events (pure logic, unit-tested on synthetic tracks):
- phase: "writing" (pen tip moving on the page), "paused" (hand or pen still there),
  "clear" (no hand or pen over the page)
- pointer: where the student points (pen tip, else the fingertip), every frame
- line_done: a writing spell ended and the hand has left the new line; carries the
  box of the region written (from the pen tip's path) — the cue for a page check
- follow: where the outlined line is now, after the page or camera moved
"""

from __future__ import annotations

import base64
import contextlib
import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np
from fastapi import WebSocket

from lens.pointing import fingertip_from_mask

log = logging.getLogger("lens.tracking")
APP_NAME = "muna-lens-sam31"

Box = tuple[float, float, float, float]  # normalised x, y, w, h


def track_token() -> str:
    """Shared secret with the Modal endpoint (same derivation as modal_apps/sam31.py)."""
    import hashlib

    explicit = os.environ.get("LENS_TRACK_TOKEN")
    if explicit:
        return explicit
    secret = os.environ.get("MODAL_TOKEN_SECRET", "")
    return hashlib.sha256(f"muna-lens-track:{secret}".encode()).hexdigest()[:32] if secret else ""


def unpack_mask(m: dict) -> np.ndarray:
    w, h = m["w"], m["h"]
    bits = np.unpackbits(np.frombuffer(base64.b64decode(m["bits"]), dtype=np.uint8))[: w * h]
    return bits.reshape(h, w).astype(bool)


@dataclass
class Tracked:
    concept: str  # "hand" | "pen"
    id: int
    score: float
    box: Box
    mask: np.ndarray | None = None


def parse_objects(objects: list[dict]) -> list[Tracked]:
    out = []
    for o in objects:
        mask = unpack_mask(o["mask"]) if o.get("mask") else None
        out.append(Tracked(o["concept"], int(o["id"]), float(o["score"]), tuple(o["box"]), mask))
    return out


def _edge_distance(p: tuple[float, float]) -> float:
    return min(p[0], p[1], 1 - p[0], 1 - p[1])


def _grow(m: np.ndarray, r: int = 2) -> np.ndarray:
    g = m.copy()
    for axis in (0, 1):
        for s in range(1, r + 1):
            g |= np.roll(m, s, axis=axis) | np.roll(m, -s, axis=axis)
    return g


def _pen_ends(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """The two ends of a pen mask, along its long axis (pixel x, y)."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return None
    pts = np.stack([xs, ys], 1).astype(float)
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    proj = (pts - c) @ vt[0]
    return pts[int(np.argmin(proj))], pts[int(np.argmax(proj))]


def _edge_contact(mask: np.ndarray, band: int = 3) -> np.ndarray | None:
    """Mean position (pixel x, y) where a mask reaches the frame border (within `band` px), if it does."""
    edge = np.zeros_like(mask)
    edge[:band, :], edge[-band:, :] = mask[:band, :], mask[-band:, :]
    edge[:, :band] |= mask[:, :band]
    edge[:, -band:] |= mask[:, -band:]
    if not edge.any():
        return None
    ey, ex = np.nonzero(edge)
    return np.array([ex.mean(), ey.mean()])


def pen_tip(pen: Tracked, hands: list[Tracked]) -> tuple[float, float] | None:
    """The writing end of a pen: one of the two ends of its long axis.

    1. The arm brings the hand in from a frame edge: the tip is the end farther from
       where the holding hand touches the border.
    2. Else, a pen running off the frame leaves by its back end: the other end.
    3. Else, the writer sits below the page (phone over the notebook, or a page held
       up to a webcam), so the pen points up and away: the upper end.
    """
    if pen.mask is None or not pen.mask.any():
        return None
    h, w = pen.mask.shape
    ends = _pen_ends(pen.mask)
    if ends is None:
        return fingertip_from_mask(pen.mask)
    a, b = ends
    grown = _grow(pen.mask, 3)
    holder = next((hd for hd in hands if hd.mask is not None and hd.mask.shape == pen.mask.shape
                   and (grown & hd.mask).any()), None)
    contact = _edge_contact(holder.mask) if holder is not None else None
    if contact is None:
        contact = _edge_contact(pen.mask)
    if contact is not None:
        tip = a if np.hypot(*(a - contact)) > np.hypot(*(b - contact)) else b
    else:
        tip = a if a[1] < b[1] else b
    return ((tip[0] + 0.5) / w, (tip[1] + 0.5) / h)


def pointer_of(objs: list[Tracked]) -> tuple[tuple[float, float] | None, str | None]:
    """Where the student points: the pen tip if a pen is seen, else the fingertip that
    reaches farthest into the frame (hands holding a notebook stay at the edges)."""
    hands = [o for o in objs if o.concept == "hand" and o.mask is not None]
    pens = sorted((o for o in objs if o.concept == "pen" and o.mask is not None), key=lambda o: -o.score)
    for pen in pens:
        tip = pen_tip(pen, hands)
        if tip is not None:
            return tip, "pen"
    best, best_d = None, -1.0
    for hand in hands:
        tip = fingertip_from_mask(hand.mask)
        if tip is not None and _edge_distance(tip) > best_d:
            best, best_d = tip, _edge_distance(tip)
    return best, ("finger" if best else None)


def box_cover(objs: list[Tracked], box: Box) -> float:
    """Fraction of `box` hidden under hand or pen masks."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return 0.0
    cover = 0.0
    for o in objs:
        if o.mask is None:
            continue
        mh, mw = o.mask.shape
        x0, y0 = int(max(0, x) * mw), int(max(0, y) * mh)
        x1, y1 = int(np.ceil(min(1, x + w) * mw)), int(np.ceil(min(1, y + h) * mh))
        if x1 <= x0 or y1 <= y0:
            continue
        cover = max(cover, float(o.mask[y0:y1, x0:x1].mean()))
    return cover


@dataclass
class EventConfig:
    move_min: float = 0.015  # pen-tip path (normalised) per second that counts as writing
    write_min_s: float = 0.8  # a writing spell shorter than this is a fidget, not a line
    settle_s: float = 0.8  # the new line must be uncovered and the pen still this long
    cover_max: float = 0.08  # "hand clear of the line": at most this much of it hidden
    work_margin: float = 0.06  # a pen tip closer than this to the frame edge is off the page
    finger_margin: float = 0.18  # a bare fingertip must reach this far in (hands holding the page stay out)
    jump_max: float = 0.08  # a tip that moves farther than this in one frame was lifted and moved, not writing
    line_pad: float = 0.025  # padding around the pen path to make the line's box
    window_s: float = 2.0  # motion is measured over this window (a slow uplink may give only ~2 fps)


@dataclass
class TrackEvents:
    """Turns tracked objects, frame by frame, into phase / pointer / line_done events."""

    cfg: EventConfig = field(default_factory=EventConfig)
    phase: str = "clear"
    _tips: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y) of the pen tip
    _spell: list[tuple[float, float, float]] = field(default_factory=list)  # tip path while writing
    _spell_s: float = 0.0
    _pending: Box | None = None  # region written, waiting for the hand to leave it
    _settled_since: float | None = None
    _last_t: float | None = None

    def _on_page(self, p: tuple[float, float] | None, kind: str | None = "pen") -> bool:
        margin = self.cfg.work_margin if kind == "pen" else self.cfg.finger_margin
        return p is not None and _edge_distance(p) >= margin

    def push(self, t: float, objs: list[Tracked]) -> dict:
        """One tracked frame at time `t` (seconds) -> {"phase", "pointer", "events": [...]}."""
        c = self.cfg
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        pointer, kind = pointer_of(objs)
        pen = pointer if kind == "pen" and self._on_page(pointer) else None
        if pen is None or (self._tips and np.hypot(pen[0] - self._tips[-1][1], pen[1] - self._tips[-1][2]) > c.jump_max):
            self._tips.clear()  # pen gone, or lifted and put down elsewhere: start a new path
        if pen is not None:
            self._tips.append((t, *pen))
        self._tips = [p for p in self._tips if t - p[0] <= c.window_s]
        path = sum(np.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(self._tips, self._tips[1:]))
        span = (self._tips[-1][0] - self._tips[0][0]) if len(self._tips) > 1 else 0.0
        moving = len(self._tips) >= 3 and path / max(span, 0.25) >= c.move_min

        events: list[dict] = []
        if pen is not None and moving:
            phase = "writing"
        elif self._on_page(pointer, kind):
            phase = "paused"
        else:
            phase = "clear"

        if phase == "writing":
            if not self._spell:
                self._spell.extend(self._tips[:-1])  # the path that showed it was writing
            self._spell.append((t, *pen))
            self._spell_s += dt
            self._settled_since = None
        elif self._spell:
            if self._spell_s >= c.write_min_s:
                xs = [p[1] for p in self._spell]
                ys = [p[2] for p in self._spell]
                new = (min(xs) - c.line_pad, min(ys) - c.line_pad,
                       max(xs) - min(xs) + 2 * c.line_pad, max(ys) - min(ys) + 2 * c.line_pad)
                self._pending = _union(self._pending, new) if self._pending else new
            self._spell.clear()
            self._spell_s = 0.0

        if self._pending is not None and phase != "writing":
            clear = box_cover(objs, self._pending) <= c.cover_max
            if clear:
                if self._settled_since is None:
                    self._settled_since = t
                if t - self._settled_since >= c.settle_s:
                    box = [round(float(v), 4) for v in _clamp(self._pending)]
                    events.append({"type": "line_done", "box": box})
                    self._pending = None
                    self._settled_since = None
            else:
                self._settled_since = None

        if phase != self.phase:
            events.append({"type": "phase", "phase": phase})
            self.phase = phase
        return {"phase": phase, "pointer": [round(float(pointer[0]), 4), round(float(pointer[1]), 4)] if pointer else None,
                "pointer_kind": kind, "events": events}


def _union(a: Box, b: Box) -> Box:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)


def _clamp(b: Box) -> Box:
    x0, y0 = max(0.0, b[0]), max(0.0, b[1])
    x1, y1 = min(1.0, b[0] + b[2]), min(1.0, b[1] + b[3])
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


# ---------------------------------------------------------------------------------------------
# Relay: browser <-> app <-> Modal. The browser never sees the endpoint or its token.


class TrackHub:
    """Latest pointer per tracking session, so the tutor can answer "this line?" without
    a fresh SAM call when live tracking already knows where the student points."""

    def __init__(self, max_age_s: float = 1.5):
        self.max_age_s = max_age_s
        self._pointers: dict[str, tuple[float, tuple[float, float] | None]] = {}

    def update(self, track_id: str, pointer) -> None:
        self._pointers[track_id] = (time.monotonic(), tuple(pointer) if pointer else None)

    def drop(self, track_id: str) -> None:
        self._pointers.pop(track_id, None)

    def pointer(self, track_id: str | None) -> tuple[float, float] | None:
        if not track_id or track_id not in self._pointers:
            return None
        t, p = self._pointers[track_id]
        return p if time.monotonic() - t <= self.max_age_s else None

    def locator(self, track_id: str | None, fallback):
        """A `locate_fingertip(image)` that prefers the tracked pointer, else asks `fallback`."""
        if not track_id:
            return fallback

        def locate(image: bytes):
            p = self.pointer(track_id)
            if p is not None:
                return p
            return fallback(image) if fallback else None

        return locate


def _now() -> float:
    """Clock for event timing (a seam for tests)."""
    return time.monotonic()


def endpoint_url() -> str:
    """wss:// URL of the Modal tracking socket (LENS_TRACK_URL overrides discovery)."""
    url = os.environ.get("LENS_TRACK_URL")
    if not url:
        import modal

        url = modal.Cls.from_name(APP_NAME, "Tracker")().web.get_web_url()
    return url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/") + "/track"


async def relay(browser, hub: TrackHub, track_id: str, url: str | None = None) -> None:
    """Pump one tracking session: frames up to Modal, events back to the browser.

    `browser` is an accepted Starlette WebSocket. Browser -> app: binary frames (4-byte
    big-endian number + JPEG) and {"type": "follow" | "unfollow", ...}. App -> browser:
    {"type": "ready"}, then one {"type": "track", ...} per processed frame, or
    {"type": "unavailable"} when the service cannot be reached (the browser then keeps
    its still-frame change detection in charge).
    """
    import asyncio
    import json

    import websockets
    from starlette.concurrency import run_in_threadpool
    from starlette.websockets import WebSocketDisconnect

    try:
        url = url or await run_in_threadpool(endpoint_url)
        gpu = await websockets.connect(f"{url}?token={track_token()}", max_size=2**22, open_timeout=120,
                                       ping_interval=20, ping_timeout=60)
    except Exception as e:
        log.warning("tracking unavailable: %s", e)
        await browser.send_json({"type": "unavailable", "reason": "tracking service unreachable"})
        return

    events = TrackEvents()
    started = _now()

    async def up():
        try:
            while True:
                msg = await browser.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    await gpu.send(msg["bytes"])
                elif msg.get("text") is not None:
                    data = json.loads(msg["text"])
                    if data.get("type") in ("follow", "unfollow"):
                        await gpu.send(json.dumps(data))
        except (WebSocketDisconnect, websockets.ConnectionClosed):
            pass

    async def down():
        try:
            async for raw in gpu:
                msg = json.loads(raw)
                if msg.get("type") == "tracks":
                    objs = parse_objects(msg["objects"])
                    out = events.push(_now() - started, objs)
                    hub.update(track_id, out["pointer"])
                    await browser.send_json({
                        "type": "track", "seq": msg.get("seq"), **out, "follow": msg.get("follow") or {},
                        "objects": [{"concept": o.concept, "id": o.id, "box": list(o.box)} for o in objs],
                        "gpu_ms": msg.get("gpu_ms"), "wall_ms": msg.get("wall_ms"),
                    })
                    for e in out["events"]:
                        if e["type"] == "line_done":
                            log.info("track %s line_done %s", track_id, e["box"])
                elif msg.get("type") == "ready":
                    log.info("track %s ready on %s (load %.0fs)", track_id, msg.get("device"), msg.get("load_s", 0))
                    await browser.send_json({"type": "ready", "device": msg.get("device")})
                else:
                    await browser.send_json(msg)
        except websockets.ConnectionClosed:
            pass

    tasks = [asyncio.create_task(up()), asyncio.create_task(down())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for t in tasks:
            t.cancel()
        await gpu.close()
        hub.drop(track_id)
        with contextlib.suppress(Exception):  # the browser may already be gone
            await browser.send_json({"type": "unavailable", "reason": "tracking ended"})


def mount(app, settings, hub: TrackHub | None = None) -> TrackHub:
    """Add `/ws/track?id=...` to the FastAPI app. Returns the hub the tutor reads pointers from."""
    hub = hub or TrackHub()

    @app.websocket("/ws/track")
    async def track(ws: WebSocket):
        await ws.accept()
        if not settings.modal_configured and not os.environ.get("LENS_TRACK_URL"):
            await ws.send_json({"type": "unavailable", "reason": "Modal is not configured"})
            await ws.close()
            return
        await relay(ws, hub, ws.query_params.get("id") or "anon")
        with contextlib.suppress(Exception):
            await ws.close()

    return hub
