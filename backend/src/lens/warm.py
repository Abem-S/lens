"""GPU services stay warm only while a student is actually using Lens.

Nothing on Modal is pinned up (no `min_containers`): every GPU app scales to zero by
itself a short while after its last request (`scaledown_window` in modal_apps/). A
tutor session keeps a service warm by pinging it now and then, and only while the
browser says the student is active (it sends `presence` heartbeats; it stops when the
tab is hidden or nobody has moved, spoken or written for a few minutes). If the app
process dies, the pings stop with it and the containers go away on their own: no
override is ever left behind.

- still-frame SAM (outlines, pointing): pinged while the student is active.
- voice (ASR + TTS): pinged while active once voice is on; before that only for the
  first few minutes of the session (most students press Talk soon).
- SAM 3.1 tracking: pinged for a few minutes after the session opens or the student
  comes back (so the 20-30 s cold start overlaps with pairing or the camera starting);
  once tracking is live its own socket keeps it up, and it closes when the browser
  stops sending frames.

The same bookkeeping gives the session's GPU-seconds estimate for the cost tally:
each ping (or live tracking) covers a service until `now + scaledown window`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

log = logging.getLogger("lens.warm")

PING_EVERY_S = 45.0  # while active; below every scaledown window below
PRESENCE_FRESH_S = 90.0  # a presence heartbeat older than this means "not active"
TRACK_PREWARM_S = 180.0  # ping the tracker this long after a session opens / the student returns
# Must match scaledown_window in modal_apps/ (seconds a container lives after its last request).
SCALEDOWN_S = {"voice": 90.0, "sam": 90.0, "sam31": 60.0, "glimmer": 180.0}
GPU_KIND = {"voice": "l4", "sam": "l4", "sam31": "l40s", "glimmer": "l40s"}  # for the cost estimate


@dataclass
class Coverage:
    """Seconds a service was kept warm on behalf of one session."""
    until: dict[str, float] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)

    def extend(self, service: str, now: float) -> None:
        end = now + SCALEDOWN_S.get(service, 60.0)
        start = max(self.until.get(service, now), now)
        if end > start:
            self.seconds[service] = self.seconds.get(service, 0.0) + (end - start)
            self.until[service] = end

    def gpu_seconds(self) -> dict[str, float]:
        """By GPU kind (each tail counted in full: the container lives that long anyway)."""
        out: dict[str, float] = {}
        for svc, s in self.seconds.items():
            out[GPU_KIND.get(svc, svc)] = out.get(GPU_KIND.get(svc, svc), 0.0) + s
        return out


class PingBoard:
    """Shared by every session in the process (and /api/warm): when each service was last
    pinged or used, and which pings are still waiting. A second request while a container
    is cold-starting can make Modal start a second container, so only one ping per service
    is ever in flight, and none within PING_EVERY_S of the last one."""

    def __init__(self) -> None:
        self.last: dict[str, float] = {}
        self.pinging: set[str] = set()

    def due(self, svc: str, now: float) -> bool:
        return svc not in self.pinging and now - self.last.get(svc, -1e9) >= PING_EVERY_S

    def claim(self, svc: str, now: float) -> bool:
        if not self.due(svc, now):
            return False
        self.last[svc] = now
        self.pinging.add(svc)
        return True


class Warmer:
    """Per tutor session. `pings` maps a service name to a blocking ping (None = not deployed).

    `notify(service, seconds)` is awaited once per service when its first ping answers
    (the browser shows "waking up" until then)."""

    def __init__(self, pings: dict[str, Callable[[], object] | None],
                 tracking_live: Callable[[], bool] = lambda: False,
                 voice_on: Callable[[], bool] = lambda: True,
                 notify: Callable[[str, float], Awaitable[None]] | None = None,
                 clock: Callable[[], float] = time.monotonic, board: PingBoard | None = None):
        self.pings = {k: v for k, v in pings.items() if v is not None}
        self.tracking_live = tracking_live
        self.voice_on = voice_on
        self.notify = notify
        self.clock = clock
        self.board = board or PingBoard()
        self.coverage = Coverage()
        self._presence: float | None = None  # when the browser last said "active"
        self._prewarm_from = clock()
        self._ready: set[str] = set()
        self._task: asyncio.Task | None = None
        self._inflight: set[asyncio.Task] = set()

    # --- presence

    def presence(self, active: bool) -> None:
        now = self.clock()
        if active:
            if not self.active():
                self._prewarm_from = now  # coming back: warm the tracker again
            self._presence = now
        else:
            self._presence = None

    def active(self) -> bool:
        return self._presence is not None and self.clock() - self._presence <= PRESENCE_FRESH_S

    # --- what to ping now (pure: unit-tested)

    def due(self) -> list[str]:
        now = self.clock()
        if not self.active():
            return []
        out = []
        for svc in self.pings:
            if svc == "sam31" and (self.tracking_live() or now - self._prewarm_from > TRACK_PREWARM_S):
                continue
            # Voice: kept warm once the student turned it on; before that only for the first
            # minutes (most students press Talk soon), not for a whole silent session.
            if svc == "voice" and not self.voice_on() and now - self._prewarm_from > TRACK_PREWARM_S:
                continue
            # A ping still waiting (cold start), or a ping or real call moments ago (by any
            # session), keeps it warm already.
            if self.board.due(svc, now):
                out.append(svc)
        return out

    def touch(self, svc: str) -> None:
        """The session really used `svc` (a transcript, speech, a mask): it is warm now."""
        now = self.clock()
        self.board.last[svc] = now
        self.coverage.extend(svc, now)

    def tick(self) -> list[str]:
        """Account coverage and return the services to ping now (claimed on the board)."""
        now = self.clock()
        if self.tracking_live():
            self.coverage.extend("sam31", now)
        svcs = [s for s in self.due() if self.board.claim(s, now)]
        for svc in svcs:
            self.coverage.extend(svc, now)
        return svcs

    # --- the loop

    async def _ping(self, svc: str) -> None:
        t = time.monotonic()
        try:
            await asyncio.to_thread(self.pings[svc])
        except Exception as e:
            log.warning("warm ping %s failed: %s", svc, e)
            return
        finally:
            self.board.pinging.discard(svc)
            self.board.last[svc] = self.clock()
        self.coverage.extend(svc, self.clock())  # a cold start ran the container all along
        if svc not in self._ready:
            self._ready.add(svc)
            secs = time.monotonic() - t
            log.info("warm %s ready after %.1fs", svc, secs)
            if self.notify is not None:
                try:
                    await self.notify(svc, secs)
                except Exception as e:  # the session may have closed meanwhile
                    log.debug("warm notify failed: %s", e)

    def run_once(self) -> None:
        if self.active():  # warm thanks to another session or a real call: say so now
            now = self.clock()
            for svc in self.pings:
                if (svc not in self._ready and svc not in self.board.pinging
                        and now - self.board.last.get(svc, -1e9) < PING_EVERY_S):
                    self._ready.add(svc)
                    if self.notify is not None:
                        task = asyncio.create_task(self.notify(svc, 0.0))
                        self._inflight.add(task)
                        task.add_done_callback(self._inflight.discard)
        for svc in self.tick():
            task = asyncio.create_task(self._ping(svc))
            self._inflight.add(task)
            task.add_done_callback(self._inflight.discard)

    def start(self, every_s: float = 5.0) -> None:
        async def loop():
            while True:
                self.run_once()
                await asyncio.sleep(every_s)

        self._task = asyncio.create_task(loop())

    def stop(self) -> None:
        if self._task:
            self._task.cancel()
        for t in list(self._inflight):
            t.cancel()


def glimmer_ping(settings) -> Callable[[], object] | None:
    """When the brain is self-hosted Muse Glimmer on Modal (switched to by hand with the
    SPARK_* settings), keep it warm the same way: only while a student is active."""
    if settings.spark_model != "muse-glimmer" or not settings.baseten_key:
        return None
    import httpx

    url = settings.spark_base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {settings.baseten_key}"}
    return lambda: httpx.get(url, headers=headers, timeout=300).raise_for_status()


def tracker_ping() -> Callable[[], object] | None:
    """Blocking ping for the SAM 3.1 tracking app (its methods and its socket share containers)."""
    try:
        import modal

        return modal.Cls.from_name("muna-lens-sam31", "Tracker")().ping.remote
    except Exception as e:
        log.warning("cannot reach the tracker for warming: %s", e)
        return None
