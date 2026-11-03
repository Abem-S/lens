"""Rate limits and spend caps for a public link: every session spends money.

All limits are settings (`LENS_*` in .env.example) and live in memory: the app runs
as one process, so one set of counters covers everyone. A restart resets them.

- concurrent tutor sessions overall, and concurrent live-tracking sessions (GPU cost)
- session starts per client IP per hour (tutor sockets and phone pairings)
- page checks and spoken turns per session per minute
- a global daily cap on Spark calls and on estimated Spark spend (UTC day)

Each `allow`-style call answers True/False; the caller shows `MESSAGES[...]` to the
student. Clocks are injectable for tests.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass

MESSAGES = {
    "busy": "Lens is helping as many students as it can right now. Please try again in a few minutes.",
    "ip": "You have started a lot of sessions in the last hour. Please wait a little and try again.",
    "checks": "You're writing fast! Lens will catch up when you finish your next step.",
    "utterances": "Let's slow down a little. Give me a moment, then ask again.",
    "daily": "Lens has reached today's limit and is resting until tomorrow. Thank you for testing it!",
    "tracking": "Live tracking is busy right now, so Lens will check your page when it goes still.",
}


def _int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


def _flt(name: str, default: float) -> float:
    v = os.environ.get(name)
    return float(v) if v not in (None, "") else default


@dataclass(frozen=True)
class Limits:
    max_sessions: int = 8  # tutor sessions open at once (0 = no limit)
    max_tracking: int = 4  # live SAM 3.1 tracking sessions at once (one L40S runs ~2-4)
    sessions_per_ip_hour: int = 40  # session starts (tutor sockets + pairings) per IP per hour
    checks_per_min: int = 6  # page checks per session per minute
    utterances_per_min: int = 20  # spoken turns per session per minute
    daily_spark_calls: int = 4000  # Spark calls per UTC day, all sessions
    daily_spend_usd: float = 15.0  # estimated Spark spend per UTC day, all sessions
    idle_min: float = 5.0  # browser: no camera movement / speech this long -> pause GPU work
    track_max_min: float = 45.0  # server: one tracking socket at most this long (client reconnects)

    @classmethod
    def from_env(cls) -> Limits:
        d = cls()
        return cls(
            max_sessions=_int("LENS_MAX_SESSIONS", d.max_sessions),
            max_tracking=_int("LENS_MAX_TRACKING", d.max_tracking),
            sessions_per_ip_hour=_int("LENS_SESSIONS_PER_IP_HOUR", d.sessions_per_ip_hour),
            checks_per_min=_int("LENS_CHECKS_PER_MIN", d.checks_per_min),
            utterances_per_min=_int("LENS_UTTERANCES_PER_MIN", d.utterances_per_min),
            daily_spark_calls=_int("LENS_DAILY_SPARK_CALLS", d.daily_spark_calls),
            daily_spend_usd=_flt("LENS_DAILY_SPEND_USD", d.daily_spend_usd),
            idle_min=_flt("LENS_IDLE_MIN", d.idle_min),
            track_max_min=_flt("LENS_TRACK_MAX_MIN", d.track_max_min),
        )


class Window:
    """At most `limit` events per `seconds`, sliding (limit 0 = unlimited)."""

    def __init__(self, limit: int, seconds: float, clock: Callable[[], float] = time.monotonic):
        self.limit, self.seconds, self.clock = limit, seconds, clock
        self.events: deque[float] = deque()

    def allow(self) -> bool:
        if self.limit <= 0:
            return True
        now = self.clock()
        while self.events and now - self.events[0] >= self.seconds:
            self.events.popleft()
        if len(self.events) >= self.limit:
            return False
        self.events.append(now)
        return True


class Slots:
    """A counter of things open at once (sessions), with a ceiling (0 = unlimited)."""

    def __init__(self, limit: int):
        self.limit, self.open = limit, 0
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            if self.limit > 0 and self.open >= self.limit:
                return False
            self.open += 1
            return True

    def give(self) -> None:
        with self._lock:
            self.open = max(0, self.open - 1)


class DailyBudget:
    """Spark calls and estimated spend per UTC day, shared by every session.

    `allow()` is asked before a call; `record(cost)` after it. Calls with an unknown
    cost count towards the call cap only."""

    def __init__(self, max_calls: int, max_usd: float, day: Callable[[], str] | None = None):
        self.max_calls, self.max_usd = max_calls, max_usd
        self.day = day or (lambda: time.strftime("%Y-%m-%d", time.gmtime()))
        self._lock = threading.Lock()
        self._today = ""
        self.calls = 0
        self.usd = 0.0

    def _roll(self) -> None:
        d = self.day()
        if d != self._today:
            self._today, self.calls, self.usd = d, 0, 0.0

    def allow(self) -> bool:
        with self._lock:
            self._roll()
            if self.max_calls > 0 and self.calls >= self.max_calls:
                return False
            return not (self.max_usd > 0 and self.usd >= self.max_usd)

    def record(self, cost_usd: float | None) -> None:
        with self._lock:
            self._roll()
            self.calls += 1
            self.usd += cost_usd or 0.0


class BudgetExceeded(RuntimeError):
    """Today's Spark cap is used up."""

    def __init__(self) -> None:
        super().__init__(MESSAGES["daily"])


def client_ip(headers: dict[str, str] | list[tuple[bytes, bytes]], peer: str | None) -> str:
    """The student's IP: the last public address in X-Forwarded-For (added by the proxy
    in front of the app: Modal's, or reverse proxy), else the socket peer."""
    if isinstance(headers, list):
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in headers}
    xff = headers.get("x-forwarded-for", "")
    import ipaddress

    for part in reversed([p.strip() for p in xff.split(",") if p.strip()]):
        try:
            if ipaddress.ip_address(part).is_global:
                return part
        except ValueError:
            continue
    return peer or "unknown"


class Limiter:
    """Every limit for one app process."""

    def __init__(self, limits: Limits, clock: Callable[[], float] = time.monotonic,
                 day: Callable[[], str] | None = None):
        self.limits, self.clock = limits, clock
        self.sessions = Slots(limits.max_sessions)
        self.tracking = Slots(limits.max_tracking)
        self.budget = DailyBudget(limits.daily_spark_calls, limits.daily_spend_usd, day)
        self._ips: dict[str, Window] = defaultdict(lambda: Window(limits.sessions_per_ip_hour, 3600, clock))

    def session_start(self, ip: str) -> bool:
        """A new session from `ip` (counted even if refused later for being busy)."""
        if len(self._ips) > 10_000:  # forget quiet addresses so the table stays small
            for k in [k for k, w in self._ips.items() if not w.events or self.clock() - w.events[-1] > 3600]:
                del self._ips[k]
        return self._ips[ip].allow()

    def checks(self) -> Window:
        return Window(self.limits.checks_per_min, 60, self.clock)

    def utterances(self) -> Window:
        return Window(self.limits.utterances_per_min, 60, self.clock)
