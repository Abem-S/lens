"""A1 Spark client: Muse Spark Open on Baseten through the OpenAI-compatible API.

Text, image and audio inputs; streaming; reasoning effort per call; retries;
token and cost logging per call.
"""

from __future__ import annotations

import base64
import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx
from openai import APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, OpenAI, RateLimitError

from lens.config import Settings

log = logging.getLogger("lens.spark")

Effort = Literal["minimal", "low", "medium", "high"]
Content = str | list[dict[str, Any]]

# Reasoning tokens count against max_tokens; too small a budget returns None.
REPLY_MAX_TOKENS = 1500
PAGE_MAX_TOKENS = 6000
_RETRYABLE = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError, httpx.RemoteProtocolError,
              httpx.ReadError)


def text_part(text: str) -> dict:
    return {"type": "text", "text": text}


def image_part(data: bytes, mime: str = "image/jpeg") -> dict:
    url = f"data:{mime};base64,{base64.b64encode(data).decode()}"
    return {"type": "image_url", "image_url": {"url": url}}


def audio_part(wav: bytes) -> dict:
    url = f"data:audio/wav;base64,{base64.b64encode(wav).decode()}"
    return {"type": "audio_url", "audio_url": {"url": url}}


@dataclass
class CallLog:
    purpose: str
    effort: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    cost_usd: float | None = None


@dataclass
class Reply:
    text: str
    log: CallLog


class EmptyReply(RuntimeError):
    """The model spent its whole token budget reasoning and returned no text."""


class SparkStalled(TimeoutError):
    """No copy of a hedged call answered within its deadline (the endpoint is queueing)."""


# A stream that sends no byte for this long is stalled (queued); the attempt is dropped
# and retried. Reasoning streams keep bytes flowing, so a live call never waits this long.
READ_TIMEOUT_S = 75.0
# A hedged call (image reads) gives up after this long in total, so nothing waits forever.
HEDGE_DEADLINE_S = 150.0


@dataclass
class CostTally:
    """Spend of one tutor session: Spark tokens and cost, plus GPU seconds kept warm."""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    spark_usd: float = 0.0
    unknown_cost: int = 0  # calls with no price set
    gpu_s: dict = field(default_factory=dict)  # e.g. {"l40s": 120.0, "l4": 300.0}

    def add(self, entry: CallLog) -> None:
        self.calls += 1
        self.prompt_tokens += entry.prompt_tokens
        self.completion_tokens += entry.completion_tokens
        if entry.cost_usd is None:
            self.unknown_cost += 1
        else:
            self.spark_usd += entry.cost_usd

    def summary(self, seconds: float, gpu_rates: dict) -> dict:
        gpu_usd = sum(s * gpu_rates.get(k, 0.0) / 3600 for k, s in self.gpu_s.items())
        total = self.spark_usd + gpu_usd
        return {"minutes": round(seconds / 60, 1), "spark_calls": self.calls, "tokens_in": self.prompt_tokens,
                "tokens_out": self.completion_tokens, "spark_usd": round(self.spark_usd, 4),
                "gpu_s": {k: round(v) for k, v in self.gpu_s.items()}, "gpu_usd": round(gpu_usd, 4),
                "total_usd": round(total, 4), "usd_per_hour": round(total * 3600 / seconds, 3) if seconds >= 60 else None,
                "unknown_cost_calls": self.unknown_cost}


@dataclass
class SparkClient:
    api_key: str
    base_url: str = "https://inference.baseten.co/v1"
    model: str = "meta-models/muse-spark-open-1026"
    price_in: float | None = None
    price_out: float | None = None
    retries: int = 3
    http_client: httpx.Client | None = None
    calls: list[CallLog] = field(default_factory=list)
    # Set to False after the server rejects `reasoning_effort` (some self-hosted
    # OpenAI-compatible servers do); later calls then leave it out.
    send_effort: bool = True
    # Shared daily cap (lens.limits.DailyBudget): asked before every call, told every cost.
    budget: Any = None
    # This session's spend (set on a per-session copy: `for_session`).
    tally: CostTally | None = None
    # Called (with the purpose) when a hedged call is slow, so the student can be told.
    on_slow: Any = None
    hedge_deadline: float = HEDGE_DEADLINE_S

    def __post_init__(self) -> None:
        # We retry ourselves so that every attempt is logged.
        timeout = httpx.Timeout(connect=15.0, read=READ_TIMEOUT_S, write=60.0, pool=15.0)
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0,
                              http_client=self.http_client, timeout=timeout)

    def for_session(self, tally: CostTally, on_slow=None) -> SparkClient:
        """A copy for one tutor session: same connection pool and daily budget, its own
        call log, cost tally and "this is slow" callback."""
        import copy

        c = copy.copy(self)
        c.calls, c.tally, c.on_slow = [], tally, on_slow
        return c

    def _admit(self) -> None:
        if self.budget is not None and not self.budget.allow():
            from lens.limits import BudgetExceeded

            raise BudgetExceeded()

    @classmethod
    def from_settings(cls, s: Settings) -> SparkClient:
        if not s.baseten_key:
            raise RuntimeError("BASETEN (or SPARK_API_KEY) is not set; add it to .env")
        return cls(api_key=s.baseten_key, base_url=s.spark_base_url, model=s.spark_model,
                   price_in=s.spark_price_in, price_out=s.spark_price_out)

    # ---- cost ----

    def _cost(self, p: int, c: int) -> float | None:
        if self.price_in is None or self.price_out is None:
            return None
        return (p * self.price_in + c * self.price_out) / 1_000_000

    def _record(self, entry: CallLog) -> None:
        entry.cost_usd = self._cost(entry.prompt_tokens, entry.completion_tokens)
        self.calls.append(entry)
        if self.budget is not None:
            self.budget.record(entry.cost_usd)
        if self.tally is not None:
            self.tally.add(entry)
        cost = f"${entry.cost_usd:.5f}" if entry.cost_usd is not None else "cost unknown"
        log.info("spark %s effort=%s in=%d out=%d %.2fs %s", entry.purpose, entry.effort,
                 entry.prompt_tokens, entry.completion_tokens, entry.seconds, cost)

    @property
    def total_cost(self) -> float | None:
        costs = [c.cost_usd for c in self.calls]
        return None if any(c is None for c in costs) else sum(costs)  # type: ignore[arg-type]

    # ---- calls ----

    def _kwargs(self, messages, effort, max_tokens, json_mode) -> dict:
        kw: dict[str, Any] = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if effort and self.send_effort:
            kw["reasoning_effort"] = effort
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        return kw

    def _live(self, kw: dict) -> dict:
        return kw if self.send_effort else {k: v for k, v in kw.items() if k != "reasoning_effort"}

    def _with_retries(self, fn):
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
                return fn()
            except BadRequestError as exc:
                if not (self.send_effort and "reasoning_effort" in str(exc)):
                    raise
                log.warning("spark: server rejects reasoning_effort; sending calls without it")
                self.send_effort = False
                return fn()
            except _RETRYABLE as exc:
                if attempt == self.retries:
                    raise
                log.warning("spark retry %d after %s", attempt + 1, type(exc).__name__)
                time.sleep(delay)
                delay *= 2

    def chat(self, messages: list[dict], *, effort: Effort | None = None, max_tokens: int = REPLY_MAX_TOKENS,
             purpose: str = "chat", json_mode: bool = False, hedge_after: float | None = None) -> Reply:
        """One reply. Streamed under the hood: a reasoning model can think for a
        minute before its first word, and a silent connection that long gets cut
        by proxies. Streaming keeps bytes flowing; we collect the text here.

        hedge_after: if no reply after this many seconds, send the same request
        again (up to twice) and take whichever finishes first. Image calls on the
        shared endpoint sometimes queue for a minute; a fresh copy usually does not."""
        self._admit()
        start = time.monotonic()
        kw = self._kwargs(messages, effort, max_tokens, json_mode)
        kw.update(stream=True, stream_options={"include_usage": True})

        def call():
            parts: list[str] = []
            usage = None
            first = None
            for chunk in self._client.chat.completions.create(**self._live(kw)):
                if chunk.usage:
                    usage = chunk.usage
                for choice in chunk.choices or []:
                    if choice.delta and choice.delta.content:
                        first = first or time.monotonic()
                        parts.append(choice.delta.content)
            return "".join(parts), usage, first

        if hedge_after:
            text, usage, first, extra = self._hedged(call, hedge_after, purpose, effort, start)
        else:
            (text, usage, first), extra = self._with_retries(call), []
        entry = CallLog(purpose, effort, seconds=time.monotonic() - start)
        if usage:
            entry.prompt_tokens, entry.completion_tokens = usage.prompt_tokens, usage.completion_tokens
        self._record(entry)
        for e in extra:  # the losing copies of a hedged call still cost money
            self._record(e)
        if first:
            log.debug("spark %s first text after %.2fs", purpose, first - start)
        if not text:
            raise EmptyReply(f"no text within max_tokens={max_tokens}; raise it")
        return Reply(text, entry)

    def _hedged(self, call, hedge_after: float, purpose: str, effort, start: float):
        import concurrent.futures as cf

        pool = cf.ThreadPoolExecutor(max_workers=3)
        futures = [pool.submit(self._with_retries, call)]
        pending = set(futures)
        extra: list[CallLog] = []
        winner = None
        deadline = start + self.hedge_deadline
        try:
            while winner is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    log.warning("spark %s: no copy answered in %.0fs; giving up", purpose, self.hedge_deadline)
                    raise SparkStalled(f"{purpose}: no reply in {self.hedge_deadline:.0f}s")
                done, pending = cf.wait(pending, timeout=min(hedge_after, left) if len(futures) < 3 else left,
                                        return_when=cf.FIRST_COMPLETED)
                winner = next((f for f in done if f.exception() is None), None)
                if winner is not None:
                    break
                if not done and time.monotonic() >= deadline:
                    continue
                if len(futures) < 3:
                    if not done:
                        log.info("spark %s: no reply after %.0fs, sending a hedge copy", purpose,
                                 time.monotonic() - start)
                        if len(futures) == 1 and self.on_slow is not None:
                            try:
                                self.on_slow(purpose)
                            except Exception:
                                log.exception("on_slow callback failed")
                    f = pool.submit(self._with_retries, call)
                    futures.append(f)
                    pending.add(f)
                elif not pending:
                    raise done.pop().exception()
        finally:
            pool.shutdown(wait=False, cancel_futures=False)

        def account(f):
            if f is winner or f.exception() is not None:
                return
            _, usage, _ = f.result()
            if usage:
                self._record(CallLog(f"{purpose}_hedge", effort, usage.prompt_tokens, usage.completion_tokens))

        for f in futures:
            f.add_done_callback(account)
        text, usage, first = winner.result()
        return text, usage, first, extra

    def stream(self, messages: list[dict], *, effort: Effort | None = None, max_tokens: int = REPLY_MAX_TOKENS,
               purpose: str = "stream") -> Iterator[str]:
        """Yield text deltas as they arrive; logs usage when the stream ends."""
        self._admit()
        start = time.monotonic()
        kw = self._kwargs(messages, effort, max_tokens, False)
        kw.update(stream=True, stream_options={"include_usage": True})
        chunks = self._with_retries(lambda: self._client.chat.completions.create(**self._live(kw)))
        entry = CallLog(purpose, effort)
        try:
            for chunk in chunks:
                if chunk.usage:
                    entry.prompt_tokens = chunk.usage.prompt_tokens
                    entry.completion_tokens = chunk.usage.completion_tokens
                for choice in chunk.choices or []:
                    if choice.delta and choice.delta.content:
                        yield choice.delta.content
        finally:
            entry.seconds = time.monotonic() - start
            self._record(entry)

    # ---- conveniences ----

    def ask(self, prompt: Content, *, system: str | None = None, **kw) -> Reply:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return self.chat(messages, **kw)
