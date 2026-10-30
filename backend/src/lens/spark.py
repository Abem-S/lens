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
from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

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

    def __post_init__(self) -> None:
        # We retry ourselves so that every attempt is logged.
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0,
                              http_client=self.http_client, timeout=240)

    @classmethod
    def from_settings(cls, s: Settings) -> SparkClient:
        if not s.baseten_key:
            raise RuntimeError("BASETEN is not set; add it to .env")
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
        if effort:
            kw["reasoning_effort"] = effort
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        return kw

    def _with_retries(self, fn):
        delay = 1.0
        for attempt in range(self.retries + 1):
            try:
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
        start = time.monotonic()
        kw = self._kwargs(messages, effort, max_tokens, json_mode)
        kw.update(stream=True, stream_options={"include_usage": True})

        def call():
            parts: list[str] = []
            usage = None
            first = None
            for chunk in self._client.chat.completions.create(**kw):
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
        try:
            while winner is None:
                done, pending = cf.wait(pending, timeout=hedge_after if len(futures) < 3 else None,
                                        return_when=cf.FIRST_COMPLETED)
                winner = next((f for f in done if f.exception() is None), None)
                if winner is not None:
                    break
                if len(futures) < 3:
                    if not done:
                        log.info("spark %s: no reply after %.0fs, sending a hedge copy", purpose,
                                 time.monotonic() - start)
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
        start = time.monotonic()
        kw = self._kwargs(messages, effort, max_tokens, False)
        kw.update(stream=True, stream_options={"include_usage": True})
        chunks = self._with_retries(lambda: self._client.chat.completions.create(**kw))
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
