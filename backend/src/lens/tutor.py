"""F1 Tutor agent + V2 voice loop (server side).

One TutorSession per Lens screen, over the /ws/tutor/{sid} WebSocket.

Client -> server
  {"type": "hello", "voice": true}                start; the tutor opens the conversation
  {"type": "frame"}            + binary JPEG       latest view of the page (for pointing)
  {"type": "utterance"}        + binary WAV        one thing the student said (VAD-cut in the browser)
  {"type": "check", "reason"}  + binary JPEG       the page changed, or "check my work"
  {"type": "interrupt"}                            the student started talking: stop speaking
Server -> client
  {"type": "state", "state": "listening" | "thinking" | "speaking" | "reading"}
  {"type": "transcript", "text"}                  what Lens heard
  {"type": "say", "text", "utt"}                  a sentence Lens is about to speak (captions)
  {"type": "audio", "utt", "sample_rate"} + binary PCM16 mono
  {"type": "audio_end", "utt"}
  {"type": "result", "page", "finding"}           a page check finished
  {"type": "pointing", "step_id", "point"}        where the student is pointing
  {"type": "timing", ...}                         latency measurements
  {"type": "error", "message"}

Rules carried from AGENTS.md: only verified findings are mentioned (the checker
guarantees that), and nothing spoken may carry the answer: every spoken sentence
passes the same answer-leak check as hints while a finding is open.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

from lens.contract import Finding, Page

log = logging.getLogger("lens.tutor")

OPENING = "Hi, I'm Lens. What are you working on today?"
FIXED = "That line works now. Keep going."
SAFE_REPLY = "Let's look at that line together. What did you do to get there from the line above?"
HISTORY_TURNS = 12
_DEICTIC = re.compile(r"\b(this|that|here|these|those)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")

TUTOR_SYSTEM = """You are Lens, a patient tutor who watches a student solve STEM problems on paper and
talks with them by voice. You never give answers.

How you speak:
- One to three short sentences, in plain spoken English. No LaTeX, no lists, no symbols to read out.
- Ask questions that help the student find things themselves.
- Never state a corrected line, a result, or a number the student has not written.
- Never tell them which operation to do; ask about what they did.
- If you do not know what they are working on, ask, and confirm what you heard instead of assuming.
- If they ask for the answer, kindly decline and ask a guiding question instead.

What you can see on their page right now:
{page}
{finding}{pointing}"""


class Speech(Protocol):
    def transcribe(self, wav: bytes) -> str: ...
    def synthesize(self, text: str) -> Iterator[tuple[bytes, int]]: ...


class Brain(Protocol):
    def stream(self, messages: list[dict], *, effort=None, max_tokens: int = 1500, purpose: str = "") -> Iterator[str]:
        ...


@dataclass
class Services:
    """Everything the tutor calls out to. Real ones in production, fakes in tests."""
    brain: Brain
    speech: Speech | None
    check: Callable[[bytes, str | None], tuple[Page, Finding | None]]  # read + check + hint + outline
    locate_fingertip: Callable[[bytes], tuple[float, float] | None] | None = None


Send = Callable[[dict], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]


def sentences(deltas: Iterator[str], first_min_words: int = 8) -> Iterator[str]:
    """Group streamed text into speakable pieces. The first piece may end at a
    comma once it is long enough, so the first audio starts sooner."""
    buf = ""
    first = True
    for d in deltas:
        buf += d
        while True:
            parts = _SENTENCE_END.split(buf, maxsplit=1)
            if len(parts) == 2:
                yield parts[0].strip()
                buf = parts[1]
                first = False
                continue
            if first and "," in buf and len(buf.split()) >= first_min_words:
                head, _, tail = buf.rpartition(",")
                if head.strip():
                    yield head.strip() + ","
                    buf = tail
                    first = False
                    continue
            break
    if buf.strip():
        yield buf.strip()


def page_summary(page: Page | None) -> str:
    if not page or not page.steps:
        return "(nothing read yet)"
    marks = {"ok": "checked", "error": "HAS THE MISTAKE", "unverifiable": "not checked", "unchecked": "not checked"}
    lines = [f"Problem: {page.problem}"] if page.problem else []
    lines += [f"Line {i + 1}: {s.latex}   [{marks[s.status]}]" for i, s in enumerate(page.steps)]
    return "\n".join(lines)


@dataclass
class TutorSession:
    services: Services
    send: Send
    send_bytes: SendBytes
    voice: bool = True
    page: Page | None = None
    finding: Finding | None = None
    problem: str | None = None  # what the student said they are working on
    history: list[dict] = field(default_factory=list)
    frame: bytes | None = None
    _speaking: asyncio.Task | None = None
    _cancel: threading.Event = field(default_factory=threading.Event)
    _utt: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    # ---------- entry points ----------

    async def start(self) -> None:
        await self._say_now([OPENING], record=True)

    async def on_frame(self, jpeg: bytes) -> None:
        self.frame = jpeg

    async def on_interrupt(self) -> None:
        await self._stop_speaking()
        await self.send({"type": "state", "state": "listening"})

    async def on_utterance(self, wav: bytes, heard_at: float | None = None) -> None:
        heard_at = heard_at or time.monotonic()
        await self._stop_speaking()
        await self.send({"type": "state", "state": "thinking"})
        if self.services.speech is None:
            return
        text = (await asyncio.to_thread(self.services.speech.transcribe, wav)).strip()
        t_asr = time.monotonic()
        await self.send({"type": "transcript", "text": text})
        if not text:
            await self.send({"type": "state", "state": "listening"})
            return
        if self.problem is None and not (self.page and self.page.problem):
            self.problem = text  # their first answer to "what are you working on"; confirmed in conversation
        pointing = await self._pointing(text)
        self.history.append({"role": "user", "content": text})
        self._speaking = asyncio.create_task(self._reply(pointing, heard_at, t_asr))

    async def on_check(self, jpeg: bytes, reason: str = "manual") -> None:
        self.frame = jpeg
        async with self._lock:  # one check at a time
            await self.send({"type": "state", "state": "reading"})
            try:
                page, finding = await asyncio.to_thread(self.services.check, jpeg, self._problem_hint())
            except Exception as exc:
                log.exception("check failed")
                await self.send({"type": "error", "message": f"Could not read the page: {exc}"})
                await self.send({"type": "state", "state": "listening"})
                return
            previous = self.finding
            prev_line = self._line_of(previous)
            self.page, self.finding = page, finding
            await self.send({"type": "result", "page": page.model_dump(),
                             "finding": finding.model_dump() if finding else None})
            if finding and (previous is None or self._line_of(finding) != prev_line):
                await self._say_when_free([finding.hint], record=True)
            elif previous and finding is None and page.steps:
                await self._say_when_free([FIXED], record=True)
            else:
                await self.send({"type": "state", "state": "listening"})

    async def close(self) -> None:
        await self._stop_speaking()

    # ---------- internals ----------

    def _problem_hint(self) -> str | None:
        return (self.page.problem if self.page and self.page.problem else None) or self.problem

    def _line_of(self, finding: Finding | None) -> str | None:
        if finding is None or self.page is None:
            return None
        s = self.page.step(finding.step_id)
        return s.latex if s else finding.step_id

    async def _pointing(self, text: str) -> str:
        if not (_DEICTIC.search(text) and self.frame and self.page and self.services.locate_fingertip):
            return ""
        from lens.pointing import nearest_step

        try:
            tip = await asyncio.to_thread(self.services.locate_fingertip, self.frame)
        except Exception:
            log.exception("pointing failed")
            return ""
        step = nearest_step(tip, self.page.steps) if tip else None
        if not step:
            return ""
        n = self.page.steps.index(step) + 1
        await self.send({"type": "pointing", "step_id": step.id, "point": list(tip)})
        return f"\nThe student is pointing at line {n}: {step.latex}"

    def _messages(self, pointing: str) -> list[dict]:
        finding = ""
        if self.finding and self.page:
            n = next((i + 1 for i, s in enumerate(self.page.steps) if s.id == self.finding.step_id), "?")
            finding = (f"\nVerified mistake on line {n} (for you only, do not reveal the fix): {self.finding.kind}."
                       f"\nYou already asked: {self.finding.hint}")
        if self.problem and not (self.page and self.page.problem):
            finding += f"\nThe student said they are working on: {self.problem}"
        system = TUTOR_SYSTEM.format(page=page_summary(self.page), finding=finding, pointing=pointing)
        return [{"role": "system", "content": system}, *self.history[-HISTORY_TURNS:]]

    def _guard(self) -> Callable[[str], bool]:
        """True if a sentence may be spoken."""
        if not (self.finding and self.page):
            return lambda s: True
        from lens.hints import answer_leak, expected_for

        try:
            expected = expected_for(self.page, self.finding)
        except Exception:
            return lambda s: True
        return lambda s: answer_leak(s, expected) is None

    async def _reply(self, pointing: str, heard_at: float, t_asr: float) -> None:
        cancel = self._cancel = threading.Event()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        messages = self._messages(pointing)

        def produce() -> None:
            try:
                deltas = self.services.brain.stream(messages, effort="minimal", max_tokens=1500, purpose="reply")
                for s in sentences(deltas):
                    if cancel.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, s)
            except Exception:
                log.exception("reply stream failed")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=produce, daemon=True).start()

        async def pieces():
            while (s := await queue.get()) is not None:
                yield s

        await self._speak(pieces(), cancel, record=True, timing={"heard_at": heard_at, "asr_done": t_asr})

    async def _say_now(self, texts: list[str], record: bool) -> None:
        await self._stop_speaking()
        cancel = self._cancel = threading.Event()

        async def pieces():
            for t in texts:
                yield t

        self._speaking = asyncio.create_task(self._speak(pieces(), cancel, record=record))
        await asyncio.sleep(0)

    async def _say_when_free(self, texts: list[str], record: bool) -> None:
        if self._speaking and not self._speaking.done():
            try:
                await self._speaking
            except asyncio.CancelledError:
                pass
        await self._say_now(texts, record)

    async def _speak(self, pieces, cancel: threading.Event, record: bool, timing: dict | None = None) -> None:
        self._utt += 1
        utt = self._utt
        allowed = self._guard()
        spoken: list[str] = []
        first_audio = None
        try:
            await self.send({"type": "state", "state": "speaking"})
            async for text in pieces:
                if cancel.is_set():
                    break
                if not allowed(text):
                    log.warning("blocked a sentence that leaked the answer: %r", text)
                    text = SAFE_REPLY
                    cancel.set()  # say the safe line, then stop
                spoken.append(text)
                await self.send({"type": "say", "text": text, "utt": utt})
                if not (self.voice and self.services.speech):
                    continue
                for pcm, sr in await asyncio.to_thread(self._synth_until_cancel, text, cancel):
                    if first_audio is None:
                        first_audio = time.monotonic()
                        if timing:
                            await self.send({"type": "timing", "asr_s": round(timing["asr_done"] - timing["heard_at"], 3),
                                             "first_audio_s": round(first_audio - timing["heard_at"], 3)})
                            log.info("voice latency: asr %.2fs, first audio %.2fs",
                                     timing["asr_done"] - timing["heard_at"], first_audio - timing["heard_at"])
                    await self.send({"type": "audio", "utt": utt, "sample_rate": sr})
                    await self.send_bytes(pcm)
        finally:
            if record and spoken:
                self.history.append({"role": "assistant", "content": " ".join(spoken)})
            await self.send({"type": "audio_end", "utt": utt})
            if not cancel.is_set() or spoken:
                await self.send({"type": "state", "state": "listening"})

    def _synth_until_cancel(self, text: str, cancel: threading.Event) -> list[tuple[bytes, int]]:
        out = []
        for chunk in self.services.speech.synthesize(text):
            if cancel.is_set():
                break
            out.append(chunk)
        return out

    async def _stop_speaking(self) -> None:
        self._cancel.set()
        task = self._speaking
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                log.exception("speaking task failed while stopping")
        self._speaking = None
