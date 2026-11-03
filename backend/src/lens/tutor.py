"""F1 Tutor agent + V2 voice loop (server side).

One TutorSession per Lens screen, over the /ws/tutor/{sid} WebSocket.

Client -> server
  {"type": "hello", "voice": true, "track"?}      start; the tutor opens the conversation. `track`: the live
                                                   tracking session id (pointer, live state, finished lines)
  {"type": "frame"}            + binary JPEG       latest view of the page (for pointing)
  {"type": "speech_start", "continues"?: bool}    the student may have started talking (Lens keeps speaking:
                                                   it stops on "interrupt" or a transcript with words).
                                                   `continues`: within a moment of their last turn, so it is
                                                   the same sentence after a pause: the two halves are heard
                                                   as one and the reply to the first half is dropped, unless
                                                   the second half adds no words (noise)
  {"type": "speech"}           + binary PCM16      audio while they talk (16 kHz mono, ~200 ms per chunk), so
                                                   nothing is left to upload when they stop
  {"type": "speech_pause"}                         ~200 ms of quiet: the server may transcribe what it has
  {"type": "speech_resume"}                        ...they went on talking: that early transcript is void
  {"type": "speech_end"}                           the turn is over (the browser's VAD waited out the silence)
  {"type": "speech_cancel"}                        it was a cough or a click: drop it
  {"type": "utterance"}        + binary WAV        one whole utterance at once (older clients, scripts)
  {"type": "activity", "phase"}                    live state from the browser's pixel detector, when live
                                                   tracking is off: "writing" | "paused" | "away"
  {"type": "check", "reason", "box"?} + JPEG       the page changed, or a line was finished ("line_done";
                                                   `box`: where, else the tracker's last line box)
  {"type": "interrupt"}                            the student is really talking over Lens: stop speaking
  {"type": "client_timing", ...}                   what the browser measured for a turn (logged)
  Accessibility (TutorSession.on_message):
  {"type": "text", "text"}                         a typed turn: answered exactly like a spoken one (same
                                                   pointing, page context, "look" frame and answer guard), no ASR.
                                                   Send a "frame" first if the camera is on
  {"type": "settings", "aloud"?, "speed"?}         `aloud`: synthesise speech for what Lens says (false = captions
                                                   only; the replies still come as "say" text). `speed`: Kokoro
                                                   speaking speed, 0.6-1.5 (1 = normal)
  {"type": "read_page"}                            read the steps already read back aloud, in words; a flagged
                                                   step is named as the outlined one, nothing more. Asking in
                                                   words ("read my page aloud") does the same
  {"type": "end"}                                  the student ends the session: reply with "summary"
Server -> client
  {"type": "state", "state": "listening" | "thinking" | "speaking" | "reading"}
  {"type": "transcript", "text", "typed"?}        what Lens heard (`typed`: the echo of a typed turn); "" for
                                                   a turn that was only noise (Lens goes on speaking)
  {"type": "say", "text", "utt"}                  a sentence Lens is about to speak (captions)
  {"type": "audio", "utt", "sample_rate"} + binary PCM16 mono (at most ~0.4 s per message)
  {"type": "audio_end", "utt"}
  {"type": "result", "page", "finding", "cleared"?}  a page check finished. `cleared`: not a check; the student
                                                   started a different problem, so the steps list starts afresh
  {"type": "steps", "page"}                        the page as read, before it is verified (all "unchecked"), so
                                                   the steps show while the check goes on; no finding yet
  {"type": "pointing", "step_id", "point"}        where the student is pointing
  {"type": "timing", ...}                         latency by stage for a spoken reply (seconds after the
                                                   turn ended): asr_s, llm_first_s, piece_s, first_audio_s
  {"type": "check_timing", "path", ...}           how a check went: "line" (fast) or "page", stage times
  {"type": "summary", "minutes", "problem", "steps_read", "steps_ok", "checks", "mistakes": [{"step", "spoken",
   "kind", "status": "fixed" | "changed" | "open"}]}  the session in plain facts, for the end screen. Kept in memory
                                                   for this socket only; nothing is stored
  {"type": "error", "message"}

Besides answering, the tutor speaks up by itself, sparingly: a hint for a verified
mistake, "that works now" after a fix, a nudge when the student has been stuck for a
while, a word of encouragement now and then after a hard-won step, a remark when a
step is crossed out, and a look at the page when something changed that is not a
step (a diagram, words). Each has its own rate limit, and none talks over the student.

Rules carried from AGENTS.md: only verified findings are mentioned (the checker
guarantees that), and nothing spoken may carry the answer: every spoken sentence
passes the same answer-leak check as hints while a finding is open. Typed turns take
the same path; "read my page aloud" passes the same check, with the student's own steps
counted as already written (quoting them is not telling).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
import re
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from typing import Protocol

from lens.attempts import moved_on
from lens.contract import Finding, Page, Step
from lens.hints import say_step
from lens.mathparse import is_connector, same_math
from lens.speakmath import count_words, plain_text, speak_latex

log = logging.getLogger("lens.tutor")

OPENING = "Hi, I'm Lens. What are you working on today?"
FIXED = "That works now. Keep going."
SAFE_REPLY = "Let's look at that step together. How did you get there from the one before it?"
HISTORY_TURNS = 20  # the problem itself is kept in the notes (lens.notes), not only in the talk
_DEICTIC = re.compile(r"\b(this|that|here|these|those)\b", re.IGNORECASE)
# The student says they are starting a different problem: the steps list starts afresh.
# (Another method for the same problem is not this: lens.attempts handles a restart.)
_NEW_PROBLEM = re.compile(
    r"\b(?:(?:new|next|another|different|other)\s+(?:problem|question|exercise|example)"
    r"|(?:moving on|move on|going on|on)\s+to\s+(?:the\s+next|(?:problem|question|exercise|number)\s*\d+)"
    r"|(?:let'?s|let us|i'?ll|i will|i'?m going to|now)\s+(?:do|try|start)\s+(?:problem|question|exercise|number)\s*\d+"
    r"|(?:starting|start)\s+(?:problem|question|exercise|number)\s*\d+)\b",
    re.IGNORECASE)
# Questions that ask Lens to look at the camera: the reply gets the current frame attached.
# Only explicit asks; "is this right?" is answered from the page Lens already read.
_VISUAL = re.compile(
    r"\b(can you see|could you see|do you see|you see (it|this|that|my|me|what)|look(ing)? at|take a look|"
    r"have a look|what do you see|see (it|this|that|my|what|me|how)|show(ing|n)? (you|me)|i'?m showing|"
    r"i'?m holding|i am holding|hold(ing)? (it|this|up)|watch(ing)?|look here|"
    r"what (did|have) i (write|written|wrote|draw|drawn|drew)|what i (wrote|drew|have written)|"
    r"what (am i|i'?m) (doing|holding|showing|pointing)|what (number|numbers|shape|colou?r|object|thing)s?|"
    r"how many|count(ing)?|(my|these|this|those) (finger|fingers|hand|hands)|finger|fingers|"
    r"the camera|my (drawing|diagram|graph|sketch|figure|notebook)|this (diagram|graph|drawing|picture))\b",
    re.IGNORECASE)  # not "is this step right?": that is answered from the page already read, without the wait
# "Watch this": the student is about to show something, so Lens keeps looking a moment longer.
_WATCH = re.compile(r"\b(watch|i'?ll show|i will show|let me show|i'?m going to show|keep looking|look now)\b",
                    re.IGNORECASE)
# Said only when an image reply is slow to start, and not again for a while.
LOOKING = ("Let me look.", "One moment.", "Looking now.")
LOOK_FILLER_AFTER_S = 1.2
LOOK_FILLER_COOLDOWN_S = 60.0
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")
# Where the first spoken piece may end early: a comma, semicolon, colon or dash followed by a space
# (so "3,000" is never cut), once it has a couple of words.
_CLAUSE_END = re.compile(r"(?:[,;:]|\s[—–-])(?=\s)")
FIRST_PIECE_MIN_WORDS = 2  # "Good question," is a natural piece; "Okay," alone is too short

# Accessibility: typed turns, voice settings, reading the page aloud, the end-of-session summary.
MAX_TYPED_CHARS = 500
SPEED_RANGE = (0.6, 1.5)
READ_MAX_STEPS = 30
# "Read my page aloud", "can you read back what I wrote", "read me my steps": a request to
# read, about the student's work, asked as a request (aloud / back / to me, or starting with
# "read"), so "did you read my work?" is answered as a question instead.
_READ_PAGE = re.compile(
    r"\bread\b[^.?!]{0,30}?\b(my|the|what i('ve| have)?|all( of)?)\b[^.?!]{0,20}?"
    r"\b(page|steps?|work(ing)?|wrote|written|answers?)\b", re.IGNORECASE)
_READ_REQUEST = re.compile(
    r"\b(aloud|out loud|read (it |them |that )?(back|out)|read (it |them )?to me|read me)\b|"
    r"^\W*(ok(ay)?|so|lens|hey lens|please|now)?\W*(can|could|would|will) you( please)? read\b|"
    r"^\W*(ok(ay)?|so|lens|hey lens|now)?\W*(please )?read\b", re.IGNORECASE)
NOTHING_TO_READ = "I haven't read any steps yet. Write your working, and I'll read each step as you finish it."
SAFE_KIND = "a mistake Lens asked you about"

# Streaming speech.
ASR_RATE = 16000
MAX_SPEECH_BYTES = 30 * ASR_RATE * 2  # the ASR model refuses clips over 40 s
# A continuation must start this soon after the turn it continues was heard (the browser
# sends one within 1.2 s of the end of speech; ASR adds a little).
MERGE_S = 3.0
_WORDS = re.compile(r"\w")


ECHO_S = 20.0  # Lens's own words can come back through the mic this long after they were sent


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))
AUDIO_SLICE_S = 0.4  # tutor audio goes down in slices this long, so playback starts on the first one

# Speaking up unasked. All times in seconds.
STUCK_AFTER_S = 20.0  # no new ink or talk this long while the student is at the page -> a gentle check-in
NUDGE_AGAIN_S = 45.0  # still stuck this much longer after the check-in -> one more, a little more concrete
MAX_NUDGES = 2  # per pause
HINT_GRACE_S = 30.0  # no check-in this soon after a hint
PRAISE_COOLDOWN_S = 12.0  # a short "good" for a correct new step, at most this often
DIRECTION_COOLDOWN_S = 120.0  # "where is this heading?" when the notes say the approach cannot reach the goal
# Looking: the browser sends a camera picture about every 0.5 s while the student talks and for a
# moment after; a turn about what the camera shows gets a run of them, in order.
LOOK_FRAMES = 6
LOOK_KEEP_S = 15.0  # pictures kept this long
LOOK_AFTER_S = 0.6  # after the turn ends, wait this long for one more picture (the hand often still moves)
WATCH_AFTER_S = 2.5  # ..."watch this": keep looking this long
ENCOURAGE_EFFORT_S = 25.0  # a correct step that took at least this long since the last one "took effort"
ENCOURAGE_COOLDOWN_S = 150.0
SCRATCH_COOLDOWN_S = 60.0
LOOK_COOLDOWN_S = 120.0  # a look at a page change that is not a step
TICK_S = 1.0
LIVE_FRESH_S = 5.0  # browser-reported activity older than this is not trusted

# When the model endpoint is slow (image reads on the shared endpoint sometimes queue
# for a minute or more): say so, keep the session responsive, and give up at a cap.
SLOW_NOTE = "This is taking longer than usual…"
SLOW_SAY = "Sorry, I'm a little slow right now. Keep going, I'll catch up."
SLOW_SAY_COOLDOWN_S = 180.0
CHECK_SLOW_S = 20.0  # a check with no result this long gets the note
CHECK_CAP_S = 170.0  # ...and is dropped after this long (the next step is read as usual)
REPLY_SLOW_S = 6.0  # a spoken reply with no words this long gets "One moment."
REPLY_CAP_S = 30.0  # ...and is replaced by REPLY_FAILED after this long
REPLY_SLOW_SAY = "One moment."
REPLY_FAILED = "Sorry, I lost my train of thought. Could you say that again?"


def check_error_message(exc: BaseException) -> str:
    """What the student sees when a check fails: friendly, never a stack trace."""
    from lens.limits import BudgetExceeded

    if isinstance(exc, BudgetExceeded):
        return str(exc)
    if isinstance(exc, TimeoutError):
        return "Reading your page is taking too long right now. Lens will look again when you write your next step."
    return "Lens couldn't read your page just now. It will try again when you write your next step."

PRAISE = (
    "Good.",
    "Yes, keep going.",
    "That's right.",
    "Good, that step works.",
    "Nice. Keep going.",
    "Right.",
)
ENCOURAGEMENTS = (
    "Nice, that step holds up.",
    "Good work, that one's right.",
    "That took some thought, and it's right.",
    "Well done. Keep going.",
    "Yes, that works.",
)
SCRATCHED = (
    "I see you crossed that out. Take it from there.",
    "Starting that step again? Good idea.",
    "Crossed out, okay. Have another go.",
)

NUDGE_NOTE = """(No words from the student. They have been at the page for {secs} seconds without writing
anything new. Say ONE short, gentle sentence or question to help them get going: ask what they are
thinking about, or which part feels tricky, or point them back at their own last step. Do not give the
answer, the next step, or which operation to do. Do not mention the camera or how long it has been.)"""
NUDGE_AGAIN_NOTE = """(No words from the student. They are still stuck: {secs} seconds at the page with nothing
new since your last check-in. Using your teaching plan, ask ONE short, concrete question that points them at
the next stage (what quantity they need, what the problem gives them, which part of their last step to work
on). Do not give the answer, a result, or the step itself. Do not mention the camera or how long it has been.)"""
DIRECTION_NOTE = """(No words from the student. Their steps so far are correct, but your notes say their approach
may not lead to what is asked: {why} Ask ONE short question that makes them check where this is heading
(for example what they are trying to find, or how this step gets them closer to it). Do not say it is a
mistake, do not name the method to use, and give no result.)"""
LOOK_NOTE = """(No words from the student. Something new appeared on their page, but it is not a written
maths step: perhaps a diagram, a graph, a table or words. The attached picture is what the camera sees
now. Say ONE short, useful thing about what they drew or wrote, or ask one question about it. If you
cannot make it out, ask them what it is. Do not give any answer or result.)"""

TUTOR_SYSTEM = """You are Lens, a warm, clear tutor who watches a student solve STEM problems on paper and
talks with them by voice. You guide them to the answer; you never hand it over.

Be direct about right and wrong. Never say you cannot tell them whether they are right.
- A step marked "checked: correct" below IS right. When they ask, say so plainly ("Yes, that step is
  right.") and, if useful, say briefly why their move works.
- The step marked "HAS THE MISTAKE" is wrong. Say so plainly ("Not quite, there's a slip in that step.")
  and say where in the step to look, then ask one focused question. Never give the fix.
- A step marked "could not be fully checked" (words, a diagram, a hard-to-read step): say in a few words
  that you could not check that one fully, and ask them to talk you through how they got it. Then judge
  their explanation: if their reasoning is sound, say so.
- A step marked "not checked yet": say you have not checked it yet and that you will once it is finished
  and the hand is clear; do not guess either way.
- When they fix a mistake, tell them clearly that it is right now.

Guide clearly, step by step (see "Guidance" below when there is a mistake):
- Level 1: name the step and ask one question about the exact move they made there.
- Level 2: point at the exact part of the step that is off (the sign, the right side, the exponent, the
  units) and ask what should happen to it.
- Level 3: remind them of the general rule or idea that applies, as a fact about maths or science, then
  ask them to apply it to their step. Still never state their corrected step or its result.
- If they explain correct reasoning, say it is correct and tell them to go on.
- Explaining a general concept, definition or rule when asked is good teaching; solving their step is not.

How you speak:
- One to three short sentences, in plain spoken English. No LaTeX, no lists, no symbols to read out.
- Never state a corrected step, a result, or a number the student has not written.
- Talk about their work by what they wrote, said in words ("where you moved the five across",
  "your second step"), never as "line 2" or "that line". Say "step", not "line".
- If you do not know what they are working on, ask, and confirm what you heard instead of assuming.
- If they ask for the answer, kindly decline and give the next level of guidance instead.
- Fit the moment (see "Right now"): while they are writing, keep it very short and let them work.
  Never mention the camera tracking, timers or how long they took.

When camera pictures are attached to the student's message, they are what the camera saw while they
spoke, in time order (the last is the most recent): the student, their hands, their page, or something
they are holding up. Use them: if they showed several things one after another (fingers, cards, objects),
say what you saw in order. Never say you cannot see when pictures are attached. If the pictures do not
show what they mean, say what you do see and ask them to show it again, holding it still.

The student's words come from speech recognition and may be garbled (accents; maths words misheard, such
as "ex" for x, "too" for 2). Work out what they most likely meant from the page, your notes and the
conversation. Only if you really cannot, ask briefly to hear it again.

If your notes say their approach cannot reach what is asked, do not call it a mistake: ask a question
that makes them check where it is heading.

Your private notes on what they are working on (never read them out; use them to teach):
{notes}

What has been read from their page so far:
{page}
{finding}{pointing}
Right now: {live}"""


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
    # read + check + hint + outline: (frame, problem, prior=lens.pipeline.Prior, on_read=callback) -> (page, finding)
    check: Callable[..., tuple[Page, Finding | None]]
    locate_fingertip: Callable[[bytes], tuple[float, float] | None] | None = None
    # Fast path for a just-finished line: (frame, last page, open finding, line box, problem, timings)
    # -> (page, finding), or None to fall back to `check` (lens.pipeline.check_line).
    check_line: Callable[..., tuple[Page, Finding | None] | None] | None = None
    # The box of the line live tracking last saw finished, if recent (lens.tracking.TrackHub).
    line_box: Callable[[], tuple[float, float, float, float] | None] | None = None
    # Live state from tracking, if live: {"phase", "present", "writing_ago_s"} (lens.tracking.TrackHub.live).
    live: Callable[[], dict | None] | None = None
    # Keep the outlined step on the live page: (check frame, step box on it), or (frame, None)
    # when nothing is outlined (lens.tracking.TrackHub.follow).
    follow: Callable[[bytes, tuple[float, float, float, float] | None], None] | None = None
    # Rewrite the private problem notes: (notes, recent conversation, page summary, event) -> notes
    # (lens.notes.update; a worker thread).
    notes: Callable[[dict, list[dict], str, str], dict] | None = None


Send = Callable[[dict], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]


def sentences(deltas: Iterator[str], first_min_words: int = FIRST_PIECE_MIN_WORDS) -> Iterator[str]:
    """Group streamed text into speakable pieces. The first piece may end at the first
    clause boundary (comma, semicolon, colon, dash) once it has a few words, so the
    first audio starts sooner; later pieces are whole sentences, for natural prosody."""
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
            if first:
                cut = next((m for m in _CLAUSE_END.finditer(buf) if len(buf[:m.start()].split()) >= first_min_words),
                           None)
                if cut is not None:
                    yield buf[:cut.end()].strip()
                    buf = buf[cut.end():]
                    first = False
                    continue
            break
    if buf.strip():
        yield buf.strip()


def page_summary(page: Page | None) -> str:
    if not page or not page.steps:
        return "(nothing read yet)"
    marks = {"ok": "checked: correct", "error": "HAS THE MISTAKE", "unverifiable": "could not be fully checked",
             "unchecked": "not checked yet"}
    lines = [f"Problem: {page.problem}"] if page.problem else []
    from lens.checker import restates_problem

    lines += [f"Step {i + 1}: {s.latex}   [{'the problem as given, copied down'
                                           if s.status != 'error' and restates_problem(s.latex, page.problem)
                                           else marks[s.status]}]" for i, s in enumerate(page.steps)]
    return "\n".join(lines)


def _key(latex: str) -> str:
    return re.sub(r"\s+|\\[,;! ]|[{}]", "", latex).lower()


HINT_REPEAT_S = 600.0  # a mistake hinted this recently is not hinted again unless it was fixed in between


def same_step(key_a: str, box_a, key_b: str, box_b, min_ratio: float = 0.75, max_shift: float = 0.12) -> bool:
    """Two readings of one written step: nearly the same text (a re-read may differ by a
    symbol or two) at about the same place on the page (a hand-held page sways)."""
    from difflib import SequenceMatcher

    if key_a != key_b:
        # Different numbers mean a new attempt ("2x=16" rewritten as "2x=17"), not a re-read.
        if re.findall(r"\d+", key_a) != re.findall(r"\d+", key_b):
            return False
        if SequenceMatcher(None, key_a, key_b).ratio() < min_ratio:
            return False
    if not box_a or not box_b or box_a[3] <= 0 or box_b[3] <= 0:
        return True  # no position known: the text decides
    ca = (box_a[0] + box_a[2] / 2, box_a[1] + box_a[3] / 2)
    cb = (box_b[0] + box_b[2] / 2, box_b[1] + box_b[3] / 2)
    return abs(ca[0] - cb[0]) <= max_shift * 2 and abs(ca[1] - cb[1]) <= max_shift


def new_steps(before: Page | None, after: Page) -> list[Step]:
    """Steps in `after` that were not on the page before (by what is written)."""
    old = {_key(s.latex) for s in before.steps} if before else set()
    return [s for s in after.steps if _key(s.latex) not in old]


def _with_whole_page(expected, page: Page):
    """The answer guard for reading the page back: every step the student wrote, even
    after the mistake, counts as already written."""
    import dataclasses

    from lens.hints import _NUMBER, _norm

    on_page, numbers = set(expected.on_page), set(expected.page_numbers)
    for s in page.steps:
        line = s.latex.lstrip("=")
        on_page.add(_norm(line))
        on_page.update(_norm(p) for p in line.split("="))
        numbers.update(_NUMBER.findall(s.latex))
    return dataclasses.replace(expected, on_page=on_page, page_numbers=numbers)


def scratched_steps(before: Page | None, after: Page) -> list[Step]:
    """Steps that were read before and are gone now, with nothing written in their place:
    crossed out (the reader leaves crossed-out writing out) or rubbed out. Only on the
    same page: most of the other steps must still be there, so a new page or a camera
    that moved does not look like crossing out."""
    if before is None or len(before.steps) < 2:
        return []
    now = {_key(s.latex) for s in after.steps}
    gone = [s for s in before.steps if _key(s.latex) not in now]
    kept = len(before.steps) - len(gone)
    if not gone or kept < max(1, len(before.steps) // 2):
        return []

    def covered(s: Step) -> bool:  # something new written where it was
        y, h = s.bbox[1], s.bbox[3]
        return h <= 0 or any(n.bbox[3] > 0 and min(y + h, n.bbox[1] + n.bbox[3]) - max(y, n.bbox[1]) > 0.5 * min(
            h, n.bbox[3]) for n in new_steps(before, after))

    return [s for s in gone if not covered(s)]


def _same_step(a: str, b: str) -> bool:
    return _key(a) == _key(b) or same_math(a, b)


def fix_confirmed(before: Page | None, previous: Finding | None, after: Page) -> bool:
    """Did the student fix the flagged step? Only when the steps before it read the same
    as before (in meaning) and the step in its place is different and checks out. A
    re-read that misreads an earlier step can make the flagged step look fine without
    anything being fixed; that is not a fix."""
    if before is None or previous is None:
        return False
    ids = [s.id for s in before.steps]
    if previous.step_id not in ids:
        return False
    i = ids.index(previous.step_id)
    wrong = before.steps[i]
    earlier = [s for s in before.steps[:i] if not is_connector(s.latex)]
    rest = iter(s for s in after.steps if not is_connector(s.latex))
    for old in earlier:
        now = next(rest, None)
        if now is None or not _same_step(old.latex, now.latex):
            return False
    replacement = next(rest, None)
    return (replacement is not None and replacement.status == "ok"
            and not _same_step(replacement.latex, wrong.latex))


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
    clock: Callable[[], float] = time.monotonic  # behaviour timers (a seam for tests)
    _speaking: asyncio.Task | None = None
    _cancel: threading.Event = field(default_factory=threading.Event)
    _utt: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _last_filler: float = -1e9
    _fillers: int = 0
    # streaming speech
    _speech: bytearray | None = None
    _prev_pcm: bytes = b""
    _prev_text: str | None = None  # the last turn, while a continuation may still replace it
    _prev_at: float = -1e9  # when that turn was heard (a continuation must come soon after)
    _merging: bool = False
    _early: tuple[int, asyncio.Task] | None = None
    talking: bool = False  # the student is mid-utterance
    # what the student is doing, and when the tutor last spoke up
    _activity: tuple[str, float] | None = None  # (phase, when) from the browser's detector
    _started: float | None = None
    _last_ink: float | None = None  # new writing seen (a check that read new steps, or the pen moving)
    _last_step_at: float | None = None
    _last_said: float = -1e9  # the tutor finished speaking
    _last_heard: float = -1e9  # the student last talked
    _hint_at: float = -1e9
    _nudged_for: float | None = None  # the pause already nudged (its _last_ink)
    _encouraged_at: float = -1e9
    _praised_hard_at: float = -1e9
    _scratch_at: float = -1e9
    _looked_at: float = -1e9
    _said_recent: list[str] = field(default_factory=list)
    _spoken_tail: list[tuple[float, str]] = field(default_factory=list)  # (when, sentence) Lens said lately
    _ticker: asyncio.Task | None = None
    # accessibility
    aloud: bool | None = None  # None: speak whenever the voice conversation is on (older clients)
    speed: float = 1.0
    _typed: bool = False  # the student has typed a turn: the conversation is on without a microphone
    _checks: int = 0
    _mistakes: list[dict] = field(default_factory=list)  # for the end-of-session summary only
    away: bool = False  # the browser says nobody is at Lens (idle or hidden tab): no speaking up
    _slow_noted: float = -1e9
    _slow_said: float = -1e9
    _hinted: list = field(default_factory=list)  # (step key, bbox, when) of mistakes already hinted
    # a different problem: the old one's steps (still on the paper) are left out of readings
    _forget: list[str] = field(default_factory=list)
    _problem_no: int = 0  # counts problems this session; a check started before a switch is dropped
    _done_steps: int = 0  # steps read and checked on earlier problems (for the summary)
    _done_ok: int = 0
    _loop: asyncio.AbstractEventLoop | None = None
    _guided: int = 0  # student turns about the open mistake since its hint (raises the guidance level)
    # private notes on the problem (lens.notes): kept up to date in the background, used in every reply
    notes: dict = field(default_factory=dict)
    _notes_task: asyncio.Task | None = None
    _notes_again: str | None = None  # an event that came while an update was running
    _direction_at: float = -1e9
    _direction_said: str | None = None  # the progress the last "where is this heading?" was about
    _nudges: int = 0  # check-ins in the current pause
    # camera pictures for looking: (when, jpeg), the newest last
    _frames: list[tuple[float, bytes]] = field(default_factory=list)
    _turn_start: float | None = None
    _turn_end: float | None = None

    # ---------- entry points ----------

    async def start(self) -> None:
        self._started = self._started or self.clock()
        await self._say_now([OPENING], record=True)

    def start_ticker(self) -> None:
        """Speak up unasked when it helps (a stuck student): checked every second."""
        self._started = self._started or self.clock()

        async def loop():
            while True:
                await asyncio.sleep(TICK_S)
                try:
                    await self.tick()
                except Exception:
                    log.exception("tick failed")

        self._ticker = asyncio.create_task(loop())

    async def on_frame(self, jpeg: bytes) -> None:
        self.frame = jpeg
        now = self.clock()
        self._frames = [f for f in self._frames if now - f[0] <= LOOK_KEEP_S][-40:] + [(now, jpeg)]

    async def on_slow(self, what: str = "check") -> None:
        """The model is slow (a queued call): tell the student, and say it aloud now and then."""
        now = self.clock()
        if now - self._slow_noted < 10.0:
            return
        self._slow_noted = now
        await self.send({"type": "note", "text": SLOW_NOTE, "what": what})
        if self.voice and self.services.speech and self._free() and now - self._slow_said >= SLOW_SAY_COOLDOWN_S:
            self._slow_said = now
            await self._say_now([SLOW_SAY], record=False)

    async def _patient(self, coro, slow_s: float, cap_s: float):
        """Await `coro`; after `slow_s` tell the student it is slow; give up after `cap_s`."""
        task = asyncio.ensure_future(coro)
        done, _ = await asyncio.wait({task}, timeout=slow_s)
        if done:
            return task.result()
        await self.on_slow("check")
        try:
            return await asyncio.wait_for(task, max(0.1, cap_s - slow_s))
        except TimeoutError:
            log.warning("check gave up after %.0fs", cap_s)
            raise

    async def on_interrupt(self) -> None:
        await self._stop_speaking()
        await self.send({"type": "state", "state": "listening"})

    async def on_activity(self, phase: str) -> None:
        if phase in ("writing", "paused", "away"):
            self._activity = (phase, self.clock())
            if phase == "writing":
                self._last_ink = self.clock()

    # --- streamed speech

    async def on_speech_start(self, continues: bool = False) -> None:
        # Nothing stops here: speech that starts may be a pen tap or a rustle. Lens stops talking
        # when the student really talks (the browser sends "interrupt") or once the turn's
        # transcript has words in it (on_utterance).
        self.talking = True
        self._last_heard = self.clock()
        self._early = None
        if not (continues and self._turn_start is not None and self.clock() - self._prev_at <= MERGE_S):
            self._turn_start = self.clock()
        self._merging = bool(continues and self._prev_text is not None and self._prev_pcm
                             and self.clock() - self._prev_at <= MERGE_S)
        # The same sentence after a pause: transcribed together with its first half. Whether the
        # reply to that half is moot is decided once the second half turns out to have words.
        self._speech = bytearray(self._prev_pcm) if self._merging else bytearray()

    async def on_speech_chunk(self, pcm: bytes) -> None:
        if self._speech is None:
            self._speech = bytearray()
        if len(self._speech) + len(pcm) <= MAX_SPEECH_BYTES:
            self._speech += pcm

    async def on_speech_pause(self) -> None:
        """Transcribe what has been said so far while the browser waits out the rest of the
        silence; if the student does not go on, the transcript is ready at speech_end."""
        if self._speech is None or not self._speech or self.services.speech is None:
            return
        from lens.voice import pcm_to_wav

        wav = pcm_to_wav(bytes(self._speech), ASR_RATE)
        self._early = (len(self._speech), asyncio.create_task(asyncio.to_thread(self.services.speech.transcribe, wav)))

    async def on_speech_resume(self) -> None:
        self._early = None

    async def on_speech_end(self) -> None:
        heard_at = time.monotonic()
        self.talking = False
        self._turn_end = self.clock()
        pcm = bytes(self._speech or b"")
        self._speech = None
        early, self._early = self._early, None
        merging, self._merging = self._merging, False
        if not pcm:
            return
        from lens.voice import pcm_to_wav

        await self.on_utterance(pcm_to_wav(pcm, ASR_RATE), heard_at, early=early[1] if early else None,
                                pcm=pcm, merging=merging)

    async def on_speech_cancel(self) -> None:
        # A cough or a click, also after a turn: the turn before it and its reply go on untouched.
        self.talking = False
        self._speech, self._early, self._merging = None, None, False

    # --- whole utterances

    async def on_utterance(self, wav: bytes, heard_at: float | None = None,
                           early: asyncio.Task | None = None, pcm: bytes | None = None,
                           merging: bool = False) -> None:
        """A finished spoken turn. Lens stops speaking only when the transcript has words in it:
        a turn that was only noise (empty, or a continuation that added nothing to the turn before)
        leaves whatever Lens is saying alone."""
        heard_at = heard_at or time.monotonic()
        self.talking = False
        self._last_heard = self.clock()
        if self.services.speech is None:
            return
        text = None
        if early is not None:
            try:
                text = (await early).strip()
            except Exception:
                log.exception("early transcript failed")
        if text is None:
            text = (await asyncio.to_thread(self.services.speech.transcribe, wav)).strip()
        t_asr = time.monotonic()
        if (not _WORDS.search(text) or (merging and _norm(text) == _norm(self._prev_text or ""))
                or self._heard_self(text)):
            # Noise or Lens hearing itself: nothing to answer. (The browser lets a lowered Lens go on.)
            log.info("ignored a turn: no new words, or Lens's own echo (%d chars, continuation=%s)", len(text), merging)
            await self.send({"type": "transcript", "text": ""})
            return
        await self._stop_speaking()
        if merging:
            self._forget_turn(self._prev_text)  # the reply to the first half is moot
        self._prev_pcm = pcm or b""  # only a streamed turn with words can be continued
        self._prev_at = self.clock()
        await self.send({"type": "transcript", "text": text})
        await self.send({"type": "state", "state": "thinking"})
        await self._respond(text, heard_at, t_asr)

    def _heard_self(self, text: str) -> bool:
        """The transcript is Lens's own voice coming back through the mic (the echo canceller let it
        through): three or more words, nearly all of them in what Lens said in the last ECHO_S."""
        words = _norm(text).split()
        if len(words) < 3:
            return False
        now = self.clock()
        recent = set(" ".join(_norm(t) for at, t in self._spoken_tail if now - at <= ECHO_S).split())
        return sum(w in recent for w in words) >= 0.8 * len(words)

    async def _respond(self, text: str, heard_at: float, t_asr: float) -> None:
        self._prev_text = text
        if _READ_PAGE.search(text) and _READ_REQUEST.search(text):
            self.history.append({"role": "user", "content": text})
            await self.read_page()
            return
        if _NEW_PROBLEM.search(text):
            await self.new_problem()
            self.notes = {}
        if self.problem is None and not (self.page and self.page.problem):
            self.problem = text  # their first answer to "what are you working on"; confirmed in conversation
        self._nudges = 0
        pointing = await self._pointing(text)
        if self.finding is not None:
            self._guided += 1
        self.history.append({"role": "user", "content": text})
        look = self._wants_image(text)
        images = await self._look_frames(text) if look else []
        messages = self._messages(pointing, images)
        self._speaking = asyncio.create_task(self._reply(messages, {"heard_at": heard_at, "asr_done": t_asr}, look))
        if len(text.split()) >= 3:
            self._update_notes("the student said something (the last student turn above)")

    def _forget_turn(self, text: str | None) -> None:
        """Take back a user turn that is being replaced by a longer one (and any reply to it)."""
        for i in range(len(self.history) - 1, max(-1, len(self.history) - 3), -1):
            if self.history[i] == {"role": "user", "content": text}:
                del self.history[i:]
                break
        if self.problem == text:
            self.problem = None

    # --- page checks

    async def on_check(self, jpeg: bytes, reason: str = "manual", box: list[float] | None = None) -> None:
        """A page check. A just-finished line (`line_done`) is read and checked on its
        own when the page is already known (seconds instead of a whole-page read);
        everything else, and any line the fast path cannot place, reads the page."""
        self.frame = jpeg
        t0 = time.monotonic()
        async with self._lock:  # one check at a time
            await self.send({"type": "state", "state": "reading"})
            timings: dict = {"reason": reason}
            waited = time.monotonic() - t0
            if waited >= 0.05:
                timings["wait_s"] = round(waited, 3)  # behind another check
            problem_no = self._problem_no
            self._loop = asyncio.get_running_loop()
            try:
                async def run():
                    result = await self._check_line(jpeg, reason, box, timings)
                    if result is None:
                        timings["path"] = "page"
                        return await asyncio.to_thread(self._check_page, jpeg, timings)
                    timings["path"] = "line"
                    return result

                page, finding = await self._patient(run(), CHECK_SLOW_S, CHECK_CAP_S)
            except Exception as exc:
                log.exception("check failed")
                await self.send({"type": "error", "message": check_error_message(exc)})
                await self.send({"type": "state", "state": "listening"})
                return
            timings["check_s"] = round(time.monotonic() - t0, 3)
            if problem_no != self._problem_no:  # the student moved to another problem meanwhile
                log.info("check %s dropped: started before a new problem", timings)
                await self.send({"type": "state", "state": "listening"})
                return
            previous, before = self.finding, self.page
            if before is not None and before.steps and moved_on(before, page):
                # A new page or a different problem: nothing about the old one carries over
                # (no "that works now", no crossed-out remark, hints may be said again).
                timings["new_problem"] = True
                self._close_problem(forget=False)
                previous = before = None
            prev_line = self._line_of(previous)
            self.page, self.finding = page, finding
            self._note_mistakes(page, finding)
            self._follow(jpeg, page, finding)
            await self.send({"type": "result", "page": page.model_dump(),
                             "finding": finding.model_dump() if finding else None})
            await self.send({"type": "check_timing", **timings})
            log.info("check %s", timings)
            await self._after_check(jpeg, reason, before, previous, prev_line, page, finding)

    def _follow(self, jpeg: bytes, page: Page, finding: Finding | None) -> None:
        """Tell live tracking which step is outlined, on the frame this check read (its box is
        in that frame's coordinates), before the browser draws it."""
        if self.services.follow is None:
            return
        step = page.step(finding.step_id) if finding else None
        try:
            self.services.follow(jpeg, tuple(step.bbox) if step is not None and step.bbox[2] > 0 else None)
        except Exception:
            log.exception("outline follow failed to start")

    async def _after_check(self, jpeg: bytes, reason: str, before: Page | None, previous: Finding | None,
                           prev_line: str | None, page: Page, finding: Finding | None) -> None:
        """What to say after a check, most important first: a new mistake's hint, the fix,
        a crossed-out step, encouragement for a hard-won step, a look at a non-step change."""
        now = self.clock()
        added = new_steps(before, page)
        took = now - self._last_step_at if self._last_step_at is not None else 0.0
        if added:
            self._last_ink = now
            self._last_step_at = now
            self._nudges = 0
        if added or (page.problem and not (before and before.problem == page.problem)):
            self._update_notes("a check read the page (new steps or the problem)")
        gone = scratched_steps(before, page)
        # The same mistake read again (a swaying page re-checked, a step re-read a little
        # differently): its hint was already said, so it is not repeated.
        repeat = finding is not None and ((previous is not None and self._line_of(finding) == prev_line)
                                          or self._hinted_before(page, finding, now))
        if finding and not repeat:
            self._remember_hint(page, finding, now)
            self._hint_at = now
            self._guided = 0
            await self._say_when_free([finding.hint], record=True)
        elif previous and finding is None and page.steps:
            self._hinted = []  # moved on: the same mistake made again later gets its question again
            if fix_confirmed(before, previous, page):
                await self._say_when_free([FIXED], record=True)
            elif gone and not added:  # the wrong step was crossed out and nothing written yet
                self._scratch_at = now
                await self._say_when_free([self._pick(SCRATCHED)], record=True)
            else:  # something else changed (e.g. a step above read differently): not a fix
                log.info("mistake no longer found, but the flagged step was not fixed; saying nothing")
                await self.send({"type": "state", "state": "listening"})
        elif gone and not added and now - self._scratch_at >= SCRATCH_COOLDOWN_S and self._free():
            self._last_ink = now
            self._scratch_at = now
            await self._say_now([self._pick(SCRATCHED)], record=True)
        elif (added and finding is None and all(s.status == "ok" for s in added) and before is not None
              and before.steps and now - self._encouraged_at >= PRAISE_COOLDOWN_S and self._free()):
            # A short word for every correct step (not too often); a warmer one for a hard-won step.
            self._encouraged_at = now
            hard = took >= ENCOURAGE_EFFORT_S and now - self._praised_hard_at >= ENCOURAGE_COOLDOWN_S
            if hard:
                self._praised_hard_at = now
            await self._say_now([self._pick(ENCOURAGEMENTS if hard else PRAISE)], record=True)
        elif (not added and not gone and not repeat and reason in ("new_ink", "page_changed", "line_done")
              and now - self._looked_at >= LOOK_COOLDOWN_S and self._free()):
            self._looked_at = now
            await self._proactive(LOOK_NOTE, image=jpeg)
        else:
            await self.send({"type": "state", "state": "listening"})

    def _remember_hint(self, page: Page, finding: Finding, now: float) -> None:
        s = page.step(finding.step_id)
        if s is not None:
            self._hinted = [h for h in self._hinted if now - h[2] < HINT_REPEAT_S][-5:] + [(_key(s.latex), s.bbox, now)]

    def _hinted_before(self, page: Page, finding: Finding, now: float) -> bool:
        s = page.step(finding.step_id)
        return s is not None and any(now - t < HINT_REPEAT_S and same_step(_key(s.latex), s.bbox, k, b)
                                     for k, b, t in self._hinted)

    async def _check_line(self, jpeg: bytes, reason: str, box, timings: dict) -> tuple[Page, Finding | None] | None:
        """The fast path for a finished line, or None when it does not apply."""
        if reason != "line_done" or self.services.check_line is None or not (self.page and self.page.steps):
            return None
        if box is None and self.services.line_box is not None:
            box = self.services.line_box()
        if not box or len(box) != 4 or box[2] <= 0 or box[3] <= 0:
            return None
        try:
            return await asyncio.to_thread(self.services.check_line, jpeg, self.page, self.finding, tuple(box),
                                           self._problem_hint(), timings)
        except Exception:
            log.exception("line check failed; reading the page instead")
            return None

    def _check_page(self, jpeg: bytes, timings: dict) -> tuple[Page, Finding | None]:
        """The whole-page check (worker thread). What the last check found goes along, so a
        mistake already verified on the same steps is not verified again, and the steps
        read show on the screen before they are checked."""
        from lens.pipeline import Prior

        loop = self._loop

        def on_read(page: Page) -> None:
            if loop is not None and page.steps:
                msg = {"type": "steps", "page": page.model_dump()}
                asyncio.run_coroutine_threadsafe(self.send(msg), loop)

        return self.services.check(jpeg, self._problem_hint(), prior=Prior(self.page, self.finding, list(self._forget)),
                                   on_read=on_read, timings=timings)

    def _close_problem(self, forget: bool) -> None:
        """The student left this problem. forget: its steps are still on the paper (they said
        they are starting another one), so later readings leave them out."""
        steps = self.page.steps if self.page else []
        self._done_steps += len(steps)
        self._done_ok += sum(s.status == "ok" for s in steps)
        if forget:
            self._forget = (self._forget + [_key(s.latex) for s in steps])[-60:]
        for m in self._mistakes:
            m["closed"] = True  # keeps its last status in the summary
        self._hinted = []
        self._problem_no += 1

    async def new_problem(self) -> None:
        """The student said they are starting a different problem: clear the steps list."""
        if not (self.page and self.page.steps) and self.finding is None:
            return
        log.info("new problem: the steps list starts afresh")
        self._close_problem(forget=True)
        self.page, self.finding, self.problem = None, None, None
        self.notes = {}
        await self.send({"type": "result", "page": Page(problem=None, steps=[]).model_dump(), "finding": None,
                         "cleared": True})

    # --- speaking up unasked

    async def tick(self) -> None:
        """Called every second: one gentle nudge when the student seems stuck."""
        now = self.clock()
        if not (self.voice or self._typed) or self.away or not self._free() or self._lock.locked():
            return  # muted or nobody there: no nagging in captions either
        live = self.live_state()
        if live is None or not live["present"]:
            return
        if live["phase"] == "writing":
            self._last_ink = now
            return
        start = self._started if self._started is not None else now
        pause = max(self._last_ink or start, start, self._last_heard)
        if self._nudged_for != pause:
            self._nudged_for, self._nudges = pause, 0  # a new pause
        if self._nudges >= MAX_NUDGES or now - self._hint_at < HINT_GRACE_S:
            return
        # The first check-in after STUCK_AFTER_S of quiet; one more NUDGE_AGAIN_S after Lens last spoke.
        wait = STUCK_AFTER_S if self._nudges == 0 else NUDGE_AGAIN_S
        if now - max(pause, self._last_said) < wait:
            return
        self._nudges += 1
        note = NUDGE_NOTE if self._nudges == 1 else NUDGE_AGAIN_NOTE
        await self._proactive(note.format(secs=int(now - pause)))

    def live_state(self) -> dict | None:
        """{"phase": writing | paused | away, "present": bool, "writing_ago_s"} from live tracking,
        else from the browser's pixel detector, else None (unknown)."""
        if self.services.live is not None:
            try:
                live = self.services.live()
            except Exception:
                live = None
            if live:
                return live
        if self._activity and self.clock() - self._activity[1] <= LIVE_FRESH_S:
            phase = self._activity[0]
            return {"phase": phase, "present": phase != "away", "writing_ago_s": None}
        return None

    def _live_line(self) -> str:
        now = self.clock()
        live = self.live_state()
        parts = []
        if live is not None:
            parts.append({"writing": "the student is writing.",
                          "paused": "the student is at the page but not writing.",
                          "away": "the student is not at the page."}.get(live["phase"], "unknown."))
        else:
            parts.append("unknown what the student is doing.")
        if self._last_step_at is not None:
            parts.append(f"Their last new step was read {int(now - self._last_step_at)} s ago.")
        n = len(self.page.steps) if self.page else 0
        parts.append(f"Steps read so far: {n}.")
        return " ".join(parts)

    def _free(self) -> bool:
        """Nobody is talking: the tutor is not speaking and the student is not mid-sentence."""
        return not self.talking and (self._speaking is None or self._speaking.done())

    def _pick(self, options: tuple[str, ...]) -> str:
        fresh = [o for o in options if o not in self._said_recent] or list(options)
        choice = random.choice(fresh)
        self._said_recent = (self._said_recent + [choice])[-3:]
        return choice

    async def _proactive(self, note: str, image: bytes | None = None) -> None:
        """Speak up with no question from the student: the model gets the usual context
        plus `note` (and a picture), and its reply passes the same guard as any reply."""
        await self._stop_speaking()
        messages = self._messages("", None)
        content: str | list[dict] = note
        if image is not None:
            url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
            content = [{"type": "text", "text": note}, {"type": "image_url", "image_url": {"url": url}}]
        messages.append({"role": "user", "content": content})
        self._speaking = asyncio.create_task(self._reply(messages, None, False))
        await asyncio.sleep(0)

    async def close(self) -> None:
        if self._ticker:
            self._ticker.cancel()
        await self._stop_speaking()

    # --- accessibility: typed turns, voice settings, reading aloud, the summary

    async def on_message(self, data: dict) -> None:
        """The accessibility messages ("text", "settings", "read_page", "end"; see the protocol above)."""
        kind = data.get("type")
        if kind == "text":
            await self.on_text(str(data.get("text") or ""))
        elif kind == "settings":
            self.on_settings(data)
        elif kind == "read_page":
            await self.read_page()
        elif kind == "end":
            await self._stop_speaking()
            await self.send(self.summary())

    async def on_text(self, text: str) -> None:
        """A typed turn: the same reply path as a spoken one, without the speech recogniser."""
        text = " ".join(text.split())[:MAX_TYPED_CHARS]
        if not text:
            return
        now = time.monotonic()
        self._typed = True
        self.talking = False
        self._last_heard = self.clock()
        self._prev_pcm, self._prev_text = b"", None  # a typed turn is never merged with speech
        await self._stop_speaking()
        await self.send({"type": "state", "state": "thinking"})
        await self.send({"type": "transcript", "text": text, "typed": True})
        await self._respond(text, now, now)

    def on_settings(self, data: dict) -> None:
        if isinstance(data.get("aloud"), bool):
            self.aloud = data["aloud"]
        speed = data.get("speed")
        if isinstance(speed, int | float) and not isinstance(speed, bool):
            self.speed = round(min(max(float(speed), SPEED_RANGE[0]), SPEED_RANGE[1]), 2)

    def _audio_out(self) -> bool:
        """Synthesise speech? Not when the student chose captions only."""
        if self.services.speech is None:
            return False
        return self.aloud if self.aloud is not None else self.voice

    def read_aloud_text(self) -> list[str]:
        """The page already read, as spoken sentences: only what the student wrote."""
        if not self.page or not self.page.steps:
            return [NOTHING_TO_READ]
        out = []
        if self.page.problem:
            out.append(f"You're working on: {speak_latex(self.page.problem)}.")
        steps = self.page.steps[:READ_MAX_STEPS]
        # Lens's own words carry no digits (counts in words): only the student's writing has numbers.
        n = count_words(len(steps))
        out.append(f"Here {'is the step' if len(steps) == 1 else f'are the {n} steps'} I've read.")
        flagged = self.finding.step_id if self.finding else None
        for i, s in enumerate(steps, 1):
            spoken = speak_latex(s.latex) or "something I couldn't read"
            mark = ", the one I've outlined" if s.id == flagged else ""
            out.append(f"Step {count_words(i)}{mark}: {spoken}.")
        return out

    async def read_page(self) -> None:
        await self._say_now(self.read_aloud_text(), record=True, quoting=True)

    def _note_mistakes(self, page: Page, finding: Finding | None) -> None:
        """Remember each mistake found and whether it was fixed, for the summary."""
        self._checks += 1
        now = {_key(s.latex): s for s in page.steps}
        first_error = next((i for i, s in enumerate(page.steps) if finding and s.id == finding.step_id), len(page.steps))
        for m in self._mistakes:
            if m.get("closed"):
                continue
            s = now.get(m["key"])
            if s is not None:
                m["status"] = {"ok": "fixed", "error": "open"}.get(s.status, m["status"])
            elif m["status"] != "fixed":
                # Gone from the page. Rewritten into a step that checks out (before any open mistake): fixed;
                # otherwise crossed out or changed.
                i = m["idx"]
                ok_now = i < len(page.steps) and page.steps[i].status == "ok" and i < first_error
                m["status"] = "fixed" if ok_now else "changed"
        step = page.step(finding.step_id) if finding else None
        if finding is None or step is None or any(m["key"] == _key(step.latex) and not m.get("closed")
                                                           for m in self._mistakes):
            return
        kind = finding.kind
        try:
            from lens.hints import answer_leak, expected_for

            if answer_leak(kind, expected_for(page, finding)) is not None:
                kind = SAFE_KIND
        except Exception:
            kind = SAFE_KIND
        self._mistakes.append({"key": _key(step.latex), "idx": page.steps.index(step), "step": step.latex,
                               "spoken": speak_latex(step.latex),
                               "kind": say_step(kind), "status": "open"})

    def summary(self) -> dict:
        """The session in plain facts: what was worked on, steps checked, mistakes and
        whether they were fixed. No hints, no corrections."""
        steps = self.page.steps if self.page else []
        started = self._started if self._started is not None else self.clock()
        return {
            "type": "summary",
            "minutes": round(max(0.0, self.clock() - started) / 60, 1),
            "problem": plain_text(self._problem_hint() or "") or None,
            "steps_read": self._done_steps + len(steps),
            "steps_ok": self._done_ok + sum(s.status == "ok" for s in steps),
            "checks": self._checks,
            "mistakes": [{k: m[k] for k in ("step", "spoken", "kind", "status")} for m in self._mistakes],
        }

    # ---------- internals ----------

    def _problem_hint(self) -> str | None:
        return ((self.page.problem if self.page and self.page.problem else None)
                or (self.notes.get("problem") if self.notes else None) or self.problem)

    def _line_of(self, finding: Finding | None) -> str | None:
        if finding is None or self.page is None:
            return None
        s = self.page.step(finding.step_id)
        return _key(s.latex) if s else finding.step_id  # "2x = 6" and "2x=6" are the same step

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
        return f"\nThe student is pointing at step {n}: {step.latex}"

    def _wants_image(self, text: str) -> bool:
        """Attach the camera frame when the question is about what the camera shows,
        or when nothing has been read yet. Image calls are slower, so plain talk stays text."""
        if not self.frame:
            return False
        return bool(_VISUAL.search(text)) or not (self.page and self.page.steps)

    async def _look_frames(self, text: str) -> list[bytes]:
        """The camera pictures for a turn about what the camera shows: from when the student began
        talking until a moment after they stopped, evenly spread, at most LOOK_FRAMES, in order.
        When the browser is sending pictures as they talk, wait briefly for one after the turn
        ended (longer for "watch this"); an older client's single picture is used as it is."""
        start = self._turn_start if self._turn_start is not None else self.clock()
        end = self._turn_end if self._turn_end is not None else self.clock()
        if self._turn_end is None or self._turn_end < start:
            end = self.clock()  # a typed turn: just now
        streaming = sum(1 for t, _ in self._frames if t >= start - 0.2) >= 2
        if streaming:
            watch = bool(_WATCH.search(text))
            wait = max(0.0, (WATCH_AFTER_S if watch else LOOK_AFTER_S) - (self.clock() - end))
            deadline = time.monotonic() + wait  # real time (self.clock may be a test's)
            while time.monotonic() < deadline and (watch or not any(t > end + 0.3 for t, _ in self._frames)):
                await asyncio.sleep(0.05)
        window = [f for t, f in self._frames if start - 0.6 <= t <= self.clock()]
        if not window:
            return [self.frame] if self.frame else []
        if len(window) <= LOOK_FRAMES:
            return window
        idx = [round(i * (len(window) - 1) / (LOOK_FRAMES - 1)) for i in range(LOOK_FRAMES)]
        return [window[i] for i in idx]

    def _update_notes(self, event: str) -> None:
        """Rewrite the private problem notes in the background (one update at a time; an event
        that comes meanwhile runs once more after it)."""
        if self.services.notes is None:
            return
        if self._notes_task is not None and not self._notes_task.done():
            self._notes_again = event
            return

        async def run(ev: str) -> None:
            while ev:
                talk = [t for t in self.history[-12:] if isinstance(t.get("content"), str)]
                before = dict(self.notes)
                problem_no = self._problem_no
                try:
                    out = await asyncio.to_thread(self.services.notes, before or {}, talk,
                                                  page_summary(self.page), ev)
                except Exception:
                    log.exception("notes update failed")
                    out = before
                if problem_no == self._problem_no:  # not a stale update for a problem just left
                    self.notes = out or {}
                    log.info("notes: %s", {k: (v[:80] if isinstance(v, str) else v) for k, v in self.notes.items()})
                    await self._maybe_redirect()
                ev, self._notes_again = self._notes_again, None

        self._notes_task = asyncio.create_task(run(event))

    async def _maybe_redirect(self) -> None:
        """Their steps are right but the notes say the approach cannot reach what is asked:
        one question about where it is heading (never called a mistake)."""
        n = self.notes
        if n.get("on_track") is False and self._speaking is not None and not self._speaking.done():
            try:  # a short "good" for the step is still being said: let it finish first
                await asyncio.wait_for(asyncio.shield(self._speaking), 8.0)
            except Exception:
                log.debug("the remark before the redirect did not finish in time")
        now = self.clock()
        if (n.get("on_track") is not False or not n.get("why_off") or self.finding is not None
                or not (self.page and self.page.steps) or now - self._direction_at < DIRECTION_COOLDOWN_S
                or n.get("progress") == self._direction_said or not (self.voice or self._typed)
                or self.away or not self._free() or self._lock.locked()):
            return
        self._direction_at = now
        self._direction_said = n.get("progress")
        await self._proactive(DIRECTION_NOTE.format(why=n["why_off"]))

    def _notes_line(self) -> str:
        from lens.notes import known, render

        if self.notes and known(self.notes):
            return render(self.notes)
        if self.problem and not (self.page and self.page.problem):
            return f"- The student said they are working on: {self.problem}"
        if self.page and self.page.problem:
            return f"- Problem (as read from the page): {self.page.problem}"
        return ("- Not known yet. Ask what they are working on, including every given value, and what they"
                " have to find.")

    def _messages(self, pointing: str, images: list[bytes] | bytes | None = None) -> list[dict]:
        finding = ""
        if self.finding and self.page:
            n = next((i + 1 for i, s in enumerate(self.page.steps) if s.id == self.finding.step_id), "?")
            level = min(3, 1 + max(0, self._guided - 1))
            finding = (f"\nVerified mistake in step {n} (for you only, do not reveal the fix): {self.finding.kind}."
                       f"\nYou already asked: {self.finding.hint}"
                       f"\nGuidance: level {level} of 3 (the student has replied {self._guided} time(s) since"
                       f" your question; go one level clearer only if they are still stuck).")
        system = TUTOR_SYSTEM.format(page=page_summary(self.page), finding=finding, pointing=pointing,
                                     live=self._live_line(), notes=self._notes_line())
        history = list(self.history[-HISTORY_TURNS:])
        if isinstance(images, bytes):
            images = [images]
        if images and history and history[-1]["role"] == "user":
            parts: list[dict] = [{"type": "text", "text": history[-1]["content"]}]
            if len(images) > 1:
                parts.append({"type": "text", "text": "(Camera pictures while I spoke, in time order, about"
                                                      " half a second apart; the last is the most recent.)"})
            for i, img in enumerate(images, 1):
                if len(images) > 1:
                    parts.append({"type": "text", "text": f"Picture {i}:"})
                url = "data:image/jpeg;base64," + base64.b64encode(img).decode()
                parts.append({"type": "image_url", "image_url": {"url": url}})
            history[-1] = {"role": "user", "content": parts}
        return [{"role": "system", "content": system}, *history]

    def _guard(self, quoting: bool = False) -> Callable[[str], bool]:
        """True if a sentence may be spoken. `quoting`: the text reads the student's own
        steps back, so everything written on the page counts as already said."""
        if not (self.finding and self.page):
            return lambda s: True
        from lens.hints import answer_leak, expected_for

        try:
            expected = expected_for(self.page, self.finding)
            if quoting:
                expected = _with_whole_page(expected, self.page)
        except Exception:
            return lambda s: True
        return lambda s: answer_leak(s, expected) is None

    async def _reply(self, messages: list[dict], timing: dict | None, look: bool = False) -> None:
        cancel = self._cancel = threading.Event()
        queue: asyncio.Queue[str | None] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        marks = timing if timing is not None else {}

        def produce() -> None:
            def timed(deltas: Iterator[str]) -> Iterator[str]:
                for d in deltas:
                    marks.setdefault("llm_first", time.monotonic())
                    yield d

            try:
                deltas = self.services.brain.stream(messages, effort="minimal", max_tokens=1500, purpose="reply")
                for s in sentences(timed(deltas)):
                    if cancel.is_set() or abandoned.is_set():
                        break
                    marks.setdefault("piece", time.monotonic())
                    loop.call_soon_threadsafe(queue.put_nowait, s)
            except Exception as exc:
                log.exception("reply stream failed")
                failure.append(exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        abandoned = threading.Event()  # the stream stalled past REPLY_CAP_S: stop reading it
        failure: list[BaseException] = []
        asked = timing is not None  # the student asked (a proactive remark just stays quiet)
        threading.Thread(target=produce, daemon=True).start()

        def failed_line() -> str | None:
            from lens.limits import BudgetExceeded

            if failure and isinstance(failure[0], BudgetExceeded):
                return str(failure[0])
            return REPLY_FAILED if asked else None

        async def pieces():
            unset = object()
            first: object = unset
            waited = False
            if look and time.monotonic() - self._last_filler > LOOK_FILLER_COOLDOWN_S:
                # An image reply starts slower. Fill the silence only if it really is slow.
                try:
                    first = await asyncio.wait_for(queue.get(), LOOK_FILLER_AFTER_S)
                except TimeoutError:
                    self._last_filler = time.monotonic()
                    self._fillers += 1
                    waited = True
                    yield LOOKING[(self._fillers - 1) % len(LOOKING)]
            if first is unset:
                # Never silent: a slow start gets "One moment.", a stalled one an apology.
                try:
                    first = await asyncio.wait_for(queue.get(), REPLY_SLOW_S)
                except TimeoutError:
                    if asked and not waited:
                        yield REPLY_SLOW_SAY
                    try:
                        first = await asyncio.wait_for(queue.get(), max(0.1, REPLY_CAP_S - REPLY_SLOW_S))
                    except TimeoutError:
                        log.warning("reply stalled for %.0fs; giving up", REPLY_CAP_S)
                        abandoned.set()
                        first = None
            if first is None:  # nothing came: an error or a stall
                line = failed_line()
                if line:
                    yield line
                return
            yield first
            while (s := await queue.get()) is not None:
                yield s

        await self._speak(pieces(), cancel, record=True, timing=timing)

    async def _say_now(self, texts: list[str], record: bool, quoting: bool = False) -> None:
        await self._stop_speaking()
        cancel = self._cancel = threading.Event()

        async def pieces():
            for t in texts:
                yield t

        self._speaking = asyncio.create_task(self._speak(pieces(), cancel, record=record, quoting=quoting))
        await asyncio.sleep(0)

    async def _say_when_free(self, texts: list[str], record: bool) -> None:
        if self._speaking and not self._speaking.done():
            try:
                await self._speaking
            except asyncio.CancelledError:
                pass
        await self._say_now(texts, record)

    async def _speak(self, pieces, cancel: threading.Event, record: bool, timing: dict | None = None,
                     quoting: bool = False) -> None:
        self._utt += 1
        utt = self._utt
        allowed = self._guard(quoting)
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
                text = say_step(text)
                spoken.append(text)
                self._spoken_tail = [*self._spoken_tail, (self.clock(), text)][-8:]
                await self.send({"type": "say", "text": text, "utt": utt})
                if not self._audio_out():
                    continue
                async for pcm, sr in self._synth(text, cancel):
                    if first_audio is None:
                        first_audio = time.monotonic()
                        if timing:
                            await self._send_timing(timing, first_audio)
                    step = max(2, int(sr * AUDIO_SLICE_S) * 2)
                    for i in range(0, len(pcm), step):
                        await self.send({"type": "audio", "utt": utt, "sample_rate": sr})
                        await self.send_bytes(pcm[i:i + step])
        finally:
            if record and spoken:
                self.history.append({"role": "assistant", "content": " ".join(spoken)})
            self._last_said = self.clock()
            await self.send({"type": "audio_end", "utt": utt})
            if not cancel.is_set() or spoken:
                await self.send({"type": "state", "state": "listening"})

    async def _send_timing(self, timing: dict, first_audio: float) -> None:
        """Where the time went between the utterance arriving and the first audio, by stage
        (seconds from `heard_at`): asr, first model text, first speakable piece, first audio."""
        t0 = timing["heard_at"]
        out = {"type": "timing", "asr_s": round(timing["asr_done"] - t0, 3),
               "first_audio_s": round(first_audio - t0, 3)}
        for key in ("llm_first", "piece"):
            if key in timing:
                out[f"{key}_s"] = round(timing[key] - t0, 3)
        await self.send(out)
        log.info("voice latency %s", {k: v for k, v in out.items() if k != "type"})

    async def _synth(self, text: str, cancel: threading.Event):
        """TTS chunks for `text`, passed on as each one is ready (not after the whole piece)."""
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def run() -> None:
            try:
                # Normal speed keeps the plain call, so any Speech without a speed option still works.
                chunks = (self.services.speech.synthesize(text) if self.speed == 1.0
                          else self.services.speech.synthesize(text, speed=self.speed))
                for chunk in chunks:
                    if cancel.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception:
                log.exception("speech synthesis failed")
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=run, daemon=True).start()
        while (chunk := await queue.get()) is not None:
            if cancel.is_set():
                break
            yield chunk

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
