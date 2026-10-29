"""F1 / V2 server logic with fake services: opening, replies, leak guard, page
checks that speak hints, barge-in, pointing."""

import asyncio
import threading
import time

import pytest

from lens.checker import check_page
from lens.contract import Page, Step
from lens.tutor import FIXED, OPENING, SAFE_REPLY, Services, TutorSession, sentences


def wrong_page():
    p = Page(problem="Solve 2x + 5 = 11", steps=[
        Step(id="s1", latex="2x + 5 = 11", bbox=(0.1, 0.2, 0.5, 0.08)),
        Step(id="s2", latex="2x = 11 + 5", bbox=(0.1, 0.35, 0.5, 0.08)),
    ])
    p, f = check_page(p)
    return p, f.model_copy(update={"hint": "What happened to the sign of the 5?"})


def fixed_page():
    p = Page(problem="Solve 2x + 5 = 11", steps=[
        Step(id="s1", latex="2x + 5 = 11", bbox=(0.1, 0.2, 0.5, 0.08)),
        Step(id="s2", latex="2x = 11 - 5", bbox=(0.1, 0.35, 0.5, 0.08)),
    ])
    return check_page(p)


class FakeBrain:
    def __init__(self, text="Good question, what did you do with the 5? Look again."):
        self.text, self.messages = text, None

    def stream(self, messages, **kw):
        self.messages = messages
        for word in self.text.split(" "):
            yield word + " "


class FakeSpeech:
    def __init__(self, delay=0.0):
        self.delay, self.said = delay, []

    def transcribe(self, wav):
        return wav.decode()

    def synthesize(self, text):
        self.said.append(text)
        for _ in range(3):
            time.sleep(self.delay)
            yield b"\x00\x01" * 10, 24000


class Wire:
    def __init__(self):
        self.msgs, self.audio = [], 0

    async def send(self, m):
        self.msgs.append(m)

    async def send_bytes(self, b):
        self.audio += 1

    def of(self, t):
        return [m for m in self.msgs if m["type"] == t]

    def said(self):
        return [m["text"] for m in self.of("say")]


def session(brain=None, speech=None, checks=None, tip=None):
    wire = Wire()
    checks = list(checks or [])
    services = Services(brain=brain or FakeBrain(), speech=speech or FakeSpeech(),
                        check=lambda jpeg, problem: checks.pop(0), locate_fingertip=tip)
    return TutorSession(services, wire.send, wire.send_bytes), wire


async def settle(s: TutorSession):
    for _ in range(200):
        if s._speaking is None or s._speaking.done():
            return
        await asyncio.sleep(0.01)


def run(coro):
    return asyncio.run(coro)


def test_sentences_cut_early_for_fast_first_audio():
    out = list(sentences(iter(["Okay, so when you look at the second line carefully, ", "what changed? ", "Tell me."])))
    assert out[0].endswith(",") and len(out) == 3
    assert list(sentences(iter(["Hi. ", "What are you ", "working on?"]))) == ["Hi.", "What are you working on?"]


def test_opening_is_proactive_and_spoken():
    async def go():
        s, wire = session()
        await s.start()
        await settle(s)
        return s, wire

    s, wire = run(go())
    assert wire.said() == [OPENING] and wire.audio == 3
    assert s.history[-1] == {"role": "assistant", "content": OPENING}


def test_utterance_is_heard_answered_and_timed():
    async def go():
        s, wire = session()
        await s.on_utterance(b"I'm solving two x plus five equals eleven")
        await settle(s)
        return s, wire

    s, wire = run(go())
    assert wire.of("transcript")[0]["text"].startswith("I'm solving")
    assert s.problem.startswith("I'm solving")  # remembered, then confirmed in conversation
    assert len(wire.said()) == 3 and wire.audio == 9  # first piece cut at a comma for speed
    assert wire.of("timing")[0]["first_audio_s"] >= 0
    assert "working on: I'm solving" in s.services.brain.messages[0]["content"]


def test_page_check_speaks_the_hint_then_praises_the_fix():
    async def go():
        s, wire = session(checks=[wrong_page(), fixed_page()])
        await s.on_check(b"jpeg1")
        await settle(s)
        first = list(wire.said())
        await s.on_check(b"jpeg2")
        await settle(s)
        return wire, first

    wire, first = run(go())
    assert first == ["What happened to the sign of the 5?"]
    assert wire.said()[-1] == FIXED
    assert [m["finding"] is None for m in wire.of("result")] == [False, True]


def test_same_mistake_is_not_repeated_on_every_check():
    async def go():
        s, wire = session(checks=[wrong_page(), wrong_page()])
        await s.on_check(b"a")
        await settle(s)
        await s.on_check(b"b")
        await settle(s)
        return wire

    assert run(go()).said() == ["What happened to the sign of the 5?"]


def test_spoken_reply_that_leaks_the_answer_is_blocked():
    async def go():
        s, wire = session(brain=FakeBrain("Close. Actually x = 3 is the answer. Then check it."),
                          checks=[wrong_page()])
        await s.on_check(b"a")
        await settle(s)
        await s.on_utterance(b"what is it")
        await settle(s)
        return wire

    said = run(go()).said()
    assert "x = 3" not in " ".join(said)
    assert said[-1] == SAFE_REPLY


def test_barge_in_stops_speech_immediately():
    async def go():
        speech = FakeSpeech(delay=0.05)
        s, wire = session(brain=FakeBrain("One. Two. Three. Four. Five. Six."), speech=speech)
        await s.on_utterance(b"tell me something long")
        await asyncio.sleep(0.12)
        await s.on_interrupt()
        before = wire.audio
        await asyncio.sleep(0.4)
        return wire, before, speech

    wire, before, speech = run(go())
    assert wire.audio == before  # nothing more after the interrupt
    assert len(speech.said) < 6
    assert wire.of("state")[-1]["state"] == "listening"


def test_this_line_question_uses_pointing():
    async def go():
        page, finding = wrong_page()
        s, wire = session(checks=[(page, finding)], tip=lambda jpeg: (0.3, 0.39))
        await s.on_check(b"a")
        await settle(s)
        await s.on_frame(b"frame")
        await s.on_utterance(b"is this line right?")
        await settle(s)
        return s, wire

    s, wire = run(go())
    assert wire.of("pointing")[0]["step_id"] == "s2"
    assert "pointing at line 2" in s.services.brain.messages[0]["content"]


def test_no_pointing_without_a_deictic_word():
    calls = []

    async def go():
        s, _wire = session(checks=[wrong_page()], tip=lambda jpeg: calls.append(1) or (0.3, 0.39))
        await s.on_check(b"a")
        await settle(s)
        await s.on_utterance(b"I am stuck on the sign")
        await settle(s)

    run(go())
    assert calls == []


@pytest.mark.parametrize("n", [3])
def test_threads_do_not_leak(n):
    before = threading.active_count()

    async def go():
        s, _ = session()
        for _ in range(n):
            await s.on_utterance(b"hello")
            await settle(s)

    run(go())
    time.sleep(0.1)
    assert threading.active_count() <= before + 1
