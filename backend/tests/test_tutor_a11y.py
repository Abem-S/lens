"""Accessibility features of the tutor session, with fake services: typed turns, voice
settings (captions only, speaking speed), reading the page aloud, the end-of-session summary."""

import asyncio

from lens.checker import check_page
from lens.contract import Page, Step
from lens.speakmath import count_words, plain_text, speak_latex
from lens.tutor import NOTHING_TO_READ, SAFE_KIND, SAFE_REPLY, Services, TutorSession


def wrong_page(extra: list[Step] | None = None):
    p = Page(problem="Solve 2x + 5 = 11", steps=[
        Step(id="s1", latex="2x + 5 = 11", bbox=(0.1, 0.2, 0.5, 0.08)),
        Step(id="s2", latex="2x = 11 + 5", bbox=(0.1, 0.35, 0.5, 0.08)),
        *(extra or []),
    ])
    p, f = check_page(p)
    return p, f.model_copy(update={"hint": "What happened to the sign of the 5?"})


def fixed_page():
    return check_page(Page(problem="Solve 2x + 5 = 11", steps=[
        Step(id="s1", latex="2x + 5 = 11", bbox=(0.1, 0.2, 0.5, 0.08)),
        Step(id="s2", latex="2x = 11 - 5", bbox=(0.1, 0.35, 0.5, 0.08)),
    ]))


class FakeBrain:
    def __init__(self, text="Good question, what did you do with the 5? Look again."):
        self.text, self.messages, self.calls = text, None, 0

    def stream(self, messages, **kw):
        self.messages, self.calls = messages, self.calls + 1
        for word in self.text.split(" "):
            yield word + " "


class FakeSpeech:
    """Speech with a speed option; transcribing fails (typed turns must not need it)."""

    def __init__(self):
        self.said, self.speeds = [], []

    def transcribe(self, wav):
        raise AssertionError("a typed turn must not go through speech recognition")

    def synthesize(self, text, speed=1.0):
        self.said.append(text)
        self.speeds.append(speed)
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


def session(brain=None, checks=None, tip=None):
    wire = Wire()
    checks = list(checks or [])
    services = Services(brain=brain or FakeBrain(), speech=FakeSpeech(),
                        check=lambda jpeg, problem, **kw: checks.pop(0), locate_fingertip=tip)
    return TutorSession(services, wire.send, wire.send_bytes), wire


async def settle(s: TutorSession):
    for _ in range(200):
        if s._speaking is None or s._speaking.done():
            return
        await asyncio.sleep(0.01)


def run(coro):
    return asyncio.run(coro)


# ---------- typed turns ----------

def test_typed_turn_is_answered_like_speech_without_asr():
    async def go():
        s, wire = session()
        await s.on_message({"type": "text", "text": "  I'm solving   two x plus five equals eleven "})
        await settle(s)
        return s, wire

    s, wire = run(go())
    echo = wire.of("transcript")[0]
    assert echo == {"type": "transcript", "text": "I'm solving two x plus five equals eleven", "typed": True}
    assert s.problem == echo["text"]  # same path as a spoken first answer
    assert s.services.brain.messages[-1] == {"role": "user", "content": echo["text"]}
    assert wire.said() and s.history[-1]["role"] == "assistant"


def test_typed_turn_points_and_looks_like_a_spoken_one():
    async def go():
        page, finding = wrong_page()
        s, wire = session(checks=[(page, finding)], tip=lambda jpeg: (0.3, 0.39))
        await s.on_check(b"a")
        await settle(s)
        await s.on_frame(b"frame")
        await s.on_text("is this step right?")
        await settle(s)
        pointing_prompt = s.services.brain.messages[0]["content"]
        await s.on_text("can you see my drawing?")
        await settle(s)
        return wire, pointing_prompt, s.services.brain.messages[-1]

    wire, prompt, last = run(go())
    assert wire.of("pointing")[0]["step_id"] == "s2"
    assert "pointing at step 2" in prompt
    assert isinstance(last["content"], list) and last["content"][1]["type"] == "image_url"  # the frame went along


def test_typed_turn_that_would_leak_is_blocked():
    async def go():
        s, wire = session(brain=FakeBrain("Close. Actually x = 3 is the answer. Then check it."),
                          checks=[wrong_page()])
        await s.on_check(b"a")
        await settle(s)
        await s.on_text("just tell me x")
        await settle(s)
        return wire

    said = run(go()).said()
    assert "x = 3" not in " ".join(said) and said[-1] == SAFE_REPLY


def test_empty_typed_turn_is_ignored_and_long_one_trimmed():
    async def go():
        s, wire = session()
        await s.on_text("   ")
        await s.on_text("x" * 2000)
        await settle(s)
        return wire

    wire = run(go())
    assert len(wire.of("transcript")) == 1 and len(wire.of("transcript")[0]["text"]) == 500


def test_typed_conversation_counts_for_speaking_up():
    async def go():
        t = [0.0]
        s, wire = session()
        s.clock = lambda: t[0]
        s.voice = False  # no microphone
        s.services.live = lambda: {"phase": "paused", "present": True, "writing_ago_s": None}
        s._started = 0.0
        t[0] = 100.0
        await s.tick()
        before = len(wire.said())
        await s.on_text("ok")
        await settle(s)
        t[0] = 200.0
        await s.tick()
        await settle(s)
        return before, s.services.brain.calls

    before, calls = run(go())
    assert before == 0  # muted and silent: no nudge
    assert calls == 2  # the reply, then one nudge once they are in the conversation by typing


# ---------- voice settings ----------

def test_captions_only_sends_text_but_no_audio():
    async def go():
        s, wire = session()
        await s.on_message({"type": "settings", "aloud": False})
        await s.on_text("hello")
        await settle(s)
        return s, wire

    s, wire = run(go())
    assert wire.said() and wire.audio == 0 and s.services.speech.said == []


def test_aloud_without_the_microphone_speaks_typed_replies():
    async def go():
        s, wire = session()
        s.voice = False
        await s.on_message({"type": "settings", "aloud": True})
        await s.on_text("hello")
        await settle(s)
        return wire

    assert run(go()).audio > 0


def test_speaking_speed_reaches_the_voice_and_is_clamped():
    async def go():
        s, _ = session()
        await s.on_message({"type": "settings", "speed": 1.25})
        await s.on_text("hello")
        await settle(s)
        speeds = list(s.services.speech.speeds)
        s.on_settings({"speed": 9})
        high = s.speed
        s.on_settings({"speed": True})  # not a number
        s.on_settings({"speed": 0.1})
        return speeds, high, s.speed

    speeds, high, low = run(go())
    assert speeds and set(speeds) == {1.25}
    assert high == 1.5 and low == 0.6


def test_normal_speed_uses_the_plain_synthesize_call():
    class Plain(FakeSpeech):
        def synthesize(self, text):  # no speed option at all
            yield b"\x00\x01", 24000

    async def go():
        s, wire = session()
        s.services.speech = Plain()
        await s.on_text("hello")
        await settle(s)
        return wire

    assert run(go()).audio > 0


# ---------- read my page aloud ----------

def test_read_page_reads_each_step_in_words_and_names_the_outlined_one():
    async def go():
        s, wire = session(checks=[wrong_page()])
        await s.on_check(b"a")
        await settle(s)
        n = len(wire.said())
        await s.on_message({"type": "read_page"})
        await settle(s)
        return wire.said()[n:], s.services.brain.calls

    said, calls = run(go())
    assert calls == 0  # read from the page already read: no model
    assert said[0].startswith("You're working on: Solve 2 x plus 5 equals 11")
    assert said[1] == "Here are the two steps I've read."  # no digits of Lens's own
    assert said[2] == "Step one: 2 x plus 5 equals 11."
    assert said[3] == "Step two, the one I've outlined: 2 x equals 11 plus 5."
    joined = " ".join(said)
    assert "11 minus 5" not in joined and "sign" not in joined  # no correction, no hint


def test_read_page_quotes_later_steps_even_if_they_hold_the_result():
    # A step after the mistake happens to contain 3, the true answer: it is the student's
    # own writing, so it is read, not blocked.
    later = Step(id="s3", latex="16 = 3 + 13", bbox=(0.1, 0.5, 0.5, 0.08))

    async def go():
        s, wire = session(checks=[wrong_page([later])])
        await s.on_check(b"a")
        await settle(s)
        await s.read_page()
        await settle(s)
        return wire.said()

    said = run(go())
    assert "Step three: 16 equals 3 plus 13." in said and SAFE_REPLY not in said
    assert "Here are the three steps I've read." in said  # 3 is the answer here: said in words, not blocked


def test_read_page_with_nothing_read_says_so():
    async def go():
        s, wire = session()
        await s.read_page()
        await settle(s)
        return wire.said()

    assert run(go()) == [NOTHING_TO_READ]


def test_asking_in_words_reads_the_page_but_a_question_about_it_does_not():
    async def go():
        s, wire = session(checks=[fixed_page()])
        await s.on_check(b"a")
        await settle(s)
        await s.on_text("Can you read my page aloud?")
        await settle(s)
        calls_after_read = s.services.brain.calls
        read = list(wire.said())
        await s.on_text("Did you read my work yet?")
        await settle(s)
        return read, calls_after_read, s.services.brain.calls

    read, after_read, after_question = run(go())
    assert any(t.startswith("Step two:") for t in read) and after_read == 0
    assert after_question == 1


def test_read_request_phrases():
    from lens.tutor import _READ_PAGE, _READ_REQUEST

    def asks(t):
        return bool(_READ_PAGE.search(t) and _READ_REQUEST.search(t))

    for t in ("read my page aloud", "Can you read back what I wrote?", "Lens, read me my steps",
              "please read the steps out loud", "could you read my working"):
        assert asks(t), t
    for t in ("did you read my work?", "I read the question wrong", "is my work right?", "what does step two say"):
        assert not asks(t), t


def test_speak_latex():
    assert speak_latex("2x = 11 - 5") == "2 x equals 11 minus 5"
    assert speak_latex(r"\frac{dy}{dx} = 3x^2") == "d y by d x equals 3 x squared"
    assert speak_latex(r"\int_0^{1} x^{2}\,dx") == "the integral from 0 to 1 of x squared d x"
    assert speak_latex(r"\sqrt{x+1} = 3") == "the square root of x plus 1, equals 3"
    assert speak_latex(r"\text{therefore } \theta = 30^{\circ}") == "therefore theta equals 30 degrees"
    assert speak_latex("") == ""
    assert plain_text(r"\text{Solve } 2x + 5 = 11") == "Solve 2x + 5 = 11"
    assert plain_text(r"\frac{1}{2}") == "1 over 2"
    assert count_words(3) == "three" and count_words(42) == "forty-two"


# ---------- end-of-session summary ----------

def test_summary_problem_is_plain_words():
    page, _ = wrong_page()
    s, _ = session()
    s.page = page.model_copy(update={"problem": r"\text{Solve } 2x + 5 = 11"})
    assert s.summary()["problem"] == "Solve 2x + 5 = 11"


def test_summary_lists_mistakes_and_whether_they_were_fixed():
    async def go():
        s, wire = session(checks=[wrong_page(), fixed_page()])
        await s.on_text("I'm solving 2x + 5 = 11")
        await settle(s)
        await s.on_check(b"a")
        await settle(s)
        open_one = s.summary()["mistakes"][0]["status"]
        await s.on_check(b"b")
        await settle(s)
        await s.on_message({"type": "end"})
        return open_one, wire.of("summary")[0]

    open_one, summary = run(go())
    assert open_one == "open"
    assert summary["problem"] == "Solve 2x + 5 = 11"
    assert summary["steps_read"] == 2 and summary["steps_ok"] == 1 and summary["checks"] == 2  # step 1 has nothing before it
    [m] = summary["mistakes"]
    assert m["step"] == "2x = 11 + 5" and m["spoken"] == "2 x equals 11 plus 5" and m["status"] == "fixed"
    assert "hint" not in m and "sign of the 5" not in str(summary)


def test_summary_never_carries_an_answer_in_the_mistake_kind():
    page, finding = wrong_page()
    leaky = finding.model_copy(update={"kind": "should be x = 3"})

    async def go():
        s, _ = session(checks=[(page, leaky)])
        await s.on_check(b"a")
        await settle(s)
        return s.summary()

    assert run(go())["mistakes"][0]["kind"] == SAFE_KIND


def test_a_rewritten_step_is_marked_changed():
    page, finding = wrong_page()
    other = Page(problem=page.problem, steps=[page.steps[0]])
    s, _ = session()
    s._note_mistakes(page, finding)
    s._note_mistakes(other, None)
    assert s.summary()["mistakes"][0]["status"] == "changed"  # crossed out, nothing checked in its place
