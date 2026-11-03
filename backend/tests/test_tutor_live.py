"""Faster turns (streamed speech, early transcription, continuation after a pause,
early first piece) and the tutor speaking up by itself (stuck nudge, encouragement,
crossed-out step, a look at a non-step change, live context)."""

import asyncio

from test_tutor import FakeBrain, FakeSpeech, run, session, settle, wrong_page

from lens.contract import Page, Step
from lens.tutor import ENCOURAGEMENTS, FIXED, PRAISE, SCRATCHED, sentences

# ---------------------------------------------------------------- B: faster turns


def test_first_piece_ends_at_a_clause_after_a_few_words_never_inside_a_number():
    assert list(sentences(iter(["I see what you did, ", "and it is close. ", "Check it."]))) == [
        "I see what you did,", "and it is close.", "Check it."]
    assert next(sentences(iter(["Okay, ", "so where did the five go? "]))) == "Okay, so where did the five go?"
    assert next(sentences(iter(["You wrote 3,000 metres there, ", "right?"]))) == "You wrote 3,000 metres there,"
    assert next(sentences(iter(["Look at your second step — ", "what changed?"]))).endswith("—")


class PcmSpeech(FakeSpeech):
    """Transcribes the PCM payload of a WAV as text (the tests send text as 'audio')."""

    def __init__(self, delay=0.0):
        super().__init__(delay)
        self.heard = []

    def transcribe(self, wav):
        self.heard.append(wav[44:].decode())
        return wav[44:].decode()


def test_streamed_speech_is_one_turn_and_transcribed_early_at_the_pause():
    async def go():
        speech = PcmSpeech()
        s, wire = session(speech=speech)
        await s.on_speech_start()
        await s.on_speech_chunk(b"is this right")
        await s.on_speech_pause()  # quiet: the server starts transcribing
        await s.on_speech_chunk(b"  ")  # the rest of the silence
        await s.on_speech_end()
        await settle(s)
        return wire, speech

    wire, speech = run(go())
    assert speech.heard == ["is this right"]  # once, early; not again at the end
    assert wire.of("transcript")[0]["text"] == "is this right"
    assert wire.of("timing") and wire.said()


def test_speech_that_goes_on_after_the_pause_voids_the_early_transcript():
    async def go():
        s, wire = session(speech=PcmSpeech())
        await s.on_speech_start()
        await s.on_speech_chunk(b"so I ")
        await s.on_speech_pause()
        await s.on_speech_resume()
        await s.on_speech_chunk(b"moved it")
        await s.on_speech_end()
        await settle(s)
        return wire

    assert run(go()).of("transcript")[0]["text"] == "so I moved it"


def test_a_continuation_after_a_pause_is_heard_as_one_turn():
    async def go():
        brain = FakeBrain("Okay. Tell me more about it.")
        s, _wire = session(brain=brain, speech=PcmSpeech(delay=0.05))
        await s.on_speech_start()
        await s.on_speech_chunk(b"I moved the five ")
        await s.on_speech_end()  # a pause long enough to end the turn...
        await asyncio.sleep(0.02)
        await s.on_speech_start(continues=True)  # ...but they went on
        await s.on_speech_chunk(b"across")
        await s.on_speech_end()
        await settle(s)
        return s, brain

    s, brain = run(go())
    assert [h["content"] for h in s.history if h["role"] == "user"] == ["I moved the five across"]
    assert brain.messages[-1]["content"] == "I moved the five across"
    assert s.problem == "I moved the five across"


def test_a_cough_after_a_turn_does_not_lose_the_turn():
    async def go():
        s, wire = session(brain=FakeBrain("What did you try?"), speech=PcmSpeech(delay=0.05))
        await s.on_speech_start()
        await s.on_speech_chunk(b"I am stuck")
        await s.on_speech_end()
        await s.on_speech_start(continues=True)
        await s.on_speech_cancel()  # it was a cough
        await settle(s)
        return s, wire

    s, wire = run(go())
    assert [h["content"] for h in s.history if h["role"] == "user"] == ["I am stuck"]
    assert wire.said()[-1] == "What did you try?"


class NoisySpeech(PcmSpeech):
    """Noise transcribes to nothing (or a stray mark); words transcribe to themselves."""

    def transcribe(self, wav):
        text = super().transcribe(wav)
        return "" if text.startswith("~") else text.replace("~", "")


def _long_reply_session(speech=None):
    brain = FakeBrain("One. Two. Three. Four. Five. Six.")
    s, wire = session(brain=brain, speech=speech or NoisySpeech(delay=0.03))
    return s, wire, brain


async def _say(s, text: bytes, continues=False):
    await s.on_speech_start(continues=continues)
    await s.on_speech_chunk(text)
    await s.on_speech_end()


def test_noise_while_lens_speaks_does_not_stop_it():
    """A pen tap or a rustle the browser sent as a turn: empty transcript, Lens says every sentence."""
    async def go():
        s, wire, _brain = _long_reply_session()
        await _say(s, b"tell me something long")
        await asyncio.sleep(0.1)
        for noise in (b"~tap", b"~rustle", b"."):
            await s.on_speech_start()
            await s.on_speech_chunk(noise)
            await s.on_speech_end()
        await s.on_speech_start()
        await s.on_speech_cancel()  # a click the browser dropped itself
        await settle(s)
        return s, wire

    s, wire = run(go())
    assert wire.said() == ["One.", "Two.", "Three.", "Four.", "Five.", "Six."]
    assert [m["text"] for m in wire.of("transcript")][1:] == ["", "", ""]
    assert [h["content"] for h in s.history if h["role"] == "user"] == ["tell me something long"]
    assert not [m for m in wire.msgs[2:] if m == {"type": "state", "state": "thinking"}]


def test_words_spoken_over_lens_stop_it_and_get_an_answer():
    async def go():
        s, wire, brain = _long_reply_session()
        await _say(s, b"tell me something long")
        await asyncio.sleep(0.1)
        brain.text = "Sure, go ahead."
        await s.on_speech_start()
        await s.on_speech_chunk(b"wait I see it")
        await s.on_speech_end()
        await settle(s)
        return s, wire

    s, wire = run(go())
    said = wire.said()
    assert said[-1] == "Sure, go ahead."
    assert "Six." not in said  # the long reply was cut for real speech
    assert [h["content"] for h in s.history if h["role"] == "user"][-1] == "wait I see it"


def test_a_noise_continuation_leaves_the_turn_and_its_reply_alone():
    """Within the merge window, noise is sent as a continuation: its merged transcript adds no
    words, so the reply to the turn goes on and is not asked for again."""
    async def go():
        s, wire, brain = _long_reply_session()
        calls = []
        stream = brain.stream
        brain.stream = lambda m, **kw: (calls.append(1), stream(m, **kw))[1]
        await s.on_speech_start()
        await s.on_speech_chunk(b"I am stuck")
        await s.on_speech_end()
        await asyncio.sleep(0.05)
        await s.on_speech_start(continues=True)
        await s.on_speech_chunk(b"~")  # rustle: "I am stuck" + noise transcribes as "I am stuck"
        await s.on_speech_end()
        await settle(s)
        return s, wire, calls

    s, wire, calls = run(go())
    assert len(calls) == 1
    assert wire.said() == ["One.", "Two.", "Three.", "Four.", "Five.", "Six."]
    assert [h["content"] for h in s.history if h["role"] == "user"] == ["I am stuck"]


def test_lens_hearing_its_own_voice_is_not_a_turn():
    """Lens's words leaking back through the mic transcribe as Lens's words: ignored, Lens goes on.
    The student's own sentence, even with some of Lens's words in it, is still answered."""
    async def go():
        brain = FakeBrain("Where did the five go when you moved it across? Look again. Then tell me.")
        s, wire = session(brain=brain, speech=NoisySpeech(delay=0.03))
        await _say(s, b"I am stuck")
        await asyncio.sleep(0.15)
        await _say(s, b"where did the five go")  # echo
        await settle(s)
        echo_ignored = list(wire.said())
        brain.text = "Okay."
        await _say(s, b"I moved the five over")
        await settle(s)
        return s, wire, echo_ignored

    s, wire, echo_ignored = run(go())
    assert echo_ignored[-1] == "Then tell me."  # the reply was said to the end
    assert [h["content"] for h in s.history if h["role"] == "user"] == ["I am stuck", "I moved the five over"]
    assert wire.said()[-1] == "Okay."


def test_a_late_continuation_is_a_new_turn_not_a_merge_with_an_old_one():
    async def go():
        s, _wire = session(speech=PcmSpeech())
        t = [1000.0]
        s.clock = lambda: t[0]
        await s.on_speech_start()
        await s.on_speech_chunk(b"I am stuck")
        await s.on_speech_end()
        await settle(s)
        t[0] += 30
        await s.on_speech_start(continues=True)
        await s.on_speech_chunk(b"on the sign")
        await s.on_speech_end()
        await settle(s)
        return s

    s = run(go())
    assert [h["content"] for h in s.history if h["role"] == "user"] == ["I am stuck", "on the sign"]


class BigSpeech(FakeSpeech):
    def synthesize(self, text):
        yield b"\x00\x00" * 24000, 24000  # one second in one chunk


def test_tutor_audio_goes_down_in_short_slices():
    async def go():
        s, wire = session(speech=BigSpeech())
        await s.start()
        await settle(s)
        return wire

    assert run(go()).audio == 3  # 1 s in 0.4 s slices


# ---------------------------------------------------------------- C: a tutor, not just a checker


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def live_session(live=None, checks=(), brain=None):
    clock = Clock()
    state = {"live": live}
    s, wire = session(brain=brain or FakeBrain("Which part feels tricky?"), checks=checks)
    s.clock = clock
    s.services.live = lambda: state["live"]
    s._started = clock.t
    return s, wire, clock, state


PAUSED = {"phase": "paused", "present": True, "writing_ago_s": 50.0}


def test_stuck_student_gets_a_check_in_then_one_more_per_pause():
    async def go():
        s, wire, clock, state = live_session(live=PAUSED)
        clock.t += 15
        await s.tick()
        assert wire.said() == []  # not yet
        clock.t += 6
        await s.tick()
        await settle(s)
        first = list(wire.said())
        clock.t += 30
        await s.tick()  # too soon after the check-in
        await settle(s)
        n_soon = len(wire.said())
        clock.t += 20
        await s.tick()  # still stuck: one more, more concrete
        await settle(s)
        second_note = s.services.brain.messages[-1]["content"]
        clock.t += 120
        await s.tick()  # at most two per pause
        await settle(s)
        n_same = len(wire.said())
        state["live"] = {"phase": "writing", "present": True, "writing_ago_s": 0.0}
        await s.tick()  # they write again...
        state["live"] = PAUSED
        clock.t += 21
        await s.tick()  # ...and get stuck again: a new check-in
        await settle(s)
        return s, wire, first, n_soon, second_note, n_same

    s, wire, first, n_soon, second_note, n_same = run(go())
    assert first == ["Which part feels tricky?"]
    assert n_soon == 1 and n_same == 2 and len(wire.said()) == 3
    assert "still stuck" in second_note and "Do not give" in second_note
    note = s.services.brain.messages[-1]["content"]
    assert "without writing" in note and "Do not give" in note


def test_no_nudge_while_anyone_talks_after_a_hint_or_when_away_or_muted():
    async def go():
        s, wire, clock, state = live_session(live=PAUSED, checks=[wrong_page()])
        s.talking = True
        clock.t += 50
        await s.tick()
        s.talking = False
        await s.on_check(b"jpeg")  # a hint is spoken
        await settle(s)
        clock.t += 25
        await s.tick()  # within the grace after a hint
        state["live"] = {"phase": "away", "present": False, "writing_ago_s": None}
        clock.t += 100
        await s.tick()
        s.voice = False
        state["live"] = PAUSED
        await s.tick()  # muted: no nagging
        await settle(s)
        return wire

    assert run(go()).said() == ["What happened to the sign of the 5?"]


def test_live_state_falls_back_to_the_browser_detector_and_reaches_the_model():
    async def go():
        s, _wire, clock, state = live_session(live=None)
        await s.on_activity("writing")
        assert s.live_state()["phase"] == "writing"
        assert "Right now: the student is writing." in s._messages("")[0]["content"]
        clock.t += 10
        assert s.live_state() is None  # stale
        state["live"] = PAUSED
        assert "at the page but not writing" in s._messages("")[0]["content"]

    run(go())


def steps_page(*latex, gap=0.1):
    return Page(problem="Solve 2x + 5 = 11", steps=[
        Step(id=f"s{i + 1}", latex=t, bbox=(0.1, 0.1 + gap * i, 0.5, 0.06), status="ok", verified_by="sympy")
        for i, t in enumerate(latex)])


def test_a_correct_step_gets_a_short_word_and_a_hard_won_one_a_warmer_one():
    async def go():
        pages = [(steps_page("2x + 5 = 11"), None), (steps_page("2x + 5 = 11", "2x = 6"), None),
                 (steps_page("2x + 5 = 11", "2x = 6", "x = 3"), None),
                 (steps_page("2x + 5 = 11", "2x = 6", "x = 3", "3 = 3"), None)]
        s, wire, clock, _ = live_session(checks=pages)
        await s.on_check(b"a", "first_view")
        clock.t += 30  # a step that took a while
        await s.on_check(b"b", "line_done")
        await settle(s)
        first = list(wire.said())
        clock.t += 30  # another hard one, but the warm word was said lately: a short one
        await s.on_check(b"c", "line_done")
        await settle(s)
        clock.t += 2  # a quick one right after: nothing (not every few seconds)
        await s.on_check(b"d", "line_done")
        await settle(s)
        return wire, first

    wire, first = run(go())
    assert len(first) == 1 and first[0] in ENCOURAGEMENTS
    assert len(wire.said()) == 2 and wire.said()[1] in PRAISE


def test_a_crossed_out_step_gets_a_short_remark():
    async def go():
        s, wire, clock, _ = live_session(checks=[
            (steps_page("2x + 5 = 11", "2x = 6", "x = 3"), None),
            (steps_page("2x + 5 = 11", "x = 3", gap=0.2), None)])  # the middle step is gone, nothing in its place
        await s.on_check(b"a", "first_view")
        clock.t += 5
        await s.on_check(b"b", "ink_removed")
        await settle(s)
        return wire

    said = run(go()).said()
    assert len(said) == 1 and said[0] in SCRATCHED


def test_crossing_out_the_wrong_step_is_not_called_a_fix():
    async def go():
        page, finding = wrong_page()
        after = Page(problem=page.problem, steps=[page.steps[0]])  # the wrong step crossed out, nothing new
        s, wire, clock, _ = live_session(checks=[(page, finding), (after, None)])
        await s.on_check(b"a", "first_view")
        await settle(s)
        clock.t += 5
        await s.on_check(b"b", "ink_removed")
        await settle(s)
        return wire

    said = run(go()).said()
    assert said[0] == "What happened to the sign of the 5?"
    assert said[-1] in SCRATCHED and FIXED not in said


def test_a_reread_with_other_spacing_is_the_same_mistake_and_a_struck_step_is_noticed():
    async def go():
        page, finding = wrong_page()
        extra = Step(id="s3", latex="2x = 16", bbox=(0.1, 0.5, 0.5, 0.08))
        before = page.model_copy(update={"steps": [*page.steps, extra]})
        # Measured live: the re-read wrote "2x=11+5" for "2x = 11 + 5", and left the struck step out.
        reread = page.model_copy(update={"steps": [page.steps[0], page.steps[1].model_copy(
            update={"latex": "2x=11+5"})]})
        s, wire, clock, _ = live_session(checks=[(before, finding), (reread, finding)])
        await s.on_check(b"a", "first_view")
        await settle(s)
        clock.t += 5
        await s.on_check(b"b", "ink_removed")
        await settle(s)
        return wire

    said = run(go()).said()
    assert said[0] == "What happened to the sign of the 5?" and len(said) == 2 and said[1] in SCRATCHED


def test_page_changed_but_no_step_read_takes_one_look():
    async def go():
        brain = FakeBrain("Nice diagram. What does the arrow show?")
        same = steps_page("2x + 5 = 11")
        s, wire, clock, _ = live_session(checks=[(same, None), (same, None), (same, None)], brain=brain)
        await s.on_check(b"a", "first_view")
        clock.t += 5
        await s.on_check(b"diagram", "new_ink")
        await settle(s)
        looked = brain.messages[-1]["content"]
        clock.t += 10
        await s.on_check(b"diagram2", "new_ink")  # rate-limited
        await settle(s)
        return wire, looked

    wire, looked = run(go())
    assert wire.said() == ["Nice diagram.", "What does the arrow show?"]
    assert isinstance(looked, list) and looked[1]["type"] == "image_url" and "not a written" in looked[0]["text"]
