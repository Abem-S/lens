"""The private problem notes, the run of camera pictures for a look, and "where is this heading?"."""

import asyncio
import json

from test_tutor import FakeBrain, run, session, settle
from test_tutor_live import live_session, steps_page

from lens import notes as lens_notes


class NotesBrain:
    """Answers the notes call with fixed JSON and records what it was given."""

    def __init__(self, data):
        self.data, self.seen = data, []

    def stream(self, messages, **kw):
        self.seen.append(messages)
        yield "```json\n" + json.dumps(self.data) + "\n```"


def test_notes_update_keeps_givens_and_tolerates_bad_output():
    brain = NotesBrain({"problem": "A ball is thrown up at 20 m/s from a 45 m cliff", "asked": "time to land",
                        "plan": "height equation, set to zero, solve", "approach": None, "progress": None,
                        "on_track": None, "why_off": None})
    out = lens_notes.update(brain, lens_notes.empty(), [{"role": "user", "content": "a ball from a cliff"}],
                            "(nothing read yet)", "the student said something")
    assert out["problem"].startswith("A ball") and out["asked"] == "time to land"
    assert "Student: a ball from a cliff" in brain.seen[0][1]["content"]
    # A reply that forgets the problem does not wipe it.
    again = lens_notes.update(NotesBrain({"problem": None, "progress": "wrote the height equation"}), out, [], "", "x")
    assert again["problem"] == out["problem"] and again["progress"] == "wrote the height equation"

    class Broken:
        def stream(self, messages, **kw):
            yield "sorry, no json"

    assert lens_notes.update(Broken(), out, [], "", "x") == out
    assert "Problem (with its givens): A ball" in lens_notes.render(out)


def test_notes_reach_every_reply_and_survive_a_long_talk():
    async def go():
        s, _wire = session()
        calls = []

        def notes(n, talk, page, event):
            calls.append(event)
            return {**lens_notes.empty(), "problem": "Solve 3x - 7 = 11", "asked": "x",
                    "plan": "undo the subtraction, then the multiplication"}

        s.services.notes = notes
        await s.on_text("I'm solving three x minus seven equals eleven")
        await settle(s)
        await asyncio.sleep(0.05)
        for i in range(15):  # a long conversation: the opening falls out of the recent turns
            await s.on_text(f"okay thinking about it number {i}")
            await settle(s)
        system = s._messages("")[0]["content"]
        return s, calls, system

    s, calls, system = run(go())
    assert calls and "Solve 3x - 7 = 11" in system and "Your teaching plan" in system
    assert all("three x minus seven" not in str(t["content"]) for t in s.history[-20:])


def test_next_question_clears_the_notes():
    async def go():
        s, _wire = session()
        s.notes = {"problem": "Solve 3x - 7 = 11"}
        await s.on_text("ok let's do the next question")
        await settle(s)
        return s

    s = run(go())
    assert "3x - 7" not in s._messages("")[0]["content"]


def test_a_look_sends_the_pictures_from_the_whole_turn_in_order():
    async def go():
        brain = FakeBrain("Two, then four, then six.")
        s, _wire, clock, _ = live_session(brain=brain)
        await s.on_speech_start()
        for i in range(10):  # a picture every half second while they talk
            await s.on_frame(f"f{i}".encode())
            clock.t += 0.5
        s._turn_end = clock.t
        s._turn_start = clock.t - 5.0
        await s.on_frame(b"after")  # one more just after they stopped
        images = await s._look_frames("tell me what numbers I'm showing with my fingers")
        msgs = s._messages("", images)
        return images, msgs

    images, msgs = run(go())
    assert len(images) == 6 and images[0] == b"f0" and images[-1] == b"after"
    assert msgs[0]["role"] == "system"  # no user turn yet: the pictures go with the student's turn


def test_finger_and_number_questions_look_but_step_questions_stay_fast():
    from lens.tutor import _VISUAL

    for q in ("tell me what numbers I'm showing", "how many fingers am I holding up", "watch this",
              "what am I doing", "can you see my diagram"):
        assert _VISUAL.search(q), q
    for q in ("is this step right", "what should I do next", "I divided both sides by two"):
        assert not _VISUAL.search(q), q


def test_a_spoken_look_attaches_several_labelled_pictures():
    async def go():
        brain = FakeBrain("I saw two, four and six.")
        s, _wire = session(brain=brain)
        s.page = steps_page("2x + 5 = 11")
        await s.on_speech_start()
        for i in range(4):
            await s.on_frame(f"f{i}".encode())
        s._turn_end = s.clock()
        await s.on_text("how many fingers am I showing")
        await settle(s)
        return brain

    brain = run(go())
    content = brain.messages[-1]["content"]
    images = [p for p in content if p["type"] == "image_url"]
    assert len(images) >= 2 and any("time order" in p.get("text", "") for p in content)


def test_off_track_notes_ask_where_it_is_heading_once():
    async def go():
        brain = FakeBrain("What are you trying to find with this step?")
        s, wire, _clock, _ = live_session(brain=brain, checks=[(steps_page("2x + 5 = 11", "2x = 6"), None)])
        s.page = steps_page("2x + 5 = 11")
        s.services.notes = lambda n, talk, page, event: {
            **lens_notes.empty(), "problem": "Solve 2x + 5 = 11", "on_track": False,
            "why_off": "they are finding y, which the problem does not ask for", "progress": "step 2"}
        await s.on_check(b"b", "line_done")
        await settle(s)
        await s._notes_task
        await settle(s)
        n = len(wire.said())
        s._direction_at -= 1000  # even past the cooldown, the same progress is not asked about twice
        s._update_notes("again")
        await s._notes_task
        await settle(s)
        return brain, wire, n

    brain, wire, n = run(go())
    assert any("where this is heading" in str(m.get("content")) for m in brain.messages)
    assert wire.said()[-1] == "What are you trying to find with this step?" and len(wire.said()) == n
