"""How Lens teaches: it says plainly whether a step is right, and its guidance on an
open mistake gets clearer each time the student is still stuck (never the fix)."""

import asyncio

from test_tutor import FakeBrain, FakeSpeech, Wire, fixed_page, wrong_page

from lens.tutor import Services, TutorSession, page_summary


def session():
    w = Wire()
    s = TutorSession(Services(brain=FakeBrain(), speech=FakeSpeech(), check=lambda *a: None), w.send, w.send_bytes)
    return s


def test_page_summary_says_which_steps_are_right():
    page, _ = fixed_page()
    text = page_summary(page)
    assert "checked: correct" in text and "not checked yet" not in text


def test_the_copied_question_is_not_called_unchecked():
    from lens.contract import Page, Step

    page = Page(problem="Solve 3x + 4 = 19", steps=[Step(id="s1", latex="3x + 4 = 19", bbox=(0, 0, 1, 0.1),
                                                         status="unverifiable")])
    assert "the problem as given" in page_summary(page)


def test_prompt_tells_the_model_to_be_direct():
    s = session()
    s.page, s.finding = fixed_page()
    system = s._messages("")[0]["content"]
    assert "Never say you cannot tell them whether they are right" in system


def test_guidance_level_rises_while_the_student_is_stuck_and_resets_on_a_new_hint():
    async def run():
        s = session()
        s.page, s.finding = wrong_page()
        levels = []
        for text in ("I don't get it", "still lost", "no idea", "help"):
            await s.on_text(text)
            await s._speaking
            levels.append(next(l for l in s._messages("")[0]["content"].splitlines() if l.startswith("Guidance")))
        assert [l.split()[2] for l in levels] == ["1", "2", "3", "3"]
        assert s._guided == 4
        return s

    s = asyncio.run(run())
    assert "level" in s._messages("")[0]["content"]


def test_no_guidance_line_without_a_mistake():
    s = session()
    s.page, s.finding = fixed_page()
    assert "Guidance: level" not in s._messages("")[0]["content"]
