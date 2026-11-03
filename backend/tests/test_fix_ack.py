"""'That works now.' is said only for a real fix: the flagged step changed to something
that checks out, with the steps before it unchanged in meaning. A re-read that misreads
the step above (so the flagged step suddenly looks fine) is not a fix."""

from test_tutor import run, session, settle, wrong_page

from lens.checker import check_page
from lens.contract import Page, Step
from lens.tutor import FIXED, fix_confirmed

BOX1, BOX2, BOX3 = (0.1, 0.2, 0.5, 0.08), (0.1, 0.35, 0.5, 0.08), (0.1, 0.5, 0.5, 0.08)


def page(*lines: str, problem: str | None = "Solve 2x + 5 = 11") -> tuple[Page, object]:
    boxes = [BOX1, BOX2, BOX3, (0.1, 0.65, 0.5, 0.08)]
    return check_page(Page(problem=problem,
                           steps=[Step(id=f"s{i + 1}", latex=t, bbox=boxes[i]) for i, t in enumerate(lines)]))


def test_flagged_step_rewritten_correctly_is_a_fix():
    before, finding = wrong_page()
    after, none = page("2x + 5 = 11", "2x = 11 - 5")
    assert none is None and fix_confirmed(before, finding, after)


def test_step_above_misread_is_not_a_fix():
    before, finding = wrong_page()
    # The first step re-read as "2x - 5 = 11": now "2x = 11 + 5" follows, but nothing was fixed.
    after, none = page("2x - 5 = 11", "2x = 11 + 5", problem=None)
    assert none is None
    assert not fix_confirmed(before, finding, after)


def test_same_flagged_step_read_again_is_not_a_fix():
    before, finding = wrong_page()
    after = before.model_copy(update={"steps": [s.model_copy(update={"status": "ok"}) for s in before.steps]})
    assert not fix_confirmed(before, finding, after)


def test_replacement_that_cannot_be_checked_is_not_a_fix():
    before, finding = wrong_page()
    after, _ = page("2x + 5 = 11", "\\text{take the five over}")
    assert after.steps[1].status != "ok"
    assert not fix_confirmed(before, finding, after)


def test_linking_words_read_differently_do_not_block_a_fix():
    before, finding = wrong_page()
    after, none = page("2x + 5 = 11", "\\text{So,}", "2x = 11 - 5")
    assert none is None and fix_confirmed(before, finding, after)


def test_earlier_step_written_another_way_still_counts():
    before, finding = wrong_page()
    after, none = page("2 x+5=11", "2x = 6")
    assert none is None and fix_confirmed(before, finding, after)


def test_session_stays_quiet_when_the_step_above_was_misread():
    async def go():
        misread = page("2x - 5 = 11", "2x = 11 + 5", problem=None)
        s, wire = session(checks=[wrong_page(), misread])
        await s.on_check(b"jpeg1")
        await settle(s)
        await s.on_check(b"jpeg2")
        await settle(s)
        return wire

    wire = run(go())
    assert FIXED not in wire.said()
    assert wire.said() == ["What happened to the sign of the 5?"]


def test_session_praises_a_real_fix():
    async def go():
        s, wire = session(checks=[wrong_page(), page("2x + 5 = 11", "2x = 6")])
        await s.on_check(b"jpeg1")
        await settle(s)
        await s.on_check(b"jpeg2")
        await settle(s)
        return wire

    assert run(go()).said()[-1] == FIXED
