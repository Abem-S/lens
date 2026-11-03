"""The steps list follows the paper: the reading shows before it is verified, the last
check goes along with a page check, and a new problem (said, or seen) starts the list
afresh without breaking "already hinted" or "that works now"."""

import asyncio

from test_tutor import FakeBrain, FakeSpeech, Wire, fixed_page, run, settle, wrong_page

from lens.attempts import current_attempt, forget_old, moved_on, other_problem, states_question
from lens.checker import check_page
from lens.contract import Page, Step
from lens.tutor import FIXED, Services, TutorSession


def other_page():
    p = Page(problem="Solve 3x - 1 = 8", steps=[
        Step(id="s1", latex="3x - 1 = 8", bbox=(0.1, 0.2, 0.5, 0.08)),
        Step(id="s2", latex="3x = 9", bbox=(0.1, 0.35, 0.5, 0.08)),
        Step(id="s3", latex="x = 3", bbox=(0.1, 0.5, 0.5, 0.08)),
    ])
    return check_page(p)


def other_wrong():
    p = Page(problem="Solve 3x - 1 = 8", steps=[
        Step(id="s1", latex="3x - 1 = 8", bbox=(0.1, 0.2, 0.5, 0.08)),
        Step(id="s2", latex="3x = 7", bbox=(0.1, 0.35, 0.5, 0.08)),
    ])
    p, f = check_page(p)
    return p, f.model_copy(update={"hint": "What happened to the one?"})


def other_fixed():
    p = Page(problem="Solve 3x - 1 = 8", steps=[
        Step(id="s1", latex="3x - 1 = 8", bbox=(0.1, 0.2, 0.5, 0.08)),
        Step(id="s2", latex="3x = 9", bbox=(0.1, 0.35, 0.5, 0.08)),
    ])
    return check_page(p)


def recording_session(results, on_read_pages=None):
    """Page checks return `results` in turn; each call's keyword arguments are recorded,
    and `on_read` is called with the page first (as the pipeline does)."""
    wire = Wire()
    results = list(results)
    calls = []

    def check(jpeg, problem, **kw):
        calls.append(kw)
        page, finding = results.pop(0)
        if kw.get("on_read"):
            kw["on_read"](page.model_copy(update={"steps": [
                s.model_copy(update={"status": "unchecked", "verified_by": None}) for s in page.steps]}))
        return page, finding

    services = Services(brain=FakeBrain(), speech=FakeSpeech(), check=check)
    return TutorSession(services, wire.send, wire.send_bytes), wire, calls


def test_steps_show_before_they_are_verified():
    async def go():
        s, wire, _ = recording_session([wrong_page()])
        await s.on_check(b"a", "first_view")
        await asyncio.sleep(0.05)
        return wire

    wire = run(go())
    kinds = [m["type"] for m in wire.msgs if m["type"] in ("steps", "result")]
    assert kinds == ["steps", "result"]
    assert all(st["status"] == "unchecked" for st in wire.of("steps")[0]["page"]["steps"])
    assert wire.of("steps")[0].get("finding") is None


def test_page_check_gets_the_last_result():
    async def go():
        s, _wire, calls = recording_session([wrong_page(), wrong_page()])
        await s.on_check(b"a")
        await settle(s)
        await s.on_check(b"b")
        return calls

    calls = run(go())
    assert calls[0]["prior"].page is None
    assert calls[1]["prior"].finding.step_id == "s2" and len(calls[1]["prior"].page.steps) == 2


def test_saying_next_question_clears_the_list_and_forgets_the_old_steps():
    async def go():
        s, wire, calls = recording_session([wrong_page(), other_wrong()])
        await s.on_check(b"a")
        await settle(s)
        await s.on_text("OK, let's do the next question")
        await settle(s)
        cleared = wire.of("result")[-1]
        await s.on_check(b"b")
        await settle(s)
        return s, wire, calls, cleared

    s, wire, calls, cleared = run(go())
    assert cleared["page"]["steps"] == [] and cleared["finding"] is None
    assert set(calls[1]["prior"].forget) == {"2x+5=11", "2x=11+5"} and calls[1]["prior"].page is None
    # The new problem's mistake gets its question; the old one's stays in the summary as it was.
    assert wire.said()[-1] == "What happened to the one?"
    summary = s.summary()
    assert summary["steps_read"] == 4 and [m["status"] for m in summary["mistakes"]] == ["open", "open"]


def test_another_method_is_not_a_new_problem():
    async def go():
        s, _wire, _ = recording_session([wrong_page()])
        await s.on_check(b"a")
        await settle(s)
        await s.on_text("Can I try another way?")
        await settle(s)
        return s

    assert len(run(go()).page.steps) == 2


def test_a_different_page_starts_afresh_without_a_fix_remark():
    async def go():
        s, wire, _ = recording_session([wrong_page(), other_page()])
        await s.on_check(b"a")
        await settle(s)
        await s.on_check(b"b", "page_changed")
        await settle(s)
        return s, wire

    s, wire = run(go())
    assert FIXED not in wire.said()  # the old mistake was not fixed; the page changed
    assert wire.of("check_timing")[-1].get("new_problem") is True
    assert [st.latex for st in s.page.steps] == ["3x - 1 = 8", "3x = 9", "x = 3"]
    assert s.summary()["mistakes"][0]["status"] == "open"


def test_hints_and_fix_still_work_after_a_new_problem():
    async def go():
        s, wire, _ = recording_session([wrong_page(), other_wrong(), other_wrong(), other_fixed()])
        await s.on_check(b"a")
        await settle(s)
        await s.on_text("I'm starting problem 2 now")
        await settle(s)
        for jpeg in (b"b", b"c", b"d"):
            await s.on_check(jpeg)
            await settle(s)
        return wire

    said = run(go()).said()
    assert said.count("What happened to the one?") == 1  # hinted once, not repeated on the re-check
    assert said[-1] == FIXED


def test_check_started_before_a_new_problem_is_dropped():
    async def go():
        wire = Wire()
        gate = asyncio.Event()

        def check(jpeg, problem, **kw):
            asyncio.run_coroutine_threadsafe(gate.wait(), loop).result(2)
            return wrong_page()

        loop = asyncio.get_running_loop()
        s = TutorSession(Services(brain=FakeBrain(), speech=FakeSpeech(), check=check), wire.send, wire.send_bytes)
        s.page, s.finding = fixed_page()
        task = asyncio.create_task(s.on_check(b"a"))
        await asyncio.sleep(0.05)
        await s.on_text("new problem")
        gate.set()
        await task
        await settle(s)
        return s, wire

    s, wire = run(go())
    assert s.page is None and wire.of("result")[-1]["page"]["steps"] == []


# ---------------------------------------------------------------- lens.attempts on its own


def _steps(*xs):
    return [Step(id=f"s{i + 1}", latex=x, bbox=(0.1, 0.1 * i, 0.5, 0.05)) for i, x in enumerate(xs)]


def test_restart_rules():
    kept, gone = current_attempt(_steps("2x + 5 = 11", "2x = 16", "x = 8", "2x + 5 = 11", "2x = 6"), None)
    assert [s.latex for s in kept] == ["2x + 5 = 11", "2x = 6"] and gone == ["s1", "s2", "s3"]
    # The line just above read again, and a factorised form of the line above, are not restarts.
    kept, _ = current_attempt(_steps("x^2 - 5x + 6 = 0", "(x-2)(x-3) = 0", "(x-2)(x-3) = 0"), None)
    assert len(kept) == 3
    kept, _ = current_attempt(_steps("2x = 6", "x = 3", "\\text{Q2. Solve } 3x - 1 = 8", "3x = 9"), None)
    assert [s.latex for s in kept] == ["\\text{Q2. Solve } 3x - 1 = 8", "3x = 9"]


def test_restart_rules_on_real_readings():
    # Spark's readings of two FERMAT pages (2026-11-01). The question copied under a
    # "Question:" label is not a restart; "We can also do it as:" and the question copied
    # again is a second method, and the first one leaves the list.
    problem = "\\text{Question: Prove that } \\cos^2 x + \\cos^2 (x) = \\frac{3}{2}."
    kept, gone = current_attempt(_steps("\\text{Question: Prove that}", "\\cos^2 x + \\cos^2 (x) = \\frac{3}{2}.",
                                        "\\text{Answer: We have}", "\\text{L.H.S} = 1"), problem)
    assert gone == []
    q = "-\\frac{4}{5} \\times \\frac{3}{7}"
    kept, gone = current_attempt(_steps("\\text{Question : find,}", q, "\\text{Answer : We have,}", q,
                                        "= \\frac{-12}{35}", "= \\frac{1}{2}", "\\text{We can also do it as :}", q,
                                        "= \\frac{-3}{4} \\times 1"), None)
    assert [s.latex for s in kept] == ["\\text{Question : find,}", q, "= \\frac{-3}{4} \\times 1"]


def test_question_and_problem_detection():
    assert states_question("\\text{Solve } x^2 = 4") and states_question("Q3") and states_question("2) Find x")
    assert not states_question("x = 3") and not states_question("\\text{so } x = 3")
    assert other_problem("Solve 2x + 5 = 11", "Solve 3x - 1 = 8")
    assert not other_problem("Solve 2x + 5 = 11", "Solve 2x+5=11.") and not other_problem(None, "Solve x")
    assert not other_problem("Solve the following equation for x: 2x + 5 = 11", "2x + 5 = 11")
    assert not other_problem("Solve for x, given that 2x + 5 = 11", "Find x when 2x+5 = 11")
    assert not other_problem("Solve 2x + 5 = 11 and check the answer", "Solve 2x + 5 = 17 and check the answer")


def test_old_problem_steps_at_the_top_are_left_out():
    steps = _steps("2x + 5 = 11", "2x = 6", "x = 3", "3x - 1 = 8", "3x = 9")
    assert [s.latex for s in forget_old(steps, ["2x+5=11", "2x=6", "x=3"])] == ["3x - 1 = 8", "3x = 9"]
    # One misread inside the old run is tolerated.
    steps = _steps("2x + 5 = 11", "2x = b", "x = 3", "3x - 1 = 8")
    assert [s.latex for s in forget_old(steps, ["2x+5=11", "2x=6", "x=3"])] == ["3x - 1 = 8"]


def test_a_swaying_page_is_not_a_new_page():
    before = Page(problem="Solve 2x + 5 = 11", steps=_steps("2x + 5 = 11", "2x = 6", "x = 3"))
    swayed = Page(problem="Solve 2x+5=11", steps=[s.model_copy(update={"bbox": (0.2, s.bbox[1] + 0.1, 0.5, 0.05)})
                                                  for s in before.steps])
    assert not moved_on(before, swayed)
    assert moved_on(before, Page(problem=None, steps=_steps("y^2 = 9", "y = 3", "y = -3")))
    assert moved_on(before, Page(problem="Find the area of the circle", steps=_steps("A = \\pi r^2")))
