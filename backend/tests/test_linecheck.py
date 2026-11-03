"""The fast check for one just-finished line: only that line is read (twice), it is
checked after the steps already known, and a mistake is claimed only when both
readings show it (rule 4). Anything it cannot place falls back to a page read."""

import json
import threading

from test_lines import TEXTS, notebook

from lens.checker import check_page
from lens.contract import Finding, Page, Step
from lens.linecheck import check_line, locate_new_line
from lens.lines import find_lines, snap_steps
from lens.pointing import bbox_outline
from lens.spark import CallLog, Reply


class Script:
    """Answers by purpose (a value, or a list used in turn); thread-safe; records what was asked."""

    def __init__(self, **answers):
        self.answers, self.asked, self.prompts = answers, [], {}
        self.lock = threading.Lock()

    def ask(self, prompt, **kw):
        p = kw["purpose"]
        with self.lock:
            self.asked.append(p)
            self.prompts.setdefault(p, []).append(prompt)
            a = self.answers[p]
            text = a.pop(0) if isinstance(a, list) else a
        return Reply(text, CallLog(p, kw.get("effort")))


def lines_json(*lines):
    return json.dumps({"lines": list(lines)})


def setup(texts=TEXTS, known=3):
    """A notebook page with `texts`; the session has read the first `known` lines."""
    jpeg, truth = notebook(texts)
    steps = [Step(id=f"s{i + 1}", latex=t, bbox=truth[i]) for i, t in enumerate(texts[:known])]
    page, _ = check_page(Page(problem="Solve 2x + 5 = 11", steps=steps))
    return jpeg, truth, page


def pen_box(b):
    return (b[0] - 0.025, b[1] - 0.025, b[2] + 0.05, b[3] + 0.05)


def test_new_line_is_found_under_the_pen():
    jpeg, truth, page = setup()
    located = locate_new_line(jpeg, page, pen_box(truth[3]))
    assert located is not None and not located.replaced
    _, y, _, h = located.region
    assert abs((y + h / 2) - (truth[3][1] + truth[3][3] / 2)) < 0.01
    assert all(abs(s.bbox[1] - t[1]) < 0.01 for s, t in zip(located.old, truth[:3]))


def test_a_correct_line_is_ok_after_one_reading():
    jpeg, truth, page = setup()
    spark = Script(read_line=lines_json("x = 3"), read_line_b=lines_json("x = 3"))
    timings = {}
    page2, finding = check_line(spark, jpeg, page, pen_box(truth[3]), timings=timings)
    assert finding is None
    assert [s.latex for s in page2.steps] == TEXTS
    new = page2.steps[-1]
    assert new.status == "ok" and new.verified_by == "sympy" and new.id == "s4"
    assert abs(new.bbox[1] - truth[3][1]) < 0.02  # boxed where it was written
    assert "read_page" not in spark.asked and "hint" not in spark.asked
    assert timings["total_s"] >= timings["locate_s"]


def test_a_mistake_both_readings_show_is_claimed_with_a_hint():
    jpeg, truth, page = setup()
    spark = Script(read_line=lines_json("x = 4"), read_line_b=lines_json("x=4"),
                   hint="What do you get when you divide both sides by 2?")
    page2, finding = check_line(spark, jpeg, page, pen_box(truth[3]), mask_for=lambda img, box: None)
    assert finding is not None and finding.step_id == "s4"
    assert page2.step("s4").status == "error" and page2.step("s4").verified_by == "sympy"
    assert finding.hint.startswith("What do you get") and finding.mask
    assert sorted(spark.asked) == ["hint", "hint", "read_line", "read_line_b"]  # two hint drafts race


def test_a_mistake_only_one_reading_shows_is_not_claimed():
    jpeg, truth, page = setup()
    spark = Script(read_line=lines_json("x = 4"), read_line_b=lines_json("x = 3"), hint="unused?")
    page2, finding = check_line(spark, jpeg, page, pen_box(truth[3]))
    # Whichever reading came first, nothing is claimed: either it read "x = 3" (fine,
    # nothing to wait for) or it read "x = 4" and the other reading did not agree.
    assert finding is None
    assert page2.steps[-1].status in ("ok", "unverifiable")


def test_a_mistake_read_two_ways_is_not_claimed():
    # Both readings show a mistake on the new line, but not the same digits: one of
    # them misread it, so neither reading is the student's mistake for sure.
    jpeg, truth, page = setup()
    spark = Script(read_line=lines_json("x = 4"), read_line_b=lines_json("x = 9"), hint="unused?")
    page2, finding = check_line(spark, jpeg, page, pen_box(truth[3]))
    assert finding is None
    assert page2.steps[-1].status == "unverifiable"


def test_both_readings_failing_asks_for_a_page_read():
    jpeg, truth, page = setup()
    spark = Script(read_line="garbled", read_line_b="{}")
    assert check_line(spark, jpeg, page, pen_box(truth[3])) is None


def test_writing_above_earlier_work_asks_for_a_page_read():
    jpeg, truth, page = setup(known=4)
    page = Page(problem=page.problem, steps=[page.steps[i] for i in (0, 2, 3)])  # line 2 is "new"
    spark = Script(read_line=lines_json("2x = 11 - 5"), read_line_b=lines_json("2x = 11 - 5"))
    assert check_line(spark, jpeg, page, pen_box(truth[1])) is None
    assert spark.asked == []


def test_words_go_to_the_judge_only_when_both_readings_agree():
    texts = ["2x + 5 = 11", "2x = 6", "x = 3", "so x is three"]
    jpeg, truth, page = setup(texts)
    spark = Script(read_line=lines_json("\\text{so x is three}"), read_line_b=lines_json("\\text{so x is thre}"),
                   judge_find=json.dumps({"line": None, "kind": ""}))
    page2, finding = check_line(spark, jpeg, page, pen_box(truth[3]))
    assert finding is None and page2.steps[-1].status == "ok" and page2.steps[-1].verified_by == "model"
    spark = Script(read_line=lines_json("\\text{so x is three}"), read_line_b=lines_json("x = 3"),
                   judge_find=json.dumps({"line": None, "kind": ""}))
    page3, finding = check_line(spark, jpeg, page, pen_box(truth[3]))
    assert finding is None and page3.steps[-1].status == "unverifiable"


def test_open_mistake_stays_until_its_line_is_rewritten():
    texts = ["2x + 5 = 11", "2x = 16", "x = 8"]
    jpeg, truth, page = setup(texts, known=2)
    page, finding = check_page(page)
    assert finding.step_id == "s2"
    spark = Script(read_line=lines_json("x = 8"), read_line_b=lines_json("x = 8"))
    page2, f2 = check_line(spark, jpeg, page, pen_box(truth[2]), previous=finding)
    # Still the first mistake; nothing new is said.
    assert (f2.step_id, f2.kind, f2.hint) == (finding.step_id, finding.kind, finding.hint)
    # Its outline is redrawn on this frame, around the step where it is found now (not the old frame's).
    assert f2.mask == bbox_outline(page2.step(finding.step_id).bbox)


def test_rewriting_the_wrong_line_below_it_counts_as_the_fix():
    texts = ["2x + 5 = 11", "2x = 16", "2x = 6"]
    jpeg, truth, page = setup(texts, known=2)
    page, finding = check_page(page)
    spark = Script(read_line=lines_json("2x = 6"), read_line_b=lines_json("2x = 6"))
    page2, f2 = check_line(spark, jpeg, page, pen_box(truth[2]), previous=finding)
    assert f2 is None
    assert [s.latex for s in page2.steps] == ["2x + 5 = 11", "2x = 6"]


def test_the_line_above_read_again_is_not_a_new_step():
    jpeg, truth, page = setup()
    spark = Script(read_line=lines_json("2x = 6", "x = 3"), read_line_b=lines_json("x = 3"))
    page2, finding = check_line(spark, jpeg, page, pen_box(truth[3]))
    assert [s.latex for s in page2.steps] == TEXTS and finding is None


def test_snapped_boxes_feed_the_line_check():
    jpeg, truth, _ = setup()
    rough = [Step(id=f"s{i + 1}", latex=t, bbox=(b[0], b[1] + 0.05, b[2], b[3] * 2)) for i, (t, b) in
             enumerate(zip(TEXTS[:3], truth))]
    snapped = snap_steps(rough, find_lines(jpeg))
    page, _ = check_page(Page(steps=snapped))
    spark = Script(read_line=lines_json("x = 3"), read_line_b=lines_json("x = 3"))
    page2, finding = check_line(spark, jpeg, page, pen_box(truth[3]))
    assert finding is None and len(page2.steps) == 4


def test_previous_finding_type_is_kept():
    f = Finding(step_id="s2", kind="k", hint="h")
    assert f.model_copy() == f


# ---------------------------------------------------------------- why it falls back; what is on the paper now


def test_fallback_says_why():
    jpeg, truth, page = setup(known=4)
    page = Page(problem=page.problem, steps=[page.steps[i] for i in (0, 2, 3)])
    timings: dict = {}
    assert check_line(Script(), jpeg, page, pen_box(truth[1]), timings=timings) is None
    assert timings["fallback"].startswith("new writing above earlier steps")


def test_a_second_column_beside_earlier_work_is_not_above_it():
    # A fresh method started in the right-hand column, level with the first steps: it is
    # not "above" the steps lower down in the left column.
    texts = ["x^2 - 5x + 6 = 0", "x = 1.5", "x = 2.5", "x = 4"]
    jpeg, truth = notebook(texts, extra=((560, 80 + 44 - 34, "(x-2)(x-3) = 0"),))
    steps = [Step(id=f"s{i + 1}", latex=t, bbox=truth[i]) for i, t in enumerate(texts)]
    page, _ = check_page(Page(problem=None, steps=steps))
    right = (560 / 900 - 0.02, (80 + 44 - 34) / 1200 - 0.01, 0.38, 0.04)
    spark = Script(read_line=lines_json("(x-2)(x-3) = 0"), read_line_b=lines_json("(x-2)(x-3)=0"))
    timings: dict = {}
    out = check_line(spark, jpeg, page, right, timings=timings)
    assert out is not None and "fallback" not in timings
    assert out[0].steps[-1].latex == "(x-2)(x-3) = 0"


def test_a_new_page_is_not_added_to_the_old_one():
    jpeg, truth, _ = setup(texts=["y = 7"], known=0)
    old = Page(problem=None, steps=[Step(id=f"s{i + 1}", latex=t, bbox=(0.1, 0.1 + 0.07 * i, 0.4, 0.03))
                                    for i, t in enumerate(TEXTS)])
    timings: dict = {}
    assert check_line(Script(), jpeg, old, pen_box(truth[0]), timings=timings) is None
    assert timings["fallback"].startswith("page changed")


def test_a_struck_through_step_leaves_the_list():
    jpeg, truth, page = setup(known=4)
    spark = Script(read_line=lines_json(), read_line_b=lines_json())  # crossed out: nothing left to read
    timings: dict = {}
    page2, finding = check_line(spark, jpeg, page, pen_box(truth[2]), timings=timings)
    assert [s.latex for s in page2.steps] == [TEXTS[0], TEXTS[1], TEXTS[3]] and finding is None
    assert timings["struck"] == ["s3"]


def test_a_copied_earlier_step_starts_the_attempt_again():
    texts = ["2x + 5 = 11", "2x = 16", "x = 8", "2x + 5 = 11"]  # wrong path, then back to the start
    jpeg, truth, page = setup(texts, known=3)
    page, finding = check_page(page)
    spark = Script(read_line=lines_json("2x + 5 = 11"), read_line_b=lines_json("2x+5=11"))
    page2, f2 = check_line(spark, jpeg, page, pen_box(truth[3]), previous=finding)
    assert [s.latex for s in page2.steps] == ["2x + 5 = 11"] and f2 is None


def test_a_new_question_is_read_with_the_page():
    jpeg, truth, page = setup()
    spark = Script(read_line=lines_json("\\text{Q2. Solve } 3x - 1 = 8"), read_line_b=lines_json("x"))
    timings: dict = {}
    assert check_line(spark, jpeg, page, pen_box(truth[3]), timings=timings) is None
    assert timings["fallback"] == "a new question"
