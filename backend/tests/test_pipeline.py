"""The shared page pipeline: a SymPy error caused by a misreading is withdrawn
after the re-read; a real one survives and gets a hint and an outline."""

import io
import json
import threading

from PIL import Image

from lens.contract import Finding, Page, Step
from lens.pipeline import Prior, check_image
from lens.spark import CallLog, Reply


def jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), "white").save(buf, "JPEG")
    return buf.getvalue()


PAGE = json.dumps({"problem": "Solve 2x + 5 = 11", "steps": [
    {"latex": "2x + 5 = 11", "bbox": [0.1, 0.1, 0.5, 0.1]},
    {"latex": "2x = 16", "bbox": [0.1, 0.25, 0.4, 0.1]},
]})


class Script:
    """Answers by purpose, records what was asked."""

    def __init__(self, **answers):
        self.answers, self.asked, self.prompts = answers, [], {}

    def ask(self, prompt, **kw):
        self.asked.append(kw["purpose"])
        self.prompts[kw["purpose"]] = prompt
        return Reply(self.answers[kw["purpose"]], CallLog(kw["purpose"], kw.get("effort")))


def lines(*xs):
    return json.dumps({"lines": list(xs)})


def test_misread_line_is_withdrawn_when_both_rereads_agree():
    spark = Script(read_page=PAGE, reread=lines("2x + 5 = 11", "2x = 6"), reread_b=lines("2x+5=11", "2x=6"))
    page, finding = check_image(spark, jpeg())
    assert finding is None
    assert page.steps[1].latex == "2x = 6" and page.steps[1].status == "ok"
    # The hint is drafted while the lines are re-read; with no claim it is never used.
    assert [a for a in spark.asked if a != "hint"] == ["read_page", "reread", "reread_b"]


def test_third_reading_that_matches_the_page_confirms():
    spark = Script(read_page=PAGE, reread=lines("2x + 5 = 11", "2x = 6"), reread_b=lines("2x + 5 = 11", "2x = 16"))
    _, finding = check_image(spark, jpeg(), hint=False)
    assert finding is not None and finding.step_id == "s2"


def test_no_two_readings_agreeing_claims_nothing():
    spark = Script(read_page=PAGE, reread=lines("2x + 5 = 11", "2x = 6"), reread_b=lines("2x + 5 = 11", "2x = 18"))
    page, finding = check_image(spark, jpeg(), hint=False)
    assert finding is None
    assert page.steps[1].status == "unverifiable"


def test_reread_lines_are_matched_by_likeness_not_position():
    # The crop slipped: the re-read also shows the line above and below, out of step.
    spark = Script(read_page=PAGE, reread=lines("\\text{Solve}", "2x + 5 = 11", "2x = 16", "x = 8"))
    _, finding = check_image(spark, jpeg(), hint=False)
    assert finding is not None and finding.step_id == "s2"
    assert spark.asked == ["read_page", "reread"]


def test_reread_of_another_line_confirms_nothing():
    spark = Script(read_page=PAGE, reread=lines("\\text{Therefore the answer is}"), reread_b=lines("y^2"))
    _, finding = check_image(spark, jpeg(), hint=False)
    assert finding is None


def test_trace_records_the_rereads():
    spark = Script(read_page=PAGE, reread=lines("2x + 5 = 11", "2x = 16"))
    trace: dict = {}
    check_image(spark, jpeg(), hint=False, trace=trace)
    assert trace["outcome"] == "confirmed"
    assert trace["rereads"][0]["lines"][1] == ["s2", "2x = 16", "2x = 16"]


def test_a_wrong_chain_needs_only_its_own_line_read_twice():
    page = json.dumps({"problem": None, "steps": [
        {"latex": "\\text{(ii)}", "bbox": [0.1, 0.1, 0.2, 0.1]},
        {"latex": "3 \\times 4 = 13", "bbox": [0.1, 0.25, 0.4, 0.1]},
    ]})
    spark = Script(read_page=page, reread=lines("3 \\cdot 4 = 13"),  # the label line is not found again
                   judge_find=json.dumps({"line": None}))
    _, finding = check_image(spark, jpeg(), hint=False)
    assert finding is not None and finding.step_id == "s2"


def test_real_error_survives_reread_and_gets_hint_and_outline():
    spark = Script(read_page=PAGE, reread=json.dumps({"lines": ["2x + 5 = 11", "2x = 16"]}),
                   hint="When the 5 crossed the equals sign, what did you do with it?")
    _page, finding = check_image(spark, jpeg(), mask_for=lambda img, box: None)
    assert finding is not None and finding.step_id == "s2"
    assert finding.hint.startswith("When the 5")
    assert finding.mask and len(finding.mask) >= 8  # box outline fallback
    assert sorted(spark.asked) == ["hint", "hint", "read_page", "reread"]  # two hint drafts race the re-read


def test_unreadable_reread_claims_nothing():
    spark = Script(read_page=PAGE, reread="no idea", reread_b="no idea")
    _, finding = check_image(spark, jpeg(), hint=False)
    assert finding is None


def test_reread_is_blind():
    spark = Script(read_page=PAGE, reread="no idea", reread_b="no idea")
    check_image(spark, jpeg(), hint=False)
    for purpose in ("reread", "reread_b"):
        prompt = spark.prompts[purpose][0]["text"]
        assert "2x = 16" not in prompt and "first read as" not in prompt


def test_model_judge_needs_confirmation():
    words = json.dumps({"problem": None, "steps": [
        {"latex": "\\\\text{Mass is } 3\\\\,\\\\mathrm{kg}", "bbox": [0.1, 0.1, 0.5, 0.1]},
        {"latex": "\\\\text{so it is } 30\\\\,\\\\mathrm{g}", "bbox": [0.1, 0.25, 0.5, 0.1]},
    ]})
    refused = Script(read_page=words, judge_find=json.dumps({"line": 2, "kind": "unit conversion"}),
                     judge_confirm=json.dumps({"wrong": False, "why": "unsure"}))
    assert check_image(refused, jpeg(), hint=False)[1] is None
    same_lines = json.dumps({"lines": [st["latex"] for st in json.loads(words)["steps"]]})
    agreed = Script(read_page=words, judge_find=json.dumps({"line": 2, "kind": "unit conversion"}),
                    judge_confirm=json.dumps({"complete": True, "wrong": True, "recovered": False,
                                              "side_remark": False, "why": "3 kg is 3000 g"}), reread=same_lines)
    page, finding = check_image(agreed, jpeg(), hint=False)
    assert finding.step_id == "s2" and page.steps[1].verified_by == "model"
    # A line of words needs two independent confirmations.
    assert agreed.asked[:2] == ["read_page", "judge_find"]
    assert sorted(agreed.asked[2:]) == ["judge_confirm", "judge_confirm", "reread"]


# ---------------------------------------------------------------- speed: overlapping calls


class Gate(Script):
    """A Script whose `wait_for` call blocks until `until` has been asked (or times out):
    passing proves the two calls ran at the same time."""

    def __init__(self, wait_for, until, **answers):
        super().__init__(**answers)
        self.wait_for, self.until, self.overlapped = wait_for, until, False
        self.seen = threading.Event()

    def ask(self, prompt, **kw):
        if kw["purpose"] == self.until:
            self.seen.set()
        if kw["purpose"] == self.wait_for:
            self.overlapped = self.seen.wait(5)
        return super().ask(prompt, **kw)


WORDS = json.dumps({"problem": None, "steps": [
    {"latex": "\\text{Mass is } 3\\,\\mathrm{kg}", "bbox": [0.1, 0.1, 0.5, 0.1]},
    {"latex": "\\text{so it is } 30\\,\\mathrm{g}", "bbox": [0.1, 0.25, 0.5, 0.1]},
]})
UPHELD = json.dumps({"complete": True, "wrong": True, "recovered": False, "side_remark": False, "why": "x"})


def test_judge_claim_is_reread_and_hinted_while_it_is_confirmed():
    same = json.dumps({"lines": [st["latex"] for st in json.loads(WORDS)["steps"]]})
    spark = Gate("judge_confirm", "reread", read_page=WORDS, reread=same, hint="What unit did you start in?",
                 judge_find=json.dumps({"line": 2, "kind": "unit conversion"}), judge_confirm=UPHELD)
    _, finding = check_image(spark, jpeg())
    assert spark.overlapped  # the re-read started before the confirmation answered
    assert finding is not None and finding.hint == "What unit did you start in?"
    assert "hint" in spark.asked and spark.asked.count("judge_find") == 1  # hint drafts race; the first safe one wins


def test_sympy_claim_hint_is_written_while_its_lines_are_reread():
    spark = Gate("reread", "hint", read_page=PAGE, reread=lines("2x + 5 = 11", "2x = 16"), hint="What about the 5?")
    _, finding = check_image(spark, jpeg())
    assert spark.overlapped and finding is not None and finding.hint == "What about the 5?"


def test_refused_claim_drops_its_early_reread_and_hint():
    spark = Script(read_page=WORDS, judge_find=json.dumps({"line": 2, "kind": "unit conversion"}),
                   judge_confirm=json.dumps({"wrong": False}), reread=lines("x"), hint="h")
    page, finding = check_image(spark, jpeg())
    assert finding is None and all(s.status != "error" for s in page.steps)


# ---------------------------------------------------------------- live: what is already known


def test_reading_is_shown_before_it_is_checked():
    shown = []
    spark = Script(read_page=PAGE, reread=lines("2x + 5 = 11", "2x = 16"), hint="h")

    def on_read(page):
        shown.append(([s.status for s in page.steps], list(spark.asked)))

    check_image(spark, jpeg(), on_read=on_read)
    assert shown == [(["unchecked", "unchecked"], ["read_page"])]


def test_open_mistake_read_the_same_again_stands_without_new_checks():
    first = Script(read_page=PAGE, reread=lines("2x + 5 = 11", "2x = 16"), hint="What about the 5?")
    page, finding = check_image(first, jpeg(), mask_for=lambda img, box: None)
    assert finding is not None
    again = json.dumps({"problem": "Solve 2x + 5 = 11", "steps": [
        {"latex": "2x+5=11", "bbox": [0.12, 0.12, 0.5, 0.1]},  # the page swayed a little
        {"latex": "2x = 16", "bbox": [0.12, 0.27, 0.4, 0.1]},
        {"latex": "x = 8", "bbox": [0.12, 0.4, 0.3, 0.1]},
    ]})
    spark = Script(read_page=again)
    timings: dict = {}
    page2, finding2 = check_image(spark, jpeg(), mask_for=lambda img, box: None, prior=Prior(page, finding),
                                  timings=timings)
    assert spark.asked == ["read_page"]  # no judge, no re-read, no hint
    assert finding2.step_id == "s2" and finding2.hint == "What about the 5?" and timings["reused"]
    assert page2.steps[1].status == "error" and page2.steps[1].bbox[0] == 0.12  # boxes from this frame
    assert page2.steps[2].status == "ok"  # x = 8 follows from 2x = 16 (SymPy)


def test_open_mistake_rewritten_is_checked_afresh():
    page = Page(problem=None, steps=[Step(id="s1", latex="2x + 5 = 11", bbox=(0.1, 0.1, 0.5, 0.1), status="ok",
                                          verified_by="sympy"),
                                     Step(id="s2", latex="2x = 16", bbox=(0.1, 0.25, 0.4, 0.1), status="error",
                                          verified_by="sympy")])
    fixed = json.dumps({"problem": None, "steps": [{"latex": "2x + 5 = 11", "bbox": [0.1, 0.1, 0.5, 0.1]},
                                                   {"latex": "2x = 6", "bbox": [0.1, 0.25, 0.4, 0.1]}]})
    spark = Script(read_page=fixed)
    page2, finding2 = check_image(spark, jpeg(), prior=Prior(page, Finding(step_id="s2", kind="k", hint="h")))
    assert finding2 is None and [s.status for s in page2.steps] == ["ok", "ok"]


def test_steps_the_judge_passed_before_are_not_judged_again():
    three = {"problem": None, "steps": [
        {"latex": "\\text{Mass is } 3\\,\\mathrm{kg}", "bbox": [0.1, 0.1, 0.5, 0.1]},
        {"latex": "\\text{in grams } 3000\\,\\mathrm{g}", "bbox": [0.1, 0.25, 0.5, 0.1]},
        {"latex": "\\text{half is } 1500\\,\\mathrm{g}", "bbox": [0.1, 0.4, 0.5, 0.1]},
    ]}
    before = Page.model_validate({"problem": None, "steps": [
        {**st, "id": f"s{i + 1}", "bbox": st["bbox"], "status": "ok", "verified_by": "model"}
        for i, st in enumerate(three["steps"][:2])]})
    spark = Script(read_page=json.dumps(three), judge_find=json.dumps({"line": None}))
    check_image(spark, jpeg(), prior=Prior(before, None))
    checked = [ln for ln in spark.prompts["judge_find"].splitlines() if ln.startswith("Line") and "[CHECK]" in ln]
    # Step 1 was judged fine and reads the same: not asked again. Step 2 was the last one
    # read (it may have been unfinished) and step 3 is new: both are judged.
    assert [ln.split(" [CHECK]")[0] for ln in checked] == ["Line 2", "Line 3"]


def test_abandoned_attempt_is_left_out_of_a_live_reading():
    restart = json.dumps({"problem": "Solve x^2 - 5x + 6 = 0", "steps": [
        {"latex": "x^2 - 5x + 6 = 0", "bbox": [0.1, 0.1, 0.5, 0.05]},
        {"latex": "x = \\frac{5 \\pm 1}{4}", "bbox": [0.1, 0.2, 0.5, 0.05]},
        {"latex": "x = 1.5", "bbox": [0.1, 0.3, 0.5, 0.05]},
        {"latex": "x^2 - 5x + 6 = 0", "bbox": [0.1, 0.4, 0.5, 0.05]},  # starts again: factorising
        {"latex": "(x-2)(x-3) = 0", "bbox": [0.1, 0.5, 0.5, 0.05]},
    ]})
    spark = Script(read_page=restart, judge_find=json.dumps({"line": None}))
    timings: dict = {}
    page, finding = check_image(spark, jpeg(), prior=Prior(), timings=timings)
    assert [s.latex for s in page.steps] == ["x^2 - 5x + 6 = 0", "(x-2)(x-3) = 0"]
    assert finding is None and timings["left_out"] == 3
    # The eval reads every page as it is: nothing left out.
    page, _ = check_image(Script(read_page=restart, judge_find=json.dumps({"line": None}),
                                 reread=lines("x"), reread_b=lines("y")), jpeg(), hint=False)
    assert len(page.steps) == 5
