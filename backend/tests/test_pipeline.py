"""The shared page pipeline: a SymPy error caused by a misreading is withdrawn
after the re-read; a real one survives and gets a hint and an outline."""

import io
import json

from PIL import Image

from lens.pipeline import check_image
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
        self.answers, self.asked = answers, []

    def ask(self, prompt, **kw):
        self.asked.append(kw["purpose"])
        return Reply(self.answers[kw["purpose"]], CallLog(kw["purpose"], kw.get("effort")))


def test_misread_line_is_withdrawn_after_reread():
    spark = Script(read_page=PAGE, reread=json.dumps({"lines": ["2x + 5 = 11", "2x = 6"]}))
    page, finding = check_image(spark, jpeg())
    assert finding is None
    assert page.steps[1].latex == "2x = 6" and page.steps[1].status == "ok"
    assert spark.asked == ["read_page", "reread"]


def test_real_error_survives_reread_and_gets_hint_and_outline():
    spark = Script(read_page=PAGE, reread=json.dumps({"lines": ["2x + 5 = 11", "2x = 16"]}),
                   hint="When the 5 crossed the equals sign, what did you do with it?")
    _page, finding = check_image(spark, jpeg(), mask_for=lambda img, box: None)
    assert finding is not None and finding.step_id == "s2"
    assert finding.hint.startswith("When the 5")
    assert finding.mask and len(finding.mask) >= 8  # box outline fallback
    assert spark.asked == ["read_page", "reread", "hint"]


def test_unreadable_reread_keeps_the_sympy_verdict():
    spark = Script(read_page=PAGE, reread="no idea")
    _, finding = check_image(spark, jpeg(), hint=False)
    assert finding is not None and finding.step_id == "s2"


def test_model_judge_needs_confirmation():
    words = json.dumps({"problem": None, "steps": [
        {"latex": "\\\\text{Mass is } 3\\\\,\\\\mathrm{kg}", "bbox": [0.1, 0.1, 0.5, 0.1]},
        {"latex": "\\\\text{so it is } 30\\\\,\\\\mathrm{g}", "bbox": [0.1, 0.25, 0.5, 0.1]},
    ]})
    refused = Script(read_page=words, judge_find=json.dumps({"line": 2, "kind": "unit conversion"}),
                     judge_confirm=json.dumps({"wrong": False, "why": "unsure"}))
    assert check_image(refused, jpeg(), hint=False)[1] is None
    agreed = Script(read_page=words, judge_find=json.dumps({"line": 2, "kind": "unit conversion"}),
                    judge_confirm=json.dumps({"wrong": True, "why": "3 kg is 3000 g"}))
    page, finding = check_image(agreed, jpeg(), hint=False)
    assert finding.step_id == "s2" and page.steps[1].verified_by == "model"
    assert agreed.asked == ["read_page", "judge_find", "judge_confirm"]
