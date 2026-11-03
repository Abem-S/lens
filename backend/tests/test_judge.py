"""The model judge's bar for a claim: the confirmation sees what follows the claimed
line; an unfinished line, a line the next lines recover from, or a side remark is not
claimed; a line of words needs two confirmations that both uphold the claim."""

import json
import threading

from lens.contract import Page, Step
from lens.judge import judge_claim, upheld
from lens.spark import CallLog, Reply

YES = {"complete": True, "wrong": True, "recovered": False, "side_remark": False, "why": "wrong"}


class Script:
    """Answers by purpose (a value, or a list used in turn); thread-safe."""

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


def page(*lines):
    return Page(problem="Show R is symmetric", steps=[Step(id=f"s{i + 1}", latex=t) for i, t in enumerate(lines)])


PROOF = page("\\text{Let } (a,b) \\in R", "(a,b) \\in R_1 \\text{ hence } R \\text{ is}", "\\text{symmetric.}")
FIND = json.dumps({"line": 2, "kind": "non sequitur"})


def test_all_four_answers_must_go_the_claims_way():
    assert upheld(YES)
    for key, bad in (("complete", False), ("wrong", False), ("recovered", True), ("side_remark", True)):
        assert not upheld({**YES, key: bad})
    assert not upheld({"wrong": True})  # missing answers do not count


def test_confirmation_sees_the_lines_after_the_claim():
    spark = Script(judge_find=FIND, judge_confirm=json.dumps(YES))
    judge_claim(spark, PROOF, ["s1", "s2", "s3"])
    prompt = spark.prompts["judge_confirm"][0]
    assert "Right after it: Line 3: \\text{symmetric.}" in prompt


def test_a_line_of_words_needs_two_confirmations():
    both = Script(judge_find=FIND, judge_confirm=json.dumps(YES))
    assert judge_claim(both, PROOF, ["s1", "s2", "s3"]) == ("s2", "non sequitur")
    assert both.asked.count("judge_confirm") == 2
    split = Script(judge_find=FIND, judge_confirm=[json.dumps(YES), json.dumps({**YES, "wrong": False})])
    assert judge_claim(split, PROOF, ["s1", "s2", "s3"]) == (None, "")


def test_an_unfinished_last_line_is_not_claimed():
    spark = Script(judge_find=FIND, judge_confirm=json.dumps({**YES, "complete": False}))
    unfinished = page("\\text{Let } (a,b) \\in R", "(a,b) \\in R_1 \\text{ hence } R \\text{ is}")
    assert judge_claim(spark, unfinished, ["s1", "s2"]) == (None, "")
    assert "nothing yet: it is the last line written so far" in spark.prompts["judge_confirm"][0]


def test_a_maths_line_needs_one_confirmation():
    spark = Script(judge_find=json.dumps({"line": 2, "kind": "wrong value"}), judge_confirm=json.dumps(YES))
    p = page("F = ma", "F = 2 \\cdot 3 = 5")
    assert judge_claim(spark, p, ["s2"]) == ("s2", "wrong value")
    assert spark.asked.count("judge_confirm") == 1


def test_recovered_line_is_not_claimed():
    spark = Script(judge_find=json.dumps({"line": 1, "kind": "dropped term"}),
                   judge_confirm=json.dumps({**YES, "recovered": True}))
    p = page("= \\frac{11}{4} \\log|x+1| + C", "= \\frac{11}{4} \\log\\left|\\frac{x+1}{x+3}\\right| + C")
    assert judge_claim(spark, p, ["s1"]) == (None, "")


def test_draft_callback_runs_before_the_confirmation():
    seen = []
    spark = Script(judge_find=FIND, judge_confirm=json.dumps({**YES, "wrong": False}))
    judge_claim(spark, PROOF, ["s1", "s2", "s3"], on_draft=lambda step, kind: seen.append((step.id, kind)))
    assert seen == [("s2", "non sequitur")]
