"""A4: the guardrail rejects hints that carry the answer; the writer never
returns one. The live done criterion (zero leaks across eval errors) runs with a key."""

import time

import pytest

from lens.checker import check_page
from lens.contract import Page, Step
from lens.hints import SAFE_HINT, expected_for, leak_reason, where_it_fails, write_hint
from lens.spark import CallLog, Reply


def checked(*lines, problem=None):
    page = Page(problem=problem, steps=[Step(id=f"s{i + 1}", latex=l) for i, l in enumerate(lines)])
    page, finding = check_page(page)
    assert finding is not None
    return page, finding


SIGN = checked("2x + 5 = 11", "2x = 11 + 5", problem="Solve 2x + 5 = 11")
EXPAND = checked("(x+1)^2", "= x^2 + 1")
DERIV = checked("\\frac{d}{dx}(3x^2 + 2x) = 6x")

LEAKS = [
    (SIGN, "Is x = 3 what you get?"),                      # the result
    (SIGN, "What if 2x = 6 instead?"),                     # corrected step
    (SIGN, "Could it be that $2x = 11 - 5$?"),             # corrected step, LaTeX
    (SIGN, "Did you mean 3?"),                             # bare result number
    (SIGN, "The answer is x equals three, do you see why?"),  # reveal phrase
    (EXPAND, "Shouldn't the middle term be there, like x^2 + 2x + 1?"),
    (DERIV, "Did you forget the 2 from 2x, so 6x + 2?"),
    (SIGN, "Subtract 5 from both sides."),                 # not a question, tells the operation
]

SAFE = [
    (SIGN, "When you moved the 5 to the other side, what happened to its sign?"),
    (SIGN, "Look at 2x + 5 = 11. What did you do with the 5?"),
    (EXPAND, "What do you get if you write (x+1)^2 as (x+1)(x+1) and multiply it out?"),
    (DERIV, "What is the derivative of each term on its own?"),
    (DERIV, "Which term in 3x^2 + 2x did you differentiate?"),
]


@pytest.mark.parametrize("case,hint", LEAKS)
def test_leaky_hints_are_rejected(case, hint):
    page, finding = case
    assert leak_reason(hint, expected_for(page, finding)) is not None, hint


@pytest.mark.parametrize("case,hint", SAFE)
def test_socratic_hints_pass(case, hint):
    page, finding = case
    assert leak_reason(hint, expected_for(page, finding)) is None, hint


class ScriptedSpark:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def ask(self, prompt, **kw):
        self.prompts.append(prompt)
        return Reply(self.replies.pop(0), CallLog("hint", kw.get("effort")))


def test_writer_regenerates_after_a_leak_and_tells_the_model_why():
    page, finding = SIGN
    spark = ScriptedSpark(["So x = 3?", "What happened to the sign of the 5 when it crossed the equals sign?"])
    hint, rejected = write_hint(spark, page, finding, drafts=1)
    assert hint.startswith("What happened")
    assert len(rejected) == 1
    assert "rejected" in spark.prompts[1]


class Racing(ScriptedSpark):
    """Drafts starting with "Slow" take 0.3 s; the rest come back at once."""

    def ask(self, prompt, **kw):
        self.prompts.append(prompt)
        text = self.replies.pop(0)
        time.sleep(0.3 if text.startswith("Slow") else 0.0)
        return Reply(text, CallLog("hint", kw.get("effort")))


def test_two_drafts_at_once_the_first_safe_one_wins():
    page, finding = SIGN
    spark = Racing(["Slow: what did you do with the 5?", "Where did the 5 go when it crossed the equals sign?"])
    t = time.monotonic()
    hint, rejected = write_hint(spark, page, finding)
    assert hint.startswith("Where did the 5 go") and rejected == []
    assert time.monotonic() - t < 0.25  # did not wait for the slow draft
    assert len(spark.prompts) == 2 and spark.prompts[0] == spark.prompts[1]


def test_a_leaking_draft_loses_to_the_safe_one():
    page, finding = SIGN
    hint, rejected = write_hint(Racing(["So x = 3?", "Slow: what happened to the sign of the 5?"]), page, finding)
    assert hint == "Slow: what happened to the sign of the 5?" and len(rejected) == 1


def test_writer_falls_back_to_safe_hint_after_repeated_leaks():
    page, finding = SIGN
    hint, rejected = write_hint(ScriptedSpark(["x = 3?"] * 3), page, finding)
    assert hint == SAFE_HINT
    assert len(rejected) == 3
    assert leak_reason(SAFE_HINT, expected_for(page, finding)) is None


# The hint names the actual slip, not an earlier correct move (live: "3x = 11 + 4" ->
# "3x = 14" got a hint about moving the four across).
ADDITION = checked("3x - 4 = 11", "3x = 11 + 4", "3x = 14", problem="Solve 3x - 4 = 11")


def test_where_names_the_side_that_was_worked_out_wrongly():
    where, forbidden = where_it_fails(*ADDITION)
    assert "11 + 4 to 14" in where and "right side" in where
    assert "15" not in where  # the true value goes to the guardrail, never into the prompt
    assert any(str(v) == "15" for v in forbidden)


def test_true_value_of_the_slip_is_a_leak():
    assert leak_reason("Is 11 + 4 really 15, then?", expected_for(*ADDITION)) is not None
    assert leak_reason("Where you wrote 14 on the right, how did you get it from 11 + 4?",
                       expected_for(*ADDITION)) is None


def test_where_reaches_the_hint_prompt():
    seen = []

    class Spark:
        def ask(self, prompt, **kw):
            seen.append(prompt)
            return Reply("Where you wrote 14, can you check how you got it?", CallLog("hint", kw.get("effort")))

    write_hint(Spark(), *ADDITION)
    assert "Exactly where" in seen[0] and "from 11 + 4 to 14" in seen[0]


def test_where_for_other_slips():
    assert "not multiplied or divided by the same number" in where_it_fails(
        *checked("\\frac{x}{4} = 3", "x = \\frac{3}{4}"))[0]
    assert "Rewriting (x+1)^2 as x^2 + 1" in where_it_fails(*EXPAND)[0]
    assert "is not true" in where_it_fails(*DERIV)[0]
    assert "without changing its sign" in where_it_fails(*SIGN)[0]
