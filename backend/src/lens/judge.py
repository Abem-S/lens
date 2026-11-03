"""The model judge for lines SymPy cannot parse (words, units, chemistry,
inequalities). Its verdicts are marked verified_by="model".

One call sees the whole page and names the first wrong candidate line, if any.
A second, stricter call must confirm that exact line before it counts: a false
alarm costs more than a missed mistake.

The confirmation is shown what comes right after the claimed line, and answers
four questions, all of which must go the claim's way:
- complete: a handwritten line often stops mid-sentence or mid-expression and goes
  on in the next line ("... hence R is" / "symmetric."). It is judged with its
  continuation; with none written yet, it is not judged at all.
- wrong: certainly a mathematical or scientific error (not a skipped reason).
- recovered: the lines after it carry on from the correct value, as if the line
  were right: the line was misread (a term lost in reading) or fixed at once.
- side remark: an aside the solution does not use is not the student's mistake.
A line of words (SymPy cannot parse it) must clear a higher bar: two independent
confirmations, both upholding the claim. Measured: every remaining false alarm on
held-out FERMAT pages was the judge on such a line.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import re
from collections.abc import Callable

from lens.checker import ModelJudge
from lens.contract import Page, Step
from lens.mathparse import analyse
from lens.reader import Asks
from lens.spark import PAGE_MAX_TOKENS

FIND_PROMPT = """You check a student's handwritten STEM working, transcribed line by line.

Problem: {problem}

{lines}

Only the lines marked [CHECK] need your judgement; the others were already checked
with a computer algebra system and are correct as far as it can tell.

Is any [CHECK] line a real mistake: a wrong calculation, a wrong fact, a wrong rule,
or a conclusion that does not follow? Handwriting wraps: a line may stop mid-sentence
or mid-expression and go on in the next line, so read each line together with the
lines after it. These are NOT mistakes: wording, notation, renamed variables, skipped
steps or reasons that are still correct, answer options being listed, restating the
question, a side remark the solution does not use, a line the following lines carry
on from as if it were right (it was misread), or a transcription that is cut off or
garbled.

Reply with JSON only:
{{"line": <number of the FIRST wrong [CHECK] line, or null>, "kind": "short name of the mistake"}}"""

CONFIRM_PROMPT = """A student's handwritten working, transcribed line by line:

Problem: {problem}

{lines}

Someone claims line {n} is a mistake ("{kind}").
Line {n}: {line}
Right after it: {after}

Check it yourself, carefully, and answer each question:
- "complete": is line {n} a finished statement, on its own or together with the lines
  right after it that continue it? false if it stops mid-sentence or mid-expression
  and nothing written after it finishes it.
- "wrong": is that statement (with its continuation) certainly wrong as written: a
  mathematical or scientific error? Wording, notation, a skipped but true reason, or a
  transcription glitch is NOT wrong.
- "recovered": do the lines after it carry on from the correct value, as if line {n}
  were right (they do not follow from its error)? Then it was most likely misread.
- "side_remark": is line {n} an aside that the rest of the solution does not use?

Be strict: a false alarm is worse than a missed mistake.
Reply with JSON only:
{{"complete": true | false, "wrong": true | false, "recovered": true | false, "side_remark": true | false,
  "why": "one short sentence"}}"""

PROSE_CONFIRMATIONS = 2  # independent confirmations a claim on a line of words needs


def _json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def _lines(page: Page, candidates: list[str]) -> str:
    return "\n".join(
        f"Line {i + 1}{' [CHECK]' if s.id in candidates else ''}: {s.latex}" for i, s in enumerate(page.steps)
    )


def is_prose(latex: str) -> bool:
    """A line SymPy cannot parse: words, a proof sentence, units, a list."""
    return analyse(latex).kind == "unparseable"


def _after(page: Page, n: int, k: int = 8) -> str:
    # Up to the end of the working, as a rule: a final answer that comes out right after
    # a "wrong" line is the plainest sign that the line was misread.
    nxt = page.steps[n:n + k]
    if not nxt:
        return "nothing yet: it is the last line written so far."
    return "\n".join(f"Line {n + 1 + i}: {s.latex}" for i, s in enumerate(nxt))


def upheld(verdict: dict) -> bool:
    """Every answer must go the claim's way; a missing answer does not count."""
    return (verdict.get("wrong") is True and verdict.get("complete") is True
            and verdict.get("recovered") is False and verdict.get("side_remark") is False)


def confirm_claim(spark: Asks, page: Page, candidates: list[str], n: int, kind: str,
                  effort: str = "medium") -> bool:
    """The strict second look at line `n` (1-based). A line of words needs every one of
    PROSE_CONFIRMATIONS independent confirmations to uphold the claim."""
    step = page.steps[n - 1]
    prompt = CONFIRM_PROMPT.format(problem=page.problem or "not stated", lines=_lines(page, candidates), n=n,
                                   kind=kind, line=step.latex, after=_after(page, n))
    times = PROSE_CONFIRMATIONS if is_prose(step.latex) else 1

    def one() -> bool:
        return upheld(_json(spark.ask(prompt, effort=effort, max_tokens=PAGE_MAX_TOKENS, purpose="judge_confirm",
                                      json_mode=True).text))

    if times == 1:
        return one()
    with cf.ThreadPoolExecutor(max_workers=times) as pool:
        return all(f.result() for f in [pool.submit(one) for _ in range(times)])


def judge_claim(spark: Asks, page: Page, candidates: list[str], *, effort: str = "low",
                confirm_effort: str = "medium",
                on_draft: Callable[[Step, str], None] | None = None) -> tuple[str | None, str]:
    """Find the first wrong candidate line and confirm it. `on_draft` is called with a
    proposed claim before the confirmation runs (to start writing its hint early)."""
    found = _json(spark.ask(FIND_PROMPT.format(problem=page.problem or "not stated",
                                               lines=_lines(page, candidates)),
                            effort=effort, max_tokens=PAGE_MAX_TOKENS, purpose="judge_find", json_mode=True).text)
    n = found.get("line")
    if not isinstance(n, int) or not 1 <= n <= len(page.steps):
        return None, ""
    step = page.steps[n - 1]
    if step.id not in candidates:
        return None, ""
    kind = str(found.get("kind") or "this step does not follow")
    if on_draft is not None:
        on_draft(step, kind)
    return (step.id, kind) if confirm_claim(spark, page, candidates, n, kind, confirm_effort) else (None, "")


def spark_judge(spark: Asks, effort: str = "low", confirm_effort: str = "medium") -> ModelJudge:
    def judge(page: Page, candidates: list[str]) -> tuple[str | None, str]:
        return judge_claim(spark, page, candidates, effort=effort, confirm_effort=confirm_effort)

    return judge
