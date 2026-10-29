"""The model judge for lines SymPy cannot parse (words, units, chemistry,
inequalities). Its verdicts are marked verified_by="model".

One call sees the whole page and names the first wrong candidate line, if any.
A second, stricter call must confirm that exact line before it counts: a false
alarm costs more than a missed mistake.
"""

from __future__ import annotations

import json
import re

from lens.checker import ModelJudge
from lens.contract import Page
from lens.reader import Asks
from lens.spark import PAGE_MAX_TOKENS

FIND_PROMPT = """You check a student's handwritten STEM working, transcribed line by line.

Problem: {problem}

{lines}

Only the lines marked [CHECK] need your judgement; the others were already checked
with a computer algebra system and are correct as far as it can tell.

Is any [CHECK] line a real mistake: a wrong calculation, a wrong fact, a wrong rule,
or a conclusion that does not follow? These are NOT mistakes: wording, notation,
renamed variables, skipped steps that are still correct, answer options being listed,
restating the question, or a transcription that is cut off or garbled.

Reply with JSON only:
{{"line": <number of the FIRST wrong [CHECK] line, or null>, "kind": "short name of the mistake"}}"""

CONFIRM_PROMPT = """A student's handwritten working, transcribed line by line:

Problem: {problem}

{lines}

Someone claims line {n} is a mistake ("{kind}"). Check it yourself, carefully.
Be strict: say yes ONLY if line {n} is certainly wrong as written, and it is a
mathematical or scientific error, not wording, notation or a transcription glitch.

Reply with JSON only: {{"wrong": true | false, "why": "one short sentence"}}"""


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


def spark_judge(spark: Asks, effort: str = "low", confirm_effort: str = "medium") -> ModelJudge:
    def judge(page: Page, candidates: list[str]) -> tuple[str | None, str]:
        lines = _lines(page, candidates)
        problem = page.problem or "not stated"
        found = _json(spark.ask(FIND_PROMPT.format(problem=problem, lines=lines), effort=effort,
                                max_tokens=PAGE_MAX_TOKENS, purpose="judge_find", json_mode=True).text)
        n = found.get("line")
        if not isinstance(n, int) or not 1 <= n <= len(page.steps):
            return None, ""
        step = page.steps[n - 1]
        if step.id not in candidates:
            return None, ""
        kind = str(found.get("kind") or "this step does not follow")
        confirm = _json(spark.ask(CONFIRM_PROMPT.format(problem=problem, lines=lines, n=n, kind=kind),
                                  effort=confirm_effort, max_tokens=PAGE_MAX_TOKENS, purpose="judge_confirm",
                                  json_mode=True).text)
        return (step.id, kind) if confirm.get("wrong") is True else (None, "")

    return judge
