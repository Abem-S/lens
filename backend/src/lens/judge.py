"""The model judge for lines SymPy cannot parse (words, units, chemistry,
inequalities). Its verdicts are marked verified_by="model"."""

from __future__ import annotations

import json
import re

from lens.checker import ModelJudge, Verdict
from lens.reader import Asks

JUDGE_PROMPT = """You check one line of a student's handwritten STEM working.

Problem: {problem}
Previous line: {prev}
Line to check: {cur}

Does the line to check follow correctly from the previous line (or, if there is
none, is it a correct start)? Only say "error" if you are certain: a false alarm
is worse than a missed mistake. Notation or style choices are not errors.

Reply with JSON only: {{"verdict": "ok" | "error" | "unsure", "kind": "short name of the mistake, or empty"}}"""


def spark_judge(spark: Asks, effort: str = "low") -> ModelJudge:
    def judge(prev: str | None, cur: str, problem: str | None) -> tuple[Verdict, str]:
        reply = spark.ask(JUDGE_PROMPT.format(problem=problem or "not stated", prev=prev or "(none)", cur=cur),
                          effort=effort, purpose="judge", json_mode=True)
        m = re.search(r"\{.*\}", reply.text, re.DOTALL)
        try:
            data = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            data = {}
        verdict = {"ok": "ok", "error": "error"}.get(str(data.get("verdict", "")).lower(), "unknown")
        return verdict, str(data.get("kind") or "")

    return judge
