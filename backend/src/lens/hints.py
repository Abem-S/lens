"""A4 Hint writer + guardrail.

A hint asks a question or points at a place; it never states the corrected step
or the result. Every model hint passes `leak_reason` before it is spoken; a
rejected hint is regenerated, and after the last attempt a fixed safe hint is
used instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp

from lens.checker import _compare_equations, equivalent
from lens.contract import Finding, Page
from lens.mathparse import StepMath, analyse
from lens.reader import Asks

SAFE_HINT = "Look at this line and the one above it. What did you do to get from one to the other?"
MAX_ATTEMPTS = 3
MAX_WORDS = 40

HINT_PROMPT = """You are a patient tutor. A student is solving a problem on paper and one
line has a mistake. Ask ONE short spoken question that helps them find it themselves.

Problem: {problem}
Line above: {prev}
Line with the mistake: {cur}
What went wrong (for you only, do not say it): {kind}

Rules:
- One question, under 30 words, plain words a student would say aloud.
- Do NOT write the corrected line, the final answer, or any number or expression
  the student has not already written.
- Do NOT tell them which operation to do. Ask about what they did.
- No LaTeX, no symbols the student would have to read.
Reply with the question only."""

_REVEAL_PHRASES = re.compile(
    r"\b(the (correct )?answer is|should (be|get|equal)|you should get|it('s| is) actually|the right (answer|value)"
    r"|correct (line|step|value) is|instead of .* write)\b",
    re.IGNORECASE,
)
_DELIMITED = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)
# Plain-text math: numbers, single-letter variables, operators and brackets.
_PLAIN_MATH = re.compile(r"(?:(?<![A-Za-z])[A-Za-z](?![A-Za-z])|\d+(?:\.\d+)?|[+\-*/^=()]|\s)+")
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


@dataclass
class Expected:
    """What the student must not be told."""
    results: list[sp.Expr]      # values / expressions that are the answer
    equations: list[StepMath]   # equations equivalent to a correct next line
    on_page: set[str]           # normalised text already written by the student
    page_numbers: set[str]


def _norm(s: str) -> str:
    return re.sub(r"\s+|\\[,;! ]|[{}]", "", s).replace("\\cdot", "*").replace("\\times", "*")


def expected_for(page: Page, finding: Finding) -> Expected:
    idx = next(i for i, s in enumerate(page.steps) if s.id == finding.step_id)
    cur = analyse(page.steps[idx].latex)
    prev = analyse(page.steps[idx - 1].latex) if idx > 0 else None
    results: list[sp.Expr] = []
    equations: list[StepMath] = []
    if cur.kind == "identity" and cur.sides:
        results.append(cur.sides[0].doit())  # the true value of what they computed
    if prev is not None and prev.kind == "equation":
        equations.append(prev)
        syms = set().union(*(sp.expand(l - r).free_symbols for l, r in prev.variants))
        if len(syms) == 1:
            var = next(iter(syms))
            for l, r in prev.variants:
                try:
                    sol = sp.solveset(l - r, var, domain=sp.S.Reals)
                except Exception:
                    continue
                if isinstance(sol, sp.FiniteSet):
                    results.extend(sol)
    elif prev is not None and prev.kind in ("expression", "identity") and prev.sides:
        results.append(prev.last)
    on_page, numbers = set(), set()
    for s in page.steps[: idx + 1]:
        line = s.latex.lstrip("=")
        on_page.add(_norm(line))
        on_page.update(_norm(p) for p in line.split("="))
        numbers.update(_NUMBER.findall(s.latex))
    if page.problem:
        numbers.update(_NUMBER.findall(page.problem))
    return Expected(results, equations, on_page, numbers)


def _candidates(hint: str) -> list[str]:
    found = [next(g for g in m.groups() if g) for m in _DELIMITED.finditer(hint)]
    plain = _DELIMITED.sub(" ", hint)
    for m in _PLAIN_MATH.finditer(plain):
        chunk = m.group(0).strip(" ()")
        if re.search(r"\d", chunk) or re.search(r"[A-Za-z]\s*[+\-*/^=]|[+\-*/^=]\s*[A-Za-z]", chunk):
            # Drop English one-letter words that sneak in ("a", "I").
            chunk = re.sub(r"^(?:[aAI]\s+)+|(?:\s+[aAI])+$", "", chunk).strip()
            if chunk:
                found.append(chunk)
    return found


def leak_reason(hint: str, expected: Expected) -> str | None:
    """Why this hint must not be spoken, or None if it is safe."""
    if not hint or not hint.strip():
        return "empty hint"
    if "?" not in hint:
        return "a hint must be a question"
    if len(hint.split()) > MAX_WORDS:
        return "too long to speak"
    return answer_leak(hint, expected)


def answer_leak(text: str, expected: Expected) -> str | None:
    """Content check shared by hints and spoken conversation replies: does this
    text give away the result, a correct next line, or something equal to them?"""
    hint = text
    if _REVEAL_PHRASES.search(hint):
        return "states what the answer should be"

    numeric_results = [r for r in expected.results if r.is_number]
    for num in _NUMBER.findall(hint):
        if num in expected.page_numbers:
            continue
        val = sp.nsimplify(num, rational=True)
        if any(equivalent(val, r) is True for r in numeric_results):
            return f"says the result {num}"

    for cand in _candidates(hint):
        if _norm(cand) in expected.on_page:
            continue  # quoting the student's own writing is pointing, not telling
        sm = analyse(cand)
        if sm.kind == "unparseable":
            continue
        if sm.kind == "equation":
            for eq in expected.equations:
                if _compare_equations(eq, sm).verdict == "ok":
                    return f"contains a correct next line ({cand})"
            for r in expected.results:
                if any(equivalent(side, r) is True for side in sm.sides[1:]) and len(sm.sides[0].free_symbols) == 1 \
                        and sm.sides[0].is_Symbol:
                    return f"states the result ({cand})"
        for side in sm.sides:
            if side.is_number and str(side) in expected.page_numbers:
                continue
            if any(equivalent(side, r) is True for r in expected.results):
                return f"contains an expression equal to the answer ({cand})"
    return None


def write_hint(spark: Asks, page: Page, finding: Finding, effort: str = "low") -> tuple[str, list[str]]:
    """Return (hint, rejected reasons). Never returns a leaking hint."""
    idx = next(i for i, s in enumerate(page.steps) if s.id == finding.step_id)
    prompt = HINT_PROMPT.format(
        problem=page.problem or "not stated",
        prev=page.steps[idx - 1].latex if idx > 0 else "(none, this is the first line)",
        cur=page.steps[idx].latex,
        kind=finding.kind,
    )
    expected = expected_for(page, finding)
    rejected: list[str] = []
    for _ in range(MAX_ATTEMPTS):
        extra = ""
        if rejected:
            extra = f"\n\nYour last question was rejected: {rejected[-1]}. Ask a different question that reveals less."
        hint = spark.ask(prompt + extra, effort=effort, purpose="hint").text.strip().strip('"')
        reason = leak_reason(hint, expected)
        if reason is None:
            return hint, rejected
        rejected.append(reason)
    return SAFE_HINT, rejected
