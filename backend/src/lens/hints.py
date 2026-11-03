"""A4 Hint writer + guardrail.

A hint asks a question or points at a place; it never states the corrected step
or the result. Every model hint passes `leak_reason` before it is spoken; a
rejected hint is regenerated, and after the last attempt a fixed safe hint is
used instead.

Speed: Spark reasons for several hundred tokens before a hint whatever the effort
setting (measured: ~850 tokens at "low" and at "minimal" alike), so a hint takes
2-5 s and varies a lot from call to call. Two drafts are asked for at once and the
first one that passes the guardrail is used: the wait is the faster of two calls.
"""

from __future__ import annotations

import concurrent.futures as cf
import re
from dataclasses import dataclass

import sympy as sp

from lens.checker import _compare_equations, _sign_error_kind, equivalent
from lens.contract import Finding, Page
from lens.mathparse import StepMath, _clean, _split_top_level, analyse
from lens.reader import Asks

SAFE_HINT = "Look at the step I marked and the one before it. What did you do to get from one to the other?"
MAX_ATTEMPTS = 3  # model calls in all, drafts included
HINT_EFFORT = "minimal"  # measured: as safe as "low" on the A4 audit, and a little faster
HINT_DRAFTS = 2  # asked for at once; the first that passes the guardrail wins
_LINE_WORD = re.compile(r"\b(this|that|the|your|each|every|a|next|previous|first|last)?\s*\blines?\b(\s+\d+)?", re.IGNORECASE)


def say_step(text: str) -> str:
    """Students write steps, not "lines": "line 2" -> "step 2", "that line" -> "that step"."""
    def swap(m: re.Match) -> str:
        word = "steps" if m.group(0).lower().rstrip("0123456789 ").endswith("lines") else "step"
        lead, num = m.group(1), m.group(2) or ""
        out = f"{lead} {word}{num}" if lead else f"{word}{num}"
        out = out if not m.group(0)[:1].isupper() else out[:1].upper() + out[1:]
        return (" " if m.group(0).startswith(" ") else "") + out
    return _LINE_WORD.sub(swap, text)
MAX_WORDS = 40

HINT_PROMPT = """You are a patient tutor. A student is solving a problem on paper and one
step has a mistake. Ask ONE short spoken question that helps them find it themselves.

Problem: {problem}
Step before (correct): {prev}
Step with the mistake: {cur}
What went wrong (for you only, do not say it): {kind}
Exactly where (for you only, do not say it): {where}

Rules:
- Ask about that exact place in the step with the mistake. Do not ask about anything
  the step before did: it is correct.
- One question, under 30 words, plain words a student would say aloud.
- Do NOT write the corrected line, the final answer, or any number or expression
  the student has not already written.
- Do NOT tell them which operation to do. Ask about what they did.
- No LaTeX, no symbols the student would have to read.
- Point to their work by what they wrote, said in words ("where you moved the five
  across"), not by a number or the word "line". The step is outlined on their screen.
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
    results.extend(where_it_fails(page, finding)[1])
    on_page, numbers = set(), set()
    for s in page.steps[: idx + 1]:
        line = s.latex.lstrip("=")
        on_page.add(_norm(line))
        on_page.update(_norm(p) for p in line.split("="))
        numbers.update(_NUMBER.findall(s.latex))
    if page.problem:
        numbers.update(_NUMBER.findall(page.problem))
    return Expected(results, equations, on_page, numbers)


def _side_texts(latex: str, n: int) -> list[str] | None:
    """The written sides of a line, as the student wrote them ("11 + 4", not 15)."""
    s = _clean(latex).lstrip("=").strip()
    parts = _split_top_level(s)
    return parts if len(parts) == n else None


def _value(e: sp.Expr) -> sp.Expr | None:
    try:
        return sp.simplify(e.doit())
    except Exception:
        return None


def where_it_fails(page: Page, finding: Finding) -> tuple[str, list[sp.Expr]]:
    """Where exactly the flagged step goes wrong, for the hint writer only, and the true
    values of the part that went wrong (which a hint must never say).

    SymPy knows which relation fails: one side of an equation changed while the other
    stayed the same (11 + 4 became 14), both sides changed by different amounts or
    factors, a link inside a chain (a = b) is false, or an expression was rewritten into
    one of a different value. Saying so keeps the hint on the actual slip instead of an
    earlier move that was right."""
    ids = [s.id for s in page.steps]
    if finding.step_id not in ids:
        return "the marked step", []
    idx = ids.index(finding.step_id)
    cur_s = page.steps[idx]
    cur = analyse(cur_s.latex)
    prev = analyse(page.steps[idx - 1].latex) if idx > 0 else None
    mine = "The slip is in the marked step itself; the steps before it are right."
    if cur.kind == "unparseable":
        return mine, []

    # A chain or calculus claim inside the one step: name the first false link.
    if cur.kind in ("identity", "equation") and len(cur.sides) >= 2 and (
            cur.kind == "identity" or not any(x.free_symbols for x in cur.sides)):
        texts = _side_texts(cur_s.latex, len(cur.sides))
        for k, (a, b) in enumerate(zip(cur.sides, cur.sides[1:])):
            if equivalent(a, b) is False or (not a.free_symbols and not b.free_symbols
                                              and equivalent(a, b) is not True):
                what = f"{texts[k]} = {texts[k + 1]}" if texts else "one of its equals signs"
                calc = " (the calculus there was worked out wrongly)" if a.has(sp.Derivative, sp.Integral,
                                                                              sp.Limit) else ""
                v = _value(a)
                return (f"Inside the marked step, {what} is not true{calc}. {mine}",
                        [v] if v is not None else [])

    if prev is None or prev.kind == "unparseable":
        return mine, []

    # An expression rewritten (expanded, simplified) into one of another value.
    if cur.kind == "expression" or (cur.kind == "identity" and prev.kind == "expression"):
        before = _side_texts(page.steps[idx - 1].latex, len(prev.sides))
        now = _side_texts(cur_s.latex, len(cur.sides))
        if before and now:
            return (f"Rewriting {before[-1]} as {now[0]} changed its value: the slip is in that "
                    f"rewriting (expanding, simplifying or working out), not earlier. {mine}"), []
        return mine, []

    if not (cur.kind == "equation" and prev.kind == "equation" and len(cur.variants) == 1
            and len(prev.variants) == 1):
        return mine, []
    (l1, r1), (l2, r2) = prev.variants[0], cur.variants[0]
    pt = _side_texts(page.steps[idx - 1].latex, 2) or [str(l1), str(r1)]
    ct = _side_texts(cur_s.latex, 2) or [str(l2), str(r2)]
    for same, other, name, was, now, true in ((equivalent(l1, l2), equivalent(r1, r2), "right", pt[1], ct[1], r1),
                                              (equivalent(r1, r2), equivalent(l1, l2), "left", pt[0], ct[0], l1)):
        if same is True and other is False:
            keep = "left" if name == "right" else "right"
            v = _value(true)
            text = (f"The {keep} side did not change. Only the {name} side changed, from {was} to {now}, "
                    f"and those are not equal: the slip is in working out the {name} side (the arithmetic, "
                    f"expanding or simplifying there), not in anything moved across the equals sign before. "
                    f"{mine}")
            return text, [v] if v is not None else []
    if _sign_error_kind(prev, cur):
        return f"A term was moved across the equals sign without changing its sign. {mine}", []
    # Both sides divided (or multiplied) by different factors, or changed by different amounts.
    try:
        fl, fr = sp.simplify(l1 / l2), sp.simplify(r1 / r2)
        if fl.is_number and fl != 0 and fr.is_number and sp.simplify(fl - fr) != 0:
            return (f"The left side went from {pt[0]} to {ct[0]} and the right side from {pt[1]} to {ct[1]}: "
                    f"the two sides were not multiplied or divided by the same number. {mine}"), []
        dl, dr = sp.simplify(l1 - l2), sp.simplify(r1 - r2)
        if sp.simplify(dl - dr) != 0:
            return (f"The left side went from {pt[0]} to {ct[0]} and the right side from {pt[1]} to {ct[1]}: "
                    f"the two sides were not changed by the same amount. {mine}"), []
    except Exception:  # SymPy could not compare the sides: say only where, not how
        return mine, []
    return mine, []


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


def write_hint(spark: Asks, page: Page, finding: Finding, effort: str = HINT_EFFORT,
               drafts: int = HINT_DRAFTS) -> tuple[str, list[str]]:
    """Return (hint, rejected reasons). Never returns a leaking hint."""
    idx = next(i for i, s in enumerate(page.steps) if s.id == finding.step_id)
    prompt = HINT_PROMPT.format(
        problem=page.problem or "not stated",
        prev=page.steps[idx - 1].latex if idx > 0 else "(none, this is the first step)",
        cur=page.steps[idx].latex,
        kind=finding.kind,
        where=where_it_fails(page, finding)[0],
    )
    expected = expected_for(page, finding)
    rejected: list[str] = []

    def draft(extra: str = "") -> str:
        return spark.ask(prompt + extra, effort=effort, purpose="hint").text.strip().strip('"')

    first = max(1, min(drafts, MAX_ATTEMPTS))
    pool = cf.ThreadPoolExecutor(max_workers=first)
    try:
        for f in cf.as_completed([pool.submit(draft) for _ in range(first)]):
            try:
                hint = f.result()
            except Exception as exc:  # one failed draft; the other may still do
                rejected.append(f"draft failed: {type(exc).__name__}")
                continue
            reason = leak_reason(hint, expected)
            if reason is None:
                return say_step(hint), rejected
            rejected.append(reason)
    finally:
        pool.shutdown(wait=False)  # a slower draft still running is simply not used
    for _ in range(MAX_ATTEMPTS - first):
        last = rejected[-1] if rejected else "no usable question"
        hint = draft(f"\n\nYour last question was rejected: {last}. Ask a different question that reveals less.")
        reason = leak_reason(hint, expected)
        if reason is None:
            return say_step(hint), rejected
        rejected.append(reason)
    return SAFE_HINT, rejected
