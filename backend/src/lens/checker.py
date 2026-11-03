"""A3 Step checker: does each line follow from the one before it?

Rules (AGENTS.md, hard rule 4): a claimed mistake is a verified mistake. A line
SymPy has proved correct can never be called wrong. Lines SymPy cannot settle
(words, units, chemistry, inequalities, values substituted from the question) go
to the model judge, if one is given, and are marked verified_by="model"; without
a judge they stay "unverifiable" and nothing is said.

Lines of only linking words ("i.e.", "So,") are skipped when finding the line a
step builds on; arrows and remarks around a line ("\\Rightarrow", "(subtracting
x)") are dropped before parsing.
"""

from __future__ import annotations

import cmath
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import sympy as sp

from lens.contract import Finding, Page, Step
from lens.mathparse import StepMath, analyse, decimal_places, is_connector, numeric_sides

Verdict = Literal["ok", "error", "unknown"]

# judge(page, candidate step ids) -> (id of the first wrong candidate | None, kind).
# The judge sees the whole page for context but may only flag a candidate: a line
# SymPy could not parse. It answers once per page.
ModelJudge = Callable[[Page, list[str]], tuple[str | None, str]]

_SAMPLES = 6
_INVERSE_TRIG = (sp.asin, sp.acos, sp.atan, sp.acot, sp.asec, sp.acsc)
_TOL = 1e-8


@dataclass
class StepResult:
    verdict: Verdict
    kind: str = ""
    by: Literal["sympy", "model"] | None = None


# ---------- equality of expressions ----------

def _numeric_points(symbols: list[sp.Symbol], rng: random.Random) -> list[dict]:
    # Positive values in a modest range dodge most domain problems (sqrt, log).
    return [{s: sp.Rational(rng.randint(11, 39), 10) for s in symbols} for _ in range(_SAMPLES)]


def _num(expr: sp.Expr, point: dict) -> complex | None:
    try:
        v = complex(expr.evalf(subs=point, n=30))
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if cmath.isnan(v) or cmath.isinf(v):
        return None
    return v


def equivalent(a: sp.Expr, b: sp.Expr, *, up_to_constant_in: sp.Symbol | None = None) -> bool | None:
    """True / False when sure, None when undecidable.

    `up_to_constant_in`: compare antiderivatives, i.e. a - b may be any constant
    with respect to that variable (students write "+ C").
    """
    try:
        a, b = a.doit(), b.doit()
        diff = a - b
        if up_to_constant_in is not None:
            diff = sp.diff(diff, up_to_constant_in)
        if diff.has(sp.Integral, sp.Limit, sp.Derivative):
            return None  # SymPy could not evaluate the calculus; do not guess
        if sp.simplify(diff) == 0:
            return True
    except Exception:
        return None
    symbols = sorted(diff.free_symbols, key=lambda s: s.name)
    rng = random.Random(1234)
    agree = disagree = 0
    for pt in _numeric_points(symbols, rng):
        v = _num(diff, pt)
        if v is None:
            continue
        scale = max(1.0, abs(_num(a, pt) or 1.0))
        if abs(v) <= _TOL * scale:
            agree += 1
        else:
            disagree += 1
    if disagree >= 2 and agree == 0:
        # sin^-1(sin 2t) = 2t holds only where the inverse is taken (|2t| <= pi/2); the
        # sample points lie outside it. With an inverse trig function, never claim.
        if diff.has(*_INVERSE_TRIG):
            return None
        return False
    if agree >= 3 and disagree == 0:
        return True
    return None


# ---------- equations ----------

def _zero_form(variant: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
    lhs, rhs = variant
    return sp.expand(lhs - rhs)


def _proportional(f: sp.Expr, g: sp.Expr) -> bool:
    if f == 0 or g == 0:
        return False
    try:
        ratio = sp.simplify(f / g)
    except Exception:
        return False
    if ratio.is_number:
        return bool(ratio != 0 and ratio.is_finite)
    # Dividing through by a symbol (e.g. a in "at = v - u") is a legitimate step.
    coeff, rest = ratio.as_coeff_Mul()
    return coeff != 0 and all(
        isinstance(f, sp.Symbol) or (isinstance(f, sp.Pow) and isinstance(f.base, sp.Symbol) and f.exp.is_number)
        for f in sp.Mul.make_args(rest)
    )


def _real_solutions(eq: StepMath, var: sp.Symbol) -> set | None:
    sols: set = set()
    for v in eq.variants:
        try:
            s = sp.solveset(_zero_form(v), var, domain=sp.S.Reals)
        except Exception:
            return None
        if not isinstance(s, sp.FiniteSet):
            return None
        sols |= {sp.nsimplify(x) for x in s}
    return sols


def _same_value(a, b) -> bool:
    return equivalent(a, b) is True


def _set_relation(prev: set, cur: set) -> Literal["equal", "superset", "disjoint", "partial"]:
    lost = [p for p in prev if not any(_same_value(p, c) for c in cur)]
    gained = [c for c in cur if not any(_same_value(c, p) for p in prev)]
    if not lost and not gained:
        return "equal"
    if not lost:
        return "superset"  # e.g. squaring both sides: extra roots, none lost
    if len(lost) == len(prev) and len(gained) == len(cur):
        return "disjoint"
    return "partial"


def _sign_error_kind(prev: StepMath, cur: StepMath) -> str | None:
    """Was a term moved across '=' without changing its sign?"""
    lhs, rhs = prev.variants[0]
    f, g = lhs - rhs, _zero_form(cur.variants[0])
    # Moving term t from the left without flipping it gives (lhs - rhs) - 2t; from the right, + 2t.
    candidates = [f - 2 * t for t in sp.Add.make_args(lhs)] + [f + 2 * t for t in sp.Add.make_args(rhs)]
    if any(_proportional(sp.expand(c), g) for c in candidates):
        return "sign error moving a term across ="
    return None


def _compare_equations(prev: StepMath, cur: StepMath) -> StepResult:
    if (len(prev.variants) == 1 and len(cur.variants) == 1
            and _proportional(_zero_form(prev.variants[0]), _zero_form(cur.variants[0]))):
        return StepResult("ok", by="sympy")
    prev_syms = set().union(*(_zero_form(v).free_symbols for v in prev.variants))
    cur_syms = set().union(*(_zero_form(v).free_symbols for v in cur.variants))
    shared = prev_syms & cur_syms
    if (not cur_syms and len(prev_syms) == 1 and len(cur.variants) == 1
            and _zero_form(cur.variants[0]).is_number and _zero_form(cur.variants[0]) != 0):
        # x = 5 + x: the variable cancelled into a contradiction, but the line above had a solution.
        ps = _real_solutions(prev, next(iter(prev_syms)))
        if ps:
            return StepResult("error", "the new line has no solution but the line above does", "sympy")
        return StepResult("unknown")
    if not shared or prev_syms != cur_syms:
        # Values were substituted (F = ma -> F = 2*3) or a new thread began.
        # SymPy cannot know the given data, so it stays quiet.
        return StepResult("unknown")
    if len(shared) == 1:
        var = next(iter(shared))
        ps, cs = _real_solutions(prev, var), _real_solutions(cur, var)
        if ps is None or cs is None or not ps:
            return StepResult("unknown")
        rel = _set_relation(ps, cs)
        if rel in ("equal", "superset"):
            return StepResult("ok", by="sympy")
        if rel == "disjoint" and cs:
            kind = _sign_error_kind(prev, cur) or "the new line has different solutions from the line above"
            return StepResult("error", kind, "sympy")
        if not cs:
            return StepResult("error", "the new line has no solution but the line above does", "sympy")
        return StepResult("unknown")  # lost some roots: could be a case split; stay quiet
    # Several variables (rearranging a formula): solve for each shared symbol.
    any_ok = False
    for var in sorted(shared, key=lambda s: s.name):
        try:
            p = sp.solve(_zero_form(prev.variants[0]), var)
            c = sp.solve(_zero_form(cur.variants[0]), var)
        except Exception:
            continue
        if not p or not c or len(prev.variants) > 1 or len(cur.variants) > 1:
            continue
        verdicts = [equivalent(pi, ci) for pi in p for ci in c]
        if any(v is True for v in verdicts):
            any_ok = True
            break
        if all(v is False for v in verdicts):
            # Two lines in several variables may be a system (A + B = 0, then 2A + B = 1),
            # not one equation rearranged. Claim an error only for a sign slip, or when
            # the new line solves for one variable (t = ...), i.e. a rearrangement.
            sign = _sign_error_kind(prev, cur)
            lhs, rhs = cur.variants[0]
            solved = isinstance(lhs, sp.Symbol) and lhs not in rhs.free_symbols
            if sign or solved:
                return StepResult("error", sign or "the rearranged formula is not equivalent to the line above",
                                  "sympy")
            return StepResult("unknown")
    return StepResult("ok", by="sympy") if any_ok else StepResult("unknown")


# ---------- one line on its own ----------

def _calculus_kind(expr: sp.Expr) -> str:
    if expr.has(sp.Derivative):
        return "derivative computed incorrectly"
    if expr.has(sp.Integral):
        return "integral computed incorrectly"
    if expr.has(sp.Limit):
        return "limit evaluated incorrectly"
    return "the two sides of this line are not equal"


def _indefinite_var(expr: sp.Expr) -> sp.Symbol | None:
    for integ in expr.atoms(sp.Integral):
        for lim in integ.limits:
            if len(lim) == 1:
                return lim[0]
    return None


def _is_name(expr: sp.Expr) -> bool:
    """A bare name: x, or letters run together like AB / QR (a length, a label).
    "QR = sqrt(34)" defines QR; it is not a claim that Q*R equals sqrt(34)."""
    return isinstance(expr, sp.Symbol) or (
        isinstance(expr, sp.Mul) and all(isinstance(f, sp.Symbol) for f in expr.args))


def _close_numbers(a: sp.Expr, b: sp.Expr, decimals: int) -> bool | None:
    """Two plain numbers, allowing the rounding a student writes: 5.83 for sqrt(34),
    154 for 49*pi. Exact rationals with no decimals written must match exactly."""
    try:
        va, vb = complex(a.evalf(30)), complex(b.evalf(30))
    except (TypeError, ValueError):
        return None
    if not all(cmath.isfinite(v) for v in (va, vb)):
        return None  # a division by zero is a misreading (or no claim at all), not a wrong value
    if abs(va.imag) > 1e-12 or abs(vb.imag) > 1e-12:
        return None
    va, vb = va.real, vb.real
    if abs(va - vb) <= 1e-9 * max(1.0, abs(va)):
        return True
    # Rounding is only possible where a plain number stands for a value: a written
    # decimal, or a whole number standing for an irrational (pi * 49 = 154).
    rounded = bool(a.is_rational) != bool(b.is_rational) or (decimals > 0 and (a.is_rational or b.is_rational))
    if not rounded:
        return False
    tol = max(10.0 ** -decimals if decimals > 0 else 0.5, 2e-3 * max(abs(va), abs(vb)))
    return abs(va - vb) <= tol


def _link(a: sp.Expr, b: sp.Expr, decimals: int) -> bool | None:
    """Does a = b hold? None when the link is not a claim SymPy can judge."""
    if _is_name(a) or _is_name(b):
        return None
    if bool(a.free_symbols) != bool(b.free_symbols):
        # Values substituted from the question (F = ma = 2*3) are data SymPy lacks:
        # only a constant expression (sin^2 x + cos^2 x = 1) can be confirmed.
        return True if equivalent(a, b) is True else None
    ia, ib = _indefinite_var(a), _indefinite_var(b)
    if ia is not None and ib is not None:
        return None  # integrating both sides (separating variables) states a step, not a result
    if not a.free_symbols and not b.free_symbols and not a.has(*_CALC) and not b.has(*_CALC):
        return _close_numbers(a, b, decimals)
    return equivalent(a, b, up_to_constant_in=ia or ib)


_CALC = (sp.Derivative, sp.Integral, sp.Limit)


def _check_within(cur: StepMath) -> StepResult:
    """A chain a = b = c, or a calculus claim, must hold on its own. Every link
    SymPy can judge is judged; one false link is an error."""
    verdicts = []
    for a, b in zip(cur.sides, cur.sides[1:]):
        eq = _link(a, b, cur.decimals)
        if eq is False:
            return StepResult("error", _calculus_kind(a) if a.has(*_CALC) else
                              "the two sides of this line are not equal", "sympy")
        verdicts.append(eq)
    if verdicts and all(v is True for v in verdicts):
        return StepResult("ok", by="sympy")
    return StepResult("unknown")


def _expr_kind(a: sp.Expr, b: sp.Expr) -> str:
    if equivalent(a, -b) is True:
        return "sign error"
    return "this expression is not equal to the one above"


# ---------- a pair of lines ----------

def _symbols(m: StepMath) -> set[sp.Symbol]:
    return set().union(*(x.free_symbols for x in m.sides)) if m.sides else set()


def check_pair(prev: StepMath | None, cur: StepMath) -> StepResult:
    if cur.kind == "unparseable":
        return StepResult("unknown", by=None)

    # Some "equations" are identities ((x+1)^2 = x^2+2x+1): treat as a claim.
    # Plain numbers on both sides (7 x 8 = 56) are a claim to check, not an equation to solve.
    if cur.kind == "equation" and len(cur.variants) == 1 and (
            not any(s.free_symbols for s in cur.sides) or equivalent(*cur.variants[0]) is True):
        cur = StepMath(cur.latex, "identity", sides=cur.sides, continuation=cur.continuation,
                       decimals=cur.decimals)

    if cur.kind == "identity":
        within = _check_within(cur)
        if within.verdict != "ok" or prev is None or prev.kind == "unparseable":
            return within
        # Also link to the previous line when this one continues its value: only
        # from a bare expression line, and never through a name or new numbers.
        if (prev.kind == "expression" and not cur.has_calculus and not _is_name(cur.sides[0])
                and cur.sides[0].free_symbols):
            linked = equivalent(prev.last, cur.sides[0])
            if linked is False:
                return StepResult("error", _expr_kind(prev.last, cur.sides[0]), "sympy")
        return within

    if prev is None:
        return StepResult("ok", by="sympy") if cur.kind in ("expression", "equation") else StepResult("unknown")
    if prev.kind == "unparseable":
        # The line it builds on is words SymPy cannot read (a setup from the question):
        # nothing to compare with, so it is not verified; the model judge sees it.
        return _check_within(cur) if len(cur.sides) > 1 and cur.kind == "expression" else StepResult("unknown")

    if cur.kind == "expression":
        # "= ..." continues the previous line's last value; so does a bare expression
        # under a bare expression. A bare expression under an equation starts a new thread.
        if prev.kind in ("expression", "identity") or (cur.continuation and prev.kind == "equation"):
            eq = _link(prev.last, cur.sides[0], max(prev.decimals, cur.decimals))
            if eq is False:
                return StepResult("error", _expr_kind(prev.last, cur.sides[0]), "sympy")
            if len(cur.sides) > 1:
                within = _check_within(cur)
                if within.verdict == "error" or eq is None:
                    return within if within.verdict == "error" else StepResult("unknown")
                return within
            return StepResult("ok", by="sympy") if eq else StepResult("unknown")
        return StepResult("unknown")

    # cur is an equation
    if prev.kind == "equation":
        return _compare_equations(prev, cur)
    if prev.kind == "identity" and len(prev.sides) == 2 and not prev.has_calculus:
        return _compare_equations(StepMath(prev.latex, "equation", prev.sides, [tuple(prev.sides)]), cur)
    return StepResult("unknown")


# Instruction words in front of the equation ("Solve", "Find x if"), not maths. A fixed
# list, because a function name ("sin x = 1/2") must stay part of the equation.
_INSTRUCTION = (r"solve|find|simplify|evaluate|compute|calculate|determine|show|prove|given|if|let|"
                r"for|the|value|values|of|that|when|where|[a-z](?=\s+(?:if|when|where|given)\b)")
_PROBLEM_WORDS = re.compile(r"^(\\text\{[^{}]*\}|[A-Za-z][A-Za-z ,]*:|(?:(?:" + _INSTRUCTION + r")\b[\s,]*)+(?=[^=]*=))\s*",
                            re.IGNORECASE)


def problem_equation(problem: str | None) -> StepMath | None:
    """The problem's equation, when it states exactly one ("Solve: 2x + 5 = 11"), so
    the first line is checked against it. A system, a list or anything unclear gives
    None: comparing a line with half of a system would be a false alarm."""
    if not problem:
        return None
    text = problem.strip()
    while (m := _PROBLEM_WORDS.match(text)) and m.end() > 0:
        text = text[m.end():]
    if not text or re.search(r"\\text|\\\\|\band\b|;", text):
        return None
    eq = analyse(text)
    return eq if eq.kind == "equation" and len(eq.variants) == 1 else None


def _squash(text: str) -> str:
    return re.sub(r"\\text|[\s{}$]", "", text).lower()


def restates_problem(latex: str, problem: str | None) -> bool:
    """The question copied onto the page ("2x+5y=13 x+2y=7" under "Solve the system...").
    It is not the student's working, so it is not checked: read as one line, a system
    looks like a broken chain."""
    line = _squash(latex)
    return bool(problem) and "=" in line and len(line) >= 5 and line in _squash(problem)


_LONE_OPERATOR = re.compile(r"^\s*\$?\s*(\+|-|\\times|\\pm|\\text\{\s*[+-]\s*\})\s*\$?\s*$")
# A line that starts with a multiplying or adding operator and has more after it.
_GOES_ON = re.compile(r"^\s*\$?\s*(\\times|\\cdot|\\div|\+|\*)\s*\S")


def check_page(page: Page, judge: ModelJudge | None = None) -> tuple[Page, Finding | None]:
    """Mark every step and return the first verified error, if any."""
    analysed = [analyse(s.latex) for s in page.steps]
    # A line of only linking words ("i.e.", "So,") is transparent: the next line
    # is compared with the maths above it.
    connector = [is_connector(s.latex) for s in page.steps]
    prevs: list[StepMath | None] = []
    # The first equation is checked against the problem's equation, when the problem
    # states one in the same unknowns (a substitution like "u = x^2" starts a new thread).
    given = problem_equation(page.problem)
    first = next((a for a, c in zip(analysed, connector) if not c), None)
    last: StepMath | None = None
    if given and first and first.kind == "equation" and _symbols(first) and _symbols(first) <= _symbols(given):
        last = given
    for i, cur in enumerate(analysed):
        prevs.append(last)
        if not connector[i]:
            last = cur
    results = [check_pair(prevs[i], cur) for i, cur in enumerate(analysed)]
    copied = {i for i, s in enumerate(page.steps) if restates_problem(s.latex, page.problem)}
    for i in copied:
        results[i] = StepResult("unknown")
    # A line that goes on in the next one ("13824 = 2 x 2 x ... x 3" / "x 3"): read on its
    # own it is unfinished, so SymPy cannot call it wrong.
    for i in range(len(analysed) - 1):
        if results[i].verdict == "error" and _GOES_ON.match(page.steps[i + 1].latex):
            results[i] = StepResult("unknown")
    # Column arithmetic: rows written one under another with a lone "+" between them,
    # and their total below. The total is not the row above it continued.
    for i, cur in enumerate(analysed):
        if (results[i].verdict == "error" and cur.kind == "expression" and not cur.continuation
                and any(_LONE_OPERATOR.match(s.latex) for s in page.steps[max(0, i - 3):i])):
            results[i] = StepResult("unknown")
    # Wordy lines can still hold plain arithmetic ("so the total is 8 x 2.5 = 18 kg"):
    # SymPy checks that part, and only a mismatch counts (a match proves nothing else).
    for i, cur in enumerate(analysed):
        runs = numeric_sides(page.steps[i].latex) if cur.kind == "unparseable" else None
        places = decimal_places(page.steps[i].latex)
        if runs and any(_close_numbers(a, b, places) is False for run in runs for a, b in pairwise(run)):
            results[i] = StepResult("error", "arithmetic: the two sides are not equal", "sympy")
    first_sympy_error = next((i for i, r in enumerate(results) if r.verdict == "error"), len(results))

    # Lines SymPy could not decide go to the model: lines it cannot parse, and parsed
    # lines it could not settle (values substituted from the question, a new thread).
    # Only those before the first SymPy-verified error (we report the first mistake only).
    candidates = [page.steps[i].id for i in range(first_sympy_error)
                  if results[i].verdict == "unknown" and i not in copied]
    if judge is not None and candidates:
        flagged, kind = judge(page, candidates)
        flagged_at = next((i for i, s in enumerate(page.steps) if s.id == flagged and s.id in candidates), None)
        for i, step in enumerate(page.steps):
            if step.id not in candidates:
                continue
            if flagged_at is None or i < flagged_at:
                results[i] = StepResult("ok", by="model")
            elif i == flagged_at:
                results[i] = StepResult("error", kind or "this step does not follow", "model")

    out: list[Step] = []
    finding: Finding | None = None
    for step, res in zip(page.steps, results):
        status = {"ok": "ok", "error": "error", "unknown": "unverifiable"}[res.verdict]
        verified_by = res.by if res.verdict != "unknown" else None
        out.append(step.model_copy(update={"status": status, "verified_by": verified_by}))
        if res.verdict == "error" and finding is None:
            finding = Finding(step_id=step.id, kind=res.kind or "this step does not follow")
    return page.model_copy(update={"steps": out}), finding
