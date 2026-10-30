"""A3 Step checker: does each line follow from the one before it?

Rules (AGENTS.md, hard rule 4): a claimed mistake is a verified mistake. If SymPy
can parse both lines, only SymPy may call a step wrong; when it cannot decide, the
step is "unverifiable" and nothing is said. Lines SymPy cannot parse (words,
units, chemistry, inequalities) go to the model judge, if one is given, and are
marked verified_by="model".
"""

from __future__ import annotations

import cmath
import random
from collections.abc import Callable
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import sympy as sp

from lens.contract import Finding, Page, Step
from lens.mathparse import StepMath, analyse, numeric_sides

Verdict = Literal["ok", "error", "unknown"]

# judge(page, candidate step ids) -> (id of the first wrong candidate | None, kind).
# The judge sees the whole page for context but may only flag a candidate: a line
# SymPy could not parse. It answers once per page.
ModelJudge = Callable[[Page, list[str]], tuple[str | None, str]]

_SAMPLES = 6
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
            kind = _sign_error_kind(prev, cur) or "the rearranged formula is not equivalent to the line above"
            return StepResult("error", kind, "sympy")
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


def _check_within(cur: StepMath) -> StepResult:
    """A chain a = b = c, or a calculus claim, must hold on its own."""
    for a, b in zip(cur.sides, cur.sides[1:]):
        var = _indefinite_var(a) or _indefinite_var(b)
        eq = equivalent(a, b, up_to_constant_in=var)
        if eq is False:
            return StepResult("error", _calculus_kind(a) if a.has(sp.Derivative, sp.Integral, sp.Limit) else
                              "the two sides of this line are not equal", "sympy")
        if eq is None:
            return StepResult("unknown")
    return StepResult("ok", by="sympy")


def _expr_kind(a: sp.Expr, b: sp.Expr) -> str:
    if equivalent(a, -b) is True:
        return "sign error"
    return "this expression is not equal to the one above"


# ---------- a pair of lines ----------

def check_pair(prev: StepMath | None, cur: StepMath) -> StepResult:
    if cur.kind == "unparseable":
        return StepResult("unknown", by=None)

    # Some "equations" are identities ((x+1)^2 = x^2+2x+1): treat as a claim.
    if cur.kind == "equation" and len(cur.variants) == 1 and equivalent(*cur.variants[0]) is True:
        cur = StepMath(cur.latex, "identity", sides=cur.sides, continuation=cur.continuation)

    if cur.kind == "identity":
        within = _check_within(cur)
        if within.verdict != "ok" or prev is None or prev.kind == "unparseable":
            return within
        # Also link to the previous line when this one continues its value.
        if prev.kind in ("expression", "identity") and not cur.has_calculus:
            linked = equivalent(prev.last, cur.sides[0])
            if linked is False:
                return StepResult("error", _expr_kind(prev.last, cur.sides[0]), "sympy")
        return within

    if prev is None or prev.kind == "unparseable":
        return StepResult("ok", by="sympy") if cur.kind in ("expression", "equation") else StepResult("unknown")

    if cur.kind == "expression":
        if prev.kind in ("expression", "identity"):
            eq = equivalent(prev.last, cur.sides[0])
            if eq is False:
                return StepResult("error", _expr_kind(prev.last, cur.sides[0]), "sympy")
            if eq is None:
                return StepResult("unknown")
            if len(cur.sides) > 1:
                return _check_within(cur)
            return StepResult("ok", by="sympy")
        return StepResult("unknown")  # equation then a bare expression: new thread

    # cur is an equation
    if prev.kind == "equation":
        return _compare_equations(prev, cur)
    if prev.kind == "identity" and len(prev.sides) == 2 and not prev.has_calculus:
        return _compare_equations(StepMath(prev.latex, "equation", prev.sides, [tuple(prev.sides)]), cur)
    return StepResult("unknown")


def check_page(page: Page, judge: ModelJudge | None = None) -> tuple[Page, Finding | None]:
    """Mark every step and return the first verified error, if any."""
    analysed = [analyse(s.latex) for s in page.steps]
    results = [check_pair(analysed[i - 1] if i > 0 else None, cur) for i, cur in enumerate(analysed)]
    # Wordy lines can still hold plain arithmetic ("so the total is 8 x 2.5 = 18 kg"):
    # SymPy checks that part, and only a mismatch counts (a match proves nothing else).
    for i, cur in enumerate(analysed):
        runs = numeric_sides(page.steps[i].latex) if cur.kind == "unparseable" else None
        if runs and any(equivalent(a, b) is False for run in runs for a, b in pairwise(run)):
            results[i] = StepResult("error", "arithmetic: the two sides are not equal", "sympy")
    first_sympy_error = next((i for i, r in enumerate(results) if r.verdict == "error"), len(results))

    # Only lines SymPy cannot parse go to the model, and only those before the
    # first SymPy-verified error (we report the first mistake only).
    candidates = [page.steps[i].id for i in range(first_sympy_error) if analysed[i].kind == "unparseable"]
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
