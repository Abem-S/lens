"""Turn one handwritten line (as LaTeX) into SymPy objects, or refuse.

Refusing is the safe outcome: a line we cannot parse with confidence goes to the
model and is never claimed as a SymPy-verified mistake.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

import sympy as sp
from sympy.core.function import AppliedUndef
from sympy.parsing.latex import parse_latex

# LaTeX commands we trust the parser with. Anything else (\text, \vec, \Delta,
# \mathrm, \partial, ...) makes the line unparseable.
_ALLOWED_COMMANDS = {
    "frac", "dfrac", "tfrac", "sqrt", "cdot", "times", "div", "left", "right",
    "sin", "cos", "tan", "sec", "csc", "cot", "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh", "ln", "log", "exp", "int", "lim", "to", "infty",
    "pi", "theta", "alpha", "beta", "phi", "omega", "lambda", "mu", "rho",
    "sigma", "tau", "pm", "mp",
}
_INEQUALITY = re.compile(r"(\\le|\\ge|\\leq|\\geq|\\neq|\\ne|<|>|\\lt|\\gt|\\approx)(?![a-zA-Z])")
_COMMAND = re.compile(r"\\([a-zA-Z]+)")
_CALCULUS = (sp.Derivative, sp.Integral, sp.Limit)

Kind = Literal["equation", "expression", "identity", "unparseable"]


@dataclass
class StepMath:
    latex: str
    kind: Kind
    # Each side of the line split on top-level "=". For an equation with \pm,
    # `variants` holds one (lhs, rhs) pair per sign.
    sides: list[sp.Expr] = field(default_factory=list)
    variants: list[tuple[sp.Expr, sp.Expr]] = field(default_factory=list)
    continuation: bool = False  # line starts with "=" and continues the previous one
    reason: str = ""

    @property
    def has_calculus(self) -> bool:
        return any(s.has(*_CALCULUS) for s in self.sides)

    @property
    def last(self) -> sp.Expr:
        return self.sides[-1]


def _clean(latex: str) -> str:
    s = latex.strip().strip("$").strip()
    s = s.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "").replace(r"\ ", " ")
    s = s.replace(r"\displaystyle", "")
    s = re.sub(r"\\(begin|end)\{[a-z*]+\}", "", s)
    s = s.rstrip(".").strip()
    return s


def _split_top_level(s: str, sep: str = "=") -> list[str]:
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        if ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts]


def _top_level_comma(s: str) -> bool:
    depth = 0
    for i, ch in enumerate(s):
        if ch in "{([":
            depth += 1
        elif ch in "})]":
            depth -= 1
        # A comma at the top level is a list, unless it is a thousands separator (1,000).
        elif ch == "," and depth == 0 and not (
            i > 0 and s[i - 1].isdigit() and re.match(r"\d{3}(?!\d)", s[i + 1 :])
        ):
            return True
    return False


def _fix_symbols(expr: sp.Expr) -> sp.Expr:
    """Map parser artefacts to the meaning a student intends."""
    subs = {}
    for sym in expr.free_symbols:
        if sym.name == "e":
            subs[sym] = sp.E
    if subs:
        expr = expr.xreplace(subs)
    # Students write decimals; work in exact rationals so 0.5 == 1/2.
    floats = expr.atoms(sp.Float)
    if floats:
        expr = expr.xreplace({f: sp.nsimplify(f, rational=True) for f in floats})
    return expr


def _parse_side(s: str) -> sp.Expr:
    if not s:
        raise ValueError("empty side")
    expr = parse_latex(s)
    if isinstance(expr, (sp.Equality, sp.Rel)):
        raise ValueError("relation inside a side")
    expr = _fix_symbols(expr)
    for sym in expr.free_symbols:
        if "'" in sym.name or len(sym.name) > 1 and "_" not in sym.name and sym.name not in _GREEK:
            raise ValueError(f"unknown symbol {sym.name}")
    for app in expr.atoms(AppliedUndef):
        raise ValueError(f"undefined function {app.func}")
    for d in expr.atoms(sp.Derivative):
        if isinstance(d.expr, sp.Symbol):  # dy/dx: y is secretly a function of x
            raise ValueError("Leibniz derivative of a bare symbol")
    return expr


_GREEK = {"pi", "theta", "alpha", "beta", "phi", "omega", "lambda", "mu", "rho", "sigma", "tau"}


def analyse(latex: str) -> StepMath:
    s = _clean(latex)
    if not s:
        return StepMath(latex, "unparseable", reason="empty line")
    for cmd in _COMMAND.findall(s):
        if cmd not in _ALLOWED_COMMANDS:
            return StepMath(latex, "unparseable", reason=f"unsupported command \\{cmd}")
    if _INEQUALITY.search(s):
        return StepMath(latex, "unparseable", reason="inequality")
    if _top_level_comma(s):
        return StepMath(latex, "unparseable", reason="list or comma")

    continuation = s.startswith("=")
    if continuation:
        s = s[1:].strip()
    raw_sides = _split_top_level(s)

    has_pm = r"\pm" in s or r"\mp" in s
    try:
        if has_pm:
            if len(raw_sides) != 2 or continuation:
                return StepMath(latex, "unparseable", reason="\\pm outside a simple equation")
            variants = []
            for sign in ("+", "-"):
                other = "-" if sign == "+" else "+"
                l_, r_ = (p.replace(r"\pm", sign).replace(r"\mp", other) for p in raw_sides)
                variants.append((_parse_side(l_), _parse_side(r_)))
            return StepMath(latex, "equation", sides=list(variants[0]), variants=variants)
        sides = [_parse_side(p) for p in raw_sides]
    except Exception as exc:  # parser raises many types; all mean "not confident"
        return StepMath(latex, "unparseable", reason=f"parse failed: {exc}"[:200])

    if continuation or len(sides) == 1:
        kind: Kind = "expression"
    elif len(sides) == 2 and not any(x.has(*_CALCULUS) for x in sides):
        kind = "equation"
    else:
        kind = "identity"  # a chain like a = b = c, or a calculus claim d/dx(..) = ..
    sm = StepMath(latex, kind, sides=sides, continuation=continuation)
    if kind == "equation":
        sm.variants = [(sides[0], sides[1])]
    return sm
