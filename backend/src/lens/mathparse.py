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
    decimals: int = 0  # most decimal places written in the line: how far a student may have rounded

    @property
    def has_calculus(self) -> bool:
        return any(s.has(*_CALCULUS) for s in self.sides)

    @property
    def last(self) -> sp.Expr:
        return self.sides[-1]


def decimal_places(latex: str) -> int:
    """Most digits written after a decimal point anywhere in the line (0 if none)."""
    return max((len(m) for m in re.findall(r"\d\.(\d+)", latex)), default=0)


# Words that only say "this line follows from the one above".
_CONNECTOR_WORDS = r"(?:i\.?\s*e\.?|so|then|thus|hence|therefore|we\s+have|we\s+get|which\s+gives|this\s+gives|giving|that\s+is)"
_LEAD_ARROW = re.compile(r"^(\\(Rightarrow|implies|Longrightarrow|therefore|Leftrightarrow|iff)(?![a-zA-Z])\s*)+")
_LEAD_CONNECTOR = re.compile(r"^\\text\{\s*" + _CONNECTOR_WORDS + r"\s*[,.:;]?\s*\}\s*", re.IGNORECASE)
# A trailing remark: "\text{ (subtracting x from both sides)}", "\quad (1)", "\text{,}".
_TRAIL_REMARK = re.compile(r"(\s*(\\q?quad|~)\s*)*(\\text\{\s*(\(.*?\)|\.\.\.\s*\(\d+\)|[,.;:]*)\s*\}|\(\d+\))\s*$")
_CONNECTOR_LINE = re.compile(r"^\s*(and\s+|now\s+)?" + _CONNECTOR_WORDS + r"\s*[,.:;]?\s*$", re.IGNORECASE)


def is_connector(latex: str) -> bool:
    """A line of only linking words ("i.e.", "So,", "Therefore,"): no claim of its own."""
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", latex.strip().strip("$"))
    text = _LEAD_ARROW.sub("", text.strip())
    return bool(_CONNECTOR_LINE.match(text))


def _balanced(s: str) -> bool:
    depth = 0
    for ch in s:
        depth += ch in "{(["
        depth -= ch in "})]"
        if depth < 0:
            return False
    return depth == 0


def _clean(latex: str) -> str:
    s = latex.strip().strip("$").strip()
    s = s.replace(r"\,", " ").replace(r"\;", " ").replace(r"\!", "").replace(r"\ ", " ")
    s = s.replace(r"\displaystyle", "")
    s = re.sub(r"\\(begin|end)\{[a-z*]+\}", "", s)
    # Linking words and arrows in front ("\Rightarrow", "\text{i.e., }") and a
    # remark at the end ("\text{ (subtracting x from both sides)}") are not maths.
    prev = None
    while prev != s:
        prev = s
        s = _LEAD_CONNECTOR.sub("", _LEAD_ARROW.sub("", s.strip())).strip()
        s = _TRAIL_REMARK.sub("", s).strip()
    s = s.rstrip(".,;").strip()
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
        elif sym.name == "pi":
            subs[sym] = sp.pi
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
    if not _balanced(s):
        return StepMath(latex, "unparseable", reason="unbalanced brackets: part of a longer line")
    if re.search(r"\d\s+\d", s):
        # "2 2 2 11": numbers side by side are a factor list with lost signs, not one number.
        return StepMath(latex, "unparseable", reason="numbers side by side")
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
    sm = StepMath(latex, kind, sides=sides, continuation=continuation, decimals=decimal_places(s))
    if kind == "equation":
        sm.variants = [(sides[0], sides[1])]
    return sm


_TEXT_GROUP = re.compile(r"\\(text|mathrm|textbf|mbox|operatorname)\s*\{([^{}]*)\}(\s*\^\s*(\{[^{}]*\}|\d))?")
_DEGREE = re.compile(r"\^\s*\{?\s*\\circ\s*\}?|\\degree")
_PERCENT = re.compile(r"\\%")


def numeric_sides(latex: str) -> list[list[sp.Expr]] | None:
    """Pull plain arithmetic out of a wordy line, for a SymPy check.

    "Total weight = (8 \\times 2.5)\\,\\text{kg} = 18\\,\\text{kg}" -> [20, 18].
    Words and units (with their exponents) are dropped, but only sides that carry
    the same unit are compared, so a conversion like 3 kg = 3000 g is never read
    as 3 = 3000. None unless at least two numeric sides are found.
    """
    s = _clean(latex)
    if _INEQUALITY.search(s) or "\\pm" in s:
        return None
    runs: list[list[sp.Expr]] = [[]]
    units: set[str] = set()
    for part in _split_top_level(s):
        unit = " ".join(m.group(2).strip() + (m.group(3) or "").replace(" ", "") for m in _TEXT_GROUP.finditer(part))
        if _DEGREE.search(part):
            unit += " deg"
        if _PERCENT.search(part):
            unit += " %"
        # Words are allowed only as a leading label or a trailing unit; words between
        # numbers ("7 and 2 x 5") join separate statements, so the chain breaks there.
        inner = _TEXT_GROUP.sub("\x00", part).strip()
        if "\x00" in inner.strip("\x00 ,;:.").strip():
            core = inner.strip("\x00 ,;:.")
            if "\x00" in core:
                runs.append([])
                continue
        body = _PERCENT.sub(" ", _DEGREE.sub(" ", _TEXT_GROUP.sub(" ", part))).strip().strip(",;:").strip()
        if (not body or "," in body or re.search(r"\d\s+\d", body)
                or re.search(r"[A-Za-z]", re.sub(r"\\(frac|dfrac|times|cdot|div|sqrt|left|right)", "", body))):
            runs.append([])  # a word, a variable or a list: not plain arithmetic; the chain breaks
            continue
        try:
            v = _fix_symbols(parse_latex(body))
        except Exception:
            runs.append([])
            continue
        if isinstance(v, sp.Expr) and not v.free_symbols and v.is_number:
            runs[-1].append(v)
            units.add(unit.strip())
        else:
            runs.append([])
    if len(units) > 1:
        return None  # different units on different sides: a conversion, not arithmetic
    runs = [r for r in runs if len(r) >= 2]
    return runs or None
