"""Read a transcribed step aloud: LaTeX -> plain spoken maths ("2x = 11 - 5" ->
"2 x equals 11 minus 5").

Used by "read my page aloud", for students who cannot easily see the page. It only
says what the student wrote, word for word, so it never adds a correction or an
answer. Plain text rewriting (no model): numbers stay as digits, which the speech
model reads naturally.
"""

from __future__ import annotations

import re

GREEK = {
    "alpha": "alpha", "beta": "beta", "gamma": "gamma", "delta": "delta", "epsilon": "epsilon",
    "varepsilon": "epsilon", "zeta": "zeta", "eta": "eta", "theta": "theta", "vartheta": "theta",
    "iota": "iota", "kappa": "kappa", "lambda": "lambda", "mu": "mu", "nu": "nu", "xi": "xi", "pi": "pi",
    "rho": "rho", "sigma": "sigma", "tau": "tau", "phi": "phi", "varphi": "phi", "chi": "chi", "psi": "psi",
    "omega": "omega", "Gamma": "capital gamma", "Delta": "capital delta", "Theta": "capital theta",
    "Lambda": "capital lambda", "Sigma": "capital sigma", "Phi": "capital phi", "Psi": "capital psi",
    "Omega": "capital omega", "Pi": "capital pi",
}
WORDS = {
    "times": " times ", "cdot": " times ", "div": " divided by ", "pm": " plus or minus ", "mp": " minus or plus ",
    "neq": " is not equal to ", "ne": " is not equal to ", "leq": " is less than or equal to ",
    "le": " is less than or equal to ", "geq": " is greater than or equal to ", "ge": " is greater than or equal to ",
    "approx": " is approximately ", "equiv": " is equivalent to ", "propto": " is proportional to ",
    "to": " goes to ", "rightarrow": " gives ", "Rightarrow": " so ", "implies": " so ", "longrightarrow": " gives ",
    "iff": " if and only if ", "therefore": " therefore ", "because": " because ", "infty": " infinity ",
    "partial": " partial ", "nabla": " del ", "circ": " degrees ", "degree": " degrees ", "angle": " angle ",
    "perp": " is perpendicular to ", "parallel": " is parallel to ", "in": " in ", "ldots": " and so on ",
    "cdots": " and so on ", "dots": " and so on ", "prime": " prime ",
}
FUNCS = {
    "sin": "sine", "cos": "cos", "tan": "tan", "sec": "sec", "csc": "cosec", "cot": "cot", "arcsin": "arc sine",
    "arccos": "arc cos", "arctan": "arc tan", "sinh": "hyperbolic sine", "cosh": "hyperbolic cos", "tanh": "hyperbolic tan", "ln": "l n",
    "log": "log", "exp": "e to the", "det": "determinant of", "max": "max", "min": "min",
}
_DROP = re.compile(r"\\(?:(?:left|right|big|Big|bigg|Bigg|displaystyle|quad|qquad)(?![A-Za-z])|[,;!: ])|\$")


def _group(s: str, i: int) -> tuple[str, int]:
    """The argument starting at s[i]: a {braced group} or one character / command."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return "", i
    if s[i] == "{":
        depth, j = 0, i
        while j < len(s):
            depth += {"{": 1, "}": -1}.get(s[j], 0)
            if depth == 0:
                return s[i + 1:j], j + 1
            j += 1
        return s[i + 1:], len(s)
    if s[i] == "\\":
        m = re.match(r"\\[A-Za-z]+", s[i:])
        if m:
            return m.group(0), i + len(m.group(0))
    return s[i], i + 1


def _simple(spoken: str) -> bool:
    return len(spoken.split()) <= 2


def _power(exp: str) -> str:
    e = _speak(exp).strip()
    if e == "2":
        return " squared "
    if e == "3":
        return " cubed "
    if e in ("degrees", "circ"):
        return " degrees "
    if e == "prime":
        return " prime "
    if _simple(e):
        return f" to the power {e} "
    return f" to the power {e}, "


def _speak(s: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if not m:  # \{ \} \% and the like
                i += 2
                out.append({"%": " percent ", "{": " ", "}": " "}.get(s[i - 1:i], " "))
                continue
            name = m.group(1)
            i += len(m.group(0))
            if name in ("frac", "dfrac", "tfrac"):
                num, i = _group(s, i)
                den, i = _group(s, i)
                n, d = _speak(num).strip(), _speak(den).strip()
                if re.fullmatch(r"d\s*\w?", n) and re.fullmatch(r"d\s*\w", d):
                    out.append(f" {n} by {d} ")  # dy/dx -> "d y by d x"
                elif _simple(n) and _simple(d):
                    out.append(f" {n} over {d} ")
                else:
                    out.append(f" the fraction {n}, over {d}, ")
            elif name == "sqrt":
                root = ""
                if i < len(s) and s[i] == "[":
                    j = s.index("]", i) if "]" in s[i:] else len(s)
                    root, i = s[i + 1:j], j + 1
                arg, i = _group(s, i)
                a = _speak(arg).strip()
                lead = "the square root of" if not root else f"the {_speak(root).strip()} root of"
                out.append(f" {lead} {a} " if _simple(a) else f" {lead} {a}, ")
            elif name in ("text", "mathrm", "textrm", "mathbf", "mathit", "operatorname", "textbf", "boldsymbol",
                          "vec", "hat", "bar", "overline"):
                arg, i = _group(s, i)
                spoken = arg if name.startswith("text") or name == "operatorname" else _speak(arg)
                suffix = {"vec": " vector", "hat": " hat", "bar": " bar", "overline": " bar"}.get(name, "")
                out.append(f" {spoken.strip()}{suffix} ")
            elif name == "int":
                lo = hi = ""
                while i < len(s) and s[i] in "_^":
                    which = s[i]
                    arg, i = _group(s, i + 1)
                    lo, hi = (arg, hi) if which == "_" else (lo, arg)
                if lo or hi:
                    out.append(f" the integral from {_speak(lo).strip()} to {_speak(hi).strip()} of ")
                else:
                    out.append(" the integral of ")
            elif name in ("sum", "prod"):
                lo = hi = ""
                while i < len(s) and s[i] in "_^":
                    which = s[i]
                    arg, i = _group(s, i + 1)
                    lo, hi = (arg, hi) if which == "_" else (lo, arg)
                word = "sum" if name == "sum" else "product"
                out.append(f" the {word} from {_speak(lo).strip()} to {_speak(hi).strip()} of " if lo or hi
                           else f" the {word} of ")
            elif name == "lim":
                sub = ""
                if i < len(s) and s[i] == "_":
                    sub, i = _group(s, i + 1)
                out.append(f" the limit as {_speak(sub).strip()} of " if sub else " the limit of ")
            elif name in FUNCS:
                out.append(f" {FUNCS[name]} ")
            elif name in GREEK:
                out.append(f" {GREEK[name]} ")
            elif name in WORDS:
                out.append(WORDS[name])
            else:
                out.append(f" {name} ")
            continue
        if c == "^":
            arg, i = _group(s, i + 1)
            out.append(_power(arg))
            continue
        if c == "_":
            arg, i = _group(s, i + 1)
            sub = _speak(arg).strip()
            out.append(f" {sub} " if _simple(sub) else f" sub {sub}, ")
            continue
        if c in "{}":
            i += 1
            out.append(" ")
            continue
        if c == "-" and s[i:i + 2] == "->":
            out.append(" gives ")
            i += 2
            continue
        sym = {"=": " equals ", "+": " plus ", "-": " minus ", "*": " times ", "/": " over ", "<": " is less than ",
               ">": " is greater than ", "(": " open bracket ", ")": " close bracket ", "[": " open bracket ",
               "]": " close bracket ", "|": " modulus ", "!": " factorial ", "%": " percent ", "&": " ",
               "~": " ", "'": " prime "}.get(c)
        if sym is not None:
            out.append(sym)
            i += 1
            continue
        if c.isdigit() or (c == "." and i + 1 < len(s) and s[i + 1].isdigit()):
            m = re.match(r"\d+(?:,\d{3})*(?:\.\d+)?|\.\d+", s[i:])
            out.append(f" {m.group(0)} ")
            i += len(m.group(0))
            continue
        if c.isalpha():
            # A run of letters: a real word stays whole, single-letter variables are said apart ("xy" -> "x y").
            m = re.match(r"[A-Za-z]+", s[i:])
            word = m.group(0)
            i += len(word)
            out.append(f" {word} " if len(word) > 3 and word.lower() in _ENGLISH else " " + " ".join(word) + " ")
            continue
        out.append(" " if c.isspace() else f" {c} ")
        i += 1
    return "".join(out)


# Words students write in their working that must not be spelled out letter by letter.
_ENGLISH = {
    "then", "therefore", "hence", "thus", "since", "where", "when", "with", "from", "let", "solve", "find", "given",
    "area", "volume", "speed", "time", "mass", "force", "work", "energy", "power", "total", "answer", "sub",
    "substitute", "into", "both", "sides", "divide", "multiply", "add", "subtract", "and", "true", "false",
    "check", "rate", "length", "width", "height", "base", "side", "angle", "cost", "price", "profit", "loss",
    "this", "that", "which", "because", "means", "so", "units", "metres", "meters", "seconds", "volts", "amps",
    "ohms", "joules", "watts", "newtons", "kilograms", "grams", "litres", "liters", "moles", "mole", "current",
    "voltage", "resistance", "charge", "distance", "velocity", "acceleration", "density", "pressure",
}


def speak_latex(latex: str) -> str:
    """One step as it would be read aloud. Empty input gives an empty string."""
    s = _DROP.sub(" ", latex or "")
    s = s.replace("\\\\", " ").replace("−", "-").replace("×", " times ").replace("÷", " divided by ")
    spoken = re.sub(r"\s+", " ", _speak(s)).strip(" ,")
    spoken = re.sub(r"\s+,", ",", spoken)
    spoken = re.sub(r",(\s*,)+", ",", spoken)
    return spoken


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = {2: "twenty", 3: "thirty", 4: "forty", 5: "fifty"}


def count_words(n: int) -> str:
    """Small counts in words ("Step three"), so Lens's own framing never says a digit that
    could be mistaken for an answer."""
    if 0 <= n < 20:
        return _ONES[n]
    if 20 <= n < 60:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")
    return str(n)


def plain_text(latex: str) -> str:
    """A problem statement for the screen and the summary file: words stay words
    ("\\text{Solve } 2x + 5 = 11" -> "Solve 2x + 5 = 11"); other LaTeX is said in words."""
    s = re.sub(r"\\(?:text|textrm|mathrm|textbf)\{([^{}]*)\}", r"\1", latex or "").replace("$", "")
    s = re.sub(r"\s+", " ", s).strip()
    return speak_latex(s) if "\\" in s else s
