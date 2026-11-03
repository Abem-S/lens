"""A3 done criterion: planted algebra, calculus and sign errors are caught; correct
derivations come back all `ok`."""

import pytest

from lens.checker import check_page
from lens.contract import Page, Step


def page(*lines: str, problem: str | None = None) -> Page:
    return Page(problem=problem, steps=[Step(id=f"s{i + 1}", latex=l) for i, l in enumerate(lines)])


CORRECT = {
    "linear": ["2x + 5 = 11", "2x = 11 - 5", "2x = 6", "x = 3"],
    "linear_decimals": ["0.5x + 1 = 2", "0.5x = 1", "x = 2"],
    "quadratic_factor": ["x^2 - 5x + 6 = 0", "(x-2)(x-3) = 0", "x = 2 \\text{ or } x = 3"],
    "quadratic_pm": ["x^2 = 9", "x = \\pm 3"],
    "fractions": ["\\frac{x}{3} + 1 = 4", "\\frac{x}{3} = 3", "x = 9"],
    "expand": ["(x+1)^2", "= x^2 + 2x + 1"],
    "simplify_chain": ["\\frac{x^2 - 1}{x - 1}", "\\frac{(x-1)(x+1)}{x-1}", "x + 1"],
    "derivative": ["\\frac{d}{dx}(3x^2 + 2x) = 6x + 2"],
    "chain_rule": ["\\frac{d}{dx}\\sin(x^2) = 2x\\cos(x^2)"],
    "integral_with_c": ["\\int 2x \\, dx = x^2 + C"],
    "definite_integral": ["\\int_0^1 x^2 dx = \\frac{1}{3}"],
    "limit": ["\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1"],
    "rearrange_formula": ["v = u + at", "v - u = at", "t = \\frac{v - u}{a}"],
    "physics_numbers": ["F = ma", "F = 2 \\cdot 3", "F = 6"],  # substitution: s2 unverifiable
    "exponentials": ["e^{2x} = e^{6}", "2x = 6", "x = 3"],
    "distribute": ["3(x-1) = 2x + 4", "3x - 3 = 2x + 4", "x = 7"],
    "reciprocal": ["\\frac{1}{x} = 2", "x = \\frac{1}{2}"],
    "rearrange_line": ["y = mx + c", "m = \\frac{y - c}{x}"],
    "trig_identity": ["\\sin^2 x + \\cos^2 x", "= 1"],
    "multiply_out_fraction": ["\\frac{x+1}{2} = \\frac{x}{3}", "3(x+1) = 2x", "x = -3"],
    "log_rules": ["\\ln(x^2)", "= 2\\ln x"],
    "squaring_adds_root": ["\\sqrt{x} = x - 2", "x = (x-2)^2", "x^2 - 5x + 4 = 0"],
}

PLANTED = {
    # name: (lines, index of first wrong line, expected substring of kind)
    "sign_moving_term": (["2x + 5 = 11", "2x = 11 + 5", "x = 8"], 1, "sign error moving a term"),
    "arithmetic": (["3x = 12", "x = 3"], 1, "different solutions"),
    "divide_wrong": (["4x - 8 = 0", "4x = 8", "x = 4"], 2, "different solutions"),
    "expand_wrong": (["(x+1)^2", "= x^2 + 1"], 1, "not equal"),
    "expression_sign": (["x - (y - 2)", "x - y - 2"], 1, "not equal"),
    "derivative_wrong": (["\\frac{d}{dx}(3x^2 + 2x) = 6x"], 0, "derivative"),
    "power_rule_wrong": (["\\frac{d}{dx} x^3 = 3x^3"], 0, "derivative"),
    "integral_wrong": (["\\int 2x \\, dx = 2x^2 + C"], 0, "integral"),
    "rearrange_wrong": (["v = u + at", "t = \\frac{v + u}{a}"], 1, "sign error"),
    "later_line_wrong": (["2x + 5 = 11", "2x = 6", "x = 2"], 2, "different solutions"),
    "chain_wrong": (["2(x+3) = 2x + 6 = 2x + 3"], 0, "not equal"),
}


@pytest.mark.parametrize("name", CORRECT)
def test_correct_derivations_have_no_error(name):
    checked, finding = check_page(page(*CORRECT[name]))
    assert finding is None, (name, finding)
    assert all(s.status in ("ok", "unverifiable") for s in checked.steps)


@pytest.mark.parametrize(
    "name",
    # quadratic_factor has \text (model territory); physics_numbers substitutes given data.
    [n for n in CORRECT if n not in ("quadratic_factor", "physics_numbers")],
)
def test_correct_math_is_verified_ok_by_sympy(name):
    checked, _ = check_page(page(*CORRECT[name]))
    assert [s.status for s in checked.steps] == ["ok"] * len(checked.steps), name
    assert all(s.verified_by == "sympy" for s in checked.steps)


@pytest.mark.parametrize("name", PLANTED)
def test_planted_errors_are_caught_at_the_right_step(name):
    lines, wrong, kind = PLANTED[name]
    checked, finding = check_page(page(*lines))
    assert finding is not None, name
    assert finding.step_id == f"s{wrong + 1}", (name, finding)
    assert kind in finding.kind, (name, finding.kind)
    assert checked.steps[wrong].status == "error"
    assert checked.steps[wrong].verified_by == "sympy"
    assert all(s.status != "error" for s in checked.steps[:wrong])


def test_unparseable_lines_are_unverifiable_without_a_judge():
    checked, finding = check_page(page("\\text{Let } x \\text{ be the speed}", "x \\leq 3"))
    assert finding is None
    assert [s.status for s in checked.steps] == ["unverifiable", "unverifiable"]


def test_model_judges_only_lines_sympy_cannot_parse():
    calls = []

    def judge(pg, candidates):
        calls.append(candidates)
        return candidates[0], "unit mismatch"

    checked, finding = check_page(page("2x = 6", "x = 3", "3\\,\\mathrm{kg} = 3000\\,\\mathrm{mg}"), judge)
    assert calls == [["s3"]]  # one call per page, only the unparseable line is a candidate
    assert finding is not None and finding.step_id == "s3"
    assert checked.steps[2].verified_by == "model"
    assert checked.steps[0].verified_by == "sympy"


def test_parsed_lines_sympy_cannot_settle_go_to_the_judge():
    # s2 substitutes given data: SymPy parses it but cannot decide, so the model may.
    calls = []
    checked, finding = check_page(page("F = ma", "F = 2 \\cdot 3", "F = 6"),
                                  lambda pg, c: calls.append(c) or ("s2", "wrong mass"))
    assert calls == [["s2"]]
    assert finding is not None and finding.step_id == "s2"
    assert checked.steps[1].verified_by == "model"


def test_model_cannot_flag_an_algebra_line():
    # The judge tries to flag a SymPy-checked line; it is ignored.
    _, finding = check_page(page("2x + 5 = 11", "2x = 6", "\\text{so } x = 3"), lambda pg, c: ("s2", "nope"))
    assert finding is None


def test_judge_is_not_asked_about_lines_after_a_sympy_error():
    calls = []
    check_page(page("2x + 5 = 11", "2x = 16", "\\text{so } x = 8"), lambda pg, c: calls.append(c) or (None, ""))
    assert calls == []


def test_lines_before_the_flagged_one_are_ok_by_model():
    checked, finding = check_page(page("\\text{Let } v = 3", "\\text{then } 2v = 7", "\\text{so } v = 3.5 \\text{ m/s}"),
                                  lambda pg, c: ("s2", "doubling error"))
    assert [s.status for s in checked.steps] == ["ok", "error", "unverifiable"]
    assert finding.step_id == "s2" and finding.kind == "doubling error"


def test_lost_root_is_not_claimed_as_error():
    # x^2 = 9 -> x = 3 may be the start of a case split. False alarms cost more.
    checked, finding = check_page(page("x^2 = 9", "x = 3"))
    assert finding is None
    assert checked.steps[1].status == "unverifiable"


@pytest.mark.parametrize("line,wrong", [
    ("\\text{Total weight increased} = (8 \\times 2.5) \\text{kg} = 18 \\text{kg}", True),
    ("\\text{Thus,} \\theta = \\frac{2}{3} \\times 360^{\\circ} = 250^{\\circ}", True),
    ("\\text{Thus,} \\theta = \\frac{2}{3} \\times 360^{\\circ} = 240^{\\circ}", False),
    ("\\text{Area} = 3 \\times 4 = 12 \\, \\text{cm}^2", False),
    ("\\text{so } x = 2 \\text{ and } y = 3", False),  # no arithmetic chain: left to the model
    ("\\text{Answer: Option C}", False),
    ("3 \\, \\text{kg} = 3000 \\, \\text{g}", False),  # a unit conversion, not 3 = 3000
    ("\\text{Speed} = \\frac{100}{4} \\text{m/s} = 20 \\text{m/s}", True),
    ("\\text{Discount} = 20\\% \\text{ of } 50 = 10", False),  # mixed: % on one side only
    ("2^2=-4a(-3), \\text{ i.e., } a=\\frac{1}{3}.", False),  # 4 and 1/3 are not claimed equal
    # Words between numbers join separate statements: SymPy leaves the line to the model
    # rather than read "7 and 2 x 5" as one product.
    ("\\text{so } 3 + 4 = 7 \\text{ and } 2 \\times 5 = 11", False),
])
def test_arithmetic_inside_words_is_checked_by_sympy(line, wrong):
    checked, finding = check_page(page(line))
    if wrong:
        assert finding is not None and checked.steps[0].verified_by == "sympy"
        assert "arithmetic" in finding.kind
    else:
        assert finding is None


# Shapes from FERMAT pages: named lengths, substituted values, rounding, separated
# variables. None of these is a mistake; each once raised (or could raise) a false alarm.
@pytest.mark.parametrize("lines", [
    ["QR = \\sqrt{(4+1)^2 + (2+1)^2} = \\sqrt{25+9} = \\sqrt{34}"],
    ["AB = \\sqrt{34}", "BC = \\sqrt{25+9} = \\sqrt{34}"],
    ["\\int \\frac{dy}{1+y^2} = \\int \\frac{dx}{1+x^2}"],
    ["\\sqrt{34} = 5.83"],
    ["A = \\pi r^2 = \\pi \\cdot 7^2 = 153.94"],
    ["\\frac{22}{7} \\times 49 = 154"],
    ["\\pi \\times 49 = 154"],
    ["\\text{Area} = 3.14 \\times 7^2 = 153.9 \\text{ cm}^2"],
    ["F = ma = 2 \\cdot 3 = 6"],
])
def test_fermat_shapes_are_not_false_alarms(lines):
    _, finding = check_page(page(*lines))
    assert finding is None, (lines, finding)


@pytest.mark.parametrize("lines", [
    ["2x - 3 = x + 2", "\\text{i.e.,}", "2x = x + 5", "\\Rightarrow x = 5"],
    ["x^2 - 5x + 6 = 0", "\\Rightarrow (x-2)(x-3) = 0 \\quad (1)"],
    ["2x = x + 5", "2x - x = x + 5 - x \\text{ (subtracting x from both sides)}", "x = 5"],
    ["\\text{So, } x + 1 = 4", "x = 3"],
    ["(a+b)^2 = a^2 + 2ab", "+b^2)"],  # a fragment of a longer line is not read as maths
    ["\\text{we get: } A + B = 0", "2A + B = 1"],  # a system of equations, not a rearrangement
    ["53240 \\div 5 = 2\\;2\\;2\\;11\\;11\\;11"],  # a factor list whose signs were lost
])
def test_linking_words_and_remarks_are_not_maths(lines):
    _, finding = check_page(page(*lines))
    assert finding is None, (lines, finding)


@pytest.mark.parametrize("lines,wrong", [
    # From FERMAT pages the checker used to miss: linking lines broke the chain,
    # "\Rightarrow" and remarks made lines unparseable, "= ..." after an equation was not linked.
    (["21(n-1)(-3) = 0", "\\text{i.e.,}", "3(n-1) = 21"], 2),
    (["10x \\times \\frac{8}{x} + 18 = 10 \\cdot \\frac{8}{x} + x", "\\Rightarrow 10x^2 \\times 8 + 18x = 80 + x^2"], 1),
    (["2x = x + 5", "2x - x = x + 5 - x \\text{ (subtracting x from both sides)}", "x = 5 + x"], 2),
    (["a_{20} = (20-1)(2-20)(3+20)", "= 19 \\times (-18) \\times 23 + 1 = -7865"], 1),
    (["7 \\times 8 = 54"], 0),  # plain numbers on both sides
    (["\\sqrt{34} = 6.83"], 0),  # rounding does not excuse a wrong value
    (["QR = \\sqrt{(4+1)^2 + (2+1)^2} = \\sqrt{25+4}"], 0),  # past the name, the arithmetic is checked
    (["F = ma = 2 \\cdot 3 = 5"], 0),
    (["x^2 = 49", "\\pi \\cdot 49 = 145"], 1),
])
def test_numeric_claims_are_checked(lines, wrong):
    checked, finding = check_page(page(*lines))
    assert finding is not None and finding.step_id == f"s{wrong + 1}", (lines, finding)
    assert checked.steps[wrong].verified_by == "sympy"


# ---- the first line is checked against the problem's own equation ----

def _first(problem, *lines):
    from lens.checker import check_page
    from lens.contract import Page, Step

    page = Page(problem=problem, steps=[Step(id=f"s{i}", latex=l, bbox=(0, 0, 1, 0.1)) for i, l in enumerate(lines)])
    return check_page(page)


def test_first_line_is_checked_against_the_problem():
    page, finding = _first(r"\text{Solve: }2x + 5 = 11", "2x = 11 + 5", "2x = 16", "x = 8")
    assert finding is not None and finding.step_id == "s0" and page.steps[0].verified_by == "sympy"


def test_correct_first_line_passes_against_the_problem():
    page, finding = _first("Solve: 2x + 5 = 11", "2x = 11 - 5", "2x = 6", "x = 3")
    assert finding is None and page.steps[0].status == "ok"


def test_a_system_in_the_problem_is_not_used():
    _, finding = _first(r"2x + 5y = 13, \; x + 2y = 7", "x = 7 - 2y", "2(7 - 2y) + 5y = 13")
    assert finding is None


def test_a_substitution_starts_a_new_thread():
    _, finding = _first("x^4 - 5x^2 + 4 = 0", "u = x^2", "u^2 - 5u + 4 = 0")
    assert finding is None


def test_no_problem_means_no_extra_check():
    _, finding = _first(None, "2x = 11 + 5", "2x = 16")
    assert finding is None


def test_problem_phrasings():
    from lens.checker import problem_equation as pe

    def sides(t):
        e = pe(t)
        return e and [str(x) for x in e.sides]

    assert sides("Solve 2x + 5 = 11") == ["2*x + 5", "11"]
    assert sides(r"\text{Solve: }2x + 5 = 11") == ["2*x + 5", "11"]
    assert sides("Find x if 3x - 2 = 7") == ["3*x - 2", "7"]
    assert sides("Solve for x: 2x+1=5") == ["2*x + 1", "5"]
    assert sides("x + 5 = 11") == ["x + 5", "11"]  # a leading variable is maths, not a word
    assert sides(r"\sin x = \frac{1}{2}") == ["sin(x)", "1/2"]  # a function name stays
    assert pe(r"2x + 5y = 13, \; x + 2y = 7") is None  # a system is never used


# ---------- two readings of one line (lens.mathparse.same_math) ----------

@pytest.mark.parametrize("a, b", [
    ("2x + 5 = 11", "2x+5=11"),
    ("= 1960000 + 9 - 8400", "=1960000 - 8400 + 9"),  # same value, terms in another order
    ("Area = d_1 . d_2 = 240", "Area = d_{1} \\cdot d_{2} = 240"),
    ("\\text{are } \\sqrt{\\frac{3}{2}}", "are $\\sqrt{\\frac{3}{2}}$"),
    ("\\frac{5-2x}{3} \\leq \\frac{x}{6} -5", "\\frac{5-2x}{3} \\le \\frac{x}{6} -5"),
    ("\\text{we divide }1188\\text{ by }2\\times 2\\times 11 = 42", "\\text{we divide 1188 by 2 \\times 2 \\times 11= 42}"),
])
def test_same_math_reads_alike(a, b):
    from lens.mathparse import same_math

    assert same_math(a, b)


@pytest.mark.parametrize("a, b", [
    ("= 1960000 - 8200 + 9", "= 1960000 - 8400 + 9"),
    ("-(5x^2 - 4y^2 + 6y - 3)", "-(5x^2 - 4y^2 + 6y - 32)"),
    ("\\text{mean} = \\frac{282}{6} = 47 = 0", "\\text{mean} = \\frac{282}{6} - 47 = 0"),
    ("= \\frac{1}{4} \\log |x+1|", "= \\frac{11}{4} \\log |x+1|"),
    ("\\text{we divide 1188 by 44}", "\\text{we divide 1188 by 42}"),
])
def test_same_math_tells_misreadings_apart(a, b):
    from lens.mathparse import same_math

    assert not same_math(a, b)


def test_column_addition_total_is_not_read_as_the_row_above_continued():
    rows = ["7xy+5yz-3zx", "+", "4yz+9zx-4y", "5xy+9yz+3zx+5x-4y"]  # a third row the reader missed
    _, finding = check_page(Page(steps=[Step(id=f"s{i}", latex=t) for i, t in enumerate(rows)]))
    assert finding is None


def test_bare_expressions_without_a_column_still_link():
    rows = ["(x+1)^2", "x^2+2x+2"]
    _, finding = check_page(Page(steps=[Step(id=f"s{i}", latex=t) for i, t in enumerate(rows)]))
    assert finding is not None and finding.step_id == "s1"


def test_the_question_copied_onto_the_page_is_not_checked():
    # Held-out FERMAT img_492: the question's two equations read as one line.
    p = page(r"\text{Question: Solve the system of equations}", "2x+5y=13 x+2y=7", "x = 7 - 2y",
             problem="Solve the system of equations 2x+5y=13 x+2y=7")
    out, finding = check_page(p)
    assert finding is None and out.steps[1].status == "unverifiable"


def test_a_wrong_step_that_only_resembles_the_question_is_still_checked():
    _, finding = check_page(page("2x + 5 = 11", "2x = 11 + 5", problem="Solve 2x + 5 = 11"))
    assert finding is not None and finding.step_id == "s2"


def test_inverse_trig_outside_its_range_is_not_claimed():
    # sin^-1(sin 2t) = 2t holds where the inverse is taken; SymPy's sample points miss it.
    _, finding = check_page(page("= \\sin^{-1}(2\\sin\\theta\\cos\\theta) = \\sin^{-1}(\\sin2\\theta) = 2\\theta"))
    assert finding is None


def test_line_that_goes_on_in_the_next_is_not_claimed():
    _, finding = check_page(page(
        "13824 = 2 \\times 2 \\times 2 \\times 2 \\times 2 \\times 2 \\times 2 \\times 2 \\times 3 \\times 3",
        "\\times 3"))
    assert finding is None
    _, finding = check_page(page("13824 = 2 \\times 2 \\times 3"))
    assert finding is not None  # finished as written: still a mistake


def test_division_by_zero_read_above_is_not_a_claim():
    _, finding = check_page(page("= \\frac{1}{2} + \\left(\\frac{0 - \\frac{1}{2}}{0}\\right)", "= \\frac{1}{4}"))
    assert finding is None
