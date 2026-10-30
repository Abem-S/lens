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


def test_model_cannot_flag_an_algebra_line():
    # The judge tries to flag a SymPy-checked line; it is ignored.
    _, finding = check_page(page("2x + 5 = 11", "2x = 6", "\\text{so } x = 3"), lambda pg, c: ("s2", "nope"))
    assert finding is None


def test_judge_is_not_asked_about_lines_after_a_sympy_error():
    calls = []
    check_page(page("2x + 5 = 11", "2x = 16", "\\text{so } x = 8"), lambda pg, c: calls.append(c) or (None, ""))
    assert calls == []


def test_lines_before_the_flagged_one_are_ok_by_model():
    checked, finding = check_page(page("\\text{Let } v = 3", "\\text{then } 2v = 7", "\\text{so } v = 3.5"),
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
