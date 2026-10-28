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

    def judge(prev, cur, problem):
        calls.append(cur)
        return "error", "unit mismatch"

    checked, finding = check_page(page("2x = 6", "x = 3", "3\\,\\mathrm{kg} = 3000\\,\\mathrm{mg}"), judge)
    assert calls == ["3\\,\\mathrm{kg} = 3000\\,\\mathrm{mg}"]
    assert finding is not None and finding.step_id == "s3"
    assert checked.steps[2].verified_by == "model"
    assert checked.steps[0].verified_by == "sympy"


def test_model_cannot_overrule_sympy_on_algebra():
    # The judge would call everything wrong; algebra lines never reach it.
    _, finding = check_page(page("2x + 5 = 11", "2x = 6"), lambda *a: ("error", "nope"))
    assert finding is None


def test_lost_root_is_not_claimed_as_error():
    # x^2 = 9 -> x = 3 may be the start of a case split. False alarms cost more.
    checked, finding = check_page(page("x^2 = 9", "x = 3"))
    assert finding is None
    assert checked.steps[1].status == "unverifiable"
