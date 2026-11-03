"""What is on the paper now: the steps panel and the checks follow the student's
current attempt at the current problem, not everything ever read.

A live session reads the same page many times. Between reads the student may
- start over on the same problem (a second method, or back to an earlier step and
  on from there) without crossing anything out: the abandoned steps would otherwise
  sit above the new ones, and the first new step would be "checked" against the last
  abandoned one;
- start a different problem (a new page, a new question written below, or they say so).

Rules (plain text and positions only, no model, so they cost nothing and cannot
claim a mistake by themselves):
- A step written the same as an earlier step that is not one of the two just above
  it, or that copies the problem's own equation after other working, starts the
  attempt again from there: the steps between are the abandoned attempt and leave the
  list. (The line just above read again is a re-read, not a restart; lens.linecheck
  drops those already.)
- A line that states a new question ("Solve ...", "Find ...", "Q2", "2)") after other
  working starts a new problem.
- A reading of a different problem statement than before, or a page that shares less
  than half of its steps with the last reading, is a new problem. A hand-held page
  that sways moves every box but keeps the text, so these rules look at text, never
  at where a step sits on the frame.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher

from lens.checker import restates_problem
from lens.contract import Page, Step
from lens.mathparse import is_connector, same_math

# Words that open a new question when they start a line.
_QUESTION = re.compile(
    r"^\s*(?:\\text\{\s*)?(?:"
    r"(?:Q|Question|Problem|Exercise|Ex\.?|Example|Task)\s*\.?\s*\d+"
    r"|\d+\s*[).]\s*(?:\\text\{\s*)?(?:Solve|Find|Prove|Show|Evaluate|Simplify|Calculate|Compute|Determine|Factori[sz]e|Differentiate|Integrate)"
    r"|(?:Solve|Find|Prove|Show that|Evaluate|Simplify|Calculate|Compute|Determine|Factori[sz]e|Differentiate|Integrate)\b"
    r")",
    re.IGNORECASE,
)


def step_key(latex: str) -> str:
    return re.sub(r"\s+|\\[,;! ]|[{}]", "", latex).lower()


_key = step_key


def same_line(a: str, b: str) -> bool:
    """Two readings of one written line (identical text, or the same maths side by side)."""
    return _key(a) == _key(b) or same_math(a, b)


def copied(a: str, b: str) -> bool:
    """The same line written again, as text (not merely equal in value: a factorised line
    equals the expanded one above it, and that is working, not a restart)."""
    return _key(a) == _key(b) and len(_key(a)) >= 3


def states_question(latex: str) -> bool:
    """A line that opens a new question rather than working on the current one."""
    return bool(_QUESTION.match(latex.strip().strip("$")))


def _plain(s: str | None) -> str:
    return re.sub(r"[^a-z0-9=+\-*/^]", "", re.sub(r"\\text\{([^{}]*)\}", r"\1", s or "").lower())


def other_problem(before: str | None, now: str | None) -> bool:
    """Two problem statements that are not two readings of the same one."""
    a, b = _plain(before), _plain(now)
    if len(a) < 4 or len(b) < 4:
        return False
    if a in b or b in a:  # one reading kept more of the wording ("Solve for x: 2x+5=11" / "2x+5=11")
        return False
    ratio = SequenceMatcher(None, a, b, autojunk=False).ratio()
    da, db = re.findall(r"\d+", a), re.findall(r"\d+", b)
    if da and da == db:  # the same numbers in the same order: worded differently
        return False
    # Other numbers and not nearly the same text: another question ("2x+5=11" / "3x-1=8");
    # one digit misread in a long statement is still the same one.
    return ratio < 0.6 or (da != db and ratio < 0.85)


def _has_math(latex: str) -> bool:
    """Working, not a label such as "Question: Prove that" (the question copied under a
    label is not a restart)."""
    return bool(re.search(r"[=<>]", re.sub(r"\\text\{[^{}]*\}", "", latex)))


def restart_at(steps: list[Step], problem: str | None) -> tuple[int, int] | None:
    """The last restart in `steps`: (start of the abandoned attempt, the restarting step)."""
    found = None
    for i, s in enumerate(steps):
        if i == 0 or is_connector(s.latex):
            continue
        if states_question(s.latex) or (restates_problem(s.latex, problem)
                                        and any(_has_math(t.latex) for t in steps[:i])):
            found = (0, i)
            continue
        for j in range(i - 2):
            if not is_connector(steps[j].latex) and copied(steps[j].latex, s.latex):
                found = (j, i)
                break
    return found


def current_attempt(steps: list[Step], problem: str | None) -> tuple[list[Step], list[str]]:
    """The steps of the attempt the student is on now, and the ids left out."""
    dropped: list[str] = []
    while (r := restart_at(steps, problem)) is not None:
        start, at = r
        dropped += [s.id for s in steps[start:at]]
        steps = steps[:start] + steps[at:]
    return steps, dropped


def forget_old(steps: list[Step], old_keys: list[str]) -> list[Step]:
    """After the student moved on to a new problem on the same sheet: the old problem's
    steps, still on the paper above, are left out (the leading run of steps that read
    like the old ones; a misread or two inside the run is tolerated)."""
    if not old_keys:
        return steps
    old = set(old_keys)
    cut, misses = 0, 0
    for i, s in enumerate(steps):
        k = _key(s.latex)
        if k in old or any(SequenceMatcher(None, k, o).ratio() >= 0.85 for o in old):
            cut, misses = i + 1, 0
        elif is_connector(s.latex):
            continue
        else:
            misses += 1
            if misses > 1:
                break
    return steps[cut:]


def moved_on(before: Page | None, after: Page) -> bool:
    """A different page, or a different problem: the last reading's steps are mostly gone
    and the new reading has steps of its own, or the problem statement changed."""
    if before is None:
        return False
    if before.problem and after.problem and other_problem(before.problem, after.problem):
        return True
    old = [s for s in before.steps if not is_connector(s.latex)]
    new = [s for s in after.steps if not is_connector(s.latex)]
    if len(old) < 2 or len(new) < 2:
        return False
    kept = sum(any(same_line(o.latex, n.latex) for n in new) for o in old)
    return kept < len(old) / 2 and kept < len(new) / 2
