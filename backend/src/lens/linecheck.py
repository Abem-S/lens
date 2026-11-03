"""The fast check for one just-finished line.

Live writing: tracking says a line was just finished (`line_done`, with the box the
pen covered). Re-reading the whole page for that takes 6-11 s on the shared endpoint
(and a minute when it queues), so the tutor spoke long after the student moved on.
Instead:

1. Find the handwritten lines in the frame (lens.lines, ~0.3 s) and take the ones
   under the pen's box: that is the new line. The steps read before are matched to
   the other lines, so the new line's place in the working is known (after the last
   step, or written over an earlier one).
2. Read only the new line, twice, in parallel and independently (two crops at two
   scales, two prompts; everything around the line is blanked so a neighbour cannot
   be read by mistake).
3. Check the new step after the earlier ones with SymPy (words: the model judge).
   Rule 4, a claimed mistake is a verified mistake: a mistake is claimed only when
   BOTH readings show it on the same line and read that line the same (as in the page
   check, lens.pipeline), or, for the judge, when both readings agree on the line. So as soon as the first reading shows the line is fine there
   is nothing left to wait for. When it shows a mistake, the hint is written while
   the second reading comes in.

Anything unusual (writing above earlier work in the same column, a page that no
longer shows the earlier steps, a new question, no line found, both reads failing)
returns None and the caller reads the whole page as before; why is put in
timings["fallback"] and logged, so slow checks can be traced to their cause.

What is on the paper now (lens.attempts): a line written over an earlier step
replaces it; writing struck through an earlier step (both readings find nothing
written there) removes it; a line that copies an earlier step starts the attempt
again from there, and the abandoned steps leave the list.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
from PIL import Image

from lens.attempts import current_attempt, states_question
from lens.checker import check_page, check_pair, restates_problem
from lens.contract import Finding, Page, Step
from lens.hints import write_hint
from lens.judge import judge_claim
from lens.lines import _union, find_lines, snap_steps
from lens.mathparse import analyse, is_connector, same_math
from lens.pipeline import MaskFor, _norm
from lens.pointing import bbox_outline, mask_outline
from lens.reader import Asks
from lens.spark import image_part, text_part

log = logging.getLogger("lens.linecheck")

Box = tuple[float, float, float, float]

LINE_PROMPT = """This is a crop of a student's handwritten work: the line they just wrote.
Read the handwritten line exactly as written, including any mistakes. Do not fix,
finish or comment on anything.

Reply with JSON only: {"lines": ["the written line as LaTeX"]}. If it is really two
lines, give two entries, top to bottom. Words go inside \\text{...}. Leave out
crossed-out writing. If the line starts with "=", keep the leading "=".
If nothing is written: {"lines": []}."""

LINE_PROMPT_B = """A student has just written the handwritten line in this picture.
Transcribe it very carefully, symbol by symbol, exactly as written, mistakes included;
never correct anything.

Reply with JSON only: {"lines": ["line as LaTeX"]} (one entry per written line, top to
bottom). Words go inside \\text{...}. Nothing written: {"lines": []}."""

LINE_EFFORT = "low"  # measured: "minimal" is no faster on a crop (the reasoning is about as long)
LINE_MAX_TOKENS = 3000  # reasoning tokens count against this; a crop needs far fewer than a page
LINE_HEDGE_S = 6.0  # a crop normally reads in 2-5 s; one stuck in the shared queue gets a fresh copy
JUDGE_FIND_EFFORT = "minimal"  # it only proposes; the medium-effort confirmation decides


# ---------------------------------------------------------------- where the new line is


def _overlap(a: float, al: float, b: float, bl: float) -> float:
    return max(0.0, min(a + al, b + bl) - max(a, b))


@dataclass
class Located:
    region: Box  # the new line(s), on this frame
    old: list[Step]  # the earlier steps, boxes re-found on this frame
    replaced: list[str]  # earlier steps the new writing sits on (rewritten in place)
    lines_elsewhere: int = 0  # handwritten lines found on the frame besides the new one
    under: list[str] = field(default_factory=list)  # earlier steps that align with the pen's line itself


def locate_new_line(image: bytes, page: Page, box: Box) -> Located | None:
    lines = find_lines(image)
    x, y, w, h = box
    height = float(np.median([ln.box[3] for ln in lines])) if lines else 0.03
    hits = [ln for ln in lines
            if _overlap(ln.box[1], ln.box[3], y - 0.01, h + 0.02) >= 0.5 * ln.box[3]
            and _overlap(ln.box[0], ln.box[2], x - 0.05, w + 0.1) > 0]
    region = _union([ln.box for ln in hits]) if hits else box
    # Up and down, keep to the pen's box (plus half a line): an ink "line" that ran
    # into its neighbours (dense writing on ruled paper) must not drag them in. Across,
    # take the ink's full width: the pen's path can miss the first or last letter.
    lo_y, hi_y = y - 0.5 * height, y + h + 0.5 * height
    x0, y0 = max(min(region[0], x), 0.0), max(region[1], lo_y, 0.0)
    x1, y1 = min(max(region[0] + region[2], x + w), 1.0), min(region[1] + region[3], hi_y, 1.0)
    if x1 <= x0 or y1 <= y0:
        return None
    region = (x0, y0, x1 - x0, y1 - y0)
    rest = [ln for ln in lines if ln not in hits]
    old = snap_steps(page.steps, rest) if rest else list(page.steps)
    replaced = [s.id for s in old
                if s.bbox[3] > 0
                and _overlap(s.bbox[1], s.bbox[3], region[1], region[3]) >= 0.5 * min(s.bbox[3], region[3])
                and _overlap(s.bbox[0], s.bbox[2], region[0], region[2]) > 0.3 * min(s.bbox[2], region[2])]
    # Aligned with every line, the pen's own line may be an earlier step (struck through
    # rather than written anew); the alignment corrects for the page having moved.
    on_all = snap_steps(page.steps, lines) if hits else []
    under = [s.id for s in on_all
             if s.bbox[3] > 0 and _overlap(s.bbox[1], s.bbox[3], region[1], region[3]) >= 0.5 * min(s.bbox[3], region[3])
             and _overlap(s.bbox[0], s.bbox[2], region[0], region[2]) > 0.3 * min(s.bbox[2], region[2])]
    return Located(region, old, replaced, len(rest), under)


def line_crop(image: bytes, region: Box, margin: float, scale: float = 1.0) -> bytes:
    """The new line with `margin` (in line heights) of blank paper around it: the
    neighbouring lines are painted over with the paper's colour."""
    img = Image.open(io.BytesIO(image)).convert("RGB")
    W, H = img.size
    x, y, w, h = region
    keep_y, keep_x = 0.15 * h, 0.01
    pad_y, pad_x = margin * h, 0.03
    cx0, cy0 = int(max(0.0, x - pad_x) * W), int(max(0.0, y - pad_y) * H)
    cx1, cy1 = int(min(1.0, x + w + pad_x) * W), int(min(1.0, y + h + pad_y) * H)
    crop = np.asarray(img.crop((cx0, cy0, max(cx0 + 1, cx1), max(cy0 + 1, cy1)))).copy()
    paper = np.median(crop.reshape(-1, 3), axis=0).astype(np.uint8)
    kx0, ky0 = int((x - keep_x) * W) - cx0, int((y - keep_y) * H) - cy0
    kx1, ky1 = int((x + w + keep_x) * W) - cx0, int((y + h + keep_y) * H) - cy0
    keep = np.zeros(crop.shape[:2], dtype=bool)
    keep[max(0, ky0):max(0, ky1), max(0, kx0):max(0, kx1)] = True
    crop[~keep] = paper
    out = Image.fromarray(crop)
    if scale != 1.0:
        out = out.resize((max(1, int(out.width * scale)), max(1, int(out.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def read_lines(spark: Asks, crop: bytes, prompt: str, purpose: str) -> list[str] | None:
    from lens.spark import SparkClient

    hedge = {"hedge_after": LINE_HEDGE_S} if isinstance(spark, SparkClient) else {}
    reply = spark.ask([text_part(prompt), image_part(crop)], effort=LINE_EFFORT, max_tokens=LINE_MAX_TOKENS,
                      purpose=purpose, json_mode=True, **hedge)
    m = re.search(r"\{.*\}", reply.text, re.DOTALL)
    try:
        lines = json.loads(m.group(0)).get("lines") if m else None
    except json.JSONDecodeError:
        return None
    if not isinstance(lines, list) or not all(isinstance(x, str) for x in lines):
        return None
    return [x.strip() for x in lines if x.strip()]


# ---------------------------------------------------------------- one reading, checked


@dataclass
class Reading:
    page: Page
    finding: Finding | None  # a SymPy-verified mistake on a new line
    new_ids: list[str]
    unknown: list[str] = field(default_factory=list)  # new lines SymPy could not settle

    def new_latex(self) -> list[str]:
        return [s.latex for s in self.page.steps if s.id in self.new_ids]

    def position(self) -> int | None:
        if self.finding is None:
            return None
        return [s.id for s in self.page.steps].index(self.finding.step_id)


@dataclass
class _Session:
    located: Located
    problem: str | None
    previous: Finding | None
    start: int

    def reading(self, lines: list[str]) -> Reading | None:
        old, replaced, region = self.located.old, self.located.replaced, self.located.region
        kept = [s for s in old if s.id not in replaced]
        recent = [s.latex for s in kept[-2:]]
        lines = [t for t in lines if not any(_same_line(t, r) for r in recent)]  # the line above, read again
        if not lines:
            return None
        x, y, w, h = region
        n = len(lines)
        new = [Step(id=f"s{self.start + i}", latex=t, bbox=(x, y + h * i / n, w, h / n)) for i, t in enumerate(lines)]
        if replaced:
            at = next(i for i, s in enumerate(old) if s.id in replaced)
            steps = old[:at] + new + [s for s in old[at:] if s.id not in replaced]
        else:
            steps = old + new
        new_ids = [s.id for s in new]
        steps, _ = current_attempt(steps, self.problem)  # a copied earlier step starts again from there
        page = _drop_superseded(Page(problem=self.problem, steps=steps), self.previous, new_ids)
        checked, finding = check_page(page, None)  # SymPy only
        if finding is not None and finding.step_id not in new_ids:
            finding = None  # an earlier line: the last check already spoke about it
        first_error = [s.id for s in checked.steps].index(finding.step_id) if finding else len(checked.steps)
        unknown = [s.id for i, s in enumerate(checked.steps)
                   if s.id in new_ids and s.status == "unverifiable" and i < first_error
                   and not restates_problem(s.latex, checked.problem)]  # the question copied: nothing to judge
        return Reading(checked, finding, new_ids, unknown)


def _drop_superseded(page: Page, previous: Finding | None, new_ids: list[str]) -> Page:
    """The student rewrote a wrong line just below it instead of crossing it out: when
    SymPy shows the new line follows from the line before the mistake, the wrong line
    counts as replaced."""
    if previous is None or not new_ids:
        return page
    ids = [s.id for s in page.steps]
    if previous.step_id not in ids or ids.index(new_ids[0]) != ids.index(previous.step_id) + 1:
        return page
    i = ids.index(previous.step_id)
    if i == 0:
        return page
    res = check_pair(analyse(page.steps[i - 1].latex), analyse(page.steps[i + 1].latex))
    if res.verdict == "ok" and res.by == "sympy":
        return page.model_copy(update={"steps": [s for s in page.steps if s.id != previous.step_id]})
    return page


def _same_line(a: str, b: str) -> bool:
    """The same written line: identical maths; for wordy lines, near-identical words."""
    if _norm(a).lower() == _norm(b).lower():
        return True
    return ("\\text" in a and "\\text" in b and _math_only(a) == _math_only(b)
            and SequenceMatcher(None, _norm(a).lower(), _norm(b).lower()).ratio() >= 0.85)


def _math_only(s: str) -> str:
    return _norm(re.sub(r"\\text\{[^}]*\}", "", s))


def agree(a: Reading, b: Reading) -> bool:
    """Two readings of the new line(s) say the same thing: identical maths, and words
    near enough (a misspelt word is not a different line); or the same maths written
    another way (lens.mathparse.same_math, the rule the page check uses too)."""
    xs, ys = a.new_latex(), b.new_latex()
    if len(xs) != len(ys):
        return False
    return all((_math_only(x) == _math_only(y) and SequenceMatcher(None, _norm(x), _norm(y)).ratio() >= 0.85)
               or same_math(x, y) for x, y in zip(xs, ys))


def same_claim(a: Reading, b: Reading) -> bool:
    """Both readings show the mistake on the same line, and read that line the same
    (8400 in one reading and 8200 in the other is a misreading, not a mistake)."""
    pos = a.position()
    if pos is None or pos != b.position():
        return False
    return same_math(a.page.steps[pos].latex, b.page.steps[pos].latex)


def _mark(page: Page, ids: list[str], status: str) -> Page:
    return page.model_copy(update={"steps": [
        s.model_copy(update={"status": status, "verified_by": None}) if s.id in ids else s for s in page.steps]})


def _judge_new(spark: Asks, page: Page, candidates: list[str], pool: cf.Executor | None = None,
               hint: bool = False) -> tuple[Page, Finding | None, cf.Future | None]:
    """The model judge on new lines SymPy could not settle: a quick look for a wrong
    line, then a strict confirmation (lens.judge: the same rules as the page check, so
    a new line that stops mid-sentence, with nothing after it yet, is not judged, and
    a line of words needs two confirmations). The hint for a flagged line is written
    while the confirmation runs."""
    hint_job: cf.Future | None = None

    def start_hint(step: Step, kind: str) -> None:
        nonlocal hint_job
        if hint and pool is not None:
            hint_job = pool.submit(write_hint, spark, page, Finding(step_id=step.id, kind=kind))

    flagged, kind = judge_claim(spark, page, candidates, effort=JUDGE_FIND_EFFORT, confirm_effort="medium",
                                on_draft=start_hint)
    finding = Finding(step_id=flagged, kind=kind) if flagged is not None else None
    out = []
    for s in page.steps:
        if s.id in candidates:
            if finding is not None and s.id == finding.step_id:
                s = s.model_copy(update={"status": "error", "verified_by": "model"})
            elif finding is None or [x.id for x in page.steps].index(s.id) < [x.id for x in page.steps].index(
                    finding.step_id):
                s = s.model_copy(update={"status": "ok", "verified_by": "model"})
        out.append(s)
    return page.model_copy(update={"steps": out}), finding, hint_job if finding is not None else None


# ---------------------------------------------------------------- the check

PAGE_GONE_MIN_STEPS = 3  # with this many steps known, a frame showing a third as many lines is another page


def _struck(page: Page, located: Located, previous: Finding | None, mark) -> tuple[Page, Finding | None]:
    """The page without the struck-through steps; nothing new is claimed (the steps
    keep what the last check said; the next check sees the new order)."""
    gone = set(located.replaced or located.under)
    steps = [s for s in page.steps if s.id not in gone]
    mark("total_s")
    still = previous if previous is not None and previous.step_id not in gone else None
    return page.model_copy(update={"steps": steps}), still



def check_line(spark: Asks, image: bytes, page: Page, box: Box, *, previous: Finding | None = None,
               problem: str | None = None, mask_for: MaskFor | None = None, hint: bool = True,
               timings: dict | None = None) -> tuple[Page, Finding | None] | None:
    """Check just the line written inside `box` against the steps of `page` (the last
    check's result). Returns the updated page and the finding to show (a new verified
    one, else `previous` while its line is still there), or None: read the whole page."""
    t = timings if timings is not None else {}
    t0 = time.monotonic()

    def mark(name: str) -> None:
        t[name] = round(time.monotonic() - t0, 3)

    def fallback(why: str) -> None:
        t["fallback"] = why
        log.info("line check: %s; reading the page", why)

    located = locate_new_line(image, page, box)
    mark("locate_s")
    if located is None:
        return fallback("no line under the pen")
    earlier = [s for s in page.steps if not is_connector(s.latex)]
    if len(earlier) >= PAGE_GONE_MIN_STEPS and located.lines_elsewhere * 3 <= len(earlier):
        # A new page (or the camera moved off the work): the earlier steps are not there.
        return fallback(f"page changed ({located.lines_elsewhere} other written lines for {len(earlier)} steps)")
    above: list[str] = []
    if not located.replaced:
        rx, ry, rw, rh = located.region
        above = [s.id for s in located.old
                 if s.bbox[3] > 0 and s.bbox[1] + s.bbox[3] / 2 > ry + rh
                 and _overlap(s.bbox[0], s.bbox[2], rx, rw) > 0.3 * min(s.bbox[2], rw)]  # same column
        if above and not located.under:
            return fallback(f"new writing above earlier steps {above}")
        # On an earlier step's own line: struck through (read below), else read the page.
    session = _Session(located, page.problem or problem, previous,
                       1 + max((int(s.id[1:]) for s in page.steps if s.id[1:].isdigit()), default=0))
    pool = cf.ThreadPoolExecutor(max_workers=4)
    try:
        futures = {
            pool.submit(read_lines, spark, line_crop(image, located.region, 0.6), LINE_PROMPT, "read_line"): "a",
            pool.submit(read_lines, spark, line_crop(image, located.region, 1.2, scale=1.5), LINE_PROMPT_B,
                        "read_line_b"): "b",
        }
        first: Reading | None = None
        got = 0
        empty = 0  # readings that found nothing written in the region
        result: Reading | None = None
        claim = False
        hint_job = judge_job = None
        for f in cf.as_completed(futures):
            got += 1
            which = futures[f]
            mark(f"read_{which}_s")
            try:
                lines = f.result()
            except Exception:
                log.exception("line read %s failed", which)
                lines = None
            if lines and any(states_question(x) for x in lines):
                return fallback("a new question")
            empty += lines == []
            if empty == 2 and (located.replaced or located.under):
                # Struck through: both readings leave the crossed-out writing out, and
                # nothing else is written there. The step leaves the list.
                t["struck"] = located.replaced or located.under
                return _struck(page, located, previous, mark)
            if lines and above:
                return fallback(f"writing on earlier step {located.under}")
            r = session.reading(lines) if lines else None
            if got == 1:
                first = r
                if r is None:
                    continue  # wait for the other reading
                if r.finding is None and not r.unknown:
                    result = r  # fine as read: a mistake now needs both readings, so nothing to wait for
                    break
                if r.finding is not None and hint:
                    hint_job = pool.submit(write_hint, spark, r.page, r.finding)  # overlaps the second read
                elif r.finding is None:
                    judge_job = pool.submit(_judge_new, spark, r.page, r.unknown, pool, hint)
                continue
            # Second reading.
            if first is None:  # the first failed: this one stands alone and may not claim anything
                if r is None:
                    return fallback("neither line reading worked")
                result = r
                if r.finding is not None:
                    result = Reading(_mark(r.page, [r.finding.step_id], "unverifiable"), None, r.new_ids)
                elif r.unknown:
                    result = Reading(r.page, None, r.new_ids)  # unknown lines stay "unverifiable"
                break
            result = first
            if first.finding is not None:
                claim = r is not None and same_claim(first, r)
                if not claim:
                    log.info("only one reading shows a mistake on %s; claiming nothing", first.finding.step_id)
                    result = Reading(_mark(first.page, [first.finding.step_id], "unverifiable"), None,
                                     first.new_ids)
            elif r is not None and agree(first, r):
                judged_page, judged, hint_job = judge_job.result()
                mark("judge_s")
                result = Reading(judged_page, judged, first.new_ids)
                claim = judged is not None
            break
        if result is None:
            return fallback("no new step in the readings")
        checked = result.page
        # Earlier steps keep what the last check said about them.
        prior = {s.id: s for s in page.steps}
        checked = checked.model_copy(update={"steps": [
            s.model_copy(update={"status": prior[s.id].status, "verified_by": prior[s.id].verified_by})
            if s.id in prior and s.id not in located.replaced else s for s in checked.steps]})
        finding = result.finding if claim else None
        kept = checked.step(previous.step_id) if previous is not None else None
        if kept is not None:
            mark("total_s")
            # The earlier (first) mistake is still on the page. Its old outline was drawn on an
            # earlier frame; outline the step where it was found on this one.
            return checked, previous.model_copy(update={"mask": bbox_outline(kept.bbox) if kept.bbox[2] > 0
                                                        else previous.mask})
        if finding is None:
            mark("total_s")
            return checked, None
        step = checked.step(finding.step_id)
        if hint:
            if hint_job is not None:
                text, rejected = hint_job.result()
            else:
                text, rejected = write_hint(spark, checked, finding)
            if rejected:
                log.info("hint guardrail rejected %d draft(s): %s", len(rejected), rejected)
            finding = finding.model_copy(update={"hint": text})
            mark("hint_s")
        outline = None
        if mask_for is not None and step is not None:
            try:
                mask = mask_for(image, step.bbox)
                outline = mask_outline(mask) if mask is not None else None
            except Exception:
                log.exception("SAM outline failed; using the box")
        finding = finding.model_copy(update={"mask": outline or (bbox_outline(step.bbox) if step else None)})
        mark("total_s")
        return checked, finding
    finally:
        pool.shutdown(wait=False)
