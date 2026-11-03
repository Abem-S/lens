"""One page check, end to end: read -> check -> confirm -> hint -> outline.

Used by the API, the tutor and the eval harness, so all three behave the same.

Confirmation: an error is only as good as the transcription it was found in. Before
claiming one, the claimed line and the line it was checked against are cropped from
the photo and read again *blind* (the second reader never sees the first reading, so
it cannot just agree with it). A mistake is claimed only when two readings of those
lines agree. If the re-read differs, the page is re-checked with it, and a mistake
that remains needs one more blind read that agrees. A re-read that cannot be read,
or that read a different line (the crop slipped), confirms nothing. A misread line
is never reported as the student's mistake.
"""

from __future__ import annotations

import concurrent.futures as cf
import io
import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from PIL import Image

from lens.attempts import current_attempt, forget_old, other_problem, same_line, step_key
from lens.checker import check_page, check_pair
from lens.contract import Finding, Page, Step
from lens.hints import write_hint
from lens.judge import judge_claim
from lens.mathparse import analyse, is_connector, same_math
from lens.pointing import bbox_outline, mask_outline
from lens.reader import Asks, _hedge, read_page
from lens.spark import PAGE_MAX_TOKENS, image_part, text_part

log = logging.getLogger("lens.pipeline")

REREAD_PROMPT = """This is a crop of a student's handwritten work.

Read every line of working in it, top to bottom, very carefully, symbol by symbol,
exactly as written, including any mistakes. Do not fix anything. Include lines
partly cut off at the edges as far as they can be read. Words go inside \\text{...}.

Reply with JSON only: {"lines": ["line as LaTeX", ...]}"""

# The second confirmation read: another prompt and a larger image, so it does not
# simply repeat the first re-read's slip.
REREAD_PROMPT_B = """A student wrote the handwritten lines in this picture. Transcribe every line,
top to bottom, symbol by symbol, exactly as written, mistakes included; never correct
anything. Copy every digit, sign and exponent exactly as it appears, including lines
partly cut off at the edges. Words go inside \\text{...}.

Reply with JSON only: {"lines": ["line as LaTeX", ...]}"""

# A re-read this unlike the first reading read another line (the crop slipped onto a
# neighbour), not the same line differently, so it cannot confirm anything.
SAME_LINE_RATIO = 0.6

# Locate a mask for a step: (image, normalised bbox) -> bool mask or None.
MaskFor = Callable[[bytes, tuple[float, float, float, float]], object]


def _mime(image: bytes) -> str:
    return "image/png" if image[:4] == b"\x89PNG" else "image/jpeg"


def _crop(image: bytes, boxes: list[tuple[float, float, float, float]], pad: float = 0.03,
          scale: float = 1.0, pad_y: float | None = None) -> bytes | None:
    boxes = [b for b in boxes if b[2] > 0 and b[3] > 0]
    if not boxes:
        return None
    pad_y = pad if pad_y is None else pad_y
    img = Image.open(io.BytesIO(image)).convert("RGB")
    w, h = img.size
    x0 = max(0.0, min(b[0] for b in boxes) - pad)
    y0 = max(0.0, min(b[1] for b in boxes) - pad_y)
    x1 = min(1.0, max(b[0] + b[2] for b in boxes) + pad)
    y1 = min(1.0, max(b[1] + b[3] for b in boxes) + pad_y)
    out = img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
    if scale != 1.0:
        out = out.resize((max(1, int(out.width * scale)), max(1, int(out.height * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    out.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def _norm(s: str) -> str:
    return re.sub(r"\s+|\\[,;! ]", "", s)


def reread(spark: Asks, image: bytes, steps: list[Step], variant: int = 0,
           raw: list[str] | None = None) -> list[str | None] | None:
    """Read the steps' lines again from a crop around them, blind, and return for each
    step the re-read line that is the same written line (None: not found in the crop).
    The crop is the page's full width and reaches a line beyond the steps' boxes, and
    every line in it is read, so a box that is off by a line still finds its line. variant 1 is the second
    confirmation: another prompt and a larger image. None: the reply was unusable.
    raw, when given, receives every line read."""
    # Full width: a step box that missed the start or end of its line must not cut
    # digits off the re-read.
    tallest = max((s.bbox[3] for s in steps), default=0.0)
    crop = _crop(image, [s.bbox for s in steps], pad=1.0, pad_y=max(0.04, tallest),
                 scale=1.0 if variant == 0 else 1.5)
    if crop is None:
        return None
    prompt = REREAD_PROMPT if variant == 0 else REREAD_PROMPT_B
    reply = spark.ask([text_part(prompt), image_part(crop)], effort="medium", max_tokens=PAGE_MAX_TOKENS,
                      purpose="reread" if variant == 0 else "reread_b", json_mode=True, **_hedge(spark))
    m = re.search(r"\{.*\}", reply.text, re.DOTALL)
    try:
        lines = json.loads(m.group(0)).get("lines") if m else None
    except json.JSONDecodeError:
        return None
    if not isinstance(lines, list) or not all(isinstance(x, str) for x in lines):
        return None
    lines = [x.strip() for x in lines if x.strip()]
    if raw is not None:
        raw.extend(lines)
    return match_lines([s.latex for s in steps], lines)


def _likeness(old: str, new: str) -> float:
    if same_math(old, new):
        return 1.0
    return SequenceMatcher(None, _norm(old).lower(), _norm(new).lower(), autojunk=False).ratio()


def match_lines(old: list[str], new: list[str]) -> list[str | None]:
    """For each line of the first reading, the most alike re-read line, if it is
    alike enough to be the same written line."""
    out: list[str | None] = []
    for o in old:
        best = max(new, key=lambda n: _likeness(o, n), default=None)
        out.append(best if best is not None and _likeness(o, best) >= SAME_LINE_RATIO else None)
    return out


def _involved(page: Page, step_id: str) -> list[Step]:
    """The claimed step and the step it was checked against (a line of linking words
    such as "Therefore," is skipped, as the checker skips it)."""
    idx = next(i for i, s in enumerate(page.steps) if s.id == step_id)
    prev = next((s for s in reversed(page.steps[:idx]) if not is_connector(s.latex)), None)
    return [prev, page.steps[idx]] if prev is not None else [page.steps[idx]]


def _rests_on(page: Page, finding: Finding) -> list[Step]:
    """The lines a claim rests on, which must read the same twice: the claimed line
    alone when it is wrong by itself (a chain whose sides differ, arithmetic inside
    words) or when the model judge flagged it; otherwise also the line it was
    compared with."""
    involved = _involved(page, finding.step_id)
    step = involved[-1]
    if step.verified_by == "model" or len(involved) == 1:
        return [step]
    cur = analyse(step.latex)
    if cur.kind == "unparseable" or check_pair(None, cur).verdict == "error":
        return [step]
    return involved


def _withhold(page: Page, step_id: str) -> Page:
    return page.model_copy(update={"steps": [
        s.model_copy(update={"status": "unverifiable", "verified_by": None}) if s.id == step_id else s
        for s in page.steps]})


def confirm_finding(spark: Asks, image: bytes, page: Page, finding: Finding, judge,
                    trace: dict | None = None, early: cf.Future | None = None) -> tuple[Page, Finding | None]:
    """Claim a finding only when two of three independent readings of its lines agree.

    Reading 1 is the page reading; reading 2 a blind crop re-read. If they agree, the
    finding stands. If not, reading 3 (another prompt, a larger image) decides: agreeing
    with the page reading, the finding stands; agreeing with the re-read, the page is
    re-checked with that reading (and a finding on those same lines stands). No two
    readings agreeing, or a line not found again, claims nothing.

    early: reading 2 already started (lens.pipeline.reread_job of the same lines, sent
    when the judge proposed this step, so it ran alongside the judge's confirmation)."""
    trace = trace if trace is not None else {}
    rereads = trace.setdefault("rereads", [])
    involved = _involved(page, finding.step_id)
    rests_on = {s.id for s in _rests_on(page, finding)}
    first = [s.latex for s in involved]

    def agree(a: list[str | None] | None, b: list[str | None] | None) -> bool:
        return (a is not None and b is not None
                and all(x is not None and y is not None and same_math(x, y)
                        for s, x, y in zip(involved, a, b) if s.id in rests_on))

    readings: list[list[str | None] | None] = []
    for variant in (0, 1):
        raw: list[str] = []
        if variant == 0 and early is not None:
            try:
                lines, raw = early.result()
            except Exception:
                log.exception("early re-read failed; reading again")
                lines = reread(spark, image, involved, variant=variant, raw=raw)
        else:
            lines = reread(spark, image, involved, variant=variant, raw=raw)
        readings.append(lines)
        rereads.append({"claimed": finding.step_id, "kind": finding.kind, "variant": variant, "raw": raw,
                        "lines": [[s.id, s.latex, new] for s, new in zip(involved, lines or [None] * len(involved))]})
        if agree(first, lines):
            trace["outcome"] = "confirmed" if variant == 0 else "confirmed by the third reading"
            return page, finding
    if not agree(readings[0], readings[1]):
        trace["outcome"] = "readings disagree"
        log.info("no two readings of the claimed lines agree; claiming nothing")
        return _withhold(page, finding.step_id), None
    # Both re-reads agree against the page reading: the page was misread. Check again
    # with every line both re-reads read alike.
    log.info("both re-reads differ from the page reading; re-checking")
    new = {s.id: a for s, a, b in zip(involved, readings[0], readings[1])  # type: ignore[arg-type]
           if a is not None and b is not None and same_math(a, b)}
    fixed = page.model_copy(update={"steps": [
        s.model_copy(update={"latex": new.get(s.id, s.latex), "status": "unchecked", "verified_by": None})
        for s in page.steps
    ]})
    page, finding = check_page(fixed, judge)
    if finding is None:
        trace["outcome"] = "withdrawn after re-read"
        return page, None
    if not {s.id for s in _rests_on(page, finding)} <= set(new):
        trace["outcome"] = "re-check claims other lines"  # read only once: not confirmed
        return _withhold(page, finding.step_id), None
    trace["outcome"] = "confirmed on the re-read"
    return page, finding


def reread_job(spark: Asks, image: bytes, steps: list[Step]) -> tuple[list[str | None] | None, list[str]]:
    """Reading 2 of `steps` (lens.pipeline.reread, variant 0), as one job for a pool."""
    raw: list[str] = []
    return reread(spark, image, steps, variant=0, raw=raw), raw


@dataclass
class Prior:
    """What a live session already knows when it reads the same page again.

    page / finding: the last check's result. forget: keys of the steps of a problem the
    student has left (new problem on the same sheet): the run of them at the top of a
    reading is left out. Only the tutor passes this; the eval reads each page fresh."""
    page: Page | None = None
    finding: Finding | None = None
    forget: list[str] = field(default_factory=list)


def on_paper(page: Page, prior: Prior) -> tuple[Page, list[str]]:
    """The reading cut down to the attempt the student is on now (lens.attempts)."""
    forget = list(prior.forget)
    if prior.page is not None and other_problem(prior.page.problem, page.problem):
        forget += [step_key(s.latex) for s in prior.page.steps]  # a new question below the old work
    steps = forget_old(list(page.steps), forget)
    steps, dropped = current_attempt(steps, page.problem)
    if len(steps) == len(page.steps):
        return page, []
    gone = [s.id for s in page.steps if s not in steps]
    return page.model_copy(update={"steps": steps}), gone or dropped


def _same_prefix(prior: Page, page: Page) -> int:
    """How many steps, from the top, read the same as in the last reading."""
    n = 0
    for a, b in zip(prior.steps, page.steps):
        if not same_line(a.latex, b.latex):
            break
        n += 1
    return n


def _still_open(prior: Prior, page: Page) -> tuple[Page, Finding] | None:
    """The last check's verified mistake is still on the page, and every step down to it
    reads as before: it stands as verified (it was claimed on these same lines read
    twice, and this reading agrees once more), without judging and re-reading again."""
    if prior.page is None or prior.finding is None:
        return None
    ids = [s.id for s in prior.page.steps]
    if prior.finding.step_id not in ids:
        return None
    k = ids.index(prior.finding.step_id)
    if _same_prefix(prior.page, page) <= k:
        return None
    checked, _ = check_page(page, None)  # SymPy marks for the steps after it
    steps = [prior.page.steps[i].model_copy(update={"bbox": page.steps[i].bbox}) if i <= k else s
             for i, s in enumerate(checked.steps)]
    return checked.model_copy(update={"steps": steps}), prior.finding


def check_image(spark: Asks, image: bytes, problem: str | None = None, *, mask_for: MaskFor | None = None,
                hint: bool = True, trace: dict | None = None, prior: Prior | None = None,
                on_read: Callable[[Page], None] | None = None,
                timings: dict | None = None) -> tuple[Page, Finding | None]:
    """trace, when given, collects what the confirmation saw (eval traces).

    Live use (the tutor) adds:
    - prior: the last check. The reading is cut to the current attempt; a verified
      mistake whose steps all read the same stands without being checked again; steps
      the judge passed before and that read the same are not judged again (except the
      last one, which may have been unfinished).
    - on_read: called with the reading (nothing checked yet) as soon as it is in, so
      the screen shows the steps while they are verified.
    Either way, while the judge's proposal is being confirmed, the blind re-read of its
    lines and its hint are already running (only used if the claim is confirmed), so a
    claim costs the slowest of the three instead of their sum."""
    t = timings if timings is not None else {}
    t0 = time.monotonic()

    def mark(name: str) -> None:
        t[name] = round(time.monotonic() - t0, 3)

    page = read_page(spark, image, mime=_mime(image))
    mark("read_s")
    if problem and not page.problem:
        page.problem = problem
    known: set[str] = set()
    if prior is not None:
        page, dropped = on_paper(page, prior)
        if dropped:
            t["left_out"] = len(dropped)
    if on_read is not None:
        try:
            on_read(page)
        except Exception:
            log.exception("on_read failed")
    if prior is not None:
        reuse = _still_open(prior, page)
        if reuse is not None:
            t["reused"] = True
            page, finding = reuse
            return page, _outline(image, page, finding, mask_for, t, mark)
        if prior.page is not None:
            n = _same_prefix(prior.page, page)
            known = {page.steps[i].id for i in range(max(0, n - 1))
                     if prior.page.steps[i].status == "ok" and prior.page.steps[i].verified_by == "model"}
    pool = cf.ThreadPoolExecutor(max_workers=4)
    drafts: dict[str, dict[str, cf.Future | None]] = {}
    try:
        def on_draft(step: Step, kind: str, judged: Page) -> None:
            drafts[step.id] = {
                "reread": pool.submit(reread_job, spark, image, _involved(judged, step.id)),
                "hint": pool.submit(write_hint, spark, judged, Finding(step_id=step.id, kind=kind)) if hint else None,
            }

        def judge(p: Page, candidates: list[str]) -> tuple[str | None, str]:
            fresh = [c for c in candidates if c not in known]
            if not fresh:
                return None, ""
            mark("judge_start_s")
            return judge_claim(spark, p, fresh, effort="low", confirm_effort="medium",
                               on_draft=lambda s, k: on_draft(s, k, p))

        page, finding = check_page(page, judge)
        mark("checked_s")
        if finding is None:
            return page, None
        if trace is not None:
            trace["first_reading"] = [s.latex for s in page.steps]
        draft = drafts.get(finding.step_id, {})
        hint_job = draft.get("hint")
        if hint and hint_job is None:  # a SymPy finding: write its hint while the lines are re-read
            hint_job = pool.submit(write_hint, spark, page, finding)
        claimed = finding
        page, finding = confirm_finding(spark, image, page, finding, judge, trace=trace, early=draft.get("reread"))
        mark("confirmed_s")
        if finding is None:
            return page, None
        if hint:
            same = (finding.step_id == claimed.step_id and hint_job is not None)
            text, rejected = hint_job.result() if same else write_hint(spark, page, finding)
            if rejected:
                log.info("hint guardrail rejected %d draft(s): %s", len(rejected), rejected)
            finding = finding.model_copy(update={"hint": text})
            mark("hint_s")
        return page, _outline(image, page, finding, mask_for, t, mark)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _outline(image: bytes, page: Page, finding: Finding, mask_for: MaskFor | None, t: dict, mark) -> Finding:
    step = page.step(finding.step_id)
    outline = None
    if mask_for is not None and step is not None:
        try:
            mask = mask_for(image, step.bbox)
            outline = mask_outline(mask) if mask is not None else None
        except Exception:
            log.exception("SAM outline failed; using the box")
    finding = finding.model_copy(update={"mask": outline or (bbox_outline(step.bbox) if step else None)})
    mark("total_s")
    return finding
