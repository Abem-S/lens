"""One page check, end to end: read -> check -> confirm -> hint -> outline.

Used by the API, the tutor and the eval harness, so all three behave the same.

Confirmation: a SymPy-verified error is only as good as the transcription it was
computed from. Before claiming one, the two lines involved are cropped from the
photo and read again, more carefully; if the second reading differs, the page is
re-checked with it. A misread line is never reported as the student's mistake.
"""

from __future__ import annotations

import io
import json
import logging
import re
from collections.abc import Callable

from PIL import Image

from lens.checker import check_page
from lens.contract import Finding, Page, Step
from lens.hints import write_hint
from lens.judge import spark_judge
from lens.pointing import bbox_outline, mask_outline
from lens.reader import Asks, _hedge, read_page
from lens.spark import PAGE_MAX_TOKENS, image_part, text_part

log = logging.getLogger("lens.pipeline")

REREAD_PROMPT = """This is a crop of a student's handwritten work. It shows {n} line(s) of working
(possibly with parts of neighbouring lines at the edges).

The lines were first read as:
{lines}

Read the {n} line(s) again, very carefully, exactly as written, including any mistakes.
Do not fix anything. Reply with JSON only: {{"lines": ["line as LaTeX", ...]}} with {n} entries, in order."""

# Locate a mask for a step: (image, normalised bbox) -> bool mask or None.
MaskFor = Callable[[bytes, tuple[float, float, float, float]], object]


def _mime(image: bytes) -> str:
    return "image/png" if image[:4] == b"\x89PNG" else "image/jpeg"


def _crop(image: bytes, boxes: list[tuple[float, float, float, float]], pad: float = 0.03) -> bytes | None:
    boxes = [b for b in boxes if b[2] > 0 and b[3] > 0]
    if not boxes:
        return None
    img = Image.open(io.BytesIO(image)).convert("RGB")
    w, h = img.size
    x0 = max(0.0, min(b[0] for b in boxes) - pad)
    y0 = max(0.0, min(b[1] for b in boxes) - pad)
    x1 = min(1.0, max(b[0] + b[2] for b in boxes) + pad)
    y1 = min(1.0, max(b[1] + b[3] for b in boxes) + pad)
    buf = io.BytesIO()
    img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h))).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def _norm(s: str) -> str:
    return re.sub(r"\s+|\\[,;! ]", "", s)


def reread(spark: Asks, image: bytes, steps: list[Step]) -> list[str] | None:
    crop = _crop(image, [s.bbox for s in steps])
    if crop is None:
        return None
    prompt = REREAD_PROMPT.format(n=len(steps), lines="\n".join(f"- {s.latex}" for s in steps))
    reply = spark.ask([text_part(prompt), image_part(crop)], effort="medium", max_tokens=PAGE_MAX_TOKENS,
                      purpose="reread", json_mode=True, **_hedge(spark))
    m = re.search(r"\{.*\}", reply.text, re.DOTALL)
    try:
        lines = json.loads(m.group(0)).get("lines") if m else None
    except json.JSONDecodeError:
        return None
    if not isinstance(lines, list) or len(lines) != len(steps) or not all(isinstance(x, str) for x in lines):
        return None
    return [x.strip() for x in lines]


def confirm_sympy_finding(spark: Asks, image: bytes, page: Page, finding: Finding,
                          judge) -> tuple[Page, Finding | None]:
    idx = next(i for i, s in enumerate(page.steps) if s.id == finding.step_id)
    involved = page.steps[max(0, idx - 1): idx + 1]
    lines = reread(spark, image, involved)
    if lines is None:
        return page, finding  # could not re-read; the SymPy verdict on the first reading stands
    changed = {s.id: new for s, new in zip(involved, lines) if _norm(new) != _norm(s.latex)}
    if not changed:
        return page, finding
    log.info("re-read changed %d line(s); re-checking", len(changed))
    fixed = page.model_copy(update={"steps": [
        s.model_copy(update={"latex": changed.get(s.id, s.latex), "status": "unchecked", "verified_by": None})
        for s in page.steps
    ]})
    return check_page(fixed, judge)


def check_image(spark: Asks, image: bytes, problem: str | None = None, *, mask_for: MaskFor | None = None,
                hint: bool = True) -> tuple[Page, Finding | None]:
    page = read_page(spark, image, mime=_mime(image))
    if problem and not page.problem:
        page.problem = problem
    judge = spark_judge(spark)
    page, finding = check_page(page, judge)
    if finding is not None:
        step = page.step(finding.step_id)
        if step and step.verified_by == "sympy":
            page, finding = confirm_sympy_finding(spark, image, page, finding, judge)
    if finding is None:
        return page, None
    step = page.step(finding.step_id)
    if hint:
        text, rejected = write_hint(spark, page, finding)
        if rejected:
            log.info("hint guardrail rejected %d draft(s): %s", len(rejected), rejected)
        finding = finding.model_copy(update={"hint": text})
    outline = None
    if mask_for is not None and step is not None:
        try:
            mask = mask_for(image, step.bbox)
            outline = mask_outline(mask) if mask is not None else None
        except Exception:
            log.exception("SAM outline failed; using the box")
    finding = finding.model_copy(update={"mask": outline or (bbox_outline(step.bbox) if step else None)})
    return page, finding
