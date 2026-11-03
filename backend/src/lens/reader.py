"""A2 Page reader: one camera frame -> Page (steps in reading order, LaTeX, boxes)."""

from __future__ import annotations

import json
import logging
import re
from typing import Protocol

from lens.contract import Page, Step
from lens.lines import snap_steps
from lens.spark import PAGE_MAX_TOKENS, Reply, image_part, text_part

log = logging.getLogger("lens.reader")

READ_PROMPT = """You are reading a photo of a student's handwritten STEM work.

Transcribe every line of working, top to bottom, exactly as written, including
mistakes. Do not fix, finish or comment on anything.

Reply with JSON only:
{
  "problem": "the question being solved if it is written on the page, else null",
  "steps": [
    {"latex": "one written line as LaTeX", "box": [left, top, right, bottom]}
  ]
}

Rules:
- One entry per written line, in reading order. Do not merge or split lines.
- If a line continues the previous one (starts with "="), keep the leading "=".
- box is the line's tight box on the photo: left, top, right, bottom edges as
  fractions of the image width and height (0 to 1). A line's box is short:
  about one line of handwriting tall.
- Words go inside \\text{...}. Leave out crossed-out writing.
- If nothing is written, return {"problem": null, "steps": []}."""


class Asks(Protocol):
    def ask(self, prompt, *, system=None, **kw) -> Reply: ...


class ReadError(ValueError):
    pass


def _extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ReadError("no JSON object in reply")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ReadError(f"bad JSON: {exc}") from exc


def _bbox(raw) -> tuple[float, float, float, float]:
    try:
        x, y, w, h = (float(v) for v in raw)
    except (TypeError, ValueError):
        return (0.0, 0.0, 0.0, 0.0)
    if max(x, y, w, h) > 1.5:  # model answered in pixels-per-thousand or percent
        scale = 1000.0 if max(x, y, w, h) > 100 else 100.0
        x, y, w, h = x / scale, y / scale, w / scale, h / scale
    return (x, y, max(w, 0.0), max(h, 0.0))


def parse_page(text: str) -> Page:
    data = _extract_json(text)
    steps = []
    for i, raw in enumerate(data.get("steps") or []):
        latex = str(raw.get("latex", "")).strip() if isinstance(raw, dict) else ""
        if not latex:
            continue
        if raw.get("box") is not None:
            l, t, r, b = _bbox(raw.get("box"))  # corners, same scaling rules
            box = (min(l, r), min(t, b), abs(r - l), abs(b - t))
        else:
            box = _bbox(raw.get("bbox"))  # older x, y, w, h form
        steps.append(Step(id=f"s{len(steps) + 1}", latex=latex, bbox=box))
    problem = data.get("problem")
    return Page(problem=str(problem) if problem else None, steps=steps)


MAX_SIDE = 1600
HEDGE_AFTER_S = 15.0


def _hedge(spark) -> dict:
    """Only the real client supports hedging; test fakes take plain keywords."""
    from lens.spark import SparkClient

    return {"hedge_after": HEDGE_AFTER_S} if isinstance(spark, SparkClient) else {}


def shrink(image: bytes, max_side: int = MAX_SIDE) -> tuple[bytes, str]:
    """Big phone photos cost tokens and time without reading better; cap the long side."""
    import io

    from PIL import Image, ImageOps

    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(image)))
    except Exception:
        return image, "image/jpeg"
    if max(img.size) <= max_side and img.format == "JPEG":
        return image, "image/jpeg"
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=88)
    return buf.getvalue(), "image/jpeg"


def read_page(spark: Asks, image: bytes, mime: str = "image/jpeg", effort: str = "low", snap: bool = True) -> Page:
    """Call Spark once; retry once with a nudge if the reply is not valid JSON.

    snap: replace Spark's rough step boxes with the handwritten lines found in the
    ink (lens.lines). The ink is analysed while Spark is reading, so it adds no time."""
    from concurrent.futures import ThreadPoolExecutor

    image, mime = shrink(image)
    pool = ThreadPoolExecutor(max_workers=1) if snap else None
    lines = pool.submit(_ink_lines, image) if pool else None
    try:
        content = [text_part(READ_PROMPT), image_part(image, mime)]
        reply = spark.ask(content, effort=effort, max_tokens=PAGE_MAX_TOKENS, purpose="read_page", json_mode=True,
                          **_hedge(spark))
        try:
            page = parse_page(reply.text)
        except ReadError:
            content.append(text_part("Your last reply was not valid JSON. Reply with the JSON object only."))
            reply = spark.ask(content, effort=effort, max_tokens=PAGE_MAX_TOKENS, purpose="read_page_retry",
                              json_mode=True, **_hedge(spark))
            page = parse_page(reply.text)
        if lines is not None:
            found = lines.result()
            if found:
                page.steps = snap_steps(page.steps, found)
        return page
    finally:
        if pool:
            pool.shutdown(wait=False)


def _ink_lines(image: bytes) -> list:
    from lens.lines import find_lines

    try:
        return find_lines(image)
    except Exception:
        log.exception("ink line detection failed; keeping Spark's boxes")
        return []
