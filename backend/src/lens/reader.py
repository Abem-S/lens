"""A2 Page reader: one camera frame -> Page (steps in reading order, LaTeX, boxes)."""

from __future__ import annotations

import json
import re
from typing import Protocol

from lens.contract import Page, Step
from lens.spark import PAGE_MAX_TOKENS, Reply, image_part, text_part

READ_PROMPT = """You are reading a photo of a student's handwritten STEM work.

Transcribe every line of working, top to bottom, exactly as written, including
mistakes. Do not fix, finish or comment on anything.

Reply with JSON only:
{
  "problem": "the question being solved if it is written on the page, else null",
  "steps": [
    {"latex": "one written line as LaTeX", "bbox": [x, y, w, h]}
  ]
}

Rules:
- One entry per written line, in reading order. Do not merge or split lines.
- If a line continues the previous one (starts with "="), keep the leading "=".
- bbox is the line's box on the photo: x, y = top-left corner; w, h = size;
  all as fractions of the image width and height (0 to 1).
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
        steps.append(Step(id=f"s{len(steps) + 1}", latex=latex, bbox=_bbox(raw.get("bbox"))))
    problem = data.get("problem")
    return Page(problem=str(problem) if problem else None, steps=steps)


def read_page(spark: Asks, image: bytes, mime: str = "image/jpeg", effort: str = "low") -> Page:
    """Call Spark once; retry once with a nudge if the reply is not valid JSON."""
    content = [text_part(READ_PROMPT), image_part(image, mime)]
    reply = spark.ask(content, effort=effort, max_tokens=PAGE_MAX_TOKENS, purpose="read_page", json_mode=True)
    try:
        return parse_page(reply.text)
    except ReadError:
        content.append(text_part("Your last reply was not valid JSON. Reply with the JSON object only."))
        reply = spark.ask(content, effort=effort, max_tokens=PAGE_MAX_TOKENS, purpose="read_page_retry",
                          json_mode=True)
        return parse_page(reply.text)
