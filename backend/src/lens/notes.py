"""The tutor's private notes on the problem the student is solving (one session only).

The conversation itself is kept short (the last turns), so what the student said at the start
("it's a ball thrown at 20 m/s from a 45 m cliff, find the time") would fall out of it. The notes
keep it: the problem with every given, what is asked, a short teaching plan, the student's
approach and where they are. A small model call rewrites them after a student turn that says
something and after a check that read new steps; the tutor puts them in every reply's context.
They are never shown and never stored past the session.
"""

from __future__ import annotations

import json
import logging
import re

log = logging.getLogger("lens.notes")

FIELDS = ("problem", "asked", "plan", "approach", "progress")

NOTES_SYSTEM = """You keep a tutor's private notes about the problem a student is solving on paper, during
one tutoring session. You get the notes so far, the latest conversation and what has been read from the
page. Return the updated notes as one JSON object with these keys (each a short string, or null if unknown):

"problem": the problem exactly as the student or the page gives it, with EVERY given value, unit and
  condition (e.g. "A ball is thrown up at 20 m/s from a 45 m cliff; g = 9.8 m/s^2").
"asked": what they must find, prove or show.
"plan": a short teaching plan: 2 to 4 stages of a sensible method, in words ("write the motion equation
  for height", "set height to zero", "solve the quadratic for t, keep the positive root"). No results,
  no numbers the student has not written.
"approach": how the student is actually going about it, from their steps and words.
"progress": where they are now (which stage, what they just did, what is still open).
"on_track": true if their approach can reach what is asked, false only if it clearly cannot (wrong
  quantity being found, an equation that cannot lead to the answer, going in circles), null if unsure.
"why_off": when on_track is false, one short sentence on why, for the tutor only; else null.

Rules:
- Keep what you already knew unless the student or the page clearly changes it. Never drop a given.
- Never invent givens. If the problem is not known yet, leave it null.
- The student's words come from speech recognition and may be garbled (accents; maths words misheard,
  e.g. "ex" for x, "too" for 2, "square" for squared). Read them for what they most likely meant.
- If the student says they are starting a different problem, replace the notes for the new one.
Reply with the JSON object only."""


def empty() -> dict:
    return {k: None for k in (*FIELDS, "on_track", "why_off")}


def known(notes: dict) -> bool:
    return any(notes.get(k) for k in FIELDS)


def render(notes: dict) -> str:
    """The notes as lines for the tutor's context."""
    names = {"problem": "Problem (with its givens)", "asked": "Asked", "plan": "Your teaching plan",
             "approach": "Their approach", "progress": "Where they are"}
    lines = [f"- {names[k]}: {notes[k]}" for k in FIELDS if notes.get(k)]
    return "\n".join(lines)


def _parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def update(brain, notes: dict, conversation: list[dict], page: str, event: str) -> dict:
    """One rewrite of the notes (worker thread). On any failure the old notes stay."""
    talk = []
    for turn in conversation:
        content = turn["content"]
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        who = "Student" if turn["role"] == "user" else "Tutor"
        talk.append(f"{who}: {content}")
    user = (f"Notes so far:\n{json.dumps(notes, ensure_ascii=False)}\n\n"
            f"Latest conversation:\n" + ("\n".join(talk) or "(none)") +
            f"\n\nRead from the page:\n{page}\n\nWhat just happened: {event}")
    messages = [{"role": "system", "content": NOTES_SYSTEM}, {"role": "user", "content": user}]
    try:
        text = "".join(brain.stream(messages, effort="minimal", max_tokens=2500, purpose="notes"))
    except Exception:
        log.exception("notes update failed")
        return notes
    data = _parse(text)
    if data is None:
        log.warning("notes update returned no JSON (%d chars)", len(text or ""))
        return notes
    out = dict(notes)
    for k in (*FIELDS, "why_off"):
        if k in data:
            v = data[k]
            out[k] = " ".join(str(v).split())[:600] if v not in (None, "", "null") else (notes.get(k) if k in ("problem", "asked") else None)
    out["on_track"] = data.get("on_track") if data.get("on_track") in (True, False) else None
    return out
