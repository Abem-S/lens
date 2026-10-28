"""The shared contract from AGENTS.md. Every lane codes against these shapes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

StepStatus = Literal["unchecked", "ok", "error", "unverifiable"]
VerifiedBy = Literal["sympy", "model"] | None

# A polygon is a list of [x, y] points, normalised 0–1 on the frame.
Polygon = list[tuple[float, float]]


def _clamp01(v: float) -> float:
    return min(1.0, max(0.0, float(v)))


class Step(BaseModel):
    id: str
    latex: str
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x, y, w, h normalised
    status: StepStatus = "unchecked"
    verified_by: VerifiedBy = None

    @field_validator("bbox")
    @classmethod
    def _normalise_bbox(cls, v: tuple[float, float, float, float]):
        x, y, w, h = (_clamp01(c) for c in v)
        return (x, y, min(w, 1.0 - x), min(h, 1.0 - y))


class Page(BaseModel):
    problem: str | None = None
    steps: list[Step] = Field(default_factory=list)

    def step(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)


class Finding(BaseModel):
    step_id: str
    kind: str
    mask: Polygon | None = None
    hint: str = ""
