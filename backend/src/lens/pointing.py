"""C2 Pointing + outlines (geometry only; masks come from SAM on Modal, C1).

- fingertip_from_mask: hand/finger mask -> the tip point.
- nearest_step: tip point -> the Step the student is pointing at.
- outline_for_step: a step bbox (and optional SAM mask) -> a clean polygon for Finding.mask.
"""

from __future__ import annotations

import numpy as np

from lens.contract import Polygon, Step


def fingertip_from_mask(mask: np.ndarray) -> tuple[float, float] | None:
    """Return the tip of a finger/pen mask as normalised (x, y).

    A hand enters the frame from an edge; the tip is the mask pixel farthest from
    where it enters. If the mask touches no edge, the topmost pixel is used.
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or not m.any():
        return None
    h, w = m.shape
    ys, xs = np.nonzero(m)
    edge = np.zeros_like(m)
    edge[0, :], edge[-1, :], edge[:, 0], edge[:, -1] = m[0, :], m[-1, :], m[:, 0], m[:, -1]
    if edge.any():
        ey, ex = np.nonzero(edge)
        cy, cx = ey.mean(), ex.mean()
        i = int(np.argmax((ys - cy) ** 2 + (xs - cx) ** 2))
    else:
        i = int(np.argmin(ys))
    return ((xs[i] + 0.5) / w, (ys[i] + 0.5) / h)


def nearest_step(point: tuple[float, float], steps: list[Step], max_distance: float = 0.15) -> Step | None:
    """The step a fingertip points at.

    Students point at or just below a line, often at its start. Vertical distance
    matters more than horizontal (lines are wide and short); a tip just below a
    line counts as pointing at it.
    """
    px, py = point
    best, best_d = None, float("inf")
    for s in steps:
        x, y, w, h = s.bbox
        if w <= 0 or h <= 0:
            continue
        dx = max(x - px, 0.0, px - (x + w))
        below = py - (y + h)
        if y <= py <= y + h:
            dy = 0.0
        elif below > 0:
            dy = max(0.0, below - 0.5 * h)  # tolerance under the line
        else:
            dy = y - py
        d = (dx * dx + (2.0 * dy) ** 2) ** 0.5
        if d < best_d:
            best, best_d = s, d
    return best if best_d <= max_distance else None


def bbox_outline(bbox: tuple[float, float, float, float], pad: float = 0.01, corner: float = 0.012,
                 segments: int = 4) -> Polygon:
    """A rounded rectangle around a bbox, padded and clamped to the frame."""
    x, y, w, h = bbox
    x0, y0 = max(0.0, x - pad), max(0.0, y - pad)
    x1, y1 = min(1.0, x + w + pad), min(1.0, y + h + pad)
    r = min(corner, (x1 - x0) / 2, (y1 - y0) / 2)
    pts: Polygon = []
    centres = [(x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)]
    for cx, cy, start in centres:
        for k in range(segments + 1):
            a = np.radians(start + 90 * k / segments)
            pts.append((round(float(cx + r * np.cos(a)), 4), round(float(cy + r * np.sin(a)), 4)))
    return pts


def _convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(set(map(tuple, points.tolist())))
    if len(pts) <= 2:
        return np.array(pts, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def mask_outline(mask: np.ndarray) -> Polygon | None:
    """Convex outline of a SAM mask, normalised. Handwriting masks are ragged; the
    hull reads as a calm outline around the written line."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return None
    h, w = m.shape
    ys, xs = np.nonzero(m)
    # Pixel corners, so a single-row mask still has area.
    corners = np.concatenate([np.stack([xs + dx, ys + dy], 1) for dx in (0, 1) for dy in (0, 1)])
    hull = _convex_hull(corners)
    return [(round(float(px / w), 4), round(float(py / h), 4)) for px, py in hull]


def outline_for_step(step: Step, mask: np.ndarray | None = None) -> Polygon:
    """Prefer the SAM mask; fall back to the reader's bbox."""
    if mask is not None:
        poly = mask_outline(mask)
        if poly and len(poly) >= 3:
            return poly
    return bbox_outline(step.bbox)
