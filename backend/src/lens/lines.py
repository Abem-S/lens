"""Handwritten lines from the ink, and step boxes snapped onto them.

Muse Spark reads the steps well, but the boxes it guesses for them are often off,
sometimes by several lines. The ink itself says where the writing is, so the boxes
on screen come from the image:

1. `find_lines(image)`: ink -> text lines. The page is flattened against its own
   paper (shadows and uneven light divide out), ink is the pixels clearly darker
   than the paper around them, and long thin straight runs (notebook rules, grid
   lines, margins) are removed. The page is deskewed, nearby ink is smeared
   together along each row, the blobs are labelled, blobs holding two touching lines
   are cut at the quietest row, and blobs side by side at the same height join
   into one line (unless a wide gap says they are separate columns).
2. `snap_steps(steps, lines)`: each step, in reading order, takes one ink line, a
   few consecutive ones (a fraction, a long step that wraps), or a share of one
   line (two short steps written side by side). Ink lines may be left over: the
   question, crossed-out work, a diagram label. A small dynamic program picks the
   assignment that best fits Spark's own rough boxes, after correcting them for a
   global shift and scale.

Only numpy and Pillow; about 0.1 s for a phone photo.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter, ImageOps

Box = tuple[float, float, float, float]  # normalised x, y, w, h

WORK_SIDE = 900  # working resolution (long side, px)
MAX_SIDE = 2000  # finest working resolution, for small writing on a big photo


@dataclass
class InkLine:
    box: Box  # normalised on the original image
    ink: float  # inked area, as a fraction of the image
    segments: list[Box] = field(default_factory=list)  # pieces separated by wide gaps, left to right

    @property
    def cy(self) -> float:
        return self.box[1] + self.box[3] / 2

    @property
    def cx(self) -> float:
        return self.box[0] + self.box[2] / 2


# ---------------------------------------------------------------- image -> ink


def _load(image) -> Image.Image:
    img = image if isinstance(image, Image.Image) else Image.open(io.BytesIO(image))
    with contextlib.suppress(Exception):  # a broken EXIF block: keep the pixels as they are
        img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    return img.convert("RGB")


def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=64, range=(0.0, 1.0))
    w = hist.astype(float)
    if w.sum() == 0:
        return 0.3
    mids = (edges[:-1] + edges[1:]) / 2
    cw = np.cumsum(w)
    cm = np.cumsum(w * mids)
    total, mean = cw[-1], cm[-1]
    w0, w1 = cw[:-1], total - cw[:-1]
    ok = (w0 > 0) & (w1 > 0)
    between = np.zeros_like(w0)
    between[ok] = (mean * w0[ok] / total - cm[:-1][ok]) ** 2 / (w0[ok] * w1[ok])
    return float(edges[1:][:-1][int(np.argmax(between))]) if ok.any() else 0.3


def ink_mask(img: Image.Image) -> np.ndarray:
    """Pixels clearly darker than the paper around them, on the paper only."""
    gray_img = img.convert("L")
    w, h = gray_img.size
    gray = np.asarray(gray_img, dtype=np.float32) / 255.0
    # Paper brightness around each pixel: the local maximum (ink is thin), smoothed.
    small = gray_img.resize((max(1, w // 8), max(1, h // 8)), Image.BILINEAR)
    small = small.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2))
    bg = np.asarray(small.resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    bg = np.maximum(bg, gray)
    contrast = np.clip((bg - gray) / np.maximum(bg, 0.05), 0.0, 1.0)
    # Paper: bright in the local median (ink is sparse; cloth, faces, the desk are not),
    # shrunk a little so the dark just past the page edge is not read as ink.
    med = gray_img.resize((max(1, w // 8), max(1, h // 8)), Image.BILINEAR).filter(ImageFilter.MedianFilter(5))
    med_a = np.asarray(med, dtype=np.float32) / 255.0
    bright = Image.fromarray(((med_a > 0.45 * float(np.percentile(med_a, 95))) * 255).astype(np.uint8))
    paper = np.asarray(bright.filter(ImageFilter.MinFilter(3)).resize((w, h), Image.NEAREST)) > 0
    cand = contrast[paper & (contrast > 0.06)]
    thr = float(np.clip(_otsu(cand), 0.16, 0.4)) if cand.size > 50 else 0.2
    return (contrast > thr) & paper


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Horizontal runs of True: (row, start, end) arrays, end exclusive, row-major order."""
    h, w = mask.shape
    pad = np.zeros((h, w + 2), dtype=np.int8)
    pad[:, 1:-1] = mask
    d = np.diff(pad, axis=1)
    rs, ss = np.nonzero(d == 1)
    _, es = np.nonzero(d == -1)
    return rs, ss, es


def _remove_rules(mask: np.ndarray, len_h: float | None = None, len_v: float | None = None,
                  thick: int = 4) -> np.ndarray:
    """Drop long thin straight runs: notebook rules, grid lines, margins, the page edge,
    circuit wires. Default lengths: about a quarter of the page."""
    out = mask.copy()
    h, w = mask.shape
    for m, length, axis in ((mask, len_h or 0.22 * w, 0), (mask.T, len_v or 0.22 * h, 1)):
        # A rule a pixel off level breaks into steps; OR-ing neighbouring rows joins them.
        joined = m | np.roll(m, 1, axis=0) | np.roll(m, -1, axis=0)
        rs, ss, es = _runs(joined)
        long = (es - ss) >= length
        for r, s, e in zip(rs[long], ss[long], es[long]):
            for rr in (r - 2, r - 1, r, r + 1, r + 2):
                if 0 <= rr < m.shape[0]:
                    if axis == 0:
                        out[rr, s:e] &= ~_thin_across(m, rr, s, e, thick)
                    else:
                        out[s:e, rr] &= ~_thin_across(m, rr, s, e, thick)
    return out


def _thin_across(m: np.ndarray, r: int, s: int, e: int, thick: int) -> np.ndarray:
    """Along a long run at row r, the pixels where the ink across it is thin (a rule,
    not a letter standing on it): a letter crossing a rule keeps its pixels."""
    lo, hi = max(0, r - thick), min(m.shape[0], r + thick + 1)
    thickness = m[lo:hi, s:e].sum(axis=0)
    return m[r, s:e] & (thickness <= thick)


def _deskew_angle(mask: np.ndarray) -> float:
    """Angle (degrees) that makes the text rows sharpest in the row projection."""
    if mask.sum() < 50:
        return 0.0
    h, w = mask.shape
    scale = min(1.0, 400 / max(h, w))
    img = Image.fromarray((mask * 255).astype(np.uint8))
    if scale < 1:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)

    def score(a: float) -> float:
        r = np.asarray(img.rotate(a, resample=Image.BILINEAR, fillcolor=0), dtype=np.float32)
        p = r.sum(axis=1)
        return float((p * p).sum())

    level = score(0.0)
    best = max(np.arange(-5.0, 5.01, 0.5), key=score)
    fine = float(max(np.arange(best - 0.375, best + 0.38, 0.125), key=score))
    # Turn the page only for a clear gain: two columns or a diagram can favour a
    # slight wrong tilt by a hair.
    return fine if score(fine) > 1.08 * level else 0.0


def _rotate(mask: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-6:
        return mask
    img = Image.fromarray((mask * 255).astype(np.uint8)).rotate(angle, resample=Image.BILINEAR, fillcolor=0)
    return np.asarray(img) > 100


# ---------------------------------------------------------------- blobs


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        p = self.p
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> None:
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def _smeared_runs(mask: np.ndarray, gap: int) -> tuple[list[int], list[int], list[int], list[int]]:
    """Runs per row after closing gaps shorter than `gap` (rows, starts, ends, ink per run)."""
    rs, ss, es = _runs(mask)
    rows: list[int] = []
    starts: list[int] = []
    ends: list[int] = []
    inks: list[int] = []
    for r, s, e in zip(rs.tolist(), ss.tolist(), es.tolist()):
        if rows and rows[-1] == r and s - ends[-1] < gap:
            ends[-1] = e
            inks[-1] += e - s
        else:
            rows.append(r)
            starts.append(s)
            ends.append(e)
            inks.append(e - s)
    return rows, starts, ends, inks


@dataclass
class _Blob:
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int
    ink: int

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def join(self, o: _Blob) -> _Blob:
        return _Blob(min(self.x0, o.x0), min(self.y0, o.y0), max(self.x1, o.x1), max(self.y1, o.y1), self.ink + o.ink)


def _blobs(mask: np.ndarray, gap: int) -> list[_Blob]:
    """Connected pieces of the row-smeared mask (8-connected)."""
    return list(_label(mask, gap)[0].values())


def _drop_frames(mask: np.ndarray, line_h: float) -> np.ndarray:
    """Remove single strokes that are big and hollow: a loop or box drawn around an
    answer, a long arrow, circuit wires. They would tie many lines into one blob."""
    blobs, runs = _label(mask, 1)
    frames = {root for root, b in blobs.items()
              if (b.h > 3 * line_h and b.w > 3 * line_h and b.ink < 0.06 * b.w * b.h)  # hollow
              or (b.h > 2.5 * line_h and b.w > line_h and b.ink > 0.5 * b.w * b.h)}  # solid: past the page edge
    if not frames:
        return mask
    out = mask.copy()
    for r, s, e, root in runs:
        if root in frames:
            out[r, s:e] = False
    return out


def _label(mask: np.ndarray, gap: int) -> tuple[dict[int, _Blob], list[tuple[int, int, int, int]]]:
    """Connected-component labelling on row runs (gaps shorter than `gap` closed first).
    Returns blobs by root id and the runs as (row, start, end, root)."""
    rows, starts, ends, inks = _smeared_runs(mask, gap)
    n = len(rows)
    dsu = _DSU(n)
    # Runs of each row are contiguous in the lists; link overlapping runs of consecutive rows.
    row_start: dict[int, int] = {}
    for i, r in enumerate(rows):
        row_start.setdefault(r, i)
    i = 0
    while i < n:
        r = rows[i]
        j = i
        while j < n and rows[j] == r:
            j += 1
        k = row_start.get(r + 1)
        if k is not None:
            a, b = i, k
            while a < j and b < n and rows[b] == r + 1:
                if starts[a] <= ends[b] and starts[b] <= ends[a]:
                    dsu.union(a, b)
                if ends[a] < ends[b]:
                    a += 1
                else:
                    b += 1
        i = j
    groups: dict[int, _Blob] = {}
    runs = []
    for idx in range(n):
        root = dsu.find(idx)
        b = _Blob(starts[idx], rows[idx], ends[idx], rows[idx] + 1, inks[idx])
        groups[root] = groups[root].join(b) if root in groups else b
        runs.append((rows[idx], starts[idx], ends[idx], root))
    return groups, runs


def _cut_tall(mask: np.ndarray, b: _Blob, line_h: float, depth: int = 0) -> list[_Blob]:
    """Split a blob holding several touching lines (cursive loops reaching into the line
    below) at its quietest row, while that row is clearly quieter than the lines on
    either side of it."""
    if b.h < 1.7 * line_h or depth > 12:
        return [b]
    prof = mask[b.y0:b.y1, b.x0:b.x1].sum(axis=1).astype(float)
    k = max(1, int(line_h * 0.15))
    smooth = np.convolve(prof, np.ones(2 * k + 1) / (2 * k + 1), mode="same")
    lo, hi = int(0.6 * line_h), b.h - int(0.6 * line_h)
    if hi <= lo:
        return [b]
    cut = lo + int(np.argmin(smooth[lo:hi]))
    reach = int(1.2 * line_h)
    above = float(smooth[max(0, cut - reach):cut].max(initial=0.0))
    below = float(smooth[cut + 1:cut + 1 + reach].max(initial=0.0))
    if smooth[cut] > 0.5 * min(above, below):
        return [b]  # no quiet row: one tall thing (a diagram, a big bracket), not lines

    def part(y0: int, y1: int) -> _Blob | None:
        sub = mask[y0:y1, b.x0:b.x1]
        if not sub.any():
            return None
        ys, xs = np.nonzero(sub)
        return _Blob(b.x0 + int(xs.min()), y0 + int(ys.min()), b.x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1,
                     int(sub.sum()))

    out: list[_Blob] = []
    for p in (part(b.y0, b.y0 + cut), part(b.y0 + cut, b.y1)):
        if p is not None:
            out += _cut_tall(mask, p, line_h, depth + 1)
    return out


def _pitch(mask: np.ndarray, line_h: float) -> float | None:
    """The spacing of the lines of writing, from the autocorrelation of the row
    profile, when the page has a clear one."""
    p = mask.sum(axis=1).astype(float)
    p -= p.mean()
    n = len(p)
    if n < 8 or not p.any():
        return None
    spec = np.fft.rfft(p, 2 * n)
    ac = np.fft.irfft(spec * np.conj(spec))[:n]
    if ac[0] <= 0:
        return None
    ac /= ac[0]
    lo, hi = max(4, int(0.6 * line_h)), min(n - 1, int(3 * line_h))
    best, val = None, 0.25
    for k in range(lo, hi):
        if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > val:
            best, val = k, float(ac[k])
    return float(best) if best is not None else None


def _line_height(blobs: list[_Blob], default: float) -> float:
    """Typical height of a word: a median weighted by the square root of the ink, so
    broken bits of faint letters do not drag it down and one big sketch does not
    drag it up."""
    # Words are dense; frames, wires and diagrams are mostly empty space.
    texty = sorted((b.h, b.ink) for b in blobs if b.w >= 0.8 * b.h and b.ink >= 20 and b.ink >= 0.1 * b.w * b.h)
    if not texty:
        return default
    weights = np.cumsum([math.sqrt(ink) for _, ink in texty])
    return float(texty[int(np.searchsorted(weights, weights[-1] / 2))][0])


def _group(blobs: list[_Blob], line_h: float, width: int) -> list[list[_Blob]]:
    """Words at the same height and not too far apart form one line. Small marks (dots,
    primes, indices) and tall ones (braces, integral signs) join the line they sit
    beside instead of tying two lines together."""
    small = [b for b in blobs if b.h < 0.45 * line_h]
    tall = [b for b in blobs if b.h > 1.7 * line_h]
    body = sorted((b for b in blobs if 0.45 * line_h <= b.h <= 1.7 * line_h), key=lambda b: b.x0)
    n = len(body)
    dsu = _DSU(n)
    max_gap = max(4.0 * line_h, 0.06 * width)
    for i in range(n):
        a = body[i]
        for j in range(i + 1, n):
            b = body[j]
            if b.x0 - a.x1 > max_gap:
                break
            # Same height: they overlap, and their middles are close (a word whose loop
            # hangs into the line below must not tie the two lines together).
            if min(a.y1, b.y1) - max(a.y0, b.y0) >= 0.5 * min(a.h, b.h) and abs(a.cy - b.cy) <= 0.6 * line_h:
                dsu.union(i, j)
    groups: dict[int, list[_Blob]] = {}
    for i, b in enumerate(body):
        groups.setdefault(dsu.find(i), []).append(b)
    lines = list(groups.values())
    boxes = []
    for g in lines:
        whole = g[0]
        for b in g[1:]:
            whole = whole.join(b)
        boxes.append(whole)

    def attach(b: _Blob, reach: float, slack: float, min_share: float = 0.0) -> bool:
        best, best_overlap = None, 0.0
        for k, w in enumerate(boxes):
            gap = max(w.x0 - b.x1, b.x0 - w.x1, 0)
            overlap = min(w.y1 + slack, b.y1) - max(w.y0 - slack, b.y0)
            if gap <= reach and overlap > best_overlap and overlap >= min_share * w.h:
                best, best_overlap = k, overlap
        if best is None:
            return False
        lines[best].append(b)
        boxes[best] = boxes[best].join(b)
        return True

    for b in sorted(tall, key=lambda b: -b.h):
        # A brace or an integral sign belongs to the line beside it; a big scrawl in
        # the margin (a note written sideways) is a line of its own.
        if b.h > 2.6 * line_h or not attach(b, 1.0 * line_h, 0.0, min_share=0.6):
            lines.append([b])
            boxes.append(b)
    for b in small:
        if not attach(b, 1.5 * line_h, 0.5 * line_h):
            lines.append([b])  # dropped later as too small unless it is a real line
            boxes.append(b)
    return lines


def _segments(group: list[_Blob], line_h: float) -> list[_Blob]:
    """Pieces of a line separated by gaps wider than a word space, left to right."""
    group = sorted(group, key=lambda b: b.x0)
    segs: list[_Blob] = []
    for b in group:
        if segs and b.x0 - segs[-1].x1 <= 1.2 * line_h:
            segs[-1] = segs[-1].join(b)
        else:
            segs.append(b)
    return segs


def find_lines(image) -> list[InkLine]:
    """Text lines of handwriting in a photo, top to bottom (then left to right)."""
    full = _load(image)
    W, H = full.size
    side = WORK_SIDE
    while True:
        scale = min(1.0, side / max(W, H))
        img = full.resize((max(1, round(W * scale)), max(1, round(H * scale))), Image.LANCZOS) if scale < 1 else full
        w, h = img.size
        mask = _remove_rules(ink_mask(img))
        if mask.sum() < 30:
            return []
        first = _blobs(mask, gap=max(3, int(0.012 * w)))
        line_h = _line_height(first, default=0.03 * h)
        # Small writing on a big photo: look again at a finer scale (letters ~20 px tall).
        if line_h >= 16 or scale >= 1 or side >= MAX_SIDE:
            break
        side = min(MAX_SIDE, int(side * 20 / max(line_h, 1)))
    mask = _drop_frames(mask, line_h)
    angle = _deskew_angle(mask)
    mask = _rotate(mask, angle)
    # Shorter straight strokes too, now that the writing size is known: rules broken up
    # by the writing on them, wires of a circuit sketch, a margin line.
    mask = _remove_rules(mask, 1.2 * line_h, 3.5 * line_h, max(4, int(0.2 * line_h)))
    blobs = _blobs(mask, gap=max(2, int(0.4 * line_h)))
    noise = max(6, int(0.02 * line_h * line_h))
    blobs = [b for b in blobs if b.ink >= noise]
    if not blobs:
        return []
    line_h = max(line_h, _line_height(blobs, default=line_h))
    pitch = _pitch(mask, line_h)
    if pitch is not None:
        # Dense writing, one line per rule: words with tall loops make the typical word
        # height look bigger than the line spacing allows.
        line_h = min(line_h, 0.75 * pitch)
    cut: list[_Blob] = []
    for b in blobs:
        cut += _cut_tall(mask, b, line_h)
    lines: list[InkLine] = []
    total = float(w * h)
    for g in _group(cut, line_h, w):
        whole = g[0]
        for b in g[1:]:
            whole = whole.join(b)
        if whole.h < 0.3 * line_h or whole.ink < 3 * noise:
            continue  # a stray mark, not a line
        if whole.ink > 0.6 * whole.w * whole.h and whole.h > 1.5 * line_h:
            continue  # a solid dark patch (a shadow edge, a sleeve), not handwriting
        if whole.h > 2 * line_h and whole.w < 0.5 * whole.h:
            continue  # a tall sliver: the page edge, a fold, a margin line
        segs = _segments(g, line_h)
        lines.append(InkLine(_unrotate(whole, angle, w, h), whole.ink / total,
                             [_unrotate(s, angle, w, h) for s in segs]))
    return _reading_order(lines)


def _reading_order(lines: list[InkLine]) -> list[InkLine]:
    """Top to bottom; lines at the same height (side by side) left to right."""
    rows: list[list[InkLine]] = []
    for ln in sorted(lines, key=lambda ln: ln.cy):
        row = rows[-1] if rows else None
        if row is not None:
            top = min(r.box[1] for r in row)
            bottom = max(r.box[1] + r.box[3] for r in row)
            overlap = min(bottom, ln.box[1] + ln.box[3]) - max(top, ln.box[1])
            if overlap >= 0.5 * min(ln.box[3], bottom - top):
                row.append(ln)
                continue
        rows.append([ln])
    return [ln for row in rows for ln in sorted(row, key=lambda ln: ln.box[0])]


def _unrotate(b: _Blob, angle: float, w: int, h: int) -> Box:
    """A box found on the deskewed mask, back on the original image (normalised)."""
    cx, cy = w / 2, h / 2
    a = math.radians(angle)  # PIL rotates counter-clockwise; undo it
    cos, sin = math.cos(a), math.sin(a)
    xs, ys = [], []
    for x, y in ((b.x0, b.y0), (b.x1, b.y0), (b.x0, b.y1), (b.x1, b.y1)):
        dx, dy = x - cx, y - cy
        xs.append(cx + dx * cos - dy * sin)
        ys.append(cy + dx * sin + dy * cos)
    x0, x1 = max(0.0, min(xs) / w), min(1.0, max(xs) / w)
    y0, y1 = max(0.0, min(ys) / h), min(1.0, max(ys) / h)
    return (x0, y0, max(0.0, x1 - x0), max(0.0, y1 - y0))


# ---------------------------------------------------------------- steps -> ink lines

SKIP = 1.0  # an ink line no step uses (the question, crossed-out work, a label)
MISS = 2.5  # a step no ink line fits (keeps Spark's box, corrected)
MERGE = 0.5  # per extra ink line one step takes
SHARE = 0.4  # per extra step on one ink line
W_ABS = 0.35  # weight of Spark's absolute position
W_GAP = 1.0  # weight of Spark's spacing between consecutive steps
MAX_SKIP = 8  # ink lines skipped between two steps
PAD = 0.004


def _union(boxes: list[Box]) -> Box:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return (x0, y0, x1 - x0, y1 - y0)


def _has_box(b: Box | None) -> bool:
    return b is not None and b[2] > 0 and b[3] > 0


@dataclass
class _Fit:
    """Spark's boxes are often shifted or squeezed as a whole: y' = a y + b, x' = x + c."""
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0

    def __call__(self, box: Box | None) -> Box | None:
        if not _has_box(box):
            return None
        x, y, w, h = box
        return (x + self.c, self.a * y + self.b, w, self.a * h)


def _cy(b: Box) -> float:
    return b[1] + b[3] / 2


def _match_cost(s: Box | None, g: Box, pitch: float) -> float:
    """How badly Spark's box `s` fits the ink box `g`, on its own. Spark's boxes drift
    by a line or two across a page, so where it puts a line counts for little; how
    far it puts it from the previous line (`_gap_cost`) counts for more."""
    if s is None:
        return 0.3
    dy = abs(_cy(s) - _cy(g)) / pitch
    dh = abs(s[3] - g[3]) / pitch
    overlap = max(0.0, min(s[0] + s[2], g[0] + g[2]) - max(s[0], g[0]))
    dx = 1.0 - overlap / max(1e-6, min(s[2], g[2]))
    return W_ABS * min(dy, 3.0) + 0.5 * min(dh, 2.0) + 0.4 * dx


def _gap_cost(prev: tuple[float, float] | None, s: Box | None, g: Box, pitch: float) -> float:
    """Spark's spacing from the previous step against the ink's spacing from its line."""
    if prev is None or s is None:
        return 0.0
    return W_GAP * min(abs((_cy(g) - prev[0]) - (_cy(s) - prev[1])) / pitch, 3.0)


def _split(line: InkLine, r: int, weights: list[float]) -> list[Box]:
    """Cut one ink line into r side-by-side pieces: at its widest gaps if it has
    enough separate pieces, else in proportion to the steps' lengths."""
    segs = sorted(line.segments or [line.box], key=lambda b: b[0])
    if len(segs) >= r:
        gaps = sorted(range(len(segs) - 1), key=lambda i: -(segs[i + 1][0] - segs[i][0] - segs[i][2]))[: r - 1]
        parts, start = [], 0
        for c in sorted(gaps):
            parts.append(_union(segs[start: c + 1]))
            start = c + 1
        parts.append(_union(segs[start:]))
        return parts
    x, y, w, h = line.box
    total = sum(weights) or 1.0
    out, at = [], x
    for wt in weights:
        pw = w * wt / total
        out.append((at, y, pw, h))
        at += pw
    return out


def _align(boxes: list[Box | None], lines: list[InkLine], pitch: float, weights: list[float],
           max_merge: int = 3, max_share: int = 3) -> list[tuple]:
    """Monotone alignment of steps to ink lines (dynamic programming). One plan entry
    per step: ("lines", j, k) | ("share", j, r) | ("miss",)."""
    n, m = len(boxes), len(lines)
    inf = float("inf")
    # State (i, j): steps before i are placed and the last placed one ends at ink line j.
    # Each state keeps its best path's last (ink centre, Spark centre) for the spacing term.
    cost = [[inf] * (m + 1) for _ in range(n + 1)]
    last: list[list[tuple[float, float] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    unions = {(j, k): _union([ln.box for ln in lines[j: j + k]]) for j in range(m) for k in range(1, max_merge + 1)
              if j + k <= m}
    match = {}
    for i in range(n):
        for key, g in unions.items():
            match[(i, key)] = _match_cost(boxes[i], g, pitch)

    def relax(ni, nj, total, prev, how):
        if total < cost[ni][nj]:
            cost[ni][nj], last[ni][nj], back[ni][nj] = total, prev, how

    for i in range(n):
        for j in range(m + 1):
            c = cost[i][j]
            if c == inf:
                continue
            prev = last[i][j]
            relax(i + 1, j, c + MISS, prev, (i, j, ("miss",)))
            for s in range((m if i == 0 else MAX_SKIP) + 1):
                j0 = j + s
                if j0 >= m:
                    break
                base = c + SKIP * s
                for k in range(1, max_merge + 1):
                    if j0 + k > m:
                        break
                    g = unions[(j0, k)]
                    add = match[(i, (j0, k))] + _gap_cost(prev, boxes[i], g, pitch) + MERGE * (k - 1)
                    nxt = (_cy(g), _cy(boxes[i])) if boxes[i] is not None else prev
                    relax(i + 1, j0 + k, base + add, nxt, (i, j, ("lines", j0, k)))
                for r in range(2, max_share + 1):
                    if i + r > n:
                        break
                    parts = _split(lines[j0], r, weights[i: i + r])
                    add, p = SHARE * (r - 1), prev
                    for t in range(r):
                        b = boxes[i + t]
                        add += _match_cost(b, parts[t], pitch) + _gap_cost(p, b, parts[t], pitch)
                        p = (_cy(parts[t]), _cy(b)) if b is not None else p
                    relax(i + r, j0 + 1, base + add, p, (i, j, ("share", j0, r)))
    end = min(range(m + 1), key=lambda j: cost[n][j] + SKIP * (m - j))
    plan: list[tuple] = []
    i, j = n, end
    while i > 0:
        pi, pj, how = back[i][j]
        plan.extend([how] * (how[2] if how[0] == "share" else 1))
        i, j = pi, pj
    plan.reverse()
    return plan


def _boxes_for(plan: list[tuple], lines: list[InkLine], weights: list[float]) -> list[Box | None]:
    out: list[Box | None] = []
    i = 0
    while i < len(plan):
        how = plan[i]
        if how[0] == "lines":
            out.append(_union([ln.box for ln in lines[how[1]: how[1] + how[2]]]))
            i += 1
        elif how[0] == "share":
            out.extend(_split(lines[how[1]], how[2], weights[i: i + how[2]]))
            i += how[2]
        else:
            out.append(None)
            i += 1
    return out


def _refit(raw: list[Box | None], snapped: list[Box | None]) -> _Fit:
    pairs = [(r, s) for r, s in zip(raw, snapped) if _has_box(r) and s is not None]
    if not pairs:
        return _Fit()
    ry = np.array([r[1] + r[3] / 2 for r, _ in pairs])
    sy = np.array([s[1] + s[3] / 2 for _, s in pairs])
    c = float(np.median([(s[0] + s[2] / 2) - (r[0] + r[2] / 2) for r, s in pairs]))
    if len(pairs) >= 3 and float(np.ptp(ry)) > 0.05:
        # Theil-Sen: the median of pairwise slopes shrugs off a few wrong matches.
        slopes = [(sy[j] - sy[i]) / (ry[j] - ry[i]) for i in range(len(ry)) for j in range(i + 1, len(ry))
                  if abs(ry[j] - ry[i]) > 0.02]
        if slopes:
            a = float(np.clip(np.median(slopes), 0.6, 1.6))
            return _Fit(a, float(np.median(sy - a * ry)), c)
    return _Fit(1.0, float(np.median(sy - ry)), c)


def snap_steps(steps: list, lines: list[InkLine]) -> list:
    """Each step's box replaced by the ink it was read from (see module doc).
    A step keeps its (shift-corrected) Spark box when no ink line fits it."""
    if not steps or not lines:
        return list(steps)
    raw = [s.bbox if _has_box(s.bbox) else None for s in steps]
    centers = sorted(ln.cy for ln in lines)
    height = float(np.median([ln.box[3] for ln in lines]))
    diffs = [b - a for a, b in itertools.pairwise(centers) if b - a > 0.5 * height]
    pitch = float(np.clip(np.median(diffs) if diffs else 2 * height, 1.2 * height, 4 * height))
    weights = [max(1.0, float(len(s.latex))) for s in steps]
    fit = _Fit()
    snapped: list[Box | None] = []
    for _ in range(3):  # align, correct Spark's boxes for a global shift/scale, align again
        snapped = _boxes_for(_align([fit(b) for b in raw], lines, pitch, weights), lines, weights)
        fit = _refit(raw, snapped)
    out = []
    for s, box, r in zip(steps, snapped, raw):
        guess = fit(r)
        if box is None:
            box = guess or s.bbox
        else:
            x, y, w, h = box
            if guess is not None and h > 2.2 * height and guess[3] < 0.6 * h:
                # A tall ink "line" is often two lines run together (dense writing on
                # ruled paper): keep the ink's sides, take the height from Spark's guess.
                half = max(guess[3], height) / 2
                top = min(max(y, _cy(guess) - half), y + h - 2 * half)
                y, h = top, 2 * half
            box = (max(0.0, x - PAD), max(0.0, y - PAD), w + 2 * PAD, h + 2 * PAD)
        out.append(s.model_copy(update={"bbox": box}))
    return out
