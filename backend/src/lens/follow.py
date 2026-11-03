"""Keep the outlined step located on the live page while the page or camera moves.

The check that found a mistake read one frame; the step's box is in that frame's
coordinates. Live tracking frames come later, from a page that may have shifted a few
centimetres, turned a few degrees or come closer. `Follow` takes the step's look in
the check frame (with a little of the writing around it, so one handwritten step on
ruled paper is not mistaken for its neighbour) and finds it in each new frame by
normalised cross-correlation over a few angles and scales. Plain numpy (FFT); no
model and no OpenCV.

It reports where the step is now (`quad`: its four corners, `box`: their bounds),
and a `state`:
- "ok": found in this frame;
- "held": not found for a moment (a hand over it), kept where it was last seen;
- "lost": not found for a while, or never found: the screen hides the outline rather
  than leave it on the wrong place.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

Box = tuple[float, float, float, float]  # normalised x, y, w, h


def gray_of(jpeg: bytes, width: int | None = None) -> np.ndarray:
    """A JPEG as a float32 grey image, optionally resized to `width` pixels wide."""
    img = Image.open(io.BytesIO(jpeg)).convert("L")
    if width and img.width != width:
        img = img.resize((width, max(1, round(img.height * width / img.width))), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def _box_sums(a: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Sum of `a` over every th x tw window (valid positions), from an integral image."""
    s = np.zeros((a.shape[0] + 1, a.shape[1] + 1), dtype=np.float64)
    s[1:, 1:] = a.cumsum(0).cumsum(1)
    return s[th:, tw:] - s[:-th, tw:] - s[th:, :-tw] + s[:-th, :-tw]


def ncc(image: np.ndarray, tpl: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of `tpl` at every valid position of `image`
    (the same as OpenCV's TM_CCOEFF_NORMED). Flat windows (blank paper) score 0."""
    H, W = image.shape
    th, tw = tpl.shape
    if th > H or tw > W:
        return np.zeros((0, 0), dtype=np.float32)
    t = tpl.astype(np.float64) - tpl.mean()
    tn = math.sqrt(float((t * t).sum()))
    if tn < 1e-6:
        return np.zeros((H - th + 1, W - tw + 1), dtype=np.float32)
    img = image.astype(np.float64)
    fi = np.fft.rfft2(img)
    ft = np.fft.rfft2(t, s=img.shape)
    corr = np.fft.irfft2(fi * np.conj(ft), s=img.shape)[: H - th + 1, : W - tw + 1]
    n = th * tw
    s1 = _box_sums(img, th, tw)
    s2 = _box_sums(img * img, th, tw)
    var = np.maximum(s2 - s1 * s1 / n, 0.0)
    den = np.sqrt(var) * tn
    out = np.where(den > 1e-3 * tn * math.sqrt(n), corr / np.maximum(den, 1e-9), 0.0)
    return out.astype(np.float32)


def box_blur(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1) x (2r+1) box around every pixel (edges repeated)."""
    h, w = a.shape
    k = 2 * r + 1
    pad = np.pad(a.astype(np.float64), r, mode="edge")
    s = np.zeros((pad.shape[0] + 1, pad.shape[1] + 1))
    s[1:, 1:] = pad.cumsum(0).cumsum(1)
    return (s[k:k + h, k:k + w] - s[:h, k:k + w] - s[k:k + h, :w] + s[:h, :w]) / (k * k)


def ink(gray: np.ndarray, r: int, soften: int = 1) -> np.ndarray:
    """Dark strokes on light paper, as positive values: the local background (a box blur of
    radius `r`) minus the image, never below 0, then softened a little. Shadows, room light
    and the brightness of a sleeve or the desk drop out; what is left is mostly the writing
    itself, which is what tells one step from another. Softening keeps a sharp check frame
    comparable with a live frame blurred by motion or compression."""
    out = np.maximum(box_blur(gray, r) - gray, 0.0)
    return (box_blur(out, soften) if soften else out).astype(np.float32)


def _second_peak(res: np.ndarray, at: tuple[int, int], ry: int, rx: int) -> float:
    """The best score outside a (2ry+1) x (2rx+1) neighbourhood of `at`."""
    y, x = at
    r = res.copy()
    r[max(0, y - ry): y + ry + 1, max(0, x - rx): x + rx + 1] = -1.0
    return float(r.max()) if r.size else -1.0


@dataclass
class FollowConfig:
    min_score: float = 0.6  # a match below this is not the step (or the step is covered)
    sure_score: float = 0.8  # a match this good needs only `sure_margin` over the next best place
    margin: float = 0.1  # else the best place must beat any other place by this much
    sure_margin: float = 0.05
    hold: int = 6  # frames a covered step is kept where it was last seen (~2 s at 3 fps)
    wide_every: int = 3  # once lost, look over the whole frame every this many frames
    reach: float = 0.12  # search this far (fraction of the frame) around the last place
    max_angle: float = 8.0  # degrees the page may have turned since the check frame
    context: float = 0.6  # template = the step box plus this many box heights above and below
    jump: float = 0.03  # a middling match may land at most this far (of the frame height, or one
    #                     step height if more) from where the step should be


@dataclass
class Follow:
    """Follow one step box from the check frame (`src`, a grey image) through later frames."""

    src: np.ndarray
    box: Box
    cfg: FollowConfig = field(default_factory=FollowConfig)
    # Pose of the step now: centre (px), turn (degrees), scale; relative to the check frame.
    cx: float = 0.0
    cy: float = 0.0
    angle: float = 0.0
    scale: float = 1.0
    found: bool = False  # located at least once
    lost: int = 0  # frames in a row without a sure match
    score: float = 0.0
    _frames: int = 0
    _size: tuple[int, int] | None = None  # (w, h) of the frames followed
    _ref: np.ndarray | None = None  # the check frame at the followed frames' size
    _tpl: dict = field(default_factory=dict)
    _vel: tuple[float, float] = (0.0, 0.0)  # last move between two frames (px)

    def _prepare(self, shape: tuple[int, int]) -> bool:
        h, w = shape
        if self._size == (w, h):
            return True
        sh, sw = self.src.shape
        if abs(sw / sh - w / h) > 0.03 * (w / h):
            return False  # another camera or the phone turned: the check frame cannot be compared
        img = Image.fromarray(self.src.astype(np.float32)).resize((w, h), Image.BILINEAR)
        self._r = max(2, round(0.01 * max(w, h)))
        self._ref = ink(np.asarray(img, dtype=np.float32), self._r)
        self._paper = 0.0  # fills any part of a template beyond the frame: no ink there
        self._size = (w, h)
        x, y, bw, bh = self.box
        self._c0 = ((x + bw / 2) * w, (y + bh / 2) * h)
        self._half = (bw * w / 2, bh * h / 2)
        pad_y = self.cfg.context * bh * h
        pad_x = min(0.5 * bh * h + 0.01 * w, 0.25 * bw * w)
        self._tsize = (max(8, round(bw * w + 2 * pad_x)), max(6, round(bh * h + 2 * pad_y)))
        if not self.found:
            self.cx, self.cy = self._c0
        # Search at half size only when the step is big enough to stay legible there.
        self._coarse = 2 if min(self._tsize[1], self._tsize[0] / 4) >= 40 else 1
        self._tpl.clear()
        return True

    def _template(self, angle: float, scale: float, step: int = 1) -> np.ndarray:
        """How the step region would look in a frame where the page turned by `angle` and
        grew by `scale`, sampled from the check frame (with its real surroundings, no fill)."""
        key = (round(angle, 2), round(scale, 3), step)
        t = self._tpl.get(key)
        if t is not None:
            return t
        tw, th = self._tsize
        ow, oh = max(4, round(tw * scale / step)), max(4, round(th * scale / step))
        a = math.radians(angle)
        ca, sa = math.cos(a), math.sin(a)
        k = step / scale
        cx0, cy0 = self._c0
        # output (u, v) -> offset q = ((u - ow/2), (v - oh/2)) * step -> source c0 + R(-a) q / scale
        coeffs = (ca * k, sa * k, cx0 - ca * k * ow / 2 - sa * k * oh / 2,
                  -sa * k, ca * k, cy0 + sa * k * ow / 2 - ca * k * oh / 2)
        img = Image.fromarray(self._ref)
        t = np.asarray(img.transform((ow, oh), Image.AFFINE, coeffs, resample=Image.BILINEAR,
                                       fillcolor=self._paper), dtype=np.float32)
        if len(self._tpl) > 64:
            self._tpl.clear()
        self._tpl[key] = t
        return t

    def _search(self, frame: np.ndarray, centre: tuple[float, float] | None, angles, scales, step: int,
                reach: float = 0.0):
        """Best (score, second, cx, cy, angle, scale) over the candidates, searched within
        `reach` (fraction of the frame, plus 3 px) of `centre`; `centre` None: the whole frame."""
        img = frame[::step, ::step] if step > 1 else frame
        H, W = img.shape
        best = None
        for a in angles:
            for s in scales:
                tpl = self._template(a, s, step)
                th, tw = tpl.shape
                if th >= H or tw >= W:
                    continue
                if centre is None:
                    x0, y0, x1, y1 = 0, 0, W, H
                else:
                    rx, ry = reach * W + 3, reach * H + 3
                    cx, cy = centre[0] / step, centre[1] / step
                    x0, y0 = int(max(0, cx - tw / 2 - rx)), int(max(0, cy - th / 2 - ry))
                    x1, y1 = int(min(W, cx + tw / 2 + rx)), int(min(H, cy + th / 2 + ry))
                win = img[y0:y1, x0:x1]
                if win.shape[0] < th or win.shape[1] < tw:
                    continue
                res = ncc(win, tpl)
                if res.size == 0:
                    continue
                iy, ix = np.unravel_index(int(np.argmax(res)), res.shape)
                score = float(res[iy, ix])
                if best is None or score > best[0]:
                    # Another place as good is another step that looks alike: not sure.
                    bh = self._half[1] * 2 * s / step
                    second = _second_peak(res, (iy, ix), max(2, int(0.6 * bh)), max(2, int(0.3 * tw)))
                    best = (score, second, (x0 + ix + tw / 2) * step, (y0 + iy + th / 2) * step, a, s)
        return best

    def _refine(self, frame: np.ndarray, hit):
        """Full-size search right around a half-size hit, over a little turn and size; keeps
        the hit's second-best score (the full-size window is too small to judge that)."""
        if hit is None:
            return None
        fine = self._search(frame, hit[2:4], [hit[4] + d for d in (-0.75, 0.0, 0.75)],
                            [hit[5] * f for f in (0.98, 1.0, 1.02)], 1, 0.0)
        if fine is None:
            return hit
        return (fine[0], hit[1] + (fine[0] - hit[0]), *fine[2:])

    def update(self, frame: np.ndarray) -> dict | None:
        """One live frame (grey, any size with the check frame's aspect) -> where the step is now."""
        if not self._prepare(frame.shape):
            return None
        frame = ink(frame, self._r)
        c = self.cfg
        self._frames += 1
        hit = None
        local = self.found and self.lost <= c.hold
        if local:
            # Near where it should be now (last place + last move): find it at half size, then
            # the exact turn and size at full size.
            px, py = self.cx + self._vel[0], self.cy + self._vel[1]
            near = self._search(frame, (px, py), [self.angle - 1.5, self.angle, self.angle + 1.5],
                                [self.scale], self._coarse, c.reach)
            hit = self._refine(frame, near)
        elif not self.found or self.lost % c.wide_every == 0:
            # First look (the page may have moved while the check ran) or lost: the whole
            # frame at half size, then refine around the best place.
            angles = np.arange(-c.max_angle, c.max_angle + 0.1, 2.0).tolist()
            coarse = self._search(frame, None, angles, (0.9, 1.0, 1.1), self._coarse)
            if coarse is not None and coarse[0] >= c.min_score - 0.15:
                mid = self._search(frame, coarse[2:4], [coarse[4] - 1, coarse[4], coarse[4] + 1],
                                   [coarse[5] * f for f in (0.95, 1.0, 1.05)], self._coarse, 0.01)
                if mid is not None:
                    mid = (mid[0], max(mid[1], coarse[1]), *mid[2:])  # the whole frame's second-best place
                hit = self._refine(frame, mid)
        if hit is not None:
            self.score = hit[0]
        need = c.sure_margin if hit is not None and hit[0] >= c.sure_score else c.margin
        sure = hit is not None and hit[0] >= c.min_score and hit[0] - hit[1] >= need
        if sure and local and hit[0] < c.sure_score:
            # A middling match far from where the step should be is more likely a step that looks
            # alike (the real one covered by a hand) than the page jumping between two frames.
            jump = math.hypot(hit[2] - px, hit[3] - py)
            sure = jump <= max(self._half[1] * 2 * self.scale, c.jump * frame.shape[0])
        if sure:
            moved = (hit[2] - self.cx, hit[3] - self.cy) if local and self.lost == 0 else (0.0, 0.0)
            _, _, self.cx, self.cy, self.angle, self.scale = hit
            self._vel = moved
            self.found, self.lost = True, 0
        else:
            self.lost += 1
            self._vel = (0.0, 0.0)
        return self.report()

    def quad(self) -> list[tuple[float, float]]:
        """The step box's corners now (tl, tr, br, bl), normalised."""
        w, h = self._size
        hw, hh = self._half
        a = math.radians(self.angle)
        ca, sa = math.cos(a), math.sin(a)
        out = []
        for qx, qy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
            px = self.cx + self.scale * (ca * qx - sa * qy)
            py = self.cy + self.scale * (sa * qx + ca * qy)
            out.append((round(float(px / w), 4), round(float(py / h), 4)))
        return out

    def report(self) -> dict:
        state = "lost" if not self.found or self.lost > self.cfg.hold else ("ok" if self.lost == 0 else "held")
        q = self.quad()
        xs, ys = [p[0] for p in q], [p[1] for p in q]
        return {"anchor": [round(v, 4) for v in self.box], "quad": [list(p) for p in q],
                "box": [round(min(xs), 4), round(min(ys), 4), round(max(xs) - min(xs), 4), round(max(ys) - min(ys), 4)],
                "angle": round(float(self.angle), 2), "score": round(float(self.score), 3), "lost": self.lost, "state": state}
