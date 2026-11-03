"""Outline follow on a synthetic moving-camera clip: does the outline stay on the mistaken step?

    uv run python scripts/outline_check.py PAGE.jpg [PAGE2.jpg ...] [--seconds 40] [--json OUT]

A real handwritten page (e.g. a FERMAT photo from the gitignored cache; nothing is
committed) is filmed by a simulated camera, the way Lens sees it in each mode:
- phone: phone over the notebook (portrait 3:4); slow drift, hand tremor, the writing
  hand crossing the page, and the notebook nudged about 3 cm and turned 4 degrees;
- webcam: page held up to the laptop camera (16:9, room around it); a steady sway of a
  few degrees and centimetres, thumbs at the page edges, a finger pointing now and then.

A check reads one frame (at full size), takes a while (a whole-page check took 20-45 s
in the owner's test), and its result outlines one step (a found ink line in the middle
of the page). From then on every tracking frame (512 px, 3 of every 5 offered frames
processed, as on the live service) moves the outline. Truth is the step's box carried
by the same camera motion. Printed per mode, before (the code as it was) and after:
- centre error (normalised, median and p90) and IoU over the frames the outline is shown,
- "wrong": frames the outline is shown solid with IoU < 0.3 (on the wrong place); "faded":
  the same while it is drawn faded (its step covered a moment, kept where last seen),
- "hidden": frames with the step in view but no outline shown.

"Before" replays the old path exactly: the template is cut from the tracking frame
numbered like the check (or, when the service skipped that frame, from the frame at
hand); a check that waited in the queue meanwhile overwrote that number; the outline
moves by shift and scale only, and stays where it was last matched when lost.
"After": lens.follow.Follow, started from the check frame itself.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from lens.follow import Follow, gray_of, ncc
from lens.lines import find_lines

FPS = 5
PROCESSED = {0, 2, 3}  # of every 5 offered frames, these reach the GPU (newest wins)


@dataclass
class Mode:
    name: str
    size: tuple[int, int]  # tracking frame (w, h)
    page_h: float  # page height as a fraction of the frame height
    sway: tuple[float, float, float, float]  # dx, dy (fraction of frame), angle (deg), zoom
    period: float  # seconds per sway cycle
    tremor: float  # px jitter
    nudge: tuple[float, float, float, float]  # at nudge time: dx, dy, angle, zoom (step change)
    keystone: float  # perspective strength (about the fraction of size change from one side to the other)


MODES = {
    "phone": Mode("phone", (512, 683), 0.95, (0.025, 0.02, 1.5, 0.03), 11.0, 1.5, (0.10, 0.03, 4.0, 1.0), 0.04),
    "webcam": Mode("webcam", (683, 384), 0.85, (0.05, 0.04, 4.0, 0.06), 5.0, 2.5, (0.06, 0.02, -5.0, 1.05), 0.05),
}


def apply(m: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = m @ np.array([x, y, 1.0])
    return v[0] / v[2], v[1] / v[2]


def affine(tx: float, ty: float, angle: float, s: float, cx: float, cy: float) -> np.ndarray:
    """3x3: rotate by `angle` and scale by `s` about (cx, cy), then shift."""
    a = math.radians(angle)
    r = np.array([[s * math.cos(a), -s * math.sin(a)], [s * math.sin(a), s * math.cos(a)]])
    m = np.eye(3)
    m[:2, :2] = r
    m[:2, 2] = np.array([cx, cy]) - r @ np.array([cx, cy]) + np.array([tx, ty])
    return m


class Camera:
    """Page pixels -> frame pixels (normalised to the frame) at time t."""

    def __init__(self, mode: Mode, page: Image.Image, nudge_at: float, seed: int):
        self.mode, self.page, self.nudge_at = mode, page, nudge_at
        self.rng = np.random.default_rng(seed)
        w, h = mode.size
        s = min(mode.page_h * h / page.height, 0.95 * w / page.width)  # the whole page in view
        self.base = np.array([[s, 0, (w - page.width * s) / 2], [0, s, (h - page.height * s) / 2], [0, 0, 1]])
        self.jit: dict[int, tuple[float, float]] = {}

    def matrix(self, t: float, n: int) -> np.ndarray:
        """Frame-pixel transform (at tracking size) for frame n at time t."""
        m = self.mode
        w, h = m.size
        p = 2 * math.pi * t / m.period
        dx, dy = m.sway[0] * w * math.sin(p), m.sway[1] * h * math.sin(2 * p + 1)
        ang, z = m.sway[2] * math.sin(p + 0.5), 1 + m.sway[3] * math.sin(p + 2)
        k = min(1.0, max(0.0, (t - self.nudge_at) / 0.5))  # the nudge takes half a second
        dx, dy = dx + k * m.nudge[0] * w, dy + k * m.nudge[1] * h
        ang, z = ang + k * m.nudge[2], z * (1 + k * (m.nudge[3] - 1))
        if n not in self.jit:
            self.jit[n] = tuple(self.rng.normal(0, m.tremor, 2))
        jx, jy = self.jit[n]
        # A camera tilted a few degrees off the page: a slight keystone, varying slowly.
        persp = np.eye(3)
        persp[2, :2] = (m.keystone * math.sin(p + 1.3) / w, m.keystone * math.cos(0.7 * p) / h)
        centre_px = np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1.0]])
        persp = centre_px @ persp @ np.linalg.inv(centre_px)
        return persp @ affine(dx + jx, dy + jy, ang, z, w / 2, h / 2) @ self.base

    def render(self, t: float, n: int, scale: float = 1.0, hands: list | None = None,
               page: Image.Image | None = None) -> Image.Image:
        w, h = self.mode.size
        mat = np.diag([scale, scale, 1.0]) @ self.matrix(t, n)
        inv = np.linalg.inv(mat)
        inv = inv / inv[2, 2]
        size = (round(w * scale), round(h * scale))
        bg = (70, 62, 58) if self.mode.name == "phone" else (120, 112, 104)
        img = (page or self.page).transform(size, Image.PERSPECTIVE, tuple(inv.ravel()[:8]),
                                            resample=Image.BILINEAR, fillcolor=bg)
        d = ImageDraw.Draw(img)
        for poly in hands or []:
            d.polygon([(x * scale, y * scale) for x, y in poly], fill=(196, 150, 120))
        # Room light and auto-exposure drift; a moving camera blurs a little.
        img = ImageEnhance.Brightness(img).enhance(1 + 0.1 * math.sin(2 * math.pi * t / 13.0))
        if self.jit[n][0] ** 2 + self.jit[n][1] ** 2 > (1.5 * self.mode.tremor) ** 2:
            img = img.filter(ImageFilter.GaussianBlur(0.8 * scale))
        return img


def hands_at(mode: Mode, t: float, target: tuple[float, float]) -> list:
    """Skin-coloured shapes over the frame (tracking-size pixels)."""
    w, h = mode.size
    out = []
    if mode.name == "phone":
        # The writing hand sweeps across below/onto the page, crossing the outlined step at times.
        phase = (t % 9.0) / 9.0
        if phase < 0.6:
            x = w * (0.15 + 1.2 * phase)
            y = target[1] + h * (0.03 + 0.10 * math.sin(6 * phase))
            out.append([(x - 40, y), (x + 40, y + 10), (x + 70, h + 10), (x - 30, h + 10)])
            out.append([(x - 8, y - 45), (x + 6, y - 45), (x + 20, y + 5), (x - 20, y + 5)])  # pen and fingers
    else:
        # Thumbs on both side edges, and every 7 s a finger pointing at the page for 2 s.
        for side in (0, 1):
            x0 = 0 if side == 0 else w
            sx = 1 if side == 0 else -1
            out.append([(x0, h * 0.55), (x0 + sx * 0.07 * w, h * 0.6), (x0 + sx * 0.07 * w, h * 0.72), (x0, h * 0.8)])
        if (t % 7.0) < 2.0:
            fx, fy = target[0] + 0.05 * w, target[1] + 0.06 * h
            out.append([(fx - 8, fy), (fx + 8, fy), (fx + 40, h + 5), (fx - 10, h + 5)])
    return out


def poly_iou(a: list, b: list, size: tuple[int, int]) -> float:
    w, h = size
    ma, mb = Image.new("1", size), Image.new("1", size)
    ImageDraw.Draw(ma).polygon([(x * w, y * h) for x, y in a], fill=1)
    ImageDraw.Draw(mb).polygon([(x * w, y * h) for x, y in b], fill=1)
    A, B = np.asarray(ma), np.asarray(mb)
    union = (A | B).sum()
    return float((A & B).sum() / union) if union else 0.0


def centre(poly: list) -> tuple[float, float]:
    return float(np.mean([p[0] for p in poly])), float(np.mean([p[1] for p in poly]))


def jpeg(img: Image.Image, q: int = 65) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    return buf.getvalue()


class OldFollow:
    """The follow logic as it was (modal_apps/sam31.py before this change), with numpy NCC."""

    def __init__(self, gray, box):
        h, w = gray.shape
        x, y, bw, bh = box
        x0, y0 = max(0, int(x * w)), max(0, int(y * h))
        x1, y1 = min(w, int((x + bw) * w)), min(h, int((y + bh) * h))
        self.tpl = gray[y0:y1, x0:x1].copy()
        self.box = [x0, y0, x1 - x0, y1 - y0]
        self.lost, self.wide = 0, True

    def update(self, gray):
        h, w = gray.shape
        x, y, bw, bh = self.box
        best = None
        for s in (0.9, 1.0, 1.1):
            tw, th = max(4, int(self.tpl.shape[1] * s)), max(4, int(self.tpl.shape[0] * s))
            if tw >= w or th >= h:
                continue
            tpl = np.asarray(Image.fromarray(self.tpl).resize((tw, th), Image.BILINEAR), dtype=np.float32)
            reach = 1.0 if self.wide else 0.25
            mx, my = max(tw, int(reach * w)), max(th, int(reach * h))
            sx0, sy0 = max(0, x + bw // 2 - tw // 2 - mx), max(0, y + bh // 2 - th // 2 - my)
            sx1, sy1 = min(w, x + bw // 2 + tw // 2 + mx), min(h, y + bh // 2 + th // 2 + my)
            win = gray[sy0:sy1, sx0:sx1]
            if win.shape[0] < th or win.shape[1] < tw:
                continue
            res = ncc(win, tpl)
            iy, ix = np.unravel_index(int(np.argmax(res)), res.shape)
            score = float(res[iy, ix])
            if best is None or score > best[0]:
                best = (score, [sx0 + ix, sy0 + iy, tw, th])
        if best is None:
            return None
        score, box = best
        if score >= 0.45:
            self.box, self.lost, self.wide = box, 0, False
        else:
            self.lost += 1
        bx, by, bw2, bh2 = self.box
        return {"box": [bx / w, by / h, bw2 / w, bh2 / h], "lost": self.lost}


def old_move(poly: list, anchor, to) -> list:
    """Frontend moveResult as it was: shift and scale from the anchor box to the followed box."""
    sx, sy = to[2] / anchor[2], to[3] / anchor[3]
    return [(to[0] + (x - anchor[0]) * sx, to[1] + (y - anchor[1]) * sy) for x, y in poly]


def new_move(poly: list, anchor, quad) -> list:
    """Frontend moveResult now: the affine map from the anchor box's corners to the followed quad."""
    (x0, y0), (x1, y1), (x3, y3) = quad[0], quad[1], quad[3]
    ax, ay, aw, ah = anchor
    out = []
    for x, y in poly:
        u, v = (x - ax) / aw, (y - ay) / ah
        out.append((x0 + u * (x1 - x0) + v * (x3 - x0), y0 + u * (y1 - y0) + v * (y3 - y0)))
    return out


def run(page_path: Path, mode: Mode, seconds: float, check_at: float, check_s: float, queued: bool, seed: int,
        snap: Path | None = None):
    page = Image.open(page_path).convert("RGB")
    cam = Camera(mode, page, nudge_at=check_at + check_s + 6.0, seed=seed)
    w, h = mode.size
    # The mistaken step: an ink line from the middle of the page, as the reader would box it.
    lines = [ln for ln in find_lines(page_path.read_bytes()) if ln.box[2] > 0.25]
    med_h = float(np.median([ln.box[3] for ln in lines]))
    single = [ln for ln in lines if ln.box[3] <= 1.5 * med_h] or lines  # one written step, not a merged block
    step_page = single[len(single) // 2].box
    n_check = round(check_at * FPS)
    m_check = cam.matrix(check_at, n_check)
    pw, ph = page.size

    def to_frame(m, pts):
        out = []
        for x, y in pts:
            fx, fy = apply(m, x * pw, y * ph)
            out.append((fx / w, fy / h))
        return out

    sx, sy, sw, sh = step_page
    corners = [(sx, sy), (sx + sw, sy), (sx + sw, sy + sh), (sx, sy + sh)]
    cq = to_frame(m_check, corners)
    xs, ys = [p[0] for p in cq], [p[1] for p in cq]
    box = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))  # the reader's box on the check frame
    inv_check = np.linalg.inv(m_check)
    bx = [(box[0], box[1]), (box[0] + box[2], box[1]), (box[0] + box[2], box[1] + box[3]), (box[0], box[1] + box[3])]
    box_corners_page = [apply(inv_check, x * w, y * h) for x, y in bx]  # truth: that box, moving with the page

    # The student goes on writing below the step after the hint: the lines under it are
    # blank on the check frame and appear one by one later (the outline's surroundings change).
    paper = tuple(int(v) for v in np.median(np.asarray(page).reshape(-1, 3), axis=0))
    below = [ln.box for ln in lines if ln.box[1] > sy + sh][:3]
    versions = []
    for k in range(len(below) + 1):
        v = page.copy()
        d = ImageDraw.Draw(v)
        for b in below[k:]:
            d.rectangle([b[0] * pw - 4, b[1] * ph - 4, (b[0] + b[2]) * pw + 4, (b[1] + b[3]) * ph + 4], fill=paper)
        versions.append(v)
    t_result = check_at + check_s

    def frame(n: int, scale: float = 1.0) -> Image.Image:
        t = n / FPS
        shown_page = versions[min(len(versions) - 1, max(0, int((t - t_result) / 8.0) + 0))] if t >= t_result \
            else versions[0]
        tgt = to_frame(cam.matrix(t, n), [((box_corners_page[0][0] + box_corners_page[2][0]) / 2 / pw,
                                           (box_corners_page[0][1] + box_corners_page[2][1]) / 2 / ph)])[0]
        cam.matrix(t, n)  # draws this frame's tremor
        return cam.render(t, n, scale, hands_at(mode, t, (tgt[0] * w, tgt[1] * h)), shown_page)

    check_jpeg = jpeg(frame(n_check, 2.0), 80)  # the check frame, grabbed at full size
    n_result = round((check_at + check_s) * FPS)
    n_queued = round((check_at + check_s / 2) * FPS)  # a second check grabbed while the first ran
    old_seq = n_queued if queued else n_check
    processed = [n for n in range(round(seconds * FPS)) if n % 5 in PROCESSED]
    grays = {n: gray_of(jpeg(frame(n))) for n in processed if n <= n_result and n >= min(n_check, old_seq)}
    old = new = None
    old_last = None
    rows = {"before": [], "after": []}
    for n in processed:
        if n < n_result:
            continue
        g = grays.get(n) if n in grays else gray_of(jpeg(frame(n)))
        mt = cam.matrix(n / FPS, n)
        truth = [(fx / w, fy / h) for fx, fy in (apply(mt, x, y) for x, y in box_corners_page)]
        if old is None:
            src = grays.get(old_seq, g)  # skipped by the service: the frame at hand
            old = OldFollow(src, box)
            new = Follow(gray_of(check_jpeg), box)
        o = old.update(g)
        old_last = o or old_last
        shown_old = old_move(bx, box, old_last["box"]) if old_last else bx
        r = new.update(g)
        shown_new = new_move(bx, box, r["quad"]) if r and r["state"] != "lost" else None
        in_view = all(0 <= x <= 1 and 0 <= y <= 1 for x, y in truth)
        if snap is not None and n in (processed[processed.index(n)] if n == n_result else -1,
                                      round(cam.nudge_at * FPS) + 5 - (round(cam.nudge_at * FPS) + 5) % 5):
            img = frame(n).convert("RGB")
            d = ImageDraw.Draw(img)
            for poly, col in ((truth, (0, 200, 0)), (shown_old, (230, 0, 0)), (shown_new, (0, 90, 255))):
                if poly:
                    d.polygon([(x * w, y * h) for x, y in poly], outline=col, width=2)
            img.save(snap.with_name(f"{snap.stem}_{mode.name}{'_q' if queued else ''}_{n}.png"))
        for key, shown, fr in (("before", shown_old, None), ("after", shown_new, r)):
            if shown is None:
                rows[key].append({"n": n, "follow": fr, "shown": False, "in_view": in_view})
                continue
            c, tc = centre(shown), centre(truth)
            rows[key].append({"n": n, "follow": fr, "shown": True, "in_view": in_view,
                              "faded": bool(fr and fr["state"] == "held"),
                              "err": math.hypot(c[0] - tc[0], c[1] - tc[1]), "iou": poly_iou(shown, truth, (w, h))})
    return rows


def summary(rows: list[dict]) -> dict:
    shown = [r for r in rows if r["shown"]]
    errs = sorted(r["err"] for r in shown) or [float("nan")]
    ious = sorted(r["iou"] for r in shown) or [float("nan")]
    return {
        "frames": len(rows),
        "centre_err_median": round(statistics.median(errs), 4),
        "centre_err_p90": round(errs[int(0.9 * (len(errs) - 1))], 4),
        "iou_median": round(statistics.median(ious), 3),
        "wrong_pct": round(100 * sum(r["iou"] < 0.3 and not r["faded"] for r in shown) / max(1, len(rows)), 1),
        "faded_wrong_pct": round(100 * sum(r["iou"] < 0.3 and r["faded"] for r in shown) / max(1, len(rows)), 1),
        "hidden_pct": round(100 * sum(not r["shown"] and r["in_view"] for r in rows) / max(1, len(rows)), 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pages", nargs="+")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--check-at", type=float, default=4.0)
    ap.add_argument("--check-s", type=float, default=25.0, help="how long the check took")
    ap.add_argument("--json")
    ap.add_argument("--snap", help="save the first result frame of each mode here (PNG), outline drawn")
    a = ap.parse_args()
    pages = []
    for p in a.pages:  # a folder means every page photo in it (not the *.boxes / *.snap / *.sam pictures)
        pp = Path(p)
        pages += sorted(pp.glob("*.jpg")) if pp.is_dir() else [pp]
    pages = [q for q in pages if not q.name.endswith((".boxes.jpg", ".snap.jpg", ".sam.jpg"))]
    report = {}
    for mode in MODES.values():
        for queued in (False, True):
            agg = {"before": [], "after": []}
            for i, p in enumerate(pages):
                rows = run(Path(p), mode, a.seconds, a.check_at, a.check_s, queued, seed=i,
                           snap=Path(a.snap) if a.snap and i == 0 else None)
                for k in agg:
                    agg[k] += rows[k]
            key = f"{mode.name}{' +queued check' if queued else ''}"
            report[key] = {k: summary(v) for k, v in agg.items()}
            for k, s in report[key].items():
                print(f"{key:22s} {k:6s} centre err median {s['centre_err_median']:.4f} p90 {s['centre_err_p90']:.4f}  "
                      f"IoU median {s['iou_median']:.2f}  wrong {s['wrong_pct']:5.1f}%  (faded {s['faded_wrong_pct']:4.1f}%)  "
                      f"hidden {s['hidden_pct']:5.1f}%  "
                      f"({s['frames']} frames)")
    if a.json:
        Path(a.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
