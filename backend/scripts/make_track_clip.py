"""Make a synthetic tracking clip: real handwriting appears line by line under a moving hand + pen.

    uv run python scripts/make_track_clip.py PAGE.png SPRITE.png OUT_DIR [--fps 5] [--drift]

PAGE is a photo or scan of handwritten work. SPRITE is an RGBA cut-out of a hand
holding a pen (the alpha covers hand + pen), with SPRITE.pen.npy beside it (the pen's
bool mask, used to find the pen tip). Both stay outside the repo (they are test data
we may not redistribute). OUT_DIR gets f0000.jpg... and truth.json with, per frame:
the phase (writing / paused / clear), the pen tip, and the boxes of finished lines.

Lines are found from the page's ink rows; each is "written" left to right with the pen
tip on the writing front, then the hand rests, and every second line it leaves the frame.
The writer sits below the page: the cut-out is turned so the pen points up, and a sleeve
joins the hand to the bottom edge, as a phone over a notebook sees it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H = 640, 480


def ink_lines(gray: np.ndarray, min_rows: int = 6) -> list[tuple[int, int, int, int]]:
    """Bands of rows that hold ink, as (x0, y0, x1, y1) in page pixels."""
    ink = gray < 110
    rows = ink.sum(1) > max(2, 0.004 * gray.shape[1])
    bands, start = [], None
    for y, on in enumerate(np.append(rows, False)):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if y - start >= min_rows:
                bands.append((start, y))
            start = None
    lines = []
    for y0, y1 in bands:
        cols = np.nonzero(ink[y0:y1].any(0))[0]
        if len(cols):
            lines.append((int(cols[0]), y0, int(cols[-1]) + 1, y1))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("page")
    ap.add_argument("sprite")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--drift", action="store_true", help="move the camera slowly (tests outlines that follow)")
    ap.add_argument("--max-lines", type=int, default=6)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    page = Image.open(a.page).convert("RGB")
    scale = min((W * 0.86) / page.width, (H * 0.86) / page.height)
    page = page.resize((int(page.width * scale), int(page.height * scale)), Image.LANCZOS)
    pw, ph = page.size
    px, py = (W - pw) // 2, (H - ph) // 2
    page_np = np.asarray(page)
    gray = np.asarray(page.convert("L"))
    blank = np.full_like(page_np, np.median(page_np.reshape(-1, 3), axis=0).astype(np.uint8))
    lines = ink_lines(gray)[: a.max_lines]

    sprite = Image.open(a.sprite).convert("RGBA")
    pen = np.load(Path(a.sprite).with_suffix(".pen.npy"))
    alpha = np.asarray(sprite)[..., 3] > 0
    ys, xs = np.nonzero(alpha)
    crop = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    sprite = sprite.crop(crop)
    pen = pen[crop[1]:crop[3], crop[0]:crop[2]][::-1, ::-1]  # turned half a turn with the sprite
    sprite = sprite.rotate(180)
    pys, pxs = np.nonzero(pen)
    tip_i = int(np.argmin(pys))  # the pen now points up, away from the writer
    s = (W * 0.42) / sprite.width
    sprite = sprite.resize((int(sprite.width * s), int(sprite.height * s)), Image.LANCZOS)
    tip = (pxs[tip_i] * s, pys[tip_i] * s)
    ays, axs = np.nonzero(np.asarray(sprite)[..., 3] > 0)
    far = int(np.argmax((axs - tip[0]) ** 2 + (ays - tip[1]) ** 2))
    wrist = (axs[far], ays[far])  # where the sleeve joins the hand

    fps = a.fps
    timeline = []  # (phase, line index, progress 0..1)
    for i in range(len(lines)):
        timeline += [("writing", i, k / (3 * fps)) for k in range(3 * fps + 1)]
        timeline += [("paused", i, 1.0)] * fps
        if i % 2 == 1:
            timeline += [("clear", i, 1.0)] * (2 * fps)
    revealed = np.zeros(gray.shape, bool)
    truth = []
    for n, (phase, li, prog) in enumerate(timeline):
        x0, y0, x1, y1 = lines[li]
        front = int(x0 + prog * (x1 - x0))
        revealed[y0:y1, x0:front] = True
        for j in range(li):  # earlier lines are complete
            a0, b0, a1, b1 = lines[j]
            revealed[b0:b1, a0:a1] = True
        canvas = Image.new("RGB", (W, H), (92, 74, 58))  # desk
        canvas.paste(Image.fromarray(np.where(revealed[..., None], page_np, blank)), (px, py))
        tip_xy = None
        if phase != "clear":
            tx, ty = px + front, py + (y0 + y1) // 2
            if phase == "paused":
                tx, ty = tx + 12, ty - 10  # pen lifted a little
            ox, oy = int(tx - tip[0]), int(ty - tip[1])
            wx, wy = ox + wrist[0], oy + wrist[1]
            ImageDraw.Draw(canvas).line([(wx, wy), (wx + 0.15 * W, H + 60)], fill=(104, 106, 112),
                                        width=int(0.16 * W))  # sleeve to the bottom edge
            canvas.paste(sprite, (ox, oy), sprite)
            tip_xy = [round(tx / W, 4), round(ty / H, 4)]
        if a.drift:
            # Slow camera drift: up to ±24 px and ±4 % zoom over the clip.
            t = n / max(1, len(timeline) - 1)
            dx, dy, z = 24 * np.sin(2 * np.pi * t), 16 * np.sin(4 * np.pi * t), 1 + 0.04 * np.sin(2 * np.pi * t)
            canvas = canvas.transform((W, H), Image.AFFINE,
                                      (1 / z, 0, (W / 2) * (1 - 1 / z) - dx, 0, 1 / z, (H / 2) * (1 - 1 / z) - dy),
                                      resample=Image.BILINEAR, fillcolor=(92, 74, 58))
            if tip_xy:
                tip_xy = [round(((tip_xy[0] * W - W / 2) * z + W / 2 + dx) / W, 4),
                          round(((tip_xy[1] * H - H / 2) * z + H / 2 + dy) / H, 4)]
        else:
            dx = dy = 0.0
            z = 1.0

        def to_frame(b, dx=dx, dy=dy, z=z):
            bx0, by0, bx1, by1 = (px + b[0], py + b[1], px + b[2], py + b[3])
            fx0, fy0 = (bx0 - W / 2) * z + W / 2 + dx, (by0 - H / 2) * z + H / 2 + dy
            fx1, fy1 = (bx1 - W / 2) * z + W / 2 + dx, (by1 - H / 2) * z + H / 2 + dy
            return [round(fx0 / W, 4), round(fy0 / H, 4), round((fx1 - fx0) / W, 4), round((fy1 - fy0) / H, 4)]

        done = [to_frame(lines[j]) for j in range(li + (1 if prog >= 1 else 0))]
        canvas.save(out / f"f{n:04d}.jpg", quality=80)
        truth.append({"frame": n, "phase": phase, "line": li, "tip": tip_xy, "lines_done": done})
    (out / "truth.json").write_text(json.dumps({"fps": fps, "size": [W, H], "frames": truth}, indent=0))
    print(f"{len(timeline)} frames, {len(lines)} lines -> {out}")


if __name__ == "__main__":
    main()
